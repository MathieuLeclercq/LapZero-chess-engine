
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import os
import math
import torch
import warnings
import hashlib
import threading
import glob
import onnx
import numpy as np

from datetime import datetime
from typing import Any
import chess_engine
from core.model import ChessNet

# Reexport : ces fonctions vivent desormais dans move_coding, importable
# sans torch. Les appelants historiques font `from core.lib import encode_move`.
from core.move_coding import (  # noqa: F401
    coords_to_uci,
    decode_move_index,
    encode_move,
    gestion_promo_dame,
    parse_uci_to_coords,
)


def move_to_san(board, orig_f, orig_r, dest_f, dest_r, promo):
    """Génère la notation SAN d'un coup AVANT qu'il soit joué sur le board."""
    piece = board.get_square(orig_f, orig_r).get_piece()
    p_type = piece.get_type()

    # Roque
    if p_type == chess_engine.PieceType.KING and abs(orig_f - dest_f) == 2:
        san = "O-O" if dest_f == 6 else "O-O-O"
    else:
        san = ""
        piece_letters = {
            chess_engine.PieceType.KNIGHT: "N",
            chess_engine.PieceType.BISHOP: "B",
            chess_engine.PieceType.ROOK: "R",
            chess_engine.PieceType.QUEEN: "Q",
            chess_engine.PieceType.KING: "K",
        }

        is_capture = board.get_square(dest_f, dest_r).is_occupied()
        # En passant
        if p_type == chess_engine.PieceType.PAWN and abs(orig_f - dest_f) == 1 and not is_capture:
            is_capture = True

        if p_type != chess_engine.PieceType.PAWN:
            san += piece_letters[p_type]

            # Désambiguïsation
            need_file, need_rank = False, False
            for f in range(8):
                for r in range(8):
                    if f == orig_f and r == orig_r:
                        continue
                    sq = board.get_square(f, r)
                    if not sq.is_occupied():
                        continue
                    if (sq.get_piece().get_type() != p_type or
                            sq.get_piece().get_color() != piece.get_color()):
                        continue
                    for m in board.get_legal_moves(f, r):
                        if (m.get_dest_square().get_file() == dest_f and
                                m.get_dest_square().get_rank() == dest_r):
                            if f == orig_f:
                                need_rank = True
                            else:
                                need_file = True

            if need_file:
                san += chr(ord('a') + orig_f)
            if need_rank:
                san += chr(ord('1') + orig_r)
        else:
            if is_capture:
                san += chr(ord('a') + orig_f)

        if is_capture:
            san += "x"

        san += chr(ord('a') + dest_f) + chr(ord('1') + dest_r)

        # Promotion
        promo_letters = {
            chess_engine.PieceType.QUEEN: "Q",
            chess_engine.PieceType.ROOK: "R",
            chess_engine.PieceType.BISHOP: "B",
            chess_engine.PieceType.KNIGHT: "N",
        }
        if promo in promo_letters:
            san += "=" + promo_letters[promo]

    return san


def print_pgn(board, san_move_list):
    # Après la boucle while, avant pygame.quit()
    pgn = ""
    for i, san in enumerate(san_move_list):
        if i % 2 == 0:
            pgn += f"{i // 2 + 1}. "
        pgn += san + " "

    results = {
        chess_engine.GameState.CHECKMATE: ("1-0" if board.turn == chess_engine.Color.BLACK
                                           else "0-1"),
        chess_engine.GameState.STALEMATE: "1/2-1/2",
        chess_engine.GameState.DRAW_REPETITION: "1/2-1/2",
        chess_engine.GameState.DRAW_50_MOVES: "1/2-1/2",
        chess_engine.GameState.DRAW_INSUFF_MATERIAL: "1/2-1/2"
    }
    pgn += results.get(board.game_state, "*")

    print("\n===== PGN =====")
    print(pgn)
    print("===============\n")


def load_supervised_model(checkpoint_path, num_res_blocks, num_filters, device):
    from training.train_supervised import AlphaZeroLightning
    """Charge le modèle depuis un checkpoint Lightning."""
    os.environ["TORCH_SKIP_WEIGHTS_ONLY_WARNING"] = "1"
    device = torch.device(device)

    lit_model = AlphaZeroLightning.load_from_checkpoint(
        checkpoint_path,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
    )
    model = lit_model.model
    model.to(device)
    model.eval()
    print(f"Modèle chargé depuis {checkpoint_path} (device: {device})")
    return model


def load_unsupervised_model(checkpoint_path, num_res_blocks, num_filters, device):
    """Charge le modèle depuis un checkpoint standard PyTorch."""
    os.environ["TORCH_SKIP_WEIGHTS_ONLY_WARNING"] = "1"

    # 1. Instanciation de l'architecture vide
    model = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)

    # 2. Chargement du fichier
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 3. Injection des poids
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"Modèle chargé depuis {checkpoint_path} (device: {device})")
    return model


def load_model(checkpoint_path, num_res_blocks, num_filters, device):
    if checkpoint_path.endswith('.ckpt'):
        # il faut load le modèle supervisé
        model = load_supervised_model(checkpoint_path, num_res_blocks, num_filters, device)
    elif checkpoint_path.endswith('.pt'):
        # il faut load le modèle "alphazero"
        model = load_unsupervised_model(checkpoint_path, num_res_blocks, num_filters, device)
    else:
        raise Exception('Extension inconnue pour le fichier checkpoint '
                        '(doit être ".pt" ou ".ckpt"')
    return model


def ai_pick_move_instant(board, model, device, temperature=0.1):
    """
    Utilise le réseau pour choisir un coup.
    1. Encode la position en tensor 119×8×8
    2. Forward pass → policy logits + value
    3. Masque les coups illégaux
    4. Sélectionne le meilleur coup (avec température optionnelle)
    5. Décode l'index en coordonnées
    """
    # Tensor d'entrée
    tensor_np = board.get_alphazero_tensor()
    x = torch.from_numpy(tensor_np).float().unsqueeze(0).to(device)  # [1, 119, 8, 8]

    # Indices des coups légaux (déjà encodés par le moteur C++)
    legal_indices = board.get_legal_move_indices()
    if not legal_indices:
        return None

    # Inférence
    with torch.no_grad():
        p_logits, v_pred = model(x)

    p_logits = p_logits.squeeze(0).cpu().numpy()  # [4672]
    value = v_pred.item()

    # Masquage : on met -inf partout sauf les coups légaux
    mask = np.full(4672, -np.inf)
    for idx in legal_indices:
        mask[idx] = p_logits[idx]

    # Sélection du coup
    if temperature <= 0:
        best_idx = legal_indices[np.argmax(mask[legal_indices])]
    else:
        # Softmax avec température sur les coups légaux uniquement
        legal_logits = np.array([p_logits[i] for i in legal_indices])
        legal_logits = legal_logits / temperature
        legal_logits -= legal_logits.max()  # stabilité numérique
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        chosen = np.random.choice(len(legal_indices), p=probs)
        best_idx = legal_indices[chosen]

    # Décodage
    is_black = (board.turn == chess_engine.Color.BLACK)
    orig_f, orig_r, dest_f, dest_r, promo = decode_move_index(board, best_idx, is_black)

    print(f"IA joue: ({orig_f},{orig_r}) -> ({dest_f},{dest_r}), promo={promo}, value={value:.3f}")

    return orig_f, orig_r, dest_f, dest_r, promo


def append_to_disk_buffer(new_data, folder_path, max_buffer_size):
    """Sauvegarde les nouvelles positions sur disque et supprime les shards les plus anciens."""
    if not new_data:
        remaining = 0
        if os.path.exists(folder_path):
            for s in glob.glob(os.path.join(folder_path, "shard_*.npz")):
                try:
                    remaining += int(os.path.splitext(s)[0].split("_")[-1])
                except (ValueError, IndexError):
                    continue
        print(f"  [Disque] +0 positions. Buffer total : ~{remaining} / {max_buffer_size}")
        return remaining

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    num_new = len(new_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shard_{timestamp}_{num_new}.npz"
    filepath = os.path.join(folder_path, filename)

    states = np.array([item[0] for item in new_data], dtype=np.float16)
    policies = np.array([item[1] for item in new_data], dtype=np.float16)
    values = np.array([item[2] for item in new_data], dtype=np.float32)

    np.savez_compressed(filepath, states=states, policies=policies, values=values)

    shards = sorted(glob.glob(os.path.join(folder_path, "shard_*.npz")))
    total = 0
    keep_from = 0
    for i in range(len(shards) - 1, -1, -1):
        try:
            size = int(os.path.splitext(shards[i])[0].split("_")[-1])
        except (ValueError, IndexError):
            continue
        total += size
        if total > max_buffer_size:
            keep_from = i + 1
            break

    for shard in shards[:keep_from]:
        try:
            os.remove(shard)
        except OSError:
            pass

    remaining = sum(
        int(os.path.splitext(s)[0].split("_")[-1])
        for s in shards[keep_from:]
        if os.path.exists(s)
    )
    print(f"  [Disque] +{num_new} positions. Buffer total : ~{remaining} / {max_buffer_size}")
    return remaining


# ============================================================
#                     EXPORT ONNX & QUANTIFICATION
# ============================================================

def export_model_to_onnx(model, onnx_path, device):
    """
    Export ONNX
    """
    model.eval()
    dummy_input = (torch.randn(1, 119, 8, 8, device=device),)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            external_data=False,
            dynamic_axes={
                'input': {0: 'batch'},
                'policy': {0: 'batch'},
                'value': {0: 'batch'}
            },
            verbose=False
        )


def get_model_hash(filepath):
    """Génère un hash unique basé sur le contenu du fichier."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def chose_move_idx(pi, tau):
    mask = pi > 0
    logits = np.full_like(pi, -1e20, dtype=np.float64)
    logits[mask] = np.log(pi[mask].astype(np.float64)) / tau

    logits -= np.max(logits)
    probs = np.exp(logits)
    probs[~mask] = 0
    probs /= np.sum(probs)

    chosen_idx = np.random.choice(4672, p=probs)
    return chosen_idx


def calculate_performance_rating(sf_elo, wins, draws, losses):
    total_games = wins + draws + losses
    if total_games == 0:
        return sf_elo

    # Calcul du score réel (1 pour win, 0.5 pour draw)
    score = wins + 0.5 * draws
    percentage = score / total_games

    # Correction pour éviter log(0) ou division par zéro sur de petits échantillons
    # On cape le pourcentage entre un demi-point de défaite et un demi-point de victoire
    percentage = max(0.5 / total_games, min((total_games - 0.5) / total_games, percentage))

    # Formule Elo inverse
    elo_diff = 400 * math.log10(percentage / (1 - percentage))

    return int(sf_elo + elo_diff)


def run_with_interrupt(fn, *args) -> Any:
    """Lance fn(*args) dans un thread pour que Ctrl+C reste réactif."""
    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = fn(*args)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target)
    t.start()

    try:
        while t.is_alive():
            t.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n[Interruption détectée] En attente de la fin du self-play C++...")
        raise

    if exc[0]:
        raise exc[0]
    return result[0]


def convert_game_results(games):
    data = []
    stats = {
        "checkmates": 0, "stalemates": 0, "repetition": 0,
        "50_moves": 0, "insuff_mat": 0, "max_moves": 0,
        "total_saved_moves": 0
    }

    # Mapping entre l'entier C++ et la clé du dictionnaire
    reason_keys = ["checkmates", "stalemates", "repetition", "50_moves", "insuff_mat", "max_moves"]

    for game in games:
        states = game.state_tensors
        policies = game.policies
        outcome = game.final_outcome
        reason_idx = game.end_reason
        n = states.shape[0]

        stats["total_saved_moves"] += n

        # Comptage détaillé
        if 0 <= reason_idx <= 5:
            key = reason_keys[reason_idx]
            stats[key] += 1

        # Conversion par partie : un passage FP16 par tenseur au lieu d'un
        # passage par position. La precision de chaque exemple est inchangee.
        states_fp16 = states.astype(np.float16)
        policies_fp16 = policies.astype(np.float16)
        white_turns = states[:, 112, 0, 0] > 0.5
        for i in range(n):
            value = outcome if white_turns[i] else -outcome
            data.append((states_fp16[i], policies_fp16[i], value))

    return data, stats
