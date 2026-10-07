
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import os
from core.data_paths import ONNX_DIR, TOURNAMENTS_DIR
import itertools
import signal
import psutil
import json
import numpy as np
import multiprocessing as mp
from whr import whole_history_rating

import chess_engine
from core.lib import (
    decode_move_index, move_to_san, get_model_hash, chose_move_idx, parse_uci_to_coords,
    coords_to_uci)
from evaluation.stockfish_player import StockfishPlayer

STOCKFISH_PATH = r"D:\logiciels\stockfish\stockfish.exe"

# ============================================================
#                     CONFIGURATION
# ============================================================
CHECKPOINT_DIR = str(ONNX_DIR)
SIMULATIONS_EVAL = 1000
GAMES_PER_PAIR = 16
MAX_WORKERS = 8  # Nombre de parties en parallèle

# Positions evaluees par inference, grace au virtual loss. Le GPU n'est pas
# active ici : huit processus se disputant la meme carte tombent sous le CPU.
MCTS_BATCH_SIZE = 8

# Virtual loss 2 : +9 a +20 % de debit mesures, qualite non-inferieure sur
# 2500 puzzles. Voir docs/etudes/2026-09-19-recherche-mcts-multicoeur/resultats/historique/2026-09-22-rebalayage-virtual-loss-resultats.md.
MCTS_VIRTUAL_LOSS = 2
WHR_STATE_FILE = str(TOURNAMENTS_DIR / "tournament_state.whr")
STATS_FILE = str(TOURNAMENTS_DIR / "tournament_stats.json")
MODE = "default"  # Options : "default", "all", "x-y", ou "endless"
STOCKFISH_ANCHOR_ELO = 2600


# ============================================================
#                     MOTEUR & MATCHS
# ============================================================
def play_game_between_two_bots(model_white, model_black, sims):
    board = chess_engine.Chessboard()
    board.set_startup_pieces()

    uci_moves = []
    san_moves = []

    while board.game_state == chess_engine.GameState.ONGOING:
        current_model = model_white if board.turn == chess_engine.Color.WHITE else model_black

        if isinstance(current_model, StockfishPlayer):
            move_uci = current_model.get_move(uci_moves)
            f_o, r_o, f_d, r_d, p = parse_uci_to_coords(move_uci)
        else:
            pi_raw = current_model.mcts_search(board, sims, 1.4, False, MCTS_BATCH_SIZE)
            pi = np.array(pi_raw, dtype=np.float32)
            move_count = len(san_moves)
            current_tau = 1.0 if move_count < 8 else 0.1

            best_idx = chose_move_idx(pi, current_tau)
            is_black = (board.turn == chess_engine.Color.BLACK)
            f_o, r_o, f_d, r_d, p = decode_move_index(board, best_idx, is_black)
            move_uci = coords_to_uci(f_o, r_o, f_d, r_d, p)

        san = move_to_san(board, f_o, r_o, f_d, r_d, p)
        board.move_piece(f_o, r_o, f_d, r_d, p)
        if board.game_state == chess_engine.GameState.CHECKMATE:
            san += "#"
        elif board.is_in_check():
            san += "+"

        san_moves.append(san)
        uci_moves.append(move_uci)

        if len(san_moves) > 300:
            break

    winner = "draw"
    if board.game_state == chess_engine.GameState.CHECKMATE:
        winner = "black" if board.turn == chess_engine.Color.WHITE else "white"
    return winner, san_moves


def format_pgn(white_name, black_name, winner, moves):
    result = "1/2-1/2"
    if winner == "white":
        result = "1-0"
    elif winner == "black":
        result = "0-1"
    pgn = f'[White "{white_name}"]\n[Black "{black_name}"]\n[Result "{result}"]\n\n'
    for i in range(0, len(moves), 2):
        move_num = (i // 2) + 1
        pgn += f"{move_num}. {moves[i]} "
        if i + 1 < len(moves): pgn += f"{moves[i + 1]} "
    pgn += f" {result}\n"
    return pgn


def play_game_worker(args):
    """
    Fonction exécutée par un sous-processus.
    Instancie les moteurs localement pour éviter les erreurs de Pickling.
    """
    white_h, black_h, white_p, black_p, sims, sf_anchor_elo = args
    sf_hash_str = f"STOCKFISH_FIXED_{sf_anchor_elo}"

    # 1. Instanciation locale du modèle Blanc
    if white_h == sf_hash_str:
        model_white = StockfishPlayer(STOCKFISH_PATH, elo=sf_anchor_elo)
    else:
        evaluator_w = chess_engine.ONNXEvaluator(os.path.join(CHECKPOINT_DIR, white_p))
        model_white = chess_engine.MCTS(evaluator_w, tt_size=131071)
        model_white.set_tuning(MCTS_VIRTUAL_LOSS)

    # 2. Instanciation locale du modèle Noir
    if black_h == sf_hash_str:
        model_black = StockfishPlayer(STOCKFISH_PATH, elo=sf_anchor_elo)
    else:
        evaluator = chess_engine.ONNXEvaluator(os.path.join(CHECKPOINT_DIR, black_p))
        model_black = chess_engine.MCTS(evaluator, tt_size=131071)
        model_black.set_tuning(MCTS_VIRTUAL_LOSS)

    # 3. Exécution de la partie
    winner, moves = play_game_between_two_bots(model_white, model_black, sims)

    # 4. Retour des données sérialisables
    return white_h, black_h, white_p, black_p, winner, moves


def generate_game_tasks(pairs_to_play, hash_to_filename):
    """Générateur qui produit les arguments de chaque partie individuelle à jouer."""
    for h1, h2 in pairs_to_play:
        p1 = hash_to_filename[h1]
        p2 = hash_to_filename[h2]
        for g in range(GAMES_PER_PAIR):
            if g % 2 == 0:
                yield h1, h2, p1, p2, SIMULATIONS_EVAL, STOCKFISH_ANCHOR_ELO
            else:
                yield h2, h1, p2, p1, SIMULATIONS_EVAL, STOCKFISH_ANCHOR_ELO


# ============================================================
#                     LOGIQUE DU TOURNOI (REFACTORISÉE)
# ============================================================
def load_tournament_state():
    TOURNAMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if os.path.exists(WHR_STATE_FILE):
        whr = whole_history_rating.Base.load_base(WHR_STATE_FILE)
    else:
        whr = whole_history_rating.Base({"w2": 14})

    stats = {}
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)

    return whr, stats


def scan_available_models():
    onnx_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".onnx")]

    hash_to_filename = {}
    for f in onnx_files:
        h = get_model_hash(os.path.join(CHECKPOINT_DIR, f))
        hash_to_filename[h] = f

    SF_HASH = f"STOCKFISH_FIXED_{STOCKFISH_ANCHOR_ELO}"
    hash_to_filename[SF_HASH] = f"STOCKFISH_{STOCKFISH_ANCHOR_ELO}_ANCHOR"

    all_hashes = list(hash_to_filename.keys())
    return hash_to_filename, all_hashes


def get_ranked_players(whr, files, stats):
    players = []
    for f in files:
        rating = whr.ratings_for_player(f)
        if rating:
            played = stats.get(f, 0)
            players.append((f, rating[-1][1], played))
    players.sort(key=lambda x: x[1], reverse=True)
    return players


def display_rankings(whr, hash_to_filename, stats, title="CLASSEMENT"):
    all_hashes = list(hash_to_filename.keys())
    ranked = get_ranked_players(whr, all_hashes, stats)

    sf_raw_elo = None
    sf_hash_str = f"STOCKFISH_FIXED_{STOCKFISH_ANCHOR_ELO}"

    for h, elo, games in ranked:
        if h == sf_hash_str:
            sf_raw_elo = elo
            break

    anchor_offset = STOCKFISH_ANCHOR_ELO - sf_raw_elo if sf_raw_elo is not None else 0

    print("\n" + "=" * 40 + f"\n {title}\n" + "=" * 40)
    for h, elo, games in ranked:
        name = hash_to_filename.get(h, "Unknown")
        print(f"{name:35} : {elo + anchor_offset:>6.1f} Elo | {games:>3} parties ({h})")

    return ranked


def build_match_schedule(whr, all_hashes, ranked_existing_hashes, hash_to_filename):
    pairs_to_play = []
    is_endless = False

    SF_HASH = f"STOCKFISH_FIXED_{STOCKFISH_ANCHOR_ELO}"
    known_hashes = [p.name for p in whr.players.values() if len(p.days) > 0]
    sf_player = whr.players.get(SF_HASH)
    sf_needs_games = sf_player is None or len(sf_player.days) == 0
    new_hashes = [h for h in all_hashes if h not in known_hashes]

    if MODE == "endless":
        if len(all_hashes) < 2:
            print("\nErreur : Pas assez de bots pour lancer un tournoi endless.")
            return [], False
        print(f"\nMode 'endless' activé : Tournoi infini entre les {len(all_hashes)} bots.")
        pairs_to_play = itertools.cycle(itertools.combinations(all_hashes, 2))
        is_endless = True

    elif sf_needs_games and ranked_existing_hashes:
        print(f" Calibration requise : Match contre l'ancre {SF_HASH}")
        champion_hash = ranked_existing_hashes[0][0]
        pairs_to_play.append((SF_HASH, champion_hash))

    elif MODE == "all":
        print(f"\nMode 'all' activé : Tournoi complet entre les {len(all_hashes)} bots.")
        pairs_to_play = list(itertools.combinations(all_hashes, 2))

    elif "-" in str(MODE):
        try:
            rank1, rank2 = map(int, str(MODE).split('-'))
            idx1, idx2 = rank1 - 1, rank2 - 1

            if max(idx1, idx2) >= len(ranked_existing_hashes):
                print(
                    f"\nErreur : Le classement ne contient que {len(ranked_existing_hashes)} bots. "
                    f"Impossible de jouer le match {MODE}.")
            elif min(idx1, idx2) < 0 or idx1 == idx2:
                print(f"\nErreur : Les rangs doivent être valides et différents (ex: '1-3').")
            else:
                h1 = ranked_existing_hashes[idx1][0]
                h2 = ranked_existing_hashes[idx2][0]
                print(f"\nMode spécifique activé : Match entre le rang {rank1} et le rang {rank2}.")
                pairs_to_play.append((h1, h2))
        except ValueError:
            print(f"\nErreur : Format de MODE invalide '{MODE}'. Utilisez un format comme '1-3'.")

    else:
        if new_hashes:
            print(f"\nNouveau(x) modèle(s) détecté(s) : {len(new_hashes)}")
            new_filenames = [hash_to_filename[_hash] for _hash in new_hashes]
            print(f"Nouveau(x) modèle(s) : {new_filenames}")
            if ranked_existing_hashes:
                champion_hash = ranked_existing_hashes[0][0]
                for nh in new_hashes:
                    pairs_to_play.append((nh, champion_hash))
            if len(new_hashes) > 1:
                for pair in itertools.combinations(new_hashes, 2):
                    pairs_to_play.append(pair)

        elif len(ranked_existing_hashes) >= 2:
            p1_h, p2_h = ranked_existing_hashes[0][0], ranked_existing_hashes[1][0]
            print(
                f"\nAucun nouveau bot. Choc des titans : {hash_to_filename[p1_h]} vs {hash_to_filename[p2_h]}")
            pairs_to_play.append((p1_h, p2_h))

    return pairs_to_play, is_endless


# ============================================================
#                     ORCHESTRATEUR PRINCIPAL
# ============================================================
def run_tournament():
    whr, stats = load_tournament_state()
    hash_to_filename, all_hashes = scan_available_models()

    ranked_existing_hashes = display_rankings(
        whr, hash_to_filename, stats,
        title=f"CLASSEMENT ACTUEL ANCRÉ (Stockfish = {STOCKFISH_ANCHOR_ELO})"
    )

    pairs_to_play, is_endless = build_match_schedule(
        whr, all_hashes, ranked_existing_hashes, hash_to_filename
    )

    if not pairs_to_play:
        print("\nAucun match à jouer avec la configuration actuelle.")
        return

    tasks = generate_game_tasks(pairs_to_play, hash_to_filename)
    games_completed = 0
    match_scores = {}
    match_games_count = {}

    print(f"\nLancement de {MAX_WORKERS} processus workers en parallèle...")

    try:
        # Multiprocessing Pool
        with mp.Pool(processes=MAX_WORKERS) as pool:
            for white_h, black_h, white_p, black_p, winner, moves in pool.imap_unordered(
                    play_game_worker, tasks):

                # Formatage et affichage du PGN
                print(format_pgn(white_p, black_p, winner, moves))

                # On utilise un tuple trié pour avoir une clé unique peu importe qui a les Blancs
                pair_key = tuple(sorted([white_h, black_h]))
                if pair_key not in match_scores:
                    match_scores[pair_key] = {white_h: 0.0, black_h: 0.0}
                    match_games_count[pair_key] = 0

                # Mise à jour stricte des statistiques
                stats[white_h] = stats.get(white_h, 0) + 1
                stats[black_h] = stats.get(black_h, 0) + 1

                if winner == "draw":
                    match_scores[pair_key][white_h] += 0.5
                    match_scores[pair_key][black_h] += 0.5
                    whr.create_game(black_h, white_h, "B", 0, 0)
                    whr.create_game(black_h, white_h, "W", 0, 0)
                else:
                    winner_h = white_h if winner == "white" else black_h
                    match_scores[pair_key][winner_h] += 1.0
                    outcome = "W" if winner == "white" else "B"
                    whr.create_game(black_h, white_h, outcome, 0, 0)

                match_games_count[pair_key] += 1
                games_completed += 1

                # --- 3. AFFICHAGE SI LE MATCH EST TERMINÉ ---
                if match_games_count[pair_key] == GAMES_PER_PAIR:

                    if MODE not in ["endless", "all"]:
                        score_w = match_scores[pair_key][white_h]
                        score_b = match_scores[pair_key][black_h]
                        print(
                            f"\n>>>> FIN DU MATCH : {white_p} ({score_w}) - {black_p} ({score_b})\n")

                    # Nettoyage de la RAM (indispensable pour le mode endless)
                    del match_scores[pair_key]
                    del match_games_count[pair_key]

                # Sauvegarde périodique pour ne pas perdre les données en cas de crash
                if games_completed % GAMES_PER_PAIR == 0:
                    whr.iterate(10)
                    whr.save_base(WHR_STATE_FILE)
                    with open(STATS_FILE, "w") as f:
                        json.dump(stats, f)
                    print("--- [SAUVEGARDE AUTOMATIQUE EFFECTUÉE] ---")

    except KeyboardInterrupt:
        print(
            "\n[Tournoi] Arrêt demandé par l'utilisateur (Ctrl+C). "
            "Terminaison forcée des processus...")

        # 1. On récupère le PID du processus principal
        current_system_pid = os.getpid()
        try:
            parent = psutil.Process(current_system_pid)
            children = parent.children(recursive=True)

            # 2. On envoie un signal d'arrêt forcé (SIGTERM/SIGKILL) à chaque enfant
            for child in children:
                child.send_signal(signal.SIGTERM)  # Ou signal.SIGKILL sous Linux

            # 3. On attend un court instant que le système nettoie les processus
            psutil.wait_procs(children, timeout=3)

        except psutil.NoSuchProcess:
            pass

        print("Processus nettoyés. Sauvegarde en cours...")

    # Affichage et sauvegarde finaux
    print("\nCalcul des Elo finaux...")
    whr.auto_iterate()
    whr.save_base(WHR_STATE_FILE)

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

    display_rankings(
        whr, hash_to_filename, stats,
        title=f"CLASSEMENT FINAL ANCRÉ (Stockfish = {STOCKFISH_ANCHOR_ELO})"
    )


if __name__ == "__main__":
    # Indispensable sous Windows pour éviter les conflits CUDA / Multiprocessing
    mp.set_start_method('spawn', force=True)
    run_tournament()
