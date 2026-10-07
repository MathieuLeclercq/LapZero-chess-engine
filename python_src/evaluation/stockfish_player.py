
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import chess
import os
import chess.engine
import numpy as np
import multiprocessing as mp
from datetime import datetime

import chess_engine
from core.lib import chose_move_idx, decode_move_index, parse_uci_to_coords, coords_to_uci, move_to_san

# Virtual loss 2 : +9 a +20 % de debit mesures, qualite non-inferieure sur
# 2500 puzzles. Voir docs/etudes/2026-09-19-recherche-mcts-multicoeur/resultats/historique/2026-09-22-rebalayage-virtual-loss-resultats.md.
MCTS_VIRTUAL_LOSS = 2


class StockfishPlayer:
    def __init__(self, path="stockfish.exe", elo=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stockfish introuvable à l'adresse : {os.path.abspath(path)}")

        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def get_move(self, board_moves_uci):
        internal_board = chess.Board()
        for move_uci in board_moves_uci:
            internal_board.push_uci(move_uci)

        result = self.engine.play(internal_board, chess.engine.Limit(nodes=325_000))
        return result.move.uci()

    def __del__(self):
        if hasattr(self, 'engine'):
            try:
                self.engine.quit()
            except chess.engine.EngineTerminatedError:
                pass


def eval_worker(args):
    """
    Worker isolé pour jouer une seule partie d'évaluation.
    Instancie les moteurs localement pour la sécurité mémoire.
    """
    onnx_path, stockfish_path, mcts_sims, sf_elo, sf_nodes, is_mcts_white = args

    sf = StockfishPlayer(stockfish_path, elo=sf_elo)

    evaluator = chess_engine.ONNXEvaluator(onnx_path, False)
    mcts = chess_engine.MCTS(evaluator, tt_size=131071)
    mcts.set_tuning(MCTS_VIRTUAL_LOSS)

    board = chess_engine.Chessboard()
    board.set_startup_pieces()

    uci_moves = []
    san_moves = []

    while board.game_state == chess_engine.GameState.ONGOING:
        is_mcts_turn = (board.turn == chess_engine.Color.WHITE and is_mcts_white) or \
                       (board.turn == chess_engine.Color.BLACK and not is_mcts_white)

        if is_mcts_turn:
            pi_raw = mcts.mcts_search(board, mcts_sims, 1.4, False)
            pi = np.array(pi_raw, dtype=np.float32)

            # Température 1.0 pour forcer des ouvertures différentes (8 demi-coups)
            if len(san_moves) < 8:
                best_idx = chose_move_idx(pi, 1.0)
            else:
                best_idx = np.argmax(pi)

            is_black = (board.turn == chess_engine.Color.BLACK)
            f_o, r_o, f_d, r_d, p = decode_move_index(board, best_idx, is_black)
            move_uci = coords_to_uci(f_o, r_o, f_d, r_d, p)
        else:
            internal_board = chess.Board()
            for u in uci_moves:
                internal_board.push_uci(u)
            result = sf.engine.play(internal_board, chess.engine.Limit(nodes=sf_nodes))
            move_uci = result.move.uci()
            f_o, r_o, f_d, r_d, p = parse_uci_to_coords(move_uci)

        san = move_to_san(board, f_o, r_o, f_d, r_d, p)
        success = board.move_piece(f_o, r_o, f_d, r_d, p)
        if not success:
            print(f"Erreur Fatale : L'IA a tenté un coup illégal ({san}) ! Abandon de la partie.")
            break
        san_moves.append(san)
        uci_moves.append(move_uci)

        if len(san_moves) > 300:
            break

    # Fermeture explicite et immédiate du moteur pour éviter les processus zombies
    sf.engine.quit()

    # Détermination du résultat
    if board.game_state == chess_engine.GameState.CHECKMATE:
        winner_is_white = (board.turn == chess_engine.Color.BLACK)
        if winner_is_white == is_mcts_white:
            result_str = "win"
            pgn_result = "1-0" if is_mcts_white else "0-1"
        else:
            result_str = "loss"
            pgn_result = "0-1" if is_mcts_white else "1-0"
    else:
        result_str = "draw"
        pgn_result = "1/2-1/2"

    # Construction du PGN
    white_name = "AlphaZero" if is_mcts_white else f"Stockfish {sf_elo}"
    black_name = f"Stockfish {sf_elo}" if is_mcts_white else "AlphaZero"

    pgn = f'[Event "Evaluation AlphaZero"]\n'
    pgn += f'[White "{white_name}"]\n'
    pgn += f'[Black "{black_name}"]\n'
    pgn += f'[Result "{pgn_result}"]\n\n'

    for i, san in enumerate(san_moves):
        if i % 2 == 0:
            pgn += f"{i // 2 + 1}. "
        pgn += f"{san} "

    pgn += f" {pgn_result}\n"

    return result_str, pgn


def evaluate_against_anchor(onnx_path, stockfish_path, num_games=16, mcts_sims=100, sf_elo=2500,
                            sf_nodes=200_000, num_workers=8):
    print(f"  Évaluation contre Stockfish {sf_elo} Elo ({num_games} parties, {mcts_sims} sims)...")
    wins, draws, losses = 0, 0, 0

    tasks = []
    for g in range(num_games):
        is_mcts_white = (g % 2 == 0)
        tasks.append((onnx_path, stockfish_path, mcts_sims, sf_elo, sf_nodes, is_mcts_white))

    # Création du dossier et du fichier PGN
    os.makedirs("eval_pgns", exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    pgn_filename = f"eval_pgns/eval_{timestamp}.pgn"

    with mp.Pool(processes=min(num_workers, num_games)) as pool:
        with open(pgn_filename, "w", encoding="utf-8") as f:
            for result_str, pgn in pool.imap_unordered(eval_worker, tasks):

                f.write(pgn + "\n\n")  # Écriture de la partie dans le fichier

                if result_str == "win":
                    wins += 1
                elif result_str == "loss":
                    losses += 1
                else:
                    draws += 1

    winrate = (wins + 0.5 * draws) / num_games
    print(f"  Résultat Éval : {wins} V | {draws} N | {losses} D (Score: {winrate * 100:.1f}%)")
    print(f"  Parties sauvegardées dans : {pgn_filename}")

    return winrate, wins, draws, losses
