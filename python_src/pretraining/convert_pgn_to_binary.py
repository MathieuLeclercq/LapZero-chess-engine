
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import numpy as np
import chess.pgn
import chess_engine
from core.lib import encode_move, gestion_promo_dame
import os
import argparse
from pathlib import Path
import io
from concurrent.futures import ProcessPoolExecutor

# ============================================================
#                     CONFIGURATION
# ============================================================
INPUT_PGN = _runtime.REPO_ROOT / "training_data" / "top_games_quality.pgn"
OUTPUT_DIR = _runtime.REPO_ROOT / "training_data" / "dataset_shards"
POSITIONS_PER_SHARD = 200_000  # Chaque fichier fera environ 1.5 Go
MAX_WORKERS = os.cpu_count()  # Utilise tous les cœurs (ex: 8 ou 16)
BATCH_SIZE_GAMES = 500  # Nombre de parties envoyées à chaque worker

PROMO_MAP = {
    chess.KNIGHT: chess_engine.PieceType.KNIGHT,
    chess.BISHOP: chess_engine.PieceType.BISHOP,
    chess.ROOK: chess_engine.PieceType.ROOK,
    chess.QUEEN: chess_engine.PieceType.QUEEN,
    None: chess_engine.PieceType.NONE
}


# ============================================================
#                     FONCTION WORKER
# ============================================================
def worker_task(game_texts, worker_id, shard_id, output_dir=OUTPUT_DIR):
    """Traitement lourd effectué par un cœur du processeur."""
    X, Y_p, Y_v = [], [], []

    for text in game_texts:
        game = chess.pgn.read_game(io.StringIO(text))
        if not game:
            continue

        res = game.headers.get("Result")
        val_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
        if res not in val_map: continue
        game_result = val_map[res]

        board = chess_engine.Chessboard()
        board.set_startup_pieces()

        for move in game.mainline_moves():
            # Input
            X.append(board.get_alphazero_tensor().astype(np.uint8))

            # Move info
            is_black = (board.turn == chess_engine.Color.BLACK)
            o_f, o_r = move.from_square % 8, move.from_square // 8
            d_f, d_r = move.to_square % 8, move.to_square // 8

            promo = PROMO_MAP.get(move.promotion, chess_engine.PieceType.NONE)
            promo = gestion_promo_dame(board, o_f, o_r, d_r, promo)

            # Target Policy
            Y_p.append(encode_move(o_f, o_r, d_f, d_r, promo, is_black))

            # Target Value
            Y_v.append(game_result if not is_black else -game_result)

            board.move_piece(o_f, o_r, d_f, d_r, promo)

    # Sauvegarde du shard
    shard_name = f"shard_{worker_id}_{shard_id}.npz"
    path = os.path.join(output_dir, shard_name)
    np.savez(path, x=np.array(X, dtype=np.uint8),
             y_p=np.array(Y_p, dtype=np.int64),
             y_v=np.array(Y_v, dtype=np.float32))
    return len(X)


# ============================================================
#                     BOUCLE PRINCIPALE
# ============================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Conversion PGN vers shards NumPy")
    parser.add_argument("--input-pgn", type=Path, default=INPUT_PGN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS or 1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers doit etre strictement positif")
    return args


def main(argv=None):
    args = parse_args(argv)
    file_size = args.input_pgn.stat().st_size
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lancement de la conversion sur {args.workers} cœurs...")
    print(f"Taille du fichier PGN : {file_size / (1024 ** 3):.2f} Go")

    with open(args.input_pgn, 'r', encoding='utf-8') as f:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            current_batch = []
            shard_id = 0
            total_positions = 0
            games_read = 0

            while True:
                if games_read >= 600_000:
                    break
                line = f.readline()
                if not line:
                    break

                if line.startswith('[Event '):
                    games_read += 1
                    if current_batch:
                        if len(current_batch) >= BATCH_SIZE_GAMES:
                            futures.append(
                                executor.submit(worker_task, current_batch, len(futures), shard_id, args.output_dir))
                            current_batch = []
                            shard_id += 1

                            # Affichage de la progression
                            current_pos = f.tell()
                            percent = (current_pos / file_size) * 100
                            print(
                                f"[{percent:5.1f}%] Parties lues : {games_read} | Shards envoyés : {shard_id}",
                                end='\r')

                current_batch.append(line)
                while True:
                    l = f.readline()
                    if not l: break
                    current_batch[-1] += l
                    if l.strip() == "": break

            # Envoyer le reliquat
            if current_batch:
                futures.append(executor.submit(worker_task, current_batch, len(futures), shard_id, args.output_dir))

            print(f"\nLecture terminée. Attente de la finalisation des {len(futures)} workers...")

            # Récupération des résultats des workers
            for i, future in enumerate(futures):
                try:
                    res = future.result()
                    total_positions += res
                    print(
                        f"Worker {i}/{len(futures)} terminé | Positions cumulées : {total_positions}",
                        end='\r')
                except Exception as e:
                    print(f"\nErreur sur le worker {i} : {e}")

    print(f"\n\nTerminé !")
    print(f"Total positions générées : {total_positions}")
    print(f"Total parties traitées : {games_read}")
    print(f"Shards sauvegardés dans : {args.output_dir}")


if __name__ == "__main__":
    main()
