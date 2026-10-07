
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import os
import re
import torch
from torch.utils.data import IterableDataset
import chess_engine

from core.lib import encode_move


def extract_result_from_pgn(filepath):
    """Extrait le résultat brut de la partie depuis l'en-tête PGN."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('[Result'):
                if '"1-0"' in line:
                    return 1.0
                elif '"0-1"' in line:
                    return -1.0
                elif '"1/2-1/2"' in line:
                    return 0.0
    return None


def extract_sans_from_pgn(filepath):
    """Extrait la liste des coups SAN d'un fichier PGN."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # DOTALL permet au point '.' d'inclure les sauts de ligne pour les commentaires multi-lignes
    content = re.sub(r'\[.*?\]', '', content, flags=re.DOTALL)
    content = re.sub(r'\{.*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\d+\.+', '', content)
    content = re.sub(r'1-0|0-1|1/2-1/2|\*', '', content)

    return content.split()


class ChessSupervisedDataset(IterableDataset):
    def __init__(self, pgn_dir):
        super().__init__()
        self.pgn_dir = pgn_dir
        self.pgn_files = [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if
                          f.endswith('.pgn')]

    def __iter__(self):
        # 1. Gestion du multi-processing PyTorch
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files_to_process = self.pgn_files  # num_workers=0 : on prend tout
        else:
            # Multi-processus : on découpe la liste des fichiers
            per_worker = len(self.pgn_files) // worker_info.num_workers
            start = worker_info.id * per_worker
            # Le dernier worker prend l'éventuel reste de la division
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(
                self.pgn_files)
            files_to_process = self.pgn_files[start:end]

        # 2. Boucle sur la portion de fichiers assignée à ce processus
        for pgn_file in files_to_process:
            game_result = extract_result_from_pgn(pgn_file)
            if game_result is None:
                continue

            sans = extract_sans_from_pgn(pgn_file)
            if not sans:
                continue

            board = chess_engine.Chessboard()
            board.set_startup_pieces()

            for san in sans:
                tensor_numpy = board.get_alphazero_tensor()
                x = torch.from_numpy(tensor_numpy).float()

                is_black = (board.turn == chess_engine.Color.BLACK)

                success = board.move_piece_san(san)
                if not success:
                    break

                orig_f, orig_r, dest_f, dest_r, promo = board.get_last_move_data()

                target_move_index = encode_move(orig_f, orig_r, dest_f, dest_r, promo, is_black)
                if target_move_index == -1:
                    break

                y_policy = torch.tensor(target_move_index, dtype=torch.long)

                current_player_result = game_result if not is_black else -game_result
                y_value = torch.tensor([current_player_result], dtype=torch.float32)

                yield x, y_policy, y_value
