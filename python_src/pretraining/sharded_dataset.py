
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import random

"""
Dataset pour entraînement supervisé
sur des parties de Lichess
"""


class ShardedChessDataset(IterableDataset):
    def __init__(self, shard_dir, shuffle=True):
        super().__init__()
        self.shard_dir = shard_dir
        self.shuffle = shuffle
        self.shard_files = [
            os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith('.npz')
        ]

        if not self.shard_files:
            raise ValueError(f"Aucun fichier .npz trouvé dans {shard_dir}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.shard_files.copy()

        # 1. Mélange de l'ordre des fichiers
        if self.shuffle:
            random.shuffle(files)

        # 2. Distribution des fichiers entre les workers
        if worker_info is not None:
            per_worker = int(np.ceil(len(files) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(files))
            files = files[start:end]

        # 3. Lecture et distribution des données
        for filepath in files:
            try:
                # On charge le shard entier en RAM (très rapide car < 500 Mo)
                data = np.load(filepath)
                X = data['x']
                Y_p = data['y_p']
                Y_v = data['y_v']
            except Exception as e:
                print(f"Erreur de lecture du shard {filepath}: {e}")
                continue

            num_samples = len(X)
            indices = np.arange(num_samples)

            # 4. Mélange interne au shard (crucial pour casser l'ordre chronologique de la partie)
            if self.shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                # Conversion uint8 -> float32 pour l'entrée du réseau
                x_tensor = torch.from_numpy(X[idx]).float()
                y_policy = torch.tensor(Y_p[idx], dtype=torch.long)
                y_value = torch.tensor([Y_v[idx]], dtype=torch.float32)

                yield x_tensor, y_policy, y_value
