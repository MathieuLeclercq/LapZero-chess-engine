"""Contrat de conversion des parties self-play en exemples d'entrainement.

La conversion reste par partie pour les tenseurs, mais chaque exemple doit
garder exactement la meme precision et le meme signe de value selon le trait.
Les parties sont factices : aucun modele, aucune generation.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

from core.lib import convert_game_results


def partie_factice(n, outcome, reason=0):
    states = np.zeros((n, 119, 8, 8), dtype=np.float32)
    policies = np.zeros((n, 4672), dtype=np.float32)
    states[:, 112, 0, 0] = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    return SimpleNamespace(
        state_tensors=states,
        policies=policies,
        final_outcome=outcome,
        end_reason=reason)


def test_conversion_garde_precision_et_signe_de_value():
    data, stats = convert_game_results([partie_factice(3, 1.0)])

    assert len(data) == 3
    assert stats["checkmates"] == 1
    assert stats["total_saved_moves"] == 3

    for index, (tensor, policy, value) in enumerate(data):
        assert tensor.dtype == np.float16
        assert policy.dtype == np.float16
        assert tensor.shape == (119, 8, 8)
        assert policy.shape == (4672,)
        attendu = 1.0 if index % 2 == 0 else -1.0
        assert value == attendu


def test_conversion_signale_les_raisons_de_fin():
    games = [partie_factice(1, 0.0, reason) for reason in range(6)]
    data, stats = convert_game_results(games)

    assert len(data) == 6
    assert stats["checkmates"] == 1
    assert stats["stalemates"] == 1
    assert stats["repetition"] == 1
    assert stats["50_moves"] == 1
    assert stats["insuff_mat"] == 1
    assert stats["max_moves"] == 1
