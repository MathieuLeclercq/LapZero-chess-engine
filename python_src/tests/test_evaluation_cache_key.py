"""Clé du cache réseau, alignée sur les informations du tensor AlphaZero."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest


RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine


def board_from_moves(moves):
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    for move in moves:
        assert board.move_piece_uci(move)
    return board


ORDRE_A = ("g1f3", "g8f6", "b1c3", "b8c6")
ORDRE_B = ("b1c3", "b8c6", "g1f3", "g8f6")


def test_compteur_exact_meme_si_le_plateau_est_identique():
    a = chess_engine.Chessboard()
    b = chess_engine.Chessboard()
    a.load_fen("8/8/8/8/8/2k5/8/R3K3 w - - 2 1")
    b.load_fen("8/8/8/8/8/2k5/8/R3K3 w - - 3 1")

    assert (a.get_evaluation_cache_key(0).position_hash
            == b.get_evaluation_cache_key(0).position_hash)
    assert (a.get_evaluation_cache_key(0).combined_hash
            != b.get_evaluation_cache_key(0).combined_hash)


def test_h0_ignore_l_historique_mais_h1_le_distingue():
    a = board_from_moves(ORDRE_A)
    b = board_from_moves(ORDRE_B)

    assert a.to_fen() == b.to_fen()
    assert (a.get_evaluation_cache_key(0).combined_hash
            == b.get_evaluation_cache_key(0).combined_hash)
    assert (a.get_evaluation_cache_key(1).combined_hash
            != b.get_evaluation_cache_key(1).combined_hash)


def test_h1_ignore_ce_qui_precede_la_position_precedente():
    a = board_from_moves(ORDRE_A + ("e2e3",))
    b = board_from_moves(ORDRE_B + ("e2e3",))

    assert a.to_fen() == b.to_fen()
    assert (a.get_evaluation_cache_key(1).combined_hash
            == b.get_evaluation_cache_key(1).combined_hash)
    assert (a.get_evaluation_cache_key(3).combined_hash
            != b.get_evaluation_cache_key(3).combined_hash)


def test_historique_ne_hash_pas_les_anciens_droits_de_roque():
    avec = chess_engine.Chessboard()
    sans = chess_engine.Chessboard()
    avec.load_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    sans.load_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    assert avec.move_piece_uci("e1e2")
    assert sans.move_piece_uci("e1e2")

    np.testing.assert_array_equal(
        avec.get_alphazero_tensor(), sans.get_alphazero_tensor())
    assert (avec.get_evaluation_cache_key(1).combined_hash
            == sans.get_evaluation_cache_key(1).combined_hash)


def test_amnesie_est_ignoree_en_h0_et_visible_en_h1():
    normal = board_from_moves(ORDRE_A)
    amnesique = board_from_moves(ORDRE_A)
    amnesique.set_amnesia_mode(True)

    assert (normal.get_evaluation_cache_key(0).combined_hash
            == amnesique.get_evaluation_cache_key(0).combined_hash)
    assert (normal.get_evaluation_cache_key(1).combined_hash
            != amnesique.get_evaluation_cache_key(1).combined_hash)


def test_repetition_et_numero_de_coup_suivent_exactement_le_tensor():
    board = board_from_moves(("g1f3", "g8f6", "f3g1", "f6g8"))
    key = board.get_evaluation_cache_key(0)
    tensor = board.get_alphazero_tensor()

    assert key.repetition_category == 1
    assert key.total_moves_bucket == 2
    np.testing.assert_array_equal(tensor[12], np.ones((8, 8)))
    np.testing.assert_allclose(tensor[113], 0.02)


@pytest.mark.parametrize("depth", [-1, 8])
def test_la_cle_refuse_une_profondeur_hors_bornes(depth):
    with pytest.raises(ValueError):
        board_from_moves(()).get_evaluation_cache_key(depth)
