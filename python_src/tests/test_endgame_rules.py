"""Regressions des regles de fin de partie utilisees par la recherche.

Ces tests restent independants du reseau. Ils permettent de distinguer une
mauvaise conversion de finale d'une erreur dans le compteur des 50 coups.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest


RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine


def plateau(fen: str):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


def jouer(board, uci: str):
    orig_file = ord(uci[0]) - ord("a")
    orig_rank = int(uci[1]) - 1
    dest_file = ord(uci[2]) - ord("a")
    dest_rank = int(uci[3]) - 1
    promotion = chess_engine.PieceType.NONE
    return board.move_piece(
        orig_file, orig_rank, dest_file, dest_rank, promotion, True)


@pytest.fixture(scope="module")
def evaluator(tmp_path_factory):
    checkpoint = (RACINE / "data" / "checkpoints"
                  / "2026_04_23_23h25_iter316_unsupervised.pt")
    if not checkpoint.exists():
        pytest.skip("checkpoint absent")

    import evaluation.puzzle_bench as puzzle_bench

    onnx, _ = puzzle_bench.resoudre_modele(
        checkpoint, tmp_path_factory.mktemp("endgame_onnx"))
    return chess_engine.ONNXEvaluator(str(onnx), False)


def test_un_coup_calme_declenche_la_regle_au_centiemes_demi_coup():
    board = plateau("7k/8/8/8/8/8/R7/K7 w - - 99 1")

    assert jouer(board, "a2a3")

    assert board.half_move_clock == 100
    assert board.game_state == chess_engine.GameState.DRAW_50_MOVES


def test_un_coup_de_pion_remet_le_compteur_a_zero():
    board = plateau("7k/8/8/8/8/8/P7/K7 w - - 99 1")

    assert jouer(board, "a2a3")

    assert board.half_move_clock == 0
    assert board.game_state == chess_engine.GameState.ONGOING


def test_une_capture_remet_le_compteur_a_zero():
    board = plateau("7k/8/8/8/8/8/Rb6/K7 w - - 99 1")

    assert jouer(board, "a2b2")

    assert board.half_move_clock == 0
    assert board.game_state == chess_engine.GameState.ONGOING


def test_undo_restaure_le_compteur_et_etat_avant_le_coup():
    fen = "7k/8/8/8/8/8/R7/K7 w - - 99 1"
    board = plateau(fen)

    assert jouer(board, "a2a3")
    assert board.game_state == chess_engine.GameState.DRAW_50_MOVES

    board.undo_move()

    assert board.to_fen() == fen
    assert board.half_move_clock == 99
    assert board.game_state == chess_engine.GameState.ONGOING


def test_un_mat_au_centiemes_demi_coup_reste_un_mat():
    board = plateau("7k/8/5KQ1/8/8/8/8/8 w - - 99 1")

    assert jouer(board, "g6g7")

    assert board.half_move_clock == 100
    assert board.game_state == chess_engine.GameState.CHECKMATE


def test_la_troisieme_occurrence_declenche_la_nulle_par_repetition():
    board = chess_engine.Chessboard()
    board.set_startup_pieces()

    for uci in (
            "g1f3", "g8f6", "f3g1", "f6g8",
            "g1f3", "g8f6", "f3g1", "f6g8"):
        assert jouer(board, uci)

    assert board.half_move_clock == 8
    assert board.game_state == chess_engine.GameState.DRAW_REPETITION


def test_le_compteur_des_50_coups_modifie_bien_le_tensor_du_reseau():
    board_0 = plateau("8/8/8/8/8/2k5/8/R3K3 w - - 0 1")
    board_99 = plateau("8/8/8/8/8/2k5/8/R3K3 w - - 99 1")

    tensor_0 = board_0.get_alphazero_tensor()
    tensor_99 = board_99.get_alphazero_tensor()
    differences = np.flatnonzero(tensor_0 != tensor_99)

    # Seul le dernier plan constant, no-progress, doit changer. Ses 64 cases
    # valent toutes 0.99 dans la seconde position.
    assert len(differences) == 64
    np.testing.assert_allclose(tensor_99[118], 0.99)


@pytest.mark.parametrize("batch_size", [0, 8])
def test_mcts_evalue_comme_nulle_une_unique_branche_atteignant_100(
        batch_size, evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192)
    board = plateau("8/8/8/8/8/k7/8/KR6 b - - 99 1")

    mcts.step_analysis(board, 8, 1.4, batch_size)
    results = mcts.get_analysis_results()

    assert len(results) == 1
    assert results[0].q_value == pytest.approx(0.0)
    assert results[0].visits == 8
    assert mcts.get_counters().terminal_hits == 8


def test_tt_ne_reutilise_pas_une_evaluation_avec_un_autre_compteur(evaluator):
    fen_0 = "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"
    fen_99 = "8/8/8/8/8/2k5/8/R3K3 w - - 99 1"

    warmed = chess_engine.MCTS(evaluator, 8192)
    warmed.mcts_search(plateau(fen_0), 64, 1.4, False, 8)
    policy_warmed = warmed.mcts_search(plateau(fen_99), 64, 1.4, False, 8)

    fresh = chess_engine.MCTS(evaluator, 8192)
    policy_fresh = fresh.mcts_search(plateau(fen_99), 64, 1.4, False, 8)

    assert list(policy_warmed) == list(policy_fresh)


def test_legacy_reproduit_le_hit_avec_un_autre_compteur(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, -1)
    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"), 0, 1.4, False, 8)
    mcts.reset_counters()

    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 99 1"), 0, 1.4, False, 8)
    c = mcts.get_counters()

    assert c.tt_hits == 1
    assert c.tt_rule50_rejects == 0


def test_compteur_different_est_un_rejet_mesure(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 0)
    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"), 0, 1.4, False, 8)
    mcts.reset_counters()

    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 99 1"), 0, 1.4, False, 8)
    c = mcts.get_counters()

    assert c.tt_hits == 0
    assert c.tt_misses == 1
    assert c.tt_position_matches == 1
    assert c.tt_rule50_rejects == 1


def test_contexte_courant_different_est_un_rejet_mesure(evaluator):
    sans_historique = plateau(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 3")
    avec_repetition = chess_engine.Chessboard()
    avec_repetition.set_startup_pieces()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
        assert avec_repetition.move_piece_uci(move)

    assert (sans_historique.get_evaluation_cache_key(0).position_hash
            == avec_repetition.get_evaluation_cache_key(0).position_hash)
    assert sans_historique.half_move_clock == avec_repetition.half_move_clock

    mcts = chess_engine.MCTS(evaluator, 8192, 0)
    mcts.mcts_search(sans_historique, 0, 1.4, False, 8)
    mcts.reset_counters()
    mcts.mcts_search(avec_repetition, 0, 1.4, False, 8)
    c = mcts.get_counters()

    assert c.tt_hits == 0
    assert c.tt_misses == 1
    assert c.tt_position_matches == 1
    assert c.tt_context_rejects == 1


def test_rejet_semantique_sous_racine_est_compte_une_fois_en_batch(evaluator):
    cible = plateau("8/8/8/8/k7/8/8/KR6 w - - 1 2")
    parent = plateau("8/8/8/8/8/k7/8/KR6 b - - 2 1")
    mcts = chess_engine.MCTS(evaluator, 131071, 0)

    mcts.mcts_search(cible, 0, 1.4, False, 8)
    mcts.reset_counters()
    mcts.step_analysis(parent, 1, 1.4, 8)
    c = mcts.get_counters()
    tree = mcts.inspect_tree()

    assert c.tt_rule50_rejects == 1
    assert c.tt_misses == 2
    assert c.nn_calls == 2
    assert tree.en_vol == 0


def test_droits_de_roque_actuels_interdisent_un_hit_tt(evaluator):
    avec_roque = plateau("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    sans_roque = plateau("4k3/8/8/8/8/8/8/4K2R w - - 0 1")

    assert (avec_roque.get_evaluation_cache_key(0).position_hash
            != sans_roque.get_evaluation_cache_key(0).position_hash)

    mcts = chess_engine.MCTS(evaluator, 8192, 0)
    mcts.mcts_search(avec_roque, 0, 1.4, False, 8)
    mcts.reset_counters()
    mcts.mcts_search(sans_roque, 0, 1.4, False, 8)
    c = mcts.get_counters()

    assert c.tt_hits == 0
    assert c.tt_misses == 1
    assert c.tt_position_matches == 0


@pytest.mark.parametrize("depth", [-2, 8])
def test_mcts_refuse_une_profondeur_de_cache_hors_bornes(evaluator, depth):
    with pytest.raises(ValueError):
        chess_engine.MCTS(evaluator, 8192, depth)
