"""Sémantique des profondeurs historiques de la clé TT."""

import os
import sys
from pathlib import Path

import pytest


RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine


CHECKPOINT = (RACINE / "data" / "checkpoints"
              / "2026_04_23_23h25_iter316_unsupervised.pt")
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(),
                                reason="checkpoint absent")

ORDRE_A = ("g1f3", "g8f6", "b1c3", "b8c6")
ORDRE_B = ("b1c3", "b8c6", "g1f3", "g8f6")


@pytest.fixture(scope="module")
def evaluator(tmp_path_factory):
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def board_from_moves(moves, fen=None):
    board = chess_engine.Chessboard()
    if fen is None:
        board.set_startup_pieces()
    else:
        board.load_fen(fen)
    for move in moves:
        assert board.move_piece_uci(move)
    return board


def remplir_racine(mcts, board):
    mcts.mcts_search(board, 0, 1.4, False, 8)


def test_h0_accepte_deux_historiques_differents(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 0)
    remplir_racine(mcts, board_from_moves(ORDRE_A))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B))

    c = mcts.get_counters()
    assert c.tt_hits == 1
    assert c.tt_history_rejects == 0


def test_profondeur_par_defaut_accepte_deux_historiques_differents(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192)
    remplir_racine(mcts, board_from_moves(ORDRE_A))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B))

    c = mcts.get_counters()
    assert c.tt_hits == 1
    assert c.tt_history_rejects == 0


@pytest.mark.parametrize("depth", [1, 3, 7])
def test_historique_recent_different_est_rejete(evaluator, depth):
    mcts = chess_engine.MCTS(evaluator, 8192, depth)
    remplir_racine(mcts, board_from_moves(ORDRE_A))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B))

    c = mcts.get_counters()
    assert c.tt_hits == 0
    assert c.tt_history_rejects == 1


def test_h1_accepte_des_historiques_anciens_differents(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 1)
    remplir_racine(mcts, board_from_moves(ORDRE_A + ("e2e3",)))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B + ("e2e3",)))

    c = mcts.get_counters()
    assert c.tt_hits == 1
    assert c.tt_history_rejects == 0


def test_h3_rejette_ces_memes_historiques_anciens(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 3)
    remplir_racine(mcts, board_from_moves(ORDRE_A + ("e2e3",)))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B + ("e2e3",)))

    assert mcts.get_counters().tt_history_rejects == 1


def test_h0_rejette_uniquement_la_repetition_courante(evaluator):
    repetition = board_from_moves(("g1f3", "g8f6", "f3g1", "f6g8"))
    sans_repetition = board_from_moves(
        ("f3g1", "f6g8", "c3b1", "c6b8"),
        "r1bqkb1r/pppppppp/2n2n2/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 0 1")

    a = repetition.get_evaluation_cache_key(0)
    b = sans_repetition.get_evaluation_cache_key(0)
    assert a.position_hash == b.position_hash
    assert a.half_move_clock == b.half_move_clock == 4
    assert a.total_moves_bucket == b.total_moves_bucket == 2
    assert a.repetition_category == 1
    assert b.repetition_category == 0

    mcts = chess_engine.MCTS(evaluator, 8192, 0)
    remplir_racine(mcts, sans_repetition)
    mcts.reset_counters()
    remplir_racine(mcts, repetition)

    c = mcts.get_counters()
    assert c.tt_context_rejects == 1
    assert c.tt_hits == 0
