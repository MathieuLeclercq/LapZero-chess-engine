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


@pytest.fixture(scope="module")
def evaluateur(tmp_path_factory):
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx_multicore"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def _plateau():
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    return board


def test_api_multicoeur_et_compteurs(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, 8192, 0)
    board = _plateau()
    fen_before = board.to_fen()

    mcts.step_analysis(board, 17, 1.4, 8, 4)

    report = mcts.inspect_tree()
    counters = mcts.get_counters()
    assert report.root_visits == 17
    assert report.pending == 0
    assert report.en_vol == 0
    assert report.violations == 0, list(report.messages)
    assert counters.waves > 0
    assert counters.completed_simulations == 17
    assert board.to_fen() == fen_before


def test_mcts_search_multicoeur_retourne_une_distribution(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, 8192, 0)
    policy = mcts.mcts_search(_plateau(), 17, 1.4, False, 8, 4)

    assert len(policy) == 4672
    assert sum(policy) == pytest.approx(1.0)


@pytest.mark.parametrize("workers", [0, -1])
def test_worker_count_invalide_est_rejete_avant_mutation(evaluateur, workers):
    mcts = chess_engine.MCTS(evaluateur, 8192, 0)

    with pytest.raises((ValueError, RuntimeError)):
        mcts.step_analysis(_plateau(), 8, 1.4, 8, workers)

    assert mcts.inspect_tree().nodes == 0


def test_multicoeur_refuse_un_batch_nul_avant_mutation(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, 8192, 0)

    with pytest.raises((ValueError, RuntimeError)):
        mcts.step_analysis(_plateau(), 8, 1.4, 0, 4)

    assert mcts.inspect_tree().nodes == 0


def test_appels_historiques_sans_worker_restent_valides(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, 8192, 0)
    mcts.step_analysis(_plateau(), 3, 1.4, 1)

    assert mcts.inspect_tree().root_visits == 3


def test_clear_evaluation_cache_equilibre_un_mcts_neuf(evaluateur):
    """En mono-worker, une recherche apres clear doit etre indiscernable d'une
    premiere recherche sur un objet neuf : meme TT vide, meme politique, memes
    compteurs. C'est la garantie que le pool chaud du banc ne mesure pas autre
    chose."""
    board = _plateau()

    chaud = chess_engine.MCTS(evaluateur, 8192, 0)
    chaud.mcts_search(board, 8, 1.4, False, 8)
    chaud.reset_counters()
    chaud.clear_evaluation_cache()
    pi_chaud = chaud.mcts_search(board, 8, 1.4, False, 8)
    c_chaud = chaud.get_counters()

    neuf = chess_engine.MCTS(evaluateur, 8192, 0)
    neuf.reset_counters()
    pi_neuf = neuf.mcts_search(board, 8, 1.4, False, 8)
    c_neuf = neuf.get_counters()

    assert pi_chaud == pi_neuf
    assert (c_chaud.nn_calls, c_chaud.nn_batches, c_chaud.tt_hits,
            c_chaud.tt_misses, c_chaud.completed_simulations) == (
        c_neuf.nn_calls, c_neuf.nn_batches, c_neuf.tt_hits,
        c_neuf.tt_misses, c_neuf.completed_simulations)
