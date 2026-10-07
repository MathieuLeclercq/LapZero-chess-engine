"""Les compteurs d'instrumentation de la recherche.

Sans eux, le nombre d'inferences par seconde et le taux de succes de la table
ne sont pas observables de l'exterieur : le moteur n'avait aucune
instrumentation.
"""
import os
import subprocess
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

TAILLE_TT = 8192


@pytest.fixture(scope="module")
def modele_onnx(tmp_path_factory):
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chemin


@pytest.fixture(scope="module")
def evaluateur(modele_onnx):
    return chess_engine.ONNXEvaluator(str(modele_onnx), False)


def _plateau():
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    return board


def test_le_mcts_python_garde_l_evaluateur_en_vie(modele_onnx):
    code = r"""
import gc
import os
import sys
os.add_dll_directory(os.getcwd())
import chess_engine

def construire_mcts():
    evaluator = chess_engine.ONNXEvaluator(sys.argv[1], False)
    return chess_engine.MCTS(evaluator, 8192, 0)

mcts = construire_mcts()
gc.collect()
board = chess_engine.Chessboard()
board.set_startup_pieces()
mcts.step_analysis(board, 2, 1.4)
"""
    resultat = subprocess.run(
        [sys.executable, "-c", code, str(modele_onnx)],
        cwd=RACINE / "python_src", capture_output=True, text=True,
        timeout=60, check=False)

    assert resultat.returncode == 0, resultat.stderr


def test_les_compteurs_partent_a_zero(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    c = mcts.get_counters()

    assert c.nn_calls == 0
    assert c.nn_batches == 0
    assert c.tt_hits == 0
    assert c.tt_misses == 0
    assert c.tt_position_matches == 0
    assert c.tt_rule50_rejects == 0
    assert c.tt_context_rejects == 0
    assert c.tt_history_rejects == 0
    assert c.terminal_hits == 0
    assert c.waves == 0
    assert c.leaf_collisions == 0
    assert c.completed_simulations == 0


def test_une_recherche_declenche_des_inferences(evaluateur):
    """Sur un arbre neuf, nn_calls vaut simulations + 1.

    Le + 1 est l'expansion de la racine, que step_analysis fait hors de la
    boucle de simulations. Chaque simulation coute ensuite
    au plus une inference : moins si elle s'arrete sur un noeud terminal ou sur
    un succes de table.
    """
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 50, 1.4)

    c = mcts.get_counters()

    assert c.nn_calls > 0, "aucune inference comptee"
    assert c.nn_calls <= 51, f"plus d'inferences que de simulations : {c.nn_calls}"
    assert c.nn_batches == c.nn_calls, (
        "le chemin sequentiel doit faire un appel par position evaluee")


@pytest.mark.parametrize("batch_size", [0, 8])
def test_chaque_defaut_de_table_declenche_exactement_une_inference(
        evaluateur, batch_size):
    """Chaque chemin compte un défaut au même endroit que son inférence.

    Le chemin séquentiel le fait dans expand_node_single, le chemin batché au
    moment où la feuille est définitivement collectée.
    """
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 200, 1.4, batch_size)

    c = mcts.get_counters()

    assert c.tt_misses == c.nn_calls, (
        f"defauts {c.tt_misses} contre inferences {c.nn_calls}")
    assert c.tt_position_matches == (
        c.tt_hits + c.tt_rule50_rejects
        + c.tt_context_rejects + c.tt_history_rejects)


def test_les_succes_de_table_ne_sont_pas_comptes_deux_fois(evaluateur):
    """select_leaf consulte la table, et sur echec fait break ; l'appelant
    enchaine alors sur expand_node_single qui refait la meme consultation sur
    la meme position. L'echec n'est donc compte que dans expand_node_single.
    Si les deux le comptaient, tt_misses depasserait nn_calls."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 200, 1.4)

    c = mcts.get_counters()

    assert c.tt_misses <= c.nn_calls, "defauts comptes deux fois"


def test_les_consultations_de_table_sont_comptees(evaluateur):
    """Sur 400 simulations depuis la position de depart, l'arbre revisite
    forcement des positions, donc la table sert."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 400, 1.4)

    c = mcts.get_counters()

    assert c.tt_hits + c.tt_misses > 0
    assert 0.0 <= c.tt_hits / (c.tt_hits + c.tt_misses) <= 1.0


def test_reset_counters_remet_tout_a_zero(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 20, 1.4)
    assert mcts.get_counters().nn_calls > 0

    mcts.reset_counters()

    c = mcts.get_counters()
    assert (c.nn_calls, c.nn_batches, c.tt_hits, c.tt_misses,
            c.tt_position_matches, c.tt_rule50_rejects,
            c.tt_context_rejects, c.tt_history_rejects,
            c.terminal_hits, c.waves, c.leaf_collisions,
            c.completed_simulations) == (0,) * 12


def test_les_compteurs_sont_par_instance(evaluateur):
    a = chess_engine.MCTS(evaluateur, TAILLE_TT)
    b = chess_engine.MCTS(evaluateur, TAILLE_TT)

    a.step_analysis(_plateau(), 20, 1.4)

    assert a.get_counters().nn_calls > 0
    assert b.get_counters().nn_calls == 0


def test_les_chronometrages_sont_optionnels_et_couvrent_l_inference(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = _plateau()

    mcts.step_analysis(board, 8, 1.4, 4)
    disabled = mcts.get_last_timing()
    assert disabled.wall_ns == 0
    assert disabled.evaluator_ns == 0

    mcts.reset_analysis()
    mcts.set_timing_enabled(True)
    mcts.step_analysis(board, 8, 1.4, 4)
    enabled = mcts.get_last_timing()

    assert enabled.wall_ns > 0
    assert enabled.selection_ns > 0
    assert enabled.evaluator_ns > 0
    assert enabled.wall_ns >= enabled.evaluator_ns


def test_clear_evaluation_cache_rend_un_ancien_hit_miss(evaluateur):
    """Le pool chaud du banc remet la TT a froid entre deux mesures, sans
    recreer le MCTS : sans clear, la seconde mesure serait un banc de hits."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT, 0)
    board = _plateau()

    mcts.mcts_search(board, 1, 1.4, False, 8)
    mcts.reset_counters()
    mcts.mcts_search(board, 1, 1.4, False, 8)
    assert mcts.get_counters().tt_position_matches > 0, \
        "la fixture ne produit aucun hit : le clear ne prouverait rien"

    mcts.reset_counters()
    mcts.clear_evaluation_cache()
    mcts.mcts_search(board, 1, 1.4, False, 8)

    assert mcts.get_counters().tt_position_matches == 0
