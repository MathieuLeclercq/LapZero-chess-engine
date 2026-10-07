"""Tests du binding d'inference brute du banc de positions.

Un minuscule modele ONNX deterministe est construit avec onnx.helper : aucun
checkpoint, aucun GPU, aucune donnee externe. Le graphe applique un Slice des
4672 premiers elements aplatis pour la policy et un ReduceMean des plans pour
la value, ce qui rend chaque sortie previsible au bit pres.
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


def _construire_modele(chemin: Path, taille_policy: int) -> Path:
    import onnx
    from onnx import TensorProto, helper

    entree = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["N", 119, 8, 8])
    policy = helper.make_tensor_value_info(
        "policy", TensorProto.FLOAT, ["N", taille_policy])
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, ["N"])
    initialiseurs = [
        helper.make_tensor("starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("ends", TensorProto.INT64, [1], [taille_policy]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("value_axes", TensorProto.INT64, [3], [1, 2, 3]),
    ]
    noeuds = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=1),
        helper.make_node("Slice", ["flat", "starts", "ends", "axes"],
                         ["policy"]),
        helper.make_node("ReduceMean", ["input", "value_axes"], ["value"],
                         keepdims=0),
    ]
    graphe = helper.make_graph(noeuds, "position_bench_tiny", [entree],
                               [policy, value], initialiseurs)
    modele = helper.make_model(
        graphe, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(modele)
    onnx.save(modele, str(chemin))
    return chemin


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    chemin = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    return _construire_modele(chemin, 4672)


@pytest.fixture(scope="module")
def mauvais_onnx(tmp_path_factory):
    """Policy de 100 sorties au lieu de 4672 : modele incompatible."""
    chemin = tmp_path_factory.mktemp("onnx") / "mauvais.onnx"
    return _construire_modele(chemin, 100)


@pytest.fixture()
def evaluateur(tiny_onnx):
    return chess_engine.ONNXEvaluator(str(tiny_onnx), False)


def test_logits_bruts_et_derniere_ligne(evaluateur):
    states = np.zeros((3, 119, 8, 8), dtype=np.float32)
    states[2].fill(.25)
    logits, values = evaluateur.predict_batch(states)

    assert logits.shape == (3, 4672)
    assert values.shape == (3,)
    np.testing.assert_array_equal(logits[0], np.zeros(4672))
    np.testing.assert_array_equal(logits[2], np.full(4672, .25))
    assert values[2] == pytest.approx(.25)


@pytest.mark.parametrize("taille", [1, 8, 3])
def test_lots_de_tailles_variees(evaluateur, taille):
    states = np.full((taille, 119, 8, 8), 0.5, dtype=np.float32)

    logits, values = evaluateur.predict_batch(states)

    assert logits.shape == (taille, 4672)
    assert values.shape == (taille,)
    np.testing.assert_allclose(values, np.full(taille, 0.5))
    np.testing.assert_allclose(logits, np.full((taille, 4672), 0.5))


def test_refuse_une_entree_non_contigue(evaluateur):
    base = np.zeros((2, 119, 8, 16), dtype=np.float32)
    vue = base[..., :8]
    assert not vue.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError):
        evaluateur.predict_batch(vue)


def test_refuse_un_mauvais_type(evaluateur):
    with pytest.raises(TypeError):
        evaluateur.predict_batch(np.zeros((1, 119, 8, 8), dtype=np.float64))


def test_refuse_une_entree_non_tableau(evaluateur):
    with pytest.raises(TypeError):
        evaluateur.predict_batch([[0.0] * (119 * 64)])


def test_refuse_une_mauvaise_forme(evaluateur):
    for forme in [(1, 118, 8, 8), (1, 119, 8, 8, 1), (0, 119, 8, 8)]:
        with pytest.raises(ValueError):
            evaluateur.predict_batch(np.zeros(forme, dtype=np.float32))


def test_refuse_une_sortie_non_finie_et_reste_utilisable(evaluateur):
    states = np.zeros((1, 119, 8, 8), dtype=np.float32)
    states[0, 0, 0, 0] = np.nan

    with pytest.raises(RuntimeError):
        evaluateur.predict_batch(states)

    # La session doit rester utilisable apres le rejet.
    logits, values = evaluateur.predict_batch(
        np.zeros((1, 119, 8, 8), dtype=np.float32))
    np.testing.assert_array_equal(logits, np.zeros((1, 4672)))
    assert values[0] == pytest.approx(0.0)


def test_refuse_une_sortie_de_modele_incompatible(mauvais_onnx):
    evaluateur = chess_engine.ONNXEvaluator(str(mauvais_onnx), False)

    with pytest.raises(RuntimeError):
        evaluateur.predict_batch(np.zeros((1, 119, 8, 8), dtype=np.float32))


def test_le_prior_du_mcts_correspond_au_softmax_masque(evaluateur):
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    tenseur = np.ascontiguousarray(
        board.get_alphazero_tensor(), dtype=np.float32)

    logits, _ = evaluateur.predict_batch(tenseur[None, ...])
    legaux = list(board.get_legal_move_indices())
    choisis = logits[0][legaux].astype(np.float64)
    choisis -= choisis.max()
    exponentiels = np.exp(choisis)
    attendu = exponentiels / exponentiels.sum()

    mcts = chess_engine.MCTS(evaluateur, 8192, 0)
    mcts.step_analysis(board, 4, 1.4)
    stats = mcts.get_analysis_results()
    assert stats, "aucun coup visite, le test ne verifierait rien"

    for s in stats:
        rang = legaux.index(s.move_idx)
        assert s.prior == pytest.approx(attendu[rang], rel=1e-5, abs=1e-6)
