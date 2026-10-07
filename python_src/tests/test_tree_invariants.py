"""Invariants de l'arbre de recherche.

Le piege principal du batching ne plante pas : select_leaf fait de l'expansion
paresseuse et continue de descendre apres un succes de table, donc une feuille
deja collectee pour le GPU peut recevoir des enfants pendant la meme collecte,
apres quoi expand_and_backup en creerait un second jeu. L'arbre serait corrompu
en silence. Ces controles sont le filet.
"""
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

TAILLE_TT = 8192

POSITIONS = [
    ("depart", None),
    ("milieu",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("finale", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


@pytest.fixture(scope="module")
def evaluateur(tmp_path_factory):
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def _plateau(fen):
    board = chess_engine.Chessboard()
    if fen is None:
        board.set_startup_pieces()
    else:
        board.load_fen(fen)
    return board


def test_un_arbre_vide_ne_signale_rien(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    rapport = mcts.inspect_tree()

    assert rapport.nodes == 0
    assert rapport.violations == 0


@pytest.mark.parametrize("nom,fen", POSITIONS)
def test_la_recherche_sequentielle_ne_viole_aucun_invariant(evaluateur, nom, fen):
    """Le passage de decouverte : la recherche actuelle fait reference.

    Si un invariant echoue ici, ne pas l'affaiblir. C'est soit un bug latent,
    soit une erreur du modele mental de la spec, et il faut determiner lequel
    avant de toucher au batching.
    """
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(fen), 400, 1.4)

    rapport = mcts.inspect_tree()

    assert rapport.violations == 0, f"{nom} : {list(rapport.messages)}"
    assert rapport.nodes > 1
    assert rapport.max_depth >= 1
    assert rapport.pending == 0
    assert rapport.root_visits == 400


def test_les_visites_de_la_racine_egalent_les_simulations(evaluateur):
    """step_analysis developpe la racine sans backup (mcts.cpp:374-377), et
    chaque simulation remonte exactement une fois par la racine. C'est le
    controle qui detectera un virtual loss mal annule."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = _plateau(None)

    mcts.step_analysis(board, 200, 1.4)
    assert sum(s.visits for s in mcts.get_analysis_results()) == 200

    mcts.step_analysis(board, 150, 1.4)
    assert sum(s.visits for s in mcts.get_analysis_results()) == 350


def test_le_rapport_grandit_avec_le_nombre_de_simulations(evaluateur):
    petit = chess_engine.MCTS(evaluateur, TAILLE_TT)
    petit.step_analysis(_plateau(None), 50, 1.4)

    grand = chess_engine.MCTS(evaluateur, TAILLE_TT)
    grand.step_analysis(_plateau(None), 400, 1.4)

    assert grand.inspect_tree().nodes > petit.inspect_tree().nodes


def test_inspect_tree_ne_voit_pas_l_arbre_de_mcts_search(evaluateur):
    """mcts_search construit sa racine en variable locale (mcts.cpp:237) et la
    detruit en revenant. C'est la raison pour laquelle la jambe invariants
    s'exerce sur step_analysis."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    mcts.mcts_search(_plateau(None), 100, 1.4, False)

    assert mcts.inspect_tree().nodes == 0


@pytest.mark.parametrize("depth", [-1, 0, 1, 3, 7])
@pytest.mark.parametrize("batch_size", [0, 8])
def test_la_politique_tt_ne_corrompt_pas_l_arbre(
        evaluateur, depth, batch_size):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT, depth)
    mcts.step_analysis(_plateau(None), 200, 1.4, batch_size)

    rapport = mcts.inspect_tree()
    assert rapport.violations == 0, list(rapport.messages)
    assert rapport.en_vol == 0
    assert rapport.pending == 0
    assert rapport.root_visits == 200
    assert sum(s.visits for s in mcts.get_analysis_results()) == 200
