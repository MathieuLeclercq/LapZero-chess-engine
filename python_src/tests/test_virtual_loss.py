"""Le virtual loss, approche LC0 : un compteur separe qui n'entre que dans le
denominateur du terme U.

Tant que rien ne l'incremente, ce champ ne doit rien changer. Les tests de
comportement reel arrivent avec la boucle batchee.
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
DEPART = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture(scope="module")
def evaluateur(tmp_path_factory):
    import evaluation.puzzle_bench as puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def _plateau(fen=DEPART):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


def test_aucun_noeud_ne_reste_en_vol_apres_une_recherche(evaluateur):
    """Un virtual loss non annule laisserait des noeuds en vol, ce qui
    fausserait le terme U de toutes les recherches suivantes."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 200, 1.4)

    assert mcts.inspect_tree().en_vol == 0


def test_les_visites_de_la_racine_egalent_les_simulations(evaluateur):
    """Rappel du filet existant : c'est le controle qui detecte un virtual loss
    mal annule, puisqu'il gonflerait ce compte."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = _plateau()

    mcts.step_analysis(board, 200, 1.4)

    assert sum(s.visits for s in mcts.get_analysis_results()) == 200


def test_la_recherche_reste_deterministe(evaluateur):
    """Deux recherches identiques doivent donner exactement la meme
    distribution : sans Dirichlet, rien n'introduit d'alea."""
    board_a = _plateau()
    board_b = _plateau()

    pi_a = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(board_a, 200, 1.4, False)
    pi_b = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(board_b, 200, 1.4, False)

    assert list(pi_a) == list(pi_b)


POSITIONS = [
    ("depart", DEPART),
    ("ouverture",
     "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
    ("milieu",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("finale", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


@pytest.mark.parametrize("nom,fen", POSITIONS)
def test_aucun_noeud_en_vol_sur_differentes_positions(evaluateur, nom, fen):
    """Verifie l'absence de noeuds en vol et la coherence des invariants
    sur des topologies d'arbres variees."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(fen), 200, 1.4)

    rapport = mcts.inspect_tree()
    assert rapport.en_vol == 0, f"{nom} : noeuds en vol = {rapport.en_vol}"
    assert rapport.violations == 0, f"{nom} : {list(rapport.messages)}"


def test_root_shifting_ne_laisse_aucun_noeud_en_vol(evaluateur):
    """En jeu reel, la racine est deplacee vers le coup joue : verifie
    qu'aucun noeud en vol ne subsiste apres extract_child ni lors du cumul."""
    from core.lib import decode_move_index

    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = _plateau()
    mcts.step_analysis(board, 100, 1.4)

    stats = mcts.get_analysis_results()
    assert len(stats) > 0
    best_move = stats[0].move_idx

    mcts.update_root(best_move)
    is_black = (board.turn == chess_engine.Color.BLACK)
    o_f, o_r, d_f, d_r, promo = decode_move_index(board, best_move, is_black)
    board.move_piece(o_f, o_r, d_f, d_r, promo)

    rapport_apres_shift = mcts.inspect_tree()
    assert rapport_apres_shift.en_vol == 0
    assert rapport_apres_shift.violations == 0

    mcts.step_analysis(board, 100, 1.4)
    rapport_apres_cumul = mcts.inspect_tree()
    assert rapport_apres_cumul.en_vol == 0
    assert rapport_apres_cumul.violations == 0


def test_position_de_mat_ne_laisse_aucun_noeud_en_vol(evaluateur):
    """Sur une position deja en echec et mat, la racine est terminale des le
    depart : elle ne doit developper aucun enfant ni laisser de noeud en vol."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    # Mat du berger termine (les Noirs sont echec et mat)
    board = _plateau("r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    mcts.step_analysis(board, 50, 1.4)

    rapport = mcts.inspect_tree()
    assert rapport.en_vol == 0
    assert rapport.violations == 0
    assert rapport.nodes == 1


@pytest.mark.parametrize("taille", [1, 2, 8, 32])
def test_la_boucle_batchee_respecte_les_invariants(evaluateur, taille):
    """Chaque lot doit rendre un arbre coherent et annuler tous ses virtual
    losses avant de rendre la main."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    mcts.step_analysis(_plateau(), 400, 1.4, taille)

    rapport = mcts.inspect_tree()
    assert rapport.violations == 0, list(rapport.messages)
    assert rapport.en_vol == 0, "virtual loss non annule"
    assert rapport.pending == 0, "feuille Pending non liberee"
    assert rapport.root_visits == 400


@pytest.mark.parametrize("taille", [1, 2, 8, 32])
def test_la_boucle_batchee_compte_exactement_les_simulations(evaluateur, taille):
    """Une simulation ne doit etre ni perdue lors d'une collision, ni comptee
    deux fois lors du backup du lot."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    mcts.step_analysis(_plateau(), 400, 1.4, taille)

    assert sum(s.visits for s in mcts.get_analysis_results()) == 400


def test_batch_1_est_identique_a_la_boucle_sequentielle(evaluateur):
    """Avec un seul element, poser puis annuler le virtual loss ne doit rien
    modifier : le chemin batche est la reference du chemin sequentiel."""
    sequentiel = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(
        _plateau(), 400, 1.4, False, 0)
    batche = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(
        _plateau(), 400, 1.4, False, 1)

    assert list(sequentiel) == list(batche)


def test_batch_1_compte_les_terminaux_comme_le_chemin_sequentiel(evaluateur):
    """La decouverte d'un mat sans coup legal est une TT miss. Elle ne devient
    terminal_hits qu'aux simulations suivantes, une fois le noeud marque."""
    fen_mat_en_un = "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"

    sequentiel = chess_engine.MCTS(evaluateur, TAILLE_TT)
    sequentiel.mcts_search(_plateau(fen_mat_en_un), 400, 1.4, False, 0)

    batche = chess_engine.MCTS(evaluateur, TAILLE_TT)
    batche.mcts_search(_plateau(fen_mat_en_un), 400, 1.4, False, 1)

    assert batche.get_counters().terminal_hits == sequentiel.get_counters().terminal_hits
