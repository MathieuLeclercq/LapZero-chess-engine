"""Identite de la position de base dans parse_position.

Deux commandes position peuvent porter la meme liste de coups sur deux bases
differentes : compteurs, droits de roque ou placement. La racine ne doit etre
decalee que si la base est la meme ; sinon le plateau et l'arbre sont
reconstruits.
"""
import os
import sys

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

import chess_engine
from core.lib import parse_uci_to_coords
from core.uci import UCIEngine


BASE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BASE_COMPTEURS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 36 19"
BASE_SANS_ROQUES = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


class _FauxMCTS:
    def __init__(self):
        self.appels = []

    def update_root(self, move_idx):
        self.appels.append(("update_root", move_idx))

    def reset_analysis(self):
        self.appels.append(("reset_analysis",))


def _moteur():
    return UCIEngine(evaluator=object(), mcts=_FauxMCTS())


def _commande_fen(fen, coups):
    return ["fen", *fen.split(), "moves", *coups]


def _rejouer(base_fen, coups):
    board = chess_engine.Chessboard()
    board.load_fen(base_fen)
    for uci_move in coups:
        orig_f, orig_r, dest_f, dest_r, promo = parse_uci_to_coords(uci_move)
        assert board.move_piece(orig_f, orig_r, dest_f, dest_r, promo)
    return board.to_fen()


def test_meme_liste_sur_deux_compteurs_reconstruit():
    moteur = _moteur()
    moteur.parse_position(_commande_fen(BASE, ["g1f3"]))
    moteur.mcts.appels.clear()

    moteur.parse_position(_commande_fen(BASE_COMPTEURS, ["g1f3"]))

    assert moteur.board.to_fen() == _rejouer(BASE_COMPTEURS, ["g1f3"])
    assert moteur.real_ply == 37
    assert ("reset_analysis",) in moteur.mcts.appels
    assert not any(appel[0] == "update_root" for appel in moteur.mcts.appels)


def test_faux_prolongement_avec_prefixe_reconstruit():
    moteur = _moteur()
    moteur.parse_position(_commande_fen(BASE, ["g1f3"]))
    moteur.mcts.appels.clear()

    moteur.parse_position(_commande_fen(BASE_COMPTEURS, ["g1f3", "g8f6"]))

    assert moteur.board.to_fen() == _rejouer(
        BASE_COMPTEURS, ["g1f3", "g8f6"])
    assert ("reset_analysis",) in moteur.mcts.appels
    assert not any(appel[0] == "update_root" for appel in moteur.mcts.appels)


def test_meme_base_et_prolongement_legal_reutilise_la_racine():
    moteur = _moteur()
    moteur.parse_position(_commande_fen(BASE, ["g1f3"]))
    moteur.mcts.appels.clear()

    moteur.parse_position(_commande_fen(BASE, ["g1f3"]))
    assert moteur.mcts.appels == []

    moteur.parse_position(_commande_fen(BASE, ["g1f3", "g8f6"]))
    assert [appel[0] for appel in moteur.mcts.appels] == ["update_root"]
    assert moteur.board.to_fen() == _rejouer(BASE, ["g1f3", "g8f6"])


def test_meme_placement_mais_roques_differents_reconstruit():
    moteur = _moteur()
    moteur.parse_position(_commande_fen(BASE, ["g1f3"]))
    moteur.mcts.appels.clear()

    moteur.parse_position(_commande_fen(BASE_SANS_ROQUES, ["g1f3"]))

    assert moteur.board.to_fen() == _rejouer(BASE_SANS_ROQUES, ["g1f3"])
    assert ("reset_analysis",) in moteur.mcts.appels


def test_le_cycle_du_bot_garde_la_reutilisation():
    """Le bot enregistre son propre coup dans last_move_list ; la commande
    suivante, meme base avec le coup adverse en plus, doit decaler la racine."""
    moteur = _moteur()
    moteur.parse_position(["startpos"])
    assert moteur.board.move_piece_uci("e2e4")
    moteur.last_move_list.append("e2e4")
    moteur.mcts.appels.clear()

    moteur.parse_position(["startpos", "moves", "e2e4", "e7e5"])

    assert [appel[0] for appel in moteur.mcts.appels] == ["update_root"]
    assert moteur.board.to_fen() == _rejouer(BASE, ["e2e4", "e7e5"])


def test_base_sans_coups_et_changement_de_base_reconstruisent():
    moteur = _moteur()
    moteur.parse_position(_commande_fen(BASE, ["g1f3"]))
    moteur.mcts.appels.clear()

    moteur.parse_position(_commande_fen(BASE_COMPTEURS, []))

    assert moteur.board.to_fen() == BASE_COMPTEURS
    assert moteur.last_move_list == []
    assert ("reset_analysis",) in moteur.mcts.appels

    moteur.mcts.appels.clear()
    moteur.parse_position(["startpos", "moves", "e2e4"])

    assert moteur.board.to_fen() == _rejouer(BASE, ["e2e4"])
    assert ("reset_analysis",) in moteur.mcts.appels
