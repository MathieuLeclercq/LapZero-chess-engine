"""Verrouille l'encodage index de policy contre coup.

C'est la brique dont depend tout le scoring du banc : un aller-retour faux
ferait mesurer autre chose que ce qu'on croit mesurer, silencieusement.
"""
import os
import subprocess
import sys

import pytest

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

import chess_engine
from core.move_coding import (
    coords_to_uci,
    decode_move_index,
    encode_move,
    gestion_promo_dame,
    parse_uci_to_coords,
)

# Positions choisies pour couvrir les noirs au trait, les promotions et le roque.
# Les deux dernieres portent les promotions : 32 et 28 coups de promotion legaux.
# Attention en les remplacant, une position en echec n'en laisse aucune.
POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "7k/PPPPPPPP/8/8/8/8/8/7K w - - 0 1",
    "7K/8/8/8/8/8/pppppppp/7k b - - 0 1",
]

POSITIONS_AVEC_PROMOTIONS = POSITIONS[3:]


def _board(fen):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


@pytest.mark.parametrize("fen", POSITIONS)
def test_encode_decode_round_trip_on_every_legal_move(fen):
    board = _board(fen)
    is_black = board.turn == chess_engine.Color.BLACK
    indices = board.get_legal_move_indices()

    assert indices, f"aucun coup legal dans {fen}"

    for index in indices:
        o_f, o_r, d_f, d_r, promo = decode_move_index(board, index, is_black)
        assert encode_move(o_f, o_r, d_f, d_r, promo, is_black) == index


@pytest.mark.parametrize("fen", POSITIONS)
def test_uci_round_trip_on_every_legal_move(fen):
    """Le banc lit des coups en UCI et doit les retrouver comme index.

    C'est le chemin exact que suivra bench_metrics.uci_to_index, y compris le
    passage par gestion_promo_dame pour la promotion dame implicite.
    """
    board = _board(fen)
    is_black = board.turn == chess_engine.Color.BLACK

    for index in board.get_legal_move_indices():
        uci = coords_to_uci(*decode_move_index(board, index, is_black))

        o_f, o_r, d_f, d_r, promo = parse_uci_to_coords(uci)
        promo = gestion_promo_dame(board, o_f, o_r, d_r, promo)

        assert encode_move(o_f, o_r, d_f, d_r, promo, is_black) == index, uci


@pytest.mark.parametrize("fen", POSITIONS_AVEC_PROMOTIONS)
def test_promotions_are_actually_covered(fen):
    """Garde fou : une position en echec ne laisse aucune promotion legale et
    viderait silencieusement la couverture des tests d'aller-retour."""
    board = _board(fen)
    is_black = board.turn == chess_engine.Color.BLACK

    promotions = [
        i for i in board.get_legal_move_indices()
        if decode_move_index(board, i, is_black)[4] != chess_engine.PieceType.NONE
    ]

    assert not board.is_in_check(), f"{fen} est en echec, plus de promotions"
    assert len(promotions) >= 20, f"seulement {len(promotions)} promotions"


def test_the_four_promotion_choices_get_distinct_indices():
    """Les sous promotions ont leurs propres plans, la dame passe par le plan
    d'avance normal. Quatre index distincts et tous legaux, sinon deux
    promotions seraient confondues au comptage."""
    board = _board("7k/PPPPPPPP/8/8/8/8/8/7K w - - 0 1")

    indices = set()
    for suffixe in ("q", "r", "b", "n"):
        o_f, o_r, d_f, d_r, promo = parse_uci_to_coords(f"a7a8{suffixe}")
        promo = gestion_promo_dame(board, o_f, o_r, d_r, promo)
        indices.add(encode_move(o_f, o_r, d_f, d_r, promo, False))

    assert len(indices) == 4
    assert indices <= set(board.get_legal_move_indices())


def test_decode_rejects_a_mismatched_is_black():
    """Le plateau fait foi : un drapeau incoherent est signale, pas ignore.

    L'ancienne table Python decodait silencieusement dans le mauvais repere.
    """
    board = _board(POSITIONS[0])  # Blancs au trait
    index = board.get_legal_move_indices()[0]

    assert decode_move_index(board, index, is_black=False)

    with pytest.raises(ValueError):
        decode_move_index(board, index, is_black=True)


def test_decode_sans_drapeau_utilise_le_trait_du_plateau():
    board = _board(POSITIONS[2])  # Noirs au trait
    index = board.get_legal_move_indices()[0]

    assert decode_move_index(board, index) == decode_move_index(
        board, index, is_black=True)


def test_move_coding_does_not_import_torch():
    """La raison d'etre du module : 16 travailleurs important torch coutent
    environ 8 Gio de RAM, contre moins de 1 Gio sans lui."""
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, r'{RACINE}')\n"
        f"os.add_dll_directory(r'{RACINE}')\n"
        "import core.move_coding as move_coding\n"
        "charges = sorted(k for k in sys.modules if k.split('.')[0] == 'torch')\n"
        "assert not charges, charges\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
