"""Encodage et decodage des coups pour la policy AlphaZero, 4672 index.

Extrait de lib.py pour etre importable sans torch : le banc de puzzles fait
tourner 16 processus travailleurs, et torch plus onnx coutent 475 Mio a
l'import contre 2 Mio pour chess_engine seul. Les tables vivent maintenant en
C++ : encode_move et decode_move_index sont de simples enveloppes des fonctions
de chess_engine, et lib les reexporte pour ne casser aucun appelant.
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import chess_engine


def encode_move(orig_f, orig_r, dest_f, dest_r, promotion_type, is_black_turn):
    """Convertit un coup en un index plat (0 a 4671), delegue au C++.

    Renvoie -1 si le coup n'est pas encodable, comme l'ancienne version locale.
    """
    return chess_engine.encode_move(
        orig_f, orig_r, dest_f, dest_r, promotion_type, bool(is_black_turn))


def decode_move_index(board, index, is_black=None):
    """Inverse de encode_move : delegue au decodeur C++ de Chessboard.

    Le parametre is_black est conserve pour les appelants historiques, mais le
    plateau fait foi : une incoherence est signalee plutot que corrigee en
    silence. Le decodeur C++ inclut l'orientation du camp au trait et
    l'inference de promotion dame. L'index doit appartenir a [0, 4671] ;
    hors bornes, le decodage n'est pas defini.
    """
    if is_black is not None:
        attendu = (board.turn == chess_engine.Color.BLACK)
        if bool(is_black) != attendu:
            raise ValueError("is_black ne correspond pas au trait du plateau")
    return board.decode_move_index(index)


def gestion_promo_dame(board, orig_f, orig_r, dest_r, promo):
    # Promotion dame implicite (convention AlphaZero)
    piece = board.get_square(orig_f, orig_r).get_piece()
    if (piece.get_type() == chess_engine.PieceType.PAWN
            and promo == chess_engine.PieceType.NONE
            and (dest_r == 0 or dest_r == 7)):
        promo = chess_engine.PieceType.QUEEN
    return promo


def parse_uci_to_coords(uci_str):
    """
    Transforme 'e2e4' ou 'a7a8q' en (orig_f, orig_r, dest_f, dest_r, promotion)
    """
    # 1. Coordonnées de base (a-h -> 0-7, 1-8 -> 0-7)
    orig_f = ord(uci_str[0]) - ord('a')
    orig_r = int(uci_str[1]) - 1
    dest_f = ord(uci_str[2]) - ord('a')
    dest_r = int(uci_str[3]) - 1

    # 2. Gestion de la promotion (si la chaîne fait 5 caractères)
    promotion = chess_engine.PieceType.NONE
    if len(uci_str) == 5:
        promo_char = uci_str[4].lower()
        mapping = {
            'q': chess_engine.PieceType.QUEEN,
            'r': chess_engine.PieceType.ROOK,
            'b': chess_engine.PieceType.BISHOP,
            'n': chess_engine.PieceType.KNIGHT
        }
        promotion = mapping.get(promo_char, chess_engine.PieceType.NONE)

    return orig_f, orig_r, dest_f, dest_r, promotion


def coords_to_uci(orig_f, orig_r, dest_f, dest_r, promotion):
    """
    Transforme les coordonnées et le type de promotion en string UCI (ex: 'e7e8q')
    """
    files = "abcdefgh"
    # Les rangs dans ton moteur sont 0-indexed, en UCI ils sont 1-8
    move_uci = f"{files[orig_f]}{orig_r + 1}{files[dest_f]}{dest_r + 1}"

    # Ajout du suffixe de promotion si nécessaire
    if promotion != chess_engine.PieceType.NONE:
        mapping = {
            chess_engine.PieceType.QUEEN: 'q',
            chess_engine.PieceType.ROOK: 'r',
            chess_engine.PieceType.BISHOP: 'b',
            chess_engine.PieceType.KNIGHT: 'n'
        }
        move_uci += mapping.get(promotion, '')

    return move_uci
