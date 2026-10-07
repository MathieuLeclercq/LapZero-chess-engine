import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess_engine


def _legal_set(board):
    result = set()
    for move in board.get_all_legal_moves():
        orig, dest = move.get_orig_square(), move.get_dest_square()
        promo = move.get_promotion()
        result.add((orig.get_file(), orig.get_rank(),
                    dest.get_file(), dest.get_rank(), promo))
    return result


def test_uci_replay_reaches_the_same_position_as_load_fen():
    """Le controle central de la tache : le rejeu doit atteindre exactement la
    position attendue, et non une position decalee d'un ply.

    On compare l'ensemble des coups legaux et les trois premiers champs de la
    FEN. On ne compare PAS les hash Zobrist : checkEnPassant positionne le
    drapeau sur toute poussee double, alors que python-chess ne renseigne la
    case que si une capture est possible.
    """
    line = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"]

    ref = chess.Board()
    for uci in line:
        ref.push(chess.Move.from_uci(uci))
    target_fen = ref.fen()

    replayed = chess_engine.Chessboard()
    replayed.set_startup_pieces()
    for uci in line:
        assert replayed.move_piece_uci(uci), uci

    direct = chess_engine.Chessboard()
    direct.load_fen(target_fen)

    assert _legal_set(replayed) == _legal_set(direct)
    assert replayed.to_fen().split()[:3] == direct.to_fen().split()[:3]


def test_uci_replay_handles_castling():
    line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1"]  # petit roque

    replayed = chess_engine.Chessboard()
    replayed.set_startup_pieces()
    for uci in line:
        assert replayed.move_piece_uci(uci), uci

    ref = chess.Board()
    for uci in line:
        ref.push(chess.Move.from_uci(uci))

    assert replayed.to_fen().split()[:3] == ref.fen().split()[:3]


def test_uci_replay_handles_promotion_explicit_and_implicit():
    """Promotion explicite ('a7a8q') et implicite ('a7a8'), la seconde etant la
    convention AlphaZero de promotion en dame par defaut."""
    setup = "8/P7/8/8/8/8/7p/K6k w - - 0 1"

    explicit = chess_engine.Chessboard()
    explicit.load_fen(setup)
    assert explicit.move_piece_uci("a7a8q")

    implicit = chess_engine.Chessboard()
    implicit.load_fen(setup)
    assert implicit.move_piece_uci("a7a8")

    assert explicit.to_fen().split()[:3] == implicit.to_fen().split()[:3]

    ref = chess.Board(setup)
    ref.push(chess.Move.from_uci("a7a8q"))
    assert explicit.to_fen().split()[:3] == ref.fen().split()[:3]

    under = chess_engine.Chessboard()
    under.load_fen(setup)
    assert under.move_piece_uci("a7a8n")
    assert under.to_fen().split()[:3] != explicit.to_fen().split()[:3]


def test_uci_rejects_malformed_and_illegal():
    board = chess_engine.Chessboard()
    board.set_startup_pieces()

    assert not board.move_piece_uci("")
    assert not board.move_piece_uci("e2")
    assert not board.move_piece_uci("z9z9")
    assert not board.move_piece_uci("e2e4x")
    assert not board.move_piece_uci("e2e5")  # illegal


def test_uci_replay_populates_board_history():
    """Le but de tout le chantier : la pile d'historique doit etre remplie.

    Une position chargee par loadFEN seul n'a qu'une entree, ce qui la rend
    structurellement identifiable comme un puzzle. Apres rejeu, elle en a
    autant que de coups joues, plus la position initiale.
    """
    line = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"]

    direct = chess_engine.Chessboard()
    direct.load_fen(chess.Board().fen())
    assert len(direct.get_board_history()) == 1

    replayed = chess_engine.Chessboard()
    replayed.load_fen(chess.Board().fen())
    for uci in line:
        assert replayed.move_piece_uci(uci), uci

    assert len(replayed.get_board_history()) == len(line) + 1
