"""
Oracle differentiel pour le generateur de coups.

OUTIL DE DIAGNOSTIC UNIQUEMENT. Ce script depend de python-chess et ne fait
pas partie de la validation du moteur : celle-ci est assuree exclusivement par
chess_perft, qui n'a aucune dependance externe. Voir
docs/superpowers/specs/2026-08-07-perft-design.md, section Architecture.

Usage :
    python dev_tools/fuzz_movegen.py [--positions 200000] [--seed 42]

Le defaut de 200000 est cale sur le critere de succes de la phase 1 : au
moins 1000 roques et 1000 promotions echantillonnes. Comptez environ 40 s.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess_engine

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

REFERENCE_FENS = [
    STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]

TACTICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "training_data", "tactics.txt")

PROMO_TO_CHAR = {
    chess_engine.PieceType.QUEEN: "q",
    chess_engine.PieceType.ROOK: "r",
    chess_engine.PieceType.BISHOP: "b",
    chess_engine.PieceType.KNIGHT: "n",
}

REF_PROMO_TO_CHAR = {
    chess.QUEEN: "q",
    chess.ROOK: "r",
    chess.BISHOP: "b",
    chess.KNIGHT: "n",
}

# Le roque est plafonne a deux par partie, quel que soit le biais de selection.
# Le nombre de roques echantillonnes depend donc du nombre de PARTIES, pas du
# nombre de positions. On ecourte les parties pour en jouer beaucoup plus a
# budget de positions constant.
MAX_PLIES_PER_GAME = 80

# Part des parties demarrant de la position initiale. Le reste part des
# positions de reference, ou le roque et la promotion sont disponibles
# immediatement.
STARTPOS_SHARE = 0.30

CHAR_TO_PROMO = {
    "q": chess_engine.PieceType.QUEEN,
    "r": chess_engine.PieceType.ROOK,
    "b": chess_engine.PieceType.BISHOP,
    "n": chess_engine.PieceType.KNIGHT,
    "": chess_engine.PieceType.NONE,
}


def engine_moves(board):
    """Ensemble des coups legaux du moteur C++, sous forme (of, orr, df, dr, promo)."""
    result = set()
    for move in board.get_all_legal_moves():
        orig, dest = move.get_orig_square(), move.get_dest_square()
        result.add((
            orig.get_file(), orig.get_rank(),
            dest.get_file(), dest.get_rank(),
            PROMO_TO_CHAR.get(move.get_promotion(), ""),
        ))
    return result


def reference_moves(fen):
    """Meme chose depuis python-chess, avec la meme convention de coordonnees."""
    ref = chess.Board(fen)
    result = set()
    for move in ref.legal_moves:
        result.add((
            move.from_square % 8, move.from_square // 8,
            move.to_square % 8, move.to_square // 8,
            REF_PROMO_TO_CHAR.get(move.promotion, ""),
        ))
    return result


def describe(move_tuple):
    files = "abcdefgh"
    of, orr, df, dr, promo = move_tuple
    return f"{files[of]}{orr + 1}{files[df]}{dr + 1}{promo}"


def is_castle(board, move_tuple):
    of, orr, df, _, _ = move_tuple
    piece = board.get_square(of, orr).get_piece()
    return piece.get_type() == chess_engine.PieceType.KING and abs(of - df) == 2


def pick_move(board, moves, rng, bias=0.35):
    """Selection aleatoire, biaisee vers les roques et les promotions.

    Le biais ne fausse pas le test, qui compare deux generateurs sur une meme
    position. Il concentre l'echantillonnage la ou les bugs se cachent :
    des parties purement aleatoires ne roquent quasiment jamais et
    n'atteignent presque jamais de promotion.
    """
    interesting = [m for m in moves if m[4] or is_castle(board, m)]
    if interesting and rng.random() < bias:
        return rng.choice(interesting)
    return rng.choice(sorted(moves))


def load_start_positions():
    positions = list(REFERENCE_FENS)
    if os.path.exists(TACTICS_PATH):
        with open(TACTICS_PATH, "r", encoding="utf-8") as handle:
            tactics = [line.strip() for line in handle if line.strip()]
        positions.extend(tactics)
        print(f"Charge {len(tactics)} FEN tactiques depuis {TACTICS_PATH}")
    else:
        print(f"Pas de fichier tactique a {TACTICS_PATH}, on continue sans.")
    return positions


def run(target_positions, seed):
    rng = random.Random(seed)
    start_positions = load_start_positions()

    checked = 0
    castles_seen = 0
    promotions_seen = 0
    games = 0

    while checked < target_positions:
        games += 1
        board = chess_engine.Chessboard()
        if rng.random() < STARTPOS_SHARE:
            board.load_fen(STARTPOS_FEN)
        else:
            board.load_fen(rng.choice(start_positions))

        for _ in range(MAX_PLIES_PER_GAME):
            if checked >= target_positions:
                break

            fen = board.to_fen()
            mine = engine_moves(board)
            theirs = reference_moves(fen)
            checked += 1

            if mine != theirs:
                print("\n*** DIVERGENCE ***")
                print(f"FEN : {fen}")
                extra = sorted(describe(m) for m in mine - theirs)
                missing = sorted(describe(m) for m in theirs - mine)
                print(f"Generes en trop par le moteur : {extra}")
                print(f"Manquants dans le moteur      : {missing}")
                return 1

            if not mine:
                break

            move = pick_move(board, mine, rng)
            if move[4]:
                promotions_seen += 1
            elif is_castle(board, move):
                castles_seen += 1

            of, orr, df, dr, promo = move
            if not board.move_piece(of, orr, df, dr, CHAR_TO_PROMO[promo], True):
                print("\n*** REFUS ***")
                print(f"FEN : {fen}")
                print(f"move_piece a refuse un coup qu'il a genere : {describe(move)}")
                return 1

            if board.game_state != chess_engine.GameState.ONGOING:
                break

            if checked % 5000 == 0:
                print(f"  {checked} positions verifiees "
                      f"({castles_seen} roques, {promotions_seen} promotions)",
                      flush=True)

    print(f"\nAucune divergence sur {checked} positions, {games} parties.")
    print(f"Roques joues      : {castles_seen}")
    print(f"Promotions jouees : {promotions_seen}")
    if castles_seen < 1000 or promotions_seen < 1000:
        print("\nATTENTION : moins de 1000 roques ou promotions echantillonnes.")
        print("Le critere de succes de la phase 1 n'est pas atteint, relancer")
        print("avec --positions plus eleve.")
        return 1
    return 0


def engine_perft(board, depth):
    """Perft de reference cote moteur C++, pilote depuis Python.

    Lent, mais la bisection ne l'appelle qu'a faible profondeur et seulement
    sur la branche fautive.
    """
    if depth == 0:
        return 1
    total = 0
    for move in sorted(engine_moves(board)):
        of, orr, df, dr, promo = move
        board.move_piece(of, orr, df, dr, CHAR_TO_PROMO[promo], False)
        total += engine_perft(board, depth - 1)
        board.undo_move()
    return total


def engine_divide(board, depth):
    """Sous-totaux par coup racine, cote moteur."""
    result = {}
    for move in sorted(engine_moves(board)):
        of, orr, df, dr, promo = move
        board.move_piece(of, orr, df, dr, CHAR_TO_PROMO[promo], False)
        result[describe(move)] = engine_perft(board, depth - 1)
        board.undo_move()
    return result


def reference_perft(ref, depth):
    if depth == 0:
        return 1
    total = 0
    for move in ref.legal_moves:
        ref.push(move)
        total += reference_perft(ref, depth - 1)
        ref.pop()
    return total


def reference_divide(fen, depth):
    """Sous-totaux par coup racine, cote python-chess."""
    ref = chess.Board(fen)
    result = {}
    for move in ref.legal_moves:
        ref.push(move)
        result[move.uci()] = reference_perft(ref, depth - 1)
        ref.pop()
    return result


def bisect(fen, depth):
    """Descend jusqu'a la position et au coup exacts ou les deux moteurs divergent."""
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    path = []

    while depth > 0:
        current_fen = board.to_fen()
        print(f"\nProfondeur {depth} : {current_fen}")

        mine = engine_moves(board)
        theirs = reference_moves(current_fen)
        if mine != theirs:
            print("\n*** DIVERGENCE SUR LA LISTE DE COUPS ***")
            print(f"FEN     : {current_fen}")
            print(f"Chemin  : {' '.join(path) if path else '(racine)'}")
            print(f"En trop : {sorted(describe(m) for m in mine - theirs)}")
            print(f"Manquant: {sorted(describe(m) for m in theirs - mine)}")
            return 1

        if depth == 1:
            print("\nListes identiques a la profondeur 1, aucune divergence trouvee.")
            return 0

        mine_div = engine_divide(board, depth)
        theirs_div = reference_divide(current_fen, depth)

        culprit = None
        for uci, count in sorted(mine_div.items()):
            if theirs_div.get(uci) != count:
                culprit = uci
                print(f"  coup fautif : {uci}, "
                      f"moteur {count}, reference {theirs_div.get(uci)}")
                break

        if culprit is None:
            print(f"\nAucun ecart : les deux moteurs concordent sur les "
                  f"{len(mine_div)} coups racine a la profondeur {depth}.")
            if not path:
                print("La position de depart est saine.")
            else:
                print("La divergence signalee plus haut n'est pas reproduite ici.")
            return 0

        of = ord(culprit[0]) - ord("a")
        orr = int(culprit[1]) - 1
        df = ord(culprit[2]) - ord("a")
        dr = int(culprit[3]) - 1
        promo = culprit[4] if len(culprit) == 5 else ""
        board.move_piece(of, orr, df, dr, CHAR_TO_PROMO[promo], False)
        path.append(culprit)
        depth -= 1

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bisect", nargs=2, metavar=("FEN", "DEPTH"),
                        help="localise la divergence sous cette position")
    args = parser.parse_args()

    if args.bisect:
        sys.exit(bisect(args.bisect[0], int(args.bisect[1])))
    sys.exit(run(args.positions, args.seed))


if __name__ == "__main__":
    main()
