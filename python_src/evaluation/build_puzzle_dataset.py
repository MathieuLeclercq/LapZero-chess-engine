"""
Construit les jeux de puzzles avec historique reel, depuis le CSV Lichess et
l'API d'export des parties.

Deux sorties disjointes :
  training_data/puzzles_train.txt  (gitignore)  rating 1300-2600
  data/puzzles_bench.txt           (committe)   rating 1000-2800, 4 tranches

Voir docs/etudes/evaluation-puzzles/protocole/2026-08-07-puzzle-pipeline-design.md
"""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import argparse
import collections
import csv
import hashlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

from core.lichess_games import (
    cached_pgn,
    fetch_games,
    game_id_from_url,
    ply_hint_from_url,
)

# Motifs tactiques uniquement. Sont exclus les libelles de phase, de longueur,
# d'issue et de provenance, qui ne decrivent pas un motif.
TACTICAL_THEMES = frozenset({
    "mateIn1", "mateIn2", "mateIn3",
    "fork", "pin", "skewer", "discoveredAttack", "doubleCheck",
    "hangingPiece", "sacrifice", "deflection", "trappedPiece",
    "attraction", "interference", "xRayAttack", "capturingDefender",
})

TRAIN_RATING_MIN = 1300
TRAIN_RATING_MAX = 2600

# Bornes incluses de chaque tranche du banc.
BENCH_BUCKETS = ((1000, 1449), (1450, 1899), (1900, 2349), (2350, 2800))

# Part des puzzles envoyes au banc. La repartition se fait par hachage du
# PuzzleId et non par tirage, pour rester identique si un CSV plus recent est
# telecharge : aucune contamination possible dans le temps.
BENCH_SHARE = 0.05


@dataclass(frozen=True)
class PuzzleRow:
    puzzle_id: str
    fen: str
    moves: list[str]
    rating: int
    themes: str
    game_url: str


def matches_themes(themes_field: str) -> bool:
    """Correspondance par jeton exact, pas par sous-chaine.

    L'ancienne extraction faisait `theme in themes_field`, qui ignore les
    frontieres de mots. Avec la liste retenue ce n'est pas un bug actif :
    aucun jeton n'est sous-chaine d'un autre theme Lichess, donc les deux
    methodes concordent sur des donnees reelles. La fragilite est latente et se
    declencherait si un nom de theme court etait ajoute. Un test synthetique
    verrouille la propriete.
    """
    return bool(TACTICAL_THEMES.intersection(themes_field.split()))


def bucket_of(rating: int) -> int | None:
    """Indice de tranche du banc, ou None si hors plage."""
    for index, (low, high) in enumerate(BENCH_BUCKETS):
        if low <= rating <= high:
            return index
    return None


def split_of(puzzle_id: str) -> str:
    """'bench' ou 'train', deterministe et stable dans le temps."""
    digest = hashlib.sha256(puzzle_id.encode("utf-8")).digest()
    # 16 bits suffisent largement pour un seuil a 5 pour cent.
    value = (digest[0] << 8 | digest[1]) / 65536.0
    return "bench" if value < BENCH_SHARE else "train"


def read_puzzle_csv(path: Path) -> Iterator[PuzzleRow]:
    """Lit le CSV Lichess et ne renvoie que les lignes passant le filtre de themes.

    Le filtre de rating n'est pas applique ici : les deux sorties ont des plages
    differentes, donc c'est l'appelant qui tranche.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            themes = row.get("Themes", "")
            if not matches_themes(themes):
                continue
            try:
                rating = int(row["Rating"])
            except (KeyError, ValueError):
                continue
            moves = row.get("Moves", "").split()
            # Un puzzle exploitable compte au moins la gaffe adverse et une
            # reponse. En dessous, les deux champs produits seraient vides.
            if len(moves) < 2:
                continue
            yield PuzzleRow(
                puzzle_id=row["PuzzleId"],
                fen=row["FEN"],
                moves=moves,
                rating=rating,
                themes=themes,
                game_url=row.get("GameUrl", ""),
            )


class MatchError:
    """Causes d'echec d'appariement, comptabilisees dans le rapport."""
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"
    GAME_MISSING = "game_missing"
    UNREADABLE = "unreadable"
    ILLEGAL_BLUNDER = "illegal_blunder"


@dataclass(frozen=True)
class MatchResult:
    start_fen: str
    moves_uci: list[str]


def position_key(board: chess.Board) -> str:
    """Trois premiers champs de la FEN : placement, trait, droits de roque.

    Sont exclus les compteurs, dont les conventions peuvent differer entre le
    CSV et le PGN reconstitue, ET la case en passant, pour la meme raison :
    Lichess ne la renseigne que si une capture est possible, python-chess a sa
    propre regle, et une divergence provoquerait un rejet a tort sur exactement
    les positions les plus interessantes.
    """
    return " ".join(board.fen().split()[:3])


def match_puzzle_in_game(pgn_text: str,
                         puzzle_fen: str,
                         ply_hint: int | None):
    """Rejoue la partie et localise le ply atteignant la position du puzzle.

    On ne se fie pas au numero de ply de la GameUrl : il ne sert qu'a lever une
    ambiguite. Le rejeu transforme une hypothese sur le format des donnees en
    verification effective.

    Renvoie un MatchResult, ou une constante de MatchError.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return MatchError.UNREADABLE

    try:
        board = game.board()
    except ValueError:
        return MatchError.UNREADABLE

    start_fen = board.fen()
    target = " ".join(puzzle_fen.split()[:3])

    moves: list[str] = []
    candidates: list[list[str]] = []

    if position_key(board) == target:
        candidates.append([])

    for move in game.mainline_moves():
        board.push(move)
        moves.append(move.uci())
        if position_key(board) == target:
            candidates.append(list(moves))

    if not candidates:
        return MatchError.NO_MATCH

    if len(candidates) == 1:
        return MatchResult(start_fen=start_fen, moves_uci=candidates[0])

    if ply_hint is None:
        return MatchError.AMBIGUOUS

    best = min(candidates, key=lambda c: abs(len(c) - ply_hint))
    return MatchResult(start_fen=start_fen, moves_uci=best)


def append_blunder(match: MatchResult, blunder_uci: str):
    """Prolonge l'historique de la gaffe adverse, premier coup du champ Moves.

    match_puzzle_in_game s'arrete SUR la position du puzzle, c'est-a-dire celle
    depuis laquelle le CSV joue Moves[0]. Cette gaffe n'est donc pas dans
    l'historique et doit y etre ajoutee : sans elle, la position ecrite precede
    le motif tactique d'un demi-coup et le premier coup de la solution y est
    illegal. C'est ce que faisait l'ancienne extraction, dont le portage a perdu
    l'etape.

    La legalite est verifiee et non supposee : un appariement leve par ply_hint
    peut avoir retenu la mauvaise occurrence de la position.

    Renvoie un MatchResult, ou MatchError.ILLEGAL_BLUNDER.
    """
    board = chess.Board(match.start_fen)
    for uci in match.moves_uci:
        board.push_uci(uci)

    try:
        move = chess.Move.from_uci(blunder_uci)
    except ValueError:
        return MatchError.ILLEGAL_BLUNDER
    if move not in board.legal_moves:
        return MatchError.ILLEGAL_BLUNDER

    return MatchResult(start_fen=match.start_fen,
                       moves_uci=[*match.moves_uci, blunder_uci])


def format_line(row: PuzzleRow, match: MatchResult) -> str:
    """Une ligne du fichier de sortie.

    Le premier coup du champ Moves du CSV est la gaffe de l'adversaire, ajoutee
    a l'historique par append_blunder. La solution commence donc au deuxieme, et
    son premier coup est legal dans la position obtenue en rejouant l'historique.
    """
    return "|".join((
        match.start_fen,
        " ".join(match.moves_uci),
        " ".join(row.moves[1:]),
        str(row.rating),
        row.themes,
    ))


def _read_token(path: Path) -> str | None:
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        return token or None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path("../training_data/lichess_db_puzzle.csv"))
    parser.add_argument("--cache", type=Path,
                        default=Path("../training_data/lichess_games_cache"))
    parser.add_argument("--out-train", type=Path,
                        default=Path("../training_data/puzzles_train.txt"))
    parser.add_argument("--out-bench", type=Path,
                        default=Path("../data/puzzles_bench.txt"))
    parser.add_argument("--train-target", type=int, default=100_000)
    parser.add_argument("--bench-per-bucket", type=int, default=1250)
    parser.add_argument("--token", type=Path,
                        default=Path("../lichess_token.txt"))
    parser.add_argument("--dry-run", action="store_true",
                        help="filtre et rapporte sans rien telecharger")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV introuvable : {args.csv}", file=sys.stderr)
        print("Telecharger depuis https://database.lichess.org/#puzzles",
              file=sys.stderr)
        return 2

    # --- 1. Selection ---
    train: list[PuzzleRow] = []
    bench: dict[int, list[PuzzleRow]] = collections.defaultdict(list)

    def bench_is_full() -> bool:
        return all(len(bench[i]) >= args.bench_per_bucket
                   for i in range(len(BENCH_BUCKETS)))

    print("Lecture et filtrage du CSV...")
    for row in read_puzzle_csv(args.csv):
        if not game_id_from_url(row.game_url):
            continue
        if split_of(row.puzzle_id) == "bench":
            index = bucket_of(row.rating)
            if index is not None and len(bench[index]) < args.bench_per_bucket:
                bench[index].append(row)
        elif len(train) < args.train_target:
            if TRAIN_RATING_MIN <= row.rating <= TRAIN_RATING_MAX:
                train.append(row)

        # Le CSV compte plusieurs millions de lignes : inutile de le lire en
        # entier quand les deux quotas sont atteints.
        if len(train) >= args.train_target and bench_is_full():
            break

    selected = train + [r for rows in bench.values() for r in rows]
    print(f"  {len(train)} pour l'entrainement, "
          f"{sum(len(v) for v in bench.values())} pour le banc")
    for index, (low, high) in enumerate(BENCH_BUCKETS):
        print(f"  tranche {low}-{high} : {len(bench[index])}")

    if args.dry_run:
        print("--dry-run : arret avant telechargement.")
        return 0

    # --- 2. Telechargement ---
    game_ids = list(dict.fromkeys(
        game_id_from_url(r.game_url) for r in selected))
    print(f"Recuperation de {len(game_ids)} parties distinctes "
          f"pour {len(selected)} puzzles...")
    fetch_games(game_ids, args.cache, token=_read_token(args.token))

    # --- 3. Appariement et ecriture ---
    counters: collections.Counter = collections.Counter()
    history_lengths: list[int] = []

    def emit(rows: list[PuzzleRow], out_path: Path) -> int:
        written = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            for row in rows:
                pgn = cached_pgn(game_id_from_url(row.game_url), args.cache)
                if pgn is None:
                    counters[MatchError.GAME_MISSING] += 1
                    continue
                result = match_puzzle_in_game(
                    pgn, row.fen, ply_hint_from_url(row.game_url))
                if isinstance(result, str):
                    counters[result] += 1
                    continue
                result = append_blunder(result, row.moves[0])
                if isinstance(result, str):
                    counters[result] += 1
                    continue
                handle.write(format_line(row, result) + "\n")
                history_lengths.append(len(result.moves_uci))
                written += 1
        return written

    written_train = emit(train, args.out_train)
    written_bench = emit(
        [r for rows in bench.values() for r in rows], args.out_bench)

    # --- 4. Rapport ---
    total = len(selected)
    rejected = sum(counters.values())
    short = sum(1 for n in history_lengths if n < 8)

    print(f"\n{'=' * 34}\n      RAPPORT DU PIPELINE\n{'=' * 34}")
    print(f"  Puzzles selectionnes   : {total}")
    print(f"  Ecrits (entrainement)  : {written_train}")
    print(f"  Ecrits (banc)          : {written_bench}")
    print(f"  Ecartes                : {rejected} "
          f"({100.0 * rejected / max(1, total):.2f} %)")
    for cause, count in sorted(counters.items()):
        print(f"    {cause:<14} : {count}")
    if history_lengths:
        print(f"  Historique < 8 plies   : {short} "
              f"({100.0 * short / len(history_lengths):.2f} %)")
    print("=" * 34)

    if rejected > 0.05 * max(1, total):
        print("\nPLUS DE 5 POUR CENT D'ECARTES. L'hypothese sur le format des "
              "donnees est a revoir avant d'aller plus loin.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
