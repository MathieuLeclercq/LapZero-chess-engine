import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.build_puzzle_dataset import (
    BENCH_BUCKETS,
    bucket_of,
    matches_themes,
    split_of,
)


def test_matches_real_lichess_theme_fields():
    assert matches_themes("mateIn1 short crushing")
    assert matches_themes("middlegame fork advantage")
    assert not matches_themes("endgame advantage long")
    assert not matches_themes("smotheredMate short crushing")


def test_theme_match_is_on_whole_tokens_not_substrings():
    """Discrimine reellement une implementation par sous-chaine.

    Avec la liste de themes retenue, aucun jeton n'est sous-chaine d'un autre
    theme Lichess : sur des donnees reelles, `theme in champ` et la
    correspondance par jeton donnent le meme resultat. La fragilite est donc
    latente, pas active. Ces cas synthetiques testent la propriete elle-meme,
    pour qu'elle reste vraie si la liste s'allonge un jour.
    """
    for field in ("xmateIn1x", "premateIn1", "mateIn1suffix", "forked", "pinned"):
        assert not matches_themes(field), field

    # Le meme jeton, isole, doit matcher.
    assert matches_themes("mateIn1")
    assert matches_themes("fork")
    assert matches_themes("pin")


def test_theme_requires_at_least_one_match():
    assert not matches_themes("")
    assert matches_themes("pin")


def test_bucket_boundaries_are_contiguous_and_inclusive():
    assert bucket_of(1000) == 0
    assert bucket_of(1449) == 0
    assert bucket_of(1450) == 1
    assert bucket_of(1899) == 1
    assert bucket_of(1900) == 2
    assert bucket_of(2349) == 2
    assert bucket_of(2350) == 3
    assert bucket_of(2800) == 3


def test_bucket_rejects_out_of_range():
    assert bucket_of(999) is None
    assert bucket_of(2801) is None


def test_buckets_cover_the_declared_range_without_overlap():
    seen = set()
    for rating in range(1000, 2801):
        b = bucket_of(rating)
        assert b is not None, rating
        seen.add(b)
    assert seen == set(range(len(BENCH_BUCKETS)))


def test_split_is_deterministic():
    assert split_of("abc12") == split_of("abc12")


def test_split_produces_both_sides_in_roughly_the_declared_ratio():
    ids = [f"p{i:05d}" for i in range(20000)]
    bench = sum(1 for i in ids if split_of(i) == "bench")
    # Cible 5 pour cent, tolerance large : on teste la mecanique, pas la
    # qualite statistique de sha256.
    assert 0.03 < bench / len(ids) < 0.07


import chess
import chess.pgn

from evaluation.build_puzzle_dataset import (
    MatchError,
    MatchResult,
    match_puzzle_in_game,
    position_key,
)


def _pgn_from_uci(uci_moves: list[str]) -> str:
    """Fabrique un PGN a partir d'une liste de coups UCI."""
    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for uci in uci_moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    game.headers["Site"] = "https://lichess.org/testtest"
    return str(game)


def _fen_after(uci_moves: list[str]) -> str:
    board = chess.Board()
    for uci in uci_moves:
        board.push(chess.Move.from_uci(uci))
    return board.fen()


def test_position_key_keeps_three_fields_only():
    board = chess.Board()
    assert position_key(board) == (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq")


def test_match_finds_the_right_ply_and_returns_the_moves():
    line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    result = match_puzzle_in_game(
        _pgn_from_uci(line), _fen_after(line), ply_hint=None)

    assert isinstance(result, MatchResult)
    assert result.moves_uci == line
    assert result.start_fen == chess.STARTING_FEN


def test_match_finds_a_position_in_the_middle_of_the_game():
    line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    target = _fen_after(line[:3])

    result = match_puzzle_in_game(_pgn_from_uci(line), target, ply_hint=None)

    assert result.moves_uci == line[:3]


def test_match_ignores_en_passant_and_counter_fields():
    """Une divergence de convention sur la case en passant ne doit pas rejeter.

    python-chess n'ecrit la case que si une capture est reellement possible :
    apres 1.e4 il produit '-'. Une source utilisant la convention
    inconditionnelle ecrirait 'e3'. On forge donc 'e3' pour exercer vraiment
    cette divergence, et non un champ identique des deux cotes.
    """
    line = ["e2e4"]
    fields = _fen_after(line).split()
    assert fields[3] == "-", "python-chess a change de convention"

    forged = f"{fields[0]} {fields[1]} {fields[2]} e3 7 42"

    result = match_puzzle_in_game(_pgn_from_uci(line), forged, ply_hint=None)

    assert result.moves_uci == line


def test_no_match_is_reported():
    line = ["e2e4", "e7e5"]
    unreachable = _fen_after(["d2d4", "d7d5", "c2c4"])

    result = match_puzzle_in_game(
        _pgn_from_uci(line), unreachable, ply_hint=None)

    assert result == MatchError.NO_MATCH


def test_repetition_without_hint_is_rejected_as_ambiguous():
    # Nf3 Nf6 Ng1 Ng8 ramene a la position de depart : deux plies partagent
    # placement, trait et droits de roque.
    line = ["g1f3", "g8f6", "f3g1", "f6g8"]
    target = chess.STARTING_FEN

    result = match_puzzle_in_game(_pgn_from_uci(line), target, ply_hint=None)

    assert result == MatchError.AMBIGUOUS


def test_repetition_with_hint_picks_the_closest_ply():
    line = ["g1f3", "g8f6", "f3g1", "f6g8"]
    target = chess.STARTING_FEN

    result = match_puzzle_in_game(_pgn_from_uci(line), target, ply_hint=4)

    assert result.moves_uci == line


from evaluation.build_puzzle_dataset import PuzzleRow, format_line


def test_format_line_has_five_fields_and_no_separator_inside():
    row = PuzzleRow(
        puzzle_id="abc12",
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        moves=["e2e4", "e7e5", "g1f3"],
        rating=1500,
        themes="mateIn2 short",
        game_url="https://lichess.org/testtest#4",
    )
    match = MatchResult(start_fen=chess.STARTING_FEN, moves_uci=["e2e4", "e7e5"])

    line = format_line(row, match)
    fields = line.split("|")

    assert len(fields) == 5
    assert fields[0] == chess.STARTING_FEN
    assert fields[1] == "e2e4 e7e5"
    # Le premier coup de Moves est la gaffe adverse, ajoutee a l'historique par
    # append_blunder. La solution commence donc au deuxieme.
    assert fields[2] == "e7e5 g1f3"
    assert fields[3] == "1500"
    assert fields[4] == "mateIn2 short"
    assert "\n" not in line


def test_solution_first_move_is_legal_after_replaying_history():
    """L'invariant que le portage de l'ancienne extraction avait perdu.

    La position ecrite doit etre celle ou le solveur a le trait : apres rejeu de
    l'historique, le premier coup de la solution doit etre legal. Sans
    append_blunder l'historique s'arrete un demi-coup trop tot, avant le motif
    tactique, et ce coup y est illegal. Aucun des tests precedents ne
    l'exercait, et les 105 000 lignes produites etaient toutes fausses.
    """
    line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    puzzle_ply = 4                      # position depuis laquelle part Moves
    puzzle_fen = _fen_after(line[:puzzle_ply])
    blunder, solution_move = line[4], line[5]

    row = PuzzleRow(
        puzzle_id="abc12",
        fen=puzzle_fen,
        moves=[blunder, solution_move],
        rating=1500,
        themes="mateIn2 short",
        game_url="https://lichess.org/testtest#4",
    )

    match = match_puzzle_in_game(_pgn_from_uci(line), puzzle_fen, ply_hint=None)
    match = bpd.append_blunder(match, row.moves[0])
    fields = format_line(row, match).split("|")

    board = chess.Board(fields[0])
    for uci in fields[1].split():
        board.push_uci(uci)

    assert fields[2].split() == [solution_move]
    assert chess.Move.from_uci(solution_move) in board.legal_moves


def test_append_blunder_adds_exactly_one_ply():
    line = ["e2e4", "e7e5", "g1f3"]
    match = MatchResult(start_fen=chess.STARTING_FEN, moves_uci=line[:2])

    result = bpd.append_blunder(match, line[2])

    assert result.moves_uci == line
    assert result.start_fen == chess.STARTING_FEN


def test_append_blunder_rejects_a_blunder_that_is_not_legal():
    """Un appariement leve par ply_hint peut avoir retenu la mauvaise occurrence.

    On veut alors un rejet comptabilise, pas une ligne silencieusement fausse.
    """
    match = MatchResult(start_fen=chess.STARTING_FEN, moves_uci=["e2e4"])

    assert bpd.append_blunder(match, "e2e4") == MatchError.ILLEGAL_BLUNDER
    assert bpd.append_blunder(match, "pouet") == MatchError.ILLEGAL_BLUNDER


def test_read_puzzle_csv_skips_rows_with_fewer_than_two_moves(tmp_path):
    """Sans gaffe ET reponse, les deux champs produits seraient vides."""
    csv_path = tmp_path / "puzzles.csv"
    header = ("PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,"
              "Themes,GameUrl,OpeningTags")
    fen = chess.STARTING_FEN
    rows = [
        f"solo,{fen},e2e4,1500,80,90,120,mateIn2 short,https://lichess.org/aaaaaaaa,",
        f"vide,{fen},,1500,80,90,120,mateIn2 short,https://lichess.org/bbbbbbbb,",
        f"bon,{fen},e2e4 e7e5,1500,80,90,120,mateIn2 short,https://lichess.org/cccccccc,",
    ]
    csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    lus = list(bpd.read_puzzle_csv(csv_path))

    assert [r.puzzle_id for r in lus] == ["bon"]


def test_train_and_bench_never_share_a_puzzle_id():
    """La disjonction est une exigence : un banc partageant des puzzles avec
    l'entrainement mesurerait la memorisation, pas la capacite tactique."""
    ids = [f"p{i:06d}" for i in range(50000)]
    train = {i for i in ids if split_of(i) == "train"}
    bench = {i for i in ids if split_of(i) == "bench"}

    assert train and bench
    assert train.isdisjoint(bench)
    assert len(train) + len(bench) == len(ids)


import evaluation.build_puzzle_dataset as bpd


def _find_id_with_split(wanted: str, prefix: str) -> str:
    """Cherche un PuzzleId tombant du cote voulu de la repartition."""
    for i in range(100000):
        candidate = f"{prefix}{i}"
        if split_of(candidate) == wanted:
            return candidate
    raise AssertionError(f"aucun id '{wanted}' trouve")


def test_end_to_end_on_synthetic_data(tmp_path, monkeypatch, capsys):
    """Execute main() de bout en bout, sans reseau.

    Le cache est prerempli, donc fetch_games ne trouve rien de manquant et ne
    fait aucun appel. Verifie l'assemblage : selection, appariement, ecriture
    des deux fichiers et code de sortie.
    """
    line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    # Convention Lichess : la FEN du puzzle est la position DEPUIS laquelle part
    # le champ Moves, dont le premier coup est la gaffe adverse. Ici Moves
    # commence par f1b5, donc la FEN est celle d'avant f1b5, soit 4 plies.
    blunder_ply = 4
    puzzle_fen = _fen_after(line[:blunder_ply])
    # Historique attendu en sortie : jusqu'a la FEN du puzzle, gaffe incluse.
    history_attendu = line[:blunder_ply + 1]
    game_id = "abcd1234"

    cache = tmp_path / "cache"
    cache.mkdir()
    pgn = _pgn_from_uci(line).replace(
        "https://lichess.org/testtest", f"https://lichess.org/{game_id}")
    (cache / f"{game_id}.pgn").write_text(pgn, encoding="utf-8")

    train_id = _find_id_with_split("train", "T")
    bench_id = _find_id_with_split("bench", "B")

    csv_path = tmp_path / "puzzles.csv"
    header = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
    rows = [
        # rating 1500 : dans la plage d'entrainement
        f"{train_id},{puzzle_fen},f1b5 a7a6 b5c6,1500,80,90,100,mateIn2 short,"
        f"https://lichess.org/{game_id}#{blunder_ply},",
        # rating 1200 : hors plage d'entrainement, mais dans la tranche 0 du banc
        f"{bench_id},{puzzle_fen},f1b5 a7a6 b5c6,1200,80,90,100,fork short,"
        f"https://lichess.org/{game_id}#{blunder_ply},",
    ]
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    out_train = tmp_path / "train.txt"
    out_bench = tmp_path / "bench.txt"

    monkeypatch.setattr(sys, "argv", [
        "build_puzzle_dataset.py",
        "--csv", str(csv_path),
        "--cache", str(cache),
        "--out-train", str(out_train),
        "--out-bench", str(out_bench),
        "--token", str(tmp_path / "absent_token.txt"),
        "--train-target", "10",
        "--bench-per-bucket", "10",
    ])

    assert bpd.main() == 0

    train_lines = out_train.read_text(encoding="utf-8").splitlines()
    bench_lines = out_bench.read_text(encoding="utf-8").splitlines()

    assert len(train_lines) == 1
    assert len(bench_lines) == 1

    fields = train_lines[0].split("|")
    assert len(fields) == 5
    assert fields[0] == chess.STARTING_FEN
    # Les coups rejoues vont du debut jusqu'a la position du puzzle, gaffe incluse.
    assert fields[1] == " ".join(history_attendu)
    # La solution est Moves[1:], la gaffe etant desormais dans les coups rejoues.
    assert fields[2] == "a7a6 b5c6"
    # L'invariant, verifie aussi de bout en bout : le solveur a le trait.
    board = chess.Board(fields[0])
    for uci in fields[1].split():
        board.push_uci(uci)
    assert chess.Move.from_uci("a7a6") in board.legal_moves
    assert fields[3] == "1500"

    assert bench_lines[0].split("|")[3] == "1200"

    out = capsys.readouterr().out
    assert "Ecartes                : 0" in out


def test_end_to_end_counts_a_missing_game(tmp_path, monkeypatch, capsys):
    """Une partie absente du cache doit etre comptee, pas ignoree en silence.

    main() appelle legitimement fetch_games. On remplace donc la fonction par
    une version inerte qui laisse le cache vide, ce qui simule exactement le
    cas reel d'une partie supprimee ou privee. Sans ce remplacement, le test
    declencherait un vrai appel a l'API Lichess : une suite de tests ne doit
    pas dependre du reseau ni solliciter un service tiers a chaque execution.
    """
    calls = []
    monkeypatch.setattr(bpd, "fetch_games",
                        lambda *a, **k: calls.append(a))

    csv_path = tmp_path / "puzzles.csv"
    header = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
    train_id = _find_id_with_split("train", "T")
    csv_path.write_text(
        header + "\n"
        f"{train_id},{chess.STARTING_FEN},e2e4 e7e5,1500,80,90,100,fork short,"
        f"https://lichess.org/zzzz9999,\n",
        encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_puzzle_dataset.py",
        "--csv", str(csv_path),
        "--cache", str(tmp_path / "vide"),
        "--out-train", str(tmp_path / "train.txt"),
        "--out-bench", str(tmp_path / "bench.txt"),
        "--token", str(tmp_path / "absent_token.txt"),
    ])

    # 100 pour cent d'ecartes : le programme doit sortir en erreur.
    assert bpd.main() == 1
    assert "game_missing" in capsys.readouterr().out
    # fetch_games a bien ete appele, mais sur notre version inerte.
    assert len(calls) == 1
