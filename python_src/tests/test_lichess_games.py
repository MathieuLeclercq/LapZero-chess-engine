import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.lichess_games as lg


def test_game_id_extraction():
    assert lg.game_id_from_url("https://lichess.org/787zsVup") == "787zsVup"
    assert lg.game_id_from_url("https://lichess.org/787zsVup/black#48") == "787zsVup"
    assert lg.game_id_from_url("https://lichess.org/787zsVup#48") == "787zsVup"
    assert lg.game_id_from_url("") is None


def test_ply_hint_extraction():
    assert lg.ply_hint_from_url("https://lichess.org/787zsVup/black#48") == 48
    assert lg.ply_hint_from_url("https://lichess.org/787zsVup") is None
    assert lg.ply_hint_from_url("https://lichess.org/787zsVup#notanumber") is None


def test_batched_respects_size_and_loses_nothing():
    items = list(range(701))
    batches = list(lg.batched(items, lg.MAX_IDS_PER_REQUEST))
    assert [len(b) for b in batches] == [300, 300, 101]
    assert [x for b in batches for x in b] == items


def _fake_pgn(game_id: str) -> str:
    return (
        f'[Event "Test"]\n'
        f'[Site "https://lichess.org/{game_id}"]\n'
        f'[Result "1-0"]\n'
        f"\n"
        f"1. e4 e5 2. Nf3 1-0\n\n"
    )


def test_fetch_writes_one_file_per_game(tmp_path):
    calls = []

    def fetcher(ids, token):
        calls.append(list(ids))
        return "".join(_fake_pgn(i) for i in ids)

    lg.fetch_games(["aaaaaaaa", "bbbbbbbb"], tmp_path, fetcher=fetcher)

    assert calls == [["aaaaaaaa", "bbbbbbbb"]]
    assert lg.cached_pgn("aaaaaaaa", tmp_path) is not None
    assert "https://lichess.org/bbbbbbbb" in lg.cached_pgn("bbbbbbbb", tmp_path)


def test_fetch_skips_what_is_already_cached(tmp_path):
    calls = []

    def fetcher(ids, token):
        calls.append(list(ids))
        return "".join(_fake_pgn(i) for i in ids)

    lg.fetch_games(["aaaaaaaa"], tmp_path, fetcher=fetcher)
    lg.fetch_games(["aaaaaaaa", "bbbbbbbb"], tmp_path, fetcher=fetcher)

    # Le second appel ne redemande que ce qui manque.
    assert calls == [["aaaaaaaa"], ["bbbbbbbb"]]


def test_fetch_with_everything_cached_makes_no_call(tmp_path):
    def fetcher(ids, token):
        return "".join(_fake_pgn(i) for i in ids)

    lg.fetch_games(["aaaaaaaa"], tmp_path, fetcher=fetcher)

    def exploding_fetcher(ids, token):
        raise AssertionError("aucun appel ne devait etre fait")

    lg.fetch_games(["aaaaaaaa"], tmp_path, fetcher=exploding_fetcher)


def test_missing_game_in_response_is_not_cached(tmp_path):
    """Une partie supprimee ou privee n'apparait pas dans la reponse."""
    def fetcher(ids, token):
        return _fake_pgn("aaaaaaaa")  # 'bbbbbbbb' manque

    lg.fetch_games(["aaaaaaaa", "bbbbbbbb"], tmp_path, fetcher=fetcher)

    assert lg.cached_pgn("aaaaaaaa", tmp_path) is not None
    assert lg.cached_pgn("bbbbbbbb", tmp_path) is None
