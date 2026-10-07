"""Configuration des chemins sans modifier les sources."""
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _runtime

import numpy as np
import pytest

from core import uci
from pretraining import convert_pgn_to_binary as converter


def test_uci_model_cli_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPZERO_MODEL_PATH", str(tmp_path / "env.onnx"))
    selected = tmp_path / "chosen.onnx"
    assert uci.parse_args(["--model", str(selected)]).model == selected
    assert uci.parse_args([]).model == tmp_path / "env.onnx"


def test_uci_missing_model_fails_with_actionable_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        uci.main(["--model", str(tmp_path / "missing.onnx")])
    assert error.value.code == 2
    assert "--model" in capsys.readouterr().err


def test_converter_accepts_paths_and_rejects_zero_workers(tmp_path):
    args = converter.parse_args([
        "--input-pgn", str(tmp_path / "games.pgn"),
        "--output-dir", str(tmp_path / "shards"), "--workers", "1"])
    assert args.input_pgn == tmp_path / "games.pgn"
    assert args.output_dir == tmp_path / "shards"
    assert args.workers == 1
    with pytest.raises(SystemExit):
        converter.parse_args(["--workers", "0"])


def test_converter_spawn_uses_selected_output_directory(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    game = '[Event "Tiny"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n'
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        count = pool.submit(converter.worker_task, [game], 0, 0, tmp_path).result(30)
    assert count == 2
    with np.load(tmp_path / "shard_0_0.npz") as shard:
        assert shard["y_v"].tolist() == [1.0, -1.0]


def test_tournament_propagates_stockfish_path_to_tasks(tmp_path):
    from evaluation import tournament_elo as tournament
    executable = str(tmp_path / "stockfish.exe")
    tasks = list(tournament.generate_game_tasks(
        [("a", "b")], {"a": "a.onnx", "b": "b.onnx"}, executable))
    assert len(tasks) == tournament.GAMES_PER_PAIR
    assert all(task[-1] == executable for task in tasks)
    assert tournament.parse_args(["--stockfish", executable]).stockfish == executable
