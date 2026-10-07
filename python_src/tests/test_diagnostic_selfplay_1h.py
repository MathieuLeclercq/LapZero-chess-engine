import importlib.util
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import psutil

SCRIPT = Path(__file__).resolve().parents[1] / "dev_tools" / "diagnostic_selfplay_1h.py"


@pytest.fixture
def bench():
    assert SCRIPT.is_file(), "Le nouveau banc doit exister"
    spec = importlib.util.spec_from_file_location("diagnostic_selfplay_1h", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_discovery_uses_iteration_number(bench, tmp_path):
    folder = tmp_path / "data" / "checkpoints" / "onnx"
    folder.mkdir(parents=True)
    for name in ("x_iter436_unsupervised.onnx", "x_iter999_unsupervised.onnx",
                 "x_iter1000_unsupervised.onnx", "x_iter9000_unsupervised.pt"):
        (folder / name).write_bytes(b"model")
    old, new = bench.resolve_models(tmp_path, None, None)
    assert old.name == "x_iter436_unsupervised.onnx"
    assert new.name == "x_iter1000_unsupervised.onnx"


def test_missing_old_checkpoint_is_actionable(bench, tmp_path):
    with pytest.raises(ValueError, match="old-model"):
        bench.resolve_models(tmp_path, None, None)


def test_partial_abba_does_not_claim_comparison(bench):
    rows = [dict(kind="fixed_result", position="start", pass_index=i,
                 model=model, elapsed_s=seconds, nodes=100, max_depth=5)
            for i, model, seconds in [(0, "old", 2), (1, "new", 4), (2, "new", 4)]]
    assert bench.fixed_comparisons(rows) == []
    rows.append(dict(kind="fixed_result", position="start", pass_index=3,
                     model="old", elapsed_s=2, nodes=200, max_depth=7))
    result = bench.fixed_comparisons(rows)[0]
    assert result["time_ratio_new_old"] == 2.0
    assert result["nodes_old"] == 150


def test_truncated_last_event_keeps_completed_results(bench, tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind":"ok"}\n{"kind":', encoding="utf-8")
    assert bench.read_events(path) == [{"kind": "ok"}]


def test_timeout_preserves_events_and_other_process(bench, tmp_path):
    events = tmp_path / "events.jsonl"
    code = ("import pathlib,time; "
            f"pathlib.Path({str(events)!r}).write_text('{{\"kind\":\"started\"}}\\n'); "
            "time.sleep(60)")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        start = time.monotonic()
        result = bench.run_bounded([sys.executable, "-c", code],
                                   tmp_path / "probe", start + 1.5,
                                   sample_seconds=0.1)
        assert result["status"] == "timeout"
        assert time.monotonic() - start < 6
        assert bench.read_events(events) == [{"kind": "started"}]
        assert unrelated.poll() is None
        assert (tmp_path / "probe.telemetry.csv").is_file()
    finally:
        for child in psutil.Process(unrelated.pid).children(recursive=True):
            child.kill()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_child_failure_is_reported(bench, tmp_path):
    result = bench.run_bounded([sys.executable, "-c", "raise SystemExit(7)"],
                               tmp_path / "failure", time.monotonic() + 10)
    assert result["status"] == "error"
    assert result["returncode"] == 7


def test_timeout_kills_interpreter_descendant_and_samples_its_memory(bench, tmp_path):
    pidfile = tmp_path / "engine.pid"
    engine = ("import os,time,pathlib; allocation=bytearray(50_000_000); "
              f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)")
    launcher = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{engine!r}]).wait()"
    try:
        result = bench.run_bounded([sys.executable, "-c", launcher], tmp_path / "launcher",
                                   time.monotonic() + 2.5, sample_seconds=0.1)
        assert result["status"] == "timeout"
        assert pidfile.is_file()
        engine_pid = int(pidfile.read_text())
        assert not psutil.pid_exists(engine_pid), "Le moteur doit etre arrete avec son lanceur"
        with (tmp_path / "launcher.telemetry.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert any(int(row.get("pid", 0)) == engine_pid and int(row["rss_bytes"]) > 40_000_000
                   for row in rows)
    finally:
        if pidfile.exists() and psutil.pid_exists(int(pidfile.read_text())):
            process = psutil.Process(int(pidfile.read_text()))
            process.kill()
            process.wait(timeout=5)


def test_expired_budget_does_not_start_child(bench, tmp_path):
    marker = tmp_path / "unexpected"
    code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    result = bench.run_bounded([sys.executable, "-c", code],
                               tmp_path / "expired", time.monotonic() - 1)
    assert result["status"] == "skipped"
    assert not marker.exists()


def test_report_keeps_incomplete_and_cleanup_memory_visible(bench, tmp_path):
    events = [dict(kind="memory", point="after_cleanup", model="new",
                   pass_index=1, private_bytes=100_000_000, rss_bytes=80_000_000,
                   threads=8)]
    bench.write_report(tmp_path, {"models": {"old": "436.onnx", "new": "673.onnx"},
                                  "games": 32, "concurrent": 32}, events,
                       [{"phase": "selfplay", "status": "timeout", "elapsed_s": 1}])
    report = (tmp_path / "rapport.md").read_text(encoding="utf-8")
    assert "incomplete" in report.lower()
    assert "100.0" in report
    assert "timeout" in report
