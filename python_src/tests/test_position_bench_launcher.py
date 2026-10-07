"""Verification du superviseur Windows, sans aucun calcul Stockfish."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "dev_tools/run_position_bench_finalize.ps1"


@pytest.mark.skipif(sys.platform != "win32", reason="Superviseur Windows")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_superviseur_conserve_la_commande_les_journaux_et_le_code_retour(
        tmp_path, exit_code):
    root = tmp_path / "root with spaces"
    source = root / "python_src/evaluation"
    work = root / "data/position_bench_work"
    source.mkdir(parents=True)
    work.mkdir(parents=True)
    annotation = work / "annotated.jsonl.zst"
    annotation.write_bytes(b"reserve intacte")
    # Le faux constructeur ne lance ni moteur ni recherche et consigne ses arguments.
    (source / "build_position_bench.py").write_text(
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]), flush=True)\n"
        "print('audit factice', file=sys.stderr, flush=True)\n"
        f"sys.exit({exit_code})\n", encoding="utf-8")
    job = root / "out/job with spaces"
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"

    result = subprocess.run(
        [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(RUNNER), "-Root", str(root), "-Python", sys.executable,
         "-Stockfish", sys.executable, "-JobDirectory", str(job), "-Workers", "8"],
        capture_output=True, timeout=30)

    assert result.returncode == exit_code, (result.stdout, result.stderr)
    status = json.loads((job / "status.json").read_text(encoding="utf-8-sig"))
    assert status["state"] == ("completed" if exit_code == 0 else "failed")
    assert status["exit_code"] == exit_code
    assert status["python_pid"] > 0
    assert status["finished_utc"] is not None
    arguments = json.loads((job / "stdout.log").read_text(encoding="utf-8-sig"))
    assert arguments == [
        "finalize", "--stage", "final", "--stockfish", sys.executable,
        "--workers", "8", "--work-dir", str(work), "--output",
        str(root / "data/position_bench/v1"), "--resume"]
    assert "audit factice" in (job / "stderr.log").read_text(encoding="utf-8-sig")
    assert annotation.read_bytes() == b"reserve intacte"
