"""Contrats de chargement des sources rangees, sans campagne reelle."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python_src"


def test_binding_depuis_un_dossier_avec_espaces_sans_torch(tmp_path):
    cwd = tmp_path / "dossier avec espaces"
    cwd.mkdir()
    env = dict(os.environ, PYTHONPATH=str(PYTHON))
    code = (
        'import _runtime, chess_engine, json, sys; '
        'board=chess_engine.Chessboard(); '
        'print(json.dumps({"binding": chess_engine.__file__, '
        '"torch": "torch" in sys.modules, '
        '"root": str(_runtime.REPO_ROOT)}))'
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert Path(data["binding"]).parent == PYTHON
    assert data["torch"] is False
    assert Path(data["root"]) == ROOT


@pytest.mark.parametrize("script", [
    "evaluation/build_position_bench.py", "dev_tools/search_bench.py",
    "run_selftrain.py", "train_self_play.py",
])
def test_aide_depuis_un_autre_dossier(script, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-B", str(PYTHON / script), "--help"],
                            cwd=tmp_path, env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_import_lanceurs_sans_execution(tmp_path):
    scripts = [str(PYTHON / name) for name in
               ("run_selftrain.py", "train_self_play.py", "uci.py")]
    code = ('import runpy, sys; '
            f'[runpy.run_path(p, run_name="layout_import_test") for p in {scripts!r}]; '
            'assert "torch" not in sys.modules; print("IMPORT_OK")')
    env = dict(os.environ, PYTHONPATH=str(PYTHON))
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IMPORT_OK"


def test_imports_evaluation_dans_un_worker_spawn(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(PYTHON))
    result = subprocess.run(
        [sys.executable, "-B", str(PYTHON / "tests/spawn_layout_probe.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SPAWN_OK"


def test_pretraining_importable_sans_traitement_ni_reseau(tmp_path):
    modules = ["clean_pgn", "download_pgns", "extract_pgn_from_lichess",
               "convert_pgn_to_binary", "dataset", "sharded_dataset",
               "convert_ckpt", "transfer_weights"]
    code = '''import importlib, socket
def deny(*args, **kwargs):
    raise AssertionError('Connexion reseau inattendue pendant un import')
socket.socket.connect = deny
'''
    code += f'for name in {modules!r}: importlib.import_module("pretraining." + name)\n'
    code += 'print("PRETRAINING_IMPORT_OK")\n'
    env = dict(os.environ, PYTHONPATH=str(PYTHON), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PRETRAINING_IMPORT_OK"
    assert list(tmp_path.iterdir()) == []
