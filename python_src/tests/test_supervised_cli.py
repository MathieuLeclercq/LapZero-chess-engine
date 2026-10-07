"""Configuration portable de l'entree d'entrainement supervise."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python_src"))

from training import train_supervised


def test_default_dataset_is_repo_relative_and_training_starts_fresh(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = train_supervised.parse_args([])
    assert args.pgn_dir == ROOT / "training_data" / "clean_pgns"
    assert args.resume is None


def test_custom_dataset_and_resume_paths():
    args = train_supervised.parse_args([
        "--pgn-dir", "my games", "--resume", "my model.ckpt",
    ])
    assert args.pgn_dir == Path("my games")
    assert args.resume == Path("my model.ckpt")
