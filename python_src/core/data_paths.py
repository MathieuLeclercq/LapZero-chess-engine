"""Emplacements locaux des modeles et donnees, independants du terminal."""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime

from pathlib import Path

ROOT = _runtime.REPO_ROOT
CHECKPOINTS_DIR = ROOT / "data" / "checkpoints"
ONNX_DIR = CHECKPOINTS_DIR / "onnx"
TOURNAMENTS_DIR = ROOT / "data" / "tournaments"
LOGS_DIR = ROOT / "data" / "logs"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "2026_04_30_09h53_iter436_unsupervised.pt"
