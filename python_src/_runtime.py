"""Initialisation legere des imports et DLL du moteur local."""
import os
from pathlib import Path
import sys

PYTHON_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PYTHON_ROOT.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
DLL_HANDLES = []
if sys.platform == "win32":
    DLL_HANDLES.append(os.add_dll_directory(str(PYTHON_ROOT)))
