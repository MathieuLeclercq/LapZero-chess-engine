# LapZero-chess-engine

A chess engine with a C++17 board and Monte Carlo tree search, a PyTorch
policy/value network, and ONNX Runtime inference. This repository contains
the engine, training and evaluation code, and tests. It does not contain
trained weights, training corpora, research reports, or experiment results.

## Components

- `src/`: board, legal moves, rules, PGN parsing, MCTS, batched inference,
  multicore search, self-play and Python bindings.
- `python_src/core/`: neural network, move encoding, data paths and UCI adapter.
- `python_src/training/`: supervised training and the self-play loop.
- `python_src/pretraining/`: PGN acquisition, filtering and dataset conversion.
- `python_src/evaluation/`: puzzle benchmarks, Stockfish-annotated position
  benchmarks, model matches and ratings.
- `python_src/monitoring/` and `python_src/dev_tools/`: monitoring and diagnostics.
- `tests/` and `python_src/tests/`: C++ and Python tests, with small fixtures.

The network uses 119 input planes, including eight board states, and outputs
a policy over 4672 move indices and one scalar value in [-1, 1]. The default
architecture has 10 residual blocks and 128 filters with squeeze-and-excitation.

## Build on Windows

Requirements: Python 3.13, uv, Visual Studio with the C++ workload, and CMake.
The currently tested configuration uses Visual Studio 2026. GPU inference
also requires the CUDA/cuDNN libraries expected by ONNX Runtime 1.24.3.
PyTorch is locked to a CUDA 13.0 build; a compatible NVIDIA driver is needed
for GPU training. These are separate runtime requirements.

From the repository root, in PowerShell:

```powershell
uv sync --locked
cmake -S . -B build -DPython3_EXECUTABLE="$PWD/.venv/Scripts/python.exe"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CMake downloads ONNX Runtime and pybind11. It builds the `chess_engine`
extension, copies its runtime DLLs into `python_src/`, and generates type stubs.
The configuration currently targets Windows x64; Linux and macOS are not
supported by the supplied CMake dependency setup.

## Tests without model weights

```powershell
.\.venv\Scripts\python.exe -B -m pytest python_src/tests
.\build\Release\chess_perft.exe bench --strict --check-fen
```

The C++ PGN smoke test uses a synthetic 20-ply game. The Python suite uses
synthetic inputs and skips some integration checks when external model
weights or benchmark data are missing. Windows performance-counter checks
depend on the counters available on the machine.

## Training

Training needs data that is not included here. Downloaders and converters in
`python_src/pretraining/` are tools from the original training pipeline;
review their input paths, filtering thresholds and output paths before use.
Several older tools resolve paths relative to the current directory and are
intended to run from `python_src/`.

For supervised training, supply a directory of prepared PGN files:

```powershell
.\.venv\Scripts\python.exe python_src/training/train_supervised.py --pgn-dir training_data/clean_pgns
```

This starts a new model. Add `--resume path/to/model.ckpt` to resume a
Lightning checkpoint. Set `WANDB_MODE=offline` or `WANDB_MODE=disabled` in
your environment if you do not want online W&B logging.

Self-play accepts a compatible model checkpoint, replay data and configuration:

```powershell
.\.venv\Scripts\python.exe python_src/run_selftrain.py --help
```

Use `--checkpoint` to choose your weights. The default historical checkpoint
is not distributed. The pipeline also uses puzzle-start data; prepare the
required files before launching training. `--eval-every 0` disables periodic
position-benchmark evaluation when that benchmark has not been built.
To retain the original replay-buffer working directory, run self-play from
`python_src/` using `../.venv/Scripts/python.exe run_selftrain.py ...`.

## Evaluation datasets

The benchmark builders are included, but not their full datasets or cached
Stockfish annotations. Small PGN fixtures under `tests/data/` are only tests,
not substitutes for the evaluation datasets.

```powershell
.\.venv\Scripts\python.exe python_src/evaluation/build_position_bench.py --help
.\.venv\Scripts\python.exe python_src/evaluation/build_puzzle_dataset.py --help
.\.venv\Scripts\python.exe python_src/evaluation/puzzle_bench.py --help
```

The position builder downloads broadcast games and annotates positions with
an external Stockfish executable supplied through `--stockfish`. Supply the
desired archive months with `--months`; defaults reflect the original
experiment, not a rolling date range. Building a full benchmark can take
hours and is not part of installation or testing.

The puzzle builder needs a Lichess puzzle CSV and fetches the corresponding
games. Supply explicit `--csv`, `--cache`, `--out-train` and `--out-bench`
paths when running from the repository root. An optional Lichess token is
read from a local file; never commit that file.

## Playing through UCI

Model weights are distributed separately in this
[shared Google Drive folder](https://drive.google.com/drive/folders/1Vt0W7RFmE_i47zv-UTGBZ2Jvtsyou10H?usp=drive_link).
Download an ONNX model for playing, or a compatible PyTorch checkpoint for
training and evaluation. Keep `.pt` and `.ckpt` files under `data/checkpoints/`
and `.onnx` files under `data/checkpoints/onnx/`. These folders are ignored
by Git, so downloading weights does not add them to the repository.

The entry point is `python_src/uci.py`. A compatible ONNX model is required
before the adapter can start. The default model filename is historical and
is not supplied. Set `MODEL_PATH` in `python_src/core/uci.py` to your local
model, then configure your chess GUI to run:

```powershell
.\.venv\Scripts\python.exe python_src/uci.py
```

The adapter currently advertises a `WeightsFile` option but does not implement
dynamic model selection through that option. Configure `MODEL_PATH` directly.
UCI per-move logs are written locally under `data/logs/`.

For Lichess integration, use the separate upstream
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) project and
configure it to launch this UCI adapter. Its vendored source, local bot
configuration and tokens are not included here. The legacy Pygame interface
is also not part of this distribution.

## Scope and dependencies

Some source comments refer to internal experiment specifications or reports.
Those documents are not distributed and are not needed by the engine or its
unit tests. This is a selected source distribution, not the full research
workspace or its historical Git commits.

ONNX Runtime, pybind11 and the Python packages are third-party dependencies
with their own licenses. No project-wide license has been selected in this
initial source distribution.
