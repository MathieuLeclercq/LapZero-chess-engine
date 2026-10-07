# LapZero-chess-engine

An AlphaZero-style chess engine built from scratch in C++, with Python
bindings and Lc0-inspired improvements. Designed for training on a laptop
with a single consumer GPU.

## Features

- C++17 board, move generation and chess rules
- Python bindings with pybind11
- Policy/value residual network
- Squeeze-and-excitation blocks
- First Play Urgency (FPU)
- PUCT Monte Carlo tree search
- Zobrist hashing and transposition table
- Multicore, batched search with virtual loss
- ONNX Runtime inference on GPU or CPU
- GPU-batched self-play
- Fast/slow search budgets and Dirichlet exploration
- PyTorch training and replay buffer
- Supervised pretraining on Lichess games, followed by grandmaster games
- UCI support

## Model weights

Model weights can be found [here](https://drive.google.com/drive/folders/1Vt0W7RFmE_i47zv-UTGBZ2Jvtsyou10H?usp=drive_link).
