# Benchmarks

This directory contains reproducible performance and equivalence benchmarks for
mmlx-whisper. Each benchmark has its own subdirectory with a runnable entry point,
methodology, supporting code, and recorded results.

## Available benchmarks

| Benchmark | Purpose | Documentation | Recorded results |
|---|---|---|---|
| [Beam selection](beam_selection/) | Compare the original NumPy beam search with MLX candidate selection, using isolated updates and public speech audio. | [Methodology and reproduction](beam_selection/README.md) | [Apple M5, 2026-11-05](beam_selection/RESULTS.md) |

## Directory conventions

Keep each benchmark's scripts, reference implementations, documentation, and
result artifacts together. Document the hardware, software and model versions,
inputs, timing procedure, correctness checks, and limitations alongside results.
Keep downloaded audio and models outside the repository.

Run benchmarks from the repository root using the commands in their specific
guides. Run GPU benchmarks sequentially to avoid resource contention affecting
the measurements.
