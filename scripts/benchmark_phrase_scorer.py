#!/usr/bin/env python3
"""Synthetic MLX scorer benchmark and numerical check; no speech-quality claims."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np
from mlx_whisper.boosting import PhraseBoostingConfig, PhraseTree, PhraseScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    results = []
    for count in (10, 50, 200, 1000, 20000):
        sequences = [
            tuple(map(int, row)) for row in rng.integers(0, 50000, size=(count, 3))
        ]
        started = time.perf_counter()
        tree = PhraseTree(sequences, 51865, PhraseBoostingConfig())
        compile_ms = (time.perf_counter() - started) * 1000
        for batch in (1, 5):
            runtime = PhraseScorer(tree, 50257, range(50258, 51865))
            # Mixed root, partial, and completed states.
            states = list(range(batch))
            device_states = mx.array(states, mx.int32)
            expected, next_states = tree.reference(states, 50257, range(50258, 51865))
            actual, actual_states = runtime.advance(device_states)
            mx.eval(actual, actual_states)
            max_error = float(np.max(np.abs(np.array(actual) - np.array(expected))))
            mismatches = int(np.count_nonzero(np.array(actual_states) != next_states))
            if max_error > 2e-6 or mismatches:
                raise AssertionError((count, batch, max_error, mismatches))
            times = []
            mx.reset_peak_memory()
            for _ in range(args.iterations):
                started = time.perf_counter()
                mx.eval(runtime.advance(device_states))
                times.append((time.perf_counter() - started) * 1000)
            results.append(
                {
                    "phrases": count,
                    "states": len(tree.edges),
                    "batch": batch,
                    "compile_ms": compile_ms,
                    "median_query_ms": statistics.median(times),
                    "max_score_error": max_error,
                    "state_mismatches": mismatches,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                }
            )
    output = {
        "device": mx.metal.device_info(),
        "mlx": mx.__version__,
        "vocab_size": 51865,
        "dtype": "float32 scores/int32 states",
        "iterations": args.iterations,
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
