"""Synchronized selection-only benchmark on fixed random vocabulary rows."""
import json
import statistics
import time

import mlx.core as mx
import numpy as np
from beam_search import select_candidates

mx.random.seed(42)
results = []
for beam in (1, 3, 5):
    x = mx.random.normal((beam, 51865))
    mx.eval(x)
    k = beam + 1
    def cpu_sort():
        rows = np.array(x)
        ids = np.argsort(rows, axis=-1)[:, -k:][:, ::-1]
        return ids, np.take_along_axis(rows, ids, axis=-1)
    def cpu_partition():
        rows = np.array(x)
        ids = np.argpartition(-rows, k - 1, axis=-1)[:, :k]
        vals = np.take_along_axis(rows, ids, axis=-1)
        order = np.argsort(-vals, axis=-1)
        return np.take_along_axis(ids, order, axis=-1), np.take_along_axis(vals, order, axis=-1)
    for name, fn in [("numpy_sort", cpu_sort), ("numpy_partition", cpu_partition)] + [
        (f"mlx_{method}", lambda method=method: select_candidates(x, k, method))
        for method in ("sort", "partition", "reduce")
    ]:
        samples = []
        for repeat in range(55):
            mx.synchronize()
            start = time.perf_counter()
            ids, vals = fn()
            mx.eval(ids, vals)
            np.array(ids), np.array(vals)
            elapsed = time.perf_counter() - start
            if repeat >= 5:
                samples.append(elapsed)
        results.append({"beam": beam, "method": name, "median_ms": 1000*statistics.median(samples), "seconds": samples})
print(json.dumps({"mlx": mx.__version__, "numpy": np.__version__, "results": results}, indent=2))
