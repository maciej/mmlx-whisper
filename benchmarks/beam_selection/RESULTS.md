# Apple M5 measurements — 2026-11-05

These measurements were captured before rebasing onto the shared cross-attention
cache changes in `7a868c1`. They compare the recorded source revision with the
original `80e6050` decoder, not the incremental performance over current `main`.
The recorded source hash and timings are historical and have not been replaced.

Apple M5, 24 GiB unified memory; macOS 27.0; Python 3.12.14. mlx 0.32.0, numpy 2.4.6, mlx-whisper 0.4.3.
FP16 models: tiny `78c52ab98ca87f570bc57ad852e15ef7060f9f76`; large-v3-turbo `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`.

Baseline commit: `80e60509090601a58acf1e6d0d4c2e34b8e5a5de`. Optimized source SHA-256: `43c078a88332d2e522d02ff04cb45be2f990eec0df6f2ebe03d23e53b8836fd6`.

Five timed warm runs per audio case, 30 per isolated selection case. Tables show medians. See [methodology](README.md) and [individual timings, memory, input hashes and output hashes](results_m5.json). The JSON retains equality checks and hashes instead of repeating the full candidate transcripts; the benchmark command emits full outputs.

## Isolated selection

No model/cache computation; 51,865 vocabulary entries, 64-token prefixes.

| Audio batch | Beam | Baseline ms | MLX ms | Speedup | Baseline / MLX peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.666 | 1.067 | 2.50× | 0.41 / 1.63 |
| 1 | 3 | 7.791 | 0.642 | 12.13× | 1.19 / 4.75 |
| 1 | 5 | 12.834 | 0.723 | 17.76× | 2.00 / 8.01 |
| 1 | 10 | 25.338 | 1.064 | 23.81× | 3.97 / 15.96 |
| 2 | 1 | 5.086 | 0.610 | 8.34× | 0.81 / 3.25 |
| 2 | 3 | 14.871 | 0.772 | 19.27× | 2.38 / 9.51 |
| 2 | 5 | 24.449 | 1.046 | 23.37× | 3.97 / 15.90 |
| 2 | 10 | 49.142 | 1.528 | 32.16× | 7.95 / 31.96 |

## Warm audio processing

Short: 6.155 s clean speech. Long: 44.43 s concatenation of the clean sample and an 8.655 s other sample, repeated three times. Batch: those two distinct samples decoded together. Timestamp: short sample with timestamps enabled. These are a small public test set, not a broad quality benchmark.

| Model | Beam | Case | Baseline s | MLX s | Speedup | Baseline / MLX peak MiB |
|---|---:|---|---:|---:|---:|---:|
| tiny | 1 | short | 0.0957 | 0.0432 | 2.21× | 281.0 / 282.2 |
| tiny | 1 | long | 0.2037 | 0.0895 | 2.28× | 304.4 / 305.7 |
| tiny | 1 | batch | 0.2106 | 0.0834 | 2.53× | 438.7 / 441.1 |
| tiny | 1 | timestamps | 0.0977 | 0.0484 | 2.02× | 281.4 / 282.6 |
| tiny | 3 | short | 0.2622 | 0.0786 | 3.33× | 338.2 / 341.7 |
| tiny | 3 | long | 0.4910 | 0.1497 | 3.28× | 361.6 / 365.2 |
| tiny | 3 | batch | 0.5263 | 0.1276 | 4.12× | 507.8 / 514.9 |
| tiny | 3 | timestamps | 0.2593 | 0.0887 | 2.92× | 339.2 / 342.8 |
| tiny | 5 | short | 0.3940 | 0.0944 | 4.17× | 386.2 / 392.2 |
| tiny | 5 | long | 1.0907 | 0.2638 | 4.14× | 409.7 / 415.7 |
| tiny | 5 | batch | 0.8733 | 0.1797 | 4.86× | 572.8 / 584.7 |
| tiny | 5 | timestamps | 0.3953 | 0.1107 | 3.57× | 388.2 / 394.2 |
| tiny | 10 | short | 0.7701 | 0.1534 | 5.02× | 507.6 / 519.6 |
| tiny | 10 | long | 2.1090 | 0.4121 | 5.12× | 531.0 / 543.0 |
| tiny | 10 | batch | 1.6883 | 0.3086 | 5.47× | 633.8 / 633.8 |
| tiny | 10 | timestamps | 0.7266 | 0.1666 | 4.36× | 511.6 / 523.6 |
| large-v3-turbo | 1 | short | 0.4478 | 0.3845 | 1.16× | 2092.1 / 2092.1 |
| large-v3-turbo | 1 | long | 1.5582 | 1.1676 | 1.33× | 2097.7 / 2097.7 |
| large-v3-turbo | 1 | batch | 0.8470 | 0.7284 | 1.16× | 2538.9 / 2538.9 |
| large-v3-turbo | 1 | timestamps | 0.4518 | 0.3957 | 1.14× | 2092.3 / 2092.3 |
| large-v3-turbo | 3 | short | 0.6537 | 0.4654 | 1.40× | 2092.1 / 2092.1 |
| large-v3-turbo | 3 | long | 1.8086 | 1.1907 | 1.52× | 2097.7 / 2097.7 |
| large-v3-turbo | 3 | batch | 1.2151 | 0.8494 | 1.43× | 2538.9 / 2538.9 |
| large-v3-turbo | 3 | timestamps | 0.6495 | 0.4869 | 1.33× | 2092.7 / 2092.7 |
| large-v3-turbo | 5 | short | 0.8108 | 0.5231 | 1.55× | 2092.1 / 2092.1 |
| large-v3-turbo | 5 | long | 2.3705 | 1.3483 | 1.76× | 2097.7 / 2097.7 |
| large-v3-turbo | 5 | batch | 1.6260 | 1.0083 | 1.61× | 2538.9 / 2538.9 |
| large-v3-turbo | 5 | timestamps | 0.8113 | 0.5465 | 1.48× | 2093.1 / 2093.1 |
| large-v3-turbo | 10 | short | 1.2515 | 0.6718 | 1.86× | 2326.3 / 2326.3 |
| large-v3-turbo | 10 | long | 4.5105 | 2.4705 | 1.83× | 2351.4 / 2351.4 |
| large-v3-turbo | 10 | batch | 2.8744 | 1.4708 | 1.95× | 2943.4 / 2943.4 |
| large-v3-turbo | 10 | timestamps | 1.3529 | 0.7851 | 1.72× | 2327.6 / 2327.6 |

All 32 configurations produced exactly equal transcripts, tokens, timestamps and ranked candidate records, including scores. Exact ties can differ on other inputs under the documented deterministic tie policy.

## Validation and review

Unchanged main: 17 passed, 3 failed (`test.py` plus existing beam unit tests). Final tests: 30 passed, the same 3 failed, including opt-in real-audio integration. The known failures are `test_decode_greedy` (no-speech expectation), `test_decode_lang` (language probability expectation), and `test_torch_mlx` (cross-backend numerics). No expectations were weakened.

Deterministic differential tests cover beams 1/3/5/10, two audio groups, eight updates, duplicate prefixes, competing parent scores, finished archives and finalization. Additional tests cover exact ties, negative-infinity suppression, EOS below the active cutoff, repeated/reversed cache ancestry, dtype preservation, reset, and seeded sampling. Real-audio integration compares full candidates with timestamps, patience 1.5 and length penalty 0.6. Existing CLI and greedy checks also pass.

Review fixed Metal’s unsupported boolean argsort keys by using integer keys, retained genuine negative-infinity candidates with a separate validity mask, and preserved token dtype when appending unsigned sort indices. The frozen baseline class was verified against its original commit (ignoring trailing blank lines).

## Limits and remaining work

Peak MLX allocations include live arrays/model memory, not CPU allocations, process RSS or allocator cache totals. GPU sorting and duplicate masks add device temporary storage; this is not a claim of lower memory use. Host-transfer counts are derived from the implementation, not measured bus traffic: without EOS, per-audio reads change from approximately `4*B*V + 4*B*prefix_length + 4*B` bytes to `B*(B+1)` mask bytes plus one parent index per beam. EOS steps additionally read only finished prefixes and scores. Explicit synchronization remains for completion/archive bookkeeping; it is included in timings.

All benchmarks ran sequentially. Fixed baseline-first order and warm caches can introduce bias; no confidence intervals or thermal controls were used. The longer input repeats two samples, fallback/silence skipping is disabled, and this does not establish accuracy improvements or performance on other hardware, large batches or very wide beams.

Reproduce the measured audio repetition count with:

```sh
python -m benchmarks.beam_selection --repeats 5 --output /tmp/beam-results.json
```
