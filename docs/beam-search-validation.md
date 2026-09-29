# Production beam-search validation

The production decoder now selects compact candidates in MLX and shares
cross-attention K/V per audio item. Queries use an explicit audio/hypothesis
layout for attention, while only self-attention caches follow parent indices.
Both beam search and sampled `best_of` groups use this layout. Cross-attention
scores keep their existing flattened batch/head/token/frame shape for alignment.

Equal scores use increasing token ID and stable parent traversal. Bit-for-bit
compatibility with NumPy's old tie ordering is intentionally not required.
FP32 cumulative scores, deduplication, EOT handling, patience, final ranking and
suppression/timestamp rules are retained.

## Fresh measurements

Measured on the same Apple M4 / 24 GB, MLX 0.32.2 environment and 6.665-second
fixture as the [investigation](beam-search-investigation.md). Each cell uses a
fresh process, one warm-up, three synchronized repetitions and cleared allocator
cache before each repetition. Model loading is excluded; decoder timing uses
precomputed features, transcription includes audio loading and the encoder.
Models, snapshots, language, FP16, suppression, prompts and fallback settings are
unchanged from the investigation. Temperature is zero, ranked candidates enabled,
and timestamps disabled in this table. Beam 1 is beam decoding, not greedy.

The benchmark freezes the old beam update/reorder routines from `8b6994f` and
reconstructs expanded encoder features in an inference adapter for the baseline.
This keeps the old cache layout measurable after the production change. The
legacy prototype variants are retained to reproduce the investigation.

Values are **baseline → production**. Memory is peak active MLX allocation in
**decimal MB**, including weights, not process RSS or total system memory.

| Model | Beam | Decoder seconds | Transcription seconds | Decoder peak MB | Transcription peak MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 1 | 0.163 → 0.057 | 0.186 → 0.117 | 97.0 → 98.3 | 291.0 → 292.3 |
| tiny | 3 | 0.270 → 0.069 | 0.352 → 0.119 | 158.0 → 106.5 | 350.6 → 299.2 |
| tiny | 5 | 0.444 → 0.085 | 0.496 → 0.133 | 209.6 → 114.5 | 403.7 → 308.6 |
| medium | 1 | 0.602 → 0.515 | 1.106 → 1.009 | 1738.7 → 1738.7 | 2210.7 → 2210.7 |
| medium | 3 | 1.165 → 0.526 | 1.583 → 0.932 | 2323.5 → 1757.8 | 2325.2 → 2210.7 |
| medium | 5 | 1.677 → 0.570 | 2.168 → 1.038 | 2697.5 → 1777.2 | 2699.2 → 2210.7 |

With segment timestamps enabled at beam 5:

| Model | Baseline transcription | Production transcription |
| --- | ---: | ---: |
| tiny | 0.520 s | 0.152 s |
| medium | 2.461 s | 1.099 s |

All 16 cells produced identical repeat outputs, and each production cell matched
its baseline's recorded tokens/text, scores, ranked candidate metadata and
segment metadata. This is evidence for this fixture, not a WER result or a
promise of identical tie outcomes. The source recording contains recognition
errors which these changes do not correct. Longer/multilingual/noisy corpus
measurements remain useful. Small timing differences are noise-sensitive;
variant order was sequential and there were no thermal controls.

## Regression coverage

- Stable tied scores, EOT ties and unique IDs in suppressed tails.
- Existing beam expansion, patience, ranking, finalization and greedy checks.
- Two distinct audio groups, repeated beam parents, incremental decoder logits
  compared with explicitly expanded K/V, and storage-shape/identity checks.
- Grouped cross-attention output and alignment scores versus expanded attention.
- Model-backed distinct speech/silence batches compared with the frozen baseline,
  for beam decoding with patience and for sampled `best_of` decoding.
- Real transcription word timestamps and forced beam-to-sampling fallback.
- Existing CLI, transcription, batched decode and candidate-output integration.

Validation passed: 21 production unit/integration tests using tiny, then all 26
production and experiment tests with integration coverage on medium (the optional
experiment speech/silence comparison used tiny). Compilation and diff whitespace
checks passed as well.

The subsequent review loop found and fixed two validation gaps: 80-bin mel inputs
were hard-coded in two integration tests (and the optional experiment check),
preventing their use with 128-bin large-v3; and the randomized multi-step
differential check covered only the prototype selector. It now exercises the
production update too. Batched comparisons also verify returned audio features
and no-speech probabilities.

After those fixes, all eight large-v3 integration tests passed. The combined
suite passed 27 tests with tiny integration coverage and the optional experiment
comparison on large-v3. A second review pass found no further issues in these
changes. The production implementation and recorded performance measurements
were unchanged by this review.

The shared-cache implementation has no single-audio restriction. The standalone
`shared` experimental variant still does; use `production` for the actual code.

## Reproduction

```sh
uv sync --locked
RUN_MLX_WHISPER_INTEGRATION=1 \
  MLX_WHISPER_TEST_MODEL=mlx-community/whisper-tiny-mlx \
  uv run pytest -q test_beam_search.py test_beam_search_integration.py
uv run python experiments/beam_search.py \
  --model mlx-community/whisper-medium-mlx --beam 5 \
  --variant baseline --output /tmp/beam-baseline.json
uv run python experiments/beam_search.py \
  --model mlx-community/whisper-medium-mlx --beam 5 \
  --variant production --output /tmp/beam-production.json
```

Use the pinned snapshot paths recorded in the JSON for strict reproduction.
Repeat for beams 1/3/5 and tiny/medium; add `--timestamps` for the timestamp cases.
[All timing/memory samples and unique outputs](../experiments/results/production-m4.json)
are retained. The [original investigation](beam-search-investigation.md) explains
the profiling rationale and remaining synchronization/full-GPU-sort costs.
