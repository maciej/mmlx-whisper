# Beam-search time and memory investigation

Measured on 2026-09-29 against `8b6994f319cc755ece256cd01f618159824302fa`, for
[issue #1](https://github.com/maciej/mmlx-whisper/issues/1).

The investigation below records the original experiments. Compact selection and
grouped per-audio cross-attention cache sharing have since been implemented in
the production decoder. See [the production validation](beam-search-validation.md)
and [current behavior](beam-search.md) for the implemented path; the old `shared`
benchmark variant remains a deliberately limited single-audio prototype.

**Recommendation: optimize candidate selection and cross-attention cache layout
as separate changes.** Compact selection is a worthwhile first step, particularly
for small models. Shared cross-attention caches are the stronger opportunity for
both time and memory at larger beam sizes. Moving all hypothesis bookkeeping to
MLX is not justified by these measurements.

These measurements used investigation prototypes. The shared-cache
prototype intentionally supports one audio item only. It is not ready for general
batched decoding, sampling fallback, or word alignment.

## Method and limits

- Apple M4 Mac mini, 10 CPU cores, 24 GB unified memory; macOS 27.0 (26A428).
- MLX 0.32.2, NumPy 2.5.3, Python 3.12.13, FP16 model execution with FP32 scores.
- Cached `mlx-community/whisper-tiny-mlx` snapshot
  `6caf9c55601caafbe6508a8b0d216bdf4783c4e8` and
  `mlx-community/whisper-medium-mlx` snapshot
  `7fc08c4eac4c316526498f147dfdee6f6303f975`.
- Existing `mlx_whisper/assets/ls_test.flac`, 6.665 seconds. `sag` could not generate
  requested additional voice samples because its configured key file,
  `/Users/maciej/.config/sag/elevenlabs.key`, was missing. No alternative TTS was used.
- Single audio item, English, transcription, temperature 0 only (no sampling
  fallback), no prompt/prefix, default suppression/patience/length penalty,
  ranked candidates enabled. Main matrix disables timestamps; additional beam-5
  runs enable segment timestamps. No word timestamps.
- Each model/beam/variant cell used a fresh process, one warm-up per measured
  operation, then three repetitions; tables show medians. Timers synchronize MLX
  before and after work. Allocator cache is cleared and peak memory reset before
  each measured repetition. Model loading/downloads are excluded.
- **Decoder** timings use precomputed encoder features. **Transcribe** timings
  include file decoding, mel calculation, encoder, decoder and result processing,
  with the model already loaded. Beam 1 still uses `BeamSearchDecoder`; it is not
  the greedy path.
- Memory is **peak active MLX allocation**, including resident model weights and
  retained benchmark inputs; decimal MB/GB. It is not total system RAM or RSS.
  Raw records also contain allocator cache bytes and process-lifetime peak RSS;
  those are separate metrics, not additive on unified memory.
- Sequential variant order, no thermal controls or confidence intervals. Small
  differences are not reliable. This short fixture is not a quality corpus or a
  long-recording performance benchmark. Encoder warm-up, cached file reads and
  `ffmpeg` startup affect these particular timings.

## Variants and measurements

| Variant | Change from baseline |
| --- | --- |
| baseline | Production NumPy full-vocabulary sort and full KV-cache reorder |
| compact | Stable MLX sort; transfer only beam+1 IDs and scores per parent; retain Python bookkeeping |
| cache | Compact selection plus reorder only self-attention KV; preserve cross-attention KV |
| shared | Cache variant plus project cross-attention K/V once and broadcast over beams; single-audio prototype |

Each timing cell is **decoder seconds / transcription seconds**.

| Model | Beam | Baseline | Compact | Cache | Shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 1 | 0.130 / 0.192 | 0.045 / 0.095 | 0.055 / 0.104 | 0.059 / 0.108 |
| tiny | 3 | 0.269 / 0.334 | 0.089 / 0.142 | 0.099 / 0.139 | 0.073 / 0.111 |
| tiny | 5 | 0.405 / 0.497 | 0.120 / 0.166 | 0.106 / 0.141 | 0.066 / 0.124 |
| medium | 1 | 0.549 / 1.062 | 0.488 / 0.970 | 0.494 / 0.986 | 0.484 / 0.971 |
| medium | 3 | 1.135 / 1.598 | 1.120 / 1.416 | 0.838 / 1.288 | 0.524 / 0.989 |
| medium | 5 | 1.652 / 2.204 | 1.429 / 1.869 | 1.252 / 1.727 | 0.598 / 1.087 |

At beam 5, compact selection alone cuts transcription time by **67% on tiny** and
**15% on medium**. The shared variant cuts it by **75% and 51%**, respectively.
The no-op cache change at beam 1 illustrates normal timing variability.

Peak active MLX allocation at beam 5, **decoder MB / transcription MB**:

| Model | Baseline | Compact | Cache | Shared |
| --- | ---: | ---: | ---: | ---: |
| tiny | 209.6 / 403.7 | 215.9 / 410.0 | 215.4 / 410.0 | 114.5 / 308.6 |
| medium | 2697.5 / 2699.2 | 2697.5 / 2699.2 | 2640.5 / 2655.6 | 1777.2 / 2210.7 |

Compact selection does **not** substantially reduce peak memory: the sort adds
GPU temporary buffers while eliminating a much smaller NumPy copy. Shared caches
cut medium decoder peak by **34%** and transcription peak by **18%**. The encoder
limits the latter. Do not equate a smaller host transfer with a smaller process.

With segment timestamps enabled, beam-5 transcription times were:

| Model | Baseline | Compact | Shared |
| --- | ---: | ---: | ---: |
| tiny | 0.544 s | 0.200 s | 0.148 s |
| medium | 2.353 s | 2.016 s | 1.129 s |

## What the profile actually says

A separate medium/beam-5 decoder profile attributed **0.362 s of 1.766 s (20.5%)**
to 140 NumPy sorts across 28 update calls. NumPy conversion itself took only about
3 ms. The rest of the large `update()` time is largely synchronization waiting
for lazy MLX execution, not proof that Python hypothesis bookkeeping is expensive.
A host profile cannot separate GPU kernels inside that wait.

For a vocabulary of 51,865 at beam 5, baseline copies 1,037,300 bytes of FP32
vocabulary scores per step. Compact selection returns 30 pairs of uint32 IDs and
FP32 scores: 240 bytes, approximately 4,322 times less candidate payload. Token
histories and cumulative scores still cross to Python, and each update still
synchronizes. There is no claim of zero-copy or asynchronous beam search.

A synchronized selection-only benchmark on fixed random vocabulary rows measured:

| Method | Beam 1 | Beam 3 | Beam 5 |
| --- | ---: | ---: | ---: |
| NumPy full sort | 2.759 ms | 8.290 ms | 13.985 ms |
| NumPy partial selection | 0.104 ms | 0.678 ms | 1.765 ms |
| MLX stable sort | 0.448 ms | 0.611 ms | 0.825 ms |
| MLX argpartition + selected sort | 0.350 ms | 0.559 ms | 0.837 ms |
| MLX repeated argmax, exploratory | 0.256 ms | 0.397 ms | 0.544 ms |

This benchmark excludes neural-network computation and uses random rather than
actual suppressed model logits. It must not be substituted for the end-to-end
measurements. NumPy partial selection is an important cheap comparison: much of
the problem is full sorting, not merely running on the CPU.

In MLX **v0.32.2**, both GPU `ArgPartition` and `Partition` call `gpu_merge_sort`.
Choosing `argpartition` does not currently avoid a full GPU sort. See the
[pinned implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/sort.cpp)
and [API contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argpartition.html).
A fused small-k selector might improve the remaining selection time, but is a
later optimization. The naive repeated-argmax experiment is incorrect when only
suppressed values remain: it can pick the same index repeatedly.

## Why cache sharing matters

`DecodingTask.run()` expands encoder features to the beam batch. Every decoder
layer then projects and retains identical cross-attention K/V for every beam.
`Inference.rearrange_kv_cache()` gathers those immutable arrays whenever it
reorders the changing self-attention cache.

For FP16 medium, cross-attention cache storage alone is
`2 * 24 layers * 1500 audio positions * 1024 channels * 2 bytes`, or **147.456 MB
per audio item**. At beam 5, storing it per beam uses **737.280 MB**. Sharing saves
**589.824 MB of logical persistent cache storage**, before considering avoided
projection work, gathers and temporary allocations. Logical tensor sizes and
measured allocator peaks differ; this formula is not a predicted peak reduction.

A production design should represent self-attention cache per hypothesis and
cross-attention cache per audio item. Preserve an explicit audio-to-beam grouping;
reshape queries to a grouped batch for broadcasted cross-attention instead of
flattening and physically repeating K/V. Only self-attention state follows parent
indices. Test distinct audio items, repeated/reordered parents, timestamp paths,
sampling and alignment. Sharing audio zero across a flattened batch would be a
serious correctness bug. The prototype deliberately rejects multiple audio items.

## Semantics and validation

Across all **30 benchmark cells**, every repetition produced identical outputs.
Against the matching baseline, all variants preserved the selected tokens/text,
all ranked candidate metadata and scores, and every recorded transcription
segment including timestamps and probabilities. This is exact equality on this
fixture, not a dataset-level WER claim. Existing recognition errors remain.

Ties need an explicit contract before production adoption. Current NumPy
`argsort` uses its default unstable sort, then reverses the result. MLX stable
sorting of negated scores orders tied tokens by increasing token ID. Even four
equal values demonstrate a difference. Do not claim bit-for-bit compatibility on
all ties, and do not add an epsilon to scores (that changes near-tie probabilities).
Define token-ID and parent tie ordering, retain genuine FP32 scores, and test ties
at the cutoff and involving EOT. Deduplication, patience and final length ranking
must remain unchanged. A flattened global top-k can discard necessary unique
candidates when multiple parent prefixes are identical.

Validation includes the existing nine unit tests, randomized multi-step/two-audio
baseline equivalence, suppressed/tied selection, cache identity/parent ordering,
and rejection of unsupported shared-cache batches. A separate model-backed test
compares speech plus generated silence in the same batch. The existing Python
integration checks were also exercised with baseline, compact and cache variants;
CLI subprocesses always use production code, not the process-local experiment.

The shared prototype has not been made compatible with general batches, sampled
fallback or word alignment. Long audio, multiple languages, silence/noise-heavy
recordings, long prompts, beam/patience extremes, quantization and other MLX
versions remain unmeasured. These are implementation/validation work, not grounds
to mark the issue completed.

## Suggested implementation order and issue changes

1. Land compact candidate selection with a documented tie contract and differential
   tests. Retain Python hypothesis bookkeeping initially. Report transfer reduction,
   synchronized wall time and memory separately.
2. Split self/cross cache reordering. Then introduce per-audio cross K/V with an
   explicit grouped layout. Treat full sharing as a separate substantial change;
   its observed gains justify prioritizing it over further bookkeeping work.
3. Expand benchmarks: `sag` speech once credentials are restored, natural speech
   with references, multiple languages, long audio, noisy/silent samples, unequal
   audio batches, prompts, fallback, word timestamps and large-v3. Record warm and
   allocator-cold regimes and randomize variant order for tighter comparisons.
4. Only then consider fused top-k, token-history bookkeeping, static/preallocated
   self-KV, and attention kernels that avoid materializing attention matrices.
   Ordinary decoding can potentially use fused attention; alignment still needs
   attention information. Those are new experiments, not measured wins here.

Two additional details deserve separate handling. The initial beam group consists
of identical prefixes, so sharing the initial decoder pass could avoid duplicate
work. Also, `_main_loop()` computes the next `_step()` before checking the previous
`completed` flag. For the Python beam decoder that extra call mutates finished
hypotheses as well as doing work. Stopping earlier should be investigated as a
correctness change, not silently folded into a performance patch.

The issue should add a cache-memory objective, precise allocation/RSS terminology,
an explicit tie contract, and baseline-vs-optimized quality comparisons. Its
current functional integration smoke tests alone do not prove semantic parity.
Keep the existing requested beam sizes and model coverage; do not require all
bookkeeping to live on the GPU in the absence of evidence.

## Reproduction and artifacts

Run from the repository root, using the same model snapshots for strict comparisons:

```sh
uv sync --locked
uv run python experiments/beam_search.py \
  --model mlx-community/whisper-medium-mlx --beam 5 \
  --variant baseline --warmups 1 --repeats 3 --profile \
  --output /tmp/medium-5-baseline.json
# Repeat with --variant compact, cache, shared; --beam 1, 3, 5;
# whisper-tiny-mlx; and --timestamps for the additional beam-5 cases.
uv run python experiments/beam_selection.py > /tmp/beam-selection.json
uv run pytest -q test_beam_search.py experiments/test_beam_search_experiments.py
EXPERIMENT_MODEL=/path/to/local/whisper-tiny-mlx/snapshot \
  uv run pytest -q experiments/test_beam_search_experiments.py
RUN_MLX_WHISPER_INTEGRATION=1 \
  MLX_WHISPER_TEST_MODEL=mlx-community/whisper-tiny-mlx \
  uv run pytest -q test_beam_search_integration.py
```

- [Prototype benchmark](../experiments/beam_search.py)
- [Selection microbenchmark](../experiments/beam_selection.py)
- [Prototype regression tests](../experiments/test_beam_search_experiments.py)
- [All timing/memory samples and unique outputs](../experiments/results/beam-m4.json)
- [Selection samples](../experiments/results/selection-m4.json)
- [Medium beam-5 CPU profiles](../experiments/results/profile-medium-5.txt)

The JSON combines per-cell outputs, retaining all time/memory samples and one copy
of each unique result. It records whether repeat outputs were identical.
