# Phrase-boosting validation — 2026-11-03

These measurements were captured before rebasing onto the MLX beam-selection
and shared-cache changes merged in `f3d570f`. The recorded timings, output
comparisons, and source hashes describe that earlier implementation; they do not
measure the rebased decoder. The rebase checks are recorded separately below.

The MLX implementation passed numerical and decoding integration checks. Public
speech results are **mixed**: the default strength is not an accuracy-promoted
preset. Boosting remains opt-in. No settings were tuned on these test samples.

## Environment and reproducibility

- Apple M5, 24 GiB unified memory; macOS 27.0.
- Python 3.12.14, MLX 0.32.0; FP16 model inference, float32 phrase scores.
- Tiny snapshot: `78c52ab98ca87f570bc57ad852e15ef7060f9f76`.
- Large-v3-turbo snapshot: `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`.
- Fixed configuration: context score 1, depth scale 1, unknown score 0,
  strength 0.1, final EOS score 1.
- Baseline source used for the disabled-path comparison:
  `80e60509090601a58acf1e6d0d4c2e34b8e5a5de`.

The new code preserves the package's existing Python 3.8+ declaration and was
syntax-checked using Python's 3.8 grammar. Runtime validation was on Python 3.12;
this does not establish every historical Python/MLX combination. Required scatter
operations also exist in MLX 0.11, the existing declared dependency minimum.

See the [evaluation commands and scoring definitions](phrase-boosting.md).
Machine-readable summaries below contain sample IDs, exact model/configuration
identities, metric values, and timing/memory definitions. Downloaded corpora and
raw audio are not part of the repository.

## Public speech evaluation

Source: [LibriSpeech/OpenSLR 12](https://www.openslr.org/12), CC BY 4.0, by
Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur.

- `test-clean`: 40 utterances / 40 speakers, 373.88 seconds, 959 normalized words,
  103 target-word occurrences. Archive MD5 `32fa31d27d2e1cad72775fee3f4849a9` verified.
- `test-other`: 40 utterances / 33 speakers, 240.08 seconds, 655 normalized words,
  86 target-word occurrences. Archive MD5 `fb5a50374b501bb3bac4815ee91d3135` verified.
- 720 measured decodes: two models on clean speech and turbo on other speech,
  each with greedy/beam-5 and baseline/relevant/distractor lists.
- Relevant lists contain up to three longest reference words (oracle context),
  with lowercase/title-case forms. Distractor lists contain up to five unrelated
  words, excluding any present in the reference. These are small-list diagnostics.

Each cell shows **baseline → relevant boosting**. WER and B-WER are percentages;
latencies are medians across utterances, not paired overhead measurements.

| Model / split | Beams | WER % | B-WER % | Target recall % | Median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny / clean | 1 | 6.78 → 6.78 | 12.62 → 12.62 | 87.38 → 87.38 | 35.2 → 49.6 |
| Tiny / clean | 5 | 6.36 → 6.47 | 9.71 → 10.68 | 90.29 → 89.32 | 373.2 → 384.5 |
| Turbo / clean | 1 | 1.67 → 1.67 | 3.88 → 3.88 | 96.12 → 96.12 | 399.8 → 422.6 |
| Turbo / clean | 5 | 1.67 → 1.77 | 3.88 → 3.88 | 96.12 → 96.12 | 871.6 → 860.7 |
| Turbo / other | 1 | 3.82 → 3.66 | 9.30 → 8.14 | 90.70 → 91.86 | 391.8 → 411.9 |
| Turbo / other | 5 | 3.66 → 3.97 | 8.14 → 9.30 | 91.86 → 90.70 | 664.0 → 656.9 |

The turbo greedy result on `test-other` recovered one additional target occurrence
(78 → 79 of 86), while beam-5 regressed by two total word errors. On clean speech,
target recall did not improve; Tiny beam-5 lost one target occurrence. This sample
is too small, and context too reference-dependent, to establish a general benefit.

There were zero emitted distractor-word occurrences across all measured modes.
Target precision was 100% on these selected target words; neither observation
establishes a population-level false-insertion bound. Irrelevant context still
changed some transcripts, consistent with EOS compensation influencing search.

The measured maximum active MLX memory was approximately 0.41 GiB for Tiny and
2.04 GiB for turbo. This is not process RSS or total system memory. Baseline and
boosted memory peaks are available in the JSON summaries. Single-run timing noise
and changed output lengths can make a boosted median appear slightly faster;
these results do not establish a speedup. GPU benchmarks were run sequentially.

Artifacts:
[Tiny / clean](validation/phrase-tiny-clean.json),
[Turbo / clean](validation/phrase-turbo-clean.json),
[Turbo / other](validation/phrase-turbo-other.json).

## Scorer microbenchmark

51,865 vocabulary entries, three-token synthetic phrases, float32 scores/int32
states, 30 timed warm queries per case. Tree-build timing excludes device-buffer
construction. The scorer returns full-vocabulary score/state arrays but stores
only sparse tree arcs.

| Phrases | Tree build ms | Query ms, 1 row | Query ms, 5 rows |
| ---: | ---: | ---: | ---: |
| 10 | 0.025 | 0.699 | 1.080 |
| 50 | 0.075 | 0.584 | 0.735 |
| 200 | 0.387 | 0.590 | 0.680 |
| 1,000 | 0.823 | 0.579 | 0.722 |
| 20,000 | 22.443 | 1.523 | 1.096 |

All cases had **zero destination-state mismatches** against the Python reference;
maximum absolute score error was below `1.9e-8`. This is a scorer benchmark, not a
claim of quality or end-to-end performance for 20,000 phrases. Long phrases can
increase fallback traversal work. [Raw summary](validation/phrase-scorer.json).

## Regression and review checks

- Randomized whole-vocabulary comparisons across every state of overlapping trees;
  hand-calculated rewards, rollback, completed suffixes, EOS, and ignored controls.
- Beam parent duplication/reordering, multi-audio isolation, synthetic termination,
  greedy/sampling state, request reuse/reset, and independent concurrent-request state.
- Separate model confidence versus fused search scores; suppression cannot be
  reversed by boosting. No-speech probability remains unchanged.
- Empty and zero-strength context exactly match unboosted tokens, confidence,
  and candidate dictionaries in both greedy and beam decoding.
- Unboosted transcription JSON, including candidates, matched the original checkout
  byte for byte for the bundled clip under greedy and beam-3 timestamp decoding.
- Real-audio CLI, word timestamps, >30-second multi-window audio, sampling retries,
  silence, and low-level noise integration tests.
- Reviewed normalization order, finished-beam bookkeeping, lazy-array lifetime,
  tokenizer boundaries, packaging, and documentation. GPU bookkeeping is realized
  each step to avoid retaining an unbounded history of lazy score arrays.

Final regression command (with `RUN_MLX_WHISPER_INTEGRATION=1`):

```sh
uv run pytest -q test_boosting.py test_boosting_evaluation.py \
  test_beam_search.py test_boosting_integration.py test_beam_search_integration.py test.py
```

**60 passed, 3 failed.** All **52 focused phrase/beam/metric/integration tests
passed**. The three failures are existing numerical checks in `test.py`:
`test_decode_greedy` (no-speech probability expectation), `test_decode_lang`
(language probability expectation), and `test_torch_mlx` (cross-backend numerical
comparison). All three reproduced on an untouched archive of the baseline commit
on the same host; they were not weakened or changed by this work. The observed
no-speech and language probabilities were identical in the original and modified
runs. This is not a claim that the entire legacy suite is green.

Ruff correctness/unused-import checks, formatting checks, Python 3.8 syntax parsing,
and `git diff --check` passed. A clean wheel build was inspected: it includes the
scorer and MIT license, excludes development benchmark scripts, and passed the
source-reference audit. Public validation data is used only by optional evaluation
scripts. [Implementation file hashes](validation/source-hashes.json) identify the
measured working-tree sources.

## Rebase validation (2026-11-05)

Rebased onto `f3d570f` (MLX beam selection, including shared cross-attention
caching). Phrase states and cumulative rewards now follow the selected MLX parent
indices. Only rewards for finished candidates enter the CPU archive.

On this Apple Silicon host, using the locked dependencies in a fresh `uv`
environment (Python 3.13.13, MLX 0.32.2), the following passed **71 tests**:

```sh
RUN_MLX_WHISPER_INTEGRATION=1 uv run python -m pytest -q \
  test_boosting.py test_boosting_evaluation.py test_boosting_integration.py \
  test_beam_search.py test_beam_selection.py test_beam_search_integration.py
```

This includes the upstream real-audio baseline candidate equivalence and shared
cache tests, a guard against NumPy array conversion during boosted beam updates,
and multi-step reward/state checks with two audio inputs and beam sizes 1, 3, and 5.
The declared Python minimum remains 3.8; this run does not establish runtime
compatibility on every supported Python version. The historical public-dataset
benchmarks above were not rerun for this rebase.
