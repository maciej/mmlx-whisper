# MLX beam selection

The production decoder performs vocabulary sorting, candidate scoring, duplicate
removal, global ranking, active-beam selection and token gathering with MLX.
It still computes the same float32 log-softmax, expands up to beam width + 1
choices from each parent, and retains the requested number of unique active
sequences for each audio independently. EOS candidates are archived only when
encountered before the last active beam, as in the original implementation.

A beam is a competing partial transcript. The parent indices identify which
partial transcript produced each new row. `selected_parent_indices` is an MLX
array in output-row order, includes batch offsets, and is cleared by `reset()`.
Other independent state can gather using it; no phrase matching is implemented.
The existing cache interface receives a small Python list of these indices.

CPU work remains for the bounded finished-hypothesis dictionary, its patience
limit, completion detection and finalization. Per token, the host reads a boolean
mask of B*(B+1) entries per audio and B parent indices, plus prefixes/scores for
new EOS candidates. It no longer reads the B*V vocabulary scores, all active
prefixes, or all cumulative scores. These are source-derived transfer counts,
not hardware bus measurements; Apple Silicon has unified memory. EOS gathers can
introduce extra synchronization. The benchmark includes those costs.

Sorting uses MLX's stable GPU `argsort`, including for the full vocabulary.
Partial selection was deliberately avoided here because its undefined tie order
would require additional cutoff-tie handling. Duplicate comparison is quadratic
in the compact candidate count B*(B+1), so this implementation targets ordinary
Whisper beam widths (tested 1/3/5/10), not very wide searches.

Tie policy: lower token ID within a parent, then earlier winning parent in the
batch. Duplicate sequences retain their highest score and, for equal scores,
the earliest parent. This deliberately replaces NumPy's unspecified quicksort
tie ordering. If a later parent improves a duplicate to tie another candidate,
its winning-parent position determines the order (the old dictionary retained
the duplicate's original insertion position). No score perturbation is used.
Genuine negative-infinity candidates remain eligible after finite candidates;
a separate validity mask prevents removed duplicates from filling those slots.
As in the original decoder, inputs must provide enough distinct non-EOS
extensions to fill the beam (normal Whisper vocabularies do). All-NaN/all-
suppressed distributions are not meaningful probability distributions.
Final archive sorting and length penalties are unchanged.

## Reproduce

From the repository root, in an environment with MLX and the project's test
requirements, on an Apple GPU:

```sh
uv run pytest -q tests/test_beam_search.py tests/test_beam_selection.py
RUN_MLX_WHISPER_INTEGRATION=1 uv run pytest -q tests/test_beam_search_integration.py
python -m benchmarks.beam_selection --output /tmp/beam-results.json
```

`--selection-only` skips model/audio work. `--models`, `--beams`, `--repeats`,
and `--data` control scope. The audio benchmark expects public LibriSpeech
`test-clean/61/70970/61-70970-0001.flac` and
`test-other/6070/86744/6070-86744-0007.flac` under `--data`; its default points to
`/tmp/mmlx-boost-validation/LibriSpeech`. Models are read from the Hugging Face
cache; the exact snapshot revisions are recorded. No downloads enter the repo.
`baseline_decoding.py` freezes the original beam class from commit
`80e60509090601a58acf1e6d0d4c2e34b8e5a5de`; both implementations use the same
current model, audio preprocessing, filters, cache and ranking code.

Each timed case excludes one warm-up. Selection uses 30 repetitions, fixed random
logits, a 51,865-token vocabulary, 64-token duplicate prefixes and no cache work.
End-to-end transcription uses audio already loaded into memory (includes mel
processing, encoding, decoding and segmentation, excludes file I/O/model loading).
Batch timings use precomputed mels and include encoding/decoding. The 44.43-second
long input concatenates the clean/other pair three times; it is a windowing test,
not an independent natural long recording. Timestamp cases use the short clean
input. Temperature is zero, language English, FP16 model, default patience and
length normalization. Transcription fallbacks and silence skipping are disabled
so both paths execute comparable work. Candidate output is enabled.

Runs are sequential, baseline then optimized, without competing GPU benchmarks.
Medians, individual times, min/max and MLX peak allocated bytes are recorded.
Peak allocation includes resident model/live arrays and is not process RSS or
system memory pressure. Runs are warm and not randomized; thermal/order effects
are possible. Exact equality compares complete returned tokens, timestamps,
text and ranked candidate records, including their scores. This is an equivalence
and performance check, not an accuracy or word-error-rate study.
