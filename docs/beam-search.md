# Beam search specification

Beam search maintains several competing partial transcripts for each audio item.
This document describes the implemented decoding behavior and its interfaces.
The implementation is in [mlx_whisper/decoding.py](../mlx_whisper/decoding.py);
[benchmark documentation](../benchmarks/beam_selection/README.md) covers the MLX
selection algorithm, reproduction commands, and performance limitations.

## Public options

Use `beam_size` with zero temperature through `decode()` or `transcribe()`:

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.flac",
    path_or_hf_repo="mlx-community/whisper-tiny",
    beam_size=5,
    temperature=0.0,
    return_candidates=True,
)
```

Install the fork using the [README setup](../README.md#setup). Run commands
with `uv run`; the CLI is `mmlx_whisper` and the Python import is `mlx_whisper`.

The corresponding CLI options are `--beam-size`, `--temperature`, `--patience`,
and `--length-penalty`.

| Option | Behavior |
|---|---|
| `beam_size` | Number of active partial transcripts per audio item. |
| `patience` | Defaults to 1.0; the finished-candidate limit is `round(beam_size * patience)` and must be positive. |
| `length_penalty` | Optional final-ranking exponent in [0, 1]; it does not change expansion scores. |
| `return_candidates` | Includes all final candidates in descending ranking-score order. |

`beam_size` and `best_of` are mutually exclusive. `patience` requires
`beam_size`, and `best_of` is invalid at zero temperature in decoding options.
During transcription temperature fallback, nonzero-temperature attempts remove
`beam_size` and `patience`; zero-temperature attempts remove `best_of`.
Greedy and sampling decoding continue to use `GreedyDecoder`.

## State and shapes

`BeamSearchDecoder` receives the beam size, end-of-text token (`eot`), inference
object, and optional patience. Let N be the audio batch size, B the beam size,
L the current prefix length, and V the vocabulary size.

| Value | Shape |
|---|---|
| Input tokens | `(N * B, L)` |
| Filtered logits | `(N * B, V)` |
| Cumulative log probabilities | `(N * B,)` |
| Output tokens | `(N * B, L + 1)` |
| Selected parent indices | `(N * B,)` |

Rows are grouped by audio item. `DecodingTask.run()` repeats initial tokens for
each group. Encoder features and cross-attention keys/values stay per audio
item; attention broadcasts them over contiguous groups of hypotheses. Only
self-attention cache rows follow selected parents. This also supports `best_of`
sampling. Results retain the original audio features, and no-speech probabilities
are reduced to one per audio item.

`reset()` clears the per-audio finished-sequence archive and sets
`selected_parent_indices` to `None`. Inference cache state is reset separately
by the decoding task.

## One decoding step

`update(tokens, logits, sum_logprobs)` returns updated tokens, a completion
boolean, and updated cumulative log probabilities. Token and cumulative-score
dtypes are preserved.

1. Validate that the row count is divisible by B. Compute log-softmax using
   float32 logits, after the existing blank, token-suppression, and timestamp
   filters have run.
2. Select up to B + 1 token extensions per parent using stable MLX sorting.
   Add each token's log probability to its parent's cumulative score.
3. Deduplicate identical extended token sequences within each audio group.
   Keep the highest cumulative score and its parent; equal duplicate scores
   retain the earlier parent.
4. Rank unique candidates by cumulative score. Retain the first B non-EOT
   candidates as active beams. Archive EOT candidates only if they occur before
   the Bth active candidate in that order.
5. Gather output prefixes and append their selected tokens. Reorder the decoder
   key/value cache using exactly the same parent order, including after the
   first full-context step.
6. Merge newly finished sequences into each audio's archive, sort by cumulative
   score, and retain at most `max_candidates`. Completion is true when every
   audio's archive reaches that limit. The decoding loop also applies its
   existing sample-length and context-length limits.

Vocabulary selection, scoring, duplicate detection, ranking, and active-token
assembly run in MLX. The host reads a compact completion mask and parent indices,
plus prefixes and scores for new EOT candidates. Finished-archive bookkeeping,
completion decisions, and finalization remain on the CPU; full vocabulary scores
are not copied to NumPy each step.

### Parent ancestry

After an update, output row `k` descends from input row
`selected_parent_indices[k]`. This MLX array includes audio-group offsets and
is in output-row order. Independent per-beam state can gather with these same
indices. The cache interface receives the equivalent Python index list.
Repeated parents are valid, and selections never cross audio groups. No phrase
matching or phrase-boosting state is implemented here.

### Ties and suppression

Within a parent, equal token scores prefer lower token IDs. Across candidates
with equal cumulative scores, the earlier winning parent takes precedence.
If a later parent improves a duplicate to tie another candidate, ordering uses
that winning parent's position rather than the duplicate's first occurrence.
This is an explicit change from the original NumPy quicksort/dictionary tie
behavior; scores are not perturbed to break ties.

A separate validity mask distinguishes removed duplicates from genuine
negative-infinity candidates. Inputs must offer enough distinct non-EOT
extensions to fill the beam; all-suppressed or NaN distributions are not
meaningful probability distributions.

## Finalization and ranking

If an audio's archive contains fewer than B sequences, `finalize()` adds the
highest-scoring unfinished beams with an appended EOT until B are available.
Extra finished candidates allowed by patience are retained for final ranking.

Finalization returns dense token arrays of shape `(N, C, padded_length)` and
score arrays of shape `(N, C)`, where C is the largest candidate count across
the batch. Tokens are padded with EOT; missing candidate slots have score
negative infinity. Post-processing removes the prompt and truncates at EOT.

`MaximumLikelihoodRanker` divides cumulative log probability by generated-token
length when `length_penalty` is absent. With exponent alpha, the divisor is
`((5 + length) / 6) ** alpha`. Ranking uses these normalized scores, while the
reported `avg_logprob` divides by `length + 1` to account for EOT.

When requested, candidate records include rank, original candidate index,
selection flag, text, tokens, cumulative/average log probabilities, ranking
score, length, EOT metadata, timestamp presence, and compression ratio.
`transcribe()` attaches candidate records to decoded segments.

## Verification

Run deterministic tests without model or dataset downloads:

```sh
uv run pytest -q test_beam_search.py test_beam_selection.py
```

The tests cover expansion, duplicate winners, ties, suppression, EOT cutoff,
patience, finalization, batch isolation, cache ancestry, dtypes, option validation,
and greedy/sampling behavior. Differential tests compare multiple decoding steps
against the frozen original decoder for beam sizes 1, 3, 5, and 10.

Model-backed tests are opt-in and default to `mlx-community/whisper-tiny`, using
the repository's audio fixture:

```sh
RUN_MLX_WHISPER_INTEGRATION=1 uv run pytest -q test_beam_search_integration.py
```

They cover API and CLI transcription, timestamps, batch decoding, candidate
outputs, and baseline equivalence with patience and length penalties. Coverage
also includes distinct speech/silence batches, sampled `best_of` groups, forced
temperature fallback, word timestamps, and shared-cache attention equivalence.

The [benchmark directory](../benchmarks/README.md) indexes performance studies.
The [Apple M5 report](../benchmarks/beam_selection/RESULTS.md) records exact
versions, timing and memory measurements, candidate equivalence, and the three
pre-existing full-suite failures reproduced on unchanged main. Those results
establish equivalence on the tested samples, not a general word-error-rate
improvement. The recorded timings precede the shared-cache changes on `main`;
they do not measure the incremental gain over that newer baseline. A broad
beam-versus-greedy accuracy study remains outside this validation suite.

## Reference implementation and adaptations

The semantic reference was OpenAI Whisper's
[`whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py),
in particular:

- `BeamSearchDecoder` for active/finished hypothesis management and patience;
- `DecodingTask.run` for grouped decoding and final ranking;
- `PyTorchInference.rearrange_kv_cache` for keeping cached attention state
  aligned with pruned beams.

The implementation ports those semantics rather than copying the PyTorch code
line for line. The main MLX-specific adaptations are:

- candidate selection, deduplication, and pruning use MLX; compact completion
  information and finished hypotheses are materialized for CPU bookkeeping;
- queries are grouped by audio item to broadcast shared cross-attention K/V
  without duplicating it across beams;
- selected parent indices are applied only to the self-attention KV cache;
- ragged finished sequences are converted back to dense EOT-padded MLX arrays
  for the existing result pipeline;
- optional ranked candidate metadata was added as a mmlx-whisper extension.

Beam search first landed in
[`38d677d`](https://github.com/maciej/mmlx-whisper/commit/38d677d295a1f5e07541a6359bbc9c10d0bbaf36).
The Whisper code and its history were then extracted from the MLX examples fork
into this standalone fork in
[`e497280`](https://github.com/maciej/mmlx-whisper/commit/e4972801adea0c3e79d619df48fc9beb39df39b0),
which also added ranked candidate output.


## Related investigations

The [beam-search investigation](beam-search-investigation.md) and
[production validation](beam-search-validation.md) document the earlier
compact-candidate selection and shared cross-attention cache work, with its
own benchmark variants and measurements. Those reports describe their recorded
revisions; the current decoder additionally performs deduplication and ranking
in MLX.
