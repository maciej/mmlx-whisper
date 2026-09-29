# Beam search

Beam-search decoding is implemented end to end in the standalone mmlx-whisper
fork. Install this repository using the [README setup](../README.md#setup)
before running the examples below. The Python import is `mlx_whisper` and the
CLI command is `mmlx_whisper`.

Beam search is available through `decode()`, `transcribe()`, and the
`mmlx_whisper` CLI. It supports batched audio and timestamps, and preserves
Whisper's temperature-fallback behavior.

This page documents the implementation as it exists today, how it was derived
from OpenAI Whisper, and the remaining trade-offs. It replaces the original
implementation spec that preceded the code.

## Using beam search

Beam search applies at zero temperature:

```sh
uv run mmlx_whisper audio_file.mp3 --beam-size 5 --temperature 0
```

The equivalent Python API is:

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio_file.mp3",
    beam_size=5,
    temperature=0.0,
)
```

The related options follow OpenAI Whisper's decoding API:

- `beam_size` sets the number of active hypotheses per audio item.
- `patience` defaults to `1.0` and changes the number of finished candidates
  retained to `round(beam_size * patience)`.
- `length_penalty` changes final candidate ranking. It is not applied while
  beams are expanded.
- `return_candidates=True` includes every ranked final candidate in
  `DecodingResult.candidates` and, when using `transcribe()`, on the associated
  segments and in the top-level `candidate_segments` view.

`beam_size` and `best_of` are mutually exclusive. `patience` requires
`beam_size`, and `length_penalty` must be between `0` and `1`.

## Current implementation

[`DecodingTask`](../mlx_whisper/decoding.py) selects `BeamSearchDecoder`
whenever `beam_size` is present. The initial token sequence is repeated by the
beam group size. Encoder features and cross-attention keys/values stay per audio
item; attention broadcasts them over that item's contiguous group of hypotheses.
This layout also supports `best_of` sampling and distinct audio items in a batch.

On every decoding step:

1. The existing blank, token-suppression, and timestamp filters are applied to
   the logits.
2. Log probabilities are calculated in `float32`.
3. MLX selects each beam's best `beam_size + 1` token extensions. Only these
   token IDs and scores, token histories and cumulative scores cross to NumPy.
4. Candidate sequences are deduplicated and ranked by cumulative log
   probability for each audio item.
5. EOT-terminated candidates move to a per-audio finished set. The best
   `beam_size` unfinished candidates remain active.
6. Self-attention KV is reordered to match the selected parent rows.
   Cross-attention KV remains shared per audio item.
7. Decoding completes when every audio item has
   `round(beam_size * patience)` finished candidates, or when the normal sample
   length/context limit is reached.

If decoding reaches a limit before enough candidates finish,
`BeamSearchDecoder.finalize()` appends EOT to the best unfinished beams until
there are at least `beam_size` candidates. The results are returned as dense
MLX arrays padded with EOT. This keeps the shared post-processing path usable
even though candidate sequences have different lengths and `patience` may
produce more candidates than active beams.

Final selection remains the responsibility of `MaximumLikelihoodRanker`.
Without an explicit `length_penalty`, it uses average log probability. With a
penalty, it uses the Google NMT length-penalty formula. Ranked candidate output
uses the same scores as final selection.

Equal token scores use increasing token ID as a deterministic tie-breaker;
equally scoring duplicate sequences retain the first parent encountered.
Cumulative-score sorting retains insertion order for ties. Scores are not
perturbed, and compatibility with the old NumPy sort's incidental tie ordering
is not required.

During transcription fallback, beam search and patience are removed when a
non-zero temperature is attempted. At temperature zero, `best_of` is removed.
This is the same separation between deterministic beam search and sampled
fallbacks used by Whisper.

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

- candidate selection uses a stable MLX sort, with only compact candidates
  materialized after `mx.eval()` for Python hypothesis management;
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

## Verification

The default unit suite in [`test_beam_search.py`](../test_beam_search.py)
covers:

- deterministic beam expansion and cumulative scores;
- EOT handling and unfinished-beam finalization;
- patience-based completion;
- KV-cache reorder indices;
- isolation between multiple audio items;
- option validation and final ranking;
- unchanged greedy argmax behavior;
- tied/EOT and suppressed candidates;
- shared-cache storage and attention equivalence after repeated parent selection.

The opt-in model-backed suite in
[`test_beam_search_integration.py`](../test_beam_search_integration.py) covers
the Python transcription path with and without timestamps, the CLI, batched
decode, ranked candidate output, distinct speech/silence batches, sampled
`best_of` groups, forced temperature fallback and word timestamps.
After following the README setup, run from
the repository root:

```sh
RUN_MLX_WHISPER_INTEGRATION=1 \
  uv run pytest -q test_beam_search_integration.py
```

It uses `mlx-community/whisper-tiny` by default. Set
`MLX_WHISPER_TEST_MODEL` to exercise another compatible model.

## Known trade-offs

Beam expansion still synchronizes once per step to materialize compact candidates
and token histories. Python manages sequence deduplication, pruning and finished
hypotheses. MLX selection currently sorts the vocabulary on the device; in MLX
0.32.2, `argpartition` also uses a full GPU sort. A fused small-k selector and less
token-history copying are possible follow-ups if further profiling justifies them.

The model-backed suite is a functional smoke test, not a quality or performance
benchmark. The [beam-search investigation](beam-search-investigation.md) includes
a repeatable benchmark for candidate selection and cross-attention cache sharing,
with measured time and memory on tiny and medium. The
[production validation](beam-search-validation.md) records the implemented gains
and batch/fallback/alignment checks. Use `--variant production` for
the current implementation or `--variant baseline` for the frozen pre-optimization
selection and expanded cache layout. Dataset-level WER and a
repeatable greedy-versus-beam comparison remain future work.
