# Phrase boosting

MMLX Whisper implements [TurboBias](https://arxiv.org/abs/2508.07014) phrase
boosting in MLX. The implementation uses existing Whisper
weights, and requires neither training nor a second model. No PyTorch operations
are used by the phrase scorer or its decoding integration.

## Python and CLI

```python
from mlx_whisper import PhraseBoostingConfig, transcribe

result = transcribe(
    "meeting.wav",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    language="en",
    temperature=0.0,
    beam_size=5,
    phrases=["NVIDIA DGX", "Kubernetes", "Kubernetes cluster"],
    phrase_boosting=PhraseBoostingConfig(strength=0.1),
    return_candidates=True,
)
```

The same `phrases` and `phrase_boosting` arguments work with `model.decode()`
and `DecodingOptions`. One phrase list applies to all audio items in a decode
batch. Use separate decode requests for different lists. No phrase state is
stored on the model. Translation tasks match phrases in the **output** language.

For the CLI, provide a UTF-8 file with one phrase per line:

```sh
uv run mmlx_whisper meeting.wav --language en --beam-size 5 --temperature 0 \
  --phrases-file phrases.txt --phrase-boost-strength 0.1
```

Greedy decoding also supports boosting: omit `beam_size` / `--beam-size`.
Nonzero-temperature sampling and best-of sampling retain boosting. Each retry
starts with fresh matching state. Higher temperature samples more randomly from
the boosted scores; it does not turn boosting off.

`phrases=None`, `phrases=[]`, or `strength=0` does not construct/query a phrase
tree and takes the baseline decoder path. Whitespace-only lists also disable the
scorer. The default is **off** because no phrases are supplied.

## How it works

Whisper generates integer tokens representing text fragments. An Aho–Corasick
prefix tree tracks which supplied phrases match the recently generated tokens.
It rewards phrase starts and continuations, subtracts rewards for abandoned
partial matches, and makes completed-node fallback edges penalty-free. Further
fallback edges can still reconcile an overlapping, incomplete suffix. Failure links allow
matching overlapping phrases without restarting from scratch.

| Configuration | Default | Meaning |
| --- | ---: | --- |
| `context_score` | 1.0 | First-token reward |
| `depth_scale` | 1.0 | For token depth `d > 1`, reward is `context_score * depth_scale + log(d)` |
| `unknown_score` | 0.0 | Root transition score for a token that starts no phrase |
| `strength` | 0.1 | Multiplier applied to every tree score |
| `final_eos_score` | 1.0 | Extra end-of-transcript reward at a completed phrase |

All parameters must be finite; only `unknown_score` may be negative. These are
heuristic score controls, not probabilities. The defaults are a conservative
starting point, not a validated accuracy preset. Tune on development recordings
and assess the chosen settings on separate recordings, including irrelevant
phrase lists and silence.

The integration preserves MMLX Whisper's existing token suppression and timestamp
rules. It normalizes the **filtered, unboosted** model scores, then adds the tree
scores without renormalization. Final ranking applies the existing length penalty
to the cumulative fused score. This is additive shallow fusion; it does not claim
identical full-decoder behavior to other Whisper implementations, whose filtering,
normalization, search, and stopping rules may differ.

The end-of-transcript token gets the largest non-negative tree reward available
at that step, plus `final_eos_score` at a completed phrase, before applying
`strength`. This counters the tendency to keep generating boosted text instead
of stopping. Suppressed tokens remain suppressed, including EOS when its mask is
active. A synthetic EOS appended at the token limit receives no additional score.

## Confidence and returned scores

`avg_logprob` and candidate `sum_logprob` exclude phrase rewards. They describe
the selected path under the existing filtered model distribution; they are not a
calibrated probability that the transcript is correct. Retry and silence decisions
continue to use these unboosted values. No-speech probability is computed from
Whisper's original first-pass logits.

When boosting is active, results and candidates also expose:

- `search_score`: cumulative fused score before final length normalization;
- `boost_score`: cumulative external phrase reward;
- candidate `score`: fused search score after the existing length normalization.

For candidates, `sum_logprob + boost_score ≈ search_score` (float32 rounding).
The scores include a naturally generated EOS transition. In `transcribe()` output,
`search_score` and `boost_score` are **decode-window** diagnostics, just as the
existing confidence diagnostics are; multiple timestamp segments from one window
share them. They must not be summed as if each segment earned a separate reward.
Disabled boosting adds no score keys to transcription segments or candidate dicts.

## Boundaries, limits, and caching

Both leading-space and no-leading-space tokenizations are compiled. Duplicate
sequences are removed. Supply spelling, capitalization, inflection, and spoken
acronym variants explicitly. Matching is token-based; it is not a pronunciation
lexicon, a hard whole-word constraint, or a canonical-output rewriting system.
A shorter phrase may also be a prefix of a longer word or phrase.

Prompt/prefix tokens do not advance the matcher. Whisper control and timestamp
tokens receive zero boost and preserve phrase state; EOS has the special policy
above. Matching resets for every decode window and retry. A phrase crossing an
audio-window boundary is not guaranteed a continuous boost. This API does not
provide a persistent streaming matcher.

All phrases use the same global rewards. Per-phrase weights are not supported.
Limits are 20,000 input phrases, 1,024 characters and 128 tokens per phrase variant,
and 200,000 total tokens across distinct variants. Special-token spellings are
rejected. Eight compiled Python trees are cached by token sequences, vocabulary
size, and configuration. Device buffers and mutable state are request-local.

The tree stores sparse arcs, while each query produces `[active rows, vocabulary]`
score and destination arrays. Float32 scores and int32 states remain in MLX.
The scorer has no device-to-host reads. Beam candidate selection, deduplication, and pruning run in MLX. Phrase states
and accumulated rewards follow its selected parent indices on-device. Only
compact finished-hypothesis rewards cross to the CPU archive; finalization and
some orchestration still use Python/NumPy. The implementation
uses standard MLX operations; no custom Metal kernels are required.

## Validation and reproducible benchmarks

```sh
uv run pytest -q tests/test_boosting.py tests/test_boosting_evaluation.py tests/test_beam_search.py
RUN_MLX_WHISPER_INTEGRATION=1 uv run pytest -q \
  tests/test_boosting_integration.py tests/test_beam_search_integration.py
uv run python scripts/benchmark_phrase_scorer.py --output /tmp/scorer.json
```

Integration tests use the bundled LibriSpeech audio and download the default tiny
model if it is not cached. `MLX_WHISPER_TEST_MODEL` selects another MLX model.

For speech evaluation, download and extract the public
[LibriSpeech test-clean corpus](https://www.openslr.org/12) (CC BY 4.0; Panayotov,
Chen, Povey, and Khudanpur). The official archive MD5 is
`32fa31d27d2e1cad72775fee3f4849a9`. Keep downloaded audio outside the repository.

```sh
uv run python scripts/evaluate_phrase_boosting.py /path/to/LibriSpeech/test-clean \
  --model mlx-community/whisper-large-v3-turbo --limit 40 --beams 1 5 \
  --output /tmp/phrase-evaluation.json
```

The script selects utterances deterministically, round-robin across speakers,
then compares baseline, relevant, and distractor lists with identical model and
decoder settings. Modes rotate their execution order. Relevant phrases are the
three longest distinct reference words of at least six letters; lowercase and
title-case forms are supplied. This is explicitly **oracle context** derived from
known transcripts, useful for a diagnostic, not an estimate of production gains.
The fixed strength is not tuned on this test set.

It records per-sample outputs in JSONL plus summary WER, B-WER, target recall/precision,
distractor insertions, latency, real-time factor, and peak MLX active memory.
B-WER here counts substitutions/deletions at target reference positions plus
insertions of target words, divided by reference target occurrences. Levenshtein
alignment ties prefer diagonal, deletion, then insertion.
Token-count recall/precision concern exact single-word targets; they do not measure
multiword phrase order or phonetic similarity. WER uses lowercase ASCII words with
internal apostrophes retained. Timing excludes loading audio files and the initial
model warmup, but includes mel processing, encoding, decoding, and request-local
phrase preparation. MLX active memory is not total process memory. Short samples
and machine load make relative timing noisy; use repeated runs for capacity planning.

See [the measured host results](phrase-boosting-validation.md) for the validation
scope and limitations. Public English read speech does not establish Polish,
conversation, noisy-room, or domain-specific quality.
