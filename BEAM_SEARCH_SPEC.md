Fully implement beam search decoding in the current mlx-whisper repository.

You are running inside the mlx-whisper repo on a Mac with an M5 chip and 24 GB RAM. You may clone reference repositories under /tmp or another temp directory, but do not vendor them into this repo and do not commit downloaded model weights, datasets, audio corpora, or generated test artifacts.

Primary objective
=================
Implement production-quality beam search for mlx-whisper so that using beam_size no longer raises NotImplementedError and works through both the Python API and CLI/transcribe path.

The implementation should match OpenAI Whisper’s beam-search semantics as closely as is practical in MLX:

- beam_size controls active hypotheses per audio item.
- patience defaults to 1.0 and sets max_candidates = round(beam_size * patience).
- length_penalty is not applied inside beam expansion; final selection should continue to use the existing MaximumLikelihoodRanker.
- EOT-ended sequences are moved into a per-audio finished set.
- Unfinished active beams are pruned back to beam_size per audio item each step.
- The decoder KV cache is reordered after pruning with Inference.rearrange_kv_cache(source_indices).
- Decoding stops once every audio item has enough finished candidates, or the normal sample_len/n_ctx stopping condition is hit.
- finalize() adds EOT-terminated unfinished beams when not enough finished candidates exist.
- Existing greedy decoding behavior must remain unchanged.

Reference material
==================
Clone OpenAI Whisper as the semantic reference:

    git clone --depth 1 https://github.com/openai/whisper.git /tmp/openai-whisper

Inspect /tmp/openai-whisper/whisper/decoding.py, especially BeamSearchDecoder, DecodingTask.run, and PyTorchInference.rearrange_kv_cache.

Do not blindly copy line-for-line. Port the behavior to the current MLX codebase and account for MLX array behavior, MLX lazy evaluation, current mlx-whisper data shapes, and current post-processing.

Implementation requirements
===========================

1. Add a BeamSearchDecoder implementation
-----------------------------------------
In the current decoding module, add a BeamSearchDecoder class implementing the TokenDecoder interface used by mlx-whisper.

Constructor:

- beam_size: int
- eot: int
- inference: Inference
- patience: Optional[float] = None
- set patience = patience or 1.0
- set max_candidates = round(beam_size * patience)
- assert max_candidates > 0
- keep self.finished_sequences, reset to None in reset()

update(tokens, logits, sum_logprobs) must return:

    tokens, completed, sum_logprobs

where:

- tokens has shape (n_audio * beam_size, current_sequence_length + 1)
- completed is a bool or MLX scalar compatible with the existing main loop
- sum_logprobs has shape (n_audio * beam_size,)

Algorithm:

- Validate tokens.shape[0] % beam_size == 0.
- n_audio = tokens.shape[0] // beam_size.
- Initialize self.finished_sequences = [{} for _ in range(n_audio)] on the first step.
- Compute logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True), using float32 logits.
- For each audio item i and active beam j:
  - parent row idx = i * beam_size + j.
  - Get the top beam_size + 1 candidate token IDs and logprobs for that row.
  - Important MLX detail: mx.topk returns values, not indices, and does not promise sorted order. Use mx.argsort, mx.argpartition + take_along_axis, or another correct MLX/NumPy path that yields both token IDs and scores in deterministic descending score order.
  - It is acceptable to start with a simple Python/NumPy bridge after mx.eval for correctness, as long as tests pass. Avoid unnecessary .tolist(), .item(), or np.array conversions in the hot path when it is easy to keep work in MLX, but correctness is the first milestone.
  - Accumulate candidate score = parent sum_logprobs + token logprob.
  - Store candidate sequence, score, and parent source index.
- Rank all candidates for each audio item by cumulative logprob descending.
- Move candidates ending in eot into that audio item’s newly finished set.
- Keep the best beam_size unfinished candidates as next active beams.
- Build source_indices in exactly the same order as the returned active tokens.
- Update sum_logprobs for returned active beams to the selected candidate scores.
- Call self.inference.rearrange_kv_cache(source_indices) after pruning, including after the first full-context decoder step.
- Merge newly finished sequences into self.finished_sequences, preserving descending score order and capping at max_candidates.
- completed is true only when every audio item has at least max_candidates finished candidates.

Be very careful that source_indices has length n_audio * beam_size and that each returned token row corresponds to the same parent cache row after reordering.

2. Replace the NotImplementedError branch
-----------------------------------------
In DecodingTask.__init__, replace the beam_size branch with:

    self.decoder = BeamSearchDecoder(
        options.beam_size,
        tokenizer.eot,
        self.inference,
        options.patience,
    )

Do not change the greedy branch except as necessary to preserve shared interfaces.

3. Make finalize() and DecodingTask.run compatible
--------------------------------------------------
Current MLX post-processing assumes decoder.finalize() returns dense MLX arrays and then does:

    tokens = tokens[..., self.sample_begin :]
    mx.eval(tokens, sum_logprobs, no_speech_probs)
    tokens = tokens.tolist()
    sum_logprobs = sum_logprobs.tolist()
    tokens = [[t[: t.index(tokenizer.eot)] for t in s] for s in tokens]

OpenAI’s BeamSearchDecoder returns ragged candidate lists. In MLX, choose one clean approach and make it robust:

Preferred approach:
- Have BeamSearchDecoder.finalize() return a dense padded mx.array of candidate tokens with shape:
      (n_audio, n_candidates, padded_sequence_length)
  where n_candidates is at least beam_size and may be max_candidates when patience > 1.
- Pad shorter candidates with eot so the existing EOT stripping logic works.
- Return sum_logprobs as an mx.array or a structure that DecodingTask.run can reliably convert to List[List[float]].
- Ensure n_candidates can differ from self.n_group after finalize; sequence_ranker.rank should still receive all candidates.

Alternative acceptable approach:
- Return ragged lists like OpenAI Whisper, but then update DecodingTask.run to branch cleanly for greedy/dense and beam/ragged outputs.
- Preserve all existing public DecodingResult fields.

Either approach must support patience > 1.0 and not silently drop extra finished candidates before ranking.

4. Fix/verify grouped audio feature handling
--------------------------------------------
When n_group > 1, tokens are repeated by group. Verify the current MLX decoder path also has audio_features shaped correctly for the decoder and for result post-processing.

Do not rely on single-audio broadcasting. Add tests with n_audio = 2 and beam_size > 1.

A robust pattern is:
- Keep original_audio_features for final DecodingResult output.
- Repeat or broadcast audio_features for decoding if needed so decoder batch dimensions match tokens.
- After decoding, restore/select one audio_features row per original audio item for DecodingResult.
- Ensure no_speech_probs is reduced/sliced to one value per original audio item.

5. Preserve logit filters and timestamp behavior
------------------------------------------------
Beam search must receive logits after existing filters are applied:

- SuppressBlank
- SuppressTokens
- ApplyTimestampRules

Do not reimplement those rules inside BeamSearchDecoder. They already operate before decoder.update(). Add tests that beam search works with both without_timestamps=True and the default timestamp mode.

6. Preserve transcribe fallback behavior
----------------------------------------
Inspect transcribe.py and CLI handling of beam_size, temperature fallback, patience, best_of, and length_penalty.

Expected behavior:
- beam search should be used when beam_size is present in decode options for temperature 0.
- existing behavior that disables beam_size for nonzero temperature fallback, if present, must remain intact.
- best_of and beam_size must remain mutually exclusive.
- patience without beam_size must remain invalid.
- length_penalty validation must remain intact.

Testing requirements
====================

Add tests. Do not rely only on manual audio transcription.

Default unit tests
------------------
These should run without downloading models or datasets.

Add focused tests for BeamSearchDecoder using fake logits and a fake Inference object whose rearrange_kv_cache(source_indices) records the calls.

Test cases:

1. Basic beam expansion:
   - n_audio = 1, beam_size = 2.
   - Provide deterministic logits for two or three steps.
   - Assert active tokens, active scores, finished sequences, completed flag, and source_indices.

2. EOT handling:
   - Ensure EOT candidates enter finished_sequences and are not kept as active beams.
   - Ensure beam_size active unfinished beams are retained when available.

3. finalize fallback:
   - Stop before enough beams finish.
   - finalize() must append eot to best unfinished active beams until at least beam_size candidates are available.

4. patience:
   - beam_size = 2, patience = 1.5 or 2.0.
   - Assert completed only after max_candidates finished candidates per audio.

5. KV-cache reorder:
   - Fake Inference must see exactly the selected parent indices, in returned-token order.

6. batch correctness:
   - n_audio = 2, beam_size = 2 or 3.
   - Assert source_indices and returned tokens are grouped per audio and do not mix audio items.

7. option validation / integration:
   - beam_size no longer raises NotImplementedError.
   - beam_size + best_of still raises ValueError.
   - patience without beam_size still raises ValueError.
   - length_penalty outside [0, 1] still raises ValueError.

8. greedy regression:
   - Existing greedy tests, if any, must still pass.
   - Add a small regression if none exists.

Run:

    python -m pytest -q

Integration smoke tests
-----------------------
Add integration tests that are skipped by default unless an environment variable is set, for example:

    RUN_MLX_WHISPER_INTEGRATION=1 python -m pytest -q tests/test_beam_search_integration.py

These tests may download a tiny MLX Whisper model and a very small amount of audio.

Use tiny/base only on this Mac. Do not use large-v3 for test gates.

Suggested integration coverage:

1. API smoke:
   - Load the default tiny MLX model or mlx-community/whisper-tiny, whichever the repo currently documents/uses.
   - Transcribe one short English audio file with:
       beam_size=3
       patience=None
       length_penalty=None
       temperature=0.0
       without_timestamps=True
       language="en"
   - Assert it returns a DecodingResult/result dict with non-empty text and no exception.

2. API with default timestamps:
   - Same audio with without_timestamps=False.
   - Assert it completes and segments/timestamps remain structurally valid through transcribe().

3. CLI smoke:
   - Run the repo’s CLI entrypoint on a short audio file with beam_size=3.
   - Assert exit code 0 and output file or stdout is produced, depending on current CLI behavior.

4. Batch decode smoke:
   - Construct or load two short mels/audio examples.
   - Call decode/transcribe path in a way that exercises n_audio=2 and beam_size=2 or 3.
   - Assert two outputs are returned and no shape assertion fails.

Whole-pipeline ASR regression
-----------------------------
Add either a pytest integration test marked slow or a script under scripts/, for example:

    scripts/eval_beam_search_librispeech.py

Use a known open ASR dataset:

- Preferred quick regression dataset: Mini LibriSpeech from OpenSLR SLR31, explicitly intended as a LibriSpeech regression subset.
- Preferred streaming/full test source: LibriSpeech test-clean via Hugging Face datasets, e.g. openslr/librispeech_asr with config "clean" and split "test", streaming=True. If the package naming differs in the current datasets version, adapt after checking the dataset card/API.
- Use only a small N by default, such as 10-25 utterances, so this is practical on an M5 with 24 GB RAM.

Evaluation behavior:

- Use jiwer for WER.
- Normalize references and hypotheses consistently. Use Whisper’s normalizer if already available in the repo or a simple documented normalizer; do not hide normalization choices.
- Compare greedy vs beam on the same examples:
    - greedy: temperature=0, beam_size=None
    - beam: temperature=0, beam_size=3 or 5
- Report:
    - N
    - model
    - beam_size
    - patience
    - greedy WER
    - beam WER
    - runtime for each
- Do not make a tight WER threshold in default CI. For the optional slow/integration gate, assert only that beam search is not catastrophically worse than greedy, for example beam WER <= greedy WER + a reasonable absolute tolerance after observing tiny-model variance on the chosen subset.
- Also compare one or two examples against OpenAI Whisper tiny on CPU as a semantic smoke test. Exact token equality is not required across frameworks, but text should be plausibly close.

Suggested command:

    RUN_MLX_WHISPER_INTEGRATION=1 \
    python scripts/eval_beam_search_librispeech.py \
      --model mlx-community/whisper-tiny \
      --dataset librispeech-test-clean \
      --num-samples 20 \
      --beam-size 5

Silence/no-speech smoke
-----------------------
Add a small generated-silence smoke test:

- Generate a short 16 kHz silent waveform or temporary WAV.
- Run transcribe with beam_size=3 or 5 and existing no_speech/logprob settings.
- Assert it completes and does not produce a long repeated hallucination.
- Do not overfit to exact empty text unless existing greedy behavior already guarantees it.

Performance check
=================
Beam search will be slower than greedy, but it should not be pathologically slow.

Add a simple benchmark note or script:

- Warm up once.
- Run one short clip with greedy, beam_size=3, and beam_size=5.
- Print wall-clock time and peak memory if easy.
- On the M5/24 GB machine, tiny/base should run comfortably.
- If the first correct implementation uses Python lists and CPU round-trips in the beam loop, leave a clear TODO and benchmark result. Optimize only after correctness tests pass.
- Avoid repeated unnecessary mx.eval/.tolist/.item calls in inner loops where a straightforward MLX version is practical.

Acceptance criteria
===================

The task is complete only when:

1. beam_size works end-to-end through decode(), transcribe(), and CLI paths.
2. beam_size=3 and beam_size=5 complete on at least one real audio file.
3. Multi-audio batch decoding with beam_size > 1 works.
4. patience and length_penalty are supported according to OpenAI Whisper semantics.
5. KV cache is reordered correctly after beam pruning.
6. Timestamp and suppress-token logic still applies before beam selection.
7. Greedy decoding behavior is unchanged.
8. Unit tests pass with python -m pytest -q.
9. Optional integration tests or scripts are added and documented, with clear skip behavior for missing network/model/dataset dependencies.
10. The implementation does not commit downloaded models, datasets, generated audio corpora, or reference repository files.

Final report
============
When finished, summarize:

- files changed
- implementation approach
- exact tests run and results
- any integration dataset/model used
- benchmark numbers, if collected
- remaining limitations or TODOs, especially any Python/CPU round-trips in the beam loop
