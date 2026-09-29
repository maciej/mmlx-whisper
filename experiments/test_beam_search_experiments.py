"""Differential checks for production decoding and the investigation prototypes."""
import mlx.core as mx
import numpy as np
import pytest

from experiments.beam_search import compact_update, select_candidates, self_cache_reorder
from mlx_whisper.decoding import BeamSearchDecoder, Inference
from experiments.beam_reference import baseline_update
from test_beam_search import FakeInference


@pytest.mark.parametrize("update", [BeamSearchDecoder.update, compact_update],
                         ids=["production", "prototype"])
def test_update_matches_baseline_for_untied_multi_audio_paths(update):
    rng = np.random.default_rng(21)
    for beam in (1, 3, 5):
        old_inf, new_inf = FakeInference(), FakeInference()
        old = BeamSearchDecoder(beam, 99, old_inf, patience=2)
        new = BeamSearchDecoder(beam, 99, new_inf, patience=2)
        old_tokens = new_tokens = mx.array([[audio] for audio in (10, 20) for _ in range(beam)])
        old_scores = new_scores = mx.zeros(2 * beam)
        for _ in range(8):
            logits = mx.array(rng.normal(size=(2 * beam, 100)), mx.float32)
            old_tokens, old_done, old_scores = baseline_update(old, old_tokens, logits, old_scores)
            new_tokens, new_done, new_scores = update(new, new_tokens, logits, new_scores)
            assert old_tokens.tolist() == new_tokens.tolist()
            assert old_scores.tolist() == new_scores.tolist()
            assert old_done == new_done
            assert old.finished_sequences == new.finished_sequences
            assert old_inf.rearrange_calls == new_inf.rearrange_calls
        ot, os = old.finalize(old_tokens.reshape(2, beam, -1), old_scores.reshape(2, beam))
        nt, ns = new.finalize(new_tokens.reshape(2, beam, -1), new_scores.reshape(2, beam))
        assert ot.tolist() == nt.tolist()
        assert os.tolist() == ns.tolist()


def test_stable_ties_and_suppressed_tail_keep_unique_ids():
    x = mx.array([[4., 4., 4., -float('inf')], [1., -float('inf'), -float('inf'), -float('inf')]])
    ids, values = select_candidates(x, 4)
    assert ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert values.tolist() == x.tolist()


def test_cross_cache_is_preserved_and_self_cache_follows_parents():
    inf = Inference(None)
    self_kv = (mx.arange(4)[:, None, None], mx.arange(4)[:, None, None] + 10)
    cross_kv = (mx.array([1, 1, 2, 2]), mx.array([3, 3, 4, 4]))
    inf.kv_cache = [(self_kv, cross_kv)]
    self_cache_reorder(inf, [1, 1, 3, 2])
    assert inf.kv_cache[0][0][0].flatten().tolist() == [1, 1, 3, 2]
    assert inf.kv_cache[0][0][1].flatten().tolist() == [11, 11, 13, 12]
    assert inf.kv_cache[0][1] is cross_kv


def test_shared_prototype_rejects_multiple_audio_items():
    from experiments.beam_search import install
    from mlx_whisper.whisper import MultiHeadAttention
    import pytest
    install('shared', 3)
    try:
        attention = MultiHeadAttention(8, 2)
        with pytest.raises(ValueError, match='one audio item'):
            attention(mx.zeros((6, 1, 8)), mx.zeros((6, 4, 8)))
    finally:
        install('production', 3)


def test_model_backed_distinct_audio_batch():
    import os
    import pytest
    from experiments.beam_search import install
    from mlx_whisper import audio
    from mlx_whisper.load_models import load_model
    model_path = os.environ.get('EXPERIMENT_MODEL')
    if not model_path:
        pytest.skip('set EXPERIMENT_MODEL to a local MLX model snapshot')
    model = load_model(model_path, mx.float16)
    speech = audio.pad_or_trim(audio.load_audio('mlx_whisper/assets/ls_test.flac'))
    batch = mx.stack([audio.log_mel_spectrogram(speech, model.dims.n_mels), audio.log_mel_spectrogram(mx.zeros_like(speech), model.dims.n_mels)])
    outputs = []
    try:
        for variant in ('baseline', 'compact', 'cache', 'production'):
            install(variant, 3)
            result = model.decode(batch, beam_size=3, temperature=0, language='en',
                                  without_timestamps=True, sample_len=32, return_candidates=True)
            outputs.append([(r.tokens, r.avg_logprob, r.candidates) for r in result])
        assert all(output == outputs[0] for output in outputs[1:])
    finally:
        install('production', 3)
