"""Deterministic differential checks against the frozen pre-optimization decoder."""
import unittest
import numpy as np
import mlx.core as mx
from benchmarks.beam_selection.baseline_decoding import BeamSearchDecoder as Reference
from mlx_whisper.decoding import BeamSearchDecoder, Inference, GreedyDecoder
from tests.test_beam_search import FakeInference


class TestSelection(unittest.TestCase):
    def test_randomized_multistep_duplicates_and_finished(self):
        rng = np.random.default_rng(314159)
        for beam in (1, 3, 5, 10):
            old = Reference(beam, 30, FakeInference(), patience=2)
            new = BeamSearchDecoder(beam, 30, FakeInference(), patience=2)
            tokens = mx.array([[i // beam] * 3 for i in range(2 * beam)])
            scores = mx.array(rng.normal(size=2 * beam).astype(np.float32))
            for step in range(8):
                logits = rng.normal(size=(2 * beam, 31)).astype(np.float32)
                logits[:, 30] += 2 if step % 2 else -2
                logits = mx.array(logits)
                a, done_a, sa = old.update(tokens, logits, scores)
                b, done_b, sb = new.update(tokens, logits, scores)
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.tolist(), b.tolist())
                np.testing.assert_allclose(sa, sb, atol=1e-6)
                self.assertEqual(done_a, done_b)
                self.assertEqual(old.finished_sequences, new.finished_sequences)
                self.assertEqual(old.inference.rearrange_calls, new.inference.rearrange_calls)
                self.assertEqual(new.selected_parent_indices.tolist(), new.inference.rearrange_calls[-1])
                tokens, scores = b, sb
            fa, fsa = old.finalize(tokens.reshape(2, beam, -1), scores.reshape(2, beam))
            fb, fsb = new.finalize(tokens.reshape(2, beam, -1), scores.reshape(2, beam))
            self.assertEqual(fa.tolist(), fb.tolist())
            np.testing.assert_array_equal(fsa, fsb)

    def test_ties_are_lower_token_then_earlier_parent(self):
        decoder = BeamSearchDecoder(3, 0, FakeInference())
        tokens, done, scores = decoder.update(mx.array([[7], [7], [8]]), mx.zeros((3, 8)), mx.zeros(3))
        self.assertEqual(tokens.tolist(), [[7, 1], [7, 2], [7, 3]])
        self.assertEqual(decoder.selected_parent_indices.tolist(), [0, 0, 0])
        self.assertEqual(list(decoder.finished_sequences[0]), [(7, 0)])
        decoder.reset()
        self.assertIsNone(decoder.selected_parent_indices)

    def test_suppressed_negative_infinity_is_not_duplicate_padding(self):
        decoder = BeamSearchDecoder(3, 9, FakeInference())
        logits = mx.full((3, 10), -mx.inf)
        logits[:, 4] = 0
        tokens, _, scores = decoder.update(mx.array([[7], [7], [7]]), logits, mx.zeros(3))
        self.assertEqual(tokens.tolist(), [[7, 4], [7, 0], [7, 1]])
        self.assertEqual(scores.tolist(), [0, -float('inf'), -float('inf')])

    def test_eos_below_active_cutoff_is_not_archived(self):
        decoder = BeamSearchDecoder(2, 4, FakeInference())
        tokens, completed, _ = decoder.update(
            mx.array([[7], [7]]), mx.array([[5., 4., 0., -1., 3.]] * 2), mx.zeros(2)
        )
        self.assertEqual(tokens.tolist(), [[7, 0], [7, 1]])
        self.assertEqual(decoder.finished_sequences, [{}])
        self.assertFalse(completed)

    def test_later_duplicate_winner_owns_cache_and_score(self):
        decoder = BeamSearchDecoder(2, 4, FakeInference())
        tokens, _, scores = decoder.update(
            mx.array([[7], [7]], dtype=mx.int32),
            mx.array([[5., 4., 0., -1., -2.]] * 2),
            mx.array([-3., 0.], dtype=mx.float16),
        )
        self.assertEqual(tokens.tolist(), [[7, 0], [7, 1]])
        self.assertEqual(decoder.selected_parent_indices.tolist(), [1, 1])
        self.assertEqual(tokens.dtype, mx.int32)
        self.assertEqual(scores.dtype, mx.float16)

    def test_actual_cache_gather_with_repeated_and_reversed_parents(self):
        inference = Inference(None)
        self_kv = (mx.arange(12).reshape(3, 4), mx.arange(6).reshape(3, 2))
        cross_kv = (mx.ones((1, 4)), mx.ones((1, 2)))
        inference.kv_cache = [(self_kv, cross_kv)]
        inference.rearrange_kv_cache([2, 0, 2])
        self.assertEqual(inference.kv_cache[0][0][0].tolist(), [[8, 9, 10, 11], [0, 1, 2, 3], [8, 9, 10, 11]])

        self.assertIs(inference.kv_cache[0][1], cross_kv)
        self.assertEqual(inference.kv_cache[0][0][1].tolist(), [[4, 5], [0, 1], [4, 5]])

    def test_sampling_is_reproducible_and_ended_rows_stay_ended(self):
        def run():
            mx.random.seed(42)
            return GreedyDecoder(0.7, 4).update(mx.array([[1], [4]]), mx.array([[1., 2., 3., 0., -1.]] * 2), mx.zeros(2))
        a, _, sa = run()
        mx.eval(a, sa)
        b, _, sb = run()
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(a.tolist()[1], [4, 4])
        self.assertEqual(sa.tolist()[1], 0)
