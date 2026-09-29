import os
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx

import mlx_whisper
import mlx_whisper.audio as audio
from mlx_whisper.load_models import load_model


RUN_INTEGRATION = os.environ.get("RUN_MLX_WHISPER_INTEGRATION") == "1"
MODEL_NAME = os.environ.get("MLX_WHISPER_TEST_MODEL", "mlx-community/whisper-tiny")
TEST_AUDIO = os.path.join(
    os.path.dirname(__file__), "mlx_whisper", "assets", "ls_test.flac"
)


@unittest.skipUnless(
    RUN_INTEGRATION,
    "set RUN_MLX_WHISPER_INTEGRATION=1 to run MLX Whisper integration tests",
)
class TestBeamSearchIntegration(unittest.TestCase):
    def test_transcribe_without_timestamps(self):
        result = mlx_whisper.transcribe(
            TEST_AUDIO,
            path_or_hf_repo=MODEL_NAME,
            beam_size=3,
            patience=None,
            length_penalty=None,
            temperature=0.0,
            without_timestamps=True,
            language="en",
        )

        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["text"].strip())

    def test_transcribe_with_timestamps(self):
        result = mlx_whisper.transcribe(
            TEST_AUDIO,
            path_or_hf_repo=MODEL_NAME,
            beam_size=3,
            temperature=0.0,
            without_timestamps=False,
            language="en",
        )

        self.assertTrue(result["segments"])
        for segment in result["segments"]:
            self.assertLessEqual(segment["start"], segment["end"])
            self.assertIsInstance(segment["tokens"], list)

    def test_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-m",
                "mlx_whisper.cli",
                TEST_AUDIO,
                "--model",
                MODEL_NAME,
                "--beam-size",
                "3",
                "--temperature",
                "0",
                "--language",
                "en",
                "--output-dir",
                tmpdir,
                "--output-format",
                "txt",
                "--verbose",
                "False",
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.listdir(tmpdir))

    def test_batch_decode_smoke(self):
        model = load_model(MODEL_NAME, mx.float16)
        data = audio.pad_or_trim(audio.load_audio(TEST_AUDIO))
        mel = audio.log_mel_spectrogram(data, model.dims.n_mels)
        batch = mx.stack([mel, mel])

        results = model.decode(
            batch,
            beam_size=2,
            temperature=0.0,
            without_timestamps=True,
            language="en",
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.text for result in results))

    def test_decode_returns_candidates_when_requested(self):
        model = load_model(MODEL_NAME, mx.float16)
        data = audio.pad_or_trim(audio.load_audio(TEST_AUDIO))
        mel = audio.log_mel_spectrogram(data, model.dims.n_mels)

        result = model.decode(
            mel,
            beam_size=2,
            temperature=0.0,
            without_timestamps=True,
            language="en",
            return_candidates=True,
        )

        self.assertTrue(result.candidates)
        self.assertTrue(any(candidate["selected"] for candidate in result.candidates))
        self.assertIn("score", result.candidates[0])
        self.assertIn("tokens", result.candidates[0])


@unittest.skipUnless(RUN_INTEGRATION, "set RUN_MLX_WHISPER_INTEGRATION=1")
class TestSharedCacheIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(MODEL_NAME, mx.float16)
        signal = audio.pad_or_trim(audio.load_audio(TEST_AUDIO))
        cls.mel = audio.log_mel_spectrogram(signal, cls.model.dims.n_mels)
        cls.silence = audio.log_mel_spectrogram(mx.zeros_like(signal), cls.model.dims.n_mels)

    def test_distinct_audio_batch_matches_expanded_reference(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        from mlx_whisper.decoding import BeamSearchDecoder, Inference
        from experiments.beam_search import baseline_logits
        from experiments.beam_reference import baseline_update, baseline_reorder
        import numpy as np
        batch = mx.stack([self.mel, self.silence])
        for sampling in (False, True):
            options = dict(language="en", sample_len=32, return_candidates=True,
                           without_timestamps=False)
            options.update(dict(temperature=0.4, best_of=3) if sampling
                           else dict(temperature=0., beam_size=3, patience=1.5))
            mx.random.seed(19)
            actual = self.model.decode(batch, **options)
            with ExitStack() as stack:
                stack.enter_context(patch.object(Inference, "logits", baseline_logits))
                stack.enter_context(patch.object(Inference, "rearrange_kv_cache", baseline_reorder))
                stack.enter_context(patch.object(BeamSearchDecoder, "update", baseline_update))
                mx.random.seed(19)
                expected = self.model.decode(batch, **options)
            for result, reference in zip(actual, expected):
                self.assertEqual(result.tokens, reference.tokens)
                self.assertEqual(result.audio_features.shape, reference.audio_features.shape)
                np.testing.assert_allclose(result.audio_features, reference.audio_features,
                                           atol=1e-4, rtol=1e-4)
                self.assertAlmostEqual(result.no_speech_prob, reference.no_speech_prob, places=5)
                self.assertEqual([c['tokens'] for c in result.candidates],
                                 [c['tokens'] for c in reference.candidates])
                np.testing.assert_allclose(
                    [c['score'] for c in result.candidates],
                    [c['score'] for c in reference.candidates], atol=1e-4, rtol=1e-4)

    def test_word_timestamps(self):
        result = mlx_whisper.transcribe(
            TEST_AUDIO, path_or_hf_repo=MODEL_NAME, beam_size=3,
            temperature=0., language="en", word_timestamps=True,
        )
        words = [w for segment in result['segments'] for w in segment['words']]
        self.assertTrue(words)
        for word in words:
            self.assertLessEqual(word['start'], word['end'])
            self.assertGreaterEqual(word['probability'], 0.)
            self.assertLessEqual(word['probability'], 1.)

    def test_forced_temperature_fallback_uses_sampling(self):
        from unittest.mock import patch
        from mlx_whisper.decoding import DecodingTask
        seen = []
        original_run = DecodingTask.run
        def record(task, mel):
            seen.append((task.options.temperature, task.options.beam_size,
                         task.options.best_of))
            return original_run(task, mel)
        with patch.object(DecodingTask, 'run', record):
            mx.random.seed(23)
            result = mlx_whisper.transcribe(
                TEST_AUDIO, path_or_hf_repo=MODEL_NAME, beam_size=3, best_of=2,
                temperature=(0., 0.4), logprob_threshold=1., no_speech_threshold=None,
                language='en', without_timestamps=True,
            )
        self.assertTrue(result['text'].strip())
        self.assertEqual(seen[:2], [(0., 3, None), (0.4, None, 2)])


if __name__ == "__main__":
    unittest.main()
