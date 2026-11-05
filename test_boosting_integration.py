"""Opt-in real-audio checks using the bundled LibriSpeech clip and an MLX model."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_whisper import PhraseBoostingConfig, transcribe
from mlx_whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from mlx_whisper.decoding import DecodingOptions, DecodingTask
from mlx_whisper.load_models import load_model

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MLX_WHISPER_INTEGRATION") != "1", reason="opt-in model integration"
)
MODEL = os.getenv("MLX_WHISPER_TEST_MODEL", "mlx-community/whisper-tiny")
AUDIO = str(Path(__file__).parent / "mlx_whisper/assets/ls_test.flac")


@pytest.fixture(scope="module")
def model_and_mel():
    model = load_model(MODEL, mx.float16)
    mel = log_mel_spectrogram(pad_or_trim(load_audio(AUDIO)), n_mels=model.dims.n_mels)
    return model, mel


@pytest.mark.parametrize("beam", [None, 3])
def test_noop_equivalence_and_score_accounting(model_and_mel, beam):
    model, mel = model_and_mel
    options = DecodingOptions(
        language="en", beam_size=beam, without_timestamps=True, return_candidates=True
    )
    baseline = model.decode(mel, options)
    for overrides in (
        {"phrases": []},
        {"phrases": ["unused"], "phrase_boosting": PhraseBoostingConfig(strength=0)},
    ):
        result = model.decode(mel, replace(options, **overrides))
        assert result.tokens == baseline.tokens
        assert result.avg_logprob == baseline.avg_logprob
        assert result.candidates == baseline.candidates
        assert result.boost_score is None
    boosted = model.decode(
        mel, replace(options, phrases=["people", "bread", "something"])
    )
    assert boosted.text.strip()
    assert boosted.boost_score is not None
    assert boosted.no_speech_prob == baseline.no_speech_prob
    chosen = next(c for c in boosted.candidates if c["selected"])
    assert chosen["sum_logprob"] + chosen["boost_score"] == pytest.approx(
        chosen["search_score"], abs=1e-5
    )
    assert boosted.avg_logprob == chosen["avg_logprob"]
    assert boosted.search_score == chosen["search_score"]


def test_batch_and_reset(model_and_mel):
    model, mel = model_and_mel
    options = DecodingOptions(
        language="en", beam_size=2, without_timestamps=True, phrases=["people"]
    )
    task = DecodingTask(model, options)
    first = task.run(mx.stack([mel, mel]))
    second = task.run(mx.stack([mel, mel]))
    assert first[0].tokens == first[1].tokens == second[0].tokens
    assert first[0].boost_score == second[0].boost_score


def test_timestamps_word_alignment_and_window_boundaries():
    audio = np.array(load_audio(AUDIO))
    # More than 30 seconds forces transcribe's independent-window lifecycle.
    audio = np.tile(audio, 6)
    assert len(audio) / 16000 > 30
    result = transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language="en",
        beam_size=2,
        temperature=0.0,
        phrases=["people"],
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    assert result["text"].strip()
    assert result["segments"]
    for segment in result["segments"]:
        assert segment["start"] <= segment["end"]
        assert np.isfinite(segment["avg_logprob"])
        assert "boost_score" in segment
        for word in segment.get("words", []):
            assert word["start"] <= word["end"]


def test_sampling_fallback_resets_state():
    result = transcribe(
        AUDIO,
        path_or_hf_repo=MODEL,
        language="en",
        beam_size=2,
        temperature=(0.0, 0.2),
        logprob_threshold=1.0,
        no_speech_threshold=None,
        phrases=["people"],
        best_of=2,
        without_timestamps=True,
    )
    assert result["segments"]
    assert all(s["temperature"] == 0.2 for s in result["segments"])
    assert all(np.isfinite(s["boost_score"]) for s in result["segments"])


def test_cli_phrase_file(tmp_path):
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("people\nbread\n", encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_whisper.cli",
            AUDIO,
            "--model",
            MODEL,
            "--phrases-file",
            str(phrases),
            "--beam-size",
            "2",
            "--language",
            "en",
            "--output-dir",
            str(tmp_path),
            "--output-format",
            "json",
            "--verbose",
            "False",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    output = json.loads((tmp_path / "ls_test.json").read_text())
    assert output["segments"] and "boost_score" in output["segments"][0]


@pytest.mark.parametrize("noise", [False, True])
def test_silence_and_low_level_noise_do_not_insert_phrases(noise):
    audio = np.zeros(8 * 16000, dtype=np.float32)
    if noise:
        audio = (
            np.random.default_rng(42).normal(0, 0.001, audio.shape).astype(np.float32)
        )
    common = dict(path_or_hf_repo=MODEL, language="en", temperature=0.0, beam_size=3)
    baseline = transcribe(audio, **common)
    boosted = transcribe(audio, phrases=["xylophone", "quasar"], **common)
    assert boosted["text"] == baseline["text"]
    assert "xylophone" not in boosted["text"].lower()
    assert "quasar" not in boosted["text"].lower()
