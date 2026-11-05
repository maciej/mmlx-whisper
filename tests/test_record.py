"""Recorder tests without microphone or macOS keyboard access.

Run with: uv run pytest -q tests/test_record.py
"""

import importlib.util
import io
import json
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def recorder(monkeypatch):
    # The standalone script owns its UI/audio dependencies, not the project.
    console = SimpleNamespace(
        print=lambda *args, **kwargs: None,
        status=lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "rich", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "rich.console", SimpleNamespace(Console=lambda: console)
    )
    spec = importlib.util.spec_from_file_location(
        "recorder", Path(__file__).resolve().parents[1] / "tools" / "record.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "terminal_mode", nullcontext)
    return module


@pytest.mark.parametrize(
    "payload",
    [
        [" Hello. ", "World."],
        {"utterances": [{"text": " Hello. "}, {"text": "World."}]},
    ],
)
def test_prompt_formats(recorder, tmp_path, payload):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(payload))
    assert recorder.load_prompts(path) == ["Hello.", "World."]


@pytest.mark.parametrize("payload", [[], {}, [" "], [None], {"utterances": "text"}])
def test_invalid_prompts(recorder, tmp_path, payload):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        recorder.load_prompts(path)


def test_preview_without_audio_backend(recorder, monkeypatch, tmp_path, capsys):
    source = tmp_path / "prompts.json"
    source.write_text('["Say [hello]."]')
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(
        sys, "argv", ["record.py", "--script", str(source), "--preview"]
    )
    recorder.main()
    assert "Recording 1: Say [hello]." in capsys.readouterr().out
    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize("frames", [0, 100, 3200])
def test_capture_pcm_and_stream_cleanup(recorder, monkeypatch, frames):
    state = {}

    class Stream:
        def __init__(self, **kwargs):
            state.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] = True

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=Stream))
    monkeypatch.setattr(sys, "stdin", io.StringIO(" "))

    def space_pressed():
        if frames:
            state["callback"](
                np.full((frames, 1), 0.5, dtype=np.float32), frames, None, "overflow"
            )
        return False

    if frames < 1600:
        with pytest.raises(RuntimeError, match="No audio|Less than"):
            recorder.capture(recorder.Console(), space_pressed, 7)
        assert state["closed"]
        return

    action, (data, seconds, peak), warnings = recorder.capture(
        recorder.Console(), space_pressed, 7
    )
    assert action == "record"
    assert state["closed"] and state["device"] == 7
    assert seconds == 0.2 and peak == 0.5
    assert warnings == ["overflow"]
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1,
            2,
            16000,
        )
        assert wav.getnframes() == 3200
        assert np.frombuffer(wav.readframes(3200), dtype="<i2")[0] == 16383


@pytest.mark.parametrize("key", ["r", "q"])
def test_finished_controls_need_no_microphone(recorder, monkeypatch, key):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(sys, "stdin", io.StringIO(key.upper()))
    assert recorder.capture(None, None, None, allow_record=False)[0] == key


def test_closed_terminal(recorder, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(EOFError):
        recorder.capture(None, None, None, allow_record=False)


@pytest.mark.parametrize(
    "ending", ["q", KeyboardInterrupt(), "audio_error", "complete"]
)
def test_session_redo_and_failure_retains_takes(
    recorder, monkeypatch, tmp_path, ending
):
    class AudioError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(PortAudioError=AudioError)
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(recorder, "space_monitor", lambda: None)
    source = tmp_path / "prompts.json"
    source.write_text('["First.", "Second."]')
    output = tmp_path / "session"
    monkeypatch.setattr(
        sys, "argv", ["record.py", "--script", str(source), "--output-dir", str(output)]
    )
    actions = iter(
        ["record", "r", "record"]
        + (["record", "q"] if ending == "complete" else [ending])
    )

    def capture(*args):
        action = next(actions)
        if isinstance(action, BaseException):
            raise action
        if action == "audio_error":
            raise AudioError("device disconnected")
        return action, (b"audio", 0.2, 0.5), []

    monkeypatch.setattr(recorder, "capture", capture)
    result = recorder.main()
    report = json.loads((output / "session.json").read_text())
    assert report["status"] == ("complete" if ending == "complete" else "partial")
    assert len(report["attempts"]) == (3 if ending == "complete" else 2)
    assert report["recordings"]["1"]["file"] == "01-take-02.wav"
    assert (output / "01-take-01.wav").read_bytes() == b"audio"
    assert result == (1 if ending == "audio_error" else None)
