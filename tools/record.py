#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "rich>=13", "sounddevice>=0.5"]
# ///
"""Local macOS push-to-talk recorder. No model or transcription server required.

Usage:
    uv run tools/record.py --list-devices
    uv run tools/record.py --script /path/outside/repo/prompts.json

Prompt JSON: {"utterances": [{"text": "Your sentence."}, ...]}.
A JSON list of strings is also accepted. Prompts and recordings are user data;
keep them outside the repository. Output defaults to a timestamped directory
under <prompt-file-directory>/recordings/. All takes and a session manifest are
saved locally as 16 kHz mono PCM WAV; nothing is uploaded.
"""

import argparse
import ctypes
import io
import json
import sys
import time
import wave
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Event

import numpy as np
from rich.console import Console

SAMPLE_RATE = 16_000


def load_prompts(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("utterances") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError(
            "Prompt JSON must contain a nonempty utterances list or be a list of strings."
        )
    prompts = []
    for index, item in enumerate(items, 1):
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prompt {index} must contain nonempty text.")
        prompts.append(text.strip())
    return prompts


@contextmanager
def terminal_mode():
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def read_key():
    key = sys.stdin.read(1)
    if not key:
        raise EOFError("Terminal input closed")
    return key.casefold()


def space_monitor():
    # Physical key state detects release, unlike terminal key-repeat characters.
    graphics = ctypes.CDLL(
        "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
    )
    key_state = graphics.CGEventSourceKeyState
    key_state.argtypes = [ctypes.c_int32, ctypes.c_uint16]
    key_state.restype = ctypes.c_bool
    return lambda: bool(key_state(0, 49))  # macOS Space key code


def capture(console, space_pressed, device, allow_record=True):
    recording = Event()
    chunks, warnings = [], []

    def callback(indata, _frames, _clock, status):
        if status:
            warnings.append(str(status))
        if recording.is_set():
            chunks.append(indata[:, 0].copy())

    with terminal_mode():
        if not allow_record:
            while True:
                key = read_key()
                if key in ("r", "q"):
                    return key, None, warnings
        import sounddevice as sd

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        ):
            while True:
                key = read_key()
                if key in ("r", "q"):
                    return key, None, warnings
                if key == " ":
                    break
            recording.set()
            try:
                with console.status("[bold red]Recording — release Space to stop[/]"):
                    while space_pressed():
                        time.sleep(0.01)
            finally:
                recording.clear()
    if not chunks:
        raise RuntimeError(
            "No audio captured. Hold Space while speaking; check terminal microphone/keyboard input permissions if needed."
        )
    audio = np.concatenate(chunks)
    if len(audio) < SAMPLE_RATE / 10:
        raise RuntimeError(
            "Less than 0.1 seconds captured. Hold Space longer, then release it."
        )
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        data = buffer.getvalue()
    return (
        "record",
        (data, len(audio) / SAMPLE_RATE, float(np.max(np.abs(audio)))),
        warnings,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--script", type=Path, help="External prompt JSON file")
    parser.add_argument(
        "--device", type=int, help="Audio input index (default: system microphone)"
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, help="New session directory; must not exist"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print prompts without accessing the microphone",
    )
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if args.script is None:
        parser.error("--script is required unless --list-devices is used")
    source = args.script.expanduser().resolve()
    try:
        prompts = load_prompts(source)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.preview:
        for index, prompt in enumerate(prompts, 1):
            print(f"Recording {index}: {prompt}")
        return
    if sys.platform != "darwin" or not sys.stdin.isatty():
        parser.error("Hold-to-record requires an interactive macOS terminal.")
    import sounddevice as sd

    space_pressed = space_monitor()
    console = Console()
    output = (
        (
            args.output_dir
            or source.parent
            / "recordings"
            / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
        .expanduser()
        .resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "source": str(source),
        "sample_rate": SAMPLE_RATE,
        "prompts": prompts,
        "attempts": [],
        "recordings": {},
        "status": "recording",
    }

    def save():
        temporary = output / "session.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "session.json")

    save()
    console.print(
        "[bold]Hold Space to record; release to stop. R = redo previous. Q = finish.[/]"
    )
    console.print("Saved locally to: " + str(output), markup=False)
    index = 0
    try:
        while True:
            if index < len(prompts):
                console.print(f"\n[bold]Recording {index + 1} / {len(prompts)}[/]")
                console.print(prompts[index], style="italic", markup=False)
                console.print("[dim]Ready — hold Space while speaking.[/]")
            else:
                console.print(
                    "\n[bold]All recordings saved.[/] R = redo last recording. Q = finish."
                )
            try:
                action, result, warnings = capture(
                    console, space_pressed, args.device, index < len(prompts)
                )
            except sd.PortAudioError as exc:
                console.print(str(exc), style="red", markup=False)
                console.print(
                    "Check --list-devices, --device, and microphone permission, then restart."
                )
                return 1
            except RuntimeError as exc:
                console.print(str(exc), style="red", markup=False)
                console.print(
                    "Retry, or Ctrl+C to finish. Check microphone permission if needed."
                )
                continue
            if action == "q":
                break
            if action == "r":
                index = max(0, index - 1)
                continue
            data, seconds, peak = result
            take = 1 + sum(a["recording"] == index + 1 for a in report["attempts"])
            filename = f"{index + 1:02d}-take-{take:02d}.wav"
            (output / filename).write_bytes(data)
            attempt = {
                "recording": index + 1,
                "reference": prompts[index],
                "file": filename,
                "seconds": seconds,
                "peak": peak,
                "audio_warnings": warnings,
            }
            report["attempts"].append(attempt)
            report["recordings"][str(index + 1)] = attempt
            save()
            console.print(f"Saved {seconds:.2f}s → {filename}", markup=False)
            if peak < 0.001:
                console.print(
                    "[yellow]Very quiet recording; check the microphone or press R to redo.[/]"
                )
            index += 1
    except (KeyboardInterrupt, EOFError):
        console.print("\nStopped; recorded takes are retained.")
    finally:
        report["status"] = (
            "complete" if len(report["recordings"]) == len(prompts) else "partial"
        )
        save()
    console.print("\nSession saved: " + str(output), markup=False)


if __name__ == "__main__":
    sys.exit(main())
