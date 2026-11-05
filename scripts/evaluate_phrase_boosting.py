#!/usr/bin/env python3
"""Reproducible LibriSpeech diagnostic; see docs/phrase-boosting.md.

Use an extracted OpenSLR test-clean/test-other directory. No downloads occur here.
Relevant terms are reference-derived (oracle context), not a deployment-quality claim.
"""

import argparse
import hashlib
import json
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from mlx_whisper import PhraseBoostingConfig, transcribe
from mlx_whisper.audio import load_audio
from huggingface_hub import snapshot_download


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def aligned_errors(reference, hypothesis, targets):
    """Levenshtein errors and target-position errors, with deterministic ties.

    Target errors count substitutions/deletions at target reference positions and
    insertions of target words. Ties prefer diagonal, deletion, then insertion.
    """
    matrix = [list(range(len(hypothesis) + 1))]
    for i, ref in enumerate(reference, 1):
        row = [i]
        for j, hyp in enumerate(hypothesis, 1):
            row.append(
                min(row[-1] + 1, matrix[-1][j] + 1, matrix[-1][j - 1] + (ref != hyp))
            )
        matrix.append(row)
    i, j, biased = len(reference), len(hypothesis), 0
    while i or j:
        if (
            i
            and j
            and matrix[i][j]
            == matrix[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
        ):
            biased += int(
                reference[i - 1] in targets and reference[i - 1] != hypothesis[j - 1]
            )
            i, j = i - 1, j - 1
        elif i and matrix[i][j] == matrix[i - 1][j] + 1:
            biased += int(reference[i - 1] in targets)
            i -= 1
        else:
            biased += int(hypothesis[j - 1] in targets)
            j -= 1
    return matrix[-1][-1], biased


def samples(root, limit):
    speakers = defaultdict(list)
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            identity, text = line.split(" ", 1)
            path = transcript.parent / (identity + ".flac")
            if path.exists():
                speakers[identity.split("-")[0]].append((identity, text, path))
    # Stable round-robin across speakers, with hashed utterance ordering.
    queues = [
        sorted(values, key=lambda x: hashlib.sha256(x[0].encode()).hexdigest())
        for _, values in sorted(speakers.items())
    ]
    selected = []
    for offset in range(max(map(len, queues), default=0)):
        for queue in queues:
            if offset < len(queue):
                selected.append(queue[offset])
                if len(selected) == limit:
                    return selected
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", default="mlx-community/whisper-tiny")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0 or any(b < 1 for b in args.beams):
        parser.error("limit and beam sizes must be positive")
    chosen = samples(args.corpus, args.limit)
    if not chosen:
        parser.error("no LibriSpeech transcripts/audio found")
    config = PhraseBoostingConfig(strength=args.strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = args.output.with_suffix(".jsonl")
    metadata = {
        "model": args.model,
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx": mx.__version__,
        "device": mx.metal.device_info(),
        "sample_ids": [x[0] for x in chosen],
        "context": "reference-derived oracle; no tuning",
        "timing": "warm transcribe from preloaded PCM; includes mel/encoder/decoder, excludes file I/O",
        "normalization": "lowercase ASCII words, internal apostrophes retained",
        "options": {
            "language": "en",
            "temperature": 0,
            "without_timestamps": True,
            "condition_on_previous_text": False,
            "fallback": False,
        },
    }
    # Warm model and kernels; exclude this from measured runs.
    first_audio = np.array(load_audio(str(chosen[0][2])))
    transcribe(
        first_audio,
        path_or_hf_repo=args.model,
        language="en",
        temperature=0.0,
        without_timestamps=True,
        verbose=None,
    )
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path(snapshot_download(args.model, local_files_only=True))
    metadata["model_snapshot"] = model_path.name
    metadata["model_config_sha256"] = hashlib.sha256(
        (model_path / "config.json").read_bytes()
    ).hexdigest()
    metadata["corpus_split"] = args.corpus.name
    records = []
    with rows_path.open("w") as handle:
        for beam in args.beams:
            for index, (identity, reference, path) in enumerate(chosen):
                audio = np.array(load_audio(str(path)))
                ref_words = words(reference)
                # Frozen selection rule: three longest distinct words >=6 letters.
                terms = sorted(
                    {w for w in ref_words if len(w) >= 6}, key=lambda w: (-len(w), w)
                )[:3]
                distractors = [
                    w
                    for w in [
                        "xylophone",
                        "quasar",
                        "microprocessor",
                        "kubernetes",
                        "photosynthesis",
                    ]
                    if w not in ref_words
                ]
                variants = [
                    ("baseline", []),
                    ("relevant", terms),
                    ("distractor", distractors),
                ]
                # Rotate order to reduce systematic timing-order bias.
                variants = variants[index % 3 :] + variants[: index % 3]
                for mode, phrases in variants:
                    forms = [
                        form for phrase in phrases for form in (phrase, phrase.title())
                    ]
                    mx.reset_peak_memory()
                    started = time.perf_counter()
                    result = transcribe(
                        audio,
                        path_or_hf_repo=args.model,
                        language="en",
                        temperature=0.0,
                        beam_size=beam if beam > 1 else None,
                        phrases=forms,
                        phrase_boosting=config,
                        without_timestamps=True,
                        condition_on_previous_text=False,
                        compression_ratio_threshold=None,
                        logprob_threshold=None,
                        no_speech_threshold=None,
                        verbose=None,
                    )
                    mx.synchronize()
                    elapsed = time.perf_counter() - started
                    hyp_words = words(result["text"])
                    ref_counts, hyp_counts = Counter(ref_words), Counter(hyp_words)
                    # Score relevant targets identically for all modes, plus distractor emissions.
                    target_total = sum(ref_counts[t] for t in terms)
                    tp = sum(min(ref_counts[t], hyp_counts[t]) for t in terms)
                    emitted = sum(hyp_counts[t] for t in terms)
                    word_errors, biased_errors = aligned_errors(
                        ref_words, hyp_words, set(terms)
                    )
                    row = {
                        "id": identity,
                        "beam": beam,
                        "mode": mode,
                        "phrases": phrases,
                        "reference": reference,
                        "text": result["text"],
                        "seconds": elapsed,
                        "audio_seconds": len(audio) / 16000,
                        "peak_mlx_bytes": mx.get_peak_memory(),
                        "word_errors": word_errors,
                        "biased_word_errors": biased_errors,
                        "reference_words": len(ref_words),
                        "target_total": target_total,
                        "target_correct": tp,
                        "target_emitted": emitted,
                        "distractor_insertions": sum(
                            hyp_counts[t] for t in distractors
                        ),
                    }
                    records.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                print(
                    "beam=%d sample=%d/%d %s"
                    % (beam, index + 1, len(chosen), identity),
                    flush=True,
                )
    summary = []
    for beam in args.beams:
        for mode in ("baseline", "relevant", "distractor"):
            group = [r for r in records if r["beam"] == beam and r["mode"] == mode]
            total = lambda key: sum(r[key] for r in group)
            emitted = total("target_emitted")
            targets = total("target_total")
            summary.append(
                {
                    "beam": beam,
                    "mode": mode,
                    "samples": len(group),
                    "wer": total("word_errors") / total("reference_words"),
                    "bwer": total("biased_word_errors") / targets if targets else None,
                    "target_recall": total("target_correct") / targets
                    if targets
                    else None,
                    "target_precision": total("target_correct") / emitted
                    if emitted
                    else None,
                    "distractor_insertions": total("distractor_insertions"),
                    "median_seconds": statistics.median(r["seconds"] for r in group),
                    "rtf": total("seconds") / total("audio_seconds"),
                    "peak_mlx_bytes": max(r["peak_mlx_bytes"] for r in group),
                }
            )
    args.output.write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
