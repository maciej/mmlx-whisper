"""Compare production beam search with a frozen baseline and investigation variants.

Run from the repository root with uv run python experiments/beam_search.py --help.
Each invocation is one fresh-process model/beam/variant comparison cell.
"""
import argparse
import cProfile
import gc
import hashlib
import importlib
import io
import json
import platform
import pstats
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mlx_whisper import audio, decoding
from mlx_whisper.load_models import load_model
from mlx_whisper.whisper import MultiHeadAttention
from experiments.beam_reference import baseline_update, baseline_reorder

PRODUCTION_UPDATE = decoding.BeamSearchDecoder.update
PRODUCTION_REORDER = decoding.Inference.rearrange_kv_cache
PRODUCTION_LOGITS = decoding.Inference.logits
BASE_ATTENTION = MultiHeadAttention.__call__


def baseline_logits(self, tokens, audio_features):
    # The old DecodingTask repeated features before entering the loop. Recreate
    # that layout here so the frozen baseline keeps per-hypothesis cross K/V.
    group = tokens.shape[0] // audio_features.shape[0]
    if group > 1:
        audio_features = mx.broadcast_to(
            audio_features[:, None],
            (audio_features.shape[0], group, *audio_features.shape[1:]),
        ).reshape(tokens.shape[0], *audio_features.shape[1:])
    return PRODUCTION_LOGITS(self, tokens, audio_features)


def select_candidates(logprobs, k, method="sort"):
    k = min(k, logprobs.shape[-1])
    if method == "sort":
        # Stable: descending probability, then ascending token id.
        indices = mx.argsort(-logprobs, axis=-1)[:, :k]
    elif method == "partition":
        indices = mx.argpartition(-logprobs, k - 1, axis=-1)[:, :k]
        values = mx.take_along_axis(logprobs, indices, axis=-1)
        order = mx.argsort(-values, axis=-1)
        indices = mx.take_along_axis(indices, order, axis=-1)
    elif method == "reduce":
        # Repeated argmax is an exploratory small-k alternative, not a general
        # selector: all-negative-infinity tails can select an id twice.
        working = logprobs
        selected = []
        for _ in range(k):
            index = mx.argmax(working, axis=-1)
            selected.append(index)
            working = mx.put_along_axis(
                working, index[:, None], mx.array(-float("inf")), axis=-1
            )
        indices = mx.stack(selected, axis=-1)
    else:
        raise ValueError(method)
    return indices, mx.take_along_axis(logprobs, indices, axis=-1)


def compact_update(self, tokens, logits, sum_logprobs):
    """Same hypothesis bookkeeping as baseline; only selection is changed."""
    if tokens.shape[0] % self.beam_size:
        raise ValueError("Batch must be divisible by beam size")
    n_audio = tokens.shape[0] // self.beam_size
    if self.finished_sequences is None:
        self.finished_sequences = [{} for _ in range(n_audio)]
    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    indices, values = select_candidates(logprobs, self.beam_size + 1)
    mx.eval(tokens, indices, values, sum_logprobs)
    tokens_np = np.array(tokens)
    indices_np, values_np = np.array(indices), np.array(values)
    sums_np = np.array(sum_logprobs)
    next_tokens, next_logprobs, source_indices, finished_sequences = [], [], [], []
    for i in range(n_audio):
        scores, sources, finished = {}, {}, {}
        for j in range(self.beam_size):
            idx = i * self.beam_size + j
            prefix = tokens_np[idx].tolist()
            for token, value in zip(indices_np[idx], values_np[idx]):
                score = float(sums_np[idx] + value)
                sequence = tuple(prefix + [int(token)])
                if sequence not in scores or score > scores[sequence]:
                    scores[sequence], sources[sequence] = score, idx
        saved = 0
        for sequence in sorted(scores, key=scores.get, reverse=True):
            if sequence[-1] == self.eot:
                finished[sequence] = scores[sequence]
            else:
                next_tokens.append(sequence)
                next_logprobs.append(scores[sequence])
                source_indices.append(sources[sequence])
                saved += 1
                if saved == self.beam_size:
                    break
        finished_sequences.append(finished)
    tokens = mx.array(next_tokens, dtype=tokens.dtype)
    sum_logprobs = mx.array(next_logprobs, dtype=sum_logprobs.dtype)
    self.inference.rearrange_kv_cache(source_indices)
    for old, new in zip(self.finished_sequences, finished_sequences):
        old.update(new)
        ordered = sorted(old.items(), key=lambda item: item[1], reverse=True)[:self.max_candidates]
        old.clear()
        old.update(ordered)
    completed = all(len(s) >= self.max_candidates for s in self.finished_sequences)
    return tokens, completed, sum_logprobs


def self_cache_reorder(self, source_indices):
    # Cross-attention entries are identical within an audio group. Beam search
    # never moves parents across audio groups. Preserve these arrays verbatim.
    if source_indices != list(range(len(source_indices))):
        self.kv_cache = [
            (tree_map(lambda x: x[source_indices], self_kv), cross_kv)
            for self_kv, cross_kv in self.kv_cache
        ]


def install(variant, beam):
    decoding.BeamSearchDecoder.update = PRODUCTION_UPDATE
    decoding.Inference.rearrange_kv_cache = PRODUCTION_REORDER
    decoding.Inference.logits = PRODUCTION_LOGITS
    MultiHeadAttention.__call__ = BASE_ATTENTION
    if variant == "production":
        return
    decoding.BeamSearchDecoder.update = baseline_update
    decoding.Inference.rearrange_kv_cache = baseline_reorder
    decoding.Inference.logits = baseline_logits
    if variant != "baseline":
        decoding.BeamSearchDecoder.update = compact_update
    if variant in ("cache", "shared"):
        decoding.Inference.rearrange_kv_cache = self_cache_reorder
    if variant == "shared":
        def shared_attention(self, x, xa=None, mask=None, kv_cache=None):
            if xa is not None and kv_cache is None:
                # Deliberately single-audio prototype. Do not silently broadcast
                # audio 0 across an unrelated batch item.
                if xa.shape[0] != beam:
                    raise ValueError("shared prototype supports one audio item only")
                kv_cache = (self.key(xa[:1]), self.value(xa[:1]))
            return BASE_ATTENTION(self, x, xa, mask, kv_cache)
        MultiHeadAttention.__call__ = shared_attention


def measure(fn, warmups, repeats):
    for _ in range(warmups):
        value = fn()
        mx.synchronize()
        del value
    samples, peaks, caches, signatures = [], [], [], []
    for _ in range(repeats):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        start = time.perf_counter()
        value = fn()
        mx.synchronize()
        samples.append(time.perf_counter() - start)
        peaks.append(mx.get_peak_memory())
        caches.append(mx.get_cache_memory())
        if isinstance(value, dict):
            signature = {k: value[k] for k in ("text", "segments")}
        else:
            signature = {"text": value.text, "tokens": value.tokens,
                         "avg_logprob": value.avg_logprob,
                         "candidates": value.candidates}
        signatures.append(signature)
        del value
    return {"seconds": samples, "median_seconds": statistics.median(samples),
            "peak_active_bytes": peaks, "cache_bytes_after": caches,
            "outputs": signatures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local snapshot or HF model id")
    p.add_argument("--audio", default="mlx_whisper/assets/ls_test.flac")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--variant", choices=("production", "baseline", "compact", "cache", "shared"), default="production")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timestamps", action="store_true")
    p.add_argument("--profile", action="store_true", help="Profile one additional decoder call")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.beam < 1 or a.repeats < 1 or a.warmups < 0:
        p.error("beam and repeats must be positive; warmups must be nonnegative")
    install(a.variant, a.beam)
    model = load_model(a.model, mx.float16)
    signal = audio.load_audio(a.audio)
    mel = audio.log_mel_spectrogram(audio.pad_or_trim(signal), model.dims.n_mels)
    features = model.encoder(mel[None].astype(mx.float16))
    mx.eval(features)
    options = dict(beam_size=a.beam, temperature=0.0, language="en", fp16=True,
                   without_timestamps=not a.timestamps, return_candidates=True)
    # Make transcribe reuse exactly this model, excluding download/load time.
    tr = importlib.import_module("mlx_whisper.transcribe")
    tr.ModelHolder.model, tr.ModelHolder.model_path = model, a.model
    result = {"config": vars(a), "mlx": mx.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "device": mx.device_info(),
              "audio_sha256": hashlib.sha256(Path(a.audio).read_bytes()).hexdigest(),
              "audio_seconds": len(signal)/audio.SAMPLE_RATE,
              "model_dims": vars(model.dims)}
    result["decoder"] = measure(lambda: model.decode(features[0], **options), a.warmups, a.repeats)
    result["transcribe"] = measure(lambda: tr.transcribe(a.audio, path_or_hf_repo=a.model,
                                     verbose=None, **options), a.warmups, a.repeats)
    if a.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        model.decode(features[0], **options)
        mx.synchronize()
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("tottime").print_stats(20)
        result["decoder_profile"] = stream.getvalue()
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"variant": a.variant, "beam": a.beam, "model": a.model,
                      "decoder": result["decoder"]["median_seconds"],
                      "transcribe": result["transcribe"]["median_seconds"],
                      "peak_MB": max(result["transcribe"]["peak_active_bytes"])/1e6}), flush=True)


if __name__ == "__main__":
    main()
