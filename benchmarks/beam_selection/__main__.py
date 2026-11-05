"""Run sequentially: python -m benchmarks.beam_selection --output /tmp/beam.json.
Public LibriSpeech audio and cached Hugging Face snapshots stay outside the repo.
"""
import argparse
import importlib.metadata
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import mlx.core as mx
import numpy as np
import mlx_whisper.decoding as decoding
from mlx_whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from mlx_whisper.transcribe import transcribe, ModelHolder
from benchmarks.beam_selection.baseline_decoding import BeamSearchDecoder as Baseline

Optimized = decoding.BeamSearchDecoder

class NoCache:
    def rearrange_kv_cache(self, indices):
        pass


def measure(fn, repeats):
    fn()  # warm-up, excluded
    times = []
    mx.reset_peak_memory()
    for _ in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        value = fn()
        mx.synchronize()
        times.append(time.perf_counter() - start)
    return dict(seconds=times, median=statistics.median(times), min=min(times),
                max=max(times), peak_mlx_bytes=mx.get_peak_memory()), value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--models', nargs='+', default=['tiny', 'large-v3-turbo'])
    parser.add_argument('--beams', nargs='+', type=int, default=[1, 3, 5, 10])
    parser.add_argument('--data', type=Path, default=Path('/tmp/mmlx-boost-validation/LibriSpeech'))
    args = parser.parse_args()
    result = dict(hardware=mx.device_info(), platform=platform.platform(), python=platform.python_version(),
                  versions={p: importlib.metadata.version(p) for p in ['mlx', 'numpy', 'mlx-whisper']},
                  baseline='80e60509090601a58acf1e6d0d4c2e34b8e5a5de',
                  optimized_sha256=hashlib.sha256(Path(decoding.__file__).read_bytes()).hexdigest(),
                  dtype='float16', selection=[], audio=[])
    def save():
        Path(args.output).write_text(json.dumps(result, indent=2))
    rng = np.random.default_rng(42)
    for batch in [1, 2]:
        for beam in args.beams:
            tokens = mx.array([[i // beam] * 64 for i in range(batch * beam)])
            logits = mx.array(rng.normal(size=(batch * beam, 51865)).astype(np.float32))
            scores = mx.zeros(batch * beam)
            mx.eval(tokens, logits, scores)
            record = dict(batch=batch, beam=beam, vocab=51865, prefix_length=64)
            for name, cls in [('baseline', Baseline), ('optimized', Optimized)]:
                decoder = cls(beam, 50257, NoCache())
                def run():
                    decoder.reset()
                    t, _, s = decoder.update(tokens, logits, scores)
                    mx.eval(t, s)
                record[name], _ = measure(run, 30)
            result['selection'].append(record)
            print('selection', batch, beam, record['baseline']['median'], record['optimized']['median'], flush=True)
            save()
    if args.selection_only:
        return
    paths = [args.data / 'test-clean/61/70970/61-70970-0001.flac',
             args.data / 'test-other/6070/86744/6070-86744-0007.flac']
    waves = [load_audio(str(p)) for p in paths]
    # Repeat the public pair to cross Whisper's 30-second window boundary.
    long_audio = np.concatenate(waves * 3)
    result['inputs'] = [{'id': str(p.relative_to(args.data)), 'seconds': len(w) / 16000, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p, w in zip(paths, waves)]
    result['long_seconds'] = len(long_audio) / 16000
    for name in args.models:
        snapshots = Path.home() / '.cache/huggingface/hub' / ('models--mlx-community--whisper-' + name) / 'snapshots'
        model_path = str(sorted(snapshots.iterdir())[-1])
        model = ModelHolder.get_model(model_path, mx.float16)
        mels = mx.stack([log_mel_spectrogram(pad_or_trim(w), n_mels=model.dims.n_mels) for w in waves])
        mx.eval(mels)
        for beam in args.beams:
            for case in ['short', 'long', 'batch', 'timestamps']:
                record = dict(model=name, revision=Path(model_path).name, beam=beam, case=case)
                outputs = {}
                for implementation, cls in [('baseline', Baseline), ('optimized', Optimized)]:
                    decoding.BeamSearchDecoder = cls
                    def run():
                        options = dict(beam_size=beam, temperature=0.0, language='en', return_candidates=True,
                                       without_timestamps=case != 'timestamps')
                        if case == 'batch':
                            decoded = model.decode(mels, **options)
                            return [dict(text=r.text, tokens=r.tokens, candidates=r.candidates) for r in decoded]
                        audio = long_audio if case == 'long' else waves[0]
                        out = transcribe(audio, path_or_hf_repo=model_path, condition_on_previous_text=False,
                                         compression_ratio_threshold=None, logprob_threshold=None, no_speech_threshold=None, **options)
                        return dict(text=out['text'], segments=[{k: v for k, v in s.items() if k in ['tokens', 'start', 'end', 'candidates']} for s in out['segments']])
                    record[implementation], outputs[implementation] = measure(run, args.repeats)
                record['exact_output_equal'] = outputs['baseline'] == outputs['optimized']
                record['outputs'] = outputs
                result['audio'].append(record)
                print(name, beam, case, record['baseline']['median'], record['optimized']['median'], record['exact_output_equal'], flush=True)
                save()
    decoding.BeamSearchDecoder = Optimized

if __name__ == '__main__':
    main()
