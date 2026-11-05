# Frozen beam decoder from 80e60509090601a58acf1e6d0d4c2e34b8e5a5de.
# Copyright © 2023 Apple Inc. See repository LICENSE.
from typing import Optional, Tuple
import mlx.core as mx
import numpy as np
from mlx_whisper.decoding import TokenDecoder, Inference

class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Inference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = logits.astype(mx.float32) - mx.logsumexp(
            logits.astype(mx.float32), axis=-1, keepdims=True
        )

        mx.eval(tokens, logprobs, sum_logprobs)
        tokens_np = np.array(tokens)
        logprobs_np = np.array(logprobs)
        sum_logprobs_np = np.array(sum_logprobs)

        next_tokens = []
        next_logprobs = []
        source_indices = []
        finished_sequences = []

        for i in range(n_audio):
            scores = {}
            sources = {}
            finished = {}

            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens_np[idx].tolist()
                row = logprobs_np[idx]
                top_indices = np.argsort(row)[-(self.beam_size + 1) :][::-1]
                for token in top_indices:
                    score = float(sum_logprobs_np[idx] + row[token])
                    sequence = tuple(prefix + [int(token)])
                    if sequence not in scores or score > scores[sequence]:
                        scores[sequence] = score
                        sources[sequence] = idx

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

        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            previously_finished.update(newly_finished)
            sorted_sequences = sorted(
                previously_finished.items(), key=lambda item: item[1], reverse=True
            )[: self.max_candidates]
            previously_finished.clear()
            previously_finished.update(sorted_sequences)

        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed, sum_logprobs

    def finalize(self, tokens: mx.array, sum_logprobs: mx.array):
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(tokens.shape[0])]

        mx.eval(tokens, sum_logprobs)
        tokens_np = np.array(tokens)
        sum_logprobs_np = np.array(sum_logprobs)

        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:
                for j in np.argsort(sum_logprobs_np[i])[::-1]:
                    sequence = tuple(tokens_np[i, j].tolist() + [self.eot])
                    sequences[sequence] = float(sum_logprobs_np[i, j])
                    if len(sequences) >= self.beam_size:
                        break

        n_candidates = max(len(sequences) for sequences in self.finished_sequences)
        max_length = max(
            len(sequence)
            for sequences in self.finished_sequences
            for sequence in sequences
        )
        padded_tokens = np.full(
            (len(self.finished_sequences), n_candidates, max_length),
            self.eot,
            dtype=tokens_np.dtype,
        )
        padded_logprobs = np.full(
            (len(self.finished_sequences), n_candidates),
            -np.inf,
            dtype=sum_logprobs_np.dtype,
        )

        for i, sequences in enumerate(self.finished_sequences):
            for j, sequence in enumerate(sequences.keys()):
                padded_tokens[i, j, : len(sequence)] = sequence
                padded_logprobs[i, j] = sequences[sequence]

        return mx.array(padded_tokens, dtype=tokens.dtype), mx.array(
            padded_logprobs, dtype=sum_logprobs.dtype
        )
