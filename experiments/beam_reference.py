# Copyright © 2023-2024 Apple Inc.
# Frozen beam baseline from 8b6994f, for differential tests and benchmarks only.
from typing import Tuple
import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

def baseline_update(
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

def baseline_reorder(self, source_indices):
    """Update the key-value cache according to the updated beams"""
    # update the key/value cache to contain the selected sequences
    if source_indices != list(range(len(source_indices))):
        self.kv_cache = tree_map(lambda x: x[source_indices], self.kv_cache)
