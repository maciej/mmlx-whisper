"""TurboBias phrase boosting on MLX (MIT license).

Algorithm: https://arxiv.org/abs/2508.07014 . No model training is required.
The Python automaton is also an independent oracle for the MLX implementation.
"""

import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import mlx.core as mx


@dataclass(frozen=True)
class PhraseBoostingConfig:
    context_score: float = 1.0
    depth_scale: float = 1.0
    unknown_score: float = 0.0
    strength: float = 0.1
    final_eos_score: float = 1.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError("%s must be finite" % name)
            if name != "unknown_score" and value < 0:
                raise ValueError("%s must be non-negative" % name)

    def reward(self, depth):
        return (
            self.context_score
            if depth == 1
            else (self.context_score * self.depth_scale + math.log(depth))
        )


class PhraseTree:
    """Sparse token automaton; immutable after construction, safe to cache/share."""

    def __init__(
        self,
        sequences: Iterable[Sequence[int]],
        vocab_size: int,
        config: PhraseBoostingConfig = PhraseBoostingConfig(),
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.config = config
        edges, depth, total, final = [{}], [0], [0.0], [False]
        for sequence in sequences:
            state = 0
            for token in sequence:
                if not isinstance(token, int) or not 0 <= token < vocab_size:
                    raise ValueError("phrase token outside vocabulary")
                if token not in edges[state]:
                    edges[state][token] = len(edges)
                    edges.append({})
                    depth.append(depth[state] + 1)
                    total.append(total[state] + config.reward(depth[-1]))
                    final.append(False)
                state = edges[state][token]
            if state:
                final[state] = True
        failure = [0] * len(edges)
        queue = deque(edges[0].values())
        while queue:
            state = queue.popleft()
            for token, child in edges[state].items():
                fallback = failure[state]
                while fallback and token not in edges[fallback]:
                    fallback = failure[fallback]
                failure[child] = edges[fallback].get(token, 0)
                queue.append(child)
        self.edges = tuple(edges)
        self.depth = tuple(depth)
        self.failure = tuple(failure)
        self.final = tuple(final)
        self.backoff = tuple(
            0.0 if final[s] else total[failure[s]] - total[s] for s in range(len(edges))
        )
        self.max_depth = max(depth)

    def transition(self, state, token):
        """Unscaled score and destination for a single ordinary token (CPU oracle)."""
        score = 0.0
        while token not in self.edges[state]:
            if state == 0:
                return score + self.config.unknown_score, 0
            score += self.backoff[state]
            state = self.failure[state]
        child = self.edges[state][token]
        return score + self.config.reward(self.depth[child]), child

    def reference(self, states, eot, ignored=()):
        scores, destinations = [], []
        for state in states:
            row = [self.transition(state, t) for t in range(self.vocab_size)]
            rewards, targets = map(list, zip(*row))
            for token in ignored:
                rewards[token], targets[token] = 0.0, state
            rewards[eot] = max(0.0, max(rewards)) + (
                self.config.final_eos_score if self.final[state] else 0.0
            )
            targets[eot] = state
            scores.append([r * self.config.strength for r in rewards])
            destinations.append(targets)
        return scores, destinations


class PhraseScorer:
    """Device-local batched queries. Arrays scale with arcs and batch × vocabulary.

    All scores are float32. No host reads or loops over vocabulary occur in advance().
    Mutable decoding state belongs to the decoder, never to this scorer.
    """

    def __init__(self, tree, eot, ignored=()):
        self.tree = tree
        self.eot = eot
        self.ignored = tuple(t for t in ignored if t != eot)
        offsets, labels, targets, rewards = [0], [], [], []
        for edges in tree.edges:
            for token, child in sorted(edges.items()):
                labels.append(token)
                targets.append(child)
                rewards.append(tree.config.reward(tree.depth[child]))
            offsets.append(len(labels))
        self.offsets = mx.array(offsets, mx.int32)
        # A sentinel arc makes empty trees and padded gathers safe.
        self.labels = mx.array(labels + [tree.vocab_size], mx.int32)
        self.targets = mx.array(targets + [-1], mx.int32)
        self.rewards = mx.array(rewards + [0.0], mx.float32)
        self.arc_count = len(labels)
        self.width = max(map(len, tree.edges))
        self.failure = mx.array(tree.failure, mx.int32)
        self.backoff = mx.array(tree.backoff, mx.float32)
        self.final = mx.array(tree.final)

    def advance(self, states):
        batch, vocab = states.shape[0], self.tree.vocab_size
        scores = mx.zeros((batch, vocab), mx.float32)
        targets = mx.full((batch, vocab), -1, mx.int32)
        current = states
        debit = mx.zeros(batch, mx.float32)
        active = mx.ones(batch, mx.bool_)
        rows = mx.arange(batch)[:, None]
        for _ in range(self.tree.max_depth + 1):
            if not self.width:
                break
            arcs = self.offsets[current, None] + mx.arange(self.width)[None, :]
            valid = active[:, None] & (arcs < self.offsets[current + 1, None])
            arcs = mx.where(valid, arcs, self.arc_count)
            labels = self.labels[arcs]
            # Invalid arcs collide only in the discarded sentinel column.
            local_scores = mx.zeros((batch, vocab + 1), mx.float32)
            local_states = mx.full((batch, vocab + 1), -1, mx.int32)
            local_scores = local_scores.at[rows, labels].add(self.rewards[arcs])
            local_states = local_states.at[rows, labels].maximum(self.targets[arcs])
            use = (targets < 0) & (local_states[:, :vocab] >= 0)
            scores = mx.where(use, debit[:, None] + local_scores[:, :vocab], scores)
            targets = mx.where(use, local_states[:, :vocab], targets)
            active = active & (current != 0)
            debit = debit + mx.where(active, self.backoff[current], 0.0)
            current = mx.where(active, self.failure[current], current)
        unresolved = targets < 0
        scores = mx.where(
            unresolved, debit[:, None] + self.tree.config.unknown_score, scores
        )
        targets = mx.where(unresolved, 0, targets)
        if self.ignored:
            ids = mx.array(self.ignored, mx.int32)
            scores[:, ids] = 0.0
            targets[:, ids] = states[:, None]
        eos = mx.maximum(mx.max(scores, axis=-1), 0.0)
        eos = eos + self.final[states] * self.tree.config.final_eos_score
        scores[:, self.eot] = eos
        targets[:, self.eot] = states
        return scores * self.tree.config.strength, targets


@lru_cache(maxsize=8)
def _cached_tree(sequences, vocab_size, config):
    # Cache only Python data, not device arrays or request state.
    return PhraseTree(sequences, vocab_size, config)


def make_phrase_scorer(tokenizer, phrases, vocab_size, config):
    if isinstance(phrases, str):
        raise TypeError("phrases must be a sequence of strings, not a string")
    if len(phrases) > 20000:
        raise ValueError("at most 20000 phrases are supported")
    sequences = set()
    token_count = 0
    for phrase in phrases:
        if not isinstance(phrase, str):
            raise TypeError("each phrase must be a string")
        phrase = phrase.strip()
        if not phrase:
            continue
        if len(phrase) > 1024:
            raise ValueError("phrases must be at most 1024 characters")
        for variant in (phrase, " " + phrase):
            # tiktoken rejects special-token spellings by default.
            sequence = tuple(tokenizer.encode(variant))
            if len(sequence) > 128:
                raise ValueError("phrases must be at most 128 tokens")
            if any(t >= tokenizer.eot or t < 0 for t in sequence):
                raise ValueError("phrases may not contain Whisper control tokens")
            if sequence and sequence not in sequences:
                sequences.add(sequence)
                token_count += len(sequence)
                if token_count > 200000:
                    raise ValueError("phrase variants exceed the 200000-token limit")
    if not sequences:
        return None
    tree = _cached_tree(tuple(sorted(sequences)), vocab_size, config)
    return PhraseScorer(tree, tokenizer.eot, range(tokenizer.eot + 1, vocab_size))
