"""Deterministic TurboBias tests; no model or dataset downloads."""

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_whisper.boosting import (
    PhraseBoostingConfig,
    PhraseTree,
    PhraseScorer,
    make_phrase_scorer,
)
from mlx_whisper.decoding import (
    BeamSearchDecoder,
    GreedyDecoder,
    DecodingOptions,
    DecodingTask,
)
from mlx_whisper.tokenizer import get_tokenizer
from test_beam_search import FakeInference


def scorer(sequences=((1, 2, 3), (1, 2, 4), (3,)), strength=0.1):
    return PhraseScorer(
        PhraseTree(sequences, 10, PhraseBoostingConfig(strength=strength)), 9, (8,)
    )


def test_hand_calculated_rewards_and_failed_prefix():
    tree = scorer(strength=1).tree
    state, reward = 0, 0
    for token, expected in [(1, 1), (2, 1 + math.log(2)), (5, -2 - math.log(2))]:
        value, state = tree.transition(state, token)
        assert value == pytest.approx(expected)
        reward += value
    assert reward == pytest.approx(0)


def test_completion_keeps_reward_and_overlap_restarts():
    tree = PhraseTree(((1, 2), (2, 3), (1, 2, 4)), 10)
    state = 0
    for token in (1, 2):
        _, state = tree.transition(state, token)
    score, state = tree.transition(state, 3)
    assert score == pytest.approx(1 + math.log(2))
    assert tree.final[state]
    assert tree.transition(state, 7)[0] == 0


@pytest.mark.parametrize("seed", range(8))
def test_mlx_matches_reference_every_state(seed):
    rng = np.random.default_rng(seed)
    sequences = [
        tuple(map(int, rng.integers(1, 12, size=rng.integers(1, 7)))) for _ in range(25)
    ]
    tree = PhraseTree(
        sequences,
        16,
        PhraseBoostingConfig(unknown_score=-0.25, strength=0.3, depth_scale=1.7),
    )
    runtime = PhraseScorer(tree, 15, (13, 14))
    states = list(range(len(tree.edges)))
    expected, target = tree.reference(states, 15, (13, 14))
    scores, next_states = runtime.advance(mx.array(states, mx.int32))
    np.testing.assert_allclose(np.array(scores), expected, atol=2e-6)
    np.testing.assert_array_equal(np.array(next_states), target)


def test_eos_compensation_and_ignored_state():
    runtime = scorer(strength=1)
    state = runtime.tree.edges[0][3]
    expected, targets = runtime.tree.reference([state], 9, (8,))
    scores, destinations = runtime.advance(mx.array([state]))
    np.testing.assert_allclose(np.array(scores), expected, atol=1e-6)
    assert scores[0, 8].item() == 0
    assert destinations[0, 8].item() == state
    assert scores[0, 9].item() >= 1


def test_empty_tree_reference():
    score, state = PhraseScorer(PhraseTree([], 10), 9).advance(mx.array([0]))
    np.testing.assert_array_equal(np.array(score), np.zeros((1, 10)))
    np.testing.assert_array_equal(np.array(state), np.zeros((1, 10)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": -1},
        {"depth_scale": float("nan")},
        {"unknown_score": float("inf")},
        {"final_eos_score": -1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PhraseBoostingConfig(**kwargs)


def test_tokenizer_boundaries_unicode_and_controls():
    tokenizer = get_tokenizer(True, language="pl")
    runtime = make_phrase_scorer(
        tokenizer, ["  Łódź ", "Łódź", "NVIDIA DGX"], 51865, PhraseBoostingConfig()
    )
    for text in ("Łódź", " Łódź", "NVIDIA DGX", " NVIDIA DGX"):
        state = 0
        for token in tokenizer.encode(text):
            state = runtime.tree.edges[state][token]
        assert runtime.tree.final[state]
    with pytest.raises(ValueError):
        make_phrase_scorer(tokenizer, ["<|endoftext|>"], 51865, PhraseBoostingConfig())
    with pytest.raises(TypeError):
        make_phrase_scorer(tokenizer, "NVIDIA", 51865, PhraseBoostingConfig())
    with pytest.raises(ValueError):
        make_phrase_scorer(tokenizer, ["x" * 1025], 51865, PhraseBoostingConfig())
    assert make_phrase_scorer(tokenizer, ["  "], 51865, PhraseBoostingConfig()) is None


def test_greedy_bonus_is_not_normalized_and_finished_rows_freeze():
    decoder = GreedyDecoder(0, 9, scorer(((1, 2),), strength=0.5))
    logits = mx.array([[0.0] * 10, [0.0] * 10])
    logits[:, 9] = -10
    tokens, _, scores = decoder.update(mx.array([[0], [9]]), logits, mx.zeros(2))
    assert tokens.tolist() == [[0, 1], [9, 9]]
    assert scores[0].item() == pytest.approx(-mx.logsumexp(logits[0]).item() + 0.5)
    assert scores[1].item() == 0
    assert decoder.boost_totals.tolist() == [0.5, 0]
    assert decoder.phrase_states[1].item() == 0


def test_beam_ancestry_score_accounting_and_finalization():
    runtime = scorer(((1, 2), (3, 4)), strength=0.2)
    inf = FakeInference()
    decoder = BeamSearchDecoder(2, 9, inf, phrase_scorer=runtime)
    logits = mx.full((4, 10), -10.0)
    logits[:, 1] = 5
    logits[:, 3] = 4
    tokens, _, scores = decoder.update(
        mx.array([[0], [0], [7], [7]]), logits, mx.array([0.0, -5.0, 0.0, -5.0])
    )
    assert inf.rearrange_calls[-1] == [0, 0, 2, 2]
    assert (
        decoder.phrase_states.tolist()
        == [runtime.tree.edges[0][1], runtime.tree.edges[0][3]] * 2
    )
    assert np.allclose(decoder.boost_totals.tolist(), 0.2)
    logits = mx.full((4, 10), -10.0)
    logits[:, 9] = 5
    tokens, _, scores = decoder.update(tokens, logits, scores)
    result, totals = decoder.finalize(tokens.reshape(2, 2, -1), scores.reshape(2, 2))
    assert result.shape[:2] == (2, 2)
    assert decoder.final_boosts.shape == totals.shape
    assert np.isfinite(np.array(totals - decoder.final_boosts)).all()
    decoder.reset()
    assert decoder.phrase_states is None and decoder.finished_boosts is None


def test_synthetic_eot_does_not_earn_bonus():
    decoder = BeamSearchDecoder(2, 9, FakeInference(), phrase_scorer=scorer())
    logits = mx.full((2, 10), -10.0)
    logits[:, 1:3] = mx.array([5.0, 4.0])
    tokens, _, scores = decoder.update(mx.array([[0], [0]]), logits, mx.zeros(2))
    before = sorted(decoder.boost_totals.tolist())
    decoder.finalize(tokens.reshape(1, 2, -1), scores.reshape(1, 2))
    assert sorted(decoder.final_boosts.tolist()[0]) == before


def fake_model():
    return SimpleNamespace(
        is_multilingual=False,
        num_languages=0,
        dims=SimpleNamespace(n_text_ctx=16, n_vocab=51864),
    )


def test_disabled_path_does_not_build_tree():
    options = DecodingOptions(language="en", without_timestamps=True)
    with patch("mlx_whisper.decoding.make_phrase_scorer", side_effect=AssertionError):
        for phrases, config in [
            (None, PhraseBoostingConfig()),
            ([], PhraseBoostingConfig()),
            (["hello"], PhraseBoostingConfig(strength=0)),
        ]:
            task = DecodingTask(
                fake_model(), replace(options, phrases=phrases, phrase_boosting=config)
            )
            assert task.phrase_scorer is None


def test_prompts_are_not_consumed_and_requests_are_independent():
    options = DecodingOptions(
        language="en", without_timestamps=True, phrases=["hello"], prompt="hello"
    )
    first = DecodingTask(fake_model(), options)
    second = DecodingTask(fake_model(), options)
    assert first.phrase_scorer.tree is second.phrase_scorer.tree
    assert first.decoder is not second.decoder
    assert first.decoder.phrase_states is None
    assert second.decoder.phrase_states is None


def test_masks_cannot_be_resurrected():
    decoder = GreedyDecoder(0, 9, scorer(((1,),), strength=100))
    logits = mx.full((1, 10), -10.0)
    logits[:, 1] = -mx.inf
    logits[:, 9] = -mx.inf
    logits[:, 2] = 1
    tokens, _, _ = decoder.update(mx.array([[0]]), logits, mx.zeros(1))
    assert tokens.tolist() == [[0, 2]]


def test_beam_scores_match_unrenormalized_numpy_oracle():
    runtime = scorer(((1, 2), (3,)), strength=0.5)
    decoder = BeamSearchDecoder(2, 9, FakeInference(), phrase_scorer=runtime)
    raw = np.array(
        [
            [0.0, 2.0, 1.0, 0.0, -4.0, -4.0, -4.0, -4.0, -4.0, -10.0],
            [0.0, 1.0, 2.0, 3.0, -4.0, -4.0, -4.0, -4.0, -4.0, -10.0],
        ],
        dtype=np.float32,
    )
    prior = np.array([0.0, -0.3], dtype=np.float32)
    expected_bias, _ = runtime.tree.reference([0, 0], 9, (8,))
    expected_model = raw - np.log(np.exp(raw).sum(axis=-1, keepdims=True))
    expected = expected_model + np.array(expected_bias) + prior[:, None]
    # Distinct prefixes allow us to recover each selected parent unambiguously.
    tokens, _, totals = decoder.update(
        mx.array([[6], [7]]), mx.array(raw), mx.array(prior)
    )
    for row, sequence in enumerate(tokens.tolist()):
        parent, token = sequence[0] - 6, sequence[1]
        assert totals[row].item() == pytest.approx(expected[parent, token], abs=1e-6)
        assert totals[row].item() - decoder.boost_totals[row].item() == pytest.approx(
            prior[parent] + expected_model[parent, token], abs=1e-6
        )


def test_empty_hypothesis_rank_is_finite():
    from mlx_whisper.decoding import MaximumLikelihoodRanker

    assert MaximumLikelihoodRanker(None).scores([[[]]], [[-1.0]]) == [[-1.0]]


def test_sampling_is_reproducible_and_tracks_selected_state():
    runtime = scorer(((1, 2),), strength=0.1)
    answers = []
    for _ in range(2):
        mx.random.seed(123)
        decoder = GreedyDecoder(0.5, 9, runtime)
        tokens, _, _ = decoder.update(mx.array([[0]]), mx.zeros((1, 10)), mx.zeros(1))
        token = tokens[0, -1].item()
        _, states = runtime.tree.reference([0], 9, (8,))
        assert decoder.phrase_states.item() == states[0][token]
        answers.append(token)
    assert answers[0] == answers[1]


def test_interleaved_decoders_share_only_immutable_scorer():
    runtime = scorer(((1, 2), (3, 4)))
    first, second = GreedyDecoder(0, 9, runtime), GreedyDecoder(0, 9, runtime)
    for decoder, token in ((first, 1), (second, 3)):
        logits = mx.full((1, 10), -100.0)
        logits[0, token] = 0
        decoder.update(mx.array([[0]]), logits, mx.zeros(1))
    assert first.phrase_states.item() == runtime.tree.edges[0][1]
    assert second.phrase_states.item() == runtime.tree.edges[0][3]
    first.reset()
    assert second.phrase_states.item() == runtime.tree.edges[0][3]


def test_completed_overlap_fallback_uses_each_edge_weight():
    # The completed node's edge is free; abandoning its incomplete suffix still
    # subtracts that suffix's potential, as in the TurboBias failure graph.
    tree = PhraseTree(((1, 2), (2, 3)), 10)
    _, state = tree.transition(0, 1)
    _, state = tree.transition(state, 2)
    assert tree.transition(state, 7) == (-1.0, 0)


def test_input_limits_and_invalid_tokens():
    tokenizer = get_tokenizer(True, language="en")
    for phrases in (["a"] * 20001, ["a " * 129]):
        with pytest.raises(ValueError):
            make_phrase_scorer(tokenizer, phrases, 51865, PhraseBoostingConfig())
    with pytest.raises(TypeError):
        make_phrase_scorer(tokenizer, [None], 51865, PhraseBoostingConfig())
    with pytest.raises(ValueError):
        PhraseTree(((10,),), 10)


def test_boosted_selection_keeps_vocabulary_arrays_in_mlx():
    runtime = scorer(((1, 2), (3, 4)))
    decoder = BeamSearchDecoder(2, 9, FakeInference(), phrase_scorer=runtime)
    tokens = mx.array([[0], [0]])
    logits = mx.array([[0.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]] * 2)
    with patch(
        "mlx_whisper.decoding.np.array", side_effect=AssertionError("host array copy")
    ):
        tokens, _, scores = decoder.update(tokens, logits, mx.zeros(2))
        logits[:, 9] = 10
        decoder.update(tokens, logits, scores)
    assert (
        decoder.selected_parent_indices.tolist()
        == decoder.inference.rearrange_calls[-1]
    )
    assert decoder.finished_boosts[0]


@pytest.mark.parametrize("beam", [1, 3, 5])
def test_boost_rewards_follow_multistep_selected_and_finished_paths(beam):
    runtime = scorer(((1, 2), (2, 3), (3,), (1, 2, 4)))
    decoder = BeamSearchDecoder(
        beam, 9, FakeInference(), patience=2, phrase_scorer=runtime
    )
    rng = np.random.default_rng(624)
    tokens = mx.array([[0]] * (2 * beam))
    scores = mx.zeros(2 * beam)

    def replay(sequence):
        state, total = 0, 0.0
        for token in sequence[1:]:
            rewards, targets = runtime.tree.reference([state], 9, (8,))
            total += rewards[0][token]
            state = targets[0][token]
        return state, total

    for step in range(7):
        logits = rng.normal(size=(2 * beam, 10)).astype(np.float32)
        logits[:, 9] += 2 if step % 2 else -2
        tokens, _, scores = decoder.update(tokens, mx.array(logits), scores)
        for sequence, state, total in zip(
            tokens.tolist(),
            decoder.phrase_states.tolist(),
            decoder.boost_totals.tolist(),
        ):
            expected_state, expected_total = replay(sequence)
            assert state == expected_state
            assert total == pytest.approx(expected_total, abs=1e-6)
        for group in decoder.finished_boosts:
            for sequence, total in group.items():
                assert total == pytest.approx(replay(sequence)[1], abs=1e-6)
