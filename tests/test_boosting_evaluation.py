"""Check the evaluation metric against small, hand-aligned transcripts."""

from scripts.evaluate_phrase_boosting import aligned_errors


def test_target_deletion_substitution_insertion_and_non_target_errors():
    assert aligned_errors(["say", "quasar"], ["say"], {"quasar"}) == (1, 1)
    assert aligned_errors(["say", "quasar"], ["say", "pulsar"], {"quasar"}) == (1, 1)
    assert aligned_errors(["say"], ["say", "quasar"], {"quasar"}) == (1, 1)
    assert aligned_errors(["say", "quasar"], ["a", "quasar"], {"quasar"}) == (1, 0)
    assert aligned_errors([], [], set()) == (0, 0)
    assert aligned_errors(["quasar"], ["quasar"], {"quasar"}) == (0, 0)
