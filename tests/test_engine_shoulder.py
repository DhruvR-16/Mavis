"""
Characterization tests for the shoulder press engine.

Fixture: fixtures/shoulder_press.json — 210 frames @ 29.97fps, four presses.

Note the Phase 2 peak/bottom-angle bug was in the *web* engine, not here —
this analyzer has always seeded its max trackers at 0 and its min trackers at
180. The equivalent guard for the web engine belongs in its own suite; this
file exists so the schema extraction can be shown not to change Python's
behaviour.
"""

import pytest

from exercises.shoulder.analyzer import ShoulderAnalyzer
from tests.replay import load_fixture, replay, summarise


@pytest.fixture(scope="module")
def replayed():
    meta, frames = load_fixture("shoulder_press")
    return summarise(replay(ShoulderAnalyzer(), frames, meta["fps"]))


@pytest.fixture(scope="module")
def tracked_extremes():
    """Per-rep (min, max) elbow angles captured at the moment of scoring."""
    meta, frames = load_fixture("shoulder_press")
    analyzer = ShoulderAnalyzer()
    captured = []
    original = analyzer._score_rep

    def spy():
        captured.append((analyzer._min_left_angle, analyzer._max_left_angle))
        return original()

    analyzer._score_rep = spy
    replay(analyzer, frames, meta["fps"])
    return captured


def test_counts_all_four_reps(replayed):
    assert replayed["reps"] == 4


def test_rep_quality_scores(replayed):
    assert replayed["qualities"] == [100, 100, 100, 90]


def test_ordinary_reps_are_not_flagged(replayed):
    # Ordinary pressing. All four reps used to be faulted "too fast" because
    # the descent alone was timed against a whole-rep threshold, and the
    # symmetry tolerance sat right on the median of normal left/right variation.
    assert replayed["faults"] == [[], [], [], []]


def test_good_bad_split(replayed):
    assert (replayed["good"], replayed["bad"]) == (4, 0)


def test_angle_trackers_capture_both_extremes(tracked_extremes):
    # A press sweeps roughly 20°-165° at the elbow. If a min/max tracker is
    # seeded at the wrong bound its comparison never fires and it stays pinned
    # at the seed value, silently disabling a whole scoring category — the
    # failure mode the web engine shipped with. Assert both ends actually move.
    assert tracked_extremes, "no reps were scored"
    for minimum, maximum in tracked_extremes:
        assert 0 < minimum < 120, f"min tracker never moved off its seed: {minimum}"
        assert 140 < maximum <= 180, f"max tracker never moved off its seed: {maximum}"


def test_final_rep_scores_lower_than_the_rest(replayed):
    # Form degrades on the last rep. Asserting the scores actually differ is
    # what catches a tracker being seeded so that a whole deduction category
    # becomes unreachable — the failure mode the web engine had, where every
    # rep scored identically regardless of form.
    qualities = replayed["qualities"]
    assert qualities[-1] < qualities[0]
    assert len(set(qualities)) > 1
