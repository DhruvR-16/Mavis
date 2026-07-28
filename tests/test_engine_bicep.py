"""
Characterization tests for the bicep curl engine.

These lock in the engine's current output on a real recorded curl so the
Phase 3 refactors (shared exercise schema, engine extraction) can be proven
behaviour-preserving. If a change here is intentional, update the expected
values in the same commit and say why.

Fixture: fixtures/bicep_curl.json — 67 frames @ 29.97fps of a barbell curl,
two reps.
"""

import pytest

from exercises.bicep.analyzer import BicepAnalyzer
from tests.replay import load_fixture, replay, summarise


@pytest.fixture(scope="module")
def replayed_analyzer():
    meta, frames = load_fixture("bicep_curl")
    return replay(BicepAnalyzer(), frames, meta["fps"])


@pytest.fixture(scope="module")
def replayed(replayed_analyzer):
    return summarise(replayed_analyzer)


def test_counts_both_reps(replayed):
    assert replayed["reps"] == 2


def test_rep_quality_scores(replayed):
    assert replayed["qualities"] == [100, 100]


def test_good_bad_split(replayed):
    assert (replayed["good"], replayed["bad"]) == (2, 0)


def test_ordinary_reps_are_not_flagged(replayed):
    # This is footage of someone curling normally. Both reps used to be
    # faulted — one for a "too fast" eccentric that was really just the
    # eccentric being timed against a whole-rep threshold, one for elbow drift
    # measured against an absolute coordinate that moves when the lifter does.
    # A coach that calls ordinary reps bad teaches users to ignore it.
    assert replayed["faults"] == [[], []]


def test_rep_duration_covers_the_whole_rep(replayed_analyzer):
    # Timed from leaving the bottom to returning, not just the lowering phase.
    from analyzer.base_analyzer import TEMPO_MIN_SECONDS

    for rep in replayed_analyzer.rep_history:
        assert rep.duration_sec >= TEMPO_MIN_SECONDS
