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
def replayed():
    meta, frames = load_fixture("bicep_curl")
    return summarise(replay(BicepAnalyzer(), frames, meta["fps"]))


def test_counts_both_reps(replayed):
    assert replayed["reps"] == 2


def test_rep_quality_scores(replayed):
    assert replayed["qualities"] == [75, 85]


def test_good_bad_split(replayed):
    # Both reps clear the quality>=60 bar, so neither is counted bad.
    assert (replayed["good"], replayed["bad"]) == (2, 0)


def test_faults_detected_per_rep(replayed):
    # Rep 1's eccentric is genuinely rushed; rep 2 shows elbow drift.
    assert replayed["faults"] == [
        ["Too fast — control the negative"],
        ["Elbow drifting forward"],
    ]


def test_quality_matches_fault_deductions(replayed):
    # Guards the scoring weights against silent drift: the tempo fault costs
    # QUALITY_WEIGHT_TEMPO (25) and the drift fault is scaled by how far past
    # DRIFT_TOLERANCE_BODY the elbow travelled.
    from exercises.bicep.analyzer import QUALITY_WEIGHT_TEMPO

    assert replayed["qualities"][0] == 100 - QUALITY_WEIGHT_TEMPO
