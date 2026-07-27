"""
Direct tests for the rep-scoring functions.

The recorded fixtures only exercise the deduction categories the filmed
athletes actually triggered — on fixtures/shoulder_press.json, for instance,
depth and lockout are always clean, so those branches never run. These tests
drive each deduction independently so no scoring category is left unverified.
"""

import time

import pytest

from analyzer.base_analyzer import (
    ANGLE_TOLERANCE_DEG,
    DRIFT_TOLERANCE_BODY,
    SYMMETRY_TOLERANCE,
)
from exercises.bicep import analyzer as bicep_mod
from exercises.shoulder import analyzer as shoulder_mod
from exercises.bicep.analyzer import BicepAnalyzer
from exercises.shoulder.analyzer import ShoulderAnalyzer
from tests.replay import virtual_clock

CLEAN_REP_SECONDS = 5.0


# ── Shoulder press ───────────────────────────────────────────────────────────

def shoulder(top=175.0, bottom=60.0, asymmetry=0.0, rep_seconds=CLEAN_REP_SECONDS):
    """Score one shoulder rep with the given per-rep extremes."""
    a = ShoulderAnalyzer()
    with virtual_clock(30.0):
        a._max_left_angle = a._max_right_angle = top
        a._min_left_angle = a._min_right_angle = bottom
        a._max_asymmetry = asymmetry
        a.rep_start_time = time.time() - rep_seconds
        return a._score_rep()


def test_shoulder_clean_rep_scores_full_marks():
    assert shoulder() == 100


def test_shoulder_missed_lockout_is_deducted():
    below_lockout = shoulder_mod.PRESS_TOP_IDEAL - ANGLE_TOLERANCE_DEG - 10
    assert shoulder(top=below_lockout) == 100 - shoulder_mod.QUALITY_WEIGHT_LOCKOUT


def test_shoulder_missed_depth_is_deducted():
    above_depth = shoulder_mod.PRESS_BOTTOM_IDEAL + ANGLE_TOLERANCE_DEG + 10
    assert shoulder(bottom=above_depth) == 100 - shoulder_mod.QUALITY_WEIGHT_DEPTH


def test_shoulder_asymmetry_is_deducted():
    assert shoulder(asymmetry=SYMMETRY_TOLERANCE * 2) < 100


def test_shoulder_fast_rep_is_deducted():
    assert shoulder(rep_seconds=0.1) == 100 - shoulder_mod.QUALITY_WEIGHT_TEMPO


def test_shoulder_score_never_leaves_bounds():
    worst = shoulder(top=10.0, bottom=179.0, asymmetry=180.0, rep_seconds=0.0)
    assert 0 <= worst <= 100


# ── Bicep curl ───────────────────────────────────────────────────────────────

def bicep(peak=40.0, bottom=155.0, drift=0.0, lean=0.0, rep_seconds=CLEAN_REP_SECONDS):
    a = BicepAnalyzer()
    with virtual_clock(30.0):
        a._peak_angle_this_rep = peak
        a._bottom_angle_this_rep = bottom
        a._max_drift_this_rep = drift
        a._max_torso_lean_this_rep = lean
        a.rep_start_time = time.time() - rep_seconds
        return a._score_rep()


def test_bicep_clean_rep_scores_full_marks():
    assert bicep() == 100


def test_bicep_missed_contraction_is_deducted():
    shallow = bicep_mod.UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG + 10
    assert bicep(peak=shallow) == 100 - bicep_mod.QUALITY_WEIGHT_RANGE


def test_bicep_partial_extension_is_deducted():
    short = bicep_mod.DOWN_ANGLE_IDEAL - ANGLE_TOLERANCE_DEG * 2 - 10
    assert bicep(bottom=short) == 100 - int(bicep_mod.QUALITY_WEIGHT_RANGE * 0.5)


def test_bicep_elbow_drift_is_deducted():
    assert bicep(drift=DRIFT_TOLERANCE_BODY * 2) < 100


def test_bicep_torso_swing_is_deducted():
    swing = 20.0 + ANGLE_TOLERANCE_DEG + 10
    assert bicep(lean=swing) == 100 - bicep_mod.QUALITY_WEIGHT_TORSO


def test_bicep_fast_rep_is_deducted():
    assert bicep(rep_seconds=0.1) == 100 - bicep_mod.QUALITY_WEIGHT_TEMPO


def test_bicep_score_never_leaves_bounds():
    worst = bicep(peak=180.0, bottom=0.0, drift=10.0, lean=90.0, rep_seconds=0.0)
    assert 0 <= worst <= 100


@pytest.mark.parametrize("scorer", [shoulder, bicep])
def test_deductions_accumulate(scorer):
    assert scorer(rep_seconds=0.1) < scorer()
