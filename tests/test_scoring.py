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


def test_shoulder_severely_missed_lockout_is_deducted():
    # Well short of lockout — the full weight should land.
    far_short = shoulder_mod.PRESS_TOP_IDEAL - ANGLE_TOLERANCE_DEG - 40
    assert shoulder(top=far_short) == 100 - shoulder_mod.QUALITY_WEIGHT_LOCKOUT


def test_shoulder_severely_missed_depth_is_deducted():
    far_short = shoulder_mod.PRESS_BOTTOM_IDEAL + ANGLE_TOLERANCE_DEG + 40
    assert shoulder(bottom=far_short) == 100 - shoulder_mod.QUALITY_WEIGHT_DEPTH


def test_shoulder_marginal_miss_costs_less_than_a_severe_one():
    # The point of graded penalties: missing lockout by 5° must not cost the
    # same as missing it by 40°.
    edge = shoulder_mod.PRESS_TOP_IDEAL - ANGLE_TOLERANCE_DEG
    marginal = shoulder(top=edge - 5)
    severe = shoulder(top=edge - 40)
    assert 100 > marginal > severe


def test_shoulder_asymmetry_is_deducted():
    assert shoulder(asymmetry=SYMMETRY_TOLERANCE * 3) < 100


def test_shoulder_bounced_rep_is_deducted():
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


def test_bicep_severely_missed_contraction_is_deducted():
    barely_bent = bicep_mod.UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG + 40
    assert bicep(peak=barely_bent) == 100 - bicep_mod.QUALITY_WEIGHT_RANGE


def test_bicep_severely_partial_extension_is_deducted():
    short = bicep_mod.DOWN_ANGLE_IDEAL - ANGLE_TOLERANCE_DEG * 2 - 35
    assert bicep(bottom=short) == 100 - int(bicep_mod.QUALITY_WEIGHT_RANGE * 0.5)


def test_bicep_marginal_miss_costs_less_than_a_severe_one():
    edge = bicep_mod.UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG
    marginal = bicep(peak=edge + 5)
    severe = bicep(peak=edge + 40)
    assert 100 > marginal > severe


def test_bicep_severe_elbow_swing_is_deducted():
    assert bicep(drift=DRIFT_TOLERANCE_BODY * 3) < 100


def test_bicep_severe_torso_swing_is_deducted():
    swing = bicep_mod.TORSO_TOLERANCE_DEG + 25
    assert bicep(lean=swing) == 100 - bicep_mod.QUALITY_WEIGHT_TORSO


def test_bicep_bounced_rep_is_deducted():
    assert bicep(rep_seconds=0.1) == 100 - bicep_mod.QUALITY_WEIGHT_TEMPO


def test_bicep_score_never_leaves_bounds():
    worst = bicep(peak=180.0, bottom=0.0, drift=10.0, lean=90.0, rep_seconds=0.0)
    assert 0 <= worst <= 100


@pytest.mark.parametrize("scorer", [shoulder, bicep])
def test_deductions_accumulate(scorer):
    assert scorer(rep_seconds=0.1) < scorer()
