"""
Guards that exercises.json really is the single source of truth.

It is easy to "fix" a threshold by editing the analyzer and leaving the shared
file stale — which is exactly how the two runtimes drifted apart originally.
These tests fail if a constant stops tracking the shared config.
"""

import json

import pytest

from analyzer import base_analyzer
from analyzer.exercise_config import CONFIG_PATH, calibration, exercise, tolerances
from exercises.bicep import analyzer as bicep
from exercises.shoulder import analyzer as shoulder


@pytest.fixture(scope="module")
def raw():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def test_config_file_is_valid_json(raw):
    assert set(raw) >= {"tolerances", "calibration", "exercises"}


def test_both_exercises_are_defined(raw):
    assert set(raw["exercises"]) == {"bicep", "shoulder"}


def test_shared_tolerances_are_not_reimplemented():
    t = tolerances()
    assert base_analyzer.ANGLE_TOLERANCE_DEG == t["angleDeg"]
    assert base_analyzer.DRIFT_TOLERANCE_BODY == t["driftBodyRatio"]
    assert base_analyzer.SYMMETRY_TOLERANCE == t["symmetryDeg"]
    assert base_analyzer.TEMPO_MIN_SECONDS == t["tempoMinSeconds"]
    assert base_analyzer.VISIBILITY_WARN_FRAMES == t["visibilityWarnFrames"]


def test_calibration_settings_come_from_config():
    c = calibration()
    assert base_analyzer.CALIB_HOLD_SECONDS == c["holdSeconds"]
    assert base_analyzer.CALIB_STABILITY_TOLERANCE == c["stabilityTolerance"]
    assert len(base_analyzer.CALIB_KEY_POINTS) == len(c["keyPoints"])


def test_bicep_thresholds_come_from_config():
    cfg = exercise("bicep")
    assert bicep.UP_ANGLE_IDEAL == cfg["thresholds"]["up"]
    assert bicep.DOWN_ANGLE_IDEAL == cfg["thresholds"]["down"]
    assert bicep.QUALITY_WEIGHT_RANGE == cfg["scoring"]["range"]
    assert bicep.QUALITY_WEIGHT_TEMPO == cfg["scoring"]["tempo"]
    assert bicep.QUALITY_WEIGHT_DRIFT == cfg["scoring"]["drift"]
    assert bicep.QUALITY_WEIGHT_TORSO == cfg["scoring"]["torso"]


def test_shoulder_thresholds_come_from_config():
    cfg = exercise("shoulder")
    assert shoulder.PRESS_TOP_IDEAL == cfg["thresholds"]["up"]
    assert shoulder.PRESS_BOTTOM_IDEAL == cfg["thresholds"]["down"]
    assert shoulder.QUALITY_WEIGHT_LOCKOUT == cfg["scoring"]["lockout"]
    assert shoulder.QUALITY_WEIGHT_DEPTH == cfg["scoring"]["depth"]
    assert shoulder.QUALITY_WEIGHT_SYMMETRY == cfg["scoring"]["symmetry"]
    assert shoulder.QUALITY_WEIGHT_TEMPO == cfg["scoring"]["tempo"]


@pytest.mark.parametrize("key", ["bicep", "shoulder"])
def test_scoring_weights_sum_to_100(key):
    # Every rep starts at 100, so the weights define a full deduction budget.
    assert sum(exercise(key)["scoring"].values()) == 100


@pytest.mark.parametrize("key", ["bicep", "shoulder"])
def test_both_arms_use_the_same_joint_triplet_shape(key):
    triplets = exercise(key)["angle"]["triplets"]
    assert set(triplets) == {"left", "right"}
    assert all(len(t) == 3 for t in triplets.values())


def test_shoulder_press_measures_the_elbow_not_the_shoulder():
    # The web engine used to measure elbow->shoulder->hip here while Python
    # measured shoulder->elbow->wrist, with the same thresholds applied to
    # both. The shared definition settles it on the elbow.
    triplets = exercise("shoulder")["angle"]["triplets"]
    assert triplets["left"] == [11, 13, 15]
    assert triplets["right"] == [12, 14, 16]


def test_unknown_exercise_raises():
    with pytest.raises(KeyError):
        exercise("deadlift")
