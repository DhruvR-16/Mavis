"""
Regression tests for active-arm detection (Phase 2).

The bicep analyzer used to read features[1] — the LEFT elbow — unconditionally,
so a right-arm curl counted zero reps and right-arm form faults were invisible.
It now latches whichever arm is more contracted while at rest and holds that
choice for the duration of the rep.
"""

from contextlib import contextmanager

import pytest

from analyzer.base_analyzer import CalibrationState, Stage
from exercises.bicep.analyzer import BicepAnalyzer
from tests.replay import virtual_clock

FPS = 30.0


def features(left_angle, right_angle, left_drift=0.0, right_drift=0.0, lean=5.0):
    """Build a feature vector directly, bypassing landmark geometry."""
    f = [0.0] * 11
    f[0] = left_drift
    f[1] = left_angle
    f[2] = right_angle
    f[6] = lean
    f[10] = right_drift
    return f


@contextmanager
def session():
    """A calibrated analyzer on a virtual clock (no real sleeping)."""
    analyzer = BicepAnalyzer()
    analyzer.calib_state = CalibrationState.COMPLETE
    with virtual_clock(FPS) as clock:
        yield analyzer, clock


def curl(analyzer, clock, side, drift=0.0, eccentric_s=0.9):
    """
    Drive one full curl on `side`, holding the other arm extended.

    `eccentric_s` is the dwell at peak contraction before lowering; the tempo
    fault keys off this interval, so keeping it above TEMPO_MIN_SECONDS gives a
    clean rep by default.
    """
    def frame(angle):
        if side == "right":
            return features(left_angle=170, right_angle=angle, right_drift=drift)
        return features(left_angle=angle, right_angle=170, left_drift=drift)

    for angle in (170, 150, 120, 90, 60, 40):
        analyzer.analyze_form(frame(angle), None)
        clock.tick()

    for _ in range(int(eccentric_s * FPS)):
        clock.tick()

    for angle in (60, 90, 120, 150, 170):
        analyzer.analyze_form(frame(angle), None)
        clock.tick()


@pytest.mark.parametrize("side", ["left", "right"])
def test_curl_is_counted_on_either_arm(side):
    with session() as (a, clock):
        curl(a, clock, side)
        assert a.total_reps == 1


def test_active_side_follows_the_working_arm():
    with session() as (a, clock):
        curl(a, clock, "right")
        curl(a, clock, "left")
        assert a.total_reps == 2


def test_right_arm_drift_fault_is_attributed_to_the_right_elbow():
    with session() as (a, clock):
        curl(a, clock, "right", drift=0.30)
        assert "Elbow drifting forward" in a.rep_history[0].faults


def test_idle_arm_drift_does_not_raise_a_fault():
    # Left arm drifts wildly but is not the working arm — must be ignored.
    with session() as (a, clock):
        def frame(angle):
            return features(left_angle=170, right_angle=angle, left_drift=0.9)

        for angle in (170, 150, 120, 90, 60, 40):
            a.analyze_form(frame(angle), None)
            clock.tick()
        for _ in range(int(0.9 * FPS)):
            clock.tick()
        for angle in (60, 90, 120, 150, 170):
            a.analyze_form(frame(angle), None)
            clock.tick()

        assert a.total_reps == 1
        assert "Elbow drifting forward" not in a.rep_history[0].faults


def test_active_side_is_latched_during_a_rep():
    # Once the rep is underway the choice must not flip, even if landmark noise
    # briefly makes the idle arm look more contracted.
    with session() as (a, clock):
        for angle in (170, 150, 120, 90, 60, 40):
            a.analyze_form(features(left_angle=170, right_angle=angle), None)
            clock.tick()

        assert a.stage == Stage.UP
        assert a._active_side == "right"

        a.analyze_form(features(left_angle=30, right_angle=45), None)
        assert a._active_side == "right"
