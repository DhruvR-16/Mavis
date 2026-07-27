"""
Regression tests for the calibration stability check (Phase 2).

Calibration used to complete purely on elapsed time, so whatever pose happened
to be held at the 3-second mark became the drift anchor — including a pose
caught mid-curl, which then corrupted every drift measurement for the rest of
the session. It now restarts the countdown when the pose moves.
"""

from analyzer.base_analyzer import (
    CALIB_STABILITY_TOLERANCE,
    BaseAnalyzer,
    CalibrationState,
)
from analyzer.feature_extractor import FeatureExtractor
from tests.replay import Landmark, load_fixture, virtual_clock


class _Analyzer(BaseAnalyzer):
    """BaseAnalyzer is abstract; these tests only exercise calibration."""

    def analyze_form(self, features, landmarks):
        raise AssertionError("not exercised by calibration tests")


def still_frame():
    _, frames = load_fixture("bicep_curl")
    return frames[0]


def shifted(frame, dx):
    return [Landmark(lm.x + dx, lm.y, lm.z, lm.visibility) for lm in frame]


def test_completes_when_pose_is_held_still():
    a, ex, frame = _Analyzer("Test"), FeatureExtractor(), still_frame()

    with virtual_clock(30.0) as clock:
        a.start_calibration()
        done = False
        for _ in range(int(30 * a.calib_countdown) + 5):
            done = a.tick_calibration(ex, frame)
            clock.tick()
            if done:
                break

    assert done
    assert a.calib_state is CalibrationState.COMPLETE
    assert ex.calibrated


def test_movement_restarts_the_countdown():
    a, ex, frame = _Analyzer("Test"), FeatureExtractor(), still_frame()
    moving = shifted(frame, CALIB_STABILITY_TOLERANCE * 3)

    with virtual_clock(30.0) as clock:
        a.start_calibration()
        # Jitter between two poses for well past the hold duration.
        for i in range(int(30 * a.calib_countdown * 3)):
            assert not a.tick_calibration(ex, moving if i % 2 else frame)
            clock.tick()

    assert a.calib_state is CalibrationState.RUNNING
    assert not ex.calibrated


def test_settling_after_movement_still_calibrates():
    a, ex, frame = _Analyzer("Test"), FeatureExtractor(), still_frame()
    moving = shifted(frame, CALIB_STABILITY_TOLERANCE * 3)

    with virtual_clock(30.0) as clock:
        a.start_calibration()
        for i in range(20):
            a.tick_calibration(ex, moving if i % 2 else frame)
            clock.tick()

        done = False
        for _ in range(int(30 * a.calib_countdown) + 5):
            done = a.tick_calibration(ex, frame)
            clock.tick()
            if done:
                break

    assert done
    assert ex.calibrated


def test_sub_tolerance_jitter_does_not_block_calibration():
    # Real landmarks are never perfectly static; small noise must not prevent
    # calibration from ever completing.
    a, ex, frame = _Analyzer("Test"), FeatureExtractor(), still_frame()
    nudged = shifted(frame, CALIB_STABILITY_TOLERANCE * 0.2)

    with virtual_clock(30.0) as clock:
        a.start_calibration()
        done = False
        for i in range(int(30 * a.calib_countdown) + 5):
            done = a.tick_calibration(ex, nudged if i % 2 else frame)
            clock.tick()
            if done:
                break

    assert done
