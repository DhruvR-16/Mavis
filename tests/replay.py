"""
Shared harness for replaying a recorded landmark fixture through an analyzer.

Fixtures in fixtures/ are captured from real workout video by
tools/extract_fixture.py, and the same files are replayed by the TypeScript
test suite. Both engines seeing identical input is what keeps them from
diverging.
"""

import json
import os
import time
from contextlib import contextmanager
from unittest import mock

from analyzer.base_analyzer import CalibrationState

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures"
)


class Landmark:
    """Stands in for a MediaPipe NormalizedLandmark."""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def load_fixture(name: str):
    """Returns (metadata, frames) where frames is a list of landmark lists."""
    with open(os.path.join(FIXTURES_DIR, f"{name}.json")) as f:
        data = json.load(f)
    frames = [[Landmark(*lm) for lm in frame] for frame in data["frames"]]
    return data, frames


class VirtualClock:
    """
    Replaces wall-clock time during replay.

    A fixture replays in milliseconds, so real time.time() would make every rep
    look faster than TEMPO_MIN_SECONDS and permanently trip the tempo fault.
    Advancing by 1/fps per frame instead reproduces the timing the analyzer
    would have seen from the live camera.
    """

    def __init__(self, fps: float):
        self.now = 10_000.0  # arbitrary non-zero epoch
        self.dt = 1.0 / fps

    def tick(self):
        self.now += self.dt

    def time(self):
        return self.now


@contextmanager
def virtual_clock(fps: float):
    clock = VirtualClock(fps)
    with mock.patch.object(time, "time", clock.time):
        yield clock


def replay(analyzer, frames, fps: float, calibrate_at: int = 0):
    """
    Feed a fixture through an analyzer's per-frame pipeline.

    Mirrors what <Analyzer>.run() does per frame, minus the camera and HUD:
    smooth -> extract features -> analyze_form. Calibration is applied directly
    at `calibrate_at` rather than run through the timed hold, so these tests
    isolate the state machine and scoring; calibration timing is covered
    separately in test_calibration.py.
    """
    with virtual_clock(fps) as clock:
        for i, raw in enumerate(frames):
            lms = analyzer.smoother.smooth(raw)

            if i == calibrate_at:
                analyzer.extractor.calibrate(lms)
                analyzer.calib_state = CalibrationState.COMPLETE

            if i >= calibrate_at:
                features = analyzer.extractor.get_features(lms)
                analyzer.analyze_form(features, lms)

            clock.tick()

    return analyzer


def summarise(analyzer) -> dict:
    """Compact, assertable view of a replayed session."""
    return {
        "reps": analyzer.total_reps,
        "good": analyzer.good_reps,
        "bad": analyzer.bad_reps,
        "qualities": [r.quality for r in analyzer.rep_history],
        "faults": [r.faults for r in analyzer.rep_history],
    }
