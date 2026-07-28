"""
Tolerance tests — the engine must leave room for human error.

Nobody hits exact angles or exact tempos. Before these tolerances were tuned
against real footage, 61% of ordinary press reps and 46% of ordinary curl reps
were flagged as faulty, which just teaches users to ignore the coach.

Loosening has an obvious failure mode in the other direction, though: a coach
that never objects is equally useless. Every test here comes in a pair —
ordinary form must pass, clearly poor form must still be caught.
"""

import pytest

from analyzer.base_analyzer import (
    DRIFT_TOLERANCE_BODY,
    FAULT_CONFIRM_FRAMES,
    TEMPO_MIN_SECONDS,
    graded_penalty,
)
from exercises.bicep.analyzer import BicepAnalyzer, TORSO_TOLERANCE_DEG
from tests.replay import load_fixture, replay
from tests.test_active_arm import FPS, curl, features, session

# ── graded_penalty: the core "room for error" primitive ──────────────────────


def test_within_tolerance_costs_nothing():
    assert graded_penalty(-5.0, 20.0, 40) == 0
    assert graded_penalty(0.0, 20.0, 40) == 0


def test_penalty_ramps_with_severity():
    small = graded_penalty(2.0, 20.0, 40)
    medium = graded_penalty(10.0, 20.0, 40)
    large = graded_penalty(20.0, 20.0, 40)
    assert 0 < small < medium < large == 40


def test_penalty_is_capped_at_the_full_weight():
    assert graded_penalty(1000.0, 20.0, 40) == 40


def test_a_hair_over_the_line_is_nearly_free():
    # The whole point: crossing a threshold by a fraction of a degree must not
    # cost the same as missing it wildly.
    assert graded_penalty(0.1, 30.0, 35) <= 1


# ── Fault debouncing ─────────────────────────────────────────────────────────


def test_momentary_noise_does_not_raise_a_fault():
    # One bad frame is landmark jitter, not bad form. Per-rep extremes are
    # tracked with max(), so without debouncing a single spike is enough to
    # fault an otherwise clean rep.
    a = BicepAnalyzer()
    for _ in range(FAULT_CONFIRM_FRAMES - 1):
        assert not a.confirm_fault("drift", True)


def test_a_sustained_problem_is_still_raised():
    a = BicepAnalyzer()
    fired = [a.confirm_fault("drift", True) for _ in range(FAULT_CONFIRM_FRAMES)]
    assert fired[-1] is True


def test_the_streak_resets_when_form_recovers():
    a = BicepAnalyzer()
    for _ in range(FAULT_CONFIRM_FRAMES - 1):
        a.confirm_fault("drift", True)
    a.confirm_fault("drift", False)
    assert not a.confirm_fault("drift", True)


# ── Ordinary form must not be flagged ────────────────────────────────────────


@pytest.mark.parametrize("fixture_name", ["bicep_curl", "shoulder_press"])
def test_real_footage_is_not_faulted(fixture_name):
    from exercises.bicep.analyzer import BicepAnalyzer as B
    from exercises.shoulder.analyzer import ShoulderAnalyzer as S

    meta, frames = load_fixture(fixture_name)
    analyzer = replay((B if fixture_name == "bicep_curl" else S)(), frames, meta["fps"])

    flagged = [r.rep_number for r in analyzer.rep_history if r.faults]
    assert not flagged, f"ordinary reps flagged: {flagged}"


@pytest.mark.parametrize("fixture_name", ["bicep_curl", "shoulder_press"])
def test_real_footage_scores_well(fixture_name):
    from exercises.bicep.analyzer import BicepAnalyzer as B
    from exercises.shoulder.analyzer import ShoulderAnalyzer as S

    meta, frames = load_fixture(fixture_name)
    analyzer = replay((B if fixture_name == "bicep_curl" else S)(), frames, meta["fps"])

    assert analyzer.bad_reps == 0
    assert min(r.quality for r in analyzer.rep_history) >= 80


def test_slightly_imperfect_form_is_not_faulted():
    # Drift and lean just under tolerance — the normal wobble of a real set.
    with session() as (a, clock):
        curl(a, clock, "right", drift=DRIFT_TOLERANCE_BODY * 0.9)
        assert a.rep_history[0].faults == []


# ── Clearly poor form must still be caught ───────────────────────────────────


def test_a_big_elbow_swing_is_still_caught():
    with session() as (a, clock):
        curl(a, clock, "right", drift=DRIFT_TOLERANCE_BODY * 3)
        assert "Elbow drifting forward" in a.rep_history[0].faults


def test_a_bounced_rep_is_still_caught():
    # Whole rep far quicker than TEMPO_MIN_SECONDS.
    with session() as (a, clock):
        curl(a, clock, "right", eccentric_s=0.0, peak_frames=0)
        assert a.rep_history[0].duration_sec < TEMPO_MIN_SECONDS
        assert any("fast" in f.lower() for f in a.rep_history[0].faults)


def test_heavy_torso_swing_is_still_caught():
    with session() as (a, clock):
        def frame(angle):
            return features(
                left_angle=170, right_angle=angle, lean=TORSO_TOLERANCE_DEG + 30
            )

        for angle in (170, 150, 120, 90, 60, 40):
            a.analyze_form(frame(angle), None)
            clock.tick()
        for _ in range(10):
            a.analyze_form(frame(40), None)
            clock.tick()
        for _ in range(int(0.9 * FPS)):
            clock.tick()
        for angle in (60, 90, 120, 150, 170):
            a.analyze_form(frame(angle), None)
            clock.tick()

        assert "Torso swinging" in a.rep_history[0].faults


def test_worse_form_always_scores_lower():
    """Scores must stay monotonic, or the number means nothing."""
    with session() as (a, clock):
        curl(a, clock, "right")
        clean = a.rep_history[0].quality
    with session() as (b, clock):
        curl(b, clock, "right", drift=DRIFT_TOLERANCE_BODY * 4)
        swung = b.rep_history[0].quality
    assert clean > swung
