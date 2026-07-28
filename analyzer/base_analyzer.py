"""
Mavis Base Analyzer
Shared logic for all exercise analyzers:
  - Calibration phase
  - Workout program mode (sets × reps with wrong-set detection)
  - Rep quality scoring (0–100) with tolerance bands
  - Session history for the trend graph
  - Landmark visibility warnings

Every exercise analyzer (BicepAnalyzer, ShoulderAnalyzer, etc.) inherits from this.
"""

import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from analyzer.exercise_config import calibration, tolerances
from analyzer.feature_extractor import LM


# ── Tolerance constants ──────────────────────────────────────────────────────
# Sourced from exercises.json so the web engine reads the identical values.
# Real humans never hit perfect angles; these bands define the acceptable
# deviation from ideal before a form fault is raised.
_TOL = tolerances()
_CALIB = calibration()

ANGLE_TOLERANCE_DEG    = _TOL["angleDeg"]          # ±° acceptable on all angle thresholds
DRIFT_TOLERANCE_BODY   = _TOL["driftBodyRatio"]    # elbow drift as a fraction of shoulder width
SYMMETRY_TOLERANCE     = _TOL["symmetryDeg"]       # acceptable bilateral asymmetry
TEMPO_MIN_SECONDS      = _TOL["tempoMinSeconds"]   # eccentric faster than this is flagged
VISIBILITY_WARN_FRAMES = _TOL["visibilityWarnFrames"]

# Calibration must be held genuinely still: if key points move more than this
# (normalised coordinate units) between frames, the countdown restarts rather
# than completing against whatever pose happens to be held at the deadline.
CALIB_STABILITY_TOLERANCE = _CALIB["stabilityTolerance"]
CALIB_HOLD_SECONDS        = _CALIB["holdSeconds"]
_INDEX_TO_NAME = {v: k for k, v in LM.items()}
CALIB_KEY_POINTS = tuple(_INDEX_TO_NAME[i] for i in _CALIB["keyPoints"])

# A fault condition must hold for this many consecutive frames before it is
# recorded. Per-rep extremes are tracked with max()/min(), which by
# construction latch onto the single worst frame in the rep — so without
# debouncing, one noisy landmark estimate is enough to fault an otherwise
# clean rep. Measured against real training footage, instantaneous spikes were
# the dominant source of false faults.
FAULT_CONFIRM_FRAMES = _TOL["faultConfirmFrames"]


def graded_penalty(excess: float, ramp: float, weight: int) -> int:
    """
    Scale a deduction by how badly the athlete actually missed.

    A binary threshold punishes missing by 1° exactly as hard as missing by
    40°, which makes scores feel arbitrary and unforgiving. This returns no
    penalty while within tolerance, then ramps linearly to the full weight once
    `ramp` units past it.
    """
    if excess <= 0:
        return 0
    return int(round(weight * min(1.0, excess / ramp)))


class Stage(Enum):
    IDLE   = "idle"
    DOWN   = "down"
    UP     = "up"
    HOLD   = "hold"


class CalibrationState(Enum):
    NEEDED    = "needed"
    RUNNING   = "running"
    COMPLETE  = "complete"


@dataclass
class RepResult:
    """All data captured for a single completed rep."""
    rep_number:   int
    quality:      int        # 0–100
    duration_sec: float
    faults:       list       # list of fault strings
    timestamp:    float = field(default_factory=time.time)


@dataclass
class SetResult:
    """All data captured for a completed set."""
    set_number:    int
    target_reps:   int
    reps:          list      # list[RepResult]
    valid:         bool      # True only if all reps were good quality (≥ threshold)
    rest_start:    float = field(default_factory=time.time)


@dataclass
class WorkoutProgram:
    """
    Describes a structured workout session.
    Wrong-set detection: if a set has too many bad reps the set is marked
    invalid and must be repeated.
    """
    exercise_name: str
    sets:          int
    reps_per_set:  int
    rest_seconds:  int   = 60
    quality_threshold: int = 60   # min quality score to count a rep as valid
    bad_rep_limit: int  = 3       # bad reps in a set before the set is invalidated


class BaseAnalyzer:
    """
    Abstract base. Subclasses must implement:
        analyze_form(features, landmarks) → None
        _score_rep()                       → int  (0–100)
    """

    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name

        # ── state ────────────────────────────────────────────────────────────
        self.stage            = Stage.IDLE
        self.calib_state      = CalibrationState.NEEDED
        self.calib_countdown  = CALIB_HOLD_SECONDS   # seconds of stable pose required
        self.calib_start_time = 0.0
        self._calib_ref_points: Optional[list] = None   # key points from the previous calibration frame

        # ── counters ─────────────────────────────────────────────────────────
        self.total_reps   = 0
        self.good_reps    = 0
        self.bad_reps     = 0
        self.current_set  = 0          # 0 = free mode (no program)

        # ── rep timing ───────────────────────────────────────────────────────
        # rep_start_time marks the top of the movement (start of the eccentric).
        # _rep_cycle_start marks when the athlete first left the bottom, so a
        # rep's reported duration covers the whole cycle rather than the
        # lowering phase alone.
        self.rep_start_time   = 0.0
        self._rep_cycle_start = 0.0

        # ── fault debouncing ─────────────────────────────────────────────────
        self._fault_streaks: dict = {}

        # ── quality history for trend graph ──────────────────────────────────
        self.rep_history: list[RepResult] = []

        # ── program mode ─────────────────────────────────────────────────────
        self.program: Optional[WorkoutProgram] = None
        self.set_history: list[SetResult] = []
        self.current_set_reps: list[RepResult] = []
        self.resting         = False
        self.rest_start_time = 0.0
        self.program_complete = False

        # ── current rep state ─────────────────────────────────────────────────
        self.current_faults: list[str] = []
        self.feedback       = f"Ready — {exercise_name}"
        self.form_color     = (0, 255, 100)   # BGR green
        self.form_quality   = 100             # live quality estimate 0–100

        # ── visibility ───────────────────────────────────────────────────────
        self.visibility_map:   dict = {}
        self.visibility_warn:  str  = ""
        self._low_vis_counter: int  = 0

    # ── Calibration ──────────────────────────────────────────────────────────

    def start_calibration(self) -> None:
        self.calib_state      = CalibrationState.RUNNING
        self.calib_start_time = time.time()
        self._calib_ref_points = None

    def tick_calibration(self, extractor, landmarks) -> bool:
        """
        Call every frame during calibration phase.
        Restarts the countdown if the pose moves more than
        CALIB_STABILITY_TOLERANCE between frames, so calibration can't
        complete against a pose that was only held for an instant (e.g.
        mid-curl) rather than genuinely stood still for calib_countdown.
        Returns True when calibration is complete.
        """
        key_points = [(landmarks[LM[name]].x, landmarks[LM[name]].y) for name in CALIB_KEY_POINTS]

        if self._calib_ref_points is not None:
            moved = max(
                math.hypot(kx - rx, ky - ry)
                for (kx, ky), (rx, ry) in zip(key_points, self._calib_ref_points)
            )
            if moved > CALIB_STABILITY_TOLERANCE:
                self.calib_start_time = time.time()

        self._calib_ref_points = key_points

        elapsed = time.time() - self.calib_start_time
        remaining = self.calib_countdown - elapsed
        self.feedback = f"Stand tall — hold still ({remaining:.1f}s)"

        if elapsed >= self.calib_countdown:
            extractor.calibrate(landmarks)
            self.calib_state = CalibrationState.COMPLETE
            self.feedback    = "Calibrated! Begin when ready."
            return True
        return False

    # ── Program Mode ─────────────────────────────────────────────────────────

    def set_program(self, sets: int, reps: int, rest: int = 60) -> None:
        self.program = WorkoutProgram(
            exercise_name     = self.exercise_name,
            sets              = sets,
            reps_per_set      = reps,
            rest_seconds      = rest,
        )
        self.current_set        = 1
        self.set_history        = []
        self.current_set_reps   = []
        self.program_complete   = False
        self.resting            = False
        self.feedback           = f"Set 1 of {sets} — do {reps} reps"

    def _check_program_progress(self, rep: RepResult) -> None:
        """
        Called after every completed rep in program mode.
        Handles set completion, wrong-set detection, rest periods, and
        program completion. A set is invalid if too many bad reps occurred.
        """
        if not self.program or self.program_complete:
            return

        self.current_set_reps.append(rep)

        reps_done       = len(self.current_set_reps)
        target          = self.program.reps_per_set
        bad_in_set      = sum(1 for r in self.current_set_reps if r.quality < self.program.quality_threshold)
        p               = self.program

        # ── wrong-set detection: too many bad reps mid-set ───────────────────
        if bad_in_set >= p.bad_rep_limit and reps_done < target:
            self.feedback = f"⚠ Too many bad reps — focus on form!"

        # ── set complete ──────────────────────────────────────────────────────
        if reps_done >= target:
            set_valid = bad_in_set < p.bad_rep_limit

            result = SetResult(
                set_number  = self.current_set,
                target_reps = target,
                reps        = list(self.current_set_reps),
                valid       = set_valid,
            )
            self.set_history.append(result)
            self.current_set_reps = []

            if not set_valid:
                # Repeat this set
                self.feedback = f"Set {self.current_set} had too many bad reps — repeat it!"
            else:
                # Advance to next set
                if self.current_set >= p.sets:
                    self.program_complete = True
                    self.feedback = "🏆 Workout complete!"
                else:
                    self.current_set += 1
                    self.resting          = True
                    self.rest_start_time  = time.time()
                    self.feedback = f"Set done! Rest {p.rest_seconds}s"

    def tick_rest(self) -> bool:
        """
        Call every frame while resting.
        Returns True when rest is over.
        """
        if not self.resting or not self.program:
            return False

        elapsed   = time.time() - self.rest_start_time
        remaining = max(0, self.program.rest_seconds - elapsed)
        self.feedback = f"Rest — {remaining:.0f}s  |  Next: Set {self.current_set} of {self.program.sets}"

        if remaining <= 0:
            self.resting = False
            self.feedback = f"Set {self.current_set} of {self.program.sets} — {self.program.reps_per_set} reps"
            return True
        return False

    # ── Rep lifecycle ─────────────────────────────────────────────────────────

    def _begin_rep(self) -> None:
        self.rep_start_time  = time.time()
        self.current_faults  = []
        self.form_quality    = 100

    def mark_cycle_start(self) -> None:
        """
        Called on the first frame the athlete leaves the bottom position.

        This is the true start of a rep. rep_start_time, by contrast, is only
        set once the top is reached, so timing from it measures the eccentric
        alone — which used to be compared against a whole-rep tempo threshold
        and reported as the rep duration.
        """
        if self._rep_cycle_start == 0.0:
            self._rep_cycle_start = time.time()

    def rep_duration(self) -> float:
        """Elapsed time for the current rep, covering the full cycle."""
        start = self._rep_cycle_start or self.rep_start_time
        return time.time() - start if start else 0.0

    def confirm_fault(self, key: str, active: bool) -> bool:
        """
        Debounce a fault condition.

        Returns True only once the condition has held for
        FAULT_CONFIRM_FRAMES consecutive frames, so a single noisy landmark
        estimate can't fault an otherwise clean rep.
        """
        if not active:
            self._fault_streaks[key] = 0
            return False
        streak = self._fault_streaks.get(key, 0) + 1
        self._fault_streaks[key] = streak
        return streak >= FAULT_CONFIRM_FRAMES

    def _reset_fault_streaks(self) -> None:
        self._fault_streaks.clear()

    def _complete_rep(self) -> RepResult:
        """
        Finalise a rep, score it, store it, and update program state.
        """
        duration = self.rep_duration()
        quality  = self._score_rep()
        self._rep_cycle_start = 0.0

        self.total_reps += 1

        if quality >= (self.program.quality_threshold if self.program else 60):
            self.good_reps += 1
        else:
            self.bad_reps += 1

        rep = RepResult(
            rep_number   = self.total_reps,
            quality      = quality,
            duration_sec = duration,
            faults       = list(self.current_faults),
        )
        self.rep_history.append(rep)

        if self.program:
            self._check_program_progress(rep)

        return rep

    def _add_fault(self, fault: str, severity: int = 10) -> None:
        """
        Record a form fault during a rep and deduct from live quality.
        severity: points to deduct (10–40 depending on how bad the fault is).
        """
        if fault not in self.current_faults:
            self.current_faults.append(fault)
        self.form_quality = max(0, self.form_quality - severity)
        self.form_color   = (0, 60, 255)   # BGR red

    def _score_rep(self) -> int:
        """
        Default scorer: deduct from 100 based on faults accumulated this rep.
        Subclasses may override to add exercise-specific scoring dimensions
        (e.g. depth, tempo, symmetry).
        """
        return max(0, self.form_quality)

    # ── Visibility ────────────────────────────────────────────────────────────

    def update_visibility(self, visibility_map: dict, required_regions: list) -> None:
        """
        Sets self.visibility_warn if any required body region is occluded
        for VISIBILITY_WARN_FRAMES consecutive frames.
        """
        self.visibility_map = visibility_map
        occluded = [r for r in required_regions if not visibility_map.get(r, True)]

        if occluded:
            self._low_vis_counter += 1
        else:
            self._low_vis_counter = 0
            self.visibility_warn  = ""

        if self._low_vis_counter >= VISIBILITY_WARN_FRAMES:
            region = occluded[0].replace("_", " ")
            self.visibility_warn = f"Can't see your {region} — adjust camera"

    # ── Session summary ───────────────────────────────────────────────────────

    def get_session_summary(self) -> dict:
        """Returns a dict suitable for JSON serialisation to the summary screen."""
        avg_quality = (
            sum(r.quality for r in self.rep_history) / len(self.rep_history)
            if self.rep_history else 0
        )
        most_common_fault = ""
        if self.rep_history:
            from collections import Counter
            all_faults = [f for r in self.rep_history for f in r.faults]
            if all_faults:
                most_common_fault = Counter(all_faults).most_common(1)[0][0]

        return {
            "exercise":          self.exercise_name,
            "total_reps":        self.total_reps,
            "good_reps":         self.good_reps,
            "bad_reps":          self.bad_reps,
            "avg_quality":       round(avg_quality),
            "most_common_fault": most_common_fault,
            "rep_qualities":     [r.quality for r in self.rep_history],
            "rep_durations":     [round(r.duration_sec, 2) for r in self.rep_history],
            "sets_completed":    len([s for s in self.set_history if s.valid]),
            "sets_repeated":     len([s for s in self.set_history if not s.valid]),
            "program_complete":  self.program_complete,
        }
