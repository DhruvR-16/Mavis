"""
Mavis — Bicep Curl Analyzer
Inherits BaseAnalyzer for program mode, quality scoring, calibration, and voice.

Form thresholds use ±ANGLE_TOLERANCE_DEG bands so real human error
(imperfect camera angle, slight asymmetry) doesn't cause false faults.
"""

import cv2
import mediapipe as mp
import time

from analyzer.base_analyzer import BaseAnalyzer, Stage, CalibrationState, ANGLE_TOLERANCE_DEG, DRIFT_TOLERANCE_BODY, TEMPO_MIN_SECONDS
from analyzer.exercise_config import exercise
from analyzer.feature_extractor import FeatureExtractor, get_visibility_map, LM

# ── Thresholds ────────────────────────────────────────────────────────────────
# All sourced from exercises.json — see analyzer/exercise_config.py.
# Tolerance band of ±ANGLE_TOLERANCE_DEG is applied at check time.
_CFG = exercise("bicep")

UP_ANGLE_IDEAL   = _CFG["thresholds"]["up"]     # full peak contraction
DOWN_ANGLE_IDEAL = _CFG["thresholds"]["down"]   # full extension

# Rep quality scoring weights (must sum to 100)
QUALITY_WEIGHT_RANGE = _CFG["scoring"]["range"]   # did they hit full ROM?
QUALITY_WEIGHT_TEMPO = _CFG["scoring"]["tempo"]   # was it controlled (not too fast)?
QUALITY_WEIGHT_DRIFT = _CFG["scoring"]["drift"]   # was the elbow stable?
QUALITY_WEIGHT_TORSO = _CFG["scoring"]["torso"]   # did the torso stay upright?

_FAULTS = _CFG["faults"]
TORSO_TOLERANCE_DEG = _FAULTS["torso"]["toleranceDeg"]


class LandmarkSmoother:
    """Exponential moving average on landmark positions to reduce jitter."""

    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self.prev  = None

    def smooth(self, landmarks):
        if self.prev is None:
            self.prev = landmarks
            return landmarks

        class _Pt:
            __slots__ = ("x", "y", "z", "visibility")
            def __init__(self, x, y, z, v):
                self.x, self.y, self.z, self.visibility = x, y, z, v

        a, b = self.alpha, 1.0 - self.alpha
        smoothed = [
            _Pt(
                a * lm.x + b * self.prev[i].x,
                a * lm.y + b * self.prev[i].y,
                a * lm.z + b * self.prev[i].z,
                a * lm.visibility + b * self.prev[i].visibility,
            )
            for i, lm in enumerate(landmarks)
        ]
        self.prev = smoothed
        return smoothed


class BicepAnalyzer(BaseAnalyzer):

    def __init__(self):
        super().__init__("Bicep Curl")

        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.extractor  = FeatureExtractor()
        self.smoother   = LandmarkSmoother(alpha=0.55)

        # Per-rep tracking
        self._peak_angle_this_rep  = 180.0   # lowest angle reached (best contraction)
        self._bottom_angle_this_rep = 0.0    # highest angle reached (best extension)
        self._max_drift_this_rep   = 0.0
        self._max_torso_lean_this_rep = 0.0
        self._fault_seen: set = set()        # tracks which faults already fired this rep

        # Which arm is curling. Re-evaluated while at rest (not mid-rep) and
        # latched for the duration of a rep so it can't flip mid-curl if the
        # two elbow angles happen to cross due to landmark noise.
        self._active_side = "left"

    # ── Form analysis ─────────────────────────────────────────────────────────

    def analyze_form(self, features: list, landmarks) -> None:
        """
        State machine for bicep curl rep counting and form analysis.

        Key design decisions:
        - Active arm (left/right) is picked from whichever elbow is more
          contracted while at rest, then latched for the rep — see
          _active_side. Tracking only the left arm meant a right-arm curl
          counted zero reps.
        - Drift is ONLY checked at peak contraction (elbow < UP_THRESH).
          During the lifting phase the elbow naturally moves — checking it
          mid-range causes constant false positives.
        - Faults are recorded at most ONCE per rep (tracked via _fault_seen)
          so the quality score isn't hammered 30× per second.
        - Torso lean tolerance is 25° (not 20°) — humans naturally micro-sway.
        - Tolerance band of ±ANGLE_TOLERANCE_DEG applied on all thresholds.
        """
        left_angle, right_angle = features[1], features[2]
        left_drift, right_drift = features[0], features[10]
        torso_lean = features[6]   # bilateral — same regardless of active arm

        if self.stage != Stage.UP:
            self._active_side = "left" if left_angle <= right_angle else "right"

        if self._active_side == "left":
            elbow_angle, drift = left_angle, left_drift
        else:
            elbow_angle, drift = right_angle, right_drift

        # Track per-rep extremes for quality scoring
        self._peak_angle_this_rep     = min(self._peak_angle_this_rep,  elbow_angle)
        self._bottom_angle_this_rep   = max(self._bottom_angle_this_rep, elbow_angle)
        self._max_drift_this_rep      = max(self._max_drift_this_rep,   drift)
        self._max_torso_lean_this_rep = max(self._max_torso_lean_this_rep, torso_lean)

        # ── DOWN: arm fully extended ─────────────────────────────────────────
        if elbow_angle > (DOWN_ANGLE_IDEAL - ANGLE_TOLERANCE_DEG):
            if self.stage == Stage.UP:
                duration = time.time() - self.rep_start_time
                if duration < TEMPO_MIN_SECONDS:
                    self._add_fault(_FAULTS["tempo"]["message"],
                                    severity=_FAULTS["tempo"]["severity"])
                rep = self._complete_rep()
                self._reset_rep_tracking()
                self.stage      = Stage.DOWN
                self.form_color = (0, 255, 100)
                self.feedback   = f"Rep {rep.rep_number} — {rep.quality}/100"
            else:
                if self.stage != Stage.DOWN:
                    self.stage = Stage.DOWN
                self.feedback = "Curl up — full range"

        # ── UP: peak contraction ─────────────────────────────────────────────
        elif elbow_angle < (UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG):
            if self.stage == Stage.DOWN:
                self.stage = Stage.UP
                self._begin_rep()
                self.feedback = "Squeeze — lower slowly"

            # Drift only meaningful at peak — elbow forward away from body
            # Only flag once per rep to avoid spamming the quality score
            if drift > DRIFT_TOLERANCE_BODY and "drift" not in self._fault_seen:
                self._fault_seen.add("drift")
                self._add_fault(_FAULTS["drift"]["message"],
                                severity=_FAULTS["drift"]["severity"])
                self.feedback = "Pin elbow to your side"

            # Torso swing — tolerance is wider than the angle band because real
            # human micro-sway is ≤15°
            elif torso_lean > TORSO_TOLERANCE_DEG and "torso" not in self._fault_seen:
                self._fault_seen.add("torso")
                self._add_fault(_FAULTS["torso"]["message"],
                                severity=_FAULTS["torso"]["severity"])
                self.feedback = "Keep torso still — no swinging"

            elif self.stage == Stage.UP and not self._fault_seen:
                self.feedback = "Lower slowly"

        # ── MID-RANGE: between down and up ───────────────────────────────────
        else:
            if self.stage == Stage.DOWN:
                self.feedback = "Curl up"
            elif self.stage == Stage.UP:
                self.feedback = "Lower slowly"

    def _reset_rep_tracking(self) -> None:
        self._peak_angle_this_rep     = 180.0
        self._bottom_angle_this_rep   = 0.0
        self._max_drift_this_rep      = 0.0
        self._max_torso_lean_this_rep = 0.0
        self._fault_seen              = set()
        self.form_quality             = 100
        self.form_color               = (0, 255, 100)

    # ── Quality scoring ────────────────────────────────────────────────────────

    def _score_rep(self) -> int:
        score = 100

        # Range of motion: did they reach near-full contraction and extension?
        peak_bonus = max(0, (UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG) - self._peak_angle_this_rep)
        # peak_bonus > 0 means they went deep enough — if not, deduct
        if self._peak_angle_this_rep > (UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG):
            score -= QUALITY_WEIGHT_RANGE    # no contraction reached

        if self._bottom_angle_this_rep < (DOWN_ANGLE_IDEAL - ANGLE_TOLERANCE_DEG * 2):
            score -= int(QUALITY_WEIGHT_RANGE * 0.5)  # partial deduction for partial extension

        # Tempo
        duration = time.time() - self.rep_start_time
        if duration < TEMPO_MIN_SECONDS:
            score -= QUALITY_WEIGHT_TEMPO

        # Elbow drift
        if self._max_drift_this_rep > DRIFT_TOLERANCE_BODY:
            excess = (self._max_drift_this_rep - DRIFT_TOLERANCE_BODY) / DRIFT_TOLERANCE_BODY
            score -= min(QUALITY_WEIGHT_DRIFT, int(QUALITY_WEIGHT_DRIFT * excess))

        # Torso lean
        if self._max_torso_lean_this_rep > 20.0 + ANGLE_TOLERANCE_DEG:
            score -= QUALITY_WEIGHT_TORSO

        return max(0, min(100, score))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)

        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                h, w, _ = image.shape

                if results.pose_landmarks:
                    raw = results.pose_landmarks.landmark
                    lms = self.smoother.smooth(raw)

                    # ── Calibration phase ─────────────────────────────────────
                    if self.calib_state == CalibrationState.NEEDED:
                        self.start_calibration()

                    if self.calib_state == CalibrationState.RUNNING:
                        self.tick_calibration(self.extractor, lms)
                    else:
                        # ── Visibility check ──────────────────────────────────
                        vis_map = get_visibility_map(lms)
                        self.update_visibility(vis_map, ["left_arm", "right_arm"])

                        # ── Rest period ───────────────────────────────────────
                        if self.resting:
                            self.tick_rest()
                        else:
                            # ── Feature extraction ────────────────────────────
                            features = self.extractor.get_features(lms)
                            self.analyze_form(features, lms)

                    # ── Draw skeleton ─────────────────────────────────────────
                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=self.form_color, thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=self.form_color, thickness=2),
                    )

                # ── HUD overlay ───────────────────────────────────────────────
                self._draw_hud(image, w, h)
                cv2.imshow("Mavis — Bicep Curl", image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary()

    def _draw_hud(self, image, w: int, h: int) -> None:
        """Minimal but informative HUD overlay on the CV window."""
        font   = cv2.FONT_HERSHEY_SIMPLEX
        white  = (255, 255, 255)
        green  = (0, 255, 100)
        red    = (0, 60, 255)
        yellow = (0, 220, 255)

        # Background strip top
        cv2.rectangle(image, (0, 0), (w, 80), (15, 15, 20), -1)

        cv2.putText(image, "MAVIS — BICEP CURL", (12, 28), font, 0.7, yellow, 2)
        cv2.putText(image, f"Reps: {self.total_reps}  Good: {self.good_reps}  Bad: {self.bad_reps}", (12, 58), font, 0.65, white, 1)

        # Program mode indicator
        if self.program:
            sets_info = f"Set {self.current_set}/{self.program.sets}"
            cv2.putText(image, sets_info, (w - 200, 28), font, 0.65, green, 2)

        # Feedback bar bottom
        cv2.rectangle(image, (0, h - 55), (w, h), (15, 15, 20), -1)
        color = green if self.form_color == (0, 255, 100) else red
        cv2.putText(image, self.feedback, (12, h - 20), font, 0.85, color, 2)

        # Visibility warning
        if self.visibility_warn:
            cv2.putText(image, f"⚠ {self.visibility_warn}", (12, h - 65), font, 0.6, yellow, 2)

    def _print_summary(self) -> None:
        summary = self.get_session_summary()
        print("\n" + "="*50)
        print("MAVIS SESSION SUMMARY — BICEP CURL")
        print("="*50)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("="*50)


if __name__ == "__main__":
    app = BicepAnalyzer()
    app.run()