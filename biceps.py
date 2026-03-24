"""
Mavis — Bicep Curl Analyzer
Inherits BaseAnalyzer for program mode, quality scoring, calibration, and voice.

Form thresholds use ±ANGLE_TOLERANCE_DEG bands so real human error
(imperfect camera angle, slight asymmetry) doesn't cause false faults.
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque

from analyzer.base_analyzer import BaseAnalyzer, Stage, CalibrationState, ANGLE_TOLERANCE_DEG, DRIFT_TOLERANCE_BODY, TEMPO_MIN_SECONDS
from analyzer.feature_extractor import FeatureExtractor, get_visibility_map, LM

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# ── Thresholds ────────────────────────────────────────────────────────────────
# Core angles — tolerance band of ±ANGLE_TOLERANCE_DEG is applied at check time
UP_ANGLE_IDEAL   = 45.0    # full peak contraction
DOWN_ANGLE_IDEAL = 160.0   # full extension

SEQUENCE_LENGTH       = 30
CONFIDENCE_THRESHOLD  = 0.7
MODELS_DIR            = os.path.join(os.path.dirname(__file__), '..', 'models')

# Rep quality scoring weights (must sum to 100)
QUALITY_WEIGHT_RANGE    = 40   # did they hit full ROM?
QUALITY_WEIGHT_TEMPO    = 25   # was it controlled (not too fast)?
QUALITY_WEIGHT_DRIFT    = 20   # was the elbow stable?
QUALITY_WEIGHT_TORSO    = 15   # did the torso stay upright?


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

        # AI classification
        self.lstm_model    = None
        self.scaler        = None
        self.label_encoder = None
        self.seq_buffer    = deque(maxlen=SEQUENCE_LENGTH)
        self.ai_label      = "Waiting..."
        self.ai_confidence = 0.0

        # Per-rep tracking
        self._peak_angle_this_rep  = 180.0   # lowest angle reached (best contraction)
        self._bottom_angle_this_rep = 0.0    # highest angle reached (best extension)
        self._max_drift_this_rep   = 0.0
        self._max_torso_lean_this_rep = 0.0

        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        if not _TF_AVAILABLE:
            print("[BicepAnalyzer] TensorFlow not available — geometry-only mode.")
            return
        try:
            model_path = os.path.join(MODELS_DIR, 'bicep_lstm.h5')
            scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
            le_path     = os.path.join(MODELS_DIR, 'label_encoder.pkl')

            self.lstm_model    = tf.keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(le_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("[BicepAnalyzer] Models loaded.")
        except Exception as e:
            print(f"[BicepAnalyzer] Model load failed: {e} — geometry-only mode.")

    # ── AI prediction ─────────────────────────────────────────────────────────

    def _predict(self, features: list) -> str:
        if self.lstm_model is None or self.scaler is None:
            return "Bicep Curl"

        scaled = self.scaler.transform([features])[0]
        self.seq_buffer.append(scaled)

        if len(self.seq_buffer) < SEQUENCE_LENGTH:
            return "Waiting..."

        seq  = np.array(self.seq_buffer)[np.newaxis, ...]
        pred = self.lstm_model.predict(seq, verbose=0)[0]
        idx  = int(np.argmax(pred))
        self.ai_confidence = float(pred[idx])

        if self.ai_confidence < CONFIDENCE_THRESHOLD:
            return "Unknown"

        return self.label_encoder.inverse_transform([idx])[0]

    # ── Form analysis ─────────────────────────────────────────────────────────

    def analyze_form(self, features: list, landmarks) -> None:
        """
        State machine for bicep curl rep counting and form analysis.
        Tolerances applied: ±ANGLE_TOLERANCE_DEG on all angle thresholds.
        Body-relative elbow drift used (not screen pixels).
        """
        elbow_angle = features[1]          # left elbow angle (degrees)
        drift       = features[0]          # body-relative elbow drift
        torso_lean  = features[6]          # degrees from vertical

        # Track per-rep extremes for quality scoring
        self._peak_angle_this_rep  = min(self._peak_angle_this_rep,  elbow_angle)
        self._bottom_angle_this_rep = max(self._bottom_angle_this_rep, elbow_angle)
        self._max_drift_this_rep   = max(self._max_drift_this_rep,   drift)
        self._max_torso_lean_this_rep = max(self._max_torso_lean_this_rep, torso_lean)

        # ── DOWN detection: arm fully extended ───────────────────────────────
        if elbow_angle > (DOWN_ANGLE_IDEAL - ANGLE_TOLERANCE_DEG):
            if self.stage == Stage.UP:
                # Rep completed — going back down
                duration = time.time() - self.rep_start_time
                if duration < TEMPO_MIN_SECONDS:
                    self._add_fault("Too fast — control the negative", severity=25)
                rep = self._complete_rep()
                self._reset_rep_tracking()
                self.stage    = Stage.DOWN
                self.form_color = (0, 255, 100)
                self.feedback   = f"Rep {rep.rep_number} — {rep.quality}/100"
            else:
                self.stage      = Stage.DOWN
                self.feedback   = "Curl up"
                self.form_color = (0, 255, 100)

        # ── UP detection: peak contraction reached ────────────────────────────
        elif elbow_angle < (UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG):
            if self.stage == Stage.DOWN:
                self.stage = Stage.UP
                self._begin_rep()
                self.feedback = "Squeeze — lower slowly"

            # Check form faults while in UP or moving through mid-range
            if drift > DRIFT_TOLERANCE_BODY:
                self._add_fault("Fix elbow — don't let it drift", severity=20)
                self.feedback = "Elbow drifting — pin it to your side"

            if torso_lean > 20.0 + ANGLE_TOLERANCE_DEG:
                self._add_fault("Torso swinging", severity=15)
                self.feedback = "No swinging — keep torso upright"

        # ── MID-RANGE guidance ────────────────────────────────────────────────
        else:
            if self.stage == Stage.DOWN:
                self.feedback = "Curl up — full range"
            elif self.stage == Stage.UP:
                self.feedback = "Lower slowly"

    def _reset_rep_tracking(self) -> None:
        self._peak_angle_this_rep     = 180.0
        self._bottom_angle_this_rep   = 0.0
        self._max_drift_this_rep      = 0.0
        self._max_torso_lean_this_rep = 0.0
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
                            # ── Feature extraction & AI ───────────────────────
                            features   = self.extractor.get_features(lms)
                            self.ai_label = self._predict(features)

                            if self.ai_label in ("Bicep Curl", "Waiting...", "Unknown") or self.lstm_model is None:
                                self.analyze_form(features, lms)
                            else:
                                self.feedback   = "Wrong exercise detected"
                                self.form_color = (0, 60, 255)

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

        # AI confidence
        cv2.putText(image, f"AI: {self.ai_confidence:.2f}", (w - 120, 58), font, 0.5, (180, 180, 180), 1)

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
