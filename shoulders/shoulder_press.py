"""
Mavis — Shoulder Press Analyzer
Tracks bilateral press symmetry, lockout validation, and controlled negative.
Inherits all program mode, quality scoring, calibration from BaseAnalyzer.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pickle
from collections import deque

from analyzer.base_analyzer import (
    BaseAnalyzer, Stage, CalibrationState,
    ANGLE_TOLERANCE_DEG, SYMMETRY_TOLERANCE, TEMPO_MIN_SECONDS
)
from analyzer.feature_extractor import FeatureExtractor, get_visibility_map, LM

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# ── Thresholds ────────────────────────────────────────────────────────────────
PRESS_TOP_IDEAL    = 165.0   # near-full lockout at top
PRESS_BOTTOM_IDEAL = 75.0    # elbows below shoulder height at bottom
SEQUENCE_LENGTH    = 30
CONFIDENCE_THRESHOLD = 0.7
MODELS_DIR         = os.path.join(os.path.dirname(__file__), '..', 'models')

# Quality weights
QUALITY_WEIGHT_LOCKOUT   = 35
QUALITY_WEIGHT_DEPTH     = 30
QUALITY_WEIGHT_SYMMETRY  = 20
QUALITY_WEIGHT_TEMPO     = 15


class LandmarkSmoother:
    def __init__(self, alpha=0.55):
        self.alpha = alpha
        self.prev  = None

    def smooth(self, landmarks):
        if self.prev is None:
            self.prev = landmarks
            return landmarks

        class _Pt:
            __slots__ = ("x","y","z","visibility")
            def __init__(self,x,y,z,v):
                self.x,self.y,self.z,self.visibility=x,y,z,v

        a,b = self.alpha, 1-self.alpha
        s = [_Pt(a*lm.x+b*self.prev[i].x, a*lm.y+b*self.prev[i].y,
                 a*lm.z+b*self.prev[i].z, a*lm.visibility+b*self.prev[i].visibility)
             for i,lm in enumerate(landmarks)]
        self.prev = s
        return s


class ShoulderAnalyzer(BaseAnalyzer):

    def __init__(self):
        super().__init__("Shoulder Press")

        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.extractor  = FeatureExtractor()
        self.smoother   = LandmarkSmoother()

        # Per-rep tracking
        self._max_left_angle     = 0.0
        self._max_right_angle    = 0.0
        self._min_left_angle     = 180.0
        self._min_right_angle    = 180.0
        self._max_asymmetry      = 0.0
        self._fault_seen: set    = set()   # prevents per-frame fault spam

        # AI (same binary model — future: retrain as multi-class)
        self.lstm_model    = None
        self.scaler        = None
        self.label_encoder = None
        self.seq_buffer    = deque(maxlen=SEQUENCE_LENGTH)
        self.ai_confidence = 0.0

        self._load_models()

    def _load_models(self):
        if not _TF_AVAILABLE:
            return
        try:
            self.lstm_model    = tf.keras.models.load_model(os.path.join(MODELS_DIR,'bicep_lstm.h5'))
            with open(os.path.join(MODELS_DIR,'scaler.pkl'),'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(MODELS_DIR,'label_encoder.pkl'),'rb') as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            print(f"[ShoulderAnalyzer] Model load failed: {e}")

    def analyze_form(self, features: list, landmarks) -> None:
        """
        State machine for shoulder press.
        Angle: elbow angle (shoulder→elbow→wrist).
        ~170° = lockout overhead. ~75° = bottom of press.
        Fault-seen set prevents per-frame quality spam.
        """
        from analyzer.feature_extractor import calculate_angle, LM

        ls = landmarks[LM["left_shoulder"]]
        rs = landmarks[LM["right_shoulder"]]
        le = landmarks[LM["left_elbow"]]
        re = landmarks[LM["right_elbow"]]
        lw = landmarks[LM["left_wrist"]]
        rw = landmarks[LM["right_wrist"]]

        left_angle  = calculate_angle(ls, le, lw)
        right_angle = calculate_angle(rs, re, rw)
        avg_angle   = (left_angle + right_angle) / 2.0
        symmetry    = abs(left_angle - right_angle)

        self._max_left_angle  = max(self._max_left_angle,  left_angle)
        self._max_right_angle = max(self._max_right_angle, right_angle)
        self._min_left_angle  = min(self._min_left_angle,  left_angle)
        self._min_right_angle = min(self._min_right_angle, right_angle)
        self._max_asymmetry   = max(self._max_asymmetry,   symmetry)

        # ── BOTTOM: elbows bent, bottom of press ─────────────────────────────
        if avg_angle < (PRESS_BOTTOM_IDEAL + ANGLE_TOLERANCE_DEG):
            if self.stage == Stage.UP:
                duration = time.time() - self.rep_start_time
                if duration < TEMPO_MIN_SECONDS:
                    self._add_fault("Too fast — control the descent", severity=20)
                rep = self._complete_rep()
                self._reset_rep_tracking()
                self.stage      = Stage.DOWN
                self.form_color = (0, 255, 100)
                self.feedback   = f"Rep {rep.rep_number} — {rep.quality}/100"
            else:
                if self.stage != Stage.DOWN:
                    self.stage = Stage.DOWN
                self.feedback = "Press overhead"

        # ── TOP: arms near lockout ────────────────────────────────────────────
        elif avg_angle > (PRESS_TOP_IDEAL - ANGLE_TOLERANCE_DEG):
            if self.stage == Stage.DOWN:
                self.stage = Stage.UP
                self._begin_rep()
                self.feedback = "Lower with control"

            if symmetry > SYMMETRY_TOLERANCE + ANGLE_TOLERANCE_DEG \
                    and "asym" not in self._fault_seen:
                self._fault_seen.add("asym")
                self._add_fault("Uneven press — balance both arms", severity=20)
                self.feedback = "Left-right imbalance — press evenly"
            elif not self._fault_seen:
                self.feedback = "Lower with control"

        # ── MID-RANGE ─────────────────────────────────────────────────────────
        else:
            if self.stage == Stage.DOWN:
                self.feedback = "Press up — full range"
            elif self.stage == Stage.UP:
                if symmetry > SYMMETRY_TOLERANCE + ANGLE_TOLERANCE_DEG \
                        and "asym" not in self._fault_seen:
                    self._fault_seen.add("asym")
                    self._add_fault("Uneven press", severity=15)
                    self.feedback = "Even out both arms"
                else:
                    self.feedback = "Lower slowly"

    def _reset_rep_tracking(self):
        self._max_left_angle  = 0.0
        self._max_right_angle = 0.0
        self._min_left_angle  = 180.0
        self._min_right_angle = 180.0
        self._max_asymmetry   = 0.0
        self._fault_seen      = set()
        self.form_quality     = 100
        self.form_color       = (0, 255, 100)

    def _score_rep(self) -> int:
        score = 100

        # Lockout quality
        avg_top = (self._max_left_angle + self._max_right_angle) / 2
        if avg_top < (PRESS_TOP_IDEAL - ANGLE_TOLERANCE_DEG):
            score -= QUALITY_WEIGHT_LOCKOUT

        # Depth
        avg_bottom = (self._min_left_angle + self._min_right_angle) / 2
        if avg_bottom > (PRESS_BOTTOM_IDEAL + ANGLE_TOLERANCE_DEG):
            score -= QUALITY_WEIGHT_DEPTH

        # Symmetry
        if self._max_asymmetry > SYMMETRY_TOLERANCE + ANGLE_TOLERANCE_DEG:
            excess = (self._max_asymmetry - SYMMETRY_TOLERANCE) / SYMMETRY_TOLERANCE
            score -= min(QUALITY_WEIGHT_SYMMETRY, int(QUALITY_WEIGHT_SYMMETRY * excess))

        # Tempo
        duration = time.time() - self.rep_start_time
        if duration < TEMPO_MIN_SECONDS:
            score -= QUALITY_WEIGHT_TEMPO

        return max(0, min(100, score))

    def run(self):
        cap = cv2.VideoCapture(0)

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

                    if self.calib_state == CalibrationState.NEEDED:
                        self.start_calibration()

                    if self.calib_state == CalibrationState.RUNNING:
                        self.tick_calibration(self.extractor, lms)
                    else:
                        vis_map = get_visibility_map(lms)
                        self.update_visibility(vis_map, ["left_arm", "right_arm", "shoulders"])

                        if self.resting:
                            self.tick_rest()
                        else:
                            features = self.extractor.get_features(lms)
                            self.analyze_form(features, lms)

                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=self.form_color, thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=self.form_color, thickness=2),
                    )

                self._draw_hud(image, w, h)
                cv2.imshow("Mavis — Shoulder Press", image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def _draw_hud(self, image, w, h):
        font   = cv2.FONT_HERSHEY_SIMPLEX
        white  = (255, 255, 255)
        green  = (0, 255, 100)
        red    = (0, 60, 255)
        yellow = (0, 220, 255)

        cv2.rectangle(image, (0, 0), (w, 80), (15, 15, 20), -1)
        cv2.putText(image, "MAVIS — SHOULDER PRESS", (12, 28), font, 0.7, yellow, 2)
        cv2.putText(image, f"Reps: {self.total_reps}  Good: {self.good_reps}  Bad: {self.bad_reps}", (12, 58), font, 0.65, white, 1)

        if self.program:
            cv2.putText(image, f"Set {self.current_set}/{self.program.sets}", (w-200, 28), font, 0.65, green, 2)

        cv2.rectangle(image, (0, h-55), (w, h), (15, 15, 20), -1)
        color = green if self.form_color == (0, 255, 100) else red
        cv2.putText(image, self.feedback, (12, h-20), font, 0.85, color, 2)

        if self.visibility_warn:
            cv2.putText(image, f"WARNING: {self.visibility_warn}", (12, h-65), font, 0.6, yellow, 2)


if __name__ == "__main__":
    app = ShoulderAnalyzer()
    app.run()