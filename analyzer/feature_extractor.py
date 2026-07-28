import numpy as np
import math


# MediaPipe landmark indices
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
}

# Visibility threshold — landmarks below this are considered unreliable
VISIBILITY_THRESHOLD = 0.55


def calculate_angle(a, b, c) -> float:
    """
    Compute the interior angle at vertex b formed by the ray b→a and ray b→c.
    Returns degrees in [0, 180].
    Human error tolerance: ±15° is acceptable for most exercises.
    """
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    dot = ax * cx + ay * cy
    mag_a = math.sqrt(ax**2 + ay**2) + 1e-9
    mag_c = math.sqrt(cx**2 + cy**2) + 1e-9
    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))


def get_shoulder_width(landmarks) -> float:
    """
    Returns Euclidean distance between shoulders as a normalisation factor.
    Falls back to 0.2 (rough average) if landmarks are occluded.
    """
    ls = landmarks[LM["left_shoulder"]]
    rs = landmarks[LM["right_shoulder"]]
    if ls.visibility < VISIBILITY_THRESHOLD or rs.visibility < VISIBILITY_THRESHOLD:
        return 0.2
    dist = math.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
    return max(dist, 0.05)  # prevent divide-by-zero


def get_visibility_map(landmarks) -> dict:
    """
    Returns per-region visibility status for the UI warning system.
    Each value is True (visible) or False (occluded).
    """
    def visible(idx):
        return landmarks[idx].visibility >= VISIBILITY_THRESHOLD

    return {
        "left_arm":    visible(LM["left_shoulder"]) and visible(LM["left_elbow"]) and visible(LM["left_wrist"]),
        "right_arm":   visible(LM["right_shoulder"]) and visible(LM["right_elbow"]) and visible(LM["right_wrist"]),
        "shoulders":   visible(LM["left_shoulder"]) and visible(LM["right_shoulder"]),
        "hips":        visible(LM["left_hip"]) and visible(LM["right_hip"]),
        "left_leg":    visible(LM["left_hip"]) and visible(LM["left_knee"]) and visible(LM["left_ankle"]),
        "right_leg":   visible(LM["right_hip"]) and visible(LM["right_knee"]) and visible(LM["right_ankle"]),
    }


class FeatureExtractor:
    """
    Extracts 11 normalised geometric features from a pose landmark set.

    Feature vector:
        0  – left elbow drift  (body-relative, vs calibrated anchor)
        1  – left elbow angle  (degrees)
        2  – right elbow angle (degrees)
        3  – left shoulder angle
        4  – right shoulder angle
        5  – left wrist-to-shoulder vertical offset (body-relative)
        6  – torso lean angle  (degrees, shoulder mid to hip mid vs vertical)
        7  – bilateral elbow symmetry  (|left_angle - right_angle|)
        8  – left wrist height  (body-relative, above shoulder = negative)
        9  – right wrist height (body-relative)
        10 – right elbow drift (body-relative, vs calibrated anchor)
    """

    def __init__(self):
        # Calibration anchor: set during the standing pose at session start
        self.calibrated = False
        self.calib_shoulder_width: float = 0.2
        self.calib_left_elbow_x: float = 0.5
        self.calib_right_elbow_x: float = 0.5
        self.calib_left_elbow_offset: float = 0.0
        self.calib_right_elbow_offset: float = 0.0
        self.calib_torso_angle: float = 90.0

    def calibrate(self, landmarks) -> None:
        """
        Call once during the T-pose / neutral standing calibration phase.
        Sets per-user body proportions so all subsequent drift calculations
        are body-relative rather than screen-pixel-relative.
        """
        self.calib_shoulder_width = get_shoulder_width(landmarks)
        self.calib_left_elbow_x   = landmarks[LM["left_elbow"]].x
        self.calib_right_elbow_x  = landmarks[LM["right_elbow"]].x
        ls = landmarks[LM["left_shoulder"]]
        rs = landmarks[LM["right_shoulder"]]
        lh = landmarks[LM["left_hip"]]
        rh = landmarks[LM["right_hip"]]

        # Neutral elbow position expressed relative to its own shoulder, so
        # drift stays meaningful if the lifter shifts around in frame.
        sw = self.calib_shoulder_width
        self.calib_left_elbow_offset  = (landmarks[LM["left_elbow"]].x - ls.x) / sw
        self.calib_right_elbow_offset = (landmarks[LM["right_elbow"]].x - rs.x) / sw

        # Torso angle: angle at shoulder midpoint between vertical and hip midpoint
        smx = (ls.x + rs.x) / 2
        smy = (ls.y + rs.y) / 2
        hmx = (lh.x + rh.x) / 2
        hmy = (lh.y + rh.y) / 2
        dx = hmx - smx
        dy = hmy - smy
        self.calib_torso_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
        self.calibrated = True

    def get_features(self, landmarks) -> list:
        """
        Returns a 10-element list matching the scaler's expected input.
        Uses body-relative normalisation for all spatial measurements.
        """
        sw = get_shoulder_width(landmarks)

        ls  = landmarks[LM["left_shoulder"]]
        rs  = landmarks[LM["right_shoulder"]]
        le  = landmarks[LM["left_elbow"]]
        re  = landmarks[LM["right_elbow"]]
        lw  = landmarks[LM["left_wrist"]]
        rw  = landmarks[LM["right_wrist"]]
        lh  = landmarks[LM["left_hip"]]
        rh  = landmarks[LM["right_hip"]]

        # ── angles ──────────────────────────────────────────────────────────
        left_elbow_angle  = calculate_angle(ls, le, lw)
        right_elbow_angle = calculate_angle(rs, re, rw)
        left_shoulder_angle  = calculate_angle(le, ls, lh)
        right_shoulder_angle = calculate_angle(re, rs, rh)

        # ── torso lean ───────────────────────────────────────────────────────
        smx = (ls.x + rs.x) / 2;  smy = (ls.y + rs.y) / 2
        hmx = (lh.x + rh.x) / 2;  hmy = (lh.y + rh.y) / 2
        torso_angle = math.degrees(math.atan2(abs(smx - hmx), abs(smy - hmy) + 1e-9))

        # ── torso-relative elbow drift ───────────────────────────────────────
        # Measured as the elbow's horizontal offset FROM ITS OWN SHOULDER,
        # normalised by shoulder width, compared against the same quantity
        # captured at calibration.
        #
        # This deliberately does not compare against an absolute calibrated
        # x-coordinate: doing so made any whole-body movement register as elbow
        # drift, so a lifter who shifted their stance or drifted across frame
        # was faulted for an elbow that never actually left their side.
        # Offset-from-shoulder is invariant to translation, and dividing by
        # shoulder width makes it invariant to distance from the camera.
        left_offset  = (le.x - ls.x) / sw
        right_offset = (re.x - rs.x) / sw
        if self.calibrated:
            left_drift  = abs(left_offset - self.calib_left_elbow_offset)
            right_drift = abs(right_offset - self.calib_right_elbow_offset)
        else:
            # Uncalibrated: fall back to the plumb line under the shoulder.
            left_drift  = abs(left_offset)
            right_drift = abs(right_offset)

        # ── wrist heights relative to shoulder (positive = below) ────────────
        left_wrist_height  = (lw.y - ls.y) / sw
        right_wrist_height = (rw.y - rs.y) / sw

        # ── wrist-to-shoulder vertical offset ───────────────────────────────
        wrist_shoulder_offset = (ls.y - lw.y) / sw   # positive when wrist above shoulder

        # ── bilateral symmetry ───────────────────────────────────────────────
        symmetry = abs(left_elbow_angle - right_elbow_angle)

        return [
            left_drift,            # 0
            left_elbow_angle,      # 1
            right_elbow_angle,     # 2
            left_shoulder_angle,   # 3
            right_shoulder_angle,  # 4
            wrist_shoulder_offset, # 5
            torso_angle,           # 6
            symmetry,              # 7
            left_wrist_height,     # 8
            right_wrist_height,    # 9
            right_drift,           # 10
        ]

    def get_angles_for_display(self, landmarks) -> dict:
        """
        Returns a dict of named angles for the UI overlay.
        These are the raw angles used by the state machines.
        """
        ls = landmarks[LM["left_shoulder"]]
        rs = landmarks[LM["right_shoulder"]]
        le = landmarks[LM["left_elbow"]]
        re = landmarks[LM["right_elbow"]]
        lw = landmarks[LM["left_wrist"]]
        rw = landmarks[LM["right_wrist"]]
        lh = landmarks[LM["left_hip"]]
        rh = landmarks[LM["right_hip"]]

        return {
            "left_elbow":      calculate_angle(ls, le, lw),
            "right_elbow":     calculate_angle(rs, re, rw),
            "left_shoulder":   calculate_angle(le, ls, lh),
            "right_shoulder":  calculate_angle(re, rs, rh),
        }