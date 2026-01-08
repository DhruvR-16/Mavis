import cv2
import mediapipe as mp
import numpy as np
from keras.models import Seqential
from keras.layers import LSTM, Dense, Dropout


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = b - a
    bc = b - c

    cosine_angle = np.dot(ba, bc) / (np.linalg(ba) * np.linalg(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_features(landmarks):
    def L(idx):
        lm = landmarks[idx]
        return [lm.x, lm.y]
    
    l_sh, r_sh = L(11), L(12)
    l_el, r_el = L(13), L(14)
    l_wr, r_wr = L(15), L(16)
    l_hip, r_hip = L(23), L(24)

    left_sh_angle = calculate_angle(l_el, l_sh, l_hip)
    right_sh_angle = calculate_angle(r_el, r_sh, r_hip)
    average_sh_angle = (left_sh_angle, right_sh_angle) / 2

    left_el_angle = calculate_angle(l_sh, l_el, l_wr)
    right_el_angle = calculate_angle(r_sh, r_el, r_wr)
    average_el_angle = (left_el_angle + right_el_angle) / 2

    shoulder_width = abs(l_sh[0] - r_sh[0]) + 1e-6
    hip_width = abs(l_hip[0] - r_hip[0])
    lean_ratio =  hip_width / shoulder_width

    shoulder_height_diff = abs(l_sh[1] - r_sh[1])

    return [
        average_sh_angle,
        average_el_angle,
        lean_ratio,
        shoulder_height_diff
    ]

model = Seqential([
    LSTM(64, return_sequences=True, input_shape = (30, 4)),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.complie(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics="accuracy"
)

