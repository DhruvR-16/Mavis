import cv2
import mediapipe as mp
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "shoulder_press")
WINDOW = 30
STRIDE = 5
EPOCHS = 40
BATCH = 8

# =========================
# MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose

# =========================
# GEOMETRY
# =========================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

# =========================
# FEATURES
# =========================
def extract_features(lm):
    def L(i): return [lm[i].x, lm[i].y]

    l_sh, r_sh = L(11), L(12)
    l_el, r_el = L(13), L(14)
    l_wr, r_wr = L(15), L(16)
    l_hip, r_hip = L(23), L(24)

    sh = (calculate_angle(l_el, l_sh, l_hip) +
          calculate_angle(r_el, r_sh, r_hip)) / 2

    el = (calculate_angle(l_sh, l_el, l_wr) +
          calculate_angle(r_sh, r_el, r_wr)) / 2

    lean = (abs(l_hip[0] - r_hip[0]) /
            (abs(l_sh[0] - r_sh[0]) + 1e-6))

    shrug = abs(l_sh[1] - r_sh[1])

    return np.array([sh, el, lean, shrug])

# =========================
# VIDEO → FEATURES
# =========================
def process_video(path):
    cap = cv2.VideoCapture(path)
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    feats = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            feats.append(extract_features(res.pose_landmarks.landmark))

    cap.release()
    pose.close()
    return np.array(feats)

# =========================
# BUILD DATASET
# =========================
X, y = [], []

for vid in os.listdir(VIDEO_DIR):
    f = process_video(os.path.join(VIDEO_DIR, vid))
    for i in range(0, len(f) - WINDOW - 1, STRIDE):
        X.append(f[i:i+WINDOW])
        y.append(f[i+WINDOW])  # next frame

X = np.array(X)
y = np.array(y)

# =========================
# NORMALIZE
# =========================
mean = X.mean(axis=(0,1))
std = X.std(axis=(0,1)) + 1e-6

X = (X - mean) / std
y = (y - mean) / std

np.save("mean.npy", mean)
np.save("std.npy", std)

# =========================
# MODEL
# =========================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, 4)),
    LSTM(64),
    Dense(4)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

model.save("one_class_pose_lstm.h5")
print("Model saved")
