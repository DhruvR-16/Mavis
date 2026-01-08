import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

WINDOW = 30
ERROR_THRESH = 0.08  # tune this

model = load_model("one_class_pose_lstm.h5")
mean = np.load("mean.npy")
std = np.load("std.npy")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(0.6, 0.6)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))

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
    lean = abs(l_hip[0] - r_hip[0]) / (abs(l_sh[0] - r_sh[0]) + 1e-6)
    shrug = abs(l_sh[1] - r_sh[1])

    return np.array([sh, el, lean, shrug])

cap = cv2.VideoCapture(0)
seq = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    label = "WAITING"
    color = (255,255,0)

    if res.pose_landmarks:
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        f = extract_features(res.pose_landmarks.landmark)
        f = (f - mean) / std

        seq.append(f)
        if len(seq) > WINDOW:
            seq.pop(0)

        if len(seq) == WINDOW:
            pred = model.predict(np.expand_dims(seq,0), verbose=0)[0]
            err = np.mean((pred - f)**2)

            if err > ERROR_THRESH:
                label = f"BAD ({err:.3f})"
                color = (0,0,255)
            else:
                label = f"GOOD ({err:.3f})"
                color = (0,255,0)

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("One-Class Pose LSTM", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
