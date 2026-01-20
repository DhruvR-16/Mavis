import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --------------------------
# TUNABLE THRESHOLDS (adjust for camera/distance)
# --------------------------
THRESH = {
    "lateral_top": 65,        # degrees shoulder angle ~ 65-90 (arm-abduction)
    "lateral_bottom": 20,
    "front_top": 65,          # degrees using front plane
    "front_bottom": 20,
    "press_top_elbow_angle": 160,   # elbow almost straight when pressed
    "press_bottom_elbow_angle": 65,
    "shrug_allowed_px": 120,  # pixel distance from hip line to shoulder line allowed
    "torso_lean_ratio": 0.7q,
    "upright_row_top_h": 30,  # pixels above shoulder for elbows
}

# --------------------------
# exercises list
# --------------------------
EXERCISES = {
    1: "Lateral Raise",
    2: "Front Raise",
    3: "Shoulder Press",
    4: "Overhead Press",
    5: "Arnold Press",
    6: "Cable Lateral Raise",
    7: "Face Pulls",
    8: "Internal/External Rotation",
    9: "Upright Rows"
}

# Per-exercise state: stage (down/up), count, bad reps
state = defaultdict(lambda: {"stage": None, "count": 0, "bad": 0, "last_feedback": ""})
selected_ex = 1

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # handle keys to switch exercise
        key = cv2.waitKey(1) & 0xFF
        if key in [ord(str(i)) for i in range(1,10)]:
            selected_ex = int(chr(key))
        if key == ord('q'):
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        feedback_lines = []
        # draw selected exercise name
        cv2.putText(frame, f"Exercise [{selected_ex}]: {EXERCISES[selected_ex]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # helper to map Mp landmarks to pixel coords
            def P(LM):
                return (int(LM.x * w), int(LM.y * h))

            # common keypoints
            left_sh = P(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            right_sh = P(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            left_el = P(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value])
            right_el = P(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
            left_wr = P(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
            right_wr = P(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
            left_hip = P(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
            right_hip = P(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
            nose = P(lm[mp_pose.PoseLandmark.NOSE.value])
            # approximate neck by midpoint between shoulders
            neck = ((left_sh[0] + right_sh[0])//2, (left_sh[1] + right_sh[1])//2)

            hip_line_y = (left_hip[1] + right_hip[1]) / 2
            shoulder_line_y = (left_sh[1] + right_sh[1]) / 2

            # small utility functions for drawing text and counting logic
            def draw_feedback(y, text, color=(0,255,0)):
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            def rep_logic(ex_id, at_top_condition, at_bottom_condition, bad_condition=None):
                s = state[ex_id]
                feedback = ""
                # stage machine: None -> "down" (start) -> "up" -> increment when returns down
                if s["stage"] is None:
                    if at_bottom_condition:
                        s["stage"] = "down"
                elif s["stage"] == "down":
                    if at_top_condition:
                        s["stage"] = "up"
                        # check bad condition at top
                        if bad_condition:
                            if bad_condition():
                                s["bad"] += 1
                                s["last_feedback"] = "Bad rep (form)"
                            else:
                                s["count"] += 1
                                s["last_feedback"] = "Good rep"
                        else:
                            s["count"] += 1
                            s["last_feedback"] = "Good rep"
                elif s["stage"] == "up":
                    if at_bottom_condition:
                        s["stage"] = "down"
                return s

            ex = selected_ex

            # ---------------------------
            # Exercise-specific logic
            # ---------------------------
            # We'll compute a few shared angles where needed:
            # shoulder angle = angle(elbow, shoulder, hip) -> abduction / front raise depending on reference
            left_shoulder_angle = calculate_angle(left_el, left_sh, left_hip)
            right_shoulder_angle = calculate_angle(right_el, right_sh, right_hip)
            avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2

            # elbow angle (for press detection)
            left_elbow_angle = calculate_angle(left_sh, left_el, left_wr)
            right_elbow_angle = calculate_angle(right_sh, right_el, right_wr)
            avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

            # torso lean detection (compare shoulder/hip dx)
            hip_dx = abs(left_hip[0] - right_hip[0])
            shoulder_dx = abs(left_sh[0] - right_sh[0])
            lean_ratio = hip_dx / (shoulder_dx + 1)

            # shrug detection: distance between hip_line_y and shoulder_line_y
            shrug_distance_px = hip_line_y - shoulder_line_y

            # convenience to draw landmarks with color
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=2)
            )

            # Now per-exercise:
            if ex == 1:  # Lateral Raise
                # arm abduction: shoulder angle formed by elbow-shoulder-hip
                top_cond = avg_shoulder_angle >= THRESH["lateral_top"]
                bottom_cond = avg_shoulder_angle <= THRESH["lateral_bottom"]
                # bad if shrugging or torso lean
                def bad():
                    return (shrug_distance_px < THRESH["shrug_allowed_px"]) is False or lean_ratio < THRESH["torso_lean_ratio"]
                s = rep_logic(1, top_cond, bottom_cond, bad_condition=bad)
                feedback_lines.append(f"Lateral: angle={int(avg_shoulder_angle)}°  Reps: {s['count']} Bad:{s['bad']} {s['last_feedback']}")

                # visual cues
                color = (0,255,0) if top_cond and not bad() else (0,0,255) if bad() else (255,165,0)
                cv2.line(frame, left_sh, left_el, color, 3)
                cv2.line(frame, right_sh, right_el, color, 3)

            elif ex == 2:  # Front Raise
                # front raise: angle between elbow-shoulder-nose (front plane)
                left_front_angle = calculate_angle(left_el, left_sh, nose)
                right_front_angle = calculate_angle(right_el, right_sh, nose)
                avg_front_angle = (left_front_angle + right_front_angle) / 2
                top_cond = avg_front_angle >= THRESH["front_top"]
                bottom_cond = avg_front_angle <= THRESH["front_bottom"]
                def bad(): 
                    return (shrug_distance_px < THRESH["shrug_allowed_px"]) is False or lean_ratio < THRESH["torso_lean_ratio"]
                s = rep_logic(2, top_cond, bottom_cond, bad_condition=bad)
                feedback_lines.append(f"Front: angle={int(avg_front_angle)}°  Reps: {s['count']} Bad:{s['bad']} {s['last_feedback']}")
                color = (0,255,0) if top_cond and not bad() else (0,0,255) if bad() else (255,165,0)
                cv2.line(frame, left_sh, left_wr, color, 3)
                cv2.line(frame, right_sh, right_wr, color, 3)

            elif ex in (3,4):  # Shoulder Press / Overhead Press
                # Counting based on elbow angle (bottom -> approx 60-90 elbow bend, top -> near straight)
                top_cond = avg_elbow_angle > THRESH["press_top_elbow_angle"]
                bottom_cond = avg_elbow_angle < THRESH["press_bottom_elbow_angle"]
                # bad if arching (torso leaning back or forward) or shrug
                def bad():
                    # wrist y should not be far behind nose (spine hyperextension not easily measured) - check torso lean
                    return lean_ratio < THRESH["torso_lean_ratio"] or (shrug_distance_px < THRESH["shrug_allowed_px"]) is False
                s = rep_logic(ex, top_cond, bottom_cond, bad_condition=bad)
                feedback_lines.append(f"Press: elbow_angle={int(avg_elbow_angle)}°  Reps: {s['count']} Bad:{s['bad']} {s['last_feedback']}")
                color = (0,255,0) if top_cond and not bad() else (0,0,255) if bad() else (255,165,0)
                # show elbow angles
                cv2.putText(frame, f"E:{int(left_elbow_angle)}/{int(right_elbow_angle)}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2)

            elif ex == 5:  # Arnold Press
                # Arnold press rotates forearm: approximate rotation by relative x positions of wrist vs elbow
                # starting: palms facing you (wrists more lateral), mid-rotation: wrist crosses
                left_rot = left_wr[0] - left_el[0]
                right_rot = right_wr[0] - right_el[0]
                # when rotating forward, sign will change for left/right depending on camera
                rot_magnitude = (abs(left_rot) + abs(right_rot)) / 2
                # use shoulder-abduction to count reps too
                top_cond = avg_shoulder_angle >= THRESH["lateral_top"]
                bottom_cond = avg_shoulder_angle <= THRESH["lateral_bottom"]
                def bad():
                    return (shrug_distance_px < THRESH["shrug_allowed_px"]) is False or lean_ratio < THRESH["torso_lean_ratio"]
                s = rep_logic(5, top_cond, bottom_cond, bad_condition=bad)
                feedback_lines.append(f"Arnold: rot={int(rot_magnitude)} px  Reps: {s['count']} Bad:{s['bad']} {s['last_feedback']}")
                color = (0,255,0) if top_cond and rot_magnitude>5 and not bad() else (0,0,255) if bad() else (255,165,0)
                cv2.line(frame, left_el, left_wr, color, 3)
                cv2.line(frame, right_el, right_wr, color, 3)

            elif ex == 6:  # Cable Lateral Raise (similar to lateral but often one-arm)
                # detect each arm separately; count when both arms pass top
                left_top = left_shoulder_angle >= THRESH["lateral_top"]
                right_top = right_shoulder_angle >= THRESH["lateral_top"]
                left_bottom = left_shoulder_angle <= THRESH["lateral_bottom"]
                right_bottom = right_shoulder_angle <= THRESH["lateral_bottom"]
                s = state[6]
                # we'll consider a rep when both arms go top from bottom
                if s["stage"] is None:
                    if left_bottom and right_bottom:
                        s["stage"] = "down"
                elif s["stage"] == "down":
                    if left_top and right_top:
                        s["stage"] = "up"
                        if (shrug_distance_px < THRESH["shrug_allowed_px"]) is False or lean_ratio < THRESH["torso_lean_ratio"]:
                            s["bad"] += 1
                            s["last_feedback"] = "Bad rep (shrug/lean)"
                        else:
                            s["count"] += 1
                            s["last_feedback"] = "Good rep"
                elif s["stage"] == "up":
                    if left_bottom and right_bottom:
                        s["stage"] = "down"
                feedback_lines.append(f"Cable Lat: L{int(left_shoulder_angle)} R{int(right_shoulder_angle)}  Reps:{s['count']} Bad:{s['bad']} {s['last_feedback']}")
                color = (0,255,0) if left_top and right_top and shrug_distance_px >= THRESH["shrug_allowed_px"] else (0,0,255)

                cv2.line(frame, left_sh, left_el, color, 3)
                cv2.line(frame, right_sh, right_el, color, 3)

            elif ex == 7:  # Face Pulls
                # face pull: elbows should be high and wrists near face (nose)
                left_elv = left_el[1] < left_sh[1]  # elbow above shoulder?
                right_elv = right_el[1] < right_sh[1]
                left_near = abs(left_wr[0] - nose[0]) < 120 and abs(left_wr[1] - nose[1]) < 120
                right_near = abs(right_wr[0] - nose[0]) < 120 and abs(right_wr[1] - nose[1]) < 120
                good = left_elv and right_elv and (left_near or right_near)
                s = state[7]
                # simple stage: arms pulled in (near face) vs extended
                pulled = (left_near or right_near)
                extended = (not pulled)
                if s["stage"] is None:
                    if extended:
                        s["stage"] = "out"
                elif s["stage"] == "out":
                    if pulled:
                        s["stage"] = "in"
                        if not good:
                            s["bad"] += 1
                            s["last_feedback"] = "Bad rep"
                        else:
                            s["count"] += 1
                            s["last_feedback"] = "Good rep"
                elif s["stage"] == "in":
                    if extended:
                        s["stage"] = "out"
                feedback_lines.append(f"FacePull: elbows_up={left_elv and right_elv} wrists_near_face={(left_near or right_near)} Reps:{s['count']} Bad:{s['bad']} {s['last_feedback']}")
                color = (0,255,0) if good else (0,0,255)

                cv2.line(frame, left_el, left_wr, color, 3)
                cv2.line(frame, right_el, right_wr, color, 3)

            elif ex == 8:  # Internal/External Rotation
                # elbow pinned at side (should be at same y as shoulder)
                left_pinned = abs(left_el[1] - left_sh[1]) < 40
                right_pinned = abs(right_el[1] - right_sh[1]) < 40
                # rotation detected by change in angle forearm-elbow-shoulder
                left_rot_angle = calculate_angle(left_wr, left_el, left_sh)
                right_rot_angle = calculate_angle(right_wr, right_el, right_sh)
                # count reps by rotation crossing thresholds (internal <-> external)
                left_top = left_rot_angle > 100
                left_bottom = left_rot_angle < 50
                right_top = right_rot_angle > 100
                right_bottom = right_rot_angle < 50
                s = state[8]
                # simple bilateral check: both pinned
                if not (left_pinned and right_pinned):
                    feedback_lines.append("Keep elbows pinned to sides")
                    color = (0,0,255)
                else:
                    # count when both sides rotate out then back
                    if s["stage"] is None:
                        if left_bottom and right_bottom:
                            s["stage"] = "neutral"
                    elif s["stage"] == "neutral":
                        if left_top and right_top:
                            s["stage"] = "rotated"
                            s["count"] += 1
                            s["last_feedback"] = "Good rep"
                    elif s["stage"] == "rotated":
                        if left_bottom and right_bottom:
                            s["stage"] = "neutral"
                    feedback_lines.append(f"Rot: L{int(left_rot_angle)} R{int(right_rot_angle)} Reps:{s['count']} Bad:{s['bad']} {s['last_feedback']}")
                    color = (0,255,0)

                cv2.line(frame, left_el, left_wr, color, 3)
                cv2.line(frame, right_el, right_wr, color, 3)

            elif ex == 9:  # Upright Rows
                # elbows travel up in front; measure elbow y relative to shoulder
                left_elv_h = left_sh[1] - left_el[1]  # positive if elbow above shoulder
                right_elv_h = right_sh[1] - right_el[1]
                # count rep when both elbows rise above shoulder by threshold
                top_cond = (left_elv_h > THRESH["upright_row_top_h"]) and (right_elv_h > THRESH["upright_row_top_h"])
                bottom_cond = (left_elv_h < 10) and (right_elv_h < 10)
                def bad():
                    return lean_ratio < THRESH["torso_lean_ratio"] or (shrug_distance_px < THRESH["shrug_allowed_px"]) is False
                s = rep_logic(9, top_cond, bottom_cond, bad_condition=bad)
                feedback_lines.append(f"Upright: ElvH L{int(left_elv_h)} R{int(right_elv_h)} Reps:{s['count']} Bad:{s['bad']} {s['last_feedback']}")
                color = (0,255,0) if top_cond and not bad() else (0,0,255) if bad() else (255,165,0)
                cv2.line(frame, left_el, left_sh, color, 3)
                cv2.line(frame, right_el, right_sh, color, 3)

            # Draw summary feedback lines on left
            y = 60
            for ln in feedback_lines:
                cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                y += 25

            # Show counts of current exercise
            s = state[ex]
            cv2.putText(frame, f"Count: {s['count']}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Bad: {s['bad']}", (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if s.get("last_feedback"):
                cv2.putText(frame, f"{s['last_feedback']}", (w - 400, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 2)

            # Extra global posture warnings
            if shrug_distance_px < THRESH["shrug_allowed_px"]:
                cv2.putText(frame, "WARNING: Shrugging shoulders!", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
            if lean_ratio < THRESH["torso_lean_ratio"]:
                cv2.putText(frame, "WARNING: Torso leaning/cheating!", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

        else:
            cv2.putText(frame, "No person detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Shoulder Exercise Trainer (press 1-9 to switch)", frame)

    cap.release()
    cv2.destroyAllWindows()
