import cv2
import mediapipe as mp
import numpy as np
from enum import Enum



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class Stage(Enum):
    DOWN = 1
    UP = 2

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point
    

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class BicepCurlAnalyzer:
    def __init__(self):
        # Thresholds and constants
        self.UP_THRESHOLD = 60      # Angle to be considered 'UP'
        self.DOWN_THRESHOLD = 130   # Angle to be considered 'DOWN'
        self.VISIBILITY_THRESHOLD = 0.5
        self.ELBOW_DRIFT_THRESHOLD_PX = 35 
        # NEW: Minimum frames for a rep to count (prevents cheating with speed)
        self.MIN_REP_DURATION_FRAMES = 10 


        self.left_arm = {
            "stage": Stage.DOWN,
            "counter": 0,
            "feedback": "",
            "anchor_elbow_x": 0,
            "form_error": False,
            "start_frame": 0 # NEW: To track rep duration
        }

        self.right_arm = {
            "stage": Stage.DOWN,
            "counter": 0,
            "feedback": "",
            "anchor_elbow_x": 0,
            "form_error": False,
            "start_frame": 0 # NEW: To track rep duration
        }
        
    def _process_arm(self, arm_state, shoulder, elbow, wrist, hip, frame_counter):
        arm_angle = calculate_angle(shoulder, elbow, wrist)
        
        #  FORM CHECKING (during the 'UP' phase) 
        if arm_state["stage"] == Stage.UP:
            drift_distance = abs(elbow[0] - arm_state["anchor_elbow_x"])
            if drift_distance > self.ELBOW_DRIFT_THRESHOLD_PX:
                arm_state["feedback"] = "ERROR: Keep Elbow Still"
                arm_state["form_error"] = True
            else:
                if arm_state["form_error"] and "Elbow" in arm_state["feedback"]:
                    arm_state["form_error"] = False
                    arm_state["feedback"] = ""

        #  STATE MACHINE (Rep Counter) 
        if arm_angle < self.UP_THRESHOLD:
            if arm_state["stage"] == Stage.DOWN:
                # Transition from DOWN to UP
                arm_state["stage"] = Stage.UP
                arm_state["anchor_elbow_x"] = elbow[0] 
                arm_state["form_error"] = False
                # NEW: Record the starting frame of the rep
                arm_state["start_frame"] = frame_counter
                
                # Initial drift check
                drift_distance = abs(elbow[0] - arm_state["anchor_elbow_x"])
                if drift_distance > self.ELBOW_DRIFT_THRESHOLD_PX:
                    arm_state["feedback"] = "ERROR: Keep Elbow Still"
                    arm_state["form_error"] = True

        elif arm_angle > self.DOWN_THRESHOLD:
            if arm_state["stage"] == Stage.UP:
                # Transition from UP to DOWN
                arm_state["stage"] = Stage.DOWN
                
                # NEW: Calculate rep duration and check for speed
                rep_duration = frame_counter - arm_state["start_frame"]
                if rep_duration < self.MIN_REP_DURATION_FRAMES:
                    arm_state["feedback"] = "ERROR: Too Fast!"
                    arm_state["form_error"] = True
                
                # Final check: If no form errors were flagged during the rep, count it.
                if not arm_state["form_error"]:
                    arm_state["counter"] += 1
                    arm_state["feedback"] = "Good Rep!"
                else:
                    # If feedback wasn't already set to "Too Fast!", use a generic message.
                    if "Fast" not in arm_state["feedback"]:
                         arm_state["feedback"] = "Bad Form - Reset"

        return arm_angle

    def analyze_frame(self, frame, frame_counter):
        h, w, _ = frame.shape

        #  Pose Detection 
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            #  Landmark Extraction and Processing 
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # LEFT ARM
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]

                    if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > self.VISIBILITY_THRESHOLD:
                        left_angle = self._process_arm(self.left_arm, l_shoulder, l_elbow, l_wrist, l_hip, frame_counter)
                        cv2.putText(frame, str(int(left_angle)), tuple(np.multiply(l_elbow, [1, 1]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    # RIGHT ARM
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
                    
                    if landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > self.VISIBILITY_THRESHOLD:
                        right_angle = self._process_arm(self.right_arm, r_shoulder, r_elbow, r_wrist, r_hip, frame_counter)
                        cv2.putText(frame, str(int(right_angle)), tuple(np.multiply(r_elbow, [1, 1]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                except Exception as e:
                    pass

                # Skeleton Coloring
                is_good_form = not (self.left_arm["form_error"] or self.right_arm["form_error"])
                skeleton_color = (0, 255, 0) if is_good_form else (0, 0, 255)
                
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2)
                connection_drawing_spec = mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2)
                
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec, connection_drawing_spec)

        self._draw_ui(frame)
        return frame

    def _draw_ui(self, frame):
        h, w, _ = frame.shape
        
        cv2.rectangle(frame, (0, 0), (250, 120), (245, 117, 16), -1)
        cv2.putText(frame, 'LEFT ARM', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'REPS', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.left_arm["counter"]), (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, self.left_arm["stage"].name, (120, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (w - 250, 0), (w, 120), (245, 117, 16), -1)
        cv2.putText(frame, 'RIGHT ARM', (w - 235, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'REPS', (w - 235, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.right_arm["counter"]), (w - 240, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (w - 130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, self.right_arm["stage"].name, (w - 130, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        feedback = self.left_arm["feedback"] if self.left_arm["feedback"] else self.right_arm["feedback"]
        color = (0, 0, 255) if "ERROR" in feedback or "Bad" in feedback else (0, 255, 0)
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, feedback, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    analyzer = BicepCurlAnalyzer()
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        processed_frame = analyzer.analyze_frame(frame, frame_counter)
        
        cv2.imshow('Bicep Curl AI Trainer', processed_frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()