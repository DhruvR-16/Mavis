import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
from collections import deque
from enum import Enum

# Import Feature Extractor
try:
    from analyzer.feature_extractor import FeatureExtractor
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'analyzer'))
    from feature_extractor import FeatureExtractor

# Constants
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.7
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# State Machine
class Stage(Enum):
    DOWN = 1
    UP = 2

# Smoothing Class
class LandmarkSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        
        smoothed = []
        for i, lm in enumerate(current_landmarks):
            prev = self.prev_landmarks[i]
            # Smooth x, y, z, and visibility
            sx = self.alpha * lm.x + (1 - self.alpha) * prev.x
            sy = self.alpha * lm.y + (1 - self.alpha) * prev.y
            sz = self.alpha * lm.z + (1 - self.alpha) * prev.z
            sv = self.alpha * lm.visibility + (1 - self.alpha) * prev.visibility
            
            # Create a mock object or struct (MediaPipe landmarks are protobufs, tough to modify directly)
            # We will return a simple object/dict wrapper for the extractor
            class SmoothPoint:
                def __init__(self, x, y, z, v):
                    self.x, self.y, self.z, self.visibility = x, y, z, v
            
            smoothed.append(SmoothPoint(sx, sy, sz, sv))
            
        self.prev_landmarks = smoothed
        return smoothed

class BicepAnalyzer:
    def __init__(self):
        # Tools
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.extractor = FeatureExtractor()
        self.smoother = LandmarkSmoother(alpha=0.6) # 0.6 = moderate smoothing
        
        # AI Logic
        self.lstm_model = None
        self.scaler = None
        self.label_encoder = None
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.current_action = "Waiting..."
        self.confidence = 0.0
        
        # Form Logic
        self.stage = Stage.DOWN
        self.counter = 0
        self.bad_reps = 0
        self.feedback = "Start Curls"
        self.form_color = (0, 255, 0) # Green
        
        # Physics state
        self.prev_time = 0
        self.rep_start_time = 0
        self.elbow_x_history = deque(maxlen=30)
        self.start_elbow_x = 0
        
        self.load_models()
        
        # Heuristics - STRICT MODE
        self.UP_THRESH = 45 # Stricter than 50
        self.DOWN_THRESH = 160 # Needs almost full extension
        self.ELBOW_DRIFT_TOLERANCE = 0.1 # 10% of frame width (approx)
        self.MIN_REP_TIME = 1.0 # Seconds

    def load_models(self):
        try:
            model_path = os.path.join(MODELS_DIR, 'bicep_lstm.h5')
            scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
            le_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
            
            # Robust Load
            try:
                self.lstm_model = tf.keras.models.load_model(model_path)
            except (AttributeError, ImportError):
                import keras
                self.lstm_model = keras.models.load_model(model_path)
                
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(le_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Analyzer will run in GEOMETRY ONLY mode.")

    def predict_action(self, features):
        if self.lstm_model is None:
            return "Bicep Curl" # Default fallback
            
        scaled_feat = self.scaler.transform([features])[0]
        self.sequence_buffer.append(scaled_feat)
        
        if len(self.sequence_buffer) == SEQUENCE_LENGTH:
            seq = np.array(self.sequence_buffer)[np.newaxis, ...]
            preds = self.lstm_model.predict(seq, verbose=0)[0]
            idx = np.argmax(preds)
            
            self.confidence = preds[idx]
            label = self.label_encoder.inverse_transform([idx])[0]
            
            # Confidence Check
            if self.confidence < CONFIDENCE_THRESHOLD:
                return "Unknown"
                
            return label
        
        return "Waiting..."

    def analyze_form(self, features, landmarks):
        import time
        current_time = time.time()
        
        # features[1] is Elbow Angle
        elbow_angle = features[1]
        
        # Extract raw coordinates for drift check (Left Elbow)
        # Using feature_extractor logic, index 13 is Left Elbow
        elbow_curr_x = landmarks[13].x 
        
        # State Machine with Strict Checks
        if elbow_angle > self.DOWN_THRESH:
            # Check if we were previously UP
            if self.stage == Stage.UP:
                duration = current_time - self.rep_start_time
                if duration > self.MIN_REP_TIME:
                    self.stage = Stage.DOWN
                    self.counter += 1
                    self.feedback = "Good Rep!"
                    self.form_color = (0, 255, 0)
                else:
                    self.feedback = "Too Fast!"
                    self.bad_reps += 1
                    self.form_color = (0, 0, 255)
                    self.stage = Stage.DOWN # Reset anyway
            else:
                self.feedback = "Start Curls"
                self.start_elbow_x = elbow_curr_x # Reset anchor

        elif elbow_angle < self.UP_THRESH:
            if self.stage == Stage.DOWN:
                self.stage = Stage.UP
                self.rep_start_time = current_time
                self.start_elbow_x = elbow_curr_x
            
            # While holding UP or moving, check drift
            drift = abs(elbow_curr_x - self.start_elbow_x)
            if drift > self.ELBOW_DRIFT_TOLERANCE:
                self.feedback = "Fix Elbow!"
                self.form_color = (0, 0, 255)
            else:
                self.feedback = "Squeeze!"

        # General Guidance
        if self.stage == Stage.UP and self.feedback != "Fix Elbow!":
            self.feedback = "Lower Slowly"
        elif self.stage == Stage.DOWN and "Good" not in self.feedback and "Fast" not in self.feedback:
             if elbow_angle < self.DOWN_THRESH:
                 self.feedback = "Full Extension"

    def run(self):
        cap = cv2.VideoCapture(0)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Preprocessing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Flip for mirror view (matches shoulders.py)
                image = cv2.flip(image, 1) 
                
                image.flags.writeable = False
                results = pose.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                h, w, _ = image.shape
                
                if results.pose_landmarks:
                    try:
                        raw_landmarks = results.pose_landmarks.landmark
                        
                        # 0. Apply Smoothing
                        landmarks = self.smoother.smooth(raw_landmarks)
                        
                        # 1. Extract Features
                        features = self.extractor.get_features(landmarks)
                        
                        # 2. AI Prediction
                        self.current_action = self.predict_action(features)
                        
                        # 3. Logic Switch
                        if self.current_action == "Bicep Curl" or self.lstm_model is None:
                            self.analyze_form(features, landmarks)
                            self.form_color = (0, 255, 0)
                        else:
                            self.feedback = "Wrong Exercise"
                            self.form_color = (0, 0, 255)
                            
                        # 4. Visualization
                        # Draw Skeleton
                        self.mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                             self.mp_drawing.DrawingSpec(color=self.form_color, thickness=2, circle_radius=2),
                             self.mp_drawing.DrawingSpec(color=self.form_color, thickness=2, circle_radius=2)
                        )
                        
                        # UI Overlay (Shoulder Trainer Style)
                        # Top Left: Exercise Name
                        cv2.putText(image, "Exercise: Bicep Curl", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        
                        # Top Right: Counters
                        cv2.putText(image, f"Count: {self.counter}", (w - 200, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image, f"Bad: {self.bad_reps}", (w - 200, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Bottom: Feedback
                        cv2.putText(image, self.feedback, (10, h - 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Debug: AI Confidence below Exercise Name
                        cv2.putText(image, f"AI Conf: {self.confidence:.2f}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    except Exception as e:
                        # print(e)
                        pass

                    except Exception as e:
                        # print(e)
                        pass
                
                cv2.imshow('Mavis - Bicep Analyzer', image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = BicepAnalyzer()
    app.run()
