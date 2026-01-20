import numpy as np
import mediapipe as mp

class FeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        """
        Calculates angle between three points a, b, c.
        b is the vertex.
        points are objects with x, y attributes.
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def calculate_ground_angle(self, a, b):
        """
        Calculates angle of segment ab with respect to vertical axis.
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        
        vec = b - a
        vertical = np.array([0, 1]) # Assuming image y-axis points down
        
        unit_vec = vec / (np.linalg.norm(vec) + 1e-6)
        
        dot_product = np.dot(unit_vec, vertical)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
        
        return angle

    def get_features(self, landmarks):
        """
        Extracts the 10 angle features used in training.
        Returns a list of 10 floats.
        """
        P = self.mp_pose.PoseLandmark
        
        # Use Left side for primary analysis/training consistency if mixed
        # or we could make it side-aware. Since previous step showed 'left' in CSV,
        # we will extract left side features by default or mirror if needed.
        # Ideally, we detect the active arm. For now, let's extract LEFT.
        
        # Helper to get point
        def p(idx): return landmarks[idx.value]
        
        # 1. Shoulder Angle (Elbow-Shoulder-Hip)
        shoulder_angle = self.calculate_angle(p(P.LEFT_ELBOW), p(P.LEFT_SHOULDER), p(P.LEFT_HIP))
        
        # 2. Elbow Angle (Shoulder-Elbow-Wrist)
        elbow_angle = self.calculate_angle(p(P.LEFT_SHOULDER), p(P.LEFT_ELBOW), p(P.LEFT_WRIST))
        
        # 3. Hip Angle (Shoulder-Hip-Knee)
        hip_angle = self.calculate_angle(p(P.LEFT_SHOULDER), p(P.LEFT_HIP), p(P.LEFT_KNEE))
        
        # 4. Knee Angle (Hip-Knee-Ankle)
        knee_angle = self.calculate_angle(p(P.LEFT_HIP), p(P.LEFT_KNEE), p(P.LEFT_ANKLE))
        
        # 5. Ankle Angle (Knee-Ankle-Foot)
        # Using LEFT_FOOT_INDEX as toe approximation
        ankle_angle = self.calculate_angle(p(P.LEFT_KNEE), p(P.LEFT_ANKLE), p(P.LEFT_FOOT_INDEX))
        
        # Ground Angles
        # 6. Shoulder Ground (Upper arm vs Vertical)
        shoulder_ground = self.calculate_ground_angle(p(P.LEFT_SHOULDER), p(P.LEFT_ELBOW))
        
        # 7. Elbow Ground (Forearm vs Vertical)
        elbow_ground = self.calculate_ground_angle(p(P.LEFT_ELBOW), p(P.LEFT_WRIST))
        
        # 8. Hip Ground (Thigh vs Vertical)
        hip_ground = self.calculate_ground_angle(p(P.LEFT_HIP), p(P.LEFT_KNEE))
        
        # 9. Knee Ground (Shin vs Vertical)
        knee_ground = self.calculate_ground_angle(p(P.LEFT_KNEE), p(P.LEFT_ANKLE))
        
        # 10. Ankle Ground (Foot vs Vertical)
        ankle_ground = self.calculate_ground_angle(p(P.LEFT_ANKLE), p(P.LEFT_FOOT_INDEX))
        
        return [
            shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle,
            shoulder_ground, elbow_ground, hip_ground, knee_ground, ankle_ground
        ]
