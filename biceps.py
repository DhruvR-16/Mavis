import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


cap = cv2.VideoCapture(0)

curr_status = 'neutral'


with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)

        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape


            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow    = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist    = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            shoulder_xy = (shoulder.x * w, shoulder.y * h)
            elbow_xy    = (elbow.x * w, elbow.y * h)
            wrist_xy    = (wrist.x * w, wrist.y * h)


            angle = calculate_angle(shoulder_xy, elbow_xy, wrist_xy)

            if angle>160:
                curr_status='Down'
            elif angle>50:
                curr_status='Up'


            if angle < 60 or angle > 140:
                color = (0, 0, 255) 
            else:
                color = (0, 255, 0)  

            

            cv2.putText(frame, str(int(angle)),
                        (int(elbow_xy[0]), int(elbow_xy[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            

            color = (0, 255, 0) if curr_status == "Up" else (0, 0, 255) if curr_status == "Down" else (255, 165, 0)

            if curr_status == "Up":
                status_color = (0, 255, 0)     
            elif curr_status == "Down":
                status_color = (0, 0, 255)    
            else:
                status_color = (255, 165, 0)   


            cv2.putText(frame,curr_status,(100, 100),cv2.FONT_HERSHEY_COMPLEX,2,status_color,2)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )


        
        cv2.imshow("Bicep Curl Form Check", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
