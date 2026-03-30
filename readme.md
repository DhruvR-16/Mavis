# Mavis - AI-Powered Personal Fitness Trainer

> **Real-time pose estimation and strict form coaching engine powered by Computer Vision and Deep Learning.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)](https://google.github.io/mediapipe/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-yellow)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Active-green)]()

## 📖 Overview

**Mavis** is an intelligent fitness assistant that provides **gym-grade coaching** right from any standard webcam device. It goes way beyond simple rep counting by utilizing **MediaPipe Landmark Tracking**, localized **Geometric Inference**, and a **Strict Coach State Machine** to enforce perfect form execution dynamically in real-time.

---

## 🚀 Key Features

### Clean Professional Dashboard
- **Standard SaaS Aesthetic**: Mavis uses a beautifully polished "Clean Dark" design (Zinc backgrounds and True Blue accents) inspired by professional analytics tools like Vercel and Stripe. 
- **Real-Time Biometrics Grid**: See live skeletal feedback and dynamic metrics (Current Phase, Flexion Angle, Total Rep Time) overlaid flawlessly onto the video feed.
- **Dynamic Skeletal Feedback**: The tracking skeleton natively changes color based on your form execution—glowing **Emerald Green** for perfect reps and flashing **Ruby Red** for bad reps or cheating.

### Expanded Exercise Library
1. **Bicep Curls**
   - **Full Range of Motion**: Demands full extension (>160°) and peak contraction (<45°).
   - **Tempo Control**: Flags reps that are "Too Fast" (<1 second) to ensure muscle time-under-tension.
   - **Elbow Stability**: Specifically tracks coordinates to alarm you if your elbow drifts forward or backwards during a heavy curl.
2. **Shoulder Press**
   - **Bilateral Asymmetry Tracking**: Measures both shoulders independently and alerts you if you lift unevenly (pressing harder on the right than the left).
   - **Controlled Lockout**: Validates top-range hold times and tracks violent, uncontrolled negative drops.

---

## ⚙️ Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DhruvR-16/Mavis
   cd Mavis
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Analyzer**
   ```bash
   # Launch interactively
   python run.py

   # Or run specific exercise directly
   python run.py bicep
   python run.py shoulder
   ```
   _Press 'q' to quit the window._

---

## 🖥️ Running the Web Interface

Mavis is built natively for modern browsers via local deployment.

1. **Start Local Server**
   ```bash
   python3 -m http.server 8000
   ```
2. **Open the Dashboard**
   Go to: [http://localhost:8000/frontend/index.html](http://localhost:8000/frontend/index.html)
3. **Usage**
   Select your exercise from the Clean Dashboard. Position yourself within the camera frame, let the MediaPipe engine track your skeleton, and start your workout. The camera will automatically pause itself if no humans are detected for 20 seconds.

---

## 🧠 How the Engine Works

1. **Input**: Webcam captures raw frames (macOS/Safari optimized manual request capturing).
2. **Pose Extraction**: MediaPipe isolates 33 advanced 3D bodily landmarks.
3. **Geometric Tracking**: Extracts crucial workout geometry (elbow flexion, glute-shoulder lines, bicep-torso deviation).
4. **State Machine Inference**: 
   - Uses precise logic to map out "DOWN", "UP", and "MID" eccentric/concentric workout states based on current skeletal ranges.
   - Triggers `BadRepCount++` whenever cheating triggers (swinging torso, elbow drifting) are flagged.
5. **Live Feedback UI**: Instantly reflects all data back into the DOM, changing the skeletal hues and rendering precise form correction notes in the Live Coaching Box.
