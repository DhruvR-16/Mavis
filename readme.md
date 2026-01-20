# Mavis - AI-Powered Personal Fitness Trainer

> **Real-time pose estimation and strict form coaching engine powered by Computer Vision and Deep Learning.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)](https://google.github.io/mediapipe/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-yellow)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Active-green)]()

## 📖 Overview

**Mavis** is an intelligent fitness assistant that provides **gym-grade coaching** on standard devices. It goes beyond simple rep counting by using **LSTM networks** to recognize actions and a **Strict Coach State Machine** to enforce perfect form.

---

## 🚀 Key Features

### Intelligent Analysis

- **Action Recognition**: Uses a custom LSTM model (`models/bicep_lstm.h5`) to verify you are performing the correct exercise (e.g., "Bicep Curl").
- **Anti-Jitter Smoothing**: Implements `LandmarkSmoother` (Exponential Moving Average) to stabilize skeletal tracking and prevent false counts.

### strict Coach Mode

Mavis enforces strict form rules required for hypertrophy:

- **Full Range of Motion**: Requires full extension (>160°) and peak contraction (<45°).
- **Tempo Control**: Flags reps that are "Too Fast" (<1 second) to ensure time-under-tension.
- **Elbow Stability**: Detects and alerts if elbows drift sideways during curls.
- **Bad Rep Counter**: distinct red counter for cheating/poor form reps.

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

3. **Run the Analyzer (Python Native)**
   ```bash
   python biceps.py
   ```
   _Press 'q' to quit the window._

---

## Running the Web Interface

Mavis includes a professional web frontend for exercise selection.

1. **Start Local Server**
   ```bash
   python3 -m http.server 8000
   ```
2. **Open in Browser**
   Go to: [http://localhost:8000/frontend/index.html](http://localhost:8000/frontend/index.html)
3. **Usage**
   Select your exercise. The camera feed will auto-pause after 20 seconds of inactivity.

---

## How It Works

1. **Input**: Webcam captures frame; flipped for mirror view.
2. **Pose Extraction**: MediaPipe tracks 33 3D landmarks.
3. **Smoothing**: `LandmarkSmoother` applies EMA filter to reduce noise.
4. **Feature Extraction**: Calculates 10 key geometric angles.
5. **AI Inference**: LSTM model confirms action is "Bicep Curl".
6. **State Machine**:
   - **DOWN**: Waiting for flexion. Check Extension > 160°.
   - **UP**: Waiting for extension. Check Contraction < 45°.
   - **Strict Checks**: If Velocity is high or Elbow Drift > 10%, increment **Bad Reps**.
7. **Feedback**: UI updates with real-time cues ("Squeeze!", "Lower Slowly").
