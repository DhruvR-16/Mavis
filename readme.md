# Mavis - AI-Powered Personal Fitness Trainer

> **Real-time pose estimation and form correction engine powered by Computer Vision and Deep Learning.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)](https://google.github.io/mediapipe/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-yellow)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Prototype-green)]()

## 📖 Overview

**Mavis** is an intelligent fitness assistant designed to analyze exercise form in real-time using a standard camera. Unlike basic rep counters, Mavis uses **MediaPipe** for high-fidelity body tracking and **LSTMs (Long Short-Term Memory networks)** to understand the temporal context of movements.

The goal is to provide **gym-grade coaching on edge devices** (mobile/laptop) without needing expensive hardware or wearables.

---

## Key Features

### Current Capabilities

- **Real-Time Pose Estimation:** Tracks 33 3D body landmarks at using MediaPipe.
- **Robust Rep Counting:** Uses Finite State Machines to count reps only when full range of motion is achieved (Start → Eccentric → Inflection → Concentric).
- **Form Correction:** Calculates geometric angles to detect common mistakes (e.g., "Elbows flaring," "Not deep enough").
- **Visual Feedback:** Dynamic skeleton overlay that changes color (Green/Red) based on form accuracy.

### Roadmap (In Progress)

- **LSTM Action Recognition:** Automatically detects _which_ exercise you are performing (Bicep Curl vs. Shoulder Press) without manual selection.
- **Edge AI Deployment:** Porting the inference engine to **React Native** for mobile deployment.
- **Voice Coaching:** Text-to-Speech integration for audio feedback ("Lower your hips!").

---

## Tech Stack

- **Computer Vision:** OpenCV, MediaPipe Pose
- **Deep Learning:** TensorFlow/Keras (for LSTM Action Classification)
- **Math & Data:** NumPy (Vectorized angle calculations)
- **Backend (Future):** FastAPI (for user profiles and history)

---

## Installation & Setup

Follow these steps to get Mavis running on your local machine.

### Prerequisites

- **Python 3.8** or higher installed.
- A working webcam.

### Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/DhruvR-16/Mavis
cd Mavis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Mavis Frontend

Mavis includes a professional web interface for selecting exercises and monitoring your form.

1. **Start the local server** (required for camera access):
   ```bash
   python3 -m http.server 8000
   ```
2. **Open the Interface**:
   Go to [http://localhost:8000/frontend/index.html](http://localhost:8000/frontend/index.html) in your browser.
3. **Usage**:
   - Allow camera access when prompted.
   - Select "Bicep Curls" to start the analysis.
   - The camera will auto-pause after 20 seconds of inactivity to save power.

---

### Pipeline

1. Input: The webcam captures a frame.
2. Pose Extraction: MediaPipe extracts 33 3D landmarks $(x, y, z)$ on the body.
3. Smoothing: A One-Euro Filter is applied to the data to remove camera jitter/shaking.
4. Geometry Analysis: The engine calculates specific joint angles (e.g., Hip-Knee-Ankle for squats).
5. State Machine Logic:
   - Phase 1 (Start): User is in a neutral position (Angle > 160°).
   - Phase 2 (Eccentric): The muscle lengthens (Angle decreases).
   - Phase 3 (Inflection): Velocity hits 0 at the bottom of the rep (Angle < Threshold).
   - Phase 4 (Concentric): The muscle contracts (Angle increases).
6. Feedback: If all phases are valid, the rep count increases and the skeleton turns green. If form breaks (e.g., swinging back), the skeleton turns red.
