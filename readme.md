# Mavis - AI-Powered Personal Fitness Trainer

> **Real-time pose estimation and strict form coaching from any standard webcam.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)](https://google.github.io/mediapipe/)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev/)
[![Status](https://img.shields.io/badge/Status-Beta-yellow)]()

## 📖 Overview

**Mavis** is a fitness assistant that coaches your lifting form in real time using a
webcam. It tracks 33 skeletal landmarks with **MediaPipe Pose**, derives joint
geometry from them, and runs each rep through a **state machine** that counts reps,
detects form faults, and scores rep quality out of 100.

All pose processing runs **entirely on your device**. No video is uploaded anywhere,
and there is no backend server.

---

## 🧩 Two ways to run it

Mavis currently ships as two separate front ends over the same idea. They do **not**
share code today — see [Known Issues](#-known-issues).

| | Web app | Desktop app |
|---|---|---|
| Location | `frontend/` | `run.py`, `analyzer/`, `exercises/` |
| Stack | React 19 + Vite + MediaPipe Tasks | Python + OpenCV + MediaPipe |
| Voice coaching | ✅ Web Speech API | ❌ (queued but never played) |
| Rep quality chart | ✅ | ❌ (text summary only) |
| Recommended | **Yes** | For development / model work |

---

## 🚀 Key Features

### Live coaching dashboard
- **Real-time metrics**: rep count, good/bad split, live rep-quality score, session timer.
- **Dynamic skeleton**: the overlay turns **green** on a clean rep and **red** the moment a fault is detected.
- **Voice cues** (web only) via the Web Speech API — rep counts, fault callouts, and rest timers.
- **Program mode**: sets × reps with rest timers and wrong-set detection — a set with 3+ bad reps is invalidated and must be repeated.

### Exercise library
1. **Bicep Curls**
   - **Range of motion**: requires full extension (>145°) and peak contraction (<60°), with a ±15° human-error tolerance band.
   - **Tempo control**: flags reps completed faster than **0.8 s** to enforce time under tension.
   - **Elbow stability**: flags lateral elbow drift beyond 18% of your shoulder width, measured against a calibrated anchor.
2. **Shoulder Press**
   - **Bilateral asymmetry**: measures both arms independently and flags an imbalance beyond 20°.
   - **Lockout and depth**: validates top-range extension and bottom-range depth.

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/DhruvR-16/Mavis && cd Mavis
```

### 2. Web app (recommended)

Requires **Node.js `^20.19` or `>=22.12`** — this is Vite 8's constraint, so Node
20.0–20.18 and 22.0–22.11 will fail. Check with `node --version`.

```bash
cd frontend && npm install && npm run dev
```

Open the URL Vite prints (usually <http://localhost:5173>) and allow camera access.

> The web app needs no Python environment at all. MediaPipe is loaded in-browser.

### 3. Desktop app

Requires **Python 3.10, 3.11, or 3.12**.

> ⚠️ **Python 3.13+ will not work.** MediaPipe 0.10.14 publishes no wheels for it,
> and `pip` fails with `No matching distribution found for mediapipe==0.10.14`.
> On many systems `python3` is already 3.13, so name the version explicitly:

```bash
python3.12 -m venv venv         # or python3.10 / python3.11
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Check what you got with `python --version` after activating.

```bash
python run.py                   # pick interactively
python run.py bicep             # free mode
python run.py shoulder --sets 4 --reps 8 --rest 90   # program mode
```

Press **`q`** to quit the OpenCV window.

> TensorFlow is **not** required and is intentionally absent from
> `requirements.txt`. The analyzers detect its absence and run in geometry-only
> mode, which is the supported path.

### 4. Retraining the models (optional)

```bash
pip install -r requirements-train.txt
python tools/train_bicep_lstm.py
```

Reads `exercises/bicep/data/bicep_angles.csv` and writes to `exercises/bicep/models/`.

---

## 🧠 How the engine works

1. **Input** — webcam frames, mirrored horizontally so the view matches a mirror.
2. **Pose extraction** — MediaPipe isolates 33 body landmarks per frame.
3. **Smoothing** — an exponential moving average over landmark positions suppresses jitter.
4. **Calibration** — a 3-second standing hold captures your shoulder width and neutral elbow position, so all subsequent drift measurements are body-relative rather than pixel-relative.
5. **Geometric features** — joint angles (elbow, shoulder), torso lean, bilateral symmetry, and normalized elbow drift.
6. **State machine** — maps the primary joint angle onto `DOWN` → `UP` → `DOWN` transitions to count reps, applying a ±15° tolerance band at each threshold.
7. **Rep scoring** — each completed rep starts at 100 and is docked for range-of-motion shortfalls, fast tempo, elbow drift, and torso swing. Reps below 60 count as bad.

---

## ⚠️ Known Issues

This project is pre-1.0 and these are tracked, not hidden:

- **The bundled bicep classifier is non-functional.** It was trained on a single-class
  dataset (every row labeled `Bicep Curl`), so it always predicts that one class at
  100% confidence. Its input features also do not match the ones the scaler was fit
  on. It is slated for removal — the geometric engine above is what actually works.
- **Shoulder-press lockout and depth scoring is inactive in the web app**, so shoulder
  reps score higher than they should.
- **Calibration does not verify you are holding still** — it waits 3 seconds and
  snapshots whatever pose you are in.
- **Desktop bicep analysis tracks the left arm only.**
- **Session history is not persisted** beyond the single most recent session
  (`localStorage`).
- The web app currently uses MediaPipe's deprecated legacy Solutions API and loads
  it from an unpinned CDN path.

---

## 📁 Layout

```
run.py                    Desktop launcher (CLI)
analyzer/                 Shared desktop engine
  base_analyzer.py          Program mode, scoring, calibration, voice queue
  feature_extractor.py      Landmark → geometric feature vector
exercises/
  bicep/analyzer.py         Bicep curl state machine + models/
  shoulder/analyzer.py      Shoulder press state machine + models/
tools/train_bicep_lstm.py Model training
frontend/                 React web app (self-contained)
  src/pages/Workout.tsx     Live session: pose loop, scoring, UI
  src/pages/Home.tsx        Exercise and program selection
```
