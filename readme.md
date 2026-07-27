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
| Voice coaching | ✅ Web Speech API | ❌ (not implemented) |
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
> `requirements.txt`. The analyzers are geometry-only — there is no model
> loading or AI classification step at runtime.

### 4. Training tooling (not currently useful — see below)

```bash
pip install -r requirements-train.txt
python tools/train_bicep_lstm.py
```

Reads `exercises/bicep/data/bicep_angles.csv` and would write to
`exercises/bicep/models/`, but the checked-in CSV has only one label
(`Bicep Curl`), so the script now refuses to save a model — training a
classifier on one class produces a model that always predicts that class,
which is not a classifier. Nothing in the app loads this model's output
regardless. The script is kept for whoever captures real multi-class data
later; see [Known Issues](#-known-issues).

---

### 5. Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite replays landmark fixtures recorded from real workout video
(`fixtures/*.json`, produced by `tools/extract_fixture.py`) through the
analyzers and asserts rep counts, quality scores, and faults. Replay uses a
virtual clock derived from the source video's frame rate, so tempo-dependent
scoring behaves as it would live rather than completing instantly.

---

## 🔗 Shared exercise definitions

Thresholds, tolerances, scoring weights, and fault messages live in one file —
[`exercises.json`](exercises.json) — read by **both** runtimes:

| Runtime | Loader |
|---|---|
| Python | `analyzer/exercise_config.py` |
| Web | `frontend/src/engine/config.ts` |

Changing a number there changes it everywhere, and `tests/test_shared_config.py`
fails if a runtime stops tracking the shared file. This exists because the two
engines had already drifted: the web app was measuring the **shoulder** joint
for a press while Python measured the **elbow**, applying identical 165°/75°
thresholds to both. The shared definition settles it on the elbow.

Adding an exercise is mostly a data change, though each runtime still needs a
state machine for it.

---

## 🧠 How the engine works

1. **Input** — webcam frames, mirrored horizontally so the view matches a mirror.
2. **Pose extraction** — MediaPipe isolates 33 body landmarks per frame.
3. **Smoothing** — an exponential moving average over landmark positions suppresses jitter.
4. **Calibration** — a 3-second standing hold captures your shoulder width and neutral elbow position, so all subsequent drift measurements are body-relative rather than pixel-relative. The hold must be genuinely still: movement past a small tolerance restarts the countdown instead of calibrating against whatever pose is held at the deadline.
5. **Geometric features** — joint angles (elbow, shoulder), torso lean, bilateral symmetry, and normalized elbow drift.
6. **Active-arm detection** (bicep curl) — whichever elbow is more contracted while at rest is picked as the working arm and locked for the duration of the rep, so curling with either arm is tracked correctly.
7. **State machine** — maps the primary joint angle onto `DOWN` → `UP` → `DOWN` transitions to count reps, applying a ±15° tolerance band at each threshold.
8. **Rep scoring** — each completed rep starts at 100 and is docked for range-of-motion shortfalls, fast tempo, elbow drift, and torso swing. Reps below 60 count as bad.

---

## ⚠️ Known Issues

This project is pre-1.0 and these are tracked, not hidden:

- **There is no AI exercise classifier.** One existed briefly but was removed: it was
  trained on a single-class dataset (every row labeled `Bicep Curl`), so it always
  predicted that one class at 100% confidence regardless of input — indistinguishable
  from not running at all, at the cost of a `model.predict()` call every frame. Both
  analyzers are geometric-only now. `tools/train_bicep_lstm.py` remains for anyone who
  captures real multi-class data.
- **"Tempo" measures only the eccentric, not the whole rep.** The timer starts at
  peak contraction / lockout and stops at full extension / the bottom, so it clocks
  the lowering phase alone. `RepResult.duration_sec` and the session summary's
  `rep_durations` nevertheless present it as the rep duration. On
  `fixtures/shoulder_press.json` — four presses over seven seconds, ~1.75 s each —
  the measured intervals are 0.53–0.60 s, so all four trip the 0.8 s threshold.
  Either the threshold wants retuning for eccentric-only measurement, or the timer
  should span the full rep. Both engines behave identically here.
- **The web engine's analysis logic still lives inside `Workout.tsx`**, so unlike
  the Python engine it has no automated test coverage. Extracting it into a pure
  module is what would let the same fixtures verify both engines agree.
- **Angles are computed in 2D**, discarding MediaPipe's `z`. Standing off-axis to
  the camera skews every measurement.
- **Session history is not persisted** beyond the single most recent session
  (`localStorage`).

---

## 📁 Layout

```
exercises.json            Shared thresholds/scoring — read by BOTH runtimes
run.py                    Desktop launcher (CLI)
analyzer/                 Shared desktop engine
  base_analyzer.py          Program mode, scoring, calibration
  feature_extractor.py      Landmark → geometric feature vector
  exercise_config.py        Loads exercises.json
exercises/
  bicep/analyzer.py         Bicep curl state machine (geometric only)
  shoulder/analyzer.py      Shoulder press state machine (geometric only)
fixtures/                 Landmark sequences recorded from real video
tests/                    pytest suite replaying those fixtures
tools/extract_fixture.py  Video → landmark fixture
tools/train_bicep_lstm.py Classifier training — not currently useful, see Known Issues
frontend/                 React web app
  src/engine/config.ts      Loads exercises.json
  src/pages/Workout.tsx     Live session: pose loop, scoring, UI
  src/pages/Home.tsx        Exercise and program selection
```
