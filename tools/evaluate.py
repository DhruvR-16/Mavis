"""
Evaluate the form engines against the recorded video library.

There are no labelled rep counts, so accuracy is measured three ways, none of
which require hand-labelling:

1. Agreement with an INDEPENDENT reference counter — scipy peak detection over
   the primary joint angle. It shares no code with the state machine, so
   agreement is evidence and disagreement flags candidates for real errors.
   It is a proxy, not truth: it can be wrong too, and both can be wrong the
   same way if MediaPipe mistracks.

2. False positives on exercises the analyzer must NOT count. A curl analyzer
   shown a squat should report zero reps. That ground truth is free.

3. Pose-detection reliability — how often MediaPipe finds a body at all, which
   upper-bounds everything downstream.

Usage:
    python tools/evaluate.py [--per-category N] [--cache DIR]
"""

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer.base_analyzer import CalibrationState  # noqa: E402
from analyzer.feature_extractor import calculate_angle  # noqa: E402
from exercises.bicep.analyzer import BicepAnalyzer  # noqa: E402
from exercises.shoulder.analyzer import ShoulderAnalyzer  # noqa: E402
from tests.replay import Landmark, virtual_clock  # noqa: E402

VIDEO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "videos"
)

# Categories the arm analyzers are expected to count.
POSITIVE = {
    "bicep": ["barbell biceps curl", "hammer curl"],
    "shoulder": ["shoulder press"],
}

# Categories where the correct answer is zero reps: either the arms barely
# articulate, or the movement is driven by the legs/torso.
NEGATIVE = ["squat", "deadlift", "leg extension", "leg raises", "russian twist", "plank"]

# Arm movements that are NOT the target exercise — the hardest false positives,
# since the elbow does flex through a similar range.
CONFUSABLE = ["lat pulldown", "tricep Pushdown", "bench press", "t bar row"]


# ── Landmark extraction (cached; this is the slow part) ──────────────────────


def extract(video_path: str, cache_dir: str, limit: int = 600):
    key = os.path.join(cache_dir, video_path.replace(os.sep, "__").replace(" ", "_") + ".npz")
    if os.path.exists(key):
        with np.load(key) as z:
            return z["frames"], float(z["fps"]), int(z["total"])

    cap = cv2.VideoCapture(os.path.join(VIDEO_ROOT, video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, total = [], 0

    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and total < limit:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            res = pose.process(img)
            if res.pose_landmarks:
                frames.append([[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark])
    cap.release()

    arr = np.array(frames, dtype=np.float32) if frames else np.zeros((0, 33, 4), dtype=np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(key, frames=arr, fps=fps, total=total)
    return arr, fps, total


def to_landmarks(arr):
    return [[Landmark(*row) for row in frame] for frame in arr]


# ── Independent reference counter ────────────────────────────────────────────


def primary_signal(frames, kind: str):
    """Elbow angle series the reference counter operates on."""
    out = []
    for f in frames:
        left = calculate_angle(f[11], f[13], f[15])
        right = calculate_angle(f[12], f[14], f[16])
        # Bicep is single-arm: follow whichever arm is working (more flexed).
        out.append(min(left, right) if kind == "bicep" else (left + right) / 2)
    return np.array(out)


def reference_reps(signal, kind: str) -> int:
    """
    Count reps by peak detection — deliberately unrelated to the state machine.

    A curl rep is a trough in elbow angle (arm flexes then extends); a press rep
    is a peak (arm locks out then bends). `prominence` demands a genuine
    excursion, so tracking jitter near a threshold isn't counted.
    """
    if len(signal) < 10:
        return 0
    if kind == "bicep":
        peaks, _ = find_peaks(-signal, prominence=45, distance=8)
        return int(sum(1 for p in peaks if signal[p] < 100))
    peaks, _ = find_peaks(signal, prominence=45, distance=8)
    return int(sum(1 for p in peaks if signal[p] > 130))


# ── Engine under test ────────────────────────────────────────────────────────


def engine_reps(frames, fps: float, kind: str):
    analyzer = BicepAnalyzer() if kind == "bicep" else ShoulderAnalyzer()
    with virtual_clock(fps) as clock:
        for i, raw in enumerate(frames):
            lms = analyzer.smoother.smooth(raw)
            if i == 0:
                analyzer.extractor.calibrate(lms)
                analyzer.calib_state = CalibrationState.COMPLETE
            analyzer.analyze_form(analyzer.extractor.get_features(lms), lms)
            clock.tick()
    return analyzer


# ── Reporting ────────────────────────────────────────────────────────────────


def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=15)
    ap.add_argument("--cache", default="/tmp/mavis_eval_cache")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    results = defaultdict(list)

    def run_category(category: str, kind: str, role: str):
        folder = os.path.join(VIDEO_ROOT, category)
        if not os.path.isdir(folder):
            return
        for name in sorted(os.listdir(folder))[: args.per_category]:
            path = os.path.join(category, name)
            try:
                arr, fps, total = extract(path, args.cache)
            except Exception:
                continue
            if total == 0:
                continue

            detection = len(arr) / total
            row = {
                "video": path, "category": category, "role": role, "kind": kind,
                "frames_total": total, "frames_with_pose": len(arr),
                "detection_rate": detection,
            }

            if len(arr) < 20:
                row.update({"engine_reps": 0, "reference_reps": 0, "too_short": True})
                results[role].append(row)
                continue

            frames = to_landmarks(arr)
            analyzer = engine_reps(frames, fps, kind)
            row["engine_reps"] = analyzer.total_reps
            row["reference_reps"] = reference_reps(primary_signal(frames, kind), kind)
            row["qualities"] = [r.quality for r in analyzer.rep_history]
            row["faults"] = [f for r in analyzer.rep_history for f in r.faults]
            results[role].append(row)
            print(".", end="", flush=True)

    print("extracting + evaluating (cached after first run)", file=sys.stderr)
    for kind, cats in POSITIVE.items():
        for c in cats:
            run_category(c, kind, f"positive:{kind}")
    for c in NEGATIVE:
        run_category(c, "bicep", "negative")
    for c in CONFUSABLE:
        run_category(c, "bicep", "confusable")
    print(file=sys.stderr)

    report(results)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({k: v for k, v in results.items()}, f, indent=1)


def report(results):
    print("\n" + "=" * 72)
    print("MAVIS FORM ENGINE — EVALUATION")
    print("=" * 72)

    # ── Detection reliability ────────────────────────────────────────────────
    every = [r for rows in results.values() for r in rows]
    det = [r["detection_rate"] for r in every]
    print(f"\nPOSE DETECTION  ({len(every)} videos)")
    print(f"  frames with a detected body: median {st.median(det)*100:.1f}%  "
          f"p10 {pct(det,10)*100:.1f}%  min {min(det)*100:.1f}%")
    print(f"  videos below 90% detection: {sum(1 for d in det if d < 0.9)}/{len(det)}")

    # ── Rep counting vs independent reference ────────────────────────────────
    for role in sorted(k for k in results if k.startswith("positive")):
        rows = [r for r in results[role] if not r.get("too_short")]
        if not rows:
            continue
        kind = role.split(":")[1]
        diffs = [r["engine_reps"] - r["reference_reps"] for r in rows]
        exact = sum(1 for d in diffs if d == 0)
        within1 = sum(1 for d in diffs if abs(d) <= 1)
        missed = sum(1 for r in rows if r["engine_reps"] == 0 and r["reference_reps"] > 0)

        print(f"\nREP COUNTING — {kind}  ({len(rows)} videos, "
              f"{sum(r['reference_reps'] for r in rows)} reference reps)")
        print(f"  exact agreement with reference: {exact}/{len(rows)} = {100*exact/len(rows):.0f}%")
        print(f"  within +/-1 rep:                {within1}/{len(rows)} = {100*within1/len(rows):.0f}%")
        print(f"  mean signed error:              {st.mean(diffs):+.2f} reps "
              f"(negative = engine under-counts)")
        print(f"  mean absolute error:            {st.mean([abs(d) for d in diffs]):.2f} reps")
        print(f"  videos where engine found 0 but reference found reps: {missed}/{len(rows)}")

        quals = [q for r in rows for q in r.get("qualities", [])]
        if quals:
            print(f"  quality scores: median {st.median(quals):.0f}  "
                  f"p10 {pct(quals,10):.0f}  p90 {pct(quals,90):.0f}  "
                  f"unique values {len(set(quals))}")

    # ── False positives ──────────────────────────────────────────────────────
    for role, label in (("negative", "NON-ARM EXERCISES (correct answer: 0 reps)"),
                        ("confusable", "OTHER ARM EXERCISES (correct answer: 0 reps)")):
        rows = [r for r in results.get(role, []) if not r.get("too_short")]
        if not rows:
            continue
        with_reps = [r for r in rows if r["engine_reps"] > 0]
        total_false = sum(r["engine_reps"] for r in rows)
        print(f"\n{label}  ({len(rows)} videos)")
        print(f"  videos with >=1 false rep: {len(with_reps)}/{len(rows)} = "
              f"{100*len(with_reps)/len(rows):.0f}%")
        print(f"  total spurious reps counted: {total_false}")
        by_cat = defaultdict(lambda: [0, 0])
        for r in rows:
            by_cat[r["category"]][0] += r["engine_reps"]
            by_cat[r["category"]][1] += 1
        for cat, (reps, n) in sorted(by_cat.items(), key=lambda kv: -kv[1][0]):
            print(f"     {cat:22} {reps:4} reps across {n:3} videos")

    # ── Feedback quality ─────────────────────────────────────────────────────
    pos_rows = [r for k, rows in results.items() if k.startswith("positive") for r in rows]
    faults = [f for r in pos_rows for f in r.get("faults", [])]
    reps = sum(len(r.get("qualities", [])) for r in pos_rows)
    print(f"\nFEEDBACK ON REAL REPS  ({reps} reps)")
    if reps:
        print(f"  reps with at least one fault: {len(faults)}/{reps} = {100*len(faults)/reps:.0f}%")
        counts = defaultdict(int)
        for f in faults:
            counts[f] += 1
        for f, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"     {c:4} x {f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
