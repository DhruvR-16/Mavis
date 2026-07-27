"""
Extract a MediaPipe pose-landmark sequence from a video into a JSON fixture.

The resulting fixture is replayed by both test suites — Python (tests/) and
TypeScript (frontend/src/engine/__tests__/) — so the two analysis engines are
verified against identical input. That is what keeps them from silently
diverging the way they did before.

Usage:
    python tools/extract_fixture.py <video> <output.json> [--name NAME]
                                    [--start N] [--max-frames N]

Fixtures are intentionally compact: coordinates are rounded to 5 decimals and
each landmark is a flat [x, y, z, visibility] array rather than an object.
"""

import argparse
import json
import os
import sys

import cv2
import mediapipe as mp

# Must match the analyzers' runtime settings, or the fixture won't represent
# what the app actually sees.
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
COORD_PRECISION = 5


def extract(video_path: str, start: int = 0, max_frames: int = 0) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    skipped = 0
    index = 0

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            index += 1
            if index <= start:
                continue

            # The analyzers mirror the webcam feed before processing; do the
            # same here so fixture coordinates match the live pipeline.
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = pose.process(image)

            if not result.pose_landmarks:
                skipped += 1
                continue

            frames.append([
                [
                    round(lm.x, COORD_PRECISION),
                    round(lm.y, COORD_PRECISION),
                    round(lm.z, COORD_PRECISION),
                    round(lm.visibility, 3),
                ]
                for lm in result.pose_landmarks.landmark
            ])

            if max_frames and len(frames) >= max_frames:
                break

    cap.release()

    if not frames:
        raise SystemExit(f"No pose detected in any frame of {video_path}")

    return {
        "source": os.path.basename(video_path),
        "fps": round(fps, 3),
        "frame_count": len(frames),
        "frames_without_pose": skipped,
        "landmark_format": ["x", "y", "z", "visibility"],
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Source video path")
    parser.add_argument("output", help="Destination .json path")
    parser.add_argument("--name", default=None, help="Fixture name (default: output filename stem)")
    parser.add_argument("--start", type=int, default=0, help="Skip this many leading frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap extracted frames (0 = all)")
    args = parser.parse_args()

    fixture = extract(args.video, start=args.start, max_frames=args.max_frames)
    fixture["name"] = args.name or os.path.splitext(os.path.basename(args.output))[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))

    size_kb = os.path.getsize(args.output) / 1024
    print(f"{fixture['name']}: {fixture['frame_count']} frames "
          f"({fixture['frames_without_pose']} without pose) -> {args.output} [{size_kb:.0f} KB]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
