#!/usr/bin/env python3
"""Extract approximate 3D pose landmarks from a video for Bunkai Explorer."""
import os
import sys

# Silence internal C++ glog / TFLite / absl messages to stderr
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_MIN_SEVERITY"] = "3"

try:
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
except Exception:
    pass

import argparse
import json
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

LANDMARKS = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

def main():
    ap = argparse.ArgumentParser(description="Extract 3D pose landmarks from video for Bunkai Explorer")
    ap.add_argument("video", help="Input video file path (.mp4, .mov, etc.)")
    ap.add_argument("-o", "--output", help="Output JSON path (default: <video_stem>_pose.json)")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (process every Nth frame, default 1)")
    ap.add_argument("--smooth", type=int, default=2, help="Temporal landmark smoothing radius (default 2)")
    ap.add_argument("--preview-video", "-p", help="Optional path to save video with skeleton overlay for visual inspection")
    args = ap.parse_args()

    src = Path(args.video)
    if not src.is_file():
        raise SystemExit(f"Video not found: {src}")
    dst = Path(args.output) if args.output else src.with_name(src.stem + "_pose.json")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Setup optional preview video writer
    writer = None
    if args.preview_video:
        prev_path = Path(args.preview_video)
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = fps / max(1, args.stride)
        writer = cv2.VideoWriter(str(prev_path), fourcc, effective_fps, (width, height))
        print(f"Recording preview video to: {prev_path}")

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles

    rows = []
    fi = 0
    analyzed = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx = fi
            fi += 1
            if idx % max(1, args.stride):
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            row = {"frame": idx, "time": round(idx / fps, 6), "detected": False, "landmarks": {}, "visibility": {}}

            if res.pose_world_landmarks:
                row["detected"] = True
                for name, li in LANDMARKS.items():
                    p = res.pose_world_landmarks.landmark[li]
                    row["landmarks"][name] = [round(float(p.x), 6), round(float(-p.y), 6), round(float(-p.z), 6)]
                    row["visibility"][name] = round(float(p.visibility), 4)

            if writer is not None:
                if res.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        res.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                writer.write(frame)

            rows.append(row)
            analyzed += 1
            if analyzed % 100 == 0:
                print(f"Processed {analyzed} frames")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Preview video saved: {args.preview_video}")

    if not rows:
        raise SystemExit("No frames read from video")

    # Smooth each detected landmark trajectory
    for name in LANDMARKS:
        ids = [i for i, r in enumerate(rows) if name in r["landmarks"]]
        if not ids or args.smooth <= 0:
            continue
        vals = np.array([rows[i]["landmarks"][name] for i in ids], dtype=float)
        sm = vals.copy()
        for j in range(len(vals)):
            lo = max(0, j - args.smooth)
            hi = min(len(vals), j + args.smooth + 1)
            sm[j] = np.mean(vals[lo:hi], axis=0)
        for j, i in enumerate(ids):
            rows[i]["landmarks"][name] = [round(float(v), 6) for v in sm[j]]

    detected = sum(r["detected"] for r in rows)
    data = {
        "schema": "bunkai-explorer-pose-capture",
        "schemaVersion": 1,
        "sourceVideo": src.name,
        "source": "MediaPipe Pose world landmarks",
        "coordinateSystem": "estimated world coordinates; x right, y up, z depth (approximate)",
        "input": {
            "fps": fps,
            "frameCount": count,
            "width": width,
            "height": height,
            "analyzedFrames": len(rows),
            "stride": args.stride
        },
        "fpsAnalyzed": fps / max(1, args.stride),
        "durationSeconds": rows[-1]["time"],
        "detectionRate": round(detected / len(rows), 4),
        "landmarkNames": LANDMARKS,
        "frames": rows,
        "notes": [
            "Estimated pose, not motion-capture ground truth.",
            "Single-camera depth and foot contact can be inaccurate, especially during turns.",
            "This JSON is intermediate data, not a Three.js AnimationClip or rig-ready GLB."
        ]
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {dst}; pose detected in {detected}/{len(rows)} frames ({100*detected/len(rows):.1f}%)")

if __name__ == "__main__":
    main()
