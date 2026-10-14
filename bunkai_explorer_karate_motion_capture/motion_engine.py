"""
Real-time Motion Capture Engine for Bunkai Explorer Karate Avatar.
Captures webcam/video, tracks 3D Pose and 3D Hands (21 finger joints per hand),
calculates natural head posture, and streams live landmarks and finger curls to Three.js.
"""
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

import time
import math
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import mediapipe as mp
import numpy as np

# Landmark Indices for MediaPipe Pose (33 points)
POSE_LM = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

POSE_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder", (255, 100, 100)),
    ("left_shoulder", "left_hip", (255, 100, 100)),
    ("right_shoulder", "right_hip", (255, 100, 100)),
    ("left_hip", "right_hip", (255, 100, 100)),
    # Left Arm
    ("left_shoulder", "left_elbow", (0, 220, 255)),
    ("left_elbow", "left_wrist", (0, 220, 255)),
    # Right Arm
    ("right_shoulder", "right_elbow", (255, 150, 0)),
    ("right_elbow", "right_wrist", (255, 150, 0)),
    # Left Leg
    ("left_hip", "left_knee", (180, 100, 255)),
    ("left_knee", "left_ankle", (180, 100, 255)),
    # Right Leg
    ("right_hip", "right_knee", (255, 100, 180)),
    ("right_knee", "right_ankle", (255, 100, 180)),
    # Head
    ("nose", "left_shoulder", (200, 200, 200)),
    ("nose", "right_shoulder", (200, 200, 200)),
]

# MediaPipe Hand Landmark Indices (21 points)
# 0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def calc_finger_curl(mcp: np.ndarray, pip: np.ndarray, dip: np.ndarray, tip: np.ndarray) -> float:
    """Calculate normalized curl (0.0=open, 1.0=fully clenched) from finger joint angles."""
    v1 = pip - mcp
    v2 = tip - pip
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    # 0 deg = straight, 130+ deg = full curl
    return float(np.clip(angle / 130.0, 0.0, 1.0))

def calc_thumb_curl(cmc: np.ndarray, mcp: np.ndarray, ip: np.ndarray, tip: np.ndarray, wrist: np.ndarray) -> float:
    """Calculate thumb flexion across palm."""
    v1 = mcp - cmc
    v2 = tip - ip
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    return float(np.clip(angle / 90.0, 0.0, 1.0))

class MotionCaptureEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Source settings
        self.source_mode = "webcam"  # "webcam" or "video"
        self.camera_index = 0
        self.video_path: Optional[str] = None
        self.is_paused = False
        self.loop_video = True

        # Processing parameters
        self.mirror_mode = True  # Natural for webcam
        self.smooth_factor = 0.35
        self.draw_skeleton = True
        self.clench_fists = True
        self.track_hands = True

        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_jpeg: Optional[bytes] = None
        self.current_pose_data: Dict[str, Any] = {}
        self.fps = 0.0

        # Recording
        self.is_recording = False
        self.recorded_frames: List[Dict[str, Any]] = []
        self.record_start_time = 0.0
        self.record_frame_count = 0

        # Posture Calibration (Neutral Metsuke Head & Torso)
        self.calibrated = False
        self.calib_samples: List[tuple] = []
        self.base_pitch = 0.0
        self.base_yaw = 0.0
        self.base_roll = 0.0

        # MediaPipe instances
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45
        )

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_source(self, mode: str, path_or_index: Any):
        with self.lock:
            self.source_mode = mode
            if mode == "webcam":
                self.camera_index = int(path_or_index)
                self.video_path = None
                self.mirror_mode = True
            else:
                self.video_path = str(path_or_index)
                self.mirror_mode = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.calibrated = False
            self.calib_samples = []

    def calibrate(self):
        with self.lock:
            self.calibrated = False
            self.calib_samples = []
        print("[MoCap Engine] Calibrating neutral posture...")

    def toggle_pause(self):
        with self.lock:
            self.is_paused = not self.is_paused
            return self.is_paused

    def start_recording(self):
        with self.lock:
            self.is_recording = True
            self.recorded_frames = []
            self.record_start_time = time.time()
            self.record_frame_count = 0

    def stop_recording(self, save_path: Optional[str] = None) -> Optional[str]:
        with self.lock:
            if not self.is_recording:
                return None
            self.is_recording = False
            frames = list(self.recorded_frames)

        if not frames:
            return None

        duration = frames[-1]["time"] if frames else 0.0
        detected_count = sum(1 for f in frames if f.get("detected"))
        data = {
            "schema": "bunkai-explorer-pose-capture",
            "schemaVersion": 1,
            "sourceVideo": "live_webcam" if self.source_mode == "webcam" else Path(self.video_path or "").name,
            "source": "Bunkai MoCap Live App",
            "coordinateSystem": "estimated world coordinates; x right, y up, z depth",
            "fpsAnalyzed": round(len(frames) / max(0.1, duration), 2),
            "durationSeconds": round(duration, 4),
            "detectionRate": round(detected_count / len(frames), 4),
            "landmarkNames": POSE_LM,
            "frames": frames,
            "notes": ["Recorded live from Bunkai MoCap Live application."]
        }

        if not save_path:
            out_dir = Path(__file__).resolve().parent / "recordings"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = str(out_dir / f"live_capture_{timestamp}.json")

        Path(save_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Recorded motion saved to: {save_path}")
        return save_path

    def _open_capture(self) -> bool:
        if self.source_mode == "webcam":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.CAP_DSHOW else cv2.CAP_ANY)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            if not self.video_path or not Path(self.video_path).is_file():
                return False
            self.cap = cv2.VideoCapture(self.video_path)
        return bool(self.cap and self.cap.isOpened())

    def _capture_loop(self):
        fps_counter = 0
        fps_timer = time.time()

        while self.running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            if self.cap is None or not self.cap.isOpened():
                if not self._open_capture():
                    time.sleep(0.5)
                    continue

            ok, frame = self.cap.read()
            if not ok:
                if self.source_mode == "video" and self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.1)
                    continue

            if self.mirror_mode and self.source_mode == "webcam":
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            # Square padding (letterbox) to satisfy MediaPipe square ROI requirement
            # and completely eliminate landmark_projection_calculator warnings
            dim_max = max(h, w)
            pad_top = (dim_max - h) // 2
            pad_bottom = dim_max - h - pad_top
            pad_left = (dim_max - w) // 2
            pad_right = dim_max - w - pad_left

            frame_square = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            rgb_square = cv2.cvtColor(frame_square, cv2.COLOR_BGR2RGB)

            # 1. Process Body Pose (Square 1:1 ROI)
            res_pose = self.mp_pose.process(rgb_square)

            # 2. Process Hands / Fingers
            res_hands = self.mp_hands.process(rgb_square) if self.track_hands else None

            pose_data: Dict[str, Any] = {
                "detected": False,
                "landmarks": {},
                "visibility": {},
                "head": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "fingers": {
                    "left": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "detected": False},
                    "right": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "detected": False}
                },
                "mode": self.source_mode,
                "fps": round(self.fps, 1),
                "time": time.time() - self.record_start_time if self.is_recording else 0.0
            }

            pts_3d: Dict[str, List[float]] = {}
            vis_dict: Dict[str, float] = {}

            if res_pose.pose_world_landmarks and res_pose.pose_landmarks:
                pose_data["detected"] = True
                lm_world = res_pose.pose_world_landmarks.landmark
                lm_2d = res_pose.pose_landmarks.landmark

                # Extract metric 3D positions in Three.js space:
                # X: avatar left (+X, right side of screen when facing avatar)
                # Y: up (+)
                # Z: forward towards camera (+)
                for name, idx in POSE_LM.items():
                    p = lm_world[idx]
                    x = p.x
                    y = -p.y  # Up is positive
                    z = -p.z  # Forward towards camera is positive
                    pts_3d[name] = [round(float(x), 4), round(float(y), 4), round(float(z), 4)]
                    vis_dict[name] = round(float(p.visibility), 3)

                pose_data["landmarks"] = pts_3d
                pose_data["visibility"] = vis_dict

                # Natural Head Rotation Calculation (Upright Metsuke gaze)
                nose = pts_3d["nose"]
                l_ear = pts_3d["left_ear"]
                r_ear = pts_3d["right_ear"]

                mid_ear = [(l_ear[i] + r_ear[i]) * 0.5 for i in range(3)]
                face_fwd = [nose[i] - mid_ear[i] for i in range(3)]
                ear_vec = [l_ear[i] - r_ear[i] for i in range(3)]

                raw_yaw = float(math.atan2(face_fwd[0], max(0.01, face_fwd[2])))
                raw_pitch = float(math.atan2(face_fwd[1], max(0.01, face_fwd[2])))
                raw_roll = float(math.atan2(ear_vec[1], max(0.05, abs(ear_vec[0]))))

                # Auto-calibration of resting head baseline posture
                if not self.calibrated:
                    self.calib_samples.append((raw_yaw, raw_pitch, raw_roll))
                    if len(self.calib_samples) >= 15:
                        self.base_yaw = float(np.mean([s[0] for s in self.calib_samples]))
                        self.base_pitch = float(np.mean([s[1] for s in self.calib_samples]))
                        self.base_roll = float(np.mean([s[2] for s in self.calib_samples]))
                        self.calibrated = True

                # Relative to neutral resting baseline
                yaw = raw_yaw - self.base_yaw
                pitch = raw_pitch - self.base_pitch
                roll = raw_roll - self.base_roll

                # Deadbands to eliminate camera jitter while user holds head steady
                if abs(yaw) < 0.04:
                    yaw = 0.0
                if abs(pitch) < 0.04:
                    pitch = 0.0
                if abs(roll) < 0.06:
                    roll = 0.0

                yaw = float(np.clip(yaw * 0.85, -0.65, 0.65))
                pitch = float(np.clip(pitch * 0.80, -0.40, 0.40))
                roll = float(np.clip(roll * 0.50, -0.30, 0.30))

                pose_data["head"] = {"yaw": round(yaw, 3), "pitch": round(pitch, 3), "roll": round(roll, 3)}

                # Draw body skeleton overlay on original frame
                if self.draw_skeleton:
                    self._draw_pose_overlay(frame, lm_2d, dim_max, pad_left, pad_top)

            # 3. Finger Tracking & Hand Overlays
            if res_hands and res_hands.multi_hand_landmarks and res_hands.multi_handedness:
                for hand_lms, handedness in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    # In mirror mode, MediaPipe classification matches body side directly
                    side = "left" if label == "Left" else "right"

                    h_pts = [np.array([p.x, p.y, p.z]) for p in hand_lms.landmark]

                    # Compute individual finger curls
                    curl_thumb = calc_thumb_curl(h_pts[1], h_pts[2], h_pts[3], h_pts[4], h_pts[0])
                    curl_index = calc_finger_curl(h_pts[5], h_pts[6], h_pts[7], h_pts[8])
                    curl_middle = calc_finger_curl(h_pts[9], h_pts[10], h_pts[11], h_pts[12])
                    curl_ring = calc_finger_curl(h_pts[13], h_pts[14], h_pts[15], h_pts[16])
                    curl_pinky = calc_finger_curl(h_pts[17], h_pts[18], h_pts[19], h_pts[20])

                    pose_data["fingers"][side] = {
                        "thumb": round(curl_thumb, 2),
                        "index": round(curl_index, 2),
                        "middle": round(curl_middle, 2),
                        "ring": round(curl_ring, 2),
                        "pinky": round(curl_pinky, 2),
                        "detected": True
                    }

                    # Draw hand skeleton overlay
                    if self.draw_skeleton:
                        self._draw_hand_overlay(frame, hand_lms, dim_max, pad_left, pad_top, (0, 255, 200) if side == "left" else (255, 200, 0))

            # Record if active
            if self.is_recording and pose_data["detected"]:
                rec_row = {
                    "frame": self.record_frame_count,
                    "time": round(time.time() - self.record_start_time, 4),
                    "detected": True,
                    "landmarks": pose_data["landmarks"],
                    "visibility": pose_data["visibility"],
                    "head": pose_data["head"],
                    "fingers": pose_data["fingers"]
                }
                self.recorded_frames.append(rec_row)
                self.record_frame_count += 1

            # Encode frame to JPEG
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                self.current_jpeg = jpeg.tobytes()
                self.current_pose_data = pose_data

            # FPS counter
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            time.sleep(0.012)

    def _draw_pose_overlay(self, frame: np.ndarray, lm_2d, dim_max: int, pad_left: int, pad_top: int):
        h, w = frame.shape[:2]
        coords: Dict[str, tuple] = {}
        for name, idx in POSE_LM.items():
            p = lm_2d[idx]
            if p.visibility > 0.35:
                px = int(p.x * dim_max - pad_left)
                py = int(p.y * dim_max - pad_top)
                if 0 <= px < w and 0 <= py < h:
                    coords[name] = (px, py)

        for a, b, color in POSE_CONNECTIONS:
            if a in coords and b in coords:
                cv2.line(frame, coords[a], coords[b], color, 3, cv2.LINE_AA)

        for name, pt in coords.items():
            color = (0, 255, 255) if "wrist" in name or "ankle" in name else (255, 255, 255)
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)

        status_text = f"FPS: {self.fps:.1f} | Tracking: OK" if coords else "Searching Pose..."
        color_hud = (0, 255, 120) if coords else (0, 140, 255)
        cv2.putText(frame, status_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, status_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_hud, 2, cv2.LINE_AA)

        if self.is_recording:
            rec_text = f"REC [{len(self.recorded_frames)} frames]"
            cv2.circle(frame, (w - 140, 30), 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, rec_text, (w - 120, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_hand_overlay(self, frame: np.ndarray, hand_lms, dim_max: int, pad_left: int, pad_top: int, color: tuple):
        h, w = frame.shape[:2]
        pts = []
        for p in hand_lms.landmark:
            px = int(p.x * dim_max - pad_left)
            py = int(p.y * dim_max - pad_top)
            pts.append((px, py))

        for a, b in HAND_CONNECTIONS:
            if 0 <= a < len(pts) and 0 <= b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for pt in pts:
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.current_jpeg

    def get_latest_pose(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.current_pose_data)
