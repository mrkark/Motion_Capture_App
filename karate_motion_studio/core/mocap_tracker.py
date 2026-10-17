"""
Unified Motion Capture Tracker for Karate Motion Studio.
Handles both Mode A (Offline Video File Analysis) and Mode B (Real-time Webcam Capture & Recording)
with MediaPipe Pose 3D, MediaPipe Hands, posture calibration, and accurate timestamps.
"""
from __future__ import annotations
import os
import sys
import time
import math
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable

class SilenceStderr:
    def __enter__(self):
        try:
            sys.stderr.flush()
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.orig_stderr = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
            self.orig_stderr = None

    def __exit__(self, *args):
        try:
            if self.orig_stderr is not None:
                os.dup2(self.orig_stderr, 2)
                os.close(self.orig_stderr)
            if self.null_fd is not None:
                os.close(self.null_fd)
        except Exception:
            pass


import cv2
import mediapipe as mp
import numpy as np


from core.project_schema import (
    STANDARD_JOINTS,
    FrameData,
    Metadata,
    KarateMotionProject,
)
from core.filters import Vector3DOneEuroFilter, OneEuroFilter

# MediaPipe Pose Landmark Indices
POSE_LM = {name: idx for idx, name in enumerate(STANDARD_JOINTS)}

POSE_DRAW_CONNECTIONS = [
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
    # Left Leg & Foot
    ("left_hip", "left_knee", (180, 100, 255)),
    ("left_knee", "left_ankle", (180, 100, 255)),
    ("left_ankle", "left_heel", (180, 100, 255)),
    ("left_heel", "left_foot_index", (180, 100, 255)),
    ("left_ankle", "left_foot_index", (180, 100, 255)),
    # Right Leg & Foot
    ("right_hip", "right_knee", (255, 100, 180)),
    ("right_knee", "right_ankle", (255, 100, 180)),
    ("right_ankle", "right_heel", (255, 100, 180)),
    ("right_heel", "right_foot_index", (255, 100, 180)),
    ("right_ankle", "right_foot_index", (255, 100, 180)),
    # Collarbones / Neck
    ("nose", "left_shoulder", (200, 200, 200)),
    ("nose", "right_shoulder", (200, 200, 200)),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]


def calc_finger_curl(
    mcp: np.ndarray,
    pip: np.ndarray,
    dip: np.ndarray,
    tip: np.ndarray,
    wrist: Optional[np.ndarray] = None,
) -> float:
    """Calculate normalized curl (0.0=open, 1.0=fully clenched) from joint angles and tip distance."""
    v1 = pip - mcp
    v2 = dip - pip
    v3 = tip - dip
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    n3 = float(np.linalg.norm(v3))
    if n1 < 1e-6 or n2 < 1e-6 or n3 < 1e-6:
        return 0.15

    dot1 = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    ang1 = math.degrees(math.acos(dot1))

    dot2 = float(np.clip(np.dot(v2, v3) / (n2 * n3), -1.0, 1.0))
    ang2 = math.degrees(math.acos(dot2))

    if wrist is not None:
        v0 = mcp - wrist
        n0 = float(np.linalg.norm(v0))
        dot0 = float(np.clip(np.dot(v0, v1) / (n0 * n1 + 1e-6), -1.0, 1.0))
        ang0 = math.degrees(math.acos(dot0))
        total_ang = ang0 * 0.40 + ang1 * 0.85 + ang2 * 0.80
    else:
        total_ang = ang1 * 0.90 + ang2 * 0.90

    finger_len = n1 + n2 + n3
    dist_tip_mcp = float(np.linalg.norm(tip - mcp))
    dist_curl = 1.0 - float(np.clip(dist_tip_mcp / (finger_len * 0.82 + 1e-6), 0.0, 1.0))

    ang_curl = float(np.clip(total_ang / 135.0, 0.0, 1.0))
    raw_curl = 0.5 * ang_curl + 0.5 * dist_curl

    # Highly responsive curve: sensitive to fine finger twitches and clenches tight to 1.0!
    curl = raw_curl ** 0.85
    if curl > 0.55:
        curl = 0.55 + (curl - 0.55) * 1.55
    return float(np.clip(curl, 0.0, 1.0))


def calc_thumb_curl(
    cmc: np.ndarray,
    mcp: np.ndarray,
    ip: np.ndarray,
    tip: np.ndarray,
    wrist: np.ndarray,
    index_mcp: Optional[np.ndarray] = None,
) -> float:
    """Calculate thumb flexion and adduction across palm."""
    v1 = mcp - cmc
    v2 = tip - ip
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.15

    dot = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    ang_curl = float(np.clip(angle / 75.0, 0.0, 1.0))

    if index_mcp is not None:
        thumb_len = n1 + float(np.linalg.norm(ip - mcp)) + n2
        dist_to_index = float(np.linalg.norm(tip - index_mcp))
        adduct_curl = 1.0 - float(np.clip(dist_to_index / (thumb_len * 1.2 + 1e-6), 0.0, 1.0))
        curl = 0.45 * ang_curl + 0.55 * adduct_curl
    else:
        curl = ang_curl

    if curl > 0.50:
        curl = 0.50 + (curl - 0.50) * 1.6
    return float(np.clip(curl, 0.0, 1.0))


def calc_phalanx_angles(
    mcp: np.ndarray,
    pip: np.ndarray,
    dip: np.ndarray,
    tip: np.ndarray,
    wrist: np.ndarray,
    u_across: np.ndarray,
) -> Tuple[List[float], float]:
    """
    Computes individual phalanx flexion angles [flex_mcp, flex_pip, flex_dip] in radians,
    and lateral abduction (spread) angle in radians.
    """
    v0 = mcp - wrist
    v1 = pip - mcp
    v2 = dip - pip
    v3 = tip - dip

    def _angle(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        dot = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return float(math.acos(dot))

    flex_mcp = _angle(v0, v1)
    flex_pip = _angle(v1, v2)
    flex_dip = _angle(v2, v3)

    # Abduction: lateral spread in palm plane
    nv1 = float(np.linalg.norm(v1))
    abd = 0.0
    if nv1 > 1e-6:
        v1_dir = v1 / nv1
        abd = float(np.dot(v1_dir, u_across))

    return [round(flex_mcp, 3), round(flex_pip, 3), round(flex_dip, 3)], round(abd, 3)


def calc_thumb_phalanx_angles(
    cmc: np.ndarray,
    mcp: np.ndarray,
    ip: np.ndarray,
    tip: np.ndarray,
    wrist: np.ndarray,
    index_mcp: Optional[np.ndarray] = None,
) -> Tuple[List[float], float]:
    """
    Computes thumb phalanx flexion angles [cmc_flex, mcp_flex, ip_flex] in radians,
    and thumb opposition across palm [0.0..1.0].
    """
    v0 = cmc - wrist
    v1 = mcp - cmc
    v2 = ip - mcp
    v3 = tip - ip

    def _angle(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        dot = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return float(math.acos(dot))

    cmc_flex = _angle(v0, v1)
    mcp_flex = _angle(v1, v2)
    ip_flex = _angle(v2, v3)

    # Opposition: thumb tip distance to index MCP
    opp = 0.0
    if index_mcp is not None:
        thumb_len = float(np.linalg.norm(v1) + np.linalg.norm(v2) + np.linalg.norm(v3))
        d_idx = float(np.linalg.norm(tip - index_mcp))
        opp = float(np.clip(1.0 - (d_idx / (thumb_len * 1.1 + 1e-6)), 0.0, 1.0))

    return [round(cmc_flex, 3), round(mcp_flex, 3), round(ip_flex, 3)], round(opp, 3)


def estimate_fist_curl_from_pose(wrist: List[float], index: List[float], elbow: List[float]) -> float:
    """Estimate hand clench/curl from MediaPipe Pose wrist-to-index distance ratio."""
    d_hand = math.sqrt(sum((index[i] - wrist[i]) ** 2 for i in range(3)))
    d_arm = math.sqrt(sum((wrist[i] - elbow[i]) ** 2 for i in range(3)))
    if d_arm < 1e-4:
        return 0.15
    ratio = d_hand / d_arm
    # Hand open: ratio ~ 0.60 - 0.75; Fist closed: ratio ~ 0.30 - 0.45
    curl = (0.62 - ratio) / (0.62 - 0.38)
    return float(np.clip(curl, 0.0, 1.0))


class MocapTracker:
    """Unified Motion Capture Tracker using MediaPipe Pose & Hands."""

    def __init__(self, track_hands: bool = True, model_complexity: int = 1):
        self.track_hands = track_hands
        self.model_complexity = model_complexity

        # MediaPipe models
        with SilenceStderr():
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.mp_hands = (
                mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=0,
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                if track_hands
                else None
            )

        # Real-time 1€ filter per joint: low min_cutoff stops static arm jitter; beta provides instant strike response
        self.joint_filters: Dict[str, Vector3DOneEuroFilter] = {
            name: Vector3DOneEuroFilter(
                freq=30.0,
                min_cutoff=0.35,
                beta=0.35 if ("wrist" in name or "elbow" in name or "index" in name or "pinky" in name) else 0.15,
            )
            for name in STANDARD_JOINTS
        }

        # Temporal smoothing for fingers
        self.finger_smooth: Dict[str, Dict[str, float]] = {
            "left": {"thumb": 0.15, "index": 0.15, "middle": 0.15, "ring": 0.15, "pinky": 0.15},
            "right": {"thumb": 0.15, "index": 0.15, "middle": 0.15, "ring": 0.15, "pinky": 0.15},
        }

        # Head Metsuke posture calibration & smoothing
        self.calibrated = False
        self.calib_samples: List[Tuple[float, float, float]] = []
        self.base_pitch: Optional[float] = None
        self.base_yaw: float = 0.0
        self.base_roll: float = 0.0
        self.head_pitch_filter = OneEuroFilter(freq=30.0, min_cutoff=0.4, beta=0.15)
        self.head_yaw_filter = OneEuroFilter(freq=30.0, min_cutoff=0.4, beta=0.15)
        self.head_roll_filter = OneEuroFilter(freq=30.0, min_cutoff=0.4, beta=0.15)

    def calibrate(self):
        """Reset head baseline posture calibration and filter states."""
        self.calibrated = False
        self.calib_samples = []
        self.base_pitch = None
        self.base_yaw = 0.0
        self.base_roll = 0.0
        self.head_pitch_filter.reset()
        self.head_yaw_filter.reset()
        self.head_roll_filter.reset()
        for f in self.joint_filters.values():
            f.reset()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp: float,
        mirror: bool = False,
        draw_skeleton: bool = True,
        fps: float = 0.0,
    ) -> Tuple[np.ndarray, FrameData]:
        """
        Processes a single BGR frame.
        Returns:
            processed_frame: BGR frame (mirrored if requested, with anti-aliased OpenCV skeleton overlay)
            frame_data: standard FrameData
        """
        # Always feed original natural frame_bgr to MediaPipe so anatomical Left and Right are NEVER swapped
        h, w = frame_bgr.shape[:2]

        # 1:1 Square padding (letterboxing) to satisfy MediaPipe square ROI
        dim_max = max(h, w)
        pad_top = (dim_max - h) // 2
        pad_bottom = dim_max - h - pad_top
        pad_left = (dim_max - w) // 2
        pad_right = dim_max - w - pad_left

        frame_square = cv2.copyMakeBorder(
            frame_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        rgb_square = cv2.cvtColor(frame_square, cv2.COLOR_BGR2RGB)

        # 1. Process Body Pose and Hands
        with SilenceStderr():
            res_pose = self.mp_pose.process(rgb_square)
            res_hands = self.mp_hands.process(rgb_square) if (self.track_hands and self.mp_hands) else None

        # Display frame: flip horizontally if mirror requested for selfie display
        display_frame = cv2.flip(frame_bgr, 1) if mirror else frame_bgr.copy()

        frame_data = FrameData(
            frame=frame_idx,
            time=round(timestamp, 4),
            rootPosition=[0.0, 0.9, 0.0],
            rootRotation=[0.0, 0.0, 0.0, 1.0],
            joints={},
            confidence={},
            tracking_loss=True,
            head={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            fingers={
                "left": {"thumb": 0.15, "index": 0.15, "middle": 0.15, "ring": 0.15, "pinky": 0.15, "detected": False},
                "right": {"thumb": 0.15, "index": 0.15, "middle": 0.15, "ring": 0.15, "pinky": 0.15, "detected": False},
            },
        )

        raw_pts_2d_unmirrored: Dict[str, List[float]] = {}

        if res_pose and res_pose.pose_world_landmarks and res_pose.pose_landmarks:
            frame_data.tracking_loss = False
            lm_world = res_pose.pose_world_landmarks.landmark

            raw_pts_3d: Dict[str, List[float]] = {}
            vis_dict: Dict[str, float] = {}

            # Three.js coordinates: X right, Y up, Z forward (towards camera)
            for name, idx in POSE_LM.items():
                if idx < len(lm_world):
                    p = lm_world[idx]
                    x = p.x
                    y = -p.y
                    z = -p.z
                    raw_pts_3d[name] = [round(float(x), 4), round(float(y), 4), round(float(z), 4)]
                    vis_dict[name] = round(float(p.visibility), 3)

            # Anatomical Torso Clearance:
            # When arms cross in front of the chest (blocks, strikes, chambers), monocular depth
            # estimation can erroneously predict negative Z values (placing wrists/elbows behind the spine),
            # causing the forearm to pierce backwards through the ribcage and dislocate the shoulder.
            # Anatomical Torso Clearance (3D Line Segment Projection):
            # Enforces that arm joints crossing in front of the chest stay physically in front of the sternum.
            # Uses true 3D projection onto the spine line segment (mid_hip -> mid_shoulder) so it works
            # identically whether standing tall, bending forward, or twisting, and leaves hands free to
            # naturally touch thighs, knees, and shins (t < 0.20).
            if (
                "left_shoulder" in raw_pts_3d
                and "right_shoulder" in raw_pts_3d
                and "left_hip" in raw_pts_3d
                and "right_hip" in raw_pts_3d
            ):
                l_sh = np.array(raw_pts_3d["left_shoulder"])
                r_sh = np.array(raw_pts_3d["right_shoulder"])
                l_hp = np.array(raw_pts_3d["left_hip"])
                r_hp = np.array(raw_pts_3d["right_hip"])

                mid_sh = (l_sh + r_sh) * 0.5
                mid_hp = (l_hp + r_hp) * 0.5
                v_spine = mid_sh - mid_hp
                spine_len_sq = float(np.dot(v_spine, v_spine))

                if spine_len_sq > 1e-4:
                    uy = v_spine / math.sqrt(spine_len_sq)
                    v_lat = l_sh - r_sh
                    ux = v_lat - float(np.dot(v_lat, uy)) * uy
                    ux_norm = float(np.linalg.norm(ux))
                    if ux_norm > 1e-4:
                        ux /= ux_norm
                        uz = np.cross(ux, uy)  # Chest forward normal from spine through sternum
                        ribcage_half_width = float(np.linalg.norm(v_lat)) * 0.55

                        for arm_j in ["left_elbow", "left_wrist", "right_elbow", "right_wrist", "left_index", "right_index"]:
                            if arm_j in raw_pts_3d:
                                pt = np.array(raw_pts_3d[arm_j])
                                v_pt = pt - mid_hp
                                t = float(np.dot(v_pt, v_spine) / spine_len_sq)

                                # Only apply to thoracic spine / ribcage (t in [0.20, 1.08]).
                                # Pelvis, groin, and legs (t < 0.20) are left completely free so hands can touch knees and legs.
                                if 0.20 <= t <= 1.08:
                                    spine_center = mid_hp + t * v_spine
                                    delta = pt - spine_center
                                    dist_lateral = abs(float(np.dot(delta, ux)))

                                    if dist_lateral < ribcage_half_width:
                                        dist_along_normal = float(np.dot(delta, uz))
                                        min_clearance = 0.08  # 8 cm in front of spine centerline
                                        if dist_along_normal < min_clearance:
                                            cleared_pt = pt + (min_clearance - dist_along_normal) * uz
                                            raw_pts_3d[arm_j] = [
                                                round(float(cleared_pt[0]), 4),
                                                round(float(cleared_pt[1]), 4),
                                                round(float(cleared_pt[2]), 4),
                                            ]

            # Real-time 1€ filter: eliminate sensor micro-jitter while preserving sharp strikes
            pts_3d: Dict[str, List[float]] = {}
            for name, p in raw_pts_3d.items():
                if name in self.joint_filters:
                    pts_3d[name] = self.joint_filters[name].filter(p, timestamp)
                else:
                    pts_3d[name] = p

            frame_data.joints = pts_3d
            frame_data.confidence = vis_dict

            # 2D Screen Landmarks (mapped back to unpadded frame coordinates in [0, 1], mirrored if display is mirrored)
            lm_2d = res_pose.pose_landmarks.landmark
            pts_2d: Dict[str, List[float]] = {}
            for name, idx in POSE_LM.items():
                if idx < len(lm_2d):
                    p2 = lm_2d[idx]
                    nx = (p2.x * dim_max - pad_left) / max(1, w)
                    ny = (p2.y * dim_max - pad_top) / max(1, h)
                    raw_pts_2d_unmirrored[name] = [float(nx), float(ny)]
                    if mirror:
                        nx = 1.0 - nx
                    pts_2d[name] = [round(float(np.clip(nx, 0.0, 1.0)), 4), round(float(np.clip(ny, 0.0, 1.0)), 4)]
            frame_data.landmarks_2d = pts_2d

            # Root Position (Hips Center)
            if "left_hip" in pts_3d and "right_hip" in pts_3d:
                l_hip = pts_3d["left_hip"]
                r_hip = pts_3d["right_hip"]
                root_x = (l_hip[0] + r_hip[0]) * 0.5
                root_y = (l_hip[1] + r_hip[1]) * 0.5 + 0.9  # standing offset
                root_z = (l_hip[2] + r_hip[2]) * 0.5
                frame_data.rootPosition = [round(root_x, 4), round(root_y, 4), round(root_z, 4)]

            # Head Rotation (Metsuke gaze with baseline calibration)
            if "nose" in pts_3d and "left_ear" in pts_3d and "right_ear" in pts_3d:
                nose = pts_3d["nose"]
                l_ear = pts_3d["left_ear"]
                r_ear = pts_3d["right_ear"]

                mid_ear = [(l_ear[i] + r_ear[i]) * 0.5 for i in range(3)]
                face_fwd = [nose[i] - mid_ear[i] for i in range(3)]
                ear_vec = [l_ear[i] - r_ear[i] for i in range(3)]

                raw_yaw = float(math.atan2(face_fwd[0], max(0.01, face_fwd[2])))
                raw_pitch = float(math.atan2(face_fwd[1], max(0.01, face_fwd[2])))
                raw_roll = float(math.atan2(ear_vec[1], max(0.05, abs(ear_vec[0]))))

                # Auto-calibrate neutral posture on first valid frames
                if self.base_pitch is None:
                    self.base_pitch = raw_pitch
                    self.base_yaw = raw_yaw
                    self.base_roll = raw_roll

                if not self.calibrated:
                    self.calib_samples.append((raw_yaw, raw_pitch, raw_roll))
                    if len(self.calib_samples) >= 15:
                        self.base_yaw = float(np.mean([s[0] for s in self.calib_samples]))
                        self.base_pitch = float(np.mean([s[1] for s in self.calib_samples]))
                        self.base_roll = float(np.mean([s[2] for s in self.calib_samples]))
                        self.calibrated = True

                yaw = raw_yaw - self.base_yaw
                pitch = raw_pitch - self.base_pitch
                roll = raw_roll - self.base_roll

                # Deadbands: filter out micro-fluctuations when looking steady
                if abs(yaw) < 0.04:
                    yaw = 0.0
                if abs(pitch) < 0.04:
                    pitch = 0.0
                if abs(roll) < 0.05:
                    roll = 0.0

                yaw = float(np.clip(yaw * 0.85, -0.65, 0.65))
                pitch = float(np.clip(pitch * 0.80, -0.45, 0.45))
                roll = float(np.clip(roll * 0.50, -0.30, 0.30))

                # Smooth head angles with OneEuroFilter to eliminate head wobble
                yaw = self.head_yaw_filter.filter(yaw, timestamp)
                pitch = self.head_pitch_filter.filter(pitch, timestamp)
                roll = self.head_roll_filter.filter(roll, timestamp)

                frame_data.head = {
                    "yaw": round(float(yaw), 3),
                    "pitch": round(float(pitch), 3),
                    "roll": round(float(roll), 3),
                }

        # Match detected hands to anatomical sides (left vs right)
        matched_hands = []
        if res_hands and res_hands.multi_hand_landmarks and res_hands.multi_handedness:
            for hand_lms, handedness in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
                label = handedness.classification[0].label
                hw_x = (hand_lms.landmark[0].x * dim_max - pad_left) / max(1, w)
                hw_y = (hand_lms.landmark[0].y * dim_max - pad_top) / max(1, h)

                side = None
                if raw_pts_2d_unmirrored and "left_wrist" in raw_pts_2d_unmirrored and "right_wrist" in raw_pts_2d_unmirrored:
                    lw = raw_pts_2d_unmirrored["left_wrist"]
                    rw = raw_pts_2d_unmirrored["right_wrist"]
                    d_left = (hw_x - lw[0]) ** 2 + (hw_y - lw[1]) ** 2
                    d_right = (hw_x - rw[0]) ** 2 + (hw_y - rw[1]) ** 2
                    side = "left" if d_left < d_right else "right"

                if side is None:
                    # In an unmirrored camera stream, MediaPipe Hands' "Left" is physically person's RIGHT hand
                    side = "right" if label == "Left" else "left"
                matched_hands.append((hand_lms, side))

        # 3. Finger Tracking (Hybrid: MediaPipe Hands if detected, fallback to Pose distance ratio)
        detected_sides = set()
        aspect = w / max(1.0, float(h))
        for hand_lms, side in matched_hands:
            detected_sides.add(side)
            h_pts = [np.array([p.x * aspect, p.y, p.z * aspect]) for p in hand_lms.landmark]

            curl_thumb = calc_thumb_curl(h_pts[1], h_pts[2], h_pts[3], h_pts[4], h_pts[0], h_pts[5])
            curl_index = calc_finger_curl(h_pts[5], h_pts[6], h_pts[7], h_pts[8], h_pts[0])
            curl_middle = calc_finger_curl(h_pts[9], h_pts[10], h_pts[11], h_pts[12], h_pts[0])
            curl_ring = calc_finger_curl(h_pts[13], h_pts[14], h_pts[15], h_pts[16], h_pts[0])
            curl_pinky = calc_finger_curl(h_pts[17], h_pts[18], h_pts[19], h_pts[20], h_pts[0])

            curls = {
                "thumb": curl_thumb,
                "index": curl_index,
                "middle": curl_middle,
                "ring": curl_ring,
                "pinky": curl_pinky,
            }
            for ftype, c_val in curls.items():
                prev = self.finger_smooth[side][ftype]
                diff = abs(c_val - prev)
                # Adaptive smoothing: responsive to fine twitches & fast punch clenches
                alpha = 0.75 if diff > 0.10 else 0.50
                smooth = prev * (1.0 - alpha) + c_val * alpha
                self.finger_smooth[side][ftype] = smooth
                frame_data.fingers[side][ftype] = round(smooth, 2)

            # 3D Palm Orientation (Wrist rotation & palm twist) in Three.js coordinates
            # Points in Three.js coordinates: X right, Y up, Z towards camera
            w_pt = np.array([h_pts[0][0], -h_pts[0][1], -h_pts[0][2]])
            m_pt = np.array([h_pts[9][0], -h_pts[9][1], -h_pts[9][2]])
            i_pt = np.array([h_pts[5][0], -h_pts[5][1], -h_pts[5][2]])
            p_pt = np.array([h_pts[17][0], -h_pts[17][1], -h_pts[17][2]])

            v_fwd = m_pt - w_pt
            len_fwd = float(np.linalg.norm(v_fwd))
            if len_fwd > 1e-4:
                v_fwd /= len_fwd
            else:
                v_fwd = np.array([0.0, 1.0, 0.0])

            # Cross product produces outward palm normal for both left and right hands
            v_across = (p_pt - i_pt) if side == "left" else (i_pt - p_pt)
            len_across = float(np.linalg.norm(v_across))
            if len_across > 1e-4:
                v_across /= len_across
            else:
                v_across = np.array([1.0, 0.0, 0.0])

            v_norm = np.cross(v_across, v_fwd)
            len_norm = float(np.linalg.norm(v_norm))
            if len_norm > 1e-4:
                v_norm /= len_norm
            else:
                v_norm = np.array([0.0, 0.0, 1.0])

            # Detailed individual phalanx angles (MCP, PIP, DIP) and abduction
            u_across_hand = (h_pts[17] - h_pts[5]) if side == "left" else (h_pts[5] - h_pts[17])
            n_ac = float(np.linalg.norm(u_across_hand))
            u_across_hand = (u_across_hand / n_ac) if n_ac > 1e-6 else np.array([1.0, 0.0, 0.0])

            ang_idx, abd_idx = calc_phalanx_angles(h_pts[5], h_pts[6], h_pts[7], h_pts[8], h_pts[0], u_across_hand)
            ang_mid, abd_mid = calc_phalanx_angles(h_pts[9], h_pts[10], h_pts[11], h_pts[12], h_pts[0], u_across_hand)
            ang_rng, abd_rng = calc_phalanx_angles(h_pts[13], h_pts[14], h_pts[15], h_pts[16], h_pts[0], u_across_hand)
            ang_pnk, abd_pnk = calc_phalanx_angles(h_pts[17], h_pts[18], h_pts[19], h_pts[20], h_pts[0], u_across_hand)
            ang_thb, opp_thb = calc_thumb_phalanx_angles(h_pts[1], h_pts[2], h_pts[3], h_pts[4], h_pts[0], h_pts[5])

            frame_data.fingers[side]["phalanges"] = {
                "thumb": ang_thb,
                "index": ang_idx,
                "middle": ang_mid,
                "ring": ang_rng,
                "pinky": ang_pnk,
            }
            frame_data.fingers[side]["abduction"] = {
                "thumb": opp_thb,
                "index": abd_idx,
                "middle": 0.0,
                "ring": abd_rng,
                "pinky": abd_pnk,
            }

            frame_data.fingers[side]["palm_normal"] = [round(float(v), 3) for v in v_norm]
            frame_data.fingers[side]["hand_direction"] = [round(float(v), 3) for v in v_fwd]
            frame_data.fingers[side]["detected"] = True

        # Fallback for hands not detected by MediaPipe Hands (e.g. at full-body distance)
        for side in ["left", "right"]:
            if side not in detected_sides:
                wrist_name = f"{side}_wrist"
                index_name = f"{side}_index"
                elbow_name = f"{side}_elbow"
                pinky_name = f"{side}_pinky"
                thumb_name = f"{side}_thumb"
                if (
                    wrist_name in frame_data.joints
                    and index_name in frame_data.joints
                    and elbow_name in frame_data.joints
                ):
                    vis_w = frame_data.confidence.get(wrist_name, 0.0)
                    vis_i = frame_data.confidence.get(index_name, 0.0)
                    if vis_w >= 0.30 and vis_i >= 0.30:
                        est_curl = estimate_fist_curl_from_pose(
                            frame_data.joints[wrist_name],
                            frame_data.joints[index_name],
                            frame_data.joints[elbow_name],
                        )
                        for ftype in ["thumb", "index", "middle", "ring", "pinky"]:
                            smooth = self.finger_smooth[side][ftype] * 0.70 + est_curl * 0.30
                            self.finger_smooth[side][ftype] = smooth
                            frame_data.fingers[side][ftype] = round(smooth, 2)
                        frame_data.fingers[side]["detected"] = True
                        frame_data.fingers[side]["general_curl"] = round(est_curl, 2)

                        # Palm orientation from Pose landmarks
                        if pinky_name in frame_data.joints:
                            w_p = np.array(frame_data.joints[wrist_name])
                            i_p = np.array(frame_data.joints[index_name])
                            p_p = np.array(frame_data.joints[pinky_name])
                            t_p = np.array(frame_data.joints.get(thumb_name, i_p))

                            v_fwd = i_p - w_p
                            len_fwd = float(np.linalg.norm(v_fwd))
                            if len_fwd > 1e-4:
                                v_fwd /= len_fwd
                            else:
                                v_fwd = np.array([0.0, 1.0, 0.0])

                            v_across = (p_p - t_p) if side == "left" else (t_p - p_p)
                            len_across = float(np.linalg.norm(v_across))
                            if len_across > 1e-4:
                                v_across /= len_across
                            else:
                                v_across = np.array([1.0, 0.0, 0.0])

                            v_norm = np.cross(v_across, v_fwd)
                            len_norm = float(np.linalg.norm(v_norm))
                            if len_norm > 1e-4:
                                v_norm /= len_norm
                            else:
                                v_norm = np.array([0.0, 0.0, 1.0])

                            frame_data.fingers[side]["palm_normal"] = [round(float(v), 3) for v in v_norm]
                            frame_data.fingers[side]["hand_direction"] = [round(float(v), 3) for v in v_fwd]
                    else:
                        for ftype in ["thumb", "index", "middle", "ring", "pinky"]:
                            smooth = self.finger_smooth[side][ftype] * 0.85 + 0.15 * 0.15
                            self.finger_smooth[side][ftype] = smooth
                            frame_data.fingers[side][ftype] = round(smooth, 2)
                else:
                    for ftype in ["thumb", "index", "middle", "ring", "pinky"]:
                        smooth = self.finger_smooth[side][ftype] * 0.85 + 0.15 * 0.15
                        self.finger_smooth[side][ftype] = smooth
                        frame_data.fingers[side][ftype] = round(smooth, 2)

        # 4. Populate 2D Hand Landmark Coordinates
        for hand_lms, side in matched_hands:
            pts_h = []
            for p in hand_lms.landmark:
                nx = (p.x * dim_max - pad_left) / max(1, w)
                ny = (p.y * dim_max - pad_top) / max(1, h)
                if mirror:
                    nx = 1.0 - nx
                pts_h.append([round(float(np.clip(nx, 0.0, 1.0)), 4), round(float(np.clip(ny, 0.0, 1.0)), 4)])
            frame_data.hands_2d[side] = pts_h

        # 5. Native OpenCV Anti-Aliased Skeleton and Hand Overlay
        if draw_skeleton:
            if res_pose.pose_landmarks:
                self._draw_pose_overlay(
                    display_frame,
                    res_pose.pose_landmarks.landmark,
                    dim_max,
                    pad_left,
                    pad_top,
                    mirror,
                    fps=fps,
                )
            for hand_lms, side in matched_hands:
                # Left hand in cyan, Right hand in amber
                color = (0, 255, 200) if side == "left" else (255, 200, 0)
                self._draw_hand_overlay(
                    display_frame,
                    hand_lms,
                    dim_max,
                    pad_left,
                    pad_top,
                    mirror,
                    color,
                )

        return display_frame, frame_data

    def _draw_pose_overlay(
        self,
        frame: np.ndarray,
        lm_2d,
        dim_max: int,
        pad_left: int,
        pad_top: int,
        mirror: bool,
        fps: float = 0.0,
    ):
        """Draws high-contrast anti-aliased karate skeleton directly on frame using OpenCV."""
        h, w = frame.shape[:2]
        coords: Dict[str, Tuple[int, int]] = {}
        for name, idx in POSE_LM.items():
            if idx < len(lm_2d):
                p = lm_2d[idx]
                if getattr(p, "visibility", 1.0) > 0.30:
                    raw_px = p.x * dim_max - pad_left
                    raw_py = p.y * dim_max - pad_top
                    px = int(w - 1 - raw_px) if mirror else int(raw_px)
                    py = int(raw_py)
                    if 0 <= px < w and 0 <= py < h:
                        coords[name] = (px, py)

        # Draw bone connections
        for a, b, color in POSE_DRAW_CONNECTIONS:
            if a in coords and b in coords:
                cv2.line(frame, coords[a], coords[b], color, 3, cv2.LINE_AA)

        # Draw dual-ring high-contrast joint spheres
        for name, pt in coords.items():
            color = (0, 255, 255) if ("wrist" in name or "ankle" in name) else (255, 255, 255)
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)

        # Status HUD
        if fps > 0.0 or coords:
            status_text = f"FPS: {fps:.1f} | Tracking: OK" if coords else "Searching Pose..."
            color_hud = (0, 255, 120) if coords else (0, 140, 255)
            cv2.putText(frame, status_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, status_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_hud, 2, cv2.LINE_AA)

    def _draw_hand_overlay(
        self,
        frame: np.ndarray,
        hand_lms,
        dim_max: int,
        pad_left: int,
        pad_top: int,
        mirror: bool,
        color: Tuple[int, int, int],
    ):
        """Draws all 21 joints and finger bones per hand using OpenCV."""
        h, w = frame.shape[:2]
        pts = []
        for p in hand_lms.landmark:
            raw_px = p.x * dim_max - pad_left
            raw_py = p.y * dim_max - pad_top
            px = int(w - 1 - raw_px) if mirror else int(raw_px)
            py = int(raw_py)
            pts.append((px, py))

        for a, b in HAND_CONNECTIONS:
            if 0 <= a < len(pts) and 0 <= b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for pt in pts:
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def analyze_video_file(
        self,
        video_path: Path | str,
        technique_name: str = "karate_technique",
        stride: int = 1,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> KarateMotionProject:
        """
        Mode A: Analyzes a video file frame-by-frame and produces a KarateMotionProject.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        self.calibrate()
        raw_frames: List[FrameData] = []
        frame_idx = 0
        analyzed_count = 0
        t0 = time.time()

        try:
            while True:
                if cancel_check and cancel_check():
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % stride == 0:
                    simulated_time = frame_idx / fps
                    _, fdata = self.process_frame(frame, analyzed_count, simulated_time, mirror=False)
                    raw_frames.append(fdata)
                    analyzed_count += 1

                    if progress_callback:
                        progress_callback(frame_idx, total_frames, analyzed_count / max(0.01, time.time() - t0))

                frame_idx += 1
        finally:
            cap.release()

        # Build Project
        meta = Metadata(
            name=path.stem,
            technique=technique_name,
            sourceType="video",
            sourceFile=path.name,
            fps=round(fps / stride, 2),
            duration=round(len(raw_frames) / (fps / stride), 4) if raw_frames else 0.0,
            frameCount=len(raw_frames),
            trimRange=[0, max(0, len(raw_frames) - 1)],
        )

        project = KarateMotionProject(
            metadata=meta,
            rawFrames=raw_frames,
            filteredFrames=[],
        )
        return project
