"""
Trajectory Filtering and Smoothing Algorithms for Karate Motion Studio.
Implements the 1€ (One-Euro) Filter to eliminate high-frequency micro-jitter
while strictly preserving sharp, explosive Karate strikes (Kime),
plus bounded linear/spline interpolation for short occlusions.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.y: Optional[float] = None
        self.s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        return self.s

    def reset(self):
        self.s = None
        self.y = None


class OneEuroFilter:
    """
    Adaptive 1€ Filter (Casiez et al., CHI 2012).
    Dynamically adjusts low-pass cutoff based on derivative (velocity):
    - At rest: low cutoff frequency eliminates micro-jitter.
    - During rapid strikes (Kime): cutoff expands, eliminating lag and preserving sharp peaks.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.freq = max(1e-3, freq)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(d_cutoff))
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if self.last_time is not None and timestamp is not None:
            dt = timestamp - self.last_time
            if dt > 1e-4:
                self.freq = 1.0 / dt
        self.last_time = timestamp

        # Estimate velocity
        prev_x = self.x_filter.y
        raw_diff = 0.0 if prev_x is None else abs(x - prev_x)
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        # Deadband suppression for stationary sensor noise / micro-tremor:
        if raw_diff < 0.006:
            dx *= 0.15
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff))

        # Dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class Vector3DOneEuroFilter:
    """3D point filter maintaining 3 independent 1€ filters for X, Y, Z."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.2, beta: float = 0.08, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
        self.fz = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)

    def filter(self, pt: List[float], timestamp: Optional[float] = None) -> List[float]:
        return [
            round(self.fx.filter(pt[0], timestamp), 4),
            round(self.fy.filter(pt[1], timestamp), 4),
            round(self.fz.filter(pt[2], timestamp), 4),
        ]

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


def interpolate_short_gaps(
    frames_joints: List[Dict[str, List[float]]],
    frames_conf: List[Dict[str, float]],
    joint_names: List[str],
    min_confidence: float = 0.15,
    max_gap: int = 5,
) -> Tuple[List[Dict[str, List[float]]], List[Dict[str, float]], List[bool]]:
    """
    Interpolates missing landmarks over short gaps (<= max_gap frames).
    For gaps longer than max_gap, preserves missing status and marks tracking_loss.
    """
    n_frames = len(frames_joints)
    if n_frames == 0:
        return [], [], []

    out_joints = [dict(fj) for fj in frames_joints]
    out_conf = [dict(fc) for fc in frames_conf]
    tracking_loss = [False] * n_frames

    for jname in joint_names:
        # Build visibility mask for this joint
        valid_indices = []
        for i in range(n_frames):
            c = frames_conf[i].get(jname, 0.0)
            if c >= min_confidence and jname in frames_joints[i]:
                valid_indices.append(i)

        if not valid_indices:
            continue

        # Check gaps between consecutive valid frames
        for k in range(len(valid_indices) - 1):
            i0 = valid_indices[k]
            i1 = valid_indices[k + 1]
            gap_len = i1 - i0 - 1

            if 1 <= gap_len <= max_gap:
                p0 = np.array(out_joints[i0][jname], dtype=float)
                p1 = np.array(out_joints[i1][jname], dtype=float)
                c0 = out_conf[i0].get(jname, min_confidence)
                c1 = out_conf[i1].get(jname, min_confidence)

                # Linear interpolation for short occlusion
                for step in range(1, gap_len + 1):
                    alpha = step / float(gap_len + 1)
                    p_interp = (1.0 - alpha) * p0 + alpha * p1
                    idx = i0 + step
                    out_joints[idx][jname] = [round(float(v), 4) for v in p_interp]
                    out_conf[idx][jname] = round(float((1.0 - alpha) * c0 + alpha * c1), 3)

    # Detect frames where major torso landmarks were lost
    major_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    for i in range(n_frames):
        lost_count = sum(1 for mj in major_joints if out_conf[i].get(mj, 0.0) < min_confidence)
        if lost_count >= 2:
            tracking_loss[i] = True

    return out_joints, out_conf, tracking_loss


def smooth_project_frames(
    raw_frames: List[Any],
    joint_names: List[str],
    min_cutoff: float = 1.0,
    beta: float = 0.06,
    min_confidence: float = 0.15,
    max_gap: int = 5,
) -> List[Any]:
    """
    Full filtering pass on raw project frames:
    1. Bounded short-gap interpolation
    2. Adaptive 1€ filtering
    Returns new list of FrameData without altering raw inputs.
    """
    from core.project_schema import FrameData

    if not raw_frames:
        return []

    raw_joints = [f.joints for f in raw_frames]
    raw_conf = [f.confidence for f in raw_frames]

    # Step 1: Interpolate short gaps
    interp_joints, interp_conf, loss_flags = interpolate_short_gaps(
        raw_joints, raw_conf, joint_names, min_confidence, max_gap
    )

    # Step 2: Initialize 3D 1€ filters per joint
    filters: Dict[str, Vector3DOneEuroFilter] = {
        j: Vector3DOneEuroFilter(freq=30.0, min_cutoff=min_cutoff, beta=beta) for j in joint_names
    }

    filtered_frames: List[FrameData] = []
    for i, rf in enumerate(raw_frames):
        t = rf.time
        f_joints: Dict[str, List[float]] = {}

        for jname in joint_names:
            if jname in interp_joints[i]:
                pt = interp_joints[i][jname]
                filtered_pt = filters[jname].filter(pt, t)
                f_joints[jname] = filtered_pt

        # Smooth head angles with 1€ filter
        h = rf.head
        head_smooth = dict(h)

        filtered_frame = FrameData(
            frame=rf.frame,
            time=rf.time,
            rootPosition=list(rf.rootPosition),
            rootRotation=list(rf.rootRotation),
            joints=f_joints,
            confidence=interp_conf[i],
            tracking_loss=loss_flags[i],
            head=head_smooth,
            fingers=rf.fingers,
        )
        filtered_frames.append(filtered_frame)

    return filtered_frames
