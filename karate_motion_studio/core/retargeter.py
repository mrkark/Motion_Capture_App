"""
Hierarchical Motion Retargeting Engine for Karate Motion Studio.
Converts 3D joint landmark trajectories into local bone quaternion rotations [x, y, z, w]
taking into account the rig's parent hierarchy, rest-pose geometric orientations, and root motion.
Default target: Karateka_Armature (compatible with Bunkai Explorer and Three.js).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Canonical rest directions in glTF world space for Karateka_Armature:
# In glTF, bones extend along their rest direction.
DEFAULT_REST_WORLD_DIRS: Dict[str, List[float]] = {
    "Spine": [0.0, 1.0, 0.0],
    "Spine1": [0.0, 1.0, 0.0],
    "Chest": [0.0, 1.0, 0.0],
    "Neck": [0.0, 1.0, 0.0],
    "Head": [0.0, 1.0, 0.0],
    "UpperArm.L": [1.0, 0.0, 0.0],   # Extends along +X (avatar left / viewer right)
    "LowerArm.L": [1.0, 0.0, 0.0],
    "Hand.L": [1.0, 0.0, 0.0],
    "UpperArm.R": [-1.0, 0.0, 0.0],  # Extends along -X (avatar right / viewer left)
    "LowerArm.R": [-1.0, 0.0, 0.0],
    "Hand.R": [-1.0, 0.0, 0.0],
    "UpperLeg.L": [0.0, -1.0, 0.0],  # Extends along -Y (downwards)
    "LowerLeg.L": [0.0, -1.0, 0.0],
    "Foot.L": [0.0, -0.6, 0.8],       # Extends forward and slightly down to toes
    "UpperLeg.R": [0.0, -1.0, 0.0],
    "LowerLeg.R": [0.0, -1.0, 0.0],
    "Foot.R": [0.0, -0.6, 0.8],
}

# Hierarchy definition: child -> parent
BONE_PARENT_MAP: Dict[str, str] = {
    "Hips": "",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Chest": "Spine1",
    "Neck": "Chest",
    "Head": "Neck",
    "Shoulder.L": "Chest",
    "UpperArm.L": "Shoulder.L",
    "LowerArm.L": "UpperArm.L",
    "Hand.L": "LowerArm.L",
    "Shoulder.R": "Chest",
    "UpperArm.R": "Shoulder.R",
    "LowerArm.R": "UpperArm.R",
    "Hand.R": "LowerArm.R",
    "UpperLeg.L": "Hips",
    "LowerLeg.L": "UpperLeg.L",
    "Foot.L": "LowerLeg.L",
    "UpperLeg.R": "Hips",
    "LowerLeg.R": "UpperLeg.R",
    "Foot.R": "LowerLeg.R",
}

# Joint mapping: bone -> (start_joint, end_joint)
DEFAULT_BONE_JOINT_MAP: Dict[str, Tuple[str, str]] = {
    "Spine": ("mid_hip", "mid_shoulder"),
    "UpperArm.L": ("left_shoulder", "left_elbow"),
    "LowerArm.L": ("left_elbow", "left_wrist"),
    "UpperArm.R": ("right_shoulder", "right_elbow"),
    "LowerArm.R": ("right_elbow", "right_wrist"),
    "UpperLeg.L": ("left_hip", "left_knee"),
    "LowerLeg.L": ("left_knee", "left_ankle"),
    "Foot.L": ("left_ankle", "left_foot_index"),
    "UpperLeg.R": ("right_hip", "right_knee"),
    "LowerLeg.R": ("right_knee", "right_ankle"),
    "Foot.R": ("right_ankle", "right_foot_index"),
}


def quat_from_unit_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Quaternion rotating unit vector v_from to unit vector v_to: [x, y, z, w]."""
    v1 = v_from / np.linalg.norm(v_from)
    v2 = v_to / np.linalg.norm(v_to)
    dot = float(np.dot(v1, v2))

    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.92:
        # Singularity protection: prevent antiparallel gimbal flips when limbs cross the torso
        v2 = np.array([v2[0], v2[1], max(float(v2[2]), 0.35)])
        v2 /= np.linalg.norm(v2)
        dot = float(np.dot(v1, v2))

    cross = np.cross(v1, v2)
    w = 1.0 + dot
    q = np.array([cross[0], cross[1], cross[2], w], dtype=float)
    return q / np.linalg.norm(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q1 * q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (conjugate for unit quaternion)."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float) / np.linalg.norm(q)


def quat_from_euler_yxz(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Euler YXZ to quaternion [x, y, z, w]."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qy = np.array([0.0, sy, 0.0, cy])
    qx = np.array([sp, 0.0, 0.0, cp])
    qz = np.array([0.0, 0.0, sr, cr])

    return quat_multiply(quat_multiply(qy, qx), qz)


@dataclass
class DiagnosticReport:
    total_frames: int = 0
    retargeted_frames: int = 0
    mapped_bones: List[str] = field(default_factory=list)
    unmapped_bones: List[str] = field(default_factory=list)
    tracking_loss_frames: int = 0
    warnings: List[str] = field(default_factory=list)


class Retargeter:
    """Solves hierarchical local bone quaternions for Karateka_Armature."""

    def __init__(self, target_rig: str = "Karateka_Armature"):
        self.target_rig = target_rig
        self.rest_world_dirs = dict(DEFAULT_REST_WORLD_DIRS)
        self.bone_parent_map = dict(BONE_PARENT_MAP)
        self.bone_joint_map = dict(DEFAULT_BONE_JOINT_MAP)

    def retarget_frame(self, frame_data: Any) -> Dict[str, List[float]]:
        """
        Retargets a single FrameData to local bone quaternions [x, y, z, w].
        """
        joints = frame_data.joints
        if not joints:
            return {}

        # Compute synthetic helper points
        pts = dict(joints)
        if "left_hip" in pts and "right_hip" in pts:
            pts["mid_hip"] = [
                (pts["left_hip"][i] + pts["right_hip"][i]) * 0.5 for i in range(3)
            ]
        if "left_shoulder" in pts and "right_shoulder" in pts:
            pts["mid_shoulder"] = [
                (pts["left_shoulder"][i] + pts["right_shoulder"][i]) * 0.5 for i in range(3)
            ]

        # Dict of world quaternions: bone -> q_world
        world_quats: Dict[str, np.ndarray] = {
            "Hips": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        local_quats: Dict[str, List[float]] = {}

        # 1. Torso / Spine
        if "mid_hip" in pts and "mid_shoulder" in pts:
            v_torso = np.array(pts["mid_shoulder"]) - np.array(pts["mid_hip"])
            if np.linalg.norm(v_torso) > 1e-4:
                # Only damp small perspective slouch when standing tall; preserve full forward bending
                is_upright = v_torso[1] > 0.45 and abs(v_torso[2]) < 0.08
                if is_upright:
                    v_torso[2] *= (abs(v_torso[2]) / 0.08) ** 2
                    if abs(v_torso[2]) < 0.02:
                        v_torso[2] = 0.0
                q_spine_world = quat_from_unit_vectors(np.array([0.0, 1.0, 0.0]), v_torso)
                world_quats["Spine"] = q_spine_world
                local_quats["Spine"] = [round(float(v), 4) for v in q_spine_world]
                world_quats["Chest"] = q_spine_world

        # 2. Head (Euler from Metsuke calculation)
        if hasattr(frame_data, "head") and frame_data.head:
            h = frame_data.head
            # -pitch correctly tilts head up/down in Three.js coordinates
            q_head_local = quat_from_euler_yxz(h.get("yaw", 0.0), -h.get("pitch", 0.0), h.get("roll", 0.0))
            local_quats["Head"] = [round(float(v), 4) for v in q_head_local]

        # 3. Limbs (Arms and Legs)
        limb_order = [
            ("UpperArm.L", "Shoulder.L"),
            ("LowerArm.L", "UpperArm.L"),
            ("UpperArm.R", "Shoulder.R"),
            ("LowerArm.R", "UpperArm.R"),
            ("UpperLeg.L", "Hips"),
            ("LowerLeg.L", "UpperLeg.L"),
            ("Foot.L", "LowerLeg.L"),
            ("UpperLeg.R", "Hips"),
            ("LowerLeg.R", "UpperLeg.R"),
            ("Foot.R", "LowerLeg.R"),
        ]

        for bone_name, parent_name in limb_order:
            if bone_name not in self.bone_joint_map or bone_name not in self.rest_world_dirs:
                continue

            j_start, j_end = self.bone_joint_map[bone_name]
            if bone_name in ["Foot.L", "Foot.R"]:
                heel_key = "left_heel" if bone_name == "Foot.L" else "right_heel"
                if heel_key in pts:
                    j_start = heel_key
            if j_start not in pts or j_end not in pts:
                continue

            v_bone = np.array(pts[j_end]) - np.array(pts[j_start])
            if np.linalg.norm(v_bone) < 1e-4:
                continue

            v_rest = np.array(self.rest_world_dirs[bone_name])
            q_bone_world = quat_from_unit_vectors(v_rest, v_bone)
            world_quats[bone_name] = q_bone_world

            # Convert to local rotation relative to parent:
            # Q_local = Q_parent_world^-1 * Q_target_world
            q_parent_world = world_quats.get(parent_name, world_quats.get("Chest", np.array([0.0, 0.0, 0.0, 1.0])))
            q_local = quat_multiply(quat_inverse(q_parent_world), q_bone_world)
            q_local /= np.linalg.norm(q_local)

            local_quats[bone_name] = [round(float(v), 4) for v in q_local]

        # 4. Finger Curls (Thumb, Index, Middle, Ring, Pinky 1..3)
        if hasattr(frame_data, "fingers") and frame_data.fingers:
            for side in ["left", "right"]:
                s_key = "L" if side == "left" else "R"
                f_data = frame_data.fingers.get(side, {})
                for ftype in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                    curl = f_data.get(ftype.lower(), 0.0) if f_data.get("detected") else 0.85
                    for idx in range(1, 4):
                        bname = f"{ftype}{idx}.{s_key}"
                        # Rotate around local +X
                        max_ang = 0.8 if ftype == "Thumb" else (1.4 if idx == 1 else 1.2)
                        ang = curl * max_ang
                        # Rotation around +X: [sin(ang/2), 0, 0, cos(ang/2)]
                        q_flex = np.array([math.sin(ang * 0.5), 0.0, 0.0, math.cos(ang * 0.5)])
                        local_quats[bname] = [round(float(v), 4) for v in q_flex]

        return local_quats

    def retarget_project(self, project: Any) -> DiagnosticReport:
        """
        Retargets all frames in a KarateMotionProject, storing results in project.retargeting.
        """
        from core.project_schema import RetargetingData

        frames = project.get_effective_frames()
        report = DiagnosticReport(
            total_frames=len(frames),
            mapped_bones=list(self.bone_joint_map.keys()) + ["Head", "Spine"],
        )

        all_local_rotations: List[Dict[str, List[float]]] = []
        for f in frames:
            if f.tracking_loss:
                report.tracking_loss_frames += 1
            local_rot = self.retarget_frame(f)
            all_local_rotations.append(local_rot)

        project.retargeting = RetargetingData(
            targetRig=self.target_rig,
            boneMapping={b: f"{j[0]}_to_{j[1]}" for b, j in self.bone_joint_map.items()},
            localRotations=all_local_rotations,
        )

        report.retargeted_frames = len(all_local_rotations)
        return report
