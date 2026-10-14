"""
Project Data Schema and Serialization for Karate Motion Studio.
Versioned JSON format for raw motion captures, filtered trajectories, and retargeted animations.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

SCHEMA_FORMAT = "karate-motion-studio"
SCHEMA_VERSION = 1

# Standard 33 MediaPipe pose joint names
STANDARD_JOINTS = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Standard bone connections [from, to, side]
STANDARD_CONNECTIONS: List[Tuple[str, str, str]] = [
    # Torso
    ("left_shoulder", "right_shoulder", "torso"),
    ("left_shoulder", "left_hip", "torso"),
    ("right_shoulder", "right_hip", "torso"),
    ("left_hip", "right_hip", "torso"),
    # Arms
    ("left_shoulder", "left_elbow", "left"),
    ("left_elbow", "left_wrist", "left"),
    ("right_shoulder", "right_elbow", "right"),
    ("right_elbow", "right_wrist", "right"),
    # Legs & Feet
    ("left_hip", "left_knee", "left"),
    ("left_knee", "left_ankle", "left"),
    ("left_ankle", "left_heel", "left"),
    ("left_heel", "left_foot_index", "left"),
    ("left_ankle", "left_foot_index", "left"),
    ("right_hip", "right_knee", "right"),
    ("right_knee", "right_ankle", "right"),
    ("right_ankle", "right_heel", "right"),
    ("right_heel", "right_foot_index", "right"),
    ("right_ankle", "right_foot_index", "right"),
    # Neck / Collarbones
    ("nose", "left_shoulder", "head"),
    ("nose", "right_shoulder", "head"),
]

SHOTOKAN_TECHNIQUE_PRESETS = [
    "gedan_barai",
    "age_uke",
    "soto_uke",
    "uchi_uke",
    "shuto_uke",
    "oi_zuki",
    "gyaku_zuki",
    "kizami_zuki",
    "uraken_uchi",
    "mae_geri",
    "yoko_geri_kekomi",
    "yoko_geri_keage",
    "mawashi_geri",
    "zenkutsu_dachi",
    "kokutsu_dachi",
    "kiba_dachi",
    "heian_shodan",
    "heian_nidan",
    "heian_sandan",
    "heian_yondan",
    "heian_godan",
    "tekki_shodan",
    "bunkai_drill",
]


@dataclass
class Metadata:
    name: str = "unnamed_motion"
    technique: str = "gedan_barai"
    tags: List[str] = field(default_factory=lambda: ["shotokan", "karate"])
    sourceType: str = "video"  # "video" or "webcam" or "synthetic"
    sourceFile: str = ""
    fps: float = 30.0
    duration: float = 0.0
    frameCount: int = 0
    trimRange: List[int] = field(default_factory=lambda: [0, 0])
    notes: str = ""


@dataclass
class CoordinateSystem:
    handedness: str = "right-handed"
    upAxis: str = "Y"
    forwardAxis: str = "Z"
    units: str = "meters"
    description: str = "Three.js compatible world coordinates: X right, Y up, Z forward"


@dataclass
class SkeletonDefinition:
    joints: List[str] = field(default_factory=lambda: list(STANDARD_JOINTS))
    boneConnections: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(STANDARD_CONNECTIONS)
    )


@dataclass
class FrameData:
    frame: int
    time: float
    rootPosition: List[float] = field(default_factory=lambda: [0.0, 0.9, 0.0])
    rootRotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    joints: Dict[str, List[float]] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    tracking_loss: bool = False
    head: Dict[str, float] = field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    fingers: Dict[str, Any] = field(
        default_factory=lambda: {
            "left": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "detected": False},
            "right": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "detected": False},
        }
    )
    landmarks_2d: Dict[str, List[float]] = field(default_factory=dict)
    hands_2d: Dict[str, List[List[float]]] = field(default_factory=dict)


@dataclass
class RetargetingData:
    targetRig: str = "Karateka_Armature"
    boneMapping: Dict[str, str] = field(default_factory=dict)
    # list of dicts: frame_idx -> {bone_name: [x, y, z, w]}
    localRotations: List[Dict[str, List[float]]] = field(default_factory=list)


@dataclass
class KarateMotionProject:
    format: str = SCHEMA_FORMAT
    version: int = SCHEMA_VERSION
    metadata: Metadata = field(default_factory=Metadata)
    coordinateSystem: CoordinateSystem = field(default_factory=CoordinateSystem)
    skeleton: SkeletonDefinition = field(default_factory=SkeletonDefinition)
    rawFrames: List[FrameData] = field(default_factory=list)
    filteredFrames: List[FrameData] = field(default_factory=list)
    retargeting: Optional[RetargetingData] = None

    def get_effective_frames(self) -> List[FrameData]:
        """Return filtered frames if available, otherwise raw frames."""
        return self.filteredFrames if self.filteredFrames else self.rawFrames

    def get_frame(self, index: int) -> Optional[FrameData]:
        frames = self.get_effective_frames()
        if 0 <= index < len(frames):
            return frames[index]
        return None

    def update_metadata(self):
        frames = self.get_effective_frames()
        self.metadata.frameCount = len(frames)
        if frames:
            self.metadata.duration = round(frames[-1].time - frames[0].time, 4)
            if self.metadata.trimRange == [0, 0] or self.metadata.trimRange[1] >= len(frames):
                self.metadata.trimRange = [0, max(0, len(frames) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "metadata": asdict(self.metadata),
            "coordinateSystem": asdict(self.coordinateSystem),
            "skeleton": asdict(self.skeleton),
            "rawFrames": [asdict(f) for f in self.rawFrames],
            "filteredFrames": [asdict(f) for f in self.filteredFrames],
            "retargeting": asdict(self.retargeting) if self.retargeting else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KarateMotionProject:
        fmt = data.get("format")
        if fmt != SCHEMA_FORMAT:
            raise ValueError(f"Invalid format: expected '{SCHEMA_FORMAT}', got '{fmt}'")

        ver = data.get("version", 1)
        if ver > SCHEMA_VERSION:
            raise ValueError(f"Unsupported future version {ver} (maximum supported is {SCHEMA_VERSION})")

        meta = Metadata(**data.get("metadata", {}))
        coords = CoordinateSystem(**data.get("coordinateSystem", {}))
        skel = SkeletonDefinition(**data.get("skeleton", {}))

        raw_frames = [FrameData(**f) for f in data.get("rawFrames", [])]
        filt_frames = [FrameData(**f) for f in data.get("filteredFrames", [])]

        retarget_dict = data.get("retargeting")
        retarget = RetargetingData(**retarget_dict) if retarget_dict else None

        proj = cls(
            format=fmt,
            version=ver,
            metadata=meta,
            coordinateSystem=coords,
            skeleton=skel,
            rawFrames=raw_frames,
            filteredFrames=filt_frames,
            retargeting=retarget,
        )
        proj.update_metadata()
        return proj

    def save_json(self, path: Path | str) -> str:
        self.update_metadata()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)

    @classmethod
    def load_json(cls, path: Path | str) -> KarateMotionProject:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Project file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def export_capture_json(self) -> Dict[str, Any]:
        """Export clean lightweight Capture JSON with landmark positions."""
        self.update_metadata()
        frames = self.get_effective_frames()
        start, end = self.metadata.trimRange if self.metadata.trimRange else (0, len(frames) - 1)
        trimmed = frames[start : end + 1] if frames else []
        t0 = trimmed[0].time if trimmed else 0.0

        return {
            "format": "bunkai-capture-json",
            "version": 1,
            "technique": self.metadata.technique,
            "name": self.metadata.name,
            "tags": self.metadata.tags,
            "fps": self.metadata.fps,
            "duration": round(trimmed[-1].time - t0, 4) if trimmed else 0.0,
            "frameCount": len(trimmed),
            "coordinateSystem": asdict(self.coordinateSystem),
            "frames": [
                {
                    "frame": i,
                    "time": round(f.time - t0, 4),
                    "rootPosition": f.rootPosition,
                    "joints": f.joints,
                    "confidence": f.confidence,
                    "tracking_loss": f.tracking_loss,
                    "head": f.head,
                    "fingers": f.fingers,
                }
                for i, f in enumerate(trimmed)
            ],
        }

    def export_retargeted_json(self) -> Dict[str, Any]:
        """Export clean Retargeted Animation JSON ready for Three.js."""
        self.update_metadata()
        frames = self.get_effective_frames()
        start, end = self.metadata.trimRange if self.metadata.trimRange else (0, len(frames) - 1)
        trimmed = frames[start : end + 1] if frames else []
        t0 = trimmed[0].time if trimmed else 0.0

        rotations = []
        if self.retargeting and self.retargeting.localRotations:
            rotations = self.retargeting.localRotations[start : end + 1]

        return {
            "format": "bunkai-retargeted-animation",
            "version": 1,
            "technique": self.metadata.technique,
            "name": self.metadata.name,
            "tags": self.metadata.tags,
            "targetRig": self.retargeting.targetRig if self.retargeting else "Karateka_Armature",
            "fps": self.metadata.fps,
            "duration": round(trimmed[-1].time - t0, 4) if trimmed else 0.0,
            "frameCount": len(trimmed),
            "frames": [
                {
                    "frame": i,
                    "time": round(f.time - t0, 4),
                    "rootPosition": f.rootPosition,
                    "rootRotation": f.rootRotation,
                    "bones": rotations[i] if i < len(rotations) else {},
                    "fingers": f.fingers,
                }
                for i, f in enumerate(trimmed)
            ],
        }
