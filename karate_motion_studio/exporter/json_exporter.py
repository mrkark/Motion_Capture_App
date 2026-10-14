"""
JSON Exporter for Bunkai Explorer.
Exports both Capture JSON (joint trajectories) and Retargeted Animation JSON (bone rotations).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from core.project_schema import KarateMotionProject


def export_capture_file(project: KarateMotionProject, output_path: Path | str) -> str:
    """Exports clean Capture JSON to disk."""
    data = project.export_capture_json()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


def export_retargeted_file(project: KarateMotionProject, output_path: Path | str) -> str:
    """Exports clean Retargeted Animation JSON to disk ready for Three.js."""
    data = project.export_retargeted_json()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)
