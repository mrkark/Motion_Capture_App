"""
Blender GLB Clip Exporter for Karate Motion Studio.
Executes headless Blender to retarget captured motion onto Karateka_Armature
and export a production-ready .glb clip with AnimationMixer tracks.
"""
from __future__ import annotations
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any


def export_glb_via_blender(
    blender_exe: str,
    project_json_path: Path | str,
    output_glb_path: Path | str,
    action_name: str = "karate_technique",
    avatar_blend_path: Optional[Path | str] = None,
) -> bool:
    """
    Executes headless Blender to bake motion onto Karateka_Armature and export GLB.
    """
    b_path = Path(blender_exe)
    if not b_path.is_file():
        raise FileNotFoundError(f"Blender executable not found at: {blender_exe}")

    base_dir = Path(__file__).resolve().parent.parent.parent / "bunkai_explorer_karate_motion_capture"
    retarget_script = base_dir / "blender_retarget.py"
    if not retarget_script.is_file():
        raise FileNotFoundError(f"blender_retarget.py not found at: {retarget_script}")

    blend_file = (
        Path(avatar_blend_path)
        if avatar_blend_path
        else base_dir / "test_data" / "karateka_lowpoly.blend"
    )

    cmd = [
        str(b_path),
        str(blend_file),
        "--background",
        "--python",
        str(retarget_script),
        "--",
        "--pose",
        str(Path(project_json_path).resolve()),
        "--export-glb",
        str(Path(output_glb_path).resolve()),
        "--action-name",
        action_name,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Blender retargeting failed with exit code {res.returncode}:\n{res.stderr}")

    return Path(output_glb_path).is_file()
