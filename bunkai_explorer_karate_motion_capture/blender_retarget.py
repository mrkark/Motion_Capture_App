"""
Blender retargeting for Bunkai Explorer.
Transfers 3D joint landmarks from capture_pose.py JSON to Karateka_Armature (or custom rig).

Usage in Blender GUI:
1. Open avatar .blend (e.g., test_data/karateka_lowpoly.blend).
2. Open this script in Scripting workspace.
3. Adjust POSE_JSON and OUTPUT_BLEND (defaults resolve to test_data/ if present).
4. Run Script.

Usage from CLI (headless):
blender -b avatar.blend -P blender_retarget.py -- --pose pose.json --output out.blend --export-glb out.glb
"""
import sys
import os
import json
import argparse
from pathlib import Path
import bpy
from mathutils import Vector, Euler, Quaternion

# Default paths (auto-resolved relative to script directory)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POSE_JSON = SCRIPT_DIR / "test_data" / "karate_test_pose.json"
DEFAULT_OUTPUT_BLEND = SCRIPT_DIR / "test_output" / "avatar_with_capture.blend"
DEFAULT_OUTPUT_GLB = SCRIPT_DIR / "test_output" / "karateka_retargeted.glb"

POSE_JSON = str(DEFAULT_POSE_JSON)
OUTPUT_BLEND = str(DEFAULT_OUTPUT_BLEND)
ACTION_NAME = "captured_karate_technique"
VISIBILITY_THRESHOLD = 0.35
SCALE = 1.0

# Supported limb segments with primary bone name and possible aliases
# Format: (landmark_start, landmark_end, [candidate_bone_names])
BONE_CANDIDATES = [
    # Arms
    ("left_shoulder", "left_elbow", ["UpperArm.L", "Upper_Arm.L", "shoulder.L"]),
    ("left_elbow", "left_wrist", ["LowerArm.L", "Forearm.L", "Lower_Arm.L"]),
    ("right_shoulder", "right_elbow", ["UpperArm.R", "Upper_Arm.R", "shoulder.R"]),
    ("right_elbow", "right_wrist", ["LowerArm.R", "Forearm.R", "Lower_Arm.R"]),
    # Legs
    ("left_hip", "left_knee", ["UpperLeg.L", "Thigh.L", "Upper_Leg.L"]),
    ("left_knee", "left_ankle", ["LowerLeg.L", "Shin.L", "Lower_Leg.L"]),
    ("right_hip", "right_knee", ["UpperLeg.R", "Thigh.R", "Upper_Leg.R"]),
    ("right_knee", "right_ankle", ["LowerLeg.R", "Shin.R", "Lower_Leg.R"]),
    # Feet
    ("left_ankle", "left_foot_index", ["Foot.L", "foot.L"]),
    ("right_ankle", "right_foot_index", ["Foot.R", "foot.R"]),
]

def parse_cli_args():
    """Parse extra arguments passed after '--' in Blender command line."""
    if "--" not in sys.argv:
        return None
    idx = sys.argv.index("--")
    parser = argparse.ArgumentParser(description="Blender Retarget CLI")
    parser.add_argument("--pose", "-p", default=POSE_JSON, help="Path to pose JSON")
    parser.add_argument("--output", "-o", default=OUTPUT_BLEND, help="Path to output .blend")
    parser.add_argument("--export-glb", "-g", default=None, help="Path to export GLB")
    parser.add_argument("--action-name", "-a", default=ACTION_NAME, help="Action name")
    parser.add_argument("--armature", default=None, help="Armature object name")
    parser.add_argument("--fists", action="store_true", default=True, help="Clench karate fists")
    parser.add_argument("--vis-thresh", type=float, default=VISIBILITY_THRESHOLD, help="Visibility threshold")
    try:
        return parser.parse_args(sys.argv[idx + 1:])
    except Exception as e:
        print(f"Warning: error parsing CLI args: {e}")
        return None

def cvt(p):
    """Convert MediaPipe coords (x right, y up, z depth) to Blender coords (x right, y depth, z up)."""
    x, y, z = p
    return Vector((x, -z, y)) * SCALE

def resolve_bone_map(arm):
    """Match candidate bone names against actual armature bones."""
    resolved = []
    bone_set = set(arm.data.bones.keys())
    for a, b, candidates in BONE_CANDIDATES:
        matched = None
        for cand in candidates:
            if cand in bone_set:
                matched = cand
                break
        if matched:
            resolved.append((a, b, matched))
    return resolved

def clench_fists(arm, frame):
    """Curl fingers into authentic karate fists."""
    pb = arm.pose.bones
    for side in ("L", "R"):
        for fname in ("Index", "Middle", "Ring", "Pinky"):
            for idx, ang in ((1, 80), (2, 100), (3, 70)):
                bone = pb.get(f"{fname}{idx}.{side}")
                if bone:
                    bone.rotation_mode = "XYZ"
                    bone.rotation_euler = Euler((ang * 3.14159265 / 180.0, 0, 0), "XYZ")
                    bone.keyframe_insert(data_path="rotation_euler", frame=frame)
        for t_idx, rot in ((1, (45, -25 if side == 'R' else 25, -30 if side == 'R' else 30)),
                           (2, (70, 0, 0)),
                           (3, (60, 0, 0))):
            t_bone = pb.get(f"Thumb{t_idx}.{side}")
            if t_bone:
                t_bone.rotation_mode = "XYZ"
                t_bone.rotation_euler = Euler([a * 3.14159265 / 180.0 for a in rot], "XYZ")
                t_bone.keyframe_insert(data_path="rotation_euler", frame=frame)

def retarget(pose_json_path, output_blend_path, export_glb_path=None, action_name=ACTION_NAME,
             armature_name=None, clench=True, vis_thresh=VISIBILITY_THRESHOLD):
    pose_file = Path(pose_json_path)
    if not pose_file.is_file():
        raise FileNotFoundError(f"Pose JSON not found: {pose_file}")

    data = json.loads(pose_file.read_text(encoding="utf-8"))

    # Locate armature
    arm = None
    if armature_name:
        arm = bpy.data.objects.get(armature_name)
    if not arm or arm.type != "ARMATURE":
        arm = bpy.context.object
        if not arm or arm.type != "ARMATURE":
            arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
            if not arms:
                raise RuntimeError("No armature object found in scene.")
            arm = arms[0]

    print(f"Target Armature: '{arm.name}'")

    # Critical fix: ensure armature is visible so Pose Mode can be activated
    arm.hide_viewport = False
    arm.hide_set(False)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)

    if arm.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")

    # Temporarily mute LIMIT_ROTATION constraints to allow pure motion solve
    for pb in arm.pose.bones:
        for c in pb.constraints:
            if c.type == "LIMIT_ROTATION":
                c.mute = True

    # Reset bones to rest pose
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    bone_map = resolve_bone_map(arm)
    print(f"Resolved bone mappings ({len(bone_map)} bones):")
    for a, b, bn in bone_map:
        print(f"  {a} -> {b} : {bn}")

    # Set up Action
    if arm.animation_data is None:
        arm.animation_data_create()

    old_action = bpy.data.actions.get(action_name)
    if old_action:
        bpy.data.actions.remove(old_action)
    action = bpy.data.actions.new(action_name)
    action.use_fake_user = True
    arm.animation_data.action = action

    scene = bpy.context.scene
    fps = float(data.get("fpsAnalyzed", 30))
    scene.render.fps = max(1, int(round(fps)))

    frames = data.get("frames", [])
    applied_bones = set()

    for row in frames:
        if not row.get("detected"):
            continue
        frame = round(float(row.get("time", 0)) * scene.render.fps) + 1
        scene.frame_set(frame)
        lm = row.get("landmarks", {})
        vis = row.get("visibility", {})

        # Solve limbs in hierarchical order (parents first)
        for a, b, bn in bone_map:
            pb = arm.pose.bones.get(bn)
            if not pb or a not in lm or b not in lm:
                continue
            if min(vis.get(a, 0), vis.get(b, 0)) < vis_thresh:
                continue

            dir_world = cvt(lm[b]) - cvt(lm[a])
            if dir_world.length < 1e-5:
                continue
            dir_world.normalize()

            # Hierarchical pose-space retargeting:
            # Reset this bone to identity rest relative to parent so current matrix reflects parent pose
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            bpy.context.view_layer.update()

            cur_rot = pb.matrix.to_3x3().normalized()
            # Direction in bone local space:
            dir_local = (cur_rot.inverted() @ dir_world).normalized()

            # In Blender bone local space, bone points along +Y
            quat = Vector((0.0, 1.0, 0.0)).rotation_difference(dir_local)
            pb.rotation_quaternion = quat
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bn)
            applied_bones.add(bn)

        if clench:
            clench_fists(arm, frame)

    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_start = 1
    scene.frame_end = max(1, round(float(data.get("durationSeconds", 1)) * scene.render.fps) + 1)

    # Save output .blend
    out_blend = Path(output_blend_path)
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"Saved .blend: {out_blend} (Action: '{action_name}', keyframed bones: {sorted(applied_bones)})")

    # Optional GLB export
    if export_glb_path:
        out_glb = Path(export_glb_path)
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(
            filepath=str(out_glb),
            export_format='GLB',
            export_animations=True,
            export_animation_mode='ACTIONS',
            export_optimize_animation_size=True,
            export_anim_slide_to_zero=False
        )
        print(f"Exported GLB: {out_glb} ({out_glb.stat().st_size} bytes)")

def main():
    cli = parse_cli_args()
    if cli:
        retarget(
            pose_json_path=cli.pose,
            output_blend_path=cli.output,
            export_glb_path=cli.export_glb,
            action_name=cli.action_name,
            armature_name=cli.armature,
            clench=cli.fists,
            vis_thresh=cli.vis_thresh
        )
    else:
        # Running inside Blender GUI or without CLI flags
        pose_p = Path(POSE_JSON)
        if not pose_p.is_file() and DEFAULT_POSE_JSON.is_file():
            pose_p = DEFAULT_POSE_JSON
        out_p = Path(OUTPUT_BLEND)
        if str(out_p).startswith(r"C:\path\to"):
            out_p = DEFAULT_OUTPUT_BLEND
        retarget(
            pose_json_path=str(pose_p),
            output_blend_path=str(out_p),
            export_glb_path=str(DEFAULT_OUTPUT_GLB)
        )

if __name__ == "__main__":
    main()
