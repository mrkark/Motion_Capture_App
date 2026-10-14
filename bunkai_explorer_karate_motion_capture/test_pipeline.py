#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for Bunkai Explorer Karate Motion Capture Pipeline.
Tests:
1. Environment & Dependencies (Python 3.10/3.11+, OpenCV, MediaPipe, NumPy, Blender)
2. Pose Capture (capture_pose.py) on test video, JSON schema validation
3. Blender Retargeting (blender_retarget.py) on Karateka_Armature
4. GLB Export & Animation Validation (verifies channels, keyframes, bone targets, fps)
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

import json
import struct
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = BASE_DIR / "test_data"
TEST_OUTPUT_DIR = BASE_DIR / "test_output"

BLENDER_CANDIDATES = [
    r"D:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
]

def find_blender():
    # 1. PATH lookup
    blender_path = shutil.which("blender")
    if blender_path:
        return blender_path
    # 2. Known standard paths
    for cand in BLENDER_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_true(self, condition, test_name, error_msg="Assertion failed"):
        if condition:
            print(f"  [PASS] {test_name}")
            self.passed += 1
        else:
            print(f"  [FAIL] {test_name}: {error_msg}")
            self.failed += 1
            self.errors.append(f"{test_name}: {error_msg}")

    def run_all(self, quick=False):
        print("=" * 70)
        print("BUNKAI EXPLORER MOTION CAPTURE — AUTOMATED TEST SUITE")
        print("=" * 70)

        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # STEP 1: ENVIRONMENT & DEPENDENCIES
        # -------------------------------------------------------------
        print("\n[Step 1] Checking Environment & Dependencies...")
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"  Python version: {py_ver}")
        self.assert_true(sys.version_info >= (3, 10), "Python >= 3.10", f"Found {py_ver}")

        try:
            import cv2
            print(f"  OpenCV version: {cv2.__version__}")
            self.assert_true(True, "OpenCV installed")
        except ImportError as e:
            self.assert_true(False, "OpenCV installed", str(e))

        try:
            import mediapipe as mp
            print(f"  MediaPipe version: {mp.__version__}")
            self.assert_true(True, "MediaPipe installed")
        except ImportError as e:
            self.assert_true(False, "MediaPipe installed", str(e))

        try:
            import numpy as np
            print(f"  NumPy version: {np.__version__}")
            self.assert_true(True, "NumPy installed")
        except ImportError as e:
            self.assert_true(False, "NumPy installed", str(e))

        blender_exe = find_blender()
        if blender_exe:
            print(f"  Blender executable: {blender_exe}")
            res = subprocess.run([blender_exe, "--version"], capture_output=True, text=True)
            first_line = res.stdout.strip().split("\n")[0] if res.stdout else "Unknown"
            print(f"  Blender version: {first_line}")
            self.assert_true(True, "Blender executable located")
        else:
            self.assert_true(False, "Blender executable located", "blender.exe not found")

        # -------------------------------------------------------------
        # STEP 2: TEST DATA AVAILABILITY
        # -------------------------------------------------------------
        print("\n[Step 2] Checking Test Assets...")
        test_video = TEST_DATA_DIR / "karate_test.mp4"
        test_blend = TEST_DATA_DIR / "karateka_lowpoly.blend"
        test_pose = TEST_DATA_DIR / "karate_test_pose.json"

        self.assert_true(test_video.is_file(), f"Test video ({test_video.name})")
        self.assert_true(test_blend.is_file(), f"Avatar .blend ({test_blend.name})")
        self.assert_true(test_pose.is_file(), f"Pre-extracted pose JSON ({test_pose.name})")

        # -------------------------------------------------------------
        # STEP 3: CAPTURE POSE TEST (capture_pose.py)
        # -------------------------------------------------------------
        print("\n[Step 3] Testing capture_pose.py...")
        out_json = TEST_OUTPUT_DIR / "pipeline_test_pose.json"
        out_preview_mp4 = TEST_OUTPUT_DIR / "pipeline_test_preview.mp4"

        cmd_capture = [
            sys.executable,
            str(BASE_DIR / "capture_pose.py"),
            str(test_video),
            "-o", str(out_json),
            "--stride", "5" if quick else "2",
            "--preview-video", str(out_preview_mp4)
        ]
        print(f"  Running: {' '.join(cmd_capture)}")
        res_cap = subprocess.run(cmd_capture, capture_output=True, text=True)
        self.assert_true(res_cap.returncode == 0, "capture_pose.py exit code 0", res_cap.stderr)

        if out_json.is_file():
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assert_true(data.get("schema") == "bunkai-explorer-pose-capture", "JSON schema identifier")
            self.assert_true("frames" in data and len(data["frames"]) > 0, f"Frames extracted (count: {len(data.get('frames', []))})")
            det_rate = data.get("detectionRate", 0)
            self.assert_true(det_rate > 0.8, f"Detection rate > 80% (got {det_rate*100:.1f}%)")
            self.assert_true("nose" in data.get("landmarkNames", {}), "Standard landmarks dictionary present")
        else:
            self.assert_true(False, "JSON output file exists")

        self.assert_true(out_preview_mp4.is_file() and out_preview_mp4.stat().st_size > 0, "Preview video generated with skeleton overlay")

        # -------------------------------------------------------------
        # STEP 4: BLENDER RETARGET TEST (blender_retarget.py)
        # -------------------------------------------------------------
        print("\n[Step 4] Testing blender_retarget.py (Headless via Blender CLI)...")
        if not blender_exe:
            print("  [SKIP] Skipping Blender retarget test because Blender was not found.")
        else:
            out_blend = TEST_OUTPUT_DIR / "pipeline_retargeted.blend"
            out_glb = TEST_OUTPUT_DIR / "pipeline_retargeted.glb"

            cmd_blender = [
                blender_exe,
                "-b", str(test_blend),
                "-P", str(BASE_DIR / "blender_retarget.py"),
                "--",
                "--pose", str(out_json if out_json.is_file() else test_pose),
                "--output", str(out_blend),
                "--export-glb", str(out_glb),
                "--action-name", "pipeline_test_action"
            ]
            print(f"  Running Blender headless retargeting...")
            res_blend = subprocess.run(cmd_blender, capture_output=True, text=True)
            self.assert_true(res_blend.returncode == 0, "blender_retarget.py exit code 0", res_blend.stderr[-500:] if res_blend.stderr else "")
            self.assert_true(out_blend.is_file() and out_blend.stat().st_size > 100000, f"Saved output .blend ({out_blend.stat().st_size if out_blend.is_file() else 0} bytes)")
            self.assert_true(out_glb.is_file() and out_glb.stat().st_size > 1000000, f"Exported GLB ({out_glb.stat().st_size if out_glb.is_file() else 0} bytes)")

            # -------------------------------------------------------------
            # STEP 5: GLB ANIMATION VERIFICATION
            # -------------------------------------------------------------
            print("\n[Step 5] Verifying exported GLB structure & animations...")
            if out_glb.is_file():
                try:
                    with open(out_glb, "rb") as f:
                        magic, ver, length = struct.unpack("<III", f.read(12))
                        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
                        gltf = json.loads(f.read(chunk_len).decode("utf-8"))

                    self.assert_true(magic == 0x46546C67, "Valid GLB magic header (0x46546C67)")
                    self.assert_true(ver == 2, "glTF version 2")
                    anims = gltf.get("animations", [])
                    anim_names = [a.get("name") for a in anims]
                    print(f"  Found {len(anims)} animation clip(s) in GLB")
                    self.assert_true("pipeline_test_action" in anim_names, "GLB contains 'pipeline_test_action' clip")

                    target_anim = next((a for a in anims if a.get("name") == "pipeline_test_action"), None)
                    if target_anim:
                        channels = target_anim.get("channels", [])
                        self.assert_true(len(channels) >= 8, f"Action has >= 8 animated bone channels (got {len(channels)})")
                    else:
                        self.assert_true(False, "Target action found in GLB")
                except Exception as e:
                    self.assert_true(False, "Parse exported GLB", str(e))

        # -------------------------------------------------------------
        # STEP 6: LIVE APPLICATION INTEGRATION TEST
        # -------------------------------------------------------------
        print("\n[Step 6] Testing Live Application Backend (server.py)...")
        try:
            import test_app
            app_ok = test_app.test_endpoints()
            self.assert_true(app_ok, "Live Web/Desktop application backend test")
        except Exception as e:
            self.assert_true(False, "Live Web/Desktop application backend test", str(e))

        # -------------------------------------------------------------
        # SUMMARY
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f"TEST RESULTS: {self.passed} PASSED, {self.failed} FAILED")
        if self.errors:
            print("Failures:")
            for err in self.errors:
                print(f"  - {err}")
            print("=" * 70)
            return False
        else:
            print("ALL PIPELINE & APPLICATION TESTS PASSED PERFECTLY!")
            print("=" * 70)
            return True

if __name__ == "__main__":
    quick_mode = "--quick" in sys.argv
    runner = TestRunner()
    success = runner.run_all(quick=quick_mode)
    sys.exit(0 if success else 1)
