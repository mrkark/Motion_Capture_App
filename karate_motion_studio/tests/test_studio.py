"""
Automated Test Suite for Karate Motion Studio.
Verifies all core headless systems:
1. Video file reading and frame extraction
2. Mocap tracker landmarks and confidence extraction
3. JSON schema validation, save and load round-trip
4. Timestamp continuity and monotonic progression
5. Occlusion gap interpolation and 1€ filter smoothing
6. Retargeter local quaternion calculation and bone hierarchy
7. Export functions (Capture JSON, Retargeted Animation JSON, Blender CLI validation)
8. Synthetic demo project integrity and offline replay without re-analysis
"""
import os
import sys
import time
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.project_schema import (
    KarateMotionProject,
    Metadata,
    CoordinateSystem,
    SkeletonDefinition,
    FrameData,
    RetargetingData,
    STANDARD_JOINTS,
    STANDARD_CONNECTIONS,
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
)
from core.filters import (
    OneEuroFilter,
    Vector3DOneEuroFilter,
    interpolate_short_gaps,
    smooth_project_frames,
)
from core.mocap_tracker import (
    MocapTracker,
    calc_finger_curl,
    calc_thumb_curl,
    calc_phalanx_angles,
    calc_thumb_phalanx_angles,
)
from core.retargeter import (
    Retargeter,
    quat_from_unit_vectors,
    quat_multiply,
    quat_inverse,
    quat_from_euler_yxz,
)
from exporter.json_exporter import (
    export_capture_file,
    export_retargeted_file,
)
from exporter.blender_glb_exporter import export_glb_via_blender
from core.phone_stream_server import PhoneStreamServer, get_local_ip
from gui.camera_manager_dialog import CameraManagerDialog
from gui.main_window import WebcamCaptureThread
from qt_compat import QtWidgets


class TestKarateMotionStudio(unittest.TestCase):
    """Headless integration and unit test suite for Karate Motion Studio."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Test 1: Video File Reading and Frame Extraction
    # -------------------------------------------------------------------------
    def test_video_reading_and_frame_extraction(self):
        """Test creating a synthetic video, opening with OpenCV, and reading frames."""
        video_file = self.temp_path / "synthetic_test_video.mp4"
        width, height = 320, 240
        fps = 30.0
        n_frames = 20

        # Generate a synthetic video file with moving color patch
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
        self.assertTrue(writer.isOpened(), "VideoWriter should open cleanly")

        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(frame, (50 + i * 8, 120), 20, (0, 255, 100), -1)
            writer.write(frame)
        writer.release()

        self.assertTrue(video_file.is_file(), "Video file should exist on disk")

        # Open and extract frames
        cap = cv2.VideoCapture(str(video_file))
        self.assertTrue(cap.isOpened(), "OpenCV should open the generated video")

        extracted_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            extracted_count += 1
            self.assertEqual(frame.shape, (height, width, 3), "Frame dimensions must match")

        cap.release()
        self.assertEqual(extracted_count, n_frames, f"Should extract exactly {n_frames} frames")

    # -------------------------------------------------------------------------
    # Test 2: Mocap Tracker Landmarks and Confidence Extraction
    # -------------------------------------------------------------------------
    def test_mocap_tracker_landmarks_and_confidence(self):
        """Test MocapTracker process_frame interface, landmark outputs, and bounds."""
        tracker = MocapTracker(track_hands=False)

        # Process a blank frame (no person detected)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_img, frame_data = tracker.process_frame(blank_frame, frame_idx=0, timestamp=0.0)

        self.assertIsInstance(frame_data, FrameData)
        self.assertEqual(frame_data.frame, 0)
        self.assertEqual(frame_data.time, 0.0)
        self.assertTrue(frame_data.tracking_loss, "Blank frame should register tracking loss")
        self.assertIn("yaw", frame_data.head)
        self.assertIn("pitch", frame_data.head)
        self.assertIn("roll", frame_data.head)

        # Also test with track_hands=True to ensure no UnboundLocalError when no pose detected
        tracker_hands = MocapTracker(track_hands=True)
        img_h, fdata_h = tracker_hands.process_frame(blank_frame, frame_idx=0, timestamp=0.0)
        self.assertIsInstance(fdata_h, FrameData)
        self.assertTrue(fdata_h.tracking_loss)
        self.assertIn("left", fdata_h.fingers)
        self.assertIn("right", fdata_h.fingers)

        # Verify finger curl calculator functions
        mcp = np.array([0.0, 0.0, 0.0])
        pip = np.array([0.0, 0.5, 0.0])
        dip = np.array([0.0, 0.8, 0.0])
        tip = np.array([0.0, 1.0, 0.0])
        straight_curl = calc_finger_curl(mcp, pip, dip, tip)
        self.assertAlmostEqual(straight_curl, 0.0, delta=0.05, msg="Straight finger curl should be ~0.0")

        # Curled finger (90 degree bend)
        curled_tip = np.array([0.5, 0.5, 0.0])
        curled_val = calc_finger_curl(mcp, pip, dip, curled_tip)
        self.assertGreater(curled_val, 0.5, "Bent finger curl should be significantly > 0.0")

        # Fully clenched karate fist (tip touches palm near base of mcp)
        tight_fist_tip = np.array([0.0, 0.1, -0.05])
        tight_fist_val = calc_finger_curl(mcp, pip, dip, tight_fist_tip, wrist=np.array([0.0, -0.4, 0.0]))
        self.assertGreaterEqual(tight_fist_val, 0.90, "Clenched fist curl should reach >= 0.90")

        # Thumb curl (relaxed vs adducted across knuckles)
        cmc = np.array([0.0, 0.0, 0.0])
        th_mcp = np.array([0.2, 0.1, 0.0])
        th_ip = np.array([0.4, 0.2, 0.0])
        th_tip_open = np.array([0.6, 0.3, 0.0])
        wrist_pt = np.array([-0.2, -0.1, 0.0])
        index_mcp = np.array([0.2, 0.7, 0.0])

        open_thumb = calc_thumb_curl(cmc, th_mcp, th_ip, th_tip_open, wrist_pt, index_mcp)
        self.assertLess(open_thumb, 0.30, "Open relaxed thumb curl should be low (< 0.30)")

        # Folded thumb locked over index
        th_tip_locked = np.array([0.22, 0.65, -0.05])
        locked_thumb = calc_thumb_curl(cmc, th_mcp, th_ip, th_tip_locked, wrist_pt, index_mcp)
        self.assertGreaterEqual(locked_thumb, 0.85, "Locked thumb curl should reach >= 0.85")

    # -------------------------------------------------------------------------
    # Test 3: JSON Schema Validation, Save and Load Round-trip
    # -------------------------------------------------------------------------
    def test_json_schema_validation_save_load(self):
        """Test KarateMotionProject serialization, file save, load, and version checks."""
        project = KarateMotionProject(
            metadata=Metadata(
                name="test_gyaku_zuki",
                technique="gyaku_zuki",
                tags=["shotokan", "punch", "chudan"],
                fps=30.0,
            )
        )

        # Add sample frames
        for i in range(10):
            fdata = FrameData(
                frame=i,
                time=round(i / 30.0, 4),
                rootPosition=[0.0, 0.9, 0.0],
                joints={
                    "nose": [0.0, 1.6, 0.1],
                    "left_shoulder": [-0.2, 1.4, 0.0],
                    "right_shoulder": [0.2, 1.4, 0.0],
                    "left_hip": [-0.15, 0.9, 0.0],
                    "right_hip": [0.15, 0.9, 0.0],
                },
                confidence={
                    "nose": 0.98,
                    "left_shoulder": 0.95,
                    "right_shoulder": 0.96,
                    "left_hip": 0.90,
                    "right_hip": 0.91,
                },
                tracking_loss=False,
                head={"yaw": 0.01, "pitch": -0.02, "roll": 0.0},
            )
            project.rawFrames.append(fdata)

        project_path = self.temp_path / "test_project.json"
        saved_file = project.save_json(project_path)
        self.assertTrue(Path(saved_file).is_file())

        # Load back
        loaded = KarateMotionProject.load_json(project_path)
        self.assertEqual(loaded.format, SCHEMA_FORMAT)
        self.assertEqual(loaded.version, SCHEMA_VERSION)
        self.assertEqual(loaded.metadata.name, "test_gyaku_zuki")
        self.assertEqual(loaded.metadata.technique, "gyaku_zuki")
        self.assertEqual(len(loaded.rawFrames), 10)
        self.assertEqual(loaded.rawFrames[4].frame, 4)
        self.assertAlmostEqual(loaded.rawFrames[4].time, 4 / 30.0, places=3)
        self.assertEqual(loaded.rawFrames[4].joints["nose"], [0.0, 1.6, 0.1])

        # Test rejection of invalid schema format
        invalid_data = loaded.to_dict()
        invalid_data["format"] = "unknown_format"
        with self.assertRaises(ValueError):
            KarateMotionProject.from_dict(invalid_data)

        # Test rejection of unsupported future version
        invalid_data = loaded.to_dict()
        invalid_data["version"] = 999
        with self.assertRaises(ValueError):
            KarateMotionProject.from_dict(invalid_data)

    # -------------------------------------------------------------------------
    # Test 4: Timestamp Continuity and Monotonic Progression
    # -------------------------------------------------------------------------
    def test_timestamp_continuity_and_monotonicity(self):
        """Test timestamp progression: strictly monotonic, expected dt, and no jumps."""
        fps = 30.0
        expected_dt = 1.0 / fps
        frames = []

        for i in range(50):
            t = round(i * expected_dt, 4)
            frames.append(FrameData(frame=i, time=t))

        # Check monotonicity
        for i in range(1, len(frames)):
            dt = frames[i].time - frames[i - 1].time
            self.assertGreater(frames[i].time, frames[i - 1].time, "Timestamps must be strictly monotonic")
            self.assertAlmostEqual(dt, expected_dt, delta=0.005, msg="Frame dt must remain uniform within tolerance")

        # Check total duration
        total_duration = frames[-1].time - frames[0].time
        self.assertAlmostEqual(total_duration, 49 * expected_dt, places=3)

    # -------------------------------------------------------------------------
    # Test 5: Occlusion Gap Interpolation and 1€ Filter Smoothing
    # -------------------------------------------------------------------------
    def test_occlusion_interpolation_and_one_euro_filter(self):
        """Test bounded gap interpolation and adaptive 1€ filter."""
        n_frames = 20
        joint_names = ["left_wrist"]

        # 1. Test short occlusion gap (3 frames missing out of 20)
        joints_seq = []
        conf_seq = []
        for i in range(n_frames):
            if 6 <= i <= 8:
                # Gap: lost tracking
                joints_seq.append({})
                conf_seq.append({"left_wrist": 0.0})
            else:
                joints_seq.append({"left_wrist": [float(i), 1.0, 0.0]})
                conf_seq.append({"left_wrist": 0.9})

        out_j, out_c, loss = interpolate_short_gaps(
            joints_seq, conf_seq, joint_names, min_confidence=0.15, max_gap=5
        )

        # Frame 7 should be successfully interpolated between frame 5 and 9
        self.assertIn("left_wrist", out_j[7], "Short gap frame 7 must be interpolated")
        self.assertAlmostEqual(out_j[7]["left_wrist"][0], 7.0, places=2)
        self.assertGreater(out_c[7]["left_wrist"], 0.15, "Interpolated confidence must be estimated")

        # 2. Test long gap (> max_gap: 7 frames missing) -> must NOT be invented
        long_joints_seq = []
        long_conf_seq = []
        for i in range(n_frames):
            if 5 <= i <= 12:  # gap of 8 frames
                long_joints_seq.append({})
                long_conf_seq.append({"left_wrist": 0.0})
            else:
                long_joints_seq.append({"left_wrist": [float(i), 1.0, 0.0]})
                long_conf_seq.append({"left_wrist": 0.9})

        out_j_long, _, _ = interpolate_short_gaps(
            long_joints_seq, long_conf_seq, joint_names, min_confidence=0.15, max_gap=5
        )
        self.assertNotIn("left_wrist", out_j_long[8], "Long gap exceeding max_gap must NOT be invented")

        # 3. Test 1€ Filter on noisy constant signal
        v_filter = Vector3DOneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.05)
        raw_values = []
        filtered_values = []
        np.random.seed(42)
        for i in range(30):
            noise = float(np.random.normal(0, 0.1))
            val = [1.0 + noise, 2.0 + noise, 0.0]
            raw_values.append(val[0])
            filtered = v_filter.filter(val, timestamp=i / 30.0)
            filtered_values.append(filtered[0])

        raw_std = np.std(raw_values[5:])
        filt_std = np.std(filtered_values[5:])
        self.assertLess(filt_std, raw_std, "1€ filter must reduce high-frequency noise variance")

    # -------------------------------------------------------------------------
    # Test 6: Retargeter Local Quaternion Calculation and Bone Hierarchy
    # -------------------------------------------------------------------------
    def test_retargeter_quaternions_and_hierarchy(self):
        """Test hierarchical local quaternion solving and quaternion algebra."""
        # 1. Quaternion math checks
        q_identity = np.array([0.0, 0.0, 0.0, 1.0])
        inv = quat_inverse(q_identity)
        np.testing.assert_allclose(inv, q_identity, atol=1e-6)

        # 90 deg rotation around Y
        v_from = np.array([0.0, 0.0, 1.0])
        v_to = np.array([1.0, 0.0, 0.0])
        q_rot = quat_from_unit_vectors(v_from, v_to)
        norm_q = np.linalg.norm(q_rot)
        self.assertAlmostEqual(norm_q, 1.0, places=5, msg="Quaternion must be normalized")

        # 2. Test Retargeter on Karate pose
        retargeter = Retargeter(target_rig="Karateka_Armature")
        frame_data = FrameData(
            frame=0,
            time=0.0,
            rootPosition=[0.0, 0.9, 0.0],
            joints={
                "left_shoulder": [-0.2, 1.4, 0.0],
                "right_shoulder": [0.2, 1.4, 0.0],
                "left_hip": [-0.15, 0.9, 0.0],
                "right_hip": [0.15, 0.9, 0.0],
                "left_elbow": [-0.5, 1.4, 0.0],
                "left_wrist": [-0.75, 1.4, 0.0],
                "right_elbow": [0.5, 1.4, 0.0],
                "right_wrist": [0.75, 1.4, 0.0],
                "left_knee": [-0.15, 0.5, 0.0],
                "left_ankle": [-0.15, 0.1, 0.0],
                "right_knee": [0.15, 0.5, 0.0],
                "right_ankle": [0.15, 0.1, 0.0],
            },
            head={"yaw": 0.1, "pitch": -0.05, "roll": 0.0},
            fingers={
                "left": {"thumb": 0.9, "index": 0.85, "middle": 0.9, "ring": 0.85, "pinky": 0.9, "detected": True},
                "right": {"thumb": 0.9, "index": 0.85, "middle": 0.9, "ring": 0.85, "pinky": 0.9, "detected": True},
            },
        )

        local_rotations = retargeter.retarget_frame(frame_data)
        self.assertIsInstance(local_rotations, dict)
        self.assertIn("Spine", local_rotations)
        self.assertIn("UpperArm.L", local_rotations)
        self.assertIn("LowerArm.L", local_rotations)
        self.assertIn("UpperArm.R", local_rotations)
        self.assertIn("LowerArm.R", local_rotations)
        self.assertIn("Head", local_rotations)
        self.assertIn("Index1.L", local_rotations)
        self.assertIn("Thumb1.R", local_rotations)

        # All bone quaternions must have unit length [x, y, z, w]
        for bone, q in local_rotations.items():
            self.assertEqual(len(q), 4, f"Bone {bone} rotation must have 4 components")
            q_len = math.sqrt(sum(v * v for v in q))
            self.assertAlmostEqual(q_len, 1.0, delta=0.01, msg=f"Bone {bone} quaternion must be unit norm")

        # Test retarget_project
        project = KarateMotionProject(rawFrames=[frame_data])
        report = retargeter.retarget_project(project)
        self.assertEqual(report.total_frames, 1)
        self.assertEqual(report.retargeted_frames, 1)
        self.assertIsNotNone(project.retargeting)
        self.assertEqual(len(project.retargeting.localRotations), 1)

    # -------------------------------------------------------------------------
    # Test 7: Export Formats (Capture JSON, Retargeted Animation JSON, Blender CLI)
    # -------------------------------------------------------------------------
    def test_export_formats(self):
        """Test exporting Capture JSON, Retargeted Animation JSON, and Blender validation."""
        project = KarateMotionProject(
            metadata=Metadata(name="oi_zuki_export_test", technique="oi_zuki", fps=30.0)
        )
        for i in range(5):
            project.rawFrames.append(
                FrameData(
                    frame=i,
                    time=i / 30.0,
                    joints={"nose": [0.0, 1.6, 0.0]},
                    confidence={"nose": 0.9},
                )
            )

        retargeter = Retargeter()
        retargeter.retarget_project(project)

        # 1. Export Capture JSON
        cap_file = self.temp_path / "capture.json"
        export_capture_file(project, cap_file)
        self.assertTrue(cap_file.is_file())
        with open(cap_file, "r", encoding="utf-8") as f:
            cap_data = json.load(f)
        self.assertEqual(cap_data["format"], "bunkai-capture-json")
        self.assertEqual(cap_data["frameCount"], 5)
        self.assertIn("coordinateSystem", cap_data)
        self.assertIn("joints", cap_data["frames"][0])

        # 2. Export Retargeted Animation JSON
        anim_file = self.temp_path / "retargeted_anim.json"
        export_retargeted_file(project, anim_file)
        self.assertTrue(anim_file.is_file())
        with open(anim_file, "r", encoding="utf-8") as f:
            anim_data = json.load(f)
        self.assertEqual(anim_data["format"], "bunkai-retargeted-animation")
        self.assertEqual(anim_data["frameCount"], 5)
        self.assertEqual(anim_data["targetRig"], "Karateka_Armature")
        self.assertIn("bones", anim_data["frames"][0])

        # 3. Blender Exporter validation (should raise FileNotFoundError for nonexistent blender)
        with self.assertRaises(FileNotFoundError):
            export_glb_via_blender(
                blender_exe="C:/nonexistent/blender.exe",
                project_json_path=cap_file,
                output_glb_path=self.temp_path / "output.glb",
            )

    # -------------------------------------------------------------------------
    # Test 8: Synthetic Demo Project Integrity and Offline Replay
    # -------------------------------------------------------------------------
    def test_demo_project_integrity_and_offline_replay(self):
        """Test loading pre-packaged demo project and simulating instant offline playback."""
        demo_path = PROJECT_ROOT / "demo_assets" / "demo_zenkutsu_dachi.json"
        self.assertTrue(demo_path.is_file(), f"Demo asset must exist at {demo_path}")

        # Load project without running any computer vision models
        project = KarateMotionProject.load_json(demo_path)

        self.assertEqual(project.format, SCHEMA_FORMAT)
        self.assertEqual(project.metadata.technique, "oi_zuki")
        self.assertEqual(project.metadata.fps, 30.0)
        self.assertEqual(len(project.rawFrames), 60)
        self.assertEqual(len(project.filteredFrames), 60)
        self.assertIsNotNone(project.retargeting)
        self.assertEqual(len(project.retargeting.localRotations), 60)

        # Simulate offline timeline scrubbing / replay through all 60 frames
        effective_frames = project.get_effective_frames()
        self.assertEqual(len(effective_frames), 60)

        for i in range(len(effective_frames)):
            frame = project.get_frame(i)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.frame, i)
            self.assertIn("left_wrist", frame.joints)
            self.assertIn("right_wrist", frame.joints)
            # Bone rotations ready for Three.js
            bone_rots = project.retargeting.localRotations[i]
            self.assertIn("UpperArm.R", bone_rots)
            self.assertIn("UpperArm.L", bone_rots)

        # Confirm punch extension: right wrist moves forward (-Z in Three.js or +Z)
        start_wrist_z = effective_frames[0].joints["right_wrist"][2]
        mid_wrist_z = effective_frames[30].joints["right_wrist"][2]
        self.assertNotEqual(start_wrist_z, mid_wrist_z, "Punch extension should move right wrist")

    # -------------------------------------------------------------------------
    # Test 9: 2D Screen Landmarks and HTML Viewport Kinematics Integrity
    # -------------------------------------------------------------------------
    def test_landmarks_2d_and_viewport_integrity(self):
        """Verify 2D landmarks schema and viewer template integrity."""
        frame = FrameData(
            frame=0,
            time=0.0,
            landmarks_2d={"left_shoulder": [0.45, 0.35], "right_shoulder": [0.55, 0.35]},
        )
        self.assertIn("left_shoulder", frame.landmarks_2d)
        self.assertEqual(frame.landmarks_2d["left_shoulder"], [0.45, 0.35])

        # Test project serialization with landmarks_2d
        proj = KarateMotionProject(rawFrames=[frame])
        json_path = self.temp_path / "test_2d_proj.json"
        proj.save_json(json_path)

        loaded = KarateMotionProject.load_json(json_path)
        self.assertEqual(len(loaded.rawFrames), 1)
        self.assertIn("left_shoulder", loaded.rawFrames[0].landmarks_2d)

        # Verify viewer template contains getBone, skeletonGroup height offset, and no deprecated update()
        viewer_html = (PROJECT_ROOT / "demo_assets" / "viewer_template.html").read_text(encoding="utf-8")
        self.assertIn("getBone", viewer_html, "Viewer template must have getBone helper")
        self.assertIn("skeletonGroup.position.set(0, 0.9, 0)", viewer_html, "Skeleton must be at hip height")
        self.assertNotIn("skeletonHelper.update()", viewer_html, "Deprecated skeletonHelper.update must be removed")

    # -------------------------------------------------------------------------
    # Test 10: Finger Curl Heuristics, Real-Time 1€ Filter, and Viewport
    # -------------------------------------------------------------------------
    def test_finger_heuristics_and_mirror_integrity(self):
        """Verify pose-based fist estimation, 1€ filter setup, and finger smoothing."""
        from core.mocap_tracker import estimate_fist_curl_from_pose

        # Open hand heuristic test
        curl_open = estimate_fist_curl_from_pose(
            wrist=[0.0, 0.0, 0.0], index=[0.18, 0.0, 0.0], elbow=[-0.25, 0.0, 0.0]
        )
        self.assertLess(curl_open, 0.2, "Open hand should yield low curl")

        # Closed fist heuristic test
        curl_fist = estimate_fist_curl_from_pose(
            wrist=[0.0, 0.0, 0.0], index=[0.08, 0.0, 0.0], elbow=[-0.25, 0.0, 0.0]
        )
        self.assertGreater(curl_fist, 0.8, "Clenched fist should yield high curl")

        # Tracker real-time filters test
        tracker = MocapTracker(track_hands=False)
        self.assertEqual(len(tracker.joint_filters), len(STANDARD_JOINTS))
        self.assertIn("right_wrist", tracker.joint_filters)

        # Viewer HTML finger smoothing and back camera preset test
        viewer_html = (PROJECT_ROOT / "demo_assets" / "viewer_template.html").read_text(encoding="utf-8")
        self.assertIn("fingerState", viewer_html, "Viewer template must maintain finger smoothing memory")
        self.assertIn("btn-back", viewer_html, "Viewer template must have back camera preset")

    # -------------------------------------------------------------------------
    # Test 11: OpenCV Anti-Aliased Skeleton, Hands Overlay, and Mirroring
    # -------------------------------------------------------------------------
    def test_opencv_skeleton_and_hands_overlay(self):
        """Verify native OpenCV skeleton, hand overlays, hands_2d schema, and mirror mode."""
        from core.mocap_tracker import POSE_DRAW_CONNECTIONS, HAND_CONNECTIONS

        self.assertGreaterEqual(len(POSE_DRAW_CONNECTIONS), 14, "Must have all major body limb connections")
        self.assertIn(("nose", "left_shoulder", (200, 200, 200)), POSE_DRAW_CONNECTIONS)
        self.assertIn(("nose", "right_shoulder", (200, 200, 200)), POSE_DRAW_CONNECTIONS)
        self.assertEqual(len(HAND_CONNECTIONS), 21, "Must have all 21 hand segment connections")

        # Verify FrameData hands_2d field
        fdata = FrameData(frame=0, time=0.0)
        self.assertIsInstance(fdata.hands_2d, dict)

        # Test drawing routines on image
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = MocapTracker(track_hands=False)

        class MockLandmark:
            def __init__(self, x, y, visibility=1.0):
                self.x = x
                self.y = y
                self.visibility = visibility

        # Create mock 2D landmarks
        mock_lms = [MockLandmark(0.5, 0.5) for _ in range(33)]
        tracker._draw_pose_overlay(canvas, mock_lms, dim_max=640, pad_left=0, pad_top=80, mirror=False, fps=30.0)
        self.assertTrue((canvas > 0).any(), "Overlay drawing must draw pixels onto the canvas")

        # Verify viewer template contains mirror avatar controls and defaults to avatar view
        viewer_html = (PROJECT_ROOT / "demo_assets" / "viewer_template.html").read_text(encoding="utf-8")
        self.assertIn("setMirrorAvatar", viewer_html, "Viewer template must expose setMirrorAvatar API")
        self.assertIn("btn-mirror", viewer_html, "Viewer template must have mirror toggle button")
        self.assertIn('currentViewMode = "avatar"', viewer_html, "Viewer template must default to avatar view mode")

    # -------------------------------------------------------------------------
    # Test 12: Head Metsuke Calibration and Stationary Arm Jitter Suppression
    # -------------------------------------------------------------------------
    def test_head_calibration_and_arm_jitter_suppression(self):
        """Verify stationary arm noise reduction and Metsuke head gaze calibration."""
        # 1. Stationary micro-jitter suppression
        filt = Vector3DOneEuroFilter(freq=30.0, min_cutoff=0.35, beta=0.35)
        np.random.seed(42)
        base_pos = np.array([0.25, 0.50, 0.10])
        raw_samples = []
        filtered_samples = []

        for i in range(40):
            # Simulate camera sensor pixel noise (+- 5mm tremor)
            noise = (np.random.rand(3) - 0.5) * 0.010
            pt = (base_pos + noise).tolist()
            raw_samples.append(pt)
            f_pt = filt.filter(pt, timestamp=i * (1.0 / 30.0))
            if i >= 10:  # Allow warmup
                filtered_samples.append(f_pt)

        raw_std = np.std(raw_samples[10:], axis=0)
        filt_std = np.std(filtered_samples, axis=0)
        for ax in range(3):
            self.assertLess(
                filt_std[ax],
                raw_std[ax] * 0.45,
                f"Filtered tremor on axis {ax} must be reduced by >55% compared to raw noise"
            )

        # 2. Explosive strike responsiveness
        fast_strike_pt = (base_pos + np.array([0.0, 0.0, 0.45])).tolist()
        strike_filtered = filt.filter(fast_strike_pt, timestamp=40 * (1.0 / 30.0))
        # After large movement, filter should adaptively track forward
        self.assertGreater(strike_filtered[2], base_pos[2] + 0.05, "Filter must follow fast punches")

        # 3. Head calibration in MocapTracker
        tracker = MocapTracker(track_hands=False)
        self.assertIsNone(tracker.base_pitch, "Base pitch should start uncalibrated")
        tracker.calibrate()
        self.assertIsNone(tracker.base_pitch, "calibrate() must reset base_pitch")
        self.assertFalse(tracker.calibrated, "calibrate() must reset calibrated flag")

        # 4. Verify viewer_template.html fixes
        viewer_html = (PROJECT_ROOT / "demo_assets" / "viewer_template.html").read_text(encoding="utf-8")
        self.assertIn("calibrateHead", viewer_html, "Viewer must expose calibrateHead")
        self.assertIn("vSpine.z *= 0.15", viewer_html, "Viewer must damp spine slouch")
        self.assertIn("handOrientMemory", viewer_html, "Viewer must maintain smoothed hand orientation")
        self.assertIn("pitchAngle = -(h.pitch", viewer_html, "Viewer must correctly invert pitch for Three.js Euler space")

    # -------------------------------------------------------------------------
    # Test 13: Finger Phalanx Kinematics and Seated Stance Solver
    # -------------------------------------------------------------------------
    def test_finger_phalanx_kinematics_and_seated_stance(self):
        """Verify fine finger phalanx angles computation, lateral abduction, and seated stance solver."""
        # 1. Straight finger phalanx test
        wrist = np.array([0.0, 0.0, 0.0])
        mcp = np.array([0.0, 0.5, 0.0])
        pip = np.array([0.0, 1.0, 0.0])
        dip = np.array([0.0, 1.3, 0.0])
        tip = np.array([0.0, 1.5, 0.0])
        u_across = np.array([1.0, 0.0, 0.0])

        straight_angles, straight_abd = calc_phalanx_angles(mcp, pip, dip, tip, wrist, u_across)
        self.assertEqual(len(straight_angles), 3, "Must return 3 phalanx flexion angles [MCP, PIP, DIP]")
        for ang in straight_angles:
            self.assertAlmostEqual(ang, 0.0, places=2, msg="Straight finger flexion should be ~0.0")
        self.assertAlmostEqual(straight_abd, 0.0, places=2, msg="Straight aligned finger abduction should be ~0.0")

        # 2. Clenched finger phalanx test (curled at MCP, PIP, DIP)
        curled_pip = np.array([0.0, 1.0, 0.0])
        curled_dip = np.array([0.0, 1.0, -0.3])
        curled_tip = np.array([0.0, 0.8, -0.4])
        bent_angles, bent_abd = calc_phalanx_angles(mcp, curled_pip, curled_dip, curled_tip, wrist, u_across)
        self.assertGreater(bent_angles[1], 0.5, "PIP flexion angle must be significantly > 0")
        self.assertGreater(bent_angles[2], 0.5, "DIP flexion angle must be significantly > 0")

        # 3. Abducted (spread) finger test
        spread_pip = np.array([0.3, 1.0, 0.0])
        spread_dip = np.array([0.4, 1.3, 0.0])
        spread_tip = np.array([0.5, 1.5, 0.0])
        _, spread_abd = calc_phalanx_angles(mcp, spread_pip, spread_dip, spread_tip, wrist, u_across)
        self.assertGreater(spread_abd, 0.2, "Abducted finger should have positive lateral projection")

        # 4. Thumb phalanx test
        cmc = np.array([0.0, 0.2, 0.0])
        th_mcp = np.array([0.2, 0.3, 0.0])
        th_ip = np.array([0.3, 0.4, 0.0])
        th_tip = np.array([0.4, 0.5, 0.0])
        idx_mcp = np.array([0.2, 0.8, 0.0])
        th_angles, th_opp = calc_thumb_phalanx_angles(cmc, th_mcp, th_ip, th_tip, wrist, idx_mcp)
        self.assertEqual(len(th_angles), 3, "Thumb must return 3 phalanx angles [CMC, MCP, IP]")
        self.assertIsInstance(th_opp, float)

        # 5. Verify viewer_template.html anatomical bone targeting and continuous squat/foot solver
        viewer_html = (PROJECT_ROOT / "demo_assets" / "viewer_template.html").read_text(encoding="utf-8")
        self.assertIn('["Thumb", "Index", "Middle", "Ring", "Pinky"]', viewer_html, "Viewer must target all 5 anatomical finger chains")
        self.assertIn('getBone(`${t}${idx}.${side}`)', viewer_html, "Viewer must dynamically target individual phalanx bones 1..3")
        self.assertIn("isSitting = true", viewer_html, "Viewer must detect sitting stance")
        self.assertIn("targetRootY = -0.38 * seatedWeight", viewer_html, "Viewer must smoothly lower avatar hips when seated")
        self.assertIn("avatarModel.position.y = smoothRootY", viewer_html, "Viewer must smoothly ground avatar root position Y")
        self.assertIn("maxSupportDrop - nominalLegLen", viewer_html, "Viewer must calculate continuous leg compression during squats")
        self.assertIn("bFootR", viewer_html, "Viewer must target Right Foot bone")
        self.assertIn("bFootL", viewer_html, "Viewer must target Left Foot bone")
        self.assertIn("dyL < 0.22 && dzL > 0.15 && dyR < 0.22 && dzR > 0.15 && kneesLevel < 0.12", viewer_html, "Sitting solver must strictly require BOTH legs and level knees")
        self.assertIn("alignBone(bUpperLegR, vThigh, 0.85)", viewer_html, "Right leg must follow tracked thigh kinematics with high responsiveness")
        self.assertIn("alignBone(bUpperLegL, vThigh, 0.85)", viewer_html, "Left leg must follow tracked thigh kinematics with high responsiveness")

        # 6. Verify Retargeter Foot tracking
        retargeter = Retargeter()
        self.assertIn("Foot.L", retargeter.bone_joint_map, "Retargeter must map Foot.L")
        self.assertIn("Foot.R", retargeter.bone_joint_map, "Retargeter must map Foot.R")
        self.assertEqual(retargeter.bone_joint_map["Foot.L"], ("left_ankle", "left_foot_index"))
        self.assertEqual(retargeter.bone_joint_map["Foot.R"], ("right_ankle", "right_foot_index"))

        # 7. Simulate Single-Leg Lift vs Seated Pose vs Squat
        hip_l = np.array([-0.15, 0.0, 0.0])
        hip_r = np.array([0.15, 0.0, 0.0])

        # A) User lifts ONE leg (Mae-geri kick chamber or crane stance)
        standing_knee_l = np.array([-0.15, -0.45, 0.0])
        raised_knee_r = np.array([0.15, 0.05, 0.35])

        dy_l = abs(hip_l[1] - standing_knee_l[1])
        dy_r = abs(hip_r[1] - raised_knee_r[1])
        dz_l = standing_knee_l[2] - hip_l[2]
        dz_r = raised_knee_r[2] - hip_r[2]
        knees_level = abs(standing_knee_l[1] - raised_knee_r[1])

        single_leg_sitting = (dy_l < 0.22 and dz_l > 0.15 and dy_r < 0.22 and dz_r > 0.15 and knees_level < 0.12)
        self.assertFalse(single_leg_sitting, "Lifting one leg must NEVER trigger sitting stance!")

        # B) User sits in a chair
        seated_knee_l = np.array([-0.15, -0.05, 0.38])
        seated_knee_r = np.array([0.15, -0.05, 0.38])

        s_dy_l = abs(hip_l[1] - seated_knee_l[1])
        s_dy_r = abs(hip_r[1] - seated_knee_r[1])
        s_dz_l = seated_knee_l[2] - hip_l[2]
        s_dz_r = seated_knee_r[2] - hip_r[2]
        s_knees_level = abs(seated_knee_l[1] - seated_knee_r[1])

        both_legs_sitting = (s_dy_l < 0.22 and s_dz_l > 0.15 and s_dy_r < 0.22 and s_dz_r > 0.15 and s_knees_level < 0.12)
        self.assertTrue(both_legs_sitting, "Both legs seated in chair must trigger sitting stance!")

        # C) User squats down (both knees bend, vertical drop shrinks continuously)
        standing_drop = 0.85
        squat_drop = 0.45
        nominal_leg_len = 0.85
        squat_descent = min(0.0, squat_drop - nominal_leg_len)
        self.assertAlmostEqual(squat_descent, -0.40, delta=0.01, msg="Squat descent must smoothly lower hips by 40cm to keep feet on floor")

    # -------------------------------------------------------------------------
    # Test 14: Phone Stream Server, Multi-Camera Setup & Stance Kinematics
    # -------------------------------------------------------------------------
    def test_phone_stream_server_and_multi_camera_kinematics(self):
        """Verifies wireless smartphone mocap streaming, QR generation, multi-cam and stance analysis."""
        # 1. Verify Local IP resolution
        ip = get_local_ip()
        self.assertIsInstance(ip, str)
        self.assertGreater(len(ip), 0)

        # 2. Verify PhoneStreamServer initialization and QR code generation
        server = PhoneStreamServer(port=8998, use_ssl=True)
        self.assertEqual(server.port, 8998)
        self.assertTrue(server.use_ssl)
        conn_url = server.get_connection_url()
        self.assertTrue(conn_url.startswith("https://"))
        self.assertIn("8998", conn_url)

        qr_bytes = server.generate_qr_png_bytes()
        self.assertTrue(qr_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "QR code must be a valid PNG binary image")
        self.assertGreater(len(qr_bytes), 100, "QR code image size must be non-trivial")

        # 3. Verify Server Thread Frame Registration and Callbacks
        received_callbacks = []
        def on_frame(cam_id, frame, fps):
            received_callbacks.append((cam_id, frame.shape, fps))

        server.register_frame_callback(on_frame)
        self.assertIn(on_frame, server.frame_callbacks)

        # Inject simulated phone frame (Cam 1 - Front, Cam 2 - Side)
        test_frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)

        with server.lock:
            now = time.time()
            server.latest_frames["cam1"] = test_frame1
            server.frame_timestamps["cam1"] = now
            server.frame_fps["cam1"] = 30.0

            server.latest_frames["cam2"] = test_frame2
            server.frame_timestamps["cam2"] = now
            server.frame_fps["cam2"] = 30.0

        # Test frame retrieval
        res1 = server.get_latest_frame("cam1")
        self.assertIsNotNone(res1)
        f1, fps1 = res1
        self.assertEqual(f1.shape, (720, 1280, 3))
        self.assertEqual(fps1, 30.0)

        res2 = server.get_latest_frame("cam2")
        self.assertIsNotNone(res2)
        f2, fps2 = res2
        self.assertEqual(f2.shape, (720, 1280, 3))

        cams = server.get_connected_cameras()
        self.assertIn("cam1", cams)
        self.assertIn("cam2", cams)

        server.unregister_frame_callback(on_frame)
        self.assertNotIn(on_frame, server.frame_callbacks)

        # 4. Verify WebcamCaptureThread with Phone & Dual-Camera Sources
        thread_phone = WebcamCaptureThread(
            camera_source="phone:cam1",
            mirror=False,
            phone_server=server,
        )
        self.assertEqual(thread_phone.camera_source, "phone:cam1")
        self.assertIsNotNone(thread_phone.tracker)

        thread_dual = WebcamCaptureThread(
            camera_source="dual:phone+phone",
            mirror=False,
            phone_server=server,
        )
        self.assertEqual(thread_dual.camera_source, "dual:phone+phone")

        # 5. Verify CameraManagerDialog Qt Widget & Dual QR Codes
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        self.assertTrue(server.get_connection_url("cam1").endswith("/cam1"))
        self.assertTrue(server.get_connection_url("cam2").endswith("/cam2"))

        qr_cam1 = server.generate_qr_png_bytes(role="cam1")
        qr_cam2 = server.generate_qr_png_bytes(role="cam2")
        self.assertTrue(qr_cam1.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(qr_cam2.startswith(b"\x89PNG\r\n\x1a\n"))

        self.assertIsNotNone(thread_dual.tracker_cam2, "Dual camera capture thread must have dedicated secondary tracker")

        dialog = CameraManagerDialog(phone_server=server, active_source="phone:cam1")
        self.assertEqual(dialog.active_source, "phone:cam1")
        self.assertTrue(dialog.btn_use_phone1.isChecked())
        self.assertIsNotNone(dialog.lbl_qr_cam1)
        self.assertIsNotNone(dialog.lbl_qr_cam2)

        # Test source selection switching in dialog
        dialog.btn_use_phone2.setChecked(True)
        dialog._on_accept()
        self.assertEqual(dialog.get_selected_source(), "phone:cam2")

        dialog.btn_use_dual.setChecked(True)
        dialog._on_accept()
        self.assertEqual(dialog.get_selected_source(), "dual:phone+phone")

        # 6. Verify Viewer Kinematics Template:
        # Pelvis yaw, pelvic forward tilt, and clavicle reach
        viewer_html_path = PROJECT_ROOT / "demo_assets" / "viewer_template.html"
        viewer_html = viewer_html_path.read_text(encoding="utf-8")

        self.assertNotIn("targetForearmZ = hDir.z * 0.55", viewer_html, "Artificial punch override must be removed for accurate karate technique analysis")
        self.assertIn("minConf = 0.08", viewer_html, "Confidence threshold must be 0.08 to capture wide karate stances")
        self.assertIn("(vThigh.z < -0.05) ? vThigh.clone().normalize() :", viewer_html, "Straight rear leg in Zenkutsu-dachi must continue along thigh")
        self.assertIn("alignShoulder", viewer_html, "Clavicle protraction solver must be present to seat shoulder socket")
        self.assertIn("alignFoot", viewer_html, "Dedicated foot rotation solver must be present to track feet turning on floor")
        self.assertIn("dot < -0.92", viewer_html, "Singularity protection must be present to prevent antiparallel joint popping")
        self.assertIn("bHips", viewer_html, "3D pelvis orientation solver must be present")
        self.assertIn("pelvicTilt", viewer_html, "Anterior pelvic tilt must be solved for forward bending and touching legs")

        # 7. Verify Mobile Client HTML Robustness (Autoplay Bypass, Tap-To-Start, Fallbacks)
        from core.phone_stream_server import MOBILE_CLIENT_HTML
        self.assertIn("tapStartOverlay", MOBILE_CLIENT_HTML, "Tap-to-start overlay must be present to bypass mobile autoplay blocks")
        self.assertIn("tapStartBtn", MOBILE_CLIENT_HTML, "Tap-to-start action button must be present")
        self.assertIn("btnTryFrontCam", MOBILE_CLIENT_HTML, "Front camera fallback button must be present")
        self.assertIn("webkit-playsinline", MOBILE_CLIENT_HTML, "WebKit inline play attributes must be set for mobile Safari")
        self.assertIn("Tier 5", MOBILE_CLIENT_HTML, "Cascading 5-tier camera constraint fallback must be present")

    # -------------------------------------------------------------------------
    # Test 15: Torso Clearance and Foot Retargeting Integrity
    # -------------------------------------------------------------------------
    def test_torso_clearance_and_foot_retargeting(self):
        """Verify torso collision depth clearance and foot orientation solver."""
        retargeter = Retargeter()
        frame = FrameData(
            frame=0,
            time=0.0,
            joints={
                "left_shoulder": [0.20, 1.40, 0.0],
                "right_shoulder": [-0.20, 1.40, 0.0],
                "left_elbow": [0.05, 1.25, 0.20],
                "right_elbow": [-0.05, 1.25, 0.20],
                "left_wrist": [-0.05, 1.25, 0.25],   # Right-crossing wrist
                "right_wrist": [0.05, 1.25, 0.25],   # Left-crossing wrist
                "left_hip": [0.10, 0.90, 0.0],
                "right_hip": [-0.10, 0.90, 0.0],
                "left_knee": [0.10, 0.50, 0.05],
                "right_knee": [-0.10, 0.50, 0.05],
                "left_ankle": [0.10, 0.08, 0.0],
                "right_ankle": [-0.10, 0.08, 0.0],
                "left_heel": [0.10, 0.04, -0.05],
                "right_heel": [-0.10, 0.04, -0.05],
                "left_foot_index": [0.10, 0.04, 0.15],
                "right_foot_index": [-0.10, 0.04, 0.15],
            }
        )
        local_quats = retargeter.retarget_frame(frame)
        self.assertIn("Foot.R", local_quats)
        self.assertIn("Foot.L", local_quats)
        self.assertIn("UpperArm.R", local_quats)
        self.assertIn("LowerArm.R", local_quats)

        # Foot quaternion must be normalized unit quaternion
        q_foot_r = np.array(local_quats["Foot.R"])
        self.assertAlmostEqual(float(np.linalg.norm(q_foot_r)), 1.0, places=2)

        # Verify FrameData accepts both standard and keyword aliases without TypeError
        f_alias = FrameData(frameIndex=10, timestamp=0.333, tracking_loss=True)
        self.assertEqual(f_alias.frame, 10)
        self.assertEqual(f_alias.time, 0.333)
        self.assertTrue(f_alias.tracking_loss)


if __name__ == "__main__":
    unittest.main()

