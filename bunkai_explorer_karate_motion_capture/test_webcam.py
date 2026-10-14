import time
from motion_engine import MotionCaptureEngine

engine = MotionCaptureEngine()
engine.set_source("webcam", 0)
engine.start()

time.sleep(2.0)
found = False
for i in range(20):
    time.sleep(0.1)
    pose = engine.get_latest_pose()
    if pose.get("detected"):
        vis = pose.get("visibility", {})
        lms = pose.get("landmarks", {})
        head = pose.get("head", {})
        print(f"Webcam Pose: detected={pose.get('detected')}, FPS={pose.get('fps')}")
        print("  Visibility:", {k: vis.get(k) for k in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_knee', 'right_knee']})
        print("  Head:", head)
        found = True
        break

if not found:
    print("No pose detected from webcam.")
engine.stop()
