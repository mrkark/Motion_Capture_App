import time
from motion_engine import MotionCaptureEngine

engine = MotionCaptureEngine()
engine.set_source("video", "test_data/karate_test.mp4")
engine.start()

# Wait for a frame to process
for _ in range(50):
    time.sleep(0.1)
    pose = engine.get_latest_pose()
    if pose and pose.get("detected"):
        break

engine.stop()

print("Pose detected:", pose.get("detected"))
print("Pose keys:", list(pose.keys()))
print("Landmarks count:", len(pose.get("landmarks", {})))
print("Visibility left_shoulder:", pose.get("visibility", {}).get("left_shoulder"))
print("Visibility left_elbow:", pose.get("visibility", {}).get("left_elbow"))
print("Visibility left_wrist:", pose.get("visibility", {}).get("left_wrist"))
print("Head:", pose.get("head"))
print("Fingers left:", pose.get("fingers", {}).get("left"))
print("Fingers right:", pose.get("fingers", {}).get("right"))

# Let's print sample landmark vectors
lms = pose.get("landmarks", {})
if "left_shoulder" in lms and "left_elbow" in lms:
    sh = lms["left_shoulder"]
    el = lms["left_elbow"]
    wr = lms["left_wrist"]
    print(f"Shoulder.L: {sh}")
    print(f"Elbow.L:    {el}")
    print(f"Wrist.L:    {wr}")
    v_up = [el[i] - sh[i] for i in range(3)]
    v_low = [wr[i] - el[i] for i in range(3)]
    print(f"Upper arm vector: {v_up}")
    print(f"Lower arm vector: {v_low}")
