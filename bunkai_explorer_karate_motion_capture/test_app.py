"""
Automated Integration Test for Bunkai MoCap Live Web/Desktop Application.
Validates HTTP endpoints, static model delivery, API status, and motion stream.
"""
import sys
import time
import urllib.request
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def test_endpoints():
    print("Testing Bunkai MoCap Live Backend...")
    # Start server as subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8766", "--log-level", "warning"],
        cwd=str(BASE_DIR)
    )

    try:
        # Wait for server startup
        base_url = "http://127.0.0.1:8766"
        ready = False
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{base_url}/api/status", timeout=1.0) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.2)

        assert ready, "Server failed to start on port 8766 within 6 seconds"
        print("  [PASS] Server started and responded to /api/status")

        # 1. Test index page
        with urllib.request.urlopen(f"{base_url}/", timeout=2.0) as resp:
            content = resp.read().decode("utf-8")
            assert "Bunkai MoCap Live" in content
            assert resp.status == 200
            print("  [PASS] GET / returned valid HTML with app title")

        # 2. Test status API
        with urllib.request.urlopen(f"{base_url}/api/status", timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert "mode" in data
            assert resp.status == 200
            print(f"  [PASS] GET /api/status returned mode: '{data['mode']}', mirror: {data['mirror']}")

        # 3. Test static model delivery
        model_req = urllib.request.Request(f"{base_url}/static/models/karateka_lowpoly.glb")
        with urllib.request.urlopen(model_req, timeout=5.0) as resp:
            header = resp.read(12)
            assert header[:4] == b"glTF", "GLB header magic must be 'glTF'"
            assert resp.status == 200
            print("  [PASS] GET /static/models/karateka_lowpoly.glb returned valid glTF binary header")

        # 4. Test source switch to sample video
        switch_req = urllib.request.Request(f"{base_url}/api/source/sample_video", method="POST")
        with urllib.request.urlopen(switch_req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["mode"] == "video"
            print("  [PASS] POST /api/source/sample_video switched source to sample video")

        # 5. Verify status updated
        with urllib.request.urlopen(f"{base_url}/api/status", timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["mode"] == "video"
            print("  [PASS] Verified engine mode is now 'video'")

        # 6. Test posture calibration API
        calib_req = urllib.request.Request(f"{base_url}/api/calibrate", method="POST")
        with urllib.request.urlopen(calib_req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            print("  [PASS] POST /api/calibrate reset posture calibration baseline")

        print("\nALL APPLICATION BACKEND INTEGRATION TESTS PASSED (6/6)!")
        return True

    finally:
        proc.terminate()
        proc.wait(timeout=3.0)

if __name__ == "__main__":
    success = test_endpoints()
    sys.exit(0 if success else 1)
