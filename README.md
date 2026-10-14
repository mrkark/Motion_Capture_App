# Motion_Capture_App

**Motion Capture App** is an AI-powered motion capture and 3D kinematics reconstruction suite designed for Shotokan Karate and martial arts motion capture. It features real-time 3D landmark extraction (MediaPipe Pose 3D + Hands), an interactive PyQt6 / Three.js desktop studio (**Karate Motion Studio**), wireless multi-camera smartphone streaming via WebSockets/HTTPS, adaptive 1€-filter stabilization, Seichusen spine alignment, anatomical shoulder seating, and precision foot heading & stance rotation tracking.

---

## 🌟 Key Features

1. **Karate Motion Studio (Desktop GUI)**
   - Built on PyQt6 with an embedded Three.js WebGL viewport.
   - Live side-by-side feed: OpenCV video stream with 3D skeleton overlay on the left, fully rigged 3D Karateka avatar (`Karateka_Armature`) on the right.
   - Real-time stance compression solver: roots descend dynamically with leg compression, keeping feet firmly planted on the tatami ground without floating or snapping.
   - Metsuke (neutral gaze) posture calibration to compensate for camera perspective tilt.

2. **Wireless Smartphone Streaming (Zero-Install)**
   - Built-in HTTPS / WebSocket streaming server on port 8088.
   - Scan a QR code from iOS Safari or Android Chrome to connect the phone as a wide-angle wireless mocap camera.
   - Supports Dual-Camera Capture (Front + Side) to eliminate limb occlusions.

3. **Biomechanical & Kinematic Integrity**
   - **Torso Collision Clearance**: Prevents arms from piercing backwards through the chest during cross-torso blocks and punches.
   - **Dynamic Clavicle Protraction (`alignShoulder`)**: Actively engages `Shoulder.L` and `Shoulder.R` to seat the humerus head naturally in the socket, eliminating arm dislocation.
   - **Dedicated Foot Orientation Solver (`alignFoot`)**: Replicates horizontal heading ($\theta_{yaw}$) and heel elevation ($\theta_{pitch}$) on the floor for stances like *Zenkutsu-dachi*, *Kiba-dachi*, and *Kokutsu-dachi*.
   - **Antiparallel Singularity Protection**: Prevents gimbal twists when limbs cross the body.
   - **Real-Time 1€ Filter**: Strips high-frequency sensor noise while preserving explosive martial arts strikes.

4. **Export & Pipeline Compatibility**
   - Export to **Capture JSON** (raw 3D landmarks + confidence) and **Retargeted Animation JSON** (hierarchical local bone quaternions).
   - Headless Blender CLI integration for baking `.glb` animations compatible with Three.js and web platforms like **Bunkai Explorer**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (64-bit)
- Optional: Blender 3.6+ or 4.x/5.x for GLB animation baking.

### 1. Launch Karate Motion Studio
Double-click `run_studio.bat` or run in PowerShell:
```powershell
.\run_studio.ps1
```

### 2. Launch Web-Based MoCap Live
Double-click `run_app.bat` or run in PowerShell:
```powershell
.\run_app.ps1
```

### 3. Run Automated Tests
```powershell
.\bunkai_explorer_karate_motion_capture\.venv\Scripts\python.exe -m unittest discover -s karate_motion_studio/tests -p "test_*.py"
```

---

## 📁 Repository Structure

```text
Motion_Capture_App/
├── karate_motion_studio/                  # Next-Gen Desktop MoCap Studio
│   ├── core/                              # Core tracking, kinematics, filters & schema
│   │   ├── mocap_tracker.py               # MediaPipe 3D Pose + Hands & Torso Clearance
│   │   ├── phone_stream_server.py         # Async HTTPS/WebSocket server for smartphones
│   │   ├── filters.py                     # 1€ Filter & gap interpolation
│   │   ├── retargeter.py                  # Hierarchical local quaternion retargeting
│   │   └── project_schema.py              # KarateMotionProject JSON schema
│   ├── gui/                               # PyQt6 Modern Dark UI
│   │   ├── main_window.py                 # Main Window & Capture thread
│   │   ├── camera_manager_dialog.py       # QR Code phone connection & camera manager
│   │   ├── video_panel.py                 # Video rendering with skeleton overlay
│   │   ├── viewer_3d_panel.py             # QWebEngine Three.js viewer
│   │   └── timeline_widget.py             # Frame scrubbing & trimming
│   ├── demo_assets/                       # 3D assets, viewer HTML & demo project
│   │   ├── karateka_lowpoly.glb           # Rigged 3D model (Karateka_Armature)
│   │   └── viewer_template.html           # Three.js viewport & kinematic solvers
│   ├── exporter/                          # JSON and Blender GLB exporters
│   └── tests/                             # Automated test suite (15 tests, 100% pass)
│
├── bunkai_explorer_karate_motion_capture/ # Web pipeline & CLI tools
│   ├── app.py                             # FastAPI web app
│   ├── server.py                          # Streaming WebSocket server
│   ├── motion_engine.py                   # MediaPipe engine
│   ├── blender_retarget.py                # Blender retargeting script
│   └── templates/                         # Web interface templates
│
├── run_studio.bat / .ps1                  # Launch Karate Motion Studio
├── run_app.bat / .ps1                     # Launch Web MoCap Live
└── README_RU.md                           # Documentation in Russian
```

---

## 📜 License & Acknowledgments
Developed for motion capture, preservation, and biomechanical analysis of traditional karate kata and bunkai techniques. Compatible with the **Bunkai Explorer** web platform.
