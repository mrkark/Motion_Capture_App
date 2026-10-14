"""
FastAPI Server for Bunkai MoCap Live Application.
Serves UI, MJPEG video feed, WebSocket real-time telemetry, and REST endpoints for control.
"""
import os
import sys
import time
import asyncio
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from motion_engine import MotionCaptureEngine

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
RECORDINGS_DIR = BASE_DIR / "recordings"

for d in [TEMPLATES_DIR, STATIC_DIR, UPLOADS_DIR, RECORDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Bunkai MoCap Live", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

engine = MotionCaptureEngine()

@app.on_event("startup")
async def startup_event():
    engine.start()

@app.on_event("shutdown")
async def shutdown_event():
    engine.stop()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Bunkai MoCap Live: templates/index.html not found</h1>", status_code=404)

def generate_mjpeg_stream():
    """Generator for streaming camera/video frames over HTTP MJPEG."""
    while True:
        jpeg = engine.get_latest_jpeg()
        if jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.02)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            pose = engine.get_latest_pose()
            if pose:
                await websocket.send_json(pose)
            await asyncio.sleep(0.025)  # ~40 Hz
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")

# --- REST API ---

class SettingsModel(BaseModel):
    mirror: Optional[bool] = None
    smooth: Optional[float] = None
    skeleton: Optional[bool] = None
    fists: Optional[bool] = None

@app.get("/api/status")
async def get_status():
    return {
        "mode": engine.source_mode,
        "is_recording": engine.is_recording,
        "record_frames": len(engine.recorded_frames) if engine.is_recording else 0,
        "fps": round(engine.fps, 1),
        "mirror": engine.mirror_mode,
        "paused": engine.is_paused,
        "smooth": engine.smooth_factor,
        "draw_skeleton": engine.draw_skeleton,
    }

@app.post("/api/source/webcam")
async def set_webcam_source(camera_index: int = 0):
    engine.set_source("webcam", camera_index)
    return {"status": "ok", "mode": "webcam", "index": camera_index}

@app.post("/api/source/sample_video")
async def set_sample_video():
    test_video = BASE_DIR / "test_data" / "karate_test.mp4"
    if not test_video.is_file():
        return JSONResponse({"status": "error", "message": "Sample video not found"}, status_code=404)
    engine.set_source("video", str(test_video))
    return {"status": "ok", "mode": "video", "path": str(test_video)}

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    save_path = UPLOADS_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    engine.set_source("video", str(save_path))
    return {"status": "ok", "mode": "video", "filename": file.filename, "path": str(save_path)}

@app.post("/api/toggle_pause")
async def toggle_pause():
    paused = engine.toggle_pause()
    return {"status": "ok", "paused": paused}

@app.post("/api/calibrate")
async def calibrate_posture():
    engine.calibrate()
    return {"status": "ok", "calibrated": False, "message": "Calibrating neutral posture"}

@app.post("/api/settings")
async def update_settings(cfg: SettingsModel):
    if cfg.mirror is not None:
        engine.mirror_mode = cfg.mirror
    if cfg.smooth is not None:
        engine.smooth_factor = max(0.0, min(0.9, cfg.smooth))
    if cfg.skeleton is not None:
        engine.draw_skeleton = cfg.skeleton
    if cfg.fists is not None:
        engine.clench_fists = cfg.fists
    return {"status": "ok"}

@app.post("/api/record/start")
async def record_start():
    engine.start_recording()
    return {"status": "ok", "recording": True}

@app.post("/api/record/stop")
async def record_stop():
    json_path = engine.stop_recording()
    if not json_path:
        return {"status": "error", "message": "No frames recorded"}
    return {
        "status": "ok",
        "recording": False,
        "json_path": json_path,
        "filename": Path(json_path).name
    }
