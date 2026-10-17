"""
Main Window for Karate Motion Studio.
Integrates Video/Webcam panel, 3D Avatar viewport, Synchronized Timeline,
Metadata controller, Project management, and Exporting.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

import concurrent.futures
from qt_compat import QtCore, QtGui, QtWidgets
from gui.draw_utils import draw_text_utf8, draw_badge_utf8
from core.project_schema import (
    KarateMotionProject,
    Metadata,
    FrameData,
    STANDARD_JOINTS,
)
from core.mocap_tracker import MocapTracker
from core.filters import smooth_project_frames
from core.retargeter import Retargeter
from gui.video_panel import VideoPanel
from gui.viewer_3d_panel import Viewer3DPanel
from gui.timeline_widget import TimelineWidget
from gui.metadata_widget import MetadataWidget
from gui.settings_dialog import SettingsDialog
from gui.camera_manager_dialog import CameraManagerDialog
from core.phone_stream_server import PhoneStreamServer
from exporter.json_exporter import export_capture_file, export_retargeted_file
from exporter.blender_glb_exporter import export_glb_via_blender

DARK_THEME_QSS = """
QMainWindow { background-color: #1a1b1e; }
QWidget { color: #dfdfe0; font-family: "Segoe UI", Arial, sans-serif; font-size: 9pt; }
QToolBar { background-color: #212328; border-bottom: 1px solid #2d3038; spacing: 6px; padding: 4px; }
QToolButton {
    background-color: #2b2d33; color: #dfdfe0; border: 1px solid #3c404c;
    border-radius: 3px; padding: 4px 10px; font-size: 9pt;
}
QToolButton:hover { background-color: #383b44; border-color: #555b6b; }
QToolButton:pressed { background-color: #1e1f23; }
QStatusBar { background-color: #1a1b1e; border-top: 1px solid #282a30; color: #8c939e; font-size: 9pt; }
QSplitter::handle { background-color: #282a30; width: 4px; }
QMenuBar { background-color: #212328; border-bottom: 1px solid #2d3038; font-size: 9pt; }
QMenuBar::item { color: #dfdfe0; padding: 4px 8px; background: transparent; }
QMenuBar::item:selected { background-color: #33363f; }
QMenu { background-color: #26282e; border: 1px solid #3c404c; color: #dfdfe0; font-size: 9pt; }
QMenu::item { padding: 4px 20px 4px 10px; }
QMenu::item:selected { background-color: #383b44; }
"""


class WebcamCaptureThread(QtCore.QThread):
    """Background thread streaming frames from webcam or wireless phone camera without blocking Qt GUI."""

    frameCaptured = QtCore.pyqtSignal(object, object, float)  # (bgr_frame, frame_data, fps)

    def __init__(
        self,
        camera_source: str = "0",
        mirror: bool = False,
        track_hands: bool = True,
        phone_server: Optional[PhoneStreamServer] = None,
    ):
        super().__init__()
        self.camera_source = str(camera_source)
        self.mirror = mirror
        self.track_hands = track_hands
        self.phone_server = phone_server
        self.running = False
        self.tracker = MocapTracker(track_hands=track_hands, model_complexity=1)
        self.tracker_cam2 = MocapTracker(track_hands=False, model_complexity=0)  # Dedicated secondary tracker for Cam 2 (Side view - Ultra-fast Lite model)
        self.dual_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="MocapDual")

        # Recording
        self.is_recording = False
        self.recorded_frames: List[FrameData] = []
        self.record_t0 = 0.0

    def run(self):
        self.running = True
        cap: Optional[cv2.VideoCapture] = None

        is_phone = self.camera_source.startswith("phone:") or self.camera_source.startswith("dual")

        if not is_phone:
            # PC webcam index or IP stream URL
            try:
                idx = int(self.camera_source)
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if cv2.CAP_DSHOW else cv2.CAP_ANY)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
            except ValueError:
                # String URL (e.g. rtsp:// or http://)
                cap = cv2.VideoCapture(self.camera_source)

        fps_timer = time.time()
        fps_counter = 0
        current_fps = 0.0
        frame_idx = 0
        waiting_phase = 0.0

        while self.running:
            t_now = time.time()
            t_rel = t_now - self.record_t0 if self.is_recording else 0.0

            if self.camera_source.startswith("dual"):
                # DUAL CAMERA MODE: Independent parallel tracking for Front (Cam1) and Side (Cam2)
                res1 = self.phone_server.get_latest_frame("cam1") if self.phone_server else None
                res2 = self.phone_server.get_latest_frame("cam2") if self.phone_server else None

                if res1 is None and res2 is None:
                    # Neither camera connected yet: Show clear waiting canvas
                    waiting_phase += 0.06
                    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.rectangle(raw_frame, (20, 20), (1260, 700), (35, 40, 50), 2)
                    url1 = self.phone_server.get_connection_url("cam1") if self.phone_server else "http://.../cam1"
                    url2 = self.phone_server.get_connection_url("cam2") if self.phone_server else "http://.../cam2"

                    draw_text_utf8(raw_frame, "KARATE MOTION STUDIO - РЕЖИМ 2 КАМЕР (СИНХРОННЫЙ ЗАХВАТ)", (230, 200), font_size=24, color=(255, 71, 87))
                    draw_text_utf8(raw_frame, "Для начала захвата подключите 2 смартфона к Wi-Fi:", (320, 260), font_size=20, color=(220, 220, 220))
                    draw_text_utf8(raw_frame, f"1. Камера 1 (ФРОНТ - лицо/корпус): {url1}", (260, 340), font_size=19, color=(16, 185, 129))
                    draw_text_utf8(raw_frame, f"2. Камера 2 (СБОКУ - профиль/глубина): {url2}", (260, 390), font_size=19, color=(6, 182, 212))
                    draw_text_utf8(raw_frame, "Или нажмите 'Камеры / Телефон (QR)' для сканирования QR-кодов", (280, 460), font_size=17, color=(160, 170, 180))

                    pulse_r = int(12 + 6 * np.sin(waiting_phase))
                    cv2.circle(raw_frame, (640, 540), max(2, pulse_r), (255, 71, 87), -1)
                    draw_text_utf8(raw_frame, "ОЖИДАНИЕ ПОТОКОВ...", (520, 580), font_size=18, color=(140, 150, 160))

                    dummy_fdata = FrameData(frame=frame_idx, time=0.0, joints={})
                    self.frameCaptured.emit(raw_frame, dummy_fdata, 0.0)
                    if self.phone_server:
                        self.phone_server.wait_for_frame(0.040)
                    else:
                        time.sleep(0.04)
                    continue

                target_h = 720
                target_half_w = 640

                # Parallel MediaPipe Pose estimation if both cameras are streaming
                if res1 is not None and res2 is not None:
                    f1, fps1 = res1
                    f2, fps2 = res2
                    future1 = self.dual_executor.submit(
                        self.tracker.process_frame,
                        f1, frame_idx, t_rel, self.mirror, True, fps1
                    )
                    future2 = self.dual_executor.submit(
                        self.tracker_cam2.process_frame,
                        f2, frame_idx, t_rel, False, True, fps2
                    )
                    proc_f1, fdata_front = future1.result()
                    proc_f2, fdata_side = future2.result()
                else:
                    if res1 is not None:
                        f1, fps1 = res1
                        proc_f1, fdata_front = self.tracker.process_frame(
                            f1, frame_idx, t_rel, mirror=self.mirror, draw_skeleton=True, fps=fps1
                        )
                    else:
                        proc_f1, fdata_front = None, None

                    if res2 is not None:
                        f2, fps2 = res2
                        proc_f2, fdata_side = self.tracker_cam2.process_frame(
                            f2, frame_idx, t_rel, mirror=False, draw_skeleton=True, fps=fps2
                        )
                    else:
                        proc_f2, fdata_side = None, None

                # Format Cam 1 Tile
                if proc_f1 is not None:
                    h1, w1 = proc_f1.shape[:2]
                    f1_scaled = cv2.resize(proc_f1, (int(w1 * (target_h / h1)), target_h))
                    if f1_scaled.shape[1] > target_half_w:
                        crop_x = (f1_scaled.shape[1] - target_half_w) // 2
                        f1_scaled = f1_scaled[:, crop_x:crop_x + target_half_w]
                    elif f1_scaled.shape[1] < target_half_w:
                        pad_l = (target_half_w - f1_scaled.shape[1]) // 2
                        pad_r = target_half_w - f1_scaled.shape[1] - pad_l
                        f1_scaled = cv2.copyMakeBorder(f1_scaled, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    # Cam 1 waiting placeholder tile on the left
                    f1_scaled = np.zeros((target_h, target_half_w, 3), dtype=np.uint8)
                    cv2.rectangle(f1_scaled, (10, 10), (target_half_w - 10, target_h - 10), (20, 45, 30), 2)
                    draw_text_utf8(f1_scaled, "КАМЕРА 1 (ФРОНТ)", (40, 60), font_size=22, color=(16, 185, 129))
                    draw_text_utf8(f1_scaled, "Ожидание подключения...", (40, 110), font_size=18, color=(160, 170, 180))
                    fdata_front = FrameData(frame=frame_idx, time=0.0, joints={}, tracking_loss=True)

                # Cam 1 badge
                draw_badge_utf8(f1_scaled, "КАМЕРА 1: ФРОНТАЛЬНАЯ", (15, 15), badge_color=(16, 185, 129))

                # Format Cam 2 Tile
                if proc_f2 is not None:
                    h2, w2 = proc_f2.shape[:2]
                    f2_scaled = cv2.resize(proc_f2, (int(w2 * (target_h / h2)), target_h))
                    if f2_scaled.shape[1] > target_half_w:
                        crop_x = (f2_scaled.shape[1] - target_half_w) // 2
                        f2_scaled = f2_scaled[:, crop_x:crop_x + target_half_w]
                    elif f2_scaled.shape[1] < target_half_w:
                        pad_l = (target_half_w - f2_scaled.shape[1]) // 2
                        pad_r = target_half_w - f2_scaled.shape[1] - pad_l
                        f2_scaled = cv2.copyMakeBorder(f2_scaled, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    # Cam 2 waiting placeholder tile on the right
                    f2_scaled = np.zeros((target_h, target_half_w, 3), dtype=np.uint8)
                    cv2.rectangle(f2_scaled, (10, 10), (target_half_w - 10, target_h - 10), (20, 35, 45), 2)
                    draw_text_utf8(f2_scaled, "КАМЕРА 2 (СБОКУ)", (40, 60), font_size=22, color=(6, 182, 212))
                    draw_text_utf8(f2_scaled, "Ожидание подключения...", (40, 110), font_size=18, color=(160, 170, 180))
                    fdata_side = FrameData(frame=frame_idx, time=0.0, joints={}, tracking_loss=True)

                # Cam 2 badge
                draw_badge_utf8(f2_scaled, "КАМЕРА 2: СБОКУ (3D ГЛУБИНА)", (15, 15), badge_color=(6, 182, 212))

                # Multi-Camera 3D Depth Cross-Validation & Fusion
                # Side camera's horizontal displacement is true physical depth (Z) in performer space!
                if res1 is not None and res2 is not None and not fdata_side.tracking_loss:
                    s_joints = fdata_side.joints
                    f_joints = fdata_front.joints
                    if "left_knee" in s_joints and "left_hip" in s_joints and "left_knee" in f_joints and "left_hip" in f_joints:
                        dz_side_l = s_joints["left_knee"][0] - s_joints["left_hip"][0]
                        target_z_l = f_joints["left_hip"][2] + dz_side_l
                        f_joints["left_knee"][2] = round(0.55 * f_joints["left_knee"][2] + 0.45 * target_z_l, 4)

                    if "right_knee" in s_joints and "right_hip" in s_joints and "right_knee" in f_joints and "right_hip" in f_joints:
                        dz_side_r = s_joints["right_knee"][0] - s_joints["right_hip"][0]
                        target_z_r = f_joints["right_hip"][2] + dz_side_r
                        f_joints["right_knee"][2] = round(0.55 * f_joints["right_knee"][2] + 0.45 * target_z_r, 4)

                proc_frame = np.hstack([f1_scaled, f2_scaled])
                fdata = fdata_front if res1 is not None else fdata_side

            elif is_phone:
                # SINGLE PHONE CAMERA MODE (Cam 1 or Cam 2)
                cam_id = "cam1" if self.camera_source == "phone:cam1" else "cam2"
                res = self.phone_server.get_latest_frame(cam_id) if self.phone_server else None

                if res is None:
                    # Waiting screen
                    waiting_phase += 0.06
                    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.rectangle(raw_frame, (20, 20), (1260, 700), (45, 48, 56), 2)
                    url_str = self.phone_server.get_connection_url(cam_id) if self.phone_server else "http://..."
                    target_role = "Камера 1 (Фронт)" if cam_id == "cam1" else "Камера 2 (Сбоку)"

                    draw_text_utf8(raw_frame, "KARATE MOTION STUDIO - ОЖИДАНИЕ ТЕЛЕФОНА", (320, 220), font_size=24, color=(255, 71, 87))
                    draw_text_utf8(raw_frame, f"Ожидание подключения: {target_role}", (360, 280), font_size=20, color=(220, 220, 220))
                    draw_text_utf8(raw_frame, f"1. Подключите телефон к Wi-Fi и откройте: {url_str}", (260, 360), font_size=19, color=(46, 213, 115))
                    draw_text_utf8(raw_frame, "2. Или нажмите 'Камеры / Телефон (QR)' в панели и отсканируйте QR-код", (230, 420), font_size=17, color=(180, 180, 180))

                    pulse_r = int(12 + 6 * np.sin(waiting_phase))
                    cv2.circle(raw_frame, (640, 510), max(2, pulse_r), (255, 71, 87), -1)
                    draw_text_utf8(raw_frame, "ОЖИДАНИЕ ВИДЕОПОТОКА...", (510, 560), font_size=18, color=(140, 150, 160))

                    dummy_fdata = FrameData(frame=frame_idx, time=0.0, joints={})
                    self.frameCaptured.emit(raw_frame, dummy_fdata, 0.0)
                    if self.phone_server:
                        self.phone_server.wait_for_frame(0.040)
                    else:
                        time.sleep(0.04)
                    continue

                raw_frame, cam_fps = res
                proc_frame, fdata = self.tracker.process_frame(
                    raw_frame,
                    frame_idx,
                    t_rel,
                    mirror=self.mirror,
                    draw_skeleton=True,
                    fps=cam_fps if cam_fps > 0 else current_fps,
                )
                badge_color = (16, 185, 129) if cam_id == "cam1" else (6, 182, 212)
                badge_title = "КАМЕРА 1 (ФРОНТ)" if cam_id == "cam1" else "КАМЕРА 2 (СБОКУ)"
                draw_badge_utf8(proc_frame, badge_title, (15, 15), badge_color=badge_color)

            else:
                # PC WEBCAM / IP STREAM
                if cap and cap.isOpened():
                    ok, frame = cap.read()
                    if ok:
                        raw_frame = frame
                    else:
                        time.sleep(0.03)
                        continue
                else:
                    time.sleep(0.05)
                    continue

                proc_frame, fdata = self.tracker.process_frame(
                    raw_frame,
                    frame_idx,
                    t_rel,
                    mirror=self.mirror,
                    draw_skeleton=True,
                    fps=current_fps,
                )

            if self.is_recording and not fdata.tracking_loss:
                self.recorded_frames.append(fdata)

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            self.frameCaptured.emit(proc_frame, fdata, current_fps)
            frame_idx += 1
            if is_phone and self.phone_server:
                self.phone_server.wait_for_frame(0.005)
            else:
                time.sleep(0.005)

        if cap:
            cap.release()

    def start_recording(self):
        self.recorded_frames = []
        self.record_t0 = time.time()
        self.is_recording = True

    def stop_recording(self) -> List[FrameData]:
        self.is_recording = False
        return list(self.recorded_frames)

    def stop(self):
        self.running = False
        self.wait(1000)
        try:
            self.dual_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


class MainWindow(QtWidgets.QMainWindow):
    """Karate Motion Studio Main Application Window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Karate Motion Studio — Захват движений карате Shotokan")
        self.resize(1380, 860)
        self.setMinimumSize(960, 620)
        self.setStyleSheet(DARK_THEME_QSS)

        # Application State
        self.project: KarateMotionProject = KarateMotionProject()
        self.mocap_tracker = MocapTracker(track_hands=True)
        self.retargeter = Retargeter()

        # Source mode: "video" or "webcam" or "idle"
        self.current_source_mode = "idle"
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.webcam_thread: Optional[WebcamCaptureThread] = None

        # Settings
        self.camera_source = "0"
        self.camera_index = 0
        self.mirror_webcam = False
        self.avatar_mirror = False
        self.blender_exe = SettingsDialog._find_blender() or ""

        # Phone Stream Server for wireless smartphone mocap (zero-app install)
        self.phone_server = PhoneStreamServer(port=8088)
        self.phone_server.start()

        self._setup_menus_and_toolbars()
        self._setup_central_layout()
        self._setup_statusbar()
        self._setup_shortcuts()

        # Load synthetic demo project on startup so UI is live immediately!
        self._load_synthetic_demo()

    def _setup_menus_and_toolbars(self):
        # Menu Bar
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("Файл")
        act_new = menu_file.addAction("Новый проект")
        act_new.setShortcut(QtGui.QKeySequence.StandardKey.New)
        act_new.triggered.connect(self.new_project)

        act_open = menu_file.addAction("Открыть проект...")
        act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_project_dialog)

        act_save = menu_file.addAction("Сохранить проект")
        act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_project)

        act_save_as = menu_file.addAction("Сохранить как...")
        act_save_as.triggered.connect(self.save_project_as)

        menu_file.addSeparator()
        act_exit = menu_file.addAction("Выход")
        act_exit.triggered.connect(self.close)

        # Export Menu
        menu_export = menubar.addMenu("Экспорт")
        act_exp_cap = menu_export.addAction("Экспорт Capture JSON...")
        act_exp_cap.triggered.connect(self.export_capture_json_dialog)

        act_exp_retarget = menu_export.addAction("Экспорт Retargeted Animation JSON...")
        act_exp_retarget.triggered.connect(self.export_retargeted_json_dialog)

        act_exp_glb = menu_export.addAction("Экспорт GLB клипа через Blender...")
        act_exp_glb.triggered.connect(self.export_glb_dialog)

        # Settings Menu
        menu_settings = menubar.addMenu("Настройки")
        act_prefs = menu_settings.addAction("Параметры...")
        act_prefs.triggered.connect(self.open_settings_dialog)

        # Main Toolbar
        toolbar = QtWidgets.QToolBar("Главная панель", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Title Label
        lbl_brand = QtWidgets.QLabel("  KARATE MOTION STUDIO  ")
        lbl_brand.setStyleSheet("font-weight: bold; color: #e05260; font-size: 12px; letter-spacing: 1px;")
        toolbar.addWidget(lbl_brand)
        toolbar.addSeparator()

        # Capture Mode Actions
        self.btn_load_video = QtWidgets.QToolButton()
        self.btn_load_video.setText("📁 Загрузить видео...")
        self.btn_load_video.clicked.connect(self.load_video_dialog)
        toolbar.addWidget(self.btn_load_video)

        self.btn_toggle_webcam = QtWidgets.QToolButton()
        self.btn_toggle_webcam.setText("🔴 Веб-камера")
        self.btn_toggle_webcam.setCheckable(True)
        self.btn_toggle_webcam.clicked.connect(self.toggle_webcam)
        toolbar.addWidget(self.btn_toggle_webcam)

        self.btn_camera_manager = QtWidgets.QToolButton()
        self.btn_camera_manager.setText("📱 Камеры / Телефон (QR)")
        self.btn_camera_manager.setStyleSheet("background-color: #2b2d33; color: #2ed573; font-weight: 600;")
        self.btn_camera_manager.clicked.connect(self.open_camera_manager_dialog)
        toolbar.addWidget(self.btn_camera_manager)

        self.combo_cam_source = QtWidgets.QComboBox()
        self.combo_cam_source.setStyleSheet(
            "QComboBox { background-color: #2b2d33; color: #dfdfe0; border: 1px solid #3c404c; border-radius: 3px; padding: 3px 6px; font-size: 8.5pt; }"
            "QComboBox::drop-down { border: none; width: 14px; }"
        )
        self.combo_cam_source.addItem("💻 Веб-камера ПК", "0")
        self.combo_cam_source.addItem("📱 Смартфон 1 (Фронт)", "phone:cam1")
        self.combo_cam_source.addItem("📱 Смартфон 2 (Сбоку)", "phone:cam2")
        self.combo_cam_source.addItem("🔀 Две камеры (Фронт + Сбоку)", "dual:phone+phone")
        self.combo_cam_source.currentIndexChanged.connect(self._on_combo_cam_source_changed)
        toolbar.addWidget(self.combo_cam_source)

        self.btn_toggle_mirror = QtWidgets.QToolButton()
        self.btn_toggle_mirror.setText("🎯 Движения: Прямые (без отзеркаливания)")
        self.btn_toggle_mirror.setCheckable(True)
        self.btn_toggle_mirror.setChecked(self.mirror_webcam)
        self.btn_toggle_mirror.clicked.connect(self._toggle_mirror_mode)
        toolbar.addWidget(self.btn_toggle_mirror)

        self.btn_record = QtWidgets.QToolButton()
        self.btn_record.setText("⏺ Запись")
        self.btn_record.setStyleSheet("background-color: #2b2d33; color: #ff6b6b;")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)
        toolbar.addWidget(self.btn_record)

        toolbar.addSeparator()

        btn_new = QtWidgets.QToolButton()
        btn_new.setText("Новый")
        btn_new.clicked.connect(self.new_project)
        toolbar.addWidget(btn_new)

        btn_open = QtWidgets.QToolButton()
        btn_open.setText("Открыть...")
        btn_open.clicked.connect(self.open_project_dialog)
        toolbar.addWidget(btn_open)

        btn_save = QtWidgets.QToolButton()
        btn_save.setText("Сохранить")
        btn_save.clicked.connect(self.save_project)
        toolbar.addWidget(btn_save)

        btn_export = QtWidgets.QToolButton()
        btn_export.setText("Экспорт JSON ▾")
        menu_exp_btn = QtWidgets.QMenu(self)
        menu_exp_btn.addAction("Capture JSON (Суставы)", self.export_capture_json_dialog)
        menu_exp_btn.addAction("Retargeted Animation JSON (Three.js)", self.export_retargeted_json_dialog)
        menu_exp_btn.addAction("GLB анимационный клип (Blender)", self.export_glb_dialog)
        btn_export.setMenu(menu_exp_btn)
        btn_export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(btn_export)

        toolbar.addSeparator()

        btn_settings = QtWidgets.QToolButton()
        btn_settings.setText("⚙ Настройки")
        btn_settings.clicked.connect(self.open_settings_dialog)
        toolbar.addWidget(btn_settings)

    def _setup_central_layout(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # 1. Technique & Smoothing Sub-bar
        self.metadata_widget = MetadataWidget(self)
        self.metadata_widget.applySmoothingRequested.connect(self.apply_smoothing)
        self.metadata_widget.calibrateRequested.connect(self.calibrate_posture)
        vbox.addWidget(self.metadata_widget)

        # 2. Central Splitter (Left: Video | Right: 3D Avatar)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.video_panel = VideoPanel(self)
        self.viewer_3d = Viewer3DPanel(self)

        self.splitter.addWidget(self.video_panel)
        self.splitter.addWidget(self.viewer_3d)
        self.splitter.setSizes([640, 640])
        vbox.addWidget(self.splitter, stretch=1)

        # 3. Bottom Timeline
        self.timeline = TimelineWidget(self)
        self.timeline.frameChanged.connect(self._on_timeline_frame_changed)
        self.timeline.trimUpdated.connect(self._on_trim_updated)
        vbox.addWidget(self.timeline)

    def _setup_statusbar(self):
        self.status = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status)
        self.status.showMessage("Готово. Выберите видеофайл или включите веб-камеру.")

    def _setup_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self, self.timeline.toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, self.timeline.step_backward)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, self.timeline.step_forward)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Home), self, self.timeline.jump_to_start)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_End), self, self.timeline.jump_to_end)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_BracketLeft), self, self.timeline.set_trim_in)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_BracketRight), self, self.timeline.set_trim_out)
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, self.calibrate_posture)

    # --- DEMO ASSET INITIALIZATION ---
    def _load_synthetic_demo(self):
        """Loads bundled Zenkutsu-dachi demo project so workspace is never empty."""
        demo_json = Path(__file__).resolve().parent.parent / "demo_assets" / "demo_zenkutsu_dachi.json"
        if demo_json.is_file():
            try:
                self.project = KarateMotionProject.load_json(demo_json)
                self._sync_project_to_ui()
                self.status.showMessage("Загружен демонстрационный проект: Дзенкуцу-дати (Ой-дзуки).")
            except Exception as e:
                print(f"Error loading demo project: {e}")

    # --- PROJECT SYNC & TIMELINE NAVIGATION ---
    def _sync_project_to_ui(self):
        frames = self.project.get_effective_frames()
        self.metadata_widget.set_metadata_values(self.project.metadata.technique, self.project.metadata.tags)
        self.timeline.load_project_frames(frames, self.project.metadata.fps, self.project.metadata.trimRange)

    def _on_timeline_frame_changed(self, frame_idx: int):
        """Synchronously updates both Video panel and 3D Avatar."""
        frame_data = self.project.get_frame(frame_idx)
        if not frame_data:
            return

        # 1. Update 3D Avatar
        self.viewer_3d.update_pose(frame_data)

        # 2. Update Video Panel if offline video is loaded
        if self.current_source_mode == "video" and self.video_cap and self.video_cap.isOpened():
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.video_cap.read()
            if ok:
                self.video_panel.update_frame(frame, frame_data)
        elif self.current_source_mode != "webcam":
            # Synthetic or no video: render clean black canvas with skeleton
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            self.video_panel.update_frame(blank, frame_data)

    def _on_trim_updated(self, trim_start: int, trim_end: int):
        self.project.metadata.trimRange = [trim_start, trim_end]
        self.status.showMessage(f"Диапазон движения ограничен: кадры [{trim_start} .. {trim_end}].")

    # --- MODE A: VIDEO FILE ANALYSIS ---
    def load_video_dialog(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите видеофайл с карате", "", "Video Files (*.mp4 *.mov *.avi *.webm *.mkv);;All Files (*.*)"
        )
        if f:
            self.process_video_file(f)

    def process_video_file(self, video_path: str):
        self._stop_webcam_if_running()
        self.current_source_mode = "video"
        self.video_path = video_path

        # Open video for scrubber playback
        if self.video_cap:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(video_path)

        # Progress dialog for asynchronous processing
        progress = QtWidgets.QProgressDialog("Анализ видео MediaPipe Pose 3D...", "Отмена", 0, 100, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(200)

        is_canceled = False

        def on_prog(curr, tot, fps):
            pct = int((curr / float(max(1, tot))) * 100)
            progress.setValue(pct)
            progress.setLabelText(f"Обработка кадра {curr}/{tot} ({fps:.1f} FPS)...")
            QtWidgets.QApplication.processEvents()

        def check_cancel():
            nonlocal is_canceled
            if progress.wasCanceled():
                is_canceled = True
                return True
            return False

        try:
            self.status.showMessage(f"Обработка видео: {Path(video_path).name}...")
            meta = self.metadata_widget.get_metadata_values()
            project = self.mocap_tracker.analyze_video_file(
                video_path=video_path,
                technique_name=meta["technique"],
                stride=1,
                progress_callback=on_prog,
                cancel_check=check_cancel,
            )
            progress.setValue(100)

            if not is_canceled and project.rawFrames:
                self.project = project
                # Auto-retarget
                self.retargeter.retarget_project(self.project)
                self._sync_project_to_ui()
                self.status.showMessage(f"Видео успешно проанализировано: {len(project.rawFrames)} кадров.")
            else:
                self.status.showMessage("Анализ видео отменен.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка анализа видео", str(e))
            self.status.showMessage("Ошибка анализа видео.")

    # --- MODE B: REAL-TIME WEBCAM CAPTURE ---
    def toggle_webcam(self, checked: bool):
        if checked:
            self.start_webcam()
        else:
            self._stop_webcam_if_running()

    def start_webcam(self):
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

        self.current_source_mode = "webcam"
        self.btn_toggle_webcam.setChecked(True)
        self.btn_toggle_webcam.setText("⏹ Выключить камеру")
        self.btn_record.setEnabled(True)

        self.webcam_thread = WebcamCaptureThread(
            camera_source=self.camera_source,
            mirror=self.mirror_webcam,
            track_hands=True,
            phone_server=self.phone_server,
        )
        self.webcam_thread.frameCaptured.connect(self._on_webcam_frame)
        self.webcam_thread.start()
        source_label = "Смартфон" if self.camera_source.startswith("phone:") else ("Две камеры" if self.camera_source.startswith("dual") else "Веб-камера")
        self.status.showMessage(f"Поток запущен ({source_label}). Аватар повторяет движения в реальном времени.")

    def open_camera_manager_dialog(self):
        dlg = CameraManagerDialog(self.phone_server, active_source=self.camera_source, parent=self)
        dlg.cameraSelected.connect(self._on_camera_source_selected)
        dlg.exec()

    def _on_camera_source_selected(self, source: str):
        self.camera_source = str(source)
        # Match combo box index if present
        idx = self.combo_cam_source.findData(self.camera_source)
        if idx >= 0:
            self.combo_cam_source.blockSignals(True)
            self.combo_cam_source.setCurrentIndex(idx)
            self.combo_cam_source.blockSignals(False)
        else:
            self.combo_cam_source.blockSignals(True)
            self.combo_cam_source.addItem(f"🌐 {self.camera_source}", self.camera_source)
            self.combo_cam_source.setCurrentIndex(self.combo_cam_source.count() - 1)
            self.combo_cam_source.blockSignals(False)

        # If currently streaming, restart with new source
        if self.current_source_mode == "webcam":
            self._stop_webcam_if_running()
            self.start_webcam()
        self.status.showMessage(f"Активный источник камеры: {self.camera_source}", 4000)

    def _on_combo_cam_source_changed(self, idx: int):
        data = self.combo_cam_source.itemData(idx)
        if data:
            self._on_camera_source_selected(str(data))

    def _stop_webcam_if_running(self):
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.webcam_thread = None
        self.btn_toggle_webcam.setChecked(False)
        self.btn_toggle_webcam.setText("🔴 Веб-камера")
        self.btn_record.setEnabled(False)
        self.btn_record.setText("⏺ Запись")
        self.current_source_mode = "idle"

    def _on_webcam_frame(self, bgr_frame: np.ndarray, frame_data: FrameData, fps: float):
        if self.current_source_mode != "webcam":
            return
        self.video_panel.set_fps(fps)
        self.video_panel.update_frame(bgr_frame, frame_data, is_pre_rendered=True)
        self.viewer_3d.update_pose(frame_data)

    def _toggle_mirror_mode(self):
        """Unified toggle between Direct (unmirrored) and Mirror (selfie) modes."""
        self.mirror_webcam = not self.mirror_webcam
        self.avatar_mirror = self.mirror_webcam
        txt = "🪞 Движения: Зеркальные (Селфи)" if self.mirror_webcam else "🎯 Движения: Прямые (без отзеркаливания)"
        self.btn_toggle_mirror.setText(txt)
        self.btn_toggle_mirror.setChecked(self.mirror_webcam)
        if self.webcam_thread:
            self.webcam_thread.mirror = self.mirror_webcam
        self.viewer_3d.set_mirror_mode(self.avatar_mirror)
        mode_str = "зеркальный (селфи)" if self.mirror_webcam else "прямой (без отзеркаливания)"
        self.status.showMessage(f"Режим отображения: {mode_str}.", 3000)

    def toggle_recording(self):
        if not self.webcam_thread or not self.webcam_thread.isRunning():
            return

        if not self.webcam_thread.is_recording:
            self.webcam_thread.start_recording()
            self.btn_record.setText("⏹ Стоп (REC)")
            self.btn_record.setStyleSheet("background-color: #e05260; color: #fff; font-weight: bold;")
            self.status.showMessage("Идет запись движения с веб-камеры...")
        else:
            recorded = self.webcam_thread.stop_recording()
            self.btn_record.setText("⏺ Запись")
            self.btn_record.setStyleSheet("background-color: #2b2d33; color: #ff6b6b;")

            if recorded:
                meta_vals = self.metadata_widget.get_metadata_values()
                meta = Metadata(
                    name="webcam_capture",
                    technique=meta_vals["technique"],
                    tags=meta_vals["tags"],
                    sourceType="webcam",
                    fps=30.0,
                    frameCount=len(recorded),
                    trimRange=[0, len(recorded) - 1],
                )
                self.project = KarateMotionProject(metadata=meta, rawFrames=recorded)
                self.retargeter.retarget_project(self.project)
                self._sync_project_to_ui()
                self.status.showMessage(f"Запись завершена: сохранено {len(recorded)} кадров на таймлайн.")
            else:
                self.status.showMessage("Запись пуста (не обнаружен человек).")

    # --- CALIBRATION & FILTERING ---
    def calibrate_posture(self):
        """Calibrates neutral Metsuke gaze posture."""
        self.mocap_tracker.calibrate()
        if self.webcam_thread:
            self.webcam_thread.tracker.calibrate()
        self.viewer_3d.calibrate_head()
        self.status.showMessage("🎯 Нейтральная осанка откалибрована (прямой взгляд вперед).")

    def apply_smoothing(self, min_cutoff: float, beta: float):
        """Applies adaptive 1€ filter to the active project frames."""
        if not self.project.rawFrames:
            QtWidgets.QMessageBox.information(self, "Сглаживание", "Проект не содержит кадров для фильтрации.")
            return

        filt = smooth_project_frames(
            raw_frames=self.project.rawFrames,
            joint_names=STANDARD_JOINTS,
            min_cutoff=min_cutoff,
            beta=beta,
        )
        self.project.filteredFrames = filt
        self.retargeter.retarget_project(self.project)
        self._sync_project_to_ui()
        self.status.showMessage("Адаптивный 1€-фильтр применен (шум удален, кимэ сохранено).")

    # --- PROJECT FILE MANAGEMENT ---
    def new_project(self):
        self._stop_webcam_if_running()
        self.project = KarateMotionProject()
        self._sync_project_to_ui()
        self.viewer_3d.reset_pose()
        self.status.showMessage("Создан новый пустой проект.")

    def open_project_dialog(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Открыть проект Karate Motion Studio", "", "Karate Motion JSON (*.json);;All Files (*.*)"
        )
        if f:
            try:
                self.project = KarateMotionProject.load_json(f)
                self._sync_project_to_ui()
                self.status.showMessage(f"Проект загружен: {Path(f).name}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Ошибка открытия проекта", str(e))

    def save_project(self):
        if not self.project.metadata.sourceFile:
            self.save_project_as()
        else:
            save_path = Path(self.project.metadata.sourceFile)
            if save_path.suffix != ".json":
                save_path = save_path.with_suffix(".json")
            self.project.save_json(save_path)
            self.status.showMessage(f"Проект сохранен: {save_path.name}")

    def save_project_as(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить проект", f"{self.project.metadata.technique}.json", "Karate Motion JSON (*.json)"
        )
        if f:
            meta = self.metadata_widget.get_metadata_values()
            self.project.metadata.technique = meta["technique"]
            self.project.metadata.tags = meta["tags"]
            self.project.metadata.sourceFile = f
            self.project.save_json(f)
            self.status.showMessage(f"Проект сохранен: {Path(f).name}")

    # --- EXPORTERS ---
    def export_capture_json_dialog(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Экспорт Capture JSON (Точки)", f"{self.project.metadata.technique}_capture.json", "JSON (*.json)"
        )
        if f:
            export_capture_file(self.project, f)
            self.status.showMessage(f"Экспортирован Capture JSON: {Path(f).name}")

    def export_retargeted_json_dialog(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Экспорт Retargeted Animation JSON (Three.js)", f"{self.project.metadata.technique}_anim.json", "JSON (*.json)"
        )
        if f:
            if not self.project.retargeting:
                self.retargeter.retarget_project(self.project)
            export_retargeted_file(self.project, f)
            self.status.showMessage(f"Экспортирован Retargeted JSON: {Path(f).name}")

    def export_glb_dialog(self):
        if not self.blender_exe or not Path(self.blender_exe).is_file():
            QtWidgets.QMessageBox.warning(
                self, "Blender не найден", "Укажите путь к blender.exe в Настройках для запекания GLB клипа."
            )
            return

        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Экспорт GLB анимационного клипа", f"{self.project.metadata.technique}.glb", "GLB Model (*.glb)"
        )
        if f:
            try:
                # Save temp project json
                tmp_json = Path(f).with_suffix(".temp.json")
                self.project.save_json(tmp_json)

                self.status.showMessage("Запекание GLB через Blender в фоновом режиме...")
                QtWidgets.QApplication.processEvents()

                export_glb_via_blender(
                    blender_exe=self.blender_exe,
                    project_json_path=tmp_json,
                    output_glb_path=f,
                    action_name=self.project.metadata.technique,
                )
                if tmp_json.is_file():
                    tmp_json.unlink()

                QtWidgets.QMessageBox.information(
                    self, "Экспорт GLB", f"Анимационный клип успешно запечен в:\n{f}\nГотов для Three.js AnimationMixer!"
                )
                self.status.showMessage(f"GLB успешно экспортирован: {Path(f).name}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Ошибка экспорта GLB", str(e))

    def open_settings_dialog(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            cfg = dlg.get_settings()
            self.camera_index = cfg["camera_index"]
            self.camera_source = str(self.camera_index)
            self.mirror_webcam = cfg["mirror"]
            self.blender_exe = cfg["blender_exe"]
            self.status.showMessage("Настройки обновлены.")

    def closeEvent(self, event: QtGui.QCloseEvent):
        self._stop_webcam_if_running()
        if self.video_cap:
            self.video_cap.release()
        if hasattr(self, "phone_server") and self.phone_server:
            self.phone_server.stop()
        event.accept()
