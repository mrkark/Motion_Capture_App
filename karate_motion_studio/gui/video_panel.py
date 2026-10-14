"""
Source Video & Webcam Display Panel with Skeleton Overlay.
Renders video frames with customizable, high-contrast karate skeleton overlays.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

from qt_compat import QtCore, QtGui, QtWidgets
from core.project_schema import STANDARD_CONNECTIONS, FrameData


class VideoPanel(QtWidgets.QWidget):
    """
    Displays the source video or live webcam image with an accurate skeleton overlay.
    """

    frameSeekRequested = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        # Image state
        self._current_qimage: Optional[QtGui.QImage] = None
        self._current_frame_data: Optional[FrameData] = None
        self._frame_width = 640
        self._frame_height = 480

        # Overlay settings
        self.show_skeleton = True
        self.joint_radius = 5
        self.bone_width = 3
        self.min_confidence = 0.20

        # Colors (RGB)
        self.color_torso = QtGui.QColor(224, 82, 96)     # Crimson
        self.color_left = QtGui.QColor(0, 210, 211)      # Cyan
        self.color_right = QtGui.QColor(255, 159, 67)    # Amber
        self.color_head = QtGui.QColor(240, 240, 245)    # White
        self.color_low_conf = QtGui.QColor(100, 100, 110, 120)

        # UI Controls Bar (Inside Panel)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        self.header = QtWidgets.QWidget(self)
        self.header.setStyleSheet("background-color: #23252a; border-bottom: 1px solid #33363f; padding: 4px;")
        h_layout = QtWidgets.QHBoxLayout(self.header)
        h_layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_title = QtWidgets.QLabel("Исходное движение (Видео / Веб-камера)")
        self.lbl_title.setStyleSheet("color: #dfdfe0; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(self.lbl_title)

        h_layout.addStretch()

        self.chk_skeleton = QtWidgets.QCheckBox("Скелет")
        self.chk_skeleton.setChecked(True)
        self.chk_skeleton.setStyleSheet("color: #b0b5be; font-size: 11px;")
        self.chk_skeleton.toggled.connect(self._toggle_skeleton)
        h_layout.addWidget(self.chk_skeleton)

        self.lbl_conf = QtWidgets.QLabel("Порог:")
        self.lbl_conf.setStyleSheet("color: #8c939e; font-size: 11px; margin-left: 8px;")
        h_layout.addWidget(self.lbl_conf)

        self.slider_conf = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_conf.setRange(5, 80)
        self.slider_conf.setValue(int(self.min_confidence * 100))
        self.slider_conf.setFixedWidth(70)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        h_layout.addWidget(self.slider_conf)

        self.lbl_fps = QtWidgets.QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color: #38ef7d; font-family: monospace; font-size: 11px; margin-left: 8px;")
        h_layout.addWidget(self.lbl_fps)

        layout.addWidget(self.header)

        # Viewport Area
        self.viewport = QtWidgets.QLabel()
        self.viewport.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.viewport.setStyleSheet("background-color: #121315;")
        self.viewport.setText("Нет видеопотока\n(Загрузите видео или включите веб-камеру)")
        layout.addWidget(self.viewport, stretch=1)

    def _toggle_skeleton(self, checked: bool):
        self.show_skeleton = checked
        self.update()

    def _on_conf_changed(self, val: int):
        self.min_confidence = val / 100.0
        self.update()

    def set_fps(self, fps: float):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def update_frame(self, frame_bgr: np.ndarray, frame_data: Optional[FrameData] = None, is_pre_rendered: bool = False):
        """Update viewport with a new OpenCV BGR frame and its landmark data."""
        h, w = frame_bgr.shape[:2]
        self._frame_width = w
        self._frame_height = h
        self._current_frame_data = frame_data

        # Convert to QImage
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()

        # If not already pre-rendered by OpenCV mocap tracker, draw skeleton overlay onto QImage
        if not is_pre_rendered and self.show_skeleton and frame_data and frame_data.joints:
            self._draw_skeleton_on_image(qimg, frame_data)

        self._current_qimage = qimg
        self._update_display()

    def _draw_skeleton_on_image(self, qimg: QtGui.QImage, frame_data: FrameData):
        painter = QtGui.QPainter(qimg)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = qimg.width()
        h = qimg.height()
        pts = frame_data.joints
        conf = frame_data.confidence

        # Exact 2D screen coordinates from MediaPipe (pixel-perfect overlay)
        coords_2d: Dict[str, QtCore.QPointF] = {}
        if getattr(frame_data, "landmarks_2d", None):
            for name, pt in frame_data.landmarks_2d.items():
                c = conf.get(name, 1.0)
                if c >= self.min_confidence:
                    px = pt[0] * w
                    py = pt[1] * h
                    coords_2d[name] = QtCore.QPointF(px, py)
        else:
            # Fallback approximate 2D projection
            for name, p3d in pts.items():
                c = conf.get(name, 1.0)
                if c >= self.min_confidence:
                    px = (p3d[0] + 0.6) / 1.2 * w
                    py = (1.1 - p3d[1]) / 1.5 * h
                    coords_2d[name] = QtCore.QPointF(px, py)

        # High-contrast palette matching motion_engine.py
        color_torso = QtGui.QColor(100, 100, 255)
        color_arm_l = QtGui.QColor(255, 220, 0)
        color_arm_r = QtGui.QColor(0, 150, 255)
        color_leg_l = QtGui.QColor(255, 100, 180)
        color_leg_r = QtGui.QColor(180, 100, 255)
        color_neck  = QtGui.QColor(200, 200, 200)

        # Draw bones
        for j_from, j_to, side in STANDARD_CONNECTIONS:
            if j_from in coords_2d and j_to in coords_2d:
                color = color_torso
                if side == "left":
                    color = color_arm_l if "elbow" in j_from or "wrist" in j_to else color_leg_l
                elif side == "right":
                    color = color_arm_r if "elbow" in j_from or "wrist" in j_to else color_leg_r
                elif side == "head":
                    color = color_neck

                pen = QtGui.QPen(color, 3, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(coords_2d[j_from], coords_2d[j_to])

        # Draw dual-ring joint spheres with black outline
        for name, pt in coords_2d.items():
            is_hl = ("wrist" in name or "ankle" in name)
            inner_color = QtGui.QColor(255, 255, 0) if is_hl else QtGui.QColor(255, 255, 255)

            painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 1))
            painter.setBrush(QtGui.QBrush(inner_color))
            painter.drawEllipse(pt, 5, 5)

        # Draw 2D hands if present
        hands_2d = getattr(frame_data, "hands_2d", {})
        if hands_2d:
            hand_conn = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]
            for side, hpts in hands_2d.items():
                h_color = QtGui.QColor(200, 255, 0) if side == "left" else QtGui.QColor(0, 200, 255)
                h_qpts = [QtCore.QPointF(p[0] * w, p[1] * h) for p in hpts]
                pen_h = QtGui.QPen(h_color, 2, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen_h)
                for a, b in hand_conn:
                    if 0 <= a < len(h_qpts) and 0 <= b < len(h_qpts):
                        painter.drawLine(h_qpts[a], h_qpts[b])
                for hp in h_qpts:
                    painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 1))
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
                    painter.drawEllipse(hp, 3, 3)

        painter.end()

    def _update_display(self):
        if self._current_qimage is None:
            return
        scaled = self._current_qimage.scaled(
            self.viewport.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.viewport.setPixmap(QtGui.QPixmap.fromImage(scaled))

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._update_display()
