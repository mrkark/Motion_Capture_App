"""
3D Avatar & Skeleton Viewport Panel for Karate Motion Studio.
Embeds hardware-accelerated WebGL Three.js scene inside QtWebEngine,
with support for custom GLB rigs, skeleton layer toggling, and camera controls.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from qt_compat import QtCore, QtGui, QtWidgets, QtWebEngineWidgets, QWebEngineSettings
from core.project_schema import FrameData
from core.retargeter import Retargeter


class Viewer3DPanel(QtWidgets.QWidget):
    """
    Right Panel: 3D Karate Avatar and Skeleton Viewport.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(380, 280)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self.retargeter = Retargeter()
        self._web_view: Optional[QtWebEngineWidgets.QWebEngineView] = None
        self._is_web_ready = False
        self._pending_pose: Optional[Dict[str, Any]] = None

        self._setup_ui()
        self._init_webengine()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header controls
        self.header = QtWidgets.QWidget(self)
        self.header.setStyleSheet("background-color: #23252a; border-bottom: 1px solid #33363f; padding: 4px;")
        h_layout = QtWidgets.QHBoxLayout(self.header)
        h_layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_title = QtWidgets.QLabel("3D Аватар & Риг (Three.js)")
        self.lbl_title.setStyleSheet("color: #dfdfe0; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(self.lbl_title)

        h_layout.addStretch()

        # View Mode Selector
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["Аватар + Скелет", "Только аватар", "Только скелет"])
        self.cmb_mode.setStyleSheet(
            "background-color: #2b2d33; color: #dfdfe0; font-size: 11px; border: 1px solid #3e424c; padding: 2px 6px;"
        )
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        h_layout.addWidget(self.cmb_mode)

        # Load Custom GLB button
        self.btn_load_glb = QtWidgets.QPushButton("Загрузить GLB...")
        self.btn_load_glb.setStyleSheet(
            "background-color: #383b44; color: #dfdfe0; font-size: 11px; border: 1px solid #4a4f5c; padding: 2px 8px; border-radius: 3px;"
        )
        self.btn_load_glb.clicked.connect(self._on_load_custom_glb)
        h_layout.addWidget(self.btn_load_glb)

        # Reset T-Pose button
        self.btn_tpose = QtWidgets.QPushButton("T-Pose")
        self.btn_tpose.setStyleSheet(
            "background-color: #383b44; color: #dfdfe0; font-size: 11px; border: 1px solid #4a4f5c; padding: 2px 8px; border-radius: 3px;"
        )
        self.btn_tpose.clicked.connect(self.reset_pose)
        h_layout.addWidget(self.btn_tpose)

        layout.addWidget(self.header)

        # 3D Container
        self.container_widget = QtWidgets.QWidget()
        self.container_layout = QtWidgets.QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.container_widget, stretch=1)

    def _init_webengine(self):
        try:
            self._web_view = QtWebEngineWidgets.QWebEngineView(self)
            self._web_view.setStyleSheet("background-color: #16171a;")

            # Configure Chromium WebGL and local file access permissions
            settings = self._web_view.settings()
            if hasattr(QWebEngineSettings.WebAttribute, "LocalContentCanAccessFileUrls"):
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            if hasattr(QWebEngineSettings.WebAttribute, "LocalContentCanAccessRemoteUrls"):
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            if hasattr(QWebEngineSettings.WebAttribute, "WebGLEnabled"):
                settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            if hasattr(QWebEngineSettings.WebAttribute, "Accelerated2dCanvasEnabled"):
                settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)

            self.container_layout.addWidget(self._web_view)

            # Load HTML template from demo_assets
            template_path = Path(__file__).resolve().parent.parent / "demo_assets" / "viewer_template.html"
            if template_path.is_file():
                self._web_view.load(QtCore.QUrl.fromLocalFile(str(template_path)))
                self._web_view.loadFinished.connect(self._on_web_loaded)
            else:
                self._show_fallback_label(f"HTML template not found: {template_path}")
        except Exception as e:
            self._show_fallback_label(f"WebEngine initialization failed: {e}")

    def _show_fallback_label(self, msg: str):
        lbl = QtWidgets.QLabel(f"3D Viewport Fallback:\n{msg}")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #e05260; background: #16171a; font-family: monospace;")
        self.container_layout.addWidget(lbl)

    def _on_web_loaded(self, success: bool):
        self._is_web_ready = success
        if success and self._pending_pose:
            self.update_pose(self._pending_pose)
            self._pending_pose = None

    def _on_mode_changed(self, index: int):
        if not self._is_web_ready or not self._web_view:
            return
        mode_str = "both" if index == 0 else ("avatar" if index == 1 else "skeleton")
        self._web_view.page().runJavaScript(f"setViewMode('{mode_str}');")

    def _on_load_custom_glb(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите 3D-модель Каратиста с ригом", "", "GLB 3D Models (*.glb);;All Files (*.*)"
        )
        if file_path:
            self.load_custom_glb(file_path)

    def load_custom_glb(self, glb_path: str):
        """Pass local GLB file URL to Three.js GLTFLoader."""
        if not self._is_web_ready or not self._web_view:
            return
        qurl = QtCore.QUrl.fromLocalFile(str(Path(glb_path).resolve()))
        url_str = qurl.toString()
        self._web_view.page().runJavaScript(f"loadModel('{url_str}');")

    def reset_pose(self):
        """Reset avatar and skeleton to rest T-pose."""
        if self._is_web_ready and self._web_view:
            self._web_view.page().runJavaScript("resetPose();")

    def calibrate_head(self):
        """Calibrate head Metsuke gaze level."""
        if self._is_web_ready and self._web_view:
            self._web_view.page().runJavaScript("if (window.calibrateHead) window.calibrateHead();")

    def set_camera_preset(self, preset: str):
        """Presets: 'front', '34', 'side', 'back'."""
        if self._is_web_ready and self._web_view:
            self._web_view.page().runJavaScript(f"setCamera('{preset}');")

    def set_mirror_mode(self, enabled: bool):
        """Toggle avatar mirroring in Three.js."""
        if self._is_web_ready and self._web_view:
            val_str = "true" if enabled else "false"
            self._web_view.page().runJavaScript(f"setMirrorAvatar({val_str});")

    def update_pose(self, frame_data: FrameData | Dict[str, Any]):
        """Update 3D avatar and neutral skeleton joints in real-time."""
        if isinstance(frame_data, dict):
            data_dict = dict(frame_data)
        else:
            # Solve local bone rotations immediately using Retargeter for instantaneous avatar mirroring!
            bones = self.retargeter.retarget_frame(frame_data)
            data_dict = {
                "joints": frame_data.joints,
                "confidence": frame_data.confidence,
                "head": frame_data.head,
                "fingers": frame_data.fingers,
                "rootPosition": frame_data.rootPosition,
                "bones": bones,
            }

        if not self._is_web_ready or not self._web_view:
            self._pending_pose = data_dict
            return

        json_str = json.dumps(data_dict)
        self._web_view.page().runJavaScript(f"window.updatePose({json_str});")

