"""
Settings Dialog for Karate Motion Studio.
Configures camera device, capture parameters, hand tracking, and Blender executable path.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional

from qt_compat import QtCore, QtGui, QtWidgets


class SettingsDialog(QtWidgets.QDialog):
    """Application Settings Dialog with dark styling."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Настройки — Karate Motion Studio")
        self.setFixedSize(460, 340)
        self.setStyleSheet("""
            QDialog { background-color: #1e1f23; color: #dfdfe0; }
            QLabel { color: #b0b5be; font-size: 11px; }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #2b2d33; color: #dfdfe0; border: 1px solid #3e424c;
                border-radius: 3px; padding: 4px; font-size: 11px;
            }
            QCheckBox { color: #dfdfe0; font-size: 11px; }
            QPushButton {
                background-color: #383b44; color: #dfdfe0; border: 1px solid #4a4f5c;
                border-radius: 3px; padding: 5px 14px; font-size: 11px;
            }
            QPushButton:hover { background-color: #4a4f5c; }
        """)

        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)

        # Camera Index
        self.spn_camera = QtWidgets.QSpinBox()
        self.spn_camera.setRange(0, 8)
        self.spn_camera.setValue(0)
        form.addRow("Индекс веб-камеры:", self.spn_camera)

        # Mirror mode
        self.chk_mirror = QtWidgets.QCheckBox("Зеркальный режим для веб-камеры")
        self.chk_mirror.setChecked(True)
        form.addRow("Зеркало:", self.chk_mirror)

        # Track Hands
        self.chk_hands = QtWidgets.QCheckBox("Отслеживание пальцев рук (MediaPipe Hands)")
        self.chk_hands.setChecked(True)
        form.addRow("Пальцы:", self.chk_hands)

        # Blender Executable Path
        blender_detected = self._find_blender() or ""
        self.txt_blender = QtWidgets.QLineEdit(blender_detected)
        btn_browse_blender = QtWidgets.QPushButton("Обзор...")
        btn_browse_blender.clicked.connect(self._browse_blender)

        blender_layout = QtWidgets.QHBoxLayout()
        blender_layout.addWidget(self.txt_blender)
        blender_layout.addWidget(btn_browse_blender)
        form.addRow("Blender.exe (для GLB):", blender_layout)

        layout.addLayout(form)
        layout.addStretch()

        # Dialog Buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _browse_blender(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Укажите путь к blender.exe", "", "Blender (blender.exe);;All Files (*.*)")
        if f:
            self.txt_blender.setText(f)

    @staticmethod
    def _find_blender() -> Optional[str]:
        p = shutil.which("blender")
        if p:
            return p
        candidates = [
            r"D:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        ]
        for c in candidates:
            if Path(c).is_file():
                return c
        return None

    def get_settings(self) -> dict:
        return {
            "camera_index": self.spn_camera.value(),
            "mirror": self.chk_mirror.isChecked(),
            "track_hands": self.chk_hands.isChecked(),
            "blender_exe": self.txt_blender.text().strip(),
        }
