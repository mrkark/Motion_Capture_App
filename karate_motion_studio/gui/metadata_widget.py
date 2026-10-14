"""
Technique Metadata & Smoothing Controller Widget for Karate Motion Studio.
Allows users to label Karate Shotokan techniques, stances, tags, and adjust 1€ smoothing filters.
"""
from __future__ import annotations
from typing import Optional, List, Callable

from qt_compat import QtCore, QtGui, QtWidgets
from core.project_schema import SHOTOKAN_TECHNIQUE_PRESETS


class MetadataWidget(QtWidgets.QWidget):
    """
    Sidebar / Tool dock for editing Karate motion metadata and adjusting filters.
    """

    applySmoothingRequested = QtCore.pyqtSignal(float, float)  # min_cutoff, beta
    calibrateRequested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet("background-color: #23252a; border-bottom: 1px solid #2d3038; padding: 4px;")

        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(10)

        # Technique Preset
        lbl_tech = QtWidgets.QLabel("Техника:")
        lbl_tech.setStyleSheet("color: #8c939e; font-size: 11px;")
        layout.addWidget(lbl_tech)

        self.cmb_technique = QtWidgets.QComboBox()
        self.cmb_technique.setEditable(True)
        self.cmb_technique.addItems(SHOTOKAN_TECHNIQUE_PRESETS)
        self.cmb_technique.setStyleSheet(
            "background-color: #2b2d33; color: #dfdfe0; font-size: 11px; border: 1px solid #3e424c; padding: 2px 6px; min-width: 140px;"
        )
        layout.addWidget(self.cmb_technique)

        # Tags
        lbl_tags = QtWidgets.QLabel("Теги:")
        lbl_tags.setStyleSheet("color: #8c939e; font-size: 11px;")
        layout.addWidget(lbl_tags)

        self.txt_tags = QtWidgets.QLineEdit("shotokan, block, zenkutsu_dachi")
        self.txt_tags.setStyleSheet(
            "background-color: #2b2d33; color: #dfdfe0; font-size: 11px; border: 1px solid #3e424c; padding: 2px 6px; min-width: 160px;"
        )
        layout.addWidget(self.txt_tags)

        # Smoothing Slider
        lbl_smooth = QtWidgets.QLabel("Сглаживание (1€):")
        lbl_smooth.setStyleSheet("color: #8c939e; font-size: 11px; margin-left: 8px;")
        layout.addWidget(lbl_smooth)

        self.slider_smooth = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_smooth.setRange(10, 100)
        self.slider_smooth.setValue(45)
        self.slider_smooth.setFixedWidth(80)
        layout.addWidget(self.slider_smooth)

        self.btn_apply_filter = QtWidgets.QPushButton("Сгладить")
        self.btn_apply_filter.setToolTip("Применить адаптивный 1€-фильтр к проекту")
        self.btn_apply_filter.setStyleSheet(
            "background-color: #383b44; color: #dfdfe0; font-size: 11px; border: 1px solid #4a4f5c; border-radius: 3px; padding: 2px 8px;"
        )
        self.btn_apply_filter.clicked.connect(self._on_apply_filter)
        layout.addWidget(self.btn_apply_filter)

        layout.addStretch()

        # Calibration Button (Metsuke Neutral Baseline)
        self.btn_calibrate = QtWidgets.QPushButton("🎯 Калибровать (C)")
        self.btn_calibrate.setToolTip("Калибровать нейтральную осанку взгляда вперед (Мэцукэ) [C]")
        self.btn_calibrate.setStyleSheet(
            "background-color: rgba(46, 213, 115, 0.85); color: #fff; font-weight: bold; font-size: 11px; border-radius: 3px; padding: 4px 10px;"
        )
        self.btn_calibrate.clicked.connect(self.calibrateRequested.emit)
        layout.addWidget(self.btn_calibrate)

    def _on_apply_filter(self):
        val = self.slider_smooth.value() / 100.0
        # Low cutoff: 0.5 to 2.5, beta: 0.02 to 0.15
        min_cutoff = 0.5 + (1.0 - val) * 2.0
        beta = 0.02 + val * 0.10
        self.applySmoothingRequested.emit(min_cutoff, beta)

    def get_metadata_values(self) -> dict:
        tags_raw = self.txt_tags.text()
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
        return {
            "technique": self.cmb_technique.currentText().strip() or "karate_technique",
            "tags": tags_list,
        }

    def set_metadata_values(self, technique: str, tags: List[str]):
        idx = self.cmb_technique.findText(technique)
        if idx >= 0:
            self.cmb_technique.setCurrentIndex(idx)
        else:
            self.cmb_technique.setCurrentText(technique)
        self.txt_tags.setText(", ".join(tags))
