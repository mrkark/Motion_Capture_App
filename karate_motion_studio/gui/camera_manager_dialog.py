"""
Camera and Phone Manager Dialog for Karate Motion Studio.
Allows connecting smartphones via QR-code / local Wi-Fi, entering IP camera URLs,
and configuring multi-camera capture (Front + Side views).
"""
from __future__ import annotations
import os
import sys
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import cv2
import numpy as np

from qt_compat import QtCore, QtGui, QtWidgets
from core.phone_stream_server import PhoneStreamServer


class CameraManagerDialog(QtWidgets.QDialog):
    """Interactive Dialog for managing Webcams, Phone Cameras, and Multi-Camera setups."""

    cameraSelected = QtCore.pyqtSignal(str)  # source identifier e.g. "0", "phone:cam1", "phone:cam2", "dual", "http://..."

    def __init__(self, phone_server: PhoneStreamServer, active_source: str = "0", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.phone_server = phone_server
        self.active_source = active_source
        self.setWindowTitle("🎥 Настройка камер и подключение телефона — Karate Motion Studio")
        self.resize(720, 560)
        self.setMinimumSize(600, 480)

        # Polling timer for live status updates
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self._refresh_connected_devices)
        self.poll_timer.start(1000)

        self._setup_ui()
        self._refresh_connected_devices()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header Title
        lbl_title = QtWidgets.QLabel("📱 Подключение камер и смартфонов")
        lbl_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #ff4757;")
        layout.addWidget(lbl_title)

        lbl_desc = QtWidgets.QLabel(
            "Для точного захвата движений и глубоких стоек карате (Дзенкуцу-дати, Киба-дати) "
            "рекомендуется использовать смартфон на штативе во весь рост или 2 камеры (спереди и сбоку)."
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #a4b0be; font-size: 9.5pt;")
        layout.addWidget(lbl_desc)

        # Tab Widget
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #3c404c; background: #1f2228; border-radius: 6px; }
            QTabBar::tab { background: #262a33; color: #c8d1dc; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: 600; }
            QTabBar::tab:selected { background: #323742; color: #ffffff; border-bottom: 2px solid #ff4757; }
        """)

        # Tab 1: Phone QR Code (Zero install)
        tab_qr = QtWidgets.QWidget()
        self._setup_tab_qr(tab_qr)
        tabs.addTab(tab_qr, "📱 Смартфон (QR-код)")

        # Tab 2: IP Webcam / DroidCam
        tab_ip = QtWidgets.QWidget()
        self._setup_tab_ip(tab_ip)
        tabs.addTab(tab_ip, "🌐 IP-камера / RTSP / DroidCam")

        # Tab 3: PC Webcams
        tab_pc = QtWidgets.QWidget()
        self._setup_tab_pc(tab_pc)
        tabs.addTab(tab_pc, "💻 Веб-камеры ПК")

        layout.addWidget(tabs)

        # Active Source Selection Box
        box_active = QtWidgets.QGroupBox("🎯 Выбор активного источника для захвата движения")
        box_active.setStyleSheet("QGroupBox { font-weight: bold; color: #ff6b81; border: 1px solid #3c404c; border-radius: 6px; margin-top: 10px; padding-top: 14px; }")
        layout_active = QtWidgets.QHBoxLayout(box_active)

        self.btn_use_phone1 = QtWidgets.QRadioButton("📱 Телефон 1 (Фронт)")
        self.btn_use_phone2 = QtWidgets.QRadioButton("📱 Телефон 2 (Сбоку)")
        self.btn_use_dual = QtWidgets.QRadioButton("🔀 Две камеры (Сплит)")
        self.btn_use_pc = QtWidgets.QRadioButton("💻 Веб-камера ПК")
        self.btn_use_ip = QtWidgets.QRadioButton("🌐 IP-поток")

        # Select matching radio
        if self.active_source == "phone:cam1":
            self.btn_use_phone1.setChecked(True)
        elif self.active_source == "phone:cam2":
            self.btn_use_phone2.setChecked(True)
        elif self.active_source.startswith("dual"):
            self.btn_use_dual.setChecked(True)
        elif self.active_source.startswith("http") or self.active_source.startswith("rtsp"):
            self.btn_use_ip.setChecked(True)
        else:
            self.btn_use_pc.setChecked(True)

        layout_active.addWidget(self.btn_use_phone1)
        layout_active.addWidget(self.btn_use_phone2)
        layout_active.addWidget(self.btn_use_dual)
        layout_active.addWidget(self.btn_use_pc)
        layout_active.addWidget(self.btn_use_ip)
        layout.addWidget(box_active)

        # Dialog Buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _setup_tab_qr(self, parent: QtWidgets.QWidget):
        layout = QtWidgets.QHBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(18)

        # Left: QR Code Display
        qr_container = QtWidgets.QVBoxLayout()
        self.lbl_qr_image = QtWidgets.QLabel()
        self.lbl_qr_image.setFixedSize(180, 180)
        self.lbl_qr_image.setStyleSheet("background: white; border-radius: 8px; padding: 6px;")
        self.lbl_qr_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        qr_container.addWidget(self.lbl_qr_image)

        lbl_qr_hint = QtWidgets.QLabel("Наведите камеру смартфона")
        lbl_qr_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_qr_hint.setStyleSheet("font-size: 8.5pt; color: #8c939e;")
        qr_container.addWidget(lbl_qr_hint)
        layout.addLayout(qr_container)

        # Right: Instructions & Live Devices
        info_col = QtWidgets.QVBoxLayout()
        info_col.setSpacing(8)

        lbl_step1 = QtWidgets.QLabel("1. Убедитесь, что телефон подключен к <b>тому же Wi-Fi</b>.")
        info_col.addWidget(lbl_step1)

        url = self.phone_server.get_connection_url()
        self.lbl_url = QtWidgets.QLabel(f"2. Или введите в браузере телефона: <a href='{url}' style='color: #2ed573; font-weight: bold;'>{url}</a>")
        self.lbl_url.setOpenExternalLinks(True)
        info_col.addWidget(self.lbl_url)

        lbl_step3 = QtWidgets.QLabel("3. Нажмите <b>«Начать эфир»</b> на экране телефона.")
        info_col.addWidget(lbl_step3)

        # Security helper box for mobile HTTPS certificates
        box_sec = QtWidgets.QFrame()
        box_sec.setStyleSheet("background-color: #232731; border: 1px solid #ff4757; border-radius: 6px; padding: 6px;")
        sec_layout = QtWidgets.QVBoxLayout(box_sec)
        sec_layout.setContentsMargins(8, 6, 8, 6)
        sec_layout.setSpacing(3)
        lbl_sec_t = QtWidgets.QLabel("<b>🔒 Доступ к камере в мобильном браузере:</b>")
        lbl_sec_t.setStyleSheet("color: #ff6b81; font-size: 8.5pt;")
        sec_layout.addWidget(lbl_sec_t)
        lbl_sec_b = QtWidgets.QLabel(
            "• Для доступа к камере используется защищенный <b>HTTPS</b>.<br>"
            "• При предупреждении о сертификате нажмите <b>«Дополнительно» → «Перейти на сайт»</b><br>"
            "  (на iPhone в Safari: <b>«Показать подробности» → «Посетить этот веб-сайт»</b>).<br>"
            "• Во всплывающем окне нажмите <b>«Разрешить»</b> доступ к камере."
        )
        lbl_sec_b.setStyleSheet("color: #ced6e0; font-size: 8pt;")
        sec_layout.addWidget(lbl_sec_b)
        info_col.addWidget(box_sec)

        info_col.addSpacing(6)
        lbl_status_head = QtWidgets.QLabel("<b>Статус подключенных телефонов:</b>")
        info_col.addWidget(lbl_status_head)

        self.lbl_phone1_status = QtWidgets.QLabel("• 📱 Камера 1 (Фронт): ⚪ Ожидание подключения...")
        self.lbl_phone2_status = QtWidgets.QLabel("• 📱 Камера 2 (Сбоку): ⚪ Ожидание подключения...")
        info_col.addWidget(self.lbl_phone1_status)
        info_col.addWidget(self.lbl_phone2_status)

        info_col.addStretch()
        layout.addLayout(info_col)

        # Generate QR code Pixmap
        self._update_qr_image()

    def _setup_tab_ip(self, parent: QtWidgets.QWidget):
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        lbl_info = QtWidgets.QLabel(
            "Подключение внешних приложений (IP Webcam, DroidCam, EpocCam, OBS Virtual Cam, RTSP):\n"
            "Примеры URL:\n"
            "• IP Webcam (Android): http://192.168.1.50:8080/video\n"
            "• DroidCam: http://192.168.1.50:4747/video\n"
            "• RTSP-поток: rtsp://192.168.1.50:8554/live"
        )
        lbl_info.setStyleSheet("color: #a4b0be; font-size: 9pt;")
        layout.addWidget(lbl_info)

        row_url = QtWidgets.QHBoxLayout()
        self.edit_ip_url = QtWidgets.QLineEdit()
        self.edit_ip_url.setPlaceholderText("http://192.168.1.X:8080/video")
        self.edit_ip_url.setText(self.active_source if self.active_source.startswith("http") or self.active_source.startswith("rtsp") else "")
        self.edit_ip_url.setStyleSheet("padding: 6px; font-size: 10pt;")
        row_url.addWidget(self.edit_ip_url)

        btn_test = QtWidgets.QPushButton("🔍 Проверить поток")
        btn_test.clicked.connect(self._test_ip_stream)
        row_url.addWidget(btn_test)
        layout.addLayout(row_url)

        self.lbl_test_result = QtWidgets.QLabel("")
        layout.addWidget(self.lbl_test_result)
        layout.addStretch()

    def _setup_tab_pc(self, parent: QtWidgets.QWidget):
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        lbl_info = QtWidgets.QLabel("Выбор встроенной или USB веб-камеры компьютера:")
        layout.addWidget(lbl_info)

        self.combo_pc_cam = QtWidgets.QComboBox()
        for i in range(4):
            self.combo_pc_cam.addItem(f"Камера #{i} (USB / Встроенная)", i)
        try:
            curr_idx = int(self.active_source)
            self.combo_pc_cam.setCurrentIndex(curr_idx)
        except ValueError:
            self.combo_pc_cam.setCurrentIndex(0)

        self.combo_pc_cam.setStyleSheet("padding: 6px; font-size: 10pt;")
        layout.addWidget(self.combo_pc_cam)
        layout.addStretch()

    def _update_qr_image(self):
        try:
            qr_bytes = self.phone_server.generate_qr_png_bytes()
            qimg = QtGui.QImage.fromData(qr_bytes)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                168, 168, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_qr_image.setPixmap(pix)
        except Exception as e:
            self.lbl_qr_image.setText(f"QR Error: {e}")

    def _refresh_connected_devices(self):
        cams = self.phone_server.get_connected_cameras()

        if "cam1" in cams:
            frame, fps = self.phone_server.get_latest_frame("cam1") or (None, 30.0)
            res_str = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "HD"
            self.lbl_phone1_status.setText(f"• 📱 Камера 1 (Фронт): <span style='color: #2ed573; font-weight: bold;'>🟢 В эфире ({fps:.1f} FPS, {res_str})</span>")
        else:
            self.lbl_phone1_status.setText("• 📱 Камера 1 (Фронт): <span style='color: #8c939e;'>⚪ Ожидание подключения...</span>")

        if "cam2" in cams:
            frame, fps = self.phone_server.get_latest_frame("cam2") or (None, 30.0)
            res_str = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "HD"
            self.lbl_phone2_status.setText(f"• 📱 Камера 2 (Сбоку): <span style='color: #2ed573; font-weight: bold;'>🟢 В эфире ({fps:.1f} FPS, {res_str})</span>")
        else:
            self.lbl_phone2_status.setText("• 📱 Камера 2 (Сбоку): <span style='color: #8c939e;'>⚪ Ожидание подключения...</span>")

    def _test_ip_stream(self):
        url = self.edit_ip_url.text().strip()
        if not url:
            self.lbl_test_result.setText("<span style='color: #ff4757;'>Введите корректный URL потока</span>")
            return

        self.lbl_test_result.setText("Подключение к потоку...")
        QtWidgets.QApplication.processEvents()

        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                self.lbl_test_result.setText(f"<span style='color: #2ed573; font-weight: bold;'>Успешно! Поток доступен ({w}x{h}).</span>")
                return

        self.lbl_test_result.setText("<span style='color: #ff4757;'>Не удалось подключиться к видеопотоку. Проверьте IP и порт.</span>")

    def _on_accept(self):
        if self.btn_use_phone1.isChecked():
            chosen = "phone:cam1"
        elif self.btn_use_phone2.isChecked():
            chosen = "phone:cam2"
        elif self.btn_use_dual.isChecked():
            chosen = "dual:phone+phone"
        elif self.btn_use_ip.isChecked():
            chosen = self.edit_ip_url.text().strip() or "phone:cam1"
        else:
            chosen = str(self.combo_pc_cam.currentData())

        self.active_source = chosen
        self.cameraSelected.emit(chosen)
        self.accept()

    def get_selected_source(self) -> str:
        return self.active_source
