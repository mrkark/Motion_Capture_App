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
            "Встроенная веб-камера ноутбука отключена для обеспечения высокой точности спортивного захвата движений. "
            "Для записи стоек (Дзенкуцу, Киба-дати) и ударов карате подключите 1 или 2 смартфона во весь рост по Wi-Fi."
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
        tabs.addTab(tab_qr, "📱 Смартфоны (QR-код)")

        # Tab 2: IP Webcam / DroidCam
        tab_ip = QtWidgets.QWidget()
        self._setup_tab_ip(tab_ip)
        tabs.addTab(tab_ip, "🌐 IP-камера смартфона / RTSP")

        layout.addWidget(tabs)

        # Active Source Selection Box
        box_active = QtWidgets.QGroupBox("🎯 Выбор активного источника для захвата движения")
        box_active.setStyleSheet("QGroupBox { font-weight: bold; color: #ff6b81; border: 1px solid #3c404c; border-radius: 6px; margin-top: 10px; padding-top: 14px; }")
        layout_active = QtWidgets.QHBoxLayout(box_active)

        self.btn_use_phone1 = QtWidgets.QRadioButton("📱 Телефон 1 (Фронт — лицо/корпус)")
        self.btn_use_phone2 = QtWidgets.QRadioButton("📱 Телефон 2 (Сбоку — профиль/глубина)")
        self.btn_use_dual = QtWidgets.QRadioButton("🔀 Две камеры смартфонов (Синхронный 3D)")
        self.btn_use_ip = QtWidgets.QRadioButton("🌐 IP-поток")

        # Select matching radio
        if self.active_source == "phone:cam2":
            self.btn_use_phone2.setChecked(True)
        elif self.active_source.startswith("dual"):
            self.btn_use_dual.setChecked(True)
        elif self.active_source.startswith("http") or self.active_source.startswith("rtsp"):
            self.btn_use_ip.setChecked(True)
        else:
            self.btn_use_phone1.setChecked(True)

        layout_active.addWidget(self.btn_use_phone1)
        layout_active.addWidget(self.btn_use_phone2)
        layout_active.addWidget(self.btn_use_dual)
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
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Top Section: Two QR Cards side by side
        qr_row = QtWidgets.QHBoxLayout()
        qr_row.setSpacing(16)

        # CARD 1: Phone 1 - Front Camera
        card_cam1 = QtWidgets.QFrame()
        card_cam1.setStyleSheet("""
            QFrame {
                background: #181c24;
                border: 2px solid #10b981;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        c1_layout = QtWidgets.QVBoxLayout(card_cam1)
        c1_layout.setSpacing(6)
        c1_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        lbl_c1_title = QtWidgets.QLabel("📱 Телефон 1: ФРОНТ (Лицо/Корпус)")
        lbl_c1_title.setStyleSheet("font-size: 10.5pt; font-weight: bold; color: #10b981;")
        lbl_c1_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c1_layout.addWidget(lbl_c1_title)

        self.lbl_qr_cam1 = QtWidgets.QLabel()
        self.lbl_qr_cam1.setFixedSize(150, 150)
        self.lbl_qr_cam1.setStyleSheet("background: white; border-radius: 6px; padding: 4px;")
        self.lbl_qr_cam1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c1_layout.addWidget(self.lbl_qr_cam1)
        self.lbl_qr_image = self.lbl_qr_cam1  # backwards compatibility

        url_cam1 = self.phone_server.get_connection_url("cam1")
        self.lbl_url_cam1 = QtWidgets.QLabel(f"<a href='{url_cam1}' style='color: #10b981; font-weight: bold; font-size: 8.5pt;'>{url_cam1}</a>")
        self.lbl_url_cam1.setOpenExternalLinks(True)
        self.lbl_url_cam1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c1_layout.addWidget(self.lbl_url_cam1)

        self.lbl_phone1_status = QtWidgets.QLabel("⚪ Ожидание подключения...")
        self.lbl_phone1_status.setStyleSheet("font-size: 8.5pt; color: #8c939e;")
        self.lbl_phone1_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c1_layout.addWidget(self.lbl_phone1_status)

        qr_row.addWidget(card_cam1)

        # CARD 2: Phone 2 - Side Camera (Depth / Stance)
        card_cam2 = QtWidgets.QFrame()
        card_cam2.setStyleSheet("""
            QFrame {
                background: #181c24;
                border: 2px solid #06b6d4;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        c2_layout = QtWidgets.QVBoxLayout(card_cam2)
        c2_layout.setSpacing(6)
        c2_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        lbl_c2_title = QtWidgets.QLabel("📱 Телефон 2: СБОКУ (Стойка/Глубина)")
        lbl_c2_title.setStyleSheet("font-size: 10.5pt; font-weight: bold; color: #06b6d4;")
        lbl_c2_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c2_layout.addWidget(lbl_c2_title)

        self.lbl_qr_cam2 = QtWidgets.QLabel()
        self.lbl_qr_cam2.setFixedSize(150, 150)
        self.lbl_qr_cam2.setStyleSheet("background: white; border-radius: 6px; padding: 4px;")
        self.lbl_qr_cam2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c2_layout.addWidget(self.lbl_qr_cam2)

        url_cam2 = self.phone_server.get_connection_url("cam2")
        self.lbl_url_cam2 = QtWidgets.QLabel(f"<a href='{url_cam2}' style='color: #06b6d4; font-weight: bold; font-size: 8.5pt;'>{url_cam2}</a>")
        self.lbl_url_cam2.setOpenExternalLinks(True)
        self.lbl_url_cam2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c2_layout.addWidget(self.lbl_url_cam2)

        self.lbl_phone2_status = QtWidgets.QLabel("⚪ Ожидание подключения...")
        self.lbl_phone2_status.setStyleSheet("font-size: 8.5pt; color: #8c939e;")
        self.lbl_phone2_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        c2_layout.addWidget(self.lbl_phone2_status)

        qr_row.addWidget(card_cam2)
        layout.addLayout(qr_row)

        # Instructions and quick tips
        box_guide = QtWidgets.QFrame()
        box_guide.setStyleSheet("background-color: #232731; border: 1px solid #3d4352; border-radius: 8px; padding: 8px 12px;")
        g_layout = QtWidgets.QVBoxLayout(box_guide)
        g_layout.setContentsMargins(6, 4, 6, 4)
        g_layout.setSpacing(4)

        lbl_gt = QtWidgets.QLabel("💡 <b>Инструкция по подключению 2 смартфонов:</b>")
        lbl_gt.setStyleSheet("color: #ffd32a; font-size: 9pt;")
        g_layout.addWidget(lbl_gt)

        lbl_gb = QtWidgets.QLabel(
            "1. Подключите оба телефона к <b>тому же Wi-Fi роутеру</b>, что и компьютер.<br>"
            "2. Отсканируйте левый QR первым телефоном (поставьте прямо перед собой во весь рост, 2-3 м).<br>"
            "3. Отсканируйте правый QR вторым телефоном (поставьте сбоку под углом 90° для захвата глубины стоек).<br>"
            "4. ⚠️ <b>В браузере телефона:</b> при предупреждении безопасности нажмите <b>«Дополнительно» → «Перейти на сайт»</b> (в Safari: <b>«Показать подробности» → «Посетить этот веб-сайт»</b>).<br>"
            "5. Нажмите <b>«Включить камеру»</b> (или <b>«Начать эфир»</b>) и разрешите доступ к камере."
        )
        lbl_gb.setStyleSheet("color: #ced6e0; font-size: 8.5pt;")
        g_layout.addWidget(lbl_gb)
        layout.addWidget(box_guide)

        # Generate QR code Pixmaps
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

    def _update_qr_image(self):
        try:
            # QR for Cam 1
            qr1_bytes = self.phone_server.generate_qr_png_bytes(role="cam1")
            qimg1 = QtGui.QImage.fromData(qr1_bytes)
            pix1 = QtGui.QPixmap.fromImage(qimg1).scaled(
                142, 142, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_qr_cam1.setPixmap(pix1)

            # QR for Cam 2
            qr2_bytes = self.phone_server.generate_qr_png_bytes(role="cam2")
            qimg2 = QtGui.QImage.fromData(qr2_bytes)
            pix2 = QtGui.QPixmap.fromImage(qimg2).scaled(
                142, 142, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_qr_cam2.setPixmap(pix2)
        except Exception as e:
            if hasattr(self, "lbl_qr_cam1"):
                self.lbl_qr_cam1.setText(f"QR Error: {e}")

    def _refresh_connected_devices(self):
        cams = self.phone_server.get_connected_cameras()

        if "cam1" in cams:
            frame, fps = self.phone_server.get_latest_frame("cam1") or (None, 30.0)
            res_str = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "HD"
            self.lbl_phone1_status.setText(f"<span style='color: #10b981; font-weight: bold;'>🟢 В эфире ({fps:.1f} FPS, {res_str})</span>")
        else:
            self.lbl_phone1_status.setText("<span style='color: #8c939e;'>⚪ Ожидание подключения...</span>")

        if "cam2" in cams:
            frame, fps = self.phone_server.get_latest_frame("cam2") or (None, 30.0)
            res_str = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "HD"
            self.lbl_phone2_status.setText(f"<span style='color: #06b6d4; font-weight: bold;'>🟢 В эфире ({fps:.1f} FPS, {res_str})</span>")
        else:
            self.lbl_phone2_status.setText("<span style='color: #8c939e;'>⚪ Ожидание подключения...</span>")

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
        if self.btn_use_phone2.isChecked():
            chosen = "phone:cam2"
        elif self.btn_use_dual.isChecked():
            chosen = "dual:phone+phone"
        elif self.btn_use_ip.isChecked():
            chosen = self.edit_ip_url.text().strip() or "phone:cam1"
        else:
            chosen = "phone:cam1"

        self.active_source = chosen
        self.cameraSelected.emit(chosen)
        self.accept()

    def get_selected_source(self) -> str:
        return self.active_source
