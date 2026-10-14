"""
Interactive Synchronized Timeline and Karate Motion Playback Controller.
Provides frame-accurate scrubbing, confidence heatmap, trim range in/out,
playback speed control, and simultaneous synchronization of video and 3D avatar.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any

from qt_compat import QtCore, QtGui, QtWidgets
from core.project_schema import FrameData


class TimelineTrack(QtWidgets.QWidget):
    """
    Custom scrubber track drawing frame ticks, trim bounds, and confidence health track.
    """

    frameSeekRequested = QtCore.pyqtSignal(int)
    trimChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.total_frames = 1
        self.current_frame = 0
        self.trim_start = 0
        self.trim_end = 0
        self.confidence_health: List[float] = []  # 0.0 to 1.0 per frame
        self.loss_flags: List[bool] = []

    def set_range(self, total_frames: int, trim_start: int = 0, trim_end: Optional[int] = None):
        self.total_frames = max(1, total_frames)
        self.trim_start = max(0, min(self.total_frames - 1, trim_start))
        self.trim_end = self.total_frames - 1 if trim_end is None else max(self.trim_start, min(self.total_frames - 1, trim_end))
        self.update()

    def set_current_frame(self, frame_idx: int):
        self.current_frame = max(0, min(self.total_frames - 1, frame_idx))
        self.update()

    def set_health_data(self, confidence_list: List[float], loss_list: List[bool]):
        self.confidence_health = list(confidence_list)
        self.loss_flags = list(loss_list)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background track
        painter.fillRect(0, 0, w, h, QtGui.QColor(22, 23, 26))

        if self.total_frames <= 1:
            return

        frame_w = w / float(self.total_frames)

        # 1. Draw Confidence Health Heatmap
        if self.confidence_health:
            for i in range(min(self.total_frames, len(self.confidence_health))):
                conf = self.confidence_health[i]
                is_lost = self.loss_flags[i] if i < len(self.loss_flags) else False

                if is_lost or conf < 0.15:
                    color = QtGui.QColor(220, 50, 50, 180)  # Red: Tracking Lost
                elif conf < 0.40:
                    color = QtGui.QColor(230, 160, 40, 180) # Amber: Weak Tracking
                else:
                    color = QtGui.QColor(46, 213, 115, 120) # Green: Confident Tracking

                x = int(i * frame_w)
                rw = max(1, int(frame_w + 0.5))
                painter.fillRect(x, 4, rw, h - 8, color)

        # 2. Dim regions outside Trim bounds
        x_trim_start = int(self.trim_start * frame_w)
        x_trim_end = int((self.trim_end + 1) * frame_w)

        dim_color = QtGui.QColor(10, 10, 12, 170)
        if x_trim_start > 0:
            painter.fillRect(0, 0, x_trim_start, h, dim_color)
        if x_trim_end < w:
            painter.fillRect(x_trim_end, 0, w - x_trim_end, h, dim_color)

        # Trim bound markers
        pen_trim = QtGui.QPen(QtGui.QColor(224, 82, 96), 2)
        painter.setPen(pen_trim)
        painter.drawLine(x_trim_start, 0, x_trim_start, h)
        painter.drawLine(x_trim_end, 0, x_trim_end, h)

        # 3. Draw Playhead Needle
        needle_x = int(self.current_frame * frame_w + frame_w * 0.5)
        needle_x = max(2, min(w - 2, needle_x))

        pen_needle = QtGui.QPen(QtGui.QColor(255, 255, 255), 2)
        painter.setPen(pen_needle)
        painter.drawLine(needle_x, 0, needle_x, h)

        # Playhead handle cap
        cap_poly = QtGui.QPolygonF([
            QtCore.QPointF(needle_x - 5, 0),
            QtCore.QPointF(needle_x + 5, 0),
            QtCore.QPointF(needle_x, 7),
        ])
        painter.setBrush(QtGui.QColor(255, 255, 255))
        painter.drawPolygon(cap_poly)

        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._handle_mouse(event.pos().x())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._handle_mouse(event.pos().x())

    def _handle_mouse(self, x: int):
        w = max(1, self.width())
        ratio = max(0.0, min(1.0, x / float(w)))
        frame_idx = int(ratio * (self.total_frames - 1) + 0.5)
        self.frameSeekRequested.emit(frame_idx)


class TimelineWidget(QtWidgets.QWidget):
    """
    Bottom Panel: Timeline scrubber, playback controls, trim in/out, and speed multiplier.
    """

    frameChanged = QtCore.pyqtSignal(int)
    trimUpdated = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(84)
        self.setStyleSheet("background-color: #1e1f23; border-top: 1px solid #2d3038;")

        # Playback state
        self.fps = 30.0
        self.speed_multiplier = 1.0
        self.is_playing = False
        self.total_frames = 0
        self.current_frame = 0
        self.trim_start = 0
        self.trim_end = 0

        # High-resolution playback timer
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_timer_tick)

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 6, 12, 6)
        main_layout.setSpacing(4)

        # 1. Custom Interactive Track
        self.track = TimelineTrack(self)
        self.track.frameSeekRequested.connect(self.seek_frame)
        main_layout.addWidget(self.track)

        # 2. Controls Toolbar
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(6)

        # Jump Start
        self.btn_start = self._create_btn("|<", "К началу (Home)", self.jump_to_start)
        ctrl_layout.addWidget(self.btn_start)

        # Step Back
        self.btn_prev = self._create_btn("<", "Предыдущий кадр (Стрелка влево)", self.step_backward)
        ctrl_layout.addWidget(self.btn_prev)

        # Play / Pause
        self.btn_play = self._create_btn("▶", "Воспроизведение / Пауза (Пробел)", self.toggle_play)
        self.btn_play.setStyleSheet(
            "background-color: #e05260; color: #fff; font-weight: bold; border-radius: 4px; padding: 4px 14px;"
        )
        ctrl_layout.addWidget(self.btn_play)

        # Step Forward
        self.btn_next = self._create_btn(">", "Следующий кадр (Стрелка вправо)", self.step_forward)
        ctrl_layout.addWidget(self.btn_next)

        # Jump End
        self.btn_end = self._create_btn(">|", "К концу (End)", self.jump_to_end)
        ctrl_layout.addWidget(self.btn_end)

        # Trim Buttons
        ctrl_layout.addSpacing(8)
        self.btn_trim_in = self._create_btn("[ In", "Установить метку Trim Start ([)", self.set_trim_in)
        ctrl_layout.addWidget(self.btn_trim_in)

        self.btn_trim_out = self._create_btn("Out ]", "Установить метку Trim End (])", self.set_trim_out)
        ctrl_layout.addWidget(self.btn_trim_out)

        self.btn_trim_reset = self._create_btn("Сброс", "Сбросить диапазон обрезки", self.reset_trim)
        ctrl_layout.addWidget(self.btn_trim_reset)

        ctrl_layout.addStretch()

        # Speed Multiplier
        lbl_speed = QtWidgets.QLabel("Скорость:")
        lbl_speed.setStyleSheet("color: #8c939e; font-size: 11px;")
        ctrl_layout.addWidget(lbl_speed)

        self.cmb_speed = QtWidgets.QComboBox()
        self.cmb_speed.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.cmb_speed.setCurrentText("1.0x")
        self.cmb_speed.setStyleSheet(
            "background-color: #2b2d33; color: #dfdfe0; font-size: 11px; border: 1px solid #3e424c; padding: 2px 6px;"
        )
        self.cmb_speed.currentTextChanged.connect(self._on_speed_changed)
        ctrl_layout.addWidget(self.cmb_speed)

        ctrl_layout.addSpacing(12)

        # Time & Frame Counter
        self.lbl_time = QtWidgets.QLabel("00:00.00 / 00:00.00 | Кадр 0 / 0")
        self.lbl_time.setStyleSheet("color: #dfdfe0; font-family: monospace; font-size: 11px;")
        ctrl_layout.addWidget(self.lbl_time)

        main_layout.addLayout(ctrl_layout)

    def _create_btn(self, text: str, tooltip: str, callback) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(26)
        btn.setStyleSheet(
            "background-color: #2b2d33; color: #dfdfe0; font-size: 11px; border: 1px solid #3e424c; border-radius: 3px; padding: 2px 8px;"
        )
        btn.clicked.connect(callback)
        return btn

    def load_project_frames(self, frames: List[FrameData], fps: float = 30.0, trim_range: Optional[List[int]] = None):
        self.pause()
        self.fps = max(1.0, fps)
        self.total_frames = len(frames)
        self.current_frame = 0

        t_start = trim_range[0] if trim_range else 0
        t_end = trim_range[1] if trim_range else max(0, self.total_frames - 1)
        self.trim_start = t_start
        self.trim_end = t_end

        # Extract confidence health track
        confs = []
        losses = []
        for f in frames:
            losses.append(f.tracking_loss)
            c_vals = list(f.confidence.values()) if f.confidence else [1.0]
            confs.append(float(sum(c_vals) / max(1, len(c_vals))))

        self.track.set_range(self.total_frames, self.trim_start, self.trim_end)
        self.track.set_health_data(confs, losses)
        self.seek_frame(self.trim_start)

    def _update_time_label(self):
        curr_sec = self.current_frame / self.fps if self.fps > 0 else 0.0
        tot_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

        c_m, c_s = divmod(curr_sec, 60.0)
        t_m, t_s = divmod(tot_sec, 60.0)

        self.lbl_time.setText(
            f"{int(c_m):02d}:{c_s:05.2f} / {int(t_m):02d}:{t_s:05.2f} | Кадр {self.current_frame + 1} / {self.total_frames}"
        )

    def seek_frame(self, frame_idx: int):
        if self.total_frames == 0:
            return
        self.current_frame = max(0, min(self.total_frames - 1, frame_idx))
        self.track.set_current_frame(self.current_frame)
        self._update_time_label()
        self.frameChanged.emit(self.current_frame)

    def toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if self.total_frames == 0:
            return
        self.is_playing = True
        self.btn_play.setText("⏸")
        interval_ms = int(1000.0 / (self.fps * self.speed_multiplier))
        self.play_timer.start(max(10, interval_ms))

    def pause(self):
        self.is_playing = False
        self.btn_play.setText("▶")
        self.play_timer.stop()

    def _on_timer_tick(self):
        next_frame = self.current_frame + 1
        # Loop inside trim bounds
        if next_frame > self.trim_end or next_frame >= self.total_frames:
            next_frame = self.trim_start

        self.seek_frame(next_frame)

    def step_forward(self):
        self.pause()
        self.seek_frame(self.current_frame + 1)

    def step_backward(self):
        self.pause()
        self.seek_frame(self.current_frame - 1)

    def jump_to_start(self):
        self.pause()
        self.seek_frame(self.trim_start)

    def jump_to_end(self):
        self.pause()
        self.seek_frame(self.trim_end)

    def set_trim_in(self):
        self.trim_start = self.current_frame
        if self.trim_end < self.trim_start:
            self.trim_end = self.trim_start
        self.track.set_range(self.total_frames, self.trim_start, self.trim_end)
        self.trimUpdated.emit(self.trim_start, self.trim_end)

    def set_trim_out(self):
        self.trim_end = self.current_frame
        if self.trim_start > self.trim_end:
            self.trim_start = self.trim_end
        self.track.set_range(self.total_frames, self.trim_start, self.trim_end)
        self.trimUpdated.emit(self.trim_start, self.trim_end)

    def reset_trim(self):
        self.trim_start = 0
        self.trim_end = max(0, self.total_frames - 1)
        self.track.set_range(self.total_frames, self.trim_start, self.trim_end)
        self.trimUpdated.emit(self.trim_start, self.trim_end)

    def _on_speed_changed(self, text: str):
        val = float(text.replace("x", ""))
        self.speed_multiplier = val
        if self.is_playing:
            self.play()  # Restart timer with new interval
