"""
Phone Stream Server for Karate Motion Studio.
Enables using one or multiple smartphones (iPhone / Android) as wireless HD cameras
without installing any mobile apps — just scan a QR code in the local Wi-Fi network!
Supports SSL (HTTPS / WSS) for mobile browser security and camera access.
"""
from __future__ import annotations
import os
import sys
import io
import time
import socket
import threading
from typing import Dict, Optional, Callable, List, Tuple, Any
from pathlib import Path
import numpy as np
import cv2

try:
    import qrcode
    from PIL import Image
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response


def get_local_ip() -> str:
    """Finds the local network IP of the host machine on the Wi-Fi / LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        # Does not actually send data over wire, just identifies outbound interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_self_signed_cert(cert_path: Path, key_path: Path, host_ip: str) -> bool:
    """Generates a self-signed RSA-2048 X.509 certificate with Subject Alternative Names for local Wi-Fi mocap."""
    try:
        import datetime
        import ipaddress
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        cert_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
        key_path.write_bytes(key_pem)

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "RU"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Karate Motion Studio"),
            x509.NameAttribute(NameOID.COMMON_NAME, host_ip),
        ])

        san_list = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(host_ip)))
        except ValueError:
            san_list.append(x509.DNSName(host_ip))

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        cert_path.write_bytes(cert_pem)
        return True
    except Exception as e:
        print(f"Warning: Failed to generate self-signed SSL cert: {e}")
        return False


MOBILE_CLIENT_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Karate Motion Studio — Мобильная Камера</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f1115;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 100vh;
      width: 100vw;
    }
    header {
      background: #1a1d24;
      padding: 10px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #2d3139;
      z-index: 10;
    }
    .title {
      font-size: 15px;
      font-weight: 700;
      color: #ff4757;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .badge-status {
      font-size: 12px;
      padding: 3px 8px;
      border-radius: 12px;
      font-weight: 600;
      background: #2ed573;
      color: #0f1115;
    }
    .badge-status.offline {
      background: #ff4757;
      color: #ffffff;
    }

    /* Warning banner for insecure context */
    .insecure-alert {
      background: #ff4757;
      color: #fff;
      padding: 10px 14px;
      font-size: 13px;
      text-align: center;
      display: none;
      font-weight: 500;
      line-height: 1.4;
      z-index: 30;
    }
    .insecure-alert a {
      color: #ffeaa7;
      font-weight: 700;
      text-decoration: underline;
    }

    .viewport {
      flex: 1;
      position: relative;
      background: #000;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    canvas { display: none; }
    .overlay-grid {
      position: absolute;
      inset: 0;
      pointer-events: none;
      border: 1px solid rgba(255, 71, 87, 0.2);
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      grid-template-rows: 1fr 1fr 1fr;
    }
    .overlay-grid div {
      border: 1px dashed rgba(255, 255, 255, 0.08);
    }

    /* Camera Permission Card */
    .permission-card {
      position: absolute;
      inset: 12px;
      background: rgba(22, 25, 32, 0.96);
      border: 1px solid #3d4352;
      border-radius: 12px;
      padding: 18px;
      display: none;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      gap: 12px;
      z-index: 25;
      backdrop-filter: blur(8px);
    }
    .permission-card h3 {
      color: #ff4757;
      font-size: 17px;
      margin: 0;
    }
    .permission-card p {
      font-size: 13px;
      color: #ced6e0;
      margin: 0;
      line-height: 1.4;
    }
    .guide-box {
      background: #121419;
      border: 1px solid #2d3139;
      border-radius: 8px;
      padding: 10px 12px;
      text-align: left;
      font-size: 12px;
      color: #a4b0be;
      width: 100%;
      box-sizing: border-box;
    }
    .guide-box ol {
      margin: 6px 0 0 18px;
      padding: 0;
    }
    .guide-box li {
      margin-bottom: 4px;
    }
    .btn-grant {
      background: #2ed573;
      color: #0f1115;
      border: none;
      border-radius: 8px;
      padding: 13px 20px;
      font-size: 15px;
      font-weight: bold;
      cursor: pointer;
      width: 100%;
      transition: background 0.2s;
    }
    .btn-grant:active {
      background: #26af5f;
    }

    .controls {
      background: #1a1d24;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      border-top: 1px solid #2d3139;
      z-index: 10;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    select, button {
      background: #2b2f3a;
      color: #ffffff;
      border: 1px solid #3d4352;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    select { flex: 1; outline: none; }
    button.btn-primary {
      background: #ff4757;
      border-color: #ff6b81;
      flex: 1;
    }
    button.btn-primary.active {
      background: #2ed573;
      border-color: #7bed9f;
      color: #0f1115;
    }
    .hud-info {
      font-size: 11px;
      color: #a4b0be;
      display: flex;
      justify-content: space-between;
      padding: 0 4px;
    }
  </style>
</head>
<body>
  <div id="insecureBanner" class="insecure-alert">
    ⚠️ Камера заблокирована браузером по HTTP.<br>
    <a id="httpsSwitchLink" href="#">Нажмите здесь для перехода на защищенный HTTPS ➔</a>
  </div>

  <header>
    <div class="title">🥋 Karate Motion Cam</div>
    <div id="statusBadge" class="badge-status offline">Отключено</div>
  </header>

  <div class="viewport">
    <video id="videoPreview" autoplay playsinline muted></video>
    <canvas id="captureCanvas"></canvas>
    <div class="overlay-grid">
      <div></div><div></div><div></div>
      <div></div><div></div><div></div>
      <div></div><div></div><div></div>
    </div>

    <!-- Permission & Camera Error Card -->
    <div id="permissionCard" class="permission-card">
      <div style="font-size: 38px;">📷🔒</div>
      <h3 id="permTitle">Разрешение доступа к камере</h3>
      <p id="permDesc">Для передачи видео на ПК необходимо разрешить доступ к камере телефона.</p>
      <button id="grantCameraBtn" class="btn-grant" type="button">🔓 Разрешить доступ к камере</button>
      <div class="guide-box">
        <b>Если камера не включается:</b>
        <ol>
          <li>Убедитесь, что страница открыта по <b><a id="permHttpsLink" href="#" style="color:#ff6b81;">HTTPS</a></b>.</li>
          <li>При предупреждении сертификата нажмите <b>«Дополнительно»</b> → <b>«Перейти на сайт»</b> (в Safari: <b>«Показать подробности»</b> → <b>«Посетить этот веб-сайт»</b>).</li>
          <li>В адресной строке нажмите значок настроек сайта 🔒 и выберите <b>«Разрешить камеру»</b>.</li>
        </ol>
      </div>
    </div>
  </div>

  <div class="controls">
    <div class="hud-info">
      <span id="resInfo">Разрешение: --</span>
      <span id="fpsInfo">FPS: 0.0</span>
      <span id="pingInfo">Кадров: 0</span>
    </div>

    <div class="row">
      <select id="roleSelect">
        <option value="cam1" selected>📱 Камера 1 (Фронт — лицо/корпус)</option>
        <option value="cam2">📱 Камера 2 (Сбоку — стойка/глубина)</option>
        <option value="cam3">📱 Камера 3 (Диагональ 45°)</option>
      </select>
      <button id="flipBtn" type="button">🔄 Камера</button>
    </div>

    <div class="row">
      <select id="resSelect">
        <option value="720p" selected>HD 720p (Рекомендуется, 30 FPS)</option>
        <option value="1080p">Full HD 1080p (Высокая четкость)</option>
        <option value="480p">SD 480p (Для слабого Wi-Fi)</option>
      </select>
      <button id="streamBtn" class="btn-primary" type="button">▶ Начать эфир</button>
    </div>
  </div>

  <script>
    const video = document.getElementById('videoPreview');
    const canvas = document.getElementById('captureCanvas');
    const ctx = canvas.getContext('2d');
    const roleSelect = document.getElementById('roleSelect');
    const resSelect = document.getElementById('resSelect');
    const flipBtn = document.getElementById('flipBtn');
    const streamBtn = document.getElementById('streamBtn');
    const statusBadge = document.getElementById('statusBadge');
    const fpsInfo = document.getElementById('fpsInfo');
    const resInfo = document.getElementById('resInfo');
    const pingInfo = document.getElementById('pingInfo');
    const permissionCard = document.getElementById('permissionCard');
    const grantCameraBtn = document.getElementById('grantCameraBtn');
    const insecureBanner = document.getElementById('insecureBanner');
    const httpsSwitchLink = document.getElementById('httpsSwitchLink');
    const permHttpsLink = document.getElementById('permHttpsLink');

    let currentStream = null;
    let ws = null;
    let isStreaming = false;
    let facingMode = 'environment'; // default to back camera for wide angle
    let frameCount = 0;
    let fpsTimer = performance.now();
    let wakeLock = null;

    // Check secure context (HTTPS / localhost required for getUserMedia on phones)
    const httpsUrl = `https://${window.location.hostname}:${window.location.port}/`;
    httpsSwitchLink.href = httpsUrl;
    permHttpsLink.href = httpsUrl;

    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      insecureBanner.style.display = 'block';
    }

    // Keep screen awake while streaming
    async function requestWakeLock() {
      try {
        if ('wakeLock' in navigator) {
          wakeLock = await navigator.wakeLock.request('screen');
        }
      } catch (e) {
        console.warn('WakeLock not supported or denied:', e);
      }
    }

    async function getCameraStream(facing, width, height) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('SECURE_CONTEXT_REQUIRED');
      }

      // Tier 1: requested resolution and facing mode
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: facing },
            width: { ideal: width },
            height: { ideal: height },
            frameRate: { ideal: 30 }
          }
        });
      } catch (e1) {
        console.warn('Tier 1 constraints failed:', e1);
      }

      // Tier 2: facing mode without resolution requirement
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: facing } }
        });
      } catch (e2) {
        console.warn('Tier 2 constraints failed:', e2);
      }

      // Tier 3: exact string facing mode
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: facing }
        });
      } catch (e3) {
        console.warn('Tier 3 constraints failed:', e3);
      }

      // Tier 4: any video camera on device
      return await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: true
      });
    }

    async function initCamera() {
      const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
      if (!hasMediaDevices) {
        showPermissionError('SECURE_CONTEXT_REQUIRED', 'Браузер заблокировал доступ к камере по HTTP. Перейдите на защищенный HTTPS.');
        return;
      }

      if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
      }

      const resChoice = resSelect.value;
      let width = 1280, height = 720;
      if (resChoice === '1080p') { width = 1920; height = 1080; }
      else if (resChoice === '480p') { width = 640; height = 480; }

      try {
        currentStream = await getCameraStream(facingMode, width, height);
        video.srcObject = currentStream;
        await video.play();

        hidePermissionError();

        const track = currentStream.getVideoTracks()[0];
        const settings = track.getSettings ? track.getSettings() : {};
        canvas.width = settings.width || video.videoWidth || width;
        canvas.height = settings.height || video.videoHeight || height;
        resInfo.textContent = `${canvas.width}x${canvas.height}`;
      } catch (err) {
        console.error('Camera init error:', err);
        showPermissionError(err.name || 'CameraError', err.message);
      }
    }

    function showPermissionError(errName, errMsg) {
      permissionCard.style.display = 'flex';
      const title = document.getElementById('permTitle');
      const desc = document.getElementById('permDesc');

      if (errName === 'SECURE_CONTEXT_REQUIRED' || window.location.protocol !== 'https:') {
        title.textContent = '🔒 Требуется защищенный HTTPS';
        desc.innerHTML = `Мобильные браузеры (Safari / Chrome) блокируют доступ к камере по протоколу HTTP.<br><br><a href="${httpsUrl}" style="display:inline-block; background:#ff4757; color:#fff; padding:10px 16px; border-radius:8px; font-weight:bold; text-decoration:none;">Перейти на HTTPS ➔</a>`;
      } else if (errName === 'NotAllowedError' || errName === 'PermissionDeniedError') {
        title.textContent = '🚫 Доступ к камере запрещен';
        desc.textContent = 'Вы запретили браузеру доступ к камере. Нажмите на значок настроек в адресной строке и включите «Камера».';
      } else {
        title.textContent = '📷 Ошибка доступа к камере';
        desc.textContent = errMsg || 'Не удалось активировать камеру телефона.';
      }
    }

    function hidePermissionError() {
      permissionCard.style.display = 'none';
    }

    grantCameraBtn.addEventListener('click', () => {
      initCamera();
    });

    flipBtn.addEventListener('click', () => {
      facingMode = (facingMode === 'environment') ? 'user' : 'environment';
      initCamera();
    });

    resSelect.addEventListener('change', initCamera);

    function connectWebSocket() {
      const role = roleSelect.value;
      const host = window.location.host;
      const protocol = (window.location.protocol === 'https:') ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${host}/ws/stream/${role}`;

      ws = new WebSocket(wsUrl);
      ws.binaryType = 'blob';

      ws.onopen = () => {
        statusBadge.textContent = '🟢 В эфире';
        statusBadge.classList.remove('offline');
        streamBtn.textContent = '⏹ Остановить';
        streamBtn.classList.add('active');
        isStreaming = true;
        requestWakeLock();
        scheduleNextFrame();
      };

      ws.onclose = () => {
        statusBadge.textContent = 'Отключено';
        statusBadge.classList.add('offline');
        streamBtn.textContent = '▶ Начать эфир';
        streamBtn.classList.remove('active');
        isStreaming = false;
      };

      ws.onerror = (e) => {
        console.error('WS Error:', e);
      };
    }

    function scheduleNextFrame() {
      if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;

      // Throttle to 30 FPS (~33ms)
      setTimeout(() => {
        if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((blob) => {
            if (blob && isStreaming && ws && ws.readyState === WebSocket.OPEN) {
              ws.send(blob);
              frameCount++;
              pingInfo.textContent = `Кадров: ${frameCount}`;

              const now = performance.now();
              if (now - fpsTimer >= 1000) {
                const fps = (frameCount * 1000) / (now - fpsTimer);
                fpsInfo.textContent = `FPS: ${fps.toFixed(1)}`;
                frameCount = 0;
                fpsTimer = now;
              }
            }
            scheduleNextFrame();
          }, 'image/jpeg', 0.72);
        } else {
          scheduleNextFrame();
        }
      }, 33);
    }

    streamBtn.addEventListener('click', () => {
      if (!isStreaming) {
        connectWebSocket();
      } else {
        if (ws) ws.close();
      }
    });

    // Auto-init on load
    initCamera();
  </script>
</body>
</html>
"""


class PhoneStreamServer:
    """
    Lightweight FastAPI + Uvicorn server running in a daemon thread.
    Receives live camera frames from mobile devices over Wi-Fi.
    Supports SSL (HTTPS / WSS) for mobile browser security.
    """

    def __init__(self, port: int = 8088, use_ssl: bool = True):
        self.port = port
        self.use_ssl = use_ssl
        self.local_ip = get_local_ip()
        self.app = FastAPI(title="Karate Motion Studio Mobile Cam Receiver")
        self._setup_routes()

        # Thread-safe storage: camera_id -> (bgr_frame, timestamp, fps)
        self.lock = threading.Lock()
        self.latest_frames: Dict[str, np.ndarray] = {}
        self.frame_timestamps: Dict[str, float] = {}
        self.frame_fps: Dict[str, float] = {}
        self.frame_counts: Dict[str, int] = {}
        self.fps_timers: Dict[str, float] = {}

        # Callbacks registered by Qt UI
        self.frame_callbacks: List[Callable[[str, np.ndarray, float], None]] = []

        self.server_thread: Optional[threading.Thread] = None
        self.server: Optional[uvicorn.Server] = None
        self.is_running = False

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def get_mobile_ui():
            return HTMLResponse(content=MOBILE_CLIENT_HTML, status_code=200)

        @self.app.get("/qr.png")
        async def get_qr_png():
            url = self.get_connection_url()
            img_bytes = self.generate_qr_png_bytes(url)
            return Response(content=img_bytes, media_type="image/png")

        @self.app.get("/status")
        async def get_status():
            with self.lock:
                cams = list(self.latest_frames.keys())
            return {
                "server": "Karate Motion Studio Phone Receiver",
                "ip": self.local_ip,
                "port": self.port,
                "ssl": self.use_ssl,
                "url": self.get_connection_url(),
                "active_cameras": cams,
            }

        @self.app.websocket("/ws/stream/{camera_id}")
        async def stream_endpoint(websocket: WebSocket, camera_id: str):
            await websocket.accept()
            t_last_fps = time.time()
            c_frames = 0
            cur_fps = 0.0

            try:
                while True:
                    data = await websocket.receive_bytes()
                    nparr = np.frombuffer(data, np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame_bgr is not None:
                        t_now = time.time()
                        c_frames += 1
                        if t_now - t_last_fps >= 1.0:
                            cur_fps = c_frames / (t_now - t_last_fps)
                            c_frames = 0
                            t_last_fps = t_now

                        with self.lock:
                            self.latest_frames[camera_id] = frame_bgr
                            self.frame_timestamps[camera_id] = t_now
                            self.frame_fps[camera_id] = cur_fps

                        # Dispatch callback
                        for cb in self.frame_callbacks:
                            try:
                                cb(camera_id, frame_bgr, cur_fps)
                            except Exception:
                                pass
            except WebSocketDisconnect:
                pass
            finally:
                with self.lock:
                    if camera_id in self.latest_frames:
                        del self.latest_frames[camera_id]

    def register_frame_callback(self, cb: Callable[[str, np.ndarray, float], None]):
        """Registers a listener for newly received phone camera frames."""
        if cb not in self.frame_callbacks:
            self.frame_callbacks.append(cb)

    def unregister_frame_callback(self, cb: Callable[[str, np.ndarray, float], None]):
        if cb in self.frame_callbacks:
            self.frame_callbacks.remove(cb)

    def get_latest_frame(self, camera_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Returns the latest (bgr_frame, fps) for camera_id if available within last 2 seconds."""
        with self.lock:
            if camera_id in self.latest_frames:
                t = self.frame_timestamps.get(camera_id, 0.0)
                if time.time() - t < 2.0:
                    return self.latest_frames[camera_id].copy(), self.frame_fps.get(camera_id, 0.0)
        return None

    def get_connected_cameras(self) -> List[str]:
        """Returns list of active camera IDs (e.g. ['cam1', 'cam2'])."""
        with self.lock:
            now = time.time()
            return [cid for cid, t in self.frame_timestamps.items() if now - t < 2.0]

    def get_connection_url(self) -> str:
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.local_ip}:{self.port}"

    def generate_qr_png_bytes(self, url: Optional[str] = None) -> bytes:
        """Generates a PNG QR code for the given URL."""
        target_url = url or self.get_connection_url()
        if HAS_QRCODE:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=2,
            )
            qr.add_data(target_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        else:
            # Fallback simple 1x1 dummy PNG if qrcode unavailable
            img = np.full((120, 120, 3), 255, dtype=np.uint8)
            cv2.putText(img, "QR", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            _, png = cv2.imencode(".png", img)
            return png.tobytes()

    def start(self):
        """Starts the FastAPI streaming server in a background daemon thread."""
        if self.is_running:
            return

        self.local_ip = get_local_ip()

        config_kwargs: Dict[str, Any] = {
            "app": self.app,
            "host": "0.0.0.0",
            "port": self.port,
            "log_level": "warning",
            "access_log": False,
        }

        if self.use_ssl:
            demo_dir = Path(__file__).resolve().parent.parent / "demo_assets"
            cert_path = demo_dir / "ssl_cert.pem"
            key_path = demo_dir / "ssl_key.pem"
            if not cert_path.is_file() or not key_path.is_file():
                generate_self_signed_cert(cert_path, key_path, self.local_ip)

            if cert_path.is_file() and key_path.is_file():
                config_kwargs["ssl_keyfile"] = str(key_path)
                config_kwargs["ssl_certfile"] = str(cert_path)

        config = uvicorn.Config(**config_kwargs)
        self.server = uvicorn.Server(config)

        def _run():
            self.server.run()

        self.server_thread = threading.Thread(target=_run, daemon=True, name="PhoneStreamServerThread")
        self.server_thread.start()
        self.is_running = True

    def stop(self):
        """Stops the streaming server."""
        self.is_running = False
        if self.server:
            self.server.should_exit = True
