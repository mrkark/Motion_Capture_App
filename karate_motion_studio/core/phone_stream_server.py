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
import json
import asyncio
import socket
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional, Callable, List, Tuple, Any, Set
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Karate Motion Studio — Мобильная Камера</title>
  <style>
    :root {
      --bg: #090a0f;
      --card-bg: rgba(18, 22, 32, 0.92);
      --border: #232a3b;
      --text: #f1f2f6;
      --text-muted: #8b95a8;
      --theme-color: #10b981; /* Emerald default for Cam1 */
      --theme-glow: rgba(16, 185, 129, 0.28);
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.35);
    }
    body[data-role="cam2"] {
      --theme-color: #06b6d4; /* Cyan for Cam2 */
      --theme-glow: rgba(6, 182, 212, 0.28);
    }
    body[data-role="cam3"] {
      --theme-color: #a855f7; /* Purple for Cam3 */
      --theme-glow: rgba(168, 85, 247, 0.28);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    html, body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      touch-action: manipulation;
    }

    /* Top Navigation Bar */
    header {
      background: rgba(13, 16, 24, 0.95);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      padding: max(8px, env(safe-area-inset-top)) 14px 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      z-index: 40;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-logo {
      font-size: 20px;
    }
    .brand-text {
      display: flex;
      flex-direction: column;
    }
    .brand-title {
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.4px;
      color: #fff;
    }
    .brand-subtitle {
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    .header-pills {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .role-pill {
      font-size: 11px;
      font-weight: 700;
      padding: 4px 9px;
      border-radius: 20px;
      background: rgba(0, 0, 0, 0.5);
      border: 1px solid var(--theme-color);
      color: var(--theme-color);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 4px;
      box-shadow: 0 0 10px var(--theme-glow);
    }
    .status-badge {
      font-size: 11px;
      font-weight: 600;
      padding: 4px 9px;
      border-radius: 20px;
      background: #272d3d;
      color: #cbd5e1;
      display: flex;
      align-items: center;
      gap: 5px;
      transition: all 0.25s ease;
    }
    .status-badge.live {
      background: var(--danger);
      color: #ffffff;
      box-shadow: 0 0 12px var(--danger-glow);
      animation: pulseLive 1.8s infinite;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: currentColor;
    }

    @keyframes pulseLive {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.85; transform: scale(0.96); }
    }

    /* Warning banner for HTTP vs HTTPS */
    .insecure-alert {
      background: linear-gradient(135deg, #ef4444, #b91c1c);
      color: #fff;
      padding: 10px 14px;
      font-size: 12.5px;
      text-align: center;
      display: none;
      font-weight: 500;
      line-height: 1.4;
      z-index: 50;
      border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    .insecure-alert a {
      color: #fef08a;
      font-weight: 700;
      text-decoration: underline;
    }

    /* Viewport Container */
    .viewport {
      flex: 1;
      position: relative;
      background: #000;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
      border-left: 2px solid transparent;
      border-right: 2px solid transparent;
      transition: border-color 0.3s;
    }
    body.is-live .viewport {
      border-color: var(--theme-color);
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform-origin: center center;
    }
    video.mirrored {
      transform: scaleX(-1);
    }
    canvas { display: none; }

    /* Stance Alignment Grid Overlay */
    .grid-overlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      grid-template-rows: 1fr 1fr 1fr;
      opacity: 0.22;
      transition: opacity 0.2s;
    }
    .grid-overlay.hidden { display: none; }
    .grid-overlay div {
      border: 1px dashed #ffffff;
    }
    /* Central stance crosshair */
    .crosshair-center {
      position: absolute;
      width: 60px;
      height: 60px;
      border: 1px solid var(--theme-color);
      border-radius: 50%;
      opacity: 0.45;
      pointer-events: none;
    }

    /* Floating On-Screen HUD */
    .floating-hud {
      position: absolute;
      top: 10px;
      left: 10px;
      right: 10px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      pointer-events: none;
      z-index: 20;
    }
    .hud-chips {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .hud-chip {
      background: rgba(10, 14, 22, 0.75);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      color: #e2e8f0;
      border: 1px solid rgba(255, 255, 255, 0.1);
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-family: monospace;
    }

    .quick-actions {
      display: flex;
      flex-direction: column;
      gap: 8px;
      pointer-events: auto;
    }
    .circle-btn {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background: rgba(18, 22, 32, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.16);
      color: #ffffff;
      display: flex;
      justify-content: center;
      align-items: center;
      font-size: 17px;
      cursor: pointer;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
      transition: all 0.15s ease;
    }
    .circle-btn:active {
      transform: scale(0.92);
      background: rgba(30, 38, 54, 0.95);
    }
    .circle-btn.active {
      background: var(--theme-color);
      color: #000;
      border-color: #fff;
    }

    /* Role Guidance Banner at bottom of viewport */
    .role-guide-banner {
      position: absolute;
      bottom: 10px;
      left: 12px;
      right: 12px;
      background: rgba(12, 16, 26, 0.82);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid var(--theme-color);
      border-radius: 10px;
      padding: 7px 12px;
      font-size: 11.5px;
      color: #e2e8f0;
      text-align: center;
      pointer-events: none;
      z-index: 15;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    }
    .role-guide-banner b {
      color: var(--theme-color);
    }

    /* Camera Permission Card */
    .permission-card {
      position: absolute;
      inset: 12px;
      background: rgba(15, 18, 26, 0.97);
      border: 1px solid #3b4254;
      border-radius: 14px;
      padding: 20px 16px;
      display: none;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      gap: 12px;
      z-index: 35;
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    }
    .permission-card h3 {
      color: #ff6b81;
      font-size: 17px;
      margin: 0;
      font-weight: 700;
    }
    .permission-card p {
      font-size: 13px;
      color: #cbd5e1;
      margin: 0;
      line-height: 1.45;
    }
    .guide-box {
      background: #0d1017;
      border: 1px solid #282f42;
      border-radius: 8px;
      padding: 10px 12px;
      text-align: left;
      font-size: 11.5px;
      color: #94a3b8;
      width: 100%;
      box-sizing: border-box;
      line-height: 1.4;
    }
    .guide-box ol {
      margin: 6px 0 0 16px;
      padding: 0;
    }
    .guide-box li {
      margin-bottom: 4px;
    }
    .btn-grant {
      background: linear-gradient(135deg, #10b981, #059669);
      color: #ffffff;
      border: none;
      border-radius: 10px;
      padding: 13px 20px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
      transition: opacity 0.2s;
    }
    .btn-grant:active { opacity: 0.85; }

    /* Tap to start camera overlay (solves mobile autoplay gesture policy) */
    .tap-start-overlay {
      position: absolute;
      inset: 0;
      background: rgba(11, 15, 23, 0.88);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      padding: 24px;
      gap: 12px;
      z-index: 32;
      cursor: pointer;
      transition: opacity 0.25s ease;
    }
    .tap-start-overlay.hidden {
      display: none !important;
    }
    .tap-icon-ring {
      width: 70px;
      height: 70px;
      border-radius: 50%;
      background: rgba(16, 185, 129, 0.15);
      border: 2px solid var(--theme-color);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 32px;
      box-shadow: 0 0 24px var(--theme-glow);
      animation: pulseTap 2s infinite ease-in-out;
    }
    @keyframes pulseTap {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.08); }
    }
    .tap-title {
      font-size: 17px;
      font-weight: 700;
      color: #ffffff;
      margin: 0;
    }
    .tap-sub {
      font-size: 12.5px;
      color: #94a3b8;
      max-width: 280px;
      line-height: 1.4;
      margin: 0;
    }
    .tap-btn {
      background: linear-gradient(135deg, var(--theme-color), #059669);
      color: #000000;
      font-weight: 800;
      font-size: 14px;
      padding: 12px 24px;
      border: none;
      border-radius: 24px;
      cursor: pointer;
      box-shadow: 0 4px 16px var(--theme-glow);
      margin-top: 4px;
    }
    body[data-role="cam2"] .tap-btn {
      background: linear-gradient(135deg, var(--theme-color), #0284c7);
      color: #ffffff;
    }
    .perm-action-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
      width: 100%;
    }
    .btn-subtle {
      background: #1e2433;
      color: #cbd5e1;
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 11px 16px;
      font-size: 13.5px;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      transition: background 0.2s;
    }
    .btn-subtle:active {
      background: #283044;
    }

    /* Bottom Control Deck */
    .controls-deck {
      background: rgba(13, 16, 24, 0.96);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      padding: 10px 14px max(10px, env(safe-area-inset-bottom));
      display: flex;
      flex-direction: column;
      gap: 10px;
      border-top: 1px solid var(--border);
      z-index: 40;
    }

    .main-action-row {
      display: flex;
      gap: 8px;
    }

    .btn-record-anim {
      flex: 1.5;
      background: linear-gradient(135deg, #ef4444, #b91c1c);
      color: #ffffff;
      border: 2px solid rgba(255, 255, 255, 0.2);
      border-radius: 12px;
      padding: 13px 14px;
      font-size: 14.5px;
      font-weight: 800;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      letter-spacing: 0.3px;
      box-shadow: 0 4px 18px rgba(239, 68, 68, 0.45);
      transition: all 0.2s ease;
    }
    .btn-record-anim:active {
      transform: scale(0.97);
    }
    .btn-record-anim.recording {
      background: #11141d;
      border: 2px solid #ef4444;
      color: #ef4444;
      box-shadow: 0 0 16px rgba(239, 68, 68, 0.6);
      animation: pulseRedRecord 1.2s infinite;
    }

    @keyframes pulseRedRecord {
      0%, 100% { box-shadow: 0 0 16px rgba(239, 68, 68, 0.7); transform: scale(1); }
      50% { box-shadow: 0 0 5px rgba(239, 68, 68, 0.3); transform: scale(0.985); }
    }

    .rec-live-badge {
      position: absolute;
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(220, 38, 38, 0.94);
      color: #ffffff;
      font-weight: 800;
      font-size: 12.5px;
      padding: 5px 14px;
      border-radius: 20px;
      display: flex;
      align-items: center;
      gap: 8px;
      z-index: 30;
      box-shadow: 0 4px 16px rgba(220, 38, 38, 0.6);
      letter-spacing: 0.5px;
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
    }
    .rec-live-badge.hidden { display: none !important; }
    .rec-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #ffffff;
      animation: blinkDot 0.9s infinite alternate;
    }
    @keyframes blinkDot {
      0% { opacity: 1; transform: scale(1); }
      100% { opacity: 0.2; transform: scale(0.7); }
    }

    .rec-toast {
      position: absolute;
      bottom: 20px;
      left: 14px;
      right: 14px;
      background: rgba(15, 23, 42, 0.97);
      border: 2px solid #10b981;
      border-radius: 14px;
      padding: 16px;
      text-align: center;
      z-index: 55;
      box-shadow: 0 10px 30px rgba(0,0,0,0.85);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }
    .rec-toast.hidden { display: none !important; }
    .rec-toast-icon { font-size: 30px; margin-bottom: 4px; }
    .rec-toast-title { font-size: 16px; font-weight: 800; color: #10b981; margin-bottom: 4px; }
    .rec-toast-desc { font-size: 12.5px; color: #cbd5e1; line-height: 1.45; margin-bottom: 10px; }
    .toast-close-btn {
      background: #10b981;
      color: #000;
      font-weight: 800;
      border: none;
      border-radius: 8px;
      padding: 8px 20px;
      cursor: pointer;
      font-size: 13px;
    }

    .btn-stream {
      flex: 0.9;
      background: linear-gradient(135deg, var(--theme-color), #059669);
      color: #ffffff;
      border: none;
      border-radius: 12px;
      padding: 13px 12px;
      font-size: 13.5px;
      font-weight: 800;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 6px;
      letter-spacing: 0.4px;
      box-shadow: 0 4px 16px var(--theme-glow);
      transition: all 0.2s ease;
    }
    body[data-role="cam2"] .btn-stream {
      background: linear-gradient(135deg, var(--theme-color), #0284c7);
    }
    .btn-stream:active {
      transform: scale(0.98);
    }
    .btn-stream.streaming {
      background: linear-gradient(135deg, #475569, #334155);
      box-shadow: none;
      color: #94a3b8;
    }

    .settings-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .select-wrapper {
      position: relative;
    }
    select {
      width: 100%;
      appearance: none;
      -webkit-appearance: none;
      background: #1b202e;
      color: #e2e8f0;
      border: 1px solid #2e374d;
      border-radius: 8px;
      padding: 9px 28px 9px 10px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      outline: none;
    }
    .select-arrow {
      position: absolute;
      right: 10px;
      top: 50%;
      transform: translateY(-50%);
      pointer-events: none;
      font-size: 10px;
      color: #94a3b8;
    }

    .footer-telemetry {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 10.5px;
      color: var(--text-muted);
      padding: 0 2px;
    }
    .wakelock-indicator {
      display: flex;
      align-items: center;
      gap: 4px;
      color: #10b981;
    }
    .wakelock-indicator.off {
      color: var(--text-muted);
    }
  </style>
</head>
<body data-role="cam1">
  <div id="insecureBanner" class="insecure-alert">
    ⚠️ Камера заблокирована браузером по HTTP.<br>
    <a id="httpsSwitchLink" href="#">Нажмите здесь для перехода на защищенный HTTPS ➔</a>
  </div>

  <header>
    <div class="brand">
      <span class="brand-logo">🥋</span>
      <div class="brand-text">
        <span class="brand-title">Karate Motion Cam</span>
        <span class="brand-subtitle">Shotokan Wireless HD</span>
      </div>
    </div>
    <div class="header-pills">
      <div id="rolePill" class="role-pill">
        <span id="roleIcon">🟢</span>
        <span id="roleText">Камера 1 (Фронт)</span>
      </div>
      <div id="statusBadge" class="status-badge">
        <div class="status-dot"></div>
        <span id="statusText">Готов</span>
      </div>
    </div>
  </header>

  <div class="viewport" id="viewportArea">
    <video id="videoPreview" autoplay playsinline muted></video>
    <canvas id="captureCanvas"></canvas>

    <!-- Floating Recording Live Badge -->
    <div id="recLiveBadge" class="rec-live-badge hidden">
      <div class="rec-dot"></div>
      <span id="recTimerText">REC 00:00</span>
    </div>

    <!-- Success Toast on Recording Complete -->
    <div id="recToast" class="rec-toast hidden">
      <div class="rec-toast-icon">🥋✨</div>
      <div class="rec-toast-title">Анимация успешно сохранена!</div>
      <div id="recToastDesc" class="rec-toast-desc">Записано кадров. Перейдите к ноутбуку — 3D аватар уже воспроизводит вашу технику!</div>
      <button id="toastCloseBtn" class="toast-close-btn" type="button">Понятно</button>
    </div>

    <!-- Stance Alignment Grid -->
    <div class="grid-overlay" id="gridOverlay">
      <div></div><div></div><div></div>
      <div></div><div></div><div></div>
      <div></div><div></div><div></div>
    </div>
    <div class="crosshair-center"></div>

    <!-- On-Screen Floating HUD -->
    <div class="floating-hud">
      <div class="hud-chips">
        <span class="hud-chip" id="chipRes">📏 --x--</span>
        <span class="hud-chip" id="chipFps">⚡ 0.0 FPS</span>
        <span class="hud-chip" id="chipCount">📦 0 кадров</span>
      </div>

      <div class="quick-actions">
        <button id="btnFlipCam" class="circle-btn" type="button" title="Переключить камеру">🔄</button>
        <button id="btnTorch" class="circle-btn" type="button" title="Фонарик / Подсветка" style="display:none;">💡</button>
        <button id="btnToggleGrid" class="circle-btn active" type="button" title="Сетка калибровки">▦</button>
        <button id="btnFullscreen" class="circle-btn" type="button" title="Полноэкранный режим">⛶</button>
      </div>
    </div>

    <!-- Dynamic Role Guidance Banner -->
    <div class="role-guide-banner" id="roleGuide">
      <b>КАМЕРА 1 (ФРОНТ):</b> Установите телефон спереди во весь рост (2-3 метра).
    </div>

    <!-- Tap to Start Camera Overlay (Bypasses Mobile Autoplay Blocks) -->
    <div id="tapStartOverlay" class="tap-start-overlay">
      <div class="tap-icon-ring">📹</div>
      <div class="tap-title">Нажмите для включения камеры</div>
      <div class="tap-sub">Браузер требует касания экрана для активации видеопотока</div>
      <button id="tapStartBtn" class="tap-btn" type="button">ВКЛЮЧИТЬ КАМЕРУ</button>
    </div>

    <!-- Permission & Camera Error Card -->
    <div id="permissionCard" class="permission-card">
      <div style="font-size: 38px;">📷🔒</div>
      <h3 id="permTitle">Доступ к камере смартфона</h3>
      <p id="permDesc">Для передачи видео на ПК необходимо предоставить доступ к камере.</p>
      <div class="perm-action-group">
        <button id="grantCameraBtn" class="btn-grant" type="button">🔓 Разрешить доступ к камере</button>
        <button id="btnTryFrontCam" class="btn-subtle" type="button">🔄 Переключить на селфи-камеру</button>
      </div>
      <div class="guide-box">
        <b>Если камера не включается:</b>
        <ol>
          <li>Убедитесь, что страница открыта по <b><a id="permHttpsLink" href="#" style="color:#ff6b81;">HTTPS</a></b>.</li>
          <li>При предупреждении сертификата нажмите <b>«Дополнительно» → «Перейти на сайт»</b> (Safari: <b>«Показать подробности» → «Посетить этот веб-сайт»</b>).</li>
          <li>В адресной строке нажмите значок 🔒 или ⚙️ / aA и выберите <b>«Камера: Разрешить»</b>.</li>
          <li>Закройте другие приложения (Instagram, WhatsApp, системную камеру), которые могут занимать объектив.</li>
        </ol>
      </div>
    </div>
  </div>

  <div class="controls-deck">
    <div class="main-action-row">
      <button id="btnRecordAnim" class="btn-record-anim" type="button">
        <span id="btnRecordIcon">🎬</span>
        <span id="btnRecordText">ЗАПИСАТЬ АНИМАЦИЮ</span>
      </button>
      <button id="streamBtn" class="btn-stream" type="button">
        <span id="streamBtnIcon">▶</span>
        <span id="streamBtnText">ЭФИР</span>
      </button>
    </div>

    <div class="settings-row">
      <div class="select-wrapper">
        <select id="roleSelect">
          <option value="cam1" selected>📱 Камера 1 (Фронт)</option>
          <option value="cam2">📱 Камера 2 (Сбоку)</option>
          <option value="cam3">📱 Камера 3 (Диагональ)</option>
        </select>
        <div class="select-arrow">▼</div>
      </div>

      <div class="select-wrapper">
        <select id="resSelect">
          <option value="540p" selected>⚡ Турбо 540p (Мин. задержка)</option>
          <option value="720p">HD 720p (Баланс)</option>
          <option value="480p">SD 480p (Макс. FPS)</option>
          <option value="1080p">Full HD 1080p</option>
        </select>
        <div class="select-arrow">▼</div>
      </div>
    </div>

    <div class="footer-telemetry">
      <span id="statusHost">Сервер: Подключение...</span>
      <span id="wakeLockStatus" class="wakelock-indicator">
        <span>☀️</span>
        <span id="wakeLockText">Экран активен</span>
      </span>
    </div>
  </div>

  <script>
    const video = document.getElementById('videoPreview');
    const canvas = document.getElementById('captureCanvas');
    const ctx = canvas.getContext('2d');
    const roleSelect = document.getElementById('roleSelect');
    const resSelect = document.getElementById('resSelect');
    const streamBtn = document.getElementById('streamBtn');
    const streamBtnIcon = document.getElementById('streamBtnIcon');
    const streamBtnText = document.getElementById('streamBtnText');
    const statusBadge = document.getElementById('statusBadge');
    const statusText = document.getElementById('statusText');
    const rolePill = document.getElementById('rolePill');
    const roleIcon = document.getElementById('roleIcon');
    const roleText = document.getElementById('roleText');
    const roleGuide = document.getElementById('roleGuide');
    const chipFps = document.getElementById('chipFps');
    const chipRes = document.getElementById('chipRes');
    const chipCount = document.getElementById('chipCount');
    const statusHost = document.getElementById('statusHost');
    const wakeLockStatus = document.getElementById('wakeLockStatus');
    const wakeLockText = document.getElementById('wakeLockText');
    const permissionCard = document.getElementById('permissionCard');
    const grantCameraBtn = document.getElementById('grantCameraBtn');
    const btnTryFrontCam = document.getElementById('btnTryFrontCam');
    const tapStartOverlay = document.getElementById('tapStartOverlay');
    const tapStartBtn = document.getElementById('tapStartBtn');
    const insecureBanner = document.getElementById('insecureBanner');
    const httpsSwitchLink = document.getElementById('httpsSwitchLink');
    const permHttpsLink = document.getElementById('permHttpsLink');
    const btnFlipCam = document.getElementById('btnFlipCam');
    const btnTorch = document.getElementById('btnTorch');
    const btnToggleGrid = document.getElementById('btnToggleGrid');
    const btnFullscreen = document.getElementById('btnFullscreen');
    const gridOverlay = document.getElementById('gridOverlay');

    const btnRecordAnim = document.getElementById('btnRecordAnim');
    const btnRecordIcon = document.getElementById('btnRecordIcon');
    const btnRecordText = document.getElementById('btnRecordText');
    const recLiveBadge = document.getElementById('recLiveBadge');
    const recTimerText = document.getElementById('recTimerText');
    const recToast = document.getElementById('recToast');
    const recToastDesc = document.getElementById('recToastDesc');
    const toastCloseBtn = document.getElementById('toastCloseBtn');

    let isRecordingAnim = false;
    let recStartTime = 0;
    let recTimerInterval = null;

    function formatTime(sec) {
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
    }

    function setRecordingUI(active, savedFrames) {
      isRecordingAnim = active;
      if (active) {
        btnRecordAnim.classList.add('recording');
        btnRecordIcon.textContent = '⏹';
        btnRecordText.textContent = 'СТОП (REC)';
        recLiveBadge.classList.remove('hidden');
        if (!recTimerInterval) {
          recStartTime = performance.now();
          recTimerInterval = setInterval(() => {
            const elapsed = (performance.now() - recStartTime) / 1000;
            const timeStr = formatTime(elapsed);
            recTimerText.textContent = `REC ${timeStr}`;
            btnRecordText.textContent = `СТОП (${timeStr})`;
          }, 250);
        }
      } else {
        btnRecordAnim.classList.remove('recording');
        btnRecordIcon.textContent = '🎬';
        btnRecordText.textContent = 'ЗАПИСАТЬ АНИМАЦИЮ';
        recLiveBadge.classList.add('hidden');
        if (recTimerInterval) {
          clearInterval(recTimerInterval);
          recTimerInterval = null;
        }
        if (savedFrames && savedFrames > 0) {
          recToastDesc.textContent = `Записано ${savedFrames} кадров движения. Перейдите к ноутбуку — 3D аватар уже воспроизводит вашу технику!`;
          recToast.classList.remove('hidden');
          setTimeout(() => { recToast.classList.add('hidden'); }, 7000);
        }
      }
    }

    toastCloseBtn.addEventListener('click', () => {
      recToast.classList.add('hidden');
    });

    async function toggleRecordingFromPhone() {
      if (!isStreaming) {
        await startCameraAndStream();
        await new Promise(r => setTimeout(r, 600));
      }

      if (!isRecordingAnim) {
        setRecordingUI(true);
        try {
          const resp = await fetch('/api/record/start', { method: 'POST' });
          if (!resp.ok) setRecordingUI(false);
        } catch (e) {
          console.error('Error starting record:', e);
        }
      } else {
        try {
          const resp = await fetch('/api/record/stop', { method: 'POST' });
          const data = await resp.json();
          setRecordingUI(false, data.frames || 0);
        } catch (e) {
          console.error('Error stopping record:', e);
          setRecordingUI(false);
        }
      }
    }

    btnRecordAnim.addEventListener('click', toggleRecordingFromPhone);

    setInterval(async () => {
      if (isStreaming) {
        try {
          const res = await fetch('/api/record/status');
          const data = await res.json();
          if (data.is_recording !== isRecordingAnim) {
            setRecordingUI(data.is_recording, data.frames || 0);
          }
        } catch (e) {}
      }
    }, 1500);

    let currentStream = null;
    let ws = null;
    let isStreaming = false;
    let facingMode = 'environment'; // back camera by default for wider angle on tripod
    let isTorchOn = false;
    let frameCount = 0;
    let totalFramesSent = 0;
    let fpsTimer = performance.now();
    let wakeLock = null;

    // Detect role from URL path (e.g. /cam1 or /cam2)
    function autoDetectRoleFromUrl() {
      const path = window.location.pathname.toLowerCase();
      if (path.includes('cam2')) {
        roleSelect.value = 'cam2';
      } else if (path.includes('cam3')) {
        roleSelect.value = 'cam3';
      } else {
        roleSelect.value = 'cam1';
      }
      applyRoleTheme(roleSelect.value);
    }

    function applyRoleTheme(role) {
      document.body.setAttribute('data-role', role);
      if (role === 'cam1') {
        roleIcon.textContent = '🟢';
        roleText.textContent = 'Камера 1 (Фронт)';
        roleGuide.innerHTML = '<b>КАМЕРА 1 (ФРОНТ):</b> Установите телефон спереди во весь рост (2-3 метра).';
      } else if (role === 'cam2') {
        roleIcon.textContent = '🔵';
        roleText.textContent = 'Камера 2 (Сбоку)';
        roleGuide.innerHTML = '<b>КАМЕРА 2 (СБОКУ):</b> Установите сбоку под углом 90° для точной глубины стоек.';
      } else {
        roleIcon.textContent = '🟣';
        roleText.textContent = 'Камера 3 (Угол)';
        roleGuide.innerHTML = '<b>КАМЕРА 3 (УГОЛ):</b> Диагональный ракурс 45° для детального захвата ударов.';
      }
    }

    roleSelect.addEventListener('change', () => {
      applyRoleTheme(roleSelect.value);
      if (isStreaming) {
        // Reconnect WebSocket to the new role endpoint immediately
        connectWebSocket();
      }
    });

    // Check secure context (HTTPS / localhost required for getUserMedia on phones)
    const httpsUrl = `https://${window.location.hostname}:${window.location.port}${window.location.pathname}`;
    httpsSwitchLink.href = httpsUrl;
    permHttpsLink.href = httpsUrl;

    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      insecureBanner.style.display = 'block';
    }

    statusHost.textContent = `Хост: ${window.location.hostname}:${window.location.port}`;

    // Keep screen awake while app is running
    async function requestWakeLock() {
      try {
        if ('wakeLock' in navigator) {
          wakeLock = await navigator.wakeLock.request('screen');
          wakeLockStatus.classList.remove('off');
          wakeLockText.textContent = 'Экран не погаснет';
          wakeLock.addEventListener('release', () => {
            wakeLockStatus.classList.add('off');
            wakeLockText.textContent = 'Экран: обычный';
          });
        }
      } catch (e) {
        wakeLockStatus.classList.add('off');
        wakeLockText.textContent = 'WakeLock недоступен';
      }
    }

    document.addEventListener('visibilitychange', () => {
      if (wakeLock !== null && document.visibilityState === 'visible') {
        requestWakeLock();
      }
    });

    async function getCameraStream(facing, width, height) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('SECURE_CONTEXT_REQUIRED');
      }

      // Tier 1: requested resolution and ideal facing mode
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

      // Tier 2: exact facing mode with ideal resolution
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: facing,
            width: { ideal: width },
            height: { ideal: height }
          }
        });
      } catch (e2) {
        console.warn('Tier 2 constraints failed:', e2);
      }

      // Tier 3: facing mode ideal without resolution restriction
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: facing } }
        });
      } catch (e3) {
        console.warn('Tier 3 constraints failed:', e3);
      }

      // Tier 4: facing mode exact string without resolution restriction
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: facing }
        });
      } catch (e4) {
        console.warn('Tier 4 constraints failed:', e4);
      }

      // Tier 5: absolute fallback to any video camera on device
      return await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: true
      });
    }

    async function initCamera() {
      const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
      if (!hasMediaDevices) {
        showPermissionError('SECURE_CONTEXT_REQUIRED', 'Браузер заблокировал доступ к камере по HTTP. Перейдите на защищенный HTTPS.');
        return false;
      }

      if (currentStream) {
        try {
          currentStream.getTracks().forEach(t => t.stop());
        } catch (e) {}
        currentStream = null;
      }

      const resChoice = resSelect.value;
      let width = 960, height = 540;
      if (resChoice === '1080p') { width = 1920; height = 1080; }
      else if (resChoice === '720p') { width = 1280; height = 720; }
      else if (resChoice === '480p') { width = 640; height = 480; }

      try {
        currentStream = await getCameraStream(facingMode, width, height);

        // Required on mobile Safari & Chrome to allow video playback without user gesture denial
        video.muted = true;
        video.defaultMuted = true;
        video.playsInline = true;
        video.setAttribute('playsinline', '');
        video.setAttribute('webkit-playsinline', '');
        video.setAttribute('muted', '');

        video.srcObject = currentStream;

        // Ensure video metadata loaded before playing
        await new Promise((resolve) => {
          if (video.readyState >= 1) {
            resolve();
          } else {
            video.onloadedmetadata = () => resolve();
            setTimeout(resolve, 800);
          }
        });

        try {
          await video.play();
        } catch (playErr) {
          console.warn('video.play() rejected (autoplay policy), showing tap overlay:', playErr);
          tapStartOverlay.classList.remove('hidden');
          return false;
        }

        tapStartOverlay.classList.add('hidden');
        hidePermissionError();

        // Apply visual mirroring only if front selfie camera is active
        video.classList.toggle('mirrored', facingMode === 'user');

        const track = currentStream.getVideoTracks()[0];
        const settings = track.getSettings ? track.getSettings() : {};
        canvas.width = settings.width || video.videoWidth || width;
        canvas.height = settings.height || video.videoHeight || height;
        chipRes.textContent = `📏 ${canvas.width}x${canvas.height}`;

        // Check torch capabilities
        const capabilities = track.getCapabilities ? track.getCapabilities() : {};
        if (capabilities.torch) {
          btnTorch.style.display = 'flex';
        } else {
          btnTorch.style.display = 'none';
        }

        requestWakeLock();
        return true;
      } catch (err) {
        console.error('Camera init error:', err);
        tapStartOverlay.classList.remove('hidden');
        showPermissionError(err.name || 'CameraError', err.message);
        return false;
      }
    }

    function showPermissionError(errName, errMsg) {
      permissionCard.style.display = 'flex';
      const title = document.getElementById('permTitle');
      const desc = document.getElementById('permDesc');

      if (errName === 'SECURE_CONTEXT_REQUIRED' || window.location.protocol !== 'https:') {
        title.textContent = '🔒 Требуется защищенный HTTPS';
        desc.innerHTML = `Мобильные браузеры (Safari / Chrome) блокируют доступ к камере по HTTP.<br><br><a href="${httpsUrl}" style="display:inline-block; background:#10b981; color:#fff; padding:10px 16px; border-radius:8px; font-weight:bold; text-decoration:none;">Перейти на HTTPS ➔</a>`;
      } else if (errName === 'NotAllowedError' || errName === 'PermissionDeniedError') {
        title.textContent = '🚫 Доступ к камере ожидает подтверждения';
        desc.textContent = 'Браузер ожидает разрешения на использование камеры. Нажмите кнопку «Разрешить» ниже или проверьте значок 🔒 в адресной строке.';
      } else if (errName === 'NotFoundError' || errName === 'DevicesNotFoundError') {
        title.textContent = '📷 Камера не обнаружена';
        desc.textContent = 'Запрашиваемый модуль камеры не найден на устройстве. Попробуйте переключиться на селфи-камеру.';
      } else if (errName === 'NotReadableError' || errName === 'TrackStartError') {
        title.textContent = '⚠️ Камера занята другим приложением';
        desc.textContent = 'Камера используется другим приложением (Instagram, WhatsApp, встроенная камера). Закройте их и повторите.';
      } else {
        title.textContent = '📷 Ошибка доступа к камере';
        desc.textContent = errMsg || 'Не удалось активировать видеопоток с камеры смартфона.';
      }
    }

    function hidePermissionError() {
      permissionCard.style.display = 'none';
    }

    async function startCameraAndStream() {
      hidePermissionError();
      const ok = await initCamera();
      if (ok || (currentStream && currentStream.active)) {
        connectWebSocket();
      }
    }

    grantCameraBtn.addEventListener('click', startCameraAndStream);

    btnTryFrontCam.addEventListener('click', async () => {
      facingMode = 'user';
      startCameraAndStream();
    });

    tapStartOverlay.addEventListener('click', startCameraAndStream);

    tapStartBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      startCameraAndStream();
    });

    btnFlipCam.addEventListener('click', async () => {
      facingMode = (facingMode === 'environment') ? 'user' : 'environment';
      await initCamera();
    });

    btnTorch.addEventListener('click', async () => {
      if (!currentStream) return;
      const track = currentStream.getVideoTracks()[0];
      if (track && track.applyConstraints) {
        try {
          isTorchOn = !isTorchOn;
          await track.applyConstraints({ advanced: [{ torch: isTorchOn }] });
          btnTorch.classList.toggle('active', isTorchOn);
        } catch (e) {
          console.warn('Torch toggle failed:', e);
        }
      }
    });

    btnToggleGrid.addEventListener('click', () => {
      const isHidden = gridOverlay.classList.toggle('hidden');
      btnToggleGrid.classList.toggle('active', !isHidden);
    });

    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });

    resSelect.addEventListener('change', async () => {
      await initCamera();
    });

    let userStoppedStreaming = false;
    let reconnectTimeout = null;
    let isEncoding = false;
    let lastFrameTime = performance.now();

    function connectWebSocket() {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }
      if (ws) {
        try { ws.close(); } catch (e) {}
      }

      userStoppedStreaming = false;
      const role = roleSelect.value;
      const host = window.location.host;
      const protocol = (window.location.protocol === 'https:') ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${host}/ws/stream/${role}`;

      ws = new WebSocket(wsUrl);
      ws.binaryType = 'blob';

      ws.onopen = () => {
        statusBadge.classList.add('live');
        statusText.textContent = 'В ЭФИРЕ';
        streamBtn.classList.add('streaming');
        streamBtnIcon.textContent = '⏹';
        streamBtnText.textContent = 'ОСТАНОВИТЬ ЭФИР';
        document.body.classList.add('is-live');
        isStreaming = true;
        isEncoding = false;
        lastFrameTime = performance.now();
        requestWakeLock();
        requestNextFrame();
      };

      ws.onclose = () => {
        statusBadge.classList.remove('live');
        document.body.classList.remove('is-live');
        isStreaming = false;
        isEncoding = false;

        if (userStoppedStreaming) {
          statusText.textContent = 'Готов';
          streamBtn.classList.remove('streaming');
          streamBtnIcon.textContent = '▶';
          streamBtnText.textContent = 'НАЧАТЬ ЭФИР';
        } else {
          // Unexpected drop / Wi-Fi jitter: Auto-reconnect seamlessly!
          statusText.textContent = 'Переподключение...';
          streamBtnIcon.textContent = '🔄';
          streamBtnText.textContent = 'ПЕРЕПОДКЛЮЧЕНИЕ...';
          reconnectTimeout = setTimeout(() => {
            if (!userStoppedStreaming) {
              connectWebSocket();
            }
          }, 1500);
        }
      };

      ws.onmessage = (event) => {
        try {
          if (typeof event.data === 'string') {
            const msg = JSON.parse(event.data);
            if (msg.type === 'record_status') {
              if (msg.is_recording !== isRecordingAnim) {
                setRecordingUI(msg.is_recording, msg.saved_frames || 0);
              }
            }
          }
        } catch (e) {}
      };

      ws.onerror = (e) => {
        console.error('WS Error:', e);
      };
    }

    function scheduleNextFrame() {
      if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;

      // Zero-Buffer Backpressure: skip capture only if queue is seriously backed up (> 256KB)
      // This guarantees minimal latency without stalling on normal Wi-Fi frame sizes
      if (ws.bufferedAmount > 262144 || isEncoding) {
        requestNextFrame();
        return;
      }

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        isEncoding = true;
        try {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const resChoice = resSelect.value;
          let quality = 0.58;
          if (resChoice === '1080p') quality = 0.75;
          else if (resChoice === '720p') quality = 0.65;
          else if (resChoice === '480p') quality = 0.50;

          canvas.toBlob((blob) => {
            isEncoding = false;
            try {
              if (blob && isStreaming && ws && ws.readyState === WebSocket.OPEN) {
                if (ws.bufferedAmount <= 262144) {
                  ws.send(blob);
                  lastFrameTime = performance.now();
                  frameCount++;
                  totalFramesSent++;
                  chipCount.textContent = `📦 ${totalFramesSent} кадров`;

                  const now = performance.now();
                  if (now - fpsTimer >= 1000) {
                    const fps = (frameCount * 1000) / (now - fpsTimer);
                    chipFps.textContent = `⚡ ${fps.toFixed(1)} FPS`;
                    frameCount = 0;
                    fpsTimer = now;
                  }
                }
              }
            } catch (err) {
              console.warn('Frame send error:', err);
            }
            requestNextFrame();
          }, 'image/jpeg', quality);
        } catch (e) {
          isEncoding = false;
          requestNextFrame();
        }
      } else {
        requestNextFrame();
      }
    }

    function requestNextFrame() {
      if (!isStreaming) return;
      if ('requestVideoFrameCallback' in video) {
        video.requestVideoFrameCallback(() => scheduleNextFrame());
      } else {
        requestAnimationFrame(() => scheduleNextFrame());
      }
    }

    // Hardware Watchdog: if stream stalls for > 350ms, kickstart next frame
    setInterval(() => {
      if (isStreaming && ws && ws.readyState === WebSocket.OPEN && performance.now() - lastFrameTime > 350) {
        isEncoding = false;
        scheduleNextFrame();
      }
    }, 250);

    streamBtn.addEventListener('click', async () => {
      if (!isStreaming) {
        userStoppedStreaming = false;
        await startCameraAndStream();
      } else {
        userStoppedStreaming = true;
        if (reconnectTimeout) clearTimeout(reconnectTimeout);
        if (ws) ws.close();
      }
    });

    // Auto-detect role from URL and initialize camera with auto-stream if permission granted
    autoDetectRoleFromUrl();
    (async () => {
      const ok = await initCamera();
      if (ok) {
        connectWebSocket();
      }
    })();
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

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.loop = asyncio.get_running_loop()
            yield

        self.app = FastAPI(title="Karate Motion Studio Mobile Cam Receiver", lifespan=lifespan)
        self._setup_routes()

        # Thread-safe storage: camera_id -> (bgr_frame, timestamp, fps)
        self.lock = threading.Lock()
        self.latest_frames: Dict[str, np.ndarray] = {}
        self.frame_timestamps: Dict[str, float] = {}
        self.frame_fps: Dict[str, float] = {}
        self.frame_counts: Dict[str, int] = {}
        self.fps_timers: Dict[str, float] = {}
        self.frame_seq: Dict[str, int] = {}
        self.new_frame_event = threading.Event()

        # Remote Recording Synchronization
        self.is_recording: bool = False
        self.record_start_t: float = 0.0
        self.last_recorded_frames: int = 0
        self.record_command_handler: Optional[Callable[[str], None]] = None
        self.active_websockets: Set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks registered by Qt UI
        self.frame_callbacks: List[Callable[[str, np.ndarray, float], None]] = []

        self.server_thread: Optional[threading.Thread] = None
        self.server: Optional[uvicorn.Server] = None
        self.is_running = False

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def get_mobile_ui():
            return HTMLResponse(content=MOBILE_CLIENT_HTML, status_code=200)

        @self.app.get("/cam1", response_class=HTMLResponse)
        async def get_mobile_cam1():
            return HTMLResponse(content=MOBILE_CLIENT_HTML, status_code=200)

        @self.app.get("/cam2", response_class=HTMLResponse)
        async def get_mobile_cam2():
            return HTMLResponse(content=MOBILE_CLIENT_HTML, status_code=200)

        @self.app.get("/cam3", response_class=HTMLResponse)
        async def get_mobile_cam3():
            return HTMLResponse(content=MOBILE_CLIENT_HTML, status_code=200)

        @self.app.get("/qr.png")
        async def get_qr_png():
            url = self.get_connection_url("cam1")
            img_bytes = self.generate_qr_png_bytes(url)
            return Response(content=img_bytes, media_type="image/png")

        @self.app.get("/qr/cam1.png")
        async def get_qr_cam1_png():
            url = self.get_connection_url("cam1")
            img_bytes = self.generate_qr_png_bytes(url)
            return Response(content=img_bytes, media_type="image/png")

        @self.app.get("/qr/cam2.png")
        async def get_qr_cam2_png():
            url = self.get_connection_url("cam2")
            img_bytes = self.generate_qr_png_bytes(url)
            return Response(content=img_bytes, media_type="image/png")

        @self.app.get("/status")
        async def get_status():
            with self.lock:
                cams = list(self.latest_frames.keys())
            dur = time.time() - self.record_start_t if self.is_recording else 0.0
            return {
                "server": "Karate Motion Studio Phone Receiver",
                "ip": self.local_ip,
                "port": self.port,
                "ssl": self.use_ssl,
                "url": self.get_connection_url(),
                "active_cameras": cams,
                "is_recording": self.is_recording,
                "recording_duration": round(dur, 2),
                "last_frames": self.last_recorded_frames,
            }

        @self.app.post("/api/record/start")
        async def api_record_start():
            self.is_recording = True
            self.record_start_t = time.time()
            self.dispatch_record_command("start")
            await self.broadcast_json({"type": "record_status", "is_recording": True})
            return {"status": "ok", "action": "start"}

        @self.app.post("/api/record/stop")
        async def api_record_stop():
            self.is_recording = False
            self.dispatch_record_command("stop")
            await asyncio.sleep(0.08)
            await self.broadcast_json({"type": "record_status", "is_recording": False, "saved_frames": self.last_recorded_frames})
            return {"status": "ok", "action": "stop", "frames": self.last_recorded_frames}

        @self.app.get("/api/record/status")
        async def api_record_status():
            dur = time.time() - self.record_start_t if self.is_recording else 0.0
            return {
                "is_recording": self.is_recording,
                "duration": round(dur, 2),
                "frames": self.last_recorded_frames,
            }

        @self.app.websocket("/ws/stream/{camera_id}")
        async def stream_endpoint(websocket: WebSocket, camera_id: str):
            await websocket.accept()
            self.active_websockets.add(websocket)
            t_last_fps = time.time()
            c_frames = 0
            cur_fps = 0.0

            try:
                while True:
                    msg = await websocket.receive()
                    if "bytes" in msg and msg["bytes"]:
                        data = msg["bytes"]
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
                                self.frame_seq[camera_id] = self.frame_seq.get(camera_id, 0) + 1
                            self.new_frame_event.set()

                            # Dispatch callback
                            for cb in self.frame_callbacks:
                                try:
                                    cb(camera_id, frame_bgr, cur_fps)
                                except Exception:
                                    pass
                    elif "text" in msg and msg["text"]:
                        try:
                            cmd = json.loads(msg["text"])
                            action = cmd.get("action")
                            if action == "start_recording":
                                self.is_recording = True
                                self.record_start_t = time.time()
                                self.dispatch_record_command("start")
                                await self.broadcast_json({"type": "record_status", "is_recording": True})
                            elif action == "stop_recording":
                                self.is_recording = False
                                self.dispatch_record_command("stop")
                                await asyncio.sleep(0.08)
                                await self.broadcast_json({"type": "record_status", "is_recording": False, "saved_frames": self.last_recorded_frames})
                        except Exception:
                            pass
            except WebSocketDisconnect:
                pass
            finally:
                self.active_websockets.discard(websocket)
                with self.lock:
                    if camera_id in self.latest_frames:
                        del self.latest_frames[camera_id]

    async def broadcast_json(self, data: dict):
        text = json.dumps(data)
        to_remove = set()
        for ws in list(self.active_websockets):
            try:
                await ws.send_text(text)
            except Exception:
                to_remove.add(ws)
        self.active_websockets -= to_remove

    def set_record_command_handler(self, handler: Callable[[str], None]):
        """Sets callback invoked when mobile user requests start/stop recording."""
        self.record_command_handler = handler

    def dispatch_record_command(self, action: str):
        if self.record_command_handler:
            try:
                self.record_command_handler(action)
            except Exception:
                pass

    def notify_recording_finished(self, frame_count: int):
        """Called by Qt main window when recording completes and animation is baked."""
        self.is_recording = False
        self.last_recorded_frames = frame_count
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.broadcast_json({"type": "record_status", "is_recording": False, "saved_frames": frame_count}),
                self.loop
            )

    def notify_recording_started(self):
        """Called by Qt main window when recording starts from PC UI."""
        self.is_recording = True
        self.record_start_t = time.time()
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.broadcast_json({"type": "record_status", "is_recording": True}),
                self.loop
            )

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
                    return self.latest_frames[camera_id], self.frame_fps.get(camera_id, 0.0)
        return None

    def get_frame_seq(self, camera_id: str) -> int:
        """Returns the monotonic frame sequence ID for camera_id."""
        with self.lock:
            return self.frame_seq.get(camera_id, 0)

    def get_latest_frame_if_new(self, camera_id: str, last_seq: int = -1) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Returns (bgr_frame, fps, current_seq) only if a newer frame than last_seq has arrived.
        If no newer frame is available, returns None immediately without duplicating memory.
        """
        with self.lock:
            seq = self.frame_seq.get(camera_id, 0)
            if last_seq >= 0 and seq <= last_seq:
                return None
            if camera_id in self.latest_frames:
                t = self.frame_timestamps.get(camera_id, 0.0)
                if time.time() - t < 2.0:
                    return self.latest_frames[camera_id], self.frame_fps.get(camera_id, 0.0), seq
        return None

    def wait_for_frame(self, timeout: float = 0.033) -> bool:
        """Blocks until a new camera frame is received or timeout expires."""
        signaled = self.new_frame_event.wait(timeout)
        self.new_frame_event.clear()
        return signaled

    def get_connected_cameras(self) -> List[str]:
        """Returns list of active camera IDs (e.g. ['cam1', 'cam2'])."""
        with self.lock:
            now = time.time()
            return [cid for cid, t in self.frame_timestamps.items() if now - t < 2.0]

    def get_connection_url(self, role: Optional[str] = None) -> str:
        protocol = "https" if self.use_ssl else "http"
        base = f"{protocol}://{self.local_ip}:{self.port}"
        if role:
            return f"{base}/{role}"
        return base

    def generate_qr_png_bytes(self, url: Optional[str] = None, role: Optional[str] = None) -> bytes:
        """Generates a PNG QR code for the given URL or role."""
        target_url = url or self.get_connection_url(role)
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
            needs_new_cert = not cert_path.is_file() or not key_path.is_file()

            if not needs_new_cert:
                try:
                    from cryptography import x509
                    cert_bytes = cert_path.read_bytes()
                    cert_obj = x509.load_pem_x509_certificate(cert_bytes)
                    san_ext = cert_obj.extensions.get_extension_for_oid(x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                    san_ips = [str(ip) for ip in san_ext.value.get_values_for_type(x509.IPAddress)]
                    san_dns = [str(d) for d in san_ext.value.get_values_for_type(x509.DNSName)]
                    if self.local_ip not in san_ips and self.local_ip not in san_dns:
                        needs_new_cert = True
                except Exception:
                    needs_new_cert = True

            if needs_new_cert:
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
