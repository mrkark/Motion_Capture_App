#!/usr/bin/env python3
"""
Bunkai MoCap Live — Desktop Motion Capture Application.
Launches the FastAPI backend and opens an interactive desktop application window.
"""
import os
import sys

# Silence internal C++ glog / TFLite / absl messages to stderr
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_MIN_SEVERITY"] = "3"

try:
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
except Exception:
    pass

import time
import socket
import threading
import webbrowser
import subprocess
from pathlib import Path

# Ensure working directory is the script directory
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}"

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0

def run_uvicorn_server():
    import uvicorn
    from server import app
    config = uvicorn.Config(
        app=app,
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False
    )
    server = uvicorn.Server(config)
    server.run()

def wait_for_server(timeout: float = 8.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(PORT):
            return True
        time.sleep(0.15)
    return False

def open_browser_window():
    # Attempt to open Chrome or Edge in standalone application window mode (--app)
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]

    for p in chrome_paths + edge_paths:
        if Path(p).is_file():
            try:
                subprocess.Popen([p, f"--app={URL}", "--window-size=1320,820"])
                print(f"Launched application window via: {Path(p).name}")
                return True
            except Exception:
                pass

    # Fallback to default system browser
    webbrowser.open(URL)
    return True

def main():
    print("=" * 70)
    print("  BUNKAI MOCAP LIVE — ЖИВОЙ ЗАХВАТ ДВИЖЕНИЯ КАРАТЕ")
    print("=" * 70)
    print(f"Запуск локального сервера на {URL}...")

    # Start FastAPI server in a background thread if not already running
    if not is_port_in_use(PORT):
        server_thread = threading.Thread(target=run_uvicorn_server, daemon=True)
        server_thread.start()
        if not wait_for_server():
            print("[ERROR] Не удалось запустить сервер uvicorn!")
            sys.exit(1)
        print("Сервер успешно запущен.")
    else:
        print(f"Сервер уже работает на порту {PORT}.")

    use_browser = "--browser" in sys.argv

    if not use_browser:
        try:
            import webview
            print("Открытие нативного окна приложения (pywebview)...")
            window = webview.create_window(
                title="Bunkai MoCap Live — Захват движений карате",
                url=URL,
                width=1340,
                height=840,
                min_size=(960, 620),
                text_select=False
            )
            webview.start()
            print("Окно приложения закрыто. Завершение работы.")
            sys.exit(0)
        except Exception as e:
            print(f"Примечание: pywebview недоступен ({e}), открываю браузерное окно...")

    # Browser window mode
    open_browser_window()
    print(f"\nПриложение работает по адресу: {URL}")
    print("Для завершения нажмите Ctrl+C в этом окне.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nЗавершение работы Bunkai MoCap Live.")

if __name__ == "__main__":
    main()
