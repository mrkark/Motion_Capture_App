@echo off
chcp 65001 >nul
title Karate Motion Studio Launcher

set ROOT_DIR=%~dp0
set STUDIO_DIR=%ROOT_DIR%karate_motion_studio
set VENV_PY=%ROOT_DIR%bunkai_explorer_karate_motion_capture\.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    set VENV_PY=%STUDIO_DIR%\.venv\Scripts\python.exe
)

if not exist "%VENV_PY%" (
    set VENV_PY=python
)

echo Запуск Karate Motion Studio...
cd /d "%STUDIO_DIR%"
"%VENV_PY%" main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ОШИБКА] Приложение завершилось с кодом ошибки %ERRORLEVEL%.
    pause
)
