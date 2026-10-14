@echo off
chcp 65001 >nul
title Karate Motion Studio

echo ========================================================
echo   Karate Motion Studio — Запуск приложения
echo ========================================================

set CURRENT_DIR=%~dp0
set VENV_PY=%CURRENT_DIR%..\bunkai_explorer_karate_motion_capture\.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    set VENV_PY=%CURRENT_DIR%.venv\Scripts\python.exe
)

if not exist "%VENV_PY%" (
    echo [ПРЕДУПРЕЖДЕНИЕ] Виртуальное окружение не найдено в стандартных путях.
    echo Используется системный python...
    set VENV_PY=python
)

echo Запуск Karate Motion Studio...
cd /d "%CURRENT_DIR%"
"%VENV_PY%" main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ОШИБКА] Приложение завершилось с кодом ошибки %ERRORLEVEL%.
    pause
)
