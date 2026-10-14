@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo   Bunkai MoCap Live - Запуск приложения живого захвата движения
echo ======================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Виртуальное окружение .venv не найдено!
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Приложение завершилось с кодом: %ERRORLEVEL%
    pause
)
