@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo   Bunkai Explorer - Running Automated Test Suite
echo ======================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    echo Please run setup first or check Python 3.11 installation.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" test_pipeline.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAIL] Some tests failed. Check output above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] All tests passed successfully!
pause
