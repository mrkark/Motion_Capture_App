@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo   Bunkai Explorer - 3D Viewer Launch
echo ======================================================================
echo Starting local web server on port 8080...
echo Opening http://localhost:8080/view_capture.html in browser...

start http://localhost:8080/view_capture.html
".venv\Scripts\python.exe" -m http.server 8080
pause
