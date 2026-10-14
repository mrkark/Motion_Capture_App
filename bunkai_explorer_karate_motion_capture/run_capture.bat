@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo   Bunkai Explorer - Extracting Karate Motion Landmarks
echo ======================================================================

set INPUT_VIDEO=%1
if "%INPUT_VIDEO%"=="" set INPUT_VIDEO=test_data\karate_test.mp4

set OUTPUT_JSON=%2
if "%OUTPUT_JSON%"=="" set OUTPUT_JSON=test_output\captured_pose.json

set PREVIEW_MP4=test_output\captured_preview.mp4

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    pause
    exit /b 1
)

echo Input video:    %INPUT_VIDEO%
echo Output JSON:    %OUTPUT_JSON%
echo Preview video:  %PREVIEW_MP4%
echo.

".venv\Scripts\python.exe" capture_pose.py "%INPUT_VIDEO%" -o "%OUTPUT_JSON%" --preview-video "%PREVIEW_MP4%" %3 %4 %5 %6
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAIL] Extraction encountered errors.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Pose extracted successfully!
echo Saved: %OUTPUT_JSON%
echo Saved preview: %PREVIEW_MP4%
pause
