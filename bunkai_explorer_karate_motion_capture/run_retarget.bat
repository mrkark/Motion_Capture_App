@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo   Bunkai Explorer - Blender Retargeting
echo ======================================================================

set BLENDER_EXE=
if exist "D:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe" (
    set "BLENDER_EXE=D:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
) else (
    where blender >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        for /f "delims=" %%i in ('where blender') do set "BLENDER_EXE=%%i"
    )
)

if "%BLENDER_EXE%"=="" (
    echo [ERROR] Blender executable not found!
    echo Please ensure Blender is installed or in PATH.
    pause
    exit /b 1
)

set AVATAR_BLEND=%1
if "%AVATAR_BLEND%"=="" set AVATAR_BLEND=test_data\karateka_lowpoly.blend

set POSE_JSON=%2
if "%POSE_JSON%"=="" (
    if exist "test_output\captured_pose.json" (
        set POSE_JSON=test_output\captured_pose.json
    ) else (
        set POSE_JSON=test_data\karate_test_pose.json
    )
)

set OUTPUT_BLEND=test_output\karateka_captured.blend
set OUTPUT_GLB=test_output\karateka_captured.glb

echo Blender:      "%BLENDER_EXE%"
echo Avatar Blend: "%AVATAR_BLEND%"
echo Pose JSON:    "%POSE_JSON%"
echo Output Blend: "%OUTPUT_BLEND%"
echo Output GLB:   "%OUTPUT_GLB%"
echo.

"%BLENDER_EXE%" -b "%AVATAR_BLEND%" -P blender_retarget.py -- --pose "%POSE_JSON%" --output "%OUTPUT_BLEND%" --export-glb "%OUTPUT_GLB%" --action-name "captured_karate_technique"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAIL] Retargeting encountered errors.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Retargeting and GLB export completed!
echo Created: %OUTPUT_BLEND%
echo Created: %OUTPUT_GLB%
pause
