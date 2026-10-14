# Karate Motion Studio Root PowerShell Launcher
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StudioDir = Join-Path $RootDir "karate_motion_studio"
$VenvPy = Join-Path $RootDir "bunkai_explorer_karate_motion_capture\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    $VenvPy = Join-Path $StudioDir ".venv\Scripts\python.exe"
}

if (-not (Test-Path $VenvPy)) {
    Write-Warning "Виртуальное окружение не найдено в стандартных путях. Используется системный python..."
    $VenvPy = "python"
}

Set-Location $StudioDir
Write-Host "Запуск Karate Motion Studio из $StudioDir..." -ForegroundColor Green
& $VenvPy main.py
