# Karate Motion Studio — PowerShell Launcher
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Karate Motion Studio — Запуск приложения" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$CurrentDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy = Join-Path $CurrentDir "..\bunkai_explorer_karate_motion_capture\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    $VenvPy = Join-Path $CurrentDir ".venv\Scripts\python.exe"
}

if (-not (Test-Path $VenvPy)) {
    Write-Warning "Виртуальное окружение не найдено в стандартных путях. Используется системный python..."
    $VenvPy = "python"
}

Set-Location $CurrentDir
Write-Host "Исполняемый файл Python: $VenvPy" -ForegroundColor Gray
Write-Host "Запуск интерфейса Karate Motion Studio..." -ForegroundColor Green

& $VenvPy main.py
