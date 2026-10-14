# PowerShell runner for 3D Motion Capture Viewer
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Bunkai Explorer - 3D Viewer (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Start-Process "http://localhost:8080/view_capture.html"
Write-Host "Web server started on http://localhost:8080/view_capture.html" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
& $pythonExe -m http.server 8080
