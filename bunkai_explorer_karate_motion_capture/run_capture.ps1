# PowerShell runner for capture_pose.py
param(
    [string]$Video = "test_data\karate_test.mp4",
    [string]$Output = "test_output\captured_pose.json",
    [int]$Stride = 1,
    [int]$Smooth = 2,
    [string]$PreviewVideo = "test_output\captured_preview.mp4"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Bunkai Explorer - Pose Capture (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment .venv not found at: $pythonExe" -ForegroundColor Red
    exit 1
}

Write-Host "Input video:   $Video"
Write-Host "Output JSON:   $Output"
Write-Host "Stride:        $Stride"
Write-Host "Preview video: $PreviewVideo`n"

& $pythonExe "capture_pose.py" $Video "-o" $Output "--stride" $Stride "--smooth" $Smooth "--preview-video" $PreviewVideo
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Extraction completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n[FAIL] Extraction failed with code $LASTEXITCODE" -ForegroundColor Red
}
exit $LASTEXITCODE
