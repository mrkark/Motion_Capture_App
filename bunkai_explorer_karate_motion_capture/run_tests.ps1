# PowerShell script to run automated test pipeline
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Bunkai Explorer - Running Automated Test Suite (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment .venv not found at: $pythonExe" -ForegroundColor Red
    exit 1
}

& $pythonExe "test_pipeline.py" @args
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] All pipeline tests passed!" -ForegroundColor Green
} else {
    Write-Host "`n[FAIL] Pipeline tests failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
exit $LASTEXITCODE
