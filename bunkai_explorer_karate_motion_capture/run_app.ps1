# PowerShell runner for Bunkai MoCap Live
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Виртуальное окружение .venv не найдено: $pythonExe" -ForegroundColor Red
    exit 1
}

& $pythonExe "app.py" @args
exit $LASTEXITCODE
