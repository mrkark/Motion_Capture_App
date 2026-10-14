# Root PowerShell wrapper to launch app.py
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$innerScript = Join-Path $scriptDir "bunkai_explorer_karate_motion_capture\run_app.ps1"
& $innerScript @args
exit $LASTEXITCODE
