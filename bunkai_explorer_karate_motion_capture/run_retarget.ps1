# PowerShell runner for blender_retarget.py
param(
    [string]$AvatarBlend = "test_data\karateka_lowpoly.blend",
    [string]$PoseJson = "test_output\captured_pose.json",
    [string]$OutputBlend = "test_output\karateka_captured.blend",
    [string]$OutputGlb = "test_output\karateka_captured.glb",
    [string]$ActionName = "captured_karate_technique"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Bunkai Explorer - Retargeting & GLB Export (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Locate Blender
$blenderCandidates = @(
    "D:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
)
$blenderExe = Get-Command blender -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $blenderExe) {
    foreach ($c in $blenderCandidates) {
        if (Test-Path $c) { $blenderExe = $c; break }
    }
}

if (-not $blenderExe) {
    Write-Host "[ERROR] Blender executable not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $PoseJson)) {
    if (Test-Path "test_data\karate_test_pose.json") {
        $PoseJson = "test_data\karate_test_pose.json"
    } else {
        Write-Host "[ERROR] Pose JSON file not found: $PoseJson" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Blender:      $blenderExe"
Write-Host "Avatar blend: $AvatarBlend"
Write-Host "Pose JSON:    $PoseJson"
Write-Host "Output blend: $OutputBlend"
Write-Host "Output GLB:   $OutputGlb`n"

& $blenderExe -b $AvatarBlend -P "blender_retarget.py" -- --pose $PoseJson --output $OutputBlend --export-glb $OutputGlb --action-name $ActionName
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Retargeting completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n[FAIL] Retargeting failed with code $LASTEXITCODE" -ForegroundColor Red
}
exit $LASTEXITCODE
