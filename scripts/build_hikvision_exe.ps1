##############################################################################
# build_hikvision_exe.ps1
#
# Builds:  dist\Harmony_JD_Block_Notch_Detection\Harmony_JD_Block_Notch_Detection.exe  (+ all DLLs)
#
# Usage (from repo root):
#   .\scripts\build_hikvision_exe.ps1
#   .\scripts\build_hikvision_exe.ps1 -Weights "runs/detect/.../best.pt"
#
# The final delivery folder is:  dist\Harmony_JD_Block_Notch_Detection\
# Hand that whole folder to the client.  They just double-click Harmony_JD_Block_Notch_Detection.exe
##############################################################################

param(
    [string]$PythonExe = "",
    [string]$SpecFile  = "hikvision_capture_gui.spec",
    [string]$Weights   = "yolo26n.pt",          # default model shipped with repo
    [string]$ExampleCfg = "client_config.example.json"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
Write-Host "`n=== HIKVISION Notch Inspector – EXE Builder ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"

# ── 1. Locate Python ──────────────────────────────────────────────────────────
if (-not $PythonExe) {
    $Candidates = @(
        (Join-Path $RepoRoot ".venv_win\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\python.exe")
    )
    foreach ($c in $Candidates) {
        if (Test-Path $c) { $PythonExe = $c; break }
    }
    if (-not $PythonExe) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $PythonExe = $cmd.Source }
    }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    throw "Python not found. Pass -PythonExe 'C:\path\to\python.exe'"
}
Write-Host "Python : $PythonExe"

# ── 2. Check PyInstaller is installed ─────────────────────────────────────────
$null = & $PythonExe -m PyInstaller --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller not found – installing …" -ForegroundColor Yellow
    & $PythonExe -m pip install pyinstaller --quiet
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PyInstaller." }
}

# ── 3. Run PyInstaller using the .spec file ───────────────────────────────────
if (-not (Test-Path $SpecFile)) {
    throw "Spec file not found: $SpecFile  (run from repo root)"
}
Write-Host "`nRunning PyInstaller …  (this can take 5-15 minutes)" -ForegroundColor Yellow

& $PythonExe -m PyInstaller --noconfirm --clean $SpecFile

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed. Check output above for errors."
}

# ── 4. Copy model weights into dist folder ────────────────────────────────────
$DistDir = Join-Path $RepoRoot "dist\Harmony_JD_Block_Notch_Detection"
$ModelsDist = Join-Path $DistDir "models"

if (-not (Test-Path $DistDir)) {
    throw "dist\Harmony_JD_Block_Notch_Detection not found – PyInstaller may have failed."
}

New-Item -ItemType Directory -Path $ModelsDist -Force | Out-Null

# Try to copy the specified weights file
$WeightsPath = Join-Path $RepoRoot $Weights
if (Test-Path $WeightsPath) {
    Copy-Item $WeightsPath (Join-Path $ModelsDist "best.pt") -Force
    Write-Host "Copied model: $Weights  ->  models\best.pt" -ForegroundColor Green
} else {
    # Fallback: look for any .pt in common locations
    $PtFiles = @(
        (Get-ChildItem -Path (Join-Path $RepoRoot "runs") -Filter "best.pt" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1),
        (Get-ChildItem -Path $RepoRoot -Filter "*.pt" -ErrorAction SilentlyContinue | Select-Object -First 1)
    ) | Where-Object { $_ -ne $null } | Select-Object -First 1

    if ($PtFiles) {
        Copy-Item $PtFiles.FullName (Join-Path $ModelsDist "best.pt") -Force
        Write-Host "Copied model: $($PtFiles.Name)  ->  models\best.pt" -ForegroundColor Green
    } else {
        Write-Host "WARNING: No model file found. Copy your best.pt to dist\Harmony_JD_Block_Notch_Detection\models\best.pt manually." -ForegroundColor Yellow
    }
}

# ── 5. Copy / create client_config.json ───────────────────────────────────────
$CfgSrc = Join-Path $RepoRoot $ExampleCfg
$CfgDst = Join-Path $DistDir "client_config.json"

if (-not (Test-Path $CfgDst)) {
    if (Test-Path $CfgSrc) {
        Copy-Item $CfgSrc $CfgDst -Force
        Write-Host "Copied:  client_config.json  (edit with your camera IP / credentials)" -ForegroundColor Green
    } else {
        # Write a sensible default
        $DefaultCfg = @{
            camera_ip      = "192.168.1.64"
            camera_port    = 554
            camera_user    = "admin"
            camera_pass    = "password"
            camera_channel = 101
            rtsp_url       = ""
            weights        = "models/best.pt"
            imgsz          = 960
            conf           = 0.25
            device         = "cpu"
            save_dir       = "results/captures"
            rtsp_tcp       = $true
            auto_connect   = $false
        } | ConvertTo-Json -Depth 3
        $DefaultCfg | Set-Content $CfgDst -Encoding UTF8
        Write-Host "Created: client_config.json (edit with your camera IP / credentials)" -ForegroundColor Green
    }
}

# ── 6. Create a README.txt for the client ─────────────────────────────────────
$ReadmeDst = Join-Path $DistDir "README.txt"
$ReadmeContent = @"
=============================================================
  HIKVISION Notch Inspector  –  Quick Start Guide
=============================================================

STEP 1 – Edit  client_config.json  (open with Notepad)
  • Set  "camera_ip"    to your HIKVISION camera IP address
  • Set  "camera_port"  (default 554 for RTSP)
  • Set  "camera_user"  (default "admin")
  • Set  "camera_pass"  to your camera password
  • Set  "camera_channel" (101 = main stream, 102 = sub stream)
  • OR set  "rtsp_url"  to a full RTSP URL (overrides fields above)

STEP 2 – Double-click  NotchInspector.exe

STEP 3 – In the GUI:
  • The camera settings from client_config.json are pre-filled
  • Click  [▶ Connect]  to connect to the HIKVISION camera
  • The LEFT panel shows the live camera feed
  • Click  [📷 CAPTURE & INSPECT]  to grab a frame and run detection
  • The RIGHT panel shows the annotated result
  • VERDICT banner shows  ✅ OK  or  ❌ NOT_OKAY  in large text
  • Every capture is saved in the  results/captures/  folder

CAPTURED IMAGES are saved next to this EXE in:
  results\captures\<timestamp>_<VERDICT>.jpg

A log CSV is also written:
  results\captures\capture_log.csv

=============================================================
"@
$ReadmeContent | Set-Content $ReadmeDst -Encoding UTF8

# ── 7. Summary ────────────────────────────────────────────────────────────────
Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "  BUILD COMPLETE" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  Delivery folder:  dist\Harmony_JD_Block_Notch_Detection\"
Write-Host "  Run:              dist\Harmony_JD_Block_Notch_Detection\Harmony_JD_Block_Notch_Detection.exe"
Write-Host ""
Write-Host "  Give the client the ENTIRE  dist\Harmony_JD_Block_Notch_Detection\  folder."
Write-Host "  They just double-click  Harmony_JD_Block_Notch_Detection.exe  – no Python needed."
Write-Host ("=" * 60) -ForegroundColor Cyan
