param(
    [string]$PythonExe = "",
    [string]$AppName = "Harmony_JD_Block_Notch_Detection",
    [string]$EntryPoint = "src/gui_detector.py",
    [string]$Weights = "runs/detect/runs/notch_face_v2/weights/best.pt"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if (-not $PythonExe) {
    $Candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv_win\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\python.exe")
    )

    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            $PythonExe = $Candidate
            break
        }
    }

    if (-not $PythonExe) {
        $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($PythonCmd) {
            $PythonExe = $PythonCmd.Source
        }
    }
}

if (-not $PythonExe) {
    throw "Python not found. Pass -PythonExe explicitly."
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $EntryPoint)) {
    throw "Entry script not found: $EntryPoint"
}

$PyInstallerCheck = & $PythonExe -m PyInstaller --version 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed for $PythonExe. Run: python -m pip install pyinstaller"
}

$PyArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--name", $AppName,
    "--onedir",
    "--windowed",
    "--collect-all", "ultralytics",
    "--collect-all", "cv2",
    "--collect-all", "torch",
    "--collect-all", "torchvision",
    "--collect-all", "PIL",
    "--collect-submodules", "numpy",
    $EntryPoint
)

Write-Host "Building GUI EXE..."
& $PythonExe @PyArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$DistDir = Join-Path $RepoRoot ("dist\" + $AppName)
if (-not (Test-Path $DistDir)) {
    throw "Expected output folder not found: $DistDir"
}

$ModelDir = Join-Path $DistDir "models"
New-Item -Path $ModelDir -ItemType Directory -Force | Out-Null

if (Test-Path $Weights) {
    Copy-Item -Path $Weights -Destination (Join-Path $ModelDir "best.pt") -Force
    Write-Host "Copied model -> $ModelDir\best.pt"
}
else {
    Write-Warning "Weights not found: $Weights. Copy best.pt manually into $ModelDir."
}

$ConfigSrc = Join-Path $RepoRoot "client_config.example.json"
if (Test-Path $ConfigSrc) {
    Copy-Item -Path $ConfigSrc -Destination (Join-Path $DistDir "client_config.example.json") -Force
    $DefaultCfg = Join-Path $DistDir "client_config.json"
    if (-not (Test-Path $DefaultCfg)) {
        Copy-Item -Path $ConfigSrc -Destination $DefaultCfg -Force
    }
}

$ReadmePath = Join-Path $DistDir "README_QUICK_START.txt"
@"
1) Edit client_config.json and set your HIKVISION RTSP URL.
2) Double-click $AppName.exe
3) You will see live camera + detection directly in GUI window.
4) Saved images and logs go to results\live
"@ | Set-Content -Path $ReadmePath -Encoding ASCII

Write-Host ""
Write-Host "Build complete."
Write-Host "Deliver this folder to client:"
Write-Host "  $DistDir"
