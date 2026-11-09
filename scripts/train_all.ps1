$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$ultralyticsConfig = Join-Path $projectRoot ".ultralytics"

Set-Location $projectRoot
New-Item -ItemType Directory -Force -Path $ultralyticsConfig | Out-Null
$env:YOLO_CONFIG_DIR = $ultralyticsConfig
& $python -m vnecheck.train_all @args
if ($LASTEXITCODE -ne 0) {
    throw "VneCheck automated training failed with exit code $LASTEXITCODE"
}
