param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$taskTemp = Join-Path $projectRoot ".tmp"

New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $projectRoot ".venv")
}

if (-not (& $venvPython -m pip --version 2>$null)) {
    $bundledPip = Get-ChildItem `
        "C:\Users\Admin\AppData\Local\Programs\Python\Python312\Lib\ensurepip\_bundled\pip-*.whl" `
        | Select-Object -First 1
    if (-not $bundledPip) {
        throw "pip is unavailable in the virtual environment and no bundled wheel was found."
    }
    python -m pip --python (Join-Path $projectRoot ".venv") install --no-index $bundledPip.FullName
}

& $venvPython -m pip install torch torchvision --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) {
    throw "Unable to install CUDA PyTorch from $TorchIndexUrl"
}
& $venvPython -m pip install -e "$projectRoot[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to install VneCheck project dependencies"
}
& $venvPython (Join-Path $projectRoot "scripts\verify_environment.py")
