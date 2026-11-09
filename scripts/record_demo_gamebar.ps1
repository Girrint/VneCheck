$ErrorActionPreference = 'Stop'

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$serverScript = Join-Path $workspace 'scripts\serve_web.py'
$stripAudioScript = Join-Path $workspace 'scripts\strip_video_audio.py'
$chrome = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$profile = Join-Path $workspace '.codex_tmp\video-demo\gamebar-chrome-profile'
$outputDir = Join-Path $workspace 'outputs\video_demo_vnecheck_e30'
$rawOutput = Join-Path $outputDir 'vnecheck_live_demo_gamebar_raw.mp4'
$finalOutput = Join-Path $outputDir 'vnecheck_live_demo_gamebar_no_audio.mp4'
$captureDir = Join-Path $env:USERPROFILE 'Videos\Captures'

foreach ($required in @($python, $serverScript, $stripAudioScript, $chrome)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required file: $required" }
}
New-Item -ItemType Directory -Path $profile, $outputDir -Force | Out-Null

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class VneCheckKeyboardInput {
    [DllImport("user32.dll", SetLastError = true)]
    public static extern void keybd_event(byte virtualKey, byte scanCode, uint flags, UIntPtr extraInfo);
}
'@

function Send-KeyChord([byte[]]$Keys) {
    foreach ($key in $Keys) {
        [VneCheckKeyboardInput]::keybd_event($key, 0, 0, [UIntPtr]::Zero)
    }
    Start-Sleep -Milliseconds 120
    for ($index = $Keys.Length - 1; $index -ge 0; $index--) {
        [VneCheckKeyboardInput]::keybd_event($Keys[$index], 0, 2, [UIntPtr]::Zero)
    }
}

function Activate-DemoWindow {
    $shell = New-Object -ComObject WScript.Shell
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        $window = Get-Process chrome -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowTitle -match 'VnECheck|VNECHECK' } |
            Sort-Object StartTime -Descending |
            Select-Object -First 1
        if ($window -and $shell.AppActivate($window.Id)) { return $window }
        Start-Sleep -Milliseconds 500
    }
    throw 'VnECheck Chrome window was not found. Run this script from the signed-in Windows desktop.'
}

function Stop-VneCheckServer {
    $listeners = netstat -ano | Select-String '127.0.0.1:8080\s+0.0.0.0:0\s+LISTENING\s+(\d+)'
    foreach ($listener in $listeners) {
        if ($listener.Matches.Count) {
            Stop-Process -Id ([int]$listener.Matches[0].Groups[1].Value) -Force -ErrorAction SilentlyContinue
        }
    }
}

$recordStart = Get-Date
$server = $null
$window = $null
try {
    Stop-VneCheckServer
    $server = Start-Process -FilePath $python -ArgumentList @($serverScript, '--host', '127.0.0.1', '--port', '8080') -WindowStyle Hidden -PassThru
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8080' -TimeoutSec 1
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        }
        catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw 'VnECheck web server did not start.' }

    $url = 'http://127.0.0.1:8080/readme_view.html?record=1'
    Start-Process -FilePath $chrome -ArgumentList @(
        '--new-window',
        '--start-maximized',
        "--user-data-dir=$profile",
        "--app=$url"
    )
    $window = Activate-DemoWindow
    Start-Sleep -Seconds 3

    Write-Host 'Opening Xbox Game Bar...'
    Send-KeyChord ([byte[]](0x5B, 0x47))
    Start-Sleep -Seconds 3
    Send-KeyChord ([byte[]](0x5B, 0x47))
    Start-Sleep -Seconds 2

    $window = Activate-DemoWindow
    Write-Host 'Starting Game Bar recording...'
    Send-KeyChord ([byte[]](0x5B, 0xA4, 0x52))

    $done = $false
    for ($elapsed = 0; $elapsed -lt 600; $elapsed += 2) {
        Start-Sleep -Seconds 2
        $doneWindow = Get-Process chrome -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowTitle -match 'VNECHECK_DEMO_DONE' } |
            Select-Object -First 1
        if ($doneWindow) { $done = $true; $window = $doneWindow; break }
        if (($elapsed % 20) -eq 0) { Write-Host "Recording: $elapsed seconds" }
    }
    if (-not $done) { throw 'The demo did not finish before timeout.' }

    Start-Sleep -Seconds 8
    $shell = New-Object -ComObject WScript.Shell
    [void]$shell.AppActivate($window.Id)
    Start-Sleep -Seconds 1
    Write-Host 'Stopping Game Bar recording...'
    Send-KeyChord ([byte[]](0x5B, 0xA4, 0x52))
    Start-Sleep -Seconds 10

    $capture = Get-ChildItem -LiteralPath $captureDir -Filter '*.mp4' -File -ErrorAction Stop |
        Where-Object { $_.LastWriteTime -ge $recordStart.AddSeconds(-5) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $capture) {
        throw "No new Game Bar capture was found in $captureDir. Open Win+G and check Capture permissions."
    }

    Copy-Item -LiteralPath $capture.FullName -Destination $rawOutput -Force
    Write-Host 'Removing the audio stream...'
    & $python $stripAudioScript $rawOutput $finalOutput
    if ($LASTEXITCODE -ne 0) { throw 'The no-audio re-encode failed.' }

    Write-Host ''
    Write-Host 'Completed:' -ForegroundColor Green
    Write-Host $finalOutput -ForegroundColor Green
}
finally {
    Stop-VneCheckServer
    if ($server) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
}
