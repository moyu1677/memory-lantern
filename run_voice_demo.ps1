[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$AppPort = 5000,

    [ValidateRange(1024, 65535)]
    [int]$TtsPort = 15000,

    [ValidateRange(1024, 65535)]
    [int]$AsrPort = 15001,

    [switch]$NoBrowser,

    # Used only for automated local verification.  It leaves the already
    # health-checked web app and controlled tunnel running after this launcher
    # exits, so a follow-up test can inspect the live demo without a visible
    # PowerShell window.
    [switch]$DetachAfterHealth,

    [string]$SshHost = 'teacher-server'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$deepseekKeyFile = Get-ChildItem -LiteralPath $projectRoot -Filter 'deepseek*.txt' -File |
    Select-Object -First 1 -ExpandProperty FullName
$tunnel = $null
$appProcess = $null

function Wait-ForTunnelHealth {
    param(
        [string]$ServiceName,
        [string]$Uri,
        [System.Diagnostics.Process]$Tunnel,
        [int]$TimeoutSeconds = 20,
        [switch]$RequireModelLoaded
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = 'No response yet.'
    while ((Get-Date) -lt $deadline) {
        if ($Tunnel.HasExited) {
            throw "$ServiceName process exited before it became reachable (exit code $($Tunnel.ExitCode))."
        }
        try {
            $response = Invoke-RestMethod -Uri $Uri -TimeoutSec 3
            if (-not $RequireModelLoaded -or $response.model_loaded) {
                return $response
            }
            $lastError = 'Service is reachable, but its model is still loading.'
        } catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$ServiceName did not become ready within $TimeoutSeconds seconds. Last error: $lastError"
}

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project virtual environment is missing. Install the project dependencies first.'
}

# Keys are kept only in this PowerShell process; they are never written to .env.
if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    if ([string]::IsNullOrWhiteSpace($deepseekKeyFile) -or -not (Test-Path -LiteralPath $deepseekKeyFile)) {
        throw 'DeepSeek key is missing. Set DEEPSEEK_API_KEY or add a deepseek*.txt key file.'
    }
    $env:DEEPSEEK_API_KEY = (Get-Content -LiteralPath $deepseekKeyFile -Raw).Trim()
}

if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    throw 'DEEPSEEK_API_KEY is empty.'
}

if (Get-NetTCPConnection -State Listen -LocalPort $TtsPort -ErrorAction SilentlyContinue) {
    throw "Local port $TtsPort is already in use. Choose another -TtsPort."
}
if (Get-NetTCPConnection -State Listen -LocalPort $AsrPort -ErrorAction SilentlyContinue) {
    throw "Local port $AsrPort is already in use. Choose another -AsrPort."
}

Write-Host 'Opening a controlled SSH tunnel to teacher-server IndexTTS2...'

# Read each remote service key from its own running directory; keep both only in memory.
$remoteVoiceKeyCommand = 'pid=$(pgrep -fo "remote_indextts_server.py"); test -n "$pid"; dir=$(readlink -f "/proc/$pid/cwd"); sed -n "s/^VOICE_API_KEY=//p" "$dir/.env" | head -n 1'
$remoteAsrKeyCommand = 'pid=$(pgrep -fo "remote_whisper_server.py"); test -n "$pid"; dir=$(readlink -f "/proc/$pid/cwd"); sed -n "s/^ASR_API_KEY=//p" "$dir/.env" | head -n 1'
$remoteVoiceApiKey = (& ssh $SshHost $remoteVoiceKeyCommand).Trim()
if ([string]::IsNullOrWhiteSpace($remoteVoiceApiKey)) {
    throw 'Remote voice credential was unavailable. Confirm IndexTTS2 is running on teacher-server.'
}
$remoteAsrApiKey = (& ssh $SshHost $remoteAsrKeyCommand).Trim()
if ([string]::IsNullOrWhiteSpace($remoteAsrApiKey)) {
    throw 'Remote ASR credential was unavailable. Confirm Whisper ASR is running on teacher-server.'
}

$forwardSpec = '{0}:127.0.0.1:5000' -f $TtsPort
$asrForwardSpec = '{0}:127.0.0.1:8000' -f $AsrPort

try {
    # One SSH process forwards both local ports. ExitOnForwardFailure prevents a
    # hidden tunnel from appearing healthy when either port could not bind.
    $tunnel = Start-Process -FilePath 'ssh' -ArgumentList @(
        '-N',
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ConnectTimeout=12',
        '-L', $forwardSpec,
        '-L', $asrForwardSpec,
        $SshHost
    ) -WindowStyle Hidden -PassThru

    $health = Wait-ForTunnelHealth 'IndexTTS2' "http://127.0.0.1:$TtsPort/health" $tunnel -TimeoutSeconds 180 -RequireModelLoaded
    if ($health.status -ne 'ok') {
        throw 'Remote voice service is reachable but IndexTTS2 is not ready.'
    }
    $asrHealth = Wait-ForTunnelHealth 'ASR' "http://127.0.0.1:$AsrPort/health" $tunnel
    if (-not $asrHealth.status) {
        throw 'Remote ASR service is not ready.'
    }

    $env:INDEXTTS2_URL = "http://127.0.0.1:$TtsPort"
    $env:INDEXTTS2_API_KEY = $remoteVoiceApiKey
    $env:INDEXTTS2_TIMEOUT_SECONDS = '180'
    $env:ASR_URL = "http://127.0.0.1:$AsrPort"
    $env:ASR_API_KEY = $remoteAsrApiKey
    $env:ASR_TIMEOUT_SECONDS = '90'
    $env:FLASK_HOST = '127.0.0.1'
    $env:FLASK_PORT = "$AppPort"
    $env:FLASK_DEBUG = 'false'
    $env:FLASK_SECRET_KEY = [guid]::NewGuid().ToString('N')

    $demoUrl = "http://127.0.0.1:$AppPort"
    Write-Host "Voice companion demo: $demoUrl"
    Write-Host 'Keep this window open. Press Ctrl+C to stop the demo and SSH tunnel.'
    $appProcess = Start-Process -FilePath $python -ArgumentList @(
        (Join-Path $projectRoot 'app.py')
    ) -NoNewWindow -PassThru

    # The audio demo has no account gateway: health-check its public family page.
    Wait-ForTunnelHealth 'Voice companion web server' "$demoUrl/family" $appProcess | Out-Null
    if (-not $NoBrowser) {
        Start-Process $demoUrl
    }
    if ($DetachAfterHealth) {
        Write-Host 'Voice companion demo is healthy and running in the background.'
        # Prevent the finally block from terminating the verified child
        # processes.  Their explicit PIDs remain discoverable by their ports.
        $appProcess = $null
        $tunnel = $null
        return
    }
    Wait-Process -Id $appProcess.Id
} finally {
    if ($null -ne $appProcess -and -not $appProcess.HasExited) {
        Stop-Process -Id $appProcess.Id -ErrorAction SilentlyContinue
    }
    if ($null -ne $tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
    }
}
