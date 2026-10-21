<#
.SYNOPSIS
    Установка зависимостей video-annotator.

.DESCRIPTION
    Ставит ffmpeg (через winget, если доступен), скачивает whisper.cpp
    (CPU и/или CUDA) и модель ggml. Тяжёлые файлы не попадают в git —
    они исключены в .gitignore.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Model large-v3 -Gpu
#>
param(
    [ValidateSet("tiny", "base", "small", "medium", "large-v3")]
    [string]$Model = "small",
    [switch]$Gpu,
    [switch]$SkipFfmpeg
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolsDir = Join-Path $Root "tools"
$ModelsDir = Join-Path $Root "models"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "  $msg" -ForegroundColor Gray }

Write-Step "Проверка Python"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "  Python не найден. Установите Python 3.10+ с сайта python.org" -ForegroundColor Red
    exit 1
}
Write-Ok "Python: $(& python --version 2>&1)"

Write-Step "ffmpeg"
if ($SkipFfmpeg) {
    Write-Info "пропущен по запросу"
} elseif (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Ok "уже установлен"
} elseif (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Info "устанавливаю через winget..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements | Out-Null
    Write-Ok "установлен (перезапустите оболочку, чтобы обновить PATH)"
} else {
    Write-Info "winget недоступен. Скачайте ffmpeg с ffmpeg.org и добавьте в PATH"
}

# --- whisper.cpp -------------------------------------------------------------
$whisperUrl = "https://github.com/ggml-org/whisper.cpp/releases/latest"

function Get-WhisperArchive {
    param([string]$Pattern)

    $json = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=20" `
        -Headers @{ "User-Agent" = "video-annotator-setup" }
    foreach ($rel in $json) {
        foreach ($asset in $rel.assets) {
            if ($asset.name -like $Pattern) {
                return [pscustomobject]@{ Url = $asset.browser_download_url; Tag = $rel.tag_name }
            }
        }
    }
    return $null
}

function Install-Whisper {
    param([string]$Pattern, [string]$Target)

    if (Test-Path $Target) {
        Write-Ok "уже скачано: $Target"
        return
    }
    $asset = Get-WhisperArchive -Pattern $Pattern
    if (-not $asset) {
        Write-Info "не нашёл релиз с '$Pattern'"
        return
    }
    Write-Info "скачиваю $($asset.Url)"
    $zip = Join-Path $env:TEMP "whisper.zip"
    Invoke-WebRequest -Uri $asset.Url -OutFile $zip -UseBasicParsing
    $dest = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Remove-Item $zip -ErrorAction SilentlyContinue
    Write-Ok "распаковано в $dest"
}

Write-Step "whisper.cpp (CPU)"
Install-Whisper -Pattern "whisper-bin-x64.zip" -Target (Join-Path $ToolsDir "whisper")

if ($Gpu) {
    Write-Step "whisper.cpp (CUDA)"
    Install-Whisper -Pattern "whisper-bin-win-cuda-*-x64.zip" -Target (Join-Path $ToolsDir "whisper-cuda")
}

# --- модель ------------------------------------------------------------------
Write-Step "Модель whisper: $Model"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
$modelFile = "ggml-$Model.bin"
$modelPath = Join-Path $ModelsDir $modelFile
if (Test-Path $modelPath) {
    Write-Ok "уже скачана"
} else {
    $url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$modelFile"
    Write-Info "скачиваю $url (может занять время)"
    Invoke-WebRequest -Uri $url -OutFile $modelPath -UseBasicParsing
    Write-Ok "скачана: $modelPath"
}

Write-Step "Готово"
Write-Host "  Проверка: python selftest.py" -ForegroundColor Cyan
Write-Host "  Запуск:   python annotate.py <video.mp4> -d 3" -ForegroundColor Cyan
