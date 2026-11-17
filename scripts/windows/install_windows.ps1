# Iraqi Voice Studio — Windows installer (PowerShell)
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\windows\install_windows.ps1
#         [-SkipAssets] [-SkipTunnel] [-RepoUrl <git-url>] [-InstallDir <path>]
# Remote bootstrap: fetch this script from raw.githubusercontent.com and run it.
[CmdletBinding()]
param(
  [switch]$SkipAssets = $false,
  [switch]$SkipTunnel = $false,
  [string]$RepoUrl = "https://github.com/Alqudimi/VoiceCall.git",
  [string]$InstallDir = (Join-Path $env:USERPROFILE "VoiceCall"),
  [string]$GpuVariant = "none"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = $null

function Log($msg)  { Write-Host "[install] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "[ok] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- system ----
Log "Windows $([System.Environment]::OSVersion.Version) | PS $($PSVersionTable.PSVersion) | arch $env:PROCESSOR_ARCHITECTURE"
if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") { Warn "only x64 is tested" }

# ---------------------------------------------------------------- winget ----
function Ensure-Command($name, $wingetId, $hint) {
  if (Get-Command $name -ErrorAction SilentlyContinue) { Ok "$name present"; return $true }
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Log "installing $name via winget ($wingetId)"
    winget install --id $wingetId -e --accept-source-agreements --accept-package-agreements
    # refresh PATH for the current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    if (Get-Command $name -ErrorAction SilentlyContinue) { Ok "$name installed"; return $true }
  }
  Warn "$name missing — $hint"
  return $false
}

if (-not (Ensure-Command "git" "Git.Git" "install Git for Windows from https://git-scm.com")) {
  throw "Git is required to download the project source. Install Git and rerun this installer."
}
Ensure-Command "ffmpeg" "Gyan.FFmpeg" "install ffmpeg (needed for MP3/M4A uploads; WAV works without it)"

# Running this script from a checkout and downloading this raw .ps1 are both supported.
$CandidateRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($CandidateRoot -and (Test-Path (Join-Path $CandidateRoot "pyproject.toml"))) {
  $ProjectRoot = (Resolve-Path $CandidateRoot).Path
} elseif (Test-Path (Join-Path $InstallDir "pyproject.toml")) {
  $ProjectRoot = (Resolve-Path $InstallDir).Path
} else {
  if (Test-Path $InstallDir) { throw "Install path exists but is not VoiceCall: $InstallDir (choose -InstallDir)." }
  Log "cloning $RepoUrl into $InstallDir"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $InstallDir) | Out-Null
  git clone --depth 1 $RepoUrl $InstallDir
  if ($LASTEXITCODE -ne 0) { throw "git clone failed with exit code $LASTEXITCODE" }
  $ProjectRoot = (Resolve-Path $InstallDir).Path
}

if (-not $SkipTunnel) {
  $CloudBinDir = Join-Path $env:LOCALAPPDATA "IraqiVoiceStudio\bin"
  $CloudBinary = Join-Path $CloudBinDir "cloudflared.exe"
  if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue) -and -not (Test-Path $CloudBinary)) {
    New-Item -ItemType Directory -Force -Path $CloudBinDir | Out-Null
    $CloudTmp = Join-Path $CloudBinDir "cloudflared.download.exe"
    Log "downloading official Cloudflare Windows x64 executable"
    Invoke-WebRequest -UseBasicParsing "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $CloudTmp
    & $CloudTmp --version
    if ($LASTEXITCODE -ne 0) { Remove-Item $CloudTmp -Force; throw "Downloaded cloudflared failed its version check." }
    Move-Item -Force $CloudTmp $CloudBinary
  }
  $UserPath = [string][System.Environment]::GetEnvironmentVariable("Path", "User")
  if (($UserPath -split ';') -notcontains $CloudBinDir) {
    $UserPath = $UserPath.TrimEnd(';')
    $NewUserPath = if ($UserPath) { "$UserPath;$CloudBinDir" } else { $CloudBinDir }
    [System.Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
  }
  $env:Path = "$CloudBinDir;$env:Path"
  if (Test-Path $CloudBinary) { & $CloudBinary --version }
} else {
  Warn "cloudflared skipped; tunnel requires manual installation."
}

# ---------------------------------------------------------------- python ----
# uv manages Python 3.12 itself; a system python is NOT required.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Log "installing uv (official astral installer)"
  $installPs1 = Invoke-WebRequest -UseBasicParsing "https://astral.sh/uv/install.ps1"
  & ([scriptblock]::Create($installPs1.Content))
  $uvDir = Join-Path $env:USERPROFILE ".local\bin"
  if (Test-Path $uvDir) { $env:Path = "$uvDir;$env:Path" }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Warn "uv still unavailable — install from https://docs.astral.sh/uv/ then re-run"
  exit 2
}
Ok "uv $(uv --version)"
Log "ensuring Python 3.12 (uv-managed)"
uv python install 3.12

# ---------------------------------------------------------------- gpu -------
$HasGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $smi = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
  if ($smi) { $HasGpu = $true; Log "NVIDIA GPU: $($smi -join ' | ')" }
}
if (-not $HasGpu) {
  Warn "no NVIDIA GPU detected — CPU mode: training VERY slow, realtime unusable"
} elseif ($GpuVariant -eq "none") {
  Log "GPU found; to install CUDA torch run: make env-gpu CUDA=cu124"
  Log "(or re-run this script with -GpuVariant cu124)"
}

# ---------------------------------------------------------------- project ---
Push-Location $ProjectRoot
Log "syncing python environment (CPU wheels default)"
uv sync --all-groups
if ($HasGpu -and $GpuVariant -ne "none") {
  Log "installing CUDA torch variant: $GpuVariant"
  uv run rvc-platform env gpu $GpuVariant
}

# ---------------------------------------------------------------- assets ----
if (-not $SkipAssets) {
  Log "downloading pretrained assets (~600 MB)"
  uv run rvc-platform download-models
} else {
  Log "skipping assets; run: uv run rvc-platform download-models"
}

# ---------------------------------------------------------------- verify ----
Log "verifying installation"
uv run rvc-platform install check
if (-not $SkipAssets) { uv run rvc-platform install doctor }
Pop-Location
Ok "installation complete"
Write-Host ""
Write-Host "  start web interface:   uv run rvc-platform web     (or: make web)"
Write-Host "  open:                  http://127.0.0.1:8320"
Write-Host "  prepare dataset:       uv run rvc-platform prepare-dataset data\raw\my_voice iraqi_voice"
Write-Host "  train:                 uv run rvc-platform train start --dataset data\raw\my_voice --experiment iraqi_voice"
Write-Host "  tunnel:                uv run rvc-platform web --tunnel  (authentication turns on automatically)"
Write-Host "  auth token:            logs\auth.token — paste into Settings on the public page"
