# Iraqi Voice Studio — Windows installer (PowerShell)
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\windows\install_windows.ps1
#         [-SkipAssets] [-GpuVariant cu124|cu121|cu118|cu128|none]
[CmdletBinding()]
param(
  [switch]$SkipAssets = $false,
  [string]$GpuVariant = "none"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

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

Ensure-Command "git"  "Git.Git"    "install Git for Windows from https://git-scm.com"
Ensure-Command "ffmpeg" "Gyan.FFmpeg" "install ffmpeg (needed for MP3/M4A uploads; WAV works without it)"
Ensure-Command "cloudflared" "Cloudflare.cloudflared" "optional — only needed for make tunnel"

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
Write-Host "  tunnel:                uv run rvc-platform web --tunnel  (auth token auto-enables)"
