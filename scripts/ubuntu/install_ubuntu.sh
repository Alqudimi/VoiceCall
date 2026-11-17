#!/usr/bin/env bash
# Iraqi Voice Studio — Ubuntu / Linux installer
# Usage:  bash scripts/ubuntu/install_ubuntu.sh [--no-apt] [--skip-assets]
# Requires: bash 4+, internet. Root not required (sudo used only for apt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NO_APT=0; SKIP_ASSETS=0
for arg in "$@"; do
  case "$arg" in
    --no-apt) NO_APT=1 ;; --skip-assets) SKIP_ASSETS=1 ;;
  esac
done

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- system ----
log "detecting system"
. /etc/os-release 2>/dev/null || { warn "/etc/os-release missing — not Ubuntu?"; }
log "OS: ${PRETTY_NAME:-unknown} | arch: $(uname -m)"
# enforce a supported Ubuntu base (20.04+) unless --any-ubuntu is passed
ANY_UBUNTU=0
for arg in "$@"; do [[ "$arg" == "--any-ubuntu" ]] && ANY_UBUNTU=1; done
if [[ "${ID:-}" == "ubuntu" ]]; then
  UBUNTU_MAJOR="${VERSION_ID%%.*}"
  if [[ -n "$UBUNTU_MAJOR" && "$UBUNTU_MAJOR" =~ ^[0-9]+$ && "$UBUNTU_MAJOR" -ge 20 ]]; then
    ok "Ubuntu ${VERSION_ID} supported"
  elif [[ $ANY_UBUNTU -eq 1 ]]; then
    warn "Ubuntu ${VERSION_ID:-unknown} older than 20.04 — continuing (--any-ubuntu)"
  else
    warn "Ubuntu ${VERSION_ID:-unknown} is below the supported floor (20.04);"
    warn "rerun with --any-ubuntu to force the install anyway"
    exit 3
  fi
fi
if [[ "$(uname -m)" != "x86_64" && "$(uname -m)" != "aarch64" ]]; then
  warn "untested architecture: $(uname -m)"
fi
log "kernel: $(uname -r) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
log "disk free: $(df -h "$PROJECT_ROOT" | awk 'NR==2{print $4}')"

SUDO=""
if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# ---------------------------------------------------------------- apt pkgs --
if [[ $NO_APT -eq 0 ]]; then
  NEED_PKGS=()
  for bin in git curl ffmpeg make gcc; do
    command -v "$bin" >/dev/null 2>&1 || NEED_PKGS+=("$bin")
  done
  if [[ ${#NEED_PKGS[@]} -gt 0 ]]; then
    log "installing missing system packages: ${NEED_PKGS[*]}"
    $SUDO apt-get update -y
    # gcc lives in build-essential; map binary names to packages
    $SUDO apt-get install -y git curl ffmpeg make build-essential
  else
    ok "all system tools present"
  fi
else
  log "skipping apt (--no-apt)"
fi
for bin in git curl ffmpeg make; do
  command -v "$bin" >/dev/null 2>&1 || warn "$bin still missing — install it manually"
done

# ---------------------------------------------------------------- python ----
if command -v python3 >/dev/null 2>&1; then
  PYV="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:2])))')"
  log "system python is $PYV (uv pins the project to 3.12 regardless)"
else
  log "no system python3 — uv will provide 3.12"
fi

# ---------------------------------------------------------------- uv --------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv (official astral installer)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version | awk '{print $2}')"

# uv guarantees Python 3.12 for the project (pyproject pins >=3.12,<3.13)
log "ensuring Python 3.12 via uv"
uv python install 3.12 2>/dev/null || true

# ---------------------------------------------------------------- gpu -------
GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  GPU=1
  log "NVIDIA GPU found: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  CUDA_RT=""
  command -v nvcc >/dev/null 2>&1 && CUDA_RT="$(nvcc --version | tail -1)"
  log "CUDA toolkit: ${CUDA_RT:-not installed (runtime wheels bundle their own CUDA)}"
else
  warn "no NVIDIA GPU detected — CPU mode: training VERY slow, realtime unusable"
fi

# ---------------------------------------------------------------- project ---
cd "$PROJECT_ROOT"
log "syncing python environment (CPU wheels by default; GPU: make env-gpu CUDA=cu124)"
uv sync --all-groups
if [[ $GPU -eq 1 ]]; then
  if [[ -n "${CUDA_VARIANT:-}" ]]; then
    uv run rvc-platform env gpu "$CUDA_VARIANT"
  else
    log "GPU detected — install CUDA torch anytime with: make env-gpu CUDA=cu124"
  fi
fi

# ---------------------------------------------------------------- assets ----
if [[ $SKIP_ASSETS -eq 0 ]]; then
  log "downloading pretrained assets (HuBERT / RMVPE / v2 base G,D — ~600 MB)"
  uv run rvc-platform download-models
else
  log "skipping assets (--skip-assets); run: uv run rvc-platform download-models"
fi

# ---------------------------------------------------------------- verify ----
log "verifying installation"
uv run rvc-platform install check
[[ $SKIP_ASSETS -eq 0 ]] && uv run rvc-platform install doctor
ok "installation complete"
echo
echo "  start the web interface:   make web        (or: uv run rvc-platform web)"
echo "  open:                      http://127.0.0.1:8320"
echo "  prepare Iraqi dataset:     make prepare-dataset DATASET=data/raw/my_voice EXPERIMENT=iraqi_voice"
echo "  train:                     make train DATASET=data/raw/my_voice EXPERIMENT=iraqi_voice"
echo "  expose via tunnel:         make web TUNNEL=1  (auth token auto-enables)"
