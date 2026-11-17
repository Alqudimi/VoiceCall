#!/usr/bin/env bash
# Iraqi Voice Studio — Ubuntu / Linux installer
# Usage:  bash scripts/ubuntu/install_ubuntu.sh [--no-apt] [--skip-assets] [--skip-tunnel]
# One-line: curl -fsSL https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/ubuntu/install_ubuntu.sh | bash
# Requires: bash 4+, internet. Root not required (sudo used only for apt).
set -euo pipefail

REPO_URL="${IRAQI_VOICE_REPO:-https://github.com/Alqudimi/VoiceCall.git}"
INSTALL_DIR="${RVC_INSTALL_DIR:-$HOME/VoiceCall}"
PROJECT_ROOT=""
NO_APT=0; SKIP_ASSETS=0; SKIP_TUNNEL=0
for arg in "$@"; do
  case "$arg" in
    --no-apt) NO_APT=1 ;; --skip-assets) SKIP_ASSETS=1 ;;
    --skip-tunnel) SKIP_TUNNEL=1 ;;
    -h|--help) echo 'Usage: install_ubuntu.sh [--no-apt] [--skip-assets] [--skip-tunnel]'; exit 0 ;;
    *) echo "[error] unknown option: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 2; }

# ---------------------------------------------------------------- system ----
log "detecting system"
. /etc/os-release 2>/dev/null || { warn "/etc/os-release missing — not Ubuntu?"; }
log "OS: ${PRETTY_NAME:-unknown} | arch: $(uname -m)"
if [[ "$(uname -m)" != "x86_64" && "$(uname -m)" != "aarch64" ]]; then
  warn "untested architecture: $(uname -m)"
fi
log "kernel: $(uname -r) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
log "disk free: $(df -h "$PROJECT_ROOT" | awk 'NR==2{print $4}')"

SUDO=""
if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# ---------------------------------------------------------------- apt pkgs --
if [[ $NO_APT -eq 0 ]]; then
  log "ensuring Ubuntu system prerequisites"
  $SUDO apt-get update -y
  $SUDO apt-get install -y git curl ca-certificates ffmpeg make build-essential ninja-build cmake pkg-config libsndfile1 libportaudio2 python3.12-dev
else
  log "skipping apt (--no-apt)"
fi
for bin in git curl ffmpeg make; do
  command -v "$bin" >/dev/null 2>&1 || warn "$bin still missing — install it manually"
done

# A curl|bash process has no script location. Use the current checkout when
# present; otherwise clone safely without deleting an existing directory.
if [[ -f "$PWD/pyproject.toml" && -d "$PWD/src/rvc_platform" ]]; then
  PROJECT_ROOT="$(pwd -P)"
elif [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$SCRIPT_DIR/../../pyproject.toml" ]]; then PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"; fi
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  command -v git >/dev/null 2>&1 || fail 'git is required to clone the source repository.'
  if [[ -f "$INSTALL_DIR/pyproject.toml" ]]; then
    PROJECT_ROOT="$(cd "$INSTALL_DIR" && pwd -P)"
  elif [[ -e "$INSTALL_DIR" ]]; then
    fail "Install path exists but is not VoiceCall: $INSTALL_DIR (set RVC_INSTALL_DIR)."
  else
    log "cloning $REPO_URL into $INSTALL_DIR"
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
    PROJECT_ROOT="$(cd "$INSTALL_DIR" && pwd -P)"
  fi
fi
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------- python ----
PY_OK=0
if command -v python3 >/dev/null 2>&1; then
  PYV="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:2])))')"
  [[ "$PYV" == "3.12" ]] && PY_OK=1 || log "system python is $PYV (uv will provide 3.12)"
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

# ---------------------------------------------------------------- tunnel ---
if [[ $SKIP_TUNNEL -eq 0 ]]; then
  # shellcheck source=../common/install_cloudflared.sh
  source scripts/common/install_cloudflared.sh
  install_cloudflared_user "$HOME/.local/bin"
else
  warn 'Skipping cloudflared install (--skip-tunnel).'
fi

# ---------------------------------------------------------------- project ---
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
echo "  expose via tunnel:         make tunnel (after starting web; authentication is enabled automatically)"
echo "  auth token:                cat logs/auth.token; enter it in Settings from the public page"
