#!/usr/bin/env bash
# Iraqi Voice Studio — Google Colab installer.
# Run inside a Colab cell:
#   !curl -fsSL https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/colab/install_colab.sh | bash
# Or after cloning the repo:
#   !bash scripts/colab/install_colab.sh
#
# Environment variables:
#   IRAQI_VOICE_REPO   override git URL (defaults to Alqudimi/VoiceCall)
#   RVC_INSTALL_DIR    clone path (default /content/VoiceCall)
#   MOUNT_DRIVE=1      mount Google Drive and wire runs/ + models/ to it (default 1)
#   DRIVE_DIR=RVC_Iraqi  folder name inside MyDrive (default RVC_Iraqi)
#   SKIP_ASSETS=1      do not download pretrained weights (default 0)
set -euo pipefail

log()  { printf '\033[1;35m[colab]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[colab-ok]\033[0m %s\n' "$*"; }

MOUNT_DRIVE="${MOUNT_DRIVE:-1}"
DRIVE_DIR="${DRIVE_DIR:-RVC_Iraqi}"
SKIP_ASSETS="${SKIP_ASSETS:-0}"
SKIP_TUNNEL="${SKIP_TUNNEL:-0}"
REPO_URL="${IRAQI_VOICE_REPO:-https://github.com/Alqudimi/VoiceCall.git}"
INSTALL_DIR="${RVC_INSTALL_DIR:-/content/VoiceCall}"

# ------------------------------------------------------------- environment --
log "detecting execution environment"
if [[ -n "${COLAB_GPU:-}" || -d /usr/local/lib/python3.*/dist-packages/google.colab  || -d /content ]]; then
  ok "Google Colab detected"
else
  log "not a Colab VM — continuing anyway (generic Linux path)"
fi
nvidia-smi -L 2>/dev/null && ok "GPU available" || log "no GPU (CPU mode only)"
python3 - <<'EOF' 2>/dev/null || true
import torch
print(f"[colab] torch={torch.__version__} cuda={torch.cuda.is_available()}")
EOF

# ---------------------------------------------------------------- project ---
if [[ -f "pyproject.toml" && -d "src/rvc_platform" ]]; then
  PROJECT_ROOT="$(pwd)"
  log "already inside the project: $PROJECT_ROOT"
elif [[ -f "$INSTALL_DIR/pyproject.toml" ]]; then
  PROJECT_ROOT="$INSTALL_DIR"
  cd "$PROJECT_ROOT"
  log "using existing checkout: $PROJECT_ROOT"
else
  [[ ! -e "$INSTALL_DIR" ]] || { echo "[colab][error] install path exists but is not VoiceCall: $INSTALL_DIR (set RVC_INSTALL_DIR)" >&2; exit 2; }
  log "cloning $REPO_URL into $INSTALL_DIR"
  git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
  PROJECT_ROOT="$INSTALL_DIR"
fi

# ---------------------------------------------------------------- drive -----
if [[ "$MOUNT_DRIVE" == "1" && -d /content/drive/MyDrive ]]; then
  ok "Google Drive already mounted"
elif [[ "$MOUNT_DRIVE" == "1" ]]; then
  log "mounting Google Drive (authorise in the popup)"
  python3 -c "from google.colab import drive; drive.mount('/content/drive')" || {
    log "drive mount failed — running without Drive persistence"; MOUNT_DRIVE=0; }
fi
DRIVE_ROOT=""
if [[ "$MOUNT_DRIVE" == "1" && -d /content/drive/MyDrive ]]; then
  DRIVE_ROOT="/content/drive/MyDrive/$DRIVE_DIR"
  mkdir -p "$DRIVE_ROOT/runs" "$DRIVE_ROOT/models" "$DRIVE_ROOT/datasets"
  ok "Drive workspace: $DRIVE_ROOT"
fi

# ---------------------------------------------------------------- Tunnel ----
if [[ "$SKIP_TUNNEL" == "0" ]]; then
  # shellcheck source=../common/install_cloudflared.sh
  source scripts/common/install_cloudflared.sh
  install_cloudflared_user "$HOME/.local/bin"
else
  log 'skipping cloudflared install (SKIP_TUNNEL=1)'
fi

# ---------------------------------------------------------------- uv + env --
command -v uv >/dev/null 2>&1 || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
log "uv $(uv --version | awk '{print $2}') — syncing environment (CPU wheels default; see below for GPU)"
uv sync --all-groups

# Colab GPU machines: install the matching CUDA variant (torch version kept)
if nvidia-smi -L >/dev/null 2>&1; then
  CU=""
  nvidia-smi | grep -qi "CUDA Version: 12.8" && CU=cu128
  nvidia-smi | grep -qi "CUDA Version: 12.4" && CU=cu124
  nvidia-smi | grep -qi "CUDA Version: 12.1" && CU=cu121
  nvidia-smi | grep -qi "CUDA Version: 11.8" && CU=cu118
  if [[ -z "$CU" ]]; then
    DRIVER_CUDA="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)"
    case "$DRIVER_CUDA" in
      12.*) CU=cu124 ;;
      11.*) CU=cu118 ;;
    esac
  fi
  [[ -n "$CU" ]] && { log "installing CUDA torch variant: $CU"; uv run rvc-platform env gpu "$CU"; } || log "could not pick a CUDA variant — using CPU wheels"
fi

# ---------------------------------------------------------------- assets ----
if [[ "$SKIP_ASSETS" == "0" ]]; then
  log "downloading pretrained assets (HuBERT / RMVPE / v2 f0G40k f0D40k)"
  uv run rvc-platform download-models
fi

# ------------------------------------------------------------ drive wiring --
if [[ -n "$DRIVE_ROOT" ]]; then
  # restore models + checkpoints from Drive into the session
  log "restoring runs/ and models/ from Drive (if present)"
  mkdir -p runs models
  rsync -a --exclude='.part' "$DRIVE_ROOT/runs/" runs/ 2>/dev/null || true
  rsync -a "$DRIVE_ROOT/models/" models/ 2>/dev/null || true
  cat > .colab_env <<EOF
# source this before training in any later cell:
export RVC_PLATFORM_ROOT="$PROJECT_ROOT"
export RVC_RUNS_DIR="$DRIVE_ROOT/runs"        # checkpoints written straight to Drive
export RVC_MODELS_DIR="$DRIVE_ROOT/models"    # exported models land in Drive
export RVC_DATA_DIR="/content/iraqi_data"     # ephemeral fast disk for datasets
EOF
  ok "wrote .colab_env — source it before running training cells"
fi

# ---------------------------------------------------------------- verify ----
log "environment verification"
uv run rvc-platform install check
[[ "$SKIP_ASSETS" == "0" ]] && uv run rvc-platform install doctor
ok "Colab setup complete"
echo
echo "  next steps (in a new cell):"
echo "    %cd $PROJECT_ROOT"
echo "    !uv run rvc-platform prepare-dataset /content/drive/MyDrive/$DRIVE_DIR/datasets/my_voice iraqi_voice"
echo "    !uv run rvc-platform train start --experiment iraqi_voice"
echo "    !uv run rvc-platform web --host 0.0.0.0 --tunnel   # protected quick tunnel"
echo "  Authentication: token saved in logs/auth.token; open Settings on the shared page and paste it."
echo
echo "  training auto-resumes the latest checkpoint on every rerun;"
echo "  checkpoints/models persist to Drive via RVC_RUNS_DIR/RVC_MODELS_DIR."
