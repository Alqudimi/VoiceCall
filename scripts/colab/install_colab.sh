#!/usr/bin/env bash
# Iraqi Voice Studio — Google Colab installer.
# Run inside a Colab cell (curl-bash, works out of the box):
#   !curl -fsSL https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/colab/install_colab.sh | bash
# Or after cloning the repo:
#   !bash scripts/colab/install_colab.sh
#
# Environment variables:
#   IRAQI_VOICE_REPO   git URL of the project (default: the public repository;
#                      override for forks; clone target /content/iraqi-voice-platform)
#   MOUNT_DRIVE=1      mount Google Drive and wire runs/ + models/ to it (default 1)
#   DRIVE_DIR=RVC_Iraqi  folder name inside MyDrive (default RVC_Iraqi)
#   SKIP_ASSETS=1      do not download pretrained weights (default 0)
set -euo pipefail

log()  { printf '\033[1;35m[colab]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[colab-ok]\033[0m %s\n' "$*"; }

MOUNT_DRIVE="${MOUNT_DRIVE:-1}"
DRIVE_DIR="${DRIVE_DIR:-RVC_Iraqi}"
SKIP_ASSETS="${SKIP_ASSETS:-0}"

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
DEFAULT_REPO="${IRAQI_VOICE_REPO:-https://github.com/Alqudimi/VoiceCall.git}"
if [[ -f "pyproject.toml" && -d "src/rvc_platform" ]]; then
  PROJECT_ROOT="$(pwd)"
  log "already inside the project: $PROJECT_ROOT"
else
  log "cloning $DEFAULT_REPO"
  rm -rf /content/iraqi-voice-platform
  git clone --depth 1 "$DEFAULT_REPO" /content/iraqi-voice-platform
  cd /content/iraqi-voice-platform
  PROJECT_ROOT=/content/iraqi-voice-platform
fi

# ---------------------------------------------------------------- drive -----
# NOTE: `drive.mount()` requires the NOTEBOOK KERNEL — running it from a bash
# subprocess (which is what `!bash install_colab.sh` is) fails with
# AttributeError: 'NoneType' object has no attribute 'kernel'. Mount Drive in
# the notebook's FIRST Python cell (see notebooks/RVC_Training_Colab.ipynb)
# and this script will detect it. The attempt below is kept only as a
# best-effort fallback for kernel contexts (e.g. %run -i).
if [[ "$MOUNT_DRIVE" == "1" && -d /content/drive/MyDrive ]]; then
  ok "Google Drive already mounted"
elif [[ "$MOUNT_DRIVE" == "1" ]]; then
  log "attempting Google Drive mount from shell (may fail — see note above)"
  python3 -c "from google.colab import drive; drive.mount('/content/drive')" 2>/dev/null || true
  if [[ ! -d /content/drive/MyDrive ]]; then
    log "drive NOT mounted"
    log "mount Drive from a PYTHON cell (not bash), then RE-RUN this install cell:"
    log "    from google.colab import drive"
    log "    drive.mount('/content/drive')"
    log "running without Drive persistence — checkpoints stay in /content (wiped at session end)"
    MOUNT_DRIVE=0
  fi
fi
DRIVE_ROOT=""
if [[ "$MOUNT_DRIVE" == "1" && -d /content/drive/MyDrive ]]; then
  DRIVE_ROOT="/content/drive/MyDrive/$DRIVE_DIR"
  mkdir -p "$DRIVE_ROOT/runs" "$DRIVE_ROOT/models" "$DRIVE_ROOT/datasets"
  ok "Drive workspace: $DRIVE_ROOT"
fi

# ---------------------------------------------------------------- uv + env --
command -v uv >/dev/null 2>&1 || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
log "uv $(uv --version | awk '{print $2}') — syncing environment (CPU wheels default; see below for GPU)"
uv sync --all-groups

# Colab GPU machines: install the matching CUDA variant (torch version kept).
# Variant availability verified against download.pytorch.org for the locked
# torch 2.7.x: ONLY cu118 / cu126 / cu128 exist (no cu121/cu124 wheels).
# Driver CUDA 13.x is fully backward-compatible with cu128 binaries.
GPU_VARIANT=""
if nvidia-smi -L >/dev/null 2>&1; then
  DRIVER_CUDA="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)"
  case "$DRIVER_CUDA" in
    13.*)         GPU_VARIANT=cu128 ;;
    12.8*|12.9*)  GPU_VARIANT=cu128 ;;
    12.[2-7]*)    GPU_VARIANT=cu126 ;;
    12.0*|12.1*)  GPU_VARIANT=cu118 ;;
    11.*)          GPU_VARIANT=cu118 ;;
    "")            GPU_VARIANT=cu128; log "driver CUDA version not readable — defaulting to cu128" ;;
    *)             GPU_VARIANT=cu128; log "unrecognised driver CUDA '$DRIVER_CUDA' — defaulting to cu128" ;;
  esac
  log "driver CUDA ${DRIVER_CUDA:-?} → torch variant $GPU_VARIANT"
fi
GPU_INSTALLED=0
if [[ -n "$GPU_VARIANT" ]]; then
  if uv run rvc-platform env gpu "$GPU_VARIANT"; then
    GPU_INSTALLED=1
    ok "CUDA torch variant installed: $GPU_VARIANT"
    # CRITICAL: plain `uv run` re-syncs the venv from the lockfile and would
    # revert these GPU wheels back to CPU (verified behaviour). Guard every
    # remaining `uv run` in this script AND later notebook cells.
    export UV_NO_SYNC=1
  else
    log "GPU variant install failed — continuing on CPU wheels (training will be VERY slow)"
  fi
fi

# ---------------------------------------------------------------- assets ----
if [[ "$SKIP_ASSETS" == "0" ]]; then
  log "downloading pretrained assets (HuBERT / RMVPE / v2 f0G40k f0D40k)"
  uv run rvc-platform download-models
fi

# ----------------------------------------------------- session env wiring --
# restore models + checkpoints from Drive into the session (resume support)
if [[ -n "$DRIVE_ROOT" ]]; then
  log "restoring runs/ and models/ from Drive (if present)"
  mkdir -p runs models
  rsync -a --exclude='.part' "$DRIVE_ROOT/runs/" runs/ 2>/dev/null || true
  rsync -a "$DRIVE_ROOT/models/" models/ 2>/dev/null || true
fi
# colab_wire_env.py is executed in the notebook kernel (cell 2) so every later
# `!uv run ...` inherits: Drive redirects AND UV_NO_SYNC=1 (a plain `uv run`
# would re-sync the venv from the lockfile and REVERT the GPU torch wheels
# back to CPU — verified behaviour, this is the guard against it).
cat > colab_wire_env.py <<EOF
import os
os.environ.setdefault("RVC_PLATFORM_ROOT", r"$PROJECT_ROOT")
EOF
if [[ -n "$DRIVE_ROOT" ]]; then
  cat >> colab_wire_env.py <<EOF
os.environ["RVC_RUNS_DIR"] = r"$DRIVE_ROOT/runs"       # checkpoints straight to Drive
os.environ["RVC_MODELS_DIR"] = r"$DRIVE_ROOT/models"   # exported models to Drive
os.environ.setdefault("RVC_DATA_DIR", "/content/iraqi_data")
EOF
fi
if [[ $GPU_INSTALLED -eq 1 ]]; then
  cat >> colab_wire_env.py <<'EOF'
os.environ["UV_NO_SYNC"] = "1"  # keep GPU torch — plain `uv run` re-syncs to CPU wheels
EOF
fi
cat > .colab_env <<EOF
# shell-only fallback (bash subshells): source this file before uv commands
export RVC_PLATFORM_ROOT="$PROJECT_ROOT"
export RVC_RUNS_DIR="${DRIVE_ROOT:+$DRIVE_ROOT/runs}"
export RVC_MODELS_DIR="${DRIVE_ROOT:+$DRIVE_ROOT/models}"
export RVC_DATA_DIR="/content/iraqi_data"
EOF
if [[ $GPU_INSTALLED -eq 1 ]]; then
  echo 'export UV_NO_SYNC=1   # keep GPU torch — plain uv run re-syncs to CPU' >> .colab_env
fi
ok "wrote colab_wire_env.py — exec it in the next cell (see notebook cell 2)"

# ---------------------------------------------------------------- verify ----
log "environment verification"
uv run rvc-platform install check
[[ "$SKIP_ASSETS" == "0" ]] && uv run rvc-platform install doctor
ok "Colab setup complete"
echo
echo "  next steps (in new cells):"
echo "    %cd $PROJECT_ROOT"
echo "    exec(open('colab_wire_env.py').read())   # Drive redirects + GPU-wheel guard"
echo "    !uv run rvc-platform prepare-dataset /content/drive/MyDrive/$DRIVE_DIR/datasets/my_voice iraqi_voice"
echo "    !uv run rvc-platform train start --experiment iraqi_voice"
echo "    !uv run rvc-platform web --host 0.0.0.0 --tunnel   # public UI via quick tunnel"
echo
echo "  NOTE: the token for the tunnel-protected API is written to logs/auth.token;"
echo "  paste it into Settings on the public page (all /api calls return 401 without it)."
echo "  training auto-resumes the latest checkpoint on every rerun;"
echo "  checkpoints/models persist to Drive via RVC_RUNS_DIR/RVC_MODELS_DIR."
if [[ $GPU_INSTALLED -eq 1 ]]; then
echo "  GPU torch is ACTIVE in this session; a later explicit 'uv sync' would revert"
echo "  it to CPU wheels — in that case re-run: uv run rvc-platform env gpu $GPU_VARIANT"
fi
echo "  before the session ends run colab snapshot (see notebook last cell)."
