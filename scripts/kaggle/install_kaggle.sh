#!/usr/bin/env bash
# Iraqi Voice Studio — Kaggle Notebook installer.
#
# Recommended Kaggle cell (after adding this file to the repository):
#   !curl -fsSL https://raw.githubusercontent.com/Alqudimi/VoiceCall/main/scripts/kaggle/install_kaggle.sh -o /kaggle/temp/install_kaggle.sh && bash /kaggle/temp/install_kaggle.sh
#
# Or, after cloning the repository:
#   !bash scripts/kaggle/install_kaggle.sh
#
# Kaggle requirements:
#   - Notebook Settings -> Internet: ON (required for clone/downloads).
#   - Notebook Settings -> Accelerator -> GPU (recommended for training).
#
# Environment variables:
#   IRAQI_VOICE_REPO       Git URL (default: https://github.com/Alqudimi/VoiceCall.git)
#   KAGGLE_PROJECT_DIR     Project checkout (default: /kaggle/working/iraqi-voice-platform)
#   KAGGLE_OUTPUT_DIR      Persistent notebook-output folder (default: /kaggle/working/iraqi_voice_output)
#   KAGGLE_RUNTIME_DIR     Fast, session-only workspace (default: /kaggle/temp/iraqi_voice_runtime)
#   KAGGLE_RESTORE_DIR     Optional read-only input folder containing runs/, models/, assets/
#   KAGGLE_LOCAL_WORKSPACE Use /kaggle/temp for runs/data/cache (default: 1)
#   SKIP_ASSETS=1          Skip pretrained asset downloads (default: 0)
#   UPDATE_REPO=1          Pull latest code if project already exists (default: 0)
#   INSTALL_CLOUDFLARED=0  Skip cloudflared installation (default: 1)
#   INSTALL_FFMPEG=0       Do not try apt installation if ffmpeg is missing (default: 1)
#   GPU_FAILURE_FATAL=0    Allow CPU fallback if CUDA wheel installation fails (default: 1)
#
# Kaggle persistence model:
#   /kaggle/input  = read-only attached datasets/models/notebook outputs
#   /kaggle/working = saved when a Notebook Version is created (output-size limited)
#   /kaggle/temp   = fast scratch space; removed when the session ends
#
set -Eeuo pipefail

# -------------------------------------------------------------------- output
log()  { printf '\033[1;35m[kaggle]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[kaggle-ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[kaggle-warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[kaggle-error]\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------- settings
REPO_URL="${IRAQI_VOICE_REPO:-https://github.com/Alqudimi/VoiceCall.git}"
PROJECT_DIR="${KAGGLE_PROJECT_DIR:-/kaggle/working/iraqi-voice-platform}"
OUTPUT_DIR="${KAGGLE_OUTPUT_DIR:-/kaggle/working/iraqi_voice_output}"
RUNTIME_DIR="${KAGGLE_RUNTIME_DIR:-/kaggle/temp/iraqi_voice_runtime}"
RESTORE_DIR="${KAGGLE_RESTORE_DIR:-}"
LOCAL_WORKSPACE="${KAGGLE_LOCAL_WORKSPACE:-1}"
SKIP_ASSETS="${SKIP_ASSETS:-0}"
UPDATE_REPO="${UPDATE_REPO:-0}"
INSTALL_CLOUDFLARED="${INSTALL_CLOUDFLARED:-1}"
INSTALL_FFMPEG="${INSTALL_FFMPEG:-1}"
GPU_FAILURE_FATAL="${GPU_FAILURE_FATAL:-1}"

# -------------------------------------------------------------- Kaggle guard
log "checking Kaggle runtime"
[[ -d /kaggle/working ]] || die "This installer must run inside a Kaggle Notebook (/kaggle/working is missing)."
mkdir -p /kaggle/working /kaggle/temp
[[ -d /kaggle/input ]] || warn "/kaggle/input is missing; attached datasets will not be available."

if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
  ok "Kaggle kernel detected: ${KAGGLE_KERNEL_RUN_TYPE}"
else
  ok "Kaggle filesystem detected"
fi

log "Python and system information"
python3 --version || true
uname -m
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  warn "nvidia-smi not found; this session appears to be CPU-only."
fi

# ------------------------------------------------------------------- project
if [[ -f "pyproject.toml" && -d "src/rvc_platform" ]]; then
  PROJECT_DIR="$(pwd -P)"
  ok "already inside the project: $PROJECT_DIR"
else
  if [[ -d "$PROJECT_DIR/.git" && -f "$PROJECT_DIR/pyproject.toml" ]]; then
    log "project already exists: $PROJECT_DIR"
    if [[ "$UPDATE_REPO" == "1" ]]; then
      log "updating existing checkout (fast-forward only)"
      git -C "$PROJECT_DIR" pull --ff-only
    fi
  elif [[ -e "$PROJECT_DIR" ]]; then
    die "$PROJECT_DIR exists but is not a valid VoiceCall checkout. Set KAGGLE_PROJECT_DIR to another path or remove that folder manually."
  else
    log "cloning $REPO_URL into $PROJECT_DIR"
    mkdir -p "$(dirname "$PROJECT_DIR")"
    git clone --depth 1 "$REPO_URL" "$PROJECT_DIR"
  fi
fi

[[ -f "$PROJECT_DIR/pyproject.toml" ]] || die "pyproject.toml not found in $PROJECT_DIR"
[[ -d "$PROJECT_DIR/src/rvc_platform" ]] || die "src/rvc_platform not found in $PROJECT_DIR"
cd "$PROJECT_DIR"

# ------------------------------------------------------------- storage paths
mkdir -p "$OUTPUT_DIR" "$RUNTIME_DIR"

if [[ "$LOCAL_WORKSPACE" == "1" ]]; then
  RUNS_DIR="$RUNTIME_DIR/runs"
  DATA_DIR="$RUNTIME_DIR/data"
  CACHE_DIR="$RUNTIME_DIR/cache"
  # Keep reusable inference models and pretrained assets under /kaggle/working.
  MODELS_DIR="$OUTPUT_DIR/models"
  ASSETS_DIR="$OUTPUT_DIR/assets"
else
  RUNS_DIR="$PROJECT_DIR/runs"
  DATA_DIR="$PROJECT_DIR/data"
  CACHE_DIR="$PROJECT_DIR/.cache"
  MODELS_DIR="$PROJECT_DIR/models"
  ASSETS_DIR="$PROJECT_DIR/assets"
fi

mkdir -p "$RUNS_DIR" "$DATA_DIR" "$CACHE_DIR" "$MODELS_DIR" "$ASSETS_DIR"

# ------------------------------------------------------- restore prior output
copy_tree() {
  local source_dir="$1" destination_dir="$2"
  [[ -d "$source_dir" ]] || return 0
  mkdir -p "$destination_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='*.part' --exclude='*.tmp' "$source_dir/" "$destination_dir/"
  else
    cp -an "$source_dir/." "$destination_dir/"
  fi
}

if [[ -n "$RESTORE_DIR" ]]; then
  if [[ ! -d "$RESTORE_DIR" ]]; then
    warn "KAGGLE_RESTORE_DIR does not exist: $RESTORE_DIR (restore skipped)"
  else
    RESTORE_BASE="$RESTORE_DIR"
    # Supports attaching either the output folder itself or its parent dataset.
    if [[ -d "$RESTORE_BASE/iraqi_voice_output" ]]; then
      RESTORE_BASE="$RESTORE_BASE/iraqi_voice_output"
    fi
    log "restoring prior artifacts from read-only Kaggle input: $RESTORE_BASE"
    copy_tree "$RESTORE_BASE/runs" "$RUNS_DIR"
    copy_tree "$RESTORE_BASE/models" "$MODELS_DIR"
    copy_tree "$RESTORE_BASE/assets" "$ASSETS_DIR"
    ok "restore step completed"
  fi
else
  log "no KAGGLE_RESTORE_DIR supplied; starting without an external checkpoint restore"
fi

# --------------------------------------------------------------- system deps
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [[ "$INSTALL_FFMPEG" == "1" ]] && command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" == "0" ]]; then
    log "ffmpeg is missing; attempting installation from the Kaggle Ubuntu repositories"
    if apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; then
      ok "ffmpeg installed"
    else
      warn "ffmpeg installation failed. WAV/FLAC workflows may still work; compressed audio formats may not."
    fi
  else
    warn "ffmpeg is missing. WAV/FLAC workflows may still work; compressed audio formats may not."
  fi
else
  ok "ffmpeg: $(ffmpeg -version 2>/dev/null | head -n 1)"
fi

# ------------------------------------------------------------ install cloudflared
TOOLS_DIR="$PROJECT_DIR/.tools"
mkdir -p "$TOOLS_DIR"
CLOUDFLARED_BIN=""
if command -v cloudflared >/dev/null 2>&1; then
  CLOUDFLARED_BIN="$(command -v cloudflared)"
elif [[ -x "$TOOLS_DIR/cloudflared" ]]; then
  CLOUDFLARED_BIN="$TOOLS_DIR/cloudflared"
elif [[ "$INSTALL_CLOUDFLARED" == "1" ]]; then
  case "$(uname -m)" in
    x86_64|amd64) CF_ASSET="cloudflared-linux-amd64" ;;
    aarch64|arm64) CF_ASSET="cloudflared-linux-arm64" ;;
    *) warn "Unsupported architecture for automatic cloudflared installation: $(uname -m)"; CF_ASSET="" ;;
  esac
  if [[ -n "$CF_ASSET" ]]; then
    log "installing cloudflared from the official Cloudflare GitHub release"
    if curl -fL --retry 3 --retry-delay 2 \
      "https://github.com/cloudflare/cloudflared/releases/latest/download/${CF_ASSET}" \
      -o "$TOOLS_DIR/cloudflared"; then
      chmod 0755 "$TOOLS_DIR/cloudflared"
      if "$TOOLS_DIR/cloudflared" --version >/dev/null 2>&1; then
        CLOUDFLARED_BIN="$TOOLS_DIR/cloudflared"
        ok "cloudflared installed: $($CLOUDFLARED_BIN --version 2>/dev/null | head -n 1)"
      else
        rm -f "$TOOLS_DIR/cloudflared"
        warn "cloudflared download was not executable; tunnel support will be unavailable."
      fi
    else
      warn "cloudflared download failed; tunnel support will be unavailable."
    fi
  fi
else
  log "cloudflared installation disabled"
fi

# -------------------------------------------------------------------- uv
export UV_CACHE_DIR="$RUNTIME_DIR/uv-cache"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_DIR/venv"
mkdir -p "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed."
ok "uv $(uv --version | awk '{print $2}')"

# The project requires Python >=3.12,<3.13. Do not use Kaggle's interpreter
# blindly: install/select a compatible managed 3.12 runtime through uv.
log "ensuring Python 3.12 (required by pyproject.toml)"
uv python install 3.12

# ------------------------------------------------------------- GPU selection
GPU_VARIANT=""
GPU_PRESENT=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_PRESENT=1
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version: ([0-9.]+).*/\1/p' | head -n 1)"
  if [[ -z "$DRIVER_CUDA" ]]; then
    GPU_VARIANT="cu128"
    warn "Could not parse the driver's CUDA compatibility version; selecting cu128."
  else
    # This project pins torch 2.7.x. Its supported CUDA wheel variants are
    # cu118, cu126 and cu128; the installed NVIDIA driver is backward-compatible.
    if [[ "$(printf '%s\n' "$DRIVER_CUDA" "12.8" | sort -V | tail -n 1)" == "$DRIVER_CUDA" ]]; then
      GPU_VARIANT="cu128"
    elif [[ "$(printf '%s\n' "$DRIVER_CUDA" "12.6" | sort -V | tail -n 1)" == "$DRIVER_CUDA" ]]; then
      GPU_VARIANT="cu126"
    else
      GPU_VARIANT="cu118"
    fi
  fi
  log "NVIDIA GPU detected; driver CUDA ${DRIVER_CUDA:-unknown} -> ${GPU_VARIANT}"
  # The project currently implements single-GPU training, even on Kaggle T4x2.
  export CUDA_VISIBLE_DEVICES=0
else
  warn "No NVIDIA GPU detected. Installation will continue with the CPU environment. Training on CPU is not recommended."
fi

# ---------------------------------------------------------- sync dependencies
log "syncing project dependencies (Python 3.12; dev + training + quality extras)"
uv sync --project "$PROJECT_DIR" --python 3.12 --all-groups --extra quality --extra training

GPU_INSTALLED=0
if [[ -n "$GPU_VARIANT" ]]; then
  log "installing the matching PyTorch CUDA wheels through the project CLI"
  if uv run --project "$PROJECT_DIR" rvc-platform env gpu "$GPU_VARIANT"; then
    GPU_INSTALLED=1
    export UV_NO_SYNC=1
    ok "CUDA PyTorch wheels installed: $GPU_VARIANT"
  else
    if [[ "$GPU_FAILURE_FATAL" == "1" ]]; then
      die "CUDA wheel installation failed on a GPU runtime. To allow CPU fallback, rerun with GPU_FAILURE_FATAL=0."
    fi
    warn "CUDA wheel installation failed; continuing with the environment as installed."
  fi
fi

# ----------------------------------------------------------- app environment
export RVC_PLATFORM_ROOT="$PROJECT_DIR"
export RVC_RUNS_DIR="$RUNS_DIR"
export RVC_MODELS_DIR="$MODELS_DIR"
export RVC_ASSETS_DIR="$ASSETS_DIR"
export RVC_DATA_DIR="$DATA_DIR"
export RVC_CACHE_DIR="$CACHE_DIR"
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$CACHE_DIR/torch"
export PATH="$TOOLS_DIR:$PATH"

# -------------------------------------------------------------- assets
if [[ "$SKIP_ASSETS" == "0" ]]; then
  log "downloading pretrained assets (HuBERT / RMVPE / v2 pretrained weights)"
  uv run --project "$PROJECT_DIR" rvc-platform download-models
else
  log "SKIP_ASSETS=1: pretrained asset download skipped"
fi

# ----------------------------------------------------------- Kaggle bridges
# A subprocess cannot change the notebook kernel's environment. Generate a
# Python bridge and a shell bridge so later notebook cells use the same paths
# and, importantly, UV_NO_SYNC=1 after CUDA wheels have been installed.
python3 - "$PROJECT_DIR" "$RUNS_DIR" "$MODELS_DIR" "$ASSETS_DIR" "$DATA_DIR" "$CACHE_DIR" "$OUTPUT_DIR" "$RUNTIME_DIR" "$TOOLS_DIR" "$GPU_INSTALLED" "$CLOUDFLARED_BIN" <<'PY'
import json
import os
import sys
from pathlib import Path

(project, runs, models, assets, data, cache, output, runtime, tools,
 gpu_installed, cloudflared) = sys.argv[1:]
values = {
    "RVC_PLATFORM_ROOT": project,
    "RVC_RUNS_DIR": runs,
    "RVC_MODELS_DIR": models,
    "RVC_ASSETS_DIR": assets,
    "RVC_DATA_DIR": data,
    "RVC_CACHE_DIR": cache,
    "HF_HOME": str(Path(cache) / "huggingface"),
    "HUGGINGFACE_HUB_CACHE": str(Path(cache) / "huggingface" / "hub"),
    "TORCH_HOME": str(Path(cache) / "torch"),
    "UV_CACHE_DIR": str(Path(runtime) / "uv-cache"),
    "UV_PROJECT_ENVIRONMENT": str(Path(runtime) / "venv"),
    "KAGGLE_PROJECT_DIR": project,
    "KAGGLE_OUTPUT_DIR": output,
    "KAGGLE_RUNTIME_DIR": runtime,
}
if gpu_installed == "1":
    values["UV_NO_SYNC"] = "1"
    values["CUDA_VISIBLE_DEVICES"] = "0"
local_bin = str(Path.home() / ".local" / "bin")
path_entries = [str(Path(tools)), local_bin, os.environ.get("PATH", "")]
values["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)

py_file = Path(project) / "kaggle_wire_env.py"
lines = ["# Generated by scripts/kaggle/install_kaggle.sh; execute in a Kaggle Python cell.", "import os"]
for key, value in values.items():
    lines.append(f"os.environ[{key!r}] = {value!r}")
lines.append(f"os.chdir({project!r})")
py_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

sh_file = Path(project) / ".kaggle_env"
with sh_file.open("w", encoding="utf-8") as handle:
    handle.write("# Generated by scripts/kaggle/install_kaggle.sh\n")
    for key, value in values.items():
        import shlex
        handle.write(f"export {key}={shlex.quote(value)}\n")
PY

# ---------------------------------------------------------- snapshot helper
cat > "$PROJECT_DIR/kaggle_snapshot.sh" <<EOF
#!/usr/bin/env bash
# Copy runtime checkpoints into /kaggle/working so they are included in a
# Kaggle Notebook Version. Run after training has reached a safe checkpoint.
set -Eeuo pipefail
OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR")
RUNS_DIR=$(printf '%q' "$RUNS_DIR")
MODELS_DIR=$(printf '%q' "$MODELS_DIR")
ASSETS_DIR=$(printf '%q' "$ASSETS_DIR")

copy_tree() {
  local source_dir="\$1" destination_dir="\$2"
  [[ -d "\$source_dir" ]] || { echo "[snapshot] skip (missing): \$source_dir"; return 0; }
  mkdir -p "\$destination_dir"
  if [[ "\$source_dir" == "\$destination_dir" ]]; then
    echo "[snapshot] already persistent: \$source_dir"
  elif command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='*.part' --exclude='*.tmp' "\$source_dir/" "\$destination_dir/"
  else
    cp -a "\$source_dir/." "\$destination_dir/"
  fi
  echo "[snapshot] saved: \$source_dir -> \$destination_dir"
}

mkdir -p "\$OUTPUT_DIR"
copy_tree "\$RUNS_DIR" "\$OUTPUT_DIR/runs"
copy_tree "\$MODELS_DIR" "\$OUTPUT_DIR/models"
copy_tree "\$ASSETS_DIR" "\$OUTPUT_DIR/assets"

python3 - "\$OUTPUT_DIR" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "contents": {
        name: sum(p.is_file() for p in (root / name).rglob("*")) if (root / name).is_dir() else 0
        for name in ("runs", "models", "assets")
    },
    "note": "Attach this saved notebook output as a Kaggle input in a later session and set KAGGLE_RESTORE_DIR to its folder."
}
(root / "kaggle_snapshot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo
echo "[snapshot] Finished. Create a Kaggle Notebook Version (Save Version) to persist /kaggle/working output."
EOF
chmod 0755 "$PROJECT_DIR/kaggle_snapshot.sh"

# ------------------------------------------------------------- verification
log "verifying the installed runtime"
uv run --project "$PROJECT_DIR" rvc-platform install check
if [[ "$SKIP_ASSETS" == "0" ]]; then
  uv run --project "$PROJECT_DIR" rvc-platform install doctor
fi

TORCH_REPORT="$("$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import json
try:
    import torch
    print(json.dumps({
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }))
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
PY
)"
printf '%s\n' "$TORCH_REPORT"

if [[ "$GPU_PRESENT" == "1" ]]; then
  CUDA_OK="$(printf '%s' "$TORCH_REPORT" | "$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import json,sys; print(str(json.load(sys.stdin).get("cuda_available", False)).lower())')"
  if [[ "$CUDA_OK" != "true" ]]; then
    if [[ "$GPU_FAILURE_FATAL" == "1" ]]; then
      die "An NVIDIA GPU is present but PyTorch cannot access CUDA. Check the CUDA wheel installation before training."
    fi
    warn "GPU runtime is present but PyTorch CUDA is unavailable; training will be CPU-only."
  else
    ok "CUDA is active; this project will use GPU 0 (single-GPU training)."
  fi
fi

ok "Kaggle setup complete"
echo
echo "  Next notebook cell (Python):"
echo "    exec(open('$PROJECT_DIR/kaggle_wire_env.py').read())"
echo
echo "  Prepare dataset (replace the input path with your attached Kaggle dataset):"
echo "    !uv run rvc-platform prepare-dataset /kaggle/input/YOUR-DATASET/ iraqi_voice"
echo
echo "  Start or resume training:"
echo "    !uv run rvc-platform train start --experiment iraqi_voice"
echo
echo "  Start the web UI with a Cloudflare Quick Tunnel:"
echo "    !uv run rvc-platform web --host 0.0.0.0 --port 8320 --tunnel"
echo
echo "  Before the Kaggle session ends, snapshot checkpoints:"
echo "    !bash $PROJECT_DIR/kaggle_snapshot.sh"
echo
echo "  Persistent output: $OUTPUT_DIR"
echo "  Runtime workspace: $RUNTIME_DIR"
echo "  Checkpoints are in the temporary runtime until the snapshot script copies them."
echo "  After snapshotting, use Save Version to persist notebook output."
echo
if [[ "$GPU_INSTALLED" == "1" ]]; then
  echo "  CUDA wheels are active. Do not run an unguarded 'uv sync' afterward;"
  echo "  UV_NO_SYNC=1 is set by kaggle_wire_env.py for later notebook cells."
fi
