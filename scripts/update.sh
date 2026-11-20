#!/usr/bin/env bash
# Generic safe updater for Linux/macOS/Colab project checkouts.
# Usage: bash scripts/update.sh [--dry-run] [--skip-deps]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
command -v uv >/dev/null 2>&1 || {
  echo "[error] uv غير موجود. هذا مُحدّث وليس مثبّتًا؛ ثبّت المشروع أو uv أولًا." >&2
  exit 2
}
exec uv run --project "$PROJECT_ROOT" --no-sync python "$PROJECT_ROOT/scripts/update_project.py" "$@"
