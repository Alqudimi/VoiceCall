#!/usr/bin/env bash
# Iraqi Voice Studio — safe in-place updater for Ubuntu/Linux.
# Usage: bash scripts/ubuntu/update_ubuntu.sh [--dry-run] [--skip-deps]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
command -v uv >/dev/null 2>&1 || {
  echo "[error] uv غير موجود. هذا مُحدّث وليس مثبّتًا؛ ثبّت المشروع أولًا أو أعد تثبيت uv." >&2
  exit 2
}
exec uv run --project "$PROJECT_ROOT" --no-sync python "$PROJECT_ROOT/scripts/update_project.py" "$@"
