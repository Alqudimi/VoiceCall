#!/usr/bin/env bash
# Iraqi Voice Studio — safe updater for an existing Colab checkout.
# Usage in a cell: !bash scripts/colab/update_colab.sh [--dry-run]
# This script never clones, deletes, mounts Drive, or touches runtime datasets.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
command -v uv >/dev/null 2>&1 || {
  echo "[error] uv غير موجود في نسخة المشروع الحالية. شغّل المثبّت الرسمي مرة واحدة أولًا." >&2
  exit 2
}
exec uv run --project "$PROJECT_ROOT" --no-sync python "$PROJECT_ROOT/scripts/update_project.py" "$@"
