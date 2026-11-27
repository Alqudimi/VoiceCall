#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — the full changes-report pipeline (as used in CI and locally)
#
#   1. parse_commits.py   : git log --numstat -> commits.json (100% coverage)
#   2. make_charts.py     : 2 statistical charts (PNG)
#   3. build_map.py       : RTL-Arabic HTML map (all commits + all files)
#   4. shoot_map.py       : full-page screenshot of the map (PNG)
#   5. build_changes_md.py: CHANGES.md + summary.json + release_notes.md
#
# Environment overrides:
#   FONTS_DIR (default .cache/changes-report-fonts)
#   OUT_DIR   (default dist/changes-report)
#   REPO_NAME (default basename of $GITHUB_REPOSITORY, else 'VoiceCall')
# =============================================================================
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/dist/changes-report}"
FONTS_DIR="${FONTS_DIR:-$REPO_ROOT/.cache/changes-report-fonts}"
RELEASE_TAG="${RELEASE_TAG:-changes-report}"

if [ -n "${GITHUB_REPOSITORY:-}" ]; then
  REPO_NAME="${REPO_NAME:-$(basename "$GITHUB_REPOSITORY")}"
else
  REPO_NAME="${REPO_NAME:-VoiceCall}"
fi

echo "== changes-report pipeline =="
echo "repo root : $REPO_ROOT"
echo "repo name: $REPO_NAME"
echo "out dir  : $OUT_DIR"
echo "fonts dir: $FONTS_DIR"
mkdir -p "$OUT_DIR"

python "$SCRIPTS_DIR/parse_commits.py" \
  --repo "$REPO_ROOT" \
  --out "$OUT_DIR/commits.json"

python "$SCRIPTS_DIR/make_charts.py" \
  --data "$OUT_DIR/commits.json" \
  --out-dir "$OUT_DIR" \
  --fonts "$FONTS_DIR" \
  --repo "$REPO_NAME"

python "$SCRIPTS_DIR/build_map.py" \
  --data "$OUT_DIR/commits.json" \
  --out "$OUT_DIR/map.html" \
  --fonts "$FONTS_DIR" \
  --repo "$REPO_NAME"

python "$SCRIPTS_DIR/shoot_map.py" \
  --html "$OUT_DIR/map.html" \
  --out "$OUT_DIR/${REPO_NAME}_Commits_Changes_Map.png" \
  --data "$OUT_DIR/commits.json"

python "$SCRIPTS_DIR/build_changes_md.py" \
  --data "$OUT_DIR/commits.json" \
  --repo-root "$REPO_ROOT" \
  --out "$REPO_ROOT/CHANGES.md" \
  --dist "$OUT_DIR" \
  --repo "$REPO_NAME" \
  --release-tag "$RELEASE_TAG"

# keep a copy of CHANGES.md with the release artifacts
cp "$REPO_ROOT/CHANGES.md" "$OUT_DIR/CHANGES.md"

echo "== pipeline done =="
ls -lh "$OUT_DIR"
