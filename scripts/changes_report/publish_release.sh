#!/usr/bin/env bash
# =============================================================================
# publish_release.sh — recreate the rolling GitHub Release with the report
#
# Strategy: ONE rolling release (tag: changes-report) that is deleted and
# recreated at the current commit on every run, so the download URLs stay
# stable (the wiki links to them) while the assets always reflect the
# newest state. A plain tag name (not v*) never collides with the repo's own
# version tags created by auto-release.yml.
#
# Required environment: GITHUB_SHA, GITHUB_REPOSITORY, GH_TOKEN (with
# contents:write). Optional: RELEASE_TAG (default changes-report).
# =============================================================================
set -euo pipefail

DIST_DIR="${DIST_DIR:-dist/changes-report}"
TAG="${RELEASE_TAG:-changes-report}"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
SHA="${GITHUB_SHA:?GITHUB_SHA is required}"

REPO_NAME="$(basename "$REPO")"
NOTES="$DIST_DIR/release_notes.md"

if [ ! -f "$NOTES" ]; then
  echo "::error::missing $NOTES - run the pipeline first"
  exit 1
fi

echo "== publishing rolling release '$TAG' at $SHA =="

# delete the previous rolling release + tag (ignore absence on first run)
gh release delete "$TAG" --repo "$REPO" --cleanup-tag --yes >/dev/null 2>&1 || true

TITLE="تقرير التغييرات الشامل — ${SHA:0:7} ($(date -u +'%Y-%m-%d %H:%M') UTC)"

gh release create "$TAG" \
  "$DIST_DIR/${REPO_NAME}_Commits_Changes_Map.png" \
  "$DIST_DIR/${REPO_NAME}_Commits_Changes_Map.html" \
  "$DIST_DIR/${REPO_NAME}_Commits_Overview_Chart.png" \
  "$DIST_DIR/${REPO_NAME}_Changes_Analytics_Dashboard.png" \
  "$DIST_DIR/CHANGES.md" \
  --repo "$REPO" \
  --target "$SHA" \
  --title "$TITLE" \
  --notes-file "$NOTES" \
  --latest=false

echo "== release published =="
gh release view "$TAG" --repo "$REPO" \
  --json tagName,isLatest,assets,url | head -40
