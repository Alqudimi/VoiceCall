# Iraqi Voice Studio — safe in-place updater for Windows.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\windows\update_windows.ps1 [-DryRun] [-SkipDeps]
[CmdletBinding()]
param(
  [switch]$DryRun,
  [switch]$SkipDeps
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "uv غير موجود. هذا مُحدّث وليس مثبّتًا؛ شغّل المثبّت الرسمي أولًا أو أعد تثبيت uv."
}
$Updater = Join-Path $ProjectRoot "scripts\update_project.py"
$UpdaterArgs = @()
if ($DryRun) { $UpdaterArgs += "--dry-run" }
if ($SkipDeps) { $UpdaterArgs += "--skip-deps" }
& uv run --project $ProjectRoot --no-sync python $Updater @UpdaterArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
