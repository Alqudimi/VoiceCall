@echo off
setlocal
set "SCRIPT=%~dp0update_windows.ps1"
if not exist "%SCRIPT%" (
  echo [error] update_windows.ps1 not found next to this wrapper.
  exit /b 2
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
