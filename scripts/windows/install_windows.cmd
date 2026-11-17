@echo off
REM Iraqi Voice Studio — Windows CMD wrapper for the PowerShell installer.
REM Usage: scripts\windows\install_windows.cmd [additional flags pass through]
setlocal
set "SCRIPT=%~dp0install_windows.ps1"
if not exist "%SCRIPT%" (
  echo [error] install_windows.ps1 not found next to this wrapper.
  exit /b 1
)
where powershell >nul 2>nul || (echo [error] PowerShell is required & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
endlocal
exit /b %ERRORLEVEL%
