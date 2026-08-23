@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Collect-HistoricalBursaNews.ps1" %*
exit /b %errorlevel%
