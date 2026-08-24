@echo off
setlocal
cd /d "%~dp0"

if not exist "runtime\python.exe" (
  echo The portable runtime is missing.
  echo Extract the complete offline ZIP before starting the app.
  pause
  exit /b 1
)

echo Starting CUS AI Reader. Please keep this window open.
echo If the browser does not open, double-click "CUS AI Reader Local Page.url".
echo.
"runtime\python.exe" "portable_launcher.py"
set "CUS_EXIT_CODE=%ERRORLEVEL%"
if not "%CUS_EXIT_CODE%"=="0" (
  echo.
  echo CUS AI Reader stopped with an error. See the messages above.
  pause
)
exit /b %CUS_EXIT_CODE%
