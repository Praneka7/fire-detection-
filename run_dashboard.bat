@echo off
cd /d "%~dp0"
echo ===================================================
echo     PyroGuard - Fire Camera Detector Dashboard
echo ===================================================
echo Starting Web Command Center at http://127.0.0.1:5000 ...
echo Opening your default browser...
echo.
echo Press Ctrl+C in this window to stop the server.
echo ===================================================

start "" "http://127.0.0.1:5000"
.\.venv\Scripts\python.exe dashboard.py %*
pause
