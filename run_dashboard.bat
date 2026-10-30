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

set PYTHON_CMD=python
if exist ".\.venv\Scripts\python.exe" (
    set PYTHON_CMD=.\.venv\Scripts\python.exe
) else (
    where py >nul 2>&1 && set PYTHON_CMD=py -3.11
)

start "" "http://127.0.0.1:5000"
%PYTHON_CMD% dashboard.py %*
pause
