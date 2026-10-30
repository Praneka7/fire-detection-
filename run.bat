@echo off
cd /d "%~dp0"
echo ===================================================
echo             Fire Camera Detector
echo ===================================================
echo Starting live detector with default camera (source 0)...
echo Press 'q' or 'Esc' in the camera window to exit.
echo.

set PYTHON_CMD=python
if exist ".\.venv\Scripts\python.exe" (
    set PYTHON_CMD=.\.venv\Scripts\python.exe
) else (
    where py >nul 2>&1 && set PYTHON_CMD=py -3.11
)

%PYTHON_CMD% fire_detector.py %*
pause
