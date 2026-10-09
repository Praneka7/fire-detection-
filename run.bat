@echo off
cd /d "%~dp0"
echo ===================================================
echo             Fire Camera Detector
echo ===================================================
echo Starting live detector with default camera (source 0)...
echo Press 'q' or 'Esc' in the camera window to exit.
echo.
.\.venv\Scripts\python.exe fire_detector.py %*
pause
