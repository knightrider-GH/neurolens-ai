@echo off
title NeuroLens AI — Running
echo ============================================================
echo   NeuroLens AI v1.0.0
echo   On-Device Visual Intelligence for Snapdragon
echo ============================================================
echo.

:: Activate venv if it exists
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)

echo Starting server at http://127.0.0.1:8765 ...
echo Press Ctrl+C to stop.
echo.

python app.py
