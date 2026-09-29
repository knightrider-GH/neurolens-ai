@echo off
title NeuroLens AI — Setup
echo ============================================================
echo   NeuroLens AI — Automated Setup Script
echo   On-Device Visual Intelligence for Snapdragon
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/3] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [3/3] Setup complete!
echo.
echo ============================================================
echo   To start NeuroLens AI:
echo     1. Activate venv: venv\Scripts\activate
echo     2. Run: python app.py
echo     3. Open: http://127.0.0.1:8765
echo ============================================================
echo.
echo   For Snapdragon NPU support, also install:
echo     pip install onnxruntime-qnn
echo ============================================================
pause
