@echo off
REM ============================================================================
REM Spectral Predict - One-Click Launcher
REM ============================================================================
REM This script launches the Spectral Predict GUI with Julia backend
REM No configuration needed - just double-click this file!
REM ============================================================================

echo.
echo ================================================================================
echo    SPECTRAL PREDICT - Automated Spectral Analysis
echo    Julia-Powered High-Performance Edition
echo ================================================================================
echo.
echo Starting GUI...
echo.

REM Set working directory to script location
cd /d "%~dp0"

REM Launch Python GUI with virtual environment Python
.venv\Scripts\python.exe "spectral_predict_gui_optimized.py"

REM Check if execution succeeded
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch GUI
    echo Check that all dependencies are installed
    pause
    exit /b 1
)

pause
