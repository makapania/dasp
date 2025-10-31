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

REM Launch Python GUI (which will use Julia backend)
python "spectral_predict_gui_optimized.py"

REM If python command doesn't work, try python3 or py
if errorlevel 1 (
    echo.
    echo Python not found in PATH. Trying alternative commands...
    python3 "spectral_predict_gui_optimized.py"
    if errorlevel 1 (
        py "spectral_predict_gui_optimized.py"
        if errorlevel 1 (
            echo.
            echo ERROR: Could not find Python interpreter.
            echo Please install Python 3.8+ from python.org
            pause
            exit /b 1
        )
    )
)

pause
