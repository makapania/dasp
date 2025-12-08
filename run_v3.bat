@echo off
REM ============================================================================
REM Spectral Predict V3 - One-Click Launcher
REM ============================================================================
REM This script launches Spectral Predict V3 (Dear PyGui version)
REM ============================================================================

echo.
echo ================================================================================
echo    SPECTRAL PREDICT V3 - Dear PyGui Edition
echo ================================================================================
echo.
echo Starting GUI...
echo.

REM Set working directory to script location
cd /d "%~dp0"

REM Launch with system Python (has dearpygui installed)
python -m spectral_predict_v3.main

REM Check if execution succeeded
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch GUI
    echo Check that dearpygui is installed: pip install dearpygui
    pause
    exit /b 1
)
