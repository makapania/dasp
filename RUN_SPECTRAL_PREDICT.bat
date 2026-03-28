@echo off
REM ============================================================================
REM Spectral Predict - One-Click Launcher
REM ============================================================================
REM This script launches the Spectral Predict GUI
REM No configuration needed - just double-click this file!
REM ============================================================================

echo.
echo ================================================================================
echo    SPECTRAL PREDICT - Automated Spectral Analysis
echo ================================================================================
echo.
echo Starting GUI...
echo.

REM Set working directory to script location
cd /d "%~dp0"

REM Ensure core package and required Omnic dependencies are installed in the venv
.venv312\Scripts\python.exe -c "import importlib.util, sys; required = ('spectral_predict', 'requests', 'spectrochempy_omnic'); missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit(0 if not missing else 1)"
if errorlevel 1 (
    echo.
    echo Required packages missing from .venv312. Installing project dependencies...
    .venv312\Scripts\python.exe -m pip install -q -e .
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install required dependencies into .venv312
        pause
        exit /b 1
    )
)

REM Launch Python GUI with virtual environment Python
.venv312\Scripts\python.exe "spectral_predict_gui_optimized.py"

REM Check if execution succeeded
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch GUI
    echo Check that all dependencies are installed
    pause
    exit /b 1
)

pause
