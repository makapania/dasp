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

REM The launcher only repairs packages; creating the environment is install.bat's job
if not exist ".venv314\Scripts\python.exe" goto :venv_missing

REM Ensure core package and required Omnic dependencies are installed in the venv
.venv314\Scripts\python.exe -c "import importlib.util, sys; required = ('spectral_predict', 'requests', 'spectrochempy_omnic'); missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit(0 if not missing else 1)"
if errorlevel 1 (
    echo.
    echo Required packages missing from .venv314. Installing project dependencies...
    .venv314\Scripts\python.exe -m pip install -q -r requirements-lock.txt
    if errorlevel 1 goto :dependency_install_failed
    .venv314\Scripts\python.exe -m pip install -q -e . --no-deps
    if errorlevel 1 goto :dependency_install_failed
)

REM Launch Python GUI with virtual environment Python
.venv314\Scripts\python.exe "spectral_predict_gui_optimized.py"

REM Check if execution succeeded
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch GUI
    echo Check that all dependencies are installed
    pause
    exit /b 1
)

pause
exit /b 0

:venv_missing
echo.
echo ERROR: Python 3.14 environment .venv314 not found.
echo Run install.bat first to create it, then launch again.
pause
exit /b 1

:dependency_install_failed
echo.
echo ERROR: Failed to install required dependencies into .venv314
pause
exit /b 1
