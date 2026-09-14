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

REM Apply requirements-lock.txt whenever the venv no longer matches it: a pull
REM that changes a pin must reach existing environments without a manual step.
.venv314\Scripts\python.exe scripts\check_env_lock.py --quiet
if errorlevel 1 (
    echo.
    echo Updating .venv314 to match requirements-lock.txt...
    .venv314\Scripts\python.exe -m pip install -q -r requirements-lock.txt
    if errorlevel 1 goto :dependency_install_failed
    .venv314\Scripts\python.exe -m pip install -q -e . --no-deps
    if errorlevel 1 goto :dependency_install_failed
)
:launch

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
REM Offline is the usual cause; an out-of-date venv may still run, so try it.
echo.
echo WARNING: Could not update .venv314 from requirements-lock.txt.
echo Launching anyway. Features whose packages are out of date may fail;
echo reconnect and run install.bat to finish the update.
echo.
goto :launch
