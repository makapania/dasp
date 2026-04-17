@echo off
REM ============================================================================
REM Spectral Predict - First-Time Install (Windows)
REM ============================================================================
REM Creates a Python 3.12 virtual environment in .venv312\ and installs all
REM dependencies. Run this once after cloning the repo. Safe to re-run any
REM time to refresh dependencies after a `git pull`.
REM ============================================================================

setlocal
cd /d "%~dp0"

echo.
echo ================================================================================
echo    SPECTRAL PREDICT - Installation
echo ================================================================================
echo.

REM --- Locate Python 3.12 -------------------------------------------------------
set "PYEXE="

py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set "PYEXE=py -3.12"
    goto :found_python
)

python --version 2>nul | findstr /C:"3.12" >nul
if not errorlevel 1 (
    set "PYEXE=python"
    goto :found_python
)

echo ERROR: Python 3.12 was not found on PATH.
echo.
echo Install Python 3.12 from:
echo    https://www.python.org/downloads/release/python-3120/
echo.
echo During installation, check the box "Add Python to PATH".
echo Then re-run this script.
echo.
pause
exit /b 1

:found_python
echo Found Python 3.12: %PYEXE%
echo.

REM --- Create venv if missing ---------------------------------------------------
if not exist ".venv312\Scripts\python.exe" (
    echo Creating virtual environment in .venv312\ ...
    %PYEXE% -m venv .venv312
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo See the output above for details.
        pause
        exit /b 1
    )
    echo.
)

REM --- Upgrade pip --------------------------------------------------------------
echo Upgrading pip ...
.venv312\Scripts\python.exe -m pip install --upgrade pip
echo.

REM --- Install project + all dependencies ---------------------------------------
echo Installing Spectral Predict and all dependencies.
echo (First install can take 5-10 minutes depending on connection speed.)
echo.
.venv312\Scripts\python.exe -m pip install -e . --upgrade
if errorlevel 1 (
    echo.
    echo ================================================================================
    echo    ERROR: Installation failed.
    echo ================================================================================
    echo.
    echo Review the output above. Common causes:
    echo   - No internet connection, or proxy/firewall blocking pip
    echo   - Missing Visual C++ Build Tools ^(some packages compile from source^)
    echo   - Disk full
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo    Installation complete.
echo ================================================================================
echo.
echo To launch the GUI: double-click RUN_SPECTRAL_PREDICT.bat
echo.
pause
