@echo off
REM ============================================================================
REM Spectral Predict - First-Time Install (Windows)
REM ============================================================================
REM Creates a Python 3.14 virtual environment in .venv314\ and installs all
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

REM --- Locate Python 3.14 -------------------------------------------------------
set "PYEXE="

py -3.14 --version >nul 2>&1
if not errorlevel 1 (
    set "PYEXE=py -3.14"
    goto :found_python
)

python --version 2>nul | findstr /C:"3.14" >nul
if not errorlevel 1 (
    set "PYEXE=python"
    goto :found_python
)

echo ERROR: Python 3.14 was not found on PATH.
echo.
echo Install Python 3.14 from:
echo    https://www.python.org/downloads/  ^(or: winget install Python.Python.3.14^)
echo    Use the ordinary build, NOT the free-threaded ^(t^) variant.
echo.
echo During installation, check the box "Add Python to PATH".
echo Then re-run this script.
echo.
pause
exit /b 1

:found_python
echo Found Python 3.14: %PYEXE%
echo.

REM --- Create venv if missing ---------------------------------------------------
if not exist ".venv314\Scripts\python.exe" (
    echo Creating virtual environment in .venv314\ ...
    %PYEXE% -m venv .venv314
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
.venv314\Scripts\python.exe -m pip install --upgrade pip
echo.

REM --- Install project + all dependencies ---------------------------------------
echo Installing Spectral Predict and all dependencies.
echo (First install can take 5-10 minutes depending on connection speed.)
echo.
.venv314\Scripts\python.exe -m pip install -r requirements-lock.txt
if errorlevel 1 goto :install_failed
.venv314\Scripts\python.exe -m pip install -e . --no-deps
if errorlevel 1 goto :install_failed

echo.
echo ================================================================================
echo    Installation complete.
echo ================================================================================
echo.
echo To launch the GUI: double-click RUN_SPECTRAL_PREDICT.bat
echo.
pause
exit /b 0

:install_failed
echo.
echo ================================================================================
echo    ERROR: Installation failed.
echo ================================================================================
echo.
echo Review the output above. Common causes:
echo   - No internet connection, or proxy/firewall blocking pip
echo   - Missing Visual C++ Build Tools ^(some packages compile from source^)
echo   - Disk full
echo   - Wrong Python: this project requires 3.14 and pip will refuse older
echo.
pause
exit /b 1
