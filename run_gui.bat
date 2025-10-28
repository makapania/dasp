@echo off
REM Spectral Predict GUI Launcher (Windows)
REM
REM This script launches the Spectral Predict GUI using the virtual environment.
REM It checks for dependencies and provides helpful error messages.

echo ========================================
echo Spectral Predict GUI Launcher
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please create the virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -e .
    echo.
    pause
    exit /b 1
)

REM Check if Python exists in venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python not found in virtual environment!
    echo.
    echo Please reinstall the virtual environment.
    pause
    exit /b 1
)

set PYTHON=.venv\Scripts\python.exe

REM Check if the package is installed
%PYTHON% -c "import spectral_predict" 2>nul
if errorlevel 1 (
    echo [WARNING] spectral_predict package not installed in venv
    echo.
    echo Installing package...
    %PYTHON% -m pip install -q -e .
    if errorlevel 1 (
        echo [ERROR] Failed to install package
        pause
        exit /b 1
    )
    echo [SUCCESS] Package installed successfully!
    echo.
)

REM Check for matplotlib
%PYTHON% -c "import matplotlib" 2>nul
if errorlevel 1 (
    echo [WARNING] matplotlib not installed
    echo Installing matplotlib...
    %PYTHON% -m pip install -q matplotlib
    if errorlevel 1 (
        echo [ERROR] Failed to install matplotlib
        pause
        exit /b 1
    )
    echo [SUCCESS] matplotlib installed successfully!
    echo.
)

REM Check for specdal (for binary ASD files)
%PYTHON% -c "import specdal" 2>nul
if errorlevel 1 (
    echo [NOTE] specdal not installed (needed for binary ASD files)
    echo Installing specdal...
    %PYTHON% -m pip install -q specdal
    if errorlevel 1 (
        echo [WARNING] Failed to install specdal (you can still use ASCII ASD files)
    ) else (
        echo [SUCCESS] specdal installed successfully!
        echo.
    )
)

REM Launch the GUI
echo [SUCCESS] Launching Spectral Predict GUI...
echo.

%PYTHON% spectral_predict_gui.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with an error
    echo Check the error messages above for details.
    pause
    exit /b 1
)
