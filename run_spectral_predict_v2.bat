@echo off
REM Spectral Predict v2 Launcher
REM ============================

cd /d "%~dp0"

echo.
echo ================================================================================
echo    SPECTRAL PREDICT v2 - Modern Automated Spectral Analysis
echo ================================================================================
echo.
echo Starting GUI...
echo.

REM Run from the spectral_predict_v2 directory so relative imports work
cd spectral_predict_v2

REM Use the same Python 3.12 venv as v1 (has all dependencies including catboost)
..\\.venv312\\Scripts\\python.exe main.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
