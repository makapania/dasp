@echo off
REM Launcher for Spectral Predict GUI with virtual environment
cd /d "%~dp0"

REM Ensure core package and required Omnic dependencies are installed in the venv
.venv311\Scripts\python.exe -c "import importlib.util, sys; required = ('spectral_predict', 'requests', 'spectrochempy_omnic'); missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit(0 if not missing else 1)"
if errorlevel 1 (
    echo Required packages missing from .venv311. Installing project dependencies...
    .venv311\Scripts\python.exe -m pip install -q -e .
    if errorlevel 1 (
        echo ERROR: Failed to install required dependencies into .venv311
        pause
        exit /b 1
    )
)

.venv311\Scripts\python.exe spectral_predict_gui_optimized.py
pause
