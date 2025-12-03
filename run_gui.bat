@echo off
REM Launcher for Spectral Predict GUI with virtual environment
cd /d "%~dp0"
.venv312\Scripts\python.exe spectral_predict_gui_optimized.py
pause
