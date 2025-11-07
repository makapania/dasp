#!/bin/bash
# Launcher for SpectralPredict GUI with Python 3.11

# Try to find Python 3.11
if command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    PYTHON=/opt/homebrew/bin/python3.11
elif command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
else
    echo "❌ Python 3.11 not found!"
    echo "Please run: ./setup_python_for_gui.sh first"
    exit 1
fi

echo "Launching SpectralPredict GUI with Python 3.11..."
echo "Python: $PYTHON"
$PYTHON --version
echo ""

cd "$(dirname "$0")"
$PYTHON spectral_predict_gui_optimized.py

