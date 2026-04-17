#!/usr/bin/env bash
# ============================================================================
# Spectral Predict - First-Time Install (macOS / Linux)
# ============================================================================
# Creates a Python 3.12 virtual environment in .venv312/ and installs all
# dependencies. Run this once after cloning the repo. Safe to re-run any
# time to refresh dependencies after a `git pull`.
# ============================================================================

set -e

cd "$(dirname "$0")"

echo
echo "================================================================================"
echo "   SPECTRAL PREDICT - Installation"
echo "================================================================================"
echo

# --- Locate Python 3.12 -------------------------------------------------------
PYEXE=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || echo "")
        if [ "$ver" = "3.12" ]; then
            PYEXE="$candidate"
            break
        fi
    fi
done

if [ -z "$PYEXE" ]; then
    echo "ERROR: Python 3.12 was not found on PATH."
    echo
    echo "Install Python 3.12 from https://www.python.org/downloads/"
    echo "or via your package manager:"
    echo "  macOS:  brew install python@3.12"
    echo "  Ubuntu: sudo apt install python3.12 python3.12-venv"
    echo
    exit 1
fi

echo "Found Python 3.12: $PYEXE"
echo

# --- Create venv if missing ---------------------------------------------------
if [ ! -x ".venv312/bin/python" ] && [ ! -x ".venv312/bin/python3" ]; then
    echo "Creating virtual environment in .venv312/ ..."
    "$PYEXE" -m venv .venv312
    echo
fi

VENV_PY=".venv312/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=".venv312/bin/python3"

# --- Upgrade pip --------------------------------------------------------------
echo "Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip
echo

# --- Install project + all dependencies ---------------------------------------
echo "Installing Spectral Predict and all dependencies."
echo "(First install can take 5-10 minutes depending on connection speed.)"
echo
"$VENV_PY" -m pip install -e . --upgrade

echo
echo "================================================================================"
echo "   Installation complete."
echo "================================================================================"
echo
echo "To launch the GUI: ./run_gui.sh"
echo
