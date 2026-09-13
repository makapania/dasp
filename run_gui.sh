#!/bin/bash
# Spectral Predict GUI Launcher (Unix/Mac/Linux)
#
# This script launches the Spectral Predict GUI using the virtual environment.
# It checks for dependencies and provides helpful error messages.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Spectral Predict GUI Launcher"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv314" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo ""
    echo "Please run ./install.sh first. It creates .venv314 with Python 3.14"
    echo "and installs the verified dependency set from requirements-lock.txt."
    echo ""
    exit 1
fi

# Check if Python exists in venv
if [ ! -f ".venv314/bin/python" ] && [ ! -f ".venv314/bin/python3" ]; then
    echo -e "${RED}Error: Python not found in virtual environment!${NC}"
    echo ""
    echo "Please reinstall the virtual environment."
    exit 1
fi

# Determine which Python to use
if [ -f ".venv314/bin/python3" ]; then
    PYTHON=".venv314/bin/python3"
else
    PYTHON=".venv314/bin/python"
fi

# Repairs install the verified lockfile, never unpinned floor resolution
install_locked() {
    $PYTHON -m pip install -q -r requirements-lock.txt || {
        echo -e "${RED}Error: Failed to install requirements-lock.txt${NC}"
        return 1
    }
    $PYTHON -m pip install -q -e . --no-deps || {
        echo -e "${RED}Error: Failed to install spectral_predict${NC}"
        return 1
    }
}

# Check if the package is installed
if ! $PYTHON -c "import spectral_predict" 2>/dev/null; then
    echo -e "${YELLOW}Warning: spectral_predict package not installed in venv${NC}"
    echo ""
    echo "Installing package..."
    install_locked || exit 1
    echo -e "${GREEN}Package installed successfully!${NC}"
    echo ""
fi

# Check for matplotlib
if ! $PYTHON -c "import matplotlib" 2>/dev/null; then
    echo -e "${YELLOW}Warning: matplotlib not installed${NC}"
    echo "Installing matplotlib..."
    install_locked || exit 1
    echo -e "${GREEN}matplotlib installed successfully!${NC}"
    echo ""
fi

# Check for specdal (for binary ASD files)
if ! $PYTHON -c "import specdal" 2>/dev/null; then
    echo -e "${YELLOW}Note: specdal not installed (needed for binary ASD files)${NC}"
    echo "Installing specdal..."
    install_locked || {
        echo -e "${YELLOW}Warning: Failed to install specdal (you can still use ASCII ASD files)${NC}"
    }
    if $PYTHON -c "import specdal" 2>/dev/null; then
        echo -e "${GREEN}specdal installed successfully!${NC}"
        echo ""
    fi
fi

# Launch the GUI
echo -e "${GREEN}Launching Spectral Predict GUI...${NC}"
echo ""

$PYTHON spectral_predict_gui_optimized.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}Error: GUI exited with an error${NC}"
    echo "Check the error messages above for details."
    exit 1
fi
