# Spectral Predict — Install Guide

This guide is for installing the **GUI version of Spectral Predict** from the source repository. If you received a bundled `.exe` or `.app`, you don't need this — just run the bundled file.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.12** | Download from [python.org](https://www.python.org/downloads/release/python-3120/). On Windows, **check "Add Python to PATH"** during install. |
| **Git** | Needed to clone the repo. [Download here](https://git-scm.com/downloads). |
| **~3 GB free disk** | For the venv + scientific Python stack (numpy, scikit-learn, xgboost, lightgbm, catboost, etc.). |
| **Internet** | Required during install to download packages from PyPI. |

> **Why Python 3.12?** Earlier versions (3.10, 3.11) will mostly work, but the maintained venv pins are tested on 3.12 and several scientific packages have better wheels there. The previously-bundled `.exe` used 3.11 because of PyInstaller compatibility — that constraint does not apply to the source install.

---

## Install (Windows)

1. **Clone the repo** (or download a zip):
   ```cmd
   git clone https://github.com/makapania/dasp.git
   cd dasp
   ```

2. **Run the installer**:
   ```cmd
   install.bat
   ```
   Or just **double-click `install.bat`** in Explorer.

   This will:
   - Find your Python 3.12 install
   - Create a virtual environment in `.venv312\`
   - Install all dependencies (5–10 minutes on first run)

3. **Launch the GUI** by double-clicking `RUN_SPECTRAL_PREDICT.bat`.

---

## Install (macOS / Linux)

1. **Clone the repo**:
   ```bash
   git clone https://github.com/makapania/dasp.git
   cd dasp
   ```

2. **Run the installer**:
   ```bash
   bash install.sh
   ```

3. **Launch the GUI**:
   ```bash
   ./run_gui.sh
   ```

---

## Updating

When you `git pull` new changes, re-run the install script. It's idempotent — it reuses the existing venv and only installs what's changed:

```cmd
git pull
install.bat        REM Windows
bash install.sh    # macOS / Linux
```

---

## Troubleshooting

### "Python 3.12 was not found on PATH"
You either don't have Python 3.12 installed, or it wasn't added to PATH.

- **Windows:** Reinstall from [python.org](https://www.python.org/downloads/release/python-3120/) and check "Add Python to PATH" on the first installer screen. You can verify with `py -3.12 --version` in a new terminal.
- **macOS:** `brew install python@3.12` then re-run `install.sh`.
- **Linux:** `sudo apt install python3.12 python3.12-venv` (Debian/Ubuntu) or your distro's equivalent.

### "Dependency installation failed" / pip errors
Re-run `install.bat` (or `install.sh`) and read the actual pip error in the output. The most common causes:

- **Corporate proxy/firewall blocking PyPI.** Configure pip to use your proxy: see [pip user guide](https://pip.pypa.io/en/stable/user_guide/#using-a-proxy-server).
- **Missing C++ Build Tools (Windows).** Some scientific packages need to compile native code if a wheel isn't available. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop development with C++".
- **Disk full.** The full install is ~3 GB; check free space.

### "ImportError" or "ModuleNotFoundError" when launching
Your venv is out of date with the current code (e.g. after `git pull`). Re-run `install.bat` / `install.sh`.

### GUI launches but a feature crashes
File an issue with:
- The full traceback from the terminal window
- Output of `.venv312\Scripts\python.exe -m pip list` (Windows) or `.venv312/bin/python -m pip list` (mac/Linux)
- Your OS and Python version

---

## What gets installed

The full dependency list is in `pyproject.toml`. The major packages:

| Package | Used for |
|---------|----------|
| numpy, pandas, scipy | Numerical core |
| scikit-learn | Models, CV, preprocessing, **one-class detection** |
| xgboost, lightgbm, catboost | Gradient-boosted regression / classification |
| matplotlib, tksheet | GUI plots and tables |
| optuna, pymoo | Bayesian / multi-objective optimization |
| imbalanced-learn | Class-imbalance handling |
| pybaselines | Baseline correction (ALS, airPLS, polynomial) |
| openpyxl, xlsxwriter, xlrd, spc-io | File-format readers |
| specdal, brukeropus, specio-py310, spectrochempy-omnic | Vendor format readers (ASD, OPUS, PerkinElmer, Omnic) |

All are pulled in automatically by `pip install -e .`.
