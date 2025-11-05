# How to Run the Spectral Predict GUI

## ✅ Quick Start (Easiest)

### Option 1: Use the Launcher Script (Recommended)

**Unix/Mac/Linux:**
```bash
./run_gui.sh
```

**Windows:**
```bash
run_gui.bat
```

**Benefits:**
- Automatically checks for dependencies
- Uses the correct virtual environment
- Installs missing packages if needed
- Shows helpful error messages

---

### Option 2: Run Directly with Python

If dependencies are already installed:

```bash
python3 spectral_predict_gui.py
```

**Note:** If you get "No module named 'matplotlib'" error, use Option 1 instead.

---

## 🔧 Troubleshooting

### "No module named 'matplotlib'" Error

**Solution 1: Use the launcher script (Recommended)**
```bash
./run_gui.sh  # or run_gui.bat on Windows
```
The launcher automatically installs all required packages.

**Solution 2: Install dependencies manually**
```bash
pip install matplotlib numpy pandas scikit-learn scipy tabulate specdal
```
Note: `specdal` is needed for reading binary ASD files.

**Solution 3: Use virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
python3 spectral_predict_gui.py
```

---

### "Permission denied" when running run_gui.sh

Make the script executable:
```bash
chmod +x run_gui.sh
./run_gui.sh
```

---

### GUI window is too small

The window should now be 850x1050 pixels. If buttons are still not visible:
1. Manually resize the window by dragging the corner
2. Or check your display resolution settings

---

### "Binary ASD file detected" Message

If you see this message, your ASD files are in binary format (not ASCII).

**Solution: The launcher script automatically installs SpecDAL**
```bash
./run_gui.sh  # Installs specdal if needed
```

**Or install manually:**
```bash
pip install specdal
```

Then restart the GUI. Binary ASD files will now be read automatically.

**Alternative:** Export your ASD files to ASCII format using ASD ViewSpec software.

---

## 📝 What the GUI Does

1. **Select Input Data**: Choose ASD files or CSV format
2. **Load Reference Data**: CSV with target variables
3. **Configure Options**: Set CV folds, complexity penalty, etc.
4. **Select Models**: Choose which models to test (PLS, RF, MLP, Neural Boosted)
5. **Run Analysis**: Automatically tests all combinations
6. **View Results**: Results CSV and markdown report in `outputs/` directory

---

## 🎯 Model Selection

You can now **choose which models to test** using checkboxes:

- ✓ **PLS** - Linear, fast, good baseline
- ✓ **Random Forest** - Nonlinear, robust
- ✓ **MLP** - Deep learning, captures complex patterns
- ✓ **Neural Boosted** - Gradient boosting, interpretable

**Tip:** Uncheck Neural Boosted if you want faster analysis (it takes the longest).

---

## 📊 Results Location

After analysis completes:
- **Results CSV**: `outputs/results_<target>_<timestamp>.csv`
- **Report**: `reports/<target>_<timestamp>.md`

Open the CSV in Excel or similar to see:
- Model rankings
- Performance metrics (RMSE, R²)
- Top important wavelengths
- Wavelength subset performance

---

## 💡 Tips

1. **First time running?** Use the launcher script (`./run_gui.sh`)
2. **Quick test?** Uncheck some models to speed up analysis
3. **High-dimensional data?** Check the results for `top250` or `top500` subsets
4. **Need help?** See the full documentation in the project root

---

## ✅ Verification

To verify everything is installed correctly:
```bash
python3 -c "import matplotlib; import numpy; import pandas; import sklearn; print('All dependencies OK!')"
```

If you see "All dependencies OK!", you can run the GUI directly with `python3 spectral_predict_gui.py`.

---

## 🆘 Still Having Issues?

1. Check that you're in the project directory: `ls -la` should show `spectral_predict_gui.py`
2. Check Python version: `python3 --version` (should be 3.10+)
3. Try reinstalling: `pip install --force-reinstall matplotlib`
4. Use the virtual environment: see Option 3 under Troubleshooting above

---

**All dependencies have been installed system-wide, so the GUI should work now!** ✨
