# SpectralPredict.jl - Quick Start Guide

## ✨ You Have a Web-Based GUI!

The easiest way to use SpectralPredict.jl is through the web GUI.

## 🚀 Launch the GUI (3 Steps)

### Step 1: Open Terminal

Open Terminal.app on your Mac

### Step 2: Navigate to Directory

```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
```

### Step 3: Start the GUI

```bash
./start_gui.sh
```

**OR:**

```bash
julia --project=. gui.jl
```

## 🌐 Access the GUI

After starting, you'll see:

```
================================================================================
Starting SpectralPredict GUI Server...
================================================================================

Server running at: http://localhost:8080

Open your web browser and navigate to: http://localhost:8080
```

**Open your web browser** (Safari, Chrome, Firefox) and go to:

```
http://localhost:8080
```

## 🎯 Quick Test

Use the GUI to test with random data:

1. **Data Input:**
   - Spectra Directory: `/Users/mattsponheimer/git/dasp/example`
   - Reference File: `/Users/mattsponheimer/git/dasp/example/BoneCollagen.csv`
   - ID Column: `File Number`
   - Target Column: `%Collagen`

2. **Models:** Check Ridge and Lasso

3. **Preprocessing:** Check SNV

4. **CV Folds:** Select 5

5. Click **Run Analysis**

6. Wait 2-5 minutes for results

7. View top 20 models in the results table

## 📁 File Locations

```
/Users/mattsponheimer/git/dasp/julia_port/SpectralPredict/
├── gui.jl                  ← GUI application
├── start_gui.sh            ← Easy startup script
├── GUI_GUIDE.md            ← Detailed GUI documentation
├── QUICK_START.md          ← This file
├── src/                    ← Source code
│   ├── SpectralPredict.jl  ← Main module
│   ├── models.jl           ← ML models
│   ├── preprocessing.jl    ← SNV, derivatives
│   └── ...
└── Project.toml            ← Dependencies
```

## ⚙️ What Works

✅ **Ridge Regression** - Excellent for spectroscopy
✅ **Lasso Regression** - Good for variable selection
✅ **ElasticNet** - Combines Ridge + Lasso
✅ **RandomForest** - Non-linear models
✅ **MLP** - Neural networks

✅ **SNV Preprocessing** - Standard Normal Variate
✅ **Derivatives** - Savitzky-Golay 1st/2nd order
✅ **Raw Data** - No preprocessing

✅ **Cross-Validation** - 3, 5, or 10 folds
✅ **Variable Subsets** - Feature selection
✅ **Region Subsets** - Spectral regions

## ❌ Known Issues

❌ **PLS Model** - Currently not working (use Ridge instead - it works great!)

## 💡 Tips

1. **Use Full Paths** - Always use complete paths like `/Users/...` not `~/...`

2. **Start Simple** - First run with:
   - Ridge only
   - SNV only
   - 5 folds
   - No subsets

3. **Then Expand** - Add more models and options once basic analysis works

4. **Check Terminal** - Error messages appear in the terminal, not the browser

5. **Be Patient** - Analysis can take 5-30 minutes depending on:
   - Number of samples
   - Number of models
   - Number of preprocessing methods
   - CV folds
   - Subsets enabled

## 🛑 Stop the GUI

Press `Ctrl+C` in the terminal window

## 📊 Results

Results are displayed in the browser AND saved to:

```
spectralpredict_results_YYYY-MM-DD_HH-MM-SS.csv
```

Open this CSV file in Excel or any spreadsheet software.

## 🆘 Troubleshooting

### GUI won't start

```bash
# Make sure you're in the right directory:
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict

# Make startup script executable:
chmod +x start_gui.sh

# Try direct command:
julia --project=. gui.jl
```

### Browser shows "Cannot connect"

- Check that the terminal shows "Server running at: http://localhost:8080"
- Try refreshing the browser page
- Try a different browser

### "File not found" errors

- Use **full paths** starting with `/Users/...`
- Check spelling of column names (case-sensitive!)
- Verify files exist

### Analysis too slow

- Use 3 CV folds instead of 10
- Select fewer models
- Disable variable/region subsets
- Use only SNV (skip derivatives)

## 📖 More Help

- **GUI Details:** See `GUI_GUIDE.md`
- **Command Line:** See main `README.md`
- **Examples:** Check `examples/` folder
- **Documentation:** Check `docs/` folder

## 🎓 Example Analysis Flow

1. Start GUI: `./start_gui.sh`
2. Open browser: `http://localhost:8080`
3. Enter your data paths
4. Select Ridge + SNV + 5 folds
5. Click "Run Analysis"
6. Wait for results
7. Check the CSV output file
8. Try again with more models/options

---

**You're ready to analyze spectral data!** 🚀

For detailed instructions, see `GUI_GUIDE.md`
