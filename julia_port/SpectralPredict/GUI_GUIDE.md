# SpectralPredict.jl GUI Guide

## 🚀 Quick Start

### Method 1: Using the Startup Script (Easiest)

```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
./start_gui.sh
```

### Method 2: Direct Julia Command

```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
julia --project=. gui.jl
```

## 📱 Accessing the GUI

After running either command above, you'll see:

```
================================================================================
Starting SpectralPredict GUI Server...
================================================================================

Server running at: http://localhost:8080

Open your web browser and navigate to: http://localhost:8080

Press Ctrl+C to stop the server
================================================================================
```

**Open your web browser** (Safari, Chrome, Firefox) and go to:

```
http://localhost:8080
```

## 🖥️ Using the GUI

### Step 1: Data Input

Fill in these fields with **full paths** to your files:

- **Spectra Directory**: Full path to folder containing your spectral files
  - Example: `/Users/mattsponheimer/data/spectra`
  - Can contain CSV or ASD files

- **Reference CSV File**: Full path to your reference file
  - Example: `/Users/mattsponheimer/data/reference.csv`
  - Must contain sample IDs and target values

- **Sample ID Column Name**: Column name in your reference file for sample IDs
  - Example: `sample_id` or `File Number`

- **Target Variable Column Name**: Column name for the value you want to predict
  - Example: `protein_pct` or `%Collagen`

### Step 2: Select Models

Check the models you want to test:

- ✅ **Ridge** - Works great, recommended for spectroscopy
- ✅ **Lasso** - Good for variable selection
- ✅ **ElasticNet** - Combines Ridge and Lasso
- ✅ **RandomForest** - Non-linear modeling
- ✅ **MLP** - Neural network

**Note:** PLS is currently disabled due to implementation issues.

### Step 3: Preprocessing

Select preprocessing methods:

- **Raw** - No preprocessing
- **SNV** - Standard Normal Variate (recommended)
- **Derivatives** - Savitzky-Golay derivatives
  - Specify orders in the "Derivative Orders" field (e.g., `1,2` for 1st and 2nd)

### Step 4: Advanced Options

- **Cross-Validation Folds**
  - 3 folds = fast testing
  - 5 folds = recommended (default)
  - 10 folds = thorough analysis

- **Variable Subsets** - Test different numbers of features
- **Region Subsets** - Test different spectral regions

### Step 5: Run Analysis

Click **"Run Analysis"** button.

The analysis will:
1. Load your data
2. Test all selected model/preprocessing combinations
3. Run cross-validation
4. Rank results by performance
5. Display top 20 models
6. Save full results to a CSV file

**Time estimate:**
- Small dataset (50 samples): 1-2 minutes
- Medium dataset (100-200 samples): 5-10 minutes
- Large dataset (500+ samples): 15-30 minutes

### Step 6: View Results

Results table shows:
- **Rank** - Best model = Rank 1
- **Model** - Model type (Ridge, Lasso, etc.)
- **Preprocessing** - Applied preprocessing
- **R²** - Higher is better (1.0 = perfect)
- **RMSE** - Lower is better
- **MAE** - Lower is better
- **Variables** - Number of features used

Results are also saved to a CSV file with timestamp:
```
spectralpredict_results_2025-10-30_14-30-45.csv
```

## 📂 Example Usage

### Example 1: BoneCollagen Data

If your data is in the example folder:

```
Spectra Directory: /Users/mattsponheimer/git/dasp/example
Reference File: /Users/mattsponheimer/git/dasp/example/BoneCollagen.csv
ID Column: File Number
Target Column: %Collagen
```

Models: Ridge, Lasso, RandomForest
Preprocessing: SNV, Derivatives
Derivative Orders: 1,2
CV Folds: 10

### Example 2: Custom Data

```
Spectra Directory: /Users/mattsponheimer/Documents/my_spectra
Reference File: /Users/mattsponheimer/Documents/reference.csv
ID Column: sample_id
Target Column: protein_content
```

Models: Ridge, ElasticNet
Preprocessing: SNV
CV Folds: 5

## 🛠️ Troubleshooting

### Issue: "Cannot connect" or "Page not loading"

**Solution:** Make sure the server is running. You should see:
```
Server running at: http://localhost:8080
```

If not, restart the GUI:
```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
./start_gui.sh
```

### Issue: "File not found" error

**Solution:** Check that you're using **full paths**, not relative paths:

❌ Wrong: `data/spectra`
✅ Correct: `/Users/mattsponheimer/data/spectra`

To get the full path:
```bash
# In Terminal, navigate to the folder and run:
pwd
```

### Issue: Analysis takes too long

**Solutions:**
1. Reduce number of CV folds (use 3 instead of 10)
2. Select fewer models
3. Disable variable/region subsets
4. Use only SNV preprocessing (skip derivatives)

### Issue: "Error loading data"

**Check:**
1. Spectra directory exists and contains CSV/ASD files
2. Reference CSV exists and is readable
3. Column names match exactly (case-sensitive!)
4. Sample IDs in reference file match filenames in spectra folder

### Issue: Want to stop the analysis

Press `Ctrl+C` in the terminal window to stop the server.

## 📊 Understanding Results

### R² (R-squared)
- Ranges from -∞ to 1.0
- 1.0 = perfect predictions
- 0.8+ = excellent
- 0.6-0.8 = good
- <0.5 = poor

### RMSE (Root Mean Squared Error)
- Lower is better
- Same units as your target variable
- Example: If predicting protein %, RMSE of 2.0 means average error of ±2%

### MAE (Mean Absolute Error)
- Lower is better
- Average absolute difference between predictions and actual values
- Less sensitive to outliers than RMSE

## 💾 Output Files

Results are automatically saved to:
```
spectralpredict_results_YYYY-MM-DD_HH-MM-SS.csv
```

This file contains:
- All tested configurations
- Complete metrics for each
- Hyperparameter details
- Ranking scores

Open in Excel, Numbers, or any spreadsheet software.

## 🔒 Security Note

The GUI runs locally on your computer only (localhost:8080).
No data is sent to the internet. All processing happens on your machine.

## 🆘 Getting Help

If you encounter issues:

1. **Check the Terminal** - Error messages appear in the terminal where you started the GUI
2. **Check paths** - Make sure all file paths are correct and complete
3. **Check data format** - Ensure CSV files are properly formatted
4. **Restart GUI** - Press Ctrl+C and restart with `./start_gui.sh`

## 📞 Common Commands

### Start GUI
```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
./start_gui.sh
```

### Stop GUI
Press `Ctrl+C` in the terminal

### Access GUI
Open browser to: `http://localhost:8080`

### Find Full Path
```bash
cd /path/to/your/folder
pwd
```

---

**Ready to analyze your spectral data!** 🚀
