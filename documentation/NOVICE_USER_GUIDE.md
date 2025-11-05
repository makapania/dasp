# 🚀 Spectral Predict - Quick Start Guide for Novice Users

**Welcome!** This guide will help you run spectral analysis in just a few clicks.

---

## ⚡ Super Quick Start (3 Steps)

### Step 1: Double-Click to Launch
1. Find the file: `RUN_SPECTRAL_PREDICT.bat`
2. Double-click it
3. Wait for the GUI to appear (10-15 seconds)

### Step 2: Load Your Data
1. Click **"Browse"** next to "Spectral File Directory"
2. Select the folder containing your spectral files (.asd, .csv, or .spc files)
3. Click **"Browse"** next to "Reference CSV"
4. Select your reference file with sample IDs and target values
5. Click **"Load Data & Generate Plots"**

### Step 3: Run Analysis
1. Go to the **"Analysis Configuration"** tab
2. Keep the default settings (they work great!)
3. Click **"▶ Run Analysis"**
4. Wait for results (shown in the **"Results"** tab)

**That's it!** You're now running world-class spectral analysis powered by Julia.

---

## 📁 What Files Do I Need?

### Spectral Data
Your spectral measurements in one of these formats:
- **ASD files** (.asd) - Field spectroradiometer data
- **CSV files** (.csv) - Spreadsheet with wavelengths as columns
- **SPC files** (.spc) - GRAMS/Thermo Galactic format

### Reference File
A CSV file with:
- Column 1: File names (matching your spectral files)
- Column 2: Sample IDs
- Column 3: Target variable (e.g., protein content, moisture, etc.)

**Example:**
```
filename,sample_id,protein_pct
sample001.asd,S001,12.5
sample002.asd,S002,15.3
sample003.asd,S003,11.8
```

---

## 🎯 Understanding the Results

### Results Tab
Shows all tested models ranked by performance:

- **Rank**: Lower is better (#1 is the best model)
- **Model**: Algorithm used (PLS, Random Forest, etc.)
- **Preprocess**: Data preprocessing method
- **RMSE**: Error (lower is better)
- **R²**: Fit quality (closer to 1.0 is better)
- **SubsetTag**: Which features/wavelengths were used

### What's a Good Result?
- **R² > 0.8**: Excellent model
- **R² 0.6-0.8**: Good model
- **R² < 0.6**: Model might need improvement

---

## ⚙️ Changing Settings (Optional)

### Tab 2: Analysis Configuration

**Most users can skip this** - defaults work great! But if you want to customize:

#### Models to Test
- ✓ **PLS**: Fast, linear, great for spectral data
- ✓ **Random Forest**: Handles nonlinear relationships
- ✓ **MLP**: Deep learning (slower but powerful)
- ✓ **Neural Boosted**: Advanced gradient boosting

#### Preprocessing Methods
- ✓ **SNV**: Removes scatter effects
- ✓ **SG1**: 1st derivative (baseline removal)
- ✓ **SG2**: 2nd derivative (peak enhancement)

#### Subset Analysis
- ✓ **Top-N Variables**: Test models with only the most important wavelengths
- ✓ **Spectral Regions**: Identify key wavelength regions

**Pro Tip**: Enable everything for comprehensive analysis, or disable subsets for faster results.

---

## 🔧 Troubleshooting

### "Python not found"
**Solution**: Install Python 3.8+ from [python.org](https://python.org)
- During installation, check ✓ "Add Python to PATH"

### GUI doesn't open
**Solution**:
1. Open Command Prompt
2. Type: `python spectral_predict_gui_optimized.py`
3. Look for error messages and send to support

### Analysis takes forever
**Solution**:
- Disable subset analysis (uncheck the boxes in Tab 2)
- Select fewer models (just PLS for quick results)
- Reduce variable counts (just N=10, N=20)

### Results look bad (low R²)
**Possible causes**:
- Poor data quality (noisy spectra)
- Wrong target variable
- Mismatched samples between spectra and reference file

**Try**:
- Check your data files are correct
- Try different preprocessing methods
- Use more samples for training (>50 recommended)

---

## 📊 Advanced Features

### Refine Model (Tab 5)
1. Go to **Results** tab
2. Double-click any result row
3. Switch to **Refine Model** tab
4. Adjust parameters and click **"Run Refined Model"**

### Save Results
Results are automatically saved to the `outputs/` folder:
- `results_[target]_[timestamp].csv` - Full results table
- See **Reports** folder for detailed analysis

---

## 💡 Tips for Best Results

### Data Preparation
1. **Remove outliers**: Bad samples hurt performance
2. **Consistent conditions**: Same instrument, same settings
3. **Enough samples**: 50+ samples recommended, 100+ is better
4. **Balanced range**: Cover full range of target values

### Model Selection
- **Start simple**: Try PLS first
- **Add complexity**: If PLS R² < 0.7, try Random Forest or MLP
- **Use subsets**: Find which wavelengths matter most

### Validation
- **R² is training performance**: Real-world may differ
- **Check top variables**: Do they make chemical sense?
- **Use held-out test set**: For final validation

---

## 🆘 Getting Help

### Common Questions
**Q: Which model should I use?**
A: Start with PLS. It's fast and works great for spectral data. If R² is low, try Random Forest next.

**Q: What's the difference between preprocessing methods?**
A:
- **SNV**: Removes sample-to-sample variation
- **Derivatives**: Removes baseline, enhances peaks
- **Raw**: No preprocessing (rarely best)

**Q: How long should analysis take?**
A:
- Small dataset (50 samples): 2-5 minutes
- Medium dataset (200 samples): 10-20 minutes
- Large dataset (500+ samples): 30-60 minutes

**Q: Can I use this for classification?**
A: Yes! If your target column has categories (e.g., "good"/"bad"), the system automatically detects classification tasks.

### Need More Help?
1. Check the detailed documentation in `docs/` folder
2. Read `README.md` for technical details
3. Contact: [support contact info]

---

## ✅ Quick Checklist

Before running analysis:
- [ ] Spectral files in a single folder
- [ ] Reference CSV file ready
- [ ] Column names known (sample ID, target variable)
- [ ] Data quality checked (no corrupted files)

For best results:
- [ ] 50+ samples
- [ ] Full wavelength range (visible to near-infrared)
- [ ] Consistent measurement conditions
- [ ] Representative sample of variation

---

## 🎓 Learning More

### Recommended Reading
- **PLS regression**: Look up "Partial Least Squares spectroscopy"
- **Spectral preprocessing**: Read about SNV and Savitzky-Golay
- **Model validation**: Learn about cross-validation

### Practice Datasets
Check the `examples/` folder for:
- Sample spectral data
- Reference files
- Expected results

---

## 🚀 You're Ready!

Remember:
1. **Double-click** `RUN_SPECTRAL_PREDICT.bat`
2. **Load data** in Tab 1
3. **Run analysis** in Tab 2
4. **View results** in Tab 4

**That's all there is to it!** The system handles all the complex machine learning automatically.

Good luck with your analysis! 🎉

---

*Last updated: October 2025*
*Version: 1.0 - Julia-Powered Edition*
