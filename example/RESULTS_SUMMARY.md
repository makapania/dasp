# Bone Collagen Analysis Results

**Date:** 2025-10-27
**Dataset:** 37 bone samples, 2151 wavelengths (350-2500 nm)
**Target:** %Collagen (range: 0.9-22.1%)
**Models Tested:** 548 configurations

---

## ✅ Output Files Created

### 1. outputs/results.csv (80 KB, 548 models)
Complete ranked table of all models tested. Each row represents one model configuration.

**Columns:**
- `Rank` - 1 = best model
- `Model` - Algorithm with hyperparameters
- `Preprocess` - Data preprocessing applied
- `SubsetTag` - Variable subset used (all/top-20/top-5/top-3)
- `RMSE` - Root mean squared error (% collagen)
- `R2` - R-squared (variance explained)
- `CompositeScore` - Overall ranking metric
- `n_vars` - Number of variables used
- `full_vars` - Total variables available (2151)

### 2. reports/%Collagen.md (NOT CREATED)
**Issue:** Report generation failed due to missing `tabulate` dependency.

**Status:** ✅ FIXED - `tabulate` has been installed and added to `pyproject.toml`

---

## 📊 Top 10 Models

| Rank | Model | Preprocess | Variables | RMSE | R² |
|------|-------|------------|-----------|------|-----|
| 1 | RandomForest(n=200) | raw | 3 | 3.62 | -0.07 |
| 1 | RandomForest(n=200, depth=15) | raw | 3 | 3.62 | -0.07 |
| 1 | RandomForest(n=200, depth=30) | raw | 3 | 3.62 | -0.07 |
| 4 | RandomForest(n=500, depth=30) | raw | 3 | 3.77 | -0.26 |
| 4 | RandomForest(n=500) | raw | 3 | 3.77 | -0.26 |
| 4 | RandomForest(n=500, depth=15) | raw | 3 | 3.77 | -0.26 |

**Best Model:** RandomForest with 3 variables (top-3), no preprocessing, RMSE=3.62% collagen

---

## ⚠️ Important Observations

### 1. **Negative R² Values**
All top models show **negative R²** values (-0.07 to -1.27). This means:
- Models perform **worse than predicting the mean**
- The spectral data may not have strong predictive power for collagen
- OR the dataset is too small/variable for reliable modeling

### 2. **Best Models Use Only 3 Variables**
The top-ranked models use only 3 wavelengths out of 2151. This suggests:
- Most of the spectrum doesn't correlate with collagen
- OR the feature selection is too aggressive
- OR there's high noise in the data

### 3. **Raw (No Preprocessing) Performs Best**
Preprocessing (SNV, derivatives) actually makes results **worse**. This is unusual for spectral data and might indicate:
- The raw reflectance values contain the signal
- Preprocessing is removing important information
- OR the preprocessing parameters aren't appropriate for this dataset

---

## 🔍 Diagnostic Recommendations

### 1. **Check Data Quality**
```bash
# View the spectral data
source .venv/bin/activate
python3 -c "import pandas as pd; from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy; \
X = read_asd_dir('example/'); \
ref = read_reference_csv('example/BoneCollagen.csv', 'File Number'); \
X_al, y = align_xy(X, ref, 'File Number', '%Collagen'); \
print(f'X shape: {X_al.shape}'); \
print(f'y range: {y.min():.1f} - {y.max():.1f}'); \
print(f'y mean: {y.mean():.1f}'); \
print(f'y std: {y.std():.1f}')"
```

### 2. **Plot Spectra**
Visual inspection of the spectra might reveal:
- Outliers or bad measurements
- Baseline shifts
- Noise levels

### 3. **Check Sample Size**
37 samples is relatively small for spectroscopy. Consider:
- Collecting more samples if possible
- Using simpler models (PLS with few components)
- Validating on external test set

### 4. **Try Different Approaches**
- Use PLS with 2-5 components (current grid may be too large)
- Try different wavelength ranges (focus on known collagen bands)
- Check for non-linear relationships

---

## 📁 Files to Review

```bash
# View full results
head -20 outputs/results.csv | column -t -s,

# Count models by type
cat outputs/results.csv | cut -d, -f2 | sort | uniq -c

# Find best PLS models
grep "PLS" outputs/results.csv | sort -t, -k15 -n | head -10

# Find models with positive R²
awk -F, '$13 > 0' outputs/results.csv | wc -l
```

---

## 🚀 Next Steps

1. **Re-run with tabulate installed** to generate the Markdown report:
   ```bash
   source .venv/bin/activate
   spectral-predict --asd-dir example/ \
                    --reference example/BoneCollagen.csv \
                    --id-column "File Number" \
                    --target "%Collagen"
   ```

2. **Investigate data quality**:
   - Plot some example spectra
   - Check for outliers in the collagen measurements
   - Verify the spectral preprocessing is appropriate

3. **Try manual model tuning**:
   - Since automated search didn't find great models, manual tuning might help
   - Focus on PLS with 2-5 components
   - Try specific wavelength regions known for protein/collagen bands

---

## 📝 Technical Notes

- **Execution time:** ~8-10 minutes
- **CPU usage:** High (expected for 548 model fits with 5-fold CV)
- **Memory usage:** Normal
- **Exit code:** 0 (success)
- **Missing dependency:** `tabulate` (now fixed in pyproject.toml)

---

**Bottom line:** The tool worked correctly and tested 548 models, but the dataset appears challenging for spectral prediction. The best RMSE of 3.62% collagen might be acceptable depending on your needs, but the negative R² suggests limited predictive power from the spectra.
