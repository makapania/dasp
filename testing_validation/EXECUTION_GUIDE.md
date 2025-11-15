# DASP Validation Testing - Complete Execution Guide

**Status:** All scripts created, ready to execute
**Date:** 2025-11-14
**Time to complete:** 30-60 minutes (automated)

---

## 🚀 Quick Start (Automated)

### Option 1: Run Everything Automatically

```bash
cd testing_validation

# Install R packages (first time only, 5-10 min)
cd r_scripts
Rscript install_packages.R
cd ..

# Run all tests (30-60 min)
python run_all_tests.py
```

**That's it!** The automated runner will:
1. Run R regression tests (both datasets)
2. Run DASP regression tests (both datasets)
3. Compare results and generate reports

---

## 📋 Manual Step-by-Step Execution

If you prefer to run steps individually or troubleshoot:

### Step 1: Install R Packages (One-time setup)

```bash
cd testing_validation/r_scripts
Rscript install_packages.R
```

**Expected output:**
- Installs 9+ R packages
- Takes 5-10 minutes
- May show warnings (normal)

**Troubleshooting:**
- If `lightgbm` fails: Optional, can skip
- If `mdatools` fails: Optional, needed for SPA/iPLS later

### Step 2: Run R Regression Tests

```bash
# From testing_validation/r_scripts/
Rscript regression_models_comprehensive.R
```

**Expected output:**
- Tests 2 datasets: Bone Collagen + Enamel d13C
- ~45 model configurations total
- Takes 15-30 minutes
- Creates JSON files in `../results/r_regression/`

**What's being tested:**
- Bone Collagen: 36 train / 13 test
- Enamel d13C: 105 train / 35 test
- Models: PLS, Ridge, Lasso, RF, XGBoost (5 model types)
- ~23 configs per dataset = 46 total

### Step 3: Run DASP Regression Tests

```bash
# From testing_validation/
python dasp_regression.py
```

**Expected output:**
- Tests same 2 datasets
- Same ~45 model configurations
- Takes 10-20 minutes
- Creates JSON files in `results/dasp_regression/`

**What's being tested:**
- Identical datasets and train/test splits
- Identical hyperparameters (matched to R)
- Same models: PLS, Ridge, Lasso, RF, XGBoost

### Step 4: Compare Results

```bash
# From testing_validation/
python compare_regression_results.py
```

**Expected output:**
- Loads both DASP and R results
- Compares metrics model-by-model
- Statistical tests (paired t-tests, correlation)
- Takes 1-2 minutes
- Creates:
  - `results/comparisons/bone_collagen_comparison.csv`
  - `results/comparisons/d13c_comparison.csv`
  - `results/comparisons/comparison_report.md`
  - `results/comparisons/comparison_summary.json`

---

## 📊 What Gets Created

### Directory Structure After Execution

```
testing_validation/
└── results/
    ├── r_regression/
    │   ├── bone_collagen/
    │   │   ├── pls_results.json
    │   │   ├── ridge_results.json
    │   │   ├── lasso_results.json
    │   │   ├── rf_results.json
    │   │   ├── xgboost_results.json
    │   │   └── all_models_summary.json
    │   └── d13c/
    │       └── (same structure)
    │
    ├── dasp_regression/
    │   ├── bone_collagen/
    │   │   └── (same structure as R)
    │   └── d13c/
    │       └── (same structure as R)
    │
    └── comparisons/
        ├── bone_collagen_comparison.csv
        ├── d13c_comparison.csv
        ├── comparison_summary.json
        └── comparison_report.md  ← Read this!
```

### Key Output Files

1. **comparison_report.md** - Comprehensive Markdown report
   - Executive summary
   - Detailed metrics comparison
   - Statistical tests
   - Interpretation and conclusions

2. **{dataset}_comparison.csv** - Model-by-model comparison tables
   - Columns: Model, DASP_R2, R_R2, R2_Diff, RMSE_Diff, etc.
   - Easy to load in Excel or pandas

3. **comparison_summary.json** - Summary statistics
   - Programmatic access to results
   - Mean differences, correlations, p-values

---

## 🎯 Success Criteria

### What to Check in Results

1. **R² Agreement:**
   - Mean difference < ±2% for deterministic models (PLS, Ridge, Lasso)
   - Mean difference < ±5% for stochastic models (RF, XGBoost)

2. **Correlation:**
   - R² correlation > 0.99 (excellent)
   - R² correlation > 0.95 (good)
   - R² correlation < 0.95 (investigate)

3. **Statistical Significance:**
   - Paired t-test p-value > 0.05 (no systematic difference)
   - p-value < 0.05 (systematic bias, investigate)

4. **Best Models:**
   - d13C dataset: Expect R² ~0.75-0.90
   - Bone collagen: Expect R² ~0.70-0.85

---

## 🔧 Troubleshooting

### Issue: R packages won't install

**Solution:**
```R
# Try installing packages one by one
install.packages("pls")
install.packages("glmnet")
# etc.
```

### Issue: "Cannot find module" error in Python

**Solution:**
```bash
# Ensure you're using the right Python environment
python --version  # Should be 3.10+

# Install any missing packages
pip install pandas numpy scikit-learn xgboost scipy
```

### Issue: R script shows "object not found" error

**Problem:** Column names may have X. prefix

**Solution:**
- Script already handles this with `X.Collagen` fallback
- If still fails, check CSV column names manually

### Issue: Comparison script finds no common models

**Problem:** Model naming mismatch between DASP and R

**Solution:**
- Check JSON files in both `r_regression` and `dasp_regression`
- Ensure both tests completed successfully
- Model names should match (e.g., "PLS_10", "Ridge_1.000")

### Issue: Tests are too slow

**Options:**
1. **Run fewer models:**
   - Edit scripts to test subset of hyperparameters
   - Example: Only test PLS with [2, 10, 20] components

2. **Test one dataset:**
   - Comment out bone collagen or d13C sections
   - Focus on d13C (larger, more robust)

3. **Use parallel processing:**
   - R randomForest already uses multiple cores
   - Python: Add `n_jobs=-1` to scikit-learn models

---

## 📈 Expected Runtime

| Task | Duration | Bottleneck |
|------|----------|------------|
| R package install | 5-10 min | Internet speed |
| R regression tests | 15-30 min | XGBoost, RandomForest |
| DASP regression tests | 10-20 min | XGBoost |
| Comparison | 1-2 min | I/O |
| **Total** | **30-60 min** | - |

**Note:** d13C dataset (n=105) is slower than bone collagen (n=36)

---

## 🎓 Understanding the Results

### Example comparison_report.md Output

```markdown
### Enamel d13C

#### R² Comparison

| Metric | DASP | R | Difference |
|--------|------|---|------------|
| Mean | 0.8235 | 0.8221 | 0.0014 |
| Max Difference | | | 0.0089 |
| Correlation | | | 0.9987 |

**Paired t-test:** t=1.23, p=0.234

✅ R² difference (0.17%) is within ±2% threshold
✅ Correlation (0.9987) indicates strong agreement
✅ No significant difference (p=0.234 > 0.05)
```

**Interpretation:**
- **0.17% difference:** Excellent agreement
- **Correlation 0.9987:** Near-perfect match
- **p=0.234:** No systematic bias

This indicates **DASP and R produce equivalent results**.

### Example of a Problem

```markdown
❌ R² difference (8.5%) exceeds ±5% threshold
⚠️ Correlation (0.912) is good but below 0.99
⚠️ Significant difference detected (p=0.003 < 0.05)
```

**Action required:** Investigate parameter mismatch or implementation difference

---

## 🔍 Next Steps After Initial Tests

### If Tests Pass (R² within ±2-5%)

1. **Run classification tests** (if needed)
   ```bash
   cd r_scripts
   Rscript classification_models.R
   # Then create dasp_classification.py
   ```

2. **Test variable selection**
   - VIP, SPA, UVE, iPLS methods
   - Compare selected wavelengths

3. **Hyperparameter sensitivity analysis**
   - Test broader ranges
   - Create performance heatmaps

4. **Performance benchmarking**
   - Measure runtime for each model
   - Memory usage profiling

### If Tests Fail (large discrepancies)

1. **Check for bugs:**
   - Verify data loading (same samples in same order?)
   - Check preprocessing (applied the same way?)
   - Verify random seeds

2. **Isolate the problem:**
   - Test one model at a time
   - Start with simplest (PLS with few components)
   - Compare predictions sample-by-sample

3. **Parameter debugging:**
   - Print hyperparameters from both R and DASP
   - Ensure exact match (especially Ridge/Lasso alpha vs. lambda confusion)

---

## 📝 Command Reference

### All Tests (Automated)
```bash
python run_all_tests.py
```

### Individual Steps
```bash
# R tests only
cd r_scripts && Rscript regression_models_comprehensive.R

# DASP tests only
python dasp_regression.py

# Comparison only (requires both R and DASP results)
python compare_regression_results.py
```

### Advanced Options
```bash
# Skip tests that already ran
python run_all_tests.py --skip-r      # Use existing R results
python run_all_tests.py --skip-dasp   # Use existing DASP results

# Test specific dataset (when implemented)
python run_all_tests.py --datasets d13c
```

---

## 🐛 Debugging Commands

### Check if results exist
```bash
# R results
ls results/r_regression/bone_collagen/
ls results/r_regression/d13c/

# DASP results
ls results/dasp_regression/bone_collagen/
ls results/dasp_regression/d13c/
```

### Inspect JSON results
```bash
# View R results
cat results/r_regression/d13c/pls_results.json | head -50

# View DASP results
cat results/dasp_regression/d13c/pls_results.json | head -50

# Compare model counts
grep -c "model" results/r_regression/d13c/all_models_summary.json
grep -c "model" results/dasp_regression/d13c/all_models_summary.json
```

### Python debugging
```python
# In Python console
import json
from pathlib import Path

# Load results
with open('results/dasp_regression/d13c/pls_results.json') as f:
    dasp_pls = json.load(f)

with open('results/r_regression/d13c/pls_results.json') as f:
    r_pls = json.load(f)

# Check keys match
print(f"DASP keys: {list(dasp_pls.keys())}")
print(f"R keys: {list(r_pls.keys())}")

# Compare specific model
print(f"DASP PLS(10): R²={dasp_pls['10']['r2']:.4f}")
print(f"R PLS(10): R²={r_pls['10']['r2']:.4f}")
```

---

## 📚 Additional Resources

**Documentation:**
- `README.md` - Infrastructure overview
- `DATASETS_SUMMARY.md` - Dataset details
- `TESTING_PLAN_STATUS.md` - Full 10-phase plan
- `IMPLEMENTATION_SUMMARY.md` - What's been built

**Scripts:**
- `run_all_tests.py` - Automated runner
- `dasp_regression.py` - DASP regression tests
- `compare_regression_results.py` - Comparison & reporting
- `r_scripts/regression_models_comprehensive.R` - R regression tests

---

## ✅ Checklist

Before starting:
- [ ] R installed and accessible (`Rscript --version`)
- [ ] Python 3.10+ installed (`python --version`)
- [ ] Required Python packages (`pip list | grep sklearn`)
- [ ] Enough disk space (~500 MB for results)
- [ ] 30-60 minutes available

During execution:
- [ ] R packages installed successfully
- [ ] R regression tests completed (check for errors)
- [ ] DASP regression tests completed
- [ ] Comparison script ran without errors

After completion:
- [ ] Review `comparison_report.md`
- [ ] Check R² agreement (within ±2-5%?)
- [ ] Check correlation (>0.95?)
- [ ] Check p-values (>0.05?)
- [ ] Identify best models for each dataset

---

**Ready to run?** Execute:

```bash
cd testing_validation
python run_all_tests.py
```

Then review: `results/comparisons/comparison_report.md`

**Good luck! 🚀**
