# R Validation Test Suite

This validation test suite verifies that our Python spectral analysis models produce results equivalent to their R package counterparts. This is critical for ensuring that we're not getting systematically worse results in Python compared to R.

## Overview

The validation suite compares Python implementations against the following R packages:

| Python Library | R Package | Models |
|---------------|-----------|---------|
| `sklearn.cross_decomposition` | `pls` | PLS Regression, PLS-DA |
| `sklearn.ensemble` | `randomForest` | Random Forest |
| `xgboost` | `xgboost` | XGBoost |
| `sklearn.linear_model` | `glmnet` | Ridge, Lasso, ElasticNet |
| `sklearn.svm` | `e1071` | SVR/SVM |

## Directory Structure

```
dasp/
├── tests/
│   └── test_r_validation.py           # Python test suite
├── r_validation_scripts/
│   ├── generate_sample_data.py        # Generate synthetic NIR data
│   ├── pls_comparison.R               # PLS validation
│   ├── random_forest_comparison.R     # Random Forest validation
│   ├── xgboost_comparison.R           # XGBoost validation
│   ├── glmnet_comparison.R            # Ridge/Lasso/ElasticNet validation
│   ├── compare_results.py             # Compare Python vs R results
│   └── results/
│       ├── python/                    # Python model results (JSON)
│       ├── r/                         # R model results (JSON)
│       └── *.csv                      # Train/test data files
└── data/
    ├── sample_nir_regression_small.csv   # 100 samples × 500 features
    ├── sample_nir_regression_medium.csv  # 150 samples × 800 features
    ├── sample_nir_regression_large.csv   # 200 samples × 1000 features
    └── sample_nir_classification.csv     # 150 samples × 800 features, 3 classes
```

## Installation

### Python Requirements

```bash
# Install Python dependencies (if not already installed)
pip install numpy pandas scikit-learn xgboost scipy pytest
```

### R Requirements

```R
# Install R packages
install.packages("pls")
install.packages("randomForest")
install.packages("xgboost")
install.packages("glmnet")
install.packages("e1071")
install.packages("jsonlite")
install.packages("caret")  # Optional, for additional validation
```

Alternatively, install all at once:

```R
install.packages(c("pls", "randomForest", "xgboost", "glmnet", "e1071", "jsonlite", "caret"))
```

## Quick Start

### Step 1: Generate Sample Data

Generate synthetic NIR spectral data with known properties:

```bash
python r_validation_scripts/generate_sample_data.py
```

This creates:
- `data/sample_nir_regression_small.csv` (100 samples, 500 wavelengths)
- `data/sample_nir_regression_medium.csv` (150 samples, 800 wavelengths)
- `data/sample_nir_regression_large.csv` (200 samples, 1000 wavelengths)
- `data/sample_nir_classification.csv` (150 samples, 800 wavelengths, 3 classes)

All datasets use **random seed 42** for reproducibility.

### Step 2: Run Python Tests

Train Python models and export results:

```bash
# Run all Python validation tests
pytest tests/test_r_validation.py -v -s

# Or run individual tests
pytest tests/test_r_validation.py::test_pls_regression -v -s
pytest tests/test_r_validation.py::test_randomforest_regression -v -s
pytest tests/test_r_validation.py::test_xgboost_regression -v -s
```

This creates results in `r_validation_scripts/results/python/`:
- `pls_regression.json`
- `rf_regression.json`
- `xgb_regression.json`
- `ridge_regression.json`
- `lasso_regression.json`
- `elasticnet_regression.json`

### Step 3: Run R Scripts

Train equivalent R models:

```bash
# Run all R validation scripts
Rscript r_validation_scripts/pls_comparison.R
Rscript r_validation_scripts/random_forest_comparison.R
Rscript r_validation_scripts/xgboost_comparison.R
Rscript r_validation_scripts/glmnet_comparison.R
```

This creates results in `r_validation_scripts/results/r/` with the same filenames.

### Step 4: Compare Results

Compare Python and R results:

```bash
# Compare all models
python r_validation_scripts/compare_results.py --model all

# Or compare individual models
python r_validation_scripts/compare_results.py --model pls_regression
python r_validation_scripts/compare_results.py --model rf_regression
python r_validation_scripts/compare_results.py --model xgb_regression
```

## Validation Criteria

The validation suite checks the following criteria for each model:

### 1. Prediction Accuracy

**Passing criteria:**
- **Deterministic models** (PLS, Ridge, Lasso, ElasticNet, XGBoost): predictions match within `1e-6`
- **Stochastic models** (Random Forest): predictions correlate > 0.95, RMSE within 10%

**Why?** Deterministic models should produce identical results with the same seed. Stochastic models may differ slightly due to different RNG implementations.

### 2. Performance Metrics

**Passing criteria:**
- RMSE matches within `1e-6` (deterministic) or 5% (stochastic)
- R² matches within `1e-4`

### 3. Feature Importances

**Passing criteria:**
- Correlation between Python and R importances > 0.9
- Top 10 important features overlap > 70%

**Why?** Feature importance calculation may differ slightly between implementations, but should be highly correlated.

### 4. Model Coefficients

**Passing criteria:**
- Coefficients match within `1e-6` (Ridge, Lasso, ElasticNet)
- Number of non-zero coefficients matches within 10% (Lasso, ElasticNet)

## Expected Results

### PLS Regression

✓ **Expected:** Predictions match exactly (< 1e-10)

PLS is a deterministic algorithm with well-defined matrix operations. Python sklearn and R pls package should produce identical results.

**Common issues:**
- Sign flips in loadings (normal, loadings are direction-agnostic)
- Different scaling defaults (`scale=False` in Python vs `scale=FALSE` in R)

### Random Forest

⚠ **Expected:** Predictions correlate highly (> 0.95) but not identical

Random Forest is stochastic even with the same seed due to:
- Different random number generators (Python vs R)
- Different tie-breaking rules
- Different splitting criteria implementations

**Passing criteria:**
- Test RMSE within 10%
- Feature importances correlation > 0.9
- Top 10 features overlap > 70%

### XGBoost

✓ **Expected:** Predictions match very closely (< 1e-4)

XGBoost is designed to be consistent across platforms. Results should match very closely.

**Common issues:**
- Parameter naming differences (`n_estimators` vs `nrounds`, `learning_rate` vs `eta`)
- Different default parameters
- Numerical precision differences

### Ridge/Lasso/ElasticNet (glmnet)

✓ **Expected:** Predictions match closely (< 1e-6)

Linear models are deterministic but may show small differences due to:
- Different convergence criteria
- Different coordinate descent implementations
- Numerical precision

**Parameter naming gotcha:**
```
Python sklearn              R glmnet
------------------         ------------------
Ridge(alpha=1.0)           glmnet(alpha=0, lambda=1.0)
Lasso(alpha=1.0)           glmnet(alpha=1, lambda=1.0)
ElasticNet(                glmnet(
  alpha=1.0,                 alpha=0.5,
  l1_ratio=0.5)              lambda=1.0)
```

**sklearn alpha = glmnet lambda** (regularization strength)
**sklearn l1_ratio = glmnet alpha** (L1/L2 mixing parameter)

## Troubleshooting

### Issue: "Data file not found"

**Solution:** Run the data generation script first:
```bash
python r_validation_scripts/generate_sample_data.py
```

### Issue: "R package not found"

**Solution:** Install missing R packages:
```R
install.packages("package_name")
```

### Issue: Predictions differ significantly

**Possible causes:**
1. **Different random seeds:** Check that both Python and R use seed 42
2. **Different hyperparameters:** Verify parameters match exactly
3. **Different preprocessing:** Ensure no scaling/normalization differences
4. **Different data:** Verify train/test split is identical

**Debug steps:**
```python
# Check Python predictions
import json
with open('r_validation_scripts/results/python/pls_regression.json') as f:
    py_results = json.load(f)
    print(py_results['predictions']['test'][:10])  # First 10 predictions
```

```R
# Check R predictions
library(jsonlite)
r_results <- read_json("r_validation_scripts/results/r/pls_regression.json")
print(r_results$predictions$test[1:10])  # First 10 predictions
```

### Issue: Feature importances poorly correlated

**For Random Forest:** This is expected due to different implementations. Check that:
- Correlation > 0.7 (acceptable)
- Top features have reasonable overlap

**For XGBoost:** Should correlate > 0.95. If not:
- Check that importance type matches ("gain" in R vs default in Python)
- Verify hyperparameters are identical

## Interpreting Results

### Example Output

```
================================================================================
COMPARING: PLS_REGRESSION
================================================================================

Loading results...
  Python: r_validation_scripts/results/python/pls_regression.json
  R: r_validation_scripts/results/r/pls_regression.json

Prediction Comparison
--------------------------------------------------------------------------------

TRAIN Predictions:
  Samples: 105
  Max absolute difference: 3.14e-11
  Mean absolute difference: 8.42e-12
  Correlation: 1.000000
  Mean relative error: 0.000%
✓ PASS: train predictions match within tolerance (1.00e-06)

TEST Predictions:
  Samples: 45
  Max absolute difference: 2.87e-11
  Mean absolute difference: 7.21e-12
  Correlation: 1.000000
  Mean relative error: 0.000%
✓ PASS: test predictions match within tolerance (1.00e-06)

Performance Metrics Comparison
--------------------------------------------------------------------------------

train_rmse:
  Python: 1.234567
  R:      1.234567
  Difference: 0.000000 (0.000%)
✓ PASS: train_rmse matches within tolerance

Overall Assessment
--------------------------------------------------------------------------------
✓ PASS: All comparisons passed!
```

### What to Look For

✓ **Green checkmarks:** Test passed
⚠ **Yellow warnings:** Test marginally passed, review recommended
✗ **Red X's:** Test failed, investigation required

## Advanced Usage

### Custom Hyperparameters

To test with different hyperparameters, edit both Python and R scripts:

**Python (test_r_validation.py):**
```python
# Change hyperparameters
n_components = 15  # Instead of 10
```

**R (pls_comparison.R):**
```R
# MUST match Python
n_components <- 15  # Instead of 10
```

### Custom Datasets

To validate on your own data:

1. Format your data as CSV with features as columns and last column as target
2. Update `load_regression_data()` in `test_r_validation.py`
3. Update data loading in R scripts
4. Ensure same train/test split (use `train_test_split` with fixed seed)

### Adding New Models

To add a new model (e.g., SVM):

1. Add test function to `test_r_validation.py`
2. Create R comparison script `r_validation_scripts/svm_comparison.R`
3. Update `compare_results.py` to include new model
4. Run validation suite

## FAQ

### Q: Why do Random Forest results differ?

A: Random Forest uses different RNG and tie-breaking rules in Python vs R. This is expected and doesn't indicate a problem. Look for correlation > 0.95.

### Q: Should loadings have the same sign?

A: No, PLS loadings can flip sign between implementations. The direction doesn't matter, only the magnitude.

### Q: What tolerance should I use?

A:
- Deterministic models (PLS, Ridge, XGBoost): `1e-6`
- Stochastic models (Random Forest): 5-10% RMSE difference
- Feature importances: correlation > 0.9

### Q: Can I use this on real spectral data?

A: Yes! Replace the synthetic data with your NIR/VIS/Raman data. Ensure:
- Same train/test split in both Python and R
- Same preprocessing (if any)
- Same random seed

### Q: How do I know if I'm getting better models in R?

A: Compare test RMSE and R² from the validation suite:
- If R consistently outperforms Python by >10%, investigate hyperparameters
- Small differences (<5%) are normal and not concerning
- XGBoost and PLS should be nearly identical

## References

### R Packages

- **pls:** Mevik, B.-H., & Wehrens, R. (2007). The pls package: Principal component and partial least squares regression in R. *Journal of Statistical Software*, 18(2), 1-23.
- **randomForest:** Liaw, A., & Wiener, M. (2002). Classification and regression by randomForest. *R News*, 2(3), 18-22.
- **xgboost:** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- **glmnet:** Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

### Python Packages

- **scikit-learn:** Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.
- **xgboost:** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review example output in this README
3. Examine the comparison script output for specific differences
4. Open an issue with:
   - Model name
   - Comparison output
   - Python and R versions
   - Package versions

## License

This validation suite is part of the DASP (Data Analysis for Spectral Prediction) project.
