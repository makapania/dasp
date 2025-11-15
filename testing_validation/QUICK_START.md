# DASP Testing Validation - Quick Start Guide

**Goal:** Validate DASP against R and Python implementations for scientific accuracy

---

## What's Been Built (Phase 1-2 ✓)

### 1. Test Datasets ✓
- **2 datasets:** Bone Collagen (49 samples), Enamel d13C (140 samples)
- **5 task types:** 2 regression + 3 classification
- **Train/test splits:** 75/25 stratified (random seed=42)
- **Spectral data:** 2,151 wavelengths (350-2500 nm)
- **Location:** `testing_validation/data/` and `testing_validation/r_data/`

**Regression tasks:**
- Bone Collagen: 36 train / 13 test
- **Enamel d13C: 105 train / 35 test** (NEW! - more robust)

### 2. R Testing Scripts ✓
- **Regression:** PLS, Ridge, Lasso, ElasticNet, RF, XGBoost (~55 configs)
- **Classification:** PLS-DA, RF, XGBoost (binary + 4-class)
- **Location:** `testing_validation/r_scripts/`

### 3. Documentation ✓
- **README.md:** Full infrastructure guide
- **TESTING_PLAN_STATUS.md:** Comprehensive status and plan
- **This file:** Quick start instructions

---

## Quick Start: Run First Tests (30 minutes)

### Step 1: Install R Packages (5-10 min)

```bash
cd testing_validation/r_scripts
Rscript install_packages.R
```

**Packages installed:**
- pls, glmnet, randomForest, xgboost
- prospectr (preprocessing)
- jsonlite, caret, pROC

### Step 2: Run R Regression Tests (10-20 min)

```bash
Rscript regression_models.R
```

**Output:** `testing_validation/results/r_regression/*.json`

**What it tests:**
- PLS: 12 configurations
- Ridge: 5 configurations
- Lasso: 4 configurations
- ElasticNet: 20 configurations
- Random Forest: 6 configurations
- XGBoost: 8 configurations

**Expected results:**
- Best R²: 0.70-0.85
- Best RMSE: 2.5-4.5% collagen

### Step 3: Run R Classification Tests (10-15 min)

```bash
Rscript classification_models.R
```

**Output:** `testing_validation/results/r_classification/{binary,4class}/*.json`

**What it tests:**
- Binary: High vs. Low collagen
- 4-class: Categories A, F, G, H
- Models: PLS-DA, Random Forest, XGBoost

**Expected results:**
- Binary accuracy: 75-85%
- 4-class accuracy: 60-75%

---

## Next Steps: DASP Comparison

### Create DASP Testing Scripts (TODO)

**Need to create:**

1. **`dasp_regression.py`** - Run DASP with same parameters as R
2. **`dasp_classification.py`** - Classification testing
3. **`compare_results.py`** - Compare DASP vs. R predictions

**Template structure:**

```python
# dasp_regression.py
import pandas as pd
from pathlib import Path
from src.spectral_predict.models import run_search

# Load train/test data
X_train = pd.read_csv("r_data/regression/X_train.csv").values
y_train = pd.read_csv("r_data/regression/y_train.csv")['%Collagen'].values
X_test = pd.read_csv("r_data/regression/X_test.csv").values
y_test = pd.read_csv("r_data/regression/y_test.csv")['%Collagen'].values

# Run DASP models with specific hyperparameters
# Example: PLS with n_components=10
from sklearn.cross_decomposition import PLSRegression
model = PLSRegression(n_components=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Export results
results = {
    'model': 'PLS_10',
    'rmse': rmse,
    'r2': r2,
    'predictions': y_pred.tolist()
}
# Save to JSON...
```

### Compare Results

**Script to create: `compare_results.py`**

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load R results
r_results = json.load(open('results/r_regression/pls_results.json'))

# Load DASP results
dasp_results = json.load(open('results/dasp_regression/pls_results.json'))

# Compare metrics
for ncomp in [2, 4, 6, 8, 10]:
    r_r2 = r_results[str(ncomp)]['r2']
    dasp_r2 = dasp_results[f'PLS_{ncomp}']['r2']

    diff = abs(r_r2 - dasp_r2)
    print(f"PLS n_comp={ncomp}: R R²={r_r2:.4f}, DASP R²={dasp_r2:.4f}, Diff={diff:.4f}")

# Scatter plot of predictions
plt.scatter(r_predictions, dasp_predictions)
plt.plot([min, max], [min, max], 'r--')  # y=x line
plt.xlabel('R Predictions')
plt.ylabel('DASP Predictions')
plt.title('DASP vs. R: PLS Predictions')
plt.savefig('results/pls_comparison.png')
```

---

## Critical Information

### Parameter Name Confusion ⚠️

**DASP vs. R glmnet:**
- DASP `alpha` (strength) = R `lambda`
- DASP `l1_ratio` (L1/L2) = R `alpha`

**Example:**
```python
# DASP
Ridge(alpha=1.0)

# R equivalent
glmnet(alpha=0, lambda=1.0)
```

### Random Seeds

- All splits use seed=42
- Set R: `set.seed(42)`
- Set Python: `np.random.seed(42)`, `random.seed(42)`

### Expected Performance

**Regression (Bone Collagen):**
- R²: 0.70-0.85
- RMSE: 2.5-4.5% collagen
- Best models: PLS (n_comp=10-20), RF, XGBoost

**Classification:**
- Binary accuracy: 75-85%
- 4-class accuracy: 60-75%

---

## File Locations

```
testing_validation/
├── README.md                     ← Full documentation
├── TESTING_PLAN_STATUS.md        ← Comprehensive status
├── QUICK_START.md               ← This file
│
├── data/                        ← Train/test CSVs
├── r_data/                      ← Spectral matrices for R
├── r_scripts/                   ← R testing scripts
│   ├── install_packages.R
│   ├── regression_models.R
│   └── classification_models.R
│
└── results/                     ← Output directory
    ├── r_regression/            ← R regression results
    └── r_classification/        ← R classification results
```

---

## Success Criteria

### Regression
- ✅ R² within ±2% for PLS/Ridge/Lasso
- ✅ R² within ±5% for RF/XGBoost
- ✅ Prediction correlation >0.99

### Classification
- ✅ Accuracy within ±3%
- ✅ AUC within ±0.05

### Reproducibility
- ✅ Exact results with fixed seed
- ✅ Consistent performance across runs

---

## Timeline

- **Phases 1-2:** ✅ COMPLETE (dataset prep + R infrastructure)
- **Phase 3:** Next - Run tests, create DASP scripts, compare
- **Phases 4-6:** Week 2 - Classification, variable selection, hyperparameters
- **Phases 7-9:** Week 3 - Edge cases, benchmarking, reproducibility
- **Phase 10:** Week 4 - Final reports

**MVP:** 1-2 weeks (core models only)
**Full validation:** 3-4 weeks (all tests + reports)

---

## Questions?

See detailed documentation:
- `README.md` - Infrastructure guide
- `TESTING_PLAN_STATUS.md` - Full plan and status
- Main DASP docs - Implementation details

---

**Ready to go!** Start with Step 1 above to run your first validation tests.
