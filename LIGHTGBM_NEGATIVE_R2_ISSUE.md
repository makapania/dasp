# LightGBM Negative R² Issue - DEBUG NEEDED

**Date:** 2025-01-10
**Status:** ⚠️ CRITICAL - LightGBM producing negative R² values
**Priority:** HIGH - Core model functionality broken

---

## Problem Summary

LightGBM is producing **negative R² values** during training, indicating performance worse than a horizontal line (mean predictor). This is happening even after:
1. Complete parameter capture fix (like XGBoost)
2. Adding proper regularization parameters for high-dimensional data
3. Ensuring random_state consistency

**Expected:** R² > 0.9 (like other models: XGBoost, RandomForest, ElasticNet)
**Actual:** R² < 0 (negative values - catastrophic failure)

---

## What We've Tried (All Failed)

### Attempt 1: Remove max_depth Parameter Conflict
**Date:** Earlier session
**Theory:** `max_depth=6` in `get_model()` conflicted with `num_leaves` from grid search
**Fix Applied:** Changed from `max_depth=6` to `num_leaves=31`
**Result:** ❌ FAILED - Still negative R²

### Attempt 2: Complete Parameter Capture (Like XGBoost Fix)
**Date:** 2025-01-10
**Theory:** Same issue as XGBoost - not all parameters being saved/restored
**Fix Applied:**
- Modified `search.py` lines 831-862 to capture ALL parameters via `get_params()`
- Replaces incomplete params dict with complete parameter set
- Includes random_state, n_jobs, verbosity, and all other params
**Result:** ❌ FAILED - Still negative R²

### Attempt 3: Add Proper Regularization for High-Dimensional Data
**Date:** 2025-01-10
**Theory:** LightGBM needs regularization like XGBoost for spectral data (2000+ features)
**Fix Applied:**
- Added `subsample=0.8` (row sampling)
- Added `colsample_bytree=0.8` (feature sampling - critical for high dimensions)
- Added `reg_alpha=0.1` (L1 regularization)
- Added `reg_lambda=1.0` (L2 regularization)
- Added `min_child_samples=20` (prevent overfitting)
- Set `max_depth=-1` (no hard limit, controlled by num_leaves)
**Files Modified:**
- `src/spectral_predict/models.py` lines 121-134 (default regressor)
- `src/spectral_predict/models.py` lines 191-204 (default classifier)
- `src/spectral_predict/models.py` lines 590-617 (grid search regressor)
- `src/spectral_predict/models.py` lines 763-789 (grid search classifier)
**Result:** ❌ FAILED - Still negative R²

---

## Current Configuration

### Default LightGBM Model (lines 121-134)
```python
LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
```

### Grid Search Configuration (model_config.py:141-146)
```python
'LightGBM': {
    'standard': {
        'n_estimators': [100, 200],  # 2 values
        'learning_rate': [0.1],      # 1 value
        'num_leaves': [31, 50],      # 2 values
        'note': 'Grid size: 2×1×2 = 4 configs'
    }
}
```

---

## Theories for Next Agent to Test

### Theory 1: LightGBM Doesn't Like Spectral Data Format
**Hypothesis:** LightGBM may be sensitive to:
- Data type (need float32 instead of float64?)
- Feature scaling (despite preprocessing?)
- Very wide data (2000+ columns, few rows)
- Highly correlated features (spectral wavelengths are sequential)

**Test:**
1. Print data shape and dtypes before LightGBM training
2. Check if X contains NaN, inf, or extreme values
3. Try converting to float32: `X = X.astype(np.float32)`
4. Compare with XGBoost - what's different in how they handle the same data?

### Theory 2: Categorical Feature Handling Issue
**Hypothesis:** LightGBM has special categorical feature handling that might be interfering

**Test:**
1. Check if any features are being interpreted as categorical
2. Explicitly set `categorical_feature=[]` or `categorical_feature='auto'`
3. Check `feature_name` parameter

### Theory 3: Preprocessing Pipeline Incompatibility
**Hypothesis:** Something in the sklearn Pipeline is breaking LightGBM specifically

**Test:**
1. Train LightGBM directly without Pipeline wrapper
2. Test with raw data (no preprocessing)
3. Test with each preprocessing step individually (SNV, derivative, etc.)
4. Compare: Does RandomForest work but LightGBM fail with same preprocessing?

### Theory 4: LightGBM Version or Installation Issue
**Hypothesis:** The LightGBM installation might be broken or incompatible

**Test:**
1. Check LightGBM version: `import lightgbm; print(lightgbm.__version__)`
2. Run simple test outside the codebase:
   ```python
   from lightgbm import LGBMRegressor
   from sklearn.datasets import make_regression
   X, y = make_regression(n_samples=100, n_features=2000, random_state=42)
   model = LGBMRegressor(random_state=42)
   model.fit(X, y)
   print(f"R² = {model.score(X, y)}")  # Should be > 0.9
   ```
3. If simple test works, issue is in the integration

### Theory 5: Boosting Rounds / Early Stopping
**Hypothesis:** LightGBM might be stopping too early or not training properly

**Test:**
1. Check if `n_estimators` is actually being used (vs early stopping)
2. Add `early_stopping_rounds=None` to disable early stopping
3. Add verbose output during training to see if trees are being built
4. Check training history: `model.evals_result_` after fit

### Theory 6: Learning Rate Too High
**Hypothesis:** learning_rate=0.1 might be too aggressive for this data

**Test:**
1. Try learning_rate=0.01 or 0.001
2. Increase n_estimators proportionally (1000 instead of 100)
3. Check if loss is exploding during training

### Theory 7: num_leaves vs Tree Depth Conflict
**Hypothesis:** Even with max_depth=-1, there might be internal conflicts

**Test:**
1. Try much smaller num_leaves (e.g., 7, 15 instead of 31, 50)
2. Try explicitly setting max_depth=6 WITH num_leaves
3. Check LightGBM docs on num_leaves recommendations for data size

### Theory 8: Sample Size Too Small
**Hypothesis:** With validation split + CV, effective training samples might be too small

**Test:**
1. Print actual sample sizes during training
2. Check min_data_in_leaf parameter
3. Try training on full dataset without validation split
4. Check if issue only happens with small datasets

---

## Debugging Steps (URGENT)

### Step 1: Enable Verbose Output
```python
# In models.py, change verbosity from -1 to 1
LGBMRegressor(
    ...
    verbosity=1  # Show training output
)
```

### Step 2: Add Debug Logging
In `search.py` around line 831 (where XGBoost diagnostic is), add:
```python
if model_name == "LightGBM":
    print(f"\n{'='*80}")
    print(f"DEBUG - LightGBM Training Data")
    print(f"{'='*80}")
    print(f"X shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y shape: {y.shape}")
    print(f"y dtype: {y.dtype}")
    print(f"X contains NaN: {np.isnan(X).any()}")
    print(f"X contains inf: {np.isinf(X).any()}")
    print(f"X min: {np.min(X)}, max: {np.max(X)}")
    print(f"y min: {np.min(y)}, max: {np.max(y)}")
    print(f"{'='*80}\n")
```

### Step 3: Test Simple Case
Create standalone test in same environment:
```python
# test_lightgbm.py
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

# Test 1: Simple dataset
X, y = make_regression(n_samples=100, n_features=50, random_state=42)
model = LGBMRegressor(random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Test 1 (simple): Mean R² = {np.mean(scores):.3f}")

# Test 2: High-dimensional like spectral
X, y = make_regression(n_samples=100, n_features=2000, random_state=42)
model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Test 2 (high-dim): Mean R² = {np.mean(scores):.3f}")

# Test 3: With preprocessing pipeline (like actual code)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])
scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"Test 3 (pipeline): Mean R² = {np.mean(scores):.3f}")
```

### Step 4: Compare with Working Model
Run XGBoost and LightGBM on EXACT same data/split:
```python
# In search.py, add comparison logging
if model_name in ["XGBoost", "LightGBM"]:
    # After CV, before computing mean:
    print(f"\n{model_name} CV fold R² scores:")
    for i, m in enumerate(cv_metrics):
        print(f"  Fold {i+1}: R² = {m['R2']:.4f}, RMSE = {m['RMSE']:.4f}")
    print(f"  Mean: R² = {mean_r2:.4f}, RMSE = {mean_rmse:.4f}\n")
```

---

## Key Files and Functions

### Grid Search & Training
1. **search.py:694-900** - `_run_single_config()` function
   - Line 748: Pipeline creation
   - Line 751-756: CV execution
   - Line 824: Model refitting for feature importances
   - Line 831-862: Diagnostic output (XGBoost/LightGBM)

2. **models.py:211-617** - `get_model_grids()` function
   - Line 584-617: LightGBM grid generation (regression)
   - Line 758-790: LightGBM grid generation (classification)

3. **models.py:56-134** - `get_model()` function
   - Line 121-134: Default LightGBM regressor
   - Line 191-204: Default LightGBM classifier

### Configuration
4. **model_config.py:141-146** - LightGBM hyperparameter grid
5. **model_config.py:24** - Standard tier model list
6. **model_config.py:30** - Comprehensive tier model list

---

## Data Flow for LightGBM

```
User Selects LightGBM → get_model_grids() creates grid configs
    ↓
run_search() iterates through configs
    ↓
_run_single_config() called for each config
    ↓
Build preprocessing pipeline (SNV, deriv, etc.)
    ↓
Add LGBMRegressor to pipeline
    ↓
Run CV with Parallel(n_jobs=-1) - NEGATIVE R² HERE!
    ↓
Compute mean_r2 (still negative)
    ↓
Refit on full data for feature importances
    ↓
Capture ALL params (via get_params())
    ↓
Save to results CSV
```

**Issue occurs:** During CV fold execution (line 751-756 in search.py)

---

## What Definitely Works

- ✅ XGBoost (fixed with complete parameter capture)
- ✅ RandomForest (works fine)
- ✅ ElasticNet, PLS, Ridge, Lasso (all working)
- ✅ Parameter capture system (works for XGBoost)
- ✅ Preprocessing pipeline (works for other models)
- ✅ CV splitting (works for other models)

## What's Broken

- ❌ LightGBM regression (negative R²)
- ❌ Likely LightGBM classification too (not confirmed)
- ❌ Ensemble model save functionality (missing metadata error)
- ❌ All buttons in ensemble model results section not working

---

## Ensemble Model Issues (Separate Problem)

### Problem Description
The ensemble model functionality in Tab 5 is broken:
1. **Save Ensemble Model** button produces "missing metadata" error
2. **All other buttons** in the ensemble results section are non-functional
3. This appears to be a separate issue from the LightGBM problem

### What Was Implemented
According to XGBOOST_REPRODUCIBILITY_ISSUE.md:
- ✅ Ensemble feature was marked as "COMPLETE - Fully functional, production-ready" (~900 lines of code)
- Implementation included:
  - Model reconstruction from results
  - Ensemble training workflow
  - Results display in Tab 5
  - Visualization integration
  - Save/load functionality
  - Tab 7 prediction support
  - Documentation updated

### Current Status
Despite being marked complete, the ensemble functionality is broken:
- Save button throws "missing metadata" error
- Other buttons in ensemble results section don't work
- Likely issue with metadata structure or serialization

### Files to Check
1. **spectral_predict_gui_optimized.py** - Tab 5 ensemble implementation
   - Search for "ensemble" to find all related code
   - Check save button callback
   - Check metadata structure being saved

2. **src/spectral_predict/model_io.py** - Save/load functions
   - `save_ensemble()` function (~230 lines added)
   - `load_ensemble()` function
   - Check what metadata fields are required vs provided

3. **docs/MACHINE_LEARNING_MODELS.md** - Ensemble documentation
   - "Using Ensembles in the GUI" section (~150 lines)
   - Check expected workflow vs actual implementation

### Debug Steps for Ensemble Issue
1. Find the save button callback in GUI code
2. Add try/except with detailed error printing
3. Check what metadata fields are being requested
4. Check what metadata is actually available
5. Compare with individual model save functionality (which works)

### Theory
The ensemble save might be trying to serialize models that include the problematic LightGBM models (with negative R²), which could be causing metadata issues. OR, the metadata structure changed during parameter capture fix and ensemble code wasn't updated.

---

## Pattern Recognition

**Interesting observation:** XGBoost and LightGBM are VERY similar gradient boosting frameworks, but:
- XGBoost works perfectly (after parameter fix)
- LightGBM completely fails (negative R²)

This suggests the issue is **specific to LightGBM's implementation details**, not the general gradient boosting approach or data format.

Possible LightGBM-specific differences:
1. LightGBM uses histogram-based binning (different from XGBoost)
2. LightGBM has different categorical feature handling
3. LightGBM leaf-wise growth vs XGBoost level-wise
4. LightGBM more sensitive to data types (float32 vs float64)
5. LightGBM has different handling of missing values

---

## System Info

- **OS:** Windows (win32)
- **Python:** Should be in .venv
- **LightGBM:** Installed via pip in .venv (check version!)
- **XGBoost:** Working correctly (same environment)
- **Working Directory:** C:\Users\sponheim\git\dasp
- **Branch:** claude/combined-format-011CUzTnzrJQP498mXKLe4vt

---

## Recommended Approach for Next Agent

1. **Start with simple test** (Step 3 above) to isolate issue
2. **Add verbose output** (Step 1) to see what LightGBM is doing
3. **Add data debugging** (Step 2) to verify data format
4. **Compare with XGBoost** (Step 4) on identical data
5. **Test theories systematically** (work through Theory 1-8)

**Priority Order:**
1. Theory 4 (Installation/Version) - Quick to test
2. Theory 1 (Data Format) - Most likely given symptoms
3. Theory 3 (Pipeline) - Test with/without preprocessing
4. Theory 5 (Training Process) - Check if trees are actually being built
5. Others as needed

---

## Success Criteria

LightGBM should achieve:
- **R² > 0.9** (comparable to XGBoost, RandomForest)
- **Stable across CV folds** (not wildly varying)
- **Reproducible** (same R² in Results Tab and Model Development Tab)

---

## Contact / Handoff

**What Works:**
- ✅ Complete parameter capture system (XGBoost proved it works)
- ✅ All other models (PLS, Ridge, Lasso, ElasticNet, RandomForest, XGBoost)
- ✅ Diagnostic logging infrastructure

**What Doesn't Work:**
- ❌ LightGBM producing negative R² values (catastrophic failure)
- ❌ Ensemble model save functionality ("missing metadata" error)
- ❌ Ensemble results section buttons non-functional

**Critical Questions:**
1. Why does LightGBM fail (negative R²) when XGBoost succeeds (R² > 0.9) on the exact same data in the same environment with similar parameters?
2. Why is ensemble save functionality broken when it was marked as complete and production-ready?
3. Are these two issues related? (e.g., ensemble trying to save broken LightGBM models?)

---

*Document created: 2025-01-10*
*For: next debugging agent*
*Related: XGBOOST_REPRODUCIBILITY_ISSUE.md (now resolved)*
