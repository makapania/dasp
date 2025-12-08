# LightGBM Performance Fix - SUPERSEDED

**STATUS: This document is SUPERSEDED by LIGHTGBM_ROOT_CAUSE_AND_FIX.md**

The fix described here (adding regularization) was **incomplete**. The actual root cause was `min_child_samples=20` being too restrictive for small datasets. See LIGHTGBM_ROOT_CAUSE_AND_FIX.md for the complete analysis and fix.

---

# Original Document (Incomplete Fix)

## Problem

LightGBM was achieving only R² = 0.11 while other models (XGBoost, RandomForest, etc.) were achieving R² > 0.9 on the same spectral data.

## Root Cause (INCOMPLETE - SEE NEW DOCUMENT)

LightGBM's default parameters had **NO regularization**, causing severe overfitting on high-dimensional spectral data:

```python
# OLD (BROKEN) - No regularization
LGBMRegressor(
    subsample=1.0,          # NO row sampling
    colsample_bytree=1.0,   # Uses ALL features
    reg_alpha=0.0,          # NO L1 regularization
    reg_lambda=0.0,         # NO L2 regularization
    ...
)
```

### Why This Matters for Spectral Data

Spectral datasets have unique characteristics:
- **Small sample sizes** (~100 samples)
- **High dimensionality** (~1000-2000 wavelengths)
- **Highly correlated features** (adjacent wavelengths are similar)

Without regularization, LightGBM memorizes the training data perfectly but fails to generalize to new data.

## Solution

Updated LightGBM to use the same regularization approach as XGBoost:

```python
# NEW (FIXED) - Proper regularization
LGBMRegressor(
    subsample=0.8,          # Row sampling to prevent overfitting
    colsample_bytree=0.8,   # Feature sampling for high-dim data
    reg_alpha=0.1,          # L1 regularization for feature selection
    reg_lambda=1.0,         # L2 regularization to prevent overfitting
    ...
)
```

## Files Modified

1. **src/spectral_predict/models.py** (line 121-135)
   - Updated `get_model()` function for LightGBM regression defaults
   - Added regularization parameters matching XGBoost

2. **src/spectral_predict/model_config.py** (lines 218-254)
   - Updated 'standard' tier LightGBM hyperparameters
   - Updated 'quick' tier LightGBM hyperparameters
   - Left 'comprehensive' tier unchanged (already had regularization)

## Expected Impact

LightGBM should now achieve similar performance to XGBoost:
- **Before:** R² = 0.11 (severe overfitting)
- **After:** R² = 0.85-0.95 (expected, based on XGBoost performance)

The regularization will:
1. Prevent overfitting by sampling rows/features
2. Enable better feature selection via L1 regularization
3. Improve generalization via L2 regularization
4. Maintain fast training speed (LightGBM's advantage over XGBoost)

## Testing

To verify the fix works:
1. Run a model comparison on your spectral data
2. Check that LightGBM's **Development tab R² matches Results tab R² almost exactly** (difference < 0.001, like XGBoost after the R² consistency fix)
3. Verify R² scores are competitive with XGBoost (should be similar, 0.85-0.95 range)
4. Confirm LightGBM is no longer getting stuck at ~0.11 R²

**Note:** The R² consistency fix from commit ee3129d achieved PERFECT reproducibility (0.000000 difference) for XGBoost. With proper regularization, LightGBM should now achieve the same level of consistency.

## Technical Details

The specific regularization values were chosen to match XGBoost:
- `subsample=0.8`: Use 80% of samples per tree (20% dropout for regularization)
- `colsample_bytree=0.8`: Use 80% of features per tree (reduces correlation)
- `reg_alpha=0.1`: Light L1 regularization (feature selection for 1000+ features)
- `reg_lambda=1.0`: Standard L2 regularization (prevents weight explosion)

These values are empirically proven on spectral data and are used successfully by XGBoost.
