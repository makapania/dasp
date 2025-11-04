# DASP Preprocessing Technical Documentation

**Date:** 2025-11-03
**Purpose:** Clarify preprocessing application order and feature selection workflow

---

## Overview

This document clarifies the exact order of preprocessing and feature selection in the DASP spectral analysis pipeline. Understanding this order is critical for reproducing results and debugging models.

---

## General Principle

**PREPROCESSING IS ALWAYS APPLIED TO THE FULL SELECTED WAVELENGTH RANGE *BEFORE* ANY SUBSET SELECTION**

This ensures:
1. Derivatives and smoothing have proper spectral context
2. Feature importance reflects actual model inputs (preprocessed data)
3. Spectral features are captured correctly across the full range

---

## Detailed Workflow for Each Model Type

### 1. Full Spectrum Models

**User Configuration:**
- Wavelength range: e.g., 1500-2300 nm (800 wavelengths)
- Preprocessing: e.g., SNV + 1st derivative (sg1)
- Model: e.g., PLS with 10 components

**Processing Steps:**
```
Step 1: Load full spectrum
  → Data shape: [n_samples, 800] wavelengths [1500.0, 1501.0, ..., 2300.0]

Step 2: Apply SNV normalization
  → Each spectrum normalized: (X - mean) / std
  → Data shape: [n_samples, 800]

Step 3: Apply Savitzky-Golay 1st derivative (window=17)
  → Scipy handles edge effects automatically
  → Data shape: [n_samples, 800] (wavelength labels unchanged)
  → Note: Edge features less reliable within ±8 points

Step 4: Train model on ALL preprocessed features
  → Model trained on 800 preprocessed features
  → Cross-validation performed
  → Performance metrics calculated

Result: Model using all 800 wavelengths (preprocessed)
```

**Code Location:** `src/spectral_predict/search.py`, lines 568-573

---

### 2. Subset Models (Top-N Variables)

**User Configuration:**
- Wavelength range: 1500-2300 nm (800 wavelengths)
- Preprocessing: 1st derivative (sg1)
- Model: PLS
- Subset: top50 variables

**Processing Steps:**
```
Step 1: Load full spectrum
  → Data shape: [n_samples, 800] wavelengths [1500.0, ..., 2300.0]

Step 2: Apply preprocessing to FULL spectrum
  → Apply 1st derivative to all 800 wavelengths
  → Data shape: [n_samples, 800] (preprocessed)

Step 3: Train temporary model on all preprocessed features
  → Temporary PLS model trained on all 800 preprocessed features
  → Used only for feature importance calculation

Step 4: Compute feature importance
  → Extract VIP scores (for PLS) or coefficients (for Ridge/Lasso)
  → Importance computed for all 800 preprocessed features

Step 5: Select top N most important features
  → Rank features by importance
  → Select top 50 → 50 wavelengths identified
  → **NEW FIX:** Save all 50 in 'all_vars' column (was only saving top 30)

Step 6: Retrain model using ONLY selected features
  → Extract columns for the 50 selected wavelengths
  → Retrain PLS model on these 50 preprocessed features
  → Cross-validation on 50-feature subset
  → Performance metrics calculated

Result: Model using 50 wavelengths (already preprocessed)
```

**Code Location:** `src/spectral_predict/search.py`, lines 376-393

**Critical Insight:** The 50 selected wavelengths are selected from the PREPROCESSED data, not raw data. The model operates on preprocessed features.

---

### 3. Region-Based Subsets

**User Configuration:**
- Wavelength range: 1500-2300 nm (800 wavelengths)
- Preprocessing: 2nd derivative (sg2)
- Subset: Identified spectral regions

**Processing Steps:**
```
Step 1: Load full spectrum
  → Data shape: [n_samples, 800] wavelengths

Step 2: Apply preprocessing to FULL spectrum
  → Apply 2nd derivative to all 800 wavelengths
  → Data shape: [n_samples, 800] (preprocessed)

Step 3: Use PLS to identify informative spectral regions
  → Train PLS on all 800 preprocessed features
  → Compute feature importance (VIP scores)
  → Identify continuous regions with high importance
  → Example result: Regions [1800-1900 nm] and [2100-2200 nm]

Step 4: Select wavelengths within identified regions
  → Extract columns corresponding to identified regions
  → E.g., ~100 + 100 = 200 wavelengths in selected regions

Step 5: Retrain model on regional subset
  → Train model on 200 preprocessed features from selected regions
  → Cross-validation performed
  → Performance metrics calculated

Result: Model using ~200 wavelengths from identified regions (preprocessed)
```

**Code Location:** `src/spectral_predict/search.py`, lines 226-237

---

## Key Technical Details

### Derivative Edge Effects

**Savitzky-Golay Filter Behavior:**
- Window size: typically 17 points
- Edge handling: Scipy automatically pads/extrapolates
- Result: Same number of features as input
- BUT: Features near edges (±8 points) are less reliable

**Example:**
```
Input:  800 wavelengths [1500.0, 1501.0, ..., 2300.0]
Output: 800 wavelengths [1500.0, 1501.0, ..., 2300.0]

Reliability:
  - Wavelengths 1500-1508 (first 9):  Lower reliability
  - Wavelengths 1509-2291 (middle):   High reliability
  - Wavelengths 2292-2300 (last 9):   Lower reliability
```

**Wavelength Labels:** Remain unchanged after derivatives. The label represents the center of the derivative calculation window.

### Why Preprocess Before Feature Selection?

**Q:** Why not select wavelengths first, then apply preprocessing?

**A:** Derivatives require neighboring points!

**Wrong Approach (Would Break Derivatives):**
```
1. Select 50 specific wavelengths from raw spectrum
   → Gaps between wavelengths: [1500, 1520, 1540, 1560, ...]
2. Try to apply derivative
   → ERROR: Derivatives need continuous, neighboring points
   → Cannot compute derivative across 20nm gaps!
```

**Correct Approach (Current Implementation):**
```
1. Apply derivative to full continuous spectrum
   → Derivatives computed correctly with proper spectral context
2. Select 50 features from preprocessed data
   → Features are derivative values at selected wavelengths
   → Makes physical/chemical sense
```

---

## Preprocessing Methods

### Available Methods

| Method | Steps | Output Features |
|--------|-------|-----------------|
| `raw` | None | Same as input |
| `snv` | Standard Normal Variate | Same as input |
| `deriv` (sg1) | 1st derivative (Savgol, window=17) | Same as input |
| `deriv2` (sg2) | 2nd derivative (Savgol, window=17) | Same as input |
| `snv_deriv` | SNV → then 1st derivative | Same as input |
| `deriv_snv` | 1st derivative → then SNV | Same as input |

**Note:** Order matters! `snv_deriv` ≠ `deriv_snv`

### Implementation Details

**File:** `src/spectral_predict/preprocess.py`

**Classes:**
- `SNV` (lines 8-40): Per-spectrum normalization
- `SavgolDerivative` (lines 43-106): Savitzky-Golay derivative filter

**Pipeline Builder:** `build_preprocessing_pipeline()` (lines 109-151)

---

## Code References

### Full Spectrum Preprocessing
**File:** `src/spectral_predict/search.py`
**Lines:** 568-573
```python
# Build preprocessing pipeline
pipe = build_preprocessing_pipeline(
    preprocess_cfg,
    n_features,
    model_class,
    params,
    task_type
)
```

### Subset Preprocessing
**File:** `src/spectral_predict/search.py`
**Lines:** 376-393
```python
# For derivatives: data already preprocessed, skip preprocessing
if preprocess_cfg["deriv"] > 0:
    result = _run_single_config(
        X_transformed,  # Already preprocessed
        y,
        wavelengths,
        model_name,
        params,
        preprocess_cfg,
        cv_splitter,
        task_type,
        is_binary_classification,
        subset_indices=subset_indices,
        subset_tag=subset_tag,
        top_n_vars=top_n_vars,
        skip_preprocessing=True  # <-- KEY FLAG
    )
```

### Feature Importance Extraction
**File:** `src/spectral_predict/search.py`
**Lines:** 650-686
```python
# Compute importances on PREPROCESSED data
importances = get_feature_importances(
    fitted_model, model_name, X_transformed, y
)

# For subset models: save ALL wavelengths (NEW FIX)
if subset_tag != "full" and subset_indices is not None:
    all_wavelengths = wavelengths[subset_indices]
    all_vars_str = ','.join([f"{w:.1f}" for w in all_wavelengths])
    result['all_vars'] = all_vars_str
```

---

## Summary for Users

### Rules of Thumb

1. **Preprocessing always happens first** - Applied to full selected range
2. **Feature selection happens second** - On preprocessed data
3. **Model trains on preprocessed features** - Never on raw data (if preprocessing selected)
4. **Edge effects are automatic** - Scipy handles them, but edges less reliable
5. **Wavelength labels don't change** - Even after derivatives

### Implications for Model Interpretation

**When you see:**
- "Model uses wavelengths: [1520, 1540, 1560, ...]"

**This means:**
- Model uses PREPROCESSED values at these wavelengths
- If derivative was applied: Model uses derivative values at these points
- These were selected because their derivative values are most informative

**Physical Interpretation:**
- For raw/SNV: Wavelengths are absorption/reflectance values
- For 1st derivative: Wavelengths are rates of spectral change
- For 2nd derivative: Wavelengths are curvature of spectral features

---

## Recent Bug Fix (2025-11-03)

### Issue
Previously, subset models (e.g., top50, top100) only saved the top 30 most important wavelengths in the `top_vars` column. When loading these models into Custom Model Development tab, only 30 wavelengths were available instead of the full 50 or 100.

### Fix
- Added new `all_vars` column to results dataframe
- For subset models: `all_vars` contains ALL wavelengths used (e.g., all 50 for top50)
- `top_vars` still contains top 30 for display purposes
- GUI loading logic now prefers `all_vars` over `top_vars`

### Code Changes
1. `src/spectral_predict/search.py` (lines 655-669): Save all wavelengths in `all_vars`
2. `src/spectral_predict/scoring.py` (line 147): Add `all_vars` column to schema
3. `spectral_predict_gui_optimized.py` (lines 2353-2380): Prefer `all_vars` when loading

---

**Document Version:** 1.0
**Last Updated:** 2025-11-03
**Author:** DASP Development Team
