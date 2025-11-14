# Task Type-Based Model Filtering - Implementation Summary

## Overview
Implemented visual filtering of model checkboxes based on selected task type (Classification vs Regression). Incompatible models are now visually disabled instead of just silently unchecked.

## Changes Made

### 1. Model Registry Update
**File:** `src/spectral_predict/model_registry.py` (line 32)

Added NeuralBoosted to classification models:
```python
CLASSIFICATION_MODELS = [
    'PLS-DA',
    'PLS',
    'RandomForest',
    'MLP',
    'NeuralBoosted',  # Now supports classification
    'SVM',
    'XGBoost',
    'LightGBM',
    'CatBoost',
]
```

### 2. Checkbox Widget References
**File:** `spectral_predict_gui_optimized.py` (lines 2478-2533)

Changed all checkbox creations from inline to storing widget references:
- `self.pls_checkbox`
- `self.plsda_checkbox`
- `self.ridge_checkbox`
- `self.lasso_checkbox`
- `self.elasticnet_checkbox`
- `self.randomforest_checkbox`
- `self.mlp_checkbox`
- `self.svr_checkbox`
- `self.xgboost_checkbox`
- `self.lightgbm_checkbox`
- `self.catboost_checkbox` (already existed)
- `self.neuralboosted_checkbox`

### 3. Widget Dictionary
**File:** `spectral_predict_gui_optimized.py` (lines 2536-2550)

Created `self.model_checkbox_widgets` dictionary mapping model names to checkbox widgets for easy enable/disable control.

### 4. Enhanced Task Type Handler
**File:** `spectral_predict_gui_optimized.py` (lines 4675-4710)

Completely rewrote `_on_task_type_changed()` to:
- **Auto mode**: Enable all checkboxes (except unavailable CatBoost)
- **Classification/Regression mode**:
  - Get supported models for task type
  - Loop through all model checkboxes
  - Visually **disable** incompatible models using `checkbox.state(['disabled'])`
  - **Enable** compatible models using `checkbox.state(['!disabled'])`
  - Uncheck disabled checkboxes
  - Preserve CatBoost disabled state if not installed

### 5. Initialization Call
**File:** `spectral_predict_gui_optimized.py` (line 5079)

Changed data loading to call `_on_task_type_changed()` instead of just `_on_tier_changed()` to ensure proper checkbox filtering after task type detection.

## Model Compatibility

### Regression-Only Models (Disabled in Classification)
- Ridge
- Lasso
- ElasticNet
- SVR

### Classification-Only Models (Disabled in Regression)
- PLS-DA

### Shared Models (Work for Both)
- PLS
- RandomForest
- MLP
- XGBoost
- LightGBM
- CatBoost
- **NeuralBoosted** (newly added)

## Behavior

### When Task Type is "Auto" (Default)
- All model checkboxes are enabled
- No filtering applied (user can select any model)
- When data loads and task type is detected, filtering is applied automatically

### When Task Type is "Regression"
- Regression-compatible models: **Enabled** ✅
- Classification-only models (PLS-DA): **Disabled and grayed out** 🚫
- User cannot click disabled checkboxes

### When Task Type is "Classification"
- Classification-compatible models: **Enabled** ✅
- Regression-only models (Ridge, Lasso, ElasticNet, SVR): **Disabled and grayed out** 🚫
- User cannot click disabled checkboxes

### Special Cases
- **CatBoost**: If Visual Studio Build Tools not installed, stays disabled regardless of task type
- **Tier Selection**: Works seamlessly with task type filtering (intersection of tier models AND compatible models)
- **Custom Tier**: Task type changes still affect enable/disable state

## User Experience Improvements

**Before:**
- Incompatible models silently unchecked
- No visual feedback
- Confusing when checkboxes mysteriously uncheck

**After:**
- Incompatible models visually disabled (grayed out)
- Clear visual feedback about model compatibility
- Cannot accidentally select incompatible models
- Better user guidance

## Testing Checklist

✅ Switch from "auto" to "regression" - verify regression-only models enabled, PLS-DA disabled
✅ Switch from "auto" to "classification" - verify classification models enabled, Ridge/Lasso/ElasticNet/SVR disabled
✅ Switch between "regression" and "classification" - verify correct enable/disable states
✅ Load classification data in "auto" mode - verify PLS-DA enabled, regression-only disabled
✅ Load regression data in "auto" mode - verify all regression models enabled
✅ Tier changes with task type selected - verify only compatible models get selected
✅ CatBoost without VS tools - verify stays disabled regardless of task type

## Files Modified
1. `src/spectral_predict/model_registry.py` - 1 line added
2. `spectral_predict_gui_optimized.py` - ~60 lines modified

---

**Date:** 2025-11-13
**Status:** Complete
**Impact:** Major UX improvement - clear visual feedback for model compatibility
