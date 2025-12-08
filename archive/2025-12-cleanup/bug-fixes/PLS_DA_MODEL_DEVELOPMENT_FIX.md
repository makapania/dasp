# PLS-DA Model Development Tab Fix

## Issue
PLS-DA works in the **Results tab** but fails in the **Model Development tab** with an error.

## Root Cause
The Model Development tab (`_run_refined_model_thread` function) was not constructing PLS-DA pipelines correctly.

**What PLS-DA needs:**
- PLS-DA = PLS (dimensionality reduction) + LogisticRegression (classification)
- Pipeline structure: `Preprocessing → PLS → LogisticRegression`

**What was happening:**
- Results tab: Correctly built PLS-DA pipelines (via `search.py` lines 807-810)
- Model Development tab: Only added PLS, forgot LogisticRegression → Error!

## The Fix

Added logic to construct PLS-DA pipelines correctly in BOTH preprocessing paths:

### Path A: Derivative + Subset (lines 10553-10564)
```python
# For PLS-DA, we need PLS + LogisticRegression
if model_name == "PLS-DA" and task_type == "classification":
    from sklearn.linear_model import LogisticRegression
    pipe_steps = [
        ('pls', model),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ]
else:
    pipe_steps = [('model', model)]
```

### Path B: Raw/SNV or Full-Spectrum (lines 10579-10588)
```python
# For PLS-DA, we need PLS + LogisticRegression
if model_name == "PLS-DA" and task_type == "classification":
    from sklearn.linear_model import LogisticRegression
    pipe_steps.append(('pls', model))
    pipe_steps.append(('lr', LogisticRegression(max_iter=1000, random_state=42)))
else:
    pipe_steps.append(('model', model))
```

### Model Saving (lines 10778-10786)
Updated to save the entire PLS-DA pipeline:
```python
if model_name == "PLS-DA" and task_type == "classification":
    # Save entire PLS-DA pipeline (both PLS and LogisticRegression)
    final_model = final_pipe
elif 'model' in final_pipe.named_steps:
    final_model = final_pipe.named_steps['model']
else:
    final_model = final_pipe
```

### Preprocessor Extraction (lines 10794-10802)
Updated to handle PLS-DA's 2-step model:
```python
elif model_name == "PLS-DA" and task_type == "classification":
    # For PLS-DA: preprocessing steps are before PLS, PLS+LR are the model
    if len(pipe_steps) > 2:  # Has preprocessing + PLS + LR
        final_preprocessor = Pipeline(pipe_steps[:-2])  # All steps except PLS and LR
        final_preprocessor.fit(X_raw)
        print("DEBUG: Fitting preprocessor on subset data (PLS-DA)")
    else:
        final_preprocessor = None
        print("DEBUG: No preprocessor for PLS-DA (raw data)")
```

### Prediction Probabilities (lines 10619-10625)
Added fallback for PLS-DA's LogisticRegression step:
```python
if hasattr(pipe_fold, 'predict_proba'):
    y_proba = pipe_fold.predict_proba(X_test)
    all_y_proba.append(y_proba)
elif 'model' in pipe_fold.named_steps and hasattr(pipe_fold.named_steps['model'], 'predict_proba'):
    y_proba = pipe_fold.named_steps['model'].predict_proba(X_test)
    all_y_proba.append(y_proba)
elif 'lr' in pipe_fold.named_steps and hasattr(pipe_fold.named_steps['lr'], 'predict_proba'):
    # For PLS-DA, LogisticRegression is named 'lr'
    y_proba = pipe_fold.named_steps['lr'].predict_proba(X_test)
    all_y_proba.append(y_proba)
```

## Files Modified

**spectral_predict_gui_optimized.py**
- Lines 10553-10564: Added PLS-DA pipeline construction (Path A)
- Lines 10579-10588: Added PLS-DA pipeline construction (Path B)
- Lines 10778-10786: Updated model extraction for PLS-DA
- Lines 10794-10802: Updated preprocessor extraction for PLS-DA
- Lines 10619-10625: Updated predict_proba extraction for PLS-DA

## Testing

1. **Load classification data**
2. **Run analysis** with PLS-DA in Results tab → Should work ✓
3. **Double-click PLS-DA result** to load in Model Development tab
4. **Click "Run Model"** → Should now work without errors ✓
5. **Verify metrics** match between Results and Model Development tabs

## Related Fixes

This fix complements the earlier PLS-DA fixes in:
- `src/spectral_predict/models.py`: Added `PLSTransformer` class
- `src/spectral_predict/search.py`: Already had correct PLS-DA pipeline construction

Now PLS-DA works consistently across:
- ✅ Results tab (via `search.py`)
- ✅ Model Development tab (via GUI refinement)
- ✅ Model saving and loading

---

**Date:** 2025-11-13
**Status:** Complete
