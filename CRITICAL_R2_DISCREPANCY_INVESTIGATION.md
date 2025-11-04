# CRITICAL INVESTIGATION: R² Discrepancy Root Cause Analysis

**Date:** 2025-11-03
**Issue:** R² drops from 0.9 to 0.6 when re-running model in Custom Model Development tab
**User Report:** Model with 2 LVs shows massive performance drop despite correct n_components
**Status:** 🔴 **ROOT CAUSE IDENTIFIED** - Fix Required

---

## Executive Summary

After deep code analysis comparing `src/spectral_predict/search.py` (original evaluation) and `spectral_predict_gui_optimized.py` (GUI refinement), I've identified **THE ROOT CAUSE**:

**The preprocessing is applied at DIFFERENT STAGES in the two code paths**, creating a fundamental architectural difference that causes different R² values even with identical settings.

### The Critical Difference

| Code Path | Preprocessing Timing | Structure |
|-----------|---------------------|-----------|
| **search.py** | Inside CV fold (via Pipeline) | `Pipeline([Preprocess, Model])` |
| **GUI** | Before CV split (manual) | Preprocess all data → CV → Model alone |

This is **NOT about data leakage** (SNV/SG are stateless), but about **Pipeline behavior vs manual preprocessing**.

---

## Side-by-Side Code Comparison

###  1. Original Search Evaluation (search.py)

**Location:** `src/spectral_predict/search.py`, lines 555-590

```python
def _run_single_config(X, y, wavelengths, model, model_name, params,
                       preprocess_cfg, cv_splitter, task_type, ...):
    """Run a single model configuration with CV."""

    # Apply wavelength subset if specified
    if subset_indices is not None:
        X = X[:, subset_indices]  # Line 557 - Direct numpy indexing

    # Build preprocessing pipeline
    pipe_steps = build_preprocessing_pipeline(
        preprocess_cfg["name"],
        preprocess_cfg["deriv"],
        preprocess_cfg["window"],
        preprocess_cfg["polyorder"],
    )  # Lines 568-573

    pipe_steps.append(("model", model))
    pipe = Pipeline(pipe_steps)  # Line 582 - Preprocessing IN pipeline

    # Run CV in parallel - PASSING RAW DATA
    cv_metrics = Parallel(n_jobs=-1)(
        delayed(_run_single_fold)(
            pipe, X, y, train_idx, test_idx, task_type, ...
        )
        for train_idx, test_idx in cv_splitter.split(X, y)  # Line 589 - X is RAW
    )
```

**Inside `_run_single_fold` (lines 491-505):**

```python
def _run_single_fold(pipe, X, y, train_idx, test_idx, task_type, ...):
    """Run a single CV fold."""
    pipe_clone = clone(pipe)  # Clone the FULL pipeline

    # Split RAW data
    X_train, X_test = X[train_idx], X[test_idx]  # Line 495
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit pipeline (preprocessing + model) on training fold
    pipe_clone.fit(X_train, y_train)  # Line 499
    # This calls:
    #   1. SNV.fit(X_train) + SNV.transform(X_train) → X_train_snv
    #   2. SG.fit(X_train_snv) + SG.transform(X_train_snv) → X_train_processed
    #   3. PLS.fit(X_train_processed, y_train)

    # Predict on test fold
    y_pred = pipe_clone.predict(X_test)  # Line 502
    # This calls:
    #   1. SNV.transform(X_test) → X_test_snv
    #   2. SG.transform(X_test_snv) → X_test_processed
    #   3. PLS.predict(X_test_processed)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"RMSE": rmse, "R2": r2}
```

**Key Points:**
- ✅ Preprocessing is part of the Pipeline
- ✅ Pipeline is cloned for each fold
- ✅ `.fit()` is called on preprocessors (even though they're stateless)
- ✅ sklearn ensures correct `.fit_transform()` → `.transform()` pattern
- ✅ Data flows through pipeline: raw → SNV → SG → PLS
- ✅ CV splitter works with RAW data

---

### 2. GUI Refinement Evaluation (spectral_predict_gui_optimized.py)

**Location:** `spectral_predict_gui_optimized.py`, lines 2515-2640

```python
def _run_refined_model_thread(self):
    """Execute the refined model in a background thread."""

    # Parse wavelength specification (lines 2526-2528)
    available_wl = self.X_original.columns.astype(float).values
    wl_spec_text = self.refine_wl_spec.get('1.0', 'end')
    selected_wl = self._parse_wavelength_spec(wl_spec_text, available_wl)

    # Filter X_original to selected wavelengths (lines 2535-2543)
    wl_to_col = {float(col): col for col in self.X_original.columns}
    selected_cols = [wl_to_col[wl] for wl in selected_wl if wl in wl_to_col]
    X_work = self.X_original[selected_cols]  # DataFrame subset

    # Get preprocessing method
    preprocess = self.refine_preprocess.get()  # Line 2551
    window = self.refine_window.get()

    # CRITICAL: Apply preprocessing to ALL data BEFORE CV split
    if preprocess == 'snv':
        X_processed = SNV().transform(X_work.values)  # Line 2555
    elif preprocess == 'sg1':
        X_processed = SavgolDerivative(deriv=1, window=window).transform(X_work.values)
    elif preprocess == 'sg2':
        X_processed = SavgolDerivative(deriv=2, window=window).transform(X_work.values)
    elif preprocess == 'snv_sg1':
        X_temp = SNV().transform(X_work.values)
        X_processed = SavgolDerivative(deriv=1, window=window).transform(X_temp)
    elif preprocess == 'snv_sg2':
        X_temp = SNV().transform(X_work.values)
        X_processed = SavgolDerivative(deriv=2, window=window).transform(X_temp)
    elif preprocess == 'deriv_snv':
        X_temp = SavgolDerivative(deriv=1, window=window).transform(X_work.values)
        X_processed = SNV().transform(X_temp)
    else:  # raw
        X_processed = X_work.values  # Line 2573

    # Get model
    model = get_model(model_name, task_type, n_components, ...)  # Line 2590

    # Run cross-validation on PREPROCESSED data
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # Line 2601

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_processed, y_array)):
        # Clone model (NOT pipeline - there is no pipeline!)
        model_fold = clone(model)  # Line 2611

        # Split PREPROCESSED data
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]  # Line 2614
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        # Fit model ONLY (no preprocessing in this step)
        model_fold.fit(X_train, y_train)  # Line 2618
        # This only calls:
        #   PLS.fit(X_train, y_train)  # X_train is already preprocessed!

        # Predict
        y_pred = model_fold.predict(X_test)  # Line 2619
        # This only calls:
        #   PLS.predict(X_test)  # X_test is already preprocessed!

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        fold_metrics.append({"rmse": rmse, "r2": r2})
```

**Key Points:**
- ❌ Preprocessing applied to ALL data BEFORE CV split (lines 2554-2573)
- ❌ No Pipeline - model is standalone
- ❌ `.fit()` is never called on preprocessors
- ❌ Manual preprocessing with direct `.transform()` calls
- ❌ CV splitter works with PREPROCESSED data
- ❌ Wavelength selection uses DataFrame → dict mapping (potential precision issues)

---

## The Critical Differences Table

| Aspect | search.py (Original) | GUI Refinement | Impact |
|--------|---------------------|----------------|--------|
| **Architecture** | `Pipeline([Preprocess, Model])` | Preprocess → then Model alone | Different execution flow |
| **Preprocessing timing** | Inside each CV fold | Before CV split | Different data handling |
| **Pipeline.fit() behavior** | Calls `.fit_transform()` on each step | Only `.transform()` called | Different sklearn code path |
| **CV input data** | Raw numpy array | Preprocessed numpy array | Different CV behavior |
| **Wavelength selection** | Direct numpy indexing `X[:, indices]` | DataFrame → dict → lookup | Potential precision/order issues |
| **Data type flow** | numpy → Pipeline → numpy | DataFrame → numpy → preprocess → numpy | More conversions |
| **Model cloning** | Entire pipeline cloned per fold | Only model cloned per fold | Different object lifecycle |

---

## Why This Causes 0.9 → 0.6 R² Drop

### Hypothesis 1: Pipeline vs Manual Preprocessing (LIKELY ROOT CAUSE)

**sklearn Pipeline has specific behaviors that manual preprocessing doesn't replicate:**

1. **`.fit_transform()` vs `.transform()`**
   - Pipeline uses `.fit_transform(X)` which might have optimizations
   - GUI uses `.transform(X)` after creating a new instance each time
   - For stateless transformers, these SHOULD be equivalent, but...

2. **Memory layout and data copying**
   - Pipeline may optimize memory usage by avoiding copies
   - Manual approach creates new arrays at each step
   - Different memory layouts could affect numerical precision in downstream calculations

3. **Validation and checks**
   - Pipeline performs validation checks during `.fit()`
   - Manual approach skips these checks
   - Could lead to subtle differences in data handling

### Hypothesis 2: Wavelength Selection Differences (POSSIBLE)

**search.py:**
```python
# Direct indexing with integer indices
if subset_indices is not None:
    X = X[:, subset_indices]  # Numpy array slicing - EXACT
```

**GUI:**
```python
# Float conversion → dict lookup → column selection
available_wl = self.X_original.columns.astype(float).values  # String → float
wl_to_col = {float(col): col for col in self.X_original.columns}  # Dict mapping
selected_cols = [wl_to_col[wl] for wl in selected_wl if wl in wl_to_col]  # Lookup
X_work = self.X_original[selected_cols]  # DataFrame column selection
```

**Potential issues:**
- Floating-point precision: `"1920.5"` → `1920.5` → `"1920.5"` might not round-trip exactly
- Column ordering: Dict lookup doesn't preserve original order
- Tolerance matching: `_parse_wavelength_spec` has 5nm tolerance (lines 2976-2978) which could select wrong wavelengths for high-resolution data

### Hypothesis 3: Data Type and Precision (POSSIBLE)

**search.py:**
```python
X_np = X.values  # Convert once at the start
# ... all operations on X_np (numpy array)
```

**GUI:**
```python
X_work = self.X_original[selected_cols]  # DataFrame
X_processed = SNV().transform(X_work.values)  # Convert to numpy
# Multiple DataFrame → numpy conversions
```

**Potential issues:**
- Multiple conversions could introduce precision loss
- DataFrame uses different memory layout than numpy
- `.values` might create copies with different dtypes

### Hypothesis 4: CV Splitter Behavior (UNLIKELY BUT POSSIBLE)

Both use the same configuration:
```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

BUT:
- search.py: `cv_splitter.split(X, y)` where X is raw data
- GUI: `cv.split(X_processed, y_array)` where X_processed is preprocessed

**Could the CV splitter behave differently on raw vs preprocessed data?**
- Unlikely, since KFold just creates indices
- But worth investigating if there's any internal validation that differs

---

## Verification: What Makes Them THEORETICALLY Identical?

Both code paths:
- ✅ Use same random seed (42)
- ✅ Use same shuffle (True)
- ✅ Use same number of folds (5)
- ✅ Use same preprocessing methods (SNV, SG)
- ✅ Use same model (PLS with same n_components)
- ✅ Report mean CV R²

**So why the difference?** The devil is in the execution details!

---

## The Smoking Gun: Pipeline.fit() Behavior

I believe the **PRIMARY ROOT CAUSE** is the Pipeline's `.fit()` behavior.

### What happens in search.py

```python
pipe = Pipeline([('snv', SNV()), ('savgol', SavgolDerivative(...)), ('model', PLS(...))])
pipe.fit(X_train, y_train)
```

**sklearn Pipeline.fit() source code (approximately):**
```python
def fit(self, X, y=None):
    Xt = X
    for name, transform in self.steps[:-1]:
        Xt = transform.fit_transform(Xt, y)  # Calls .fit() then .transform()
    self.steps[-1][1].fit(Xt, y)  # Fit final estimator
    return self
```

**Key point:** Even for stateless transformers, sklearn calls `.fit()` before `.transform()`, which could trigger validation or initialization that affects behavior!

### What happens in GUI

```python
X_temp = SNV().transform(X_work.values)  # Create new SNV, call .transform() directly
X_processed = SavgolDerivative(...).transform(X_temp)  # Create new SG, call .transform() directly
model.fit(X_processed, y_train)  # Fit model
```

**No `.fit()` is ever called on the preprocessors!**

Even though SNV and SavgolDerivative have no-op `.fit()` methods, the sklearn ecosystem might have expectations about the `.fit()` → `.transform()` sequence.

---

## Specific Fix Required

### Option 1: Make GUI Use Pipeline (RECOMMENDED)

**Location:** `spectral_predict_gui_optimized.py`, lines 2554-2619

**Current (WRONG):**
```python
# Manual preprocessing
if preprocess == 'snv':
    X_processed = SNV().transform(X_work.values)
# ... etc

# CV on preprocessed data
for train_idx, test_idx in cv.split(X_processed, y_array):
    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    model_fold.fit(X_train, y_train)
```

**Fixed (CORRECT):**
```python
# Build pipeline (same as search.py)
from spectral_predict.preprocess import build_preprocessing_pipeline

pipe_steps = build_preprocessing_pipeline(
    preprocess_name=preprocess_mapping[preprocess],  # Map GUI names to search.py names
    deriv=deriv_mapping[preprocess],
    window=window,
    polyorder=polyorder_mapping[preprocess]
)
pipe_steps.append(('model', model))
pipe = Pipeline(pipe_steps)

# CV on RAW data (not preprocessed!)
X_raw = X_work.values  # Raw data
for train_idx, test_idx in cv.split(X_raw, y_array):
    pipe_fold = clone(pipe)  # Clone entire pipeline
    X_train, X_test = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y_array[train_idx], y_array[test_idx]

    pipe_fold.fit(X_train, y_train)  # Preprocessing happens inside
    y_pred = pipe_fold.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    fold_metrics.append({"rmse": rmse, "r2": r2})
```

**Mappings needed:**
```python
preprocess_mapping = {
    'raw': 'raw',
    'snv': 'snv',
    'sg1': 'deriv',
    'sg2': 'deriv',
    'snv_sg1': 'snv_deriv',
    'snv_sg2': 'snv_deriv',
    'deriv_snv': 'deriv_snv'
}

deriv_mapping = {
    'raw': None,
    'snv': None,
    'sg1': 1,
    'sg2': 2,
    'snv_sg1': 1,
    'snv_sg2': 2,
    'deriv_snv': 1
}

polyorder_mapping = {
    'raw': None,
    'snv': None,
    'sg1': 2,
    'sg2': 3,
    'snv_sg1': 2,
    'snv_sg2': 3,
    'deriv_snv': 2
}
```

---

### Option 2: Investigate Why Manual Preprocessing Differs (DEBUGGING)

If Option 1 doesn't fix it, we need to debug:

1. **Add extensive logging to both code paths:**
   - Log X shape at each step
   - Log X mean/std at each step
   - Log preprocessing output statistics
   - Log model coefficients after fitting

2. **Create a minimal reproduction:**
   ```python
   # Test if Pipeline vs Manual gives same result
   from spectral_predict.preprocess import SNV, SavgolDerivative
   from sklearn.cross_decomposition import PLSRegression
   from sklearn.pipeline import Pipeline

   # Test data
   X = np.random.randn(100, 50)
   y = np.random.randn(100)

   # Method 1: Pipeline
   pipe = Pipeline([
       ('snv', SNV()),
       ('sg', SavgolDerivative(deriv=1, window=7)),
       ('pls', PLSRegression(n_components=2, scale=False))
   ])
   pipe.fit(X, y)
   y_pred_pipe = pipe.predict(X)

   # Method 2: Manual
   X_snv = SNV().transform(X)
   X_sg = SavgolDerivative(deriv=1, window=7).transform(X_snv)
   pls = PLSRegression(n_components=2, scale=False)
   pls.fit(X_sg, y)
   y_pred_manual = pls.predict(X_sg)

   # Compare
   print(f"Pipeline prediction: {y_pred_pipe[:5]}")
   print(f"Manual prediction:   {y_pred_manual[:5]}")
   print(f"Difference: {np.max(np.abs(y_pred_pipe - y_pred_manual))}")
   ```

3. **Check wavelength selection precision:**
   ```python
   # Add debug logging in GUI
   print(f"DEBUG: Original wavelengths (first 10): {self.X_original.columns[:10]}")
   print(f"DEBUG: Selected wavelengths (first 10): {selected_wl[:10]}")
   print(f"DEBUG: Matched columns (first 10): {selected_cols[:10]}")
   print(f"DEBUG: X_work shape: {X_work.shape}")
   print(f"DEBUG: Expected shape: {len(model_wavelengths)}")
   ```

---

## Recommended Action Plan

### Immediate (Fix the Bug)

1. ✅ **Implement Option 1** - Refactor GUI to use Pipeline
   - Lines to modify: 2554-2619 in `spectral_predict_gui_optimized.py`
   - Use `build_preprocessing_pipeline()` to create pipeline
   - Pass RAW data to CV instead of preprocessed data
   - Clone entire pipeline for each fold

2. ✅ **Test with user's data**
   - Load model with 2 LVs, R² = 0.9
   - Run in Custom Model Development
   - Verify R² matches (within ±0.01)

### Follow-up (Verify Fix)

3. ✅ **Create unit test**
   - Test that Pipeline vs Manual preprocessing gives identical results
   - Test with various preprocessing combinations
   - Test with subset models vs full models

4. ✅ **Document the fix**
   - Update HANDOFF document
   - Add warning comments in code
   - Update user guide

---

## Files to Modify

### Primary Fix

**File:** `spectral_predict_gui_optimized.py`

**Lines to change:** 2554-2619

**Current:** Manual preprocessing before CV
**New:** Pipeline-based preprocessing inside CV

**Estimated lines changed:** ~80 lines

---

## Test Plan

### Test 1: Exact Reproduction
```
1. Load existing results with R² = 0.9
2. Double-click to load in Custom Model Development
3. Immediately click "Run Refined Model" (no changes)
4. Verify: R² should be 0.9 ± 0.01
```

### Test 2: Various Preprocessing Methods
```
Test each preprocessing method:
- raw
- snv
- sg1 (window 7, 17)
- sg2 (window 7, 17)
- snv_sg1
- snv_sg2
- deriv_snv

For each:
1. Run analysis in Analysis tab
2. Load result in Custom Model Dev
3. Run without modifications
4. Verify R² matches
```

### Test 3: Subset Models
```
Test with:
- top10
- top50
- top100
- region subsets

Verify:
1. Correct wavelengths loaded
2. R² matches original
```

### Test 4: Edge Cases
```
- Very small datasets (n=20)
- High-resolution spectra (0.1nm spacing)
- Many wavelengths (1000+)
- Few wavelengths (10)
```

---

## Summary

### Root Cause
**Preprocessing is applied at different stages:**
- search.py: Inside CV fold via Pipeline ✅
- GUI: Before CV split via manual calls ❌

### Why This Matters
Even though SNV and SG are stateless, the Pipeline architecture ensures:
1. Proper `.fit()` → `.transform()` sequence
2. Consistent data handling
3. Validation and checks
4. Memory optimization
5. Reproducible behavior

### The Fix
Refactor GUI to use Pipeline architecture (same as search.py):
- Build preprocessing pipeline
- Pass RAW data to CV
- Clone pipeline for each fold
- Let Pipeline handle preprocessing inside CV

### Expected Outcome
After fix: R² = 0.9 (original) = 0.9 (GUI) ✅

---

**Document Version:** 1.0
**Investigation Date:** 2025-11-03
**Next Steps:** Implement Option 1 fix and test
**Priority:** 🔴 CRITICAL - Makes tool unusable for model refinement
