# R² Discrepancy Fix Plan - COMPREHENSIVE ANALYSIS

**Date:** 2025-11-03
**Status:** 🎯 READY TO IMPLEMENT
**Priority:** CRITICAL

---

## Executive Summary

**Problem:** When models with derivative preprocessing + variable selection are loaded into Model Development and re-run, R² is significantly lower than the original Results tab value.

**Root Cause:** Different preprocessing order between main analysis and model development.

**User's Key Observation:**
- ✅ **iPLS + derivatives**: Results tab R² ≈ Model Development R² (CORRECT)
- ❌ **Non-iPLS variable selection + derivatives**: Results tab R² >> Model Development R² (BUG)

**Why iPLS works:** iPLS selects **contiguous wavelength intervals**, preserving derivative context
**Why feature selection fails:** Selects **scattered wavelengths**, breaking derivative context when recomputed

---

## The Exact Bug

### Main Analysis (search.py) - CORRECT for derivatives + subsets

```python
# Line 434-449: For derivative preprocessing
if preprocess_cfg["deriv"] is not None:
    # Use ALREADY PREPROCESSED data (X_transformed)
    subset_result = _run_single_config(
        X_transformed,  # ← Already has derivatives computed on FULL spectrum
        y_np,
        wavelengths,
        model,
        ...,
        subset_indices=selected_indices,  # ← Subset AFTER preprocessing
        skip_preprocessing=True,  # ← Don't reapply preprocessing in CV
    )
```

**Flow:**
1. Preprocess **full spectrum** → derivatives computed with proper context
2. Subset to selected wavelengths → keeps derivative values from full spectrum
3. Run CV with `skip_preprocessing=True` → no preprocessing inside CV
4. **Result:** Correct R² (e.g., 0.9443)

### Model Development (GUI) - BROKEN for derivatives + subsets

```python
# Line 2630-2746: Current implementation
# Filter X_original to selected wavelengths
X_work = self.X_original[selected_cols]  # ← Subset RAW data FIRST

# Build preprocessing pipeline
pipe_steps = build_preprocessing_pipeline(...)
pipe_steps.append(('model', model))
pipe = Pipeline(pipe_steps)

# Run CV with preprocessing INSIDE each fold
X_raw = X_work.values  # ← Raw subset data
for fold in cv.split(X_raw, y_array):
    pipe_fold = clone(pipe)
    pipe_fold.fit(X_train, y_train)  # ← Recomputes derivatives on SUBSET
```

**Flow:**
1. Subset **raw data** to selected wavelengths FIRST
2. Build preprocessing pipeline with derivatives
3. Run CV → derivatives computed on **subset only** (WRONG CONTEXT!)
4. **Result:** Lower R² (e.g., 0.8266)

---

## Why iPLS Works But Feature Selection Doesn't

### Diagnostic Test Results

```
TEST 1: Feature Differences
  Contiguous (iPLS):        Max diff = 3.00e-01
  Non-contiguous (VarSel):  Max diff = 4.68e-01 (19.4x worse!)

TEST 2: R² Impact
  Contiguous (iPLS):        Delta R² = 0.18
  Non-contiguous (VarSel):  Delta R² = 1.62 (9x worse!)

TEST 3: Contiguity Detection
  iPLS intervals:           Contiguous = True, Max gap = 2.04 nm
  Feature selection:        Contiguous = False, Max gap = 333.33 nm
```

### Why Contiguity Matters for Derivatives

**Savitzky-Golay derivatives** use a sliding window across wavelengths:

**Full spectrum (1000 wavelengths):**
```
Wavelengths: [1500, 1501, 1502, ..., 2499, 2500]
Derivative at 1534 (window=17) uses: [1526, 1527, ..., 1541, 1542]
                                      └─────────────────────────┘
                                         8 neighbors each side
```

**iPLS selects contiguous interval** (e.g., 1700-1800 nm):
```
Selected: [1700, 1701, 1702, ..., 1799, 1800]
Derivative at 1750 uses: [1742, 1743, ..., 1757, 1758] ✅ Correct context
```

**Feature selection picks scattered wavelengths**:
```
Selected: [1502, 1518, 1534, 1550, 1566, ...]
                    ↓ When derivatives are recomputed on this subset ↓
Derivative at "1534" uses: ["1518", "1534", "1550"] ❌ WRONG! Missing 1526-1542!
```

The derivative calculation **fundamentally changes** because the neighboring wavelengths don't exist in the subset!

---

## Additional Issue: SNV Normalization

Even for **contiguous wavelengths**, there's still a difference (Delta R² = 0.18 in the test).

**Cause:** SNV normalizes each spectrum using mean/std of the **available wavelengths**.

- **Full spectrum → SNV → subset**: Normalized using all 1000 wavelengths, then subset
- **Subset → SNV**: Normalized using only the 50 selected wavelengths

While this is a smaller effect than the derivative context issue, it compounds the problem.

---

## The Fix: Match Main Analysis Behavior

### Implementation in spectral_predict_gui_optimized.py

**Location:** `_run_refined_model_thread` function (line 2612)

**Strategy:** Detect when we have derivative preprocessing + variable subset, then use the same approach as search.py.

### Fix Code (Pseudocode)

```python
def _run_refined_model_thread(self):
    # ... existing code to parse wavelengths and get parameters ...

    # Get selected wavelengths
    selected_wl = self._parse_wavelength_spec(...)
    selected_cols = [wl_to_col[wl] for wl in selected_wl]

    # Check if we have derivatives + subset
    is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']
    is_subset = len(selected_wl) < len(self.X_original.columns)

    if is_derivative and is_subset:
        # === SPECIAL PATH: Match search.py behavior ===
        print("DEBUG: Derivative + subset detected. Using full-spectrum preprocessing.")

        # 1. Build preprocessing pipeline (NO model)
        from spectral_predict.preprocess import build_preprocessing_pipeline
        prep_steps = build_preprocessing_pipeline(
            preprocess_name, deriv, window, polyorder
        )
        prep_pipeline = Pipeline(prep_steps)

        # 2. Preprocess FULL spectrum
        X_full = self.X_original.values
        X_full_preprocessed = prep_pipeline.fit_transform(X_full)

        # 3. Find indices of selected wavelengths in original data
        all_wavelengths = self.X_original.columns.astype(float).values
        wavelength_indices = [
            np.where(all_wavelengths == wl)[0][0] for wl in selected_wl
        ]

        # 4. Subset the PREPROCESSED data
        X_work = X_full_preprocessed[:, wavelength_indices]

        # 5. Build pipeline with ONLY the model (skip preprocessing)
        pipe_steps = [('model', model)]
        pipe = Pipeline(pipe_steps)

        # 6. Run CV on preprocessed subset (matches search.py!)
        # ... rest of CV code ...

    else:
        # === NORMAL PATH: For raw/SNV or full-spectrum models ===
        # Subset raw data, then preprocess inside CV
        X_work = self.X_original[selected_cols]

        # Build full pipeline with preprocessing + model
        pipe_steps = build_preprocessing_pipeline(...)
        pipe_steps.append(('model', model))
        pipe = Pipeline(pipe_steps)

        # Run CV
        # ... existing CV code ...
```

### Key Changes

1. **Detect derivative + subset scenario**
2. **Preprocess full spectrum first** (same as search.py line 437)
3. **Subset after preprocessing** (same as search.py subset_indices)
4. **Skip preprocessing in CV** (same as search.py skip_preprocessing=True)
5. **Keep existing behavior for raw/SNV or full-spectrum models**

---

## Expected Results After Fix

### Before Fix
```
Main Analysis (Results Tab):  R² = 0.9443
Model Development:            R² = 0.8266
Difference:                   -0.1176 (11.8% drop!) ❌
```

### After Fix
```
Main Analysis (Results Tab):  R² = 0.9443
Model Development:            R² = 0.9441 ± 0.002 (within CV variance)
Difference:                   ~0.0002 ✅
```

The small difference (< 0.01) is expected due to:
- CV fold randomness (even with same seed, numerical precision)
- Floating point precision

---

## Testing Plan

### Test 1: User's Actual Data (Critical)

Load the problematic model and verify R² matches:
- Original R²: 0.9443
- Expected after fix: 0.9440 ± 0.005

### Test 2: iPLS Models (Should still work)

Verify iPLS models continue to work correctly (shouldn't regress)

### Test 3: Raw/SNV Models (Should be unchanged)

Verify non-derivative models work exactly as before

### Test 4: Full-Spectrum Models (Should be unchanged)

Verify models without variable selection work exactly as before

---

## Implementation Steps

1. ✅ Understand the problem (DONE)
2. ✅ Verify hypothesis with diagnostic tests (DONE)
3. ⏳ Implement fix in spectral_predict_gui_optimized.py
4. ⏳ Test with user's actual data
5. ⏳ Add regression tests to prevent future breaks
6. ⏳ Update documentation

---

## Related Issues

- Variable selection GUI integration (VARIABLE_SELECTION_IMPLEMENTATION.md)
- Previous R² bug mentioned in VARIABLE_SELECTION_GUI_HANDOFF.md
- This issue was likely fixed before, then re-introduced during refactoring

---

## Questions Answered

**Q: Why does iPLS work but other variable selection methods don't?**
A: iPLS selects **contiguous intervals**, so derivative context is preserved. Other methods select **scattered wavelengths**, breaking derivative context.

**Q: Is the main analysis R² of 0.9443 correct or inflated?**
A: **Correct!** It's not data leakage. Derivatives are per-spectrum transformations that don't use information from other samples.

**Q: Should we warn users about this limitation?**
A: After the fix, no warning needed! The fix makes Model Development reproduce main analysis behavior correctly.

---

## Next Step

**IMPLEMENT THE FIX** in spectral_predict_gui_optimized.py::_run_refined_model_thread()
