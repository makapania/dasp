# R² Difference Analysis: Results Tab vs Custom Model Development

**Date:** 2025-11-03
**Issue:** Users seeing different R² values between Results tab and Custom Model Development tab

---

## Quick Answer

**Both tabs show CV mean R² values** (averaged across all folds), NOT single-fold results.

The R² differences you're seeing are likely due to:
1. **[JUST FIXED] Variable Count Bug** - Subset models were loading incomplete wavelengths
2. **Different wavelength specifications** - Subtle differences in which wavelengths are used
3. **Different preprocessing settings** - Window size or method differences

---

## Detailed Analysis

### What the Results Tab Shows

**File:** `src/spectral_predict/search.py`
**Lines:** 585-598

```python
# Run CV in parallel
cv_metrics = Parallel(n_jobs=-1)(...)

# Average metrics across ALL folds
mean_rmse = np.mean([m["RMSE"] for m in cv_metrics])
mean_r2 = np.mean([m["R2"] for m in cv_metrics])
```

**CV Configuration:**
- Number of folds: `folds=5` (default)
- Random seed: `random_state=42` (deterministic)
- Shuffle: `shuffle=True`

**Result:** The R² shown in Results tab is the **MEAN across 5 CV folds**

---

### What Custom Model Development Shows

**File:** `spectral_predict_gui_optimized.py`
**Lines:** 2578-2612

```python
# Run cross-validation manually
n_folds = self.refine_folds.get()  # Default: 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Collect metrics for each fold
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(...)):
    # ... train and evaluate ...
    fold_metrics.append({"rmse": rmse, "r2": r2, "mae": mae})

# Compute mean and std across folds
results['r2_mean'] = np.mean([m['r2'] for m in fold_metrics])
```

**CV Configuration:**
- Number of folds: `5` (default, user can change)
- Random seed: `random_state=42` (same as Results tab)
- Shuffle: `shuffle=True` (same as Results tab)

**Result:** Custom Model Development also shows **MEAN across CV folds**

---

## Why Are You Seeing Different R² Values?

### Reason 1: Variable Count Bug (JUST FIXED!) ⚠️

**This was the most likely cause and has been FIXED!**

**Before the fix:**
```
1. Run analysis with subset="top50"
   → Model trained on 50 wavelengths
   → Results tab shows: R² = 0.95, n_vars = 50

2. Double-click to load in Custom Model Development
   → BUG: Only 30 wavelengths loaded (not all 50!)
   → User unknowingly trains on 30 wavelengths
   → Shows: R² = 0.87 (lower because fewer features)
```

**After the fix (today's implementation):**
```
1. Run analysis with subset="top50"
   → Model trained on 50 wavelengths
   → Results tab shows: R² = 0.95, n_vars = 50

2. Double-click to load in Custom Model Development
   → FIXED: All 50 wavelengths loaded correctly
   → Trains on same 50 wavelengths
   → Shows: R² = 0.95 (same performance!)
```

**Files Modified:**
- `src/spectral_predict/search.py` - Now saves all wavelengths in `all_vars` column
- `spectral_predict_gui_optimized.py` - Now loads `all_vars` preferentially

---

### Reason 2: Different Wavelength Specifications

Even with the bug fix, you might see differences if:

**Scenario A: Different wavelength ranges**
```
Results tab (original analysis):
  - Wavelengths: 1500-2300 nm (800 wavelengths)
  - Subset: top50 → specific 50 wavelengths selected

Custom Model Development (after loading):
  - User modifies wavelength spec to "1500-2000" instead of original range
  - Different wavelengths selected
  - Different R²
```

**Solution:** Always verify the wavelength specification matches exactly!

---

### Reason 3: Different Preprocessing Settings

**Scenario B: Preprocessing mismatch**
```
Results tab:
  - Preprocessing: sg1 (1st derivative)
  - Window: 17

Custom Model Development:
  - User selects: sg1 (1st derivative)
  - Window: 7 (DIFFERENT!)
  - Different smoothing → different results
```

**Window size matters!**
- Window 7: Less smoothing, more noise, might capture finer details
- Window 17: More smoothing, less noise, might miss fine details
- Different windows = different R²

**Solution:** Check that preprocessing settings match exactly:
- Same method (raw, SNV, sg1, sg2, deriv_snv)
- Same window size
- Same polyorder

---

### Reason 4: Different Number of CV Folds

**Less likely but possible:**

```
Results tab: 5 folds (default)
Custom Model Development: User changed to 3 or 10 folds
```

Different fold counts can lead to slightly different mean R² values due to:
- Different train/test splits
- Different sample stratification
- More folds = more stable estimate but smaller test sets

**Solution:** Keep folds the same (5 is standard)

---

## How to Debug R² Differences

### Step 1: Check if bug fix solved it

**Try this NOW (after bug fix):**
1. Run a fresh analysis with subset models (top50, top100)
2. Double-click a result
3. Immediately click "Run Refined Model" without changing anything
4. Compare R² values

**Expected:** Values should be very similar (within ±0.01 due to numerical precision)

---

### Step 2: Verify configuration matches

**Check these in Custom Model Development tab:**

1. **Wavelengths:**
   - Look at the wavelength specification text box
   - Count: Should match `n_vars` from Results tab
   - For subsets: Should be exactly the wavelengths from `all_vars` column

2. **Preprocessing:**
   - Method dropdown: Should match "Preprocess" column
   - Window: Should match "Window" column
   - Verify sg1 vs sg2

3. **Model Type:**
   - Should match "Model" column (PLS, Ridge, Lasso, etc.)

4. **CV Folds:**
   - Default is 5 (same as analysis)
   - Check if you've changed it

---

### Step 3: Check for user modifications

**Common mistakes:**

```
❌ User loads model with 50 wavelengths
❌ User thinks "maybe I should try 1500-1700 range instead"
❌ User edits wavelength spec
❌ Trains on different wavelengths
❌ Gets different R²
❌ Confused why results differ
```

**Solution:** Don't modify loaded configurations unless you intend to experiment!

---

## Expected R² Variability

### When should values be identical?

**Never exactly identical, but very close:**

Even with IDENTICAL configurations, you might see small differences:
```
Results tab:   R² = 0.9523
Custom Model:  R² = 0.9521
Difference:    0.0002 ← ACCEPTABLE
```

**Why?**
- Floating-point precision
- Numerical computation order
- Platform differences (Windows vs Linux)

**Acceptable difference:** ±0.01

---

### When are larger differences expected?

**Legitimately different configurations:**

| Scenario | R² Difference | Reason |
|----------|---------------|--------|
| Different wavelengths | Large (±0.10+) | Fundamentally different features |
| Different window size | Medium (±0.05) | Different smoothing |
| Different preprocessing | Medium (±0.05) | Different feature transformations |
| Different model type | Variable | Different algorithms |
| Fewer wavelengths (bug) | Large (±0.10+) | Missing important features |

---

## Recommended Workflow

### For Reproducibility

**To get the SAME R² in Custom Model Development:**

1. ✅ Run analysis in Analysis Configuration tab
2. ✅ Wait for results
3. ✅ **Double-click** the result you want to refine
   - This auto-loads all settings
4. ✅ **DO NOT MODIFY** any settings initially
5. ✅ Click "Run Refined Model"
6. ✅ Verify R² matches (within ±0.01)

**THEN experiment:**
7. ✅ Modify settings as desired
8. ✅ Re-run to see impact
9. ✅ Compare to original

---

### For Experimentation

**To intentionally try variations:**

1. Load a model configuration (double-click result)
2. Note the original R² from Results tab
3. Modify ONE thing at a time:
   - Try different window sizes: 7, 11, 15, 19
   - Try different preprocessing: raw → SNV → sg1 → sg2
   - Try different wavelength ranges
4. Run and compare R² to original
5. Track what improves or hurts performance

---

## Verification Test

**Test if the bug fix resolved your issue:**

```python
# After running this session's code updates:

1. Start GUI
2. Load your spectral data
3. Configure analysis:
   - Model: PLS
   - Subset: top50
   - Preprocessing: sg1, window=17
4. Run Analysis
5. Note R² from Results tab (e.g., R² = 0.923)
6. Double-click that result
7. Custom Model Development should show:
   - n_vars = 50 ✓ (not 30!)
   - Wavelengths = 50 values ✓
8. Click "Run Refined Model" WITHOUT changing anything
9. Note R² from Custom Model Development
10. Compare:
    - If difference < 0.01: ✅ Working correctly!
    - If difference > 0.05: ⚠️ Something still different
```

---

## Summary

### Before Today's Fix
❌ Subset models lost wavelengths (only 30 saved)
❌ Custom Model Development trained on incomplete data
❌ R² differences were REAL and caused by missing features

### After Today's Fix
✅ All wavelengths saved in `all_vars` column
✅ Custom Model Development loads complete wavelength list
✅ R² should match (within ±0.01) when configs are identical

### If You Still See Differences

Check these in order:
1. ✓ Are you using results from BEFORE today's fix? (Re-run analysis!)
2. ✓ Did you modify wavelength specification?
3. ✓ Are preprocessing settings identical?
4. ✓ Is the same model type selected?
5. ✓ Are CV folds the same?

---

## Technical Details

### CV Splitting

Both Results and Custom Model Development use:
```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

**Key points:**
- `shuffle=True`: Samples are shuffled before splitting
- `random_state=42`: Same shuffle every time (reproducible)
- `n_splits=5`: 5-fold cross-validation (80% train, 20% test)

**Fold structure:**
```
Fold 1: Train on 80% (indices 20-99), Test on 20% (indices 0-19)
Fold 2: Train on 80% (indices 0-19, 40-99), Test on 20% (indices 20-39)
Fold 3: Train on 80% (indices 0-39, 60-99), Test on 20% (indices 40-59)
Fold 4: Train on 80% (indices 0-59, 80-99), Test on 20% (indices 60-79)
Fold 5: Train on 80% (indices 0-79), Test on 20% (indices 80-99)

Final R²: Mean of 5 test-set R² values
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-03
**Status:** Bug fixed, analysis complete
