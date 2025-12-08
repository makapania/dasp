# Reproducibility Issue - Handoff Document

## Problem Statement

PLS cross-validation produces **different R² values and different top models** across GUI restarts with identical data and settings.

### Observed Behavior
- User performs complete GUI restart (close → reopen → load data → run)
- Same BoneCollagen dataset, same settings
- Gets wildly different results each time:
  - Run 1: R² = 0.924 (4 LVs, deriv)
  - Run 2: R² = 0.937 (4 LVs, deriv)
  - Run 3: R² = 0.904 (4 LVs, deriv)
  - Earlier runs: 0.9584, 0.9558, 0.9448 (different LVs: 7, 7, 3)

### User Requirements
- Results must be **identical** when running in Results tab vs Model Development tab
- Only using default methods: "top variables" and "regional subsets" (NOT spa/uve)
- All runs use same preprocessing methods (deriv, snv_deriv, deriv_snv)

---

## What We've Tried (Chronologically)

### ❌ Attempt 1: Random Seed Initialization
**Hypothesis**: NumPy/Python random state not initialized consistently

**Changes Made**:
- Added `np.random.seed(42)`, `random.seed(42)` in multiple locations:
  - `run_search()` in search.py
  - `_run_analysis_thread()` in GUI
  - `_run_refined_model_thread()` in GUI
  - `_train_ensembles()` in GUI
- Fixed ensemble.py to use `KFold(n_splits=5, shuffle=True, random_state=42)` instead of `cv=5`

**Result**: ❌ **Failed** - Still got 3 different top models (R² = 0.9584, 0.9558, 0.9448)

**Action Taken**: Reverted all changes with `git stash`

---

### ❌ Attempt 2: Stable Sorting for Feature Selection
**Hypothesis**: Unstable `np.argsort()` causes different feature orderings → different NIPALS convergence → different R²

**Evidence**:
- sklearn's PLSRegression uses hardcoded NIPALS algorithm (iterative, sequential feature processing)
- Found `np.argsort()` without `kind='stable'` at:
  - Line 725 in search.py (top variable selection)
  - Line 1273 in search.py (variable importance ranking)

**Changes Made**:
```python
# Before
top_indices = np.argsort(importances)[-n_top:][::-1]

# After
top_indices = np.argsort(importances, kind='stable')[-n_top:][::-1]
```

**Result**: ❌ **Failed** - Still got different R² (0.924, 0.937, 0.904) with 3.3% variance

**Status**: Changes kept in place (not harmful, but didn't solve the problem)

---

### ❌ Attempt 3: File Loading Order (Glob Sorting)
**Hypothesis**: `path.glob()` returns files in **arbitrary filesystem-dependent order** → different sample ordering → CV fold indices point to different samples → different R²

**Evidence**:
- Python docs confirm `Path.glob()` returns files in arbitrary order
- Even with deterministic CV fold indices (KFold with random_state=42), if samples are loaded in different order, fold composition changes

**Changes Made**: Wrapped ALL `glob()` calls with `sorted()`:
- Lines 6573-6708: Data loading tab (ASD, CSV, SPC, JCAMP, ASCII, SP files)
- Line 6846: Reference file detection
- Lines 18522-18524: Prediction tab file detection
- Lines 22218-22220: Directory loading helper
- Lines 22853-22857, 23216-23218: Calibration Transfer tabs
- Lines 23023, 23247: Reference file auto-detection

**Result**: ❌ **Failed** - User reports "no, they are not the same. just as different as last time"

---

## Diagnostic Evidence Created

### 1. `test_cv_determinism.py`
Proves CV folds ARE deterministic when using `random_state=42`:
```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Shows IDENTICAL fold indices across multiple runs
```

### 2. `diagnose_simple.py`
Tests with synthetic data (49 samples, 2151 features):
- ✅ CV folds identical with random_state=42
- ✅ CV folds identical even WITHOUT global seed
- ✅ Dictionary iteration order consistent (Python 3.7+)
- ✅ Different folds → different R² (expected)

**Key Finding**: KFold creates INDEPENDENT random state - global `np.random.seed()` doesn't affect it

### 3. `diagnose_reproducibility.py`
Tests with real BoneCollagen data:
- Confirms CV fold determinism
- Tests impact of global random state
- Tests dictionary iteration order

---

## Current Code State

### Files Modified
1. **`src/spectral_predict/search.py`**:
   - Line 5: Added `import random` (NOT USED - can remove)
   - Line 725: Changed to `np.argsort(importances, kind='stable')`
   - Line 1273: Changed to `np.argsort(importances, kind='stable')`

2. **`spectral_predict_gui_optimized.py`**:
   - Multiple `sorted(list(path.glob(...)))` wrappers added throughout
   - All glob() calls now return deterministic file ordering
   - File compiles successfully (verified with py_compile)

### Git Status
```
M src/spectral_predict/search.py
M spectral_predict_gui_optimized.py
```

No commits made - all changes uncommitted

---

## What We Know For Sure

### ✅ NOT the Problem
1. **CV fold randomness** - Folds are deterministic with random_state=42
2. **Dictionary iteration order** - Python 3.7+ guarantees insertion order
3. **Global NumPy seed** - KFold uses independent RandomState
4. **Model testing order** - Consistent across runs

### ❓ Still Unexplained
1. **Why do R² values vary by 3-4%?** (0.904 to 0.937)
2. **Why do different LVs get selected?** (3 LVs vs 7 LVs vs 4 LVs)
3. **What causes different preprocessing methods to win?** (sometimes deriv, sometimes snv_deriv)

---

## Potential Remaining Issues

### 1. **Data Preprocessing Pipeline Order**
**Hypothesis**: Preprocessing methods might be tested in non-deterministic order

**Evidence Needed**:
- Check if preprocessing method iteration is deterministic
- Verify SNV/derivatives don't have randomness
- Check if MSC reference spectrum selection is deterministic

**Investigation**:
```python
# Search for preprocessing iteration in search.py
# Look for:
# - How preprocessing methods are selected
# - Order of deriv, snv_deriv, deriv_snv testing
# - Any set() usage (non-deterministic iteration)
```

### 2. **Feature/Variable Selection Randomness**
**Hypothesis**: "top variables" or "regional subsets" selection has hidden randomness

**User Confirmed**: Only using default methods:
- Top variables subset
- Regional subsets
- NOT using: spa, uve

**Investigation Needed**:
- Trace through "top variables" selection logic
- Check "regional subsets" generation
- Look for any shuffle/sample operations without random_state

**Key Code Locations**:
```
src/spectral_predict/search.py:
  - run_search() function
  - Feature selection methods
  - Subset generation
```

### 3. **Parallel Processing / Threading Issues**
**Hypothesis**: Multi-threading could introduce non-determinism if results are collected in arrival order

**Investigation Needed**:
- Check if search uses joblib/multiprocessing
- Verify if results are sorted deterministically after parallel execution
- Look for `n_jobs` parameter usage

### 4. **Data Loading from ASD Files**
**Hypothesis**: Even though file ORDER is now sorted, maybe file CONTENT reading has randomness?

**Investigation Needed**:
- Check `read_asd_file()` implementation in io_utils
- Verify wavelength alignment is deterministic
- Check if any interpolation/resampling has randomness

### 5. **Model Grid Iteration Order**
**Hypothesis**: The order in which models (different LV counts, preprocessing combos) are tested might not be deterministic

**Investigation Needed**:
```python
# In search.py, check how model grid is generated
# Look for:
# - How n_components values are iterated (range? list?)
# - How preprocessing combinations are generated
# - Any dict.keys() or set() usage
```

### 6. **Ties in Model Selection**
**Hypothesis**: If multiple models have identical R² (within floating point precision), the "best" model selection might be arbitrary

**Investigation Needed**:
- Check model selection logic in `run_search()`
- Add tie-breaking logic if needed (e.g., prefer fewer LVs, simpler preprocessing)

---

## Next Steps for Investigation

### Step 1: Add Comprehensive Logging
Add logging throughout `run_search()` to capture:
```python
print(f"Testing model: {preprocessing_method}, {n_components} LVs")
print(f"  CV R² = {r2_score:.6f}")
print(f"  Features selected: {selected_features}")
print(f"  Testing order: {model_index}/{total_models}")
```

This will show if:
- Model testing order changes between runs
- Same models produce different R² values
- Feature selection differs between runs

### Step 2: Test with Fixed Single Configuration
Force a single configuration to isolate the issue:
```python
# In search.py run_search(), hardcode:
preprocessing_methods = ['deriv']  # Only one method
n_components_range = [7]  # Only one LV count
subset_methods = [None]  # No subsetting

# Run 3 times - do you get IDENTICAL R²?
```

If YES → problem is in method/subset iteration
If NO → problem is deeper (in PLS fitting or CV itself)

### Step 3: Verify sklearn Determinism
Create minimal test:
```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold

# Load same data 3 times, fit same model
for run in range(3):
    model = PLSRegression(n_components=7)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"Run {run+1}: {scores.mean():.10f}")

# Should be IDENTICAL to 10 decimal places
```

If different → sklearn installation issue or hardware non-determinism
If same → problem is in dasp code

### Step 4: Binary Search the Pipeline
Systematically disable components:
1. Disable all preprocessing → test raw spectra
2. Disable subset selection → test full spectrum
3. Disable ensemble → test single model
4. Disable cross-validation → test train/test split

Find which component introduces non-determinism.

---

## Code References

### Key Files
- **`src/spectral_predict/search.py`**: Main search logic, CV, model selection
- **`spectral_predict_gui_optimized.py`**: GUI, data loading, threading
- **`src/spectral_predict/ensemble.py`**: Ensemble model training
- **`src/spectral_predict/io_utils.py`**: ASD file reading

### Key Functions
- `search.py:run_search()`: Main model search entry point (line ~500+)
- `search.py:_select_top_variables()`: Feature selection (line ~724)
- `gui:_run_analysis_thread()`: Results tab execution (line ~6000+)
- `gui:_run_refined_model_thread()`: Model Development tab execution

---

## Test Data
**Dataset**: `example/BoneCollagen.csv` + `example/*.asd`
- 49 samples
- ~2151 wavelengths
- Target: %Collagen

**Test Protocol**:
1. Close GUI completely
2. Reopen GUI
3. Load BoneCollagen data
4. Run with default settings (deriv, snv_deriv, deriv_snv)
5. Record top model R² and LVs
6. Repeat 3 times minimum

**Expected**: Identical R², LVs, preprocessing method
**Actual**: Different every time (variance up to 3-4%)

---

## Uncommitted Changes Summary

### search.py (3 changes)
```python
# Line 5
import random  # Added (unused - can remove)

# Line 725
top_indices = np.argsort(importances, kind='stable')[-n_top:][::-1]  # Added kind='stable'

# Line 1273
top_indices = np.argsort(importances, kind='stable')[-n_to_select:][::-1]  # Added kind='stable'
```

### spectral_predict_gui_optimized.py (15+ changes)
All `path.glob()` and `glob.glob()` calls wrapped with `sorted()`:
- Data loading: lines 6573, 6597, 6619, 6642, 6663, 6708, 6846
- Prediction tab: lines 18522-18524
- Helper functions: lines 22218-22220
- Calibration Transfer: lines 22853-22857, 23216-23218
- Reference detection: lines 23023, 23247

---

## Recommendations

### Option A: Deep Dive into Search Logic
1. Add extensive logging to `run_search()`
2. Capture exact model testing order, R² values, feature selections
3. Compare logs from 3 runs side-by-side
4. Find where divergence starts

### Option B: Simplify to Minimal Reproducible Case
1. Create standalone script that mimics search.py logic
2. Strip down to bare minimum (single preprocessing, single LV count)
3. Test if that's reproducible
4. Add complexity back piece by piece

### Option C: Alternative Approach - Force Determinism Everywhere
1. Set global seeds at every function entry point
2. Force all iteration to use sorted keys/indices
3. Add explicit sorting to ALL list operations
4. Heavy-handed but might work

### Option D: Investigate External Factors
1. Check if sklearn version has known non-determinism issues
2. Test on different machine/OS
3. Check BLAS/LAPACK backend (some use random initialization)
4. Verify NumPy MKL settings

---

## Contact Points

**User Confirmed**:
- Using default methods only (top variables, regional subsets)
- NOT using spa or uve
- Performing complete GUI restarts between tests
- Same data file every time (BoneCollagen)

**Last Output Files**:
- `outputs/results_%Collagen_20251122_124828.csv`: R²=0.924, 4 LVs, deriv
- `outputs/results_%Collagen_20251122_124734.csv`: R²=0.937, 4 LVs, deriv
- `outputs/results_%Collagen_20251122_124623.csv`: R²=0.904, 4 LVs, deriv

---

## Status: UNRESOLVED

**What Worked**:
- Stable sorting (safe to keep)
- File ordering (safe to keep, but didn't fix issue)

**What Didn't Work**:
- Random seed initialization
- File glob ordering

**Root Cause**: Still unknown

**Next Agent Should**: Follow "Next Steps for Investigation" section above, starting with comprehensive logging to identify where non-determinism enters the pipeline.
