# XGBoost R² Reproducibility Issue - DEBUG NEEDED

**Date:** 2025-01-10 (Updated: Session 2)
**Status:** ⚠️ TREE_METHOD FIX APPLIED - ISSUE STILL PERSISTS
**Priority:** HIGH - Core functionality issue

---

## Update 2025-01-10 Session 2: Tree Method Fix Applied - Issue Persists

### What We Tried (Latest Attempt)

Following Theory 4 from the original investigation, we implemented the `tree_method='hist'` fix:

**Files Modified:**
- `src/spectral_predict/models.py` (lines 556-564, 712-720)

**Changes Made:**
```python
# BEFORE (lines 556-564):
{
    "n_estimators": n_est,
    "learning_rate": lr,
    "max_depth": max_depth,
    "subsample": subsample,
    "colsample_bytree": colsample,
    "reg_alpha": reg_alpha,
    "reg_lambda": reg_lambda
}

# AFTER (added tree_method):
{
    "n_estimators": n_est,
    "learning_rate": lr,
    "max_depth": max_depth,
    "subsample": subsample,
    "colsample_bytree": colsample,
    "reg_alpha": reg_alpha,
    "reg_lambda": reg_lambda,
    "tree_method": "hist"  # ← ADDED THIS LINE
}
```

**Why We Thought This Would Work:**
- Grid search uses `tree_method='hist'` (line 543)
- This was NOT being saved to params dict
- `set_params()` would silently reset it to default 'auto'
- Different tree algorithms produce different results
- Expected: 0.04 R² difference explained

**Result:**
⚠️ **ISSUE STILL PERSISTS** - R² drop remains ~0.04 even after adding tree_method to params dict

### Other Issues Fixed This Session

While investigating, we also fixed two related model issues:

#### ⚠️ LightGBM Fix Attempted - Still Not Working
**Problem:** Models producing ridiculously low R² (~0.1 when other models get >0.9)
**Root Cause (Initial Theory):** `max_depth=6` in `get_model()` conflicted with `num_leaves` from grid search
**Fix Applied:**
- File: `src/spectral_predict/models.py` (lines 124, 188)
- Changed from `max_depth=6` to `num_leaves=31`
- LightGBM now uses `num_leaves` as primary complexity control (correct approach)

**Result:** ⚠️ **STILL NOT WORKING** - User reports LightGBM R² still ~0.1 (other models >0.9)

**Additional Investigation Needed:**
- Check if there are other LightGBM-specific parameters causing issues
- Verify LightGBM is actually using the grid search parameters
- Check for data type incompatibilities (LightGBM can be picky about dtypes)
- Verify categorical features aren't being mishandled
- Check for silent parameter rejection (like XGBoost issue)
- Compare LightGBM default params vs what's being set

**Note:** This is a SEPARATE issue from XGBoost but similar pattern (model not performing as expected)

#### ✅ CatBoost Disabled (Can't Install)
**Problem:** CatBoost won't run (ImportError)
**Root Cause:** Requires Visual Studio 2022 Build Tools (C++ components) on Windows
**Fix Applied:**
- File: `spectral_predict_gui_optimized.py` (lines 63-69, 1117-1123, 2077-2081)
- Added `HAS_CATBOOST` availability check at import
- CatBoost checkbox now disabled if not available
- Shows red message: "Requires Visual Studio 2022 Build Tools (not installed)"
- Tier selection won't auto-enable CatBoost if unavailable

**Result:** ✅ FIXED - CatBoost gracefully disabled when not available

#### ✅ Ensemble Feature Completed
**Status:** Fully implemented (~900 lines of code across 3 files)
- Model reconstruction from results
- Ensemble training workflow
- Results display in Tab 5
- Visualization integration
- Save/load functionality
- Tab 7 prediction support
- Documentation updated

**Result:** ✅ COMPLETE - Ensemble feature now production-ready (separate from XGBoost issue)

---

## Problem Summary

After implementing the complete XGBoost hyperparameter fix (adding subsample, colsample_bytree, reg_alpha to grid search), the R² drop between Results Tab and Model Development Tab **still persists at ~0.04**.

**Expected:** R² drop of 0.005-0.01 (like ElasticNet)
**Actual:** R² drop of 0.04 (no improvement from before)

---

## What We've Implemented (COMPLETE)

### ✅ Backend Implementation (All Working)

1. **model_config.py** - Added 4 new XGBoost parameters to all tiers
   - `subsample`: [0.8, 1.0] (standard), [0.7, 0.85, 1.0] (comprehensive)
   - `colsample_bytree`: [0.8, 1.0] (standard), [0.7, 0.85, 1.0] (comprehensive)
   - `reg_alpha`: [0, 0.1] (standard), [0, 0.1, 0.5] (comprehensive)
   - `reg_lambda`: [1.0, 5.0] (comprehensive only)
   - Location: Lines 105-134

2. **models.py** - Updated grid generation
   - Added all 7 parameters to XGBRegressor instantiation (lines 543-554)
   - **All 7 params ARE stored in params dict** (lines 556-564):
     ```python
     {
         "n_estimators": n_est,
         "learning_rate": lr,
         "max_depth": max_depth,
         "subsample": subsample,           # ✅ STORED
         "colsample_bytree": colsample,    # ✅ STORED
         "reg_alpha": reg_alpha,           # ✅ STORED
         "reg_lambda": reg_lambda          # ✅ STORED
     }
     ```
   - Updated default models to use optimized values (lines 106-118, 170-182)
   - Grid size: 8 → 64 configs (standard tier)

3. **search.py** - Updated function signatures
   - Added all 7 XGBoost parameters to `run_search()` (lines 21-38)
   - Passes parameters to `get_model_grids()` (lines 160-177)

### ✅ Frontend Implementation (All Working)

4. **spectral_predict_gui_optimized.py** - Complete XGBoost UI
   - **Variables created** (lines 298-334): 20+ BooleanVar/StringVar
   - **UI controls created** (lines 1156-1259): Full hyperparameter panel with 7 parameters
   - **Collection code added** (lines 3756-3912): ~160 lines collecting/validating all params
   - **Parameters passed** (lines 4050-4056): All 7 params sent to `run_search()`

5. **Model Development Tab** - Loads and applies saved params
   - **Loads 'Params' column** from results CSV (lines 5417-5425)
   - **Applies via set_params()** (lines 5427-5432)
   - Code path: `_load_model_config()` → `_run_refined_model_thread()` → `set_params()`

### ✅ Tier Optimization (Complete)

6. **Moved XGBoost to Comprehensive tier** due to speed
   - Quick: PLS, Ridge, ElasticNet (1-5 min)
   - Standard: PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM (5-15 min)
   - Comprehensive: +XGBoost, CatBoost, NeuralBoosted (45-60 min)
   - XGBoost with 64 configs takes ~27 minutes alone

---

## The Mystery: Why R² Still Drops 0.04

### What We Know

1. **All 7 parameters ARE being stored** ✅
   - Verified in models.py lines 556-564
   - The params dict includes all new parameters

2. **Model Development DOES load saved params** ✅
   - Code at lines 5417-5432 loads and applies params
   - Uses `ast.literal_eval()` to parse params string
   - Uses `model.set_params(**params_from_search)` to apply

3. **User confirms running fresh searches** ✅
   - Not loading old results
   - Running complete new grid search each time

4. **Backend tests all passed** ✅
   - Parameter flow verified
   - Grid generation verified
   - All imports working

### What Might Be Wrong (DEBUG NEEDED)

**Theory 1: Params Not Actually in CSV**
- Maybe the params dict isn't being serialized correctly to CSV?
- Need to check actual CSV file content
- Look at 'Params' column in results CSV
- **STATUS:** ⏳ NOT YET TESTED

**Theory 2: set_params() Failing Silently**
- XGBoost's `set_params()` might reject some parameters
- Check if exception is being caught and ignored
- Add logging to see what params actually get applied
- **STATUS:** ⏳ NOT YET TESTED

**Theory 3: Different Random State / CV Split**
- Maybe Model Development tab uses different CV split?
- Check if random_state is consistent
- Check if validation set exclusion is working
- **STATUS:** ⏳ NOT YET TESTED

**~~Theory 4: tree_method='hist' Not Saved~~** ❌ RULED OUT
- ~~`tree_method='hist'` is set in model but NOT saved to params dict~~
- ~~If default is different, this could cause issues~~
- **FIX ATTEMPTED:** Added `"tree_method": "hist"` to params dict (lines 556-564, 712-720)
- **RESULT:** Issue still persists - R² drop still ~0.04
- **CONCLUSION:** tree_method was likely not the cause, or there's another parameter issue

**Theory 5: Data Preprocessing Mismatch**
- Maybe wavelength filtering is different?
- Maybe preprocessing order is different?
- Check if full-spectrum preprocessing logic matches
- **STATUS:** ⏳ NOT YET TESTED

**NEW Theory 6: Other Implicit Parameters Missing**
- XGBoost has many parameters beyond the 8 we're saving
- Examples: `gamma`, `min_child_weight`, `max_delta_step`, `scale_pos_weight`, `base_score`
- Some might have non-default values during training but not saved
- Need to dump ALL params from trained model and compare with loaded model
- **STATUS:** ⏳ NEEDS INVESTIGATION

**NEW Theory 7: sklearn Pipeline Interference**
- Model Development might wrap model in Pipeline differently than grid search
- Pipeline preprocessing could add/remove features
- Check if preprocessing steps are exactly identical
- **STATUS:** ⏳ NEEDS INVESTIGATION

---

## Debugging Steps (URGENT)

### Step 1: Verify CSV Storage
```python
# After running grid search, check results CSV:
import pandas as pd
df = pd.read_csv('path/to/results.csv')
xgb_row = df[df['Model'] == 'XGBoost'].iloc[0]
print("Params column:", xgb_row['Params'])
# Should show: {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6,
#               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0}
```

### Step 2: Add Debug Logging to Model Development
In `spectral_predict_gui_optimized.py` around line 5429:
```python
if params_from_search:
    try:
        print(f"DEBUG: Before set_params: {model.get_params()}")
        model.set_params(**params_from_search)
        print(f"DEBUG: After set_params: {model.get_params()}")
        print(f"DEBUG: Applied saved search parameters: {params_from_search}")
    except Exception as e:
        print(f"WARNING: Failed to apply saved parameters {params_from_search}: {e}")
```

### Step 3: Compare Actual Model Params
After loading in Model Development, print the actual model params:
```python
# After line 5432
actual_params = model.get_params()
print("\nDEBUG: Actual XGBoost params after loading:")
for key in ['n_estimators', 'learning_rate', 'max_depth', 'subsample',
            'colsample_bytree', 'reg_alpha', 'reg_lambda']:
    print(f"  {key}: {actual_params.get(key, 'NOT SET')}")
```

### Step 4: Check Results Tab Params
Add debug output when saving results in search.py:
```python
# After model training, before adding to results:
if model_name == "XGBoost":
    print(f"DEBUG: Saving XGBoost params to results: {current_params}")
    print(f"DEBUG: Actual model params: {model.get_params()}")
```

---

## Technical Details for Debugger

### Key Files and Functions

1. **Grid Search Storage**: `src/spectral_predict/search.py`
   - Function: `run_search()` (lines 21-400+)
   - Stores params in results DataFrame
   - Params dict comes from `get_model_grids()` return value

2. **Model Development Loading**: `spectral_predict_gui_optimized.py`
   - Function: `_load_model_config()` (lines 4621-4780)
   - Function: `_run_refined_model_thread()` (lines 5228-5700)
   - Loads params at lines 5417-5425
   - Applies params at lines 5427-5432

3. **Grid Generation**: `src/spectral_predict/models.py`
   - Function: `get_model_grids()` (lines 211-728)
   - XGBoost grid: lines 520-567
   - Returns list of tuples: `[(model_instance, params_dict), ...]`

### Data Flow

```
Grid Search:
  get_model_grids() → returns [(XGBRegressor(...), {params_dict}), ...]
  ↓
  search.py iterates through configs
  ↓
  Trains model, stores params_dict in results DataFrame 'Params' column
  ↓
  Saves to CSV with to_csv()

Model Development:
  User double-clicks result row
  ↓
  _load_model_config() reads row from DataFrame
  ↓
  _run_refined_model_thread() called
  ↓
  Parses 'Params' column with ast.literal_eval()
  ↓
  Creates fresh model with get_model()
  ↓
  Applies saved params with model.set_params(**params_from_search)
  ↓
  Trains on same data/CV split
```

### Expected vs Actual Behavior

**Expected:**
- Results Tab R² = 0.95 (trained with subsample=0.8, colsample=0.8, reg_alpha=0.1)
- Model Dev R² = 0.945-0.95 (loaded same params, same data)
- Drop: 0.005-0.01 ✅

**Actual:**
- Results Tab R² = 0.95
- Model Dev R² = 0.91
- Drop: 0.04 ❌

This 0.04 drop is EXACTLY what we were trying to fix!

---

## What Still Needs Implementation (Lower Priority)

### Option B: Remaining 5 Models UI Implementation

The XGBoost fix is the critical priority, but for completeness, we still need:

1. **Add UI Controls** (similar to XGBoost panel)
   - ElasticNet: alpha [0.01, 0.1, 1.0], l1_ratio [0.3, 0.5, 0.7]
   - LightGBM: n_estimators [100, 200], learning_rate [0.1], num_leaves [31, 50]
   - CatBoost: iterations [100, 200], learning_rate [0.1], depth [4, 6]
   - SVR: kernel [rbf, linear], C [1.0, 10.0], gamma [scale, auto]
   - MLP: hidden_layer_sizes [(64,), (128,64)], alpha [1e-3], learning_rate_init [1e-3]

2. **Add Collection Code** (like XGBoost lines 3756-3912)
   - Parse checkboxes + custom inputs
   - Build parameter lists
   - Apply defaults if none selected

3. **Pass to run_search()** (like XGBoost lines 4050-4056)
   - Add all parameter lists as kwargs

**NOTE:** All 5 models already have:
- ✅ Variables created (lines 336-406)
- ✅ Backend support (models.py, search.py, model_config.py)
- ❌ UI controls NOT created (would add ~500 lines)
- ❌ Collection code NOT added (would add ~300 lines)

**Estimated Work:** ~3-4 hours for all 5 models

---

## Recommendations for Debugger

1. **Start with Step 1**: Check actual CSV file content
   - Open results CSV in text editor
   - Find XGBoost row
   - Verify 'Params' column contains all 7 parameters

2. **Add Debug Logging**: Modify lines as suggested above
   - Before/after set_params()
   - Print actual model params
   - Compare with saved params

3. **Test with Single Config**: Simplify to isolate issue
   - Use Quick tier (1 config only)
   - XGBoost with explicit params: n_estimators=100, subsample=0.8, etc.
   - Check if R² still drops

4. **Compare with ElasticNet**: Working model for reference
   - ElasticNet R² is stable (0.005-0.01 drop)
   - Compare its parameter loading vs XGBoost
   - What's different?

---

## System Info

- **OS:** Windows (win32)
- **Python:** Should be in .venv or C:\Python314
- **XGBoost:** Installed via pip in .venv
- **Working Directory:** C:\Users\sponheim\git\dasp
- **Branch:** claude/combined-format-011CUzTnzrJQP498mXKLe4vt

---

## Contact / Handoff

**What Works:**
- ✅ All backend code (models.py, search.py, model_config.py)
- ✅ All XGBoost UI (variables, controls, collection, passing)
- ✅ Tier optimization (XGBoost moved to comprehensive)
- ✅ All tests pass

**What Doesn't Work:**
- ❌ R² still drops 0.04 in Model Development tab (MYSTERY!)

**What's Not Done:**
- ⏳ UI controls for ElasticNet, LightGBM, CatBoost, SVR, MLP (optional)
- ⏳ Collection/passing code for above 5 models (optional)

**Critical Question:**
Why does Model Development tab produce 0.04 lower R² when it's supposedly loading and applying all 7 XGBoost parameters correctly?

---

## Summary for Next Agent

### Current State (2025-01-10 Session 2)

**XGBoost R² Drop Issue:**
- ❌ STILL NOT FIXED after tree_method addition
- All 8 parameters now in params dict (including tree_method)
- Issue persists: R² drops 0.04 between Results Tab and Model Development Tab
- Next steps: Test Theories 1, 2, 3, 6, 7 (listed above)

**LightGBM Low R² Issue:**
- ❌ STILL NOT FIXED after max_depth removal
- Gets R² of ~0.1 when other models get >0.9
- Likely a separate but similar parameter issue
- Needs full investigation (see Theory 6 approach for XGBoost)

**CatBoost:**
- ✅ FIXED - Gracefully disabled when VS2022 not available

**Ensemble Feature:**
- ✅ COMPLETE - Fully functional, production-ready

### Critical Debugging Approach for Next Agent

1. **Add comprehensive logging** to both tabs:
   - Log ALL XGBoost params after training (Results Tab)
   - Log ALL XGBoost params after loading (Model Development Tab)
   - Use `model.get_params()` to dump EVERYTHING, not just the 8 we think we're saving
   - Compare the two outputs to find what's different

2. **Do the same for LightGBM** - likely same root cause

3. **Check the actual CSV file** - open in text editor, verify params column

4. **Simplify test case**:
   - Use Quick tier with just XGBoost
   - Single config: n_estimators=100, everything else default
   - See if issue still happens with minimal configuration

5. **Compare with working model (ElasticNet)**:
   - ElasticNet R² is stable (0.005-0.01 drop)
   - What's different in how its params are saved/loaded?
   - Use same debugging approach to compare

### Files That Were Changed This Session

1. **src/spectral_predict/models.py**
   - Added tree_method to XGBoost params dict (lines 564, 720)
   - Changed LightGBM from max_depth to num_leaves (lines 124, 188)

2. **spectral_predict_gui_optimized.py**
   - Added HAS_CATBOOST check (lines 63-69)
   - Disabled CatBoost checkbox when not available (lines 1117-1123)
   - Added ensemble feature (~600 lines)

3. **src/spectral_predict/model_io.py**
   - Added save_ensemble() and load_ensemble() functions (~230 lines)

4. **docs/MACHINE_LEARNING_MODELS.md**
   - Added "Using Ensembles in the GUI" section (~150 lines)

### What Definitely Works

- ✅ Backend parameter infrastructure (all params defined in model_config.py)
- ✅ Grid generation (models.py creates grids with all params)
- ✅ UI collection (XGBoost params collected from GUI)
- ✅ ElasticNet, PLS, Ridge, RandomForest (all working normally)
- ✅ Ensemble feature (independent from this issue)

### What's Broken

- ❌ XGBoost R² reproducibility (0.04 drop)
- ❌ LightGBM performance (R² ~0.1 instead of >0.9)

### Key Insight

**Pattern Recognition:** Both XGBoost and LightGBM have similar issues where they don't perform as expected. This suggests:
- Possible common parameter handling bug
- Possible data type/preprocessing issue affecting gradient boosting models specifically
- Possible sklearn Pipeline interaction issue
- NOT just a single missing parameter (we've added many, issue persists)

The next agent should focus on **systematic parameter dumping and comparison** rather than guessing which parameter might be missing.

---

*Document created: 2025-01-10*
*Updated: 2025-01-10 Session 2*
*For: next debugging agent*
*Related: xboostfix.md, INVESTIGATION_SUMMARY.md*
