# Random Forest Phase 1 Hyperparameter Implementation - Completion Instructions

## Summary

The Tab 4C Random Forest Specialist has successfully implemented 6 new Random Forest hyperparameter controls in the GUI. However, due to concurrent file modifications by other specialists, one manual step is required to complete the integration.

## Completed Work

### 1. GUI Controls Added (COMPLETED)
**File:** `spectral_predict_gui_optimized.py`
**Lines:** 2420-2492
**Status:** Successfully added to Tab 4C Random Forest section

The following 6 parameter controls were added using `_create_parameter_grid_control()`:
- `self.rf_min_samples_split_control` - Checkboxes: [2, 5, 10, 20], Default: [2]
- `self.rf_min_samples_leaf_control` - Checkboxes: [1, 2, 4, 8], Default: [1]
- `self.rf_max_features_control` - Checkboxes: ['sqrt', 'log2', None], Default: ['sqrt']
- `self.rf_bootstrap_control` - Checkboxes: [True, False], Default: [True]
- `self.rf_max_leaf_nodes_control` - Checkboxes: [None, 10, 50, 100, 500], Default: [None]
- `self.rf_min_impurity_decrease_control` - Checkboxes: [0.0, 0.01, 0.05, 0.1], Default: [0.0]

### 2. Backend Updated (COMPLETED)
**File:** `src/spectral_predict/search.py`
**Status:** Successfully updated

- **run_search() signature updated** (lines 25-27): Added 6 new RF parameters
- **get_model_grids() call updated** (lines 186-191): Passes new RF parameters

### 3. Backend Already Supports These Parameters (VERIFIED)
**File:** `src/spectral_predict/models.py`
**Status:** Already implemented in Phase 1

The `get_model_grids()` function already includes full support for all 6 parameters (lines 225-227, 611-646, 971-976).

## Manual Step Required

### Parameter Extraction in _run_analysis()

**Action Required:** Insert the parameter extraction code into `spectral_predict_gui_optimized.py`

**Location:** In the `_run_analysis()` method, after line ~6681
**After:** `rf_max_depth_list = sorted(rf_max_depth_list, key=lambda x: (x is not None, x))`
**Before:** `# Collect Ridge alpha values`

**Code to Insert:** See `RF_PARAMETER_EXTRACTION_CODE.txt` for the complete 74-line code block.

**Why Manual?** The file `spectral_predict_gui_optimized.py` is under concurrent modification by multiple specialists (XGBoost, LightGBM, SVR, MLP parameter specialists). To avoid conflicts, this extraction code should be manually inserted when the file stabilizes.

### How to Insert

1. Open `spectral_predict_gui_optimized.py`
2. Search for: `# Sort for consistent ordering (None sorts first)`
3. Find the line: `rf_max_depth_list = sorted(rf_max_depth_list, key=lambda x: (x is not None, x))`
4. After the blank line following this sort, insert the code from `RF_PARAMETER_EXTRACTION_CODE.txt`

### Then Update run_search() Call

**File:** `spectral_predict_gui_optimized.py`
**Location:** Around line 6580-6590 (in `_run_analysis()` method)
**Current code:**
```python
results_df, label_encoder = run_search(
    ...
    rf_n_trees_list=rf_n_trees_list,
    rf_max_depth_list=rf_max_depth_list,
    ridge_alphas_list=ridge_alphas_list,
    ...
)
```

**Add these 6 lines after `rf_max_depth_list=rf_max_depth_list,`:**
```python
rf_min_samples_split_list=rf_min_samples_split_list,
rf_min_samples_leaf_list=rf_min_samples_leaf_list,
rf_max_features_list=rf_max_features_list,
rf_bootstrap_list=rf_bootstrap_list,
rf_max_leaf_nodes_list=rf_max_leaf_nodes_list,
rf_min_impurity_decrease_list=rf_min_impurity_decrease_list,
```

## Verification Steps

After manual insertion:

1. **Start the GUI:**
   ```bash
   python spectral_predict_gui_optimized.py
   ```

2. **Navigate to Tab 4C (Analysis Configuration)**

3. **Expand "Random Forest Hyperparameters" section**

4. **Verify all 6 new controls are visible:**
   - Minimum Samples to Split (min_samples_split)
   - Minimum Samples in Leaf (min_samples_leaf)
   - Max Features for Splits (max_features)
   - Bootstrap Sampling (bootstrap)
   - Max Leaf Nodes (max_leaf_nodes)
   - Min Impurity Decrease (min_impurity_decrease)

5. **Test functionality:**
   - Check/uncheck boxes
   - Enter custom values (e.g., "10, 15, 20" or "5-20 step 5")
   - Run an analysis with Random Forest selected
   - Verify parameters appear in results

## Files Modified

1. ✅ `spectral_predict_gui_optimized.py` (lines 2420-2492) - GUI controls added
2. ⚠️ `spectral_predict_gui_optimized.py` (lines ~6683-6756) - **MANUAL INSERTION REQUIRED**
3. ⚠️ `spectral_predict_gui_optimized.py` (lines ~6580-6590) - **MANUAL PARAMETER PASSING REQUIRED**
4. ✅ `src/spectral_predict/search.py` (lines 25-27, 186-191) - Backend updated
5. ✅ `src/spectral_predict/models.py` - Already supports parameters (no changes needed)

## Reference Files

- `RF_PARAMETER_EXTRACTION_CODE.txt` - Complete extraction code to insert
- `RF_PHASE1_COMPLETION_INSTRUCTIONS.md` - This file

## Expected Behavior

Once complete, users can:

1. See 6 new Random Forest hyperparameter controls in Tab 4C
2. Select multiple values for each parameter using checkboxes
3. Enter custom values using the text entry field
4. Have these parameters passed through to the Random Forest model
5. See these parameters reflected in the results CSV

## Notes

- The GUI controls use the modern `_create_parameter_grid_control()` helper function
- Extraction uses `_extract_parameter_values()` for consistent parsing
- Backend already fully supports these parameters (implemented in Phase 1)
- Default values match scikit-learn Random Forest defaults
- Checkbox values chosen based on common hyperparameter tuning ranges

## Questions or Issues?

If you encounter any issues:
1. Verify the line numbers haven't shifted due to other changes
2. Check that all 6 controls are present in the GUI
3. Ensure the extraction code is properly indented (12 spaces for each line)
4. Confirm the run_search() call includes all 6 new parameters

---
**Implementation Date:** 2025-11-12
**Specialist:** Tab 4C Random Forest Specialist
**Phase:** Phase 1 Hyperparameter Completion
