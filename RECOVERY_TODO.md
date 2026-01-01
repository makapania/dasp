# Recovery TODO - December 16-20, 2025 Work

> Tracking 4 days of work to be preserved/restored.
> Created: 2025-12-20
> Status: **BACKED UP** to `backup_2025-12-20/`

---

## 🚨 TOP PRIORITY: BAYESIAN PLS UNDERPERFORMS (2025-12-30)

**STATUS: PLS-specific issue, not general Bayesian problem**

- **Bayesian Ridge:** R² > 0.97 (BETTER than grid search) ✓
- **Bayesian PLS:** R² 0.90-0.95 (worse than grid search ~0.96) ✗
- **Root cause:** Unknown - not the "best of 29" approach (Ridge uses same approach and works)
- **See:** "Bayesian Optimization - PLS-SPECIFIC ISSUE" section below for details

**Secondary issue:** R2 concordance between Results tab and Model Development tab - **FIXED (2026-01-01)** - See "R² Mismatch Fix" section below

---

## ✅ R² MISMATCH FIX (2026-01-01) - FIXED

**Problem:** When double-clicking a model in Results tab to load into Model Development, R² doesn't match for ElasticNet, RandomForest, LightGBM (but PLS/Ridge work fine).

**Root Cause:** The format→text_widget→parse round-trip destroys wavelength order. Additionally, `_update_wavelength_count()` event handler was clearing `_original_wavelength_order` during programmatic text widget updates.

**Fix Applied (commits 89fe683, 9212f02, fa551ce, 76bdc24):**

1. **Bypass the fragile parse** - Use stored wavelength order directly when available
2. **Add programmatic update flag** - `_programmatic_wavelength_update` distinguishes programmatic vs manual edits
3. **Move _update_wavelength_count() inside protected block** - Prevents clearing wavelength order during load

**Status:**
- [x] Code committed and pushed
- [x] **TESTED AND VERIFIED** - ElasticNet and LightGBM with 250+ variables now match perfectly

---

## USER PRIORITY FEATURES (Implement First)

### UI/Layout Fixes
- [x] **1. Move Data Management to Advanced section** - Currently in Data section, should be at bottom in Advanced
- [x] **2. Remove extra space in Data Management** - Space between icon and text
- [x] **3. Data Management opens on Import & Preview** - Should always open on Import & Preview tab

### Imbalance Handling
- [x] **4. Add SMOTE for regression** - RegressionResampler with 'smogn' method + GUI
- [x] **5. Add SMOTE-Tomek for regression** - RegressionResampler with 'smotetomek' method + GUI
- [x] **6. Add artificial sampling for regression imbalances** - RegressionResampler with 'oversample' method + GUI

### Variable Selection
- [x] **7. Add CARS** - Already exists in variable_selection.py and GUI
- [x] **8. Add VCPA-IRIV** - Already exists in wavelength_selection.py and GUI

### Genetic Algorithm Features
- [x] **9. Add GA Variable Selection** - Working (slow but functional)
- [x] **10. Add GA Preprocessing** - Already fully implemented in GUI + backend

### Ensemble/Buttons
- [x] **11. Redo all buttons after Ensemble Models** - COMPLETELY REWROTE ensemble_viz.py from scratch:
  - All 4 visualization functions rewritten with self-explanatory figures
  - Large readable fonts (12pt+ minimum), auto-scaling figure sizes
  - Title blocks with descriptions explaining what each figure shows
  - Interpretation boxes explaining how to read colors/values
  - All models shown with rankings and exact values
  - Key insights auto-generated (specialists, generalists, best per region)
  - Smart button enabling based on ensemble capabilities
  - Graceful failure with clear user-friendly messages for unsupported ensembles
- [x] **12. Check Ensembles for classification** - Ensembles do NOT support classification (all ensemble classes use RegressorMixin). Added "Regression Only" label to checkbox, added validation to skip ensembles for classification tasks with clear message, added validation to Train Ensemble button for classification.

### Optimization
- [x] **13. Add NSGA-II** - Fully integrated as third optimization option alongside Grid Search and Bayesian:
  - Added NSGA-II radio button and parameters (Pop/Gens) to UI
  - Added dispatch logic to run NSGA-II multi-objective optimization
  - Works with ALL models (PLS, Ridge, Lasso, ElasticNet, RF, LightGBM, XGBoost, CatBoost, SVR, MLP)
  - Optimizes 3 objectives: error, wavelength count, model complexity
  - Returns Pareto front with knee point detection for automatic "best compromise" selection

### Model Config Cleanup
- [x] **14. Remove 'Modern' from Gradient Boosting** - Changed to just "Gradient Boosting", removed green color override
- [x] **15. Remove all "new" logos** - Removed all 🆕 emojis from UI labels
- [x] **16. Rename 'Advanced Models' to 'Other Models'** - Done

### Preprocessing
- [x] **17. Verify SG3 and SG4 integration** - Verified: preprocess.py, search.py, and GUI all properly support 3rd/4th derivatives

### Performance
- [x] **18. Add GA Quick Mode** - Added "Quick" checkbox for GA Variable Selection (~10× faster: pop=32, gen=50, runs=2)

### Search Control
- [x] **19. Add Pause/Resume/Stop** - Added search controller with Pause/Resume/Stop buttons for long-running analyses

---

## DECEMBER 16-20 FEATURES (From Backup)

*Note: Some overlap with user features above - code exists in `backup_2025-12-20/`*

### Backend Modules (COMMITTED 2025-12-21)
- [x] **NSGA-II** - `src/spectral_predict/nsga2_search.py` - Multi-objective optimization
- [x] **GA-PLS** - `src/spectral_predict/ga_pls.py` - GA wavelength selection for linear models
- [x] **GA-LightGBM** - `src/spectral_predict/ga_lightgbm.py` - GA wavelength selection for tree models
- [x] **Library Search** - `src/spectral_predict/library_search.py` - Persistent spectral library
- [x] **Similarity Metrics** - `src/spectral_predict/similarity_metrics.py` - HQI, SAM, Euclidean, etc.
- [x] **Search Controller** - `src/spectral_predict/search_controller.py` - Pause/Resume/Stop

### GUI Integration (COMPLETED 2025-12-21)
- [x] **Library Search GUI** - Added Tab 12 "Spectral Library" with Library Management sub-tab
- [x] **Similarity Metrics GUI** - Added Similarity Search sub-tab with 7 metrics (HQI, SAM, Euclidean, Cosine, Deriv1/Deriv2 Correlation, SID)

### Test Suite (Pending)
- [ ] **Test Suite** - Integrate pytest tests from `backup_2025-12-20/tests/`

---

## Bug Fixes (COMPLETED 2025-12-21)

- [x] Fix 1: Edge masking for top_vars display
- [x] Fix 2: All-zero importances tracking
- [x] Fix 3: Baseline/smoothing in grid search
- [x] Fix 4: GA preprocessing in Model Development
- [x] Fix 5: Expand GA preprocessing to 4 model groups

---

## Dependency Cleanup (COMPLETED 2025-12-21)

- [x] Make all dependencies required (no optional imports)
- [x] Add: imbalanced-learn, platformdirs, threadpoolctl
- [x] Add: specdal, brukeropus, specio-py310, agilent-ir-formats
- [x] Remove all HAS_* conditional patterns from code

---

## Remaining Work

### VCPA-IRIV Performance Issue (FIXED 2025-12-22)
- [x] **VCPA-IRIV completely rewritten** - Now uses true algorithm from Yun et al. (2014):
  - Uses Mann-Whitney U test to compare include vs exclude RMSECV distributions
  - Classifies variables into 4 categories: strong, weak, uninformative, interfering
  - Removes uninformative/interfering iteratively until convergence
  - Now achieves R2=0.9984 vs baseline 0.6112, finds all true informative variables
  - More compact selection than CARS (7 vs 12 variables in test)

### GA Preprocessing Issue (FIXED 2025-12-22)
- [x] **GA Preprocessing gives lower R2 than default grid search** - Fixed two parameter mismatches:
  1. `n_components`: GA was using default 10, now uses `safe_max_components` (matches grid search)
  2. `cv_folds`: GA was using `ga_preprocess_cv_folds`, now uses `folds` (matches main search)

### Ensemble Issue (FIXED 2025-12-22)
- [x] **Ensemble gives lower R2 than individual models** - Fixed GA preprocessing reconstruction in `_reconstruct_models_from_results()`:
  - Added parsing of GA preprocessing names (e.g., `deriv1_w7_pls`)
  - Added `GAPreprocessWrapper` class to apply preprocessing during predict
  - Ensemble now uses same preprocessed data as individual models

### Edge Masking for Derivatives (PARTIAL - 2025-12-22)
- [x] **Edge masking added to SEARCH methods** - Savitzky-Golay derivatives create boundary artifacts at spectrum edges. Edge zone = window // 2 wavelengths on each side:
  - Added `_get_edge_zone_size()` and `_apply_edge_mask_to_data()` to `search.py`
  - Grid Search: Edge masking applied after preprocessing (lines 1194-1215)
  - Bayesian Search: Edge masking applied after preprocessing (lines 2191-2207)
  - NSGA-II: Already had correct edge masking (lines 575-581)
  - Example: window=15 → edge_zone=7 → first/last 7 wavelengths excluded
  - **Note**: Model Development Tab does NOT apply edge masking - it uses the wavelengths from `all_vars` which are already edge-masked by the search functions
- [x] **Bug fix: Removed erroneous double edge masking** - Edge masking was incorrectly added to Model Development tab (lines 20557-20577), causing wavelengths to be masked twice. This caused small R2 mismatches (0.001-0.01) for ElasticNet. Fixed by removing the Model Development edge masking block.
- [x] **ORIGINAL PROBLEM FIXED (2026-01-01)** - R2 mismatch between Results tab and Model Development tab was caused by wavelength order corruption. **FIXED** - see "R² MISMATCH FIX" section above. Commits 89fe683, 9212f02, fa551ce, 76bdc24. **VERIFIED WORKING.**

### Bayesian Optimization - PLS-SPECIFIC ISSUE (2025-12-30)

**STATUS: PLS UNDERPERFORMS, RIDGE WORKS GREAT**

**Observed behavior:**
- **Bayesian Ridge: R² > 0.97** (BETTER than grid search) ✓
- **Bayesian PLS: R² 0.90-0.95** (WORSE than grid search which gets ~0.96) ✗

**Key insight:** The issue is PLS-SPECIFIC, not a general problem with the optimization structure. Ridge uses the same "best of 29" approach and works great.

**Previous analysis (may be wrong):**
- Thought "best of 29 subsets" return caused noisy TPE learning
- Thought matching spectral-predict's "1 eval per trial" would fix it
- BUT if that were true, Ridge should also underperform (it doesn't)

**Outstanding questions:**
1. Why does PLS specifically underperform?
2. Is there something in PLS model building or evaluation that's different?
3. Is VIP-based importance for PLS causing issues vs simple coefficients for Ridge?
4. Does n_components constraint behave differently with variable subsets?

**Plan file:** `.claude/plans/memoized-wondering-cosmos.md` contains detailed analysis

**Previous fix (2025-12-22):** Double edge masking was fixed (commit d22a528).

---

## 🚨 NEW ANALYSIS (2025-12-22 Evening) - ROOT CAUSE FOUND

### Symptoms
- R2 dropped from 0.97 to 0.91 after recent uncommitted changes
- CARS subsets may not be appearing in results

### ROOT CAUSE: DOUBLE EDGE MASKING

There are TWO types of edge masking being applied in Bayesian:

1. **`_apply_edge_mask_to_data()`** in `run_bayesian_search()` (lines 2191-2207)
   - REMOVES columns from X (makes array smaller)
   - Added in uncommitted changes to search.py

2. **`_apply_edge_mask()`** in `_run_single_config()` (line 2852)
   - ZEROS OUT importance values at edges (doesn't change array size)
   - This was already there before

**What happens when both are applied:**
1. X goes from 2000 → 1986 columns (edges physically removed by step 1)
2. Then importances[0:7] and importances[-7:] get zeroed (step 2)
3. But those are NOW VALID wavelengths (not the original edges!)
4. Result: 14 valid wavelengths have their importances incorrectly zeroed
5. Variable selection picks wrong wavelengths → R2 drops

### FIX REQUIRED

Remove the column-removal edge masking from `run_bayesian_search()`:

**DELETE lines ~2191-2207 in search.py:**
```python
wavelengths_for_model = wavelengths
n_features_for_model = n_features
if preprocess_cfg.get("deriv") and preprocess_cfg.get("window"):
    X_preprocessed, wavelengths_for_model, edge_zone_applied = _apply_edge_mask_to_data(...)
    if edge_zone_applied > 0:
        n_features_for_model = X_preprocessed.shape[1]
        print(f"  Edge masking: {edge_zone_applied} wavelengths removed per side")
        print(f"  Wavelengths after masking: {n_features_for_model}")
        print(f"  Range: {wavelengths_for_model[0]:.1f} - {wavelengths_for_model[-1]:.1f} nm")
```

**REVERT** `wavelengths_for_model` → `wavelengths` and `n_features_for_model` → `n_features` in:
- Lines ~2228-2254 (create_objective_function call)
- Lines ~2289-2290 (convert_optuna_result_to_dasp_format call)

### Why This Works
- `_apply_edge_mask()` inside `_run_single_config()` already handles edge masking for importances
- It zeros out edge values without changing array size
- No double masking occurs
- This matches how Grid Search handles it (Grid Search does NOT remove columns)

### Additional Investigation Needed
- Check why CARS subsets are not appearing (could be a separate issue)
- Verify the variable selection methods are all running without exceptions

---

## Previous Fixes (for reference only)

**Commits that were investigated (but NOT the root cause):**
1. c6a9dd1 - SubsetTag key mismatch (harmless fix)
2. ca3d50b - Added cars-aware and vcpa-iriv support (harmless addition)
3. 6d9938f - Expanded default variable selection (changed behavior but not root cause)

**The actual problem is UNCOMMITTED changes to search.py that add _apply_edge_mask_to_data() to run_bayesian_search().**

**Files with uncommitted changes:**
- `src/spectral_predict/search.py` - Lines 2191-2207, 2228-2254, 2289-2290
- `src/spectral_predict/bayesian_utils.py` - Lines 203, 215, 370-423

### NSGA-II - BROKEN (2025-12-22)

**STATUS: DOES NOT WORK** - Gives much worse RMSE than Grid Search for regression.

**ROOT CAUSES IDENTIFIED:**

1. **CRITICAL: Data Leakage in Preprocessing** (`nsga2_search.py` lines 636-644)
   - Grid/Bayesian: Applies preprocessing **per-fold inside CV** (correct)
   - NSGA2: Applies preprocessing **globally before CV** (wrong - data leakage)
   - Test folds "see" training data statistics via preprocessing fit
   - This causes optimistic fitness during optimization, then worse real-world performance

2. **HIGH: Different Model Hyperparameters** (`_build_model()` vs `get_model()`)
   - Ridge alpha: Grid uses 1.0, NSGA2 searches 0.01-1000 (may pick extremes)
   - XGBoost max_depth: Grid uses 6, NSGA2 uses 3-7
   - XGBoost subsample: Grid uses 0.8, NSGA2 only has 1.0 or 0.8
   - LightGBM min_child_samples: Grid uses 5, NSGA2 doesn't tune it

3. **MEDIUM: Learning Rate Formula** (line 269)
   - Changed from 3.0^(gene/14) to 30.0^(gene/14) in commit 649ee9c
   - New formula is mathematically correct (0.01-0.3 range)
   - But comment claiming gene=7 gives 0.1 is wrong (actually 0.055)

4. **LOW: Edge Masking Constraint** (lines 577-597)
   - Edge masking reduces wavelength count after selection
   - Solutions penalized when n_selected < min_wavelengths after masking
   - GA doesn't know to compensate for edge wavelength loss

**FIX OPTIONS:**

- **Quick fix:** Align `_build_model()` defaults with `get_model()` (Ridge alpha=1.0, etc.)
- **Proper fix:** Refactor `_compute_prediction_error()` to use sklearn Pipeline with per-fold preprocessing

**FILES:** `src/spectral_predict/nsga2_search.py`

**See:** `.claude/plans/crystalline-herding-lovelace.md` for detailed investigation notes

### Test Suite (Low Priority)
- [ ] Integrate pytest tests from `backup_2025-12-20/tests/`
- [ ] 50+ test files for regression testing

---

## Completed Summary (2025-12-21)

All major features from December 16-20 work have been restored:
- 19 user priority features
- 6 backend modules
- 2 GUI integration items (Library Search + Similarity Metrics)
- 5 bug fixes
- Dependency cleanup (all required, no optional)

### Backup Location
`C:\Users\sponheim\git\dasp\backup_2025-12-20\`
