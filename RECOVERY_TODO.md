# Recovery TODO - December 16-20, 2025 Work

> Tracking 4 days of work to be preserved/restored.
> Created: 2025-12-20
> Status: **BACKED UP** to `backup_2025-12-20/`

---

## ✅ FIXED: NSGA-II Result Discrepancy (2026-01-06)

**Problem:** Optimization reports one RMSE but results display shows different value.

**Observed:**
- During optimization: Best error converged to **1.04**
- In results table: RMSE = **1.17**, R² = **0.95**
- Discrepancy: ~0.13 RMSE difference

**Root Cause Found (2nd investigation):**
`_compute_display_rmse()` and `_compute_solution_r2()` were **missing edge masking** that `_compute_prediction_error()` applies. During optimization, edge wavelengths (first/last `window//2`) are excluded because Savitzky-Golay interpolation creates unreliable boundary artifacts. But the display functions were including these unreliable edges, causing worse RMSE.

**Fix Applied (2026-01-06):**
Added edge masking to both display functions in `nsga2_search.py`:
1. `_compute_display_rmse()` - lines 2156-2162
2. `_compute_solution_r2()` - lines 2052-2058

```python
# Apply edge masking (same as _compute_prediction_error)
edge_zone = _get_edge_zone_size(preproc_idx, window_idx)
if edge_zone > 0:
    wavelength_mask = wavelength_mask.copy()
    wavelength_mask[:edge_zone] = False
    wavelength_mask[-edge_zone:] = False
```

**Result:** Optimization RMSE and display RMSE now match exactly.

**Previous partial fix:** Also added `include_best_from_all` parameter to `convert_nsga2_to_v1_format()` to ensure minimum error solution from all evaluations appears in results table.

**Added:** 2026-01-06 after importance-biased mutation implementation

---

## ✅ FIXED: NSGA-II Wavelength Reduction (2026-01-06)

### Issue 1: NSGA-II Selects Too Many Wavelengths - FIXED

**Problem:** NSGA-II consistently selected 800-1100+ wavelengths, while Grid Search's most competitive models use much smaller subsets (typically 50-300 wavelengths via CARS, SPA, iPLS, importance selection).

**Root Causes Found:**
1. Initial population started with 100% wavelengths selected
2. Wavelength count objective was linear (no extra penalty for large counts)
3. Mutation had no sparsity bias (equally likely to add/drop)

**Fixes Applied (2026-01-06):**

1. **CARS-Tree importance guidance** (lines 1527-1554)
   - Pre-compute CARS-Tree importance scores before NSGA-II starts
   - Uses hybrid split+gain importance (denser than plain CARS)
   - Provides principled wavelength importance for all downstream operations

2. **Importance-weighted population initialization** (SeededWavelengthSampling class, lines 303-426)
   - Population starts with ~250 wavelengths (configurable via `target_n_wavelengths`)
   - Wavelength selection probability proportional to CARS importance
   - Random solutions get 125-375 wavelengths (target/2 to target*1.5)

3. **Sparsity bias in mutation** (SmartMutation class, lines 157-200)
   - Added `sparsity_bias` parameter (default 2.0, adjusted to 1.9)
   - Dropping wavelengths is 1.9x more likely than adding
   - Combined with importance protection: high-importance wavelengths harder to drop

4. **Quadratic wavelength penalty** (line 1141-1144)
   - Changed from `n_selected / n_total` to `(n_selected / n_total)²`
   - 800 wavelengths penalized 4x more than 400 wavelengths

**Results:**
- Before: 800-1100+ wavelengths
- After: 48-201 wavelengths (mean ~115)
- Test validation: PASS (mean < 300, max < 400)

**Test script:** `scripts/test_nsga2_wavelength_reduction.py`

**Files modified:** `src/spectral_predict/nsga2_search.py`

---

### Issue 2: R² Appears Low Relative to RMSE

**Problem:** The R² values in NSGA-II results seem low compared to what the RMSE would suggest.

**TODO:** Verify `_compute_solution_r2()` calculation is correct:
- Check if it uses the same edge masking (now added)
- Check if cross-validation setup matches optimization
- Compare R² formula with Grid Search R² calculation
- Verify same wavelengths and preprocessing are used

**Files:** `src/spectral_predict/nsga2_search.py` - `_compute_solution_r2()` (lines 1996-2092)

---

### Issue 3: Model Won't Run in Model Development - PARTIALLY FIXED

**Error:** `slice indices must be integers or None or have an __index__ method`

**Root Cause:** Window parameter loaded as float (e.g., `31.0`) instead of int (`31`).

**Fix Applied (2026-01-06):** Added `int()` cast at line 21103-21105:
```python
window_from_config = self.selected_model_config.get('Window', None)
if window_from_config is not None and not pd.isna(window_from_config):
    window_from_config = int(window_from_config)
```

**Status:**
- ✅ NSGA-II results with derivatives now load into Model Development
- ❌ NSGA-II results with raw preprocessing still fail (see Issue 4)
- ❌ R² mismatch between Results and Model Development for derivatives (see Issue 5)

---

### Issue 4: NSGA-II Raw Preprocessing Fails in Model Development

**Problem:** When NSGA-II selects 'raw' preprocessing, Model Development fails to run the model.

**Status:** NOT INVESTIGATED

**TODO:** Debug what happens when preprocessing is 'raw' - may be missing handling in `_parse_coupled_preprocessing()` or similar.

---

### Issue 5: NSGA-II R² Mismatch with Derivatives

**Problem:** R² shown in Results tab does not match R² computed in Model Development tab when derivatives are used.

**Status:** NOT INVESTIGATED

**Possible Causes:**
1. Edge masking differences between NSGA-II and Model Development
2. Wavelength order issues (similar to previous Unified Bayesian bug)
3. Different CV setup between optimization and Model Development

**TODO:** Compare how preprocessing + wavelengths are applied in:
- `nsga2_search.py` - `_compute_solution_r2()`
- `spectral_predict_gui_optimized.py` - `_run_refined_model_thread()`

---

### Issue 6: NSGA-II Only Shows PLS Results When Multiple Models Selected - LOGGING ADDED (2026-01-06)

**Problem:** User selects multiple models (PLS, Ridge, LightGBM, etc.) for regression, but results only contain PLS models. Runs quickly, suggesting other models aren't being evaluated.

**Status:** LOGGING ADDED - Ready to diagnose

**Root Cause Found:** Non-PLS models were failing silently in `_compute_prediction_error()`. The exception handler returned 1e10 (penalty) with NO logging, making it impossible to debug.

**Fix Applied (2026-01-06):**

1. **Added logging module** (line 54-56):
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```

2. **Replaced silent exception handler** (lines 1247-1268):
   - Tracks failures per model type in `_failure_counts` dict
   - Logs WARNING on first failure of each model type (visible by default)
   - Logs DEBUG for subsequent failures
   - Includes model type, preprocessing, wavelength count, and error message

3. **Added failure summary after optimization** (lines 1707-1716):
   - Prints model failure counts when verbose >= 1
   - Logs INFO level summary

4. **Fixed XGBoost deprecated parameter** (line 866):
   - Removed `use_label_encoder=False` (deprecated in XGBoost 1.3+, removed in 2.0)

**Next Steps:**
1. Run NSGA-II with multiple models
2. Check console for failure warnings (e.g., "NSGA-II: First failure for LightGBM: ...")
3. Failure summary at end shows which models failed and how many times
4. Fix the specific model failures based on the logged error messages

---

## ✅ RESOLVED: CARS vs CARS-Tree Investigation (2026-01-06)

### Conclusion: They ARE Different Algorithms - Keep Both

**File:** `src/spectral_predict/variable_selection.py`

**Findings:**

1. **Different internal models:**
   - CARS: Uses PLS regression coefficients for importance
   - CARS-Tree: Uses LightGBM with hybrid split+gain importance blend

2. **Different variable selection behavior:**
   - CARS-Tree selects MORE variables (best RMSECV at earlier iterations)
   - Tree models tolerate high dimensionality; PLS needs cleaner variable sets
   - Minimum weight floor (`1e-6`) in CARS-Tree keeps variables alive longer

3. **Different use cases:**
   - CARS: Best for linear models (PLS, Ridge, Lasso, ElasticNet)
   - CARS-Tree: Best for tree models (LightGBM, RF, XGBoost, CatBoost)

**Decision:** Keep both as separate options. User observation confirmed - tree models prefer CARS-Tree subsets.

---

## ✅ CLASSIFICATION METRICS FIXES (2026-01-06)

### Issue: Rankings Decoupled from Accuracy
Models with high accuracy were ranked poorly because ranking used ROC_AUC instead of Accuracy.

### Fixes Applied

**1. Changed Classification Ranking (scoring.py lines 83-100)**
- BEFORE: Ranked by ROC_AUC only
- AFTER: Ranked by Accuracy (primary) + F1 (tiebreaker, 0.0001 weight)

**2. Changed CV Metrics Averaging (search.py line 2812)**
- BEFORE: `'weighted'` averaging for F1/Precision/Recall
- AFTER: `'macro'` averaging (treats all classes equally, consistent with ROC_AUC)

**3. Fixed Validation ROC_AUC Class Mismatch (search.py lines 507-541)**
- Problem: Training has N classes, validation has fewer (e.g., train=3, val=2)
- Model's `predict_proba()` returns columns for ALL training classes
- Fix: Always subset probabilities to classes present in validation, then renormalize
- Works for any number of classes (binary or multiclass)

**4. Fixed CV ROC_AUC Single-Class Folds (search.py lines 2877-2881)**
- Problem: Some CV folds might have only one class by chance
- Fix: Check `n_classes_test < 2` before computing ROC_AUC, return NaN if single class

**5. Fixed Pipeline Not Fitted Warning (search.py lines 2738-2902)**
- Problem: When sample weights are used, Pipeline steps are fitted manually but Pipeline itself not marked as fitted
- Fix: Track manual fitting with `manual_fit_used` flag, use manual transform/predict when needed

### Verification
- Validation ROC_AUC now computes correctly even when val set has fewer classes
- ROC_AUC = 1.0 with F1 < 1.0 is valid (perfect ranking but suboptimal threshold)

---

## ✅ VALIDATION METRICS IN RESULTS TABLE (2026-01-03) - COMPLETE

**Feature:** Checkbox in Validation subtab to compute and display validation metrics for top N models in Results table.

### Regression - WORKING
- `val_R2` and `val_RMSE` columns added to results
- Verified to match Predict tab exactly (tested to 4+ decimal places)
- Uses correct preprocessing order: preprocess FULL spectrum THEN subset
- Uses `ast.literal_eval()` + `set_params()` for model reconstruction
- Columns appear in CSV export

### Classification - COMPLETE (2026-01-03, updated 2026-01-06)
- `val_Accuracy` - accuracy score
- `val_ROC_AUC` - ROC AUC (handles class mismatch between train/val)
- `val_F1` - F1 score (macro averaging for multiclass)
- `val_Precision` - Precision score (macro averaging for multiclass)
- `val_Recall` - Recall score (macro averaging for multiclass)

**PLS-DA ISSUE - FIXED:**
- `_rebuild_model_from_row()` now wraps PLSTransformer with LogisticRegression for PLS-DA classification
- Matches the pipeline structure used during search (`search.py:2933-2940`)

### Files
- `src/spectral_predict/search.py`:
  - `_rebuild_model_from_row()` (lines 231-294): Rebuilds model, PLS-DA pipeline fix at lines 283-292
  - `compute_validation_metrics_for_top_models()` (lines 297-547): Computes all validation metrics
  - Column initialization: lines 342-351
  - Classification metrics calculation: lines 463-505
  - Column reordering: lines 524-545
- `spectral_predict_gui_optimized.py`:
  - UI variables: lines 1402-1403
  - Checkbox/spinbox: lines 7289-7310
  - Checkbox text updated to generic "Show validation metrics in results table"

### Implementation Details
1. **Binary vs Multiclass Detection:** Uses `len(np.unique(y_val))` to determine averaging method
2. **Error Handling:** Each metric wrapped in try/except for graceful failure
3. **ROC AUC:** Checks `hasattr(model, 'predict_proba')` before computing
4. **Column Ordering:** Validation metrics placed after CV metrics (Accuracy, ROC_AUC)

### All Classification Models Support predict_proba
- PLS-DA: ✅ (via LogisticRegression wrapper)
- RandomForest: ✅
- MLP: ✅
- NeuralBoosted: ✅
- SVM: ✅ (probability=True set explicitly)
- XGBoost: ✅
- LightGBM: ✅
- CatBoost: ✅

---

## ✅ PLS DEFAULT FOR CLASSIFICATION FIX (2026-01-03)

**Problem:** When loading classification data with task_type="auto", PLS (regression model) remained checked and tier switched to "Custom" instead of properly updating to classification models.

### Fix 1: Move `_on_task_type_changed()` Outside Conditional
**File:** `spectral_predict_gui_optimized.py` line 10541-10544

Moved `_on_task_type_changed()` OUTSIDE the conditional block so it ALWAYS runs after data loads.

### Fix 2: Add `_updating_from_tier` Flag Protection
**File:** `spectral_predict_gui_optimized.py` lines 8873-8894

**Root Cause:** `_on_task_type_changed()` was modifying checkboxes without the `_updating_from_tier` flag. This triggered `_on_model_checkbox_changed()` which switched tier to "Custom" BEFORE `_on_tier_changed()` could run.

**Fix:** Added `self._updating_from_tier = True` before the checkbox loop and `self._updating_from_tier = False` after, matching the pattern in `_on_tier_changed()`.

**Verification:** Classification tier defaults correctly exclude PLS:
- Quick: `['PLS-DA', 'LightGBM', 'RandomForest']`
- Standard: `['PLS-DA', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost']`
- Comprehensive: `['PLS-DA', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost', 'SVM', 'MLP', 'NeuralBoosted']`

---

## 🔗 THREE-TIER SPECTRAL OPTIMIZATION (2026-01-03 Session Update)

**PLAN:** `C:\Users\sponheim\.claude\plans\wild-puzzling-lemon.md`

**STATUS: Coupled Working, Ensemble Marginal, Learned Broken**

### Overview
Three optimization modules for spectral modeling - ADDITIVE to existing search.py, bayesian_config.py, nsga2_search.py.

### Modules Status

| Module | File | Status | Notes |
|--------|------|--------|-------|
| **Coupled Search** | `coupled_search.py` | ✅ Working | Optuna joint preprocessing+model optimization |
| **Ensemble Preprocessing** | `ensemble_preprocessing.py` | ⚠️ Marginal | Slightly worse than Grid Search (~1-2% lower R²) |
| **Learned Preprocessing** | `learned_preprocessing.py` | ❌ Broken | Produces terrible models - should be disabled |

---

### Coupled Search (2026-01-03 Updates)

**What it does:**
- Jointly optimizes preprocessing + model hyperparameters using Optuna
- Supports ALL models: PLS, Ridge, Lasso, ElasticNet, RF, XGBoost, LightGBM, CatBoost, SVR, MLP
- Wide hyperparameter search ranges (intentionally broader than Grid Search)

**Preprocessing options:**
- 14 types: raw, snv, deriv1-4, snv_deriv1-4, deriv1-4_snv
- Baseline correction: ALS, polynomial, airPLS (only when NOT using derivatives)
- Window size: 5-51 for Savitzky-Golay

**Fix applied (2026-01-03):** Baseline correction now mutually exclusive with derivatives
- **Rationale:** Derivatives inherently remove polynomial baselines; applying baseline BEFORE derivatives is redundant and can distort signal
- **Location:** `src/spectral_predict/coupled_search.py` lines 62-91
- **Logic:** If `'deriv' in preprocess_type`, skip baseline suggestion entirely

**Model hyperparameter ranges** (intentionally wide for exploration):
- PLS: n_components 2-20
- Ridge: alpha 1e-4 to 100 (log scale)
- XGBoost/LightGBM: 7-8 hyperparameters each with wide ranges
- These ranges are correct - Grid Search values are narrower but not "better"

---

### Ensemble Preprocessing (2026-01-03 Updates)

**What it does:**
- Stacks 8 preprocessing methods (raw, snv, deriv1, deriv2, snv_deriv1, snv_deriv2, als, als_snv)
- Trains same base model on each preprocessed version
- Combines predictions using RidgeCV meta-model

**Fixes applied:**
1. **Empty params bug:** Was calling `build_model(model_name, {})` → sklearn defaults (PLS n_components=2)
2. **Mini grid search added:** Now tunes base model hyperparameters BEFORE building ensemble
   - Location: `spectral_predict_gui_optimized.py` lines 15993-16071
   - Tests: PLS n_components 2-15, Ridge alpha [0.01-100], etc.

**Current status:**
- Works but slightly underperforms Grid Search (~1-2% lower R²)
- Value proposition unclear - theoretically more robust but not proven
- User wants to test generalization on held-out data to validate

---

### Learned Preprocessing (PyTorch) - BROKEN

**Status:** ❌ NOT FUNCTIONAL - Should be disabled

**Problems:**
1. Replaces BOTH preprocessing AND model with neural network (ignores user's selected model)
2. sklearn's `cross_val_score` doesn't pass `epochs` to `fit()` - minimal training
3. Neural networks need 1000+ samples; spectral data typically has 50-500
4. Fixed architecture not tuned for spectral data

**Recommendation:** Disable the checkbox or remove feature entirely. PLS/Ridge were designed for "n << p" problems.

---

### GUI Changes (2026-01-03)

**Files modified:** `spectral_predict_gui_optimized.py`

1. **Checkbox disable logic (lines 13277-13298):**
   - `_update_coupled_options_state()` method
   - Disables Ensemble/Learned checkboxes when non-Coupled mode selected
   - Trace on `optimization_method` StringVar

2. **PyTorch check fix (lines 115-116):**
   - Now imports `PYTORCH_AVAILABLE` from module
   - `HAS_LEARNED_PREPROCESSING = PYTORCH_AVAILABLE`
   - PyTorch installed in `.venv312` (was missing)

3. **Ensemble hyperparameter tuning (lines 15993-16071):**
   - Mini grid search before ensemble creation
   - Tests key hyperparameters for each model type

---

### Known Issues / TODO for Next Agent

1. **State/Thread Bug:** After running Coupled then Grid Search, "Optimizing Ridge" messages may appear. Restart fixes. Root cause unknown.

2. **Deep Learning Model:** ~~Decide whether to disable~~ **DISABLED** (2026-01-03) - Checkbox disabled until debugged:
   - **Symptom:** Produces very poor models compared to other methods
   - **Possible cause:** sklearn's `cross_val_score` may not pass `epochs` to `fit()` correctly
   - **TODO:** Debug why results are poor - root cause unknown

3. **Ensemble value proposition:** User wants to test if ensemble actually generalizes better on held-out data (not just CV)

4. **Coupled - multiple models issue:** User reported only PLS ran when PLS+Ridge selected. May be silent exception in Ridge. Needs investigation.

5. **NEW (2026-01-03): Coupled R² mismatch in Model Development:**
   - When loading Coupled result into Model Development tab, R² doesn't match
   - Suggests preprocessing, variables, or variable order NOT preserved correctly
   - Similar to previous R² mismatch bug (fixed for Grid Search) but specific to Coupled
   - **Investigation needed:** Check how Coupled results are serialized and reconstructed in `_populate_from_result()`

### Key Files for Context
- `src/spectral_predict/coupled_search.py` - Joint Optuna optimization
- `src/spectral_predict/ensemble_preprocessing.py` - Stacked preprocessing
- `src/spectral_predict/learned_preprocessing.py` - PyTorch (broken)
- `spectral_predict_gui_optimized.py` - GUI integration (lines 15870-16165 for Coupled mode)

---


## ✅ UNIFIED BAYESIAN OPTIMIZATION (2026-01-05) - COMPLETE

**STATUS: FULLY WORKING - Regression and Classification both confirmed working**

A completely new "Unified Bayesian" optimization method that performs TRUE joint optimization of preprocessing + hyperparameters + variable selection + subset size together.

### Regression - COMPLETE ✅

All regression issues fixed (see commits below).

### Classification - COMPLETE ✅ (2026-01-04)

**Problem (FIXED):** Unified Bayesian classification was failing with multiple errors:
- `could not convert string to float: 'High'` - string labels not encoded
- `KeyError: 'RMSE'` - GUI hardcoded for regression columns
- `KeyError: 'Accuracy'` - missing CV Error column for GUI

**Fixes Applied:**

1. **unified_bayesian.py:**
   - Added `task_type` param to `compute_importances()` (was hardcoded to 'regression')
   - Added LabelEncoder for string class labels (e.g., "High", "Low", "Medium")
   - Wrap PLS-DA with LogisticRegression pipeline for classification
   - Added `CV Error = 1 - Accuracy` column for GUI compatibility
   - Added `Score` column for GUI compatibility
   - Handle empty DataFrame gracefully

2. **spectral_predict_gui_optimized.py:**
   - Line 16319-16322: Task-type aware logging (RMSE vs Accuracy)

3. **report.py:**
   - Handle empty results DataFrame

**Testing:**
- ✅ Standalone test with synthetic data PASSED
- ✅ String labels work correctly
- ✅ GUI integration confirmed working by user (2026-01-04)

### Progress Feedback - COMPLETE ✅ (2026-01-05)

**Problem:** Unified Bayesian showed no progress in Progress tab until completion (unlike Grid Search).

**Fixes Applied:**
1. Connected `progress_callback` to `run_unified_bayesian()` call (line 16347)
2. Added `best_model` tracking to progress_wrapper for "Best Model So Far" display
3. Fixed preprocessing param key: `'preprocess_type'` → `'preprocessing'`
4. Added `unified_progress_wrapper` to track best across ALL models (not just current)

**Progress tab now shows:**
- Trial-by-trial updates: `"PLS: Trial 1/50 - RMSE: 0.0523"`
- Time estimation (elapsed/remaining)
- "Best Model So Far" with correct preprocessing and overall best across models

**Commits:**
- `0fb179f` - feat: Add progress feedback to Unified Bayesian optimization
- `60aa6ea` - fix: Unified Bayesian progress shows correct preprocessing and overall best

### Confusion Matrix Class Names - FIXED ✅ (2026-01-05)

**Problem:** Confusion matrix in Model Development showed numbers (0, 1, 2) instead of class names ("High", "Low", "Medium").

**Root Cause:** Code checked `self.label_encoder` (from Results search) but Model Development stores its encoder in `self.refined_label_encoder`.

**Fix:** Updated 4 locations to check `refined_label_encoder` first, fallback to `label_encoder`:
- Confusion matrix plot (line 19764)
- ROC curve plot (line 20062)
- Annotation tooltip Y value (line 11324)
- Annotation tooltip predicted value (line 11346)

**Commit:** `b104199` - fix: Show class names instead of numbers in confusion matrix

### LVs Integer Display - FIXED ✅ (2026-01-05)

**Problem:** LVs column showed decimals (e.g., "5.0") instead of integers ("5").

**Root Cause:** `np.nan` for non-PLS models caused pandas to convert column to float.

**Fix:** Use `int()` conversion and `None` instead of `np.nan` in both:
- `unified_bayesian.py` line 972
- `search.py` line 3149

**Commit:** `bc716ea` - fix: Display LVs as integers instead of decimals in results

---

### Previous Regression Fixes (COMPLETE)

#### 1. Wavelength Ordering - FIXED (Commit 7ccdfd4)
**Problem:** Wavelengths were stored in spectral order but training used importance order.
**Fix:** Removed `np.sort(top_indices)` - wavelengths now stored in training order.
**File:** `src/spectral_predict/unified_bayesian.py` lines 625-638

#### 2. Window Size Bug - FIXED (Commit 86d0ddb)
**Problem:** Model Development used hardcoded window=11 instead of reading from config.
**Root Cause:** `_parse_coupled_preprocessing()` had hardcoded `'window': 11` default.
**Fix:** Added `window_config` parameter to pass Window from config.
**File:** `spectral_predict_gui_optimized.py` lines 20990, 21002-21010, 21105-21109

**Test Result:**
```
Original:          Window=31, R²=0.913829
Model Development: Window=31, R²=0.913829
Difference:        0.000000 ✅
```

#### 3. Previously Fixed (Still Working)
- `Poly` column - required by report.py
- `ROC_AUC` placeholder for classification tasks
- `all_vars` column - ALL wavelengths for model reconstruction
- Report generation - no longer crashes

### Commits
```
86d0ddb - fix: Read Window from config for Unified Bayesian Model Development
7ccdfd4 - fix: Store wavelengths in training order (removed np.sort)
a757086 - revert: Remove preprocessing name stripping that broke Model Development
361f8aa - fix: Sort wavelengths only for storage, not during training
e6b631a - fix: Match grid search preprocessing name format for Model Development
08a3c30 - fix: Add missing columns and spectral order for Model Development
```

### Test Script
Regression test at `scripts/test_unified_bayesian_model_dev.py` verifies R² matches for all window sizes.

### Technical Note: Why Unified Bayesian Uses "Coupled" Code Path
Unified Bayesian uses preprocessing names like `'deriv1'`, `'snv_deriv2'` (with numbers).
The fallback regex at line 21090 (`deriv\d`) matches these, causing it to be detected as "coupled".
The Window Size fix compensates by reading Window from the config's Window column.
Grid Search uses names like `'deriv'`, `'snv_deriv'` (no numbers) and goes through a different path.

---


## ~~🚨 TOP PRIORITY: BAYESIAN PLS UNDERPERFORMS~~ (2025-12-30) - SUPERSEDED

**STATUS: SUPERSEDED by Unified Bayesian (above)**

The original Bayesian optimization issues are now resolved by the new Unified Bayesian approach which performs true joint optimization.

- **Bayesian Ridge:** R² > 0.97 (BETTER than grid search) ✓
- **Bayesian PLS:** R² 0.90-0.95 (worse than grid search ~0.96) ✗

**Secondary issue:** R2 concordance between Results tab and Model Development tab - **FIXED (2026-01-01)** - See "R² Mismatch Fix" section below

---

## ✅ CARS-TREE IMPLEMENTATION (2026-01-01) - COMPLETE

**Problem:** CARS variable selection didn't work for tree models (RandomForest, XGBoost, LightGBM). The existing "Model-Aware" checkbox was confusing and `cars-aware` mode underperformed native TopN selection.

**Root Cause:**
1. **Bug:** LightGBM `feature_importances_` are sparse (many zeros). After iteration 1, probability weights became zero, causing `rng.choice()` to fail with "Fewer non-zero entries in p than size".
2. **Design:** CARS was designed for PLS (Li et al., 2009). Tree importance is context-dependent and sparse, not suitable for original CARS algorithm.

**Solution - CARS-Tree (Novel Method):**
- Hybrid importance blending split-based (frequency) + gain-based (quality) importance
- Enhanced LightGBM config: 100 trees, unlimited depth, regularization, subsampling
- Produces denser importance distributions suitable for CARS reweighting

**Commits:**
- `e9b0584` - fix: Add minimum weight floor for tree model CARS (fixes crash)
- `c865e70` - feat: Implement CARS-Tree variable selection for tree models
- `e400eb6` - feat: Add CARS-Tree as separate GUI option

**GUI Changes:**
- Replaced confusing "Model-Aware" checkbox with two distinct options:
  - **CARS (PLS-based)** - Best for linear models (PLS, Ridge, ElasticNet)
  - **CARS-Tree (Hybrid Importance)** - Best for tree models (LightGBM, RF, XGBoost)

**Test Results (synthetic data):**
| Method | Variables Selected | RMSECV |
|--------|-------------------|--------|
| CARS-aware | 52 | 4.6738 |
| CARS-Tree | 64 | 4.4926 |

**No regression:** PLS R²=0.593185, Ridge R²=0.356646 (unchanged from baseline)

**Files Modified:**
- `src/spectral_predict/variable_selection.py` - Added `use_hybrid_importance`, `hybrid_importance_weight` params
- `src/spectral_predict/search.py` - Added 'cars-tree' to method list and handler
- `spectral_predict_gui_optimized.py` - Added CARS-Tree checkbox, removed Model-Aware checkbox
- `scripts/diagnose_cars_tree.py` - Diagnostic script for testing all CARS variants

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

### Bayesian Optimization - FIXED (2026-01-02)

**STATUS: FIXED - Multiple Issues Resolved**

**Original problem:**
- **Bayesian Ridge: R² > 0.97** (BETTER than grid search) ✓
- **Bayesian PLS: R² 0.90-0.95** (WORSE than grid search which gets ~0.96) ✗
- **Current state (pre-fix):** PLS models don't appear until rank 1000+ in Bayesian results

#### Fix 1: Clean Hyperparameter Signal (2026-01-02 morning)

**ROOT CAUSE:** The objective function was returning "best of all subsets" score to TPE, creating noisy learning signals.

**FIX:** Changed `create_objective_function()` in `bayesian_utils.py` to return `full_model_rmse` to TPE (not `best_subset_rmse`).

**Result:** Ridge now works great. But PLS still failed - not appearing in top 664 results even with 100 trials.

#### Fix 2: PLS n_components Categorical (2026-01-02 afternoon)

**ROOT CAUSE:** PLS still failed because `suggest_int` for n_components allows TPE to sample scattered values, while grid search tests ALL values (2, 3, 4, ..., 12) exhaustively. n_components is discrete and non-monotonic - different values work best for different variable subsets, confusing TPE.

**FIX:** Changed `_get_pls_space()` in `bayesian_config.py`:
```python
# FROM: suggest_int (TPE samples scattered values)
n_components = trial.suggest_int('n_components', min_components, max_components)

# TO: suggest_categorical (guarantees ALL values tested equally like grid search)
component_choices = list(range(min_components, max_components + 1))
n_components = trial.suggest_categorical('n_components', component_choices)
```

#### Fix 3: GUI-Controlled n_trials (2026-01-02 afternoon)

- Removed hardcoded default from `run_bayesian_search()` - now requires `n_trials` parameter
- GUI default changed from 50 to 100 in `spectral_predict_gui_optimized.py`
- Backend raises `ValueError` if `n_trials` not provided (user controls via GUI)

#### Fix 4: All Model Search Spaces Optimized (2026-01-02)

Fixed wasteful convergence/speed parameters across ALL models:

| Model | Wasteful Params Fixed |
|-------|----------------------|
| PLS | `max_iter=500`, `tol=1e-6` (fixed), n_components → categorical |
| Ridge | `solver='auto'`, `tol=1e-6`, `max_iter=10000` (fixed) |
| Lasso | `selection='random'`, `tol=1e-6`, `max_iter=10000` (fixed) |
| ElasticNet | `tol=1e-6`, `max_iter=10000` (fixed) |
| RandomForest | `min_impurity_decrease=0.0`, `ccp_alpha=0.0` (fixed) |
| SVR | `shrinking=True`, `tol=1e-4`, `cache_size=500`, `max_iter=10000` (fixed) |
| MLP | `max_iter=1000`, `tol=1e-4` (fixed) |
| LightGBM | Added `num_leaves < 2^max_depth` constraint |

#### Fix 5: Syntax Errors in nsga2_search.py (2026-01-02)

Fixed unterminated f-string literals at lines 1316, 1339, 1349 (literal newlines → `\n` escapes).

**Files modified:**
- `src/spectral_predict/bayesian_config.py` - All model search spaces, PLS categorical
- `src/spectral_predict/bayesian_utils.py` - Objective function return value
- `src/spectral_predict/search.py` - Removed hardcoded n_trials default
- `src/spectral_predict/nsga2_search.py` - Fixed f-string syntax errors
- `spectral_predict_gui_optimized.py` - GUI default n_trials=100

**Expected outcome:**
- PLS: Should now appear in top results (categorical ensures all n_components tested)
- Ridge: R² > 0.97 (Bayesian) ≥ Grid ✓ (verified working)
- All models: Focus trials on quality parameters, not convergence params

**Testing needed:**
User should re-test Bayesian PLS with the categorical n_components fix.

**Previous fixes:**
- (2025-12-22) Double edge masking was fixed (commit d22a528)

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

### NSGA-II - BROKEN (ROOT CAUSE FOUND 2026-01-01)

**STATUS: DOES NOT WORK** - Gives much worse RMSE than Grid Search for regression.

**ROOT CAUSE: Missing Hyperparameters in `_build_model()`**

The previous "data leakage" diagnosis was **WRONG**. The real issue is NSGA-II's `_build_model()` function is missing critical hyperparameters that Grid Search uses.

**CONCRETE DIFFERENCES FOUND:**

#### LightGBM - MISSING 5 CRITICAL PARAMETERS
```python
# Grid Search (models.py line 186-201) uses:
LGBMRegressor(
    min_child_samples=5,      # MISSING in NSGA-II
    subsample=0.8,            # MISSING in NSGA-II
    bagging_freq=1,           # MISSING in NSGA-II
    colsample_bytree=0.8,     # MISSING in NSGA-II
    reg_alpha=0.1,            # MISSING in NSGA-II
    reg_lambda=1.0,           # NSGA-II uses 0.1 (wrong)
)

# NSGA-II (nsga2_search.py line 357-360) uses:
LGBMRegressor(
    # Missing all regularization parameters!
    reg_lambda=0.1,  # Wrong default (should be 1.0)
)
```

#### XGBoost - DIFFERENT DEFAULTS
- Grid: `max_depth=6` (fixed) vs NSGA-II: `3-7` (varies by gene)
- Grid: `subsample=0.8` (fixed) vs NSGA-II: `1.0 or 0.8` (varies)
- Grid: `colsample_bytree=0.8` vs NSGA-II: varies

#### RandomForest
- Grid: `n_jobs=-1` (all cores)
- NSGA-II: `n_jobs=1` (intentional for GA parallelism, acceptable)

**WHY THIS CAUSES POOR RESULTS:**
Missing regularization parameters (subsample, colsample_bytree, min_child_samples) cause overfitting. NSGA-II explores hyperparameter regions that Grid Search's proven defaults avoid.

**FIX OPTIONS:**

1. **Quick Fix** - Add missing parameters to `_build_model()` to match Grid Search defaults
2. **Better Fix** - Replace `_build_model()` with calls to `get_model()` from models.py
3. **Best Fix** - Constrain gene search space around proven defaults

**RECOMMENDED: Option 1 (Quick Fix)**

Add these to LightGBM in `_build_model()` (line 357-360):
```python
LGBMRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    num_leaves=num_leaves,
    min_child_samples=5,       # ADD
    subsample=0.8,             # ADD
    bagging_freq=1,            # ADD
    colsample_bytree=0.8,      # ADD
    reg_alpha=0.1,             # ADD
    reg_lambda=reg_lambda,     # Keep gene-controlled
    random_state=random_state,
    n_jobs=1,
    verbose=-1
)
```

**FILES:** `src/spectral_predict/nsga2_search.py` - `_build_model()` function (lines 289-430)

### INVESTIGATION: spectral-predict has working NSGA-II (2026-01-01)

**Reference:** `C:\Users\sponheim\git\spectral-predict\spectral_predict\core\nsga2_search.py`

**KEY DIFFERENCES FOUND:**

| Feature | spectral-predict (V3) | dasp (BROKEN) |
|---------|----------------------|---------------|
| Hyperparameter genes | **4 genes** | 13 genes |
| Search space | Small, focused | Massive, diffuse |
| Model building | Simple + `get_model()` | Complex `_build_model()` |
| Initial sampling | `IntegerRandomSampling()` | `SeededWavelengthSampling` |
| RMSE formula | `sqrt(mean(MSE))` | `mean(sqrt(MSE))` |
| Penalty value | 1e6 | 1e10 |
| Complexity floors | 0.15-0.25 base | 0.3-0.5 base |

**ROOT CAUSE HYPOTHESIS:**

1. **Search Space Explosion** - dasp has 13 hyperparameter genes (lr, reg_alpha, reg_lambda, l1, subsample, colsample, min_samples, gamma, max_features) vs V3's 4 genes. This creates a search space ~10^9 times larger, making convergence nearly impossible.

2. **Over-engineering** - V3 keeps it simple: preprocessing + model_type + model_param + wavelengths. dasp tries to optimize EVERY hyperparameter simultaneously.

3. **Complexity Penalty** - Higher complexity floors (0.3-0.5) in dasp may over-penalize tree models, preventing good solutions from appearing in Pareto front.

**FIX RECOMMENDATION:**

**Option A (Quick): Revert to V3 approach**
- Remove genes 4-12 (hyperparameter genes)
- Use fixed/default hyperparameters from `get_model()`
- Reduce chromosome to: [preproc, window, model, param, wavelengths...]

**Option B (Staged optimization)**
- Phase 1: Optimize preprocessing + wavelengths (small search space)
- Phase 2: Fine-tune hyperparameters on best solutions from Phase 1

### NSGA-II vs Grid Search - Real Value Proposition (2026-01-01)

**Grid Search preprocessing coverage:**
- Tests ~4-8 preprocessing combos (raw, SNV, deriv1_w11, deriv2_w11)
- Fixed window sizes per derivative
- Sequential: finds best preprocessing THEN does variable selection

**NSGA-II (V3-style) preprocessing coverage:**
- 10-14 preprocessing types × 15 window sizes = **150+ preprocessing combos**
- Tests w5, w7, w9, ..., w33 for each derivative
- Tests SNV+deriv orderings (snv_deriv1 vs deriv1_snv - different results!)
- Simultaneous: optimizes preprocessing AND wavelengths together

**Why NSGA-II should outperform Grid Search (even Option A):**
1. **20× larger preprocessing search space** - Finds optimal window size, not fixed w11
2. **Finds optimal derivative order** - Including deriv3, deriv4 that Grid Search skips
3. **Simultaneous optimization** - May find preprocessing+wavelength combos that sequential search misses

**Why Option B would be even better:**
- Option A uses fixed hyperparameters from `get_model()` - same as Grid Search
- Option B adds hyperparameter tuning in Phase 2, potentially finding better LightGBM/XGBoost configs
- This is the only way NSGA-II can beat Grid Search on BOTH preprocessing AND hyperparameters

**Why current dasp NSGA-II fails:**
- 13-gene search space is so massive it can't even explore preprocessing space effectively
- Tries to optimize everything at once → converges to nothing

**RECOMMENDED FIX ORDER:**
1. First: Implement Option A (V3-style, 4 genes) - proves NSGA-II works, beats Grid Search on preprocessing
2. Later: Add Option B (staged) - adds hyperparameter tuning for additional gains

### ❌ GA Preprocessing Running Too Fast - FIX ATTEMPT BROKE GA COMPLETELY (2026-01-05)

**Original Problem:** GA preprocessing ran in < 1 minute and produced worse models than Grid Search. It should run AT LEAST as much as Grid Search.

**Fix Attempted:**
1. Modified `ga_preprocessing.py` to return top 5 results per model group (not just best 1)
2. Modified `search.py` to use ALL top-N configs from each GA result
3. Added deduplication to prevent testing the same preprocessing twice

**Result: FIX BROKE GA COMPLETELY**
- GA now produces NO results at all
- RMSECV=inf throughout entire run
- See "CRITICAL: GA Preprocessing COMPLETELY BROKEN" section above

**Files Modified (WITH UNCOMMITTED CHANGES):**
- `src/spectral_predict/ga_preprocessing.py` - Added `configs` list with `deriv`, `window`, `polyorder` fields
- `src/spectral_predict/search.py` - Loop through `configs` list, deduplicate by preprocessing key

**TODO:** Either revert these changes or debug why fitness returns -inf

---

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

---

## ✅ REPRODUCIBILITY MODE (2026-01-05) - REMOVED

**STATUS: REMOVED** - Reproducibility toggle deleted from GUI; feature no longer needed.

### History
**Original Problem:** Different results between program restarts.

**Root Cause Found (Commit fe152f1):** Python's hash randomization (PYTHONHASHSEED) caused set iteration order to vary between program restarts during fuzzy filename matching in `io.py`. This changed the row order of aligned data, causing different CV fold assignments even with the same seed.

**Fix Applied:** `io.py:277` - Sort set iteration in fuzzy matching: `for norm_id in sorted(common_norm_ids)`

### Analysis After Fix (Commit be1f18e)
With io.py fixed, we tested whether BLAS threading still affected results:

**Test:** 1000 samples × 2500 wavelengths (PLS, Ridge, LightGBM)
**Result:** All differences = 0.00e+00 (IDENTICAL regardless of threading)

**Conclusion:** Reproducibility mode provided no additional value. The io.py fix was the only thing needed.

### Changes Made (Commit be1f18e)
- Removed reproducibility toggle from GUI (Settings tab)
- Removed `reproducible` parameter from `run_search()`
- Deleted `src/spectral_predict/reproducibility.py` module
- Deleted `examples/reproducibility_demo.py`
- Deleted `tests/test_reproducibility.py`
- Added `src/spectral_predict/constants.py` with `RANDOM_STATE = 42`
- Added `scripts/test_threading_impact.py` to verify threading has no effect

**Note:** `random_state=42` is still used internally throughout the codebase for CV splits and model initialization. Users simply no longer need to manage a toggle.

---

## ✅ GA PREPROCESSING OPTIMIZATION SIMPLIFICATION (2026-01-05) - COMPLETE

**What was done:**

### 1. Simplified GA from 6 genes to 2 genes

**Removed (redundant with derivatives):**
- Baseline method (none, polynomial, ALS, airPLS)
- Baseline lambda (1e2 to 1e9)
- Smoothing enabled (True/False)
- Smoothing window (5-21)

**Kept:**
- Preprocessing type (14 options: raw, snv, deriv1-4, snv_deriv1-4, deriv1-4_snv)
- S-G window size (17 options: 5-51 odd values)

**Search space reduction:** 136,416 → 238 combinations (99.8% reduction)

**Why redundant:**
- Derivatives mathematically remove baselines (1st removes constant, 2nd removes linear, etc.)
- Savitzky-Golay derivatives ARE smoothing operations - additional smoothing is redundant

### 2. Added Exhaustive Search Option

New `method` parameter in `optimize_preprocessing()`:
- `method='ga'` - Genetic algorithm (default, uses population/generations)
- `method='exhaustive'` - Tests all 238 combinations in parallel (guaranteed optimal)

Parallel evaluation via joblib when `n_jobs=-1`.

### 3. Fixed Ensemble Preprocessing

- Added multiple window sizes (w11, w17, w25) for derivatives
- Changed default `include_baseline=False` (redundant with derivatives)
- Added `compact=True` option for faster training
- Expanded RidgeCV alpha range: 5 values → 17 (logspace -4 to 4)
- Increased meta-model CV folds: 3 → 5

### 4. Fixed Learned Preprocessing

- Added early stopping with `patience=10`, `min_delta=1e-4`
- Saves and restores best model weights
- Added standalone `LearnedPreprocessor` class (sklearn-compatible transformer)
  - Can be used with any sklearn model (PLS, RF, etc.)
  - `preprocessor.fit(X_train, y_train)` → `preprocessor.transform(X)`

### 5. GUI Changes

- Added "Search Method" dropdown: `ga` or `exhaustive`
- Updated defaults: population=48, generations=30
- Updated labels to reflect simplified search space (238 combinations)
- Labels note that optimization runs separately for each model group

### Files Modified:
- `src/spectral_predict/ga_preprocessing.py` - Complete rewrite with 2 genes
- `src/spectral_predict/ensemble_preprocessing.py` - Multiple windows, better RidgeCV
- `src/spectral_predict/learned_preprocessing.py` - Early stopping, LearnedPreprocessor class
- `src/spectral_predict/search.py` - Added `ga_preprocess_method` parameter
- `spectral_predict_gui_optimized.py` - Method dropdown, updated defaults/labels

---

## 🚨 OUTSTANDING: Validation ROC_AUC Still Shows NaN (2026-01-06)

**Problem:** Validation ROC_AUC shows NaN for classification, even though other validation metrics (Accuracy, F1, Precision, Recall) work correctly.

**Status:** PARTIALLY INVESTIGATED - needs more debugging

### What Was Fixed (2026-01-06)

1. **CV Metrics Ranking** - Changed from ROC_AUC to Accuracy (primary) + F1 (tiebreaker)
   - File: `src/spectral_predict/scoring.py` lines 83-100

2. **CV Metrics Averaging** - Changed from `'weighted'` to `'macro'` for multiclass
   - File: `src/spectral_predict/search.py` line 2812

3. **Validation Metrics Averaging** - Changed from `'weighted'` to `'macro'` for multiclass
   - File: `src/spectral_predict/search.py` line 471

4. **CV ROC_AUC label_binarize bug** - Removed incorrect `label_binarize` usage
   - File: `src/spectral_predict/search.py` lines 2840-2845

5. **CV ROC_AUC class alignment** - Added `labels=pipe_clone.classes_` parameter
   - File: `src/spectral_predict/search.py` lines 2835-2838

6. **Validation ROC_AUC simplification** - Removed `labels` parameter that was causing issues
   - File: `src/spectral_predict/search.py` lines 513-521

### What Still Doesn't Work

**Validation ROC_AUC** - Still returns NaN despite:
- Other validation metrics (Accuracy, F1, Precision, Recall) working correctly
- Model having `predict_proba` method
- Simplified code without `labels` parameter

### TODO for Next Session

1. Add debug print statements to see actual error in exception handler (line 522-523)
2. Check if `y_proba` shape matches expected dimensions
3. Check if `y_val` contains valid encoded labels (0, 1, 2...) not strings
4. Check if model is a Pipeline vs raw model (Pipeline needs different handling)
5. Consider if multiclass `roc_auc_score` has edge case issues with validation data

### Files Modified (2026-01-06)
- `src/spectral_predict/scoring.py` - Ranking by Accuracy+F1
- `src/spectral_predict/search.py` - Multiple ROC_AUC and averaging fixes

---

## 🚨 CRITICAL: GA Preprocessing COMPLETELY BROKEN (2026-01-05)

**STATUS: NON-FUNCTIONAL** - GA preprocessing produces NO RESULTS at all.

### Symptoms
- Selecting "Genetic Algorithm" method produces ZERO results
- RMSECV=inf throughout entire run
- Finishes in ~10 seconds with empty output

### Root Cause (Suspected)
Recent changes to `ga_preprocessing.py` and `search.py` have broken the GA preprocessing flow. The exact breaking change is unknown.

### Files With Uncommitted Changes
- `src/spectral_predict/ga_preprocessing.py`
- `src/spectral_predict/search.py`

### TODO to Fix
1. **Option A:** Revert changes with `git restore src/spectral_predict/ga_preprocessing.py src/spectral_predict/search.py`
2. **Option B:** Debug what's causing fitness evaluations to return -inf

### Secondary Issue: Only 2nd Derivative Appears
User reports only ever seeing 2nd derivative in GA genes - never 1st, 3rd, or 4th. Code shows all derivatives ARE in PREPROC_TYPES, but may not be evaluated or may all return worse fitness than deriv2.

---

## 🚨 OUTSTANDING: GA/Exhaustive Preprocessing Issues (2026-01-05)

### ❌ Issue 1: Exhaustive Search Performance - COMPREHENSIVE FIX FAILED (2026-01-05)

**Original Problem:** User reports exhaustive search (guaranteed optimal) does not clearly outperform default grid search preprocessing.

**Fix Attempt #2 (FAILED 2026-01-05):**

Implemented comprehensive 5-phase fix:
1. Reduced PREPROC_TYPES from 14 to 10 (removed snv_deriv3/4, deriv3/4_snv)
2. Modified `optimize_preprocessing()` to return top 5 configs instead of just 1
3. Added `_evaluate_with_actual_model()` to use real model with user hyperparameters
4. Changed GA to run per-model instead of per-model-group
5. Created multiple preprocessing configs per model

**Result: DOES NOT WORK**
- Produces only **3 models** (should produce many more)
- Models have **awful performance** (worse than before)
- Uncommitted changes remain in `ga_preprocessing.py` and `search.py`

**Files created (can be deleted):**
- `scripts/verify_grid_search_unchanged.py`
- `COUPLED_OPTIMIZATION_FIX_SUMMARY.md`

**TODO for next session:**
1. Revert changes: `git restore src/spectral_predict/ga_preprocessing.py src/spectral_predict/search.py`
2. Investigate why only 3 models produced - debug actual execution flow
3. Investigate awful performance - check if fitness evaluation working correctly
4. Consider simpler approach - fix fitness model mismatch first without per-model architecture

**Root cause hypothesis:** The per-model architecture change may have broken something fundamental in how preprocessing configs map to models.

**Original possible causes (still valid):**
1. Grid search defaults are actually near-optimal for most data
2. Fitness model (PLS/LightGBM/MLP) may not correlate with actual model performance
3. Window sizes included (5-51 odd) may not include Grid Search defaults

### ✅ Issue 2 & 3: R² Mismatch / Edge Masking for GA Preprocessing - FIXED (2026-01-05)

**Problem:** When using GA or Exhaustive preprocessing, R² in Model Development tab did NOT match R² in Results tab. Edge masking was not being applied for GA preprocessing.

**Root Cause:**
- GA preprocessing stored `deriv=None` and `window=None` in results
- Edge masking check (`if preprocess_cfg.get("deriv") and preprocess_cfg.get("window")`) FAILED
- Edge wavelengths were NOT excluded for GA preprocessing (but WERE excluded for Grid Search)

**Fix Applied:**
- Added `_decode_ga_genes()` helper function in `search.py` (lines 1132-1160)
- Decodes `ga_genes` array to extract actual `deriv` order (1-4) and `window` size (5-51)
- Updated all 4 GA config creations (pls, neural_svm, tree, neuralboosted) to store actual values
- Existing edge masking code now works automatically for GA preprocessing

**Files Modified:**
- `src/spectral_predict/search.py` - Lines 1132-1240 (GA config creation)

### ❌ Issue 4: GA Fitness Hyperparameters - FIX REVERTED (2026-01-05)

**Problem:** GA/Exhaustive fitness evaluation uses hardcoded simplified hyperparameters that don't match user's actual configuration.

| Model | GA Fitness Uses (WRONG) | Should Use |
|-------|------------------------|-----------|
| LightGBM | `max_depth=5` hardcoded | User's configured values |
| MLP | `hidden_layer_sizes=(100, 50)` hardcoded | User's configured values |
| NeuralBoosted | sklearn defaults | User's configured values |
| Ridge/Lasso/ElasticNet | Uses PLS proxy | Should use their own fitness |
| RandomForest/XGBoost/CatBoost | Uses LightGBM proxy | Should use their own fitness |
| SVR | Uses MLP proxy | Should use its own fitness |

**Failed Fix Attempt (Commits 210a71f, 295306d - REVERTED):**

Attempted to add `model_config` parameter to GA fitness functions and extract user hyperparameters in search.py. This caused Grid Search to run far fewer configurations when GA preprocessing was enabled.

**Revert Commit:** 4dfaba1

**Why It Broke:**
Unknown - need to investigate. The changes only added parameters to GA functions and extracted config dicts inside the `if ga_preprocess:` block. Should not have affected Grid Search hyperparameter grids.

**TODO - Correct Implementation:**

1. **Investigate root cause** - Why did adding model_config to GA functions affect Grid Search?
2. **Per-model fitness functions** - Add individual fitness for each model type:
   - `_evaluate_ridge()` - uses user's Ridge alpha
   - `_evaluate_lasso()` - uses user's Lasso alpha
   - `_evaluate_elasticnet()` - uses user's alpha/l1_ratio
   - `_evaluate_randomforest()` - uses user's RF config
   - `_evaluate_xgboost()` - uses user's XGBoost config
   - `_evaluate_catboost()` - uses user's CatBoost config
   - `_evaluate_svr()` - uses user's SVR config
3. **Run GA per model** - Not per model group (since users typically run 3-4 models)
4. **Pass user hyperparameters** - Extract first value from each user list

---

### Issue 5: Ensemble Preprocessing Still Underperforms Grid Search

**Problem:** Even after expanding to 24 preprocessings, ensemble stacking still produces worse models than simple Grid Search.

**Current ensemble setup:**
- 24 preprocessings: raw, SNV, deriv1-3 with multiple windows (7,11,17,25), SNV+deriv combos, deriv+SNV combos, baseline variants
- RidgeCV meta-model with wide alpha range (1e-4 to 1e4)
- 5-fold CV for meta-model
- Hyperparameter tuning for base model before stacking

**Possible fundamental issues:**
1. **Stacking may not suit spectral data** - Different preprocessings emphasize different spectral features, but the meta-model can only see predictions, not which features drove them
2. **Same base model for all preprocessings** - But different preprocessings may work better with different models
3. **Poor preprocessing predictions can harm ensemble** - Unlike model selection which picks the best, stacking averages all (weighted)
4. **OOF predictions noisy with small datasets** - Typical spectral datasets (50-500 samples) may not have enough data for stable OOF predictions

**Possible fixes to try:**
1. Use different base models for different preprocessing types (PLS for derivatives, RF for raw)
2. Add feature selection within each preprocessing branch
3. Use nonlinear meta-model (e.g., LightGBM instead of Ridge)
4. Add threshold to drop poor preprocessing branches before stacking
5. Consider if ensemble approach is fundamentally unsuited for spectral data

**Literature note:** The SPRR paper (Talanta 2024) showed ensemble preprocessing outperformed single preprocessing on 6 datasets - but those may have had more samples or different data characteristics.

---

## ✅ CLASSIFICATION METRICS FOR CV AND VALIDATION (2026-01-05) - COMPLETE

**Problem:** F1, Precision, and Recall metrics only appeared for validation data, not for calibration (CV) results. Additionally, validation metrics showed all NaN for classification.

### CV (Calibration) Metrics - ADDED
- Added F1, Precision, Recall computation in `_run_single_fold()` for each CV fold
- Added averaging across folds in `_run_single_config()`
- Uses `is_binary_classification` flag (from full dataset) for consistent averaging
- Binary: `average='binary'`, Multiclass: `average='weighted'`

### Validation Metrics - FIXED
Multiple bugs causing NaN validation metrics:

1. **Label encoding mismatch** - Training used encoded labels (`y_np` via LabelEncoder) but validation passed original un-encoded labels. Fixed by:
   - Using `y_np` (encoded) for training data in validation
   - Encoding validation labels with same `label_encoder.transform()`

2. **Undefined variable** - `n_classes` renamed to `n_classes_train` but ROC AUC still used old name

3. **Class count mismatch in ROC AUC** - Validation set had fewer classes than training, causing "Number of classes in y_true not equal to columns in y_score". Fixed by:
   - Detecting when validation has fewer classes
   - Subsetting probability matrix to matching classes
   - Relabeling validation labels for ROC AUC computation

4. **Column ordering** - Calibration F1/Precision/Recall weren't ordered correctly. Fixed reordering to: Accuracy, ROC_AUC, F1, Precision, Recall (cal), then val_* columns

### Files Modified
- `src/spectral_predict/search.py`:
  - `_run_single_fold()` lines 2860-2896: Added F1/Precision/Recall computation
  - `_run_single_config()` lines 3107-3111: Added metric averaging
  - `compute_validation_metrics_for_top_models()`: Label encoding fixes, ROC AUC class handling
  - Column reordering logic for classification
- `src/spectral_predict/scoring.py`:
  - `create_results_dataframe()`: Added F1/Precision/Recall to classification columns

### Commits
- `7957a01` - feat: Add F1, Precision, Recall to classification CV results
- `736725d` - fix: Classification validation metrics and column ordering
- `217209e` - fix: Use encoded training labels for validation metrics
- `19afa6a` - fix: Robust validation F1/Precision/Recall with fallback averaging
- `4e5d024` - fix: Use n_classes_train for validation ROC AUC
- `2b6d615` - fix: Handle class mismatch in validation ROC AUC

### Note on Small Validation Sets
With < 20 validation samples across 3+ classes, metrics may appear perfect (1.0) or jump dramatically (0.25). This is expected behavior with small samples, not a bug. Recommend 10-20+ samples per class for meaningful validation metrics.

---

## ✅ MINOR CHANGES (2026-01-05)

### Regional Subset Analysis Default Changed
- Changed default from 5 regions to 10 regions
- Files: `spectral_predict_gui_optimized.py` line 1486, `search.py` line 616 and docstring
