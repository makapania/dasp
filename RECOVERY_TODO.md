# Recovery TODO - December 16-20, 2025 Work

> Tracking 4 days of work to be preserved/restored.
> Created: 2025-12-20
> Status: **BACKED UP** to `backup_2025-12-20/`

---

## 🚨 TOP PRIORITY: BAYESIAN BROKEN BY RECENT FIXES (2025-12-22)

**CRITICAL ISSUE:** Recent commits broke Bayesian - now gives much worse models.

- **BEFORE fixes:** Bayesian was achieving R2 = 0.97
- **AFTER fixes:** Bayesian now achieves R2 = 0.91 (significantly worse)
- Commits that broke it: c6a9dd1, ca3d50b, 6d9938f
- **ACTION NEEDED:** Revert these commits to restore Bayesian performance

**Secondary issue:** R2 concordance between Results tab and Model Development tab (separate problem, lower priority)

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
- [ ] **ORIGINAL PROBLEM NOT RESOLVED** - The investigation that led to edge masking changes was prompted by R2 mismatch between Results tab and Model Development tab. The root cause of this mismatch was NOT fully diagnosed. Potential remaining issues:
  - Wavelength order corruption: `_format_wavelengths_as_spec()` sorts wavelengths, `_parse_wavelength_spec()` reorders to X_original order
  - Variable order matters for many models but may not be preserved correctly

### Bayesian Optimization - BROKEN (2025-12-22)

**STATUS: MADE WORSE BY FIXES** - All attempted fixes have degraded Bayesian performance. Consider reverting commits: c6a9dd1, ca3d50b, 6d9938f.

**Models Explored (based on tier setting):**
- quick: PLS, Ridge, ElasticNet
- standard: PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM
- comprehensive: PLS, Ridge, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, NeuralBoosted
- experimental: All 11 models (adds SVR, MLP)

**Preprocessing Explored:** raw, snv, deriv1, deriv2 (window sizes 7, 11, 15, 21)

**Variable Selection Explored:** importance, spa, uve, cars (after fix 6d9938f)

**Fixes Attempted (none resolved the core issues):**

1. **SubsetTag key mismatch** (commit c6a9dd1)
   - Changed `'subset_tag'` to `'SubsetTag'` in `bayesian_utils.py:571`
   - Issue: Results were showing "full" even for subsets
   - Result: Tags display correctly but R2 still wrong

2. **Added missing variable selection methods** (commit ca3d50b)
   - Added `'cars-aware'` support (was only checking `== 'cars'`)
   - Added `'vcpa-iriv'` support with importance score mapping
   - Issue: CARS subsets weren't appearing in results
   - Result: Methods run but results still incorrect

3. **Expanded default variable selection** (commit 6d9938f)
   - Changed default from `['importance']` to `['importance', 'spa', 'uve', 'cars']`
   - Issue: Bayesian should auto-explore multiple methods
   - Result: More methods tested but R2 still doesn't match Grid Search

4. **Added missing parameters to run_bayesian_search()**
   - Added: `baseline_method`, `baseline_params`, `smoothing`, `smoothing_window`, `smoothing_polyorder`
   - Issue: NameError on missing parameters
   - Result: No longer crashes but results still wrong

**Remaining Problems:**
- [ ] R2 from Bayesian Results tab does NOT match Model Development tab
- [ ] R2 from Bayesian is significantly worse than Grid Search for same data
- [ ] full_vars values inconsistent (saw 2121 vs 2131 - 100 wavelength difference)
- [ ] Variable selection methods may not be running correctly
- [ ] Possible issues with how Optuna trials are being converted to DASP format

**Files Modified:**
- `src/spectral_predict/bayesian_utils.py` - Lines 203, 215, 370-423, 571
- `src/spectral_predict/search.py` - run_bayesian_search parameter additions

### NSGA-II - BROKEN (2025-12-22)

**STATUS: DOES NOT WORK** - Gives much worse results than Grid Search.

Possible issues:
- Pareto front knee point selection may not pick optimal model
- Multi-objective tradeoffs sacrifice accuracy for other objectives
- Hyperparameter search space not well-tuned
- Population/generation settings insufficient for convergence

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
