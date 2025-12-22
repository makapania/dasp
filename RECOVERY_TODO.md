# Recovery TODO - December 16-20, 2025 Work

> Tracking 4 days of work to be preserved/restored.
> Created: 2025-12-20
> Status: **BACKED UP** to `backup_2025-12-20/`

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

### VCPA-IRIV Performance Issue (High Priority)
- [ ] **VCPA gives worse results than baseline** - Despite fixes to inclusion_prob (capped at 0.9) and display bug (index reset), VCPA still performs poorly:
  - Top VCPA model ranks #53, worse than before
  - Algorithm may need fundamental redesign:
    - importance_scores reset each outer iteration (loses history)
    - Threshold-based elimination may be too aggressive
    - Consider: cumulative importance across iterations, or different elimination strategy
  - Compare to CARS which works well - understand why CARS succeeds but VCPA fails

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
