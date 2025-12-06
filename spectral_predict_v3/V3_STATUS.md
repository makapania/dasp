# Spectral Predict v3 - Development Status

**Last Updated:** 2024-12-05

## Quick Start
```bash
cd C:\Users\sponheim\git\dasp
python -m spectral_predict_v3.main
```

## Full Plan Location
The detailed architecture plan is at:
`C:\Users\sponheim\.claude\plans\frolicking-tinkering-blanket.md`

---

## COMPLETED

### Core Backend (Phase 5)
- [x] `core/preprocess.py` - SNV, SavgolDerivative
- [x] `core/model_config.py` - Tier definitions, hyperparameter grids (flat, same for all tiers)
- [x] `core/models.py` - Model factory with optional imports (LightGBM, XGBoost, CatBoost)
- [x] `core/variable_selection.py` - UVE, SPA, iPLS, UVE-SPA algorithms
- [x] `core/regions.py` - Region correlation analysis
- [x] `core/search.py` - Full automation with run_auto_search, run_manual_search

### UI (Phases 1-4)
- [x] Import panel with file/folder browser
- [x] Column config dialog
- [x] Data grid with virtual scrolling
- [x] Spectra plot with zoom/pan and derivative overlays
- [x] PCA plot with color by target
- [x] Plot ↔ Grid selection sync
- [x] "Assign to Group" dialog

### Build Panel (Phase 5)
- [x] Manual vs Auto mode toggle
- [x] Tier selection (Quick/Standard/Comprehensive) - only affects which models
- [x] Preprocessing checkboxes (all SNV/SG combinations enabled by default, raw disabled)
- [x] Advanced Options: window sizes, variable selection, variable counts, region analysis
- [x] **ALL hyperparameters exposed** in Advanced Options → Hyperparameter Grids
- [x] Results table with ranked models

---

## TODO (Remaining Work)

### High Priority
1. **Make hyperparameter edits functional** - Currently hyperparameters are visible/editable but editing doesn't affect the search yet. Need to read UI values and pass to search.py

### Medium Priority
2. **Predict Panel** - Load trained model and apply to new data
3. **Export functionality** - Export results, models, predictions
4. **Data grid features** - Fill down, copy/paste, add/delete columns, row flagging

### Low Priority
5. **Performance testing** - Verify smooth scrolling with 10k+ samples
6. **Bug fixes and polish**

---

## Key Design Decisions

1. **Tier only affects models** - Quick/Standard/Comprehensive tiers only change which models are tested. Hyperparameters are identical across all tiers.

2. **Preprocessing is user-selectable** - Not tied to tier. Defaults: all SNV/SG1/SG2 combinations enabled, raw disabled.

3. **All hyperparameters exposed** - Users can see and (eventually) edit all hyperparameters for every model type in Advanced Options.

4. **Dear PyGui** - GPU-accelerated UI framework for performance with large datasets.

5. **Standalone from v1** - v3 is completely standalone, no imports from v1 (forked code instead).

---

## Hyperparameters Exposed Per Model

| Model | Parameters |
|-------|------------|
| PLS/PLS-DA | n_components, scale |
| Ridge | alpha, fit_intercept, solver |
| Lasso | alpha, max_iter, tol, fit_intercept |
| ElasticNet | alpha, l1_ratio, max_iter, tol, fit_intercept |
| RandomForest | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap |
| LightGBM | n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda |
| XGBoost | n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda |
| CatBoost | iterations, learning_rate, depth, l2_leaf_reg, border_count, bagging_temperature, random_strength |
| SVR/SVM | kernel, C, gamma, epsilon, degree, coef0, max_iter |
