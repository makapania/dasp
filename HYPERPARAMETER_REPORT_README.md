# HYPERPARAMETER ANALYSIS REPORT - README

## Reports Generated

This analysis includes **2 comprehensive documents**:

### 1. **HYPERPARAMETER_ANALYSIS.md** (567 lines)
Complete technical analysis of all 12 models with:
- File locations and line numbers for all parameters
- Exposed vs hard-coded parameters for each model
- Configuration grid details
- Tier-based defaults (quick, standard, comprehensive, experimental)
- Issues and inconsistencies with detailed examples

### 2. **CRITICAL_ISSUES_SUMMARY.md** (222 lines)
Quick reference guide focused on:
- 3 Critical issues requiring immediate attention
- 1 Moderate issue affecting multiple models
- Parameter location reference
- PLS-DA implementation details
- Specific code examples and fixes

## Key Findings Summary

### Models Analyzed
1. PLS (Partial Least Squares)
2. PLS-DA (PLS Discriminant Analysis)
3. Random Forest
4. XGBoost
5. LightGBM
6. CatBoost
7. MLP (Neural Network)
8. NeuralBoosted (Custom)
9. Ridge Regression
10. Lasso Regression
11. ElasticNet
12. SVR (Support Vector Regression)

---

## CRITICAL ISSUES FOUND

### Issue #1: LightGBM Hard-coded Parameters (CRITICAL)
**Status**: Parameters never vary during grid search
**Impact**: 5 critical parameters are fixed instead of tuned
- min_child_samples=5 (always)
- subsample=0.8 (always)
- colsample_bytree=0.8 (always)
- reg_alpha=0.1 (always)
- reg_lambda=1.0 (always)

**Files**: models.py lines 595-613, 768-786
**Fix**: Expose in config and GUI or document as intentional defaults

### Issue #2: NeuralBoosted Triple Mismatch (CRITICAL)
**Status**: GUI, Config, and Model use different values
**Impact**: Grid search uses different parameters than GUI shows
- n_estimators: GUI [50,100] vs Config [100,150]
- hidden_layer_size: Config [3,5] but GUI hardcoded to 3
- activation: Config ['tanh','identity'] but GUI hardcoded to 'tanh'

**Files**: gui.py:312-319, model_config.py:215-238, models.py:89-101
**Fix**: Unify values between GUI and config

### Issue #3: RandomForest Triple Mismatch (CRITICAL)
**Status**: Default, Config, and GUI all use different values
**Impact**: Users can't access all available tuning options
- n_estimators: Default 200 vs Config [100,200,500] vs GUI [50,100]
- max_depth: Config [None,15,30] vs GUI [None,30]

**Files**: models.py:73, model_config.py:267, gui.py:313-328
**Fix**: Update GUI to match config values

### Issue #4: GUI-Config Parameter Mismatches (MODERATE)
**Status**: Multiple parameters have different ranges in GUI vs Config
**Impact**: Users can't test all parameter combinations defined in config

Models affected:
- ElasticNet (alpha missing 0.001, l1_ratio missing 0.1/0.9)
- SVR (C missing 0.1)
- RandomForest (max_depth missing 15)

---

## "LATENT NUMBER" DEFINITION (From PLS/PLS-DA Analysis)

**"Latent number" = number of PLS components (n_components)**
- PLS extracts n_components latent variables
- Also called "Latent Vectors" or "LVs"
- Not specific to PLS - any model can use these latent variables
- In PLS-DA: PLS extracts latents, LogisticRegression classifies them
- Range: 2-50 components (configurable via max_n_components, default 8)

---

## PLS-DA IMPLEMENTATION DETAILS

**What it actually is**:
- NOT a single model class
- Pipeline of 2 components:
  1. PLS (dimensionality reduction) - tunable: n_components
  2. LogisticRegression (classification) - all parameters hard-coded

**Tuned Parameters**:
- n_components: [2-50]

**Hard-coded Parameters**:
- LogisticRegression max_iter=1000
- LogisticRegression random_state=42
- All other sklearn LogisticRegression defaults

**Location**: search.py lines 765-767

---

## PARAMETER EXPOSURE BY MODEL

### Best Exposed (XGBoost)
7 parameters tunable via GUI + config

### Worst Exposed (LightGBM, NeuralBoosted, RandomForest)
Multiple parameters hard-coded despite being in config or needing tuning

### Summary Table
| Model | GUI Params | Config Params | Hard-coded | Issue |
|-------|-----------|---------------|-----------|-------|
| **RandomForest** | 2 | 2 | 4+ | MISMATCH: GUI doesn't match config values |
| **LightGBM** | 3 | 3 | 5 | CRITICAL: 5 params never varied in search |
| **NeuralBoosted** | 2 | 4 | 6+ | MISMATCH: Config params not in GUI |
| **XGBoost** | 7 | 7 | 4 | Good - mostly consistent |
| **ElasticNet** | 2 | 2 | 2 | MISMATCH: GUI missing some config values |

---

## HOW TO USE THESE REPORTS

### For Code Review
- **Start with**: CRITICAL_ISSUES_SUMMARY.md (faster read)
- **Then read**: Relevant sections of HYPERPARAMETER_ANALYSIS.md

### For Fixing Issues
1. **LightGBM**: model_config.py lines 173-192 + models.py lines 595-613, 768-786
2. **NeuralBoosted**: gui.py + model_config.py + models.py (3 files to sync)
3. **RandomForest**: gui.py (update checkboxes to match config values)

### For Parameter Locations
- See "FILE LOCATIONS SUMMARY" section at end of HYPERPARAMETER_ANALYSIS.md
- Or quick reference table in CRITICAL_ISSUES_SUMMARY.md

### For Understanding Parameters
1. **GUI defaults**: spectral_predict_gui_optimized.py lines 312-453
2. **Grid search values**: model_config.py lines 83-319
3. **Hard-coded defaults**: models.py lines 33-220
4. **NeuralBoosted specifics**: neural_boosted.py lines 120-166

---

## KEY STATISTICS

- **12 models** analyzed
- **3 critical issues** found
- **1 moderate issue** (multiple model inconsistencies)
- **Worst case**: LightGBM - 5 parameters never tuned
- **Best case**: XGBoost - well-documented and exposed
- **Total parameters analyzed**: ~80+
- **Hardcoded parameters**: ~40+
- **GUI-Config mismatches**: 8+ instances

---

## RECOMMENDATIONS

### Immediate Fixes (High Priority)
1. Fix LightGBM parameter duplication (models.py:595-613)
2. Sync NeuralBoosted GUI and config values
3. Update RandomForest GUI to match config

### Medium Priority
1. Add missing parameter ranges to GUI (ElasticNet, SVR)
2. Document intentional hard-coded parameters
3. Add min_samples_split, min_samples_leaf to RandomForest

### Long-term Improvements
1. Standardize parameter definition in single location
2. Auto-generate GUI from config
3. Add parameter documentation in config file
4. Create parameter validation layer

---

## FILE CHECKLIST

Both reports are saved in `/home/user/dasp/`:
- ✓ HYPERPARAMETER_ANALYSIS.md (main report, 567 lines)
- ✓ CRITICAL_ISSUES_SUMMARY.md (quick reference, 222 lines)
- ✓ HYPERPARAMETER_REPORT_README.md (this file)

---

## NEXT STEPS

1. Read CRITICAL_ISSUES_SUMMARY.md first (5-10 min read)
2. Review specific files mentioned for each issue
3. Use HYPERPARAMETER_ANALYSIS.md for complete technical details
4. Implement fixes based on severity (critical → medium → low)

**Questions?** All file locations, line numbers, and code examples are provided in both reports.

