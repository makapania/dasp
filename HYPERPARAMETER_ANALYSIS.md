# SPECTRAL PREDICT - COMPREHENSIVE HYPERPARAMETER ANALYSIS REPORT

## Executive Summary
This document provides a complete analysis of all model hyperparameters in the spectral prediction application, including what is currently exposed in the GUI, what is hard-coded in model implementations, and recommendations for parameter unification.

---

## 1. PLS (Partial Least Squares) - Regression & Classification

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (lines 59-60, 152-153)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 88-101)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (no direct GUI for PLS parameters)

### Exposed Parameters (in GUI)
- `max_n_components`: [2-100] via Spinbox at line 2015 (default: 8)
- Hyperparameter grid values from model_config.py:
  - `n_components`: [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50] (all tiers use same)

### Hard-coded Parameters
- `scale=False` (line 60): Always disabled
- No penalty parameters

### Notes
- PLS is used as a transformer and regressor
- For classification, "latent" variables = `n_components` in PLS
- "Latent number" refers to the number of PLS components extracted
- Same hyperparameter grid used across all tiers (quick, standard, comprehensive)

---

## 2. PLS-DA (Partial Least Squares Discriminant Analysis)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (line 151-153 for regression base)
- **Classification Implementation**: `/home/user/dasp/src/spectral_predict/search.py` (lines 765-767)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (line 493)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (line 250, checkbox: use_plsda)

### Exposed Parameters (in GUI)
- Same as PLS: `max_n_components` [2-100] (default: 8)
- `n_components`: [2-50] grid (same as PLS)

### Hard-coded Parameters
**Classification Pipeline (search.py lines 765-767)**:
- PLS transformer: `n_components` (tunable)
- LogisticRegression classifier: 
  - `max_iter=1000` (hard-coded, line 767)
  - `random_state=42` (hard-coded)
  - All other parameters use sklearn defaults

### Implementation Details
- **NOT a separate model** - PLS-DA is implemented as a Pipeline: PLS + LogisticRegression
- PLS performs dimensionality reduction
- LogisticRegression performs final classification
- PLS component extraction is the only tuned parameter
- LogisticRegression has no exposed hyperparameters

### Missing/Exposed Parameters
- LogisticRegression penalty (l1/l2): Hard-coded to default (l2)
- LogisticRegression C parameter: Hard-coded to default (1.0)
- LogisticRegression solver: Hard-coded to default (lbfgs)

---

## 3. Random Forest (Regression & Classification)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (lines 71-77 regression, 155-161 classification)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 265-281)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 321-328)

### Exposed Parameters (in GUI)

#### n_estimators
- **Location**: GUI lines 2353-2356, variable names: n_estimators_50, n_estimators_100, n_estimators_custom
- **Values**: [50, 100] + custom
- **Default**: 100 checked
- **Note**: Default model (line 73) uses 200, but GUI only exposes [50, 100]

#### max_depth
- **Location**: GUI lines 2410-2413, variable names: rf_max_depth_none, rf_max_depth_30, rf_max_depth_custom
- **Values**: [None, 30] + custom
- **Default**: Both None and 30 checked
- **Note**: Default model (line 74) uses None

### Hard-coded Parameters
- `random_state=42` (lines 75, 159)
- `n_jobs=-1` (lines 76, 160) - parallel processing
- **Missing from GUI**:
  - `min_samples_split` (default: 2)
  - `min_samples_leaf` (default: 1)
  - `max_features` (default: "sqrt")
  - `bootstrap` (default: True)
  - `class_weight` (default: None)
  - `criterion` (default: "mse" for regression, "gini" for classification)

### Config Grid (model_config.py lines 265-281)
- `n_estimators`: [100, 200, 500] (all tiers same)
- `max_depth`: [None, 15, 30] (all tiers same)
- **Grid Size**: 3×3 = 9 configs

### Issues
- GUI n_estimators [50, 100] differs from config [100, 200, 500]
- Only 2 depth options in GUI vs 3 in config
- Missing min_samples_split, min_samples_leaf, max_features

---

## 4. XGBoost (Regression & Classification)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (lines 106-118 regression, 176-188 classification)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 136-167)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 345-381)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **n_estimators** | xgb_n_estimators_100/200/custom | [100, 200] + custom | Both checked |
| **learning_rate** | xgb_lr_005/01/custom | [0.05, 0.1] + custom | Both checked |
| **max_depth** | xgb_max_depth_3/6/9/custom | [3, 6, 9] + custom | 3,6 checked |
| **subsample** | xgb_subsample_08/10/custom | [0.8, 1.0] + custom | Both checked |
| **colsample_bytree** | xgb_colsample_08/10/custom | [0.8, 1.0] + custom | Both checked |
| **reg_alpha** | xgb_reg_alpha_0/01/05/custom | [0, 0.1, 0.5] + custom | 0,0.1 checked |
| **reg_lambda** | xgb_reg_lambda_10/50/custom | [1.0, 5.0] + custom | Not checked |

### Hard-coded Parameters
- `tree_method='hist'` (lines 114, 184) - for high-dimensional data
- `random_state=42` (lines 115, 185)
- `n_jobs=-1` (lines 116, 186)
- `verbosity=0` (lines 117, 187)

### Config Grid (model_config.py lines 136-167)
- **Grid Size**: 2×2×3×3×3×3×2 = 648 configs
- `n_estimators`: [100, 200]
- `learning_rate`: [0.05, 0.1]
- `max_depth`: [3, 6, 9]
- `subsample`: [0.7, 0.85, 1.0]
- `colsample_bytree`: [0.7, 0.85, 1.0]
- `reg_alpha`: [0, 0.1, 0.5]
- `reg_lambda`: [1.0, 5.0]

### Missing Parameters
- `gamma` (min loss reduction to split)
- `min_child_weight` (default: 1)
- `sample_weight` support
- `objective` (default: "reg:squarederror")

---

## 5. LightGBM (Regression & Classification)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (lines 120-134 regression, 190-204 classification)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 173-192)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 396-409)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **n_estimators** | lightgbm_n_estimators_100/200/custom | [100, 200] + custom | Both checked |
| **learning_rate** | lightgbm_lr_01/custom | [0.1] + custom | Checked |
| **num_leaves** | lightgbm_num_leaves_31/50/custom | [31, 50] + custom | Both checked |

### Hard-coded Parameters (in get_model, lines 120-134, 190-204)
- `max_depth=-1` (no limit, controlled by num_leaves)
- `min_child_samples=5` (reduced from default 20)
- `subsample=0.8` (row sampling)
- `colsample_bytree=0.8` (feature sampling)
- `reg_alpha=0.1` (L1 regularization)
- `reg_lambda=1.0` (L2 regularization)
- `random_state=42`
- `n_jobs=-1`
- `verbosity=-1`

### Hard-coded Parameters (in get_model_grids, lines 593-616, 766-789)
- `max_depth=-1`
- `min_child_samples=5` ❌ **HARD-CODED - NOT TUNABLE**
- `subsample=0.8` ❌ **HARD-CODED - NOT TUNABLE**
- `colsample_bytree=0.8` ❌ **HARD-CODED - NOT TUNABLE**
- `reg_alpha=0.1` ❌ **HARD-CODED - NOT TUNABLE**
- `reg_lambda=1.0` ❌ **HARD-CODED - NOT TUNABLE**

### Config Grid (model_config.py lines 173-192)
- **Grid Size**: 3×3×3 = 27 configs
- `n_estimators`: [50, 100, 200]
- `learning_rate`: [0.05, 0.1, 0.2]
- `num_leaves`: [31, 50, 70]

### CRITICAL ISSUE ⚠️
**Parameter Duplication**: min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda are:
1. **Hard-coded in get_model()** (default model creation)
2. **Hard-coded AGAIN in get_model_grids()** (grid search) with SAME values
3. **NOT exposed in GUI** for tuning
4. **NOT in config file** for tier-based defaults

This means these parameters are never actually varied during hyperparameter search!

---

## 6. CatBoost (Regression & Classification)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (lines 136-145 regression, 206-215 classification)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 244-263)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 411-424)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **iterations** | catboost_iterations_100/200/custom | [100, 200] + custom | Both checked |
| **learning_rate** | catboost_lr_01/custom | [0.1] + custom | Checked |
| **depth** | catboost_depth_4/6/custom | [4, 6] + custom | Both checked |

### Hard-coded Parameters
- `random_state=42` (lines 143, 212)
- `verbose=False` (lines 144, 214)

### Config Grid (model_config.py lines 244-263)
- **Grid Size**: 3×3×3 = 27 configs
- `iterations`: [50, 100, 200]
- `learning_rate`: [0.05, 0.1, 0.2]
- `depth`: [4, 6, 8]

### Issues
- Missing parameters:
  - `loss_function`
  - `early_stopping_rounds`
  - `subsample`
  - `colsample_bylevel`

---

## 7. Neural Networks - MLP (Multi-Layer Perceptron)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (lines 79-87 regression, 163-171 classification)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 283-302)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 441-453)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **hidden_layer_sizes** | mlp_hidden_64/128_64/custom | [(64,), (128,64)] + custom | Both checked |
| **alpha** | mlp_alpha_1e3/custom | [0.001] + custom | Checked |
| **learning_rate_init** | mlp_lr_init_1e3/custom | [0.001] + custom | Checked |

### Hard-coded Parameters
- `max_iter=500` (lines 84, 168) - now 100 in GUI (line 217)
- `random_state=42` (lines 85, 169)
- `early_stopping=True` (lines 86, 170)
- `solver='lbfgs'` (default in sklearn)
- `activation='relu'` (default in sklearn)

### Config Grid (model_config.py lines 283-302)
- **Grid Size**: 2×2×2 = 8 configs
- `hidden_layer_sizes`: [(64,), (128, 64)]
- `alpha`: [1e-4, 1e-3]
- `learning_rate_init`: [1e-3, 1e-2]

### Issues
- GUI only exposes [0.001] for alpha, but config has [1e-4, 1e-3]
- Missing parameters:
  - `batch_size` (default: auto)
  - `learning_rate` schedule (default: constant)
  - `momentum` (default: 0.9)
  - `nesterov` (default: True)

---

## 8. NeuralBoosted (Custom Ensemble)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/neural_boosted.py` (lines 120-166)
- **Model Instance Creation**: `/home/user/dasp/src/spectral_predict/models.py` (lines 89-101)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 215-238)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 312-319)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **n_estimators** | n_estimators_50/100/custom | [50, 100] + custom | 100 checked |
| **learning_rate** | lr_005/01/02/03 | [0.05, 0.1, 0.2, 0.3] | 0.1, 0.2, 0.3 checked |

### Hard-coded Parameters (in NeuralBoostedRegressor.__init__, neural_boosted.py)

**Not tuned in grid search**:
- `hidden_layer_size=3` (line 93 in models.py)
- `activation='tanh'` (line 94 in models.py)
- `alpha=1e-4` (line 98 in models.py, weight decay)
- `max_iter=100` (line 127, optimized from 500)
- `early_stopping=True` (line 128)
- `validation_fraction=0.15` (line 129)
- `n_iter_no_change=10` (line 130)
- `loss='mse'` (line 131)
- `huber_delta=1.35` (line 132)
- `random_state=42`
- `verbose=0`

### Config Grid (model_config.py lines 215-238)
- **Grid Size**: 2×3×2×2 = 24 configs
- `n_estimators`: [100, 150]
- `learning_rate`: [0.1, 0.2, 0.3]
- `hidden_layer_size`: [3, 5]
- `activation`: ['tanh', 'identity']

### CRITICAL ISSUE ⚠️
**Parameter Duplication in NeuralBoosted**:
1. **n_estimators**: GUI [50, 100] vs Config [100, 150] - MISMATCH
2. **learning_rate**: GUI includes 0.05, Config [0.1, 0.2, 0.3] - MISMATCH
3. **hidden_layer_size**: Exposed in config [3, 5] but GUI doesn't expose (uses hardcoded 3)
4. **activation**: Exposed in config ['tanh', 'identity'] but GUI doesn't expose (uses hardcoded 'tanh')

### Fully Exposed Parameters (all controllable via Config/GUI)
- `n_estimators`
- `learning_rate`

### Partially Exposed (in config but not GUI)
- `hidden_layer_size`: [3, 5]
- `activation`: ['tanh', 'identity']

### Hard-coded (Not Exposed)
- `alpha`: 1e-4 (L2 regularization)
- `max_iter`: 100 (weak learner training iterations)
- `early_stopping`: True
- `validation_fraction`: 0.15
- `n_iter_no_change`: 10
- `loss`: 'mse'

---

## 9. Ridge Regression

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (line 62)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 103-116)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 330-336)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **alpha** | ridge_alpha_0001/001/01/1/10/custom | [0.001, 0.01, 0.1, 1.0, 10.0] + custom | All checked |

### Hard-coded Parameters
- `random_state=42` (line 63)

### Config Grid (model_config.py lines 103-116)
- **Grid Size**: 5 configs
- `alpha`: [0.001, 0.01, 0.1, 1.0, 10.0]

---

## 10. Lasso Regression

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (line 65)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 304-318)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 338-343)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **alpha** | lasso_alpha_0001/001/01/1/custom | [0.001, 0.01, 0.1, 1.0] + custom | All checked |

### Hard-coded Parameters
- `random_state=42` (line 66)
- `max_iter=500` (line 66, overridden by GUI max_iter default 100)

### Config Grid (model_config.py lines 304-318)
- **Grid Size**: 4 configs
- `alpha`: [0.001, 0.01, 0.1, 1.0]

---

## 11. ElasticNet Regression

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (line 68)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 118-134)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 383-394)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **alpha** | elasticnet_alpha_001/01/10/custom | [0.01, 0.1, 1.0] + custom | 0.01, 0.1, 1.0 checked |
| **l1_ratio** | elasticnet_l1_ratio_03/05/07/custom | [0.3, 0.5, 0.7] + custom | All checked |

### Hard-coded Parameters
- `random_state=42` (line 69)
- `max_iter=500` (line 69)

### Config Grid (model_config.py lines 118-134)
- **Grid Size**: 4×5 = 20 configs
- `alpha`: [0.001, 0.01, 0.1, 1.0]
- `l1_ratio`: [0.1, 0.3, 0.5, 0.7, 0.9]

### Issues
- GUI alpha [0.01, 0.1, 1.0] vs Config [0.001, 0.01, 0.1, 1.0] - missing 0.001
- GUI l1_ratio [0.3, 0.5, 0.7] vs Config [0.1, 0.3, 0.5, 0.7, 0.9] - missing 0.1, 0.9

---

## 12. SVR (Support Vector Regression)

### Files
- **Model Definition**: `/home/user/dasp/src/spectral_predict/models.py` (line 104)
- **Config**: `/home/user/dasp/src/spectral_predict/model_config.py` (lines 194-213)
- **GUI**: `/home/user/dasp/spectral_predict_gui_optimized.py` (lines 426-439)

### Exposed Parameters (in GUI)

| Parameter | GUI Variables | Values | Default |
|-----------|---------------|--------|---------|
| **kernel** | svr_kernel_rbf/linear | ['rbf', 'linear'] | Both checked |
| **C** | svr_C_10/100/custom | [1.0, 10.0] + custom | Both checked |
| **gamma** | svr_gamma_scale/auto/custom | ['scale', 'auto'] + custom | scale checked |

### Hard-coded Parameters
- None (uses sklearn defaults)

### Config Grid (model_config.py lines 194-213)
- **Grid Size**: 9 configs
- `kernel`: ['rbf', 'linear']
- `C`: [0.1, 1.0, 10.0]
- `gamma`: ['scale', 'auto']

### Issues
- GUI C [1.0, 10.0] vs Config [0.1, 1.0, 10.0] - missing 0.1

---

## SUMMARY TABLE: PARAMETER DUPLICATION & MISMATCH ISSUES

| Model | Issue | Impact | Files |
|-------|-------|--------|-------|
| **NeuralBoosted** | n_estimators: GUI [50,100] vs Config [100,150] | MISMATCH - Grid search uses [100,150] not [50,100] | models.py, model_config.py, GUI |
| **NeuralBoosted** | hidden_layer_size: Exposed in config [3,5] but not in GUI | Partially exposed - always uses 3 | model_config.py, models.py:93 |
| **NeuralBoosted** | activation: Exposed in config ['tanh','identity'] but not in GUI | Partially exposed - always uses 'tanh' | model_config.py, models.py:94 |
| **LightGBM** | min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda | HARD-CODED in grid search with fixed values | models.py:595-613, 768-786 |
| **XGBoost** | subsample, colsample_bytree in models.py (default) vs GUI exposed | GUI allows override, good! | models.py:111-112, GUI |
| **ElasticNet** | alpha: GUI [0.01,0.1,1.0] vs Config [0.001,0.01,0.1,1.0] | MISMATCH - Config has 0.001 not in GUI | model_config.py:125, GUI:386 |
| **ElasticNet** | l1_ratio: GUI [0.3,0.5,0.7] vs Config [0.1,0.3,0.5,0.7,0.9] | MISMATCH - Config has 0.1, 0.9 not in GUI | model_config.py:126, GUI:391-393 |
| **RandomForest** | n_estimators: GUI [50,100] vs Config [100,200,500] vs Default 200 | TRIPLE MISMATCH! | models.py:73, model_config.py:267, GUI:313-315 |
| **RandomForest** | max_depth: GUI [None,30] vs Config [None,15,30] | MISMATCH - Config has 15 not in GUI | model_config.py:268, GUI:326-328 |
| **SVR** | C: GUI [1.0,10.0] vs Config [0.1,1.0,10.0] | MISMATCH - Config has 0.1 not in GUI | model_config.py:197, GUI:432-433 |

---

## KEY FINDINGS & RECOMMENDATIONS

### 1. **CRITICAL: LightGBM Parameter Duplication**
**Problem**: min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda are hard-coded in `get_model_grids()` at lines 595-613 and 768-786, meaning they are NEVER varied during grid search.

**Current Values** (hard-coded):
```python
min_child_samples=5
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=1.0
```

**Recommendation**: Either:
- Option A: Expose these parameters in the config and GUI if they should be tuned
- Option B: Accept current values and document them as fixed defaults

### 2. **CRITICAL: NeuralBoosted Parameter Mismatch**
**Problem**: 
- n_estimators: GUI supports [50, 100] but config has [100, 150]
- hidden_layer_size: Config supports [3, 5] but GUI hard-codes to 3
- activation: Config supports ['tanh', 'identity'] but GUI hard-codes to 'tanh'

**Recommendation**: Unify GUI and config to use same values. Suggest:
```python
n_estimators: [100, 150, 200]
hidden_layer_size: [3, 5]  # Expose in GUI
activation: ['tanh', 'identity']  # Expose in GUI
learning_rate: [0.1, 0.2, 0.3]  # Expose in GUI
```

### 3. **MODERATE: RandomForest Parameter Triple Mismatch**
**Problem**: Three different n_estimators values across codebase:
- Default model (models.py:73): 200
- Config (model_config.py:267): [100, 200, 500]
- GUI (gui.py:313-315): [50, 100]

**Recommendation**: Use config values [100, 200, 500] in GUI:
```python
self.rf_n_trees_100 = tk.BooleanVar(value=True)
self.rf_n_trees_200 = tk.BooleanVar(value=True)  # Current default
self.rf_n_trees_500 = tk.BooleanVar(value=True)
```

### 4. **MODERATE: Missing RandomForest Parameters**
Not exposed in GUI:
- `min_samples_split` (default: 2) - controls tree depth
- `min_samples_leaf` (default: 1) - controls overfitting
- `max_features` (default: "sqrt") - important for high-dimensional data

### 5. **Parameter Inconsistencies Between GUI and Config**

**ElasticNet**:
- GUI alpha missing 0.001
- GUI l1_ratio missing 0.1 and 0.9

**SVR**:
- GUI C missing 0.1

**RandomForest**:
- GUI max_depth missing 15 (has None, 30; config has None, 15, 30)

### 6. **"Latent" Terminology**
**Finding**: "Latent number" (latent variables) in PLS context = `n_components`
- PLS extracts `n_components` latent variables
- Also called "Latent Vectors" or "LVs"
- File references: search.py:818 ("LVs"), model_config.py (n_components)
- Not PLS-specific - LVs are output of PLS that can be used by downstream classifiers

### 7. **PLS-DA Implementation**
- **NOT a single model** - Pipeline of two components
- Component 1: PLS for dimensionality reduction (n_components is tuned)
- Component 2: LogisticRegression for classification (all parameters hard-coded)
- LogisticRegression parameters that should be considered for tuning:
  - `C` (regularization strength, default: 1.0)
  - `penalty` (l1 or l2, default: l2)
  - `solver` (default: lbfgs)

---

## FILE LOCATIONS SUMMARY

### Configuration Files
- `/home/user/dasp/src/spectral_predict/model_config.py` - All tier-based hyperparameter grids
- `/home/user/dasp/src/spectral_predict/model_registry.py` - Model lists and metadata

### Model Implementation Files
- `/home/user/dasp/src/spectral_predict/models.py` - All model instantiation (get_model, get_model_grids)
- `/home/user/dasp/src/spectral_predict/neural_boosted.py` - Custom NeuralBoosted implementation

### Execution Files
- `/home/user/dasp/src/spectral_predict/search.py` - Model grid search (lines 764-814 for PLS-DA handling)

### GUI Files
- `/home/user/dasp/spectral_predict_gui_optimized.py` - All hyperparameter GUI controls

---

## RECOMMENDATIONS FOR CLEANUP

1. **Synchronize RandomForest parameters** between config and GUI
2. **Document or expose** the hard-coded LightGBM parameters
3. **Fix NeuralBoosted mismatch** between GUI and config
4. **Consider exposing**:
   - RandomForest: min_samples_split, min_samples_leaf
   - LightGBM: min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
   - PLS-DA LogisticRegression: C, penalty, solver
   - NeuralBoosted: hidden_layer_size, activation (currently in config but not GUI)

