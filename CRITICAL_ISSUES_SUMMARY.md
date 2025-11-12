# CRITICAL HYPERPARAMETER ISSUES - QUICK REFERENCE

## 🔴 CRITICAL FINDINGS

### 1. LightGBM: Parameters Never Tuned During Grid Search
**Location**: `/home/user/dasp/src/spectral_predict/models.py` lines 595-613, 768-786

**Issue**: 5 parameters are hard-coded with FIXED values in the grid search function:
- `min_child_samples=5` (always 5, never varies)
- `subsample=0.8` (always 0.8, never varies)
- `colsample_bytree=0.8` (always 0.8, never varies)
- `reg_alpha=0.1` (always 0.1, never varies)
- `reg_lambda=1.0` (always 1.0, never varies)

**Current Grid Search**: Only varies n_estimators, learning_rate, num_leaves (3x3x3 = 27 configs)
**What's Actually Tested**: 27 configs with FIXED regularization

**Fix Options**:
1. Add these to model_config.py and expose in GUI
2. Or document them as fixed defaults and mark code as intended

```python
# Current problematic code (models.py:595-613):
LGBMRegressor(
    n_estimators=n_est,
    learning_rate=lr,
    num_leaves=num_leaves,
    max_depth=-1,
    min_child_samples=5,        # HARD-CODED - NOT VARIED
    subsample=0.8,              # HARD-CODED - NOT VARIED
    colsample_bytree=0.8,       # HARD-CODED - NOT VARIED
    reg_alpha=0.1,              # HARD-CODED - NOT VARIED
    reg_lambda=1.0,             # HARD-CODED - NOT VARIED
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
```

---

### 2. NeuralBoosted: Triple Parameter Mismatch (GUI vs Config)
**Location**: 
- GUI: `/home/user/dasp/spectral_predict_gui_optimized.py` lines 312-319
- Config: `/home/user/dasp/src/spectral_predict/model_config.py` lines 215-238
- Models: `/home/user/dasp/src/spectral_predict/models.py` lines 89-101

**Issues**:

#### Issue 2a: n_estimators Mismatch
```
GUI defaults:          [50, 100] with custom option
Config grid:           [100, 150]
RESULT: Grid search uses [100, 150], not what user selected in GUI
```

#### Issue 2b: hidden_layer_size Not Exposed in GUI
```
Config supports:       [3, 5]
GUI hardcodes:         3
RESULT: Can't tune hidden layer size despite config support
Location: models.py:93
```

#### Issue 2c: activation Not Exposed in GUI  
```
Config supports:       ['tanh', 'identity']
GUI hardcodes:         'tanh'
RESULT: Can't use identity activation despite config support
Location: models.py:94
```

**Fix**: Unify GUI and Config values:
```python
# Recommended unified values:
n_estimators: [100, 150, 200]
hidden_layer_size: [3, 5]          # Expose in GUI
activation: ['tanh', 'identity']   # Expose in GUI
learning_rate: [0.1, 0.2, 0.3]    # Already exposed
```

---

### 3. RandomForest: Triple Mismatch (Default vs Config vs GUI)
**Location**:
- Default: `/home/user/dasp/src/spectral_predict/models.py` line 73
- Config: `/home/user/dasp/src/spectral_predict/model_config.py` line 267
- GUI: `/home/user/dasp/spectral_predict_gui_optimized.py` lines 313-315, 326-328

**Issues**:

#### n_estimators Triple Mismatch
```
Default model uses:    200
Config grid has:       [100, 200, 500]
GUI exposes:           [50, 100]
RESULT: User can't select 500 trees from GUI!
```

#### max_depth GUI Missing Middle Value
```
Config has:    [None, 15, 30]
GUI shows:     [None, 30]
RESULT: Can't select depth=15 from GUI
```

**Fix**:
```python
# Update GUI lines 313-315 to match config:
self.rf_n_trees_100 = tk.BooleanVar(value=True)
self.rf_n_trees_200 = tk.BooleanVar(value=True)
self.rf_n_trees_500 = tk.BooleanVar(value=True)

# Update GUI lines 326-328 to add 15:
self.rf_max_depth_none = tk.BooleanVar(value=True)
self.rf_max_depth_15 = tk.BooleanVar(value=True)    # ADD THIS
self.rf_max_depth_30 = tk.BooleanVar(value=True)
```

---

## 🟡 MODERATE ISSUES

### 4. GUI Parameter Values Don't Match Config
Multiple models have GUI checkboxes that don't cover all config values:

| Model | Parameter | Config Values | GUI Values | Missing |
|-------|-----------|---------------|-----------|---------|
| ElasticNet | alpha | [0.001, 0.01, 0.1, 1.0] | [0.01, 0.1, 1.0] | 0.001 |
| ElasticNet | l1_ratio | [0.1, 0.3, 0.5, 0.7, 0.9] | [0.3, 0.5, 0.7] | 0.1, 0.9 |
| SVR | C | [0.1, 1.0, 10.0] | [1.0, 10.0] | 0.1 |

---

## 📋 PARAMETER LOCATION REFERENCE

### Where Hyperparameters Are Defined (in priority order):

1. **GUI** (user-facing): `/home/user/dasp/spectral_predict_gui_optimized.py` lines 312-453
   - Default values and options shown to user
   - If GUI checkbox doesn't exist, parameter not user-tunable

2. **Config** (grid search values): `/home/user/dasp/src/spectral_predict/model_config.py` lines 83-319
   - Tier-based default grids (quick, standard, comprehensive, experimental)
   - These are actually searched during hyperparameter search

3. **Models.py** (hard-coded defaults): `/home/user/dasp/src/spectral_predict/models.py` lines 33-220
   - Two functions: `get_model()` and `get_model_grids()`
   - Hard-coded values that never vary

4. **Custom Models**: `/home/user/dasp/src/spectral_predict/neural_boosted.py` lines 120-166
   - NeuralBoosted class definition with all parameters

---

## PLS-DA KEY FINDINGS

### What is PLS-DA?
**NOT a single model** - It's a Pipeline:
1. **PLS Transformer** (sklearn PLSRegression): Extracts latent variables
2. **LogisticRegression Classifier** (sklearn LogisticRegression): Performs classification

### Current Implementation (search.py:765-767)
```python
if model_name == "PLS-DA":
    pipe_steps.append(("pls", model))  # Tunable: n_components
    pipe_steps.append(("lr", LogisticRegression(max_iter=1000, random_state=42)))
```

### What's Tuned
- Only `n_components` in PLS (values: [2-50])

### What's Hard-coded
- LogisticRegression: `max_iter=1000`, `random_state=42`
- All other LogisticRegression parameters use sklearn defaults:
  - `penalty='l2'` (could be 'l1', 'elasticnet')
  - `C=1.0` (regularization strength, could be tuned)
  - `solver='lbfgs'` (could be 'liblinear', 'saga', etc.)

### "Latent Number" Definition
- **"Latent number"** = number of PLS components (`n_components`)
- PLS extracts `n_components` latent variables before classification
- These latent variables are the features seen by LogisticRegression

---

## EXPOSED VS HARD-CODED PARAMETERS BY MODEL

### Models with Most Complete GUI Exposure
1. **XGBoost**: 7 parameters exposed (n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda)
2. **Ridge/Lasso**: 1-2 parameters (alpha)
3. **ElasticNet**: 2 parameters (alpha, l1_ratio)

### Models with MISSING GUI Exposure
1. **RandomForest**: Missing min_samples_split, min_samples_leaf, max_features
2. **LightGBM**: Missing min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda (5 critical params!)
3. **NeuralBoosted**: Missing hidden_layer_size, activation (in config but not GUI)
4. **CatBoost**: Missing loss_function, early_stopping_rounds, subsample
5. **MLP**: Missing batch_size, learning_rate schedule, momentum

---

## FILE REFERENCES FOR ALL CRITICAL LINES

### Model Definitions
- RandomForest: models.py lines 71-77 (regression), 155-161 (classification)
- XGBoost: models.py lines 106-118 (regression), 176-188 (classification)
- LightGBM: models.py lines 120-134 (regression), 190-204 (classification)
- NeuralBoosted: models.py lines 89-101
- PLS-DA: search.py lines 765-767

### Configuration Grids
- All models: model_config.py lines 83-319
- NeuralBoosted specifically: model_config.py lines 215-238

### GUI Controls
- All hyperparameter checkboxes: gui.py lines 312-453

### Hard-coded Parameter Issues
- NeuralBoosted: models.py lines 93-94 (hardcoded hidden_layer_size=3, activation='tanh')
- LightGBM: models.py lines 595-613, 768-786 (hardcoded in grid search)

