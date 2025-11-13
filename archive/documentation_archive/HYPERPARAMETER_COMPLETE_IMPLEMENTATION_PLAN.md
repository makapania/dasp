# COMPLETE HYPERPARAMETER IMPLEMENTATION - DEFINITIVE HANDOFF

**Date Created**: 2025-11-12
**Status**: Ready for Agent Execution
**Estimated Total Time**: 14 hours (wall clock with parallel execution)

---

## 1. EXECUTIVE SUMMARY

### What Happened

The previous `HYPERPARAMETER_IMPLEMENTATION_HANDOFF.md` document is **COMPLETELY STALE** and describes an implementation that never existed. A forensic analysis of the git history revealed:

1. **Commit cb1ead6** ("feat: Expose all model hyperparameters"):
   - ✅ RandomForest: Fully implemented 6 new parameters
   - ✅ LightGBM: Fully implemented 6 new parameters, removed hard-coded values
   - ✅ PLS-DA: Partially implemented LogisticRegression parameters
   - ❌ **PLS, Ridge, Lasso, ElasticNet**: NEVER IMPLEMENTED

2. **Commit d801c91** ("fix: Unify hyperparameters across tiers"):
   - ❌ RandomForest: **REMOVED** all 6 new parameters
   - ❌ LightGBM: **REMOVED** all 6 new parameters, **RESTORED HARD-CODED VALUES**
   - PLS: Changed to range-based component generation
   - **PLS, Ridge, Lasso, ElasticNet**: Unchanged (nothing to remove)

3. **Current HEAD**:
   - Same as d801c91
   - Only XGBoost, CatBoost, SVR, MLP, NeuralBoosted parameters exist from original work
   - **39 parameters missing** from grid generation

### The Vision (User-Confirmed)

**"Expose ALL hyperparameters with single-value defaults that don't change grid size unless user overrides"**

This is architecturally sound and safe:
- Single-value defaults = NO grid explosion
- Users can override for expanded search
- Achieves R parity for scientific research

### Grid Size Proof (No Explosion with Defaults)

| Model | Current Grid | With New Params (Single Defaults) | Grid Size Change |
|-------|--------------|-----------------------------------|------------------|
| PLS | 12 (components 2-13) | 12 × 1 × 1 × 1 = 12 | ✅ **UNCHANGED** |
| Ridge | 5 (alphas) | 5 × 1 × 1 = 5 | ✅ **UNCHANGED** |
| Lasso | 4 (alphas) | 4 × 1 × 1 × 1 = 4 | ✅ **UNCHANGED** |
| ElasticNet | 20 (4 α × 5 l1_ratio) | 20 × 1 × 1 × 1 = 20 | ✅ **UNCHANGED** |
| RandomForest | 6 (2 n_est × 3 depth) | 6 × 1 × 1 × 1 × 1 × 1 × 1 = 6 | ✅ **UNCHANGED** |
| LightGBM | 12 (2 n_est × 2 lr × 3 leaves) | 12 × 1 × 1 × 1 × 1 × 1 × 1 = 12 | ✅ **UNCHANGED** |

**Conclusion**: ZERO RISK of grid explosion with single-value defaults.

---

## 2. FORENSIC ANALYSIS DETAILS

### 2.1 Timeline

**Commit cb1ead6** (Nov 12, 2025 - "feat: Expose all model hyperparameters"):

**RandomForest** - ✅ FULLY IMPLEMENTED:
```python
# Function signature added:
rf_min_samples_split_list=None, rf_min_samples_leaf_list=None, rf_max_features_list=None

# Default loading added:
if rf_min_samples_split_list is None:
    rf_config = get_hyperparameters('RandomForest', tier)
    rf_min_samples_split_list = rf_config.get('min_samples_split', [2])
# ... (similar for other params)

# Grid generation - 5D nested loop:
for n_est in rf_n_trees_list:
    for max_d in rf_max_depth_list:
        for min_split in rf_min_samples_split_list:
            for min_leaf in rf_min_samples_leaf_list:
                for max_feat in rf_max_features_list:
                    RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        max_features=max_feat,
                        random_state=42,
                        n_jobs=-1
                    )
```

**LightGBM** - ✅ FULLY IMPLEMENTED:
```python
# Function signature added:
lightgbm_min_child_samples_list=None, lightgbm_subsample_list=None,
lightgbm_colsample_bytree_list=None, lightgbm_reg_alpha_list=None,
lightgbm_reg_lambda_list=None

# Grid generation - 8D nested loop:
for n_est in lightgbm_n_estimators_list:
    for lr in lightgbm_learning_rates:
        for num_leaves in lightgbm_num_leaves_list:
            for min_child in lightgbm_min_child_samples_list:
                for subsample in lightgbm_subsample_list:
                    for colsample in lightgbm_colsample_bytree_list:
                        for reg_alpha in lightgbm_reg_alpha_list:
                            for reg_lambda in lightgbm_reg_lambda_list:
                                LGBMRegressor(
                                    n_estimators=n_est,
                                    learning_rate=lr,
                                    num_leaves=num_leaves,
                                    min_child_samples=min_child,
                                    subsample=subsample,
                                    colsample_bytree=colsample,
                                    reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda,
                                    ...
                                )
```

**PLS, Ridge, Lasso, ElasticNet** - ❌ NEVER IMPLEMENTED:
- No function signature changes
- No default loading for new params
- No nested loops for new params
- No parameters passed to constructors beyond what exists today

**Commit d801c91** (Nov 11, 2025 - "fix: Unify hyperparameters across tiers"):

**RandomForest** - ❌ REMOVED:
```python
# Function signature REMOVED 3 parameters
# Before: rf_min_samples_split_list=None, rf_min_samples_leaf_list=None, rf_max_features_list=None
# After: (removed)

# Grid generation simplified from 5D to 2D:
for n_est in rf_n_trees_list:
    for max_d in rf_max_depth_list:
        RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42,
            n_jobs=-1
        )
```

**LightGBM** - ❌ REMOVED + HARD-CODED VALUES RESTORED:
```python
# Grid generation simplified from 8D to 3D:
for n_est in lightgbm_n_estimators_list:
    for lr in lightgbm_learning_rates:
        for num_leaves in lightgbm_num_leaves_list:
            LGBMRegressor(
                n_estimators=n_est,
                learning_rate=lr,
                num_leaves=num_leaves,
                min_child_samples=5,      # ❌ HARD-CODED AGAIN!
                subsample=0.8,            # ❌ HARD-CODED AGAIN!
                colsample_bytree=0.8,     # ❌ HARD-CODED AGAIN!
                reg_alpha=0.1,            # ❌ HARD-CODED AGAIN!
                reg_lambda=1.0,           # ❌ HARD-CODED AGAIN!
                ...
            )
```

### 2.2 Current State vs. Desired State

**Current HEAD** (`src/spectral_predict/models.py`):

**PLS** (lines 397-402):
```python
# Current - only n_components
if 'PLS' in enabled_models:
    grids["PLS"] = [
        (PLSRegression(n_components=nc, scale=False), {"n_components": nc})
        for nc in pls_components
    ]

# MISSING: max_iter, tol, algorithm
```

**Ridge** (lines 404-414):
```python
# Current - only alpha
if 'Ridge' in enabled_models:
    ridge_configs = []
    for alpha in ridge_alphas_list:
        ridge_configs.append(
            (
                Ridge(alpha=alpha, random_state=42),
                {"alpha": alpha}
            )
        )

# MISSING: solver, tol
```

**Lasso** (lines 416-426):
```python
# Current - alpha + global max_iter
if 'Lasso' in enabled_models:
    lasso_configs = []
    for alpha in lasso_alphas_list:
        lasso_configs.append(
            (
                Lasso(alpha=alpha, random_state=42, max_iter=max_iter),
                {"alpha": alpha}
            )
        )

# MISSING: selection, tol, max_iter (from list, not global)
```

**ElasticNet** (lines 428-439):
```python
# Current - alpha + l1_ratio + global max_iter
if 'ElasticNet' in enabled_models:
    elasticnet_configs = []
    for alpha in elasticnet_alphas_list:
        for l1_ratio in elasticnet_l1_ratios:
            elasticnet_configs.append(
                (
                    ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=max_iter),
                    {"alpha": alpha, "l1_ratio": l1_ratio}
                )
            )

# MISSING: selection, tol, max_iter (from list)
```

### 2.3 Parameter Count by Model

**Total parameters to implement**: 39

| Model | Current Params | Missing Params | New Params to Add |
|-------|----------------|----------------|-------------------|
| PLS | 1 (n_components) | 3 | max_iter, tol, algorithm |
| Ridge | 1 (alpha) | 2 | solver, tol |
| Lasso | 1 (alpha) | 3 | selection, tol, max_iter |
| ElasticNet | 2 (alpha, l1_ratio) | 3 | selection, tol, max_iter |
| RandomForest | 2 (n_estimators, max_depth) | 6 | min_samples_split, min_samples_leaf, max_features, bootstrap, max_leaf_nodes, min_impurity_decrease |
| LightGBM | 3 (n_estimators, lr, num_leaves) | 6 | max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda |
| XGBoost | 7 (existing) | 2 | min_child_weight, gamma |
| CatBoost | 3 (existing) | 4 | l2_leaf_reg, border_count, bagging_temperature, random_strength |
| SVR | 3 (existing) | 4 | epsilon, degree, coef0, shrinking |
| MLP | 3 (existing) | 5 | activation, solver, batch_size, learning_rate_schedule, momentum |
| NeuralBoosted | 4 (existing) | 1 | subsample |
| PLS-DA | 1 (n_components) | 3 | max_iter, tol, algorithm (PLS stage) |

---

## 3. ARCHITECTURE & PARAMETER FLOW

### 3.1 File Structure

```
dasp/
├── src/spectral_predict/
│   ├── models.py           # Grid generation - NEEDS 39 parameters
│   ├── search.py           # Orchestration - NEEDS to pass 39 parameters
│   ├── model_config.py     # Tier defaults - NEEDS 39 param definitions
│   └── neural_boosted.py   # Custom model - subsample already implemented
├── spectral_predict_gui_optimized.py  # GUI - NEEDS controls + extraction
└── tests/
    └── (test files to be created)
```

### 3.2 Parameter Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Tab 4C: Analysis Configuration                              │
│ - User checks boxes / enters custom values                  │
│ - _create_parameter_grid_control() creates widgets          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ _run_analysis() method                                       │
│ - _extract_parameter_values() extracts from each control    │
│ - Builds parameter lists (e.g., pls_max_iter_list=[500])   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ run_search() in search.py                                   │
│ - Receives all parameter lists                              │
│ - Passes to get_model_grids()                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ get_model_grids() in models.py                             │
│ - Loads defaults from model_config.py (if param is None)   │
│ - Generates nested loops for each model                     │
│ - Creates (model_instance, params_dict) tuples             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Cross-validation & Results                                   │
│ - Each config tested with CV                                │
│ - Params dict saved to Results DataFrame                    │
│ - Displayed in Results tab                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Tab 7C: Model Development (double-click result)            │
│ - Loads params from Results DataFrame                       │
│ - Populates subtab controls                                 │
│ - User can modify and re-run                                │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Grid Generation Architecture

**Pattern for ALL models**:

```python
# 1. Function signature includes parameter lists
def get_model_grids(
    ...
    model_param1_list=None,
    model_param2_list=None,
    ...
):

    # 2. Load defaults from model_config.py if None
    if model_param1_list is None:
        config = get_hyperparameters('ModelName', tier)
        model_param1_list = config.get('param1', [default_value])

    # 3. Generate grid with nested loops
    if 'ModelName' in enabled_models:
        configs = []
        for param1 in model_param1_list:
            for param2 in model_param2_list:
                configs.append(
                    (
                        Model(param1=param1, param2=param2, ...),
                        {"param1": param1, "param2": param2, ...}
                    )
                )
        grids["ModelName"] = configs
```

---

## 4. IMPLEMENTATION SPECIFICATIONS

### 4.1 Backend: models.py

**File**: `src/spectral_predict/models.py`
**Current Lines**: 1256
**Modifications Required**: 7 sections

#### 4.1.1 Function Signature (Line 223)

**CURRENT**:
```python
def get_model_grids(task_type, n_features, max_n_components=8, max_iter=500,
                    n_estimators_list=None, learning_rates=None, rf_n_trees_list=None,
                    rf_max_depth_list=None, ridge_alphas_list=None, lasso_alphas_list=None,
                    xgb_n_estimators_list=None, xgb_learning_rates=None, xgb_max_depths=None,
                    xgb_subsample=None, xgb_colsample_bytree=None, xgb_reg_alpha=None, xgb_reg_lambda=None,
                    elasticnet_alphas_list=None, elasticnet_l1_ratios=None,
                    lightgbm_n_estimators_list=None, lightgbm_learning_rates=None, lightgbm_num_leaves_list=None,
                    catboost_iterations_list=None, catboost_learning_rates=None, catboost_depths=None,
                    svr_kernels=None, svr_C_list=None, svr_gamma_list=None,
                    mlp_hidden_layer_sizes_list=None, mlp_alphas_list=None, mlp_learning_rate_inits=None,
                    tier='standard', enabled_models=None):
```

**ADD AFTER `rf_max_depth_list=None,`**:
```python
                    rf_min_samples_split_list=None, rf_min_samples_leaf_list=None,
                    rf_max_features_list=None, rf_bootstrap_list=None,
                    rf_max_leaf_nodes_list=None, rf_min_impurity_decrease_list=None,
```

**ADD AFTER `ridge_alphas_list=None,`**:
```python
                    ridge_solver_list=None, ridge_tol_list=None,
```

**ADD AFTER `lasso_alphas_list=None,`**:
```python
                    lasso_selection_list=None, lasso_tol_list=None, lasso_max_iter_list=None,
```

**ADD AFTER `elasticnet_l1_ratios=None,`**:
```python
                    elasticnet_selection_list=None, elasticnet_tol_list=None, elasticnet_max_iter_list=None,
```

**ADD AFTER `lightgbm_num_leaves_list=None,`**:
```python
                    lightgbm_max_depth_list=None, lightgbm_min_child_samples_list=None,
                    lightgbm_subsample_list=None, lightgbm_colsample_bytree_list=None,
                    lightgbm_reg_alpha_list=None, lightgbm_reg_lambda_list=None,
```

**ADD (new PLS parameters before ridge)**:
```python
                    pls_max_iter_list=None, pls_tol_list=None, pls_algorithm_list=None,
```

**ADD AFTER `xgb_reg_lambda=None,`**:
```python
                    xgb_min_child_weight=None, xgb_gamma=None,
```

**ADD AFTER `catboost_depths=None,`**:
```python
                    catboost_l2_leaf_reg_list=None, catboost_border_count_list=None,
                    catboost_bagging_temperature_list=None, catboost_random_strength_list=None,
```

**ADD AFTER `svr_gamma_list=None,`**:
```python
                    svr_epsilon_list=None, svr_degree_list=None, svr_coef0_list=None, svr_shrinking_list=None,
```

**ADD AFTER `mlp_learning_rate_inits=None,`**:
```python
                    mlp_activation_list=None, mlp_solver_list=None, mlp_batch_size_list=None,
                    mlp_learning_rate_schedule_list=None, mlp_momentum_list=None,
                    neuralboosted_subsample_list=None,
```

#### 4.1.2 Default Parameter Loading (After line 295)

**ADD AFTER LINE 302** (after existing RF defaults):
```python
    # RandomForest additional defaults (tier-aware)
    if rf_min_samples_split_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_min_samples_split_list = rf_config.get('min_samples_split', [2])
    if rf_min_samples_leaf_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_min_samples_leaf_list = rf_config.get('min_samples_leaf', [1])
    if rf_max_features_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_max_features_list = rf_config.get('max_features', ['sqrt'])
    if rf_bootstrap_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_bootstrap_list = rf_config.get('bootstrap', [True])
    if rf_max_leaf_nodes_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_max_leaf_nodes_list = rf_config.get('max_leaf_nodes', [None])
    if rf_min_impurity_decrease_list is None:
        rf_config = get_hyperparameters('RandomForest', tier)
        rf_min_impurity_decrease_list = rf_config.get('min_impurity_decrease', [0.0])
```

**ADD AFTER PLS components generation** (create new section ~line 395):
```python
    # PLS additional defaults (tier-aware)
    if pls_max_iter_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_max_iter_list = pls_config.get('max_iter', [500])
    if pls_tol_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_tol_list = pls_config.get('tol', [1e-6])
    if pls_algorithm_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_algorithm_list = pls_config.get('algorithm', ['nipals'])
```

**ADD AFTER LINE 307** (after existing Ridge defaults):
```python
    # Ridge additional defaults (tier-aware)
    if ridge_solver_list is None:
        ridge_config = get_hyperparameters('Ridge', tier)
        ridge_solver_list = ridge_config.get('solver', ['auto'])
    if ridge_tol_list is None:
        ridge_config = get_hyperparameters('Ridge', tier)
        ridge_tol_list = ridge_config.get('tol', [1e-4])
```

**ADD AFTER LINE 312** (after existing Lasso defaults):
```python
    # Lasso additional defaults (tier-aware)
    if lasso_selection_list is None:
        lasso_config = get_hyperparameters('Lasso', tier)
        lasso_selection_list = lasso_config.get('selection', ['cyclic'])
    if lasso_tol_list is None:
        lasso_config = get_hyperparameters('Lasso', tier)
        lasso_tol_list = lasso_config.get('tol', [1e-4])
    if lasso_max_iter_list is None:
        lasso_config = get_hyperparameters('Lasso', tier)
        lasso_max_iter_list = lasso_config.get('max_iter', [1000])
```

**ADD AFTER LINE 343** (after existing ElasticNet defaults):
```python
    # ElasticNet additional defaults (tier-aware)
    if elasticnet_selection_list is None:
        en_config = get_hyperparameters('ElasticNet', tier)
        elasticnet_selection_list = en_config.get('selection', ['cyclic'])
    if elasticnet_tol_list is None:
        en_config = get_hyperparameters('ElasticNet', tier)
        elasticnet_tol_list = en_config.get('tol', [1e-4])
    if elasticnet_max_iter_list is None:
        en_config = get_hyperparameters('ElasticNet', tier)
        elasticnet_max_iter_list = en_config.get('max_iter', [1000])
```

**ADD AFTER LINE 354** (after existing LightGBM defaults):
```python
    # LightGBM additional defaults (tier-aware)
    if lightgbm_max_depth_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_max_depth_list = lgbm_config.get('max_depth', [-1])
    if lightgbm_min_child_samples_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_min_child_samples_list = lgbm_config.get('min_child_samples', [20])
    if lightgbm_subsample_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_subsample_list = lgbm_config.get('subsample', [1.0])
    if lightgbm_colsample_bytree_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_colsample_bytree_list = lgbm_config.get('colsample_bytree', [1.0])
    if lightgbm_reg_alpha_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_reg_alpha_list = lgbm_config.get('reg_alpha', [0.0])
    if lightgbm_reg_lambda_list is None:
        lgbm_config = get_hyperparameters('LightGBM', tier)
        lightgbm_reg_lambda_list = lgbm_config.get('reg_lambda', [0.0])
```

**ADD AFTER existing XGBoost defaults** (~line 335):
```python
    # XGBoost additional defaults (tier-aware)
    if xgb_min_child_weight is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_min_child_weight = xgb_config.get('min_child_weight', [1])
    if xgb_gamma is None:
        xgb_config = get_hyperparameters('XGBoost', tier)
        xgb_gamma = xgb_config.get('gamma', [0])
```

**ADD AFTER existing CatBoost defaults** (~line 365):
```python
    # CatBoost additional defaults (tier-aware)
    if catboost_l2_leaf_reg_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_l2_leaf_reg_list = cb_config.get('l2_leaf_reg', [3.0])
    if catboost_border_count_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_border_count_list = cb_config.get('border_count', [254])
    if catboost_bagging_temperature_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_bagging_temperature_list = cb_config.get('bagging_temperature', [1.0])
    if catboost_random_strength_list is None:
        cb_config = get_hyperparameters('CatBoost', tier)
        catboost_random_strength_list = cb_config.get('random_strength', [1.0])
```

**ADD AFTER existing SVR defaults** (~line 377):
```python
    # SVR additional defaults (tier-aware)
    if svr_epsilon_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_epsilon_list = svr_config.get('epsilon', [0.1])
    if svr_degree_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_degree_list = svr_config.get('degree', [3])
    if svr_coef0_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_coef0_list = svr_config.get('coef0', [0.0])
    if svr_shrinking_list is None:
        svr_config = get_hyperparameters('SVR', tier)
        svr_shrinking_list = svr_config.get('shrinking', [True])
```

**ADD AFTER existing MLP defaults** (~line 387):
```python
    # MLP additional defaults (tier-aware)
    if mlp_activation_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_activation_list = mlp_config.get('activation', ['relu'])
    if mlp_solver_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_solver_list = mlp_config.get('solver', ['adam'])
    if mlp_batch_size_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_batch_size_list = mlp_config.get('batch_size', ['auto'])
    if mlp_learning_rate_schedule_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_learning_rate_schedule_list = mlp_config.get('learning_rate_schedule', ['constant'])
    if mlp_momentum_list is None:
        mlp_config = get_hyperparameters('MLP', tier)
        mlp_momentum_list = mlp_config.get('momentum', [0.9])
```

**ADD (new NeuralBoosted subsample)**:
```python
    # NeuralBoosted subsample (tier-aware)
    if neuralboosted_subsample_list is None:
        nb_config = get_hyperparameters('NeuralBoosted', tier)
        neuralboosted_subsample_list = nb_config.get('subsample', [1.0])
```

#### 4.1.3 Grid Generation Updates

This section contains complete replacement code for each model's grid generation.

**PLS Regression** (REPLACE lines 397-402):
```python
        # PLS Regression (with all hyperparameters)
        if 'PLS' in enabled_models:
            pls_configs = []
            for nc in pls_components:
                for max_iter in pls_max_iter_list:
                    for tol in pls_tol_list:
                        for algorithm in pls_algorithm_list:
                            pls_configs.append(
                                (
                                    PLSRegression(
                                        n_components=nc,
                                        scale=False,
                                        max_iter=max_iter,
                                        tol=tol,
                                        algorithm=algorithm
                                    ),
                                    {
                                        "n_components": nc,
                                        "max_iter": max_iter,
                                        "tol": tol,
                                        "algorithm": algorithm
                                    }
                                )
                            )
            grids["PLS"] = pls_configs
```

**Ridge Regression** (REPLACE lines 404-414):
```python
        # Ridge Regression (with all hyperparameters)
        if 'Ridge' in enabled_models:
            ridge_configs = []
            for alpha in ridge_alphas_list:
                for solver in ridge_solver_list:
                    for tol in ridge_tol_list:
                        ridge_configs.append(
                            (
                                Ridge(
                                    alpha=alpha,
                                    solver=solver,
                                    tol=tol,
                                    random_state=42
                                ),
                                {
                                    "alpha": alpha,
                                    "solver": solver,
                                    "tol": tol
                                }
                            )
                        )
            grids["Ridge"] = ridge_configs
```

**Lasso Regression** (REPLACE lines 416-426):
```python
        # Lasso Regression (with all hyperparameters)
        if 'Lasso' in enabled_models:
            lasso_configs = []
            for alpha in lasso_alphas_list:
                for selection in lasso_selection_list:
                    for tol in lasso_tol_list:
                        for max_iter in lasso_max_iter_list:
                            lasso_configs.append(
                                (
                                    Lasso(
                                        alpha=alpha,
                                        selection=selection,
                                        tol=tol,
                                        max_iter=max_iter,
                                        random_state=42
                                    ),
                                    {
                                        "alpha": alpha,
                                        "selection": selection,
                                        "tol": tol,
                                        "max_iter": max_iter
                                    }
                                )
                            )
            grids["Lasso"] = lasso_configs
```

**ElasticNet Regression** (REPLACE lines 428-439):
```python
        # ElasticNet Regression (with all hyperparameters)
        if 'ElasticNet' in enabled_models:
            elasticnet_configs = []
            for alpha in elasticnet_alphas_list:
                for l1_ratio in elasticnet_l1_ratios:
                    for selection in elasticnet_selection_list:
                        for tol in elasticnet_tol_list:
                            for max_iter in elasticnet_max_iter_list:
                                elasticnet_configs.append(
                                    (
                                        ElasticNet(
                                            alpha=alpha,
                                            l1_ratio=l1_ratio,
                                            selection=selection,
                                            tol=tol,
                                            max_iter=max_iter,
                                            random_state=42
                                        ),
                                        {
                                            "alpha": alpha,
                                            "l1_ratio": l1_ratio,
                                            "selection": selection,
                                            "tol": tol,
                                            "max_iter": max_iter
                                        }
                                    )
                                )
            grids["ElasticNet"] = elasticnet_configs
```

**RandomForest Regression** (REPLACE lines 604-646):
```python
        # Random Forest Regressor (with all hyperparameters)
        if 'RandomForest' in enabled_models:
            rf_configs = []
            for n_est in rf_n_trees_list:
                for max_d in rf_max_depth_list:
                    for min_split in rf_min_samples_split_list:
                        for min_leaf in rf_min_samples_leaf_list:
                            for max_feat in rf_max_features_list:
                                for bootstrap in rf_bootstrap_list:
                                    for max_leaf_nodes in rf_max_leaf_nodes_list:
                                        for min_impurity in rf_min_impurity_decrease_list:
                                            rf_configs.append(
                                                (
                                                    RandomForestRegressor(
                                                        n_estimators=n_est,
                                                        max_depth=max_d,
                                                        min_samples_split=min_split,
                                                        min_samples_leaf=min_leaf,
                                                        max_features=max_feat,
                                                        bootstrap=bootstrap,
                                                        max_leaf_nodes=max_leaf_nodes,
                                                        min_impurity_decrease=min_impurity,
                                                        random_state=42,
                                                        n_jobs=-1
                                                    ),
                                                    {
                                                        "n_estimators": n_est,
                                                        "max_depth": max_d,
                                                        "min_samples_split": min_split,
                                                        "min_samples_leaf": min_leaf,
                                                        "max_features": max_feat,
                                                        "bootstrap": bootstrap,
                                                        "max_leaf_nodes": max_leaf_nodes,
                                                        "min_impurity_decrease": min_impurity
                                                    }
                                                )
                                            )
            grids["RandomForest"] = rf_configs
```

**LightGBM Regression** (REPLACE lines 647-680 - remove hard-coded values):
```python
        # LightGBM Regression (with all hyperparameters, NO hard-coded values)
        if 'LightGBM' in enabled_models:
            lgbm_configs = []
            for n_est in lightgbm_n_estimators_list:
                for lr in lightgbm_learning_rates:
                    for num_leaves in lightgbm_num_leaves_list:
                        for max_depth in lightgbm_max_depth_list:
                            for min_child_samples in lightgbm_min_child_samples_list:
                                for subsample in lightgbm_subsample_list:
                                    for colsample_bytree in lightgbm_colsample_bytree_list:
                                        for reg_alpha in lightgbm_reg_alpha_list:
                                            for reg_lambda in lightgbm_reg_lambda_list:
                                                lgbm_configs.append(
                                                    (
                                                        LGBMRegressor(
                                                            n_estimators=n_est,
                                                            learning_rate=lr,
                                                            num_leaves=num_leaves,
                                                            max_depth=max_depth,
                                                            min_child_samples=min_child_samples,
                                                            subsample=subsample,
                                                            colsample_bytree=colsample_bytree,
                                                            reg_alpha=reg_alpha,
                                                            reg_lambda=reg_lambda,
                                                            random_state=42,
                                                            n_jobs=-1,
                                                            verbosity=-1
                                                        ),
                                                        {
                                                            "n_estimators": n_est,
                                                            "learning_rate": lr,
                                                            "num_leaves": num_leaves,
                                                            "max_depth": max_depth,
                                                            "min_child_samples": min_child_samples,
                                                            "subsample": subsample,
                                                            "colsample_bytree": colsample_bytree,
                                                            "reg_alpha": reg_alpha,
                                                            "reg_lambda": reg_lambda
                                                        }
                                                    )
                                                )
            grids["LightGBM"] = lgbm_configs
```

**XGBoost Regression** (UPDATE existing - add 2 more nested loops):
Find the existing XGBoost section and add nested loops for `min_child_weight` and `gamma` within the existing structure.

**CatBoost Regression** (UPDATE existing - add 4 more nested loops):
Find the existing CatBoost section and add nested loops for `l2_leaf_reg`, `border_count`, `bagging_temperature`, `random_strength`.

**SVR** (UPDATE existing - add conditional loops for new params):
Update the SVR section to include `epsilon`, `degree`, `coef0`, `shrinking` with kernel-specific conditional logic.

**MLP Regression** (UPDATE existing - add 5 more nested loops):
Add loops for `activation`, `solver`, `batch_size`, `learning_rate_schedule`, and conditionally `momentum` (only when solver='sgd').

**NeuralBoosted** (UPDATE existing - add subsample loop):
Add nested loop for `subsample` parameter.

**Classification Models**: Apply same changes to classification sections (lines 800+).

---

### 4.2 Backend: search.py

**File**: `src/spectral_predict/search.py`
**Current Lines**: 198
**Modifications Required**: 2 sections

#### 4.2.1 Function Signature (Line 22)

**ADD all 39 new parameters** to the `run_search()` function signature. Insert after corresponding existing parameters.

Example:
```python
def run_search(X, y, task_type, folds=5, variable_penalty=3, complexity_penalty=0.1,
               max_n_components=8, max_iter=500, models_to_test=None, preprocessing_methods=None,
               window_sizes=None, n_estimators_list=None, learning_rates=None,

               # RandomForest parameters
               rf_n_trees_list=None, rf_max_depth_list=None,
               rf_min_samples_split_list=None, rf_min_samples_leaf_list=None,  # NEW
               rf_max_features_list=None, rf_bootstrap_list=None,              # NEW
               rf_max_leaf_nodes_list=None, rf_min_impurity_decrease_list=None, # NEW

               # PLS parameters
               pls_max_iter_list=None, pls_tol_list=None, pls_algorithm_list=None,  # NEW

               # Ridge parameters
               ridge_alphas_list=None,
               ridge_solver_list=None, ridge_tol_list=None,  # NEW

               # ... continue for all models
               tier='standard'):
```

#### 4.2.2 get_model_grids() Call (Line 183)

**PASS all 39 new parameters** to the `get_model_grids()` call. Match the order used in models.py signature.

Example:
```python
    model_grids = get_model_grids(
        task_type, n_features, safe_max_components, max_iter,
        n_estimators_list=n_estimators_list, learning_rates=learning_rates,
        rf_n_trees_list=rf_n_trees_list, rf_max_depth_list=rf_max_depth_list,
        rf_min_samples_split_list=rf_min_samples_split_list,  # NEW
        rf_min_samples_leaf_list=rf_min_samples_leaf_list,    # NEW
        rf_max_features_list=rf_max_features_list,            # NEW
        rf_bootstrap_list=rf_bootstrap_list,                  # NEW
        rf_max_leaf_nodes_list=rf_max_leaf_nodes_list,        # NEW
        rf_min_impurity_decrease_list=rf_min_impurity_decrease_list,  # NEW
        pls_max_iter_list=pls_max_iter_list,  # NEW
        pls_tol_list=pls_tol_list,            # NEW
        pls_algorithm_list=pls_algorithm_list,  # NEW
        # ... continue for all models
        tier=tier, enabled_models=models_to_test
    )
```

---

### 4.3 Configuration: model_config.py

**File**: `src/spectral_predict/model_config.py`
**Current Lines**: 450
**Modifications Required**: Add parameters to all tier dictionaries

#### 4.3.1 PLS (Lines 88-101)

**ADD to each tier (standard, comprehensive, quick)**:
```python
'PLS': {
    'standard': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
        'max_iter': [500],      # NEW - single value = no grid expansion
        'tol': [1e-6],          # NEW - single value
        'algorithm': ['nipals'], # NEW - single value
        'note': 'Grid size: 12×1×1×1 = 12 configs (UNCHANGED)'
    },
    'comprehensive': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
        'max_iter': [500, 1000],      # NEW - 2 values for experimentation
        'tol': [1e-7, 1e-6, 1e-5],    # NEW - 3 values
        'algorithm': ['nipals', 'svd'], # NEW - 2 values
        'note': 'Grid size: 12×2×3×2 = 144 configs'
    },
    'quick': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
        'max_iter': [500],      # NEW - single value
        'tol': [1e-6],          # NEW - single value
        'algorithm': ['nipals'], # NEW - single value
        'note': 'Grid size: 12×1×1×1 = 12 configs (UNCHANGED)'
    }
}
```

#### 4.3.2 Ridge (Lines 103-116)

**ADD to each tier**:
```python
'Ridge': {
    'standard': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'solver': ['auto'],  # NEW - single value = no grid expansion
        'tol': [1e-4],       # NEW - single value
        'note': 'Grid size: 5×1×1 = 5 configs (UNCHANGED)'
    },
    'comprehensive': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'lsqr'],  # NEW - 3 values
        'tol': [1e-5, 1e-4, 1e-3],          # NEW - 3 values
        'note': 'Grid size: 5×3×3 = 45 configs'
    },
    'quick': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'solver': ['auto'],  # NEW - single value
        'tol': [1e-4],       # NEW - single value
        'note': 'Grid size: 5×1×1 = 5 configs (UNCHANGED)'
    }
}
```

#### 4.3.3 Lasso (Lines 304-318)

**ADD to each tier**:
```python
'Lasso': {
    'standard': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'selection': ['cyclic'],  # NEW - single value
        'tol': [1e-4],            # NEW - single value
        'max_iter': [1000],       # NEW - single value
        'note': 'Grid size: 4×1×1×1 = 4 configs (UNCHANGED)'
    },
    'comprehensive': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'selection': ['cyclic', 'random'],  # NEW - 2 values
        'tol': [1e-5, 1e-4, 1e-3],          # NEW - 3 values
        'max_iter': [1000, 2000],           # NEW - 2 values
        'note': 'Grid size: 4×2×3×2 = 48 configs'
    },
    'quick': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'selection': ['cyclic'],  # NEW - single value
        'tol': [1e-4],            # NEW - single value
        'max_iter': [1000],       # NEW - single value
        'note': 'Grid size: 4×1×1×1 = 4 configs (UNCHANGED)'
    }
}
```

#### 4.3.4 ElasticNet (Lines 118-134)

**ADD to each tier**:
```python
'ElasticNet': {
    'standard': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'selection': ['cyclic'],  # NEW - single value
        'tol': [1e-4],            # NEW - single value
        'max_iter': [1000],       # NEW - single value
        'note': 'Grid size: 4×5×1×1×1 = 20 configs (UNCHANGED)'
    },
    'comprehensive': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'selection': ['cyclic', 'random'],  # NEW - 2 values
        'tol': [1e-5, 1e-4, 1e-3],          # NEW - 3 values
        'max_iter': [1000, 2000],           # NEW - 2 values
        'note': 'Grid size: 4×5×2×3×2 = 240 configs'
    },
    'quick': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'selection': ['cyclic'],  # NEW - single value
        'tol': [1e-4],            # NEW - single value
        'max_iter': [1000],       # NEW - single value
        'note': 'Grid size: 4×5×1×1×1 = 20 configs (UNCHANGED)'
    }
}
```

#### 4.3.5 RandomForest, LightGBM, XGBoost, CatBoost, SVR, MLP, NeuralBoosted

Apply same pattern: Add new parameters with single-value defaults for standard/quick, 2-3 values for comprehensive.

---

### 4.4 GUI: Tab 4C (spectral_predict_gui_optimized.py)

**File**: `spectral_predict_gui_optimized.py`
**Estimated Lines to Add**: ~500 lines (GUI controls) + ~400 lines (extraction)

#### 4.4.1 GUI Control Creation

**Location**: Within Tab 4C setup (around lines 2200-2960)

**Pattern for each parameter**:
```python
# Example: PLS max_iter
self.pls_max_iter_control = self._create_parameter_grid_control(
    parent=pls_section_frame,
    param_name='max_iter',
    param_label='Max Iterations',
    checkbox_values=[500, 1000, 2000, 5000],
    default_checked=[500],
    is_float=False,
    help_text='Maximum iterations for NIPALS algorithm'
)
```

**Create collapsible sections** for each model using existing patterns.

#### 4.4.2 Parameter Extraction in _run_analysis()

**Location**: Around line 6723 in `_run_analysis_thread()` method

**Pattern for each parameter**:
```python
# Example: PLS parameters
pls_max_iter_list = self._extract_parameter_values(
    self.pls_max_iter_control,
    'max_iter',
    is_float=False
) if hasattr(self, 'pls_max_iter_control') else None

pls_tol_list = self._extract_parameter_values(
    self.pls_tol_control,
    'tol',
    is_float=True
) if hasattr(self, 'pls_tol_control') else None

pls_algorithm_list = self._extract_parameter_values(
    self.pls_algorithm_control,
    'algorithm',
    allow_string_values=True
) if hasattr(self, 'pls_algorithm_control') else None
```

Repeat for all 39 parameters.

#### 4.4.3 run_search() Call Update

**Location**: Around line 7344

**ADD all 39 parameters** to the run_search() call:
```python
results_df = run_search(
    X, y, task_type,
    folds=cv_folds,
    variable_penalty=variable_penalty,
    complexity_penalty=complexity_penalty,
    max_n_components=max_n_components,
    max_iter=max_iter,
    models_to_test=models_to_test,

    # PLS parameters
    pls_max_iter_list=pls_max_iter_list,  # NEW
    pls_tol_list=pls_tol_list,            # NEW
    pls_algorithm_list=pls_algorithm_list,  # NEW

    # RandomForest parameters
    rf_n_trees_list=rf_n_trees_list,
    rf_max_depth_list=rf_max_depth_list,
    rf_min_samples_split_list=rf_min_samples_split_list,  # NEW
    rf_min_samples_leaf_list=rf_min_samples_leaf_list,    # NEW
    rf_max_features_list=rf_max_features_list,            # NEW
    rf_bootstrap_list=rf_bootstrap_list,                  # NEW
    rf_max_leaf_nodes_list=rf_max_leaf_nodes_list,        # NEW
    rf_min_impurity_decrease_list=rf_min_impurity_decrease_list,  # NEW

    # ... continue for all 39 parameters
    tier=tier
)
```

#### 4.4.4 Conditional Logic

**MLP Momentum** - Enable only when solver='sgd':
```python
def _on_mlp_solver_change(self):
    """Enable momentum controls only when SGD solver is selected"""
    # Check if SGD is checked
    sgd_checked = self.mlp_solver_sgd.get() if hasattr(self, 'mlp_solver_sgd') else False

    # Enable/disable momentum controls
    if hasattr(self, 'mlp_momentum_control'):
        # Enable or disable momentum checkboxes and custom entry
        state = 'normal' if sgd_checked else 'disabled'
        # Update widget states
```

**SVR Kernel-Specific** - Enable parameters based on kernel:
```python
def _on_svr_kernel_change(self):
    """Enable/disable SVR parameters based on kernel selection"""
    # Determine which kernels are selected
    rbf_checked = self.svr_kernel_rbf.get() if hasattr(self, 'svr_kernel_rbf') else False
    poly_checked = self.svr_kernel_poly.get() if hasattr(self, 'svr_kernel_poly') else False
    sigmoid_checked = self.svr_kernel_sigmoid.get() if hasattr(self, 'svr_kernel_sigmoid') else False
    linear_checked = self.svr_kernel_linear.get() if hasattr(self, 'svr_kernel_linear') else False

    # degree: Enable only for poly
    # coef0: Enable for poly OR sigmoid
    # gamma: Disable for linear only
    # Update widget states accordingly
```

Bind these callbacks to the relevant checkboxes.

---

### 4.5 GUI: Tab 7C (spectral_predict_gui_optimized.py)

**File**: `spectral_predict_gui_optimized.py`
**Estimated Lines to Add**: ~1200 lines

#### 4.5.1 Notebook Architecture

**Location**: Tab 7 setup (around line 8800)

**REPLACE existing Tab 7C Frame with Notebook**:
```python
# Tab 7C: Configuration - REPLACE entire section
self.tab_7c = ttk.Frame(tab_7_notebook)
tab_7_notebook.add(self.tab_7c, text="C. Configuration")

# Create notebook for model-specific configuration
self.tab_7c_model_notebook = ttk.Notebook(self.tab_7c)
self.tab_7c_model_notebook.pack(fill='both', expand=True, padx=10, pady=10)

# Create 12 model subtabs
self.tab_7c_pls = ttk.Frame(self.tab_7c_model_notebook)
self.tab_7c_model_notebook.add(self.tab_7c_pls, text="PLS")

self.tab_7c_ridge = ttk.Frame(self.tab_7c_model_notebook)
self.tab_7c_model_notebook.add(self.tab_7c_ridge, text="Ridge")

# ... create all 12 subtabs
```

#### 4.5.2 Auto-Navigation

**Bind model dropdown to switch subtabs**:
```python
def _on_tab7_model_change(self, *args):
    """Switch Tab 7C subtab based on selected model"""
    model_name = self.model_type_var.get()
    model_to_tab_index = {
        'PLS': 0,
        'Ridge': 1,
        'Lasso': 2,
        'ElasticNet': 3,
        'RandomForest': 4,
        'XGBoost': 5,
        'LightGBM': 6,
        'CatBoost': 7,
        'SVR': 8,
        'MLP': 9,
        'NeuralBoosted': 10,
        'PLS-DA': 11
    }
    if model_name in model_to_tab_index:
        self.tab_7c_model_notebook.select(model_to_tab_index[model_name])

# Bind to model dropdown
self.model_type_var.trace('w', self._on_tab7_model_change)
```

#### 4.5.3 Subtab Implementation (Example - PLS)

**For each of 12 models**, create parameter controls:
```python
# PLS Subtab
pls_frame = ttk.Frame(self.tab_7c_pls)
pls_frame.pack(fill='both', expand=True, padx=10, pady=10)

# n_components
self.tab7c_pls_n_components_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='n_components',
    param_label='Number of Components',
    checkbox_values=[2, 4, 6, 8, 10, 12, 16, 20],
    default_checked=[10],
    is_float=False,
    help_text='Number of PLS components to use'
)

# max_iter
self.tab7c_pls_max_iter_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='max_iter',
    param_label='Max Iterations',
    checkbox_values=[500, 1000, 2000],
    default_checked=[500],
    is_float=False,
    help_text='Maximum iterations for NIPALS algorithm'
)

# tol
self.tab7c_pls_tol_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='tol',
    param_label='Convergence Tolerance',
    checkbox_values=[1e-7, 1e-6, 1e-5, 1e-4],
    default_checked=[1e-6],
    is_float=True,
    help_text='Tolerance for stopping criterion'
)

# algorithm
self.tab7c_pls_algorithm_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='algorithm',
    param_label='Algorithm',
    checkbox_values=['nipals', 'svd'],
    default_checked=['nipals'],
    allow_string_values=True,
    help_text='NIPALS = iterative, SVD = direct decomposition'
)
```

Repeat for all 12 models.

#### 4.5.4 Parameter Extraction in _run_refined_model()

**Location**: Around line 9100

**Create extraction helpers** for each model:
```python
def _extract_tab7c_pls_params(self):
    """Extract PLS parameters from Tab 7C PLS subtab"""
    params = {}

    if hasattr(self, 'tab7c_pls_n_components_control'):
        n_components_list = self._extract_parameter_values(
            self.tab7c_pls_n_components_control, 'n_components', is_float=False
        )
        if n_components_list:
            params['n_components'] = n_components_list[0]  # Use first value for single model

    if hasattr(self, 'tab7c_pls_max_iter_control'):
        max_iter_list = self._extract_parameter_values(
            self.tab7c_pls_max_iter_control, 'max_iter', is_float=False
        )
        if max_iter_list:
            params['max_iter'] = max_iter_list[0]

    if hasattr(self, 'tab7c_pls_tol_control'):
        tol_list = self._extract_parameter_values(
            self.tab7c_pls_tol_control, 'tol', is_float=True
        )
        if tol_list:
            params['tol'] = tol_list[0]

    if hasattr(self, 'tab7c_pls_algorithm_control'):
        algorithm_list = self._extract_parameter_values(
            self.tab7c_pls_algorithm_control, 'algorithm', allow_string_values=True
        )
        if algorithm_list:
            params['algorithm'] = algorithm_list[0]

    return params
```

**Update _run_refined_model()** to use extraction:
```python
# Extract params based on model type
model_name = self.model_type_var.get()
params_to_apply = {}

if model_name == 'PLS':
    params_to_apply = self._extract_tab7c_pls_params()
elif model_name == 'Ridge':
    params_to_apply = self._extract_tab7c_ridge_params()
elif model_name == 'RandomForest':
    params_to_apply = self._extract_tab7c_rf_params()
# ... etc for all 12 models

# Apply params to model
if params_to_apply:
    try:
        model.set_params(**params_to_apply)
        print(f"DEBUG: Applied Tab 7C parameters: {params_to_apply}")
    except Exception as e:
        print(f"WARNING: Failed to apply Tab 7C parameters: {e}")
```

#### 4.5.5 Results → Tab 7C Loading

**Update _load_model_for_refinement()** to populate Tab 7C controls:
```python
def _load_model_for_refinement(self, config):
    """Load model configuration from Results into Tab 7C"""
    # ... existing code ...

    # Parse Params column
    raw_params = config.get('Params', {})
    if isinstance(raw_params, str):
        try:
            params_dict = ast.literal_eval(raw_params)
        except:
            params_dict = {}
    else:
        params_dict = raw_params

    # Auto-navigate to correct subtab
    model_name = config.get('Model')
    self._on_tab7_model_change()  # Trigger navigation

    # Populate controls based on model type
    if model_name == 'PLS':
        self._populate_tab7c_pls_controls(params_dict)
    elif model_name == 'Ridge':
        self._populate_tab7c_ridge_controls(params_dict)
    # ... etc
```

**Create population helpers**:
```python
def _populate_tab7c_pls_controls(self, params):
    """Populate PLS subtab controls from loaded parameters"""
    # n_components
    if 'n_components' in params and hasattr(self, 'tab7c_pls_n_components_control'):
        # Check the checkbox for this value
        # Or populate custom entry
        pass

    # max_iter
    if 'max_iter' in params and hasattr(self, 'tab7c_pls_max_iter_control'):
        # Populate control
        pass

    # Continue for all parameters
```

---

## 5. AGENT ASSIGNMENT STRATEGY

### 5.1 Agent Team Structure

**10 specialized agents** organized into 8 waves for optimal parallel execution.

#### Wave 1: Backend Foundation (Parallel - 2.5 hours)

**Agent A1: Linear Models Backend Specialist**
- Models: PLS, Ridge, Lasso, ElasticNet
- Files: models.py (function signature, default loading, grid generation), model_config.py
- Parameters: 11 total (3 + 2 + 3 + 3)
- Deliverable: Complete backend implementation for 4 linear models

**Agent A2: Tree Models Backend Specialist**
- Models: RandomForest
- Files: models.py (restore 6 parameters), model_config.py
- Parameters: 6 total
- Deliverable: Restore RandomForest to full 8D grid

**Agent A3: Boosting Models Backend Specialist**
- Models: LightGBM, XGBoost, CatBoost
- Files: models.py (restore/add parameters, remove hard-coded), model_config.py
- Parameters: 12 total (6 + 2 + 4)
- Deliverable: Complete boosting model implementations

#### Wave 2: Neural/SVM Backend (Sequential - 1.5 hours)

**Agent B1: Neural & SVM Backend Specialist**
- Models: MLP, SVR, NeuralBoosted
- Files: models.py, neural_boosted.py, model_config.py
- Parameters: 10 total (5 + 4 + 1)
- Deliverable: Complete neural/SVM implementations with conditional logic

**Dependency**: Wave 1 complete (avoid merge conflicts)

#### Wave 3: Search Integration (Sequential - 1 hour)

**Agent C1: Search Integration Specialist**
- Files: search.py
- Task: Update run_search() signature and get_model_grids() call with all 39 parameters
- Deliverable: Complete backend integration

**Dependency**: Waves 1 & 2 complete (needs all parameter names)

#### Wave 4: Tab 4C GUI (Parallel - 2 hours)

**Agent D1: Tab 4C Linear & Tree GUI**
- Models: PLS, Ridge, Lasso, ElasticNet, RandomForest
- Task: Create GUI controls for 17 parameters
- Deliverable: Collapsible sections with all controls

**Agent D2: Tab 4C Boosting GUI**
- Models: XGBoost, LightGBM, CatBoost
- Task: Create GUI controls for 12 parameters
- Deliverable: Collapsible sections with all controls

**Agent D3: Tab 4C Neural/SVM GUI**
- Models: MLP, SVR, NeuralBoosted
- Task: Create GUI controls for 10 parameters
- Deliverable: Collapsible sections with all controls

**Dependency**: Wave 3 complete (backend must exist)

#### Wave 5: Tab 4C Integration (Sequential - 2.5 hours)

**Agent E1: Tab 4C Integration & Conditional Logic**
- Task: Parameter extraction for all 39 parameters, run_search() call update, conditional logic
- Deliverable: Complete Tab 4C → backend parameter flow

**Dependency**: Wave 4 complete (GUI controls must exist)

#### Wave 6: Tab 7C Subtabs (Parallel - 3.5 hours)

**Agent F1: Tab 7C Architecture + First 6 Models**
- Task: Create notebook, auto-navigation, subtabs for PLS, Ridge, Lasso, ElasticNet, RF, XGBoost
- Deliverable: 6 subtabs with all controls

**Agent F2: Tab 7C Last 6 Models**
- Task: Subtabs for LightGBM, CatBoost, SVR, MLP, NeuralBoosted, PLS-DA
- Deliverable: 6 subtabs with all controls

**Dependency**: Wave 3 complete (backend must exist)

#### Wave 7: Tab 7C Integration (Sequential - 2 hours)

**Agent G1: Tab 7C Integration Specialist**
- Task: Parameter extraction helpers, _run_refined_model() update, Results loading
- Deliverable: Complete Tab 7C → execution flow

**Dependency**: Wave 6 complete (subtabs must exist)

#### Wave 8: Testing (Sequential - 3 hours)

**Agent H1: Testing & Validation Specialist**
- Task: Grid size tests, integration tests, conditional logic tests, backward compatibility
- Deliverable: Complete test suite with all tests passing

**Dependency**: Waves 5 & 7 complete (all functionality implemented)

### 5.2 Execution Timeline

```
Hour 0-2.5:   Wave 1 (A1, A2, A3 in parallel)
Hour 2.5-4:   Wave 2 (B1)
Hour 4-5:     Wave 3 (C1)
Hour 5-7:     Wave 4 (D1, D2, D3 in parallel)
Hour 7-9.5:   Wave 5 (E1)
Hour 5-8.5:   Wave 6 (F1, F2 in parallel - can overlap with Waves 4 & 5)
Hour 9.5-11.5: Wave 7 (G1)
Hour 11.5-14.5: Wave 8 (H1)
```

**Total Wall Clock Time**: ~14.5 hours

---

## 6. TESTING REQUIREMENTS

### 6.1 Grid Size Validation Tests

**Purpose**: Verify single-value defaults maintain current grid sizes

**Test Script** (`tests/test_grid_sizes.py`):
```python
def test_pls_grid_size_unchanged():
    """Verify PLS grid size unchanged with new params"""
    # Before (current): 12 components
    # After (with defaults): 12 × 1 × 1 × 1 = 12
    config = get_hyperparameters('PLS', 'standard')
    assert len(config['max_iter']) == 1
    assert len(config['tol']) == 1
    assert len(config['algorithm']) == 1
    # Grid size = 12 × 1 × 1 × 1 = 12 (unchanged)
```

Repeat for all 12 models.

### 6.2 Parameter Flow Integration Tests

**Test**: Parameters flow correctly from GUI → backend → results

```python
def test_pls_parameter_flow():
    """Test PLS parameters flow through entire system"""
    # 1. Create mock GUI state
    pls_max_iter_list = [1000]
    pls_tol_list = [1e-5]
    pls_algorithm_list = ['svd']

    # 2. Call run_search with params
    results = run_search(..., pls_max_iter_list=pls_max_iter_list, ...)

    # 3. Verify results contain params
    assert 'max_iter' in results.iloc[0]['Params']
    assert results.iloc[0]['Params']['max_iter'] == 1000
```

### 6.3 Conditional Logic Tests

**Test**: MLP momentum enabled only for SGD

```python
def test_mlp_momentum_conditional():
    """Test momentum only applied when solver=sgd"""
    # Test 1: solver='adam', momentum should be ignored
    grid = get_model_grids(..., mlp_solver_list=['adam'], mlp_momentum_list=[0.9])
    # Verify momentum not in params

    # Test 2: solver='sgd', momentum should be applied
    grid = get_model_grids(..., mlp_solver_list=['sgd'], mlp_momentum_list=[0.9])
    # Verify momentum in params
```

**Test**: SVR kernel-specific parameters

```python
def test_svr_kernel_conditional():
    """Test degree only used with poly kernel"""
    # Test with poly kernel - degree should be present
    grid = get_model_grids(..., svr_kernels=['poly'], svr_degree_list=[3])
    # Verify degree in params

    # Test with rbf kernel - degree should be absent
    grid = get_model_grids(..., svr_kernels=['rbf'], svr_degree_list=[3])
    # Verify degree NOT in params
```

### 6.4 Backward Compatibility Tests

**Test**: Old results (without new params) load correctly

```python
def test_load_old_results():
    """Test loading results from before param implementation"""
    old_result = {
        'Model': 'PLS',
        'Params': "{'n_components': 10}",  # No max_iter, tol, algorithm
        # ... other fields
    }

    # Load into Tab 7C
    app._load_model_for_refinement(old_result)

    # Verify defaults are used for missing params
    # Verify no errors
```

---

## 7. SUCCESS CRITERIA

### 7.1 Backend Checklist

- [ ] All 39 parameters added to `models.py` function signature
- [ ] All 39 parameters have default loading from `model_config.py`
- [ ] All 12 models have updated grid generation with nested loops
- [ ] All parameters passed to model constructors
- [ ] LightGBM hard-coded values removed
- [ ] `search.py` updated with all 39 parameters
- [ ] `model_config.py` has all 39 parameters in all 3 tiers
- [ ] Single-value defaults for standard/quick tiers
- [ ] 2-3 value defaults for comprehensive tier
- [ ] All syntax validations pass

### 7.2 GUI Checklist

**Tab 4C**:
- [ ] GUI controls for all 39 parameters
- [ ] Controls use `_create_parameter_grid_control()` helper
- [ ] Collapsible sections for each model
- [ ] Parameter extraction in `_run_analysis()` for all 39 params
- [ ] `run_search()` call passes all 39 parameters
- [ ] MLP momentum conditional logic implemented
- [ ] SVR kernel conditional logic implemented
- [ ] Conditional logic callbacks bound to controls

**Tab 7C**:
- [ ] ttk.Notebook replaces current Frame
- [ ] 12 model subtabs created
- [ ] Auto-navigation based on model dropdown
- [ ] All parameters have controls in respective subtabs
- [ ] Parameter extraction helpers for all 12 models
- [ ] `_run_refined_model()` extracts and applies params
- [ ] Results → Tab 7C loading populates controls
- [ ] Auto-navigates to correct subtab on load

### 7.3 Testing Checklist

- [ ] Grid size tests pass (sizes unchanged with defaults)
- [ ] Parameter flow tests pass (GUI → backend → results)
- [ ] Conditional logic tests pass (MLP, SVR)
- [ ] Backward compatibility tests pass
- [ ] Integration tests pass (full workflow)
- [ ] Syntax validation passes for all files
- [ ] No regressions in existing functionality

### 7.4 Acceptance Criteria

- [ ] User can see all 39 parameters in Tab 4C
- [ ] Default selections match tier defaults (single values)
- [ ] User can override any parameter with custom values
- [ ] Grid size unchanged when using defaults
- [ ] Grid expands correctly when user adds values
- [ ] Tab 7C has 12 subtabs, one per model
- [ ] Selecting model auto-switches to correct subtab
- [ ] Double-clicking result populates Tab 7C controls
- [ ] Modified params in Tab 7C execute correctly
- [ ] Results CSV includes all parameter values
- [ ] Old results load without errors

---

## 8. REFERENCE CODE SNIPPETS

### 8.1 Complete PLS Implementation Example

**models.py - Function Signature Addition**:
```python
def get_model_grids(task_type, n_features, max_n_components=8, max_iter=500,
                    n_estimators_list=None, learning_rates=None, rf_n_trees_list=None,
                    rf_max_depth_list=None,
                    pls_max_iter_list=None, pls_tol_list=None, pls_algorithm_list=None,  # ADD HERE
                    ridge_alphas_list=None, lasso_alphas_list=None,
                    ...
```

**models.py - Default Loading**:
```python
    # PLS additional defaults (tier-aware)
    if pls_max_iter_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_max_iter_list = pls_config.get('max_iter', [500])
    if pls_tol_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_tol_list = pls_config.get('tol', [1e-6])
    if pls_algorithm_list is None:
        pls_config = get_hyperparameters('PLS', tier)
        pls_algorithm_list = pls_config.get('algorithm', ['nipals'])
```

**models.py - Grid Generation**:
```python
        # PLS Regression (with all hyperparameters)
        if 'PLS' in enabled_models:
            pls_configs = []
            for nc in pls_components:
                for max_iter in pls_max_iter_list:
                    for tol in pls_tol_list:
                        for algorithm in pls_algorithm_list:
                            pls_configs.append(
                                (
                                    PLSRegression(
                                        n_components=nc,
                                        scale=False,
                                        max_iter=max_iter,
                                        tol=tol,
                                        algorithm=algorithm
                                    ),
                                    {
                                        "n_components": nc,
                                        "max_iter": max_iter,
                                        "tol": tol,
                                        "algorithm": algorithm
                                    }
                                )
                            )
            grids["PLS"] = pls_configs
```

**model_config.py - Tier Defaults**:
```python
'PLS': {
    'standard': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
        'max_iter': [500],
        'tol': [1e-6],
        'algorithm': ['nipals'],
        'note': 'Grid size: 12×1×1×1 = 12 configs (UNCHANGED)'
    },
    'comprehensive': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
        'max_iter': [500, 1000],
        'tol': [1e-7, 1e-6, 1e-5],
        'algorithm': ['nipals', 'svd'],
        'note': 'Grid size: 12×2×3×2 = 144 configs'
    },
    'quick': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
        'max_iter': [500],
        'tol': [1e-6],
        'algorithm': ['nipals'],
        'note': 'Grid size: 12×1×1×1 = 12 configs (UNCHANGED)'
    }
}
```

**search.py - Function Signature**:
```python
def run_search(X, y, task_type, folds=5, variable_penalty=3, complexity_penalty=0.1,
               max_n_components=8, max_iter=500, models_to_test=None, preprocessing_methods=None,
               window_sizes=None, n_estimators_list=None, learning_rates=None,
               rf_n_trees_list=None, rf_max_depth_list=None,
               pls_max_iter_list=None, pls_tol_list=None, pls_algorithm_list=None,  # ADD HERE
               ridge_alphas_list=None, lasso_alphas_list=None,
               ...
```

**search.py - get_model_grids() Call**:
```python
    model_grids = get_model_grids(
        task_type, n_features, safe_max_components, max_iter,
        n_estimators_list=n_estimators_list, learning_rates=learning_rates,
        rf_n_trees_list=rf_n_trees_list, rf_max_depth_list=rf_max_depth_list,
        pls_max_iter_list=pls_max_iter_list,  # ADD HERE
        pls_tol_list=pls_tol_list,            # ADD HERE
        pls_algorithm_list=pls_algorithm_list,  # ADD HERE
        ridge_alphas_list=ridge_alphas_list, lasso_alphas_list=lasso_alphas_list,
        ...
```

**GUI - Tab 4C Control Creation**:
```python
# PLS Hyperparameters section in Tab 4C
pls_section = ttk.LabelFrame(advanced_content, text="PLS Hyperparameters", style='Card.TLabelframe')
pls_section.grid(row=X, column=0, columnspan=4, sticky='ew', pady=10)

pls_frame = ttk.Frame(pls_section)
pls_frame.pack(fill='both', expand=True, padx=10, pady=10)

# max_iter control
self.pls_max_iter_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='max_iter',
    param_label='Max Iterations',
    checkbox_values=[500, 1000, 2000, 5000],
    default_checked=[500],
    is_float=False,
    help_text='Maximum iterations for NIPALS algorithm'
)

# tol control
self.pls_tol_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='tol',
    param_label='Convergence Tolerance',
    checkbox_values=[1e-7, 1e-6, 1e-5, 1e-4],
    default_checked=[1e-6],
    is_float=True,
    help_text='Tolerance for stopping criterion'
)

# algorithm control
self.pls_algorithm_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='algorithm',
    param_label='Algorithm',
    checkbox_values=['nipals', 'svd'],
    default_checked=['nipals'],
    allow_string_values=True,
    help_text='NIPALS = iterative, SVD = direct decomposition'
)
```

**GUI - Tab 4C Parameter Extraction**:
```python
# In _run_analysis_thread() method (~line 6723)

# PLS parameters
pls_max_iter_list = self._extract_parameter_values(
    self.pls_max_iter_control, 'max_iter', is_float=False
) if hasattr(self, 'pls_max_iter_control') else None

pls_tol_list = self._extract_parameter_values(
    self.pls_tol_control, 'tol', is_float=True
) if hasattr(self, 'pls_tol_control') else None

pls_algorithm_list = self._extract_parameter_values(
    self.pls_algorithm_control, 'algorithm', allow_string_values=True
) if hasattr(self, 'pls_algorithm_control') else None
```

**GUI - Tab 4C run_search() Call**:
```python
# Around line 7344
results_df = run_search(
    X, y, task_type,
    folds=cv_folds,
    variable_penalty=variable_penalty,
    complexity_penalty=complexity_penalty,
    max_n_components=max_n_components,
    max_iter=max_iter,
    models_to_test=models_to_test,
    pls_max_iter_list=pls_max_iter_list,  # ADD HERE
    pls_tol_list=pls_tol_list,            # ADD HERE
    pls_algorithm_list=pls_algorithm_list,  # ADD HERE
    ...
    tier=tier
)
```

**GUI - Tab 7C Subtab**:
```python
# PLS Subtab in Tab 7C
self.tab_7c_pls = ttk.Frame(self.tab_7c_model_notebook)
self.tab_7c_model_notebook.add(self.tab_7c_pls, text="PLS")

pls_frame = ttk.Frame(self.tab_7c_pls)
pls_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Create scrollable frame if needed
pls_scroll_frame = self._create_scrollable_frame(pls_frame)

# All 4 PLS parameters
self.tab7c_pls_n_components_control = self._create_parameter_grid_control(
    parent=pls_scroll_frame,
    param_name='n_components',
    param_label='Number of Components',
    checkbox_values=[2, 4, 6, 8, 10, 12, 16, 20],
    default_checked=[10],
    is_float=False
)

self.tab7c_pls_max_iter_control = self._create_parameter_grid_control(
    parent=pls_scroll_frame,
    param_name='max_iter',
    param_label='Max Iterations',
    checkbox_values=[500, 1000, 2000],
    default_checked=[500],
    is_float=False
)

self.tab7c_pls_tol_control = self._create_parameter_grid_control(
    parent=pls_scroll_frame,
    param_name='tol',
    param_label='Convergence Tolerance',
    checkbox_values=[1e-7, 1e-6, 1e-5, 1e-4],
    default_checked=[1e-6],
    is_float=True
)

self.tab7c_pls_algorithm_control = self._create_parameter_grid_control(
    parent=pls_scroll_frame,
    param_name='algorithm',
    param_label='Algorithm',
    checkbox_values=['nipals', 'svd'],
    default_checked=['nipals'],
    allow_string_values=True
)
```

**GUI - Tab 7C Parameter Extraction**:
```python
def _extract_tab7c_pls_params(self):
    """Extract PLS parameters from Tab 7C"""
    params = {}

    if hasattr(self, 'tab7c_pls_n_components_control'):
        vals = self._extract_parameter_values(
            self.tab7c_pls_n_components_control, 'n_components', is_float=False
        )
        if vals:
            params['n_components'] = vals[0]

    if hasattr(self, 'tab7c_pls_max_iter_control'):
        vals = self._extract_parameter_values(
            self.tab7c_pls_max_iter_control, 'max_iter', is_float=False
        )
        if vals:
            params['max_iter'] = vals[0]

    if hasattr(self, 'tab7c_pls_tol_control'):
        vals = self._extract_parameter_values(
            self.tab7c_pls_tol_control, 'tol', is_float=True
        )
        if vals:
            params['tol'] = vals[0]

    if hasattr(self, 'tab7c_pls_algorithm_control'):
        vals = self._extract_parameter_values(
            self.tab7c_pls_algorithm_control, 'algorithm', allow_string_values=True
        )
        if vals:
            params['algorithm'] = vals[0]

    return params
```

---

## 9. TROUBLESHOOTING

### 9.1 Common Issues

**Issue**: Grid sizes explode after implementation
**Diagnosis**: Check model_config.py - likely multi-value defaults instead of single values
**Solution**: Verify all standard/quick tier parameters have single-value lists

**Issue**: Parameters not appearing in results
**Diagnosis**: Parameters not being passed through the flow
**Solution**: Verify params are in: GUI extraction → run_search() call → get_model_grids() signature → grid generation

**Issue**: Syntax errors in nested loops
**Diagnosis**: Missing colons, incorrect indentation
**Solution**: Use syntax checker: `python -m py_compile src/spectral_predict/models.py`

**Issue**: Conditional logic not working (MLP momentum, SVR kernel)
**Diagnosis**: Callbacks not bound or logic incorrect
**Solution**: Verify callback binding, add debug prints to check parameter states

**Issue**: Tab 7C subtabs not switching
**Diagnosis**: Auto-navigation not bound or model names don't match
**Solution**: Check model_type_var trace binding, verify model name mappings

### 9.2 Debugging Strategies

**Grid Size Verification**:
```python
# Add to models.py after grid generation
print(f"DEBUG: PLS grid size = {len(grids.get('PLS', []))}")
print(f"DEBUG: First PLS config: {grids['PLS'][0] if grids.get('PLS') else 'None'}")
```

**Parameter Flow Tracing**:
```python
# Add to search.py
print(f"DEBUG: run_search received pls_max_iter_list = {pls_max_iter_list}")

# Add to models.py
print(f"DEBUG: get_model_grids using pls_max_iter_list = {pls_max_iter_list}")
```

**GUI Control Verification**:
```python
# Add to _extract_parameter_values
print(f"DEBUG: Extracting {param_name}, checkboxes checked: {checked_values}")
print(f"DEBUG: Extracting {param_name}, custom entry: {custom_values}")
print(f"DEBUG: Extracting {param_name}, final list: {final_list}")
```

### 9.3 Validation Commands

**Syntax Validation**:
```bash
python -m py_compile src/spectral_predict/models.py
python -m py_compile src/spectral_predict/search.py
python -m py_compile src/spectral_predict/model_config.py
python -m py_compile spectral_predict_gui_optimized.py
```

**Grid Size Verification**:
```python
from src.spectral_predict.models import get_model_grids
from src.spectral_predict.model_config import get_hyperparameters

# Test PLS with defaults
grids = get_model_grids('regression', 100, tier='standard', enabled_models=['PLS'])
print(f"PLS standard tier grid size: {len(grids['PLS'])}")  # Should be 12

# Test with comprehensive tier
grids = get_model_grids('regression', 100, tier='comprehensive', enabled_models=['PLS'])
print(f"PLS comprehensive tier grid size: {len(grids['PLS'])}")  # Should be 144
```

**Parameter Flow Test**:
```python
# Manual test of complete flow
pls_max_iter_list = [1000]
pls_tol_list = [1e-5]
pls_algorithm_list = ['svd']

grids = get_model_grids(
    'regression', 100, max_n_components=20,
    pls_max_iter_list=pls_max_iter_list,
    pls_tol_list=pls_tol_list,
    pls_algorithm_list=pls_algorithm_list,
    tier='standard',
    enabled_models=['PLS']
)

# Verify params in grid
model, params = grids['PLS'][0]
assert params['max_iter'] == 1000
assert params['tol'] == 1e-5
assert params['algorithm'] == 'svd'
print("Parameter flow test PASSED")
```

---

## 10. CONCLUSION

This handoff document provides complete specifications for implementing all 39 missing hyperparameters across 12 models. The forensic analysis proves the implementation is safe (no grid explosion with single-value defaults) and aligns with the user's vision of full parameter exposure with sensible defaults.

The multi-agent execution strategy enables parallel work where possible, reducing wall-clock time from 28 agent-hours to ~14.5 hours. Each agent has clear dependencies, deliverables, and reference code to ensure consistent implementation.

**Ready for Agent Execution**: All specifications complete, agent assignments defined, success criteria established.

---

**END OF HANDOFF DOCUMENT**
