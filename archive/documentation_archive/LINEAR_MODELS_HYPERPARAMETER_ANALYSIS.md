# Linear Models Hyperparameter Implementation Analysis

## Executive Summary

This document provides a comprehensive analysis of the current hyperparameter implementation for linear models (PLS, PLS-DA, Ridge, Lasso, ElasticNet) and identifies what needs to be added to complete Phase 1 backend implementation.

## Current State Analysis

### 1. PLS (Partial Least Squares Regression)

**Location in code**:
- Config: `C:\Users\sponheim\git\dasp\src\spectral_predict\model_config.py` Lines 88-110
- Implementation: `C:\Users\sponheim\git\dasp\src\spectral_predict\models.py` Lines 447-452

**Current Implementation**:
```python
# In models.py (Line 449-451)
grids["PLS"] = [
    (PLSRegression(n_components=nc, scale=False), {"n_components": nc})
    for nc in pls_components
]
```

**Parameters in model_config.py**:
```python
'PLS': {
    'standard': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
        'max_iter': [500],  # DEFINED BUT NOT USED
        'tol': [1e-6],  # DEFINED BUT NOT USED
        'algorithm': ['nipals'],  # DEFINED BUT NOT USED
    }
}
```

**ISSUE**: PLS has 3 parameters defined in config but only `n_components` is used in the model creation.

**Required Parameters** (from task):
1. **max_iter**: Maximum iterations for algorithm
   - Checkboxes: [500, 1000, 2000, 5000]
   - Default: [500]
   - is_float: False

2. **tol**: Convergence tolerance
   - Checkboxes: [1e-7, 1e-6, 1e-5, 1e-4]
   - Default: [1e-6]
   - is_float: True

3. **algorithm**: PLS algorithm to use
   - Checkboxes: ['nipals', 'svd']
   - Default: ['nipals']
   - allow_string_values: True

### 2. PLS-DA (PLS Discriminant Analysis)

**Location in code**:
- Config: `C:\Users\sponheim\git\dasp\src\spectral_predict\model_config.py` Lines 609-646
- Implementation: `C:\Users\sponheim\git\dasp\src\spectral_predict\models.py` Lines 692-697
- Search: `C:\Users\sponheim\git\dasp\src\spectral_predict\search.py` Lines 764-780

**Current Implementation**:
```python
# In models.py (Line 694-696)
grids["PLS-DA"] = [
    (PLSRegression(n_components=nc, scale=False), {"n_components": nc})
    for nc in pls_components
]

# In search.py (Lines 766-780) - TWO-STAGE MODEL
pipe_steps.append(("pls", model))
# Extract LogisticRegression parameters from params dict
lr_C = params.get("lr_C", 1.0)
lr_solver = params.get("lr_solver", "lbfgs")
lr_penalty = params.get("lr_penalty", "l2")
lr_max_iter = params.get("lr_max_iter", 1000)
lr_class_weight = params.get("lr_class_weight", None)
pipe_steps.append(("lr", LogisticRegression(
    C=lr_C,
    solver=lr_solver,
    penalty=lr_penalty,
    max_iter=lr_max_iter,
    class_weight=lr_class_weight,
    random_state=42
)))
```

**Parameters in model_config.py**:
```python
'PLS-DA': {
    'standard': {
        'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # PLS components
        'max_iter': [500],  # PLS parameter - DEFINED BUT NOT USED IN GRID
        'tol': [1e-6],  # PLS parameter - DEFINED BUT NOT USED IN GRID
        'algorithm': ['nipals'],  # PLS parameter - DEFINED BUT NOT USED IN GRID
        'lr_C': [1.0],  # LogisticRegression C parameter - DEFINED AND USED IN SEARCH.PY
        'lr_solver': ['lbfgs'],  # LogisticRegression solver - DEFINED AND USED IN SEARCH.PY
        'lr_penalty': ['l2'],  # LogisticRegression penalty - DEFINED AND USED IN SEARCH.PY
        'lr_max_iter': [1000],  # LogisticRegression max_iter - DEFINED AND USED IN SEARCH.PY
        'lr_class_weight': [None],  # LogisticRegression class_weight - DEFINED AND USED IN SEARCH.PY
    }
}
```

**ISSUE**: PLS-DA has PLS parameters (max_iter, tol, algorithm) defined but NOT passed to model grid. LogisticRegression parameters (lr_*) are defined and correctly used in search.py.

**Status**:
- **PLS stage parameters (3)**: Need to be added to model grid in models.py
- **LogisticRegression parameters (5)**: ✅ Already implemented in search.py

### 3. Ridge Regression

**Location in code**:
- Config: `C:\Users\sponheim\git\dasp\src\spectral_predict\model_config.py` Lines 112-131
- Implementation: `C:\Users\sponheim\git\dasp\src\spectral_predict\models.py` Lines 454-464

**Current Implementation**:
```python
# In models.py (Lines 455-464)
if 'Ridge' in enabled_models:
    ridge_configs = []
    for alpha in ridge_alphas_list:
        ridge_configs.append(
            (
                Ridge(alpha=alpha, random_state=42),
                {"alpha": alpha}
            )
        )
    grids["Ridge"] = ridge_configs
```

**Parameters in model_config.py**:
```python
'Ridge': {
    'standard': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'solver': ['auto'],  # DEFINED BUT NOT USED
        'tol': [1e-4],  # DEFINED BUT NOT USED
    }
}
```

**ISSUE**: Ridge has 2 parameters defined in config (solver, tol) but NOT used in model creation.

**Required Parameters** (from task):
1. **solver**: Solver to use
   - Checkboxes: ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
   - Default: ['auto']
   - allow_string_values: True

2. **tol**: Convergence tolerance
   - Checkboxes: [1e-5, 1e-4, 1e-3]
   - Default: [1e-4]
   - is_float: True

### 4. Lasso Regression

**Location in code**:
- Config: `C:\Users\sponheim\git\dasp\src\spectral_predict\model_config.py` Lines 409-429
- Implementation: `C:\Users\sponheim\git\dasp\src\spectral_predict\models.py` Lines 466-476

**Current Implementation**:
```python
# In models.py (Lines 467-476)
if 'Lasso' in enabled_models:
    lasso_configs = []
    for alpha in lasso_alphas_list:
        lasso_configs.append(
            (
                Lasso(alpha=alpha, random_state=42, max_iter=max_iter),
                {"alpha": alpha}
            )
        )
    grids["Lasso"] = lasso_configs
```

**Parameters in model_config.py**:
```python
'Lasso': {
    'standard': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'selection': ['cyclic'],  # DEFINED BUT NOT USED
        'tol': [1e-4],  # DEFINED BUT NOT USED
    }
}
```

**ISSUE**: Lasso has 2 parameters defined in config (selection, tol) but NOT used in model creation.

**Required Parameters** (from task):
1. **selection**: Coefficient update method
   - Checkboxes: ['cyclic', 'random']
   - Default: ['cyclic']
   - allow_string_values: True

2. **tol**: Convergence tolerance
   - Checkboxes: [1e-5, 1e-4, 1e-3]
   - Default: [1e-4]
   - is_float: True

### 5. ElasticNet Regression

**Location in code**:
- Config: `C:\Users\sponheim\git\dasp\src\spectral_predict\model_config.py` Lines 133-155
- Implementation: `C:\Users\sponheim\git\dasp\src\spectral_predict\models.py` Lines 478-489

**Current Implementation**:
```python
# In models.py (Lines 479-489)
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
    grids["ElasticNet"] = elasticnet_configs
```

**Parameters in model_config.py**:
```python
'ElasticNet': {
    'standard': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'selection': ['cyclic'],  # DEFINED BUT NOT USED
        'tol': [1e-4],  # DEFINED BUT NOT USED
    }
}
```

**ISSUE**: ElasticNet has 2 parameters defined in config (selection, tol) but NOT used in model creation.

**Required Parameters** (from task):
1. **selection**: Coefficient update method
   - Checkboxes: ['cyclic', 'random']
   - Default: ['cyclic']
   - allow_string_values: True

2. **tol**: Convergence tolerance
   - Checkboxes: [1e-5, 1e-4, 1e-3]
   - Default: [1e-4]
   - is_float: True

## Implementation Plan

### Phase 1: Update model_config.py (Already Complete)

The config file already has ALL required parameters defined in the comprehensive tier. The only missing element is expanding the checkboxes in standard tier to match the task requirements.

**Required Updates to model_config.py**:

1. **PLS** - Update Lines 88-110:
   ```python
   'standard': {
       'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
       'max_iter': [500, 1000, 2000, 5000],  # ADD MORE OPTIONS
       'tol': [1e-7, 1e-6, 1e-5, 1e-4],  # ADD MORE OPTIONS
       'algorithm': ['nipals', 'svd'],  # ADD SVD OPTION
   }
   ```

2. **PLS-DA** - Update Lines 609-646:
   ```python
   'standard': {
       'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],
       'max_iter': [500, 1000, 2000, 5000],  # ADD MORE OPTIONS (PLS)
       'tol': [1e-7, 1e-6, 1e-5, 1e-4],  # ADD MORE OPTIONS (PLS)
       'algorithm': ['nipals', 'svd'],  # ADD SVD OPTION (PLS)
       'lr_C': [0.01, 0.1, 1.0, 10.0, 100.0],  # ADD MORE OPTIONS
       'lr_solver': ['lbfgs', 'liblinear', 'saga', 'sag'],  # ADD MORE OPTIONS
       'lr_penalty': ['l2', 'l1', 'elasticnet', 'none'],  # ADD MORE OPTIONS
       'lr_max_iter': [100, 500, 1000, 2000],  # ADD MORE OPTIONS
       'lr_class_weight': [None, 'balanced'],  # ADD BALANCED OPTION
   }
   ```

3. **Ridge** - Update Lines 112-131:
   ```python
   'standard': {
       'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
       'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],  # ADD MORE OPTIONS
       'tol': [1e-5, 1e-4, 1e-3],  # ADD MORE OPTIONS
   }
   ```

4. **Lasso** - Update Lines 409-429:
   ```python
   'standard': {
       'alpha': [0.001, 0.01, 0.1, 1.0],
       'selection': ['cyclic', 'random'],  # ADD RANDOM OPTION
       'tol': [1e-5, 1e-4, 1e-3],  # ADD MORE OPTIONS
   }
   ```

5. **ElasticNet** - Update Lines 133-155:
   ```python
   'standard': {
       'alpha': [0.001, 0.01, 0.1, 1.0],
       'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
       'selection': ['cyclic', 'random'],  # ADD RANDOM OPTION
       'tol': [1e-5, 1e-4, 1e-3],  # ADD MORE OPTIONS
   }
   ```

### Phase 2: Update models.py

**Critical Updates Required**:

1. **Add missing parameter lists to function signature** (Line 247):
   ```python
   pls_max_iter_list=None, pls_tol_list=None, pls_algorithm_list=None,  # ADD pls_algorithm_list
   ```

2. **Add parameter loading code** (After Line 517):
   ```python
   if pls_algorithm_list is None:
       pls_config = get_hyperparameters('PLS', tier)
       pls_algorithm_list = pls_config.get('algorithm', ['nipals'])
   ```

3. **Update PLS Regression grid creation** (Lines 447-452):
   ```python
   # REPLACE LINES 449-451 WITH:
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

4. **Update Ridge Regression grid creation** (Lines 454-464):
   ```python
   # ADD THESE LINES BEFORE Line 456:
   if ridge_solver_list is None:
       ridge_config = get_hyperparameters('Ridge', tier)
       ridge_solver_list = ridge_config.get('solver', ['auto'])
   if ridge_tol_list is None:
       ridge_config = get_hyperparameters('Ridge', tier)
       ridge_tol_list = ridge_config.get('tol', [1e-4])

   # REPLACE LINES 456-464 WITH:
   ridge_configs = []
   for alpha in ridge_alphas_list:
       for solver in ridge_solver_list:
           for tol in ridge_tol_list:
               ridge_configs.append(
                   (
                       Ridge(alpha=alpha, solver=solver, tol=tol, random_state=42),
                       {"alpha": alpha, "solver": solver, "tol": tol}
                   )
               )
   grids["Ridge"] = ridge_configs
   ```

5. **Update Lasso Regression grid creation** (Lines 466-476):
   ```python
   # ADD THESE LINES BEFORE Line 468:
   if lasso_selection_list is None:
       lasso_config = get_hyperparameters('Lasso', tier)
       lasso_selection_list = lasso_config.get('selection', ['cyclic'])
   if lasso_tol_list is None:
       lasso_config = get_hyperparameters('Lasso', tier)
       lasso_tol_list = lasso_config.get('tol', [1e-4])

   # REPLACE LINES 468-476 WITH:
   lasso_configs = []
   for alpha in lasso_alphas_list:
       for selection in lasso_selection_list:
           for tol in lasso_tol_list:
               lasso_configs.append(
                   (
                       Lasso(alpha=alpha, selection=selection, tol=tol, random_state=42, max_iter=max_iter),
                       {"alpha": alpha, "selection": selection, "tol": tol}
                   )
               )
   grids["Lasso"] = lasso_configs
   ```

6. **Update ElasticNet Regression grid creation** (Lines 478-489):
   ```python
   # ADD THESE LINES BEFORE Line 480:
   if elasticnet_selection_list is None:
       en_config = get_hyperparameters('ElasticNet', tier)
       elasticnet_selection_list = en_config.get('selection', ['cyclic'])
   if elasticnet_tol_list is None:
       en_config = get_hyperparameters('ElasticNet', tier)
       elasticnet_tol_list = en_config.get('tol', [1e-4])

   # REPLACE LINES 480-489 WITH:
   elasticnet_configs = []
   for alpha in elasticnet_alphas_list:
       for l1_ratio in elasticnet_l1_ratios:
           for selection in elasticnet_selection_list:
               for tol in elasticnet_tol_list:
                   elasticnet_configs.append(
                       (
                           ElasticNet(alpha=alpha, l1_ratio=l1_ratio, selection=selection, tol=tol, random_state=42, max_iter=max_iter),
                           {"alpha": alpha, "l1_ratio": l1_ratio, "selection": selection, "tol": tol}
                       )
                   )
   grids["ElasticNet"] = elasticnet_configs
   ```

7. **Update PLS-DA Classification grid creation** (Lines 692-697):
   ```python
   # REPLACE LINES 694-696 WITH:
   pls_da_configs = []
   for nc in pls_components:
       for pls_max_iter in pls_max_iter_list:
           for pls_tol in pls_tol_list:
               for pls_algorithm in pls_algorithm_list:
                   for lr_C in plsda_C_list:
                       for lr_solver in plsda_solver_list:
                           for lr_penalty in plsda_penalty_list:
                               for lr_max_iter in plsda_max_iter_list:
                                   for lr_class_weight in plsda_class_weight_list:
                                       pls_da_configs.append(
                                           (
                                               PLSRegression(
                                                   n_components=nc,
                                                   scale=False,
                                                   max_iter=pls_max_iter,
                                                   tol=pls_tol,
                                                   algorithm=pls_algorithm
                                               ),
                                               {
                                                   "n_components": nc,
                                                   "max_iter": pls_max_iter,
                                                   "tol": pls_tol,
                                                   "algorithm": pls_algorithm,
                                                   "lr_C": lr_C,
                                                   "lr_solver": lr_solver,
                                                   "lr_penalty": lr_penalty,
                                                   "lr_max_iter": lr_max_iter,
                                                   "lr_class_weight": lr_class_weight
                                               }
                                           )
                                       )
   grids["PLS-DA"] = pls_da_configs
   ```

8. **Update function signature to include missing parameters** (Lines 223-250):
   ```python
   # ADD TO Line 228 (after ridge_alphas_list):
   ridge_solver_list=None, ridge_tol_list=None,

   # ADD TO Line 229 (after lasso_alphas_list):
   lasso_selection_list=None, lasso_tol_list=None,

   # ADD TO Line 233 (after elasticnet_l1_ratios):
   elasticnet_selection_list=None, elasticnet_tol_list=None,
   ```

### Phase 3: No Changes Needed for search.py

The search.py file already correctly handles PLS-DA LogisticRegression parameters (Lines 764-780). No changes needed.

## Summary

**Total Parameters to Add**:
- PLS: 3 parameters (max_iter, tol, algorithm)
- PLS-DA: 8 parameters (3 PLS + 5 LogisticRegression, but LR already implemented)
- Ridge: 2 parameters (solver, tol)
- Lasso: 2 parameters (selection, tol)
- ElasticNet: 2 parameters (selection, tol)

**Implementation Status**:
- model_config.py: Parameters defined but need to be expanded with more options ⚠️
- models.py: Parameters defined but NOT used in model grid creation ❌
- search.py: PLS-DA LogisticRegression parameters correctly implemented ✅

**Next Steps**:
1. Update model_config.py with expanded options (5 models)
2. Update models.py function signature and grid creation (5 models)
3. Test thoroughly with different tier selections

---

**File Paths**:
- Config: `C:\Users\sponheim\git\dasp\src\spectral_predict\model_config.py`
- Implementation: `C:\Users\sponheim\git\dasp\src\spectral_predict\models.py`
- Search Logic: `C:\Users\sponheim\git\dasp\src\spectral_predict\search.py`
