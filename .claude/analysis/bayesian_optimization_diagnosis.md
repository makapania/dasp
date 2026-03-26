# Bayesian Optimization Diagnosis and Fix Plan
**Date:** 2026-01-02
**Issue:** PLS underperforms with Bayesian optimization (R² ~0.90-0.95) vs grid search (~0.96), while Ridge works great (R² > 0.97)

---

## ROOT CAUSE ANALYSIS

### The Problem: "Best of Many" Return Value

**Current behavior (bayesian_utils.py lines 270-510):**
```python
# STEP 1: Test full model → RMSE = X
full_result = run_single_config_fn(...)  # n_components=5, all features
best_metric = full_result['RMSE']  # e.g., 2.5

# STEP 2: Test variable subsets (4 methods × 7 counts = 28 subsets)
for varsel_method in ['importance', 'spa', 'uve', 'cars']:
    for n_top in [10, 20, 50, 100, 250, 500, 1000]:
        subset_result = run_single_config_fn(...)  # same n_components=5, subset features
        if subset_result['RMSE'] < best_metric:
            best_metric = subset_result['RMSE']  # Update if better

# STEP 3: Return best of ~29 configs to TPE
return best_metric  # Returns e.g., 1.8 (from top50_spa)
```

**What TPE sees:**
- Trial 1: n_components=3 → RMSE=2.1 (best of 29 subsets)
- Trial 2: n_components=5 → RMSE=1.8 (best of 29 subsets)
- Trial 3: n_components=7 → RMSE=2.3 (best of 29 subsets)
- Trial 4: n_components=4 → RMSE=1.9 (best of 29 subsets)

**TPE's confusion:**
TPE thinks it's learning "which n_components is best", but it's actually learning "which n_components + variable subset combo is best". Since variable subsets change pseudo-randomly based on importance rankings (which vary with n_components), TPE can't learn the true relationship.

### Why Ridge Works But PLS Doesn't

**Ridge hyperparameters (bayesian_config.py lines 127-139):**
- `alpha` (continuous, log-scale): Controls regularization strength
- `solver`: Categorical choice
- `tol`: Convergence tolerance
- `max_iter`: Iteration limit

**Key difference:** Ridge's `alpha` parameter has a **monotonic relationship** with performance across variable subsets. If alpha=1.0 is good for the full model, it's likely good for subsets too.

**PLS hyperparameters (bayesian_config.py lines 106-124):**
- `n_components` (discrete, 1-8): Number of latent variables
- `max_iter`: Iteration limit
- `tol`: Convergence tolerance

**Key difference:** PLS's `n_components` has a **non-monotonic, subset-dependent relationship** with performance:
- Full model (2000 features): n_components=8 might be best
- Top-50 subset: n_components=3 might be best (fewer features need fewer components)
- Top-500 subset: n_components=6 might be best

The "best n_components" **changes with each subset**. TPE sees:
- Trial with n_components=8 returns best_metric from top50 (which prefers n_components=3) → looks bad
- Trial with n_components=3 returns best_metric from top500 (which prefers n_components=6) → looks bad
- **Result:** TPE can't learn which n_components is actually good

---

## MODEL-BY-MODEL ANALYSIS

### PLS (BROKEN)
- **Hyperparameter:** `n_components` (1-8)
- **Issue:** Different subsets need different n_components
- **Why it fails:** Non-monotonic relationship across subsets

### Ridge (WORKS)
- **Hyperparameter:** `alpha` (0.001-100, log-scale)
- **Why it works:** Alpha has monotonic effect across subsets

### Lasso/ElasticNet (LIKELY BROKEN)
- **Hyperparameters:** `alpha`, `l1_ratio`
- **Prediction:** Similar to Ridge, should work if alpha dominates

### RandomForest (UNCERTAIN)
- **Hyperparameters:** `n_estimators`, `max_depth`, `min_samples_split`, etc.
- **Prediction:** Tree depth might have subset-dependent relationship (deeper trees for more features)
- **Likely issue:** `max_depth` may behave like n_components

### XGBoost/LightGBM/CatBoost (UNCERTAIN)
- **Hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, regularization
- **Prediction:** `max_depth` and `num_leaves` (LightGBM) likely subset-dependent
- **Likely issue:** Depth parameters need different values for different subset sizes

### SVR (UNCERTAIN)
- **Hyperparameters:** `C`, `epsilon`, `gamma`, `kernel`
- **Prediction:** `C` (regularization) likely monotonic like Ridge's alpha
- **Should work:** Similar to Ridge

### MLP (UNCERTAIN)
- **Hyperparameters:** `hidden_layer_sizes`, `alpha`, `learning_rate_init`
- **Prediction:** Layer sizes might be subset-dependent
- **Potential issue:** Network capacity needs differ between full/subset data

---

## TRIAL ALLOCATION ISSUE

**Current behavior (search.py lines 2119-2135):**
```python
total_tasks = len(models_to_test) * len(preprocessing_methods)
total_trials = total_tasks * n_trials
print(f"Total optimizations: {total_tasks} (models × preprocessing)")
```

**Problem:** User expects "30 trials per model", but gets "30 trials per (model × preprocessing)"

**Example:**
- Models: PLS, Ridge
- Preprocessing: raw, SNV, deriv1, deriv2 (4 methods)
- n_trials: 30
- **Expected:** 2 models × 30 trials = 60 trials total
- **Actual:** 2 models × 4 preprocessing × 30 trials = 240 trials total

**Impact:** This is actually GOOD for exploration (more trials per model group), but confusing for users.

---

## PROPOSED FIXES

### Option A: Return Full Model Score Only (CLEAN HYPERPARAMETER SIGNAL)

**Change lines 270-510 in bayesian_utils.py:**
```python
# STEP 1: Test full model (all features)
full_result = run_single_config_fn(...)
full_metric = full_result['RMSE']  # Store separately

# STEP 2: Test variable subsets
best_subset_metric = full_metric  # Initialize with full model
for varsel_method in variable_selection_methods:
    for n_top in valid_variable_counts:
        subset_result = run_single_config_fn(...)
        if subset_result['RMSE'] < best_subset_metric:
            best_subset_metric = subset_result['RMSE']

# STEP 3: Return FULL MODEL score to TPE (ignore subsets for learning)
# Store best_subset_metric in trial attributes for final ranking
trial.set_user_attr('full_model_rmse', full_metric)
trial.set_user_attr('best_subset_rmse', best_subset_metric)
return full_metric  # ← TPE sees clean hyperparameter signal
```

**Pros:**
- TPE learns true hyperparameter→performance relationship
- Works for ALL models (PLS, Ridge, trees, etc.)
- Subset exploration still happens (stored in trial_results)
- Final ranking uses best_subset_rmse (best of all subsets)

**Cons:**
- TPE doesn't directly optimize for best subset (but it optimizes for best full model, which is correlated)

### Option B: Separate Optimization Per Subset Size (STAGED APPROACH)

**Change structure: Run separate Optuna studies for each subset size**
```python
for n_top in [None, 50, 100, 250]:  # None = full model
    study = create_optuna_study(...)
    objective = create_objective_for_subset_size(n_top)
    study.optimize(objective, n_trials=trials_per_subset)
```

**Pros:**
- TPE can learn hyperparameters for each subset size independently
- Optimal n_components for top-50, optimal for top-500, etc.

**Cons:**
- Much more complex
- Multiplies trial count by number of subset sizes
- Doesn't match grid search's "single sweep" approach

### Option C: Model-Specific Return Logic

**Use Option A for models with subset-dependent hyperparameters (PLS, trees), keep current behavior for others (Ridge)**

**Pros:**
- Tailored to each model's characteristics

**Cons:**
- Complex, fragile, model-specific logic
- Hard to maintain

---

## RECOMMENDED FIX: Option A

**Rationale:**
1. **Simple and general** - Works for all models without model-specific code
2. **Clean signal to TPE** - Learns hyperparameters from full model (no subset noise)
3. **Preserves subset exploration** - All subsets still tested, results stored
4. **Best-of-both-worlds ranking** - Final results ranked by best_subset_rmse
5. **Matches grid search philosophy** - Grid search also optimizes hyperparameters on full model first, then applies to subsets

**Implementation steps:**
1. Modify `create_objective_function()` to return full model score to TPE
2. Store both `full_model_rmse` and `best_subset_rmse` as trial attributes
3. Modify `convert_optuna_result_to_dasp_format()` to return ALL configs (unchanged)
4. Final ranking already uses best_subset_rmse from stored results (unchanged)

---

## TRIAL ALLOCATION FIX

**Change search.py lines 2119-2135:**
```python
# OLD: n_trials per (model × preprocessing)
total_tasks = len(models_to_test) * len(preprocessing_methods)
total_trials = total_tasks * n_trials

# NEW: n_trials per model (split across preprocessing methods)
trials_per_combination = max(1, n_trials // len(preprocessing_methods))
total_trials_actual = len(models_to_test) * len(preprocessing_methods) * trials_per_combination
print(f"Trials per model: {n_trials} (split as {trials_per_combination} per preprocessing method)")
```

**User expectation alignment:**
- User says "30 trials per model"
- If 4 preprocessing methods: 30 // 4 = 7 trials per (model × preprocessing)
- Total: 2 models × 4 preprocessing × 7 trials = 56 trials (close to 2 × 30 = 60)

**Alternative (keep current, clarify messaging):**
```python
print(f"Trials per model×preprocessing: {n_trials}")
print(f"Total trials: {total_trials} ({len(models_to_test)} models × {len(preprocessing_methods)} preprocessing × {n_trials} trials)")
```

---

## TESTING PLAN

1. **Baseline comparison** (before fix):
   - Run Bayesian PLS with 30 trials → measure R²
   - Run Bayesian Ridge with 30 trials → measure R²
   - Run Grid Search PLS → measure R²
   - Run Grid Search Ridge → measure R²

2. **Apply Option A fix**

3. **Post-fix comparison** (after fix):
   - Run Bayesian PLS with 30 trials → measure R²
   - Run Bayesian Ridge with 30 trials → measure R²
   - Verify PLS R² matches or beats grid search
   - Verify Ridge R² doesn't regress

4. **Tree models** (if time permits):
   - Test LightGBM, RandomForest with Bayesian
   - Compare to grid search

---

## EXPECTED OUTCOMES

**Before fix:**
- PLS: R² 0.90-0.95 (Bayesian) < 0.96 (Grid) ✗
- Ridge: R² > 0.97 (Bayesian) ≥ Grid ✓

**After fix:**
- PLS: R² ≥ 0.96 (Bayesian) ≥ Grid ✓
- Ridge: R² > 0.97 (Bayesian) ≥ Grid ✓ (no regression)
- Trees: R² comparable or better than Grid ✓

---

## FILES TO MODIFY

1. **C:\Users\sponheim\git\dasp\src\spectral_predict\bayesian_utils.py**
   - Modify `create_objective_function()` to return full model score (lines 231-522)
   - Store both full_model and best_subset metrics as trial attributes

2. **C:\Users\sponheim\git\dasp\src\spectral_predict\search.py** (optional)
   - Clarify trial allocation messaging (lines 2119-2135)

3. **Tests** (create new)
   - `test_bayesian_pls_matches_grid.py` - Verify fix works
