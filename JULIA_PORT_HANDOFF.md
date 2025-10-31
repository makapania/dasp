# Julia Port Handoff - Phase 1: Core Algorithms

**Date:** October 29, 2025
**Status:** Ready for Julia port
**Strategy:** Option 1 - Port core algorithms, simple CLI/basic GUI in Julia, keep Python GUI for advanced features

---

## Executive Summary

The Python implementation is **debugged and validated**. Today (Oct 29) we fixed:
1. ✅ Variable subsets with derivatives (double-preprocessing bug)
2. ✅ Region subsets now work for derivatives (was disabled)
3. ✅ Region subsets run for ALL models (was only PLS/RF/MLP)
4. ✅ Ranking fixed (90% performance, 10% complexity)
5. ✅ Preprocessing labels corrected for subsets

**Core algorithms are ready to port to Julia.**

---

## Port Strategy: Phase 1 vs Phase 2

### Phase 1: Port Now (Core Algorithms)
- ✅ Preprocessing pipeline
- ✅ Model definitions (PLS, Ridge, Lasso, RF, MLP, NeuralBoosted)
- ✅ Cross-validation with subset selection
- ✅ Variable subset analysis (top-N feature selection)
- ✅ Region subset analysis (spectral region detection)
- ✅ Scoring/ranking system
- ✅ Basic CLI or simple GUI (file I/O, run analysis, save results)

### Phase 2: Port Later (Advanced GUI)
- ⏸️ Interactive plots (matplotlib → Makie.jl/Plots.jl)
- ⏸️ Cursor region selection on refinement page
- ⏸️ Multi-tab GUI with live progress
- ⏸️ Results table with sorting/filtering
- ⏸️ Model refinement interface

**Keep Python GUI for advanced features during Phase 1.**

---

## Architecture Overview

### File Structure (Recommended Julia Equivalent)

```
SpectralPredict.jl/
├── src/
│   ├── SpectralPredict.jl          # Main module
│   ├── preprocessing.jl             # From preprocess.py
│   ├── models.jl                    # From models.py
│   ├── search.jl                    # From search.py ⭐ CORE
│   ├── regions.jl                   # From regions.py
│   ├── scoring.jl                   # From scoring.py
│   └── cli.jl                       # Simple CLI interface (NEW)
├── test/
│   └── runtests.jl
└── examples/
    └── basic_analysis.jl
```

### Data Flow

```
Input (CSV/SPC)
    ↓
Preprocessing Pipeline
    ↓
Model Search Loop
    ├── Full model (all features)
    ├── Variable subsets (top-N features)
    └── Region subsets (spectral regions)
    ↓
Scoring & Ranking
    ↓
Output (CSV + summary)
```

---

## Core Algorithm Details

### 1. Preprocessing Pipeline (preprocess.py)

**Supported methods:**
- `raw` - No preprocessing
- `snv` - Standard Normal Variate
- `deriv` - Savitzky-Golay derivatives (1st, 2nd)
- `snv_deriv` - SNV then derivative
- `deriv_snv` - Derivative then SNV

**Key function:** `build_preprocessing_pipeline(name, deriv, window, polyorder)`

**Returns:** List of sklearn transformers (in Julia: create similar pipeline)

**Julia packages:**
- SNV: Implement manually (simple: `(x - mean(x)) / std(x)`)
- Derivatives: Use `SavitzkyGolay.jl` or implement manually

**Configuration combinations:**
```python
# User selects: SNV ✓, SG2 ✓
# System generates:
configs = [
    {"name": "snv", "deriv": None, "window": None, "polyorder": None},
    {"name": "deriv", "deriv": 2, "window": 17, "polyorder": 3},
    {"name": "snv_deriv", "deriv": 2, "window": 17, "polyorder": 3},
    # Optional: deriv_snv if user enables it
]
```

---

### 2. Model Search with Subset Analysis (search.py)

**Main function:** `run_search(X, y, task_type, ...)`

**Critical insight from today's debugging:**
> Derivatives change the feature space! Cannot apply preprocessing twice.

**Algorithm structure:**

```python
for preprocess_cfg in preprocess_configs:
    # 1. Compute region subsets on PREPROCESSED data
    X_preprocessed = apply_preprocessing(X, preprocess_cfg)
    region_subsets = create_region_subsets(X_preprocessed, y, wavelengths)

    for model in models:
        # 2. Run full model
        result_full = run_cv(X, y, model, preprocess_cfg)

        # 3. Variable subsets (for models with feature importance)
        if model in ["PLS", "RandomForest", "MLP", "NeuralBoosted"]:
            # Fit on full preprocessed data
            model.fit(X_preprocessed, y)
            importances = get_feature_importances(model)

            for n_top in [10, 20, 50, 100, 250]:
                top_indices = argsort(importances)[-n_top:]

                # CRITICAL: For derivatives, use preprocessed data!
                if preprocess_cfg["deriv"] is not None:
                    result = run_cv(
                        X_preprocessed[:, top_indices],  # Already preprocessed
                        y, model,
                        preprocess_cfg,  # For labeling only
                        skip_preprocessing=True  # Don't reapply!
                    )
                else:
                    # For raw/SNV: subset raw data, reapply preprocessing
                    result = run_cv(
                        X[:, top_indices],  # Raw data
                        y, model,
                        preprocess_cfg  # Will reapply
                    )

        # 4. Region subsets (for ALL models)
        for region in region_subsets:
            # Same logic: preprocessed vs raw
            if preprocess_cfg["deriv"] is not None:
                result = run_cv(X_preprocessed[:, region['indices']], ...)
            else:
                result = run_cv(X[:, region['indices']], ...)
```

**Why this matters:**
- **Derivatives**: 101 features → 84 features (window=17 reduces count)
- **Feature indices** refer to the 84 derivative features, not original 101
- **Must use preprocessed data** for subsets, or you get: `window(17) > n_features(10)` error

**Julia implementation notes:**
- Use `skip_preprocessing` flag in CV function
- Keep `preprocess_cfg` for result labeling even when skipping

---

### 3. Variable Subset Analysis (search.py:298-410)

**Feature importance methods:**

```python
def get_feature_importances(model, model_name, X, y):
    if model_name in ["PLS", "PLS-DA"]:
        # Use VIP scores (Variable Importance in Projection)
        return compute_vip_scores(model, X)

    elif model_name == "RandomForest":
        return model.feature_importances_

    elif model_name in ["MLP", "NeuralBoosted"]:
        # Permutation importance
        return permutation_importance(model, X, y)
```

**VIP Score formula:**
```python
def compute_vip_scores(pls_model, X):
    W = pls_model.x_weights_  # (n_features, n_components)
    T = pls_model.x_scores_   # (n_samples, n_components)
    Q = pls_model.y_loadings_ # (n_targets, n_components)

    # Sum of squares explained by each component
    s = np.sum(T**2 * (Q**2).T, axis=0)

    # VIP score for each variable
    vip = np.sqrt(n_features * np.sum(s * W**2, axis=1) / np.sum(s))
    return vip
```

**Julia packages:**
- PLS: `PartialLeastSquares.jl` or `MultivariateStats.jl`
- VIP: Implement manually (simple formula above)
- Permutation: Implement or use `MLJ.jl`

**Validation counts:**
```python
# Only test counts less than available features
n_features_preprocessed = X_preprocessed.shape[1]
valid_counts = [n for n in [10, 20, 50, 100, 250] if n < n_features_preprocessed]
```

---

### 4. Region Subset Analysis (regions.py)

**Today's enhancement:** Now supports 5, 10, 15, or 20 regions (was fixed at 5)

**Algorithm:**

```python
def create_region_subsets(X, y, wavelengths, n_top_regions=5):
    # 1. Divide spectrum into 50nm windows (25nm overlap)
    regions = compute_region_correlations(X, y, wavelengths,
                                          region_size=50, overlap=25)

    # 2. Rank by correlation with target
    top_regions = get_top_regions(regions, n_top=n_top_regions)

    # 3. Create subsets:
    #    - Individual top regions (3, 5, 7, or 10 depending on n_top_regions)
    #    - Combined regions (top-2, top-5, top-10, top-15, top-20)

    subsets = []

    # Individual regions
    n_individual = 3 if n_top_regions <= 5 else
                   5 if n_top_regions <= 10 else
                   7 if n_top_regions <= 15 else 10

    for i, region in enumerate(top_regions[:n_individual]):
        subsets.append({
            'indices': region['indices'],
            'tag': f'region_{region["start"]:.0f}-{region["end"]:.0f}nm',
            'description': f"Region {i+1}: {region['start']:.0f}-{region['end']:.0f}nm"
        })

    # Combinations
    for combo_size in [2, 5, 10, 15, 20]:
        if combo_size <= n_top_regions:
            indices = combine_region_indices(top_regions[:combo_size])
            subsets.append({
                'indices': indices,
                'tag': f'top{combo_size}regions',
                'description': f"Top {combo_size} regions combined"
            })

    return subsets
```

**Region correlation:**
```python
def compute_region_correlations(X, y, wavelengths, region_size=50, overlap=25):
    regions = []
    start_wl = wavelengths.min()

    while start_wl < wavelengths.max():
        end_wl = start_wl + region_size

        # Find features in this region
        mask = (wavelengths >= start_wl) & (wavelengths < end_wl)
        indices = np.where(mask)[0]

        # Compute mean correlation with target
        correlations = [pearsonr(X[:, i], y)[0] for i in indices]

        regions.append({
            'start': start_wl,
            'end': end_wl,
            'indices': indices,
            'mean_corr': np.mean(np.abs(correlations)),
            'max_corr': np.max(np.abs(correlations))
        })

        start_wl += (region_size - overlap)

    return regions
```

**Julia packages:**
- Correlations: `Statistics.cor` (built-in)
- Sliding windows: Implement manually

---

### 5. Scoring & Ranking System (scoring.py)

**Fixed today:** 90% performance, 10% complexity (was broken with harsh sparsity penalties)

**Formula:**

```python
# Performance score (z-scores, lower is better)
if task_type == "regression":
    z_rmse = (RMSE - mean(RMSE)) / std(RMSE)
    z_r2 = (R2 - mean(R2)) / std(R2)
    performance_score = 0.5 * z_rmse - 0.5 * z_r2
else:  # classification
    z_auc = (ROC_AUC - mean(ROC_AUC)) / std(ROC_AUC)
    z_acc = (Accuracy - mean(Accuracy)) / std(Accuracy)
    performance_score = -z_auc - 0.3 * z_acc

# Complexity penalty (linear, normalized to [0, 1])
lvs_penalty = LVs / 25.0
vars_penalty = n_vars / full_vars

# Scale to ~10% of performance (z-scores range ±3, so ~6 total)
complexity_scale = 0.3 * lambda_penalty / 0.15
complexity_penalty = complexity_scale * (lvs_penalty + vars_penalty)

# Final composite score (lower is better)
composite_score = performance_score + complexity_penalty

# Rank (1 = best)
rank = rank(composite_score)
```

**Key principles:**
- ✅ Models with similar R² stay close in ranking
- ✅ Complexity only as tiebreaker (~10% influence)
- ✅ LVs and variables weighted equally
- ❌ NO harsh sparsity penalties (old code had huge penalties for <10 vars)

**Julia implementation:**
- Z-scores: `(x .- mean(x)) ./ std(x)`
- Ranking: `sortperm(scores)`

---

### 6. Cross-Validation (search.py:453-522)

**Current:** 5-fold CV with parallel processing (joblib)

**Structure:**
```python
def _run_single_fold(pipe, X, y, train_idx, test_idx, task_type):
    pipe_clone = clone(pipe)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe_clone.fit(X_train, y_train)
    y_pred = pipe_clone.predict(X_test)

    if task_type == "regression":
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": rmse, "R2": r2}
    else:
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        return {"Accuracy": acc, "ROC_AUC": auc}

# Parallel execution
cv_metrics = Parallel(n_jobs=-1)(
    delayed(_run_single_fold)(pipe, X, y, train_idx, test_idx, task_type)
    for train_idx, test_idx in cv_splitter.split(X, y)
)

# Average metrics
mean_rmse = mean([m["RMSE"] for m in cv_metrics])
mean_r2 = mean([m["R2"] for m in cv_metrics])
```

**Julia packages:**
- CV splitting: `MLJ.jl` (has built-in CV)
- Parallel: `Distributed.jl` or `Threads.@threads`
- Metrics: Implement manually or use `MLJ.jl`

---

### 7. Model Definitions (models.py)

**Model hyperparameter grids:**

```python
# PLS: Test multiple component counts
n_components_list = [1, 2, 3, 5, 7, 10, 15, 20]  # Capped at min(n_features, n_samples_in_fold)

# Ridge/Lasso/ElasticNet: Alpha values
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# RandomForest
n_estimators = [50, 100, 200]
max_features = ['sqrt', 'log2']

# MLP
hidden_layer_sizes = [(50,), (100,), (50, 50)]
learning_rate_init = [0.001, 0.01]

# NeuralBoosted (custom model)
n_estimators = [50, 100]
learning_rate = [0.01, 0.1]
```

**Julia packages:**
- PLS: `PartialLeastSquares.jl` or `MultivariateStats.jl`
- Ridge/Lasso: `GLMNet.jl` or `MLJ.jl`
- RandomForest: `DecisionTree.jl`
- MLP: `Flux.jl`
- NeuralBoosted: Port from Python or use `EvoTrees.jl` (gradient boosting)

**NeuralBoosted architecture (if porting):**
```python
# Hybrid: Neural feature extractor + Gradient boosting
class NeuralBoostedRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, hidden_size=50):
        self.neural_net = MLP(hidden_layer_sizes=(hidden_size,))
        self.boosting = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

    def fit(self, X, y):
        # 1. Train neural net
        self.neural_net.fit(X, y)

        # 2. Extract features from hidden layer
        X_features = self.neural_net.hidden_layer_output(X)

        # 3. Train boosting on neural features
        self.boosting.fit(X_features, y)

    def predict(self, X):
        X_features = self.neural_net.hidden_layer_output(X)
        return self.boosting.predict(X_features)
```

---

## Critical Bugs Fixed Today (Oct 29, 2025)

### Bug 1: Double Preprocessing with Derivatives

**Symptom:** `Window length (17) must be <= number of features (10)`

**Cause:**
1. Derivatives applied: 101 features → 84 features
2. Importances computed on 84 features
3. Top-10 indices selected from 84 features
4. **BUG:** Indices used to subset raw 101-feature data
5. **BUG:** Derivative reapplied to 10 features → ERROR

**Fix:** Use preprocessed data for derivative subsets, skip reapplying preprocessing

**Code pattern:**
```python
if preprocess_cfg["deriv"] is not None:
    # Use already-preprocessed data
    result = run_cv(X_preprocessed[:, indices], y, model,
                    preprocess_cfg, skip_preprocessing=True)
else:
    # Use raw data, will reapply preprocessing
    result = run_cv(X[:, indices], y, model, preprocess_cfg)
```

**Julia implementation:** Add `skip_preprocessing::Bool=false` parameter to CV function

---

### Bug 2: Region Subsets Disabled for Derivatives

**Symptom:** No region subsets when only derivatives selected

**Cause:** Code explicitly skipped derivatives: `if preprocess_cfg["deriv"] is None`

**Why it existed:** Original developer thought region analysis didn't make sense for derivatives

**Why it's wrong:** Finding important regions in DERIVATIVE SPACE is critical!

**Fix:** Removed the check, now computes regions on preprocessed (derivative) data

**Code:**
```python
# OLD:
if enable_region_subsets and preprocess_cfg["deriv"] is None:
    region_subsets = create_region_subsets(X_preprocessed, ...)

# NEW:
if enable_region_subsets:
    region_subsets = create_region_subsets(X_preprocessed, ...)  # Works for all preprocessing
```

---

### Bug 3: Region Subsets Only for Specific Models

**Symptom:** Ridge, Lasso, ElasticNet never got region subsets

**Cause:** Region subset code was nested inside variable subset block (PLS/RF/MLP only)

**Fix:** Moved region subsets outside model-specific block

**Code structure:**
```python
# OLD (nested):
if model_name in ["PLS", "RandomForest", "MLP", "NeuralBoosted"]:
    # Variable subsets
    for n_top in [10, 20, 50]:
        ...

    # Region subsets (WRONG - only for these models!)
    for region in region_subsets:
        ...

# NEW (parallel):
if model_name in ["PLS", "RandomForest", "MLP", "NeuralBoosted"]:
    # Variable subsets
    for n_top in [10, 20, 50]:
        ...

# Region subsets for ALL models
for region in region_subsets:
    ...
```

---

### Bug 4: Broken Ranking System

**Symptom:** Models with terrible R² ranked at top because they used fewer variables

**Cause:** Harsh sparsity penalties overwhelmed performance
- `n_vars < 10`: +2.0 penalty
- `10 ≤ n_vars < 25`: +1.0 penalty
- `< 1% of vars`: +1.5 penalty

**Example:** Model with 10 vars, R²=0.3 ranked higher than 50 vars, R²=0.9

**Fix:** Removed all sparsity penalties, scaled complexity to ~10% of performance

**Formula:**
```python
# OLD: complexity could be 0-4.5 (huge!)
complexity = lambda * (lvs_penalty + vars_penalty + sparsity_penalty)
# sparsity_penalty could be 0, 1.0, 2.0, or 4.5 (sum of multiple penalties)

# NEW: complexity is ~0.6 (10% of performance range ~6)
complexity = 0.3 * (lambda/0.15) * (lvs_penalty + vars_penalty)
# No sparsity penalty at all
```

---

### Bug 5: Incorrect Preprocessing Labels for Subsets

**Symptom:** All derivative subsets showed as "raw" in results

**Cause:** Passing `{"name": "raw", ...}` to skip preprocessing, which also changed labels

**Fix:** Keep original `preprocess_cfg` for labeling, use `skip_preprocessing=True` flag

**Code:**
```python
# OLD:
_run_single_config(X_preprocessed, y, model,
                   {"name": "raw", "deriv": None, ...})  # Wrong label!

# NEW:
_run_single_config(X_preprocessed, y, model,
                   preprocess_cfg,  # Correct label (deriv_d2)
                   skip_preprocessing=True)  # Flag to skip reapplying
```

---

## Validation & Testing

### Test Dataset Requirements

**Minimum for testing:**
- 50+ samples
- 100+ wavelengths
- Continuous target (regression) or discrete (classification)

**Recommended test files:**
- Small dataset: 50 samples × 100 wavelengths (fast validation)
- Medium dataset: 100 samples × 500 wavelengths (realistic)
- Large dataset: 500 samples × 2000 wavelengths (performance test)

### Test Cases

**1. Basic preprocessing:**
```julia
X = randn(50, 100)
y = randn(50)

# Test SNV
X_snv = apply_snv(X)
@test all(abs.(mean(X_snv, dims=1)) .< 1e-10)  # Mean ≈ 0
@test all(abs.(std(X_snv, dims=1) .- 1) .< 1e-10)  # Std ≈ 1

# Test derivative
X_deriv = apply_derivative(X, deriv=2, window=17)
@test size(X_deriv, 2) < size(X, 2)  # Features reduced
```

**2. Subset analysis:**
```julia
# Test that variable subsets work with derivatives
X = randn(50, 101)
y = randn(50)

# Apply derivative: 101 → ~84 features
X_deriv = apply_derivative(X, deriv=2, window=17)

# Get feature importances on derivative data
importances = compute_importances(X_deriv, y)
@test length(importances) == size(X_deriv, 2)

# Select top-10
top_indices = sortperm(importances, rev=true)[1:10]
X_subset = X_deriv[:, top_indices]
@test size(X_subset) == (50, 10)

# Should NOT reapply derivative to subset
# (This was the bug!)
```

**3. Region analysis:**
```julia
# Test region detection
wavelengths = collect(1000:10:2000)  # 1000-2000nm, 10nm spacing
X = randn(50, length(wavelengths))
y = randn(50)

regions = compute_region_correlations(X, y, wavelengths,
                                      region_size=50, overlap=25)

@test length(regions) > 0
@test all(r["end"] - r["start"] == 50 for r in regions)
```

**4. Scoring:**
```julia
# Test that better R² ranks higher
results = [
    (R2=0.9, n_vars=50, LVs=5),
    (R2=0.3, n_vars=10, LVs=2),
    (R2=0.85, n_vars=30, LVs=4)
]

scores = compute_composite_scores(results)
ranks = sortperm(scores)  # Lower score = better

@test ranks[1] == 1  # R²=0.9 should be rank 1
@test ranks[3] == 2  # R²=0.3 should be rank 3 (worst)
```

**5. Full integration test:**
```julia
# Load test data
X, y, wavelengths = load_test_data()

# Run full search
results = run_search(
    X, y,
    task_type="regression",
    models=["PLS", "Ridge"],
    preprocessing=["raw", "snv", "deriv"],
    enable_variable_subsets=true,
    variable_counts=[10, 20, 50],
    enable_region_subsets=true,
    n_top_regions=5
)

# Validate results
@test size(results, 1) > 0
@test "Rank" in names(results)
@test "R2" in names(results)
@test "SubsetTag" in names(results)

# Check that best model has best R²
best_model = results[results.Rank .== 1, :]
@test best_model.R2[1] == maximum(results.R2)
```

---

## Performance Considerations

### Python Performance (Current)

**Bottlenecks:**
1. Cross-validation loops (mitigated with `joblib` parallel)
2. Feature importance computation (especially permutation for MLP)
3. Region correlation computation (nested loops)

**Timing (100 samples × 500 wavelengths):**
- PLS with 3 preprocessing × 8 components: ~30 seconds
- RandomForest with 3 preprocessing × 6 configs: ~2 minutes
- Full search (5 models, 3 preprocessing, subsets): ~10 minutes

### Julia Optimization Opportunities

**1. Type stability:**
```julia
# BAD (type-unstable)
function process_data(X, method)
    if method == "snv"
        return apply_snv(X)  # Returns Matrix{Float64}
    else
        return X  # Could be different type
    end
end

# GOOD (type-stable)
function process_data(X::Matrix{Float64}, method::String)::Matrix{Float64}
    if method == "snv"
        return apply_snv(X)
    else
        return copy(X)  # Ensure same type
    end
end
```

**2. Preallocate arrays:**
```julia
# BAD
results = []
for i in 1:n
    push!(results, compute_metric(data[i]))
end

# GOOD
results = Vector{Float64}(undef, n)
for i in 1:n
    results[i] = compute_metric(data[i])
end
```

**3. Use views instead of copies:**
```julia
# BAD
X_subset = X[:, indices]  # Copies data

# GOOD
X_subset = @view X[:, indices]  # No copy, just reference
```

**4. Parallelize CV folds:**
```julia
using Distributed

# Parallel CV
fold_results = @distributed (vcat) for fold in folds
    run_fold(X, y, fold.train_idx, fold.test_idx)
end
```

**Expected speedup:** 2-5x faster than Python (with proper optimization)

---

## Recommended Julia Packages

### Core ML
- `MLJ.jl` - Machine learning framework (like sklearn)
- `MultivariateStats.jl` - PCA, PLS, etc.
- `GLMNet.jl` - Ridge, Lasso, ElasticNet
- `DecisionTree.jl` - RandomForest
- `Flux.jl` - Neural networks (for MLP, NeuralBoosted)

### Numerical Computing
- `Statistics` (built-in) - Mean, std, cor
- `StatsBase.jl` - Additional stats functions
- `LinearAlgebra` (built-in) - Matrix operations

### Data Processing
- `DataFrames.jl` - Like pandas
- `CSV.jl` - CSV file I/O
- `Tables.jl` - Table interface

### Preprocessing
- `SavitzkyGolay.jl` - Derivative filters (or implement manually)
- `DSP.jl` - Signal processing

### Parallel Computing
- `Distributed` (built-in) - Multi-process parallelism
- `Threads` (built-in) - Multi-thread parallelism

### CLI (Phase 1)
- `ArgParse.jl` - Command-line argument parsing
- `ProgressMeter.jl` - Progress bars

### GUI (Phase 2 - Later)
- `Makie.jl` - Modern plotting (recommended)
- `Plots.jl` - Simple plotting
- `Gtk.jl` or `QML.jl` - Desktop GUI
- `Genie.jl` - Web-based GUI (alternative)

---

## Phase 1 Implementation Roadmap

### Week 1: Core Preprocessing
- [ ] SNV implementation
- [ ] Savitzky-Golay derivatives
- [ ] Pipeline builder
- [ ] Unit tests for preprocessing

### Week 2: Model Definitions
- [ ] PLS wrapper
- [ ] Ridge/Lasso/ElasticNet wrapper
- [ ] RandomForest wrapper
- [ ] Model grid generator
- [ ] Unit tests for models

### Week 3: Cross-Validation
- [ ] CV splitter
- [ ] Single-fold runner
- [ ] Parallel CV execution
- [ ] Metrics computation (RMSE, R², Accuracy, AUC)
- [ ] Unit tests for CV

### Week 4: Subset Analysis
- [ ] Feature importance computation
- [ ] Variable subset selection
- [ ] Region correlation computation
- [ ] Region subset creation
- [ ] Handle derivative subsets correctly (no double-preprocessing)
- [ ] Unit tests for subsets

### Week 5: Scoring & Search
- [ ] Composite scoring function (90/10 split)
- [ ] Ranking system
- [ ] Main search loop
- [ ] Result dataframe creation
- [ ] Integration tests

### Week 6: CLI & File I/O
- [ ] CSV/SPC file loading
- [ ] Command-line interface
- [ ] Result CSV export
- [ ] Basic text-based progress output
- [ ] End-to-end tests

### Week 7: Polish & Validation
- [ ] Performance optimization
- [ ] Compare results with Python implementation
- [ ] Documentation
- [ ] Examples

**Total estimated time:** 6-8 weeks for Phase 1

---

## Phase 2 (Later): Advanced GUI

**Not needed for initial port.** Keep using Python GUI for:
- Interactive plots
- Cursor region selection
- Multi-tab interface
- Real-time progress monitoring
- Results table with sorting
- Model refinement UI

**When to implement Phase 2:**
- After Phase 1 is validated and stable
- When Julia implementation is primary tool
- When Python GUI maintenance becomes burden

**Estimated Phase 2 time:** 4-6 weeks

---

## Validation Strategy

### 1. Unit Tests (Each Module)
- Test each function independently
- Cover edge cases
- Fast execution (<1 second per test)

### 2. Integration Tests (Full Pipeline)
- Test complete workflows
- Use small synthetic datasets
- Compare with Python results

### 3. Regression Tests (Real Data)
- Use your actual spectral data
- Run same analysis in Python and Julia
- Compare results (should match within numerical precision)

**Acceptance criteria:**
- RMSE/R² differences < 1e-6 (numerical precision)
- Rankings identical (or differ only in ties)
- All subsets generated correctly
- No double-preprocessing bugs

---

## Python GUI Enhancement (Parallel Work)

While porting to Julia, you can enhance Python GUI:

### Add Plots
- Spectra plot (wavelength vs intensity)
- Predictions vs actual plot
- Residuals plot
- Feature importance bar chart

### Add Cursor Region Selection
- Click-and-drag to select wavelength range
- Highlight selected region on spectrum
- Display selected wavelength range
- "Run with selected region" button

**These features stay in Python GUI for now.** Port to Julia in Phase 2.

---

## File Format Compatibility

**Ensure Julia can read Python outputs:**

### Results CSV Format
```csv
Rank,Task,Model,Preprocess,Deriv,Window,Poly,LVs,n_vars,full_vars,SubsetTag,RMSE,R2,CompositeScore,top_vars
1,regression,PLS,deriv_d2,2,17,3,5,84,101,full,0.123,0.89,-2.1,"450nm,550nm,..."
2,regression,PLS,deriv_d2,2,17,3,3,20,101,top20,0.145,0.85,-1.9,"450nm,550nm,..."
```

**Column types:**
- `Rank`: Integer
- `Deriv`, `Window`, `Poly`, `LVs`, `n_vars`, `full_vars`: Integer or NaN
- `RMSE`, `R2`, `CompositeScore`: Float
- Rest: String

**Julia can read with:**
```julia
using CSV, DataFrames
results = CSV.read("results.csv", DataFrame)
```

---

## Common Pitfalls to Avoid

### 1. Feature Indexing After Derivatives
**❌ Wrong:**
```julia
X_deriv = apply_derivative(X)  # 101 → 84 features
importances = compute_importances(X_deriv, y)
top_indices = top_n_indices(importances, 10)
X_subset = X[:, top_indices]  # BUG! Indices are for X_deriv, not X
X_subset_deriv = apply_derivative(X_subset)  # BUG! Double preprocessing
```

**✅ Correct:**
```julia
X_deriv = apply_derivative(X)  # 101 → 84 features
importances = compute_importances(X_deriv, y)
top_indices = top_n_indices(importances, 10)
X_subset = X_deriv[:, top_indices]  # Use derivative data
# Don't reapply derivative!
```

### 2. Region Subsets with Raw Indices
**❌ Wrong:**
```julia
regions = compute_regions(X, y, wavelengths)  # Computed on raw
X_deriv = apply_derivative(X)  # 101 → 84 features
X_region = X_deriv[:, regions[1].indices]  # BUG! Indices don't match
```

**✅ Correct:**
```julia
X_deriv = apply_derivative(X)  # 101 → 84 features
regions = compute_regions(X_deriv, y, wavelengths)  # Compute on derivative
X_region = X_deriv[:, regions[1].indices]  # Indices match
```

### 3. Forgetting to Skip Preprocessing
**❌ Wrong:**
```julia
X_deriv = apply_derivative(X)
X_subset = X_deriv[:, indices]
# Now run CV which will apply derivative again!
result = run_cv(X_subset, y, preprocess_cfg)  # BUG! Double preprocessing
```

**✅ Correct:**
```julia
X_deriv = apply_derivative(X)
X_subset = X_deriv[:, indices]
result = run_cv(X_subset, y, preprocess_cfg, skip_preprocessing=true)
```

### 4. Wrong Complexity Penalty Scale
**❌ Wrong:**
```julia
# Performance z-scores: range ±3
# Complexity penalty: range 0-10
# Complexity dominates! (like old Python bug)
score = performance + 2.0 * complexity
```

**✅ Correct:**
```julia
# Performance z-scores: range ±3 (total range ~6)
# Complexity: scaled to ~0.6 (10% of 6)
complexity_scale = 0.3 * lambda_penalty / 0.15
score = performance + complexity_scale * complexity
```

---

## Documentation Requirements

### Code Comments
- Docstrings for all public functions
- Explain why, not just what
- Note any non-obvious behavior
- Reference Python equivalent function

### README
- Installation instructions
- Quick start example
- Link to full docs

### Examples
- `examples/basic_analysis.jl` - Simple workflow
- `examples/custom_models.jl` - Adding custom models
- `examples/compare_with_python.jl` - Validation script

### API Reference
- Auto-generated from docstrings
- Organized by module
- Cross-references

---

## Next Steps

### Before Starting Port
1. ✅ Review this document thoroughly
2. ✅ Set up Julia development environment
3. ✅ Install recommended packages
4. ✅ Create test datasets (small, medium, large)
5. ✅ Run Python version, save results for validation

### During Port
1. Follow implementation roadmap (weeks 1-7)
2. Write tests alongside implementation
3. Compare with Python results frequently
4. Document as you go

### After Phase 1 Complete
1. Performance benchmarking
2. Real-world validation
3. User feedback
4. Decide on Phase 2 timing

### Python GUI Work (Parallel)
1. Add plots to Tab 3 (Progress)
2. Add cursor selection to Tab 5 (Refine Model)
3. Keep Python GUI as primary interface during Phase 1

---

## Questions to Consider

1. **Package choice:** MLJ.jl vs custom wrappers?
   - MLJ: More features, steeper learning curve
   - Custom: More control, more work

2. **Parallelization:** Distributed vs Threads?
   - Distributed: Better for CPU-bound tasks, more overhead
   - Threads: Lower overhead, may have GIL-like issues with some packages

3. **NeuralBoosted:** Port or use existing Julia boosting?
   - Port: Matches Python exactly
   - EvoTrees.jl: Faster, well-maintained, but different architecture

4. **CLI vs Notebook:** Primary interface for Phase 1?
   - CLI: Traditional, scriptable
   - Jupyter/Pluto notebook: Interactive, better for exploration

---

## Success Criteria

**Phase 1 is complete when:**
- ✅ All core algorithms ported
- ✅ All unit tests pass
- ✅ Integration tests match Python results (within numerical precision)
- ✅ Regression tests pass on real data
- ✅ CLI works for basic workflows
- ✅ Documentation complete
- ✅ Performance is competitive (ideally 2-5x faster than Python)

**Phase 2 begins when:**
- Phase 1 validated and stable
- Julia implementation is primary tool
- Need for advanced GUI features emerges

---

## Contact Points

**For questions during port:**
- Refer to this document
- Check Python source code (search.py, scoring.py, regions.py, etc.)
- Review today's bug fixes (see "Critical Bugs Fixed Today" section)
- Compare with Python test outputs

**Documentation sources:**
- Python docstrings
- DOCUMENTATION_INDEX.md (for broader context)
- RESTART_README.md (recent bug fixes)
- JULIA_PORT_GUIDE.md (if it exists, older version)

---

## Final Notes

**What makes this port straightforward:**
- ✅ Python implementation is debugged and working
- ✅ Algorithms are well-documented
- ✅ Clear separation between core and GUI
- ✅ Test data available for validation

**What to watch out for:**
- ⚠️ Derivative preprocessing changes feature space
- ⚠️ Feature indices don't map between raw and preprocessed data
- ⚠️ Complexity penalty must be scaled properly (~10% of performance)
- ⚠️ Region subsets need to run for ALL models, not just PLS/RF/MLP

**Philosophy:**
- **Performance first** - Complexity only as tiebreaker
- **No arbitrary penalties** - Let data drive ranking
- **Comprehensive search** - Test full spectrum + subsets
- **Clear results** - Rank 1 should be obviously best

---

**Good luck with the Julia port! 🚀**

**Phase 1 estimate:** 6-8 weeks
**Phase 2 estimate:** 4-6 weeks (later)

**Status:** ✅ Ready to begin
