# Cross-Validation Module Guide

## Overview

The `cv.jl` module provides a comprehensive k-fold cross-validation framework for the Julia spectral prediction system. It supports both regression and classification tasks with parallel processing capabilities.

## Table of Contents

1. [Key Features](#key-features)
2. [Quick Start](#quick-start)
3. [Core Functions](#core-functions)
4. [Critical Implementation Details](#critical-implementation-details)
5. [Metrics](#metrics)
6. [Parallel Execution](#parallel-execution)
7. [Integration with Other Modules](#integration-with-other-modules)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

---

## Key Features

- **K-fold cross-validation** with automatic fold creation
- **Skip preprocessing mode** for derivative subsets (critical bug prevention)
- **Parallel execution** via multi-threading
- **Comprehensive metrics**:
  - Regression: RMSE, R², MAE
  - Classification: Accuracy, ROC AUC, Precision, Recall
- **Type-stable implementation** for optimal performance
- **Reproducible results** via fixed random seeds

---

## Quick Start

```julia
using Random
include("src/cv.jl")
include("src/models.jl")
include("src/preprocessing.jl")

# Create data
X = rand(100, 50)  # 100 samples, 50 features
y = rand(100)       # Target values

# Build model and preprocessing config
model = PLSModel(10)
preprocess_config = Dict("name" => "snv")

# Run 5-fold cross-validation
results = run_cross_validation(
    X, y, model, "PLS",
    preprocess_config, "regression",
    n_folds=5
)

# View results
println("RMSE: $(results["RMSE_mean"]) ± $(results["RMSE_std"])")
println("R²: $(results["R2_mean"]) ± $(results["R2_std"])")
```

---

## Core Functions

### 1. `create_cv_folds`

Creates k-fold cross-validation splits.

**Signature:**
```julia
create_cv_folds(n_samples::Int, n_folds::Int=5)::Vector{Tuple{Vector{Int}, Vector{Int}}}
```

**Arguments:**
- `n_samples`: Total number of samples
- `n_folds`: Number of folds (default: 5)

**Returns:**
- Array of `(train_indices, test_indices)` tuples

**Example:**
```julia
folds = create_cv_folds(100, 5)
# Returns 5 tuples, each with ~80 train and ~20 test indices

# Access first fold
train_idx, test_idx = folds[1]
```

**Notes:**
- Uses fixed random seed (42) for reproducibility
- Handles uneven divisions gracefully
- Train and test sets are always disjoint

---

### 2. `run_single_fold`

Executes a single cross-validation fold.

**Signature:**
```julia
run_single_fold(
    X::Matrix{Float64},
    y::Vector{Float64},
    train_idx::Vector{Int},
    test_idx::Vector{Int},
    model,
    model_name::String,
    preprocess_config::Dict{String, Any},
    task_type::String;
    skip_preprocessing::Bool=false
)::Dict{String, Float64}
```

**Arguments:**
- `X`: Full feature matrix
- `y`: Full target vector
- `train_idx`: Training set indices
- `test_idx`: Test set indices
- `model`: Model instance
- `model_name`: Model type name
- `preprocess_config`: Preprocessing configuration
- `task_type`: "regression" or "classification"
- `skip_preprocessing`: Skip preprocessing flag (default: false)

**Returns:**
- Dictionary of metrics

**Example:**
```julia
folds = create_cv_folds(100, 5)
train_idx, test_idx = folds[1]

model = PLSModel(10)
metrics = run_single_fold(
    X, y, train_idx, test_idx,
    model, "PLS",
    Dict("name" => "snv"),
    "regression"
)
# Returns: Dict("RMSE" => ..., "R2" => ..., "MAE" => ...)
```

---

### 3. `run_cross_validation`

Main cross-validation function (sequential execution).

**Signature:**
```julia
run_cross_validation(
    X::Matrix{Float64},
    y::Vector{Float64},
    model,
    model_name::String,
    preprocess_config::Dict{String, Any},
    task_type::String;
    n_folds::Int=5,
    skip_preprocessing::Bool=false
)::Dict{String, Any}
```

**Arguments:**
- Same as `run_single_fold` but without indices
- `n_folds`: Number of CV folds

**Returns:**
- Dictionary with:
  - Mean metrics: `"RMSE_mean"`, `"R2_mean"`, etc.
  - Std metrics: `"RMSE_std"`, `"R2_std"`, etc.
  - Individual fold results: `"cv_scores"`
  - Metadata: `"n_folds"`, `"task_type"`

**Example:**
```julia
results = run_cross_validation(
    X, y, PLSModel(10), "PLS",
    Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2),
    "regression",
    n_folds=5
)

println("Mean RMSE: ", results["RMSE_mean"])
println("Std RMSE: ", results["RMSE_std"])

# Access individual folds
for (i, fold) in enumerate(results["cv_scores"])
    println("Fold $i: RMSE = ", fold["RMSE"])
end
```

---

### 4. `run_cross_validation_parallel`

Parallel version using multi-threading.

**Signature:**
Same as `run_cross_validation`

**Usage:**
```bash
# Start Julia with multiple threads
julia -t 4 your_script.jl
```

```julia
# Will automatically use 4 threads
results = run_cross_validation_parallel(
    X, y, model, "PLS",
    preprocess_config, "regression"
)
```

**Speedup:**
- Approximately linear with number of threads
- Best for large models (RandomForest, MLP)
- Minimal overhead for simple models (Ridge, Lasso)

---

## Critical Implementation Details

### Skip Preprocessing Mode

**Problem:** When working with derivative subsets, data is already preprocessed at the parent level. Re-applying preprocessing causes incorrect results.

**Solution:** Use `skip_preprocessing=true`

**Example:**
```julia
# Scenario: Derivative subset workflow

# 1. Parent preprocessing (applied once)
X_preprocessed = apply_preprocessing(X, preprocess_config)

# 2. Create derivative subset (e.g., wavelength selection)
selected_wavelengths = [10, 20, 30, 40, 50]
X_subset = X_preprocessed[:, selected_wavelengths]

# 3. CV on subset - MUST skip preprocessing!
results = run_cross_validation(
    X_subset, y, model, "PLS",
    preprocess_config, "regression",
    skip_preprocessing=true  # CRITICAL!
)
```

**What Happens Under the Hood:**

```julia
# When skip_preprocessing=false (DEFAULT):
X_train = X[train_idx, :]
X_test = X[test_idx, :]
X_train_processed = apply_preprocessing(X_train, config)  # Applied
X_test_processed = apply_preprocessing(X_test, config)    # Applied

# When skip_preprocessing=true:
X_train = X[train_idx, :]  # Used as-is
X_test = X[test_idx, :]    # Used as-is
# NO preprocessing applied!
```

---

## Metrics

### Regression Metrics

Computed by `compute_regression_metrics`:

1. **RMSE** (Root Mean Squared Error)
   - Formula: `√(mean((y_true - y_pred)²))`
   - Units: Same as target variable
   - Lower is better

2. **R²** (Coefficient of Determination)
   - Formula: `1 - (SS_res / SS_tot)`
   - Range: (-∞, 1.0], typically [0, 1]
   - Higher is better
   - Can be negative if model is worse than mean

3. **MAE** (Mean Absolute Error)
   - Formula: `mean(|y_true - y_pred|)`
   - Units: Same as target variable
   - Robust to outliers
   - Lower is better

### Classification Metrics

Computed by `compute_classification_metrics`:

1. **Accuracy**
   - Formula: `(TP + TN) / (TP + TN + FP + FN)`
   - Range: [0, 1]
   - Threshold: 0.5

2. **ROC AUC** (Area Under ROC Curve)
   - Range: [0, 1]
   - 0.5 = random classifier
   - 1.0 = perfect classifier

3. **Precision**
   - Formula: `TP / (TP + FP)`
   - Range: [0, 1]

4. **Recall**
   - Formula: `TP / (TP + FN)`
   - Range: [0, 1]

---

## Parallel Execution

### Threading Setup

```bash
# Set environment variable
export JULIA_NUM_THREADS=4

# Or start Julia with -t flag
julia -t 4
julia -t auto  # Use all cores
```

### Check Thread Count

```julia
println("Threads available: ", Threads.nthreads())
```

### Performance Comparison

```julia
using BenchmarkTools

# Sequential
@time results_seq = run_cross_validation(
    X, y, model, "PLS", config, "regression", n_folds=10
)

# Parallel
@time results_par = run_cross_validation_parallel(
    X, y, model, "PLS", config, "regression", n_folds=10
)
```

**Expected Speedups:**
- 4 threads: ~3.5x (accounting for overhead)
- 8 threads: ~6-7x
- 16 threads: ~12-14x

---

## Integration with Other Modules

### With models.jl

```julia
# Get model configurations
configs = get_model_configs("PLS")
# Returns: [Dict("n_components" => 1), Dict("n_components" => 2), ...]

# Test each configuration
for config in configs
    model = build_model("PLS", config, "regression")
    results = run_cross_validation(X, y, model, "PLS", preprocess_config, "regression")
    println("Config: $config, RMSE: $(results["RMSE_mean"])")
end
```

### With preprocessing.jl

```julia
# Define preprocessing options
preprocess_options = [
    Dict("name" => "raw"),
    Dict("name" => "snv"),
    Dict("name" => "deriv", "deriv" => 1, "window" => 11, "polyorder" => 2),
    Dict("name" => "snv_deriv", "deriv" => 2, "window" => 17, "polyorder" => 3)
]

# Test each
for config in preprocess_options
    results = run_cross_validation(X, y, model, "PLS", config, "regression")
    println("$(config["name"]): RMSE = $(results["RMSE_mean"])")
end
```

### With scoring.jl

```julia
using DataFrames

# Run CV for multiple configurations
cv_results = []
for config in model_configs
    model = build_model("PLS", config, "regression")
    results = run_cross_validation(X, y, model, "PLS", preprocess_config, "regression")
    push!(cv_results, (config, results))
end

# Create DataFrame for scoring
df = DataFrame(
    RMSE = [r["RMSE_mean"] for (_, r) in cv_results],
    R2 = [r["R2_mean"] for (_, r) in cv_results],
    n_vars = fill(size(X, 2), length(cv_results)),
    full_vars = fill(size(X, 2), length(cv_results)),
    lvs = [c["n_components"] for (c, _) in cv_results],
    task_type = fill("regression", length(cv_results))
)

# Rank using scoring module
rank_results!(df)
```

---

## Common Patterns

### Pattern 1: Hyperparameter Tuning

```julia
# Grid search over PLS components
best_rmse = Inf
best_n_components = 0

for n_comp in [1, 2, 3, 5, 7, 10, 15, 20]
    model = PLSModel(n_comp)
    results = run_cross_validation(
        X, y, model, "PLS",
        Dict("name" => "snv"), "regression"
    )

    if results["RMSE_mean"] < best_rmse
        best_rmse = results["RMSE_mean"]
        best_n_components = n_comp
    end
end

println("Best: n_components = $best_n_components")
```

### Pattern 2: Nested Cross-Validation

```julia
# Outer loop: model evaluation
outer_folds = create_cv_folds(n_samples, 5)

for (i, (train_idx, test_idx)) in enumerate(outer_folds)
    X_train = X[train_idx, :]
    y_train = y[train_idx]

    # Inner loop: hyperparameter tuning
    best_model = nothing
    best_score = Inf

    for n_comp in [5, 10, 15]
        model = PLSModel(n_comp)
        results = run_cross_validation(
            X_train, y_train, model, "PLS",
            Dict("name" => "snv"), "regression",
            n_folds=3  # Inner CV
        )

        if results["RMSE_mean"] < best_score
            best_score = results["RMSE_mean"]
            best_model = model
        end
    end

    # Evaluate best model on outer test fold
    # ... (evaluation code)
end
```

### Pattern 3: Model Comparison

```julia
models = [
    ("PLS", PLSModel(10)),
    ("Ridge", RidgeModel(1.0)),
    ("Lasso", LassoModel(0.1)),
    ("RandomForest", RandomForestModel(100, "sqrt"))
]

results_comparison = Dict()

for (name, model) in models
    results = run_cross_validation(
        X, y, model, name,
        Dict("name" => "snv"), "regression"
    )
    results_comparison[name] = results
end

# Print comparison
for (name, results) in sort(collect(results_comparison), by=x->x[2]["RMSE_mean"])
    println("$name: RMSE = $(results["RMSE_mean"]) ± $(results["RMSE_std"])")
end
```

---

## Troubleshooting

### Issue 1: Double Preprocessing

**Symptom:** Results are worse than expected, or errors occur during preprocessing.

**Cause:** Data is already preprocessed but `skip_preprocessing=false` (default).

**Solution:**
```julia
# If X is already preprocessed:
results = run_cross_validation(
    X, y, model, "PLS", config, "regression",
    skip_preprocessing=true  # Add this!
)
```

### Issue 2: Negative R²

**Symptom:** R² is negative.

**Explanation:** Model is worse than predicting the mean. This is normal for poor models.

**Solution:** Try different models, preprocessing, or hyperparameters.

### Issue 3: High Variance in Results

**Symptom:** Large standard deviations in metrics.

**Causes:**
- Small dataset
- Outliers in specific folds
- Unstable model

**Solutions:**
```julia
# Increase number of folds
results = run_cross_validation(X, y, model, "PLS", config, "regression", n_folds=10)

# Use more stable model
model = RidgeModel(1.0)  # Instead of Lasso

# Remove outliers before CV
```

### Issue 4: Slow Performance

**Symptom:** CV takes too long.

**Solutions:**

1. Use parallel version:
```julia
# julia -t 4
results = run_cross_validation_parallel(...)
```

2. Reduce folds:
```julia
results = run_cross_validation(..., n_folds=3)
```

3. Use simpler model:
```julia
# Use Ridge instead of RandomForest for initial testing
model = RidgeModel(1.0)
```

### Issue 5: Memory Issues

**Symptom:** Out of memory errors with large datasets.

**Solutions:**

1. Reduce parallel threads:
```julia
# julia -t 2  # Instead of -t 8
```

2. Use smaller models:
```julia
# Reduce components
model = PLSModel(5)  # Instead of PLSModel(20)

# Reduce trees
model = RandomForestModel(50, "sqrt")  # Instead of 200
```

3. Process in batches (advanced):
```julia
# Split data into chunks and process separately
```

---

## Performance Tips

1. **Type Stability**: All functions are type-stable for maximum performance
2. **Preallocations**: Results arrays are preallocated
3. **Fixed Seeds**: Use `Random.seed!(42)` for reproducibility
4. **Parallel vs Sequential**: Use parallel only for large models/datasets
5. **Fold Count**: 5-10 folds is usually sufficient; more doesn't always improve estimates

---

## API Reference Summary

| Function | Purpose | Parallel |
|----------|---------|----------|
| `create_cv_folds` | Create fold indices | No |
| `compute_regression_metrics` | Compute regression metrics | No |
| `compute_classification_metrics` | Compute classification metrics | No |
| `run_single_fold` | Execute one fold | No |
| `run_cross_validation` | Full CV (sequential) | No |
| `run_cross_validation_parallel` | Full CV (parallel) | Yes |

---

## Version History

- **v1.0** (2025-01-29): Initial implementation
  - K-fold CV for regression and classification
  - Skip preprocessing mode
  - Parallel execution support
  - Comprehensive metrics

---

## See Also

- `models.jl`: Model definitions and hyperparameter configs
- `preprocessing.jl`: SNV and derivative transformations
- `scoring.jl`: Composite scoring and ranking
- `examples/cv_usage_examples.jl`: Complete usage examples
