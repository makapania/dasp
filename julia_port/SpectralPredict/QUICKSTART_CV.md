# Cross-Validation Quick Start Guide

## 5-Minute Setup

### 1. Import the Module

```julia
include("src/cv.jl")
include("src/models.jl")
include("src/preprocessing.jl")
```

### 2. Prepare Your Data

```julia
# Load your spectral data
X = your_spectral_matrix  # n_samples × n_wavelengths
y = your_target_vector    # n_samples

# Example:
# X = rand(100, 50)  # 100 samples, 50 wavelengths
# y = rand(100)       # 100 target values
```

### 3. Run Cross-Validation

```julia
# Build model
model = PLSModel(10)  # PLS with 10 components

# Configure preprocessing
preprocess_config = Dict("name" => "snv")

# Run 5-fold CV
results = run_cross_validation(
    X, y, model, "PLS",
    preprocess_config, "regression"
)

# View results
println("RMSE: $(results["RMSE_mean"]) ± $(results["RMSE_std"])")
println("R²: $(results["R2_mean"]) ± $(results["R2_std"])")
```

**Done!** That's all you need for basic CV.

---

## Common Use Cases

### Use Case 1: Hyperparameter Tuning

**Goal:** Find best number of PLS components

```julia
best_rmse = Inf
best_n_components = 0

for n_comp in [1, 3, 5, 10, 15, 20]
    model = PLSModel(n_comp)
    results = run_cross_validation(
        X, y, model, "PLS",
        Dict("name" => "snv"), "regression"
    )

    println("n_components=$n_comp: RMSE=$(results["RMSE_mean"])")

    if results["RMSE_mean"] < best_rmse
        best_rmse = results["RMSE_mean"]
        best_n_components = n_comp
    end
end

println("\nBest: n_components=$best_n_components")
```

---

### Use Case 2: Compare Preprocessing Methods

**Goal:** Find best preprocessing approach

```julia
preprocessing_options = [
    ("Raw", Dict("name" => "raw")),
    ("SNV", Dict("name" => "snv")),
    ("1st Derivative", Dict("name" => "deriv", "deriv" => 1, "window" => 11, "polyorder" => 2)),
    ("SNV + 1st Deriv", Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2))
]

model = PLSModel(10)

for (name, config) in preprocessing_options
    results = run_cross_validation(
        X, y, model, "PLS", config, "regression"
    )
    println("$name: RMSE=$(results["RMSE_mean"])")
end
```

---

### Use Case 3: Compare Models

**Goal:** Find best model type

```julia
models = [
    ("PLS", PLSModel(10)),
    ("Ridge", RidgeModel(1.0)),
    ("Lasso", LassoModel(0.1)),
    ("RandomForest", RandomForestModel(50, "sqrt"))
]

preprocess_config = Dict("name" => "snv")

for (name, model) in models
    results = run_cross_validation(
        X, y, model, name,
        preprocess_config, "regression"
    )
    println("$name: RMSE=$(results["RMSE_mean"])")
end
```

---

### Use Case 4: Derivative Subset (CRITICAL - Skip Preprocessing)

**Goal:** CV on a wavelength subset from preprocessed data

```julia
# 1. Preprocess full dataset ONCE
preprocess_config = Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2)
X_preprocessed = apply_preprocessing(X, preprocess_config)

# 2. Select wavelength subset (e.g., from feature selection)
selected_wavelengths = [10, 20, 30, 40, 50]  # Indices of important wavelengths
X_subset = X_preprocessed[:, selected_wavelengths]

# 3. Run CV with skip_preprocessing=true
model = PLSModel(5)
results = run_cross_validation(
    X_subset, y, model, "PLS",
    preprocess_config, "regression",
    skip_preprocessing=true  # ⚠️ CRITICAL! Don't re-preprocess
)

println("Subset RMSE: $(results["RMSE_mean"])")
```

**Why skip_preprocessing=true?**
- Data is already preprocessed in step 1
- Re-preprocessing would apply SNV twice (incorrect!)
- This is the most common bug - always set `skip_preprocessing=true` for subsets

---

### Use Case 5: Classification

**Goal:** Classify samples into categories

```julia
# Binary classification (0.0 or 1.0)
y_class = rand([0.0, 1.0], 100)

model = RidgeModel(1.0)
results = run_cross_validation(
    X, y_class, model, "Ridge",
    Dict("name" => "raw"),
    "classification"  # ⚠️ Note: "classification" not "regression"
)

println("Accuracy: $(results["Accuracy_mean"])")
println("ROC AUC: $(results["ROC_AUC_mean"])")
```

---

### Use Case 6: Parallel Execution (Fast!)

**Goal:** Speed up CV with multi-threading

**Setup:**
```bash
# Start Julia with 4 threads
julia -t 4 your_script.jl
```

**Code:**
```julia
# Use parallel version
results = run_cross_validation_parallel(
    X, y, model, "PLS",
    preprocess_config, "regression"
)
# ⚠️ Same interface as regular run_cross_validation!
```

**Expected Speedup:**
- 4 threads: ~3-4x faster
- 8 threads: ~6-7x faster

---

## Model Options

### PLS
```julia
model = PLSModel(n_components)
# Try n_components: 1, 3, 5, 10, 15, 20
```

### Ridge
```julia
model = RidgeModel(alpha)
# Try alpha: 0.001, 0.01, 0.1, 1.0, 10.0
```

### Lasso
```julia
model = LassoModel(alpha)
# Try alpha: 0.001, 0.01, 0.1, 1.0, 10.0
```

### Elastic Net
```julia
model = ElasticNetModel(alpha, l1_ratio)
# Try alpha: 0.01, 0.1, 1.0
# Try l1_ratio: 0.1, 0.5, 0.9
```

### Random Forest
```julia
model = RandomForestModel(n_trees, max_features)
# Try n_trees: 50, 100, 200
# Try max_features: "sqrt", "log2"
```

### MLP (Neural Network)
```julia
model = MLPModel(hidden_layers, learning_rate)
# Try hidden_layers: (50,), (100,), (50, 50)
# Try learning_rate: 0.001, 0.01
```

---

## Preprocessing Options

### No Preprocessing
```julia
config = Dict("name" => "raw")
```

### SNV Only
```julia
config = Dict("name" => "snv")
```

### Derivative Only
```julia
config = Dict(
    "name" => "deriv",
    "deriv" => 1,        # 1 or 2 (1st or 2nd derivative)
    "window" => 11,      # Window size (odd number)
    "polyorder" => 2     # Polynomial order
)
```

### SNV + Derivative
```julia
config = Dict(
    "name" => "snv_deriv",
    "deriv" => 1,
    "window" => 11,
    "polyorder" => 2
)
```

### Derivative + SNV
```julia
config = Dict(
    "name" => "deriv_snv",
    "deriv" => 2,
    "window" => 17,
    "polyorder" => 3
)
```

---

## Understanding Results

### Result Structure
```julia
results = Dict(
    "RMSE_mean" => 0.5,      # Mean RMSE across folds
    "RMSE_std" => 0.1,       # Std dev of RMSE
    "R2_mean" => 0.85,       # Mean R²
    "R2_std" => 0.05,        # Std dev of R²
    "MAE_mean" => 0.4,       # Mean MAE
    "MAE_std" => 0.08,       # Std dev of MAE
    "cv_scores" => [...],    # Individual fold results
    "n_folds" => 5,          # Number of folds used
    "task_type" => "regression"
)
```

### Access Individual Folds
```julia
for (i, fold) in enumerate(results["cv_scores"])
    println("Fold $i:")
    println("  RMSE: $(fold["RMSE"])")
    println("  R²: $(fold["R2"])")
    println("  MAE: $(fold["MAE"])")
end
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Not Using skip_preprocessing for Subsets
```julia
# WRONG!
X_subset = X_preprocessed[:, selected_wavelengths]
results = run_cross_validation(
    X_subset, y, model, "PLS", config, "regression"
    # Missing skip_preprocessing=true!
)
```

**Fix:**
```julia
# CORRECT
results = run_cross_validation(
    X_subset, y, model, "PLS", config, "regression",
    skip_preprocessing=true  # Add this!
)
```

### ❌ Mistake 2: Wrong Task Type
```julia
# WRONG for classification!
results = run_cross_validation(
    X, y_class, model, "Ridge",
    config, "regression"  # Should be "classification"
)
```

**Fix:**
```julia
# CORRECT
results = run_cross_validation(
    X, y_class, model, "Ridge",
    config, "classification"  # Correct task type
)
```

### ❌ Mistake 3: Mismatched Dimensions
```julia
# WRONG!
X = rand(100, 50)
y = rand(90)  # Different size!
results = run_cross_validation(...)  # Will error
```

**Fix:**
```julia
# CORRECT
X = rand(100, 50)
y = rand(100)  # Same number of samples
```

---

## Troubleshooting

### Problem: "n_folds cannot exceed number of samples"

**Cause:** Too many folds for dataset size

**Fix:**
```julia
# If you have 50 samples, use max 50 folds (usually use 5-10)
results = run_cross_validation(..., n_folds=5)
```

### Problem: Negative R²

**Cause:** Model is performing worse than predicting the mean

**Fix:**
- Try different preprocessing
- Try different model type
- Tune hyperparameters
- Check if data is appropriate for the task

### Problem: High standard deviation in results

**Cause:** Results vary a lot between folds

**Fix:**
```julia
# Increase number of folds
results = run_cross_validation(..., n_folds=10)

# Or use more stable model
model = RidgeModel(1.0)  # Instead of Lasso
```

### Problem: CV is very slow

**Fix 1: Use Parallel CV**
```bash
julia -t 4 your_script.jl
```
```julia
results = run_cross_validation_parallel(...)
```

**Fix 2: Reduce Folds**
```julia
results = run_cross_validation(..., n_folds=3)  # Instead of 10
```

**Fix 3: Use Simpler Model**
```julia
model = RidgeModel(1.0)  # Instead of RandomForest
```

---

## Next Steps

1. **Read Full Documentation:** `docs/CV_MODULE_GUIDE.md`
2. **Run Examples:** `examples/cv_usage_examples.jl`
3. **Run Tests:** `test/test_cv.jl`
4. **Integration:** Use with `models.jl`, `preprocessing.jl`, `scoring.jl`

---

## Quick Reference Card

```julia
# ========================================
# BASIC TEMPLATE
# ========================================

# Import
include("src/cv.jl")
include("src/models.jl")
include("src/preprocessing.jl")

# Build
model = PLSModel(10)
config = Dict("name" => "snv")

# Run CV
results = run_cross_validation(
    X, y, model, "PLS", config, "regression"
)

# View
println("RMSE: $(results["RMSE_mean"]) ± $(results["RMSE_std"])")
println("R²: $(results["R2_mean"]) ± $(results["R2_std"])")

# ========================================
# DERIVATIVE SUBSET TEMPLATE
# ========================================

# Preprocess ONCE
X_preprocessed = apply_preprocessing(X, config)

# Select subset
X_subset = X_preprocessed[:, selected_wavelengths]

# CV with skip_preprocessing=true
results = run_cross_validation(
    X_subset, y, model, "PLS", config, "regression",
    skip_preprocessing=true  # ⚠️ CRITICAL!
)

# ========================================
# PARALLEL TEMPLATE
# ========================================

# Start Julia with: julia -t 4

# Run parallel CV
results = run_cross_validation_parallel(
    X, y, model, "PLS", config, "regression"
)
```

---

**Remember:** When in doubt, check the full documentation in `docs/CV_MODULE_GUIDE.md`!
