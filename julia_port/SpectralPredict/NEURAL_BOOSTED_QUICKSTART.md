# NeuralBoosted Quick Start Guide

## What is NeuralBoosted?

NeuralBoosted is a gradient boosting regressor that uses small neural networks as weak learners. It combines the power of ensemble methods (like Random Forest) with the flexibility of neural networks.

**Key Features:**
- Gradient boosting with neural network weak learners
- Automatic early stopping to prevent overfitting
- Feature importance based on network weights
- Ideal for high-dimensional spectral data

## Quick Usage

### 1. Simple Example
```julia
using SpectralPredict

# Your data
X, y, wavelengths = load_spectral_dataset(...)

# Run with NeuralBoosted
results = run_search(
    X, y, wavelengths,
    models=["NeuralBoosted"],
    preprocessing=["snv"],
    n_folds=5
)
```

### 2. Compare Multiple Models
```julia
results = run_search(
    X, y, wavelengths,
    models=["PLS", "Ridge", "NeuralBoosted"],
    preprocessing=["raw", "snv", "deriv"],
    n_folds=5
)

# Find best NeuralBoosted model
using DataFrames
df = DataFrame(results)
nb_models = filter(row -> row.Model == "NeuralBoosted", df)
best_nb = first(sort(nb_models, :Rank), 1)
println(best_nb)
```

### 3. Manual Configuration
```julia
# Build specific config
config = Dict(
    "n_estimators" => 100,
    "learning_rate" => 0.1,
    "hidden_layer_size" => 5,
    "activation" => "tanh"
)

model = build_model("NeuralBoosted", config, "regression")
fit_model!(model, X_train, y_train)
y_pred = predict_model(model, X_test)

# Get importances
importances = get_feature_importances(model, "NeuralBoosted", X_train, y_train)
```

## Hyperparameters Explained

### Search Grid (36 configurations)
| Parameter | Values | Meaning |
|-----------|--------|---------|
| `n_estimators` | 50, 100, 200 | Number of boosting stages (weak learners) |
| `learning_rate` | 0.05, 0.1, 0.2 | Shrinkage factor (lower = slower learning) |
| `hidden_layer_size` | 3, 5 | Neurons in weak learner (keep small!) |
| `activation` | "tanh", "relu" | Activation function for weak learners |

### Fixed Parameters (auto-configured)
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `alpha` | 0.0001 | L2 regularization strength |
| `max_iter` | 100 | Training iterations per weak learner |
| `early_stopping` | true | Stop if validation doesn't improve |
| `verbose` | 0 | Logging level (0=silent, 1=progress) |

## When to Use NeuralBoosted

### Good For:
- ✓ High-dimensional spectral data (many wavelengths)
- ✓ Non-linear relationships between spectra and properties
- ✓ Capturing complex patterns PLS might miss
- ✓ Robust predictions with feature importance

### Consider Alternatives When:
- Small datasets (< 50 samples) → Use PLS or Ridge
- Need fast training → Use PLS or Ridge
- Linear relationships → Use PLS
- Interpretability is critical → Use PLS (loadings) or Ridge (coefficients)

## Performance Tips

### 1. Start Small
```julia
# Quick test with small grid
results = run_search(
    X, y, wavelengths,
    models=["NeuralBoosted"],
    preprocessing=["snv"],  # Just one preprocessing
    n_folds=3  # Fewer folds for testing
)
```

### 2. Monitor Early Stopping
After fitting, check how many estimators were actually used:
```julia
println("Estimators trained: $(model.model.n_estimators_)")
println("vs max: $(model.n_estimators)")
```
If much lower, consider reducing n_estimators in grid.

### 3. Debug with Verbose
```julia
config = Dict(
    "n_estimators" => 50,
    "learning_rate" => 0.1,
    "hidden_layer_size" => 3,
    "activation" => "tanh",
    "verbose" => 1  # Enable progress logging
)
```

## Interpreting Results

### Feature Importance
```julia
importances = get_feature_importances(model, "NeuralBoosted", X_train, y_train)

# Find top 10 wavelengths
top_10_idx = sortperm(importances, rev=true)[1:10]
top_10_wavelengths = wavelengths[top_10_idx]

println("Most important wavelengths:")
for (i, wl) in zip(top_10_idx, top_10_wavelengths)
    println("  $wl nm: importance = $(round(importances[i], digits=4))")
end
```

### Model Performance
```julia
# After run_search()
using DataFrames
df = DataFrame(results)

# Best NeuralBoosted for each preprocessing
for prep in unique(df.Preprocessing)
    subset = filter(row -> row.Preprocessing == prep && row.Model == "NeuralBoosted", df)
    if !isempty(subset)
        best = first(sort(subset, :RMSE), 1)
        println("$prep: RMSE=$(best.RMSE[1]), R²=$(best.R2[1])")
    end
end
```

## Comparison with Other Models

| Model | Speed | Interpretability | Accuracy | Handles Non-linearity |
|-------|-------|------------------|----------|----------------------|
| PLS | ⚡⚡⚡ Fast | ⭐⭐⭐ High | ⭐⭐ Good | ❌ Limited |
| Ridge | ⚡⚡⚡ Fast | ⭐⭐ Medium | ⭐⭐ Good | ❌ Limited |
| RandomForest | ⚡⚡ Medium | ⭐ Low | ⭐⭐⭐ Excellent | ✓ Yes |
| MLP | ⚡ Slow | ⭐ Low | ⭐⭐⭐ Excellent | ✓ Yes |
| **NeuralBoosted** | ⚡ Slow | ⭐⭐ Medium | ⭐⭐⭐ Excellent | ✓ Yes |

## Common Issues & Solutions

### Issue: Training is very slow
**Solution:**
- Reduce n_estimators (try 50)
- Use smaller hidden_layer_size (3)
- Enable early_stopping (default)
- Reduce n_folds in cross-validation

### Issue: Overfitting
**Solution:**
- Lower learning_rate (try 0.05)
- Reduce n_estimators
- Increase alpha (L2 regularization)
- Ensure early_stopping=true

### Issue: Underfitting
**Solution:**
- Increase n_estimators (try 200)
- Higher learning_rate (try 0.2)
- Larger hidden_layer_size (try 5)
- Try different activation ("relu" vs "tanh")

### Issue: NaN predictions
**Solution:**
- Check for NaN/Inf in input data
- Try different activation (tanh is more stable than relu)
- Increase alpha for more regularization
- Reduce learning_rate

## Advanced Usage

### Custom Search Grid
```julia
# Modify models.jl to add custom values
# Line 128-148 in models.jl:
elseif model_name == "NeuralBoosted"
    n_estimators_list = [30, 75, 150]  # Custom values
    learning_rate_list = [0.03, 0.08, 0.15]  # Custom values
    hidden_layer_size_list = [4, 7]  # Custom values
    activation_list = ["tanh", "sigmoid"]  # Try sigmoid
    # ... rest of code
```

### Combine with Variable Selection
```julia
# Use SPA to select wavelengths, then NeuralBoosted
results = run_search(
    X, y, wavelengths,
    models=["NeuralBoosted"],
    preprocessing=["snv"],
    enable_variable_subsets=true,
    variable_methods=["SPA"],
    variable_counts=[10, 20, 50],
    n_folds=5
)

# Best sparse model
df = DataFrame(results)
sparse = filter(row -> row.n_vars < 50, df)
best_sparse = first(sort(sparse, :Rank), 1)
println("Best sparse NeuralBoosted: $(best_sparse.n_vars) features, RMSE=$(best_sparse.RMSE)")
```

## Example Workflow

```julia
using SpectralPredict
using DataFrames

# 1. Load data
X, y, wavelengths = load_spectral_dataset(
    "data/spectra",
    "data/reference.csv",
    "sample_id",
    "protein_pct"
)

# 2. Quick exploration (fast models)
quick_results = run_search(
    X, y, wavelengths,
    models=["PLS", "Ridge"],
    preprocessing=["raw", "snv", "deriv"],
    n_folds=5
)

# 3. If quick models show promise, try NeuralBoosted
full_results = run_search(
    X, y, wavelengths,
    models=["PLS", "Ridge", "NeuralBoosted"],
    preprocessing=["snv", "deriv"],  # Best from step 2
    enable_variable_subsets=true,
    n_folds=5
)

# 4. Analyze results
df = DataFrame(full_results)
top_10 = first(sort(df, :Rank), 10)
println(top_10[:, [:Rank, :Model, :Preprocessing, :RMSE, :R2]])

# 5. Extract best model details
best = first(top_10, 1)
println("\nBest Model: $(best.Model[1])")
println("Config: $(best.Config[1])")
println("RMSE: $(best.RMSE[1])")
println("R²: $(best.R2[1])")

# 6. If NeuralBoosted won, get feature importances
if best.Model[1] == "NeuralBoosted"
    # Retrain on full data to get importances
    config = best.Config[1]
    model = build_model("NeuralBoosted", config, "regression")
    fit_model!(model, X, y)
    importances = get_feature_importances(model, "NeuralBoosted", X, y)

    top_wavelengths = wavelengths[sortperm(importances, rev=true)[1:10]]
    println("\nTop 10 important wavelengths: $top_wavelengths")
end
```

## References

- Original Paper: Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine."
- JMP Documentation: Neural Boosted methodology
- Implementation: `src/neural_boosted.jl`
- Integration: `src/models.jl`

## Getting Help

1. Check `NEURAL_BOOSTED_INTEGRATION_SUMMARY.md` for technical details
2. Run `test_neural_boosted_integration.jl` for working examples
3. See `src/neural_boosted.jl` for implementation details
4. Review `START_HERE.md` for general SpectralPredict usage

---

**Last Updated:** November 2025
**Status:** Production Ready
