# Quick Start: Using Neural Boosted After Fixes

## ✅ What Was Fixed

The Julia Neural Boosted implementation had critical bugs causing "too few samples" errors. These are now **RESOLVED**.

### Fixed Issues:
- ✅ Empty validation set crashes
- ✅ Small dataset handling
- ✅ Cross-validation compatibility
- ✅ Automatic early stopping adjustment

## 🚀 Testing the Fixes

### Option 1: Run Unit Tests

```bash
cd /home/user/dasp/julia_port/SpectralPredict
julia --project=. test/test_neural_boosted_fixes.jl
```

**Expected**: All 7 tests pass ✓

### Option 2: Test in GUI

```bash
cd /home/user/dasp/julia_port/SpectralPredict
julia --project=. gui.jl
```

Then open: http://localhost:8080

**Steps**:
1. Load your spectral data
2. Check "NeuralBoosted (Gradient Boosting)" model
3. Select preprocessing (SNV recommended)
4. Run analysis

**Expected**: No "too few samples" errors!

## 📊 When to Use Neural Boosted

### ✅ Good For:
- Complex spectral patterns
- Non-linear relationships
- High-dimensional data (100+ wavelengths)
- When you need interpretability (feature importances)

### ⚠️ Consider Alternatives If:
- Very small datasets (< 20 samples) → Use Ridge/Lasso
- Simple linear relationships → Use Ridge/Lasso
- Speed is critical → Use Ridge/Lasso

## ⚙️ Recommended Settings

### For Small Datasets (20-50 samples):
```julia
NeuralBoostedRegressor(
    n_estimators=50,           # Fewer estimators
    learning_rate=0.1,
    hidden_layer_size=3,       # Keep small
    early_stopping=true,       # Auto-adjusts if needed
    verbose=1
)
```

### For Medium Datasets (50-200 samples):
```julia
NeuralBoostedRegressor(
    n_estimators=100,          # Default
    learning_rate=0.1,
    hidden_layer_size=3,
    early_stopping=true,
    verbose=1
)
```

### For Large Datasets (200+ samples):
```julia
NeuralBoostedRegressor(
    n_estimators=200,          # More estimators
    learning_rate=0.05,        # Lower learning rate
    hidden_layer_size=5,       # Can increase
    early_stopping=true,
    verbose=1
)
```

## 🔍 Troubleshooting

### Error: "requires at least X samples"
**Cause**: Dataset too small for the network architecture
**Fix**: Reduce `hidden_layer_size` or use a different model

### Warning: "Disabling early stopping"
**Cause**: Dataset too small for validation split
**Effect**: Model continues training on full dataset (normal behavior)
**Action**: None needed - this is expected and safe

### Model trains but predictions are poor
**Check**:
1. Try different preprocessing (SNV, derivatives)
2. Adjust `learning_rate` (try 0.05, 0.1, 0.2)
3. Increase `n_estimators` (100 → 200)
4. Check for outliers in data

## 📈 Interpreting Results

### Feature Importances

```julia
# After fitting
importances = feature_importances(model)

# Find top wavelengths
top_10_indices = sortperm(importances, rev=true)[1:10]
top_10_wavelengths = wavelengths[top_10_indices]

println("Most important wavelengths: $top_10_wavelengths")
```

### Training Curves

```julia
using Plots

# Plot training progress
plot(model.train_score_, label="Training Loss",
     xlabel="Iteration", ylabel="Loss",
     title="Neural Boosted Training Progress")

if !isempty(model.validation_score_)
    plot!(model.validation_score_, label="Validation Loss")
end
```

## 📚 Advanced Usage

### Custom Activation Functions

```julia
# Try different activations
for activation in ["tanh", "relu", "identity"]
    model = NeuralBoostedRegressor(
        activation=activation,
        n_estimators=50,
        verbose=0
    )
    # ... fit and evaluate
end
```

### Disable Early Stopping (for very small datasets)

```julia
model = NeuralBoostedRegressor(
    n_estimators=50,
    early_stopping=false,  # Use all data for training
    verbose=1
)
```

### Huber Loss (for outlier robustness)

```julia
model = NeuralBoostedRegressor(
    n_estimators=100,
    loss="huber",          # Robust to outliers
    huber_delta=1.35,
    verbose=1
)
```

## 🎯 Performance Tips

1. **Start simple**: Default settings work well for most cases
2. **Preprocess**: SNV or derivatives often improve results
3. **Tune gradually**: Adjust one parameter at a time
4. **Use cross-validation**: Always evaluate with CV, not just training fit
5. **Check importances**: Feature importance can guide wavelength selection

## ✨ Example: Complete Workflow

```julia
using SpectralPredict
using Statistics

# 1. Load data
X, y, wavelengths, sample_ids = load_spectral_dataset(
    "path/to/spectra",
    "path/to/reference.csv",
    "sample_id",
    "protein_pct"
)

# 2. Apply preprocessing
X_snv = apply_snv(X)

# 3. Create and fit model
model = NeuralBoostedRegressor(
    n_estimators=100,
    learning_rate=0.1,
    hidden_layer_size=3,
    early_stopping=true,
    verbose=1
)

fit!(model, X_snv, y)

# 4. Make predictions
predictions = predict(model, X_snv)

# 5. Evaluate
rmse = sqrt(mean((y .- predictions).^2))
r2 = 1 - sum((y .- predictions).^2) / sum((y .- mean(y)).^2)

println("RMSE: $rmse")
println("R²: $r2")

# 6. Get feature importances
importances = feature_importances(model)
top_5 = sortperm(importances, rev=true)[1:5]
println("Top 5 wavelengths: $(wavelengths[top_5])")
```

## 📞 Support

If you still encounter issues:
1. Check error messages (now more informative)
2. Try with `verbose=1` to see training progress
3. Review `NEURAL_BOOSTED_FIXES.md` for technical details
4. Compare with Python implementation if needed

---

**Status**: Ready to use! 🎉
**Last Updated**: 2025-11-06
