"""
Example usage of the models module

This script demonstrates how to use the different ML models for spectral prediction.
"""

using Random
using Statistics
using Printf

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/models.jl")

# Set random seed
Random.seed!(42)

println("="^70)
println("SpectralPredict Models Module - Example Usage")
println("="^70)
println()


# ============================================================================
# Generate Synthetic Spectral Data
# ============================================================================

println("1. Generating synthetic spectral data...")
println()

# Simulate spectral data: 100 samples, 200 wavelengths
n_samples = 100
n_wavelengths = 200

# Create synthetic spectra (random with some structure)
X = randn(n_samples, n_wavelengths) .+
    sin.(collect(1:n_wavelengths)' ./ 10) .+
    0.5 * randn(n_samples, n_wavelengths)

# Create target variable (e.g., protein content)
# True relationship: depends on certain spectral regions
true_weights = zeros(n_wavelengths)
true_weights[50:60] .= 2.0      # Important region 1
true_weights[120:130] .= -1.5   # Important region 2
true_weights[180:190] .= 1.0    # Important region 3

y = X * true_weights + 0.5 * randn(n_samples)

# Split into train/test
train_idx = 1:80
test_idx = 81:100

X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]

println("  Training samples: $(length(train_idx))")
println("  Testing samples: $(length(test_idx))")
println("  Number of wavelengths: $n_wavelengths")
println()


# ============================================================================
# Example 1: PLS Regression
# ============================================================================

println("2. Training PLS Regression Model")
println("-"^70)

# Get available configurations
pls_configs = get_model_configs("PLS")
println("  Available PLS configurations: $(length(pls_configs))")
println("  n_components options: ", [c["n_components"] for c in pls_configs])
println()

# Select a configuration
config = Dict("n_components" => 10)
println("  Selected configuration: n_components = $(config["n_components"])")

# Build model
pls_model = build_model("PLS", config, "regression")
println("  Model type: $(typeof(pls_model))")

# Train model
println("  Fitting model...")
fit_model!(pls_model, X_train, y_train)
println("  Model fitted successfully!")

# Make predictions
y_pred = predict_model(pls_model, X_test)

# Evaluate
mse = mean((y_test .- y_pred).^2)
rmse = sqrt(mse)
r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

println()
println("  Performance on test set:")
@printf("    RMSE: %.4f\n", rmse)
@printf("    R²: %.4f\n", r2)

# Get feature importances (VIP scores)
importances = get_feature_importances(pls_model, "PLS", X_train, y_train)
top_10_idx = sortperm(importances, rev=true)[1:10]

println()
println("  Top 10 important wavelengths (VIP scores):")
for (i, idx) in enumerate(top_10_idx)
    @printf("    %2d. Wavelength %3d: VIP = %.3f\n", i, idx, importances[idx])
end
println()


# ============================================================================
# Example 2: Ridge Regression
# ============================================================================

println("3. Training Ridge Regression Model")
println("-"^70)

# Get configurations
ridge_configs = get_model_configs("Ridge")
println("  Available Ridge configurations: $(length(ridge_configs))")
println("  Alpha (regularization) options: ", [c["alpha"] for c in ridge_configs])
println()

# Select configuration
config = Dict("alpha" => 1.0)
println("  Selected configuration: alpha = $(config["alpha"])")

# Build and train
ridge_model = build_model("Ridge", config, "regression")
fit_model!(ridge_model, X_train, y_train)

# Predict and evaluate
y_pred = predict_model(ridge_model, X_test)
mse = mean((y_test .- y_pred).^2)
rmse = sqrt(mse)
r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

println()
println("  Performance on test set:")
@printf("    RMSE: %.4f\n", rmse)
@printf("    R²: %.4f\n", r2)

# Get feature importances (coefficient magnitudes)
importances = get_feature_importances(ridge_model, "Ridge", X_train, y_train)
top_10_idx = sortperm(importances, rev=true)[1:10]

println()
println("  Top 10 important wavelengths (coefficient magnitudes):")
for (i, idx) in enumerate(top_10_idx)
    @printf("    %2d. Wavelength %3d: |coef| = %.3f\n", i, idx, importances[idx])
end
println()


# ============================================================================
# Example 3: Random Forest
# ============================================================================

println("4. Training Random Forest Model")
println("-"^70)

# Get configurations
rf_configs = get_model_configs("RandomForest")
println("  Available RandomForest configurations: $(length(rf_configs))")
println()

# Select configuration
config = Dict("n_trees" => 100, "max_features" => "sqrt")
println("  Selected configuration:")
println("    n_trees = $(config["n_trees"])")
println("    max_features = $(config["max_features"])")

# Build and train
rf_model = build_model("RandomForest", config, "regression")
println()
println("  Fitting model (this may take a moment)...")
fit_model!(rf_model, X_train, y_train)
println("  Model fitted successfully!")

# Predict and evaluate
y_pred = predict_model(rf_model, X_test)
mse = mean((y_test .- y_pred).^2)
rmse = sqrt(mse)
r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

println()
println("  Performance on test set:")
@printf("    RMSE: %.4f\n", rmse)
@printf("    R²: %.4f\n", r2)

# Get feature importances
importances = get_feature_importances(rf_model, "RandomForest", X_train, y_train)
top_10_idx = sortperm(importances, rev=true)[1:10]

println()
println("  Top 10 important wavelengths (tree-based importance):")
for (i, idx) in enumerate(top_10_idx)
    @printf("    %2d. Wavelength %3d: importance = %.4f\n", i, idx, importances[idx])
end
println()


# ============================================================================
# Example 4: MLP (Neural Network)
# ============================================================================

println("5. Training Multi-Layer Perceptron (MLP) Model")
println("-"^70)

# Get configurations
mlp_configs = get_model_configs("MLP")
println("  Available MLP configurations: $(length(mlp_configs))")
println()

# Select configuration
config = Dict("hidden_layers" => (50,), "learning_rate" => 0.01)
println("  Selected configuration:")
println("    hidden_layers = $(config["hidden_layers"])")
println("    learning_rate = $(config["learning_rate"])")

# Build and train
mlp_model = build_model("MLP", config, "regression")
println()
println("  Fitting model (training neural network)...")
fit_model!(mlp_model, X_train, y_train)
println("  Model fitted successfully!")

# Predict and evaluate
y_pred = predict_model(mlp_model, X_test)
mse = mean((y_test .- y_pred).^2)
rmse = sqrt(mse)
r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

println()
println("  Performance on test set:")
@printf("    RMSE: %.4f\n", rmse)
@printf("    R²: %.4f\n", r2)

# Get feature importances
importances = get_feature_importances(mlp_model, "MLP", X_train, y_train)
top_10_idx = sortperm(importances, rev=true)[1:10]

println()
println("  Top 10 important wavelengths (weight magnitudes):")
for (i, idx) in enumerate(top_10_idx)
    @printf("    %2d. Wavelength %3d: weight = %.4f\n", i, idx, importances[idx])
end
println()


# ============================================================================
# Example 5: Comparing Multiple Models
# ============================================================================

println("6. Comparing All Models")
println("-"^70)

model_specs = [
    ("PLS", Dict("n_components" => 10)),
    ("Ridge", Dict("alpha" => 1.0)),
    ("Lasso", Dict("alpha" => 0.1)),
    ("ElasticNet", Dict("alpha" => 1.0, "l1_ratio" => 0.5)),
    ("RandomForest", Dict("n_trees" => 50, "max_features" => "sqrt")),
    ("MLP", Dict("hidden_layers" => (50,), "learning_rate" => 0.01))
]

println()
println("  Model                    RMSE      R²     Training Time")
println("  " * "-"^60)

results = []

for (model_name, config) in model_specs
    # Build model
    model = build_model(model_name, config, "regression")

    # Time the training
    t_start = time()
    fit_model!(model, X_train, y_train)
    t_elapsed = time() - t_start

    # Predict and evaluate
    y_pred = predict_model(model, X_test)
    rmse = sqrt(mean((y_test .- y_pred).^2))
    r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

    @printf("  %-22s %7.4f %7.4f    %6.3f s\n", model_name, rmse, r2, t_elapsed)

    push!(results, (model_name, rmse, r2, t_elapsed))
end

println()

# Find best model
best_idx = argmin([r[2] for r in results])  # Lowest RMSE
best_name, best_rmse, best_r2, _ = results[best_idx]

println("  Best performing model: $best_name")
@printf("    RMSE: %.4f\n", best_rmse)
@printf("    R²: %.4f\n", best_r2)
println()


# ============================================================================
# Example 6: Hyperparameter Grid Search (Simplified)
# ============================================================================

println("7. Simple Hyperparameter Search for PLS")
println("-"^70)

# Try all PLS configurations
pls_configs = get_model_configs("PLS")
println("  Testing $(length(pls_configs)) different configurations...")
println()

best_r2 = -Inf
best_config = nothing
best_model = nothing

for config in pls_configs
    model = build_model("PLS", config, "regression")
    fit_model!(model, X_train, y_train)

    y_pred = predict_model(model, X_test)
    r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

    @printf("    n_components = %2d: R² = %.4f\n",
            config["n_components"], r2)

    if r2 > best_r2
        best_r2 = r2
        best_config = config
        best_model = model
    end
end

println()
println("  Best configuration:")
println("    n_components = $(best_config["n_components"])")
@printf("    R² = %.4f\n", best_r2)
println()


# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("Example completed successfully!")
println()
println("Key takeaways:")
println("  - Multiple model types available: PLS, Ridge, Lasso, ElasticNet, RF, MLP")
println("  - Each model has configurable hyperparameters")
println("  - Unified interface: build_model, fit_model!, predict_model")
println("  - Feature importance extraction available for all models")
println("  - Easy to compare models and search hyperparameters")
println("="^70)
