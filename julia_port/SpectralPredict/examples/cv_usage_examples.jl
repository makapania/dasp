"""
    cv_usage_examples.jl

Comprehensive usage examples for the cross-validation framework.

This file demonstrates all major features of the CV module including:
- Basic k-fold cross-validation
- Skip preprocessing mode for derivative subsets
- Parallel execution
- Different model types and preprocessing methods
- Results interpretation
"""

using Random
using Statistics

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/cv.jl")
include("../src/models.jl")
include("../src/preprocessing.jl")


println("="^80)
println("Cross-Validation Framework Usage Examples")
println("="^80)


# ============================================================================
# Example 1: Basic 5-Fold Cross-Validation
# ============================================================================

println("\n" * "="^80)
println("Example 1: Basic 5-Fold Cross-Validation")
println("="^80)

# Create synthetic spectral data
Random.seed!(42)
n_samples = 100
n_features = 50
X = rand(n_samples, n_features) .* 1000 .+ 2000  # Simulate spectral data
y = rand(n_samples) .* 10 .+ 5  # Simulate target values (e.g., protein content)

println("\nData shape: $(size(X))")
println("Target range: $(minimum(y)) to $(maximum(y))")

# Build PLS model
model = PLSModel(10)
preprocess_config = Dict("name" => "snv")

# Run 5-fold CV
results = run_cross_validation(
    X, y, model, "PLS",
    preprocess_config, "regression",
    n_folds=5
)

println("\nCross-Validation Results:")
println("  RMSE: $(round(results["RMSE_mean"], digits=4)) ± $(round(results["RMSE_std"], digits=4))")
println("  R²:   $(round(results["R2_mean"], digits=4)) ± $(round(results["R2_std"], digits=4))")
println("  MAE:  $(round(results["MAE_mean"], digits=4)) ± $(round(results["MAE_std"], digits=4))")

println("\nIndividual Fold Results:")
for (i, fold_metrics) in enumerate(results["cv_scores"])
    println("  Fold $i: RMSE = $(round(fold_metrics["RMSE"], digits=4)), " *
            "R² = $(round(fold_metrics["R2"], digits=4))")
end


# ============================================================================
# Example 2: Skip Preprocessing Mode (Critical for Derivative Subsets)
# ============================================================================

println("\n" * "="^80)
println("Example 2: Skip Preprocessing Mode")
println("="^80)

# Scenario: We have already preprocessed data (e.g., from a derivative subset)
# We want to run CV without re-applying preprocessing

# Preprocess the entire dataset once
X_preprocessed = apply_preprocessing(X, preprocess_config)
println("\nData preprocessed with SNV")

# Run CV with skip_preprocessing=true
model_skip = PLSModel(10)
results_skip = run_cross_validation(
    X_preprocessed, y, model_skip, "PLS",
    preprocess_config, "regression",
    n_folds=5,
    skip_preprocessing=true  # CRITICAL: Don't re-apply preprocessing
)

println("\nResults with skip_preprocessing=true:")
println("  RMSE: $(round(results_skip["RMSE_mean"], digits=4))")
println("  R²:   $(round(results_skip["R2_mean"], digits=4))")

# Compare with normal mode (would apply SNV twice - incorrect!)
model_normal = PLSModel(10)
results_normal = run_cross_validation(
    X_preprocessed, y, model_normal, "PLS",
    preprocess_config, "regression",
    n_folds=5,
    skip_preprocessing=false  # This would double-preprocess!
)

println("\nResults with skip_preprocessing=false (WRONG - double preprocessing):")
println("  RMSE: $(round(results_normal["RMSE_mean"], digits=4))")
println("  R²:   $(round(results_normal["R2_mean"], digits=4))")

println("\nNotice the difference! Always use skip_preprocessing=true for derivative subsets.")


# ============================================================================
# Example 3: Different Model Types
# ============================================================================

println("\n" * "="^80)
println("Example 3: Comparing Different Model Types")
println("="^80)

# Test multiple models with same preprocessing
preprocess_config = Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2)

model_configs = [
    ("PLS", PLSModel(5)),
    ("Ridge", RidgeModel(1.0)),
    ("Lasso", LassoModel(0.1)),
    ("RandomForest", RandomForestModel(50, "sqrt"))
]

println("\nComparing models with SNV + 1st derivative preprocessing:")
for (model_name, model) in model_configs
    results = run_cross_validation(
        X, y, model, model_name,
        preprocess_config, "regression",
        n_folds=5
    )

    println("\n$model_name:")
    println("  RMSE: $(round(results["RMSE_mean"], digits=4)) ± $(round(results["RMSE_std"], digits=4))")
    println("  R²:   $(round(results["R2_mean"], digits=4)) ± $(round(results["R2_std"], digits=4))")
end


# ============================================================================
# Example 4: Different Preprocessing Methods
# ============================================================================

println("\n" * "="^80)
println("Example 4: Comparing Preprocessing Methods")
println("="^80)

model = PLSModel(10)

preprocessing_configs = [
    ("Raw", Dict("name" => "raw")),
    ("SNV", Dict("name" => "snv")),
    ("1st Derivative", Dict("name" => "deriv", "deriv" => 1, "window" => 11, "polyorder" => 2)),
    ("SNV + 1st Deriv", Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2)),
    ("2nd Derivative", Dict("name" => "deriv", "deriv" => 2, "window" => 17, "polyorder" => 3))
]

println("\nComparing preprocessing methods (PLS model with 10 components):")
for (name, config) in preprocessing_configs
    results = run_cross_validation(
        X, y, PLSModel(10), "PLS",
        config, "regression",
        n_folds=5
    )

    println("\n$name:")
    println("  RMSE: $(round(results["RMSE_mean"], digits=4)) ± $(round(results["RMSE_std"], digits=4))")
    println("  R²:   $(round(results["R2_mean"], digits=4)) ± $(round(results["R2_std"], digits=4))")
end


# ============================================================================
# Example 5: Hyperparameter Tuning with CV
# ============================================================================

println("\n" * "="^80)
println("Example 5: Hyperparameter Tuning")
println("="^80)

# Test different numbers of PLS components
preprocess_config = Dict("name" => "snv")

println("\nTuning PLS components:")
best_score = Inf
best_n_components = 0

for n_components in [1, 3, 5, 10, 15, 20]
    model = PLSModel(n_components)
    results = run_cross_validation(
        X, y, model, "PLS",
        preprocess_config, "regression",
        n_folds=5
    )

    rmse = results["RMSE_mean"]
    println("  n_components=$n_components: RMSE = $(round(rmse, digits=4))")

    if rmse < best_score
        best_score = rmse
        best_n_components = n_components
    end
end

println("\nBest configuration: n_components=$best_n_components (RMSE = $(round(best_score, digits=4)))")


# ============================================================================
# Example 6: Classification Task
# ============================================================================

println("\n" * "="^80)
println("Example 6: Classification Task")
println("="^80)

# Create binary classification data
Random.seed!(42)
y_class = rand([0.0, 1.0], n_samples)

println("\nClass distribution:")
println("  Class 0: $(sum(y_class .== 0.0)) samples")
println("  Class 1: $(sum(y_class .== 1.0)) samples")

# Run classification CV (note: using Ridge for simplicity, would normally use logistic regression)
model = RidgeModel(1.0)
preprocess_config = Dict("name" => "raw")

results = run_cross_validation(
    X, y_class, model, "Ridge",
    preprocess_config, "classification",
    n_folds=5
)

println("\nClassification Results:")
println("  Accuracy: $(round(results["Accuracy_mean"], digits=4)) ± $(round(results["Accuracy_std"], digits=4))")
println("  ROC AUC:  $(round(results["ROC_AUC_mean"], digits=4)) ± $(round(results["ROC_AUC_std"], digits=4))")
println("  Precision: $(round(results["Precision_mean"], digits=4)) ± $(round(results["Precision_std"], digits=4))")
println("  Recall:   $(round(results["Recall_mean"], digits=4)) ± $(round(results["Recall_std"], digits=4))")


# ============================================================================
# Example 7: Parallel Cross-Validation
# ============================================================================

println("\n" * "="^80)
println("Example 7: Parallel Cross-Validation")
println("="^80)

println("\nNote: To use parallel CV, start Julia with multiple threads:")
println("  julia -t 4 cv_usage_examples.jl")
println("\nCurrent number of threads: $(Threads.nthreads())")

if Threads.nthreads() > 1
    println("\nRunning parallel CV...")

    # Time parallel version
    start_time = time()
    results_parallel = run_cross_validation_parallel(
        X, y, PLSModel(10), "PLS",
        Dict("name" => "snv"), "regression",
        n_folds=5
    )
    parallel_time = time() - start_time

    # Time sequential version
    start_time = time()
    results_sequential = run_cross_validation(
        X, y, PLSModel(10), "PLS",
        Dict("name" => "snv"), "regression",
        n_folds=5
    )
    sequential_time = time() - start_time

    println("\nParallel time: $(round(parallel_time, digits=3))s")
    println("Sequential time: $(round(sequential_time, digits=3))s")
    println("Speedup: $(round(sequential_time/parallel_time, digits=2))x")
else
    println("\nRunning with single thread (parallel CV would have same performance)")
end


# ============================================================================
# Example 8: Manual Fold Creation and Execution
# ============================================================================

println("\n" * "="^80)
println("Example 8: Manual Fold Creation (Advanced)")
println("="^80)

# Create custom CV folds
folds = create_cv_folds(n_samples, 5)

println("\nFold structure:")
for (i, (train_idx, test_idx)) in enumerate(folds)
    println("  Fold $i: $(length(train_idx)) train, $(length(test_idx)) test")
end

# Run single fold manually
train_idx, test_idx = folds[1]
model = PLSModel(10)
preprocess_config = Dict("name" => "snv")

fold_metrics = run_single_fold(
    X, y, train_idx, test_idx,
    model, "PLS", preprocess_config, "regression"
)

println("\nSingle Fold Results (Fold 1):")
println("  RMSE: $(round(fold_metrics["RMSE"], digits=4))")
println("  R²:   $(round(fold_metrics["R2"], digits=4))")
println("  MAE:  $(round(fold_metrics["MAE"], digits=4))")


# ============================================================================
# Example 9: Error Handling
# ============================================================================

println("\n" * "="^80)
println("Example 9: Error Handling")
println("="^80)

println("\nTesting error conditions:")

# Test 1: Too few folds
try
    run_cross_validation(X, y, PLSModel(5), "PLS", Dict("name" => "raw"), "regression", n_folds=1)
    println("  ✗ Should have thrown error for n_folds=1")
catch e
    println("  ✓ Correctly caught: n_folds must be at least 2")
end

# Test 2: Mismatched X and y
try
    X_wrong = X[1:50, :]
    run_cross_validation(X_wrong, y, PLSModel(5), "PLS", Dict("name" => "raw"), "regression")
    println("  ✗ Should have thrown error for mismatched dimensions")
catch e
    println("  ✓ Correctly caught: X and y dimension mismatch")
end

# Test 3: Too many folds
try
    run_cross_validation(X[1:10, :], y[1:10], PLSModel(5), "PLS", Dict("name" => "raw"), "regression", n_folds=11)
    println("  ✗ Should have thrown error for n_folds > n_samples")
catch e
    println("  ✓ Correctly caught: n_folds exceeds n_samples")
end


# ============================================================================
# Example 10: Real-World Workflow
# ============================================================================

println("\n" * "="^80)
println("Example 10: Complete Real-World Workflow")
println("="^80)

println("\nSimulating complete model selection workflow:")

# 1. Define search space
preprocessing_options = [
    Dict("name" => "raw"),
    Dict("name" => "snv"),
    Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2)
]

model_options = [
    ("PLS", [1, 5, 10, 15]),
    ("Ridge", [0.01, 0.1, 1.0, 10.0]),
]

# 2. Grid search with CV
println("\nPerforming grid search...")
best_rmse = Inf
best_config = nothing

for preprocess_config in preprocessing_options
    for (model_name, param_values) in model_options
        for param_value in param_values
            # Build model
            if model_name == "PLS"
                model = PLSModel(param_value)
                config_str = "n_components=$param_value"
            else  # Ridge
                model = RidgeModel(param_value)
                config_str = "alpha=$param_value"
            end

            # Run CV
            results = run_cross_validation(
                X, y, model, model_name,
                preprocess_config, "regression",
                n_folds=5
            )

            rmse = results["RMSE_mean"]

            # Track best
            if rmse < best_rmse
                best_rmse = rmse
                best_config = (model_name, config_str, preprocess_config["name"])
            end
        end
    end
end

println("\nBest configuration found:")
println("  Model: $(best_config[1])")
println("  Hyperparameters: $(best_config[2])")
println("  Preprocessing: $(best_config[3])")
println("  CV RMSE: $(round(best_rmse, digits=4))")


println("\n" * "="^80)
println("All Examples Completed Successfully!")
println("="^80)
