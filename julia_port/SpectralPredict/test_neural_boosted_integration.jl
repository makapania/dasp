"""
Test script for NeuralBoostedRegressor integration with models.jl

This script tests:
1. Model configuration generation
2. Model building
3. Model fitting
4. Model prediction
5. Feature importance extraction
"""

using Random
Random.seed!(42)

# Add the SpectralPredict module to the path
push!(LOAD_PATH, @__DIR__)
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Load the models module
include("src/models.jl")

println("="^80)
println("Testing NeuralBoostedRegressor Integration")
println("="^80)
println()

# Create synthetic test data
println("Creating synthetic test data...")
n_samples = 100
n_features = 50

X = randn(n_samples, n_features)
# True relationship: y depends on features 10 and 20
y = 2.0 .* X[:, 10] .+ 1.5 .* X[:, 20] .+ 0.5 .* X[:, 30] .+ randn(n_samples) .* 0.2
println("  Dataset: $n_samples samples × $n_features features")
println()

# Test 1: Get model configurations
println("Test 1: Getting model configurations...")
try
    configs = get_model_configs("NeuralBoosted")
    println("  ✓ Success: Generated $(length(configs)) configurations")
    println("  Example config: $(configs[1])")
catch e
    println("  ✗ Failed: $e")
    rethrow(e)
end
println()

# Test 2: Build model
println("Test 2: Building model...")
try
    config = Dict(
        "n_estimators" => 50,
        "learning_rate" => 0.1,
        "hidden_layer_size" => 3,
        "activation" => "tanh"
    )
    model = build_model("NeuralBoosted", config, "regression")
    println("  ✓ Success: Built model of type $(typeof(model))")
    println("  Model params: n_estimators=$(model.n_estimators), lr=$(model.learning_rate)")
catch e
    println("  ✗ Failed: $e")
    rethrow(e)
end
println()

# Test 3: Fit model
println("Test 3: Fitting model...")
try
    config = Dict(
        "n_estimators" => 20,  # Small for fast test
        "learning_rate" => 0.1,
        "hidden_layer_size" => 3,
        "activation" => "tanh",
        "verbose" => 0
    )
    model = build_model("NeuralBoosted", config, "regression")

    println("  Fitting with $(config["n_estimators"]) estimators...")
    fit_model!(model, X, y)

    println("  ✓ Success: Model fitted")
    println("  Actual estimators trained: $(model.model.n_estimators_)")
catch e
    println("  ✗ Failed: $e")
    rethrow(e)
end
println()

# Test 4: Make predictions
println("Test 4: Making predictions...")
try
    X_test = randn(10, n_features)
    predictions = predict_model(model, X_test)

    println("  ✓ Success: Generated $(length(predictions)) predictions")
    println("  Prediction range: [$(minimum(predictions)), $(maximum(predictions))]")
    println("  Prediction type: $(typeof(predictions))")
catch e
    println("  ✗ Failed: $e")
    rethrow(e)
end
println()

# Test 5: Feature importances
println("Test 5: Computing feature importances...")
try
    importances = get_feature_importances(model, "NeuralBoosted", X, y)

    println("  ✓ Success: Computed importances for $(length(importances)) features")
    println("  Sum of importances: $(sum(importances)) (should be ~1.0)")

    # Find top 5 features
    top_indices = sortperm(importances, rev=true)[1:5]
    println("  Top 5 features: $top_indices")
    println("  Their importances: $(importances[top_indices])")

    # Check if important features (10, 20, 30) are in top 10
    top_10 = sortperm(importances, rev=true)[1:10]
    if 10 in top_10 && 20 in top_10
        println("  ✓ Model correctly identified important features (10, 20)")
    else
        println("  ⚠ Warning: Important features not in top 10")
    end
catch e
    println("  ✗ Failed: $e")
    rethrow(e)
end
println()

# Test 6: Cross-validation compatibility
println("Test 6: Testing cross-validation compatibility...")
try
    # Split into train/test
    train_idx = 1:80
    test_idx = 81:100

    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    # Build and fit
    config = Dict(
        "n_estimators" => 30,
        "learning_rate" => 0.1,
        "hidden_layer_size" => 3,
        "activation" => "tanh",
        "verbose" => 0
    )
    cv_model = build_model("NeuralBoosted", config, "regression")
    fit_model!(cv_model, X_train, y_train)

    # Predict on test set
    y_pred = predict_model(cv_model, X_test)

    # Compute RMSE
    rmse = sqrt(sum((y_test .- y_pred).^2) / length(y_test))
    r2 = 1.0 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

    println("  ✓ Success: Cross-validation test completed")
    println("  Test RMSE: $(round(rmse, digits=4))")
    println("  Test R²: $(round(r2, digits=4))")
catch e
    println("  ✗ Failed: $e")
    rethrow(e)
end
println()

println("="^80)
println("All tests completed successfully!")
println("="^80)
println()
println("NeuralBoosted model is ready to use in run_search()")
println()
println("Example usage:")
println("""
results = run_search(
    X, y, wavelengths,
    models=["PLS", "Ridge", "NeuralBoosted"],
    preprocessing=["raw", "snv"],
    n_folds=5
)
""")
