"""
Test suite for models.jl

This file demonstrates usage of all model types and validates their implementation.
Run with: julia --project=. test/test_models.jl
"""

using Test
using Random
using Statistics
using LinearAlgebra

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SpectralPredict

# Set random seed for reproducibility
Random.seed!(42)


# ============================================================================
# Test Data Generation
# ============================================================================

"""Generate synthetic regression data for testing"""
function generate_test_data(n_samples=100, n_features=20)
    X = randn(n_samples, n_features)
    # True model: y = sum of first 5 features + noise
    true_coef = zeros(n_features)
    true_coef[1:5] .= [2.0, -1.5, 1.0, -0.5, 0.8]
    y = X * true_coef + 0.1 * randn(n_samples)
    return X, y, true_coef
end


# ============================================================================
# Test Model Configuration
# ============================================================================

@testset "Model Configuration Generation" begin
    @testset "PLS Configurations" begin
        configs = get_model_configs("PLS")
        @test length(configs) == 8
        @test all(haskey(c, "n_components") for c in configs)
        @test configs[1]["n_components"] == 1
        @test configs[end]["n_components"] == 20
    end

    @testset "Ridge Configurations" begin
        configs = get_model_configs("Ridge")
        @test length(configs) == 6
        @test all(haskey(c, "alpha") for c in configs)
        @test configs[1]["alpha"] == 0.001
        @test configs[end]["alpha"] == 100.0
    end

    @testset "Lasso Configurations" begin
        configs = get_model_configs("Lasso")
        @test length(configs) == 6
        @test all(haskey(c, "alpha") for c in configs)
    end

    @testset "ElasticNet Configurations" begin
        configs = get_model_configs("ElasticNet")
        @test length(configs) == 12  # 4 alphas × 3 l1_ratios
        @test all(haskey(c, "alpha") && haskey(c, "l1_ratio") for c in configs)
    end

    @testset "RandomForest Configurations" begin
        configs = get_model_configs("RandomForest")
        @test length(configs) == 6  # 3 n_trees × 2 max_features
        @test all(haskey(c, "n_trees") && haskey(c, "max_features") for c in configs)
    end

    @testset "MLP Configurations" begin
        configs = get_model_configs("MLP")
        @test length(configs) == 6  # 3 architectures × 2 learning rates
        @test all(haskey(c, "hidden_layers") && haskey(c, "learning_rate") for c in configs)
    end

    @testset "Invalid Model Name" begin
        @test_throws ArgumentError get_model_configs("InvalidModel")
    end
end


# ============================================================================
# Test Model Building
# ============================================================================

@testset "Model Building" begin
    @testset "Build PLS Model" begin
        config = Dict("n_components" => 5)
        model = build_model("PLS", config, "regression")
        @test isa(model, PLSModel)
        @test model.n_components == 5
    end

    @testset "Build Ridge Model" begin
        config = Dict("alpha" => 1.0)
        model = build_model("Ridge", config, "regression")
        @test isa(model, RidgeModel)
        @test model.alpha == 1.0
    end

    @testset "Build Lasso Model" begin
        config = Dict("alpha" => 0.1)
        model = build_model("Lasso", config, "regression")
        @test isa(model, LassoModel)
        @test model.alpha == 0.1
    end

    @testset "Build ElasticNet Model" begin
        config = Dict("alpha" => 1.0, "l1_ratio" => 0.5)
        model = build_model("ElasticNet", config, "regression")
        @test isa(model, ElasticNetModel)
        @test model.alpha == 1.0
        @test model.l1_ratio == 0.5
    end

    @testset "Build RandomForest Model" begin
        config = Dict("n_trees" => 100, "max_features" => "sqrt")
        model = build_model("RandomForest", config, "regression")
        @test isa(model, RandomForestModel)
        @test model.n_trees == 100
        @test model.max_features == "sqrt"
    end

    @testset "Build MLP Model" begin
        config = Dict("hidden_layers" => (50, 50), "learning_rate" => 0.001)
        model = build_model("MLP", config, "regression")
        @test isa(model, MLPModel)
        @test model.hidden_layers == (50, 50)
        @test model.learning_rate == 0.001
    end

    @testset "Classification Not Supported" begin
        config = Dict("n_components" => 5)
        @test_throws ArgumentError build_model("PLS", config, "classification")
    end
end


# ============================================================================
# Test Model Fitting and Prediction
# ============================================================================

@testset "PLS Model Fit and Predict" begin
    X, y, true_coef = generate_test_data()
    X_train, X_test = X[1:80, :], X[81:end, :]
    y_train, y_test = y[1:80], y[81:end]

    config = Dict("n_components" => 5)
    model = build_model("PLS", config, "regression")

    # Test fitting
    fit_model!(model, X_train, y_train)
    @test !isnothing(model.model)
    @test !isnothing(model.mean_X)
    @test !isnothing(model.mean_y)

    # Test prediction
    y_pred = predict_model(model, X_test)
    @test length(y_pred) == length(y_test)
    @test all(isfinite.(y_pred))

    # Test reasonable performance (R² > 0.5 on synthetic data)
    ss_res = sum((y_test .- y_pred).^2)
    ss_tot = sum((y_test .- mean(y_test)).^2)
    r2 = 1 - ss_res / ss_tot
    @test r2 > 0.5
end


@testset "Ridge Model Fit and Predict" begin
    X, y, true_coef = generate_test_data()
    X_train, X_test = X[1:80, :], X[81:end, :]
    y_train, y_test = y[1:80], y[81:end]

    config = Dict("alpha" => 1.0)
    model = build_model("Ridge", config, "regression")

    fit_model!(model, X_train, y_train)
    @test !isnothing(model.model)

    y_pred = predict_model(model, X_test)
    @test length(y_pred) == length(y_test)
    @test all(isfinite.(y_pred))

    # Check prediction accuracy
    ss_res = sum((y_test .- y_pred).^2)
    ss_tot = sum((y_test .- mean(y_test)).^2)
    r2 = 1 - ss_res / ss_tot
    @test r2 > 0.5
end


@testset "Lasso Model Fit and Predict" begin
    X, y, true_coef = generate_test_data()
    X_train, X_test = X[1:80, :], X[81:end, :]
    y_train, y_test = y[1:80], y[81:end]

    config = Dict("alpha" => 0.1)
    model = build_model("Lasso", config, "regression")

    fit_model!(model, X_train, y_train)
    y_pred = predict_model(model, X_test)
    @test all(isfinite.(y_pred))
end


@testset "ElasticNet Model Fit and Predict" begin
    X, y, true_coef = generate_test_data()
    X_train, X_test = X[1:80, :], X[81:end, :]
    y_train, y_test = y[1:80], y[81:end]

    config = Dict("alpha" => 1.0, "l1_ratio" => 0.5)
    model = build_model("ElasticNet", config, "regression")

    fit_model!(model, X_train, y_train)
    y_pred = predict_model(model, X_test)
    @test all(isfinite.(y_pred))
end


@testset "RandomForest Model Fit and Predict" begin
    X, y, true_coef = generate_test_data()
    X_train, X_test = X[1:80, :], X[81:end, :]
    y_train, y_test = y[1:80], y[81:end]

    config = Dict("n_trees" => 50, "max_features" => "sqrt")
    model = build_model("RandomForest", config, "regression")

    fit_model!(model, X_train, y_train)
    @test !isnothing(model.forest)

    y_pred = predict_model(model, X_test)
    @test length(y_pred) == length(y_test)
    @test all(isfinite.(y_pred))

    # Random forest should perform well on this data
    ss_res = sum((y_test .- y_pred).^2)
    ss_tot = sum((y_test .- mean(y_test)).^2)
    r2 = 1 - ss_res / ss_tot
    @test r2 > 0.5
end


@testset "MLP Model Fit and Predict" begin
    X, y, true_coef = generate_test_data()
    X_train, X_test = X[1:80, :], X[81:end, :]
    y_train, y_test = y[1:80], y[81:end]

    config = Dict("hidden_layers" => (50,), "learning_rate" => 0.01)
    model = build_model("MLP", config, "regression")

    fit_model!(model, X_train, y_train)
    @test !isnothing(model.model)
    @test !isnothing(model.mean_X)
    @test !isnothing(model.std_X)

    y_pred = predict_model(model, X_test)
    @test length(y_pred) == length(y_test)
    @test all(isfinite.(y_pred))
end


# ============================================================================
# Test Feature Importances
# ============================================================================

@testset "Feature Importances" begin
    X, y, true_coef = generate_test_data()
    X_train = X[1:80, :]
    y_train = y[1:80]

    @testset "PLS VIP Scores" begin
        config = Dict("n_components" => 5)
        model = build_model("PLS", config, "regression")
        fit_model!(model, X_train, y_train)

        importances = get_feature_importances(model, "PLS", X_train, y_train)
        @test length(importances) == size(X_train, 2)
        @test all(importances .>= 0)  # VIP scores are non-negative

        # Top features should include true predictors (features 1-5)
        top_5 = sortperm(importances, rev=true)[1:5]
        @test length(intersect(top_5, 1:5)) >= 3  # At least 3 of top 5 are true
    end

    @testset "Ridge Coefficient Importances" begin
        config = Dict("alpha" => 1.0)
        model = build_model("Ridge", config, "regression")
        fit_model!(model, X_train, y_train)

        importances = get_feature_importances(model, "Ridge", X_train, y_train)
        @test length(importances) == size(X_train, 2)
        @test all(importances .>= 0)  # Absolute values are non-negative

        # Features with true coefficients should have higher importance
        top_5 = sortperm(importances, rev=true)[1:5]
        @test length(intersect(top_5, 1:5)) >= 3
    end

    @testset "Lasso Coefficient Importances" begin
        config = Dict("alpha" => 0.1)
        model = build_model("Lasso", config, "regression")
        fit_model!(model, X_train, y_train)

        importances = get_feature_importances(model, "Lasso", X_train, y_train)
        @test length(importances) == size(X_train, 2)
        @test all(importances .>= 0)

        # Lasso should zero out some coefficients
        @test sum(importances .== 0) > 0
    end

    @testset "ElasticNet Coefficient Importances" begin
        config = Dict("alpha" => 1.0, "l1_ratio" => 0.5)
        model = build_model("ElasticNet", config, "regression")
        fit_model!(model, X_train, y_train)

        importances = get_feature_importances(model, "ElasticNet", X_train, y_train)
        @test length(importances) == size(X_train, 2)
        @test all(importances .>= 0)
    end

    @testset "RandomForest Importances" begin
        config = Dict("n_trees" => 50, "max_features" => "sqrt")
        model = build_model("RandomForest", config, "regression")
        fit_model!(model, X_train, y_train)

        importances = get_feature_importances(model, "RandomForest", X_train, y_train)
        @test length(importances) == size(X_train, 2)
        @test all(importances .>= 0)

        # Should be normalized
        @test sum(importances) ≈ 1.0 atol=1e-6
    end

    @testset "MLP Weight Importances" begin
        config = Dict("hidden_layers" => (50,), "learning_rate" => 0.01)
        model = build_model("MLP", config, "regression")
        fit_model!(model, X_train, y_train)

        importances = get_feature_importances(model, "MLP", X_train, y_train)
        @test length(importances) == size(X_train, 2)
        @test all(importances .>= 0)
    end

    @testset "Unfitted Model Error" begin
        config = Dict("n_components" => 5)
        model = build_model("PLS", config, "regression")
        @test_throws ArgumentError get_feature_importances(model, "PLS", X_train, y_train)
    end
end


# ============================================================================
# Test Edge Cases
# ============================================================================

@testset "Edge Cases" begin
    @testset "Single Sample Prediction" begin
        X, y, _ = generate_test_data()
        X_train = X[1:80, :]
        y_train = y[1:80]
        X_single = reshape(X[81, :], 1, :)

        config = Dict("n_components" => 5)
        model = build_model("PLS", config, "regression")
        fit_model!(model, X_train, y_train)

        y_pred = predict_model(model, X_single)
        @test length(y_pred) == 1
        @test isfinite(y_pred[1])
    end

    @testset "High Dimensional Data" begin
        # More features than samples
        X = randn(50, 100)
        y = randn(50)

        config = Dict("alpha" => 1.0)
        model = build_model("Ridge", config, "regression")
        fit_model!(model, X, y)

        y_pred = predict_model(model, X)
        @test all(isfinite.(y_pred))
    end

    @testset "Constant Features" begin
        X, y, _ = generate_test_data()
        X[:, 1] .= 1.0  # Make first feature constant

        config = Dict("n_components" => 3)
        model = build_model("PLS", config, "regression")
        fit_model!(model, X, y)

        y_pred = predict_model(model, X)
        @test all(isfinite.(y_pred))
    end

    @testset "Zero Variance Target" begin
        X = randn(100, 20)
        y = ones(100)  # Constant target

        config = Dict("n_components" => 3)
        model = build_model("PLS", config, "regression")
        fit_model!(model, X, y)

        y_pred = predict_model(model, X)
        @test all(y_pred .≈ 1.0)
    end
end


# ============================================================================
# Integration Test: Full Pipeline
# ============================================================================

@testset "Full Pipeline Integration" begin
    # Generate larger dataset
    X, y, true_coef = generate_test_data(200, 30)
    X_train, X_test = X[1:150, :], X[151:end, :]
    y_train, y_test = y[1:150], y[151:end]

    # Test multiple models
    model_names = ["PLS", "Ridge", "Lasso", "RandomForest"]
    results = Dict()

    for model_name in model_names
        # Get first configuration
        configs = get_model_configs(model_name)
        config = configs[1]

        # Build and train
        model = build_model(model_name, config, "regression")
        fit_model!(model, X_train, y_train)

        # Predict
        y_pred = predict_model(model, X_test)

        # Evaluate
        mse = mean((y_test .- y_pred).^2)
        r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

        # Get importances
        importances = get_feature_importances(model, model_name, X_train, y_train)

        results[model_name] = Dict(
            "mse" => mse,
            "r2" => r2,
            "importances" => importances
        )

        @test isfinite(mse)
        @test r2 > 0.3  # Reasonable performance
        @test length(importances) == size(X_train, 2)
    end

    # Compare models
    println("\nModel Performance Comparison:")
    for (name, res) in results
        println("  $name: R² = $(round(res["r2"], digits=3)), MSE = $(round(res["mse"], digits=3))")
    end
end


println("\nAll tests completed successfully!")
