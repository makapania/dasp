"""
Test suite for neural_boosted.jl

Comprehensive tests for Neural Boosted Regressor:
- Model construction and parameter validation
- Fitting and prediction
- Early stopping
- Feature importances
- Different loss functions
- Edge cases and numerical stability

Run with: julia --project=. test/test_neural_boosted.jl
"""

using Test
using Random
using Statistics
using LinearAlgebra

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Import neural boosted module
include(joinpath(@__DIR__, "..", "src", "neural_boosted.jl"))
using .NeuralBoosted


# ============================================================================
# Test Data Generators
# ============================================================================

"""
Generate synthetic regression data for boosting tests.
"""
function generate_boosting_data(;
    n_samples=100,
    n_features=20,
    n_informative=5,
    noise_level=0.2,
    random_state=42
)
    Random.seed!(random_state)

    X = randn(n_samples, n_features)

    # Create non-linear target with interactions
    true_coef = zeros(n_features)
    true_coef[1:n_informative] = randn(n_informative)

    # Linear component
    y = X[:, 1:n_informative] * true_coef[1:n_informative]

    # Add non-linearity
    y .+= 0.5 .* sin.(X[:, 1])
    y .+= 0.3 .* (X[:, 2].^2)

    # Add noise
    y .+= randn(n_samples) * noise_level

    return X, y, true_coef
end

"""
Generate data with outliers for robust loss testing.
"""
function generate_data_with_outliers(;
    n_samples=100,
    n_features=10,
    n_outliers=5,
    random_state=42
)
    Random.seed!(random_state)

    X = randn(n_samples, n_features)
    y = X[:, 1] + 2 * X[:, 2] + randn(n_samples) * 0.1

    # Add outliers
    outlier_indices = randperm(n_samples)[1:n_outliers]
    y[outlier_indices] .+= randn(n_outliers) .* 10.0

    return X, y, outlier_indices
end

"""
Generate simple linear data for basic tests.
"""
function generate_linear_data(;
    n_samples=80,
    n_features=10,
    noise_level=0.1,
    random_state=42
)
    Random.seed!(random_state)

    X = randn(n_samples, n_features)
    true_coef = randn(n_features)
    y = X * true_coef + randn(n_samples) * noise_level

    return X, y, true_coef
end


# ============================================================================
# Test Model Construction
# ============================================================================

@testset "Model Construction" begin

    @testset "Default Parameters" begin
        model = NeuralBoostedRegressor()

        @test model.n_estimators == 100
        @test model.learning_rate == 0.1
        @test model.hidden_layer_size == 3
        @test model.activation == "tanh"
        @test model.early_stopping == true
        @test model.loss == "mse"
        @test model.verbose == 0

        # Fitted attributes should be initialized
        @test isempty(model.estimators_)
        @test isempty(model.train_score_)
        @test model.n_estimators_ == 0
    end

    @testset "Custom Parameters" begin
        model = NeuralBoostedRegressor(
            n_estimators=50,
            learning_rate=0.2,
            hidden_layer_size=5,
            activation="relu",
            early_stopping=false,
            loss="huber",
            verbose=1
        )

        @test model.n_estimators == 50
        @test model.learning_rate == 0.2
        @test model.hidden_layer_size == 5
        @test model.activation == "relu"
        @test model.early_stopping == false
        @test model.loss == "huber"
        @test model.verbose == 1
    end

    @testset "Parameter Validation" begin
        # Invalid learning rate
        @test_throws ErrorException NeuralBoostedRegressor(learning_rate=0.0)
        @test_throws ErrorException NeuralBoostedRegressor(learning_rate=1.5)

        # Invalid hidden layer size
        @test_throws ErrorException NeuralBoostedRegressor(hidden_layer_size=0)

        # Invalid activation
        @test_throws ErrorException NeuralBoostedRegressor(activation="invalid")

        # Invalid loss
        @test_throws ErrorException NeuralBoostedRegressor(loss="invalid")

        # Invalid validation fraction
        @test_throws ErrorException NeuralBoostedRegressor(validation_fraction=0.0)
        @test_throws ErrorException NeuralBoostedRegressor(validation_fraction=1.0)
    end

    @testset "Warning for Large Hidden Layer" begin
        # Should warn but not error
        @test_logs (:warn, r"hidden_layer_size") NeuralBoostedRegressor(hidden_layer_size=15)
    end
end


# ============================================================================
# Test Fitting
# ============================================================================

@testset "Model Fitting" begin

    @testset "Basic Fit" begin
        X, y, _ = generate_linear_data(n_samples=50, n_features=5)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=false,
            verbose=0
        )

        # Fit model
        fit!(model, X, y)

        # Check fitted attributes
        @test length(model.estimators_) == 10
        @test length(model.train_score_) == 10
        @test model.n_estimators_ == 10

        # Training loss should decrease
        @test model.train_score_[end] <= model.train_score_[1]
    end

    @testset "Early Stopping" begin
        X, y, _ = generate_linear_data(n_samples=80, n_features=10)

        model = NeuralBoostedRegressor(
            n_estimators=100,
            learning_rate=0.1,
            early_stopping=true,
            validation_fraction=0.2,
            n_iter_no_change=5,
            verbose=0
        )

        fit!(model, X, y)

        # Should stop early (before 100 estimators)
        @test model.n_estimators_ < 100

        # Should have validation scores
        @test length(model.validation_score_) == model.n_estimators_
    end

    @testset "No Early Stopping" begin
        X, y, _ = generate_linear_data(n_samples=60, n_features=8)

        model = NeuralBoostedRegressor(
            n_estimators=20,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        # Should fit all estimators
        @test model.n_estimators_ == 20

        # Should not have validation scores
        @test isempty(model.validation_score_)
    end

    @testset "Different Activations" begin
        X, y, _ = generate_linear_data(n_samples=50, n_features=6)

        for activation in ["tanh", "relu", "sigmoid", "identity"]
            model = NeuralBoostedRegressor(
                n_estimators=10,
                activation=activation,
                early_stopping=false,
                verbose=0
            )

            @test_nowarn fit!(model, X, y)
            @test length(model.estimators_) == 10
        end
    end

    @testset "MSE Loss" begin
        X, y, _ = generate_linear_data(n_samples=60, n_features=8)

        model = NeuralBoostedRegressor(
            n_estimators=15,
            loss="mse",
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        @test length(model.estimators_) == 15
        @test all(model.train_score_ .>= 0)  # MSE is non-negative
    end

    @testset "Huber Loss" begin
        X, y, outlier_indices = generate_data_with_outliers(
            n_samples=80, n_features=8, n_outliers=5
        )

        model = NeuralBoostedRegressor(
            n_estimators=20,
            loss="huber",
            huber_delta=1.35,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        @test length(model.estimators_) == 20
        @test all(model.train_score_ .>= 0)  # Huber loss is non-negative
    end

    @testset "Small Dataset Warning" begin
        X_small = randn(15, 5)
        y_small = randn(15)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            early_stopping=true,
            verbose=0
        )

        # Should warn about small dataset with early stopping
        @test_logs (:warn, r"very small") fit!(model, X_small, y_small)
    end
end


# ============================================================================
# Test Prediction
# ============================================================================

@testset "Model Prediction" begin

    @testset "Basic Prediction" begin
        X, y, _ = generate_linear_data(n_samples=60, n_features=8)

        model = NeuralBoostedRegressor(
            n_estimators=15,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        # Predict on training data
        y_pred = predict(model, X)

        @test length(y_pred) == 60
        @test !any(isnan.(y_pred))
        @test !any(isinf.(y_pred))

        # Predictions should be reasonable
        @test cor(y, y_pred) > 0.5  # At least moderate correlation
    end

    @testset "Prediction on New Data" begin
        X_train, y_train, _ = generate_linear_data(n_samples=80, n_features=10)
        X_test, y_test, _ = generate_linear_data(n_samples=20, n_features=10)

        model = NeuralBoostedRegressor(
            n_estimators=20,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X_train, y_train)

        # Predict on test data
        y_pred = predict(model, X_test)

        @test length(y_pred) == 20
        @test !any(isnan.(y_pred))
        @test !any(isinf.(y_pred))
    end

    @testset "Prediction Before Fitting" begin
        X = randn(50, 10)

        model = NeuralBoostedRegressor()

        # Should error if predicting before fitting
        @test_throws ErrorException predict(model, X)
    end

    @testset "Single Sample Prediction" begin
        X, y, _ = generate_linear_data(n_samples=50, n_features=8)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        # Predict on single sample
        X_single = reshape(X[1, :], 1, :)
        y_pred = predict(model, X_single)

        @test length(y_pred) == 1
        @test !isnan(y_pred[1])
        @test !isinf(y_pred[1])
    end
end


# ============================================================================
# Test Feature Importances
# ============================================================================

@testset "Feature Importances" begin

    @testset "Basic Importances" begin
        X, y, true_coef = generate_boosting_data(
            n_samples=100, n_features=20, n_informative=5
        )

        model = NeuralBoostedRegressor(
            n_estimators=20,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        importances = feature_importances(model)

        # Check dimensions
        @test length(importances) == 20

        # All importances should be non-negative
        @test all(importances .>= 0)

        # Should sum to 1 (normalized)
        @test sum(importances) ≈ 1.0 atol=1e-10

        # Informative features should have higher importance
        # (First 5 features are informative)
        informative_importance = mean(importances[1:5])
        uninformative_importance = mean(importances[6:20])

        @test informative_importance > uninformative_importance
    end

    @testset "Importances Before Fitting" begin
        model = NeuralBoostedRegressor()

        # Should error if computing importances before fitting
        @test_throws ErrorException feature_importances(model)
    end

    @testset "Importances Vary by Architecture" begin
        X, y, _ = generate_linear_data(n_samples=80, n_features=15)

        # Small hidden layer
        model_small = NeuralBoostedRegressor(
            n_estimators=15,
            hidden_layer_size=2,
            early_stopping=false,
            verbose=0
        )
        fit!(model_small, X, y)
        imp_small = feature_importances(model_small)

        # Large hidden layer
        model_large = NeuralBoostedRegressor(
            n_estimators=15,
            hidden_layer_size=8,
            early_stopping=false,
            verbose=0
        )
        fit!(model_large, X, y)
        imp_large = feature_importances(model_large)

        # Both should be valid
        @test length(imp_small) == 15
        @test length(imp_large) == 15
        @test sum(imp_small) ≈ 1.0
        @test sum(imp_large) ≈ 1.0
    end
end


# ============================================================================
# Test Learning Rate and Boosting Behavior
# ============================================================================

@testset "Learning Rate Effects" begin

    @testset "Different Learning Rates" begin
        X, y, _ = generate_linear_data(n_samples=80, n_features=10)

        # Small learning rate
        model_slow = NeuralBoostedRegressor(
            n_estimators=30,
            learning_rate=0.05,
            early_stopping=false,
            verbose=0
        )
        fit!(model_slow, X, y)

        # Large learning rate
        model_fast = NeuralBoostedRegressor(
            n_estimators=30,
            learning_rate=0.3,
            early_stopping=false,
            verbose=0
        )
        fit!(model_fast, X, y)

        # Both should complete successfully
        @test length(model_slow.estimators_) == 30
        @test length(model_fast.estimators_) == 30

        # Fast learning should achieve lower training loss sooner
        @test model_fast.train_score_[10] < model_slow.train_score_[10]
    end

    @testset "Residual Reduction" begin
        X, y, _ = generate_linear_data(n_samples=60, n_features=8)

        model = NeuralBoostedRegressor(
            n_estimators=25,
            learning_rate=0.1,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        # Training loss should monotonically decrease (or stay flat)
        for i in 2:length(model.train_score_)
            @test model.train_score_[i] <= model.train_score_[i-1] * 1.1  # Allow small increases
        end
    end
end


# ============================================================================
# Test Edge Cases
# ============================================================================

@testset "Edge Cases" begin

    @testset "Very Small Dataset" begin
        X_small = randn(10, 5)
        y_small = randn(10)

        model = NeuralBoostedRegressor(
            n_estimators=5,
            early_stopping=false,
            verbose=0
        )

        @test_nowarn fit!(model, X_small, y_small)
        @test length(model.estimators_) == 5

        y_pred = predict(model, X_small)
        @test length(y_pred) == 10
    end

    @testset "Many Features, Few Samples" begin
        X_wide = randn(30, 100)
        y_wide = randn(30)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            hidden_layer_size=3,
            early_stopping=false,
            verbose=0
        )

        @test_nowarn fit!(model, X_wide, y_wide)

        y_pred = predict(model, X_wide)
        @test length(y_pred) == 30
    end

    @testset "Single Estimator" begin
        X, y, _ = generate_linear_data(n_samples=50, n_features=8)

        model = NeuralBoostedRegressor(
            n_estimators=1,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        @test length(model.estimators_) == 1
        @test length(model.train_score_) == 1

        y_pred = predict(model, X)
        @test length(y_pred) == 50
    end

    @testset "Constant Target" begin
        X = randn(50, 10)
        y = ones(50)  # All targets are 1.0

        model = NeuralBoostedRegressor(
            n_estimators=10,
            early_stopping=false,
            verbose=0
        )

        # Should handle gracefully
        @test_nowarn fit!(model, X, y)

        y_pred = predict(model, X)

        # All predictions should be close to 1.0
        @test mean(y_pred) ≈ 1.0 atol=0.5
    end

    @testset "Zero Target" begin
        X = randn(50, 10)
        y = zeros(50)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        y_pred = predict(model, X)

        # All predictions should be close to 0.0
        @test mean(abs.(y_pred)) < 0.5
    end
end


# ============================================================================
# Test Reproducibility
# ============================================================================

@testset "Reproducibility" begin

    @testset "Same Seed Same Results" begin
        X, y, _ = generate_linear_data(n_samples=60, n_features=10, random_state=42)

        # Fit two models with same random state
        model1 = NeuralBoostedRegressor(
            n_estimators=15,
            random_state=42,
            early_stopping=false,
            verbose=0
        )
        fit!(model1, X, y)

        model2 = NeuralBoostedRegressor(
            n_estimators=15,
            random_state=42,
            early_stopping=false,
            verbose=0
        )
        fit!(model2, X, y)

        # Predictions should be identical
        y_pred1 = predict(model1, X)
        y_pred2 = predict(model2, X)

        @test y_pred1 ≈ y_pred2 atol=1e-6
    end

    @testset "Different Seed Different Results" begin
        X, y, _ = generate_linear_data(n_samples=60, n_features=10, random_state=42)

        model1 = NeuralBoostedRegressor(
            n_estimators=15,
            random_state=42,
            early_stopping=false,
            verbose=0
        )
        fit!(model1, X, y)

        model2 = NeuralBoostedRegressor(
            n_estimators=15,
            random_state=123,
            early_stopping=false,
            verbose=0
        )
        fit!(model2, X, y)

        y_pred1 = predict(model1, X)
        y_pred2 = predict(model2, X)

        # Predictions should differ
        @test !(y_pred1 ≈ y_pred2)
    end
end


# ============================================================================
# Test Numerical Stability
# ============================================================================

@testset "Numerical Stability" begin

    @testset "No NaN or Inf in Training" begin
        X, y, _ = generate_linear_data(n_samples=80, n_features=12)

        model = NeuralBoostedRegressor(
            n_estimators=20,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        # Training scores should be finite
        @test all(isfinite.(model.train_score_))
        @test !any(isnan.(model.train_score_))
        @test !any(isinf.(model.train_score_))

        # Predictions should be finite
        y_pred = predict(model, X)
        @test all(isfinite.(y_pred))
        @test !any(isnan.(y_pred))
        @test !any(isinf.(y_pred))
    end

    @testset "No NaN or Inf in Importances" begin
        X, y, _ = generate_linear_data(n_samples=80, n_features=15)

        model = NeuralBoostedRegressor(
            n_estimators=15,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X, y)

        importances = feature_importances(model)

        @test all(isfinite.(importances))
        @test !any(isnan.(importances))
        @test !any(isinf.(importances))
    end

    @testset "Extreme Values" begin
        # Data with large values
        X_large = randn(50, 10) .* 1000
        y_large = randn(50) .* 1000

        model = NeuralBoostedRegressor(
            n_estimators=10,
            early_stopping=false,
            verbose=0
        )

        @test_nowarn fit!(model, X_large, y_large)

        y_pred = predict(model, X_large)
        @test all(isfinite.(y_pred))
    end
end


# ============================================================================
# Integration Tests
# ============================================================================

@testset "Integration Tests" begin

    @testset "Full Workflow" begin
        # Generate train and test data
        X_train, y_train, _ = generate_boosting_data(
            n_samples=100, n_features=20, n_informative=5
        )
        X_test, y_test, _ = generate_boosting_data(
            n_samples=30, n_features=20, n_informative=5, random_state=123
        )

        # Create and fit model
        model = NeuralBoostedRegressor(
            n_estimators=30,
            learning_rate=0.1,
            hidden_layer_size=4,
            early_stopping=true,
            validation_fraction=0.15,
            verbose=0
        )

        fit!(model, X_train, y_train)

        # Make predictions
        y_pred_train = predict(model, X_train)
        y_pred_test = predict(model, X_test)

        # Get feature importances
        importances = feature_importances(model)

        # Verify outputs
        @test length(y_pred_train) == 100
        @test length(y_pred_test) == 30
        @test length(importances) == 20

        # Check performance
        train_corr = cor(y_train, y_pred_train)
        test_corr = cor(y_test, y_pred_test)

        @test train_corr > 0.5  # Reasonable training performance
        @test test_corr > 0.3   # Some generalization
    end

    @testset "Compare MSE vs Huber Loss" begin
        X, y, outlier_indices = generate_data_with_outliers(
            n_samples=100, n_features=10, n_outliers=8
        )

        # MSE loss
        model_mse = NeuralBoostedRegressor(
            n_estimators=25,
            loss="mse",
            early_stopping=false,
            verbose=0
        )
        fit!(model_mse, X, y)

        # Huber loss
        model_huber = NeuralBoostedRegressor(
            n_estimators=25,
            loss="huber",
            early_stopping=false,
            verbose=0
        )
        fit!(model_huber, X, y)

        # Both should complete successfully
        @test length(model_mse.estimators_) == 25
        @test length(model_huber.estimators_) == 25

        # Get predictions
        y_pred_mse = predict(model_mse, X)
        y_pred_huber = predict(model_huber, X)

        # Both should produce valid predictions
        @test all(isfinite.(y_pred_mse))
        @test all(isfinite.(y_pred_huber))
    end
end


# ============================================================================
# Run all tests
# ============================================================================

println("\n" * "="^70)
println("Neural Boosted Test Suite Complete")
println("="^70)
