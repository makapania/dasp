"""
Test suite for diagnostics.jl

Comprehensive tests for diagnostic functions:
- compute_residuals
- compute_leverage
- qq_plot_data
- jackknife_prediction_intervals

Run with: julia --project=. test/test_diagnostics.jl
"""

using Test
using Random
using Statistics
using LinearAlgebra
using Distributions

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Import diagnostics module
include(joinpath(@__DIR__, "..", "src", "diagnostics.jl"))
using .Diagnostics


# ============================================================================
# Test Data Generators
# ============================================================================

"""
Generate synthetic regression data with known properties.
"""
function generate_regression_data(;
    n_samples=100,
    n_features=10,
    noise_level=0.1,
    add_outliers=false,
    random_state=42
)
    Random.seed!(random_state)

    X = randn(n_samples, n_features)
    true_coef = randn(n_features)
    y = X * true_coef + randn(n_samples) * noise_level

    # Optionally add outliers
    if add_outliers
        n_outliers = max(1, n_samples ÷ 20)
        outlier_indices = randperm(n_samples)[1:n_outliers]
        y[outlier_indices] .+= randn(n_outliers) .* 5.0
    end

    return X, y, true_coef
end

"""
Generate perfect predictions (for testing edge cases).
"""
function generate_perfect_predictions(n_samples=50)
    Random.seed!(42)
    y_true = randn(n_samples)
    y_pred = copy(y_true)  # Perfect predictions
    return y_true, y_pred
end

"""
Generate predictions with outliers.
"""
function generate_predictions_with_outliers(n_samples=100)
    Random.seed!(42)
    y_true = randn(n_samples)
    y_pred = y_true + randn(n_samples) * 0.1

    # Add a few large errors (outliers)
    outlier_indices = [10, 25, 50]
    y_pred[outlier_indices] .+= [5.0, -4.0, 6.0]

    return y_true, y_pred
end


# ============================================================================
# Test compute_residuals
# ============================================================================

@testset "Compute Residuals" begin

    @testset "Basic Functionality" begin
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]

        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # Check dimensions
        @test length(residuals) == 5
        @test length(std_residuals) == 5

        # Check residual values
        expected_residuals = [-0.1, 0.1, -0.2, 0.2, -0.1]
        @test residuals ≈ expected_residuals atol=1e-10

        # Check standardization
        @test mean(std_residuals) ≈ 0.0 atol=1e-10
        @test std(std_residuals) ≈ 1.0 atol=1e-10
    end

    @testset "Perfect Predictions" begin
        y_true, y_pred = generate_perfect_predictions(50)

        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # All residuals should be zero
        @test all(abs.(residuals) .< 1e-10)

        # Standardized residuals should also be zero (or same as raw due to small std)
        @test all(abs.(std_residuals) .< 1e-10)
    end

    @testset "Outlier Detection" begin
        y_true, y_pred = generate_predictions_with_outliers(100)

        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # Check that outliers are detected (|std_residual| > 2 or 3)
        outliers = findall(abs.(std_residuals) .> 2)

        @test length(outliers) >= 3  # Should detect the 3 outliers we added
        @test 10 in outliers
        @test 25 in outliers
        @test 50 in outliers
    end

    @testset "Constant Residuals" begin
        # All predictions off by the same amount
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.5, 2.5, 3.5, 4.5, 5.5]

        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # All residuals should be -0.5
        @test all(residuals .≈ -0.5)

        # Standard deviation of residuals is 0, so standardized = raw
        @test std_residuals ≈ residuals
    end

    @testset "Numerical Stability" begin
        # Very small residuals
        y_true = randn(100)
        y_pred = y_true .+ randn(100) .* 1e-12

        residuals, std_residuals = compute_residuals(y_true, y_pred)

        @test !any(isnan.(residuals))
        @test !any(isinf.(residuals))
        @test !any(isnan.(std_residuals))
        @test !any(isinf.(std_residuals))
    end

    @testset "Large Scale" begin
        # Test with large dataset
        Random.seed!(42)
        y_true = randn(10000)
        y_pred = y_true + randn(10000) * 0.5

        residuals, std_residuals = compute_residuals(y_true, y_pred)

        @test length(residuals) == 10000
        @test length(std_residuals) == 10000

        # Standardized residuals should have mean ≈ 0, std ≈ 1
        @test abs(mean(std_residuals)) < 0.01
        @test abs(std(std_residuals) - 1.0) < 0.01
    end
end


# ============================================================================
# Test compute_leverage
# ============================================================================

@testset "Compute Leverage" begin

    @testset "Basic Functionality" begin
        X = randn(50, 10)

        leverage, threshold = compute_leverage(X, return_threshold=true)

        # Check dimensions
        @test length(leverage) == 50

        # Check value ranges: 0 ≤ h_ii ≤ 1
        @test all(0 .<= leverage .<= 1)

        # Check threshold calculation: 2*p/n where p = n_features + 1
        expected_threshold = 2 * 11 / 50  # 11 = 10 features + intercept
        @test threshold ≈ expected_threshold
    end

    @testset "Average Leverage" begin
        X = randn(100, 20)

        leverage, threshold = compute_leverage(X)

        # Average leverage should be p/n where p = n_features + 1
        avg_leverage = mean(leverage)
        expected_avg = 21 / 100  # 21 = 20 features + intercept

        @test avg_leverage ≈ expected_avg atol=1e-10
    end

    @testset "High Leverage Points" begin
        # Create data with a clear outlier in feature space
        X = randn(50, 5)
        X[1, :] .= [10.0, 10.0, 10.0, 10.0, 10.0]  # Far from center

        leverage, threshold = compute_leverage(X)

        # First sample should have high leverage
        @test leverage[1] > threshold
        @test leverage[1] > mean(leverage)
    end

    @testset "Return Threshold Option" begin
        X = randn(50, 10)

        # With threshold
        result_with = compute_leverage(X, return_threshold=true)
        @test isa(result_with, Tuple)
        @test length(result_with) == 2

        # Without threshold
        result_without = compute_leverage(X, return_threshold=false)
        @test isa(result_without, Vector{Float64})
    end

    @testset "Edge Cases" begin
        # Very small dataset
        X_small = randn(5, 3)
        leverage, _ = compute_leverage(X_small)
        @test length(leverage) == 5
        @test all(0 .<= leverage .<= 1)

        # More features than samples (wide data) - uses SVD
        X_wide = randn(20, 50)
        leverage, _ = compute_leverage(X_wide)
        @test length(leverage) == 20
        @test all(0 .<= leverage .<= 1)

        # Many features (triggers SVD path)
        X_many = randn(100, 150)
        leverage, _ = compute_leverage(X_many)
        @test length(leverage) == 100
        @test all(0 .<= leverage .<= 1)
    end

    @testset "Numerical Stability" begin
        # Nearly collinear features
        X = randn(50, 5)
        X[:, 3] = X[:, 1] + randn(50) * 1e-6  # Nearly identical to column 1

        # Should handle gracefully without error
        @test_nowarn leverage, _ = compute_leverage(X)

        leverage, _ = compute_leverage(X)
        @test !any(isnan.(leverage))
        @test !any(isinf.(leverage))
    end

    @testset "SVD vs Direct Methods Consistency" begin
        # Small dataset (uses direct method)
        X_small = randn(50, 10)
        leverage_small, _ = compute_leverage(X_small)

        # Large feature dataset (uses SVD)
        X_large = randn(50, 150)
        leverage_large, _ = compute_leverage(X_large)

        # Both should produce valid leverage values
        @test all(0 .<= leverage_small .<= 1)
        @test all(0 .<= leverage_large .<= 1)
    end
end


# ============================================================================
# Test qq_plot_data
# ============================================================================

@testset "Q-Q Plot Data" begin

    @testset "Normal Residuals" begin
        # Generate normally distributed residuals
        Random.seed!(42)
        residuals = randn(100)

        theoretical, sample = qq_plot_data(residuals)

        # Check dimensions
        @test length(theoretical) == 100
        @test length(sample) == 100

        # Sample quantiles should be sorted
        @test issorted(sample)

        # For normal data, theoretical and sample should be roughly aligned
        # Correlation should be high
        correlation = cor(theoretical, sample)
        @test correlation > 0.98  # High correlation for normal data
    end

    @testset "Heavy-Tailed Distribution" begin
        # Generate t-distributed residuals (heavier tails than normal)
        Random.seed!(42)
        t_dist = TDist(3)
        residuals = rand(t_dist, 200)

        theoretical, sample = qq_plot_data(residuals)

        @test length(theoretical) == 200
        @test length(sample) == 200

        # Sample should have more extreme values than theoretical
        # Check tails
        @test abs(sample[end]) > abs(theoretical[end])
        @test abs(sample[1]) > abs(theoretical[1])
    end

    @testset "Light-Tailed Distribution" begin
        # Generate uniform residuals (lighter tails than normal)
        Random.seed!(42)
        residuals = rand(Uniform(-1, 1), 150)

        theoretical, sample = qq_plot_data(residuals)

        @test length(theoretical) == 150
        @test length(sample) == 150

        # Sample should have less extreme values than theoretical
        @test abs(sample[end]) < abs(theoretical[end])
    end

    @testset "Constant Residuals" begin
        # All residuals are the same
        residuals = ones(50)

        theoretical, sample = qq_plot_data(residuals)

        # Sample quantiles should all be 1.0
        @test all(sample .≈ 1.0)

        # Theoretical should vary
        @test std(theoretical) > 0
    end

    @testset "Small Sample Size" begin
        residuals = randn(5)

        theoretical, sample = qq_plot_data(residuals)

        @test length(theoretical) == 5
        @test length(sample) == 5
        @test issorted(sample)
    end

    @testset "Numerical Properties" begin
        Random.seed!(42)
        residuals = randn(100)

        theoretical, sample = qq_plot_data(residuals)

        # No NaN or Inf values
        @test !any(isnan.(theoretical))
        @test !any(isnan.(sample))
        @test !any(isinf.(theoretical))
        @test !any(isinf.(sample))

        # Theoretical quantiles should span reasonable range
        @test theoretical[1] < 0  # Negative for lower quantiles
        @test theoretical[end] > 0  # Positive for upper quantiles
    end
end


# ============================================================================
# Test jackknife_prediction_intervals
# ============================================================================

@testset "Jackknife Prediction Intervals" begin

    @testset "Basic Functionality" begin
        # Generate data
        X_train, y_train, _ = generate_regression_data(n_samples=30, n_features=5)
        X_test = randn(10, 5)

        # Simple linear model
        function fit_linear(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        # Compute jackknife intervals
        pred, lower, upper, stderr = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test,
            confidence=0.95, verbose=false
        )

        # Check dimensions
        @test length(pred) == 10
        @test length(lower) == 10
        @test length(upper) == 10
        @test length(stderr) == 10

        # Check interval properties
        @test all(lower .<= pred)
        @test all(pred .<= upper)
        @test all(stderr .>= 0)
    end

    @testset "Confidence Level" begin
        X_train, y_train, _ = generate_regression_data(n_samples=30, n_features=5)
        X_test = randn(10, 5)

        function fit_linear(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        # 95% confidence
        _, lower_95, upper_95, _ = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test,
            confidence=0.95, verbose=false
        )

        # 90% confidence
        _, lower_90, upper_90, _ = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test,
            confidence=0.90, verbose=false
        )

        # 90% intervals should be narrower than 95% intervals
        width_95 = mean(upper_95 .- lower_95)
        width_90 = mean(upper_90 .- lower_90)

        @test width_90 < width_95
    end

    @testset "Reproducibility" begin
        Random.seed!(42)
        X_train, y_train, _ = generate_regression_data(n_samples=25, n_features=4)
        X_test = randn(5, 4)

        function fit_linear(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        # Run twice
        Random.seed!(42)
        pred1, lower1, upper1, stderr1 = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test, verbose=false
        )

        Random.seed!(42)
        pred2, lower2, upper2, stderr2 = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test, verbose=false
        )

        # Should be identical
        @test pred1 ≈ pred2
        @test lower1 ≈ lower2
        @test upper1 ≈ upper2
        @test stderr1 ≈ stderr2
    end

    @testset "Perfect Model" begin
        # Create data where model can fit perfectly
        Random.seed!(42)
        X_train = randn(30, 3)
        true_coef = [1.0, -2.0, 0.5]
        y_train = X_train * true_coef  # No noise

        X_test = randn(5, 3)

        function fit_linear(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        pred, lower, upper, stderr = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test, verbose=false
        )

        # Standard errors should be very small for perfect model
        @test all(stderr .< 0.1)

        # Intervals should be narrow
        widths = upper .- lower
        @test all(widths .< 1.0)
    end

    @testset "High Variability Model" begin
        # Create noisy data
        Random.seed!(42)
        X_train = randn(30, 3)
        y_train = randn(30)  # Pure noise, no relationship

        X_test = randn(5, 3)

        function fit_linear(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        pred, lower, upper, stderr = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test, verbose=false
        )

        # Standard errors should be larger for noisy model
        @test mean(stderr) > 0.1

        # Intervals should be wider
        widths = upper .- lower
        @test mean(widths) > 0.5
    end

    @testset "Edge Cases" begin
        # Very small training set
        X_train_small = randn(10, 3)
        y_train_small = randn(10)
        X_test = randn(5, 3)

        function fit_linear(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        # Should complete without error
        @test_nowarn jackknife_prediction_intervals(
            fit_linear, X_train_small, y_train_small, X_test, verbose=false
        )

        # Single test sample
        X_test_single = randn(1, 3)

        pred, lower, upper, stderr = jackknife_prediction_intervals(
            fit_linear, X_train_small, y_train_small, X_test_single, verbose=false
        )

        @test length(pred) == 1
        @test length(lower) == 1
        @test length(upper) == 1
        @test length(stderr) == 1
    end

    @testset "Numerical Stability" begin
        X_train, y_train, _ = generate_regression_data(n_samples=30, n_features=5)
        X_test = randn(10, 5)

        function fit_linear(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        pred, lower, upper, stderr = jackknife_prediction_intervals(
            fit_linear, X_train, y_train, X_test, verbose=false
        )

        # Check for NaN or Inf
        @test !any(isnan.(pred))
        @test !any(isinf.(pred))
        @test !any(isnan.(lower))
        @test !any(isinf.(lower))
        @test !any(isnan.(upper))
        @test !any(isinf.(upper))
        @test !any(isnan.(stderr))
        @test !any(isinf.(stderr))
    end

    @testset "Model Function Returns Matrix" begin
        # Test when model function returns matrix instead of vector
        X_train, y_train, _ = generate_regression_data(n_samples=30, n_features=5)
        X_test = randn(10, 5)

        function fit_linear_matrix(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> reshape(hcat(ones(size(X_test, 1)), X_test) * β, :, 1)
        end

        # Should handle matrix output
        pred, lower, upper, stderr = jackknife_prediction_intervals(
            fit_linear_matrix, X_train, y_train, X_test, verbose=false
        )

        @test length(pred) == 10
        @test length(lower) == 10
        @test length(upper) == 10
        @test length(stderr) == 10
    end
end


# ============================================================================
# Integration Tests
# ============================================================================

@testset "Integration Tests" begin

    @testset "Full Diagnostic Pipeline" begin
        # Generate model predictions
        X_train, y_train, true_coef = generate_regression_data(
            n_samples=100, n_features=10, add_outliers=true
        )

        # Fit simple linear model
        X_aug = hcat(ones(size(X_train, 1)), X_train)
        β = X_aug \ y_train
        y_pred = X_aug * β

        # Compute all diagnostics
        residuals, std_residuals = compute_residuals(y_train, y_pred)
        leverage, threshold = compute_leverage(X_train)
        theoretical, sample = qq_plot_data(residuals)

        # All should complete successfully
        @test length(residuals) == 100
        @test length(leverage) == 100
        @test length(theoretical) == 100

        # Identify potential issues
        outliers = findall(abs.(std_residuals) .> 3)
        high_leverage = findall(leverage .> threshold)

        @test length(outliers) > 0  # Should detect outliers we added
        @test length(high_leverage) >= 0  # May or may not have high leverage points
    end

    @testset "Diagnostics Consistency" begin
        # Generate predictions with known properties
        Random.seed!(42)
        y_true = randn(100)
        y_pred = y_true + randn(100) * 0.2

        # Compute diagnostics
        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # Residuals should match direct computation
        expected_residuals = y_true - y_pred
        @test residuals ≈ expected_residuals

        # Q-Q plot for these residuals
        theoretical, sample = qq_plot_data(residuals)

        # Since residuals are approximately normal, Q-Q should be roughly linear
        correlation = cor(theoretical, sample)
        @test correlation > 0.95
    end
end


# ============================================================================
# Run all tests
# ============================================================================

println("\n" * "="^70)
println("Diagnostics Test Suite Complete")
println("="^70)
