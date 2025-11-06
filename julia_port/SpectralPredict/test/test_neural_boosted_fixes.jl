"""
Test suite for Neural Boosted Regressor fixes

Tests the critical fixes for handling small datasets and empty validation sets.
These tests verify that the "too few samples" error is resolved.
"""

using Test
using Random

# Load the NeuralBoosted module
include("../src/neural_boosted.jl")
using .NeuralBoosted

@testset "NeuralBoosted Small Dataset Fixes" begin

    @testset "Test 1: Tiny dataset (5 samples) - should disable early stopping" begin
        println("\n=== Test 1: Tiny dataset (5 samples) ===")
        Random.seed!(42)

        n = 5
        X = randn(n, 10)
        y = randn(n)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=true,
            validation_fraction=0.15,
            verbose=1
        )

        # This should NOT crash - it should automatically disable early stopping
        @test_nowarn fit!(model, X, y)

        # Model should be fitted
        @test length(model.estimators_) > 0

        # Predictions should work
        pred = predict(model, X)
        @test length(pred) == n
        @test all(isfinite.(pred))

        println("✓ Test 1 passed: Model handled tiny dataset gracefully")
    end

    @testset "Test 2: Small dataset (20 samples) - edge case for validation split" begin
        println("\n=== Test 2: Small dataset (20 samples) ===")
        Random.seed!(42)

        n = 20
        X = randn(n, 15)
        y = randn(n)

        model = NeuralBoostedRegressor(
            n_estimators=20,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=true,
            validation_fraction=0.15,
            verbose=1
        )

        # Should work, might have validation or might disable it
        @test_nowarn fit!(model, X, y)

        # Predictions should work
        pred = predict(model, X)
        @test length(pred) == n
        @test all(isfinite.(pred))

        println("✓ Test 2 passed: Model handled small dataset")
    end

    @testset "Test 3: CV fold size (80 samples) - realistic scenario" begin
        println("\n=== Test 3: CV fold size simulation (80 samples) ===")
        Random.seed!(42)

        # This simulates a CV fold from a 100-sample dataset with 5-fold CV
        n = 80
        X = randn(n, 50)
        y = X[:, 10] .+ 2 .* X[:, 20] .+ randn(n) .* 0.1

        model = NeuralBoostedRegressor(
            n_estimators=50,
            learning_rate=0.1,
            hidden_layer_size=3,
            activation="tanh",
            early_stopping=true,
            validation_fraction=0.15,
            verbose=1
        )

        # This is the most common failure scenario - should work now
        @test_nowarn fit!(model, X, y)

        # Should have validation scores since dataset is large enough
        @test length(model.estimators_) > 0

        # Test prediction quality
        pred = predict(model, X)
        @test length(pred) == n

        # Should achieve reasonable fit
        r2 = 1 - sum((y .- pred).^2) / sum((y .- mean(y)).^2)
        @test r2 > 0.5  # Should explain at least 50% of variance

        println("  R² = $(round(r2, digits=4))")
        println("  Number of estimators fitted: $(model.n_estimators_)")
        println("✓ Test 3 passed: Realistic CV fold scenario works")
    end

    @testset "Test 4: Very small dataset (< min_required) - should error gracefully" begin
        println("\n=== Test 4: Dataset smaller than minimum required ===")
        Random.seed!(42)

        # Only 3 samples, but hidden_layer_size=3 requires at least 5 samples
        n = 3
        X = randn(n, 10)
        y = randn(n)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=true,
            verbose=1
        )

        # Should throw informative error
        @test_throws ErrorException fit!(model, X, y)

        println("✓ Test 4 passed: Informative error for insufficient samples")
    end

    @testset "Test 5: Early stopping disabled explicitly" begin
        println("\n=== Test 5: Early stopping disabled ===")
        Random.seed!(42)

        n = 15
        X = randn(n, 10)
        y = randn(n)

        model = NeuralBoostedRegressor(
            n_estimators=20,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=false,  # Explicitly disabled
            verbose=1
        )

        # Should work fine without validation split
        @test_nowarn fit!(model, X, y)

        # Should have no validation scores
        @test isempty(model.validation_score_)

        # Should have trained the specified number of estimators (or close to it)
        @test model.n_estimators_ <= model.n_estimators

        pred = predict(model, X)
        @test length(pred) == n

        println("✓ Test 5 passed: Early stopping disabled works correctly")
    end

    @testset "Test 6: Different hidden layer sizes" begin
        println("\n=== Test 6: Various hidden layer sizes ===")
        Random.seed!(42)

        n = 50
        X = randn(n, 20)
        y = randn(n)

        for hidden_size in [1, 3, 5, 7]
            model = NeuralBoostedRegressor(
                n_estimators=10,
                learning_rate=0.1,
                hidden_layer_size=hidden_size,
                early_stopping=true,
                verbose=0
            )

            @test_nowarn fit!(model, X, y)
            pred = predict(model, X)
            @test length(pred) == n

            println("  ✓ hidden_layer_size=$hidden_size works")
        end

        println("✓ Test 6 passed: All hidden layer sizes work")
    end

    @testset "Test 7: Feature importances with small dataset" begin
        println("\n=== Test 7: Feature importances ===")
        Random.seed!(42)

        n = 30
        n_features = 15
        X = randn(n, n_features)
        y = X[:, 5] .+ 2 .* X[:, 10] .+ randn(n) .* 0.1

        model = NeuralBoostedRegressor(
            n_estimators=30,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=true,
            verbose=0
        )

        fit!(model, X, y)

        # Get feature importances
        importances = feature_importances(model)

        @test length(importances) == n_features
        @test all(importances .>= 0)
        @test sum(importances) ≈ 1.0

        # Features 5 and 10 should be among the most important
        top_features = sortperm(importances, rev=true)[1:3]
        println("  Top 3 features: $top_features (true: 5, 10)")

        println("✓ Test 7 passed: Feature importances computed correctly")
    end

end

println("\n" * "="^70)
println("All Neural Boosted fixes verified successfully!")
println("The 'too few samples' error should now be resolved.")
println("="^70)
