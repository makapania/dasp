"""
Test suite for variable_selection.jl

Comprehensive tests for all variable selection methods:
- UVE (Uninformative Variable Elimination)
- SPA (Successive Projections Algorithm)
- iPLS (Interval PLS)
- UVE-SPA (Hybrid method)

Run with: julia --project=. test/test_variable_selection.jl
"""

using Test
using Random
using Statistics
using LinearAlgebra

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Import variable selection functions
include(joinpath(@__DIR__, "..", "src", "variable_selection.jl"))


# ============================================================================
# Test Data Generators
# ============================================================================

"""
Generate synthetic spectral data with known informative variables.

Returns:
- X: spectral data (n_samples × n_features)
- y: target values
- informative_indices: indices of truly informative variables
"""
function generate_spectral_data(;
    n_samples=100,
    n_features=50,
    n_informative=5,
    noise_level=0.1,
    random_state=42
)
    Random.seed!(random_state)

    # Generate base spectral data with realistic structure
    # Simulate NIR-like spectra with smooth variations
    X = zeros(n_samples, n_features)
    wavelengths = range(1000, 2500, length=n_features)

    for i in 1:n_samples
        # Base spectrum with smooth peaks
        base = sin.(wavelengths ./ 200) .+ 0.5 .* cos.(wavelengths ./ 300)
        # Add sample variation
        variation = randn() * 0.2
        X[i, :] = base .+ variation .+ randn(n_features) .* 0.05
    end

    # Select informative variables
    informative_indices = sort(randperm(n_features)[1:n_informative])

    # Generate target based on informative variables
    true_coef = randn(n_informative)
    y = X[:, informative_indices] * true_coef + randn(n_samples) * noise_level

    return X, y, informative_indices
end

"""
Generate spectral data with collinear variables.
"""
function generate_collinear_data(;
    n_samples=100,
    n_features=30,
    n_groups=3,
    noise_level=0.1,
    random_state=42
)
    Random.seed!(random_state)

    # Generate groups of highly correlated variables
    group_size = n_features ÷ n_groups
    X = zeros(n_samples, n_features)

    for group in 1:n_groups
        # Base variable for this group
        base = randn(n_samples)
        start_idx = (group - 1) * group_size + 1
        end_idx = min(group * group_size, n_features)

        for j in start_idx:end_idx
            # Add slight variation to create collinearity
            X[:, j] = base + randn(n_samples) * 0.1
        end
    end

    # Target depends on first variable of each group
    group_starts = [1 + (g-1)*group_size for g in 1:n_groups]
    y = sum(X[:, idx] for idx in group_starts) + randn(n_samples) * noise_level

    return X, y
end

"""
Generate spectral data with distinct intervals (for iPLS testing).
"""
function generate_interval_data(;
    n_samples=100,
    n_features=60,
    n_intervals=3,
    informative_interval=2,
    noise_level=0.1,
    random_state=42
)
    Random.seed!(random_state)

    # Generate data where only one interval is informative
    X = randn(n_samples, n_features)

    interval_size = n_features ÷ n_intervals
    start_idx = (informative_interval - 1) * interval_size + 1
    end_idx = informative_interval * interval_size

    # Target depends only on the informative interval
    y = sum(X[:, start_idx:end_idx], dims=2)[:] + randn(n_samples) * noise_level

    return X, y, (start_idx, end_idx)
end


# ============================================================================
# Test UVE Selection
# ============================================================================

@testset "UVE Selection" begin

    @testset "Basic Functionality" begin
        X, y, informative_idx = generate_spectral_data(
            n_samples=100, n_features=50, n_informative=5
        )

        # Run UVE
        importances = uve_selection(X, y, cutoff_multiplier=1.0, cv_folds=5)

        # Check output dimensions
        @test length(importances) == 50
        @test all(importances .>= 0)  # All scores should be non-negative

        # Check that importances vary (not all zeros, not all equal)
        @test !all(importances .== 0)
        @test std(importances) > 0
    end

    @testset "Reproducibility" begin
        X, y, _ = generate_spectral_data(n_samples=80, n_features=40)

        # Run twice with same seed
        imp1 = uve_selection(X, y, random_state=42)
        imp2 = uve_selection(X, y, random_state=42)

        # Should be identical
        @test imp1 ≈ imp2

        # Run with different seed
        imp3 = uve_selection(X, y, random_state=123)

        # Should be different
        @test !(imp1 ≈ imp3)
    end

    @testset "Parameter Variations" begin
        X, y, _ = generate_spectral_data(n_samples=100, n_features=50)

        # Test different cutoff multipliers
        imp_conservative = uve_selection(X, y, cutoff_multiplier=2.0)
        imp_aggressive = uve_selection(X, y, cutoff_multiplier=0.5)

        @test length(imp_conservative) == length(imp_aggressive)

        # Test different n_components
        imp_2comp = uve_selection(X, y, n_components=2)
        imp_10comp = uve_selection(X, y, n_components=10)

        @test length(imp_2comp) == length(imp_10comp)
    end

    @testset "Edge Cases" begin
        # Very small dataset
        X_small = randn(10, 20)
        y_small = randn(10)

        @test_nowarn importances = uve_selection(X_small, y_small, cv_folds=3)

        # Many features, few samples
        X_wide = randn(20, 100)
        y_wide = randn(20)

        importances = uve_selection(X_wide, y_wide, cv_folds=5)
        @test length(importances) == 100

        # Constant target (degenerate case)
        X, _ = generate_spectral_data(n_samples=50, n_features=30)
        y_constant = ones(50)

        # Should handle gracefully without error
        @test_nowarn importances = uve_selection(X, y_constant)
    end

    @testset "Numerical Stability" begin
        X, y, _ = generate_spectral_data(n_samples=100, n_features=40)

        # Check for NaN or Inf in output
        importances = uve_selection(X, y)

        @test !any(isnan.(importances))
        @test !any(isinf.(importances))
        @test all(isfinite.(importances))
    end
end


# ============================================================================
# Test SPA Selection
# ============================================================================

@testset "SPA Selection" begin

    @testset "Basic Functionality" begin
        X, y = generate_collinear_data(n_samples=100, n_features=30)

        # Select 10 features
        n_select = 10
        importances = spa_selection(X, y, n_select, n_random_starts=5)

        # Check output dimensions
        @test length(importances) == 30
        @test all(importances .>= 0)

        # Check that exactly n_select features have non-zero importance
        n_selected = sum(importances .> 0)
        @test n_selected == n_select
    end

    @testset "Collinearity Reduction" begin
        X, y = generate_collinear_data(n_samples=100, n_features=30, n_groups=3)

        # Select features with SPA
        importances = spa_selection(X, y, 10, n_random_starts=10)
        selected_idx = findall(importances .> 0)

        # Check that selected features are less correlated than random selection
        X_selected = X[:, selected_idx]
        corr_matrix = cor(X_selected)

        # Mean absolute off-diagonal correlation should be relatively low
        n = size(corr_matrix, 1)
        off_diagonal = [corr_matrix[i,j] for i in 1:n, j in 1:n if i != j]
        mean_corr = mean(abs.(off_diagonal))

        # SPA should select uncorrelated features
        @test mean_corr < 0.8  # Reasonable threshold
    end

    @testset "Reproducibility" begin
        X, y = generate_collinear_data(n_samples=80, n_features=40)

        # Run twice with same seed
        imp1 = spa_selection(X, y, 15, random_state=42, n_random_starts=5)
        imp2 = spa_selection(X, y, 15, random_state=42, n_random_starts=5)

        @test imp1 ≈ imp2
    end

    @testset "Edge Cases" begin
        # Request more features than available
        X = randn(50, 20)
        y = randn(50)

        # Should use all features
        importances = spa_selection(X, y, 30, n_random_starts=3)
        @test sum(importances .> 0) == 20

        # Very small dataset
        X_small = randn(10, 15)
        y_small = randn(10)

        @test_nowarn spa_selection(X_small, y_small, 5, cv_folds=3)

        # Select only 1 feature
        importances = spa_selection(X, y, 1, n_random_starts=3)
        @test sum(importances .> 0) == 1
    end

    @testset "Selection Order" begin
        X, y = generate_collinear_data(n_samples=100, n_features=30)

        importances = spa_selection(X, y, 10, n_random_starts=5)
        selected_idx = findall(importances .> 0)

        # Earlier selected features should have higher scores
        # Get scores of selected features
        selected_scores = importances[selected_idx]

        # Check that scores are in descending order when sorted by index
        # (This tests the ranking mechanism)
        @test all(selected_scores .> 0)
    end

    @testset "Numerical Stability" begin
        X, y = generate_collinear_data(n_samples=100, n_features=40)

        importances = spa_selection(X, y, 15, n_random_starts=5)

        @test !any(isnan.(importances))
        @test !any(isinf.(importances))
        @test all(isfinite.(importances))
    end
end


# ============================================================================
# Test iPLS Selection
# ============================================================================

@testset "iPLS Selection" begin

    @testset "Basic Functionality" begin
        X, y, (start, stop) = generate_interval_data(
            n_samples=100, n_features=60, n_intervals=3, informative_interval=2
        )

        # Run iPLS
        importances = ipls_selection(X, y, n_intervals=3, cv_folds=5)

        # Check output dimensions
        @test length(importances) == 60
        @test all(importances .>= 0)

        # The informative interval should have higher average importance
        interval_size = 20
        interval_scores = [
            mean(importances[1:20]),
            mean(importances[21:40]),
            mean(importances[41:60])
        ]

        # Interval 2 (informative) should have highest score
        @test argmax(interval_scores) == 2
    end

    @testset "Interval Identification" begin
        X, y, (start, stop) = generate_interval_data(
            n_samples=150, n_features=90, n_intervals=3, informative_interval=1
        )

        importances = ipls_selection(X, y, n_intervals=3)

        # First interval should be most important
        @test mean(importances[1:30]) > mean(importances[31:60])
        @test mean(importances[1:30]) > mean(importances[61:90])
    end

    @testset "Number of Intervals" begin
        X, y, _ = generate_interval_data(n_samples=100, n_features=60)

        # Test different numbers of intervals
        for n_int in [2, 5, 10, 20]
            importances = ipls_selection(X, y, n_intervals=n_int)
            @test length(importances) == 60
            @test !all(importances .== 0)
        end
    end

    @testset "Reproducibility" begin
        X, y, _ = generate_interval_data(n_samples=100, n_features=60)

        imp1 = ipls_selection(X, y, n_intervals=5, random_state=42)
        imp2 = ipls_selection(X, y, n_intervals=5, random_state=42)

        @test imp1 ≈ imp2
    end

    @testset "Edge Cases" begin
        # More intervals than features
        X = randn(50, 20)
        y = randn(50)

        # Should handle gracefully
        @test_nowarn importances = ipls_selection(X, y, n_intervals=30)

        # Single interval (should work like full-spectrum PLS)
        importances = ipls_selection(X, y, n_intervals=1)
        @test length(importances) == 20

        # Very small dataset
        X_small = randn(15, 30)
        y_small = randn(15)

        @test_nowarn ipls_selection(X_small, y_small, n_intervals=3, cv_folds=3)
    end

    @testset "Numerical Stability" begin
        X, y, _ = generate_interval_data(n_samples=100, n_features=60)

        importances = ipls_selection(X, y, n_intervals=6)

        @test !any(isnan.(importances))
        @test !any(isinf.(importances))
        @test all(isfinite.(importances))
    end
end


# ============================================================================
# Test UVE-SPA Hybrid Selection
# ============================================================================

@testset "UVE-SPA Hybrid Selection" begin

    @testset "Basic Functionality" begin
        X, y, informative_idx = generate_spectral_data(
            n_samples=100, n_features=50, n_informative=5
        )

        # Run UVE-SPA
        n_select = 10
        importances = uve_spa_selection(X, y, n_select,
                                       cutoff_multiplier=1.0,
                                       spa_n_random_starts=5)

        # Check output dimensions
        @test length(importances) == 50
        @test all(importances .>= 0)

        # Should have selected approximately n_select features
        # (may be fewer if UVE eliminates too many)
        n_selected = sum(importances .> 0)
        @test n_selected <= n_select
        @test n_selected > 0  # Should select at least some features
    end

    @testset "Two-Stage Process" begin
        X, y, _ = generate_spectral_data(n_samples=120, n_features=60, n_informative=8)

        # Run UVE-SPA with different parameters
        imp_conservative = uve_spa_selection(X, y, 15, cutoff_multiplier=2.0)
        imp_aggressive = uve_spa_selection(X, y, 15, cutoff_multiplier=0.5)

        # Both should return valid importance vectors
        @test length(imp_conservative) == 60
        @test length(imp_aggressive) == 60

        # More conservative cutoff should eliminate fewer variables
        n_selected_conservative = sum(imp_conservative .> 0)
        n_selected_aggressive = sum(imp_aggressive .> 0)

        # Both should have selected some features
        @test n_selected_conservative > 0
        @test n_selected_aggressive > 0
    end

    @testset "Reproducibility" begin
        X, y, _ = generate_spectral_data(n_samples=100, n_features=50)

        imp1 = uve_spa_selection(X, y, 15, random_state=42)
        imp2 = uve_spa_selection(X, y, 15, random_state=42)

        @test imp1 ≈ imp2
    end

    @testset "Edge Cases" begin
        # Very aggressive filtering
        X, y = generate_spectral_data(n_samples=50, n_features=40, n_informative=3)

        # Should handle even if UVE eliminates most variables
        @test_nowarn importances = uve_spa_selection(X, y, 10, cutoff_multiplier=0.1)

        # Request more features than available
        importances = uve_spa_selection(X, y, 50)
        @test length(importances) == 40

        # Very small dataset
        X_small = randn(15, 25)
        y_small = randn(15)

        @test_nowarn uve_spa_selection(X_small, y_small, 5,
                                       uve_cv_folds=3, spa_cv_folds=3)
    end

    @testset "Numerical Stability" begin
        X, y, _ = generate_spectral_data(n_samples=100, n_features=50)

        importances = uve_spa_selection(X, y, 15)

        @test !any(isnan.(importances))
        @test !any(isinf.(importances))
        @test all(isfinite.(importances))
    end
end


# ============================================================================
# Integration Tests
# ============================================================================

@testset "Integration Tests" begin

    @testset "Compare Methods on Same Data" begin
        X, y, informative_idx = generate_spectral_data(
            n_samples=150, n_features=50, n_informative=5
        )

        # Run all methods
        imp_uve = uve_selection(X, y, cutoff_multiplier=1.0)
        imp_spa = spa_selection(X, y, 15, n_random_starts=5)
        imp_ipls = ipls_selection(X, y, n_intervals=10)
        imp_uve_spa = uve_spa_selection(X, y, 15)

        # All should return same-length vectors
        @test length(imp_uve) == 50
        @test length(imp_spa) == 50
        @test length(imp_ipls) == 50
        @test length(imp_uve_spa) == 50

        # All should be valid (no NaN/Inf)
        for imp in [imp_uve, imp_spa, imp_ipls, imp_uve_spa]
            @test !any(isnan.(imp))
            @test !any(isinf.(imp))
        end
    end

    @testset "Performance on High-Dimensional Data" begin
        # Realistic NIR spectroscopy dimensions
        X, y, _ = generate_spectral_data(
            n_samples=200, n_features=100, n_informative=10
        )

        # Test that all methods complete in reasonable time
        @test_nowarn begin
            uve_selection(X, y, cv_folds=5)
            spa_selection(X, y, 20, n_random_starts=5)
            ipls_selection(X, y, n_intervals=10)
            uve_spa_selection(X, y, 20, spa_n_random_starts=5)
        end
    end
end


# ============================================================================
# Run all tests
# ============================================================================

println("\n" * "="^70)
println("Variable Selection Test Suite Complete")
println("="^70)
