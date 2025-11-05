"""
Test suite for MSC (Multiplicative Scatter Correction) preprocessing

Comprehensive tests for MSC functions in preprocessing.jl:
- apply_msc
- fit_msc
- Reference spectrum computation
- Scatter correction properties
- Edge cases and numerical stability

Run with: julia --project=. test/test_msc.jl
"""

using Test
using Random
using Statistics
using LinearAlgebra

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Import preprocessing functions
include(joinpath(@__DIR__, "..", "src", "preprocessing.jl"))


# ============================================================================
# Test Data Generators
# ============================================================================

"""
Generate synthetic spectral data with realistic structure.
"""
function generate_spectral_data(;
    n_samples=100,
    n_wavelengths=50,
    random_state=42
)
    Random.seed!(random_state)

    # Create base spectrum (smooth, realistic shape)
    wavelengths = range(1000, 2500, length=n_wavelengths)
    base_spectrum = sin.(wavelengths ./ 200) .+ 0.5 .* cos.(wavelengths ./ 300)

    # Generate sample spectra with variations
    X = zeros(n_samples, n_wavelengths)
    for i in 1:n_samples
        # Add sample-specific variation
        variation = randn() * 0.2
        noise = randn(n_wavelengths) * 0.05
        X[i, :] = base_spectrum .+ variation .+ noise
    end

    return X, base_spectrum
end

"""
Generate spectral data with known scatter effects.

Returns data with additive and multiplicative scatter,
plus the original clean spectra for comparison.
"""
function generate_scattered_data(;
    n_samples=100,
    n_wavelengths=50,
    scatter_strength=0.5,
    random_state=42
)
    Random.seed!(random_state)

    # Generate clean base spectra
    wavelengths = range(1000, 2500, length=n_wavelengths)
    base = sin.(wavelengths ./ 200) .+ 0.5 .* cos.(wavelengths ./ 300)

    X_clean = zeros(n_samples, n_wavelengths)
    X_scattered = zeros(n_samples, n_wavelengths)

    for i in 1:n_samples
        # Clean spectrum
        clean = base .+ randn() * 0.1 .+ randn(n_wavelengths) * 0.02
        X_clean[i, :] = clean

        # Add scatter effects
        a = randn() * scatter_strength  # Additive (baseline shift)
        b = 1.0 + randn() * scatter_strength  # Multiplicative (scaling)

        X_scattered[i, :] = a .+ b .* clean
    end

    return X_clean, X_scattered
end

"""
Generate data with constant spectra (edge case).
"""
function generate_constant_spectra(n_samples=50, n_wavelengths=30)
    # All spectra are identical
    base = ones(n_wavelengths) .* 100.0
    X = repeat(base', n_samples, 1)
    return X
end


# ============================================================================
# Test apply_msc - Basic Functionality
# ============================================================================

@testset "MSC Basic Functionality" begin

    @testset "Mean Reference (Default)" begin
        X, _ = generate_spectral_data(n_samples=50, n_wavelengths=30)

        X_corrected = apply_msc(X, reference=:mean)

        # Check dimensions preserved
        @test size(X_corrected) == size(X)

        # Check numerical stability
        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))

        # After MSC, mean spectrum should be close to zero
        mean_spectrum = vec(mean(X_corrected, dims=1))
        @test mean(abs.(mean_spectrum)) < mean(abs.(vec(mean(X, dims=1))))
    end

    @testset "Median Reference" begin
        X, _ = generate_spectral_data(n_samples=60, n_wavelengths=40)

        X_corrected = apply_msc(X, reference=:median)

        @test size(X_corrected) == size(X)
        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))
    end

    @testset "Custom Reference Vector" begin
        X, base_spectrum = generate_spectral_data(n_samples=50, n_wavelengths=30)

        # Use base spectrum as reference
        X_corrected = apply_msc(X, reference=base_spectrum)

        @test size(X_corrected) == size(X)
        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))
    end

    @testset "Pre-computed Reference Spectrum" begin
        X, _ = generate_spectral_data(n_samples=50, n_wavelengths=30)

        # Compute reference
        ref = vec(mean(X, dims=1))

        # Apply with reference_spectrum parameter
        X_corrected = apply_msc(X, reference_spectrum=ref)

        @test size(X_corrected) == size(X)
        @test !any(isnan.(X_corrected))

        # Should be same as using reference=:mean
        X_corrected_mean = apply_msc(X, reference=:mean)
        @test X_corrected ≈ X_corrected_mean
    end
end


# ============================================================================
# Test MSC Scatter Correction Properties
# ============================================================================

@testset "MSC Scatter Correction Properties" begin

    @testset "Removes Additive Effects" begin
        X_clean, X_scattered = generate_scattered_data(
            n_samples=80, n_wavelengths=40, scatter_strength=0.5
        )

        X_corrected = apply_msc(X_scattered, reference=:mean)

        # After correction, variance should be closer to clean data
        var_scattered = var(X_scattered, dims=1)
        var_corrected = var(X_corrected, dims=1)

        # Mean variance should decrease after correction
        @test mean(var_corrected) < mean(var_scattered)
    end

    @testset "Removes Multiplicative Effects" begin
        # Create data with only multiplicative scatter
        Random.seed!(42)
        n_samples = 60
        n_wavelengths = 30

        base = sin.(range(0, 2π, length=n_wavelengths))
        X = zeros(n_samples, n_wavelengths)

        for i in 1:n_samples
            scale = 1.0 + randn() * 0.3
            X[i, :] = scale .* base
        end

        X_corrected = apply_msc(X, reference=:mean)

        # Variance should be reduced
        @test var(vec(X_corrected)) < var(vec(X))
    end

    @testset "Preserves Spectral Information" begin
        X_clean, X_scattered = generate_scattered_data(
            n_samples=100, n_wavelengths=50, scatter_strength=0.4
        )

        X_corrected = apply_msc(X_scattered, reference=:mean)

        # Correlation with clean data should be high
        for i in 1:size(X_clean, 1)
            corr = cor(X_clean[i, :], X_corrected[i, :])
            @test corr > 0.8  # Strong correlation preserved
        end
    end

    @testset "Variance Reduction" begin
        X_clean, X_scattered = generate_scattered_data(
            n_samples=80, n_wavelengths=40, scatter_strength=0.6
        )

        # Variance before correction
        var_before = var(X_scattered, dims=1)

        # Apply MSC
        X_corrected = apply_msc(X_scattered, reference=:mean)

        # Variance after correction
        var_after = var(X_corrected, dims=1)

        # Most wavelengths should have reduced variance
        reduction_count = sum(var_after .< var_before)
        @test reduction_count > size(X_scattered, 2) * 0.6  # At least 60% reduced
    end
end


# ============================================================================
# Test fit_msc
# ============================================================================

@testset "fit_msc Function" begin

    @testset "Compute Mean Reference" begin
        X, _ = generate_spectral_data(n_samples=50, n_wavelengths=30)

        ref = fit_msc(X, reference=:mean)

        # Should return vector of correct length
        @test length(ref) == 30

        # Should match manual mean computation
        expected_ref = vec(mean(X, dims=1))
        @test ref ≈ expected_ref
    end

    @testset "Compute Median Reference" begin
        X, _ = generate_spectral_data(n_samples=60, n_wavelengths=40)

        ref = fit_msc(X, reference=:median)

        @test length(ref) == 40

        expected_ref = vec(median(X, dims=1))
        @test ref ≈ expected_ref
    end

    @testset "Pass Through Custom Reference" begin
        X, _ = generate_spectral_data(n_samples=50, n_wavelengths=30)

        custom_ref = sin.(range(0, 2π, length=30))
        ref = fit_msc(X, reference=custom_ref)

        @test length(ref) == 30
        @test ref ≈ custom_ref
    end

    @testset "Invalid Reference Dimension" begin
        X, _ = generate_spectral_data(n_samples=50, n_wavelengths=30)

        wrong_ref = ones(20)  # Wrong size

        @test_throws ArgumentError fit_msc(X, reference=wrong_ref)
    end

    @testset "Train/Test Consistency" begin
        X_train, _ = generate_spectral_data(n_samples=80, n_wavelengths=40, random_state=42)
        X_test, _ = generate_spectral_data(n_samples=20, n_wavelengths=40, random_state=123)

        # Fit on training data
        ref = fit_msc(X_train, reference=:mean)

        # Apply to both train and test
        X_train_corrected = apply_msc(X_train, reference_spectrum=ref)
        X_test_corrected = apply_msc(X_test, reference_spectrum=ref)

        # Both should be valid
        @test size(X_train_corrected) == size(X_train)
        @test size(X_test_corrected) == size(X_test)
        @test !any(isnan.(X_train_corrected))
        @test !any(isnan.(X_test_corrected))
    end
end


# ============================================================================
# Test Edge Cases
# ============================================================================

@testset "MSC Edge Cases" begin

    @testset "Constant Spectra" begin
        X = generate_constant_spectra(50, 30)

        # Should handle without error
        @test_nowarn X_corrected = apply_msc(X, reference=:mean)

        X_corrected = apply_msc(X, reference=:mean)

        # Output should be valid (likely mean-centered)
        @test size(X_corrected) == size(X)
        @test !any(isnan.(X_corrected))
    end

    @testset "Single Sample" begin
        X = randn(1, 50)

        X_corrected = apply_msc(X, reference=:mean)

        @test size(X_corrected) == (1, 50)
        @test !any(isnan.(X_corrected))
    end

    @testset "Two Samples" begin
        X = randn(2, 30)

        X_corrected = apply_msc(X, reference=:mean)

        @test size(X_corrected) == (2, 30)
        @test !any(isnan.(X_corrected))
    end

    @testset "Large Dataset" begin
        X = randn(1000, 100)

        # Should complete without error
        @test_nowarn X_corrected = apply_msc(X, reference=:mean)

        X_corrected = apply_msc(X, reference=:mean)
        @test size(X_corrected) == (1000, 100)
    end

    @testset "Few Wavelengths" begin
        X = randn(50, 5)  # Only 5 wavelengths

        X_corrected = apply_msc(X, reference=:mean)

        @test size(X_corrected) == (50, 5)
        @test !any(isnan.(X_corrected))
    end

    @testset "Collinear Features" begin
        # Create data where some features are nearly identical
        X = randn(50, 30)
        X[:, 10] = X[:, 5] .+ randn(50) .* 1e-6

        # Should handle gracefully
        X_corrected = apply_msc(X, reference=:mean)

        @test size(X_corrected) == size(X)
        @test !any(isnan.(X_corrected))
    end

    @testset "Zero Variance Feature" begin
        X = randn(50, 30)
        X[:, 15] .= 100.0  # Constant feature

        X_corrected = apply_msc(X, reference=:mean)

        @test size(X_corrected) == size(X)
        @test !any(isnan.(X_corrected))
    end

    @testset "Negative Values" begin
        # Spectral data can sometimes have negative values (e.g., after preprocessing)
        X = randn(50, 30)  # Mix of positive and negative

        X_corrected = apply_msc(X, reference=:mean)

        @test size(X_corrected) == size(X)
        @test !any(isnan.(X_corrected))
    end
end


# ============================================================================
# Test Numerical Stability
# ============================================================================

@testset "MSC Numerical Stability" begin

    @testset "Very Small Values" begin
        X = randn(50, 30) .* 1e-6

        X_corrected = apply_msc(X, reference=:mean)

        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))
        @test all(isfinite.(X_corrected))
    end

    @testset "Very Large Values" begin
        X = randn(50, 30) .* 1e6

        X_corrected = apply_msc(X, reference=:mean)

        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))
        @test all(isfinite.(X_corrected))
    end

    @testset "Mixed Magnitude Features" begin
        X = randn(50, 30)
        X[:, 1:10] .*= 1e-6    # Very small features
        X[:, 11:20] .*= 1.0    # Normal features
        X[:, 21:30] .*= 1e6    # Very large features

        X_corrected = apply_msc(X, reference=:mean)

        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))
    end

    @testset "Near-Singular Cases" begin
        # Create nearly singular design matrix scenario
        X = randn(50, 30)
        # Make first row very similar to reference
        ref = vec(mean(X, dims=1))
        X[1, :] = ref .+ randn(30) .* 1e-10

        # Should handle gracefully
        X_corrected = apply_msc(X, reference=:mean)

        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))
    end

    @testset "Flat Spectrum (Zero Slope)" begin
        # Spectrum with nearly zero slope relative to reference
        X = randn(50, 30)
        ref = vec(mean(X, dims=1))

        # Make one spectrum flat (constant) and equal to mean of reference
        X[1, :] .= mean(ref)

        X_corrected = apply_msc(X, reference=:mean)

        @test !any(isnan.(X_corrected))
        @test !any(isinf.(X_corrected))
    end
end


# ============================================================================
# Test Parameter Validation
# ============================================================================

@testset "MSC Parameter Validation" begin

    @testset "Invalid Reference Type" begin
        X = randn(50, 30)

        @test_throws ErrorException apply_msc(X, reference=:invalid)
    end

    @testset "Wrong Reference Dimension" begin
        X = randn(50, 30)
        wrong_ref = ones(20)  # Should be 30

        @test_throws ArgumentError apply_msc(X, reference=wrong_ref)
    end

    @testset "Reference Spectrum Overrides Reference Type" begin
        X, _ = generate_spectral_data(n_samples=50, n_wavelengths=30)

        custom_ref = sin.(range(0, 2π, length=30))

        # Even though we specify reference=:mean, reference_spectrum should take precedence
        X_corrected = apply_msc(X, reference=:mean, reference_spectrum=custom_ref)

        # Should use custom_ref, not mean
        X_corrected_custom = apply_msc(X, reference=custom_ref)

        @test X_corrected ≈ X_corrected_custom
    end
end


# ============================================================================
# Test Comparison with SNV
# ============================================================================

@testset "MSC vs SNV Comparison" begin

    @testset "Both Reduce Scatter" begin
        X_clean, X_scattered = generate_scattered_data(
            n_samples=80, n_wavelengths=40, scatter_strength=0.5
        )

        # Apply MSC
        X_msc = apply_msc(X_scattered, reference=:mean)

        # Apply SNV
        X_snv = apply_snv(X_scattered)

        # Both should reduce variance compared to scattered
        var_scattered = mean(var(X_scattered, dims=1))
        var_msc = mean(var(X_msc, dims=1))
        var_snv = mean(var(X_snv, dims=1))

        # Both correction methods should reduce variance
        @test var_msc < var_scattered
        @test var_snv < var_scattered
    end

    @testset "Different Mechanisms" begin
        X_clean, X_scattered = generate_scattered_data(
            n_samples=60, n_wavelengths=30
        )

        X_msc = apply_msc(X_scattered, reference=:mean)
        X_snv = apply_snv(X_scattered)

        # Results should be different (different correction mechanisms)
        @test !(X_msc ≈ X_snv)

        # But both should preserve spectral shape
        # Check correlation with clean for a sample
        for i in [1, 10, 30]
            corr_msc = cor(X_clean[i, :], X_msc[i, :])
            corr_snv = cor(X_clean[i, :], X_snv[i, :])

            @test corr_msc > 0.7
            @test corr_snv > 0.7
        end
    end
end


# ============================================================================
# Integration Tests
# ============================================================================

@testset "MSC Integration Tests" begin

    @testset "Full Preprocessing Workflow" begin
        # Generate training and test data with scatter
        X_train_clean, X_train_scattered = generate_scattered_data(
            n_samples=100, n_wavelengths=50, random_state=42
        )
        X_test_clean, X_test_scattered = generate_scattered_data(
            n_samples=30, n_wavelengths=50, random_state=123
        )

        # Fit MSC on training data
        ref = fit_msc(X_train_scattered, reference=:mean)

        # Apply to both train and test
        X_train_corrected = apply_msc(X_train_scattered, reference_spectrum=ref)
        X_test_corrected = apply_msc(X_test_scattered, reference_spectrum=ref)

        # Check dimensions
        @test size(X_train_corrected) == (100, 50)
        @test size(X_test_corrected) == (30, 50)

        # Check validity
        @test !any(isnan.(X_train_corrected))
        @test !any(isnan.(X_test_corrected))

        # Check that correction improves similarity to clean data
        # Training data
        corr_before_train = mean([cor(X_train_clean[i, :], X_train_scattered[i, :])
                                  for i in 1:10])
        corr_after_train = mean([cor(X_train_clean[i, :], X_train_corrected[i, :])
                                 for i in 1:10])
        @test corr_after_train >= corr_before_train * 0.9  # Should maintain or improve

        # Test data
        corr_before_test = mean([cor(X_test_clean[i, :], X_test_scattered[i, :])
                                for i in 1:10])
        corr_after_test = mean([cor(X_test_clean[i, :], X_test_corrected[i, :])
                               for i in 1:10])
        @test corr_after_test >= corr_before_test * 0.9
    end

    @testset "MSC with Different Preprocessing Combinations" begin
        X, _ = generate_spectral_data(n_samples=80, n_wavelengths=50)

        # MSC only
        X_msc = apply_msc(X, reference=:mean)
        @test size(X_msc) == size(X)

        # MSC then SNV
        X_msc_snv = apply_snv(X_msc)
        @test size(X_msc_snv) == size(X)

        # MSC then derivative (would need derivative function)
        # X_msc_deriv = apply_derivative(X_msc, deriv=1, window=11, polyorder=2)
        # @test size(X_msc_deriv) == size(X)

        # All should be valid
        @test !any(isnan.(X_msc))
        @test !any(isnan.(X_msc_snv))
    end
end


# ============================================================================
# Run all tests
# ============================================================================

println("\n" * "="^70)
println("MSC Test Suite Complete")
println("="^70)
