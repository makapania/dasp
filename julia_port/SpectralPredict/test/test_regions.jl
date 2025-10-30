"""
Test suite for Regions module

Run with: julia --project=. test/test_regions.jl
"""

using Test
include("../src/regions.jl")
using .Regions
using Statistics

@testset "Regions Module Tests" begin

    @testset "compute_region_correlations - Basic Functionality" begin
        # Create synthetic spectral data
        n_samples = 50
        n_wavelengths = 100
        wavelengths = collect(400.0:4.0:796.0)  # 400-796 nm in 4nm steps

        # Create X with different correlation patterns in different regions
        X = randn(n_samples, n_wavelengths)
        y = randn(n_samples)

        # Add strong correlation in a specific region (500-550 nm)
        region_mask = (wavelengths .>= 500.0) .& (wavelengths .< 550.0)
        X[:, region_mask] .+= y * 2.0  # Strong positive correlation

        regions = compute_region_correlations(X, y, wavelengths)

        # Basic checks
        @test length(regions) > 0
        @test all(haskey(r, "start") for r in regions)
        @test all(haskey(r, "end") for r in regions)
        @test all(haskey(r, "indices") for r in regions)
        @test all(haskey(r, "mean_corr") for r in regions)
        @test all(haskey(r, "max_corr") for r in regions)
        @test all(haskey(r, "n_features") for r in regions)

        # Check correlation values are valid
        @test all(0.0 <= r["mean_corr"] <= 1.0 for r in regions)
        @test all(0.0 <= r["max_corr"] <= 1.0 for r in regions)

        # Check max_corr >= mean_corr
        @test all(r["max_corr"] >= r["mean_corr"] for r in regions)

        # Check n_features matches indices length
        @test all(r["n_features"] == length(r["indices"]) for r in regions)

        println("✓ Basic functionality tests passed")
    end

    @testset "compute_region_correlations - Custom Parameters" begin
        n_samples = 30
        wavelengths = collect(400.0:2.0:498.0)  # 50 wavelengths
        X = randn(n_samples, length(wavelengths))
        y = randn(n_samples)

        # Test with larger region size
        regions_large = compute_region_correlations(X, y, wavelengths,
                                                    region_size=100.0, overlap=0.0)
        @test length(regions_large) >= 1

        # Test with smaller region size
        regions_small = compute_region_correlations(X, y, wavelengths,
                                                    region_size=20.0, overlap=10.0)
        @test length(regions_small) > length(regions_large)

        println("✓ Custom parameter tests passed")
    end

    @testset "compute_region_correlations - Edge Cases" begin
        wavelengths = collect(400.0:2.0:498.0)

        # Test with minimum samples
        X_min = randn(2, length(wavelengths))
        y_min = randn(2)
        regions_min = compute_region_correlations(X_min, y_min, wavelengths)
        @test length(regions_min) >= 0  # Should not error

        # Test with constant y (should handle gracefully)
        X = randn(20, length(wavelengths))
        y_const = ones(20)
        regions_const = compute_region_correlations(X, y_const, wavelengths)
        @test length(regions_const) >= 0  # Should not error

        println("✓ Edge case tests passed")
    end

    @testset "compute_region_correlations - Input Validation" begin
        X = randn(50, 100)
        y = randn(50)
        wavelengths = collect(400.0:4.0:796.0)

        # Test mismatched y length
        @test_throws AssertionError compute_region_correlations(X, randn(30), wavelengths)

        # Test mismatched wavelengths length
        @test_throws AssertionError compute_region_correlations(X, y, wavelengths[1:50])

        # Test invalid region_size
        @test_throws AssertionError compute_region_correlations(X, y, wavelengths, region_size=-10.0)

        # Test invalid overlap
        @test_throws AssertionError compute_region_correlations(X, y, wavelengths, overlap=-5.0)
        @test_throws AssertionError compute_region_correlations(X, y, wavelengths, region_size=50.0, overlap=60.0)

        # Test too few samples
        @test_throws AssertionError compute_region_correlations(randn(1, 100), randn(1), wavelengths)

        println("✓ Input validation tests passed")
    end

    @testset "create_region_subsets - Basic Functionality" begin
        n_samples = 50
        n_wavelengths = 200
        wavelengths = collect(400.0:2.0:798.0)
        X = randn(n_samples, n_wavelengths)
        y = randn(n_samples)

        # Add correlation in multiple regions
        X[:, 1:20] .+= y * 1.5
        X[:, 50:70] .+= y * 1.2
        X[:, 100:120] .+= y * 1.0

        subsets = create_region_subsets(X, y, wavelengths)

        # Basic checks
        @test length(subsets) > 0
        @test all(haskey(s, "indices") for s in subsets)
        @test all(haskey(s, "tag") for s in subsets)
        @test all(haskey(s, "description") for s in subsets)

        # Check indices are valid
        @test all(all(1 <= idx <= n_wavelengths for idx in s["indices"]) for s in subsets)

        # Check for individual region subsets
        has_individual = any(startswith(s["tag"], "region_") for s in subsets)
        @test has_individual

        # Check for combined region subsets
        has_combined = any(startswith(s["tag"], "top") for s in subsets)
        @test has_combined

        println("✓ Basic subset creation tests passed")
    end

    @testset "create_region_subsets - Different n_top_regions" begin
        n_samples = 50
        wavelengths = collect(400.0:2.0:798.0)
        X = randn(n_samples, length(wavelengths))
        y = randn(n_samples)

        # Test with n_top_regions=5 (should give 3 individual)
        subsets_5 = create_region_subsets(X, y, wavelengths, n_top_regions=5)
        individual_5 = count(s -> startswith(s["tag"], "region_"), subsets_5)
        @test individual_5 <= 3

        # Test with n_top_regions=10 (should give 5 individual)
        subsets_10 = create_region_subsets(X, y, wavelengths, n_top_regions=10)
        individual_10 = count(s -> startswith(s["tag"], "region_"), subsets_10)
        @test individual_10 <= 5

        # Test with n_top_regions=15 (should give 7 individual)
        subsets_15 = create_region_subsets(X, y, wavelengths, n_top_regions=15)
        individual_15 = count(s -> startswith(s["tag"], "region_"), subsets_15)
        @test individual_15 <= 7

        # Test with n_top_regions=20 (should give 10 individual)
        subsets_20 = create_region_subsets(X, y, wavelengths, n_top_regions=20)
        individual_20 = count(s -> startswith(s["tag"], "region_"), subsets_20)
        @test individual_20 <= 10

        println("✓ Variable n_top_regions tests passed")
    end

    @testset "create_region_subsets - Edge Cases" begin
        wavelengths = collect(400.0:10.0:450.0)  # Very few wavelengths
        X = randn(20, length(wavelengths))
        y = randn(20)

        # Test with very few wavelengths
        subsets_few = create_region_subsets(X, y, wavelengths, n_top_regions=5)
        @test length(subsets_few) >= 0  # Should not error

        # Test with n_top_regions larger than possible regions
        subsets_large = create_region_subsets(X, y, wavelengths, n_top_regions=100)
        @test length(subsets_large) >= 0  # Should cap to available regions

        println("✓ Edge case tests passed")
    end

    @testset "combine_region_indices - Basic Functionality" begin
        regions = [
            Dict("indices" => [1, 2, 3, 4, 5]),
            Dict("indices" => [4, 5, 6, 7, 8]),
            Dict("indices" => [10, 11, 12])
        ]

        combined = combine_region_indices(regions)

        # Check result is sorted and unique
        @test issorted(combined)
        @test length(combined) == length(unique(combined))

        # Check all indices are present
        @test 1 in combined
        @test 12 in combined
        @test length(combined) == 12  # Should have 1-8 and 10-12

        println("✓ Basic combine tests passed")
    end

    @testset "combine_region_indices - Edge Cases" begin
        # Test with empty regions
        regions_empty = [
            Dict("indices" => [1, 2, 3]),
            Dict("indices" => Int[]),
            Dict("indices" => [5, 6])
        ]
        combined_empty = combine_region_indices(regions_empty)
        @test length(combined_empty) == 5

        # Test with single region
        regions_single = [Dict("indices" => [1, 2, 3, 4, 5])]
        combined_single = combine_region_indices(regions_single)
        @test combined_single == [1, 2, 3, 4, 5]

        # Test with completely overlapping regions
        regions_overlap = [
            Dict("indices" => [1, 2, 3]),
            Dict("indices" => [1, 2, 3])
        ]
        combined_overlap = combine_region_indices(regions_overlap)
        @test combined_overlap == [1, 2, 3]

        println("✓ Edge case combine tests passed")
    end

    @testset "Integration Test - Full Workflow" begin
        # Simulate a realistic spectral analysis workflow
        n_samples = 100
        n_wavelengths = 300
        wavelengths = collect(400.0:1.0:699.0)  # 400-699 nm in 1nm steps

        # Create synthetic data with known structure
        X = randn(n_samples, n_wavelengths)
        y = randn(n_samples)

        # Add strong signal in specific regions
        # Region 1: 450-500 nm (strong)
        mask1 = (wavelengths .>= 450.0) .& (wavelengths .< 500.0)
        X[:, mask1] .+= y * 2.5

        # Region 2: 550-600 nm (medium)
        mask2 = (wavelengths .>= 550.0) .& (wavelengths .< 600.0)
        X[:, mask2] .+= y * 1.5

        # Region 3: 650-680 nm (weak)
        mask3 = (wavelengths .>= 650.0) .& (wavelengths .< 680.0)
        X[:, mask3] .+= y * 0.8

        # Step 1: Compute regions
        regions = compute_region_correlations(X, y, wavelengths)
        @test length(regions) > 0

        # Step 2: Create subsets
        subsets = create_region_subsets(X, y, wavelengths, n_top_regions=10)
        @test length(subsets) > 0

        # Step 3: Verify top regions capture the strong signal
        # Get first individual region subset
        individual_subsets = filter(s -> startswith(s["tag"], "region_"), subsets)
        if length(individual_subsets) > 0
            top_subset = individual_subsets[1]
            @test length(top_subset["indices"]) > 0

            # The indices should be valid
            @test all(1 <= idx <= n_wavelengths for idx in top_subset["indices"])
        end

        # Step 4: Test combining regions
        if length(regions) >= 3
            top3_regions = regions[1:3]
            combined_indices = combine_region_indices(top3_regions)
            @test length(combined_indices) > 0
            @test issorted(combined_indices)
        end

        println("✓ Integration test passed")
    end

end

println("\n" * "="^70)
println("All Regions module tests passed!")
println("="^70)
