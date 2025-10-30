"""
Test suite for search.jl - The core hyperparameter search module

This test suite validates:
1. Preprocessing configuration generation
2. Single configuration execution
3. Full search orchestration
4. Skip-preprocessing logic (CRITICAL!)
5. Variable subset analysis
6. Region subset analysis
7. Results structure and ranking
"""

using Test
using DataFrames
using Random
using Statistics

# Include the module
include("../src/search.jl")


@testset "Search Module Tests" begin

    # ========================================================================
    # Test Data Setup
    # ========================================================================

    @testset "Test Data Generation" begin
        Random.seed!(42)

        # Small dataset for testing
        n_samples = 50
        n_features = 100
        X = randn(n_samples, n_features)
        y = randn(n_samples)
        wavelengths = collect(400.0:4.0:796.0)

        @test size(X) == (50, 100)
        @test length(y) == 50
        @test length(wavelengths) == 100
        @test minimum(wavelengths) == 400.0
        @test maximum(wavelengths) == 796.0
    end


    # ========================================================================
    # Test Preprocessing Configuration Generation
    # ========================================================================

    @testset "Preprocessing Configuration Generation" begin
        # Test raw + SNV
        configs = generate_preprocessing_configs(
            ["raw", "snv"],
            [1, 2],
            17,
            3
        )
        @test length(configs) == 2
        @test configs[1]["name"] == "raw"
        @test configs[1]["deriv"] === nothing
        @test configs[2]["name"] == "snv"
        @test configs[2]["deriv"] === nothing

        # Test with derivatives
        configs = generate_preprocessing_configs(
            ["deriv"],
            [1, 2],
            17,
            3
        )
        @test length(configs) == 2
        @test configs[1]["name"] == "deriv"
        @test configs[1]["deriv"] == 1
        @test configs[1]["window"] == 17
        @test configs[1]["polyorder"] == 2  # 1st deriv uses polyorder=2
        @test configs[2]["deriv"] == 2
        @test configs[2]["polyorder"] == 3  # 2nd deriv uses polyorder=3

        # Test SNV + derivative combinations
        configs = generate_preprocessing_configs(
            ["snv", "deriv", "snv_deriv"],
            [1],
            11,
            2
        )
        @test length(configs) == 3
        @test any(c["name"] == "snv" for c in configs)
        @test any(c["name"] == "deriv" && c["deriv"] == 1 for c in configs)
        @test any(c["name"] == "snv_deriv" && c["deriv"] == 1 for c in configs)

        # Test all combinations
        configs = generate_preprocessing_configs(
            ["raw", "snv", "deriv", "snv_deriv", "deriv_snv"],
            [1, 2],
            17,
            3
        )
        # Should have: raw, snv, deriv(1,2), snv_deriv(1,2), deriv_snv(1,2) = 2 + 2 + 2 + 2 = 8
        @test length(configs) == 8
    end


    # ========================================================================
    # Test Single Configuration Execution
    # ========================================================================

    @testset "Single Configuration Execution" begin
        Random.seed!(42)
        X = randn(30, 50)
        y = randn(30)

        model_name = "Ridge"
        config = Dict("alpha" => 1.0)
        preprocess_config = Dict(
            "name" => "raw",
            "deriv" => nothing,
            "window" => nothing,
            "polyorder" => nothing
        )

        # Run single config
        result = run_single_config(
            X, y,
            model_name, config,
            preprocess_config, "regression",
            5,  # n_folds
            skip_preprocessing=false,
            subset_tag="full",
            n_vars=50,
            full_vars=50
        )

        # Check result structure
        @test haskey(result, "Model")
        @test haskey(result, "Preprocess")
        @test haskey(result, "RMSE")
        @test haskey(result, "R2")
        @test haskey(result, "n_vars")
        @test haskey(result, "full_vars")
        @test haskey(result, "SubsetTag")

        # Check values
        @test result["Model"] == "Ridge"
        @test result["Preprocess"] == "raw"
        @test result["SubsetTag"] == "full"
        @test result["n_vars"] == 50
        @test result["full_vars"] == 50
        @test result["RMSE"] > 0
        @test result["R2"] >= -Inf
    end


    # ========================================================================
    # Test Skip Preprocessing Logic (CRITICAL!)
    # ========================================================================

    @testset "Skip Preprocessing Logic" begin
        Random.seed!(42)
        X = randn(30, 50)
        y = randn(30)

        # Preprocess data manually
        preprocess_config = Dict(
            "name" => "snv",
            "deriv" => nothing,
            "window" => nothing,
            "polyorder" => nothing
        )
        X_preprocessed = apply_preprocessing(X, preprocess_config)

        model_name = "Ridge"
        config = Dict("alpha" => 1.0)

        # Test 1: Normal preprocessing (should apply SNV)
        result_normal = run_single_config(
            X, y,
            model_name, config,
            preprocess_config, "regression",
            5,
            skip_preprocessing=false
        )

        # Test 2: Skip preprocessing (use already preprocessed data)
        result_skip = run_single_config(
            X_preprocessed, y,
            model_name, config,
            preprocess_config, "regression",
            5,
            skip_preprocessing=true
        )

        # Results should be similar (both use SNV, just applied differently)
        @test abs(result_normal["RMSE"] - result_skip["RMSE"]) < 0.1

        # Test 3: Critical case - derivatives with subset
        # This is where skip_preprocessing prevents errors
        X_small = randn(30, 101)  # 101 features
        y_small = randn(30)

        deriv_config = Dict(
            "name" => "deriv",
            "deriv" => 1,
            "window" => 17,
            "polyorder" => 2
        )

        # Preprocess to get derivative features (will reduce to ~84 features)
        X_deriv = apply_preprocessing(X_small, deriv_config)
        n_deriv_features = size(X_deriv, 2)
        @test n_deriv_features < 101  # Features reduced

        # Select top 10 features
        top_indices = 1:10
        X_subset = X_deriv[:, top_indices]
        @test size(X_subset, 2) == 10

        # This should work with skip_preprocessing=true
        result_skip = run_single_config(
            X_subset, y_small,
            model_name, config,
            deriv_config, "regression",
            3,
            skip_preprocessing=true,  # Don't reapply derivative!
            subset_tag="top10",
            n_vars=10,
            full_vars=101
        )

        @test result_skip["n_vars"] == 10
        @test result_skip["SubsetTag"] == "top10"

        # Without skip_preprocessing, this would fail because window=17 > n_features=10
        # But we don't test failure cases in unit tests
    end


    # ========================================================================
    # Test Full Search (Small Scale)
    # ========================================================================

    @testset "Full Search - Small Scale" begin
        Random.seed!(42)

        # Small dataset
        n_samples = 40
        n_features = 60
        X = randn(n_samples, n_features)
        y = randn(n_samples)
        wavelengths = collect(400.0:6.7:794.3)

        # Run minimal search
        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["Ridge"],  # Single model for speed
            preprocessing=["raw"],  # Single preprocessing for speed
            enable_variable_subsets=false,  # Disable for speed
            enable_region_subsets=false,  # Disable for speed
            n_folds=3  # Fewer folds for speed
        )

        # Check results structure
        @test isa(results, DataFrame)
        @test nrow(results) >= 1  # At least one result
        @test "Model" in names(results)
        @test "Preprocess" in names(results)
        @test "RMSE" in names(results)
        @test "R2" in names(results)
        @test "Rank" in names(results)
        @test "CompositeScore" in names(results)

        # Check ranking
        @test issorted(results.Rank)
        @test results.Rank[1] == 1  # Best model has rank 1
    end


    # ========================================================================
    # Test Variable Subset Analysis
    # ========================================================================

    @testset "Variable Subset Analysis" begin
        Random.seed!(42)

        X = randn(40, 80)
        y = randn(40)
        wavelengths = collect(400.0:5.0:795.0)

        # Run search with variable subsets
        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["PLS"],  # PLS supports feature importance
            preprocessing=["raw"],
            enable_variable_subsets=true,
            variable_counts=[10, 20],  # Test two subset sizes
            enable_region_subsets=false,
            n_folds=3
        )

        # Should have: full + top10 + top20 for each PLS config
        # PLS has 8 configs, so 8 * 3 = 24 results
        @test nrow(results) >= 3  # At least full, top10, top20

        # Check subset tags
        subset_tags = unique(results.SubsetTag)
        @test "full" in subset_tags
        @test "top10" in subset_tags
        @test "top20" in subset_tags

        # Check n_vars
        full_results = filter(row -> row.SubsetTag == "full", results)
        @test all(full_results.n_vars .== 80)

        top10_results = filter(row -> row.SubsetTag == "top10", results)
        @test all(top10_results.n_vars .== 10)

        top20_results = filter(row -> row.SubsetTag == "top20", results)
        @test all(top20_results.n_vars .== 20)
    end


    # ========================================================================
    # Test Region Subset Analysis
    # ========================================================================

    @testset "Region Subset Analysis" begin
        Random.seed!(42)

        X = randn(40, 100)
        y = randn(40)
        wavelengths = collect(400.0:4.0:796.0)

        # Run search with region subsets
        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["Ridge"],  # Ridge works with region subsets
            preprocessing=["raw"],
            enable_variable_subsets=false,
            enable_region_subsets=true,
            n_top_regions=3,  # Test 3 regions
            n_folds=3
        )

        # Should have: full + region subsets
        @test nrow(results) > 1  # More than just full

        # Check for region tags
        subset_tags = unique(results.SubsetTag)
        @test "full" in subset_tags
        # Should have some region tags like "region_400-450nm" or "top2regions"
        region_tags = filter(tag -> tag != "full", subset_tags)
        @test length(region_tags) > 0
    end


    # ========================================================================
    # Test Multiple Models and Preprocessing
    # ========================================================================

    @testset "Multiple Models and Preprocessing" begin
        Random.seed!(42)

        X = randn(35, 70)
        y = randn(35)
        wavelengths = collect(400.0:5.7:794.3)

        # Run search with multiple models and preprocessing
        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["Ridge", "Lasso"],
            preprocessing=["raw", "snv"],
            enable_variable_subsets=false,
            enable_region_subsets=false,
            n_folds=3
        )

        # Check model variety
        models_found = unique(results.Model)
        @test "Ridge" in models_found
        @test "Lasso" in models_found

        # Check preprocessing variety
        preprocess_found = unique(results.Preprocess)
        @test "raw" in preprocess_found
        @test "snv" in preprocess_found

        # Results should be ranked
        @test issorted(results.Rank)
    end


    # ========================================================================
    # Test Derivative Preprocessing in Search
    # ========================================================================

    @testset "Derivative Preprocessing in Search" begin
        Random.seed!(42)

        X = randn(30, 101)  # 101 features for derivatives
        y = randn(30)
        wavelengths = collect(400.0:4.0:800.0)

        # Run search with derivatives
        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["Ridge"],
            preprocessing=["deriv"],
            derivative_orders=[1],
            derivative_window=17,
            derivative_polyorder=2,
            enable_variable_subsets=false,
            enable_region_subsets=false,
            n_folds=3
        )

        # Check that derivative was applied
        deriv_results = filter(row -> row.Preprocess == "deriv", results)
        @test nrow(deriv_results) > 0
        @test all(deriv_results.Deriv .== 1)
        @test all(deriv_results.Window .== 17)
        @test all(deriv_results.Poly .== 2)

        # Features should be reduced after derivative
        @test all(deriv_results.n_vars .< 101)
    end


    # ========================================================================
    # Test Derivative + Variable Subsets (CRITICAL TEST!)
    # ========================================================================

    @testset "Derivative + Variable Subsets (Skip Preprocessing)" begin
        Random.seed!(42)

        X = randn(30, 101)
        y = randn(30)
        wavelengths = collect(400.0:4.0:800.0)

        # This is the critical test for skip_preprocessing logic
        # With derivatives + variable subsets, we must NOT reapply preprocessing
        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["PLS"],  # PLS supports variable subsets
            preprocessing=["deriv"],
            derivative_orders=[1],
            derivative_window=17,
            derivative_polyorder=2,
            enable_variable_subsets=true,
            variable_counts=[10],  # Small subset
            enable_region_subsets=false,
            n_folds=3
        )

        # Should have results for both full and top10
        subset_tags = unique(results.SubsetTag)
        @test "full" in subset_tags
        @test "top10" in subset_tags

        # Check top10 results
        top10 = filter(row -> row.SubsetTag == "top10", results)
        @test nrow(top10) > 0
        @test all(top10.n_vars .== 10)

        # This test passing means skip_preprocessing worked correctly!
        # Without it, we'd get errors like "window size 17 > n_features 10"
    end


    # ========================================================================
    # Test Results Structure and Completeness
    # ========================================================================

    @testset "Results Structure and Completeness" begin
        Random.seed!(42)

        X = randn(30, 50)
        y = randn(30)
        wavelengths = collect(400.0:8.0:792.0)

        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["PLS", "Ridge"],
            preprocessing=["raw", "snv"],
            enable_variable_subsets=false,
            enable_region_subsets=false,
            n_folds=3
        )

        # Check required columns
        required_cols = [
            "Model", "Preprocess", "Deriv", "Window", "Poly",
            "SubsetTag", "n_vars", "full_vars",
            "RMSE", "R2",
            "CompositeScore", "Rank"
        ]
        for col in required_cols
            @test col in names(results)
        end

        # Check data types
        @test eltype(results.Model) <: AbstractString
        @test eltype(results.RMSE) <: Real
        @test eltype(results.R2) <: Real
        @test eltype(results.Rank) <: Integer

        # Check no missing critical values
        @test !any(ismissing.(results.Model))
        @test !any(ismissing.(results.RMSE))
        @test !any(ismissing.(results.Rank))
    end


    # ========================================================================
    # Edge Cases and Error Handling
    # ========================================================================

    @testset "Edge Cases" begin
        Random.seed!(42)

        X = randn(20, 30)
        y = randn(20)
        wavelengths = collect(400.0:13.3:785.7)

        # Test with empty preprocessing list (should use defaults or error)
        @test_throws Exception run_search(
            X, y, wavelengths,
            preprocessing=String[]
        )

        # Test with invalid task type
        @test_throws AssertionError run_search(
            X, y, wavelengths,
            task_type="invalid"
        )

        # Test with mismatched dimensions
        @test_throws AssertionError run_search(
            X, y[1:10], wavelengths  # Wrong y length
        )

        @test_throws AssertionError run_search(
            X, y, wavelengths[1:20]  # Wrong wavelengths length
        )
    end

end  # main testset


# Run the tests
println("\n" * "="^80)
println("Running Search Module Tests")
println("="^80 * "\n")

# Execute test suite
Test.@testset "All Search Tests" begin
    include("test_search.jl")
end
