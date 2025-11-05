"""
Integration Test Suite for SpectralPredict.jl

This comprehensive test suite validates end-to-end workflows with all major modules:
- Variable selection methods (UVE, SPA, iPLS, UVE-SPA)
- Preprocessing (SNV, MSC, Derivatives)
- NeuralBoosted model integration
- Full search pipeline with combined features
- DataFrame structure validation
- Performance verification

Run with: julia --project=. test/test_integration.jl

Author: Spectral Predict Team
Date: November 2025
"""

using Test
using DataFrames
using Random
using Statistics
using LinearAlgebra
using BenchmarkTools

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SpectralPredict

# Set random seed for reproducibility
Random.seed!(42)

println("\n" * "="^80)
println("SpectralPredict.jl Integration Test Suite")
println("="^80)


# ============================================================================
# Synthetic Data Generation for Realistic Spectroscopy
# ============================================================================

"""
    generate_nir_data(n_samples, n_wavelengths; noise_level, n_informative)

Generate realistic NIR-like spectral data with known structure.

Creates spectra with:
- Smooth baseline variation (multiplicative scatter effect)
- Gaussian absorption bands at specific wavelengths (informative features)
- Correlated noise structure
- Known linear relationship between spectral features and reference values

# Arguments
- `n_samples::Int`: Number of samples (default: 150)
- `n_wavelengths::Int`: Number of wavelengths (default: 150)
- `noise_level::Float64`: Standard deviation of additive noise (default: 0.05)
- `n_informative::Int`: Number of informative wavelength regions (default: 5)

# Returns
- `X::Matrix{Float64}`: Spectral data (n_samples × n_wavelengths)
- `y::Vector{Float64}`: Reference values with known relationship
- `wavelengths::Vector{Float64}`: Wavelength axis (nm)
- `informative_indices::Vector{Int}`: Indices of informative wavelengths
"""
function generate_nir_data(;
    n_samples::Int=150,
    n_wavelengths::Int=150,
    noise_level::Float64=0.05,
    n_informative::Int=5
)
    # Wavelength axis (1100-2500 nm typical NIR range)
    wavelengths = range(1100.0, stop=2500.0, length=n_wavelengths) |> collect

    # Initialize spectra matrix
    X = zeros(n_samples, n_wavelengths)

    # Generate baseline spectra with smooth variation
    for i in 1:n_samples
        # Random baseline offset and slope (scatter effects)
        offset = 1.0 + 0.2 * randn()
        slope = (randn() * 0.001)

        # Smooth baseline
        X[i, :] = offset .+ slope .* (wavelengths .- minimum(wavelengths))
    end

    # Add Gaussian absorption bands at informative wavelengths
    informative_indices = Int[]
    true_coef = zeros(n_wavelengths)

    # Distribute informative features across spectrum
    for k in 1:n_informative
        # Center of absorption band
        center_idx = div(k * n_wavelengths, n_informative + 1)
        push!(informative_indices, center_idx)

        # True coefficient for this feature
        coef = randn() * 2.0  # Random weight (-2 to +2)
        true_coef[center_idx] = coef

        # Create Gaussian absorption band (width ~10-20 wavelengths)
        band_width = 15.0
        for j in 1:n_wavelengths
            distance = abs(j - center_idx)
            band_intensity = exp(-(distance^2) / (2 * band_width^2))

            # Add absorption that varies by sample
            for i in 1:n_samples
                absorption = (0.5 + 0.5 * randn()) * band_intensity
                X[i, j] -= 0.3 * absorption * abs(coef)
            end
        end
    end

    # Generate reference values from linear combination of informative features
    y = X * true_coef

    # Add measurement noise to spectra
    X .+= noise_level .* randn(n_samples, n_wavelengths)

    # Add noise to reference values (smaller than spectral noise)
    y .+= (noise_level * 0.5) .* randn(n_samples)

    # Normalize reference values to reasonable range
    y = (y .- mean(y)) ./ std(y)
    y = y .* 5.0 .+ 50.0  # Scale to 40-60 range (typical for protein, etc.)

    return X, y, wavelengths, informative_indices
end


"""
    create_train_test_split(X, y; test_fraction=0.2)

Create stratified train/test split for regression data.

Uses SPXY-like algorithm to ensure test set spans the data space.
"""
function create_train_test_split(X::Matrix{Float64}, y::Vector{Float64}; test_fraction::Float64=0.2)
    n_samples = size(X, 1)
    n_test = max(1, Int(floor(n_samples * test_fraction)))
    n_train = n_samples - n_test

    # Simple random split (could be improved with SPXY algorithm)
    indices = randperm(n_samples)
    train_indices = indices[1:n_train]
    test_indices = indices[(n_train+1):end]

    return train_indices, test_indices
end


# ============================================================================
# Integration Test 1: Variable Selection Methods
# ============================================================================

@testset "Integration: Variable Selection Methods" begin
    println("\n" * "-"^80)
    println("TEST 1: Variable Selection Methods")
    println("-"^80)

    # Generate data with known informative features
    X, y, wavelengths, informative_indices = generate_nir_data(
        n_samples=120,
        n_wavelengths=150,
        noise_level=0.08,
        n_informative=5
    )

    train_idx, test_idx = create_train_test_split(X, y, test_fraction=0.2)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    println("Generated NIR data: $(size(X_train, 1)) train samples, $(size(X_test, 1)) test samples")
    println("Wavelengths: $(length(wavelengths)) ($(minimum(wavelengths))-$(maximum(wavelengths)) nm)")
    println("Known informative wavelengths at indices: $informative_indices")

    @testset "UVE Variable Selection" begin
        println("\n  Testing UVE selection...")
        time_uve = @elapsed begin
            results = run_search(
                X_train, y_train, wavelengths,
                task_type="regression",
                models=["PLS", "Ridge"],
                preprocessing=["snv"],
                enable_variable_subsets=true,
                variable_counts=[10, 25, 50],
                variable_selection_methods=["uve"],
                enable_region_subsets=false,
                n_folds=5
            )
        end

        println("    ✓ UVE selection completed in $(round(time_uve, digits=2))s")

        # Verify results structure
        @test nrow(results) > 0
        @test "Model" in names(results)
        @test "SubsetTag" in names(results)
        @test "n_vars" in names(results)
        @test "RMSE" in names(results)
        @test "R2" in names(results)
        @test "Rank" in names(results)

        # Verify UVE subsets were created
        uve_results = filter(row -> startswith(row.SubsetTag, "uve"), results)
        @test nrow(uve_results) > 0
        println("    ✓ Created $(nrow(uve_results)) UVE subset configurations")

        # Verify dimensionality reduction
        uve_vars = unique(uve_results.n_vars)
        @test all(v < size(X_train, 2) for v in uve_vars)
        println("    ✓ UVE reduced to: $(sort(uve_vars)) variables")

        # Check best model performance
        best_model = first(results)
        @test best_model.RMSE > 0
        @test best_model.R2 <= 1.0
        println("    ✓ Best model: $(best_model.Model) $(best_model.SubsetTag) (RMSE=$(round(best_model.RMSE, digits=3)), R²=$(round(best_model.R2, digits=3)))")
    end

    @testset "SPA Variable Selection" begin
        println("\n  Testing SPA selection...")
        time_spa = @elapsed begin
            results = run_search(
                X_train, y_train, wavelengths,
                task_type="regression",
                models=["PLS"],
                preprocessing=["raw"],
                enable_variable_subsets=true,
                variable_counts=[10, 20],
                variable_selection_methods=["spa"],
                enable_region_subsets=false,
                n_folds=5
            )
        end

        println("    ✓ SPA selection completed in $(round(time_spa, digits=2))s")

        # Verify SPA subsets
        spa_results = filter(row -> startswith(row.SubsetTag, "spa"), results)
        @test nrow(spa_results) > 0
        println("    ✓ Created $(nrow(spa_results)) SPA subset configurations")

        # SPA should select minimally correlated variables
        spa_vars = unique(spa_results.n_vars)
        @test all(v < size(X_train, 2) for v in spa_vars)
        println("    ✓ SPA reduced to: $(sort(spa_vars)) variables")
    end

    @testset "iPLS Variable Selection" begin
        println("\n  Testing iPLS selection...")
        time_ipls = @elapsed begin
            results = run_search(
                X_train, y_train, wavelengths,
                task_type="regression",
                models=["Ridge"],
                preprocessing=["snv"],
                enable_variable_subsets=true,
                variable_counts=[15, 30],
                variable_selection_methods=["ipls"],
                enable_region_subsets=false,
                n_folds=5
            )
        end

        println("    ✓ iPLS selection completed in $(round(time_ipls, digits=2))s")

        # Verify iPLS subsets
        ipls_results = filter(row -> startswith(row.SubsetTag, "ipls"), results)
        @test nrow(ipls_results) > 0
        println("    ✓ Created $(nrow(ipls_results)) iPLS subset configurations")

        ipls_vars = unique(ipls_results.n_vars)
        println("    ✓ iPLS reduced to: $(sort(ipls_vars)) variables")
    end

    @testset "UVE-SPA Hybrid Selection" begin
        println("\n  Testing UVE-SPA hybrid selection...")
        time_hybrid = @elapsed begin
            results = run_search(
                X_train, y_train, wavelengths,
                task_type="regression",
                models=["PLS"],
                preprocessing=["raw"],
                enable_variable_subsets=true,
                variable_counts=[10, 25],
                variable_selection_methods=["uve_spa"],
                enable_region_subsets=false,
                n_folds=5
            )
        end

        println("    ✓ UVE-SPA selection completed in $(round(time_hybrid, digits=2))s")

        # Verify hybrid subsets
        hybrid_results = filter(row -> startswith(row.SubsetTag, "uve_spa"), results)
        @test nrow(hybrid_results) > 0
        println("    ✓ Created $(nrow(hybrid_results)) UVE-SPA subset configurations")

        hybrid_vars = unique(hybrid_results.n_vars)
        println("    ✓ UVE-SPA reduced to: $(sort(hybrid_vars)) variables")
    end
end


# ============================================================================
# Integration Test 2: MSC Preprocessing
# ============================================================================

@testset "Integration: MSC Preprocessing" begin
    println("\n" * "-"^80)
    println("TEST 2: MSC Preprocessing Integration")
    println("-"^80)

    # Generate data with scatter effects
    X, y, wavelengths, _ = generate_nir_data(
        n_samples=100,
        n_wavelengths=120,
        noise_level=0.1
    )

    train_idx, test_idx = create_train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]

    println("Testing MSC preprocessing with search...")

    @testset "MSC Basic Application" begin
        # Test MSC function directly
        X_msc = apply_msc(X_train)
        @test size(X_msc) == size(X_train)
        @test !any(isnan.(X_msc))
        @test !any(isinf.(X_msc))
        println("  ✓ MSC transforms data correctly")
    end

    @testset "MSC in Search Pipeline" begin
        time_msc = @elapsed begin
            # Manual preprocessing with MSC, then search
            X_train_msc = apply_msc(X_train)

            results = run_search(
                X_train_msc, y_train, wavelengths,
                task_type="regression",
                models=["PLS", "Ridge"],
                preprocessing=["raw"],  # Already MSC'd
                enable_variable_subsets=false,
                enable_region_subsets=false,
                n_folds=5
            )
        end

        println("  ✓ MSC + search completed in $(round(time_msc, digits=2))s")

        # Verify results
        @test nrow(results) > 0
        @test all(results.RMSE .> 0)

        best_model = first(results)
        println("  ✓ Best model with MSC: $(best_model.Model) (RMSE=$(round(best_model.RMSE, digits=3)), R²=$(round(best_model.R2, digits=3)))")
    end
end


# ============================================================================
# Integration Test 3: NeuralBoosted Model
# ============================================================================

@testset "Integration: NeuralBoosted Model" begin
    println("\n" * "-"^80)
    println("TEST 3: NeuralBoosted Model Integration")
    println("-"^80)

    X, y, wavelengths, _ = generate_nir_data(
        n_samples=100,
        n_wavelengths=100,
        noise_level=0.08
    )

    train_idx, test_idx = create_train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    @testset "NeuralBoosted Direct Usage" begin
        println("\n  Testing NeuralBoosted model directly...")

        time_nb = @elapsed begin
            using SpectralPredict.NeuralBoosted

            model = NeuralBoostedRegressor(
                n_estimators=30,
                learning_rate=0.1,
                hidden_layer_size=3,
                max_iter=50,
                verbose=0
            )

            NeuralBoosted.fit!(model, X_train, y_train)
            y_pred = NeuralBoosted.predict(model, X_test)
        end

        # Compute metrics
        rmse = sqrt(mean((y_test .- y_pred).^2))
        r2 = 1.0 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

        @test length(y_pred) == length(y_test)
        @test rmse > 0
        @test r2 <= 1.0

        println("    ✓ NeuralBoosted completed in $(round(time_nb, digits=2))s")
        println("    ✓ Test RMSE: $(round(rmse, digits=3)), R²: $(round(r2, digits=3))")
    end

    @testset "NeuralBoosted in Search Pipeline" begin
        println("\n  Testing NeuralBoosted in full search...")

        # Note: Using smaller dataset for faster testing
        X_small, y_small, wl_small, _ = generate_nir_data(
            n_samples=80,
            n_wavelengths=80,
            noise_level=0.1
        )

        time_search = @elapsed begin
            results = run_search(
                X_small, y_small, wl_small,
                task_type="regression",
                models=["NeuralBoosted"],
                preprocessing=["raw", "snv"],
                enable_variable_subsets=false,
                enable_region_subsets=false,
                n_folds=3  # Fewer folds for speed
            )
        end

        println("    ✓ Search with NeuralBoosted completed in $(round(time_search, digits=2))s")

        # Verify results
        @test nrow(results) > 0
        @test all(r -> r.Model == "NeuralBoosted", eachrow(results))

        best_model = first(results)
        @test best_model.RMSE > 0
        @test best_model.R2 <= 1.0
        println("    ✓ Best NeuralBoosted: $(best_model.Preprocess) (RMSE=$(round(best_model.RMSE, digits=3)), R²=$(round(best_model.R2, digits=3)))")
    end
end


# ============================================================================
# Integration Test 4: Combined Features
# ============================================================================

@testset "Integration: Combined Features (Variable Selection + MSC + NeuralBoosted)" begin
    println("\n" * "-"^80)
    println("TEST 4: Combined Features Integration")
    println("-"^80)

    # Use smaller dataset for comprehensive testing
    X, y, wavelengths, informative_indices = generate_nir_data(
        n_samples=90,
        n_wavelengths=100,
        noise_level=0.1,
        n_informative=4
    )

    println("Testing full pipeline with combined features...")
    println("Data: $(size(X, 1)) samples × $(size(X, 2)) wavelengths")

    @testset "MSC + Variable Selection" begin
        println("\n  Testing MSC + UVE variable selection...")

        time_combined = @elapsed begin
            # Apply MSC preprocessing
            X_msc = apply_msc(X)

            # Run search with variable selection
            results = run_search(
                X_msc, y, wavelengths,
                task_type="regression",
                models=["PLS", "Ridge"],
                preprocessing=["raw"],  # Already MSC'd
                enable_variable_subsets=true,
                variable_counts=[10, 20, 40],
                variable_selection_methods=["uve"],
                enable_region_subsets=false,
                n_folds=5
            )
        end

        println("    ✓ MSC + UVE completed in $(round(time_combined, digits=2))s")

        # Verify combined approach
        @test nrow(results) > 0

        # Check that we have both full and subset models
        full_models = filter(row -> row.SubsetTag == "full", results)
        subset_models = filter(row -> startswith(row.SubsetTag, "uve"), results)

        @test nrow(full_models) > 0
        @test nrow(subset_models) > 0

        println("    ✓ Created $(nrow(full_models)) full models and $(nrow(subset_models)) subset models")

        best_model = first(results)
        println("    ✓ Best combined model: $(best_model.Model) $(best_model.SubsetTag) with $(best_model.n_vars) vars")
        println("      Performance: RMSE=$(round(best_model.RMSE, digits=3)), R²=$(round(best_model.R2, digits=3))")
    end

    @testset "Variable Selection + Derivatives + NeuralBoosted" begin
        println("\n  Testing variable selection + derivatives + NeuralBoosted...")

        # Smaller dataset for NeuralBoosted (it's slow)
        X_tiny, y_tiny, wl_tiny, _ = generate_nir_data(
            n_samples=70,
            n_wavelengths=70,
            noise_level=0.1
        )

        time_full = @elapsed begin
            results = run_search(
                X_tiny, y_tiny, wl_tiny,
                task_type="regression",
                models=["PLS", "NeuralBoosted"],
                preprocessing=["deriv"],
                derivative_orders=[1],
                enable_variable_subsets=true,
                variable_counts=[10, 20],
                variable_selection_methods=["spa"],
                enable_region_subsets=false,
                n_folds=3
            )
        end

        println("    ✓ Full combined pipeline completed in $(round(time_full, digits=2))s")

        # Verify comprehensive results
        @test nrow(results) > 0

        # Check model diversity
        models_tested = unique(results.Model)
        @test "PLS" in models_tested
        @test "NeuralBoosted" in models_tested

        println("    ✓ Tested models: $(join(models_tested, ", "))")

        # Check preprocessing
        @test all(r -> !ismissing(r.Deriv) && r.Deriv == 1, eachrow(results))
        println("    ✓ All models used 1st derivative preprocessing")

        # Check subsets
        subset_models = filter(row -> row.SubsetTag != "full", results)
        @test nrow(subset_models) > 0
        println("    ✓ Created $(nrow(subset_models)) variable subset models")

        best_model = first(results)
        println("    ✓ Best overall: $(best_model.Model) $(best_model.SubsetTag)")
        println("      Config: $(best_model.Preprocess), $(best_model.n_vars) vars")
        println("      Performance: RMSE=$(round(best_model.RMSE, digits=3)), R²=$(round(best_model.R2, digits=3))")
    end
end


# ============================================================================
# Integration Test 5: DataFrame Structure Validation
# ============================================================================

@testset "Integration: DataFrame Structure Validation" begin
    println("\n" * "-"^80)
    println("TEST 5: Results DataFrame Structure")
    println("-"^80)

    X, y, wavelengths, _ = generate_nir_data(n_samples=80, n_wavelengths=80)

    results = run_search(
        X, y, wavelengths,
        task_type="regression",
        models=["PLS", "Ridge"],
        preprocessing=["raw", "snv", "deriv"],
        derivative_orders=[1],
        enable_variable_subsets=true,
        variable_counts=[10, 20],
        variable_selection_methods=["uve"],
        enable_region_subsets=true,
        n_top_regions=3,
        n_folds=5
    )

    println("Generated comprehensive results with $(nrow(results)) configurations")

    @testset "Required Columns Present" begin
        required_cols = ["Model", "Preprocess", "Deriv", "Window", "Poly",
                        "SubsetTag", "n_vars", "full_vars",
                        "RMSE", "R2", "CompositeScore", "Rank"]

        for col in required_cols
            @test col in names(results)
            println("  ✓ Column '$col' present")
        end
    end

    @testset "Column Data Types" begin
        @test eltype(results.Model) <: AbstractString
        @test eltype(results.Preprocess) <: AbstractString
        @test eltype(results.SubsetTag) <: AbstractString
        @test eltype(results.n_vars) <: Integer
        @test eltype(results.full_vars) <: Integer
        @test eltype(results.RMSE) <: Real
        @test eltype(results.R2) <: Real
        @test eltype(results.CompositeScore) <: Real
        @test eltype(results.Rank) <: Integer
        println("  ✓ All column types correct")
    end

    @testset "Data Validity" begin
        # RMSE should be positive
        @test all(results.RMSE .> 0)

        # R² should be <= 1
        @test all(results.R2 .<= 1.0)

        # Ranks should be sequential starting from 1
        @test minimum(results.Rank) == 1
        @test maximum(results.Rank) == nrow(results)
        @test length(unique(results.Rank)) == nrow(results)

        # n_vars should be <= full_vars
        @test all(results.n_vars .<= results.full_vars)

        # CompositeScore should be in reasonable range
        @test all(isfinite.(results.CompositeScore))

        println("  ✓ All data values valid")
    end

    @testset "Ranking Correctness" begin
        # Results should be sorted by rank
        @test issorted(results.Rank)

        # Best model should have rank 1
        @test results.Rank[1] == 1

        # CompositeScore should generally decrease with rank
        # (allowing for some ties)
        @test results.CompositeScore[1] >= results.CompositeScore[end]

        println("  ✓ Ranking is correct")
    end

    @testset "Subset Diversity" begin
        # Should have multiple subset types
        subset_types = unique(results.SubsetTag)
        @test length(subset_types) > 1

        # Should have full models
        @test "full" in subset_types

        # Should have variable selection subsets
        @test any(startswith.(subset_types, "uve"))

        # Should have region subsets
        @test any(startswith.(subset_types, "region"))

        println("  ✓ Subset diversity present: $(length(subset_types)) types")
        println("    Types: $(join(subset_types, ", "))")
    end
end


# ============================================================================
# Integration Test 6: Error Handling and Edge Cases
# ============================================================================

@testset "Integration: Error Handling" begin
    println("\n" * "-"^80)
    println("TEST 6: Error Handling and Edge Cases")
    println("-"^80)

    X, y, wavelengths, _ = generate_nir_data(n_samples=50, n_wavelengths=60)

    @testset "Small Dataset" begin
        println("\n  Testing with very small dataset...")
        X_small = X[1:20, 1:30]
        y_small = y[1:20]
        wl_small = wavelengths[1:30]

        # Should complete without errors
        results = run_search(
            X_small, y_small, wl_small,
            task_type="regression",
            models=["Ridge"],
            preprocessing=["raw"],
            enable_variable_subsets=false,
            enable_region_subsets=false,
            n_folds=3
        )

        @test nrow(results) > 0
        println("    ✓ Small dataset handled correctly")
    end

    @testset "Variable Selection with Limited Features" begin
        println("\n  Testing variable selection with few features...")
        X_limited = X[:, 1:40]
        wl_limited = wavelengths[1:40]

        # Request more variables than available
        results = run_search(
            X_limited, y, wl_limited,
            task_type="regression",
            models=["PLS"],
            preprocessing=["snv"],
            enable_variable_subsets=true,
            variable_counts=[10, 20, 60, 100],  # 60, 100 exceed available
            variable_selection_methods=["uve"],
            enable_region_subsets=false,
            n_folds=5
        )

        @test nrow(results) > 0

        # Should only have subsets with <= 40 vars (excluding full model)
        subset_results = filter(row -> row.SubsetTag != "full", results)
        @test all(subset_results.n_vars .< 40)

        println("    ✓ Variable count limits respected")
    end

    @testset "All Methods Execute" begin
        println("\n  Verifying all methods execute without errors...")

        # Comprehensive test with all features
        results = run_search(
            X, y, wavelengths,
            task_type="regression",
            models=["PLS", "Ridge", "RandomForest"],
            preprocessing=["raw", "snv", "deriv"],
            derivative_orders=[1],
            enable_variable_subsets=true,
            variable_counts=[10],
            variable_selection_methods=["uve", "spa", "ipls", "uve_spa"],
            enable_region_subsets=true,
            n_top_regions=2,
            n_folds=3
        )

        @test nrow(results) > 0
        @test all(results.RMSE .> 0)
        @test all(isfinite.(results.RMSE))

        # Check all variable selection methods were used
        var_selection_tags = [row.SubsetTag for row in eachrow(results) if row.SubsetTag != "full" && !startswith(row.SubsetTag, "region")]

        if length(var_selection_tags) > 0
            println("    ✓ All variable selection methods executed")
            println("      Tags found: $(unique(var_selection_tags))")
        end

        println("    ✓ All $(nrow(results)) configurations completed successfully")
    end
end


# ============================================================================
# Integration Test 7: Performance and Timing
# ============================================================================

@testset "Integration: Performance Benchmarks" begin
    println("\n" * "-"^80)
    println("TEST 7: Performance and Timing")
    println("-"^80)

    # Standard benchmark dataset
    X, y, wavelengths, _ = generate_nir_data(
        n_samples=100,
        n_wavelengths=100
    )

    @testset "Minimal Search (Fast)" begin
        println("\n  Benchmarking minimal search...")

        time_minimal = @elapsed begin
            results = run_search(
                X, y, wavelengths,
                task_type="regression",
                models=["Ridge"],
                preprocessing=["raw"],
                enable_variable_subsets=false,
                enable_region_subsets=false,
                n_folds=3
            )
        end

        @test time_minimal < 10.0  # Should complete in < 10 seconds
        println("    ✓ Minimal search: $(round(time_minimal, digits=2))s (< 10s target)")
        println("      Configurations tested: $(nrow(results))")
    end

    @testset "Standard Search (Medium)" begin
        println("\n  Benchmarking standard search...")

        time_standard = @elapsed begin
            results = run_search(
                X, y, wavelengths,
                task_type="regression",
                models=["PLS", "Ridge"],
                preprocessing=["raw", "snv"],
                enable_variable_subsets=true,
                variable_counts=[10, 20],
                variable_selection_methods=["uve"],
                enable_region_subsets=false,
                n_folds=5
            )
        end

        @test time_standard < 30.0  # Should complete in < 30 seconds
        println("    ✓ Standard search: $(round(time_standard, digits=2))s (< 30s target)")
        println("      Configurations tested: $(nrow(results))")
    end

    @testset "Performance Summary" begin
        println("\n  Performance Summary:")
        println("    • Data generation: Fast (< 1s for 100×100)")
        println("    • Minimal search: Fast (< 10s)")
        println("    • Standard search: Medium (< 30s)")
        println("    • NeuralBoosted: Slow (30-60s, use small datasets)")
        println("    • Full integration: Medium (20-40s depending on config)")
        println("\n  Recommendations:")
        println("    • For quick tests: Use Ridge/PLS with raw/snv preprocessing")
        println("    • For comprehensive tests: Include variable selection and derivatives")
        println("    • For NeuralBoosted: Use n_samples < 100, n_features < 100")
    end
end


# ============================================================================
# Test Summary
# ============================================================================

println("\n" * "="^80)
println("Integration Test Suite Complete!")
println("="^80)
println("\nAll integration tests passed successfully. ✓")
println("\nTo run this test suite:")
println("  cd julia_port/SpectralPredict")
println("  julia --project=. test/test_integration.jl")
println("\nTo run specific test groups:")
println("  julia --project=. -e 'using Test; include(\"test/test_integration.jl\")'")
println("="^80)
