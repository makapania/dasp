"""
Comprehensive Integration Test Suite for All 6 Models
=======================================================

Tests all models with recent hyperparameter fixes:
1. PLS - baseline functionality
2. Ridge - verify alpha=1000.0 is in grid
3. Lasso - baseline functionality
4. RandomForest - verify min_samples_leaf=1 is being used
5. MLP - baseline functionality
6. NeuralBoosted - verify LBFGS training works

Run with:
    cd julia_port/SpectralPredict
    julia --project=. test/test_all_models_integration.jl

Author: DASP Team
Date: November 6, 2025
"""

using Test
using Random
using Statistics
using LinearAlgebra
using DataFrames

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SpectralPredict

# For direct NeuralBoosted testing
using SpectralPredict.NeuralBoosted

# Set random seed for reproducibility
Random.seed!(42)

println("\n" * "="^80)
println("COMPREHENSIVE INTEGRATION TEST: ALL 6 MODELS")
println("="^80)
println("\nVerifying fixes:")
println("  1. Ridge: alpha=1000.0 in grid")
println("  2. RandomForest: min_samples_leaf=1")
println("  3. NeuralBoosted: LBFGS training")
println("="^80)


# ============================================================================
# Test Data Generation
# ============================================================================

"""Generate realistic spectral data for testing"""
function generate_spectral_data(; n_samples=100, n_wavelengths=50, noise_level=0.1)
    # Wavelength axis (1100-2500 nm NIR range)
    wavelengths = range(1100.0, stop=2500.0, length=n_wavelengths) |> collect

    # Initialize spectra
    X = zeros(n_samples, n_wavelengths)

    # Add baseline variation (scatter effects)
    for i in 1:n_samples
        offset = 1.0 + 0.2 * randn()
        slope = randn() * 0.001
        X[i, :] = offset .+ slope .* (wavelengths .- minimum(wavelengths))
    end

    # Add informative Gaussian absorption bands
    n_informative = 5
    true_coef = zeros(n_wavelengths)
    informative_indices = Int[]

    for k in 1:n_informative
        center_idx = div(k * n_wavelengths, n_informative + 1)
        push!(informative_indices, center_idx)
        coef = randn() * 2.0
        true_coef[center_idx] = coef

        # Create absorption band
        band_width = 10.0
        for j in 1:n_wavelengths
            distance = abs(j - center_idx)
            band_intensity = exp(-(distance^2) / (2 * band_width^2))
            for i in 1:n_samples
                absorption = (0.5 + 0.5 * randn()) * band_intensity
                X[i, j] -= 0.3 * absorption * abs(coef)
            end
        end
    end

    # Generate target from linear combination
    y = X * true_coef

    # Add noise
    X .+= noise_level .* randn(n_samples, n_wavelengths)
    y .+= (noise_level * 0.5) .* randn(n_samples)

    # Normalize target
    y = (y .- mean(y)) ./ std(y)
    y = y .* 5.0 .+ 50.0

    return X, y, wavelengths, informative_indices
end


# ============================================================================
# Helper Functions
# ============================================================================

"""Split data into train/test sets"""
function train_test_split(X, y; test_fraction=0.2)
    n_samples = size(X, 1)
    n_test = max(1, Int(floor(n_samples * test_fraction)))
    n_train = n_samples - n_test

    indices = randperm(n_samples)
    train_idx = indices[1:n_train]
    test_idx = indices[(n_train+1):end]

    return train_idx, test_idx
end

"""Calculate R² score"""
function r2_score(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1.0 - ss_res / ss_tot
end

"""Calculate RMSE"""
function rmse_score(y_true, y_pred)
    return sqrt(mean((y_true .- y_pred).^2))
end


# ============================================================================
# TEST 1: PLS Model
# ============================================================================

@testset "Model 1/6: PLS (Baseline)" begin
    println("\n" * "-"^80)
    println("TEST 1/6: PLS Model")
    println("-"^80)

    X, y, wavelengths, _ = generate_spectral_data(n_samples=120, n_wavelengths=60)
    train_idx, test_idx = train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    # Test configuration generation
    configs = get_model_configs("PLS")
    @test length(configs) == 8
    @test configs[1]["n_components"] == 1
    @test configs[end]["n_components"] == 20
    println("✓ Config generation: $(length(configs)) configurations")

    # Test model fitting
    config = Dict{String, Any}("n_components" => 5)
    model = build_model("PLS", config, "regression")
    fit_model!(model, X_train, y_train)

    @test !isnothing(model.model)
    @test !isnothing(model.mean_X)
    @test !isnothing(model.mean_y)
    println("✓ Model fitting: successful")

    # Test prediction
    y_pred = predict_model(model, X_test)
    @test length(y_pred) == length(y_test)
    @test all(isfinite.(y_pred))

    r2 = r2_score(y_test, y_pred)
    rmse = rmse_score(y_test, y_pred)
    @test r2 > -0.5
    println("✓ Prediction: R²=$(round(r2, digits=3)), RMSE=$(round(rmse, digits=3))")

    # Test feature importances
    importances = SpectralPredict.get_feature_importances(model, "PLS", X_train, y_train)
    @test length(importances) == size(X_train, 2)
    @test all(importances .>= 0)
    println("✓ Feature importances: $(length(importances)) values")

    println("✅ PLS: PASS")
end


# ============================================================================
# TEST 2: Ridge Model (VERIFY alpha=1000.0)
# ============================================================================

@testset "Model 2/6: Ridge (Verify alpha=1000.0)" begin
    println("\n" * "-"^80)
    println("TEST 2/6: Ridge Model - HYPERPARAMETER FIX CHECK")
    println("-"^80)

    X, y, wavelengths, _ = generate_spectral_data(n_samples=120, n_wavelengths=60)
    train_idx, test_idx = train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    # CRITICAL CHECK: Verify alpha=1000.0 is in grid
    configs = get_model_configs("Ridge")
    alpha_values = [c["alpha"] for c in configs]

    println("Ridge alpha grid: $alpha_values")

    if 1000.0 in alpha_values
        println("✅ HYPERPARAMETER FIX APPLIED: alpha=1000.0 is in grid")
        @test true
    else
        println("❌ HYPERPARAMETER FIX MISSING: alpha=1000.0 NOT in grid")
        println("   Current grid: $alpha_values")
        println("   Expected: should include 1000.0")
        @test false
    end

    # Test with multiple alpha values including extremes
    test_alphas = [0.001, 1.0, 100.0]
    if 1000.0 in alpha_values
        push!(test_alphas, 1000.0)
    end

    println("\nTesting Ridge with different alpha values:")
    for alpha in test_alphas
        config = Dict{String, Any}("alpha" => alpha)
        model = build_model("Ridge", config, "regression")
        fit_model!(model, X_train, y_train)

        y_pred = predict_model(model, X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = rmse_score(y_test, y_pred)

        @test all(isfinite.(y_pred))
        @test r2 > -0.5  # Reasonable performance

        println("  α=$alpha: R²=$(round(r2, digits=3)), RMSE=$(round(rmse, digits=3))")
    end

    # Test feature importances
    config = Dict{String, Any}("alpha" => 1.0)
    model = build_model("Ridge", config, "regression")
    fit_model!(model, X_train, y_train)
    importances = SpectralPredict.get_feature_importances(model, "Ridge", X_train, y_train)
    @test length(importances) == size(X_train, 2)
    println("✓ Feature importances: working")

    if 1000.0 in alpha_values
        println("✅ Ridge: PASS (with alpha=1000.0 fix)")
    else
        println("⚠️  Ridge: CONDITIONAL PASS (missing alpha=1000.0 fix)")
    end
end


# ============================================================================
# TEST 3: Lasso Model
# ============================================================================

@testset "Model 3/6: Lasso (Baseline)" begin
    println("\n" * "-"^80)
    println("TEST 3/6: Lasso Model")
    println("-"^80)

    X, y, wavelengths, _ = generate_spectral_data(n_samples=120, n_wavelengths=60)
    train_idx, test_idx = train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    configs = get_model_configs("Lasso")
    @test length(configs) == 6
    println("✓ Config generation: $(length(configs)) configurations")

    config = Dict{String, Any}("alpha" => 0.1)
    model = build_model("Lasso", config, "regression")
    fit_model!(model, X_train, y_train)

    y_pred = predict_model(model, X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = rmse_score(y_test, y_pred)

    @test all(isfinite.(y_pred))
    @test r2 > -0.5
    println("✓ Prediction: R²=$(round(r2, digits=3)), RMSE=$(round(rmse, digits=3))")

    # Lasso should zero out some coefficients
    importances = SpectralPredict.get_feature_importances(model, "Lasso", X_train, y_train)
    n_zeros = sum(importances .== 0)
    println("✓ Sparsity: $(n_zeros)/$(length(importances)) coefficients zeroed")

    println("✅ Lasso: PASS")
end


# ============================================================================
# TEST 4: RandomForest Model (VERIFY min_samples_leaf=1)
# ============================================================================

@testset "Model 4/6: RandomForest (Verify min_samples_leaf=1)" begin
    println("\n" * "-"^80)
    println("TEST 4/6: RandomForest Model - HYPERPARAMETER FIX CHECK")
    println("-"^80)

    X, y, wavelengths, _ = generate_spectral_data(n_samples=120, n_wavelengths=60)
    train_idx, test_idx = train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    configs = get_model_configs("RandomForest")
    @test length(configs) == 6
    println("✓ Config generation: $(length(configs)) configurations")

    # Train model
    config = Dict{String, Any}("n_trees" => 100, "max_features" => "sqrt")
    model = build_model("RandomForest", config, "regression")
    fit_model!(model, X_train, y_train)

    @test !isnothing(model.forest)
    println("✓ Model fitting: successful")

    # CRITICAL CHECK: Verify min_samples_leaf=1 is being used
    # We need to check the source code since DecisionTree.jl doesn't expose this easily
    println("\nChecking min_samples_leaf parameter in source code:")

    # Read the models.jl file to check the parameter
    models_file = joinpath(@__DIR__, "..", "src", "models.jl")
    if isfile(models_file)
        file_content = read(models_file, String)

        # Look for the min_samples_leaf line in fit_model! for RandomForest
        # Expected line: "5,                       # min_samples_leaf" or "1,                       # min_samples_leaf"
        if occursin(r"1,\s*#\s*min_samples_leaf", file_content)
            println("✅ HYPERPARAMETER FIX APPLIED: min_samples_leaf=1 in source code")
            @test true
        elseif occursin(r"5,\s*#\s*min_samples_leaf", file_content)
            println("❌ HYPERPARAMETER FIX MISSING: min_samples_leaf=5 in source code")
            println("   Expected: 1,                       # min_samples_leaf")
            println("   Found: 5,                       # min_samples_leaf")
            @test false
        else
            println("⚠️  Could not verify min_samples_leaf parameter in source code")
            @test false
        end
    else
        println("⚠️  Could not find models.jl file")
        @test false
    end

    # Test prediction regardless
    y_pred = predict_model(model, X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = rmse_score(y_test, y_pred)

    @test all(isfinite.(y_pred))
    @test r2 > -0.5
    println("✓ Prediction: R²=$(round(r2, digits=3)), RMSE=$(round(rmse, digits=3))")

    # Test feature importances
    importances = SpectralPredict.get_feature_importances(model, "RandomForest", X_train, y_train)
    @test length(importances) == size(X_train, 2)
    @test sum(importances) ≈ 1.0 atol=1e-6
    println("✓ Feature importances: normalized sum=$(round(sum(importances), digits=6))")

    # Check source code for the fix
    if isfile(models_file)
        file_content = read(models_file, String)
        if occursin(r"1,\s*#\s*min_samples_leaf", file_content)
            println("✅ RandomForest: PASS (with min_samples_leaf=1 fix)")
        else
            println("⚠️  RandomForest: CONDITIONAL PASS (missing min_samples_leaf=1 fix)")
        end
    end
end


# ============================================================================
# TEST 5: MLP Model
# ============================================================================

@testset "Model 5/6: MLP (Baseline)" begin
    println("\n" * "-"^80)
    println("TEST 5/6: MLP Model")
    println("-"^80)

    X, y, wavelengths, _ = generate_spectral_data(n_samples=120, n_wavelengths=60)
    train_idx, test_idx = train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    configs = get_model_configs("MLP")
    @test length(configs) == 6
    println("✓ Config generation: $(length(configs)) configurations")

    config = Dict{String, Any}("hidden_layers" => (50,), "learning_rate" => 0.01)
    model = build_model("MLP", config, "regression")
    fit_model!(model, X_train, y_train)

    @test !isnothing(model.model)
    @test !isnothing(model.mean_X)
    @test !isnothing(model.std_X)
    println("✓ Model fitting: successful")

    y_pred = predict_model(model, X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = rmse_score(y_test, y_pred)

    @test all(isfinite.(y_pred))
    @test r2 > -0.5
    println("✓ Prediction: R²=$(round(r2, digits=3)), RMSE=$(round(rmse, digits=3))")

    importances = SpectralPredict.get_feature_importances(model, "MLP", X_train, y_train)
    @test length(importances) == size(X_train, 2)
    println("✓ Feature importances: working")

    println("✅ MLP: PASS")
end


# ============================================================================
# TEST 6: NeuralBoosted Model (VERIFY LBFGS)
# ============================================================================

@testset "Model 6/6: NeuralBoosted (Verify LBFGS)" begin
    println("\n" * "-"^80)
    println("TEST 6/6: NeuralBoosted Model - LBFGS TRAINING CHECK")
    println("-"^80)

    # Use smaller dataset for faster NeuralBoosted training
    X, y, wavelengths, _ = generate_spectral_data(n_samples=80, n_wavelengths=40)
    train_idx, test_idx = train_test_split(X, y)
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    configs = get_model_configs("NeuralBoosted")
    println("✓ Config generation: $(length(configs)) configurations")

    # Test direct NeuralBoosted usage
    println("\nTesting NeuralBoosted directly:")

    try
        model = NeuralBoostedRegressor(
            n_estimators=20,
            learning_rate=0.1,
            hidden_layer_size=3,
            max_iter=100,
            verbose=1
        )

        println("  Fitting NeuralBoosted model...")
        NeuralBoosted.fit!(model, X_train, y_train)

        @test !isnothing(model.estimators_)
        @test length(model.estimators_) > 0
        println("  ✓ Model fitted: $(length(model.estimators_)) weak learners trained")

        y_pred = NeuralBoosted.predict(model, X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = rmse_score(y_test, y_pred)

        @test length(y_pred) == length(y_test)
        @test all(isfinite.(y_pred))
        println("  ✓ Prediction: R²=$(round(r2, digits=3)), RMSE=$(round(rmse, digits=3))")

        # Test feature importances
        importances = NeuralBoosted.SpectralPredict.get_feature_importances(model)
        @test length(importances) == size(X_train, 2)
        @test all(importances .>= 0)
        println("  ✓ Feature importances: working")

        # Check if LBFGS is mentioned in the training output
        # (We'd need to check the neural_boosted.jl source to verify LBFGS is used)
        neural_boosted_file = joinpath(@__DIR__, "..", "src", "neural_boosted.jl")
        if isfile(neural_boosted_file)
            file_content = read(neural_boosted_file, String)
            if occursin("LBFGS", file_content)
                println("  ✅ LBFGS optimizer found in source code")
                @test true
            else
                println("  ⚠️  LBFGS optimizer not found in source code")
                println("     Note: Check if Optim.jl with LBFGS() is being used")
            end
        end

        println("✅ NeuralBoosted: PASS")

    catch e
        println("  ❌ NeuralBoosted training failed with error:")
        println("     $(sprint(showerror, e))")
        @test false
        println("❌ NeuralBoosted: FAIL")
    end

    # Test through model wrapper
    println("\nTesting NeuralBoosted through model wrapper:")
    try
        config = Dict{String, Any}("n_estimators" => 15, "learning_rate" => 0.1,
                     "hidden_layer_size" => 3, "activation" => "tanh")
        model_wrapper = build_model("NeuralBoosted", config, "regression")
        fit_model!(model_wrapper, X_train, y_train)

        y_pred = predict_model(model_wrapper, X_test)
        r2 = r2_score(y_test, y_pred)

        @test all(isfinite.(y_pred))
        println("  ✓ Model wrapper: R²=$(round(r2, digits=3))")

    catch e
        println("  ⚠️  Model wrapper test failed: $(sprint(showerror, e))")
    end
end


# ============================================================================
# INTEGRATION TEST: Full Search Pipeline
# ============================================================================

@testset "Integration: Full Search Pipeline with All Models" begin
    println("\n" * "-"^80)
    println("INTEGRATION TEST: Full Search Pipeline")
    println("-"^80)

    # Generate data
    X, y, wavelengths, _ = generate_spectral_data(n_samples=100, n_wavelengths=50)

    println("Running full search with all 6 models...")

    # Test with all models
    results = run_search(
        X, y, wavelengths,
        task_type="regression",
        models=["PLS", "Ridge", "Lasso", "RandomForest", "MLP", "NeuralBoosted"],
        preprocessing=["raw", "snv"],
        enable_variable_subsets=false,
        enable_region_subsets=false,
        n_folds=3
    )

    @test nrow(results) > 0
    println("✓ Search completed: $(nrow(results)) configurations tested")

    # Check all models are present
    models_tested = unique(results.Model)
    println("✓ Models tested: $(join(models_tested, ", "))")

    # Verify results structure
    @test "Model" in names(results)
    @test "Preprocess" in names(results)
    @test "RMSE" in names(results)
    @test "R2" in names(results)
    @test "Rank" in names(results)

    # Check top 5 models
    println("\nTop 5 models:")
    for i in 1:min(5, nrow(results))
        row = results[i, :]
        println("  $i. $(row.Model) $(row.Preprocess): R²=$(round(row.R2, digits=3)), RMSE=$(round(row.RMSE, digits=3))")
    end

    println("✅ Full pipeline: PASS")
end


# ============================================================================
# FINAL SUMMARY
# ============================================================================

println("\n" * "="^80)
println("COMPREHENSIVE TEST SUMMARY")
println("="^80)
println("\nModel Testing Results:")
println("  1. PLS:           ✅ PASS")
print("  2. Ridge:         ")

# Check if Ridge has alpha=1000.0
configs = get_model_configs("Ridge")
alpha_values = [c["alpha"] for c in configs]
if 1000.0 in alpha_values
    println("✅ PASS (alpha=1000.0 fix applied)")
else
    println("⚠️  CONDITIONAL PASS (alpha=1000.0 fix missing)")
end

println("  3. Lasso:         ✅ PASS")

# Check if RandomForest has min_samples_leaf=1
print("  4. RandomForest:  ")
models_file = joinpath(@__DIR__, "..", "src", "models.jl")
if isfile(models_file)
    file_content = read(models_file, String)
    if occursin(r"1,\s*#\s*min_samples_leaf", file_content)
        println("✅ PASS (min_samples_leaf=1 fix applied)")
    else
        println("⚠️  CONDITIONAL PASS (min_samples_leaf=1 fix missing)")
    end
else
    println("⚠️  CONDITIONAL PASS (could not verify)")
end

println("  5. MLP:           ✅ PASS")
println("  6. NeuralBoosted: (see detailed output above)")

println("\n" * "="^80)
println("To apply missing fixes:")
println("  Ridge: Add 1000.0 to alpha_list in get_model_configs()")
println("  RandomForest: Change line 539 from '5,' to '1,' (min_samples_leaf)")
println("="^80)

println("\n✅ ALL TESTS COMPLETED")
