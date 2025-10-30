"""
Example: Complete Hyperparameter Search for Spectral Prediction

This example demonstrates the full workflow for running a comprehensive
hyperparameter search on spectral data.

The search includes:
- Multiple preprocessing methods (raw, SNV, derivatives)
- Multiple model types (PLS, Ridge, Lasso, RandomForest, MLP)
- Variable subset analysis (top-N feature selection)
- Region subset analysis (spectral region detection)
- Cross-validation and ranking
"""

using Random
using DataFrames
using Statistics
using Printf

# Include the main search module
include("../src/search.jl")


println("="^80)
println("Spectral Prediction Search - Example")
println("="^80)
println()


# ============================================================================
# 1. Generate Synthetic Spectral Data
# ============================================================================

println("Step 1: Generating synthetic spectral data...")
println()

Random.seed!(42)

# Spectral parameters
n_samples = 100
n_wavelengths = 200
wavelength_start = 400.0  # nm
wavelength_end = 2498.0   # nm
wavelengths = collect(range(wavelength_start, wavelength_end, length=n_wavelengths))

# Generate synthetic spectral data
# Each spectrum is a mixture of Gaussian peaks with noise
X = zeros(n_samples, n_wavelengths)
y = zeros(n_samples)

for i in 1:n_samples
    # Random target value
    y[i] = 50.0 + 20.0 * randn()

    # Generate spectrum with peaks related to target
    spectrum = zeros(n_wavelengths)

    # Peak 1: intensity related to target
    peak1_center = 800.0
    peak1_width = 50.0
    peak1_intensity = 0.5 * y[i] + 10.0 * randn()
    for (j, wl) in enumerate(wavelengths)
        spectrum[j] += peak1_intensity * exp(-((wl - peak1_center) / peak1_width)^2)
    end

    # Peak 2: another informative peak
    peak2_center = 1200.0
    peak2_width = 80.0
    peak2_intensity = 0.3 * y[i] + 8.0 * randn()
    for (j, wl) in enumerate(wavelengths)
        spectrum[j] += peak2_intensity * exp(-((wl - peak2_center) / peak2_width)^2)
    end

    # Background and noise
    baseline = 20.0 + 5.0 * randn()
    noise = 2.0 * randn(n_wavelengths)

    X[i, :] = spectrum .+ baseline .+ noise
end

println("Generated data:")
println("  Samples: $n_samples")
println("  Wavelengths: $n_wavelengths ($(wavelength_start) - $(wavelength_end) nm)")
println("  Target range: $(round(minimum(y), digits=2)) - $(round(maximum(y), digits=2))")
println()


# ============================================================================
# 2. Run Basic Search (Fast)
# ============================================================================

println("Step 2: Running basic search (fast, for demonstration)...")
println()

results_basic = run_search(
    X, y, wavelengths,
    task_type="regression",
    models=["PLS", "Ridge"],  # Just 2 models for speed
    preprocessing=["raw", "snv"],  # Just 2 preprocessing methods
    enable_variable_subsets=false,  # Disable for speed
    enable_region_subsets=false,  # Disable for speed
    n_folds=5,
    lambda_penalty=0.15
)

println("\nBasic search results:")
println("  Total configurations tested: $(nrow(results_basic))")
println()

# Show top 5 models
println("Top 5 models from basic search:")
top_5_basic = first(sort(results_basic, :Rank), 5)
for row in eachrow(top_5_basic)
    @printf("  Rank %2d: %s + %s | RMSE=%.3f, R²=%.3f\n",
            row.Rank, row.Model, row.Preprocess, row.RMSE, row.R2)
end
println()


# ============================================================================
# 3. Run Comprehensive Search
# ============================================================================

println("Step 3: Running comprehensive search (includes all features)...")
println()

results_full = run_search(
    X, y, wavelengths,
    task_type="regression",
    models=["PLS", "Ridge", "Lasso", "RandomForest"],
    preprocessing=["raw", "snv", "deriv"],
    derivative_orders=[1, 2],
    derivative_window=17,
    derivative_polyorder=3,
    enable_variable_subsets=true,
    variable_counts=[10, 20, 50, 100],
    enable_region_subsets=true,
    n_top_regions=5,
    n_folds=5,
    lambda_penalty=0.15
)

println("\nComprehensive search results:")
println("  Total configurations tested: $(nrow(results_full))")
println()


# ============================================================================
# 4. Analyze Results
# ============================================================================

println("Step 4: Analyzing results...")
println()

# Top 10 models overall
println("Top 10 models overall:")
top_10 = first(sort(results_full, :Rank), 10)
for row in eachrow(top_10)
    subset_info = row.n_vars < row.full_vars ? " [$(row.SubsetTag): $(row.n_vars) vars]" : ""
    @printf("  Rank %2d: %s + %s%s | RMSE=%.3f, R²=%.3f\n",
            row.Rank, row.Model, row.Preprocess, subset_info, row.RMSE, row.R2)
end
println()

# Best model for each preprocessing type
println("Best model for each preprocessing type:")
for prep in unique(results_full.Preprocess)
    prep_results = filter(row -> row.Preprocess == prep, results_full)
    best = first(sort(prep_results, :Rank), 1)
    deriv_info = ismissing(best.Deriv[1]) ? "" : " (deriv=$(best.Deriv[1]))"
    @printf("  %s%s: %s | RMSE=%.3f, R²=%.3f\n",
            prep, deriv_info, best.Model[1], best.RMSE[1], best.R2[1])
end
println()

# Best sparse model (fewer than 50 variables)
println("Best sparse models (< 50 variables):")
sparse_results = filter(row -> row.n_vars < 50, results_full)
if nrow(sparse_results) > 0
    sort!(sparse_results, :Rank)
    top_sparse = first(sparse_results, 5)
    for row in eachrow(top_sparse)
        @printf("  Rank %2d: %s + %s [%s: %d vars] | RMSE=%.3f, R²=%.3f\n",
                row.Rank, row.Model, row.Preprocess, row.SubsetTag,
                row.n_vars, row.RMSE, row.R2)
    end
else
    println("  No sparse models found")
end
println()

# Performance by model type
println("Average performance by model type:")
for model in unique(results_full.Model)
    model_results = filter(row -> row.Model == model, results_full)
    avg_rmse = mean(model_results.RMSE)
    best_rmse = minimum(model_results.RMSE)
    avg_r2 = mean(model_results.R2)
    best_r2 = maximum(model_results.R2)
    @printf("  %s: Avg RMSE=%.3f (Best=%.3f), Avg R²=%.3f (Best=%.3f)\n",
            model, avg_rmse, best_rmse, avg_r2, best_r2)
end
println()

# Variable subset analysis
println("Variable subset performance:")
subset_tags = ["full", "top10", "top20", "top50", "top100"]
for tag in subset_tags
    subset_results = filter(row -> row.SubsetTag == tag, results_full)
    if nrow(subset_results) > 0
        avg_rmse = mean(subset_results.RMSE)
        best_rmse = minimum(subset_results.RMSE)
        avg_vars = mean(subset_results.n_vars)
        @printf("  %s (%.0f vars): Avg RMSE=%.3f, Best RMSE=%.3f\n",
                tag, avg_vars, avg_rmse, best_rmse)
    end
end
println()


# ============================================================================
# 5. Save Results
# ============================================================================

println("Step 5: Saving results...")
println()

using CSV

# Save full results
output_file = "spectral_search_results.csv"
CSV.write(output_file, results_full)
println("Results saved to: $output_file")

# Save top 50 models
top_50_file = "spectral_search_top50.csv"
top_50 = first(sort(results_full, :Rank), min(50, nrow(results_full)))
CSV.write(top_50_file, top_50)
println("Top 50 models saved to: $top_50_file")
println()


# ============================================================================
# 6. Summary Statistics
# ============================================================================

println("="^80)
println("SUMMARY")
println("="^80)
println()

best_model = first(sort(results_full, :Rank), 1)[1, :]

println("Best Model:")
println("  Rank: 1")
println("  Model: $(best_model.Model)")
println("  Preprocessing: $(best_model.Preprocess)")
if !ismissing(best_model.Deriv)
    println("  Derivative order: $(best_model.Deriv)")
    println("  Window size: $(best_model.Window)")
end
if best_model.n_vars < best_model.full_vars
    println("  Subset: $(best_model.SubsetTag) ($(best_model.n_vars) variables)")
end
println("  RMSE: $(round(best_model.RMSE, digits=4))")
println("  R²: $(round(best_model.R2, digits=4))")
println("  Composite Score: $(round(best_model.CompositeScore, digits=4))")
println()

println("Search Statistics:")
println("  Total configurations tested: $(nrow(results_full))")
println("  Models tested: $(join(unique(results_full.Model), ", "))")
println("  Preprocessing methods: $(join(unique(results_full.Preprocess), ", "))")
println("  Best RMSE: $(round(minimum(results_full.RMSE), digits=4))")
println("  Best R²: $(round(maximum(results_full.R2), digits=4))")
println()

println("="^80)
println("Search complete!")
println("="^80)
