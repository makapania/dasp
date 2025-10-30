"""
Example: Using the Regions module for spectral region analysis

This example demonstrates how to:
1. Compute region correlations on spectral data
2. Identify top spectral regions
3. Create variable subsets for model testing
4. Combine regions for comprehensive analysis

Run with: julia --project=. examples/regions_example.jl
"""

using Pkg
Pkg.activate(".")

include("../src/regions.jl")
using .Regions
using Statistics
using Random

println("="^70)
println("Spectral Regions Module - Usage Example")
println("="^70)
println()

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================================
# 1. Create Synthetic Spectral Data
# ============================================================================

println("1. Creating synthetic spectral data...")
println()

n_samples = 100
n_wavelengths = 200
wavelengths = collect(400.0:2.0:798.0)  # 400-798 nm in 2nm steps

# Create base spectral data
X = randn(n_samples, n_wavelengths) * 0.5

# Create target variable (e.g., protein content)
y = randn(n_samples)

# Add strong correlation in specific spectral regions
# These simulate real spectral features related to the target

# Region 1: 500-550 nm (strong signal - e.g., chlorophyll absorption)
mask1 = (wavelengths .>= 500.0) .& (wavelengths .< 550.0)
X[:, mask1] .+= y * 2.0

# Region 2: 650-700 nm (medium signal - e.g., red edge)
mask2 = (wavelengths .>= 650.0) .& (wavelengths .< 700.0)
X[:, mask2] .+= y * 1.2

# Region 3: 450-480 nm (weak signal)
mask3 = (wavelengths .>= 450.0) .& (wavelengths .< 480.0)
X[:, mask3] .+= y * 0.6

println("  Samples: ", n_samples)
println("  Wavelengths: ", n_wavelengths, " (", minimum(wavelengths), "-", maximum(wavelengths), " nm)")
println("  Target variable range: ", round(minimum(y), digits=2), " to ", round(maximum(y), digits=2))
println()

# ============================================================================
# 2. Compute Region Correlations
# ============================================================================

println("2. Computing region correlations...")
println()

# Default parameters: 50nm regions with 25nm overlap
regions = compute_region_correlations(X, y, wavelengths)

println("  Total regions identified: ", length(regions))
println()

# Display top 10 regions
println("  Top 10 Regions by Mean Correlation:")
println("  " * "-"^70)
println("  Rank  Region (nm)           Mean |r|    Max |r|     N features")
println("  " * "-"^70)

# Sort by mean correlation
sorted_regions = sort(regions, by=r -> r["mean_corr"], rev=true)

for (i, region) in enumerate(sorted_regions[1:min(10, length(sorted_regions))])
    region_str = string(Int(round(region["start"])), "-", Int(round(region["end"])))
    println(
        "  ", lpad(i, 4), "  ",
        rpad(region_str, 20), "  ",
        rpad(round(region["mean_corr"], digits=4), 10), "  ",
        rpad(round(region["max_corr"], digits=4), 10), "  ",
        region["n_features"]
    )
end
println()

# ============================================================================
# 3. Create Region Subsets for Model Testing
# ============================================================================

println("3. Creating region subsets for model testing...")
println()

# Create subsets with n_top_regions=10
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=10)

println("  Total subsets created: ", length(subsets))
println()

# Display individual region subsets
println("  Individual Region Subsets:")
individual_subsets = filter(s -> startswith(s["tag"], "region_"), subsets)
for subset in individual_subsets
    println("    ", subset["tag"])
    println("      ", subset["description"])
end
println()

# Display combined region subsets
println("  Combined Region Subsets:")
combined_subsets = filter(s -> startswith(s["tag"], "top"), subsets)
for subset in combined_subsets
    println("    ", subset["tag"])
    println("      ", subset["description"])
end
println()

# ============================================================================
# 4. Using Subsets for Model Training
# ============================================================================

println("4. Example: Using subsets for model training...")
println()

# Select a subset to use
example_subset = subsets[1]
println("  Selected subset: ", example_subset["tag"])
println("  Description: ", example_subset["description"])
println()

# Extract data for this subset
X_subset = X[:, example_subset["indices"]]

println("  Original data shape: ", size(X))
println("  Subset data shape: ", size(X_subset))
println("  Reduction: ", round((1 - size(X_subset, 2) / size(X, 2)) * 100, digits=1), "%")
println()

# Calculate correlation improvement
original_corrs = [abs(cor(X[:, i], y)) for i in 1:size(X, 2)]
subset_corrs = [abs(cor(X_subset[:, i], y)) for i in 1:size(X_subset, 2)]

println("  Average correlation:")
println("    All wavelengths: ", round(mean(original_corrs), digits=4))
println("    Selected subset: ", round(mean(subset_corrs), digits=4))
println("    Improvement: ", round((mean(subset_corrs) - mean(original_corrs)) / mean(original_corrs) * 100, digits=1), "%")
println()

# ============================================================================
# 5. Custom Region Parameters
# ============================================================================

println("5. Using custom region parameters...")
println()

# Try larger regions with no overlap
regions_large = compute_region_correlations(X, y, wavelengths,
                                           region_size=100.0, overlap=0.0)
println("  Large regions (100nm, no overlap): ", length(regions_large), " regions")

# Try smaller regions with more overlap
regions_small = compute_region_correlations(X, y, wavelengths,
                                           region_size=30.0, overlap=20.0)
println("  Small regions (30nm, 20nm overlap): ", length(regions_small), " regions")
println()

# ============================================================================
# 6. Combining Specific Regions
# ============================================================================

println("6. Manually combining specific regions...")
println()

# Get top 3 regions
top3_regions = sorted_regions[1:min(3, length(sorted_regions))]

# Combine their indices
combined_indices = combine_region_indices(top3_regions)

println("  Top 3 regions:")
for (i, region) in enumerate(top3_regions)
    println("    Region ", i, ": ",
            Int(round(region["start"])), "-", Int(round(region["end"])), " nm ",
            "(", length(region["indices"]), " features)")
end
println()
println("  Combined: ", length(combined_indices), " unique features")
println()

# Verify wavelength coverage
combined_wls = wavelengths[combined_indices]
println("  Wavelength coverage: ",
        round(minimum(combined_wls), digits=1), "-",
        round(maximum(combined_wls), digits=1), " nm")
println()

# ============================================================================
# 7. Practical Workflow Example
# ============================================================================

println("7. Complete workflow for hyperparameter search...")
println()

println("  In a typical hyperparameter search, you would:")
println("    1. Compute region subsets on training data")
println("    2. Test each subset with different preprocessing combinations")
println("    3. Evaluate using cross-validation")
println("    4. Rank models by performance")
println()

println("  Example search space:")
subsets_for_search = create_region_subsets(X, y, wavelengths, n_top_regions=5)
preprocessing_methods = ["None", "SNV", "Derivative1", "Derivative2"]

total_combinations = length(subsets_for_search) * length(preprocessing_methods)
println("    ", length(subsets_for_search), " region subsets")
println("    × ", length(preprocessing_methods), " preprocessing methods")
println("    = ", total_combinations, " combinations to test")
println()

println("  Region subsets would be tested:")
for subset in subsets_for_search
    println("    - ", subset["tag"], " (", length(subset["indices"]), " features)")
end
println()

# ============================================================================
# Summary
# ============================================================================

println("="^70)
println("Summary")
println("="^70)
println()
println("The Regions module provides:")
println("  ✓ Automatic identification of important spectral regions")
println("  ✓ Flexible region sizing and overlap parameters")
println("  ✓ Strategic subset creation for model testing")
println("  ✓ Significant reduction in search space while preserving information")
println()
println("Key advantages:")
println("  • Focus on wavelengths with high correlation to target")
println("  • Reduce dimensionality without losing predictive power")
println("  • Test multiple region configurations efficiently")
println("  • Interpretable results (wavelength ranges)")
println()
println("="^70)
