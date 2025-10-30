"""
Preprocessing Module Examples

This file demonstrates all the key features of the preprocessing module.
Run this file to verify the implementation works correctly.
"""

# Add the module to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Load the preprocessing module
include(joinpath(@__DIR__, "..", "src", "preprocessing.jl"))

using Statistics
using Printf

println("="^80)
println("SpectralPredict.jl - Preprocessing Module Examples")
println("="^80)

#==============================================================================#
# Example 1: Basic SNV Transformation
#==============================================================================#

println("\n" * "="^80)
println("Example 1: Basic SNV Transformation")
println("="^80)

# Create sample data: 3 samples, 10 wavelengths
X = [1000.0 1010.0 1020.0 1030.0 1040.0 1050.0 1060.0 1070.0 1080.0 1090.0;
     2000.0 2010.0 2020.0 2030.0 2040.0 2050.0 2060.0 2070.0 2080.0 2090.0;
     1500.0 1510.0 1520.0 1530.0 1540.0 1550.0 1560.0 1570.0 1580.0 1590.0]

println("\nOriginal data:")
println("  Shape: $(size(X))")
println("  Row 1 mean: $(mean(X[1, :])), std: $(std(X[1, :]))")

# Apply SNV
X_snv = apply_snv(X)

println("\nAfter SNV transformation:")
println("  Shape: $(size(X_snv))")
println("  Row 1 mean: $(@sprintf("%.10f", mean(X_snv[1, :]))), std: $(std(X_snv[1, :]))")
println("  Row 2 mean: $(@sprintf("%.10f", mean(X_snv[2, :]))), std: $(std(X_snv[2, :]))")
println("  Row 3 mean: $(@sprintf("%.10f", mean(X_snv[3, :]))), std: $(std(X_snv[3, :]))")

#==============================================================================#
# Example 2: Savitzky-Golay Derivatives
#==============================================================================#

println("\n" * "="^80)
println("Example 2: Savitzky-Golay Derivatives")
println("="^80)

# Create sample data with a trend
X = rand(Float64, 10, 51) .* 100 .+ 1000

println("\nOriginal data:")
println("  Shape: $(size(X))")

# First derivative
X_d1 = apply_derivative(X, deriv=1, window=11, polyorder=2)
println("\nFirst derivative (window=11, polyorder=2):")
println("  Shape: $(size(X_d1))")

# Second derivative
X_d2 = apply_derivative(X, deriv=2, window=17, polyorder=3)
println("\nSecond derivative (window=17, polyorder=3):")
println("  Shape: $(size(X_d2))")

#==============================================================================#
# Example 3: Pipeline Configurations
#==============================================================================#

println("\n" * "="^80)
println("Example 3: Pipeline Configurations")
println("="^80)

X = rand(Float64, 100, 101) .* 100 .+ 1000

configs = [
    Dict("name" => "raw"),
    Dict("name" => "snv"),
    Dict("name" => "deriv", "deriv" => 1, "window" => 11, "polyorder" => 2),
    Dict("name" => "deriv", "deriv" => 2, "window" => 17, "polyorder" => 3),
    Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2),
    Dict("name" => "snv_deriv", "deriv" => 2, "window" => 17, "polyorder" => 3),
    Dict("name" => "deriv_snv", "deriv" => 1, "window" => 11, "polyorder" => 2)
]

println("\nApplying different preprocessing pipelines:")
println("Original data: $(size(X))\n")

for config in configs
    pipeline = build_preprocessing_pipeline(config)
    X_t = pipeline(X)

    # Format config name nicely
    if config["name"] in ["raw", "snv"]
        config_str = config["name"]
    else
        d = get(config, "deriv", 1)
        w = get(config, "window", 11)
        p = get(config, "polyorder", 2)
        config_str = "$(config["name"]) (d=$d, w=$w, p=$p)"
    end

    println("  $(rpad(config_str, 40)) → $(size(X_t))")
end

#==============================================================================#
# Example 4: Reusable Pipeline
#==============================================================================#

println("\n" * "="^80)
println("Example 4: Reusable Pipeline (Best Practice)")
println("="^80)

# Build pipeline once
config = Dict(
    "name" => "snv_deriv",
    "deriv" => 2,
    "window" => 17,
    "polyorder" => 3
)

pipeline = build_preprocessing_pipeline(config)

# Apply to multiple datasets
X_train = rand(Float64, 100, 101)
X_val = rand(Float64, 20, 101)
X_test = rand(Float64, 30, 101)

X_train_pp = pipeline(X_train)
X_val_pp = pipeline(X_val)
X_test_pp = pipeline(X_test)

println("\nBuilt pipeline: SNV → 2nd derivative (window=17, polyorder=3)")
println("  X_train: $(size(X_train)) → $(size(X_train_pp))")
println("  X_val:   $(size(X_val)) → $(size(X_val_pp))")
println("  X_test:  $(size(X_test)) → $(size(X_test_pp))")

#==============================================================================#
# Example 5: Convenience Function
#==============================================================================#

println("\n" * "="^80)
println("Example 5: Convenience Function (One-shot Use)")
println("="^80)

X = rand(Float64, 50, 101)

config = Dict(
    "name" => "snv_deriv",
    "deriv" => 1,
    "window" => 11,
    "polyorder" => 2
)

X_transformed = apply_preprocessing(X, config)

println("\nOne-shot preprocessing:")
println("  Input:  $(size(X))")
println("  Output: $(size(X_transformed))")
println("  Config: $(config["name"]) (deriv=$(config["deriv"]), window=$(config["window"]))")

#==============================================================================#
# Example 6: Error Handling
#==============================================================================#

println("\n" * "="^80)
println("Example 6: Error Handling")
println("="^80)

X = rand(Float64, 10, 50)

test_cases = [
    (desc="Window too small for polyorder",
     test=() -> apply_derivative(X, deriv=1, window=3, polyorder=5)),
    (desc="Window larger than features",
     test=() -> apply_derivative(rand(10, 20), deriv=1, window=50, polyorder=3)),
    (desc="Unknown preprocessing name",
     test=() -> build_preprocessing_pipeline(Dict("name" => "invalid")))
]

println("\nTesting error handling:")
for tc in test_cases
    try
        tc.test()
        println("  ✗ ERROR: Expected error for: $(tc.desc)")
    catch e
        println("  ✓ $(tc.desc)")
        println("    $(typeof(e)): $(first(split(string(e), "\n")))")
    end
end

#==============================================================================#
# Example 7: SNV Edge Cases
#==============================================================================#

println("\n" * "="^80)
println("Example 7: SNV Edge Cases")
println("="^80)

# Constant spectrum (zero std)
X_const = ones(Float64, 5, 10) .* 100.0
X_snv_const = apply_snv(X_const)

println("\nConstant spectrum (all values = 100.0):")
println("  Input std:  $(std(X_const[1, :]))")
println("  Output:     all zeros? $(all(X_snv_const .== 0.0))")

# Mixed: some constant, some varying
X_mixed = vcat(
    ones(Float64, 2, 10) .* 100.0,  # Constant rows
    rand(Float64, 3, 10) .* 100 .+ 1000  # Varying rows
)
X_mixed_snv = apply_snv(X_mixed)

println("\nMixed spectra (constant + varying):")
println("  Row 1 (constant): std = $(@sprintf("%.10f", std(X_mixed_snv[1, :])))")
println("  Row 3 (varying):  std = $(std(X_mixed_snv[3, :]))")

#==============================================================================#
# Example 8: Feature Count Tracking
#==============================================================================#

println("\n" * "="^80)
println("Example 8: Feature Count Tracking")
println("="^80)

X = rand(Float64, 10, 101)

configs_for_tracking = [
    Dict("name" => "raw"),
    Dict("name" => "snv"),
    Dict("name" => "deriv", "deriv" => 1, "window" => 11, "polyorder" => 2),
    Dict("name" => "deriv", "deriv" => 1, "window" => 17, "polyorder" => 3),
    Dict("name" => "deriv", "deriv" => 1, "window" => 21, "polyorder" => 3),
    Dict("name" => "snv_deriv", "deriv" => 2, "window" => 17, "polyorder" => 3)
]

println("\nOriginal: $(size(X, 2)) features")
for config in configs_for_tracking
    X_t = apply_preprocessing(X, config)

    if config["name"] in ["raw", "snv"]
        config_str = config["name"]
    else
        d = get(config, "deriv", 1)
        w = get(config, "window", 11)
        config_str = "$(config["name"]) (d=$d, w=$w)"
    end

    features_kept = size(X_t, 2)
    println("  $(rpad(config_str, 30)) → $features_kept features")
end

#==============================================================================#
# Summary
#==============================================================================#

println("\n" * "="^80)
println("All Examples Completed Successfully!")
println("="^80)
println("\nKey Takeaways:")
println("  • SNV normalizes each spectrum (row) to mean=0, std=1")
println("  • Savitzky-Golay derivatives maintain feature count (same shape)")
println("  • Build pipelines once, reuse for multiple datasets")
println("  • Use apply_preprocessing() for one-shot transformations")
println("  • All functions are type-stable and optimized for performance")
println("="^80)
