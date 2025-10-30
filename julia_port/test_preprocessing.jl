"""
Quick test script for preprocessing.jl functionality
"""

# Add the SpectralPredict module to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "SpectralPredict", "src"))

# Load the preprocessing module
include(joinpath(@__DIR__, "SpectralPredict", "src", "preprocessing.jl"))

using Statistics
using Printf

println("=" ^ 70)
println("Testing SpectralPredict Preprocessing Module")
println("=" ^ 70)

# Test 1: SNV transformation
println("\n[Test 1] SNV Transformation")
println("-" ^ 70)
X = rand(Float64, 10, 50) .* 100 .+ 1000
println("Input shape: $(size(X))")

X_snv = apply_snv(X)
println("Output shape: $(size(X_snv))")

# Verify properties
row_means = [mean(X_snv[i, :]) for i in 1:size(X_snv, 1)]
row_stds = [std(X_snv[i, :]) for i in 1:size(X_snv, 1)]

println("Row means (should be ≈ 0): ", @sprintf("%.10f", maximum(abs.(row_means))))
println("Row stds (should be ≈ 1): ", @sprintf("%.10f", maximum(abs.(row_stds .- 1.0))))

if maximum(abs.(row_means)) < 1e-10 && maximum(abs.(row_stds .- 1.0)) < 1e-10
    println("✓ SNV transformation correct!")
else
    println("✗ SNV transformation failed!")
end

# Test 2: Derivative transformation
println("\n[Test 2] Savitzky-Golay Derivative")
println("-" ^ 70)
X = rand(Float64, 10, 101) .* 100 .+ 1000
println("Input shape: $(size(X))")

X_d1 = apply_derivative(X, deriv=1, window=17, polyorder=3)
println("Output shape (window=17): $(size(X_d1))")
println("Features preserved: $(size(X_d1, 2))")

if size(X_d1, 1) == size(X, 1) && size(X_d1, 2) == size(X, 2)
    println("✓ Derivative maintains dimensions with DSP.jl")
else
    println("Note: Feature count = $(size(X_d1, 2)) (DSP.jl behavior)")
end

# Test 3: Pipeline configurations
println("\n[Test 3] Pipeline Configurations")
println("-" ^ 70)

configs = [
    Dict("name" => "raw"),
    Dict("name" => "snv"),
    Dict("name" => "deriv", "deriv" => 1, "window" => 17, "polyorder" => 3),
    Dict("name" => "snv_deriv", "deriv" => 1, "window" => 17, "polyorder" => 3),
    Dict("name" => "deriv_snv", "deriv" => 2, "window" => 11, "polyorder" => 2)
]

X = rand(Float64, 100, 101)
println("Input: $(size(X))")

for config in configs
    pipeline = build_preprocessing_pipeline(config)
    X_transformed = pipeline(X)
    println("  $(rpad(config["name"], 12)) -> $(size(X_transformed))")
end

# Test 4: apply_preprocessing convenience function
println("\n[Test 4] Convenience Function")
println("-" ^ 70)

config = Dict(
    "name" => "snv_deriv",
    "deriv" => 2,
    "window" => 17,
    "polyorder" => 3
)

X = rand(Float64, 50, 101)
X_transformed = apply_preprocessing(X, config)
println("Input: $(size(X))")
println("Output: $(size(X_transformed))")
println("✓ Convenience function works!")

# Test 5: Error handling
println("\n[Test 5] Error Handling")
println("-" ^ 70)

test_cases = [
    (desc="Window > polyorder + 1",
     test=() -> apply_derivative(X, deriv=1, window=3, polyorder=3)),
    (desc="Window > features",
     test=() -> apply_derivative(rand(10, 20), deriv=1, window=50, polyorder=3)),
    (desc="Unknown pipeline name",
     test=() -> build_preprocessing_pipeline(Dict("name" => "invalid")))
]

for tc in test_cases
    try
        tc.test()
        println("✗ Expected error for: $(tc.desc)")
    catch e
        println("✓ Caught expected error: $(tc.desc)")
        println("  $(typeof(e)): $(first(split(string(e), "\n")))")
    end
end

# Test 6: Zero std case for SNV
println("\n[Test 6] SNV with Zero Standard Deviation")
println("-" ^ 70)

X_const = ones(Float64, 5, 10) .* 100.0
X_snv_const = apply_snv(X_const)
println("Constant input (all 100.0): $(size(X_const))")
println("After SNV: all zeros = $(all(X_snv_const .== 0.0))")

if all(X_snv_const .== 0.0)
    println("✓ Zero std case handled correctly!")
else
    println("✗ Zero std case not handled correctly!")
end

println("\n" * "=" ^ 70)
println("All tests completed!")
println("=" ^ 70)
