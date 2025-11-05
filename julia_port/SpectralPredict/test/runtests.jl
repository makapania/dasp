"""
Master test runner for SpectralPredict.jl

Runs all test suites for the newly implemented modules:
- Variable Selection (UVE, SPA, iPLS, UVE-SPA)
- Diagnostics (residuals, leverage, Q-Q plots, jackknife)
- Neural Boosted (gradient boosting with MLP weak learners)
- MSC (Multiplicative Scatter Correction preprocessing)
- Integration Tests (end-to-end workflows)

Usage:
    julia --project=. test/runtests.jl

Or to run individual test files:
    julia --project=. test/test_variable_selection.jl
    julia --project=. test/test_diagnostics.jl
    julia --project=. test/test_neural_boosted.jl
    julia --project=. test/test_msc.jl
    julia --project=. test/test_integration.jl

To run without integration tests (faster):
    SKIP_INTEGRATION=1 julia --project=. test/runtests.jl
"""

using Test

println("="^70)
println("SpectralPredict.jl - Comprehensive Test Suite")
println("="^70)
println()

# Track test results
test_results = Dict{String, Bool}()

# Check if integration tests should be skipped
skip_integration = haskey(ENV, "SKIP_INTEGRATION") && ENV["SKIP_INTEGRATION"] == "1"

# Test files to run
test_files = [
    ("Variable Selection", "test_variable_selection.jl"),
    ("Diagnostics", "test_diagnostics.jl"),
    ("Neural Boosted", "test_neural_boosted.jl"),
    ("MSC Preprocessing", "test_msc.jl")
]

# Add integration tests if not skipped
if !skip_integration
    push!(test_files, ("Integration Tests", "test_integration.jl"))
else
    println("⚠️  Skipping integration tests (SKIP_INTEGRATION=1)")
    println()
end

println("Running $(length(test_files)) test suites...\n")

# Run each test file
for (name, file) in test_files
    println("\n" * "="^70)
    println("Testing: $name")
    println("="^70)

    try
        include(file)
        test_results[name] = true
        println("✓ $name: PASSED")
    catch e
        test_results[name] = false
        println("✗ $name: FAILED")
        println("Error: $e")
        # Print stacktrace for debugging
        showerror(stdout, e, catch_backtrace())
        println()
    end
end

# Summary
println("\n" * "="^70)
println("Test Summary")
println("="^70)

n_passed = sum(values(test_results))
n_total = length(test_results)

for (name, passed) in test_results
    status = passed ? "✓ PASSED" : "✗ FAILED"
    println("  $status: $name")
end

println()
println("Total: $n_passed/$n_total test suites passed")
println("="^70)

# Exit with appropriate code
if n_passed == n_total
    println("\n🎉 All tests passed!")
    exit(0)
else
    println("\n⚠️  Some tests failed. See details above.")
    exit(1)
end
