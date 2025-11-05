"""
    check_setup.jl

Quick setup verification script for benchmarks.

Run this before executing benchmarks to ensure everything is configured correctly.

Usage:
    julia benchmark/check_setup.jl
"""

println("="^80)
println("Benchmark Setup Verification")
println("="^80)
println()

# Check Julia version
println("1. Julia Version")
println("-"^80)
println("   Version: $(VERSION)")
if VERSION >= v"1.9"
    println("   Status: ✓ Julia 1.9+ (meets requirement)")
else
    println("   Status: ✗ Julia 1.9+ required (have $(VERSION))")
end
println()

# Check threads
println("2. Threading Configuration")
println("-"^80)
n_threads = Threads.nthreads()
println("   Julia threads: $n_threads")
if n_threads == 1
    println("   Status: ⚠ Only 1 thread available")
    println("   Recommendation: Run with julia --threads=auto")
    println("   Or set JULIA_NUM_THREADS environment variable")
elseif n_threads >= 4
    println("   Status: ✓ $n_threads threads (excellent for parallelization)")
else
    println("   Status: ⚠ $n_threads threads (will work, but 4+ recommended)")
end
println()

# Check BLAS
println("3. BLAS Configuration")
println("-"^80)
try
    using LinearAlgebra
    if isdefined(LinearAlgebra.BLAS, :get_num_threads)
        blas_threads = LinearAlgebra.BLAS.get_num_threads()
        println("   BLAS threads: $blas_threads")
        if blas_threads >= n_threads
            println("   Status: ✓ BLAS threads configured")
        else
            println("   Status: ⚠ BLAS threads < Julia threads")
            println("   Suggestion: LinearAlgebra.BLAS.set_num_threads($n_threads)")
        end
    else
        println("   Status: ℹ BLAS thread control not available")
    end
catch e
    println("   Status: ✗ Error checking BLAS: $e")
end
println()

# Check required packages
println("4. Required Packages")
println("-"^80)

required_packages = [
    "Statistics",
    "Random",
    "LinearAlgebra",
    "Printf",
    "Dates",
    "JSON",
    "Distributions",
    "MultivariateStats",
    "Flux",
    "SavitzkyGolay"
]

all_packages_ok = true

for pkg in required_packages
    try
        @eval using $(Symbol(pkg))
        println("   ✓ $pkg")
    catch e
        println("   ✗ $pkg (NOT AVAILABLE)")
        all_packages_ok = false
    end
end

if all_packages_ok
    println("\n   Status: ✓ All required packages available")
else
    println("\n   Status: ✗ Some packages missing")
    println("   Fix: Run 'julia> ] activate . ; instantiate' in SpectralPredict/")
end
println()

# Check SpectralPredict module
println("5. SpectralPredict Module")
println("-"^80)

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

try
    @eval using SpectralPredict
    println("   ✓ SpectralPredict module loaded")

    # Check for key functions
    functions_to_check = [
        :uve_selection,
        :spa_selection,
        :ipls_selection,
        :uve_spa_selection
    ]

    functions_ok = true
    for func in functions_to_check
        if isdefined(SpectralPredict, func)
            println("   ✓ $func available")
        else
            println("   ✗ $func NOT FOUND")
            functions_ok = false
        end
    end

    if functions_ok
        println("\n   Status: ✓ All benchmark functions available")
    else
        println("\n   Status: ✗ Some functions missing")
        println("   Check: Ensure Julia port implementation is complete")
    end
catch e
    println("   ✗ Cannot load SpectralPredict module")
    println("   Error: $e")
    println("\n   Status: ✗ Module not available")
    println("   Fix: Ensure you're in the SpectralPredict/ directory")
end
println()

# Check benchmark files
println("6. Benchmark Files")
println("-"^80)

benchmark_files = [
    "bench_variable_selection.jl",
    "bench_diagnostics.jl",
    "bench_neural_boosted.jl",
    "bench_msc.jl",
    "bench_comprehensive.jl"
]

all_files_present = true

for file in benchmark_files
    file_path = joinpath(@__DIR__, file)
    if isfile(file_path)
        size_kb = round(filesize(file_path) / 1024, digits=1)
        println("   ✓ $file ($size_kb KB)")
    else
        println("   ✗ $file (NOT FOUND)")
        all_files_present = false
    end
end

if all_files_present
    println("\n   Status: ✓ All benchmark files present")
else
    println("\n   Status: ✗ Some benchmark files missing")
end
println()

# System information
println("7. System Information")
println("-"^80)
println("   OS: $(Sys.KERNEL)")
println("   CPU: $(Sys.CPU_NAME)")
println("   CPU threads: $(Sys.CPU_THREADS)")
println("   Physical cores: $(Sys.CPU_THREADS ÷ 2)")  # Estimate
println()

# Memory check
println("8. Memory")
println("-"^80)
total_mem_gb = round(Sys.total_memory() / (1024^3), digits=1)
println("   Total RAM: $total_mem_gb GB")
if total_mem_gb >= 8
    println("   Status: ✓ Sufficient for all benchmarks")
elseif total_mem_gb >= 4
    println("   Status: ⚠ Sufficient for small/medium benchmarks")
    println("   Note: Large benchmarks may be slow or fail")
else
    println("   Status: ⚠ Low memory detected")
    println("   Recommendation: Run only small benchmarks")
end
println()

# Overall status
println("="^80)
println("OVERALL STATUS")
println("="^80)

checks_passed = 0
checks_total = 6

# Count checks
checks_passed += (VERSION >= v"1.9") ? 1 : 0
checks_passed += (n_threads >= 1) ? 1 : 0
checks_passed += all_packages_ok ? 1 : 0
checks_passed += all_files_present ? 1 : 0
checks_passed += (total_mem_gb >= 4) ? 1 : 0

# Try to load module
module_loaded = false
try
    @eval using SpectralPredict
    module_loaded = true
    checks_passed += 1
catch
end

println()
if checks_passed == checks_total
    println("✓ All checks passed! Ready to run benchmarks.")
    println()
    println("Recommended commands:")
    println("  julia --threads=auto benchmark/bench_comprehensive.jl")
    println("  julia --threads=auto benchmark/bench_variable_selection.jl")
    println("  julia --threads=auto benchmark/bench_diagnostics.jl")
elseif checks_passed >= checks_total - 1
    println("⚠ Most checks passed. You can run benchmarks, but some may have issues.")
    println()
    println("Review warnings above and fix if possible.")
    println("Then run:")
    println("  julia --threads=auto benchmark/bench_comprehensive.jl")
else
    println("✗ Multiple issues detected. Please fix the problems above before running benchmarks.")
    println()
    println("Common fixes:")
    println("  1. Install packages: julia> ] activate . ; instantiate")
    println("  2. Enable threads: export JULIA_NUM_THREADS=auto")
    println("  3. Check you're in SpectralPredict/ directory")
end

println()
println("="^80)
println("For help, see:")
println("  - benchmark/QUICKSTART.md")
println("  - benchmark/README.md")
println("  - ../TROUBLESHOOTING.md")
println("="^80)
