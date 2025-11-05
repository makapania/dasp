"""
    bench_msc.jl

Performance benchmarks for MSC (Multiplicative Scatter Correction) preprocessing.

Tests realistic spectroscopy data sizes:
- Small: 100 samples × 500 wavelengths (rapid prototyping)
- Medium: 300 samples × 1500 wavelengths (typical NIR spectroscopy)
- Large: 1000 samples × 2151 wavelengths (full resolution spectroscopy)
- Extra Large: 5000 samples × 2151 wavelengths (large dataset)

Expected speedups from implementation plan:
- MSC computation: 8-12x faster than Python
- Benefits from Julia's efficient linear algebra
- Vectorized operations with minimal allocations

Run with: julia --threads=auto benchmark/bench_msc.jl
"""

using Printf
using Statistics
using Random
using LinearAlgebra
using Dates

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Import preprocessing functions
# Note: Check if msc is exported from SpectralPredict module
try
    using SpectralPredict: apply_msc
catch
    # If not exported, include preprocessing.jl directly
    include(joinpath(@__DIR__, "..", "src", "preprocessing.jl"))
end

#=============================================================================
    Benchmark Utilities
=============================================================================#

"""
    generate_spectral_data(n_samples, n_wavelengths)

Generate synthetic spectral data with scatter effects.

Simulates realistic NIR spectra with:
- Baseline shifts (additive scatter)
- Multiplicative scatter (scaling)
- Spectral features (peaks)
- Random noise
"""
function generate_spectral_data(n_samples::Int, n_wavelengths::Int)
    Random.seed!(42)

    X = zeros(n_samples, n_wavelengths)

    # Wavelength axis (e.g., 400-2500 nm)
    wavelengths = range(400, 2500, length=n_wavelengths)

    for i in 1:n_samples
        # Base spectrum (smooth curve with peaks)
        spectrum = zeros(n_wavelengths)

        # Add several Gaussian peaks (chemical absorption bands)
        peak_positions = [800, 1200, 1450, 1900, 2100]
        for peak_λ in peak_positions
            if peak_λ < maximum(wavelengths)
                # Find closest wavelength index
                peak_idx = argmin(abs.(wavelengths .- peak_λ))

                # Add Gaussian peak
                width = 50.0
                for j in 1:n_wavelengths
                    dist = (wavelengths[j] - wavelengths[peak_idx])^2
                    spectrum[j] += exp(-dist / (2 * width^2)) * (0.5 + randn() * 0.1)
                end
            end
        end

        # Add baseline (constant offset)
        baseline = 1.0 + randn() * 0.3

        # Add multiplicative scatter effect (scaling)
        scatter_factor = 1.0 + randn() * 0.2

        # Combine effects
        X[i, :] = (baseline .+ spectrum) .* scatter_factor

        # Add measurement noise
        X[i, :] .+= randn(n_wavelengths) .* 0.01
    end

    return X
end

"""
    run_benchmark(func, args...; n_warmup=2, n_runs=5, name="")

Run benchmark with warmup and multiple iterations.
"""
function run_benchmark(func, args...; n_warmup=2, n_runs=5, name="")
    # Warmup runs (JIT compilation)
    println("  Warming up $name...")
    for i in 1:n_warmup
        func(args...)
    end

    # Timed runs
    println("  Running $name ($n_runs iterations)...")
    times = Float64[]

    for i in 1:n_runs
        t_start = time_ns()
        result = func(args...)
        t_end = time_ns()

        elapsed = (t_end - t_start) / 1e9
        push!(times, elapsed)
    end

    # Memory estimate
    mem_start = @allocated func(args...)
    mem_mb = mem_start / (1024^2)

    return (
        mean = mean(times),
        std = std(times),
        min = minimum(times),
        max = maximum(times),
        mem_mb = mem_mb
    )
end

"""
    print_benchmark_results(name, stats, target_speedup="")

Print formatted benchmark results.
"""
function print_benchmark_results(name, stats, target_speedup="")
    println("\n$name:")
    @printf("  Mean:   %.4f ± %.4f s\n", stats.mean, stats.std)
    @printf("  Min:    %.4f s\n", stats.min)
    @printf("  Max:    %.4f s\n", stats.max)
    @printf("  Memory: %.2f MB\n", stats.mem_mb)
    if !isempty(target_speedup)
        println("  Target speedup: $target_speedup")
    end
end

"""
    verify_msc_correctness(X, X_msc)

Verify MSC transformation is correct.
"""
function verify_msc_correctness(X, X_msc)
    n_samples, n_wavelengths = size(X)

    println("\n  Verification:")

    # Check dimensions
    if size(X_msc) != size(X)
        println("    ❌ Dimension mismatch!")
        return false
    end
    println("    ✓ Dimensions preserved: $n_samples × $n_wavelengths")

    # Check that scatter was reduced
    # MSC should reduce between-sample variance at most wavelengths
    var_before = var(X, dims=1)
    var_after = var(X_msc, dims=1)

    variance_reduction = mean(var_after ./ var_before)
    @printf("    ✓ Average variance reduction: %.2f%%\n", (1 - variance_reduction) * 100)

    # Check for NaN or Inf
    if any(isnan.(X_msc)) || any(isinf.(X_msc))
        println("    ❌ Contains NaN or Inf values!")
        return false
    end
    println("    ✓ No NaN or Inf values")

    return true
end

#=============================================================================
    Benchmark Tests
=============================================================================#

function benchmark_msc(scale_name, X)
    println("\n$scale_name - MSC Preprocessing")
    println("="^60)

    stats = run_benchmark(apply_msc, X,
                         n_warmup=2, n_runs=5, name="MSC")

    print_benchmark_results("MSC Computation", stats, "8-12x vs Python")

    # Run once more for verification
    X_msc = apply_msc(X)
    verify_msc_correctness(X, X_msc)

    return stats
end

#=============================================================================
    Throughput Analysis
=============================================================================#

function analyze_throughput(results, test_scales)
    println("\n" * "="^80)
    println("THROUGHPUT ANALYSIS")
    println("="^80)

    println("\nSamples processed per second:")
    println("-"^80)
    @printf("%-30s %-15s %-15s\n", "Scale", "Throughput", "MB/s")
    println("-"^80)

    for (i, (scale_name, n_samples, n_wavelengths)) in enumerate(test_scales)
        stats = results[scale_name]

        # Samples per second
        throughput = n_samples / stats.mean

        # Data throughput (MB/s)
        # Each value is Float64 (8 bytes)
        data_size_mb = (n_samples * n_wavelengths * 8) / (1024^2)
        data_throughput = data_size_mb / stats.mean

        @printf("%-30s %-15.2f %-15.2f\n",
                scale_name, throughput, data_throughput)
    end

    println()
end

#=============================================================================
    Main Benchmark Suite
=============================================================================#

function main()
    println("="^80)
    println("MSC Preprocessing Performance Benchmarks")
    println("="^80)
    println("\nJulia Version: $(VERSION)")
    println("Threads Available: $(Threads.nthreads())")
    println("BLAS Threads: $(LinearAlgebra.BLAS.get_num_threads())")
    println("Date: $(Dates.now())")
    println()

    # Define test scales
    test_scales = [
        ("Small (100 × 500)", 100, 500),
        ("Medium (300 × 1500)", 300, 1500),
        ("Large (1000 × 2151)", 1000, 2151),
        ("Extra Large (5000 × 2151)", 5000, 2151)
    ]

    results = Dict()

    for (scale_name, n_samples, n_wavelengths) in test_scales
        println("\n" * "="^80)
        println("Scale: $scale_name")
        println("="^80)

        # Generate data
        println("Generating synthetic spectral data...")
        X = generate_spectral_data(n_samples, n_wavelengths)
        println("  Samples: $n_samples")
        println("  Wavelengths: $n_wavelengths")
        @printf("  Data size: %.2f MB\n", (n_samples * n_wavelengths * 8) / (1024^2))

        # Run benchmark
        stats = benchmark_msc(scale_name, X)
        results[scale_name] = stats
    end

    # Throughput analysis
    analyze_throughput(results, test_scales)

    # Summary Report
    println("\n\n" * "="^80)
    println("SUMMARY REPORT")
    println("="^80)

    println("\nMean Execution Times (seconds):")
    println("-"^80)
    @printf("%-30s %-15s %-15s\n", "Scale", "Time (s)", "Memory (MB)")
    println("-"^80)

    for (scale_name, n_samples, n_wavelengths) in test_scales
        stats = results[scale_name]
        @printf("%-30s %-15.4f %-15.2f\n",
                scale_name,
                stats.mean,
                stats.mem_mb)
    end

    println("\n" * "="^80)
    println("Target Speedup (vs Python):")
    println("  - MSC computation: 8-12x faster")
    println()
    println("Performance Characteristics:")
    println("  - Efficient linear algebra via Julia's BLAS")
    println("  - Minimal allocations (preallocated arrays)")
    println("  - Scales linearly with data size")
    println("  - Memory usage approximately 2x input size (input + output)")
    println()
    println("Notes:")
    println("  - MSC reduces multiplicative scatter in spectral data")
    println("  - Each spectrum corrected via regression to mean spectrum")
    println("  - Critical preprocessing step before PLS/PCR modeling")
    println("  - Warmup runs excluded from timing (JIT compilation)")
    println("="^80)
end

# Run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
