"""
    bench_variable_selection.jl

Performance benchmarks for variable selection methods (UVE, SPA, iPLS, UVE-SPA).

Tests realistic spectroscopy data sizes with multiple data scales:
- Small: 100 samples × 500 wavelengths (rapid prototyping)
- Medium: 300 samples × 1500 wavelengths (typical NIR spectroscopy)
- Large: 1000 samples × 2151 wavelengths (full resolution spectroscopy)

Expected speedups from implementation plan:
- SPA selection: 10-20x faster (parallelized)
- UVE selection: 6-10x faster
- iPLS selection: 8-12x faster
- UVE-SPA: 8-15x faster

Run with: julia --threads=auto benchmark/bench_variable_selection.jl
"""

using Printf
using Statistics
using Random
using LinearAlgebra

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SpectralPredict: uve_selection, spa_selection, ipls_selection, uve_spa_selection

#=============================================================================
    Benchmark Utilities
=============================================================================#

"""
    generate_synthetic_spectral_data(n_samples, n_wavelengths, n_informative)

Generate synthetic spectral data with known informative variables.

# Arguments
- `n_samples::Int`: Number of samples
- `n_wavelengths::Int`: Number of wavelengths (features)
- `n_informative::Int`: Number of truly informative wavelengths

# Returns
- `X::Matrix{Float64}`: Spectral data (n_samples × n_wavelengths)
- `y::Vector{Float64}`: Target values
- `true_indices::Vector{Int}`: Indices of truly informative wavelengths
"""
function generate_synthetic_spectral_data(n_samples::Int, n_wavelengths::Int, n_informative::Int)
    Random.seed!(42)

    # Create informative wavelengths with structure
    true_indices = sort(rand(1:n_wavelengths, n_informative))

    # Generate spectra with baseline + peaks + noise
    X = zeros(n_samples, n_wavelengths)

    for i in 1:n_samples
        # Baseline
        baseline = 1.0 + 0.1 * randn()

        # Add smooth spectral curve
        wavelengths = range(0, 2π, length=n_wavelengths)
        smooth_curve = sin.(wavelengths .+ randn() * 0.5) .* 0.3

        # Add informative peaks at true indices
        informative_signal = zeros(n_wavelengths)
        for idx in true_indices
            # Gaussian peak centered at informative wavelength
            width = 10.0
            for j in 1:n_wavelengths
                dist = (j - idx)^2
                informative_signal[j] += exp(-dist / (2 * width^2)) * randn()
            end
        end

        # Add noise
        noise = randn(n_wavelengths) .* 0.05

        # Combine
        X[i, :] = baseline .+ smooth_curve .+ informative_signal .+ noise
    end

    # Generate target based on informative wavelengths
    β = randn(n_informative)
    y = X[:, true_indices] * β .+ randn(n_samples) .* 0.1

    return X, y, true_indices
end

"""
    run_benchmark(func, X, y, args...; n_warmup=2, n_runs=5, name="")

Run benchmark with warmup and multiple iterations.

# Returns
- Named tuple with timing statistics and memory estimate
"""
function run_benchmark(func, X, y, args...; n_warmup=2, n_runs=5, name="")
    # Warmup runs (JIT compilation)
    println("  Warming up $name...")
    for i in 1:n_warmup
        func(X, y, args...)
    end

    # Timed runs
    println("  Running $name ($n_runs iterations)...")
    times = Float64[]

    for i in 1:n_runs
        t_start = time_ns()
        result = func(X, y, args...)
        t_end = time_ns()

        elapsed = (t_end - t_start) / 1e9  # Convert to seconds
        push!(times, elapsed)
    end

    # Memory estimate (rough)
    mem_start = @allocated func(X, y, args...)
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

#=============================================================================
    Benchmark Tests
=============================================================================#

function benchmark_uve(scale_name, X, y)
    println("\n$scale_name - UVE Selection")
    println("="^60)

    # UVE parameters
    n_components = min(10, size(X, 2) ÷ 10)
    cv_folds = 5
    noise_factor = 1.0

    stats = run_benchmark(uve_selection, X, y, n_components, cv_folds, noise_factor,
                         n_warmup=2, n_runs=5, name="UVE")

    print_benchmark_results("UVE Selection", stats, "6-10x vs Python")

    return stats
end

function benchmark_spa(scale_name, X, y)
    println("\n$scale_name - SPA Selection")
    println("="^60)

    # SPA parameters
    n_vars = min(50, size(X, 2) ÷ 10)

    stats = run_benchmark(spa_selection, X, y, n_vars,
                         n_warmup=2, n_runs=5, name="SPA")

    print_benchmark_results("SPA Selection", stats, "10-20x vs Python (parallelized)")

    return stats
end

function benchmark_ipls(scale_name, X, y)
    println("\n$scale_name - iPLS Selection")
    println("="^60)

    # iPLS parameters
    n_intervals = 10
    n_components = min(5, size(X, 2) ÷ 50)
    cv_folds = 5

    stats = run_benchmark(ipls_selection, X, y, n_intervals, n_components, cv_folds,
                         n_warmup=2, n_runs=3, name="iPLS")  # Fewer runs (slower)

    print_benchmark_results("iPLS Selection", stats, "8-12x vs Python")

    return stats
end

function benchmark_uve_spa(scale_name, X, y)
    println("\n$scale_name - UVE-SPA Selection")
    println("="^60)

    # UVE-SPA parameters
    n_vars = min(30, size(X, 2) ÷ 20)
    n_components = min(10, size(X, 2) ÷ 10)
    cv_folds = 5
    noise_factor = 1.0

    stats = run_benchmark(uve_spa_selection, X, y, n_vars, n_components, cv_folds, noise_factor,
                         n_warmup=2, n_runs=5, name="UVE-SPA")

    print_benchmark_results("UVE-SPA Selection", stats, "8-15x vs Python")

    return stats
end

#=============================================================================
    Main Benchmark Suite
=============================================================================#

function main()
    println("="^80)
    println("Variable Selection Performance Benchmarks")
    println("="^80)
    println("\nJulia Version: $(VERSION)")
    println("Threads Available: $(Threads.nthreads())")
    println("Date: $(now())")
    println()

    # Define test scales
    test_scales = [
        ("Small (100 × 500)", 100, 500, 20),
        ("Medium (300 × 1500)", 300, 1500, 50),
        ("Large (1000 × 2151)", 1000, 2151, 100)
    ]

    results = Dict()

    for (scale_name, n_samples, n_wavelengths, n_informative) in test_scales
        println("\n" * "="^80)
        println("Scale: $scale_name")
        println("="^80)

        # Generate data
        println("Generating synthetic data...")
        X, y, true_indices = generate_synthetic_spectral_data(n_samples, n_wavelengths, n_informative)
        println("  Samples: $n_samples")
        println("  Wavelengths: $n_wavelengths")
        println("  Informative wavelengths: $n_informative")

        # Run benchmarks
        scale_results = Dict()
        scale_results["uve"] = benchmark_uve(scale_name, X, y)
        scale_results["spa"] = benchmark_spa(scale_name, X, y)
        scale_results["ipls"] = benchmark_ipls(scale_name, X, y)
        scale_results["uve_spa"] = benchmark_uve_spa(scale_name, X, y)

        results[scale_name] = scale_results
    end

    # Summary Report
    println("\n\n" * "="^80)
    println("SUMMARY REPORT")
    println("="^80)

    println("\nMean Execution Times (seconds):")
    println("-"^80)
    @printf("%-25s %-12s %-12s %-12s %-12s\n", "Scale", "UVE", "SPA", "iPLS", "UVE-SPA")
    println("-"^80)

    for (scale_name, n_samples, n_wavelengths, n_informative) in test_scales
        scale_results = results[scale_name]
        @printf("%-25s %-12.4f %-12.4f %-12.4f %-12.4f\n",
                scale_name,
                scale_results["uve"].mean,
                scale_results["spa"].mean,
                scale_results["ipls"].mean,
                scale_results["uve_spa"].mean)
    end

    println("\n" * "="^80)
    println("Target Speedups (vs Python):")
    println("  - UVE: 6-10x faster")
    println("  - SPA: 10-20x faster (with parallelization)")
    println("  - iPLS: 8-12x faster")
    println("  - UVE-SPA: 8-15x faster")
    println()
    println("Notes:")
    println("  - Run with multiple threads for best SPA performance")
    println("  - Use: julia --threads=auto benchmark/bench_variable_selection.jl")
    println("  - Warmup runs excluded from timing (JIT compilation)")
    println("  - Statistics based on 5 iterations per test (3 for iPLS)")
    println("="^80)
end

# Run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
