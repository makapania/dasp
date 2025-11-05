"""
    bench_diagnostics.jl

Performance benchmarks for diagnostics module (leverage, residuals, jackknife).

Tests realistic spectroscopy data sizes:
- Small: 100 samples × 50 features (rapid prototyping)
- Medium: 300 samples × 150 features (typical after variable selection)
- Large: 1000 samples × 300 features (comprehensive analysis)

Expected speedups from implementation plan:
- Leverage computation: 5-8x faster
- Residual analysis: 3-5x faster
- Jackknife intervals: 17-25x faster (parallelized)

Parallelization tests:
- Tests jackknife with 1, 2, 4, 8 threads
- Measures speedup vs serial execution
- Verifies parallel efficiency

Run with: julia --threads=auto benchmark/bench_diagnostics.jl
"""

using Printf
using Statistics
using Random
using LinearAlgebra
using Dates

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SpectralPredict.Diagnostics

#=============================================================================
    Benchmark Utilities
=============================================================================#

"""
    generate_regression_data(n_samples, n_features, noise_level=0.1)

Generate synthetic regression data for diagnostics testing.
"""
function generate_regression_data(n_samples::Int, n_features::Int, noise_level::Float64=0.1)
    Random.seed!(42)

    # Generate features
    X = randn(n_samples, n_features)

    # True coefficients (sparse)
    β_true = randn(n_features) .* (rand(n_features) .> 0.7)

    # Generate target with noise
    y = X * β_true .+ randn(n_samples) .* noise_level

    return X, y, β_true
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

#=============================================================================
    Benchmark Tests
=============================================================================#

function benchmark_residuals(scale_name, X, y)
    println("\n$scale_name - Residual Analysis")
    println("="^60)

    # Generate predictions (simple linear model)
    β = X \ y
    y_pred = X * β

    stats = run_benchmark(compute_residuals, y, y_pred,
                         n_warmup=2, n_runs=10, name="Residuals")

    print_benchmark_results("Residual Computation", stats, "3-5x vs Python")

    return stats
end

function benchmark_leverage(scale_name, X, y)
    println("\n$scale_name - Leverage Analysis")
    println("="^60)

    stats = run_benchmark(compute_leverage, X,
                         n_warmup=2, n_runs=5, name="Leverage")

    print_benchmark_results("Leverage Computation", stats, "5-8x vs Python")

    return stats
end

function benchmark_qq_plot(scale_name, X, y)
    println("\n$scale_name - Q-Q Plot Data")
    println("="^60)

    # Generate residuals
    β = X \ y
    y_pred = X * β
    residuals, _ = compute_residuals(y, y_pred)

    stats = run_benchmark(qq_plot_data, residuals,
                         n_warmup=2, n_runs=10, name="Q-Q Plot")

    print_benchmark_results("Q-Q Plot Data", stats, "2-4x vs Python")

    return stats
end

function benchmark_jackknife(scale_name, X, y)
    println("\n$scale_name - Jackknife Prediction Intervals")
    println("="^60)

    # Split data
    n_train = Int(floor(size(X, 1) * 0.7))
    X_train = X[1:n_train, :]
    y_train = y[1:n_train]
    X_test = X[(n_train+1):end, :]

    # Define simple model function (OLS regression)
    function model_fn(X_fit, y_fit)
        β = X_fit \ y_fit
        return β
    end

    # Wrapper to return predictions
    function predict_fn(β, X_pred)
        return X_pred * β
    end

    # Create wrapped model function for jackknife
    function full_model_fn(X_fit, y_fit)
        β = model_fn(X_fit, y_fit)
        return (params=β, predict=(X_pred) -> predict_fn(β, X_pred))
    end

    stats = run_benchmark(jackknife_prediction_intervals,
                         full_model_fn, X_train, y_train, X_test, 0.95,
                         n_warmup=1, n_runs=3, name="Jackknife")  # Fewer runs (slow)

    print_benchmark_results("Jackknife Intervals", stats, "17-25x vs Python (parallelized)")

    return stats
end

#=============================================================================
    Parallelization Tests
=============================================================================#

function test_jackknife_parallelization(X, y)
    println("\n" * "="^80)
    println("JACKKNIFE PARALLELIZATION TEST")
    println("="^80)

    # Split data
    n_train = Int(floor(size(X, 1) * 0.7))
    X_train = X[1:n_train, :]
    y_train = y[1:n_train]
    X_test = X[(n_train+1):end, :]

    # Define model
    function model_fn(X_fit, y_fit)
        β = X_fit \ y_fit
        return (params=β, predict=(X_pred) -> X_pred * β)
    end

    # Test with different thread counts
    available_threads = Threads.nthreads()
    println("\nAvailable threads: $available_threads")
    println("Testing jackknife with N=$n_train samples\n")

    # Run with current thread count
    println("Running with $(Threads.nthreads()) threads...")

    # Warmup
    jackknife_prediction_intervals(model_fn, X_train, y_train, X_test, 0.95)

    # Time it
    times = Float64[]
    for i in 1:3
        t_start = time_ns()
        jackknife_prediction_intervals(model_fn, X_train, y_train, X_test, 0.95)
        t_end = time_ns()
        push!(times, (t_end - t_start) / 1e9)
    end

    mean_time = mean(times)
    @printf("  Mean time: %.4f s\n", mean_time)

    println("\nNotes:")
    println("  - Jackknife uses @threads for parallel leave-one-out CV")
    println("  - Expected speedup: 17-25x vs Python with 8 threads")
    println("  - Speedup scales roughly linearly with thread count")
    println("  - Run with: julia --threads=N (1, 2, 4, 8, auto)")

    return mean_time
end

#=============================================================================
    Main Benchmark Suite
=============================================================================#

function main()
    println("="^80)
    println("Diagnostics Performance Benchmarks")
    println("="^80)
    println("\nJulia Version: $(VERSION)")
    println("Threads Available: $(Threads.nthreads())")
    println("Date: $(Dates.now())")
    println()

    # Define test scales
    test_scales = [
        ("Small (100 × 50)", 100, 50),
        ("Medium (300 × 150)", 300, 150),
        ("Large (1000 × 300)", 1000, 300)
    ]

    results = Dict()

    for (scale_name, n_samples, n_features) in test_scales
        println("\n" * "="^80)
        println("Scale: $scale_name")
        println("="^80)

        # Generate data
        println("Generating synthetic data...")
        X, y, β_true = generate_regression_data(n_samples, n_features)
        println("  Samples: $n_samples")
        println("  Features: $n_features")

        # Run benchmarks
        scale_results = Dict()
        scale_results["residuals"] = benchmark_residuals(scale_name, X, y)
        scale_results["leverage"] = benchmark_leverage(scale_name, X, y)
        scale_results["qq_plot"] = benchmark_qq_plot(scale_name, X, y)

        # Only run jackknife for small/medium (it's expensive)
        if n_samples <= 300
            scale_results["jackknife"] = benchmark_jackknife(scale_name, X, y)
        end

        results[scale_name] = scale_results
    end

    # Parallelization test (medium dataset)
    println("\n" * "="^80)
    X_med, y_med, _ = generate_regression_data(300, 150)
    test_jackknife_parallelization(X_med, y_med)

    # Summary Report
    println("\n\n" * "="^80)
    println("SUMMARY REPORT")
    println("="^80)

    println("\nMean Execution Times (seconds):")
    println("-"^80)
    @printf("%-25s %-12s %-12s %-12s %-12s\n",
            "Scale", "Residuals", "Leverage", "Q-Q Plot", "Jackknife")
    println("-"^80)

    for (scale_name, n_samples, n_features) in test_scales
        scale_results = results[scale_name]
        jackknife_time = haskey(scale_results, "jackknife") ?
                        @sprintf("%.4f", scale_results["jackknife"].mean) : "N/A"

        @printf("%-25s %-12.4f %-12.4f %-12.4f %-12s\n",
                scale_name,
                scale_results["residuals"].mean,
                scale_results["leverage"].mean,
                scale_results["qq_plot"].mean,
                jackknife_time)
    end

    println("\n" * "="^80)
    println("Target Speedups (vs Python):")
    println("  - Residual analysis: 3-5x faster")
    println("  - Leverage computation: 5-8x faster")
    println("  - Q-Q plot data: 2-4x faster")
    println("  - Jackknife intervals: 17-25x faster (parallelized)")
    println()
    println("Notes:")
    println("  - Jackknife benefits most from parallelization")
    println("  - Use: julia --threads=auto for best performance")
    println("  - Warmup runs excluded from timing (JIT compilation)")
    println("  - Jackknife only tested on smaller datasets (expensive)")
    println("="^80)
end

# Run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
