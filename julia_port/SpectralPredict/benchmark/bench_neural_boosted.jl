"""
    bench_neural_boosted.jl

Performance benchmarks for Neural Boosted Regressor.

Tests realistic spectroscopy data sizes:
- Small: 100 samples × 50 features (rapid prototyping)
- Medium: 300 samples × 150 features (typical after variable selection)
- Large: 1000 samples × 300 features (comprehensive modeling)

Expected speedups from implementation plan:
- Neural Boosted training: 2-3x faster than Python sklearn
- Prediction: 3-5x faster
- Feature importance: 2-3x faster

Tests multiple configurations:
- Different numbers of estimators (50, 100, 200)
- Different hidden layer sizes (3, 5, 10)
- With and without early stopping

Run with: julia --threads=auto benchmark/bench_neural_boosted.jl
"""

using Printf
using Statistics
using Random
using LinearAlgebra
using Dates

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using SpectralPredict.NeuralBoosted

#=============================================================================
    Benchmark Utilities
=============================================================================#

"""
    generate_regression_data(n_samples, n_features, noise_level=0.1)

Generate synthetic regression data with nonlinear relationships.
"""
function generate_regression_data(n_samples::Int, n_features::Int, noise_level::Float64=0.1)
    Random.seed!(42)

    # Generate features
    X = randn(n_samples, n_features)

    # Nonlinear target (good for neural boosted)
    y = zeros(n_samples)
    for i in 1:n_samples
        # Nonlinear combination of features
        y[i] = sin(X[i, 1]) + X[i, 2]^2 + tanh(X[i, min(3, n_features)])

        # Add interactions
        if n_features >= 5
            y[i] += X[i, 4] * X[i, 5]
        end
    end

    # Add noise
    y .+= randn(n_samples) .* noise_level

    return X, y
end

"""
    run_benchmark(func, args...; n_warmup=1, n_runs=3, name="")

Run benchmark with warmup and multiple iterations.

Note: Fewer runs for neural network training (slower).
"""
function run_benchmark(func, args...; n_warmup=1, n_runs=3, name="")
    # Warmup runs (JIT compilation)
    println("  Warming up $name...")
    for i in 1:n_warmup
        func(args...)
    end

    # Timed runs
    println("  Running $name ($n_runs iterations)...")
    times = Float64[]
    results = []

    for i in 1:n_runs
        t_start = time_ns()
        result = func(args...)
        t_end = time_ns()

        elapsed = (t_end - t_start) / 1e9
        push!(times, elapsed)
        push!(results, result)
    end

    # Memory estimate
    mem_start = @allocated func(args...)
    mem_mb = mem_start / (1024^2)

    return (
        mean = mean(times),
        std = std(times),
        min = minimum(times),
        max = maximum(times),
        mem_mb = mem_mb,
        results = results
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

function benchmark_training(scale_name, X, y, n_estimators, hidden_size, early_stopping)
    config_str = "n_est=$n_estimators, hidden=$hidden_size, early_stop=$early_stopping"
    println("\n$scale_name - Training ($config_str)")
    println("="^60)

    # Split data
    n_train = Int(floor(size(X, 1) * 0.7))
    X_train = X[1:n_train, :]
    y_train = y[1:n_train]

    # Training function
    function train_model()
        model = NeuralBoostedRegressor(
            n_estimators = n_estimators,
            learning_rate = 0.1,
            hidden_layer_size = hidden_size,
            activation = "tanh",
            early_stopping = early_stopping,
            validation_fraction = 0.15,
            verbose = 0,
            random_state = 42
        )
        NeuralBoosted.fit!(model, X_train, y_train)
        return model
    end

    stats = run_benchmark(train_model,
                         n_warmup=1, n_runs=3, name="Training")

    print_benchmark_results("Neural Boosted Training", stats, "2-3x vs Python")

    # Return trained model from last run
    return stats, stats.results[end]
end

function benchmark_prediction(scale_name, model, X_test)
    println("\n$scale_name - Prediction")
    println("="^60)

    function predict_all()
        return NeuralBoosted.predict(model, X_test)
    end

    stats = run_benchmark(predict_all,
                         n_warmup=2, n_runs=10, name="Prediction")

    print_benchmark_results("Neural Boosted Prediction", stats, "3-5x vs Python")

    return stats
end

function benchmark_feature_importance(scale_name, model)
    println("\n$scale_name - Feature Importance")
    println("="^60)

    function compute_importance()
        return NeuralBoosted.feature_importances(model)
    end

    stats = run_benchmark(compute_importance,
                         n_warmup=2, n_runs=5, name="Feature Importance")

    print_benchmark_results("Feature Importance", stats, "2-3x vs Python")

    return stats
end

#=============================================================================
    Configuration Tests
=============================================================================#

function test_configurations(X, y)
    println("\n" * "="^80)
    println("CONFIGURATION COMPARISON TEST")
    println("="^80)

    configs = [
        (50, 3, true, "Fast (50 est, 3 hidden, early stop)"),
        (100, 3, false, "Standard (100 est, 3 hidden, no early stop)"),
        (100, 5, false, "Complex (100 est, 5 hidden, no early stop)"),
    ]

    # Split data
    n_train = Int(floor(size(X, 1) * 0.7))
    X_train = X[1:n_train, :]
    y_train = y[1:n_train]

    println("\nTraining with different configurations...")
    println()

    results = []

    for (n_est, hidden, early_stop, desc) in configs
        println("Configuration: $desc")
        println("-"^60)

        # Train
        function train_config()
            model = NeuralBoostedRegressor(
                n_estimators = n_est,
                learning_rate = 0.1,
                hidden_layer_size = hidden,
                activation = "tanh",
                early_stopping = early_stop,
                validation_fraction = 0.15,
                verbose = 0,
                random_state = 42
            )
            NeuralBoosted.fit!(model, X_train, y_train)
            return model
        end

        # Warmup
        train_config()

        # Time
        t_start = time_ns()
        model = train_config()
        t_end = time_ns()
        elapsed = (t_end - t_start) / 1e9

        @printf("  Training time: %.4f s\n", elapsed)
        @printf("  Estimators trained: %d\n", length(model.estimators_))
        println()

        push!(results, (desc, elapsed, length(model.estimators_)))
    end

    return results
end

#=============================================================================
    Main Benchmark Suite
=============================================================================#

function main()
    println("="^80)
    println("Neural Boosted Regressor Performance Benchmarks")
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
        X, y = generate_regression_data(n_samples, n_features)
        println("  Samples: $n_samples")
        println("  Features: $n_features")

        # Split for testing
        n_train = Int(floor(n_samples * 0.7))
        X_test = X[(n_train+1):end, :]

        # Run benchmarks with standard configuration
        scale_results = Dict()

        # Training
        n_estimators = 100
        hidden_size = 3
        early_stopping = true

        train_stats, trained_model = benchmark_training(
            scale_name, X, y, n_estimators, hidden_size, early_stopping
        )
        scale_results["training"] = train_stats

        # Prediction
        pred_stats = benchmark_prediction(scale_name, trained_model, X_test)
        scale_results["prediction"] = pred_stats

        # Feature importance
        feat_stats = benchmark_feature_importance(scale_name, trained_model)
        scale_results["feature_importance"] = feat_stats

        results[scale_name] = scale_results
    end

    # Configuration comparison (medium dataset)
    println("\n" * "="^80)
    X_med, y_med = generate_regression_data(300, 150)
    config_results = test_configurations(X_med, y_med)

    # Summary Report
    println("\n\n" * "="^80)
    println("SUMMARY REPORT")
    println("="^80)

    println("\nMean Execution Times (seconds):")
    println("-"^80)
    @printf("%-25s %-15s %-15s %-15s\n",
            "Scale", "Training", "Prediction", "Feature Imp.")
    println("-"^80)

    for (scale_name, n_samples, n_features) in test_scales
        scale_results = results[scale_name]
        @printf("%-25s %-15.4f %-15.4f %-15.4f\n",
                scale_name,
                scale_results["training"].mean,
                scale_results["prediction"].mean,
                scale_results["feature_importance"].mean)
    end

    println("\n\nConfiguration Comparison (Medium dataset):")
    println("-"^80)
    for (desc, time, n_est) in config_results
        @printf("%-50s %.4f s (%d estimators)\n", desc, time, n_est)
    end

    println("\n" * "="^80)
    println("Target Speedups (vs Python sklearn):")
    println("  - Training: 2-3x faster")
    println("  - Prediction: 3-5x faster")
    println("  - Feature importance: 2-3x faster")
    println()
    println("Notes:")
    println("  - Uses Flux.jl for neural network implementation")
    println("  - Early stopping can significantly reduce training time")
    println("  - Hidden layer size impacts both speed and accuracy")
    println("  - Keep hidden_layer_size low (3-5) for weak learners")
    println("  - Warmup runs excluded from timing (JIT compilation)")
    println("="^80)
end

# Run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
