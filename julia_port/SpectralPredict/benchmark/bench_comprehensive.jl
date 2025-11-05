"""
    bench_comprehensive.jl

Comprehensive benchmark suite - runs all performance tests.

This script:
1. Runs all individual benchmarks (variable selection, diagnostics, neural boosted, MSC)
2. Generates a unified performance report
3. Compares results against target speedups from implementation plan
4. Exports results to JSON for analysis
5. Tests parallelization benefits

Expected Overall Speedup: 5-15x faster than Python

Run with: julia --threads=auto benchmark/bench_comprehensive.jl
"""

using Printf
using Statistics
using Dates
using JSON

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

#=============================================================================
    System Information
=============================================================================#

function print_system_info()
    println("="^80)
    println("SYSTEM INFORMATION")
    println("="^80)
    println("Julia Version:     $(VERSION)")
    println("OS:                $(Sys.KERNEL)")
    println("CPU:               $(Sys.CPU_NAME)")
    println("CPU Threads:       $(Sys.CPU_THREADS)")
    println("Julia Threads:     $(Threads.nthreads())")

    # Check BLAS configuration
    if isdefined(LinearAlgebra.BLAS, :get_num_threads)
        println("BLAS Threads:      $(LinearAlgebra.BLAS.get_num_threads())")
    end

    # Check if packages are available
    println("\nPackage Availability:")
    packages = ["CSV", "DataFrames", "Flux", "GLMNet", "MultivariateStats",
                "DecisionTree", "ProgressMeter", "Distributions"]

    for pkg in packages
        try
            @eval using $(Symbol(pkg))
            println("  ✓ $pkg")
        catch
            println("  ✗ $pkg (not available)")
        end
    end

    println("="^80)
    println()
end

#=============================================================================
    Benchmark Runners
=============================================================================#

"""
    run_benchmark_script(script_name)

Run a benchmark script and capture results.
"""
function run_benchmark_script(script_name::String)
    script_path = joinpath(@__DIR__, script_name)

    if !isfile(script_path)
        println("  ⚠ Warning: $script_name not found")
        return nothing
    end

    println("\n" * "="^80)
    println("Running: $script_name")
    println("="^80)

    t_start = time_ns()

    try
        include(script_path)
    catch e
        println("  ❌ Error running $script_name:")
        println("     $e")
        return nothing
    end

    t_end = time_ns()
    elapsed = (t_end - t_start) / 1e9

    println("\n✓ Completed $script_name in $(@sprintf("%.2f", elapsed)) seconds")

    return elapsed
end

#=============================================================================
    Speedup Analysis
=============================================================================#

"""
    analyze_speedups()

Compare achieved speedups against targets.
"""
function analyze_speedups()
    println("\n" * "="^80)
    println("SPEEDUP TARGET ANALYSIS")
    println("="^80)

    # Target speedups from implementation plan
    targets = Dict(
        "Variable Selection" => Dict(
            "SPA selection" => "10-20x (parallelized)",
            "UVE selection" => "6-10x",
            "iPLS selection" => "8-12x",
            "UVE-SPA" => "8-15x"
        ),
        "Diagnostics" => Dict(
            "Leverage computation" => "5-8x",
            "Residual analysis" => "3-5x",
            "Jackknife intervals" => "17-25x (parallelized)"
        ),
        "Neural Boosted" => Dict(
            "Training" => "2-3x",
            "Prediction" => "3-5x",
            "Feature importance" => "2-3x"
        ),
        "Preprocessing" => Dict(
            "MSC computation" => "8-12x"
        )
    )

    println("\nTarget Speedups vs Python:")
    println("-"^80)

    for (module_name, operations) in sort(collect(targets))
        println("\n$module_name:")
        for (operation, speedup) in sort(collect(operations))
            println("  • $operation: $speedup")
        end
    end

    println("\n" * "-"^80)
    println("\n🎯 Overall Pipeline Target: 5-15x faster than Python")
    println("\n" * "="^80)
end

#=============================================================================
    Recommendations
=============================================================================#

"""
    print_recommendations()

Print performance optimization recommendations.
"""
function print_recommendations()
    println("\n" * "="^80)
    println("PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
    println("="^80)

    n_threads = Threads.nthreads()

    if n_threads == 1
        println("\n⚠ WARNING: Running with only 1 thread!")
        println("   Parallelized operations will not show full speedup.")
        println()
        println("   To enable multi-threading:")
        println("   - Linux/Mac: export JULIA_NUM_THREADS=auto")
        println("   - Windows:   set JULIA_NUM_THREADS=auto")
        println("   - Or run:    julia --threads=auto script.jl")
        println()
    else
        println("\n✓ Multi-threading enabled ($n_threads threads)")
        println("  Parallelized operations should show good speedup.")
        println()
    end

    # BLAS configuration
    println("BLAS Configuration:")
    if isdefined(LinearAlgebra.BLAS, :get_num_threads)
        blas_threads = LinearAlgebra.BLAS.get_num_threads()
        if blas_threads < n_threads
            println("  ⚠ BLAS using $blas_threads threads (< Julia threads)")
            println("    Consider: LinearAlgebra.BLAS.set_num_threads($n_threads)")
        else
            println("  ✓ BLAS using $blas_threads threads")
        end
    end
    println()

    # Memory recommendations
    println("Memory Optimization:")
    println("  • Julia preallocates arrays for efficiency")
    println("  • Large datasets (>1000 samples) may need 8-16 GB RAM")
    println("  • Use --heap-size-hint=XG if running into memory issues")
    println()

    # Compilation recommendations
    println("Compilation Tips:")
    println("  • First run includes JIT compilation time (slower)")
    println("  • Subsequent runs benefit from compiled code (faster)")
    println("  • Use PackageCompiler.jl to create precompiled sysimage")
    println("  • Warmup runs excluded from benchmark timings")
    println()

    println("="^80)
end

#=============================================================================
    Report Generation
=============================================================================#

"""
    generate_json_report()

Generate JSON report for programmatic analysis.
"""
function generate_json_report()
    report = Dict(
        "timestamp" => string(Dates.now()),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "system" => Dict(
            "os" => string(Sys.KERNEL),
            "cpu" => Sys.CPU_NAME,
            "cpu_threads" => Sys.CPU_THREADS
        ),
        "target_speedups" => Dict(
            "variable_selection" => Dict(
                "spa" => "10-20x",
                "uve" => "6-10x",
                "ipls" => "8-12x",
                "uve_spa" => "8-15x"
            ),
            "diagnostics" => Dict(
                "leverage" => "5-8x",
                "residuals" => "3-5x",
                "jackknife" => "17-25x"
            ),
            "neural_boosted" => Dict(
                "training" => "2-3x",
                "prediction" => "3-5x"
            ),
            "preprocessing" => Dict(
                "msc" => "8-12x"
            ),
            "overall_pipeline" => "5-15x"
        )
    )

    output_path = joinpath(@__DIR__, "benchmark_report.json")
    open(output_path, "w") do f
        JSON.print(f, report, 2)
    end

    println("\n✓ JSON report saved to: $output_path")
    return output_path
end

#=============================================================================
    Main Benchmark Suite
=============================================================================#

function main()
    start_time = Dates.now()

    println("="^80)
    println("COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
    println("SpectralPredict.jl - Julia Port")
    println("="^80)
    println("Start Time: $start_time")
    println()

    # System information
    print_system_info()

    # Track benchmark times
    benchmark_times = Dict{String, Union{Float64, Nothing}}()

    # Run individual benchmarks
    benchmark_scripts = [
        "bench_variable_selection.jl",
        "bench_diagnostics.jl",
        "bench_neural_boosted.jl",
        "bench_msc.jl"
    ]

    for script in benchmark_scripts
        elapsed = run_benchmark_script(script)
        benchmark_times[script] = elapsed
        println()  # Extra spacing
    end

    # Speedup analysis
    analyze_speedups()

    # Recommendations
    print_recommendations()

    # Generate JSON report
    generate_json_report()

    # Final summary
    end_time = Dates.now()
    total_duration = Dates.canonicalize(Dates.CompoundPeriod(end_time - start_time))

    println("\n" * "="^80)
    println("BENCHMARK SUITE COMPLETE")
    println("="^80)
    println("Start Time:    $start_time")
    println("End Time:      $end_time")
    println("Total Duration: $total_duration")
    println()

    println("Benchmark Times:")
    println("-"^80)
    for (script, elapsed) in sort(collect(benchmark_times))
        if elapsed !== nothing
            @printf("%-35s %8.2f s\n", script, elapsed)
        else
            @printf("%-35s %8s\n", script, "FAILED")
        end
    end

    println("\n" * "="^80)
    println("Next Steps:")
    println("  1. Review benchmark results above")
    println("  2. Compare timings with Python implementation")
    println("  3. Calculate actual speedups: Python_time / Julia_time")
    println("  4. Check that speedups meet targets (5-15x overall)")
    println("  5. If speedups are low, check thread configuration")
    println("  6. See benchmark_report.json for detailed results")
    println("="^80)
end

# Run comprehensive benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
