"""
    cli.jl

Command-line interface for SpectralPredict.jl

Provides a user-friendly command-line interface for running spectral prediction analyses.

# Usage

```bash
julia --project=. src/cli.jl \\
    --spectra-dir data/spectra \\
    --reference data/reference.csv \\
    --id-column sample_id \\
    --target protein_pct \\
    --output results.csv \\
    --models PLS,Ridge,RandomForest \\
    --preprocessing snv,deriv \\
    --enable-subsets
```
"""

using ArgParse
using DataFrames
using CSV

# Include main module
include("SpectralPredict.jl")
using .SpectralPredict

"""
    parse_commandline()

Parse command-line arguments for SpectralPredict.
"""
function parse_commandline()
    s = ArgParseSettings(
        description = "SpectralPredict.jl - Spectral prediction with automated model search",
        version = "0.1.0",
        add_version = true
    )

    @add_arg_table! s begin
        "--spectra-dir"
            help = "Directory containing spectral data files (CSV or SPC)"
            arg_type = String
            required = true

        "--reference"
            help = "CSV file with reference values (targets)"
            arg_type = String
            required = true

        "--id-column"
            help = "Column name for sample IDs in reference file"
            arg_type = String
            required = true

        "--target"
            help = "Column name for target variable in reference file"
            arg_type = String
            required = true

        "--output"
            help = "Output CSV file for results"
            arg_type = String
            default = "spectral_predict_results.csv"

        "--task-type"
            help = "Task type: regression or classification"
            arg_type = String
            default = "regression"

        "--models"
            help = "Comma-separated list of models (PLS,Ridge,Lasso,ElasticNet,RandomForest,MLP)"
            arg_type = String
            default = "PLS,Ridge,Lasso,RandomForest,MLP"

        "--preprocessing"
            help = "Comma-separated list of preprocessing methods (raw,snv,deriv)"
            arg_type = String
            default = "raw,snv,deriv"

        "--derivative-orders"
            help = "Comma-separated derivative orders (1,2)"
            arg_type = String
            default = "1,2"

        "--derivative-window"
            help = "Savitzky-Golay window length"
            arg_type = Int
            default = 17

        "--derivative-polyorder"
            help = "Savitzky-Golay polynomial order"
            arg_type = Int
            default = 3

        "--enable-subsets"
            help = "Enable variable and region subsets"
            action = :store_true

        "--variable-counts"
            help = "Comma-separated variable counts for subsets (10,20,50,100,250)"
            arg_type = String
            default = "10,20,50,100,250"

        "--n-top-regions"
            help = "Number of top regions to analyze"
            arg_type = Int
            default = 5

        "--n-folds"
            help = "Number of cross-validation folds"
            arg_type = Int
            default = 5

        "--lambda-penalty"
            help = "Complexity penalty weight (0.0-1.0)"
            arg_type = Float64
            default = 0.15

        "--file-extension"
            help = "File extension for spectral files (.csv or .spc)"
            arg_type = String
            default = ".csv"

        "--verbose"
            help = "Enable verbose output"
            action = :store_true
    end

    return parse_args(s)
end

"""
    parse_list(s::String, type::Type)

Parse comma-separated string into vector of given type.
"""
function parse_list(s::String, type::Type)
    items = strip.(split(s, ','))
    if type == String
        return items
    elseif type == Int
        return parse.(Int, items)
    else
        error("Unsupported type: $type")
    end
end

"""
    main()

Main entry point for CLI.
"""
function main()
    # Parse arguments
    args = parse_commandline()

    verbose = args["verbose"]

    if verbose
        println("=" ^ 70)
        println("SpectralPredict.jl - Spectral Prediction Analysis")
        println("=" ^ 70)
        println()
    end

    # Parse list arguments
    models = parse_list(args["models"], String)
    preprocessing = parse_list(args["preprocessing"], String)
    derivative_orders = parse_list(args["derivative-orders"], Int)
    variable_counts = parse_list(args["variable-counts"], Int)

    # Load data
    if verbose
        println("Loading data...")
        println("  Spectra directory: $(args["spectra-dir"])")
        println("  Reference file: $(args["reference"])")
        println("  ID column: $(args["id-column"])")
        println("  Target column: $(args["target"])")
        println()
    end

    try
        X, y, wavelengths, sample_ids = load_spectral_dataset(
            args["spectra-dir"],
            args["reference"],
            args["id-column"],
            args["target"],
            file_extension=args["file-extension"]
        )

        if verbose
            println("Data loaded successfully:")
            println("  Samples: $(size(X, 1))")
            println("  Wavelengths: $(size(X, 2))")
            println("  Range: $(wavelengths[1]) - $(wavelengths[end]) nm")
            println()
        end

        # Run search
        if verbose
            println("Running hyperparameter search...")
            println("  Task type: $(args["task-type"])")
            println("  Models: $(join(models, ", "))")
            println("  Preprocessing: $(join(preprocessing, ", "))")
            println("  CV folds: $(args["n-folds"])")
            println("  Subsets enabled: $(args["enable-subsets"])")
            println()
            println("This may take several minutes...")
            println()
        end

        results = run_search(
            X, y, wavelengths,
            task_type=args["task-type"],
            models=models,
            preprocessing=preprocessing,
            derivative_orders=derivative_orders,
            derivative_window=args["derivative-window"],
            derivative_polyorder=args["derivative-polyorder"],
            enable_variable_subsets=args["enable-subsets"],
            variable_counts=variable_counts,
            enable_region_subsets=args["enable-subsets"],
            n_top_regions=args["n-top-regions"],
            n_folds=args["n-folds"],
            lambda_penalty=args["lambda-penalty"]
        )

        # Save results
        if verbose
            println("Saving results to: $(args["output"])")
        end

        save_results(results, args["output"])

        # Display summary
        if verbose
            println()
            println("=" ^ 70)
            println("Analysis Complete!")
            println("=" ^ 70)
            println()
            println("Total configurations tested: $(nrow(results))")
            println()
            println("Top 10 models:")
            println("-" ^ 70)

            top_10 = first(results, 10)
            for i in 1:nrow(top_10)
                row = top_10[i, :]
                println("Rank $(row.Rank): $(row.Model) + $(row.Preprocess) ($(row.SubsetTag))")
                if args["task-type"] == "regression"
                    println("  R² = $(round(row.R2, digits=4)), RMSE = $(round(row.RMSE, digits=4))")
                else
                    println("  Accuracy = $(round(row.Accuracy, digits=4)), AUC = $(round(row.ROC_AUC, digits=4))")
                end
                println()
            end

            println("Full results saved to: $(args["output"])")
            println()
        end

        # Exit successfully
        return 0

    catch e
        println("ERROR: Analysis failed")
        println()
        println("Error message:")
        println(e)
        println()

        if verbose
            println("Stack trace:")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end

        return 1
    end
end

# Run main if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
