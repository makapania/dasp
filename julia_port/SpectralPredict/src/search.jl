"""
    search.jl

Hyperparameter search orchestration for spectral prediction models.

This is THE CORE MODULE that orchestrates the entire hyperparameter search with
preprocessing, models, cross-validation, and subset analysis. It implements the
EXACT algorithm from the Python version, including the critical fix for avoiding
double-preprocessing with derivative subsets.

Key Features:
- Full hyperparameter grid search across models and preprocessing methods
- Variable subset analysis (top-N feature selection)
- Region subset analysis (spectral region detection)
- Skip-preprocessing logic for derivative subsets (CRITICAL!)
- Composite scoring and ranking
- Progress tracking and reporting

Critical Implementation Notes:
- Preprocessing is applied ONCE at the beginning of each preprocessing config loop
- Variable/region subsets for derivatives use PREPROCESSED data with skip_preprocessing=true
- Variable/region subsets for raw/SNV use RAW data with skip_preprocessing=false
- This prevents the double-preprocessing bug that was fixed in Python on Oct 29, 2025
"""

using DataFrames
using Statistics
using ProgressMeter

# Note: All module files are included in parent SpectralPredict.jl module
# No need to re-include them here

using .Regions
using .Scoring

# Import variable selection functions
import .uve_selection, .spa_selection, .ipls_selection, .uve_spa_selection


# ============================================================================
# Main Search Function
# ============================================================================

"""
    run_search(
        X::Matrix{Float64},
        y::Vector{Float64},
        wavelengths::Vector{Float64};
        task_type::String="regression",
        models::Vector{String}=["PLS", "Ridge", "Lasso", "RandomForest", "MLP"],
        preprocessing::Vector{String}=["raw", "snv", "deriv"],
        derivative_orders::Vector{Int}=[1, 2],
        derivative_window::Int=17,
        derivative_polyorder::Int=3,
        enable_variable_subsets::Bool=true,
        variable_counts::Vector{Int}=[10, 20, 50, 100, 250],
        variable_selection_methods::Vector{String}=String[],
        enable_region_subsets::Bool=true,
        n_top_regions::Int=5,
        n_folds::Int=5,
        lambda_penalty::Float64=0.15
    )::DataFrame

Run comprehensive hyperparameter search with preprocessing, models, and subset analysis.

This is the main entry point for the spectral prediction search. It orchestrates:
1. Preprocessing configurations (raw, SNV, derivatives)
2. Model hyperparameter grids (PLS, Ridge, Lasso, RF, MLP)
3. Variable subset analysis (top-N feature selection)
4. Region subset analysis (spectral region detection)
5. Cross-validation and performance evaluation
6. Composite scoring and ranking

# Algorithm (CRITICAL - Matches Python exactly)

For each preprocessing configuration:
1. Apply preprocessing ONCE to get X_preprocessed
2. Compute region subsets on X_preprocessed (for this preprocessing only)

For each model:
    A. Full model (all features)
        - Run CV with preprocessing

    B. Variable subsets (for models with feature importance)
        - Fit model on FULL X_preprocessed to get importances
        - For each variable count (10, 20, 50, ...):
            - Select top-N features based on importances
            - If using derivatives: use X_preprocessed[:, top_indices], skip_preprocessing=true
            - If using raw/SNV: use X[:, top_indices], skip_preprocessing=false

    C. Region subsets (for ALL models)
        - For each region:
            - If using derivatives: use X_preprocessed[:, region_indices], skip_preprocessing=true
            - If using raw/SNV: use X[:, region_indices], skip_preprocessing=false

3. Compute composite scores and rank all results

# Arguments
- `X::Matrix{Float64}`: Spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples,)
- `wavelengths::Vector{Float64}`: Wavelength values for each feature (n_features,)
- `task_type::String`: Either "regression" or "classification" (default: "regression")
- `models::Vector{String}`: Model types to test (default: ["PLS", "Ridge", "Lasso", "RandomForest", "MLP"])
- `preprocessing::Vector{String}`: Preprocessing methods to test (default: ["raw", "snv", "deriv"])
  - Options: "raw", "snv", "deriv", "snv_deriv", "deriv_snv"
- `derivative_orders::Vector{Int}`: Derivative orders to test when "deriv" is included (default: [1, 2])
- `derivative_window::Int`: Window size for Savitzky-Golay filter (default: 17)
- `derivative_polyorder::Int`: Polynomial order for Savitzky-Golay filter (default: 3)
- `enable_variable_subsets::Bool`: Enable top-N variable subset analysis (default: true)
- `variable_counts::Vector{Int}`: Variable counts to test (default: [10, 20, 50, 100, 250])
- `variable_selection_methods::Vector{String}`: Variable selection methods to use (default: String[])
  - Options: "uve", "spa", "ipls", "uve_spa"
  - Empty array means use model-based feature importance (default behavior)
  - Variable selection happens BEFORE preprocessing for each method
- `enable_region_subsets::Bool`: Enable spectral region subset analysis (default: true)
- `n_top_regions::Int`: Number of top regions to analyze (default: 5)
- `n_folds::Int`: Number of cross-validation folds (default: 5)
- `lambda_penalty::Float64`: Complexity penalty weight for scoring (default: 0.15)

# Returns
- `DataFrame`: Results table with columns:
  - Model info: Model, Preprocess, Deriv, Window, Poly, LVs
  - Subset info: SubsetTag, n_vars, full_vars
  - Metrics: RMSE, R2 (regression) or Accuracy, ROC_AUC (classification)
  - Scoring: CompositeScore, Rank
  - All hyperparameters from model configs

# Examples
```julia
# Basic usage
X = rand(100, 200)
y = rand(100)
wavelengths = collect(400.0:2.0:798.0)

results = run_search(X, y, wavelengths)

# Custom configuration
results = run_search(
    X, y, wavelengths,
    task_type="regression",
    models=["PLS", "RandomForest"],
    preprocessing=["raw", "snv", "deriv"],
    derivative_orders=[1, 2],
    enable_variable_subsets=true,
    variable_counts=[10, 20, 50],
    enable_region_subsets=true,
    n_top_regions=10,
    n_folds=10,
    lambda_penalty=0.15
)

# Using variable selection methods
results = run_search(
    X, y, wavelengths,
    task_type="regression",
    models=["PLS", "Ridge"],
    preprocessing=["snv", "deriv"],
    enable_variable_subsets=true,
    variable_counts=[10, 20, 50, 100],
    variable_selection_methods=["uve", "spa", "ipls", "uve_spa"],
    enable_region_subsets=true,
    n_folds=5
)

# View top models
first(sort(results, :Rank), 10)
```

# Notes
- Results are sorted by Rank (1 = best model)
- Composite score is 90% performance + 10% complexity
- Variable subsets only for models with feature importance (PLS, RF, MLP)
- Region subsets run for ALL models
- Skip-preprocessing logic prevents double-preprocessing bug
"""
function run_search(
    X::Matrix{Float64},
    y::Vector{Float64},
    wavelengths::Vector{Float64};
    task_type::String="regression",
    models::Vector{String}=["PLS", "Ridge", "Lasso", "RandomForest", "MLP"],
    preprocessing::Vector{String}=["raw", "snv", "deriv"],
    derivative_orders::Vector{Int}=[1, 2],
    derivative_window::Int=17,
    derivative_polyorder::Int=3,
    enable_variable_subsets::Bool=true,
    variable_counts::Vector{Int}=[10, 20, 50, 100, 250],
    variable_selection_methods::Vector{String}=String[],
    enable_region_subsets::Bool=true,
    n_top_regions::Int=5,
    n_folds::Int=5,
    lambda_penalty::Float64=0.15
)::DataFrame

    # Validate inputs
    @assert task_type in ["regression", "classification"] "task_type must be 'regression' or 'classification'"
    @assert size(X, 1) == length(y) "X rows ($(size(X, 1))) must match y length ($(length(y)))"
    @assert size(X, 2) == length(wavelengths) "X columns ($(size(X, 2))) must match wavelengths length ($(length(wavelengths)))"
    @assert n_folds >= 2 "n_folds must be at least 2"
    @assert lambda_penalty >= 0.0 "lambda_penalty must be non-negative"

    n_samples, n_features = size(X)
    full_vars = n_features

    println("="^80)
    println("Spectral Prediction Hyperparameter Search")
    println("="^80)
    println("Task type: $task_type")
    println("Data: $n_samples samples × $n_features features")
    println("Wavelength range: $(minimum(wavelengths)) - $(maximum(wavelengths)) nm")
    println("Models: ", join(models, ", "))
    println("Preprocessing: ", join(preprocessing, ", "))
    println("CV folds: $n_folds")
    println("Variable subsets: $enable_variable_subsets")
    println("Region subsets: $enable_region_subsets")
    println()

    # Generate preprocessing configurations
    preprocess_configs = generate_preprocessing_configs(
        preprocessing,
        derivative_orders,
        derivative_window,
        derivative_polyorder
    )

    println("Preprocessing configurations: $(length(preprocess_configs))")
    for cfg in preprocess_configs
        if cfg["deriv"] !== nothing
            println("  - $(cfg["name"]) (deriv=$(cfg["deriv"]), window=$(cfg["window"]), poly=$(cfg["polyorder"]))")
        else
            println("  - $(cfg["name"])")
        end
    end
    println()

    # Results collection
    results = Dict{String, Any}[]

    # Progress tracking
    total_preprocess = length(preprocess_configs)
    total_models = sum(length(get_model_configs(m)) for m in models)
    total_configs = total_preprocess * total_models
    current_config = 0

    println("Total configurations to test: $total_configs")
    println()

    # Create progress meter
    progress = Progress(total_configs, desc="Running search: ", barlen=50)

    # Main search loop
    for (preprocess_idx, preprocess_cfg) in enumerate(preprocess_configs)
        prep_name = preprocess_cfg["name"]
        prep_display = if preprocess_cfg["deriv"] !== nothing
            "$(prep_name)_d$(preprocess_cfg["deriv"])"
        else
            prep_name
        end

        println()
        println("-"^80)
        println("Preprocessing [$preprocess_idx/$total_preprocess]: $prep_display")
        println("-"^80)

        # 1. Apply preprocessing ONCE to get preprocessed data
        X_preprocessed = apply_preprocessing(X, preprocess_cfg)
        n_features_preprocessed = size(X_preprocessed, 2)

        println("Features after preprocessing: $n_features_preprocessed")

        # 2. Compute region subsets on PREPROCESSED data (if enabled)
        region_subsets = Dict{String, Any}[]
        if enable_region_subsets
            try
                region_subsets = Regions.create_region_subsets(
                    X_preprocessed,
                    y,
                    wavelengths,
                    n_top_regions=n_top_regions
                )
                println("Region subsets identified: $(length(region_subsets))")
            catch e
                println("Warning: Could not compute region subsets for $prep_display: $e")
            end
        end

        # 3. For each model type
        for model_name in models
            # Get hyperparameter configurations for this model
            configs = get_model_configs(model_name)

            println("\n  Model: $model_name ($(length(configs)) configurations)")

            for config in configs
                current_config += 1

                # A. Full model (all features)
                result_full = run_single_config(
                    X, y,
                    model_name, config,
                    preprocess_cfg, task_type,
                    n_folds,
                    skip_preprocessing=false,
                    subset_tag="full",
                    n_vars=n_features_preprocessed,
                    full_vars=full_vars
                )
                # Only add result if training succeeded (not nothing)
                if !isnothing(result_full)
                    push!(results, result_full)
                end

                # Update progress
                next!(progress, showvalues=[
                    (:Preprocessing, prep_display),
                    (:Model, model_name),
                    (:Config, config),
                    (:Status, "Full model")
                ])

                # B. Variable subsets
                if enable_variable_subsets
                    # Determine which variable selection methods to use
                    methods_to_use = String[]

                    # If variable_selection_methods is specified, use those
                    if !isempty(variable_selection_methods)
                        methods_to_use = variable_selection_methods
                    elseif model_name in ["PLS", "RandomForest", "MLP"]
                        # Default: use model-based feature importance for compatible models
                        push!(methods_to_use, "importance")
                    end

                    # Process each variable selection method
                    for method in methods_to_use
                        # Get valid variable counts (must be less than available features)
                        valid_counts = [n for n in variable_counts if n < n_features_preprocessed]

                        if isempty(valid_counts)
                            continue
                        end

                        println("    → Variable selection ($method): testing $(length(valid_counts)) counts")

                        for n_top in valid_counts
                            local importances::Vector{Float64}
                            local selected_indices::Vector{Int}
                            local subset_tag::String
                            local selection_succeeded = false

                            # Apply variable selection method (wrapped in try-catch for safety)
                            try
                                if method == "importance"
                                # Model-based feature importance (original method)
                                # Wrap in try-catch to handle models that fail on small datasets
                                local model_trained = false
                                try
                                    model = build_model(model_name, config, task_type)
                                    fit_model!(model, X_preprocessed, y)
                                    importances = get_feature_importances(model, model_name, X_preprocessed, y)
                                    selected_indices = sortperm(importances, rev=true)[1:n_top]
                                    subset_tag = "top$(n_top)"
                                    model_trained = true
                                catch e
                                    @warn "Failed to compute feature importances for $model_name: $(sprint(showerror, e)). Skipping variable selection for this model/count."
                                end

                                # Skip this iteration if model training failed
                                if !model_trained
                                    continue
                                end

                            elseif method == "uve"
                                # UVE selection on preprocessed data
                                importances = uve_selection(
                                    X_preprocessed, y;
                                    cutoff_multiplier=1.0,
                                    cv_folds=n_folds
                                )
                                # Select top n_top variables by UVE score
                                selected_indices = sortperm(importances, rev=true)[1:min(n_top, count(x -> x > 0, importances))]
                                subset_tag = "uve$(n_top)"

                            elseif method == "spa"
                                # SPA selection on preprocessed data
                                importances = spa_selection(
                                    X_preprocessed, y, n_top;
                                    n_random_starts=10,
                                    cv_folds=n_folds
                                )
                                # SPA returns scores, select non-zero ones
                                selected_indices = findall(x -> x > 0, importances)
                                if length(selected_indices) > n_top
                                    # If SPA selected more than requested, take top by score
                                    selected_indices = sortperm(importances, rev=true)[1:n_top]
                                end
                                subset_tag = "spa$(n_top)"

                            elseif method == "ipls"
                                # iPLS selection on preprocessed data
                                importances = ipls_selection(
                                    X_preprocessed, y;
                                    n_intervals=20,
                                    cv_folds=n_folds
                                )
                                # Select top n_top variables by interval score
                                selected_indices = sortperm(importances, rev=true)[1:n_top]
                                subset_tag = "ipls$(n_top)"

                            elseif method == "uve_spa"
                                # UVE-SPA hybrid selection on preprocessed data
                                importances = uve_spa_selection(
                                    X_preprocessed, y, n_top;
                                    cutoff_multiplier=1.0,
                                    spa_n_random_starts=10,
                                    cv_folds=n_folds
                                )
                                # UVE-SPA returns combined scores
                                selected_indices = findall(x -> x > 0, importances)
                                subset_tag = "uve_spa$(n_top)"

                                else
                                    @warn "Unknown variable selection method: $method (skipping)"
                                    continue
                                end

                                # Mark selection as successful
                                selection_succeeded = true

                            catch e
                                @warn "Variable selection failed for $model_name with method $method, n_top=$n_top: $(sprint(showerror, e)). Skipping."
                                continue
                            end

                            # Skip if selection didn't succeed
                            if !selection_succeeded
                                continue
                            end

                            # Handle edge case: no variables selected
                            if isempty(selected_indices)
                                @warn "No variables selected by $method for $n_top variables (skipping)"
                                continue
                            end

                            # CRITICAL: Check if derivatives are used
                            if preprocess_cfg["deriv"] !== nothing
                                # For derivatives: use preprocessed data, skip reapplying
                                result = run_single_config(
                                    X_preprocessed[:, selected_indices], y,
                                    model_name, config,
                                    preprocess_cfg, task_type,
                                    n_folds,
                                    skip_preprocessing=true,  # Don't reapply!
                                    subset_tag=subset_tag,
                                    n_vars=length(selected_indices),
                                    full_vars=full_vars,
                                    var_selection_method=method,
                                    var_selection_indices=selected_indices
                                )
                            else
                                # For raw/SNV: use raw data, will reapply preprocessing
                                result = run_single_config(
                                    X[:, selected_indices], y,
                                    model_name, config,
                                    preprocess_cfg, task_type,
                                    n_folds,
                                    skip_preprocessing=false,
                                    subset_tag=subset_tag,
                                    n_vars=length(selected_indices),
                                    full_vars=full_vars,
                                    var_selection_method=method,
                                    var_selection_indices=selected_indices
                                )
                            end
                            # Only add result if training succeeded (not nothing)
                            if !isnothing(result)
                                push!(results, result)
                            end
                        end
                    end
                end

                # C. Region subsets (for ALL models)
                if enable_region_subsets && length(region_subsets) > 0
                    println("    → Region subsets: testing $(length(region_subsets)) regions")

                    for region in region_subsets
                        # Same logic: check if derivatives used
                        if preprocess_cfg["deriv"] !== nothing
                            # For derivatives: use preprocessed data, skip reapplying
                            result = run_single_config(
                                X_preprocessed[:, region["indices"]], y,
                                model_name, config,
                                preprocess_cfg, task_type,
                                n_folds,
                                skip_preprocessing=true,
                                subset_tag=region["tag"],
                                n_vars=length(region["indices"]),
                                full_vars=full_vars
                            )
                        else
                            # For raw/SNV: use raw data, reapply preprocessing
                            result = run_single_config(
                                X[:, region["indices"]], y,
                                model_name, config,
                                preprocess_cfg, task_type,
                                n_folds,
                                skip_preprocessing=false,
                                subset_tag=region["tag"],
                                n_vars=length(region["indices"]),
                                full_vars=full_vars
                            )
                        end
                        # Only add result if training succeeded (not nothing)
                        if !isnothing(result)
                            push!(results, result)
                        end
                    end
                end
            end
        end
    end

    finish!(progress)

    println()
    println("="^80)
    println("Search complete! Total results: $(length(results))")
    println("="^80)
    println()

    # 4. Convert to DataFrame
    results_df = DataFrame(results)

    # 5. Compute composite scores and ranks
    rank_results!(results_df)

    # 6. Sort by rank
    sort!(results_df, :Rank)

    return results_df
end


# ============================================================================
# Helper Functions
# ============================================================================

"""
    generate_preprocessing_configs(
        preprocessing::Vector{String},
        derivative_orders::Vector{Int},
        window::Int,
        polyorder::Int
    )::Vector{Dict{String, Any}}

Generate preprocessing configuration dictionaries from user selections.

Converts user-friendly preprocessing selections (e.g., ["raw", "snv", "deriv"])
into full configuration dictionaries with all parameters.

# Preprocessing Methods
- **"raw"**: No preprocessing
- **"snv"**: Standard Normal Variate only
- **"deriv"**: Savitzky-Golay derivative only (generates one config per derivative order)
- **"snv_deriv"**: SNV then derivative (generates one config per derivative order)
- **"deriv_snv"**: Derivative then SNV (generates one config per derivative order)

# Arguments
- `preprocessing::Vector{String}`: Preprocessing method names
- `derivative_orders::Vector{Int}`: Derivative orders to generate (e.g., [1, 2])
- `window::Int`: Window size for Savitzky-Golay filter
- `polyorder::Int`: Polynomial order for Savitzky-Golay filter

# Returns
- `Vector{Dict{String, Any}}`: Configuration dictionaries with keys:
  - `"name"::String`: Preprocessing method name
  - `"deriv"::Union{Int, Nothing}`: Derivative order (or nothing for raw/SNV)
  - `"window"::Union{Int, Nothing}`: Window size (or nothing for raw/SNV)
  - `"polyorder"::Union{Int, Nothing}`: Polynomial order (or nothing for raw/SNV)

# Examples
```julia
# Basic configurations
configs = generate_preprocessing_configs(
    ["raw", "snv", "deriv"],
    [1, 2],
    17,
    3
)
# Returns:
# [
#   {"name" => "raw", "deriv" => nothing, ...},
#   {"name" => "snv", "deriv" => nothing, ...},
#   {"name" => "deriv", "deriv" => 1, "window" => 17, "polyorder" => 3},
#   {"name" => "deriv", "deriv" => 2, "window" => 17, "polyorder" => 3}
# ]

# With combinations
configs = generate_preprocessing_configs(
    ["snv", "deriv", "snv_deriv"],
    [1, 2],
    11,
    2
)
# Returns: snv, deriv(1st), deriv(2nd), snv_deriv(1st), snv_deriv(2nd)
```

# Notes
- "deriv" generates one config per derivative order
- "snv_deriv" and "deriv_snv" also generate one config per derivative order
- Window and polyorder are used for all derivative-based methods
- Returns empty vector if no methods selected
"""
function generate_preprocessing_configs(
    preprocessing::Vector{String},
    derivative_orders::Vector{Int},
    window::Int,
    polyorder::Int
)::Vector{Dict{String, Any}}

    configs = Dict{String, Any}[]

    # Process each preprocessing method
    for method in preprocessing
        if method == "raw"
            # No preprocessing
            push!(configs, Dict{String, Any}(
                "name" => "raw",
                "deriv" => nothing,
                "window" => nothing,
                "polyorder" => nothing
            ))

        elseif method == "snv"
            # SNV only
            push!(configs, Dict{String, Any}(
                "name" => "snv",
                "deriv" => nothing,
                "window" => nothing,
                "polyorder" => nothing
            ))

        elseif method == "deriv"
            # Derivative only - one config per order
            for deriv_order in derivative_orders
                # Adjust polyorder for derivative order
                poly = deriv_order == 1 ? 2 : 3
                push!(configs, Dict{String, Any}(
                    "name" => "deriv",
                    "deriv" => deriv_order,
                    "window" => window,
                    "polyorder" => poly
                ))
            end

        elseif method == "snv_deriv"
            # SNV then derivative - one config per order
            for deriv_order in derivative_orders
                poly = deriv_order == 1 ? 2 : 3
                push!(configs, Dict{String, Any}(
                    "name" => "snv_deriv",
                    "deriv" => deriv_order,
                    "window" => window,
                    "polyorder" => poly
                ))
            end

        elseif method == "deriv_snv"
            # Derivative then SNV - one config per order
            for deriv_order in derivative_orders
                poly = deriv_order == 1 ? 2 : 3
                push!(configs, Dict{String, Any}(
                    "name" => "deriv_snv",
                    "deriv" => deriv_order,
                    "window" => window,
                    "polyorder" => poly
                ))
            end

        else
            @warn "Unknown preprocessing method: $method (skipping)"
        end
    end

    return configs
end


"""
    run_single_config(
        X::Matrix{Float64},
        y::Vector{Float64},
        model_name::String,
        config::Dict{String, Any},
        preprocess_config::Dict{String, Any},
        task_type::String,
        n_folds::Int;
        skip_preprocessing::Bool=false,
        subset_tag::String="full",
        n_vars::Int=size(X, 2),
        full_vars::Int=size(X, 2),
        var_selection_method::Union{String,Nothing}=nothing,
        var_selection_indices::Union{Vector{Int},Nothing}=nothing
    )::Dict{String, Any}

Run a single model configuration with cross-validation.

This function handles the complete workflow for one model configuration:
1. Apply preprocessing (unless skip_preprocessing=true)
2. Run k-fold cross-validation
3. Compute metrics
4. Extract model parameters
5. Return results dictionary

# Critical: Skip Preprocessing Logic

When `skip_preprocessing=true`, the function uses data as-is without transformation.
This is essential for derivative subsets where preprocessing was already applied.

```julia
if skip_preprocessing
    # Data is already preprocessed, use as-is
    # This is critical for derivative subsets
    # (avoids window_size > n_features errors)
else
    # Apply preprocessing to raw data
end
```

# Arguments
- `X::Matrix{Float64}`: Feature matrix (n_samples × n_features)
- `y::Vector{Float64}`: Target vector (n_samples,)
- `model_name::String`: Name of model type ("PLS", "Ridge", etc.)
- `config::Dict{String, Any}`: Model hyperparameters
- `preprocess_config::Dict{String, Any}`: Preprocessing configuration
- `task_type::String`: Either "regression" or "classification"
- `n_folds::Int`: Number of CV folds
- `skip_preprocessing::Bool`: If true, skip preprocessing (data already processed)
- `subset_tag::String`: Tag for this subset ("full", "top10", "region_400-450nm")
- `n_vars::Int`: Number of variables in this subset
- `full_vars::Int`: Total number of variables in full dataset
- `var_selection_method::Union{String,Nothing}`: Variable selection method used (if any)
- `var_selection_indices::Union{Vector{Int},Nothing}`: Indices of selected variables (if any)

# Returns
- `Dict{String, Any}`: Results dictionary with keys:
  - Model info: "Model", "Preprocess", "Deriv", "Window", "Poly", "LVs"
  - Subset info: "SubsetTag", "n_vars", "full_vars"
  - Metrics: "RMSE", "R2" (regression) or "Accuracy", "ROC_AUC" (classification)
  - All hyperparameters from config

# Examples
```julia
# Full model with preprocessing
config = Dict("n_components" => 5)
preprocess_config = Dict("name" => "snv", "deriv" => nothing, ...)
result = run_single_config(
    X, y, "PLS", config, preprocess_config, "regression", 5
)

# Subset with preprocessing already applied
result = run_single_config(
    X_preprocessed[:, top_indices], y,
    "PLS", config, preprocess_config, "regression", 5,
    skip_preprocessing=true,
    subset_tag="top20",
    n_vars=20,
    full_vars=200
)
```
"""
function run_single_config(
    X::Matrix{Float64},
    y::Vector{Float64},
    model_name::String,
    config::Dict{String, Any},
    preprocess_config::Dict{String, Any},
    task_type::String,
    n_folds::Int;
    skip_preprocessing::Bool=false,
    subset_tag::String="full",
    n_vars::Int=size(X, 2),
    full_vars::Int=size(X, 2),
    var_selection_method::Union{String,Nothing}=nothing,
    var_selection_indices::Union{Vector{Int},Nothing}=nothing
)::Union{Dict{String, Any}, Nothing}

    # Build model
    model = build_model(model_name, config, task_type)

    # Run cross-validation with error handling
    local cv_results
    try
        cv_results = run_cross_validation(
            X, y,
            model, model_name,
            preprocess_config, task_type,
            n_folds=n_folds,
            skip_preprocessing=skip_preprocessing
        )
    catch e
        # Log the error and return nothing to indicate failure
        @warn "Model training failed for $model_name with config $config: $(sprint(showerror, e))"
        return nothing  # Signal that this configuration failed
    end

    # Extract metrics
    if task_type == "regression"
        rmse = cv_results["RMSE_mean"]
        r2 = cv_results["R2_mean"]
    else  # classification
        accuracy = cv_results["Accuracy_mean"]
        roc_auc = cv_results["ROC_AUC_mean"]
    end

    # Extract LVs (for PLS models)
    lvs = if model_name == "PLS"
        config["n_components"]
    else
        missing
    end

    # Build result dictionary
    result = Dict{String, Any}(
        "Model" => model_name,
        "Preprocess" => preprocess_config["name"],
        "Deriv" => preprocess_config["deriv"],
        "Window" => preprocess_config["window"],
        "Poly" => preprocess_config["polyorder"],
        "LVs" => lvs,
        "SubsetTag" => subset_tag,
        "n_vars" => n_vars,
        "full_vars" => full_vars,
        "task_type" => task_type,
        "VarSelectionMethod" => var_selection_method,
        "VarSelectionIndices" => var_selection_indices
    )

    # Add metrics
    if task_type == "regression"
        result["RMSE"] = rmse
        result["R2"] = r2
    else
        result["Accuracy"] = accuracy
        result["ROC_AUC"] = roc_auc
    end

    # Add all hyperparameters from config
    for (key, value) in config
        if key != "n_components"  # Already captured as "LVs"
            result[key] = value
        end
    end

    return result
end


# ============================================================================
# Exports
# ============================================================================

export run_search
export generate_preprocessing_configs
export run_single_config


#=
USAGE EXAMPLES
==============

# Example 1: Basic search with defaults
using SpectralPredict

X = rand(100, 200)  # 100 samples, 200 wavelengths
y = rand(100)
wavelengths = collect(400.0:2.0:798.0)

results = run_search(X, y, wavelengths)

# View top 10 models
first(sort(results, :Rank), 10)

# Example 2: Custom configuration
results = run_search(
    X, y, wavelengths,
    task_type="regression",
    models=["PLS", "Ridge", "RandomForest"],
    preprocessing=["raw", "snv", "deriv"],
    derivative_orders=[1, 2],
    derivative_window=11,
    derivative_polyorder=2,
    enable_variable_subsets=true,
    variable_counts=[10, 20, 50, 100],
    enable_region_subsets=true,
    n_top_regions=10,
    n_folds=10,
    lambda_penalty=0.15
)

# Example 3: Only test specific models with raw/SNV
results = run_search(
    X, y, wavelengths,
    models=["PLS", "Ridge"],
    preprocessing=["raw", "snv"],
    enable_variable_subsets=true,
    enable_region_subsets=false
)

# Example 4: Derivative-focused search
results = run_search(
    X, y, wavelengths,
    models=["PLS", "RandomForest"],
    preprocessing=["deriv", "snv_deriv"],
    derivative_orders=[1, 2],
    derivative_window=17,
    derivative_polyorder=3,
    enable_variable_subsets=true,
    variable_counts=[10, 20, 50],
    enable_region_subsets=true,
    n_top_regions=5
)

# Example 5: Analyze results
using DataFrames, Statistics

# Top 10 models
top_10 = first(sort(results, :Rank), 10)
println(top_10)

# Best model for each preprocessing type
for prep in unique(results.Preprocess)
    prep_results = filter(row -> row.Preprocess == prep, results)
    best = first(sort(prep_results, :Rank), 1)
    println("Best model for $prep: ", best.Model, " (RMSE: ", best.RMSE, ")")
end

# Models using fewer than 50 variables
sparse_models = filter(row -> row.n_vars < 50, results)
sort!(sparse_models, :Rank)
println("Top sparse models: ", first(sparse_models, 5))

# Example 6: Save results
using CSV
CSV.write("spectral_prediction_results.csv", results)

# Example 7: Classification task
y_class = rand([0.0, 1.0], 100)
results_class = run_search(
    X, y_class, wavelengths,
    task_type="classification",
    models=["Ridge", "RandomForest", "MLP"],
    preprocessing=["raw", "snv"],
    n_folds=5
)

# View by accuracy
sort!(results_class, :Rank)
println("Best classifier: ", results_class[1, :Model], " (Accuracy: ", results_class[1, :Accuracy], ")")

# Example 8: Performance comparison
function compare_preprocessing(results::DataFrame)
    for prep in unique(results.Preprocess)
        prep_results = filter(row -> row.Preprocess == prep, results)
        avg_rmse = mean(prep_results.RMSE)
        best_rmse = minimum(prep_results.RMSE)
        println("$prep: Avg RMSE=$avg_rmse, Best RMSE=$best_rmse")
    end
end

compare_preprocessing(results)

=#
