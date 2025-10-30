"""
    scoring.jl

Model ranking based on composite score: 90% performance + 10% complexity.

This module implements the scoring algorithm for ranking spectral prediction models.
The scoring system prioritizes predictive performance while using complexity as a
tiebreaker, avoiding harsh penalties that would favor overly simple models.

Key Principles:
- Performance dominates (90% weight via z-score normalization)
- Complexity as tiebreaker only (10% weight)
- No arbitrary bonuses for small models
- No harsh sparsity penalties
"""

module Scoring

using Statistics
using DataFrames

export compute_composite_score, rank_results!

"""
    compute_composite_score(
        metrics::Dict{String, Float64},
        all_metrics::Vector{Dict{String, Float64}},
        task_type::String,
        n_vars::Int,
        full_vars::Int,
        lvs::Union{Int, Nothing},
        lambda_penalty::Float64=0.15
    )::Float64

Compute composite score for a single model based on performance and complexity.

The composite score combines normalized performance metrics with a small complexity
penalty. Lower scores are better.

# Algorithm

For regression tasks:
```
z_rmse = (RMSE - μ_rmse) / σ_rmse
z_r2 = (R² - μ_r2) / σ_r2
performance_score = 0.5 * z_rmse - 0.5 * z_r2
```

For classification tasks:
```
z_auc = (AUC - μ_auc) / σ_auc
z_acc = (Accuracy - μ_acc) / σ_acc
performance_score = -z_auc - 0.3 * z_acc
```

Complexity penalty:
```
lvs_penalty = lvs / 25.0  (normalized to typical range)
vars_penalty = n_vars / full_vars  (fraction of variables used)
complexity_scale = 0.3 * (lambda_penalty / 0.15)
complexity_penalty = complexity_scale * (lvs_penalty + vars_penalty)
```

Final score:
```
composite_score = performance_score + complexity_penalty
```

# Arguments
- `metrics::Dict{String, Float64}`: Performance metrics for this model
- `all_metrics::Vector{Dict{String, Float64}}`: Metrics for all models (for normalization)
- `task_type::String`: Either "regression" or "classification"
- `n_vars::Int`: Number of variables used in this model
- `full_vars::Int`: Total number of variables available
- `lvs::Union{Int, Nothing}`: Number of latent variables (PLS models only)
- `lambda_penalty::Float64`: Penalty weight (default 0.15 = ~10% of performance)

# Returns
- `Float64`: Composite score (lower is better)

# Notes
- Z-scores typically range from -3 to +3 (total range ~6)
- Complexity is scaled to ~10% of performance range
- If standard deviation is 0 (all models identical), z-score is 0
- Performance always dominates; complexity only breaks ties

# Examples
```julia
# Regression model
metrics = Dict("RMSE" => 0.5, "R2" => 0.85)
all_metrics = [
    Dict("RMSE" => 0.6, "R2" => 0.80),
    Dict("RMSE" => 0.5, "R2" => 0.85),
    Dict("RMSE" => 0.4, "R2" => 0.90)
]
score = compute_composite_score(metrics, all_metrics, "regression", 50, 100, 10)

# Classification model
metrics = Dict("ROC_AUC" => 0.92, "Accuracy" => 0.88)
all_metrics = [
    Dict("ROC_AUC" => 0.90, "Accuracy" => 0.85),
    Dict("ROC_AUC" => 0.92, "Accuracy" => 0.88),
    Dict("ROC_AUC" => 0.95, "Accuracy" => 0.91)
]
score = compute_composite_score(metrics, all_metrics, "classification", 75, 100, nothing)
```
"""
function compute_composite_score(
    metrics::Dict{String, Float64},
    all_metrics::Vector{Dict{String, Float64}},
    task_type::String,
    n_vars::Int,
    full_vars::Int,
    lvs::Union{Int, Nothing},
    lambda_penalty::Float64=0.15
)::Float64

    # Validate inputs
    @assert task_type in ["regression", "classification"] "task_type must be 'regression' or 'classification'"
    @assert n_vars > 0 "n_vars must be positive"
    @assert full_vars > 0 "full_vars must be positive"
    @assert n_vars <= full_vars "n_vars cannot exceed full_vars"
    @assert lambda_penalty >= 0.0 "lambda_penalty must be non-negative"
    @assert length(all_metrics) > 0 "all_metrics cannot be empty"

    # Compute performance score based on task type
    performance_score = if task_type == "regression"
        compute_regression_performance(metrics, all_metrics)
    else  # classification
        compute_classification_performance(metrics, all_metrics)
    end

    # Compute complexity penalty
    complexity_penalty = compute_complexity_penalty(n_vars, full_vars, lvs, lambda_penalty)

    # Combine performance and complexity (lower is better)
    composite_score = performance_score + complexity_penalty

    return composite_score
end


"""
    compute_regression_performance(
        metrics::Dict{String, Float64},
        all_metrics::Vector{Dict{String, Float64}}
    )::Float64

Compute performance score for regression models using RMSE and R².

Lower RMSE is better, higher R² is better.
Performance score = 0.5 * z_rmse - 0.5 * z_r2 (lower is better)

# Arguments
- `metrics::Dict{String, Float64}`: Metrics for this model
- `all_metrics::Vector{Dict{String, Float64}}`: Metrics for all models

# Returns
- `Float64`: Performance score (lower is better)
"""
function compute_regression_performance(
    metrics::Dict{String, Float64},
    all_metrics::Vector{Dict{String, Float64}}
)::Float64

    # Extract all RMSE and R² values
    all_rmse = [m["RMSE"] for m in all_metrics]
    all_r2 = [m["R2"] for m in all_metrics]

    # Compute z-scores with safe division
    z_rmse = safe_zscore(metrics["RMSE"], all_rmse)
    z_r2 = safe_zscore(metrics["R2"], all_r2)

    # Performance score (lower is better)
    # - Lower RMSE is better → positive z_rmse
    # - Higher R² is better → negative z_r2 (subtract to make lower better)
    performance_score = 0.5 * z_rmse - 0.5 * z_r2

    return performance_score
end


"""
    compute_classification_performance(
        metrics::Dict{String, Float64},
        all_metrics::Vector{Dict{String, Float64}}
    )::Float64

Compute performance score for classification models using ROC AUC and Accuracy.

Higher AUC is better (70% weight), higher accuracy is better (30% weight).
Performance score = -z_auc - 0.3 * z_acc (lower is better)

# Arguments
- `metrics::Dict{String, Float64}`: Metrics for this model
- `all_metrics::Vector{Dict{String, Float64}}`: Metrics for all models

# Returns
- `Float64`: Performance score (lower is better)
"""
function compute_classification_performance(
    metrics::Dict{String, Float64},
    all_metrics::Vector{Dict{String, Float64}}
)::Float64

    # Extract all AUC and Accuracy values
    all_auc = [m["ROC_AUC"] for m in all_metrics]
    all_acc = [m["Accuracy"] for m in all_metrics]

    # Compute z-scores
    z_auc = safe_zscore(metrics["ROC_AUC"], all_auc)
    z_acc = safe_zscore(metrics["Accuracy"], all_acc)

    # Performance score (lower is better)
    # - Higher AUC is better → negate z_auc (70% weight)
    # - Higher accuracy is better → negate z_acc (30% weight)
    performance_score = -z_auc - 0.3 * z_acc

    return performance_score
end


"""
    compute_complexity_penalty(
        n_vars::Int,
        full_vars::Int,
        lvs::Union{Int, Nothing},
        lambda_penalty::Float64
    )::Float64

Compute complexity penalty based on number of variables and latent variables.

The penalty is scaled to be approximately 10% of the performance score range.
With lambda_penalty=0.15 and z-scores ranging ±3 (total range ~6), the complexity
scale is 0.3, making the penalty range about 10% of performance.

# Algorithm
```
lvs_penalty = lvs / 25.0  (normalized, typically lvs ≤ 25)
vars_penalty = n_vars / full_vars  (fraction of variables)
complexity_scale = 0.3 * (lambda_penalty / 0.15)
complexity_penalty = complexity_scale * (lvs_penalty + vars_penalty)
```

# Arguments
- `n_vars::Int`: Number of variables used
- `full_vars::Int`: Total variables available
- `lvs::Union{Int, Nothing}`: Latent variables (nothing for non-PLS models)
- `lambda_penalty::Float64`: Penalty weight

# Returns
- `Float64`: Complexity penalty (always ≥ 0)
"""
function compute_complexity_penalty(
    n_vars::Int,
    full_vars::Int,
    lvs::Union{Int, Nothing},
    lambda_penalty::Float64
)::Float64

    # Latent variables penalty (normalized to typical range of 1-25)
    lvs_penalty = if lvs !== nothing
        lvs / 25.0
    else
        0.0
    end

    # Variables penalty (fraction of total variables used)
    vars_penalty = n_vars / full_vars

    # Scale complexity to ~10% of performance range
    # Z-scores typically range ±3 (total range ~6)
    # With lambda_penalty=0.15, complexity_scale=0.3
    # This makes complexity ~10% of performance
    complexity_scale = 0.3 * (lambda_penalty / 0.15)

    # Total complexity penalty
    complexity_penalty = complexity_scale * (lvs_penalty + vars_penalty)

    return complexity_penalty
end


"""
    safe_zscore(value::Float64, all_values::Vector{Float64})::Float64

Compute z-score with safe handling of zero standard deviation.

If standard deviation is 0 (all values identical), returns 0.0.

# Arguments
- `value::Float64`: The value to normalize
- `all_values::Vector{Float64}`: All values for computing mean and std

# Returns
- `Float64`: Z-score, or 0.0 if std is 0
"""
function safe_zscore(value::Float64, all_values::Vector{Float64})::Float64
    μ = mean(all_values)
    σ = std(all_values)

    # Handle edge case where all values are identical
    if σ == 0.0 || isnan(σ)
        return 0.0
    end

    z = (value - μ) / σ
    return z
end


"""
    rank_results!(results::DataFrame)

Add composite scores and ranks to results DataFrame (in-place).

This function computes composite scores for all models in the results DataFrame
and adds two new columns:
- `CompositeScore`: The computed composite score (lower is better)
- `Rank`: Integer rank (1 = best model with lowest composite score)

The function modifies the DataFrame in-place and also returns it.

# Arguments
- `results::DataFrame`: Results DataFrame with columns:
  - Model-specific metrics (RMSE, R2 for regression; ROC_AUC, Accuracy for classification)
  - `n_vars`: Number of variables
  - `full_vars`: Total variables available
  - `lvs`: Latent variables (can be missing for non-PLS models)
  - `task_type`: "regression" or "classification"
  - Optional: `lambda_penalty` (defaults to 0.15 if not present)

# Returns
- `DataFrame`: The modified results DataFrame (same as input)

# Notes
- The DataFrame is modified in-place
- Existing `CompositeScore` and `Rank` columns are overwritten
- Models are ranked by composite score (ascending)
- Ties in composite score receive the same rank (average rank)

# Examples
```julia
# Create sample results
results = DataFrame(
    RMSE = [0.5, 0.6, 0.4],
    R2 = [0.85, 0.80, 0.90],
    n_vars = [50, 75, 30],
    full_vars = [100, 100, 100],
    lvs = [10, 15, 5],
    task_type = ["regression", "regression", "regression"]
)

# Add scores and ranks
rank_results!(results)

# Results now have CompositeScore and Rank columns
# Rank 1 = best model (lowest composite score)
```
"""
function rank_results!(results::DataFrame)

    # Validate required columns
    required_cols = ["n_vars", "full_vars", "task_type"]
    for col in required_cols
        @assert col in names(results) "Results DataFrame must have column: $col"
    end

    # Check for task-specific metrics
    if "regression" in results.task_type
        @assert "RMSE" in names(results) "Regression results must have RMSE column"
        @assert "R2" in names(results) "Regression results must have R2 column"
    end
    if "classification" in results.task_type
        @assert "ROC_AUC" in names(results) "Classification results must have ROC_AUC column"
        @assert "Accuracy" in names(results) "Classification results must have Accuracy column"
    end

    n_results = nrow(results)
    if n_results == 0
        # Empty DataFrame - add empty columns
        results.CompositeScore = Float64[]
        results.Rank = Int[]
        return results
    end

    # Get lambda penalty (default to 0.15 if not specified)
    lambda_penalty = if "lambda_penalty" in names(results)
        results.lambda_penalty[1]  # Assume constant across all rows
    else
        0.15
    end

    # Collect all metrics for normalization
    all_metrics = Vector{Dict{String, Float64}}(undef, n_results)

    for i in 1:n_results
        task_type = results.task_type[i]

        if task_type == "regression"
            all_metrics[i] = Dict(
                "RMSE" => results.RMSE[i],
                "R2" => results.R2[i]
            )
        else  # classification
            all_metrics[i] = Dict(
                "ROC_AUC" => results.ROC_AUC[i],
                "Accuracy" => results.Accuracy[i]
            )
        end
    end

    # Compute composite scores
    composite_scores = Vector{Float64}(undef, n_results)

    for i in 1:n_results
        # Get model parameters
        task_type = results.task_type[i]
        n_vars = results.n_vars[i]
        full_vars = results.full_vars[i]

        # Handle missing lvs values
        lvs = if "lvs" in names(results) && !ismissing(results.lvs[i])
            Int(results.lvs[i])
        else
            nothing
        end

        # Compute composite score
        composite_scores[i] = compute_composite_score(
            all_metrics[i],
            all_metrics,
            task_type,
            n_vars,
            full_vars,
            lvs,
            lambda_penalty
        )
    end

    # Add composite scores to DataFrame
    results.CompositeScore = composite_scores

    # Compute ranks (1 = best, i.e., lowest composite score)
    # Use ordinalrank for ties (gives same rank to tied values)
    sorted_indices = sortperm(composite_scores)
    ranks = Vector{Int}(undef, n_results)

    current_rank = 1
    for i in 1:n_results
        idx = sorted_indices[i]

        # Check for ties
        if i > 1 && composite_scores[idx] ≈ composite_scores[sorted_indices[i-1]]
            # Same score as previous - use same rank
            ranks[idx] = ranks[sorted_indices[i-1]]
        else
            # New score - assign current rank
            ranks[idx] = current_rank
        end

        current_rank += 1
    end

    results.Rank = ranks

    return results
end

end  # module Scoring
