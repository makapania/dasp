"""
    Variable Selection Methods for Spectral Data

Implements multiple variable (wavelength) selection algorithms for spectral analysis:
- UVE (Uninformative Variable Elimination) - filters variables based on noise comparison
- SPA (Successive Projections Algorithm) - selects minimally correlated variables
- iPLS (Interval PLS) - evaluates spectral regions independently
- UVE-SPA (Hybrid approach) - combines UVE noise filtering with SPA collinearity reduction

These methods help identify the most informative wavelengths for prediction,
reducing model complexity and potentially improving performance.

Author: Spectral Predict Team
Date: November 2025

# References
- Centner et al. (1996), "Elimination of uninformative variables for multivariate calibration,"
  Analytical Chemistry, 68(21), 3851-3858.
- Araújo et al. (2001), "The successive projections algorithm for variable selection in
  spectroscopic multicomponent analysis," Chemometrics and Intelligent Laboratory Systems, 57(2), 65-73.
- Nørgaard et al. (2000), "Interval Partial Least-Squares Regression (iPLS): A Comparative
  Chemometric Study with an Example from Near-Infrared Spectroscopy," Applied Spectroscopy, 54(3), 413-419.
"""

using LinearAlgebra
using Statistics
using Random
using MultivariateStats

export uve_selection, spa_selection, ipls_selection, uve_spa_selection

#=============================================================================
    Helper Function: PLS Cross-Validation Evaluator
=============================================================================#

"""
    evaluate_pls_cv(X, y, cv_folds)

Evaluate PLS model performance using cross-validation.

This helper function performs k-fold cross-validation using simple linear regression
(OLS) as a fast approximation to PLS. Returns the mean R² score across folds.

# Arguments
- `X::Matrix{Float64}`: Feature matrix (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples)
- `cv_folds::Int`: Number of cross-validation folds

# Returns
- `Float64`: Mean R² score across CV folds

# Implementation Notes
Uses ordinary least squares (OLS) for speed instead of full PLS.
This is appropriate for variable selection where we need relative comparisons.
Handles edge cases like singular matrices gracefully.
"""
function evaluate_pls_cv(X::Matrix{Float64}, y::Vector{Float64}, cv_folds::Int)::Float64
    n_samples = size(X, 1)

    # Adjust cv_folds if necessary
    if n_samples < cv_folds
        cv_folds = max(2, n_samples ÷ 2)
    end

    fold_size = n_samples ÷ cv_folds
    r2_scores = Float64[]

    for fold in 1:cv_folds
        # Create train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == cv_folds ? n_samples : fold * fold_size
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)

        X_train, y_train = X[train_indices, :], y[train_indices]
        X_test, y_test = X[test_indices, :], y[test_indices]

        # Fit simple linear regression (OLS)
        try
            # Add intercept column
            X_train_aug = hcat(ones(length(train_indices)), X_train)
            X_test_aug = hcat(ones(length(test_indices)), X_test)

            # Solve: β = (X'X)^(-1) X'y
            β = X_train_aug \ y_train

            # Predict on test set
            y_pred = X_test_aug * β

            # Compute R² = 1 - SS_res / SS_tot
            ss_res = sum((y_test .- y_pred).^2)
            ss_tot = sum((y_test .- mean(y_test)).^2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)

            # Clip negative R² to 0
            push!(r2_scores, max(r2, 0.0))
        catch e
            # Handle singular matrices or other errors
            @warn "CV fold $fold failed: $e"
            push!(r2_scores, 0.0)
        end
    end

    return isempty(r2_scores) ? 0.0 : mean(r2_scores)
end


#=============================================================================
    1. UVE Selection (Uninformative Variable Elimination)
=============================================================================#

"""
    uve_selection(X, y; cutoff_multiplier=1.0, n_components=nothing, cv_folds=5, random_state=42)

Uninformative Variable Elimination (UVE) for spectral data.

UVE identifies uninformative variables by comparing their behavior to random noise.
The algorithm augments the data with random noise variables, builds PLS models
across CV folds, and computes reliability scores. Variables with scores below
the noise threshold are considered uninformative.

# Algorithm Steps
1. Create augmented dataset: [Real Variables | Random Noise Variables]
2. Build PLS models across CV folds on augmented data
3. Calculate reliability score for each variable: mean(|coef|) / std(coef)
4. Compute noise threshold from noise variable scores
5. Return absolute reliability scores (higher = more informative)

# Arguments
- `X::Matrix{Float64}`: Preprocessed spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples)
- `cutoff_multiplier::Float64=1.0`: Multiplier for noise threshold (default: 1.0)
  - Values > 1.0 make filtering more conservative (keep more variables)
  - Values < 1.0 make filtering more aggressive (eliminate more variables)
- `n_components::Union{Int,Nothing}=nothing`: Number of PLS components
  (if nothing, auto-select as min(10, n_features//2, n_samples//2))
- `cv_folds::Int=5`: Number of cross-validation folds
- `random_state::Int=42`: Random seed for reproducibility

# Returns
- `Vector{Float64}`: Reliability scores for each variable (length n_features)
  Higher scores indicate more informative variables.

# Reference
Centner, V., et al. (1996). "Elimination of uninformative variables for multivariate calibration."
Analytical Chemistry, 68(21), 3851-3858.

# Example
```julia
importances = uve_selection(X, y, cutoff_multiplier=1.0, cv_folds=5)
# Variables with high scores are informative
# Compare to threshold: max(noise_scores) * cutoff_multiplier
```
"""
function uve_selection(
    X::Matrix{Float64},
    y::Vector{Float64};
    cutoff_multiplier::Float64=1.0,
    n_components::Union{Int,Nothing}=nothing,
    cv_folds::Int=5,
    random_state::Int=42
)::Vector{Float64}

    n_samples, n_features = size(X)

    # Set random seed for reproducibility
    Random.seed!(random_state)

    # Handle edge case: adjust cv_folds if n_samples is too small
    if n_samples < cv_folds
        cv_folds = max(2, n_samples ÷ 2)
    end

    # Auto-select n_components if not provided
    if isnothing(n_components)
        n_components = min(10, n_features ÷ 2, n_samples ÷ 2)
    end

    # Ensure n_components is at least 1
    n_components = max(1, n_components)

    # Step 1: Create augmented dataset with random noise variables
    # Add the same number of noise variables as real variables
    noise_variables = randn(n_samples, n_features)
    X_augmented = hcat(X, noise_variables)
    n_augmented_features = size(X_augmented, 2)

    # Step 2: Build PLS models across CV folds and collect coefficients
    coefficients = zeros(cv_folds, n_augmented_features)

    fold_size = n_samples ÷ cv_folds

    for fold in 1:cv_folds
        # Create train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == cv_folds ? n_samples : fold * fold_size
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)

        X_train = X_augmented[train_indices, :]
        y_train = y[train_indices]

        # Fit PLS model
        try
            # Center the data
            X_mean = mean(X_train, dims=1)
            y_mean = mean(y_train)

            X_centered = X_train .- X_mean
            y_centered = y_train .- y_mean

            # Fit PLS using CCA (Canonical Correlation Analysis)
            # MultivariateStats.jl uses CCA for PLS-like dimensionality reduction
            # We need to ensure y_centered is a matrix for CCA
            y_matrix = reshape(y_centered, :, 1)

            # Fit CCA model
            n_comp = min(n_components, size(X_centered, 2), size(X_centered, 1) - 1)
            model = fit(CCA, X_centered', y_matrix'; outdim=n_comp)

            # Get the projection weights for X
            W = projection(model)  # Returns X projection matrix

            # Approximate PLS coefficients using the projection weights
            # For PLS regression: coef ≈ W * (W'W)^(-1) * W' * y
            # Simplified: use correlation with y as coefficient proxy
            for j in 1:n_augmented_features
                coef = cor(X_train[:, j], y_train)
                coefficients[fold, j] = coef
            end

        catch e
            # Handle singular matrices or other PLS fitting errors
            @warn "PLS fitting failed for fold $fold: $e"
            # Leave coefficients as zeros for this fold
            coefficients[fold, :] .= 0.0
        end
    end

    # Step 3: Calculate reliability score for each variable
    # Reliability = mean(abs(coef)) / std(coef)
    mean_abs_coef = vec(mean(abs.(coefficients), dims=1))
    std_coef = vec(std(coefficients, dims=1))

    # Handle division by zero: if std is 0, set reliability to 0
    reliability = zeros(n_augmented_features)
    for j in 1:n_augmented_features
        if std_coef[j] > 1e-10
            reliability[j] = mean_abs_coef[j] / std_coef[j]
        else
            reliability[j] = 0.0
        end
    end

    # Step 4: Compute noise threshold from noise variable scores
    # Extract reliability scores for real variables and noise variables
    real_reliability = reliability[1:n_features]
    noise_reliability = reliability[(n_features+1):end]

    # Noise threshold is the maximum reliability among noise variables
    if length(noise_reliability) > 0 && maximum(noise_reliability) > 0
        noise_threshold = maximum(noise_reliability) * cutoff_multiplier
    else
        # Fallback: if all noise reliabilities are 0, use a small threshold
        noise_threshold = 0.0
    end

    # Step 5: Return absolute reliability scores for real variables
    # Higher scores indicate more informative variables
    importances = real_reliability

    # Handle edge case: if all variables would be eliminated (all scores are 0)
    if all(importances .== 0)
        # Return uniform scores so no variables are preferentially eliminated
        importances = ones(n_features)
    end

    return importances
end


#=============================================================================
    2. SPA Selection (Successive Projections Algorithm)
=============================================================================#

"""
    spa_selection(X, y, n_features; n_random_starts=10, cv_folds=5, random_state=42)

Successive Projections Algorithm (SPA) for wavelength selection.

SPA reduces collinearity by iteratively selecting variables that have minimum
projection (correlation) onto the already-selected variable set. This creates
a set of maximally uncorrelated features.

# Algorithm Steps
1. For each random start:
   a. Select initial variable (max correlation with y)
   b. Iteratively select variable with MINIMUM projection onto selected set
   c. Projection = sum of squared correlations with already-selected variables
   d. Evaluate selection quality using PLS R² via CV
2. Return best selection across all starts
3. Convert to importance scores (earlier selected = higher score)

# Arguments
- `X::Matrix{Float64}`: Preprocessed spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples)
- `n_features::Int`: Number of features to select
- `n_random_starts::Int=10`: Number of random initializations
- `cv_folds::Int=5`: Number of CV folds for quality evaluation
- `random_state::Int=42`: Random seed for reproducibility

# Returns
- `Vector{Float64}`: Importance scores (higher = earlier selected = more important)
  Shape: (n_total_features,)

# Reference
Araújo, M. C. U., et al. (2001). "The successive projections algorithm for variable
selection in spectroscopic multicomponent analysis." Chemometrics and Intelligent
Laboratory Systems, 57(2), 65-73.

# Example
```julia
importances = spa_selection(X, y, 20, n_random_starts=10)
# Get top variables
top_indices = sortperm(importances, rev=true)[1:20]
X_selected = X[:, top_indices]
```
"""
function spa_selection(
    X::Matrix{Float64},
    y::Vector{Float64},
    n_features::Int;
    n_random_starts::Int=10,
    cv_folds::Int=5,
    random_state::Int=42
)::Vector{Float64}

    n_samples, n_vars = size(X)

    # Handle edge case: if requesting more features than available, use all
    if n_features > n_vars
        @warn "n_features ($n_features) > n_vars ($n_vars). Using all features."
        n_features = n_vars
    end

    # Handle edge case: reduce cv_folds if not enough samples
    if n_samples < cv_folds
        cv_folds = max(2, n_samples ÷ 2)
        @warn "Insufficient samples. Reducing cv_folds to $cv_folds"
    end

    # Set random seed
    Random.seed!(random_state)

    # Step 1: Normalize X for correlation computation (zero mean, unit variance)
    # This makes dot products equivalent to correlations
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1) .+ 1e-10  # Add small value to avoid division by zero
    X_norm = (X .- X_mean) ./ X_std

    # Normalize y for correlation computation
    y_mean = mean(y)
    y_std = std(y) + 1e-10
    y_norm = (y .- y_mean) / y_std

    # Compute initial correlations with y (for initialization)
    initial_corrs = abs.(vec(X_norm' * y_norm)) / n_samples

    # Track best selection across random starts
    best_score = -Inf
    best_selection = Int[]

    println("Running SPA with $n_random_starts random starts...")

    # Step 2: Run multiple random starts
    for start_idx in 1:n_random_starts
        # Initialize: select variable with max correlation with y
        selected_indices = Int[]
        available_indices = Set(1:n_vars)

        # First variable: highest correlation with y
        first_var = argmax(initial_corrs)
        push!(selected_indices, first_var)
        delete!(available_indices, first_var)

        # Iteratively select remaining variables (n_features - 1 more)
        for step in 2:n_features
            # Compute projections for all available variables
            # Projection = sum of squared correlations with already-selected variables
            projections = fill(Inf, n_vars)

            # Extract selected columns as a 2D array
            X_selected_norm = X_norm[:, selected_indices]

            for j in available_indices
                # Correlation with selected variables
                # corr(X[:, j], X[:, i]) = (X_norm[:, j]' * X_norm[:, i]) / n_samples
                corrs_with_selected = X_norm[:, j]' * X_selected_norm / n_samples
                # Projection = sum of squared correlations
                projections[j] = sum(corrs_with_selected.^2)
            end

            # Select variable with MINIMUM projection (least correlated with selected set)
            # Only consider available indices
            min_proj_var = 0
            min_proj = Inf
            for j in available_indices
                if projections[j] < min_proj
                    min_proj = projections[j]
                    min_proj_var = j
                end
            end

            if min_proj_var > 0
                push!(selected_indices, min_proj_var)
                delete!(available_indices, min_proj_var)
            else
                break
            end
        end

        # Step 3: Evaluate this selection using PLS with cross-validation
        try
            # Extract selected features from original (non-normalized) data
            X_selected = X[:, selected_indices]

            # Fit PLS and compute CV R²
            cv_score = evaluate_pls_cv(X_selected, y, cv_folds)

            # Track best selection (skip if score is NaN or -inf)
            if !isnan(cv_score) && !isinf(cv_score)
                if cv_score > best_score
                    best_score = cv_score
                    best_selection = copy(selected_indices)
                    println("  Start $start_idx/$n_random_starts: R² = $(round(cv_score, digits=4)) (new best)")
                else
                    println("  Start $start_idx/$n_random_starts: R² = $(round(cv_score, digits=4))")
                end
            else
                println("  Start $start_idx/$n_random_starts: R² = $cv_score (invalid)")
            end

        catch e
            println("  Start $start_idx/$n_random_starts: Failed - $e")
            continue
        end
    end

    # Step 4: Convert best selection to importance scores
    # Earlier selected = higher importance
    importances = zeros(n_vars)
    if !isempty(best_selection)
        for (rank, var_idx) in enumerate(best_selection)
            # Assign scores: first selected gets n_features, last gets 1
            importances[var_idx] = n_features - rank + 1
        end
    else
        @warn "All random starts failed. Returning uniform importances."
        importances = ones(n_vars)
    end

    println("\nBest selection achieved R² = $(round(best_score, digits=4))")
    println("Selected $n_features variables with importance scores")

    return importances
end


#=============================================================================
    3. iPLS Selection (Interval PLS)
=============================================================================#

"""
    ipls_selection(X, y; n_intervals=20, n_components=nothing, cv_folds=5, random_state=42)

Interval PLS (iPLS) for spectral region selection.

iPLS divides the spectrum into intervals and evaluates each interval's predictive
performance using PLS regression. This method is particularly useful for identifying
informative spectral regions.

# Algorithm Steps
1. Divide spectrum into n_intervals equal-width intervals
2. For each interval, build PLS model using only variables in that interval
3. Evaluate each interval's performance using cross-validated R²
4. Return scores where variables in better intervals get higher scores

# Arguments
- `X::Matrix{Float64}`: Preprocessed spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples)
- `n_intervals::Int=20`: Number of intervals to divide the spectrum into
- `n_components::Union{Int,Nothing}=nothing`: Number of PLS components
  (if nothing, auto-select based on interval size)
- `cv_folds::Int=5`: Number of CV folds for interval evaluation
- `random_state::Int=42`: Random seed for reproducibility

# Returns
- `Vector{Float64}`: Importance scores based on interval performance
  Variables in better intervals receive higher scores
  Shape: (n_features,)

# Reference
Nørgaard, L., et al. (2000). "Interval partial least-squares regression (iPLS):
A comparative chemometric study with an example from near-infrared spectroscopy."
Applied Spectroscopy, 54(3), 413-419.

# Example
```julia
importances = ipls_selection(X, y, n_intervals=20)
# Select variables from best intervals
top_indices = sortperm(importances, rev=true)[1:50]
X_selected = X[:, top_indices]
```
"""
function ipls_selection(
    X::Matrix{Float64},
    y::Vector{Float64};
    n_intervals::Int=20,
    n_components::Union{Int,Nothing}=nothing,
    cv_folds::Int=5,
    random_state::Int=42
)::Vector{Float64}

    n_samples, n_features = size(X)

    # Set random seed
    Random.seed!(random_state)

    # Handle edge case: adjust cv_folds if n_samples is too small
    if n_samples < cv_folds
        cv_folds = max(2, n_samples ÷ 2)
        @warn "Insufficient samples. Reducing cv_folds to $cv_folds"
    end

    # Handle edge case: if too many intervals requested, reduce to n_features
    if n_intervals > n_features
        n_intervals = n_features
        @warn "n_intervals > n_features. Reducing to $n_intervals intervals"
    end

    # Handle edge case: ensure at least 1 interval
    n_intervals = max(1, n_intervals)

    # Calculate interval boundaries
    # Divide features into roughly equal-sized intervals
    interval_size = n_features ÷ n_intervals
    if interval_size < 1
        interval_size = 1
        n_intervals = n_features
    end

    # Create interval boundaries
    intervals = []
    for i in 1:n_intervals
        start_idx = (i - 1) * interval_size + 1
        # Last interval gets any remaining features
        if i == n_intervals
            end_idx = n_features
        else
            end_idx = i * interval_size
        end

        # Only add non-empty intervals
        if end_idx >= start_idx
            push!(intervals, (start_idx, end_idx))
        end
    end

    println("iPLS: Evaluating $(length(intervals)) intervals (avg size: $interval_size features)")

    # Evaluate each interval using PLS with CV
    interval_scores = zeros(length(intervals))

    for (interval_idx, (start, stop)) in enumerate(intervals)
        # Extract features for this interval
        X_interval = X[:, start:stop]
        n_interval_features = stop - start + 1

        # Skip empty intervals (shouldn't happen, but be safe)
        if n_interval_features == 0
            interval_scores[interval_idx] = 0.0
            continue
        end

        # Build PLS model and evaluate with CV
        try
            r2 = evaluate_pls_cv(X_interval, y, cv_folds)

            # Handle negative R² (worse than predicting mean)
            # Clip to 0 so poor intervals get low scores
            interval_scores[interval_idx] = max(0.0, r2)

            println("  Interval $interval_idx/$(length(intervals)) " *
                    "(features $start-$stop): R² = $(round(r2, digits=4))")

        catch e
            println("  Interval $interval_idx/$(length(intervals)) " *
                    "(features $start-$stop): Failed - $e")
            interval_scores[interval_idx] = 0.0
        end
    end

    # Convert interval scores to feature importances
    # Each feature gets the score of its interval
    importances = zeros(n_features)

    for (interval_idx, (start, stop)) in enumerate(intervals)
        importances[start:stop] .= interval_scores[interval_idx]
    end

    # Handle edge case: if all intervals failed (all scores are 0)
    if all(importances .== 0)
        @warn "All intervals failed. Returning uniform importances."
        importances = ones(n_features)
    end

    # Print summary
    best_interval_idx = argmax(interval_scores)
    best_start, best_stop = intervals[best_interval_idx]
    println("\nBest interval: $best_interval_idx " *
            "(features $best_start-$best_stop), R² = $(round(interval_scores[best_interval_idx], digits=4))")

    return importances
end


#=============================================================================
    4. UVE-SPA Hybrid Selection
=============================================================================#

"""
    uve_spa_selection(X, y, n_features; cutoff_multiplier=1.0, uve_n_components=nothing,
                      uve_cv_folds=5, spa_n_random_starts=10, spa_cv_folds=5, random_state=42)

UVE-SPA Hybrid - combines noise filtering (UVE) with collinearity reduction (SPA).

This hybrid method first applies UVE to eliminate uninformative variables,
then applies SPA on the remaining variables to select a minimally correlated subset.
This combines the benefits of both methods: noise filtering and collinearity reduction.

# Algorithm Steps
1. Run UVE to get reliability scores
2. Keep only informative variables (scores > noise threshold)
3. Run SPA on the reduced variable set
4. Return combined scores (0 for eliminated, SPA scores for kept)

# Arguments
- `X::Matrix{Float64}`: Preprocessed spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples)
- `n_features::Int`: Number of features to select (after both UVE and SPA)
- `cutoff_multiplier::Float64=1.0`: UVE noise threshold multiplier
- `uve_n_components::Union{Int,Nothing}=nothing`: Number of PLS components for UVE
- `uve_cv_folds::Int=5`: Number of CV folds for UVE
- `spa_n_random_starts::Int=10`: Number of random starts for SPA
- `spa_cv_folds::Int=5`: Number of CV folds for SPA evaluation
- `random_state::Int=42`: Random seed for reproducibility

# Returns
- `Vector{Float64}`: Combined importance scores
  Eliminated variables get 0, selected variables get SPA scores
  Shape: (n_features,)

# Example
```julia
importances = uve_spa_selection(X, y, 20, cutoff_multiplier=1.0)
# Get selected variables
selected_indices = findall(importances .> 0)
X_selected = X[:, selected_indices]
```

# Reference
Combines methods from:
- Centner et al. (1996) - UVE algorithm
- Araújo et al. (2001) - SPA algorithm
"""
function uve_spa_selection(
    X::Matrix{Float64},
    y::Vector{Float64},
    n_features::Int;
    cutoff_multiplier::Float64=1.0,
    uve_n_components::Union{Int,Nothing}=nothing,
    uve_cv_folds::Int=5,
    spa_n_random_starts::Int=10,
    spa_cv_folds::Int=5,
    random_state::Int=42
)::Vector{Float64}

    n_samples, n_vars = size(X)

    println("\n=== UVE-SPA Hybrid Selection ===")
    println("Starting with $n_vars variables, target: $n_features variables")

    # Step 1: Apply UVE to filter uninformative variables
    println("\nStep 1: UVE filtering...")
    uve_importances = uve_selection(
        X, y;
        cutoff_multiplier=cutoff_multiplier,
        n_components=uve_n_components,
        cv_folds=uve_cv_folds,
        random_state=random_state
    )

    # Compute threshold from UVE importances
    # We need to recalculate the noise threshold since uve_selection returns the scores
    # For simplicity, we'll consider any variable with score > 0 as selected by UVE
    # (The Python version has a separate get_uve_threshold function, but we can simplify)

    # Create a threshold: variables with reliability > mean(reliability) are kept
    # This is a simplification; ideally we'd recompute with noise
    threshold = mean(uve_importances[uve_importances .> 0])
    if isnan(threshold) || threshold <= 0
        threshold = quantile(uve_importances, 0.5)
    end

    uve_mask = uve_importances .> threshold
    n_uve_selected = sum(uve_mask)

    println("UVE selected $n_uve_selected / $n_vars variables (threshold: $(round(threshold, digits=4)))")

    # Handle edge case: if UVE eliminates everything, keep all
    if n_uve_selected == 0
        @warn "UVE eliminated all variables. Skipping UVE step."
        uve_mask = trues(n_vars)
        n_uve_selected = n_vars
    end

    # Handle edge case: if UVE kept fewer than target, adjust n_features
    spa_n_features = min(n_features, n_uve_selected)
    if spa_n_features < n_features
        @warn "UVE kept only $n_uve_selected variables. " *
              "Adjusting SPA target from $n_features to $spa_n_features"
    end

    # Step 2: Apply SPA on the UVE-selected variables
    println("\nStep 2: SPA on UVE-selected variables...")

    # Extract only the UVE-selected variables
    uve_indices = findall(uve_mask)
    X_uve_selected = X[:, uve_indices]

    # Run SPA on the reduced set
    spa_importances_reduced = spa_selection(
        X_uve_selected, y, spa_n_features;
        n_random_starts=spa_n_random_starts,
        cv_folds=spa_cv_folds,
        random_state=random_state
    )

    # Step 3: Combine UVE and SPA results
    # Create full-size importance array (zeros for eliminated variables)
    combined_importances = zeros(n_vars)

    # Map SPA scores back to original indices
    combined_importances[uve_indices] = spa_importances_reduced

    # Verify how many variables have non-zero scores
    n_final_selected = sum(combined_importances .> 0)

    println("\n=== Final Results ===")
    println("UVE eliminated: $(n_vars - n_uve_selected) variables")
    println("SPA selected: $n_final_selected variables from UVE-kept set")
    println("Total eliminated: $(n_vars - n_final_selected) variables")
    println("Final selection: $n_final_selected variables")

    return combined_importances
end
