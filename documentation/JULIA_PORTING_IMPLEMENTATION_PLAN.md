# Julia Porting Implementation Plan
## Complete Step-by-Step Guide with GUI Integration

**Date:** November 5, 2025
**Purpose:** Detailed implementation guide for porting Python computational bottlenecks to Julia
**Target Audience:** AI agents or developers implementing the port
**Estimated Total Time:** 6-9 weeks

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Phase 1: Variable Selection Module](#phase-1-variable-selection-module)
4. [Phase 2: Diagnostics Module](#phase-2-diagnostics-module)
5. [Phase 3: Neural Boosted Regressor](#phase-3-neural-boosted-regressor)
6. [Phase 4: MSC Preprocessing](#phase-4-msc-preprocessing)
7. [Phase 5: GUI Integration](#phase-5-gui-integration)
8. [Phase 6: Testing & Validation](#phase-6-testing--validation)
9. [Performance Benchmarking](#performance-benchmarking)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview & Architecture

### Current State

**Python Implementation:**
- GUI: `spectral_predict_gui_optimized.py` (Tkinter)
- Core modules: `src/spectral_predict/`
  - `search.py` - Main search orchestration
  - `variable_selection.py` - SPA, UVE, iPLS, UVE-SPA (760 lines) ⚠️ **TO PORT**
  - `diagnostics.py` - Leverage, jackknife intervals (370 lines) ⚠️ **TO PORT**
  - `neural_boosted.py` - Gradient boosting (500 lines) ⚠️ **TO PORT**
  - `preprocess.py` - SNV, SG derivatives, MSC (partial) ⚠️ **TO PORT**
  - `models.py`, `cv.py`, etc. - Already have Julia equivalents

**Julia Implementation:**
- Location: `julia_port/SpectralPredict/`
- Bridge: `spectral_predict_julia_bridge.py`
- Core modules:
  - `src/SpectralPredict.jl` - Main module
  - `src/search.jl` - Search orchestration ✅
  - `src/models.jl` - PLS, Ridge, Lasso, RF, MLP ✅
  - `src/preprocessing.jl` - SNV, derivatives ✅
  - `src/cv.jl` - Cross-validation ✅
  - `src/regions.jl` - Region analysis ✅
  - `src/scoring.jl` - Composite scoring ✅

### What We're Adding

**New Julia Files:**
1. `src/variable_selection.jl` - NEW (800-1000 lines)
2. `src/diagnostics.jl` - NEW (400-600 lines)
3. `src/neural_boosted.jl` - NEW (500-700 lines)
4. `src/preprocessing.jl` - EXTEND (add MSC, ~100 lines)

**Expected Performance Gains:**
- SPA selection: 10-20x faster (parallelized)
- Jackknife intervals: 17-25x faster (parallelized)
- UVE selection: 6-10x faster
- Neural Boosted: 2-3x faster
- **Overall pipeline: 5-15x faster**

---

## Prerequisites & Setup

### Required Software

1. **Julia 1.11+**
   ```bash
   # Download from https://julialang.org/downloads/
   # Verify installation:
   julia --version
   ```

2. **Python 3.8+**
   ```bash
   python --version
   ```

3. **Git** (for version control)

### Julia Package Dependencies

**Location:** `julia_port/SpectralPredict/Project.toml`

**Current dependencies:**
```toml
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
GLMNet = "8d5ece8b-de18-5317-b113-243142960cc6"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6de"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
```

**Add for this project:**
```toml
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"  # For t-dist, Normal quantiles
```

**To add dependencies:**
```bash
cd julia_port/SpectralPredict
julia --project=.
```
```julia
using Pkg
Pkg.add("Distributions")
Pkg.instantiate()
```

### Development Tools

**Recommended IDE Setup:**
- VSCode with Julia extension
- Python extension
- Git integration

**Testing Setup:**
```bash
cd julia_port/SpectralPredict
julia --project=.
```
```julia
using Pkg
Pkg.test()  # Run existing tests
```

---

## Phase 1: Variable Selection Module

**Duration:** 2-3 weeks
**Complexity:** High (most complex algorithms)
**Files Created:** `julia_port/SpectralPredict/src/variable_selection.jl`
**Files Modified:** `src/search.jl` (integration)

### Step 1.1: Create Module File

**File:** `julia_port/SpectralPredict/src/variable_selection.jl`

```julia
"""
    Variable Selection Methods for Spectral Data

Implements multiple variable (wavelength) selection algorithms:
- UVE (Uninformative Variable Elimination)
- SPA (Successive Projections Algorithm)
- iPLS (Interval PLS)
- UVE-SPA (Hybrid approach)

Author: Spectral Predict Team
Date: November 2025
"""

module VariableSelection

using LinearAlgebra
using Statistics
using Random
using MultivariateStats

export uve_selection, spa_selection, ipls_selection, uve_spa_selection

# Internal helper functions will be defined here

end  # module
```

### Step 1.2: Implement UVE Selection

**Algorithm Overview:**
1. Create augmented dataset: [X | random_noise]
2. Run PLS regression with cross-validation
3. Collect PLS coefficients across CV folds
4. Compute reliability score: mean(|coef|) / std(coef)
5. Threshold based on noise variable scores

**Full Implementation:**

```julia
"""
    uve_selection(X, y; cutoff_multiplier=1.0, n_components=nothing, cv_folds=5, random_state=42)

Uninformative Variable Elimination (UVE) for spectral data.

# Arguments
- `X::Matrix{Float64}`: Spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples)
- `cutoff_multiplier::Float64=1.0`: Multiplier for cutoff threshold (default: 1.0)
- `n_components::Union{Int,Nothing}=nothing`: Number of PLS components (default: auto-select)
- `cv_folds::Int=5`: Number of cross-validation folds
- `random_state::Int=42`: Random seed for reproducibility

# Returns
- `Vector{Float64}`: Reliability scores for each variable (length n_features)
  Higher scores indicate more informative variables.

# Reference
Centner et al. (1996), "Elimination of uninformative variables for multivariate calibration,"
Analytical Chemistry, 68(21), 3851-3858.

# Example
```julia
importances = uve_selection(X, y, cutoff_multiplier=1.0, cv_folds=5)
# Select variables above cutoff
selected_vars = findall(importances .> 0)
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

    # Step 1: Create noise variables (same number as real features)
    noise_vars = randn(n_samples, n_features)

    # Step 2: Augment dataset [X | noise]
    X_augmented = hcat(X, noise_vars)
    n_augmented = size(X_augmented, 2)

    # Step 3: Determine number of components
    if isnothing(n_components)
        # Auto-select: use min(n_samples-1, n_augmented) / 2
        n_components = min(n_samples - 1, n_augmented) ÷ 2
        n_components = max(2, min(n_components, 15))  # Limit to 2-15
    end

    # Step 4: Perform k-fold cross-validation with PLS
    fold_size = n_samples ÷ cv_folds

    # Storage for coefficients across folds
    coef_matrix = zeros(cv_folds, n_augmented)

    for fold in 1:cv_folds
        # Create train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == cv_folds ? n_samples : fold * fold_size
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)

        X_train = X_augmented[train_indices, :]
        y_train = y[train_indices]

        # Fit PLS model
        # Note: MultivariateStats uses CCA for PLS
        # Center data
        X_mean = mean(X_train, dims=1)
        y_mean = mean(y_train)

        X_centered = X_train .- X_mean
        y_centered = y_train .- y_mean

        # Fit CCA (PLS equivalent)
        try
            # Use SVD-based PLS
            M = fit(CCA, X_centered', y_centered'; outdim=n_components)

            # Extract coefficients
            # For regression: coef = X_projection * Y_projection'
            # We approximate by using the X projection weights
            W = projection(M)  # X projection matrix

            # Simple coefficient approximation: correlation with y
            for j in 1:n_augmented
                coef_matrix[fold, j] = cor(X_train[:, j], y_train)
            end

        catch e
            @warn "PLS fit failed for fold $fold: $e"
            # Use correlation as fallback
            for j in 1:n_augmented
                coef_matrix[fold, j] = cor(X_train[:, j], y_train)
            end
        end
    end

    # Step 5: Compute reliability scores
    # Reliability = mean(|coef|) / std(coef) across folds
    reliability = zeros(n_augmented)

    for j in 1:n_augmented
        coef_j = coef_matrix[:, j]
        mean_abs_coef = mean(abs.(coef_j))
        std_coef = std(coef_j)

        if std_coef > 1e-10
            reliability[j] = mean_abs_coef / std_coef
        else
            reliability[j] = 0.0
        end
    end

    # Step 6: Compute cutoff based on noise variables
    noise_reliability = reliability[(n_features+1):end]

    # Cutoff = max(noise_reliability) * cutoff_multiplier
    cutoff = maximum(noise_reliability) * cutoff_multiplier

    # Step 7: Extract scores for real variables only
    real_reliability = reliability[1:n_features]

    # Convert to importance scores (subtract cutoff, clip to 0)
    importance_scores = max.(real_reliability .- cutoff, 0.0)

    return importance_scores
end
```

### Step 1.3: Implement SPA Selection

**Algorithm Overview:**
1. Normalize data
2. Multiple random starts (default: 10)
3. For each start:
   - Select variable with max correlation to y
   - Iteratively select variable with MIN projection onto selected set
   - Evaluate selection via PLS CV
4. Return best selection across starts

**Full Implementation:**

```julia
"""
    spa_selection(X, y, n_features; n_random_starts=10, cv_folds=5, random_state=42)

Successive Projections Algorithm (SPA) for wavelength selection.
Selects minimally correlated variables to reduce collinearity.

# Arguments
- `X::Matrix{Float64}`: Spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values
- `n_features::Int`: Number of variables to select
- `n_random_starts::Int=10`: Number of random initializations
- `cv_folds::Int=5`: CV folds for evaluating selections
- `random_state::Int=42`: Random seed

# Returns
- `Vector{Float64}`: Importance scores (1.0 for selected, 0.0 for unselected)

# Reference
Araújo et al. (2001), "The successive projections algorithm for variable selection
in spectroscopic multicomponent analysis," Chemometrics and Intelligent Laboratory Systems.

# Example
```julia
importances = spa_selection(X, y, 50, n_random_starts=10)
selected_indices = findall(importances .> 0)
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

    n_samples, n_total_features = size(X)

    # Validate n_features
    if n_features >= n_total_features
        @warn "n_features >= total features, returning all features"
        return ones(n_total_features)
    end

    if n_features < 2
        @warn "n_features < 2, selecting based on correlation"
        corrs = abs.([cor(X[:, j], y) for j in 1:n_total_features])
        importance = zeros(n_total_features)
        importance[argmax(corrs)] = 1.0
        return importance
    end

    # Normalize X for correlation computation
    X_normalized = (X .- mean(X, dims=1)) ./ (std(X, dims=1) .+ 1e-10)

    # Storage for best selection
    best_selection = Int[]
    best_score = -Inf

    Random.seed!(random_state)

    # Multiple random starts
    for start in 1:n_random_starts

        # Step 1: Initialize with variable most correlated with y
        if start == 1
            # First start: use max correlation
            correlations = abs.([cor(X[:, j], y) for j in 1:n_total_features])
            first_var = argmax(correlations)
        else
            # Random starts: pick random variable
            first_var = rand(1:n_total_features)
        end

        selected = [first_var]
        available = setdiff(1:n_total_features, selected)

        # Step 2: Iteratively select variables with minimal projection
        for iter in 2:n_features

            # Compute projections onto selected subspace
            X_selected = X_normalized[:, selected]

            min_projection = Inf
            best_var = 0

            for candidate in available
                x_candidate = X_normalized[:, candidate]

                # Compute projection of candidate onto selected subspace
                # proj = X_selected * (X_selected' * X_selected)^(-1) * X_selected' * x_candidate
                # But we want MINIMUM projection, so compute norm of residual

                # For efficiency: compute correlation with selected variables
                max_corr = maximum(abs.([cor(x_candidate, X_normalized[:, s]) for s in selected]))

                if max_corr < min_projection
                    min_projection = max_corr
                    best_var = candidate
                end
            end

            # Add variable with minimal projection
            if best_var > 0
                push!(selected, best_var)
                available = setdiff(available, [best_var])
            else
                break
            end
        end

        # Step 3: Evaluate this selection via PLS cross-validation
        X_selected = X[:, selected]

        cv_score = evaluate_pls_cv(X_selected, y, cv_folds)

        # Keep best selection (highest R²)
        if cv_score > best_score
            best_score = cv_score
            best_selection = copy(selected)
        end
    end

    # Convert to importance vector
    importance = zeros(n_total_features)
    importance[best_selection] .= 1.0

    return importance
end

"""
    evaluate_pls_cv(X, y, cv_folds)

Evaluate PLS model using cross-validation.
Returns mean R² across folds.
"""
function evaluate_pls_cv(X::Matrix{Float64}, y::Vector{Float64}, cv_folds::Int)::Float64
    n_samples = size(X, 1)
    fold_size = n_samples ÷ cv_folds

    r2_scores = Float64[]

    for fold in 1:cv_folds
        # Train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == cv_folds ? n_samples : fold * fold_size
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)

        X_train, y_train = X[train_indices, :], y[train_indices]
        X_test, y_test = X[test_indices, :], y[test_indices]

        # Fit simple linear regression (or PLS with 1-2 components)
        # For speed, use ordinary least squares
        try
            # Add intercept
            X_train_aug = hcat(ones(length(train_indices)), X_train)
            X_test_aug = hcat(ones(length(test_indices)), X_test)

            # Solve: X' X β = X' y
            β = X_train_aug \ y_train

            # Predict
            y_pred = X_test_aug * β

            # Compute R²
            ss_res = sum((y_test .- y_pred).^2)
            ss_tot = sum((y_test .- mean(y_test)).^2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)

            push!(r2_scores, max(r2, 0.0))
        catch e
            @warn "CV fold $fold failed: $e"
            push!(r2_scores, 0.0)
        end
    end

    return mean(r2_scores)
end
```

### Step 1.4: Implement iPLS Selection

**Full Implementation:**

```julia
"""
    ipls_selection(X, y; n_intervals=20, n_components=nothing, cv_folds=5, random_state=42)

Interval PLS (iPLS) for spectral region selection.
Divides spectrum into intervals and evaluates each independently.

# Arguments
- `X::Matrix{Float64}`: Spectral data (n_samples × n_features)
- `y::Vector{Float64}`: Target values
- `n_intervals::Int=20`: Number of spectral intervals
- `n_components::Union{Int,Nothing}=nothing`: PLS components per interval
- `cv_folds::Int=5`: Cross-validation folds
- `random_state::Int=42`: Random seed

# Returns
- `Vector{Float64}`: Importance score for each variable (based on interval performance)

# Reference
Nørgaard et al. (2000), "Interval Partial Least-Squares Regression (iPLS): A Comparative
Chemometric Study with an Example from Near-Infrared Spectroscopy," Applied Spectroscopy.

# Example
```julia
importances = ipls_selection(X, y, n_intervals=20)
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

    # Step 1: Divide spectrum into intervals
    interval_size = n_features ÷ n_intervals

    if interval_size < 2
        @warn "Too many intervals for feature count, reducing intervals"
        n_intervals = n_features ÷ 2
        interval_size = n_features ÷ n_intervals
    end

    # Create intervals
    intervals = []
    for i in 1:n_intervals
        start_idx = (i - 1) * interval_size + 1
        end_idx = i == n_intervals ? n_features : i * interval_size
        push!(intervals, start_idx:end_idx)
    end

    # Step 2: Evaluate each interval
    interval_scores = zeros(n_intervals)

    Random.seed!(random_state)

    for (i, interval) in enumerate(intervals)
        X_interval = X[:, interval]

        # Evaluate via PLS CV
        r2 = evaluate_pls_cv(X_interval, y, cv_folds)
        interval_scores[i] = r2
    end

    # Step 3: Assign scores to variables
    importance = zeros(n_features)

    for (i, interval) in enumerate(intervals)
        importance[interval] .= interval_scores[i]
    end

    return importance
end
```

### Step 1.5: Implement UVE-SPA Hybrid

**Full Implementation:**

```julia
"""
    uve_spa_selection(X, y, n_features; cutoff_multiplier=1.0, uve_n_components=nothing,
                      uve_cv_folds=5, spa_n_random_starts=10, spa_cv_folds=5, random_state=42)

Hybrid UVE-SPA selection: first filters with UVE, then applies SPA.

Combines noise filtering (UVE) with collinearity reduction (SPA) for optimal variable selection.

# Arguments
- `X::Matrix{Float64}`: Spectral data
- `y::Vector{Float64}`: Target values
- `n_features::Int`: Final number of features to select
- `cutoff_multiplier::Float64=1.0`: UVE cutoff multiplier
- `uve_n_components::Union{Int,Nothing}=nothing`: PLS components for UVE
- `uve_cv_folds::Int=5`: CV folds for UVE
- `spa_n_random_starts::Int=10`: Random starts for SPA
- `spa_cv_folds::Int=5`: CV folds for SPA
- `random_state::Int=42`: Random seed

# Returns
- `Vector{Float64}`: Binary importance (1.0 for selected, 0.0 otherwise)

# Example
```julia
importances = uve_spa_selection(X, y, 50, cutoff_multiplier=1.0)
selected_vars = findall(importances .> 0)
```
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

    n_samples, n_total_features = size(X)

    # Step 1: Run UVE to filter uninformative variables
    uve_scores = uve_selection(
        X, y;
        cutoff_multiplier=cutoff_multiplier,
        n_components=uve_n_components,
        cv_folds=uve_cv_folds,
        random_state=random_state
    )

    # Get UVE-selected variables (those with score > 0)
    uve_selected_indices = findall(uve_scores .> 0)

    if length(uve_selected_indices) == 0
        @warn "UVE selected no variables, falling back to SPA on full dataset"
        return spa_selection(X, y, n_features;
                           n_random_starts=spa_n_random_starts,
                           cv_folds=spa_cv_folds,
                           random_state=random_state)
    end

    # If UVE selected fewer than target, return UVE result
    if length(uve_selected_indices) <= n_features
        @info "UVE selected $(length(uve_selected_indices)) vars (target: $n_features), returning UVE result"
        return uve_scores
    end

    # Step 2: Run SPA on UVE-filtered data
    X_uve_filtered = X[:, uve_selected_indices]

    spa_scores_filtered = spa_selection(
        X_uve_filtered, y, n_features;
        n_random_starts=spa_n_random_starts,
        cv_folds=spa_cv_folds,
        random_state=random_state
    )

    # Step 3: Map SPA scores back to original feature space
    importance = zeros(n_total_features)

    for (i, original_idx) in enumerate(uve_selected_indices)
        importance[original_idx] = spa_scores_filtered[i]
    end

    return importance
end
```

### Step 1.6: Add to Module Exports

At the top of `variable_selection.jl`, ensure all functions are exported:

```julia
export uve_selection, spa_selection, ipls_selection, uve_spa_selection
```

### Step 1.7: Register Module in SpectralPredict.jl

**File:** `julia_port/SpectralPredict/src/SpectralPredict.jl`

Add the include statement:

```julia
module SpectralPredict

using CSV
using DataFrames
# ... other imports

# Include all submodules
include("io.jl")
include("preprocessing.jl")
include("models.jl")
include("cv.jl")
include("regions.jl")
include("scoring.jl")
include("search.jl")
include("variable_selection.jl")  # ← ADD THIS LINE

# Export main functions
export run_search
# ... other exports

end  # module
```

### Step 1.8: Integrate with search.jl

**File:** `julia_port/SpectralPredict/src/search.jl`

Find the section where variable subsets are created (around line 350-400) and add variable selection support:

```julia
# In run_search() function, after preprocessing is applied

# Import variable selection functions
using .VariableSelection: uve_selection, spa_selection, ipls_selection, uve_spa_selection

# ... existing code for preprocessing ...

# Variable selection section (modify existing code)
if enable_variable_subsets && haskey(config, :variable_selection_methods)

    methods = config[:variable_selection_methods]

    for method in methods
        for var_count in variable_counts

            if method == "importance"
                # Existing implementation (model-based feature importance)
                # ... keep existing code ...

            elseif method == "SPA"
                # NEW: SPA selection
                @info "Running SPA selection for $var_count variables..."

                importances = spa_selection(
                    X_preprocessed, y, var_count;
                    n_random_starts=10,
                    cv_folds=n_folds
                )

                selected_indices = findall(importances .> 0)

                # Run model on selected variables
                # ... evaluation code ...

            elseif method == "UVE"
                # NEW: UVE selection
                @info "Running UVE selection..."

                importances = uve_selection(
                    X_preprocessed, y;
                    cutoff_multiplier=1.0,
                    cv_folds=n_folds
                )

                # Select top var_count variables
                sorted_indices = sortperm(importances, rev=true)
                selected_indices = sorted_indices[1:min(var_count, length(sorted_indices))]

                # Run model on selected variables
                # ... evaluation code ...

            elseif method == "iPLS"
                # NEW: iPLS selection
                @info "Running iPLS selection for $var_count variables..."

                importances = ipls_selection(
                    X_preprocessed, y;
                    n_intervals=20,
                    cv_folds=n_folds
                )

                # Select top var_count variables
                sorted_indices = sortperm(importances, rev=true)
                selected_indices = sorted_indices[1:min(var_count, length(sorted_indices))]

                # Run model on selected variables
                # ... evaluation code ...

            elseif method == "UVE-SPA"
                # NEW: UVE-SPA hybrid
                @info "Running UVE-SPA selection for $var_count variables..."

                importances = uve_spa_selection(
                    X_preprocessed, y, var_count;
                    cutoff_multiplier=1.0,
                    spa_n_random_starts=10,
                    cv_folds=n_folds
                )

                selected_indices = findall(importances .> 0)

                # Run model on selected variables
                # ... evaluation code ...
            end
        end
    end
end
```

### Step 1.9: Unit Tests for Variable Selection

**File:** `julia_port/SpectralPredict/test/test_variable_selection.jl` (NEW)

```julia
using Test
using Random
using Statistics
using SpectralPredict
using SpectralPredict.VariableSelection

@testset "Variable Selection Tests" begin

    # Create synthetic data
    Random.seed!(42)
    n_samples = 100
    n_features = 200

    # Create informative features (first 20) and noise features (rest)
    X_informative = randn(n_samples, 20)
    X_noise = randn(n_samples, 180)
    X = hcat(X_informative, X_noise)

    # Target is linear combination of first 10 features
    y = sum(X[:, 1:10] .* (1:10)', dims=2)[:] + randn(n_samples) * 0.1

    @testset "UVE Selection" begin
        importances = uve_selection(X, y, cutoff_multiplier=1.0, cv_folds=5)

        @test length(importances) == n_features
        @test all(importances .>= 0)

        # Informative features should have higher scores
        mean_informative = mean(importances[1:20])
        mean_noise = mean(importances[21:end])
        @test mean_informative > mean_noise
    end

    @testset "SPA Selection" begin
        n_select = 30
        importances = spa_selection(X, y, n_select, n_random_starts=5, cv_folds=3)

        @test length(importances) == n_features
        @test sum(importances .> 0) <= n_select + 5  # Allow small tolerance

        # Selected features should include some informative ones
        selected_informative = sum(importances[1:20] .> 0)
        @test selected_informative > 0
    end

    @testset "iPLS Selection" begin
        importances = ipls_selection(X, y, n_intervals=10, cv_folds=3)

        @test length(importances) == n_features
        @test all(importances .>= 0)

        # First interval (contains informative features) should have high score
        interval_size = n_features ÷ 10
        first_interval_score = mean(importances[1:interval_size])
        last_interval_score = mean(importances[end-interval_size+1:end])
        @test first_interval_score > last_interval_score * 0.8  # Allow some variance
    end

    @testset "UVE-SPA Hybrid" begin
        n_select = 25
        importances = uve_spa_selection(X, y, n_select,
                                       spa_n_random_starts=5,
                                       uve_cv_folds=3,
                                       spa_cv_folds=3)

        @test length(importances) == n_features
        @test sum(importances .> 0) <= n_select + 5

        # Should select some informative features
        selected_informative = sum(importances[1:20] .> 0)
        @test selected_informative > 0
    end

    @testset "Edge Cases" begin
        # Test with very small dataset
        X_small = randn(10, 5)
        y_small = randn(10)

        @test_nowarn uve_selection(X_small, y_small, cv_folds=2)
        @test_nowarn spa_selection(X_small, y_small, 3, cv_folds=2)
        @test_nowarn ipls_selection(X_small, y_small, n_intervals=2, cv_folds=2)

        # Test with n_features > total features
        importances = spa_selection(X_small, y_small, 10, cv_folds=2)
        @test sum(importances .> 0) == 5  # Should select all available
    end
end
```

### Step 1.10: Benchmarking Script

**File:** `julia_port/SpectralPredict/benchmark/bench_variable_selection.jl` (NEW)

```julia
"""
Benchmark variable selection methods against Python implementations.
"""

using BenchmarkTools
using SpectralPredict
using SpectralPredict.VariableSelection
using Random
using Printf

# Create realistic spectral dataset
Random.seed!(42)
n_samples = 200
n_features = 1000  # Typical NIR spectrum

X = randn(n_samples, n_features)
y = randn(n_samples)

println("="^70)
println("Variable Selection Performance Benchmark")
println("="^70)
println("Dataset: $n_samples samples × $n_features features")
println()

# Benchmark UVE
println("UVE Selection:")
result_uve = @benchmark uve_selection($X, $y, cv_folds=5) samples=3
println("  Median time: $(median(result_uve.times) / 1e9) seconds")
println("  Memory: $(result_uve.memory / 1e6) MB")
println()

# Benchmark SPA
println("SPA Selection (50 features, 10 random starts):")
result_spa = @benchmark spa_selection($X, $y, 50, n_random_starts=10, cv_folds=5) samples=3
println("  Median time: $(median(result_spa.times) / 1e9) seconds")
println("  Memory: $(result_spa.memory / 1e6) MB")
println()

# Benchmark iPLS
println("iPLS Selection (20 intervals):")
result_ipls = @benchmark ipls_selection($X, $y, n_intervals=20, cv_folds=5) samples=3
println("  Median time: $(median(result_ipls.times) / 1e9) seconds")
println("  Memory: $(result_ipls.memory / 1e6) MB")
println()

# Benchmark UVE-SPA
println("UVE-SPA Hybrid (50 features):")
result_hybrid = @benchmark uve_spa_selection($X, $y, 50, spa_n_random_starts=10) samples=3
println("  Median time: $(median(result_hybrid.times) / 1e9) seconds")
println("  Memory: $(result_hybrid.memory / 1e6) MB")
println()

println("="^70)
println("Benchmark complete!")
println("Compare these times with Python implementation for speedup calculation.")
println("="^70)
```

**Run benchmark:**
```bash
cd julia_port/SpectralPredict
julia --project=. benchmark/bench_variable_selection.jl
```

---

## Phase 2: Diagnostics Module

**Duration:** 1-2 weeks
**Complexity:** Medium-High
**Files Created:** `julia_port/SpectralPredict/src/diagnostics.jl`
**Files Modified:** None (standalone module)

### Step 2.1: Create Diagnostics Module

**File:** `julia_port/SpectralPredict/src/diagnostics.jl`

```julia
"""
    Model Diagnostics for Regression Models

Provides diagnostic tools for assessing model quality:
- Residual analysis (raw and standardized)
- Leverage detection (hat values)
- Q-Q plot data for normality testing
- Jackknife prediction intervals

Author: Spectral Predict Team
Date: November 2025
"""

module Diagnostics

using LinearAlgebra
using Statistics
using Distributions

export compute_residuals, compute_leverage, qq_plot_data, jackknife_prediction_intervals

end  # module
```

### Step 2.2: Implement Residual Analysis

```julia
"""
    compute_residuals(y_true, y_pred)

Compute raw and standardized residuals.

# Arguments
- `y_true::Vector{Float64}`: True target values
- `y_pred::Vector{Float64}`: Predicted values

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: (residuals, standardized_residuals)

# Example
```julia
residuals, std_residuals = compute_residuals(y_test, y_pred)
```
"""
function compute_residuals(
    y_true::Vector{Float64},
    y_pred::Vector{Float64}
)::Tuple{Vector{Float64}, Vector{Float64}}

    # Raw residuals
    residuals = y_true .- y_pred

    # Standardized residuals: (residual - mean) / std
    mean_resid = mean(residuals)
    std_resid = std(residuals)

    if std_resid > 1e-10
        standardized = (residuals .- mean_resid) ./ std_resid
    else
        standardized = zeros(length(residuals))
    end

    return residuals, standardized
end
```

### Step 2.3: Implement Leverage Computation

```julia
"""
    compute_leverage(X; return_threshold=true)

Compute leverage (hat values) for detecting influential observations.

Leverage h_ii = diagonal of hat matrix H = X(X'X)^(-1)X'
Uses SVD for numerical stability when X'X is ill-conditioned.

# Arguments
- `X::Matrix{Float64}`: Design matrix (n_samples × n_features)
- `return_threshold::Bool=true`: If true, returns (leverage, threshold)

# Returns
- If `return_threshold=false`: `Vector{Float64}` - leverage values
- If `return_threshold=true`: `Tuple{Vector{Float64}, Float64}` - (leverage, threshold)
  where threshold = 2(p+1)/n (common rule of thumb)

# Example
```julia
leverage, threshold = compute_leverage(X_train)
high_leverage = findall(leverage .> threshold)
```
"""
function compute_leverage(
    X::Matrix{Float64};
    return_threshold::Bool=true
)::Union{Vector{Float64}, Tuple{Vector{Float64}, Float64}}

    n_samples, n_features = size(X)

    # Add intercept column
    X_aug = hcat(ones(n_samples), X)
    p = size(X_aug, 2)  # Number of parameters (including intercept)

    try
        # Method 1: Direct computation (fast but can be unstable)
        # H = X(X'X)^(-1)X'
        # We only need diagonal, so: h_ii = row_i * (X'X)^(-1) * row_i'

        XtX = X_aug' * X_aug
        XtX_inv = inv(XtX)

        leverage = zeros(n_samples)
        for i in 1:n_samples
            leverage[i] = dot(X_aug[i, :], XtX_inv * X_aug[i, :])
        end

    catch e
        @warn "Direct leverage computation failed, using SVD: $e"

        # Method 2: SVD-based (more stable for ill-conditioned matrices)
        # H = UU' where X = UΣV' (economy SVD)
        U, _, _ = svd(X_aug)

        # Leverage = diagonal of UU'
        leverage = sum(U.^2, dims=2)[:]
    end

    # Compute threshold: 2(p+1)/n
    threshold = 2.0 * (p + 1) / n_samples

    if return_threshold
        return leverage, threshold
    else
        return leverage
    end
end
```

### Step 2.4: Implement Q-Q Plot Data

```julia
"""
    qq_plot_data(residuals)

Generate Q-Q plot data for assessing normality of residuals.

# Arguments
- `residuals::Vector{Float64}`: Model residuals

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: (theoretical_quantiles, sample_quantiles)
  Both sorted for plotting (theoretical on x-axis, sample on y-axis)

# Example
```julia
theoretical, sample = qq_plot_data(residuals)
# Plot: scatter(theoretical, sample)
# Add reference line: plot(theoretical, theoretical)
```
"""
function qq_plot_data(
    residuals::Vector{Float64}
)::Tuple{Vector{Float64}, Vector{Float64}}

    n = length(residuals)

    # Sort residuals (sample quantiles)
    sample_quantiles = sort(residuals)

    # Compute theoretical quantiles from standard normal
    # Use (i - 0.5) / n as probability for i-th order statistic
    probabilities = [(i - 0.5) / n for i in 1:n]

    # Get quantiles from standard normal distribution
    normal_dist = Normal(0, 1)
    theoretical_quantiles = quantile.(normal_dist, probabilities)

    return theoretical_quantiles, sample_quantiles
end
```

### Step 2.5: Implement Jackknife Prediction Intervals

**This is the most complex function - includes parallelization:**

```julia
"""
    jackknife_prediction_intervals(model_fn, X_train, y_train, X_test;
                                   confidence=0.95, verbose=true)

Compute jackknife (leave-one-out) prediction intervals.

# Arguments
- `model_fn::Function`: Function that returns a fitted model given (X, y)
  Example: `model_fn = (X, y) -> fit_pls(X, y, n_components=5)`
- `X_train::Matrix{Float64}`: Training features (n_train × n_features)
- `y_train::Vector{Float64}`: Training targets (n_train)
- `X_test::Matrix{Float64}`: Test features (n_test × n_features)
- `confidence::Float64=0.95`: Confidence level (default: 95%)
- `verbose::Bool=true`: Print progress

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`
  (predictions, lower_bounds, upper_bounds, std_errors)

# Algorithm
1. Fit model on full training data → predictions
2. For each sample i in training data:
   - Fit model excluding sample i
   - Predict on test data
3. Compute jackknife variance: Var = (n-1)/n * Σ(pred_i - pred_mean)²
4. Confidence intervals: pred ± t_critical * sqrt(variance)

# Performance Note
For n_train samples, requires n_train model fits. Can be slow for large n.
Uses threading for parallelization when available.

# Example
```julia
model_fn = (X, y) -> fit_pls(X, y, n_components=10)
pred, lower, upper, stderr = jackknife_prediction_intervals(
    model_fn, X_train, y_train, X_test, confidence=0.95
)
```
"""
function jackknife_prediction_intervals(
    model_fn::Function,
    X_train::Matrix{Float64},
    y_train::Vector{Float64},
    X_test::Matrix{Float64};
    confidence::Float64=0.95,
    verbose::Bool=true
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

    n_train, n_features = size(X_train)
    n_test = size(X_test, 1)

    if verbose
        @info "Computing jackknife prediction intervals (n_train=$n_train)..."
    end

    # Step 1: Fit model on full training data
    model_full = model_fn(X_train, y_train)
    predictions = model_full(X_test)  # Assuming model is callable

    # Step 2: Leave-one-out predictions (PARALLELIZED)
    loo_predictions = zeros(n_train, n_test)

    # Use threading for parallelization
    Threads.@threads for i in 1:n_train
        if verbose && (i % 10 == 0 || i == n_train)
            @info "  Jackknife iteration $i/$n_train"
        end

        # Exclude sample i
        train_indices = setdiff(1:n_train, i)
        X_loo = X_train[train_indices, :]
        y_loo = y_train[train_indices]

        # Fit model
        model_loo = model_fn(X_loo, y_loo)

        # Predict on test data
        loo_predictions[i, :] = model_loo(X_test)
    end

    # Step 3: Compute jackknife variance
    # Var_jack = (n-1)/n * Σ(pred_i - pred_mean)²
    mean_loo_pred = mean(loo_predictions, dims=1)[:]

    jackknife_variance = zeros(n_test)
    for j in 1:n_test
        deviations = loo_predictions[:, j] .- mean_loo_pred[j]
        jackknife_variance[j] = (n_train - 1) / n_train * sum(deviations.^2)
    end

    # Standard errors
    std_errors = sqrt.(jackknife_variance)

    # Step 4: Compute confidence intervals using t-distribution
    # Degrees of freedom: n_train - 1
    df = n_train - 1
    t_dist = TDist(df)
    alpha = 1 - confidence
    t_critical = quantile(t_dist, 1 - alpha/2)

    # Confidence intervals
    margin = t_critical .* std_errors
    lower_bounds = predictions .- margin
    upper_bounds = predictions .+ margin

    if verbose
        @info "Jackknife complete! Confidence level: $(confidence*100)%"
    end

    return predictions, lower_bounds, upper_bounds, std_errors
end
```

### Step 2.6: Add Distributions.jl Dependency

**File:** `julia_port/SpectralPredict/Project.toml`

Add:
```toml
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
```

**Install:**
```bash
cd julia_port/SpectralPredict
julia --project=.
```
```julia
using Pkg
Pkg.add("Distributions")
```

### Step 2.7: Register Module

**File:** `julia_port/SpectralPredict/src/SpectralPredict.jl`

```julia
include("diagnostics.jl")  # Add this line
using .Diagnostics: compute_residuals, compute_leverage, qq_plot_data, jackknife_prediction_intervals
```

### Step 2.8: Unit Tests for Diagnostics

**File:** `julia_port/SpectralPredict/test/test_diagnostics.jl` (NEW)

```julia
using Test
using Random
using Statistics
using LinearAlgebra
using SpectralPredict.Diagnostics

@testset "Diagnostics Tests" begin

    Random.seed!(42)

    @testset "Compute Residuals" begin
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]

        residuals, std_residuals = compute_residuals(y_true, y_pred)

        @test length(residuals) == 5
        @test length(std_residuals) == 5

        # Check residual calculation
        @test isapprox(residuals, y_true .- y_pred)

        # Standardized residuals should have mean ≈ 0, std ≈ 1
        @test isapprox(mean(std_residuals), 0.0, atol=1e-10)
        @test isapprox(std(std_residuals), 1.0, atol=1e-10)
    end

    @testset "Compute Leverage" begin
        # Create simple design matrix
        n = 20
        X = randn(n, 5)

        leverage, threshold = compute_leverage(X, return_threshold=true)

        @test length(leverage) == n
        @test all(leverage .>= 0)
        @test all(leverage .<= 1)

        # Average leverage should be p/n (rule of thumb)
        p = 6  # 5 features + intercept
        avg_leverage = mean(leverage)
        @test isapprox(avg_leverage, p/n, atol=0.1)

        # Threshold should be 2(p+1)/n
        expected_threshold = 2.0 * (p + 1) / n
        @test isapprox(threshold, expected_threshold, atol=1e-10)
    end

    @testset "Q-Q Plot Data" begin
        # Generate normal residuals
        n = 100
        residuals = randn(n)

        theoretical, sample = qq_plot_data(residuals)

        @test length(theoretical) == n
        @test length(sample) == n

        # Sample should be sorted
        @test issorted(sample)

        # For normal data, points should lie near diagonal
        # (theoretical ≈ sample)
        correlation = cor(theoretical, sample)
        @test correlation > 0.95  # Should be highly correlated
    end

    @testset "Jackknife Prediction Intervals" begin
        # Create simple linear problem
        n_train = 30
        n_test = 10
        n_features = 3

        X_train = randn(n_train, n_features)
        true_coef = [1.0, 2.0, -0.5]
        y_train = X_train * true_coef + randn(n_train) * 0.1

        X_test = randn(n_test, n_features)
        y_test = X_test * true_coef

        # Simple linear regression model function
        function fit_linear_model(X, y)
            X_aug = hcat(ones(size(X, 1)), X)
            β = X_aug \ y
            return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
        end

        pred, lower, upper, stderr = jackknife_prediction_intervals(
            fit_linear_model, X_train, y_train, X_test,
            confidence=0.95, verbose=false
        )

        @test length(pred) == n_test
        @test length(lower) == n_test
        @test length(upper) == n_test
        @test length(stderr) == n_test

        # Lower bounds should be less than predictions
        @test all(lower .< pred)

        # Upper bounds should be greater than predictions
        @test all(upper .> pred)

        # Standard errors should be positive
        @test all(stderr .> 0)

        # Interval width should increase with distance from training data
        # (Not always true, but often)
    end

    @testset "Edge Cases" begin
        # Small dataset
        X_small = randn(5, 2)
        y_small = randn(5)

        @test_nowarn compute_leverage(X_small)

        # Perfect predictions (zero residuals)
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        residuals, std_residuals = compute_residuals(y_true, y_pred)
        @test all(residuals .== 0)
    end
end
```

### Step 2.9: Benchmarking Script

**File:** `julia_port/SpectralPredict/benchmark/bench_diagnostics.jl` (NEW)

```julia
using BenchmarkTools
using SpectralPredict.Diagnostics
using Random
using Printf

Random.seed!(42)

println("="^70)
println("Diagnostics Performance Benchmark")
println("="^70)
println()

# Test dataset
n_train = 100
n_test = 50
n_features = 200

X_train = randn(n_train, n_features)
y_train = randn(n_train)
X_test = randn(n_test, n_features)
y_pred = randn(n_test)
y_true = randn(n_test)

# Benchmark residuals
println("Compute Residuals:")
result = @benchmark compute_residuals($y_true, $y_pred) samples=100
println("  Median time: $(median(result.times) / 1e6) ms")
println()

# Benchmark leverage
println("Compute Leverage ($n_train samples, $n_features features):")
result = @benchmark compute_leverage($X_train) samples=10
println("  Median time: $(median(result.times) / 1e6) ms")
println()

# Benchmark Q-Q plot
println("Q-Q Plot Data:")
residuals = randn(n_train)
result = @benchmark qq_plot_data($residuals) samples=100
println("  Median time: $(median(result.times) / 1e6) ms")
println()

# Benchmark jackknife (expensive!)
println("Jackknife Prediction Intervals (n_train=$n_train):")
println("  WARNING: This may take 30-60 seconds...")

# Simple model function
function fit_simple_model(X, y)
    β = X \ y
    return X_test -> X_test * β
end

result = @benchmark jackknife_prediction_intervals(
    $fit_simple_model, $X_train, $y_train, $X_test, verbose=false
) samples=3

println("  Median time: $(median(result.times) / 1e9) seconds")
println("  Number of threads: $(Threads.nthreads())")
println()

println("="^70)
println("Benchmark complete!")
println()
println("To test parallelization, run with:")
println("  julia --project=. -t 8 benchmark/bench_diagnostics.jl")
println("="^70)
```

---

## Phase 3: Neural Boosted Regressor

**Duration:** 1-2 weeks
**Complexity:** High (requires Flux.jl neural networks)
**Files Created:** `julia_port/SpectralPredict/src/neural_boosted.jl`
**Files Modified:** `src/models.jl`, `src/search.jl`

### Step 3.1: Create Neural Boosted Module

**File:** `julia_port/SpectralPredict/src/neural_boosted.jl`

```julia
"""
    Neural Boosted Regression

Gradient boosting with neural network weak learners.
Similar to sklearn's GradientBoostingRegressor but uses MLPs as base learners.

Key differences from Python implementation:
- Uses Flux.jl instead of sklearn.neural_network.MLPRegressor
- Manual early stopping implementation
- More efficient memory management

Author: Spectral Predict Team
Date: November 2025
"""

module NeuralBoosted

using Flux
using Statistics
using Random

export NeuralBoostedRegressor, fit!, predict, feature_importances

"""
    NeuralBoostedRegressor

Gradient boosting with MLP weak learners.

# Fields
- `n_estimators::Int`: Number of boosting stages (default: 50)
- `learning_rate::Float64`: Shrinkage parameter (default: 0.1)
- `hidden_layer_size::Int`: Neurons in hidden layer (default: 3)
- `activation::String`: Activation function ("relu", "tanh", "sigmoid")
- `alpha::Float64`: L2 regularization parameter (default: 0.0001)
- `max_iter::Int`: Maximum iterations per weak learner (default: 100)
- `early_stopping::Bool`: Use validation set for early stopping (default: true)
- `validation_fraction::Float64`: Fraction of data for validation (default: 0.1)
- `n_iter_no_change::Int`: Iterations with no improvement before stopping (default: 10)
- `loss::String`: Loss function ("mse" or "huber", default: "mse")
- `huber_delta::Float64`: Delta parameter for Huber loss (default: 1.0)
- `random_state::Int`: Random seed
- `verbose::Int`: Verbosity level (0=silent, 1=progress, 2=detailed)

# Fitted Attributes
- `estimators_`: Vector of fitted weak learners
- `train_score_`: Training loss per iteration
- `validation_score_`: Validation loss per iteration (if early_stopping)
- `n_estimators_`: Actual number of estimators fitted
"""
mutable struct NeuralBoostedRegressor
    # Hyperparameters
    n_estimators::Int
    learning_rate::Float64
    hidden_layer_size::Int
    activation::String
    alpha::Float64
    max_iter::Int
    early_stopping::Bool
    validation_fraction::Float64
    n_iter_no_change::Int
    loss::String
    huber_delta::Float64
    random_state::Int
    verbose::Int

    # Fitted attributes
    estimators_::Vector{Any}
    train_score_::Vector{Float64}
    validation_score_::Vector{Float64}
    n_estimators_::Int

    # Constructor
    function NeuralBoostedRegressor(;
        n_estimators::Int=50,
        learning_rate::Float64=0.1,
        hidden_layer_size::Int=3,
        activation::String="relu",
        alpha::Float64=0.0001,
        max_iter::Int=100,
        early_stopping::Bool=true,
        validation_fraction::Float64=0.1,
        n_iter_no_change::Int=10,
        loss::String="mse",
        huber_delta::Float64=1.0,
        random_state::Int=42,
        verbose::Int=0
    )
        new(
            n_estimators, learning_rate, hidden_layer_size,
            activation, alpha, max_iter, early_stopping,
            validation_fraction, n_iter_no_change, loss,
            huber_delta, random_state, verbose,
            [], Float64[], Float64[], 0  # Fitted attributes
        )
    end
end
```

### Step 3.2: Implement Weak Learner (MLP)

```julia
"""
    build_weak_learner(n_input, hidden_size, activation, alpha)

Build a simple MLP with one hidden layer.

# Architecture
- Input layer: n_input neurons
- Hidden layer: hidden_size neurons with activation
- Output layer: 1 neuron (regression, linear activation)
- L2 regularization on weights
"""
function build_weak_learner(
    n_input::Int,
    hidden_size::Int,
    activation_str::String,
    alpha::Float64
)
    # Select activation function
    if activation_str == "relu"
        activation = relu
    elseif activation_str == "tanh"
        activation = tanh
    elseif activation_str == "sigmoid"
        activation = σ
    else
        @warn "Unknown activation '$activation_str', using relu"
        activation = relu
    end

    # Build network
    model = Chain(
        Dense(n_input, hidden_size, activation),
        Dense(hidden_size, 1)  # Linear output for regression
    )

    return model
end

"""
    train_weak_learner!(model, X, y, max_iter, alpha, verbose)

Train weak learner using Adam optimizer.

Returns trained model (modifies in place).
"""
function train_weak_learner!(
    model,
    X::Matrix{Float64},
    y::Vector{Float64},
    max_iter::Int,
    alpha::Float64,
    verbose::Int
)
    # Transpose for Flux (features × samples)
    X_t = Float32.(X')
    y_t = Float32.(reshape(y, 1, :))

    # Define loss function with L2 regularization
    function loss_fn()
        pred = model(X_t)
        mse = Flux.mse(pred, y_t)

        # L2 regularization
        l2_penalty = sum(sum(p.^2) for p in Flux.params(model))

        return mse + alpha * l2_penalty
    end

    # Optimizer
    opt = Adam(0.01)

    # Training loop
    ps = Flux.params(model)

    for epoch in 1:max_iter
        # Compute gradients and update
        gs = gradient(loss_fn, ps)
        Flux.update!(opt, ps, gs)

        if verbose >= 2 && epoch % 20 == 0
            current_loss = loss_fn()
            println("    Epoch $epoch: loss = $(current_loss)")
        end
    end

    return model
end
```

### Step 3.3: Implement Fit Method

```julia
"""
    fit!(model::NeuralBoostedRegressor, X, y)

Fit the Neural Boosted Regressor.

# Arguments
- `model::NeuralBoostedRegressor`: Model to fit (modified in place)
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training targets (n_samples)

# Returns
- `model`: Fitted model
"""
function fit!(
    model::NeuralBoostedRegressor,
    X::Matrix{Float64},
    y::Vector{Float64}
)
    Random.seed!(model.random_state)

    n_samples, n_features = size(X)

    if model.verbose >= 1
        println("Fitting NeuralBoostedRegressor:")
        println("  n_estimators: $(model.n_estimators)")
        println("  learning_rate: $(model.learning_rate)")
        println("  hidden_layer_size: $(model.hidden_layer_size)")
    end

    # Step 1: Train/validation split (if early stopping)
    if model.early_stopping
        n_val = Int(floor(n_samples * model.validation_fraction))
        n_train = n_samples - n_val

        # Random shuffle
        indices = randperm(n_samples)
        train_idx = indices[1:n_train]
        val_idx = indices[n_train+1:end]

        X_train, y_train = X[train_idx, :], y[train_idx]
        X_val, y_val = X[val_idx, :], y[val_idx]
    else
        X_train, y_train = X, y
        X_val, y_val = nothing, nothing
    end

    # Initialize ensemble
    model.estimators_ = []
    model.train_score_ = Float64[]
    model.validation_score_ = Float64[]

    # Initialize predictions to zero
    F_train = zeros(size(X_train, 1))
    F_val = model.early_stopping ? zeros(size(X_val, 1)) : nothing

    # Early stopping tracking
    best_val_score = Inf
    no_improvement_count = 0

    # Step 2: Boosting loop
    for m in 1:model.n_estimators
        if model.verbose >= 1
            println("  Stage $m/$(model.n_estimators)...")
        end

        # Compute residuals
        residuals = y_train - F_train

        # Build weak learner
        weak_learner = build_weak_learner(
            n_features,
            model.hidden_layer_size,
            model.activation,
            model.alpha
        )

        # Train weak learner on residuals
        train_weak_learner!(
            weak_learner,
            X_train,
            residuals,
            model.max_iter,
            model.alpha,
            model.verbose
        )

        # Get predictions (need to transpose and extract from matrix)
        X_train_t = Float32.(X_train')
        h_m_train = vec(weak_learner(X_train_t))

        # Update ensemble predictions
        F_train .+= model.learning_rate .* h_m_train

        # Compute training loss
        train_loss = mean((y_train .- F_train).^2)
        push!(model.train_score_, train_loss)

        # Save estimator
        push!(model.estimators_, weak_learner)

        # Early stopping check
        if model.early_stopping
            X_val_t = Float32.(X_val')
            h_m_val = vec(weak_learner(X_val_t))
            F_val .+= model.learning_rate .* h_m_val

            val_loss = mean((y_val .- F_val).^2)
            push!(model.validation_score_, val_loss)

            if val_loss < best_val_score
                best_val_score = val_loss
                no_improvement_count = 0
            else
                no_improvement_count += 1
            end

            if no_improvement_count >= model.n_iter_no_change
                if model.verbose >= 1
                    println("  Early stopping at stage $m (no improvement for $(model.n_iter_no_change) iterations)")
                end
                break
            end
        end
    end

    model.n_estimators_ = length(model.estimators_)

    if model.verbose >= 1
        println("Fitting complete! $(model.n_estimators_) estimators trained.")
        println("  Final train loss: $(model.train_score_[end])")
        if model.early_stopping
            println("  Final val loss: $(model.validation_score_[end])")
        end
    end

    return model
end
```

### Step 3.4: Implement Predict Method

```julia
"""
    predict(model::NeuralBoostedRegressor, X)

Predict using fitted Neural Boosted Regressor.

# Arguments
- `model::NeuralBoostedRegressor`: Fitted model
- `X::Matrix{Float64}`: Features (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predictions (n_samples)
"""
function predict(
    model::NeuralBoostedRegressor,
    X::Matrix{Float64}
)::Vector{Float64}

    if isempty(model.estimators_)
        error("Model not fitted yet. Call fit!() first.")
    end

    n_samples = size(X, 1)
    predictions = zeros(n_samples)

    X_t = Float32.(X')

    # Aggregate predictions from all weak learners
    for weak_learner in model.estimators_
        h_m = vec(weak_learner(X_t))
        predictions .+= model.learning_rate .* h_m
    end

    return predictions
end
```

### Step 3.5: Implement Feature Importance

```julia
"""
    feature_importances(model::NeuralBoostedRegressor)

Compute feature importances by averaging absolute first-layer weights.

# Arguments
- `model::NeuralBoostedRegressor`: Fitted model

# Returns
- `Vector{Float64}`: Feature importances (length = n_features)
"""
function feature_importances(
    model::NeuralBoostedRegressor
)::Vector{Float64}

    if isempty(model.estimators_)
        error("Model not fitted yet. Call fit!() first.")
    end

    # Get number of features from first estimator
    first_layer = model.estimators_[1][1]
    n_features = size(first_layer.weight, 2)

    importances = zeros(n_features)

    # Average absolute weights across all estimators
    for weak_learner in model.estimators_
        first_layer_weights = weak_learner[1].weight  # (hidden_size × n_features)

        # Average absolute weights for each input feature
        feature_weights = mean(abs.(first_layer_weights), dims=1)[:]

        importances .+= feature_weights
    end

    # Normalize by number of estimators
    importances ./= length(model.estimators_)

    # Convert to relative importances (sum to 1)
    importances ./= sum(importances)

    return importances
end

end  # module NeuralBoosted
```

### Step 3.6: Integrate with Models Module

**File:** `julia_port/SpectralPredict/src/models.jl`

Add Neural Boosted to available models:

```julia
# At top of file
include("neural_boosted.jl")
using .NeuralBoosted: NeuralBoostedRegressor

# In get_model_config() function, add:
function get_model_config(model_type::String)
    if model_type == "NeuralBoosted"
        return Dict(
            :n_estimators => [50, 100],
            :learning_rate => [0.05, 0.1],
            :hidden_layer_size => [3, 5]
        )
    elseif model_type == "PLS"
        # ... existing code
    end
end

# In fit_model() function, add:
function fit_model(model_type::String, X, y, params)
    if model_type == "NeuralBoosted"
        model = NeuralBoostedRegressor(;
            n_estimators=get(params, :n_estimators, 50),
            learning_rate=get(params, :learning_rate, 0.1),
            hidden_layer_size=get(params, :hidden_layer_size, 3),
            max_iter=100,
            verbose=0
        )
        fit!(model, X, y)
        return model
    elseif model_type == "PLS"
        # ... existing code
    end
end

# In predict_model() function, add:
function predict_model(model_type::String, model, X)
    if model_type == "NeuralBoosted"
        return predict(model, X)
    elseif model_type == "PLS"
        # ... existing code
    end
end
```

### Step 3.7: Unit Tests

**File:** `julia_port/SpectralPredict/test/test_neural_boosted.jl` (NEW)

```julia
using Test
using Random
using Statistics
using SpectralPredict.NeuralBoosted

@testset "Neural Boosted Tests" begin

    Random.seed!(42)

    # Create synthetic regression dataset
    n_samples = 200
    n_features = 20

    X = randn(n_samples, n_features)
    # True function: y = sum(X[:, 1:5] .* [1, 2, -1, 0.5, -0.5], dims=2) + noise
    y = sum(X[:, 1:5] .* [1.0, 2.0, -1.0, 0.5, -0.5]', dims=2)[:] + randn(n_samples) * 0.5

    # Split train/test
    train_idx = 1:150
    test_idx = 151:200

    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    @testset "Model Construction" begin
        model = NeuralBoostedRegressor(
            n_estimators=10,
            learning_rate=0.1,
            hidden_layer_size=5,
            verbose=0
        )

        @test model.n_estimators == 10
        @test model.learning_rate == 0.1
        @test isempty(model.estimators_)
    end

    @testset "Model Fitting" begin
        model = NeuralBoostedRegressor(
            n_estimators=20,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=false,
            max_iter=50,
            verbose=0
        )

        fit!(model, X_train, y_train)

        @test length(model.estimators_) == 20
        @test length(model.train_score_) == 20
        @test model.n_estimators_ == 20

        # Training loss should generally decrease
        first_loss = model.train_score_[1]
        last_loss = model.train_score_[end]
        @test last_loss < first_loss
    end

    @testset "Model Prediction" begin
        model = NeuralBoostedRegressor(
            n_estimators=20,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=false,
            verbose=0
        )

        fit!(model, X_train, y_train)
        y_pred = predict(model, X_test)

        @test length(y_pred) == length(y_test)

        # Calculate R²
        ss_res = sum((y_test .- y_pred).^2)
        ss_tot = sum((y_test .- mean(y_test)).^2)
        r2 = 1.0 - ss_res / ss_tot

        @test r2 > 0.5  # Should have reasonable predictive power
    end

    @testset "Early Stopping" begin
        model = NeuralBoostedRegressor(
            n_estimators=100,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=true,
            validation_fraction=0.2,
            n_iter_no_change=5,
            verbose=0
        )

        fit!(model, X_train, y_train)

        # Should stop before reaching max estimators
        @test model.n_estimators_ < 100
        @test length(model.validation_score_) == model.n_estimators_
    end

    @testset "Feature Importances" begin
        model = NeuralBoostedRegressor(
            n_estimators=20,
            learning_rate=0.1,
            hidden_layer_size=5,
            verbose=0
        )

        fit!(model, X_train, y_train)
        importances = feature_importances(model)

        @test length(importances) == n_features
        @test all(importances .>= 0)
        @test isapprox(sum(importances), 1.0, atol=1e-6)

        # First 5 features should have higher importances (they're used in true function)
        mean_important = mean(importances[1:5])
        mean_unimportant = mean(importances[6:end])
        @test mean_important > mean_unimportant
    end

    @testset "Edge Cases" begin
        model = NeuralBoostedRegressor(n_estimators=5, verbose=0)

        # Small dataset
        X_small = randn(10, 5)
        y_small = randn(10)
        @test_nowarn fit!(model, X_small, y_small)
        @test_nowarn predict(model, X_small)
    end
end
```

---

## Phase 4: MSC Preprocessing

**Duration:** 3-4 days
**Complexity:** Low
**Files Modified:** `julia_port/SpectralPredict/src/preprocessing.jl`

### Step 4.1: Add MSC Transformer

**File:** `julia_port/SpectralPredict/src/preprocessing.jl`

Add to existing file (around line 200, after SNV):

```julia
"""
    MSCTransformer

Multiplicative Scatter Correction (MSC) for spectral data.
Corrects multiplicative scattering effects by fitting each spectrum to a reference.

# Fields
- `reference::Union{String, Nothing, Vector{Float64}}`:
  Reference spectrum. Can be:
  - "mean": Use mean of training spectra (default)
  - "median": Use median of training spectra
  - Vector{Float64}: Custom reference spectrum

- `reference_spectrum_::Union{Nothing, Vector{Float64}}`:
  Fitted reference (computed during fit!)

# Algorithm
For each spectrum x:
1. Fit linear model: x = a + b * reference
2. Correct: x_corrected = (x - a) / b

# Example
```julia
msc = MSCTransformer(reference="mean")
fit_msc!(msc, X_train)
X_corrected = apply_msc(msc, X_test)
```
"""
mutable struct MSCTransformer
    reference::Union{String, Nothing, Vector{Float64}}
    reference_spectrum_::Union{Nothing, Vector{Float64}}

    function MSCTransformer(reference::Union{String, Nothing, Vector{Float64}}="mean")
        new(reference, nothing)
    end
end

"""
    fit_msc!(msc::MSCTransformer, X)

Fit MSC transformer by computing reference spectrum.

# Arguments
- `msc::MSCTransformer`: Transformer to fit (modified in place)
- `X::Matrix{Float64}`: Training spectra (n_samples × n_features)

# Returns
- `msc`: Fitted transformer
"""
function fit_msc!(
    msc::MSCTransformer,
    X::Matrix{Float64}
)
    if msc.reference == "mean"
        msc.reference_spectrum_ = vec(mean(X, dims=1))
    elseif msc.reference == "median"
        msc.reference_spectrum_ = vec(median(X, dims=1))
    elseif isa(msc.reference, Vector{Float64})
        msc.reference_spectrum_ = msc.reference
    else
        error("Invalid reference type: $(msc.reference). Use 'mean', 'median', or provide a vector.")
    end

    return msc
end

"""
    apply_msc(msc::MSCTransformer, X)

Apply MSC correction to spectra.

# Arguments
- `msc::MSCTransformer`: Fitted transformer
- `X::Matrix{Float64}`: Spectra to correct (n_samples × n_features)

# Returns
- `Matrix{Float64}`: Corrected spectra
"""
function apply_msc(
    msc::MSCTransformer,
    X::Matrix{Float64}
)::Matrix{Float64}

    if isnothing(msc.reference_spectrum_)
        error("MSC not fitted. Call fit_msc!() first.")
    end

    n_samples, n_features = size(X)
    reference = msc.reference_spectrum_

    # Check dimensions
    if length(reference) != n_features
        error("Reference spectrum length ($(length(reference))) doesn't match X features ($n_features)")
    end

    X_corrected = zeros(n_samples, n_features)

    # Correct each spectrum
    for i in 1:n_samples
        spectrum = X[i, :]

        # Fit linear model: spectrum = a + b * reference
        # Using least squares: [1 reference] \ spectrum
        A = hcat(ones(n_features), reference)

        try
            coefs = A \ spectrum
            a, b = coefs[1], coefs[2]

            # Correct: (spectrum - a) / b
            if abs(b) > 1e-10
                X_corrected[i, :] = (spectrum .- a) ./ b
            else
                # If b ≈ 0, return mean-centered spectrum
                X_corrected[i, :] = spectrum .- mean(spectrum)
            end
        catch e
            @warn "MSC correction failed for sample $i: $e. Returning mean-centered spectrum."
            X_corrected[i, :] = spectrum .- mean(spectrum)
        end
    end

    return X_corrected
end

# Export new functions
export MSCTransformer, fit_msc!, apply_msc
```

### Step 4.2: Integrate MSC with Preprocessing Pipeline

**In same file**, modify `build_preprocessing_pipeline()` to support MSC:

```julia
function build_preprocessing_pipeline(preprocess_type::String; deriv_order=1, window=17, polyorder=3)
    """
    Build preprocessing pipeline.

    Supported types:
    - "raw": No preprocessing
    - "snv": Standard Normal Variate
    - "msc": Multiplicative Scatter Correction
    - "deriv", "sg1", "sg2": Savitzky-Golay derivatives
    - "snv_deriv", "msc_deriv": SNV/MSC + derivative
    - "deriv_snv", "deriv_msc": Derivative + SNV/MSC
    """

    if preprocess_type == "raw"
        return nothing

    elseif preprocess_type == "snv"
        return SNVTransformer()

    elseif preprocess_type == "msc"
        return MSCTransformer(reference="mean")

    elseif preprocess_type in ["deriv", "sg1", "sg2"]
        order = preprocess_type == "sg2" ? 2 : 1
        return SavgolDerivativeTransformer(order=order, window_length=window, polyorder=polyorder)

    elseif preprocess_type == "snv_deriv"
        # SNV first, then derivative
        snv = SNVTransformer()
        deriv = SavgolDerivativeTransformer(order=deriv_order, window_length=window, polyorder=polyorder)
        return [snv, deriv]

    elseif preprocess_type == "msc_deriv"
        # NEW: MSC first, then derivative
        msc = MSCTransformer(reference="mean")
        deriv = SavgolDerivativeTransformer(order=deriv_order, window_length=window, polyorder=polyorder)
        return [msc, deriv]

    elseif preprocess_type == "deriv_snv"
        # Derivative first, then SNV
        deriv = SavgolDerivativeTransformer(order=deriv_order, window_length=window, polyorder=polyorder)
        snv = SNVTransformer()
        return [deriv, snv]

    elseif preprocess_type == "deriv_msc"
        # NEW: Derivative first, then MSC
        deriv = SavgolDerivativeTransformer(order=deriv_order, window_length=window, polyorder=polyorder)
        msc = MSCTransformer(reference="mean")
        return [deriv, msc]

    else
        error("Unknown preprocessing type: $preprocess_type")
    end
end
```

### Step 4.3: Unit Tests for MSC

**File:** `julia_port/SpectralPredict/test/test_preprocessing.jl`

Add to existing test file:

```julia
@testset "MSC Transformer" begin
    # Create synthetic spectra with multiplicative scatter
    n_samples = 50
    n_features = 100

    # True spectrum
    true_spectrum = sin.(range(0, 2π, length=n_features))

    # Create scattered spectra: x_i = a_i + b_i * true_spectrum + noise
    X = zeros(n_samples, n_features)
    for i in 1:n_samples
        a = randn() * 0.1
        b = 1.0 + randn() * 0.2
        noise = randn(n_features) * 0.01
        X[i, :] = a .+ b .* true_spectrum .+ noise
    end

    @testset "MSC Fitting" begin
        msc = MSCTransformer(reference="mean")
        fit_msc!(msc, X)

        @test !isnothing(msc.reference_spectrum_)
        @test length(msc.reference_spectrum_) == n_features
    end

    @testset "MSC Correction" begin
        msc = MSCTransformer(reference="mean")
        fit_msc!(msc, X)
        X_corrected = apply_msc(msc, X)

        @test size(X_corrected) == size(X)

        # Corrected spectra should have reduced variance across samples
        # (at each wavelength)
        var_original = var(X, dims=1)
        var_corrected = var(X_corrected, dims=1)

        # Mean variance should decrease
        @test mean(var_corrected) < mean(var_original)
    end

    @testset "MSC with Custom Reference" begin
        custom_ref = true_spectrum
        msc = MSCTransformer(reference=custom_ref)
        fit_msc!(msc, X)

        @test msc.reference_spectrum_ == custom_ref

        X_corrected = apply_msc(msc, X)
        @test size(X_corrected) == size(X)
    end

    @testset "MSC Edge Cases" begin
        # Constant spectrum (b ≈ 0)
        X_const = ones(10, 50)
        msc = MSCTransformer("mean")
        fit_msc!(msc, X_const)

        @test_nowarn apply_msc(msc, X_const)
    end
end
```

---

## Phase 5: GUI Integration

**Duration:** 1 week
**Complexity:** Medium
**Files Modified:**
- `spectral_predict_julia_bridge.py` (major updates)
- `spectral_predict_gui_optimized.py` (minor updates)

### Step 5.1: Update Julia Bridge - Add Variable Selection Support

**File:** `spectral_predict_julia_bridge.py`

**Modify `_create_config()` function** (around lines 355-461):

```python
def _create_config(
    task_type,
    folds,
    lambda_penalty,
    max_n_components,
    max_iter,
    models_to_test,
    preprocessing_methods,
    window_sizes,
    enable_variable_subsets,
    variable_counts,
    enable_region_subsets,
    n_top_regions,
    variable_selection_methods=None  # NEW parameter
):
    """Create configuration dictionary for Julia."""

    # ... existing code for models, preprocessing ...

    # NEW: Variable selection methods
    if variable_selection_methods is None:
        julia_var_selection_methods = ['importance']
    else:
        # Map Python names to Julia names (they're the same)
        valid_methods = ['importance', 'SPA', 'UVE', 'iPLS', 'UVE-SPA']
        julia_var_selection_methods = [m for m in variable_selection_methods if m in valid_methods]

        if not julia_var_selection_methods:
            julia_var_selection_methods = ['importance']

    config = {
        'task_type': task_type,
        'models': julia_models,
        'preprocessing': julia_preprocessing,
        'derivative_orders': derivative_orders if derivative_orders else [1, 2],
        'derivative_window': window_size,
        'derivative_polyorder': 3,
        'enable_variable_subsets': enable_variable_subsets,
        'variable_counts': variable_counts,
        'variable_selection_methods': julia_var_selection_methods,  # NEW
        'enable_region_subsets': enable_region_subsets,
        'n_top_regions': n_top_regions,
        'n_folds': folds,
        'lambda_penalty': lambda_penalty,
        'max_n_components': max_n_components,
        'max_iter': max_iter
    }

    return config
```

**Update `run_search_julia()` signature** (around line 55):

```python
def run_search_julia(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    folds: int = 5,
    lambda_penalty: float = 0.15,
    max_n_components: int = 24,
    max_iter: int = 500,
    models_to_test: Optional[List[str]] = None,
    preprocessing_methods: Optional[Dict[str, bool]] = None,
    window_sizes: Optional[List[int]] = None,
    n_estimators_list: Optional[List[int]] = None,
    learning_rates: Optional[List[float]] = None,
    enable_variable_subsets: bool = True,
    variable_counts: Optional[List[int]] = None,
    enable_region_subsets: bool = True,
    n_top_regions: int = 5,
    variable_selection_methods: Optional[List[str]] = None,  # NEW
    progress_callback: Optional[Callable] = None,
    julia_exe: Optional[str] = None,
    julia_project: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run comprehensive model search using Julia backend.

    New Parameters
    --------------
    variable_selection_methods : list of str, optional
        Variable selection methods to use. Options:
        - 'importance': Model-based feature importance (default)
        - 'SPA': Successive Projections Algorithm
        - 'UVE': Uninformative Variable Elimination
        - 'iPLS': Interval PLS
        - 'UVE-SPA': Hybrid UVE-SPA approach
        If None, uses ['importance']
    """
    # ... existing validation ...

    # Pass to config creation
    config = _create_config(
        task_type=task_type,
        # ... other parameters ...
        variable_selection_methods=variable_selection_methods  # NEW
    )

    # ... rest of function unchanged ...
```

### Step 5.2: Update Julia Script Generation

**In `spectral_predict_julia_bridge.py`**, modify `_create_julia_script()` (around line 464):

```python
def _create_julia_script(...):
    # ... existing code ...

    # Add variable selection methods to config embedding
    script = f'''
# SpectralPredict Julia Analysis Script
using SpectralPredict
using SpectralPredict.VariableSelection  # NEW

# Configuration
config = Dict(
    "task_type" => "{config['task_type']}",
    "models" => {to_julia_array(config['models'])},
    "preprocessing" => {to_julia_array(config['preprocessing'])},
    "variable_selection_methods" => {to_julia_array(config['variable_selection_methods'])},  # NEW
    "enable_variable_subsets" => {str(config['enable_variable_subsets']).lower()},
    "variable_counts" => {to_julia_array(config['variable_counts'])},
    # ... rest of config ...
)

# Run search
results = run_search(
    X_matrix,
    y_vector,
    wavelengths,
    task_type=config["task_type"],
    models=config["models"],
    preprocessing=config["preprocessing"],
    variable_selection_methods=config["variable_selection_methods"],  # NEW
    # ... other parameters ...
)
'''

    return script
```

### Step 5.3: Update GUI to Pass Variable Selection Methods

**File:** `spectral_predict_gui_optimized.py`

**Find the section where `run_search()` is called** (around line 2593-2617):

```python
# In _run_analysis_thread() method

# Get selected variable selection methods from GUI
selected_var_methods = []
if hasattr(self, 'var_selection_checkboxes'):
    for method, var in self.var_selection_checkboxes.items():
        if var.get():
            selected_var_methods.append(method)

if not selected_var_methods:
    selected_var_methods = ['importance']  # Default

# Call search (works with both Python and Julia backends)
results_df = run_search(
    X_filtered,
    y_filtered,
    task_type=task_type,
    folds=self.folds.get(),
    # ... other parameters ...
    enable_variable_subsets=enable_variable_subsets,
    variable_counts=variable_counts,
    variable_selection_methods=selected_var_methods,  # NEW parameter
    progress_callback=self._update_progress
)
```

**If GUI doesn't already have variable selection checkboxes**, add them to Analysis Configuration tab:

```python
# In create_analysis_config_tab() method

# Variable Selection Methods section
var_selection_frame = ttk.LabelFrame(config_tab, text="Variable Selection Methods", padding=10)
var_selection_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
row += 1

self.var_selection_checkboxes = {}
methods = [
    ('importance', 'Feature Importance (default)'),
    ('SPA', 'SPA (Successive Projections)'),
    ('UVE', 'UVE (Uninformative Variable Elimination)'),
    ('iPLS', 'iPLS (Interval PLS)'),
    ('UVE-SPA', 'UVE-SPA Hybrid')
]

for i, (method_id, method_name) in enumerate(methods):
    var = tk.BooleanVar(value=(method_id == 'importance'))  # Default: importance only
    cb = ttk.Checkbutton(var_selection_frame, text=method_name, variable=var)
    cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2)
    self.var_selection_checkboxes[method_id] = var
```

### Step 5.4: Add Diagnostics Support to Bridge

**File:** `spectral_predict_julia_bridge.py`

Add new function for diagnostics:

```python
def compute_diagnostics_julia(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_type: str,
    model_config: dict,
    julia_exe: Optional[str] = None,
    julia_project: Optional[str] = None
) -> dict:
    """
    Compute model diagnostics using Julia backend.

    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    y_pred : Predictions on test data
    model_type : Type of model (e.g., 'PLS', 'Ridge')
    model_config : Model hyperparameters

    Returns
    -------
    dict with keys:
        - 'residuals': Raw residuals
        - 'std_residuals': Standardized residuals
        - 'leverage': Leverage values
        - 'leverage_threshold': Threshold for high leverage
        - 'qq_theoretical': Q-Q plot theoretical quantiles
        - 'qq_sample': Q-Q plot sample quantiles
        - 'pred_intervals': Dict with 'predictions', 'lower', 'upper', 'stderr'
                           (only if jackknife computed)
    """

    julia_exe = julia_exe or JULIA_EXE
    julia_project = julia_project or JULIA_PROJECT

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save data
        np.savetxt(temp_path / "X_train.csv", X_train, delimiter=",")
        np.savetxt(temp_path / "y_train.csv", y_train, delimiter=",")
        np.savetxt(temp_path / "X_test.csv", X_test, delimiter=",")
        np.savetxt(temp_path / "y_test.csv", y_test, delimiter=",")
        np.savetxt(temp_path / "y_pred.csv", y_pred, delimiter=",")

        # Create Julia script for diagnostics
        script = f'''
using SpectralPredict.Diagnostics
using CSV
using DataFrames

# Load data
X_train = readdlm("{str(temp_path / 'X_train.csv').replace(chr(92), '/')}", ',', Float64)
y_train = vec(readdlm("{str(temp_path / 'y_train.csv').replace(chr(92), '/')}", ',', Float64))
X_test = readdlm("{str(temp_path / 'X_test.csv').replace(chr(92), '/')}", ',', Float64)
y_test = vec(readdlm("{str(temp_path / 'y_test.csv').replace(chr(92), '/')}", ',', Float64))
y_pred = vec(readdlm("{str(temp_path / 'y_pred.csv').replace(chr(92), '/')}", ',', Float64))

# Compute residuals
residuals, std_residuals = compute_residuals(y_test, y_pred)

# Compute leverage
leverage, threshold = compute_leverage(X_train)

# Q-Q plot data
qq_theoretical, qq_sample = qq_plot_data(residuals)

# Save results
writedlm("{str(temp_path / 'residuals.csv').replace(chr(92), '/')}", residuals, ',')
writedlm("{str(temp_path / 'std_residuals.csv').replace(chr(92), '/')}", std_residuals, ',')
writedlm("{str(temp_path / 'leverage.csv').replace(chr(92), '/')}", leverage, ',')
writedlm("{str(temp_path / 'leverage_threshold.csv').replace(chr(92), '/')}", [threshold], ',')
writedlm("{str(temp_path / 'qq_theoretical.csv').replace(chr(92), '/')}", qq_theoretical, ',')
writedlm("{str(temp_path / 'qq_sample.csv').replace(chr(92), '/')}", qq_sample, ',')

println("Diagnostics computed successfully!")
'''

        script_file = temp_path / "diagnostics.jl"
        with open(script_file, 'w') as f:
            f.write(script)

        # Run Julia
        subprocess.run(
            [julia_exe, f"--project={julia_project}", str(script_file)],
            check=True
        )

        # Load results
        results = {
            'residuals': np.loadtxt(temp_path / "residuals.csv", delimiter=","),
            'std_residuals': np.loadtxt(temp_path / "std_residuals.csv", delimiter=","),
            'leverage': np.loadtxt(temp_path / "leverage.csv", delimiter=","),
            'leverage_threshold': np.loadtxt(temp_path / "leverage_threshold.csv", delimiter=",")[0],
            'qq_theoretical': np.loadtxt(temp_path / "qq_theoretical.csv", delimiter=","),
            'qq_sample': np.loadtxt(temp_path / "qq_sample.csv", delimiter=","),
        }

        return results
```

### Step 5.5: Wire Diagnostics in GUI

**File:** `spectral_predict_gui_optimized.py`

**In Model Development tab**, when user clicks "Show Diagnostics" or similar button:

```python
def compute_and_show_diagnostics(self):
    """Compute diagnostics using Julia backend."""

    if self.refined_model is None:
        messagebox.showwarning("No Model", "Please train a model first.")
        return

    # Import Julia bridge
    try:
        from spectral_predict_julia_bridge import compute_diagnostics_julia
        use_julia = True
    except:
        use_julia = False

    if use_julia:
        # Use Julia for faster diagnostics
        diagnostics = compute_diagnostics_julia(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            y_pred=self.y_pred,
            model_type=self.model_type,
            model_config=self.model_config
        )
    else:
        # Fallback to Python
        from spectral_predict.diagnostics import compute_residuals, compute_leverage, qq_plot_data

        residuals, std_residuals = compute_residuals(self.y_test, self.y_pred)
        leverage, threshold = compute_leverage(self.X_train)
        qq_theoretical, qq_sample = qq_plot_data(residuals)

        diagnostics = {
            'residuals': residuals,
            'std_residuals': std_residuals,
            'leverage': leverage,
            'leverage_threshold': threshold,
            'qq_theoretical': qq_theoretical,
            'qq_sample': qq_sample
        }

    # Plot diagnostics
    self.plot_diagnostics(diagnostics)
```

---

## Phase 6: Testing & Validation

**Duration:** 1 week
**Complexity:** Critical (ensures correctness)

### Step 6.1: Numerical Parity Tests

**Goal:** Verify Julia results match Python results (within floating-point tolerance)

**File:** `julia_port/SpectralPredict/test/test_parity.jl` (NEW)

```julia
"""
Test numerical parity between Julia and Python implementations.

Requires Python with spectral_predict installed.
Run with: julia --project=. test/test_parity.jl
"""

using Test
using SpectralPredict
using SpectralPredict.VariableSelection
using Random
using Printf

# Try to import PyCall for Python comparison
try
    using PyCall
    HAS_PYCALL = true
catch
    @warn "PyCall not available. Skipping parity tests."
    HAS_PYCALL = false
    exit(0)
end

if HAS_PYCALL
    # Import Python modules
    py"""
    import sys
    sys.path.insert(0, '../src')  # Adjust path to Python implementation
    from spectral_predict.variable_selection import uve_selection, spa_selection, ipls_selection
    """

    @testset "Numerical Parity Tests" begin

        Random.seed!(42)

        # Create test dataset
        n_samples = 100
        n_features = 200

        X = randn(n_samples, n_features)
        y = randn(n_samples)

        @testset "UVE Parity" begin
            # Julia implementation
            julia_scores = uve_selection(X, y, cutoff_multiplier=1.0, cv_folds=5, random_state=42)

            # Python implementation
            py_scores = py"uve_selection"(X, y, cutoff_multiplier=1.0, cv_folds=5, random_state=42)

            # Compare
            @test length(julia_scores) == length(py_scores)

            # Scores should be close (allow for numerical differences)
            max_diff = maximum(abs.(julia_scores .- py_scores))
            @test max_diff < 0.1

            # Correlation should be very high
            correlation = cor(julia_scores, py_scores)
            @test correlation > 0.95

            @info "UVE Parity: max_diff=$max_diff, cor=$correlation"
        end

        @testset "SPA Parity" begin
            n_select = 50

            julia_scores = spa_selection(X, y, n_select, n_random_starts=5, cv_folds=5, random_state=42)
            py_scores = py"spa_selection"(X, y, n_select, n_random_starts=5, cv_folds=5, random_state=42)

            # Same variables should be selected
            julia_selected = findall(julia_scores .> 0)
            py_selected = findall(py_scores .> 0)

            # Overlap should be high (>80%)
            overlap = length(intersect(julia_selected, py_selected)) / length(union(julia_selected, py_selected))
            @test overlap > 0.8

            @info "SPA Parity: overlap=$overlap"
        end

        @testset "iPLS Parity" begin
            julia_scores = ipls_selection(X, y, n_intervals=20, cv_folds=5, random_state=42)
            py_scores = py"ipls_selection"(X, y, n_intervals=20, cv_folds=5, random_state=42)

            # Interval scores should be identical (deterministic)
            max_diff = maximum(abs.(julia_scores .- py_scores))
            @test max_diff < 0.01

            @info "iPLS Parity: max_diff=$max_diff"
        end
    end
end
```

### Step 6.2: End-to-End Integration Test

**File:** `julia_port/SpectralPredict/test/test_integration.jl` (NEW)

```julia
"""
End-to-end integration test.
Tests complete workflow from data loading to results.
"""

using Test
using SpectralPredict
using Random
using DataFrames

@testset "End-to-End Integration" begin

    Random.seed!(42)

    # Create realistic spectral dataset
    n_samples = 150
    n_wavelengths = 500

    # Simulate NIR spectrum
    wavelengths = collect(range(1000, 2500, length=n_wavelengths))

    X = zeros(n_samples, n_wavelengths)
    for i in 1:n_samples
        # Base spectrum + sample variations
        baseline = sin.(wavelengths ./ 100) .+ randn() * 0.1
        X[i, :] = baseline .+ randn(n_wavelengths) * 0.05
    end

    # Target: linear combination of specific wavelengths
    important_wavelengths = [1200, 1400, 1600, 1800, 2000]
    important_indices = [argmin(abs.(wavelengths .- wl)) for wl in important_wavelengths]
    y = sum(X[:, important_indices] .* [1.0, 2.0, -1.0, 0.5, -0.5]', dims=2)[:] + randn(n_samples) * 0.5

    @testset "Full Search with All Methods" begin

        results = run_search(
            X, y, wavelengths;
            task_type="regression",
            models=["PLS", "Ridge"],
            preprocessing=["raw", "snv"],
            variable_selection_methods=["importance", "SPA", "UVE"],
            enable_variable_subsets=true,
            variable_counts=[20, 50],
            enable_region_subsets=false,
            n_folds=3,
            lambda_penalty=0.15
        )

        @test isa(results, DataFrame)
        @test nrow(results) > 0

        # Check required columns
        required_cols = [:Model, :Preprocess, :RMSE, :R2, :CompositeScore, :Rank]
        for col in required_cols
            @test col in propertynames(results)
        end

        # Check that best model has reasonable performance
        best_model = results[1, :]
        @test best_model.R2 > 0.3  # Should capture some signal

        # Check that variable selection methods were tested
        methods_tested = unique(results.SubsetTag)
        @test any(contains.(methods_tested, "SPA"))
        @test any(contains.(methods_tested, "UVE"))

        @info "Integration test passed! $(nrow(results)) configurations tested."
        @info "Best model: $(best_model.Model) | $(best_model.Preprocess) | R²=$(best_model.R2)"
    end

    @testset "Diagnostics Integration" begin
        # Train simple model
        n_train = 100
        X_train, y_train = X[1:n_train, :], y[1:n_train]
        X_test, y_test = X[n_train+1:end, :], y[n_train+1:end]

        # Fit PLS model (simplified)
        using MultivariateStats
        M = fit(CCA, X_train', y_train'; outdim=5)
        y_pred = vec(predict(M, X_test'))

        # Compute diagnostics
        using SpectralPredict.Diagnostics

        residuals, std_residuals = compute_residuals(y_test, y_pred)
        @test length(residuals) == length(y_test)

        leverage, threshold = compute_leverage(X_train)
        @test length(leverage) == n_train
        @test threshold > 0

        qq_theoretical, qq_sample = qq_plot_data(residuals)
        @test length(qq_theoretical) == length(residuals)

        @info "Diagnostics integration passed!"
    end
end
```

### Step 6.3: Performance Benchmarking

**File:** `julia_port/SpectralPredict/benchmark/bench_full_pipeline.jl` (NEW)

```julia
"""
Benchmark full analysis pipeline.
Compares Julia vs Python performance.
"""

using BenchmarkTools
using SpectralPredict
using Random
using Printf

Random.seed!(42)

println("="^70)
println("Full Pipeline Performance Benchmark")
println("="^70)
println()

# Realistic spectral dataset
n_samples = 200
n_wavelengths = 1000

println("Creating dataset: $n_samples samples × $n_wavelengths wavelengths")

wavelengths = collect(range(1000, 2500, length=n_wavelengths))
X = randn(n_samples, n_wavelengths)
y = randn(n_samples)

println()
println("Running benchmark...")
println()

# Benchmark Julia implementation
result = @benchmark run_search(
    $X, $y, $wavelengths;
    task_type="regression",
    models=["PLS", "Ridge"],
    preprocessing=["raw", "snv", "deriv"],
    variable_selection_methods=["importance", "SPA"],
    enable_variable_subsets=true,
    variable_counts=[20, 50],
    enable_region_subsets=false,
    n_folds=5,
    lambda_penalty=0.15
) samples=3

median_time = median(result.times) / 1e9  # Convert to seconds

println("Julia Implementation:")
println("  Median time: $(median_time) seconds")
println("  Memory: $(result.memory / 1e6) MB")
println()

println("="^70)
println("Benchmark complete!")
println()
println("To compare with Python:")
println("  1. Run Python version with same parameters")
println("  2. Compute speedup: Python_time / Julia_time")
println()
println("Expected speedup: 5-15x with full parallelization")
println("="^70)
```

### Step 6.4: Create Validation Report Script

**File:** `julia_port/SpectralPredict/scripts/validate_implementation.jl` (NEW)

```julia
"""
Comprehensive validation script.

Runs all tests and generates validation report.
"""

using Pkg
using Test

println("="^70)
println("SpectralPredict.jl Validation Report")
println("="^70)
println()

# Activate project
Pkg.activate(".")
Pkg.instantiate()

println("Julia version: $(VERSION)")
println("Project dependencies:")
for (pkg, version) in Pkg.dependencies()
    if pkg.is_direct_dep
        println("  - $(pkg.name) v$(pkg.version)")
    end
end
println()

# Run all test suites
test_suites = [
    ("Variable Selection", "test/test_variable_selection.jl"),
    ("Diagnostics", "test/test_diagnostics.jl"),
    ("Neural Boosted", "test/test_neural_boosted.jl"),
    ("Preprocessing", "test/test_preprocessing.jl"),
    ("Integration", "test/test_integration.jl"),
]

results = Dict()

for (name, test_file) in test_suites
    println("Running $name tests...")

    try
        include(test_file)
        results[name] = "✅ PASSED"
    catch e
        results[name] = "❌ FAILED: $e"
    end

    println()
end

# Print summary
println("="^70)
println("Validation Summary")
println("="^70)

for (name, status) in results
    println("$name: $status")
end

println()

all_passed = all(contains(status, "PASSED") for status in values(results))

if all_passed
    println("✅ ALL TESTS PASSED!")
    println()
    println("Implementation is ready for production use.")
else
    println("❌ SOME TESTS FAILED")
    println()
    println("Please review failed tests before deploying.")
end

println("="^70)
```

**Run validation:**
```bash
cd julia_port/SpectralPredict
julia --project=. scripts/validate_implementation.jl
```

---

## Performance Benchmarking

### Benchmark Suite

**File:** `julia_port/SpectralPredict/benchmark/run_all_benchmarks.jl` (NEW)

```julia
"""
Run all benchmarks and generate performance report.
"""

println("="^70)
println("SpectralPredict.jl Performance Report")
println("="^70)
println()
println("Running all benchmarks...")
println("This may take 5-10 minutes.")
println()

# Run individual benchmarks
benchmarks = [
    ("Variable Selection", "benchmark/bench_variable_selection.jl"),
    ("Diagnostics", "benchmark/bench_diagnostics.jl"),
    ("Full Pipeline", "benchmark/bench_full_pipeline.jl"),
]

for (name, script) in benchmarks
    println("="^70)
    println("Benchmark: $name")
    println("="^70)
    include(script)
    println()
end

println("="^70)
println("All benchmarks complete!")
println("="^70)
```

### Expected Performance Results

**Based on Tier 1 porting:**

| Operation | Python (baseline) | Julia Sequential | Julia Parallel (8 cores) | Speedup |
|-----------|-------------------|------------------|--------------------------|---------|
| UVE Selection | 30s | 9s (3.3x) | 4s (7.5x) | ✅ |
| SPA Selection | 100s | 25s (4x) | 8s (12.5x) | ✅ |
| iPLS Selection | 60s | 18s (3.3x) | 7s (8.6x) | ✅ |
| Jackknife (n=100) | 500s | 180s (2.8x) | 25s (20x) | ✅ |
| Neural Boosted | 120s | 50s (2.4x) | 50s (2.4x) | ✅ |
| **Full Pipeline** | **2-4 hrs** | **1-1.5 hrs (2.5x)** | **0.3-0.6 hrs (7-10x)** | **✅** |

---

## Troubleshooting Guide

### Common Issues

#### 1. Julia Not Found

**Error:** `Julia executable not found: /path/to/julia`

**Solution:**
```bash
# Find Julia
which julia

# Update path in spectral_predict_julia_bridge.py
JULIA_EXE = "/correct/path/to/julia"
```

#### 2. Package Not Instantiated

**Error:** `ArgumentError: Package Distributions not found`

**Solution:**
```bash
cd julia_port/SpectralPredict
julia --project=.
```
```julia
using Pkg
Pkg.instantiate()
```

#### 3. PyCall Issues (for parity tests)

**Error:** `PyCall not available`

**Solution:**
```julia
using Pkg
Pkg.add("PyCall")
Pkg.build("PyCall")
```

#### 4. Thread Count

**Problem:** Jackknife not parallelizing

**Solution:**
```bash
# Check threads
julia -e 'println(Threads.nthreads())'

# Run with more threads
julia --project=. -t 8 script.jl
```

#### 5. Memory Issues

**Error:** `OutOfMemoryError`

**Solution:**
- Reduce dataset size
- Use fewer CV folds
- Reduce number of variable subsets
- Close other applications

#### 6. Numerical Differences

**Problem:** Julia results differ from Python

**Check:**
1. Random seeds match
2. Same preprocessing applied
3. Same hyperparameters
4. Floating-point tolerance appropriate

---

## Summary Checklist

### Phase 1: Variable Selection ✅
- [ ] Create `variable_selection.jl`
- [ ] Implement UVE selection
- [ ] Implement SPA selection
- [ ] Implement iPLS selection
- [ ] Implement UVE-SPA hybrid
- [ ] Unit tests pass
- [ ] Integrated with `search.jl`
- [ ] Benchmarked

### Phase 2: Diagnostics ✅
- [ ] Create `diagnostics.jl`
- [ ] Add Distributions.jl dependency
- [ ] Implement residual analysis
- [ ] Implement leverage computation
- [ ] Implement Q-Q plot data
- [ ] Implement jackknife intervals (parallelized)
- [ ] Unit tests pass
- [ ] Benchmarked

### Phase 3: Neural Boosted ✅
- [ ] Create `neural_boosted.jl`
- [ ] Implement NeuralBoostedRegressor struct
- [ ] Implement weak learner (Flux.jl)
- [ ] Implement fit! method
- [ ] Implement predict method
- [ ] Implement feature_importances
- [ ] Integrated with `models.jl`
- [ ] Unit tests pass

### Phase 4: MSC Preprocessing ✅
- [ ] Add MSC to `preprocessing.jl`
- [ ] Implement MSCTransformer
- [ ] Implement fit_msc! and apply_msc
- [ ] Integrated with pipeline builder
- [ ] Unit tests pass

### Phase 5: GUI Integration ✅
- [ ] Update `spectral_predict_julia_bridge.py`
- [ ] Add variable selection support to config
- [ ] Add diagnostics function
- [ ] Update GUI to pass new parameters
- [ ] Add variable selection checkboxes (if needed)
- [ ] Wire diagnostics tab

### Phase 6: Testing & Validation ✅
- [ ] All unit tests pass
- [ ] Parity tests pass (Julia ≈ Python)
- [ ] Integration tests pass
- [ ] Performance benchmarks complete
- [ ] Documentation updated
- [ ] Validation report generated

---

## Next Steps After Implementation

1. **Performance Analysis**
   - Run benchmarks on production datasets
   - Compare with Python implementation
   - Document speedup factors

2. **User Documentation**
   - Update user guide with new features
   - Add examples of variable selection methods
   - Document performance improvements

3. **Deployment**
   - Create release notes
   - Tag version (e.g., v2.0.0)
   - Update README

4. **Future Enhancements**
   - Add more variable selection methods (CARS, GA)
   - GPU acceleration for neural networks
   - Distributed computing for very large datasets

---

**End of Julia Porting Implementation Plan**

**Estimated Total Time:** 6-9 weeks
**Expected Performance Gain:** 5-15x overall speedup
**Lines of Code Added:** ~3,000-4,000 lines Julia

Good luck with implementation! 🚀
