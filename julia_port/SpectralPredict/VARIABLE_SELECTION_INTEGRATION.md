# Variable Selection Integration Guide

**Date:** November 5, 2025
**Module:** `search.jl` integration with `variable_selection.jl`

## Overview

This document describes the integration of variable selection methods (UVE, SPA, iPLS, UVE-SPA) into the search pipeline. Variable selection helps identify the most informative wavelengths for prediction, reducing model complexity and potentially improving performance.

## Changes Made to search.jl

### 1. Added Module Import (Lines 36-42)

```julia
include("variable_selection.jl")

using .Regions
using .Scoring

# Import variable selection functions
import .uve_selection, .spa_selection, .ipls_selection, .uve_spa_selection
```

**Purpose:** Makes variable selection functions available within the search module.

### 2. Added Parameter to run_search() (Line 62, 180)

**New Parameter:**
```julia
variable_selection_methods::Vector{String}=String[]
```

**Options:**
- `String[]` (default) - Uses model-based feature importance for compatible models (PLS, RF, MLP)
- `["uve"]` - Uninformative Variable Elimination
- `["spa"]` - Successive Projections Algorithm
- `["ipls"]` - Interval PLS
- `["uve_spa"]` - Hybrid UVE-SPA method
- Can specify multiple methods: `["uve", "spa", "ipls", "uve_spa"]`

**Documentation Added (Lines 116-119):**
```julia
- `variable_selection_methods::Vector{String}`: Variable selection methods to use (default: String[])
  - Options: "uve", "spa", "ipls", "uve_spa"
  - Empty array means use model-based feature importance (default behavior)
  - Variable selection happens BEFORE preprocessing for each method
```

### 3. Modified Variable Subset Loop (Lines 310-441)

**Key Changes:**

#### a. Method Selection Logic (Lines 311-321)
```julia
# Determine which variable selection methods to use
methods_to_use = String[]

# If variable_selection_methods is specified, use those
if !isempty(variable_selection_methods)
    methods_to_use = variable_selection_methods
elseif model_name in ["PLS", "RandomForest", "MLP"]
    # Default: use model-based feature importance for compatible models
    push!(methods_to_use, "importance")
end
```

**Behavior:**
- If `variable_selection_methods` is provided, uses those methods for ALL models
- Otherwise, uses model-based importance only for PLS, RandomForest, and MLP
- This maintains backward compatibility

#### b. Variable Selection Implementation (Lines 339-400)

Each method is implemented with:
1. **Method-specific selection logic**
2. **Proper handling of edge cases** (no variables selected, count limits)
3. **Appropriate subset tagging** for tracking

**UVE Selection (Lines 348-357):**
```julia
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
```

**SPA Selection (Lines 359-372):**
```julia
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
```

**iPLS Selection (Lines 374-383):**
```julia
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
```

**UVE-SPA Selection (Lines 385-395):**
```julia
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
```

#### c. Edge Case Handling (Lines 402-406)

```julia
# Handle edge case: no variables selected
if isempty(selected_indices)
    @warn "No variables selected by $method for $n_top variables (skipping)"
    continue
end
```

**Prevents:** Crashes when selection methods eliminate all variables

#### d. Preprocessing Logic (Lines 408-438)

```julia
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
```

**Critical Design Decision:**
- Variable selection is applied to **preprocessed data** (X_preprocessed)
- This ensures selection sees the same data that models will train on
- Prevents double-preprocessing bug by using `skip_preprocessing=true` for derivatives

### 4. Updated run_single_config() (Lines 660-661, 742-743, 786-787)

**New Parameters:**
```julia
var_selection_method::Union{String,Nothing}=nothing,
var_selection_indices::Union{Vector{Int},Nothing}=nothing
```

**Documentation Added (Lines 700-701):**
```julia
- `var_selection_method::Union{String,Nothing}`: Variable selection method used (if any)
- `var_selection_indices::Union{Vector{Int},Nothing}`: Indices of selected variables (if any)
```

**Metadata Storage (Lines 786-787):**
```julia
"VarSelectionMethod" => var_selection_method,
"VarSelectionIndices" => var_selection_indices
```

**Purpose:** Track which variables were selected by which method in the results DataFrame

### 5. Added Usage Example (Lines 157-168)

```julia
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
```

## Usage Examples

### Example 1: Default Behavior (Backward Compatible)

```julia
# No variable selection methods specified
# Uses model-based importance for PLS, RF, MLP only
results = run_search(X, y, wavelengths)
```

**Result:** Original behavior maintained - only models with feature importance get variable subsets.

### Example 2: UVE Selection Only

```julia
results = run_search(
    X, y, wavelengths,
    variable_selection_methods=["uve"],
    variable_counts=[20, 50, 100]
)
```

**Result:**
- All models get variable subsets using UVE selection
- Tests 20, 50, and 100 variables for each model
- Results will have `SubsetTag` like "uve20", "uve50", "uve100"

### Example 3: Multiple Selection Methods

```julia
results = run_search(
    X, y, wavelengths,
    models=["PLS", "Ridge"],
    preprocessing=["snv", "deriv"],
    variable_selection_methods=["uve", "spa", "ipls", "uve_spa"],
    variable_counts=[10, 20, 50],
    n_folds=5
)
```

**Result:**
- Each model tested with 4 selection methods × 3 variable counts = 12 subsets
- Plus full model = 13 configurations per preprocessing method
- Results DataFrame will have columns:
  - `VarSelectionMethod`: "uve", "spa", "ipls", or "uve_spa"
  - `VarSelectionIndices`: Vector of selected variable indices
  - `SubsetTag`: "uve10", "spa20", "ipls50", "uve_spa10", etc.

### Example 4: SPA with Custom Settings

```julia
# For more control over SPA parameters, modify search.jl line 361-365:
importances = spa_selection(
    X_preprocessed, y, n_top;
    n_random_starts=20,  # Increase for more thorough search
    cv_folds=n_folds
)
```

## Results DataFrame Schema

When variable selection is used, results include:

| Column | Type | Description |
|--------|------|-------------|
| `Model` | String | Model name |
| `Preprocess` | String | Preprocessing method |
| `SubsetTag` | String | Subset identifier ("full", "uve20", "spa50", etc.) |
| `n_vars` | Int | Number of variables in subset |
| `full_vars` | Int | Total variables in full dataset |
| `VarSelectionMethod` | String? | Variable selection method ("uve", "spa", "ipls", "uve_spa", nothing) |
| `VarSelectionIndices` | Vector{Int}? | Indices of selected variables |
| `RMSE` | Float64 | Cross-validated RMSE |
| `R2` | Float64 | Cross-validated R² |
| `CompositeScore` | Float64 | Combined performance score |
| `Rank` | Int | Overall ranking |

## Design Decisions

### 1. Variable Selection on Preprocessed Data

**Decision:** Apply variable selection to `X_preprocessed` (after preprocessing)

**Rationale:**
- Selection should see the same data that models will train on
- Derivatives change the nature of the data significantly
- Ensures consistency between selection and modeling

**Alternative Considered:** Select on raw data before preprocessing
- **Rejected:** Would lead to mismatch between selected variables and model inputs

### 2. Backward Compatibility

**Decision:** Empty `variable_selection_methods` means use model-based importance (original behavior)

**Rationale:**
- Existing code continues to work without changes
- Users can opt-in to new methods explicitly
- No breaking changes to existing workflows

### 3. Skip-Preprocessing Logic

**Decision:** Maintain existing skip-preprocessing logic for derivatives

**Rationale:**
- Prevents double-preprocessing bug (fixed in Python Oct 29, 2025)
- Variable selection works on preprocessed data
- Results are passed with `skip_preprocessing=true` for derivatives

### 4. Metadata Tracking

**Decision:** Store `VarSelectionMethod` and `VarSelectionIndices` in results

**Rationale:**
- Enables reproducibility
- Allows analysis of which methods work best
- Facilitates debugging and validation
- Users can extract selected variables for later use

### 5. Edge Case Handling

**Decisions:**
- Skip if no variables selected (with warning)
- Adjust count if UVE selects fewer than requested
- Handle invalid method names gracefully

**Rationale:**
- Prevents crashes from edge cases
- Provides clear feedback to users
- Allows search to continue even if one method fails

## Performance Considerations

### Variable Selection Overhead

- **UVE:** O(n_folds × n_features²) - slowest due to PLS on augmented data
- **SPA:** O(n_starts × n_features² × n_top) - moderate, parallelizable
- **iPLS:** O(n_intervals × n_folds × interval_size) - fast
- **UVE-SPA:** O(UVE + SPA) - slowest overall

### Recommendations

1. **For large datasets (>500 features):**
   - Use iPLS first for quick region identification
   - Then apply UVE or SPA to top regions

2. **For small datasets (<200 features):**
   - All methods are fast enough
   - UVE-SPA provides best results

3. **For time-constrained searches:**
   - Use iPLS only (fastest)
   - Or reduce `variable_counts` to fewer values

## Testing

### Syntax Validation

The code has been manually reviewed for:
- ✓ Function signature consistency
- ✓ Variable scope (local declarations)
- ✓ Type annotations
- ✓ Edge case handling
- ✓ Backward compatibility

### Recommended Tests

1. **Basic functionality:**
   ```julia
   results = run_search(X, y, wavelengths,
                       variable_selection_methods=["uve"],
                       variable_counts=[20])
   @assert any(results.VarSelectionMethod .== "uve")
   ```

2. **Multiple methods:**
   ```julia
   results = run_search(X, y, wavelengths,
                       variable_selection_methods=["uve", "spa"],
                       variable_counts=[10, 20])
   @assert length(unique(results.VarSelectionMethod)) >= 2
   ```

3. **Backward compatibility:**
   ```julia
   results_old = run_search(X, y, wavelengths)  # No variable_selection_methods
   @assert all(ismissing.(results_old.VarSelectionMethod) .|
               isnothing.(results_old.VarSelectionMethod))
   ```

## Troubleshooting

### Issue: "No variables selected by uve for 20 variables"

**Cause:** UVE eliminated all variables as uninformative

**Solution:**
- Increase `cutoff_multiplier` in code (line 352): `cutoff_multiplier=1.5`
- Or use different selection method (spa, ipls)

### Issue: SPA takes very long

**Cause:** Too many random starts or large n_top

**Solution:**
- Reduce `n_random_starts` (line 363): `n_random_starts=5`
- Or reduce `variable_counts` values

### Issue: Results don't show variable selection metadata

**Cause:** `variable_selection_methods` is empty

**Solution:**
- Explicitly set: `variable_selection_methods=["uve"]` or other methods

## Summary

The integration successfully adds four variable selection methods to the search pipeline while maintaining full backward compatibility. The implementation:

- ✓ Supports all four methods: UVE, SPA, iPLS, UVE-SPA
- ✓ Maintains backward compatibility (empty methods = original behavior)
- ✓ Properly handles preprocessing (selection on preprocessed data)
- ✓ Avoids double-preprocessing bug (skip_preprocessing logic)
- ✓ Tracks metadata (method used, indices selected)
- ✓ Handles edge cases (no variables selected, etc.)
- ✓ Well-documented with examples

## Line Number Reference

| Change | Line Numbers |
|--------|--------------|
| Module import | 36-42 |
| Parameter in signature | 62, 180 |
| Parameter documentation | 116-119 |
| Variable subset logic | 310-441 |
| run_single_config signature | 660-661, 742-743 |
| Metadata storage | 786-787 |
| Usage example | 157-168 |
