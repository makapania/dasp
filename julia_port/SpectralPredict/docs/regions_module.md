# Regions Module

The Regions module identifies important spectral regions based on correlation with the target variable. It's a key component of the hyperparameter search strategy, enabling efficient variable selection by focusing on wavelength ranges with high predictive power.

## Overview

Spectral data often contains hundreds of wavelengths, but not all are equally informative. The Regions module:

1. **Divides the spectrum** into overlapping windows (e.g., 50nm regions with 25nm overlap)
2. **Computes correlations** between each wavelength and the target variable
3. **Ranks regions** by their mean absolute correlation
4. **Creates subsets** combining individual and grouped top regions for model testing

## Core Functions

### `compute_region_correlations`

Divides the spectrum into overlapping regions and computes correlation statistics.

```julia
regions = compute_region_correlations(
    X::Matrix{Float64},           # Spectral data (n_samples × n_wavelengths)
    y::Vector{Float64},           # Target values
    wavelengths::Vector{Float64}; # Wavelength values
    region_size::Float64=50.0,    # Region size in nm
    overlap::Float64=25.0         # Overlap in nm
)
```

**Returns:** Vector of dictionaries with keys:
- `"start"`: Start wavelength (nm)
- `"end"`: End wavelength (nm)
- `"indices"`: Feature indices in region (1-indexed)
- `"mean_corr"`: Mean absolute correlation with target
- `"max_corr"`: Maximum absolute correlation
- `"n_features"`: Number of features in region

**Example:**
```julia
# Basic usage
regions = compute_region_correlations(X, y, wavelengths)

# Custom parameters - larger regions, no overlap
regions = compute_region_correlations(X, y, wavelengths,
                                     region_size=100.0, overlap=0.0)

# Find top region
top_region = maximum(regions, by=r -> r["mean_corr"])
println("Best region: ", top_region["start"], "-", top_region["end"], " nm")
```

### `create_region_subsets`

Creates strategic combinations of top regions for model testing.

```julia
subsets = create_region_subsets(
    X::Matrix{Float64},
    y::Vector{Float64},
    wavelengths::Vector{Float64};
    n_top_regions::Int=5  # Number of top regions to use
)
```

**Returns:** Vector of dictionaries with keys:
- `"indices"`: Variable indices for subset (1-indexed)
- `"tag"`: Descriptive tag (e.g., "region_450-500nm", "top5regions")
- `"description"`: Human-readable description

**Strategy:**

The function creates two types of subsets:

1. **Individual regions** (test separately):
   - n_top_regions ≤ 5: Test top 3 individual regions
   - n_top_regions ≤ 10: Test top 5 individual regions
   - n_top_regions ≤ 15: Test top 7 individual regions
   - n_top_regions > 15: Test top 10 individual regions

2. **Combined regions** (test together):
   - Top 2 regions
   - Top 5 regions
   - Top 10 regions
   - Top 15 regions
   - Top 20 regions

**Example:**
```julia
# Create subsets for testing
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=10)

# Use in model training
for subset in subsets
    X_subset = X[:, subset["indices"]]
    # Train model on X_subset...
    println("Testing: ", subset["tag"])
end
```

### `combine_region_indices`

Combines indices from multiple regions into a sorted unique list.

```julia
combined = combine_region_indices(regions::Vector{Dict{String, Any}})
```

**Example:**
```julia
# Get top 3 regions
top_regions = sort(regions, by=r -> r["mean_corr"], rev=true)[1:3]

# Combine their indices
indices = combine_region_indices(top_regions)

# Use combined indices
X_combined = X[:, indices]
```

## Typical Workflow

### 1. Training Phase - Identify Regions

```julia
using Statistics
include("src/regions.jl")
using .Regions

# Load training data (preprocessed!)
X_train, y_train, wavelengths = load_training_data()

# Compute region correlations
regions = compute_region_correlations(X_train, y_train, wavelengths)

# Create subsets for hyperparameter search
subsets = create_region_subsets(X_train, y_train, wavelengths, n_top_regions=10)

println("Created ", length(subsets), " subsets for testing")
```

### 2. Hyperparameter Search

```julia
# Test each subset with different model configurations
results = []

for subset in subsets
    for preprocessing in ["None", "SNV", "Derivative1", "Derivative2"]
        for n_lvs in [5, 10, 15, 20]
            # Get subset data
            X_subset = X_train[:, subset["indices"]]

            # Apply preprocessing
            X_processed = preprocess(X_subset, preprocessing)

            # Train and evaluate model
            score = train_and_evaluate(X_processed, y_train, n_lvs)

            push!(results, (
                subset=subset["tag"],
                preprocessing=preprocessing,
                n_lvs=n_lvs,
                score=score
            ))
        end
    end
end

# Find best configuration
best = minimum(results, by=r -> r.score)
```

### 3. Analysis - Inspect Top Regions

```julia
# Sort regions by correlation
sorted_regions = sort(regions, by=r -> r["mean_corr"], rev=true)

# Display top 5 regions
println("Top 5 Spectral Regions:")
println("-"^70)
for (i, region) in enumerate(sorted_regions[1:5])
    println("$i. $(Int(region["start"]))-$(Int(region["end"])) nm: ",
            "r=$(round(region["mean_corr"], digits=3)), ",
            "n=$(region["n_features"]) features")
end
```

## Important Notes

### Preprocessing Order

**CRITICAL:** Region analysis must be performed on **preprocessed** data, not raw spectra.

```julia
# CORRECT: Preprocess first, then compute regions
X_preprocessed = apply_snv(X_raw)
regions = compute_region_correlations(X_preprocessed, y, wavelengths)

# WRONG: Computing regions on raw data
regions = compute_region_correlations(X_raw, y, wavelengths)  # ❌
```

This is because:
- SNV/derivatives change correlation patterns
- Different preprocessing methods may highlight different regions
- Regions should match the data used for model training

### Indexing

Julia uses 1-based indexing. All returned indices are 1-indexed:

```julia
regions = compute_region_correlations(X, y, wavelengths)
indices = regions[1]["indices"]  # Indices are 1-based
X_subset = X[:, indices]  # Directly usable in Julia
```

### Memory Considerations

For large datasets, region analysis is efficient:

```julia
# Example: 1000 samples × 2000 wavelengths
X = randn(1000, 2000)  # ~16 MB
y = randn(1000)
wavelengths = collect(400.0:0.2:799.8)

# Region computation is fast (< 1 second typically)
@time regions = compute_region_correlations(X, y, wavelengths)

# Subsets don't copy data, just store indices
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=10)
```

### Parameter Selection

**Region Size:**
- Default 50nm works well for most NIR/visible spectra
- Smaller regions (20-30nm): More granular, more subsets
- Larger regions (75-100nm): Fewer subsets, broader features

**Overlap:**
- Default 25nm (50% overlap) balances coverage and redundancy
- More overlap: Smoother transitions, more regions
- Less overlap: Faster computation, fewer regions
- No overlap: Independent regions, fastest

**n_top_regions:**
- Start with 5-10 for initial exploration
- Increase to 15-20 for comprehensive search
- Too many regions → slow search with diminishing returns
- Too few regions → may miss important features

```julia
# Quick exploration
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=5)

# Comprehensive search
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=15)
```

## Performance

The Regions module is optimized for efficiency:

| Dataset Size | Region Computation | Subset Creation |
|--------------|-------------------|-----------------|
| 100 × 200    | < 0.1s           | < 0.01s         |
| 500 × 1000   | < 0.5s           | < 0.05s         |
| 1000 × 2000  | < 1.0s           | < 0.1s          |

## Integration with Search

The Regions module integrates with the hyperparameter search:

```julia
# 1. Create region subsets
region_subsets = create_region_subsets(X_train, y_train, wavelengths, n_top_regions=10)

# 2. Add to search space
search_space = [
    # Full spectrum
    Dict("indices" => collect(1:size(X_train, 2)), "tag" => "full"),

    # Region subsets
    region_subsets...
]

# 3. Search over subsets × preprocessing × model params
for subset in search_space
    for preproc in preprocessing_methods
        for params in model_params
            # Test configuration...
        end
    end
end
```

## Examples

See `examples/regions_example.jl` for a complete working example with:
- Synthetic data generation
- Region computation and analysis
- Subset creation and usage
- Integration with model training
- Performance comparison

Run with:
```bash
julia --project=. examples/regions_example.jl
```

## Testing

Run the test suite:
```bash
julia --project=. test/test_regions.jl
```

Tests cover:
- Basic functionality
- Edge cases (few samples, constant values)
- Input validation
- Custom parameters
- Integration scenarios

## References

This implementation is based on the Python version in `src/spectral_predict/regions.py`, with optimizations for Julia:
- Uses native Julia types (Vector, Dict)
- Leverages Julia's efficient array operations
- Maintains exact algorithmic compatibility
- 1-based indexing (Julia convention)
