# Regions Module - Quick Reference

## Installation

```julia
# Already included in SpectralPredict package
# No additional dependencies needed
```

## Import

```julia
include("src/regions.jl")
using .Regions
```

## Basic Usage

### 1. Compute Region Correlations

```julia
regions = compute_region_correlations(X, y, wavelengths)
# Default: 50nm regions, 25nm overlap

# Custom parameters
regions = compute_region_correlations(X, y, wavelengths,
                                     region_size=100.0, overlap=50.0)
```

### 2. Create Region Subsets

```julia
subsets = create_region_subsets(X, y, wavelengths)
# Default: n_top_regions=5

# More comprehensive
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=15)
```

### 3. Use in Model Training

```julia
for subset in subsets
    X_subset = X[:, subset["indices"]]
    # Train model on X_subset
    println("Testing: ", subset["tag"])
end
```

## Complete Example

```julia
using Statistics
include("src/regions.jl")
using .Regions

# Load data (PREPROCESSED!)
X = randn(100, 200)  # 100 samples, 200 wavelengths
y = randn(100)
wavelengths = collect(400.0:2.0:798.0)

# Find important regions
regions = compute_region_correlations(X, y, wavelengths)
println("Found ", length(regions), " regions")

# Create subsets for testing
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=10)

# Use best region
best_subset = subsets[1]
X_best = X[:, best_subset["indices"]]
println("Best subset: ", best_subset["tag"])
println("Features: ", length(best_subset["indices"]))
```

## Return Value Structure

### `compute_region_correlations` returns:

```julia
[
    Dict(
        "start" => 400.0,              # Start wavelength
        "end" => 450.0,                # End wavelength
        "indices" => [1, 2, 3, ...],   # Feature indices (1-based)
        "mean_corr" => 0.75,           # Mean absolute correlation
        "max_corr" => 0.88,            # Max absolute correlation
        "n_features" => 25             # Number of features
    ),
    ...
]
```

### `create_region_subsets` returns:

```julia
[
    Dict(
        "indices" => [1, 2, 3, ...],         # Feature indices
        "tag" => "region_400-450nm",         # Short tag
        "description" => "Region 1: ..."     # Full description
    ),
    Dict(
        "indices" => [1, 2, ..., 50],
        "tag" => "top5regions",
        "description" => "Top 5 regions combined (n=120)"
    ),
    ...
]
```

## Common Patterns

### Find Top Regions

```julia
regions = compute_region_correlations(X, y, wavelengths)
sorted = sort(regions, by=r -> r["mean_corr"], rev=true)
top5 = sorted[1:5]

for (i, region) in enumerate(top5)
    println("$i. $(region["start"])-$(region["end"]) nm: r=$(region["mean_corr"])")
end
```

### Combine Specific Regions

```julia
# Get top 3 regions
top3 = sorted_regions[1:3]

# Combine their indices
combined = combine_region_indices(top3)

# Use combined data
X_combined = X[:, combined]
```

### Filter by Correlation Threshold

```julia
high_corr_regions = filter(r -> r["mean_corr"] > 0.5, regions)
println("Found ", length(high_corr_regions), " high-correlation regions")
```

## Parameter Guidelines

### region_size
- **50nm** (default): Good for most NIR/visible spectra
- **20-30nm**: More granular, more subsets
- **75-100nm**: Broader features, fewer subsets

### overlap
- **25nm** (default): 50% overlap, good balance
- **0nm**: No overlap, independent regions
- **>50%**: Heavy overlap, smoother transitions

### n_top_regions
- **5** (default): Quick exploration, ~6-8 subsets
- **10**: Standard search, ~8-10 subsets
- **15**: Comprehensive, ~10-12 subsets
- **20**: Exhaustive, ~13-15 subsets

## Subset Strategy

| n_top_regions | Individual Regions | Combined Regions |
|---------------|-------------------|------------------|
| ≤ 5 | Top 3 | 2, 5 |
| ≤ 10 | Top 5 | 2, 5, 10 |
| ≤ 15 | Top 7 | 2, 5, 10, 15 |
| > 15 | Top 10 | 2, 5, 10, 15, 20 |

## Integration with Search

```julia
# Create search space
subsets = create_region_subsets(X_train, y_train, wavelengths, n_top_regions=10)

preprocessing = ["None", "SNV", "Derivative1", "Derivative2"]
n_lvs_values = [5, 10, 15, 20]

# Search
for subset in subsets
    X_sub = X_train[:, subset["indices"]]
    for prep in preprocessing
        X_prep = preprocess(X_sub, prep)
        for n_lvs in n_lvs_values
            score = train_evaluate(X_prep, y_train, n_lvs)
            # Store result...
        end
    end
end
```

## Important Notes

### ⚠️ Use Preprocessed Data

```julia
# CORRECT
X_preprocessed = apply_snv(X_raw)
regions = compute_region_correlations(X_preprocessed, y, wavelengths)

# WRONG
regions = compute_region_correlations(X_raw, y, wavelengths)  # ❌
```

### ⚠️ 1-Based Indexing

```julia
# Julia uses 1-based indexing
indices = [1, 2, 3, ...]  # First element is 1, not 0
X_subset = X[:, indices]  # Works directly
```

### ⚠️ Minimum Requirements

```julia
# Need at least 2 samples for correlation
n_samples >= 2  # Required

# Need at least 1 wavelength
n_wavelengths >= 1  # Required
```

## Error Handling

```julia
try
    regions = compute_region_correlations(X, y, wavelengths)
catch e
    if isa(e, AssertionError)
        println("Input validation failed: ", e)
    else
        println("Error: ", e)
    end
end
```

## Testing

```bash
# Run test suite
julia --project=. test/test_regions.jl

# Run example
julia --project=. examples/regions_example.jl
```

## Performance Tips

1. **Precompile:** First run compiles, subsequent runs are fast
2. **Type stability:** Use Matrix{Float64}, not Matrix{Any}
3. **Preallocate:** Regions are created efficiently
4. **Views:** Use views (@view) for large arrays if needed

## Troubleshooting

### No regions found
- Check wavelength range vs region_size
- Verify data is not all NaN/Inf
- Try larger region_size

### Few subsets created
- Increase n_top_regions
- Check if enough regions exist
- Verify data has correlation signal

### High memory usage
- Subsets store indices only (minimal memory)
- Don't store all X_subset copies
- Use views when possible

## API Summary

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `compute_region_correlations` | X, y, wavelengths | regions | Find correlated regions |
| `create_region_subsets` | X, y, wavelengths | subsets | Create testing subsets |
| `combine_region_indices` | regions | indices | Merge region indices |

## Files

- **Source:** `src/regions.jl`
- **Tests:** `test/test_regions.jl`
- **Example:** `examples/regions_example.jl`
- **Docs:** `docs/regions_module.md`

## Support

See full documentation: `docs/regions_module.md`

---

**Quick Start:**
```julia
include("src/regions.jl")
using .Regions
subsets = create_region_subsets(X, y, wavelengths)
X_best = X[:, subsets[1]["indices"]]
```
