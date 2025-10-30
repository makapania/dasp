"""
    regions.jl

Spectral region analysis and selection utilities.

This module identifies important spectral regions based on correlation with the target
variable. It divides the spectrum into overlapping windows and computes correlations
to find the most informative spectral regions.

Key Features:
- Automatic region division with configurable size and overlap
- Correlation-based region ranking
- Individual and combined region subsets for model testing
- Works on preprocessed data (after SNV/derivatives)

Typical Usage:
1. Compute region correlations on training data
2. Create region subsets for variable selection
3. Use subsets in model hyperparameter search
"""

module Regions

using Statistics

export compute_region_correlations, create_region_subsets, combine_region_indices


"""
    compute_region_correlations(
        X::Matrix{Float64},
        y::Vector{Float64},
        wavelengths::Vector{Float64};
        region_size::Float64=50.0,
        overlap::Float64=25.0
    )::Vector{Dict{String, Any}}

Divide spectrum into overlapping regions and compute correlation with target.

This function segments the spectral data into overlapping windows and computes the
correlation between each wavelength and the target variable. Regions are characterized
by their mean and maximum absolute correlations.

# Algorithm

1. Determine spectral range from wavelengths
2. Create overlapping windows: [start, start+size), [start+size-overlap, ...)
3. For each region:
   - Find wavelength indices in window
   - Compute Pearson correlation for each wavelength with target
   - Calculate mean and max absolute correlations

# Arguments
- `X::Matrix{Float64}`: Spectral data (n_samples × n_wavelengths), **preprocessed**
- `y::Vector{Float64}`: Target values (length n_samples)
- `wavelengths::Vector{Float64}`: Wavelength values for each feature (length n_wavelengths)
- `region_size::Float64`: Size of each region in nm (default: 50.0)
- `overlap::Float64`: Overlap between adjacent regions in nm (default: 25.0)

# Returns
- `Vector{Dict{String, Any}}`: List of region information dictionaries with keys:
  - `"start"::Float64`: Start wavelength
  - `"end"::Float64`: End wavelength
  - `"indices"::Vector{Int}`: Feature indices in this region (1-indexed)
  - `"mean_corr"::Float64`: Mean absolute correlation with target
  - `"max_corr"::Float64`: Maximum absolute correlation with target
  - `"n_features"::Int`: Number of features in region

# Notes
- Regions with no features are skipped
- NaN correlations (e.g., from constant features) are excluded
- Indices are 1-based (Julia convention)
- Operates on **preprocessed** data, not raw spectra
- Requires at least 2 samples for correlation computation

# Examples
```julia
# Basic usage with default parameters
X = randn(100, 200)  # 100 samples, 200 wavelengths
y = randn(100)
wavelengths = collect(400.0:2.0:798.0)  # 400-798 nm in 2nm steps

regions = compute_region_correlations(X, y, wavelengths)

# First region information
println("Region: ", regions[1]["start"], "-", regions[1]["end"], " nm")
println("Mean correlation: ", regions[1]["mean_corr"])
println("Features: ", length(regions[1]["indices"]))

# Custom region size and overlap
regions = compute_region_correlations(X, y, wavelengths,
                                     region_size=100.0, overlap=50.0)
```
"""
function compute_region_correlations(
    X::Matrix{Float64},
    y::Vector{Float64},
    wavelengths::Vector{Float64};
    region_size::Float64=50.0,
    overlap::Float64=25.0
)::Vector{Dict{String, Any}}

    # Validate inputs
    n_samples, n_wavelengths = size(X)
    @assert length(y) == n_samples "y length ($(length(y))) must match X rows ($n_samples)"
    @assert length(wavelengths) == n_wavelengths "wavelengths length ($(length(wavelengths))) must match X columns ($n_wavelengths)"
    @assert region_size > 0 "region_size must be positive"
    @assert overlap >= 0 "overlap must be non-negative"
    @assert overlap < region_size "overlap must be less than region_size"
    @assert n_samples >= 2 "Need at least 2 samples to compute correlations"

    # Get wavelength range
    min_wl = minimum(wavelengths)
    max_wl = maximum(wavelengths)

    regions = Dict{String, Any}[]
    start_wl = min_wl

    # Create overlapping regions
    while start_wl < max_wl
        end_wl = start_wl + region_size

        # Find features in this region
        region_mask = (wavelengths .>= start_wl) .& (wavelengths .< end_wl)
        region_indices = findall(region_mask)

        if length(region_indices) == 0
            # No features in this region, move to next
            start_wl += (region_size - overlap)
            continue
        end

        # Compute correlations for this region
        correlations = Float64[]
        for idx in region_indices
            try
                # Compute Pearson correlation
                corr_val = cor(X[:, idx], y)

                # Only include valid correlations
                if !isnan(corr_val) && !isinf(corr_val)
                    push!(correlations, abs(corr_val))
                end
            catch
                # Skip features that cause errors (e.g., constant values)
                continue
            end
        end

        # Only add region if it has valid correlations
        if length(correlations) > 0
            push!(regions, Dict{String, Any}(
                "start" => start_wl,
                "end" => end_wl,
                "indices" => region_indices,
                "mean_corr" => mean(correlations),
                "max_corr" => maximum(correlations),
                "n_features" => length(region_indices)
            ))
        end

        # Move to next region (with overlap)
        start_wl += (region_size - overlap)
    end

    return regions
end


"""
    create_region_subsets(
        X::Matrix{Float64},
        y::Vector{Float64},
        wavelengths::Vector{Float64};
        n_top_regions::Int=5
    )::Vector{Dict{String, Any}}

Create variable subsets based on spectral regions for model testing.

This function identifies important spectral regions and creates multiple subset
configurations for hyperparameter search. It generates both individual region
subsets and strategic combinations of top regions.

# Strategy

The function creates two types of subsets:

1. **Individual Regions**: Test each of the top N individual regions separately
   - n_top_regions ≤ 5: Test top 3 individual regions
   - n_top_regions ≤ 10: Test top 5 individual regions
   - n_top_regions ≤ 15: Test top 7 individual regions
   - n_top_regions > 15: Test top 10 individual regions

2. **Combined Regions**: Test strategic combinations
   - Top 2 regions combined
   - Top 5 regions combined
   - Top 10 regions combined
   - Top 15 regions combined
   - Top 20 regions combined
   (Only creates combinations that fit within n_top_regions)

# Arguments
- `X::Matrix{Float64}`: Spectral data (n_samples × n_wavelengths), **preprocessed**
- `y::Vector{Float64}`: Target values (length n_samples)
- `wavelengths::Vector{Float64}`: Wavelength values (length n_wavelengths)
- `n_top_regions::Int`: Maximum number of top regions to use (default: 5, can be up to 20)

# Returns
- `Vector{Dict{String, Any}}`: List of subset configurations with keys:
  - `"indices"::Vector{Int}`: Variable indices for this subset (1-indexed)
  - `"tag"::String`: Descriptive tag (e.g., "region_400-450nm", "top5regions")
  - `"description"::String`: Human-readable description

# Edge Cases
- Returns empty vector if no regions found
- Caps n_top_regions to available regions
- Skips empty regions
- Handles cases where fewer regions exist than requested

# Notes
- Regions are ranked by mean absolute correlation
- Combined regions use sorted unique indices
- Region tags include wavelength ranges for interpretability
- Works on **preprocessed** data (after SNV/derivatives)

# Examples
```julia
# Basic usage - creates ~6-8 subsets
X = randn(100, 200)
y = randn(100)
wavelengths = collect(400.0:2.0:798.0)

subsets = create_region_subsets(X, y, wavelengths)

for subset in subsets
    println(subset["tag"], ": ", length(subset["indices"]), " features")
    println("  ", subset["description"])
end

# Request more regions for comprehensive search
subsets = create_region_subsets(X, y, wavelengths, n_top_regions=15)
println("Created ", length(subsets), " subsets")

# Use subsets in model search
for subset in subsets
    X_subset = X[:, subset["indices"]]
    # Train model on X_subset...
end
```
"""
function create_region_subsets(
    X::Matrix{Float64},
    y::Vector{Float64},
    wavelengths::Vector{Float64};
    n_top_regions::Int=5
)::Vector{Dict{String, Any}}

    # Validate inputs
    @assert n_top_regions > 0 "n_top_regions must be positive"

    # Compute region correlations
    regions = compute_region_correlations(X, y, wavelengths)

    # Handle edge case: no regions found
    if length(regions) == 0
        return Dict{String, Any}[]
    end

    # Cap n_top_regions to available regions
    n_top_regions = min(n_top_regions, length(regions))

    # Sort regions by mean correlation (descending)
    sorted_regions = sort(regions, by=r -> r["mean_corr"], rev=true)
    top_regions = sorted_regions[1:n_top_regions]

    subsets = Dict{String, Any}[]

    # Determine how many individual regions to test
    n_individual = if n_top_regions <= 5
        3
    elseif n_top_regions <= 10
        5
    elseif n_top_regions <= 15
        7
    else  # n_top_regions > 15
        10
    end

    # Cap to available regions
    n_individual = min(n_individual, n_top_regions)

    # Create individual region subsets
    for (i, region) in enumerate(top_regions[1:n_individual])
        if length(region["indices"]) > 0
            # Include wavelength range in tag for interpretability
            wl_tag = string(Int(round(region["start"])), "-", Int(round(region["end"])), "nm")

            push!(subsets, Dict{String, Any}(
                "indices" => region["indices"],
                "tag" => "region_" * wl_tag,
                "description" => string(
                    "Region ", i, ": ",
                    Int(round(region["start"])), "-", Int(round(region["end"])), "nm ",
                    "(r=", round(region["mean_corr"], digits=3),
                    ", n=", length(region["indices"]), ")"
                )
            ))
        end
    end

    # Create combined region subsets at strategic intervals
    combination_sizes = [2, 5, 10, 15, 20]

    for combo_size in combination_sizes
        if combo_size <= n_top_regions && combo_size > 1
            # Combine indices from top N regions
            indices_combo = combine_region_indices(top_regions[1:combo_size])

            if length(indices_combo) > 0
                # For small combinations, include wavelength ranges
                tag_suffix = ""
                if combo_size <= 5
                    wl_ranges = join([
                        string(Int(round(r["start"])), "-", Int(round(r["end"])))
                        for r in top_regions[1:combo_size]
                    ], ",")
                    tag_suffix = "_" * wl_ranges * "nm"
                end

                push!(subsets, Dict{String, Any}(
                    "indices" => indices_combo,
                    "tag" => "top" * string(combo_size) * "regions" * tag_suffix,
                    "description" => "Top " * string(combo_size) * " regions combined (n=" *
                                   string(length(indices_combo)) * ")"
                ))
            end
        end
    end

    return subsets
end


"""
    combine_region_indices(regions::Vector{Dict{String, Any}})::Vector{Int}

Combine indices from multiple regions into a single sorted unique list.

This is a utility function for merging wavelength indices from multiple spectral
regions. Duplicate indices are removed, and the result is sorted.

# Arguments
- `regions::Vector{Dict{String, Any}}`: Vector of region dictionaries with "indices" key

# Returns
- `Vector{Int}`: Sorted unique indices (1-indexed)

# Notes
- Removes duplicates automatically
- Returns sorted indices for consistent ordering
- Empty regions are handled gracefully
- Maintains 1-based indexing (Julia convention)

# Examples
```julia
# Combine indices from multiple regions
regions = [
    Dict("indices" => [1, 2, 3, 4, 5]),
    Dict("indices" => [4, 5, 6, 7, 8]),  # Overlap with first region
    Dict("indices" => [10, 11, 12])
]

combined = combine_region_indices(regions)
# Result: [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

# Use in model selection
X_combined = X[:, combined]

# Handle empty regions gracefully
regions = [
    Dict("indices" => [1, 2, 3]),
    Dict("indices" => Int[]),  # Empty region
    Dict("indices" => [5, 6])
]
combined = combine_region_indices(regions)  # [1, 2, 3, 5, 6]
```
"""
function combine_region_indices(regions::Vector{Dict{String, Any}})::Vector{Int}
    # Collect all indices
    all_indices = Int[]

    for region in regions
        if haskey(region, "indices")
            append!(all_indices, region["indices"])
        end
    end

    # Return sorted unique indices
    return sort(unique(all_indices))
end


end  # module Regions
