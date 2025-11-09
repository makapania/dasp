"""Spectral region analysis and selection utilities."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_region_correlations(X, y, wavelengths, region_size=50, overlap=25):
    """
    Divide spectrum into overlapping regions and compute correlation with target.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    region_size : float
        Size of each region in nm (default: 50)
    overlap : float
        Overlap between adjacent regions in nm (default: 25)

    Returns
    -------
    regions : list of dict
        List of region information with keys:
        - 'start': Start wavelength
        - 'end': End wavelength
        - 'indices': Feature indices in this region
        - 'mean_corr': Mean absolute correlation with target
        - 'max_corr': Maximum absolute correlation with target
        - 'n_features': Number of features in region
    """
    min_wl = wavelengths.min()
    max_wl = wavelengths.max()

    regions = []
    start_wl = min_wl

    while start_wl < max_wl:
        end_wl = start_wl + region_size

        # Find features in this region
        region_mask = (wavelengths >= start_wl) & (wavelengths < end_wl)
        region_indices = np.where(region_mask)[0]

        if len(region_indices) == 0:
            start_wl += (region_size - overlap)
            continue

        # Compute correlations for this region (vectorized for performance)
        # This computes all correlations at once instead of looping
        try:
            region_data = X[:, region_indices]

            # Stack region features with y and compute correlation matrix
            combined = np.column_stack([region_data, y.ravel()])
            corr_matrix = np.corrcoef(combined, rowvar=False)

            # Extract correlations between each feature and y (last row, excluding y vs y)
            feature_y_corrs = corr_matrix[:-1, -1]

            # Take absolute value and filter out NaNs
            correlations = np.abs(feature_y_corrs)
            correlations = correlations[~np.isnan(correlations)].tolist()
        except:
            correlations = []

        if len(correlations) > 0:
            regions.append({
                'start': start_wl,
                'end': end_wl,
                'indices': region_indices,
                'mean_corr': np.mean(correlations),
                'max_corr': np.max(correlations),
                'n_features': len(region_indices)
            })

        # Move to next region (with overlap)
        start_wl += (region_size - overlap)

    return regions


def get_top_regions(regions, n_top=5, criterion='mean_corr'):
    """
    Get top N regions by correlation.

    Parameters
    ----------
    regions : list of dict
        Region information from compute_region_correlations
    n_top : int
        Number of top regions to return
    criterion : str
        'mean_corr' or 'max_corr'

    Returns
    -------
    top_regions : list of dict
        Top N regions sorted by criterion
    """
    sorted_regions = sorted(regions, key=lambda r: r[criterion], reverse=True)
    return sorted_regions[:n_top]


def get_region_variable_indices(regions, return_combined=True):
    """
    Get variable indices for top regions.

    Parameters
    ----------
    regions : list of dict
        Region information (typically from get_top_regions)
    return_combined : bool
        If True, return combined indices from all regions
        If False, return list of indices for each region separately

    Returns
    -------
    indices : np.ndarray or list of np.ndarray
        Variable indices for regions
    """
    if return_combined:
        # Combine all indices from all regions
        all_indices = []
        for region in regions:
            all_indices.extend(region['indices'])
        return np.unique(all_indices)
    else:
        # Return separate indices for each region
        return [region['indices'] for region in regions]


def create_region_subsets(X, y, wavelengths, n_top_regions=5):
    """
    Create variable subsets based on spectral regions.

    This function identifies important spectral regions and creates
    multiple subset configurations for testing.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values
    n_top_regions : int
        Number of top regions to use (default: 5, can be up to 20)

    Returns
    -------
    subsets : list of dict
        List of subset configurations with keys:
        - 'indices': Variable indices
        - 'tag': Descriptive name (e.g., 'region1', 'top3regions')
        - 'description': Human-readable description
    """
    # Compute region correlations
    regions = compute_region_correlations(X, y, wavelengths)

    if len(regions) == 0:
        return []

    # Cap n_top_regions to available regions
    n_top_regions = min(n_top_regions, len(regions))

    # Get top regions
    top_regions = get_top_regions(regions, n_top=n_top_regions)

    subsets = []

    # Strategy: Test individual regions and strategic combinations
    # For n_top_regions=5: test top 3 individual + combinations (5-6 subsets)
    # For n_top_regions=10: test top 5 individual + combinations (8-10 subsets)
    # For n_top_regions=15: test top 7 individual + combinations (10-12 subsets)
    # For n_top_regions=20: test top 10 individual + combinations (13-15 subsets)

    # Determine how many individual regions to test
    if n_top_regions <= 5:
        n_individual = 3
    elif n_top_regions <= 10:
        n_individual = 5
    elif n_top_regions <= 15:
        n_individual = 7
    else:  # n_top_regions > 15
        n_individual = 10

    # Individual top regions (test each separately)
    for i, region in enumerate(top_regions[:n_individual], 1):
        if len(region['indices']) > 0:
            # Include actual wavelength range in tag for immediate interpretability
            wl_tag = f"{region['start']:.0f}-{region['end']:.0f}nm"
            subsets.append({
                'indices': region['indices'],
                'tag': f'region_{wl_tag}',
                'description': f"Region {i}: {region['start']:.0f}-{region['end']:.0f}nm "
                             f"(r={region['mean_corr']:.3f}, n={len(region['indices'])})"
            })

    # Combined top regions at strategic intervals
    # Test combinations: top-2, top-5, top-10, top-15, top-20 (as available)
    combination_sizes = [2, 5, 10, 15, 20]
    for combo_size in combination_sizes:
        if combo_size <= n_top_regions and combo_size > 1:
            indices_combo = get_region_variable_indices(top_regions[:combo_size])
            if len(indices_combo) > 0:
                # For readability, only show wavelength ranges for small combinations
                if combo_size <= 5:
                    wl_ranges = ','.join([f"{r['start']:.0f}-{r['end']:.0f}" for r in top_regions[:combo_size]])
                    tag_suffix = f"_{wl_ranges}nm"
                else:
                    tag_suffix = ""

                subsets.append({
                    'indices': indices_combo,
                    'tag': f'top{combo_size}regions{tag_suffix}',
                    'description': f"Top {combo_size} regions combined (n={len(indices_combo)})"
                })

    return subsets


def format_region_report(regions, wavelengths, n_top=10):
    """
    Create a formatted report of top spectral regions.

    Parameters
    ----------
    regions : list of dict
        Region information from compute_region_correlations
    wavelengths : np.ndarray
        Wavelength values
    n_top : int
        Number of top regions to include in report

    Returns
    -------
    report : str
        Formatted text report
    """
    top_regions = get_top_regions(regions, n_top=n_top)

    lines = []
    lines.append("=" * 70)
    lines.append("Top Spectral Regions (by correlation with target)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Region (nm)':<20} {'Mean |r|':<12} {'Max |r|':<12} {'N vars':<8}")
    lines.append("-" * 70)

    for i, region in enumerate(top_regions, 1):
        region_str = f"{region['start']:.0f}-{region['end']:.0f}"
        lines.append(
            f"{i:<6} {region_str:<20} {region['mean_corr']:<12.4f} "
            f"{region['max_corr']:<12.4f} {region['n_features']:<8}"
        )

    lines.append("")
    lines.append("Note: Regions with high correlations may indicate important")
    lines.append("spectral features related to the target variable.")
    lines.append("=" * 70)

    return "\n".join(lines)
