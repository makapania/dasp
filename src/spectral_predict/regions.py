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

        # Compute correlations for this region
        correlations = []
        for idx in region_indices:
            try:
                corr, _ = pearsonr(X[:, idx], y)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            except:
                pass

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
        Number of top regions to use (default: 5)

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

    # Get top regions
    top_regions = get_top_regions(regions, n_top=n_top_regions)

    subsets = []

    # Individual top regions (test each separately)
    for i, region in enumerate(top_regions[:3], 1):  # Top 3 individual regions
        if len(region['indices']) > 0:
            subsets.append({
                'indices': region['indices'],
                'tag': f'region{i}',
                'description': f"Region {i}: {region['start']:.0f}-{region['end']:.0f}nm "
                             f"(r={region['mean_corr']:.3f}, n={len(region['indices'])})"
            })

    # Combined top regions
    if n_top_regions >= 2:
        # Top 2 regions combined
        indices_top2 = get_region_variable_indices(top_regions[:2])
        if len(indices_top2) > 0:
            subsets.append({
                'indices': indices_top2,
                'tag': 'top2regions',
                'description': f"Top 2 regions combined (n={len(indices_top2)})"
            })

    if n_top_regions >= 3:
        # Top 3 regions combined
        indices_top3 = get_region_variable_indices(top_regions[:3])
        if len(indices_top3) > 0:
            subsets.append({
                'indices': indices_top3,
                'tag': 'top3regions',
                'description': f"Top 3 regions combined (n={len(indices_top3)})"
            })

    if n_top_regions >= 5:
        # Top 5 regions combined
        indices_top5 = get_region_variable_indices(top_regions[:5])
        if len(indices_top5) > 0:
            subsets.append({
                'indices': indices_top5,
                'tag': 'top5regions',
                'description': f"Top 5 regions combined (n={len(indices_top5)})"
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
