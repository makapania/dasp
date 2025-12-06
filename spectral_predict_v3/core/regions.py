"""
Spectral region analysis and selection utilities (v3 standalone).

Forked from v1 - standalone implementation for v3's numpy-first approach.
Supports both regression (correlation-based) and classification (discriminant-based).
"""

import numpy as np


def compute_region_correlations(X, y, wavelengths, region_size=50, overlap=25, task_type='regression'):
    """
    Divide spectrum into overlapping regions and compute importance score with target.

    For regression: uses absolute correlation with target.
    For classification: uses Fisher's discriminant ratio (between-class / within-class variance).

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values (continuous for regression, class labels for classification)
    wavelengths : np.ndarray
        Wavelength values for each feature
    region_size : float
        Size of each region in nm (default: 50)
    overlap : float
        Overlap between adjacent regions in nm (default: 25)
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    regions : list of dict
        List of region information with keys:
        - 'start': Start wavelength
        - 'end': End wavelength
        - 'indices': Feature indices in this region
        - 'mean_corr': Mean importance score (correlation for regression, Fisher ratio for classification)
        - 'max_corr': Maximum importance score
        - 'n_features': Number of features in region
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    wavelengths = np.asarray(wavelengths)

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

        # Compute importance scores for this region
        try:
            region_data = X[:, region_indices]

            if task_type == 'classification':
                # Fisher's discriminant ratio for classification
                # Higher ratio = better class separation
                scores = _compute_fisher_scores(region_data, y)
            else:
                # Correlation for regression
                # Stack region features with y and compute correlation matrix
                combined = np.column_stack([region_data, y.ravel()])
                corr_matrix = np.corrcoef(combined, rowvar=False)

                # Extract correlations between each feature and y
                feature_y_corrs = corr_matrix[:-1, -1]

                # Take absolute value
                scores = np.abs(feature_y_corrs)

            # Filter out NaNs
            scores = scores[~np.isnan(scores)].tolist()
        except:
            scores = []

        if len(scores) > 0:
            regions.append({
                'start': start_wl,
                'end': end_wl,
                'indices': region_indices,
                'mean_corr': np.mean(scores),
                'max_corr': np.max(scores),
                'n_features': len(region_indices)
            })

        # Move to next region (with overlap)
        start_wl += (region_size - overlap)

    return regions


def _compute_fisher_scores(X, y):
    """
    Compute Fisher's discriminant ratio for each feature.

    Fisher ratio = (between-class variance) / (within-class variance)
    Higher values indicate better class separation.

    Parameters
    ----------
    X : np.ndarray
        Feature data (n_samples, n_features)
    y : np.ndarray
        Class labels

    Returns
    -------
    scores : np.ndarray
        Fisher ratio for each feature
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    scores = np.zeros(n_features)

    # Global mean for each feature
    global_mean = np.mean(X, axis=0)

    for j in range(n_features):
        # Between-class variance
        between_var = 0.0
        within_var = 0.0

        for c in classes:
            class_mask = (y == c)
            class_data = X[class_mask, j]
            n_c = len(class_data)

            if n_c == 0:
                continue

            class_mean = np.mean(class_data)
            class_var = np.var(class_data)

            # Between-class: weighted squared distance from global mean
            between_var += n_c * (class_mean - global_mean[j]) ** 2

            # Within-class: sum of class variances
            within_var += n_c * class_var

        # Fisher ratio (add small epsilon to avoid division by zero)
        if within_var > 1e-10:
            scores[j] = between_var / within_var
        else:
            scores[j] = 0.0

    return scores


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


def create_region_subsets(X, y, wavelengths, n_top_regions=5, task_type='regression'):
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
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    subsets : list of dict
        List of subset configurations with keys:
        - 'indices': Variable indices
        - 'tag': Descriptive name (e.g., 'region1', 'top3regions')
        - 'description': Human-readable description
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    wavelengths = np.asarray(wavelengths)

    # Compute region importance scores
    regions = compute_region_correlations(X, y, wavelengths, task_type=task_type)

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
