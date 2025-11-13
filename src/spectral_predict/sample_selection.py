"""
spectral_predict.sample_selection
==================================

Sample selection algorithms for calibration transfer and model development.

This module implements several algorithms for selecting representative samples
from a dataset:

- Kennard-Stone (KS): Selects samples to maximize diversity in X-space
- DUPLEX: Splits dataset into calibration and validation sets
- SPXY: Selects samples based on joint X-Y space diversity
- Random: Baseline random selection

These algorithms are particularly useful for:
1. Selecting transfer samples for calibration transfer (TSR, JYPLS-inv)
2. Creating representative calibration/validation splits
3. Optimal sample selection for experimental design
"""

from __future__ import annotations

from typing import Tuple, Dict, Literal
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


SelectionMethod = Literal["kennard-stone", "duplex", "spxy", "random"]


def kennard_stone(
    X: np.ndarray,
    n_samples: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Kennard-Stone algorithm for selecting representative samples.

    The Kennard-Stone (KS) algorithm selects samples that are maximally
    diverse in the feature space. It starts by selecting the two samples
    with the maximum distance, then iteratively adds samples that are
    farthest from the already-selected set.

    Algorithm:
    1. Find two samples with maximum Euclidean distance
    2. For i = 3 to n_samples:
       - For each remaining sample, compute minimum distance to selected set
       - Add the sample with maximum minimum-distance
    3. Return indices of selected samples

    Parameters
    ----------
    X : np.ndarray, shape (n_total_samples, n_features)
        Feature matrix (e.g., spectra).
    n_samples : int
        Number of samples to select.
    metric : str, default='euclidean'
        Distance metric to use (see scipy.spatial.distance.cdist).

    Returns
    -------
    selected_indices : np.ndarray, shape (n_samples,)
        Indices of selected samples in original dataset.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.sample_selection import kennard_stone
    >>>
    >>> # Create synthetic dataset
    >>> X = np.random.randn(100, 50)
    >>>
    >>> # Select 15 representative samples
    >>> indices = kennard_stone(X, n_samples=15)
    >>> X_selected = X[indices]
    >>>
    >>> print(f"Selected {len(indices)} samples")
    >>> print(f"Index range: {indices.min()} to {indices.max()}")

    References
    ----------
    .. [1] Kennard, R. W., & Stone, L. A. (1969). Computer aided design of
           experiments. Technometrics, 11(1), 137-148.

    Notes
    -----
    - Computational complexity: O(n^2 * p) where n is total samples, p is features
    - For large datasets (>5000 samples), consider using a subset first
    - The algorithm is deterministic given the same input
    """
    n_total, n_features = X.shape

    # Validate inputs
    if n_samples > n_total:
        raise ValueError(
            f"Cannot select {n_samples} samples from dataset with only {n_total} samples"
        )
    if n_samples < 2:
        raise ValueError("Must select at least 2 samples")

    # Step 1: Find two samples with maximum distance
    # Compute pairwise distances
    distances = pdist(X, metric=metric)
    distance_matrix = squareform(distances)

    # Find the pair with maximum distance
    max_dist_idx = np.argmax(distances)
    # Convert condensed distance matrix index to (i, j) pair
    n = n_total
    i = int(np.floor(0.5 * (1 + np.sqrt(1 + 8 * max_dist_idx))))
    j = max_dist_idx - i * (i - 1) // 2

    selected_indices = [i, j]
    remaining_indices = list(range(n_total))
    remaining_indices.remove(i)
    remaining_indices.remove(j)

    # Step 2: Iteratively add samples
    for _ in range(n_samples - 2):
        # Compute distances from remaining samples to selected samples
        X_selected = X[selected_indices]
        X_remaining = X[remaining_indices]

        # For each remaining sample, find minimum distance to selected set
        dist_to_selected = cdist(X_remaining, X_selected, metric=metric)
        min_distances = np.min(dist_to_selected, axis=1)

        # Select the sample with maximum minimum-distance
        max_min_idx = np.argmax(min_distances)
        selected_sample = remaining_indices[max_min_idx]

        selected_indices.append(selected_sample)
        remaining_indices.remove(selected_sample)

    return np.array(selected_indices, dtype=int)


def duplex(
    X: np.ndarray,
    y: np.ndarray | None = None,
    n_cal: int | None = None,
    cal_ratio: float = 0.75,
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DUPLEX algorithm for splitting dataset into calibration and validation sets.

    DUPLEX extends Kennard-Stone by alternately assigning selected samples
    to calibration and validation sets, ensuring both sets are representative
    of the full dataset.

    Algorithm:
    1. Use KS to select samples alternately for cal/val sets
    2. First sample -> calibration
    3. Second sample -> validation
    4. Third sample -> calibration, etc.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,), optional
        Target values. If provided, can be used for SPXY-like selection.
        Currently not used, reserved for future enhancement.
    n_cal : int, optional
        Number of calibration samples. If None, uses cal_ratio.
    cal_ratio : float, default=0.75
        Ratio of calibration samples (between 0 and 1).
    metric : str, default='euclidean'
        Distance metric for KS algorithm.

    Returns
    -------
    cal_indices : np.ndarray
        Indices of calibration samples.
    val_indices : np.ndarray
        Indices of validation samples.

    Examples
    --------
    >>> from spectral_predict.sample_selection import duplex
    >>>
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>>
    >>> # Split into 75/25 calibration/validation
    >>> cal_idx, val_idx = duplex(X, y, cal_ratio=0.75)
    >>>
    >>> X_cal, y_cal = X[cal_idx], y[cal_idx]
    >>> X_val, y_val = X[val_idx], y[val_idx]
    >>>
    >>> print(f"Calibration: {len(cal_idx)} samples")
    >>> print(f"Validation: {len(val_idx)} samples")

    References
    ----------
    .. [1] Snee, R. D. (1977). Validation of regression models: methods and
           examples. Technometrics, 19(4), 415-428.

    Notes
    -----
    - DUPLEX ensures both cal and val sets span the feature space
    - More robust than random splitting
    - Deterministic (given same input)
    """
    n_total = X.shape[0]

    # Determine number of calibration samples
    if n_cal is None:
        n_cal = int(n_total * cal_ratio)

    n_val = n_total - n_cal

    if n_cal < 1 or n_val < 1:
        raise ValueError(
            f"Invalid split: {n_cal} cal, {n_val} val. "
            f"Adjust cal_ratio or n_cal."
        )

    # Use Kennard-Stone to select all samples in order
    all_selected = kennard_stone(X, n_samples=n_total, metric=metric)

    # Alternate assignment: odd indices -> cal, even indices -> val
    # This ensures both sets are representative
    cal_indices = []
    val_indices = []

    for i, idx in enumerate(all_selected):
        if len(cal_indices) < n_cal and len(val_indices) < n_val:
            # Alternate assignment
            if i % 2 == 0:
                cal_indices.append(idx)
            else:
                val_indices.append(idx)
        elif len(cal_indices) < n_cal:
            cal_indices.append(idx)
        else:
            val_indices.append(idx)

    return np.array(cal_indices, dtype=int), np.array(val_indices, dtype=int)


def spxy(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Sample set Partitioning based on joint X-Y distance (SPXY).

    SPXY extends Kennard-Stone by considering both feature space (X) and
    target space (Y) when selecting samples. This is particularly useful
    when you want samples that are diverse in both spectral and reference
    value spaces.

    Algorithm:
    1. Normalize X and Y to [0, 1] range
    2. Compute combined distance: d_xy = d_x + d_y
    3. Apply KS algorithm on combined distance

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
        Target values.
    n_samples : int
        Number of samples to select.
    metric : str, default='euclidean'
        Distance metric for X space.

    Returns
    -------
    selected_indices : np.ndarray, shape (n_samples,)
        Indices of selected samples.

    Examples
    --------
    >>> from spectral_predict.sample_selection import spxy
    >>>
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>>
    >>> # Select 20 samples diverse in both X and Y
    >>> indices = spxy(X, y, n_samples=20)
    >>>
    >>> X_selected = X[indices]
    >>> y_selected = y[indices]
    >>>
    >>> print(f"Selected samples Y range: {y_selected.min():.2f} to {y_selected.max():.2f}")

    References
    ----------
    .. [1] Galvão, R. K., et al. (2005). A method for calibration and
           validation subset partitioning. Talanta, 67(4), 736-740.

    Notes
    -----
    - SPXY is particularly useful for calibration transfer sample selection
    - Ensures selected samples span both spectral and concentration ranges
    - Normalization is critical for balancing X and Y contributions
    """
    n_total = X.shape[0]

    # Validate inputs
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
        )

    if n_samples > n_total:
        raise ValueError(
            f"Cannot select {n_samples} samples from dataset with only {n_total} samples"
        )

    # Step 1: Normalize X and y to [0, 1]
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    y_norm = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0) + 1e-10)

    # Step 2: Compute pairwise distances in X and Y spaces
    dist_X = squareform(pdist(X_norm, metric=metric))
    dist_Y = squareform(pdist(y_norm, metric='euclidean'))

    # Step 3: Combine distances (equal weighting)
    dist_XY = dist_X + dist_Y

    # Step 4: Apply KS algorithm on combined distance
    # Find initial pair with maximum combined distance
    max_dist_idx = np.argmax(dist_XY)
    i, j = np.unravel_index(max_dist_idx, dist_XY.shape)

    selected_indices = [i, j]
    remaining_indices = list(range(n_total))
    remaining_indices.remove(i)
    remaining_indices.remove(j)

    # Iteratively add samples
    for _ in range(n_samples - 2):
        # For each remaining sample, find minimum distance to selected set
        min_distances = []
        for idx in remaining_indices:
            distances_to_selected = [dist_XY[idx, s] for s in selected_indices]
            min_distances.append(min(distances_to_selected))

        # Select sample with maximum minimum-distance
        max_min_idx = np.argmax(min_distances)
        selected_sample = remaining_indices[max_min_idx]

        selected_indices.append(selected_sample)
        remaining_indices.remove(selected_sample)

    return np.array(selected_indices, dtype=int)


def random_selection(
    n_total: int,
    n_samples: int,
    random_state: int | None = None
) -> np.ndarray:
    """
    Random sample selection (baseline method).

    Parameters
    ----------
    n_total : int
        Total number of samples available.
    n_samples : int
        Number of samples to select.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    selected_indices : np.ndarray
        Randomly selected sample indices.

    Examples
    --------
    >>> from spectral_predict.sample_selection import random_selection
    >>>
    >>> # Select 20 random samples from 100
    >>> indices = random_selection(100, 20, random_state=42)
    >>> print(len(indices))  # 20
    """
    if random_state is not None:
        np.random.seed(random_state)

    if n_samples > n_total:
        raise ValueError(
            f"Cannot select {n_samples} samples from {n_total} total samples"
        )

    return np.random.choice(n_total, size=n_samples, replace=False)


def compare_selection_methods(
    X: np.ndarray,
    y: np.ndarray | None = None,
    n_samples: int = 20,
    methods: list[SelectionMethod] | None = None,
    metric: str = 'euclidean',
    random_state: int | None = 42
) -> Dict[str, Dict]:
    """
    Compare different sample selection methods.

    Evaluates how well each method covers the feature space by computing
    diversity metrics on the selected samples.

    Parameters
    ----------
    X : np.ndarray, shape (n_total, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_total,), optional
        Target values (required for SPXY).
    n_samples : int, default=20
        Number of samples to select.
    methods : list of str, optional
        Methods to compare. Default: ['kennard-stone', 'duplex', 'spxy', 'random']
    metric : str, default='euclidean'
        Distance metric.
    random_state : int, optional
        Random seed for reproducible random selection.

    Returns
    -------
    results : dict
        Dictionary with method names as keys and metrics as values:
        {
            'kennard-stone': {
                'indices': np.ndarray,
                'mean_distance': float,
                'min_distance': float,
                'coverage': float
            },
            ...
        }

    Examples
    --------
    >>> from spectral_predict.sample_selection import compare_selection_methods
    >>>
    >>> X = np.random.randn(200, 50)
    >>> y = np.random.randn(200)
    >>>
    >>> results = compare_selection_methods(X, y, n_samples=30)
    >>>
    >>> for method, metrics in results.items():
    >>>     print(f"{method}: mean_dist={metrics['mean_distance']:.3f}")

    Notes
    -----
    - Higher mean_distance indicates better coverage
    - Higher min_distance indicates no sample clustering
    - Coverage metric shows fraction of feature space spanned
    """
    if methods is None:
        methods = ['kennard-stone', 'random']
        if y is not None:
            methods.extend(['duplex', 'spxy'])

    n_total = X.shape[0]
    results = {}

    for method in methods:
        if method == 'kennard-stone':
            indices = kennard_stone(X, n_samples, metric=metric)

        elif method == 'duplex':
            if y is None:
                print(f"Skipping {method}: requires y values")
                continue
            cal_idx, val_idx = duplex(X, y, n_cal=n_samples, metric=metric)
            indices = cal_idx  # Use calibration set for comparison

        elif method == 'spxy':
            if y is None:
                print(f"Skipping {method}: requires y values")
                continue
            indices = spxy(X, y, n_samples, metric=metric)

        elif method == 'random':
            indices = random_selection(n_total, n_samples, random_state=random_state)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute diversity metrics
        X_selected = X[indices]
        distances = pdist(X_selected, metric=metric)

        mean_distance = np.mean(distances)
        min_distance = np.min(distances) if len(distances) > 0 else 0.0

        # Coverage: fraction of feature space spanned
        # Compute ratio of selected range to total range per feature
        ranges_selected = X_selected.max(axis=0) - X_selected.min(axis=0)
        ranges_total = X.max(axis=0) - X.min(axis=0)
        coverage = np.mean(ranges_selected / (ranges_total + 1e-10))

        results[method] = {
            'indices': indices,
            'mean_distance': mean_distance,
            'min_distance': min_distance,
            'coverage': coverage,
            'n_samples': len(indices)
        }

    return results


if __name__ == "__main__":
    # Simple demonstration
    print("Sample Selection Module")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    print(f"\nDataset: {n_samples} samples, {n_features} features")

    # Test Kennard-Stone
    print("\n1. Kennard-Stone Selection:")
    ks_indices = kennard_stone(X, n_samples=15)
    print(f"   Selected {len(ks_indices)} samples: {ks_indices[:5]}...")

    # Test DUPLEX
    print("\n2. DUPLEX Split:")
    cal_idx, val_idx = duplex(X, y, cal_ratio=0.75)
    print(f"   Calibration: {len(cal_idx)} samples")
    print(f"   Validation: {len(val_idx)} samples")

    # Test SPXY
    print("\n3. SPXY Selection:")
    spxy_indices = spxy(X, y, n_samples=15)
    print(f"   Selected {len(spxy_indices)} samples: {spxy_indices[:5]}...")

    # Compare methods
    print("\n4. Method Comparison:")
    comparison = compare_selection_methods(X, y, n_samples=20)
    for method, metrics in comparison.items():
        print(f"   {method:20s}: mean_dist={metrics['mean_distance']:.3f}, "
              f"coverage={metrics['coverage']:.3f}")

    print("\n" + "=" * 50)
    print("Sample selection module loaded successfully!")
