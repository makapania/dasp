"""
Sample selection algorithms for Spectral Predict v3.

Implements methods for selecting representative samples for validation holdout:
- SPXY: Diverse in both X (spectra) AND Y (target) spaces (recommended default)
- DUPLEX: Splits into cal/val sets where both are representative
- Kennard-Stone: Maximizes diversity in X-space only
- Random: Baseline random selection

Port from v1's sample_selection.py, simplified for v3's needs.
"""

from __future__ import annotations

from typing import Tuple, Literal
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


SelectionMethod = Literal["spxy", "duplex", "kennard-stone", "random"]


def kennard_stone(
    X: np.ndarray,
    n_samples: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Kennard-Stone algorithm for selecting representative samples.

    Selects samples that are maximally diverse in feature space.
    Starts by selecting two samples with maximum distance, then iteratively
    adds samples that are farthest from the already-selected set.

    Parameters
    ----------
    X : np.ndarray, shape (n_total_samples, n_features)
        Feature matrix (e.g., spectra).
    n_samples : int
        Number of samples to select.
    metric : str, default='euclidean'
        Distance metric to use.

    Returns
    -------
    selected_indices : np.ndarray, shape (n_samples,)
        Indices of selected samples in original dataset.
    """
    n_total = X.shape[0]

    if n_samples > n_total:
        raise ValueError(
            f"Cannot select {n_samples} samples from dataset with only {n_total} samples"
        )
    if n_samples < 2:
        raise ValueError("Must select at least 2 samples")

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

    # Iteratively add samples
    for _ in range(n_samples - 2):
        X_selected = X[selected_indices]
        X_remaining = X[remaining_indices]

        dist_to_selected = cdist(X_remaining, X_selected, metric=metric)
        min_distances = np.min(dist_to_selected, axis=1)

        max_min_idx = np.argmax(min_distances)
        selected_sample = remaining_indices[max_min_idx]

        selected_indices.append(selected_sample)
        remaining_indices.remove(selected_sample)

    return np.array(selected_indices, dtype=int)


def spxy(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Sample set Partitioning based on joint X-Y distance (SPXY).

    Extends Kennard-Stone by considering both feature space (X) and
    target space (Y) when selecting samples. This ensures selected samples
    are diverse in both spectral and reference value spaces.

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
    """
    n_total = X.shape[0]

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

    # Normalize X and y to [0, 1]
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    y_norm = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0) + 1e-10)

    # Compute pairwise distances in X and Y spaces
    dist_X = squareform(pdist(X_norm, metric=metric))
    dist_Y = squareform(pdist(y_norm, metric='euclidean'))

    # Combine distances (equal weighting)
    dist_XY = dist_X + dist_Y

    # Find initial pair with maximum combined distance
    max_dist_idx = np.argmax(dist_XY)
    i, j = np.unravel_index(max_dist_idx, dist_XY.shape)

    selected_indices = [i, j]
    remaining_indices = list(range(n_total))
    remaining_indices.remove(i)
    remaining_indices.remove(j)

    # Iteratively add samples
    for _ in range(n_samples - 2):
        min_distances = []
        for idx in remaining_indices:
            distances_to_selected = [dist_XY[idx, s] for s in selected_indices]
            min_distances.append(min(distances_to_selected))

        max_min_idx = np.argmax(min_distances)
        selected_sample = remaining_indices[max_min_idx]

        selected_indices.append(selected_sample)
        remaining_indices.remove(selected_sample)

    return np.array(selected_indices, dtype=int)


def duplex(
    X: np.ndarray,
    y: np.ndarray = None,
    n_cal: int = None,
    cal_ratio: float = 0.75,
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DUPLEX algorithm for splitting dataset into calibration and validation sets.

    Extends Kennard-Stone by alternately assigning selected samples
    to calibration and validation sets, ensuring both sets are representative.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, optional
        Target values (not used, kept for API compatibility).
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
    """
    n_total = X.shape[0]

    if n_cal is None:
        n_cal = int(n_total * cal_ratio)

    n_val = n_total - n_cal

    if n_cal < 1 or n_val < 1:
        raise ValueError(
            f"Invalid split: {n_cal} cal, {n_val} val. Adjust cal_ratio or n_cal."
        )

    # Use Kennard-Stone to select all samples in order
    all_selected = kennard_stone(X, n_samples=n_total, metric=metric)

    # Alternate assignment
    cal_indices = []
    val_indices = []

    for i, idx in enumerate(all_selected):
        if len(cal_indices) < n_cal and len(val_indices) < n_val:
            if i % 2 == 0:
                cal_indices.append(idx)
            else:
                val_indices.append(idx)
        elif len(cal_indices) < n_cal:
            cal_indices.append(idx)
        else:
            val_indices.append(idx)

    return np.array(cal_indices, dtype=int), np.array(val_indices, dtype=int)


def random_selection(
    n_total: int,
    n_samples: int,
    random_state: int = None
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
    """
    rng = np.random.RandomState(random_state)

    if n_samples > n_total:
        raise ValueError(
            f"Cannot select {n_samples} samples from {n_total} total samples"
        )

    return rng.choice(n_total, size=n_samples, replace=False)


def split_validation(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    method: str = 'spxy',
    random_state: int = 42,
    task_type: str = 'regression'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split dataset into calibration and validation sets.

    This is the main function for validation holdout in the Build tab.

    Note: For classification tasks, stratified random sampling is used by default
    to preserve class proportions, regardless of the method specified.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (spectra).
    y : np.ndarray, shape (n_samples,)
        Target values.
    val_ratio : float, default=0.2
        Fraction of samples to hold out for validation (0.1 to 0.25).
    method : str, default='spxy'
        Selection method: 'spxy' (recommended), 'duplex', 'kennard-stone', 'random'
    random_state : int, default=42
        Random seed for reproducibility (only used by 'random' method).

    Returns
    -------
    cal_indices : np.ndarray
        Indices of calibration (training) samples.
    val_indices : np.ndarray
        Indices of validation (holdout) samples.

    Examples
    --------
    >>> X = spectra  # shape (100, 500)
    >>> y = reference_values  # shape (100,)
    >>> cal_idx, val_idx = split_validation(X, y, val_ratio=0.2, method='spxy')
    >>> X_cal, y_cal = X[cal_idx], y[cal_idx]  # 80 samples for training
    >>> X_val, y_val = X[val_idx], y[val_idx]  # 20 samples for validation
    """
    from sklearn.model_selection import train_test_split

    n_total = X.shape[0]
    n_val = int(n_total * val_ratio)
    n_cal = n_total - n_val

    if n_val < 1:
        raise ValueError(f"val_ratio {val_ratio} results in 0 validation samples")
    if n_cal < 2:
        raise ValueError(f"val_ratio {val_ratio} leaves less than 2 calibration samples")

    all_indices = np.arange(n_total)

    # For classification, use stratified sampling to preserve class proportions
    if task_type == 'classification':
        if method == 'stratified' or method == 'random':
            # Stratified random split - preserves class proportions
            cal_indices, val_indices = train_test_split(
                all_indices, test_size=val_ratio, stratify=y, random_state=random_state
            )
            return np.array(cal_indices), np.array(val_indices)
        elif method == 'kennard-stone':
            # Kennard-Stone on X only (ignores class structure)
            val_indices = kennard_stone(X, n_samples=n_val)
            cal_indices = np.setdiff1d(all_indices, val_indices)
        else:
            # For other methods, fall back to stratified random (best for classification)
            cal_indices, val_indices = train_test_split(
                all_indices, test_size=val_ratio, stratify=y, random_state=random_state
            )
            return np.array(cal_indices), np.array(val_indices)
    else:
        # Regression methods
        if method == 'spxy':
            # Select validation samples using SPXY (diverse in X-Y space)
            val_indices = spxy(X, y, n_samples=n_val)
            cal_indices = np.setdiff1d(all_indices, val_indices)

        elif method == 'duplex':
            # DUPLEX alternates assignment
            cal_indices, val_indices = duplex(X, y, cal_ratio=(1 - val_ratio))

        elif method == 'kennard-stone':
            # Select validation samples using Kennard-Stone (diverse in X space)
            val_indices = kennard_stone(X, n_samples=n_val)
            cal_indices = np.setdiff1d(all_indices, val_indices)

        elif method == 'random':
            # Random selection
            val_indices = random_selection(n_total, n_val, random_state=random_state)
            cal_indices = np.setdiff1d(all_indices, val_indices)

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: 'spxy', 'duplex', 'kennard-stone', 'random'"
            )

    return cal_indices, val_indices
