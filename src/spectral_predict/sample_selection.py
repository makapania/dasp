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
   (:func:`split_calibration_holdout`)
3. Optimal sample selection for experimental design

Direction matters. KS and SPXY pick the *representative* samples, including the
extremes. For a calibration / holdout split those samples are the calibration
set and the remainder is the holdout; picking the holdout with KS instead would
put the extremes in validation and force the model to extrapolate.
"""

from __future__ import annotations

from typing import Tuple, Dict, Literal
import numpy as np
from scipy.spatial.distance import pdist, squareform


SelectionMethod = Literal["kennard-stone", "duplex", "spxy", "random"]


def _farthest_pair(D: np.ndarray) -> tuple[int, int]:
    """Positions of the two most distant rows of a square distance matrix."""
    i, j = np.unravel_index(np.argmax(D), D.shape)
    i, j = int(i), int(j)
    if i == j:  # every distance is zero (identical rows): any pair is farthest
        i, j = 0, 1
    return i, j


def _max_min_order(D: np.ndarray, n_select: int) -> list[int]:
    """Kennard-Stone max-min ordering on a precomputed square distance matrix.

    Starts from the farthest pair, then repeatedly adds the row whose distance to
    its nearest already-chosen row is largest. Ties go to the lowest row index.
    """
    i, j = _farthest_pair(D)
    selected = [i, j]
    # Distance from every row to its nearest selected row; chosen rows are -inf
    # so they can never be picked again.
    min_d = np.minimum(D[i], D[j]).astype(float)
    min_d[selected] = -np.inf
    while len(selected) < n_select:
        k = int(np.argmax(min_d))
        selected.append(k)
        min_d = np.minimum(min_d, D[k])
        min_d[k] = -np.inf
    return selected


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

    The selected samples are the *representative* ones: they include the
    extremes and cover the boundary of the data. Used for a calibration /
    holdout split, they are the **calibration** set and the remaining samples
    are the holdout (see :func:`split_calibration_holdout`). Used to choose
    calibration-transfer standards, they are the standards.

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
        Distance metric to use (see scipy.spatial.distance.pdist).

    Returns
    -------
    selected_indices : np.ndarray, shape (n_samples,)
        Indices of selected samples in original dataset, in selection order.

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
    n_total = X.shape[0]

    # Validate inputs
    if n_samples > n_total:
        raise ValueError(
            f"Cannot select {n_samples} samples from dataset with only {n_total} samples"
        )
    if n_samples < 2:
        raise ValueError("Must select at least 2 samples")

    # pdist's condensed vector is in upper-triangular row-major order, so work on
    # the square matrix rather than inverting the condensed index by hand (R085).
    distance_matrix = squareform(pdist(X, metric=metric))
    return np.array(_max_min_order(distance_matrix, n_samples), dtype=int)


def duplex(
    X: np.ndarray,
    y: np.ndarray | None = None,
    n_cal: int | None = None,
    cal_ratio: float = 0.75,
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DUPLEX algorithm for splitting dataset into calibration and validation sets.

    DUPLEX (Snee 1977) grows two sets at once so that *both* span the data:

    1. The two samples farthest apart go to the calibration set.
    2. Of the samples left, the two farthest apart go to the validation set.
    3. Then, alternately, the remaining sample farthest from the calibration
       set (largest distance to its nearest calibration sample) joins the
       calibration set, and the remaining sample farthest from the validation
       set joins the validation set.
    4. When one set reaches its size, every remaining sample joins the other.

    Each set runs its own max-min selection against its own members; this is
    not the same as alternating a single Kennard-Stone order.

    A set of size one cannot start from a pair. A one-sample validation set
    takes the sample farthest from the calibration seed pair (largest distance
    to its nearest calibration sample); a one-sample calibration set takes the
    lower-index end of the farthest pair. Ties go to the lowest row index.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,), optional
        Accepted for API symmetry with :func:`spxy`; DUPLEX uses X only.
    n_cal : int, optional
        Number of calibration samples. If None, uses cal_ratio.
    cal_ratio : float, default=0.75
        Ratio of calibration samples (between 0 and 1).
    metric : str, default='euclidean'
        Distance metric (see scipy.spatial.distance.pdist).

    Returns
    -------
    cal_indices : np.ndarray
        Indices of calibration samples, in the order they were assigned.
    val_indices : np.ndarray
        Indices of validation samples, in the order they were assigned.

    Examples
    --------
    >>> from spectral_predict.sample_selection import duplex
    >>>
    >>> X = np.random.randn(100, 50)
    >>> cal_idx, val_idx = duplex(X, cal_ratio=0.75)

    References
    ----------
    .. [1] Snee, R. D. (1977). Validation of regression models: methods and
           examples. Technometrics, 19(4), 415-428.

    Notes
    -----
    - Both sets span the feature space, so the validation set is not forced to
      extrapolate, but neither is it confined to the interior.
    - Deterministic (given same input); ties go to the lowest row index.
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

    D = squareform(pdist(X, metric=metric))
    available = np.ones(n_total, dtype=bool)

    def seed(quota: int, opposite: list[int]) -> list[int]:
        rows = np.flatnonzero(available)
        if quota == 1 and opposite:
            # A one-sample set has no pair: take the available sample farthest
            # from the other set (largest distance to its nearest member).
            near = D[np.ix_(rows, opposite)].min(axis=1)
            return [int(rows[int(np.argmax(near))])]
        if len(rows) == 1:
            return [int(rows[0])]
        a, b = _farthest_pair(D[np.ix_(rows, rows)])
        return [int(rows[a]), int(rows[b])][:quota]

    def take(members: list[int], k: int) -> None:
        members.append(k)
        available[k] = False

    cal: list[int] = []
    val: list[int] = []
    for k in seed(n_cal, val):
        take(cal, k)
    for k in seed(n_val, cal):
        take(val, k)

    # Distance from every sample to its nearest member of each set.
    near_cal = D[cal].min(axis=0)
    near_val = D[val].min(axis=0)

    def farthest_from(near: np.ndarray) -> int:
        return int(np.argmax(np.where(available, near, -np.inf)))

    while available.any():
        if len(cal) < n_cal:
            k = farthest_from(near_cal)
            take(cal, k)
            near_cal = np.minimum(near_cal, D[k])
        if available.any() and len(val) < n_val:
            k = farthest_from(near_val)
            take(val, k)
            near_val = np.minimum(near_val, D[k])

    return np.array(cal, dtype=int), np.array(val, dtype=int)


def spxy_distance_matrix(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Joint X-y distance matrix used by SPXY (Galvão et al. 2005).

    ``d_xy(p, q) = d_x(p, q) / max d_x + d_y(p, q) / max d_y``, where ``d_x`` is
    the distance between spectra and ``d_y`` the distance between reference
    values. Each matrix is divided by its own maximum so that X and y carry
    equal weight; the spectra themselves are not rescaled column by column.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
        Target values (numeric).
    metric : str, default='euclidean'
        Distance metric for X space. y always uses Euclidean distance.

    Returns
    -------
    np.ndarray, shape (n_samples, n_samples)
        Symmetric joint distance matrix with a zero diagonal.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("SPXY needs finite y values; drop samples with missing y first")

    dist_X = squareform(pdist(X, metric=metric))
    dist_Y = squareform(pdist(y, metric='euclidean'))
    max_X = dist_X.max()
    max_Y = dist_Y.max()
    if max_X > 0:
        dist_X = dist_X / max_X
    if max_Y > 0:
        dist_Y = dist_Y / max_Y
    return dist_X + dist_Y


def spxy(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Sample set Partitioning based on joint X-Y distance (SPXY).

    SPXY extends Kennard-Stone by considering both feature space (X) and
    target space (Y) when selecting samples. The selected samples are the
    representative ones and, for a calibration / holdout split, form the
    **calibration** set (see :func:`split_calibration_holdout`).

    Algorithm (Galvão et al. 2005):
    1. Compute the X distance matrix and the y distance matrix
    2. Divide each by its maximum and add them (:func:`spxy_distance_matrix`)
    3. Apply the KS max-min selection to the combined distance

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
        Indices of selected samples, in selection order.

    Examples
    --------
    >>> from spectral_predict.sample_selection import spxy
    >>>
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> indices = spxy(X, y, n_samples=20)

    References
    ----------
    .. [1] Galvão, R. K. H., Araujo, M. C. U., José, G. E., Pontes, M. J. C.,
           Silva, E. C., & Saldanha, T. C. B. (2005). A method for calibration
           and validation subset partitioning. Talanta, 67(4), 736-740.
    """
    n_total = X.shape[0]
    y_arr = np.asarray(y)
    if y_arr.shape[0] != n_total:
        raise ValueError(
            f"X and y must have same number of samples: {X.shape[0]} vs {y_arr.shape[0]}"
        )
    if n_samples > n_total:
        raise ValueError(
            f"Cannot select {n_samples} samples from dataset with only {n_total} samples"
        )
    if n_samples < 2:
        raise ValueError("Must select at least 2 samples")

    D = spxy_distance_matrix(X, y_arr, metric=metric)
    return np.array(_max_min_order(D, n_samples), dtype=int)


HoldoutMethod = Literal["kennard-stone", "spxy", "duplex"]


def split_calibration_holdout(
    X: np.ndarray,
    n_holdout: int,
    method: HoldoutMethod = "kennard-stone",
    y: np.ndarray | None = None,
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split samples into a calibration set and a fixed external holdout.

    Kennard-Stone and SPXY choose the **calibration** set: the
    ``n_total - n_holdout`` most representative samples, which include the
    extremes. The samples left over are the holdout, so validation samples lie
    inside the calibration range and the model is not asked to extrapolate.
    DUPLEX grows both sets at once so each spans the data (:func:`duplex`).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Spectra (or any feature matrix, e.g. preprocessed spectra or PCA scores).
    n_holdout : int
        Number of samples to hold out. At least 1, and at least 2 must remain
        for calibration.
    method : {"kennard-stone", "spxy", "duplex"}, default="kennard-stone"
        Selection algorithm.
    y : np.ndarray, optional
        Numeric target values; required for ``"spxy"``.
    metric : str, default='euclidean'
        Distance metric for X.

    Returns
    -------
    cal_indices : np.ndarray
        Positions of the calibration samples, sorted ascending.
    holdout_indices : np.ndarray
        Positions of the holdout samples, sorted ascending.
    """
    X = np.asarray(X, dtype=float)
    n_total = X.shape[0]
    n_holdout = int(n_holdout)
    n_cal = n_total - n_holdout
    if n_holdout < 1 or n_cal < 2:
        raise ValueError(
            f"Cannot hold out {n_holdout} of {n_total} samples: need at least 1 holdout "
            "and 2 calibration samples"
        )

    if method == "kennard-stone":
        cal = kennard_stone(X, n_samples=n_cal, metric=metric)
    elif method == "spxy":
        if y is None:
            raise ValueError("SPXY needs y values")
        cal = spxy(X, y, n_samples=n_cal, metric=metric)
    elif method == "duplex":
        cal, _ = duplex(X, n_cal=n_cal, metric=metric)
    else:
        raise ValueError(f"Unknown holdout selection method: {method!r}")

    cal = np.sort(np.asarray(cal, dtype=int))
    holdout = np.setdiff1d(np.arange(n_total), cal)
    return cal, holdout


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
