"""
spectral_predict.wavelength_selection
=====================================

Wavelength/variable selection algorithms for spectral analysis and
calibration transfer enhancement.

This module implements several algorithms for selecting informative
wavelengths from spectral data:

- SPA (Successive Projections Algorithm): Fast, greedy selection
- CARS (Competitive Adaptive Reweighted Sampling): Monte Carlo-based
- VCPA-IRIV (Variable Combination Population Analysis): Advanced iterative method

These algorithms are particularly useful for:
1. Reducing model complexity and overfitting
2. Improving prediction performance
3. Enhancing calibration transfer (especially NS-PFCE)
4. Identifying key spectral regions
"""

from __future__ import annotations

from typing import Dict, Tuple, Literal
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


SelectionMethod = Literal["spa", "cars", "vcpa-iriv"]


def spa(
    X: np.ndarray,
    y: np.ndarray,
    n_vars: int = 20,
    max_iterations: int = 10000
) -> Dict:
    """
    Successive Projections Algorithm (SPA) for wavelength selection.

    SPA is a forward selection method that uses projection operations
    to minimize collinearity among selected variables. It's fast and
    effective for removing redundant wavelengths.

    Algorithm:
    1. Start with one wavelength (maximum variation or user-specified)
    2. For each iteration:
       - Compute projections of remaining wavelengths onto selected space
       - Select wavelength with maximum projection norm
       - Add to selected set
    3. Continue until n_vars wavelengths are selected

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)
        Spectral data matrix.
    y : np.ndarray, shape (n_samples,)
        Target values (can be None, SPA works on X only).
    n_vars : int, default=20
        Number of wavelengths to select.
    max_iterations : int, default=10000
        Maximum iterations to prevent infinite loops.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'selected_indices': np.ndarray, selected wavelength indices
        - 'selected_order': list, order in which wavelengths were selected
        - 'projection_norms': np.ndarray, projection norms for each selected var
        - 'n_selected': int, number of wavelengths selected

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.wavelength_selection import spa
    >>>
    >>> X = np.random.randn(100, 200)
    >>> y = np.random.randn(100)
    >>>
    >>> result = spa(X, y, n_vars=30)
    >>> print(f"Selected {result['n_selected']} wavelengths")
    >>> print(f"Indices: {result['selected_indices'][:10]}...")  # First 10

    References
    ----------
    .. [1] Araújo, M. C. U., et al. (2001). The successive projections
           algorithm for variable selection in spectroscopic multicomponent
           analysis. Chemometrics and Intelligent Laboratory Systems,
           57(2), 65-73.

    Notes
    -----
    - SPA is fast (O(n * p^2) worst case)
    - Focuses on reducing collinearity, not prediction performance
    - Good for preprocessing before building models
    - Does not require y values (unsupervised selection)
    """
    n_samples, n_wavelengths = X.shape

    # Validate inputs
    if n_vars > n_wavelengths:
        raise ValueError(f"Cannot select {n_vars} from {n_wavelengths} wavelengths")
    if n_vars < 1:
        raise ValueError("Must select at least 1 wavelength")

    # Initialize with wavelength of maximum variance
    variances = np.var(X, axis=0)
    initial_idx = np.argmax(variances)

    selected_indices = [initial_idx]
    projection_norms = [np.sqrt(variances[initial_idx])]
    remaining_indices = list(range(n_wavelengths))
    remaining_indices.remove(initial_idx)

    # Iteratively select wavelengths
    for iteration in range(min(n_vars - 1, max_iterations)):
        if len(remaining_indices) == 0:
            break

        # Get selected subspace
        X_selected = X[:, selected_indices]

        # Compute projection of remaining wavelengths onto selected subspace
        # Projection: P = X_selected @ (X_selected^T X_selected)^-1 @ X_selected^T
        # For efficiency, use QR decomposition
        Q, R = np.linalg.qr(X_selected)

        max_norm = -np.inf
        best_idx = None

        for idx in remaining_indices:
            x_col = X[:, idx:idx+1]  # Column vector

            # Project onto orthogonal complement of selected space
            # projection_orth = x_col - Q @ (Q^T @ x_col)
            projection_orth = x_col - Q @ (Q.T @ x_col)
            norm = np.linalg.norm(projection_orth)

            if norm > max_norm:
                max_norm = norm
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        projection_norms.append(max_norm)
        remaining_indices.remove(best_idx)

    result = {
        'selected_indices': np.array(selected_indices),
        'selected_order': selected_indices,
        'projection_norms': np.array(projection_norms),
        'n_selected': len(selected_indices)
    }

    return result


def cars(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 50,
    pls_components: int = 5,
    cv_folds: int = 5,
    monte_carlo_samples: int = 80,
    random_state: int | None = None
) -> Dict:
    """
    Competitive Adaptive Reweighted Sampling (CARS) for wavelength selection.

    CARS is a Monte Carlo-based method that uses an adaptive reweighted
    sampling (ARS) strategy combined with exponential decay to select
    optimal wavelengths. It balances exploration and exploitation.

    Algorithm:
    1. Initialize all wavelengths with equal weights
    2. For each Monte Carlo iteration:
       - Sample wavelengths based on current weights
       - Build PLS model and evaluate via cross-validation
       - Update weights based on PLS regression coefficients
       - Apply exponential decay to force elimination
    3. Select wavelengths from iteration with lowest RMSECV

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)
        Spectral data matrix.
    y : np.ndarray, shape (n_samples,)
        Target values (required for CARS).
    n_iterations : int, default=50
        Number of Monte Carlo sampling iterations.
    pls_components : int, default=5
        Number of PLS components to use in evaluation.
    cv_folds : int, default=5
        Number of cross-validation folds.
    monte_carlo_samples : int, default=80
        Percentage of wavelengths to sample in each iteration (as integer).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'selected_indices': np.ndarray, optimal wavelength indices
        - 'best_iteration': int, iteration with lowest RMSECV
        - 'rmsecv_history': np.ndarray, RMSECV for each iteration
        - 'n_selected_history': list, number of vars at each iteration
        - 'weights_history': list, weight vectors over iterations

    Examples
    --------
    >>> from spectral_predict.wavelength_selection import cars
    >>>
    >>> X = np.random.randn(80, 150)
    >>> y = X[:, 50] + X[:, 100] + 0.1 * np.random.randn(80)
    >>>
    >>> result = cars(X, y, n_iterations=40)
    >>> print(f"Selected {len(result['selected_indices'])} wavelengths")
    >>> print(f"Best iteration: {result['best_iteration']}")
    >>> print(f"Best RMSECV: {min(result['rmsecv_history']):.4f}")

    References
    ----------
    .. [1] Li, H. D., et al. (2009). Key wavelengths screening using
           competitive adaptive reweighted sampling method for multivariate
           calibration. Analytica Chimica Acta, 648(1), 77-84.

    Notes
    -----
    - CARS balances variable selection with prediction performance
    - Computationally more expensive than SPA (Monte Carlo iterations)
    - Often produces very compact variable sets
    - Requires target values (supervised selection)
    - Performance depends on good PLS component selection
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_wavelengths = X.shape

    # Validate inputs
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if pls_components > min(n_samples, n_wavelengths):
        raise ValueError(f"pls_components ({pls_components}) too large")

    # Initialize weights
    weights = np.ones(n_wavelengths)

    # Storage for history
    rmsecv_history = []
    n_selected_history = []
    weights_history = []
    selected_vars_history = []

    # Monte Carlo iterations
    for iteration in range(n_iterations):
        # Exponential decay function for forcing removal
        # r(k) = a * exp(-k/b) where k is iteration
        r = 0.8 * np.exp(-2 * iteration / n_iterations)

        # Number of wavelengths to sample in this iteration
        n_sample = max(int(n_wavelengths * (monte_carlo_samples / 100) * r), pls_components + 1)
        n_sample = min(n_sample, n_wavelengths)

        # Sample wavelengths based on current weights
        # Higher weight = higher probability of selection
        probabilities = weights / weights.sum()
        selected_vars = np.random.choice(
            n_wavelengths,
            size=n_sample,
            replace=False,
            p=probabilities
        )
        selected_vars = np.sort(selected_vars)

        X_subset = X[:, selected_vars]

        # Build PLS model and evaluate
        try:
            pls = PLSRegression(n_components=min(pls_components, n_sample-1, X_subset.shape[0]-1))

            # Cross-validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_errors = []

            for train_idx, val_idx in kf.split(X_subset):
                X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                pls.fit(X_train, y_train)
                y_pred = pls.predict(X_val)
                mse = np.mean((y_val - y_pred.ravel()) ** 2)
                cv_errors.append(mse)

            rmsecv = np.sqrt(np.mean(cv_errors))

            # Fit on full subset to get coefficients
            pls.fit(X_subset, y)

            # Update weights based on PLS coefficients
            # Larger absolute coefficient = more important
            coef = pls.coef_.ravel()
            new_weights = np.abs(coef)

            # Update only the sampled variables' weights
            temp_weights = weights.copy()
            temp_weights[selected_vars] = new_weights
            weights = temp_weights

            # Normalize weights
            weights = weights / (weights.sum() + 1e-10)

        except Exception as e:
            # If PLS fails (e.g., singular matrix), skip this iteration
            rmsecv = np.inf

        # Record history
        rmsecv_history.append(rmsecv)
        n_selected_history.append(n_sample)
        weights_history.append(weights.copy())
        selected_vars_history.append(selected_vars)

    # Find iteration with lowest RMSECV
    rmsecv_history = np.array(rmsecv_history)
    valid_iterations = ~np.isinf(rmsecv_history)

    if not np.any(valid_iterations):
        raise RuntimeError("CARS failed: no valid iterations")

    best_iteration = np.argmin(rmsecv_history[valid_iterations])
    best_iteration_idx = np.where(valid_iterations)[0][best_iteration]

    selected_indices = selected_vars_history[best_iteration_idx]

    result = {
        'selected_indices': selected_indices,
        'best_iteration': best_iteration_idx,
        'rmsecv_history': rmsecv_history,
        'n_selected_history': n_selected_history,
        'weights_history': weights_history,
        'n_selected': len(selected_indices)
    }

    return result


def vcpa_iriv(
    X: np.ndarray,
    y: np.ndarray,
    n_outer_iterations: int = 10,
    n_inner_iterations: int = 50,
    pls_components: int = 5,
    cv_folds: int = 5,
    binary_matrix_samples: int = 100,
    importance_threshold: float = 0.5,
    random_state: int | None = None
) -> Dict:
    """
    Variable Combination Population Analysis - Iteratively Retains
    Informative Variables (VCPA-IRIV).

    VCPA-IRIV is an advanced iterative method that combines binary matrix
    sampling with variable importance analysis. It's particularly effective
    for NS-PFCE calibration transfer enhancement.

    Algorithm (simplified):
    1. Generate binary matrix (BM) for variable combinations
    2. For each combination, build PLS model and evaluate
    3. Compute variable importance scores based on model performance
    4. Remove uninformative variables (IRIV step)
    5. Repeat until convergence or max iterations

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)
        Spectral data matrix.
    y : np.ndarray, shape (n_samples,)
        Target values (required).
    n_outer_iterations : int, default=10
        Number of IRIV outer iterations (variable elimination rounds).
    n_inner_iterations : int, default=50
        Number of BM sampling iterations per outer iteration.
    pls_components : int, default=5
        Number of PLS components for model building.
    cv_folds : int, default=5
        Cross-validation folds.
    binary_matrix_samples : int, default=100
        Number of binary combinations to generate per iteration.
    importance_threshold : float, default=0.5
        Threshold for removing low-importance variables (0-1).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'selected_indices': np.ndarray, final selected wavelength indices
        - 'importance_scores': np.ndarray, final importance scores
        - 'convergence_history': list, RMSECV at each outer iteration
        - 'n_vars_history': list, number of variables at each iteration
        - 'final_rmsecv': float, cross-validation error with selected vars

    Examples
    --------
    >>> from spectral_predict.wavelength_selection import vcpa_iriv
    >>>
    >>> X = np.random.randn(100, 200)
    >>> y = X[:, [30, 80, 150]].sum(axis=1) + 0.1 * np.random.randn(100)
    >>>
    >>> result = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=30)
    >>> print(f"Selected {len(result['selected_indices'])} wavelengths")
    >>> print(f"Final RMSECV: {result['final_rmsecv']:.4f}")
    >>> # Check if true informative vars (30, 80, 150) were selected
    >>> print(f"Key vars in selection: {np.isin([30, 80, 150], result['selected_indices'])}")

    References
    ----------
    .. [1] Yun, Y. H., et al. (2015). An efficient method of wavelength
           interval selection based on random frog for multivariate spectral
           calibration. Spectrochimica Acta Part A, 148, 375-381.

    Notes
    -----
    - VCPA-IRIV is the most sophisticated method in this module
    - Computationally expensive (many PLS models built)
    - Often finds very compact, informative variable sets
    - Well-suited for high-dimensional data (p >> n)
    - Requires careful parameter tuning for best results
    - Used in NS-PFCE for optimal calibration transfer
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_wavelengths = X.shape

    # Validate inputs
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    # Initialize: all variables are candidates
    active_indices = np.arange(n_wavelengths)
    convergence_history = []
    n_vars_history = []

    # Outer loop: Iterative variable removal (IRIV)
    for outer_iter in range(n_outer_iterations):
        n_active = len(active_indices)
        n_vars_history.append(n_active)

        if n_active <= pls_components:
            print(f"  VCPA-IRIV: Stopped at iteration {outer_iter} (too few variables)")
            break

        # Initialize importance scores for active variables
        importance_scores = np.zeros(n_active)
        performance_with_var = []

        # Inner loop: Binary matrix sampling (VCPA)
        for inner_iter in range(n_inner_iterations):
            # Generate binary vector (random subset of variables)
            # Probability of including each variable decreases over iterations
            # to encourage exploration early, exploitation later
            inclusion_prob = 0.9 * (1 - outer_iter / n_outer_iterations) + 0.2

            binary_vector = np.random.rand(n_active) < inclusion_prob
            n_selected = np.sum(binary_vector)

            # Need at least pls_components+1 variables
            if n_selected <= pls_components:
                continue

            selected_vars = active_indices[binary_vector]
            X_subset = X[:, selected_vars]

            # Build PLS model and evaluate
            try:
                pls = PLSRegression(n_components=min(pls_components, n_selected-1))

                # Cross-validation
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_errors = []

                for train_idx, val_idx in kf.split(X_subset):
                    X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    pls.fit(X_train, y_train)
                    y_pred = pls.predict(X_val)
                    mse = np.mean((y_val - y_pred.ravel()) ** 2)
                    cv_errors.append(mse)

                rmsecv = np.sqrt(np.mean(cv_errors))

                # If this combination performed well, increase importance of included vars
                if rmsecv < np.inf:
                    # Lower RMSECV = higher importance
                    weight = 1.0 / (rmsecv + 1e-10)
                    importance_scores[binary_vector] += weight

            except Exception:
                # Skip if PLS fails
                continue

        # Normalize importance scores
        if importance_scores.sum() > 0:
            importance_scores = importance_scores / importance_scores.sum()

        # Compute median RMSECV for this outer iteration (approximate)
        current_rmsecv = 1.0 / (importance_scores.mean() + 1e-10) if importance_scores.mean() > 0 else np.inf
        convergence_history.append(current_rmsecv)

        # IRIV step: Remove variables below importance threshold
        threshold_value = importance_threshold * importance_scores.max()
        keep_mask = importance_scores >= threshold_value

        if np.sum(keep_mask) <= pls_components:
            # Don't remove more variables
            print(f"  VCPA-IRIV: Stopped removal at iteration {outer_iter} (threshold too high)")
            break

        active_indices = active_indices[keep_mask]

        # Check convergence: if no variables removed, stop
        if np.sum(keep_mask) == n_active:
            print(f"  VCPA-IRIV: Converged at iteration {outer_iter} (no variables removed)")
            break

    # Final evaluation with selected variables
    X_final = X[:, active_indices]

    try:
        pls_final = PLSRegression(n_components=min(pls_components, len(active_indices)-1))

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_errors = []

        for train_idx, val_idx in kf.split(X_final):
            X_train, X_val = X_final[train_idx], X_final[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pls_final.fit(X_train, y_train)
            y_pred = pls_final.predict(X_val)
            mse = np.mean((y_val - y_pred.ravel()) ** 2)
            cv_errors.append(mse)

        final_rmsecv = np.sqrt(np.mean(cv_errors))

    except Exception as e:
        final_rmsecv = np.inf

    result = {
        'selected_indices': active_indices,
        'importance_scores': importance_scores if len(importance_scores) == len(active_indices) else np.ones(len(active_indices)),
        'convergence_history': convergence_history,
        'n_vars_history': n_vars_history,
        'final_rmsecv': final_rmsecv,
        'n_selected': len(active_indices)
    }

    return result


def compare_selection_methods(
    X: np.ndarray,
    y: np.ndarray,
    methods: list[SelectionMethod] | None = None,
    target_n_vars: int = 30,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Compare different wavelength selection methods.

    Evaluates SPA, CARS, and VCPA-IRIV on the same dataset and compares
    their performance in terms of:
    - Number of variables selected
    - Cross-validation performance
    - Computation time

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)
        Spectral data.
    y : np.ndarray, shape (n_samples,)
        Target values.
    methods : list of str, optional
        Methods to compare. Default: all three methods.
    target_n_vars : int, default=30
        Target number of variables for SPA (others auto-select).
    random_state : int, default=42
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with method names as keys, results as values.

    Examples
    --------
    >>> from spectral_predict.wavelength_selection import compare_selection_methods
    >>>
    >>> X = np.random.randn(80, 200)
    >>> y = np.random.randn(80)
    >>>
    >>> results = compare_selection_methods(X, y, target_n_vars=25)
    >>> for method, res in results.items():
    >>>     print(f"{method}: {res['n_selected']} vars, RMSECV={res.get('final_rmsecv', 'N/A')}")
    """
    import time

    if methods is None:
        methods = ['spa', 'cars', 'vcpa-iriv']

    results = {}

    for method in methods:
        print(f"\nTesting {method.upper()}...")
        start_time = time.time()

        try:
            if method == 'spa':
                result = spa(X, y, n_vars=target_n_vars)
            elif method == 'cars':
                result = cars(X, y, n_iterations=40, random_state=random_state)
            elif method == 'vcpa-iriv':
                result = vcpa_iriv(X, y, n_outer_iterations=8, n_inner_iterations=30, random_state=random_state)
            else:
                print(f"  Unknown method: {method}")
                continue

            elapsed = time.time() - start_time
            result['computation_time'] = elapsed
            result['success'] = True

            print(f"  ✓ Completed in {elapsed:.2f}s")
            print(f"  Selected {result['n_selected']} wavelengths")
            if 'final_rmsecv' in result:
                print(f"  Final RMSECV: {result['final_rmsecv']:.4f}")

            results[method] = result

        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results[method] = {'success': False, 'error': str(e)}

    return results


if __name__ == "__main__":
    print("Wavelength Selection Module")
    print("=" * 60)

    # Quick demonstration
    np.random.seed(42)

    n_samples, n_wavelengths = 80, 150
    X = np.random.randn(n_samples, n_wavelengths)

    # Create y with known important wavelengths
    important_wavelengths = [30, 75, 120]
    y = X[:, important_wavelengths].sum(axis=1) + 0.1 * np.random.randn(n_samples)

    print(f"\nGenerated data: {n_samples} samples, {n_wavelengths} wavelengths")
    print(f"True important wavelengths: {important_wavelengths}")

    # Test SPA
    print("\n1. Testing SPA...")
    spa_result = spa(X, y, n_vars=20)
    print(f"   Selected {spa_result['n_selected']} wavelengths")
    overlap_spa = np.isin(important_wavelengths, spa_result['selected_indices'])
    print(f"   Found {overlap_spa.sum()}/{len(important_wavelengths)} true important vars")

    # Test CARS
    print("\n2. Testing CARS...")
    cars_result = cars(X, y, n_iterations=30, random_state=42)
    print(f"   Selected {cars_result['n_selected']} wavelengths")
    print(f"   Best iteration: {cars_result['best_iteration']}")
    overlap_cars = np.isin(important_wavelengths, cars_result['selected_indices'])
    print(f"   Found {overlap_cars.sum()}/{len(important_wavelengths)} true important vars")

    # Test VCPA-IRIV
    print("\n3. Testing VCPA-IRIV...")
    vcpa_result = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=20, random_state=42)
    print(f"   Selected {vcpa_result['n_selected']} wavelengths")
    print(f"   Final RMSECV: {vcpa_result['final_rmsecv']:.4f}")
    overlap_vcpa = np.isin(important_wavelengths, vcpa_result['selected_indices'])
    print(f"   Found {overlap_vcpa.sum()}/{len(important_wavelengths)} true important vars")

    print("\n" + "=" * 60)
    print("Wavelength selection module loaded successfully!")
