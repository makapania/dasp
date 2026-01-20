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
            pls = PLSRegression(n_components=min(pls_components, n_sample-1, X_subset.shape[0]-1), scale=False)

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
    model_type: str | None = None,
    random_state: int | None = None
) -> Dict:
    """
    Variable Combination Population Analysis - Iteratively Retains
    Informative Variables (VCPA-IRIV).

    True VCPA-IRIV algorithm based on Yun et al. (2014). Uses statistical
    include/exclude comparison with Mann-Whitney U test to classify variables
    into four categories: strongly informative, weakly informative,
    uninformative, and interfering.

    Algorithm:
    1. Generate binary matrix (BM) for variable combinations
    2. For each variable, compare RMSECV when included vs excluded
    3. Apply Mann-Whitney U test to determine significance
    4. Classify variables into 4 categories based on test results
    5. Remove uninformative and interfering variables
    6. Repeat until convergence or max iterations

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)
        Spectral data matrix.
    y : np.ndarray, shape (n_samples,)
        Target values (required).
    n_outer_iterations : int, default=10
        Maximum number of IRIV outer iterations (variable elimination rounds).
    n_inner_iterations : int, default=50
        Number of BM sampling iterations per outer iteration (deprecated,
        binary_matrix_samples is used instead).
    pls_components : int, default=5
        Number of PLS components for model building (used when model_type is PLS).
    cv_folds : int, default=5
        Cross-validation folds.
    binary_matrix_samples : int, default=100
        Number of binary combinations to generate per iteration.
    importance_threshold : float, default=0.5
        Deprecated. Statistical test (p < 0.05) is used instead.
    model_type : str, optional
        Model type to use for variable evaluation. Options:
        - None or 'PLS': Use PLS regression (default, recommended for spectroscopy)
        - 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost': Use LightGBM
          (faster tree-based evaluation for tree model final models)
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'selected_indices': np.ndarray, final selected wavelength indices
        - 'variable_categories': dict, classification of each variable
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
    >>> # Default: PLS-based selection
    >>> result = vcpa_iriv(X, y, n_outer_iterations=5)
    >>> print(f"Selected {len(result['selected_indices'])} wavelengths")
    >>>
    >>> # For tree-based final models: LightGBM-based selection
    >>> result = vcpa_iriv(X, y, model_type='RandomForest')

    References
    ----------
    .. [1] Yun, Y. H., et al. (2014). A strategy that iteratively retains
           informative variables for selecting optimal variable subset in
           multivariate calibration. Analytica Chimica Acta, 807, 36-43.

    Notes
    -----
    - Uses Mann-Whitney U test for statistical significance (p < 0.05)
    - Variables classified as: strong, weak, uninformative, interfering
    - Strongly/weakly informative variables are retained
    - Uninformative/interfering variables are removed each iteration
    - PLS is recommended for spectroscopy (handles collinearity)
    - Tree-based option useful when final model is RF/XGBoost/LightGBM
    """
    from scipy.stats import mannwhitneyu

    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_wavelengths = X.shape

    # Validate inputs
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    # Determine which model to use for evaluation
    TREE_MODELS = {'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'}
    use_tree_model = model_type in TREE_MODELS if model_type else False

    # Initialize: all variables are candidates
    active_indices = np.arange(n_wavelengths)
    convergence_history = []
    n_vars_history = []
    variable_categories = {}  # Track final category of each variable

    # Helper function to compute RMSECV
    def compute_rmsecv(X_subset: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-validated RMSECV for a variable subset."""
        try:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_errors = []

            if use_tree_model:
                # Use LightGBM for tree-based models (faster than RF/XGBoost for CV)
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    n_estimators=50,
                    max_depth=5,
                    verbosity=-1,
                    random_state=random_state
                )
            else:
                # Use PLS (default, recommended for spectroscopy)
                n_comp = min(pls_components, X_subset.shape[1] - 1, X_subset.shape[0] - 1)
                if n_comp < 1:
                    return np.inf
                model = PLSRegression(n_components=n_comp, scale=False)

            for train_idx, val_idx in kf.split(X_subset):
                X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mse = np.mean((y_val - y_pred.ravel()) ** 2)
                cv_errors.append(mse)

            return np.sqrt(np.mean(cv_errors))
        except Exception:
            return np.inf

    # Outer loop: Iterative variable classification and removal
    for outer_iter in range(n_outer_iterations):
        n_active = len(active_indices)
        n_vars_history.append(n_active)

        # Minimum variables: pls_components+1 for PLS, 3 for tree models
        min_vars = pls_components + 1 if not use_tree_model else 3
        if n_active <= min_vars:
            print(f"  VCPA-IRIV: Stopped at iteration {outer_iter} (too few variables: {n_active})")
            break

        # Adaptive number of binary matrix rows based on variable count
        if n_active >= 500:
            n_rows = 200
        elif n_active >= 100:
            n_rows = 150
        elif n_active >= 50:
            n_rows = 100
        else:
            n_rows = max(50, n_active * 2)

        # Override with user parameter if specified
        n_rows = max(n_rows, binary_matrix_samples)

        # Generate binary matrix: each row is a random subset of variables
        # ~50% inclusion probability ensures each variable is included/excluded enough times
        binary_matrix = np.random.rand(n_rows, n_active) < 0.5

        # Ensure each row has enough variables for PLS
        for i in range(n_rows):
            while np.sum(binary_matrix[i]) <= pls_components:
                # Add random variables until we have enough
                zero_indices = np.where(~binary_matrix[i])[0]
                if len(zero_indices) == 0:
                    break
                add_idx = np.random.choice(zero_indices)
                binary_matrix[i, add_idx] = True

        # Collect RMSECV for include/exclude scenarios
        rmsecv_include = [[] for _ in range(n_active)]  # Errors when var is included
        rmsecv_exclude = [[] for _ in range(n_active)]  # Errors when var is excluded

        successful_samples = 0
        for row_idx, row in enumerate(binary_matrix):
            selected_local = np.where(row)[0]
            if len(selected_local) <= pls_components:
                continue

            selected_vars = active_indices[selected_local]
            X_subset = X[:, selected_vars]
            rmsecv = compute_rmsecv(X_subset, y)

            if np.isinf(rmsecv):
                continue

            successful_samples += 1

            # Record RMSECV for each variable based on whether it was included
            for i in range(n_active):
                if row[i]:
                    rmsecv_include[i].append(rmsecv)
                else:
                    rmsecv_exclude[i].append(rmsecv)

        if successful_samples < 10:
            print(f"  VCPA-IRIV iter {outer_iter}: Too few successful samples ({successful_samples})")
            continue

        # Statistical classification of each variable
        categories = ['unknown'] * n_active
        keep_mask = np.ones(n_active, dtype=bool)
        n_strong = n_weak = n_uninformative = n_interfering = 0

        for i in range(n_active):
            inc = rmsecv_include[i]
            exc = rmsecv_exclude[i]

            # Need enough samples for statistical test
            if len(inc) < 5 or len(exc) < 5:
                categories[i] = 'weak'  # Not enough data, assume weakly informative
                continue

            # Mann-Whitney U test: compare RMSECV distributions
            # H0: distributions are the same
            # H1: distributions are different
            try:
                stat, p = mannwhitneyu(exc, inc, alternative='two-sided')
                h = 1 if p < 0.05 else 0
            except Exception:
                h = 0
                p = 1.0

            # DMEAN = mean(exclude) - mean(include)
            # POSITIVE DMEAN means RMSECV is higher when excluded
            # → including the variable IMPROVES performance (lower RMSECV)
            # NEGATIVE DMEAN means RMSECV is lower when excluded
            # → including the variable WORSENS performance
            dmean = np.mean(exc) - np.mean(inc)

            if h == 1 and dmean > 0:
                # Significant improvement when included → strongly informative
                categories[i] = 'strong'
                n_strong += 1
            elif h == 0 and dmean > 0:
                # Non-significant improvement when included → weakly informative
                categories[i] = 'weak'
                n_weak += 1
            elif h == 0 and dmean <= 0:
                # No significant effect or slight harm → uninformative
                categories[i] = 'uninformative'
                keep_mask[i] = False
                n_uninformative += 1
            else:  # h == 1 and dmean < 0
                # Significant worsening when included → interfering
                categories[i] = 'interfering'
                keep_mask[i] = False
                n_interfering += 1

        # Store categories for current active variables
        for i, var_idx in enumerate(active_indices):
            variable_categories[var_idx] = categories[i]

        # Compute current iteration's RMSECV (using all active variables)
        current_rmsecv = compute_rmsecv(X[:, active_indices], y)
        convergence_history.append(current_rmsecv)

        # Report iteration results
        n_removed = n_uninformative + n_interfering
        print(f"  VCPA-IRIV iter {outer_iter}: {n_strong} strong, {n_weak} weak, "
              f"{n_uninformative} uninformative, {n_interfering} interfering "
              f"(removing {n_removed}, RMSECV={current_rmsecv:.4f})")

        # Remove uninformative and interfering variables
        if np.sum(keep_mask) <= pls_components:
            print(f"  VCPA-IRIV: Would remove too many variables, stopping")
            break

        if n_removed == 0:
            print(f"  VCPA-IRIV: Converged at iteration {outer_iter} (no variables removed)")
            break

        active_indices = active_indices[keep_mask]

    # Final evaluation with selected variables
    selected_indices = active_indices
    final_rmsecv = compute_rmsecv(X[:, selected_indices], y)

    # Compute importance scores based on final categories
    importance_scores = np.zeros(len(selected_indices))
    for i, var_idx in enumerate(selected_indices):
        cat = variable_categories.get(var_idx, 'unknown')
        if cat == 'strong':
            importance_scores[i] = 1.0
        elif cat == 'weak':
            importance_scores[i] = 0.5
        else:
            importance_scores[i] = 0.25

    result = {
        'selected_indices': selected_indices,
        'importance_scores': importance_scores,
        'variable_categories': variable_categories,
        'convergence_history': convergence_history,
        'n_vars_history': n_vars_history,
        'final_rmsecv': final_rmsecv,
        'n_selected': len(selected_indices)
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
