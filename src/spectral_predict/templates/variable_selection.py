"""
Variable selection algorithm templates for generated scripts.

These are complete, standalone implementations of each algorithm
using only numpy and sklearn, suitable for publication.
"""

VIP_TEMPLATE = '''
def compute_vip(X, y, n_components=None):
    """
    Compute VIP (Variable Importance in Projection) scores for PLS.

    VIP measures each variable's contribution to the PLS model.
    Variables with VIP > 1 are typically considered important.

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    n_components : int, optional
        Number of PLS components. If None, auto-selects.

    Returns
    -------
    np.ndarray
        VIP scores for each wavelength (higher = more important)

    Reference
    ---------
    Wold, S., Sjostrom, M., & Eriksson, L. (2001). PLS-regression: a basic
    tool of chemometrics. Chemometrics and Intelligent Laboratory Systems,
    58(2), 109-130.
    """
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    if n_components is None:
        n_components = min(10, n_features // 2, n_samples // 2)
    n_components = max(1, n_components)

    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)

    W = pls.x_weights_  # (n_features, n_components)
    T = pls.x_scores_   # (n_samples, n_components)

    # Explained variance per component
    y_reshaped = y.reshape(-1, 1)
    ssy_comp = np.sum(T**2, axis=0) * np.var(y_reshaped, axis=0)
    ssy_total = np.sum(ssy_comp)

    if ssy_total < 1e-10:
        return np.ones(n_features)

    # VIP calculation
    weight = np.sum((W ** 2) * ssy_comp, axis=1)
    vip_scores = np.sqrt(n_features * weight / ssy_total)

    return vip_scores


def select_by_vip(X, y, n_variables, n_components=None):
    """
    Select top variables by VIP score.

    Parameters
    ----------
    X : np.ndarray
        Spectral data
    y : np.ndarray
        Target values
    n_variables : int
        Number of variables to select
    n_components : int, optional
        Number of PLS components

    Returns
    -------
    np.ndarray
        Indices of selected variables (sorted by importance, descending)
    """
    vip_scores = compute_vip(X, y, n_components)
    selected_indices = np.argsort(vip_scores)[::-1][:n_variables]
    return np.sort(selected_indices)
'''

SPA_TEMPLATE = '''
def spa_selection(X, y, n_variables, random_state=42):
    """
    Successive Projections Algorithm (SPA) for variable selection.

    Selects variables with minimum collinearity by iteratively choosing
    variables that have maximum projection orthogonal to already-selected variables.

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    n_variables : int
        Number of variables to select
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Indices of selected variables (in selection order)

    Reference
    ---------
    Araujo, M.C.U., et al. (2001). The successive projections algorithm for
    variable selection in spectroscopic multicomponent analysis.
    Chemometrics and Intelligent Laboratory Systems, 57(2), 65-73.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    n_variables = min(n_variables, n_features)

    # Normalize X for correlation computation
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    # Normalize y
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-10
    y_norm = (y - y_mean) / y_std

    # Initial correlations with y
    initial_corrs = np.abs(X_norm.T @ y_norm) / n_samples

    # Start with variable most correlated with y
    selected = [int(np.argmax(initial_corrs))]
    available = set(range(n_features)) - set(selected)

    # Iteratively select remaining variables
    for _ in range(1, n_variables):
        projections = np.zeros(n_features)
        X_selected = X_norm[:, selected]

        for j in available:
            # Correlation with already selected variables
            corrs = X_norm[:, j] @ X_selected / n_samples
            projections[j] = np.sum(corrs ** 2)

        # Select variable with minimum projection (least correlated with selected)
        min_idx = min(available, key=lambda j: projections[j])
        selected.append(min_idx)
        available.remove(min_idx)

    return np.array(selected)
'''

UVE_TEMPLATE = '''
def uve_selection(X, y, n_variables=None, cutoff_multiplier=1.0, cv_folds=5, random_state=42):
    """
    Uninformative Variable Elimination (UVE).

    Eliminates variables that contribute no more than random noise by comparing
    variable reliability scores against noise variable scores.

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    n_variables : int, optional
        Number of variables to select. If None, uses threshold-based selection.
    cutoff_multiplier : float
        Multiplier for noise threshold (higher = more conservative)
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Indices of selected variables

    Reference
    ---------
    Centner, V., et al. (1996). Elimination of uninformative variables for
    multivariate calibration. Analytical Chemistry, 68(21), 3851-3858.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    rng = np.random.RandomState(random_state)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    cv_folds = min(cv_folds, n_samples // 2)
    n_components = min(10, n_features // 2, n_samples // 2)
    n_components = max(1, n_components)

    # Create augmented dataset with noise variables
    noise = rng.randn(n_samples, n_features)
    X_aug = np.hstack([X, noise])

    # Collect PLS coefficients across CV folds
    n_aug = X_aug.shape[1]
    coefficients = np.zeros((cv_folds, n_aug))

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, _) in enumerate(kf.split(X_aug)):
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_aug[train_idx], y[train_idx])
        coef = pls.coef_.ravel()
        coefficients[fold_idx] = coef

    # Calculate reliability: mean(|coef|) / std(coef)
    mean_abs = np.mean(np.abs(coefficients), axis=0)
    std_coef = np.std(coefficients, axis=0)

    reliability = np.zeros(n_aug)
    valid = std_coef > 1e-10
    reliability[valid] = mean_abs[valid] / std_coef[valid]

    # Split real and noise variables
    real_reliability = reliability[:n_features]
    noise_reliability = reliability[n_features:]

    # Noise threshold
    threshold = np.max(noise_reliability) * cutoff_multiplier if np.max(noise_reliability) > 0 else 0

    # Select variables
    if n_variables is not None:
        # Select top n_variables by reliability
        selected_indices = np.argsort(real_reliability)[::-1][:n_variables]
    else:
        # Threshold-based selection
        selected_indices = np.where(real_reliability > threshold)[0]
        if len(selected_indices) == 0:
            selected_indices = np.argsort(real_reliability)[::-1][:10]

    return np.sort(selected_indices)
'''

CARS_TEMPLATE = '''
def cars_selection(X, y, n_variables=None, n_iterations=50, cv_folds=5, random_state=42):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for variable selection.

    Uses Monte Carlo sampling with exponential decay to select optimal variables
    by balancing exploration and exploitation.

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    n_variables : int, optional
        Target number of variables (used for final selection)
    n_iterations : int
        Number of Monte Carlo iterations
    cv_folds : int
        Cross-validation folds for evaluation
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Indices of selected variables

    Reference
    ---------
    Li, H.D., et al. (2009). Key wavelengths screening using competitive
    adaptive reweighted sampling method for multivariate calibration.
    Analytica Chimica Acta, 648(1), 77-84.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    rng = np.random.RandomState(random_state)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    pls_components = min(10, n_samples // 2, n_features // 2)
    cv_folds = min(cv_folds, n_samples // 2)

    weights = np.ones(n_features)

    best_rmsecv = np.inf
    best_selection = None

    for iteration in range(n_iterations):
        # Exponential decay
        r = 0.8 * np.exp(-2 * iteration / n_iterations)
        n_sample = max(int(n_features * 0.8 * r), pls_components + 1)
        n_sample = min(n_sample, n_features)

        # Sample variables based on weights
        probs = weights / weights.sum()
        selected = rng.choice(n_features, size=n_sample, replace=False, p=probs)
        selected = np.sort(selected)

        X_sub = X[:, selected]

        # Cross-validation
        try:
            n_comp = min(pls_components, n_sample - 1)
            pls = PLSRegression(n_components=n_comp)

            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            errors = []

            for train_idx, val_idx in kf.split(X_sub):
                pls.fit(X_sub[train_idx], y[train_idx])
                pred = pls.predict(X_sub[val_idx])
                errors.append(np.mean((y[val_idx] - pred.ravel()) ** 2))

            rmsecv = np.sqrt(np.mean(errors))

            if rmsecv < best_rmsecv:
                best_rmsecv = rmsecv
                best_selection = selected.copy()

            # Update weights
            pls.fit(X_sub, y)
            coef = np.abs(pls.coef_.ravel())
            temp_weights = weights.copy()
            temp_weights[selected] = coef
            weights = temp_weights / (temp_weights.sum() + 1e-10)

        except Exception:
            continue

    if best_selection is None:
        # Fallback: select by weight
        best_selection = np.argsort(weights)[::-1][:n_variables or 50]

    if n_variables is not None and len(best_selection) > n_variables:
        # Trim to requested size
        best_selection = best_selection[:n_variables]

    return np.sort(best_selection)
'''


def get_variable_selection_template(method: str) -> str:
    """
    Get the variable selection template for a given method.

    Parameters
    ----------
    method : str
        Variable selection method name: 'vip', 'spa', 'uve', 'cars', 'importance'

    Returns
    -------
    str
        Template code for the variable selection method
    """
    templates = {
        'vip': VIP_TEMPLATE,
        'spa': SPA_TEMPLATE,
        'uve': UVE_TEMPLATE,
        'cars': CARS_TEMPLATE,
        'uve_spa': UVE_TEMPLATE + '\n' + SPA_TEMPLATE,
        'importance': '',  # Uses sklearn's feature_importances_ directly
    }

    return templates.get(method.lower(), '')
