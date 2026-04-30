"""
Variable selection methods for spectral analysis.

This module implements various variable selection algorithms to identify
the most informative spectral variables for prediction.
"""

import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


def _get_cv_n_jobs():
    """Get n_jobs for CV, respecting frozen app constraints."""
    from spectral_predict.search import _frozen_needs_threading_fallback
    return 1 if _frozen_needs_threading_fallback() else -1


def _spa_seed_n_jobs():
    """Worker count for the SPA seed loop.

    Threading backend is used (numpy/sklearn release the GIL on the heavy
    matmul + PLS CV work, and threading is PyInstaller-bundle-safe — loky
    has known argv-parse issues in the frozen runtime per
    `search._frozen_needs_threading_fallback`). Returns the cpu count
    capped at 8 to avoid pathological oversubscription on 32-core dev boxes
    (each thread fights for the same numpy BLAS pool).
    """
    return min(os.cpu_count() or 1, 8)


def uve_selection(X, y, cutoff_multiplier=1.0, n_components=None, cv_folds=5, random_state=42):
    """
    Uninformative Variable Elimination (UVE) - eliminates variables that contribute no more than noise.

    The UVE algorithm augments the original data with random noise variables, then uses
    cross-validated PLS regression to determine which variables are more informative than
    random noise. Variables with reliability scores below the noise threshold are considered
    uninformative.

    Algorithm:
    1. Create augmented dataset: [Real Variables | Random Noise Variables]
    2. Build PLS models across CV folds on augmented data
    3. Calculate reliability score for each variable: mean(abs(coef)) / std(coef)
    4. Compute noise threshold from noise variable scores
    5. Return absolute reliability scores (higher = more informative)

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cutoff_multiplier : float, default=1.0
        Multiplier for noise threshold (higher = more aggressive filtering)
        Values > 1.0 make filtering more conservative (keep more variables)
        Values < 1.0 make filtering more aggressive (eliminate more variables)
    n_components : int or None
        Number of PLS components (if None, auto-select as min(10, n_features//2, n_samples//2))
    cv_folds : int, default=5
        Number of CV folds for cross-validation
    random_state : int, default=42
        Random seed for noise variable generation (for reproducibility)

    Returns
    -------
    importances : np.ndarray
        Reliability scores for each variable (higher = more informative variable)
        Shape: (n_features,)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import uve_selection
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(50, 100)  # 50 samples, 100 variables
    >>> y = np.random.randn(50)
    >>>
    >>> # Calculate UVE importances
    >>> importances = uve_selection(X, y, cutoff_multiplier=1.0)
    >>>
    >>> # Select variables above noise threshold
    >>> # (In practice, you'd compare importances to the threshold from noise variables)
    >>> selected_vars = importances > np.median(importances)
    >>> X_selected = X[:, selected_vars]

    References
    ----------
    Centner, V., Massart, D. L., de Noord, O. E., de Jong, S., Vandeginste, B. M., & Sterna, C. (1996).
    Elimination of uninformative variables for multivariate calibration.
    Analytical Chemistry, 68(21), 3851-3858.
    """
    # Convert inputs to numpy arrays and ensure proper shapes
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    # Graceful degradation for small feature counts
    # UVE doubles features with noise and builds PLS, so needs reasonable feature count
    MIN_UVE_FEATURES = 3  # Minimum for meaningful UVE (need n_components >= 1 with n_features//2)
    if n_features < MIN_UVE_FEATURES:
        print(f"WARNING: UVE skipped - only {n_features} wavelengths available "
              f"(minimum {MIN_UVE_FEATURES} required). Returning uniform importance.")
        return np.ones(n_features)

    # Handle edge case: adjust cv_folds if n_samples is too small
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        print(f"Warning: UVE adjusted cv_folds to {cv_folds} due to small sample size")

    # Auto-select n_components if not provided
    if n_components is None:
        n_components = min(10, n_features // 2, n_samples // 2)

    # Ensure n_components is at least 1
    n_components = max(1, n_components)

    # Step 1: Create augmented dataset with random noise variables
    # Add the same number of noise variables as real variables
    rng = np.random.RandomState(random_state)
    noise_variables = rng.randn(n_samples, n_features)
    X_augmented = np.hstack([X, noise_variables])

    # Step 2: Build PLS models across CV folds and collect coefficients
    # Initialize array to store coefficients from each fold
    n_augmented_features = X_augmented.shape[1]
    coefficients = np.zeros((cv_folds, n_augmented_features))

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_idx = 0

    for train_idx, _ in kfold.split(X_augmented):
        # Get training data for this fold
        X_train = X_augmented[train_idx]
        y_train = y[train_idx]

        # Fit PLS model
        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_train, y_train)

            # Extract coefficients
            coefficients[fold_idx] = pls.coef_.ravel()

        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle singular matrices or other PLS fitting errors
            # Leave coefficients as zeros for this fold
            print(f"Warning: PLS fitting failed for fold {fold_idx + 1}: {e}")
            coefficients[fold_idx] = 0.0

        fold_idx += 1

    # Step 3: Calculate reliability score for each variable
    # Reliability = mean(abs(coef)) / std(coef)
    mean_abs_coef = np.mean(np.abs(coefficients), axis=0)
    std_coef = np.std(coefficients, axis=0)

    # Handle division by zero: if std is 0, set reliability to 0
    reliability = np.zeros(n_augmented_features)
    non_zero_std = std_coef > 1e-10  # Use small threshold to avoid numerical issues
    reliability[non_zero_std] = mean_abs_coef[non_zero_std] / std_coef[non_zero_std]

    # Step 4: Compute noise threshold from noise variable scores
    # Extract reliability scores for real variables and noise variables
    real_reliability = reliability[:n_features]
    noise_reliability = reliability[n_features:]

    # Noise threshold is the maximum reliability among noise variables
    if len(noise_reliability) > 0 and np.max(noise_reliability) > 0:
        noise_threshold = np.max(noise_reliability) * cutoff_multiplier
    else:
        # Fallback: if all noise reliabilities are 0, use a small threshold
        noise_threshold = 0.0

    # Step 5: Return absolute reliability scores for real variables
    # Higher scores indicate more informative variables
    # Note: We return the raw reliability scores, not a binary mask
    # This allows the caller to decide on filtering based on the threshold
    importances = real_reliability

    # Handle edge case: if all variables would be eliminated (all scores are 0)
    if np.all(importances == 0):
        # Return uniform scores so no variables are preferentially eliminated
        importances = np.ones(n_features)

    return importances


def get_uve_threshold(X, y, cutoff_multiplier=1.0, n_components=None, cv_folds=5, random_state=42):
    """
    Calculate the UVE noise threshold for variable selection.

    This is a helper function that returns both the importances and the threshold
    value that can be used to filter variables.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cutoff_multiplier : float, default=1.0
        Multiplier for noise threshold
    n_components : int or None
        Number of PLS components
    cv_folds : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed for noise variable generation (for reproducibility)

    Returns
    -------
    importances : np.ndarray
        Reliability scores for each variable
    threshold : float
        The noise threshold value
    selected_mask : np.ndarray
        Boolean mask of selected variables (True = informative, False = noise)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import get_uve_threshold
    >>>
    >>> X = np.random.randn(50, 100)
    >>> y = np.random.randn(50)
    >>>
    >>> importances, threshold, mask = get_uve_threshold(X, y)
    >>> X_selected = X[:, mask]
    >>> print(f"Selected {np.sum(mask)} out of {len(mask)} variables")
    """
    # Convert inputs
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    # Adjust parameters
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)

    if n_components is None:
        n_components = min(10, n_features // 2, n_samples // 2)

    n_components = max(1, n_components)

    # Create augmented dataset
    rng = np.random.RandomState(random_state)
    noise_variables = rng.randn(n_samples, n_features)
    X_augmented = np.hstack([X, noise_variables])

    # Collect coefficients
    n_augmented_features = X_augmented.shape[1]
    coefficients = np.zeros((cv_folds, n_augmented_features))

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_idx = 0

    for train_idx, _ in kfold.split(X_augmented):
        X_train = X_augmented[train_idx]
        y_train = y[train_idx]

        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_train, y_train)

            coefficients[fold_idx] = pls.coef_.ravel()

        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: PLS fitting failed for fold {fold_idx + 1}: {e}")
            coefficients[fold_idx] = 0.0

        fold_idx += 1

    # Calculate reliability
    mean_abs_coef = np.mean(np.abs(coefficients), axis=0)
    std_coef = np.std(coefficients, axis=0)

    reliability = np.zeros(n_augmented_features)
    non_zero_std = std_coef > 1e-10
    reliability[non_zero_std] = mean_abs_coef[non_zero_std] / std_coef[non_zero_std]

    # Split into real and noise
    real_reliability = reliability[:n_features]
    noise_reliability = reliability[n_features:]

    # Calculate threshold
    if len(noise_reliability) > 0 and np.max(noise_reliability) > 0:
        threshold = np.max(noise_reliability) * cutoff_multiplier
    else:
        threshold = 0.0

    # Create selection mask
    selected_mask = real_reliability > threshold

    # Handle edge case: if all eliminated, select all
    if not np.any(selected_mask):
        selected_mask = np.ones(n_features, dtype=bool)

    return real_reliability, threshold, selected_mask


def _evaluate_spa_seed(first_var, X, X_norm, y, n_samples, n_vars, n_features, cv_folds):
    """Run one canonical SPA chain seeded at `first_var` and return (selection, CV R²).

    Pure function — reads only the passed-in arrays (which are not mutated).
    Designed to run concurrently across all J candidate seeds in
    `spa_selection`. Returns (None, -inf) if the chain's CV scoring fails.
    """
    selected = [first_var]
    available = set(range(n_vars))
    available.remove(first_var)

    for _ in range(1, n_features):
        X_selected_norm = X_norm[:, selected]
        if X_selected_norm.ndim == 1:
            X_selected_norm = X_selected_norm.reshape(-1, 1)
        avail_idx = np.array(sorted(available))
        X_avail_norm = X_norm[:, avail_idx]
        corr_matrix = (X_avail_norm.T @ X_selected_norm) / n_samples
        proj_values = np.sum(corr_matrix ** 2, axis=1)
        min_proj_var = int(avail_idx[np.argmin(proj_values)])
        selected.append(min_proj_var)
        available.remove(min_proj_var)

    try:
        n_components = min(n_features, n_samples - 1, 10)
        pls = PLSRegression(n_components=n_components, scale=False)
        cv_scores = cross_val_score(
            pls, X[:, selected], y,
            cv=cv_folds, scoring="r2", n_jobs=1,
        )
        mean_score = float(np.mean(cv_scores))
        if np.isfinite(mean_score):
            return selected, mean_score
    except Exception:
        pass
    return None, -np.inf


def spa_selection(X, y, n_features, cv_folds=5):
    """
    Successive Projections Algorithm (SPA) - selects minimally correlated variables.

    SPA reduces collinearity by iteratively selecting variables that have minimum
    projection (correlation) onto the already-selected variable set. This creates
    a set of maximally uncorrelated features.

    Algorithm (canonical Araújo 2001):
    1. For every variable k=1..J as candidate first variable:
       a. Initialize chain with k as the seed
       b. Iteratively select variable with MINIMUM projection onto selected set
       c. Projection = sum of squared correlations with already-selected variables
       d. Evaluate selection quality using PLS R² via CV
    2. Return the chain with the best CV criterion across all J seeds
    3. Convert to importance scores (earlier selected = higher score)

    Note on determinism: SPA is fully deterministic. There is no random
    initialization in the canonical algorithm; the seed is enumerated, not
    randomized. dasp prior to T-06 had an `n_random_starts` parameter that
    looped a single argmax-correlation chain N times producing identical
    output — that knob has been removed.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    n_features : int
        Number of features to select
    cv_folds : int, default=5
        Number of CV folds for quality evaluation

    Returns
    -------
    importances : np.ndarray
        Importance scores (higher = earlier selected = more important)
        Shape: (X.shape[1],)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import spa_selection
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(50, 100)  # 50 samples, 100 variables
    >>> y = np.random.randn(50)
    >>>
    >>> # Select 20 minimally correlated variables
    >>> importances = spa_selection(X, y, n_features=20)
    >>>
    >>> # Get top variables
    >>> top_indices = np.argsort(importances)[-20:]
    >>> X_selected = X[:, top_indices]

    References
    ----------
    Araújo, M. C. U., et al. "The successive projections algorithm for variable
    selection in spectroscopic multicomponent analysis." Chemometrics and
    Intelligent Laboratory Systems 57.2 (2001): 65-73.
    """
    # Convert inputs to numpy arrays and ensure proper shapes
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_vars = X.shape

    # Graceful degradation for very small feature counts
    MIN_SPA_FEATURES = 3  # Minimum for meaningful SPA selection
    if n_vars < MIN_SPA_FEATURES:
        print(f"WARNING: SPA skipped - only {n_vars} wavelengths available "
              f"(minimum {MIN_SPA_FEATURES} required for meaningful selection). "
              f"Returning uniform importance for all wavelengths.")
        return np.ones(n_vars)

    # Handle edge case: if requesting more features than available, use all
    if n_features > n_vars:
        print(f"WARNING: SPA requested {n_features} features but only {n_vars} wavelengths available. "
              f"Using all {n_vars} wavelengths.")
        n_features = n_vars

    # Handle edge case: reduce cv_folds if not enough samples
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        print(f"Warning: Insufficient samples. Reducing cv_folds to {cv_folds}")

    # Step 1: Normalize X for projection computation (zero mean, unit variance).
    # This makes dot products equivalent to correlations.
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10  # Avoid division by zero on flat columns
    X_norm = (X - X_mean) / X_std

    # Note: y is no longer normalized — Araújo 2001 SPA enumerates every variable
    # as a candidate first variable and selects the chain with the best CV
    # criterion. The pre-T-06 implementation seeded only at
    # argmax(|corr(X[:, j], y)|) (which required y normalization); canon does not.

    print(
        f"Running canonical SPA over {n_vars} candidate seeds "
        f"(threading n_jobs={_spa_seed_n_jobs()})..."
    )

    # Step 2: Evaluate every variable as a candidate first variable (Araújo 2001).
    # Each seed's chain is independent — read-only access to X, X_norm, y.
    # Parallelize via joblib with threading backend (GIL-free for numpy/sklearn
    # work, PyInstaller-bundle-safe; loky has frozen-runtime argv-parse issues).
    results = Parallel(n_jobs=_spa_seed_n_jobs(), backend="threading")(
        delayed(_evaluate_spa_seed)(
            first_var, X, X_norm, y, n_samples, n_vars, n_features, cv_folds
        )
        for first_var in range(n_vars)
    )

    # Sequentially pick the best chain across all seeds.
    best_score = -np.inf
    best_selection = None
    for selected, score in results:
        if selected is not None and score > best_score:
            best_score = score
            best_selection = selected

    # Step 4: Convert best selection to importance scores
    # Earlier selected = higher importance
    importances = np.zeros(n_vars)
    if best_selection is not None:
        for rank, var_idx in enumerate(best_selection):
            # Assign scores: first selected gets n_features, last gets 1
            importances[var_idx] = n_features - rank
    else:
        print("Warning: All SPA seeds failed. Returning uniform importances.")
        importances = np.ones(n_vars)

    print(f"\nBest selection achieved R² = {best_score:.4f}")
    print(f"Selected {n_features} variables with importance scores")

    return importances


def ipls_selection(X, y, n_intervals=20, n_components=None, cv_folds=5, random_state=42):
    """
    Interval PLS (iPLS) - selects spectral variables based on interval performance.

    iPLS divides the spectrum into intervals and evaluates each interval's predictive
    performance using PLS regression. This method is particularly useful for identifying
    informative spectral regions.

    Algorithm:
    1. Divide spectrum into n_intervals equal-width intervals
    2. For each interval, build PLS model using only variables in that interval
    3. Evaluate each interval's performance using cross-validated R²
    4. Return scores where variables in better intervals get higher scores

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    n_intervals : int, default=20
        Number of intervals to divide the spectrum into
    n_components : int or None
        Number of PLS components (if None, auto-select based on interval size)
    cv_folds : int, default=5
        Number of CV folds for interval evaluation
    random_state : int, default=42
        Random seed for reproducibility (currently iPLS is deterministic, but
        this parameter is included for API consistency)

    Returns
    -------
    importances : np.ndarray
        Importance scores based on interval performance
        Variables in better intervals receive higher scores
        Shape: (n_features,)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import ipls_selection
    >>>
    >>> # Generate sample spectral data
    >>> X = np.random.randn(50, 200)  # 50 samples, 200 wavelengths
    >>> y = np.random.randn(50)
    >>>
    >>> # Evaluate spectral intervals
    >>> importances = ipls_selection(X, y, n_intervals=20)
    >>>
    >>> # Select variables from best intervals
    >>> top_indices = np.argsort(importances)[-50:]
    >>> X_selected = X[:, top_indices]

    References
    ----------
    Nørgaard, L., et al. "Interval partial least-squares regression (iPLS):
    A comparative chemometric study with an example from near-infrared spectroscopy."
    Applied Spectroscopy 54.3 (2000): 413-419.
    """
    # Convert inputs to numpy arrays and ensure proper shapes
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    # Graceful degradation for small feature counts
    # iPLS needs enough features for meaningful intervals (at least 2 features per interval)
    MIN_IPLS_FEATURES = 6  # Minimum for meaningful iPLS (at least 2-3 intervals with 2+ features each)
    if n_features < MIN_IPLS_FEATURES:
        print(f"WARNING: iPLS skipped - only {n_features} wavelengths available "
              f"(minimum {MIN_IPLS_FEATURES} required for meaningful interval selection). "
              f"Returning uniform importance for all wavelengths.")
        # Return uniform importance (all wavelengths equally important)
        return np.ones(n_features)

    # Handle edge case: adjust cv_folds if n_samples is too small
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        print(f"Warning: Insufficient samples. Reducing cv_folds to {cv_folds}")

    # Auto-adjust n_intervals based on available features
    # Goal: each interval should have at least 2-3 features for meaningful PLS
    MIN_FEATURES_PER_INTERVAL = 2
    max_sensible_intervals = n_features // MIN_FEATURES_PER_INTERVAL

    if n_intervals > max_sensible_intervals:
        original_intervals = n_intervals
        n_intervals = max(2, max_sensible_intervals)  # At least 2 intervals
        print(f"WARNING: Auto-adjusted n_intervals from {original_intervals} to {n_intervals} "
              f"(only {n_features} wavelengths available, need at least {MIN_FEATURES_PER_INTERVAL} per interval)")

    # Handle edge case: ensure at least 1 interval
    n_intervals = max(1, n_intervals)

    # Calculate interval boundaries
    # Divide features into roughly equal-sized intervals
    interval_size = n_features // n_intervals
    if interval_size < 1:
        interval_size = 1
        n_intervals = n_features

    # Create interval boundaries
    intervals = []
    for i in range(n_intervals):
        start_idx = i * interval_size
        # Last interval gets any remaining features
        if i == n_intervals - 1:
            end_idx = n_features
        else:
            end_idx = (i + 1) * interval_size

        # Only add non-empty intervals
        if end_idx > start_idx:
            intervals.append((start_idx, end_idx))

    print(f"iPLS: Evaluating {len(intervals)} intervals (avg size: {interval_size} features)")

    # Evaluate each interval using PLS with CV
    interval_scores = np.zeros(len(intervals))

    for interval_idx, (start, end) in enumerate(intervals):
        # Extract features for this interval
        X_interval = X[:, start:end]
        n_interval_features = end - start

        # Skip empty intervals (shouldn't happen, but be safe)
        if n_interval_features == 0:
            interval_scores[interval_idx] = 0.0
            continue

        # Auto-select n_components if not provided
        # Use min of: specified value, half of interval features, half of samples, 10
        if n_components is None:
            interval_n_components = min(n_interval_features // 2, n_samples // 2, 10)
        else:
            interval_n_components = min(n_components, n_interval_features, n_samples - 1)

        # Ensure at least 1 component
        interval_n_components = max(1, interval_n_components)

        # Build PLS model and evaluate with CV
        try:
            pls = PLSRegression(n_components=interval_n_components, scale=False)

            # Cross-validation R² score
            cv_scores = cross_val_score(
                pls, X_interval, y,
                cv=cv_folds,
                scoring='r2',
                n_jobs=_get_cv_n_jobs()
            )

            # Use mean R² as interval score
            mean_r2 = np.mean(cv_scores)

            # Handle negative R² (worse than predicting mean)
            # Clip to 0 so poor intervals get low scores
            interval_scores[interval_idx] = max(0.0, mean_r2)

            print(f"  Interval {interval_idx+1}/{len(intervals)} "
                  f"(features {start}-{end}): R² = {mean_r2:.4f}")

        except Exception as e:
            print(f"  Interval {interval_idx+1}/{len(intervals)} "
                  f"(features {start}-{end}): Failed - {str(e)}")
            interval_scores[interval_idx] = 0.0

    # Convert interval scores to feature importances
    # Each feature gets the score of its interval
    importances = np.zeros(n_features)

    for interval_idx, (start, end) in enumerate(intervals):
        importances[start:end] = interval_scores[interval_idx]

    # Handle edge case: if all intervals failed (all scores are 0)
    if np.all(importances == 0):
        print("Warning: All intervals failed. Returning uniform importances.")
        importances = np.ones(n_features)

    # Print summary
    best_interval_idx = np.argmax(interval_scores)
    best_start, best_end = intervals[best_interval_idx]
    print(f"\nBest interval: {best_interval_idx+1} "
          f"(features {best_start}-{best_end}), R² = {interval_scores[best_interval_idx]:.4f}")

    return importances


def uve_spa_selection(X, y, n_features, cutoff_multiplier=1.0,
                      uve_n_components=None, uve_cv_folds=5,
                      spa_cv_folds=5, random_state=42):
    """
    UVE-SPA Hybrid - combines noise filtering (UVE) with collinearity reduction (SPA).

    This hybrid method first applies UVE to eliminate uninformative variables,
    then applies SPA on the remaining variables to select a minimally correlated subset.
    This combines the benefits of both methods: noise filtering and collinearity reduction.

    Algorithm:
    1. Run UVE to get reliability scores
    2. Keep only informative variables (scores > noise threshold)
    3. Run SPA on the reduced variable set
    4. Return combined scores (0 for eliminated, SPA scores for kept)

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    n_features : int
        Number of features to select (after both UVE and SPA)
    cutoff_multiplier : float, default=1.0
        UVE noise threshold multiplier
    uve_n_components : int or None
        Number of PLS components for UVE
    uve_cv_folds : int, default=5
        Number of CV folds for UVE
    spa_cv_folds : int, default=5
        Number of CV folds for SPA evaluation
    random_state : int, default=42
        Random seed for noise variable generation in UVE (for reproducibility)

    Returns
    -------
    importances : np.ndarray
        Combined importance scores
        Eliminated variables get 0, selected variables get SPA scores
        Shape: (n_features,)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import uve_spa_selection
    >>>
    >>> # Generate sample data with noise
    >>> X = np.random.randn(50, 100)
    >>> y = np.random.randn(50)
    >>>
    >>> # Apply hybrid method: filter noise, then reduce collinearity
    >>> importances = uve_spa_selection(X, y, n_features=20)
    >>>
    >>> # Get selected variables
    >>> top_indices = np.argsort(importances)[-20:]
    >>> X_selected = X[:, top_indices]

    References
    ----------
    Combines methods from:
    - Centner et al. (1996) - UVE algorithm
    - Araújo et al. (2001) - SPA algorithm
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_vars = X.shape

    print(f"\n=== UVE-SPA Hybrid Selection ===")
    print(f"Starting with {n_vars} variables, target: {n_features} variables")

    # Step 1: Apply UVE to filter uninformative variables
    print(f"\nStep 1: UVE filtering...")
    uve_importances, threshold, uve_mask = get_uve_threshold(
        X, y,
        cutoff_multiplier=cutoff_multiplier,
        n_components=uve_n_components,
        cv_folds=uve_cv_folds,
        random_state=random_state
    )

    n_uve_selected = np.sum(uve_mask)
    print(f"UVE selected {n_uve_selected} / {n_vars} variables (threshold: {threshold:.4f})")

    # Handle edge case: if UVE eliminates everything, keep all
    if n_uve_selected == 0:
        print("Warning: UVE eliminated all variables. Skipping UVE step.")
        uve_mask = np.ones(n_vars, dtype=bool)
        n_uve_selected = n_vars

    # Handle edge case: if UVE kept fewer than target, adjust n_features
    spa_n_features = min(n_features, n_uve_selected)
    if spa_n_features < n_features:
        print(f"Warning: UVE kept only {n_uve_selected} variables. "
              f"Adjusting SPA target from {n_features} to {spa_n_features}")

    # Step 2: Apply SPA on the UVE-selected variables
    print(f"\nStep 2: SPA on UVE-selected variables...")

    # Extract only the UVE-selected variables
    X_uve_selected = X[:, uve_mask]

    # Run SPA on the reduced set
    spa_importances_reduced = spa_selection(
        X_uve_selected, y,
        n_features=spa_n_features,
        cv_folds=spa_cv_folds,
    )

    # Step 3: Combine UVE and SPA results
    # Create full-size importance array (zeros for eliminated variables)
    combined_importances = np.zeros(n_vars)

    # Map SPA scores back to original indices
    uve_indices = np.where(uve_mask)[0]
    combined_importances[uve_indices] = spa_importances_reduced

    # Verify how many variables have non-zero scores
    n_final_selected = np.sum(combined_importances > 0)

    print(f"\n=== Final Results ===")
    print(f"UVE eliminated: {n_vars - n_uve_selected} variables")
    print(f"SPA selected: {n_final_selected} variables from UVE-kept set")
    print(f"Total eliminated: {n_vars - n_final_selected} variables")
    print(f"Final selection: {n_final_selected} variables")

    return combined_importances


def uve_cars_selection(X, y, cutoff_multiplier=1.0, uve_n_components=None, uve_cv_folds=5,
                       n_iterations=50, pls_components=5, cars_cv_folds=5,
                       monte_carlo_samples=80, random_state=42,
                       model_type=None, use_hybrid_importance=False,
                       hybrid_importance_weight=0.5, task_type='regression'):
    """
    UVE-CARS Hybrid - combines noise filtering (UVE) with adaptive selection (CARS).

    First applies UVE to eliminate uninformative variables, then runs CARS on the
    surviving variables. Also handles UVE-CARS-tree when model_type and
    use_hybrid_importance are set.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cutoff_multiplier : float, default=1.0
        UVE noise threshold multiplier
    uve_n_components : int or None
        Number of PLS components for UVE
    uve_cv_folds : int, default=5
        Number of CV folds for UVE
    n_iterations : int, default=50
        CARS Monte Carlo iterations
    pls_components : int, default=5
        PLS components for CARS
    cars_cv_folds : int, default=5
        CV folds for CARS
    monte_carlo_samples : int, default=80
        CARS Monte Carlo samples
    random_state : int, default=42
        Random seed
    model_type : str or None
        Model name for model-aware CARS (None = standard PLS-CARS)
    use_hybrid_importance : bool, default=False
        Use hybrid importance for tree models (CARS-tree variant)
    hybrid_importance_weight : float, default=0.5
        Weight for hybrid importance blending
    task_type : str, default='regression'
        Task type ('regression' or 'classification')

    Returns
    -------
    importances : np.ndarray
        Combined importance scores. Shape: (n_features,)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_vars = X.shape

    variant = "UVE-CARS-Tree" if use_hybrid_importance else "UVE-CARS"
    print(f"\n=== {variant} Hybrid Selection ===")
    print(f"Starting with {n_vars} variables")

    # Step 1: UVE filtering
    print(f"\nStep 1: UVE filtering...")
    uve_importances, threshold, uve_mask = get_uve_threshold(
        X, y,
        cutoff_multiplier=cutoff_multiplier,
        n_components=uve_n_components,
        cv_folds=uve_cv_folds,
        random_state=random_state
    )

    n_uve_selected = int(np.sum(uve_mask))
    print(f"UVE selected {n_uve_selected} / {n_vars} variables (threshold: {threshold:.4f})")

    if n_uve_selected == 0:
        print("Warning: UVE eliminated all variables. Returning UVE importances only.")
        return uve_importances

    if n_uve_selected < 10:
        print(f"Warning: UVE kept only {n_uve_selected} variables (< 10). Returning UVE importances only.")
        return uve_importances

    # Step 2: CARS on UVE survivors
    print(f"\nStep 2: CARS on {n_uve_selected} UVE-selected variables...")
    X_uve_selected = X[:, uve_mask]

    cars_importances_reduced = cars_selection(
        X_uve_selected, y,
        n_iterations=n_iterations,
        pls_components=pls_components,
        cv_folds=cars_cv_folds,
        monte_carlo_samples=monte_carlo_samples,
        random_state=random_state,
        model_type=model_type,
        use_hybrid_importance=use_hybrid_importance,
        hybrid_importance_weight=hybrid_importance_weight,
        task_type=task_type
    )

    # Map back to full indices
    combined_importances = np.zeros(n_vars)
    uve_indices = np.where(uve_mask)[0]
    combined_importances[uve_indices] = cars_importances_reduced

    n_final = int(np.sum(combined_importances > 0))
    print(f"\n=== {variant} Final Results ===")
    print(f"UVE eliminated: {n_vars - n_uve_selected} variables")
    print(f"CARS selected: {n_final} variables from UVE-kept set")

    return combined_importances


def uve_cars_spa_selection(X, y, cutoff_multiplier=1.0, uve_n_components=None, uve_cv_folds=5,
                           n_iterations=50, pls_components=5, cars_cv_folds=5,
                           monte_carlo_samples=80, spa_n_features=None,
                           spa_cv_folds=5,
                           random_state=42, task_type='regression'):
    """
    UVE-CARS-SPA 3-stage hybrid: noise filtering → adaptive selection → collinearity reduction.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cutoff_multiplier : float, default=1.0
        UVE noise threshold multiplier
    uve_n_components : int or None
        Number of PLS components for UVE
    uve_cv_folds : int, default=5
        CV folds for UVE
    n_iterations : int, default=50
        CARS Monte Carlo iterations
    pls_components : int, default=5
        PLS components for CARS
    cars_cv_folds : int, default=5
        CV folds for CARS
    monte_carlo_samples : int, default=80
        CARS Monte Carlo samples
    spa_n_features : int or None
        Target features for SPA (None = use all CARS survivors)
    spa_cv_folds : int, default=5
        CV folds for SPA
    random_state : int, default=42
        Random seed
    task_type : str, default='regression'
        Task type

    Returns
    -------
    importances : np.ndarray
        Combined importance scores. Shape: (n_features,)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_vars = X.shape

    print(f"\n=== UVE-CARS-SPA 3-Stage Hybrid Selection ===")
    print(f"Starting with {n_vars} variables")

    # Step 1: UVE filtering
    print(f"\nStep 1: UVE filtering...")
    uve_importances, threshold, uve_mask = get_uve_threshold(
        X, y,
        cutoff_multiplier=cutoff_multiplier,
        n_components=uve_n_components,
        cv_folds=uve_cv_folds,
        random_state=random_state
    )

    n_uve_selected = int(np.sum(uve_mask))
    print(f"UVE selected {n_uve_selected} / {n_vars} variables")

    if n_uve_selected == 0:
        print("Warning: UVE eliminated all variables. Returning UVE importances only.")
        return uve_importances

    if n_uve_selected < 10:
        print(f"Warning: UVE kept only {n_uve_selected} variables (< 10). Returning UVE importances only.")
        return uve_importances

    # Step 2: CARS on UVE survivors
    print(f"\nStep 2: CARS on {n_uve_selected} UVE-selected variables...")
    X_uve_selected = X[:, uve_mask]

    cars_importances_reduced = cars_selection(
        X_uve_selected, y,
        n_iterations=n_iterations,
        pls_components=pls_components,
        cv_folds=cars_cv_folds,
        monte_carlo_samples=monte_carlo_samples,
        random_state=random_state,
        task_type=task_type
    )

    # Get CARS survivors (non-zero importances)
    cars_survivor_mask = cars_importances_reduced > 0
    n_cars_survivors = int(np.sum(cars_survivor_mask))
    print(f"CARS selected {n_cars_survivors} variables from UVE-kept set")

    if n_cars_survivors < 3:
        print(f"Warning: Too few CARS survivors ({n_cars_survivors}) for SPA. Returning UVE-CARS importances.")
        combined = np.zeros(n_vars)
        uve_indices = np.where(uve_mask)[0]
        combined[uve_indices] = cars_importances_reduced
        return combined

    # Step 3: SPA on CARS survivors
    print(f"\nStep 3: SPA on {n_cars_survivors} CARS-selected variables...")
    X_cars_selected = X_uve_selected[:, cars_survivor_mask]

    spa_n = min(spa_n_features or n_cars_survivors, n_cars_survivors)
    spa_importances_reduced = spa_selection(
        X_cars_selected, y,
        n_features=spa_n,
        cv_folds=spa_cv_folds,
    )

    # Map back through both stages to full indices
    combined_importances = np.zeros(n_vars)
    uve_indices = np.where(uve_mask)[0]
    cars_indices_in_uve = np.where(cars_survivor_mask)[0]
    full_indices = uve_indices[cars_indices_in_uve]
    combined_importances[full_indices] = spa_importances_reduced

    n_final = int(np.sum(combined_importances > 0))
    print(f"\n=== UVE-CARS-SPA Final Results ===")
    print(f"UVE: {n_vars} → {n_uve_selected} | CARS: → {n_cars_survivors} | SPA: → {n_final}")

    return combined_importances


def fipls_spa_selection(X, y, wavelengths, n_intervals=20, max_combine=5,
                        ipls_cv_folds=5, spa_n_features=None,
                        spa_cv_folds=5,
                        random_state=42):
    """
    Forward iPLS → SPA hybrid: region selection followed by collinearity reduction.

    First runs forward iPLS to find the best spectral interval combination, then
    applies SPA on those variables to select a minimally correlated subset.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    n_intervals : int, default=20
        Number of intervals for iPLS
    max_combine : int, default=5
        Max intervals to combine in forward selection
    ipls_cv_folds : int, default=5
        CV folds for iPLS
    spa_n_features : int or None
        Target features for SPA (None = use all iPLS-selected)
    spa_cv_folds : int, default=5
        CV folds for SPA
    random_state : int, default=42
        Random seed

    Returns
    -------
    importances : np.ndarray
        Combined importance scores. Shape: (n_features,)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_vars = X.shape

    print(f"\n=== Forward iPLS-SPA Hybrid Selection ===")
    print(f"Starting with {n_vars} variables")

    # Step 1: Forward iPLS
    print(f"\nStep 1: Forward iPLS ({n_intervals} intervals, max_combine={max_combine})...")
    subsets = ipls_forward(
        X, y, wavelengths,
        n_intervals=n_intervals,
        max_combine=max_combine,
        cv_folds=ipls_cv_folds,
        random_state=random_state
    )

    if not subsets:
        print("Warning: Forward iPLS returned no subsets. Returning uniform importances.")
        return np.ones(n_vars)

    # Find best subset (lowest RMSECV)
    best = min(subsets, key=lambda s: s['rmsecv'])
    best_indices = best['indices']
    n_ipls_selected = len(best_indices)
    print(f"Best iPLS subset: {best['tag']} ({n_ipls_selected} vars, RMSECV={best['rmsecv']:.4f})")

    if n_ipls_selected < 3:
        print(f"Warning: Too few iPLS variables ({n_ipls_selected}) for SPA. Returning iPLS-only importances.")
        importances = np.zeros(n_vars)
        importances[best_indices] = 1.0
        return importances

    # Step 2: SPA on iPLS-selected variables
    print(f"\nStep 2: SPA on {n_ipls_selected} iPLS-selected variables...")
    X_ipls_selected = X[:, best_indices]

    spa_n = min(spa_n_features or n_ipls_selected, n_ipls_selected)
    spa_importances_reduced = spa_selection(
        X_ipls_selected, y,
        n_features=spa_n,
        cv_folds=spa_cv_folds,
    )

    # Map SPA scores back to full indices
    combined_importances = np.zeros(n_vars)
    combined_importances[best_indices] = spa_importances_reduced

    n_final = int(np.sum(combined_importances > 0))
    print(f"\n=== Fwd iPLS-SPA Final Results ===")
    print(f"iPLS: {n_vars} → {n_ipls_selected} | SPA: → {n_final}")

    return combined_importances


def fipls_cars_selection(X, y, wavelengths, n_intervals=20, max_combine=5,
                         ipls_cv_folds=5, n_iterations=50, pls_components=5,
                         cars_cv_folds=5, monte_carlo_samples=80,
                         random_state=42, task_type='regression'):
    """
    Forward iPLS → CARS hybrid: region selection followed by adaptive variable selection.

    First runs forward iPLS to find the best spectral interval combination, then
    applies CARS on those variables for further refinement.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    n_intervals : int, default=20
        Number of intervals for iPLS
    max_combine : int, default=5
        Max intervals to combine in forward selection
    ipls_cv_folds : int, default=5
        CV folds for iPLS
    n_iterations : int, default=50
        CARS Monte Carlo iterations
    pls_components : int, default=5
        PLS components for CARS
    cars_cv_folds : int, default=5
        CV folds for CARS
    monte_carlo_samples : int, default=80
        CARS Monte Carlo samples
    random_state : int, default=42
        Random seed
    task_type : str, default='regression'
        Task type

    Returns
    -------
    importances : np.ndarray
        Combined importance scores. Shape: (n_features,)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_vars = X.shape

    print(f"\n=== Forward iPLS-CARS Hybrid Selection ===")
    print(f"Starting with {n_vars} variables")

    # Step 1: Forward iPLS
    print(f"\nStep 1: Forward iPLS ({n_intervals} intervals, max_combine={max_combine})...")
    subsets = ipls_forward(
        X, y, wavelengths,
        n_intervals=n_intervals,
        max_combine=max_combine,
        cv_folds=ipls_cv_folds,
        random_state=random_state
    )

    if not subsets:
        print("Warning: Forward iPLS returned no subsets. Returning uniform importances.")
        return np.ones(n_vars)

    # Find best subset (lowest RMSECV)
    best = min(subsets, key=lambda s: s['rmsecv'])
    best_indices = best['indices']
    n_ipls_selected = len(best_indices)
    print(f"Best iPLS subset: {best['tag']} ({n_ipls_selected} vars, RMSECV={best['rmsecv']:.4f})")

    if n_ipls_selected < 10:
        print(f"Warning: Too few iPLS variables ({n_ipls_selected}) for CARS. Returning iPLS-only importances.")
        importances = np.zeros(n_vars)
        importances[best_indices] = 1.0
        return importances

    # Step 2: CARS on iPLS-selected variables
    print(f"\nStep 2: CARS on {n_ipls_selected} iPLS-selected variables...")
    X_ipls_selected = X[:, best_indices]

    cars_importances_reduced = cars_selection(
        X_ipls_selected, y,
        n_iterations=n_iterations,
        pls_components=pls_components,
        cv_folds=cars_cv_folds,
        monte_carlo_samples=monte_carlo_samples,
        random_state=random_state,
        task_type=task_type
    )

    # Map CARS scores back to full indices
    combined_importances = np.zeros(n_vars)
    combined_importances[best_indices] = cars_importances_reduced

    n_final = int(np.sum(combined_importances > 0))
    print(f"\n=== Fwd iPLS-CARS Final Results ===")
    print(f"iPLS: {n_vars} → {n_ipls_selected} | CARS: → {n_final}")

    return combined_importances


def cars_selection(X, y, n_iterations=50, pls_components=5, cv_folds=5,
                   monte_carlo_samples=80, random_state=42, model_type=None,
                   use_hybrid_importance=False, hybrid_importance_weight=0.5, task_type='regression'):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for variable selection.

    CARS is a Monte Carlo-based method that uses an adaptive reweighted
    sampling (ARS) strategy combined with exponential decay to select
    optimal variables. It balances exploration and exploitation.

    Algorithm:
    1. Initialize all variables with equal weights
    2. For each Monte Carlo iteration:
       - Sample variables based on current weights
       - Build model (PLS or LightGBM) and evaluate via cross-validation
       - Update weights based on model coefficients/importances
       - Apply exponential decay to force elimination
    3. Select variables from iteration with lowest RMSECV

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values (required for CARS)
    n_iterations : int, default=50
        Number of Monte Carlo sampling iterations
    pls_components : int, default=5
        Number of PLS components to use in evaluation
    cv_folds : int, default=5
        Number of cross-validation folds
    monte_carlo_samples : int, default=80
        Percentage of variables to sample in each iteration (as integer)
    random_state : int, default=42
        Random seed for reproducibility
    model_type : str, optional
        Model name for model-aware selection. If provided and is a tree model
        (RandomForest, XGBoost, LightGBM, CatBoost), uses LightGBM for evaluation.
        Otherwise uses PLS (default behavior).
    use_hybrid_importance : bool, default=False
        If True, uses hybrid importance (blend of split + gain) for tree models.
        This is the CARS-Tree mode which produces denser importance distributions.
        Only effective when model_type is a tree model.
    hybrid_importance_weight : float, default=0.5
        Weight for blending split-based and gain-based importance.
        Final importance = weight * split_norm + (1-weight) * gain_norm.
        Only used when use_hybrid_importance=True.
    task_type : str, default='regression'
        Task type ('regression' or 'classification'). Determines whether to use
        LGBMRegressor or LGBMClassifier for tree-based evaluation.

    Returns
    -------
    importances : np.ndarray
        Variable importance scores (higher = more important)
        Variables selected in the best iteration get higher scores
        Shape: (n_features,)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import cars_selection
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(50, 100)  # 50 samples, 100 variables
    >>> y = np.random.randn(50)
    >>>
    >>> # Run CARS variable selection
    >>> importances = cars_selection(X, y, n_iterations=50)
    >>>
    >>> # Get top variables
    >>> top_indices = np.argsort(importances)[-50:]
    >>> X_selected = X[:, top_indices]

    References
    ----------
    Li, H. D., et al. (2009). "Key wavelengths screening using competitive
    adaptive reweighted sampling method for multivariate calibration."
    Analytica Chimica Acta, 648(1), 77-84.

    Notes
    -----
    - CARS balances variable selection with prediction performance
    - Computationally more expensive than SPA (Monte Carlo iterations)
    - Often produces very compact variable sets (20-50 variables)
    - Requires target values (supervised selection)
    - Performance depends on good PLS component selection
    """
    # Set random seed
    rng = np.random.RandomState(random_state)

    # Convert inputs to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_variables = X.shape

    # Validate inputs
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    # Graceful degradation for small feature counts
    # CARS needs enough variables for sampling AND PLS/LightGBM model building
    MIN_CARS_FEATURES = 7  # Minimum for meaningful CARS (pls_components=2 + buffer)
    if n_variables < MIN_CARS_FEATURES:
        print(f"WARNING: CARS skipped - only {n_variables} wavelengths available "
              f"(minimum {MIN_CARS_FEATURES} required for meaningful selection). "
              f"Returning uniform importance for all wavelengths.")
        # Return uniform importance (all wavelengths equally important)
        return np.ones(n_variables)

    # Handle None for pls_components (use default)
    if pls_components is None:
        pls_components = 5

    # Auto-adjust pls_components for small datasets
    if pls_components > min(n_samples, n_variables):
        pls_components = min(n_samples // 2, n_variables // 2, 10)
        print(f"Warning: Adjusted pls_components to {pls_components}")

    # Additional safeguard: ensure pls_components is at least 1 and leaves room for sampling
    if pls_components < 1:
        pls_components = 1
    if pls_components >= n_variables:
        pls_components = max(1, n_variables - 2)
        print(f"Warning: Further adjusted pls_components to {pls_components} for small dataset")

    # Adjust cv_folds if needed
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        print(f"Warning: Adjusted cv_folds to {cv_folds}")

    # Determine if using tree-based evaluation (model-aware mode)
    TREE_MODELS = {'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'}
    use_tree_model = model_type in TREE_MODELS if model_type else False

    # CARS-Tree uses hybrid importance (only valid for tree models)
    use_hybrid = use_hybrid_importance and use_tree_model

    if use_tree_model:
        from lightgbm import LGBMRegressor, LGBMClassifier
        if use_hybrid:
            print(f"CARS-Tree: Using hybrid importance (split+gain) for '{model_type}'")
        else:
            print(f"CARS: Using LightGBM-based evaluation for tree model '{model_type}'")
    else:
        print(f"CARS: Using PLS-based evaluation")

    # Initialize weights
    weights = np.ones(n_variables)

    # Storage for history
    rmsecv_history = []
    n_selected_history = []
    selected_vars_history = []

    print(f"CARS: Running {n_iterations} Monte Carlo iterations...")

    # Monte Carlo iterations
    for iteration in range(n_iterations):
        # Exponential decay function for forcing removal
        # r(k) = a * exp(-k/b) where k is iteration
        r = 0.8 * np.exp(-2 * iteration / n_iterations)

        # Number of variables to sample in this iteration
        n_sample = max(int(n_variables * (monte_carlo_samples / 100) * r), pls_components + 1)
        n_sample = min(n_sample, n_variables)

        # Sample variables based on current weights
        # Higher weight = higher probability of selection
        probabilities = weights / weights.sum()
        selected_vars = rng.choice(
            n_variables,
            size=n_sample,
            replace=False,
            p=probabilities
        )
        selected_vars = np.sort(selected_vars)

        X_subset = X[:, selected_vars]

        # Build model and evaluate
        try:
            # Cross-validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_errors = []

            if use_tree_model:
                # LightGBM-based evaluation for tree models
                # CARS-Tree uses enhanced config for more stable importance
                if use_hybrid:
                    if task_type == 'regression':
                        lgb_model = LGBMRegressor(
                            n_estimators=100,
                            max_depth=-1,  # Unlimited, controlled by num_leaves
                            num_leaves=31,
                            min_child_samples=5,
                            subsample=0.8,
                            subsample_freq=1,
                            colsample_bytree=0.8,
                            reg_lambda=1.0,
                            random_state=random_state,
                            verbose=-1,
                            n_jobs=1
                        )
                    else:
                        lgb_model = LGBMClassifier(
                            n_estimators=100,
                            max_depth=-1,  # Unlimited, controlled by num_leaves
                            num_leaves=31,
                            min_child_samples=5,
                            subsample=0.8,
                            subsample_freq=1,
                            colsample_bytree=0.8,
                            reg_lambda=1.0,
                            random_state=random_state,
                            verbose=-1,
                            n_jobs=1
                        )
                else:
                    if task_type == 'regression':
                        lgb_model = LGBMRegressor(
                            n_estimators=50,
                            max_depth=5,
                            random_state=random_state,
                            verbose=-1,
                            n_jobs=1
                        )
                    else:
                        lgb_model = LGBMClassifier(
                            n_estimators=50,
                            max_depth=5,
                            random_state=random_state,
                            verbose=-1,
                            n_jobs=1
                        )

                for train_idx, val_idx in kf.split(X_subset):
                    X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    lgb_model.fit(X_train, y_train)
                    y_pred = lgb_model.predict(X_val)

                    if task_type == 'regression':
                        mse = np.mean((y_val - y_pred.ravel()) ** 2)
                        cv_errors.append(mse)
                    else:
                        # Classification: compute error as 1 - accuracy
                        accuracy = np.mean(y_val == y_pred)
                        cv_errors.append(1.0 - accuracy)

                if task_type == 'regression':
                    rmsecv = np.sqrt(np.mean(cv_errors))
                else:
                    rmsecv = np.mean(cv_errors)  # Mean error for classification

                # Fit on full subset to get feature importances
                lgb_model.fit(X_subset, y)

                if use_hybrid:
                    # CARS-Tree: Compute hybrid importance (split + gain blend)
                    # This produces denser distributions than split-only importance
                    booster = lgb_model.booster_
                    split_imp = booster.feature_importance(importance_type='split').astype(float)
                    gain_imp = booster.feature_importance(importance_type='gain').astype(float)

                    # Normalize each importance type to sum to 1
                    split_norm = split_imp / (split_imp.sum() + 1e-10)
                    gain_norm = gain_imp / (gain_imp.sum() + 1e-10)

                    # Blend with configurable weight
                    feature_imp = (hybrid_importance_weight * split_norm +
                                   (1 - hybrid_importance_weight) * gain_norm)
                else:
                    feature_imp = lgb_model.feature_importances_.astype(float)

                # Add minimum floor to prevent complete elimination of variables
                # Tree models have sparse feature importances (many zeros) which
                # breaks probability sampling in subsequent iterations
                feature_imp = np.maximum(feature_imp, 1e-6)

                # Update weights based on feature importances
                weights[selected_vars] = feature_imp / (feature_imp.sum() + 1e-10)

            else:
                # PLS-based evaluation (default)
                n_comp = min(pls_components, n_sample - 1, X_subset.shape[0] - 1)
                pls = PLSRegression(n_components=n_comp)

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
                weights[selected_vars] = np.abs(coef)

            # Normalize weights to sum to 1
            weights = weights / (weights.sum() + 1e-10)

        except Exception as e:
            # If model fails (e.g., singular matrix), skip this iteration
            rmsecv = np.inf
            print(f"  Iteration {iteration+1}/{n_iterations}: Failed - {str(e)[:50]}")

        # Record history
        rmsecv_history.append(rmsecv)
        n_selected_history.append(n_sample)
        selected_vars_history.append(selected_vars)

        if iteration % 10 == 0 and rmsecv != np.inf:
            print(f"  Iteration {iteration+1}/{n_iterations}: {n_sample} vars, RMSECV={rmsecv:.4f}")

    # Find iteration with lowest RMSECV
    rmsecv_history = np.array(rmsecv_history)
    valid_iterations = ~np.isinf(rmsecv_history)

    if not np.any(valid_iterations):
        raise RuntimeError("CARS failed: no valid iterations")

    best_iteration = np.argmin(rmsecv_history[valid_iterations])
    best_iteration_idx = np.where(valid_iterations)[0][best_iteration]

    selected_indices = selected_vars_history[best_iteration_idx]
    best_rmsecv = rmsecv_history[best_iteration_idx]

    # Create importance scores based on final accumulated weights
    # The weights have been adaptively updated across all iterations
    # Higher weight = variable was consistently important across iterations
    importances = weights.copy()

    # Zero out variables not selected in best iteration
    # This ensures only the best iteration's variables get non-zero importance
    mask = np.zeros(n_variables, dtype=bool)
    mask[selected_indices] = True
    importances[~mask] = 0

    print(f"\nCARS complete:")
    print(f"  Best iteration: {best_iteration_idx+1}/{n_iterations}")
    print(f"  Selected {len(selected_indices)} variables")
    print(f"  Best RMSECV: {best_rmsecv:.4f}")

    return importances


# =============================================================================
# Enhanced iPLS Helper Functions
# =============================================================================

def _create_intervals(wavelengths, n_intervals):
    """
    Create equal-width intervals based on wavelength range.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength values for each feature
    n_intervals : int
        Number of intervals to create

    Returns
    -------
    intervals : list of tuple
        Each tuple contains (start_idx, end_idx, start_wl, end_wl)
    """
    wavelengths = np.asarray(wavelengths)
    n_features = len(wavelengths)

    # Calculate interval size in features
    interval_size = n_features // n_intervals

    intervals = []
    for i in range(n_intervals):
        start_idx = i * interval_size
        if i == n_intervals - 1:
            end_idx = n_features  # Last interval gets remaining features
        else:
            end_idx = (i + 1) * interval_size

        if end_idx > start_idx:
            start_wl = wavelengths[start_idx]
            end_wl = wavelengths[end_idx - 1]  # -1 because end_idx is exclusive
            intervals.append((start_idx, end_idx, start_wl, end_wl))

    return intervals


def _evaluate_interval_pls(X, y, cv_folds):
    """
    Evaluate PLS model on given variables using cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Spectral data subset (n_samples, n_features_subset)
    y : np.ndarray
        Target values
    cv_folds : int
        Number of cross-validation folds

    Returns
    -------
    rmsecv : float
        Cross-validated RMSE
    r2 : float
        Cross-validated R²
    """
    n_samples, n_features = X.shape

    # Auto-select number of components
    n_components = min(10, n_features // 2, n_samples // 2)
    n_components = max(1, n_components)

    try:
        pls = PLSRegression(n_components=n_components, scale=False)
        y_pred = cross_val_predict(pls, X, y, cv=cv_folds)

        rmsecv = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return rmsecv, r2
    except Exception:
        # Return poor scores on failure
        return np.inf, -1.0


def _get_combined_indices(intervals, interval_ids):
    """
    Get combined variable indices for selected intervals.

    Parameters
    ----------
    intervals : list of tuple
        Interval definitions from _create_intervals()
    interval_ids : list of int
        Indices of intervals to combine

    Returns
    -------
    indices : np.ndarray
        Combined variable indices (sorted)
    """
    indices = []
    for interval_id in sorted(interval_ids):
        start_idx, end_idx, _, _ = intervals[interval_id]
        indices.extend(range(start_idx, end_idx))
    return np.array(indices, dtype=int)


def _get_wavelength_ranges(intervals, interval_ids):
    """
    Get wavelength ranges for selected intervals.

    Parameters
    ----------
    intervals : list of tuple
        Interval definitions from _create_intervals()
    interval_ids : list of int
        Indices of selected intervals

    Returns
    -------
    ranges : list of tuple
        Each tuple contains (start_wl, end_wl)
    """
    ranges = []
    for interval_id in sorted(interval_ids):
        _, _, start_wl, end_wl = intervals[interval_id]
        ranges.append((start_wl, end_wl))
    return ranges


# =============================================================================
# Enhanced iPLS Functions
# =============================================================================

def ipls_forward(X, y, wavelengths, n_intervals=20, max_combine=5, cv_folds=5, random_state=42):
    """
    Forward Interval PLS (iPLS) - iteratively adds best intervals.

    This implements the proper iPLS algorithm from Nørgaard et al. (2000).
    The spectrum is divided into intervals, each evaluated independently,
    then intervals are iteratively combined to find the best subset.

    Algorithm:
    1. Divide spectrum into n_intervals equal segments
    2. Evaluate each interval independently using PLS with CV
    3. Rank intervals by RMSECV (lower = better)
    4. Forward selection: start with best, iteratively add intervals that improve model
    5. Return all individual intervals + best combinations

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature (used for labeling)
    n_intervals : int, default=20
        Number of intervals to divide spectrum into
    max_combine : int, default=5
        Maximum number of intervals to combine in forward selection
    cv_folds : int, default=5
        Number of CV folds for evaluation
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    subsets : list of dict
        Each dict contains:
        - 'indices': np.ndarray of variable indices
        - 'tag': str like 'fwd_iPLS_1400-1500nm' or 'fwd_iPLS_2int'
        - 'rmsecv': float (CV RMSE)
        - 'r2': float (CV R²)
        - 'n_intervals': int (number of intervals in this subset)
        - 'interval_ids': list of interval indices

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import ipls_forward
    >>>
    >>> # Generate sample spectral data
    >>> X = np.random.randn(50, 200)  # 50 samples, 200 wavelengths
    >>> y = np.random.randn(50)
    >>> wavelengths = np.linspace(1000, 2500, 200)
    >>>
    >>> # Run forward iPLS
    >>> subsets = ipls_forward(X, y, wavelengths, n_intervals=20, max_combine=5)
    >>>
    >>> # Get best subset (lowest RMSECV)
    >>> best_subset = min(subsets, key=lambda s: s['rmsecv'])
    >>> X_selected = X[:, best_subset['indices']]

    References
    ----------
    Nørgaard, L., et al. (2000). "Interval Partial Least-Squares Regression (iPLS)."
    Applied Spectroscopy 54(3): 413-419.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    wavelengths = np.asarray(wavelengths)

    n_samples, n_features = X.shape

    # Adjust cv_folds if needed
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)

    # Create intervals
    intervals = _create_intervals(wavelengths, n_intervals)

    print(f"Forward iPLS: Evaluating {len(intervals)} intervals...")

    # Step 1: Evaluate each interval independently
    interval_scores = []
    for i, interval in enumerate(intervals):
        start_idx, end_idx, start_wl, end_wl = interval
        X_interval = X[:, start_idx:end_idx]

        rmsecv, r2 = _evaluate_interval_pls(X_interval, y, cv_folds)

        interval_scores.append({
            'interval_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_wl': start_wl,
            'end_wl': end_wl,
            'rmsecv': rmsecv,
            'r2': r2
        })

        print(f"  Interval {i+1}/{len(intervals)} ({start_wl:.0f}-{end_wl:.0f}nm): "
              f"RMSECV={rmsecv:.4f}, R²={r2:.4f}")

    # Sort intervals by RMSECV (lower = better)
    interval_scores.sort(key=lambda x: x['rmsecv'])

    # Build result list
    subsets = []

    # Add all individual intervals (ranked by RMSECV)
    for rank, interval in enumerate(interval_scores):
        indices = np.arange(interval['start_idx'], interval['end_idx'])
        tag = f"fwd_iPLS_{interval['start_wl']:.0f}-{interval['end_wl']:.0f}nm"

        subsets.append({
            'indices': indices,
            'tag': tag,
            'rmsecv': interval['rmsecv'],
            'r2': interval['r2'],
            'n_intervals': 1,
            'interval_ids': [interval['interval_id']],
            'rank': rank + 1
        })

    # Step 2: Forward selection - combine intervals
    print(f"\nForward selection (combining up to {max_combine} intervals)...")

    # Start with best single interval
    selected_intervals = [interval_scores[0]['interval_id']]
    best_rmsecv = interval_scores[0]['rmsecv']

    for n_selected in range(2, min(max_combine + 1, len(intervals) + 1)):
        # Try adding each remaining interval
        best_addition = None
        best_new_rmsecv = best_rmsecv

        remaining = [s for s in interval_scores if s['interval_id'] not in selected_intervals]

        for candidate in remaining:
            # Combine indices
            test_intervals = selected_intervals + [candidate['interval_id']]
            combined_indices = _get_combined_indices(intervals, test_intervals)

            X_combined = X[:, combined_indices]
            rmsecv, r2 = _evaluate_interval_pls(X_combined, y, cv_folds)

            if rmsecv < best_new_rmsecv:
                best_new_rmsecv = rmsecv
                best_addition = {
                    'interval_id': candidate['interval_id'],
                    'rmsecv': rmsecv,
                    'r2': r2,
                    'test_intervals': test_intervals
                }

        if best_addition is None:
            # No improvement, stop
            print(f"  {n_selected} intervals: No improvement, stopping")
            break

        # Add best interval
        selected_intervals.append(best_addition['interval_id'])
        best_rmsecv = best_addition['rmsecv']

        # Create subset entry
        combined_indices = _get_combined_indices(intervals, selected_intervals)
        wl_ranges = _get_wavelength_ranges(intervals, selected_intervals)

        if len(wl_ranges) <= 3:
            wl_str = '+'.join([f"{s:.0f}-{e:.0f}" for s, e in wl_ranges])
            tag = f"fwd_iPLS_{n_selected}int_{wl_str}nm"
        else:
            tag = f"fwd_iPLS_{n_selected}int"

        subsets.append({
            'indices': combined_indices,
            'tag': tag,
            'rmsecv': best_addition['rmsecv'],
            'r2': best_addition['r2'],
            'n_intervals': n_selected,
            'interval_ids': selected_intervals.copy()
        })

        print(f"  {n_selected} intervals: RMSECV={best_addition['rmsecv']:.4f}, "
              f"R²={best_addition['r2']:.4f}")

    print(f"\nForward iPLS complete: {len(subsets)} subsets generated")
    return subsets


def ipls_backward(X, y, wavelengths, n_intervals=20, cv_folds=5, random_state=42, min_intervals=1):
    """
    Backward Interval PLS (biPLS) - iteratively removes worst intervals.

    This implements backward interval PLS from Leardi & Nørgaard (2004).
    Starts with all intervals and iteratively removes intervals one at a time,
    always removing the interval whose removal causes the least degradation
    (or most improvement) in RMSECV.

    Algorithm:
    1. Divide spectrum into n_intervals equal segments
    2. Start with all intervals included (baseline)
    3. At each step, remove the interval whose removal results in lowest RMSECV
    4. Continue until min_intervals remain
    5. Return all steps and mark the one with best RMSECV as optimal

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    n_intervals : int, default=20
        Number of intervals to divide spectrum into
    cv_folds : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    min_intervals : int, default=1
        Minimum number of intervals to keep (stop when reached)

    Returns
    -------
    subsets : list of dict
        Each dict contains:
        - 'indices': np.ndarray of variable indices
        - 'tag': str like 'bwd_iPLS_18int' or 'bwd_iPLS_optimal'
        - 'rmsecv': float
        - 'r2': float
        - 'n_intervals': int
        - 'interval_ids': list of remaining interval indices
        - 'is_optimal': bool (True for the subset with best RMSECV)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.variable_selection import ipls_backward
    >>>
    >>> # Generate sample spectral data
    >>> X = np.random.randn(50, 200)  # 50 samples, 200 wavelengths
    >>> y = np.random.randn(50)
    >>> wavelengths = np.linspace(1000, 2500, 200)
    >>>
    >>> # Run backward iPLS
    >>> subsets = ipls_backward(X, y, wavelengths, n_intervals=20)
    >>>
    >>> # Get optimal subset (best RMSECV along elimination path)
    >>> optimal = [s for s in subsets if s.get('is_optimal', False)][0]

    References
    ----------
    Leardi, R. & Nørgaard, L. (2004). "Sequential application of backward interval
    PLS and genetic algorithms for the selection of relevant spectral regions."
    Journal of Chemometrics.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    wavelengths = np.asarray(wavelengths)

    n_samples, n_features = X.shape

    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)

    # Create intervals
    intervals = _create_intervals(wavelengths, n_intervals)
    actual_n_intervals = len(intervals)

    print(f"Backward iPLS: Starting with {actual_n_intervals} intervals...")

    # Start with all intervals
    remaining_intervals = list(range(actual_n_intervals))

    # Evaluate full model (baseline)
    full_indices = _get_combined_indices(intervals, remaining_intervals)
    full_rmsecv, full_r2 = _evaluate_interval_pls(X[:, full_indices], y, cv_folds)

    print(f"  Full spectrum ({actual_n_intervals} intervals): RMSECV={full_rmsecv:.4f}, R²={full_r2:.4f}")

    subsets = []

    # Add full model as baseline
    wl_ranges_full = _get_wavelength_ranges(intervals, remaining_intervals)
    min_wl = min(s for s, e in wl_ranges_full)
    max_wl = max(e for s, e in wl_ranges_full)

    subsets.append({
        'indices': full_indices.copy(),
        'tag': f"bwd_iPLS_full_{min_wl:.0f}-{max_wl:.0f}nm",
        'rmsecv': full_rmsecv,
        'r2': full_r2,
        'n_intervals': actual_n_intervals,
        'interval_ids': remaining_intervals.copy(),
        'is_optimal': False  # Will be updated later
    })

    # Track best RMSECV for marking optimal
    best_rmsecv_overall = full_rmsecv
    best_subset_idx = 0

    # Iteratively remove intervals until min_intervals remain
    while len(remaining_intervals) > min_intervals:
        # Find the interval whose removal results in lowest RMSECV
        best_removal = None
        best_new_rmsecv = float('inf')

        for interval_id in remaining_intervals:
            # Try removing this interval
            test_intervals = [i for i in remaining_intervals if i != interval_id]
            test_indices = _get_combined_indices(intervals, test_intervals)

            rmsecv, r2 = _evaluate_interval_pls(X[:, test_indices], y, cv_folds)

            # Select the removal that gives lowest RMSECV (best model)
            if rmsecv < best_new_rmsecv:
                best_new_rmsecv = rmsecv
                best_removal = {
                    'removed_id': interval_id,
                    'remaining': test_intervals,
                    'rmsecv': rmsecv,
                    'r2': r2
                }

        if best_removal is None:
            # Should not happen unless all intervals fail, but handle gracefully
            print(f"  {len(remaining_intervals)} intervals: All removals failed, stopping")
            break

        # Apply the removal (always remove something, even if it hurts)
        removed_interval = intervals[best_removal['removed_id']]
        remaining_intervals = best_removal['remaining']

        print(f"  Removed interval {removed_interval[2]:.0f}-{removed_interval[3]:.0f}nm -> "
              f"{len(remaining_intervals)} intervals: RMSECV={best_removal['rmsecv']:.4f}, "
              f"R²={best_removal['r2']:.4f}")

        # Create subset entry
        combined_indices = _get_combined_indices(intervals, remaining_intervals)

        tag = f"bwd_iPLS_{len(remaining_intervals)}int"

        subsets.append({
            'indices': combined_indices,
            'tag': tag,
            'rmsecv': best_removal['rmsecv'],
            'r2': best_removal['r2'],
            'n_intervals': len(remaining_intervals),
            'interval_ids': remaining_intervals.copy(),
            'is_optimal': False
        })

        # Track best
        if best_removal['rmsecv'] < best_rmsecv_overall:
            best_rmsecv_overall = best_removal['rmsecv']
            best_subset_idx = len(subsets) - 1

    # Mark the optimal subset (lowest RMSECV along the path)
    if subsets:
        subsets[best_subset_idx]['is_optimal'] = True

        # Add a final entry with descriptive tag for the optimal subset
        optimal = subsets[best_subset_idx]
        wl_ranges = _get_wavelength_ranges(intervals, optimal['interval_ids'])

        if len(wl_ranges) <= 3:
            wl_str = '+'.join([f"{s:.0f}-{e:.0f}" for s, e in wl_ranges])
            optimal_tag = f"bwd_iPLS_optimal_{wl_str}nm"
        else:
            min_wl = min(s for s, e in wl_ranges)
            max_wl = max(e for s, e in wl_ranges)
            optimal_tag = f"bwd_iPLS_optimal_{min_wl:.0f}-{max_wl:.0f}nm"

        subsets.append({
            'indices': optimal['indices'].copy(),
            'tag': optimal_tag,
            'rmsecv': optimal['rmsecv'],
            'r2': optimal['r2'],
            'n_intervals': optimal['n_intervals'],
            'interval_ids': optimal['interval_ids'].copy(),
            'is_optimal': True
        })

    print(f"\nBackward iPLS complete: {len(subsets)} subsets generated")
    print(f"  Optimal: {subsets[best_subset_idx]['n_intervals']} intervals, "
          f"RMSECV={subsets[best_subset_idx]['rmsecv']:.4f}")
    return subsets


# =============================================================================
# MC-siPLS (Monte Carlo Synergy Interval PLS)
# =============================================================================

def mc_sipls(X, y, wavelengths, n_intervals=20, n_combinations=500,
             max_combine=4, cv_folds=5, random_state=42):
    """
    Monte Carlo Synergy Interval PLS — randomly samples interval combinations
    to find synergistic spectral regions that work better together than individually.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    n_intervals : int
        Number of intervals to divide spectrum into
    n_combinations : int
        Total number of random combinations to evaluate
    max_combine : int
        Maximum number of intervals to combine (2..max_combine)
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    subsets : list[dict]
        Each dict has keys: indices, tag, rmsecv, r2, n_intervals, interval_ids, rank
    """
    from math import comb

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    wavelengths = np.asarray(wavelengths)
    n_features = X.shape[1]

    # Clamp n_intervals so each interval has at least 1 feature
    n_intervals = min(n_intervals, n_features)

    intervals = _create_intervals(wavelengths, n_intervals)
    n_actual = len(intervals)

    if n_actual < 2:
        print("MC-siPLS: Not enough intervals (need >= 2), skipping")
        return []

    # Clamp max_combine to available intervals
    max_combine = min(max_combine, n_actual)
    if max_combine < 2:
        max_combine = 2

    rng = np.random.RandomState(random_state)

    # Distribute combinations across sizes 2..max_combine
    sizes = list(range(2, max_combine + 1))
    combos_per_size = max(1, n_combinations // len(sizes))

    seen = set()
    results = []

    print(f"\nMC-siPLS: {n_actual} intervals, sampling up to {n_combinations} combinations "
          f"(sizes {sizes[0]}-{sizes[-1]})")

    for size in sizes:
        # Cap at the actual number of possible combos C(n_actual, size)
        max_possible = comb(n_actual, size)
        budget = min(combos_per_size, max_possible)
        found = 0

        for _ in range(budget * 3):  # allow retries for hash collisions
            if found >= budget:
                break
            combo = tuple(sorted(rng.choice(n_actual, size=size, replace=False)))
            if combo in seen:
                continue
            seen.add(combo)
            found += 1

            combo_list = list(combo)
            combined_indices = _get_combined_indices(intervals, combo_list)

            if len(combined_indices) < 2:
                continue

            rmsecv, r2 = _evaluate_interval_pls(X[:, combined_indices], y, cv_folds)

            if np.isinf(rmsecv):
                continue

            wl_ranges = _get_wavelength_ranges(intervals, combo_list)
            if len(wl_ranges) <= 3:
                wl_str = '+'.join([f"{s:.0f}-{e:.0f}" for s, e in wl_ranges])
                tag = f"siPLS_{size}int_{wl_str}nm"
            else:
                min_wl = min(s for s, e in wl_ranges)
                max_wl = max(e for s, e in wl_ranges)
                tag = f"siPLS_{size}int_{min_wl:.0f}-{max_wl:.0f}nm"

            results.append({
                'indices': combined_indices,
                'tag': tag,
                'rmsecv': rmsecv,
                'r2': r2,
                'n_intervals': size,
                'interval_ids': combo_list,
            })

    # Sort by RMSECV and assign ranks
    results.sort(key=lambda s: s['rmsecv'])
    for i, s in enumerate(results):
        s['rank'] = i + 1

    print(f"MC-siPLS complete: {len(results)} unique subsets evaluated")
    if results:
        print(f"  Best: {results[0]['tag']}, RMSECV={results[0]['rmsecv']:.4f}")
    return results


# =============================================================================
# MWPLS (Moving Window PLS)
# =============================================================================

def mwpls(X, y, wavelengths, window_sizes=None, step_size=None, cv_folds=5):
    """
    Moving Window PLS — slides windows of different sizes across the spectrum
    to find optimal contiguous regions.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values for each feature
    window_sizes : list[int] or None
        Window sizes (in number of variables). None = auto-select based on feature count.
    step_size : int or None
        Step size for sliding. None = window_size // 4 for each window.
    cv_folds : int
        Number of cross-validation folds

    Returns
    -------
    subsets : list[dict]
        Each dict has keys: indices, tag, rmsecv, r2, n_intervals, interval_ids, rank
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    wavelengths = np.asarray(wavelengths)
    n_features = X.shape[1]

    # Auto-select window sizes if not provided
    if window_sizes is None:
        candidates = [10, 20, 40, 80]
        window_sizes = [w for w in candidates if w < n_features]
        if not window_sizes:
            window_sizes = [max(2, n_features // 4)]

    results = []

    # Filter out invalid window sizes
    valid_sizes = [w for w in window_sizes if 2 <= w < n_features]
    if not valid_sizes:
        print(f"\nMWPLS: No valid window sizes for {n_features} features "
              f"(requested: {window_sizes}), skipping")
        return []

    print(f"\nMWPLS: {n_features} features, window sizes: {valid_sizes}")

    for wsize in valid_sizes:

        step = step_size if step_size is not None else max(1, wsize // 4)

        start = 0
        while start + wsize <= n_features:
            end = start + wsize
            indices = np.arange(start, end, dtype=int)

            rmsecv, r2 = _evaluate_interval_pls(X[:, indices], y, cv_folds)

            if not np.isinf(rmsecv):
                start_wl = wavelengths[start]
                end_wl = wavelengths[end - 1]
                tag = f"MWPLS_w{wsize}_{start_wl:.0f}-{end_wl:.0f}nm"

                results.append({
                    'indices': indices,
                    'tag': tag,
                    'rmsecv': rmsecv,
                    'r2': r2,
                    'n_intervals': 1,
                    'interval_ids': [],
                })

            start += step

    # Sort by RMSECV and assign ranks
    results.sort(key=lambda s: s['rmsecv'])
    for i, s in enumerate(results):
        s['rank'] = i + 1

    print(f"MWPLS complete: {len(results)} windows evaluated")
    if results:
        print(f"  Best: {results[0]['tag']}, RMSECV={results[0]['rmsecv']:.4f}")
    return results
