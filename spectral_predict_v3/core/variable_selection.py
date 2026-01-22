"""
Variable selection methods for spectral analysis (v3 standalone).

This module implements various variable selection algorithms to identify
the most informative spectral variables for prediction.

Forked from v1 - standalone implementation for v3's numpy-first approach.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score

# Optional GA-PLS import (requires ga_pls.py module)
try:
    from .ga_pls import ga_pls_selection
    GA_PLS_AVAILABLE = True
except ImportError:
    GA_PLS_AVAILABLE = False


def compute_vip(X, y, n_components=None):
    """
    Compute VIP (Variable Importance in Projection) scores for PLS variable selection.

    VIP measures each variable's contribution to the PLS model. Variables with
    VIP > 1 are typically considered important. This is the chemometrics standard
    for variable selection with PLS models.

    Algorithm:
    1. Fit PLS model on X, y
    2. Extract x_weights (W) and x_scores (T) from fitted model
    3. Compute explained variance per component
    4. Calculate VIP as weighted sum of squared weights

    VIP formula:
        VIP_j = sqrt(p * sum_a(W_ja^2 * SSY_a) / SSY_total)

    where:
        - p = number of variables
        - W_ja = weight of variable j for component a
        - SSY_a = variance explained by component a
        - SSY_total = total variance explained

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    n_components : int or None
        Number of PLS components. If None, auto-select as min(10, n_features//2, n_samples//2)

    Returns
    -------
    vip_scores : np.ndarray
        VIP score for each variable (higher = more important)
        Shape: (n_features,)

    References
    ----------
    Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool
    of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 100)
    >>> y = np.random.randn(50)
    >>> vip_scores = compute_vip(X, y)
    >>> # Select variables with VIP > 1
    >>> important_vars = np.where(vip_scores > 1.0)[0]
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    # Auto-select n_components if not provided
    if n_components is None:
        n_components = min(10, n_features // 2, n_samples // 2)
    n_components = max(1, n_components)

    # Fit PLS model
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)

    # Get PLS components
    W = pls.x_weights_  # (n_features, n_components)
    T = pls.x_scores_   # (n_samples, n_components)

    # Get explained variance by each component
    # SSY: sum of squares of y explained by each component
    y_reshaped = y.reshape(-1, 1)
    ssy_comp = np.sum(T**2, axis=0) * np.var(y_reshaped, axis=0)

    # Total SSY
    ssy_total = np.sum(ssy_comp)

    if ssy_total < 1e-10:
        # If no variance explained, return uniform importances
        return np.ones(n_features)

    # VIP calculation (vectorized for performance)
    # Sum over components for each feature
    weight = np.sum((W ** 2) * ssy_comp, axis=1)
    vip_scores = np.sqrt(n_features * weight / ssy_total)

    return vip_scores


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

    # Handle edge case: adjust cv_folds if n_samples is too small
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)

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

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_idx = 0

    for train_idx, _ in kfold.split(X_augmented):
        # Get training data for this fold
        X_train = X_augmented[train_idx]
        y_train = y[train_idx]

        # Fit PLS model
        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_train, y_train)

            # Extract coefficients (first column if y is 1D)
            if pls.coef_.ndim == 2:
                coefficients[fold_idx] = pls.coef_[:, 0]
            else:
                coefficients[fold_idx] = pls.coef_

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

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_idx = 0

    for train_idx, _ in kfold.split(X_augmented):
        X_train = X_augmented[train_idx]
        y_train = y[train_idx]

        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_train, y_train)

            if pls.coef_.ndim == 2:
                coefficients[fold_idx] = pls.coef_[:, 0]
            else:
                coefficients[fold_idx] = pls.coef_

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


def spa_selection(X, y, n_features, n_random_starts=10, cv_folds=5, random_state=42):
    """
    Successive Projections Algorithm (SPA) - selects minimally correlated variables.

    SPA reduces collinearity by iteratively selecting variables that have minimum
    projection (correlation) onto the already-selected variable set. This creates
    a set of maximally uncorrelated features.

    Algorithm:
    1. For each random start:
       a. Select initial variable (max correlation with y, or random)
       b. Iteratively select variable with MINIMUM projection onto selected set
       c. Projection = sum of squared correlations with already-selected variables
       d. Evaluate selection quality using PLS R² via CV
    2. Return best selection across all starts
    3. Convert to importance scores (earlier selected = higher score)

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    n_features : int
        Number of features to select
    n_random_starts : int, default=10
        Number of random initializations
    cv_folds : int, default=5
        Number of CV folds for quality evaluation
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    importances : np.ndarray
        Importance scores (higher = earlier selected = more important)
        Shape: (X.shape[1],)

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

    # Handle edge case: if requesting more features than available, use all
    if n_features > n_vars:
        print(f"Warning: n_features ({n_features}) > n_vars ({n_vars}). Using all features.")
        n_features = n_vars

    # Handle edge case: reduce cv_folds if not enough samples
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        print(f"Warning: Insufficient samples. Reducing cv_folds to {cv_folds}")

    # Step 1: Normalize X for correlation computation (zero mean, unit variance)
    # This makes dot products equivalent to correlations
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10  # Add small value to avoid division by zero
    X_norm = (X - X_mean) / X_std

    # Normalize y for correlation computation
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-10
    y_norm = (y - y_mean) / y_std

    # Compute initial correlations with y (for initialization)
    # corr(X[:, j], y) = (X_norm[:, j] @ y_norm) / n_samples
    initial_corrs = np.abs(X_norm.T @ y_norm) / n_samples

    # Track best selection across random starts
    best_score = -np.inf
    best_selection = None

    print(f"Running SPA with {n_random_starts} random starts...")

    # Step 2: Run multiple random starts
    for start_idx in range(n_random_starts):
        # Initialize: select variable with max correlation with y
        selected_indices = []
        available_indices = set(range(n_vars))

        # First variable: highest correlation with y
        first_var = np.argmax(initial_corrs)
        selected_indices.append(first_var)
        available_indices.remove(first_var)

        # Iteratively select remaining variables (n_features - 1 more)
        for step in range(1, n_features):
            # Compute projections for all available variables
            # Projection = sum of squared correlations with already-selected variables
            projections = np.zeros(n_vars)

            # Vectorized computation of correlations
            # For each unselected variable j, compute corr² with all selected variables
            # Extract selected columns as a 2D array
            X_selected_norm = X_norm[:, selected_indices]
            if X_selected_norm.ndim == 1:
                X_selected_norm = X_selected_norm.reshape(-1, 1)

            for j in available_indices:
                # Correlation with selected variables
                # corr(X[:, j], X[:, i]) = (X_norm[:, j] @ X_norm[:, i]) / n_samples
                corrs_with_selected = X_norm[:, j] @ X_selected_norm / n_samples
                # Projection = sum of squared correlations
                projections[j] = np.sum(corrs_with_selected ** 2)

            # Select variable with MINIMUM projection (least correlated with selected set)
            # Only consider available indices
            min_proj_var = None
            min_proj = np.inf
            for j in available_indices:
                if projections[j] < min_proj:
                    min_proj = projections[j]
                    min_proj_var = j

            selected_indices.append(min_proj_var)
            available_indices.remove(min_proj_var)

        # Step 3: Evaluate this selection using PLS with cross-validation
        try:
            # Extract selected features from original (non-normalized) data
            X_selected = X[:, selected_indices]

            # Fit PLS and compute CV R²
            # Use n_components = min(n_features, n_samples-1) to avoid overfitting
            n_components = min(n_features, n_samples - 1, 10)
            pls = PLSRegression(n_components=n_components, scale=False)

            # Cross-validation score (R²)
            cv_scores = cross_val_score(
                pls, X_selected, y,
                cv=cv_folds,
                scoring='r2',
                n_jobs=1
            )
            mean_score = np.mean(cv_scores)

            # Track best selection (skip if score is NaN or -inf)
            if not np.isnan(mean_score) and not np.isinf(mean_score):
                if mean_score > best_score:
                    best_score = mean_score
                    best_selection = selected_indices.copy()
                    print(f"  Start {start_idx+1}/{n_random_starts}: R² = {mean_score:.4f} (new best)")
                else:
                    print(f"  Start {start_idx+1}/{n_random_starts}: R² = {mean_score:.4f}")
            else:
                print(f"  Start {start_idx+1}/{n_random_starts}: R² = {mean_score:.4f} (invalid)")

        except Exception as e:
            print(f"  Start {start_idx+1}/{n_random_starts}: Failed - {str(e)}")
            continue

    # Step 4: Convert best selection to importance scores
    # Earlier selected = higher importance
    importances = np.zeros(n_vars)
    if best_selection is not None:
        for rank, var_idx in enumerate(best_selection):
            # Assign scores: first selected gets n_features, last gets 1
            importances[var_idx] = n_features - rank
    else:
        print("Warning: All random starts failed. Returning uniform importances.")
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
        Random seed for reproducibility

    Returns
    -------
    importances : np.ndarray
        Importance scores based on interval performance
        Variables in better intervals receive higher scores
        Shape: (n_features,)

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

    # Handle edge case: adjust cv_folds if n_samples is too small
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        print(f"Warning: Insufficient samples. Reducing cv_folds to {cv_folds}")

    # Handle edge case: if too many intervals requested, reduce to n_features
    if n_intervals > n_features:
        n_intervals = n_features
        print(f"Warning: n_intervals > n_features. Reducing to {n_intervals} intervals")

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
                n_jobs=1
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
                      spa_n_random_starts=10, spa_cv_folds=5, random_state=42):
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
    spa_n_random_starts : int, default=10
        Number of random starts for SPA
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
        n_random_starts=spa_n_random_starts,
        cv_folds=spa_cv_folds,
        random_state=random_state
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


# =============================================================================
# INTERVAL PLS (iPLS) - Proper Implementation
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

    References
    ----------
    Nørgaard, L., et al. (2000). "Interval Partial Least-Squares Regression (iPLS)."
    Applied Spectroscopy 54(3): 413-419.
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score

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


def _create_intervals(wavelengths, n_intervals):
    """
    Create equal-width intervals based on wavelength range.

    Returns list of (start_idx, end_idx, start_wl, end_wl) tuples.
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

    Returns (rmsecv, r2).
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score

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
    except Exception as e:
        # Return poor scores on failure
        return np.inf, -1.0


def _get_combined_indices(intervals, interval_ids):
    """
    Get combined variable indices for selected intervals.
    """
    indices = []
    for interval_id in sorted(interval_ids):
        start_idx, end_idx, _, _ = intervals[interval_id]
        indices.extend(range(start_idx, end_idx))
    return np.array(indices, dtype=int)


def _get_wavelength_ranges(intervals, interval_ids):
    """
    Get wavelength ranges for selected intervals.

    Returns list of (start_wl, end_wl) tuples.
    """
    ranges = []
    for interval_id in sorted(interval_ids):
        _, _, start_wl, end_wl = intervals[interval_id]
        ranges.append((start_wl, end_wl))
    return ranges


def cars_selection(
    X, y,
    n_iterations=50,
    pls_components=5,
    cv_folds=5,
    monte_carlo_samples=80,
    random_state=42
):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for variable selection.

    CARS is a Monte Carlo-based method that uses an adaptive reweighted
    sampling (ARS) strategy combined with exponential decay to select
    optimal variables. It balances exploration and exploitation.

    Algorithm:
    1. Initialize all variables with equal weights
    2. For each Monte Carlo iteration:
       - Sample variables based on current weights
       - Build PLS model and evaluate via cross-validation
       - Update weights based on PLS regression coefficients
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

    Returns
    -------
    importances : np.ndarray
        Variable importance scores (higher = more important)
        Variables selected in the best iteration get higher scores
        Shape: (n_features,)

    References
    ----------
    Li, H. D., et al. (2009). "Key wavelengths screening using competitive
    adaptive reweighted sampling method for multivariate calibration."
    Analytica Chimica Acta, 648(1), 77-84.

    Notes
    -----
    - CARS balances variable selection with prediction performance
    - Computationally more expensive than SPA (Monte Carlo iterations)
    - Often produces very compact variable sets
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
    if pls_components > min(n_samples, n_variables):
        pls_components = min(n_samples // 2, n_variables // 2, 10)
        print(f"Warning: Adjusted pls_components to {pls_components}")

    # Adjust cv_folds if needed
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        print(f"Warning: Adjusted cv_folds to {cv_folds}")

    # Initialize weights
    weights = np.ones(n_variables)

    # Storage for history
    rmsecv_history = []
    n_selected_history = []
    weights_history = []
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

        # Build PLS model and evaluate
        try:
            n_comp = min(pls_components, n_sample - 1, X_subset.shape[0] - 1)
            pls = PLSRegression(n_components=n_comp)

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
            print(f"  Iteration {iteration+1}/{n_iterations}: Failed - {str(e)[:50]}")

        # Record history
        rmsecv_history.append(rmsecv)
        n_selected_history.append(n_sample)
        weights_history.append(weights.copy())
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

    # Create importance scores
    # Variables selected in best iteration get scores based on their selection order
    importances = np.zeros(n_variables)
    for rank, var_idx in enumerate(selected_indices):
        # Assign scores: first selected gets n_sample, last gets 1
        importances[var_idx] = len(selected_indices) - rank

    print(f"\nCARS complete:")
    print(f"  Best iteration: {best_iteration_idx+1}/{n_iterations}")
    print(f"  Selected {len(selected_indices)} variables")
    print(f"  Best RMSECV: {best_rmsecv:.4f}")

    return importances
