"""
Variable selection methods for spectral analysis.

This module implements various variable selection algorithms to identify
the most informative spectral variables for prediction.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score


def uve_selection(X, y, cutoff_multiplier=1.0, n_components=None, cv_folds=5):
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
    noise_variables = np.random.randn(n_samples, n_features)
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
    # Note: We return the raw reliability scores, not a binary mask
    # This allows the caller to decide on filtering based on the threshold
    importances = real_reliability

    # Handle edge case: if all variables would be eliminated (all scores are 0)
    if np.all(importances == 0):
        # Return uniform scores so no variables are preferentially eliminated
        importances = np.ones(n_features)

    return importances


def get_uve_threshold(X, y, cutoff_multiplier=1.0, n_components=None, cv_folds=5):
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
    noise_variables = np.random.randn(n_samples, n_features)
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


def spa_selection(X, y, n_features, n_random_starts=10, cv_folds=5):
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


def ipls_selection(X, y, n_intervals=20, n_components=None, cv_folds=5):
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
                      spa_n_random_starts=10, spa_cv_folds=5):
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
        cv_folds=uve_cv_folds
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
        cv_folds=spa_cv_folds
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
