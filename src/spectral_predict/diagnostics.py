"""
Model diagnostics utilities for spectral analysis.

Provides:
- Residual analysis (compute_residuals)
- Leverage detection (compute_leverage)
- Prediction intervals (jackknife_prediction_intervals)
- Q-Q plot data generation (qq_plot_data)

References
----------
- Weisberg, S. (2005). Applied Linear Regression. Wiley.
- Fox, J. (2008). Applied Regression Analysis and Generalized Linear Models. Sage.
- Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
"""

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline


def compute_residuals(y_true, y_pred):
    """
    Compute residuals for regression models.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values
    y_pred : array-like, shape (n_samples,)
        Predicted values

    Returns
    -------
    residuals : ndarray
        y_true - y_pred
    standardized_residuals : ndarray
        Residuals divided by their standard deviation
    """
    residuals = np.array(y_true) - np.array(y_pred)
    std_resid = residuals / np.std(residuals) if np.std(residuals) > 1e-10 else residuals
    return residuals, std_resid


def compute_leverage(X, return_threshold=True):
    """
    Compute leverage (hat values) for samples.

    Leverage h_ii = diag(X(X'X)^-1X')
    High leverage points have h_ii > 2p/n or 3p/n (thresholds)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (preprocessed data used for model fitting)
    return_threshold : bool, default=True
        If True, also return leverage threshold (2p/n)

    Returns
    -------
    leverage : ndarray, shape (n_samples,)
        Hat values for each sample
    threshold_2p : float (optional)
        2p/n threshold for moderate leverage

    Notes
    -----
    For large n_features, uses SVD-based approach for numerical stability.
    """
    X = np.asarray(X)
    n, p = X.shape

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])
    p_with_intercept = p + 1

    # Use SVD for numerical stability when p is large
    if p > 100 or n <= p_with_intercept:
        # H = U @ U.T where U comes from SVD
        U, s, Vt = np.linalg.svd(X_with_intercept, full_matrices=False)
        leverage = np.sum(U**2, axis=1)
    else:
        # Standard formula: H = X(X'X)^-1X'
        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            H = X_with_intercept @ XtX_inv @ X_with_intercept.T
            leverage = np.diag(H)
        except np.linalg.LinAlgError:
            # Fallback to SVD if matrix is singular
            U, s, Vt = np.linalg.svd(X_with_intercept, full_matrices=False)
            leverage = np.sum(U**2, axis=1)

    if return_threshold:
        threshold_2p = 2 * p_with_intercept / n
        return leverage, threshold_2p

    return leverage


def qq_plot_data(residuals):
    """
    Compute Q-Q plot coordinates for normality assessment.

    Parameters
    ----------
    residuals : array-like
        Model residuals

    Returns
    -------
    theoretical_quantiles : ndarray
        Expected quantiles from normal distribution
    sample_quantiles : ndarray
        Observed quantiles from residuals (sorted)
    """
    from scipy import stats

    residuals = np.asarray(residuals)
    sample_quantiles = np.sort(residuals)

    # Compute theoretical quantiles
    n = len(residuals)
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(1/(n+1), n/(n+1), n)
    )

    return theoretical_quantiles, sample_quantiles


def jackknife_prediction_intervals(model, X_train, y_train, X_test, confidence=0.95):
    """
    Compute prediction intervals using jack-knife (leave-one-out) resampling.

    Faster than bootstrap for small-to-moderate sample sizes.
    Suitable for PLS regression models.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Fitted model or pipeline (e.g., PLSRegression or Pipeline with preprocessing)
        CRITICAL: Pass the entire pipeline, not just the extracted model.
        This ensures preprocessing is applied correctly during jackknife resampling.
    X_train : array-like, shape (n_train, n_features)
        Training features
    y_train : array-like, shape (n_train,)
        Training targets
    X_test : array-like, shape (n_test, n_features)
        Test features for prediction
    confidence : float, default=0.95
        Confidence level (0.95 = 95% interval)

    Returns
    -------
    predictions : ndarray, shape (n_test,)
        Point predictions for X_test
    lower_bounds : ndarray, shape (n_test,)
        Lower confidence bounds
    upper_bounds : ndarray, shape (n_test,)
        Upper confidence bounds
    std_errors : ndarray, shape (n_test,)
        Standard errors of predictions

    Notes
    -----
    Uses delete-1 jackknife:
    1. For each training sample i, fit model on data excluding sample i
    2. Predict on X_test with this model
    3. Compute variance across jackknife replications
    4. Construct intervals using t-distribution

    Computational cost: O(n_train * fit_time)
    WARNING: Can be slow for n_train > 200
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).flatten()
    X_test = np.asarray(X_test)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Get point predictions from full model
    predictions = model.predict(X_test).flatten()

    # Jackknife resampling: leave-one-out predictions
    jackknife_preds = np.zeros((n_train, n_test))

    for i in range(n_train):
        # Create leave-one-out dataset
        mask = np.ones(n_train, dtype=bool)
        mask[i] = False

        X_loo = X_train[mask]
        y_loo = y_train[mask]

        # Clone and fit model
        model_loo = clone(model)
        model_loo.fit(X_loo, y_loo)

        # Predict on test set
        jackknife_preds[i, :] = model_loo.predict(X_test).flatten()

    # Compute jackknife variance
    # Variance = (n-1)/n * sum((theta_i - theta_mean)^2)
    mean_preds = np.mean(jackknife_preds, axis=0)
    jackknife_var = ((n_train - 1) / n_train) * np.sum(
        (jackknife_preds - mean_preds)**2, axis=0
    )
    std_errors = np.sqrt(jackknife_var)

    # Construct confidence intervals using t-distribution
    # df = n_train - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n_train - 1)

    lower_bounds = predictions - t_critical * std_errors
    upper_bounds = predictions + t_critical * std_errors

    return predictions, lower_bounds, upper_bounds, std_errors
