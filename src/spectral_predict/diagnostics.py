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


# =============================================================================
# Model Complexity Analysis Functions
# =============================================================================

def compute_pls_complexity_curve(X, y, max_components, cv, task='regression', base_params=None):
    """
    Compute RMSEC/RMSECV or error rate at each n_components for PLS models.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (already preprocessed)
    y : array-like, shape (n_samples,)
        Target values
    max_components : int
        Maximum number of components to evaluate
    cv : int or cross-validation generator
        Cross-validation splitting strategy
    task : str, default='regression'
        'regression' or 'classification'
    base_params : dict, optional
        Additional parameters for PLSRegression

    Returns
    -------
    dict with keys:
        - param_values: list of n_components values tested
        - train_scores: array of training scores (RMSE or 1-accuracy)
        - cv_scores: array of CV scores
        - cv_std: array of CV score standard deviations
        - param_name: 'n_components'
        - metric_name: 'RMSE' or '1-Accuracy'
        - optimal_idx: index of optimal complexity (min CV error)
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    # Determine valid range for n_components
    max_valid = min(max_components, n_features - 1, n_samples - 1)
    param_values = list(range(1, max_valid + 1))

    if len(param_values) == 0:
        return {
            'param_values': [1],
            'train_scores': [np.nan],
            'cv_scores': [np.nan],
            'cv_std': [np.nan],
            'param_name': 'n_components',
            'metric_name': 'RMSE' if task == 'regression' else '1-Accuracy',
            'optimal_idx': 0
        }

    train_scores = []
    cv_scores = []
    cv_std = []

    base_params = base_params or {}

    for n_comp in param_values:
        pls = PLSRegression(n_components=n_comp, scale=False, **base_params)

        if task == 'regression':
            # Training score
            pls.fit(X, y)
            y_train_pred = pls.predict(X).ravel()
            train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
            train_scores.append(train_rmse)

            # CV score
            y_cv_pred = cross_val_predict(pls, X, y, cv=cv)
            cv_rmse = np.sqrt(mean_squared_error(y, y_cv_pred))
            cv_scores.append(cv_rmse)

            # Compute per-fold std (approximate from aggregated prediction)
            fold_errors = []
            if hasattr(cv, 'split'):
                for train_idx, test_idx in cv.split(X, y):
                    fold_pls = clone(pls)
                    fold_pls.fit(X[train_idx], y[train_idx])
                    fold_pred = fold_pls.predict(X[test_idx]).ravel()
                    fold_rmse = np.sqrt(mean_squared_error(y[test_idx], fold_pred))
                    fold_errors.append(fold_rmse)
                cv_std.append(np.std(fold_errors))
            else:
                cv_std.append(0.0)
        else:
            # Classification: PLS + LogisticRegression
            pls.fit(X, y)
            X_scores_train = pls.transform(X)
            scaler = StandardScaler()
            X_scores_scaled = scaler.fit_transform(X_scores_train)

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_scores_scaled, y)
            y_train_pred = lr.predict(X_scores_scaled)
            train_acc = accuracy_score(y, y_train_pred)
            train_scores.append(1 - train_acc)  # Error rate

            # CV score
            fold_errors = []
            if hasattr(cv, 'split'):
                for train_idx, test_idx in cv.split(X, y):
                    fold_pls = PLSRegression(n_components=n_comp, scale=False, **base_params)
                    fold_pls.fit(X[train_idx], y[train_idx])
                    X_scores_train_fold = fold_pls.transform(X[train_idx])
                    X_scores_test_fold = fold_pls.transform(X[test_idx])

                    fold_scaler = StandardScaler()
                    X_scores_train_scaled = fold_scaler.fit_transform(X_scores_train_fold)
                    X_scores_test_scaled = fold_scaler.transform(X_scores_test_fold)

                    fold_lr = LogisticRegression(max_iter=1000, random_state=42)
                    fold_lr.fit(X_scores_train_scaled, y[train_idx])
                    fold_pred = fold_lr.predict(X_scores_test_scaled)
                    fold_errors.append(1 - accuracy_score(y[test_idx], fold_pred))

                cv_scores.append(np.mean(fold_errors))
                cv_std.append(np.std(fold_errors))
            else:
                cv_scores.append(np.nan)
                cv_std.append(0.0)

    train_scores = np.array(train_scores)
    cv_scores = np.array(cv_scores)
    cv_std = np.array(cv_std)

    # Find optimal (minimum CV error)
    valid_idx = np.where(~np.isnan(cv_scores))[0]
    if len(valid_idx) > 0:
        optimal_idx = valid_idx[np.argmin(cv_scores[valid_idx])]
    else:
        optimal_idx = 0

    return {
        'param_values': param_values,
        'train_scores': train_scores,
        'cv_scores': cv_scores,
        'cv_std': cv_std,
        'param_name': 'n_components',
        'metric_name': 'RMSE' if task == 'regression' else '1-Accuracy',
        'optimal_idx': optimal_idx
    }


def compute_sklearn_validation_curve(estimator, X, y, param_name, param_range, cv,
                                     task='regression', scoring=None):
    """
    Compute validation curve using sklearn's validation_curve.

    Wrapper for sklearn.model_selection.validation_curve with error handling.

    Parameters
    ----------
    estimator : sklearn estimator
        Model to evaluate
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target values
    param_name : str
        Name of hyperparameter to vary
    param_range : array-like
        Values of hyperparameter to evaluate
    cv : int or cross-validation generator
        Cross-validation splitting strategy
    task : str, default='regression'
        'regression' or 'classification'
    scoring : str, optional
        Scoring metric (default: neg_root_mean_squared_error for regression)

    Returns
    -------
    dict with keys:
        - param_values: list of parameter values tested
        - train_scores: array of training scores (converted to RMSE or error)
        - cv_scores: array of CV scores
        - cv_std: array of CV score standard deviations
        - param_name: parameter name
        - metric_name: 'RMSE' or '1-Accuracy'
        - optimal_idx: index of optimal complexity
    """
    from sklearn.model_selection import validation_curve

    if scoring is None:
        if task == 'regression':
            scoring = 'neg_root_mean_squared_error'
        else:
            scoring = 'accuracy'

    try:
        train_scores_raw, cv_scores_raw = validation_curve(
            estimator, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        # Convert scores (sklearn returns negated RMSE for regression)
        if task == 'regression':
            train_scores = -np.mean(train_scores_raw, axis=1)  # Convert to positive RMSE
            cv_scores = -np.mean(cv_scores_raw, axis=1)
            cv_std = np.std(-cv_scores_raw, axis=1)
            metric_name = 'RMSE'
        else:
            train_scores = 1 - np.mean(train_scores_raw, axis=1)  # Convert to error rate
            cv_scores = 1 - np.mean(cv_scores_raw, axis=1)
            cv_std = np.std(1 - cv_scores_raw, axis=1)
            metric_name = '1-Accuracy'

        # Find optimal
        valid_idx = np.where(~np.isnan(cv_scores))[0]
        if len(valid_idx) > 0:
            optimal_idx = valid_idx[np.argmin(cv_scores[valid_idx])]
        else:
            optimal_idx = 0

        return {
            'param_values': list(param_range),
            'train_scores': train_scores,
            'cv_scores': cv_scores,
            'cv_std': cv_std,
            'param_name': param_name,
            'metric_name': metric_name,
            'optimal_idx': optimal_idx
        }

    except Exception as e:
        print(f"Warning: validation_curve failed: {e}")
        return {
            'param_values': list(param_range),
            'train_scores': np.full(len(param_range), np.nan),
            'cv_scores': np.full(len(param_range), np.nan),
            'cv_std': np.zeros(len(param_range)),
            'param_name': param_name,
            'metric_name': 'RMSE' if task == 'regression' else '1-Accuracy',
            'optimal_idx': 0
        }


def compute_ensemble_validation_curve(model_type, X, y, base_params, cv,
                                       task='regression', n_points=8):
    """
    Compute validation curve for tree-based ensemble models (n_estimators sweep).

    Parameters
    ----------
    model_type : str
        One of: 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target values
    base_params : dict
        Base hyperparameters for the model (including the selected n_estimators)
    cv : cross-validation generator
        Cross-validation splitting strategy
    task : str, default='regression'
        'regression' or 'classification'
    n_points : int, default=8
        Number of points in the sweep (from 10% to 150% of base n_estimators)

    Returns
    -------
    dict with keys:
        - param_values: list of n_estimators values tested
        - train_scores: array of training scores
        - cv_scores: array of CV scores
        - cv_std: array of CV score standard deviations
        - param_name: 'n_estimators' or 'iterations'
        - metric_name: 'RMSE' or '1-Accuracy'
        - optimal_idx: index of optimal complexity
        - selected_idx: index of the selected/base value
    """
    from sklearn.metrics import mean_squared_error, accuracy_score

    # Get base n_estimators value
    if model_type == 'CatBoost':
        param_name = 'iterations'
        base_n = base_params.get('iterations', 100)
    else:
        param_name = 'n_estimators'
        base_n = base_params.get('n_estimators', 100)

    # Generate sweep range (10% to 150% of base)
    min_n = max(10, int(base_n * 0.1))
    max_n = int(base_n * 1.5)

    # Create log-spaced points for smoother curve
    param_range = np.unique(np.logspace(
        np.log10(min_n), np.log10(max_n), n_points
    ).astype(int))

    # Ensure base value is included
    if base_n not in param_range:
        param_range = np.sort(np.append(param_range, base_n))

    train_scores = []
    cv_scores = []
    cv_std = []

    for n_est in param_range:
        fold_train = []
        fold_cv = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create model with modified n_estimators
            model_params = base_params.copy()
            model_params[param_name] = n_est

            # Disable early stopping for consistent comparison
            if model_type == 'XGBoost':
                model_params['early_stopping_rounds'] = None
            elif model_type == 'LightGBM':
                model_params.pop('early_stopping_rounds', None)
            elif model_type == 'CatBoost':
                model_params['early_stopping_rounds'] = None

            model = _create_ensemble_model(model_type, model_params, task)

            try:
                if model_type == 'CatBoost':
                    model.fit(X_train, y_train, verbose=False)
                else:
                    model.fit(X_train, y_train)

                # Training predictions
                y_train_pred = model.predict(X_train)
                # Test predictions
                y_test_pred = model.predict(X_test)

                if task == 'regression':
                    fold_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
                    fold_cv.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
                else:
                    fold_train.append(1 - accuracy_score(y_train, y_train_pred))
                    fold_cv.append(1 - accuracy_score(y_test, y_test_pred))
            except Exception as e:
                print(f"Warning: model fit failed for {param_name}={n_est}: {e}")
                fold_train.append(np.nan)
                fold_cv.append(np.nan)

        train_scores.append(np.nanmean(fold_train))
        cv_scores.append(np.nanmean(fold_cv))
        cv_std.append(np.nanstd(fold_cv))

    train_scores = np.array(train_scores)
    cv_scores = np.array(cv_scores)
    cv_std = np.array(cv_std)

    # Find optimal and selected indices
    valid_idx = np.where(~np.isnan(cv_scores))[0]
    if len(valid_idx) > 0:
        optimal_idx = valid_idx[np.argmin(cv_scores[valid_idx])]
    else:
        optimal_idx = 0

    selected_idx = np.argmin(np.abs(param_range - base_n))

    return {
        'param_values': param_range.tolist(),
        'train_scores': train_scores,
        'cv_scores': cv_scores,
        'cv_std': cv_std,
        'param_name': param_name,
        'metric_name': 'RMSE' if task == 'regression' else '1-Accuracy',
        'optimal_idx': optimal_idx,
        'selected_idx': selected_idx
    }


def _create_ensemble_model(model_type, params, task):
    """Create an ensemble model with given parameters."""
    params = params.copy()

    if model_type == 'RandomForest':
        if task == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)

    elif model_type == 'XGBoost':
        try:
            import xgboost as xgb
            if task == 'regression':
                return xgb.XGBRegressor(**params)
            else:
                return xgb.XGBClassifier(**params)
        except ImportError:
            raise ImportError("XGBoost not installed")

    elif model_type == 'LightGBM':
        try:
            import lightgbm as lgb
            params.setdefault('verbose', -1)
            if task == 'regression':
                return lgb.LGBMRegressor(**params)
            else:
                return lgb.LGBMClassifier(**params)
        except ImportError:
            raise ImportError("LightGBM not installed")

    elif model_type == 'CatBoost':
        try:
            import catboost as cb
            params.setdefault('verbose', False)
            if task == 'regression':
                return cb.CatBoostRegressor(**params)
            else:
                return cb.CatBoostClassifier(**params)
        except ImportError:
            raise ImportError("CatBoost not installed")

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_regularization_validation_curve(model_class, X, y, base_alpha, cv,
                                            task='regression', n_points=8, orders=2):
    """
    Compute validation curve for regularization parameter (alpha/C).

    Parameters
    ----------
    model_class : class
        sklearn estimator class (Ridge, Lasso, ElasticNet, SVR, etc.)
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target values
    base_alpha : float
        Selected regularization parameter value
    cv : cross-validation generator
        Cross-validation splitting strategy
    task : str, default='regression'
        'regression' or 'classification'
    n_points : int, default=8
        Number of points in the sweep
    orders : int, default=2
        Number of orders of magnitude to span (±orders around base_alpha)

    Returns
    -------
    dict with validation curve results
    """
    # Determine parameter name based on model class
    model_name = model_class.__name__
    if model_name in ['SVC', 'SVR']:
        param_name = 'C'
    else:
        param_name = 'alpha'

    # Generate log-spaced sweep range
    min_alpha = base_alpha / (10 ** orders)
    max_alpha = base_alpha * (10 ** orders)
    param_range = np.logspace(np.log10(min_alpha), np.log10(max_alpha), n_points)

    # Use sklearn's validation_curve
    if task == 'regression':
        estimator = model_class()
    else:
        estimator = model_class()

    return compute_sklearn_validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        task=task
    )


def compute_learning_curve(estimator, X, y, cv, task='regression', train_sizes=None):
    """
    Compute learning curve (performance vs training set size).

    Parameters
    ----------
    estimator : sklearn estimator or Pipeline
        Model to evaluate (should be unfitted)
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target values
    cv : cross-validation generator
        Cross-validation splitting strategy
    task : str, default='regression'
        'regression' or 'classification'
    train_sizes : array-like, optional
        Fractions or absolute numbers of training samples
        Default: np.linspace(0.2, 1.0, 8)

    Returns
    -------
    dict with keys:
        - train_sizes_abs: absolute training sample counts
        - train_scores: array of mean training scores (RMSE or error rate)
        - train_std: array of training score standard deviations
        - cv_scores: array of mean CV scores
        - cv_std: array of CV score standard deviations
        - metric_name: 'RMSE' or '1-Accuracy'
        - interpretation: text explaining what the curve suggests
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 8)

    if task == 'regression':
        scoring = 'neg_root_mean_squared_error'
    else:
        scoring = 'accuracy'

    try:
        train_sizes_abs, train_scores_raw, cv_scores_raw = learning_curve(
            estimator, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=42
        )

        # Convert scores
        if task == 'regression':
            train_scores = -np.mean(train_scores_raw, axis=1)
            train_std = np.std(-train_scores_raw, axis=1)
            cv_scores = -np.mean(cv_scores_raw, axis=1)
            cv_std = np.std(-cv_scores_raw, axis=1)
            metric_name = 'RMSE'
        else:
            train_scores = 1 - np.mean(train_scores_raw, axis=1)
            train_std = np.std(1 - train_scores_raw, axis=1)
            cv_scores = 1 - np.mean(cv_scores_raw, axis=1)
            cv_std = np.std(1 - cv_scores_raw, axis=1)
            metric_name = '1-Accuracy'

        # Generate interpretation
        interpretation = _interpret_learning_curve(train_scores, cv_scores)

        return {
            'train_sizes_abs': train_sizes_abs.tolist(),
            'train_scores': train_scores,
            'train_std': train_std,
            'cv_scores': cv_scores,
            'cv_std': cv_std,
            'metric_name': metric_name,
            'interpretation': interpretation
        }

    except Exception as e:
        print(f"Warning: learning_curve failed: {e}")
        return {
            'train_sizes_abs': [],
            'train_scores': np.array([]),
            'train_std': np.array([]),
            'cv_scores': np.array([]),
            'cv_std': np.array([]),
            'metric_name': 'RMSE' if task == 'regression' else '1-Accuracy',
            'interpretation': f"Learning curve computation failed: {str(e)}"
        }


def _interpret_learning_curve(train_scores, cv_scores):
    """Generate interpretation text for learning curve."""
    if len(train_scores) < 2 or len(cv_scores) < 2:
        return "Insufficient data points for interpretation."

    # Calculate gap between train and CV at end
    final_gap = cv_scores[-1] - train_scores[-1]

    # Calculate trend (slope) of CV error
    cv_trend = cv_scores[-1] - cv_scores[0]

    # Calculate if gap is closing
    initial_gap = cv_scores[0] - train_scores[0]
    gap_closing = final_gap < initial_gap

    interpretations = []

    if gap_closing and cv_trend < 0:
        interpretations.append(
            "Gap is closing as training size increases - more data would likely improve model performance."
        )
    elif abs(cv_trend) < 0.01 * np.mean(cv_scores):
        interpretations.append(
            "Both curves have flattened - model has likely saturated and more data may not help significantly."
        )
    if final_gap > 0.1 * np.mean(train_scores):
        interpretations.append(
            "Large gap between training and CV error suggests overfitting - "
            "consider more regularization or simpler model."
        )
    elif final_gap < 0.02 * np.mean(train_scores):
        interpretations.append(
            "Small gap between training and CV error indicates good generalization."
        )

    if train_scores[-1] > train_scores[0]:
        interpretations.append(
            "Training error increasing with data - possible underfitting or data quality issues."
        )

    if not interpretations:
        interpretations.append("Learning curve appears normal.")

    return " ".join(interpretations)
