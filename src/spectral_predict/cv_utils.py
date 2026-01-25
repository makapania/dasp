"""Cross-validation utilities with early stopping support for boosting models.

This module provides helper functions for cross-validation that support early stopping
for gradient boosting models (XGBoost, CatBoost, LightGBM). These models benefit from
early stopping which typically saves 30-50% training time and often improves model
quality by preventing overfitting.

The main function `cross_validate_with_early_stopping` automatically detects boosting
models and applies early stopping, while falling back to standard sklearn cross_validate
for other models.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.model_selection import cross_validate, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score, mean_absolute_error,
    log_loss
)
from typing import Optional, Dict, Any, Union, List
import warnings

# Import boosting model types for detection
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb

# CatBoost may not be available in all environments
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None
    CatBoostClassifier = None

# Tuple of all boosting model types for isinstance checks
XGBOOST_MODELS = (XGBRegressor, XGBClassifier)
LIGHTGBM_MODELS = (LGBMRegressor, LGBMClassifier)
if CATBOOST_AVAILABLE:
    CATBOOST_MODELS = (CatBoostRegressor, CatBoostClassifier)
    BOOSTING_MODELS = XGBOOST_MODELS + LIGHTGBM_MODELS + CATBOOST_MODELS
else:
    CATBOOST_MODELS = ()
    BOOSTING_MODELS = XGBOOST_MODELS + LIGHTGBM_MODELS


def is_boosting_model(model) -> bool:
    """Check if a model is a boosting model that supports early stopping.

    Parameters
    ----------
    model : estimator
        Model to check

    Returns
    -------
    bool
        True if model is XGBoost, LightGBM, or CatBoost
    """
    return isinstance(model, BOOSTING_MODELS)


def _get_model_from_pipeline(pipeline_or_model):
    """Extract the final model from a pipeline, or return the model if not a pipeline.

    Parameters
    ----------
    pipeline_or_model : estimator or Pipeline
        Either a sklearn estimator or a Pipeline

    Returns
    -------
    estimator
        The final estimator (model)
    """
    if hasattr(pipeline_or_model, 'steps'):
        # It's a pipeline - get the final step
        return pipeline_or_model.steps[-1][1]
    return pipeline_or_model


def _fit_with_early_stopping(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = 15
) -> None:
    """Fit a boosting model with early stopping.

    Parameters
    ----------
    model : estimator
        Boosting model (XGBoost, LightGBM, or CatBoost)
    X_train : ndarray
        Training features
    y_train : ndarray
        Training targets
    X_val : ndarray
        Validation features for early stopping
    y_val : ndarray
        Validation targets for early stopping
    early_stopping_rounds : int, default=15
        Number of rounds without improvement before stopping
    """
    if isinstance(model, XGBOOST_MODELS):
        # Set early_stopping_rounds before fitting
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    elif isinstance(model, LIGHTGBM_MODELS):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)  # Suppress output
            ]
        )

    elif CATBOOST_AVAILABLE and isinstance(model, CATBOOST_MODELS):
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),  # CatBoost uses tuple, not list
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
    else:
        # Not a boosting model - standard fit
        model.fit(X_train, y_train)


def _compute_score(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    """Compute a score given true and predicted values.

    Parameters
    ----------
    y_true : ndarray
        True target values
    y_pred : ndarray
        Predicted values
    scoring : str
        Scoring method name (sklearn style, e.g., 'neg_root_mean_squared_error')

    Returns
    -------
    float
        Computed score (following sklearn convention where higher is better)
    """
    if scoring == 'neg_root_mean_squared_error':
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    elif scoring == 'neg_mean_squared_error':
        return -mean_squared_error(y_true, y_pred)
    elif scoring == 'r2':
        return r2_score(y_true, y_pred)
    elif scoring == 'neg_mean_absolute_error':
        return -mean_absolute_error(y_true, y_pred)
    elif scoring == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif scoring == 'f1':
        return f1_score(y_true, y_pred, average='binary', zero_division=0)
    elif scoring == 'f1_weighted':
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)
    elif scoring == 'f1_macro':
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    elif scoring == 'precision':
        return precision_score(y_true, y_pred, average='binary', zero_division=0)
    elif scoring == 'recall':
        return recall_score(y_true, y_pred, average='binary', zero_division=0)
    else:
        raise ValueError(f"Unsupported scoring method: {scoring}")


def cross_validate_with_early_stopping(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: Union[str, Dict[str, str]] = 'neg_root_mean_squared_error',
    early_stopping_rounds: int = 15,
    n_jobs: int = 1,
    return_train_score: bool = False,
    return_estimator: bool = False
) -> Dict[str, np.ndarray]:
    """Cross-validate with early stopping support for boosting models.

    For boosting models (XGBoost, CatBoost, LightGBM), this function uses a manual
    CV loop that passes eval_set to enable early stopping. For other models, it
    falls back to sklearn's standard cross_validate.

    Parameters
    ----------
    model : estimator
        Model to cross-validate. Can be a boosting model or any sklearn estimator.
    X : ndarray
        Feature matrix (n_samples, n_features)
    y : ndarray
        Target vector (n_samples,)
    cv : int or cross-validator
        Number of folds or sklearn CV splitter
    scoring : str or dict, default='neg_root_mean_squared_error'
        Scoring method. Can be a string (e.g., 'accuracy', 'neg_root_mean_squared_error')
        or a dict mapping names to scorers (e.g., {'rmse': 'neg_root_mean_squared_error'}).
    early_stopping_rounds : int, default=15
        Number of rounds without improvement before stopping. Only used for
        boosting models. Set to 0 or None to disable early stopping.
    n_jobs : int, default=1
        Number of parallel jobs. Note: early stopping CV is always serial for
        boosting models to ensure reproducibility.
    return_train_score : bool, default=False
        Whether to return training scores
    return_estimator : bool, default=False
        Whether to return fitted estimators

    Returns
    -------
    dict
        Dictionary with keys:
        - 'test_score' or 'test_{scorer_name}': Array of test scores per fold
        - 'train_score' or 'train_{scorer_name}': Array of train scores (if return_train_score)
        - 'fit_time': Array of fit times per fold
        - 'score_time': Array of score times per fold
        - 'estimator': List of fitted estimators (if return_estimator)

    Examples
    --------
    >>> from xgboost import XGBRegressor
    >>> from sklearn.model_selection import KFold
    >>> model = XGBRegressor(n_estimators=100)
    >>> cv = KFold(n_splits=5, shuffle=True, random_state=42)
    >>> results = cross_validate_with_early_stopping(
    ...     model, X, y, cv=cv, scoring='neg_root_mean_squared_error',
    ...     early_stopping_rounds=15
    ... )
    >>> print(f"Mean RMSE: {-results['test_score'].mean():.4f}")
    """
    import time

    # Check if model is a boosting model that benefits from early stopping
    final_model = _get_model_from_pipeline(model)
    use_early_stopping = (
        is_boosting_model(final_model) and
        early_stopping_rounds is not None and
        early_stopping_rounds > 0
    )

    # If not a boosting model or early stopping disabled, use standard cross_validate
    if not use_early_stopping:
        return cross_validate(
            model, X, y, cv=cv, scoring=scoring,
            n_jobs=n_jobs, return_train_score=return_train_score,
            return_estimator=return_estimator, error_score='raise'
        )

    # Handle scoring - convert string to dict for uniform handling
    if isinstance(scoring, str):
        scoring_dict = {'score': scoring}
        single_scorer = True
    else:
        scoring_dict = scoring
        single_scorer = False

    # Initialize result containers
    results = {
        'fit_time': [],
        'score_time': []
    }

    for scorer_name in scoring_dict.keys():
        results[f'test_{scorer_name}'] = []
        if return_train_score:
            results[f'train_{scorer_name}'] = []

    if return_estimator:
        results['estimator'] = []

    # Manual CV loop with early stopping
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone model for this fold
        model_clone = clone(model)
        final_model_clone = _get_model_from_pipeline(model_clone)

        # Fit with early stopping
        fit_start = time.time()

        if hasattr(model_clone, 'steps'):
            # It's a pipeline - fit preprocessing steps first
            X_train_transformed = X_train.copy()
            X_val_transformed = X_val.copy()

            for step_name, step in model_clone.steps[:-1]:
                step.fit(X_train_transformed, y_train)
                X_train_transformed = step.transform(X_train_transformed)
                X_val_transformed = step.transform(X_val_transformed)

            # Fit final model with early stopping
            _fit_with_early_stopping(
                final_model_clone,
                X_train_transformed, y_train,
                X_val_transformed, y_val,
                early_stopping_rounds
            )
        else:
            # Direct model - fit with early stopping
            _fit_with_early_stopping(
                model_clone, X_train, y_train, X_val, y_val, early_stopping_rounds
            )

        fit_time = time.time() - fit_start
        results['fit_time'].append(fit_time)

        # Score
        score_start = time.time()

        if hasattr(model_clone, 'steps'):
            # Transform test data through pipeline
            X_val_transformed = X_val.copy()
            for step_name, step in model_clone.steps[:-1]:
                X_val_transformed = step.transform(X_val_transformed)
            y_pred = final_model_clone.predict(X_val_transformed)
        else:
            y_pred = model_clone.predict(X_val)

        y_pred = np.ravel(y_pred)

        for scorer_name, scorer in scoring_dict.items():
            score = _compute_score(y_val, y_pred, scorer)
            results[f'test_{scorer_name}'].append(score)

        if return_train_score:
            if hasattr(model_clone, 'steps'):
                X_train_transformed = X_train.copy()
                for step_name, step in model_clone.steps[:-1]:
                    X_train_transformed = step.transform(X_train_transformed)
                y_train_pred = final_model_clone.predict(X_train_transformed)
            else:
                y_train_pred = model_clone.predict(X_train)
            y_train_pred = np.ravel(y_train_pred)

            for scorer_name, scorer in scoring_dict.items():
                train_score = _compute_score(y_train, y_train_pred, scorer)
                results[f'train_{scorer_name}'].append(train_score)

        score_time = time.time() - score_start
        results['score_time'].append(score_time)

        if return_estimator:
            results['estimator'].append(model_clone)

    # Convert lists to numpy arrays
    for key in results:
        if key != 'estimator':
            results[key] = np.array(results[key])

    # If single scorer, also provide 'test_score' key for compatibility
    if single_scorer:
        results['test_score'] = results['test_score']

    return results


def cross_val_predict_with_early_stopping(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    early_stopping_rounds: int = 15,
    method: str = 'predict'
) -> np.ndarray:
    """Get cross-validated predictions with early stopping support.

    Similar to sklearn's cross_val_predict, but supports early stopping for
    boosting models.

    Parameters
    ----------
    model : estimator
        Model to cross-validate
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    cv : int or cross-validator
        Number of folds or CV splitter
    early_stopping_rounds : int, default=15
        Early stopping rounds for boosting models
    method : str, default='predict'
        Method to call ('predict' or 'predict_proba')

    Returns
    -------
    ndarray
        Cross-validated predictions
    """
    # Check if model is a boosting model
    final_model = _get_model_from_pipeline(model)
    use_early_stopping = (
        is_boosting_model(final_model) and
        early_stopping_rounds is not None and
        early_stopping_rounds > 0
    )

    # If not a boosting model, use standard cross_val_predict
    if not use_early_stopping:
        return cross_val_predict(model, X, y, cv=cv, method=method)

    # Manual CV loop
    n_samples = X.shape[0]

    # Determine output shape
    if method == 'predict_proba':
        # Get number of classes
        n_classes = len(np.unique(y))
        predictions = np.zeros((n_samples, n_classes))
    else:
        predictions = np.zeros(n_samples)

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_clone = clone(model)
        final_model_clone = _get_model_from_pipeline(model_clone)

        if hasattr(model_clone, 'steps'):
            # Pipeline
            X_train_transformed = X_train.copy()
            X_val_transformed = X_val.copy()

            for step_name, step in model_clone.steps[:-1]:
                step.fit(X_train_transformed, y_train)
                X_train_transformed = step.transform(X_train_transformed)
                X_val_transformed = step.transform(X_val_transformed)

            _fit_with_early_stopping(
                final_model_clone,
                X_train_transformed, y_train,
                X_val_transformed, y_val,
                early_stopping_rounds
            )

            if method == 'predict_proba':
                preds = final_model_clone.predict_proba(X_val_transformed)
            else:
                preds = final_model_clone.predict(X_val_transformed)
        else:
            _fit_with_early_stopping(
                model_clone, X_train, y_train, X_val, y_val, early_stopping_rounds
            )

            if method == 'predict_proba':
                preds = model_clone.predict_proba(X_val)
            else:
                preds = model_clone.predict(X_val)

        predictions[val_idx] = preds

    return predictions


def cross_val_score_with_early_stopping(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: str = 'neg_root_mean_squared_error',
    early_stopping_rounds: int = 15,
    n_jobs: int = 1
) -> np.ndarray:
    """Cross-validation scores with early stopping support.

    Simplified interface returning only test scores. For more options,
    use cross_validate_with_early_stopping.

    Parameters
    ----------
    model : estimator
        Model to cross-validate
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    cv : int or cross-validator
        CV splitter
    scoring : str, default='neg_root_mean_squared_error'
        Scoring method
    early_stopping_rounds : int, default=15
        Early stopping rounds
    n_jobs : int, default=1
        Number of parallel jobs (only used for non-boosting models)

    Returns
    -------
    ndarray
        Array of scores per fold
    """
    results = cross_validate_with_early_stopping(
        model, X, y, cv=cv, scoring=scoring,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs
    )
    return results['test_score']
