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
from collections import Counter
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

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


def validate_cv_strategy_for_task(
    strategy: str,
    task_type: str,
    y: np.ndarray,
    n_folds: int,
    n_repeats: int | None = None,
    inlier_label=None,
) -> None:
    """Upfront guard for CV strategies that can fail inside fold loops.

    Catches cases where LOO / K-fold will hit a single-class training fold —
    the sklearn error comes out as an opaque ValueError deep in the fit loop,
    which gets swallowed by pooled helpers or degrades into silent NaN metrics
    in the Bayesian objective. Validation must run BEFORE training starts.

    Parameters
    ----------
    strategy : str
        'kfold', 'repeated_kfold', or 'loo'.
    task_type : str
        'regression', 'classification', or 'one_class'.
    y : ndarray
        Target vector. For one-class, the original labels (used with `inlier_label`).
    n_folds : int
        Number of folds.
    n_repeats : int, optional
        Number of repeats. Required (and must be >= 1) when strategy=='repeated_kfold'.
    inlier_label : optional
        Inlier class label for one-class tasks. If provided, validates inlier count
        is sufficient for the strategy.

    Raises
    ------
    ValueError
        If the dataset is too small or imbalanced for the requested strategy.
    """
    if strategy == 'repeated_kfold':
        if n_repeats is None or int(n_repeats) < 1:
            raise ValueError(
                f"Repeated K-Fold requires n_repeats >= 1 (got {n_repeats!r})."
            )

    if task_type not in ('classification', 'one_class'):
        return

    y_arr = np.asarray(y)
    n = len(y_arr)
    if n < 2:
        raise ValueError(f"Need at least 2 samples for {task_type} CV (got {n}).")

    if task_type == 'classification':
        # Enumerate class counts — sklearn needs ≥2 classes in every train fold
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2:
            raise ValueError(
                f"Classification requires at least 2 classes, got only {len(classes)} "
                f"({classes.tolist()})."
            )
        min_class = int(counts.min())
        if strategy == 'loo' and min_class < 2:
            rarest = classes[int(counts.argmin())]
            raise ValueError(
                f"LOO CV requires at least 2 samples per class (class {rarest!r} has "
                f"{min_class}). Leaving out its only sample yields a single-class "
                f"training fold. Use K-fold or add more samples for that class."
            )
        if strategy in ('kfold', 'repeated_kfold') and min_class < n_folds:
            rarest = classes[int(counts.argmin())]
            raise ValueError(
                f"{n_folds}-fold CV requires at least {n_folds} samples per class "
                f"(class {rarest!r} has {min_class}). Reduce folds or add samples."
            )
        return

    # One-class: validate inlier count against strategy.
    # If inlier_label is provided, caller is passing raw labels — coerce both
    # sides to str for comparison (matches search.py:~4878's convention for
    # one-class label encoding; prevents "too few inliers" errors when
    # inlier_label dtype differs from y_arr dtype, e.g. int vs numpy string).
    # If inlier_label is None, labels are assumed to be +1/-1 encoded (matches
    # contamination.run_one_class_cv after conversion).
    if inlier_label is not None:
        y_str = np.asarray(y_arr, dtype=str)
        n_inliers = int(np.sum(y_str == str(inlier_label)))
    else:
        n_inliers = int(np.sum(y_arr == 1))
    if strategy == 'loo':
        # 2 inliers minimum; PCA-SIMCA needs more (enforced model-side in contamination.py)
        if n_inliers < 2:
            raise ValueError(
                f"LOO one-class CV requires at least 2 inliers (got {n_inliers})."
            )
    elif strategy in ('kfold', 'repeated_kfold'):
        if n_inliers < n_folds:
            raise ValueError(
                f"{n_folds}-fold one-class CV requires at least {n_folds} inliers "
                f"(got {n_inliers}). Reduce folds or use LOO."
            )


def compute_min_train_fold_size(
    cv_strategy: str,
    n_samples: int,
    n_folds: int,
) -> int:
    """Exact lower bound on the smallest training-fold size.

    PLS regression requires n_components <= min(n_features, n_samples_train_fold).
    The grid for ``n_components`` must be clamped using a value that is no greater
    than the smallest training-fold size any CV split will produce, otherwise
    sklearn raises silently inside the fold or returns NaN metrics that are
    swallowed by the search aggregator.

    Strategy semantics
    ------------------
    - ``'kfold'``:  train fold size = ``n_samples - ceil(n_samples / n_folds)``.
      The formula ``n_samples * (n_folds - 1) // n_folds`` used here equals
      ``n_samples - ceil(n_samples / n_folds)`` for all positive integers and
      is therefore exact (not merely conservative) for sklearn ``KFold``.
    - ``'repeated_kfold'``: identical to kfold.  ``RepeatedKFold`` reuses
      ``KFold`` partitions across repeats; per-fold geometry is the same.
      ``n_repeats`` does NOT affect train-fold size.
    - ``'loo'``: train fold size = ``n_samples - 1``.

    Group splitters (``GroupKFold``, ``LeaveOneGroupOut``) are NOT covered
    here and will raise ``NotImplementedError``; T-15 will plumb group-aware
    sizing through a separate path.

    Parameters
    ----------
    cv_strategy : str
        One of ``'kfold'``, ``'repeated_kfold'``, ``'loo'``.
    n_samples : int
        Total samples in the calibration set.
    n_folds : int
        Number of folds.  Ignored when ``cv_strategy == 'loo'``.

    Returns
    -------
    int
        Exact lower bound on the smallest training-fold size, >= 1.

    Raises
    ------
    ValueError
        If ``n_samples < 2`` or ``n_folds < 2`` (kfold/repeated_kfold)
        or ``n_folds > n_samples`` (kfold/repeated_kfold — invalid geometry
        that would produce empty training folds).
    NotImplementedError
        If ``cv_strategy`` is ``'group_kfold'`` or ``'leave_one_group_out'``
        (T-15 scope).
    """
    if n_samples < 2:
        raise ValueError(
            f"PLS clamp requires n_samples >= 2 (got {n_samples})."
        )
    if cv_strategy == 'loo':
        return n_samples - 1
    if cv_strategy in ('kfold', 'repeated_kfold'):
        if n_folds < 2:
            raise ValueError(
                f"K-fold CV requires n_folds >= 2 (got {n_folds})."
            )
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have more folds ({n_folds}) than samples "
                f"({n_samples}); reduce folds or use LOO."
            )
        return max(1, n_samples * (n_folds - 1) // n_folds)
    if cv_strategy in ('group_kfold', 'leave_one_group_out'):
        raise NotImplementedError(
            f"compute_min_train_fold_size: {cv_strategy!r} not supported yet "
            "(T-15 will add group-aware sizing)."
        )
    raise ValueError(
        f"Unknown cv_strategy: {cv_strategy!r}. "
        "Expected 'kfold', 'repeated_kfold', or 'loo'."
    )


def estimate_total_cv_fits(
    strategy: str,
    n_folds: int,
    n_repeats: int,
    n_samples: int,
    n_trials: int = 1,
    n_models: int = 1,
    n_preprocessing: int = 1,
) -> int:
    """Estimate total model fits for a CV-based search.

    Parameters
    ----------
    strategy : str
        CV strategy: 'kfold', 'repeated_kfold', or 'loo'.
    n_folds : int
        Number of folds (ignored for 'loo').
    n_repeats : int
        Number of repeats (used only for 'repeated_kfold').
    n_samples : int
        Number of training samples.
    n_trials : int
        Number of Bayesian trials or grid configurations.
    n_models : int
        Number of model types being tested.
    n_preprocessing : int
        Number of preprocessing configurations.

    Returns
    -------
    int
        Estimated total number of individual model fits.
    """
    if strategy == 'loo':
        cv_fits = n_samples
    elif strategy == 'repeated_kfold':
        cv_fits = n_folds * n_repeats
    else:
        cv_fits = n_folds
    return cv_fits * max(1, n_trials) * max(1, n_models) * max(1, n_preprocessing)


def build_cv_splitter(
    strategy: str,
    n_folds: int,
    task_type: str,
    n_repeats: int = 5,
    random_state: int = 42,
    y=None,
):
    """Build a sklearn CV splitter for the requested strategy.

    Parameters
    ----------
    strategy : str
        One of 'kfold', 'repeated_kfold', 'loo'.
    n_folds : int
        Number of folds. Ignored when strategy == 'loo'.
    task_type : str
        One of 'regression', 'classification', 'one_class'. Controls stratification.
    n_repeats : int, default=5
        Number of repeats. Used only when strategy == 'repeated_kfold'.
    random_state : int, default=42
        Random state for reproducibility.
    y : array-like, optional
        Target array. When provided and task_type is 'classification', validates
        that y contains discrete labels (not continuous). Raises ValueError if
        StratifiedKFold would fail on the given y.

    Returns
    -------
    sklearn.model_selection.BaseCrossValidator
        A splitter object usable with cross_validate, cross_val_predict, etc.
    """
    use_stratified = task_type == 'classification' and strategy in ('kfold', 'repeated_kfold')
    if use_stratified and y is not None:
        from sklearn.utils.multiclass import type_of_target
        try:
            y_kind = type_of_target(y)
        except (TypeError, ValueError):
            y_kind = None
        if y_kind == 'continuous':
            raise ValueError(
                "task_type='classification' but y is continuous "
                "(sklearn type_of_target='continuous'). StratifiedKFold requires "
                "binary or multiclass labels. Either change task_type to "
                "'regression' or use a categorical target column."
            )
    if strategy == 'loo':
        from sklearn.model_selection import LeaveOneOut
        return LeaveOneOut()
    if strategy == 'repeated_kfold':
        from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
        if use_stratified:
            return RepeatedStratifiedKFold(
                n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
            )
        return RepeatedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
        )
    if strategy == 'kfold':
        if use_stratified:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    raise ValueError(
        f"Unknown CV strategy: {strategy!r}. Expected 'kfold', 'repeated_kfold', or 'loo'."
    )


def _is_repeated_cv(cv) -> bool:
    """Check if a CV splitter produces overlapping test sets (repeated splits)."""
    return isinstance(cv, (RepeatedKFold, RepeatedStratifiedKFold))


def _model_is_classifier(model) -> bool:
    """Detect whether an estimator (possibly wrapped in a Pipeline) is a classifier."""
    inner = _get_model_from_pipeline(model) if hasattr(model, 'steps') else model
    try:
        return is_classifier(inner)
    except (AttributeError, TypeError) as e:
        # Custom estimator with broken tags — surface so we don't silently fall back
        # to numeric averaging of integer labels under repeated CV.
        warnings.warn(
            f"Could not determine classifier status for {type(inner).__name__}: {e}. "
            "Treating as non-classifier; check if repeated-CV predict averaging is safe.",
            stacklevel=2,
        )
        return False


def reduce_repeated_cv_predictions(
    cv_metrics: list,
    splits: list,
    n_samples: int,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-fold (y_test, y_pred) outputs to one prediction per sample.

    Used by the grid-search aggregation path in search.py. Under repeated CV
    each sample appears in multiple test folds; flat concatenation duplicates
    rows and biases pooled metrics. For regression we average repeated
    predictions; for classification we take the majority vote (averaging
    integer labels would yield fractional pseudo-labels).

    ORDER MUST MATCH: cv_metrics[i] must correspond to splits[i]. Silent
    miscorrespondence corrupts per-sample attribution without raising.

    Parameters
    ----------
    cv_metrics : list of dict
        Per-fold output from `_run_single_fold` — each must have 'y_test', 'y_pred'.
    splits : list of (train_idx, test_idx)
        Realized fold indices (same order as cv_metrics).
    n_samples : int
        Total samples in the original X (not the pooled count).
    task_type : str
        'regression' or 'classification'. Drives reduction strategy.

    Returns
    -------
    all_y_test, all_y_pred : ndarray
        One row per sample that received at least one prediction, in sample-index order.
    """
    if len(cv_metrics) != len(splits):
        raise ValueError(
            f"cv_metrics ({len(cv_metrics)}) and splits ({len(splits)}) length mismatch"
        )

    if task_type == 'regression':
        pred_sum = np.zeros(n_samples, dtype=float)
        truth = np.full(n_samples, np.nan, dtype=float)
        pred_count = np.zeros(n_samples, dtype=int)
        for m, (_train_idx, test_idx) in zip(cv_metrics, splits):
            preds = np.asarray(m['y_pred']).ravel()
            tests = np.asarray(m['y_test']).ravel()
            pred_sum[test_idx] += preds
            pred_count[test_idx] += 1
            truth[test_idx] = tests
        mask = pred_count > 0
        return truth[mask], pred_sum[mask] / pred_count[mask]

    # Classification: majority vote per sample
    votes_per_sample: List[list] = [[] for _ in range(n_samples)]
    truth_label = [None] * n_samples
    for m, (_train_idx, test_idx) in zip(cv_metrics, splits):
        preds = np.asarray(m['y_pred']).ravel()
        tests = np.asarray(m['y_test']).ravel()
        for i, sample_idx in enumerate(test_idx):
            votes_per_sample[sample_idx].append(preds[i])
            truth_label[sample_idx] = tests[i]
    mask = [len(v) > 0 for v in votes_per_sample]
    truth_arr = np.array([t for t, k in zip(truth_label, mask) if k])
    pred_arr = np.array([
        Counter(v).most_common(1)[0][0]
        for v, k in zip(votes_per_sample, mask) if k
    ])
    return truth_arr, pred_arr


def _majority_vote(votes_per_sample: list, dtype) -> np.ndarray:
    """Reduce a list of vote-lists to a single prediction per sample via mode.

    Used for repeated-CV classifier predictions where numeric averaging would
    produce nonsensical fractional labels (e.g. averaging [0, 1] → 0.5).
    """
    n_samples = len(votes_per_sample)
    out = np.empty(n_samples, dtype=dtype)
    for i, votes in enumerate(votes_per_sample):
        if votes:
            out[i] = Counter(votes).most_common(1)[0][0]
    return out


def _slice_fit_params(fit_params: Optional[Dict[str, Any]], train_idx, n_samples: int) -> Dict[str, Any]:
    """Slice array-valued fit_params per train_idx, leaving scalars untouched.

    Used by the manual repeated-CV loop in cross_val_predict_pooled (and friends)
    to mirror sklearn's auto-slicing of `params` in cross_val_predict for arrays
    whose length matches the sample count (sample_weight, sample-indexed metadata).
    Non-array values pass through verbatim.
    """
    if not fit_params:
        return {}
    sliced: Dict[str, Any] = {}
    for key, value in fit_params.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == n_samples:
            sliced[key] = value[train_idx]
        else:
            sliced[key] = value
    return sliced


def cross_val_predict_pooled(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    n_jobs: int = 1,
    method: str = 'predict',
    fit_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Cross-validated predictions that work for all CV strategies including repeated CV.

    For standard CV (KFold, StratifiedKFold, LOO), this delegates to sklearn's
    cross_val_predict. For repeated CV (RepeatedKFold, RepeatedStratifiedKFold),
    it runs a manual loop and averages predictions per sample across repeats.

    Parameters
    ----------
    model : estimator
        Model to cross-validate.
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    cv : cross-validator
        CV splitter (any sklearn splitter).
    n_jobs : int, default=1
        Number of parallel jobs (only for non-repeated CV).
    method : str, default='predict'
        Prediction method ('predict' or 'predict_proba').
    fit_params : dict or None, default=None
        Per-fold fit kwargs forwarded to ``model.fit`` (e.g.
        ``{'model__sample_weight': sw}`` for a Pipeline whose terminal step is
        named ``model``). Array values whose length matches X are sliced per
        train_idx; scalars pass through unchanged. Required for sample_weight-
        only classifiers (XGBoost, RidgeClassifier) under
        imbalance_method='class_weight'.

    Returns
    -------
    ndarray
        Per-sample predictions, averaged across repeats for repeated CV.
    """
    # When no fit_params, delegate to sklearn's optimized cross_val_predict
    # for non-repeated CV. When fit_params IS supplied, force the manual loop
    # so per-fold slicing goes through _slice_fit_params — sklearn 1.8 removed
    # the legacy `fit_params=` kwarg in favour of metadata routing
    # (`set_config(enable_metadata_routing=True)` + `set_fit_request(...)`),
    # which would require global state mutation we'd rather avoid here.
    if not _is_repeated_cv(cv) and not fit_params:
        return cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs, method=method)

    # Manual loop: handles repeated CV AND any cv with fit_params.
    # For classifier predict, reduce by majority vote (averaging integer class
    # labels produces nonsensical fractional "predictions"). For regression or
    # predict_proba, average across repeats (or just place if non-repeated).
    n_samples = X.shape[0]
    use_majority_vote = method == 'predict' and _model_is_classifier(model)

    if use_majority_vote:
        votes_per_sample: List[list] = [[] for _ in range(n_samples)]
        for train_idx, test_idx in cv.split(X, y):
            model_clone = clone(model)
            model_clone.fit(
                X[train_idx], y[train_idx],
                **_slice_fit_params(fit_params, train_idx, n_samples),
            )
            preds = np.ravel(model_clone.predict(X[test_idx]))
            for i, sample_idx in enumerate(test_idx):
                votes_per_sample[sample_idx].append(preds[i])
        return _majority_vote(votes_per_sample, dtype=np.asarray(y).dtype)

    if method == 'predict_proba':
        n_classes = len(np.unique(y))
        pred_sum = np.zeros((n_samples, n_classes))
    else:
        pred_sum = np.zeros(n_samples)
    pred_count = np.zeros(n_samples)

    for train_idx, test_idx in cv.split(X, y):
        model_clone = clone(model)
        model_clone.fit(
            X[train_idx], y[train_idx],
            **_slice_fit_params(fit_params, train_idx, n_samples),
        )
        if method == 'predict_proba':
            preds = model_clone.predict_proba(X[test_idx])
        else:
            preds = np.ravel(model_clone.predict(X[test_idx]))
        pred_sum[test_idx] += preds
        pred_count[test_idx] += 1

    mask = pred_count > 0
    if method == 'predict_proba':
        pred_sum[mask] /= pred_count[mask, np.newaxis]
    else:
        pred_sum[mask] /= pred_count[mask]
    return pred_sum


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
    early_stopping_rounds: int = 40,
    sample_weight: np.ndarray | None = None,
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
    early_stopping_rounds : int, default=50
        Number of rounds without improvement before stopping
    sample_weight : ndarray or None, default=None
        Optional per-sample weights threaded into the booster's ``fit``.
        XGBoost has no native ``class_weight`` kwarg; per-sample balanced
        weights via ``compute_sample_weight('balanced', y)`` is the canonical
        path for imbalanced XGBoost classification (mirrors search.py:4427).
    """
    fit_kwargs: dict = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight

    if isinstance(model, XGBOOST_MODELS):
        # Set early_stopping_rounds before fitting
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            **fit_kwargs,
        )

    elif isinstance(model, LIGHTGBM_MODELS):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)  # Suppress output
            ],
            **fit_kwargs,
        )

    elif CATBOOST_AVAILABLE and isinstance(model, CATBOOST_MODELS):
        # Classification requires explicit eval_metric for early stopping
        # Use Accuracy (not Logloss) so early stopping monitors actual classification performance
        # Set via set_params() since it's a constructor parameter, not fit parameter
        if isinstance(model, CatBoostClassifier):
            model.set_params(eval_metric='Accuracy')
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),  # CatBoost uses tuple, not list
            early_stopping_rounds=early_stopping_rounds,
            verbose=0,  # CatBoost prefers int over bool
            **fit_kwargs,
        )
    else:
        # Not a boosting model - standard fit
        model.fit(X_train, y_train, **fit_kwargs)


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
    early_stopping_rounds: int = 40,
    n_jobs: int = 1,
    return_train_score: bool = False,
    return_estimator: bool = False,
    sample_weight: Optional[np.ndarray] = None,
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
    early_stopping_rounds : int, default=50
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
    ...     early_stopping_rounds=50
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

    # LeaveOneOut has 1-sample test folds that cannot serve as an eval_set for
    # early stopping. Disable and warn so the boosting model trains normally.
    from sklearn.model_selection import LeaveOneOut
    if isinstance(cv, LeaveOneOut) and use_early_stopping:
        warnings.warn(
            "Early stopping disabled under LeaveOneOut CV: "
            "single-sample test folds cannot serve as an eval_set. "
            "Boosting model will train without early stopping.",
            stacklevel=2,
        )
        use_early_stopping = False

    # If not a boosting model or early stopping disabled, use standard cross_validate.
    # When sample_weight is provided, route via sklearn's metadata routing API
    # (sklearn >= 1.4; in 1.8+ the legacy `fit_params=` kwarg was removed). We
    # scope `set_config(enable_metadata_routing=True)` to a context manager so
    # global config isn't mutated for callers who are not opted in. The terminal
    # estimator opts in to receiving sample_weight via `set_fit_request`.
    if not use_early_stopping:
        cv_kwargs: Dict[str, Any] = dict(
            cv=cv, scoring=scoring,
            n_jobs=n_jobs, return_train_score=return_train_score,
            return_estimator=return_estimator, error_score='raise',
        )
        if sample_weight is not None:
            import sklearn
            # Clone first so set_fit_request doesn't mutate the caller's
            # estimator instance (the routing-state setter is sticky and
            # would persist past the config_context exit, polluting the
            # original model for any downstream caller).
            model_for_routing = clone(model)
            _inner = _get_model_from_pipeline(model_for_routing)
            if hasattr(_inner, 'set_fit_request'):
                _inner.set_fit_request(sample_weight=True)
            with sklearn.config_context(enable_metadata_routing=True):
                return cross_validate(
                    model_for_routing, X, y,
                    params={'sample_weight': sample_weight},
                    **cv_kwargs,
                )
        return cross_validate(model, X, y, **cv_kwargs)

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
        # Per-fold slice of sample_weight when provided
        sw_train = sample_weight[train_idx] if sample_weight is not None else None

        # Clone model for this fold
        model_clone = clone(model)
        final_model_clone = _get_model_from_pipeline(model_clone)

        # Fit with early stopping
        fit_start = time.time()

        if hasattr(model_clone, 'steps'):
            # It's a pipeline - fit preprocessing steps first
            X_train_transformed = X_train
            X_val_transformed = X_val

            for step_name, step in model_clone.steps[:-1]:
                if hasattr(step, 'fit_resample'):
                    # Resampler changes y AND sample count; recompute sw_train
                    # from post-resample y so balanced weighting matches the
                    # reduced/augmented training set.
                    X_train_transformed, y_train = step.fit_resample(X_train_transformed, y_train)
                    if sw_train is not None:
                        from sklearn.utils.class_weight import compute_sample_weight
                        sw_train = compute_sample_weight('balanced', y_train)
                elif hasattr(step, 'transform'):
                    step.fit(X_train_transformed, y_train)
                    X_train_transformed = step.transform(X_train_transformed)
                    X_val_transformed = step.transform(X_val_transformed)

            _fit_with_early_stopping(
                final_model_clone,
                X_train_transformed, y_train,
                X_val_transformed, y_val,
                early_stopping_rounds,
                sample_weight=sw_train,
            )
        else:
            _fit_with_early_stopping(
                model_clone, X_train, y_train, X_val, y_val, early_stopping_rounds,
                sample_weight=sw_train,
            )

        fit_time = time.time() - fit_start
        results['fit_time'].append(fit_time)

        # Score
        score_start = time.time()

        if hasattr(model_clone, 'steps'):
            X_val_transformed = X_val
            for step_name, step in model_clone.steps[:-1]:
                if hasattr(step, 'fit_resample'):
                    continue
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
                X_train_transformed = X_train
                for step_name, step in model_clone.steps[:-1]:
                    if hasattr(step, 'fit_resample'):
                        continue
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
    early_stopping_rounds: int = 40,
    method: str = 'predict',
    sample_weight: Optional[np.ndarray] = None,
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
    early_stopping_rounds : int, default=50
        Early stopping rounds for boosting models
    method : str, default='predict'
        Method to call ('predict' or 'predict_proba')
    sample_weight : ndarray or None, default=None
        Per-sample weights of length n_samples. Sliced per train_idx and
        threaded into ``_fit_with_early_stopping`` for the boosting fold-fit
        (where it routes to the booster's native ``sample_weight`` fit kwarg).
        For the non-boosting fallback path, the caller-provided ``sample_weight``
        is wrapped into ``fit_params={'model__sample_weight': sample_weight}``
        so the Pipeline's per-fold sklearn fit slices it correctly. Required
        for sample_weight-only classifiers (XGBoost) under
        imbalance_method='class_weight'.

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

    # If not a boosting model, use pooled cross_val_predict (handles repeated CV).
    # The 'model' step prefix matches the codebase convention used by
    # unified_bayesian and nsga2_search when constructing the per-trial Pipeline.
    if not use_early_stopping:
        if sample_weight is not None:
            return cross_val_predict_pooled(
                model, X, y, cv=cv, method=method,
                fit_params={'model__sample_weight': sample_weight},
            )
        return cross_val_predict_pooled(model, X, y, cv=cv, method=method)

    # Manual CV loop with early stopping (handles repeated CV via accumulation)
    repeated = _is_repeated_cv(cv)
    n_samples = X.shape[0]
    use_majority_vote = repeated and method == 'predict' and _model_is_classifier(model)

    # Determine output shape
    if method == 'predict_proba':
        n_classes = len(np.unique(y))
        predictions = np.zeros((n_samples, n_classes))
    else:
        predictions = np.zeros(n_samples)
    if repeated:
        pred_count = np.zeros(n_samples)
        if use_majority_vote:
            votes_per_sample: List[list] = [[] for _ in range(n_samples)]

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # Per-fold slice of sample_weight when provided (mirrors sklearn's
        # native auto-slicing of fit_params arrays whose length matches X).
        sw_train = sample_weight[train_idx] if sample_weight is not None else None

        model_clone = clone(model)
        final_model_clone = _get_model_from_pipeline(model_clone)

        if hasattr(model_clone, 'steps'):
            # Pipeline — transform through preprocessing steps
            X_train_transformed = X_train
            X_val_transformed = X_val

            for step_name, step in model_clone.steps[:-1]:
                if hasattr(step, 'fit_resample'):
                    # Resampler changes y AND sample count; recompute sw_train
                    # from the post-resample y so balanced weighting matches
                    # the reduced/augmented training set.
                    X_train_transformed, y_train = step.fit_resample(X_train_transformed, y_train)
                    if sw_train is not None:
                        from sklearn.utils.class_weight import compute_sample_weight
                        sw_train = compute_sample_weight('balanced', y_train)
                elif hasattr(step, 'transform'):
                    step.fit(X_train_transformed, y_train)
                    X_train_transformed = step.transform(X_train_transformed)
                    X_val_transformed = step.transform(X_val_transformed)

            _fit_with_early_stopping(
                final_model_clone,
                X_train_transformed, y_train,
                X_val_transformed, y_val,
                early_stopping_rounds,
                sample_weight=sw_train,
            )

            if method == 'predict_proba':
                preds = final_model_clone.predict_proba(X_val_transformed)
            else:
                preds = final_model_clone.predict(X_val_transformed)
        else:
            _fit_with_early_stopping(
                model_clone, X_train, y_train, X_val, y_val, early_stopping_rounds,
                sample_weight=sw_train,
            )

            if method == 'predict_proba':
                preds = model_clone.predict_proba(X_val)
            else:
                preds = model_clone.predict(X_val)

        if method != 'predict_proba':
            preds = np.ravel(preds)

        if repeated:
            if use_majority_vote:
                for i, sample_idx in enumerate(val_idx):
                    votes_per_sample[sample_idx].append(preds[i])
            else:
                predictions[val_idx] += preds
            pred_count[val_idx] += 1
        else:
            predictions[val_idx] = preds

    if repeated:
        if use_majority_vote:
            return _majority_vote(votes_per_sample, dtype=np.asarray(y).dtype)
        mask = pred_count > 0
        if method == 'predict_proba':
            predictions[mask] /= pred_count[mask, np.newaxis]
        else:
            predictions[mask] /= pred_count[mask]

    return predictions


def cross_val_score_with_early_stopping(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: str = 'neg_root_mean_squared_error',
    early_stopping_rounds: int = 40,
    n_jobs: int = 1,
    sample_weight: Optional[np.ndarray] = None,
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
    early_stopping_rounds : int, default=50
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
        n_jobs=n_jobs,
        sample_weight=sample_weight,
    )
    return results['test_score']
