"""Cross-validation utilities, including boosting-round selection for boosters.

For gradient boosting models (XGBoost, CatBoost, LightGBM) the number of
boosting rounds is chosen like the number of PLS latent variables: each CV fold
is fitted once with the maximum round count (no eval_set, so the scored fold's
y never reaches the fit), the folds' predictions at every round count are
pooled, and ONE round count is picked from the pooled CV curve. See
``cross_val_boosting_rounds``. The ``*_with_early_stopping`` wrappers keep
their names for compatibility; ``early_stopping_rounds`` is the patience used
when scanning the pooled curve. Other models fall back to sklearn's standard
cross-validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier
from sklearn.model_selection import cross_validate, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score, mean_absolute_error,
    log_loss
)
from typing import Optional, Dict, Any, Union, List
import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

logger = logging.getLogger(__name__)

# Import boosting model types for detection
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

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
        _majority_label(v)
        for v, k in zip(votes_per_sample, mask) if k
    ])
    return truth_arr, pred_arr


def _majority_label(votes: list):
    """The one repeated-CV voting rule: most votes; a tie goes to the label voted first.

    Votes are appended in fold order, and ``Counter.most_common`` keeps insertion
    order among equal counts. :func:`pooled_round_curve` (booster round selection)
    reproduces exactly this rule, and the exported templates use the same Counter.
    """
    return Counter(votes).most_common(1)[0][0]


def _proba_in_class_order(fitted, proba, classes) -> np.ndarray:
    """``predict_proba`` columns mapped onto the dataset's sorted ``classes``.

    Uses the fitted model's ``classes_`` (zero column for a class the fold model
    never saw), so per-fold probabilities can be accumulated across folds and
    repeats. See ``scoring.align_proba_to_classes``.
    """
    from spectral_predict.scoring import align_proba_to_classes

    model_classes = getattr(fitted, 'classes_', None)
    if model_classes is None and hasattr(fitted, 'steps'):
        model_classes = getattr(fitted.steps[-1][1], 'classes_', None)
    return align_proba_to_classes(proba, model_classes, classes)


def _majority_vote(votes_per_sample: list, dtype) -> np.ndarray:
    """Reduce a list of vote-lists to a single prediction per sample via mode.

    Used for repeated-CV classifier predictions where numeric averaging would
    produce nonsensical fractional labels (e.g. averaging [0, 1] → 0.5).
    """
    n_samples = len(votes_per_sample)
    out = np.empty(n_samples, dtype=dtype)
    for i, votes in enumerate(votes_per_sample):
        if votes:
            out[i] = _majority_label(votes)
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
    balanced_weight_param: Optional[str] = None,
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
    balanced_weight_param : str or None, default=None
        Fit kwarg (e.g. ``'model__sample_weight'``) that receives balanced class
        weights computed from each training fold's y alone. Use this instead of
        slicing weights computed from all of y, which would make a fold's fit
        depend on its test labels.

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
    # sklearn's cross_val_predict(method='predict_proba') label-encodes y before
    # fitting. For labels that are not already 0..K-1 that fits a different
    # model (PLS-DA regresses on the label values), so use the manual loop.
    _y_classes = np.unique(y)
    _labels_are_codes = np.array_equal(_y_classes, np.arange(len(_y_classes)))
    if (
        not _is_repeated_cv(cv)
        and not fit_params
        and balanced_weight_param is None
        and (method != 'predict_proba' or _labels_are_codes)
    ):
        return cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs, method=method)

    def _fold_fit_kwargs(train_idx):
        kwargs = _slice_fit_params(fit_params, train_idx, n_samples)
        if balanced_weight_param is not None:
            from sklearn.utils.class_weight import compute_sample_weight

            kwargs[balanced_weight_param] = compute_sample_weight('balanced', y[train_idx])
        return kwargs

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
            model_clone.fit(X[train_idx], y[train_idx], **_fold_fit_kwargs(train_idx))
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
        model_clone.fit(X[train_idx], y[train_idx], **_fold_fit_kwargs(train_idx))
        if method == 'predict_proba':
            # One column per dataset class: a resampler (e.g. SMOTE-ENN) can
            # remove a class from a training fold, and broadcasting its
            # narrower proba into every column gave rows summing to K.
            preds = _proba_in_class_order(
                model_clone, model_clone.predict_proba(X[test_idx]), _y_classes
            )
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


# ---------------------------------------------------------------------------
# Boosting-round selection from the pooled CV curve
# ---------------------------------------------------------------------------
# The number of boosting rounds is chosen the way chemometrics software chooses
# the number of PLS latent variables: ONE value, read off the pooled
# cross-validation curve. Every CV fold fits the booster once with the maximum
# round count and no eval_set, so the scored fold's y never reaches the fit.
# The fold's test rows are then predicted at every round count, the
# predictions are pooled across folds, and the round count that minimises the
# pooled RMSECV (regression) or maximises pooled accuracy (classification;
# exact ties broken by pooled log-loss) is selected. All CV metrics are
# reported at that single count. The final model
# is the scored configuration truncated: fitted on all calibration data at the
# same maximum round count, then cut to the selected count (truncate_booster),
# so every round-dependent default (CatBoost's automatic learning rate and
# leaf-estimation iterations) is the one the CV folds used.
#
# ``early_stopping_rounds`` keeps its meaning as a patience: the pooled curve
# is scanned from round 1 and the scan stops once that many consecutive rounds
# bring no improvement, keeping the best count seen so far. 0/None scans the
# whole curve.
#
# The previous implementation (``_fit_with_early_stopping``) passed the scored
# test fold as the booster's eval_set, so each fold chose its own tree count
# from the labels it was then scored on (review findings R028/R003/R022). It
# was removed rather than kept as an option.

#: Identity of the round-selection policy. Persisted Bayesian studies that
#: scored boosters include it in their name so trials scored under the old
#: test-fold early stopping are never resumed alongside corrected ones.
BOOSTING_ROUND_POLICY = "pooled_cv_curve_v1"

# CatBoost accepts several aliases for its round count.
_CATBOOST_ROUND_KEYS = ("iterations", "n_estimators", "num_boost_round", "num_trees")
# LightGBM aliases of num_iterations; any of them overrides n_estimators at fit time.
_LGBM_ROUND_ALIASES = (
    "num_iterations",
    "num_iteration",
    "n_iter",
    "num_tree",
    "num_trees",
    "num_round",
    "num_rounds",
    "num_boost_round",
    "nrounds",
    "max_iter",
)
# LightGBM aliases of early_stopping_round (they need an eval set, which is never passed).
_LGBM_EARLY_STOP_KEYS = (
    "early_stopping_round",
    "early_stopping_rounds",
    "early_stopping",
    "n_iter_no_change",
)
_LGBM_BOOSTING_KEYS = ("boosting_type", "boosting", "boost", "boosting_algorithm")
# CatBoost overfitting-detector settings (they also need an eval set).
_CATBOOST_EVAL_KEYS = ("early_stopping_rounds", "od_wait", "od_type", "od_pval")


def _lgbm_round_aliases(params: dict) -> dict:
    """LightGBM round-count aliases set on an estimator, ``{name: value}``."""
    return {k: params[k] for k in _LGBM_ROUND_ALIASES if params.get(k) is not None}


def booster_max_rounds(model) -> int:
    """Return the configured (maximum) number of boosting rounds of a booster.

    LightGBM round aliases (``num_iterations`` etc.) override ``n_estimators`` at
    fit time, so they win here too: ``num_iterations`` (LightGBM's main name) wins
    over every other alias, as LightGBM resolves it.

    Args:
        model: An XGBoost, LightGBM or CatBoost estimator, or a Pipeline ending in one.

    Returns:
        The round count the estimator will fit (its library default when unset).

    Raises:
        ValueError: Conflicting values among LightGBM aliases other than
            ``num_iterations`` (LightGBM's own resolution between them is not defined).
    """
    est = _final_estimator(model)
    params = est.get_params()
    if isinstance(est, XGBOOST_MODELS):
        value = params.get("n_estimators")
        return int(value) if value is not None else 100
    if isinstance(est, LIGHTGBM_MODELS):
        aliases = _lgbm_round_aliases(params)
        if "num_iterations" in aliases:
            return int(aliases["num_iterations"])
        if aliases:
            values = {int(v) for v in aliases.values()}
            if len(values) > 1:
                raise ValueError(f"Conflicting LightGBM round-count aliases: {aliases}")
            return values.pop()
        value = params.get("n_estimators")
        return int(value) if value is not None else 100
    if CATBOOST_AVAILABLE and isinstance(est, CATBOOST_MODELS):
        for key in _CATBOOST_ROUND_KEYS:
            if params.get(key) is not None:
                return int(params[key])
        return 1000
    raise TypeError(f"{type(est).__name__} is not a supported boosting model")


def set_booster_rounds(model, n_rounds: int) -> None:
    """Set the number of boosting rounds on a booster (or a Pipeline's final step) in place.

    LightGBM round aliases present on the estimator are set to the same count, so
    none of them can override it.

    Args:
        model: An XGBoost, LightGBM or CatBoost estimator, or a Pipeline ending in one.
        n_rounds: Round count to fit.
    """
    est = _final_estimator(model)
    n_rounds = int(n_rounds)
    if isinstance(est, XGBOOST_MODELS):
        est.set_params(n_estimators=n_rounds)
        return
    if isinstance(est, LIGHTGBM_MODELS):
        updates = {k: n_rounds for k in _lgbm_round_aliases(est.get_params())}
        updates["n_estimators"] = n_rounds
        est.set_params(**updates)
        return
    if CATBOOST_AVAILABLE and isinstance(est, CATBOOST_MODELS):
        params = est.get_params()
        key = next((k for k in _CATBOOST_ROUND_KEYS if params.get(k) is not None), "iterations")
        est.set_params(**{key: n_rounds})
        return
    raise TypeError(f"{type(est).__name__} is not a supported boosting model")


def _final_estimator(model):
    """Innermost estimator: unwraps any target-transform wrapper exposing
    ``regressor_`` / ``regressor`` (fitted or not) and a Pipeline's final step.

    The exported scripts copy this function's source, so its text must not name
    export-only classes (a script without a Y-transform must not mention one).
    """
    inner = model
    for _ in range(4):
        if hasattr(inner, "regressor_"):
            inner = inner.regressor_
        elif hasattr(inner, "regressor") and not hasattr(inner, "steps"):
            inner = inner.regressor
        elif hasattr(inner, "steps"):
            inner = inner.steps[-1][1]
        else:
            break
    return inner


def _split_target_wrapper(model):
    """``(inner model, target transformer or None)`` for a TransformedTargetRegressor.

    The transformer reproduces the wrapper's own (``transformer``, or ``func`` /
    ``inverse_func``), so CV of the inner model with it equals CV of the wrapper.
    Anything else is returned unchanged with None.
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import FunctionTransformer

    if not isinstance(model, TransformedTargetRegressor):
        return model, None
    if model.transformer is not None:
        return model.regressor, clone(model.transformer)
    if model.func is None and model.inverse_func is None:
        return model.regressor, None
    return model.regressor, FunctionTransformer(
        func=model.func, inverse_func=model.inverse_func, validate=True, check_inverse=False
    )


def truncate_booster(model, n_rounds: int) -> None:
    """Keep only the first ``n_rounds`` rounds of a FITTED booster, in place.

    The final model of a round-selected booster is the scored configuration fitted on
    all calibration data at the maximum round count and then truncated, so its
    predictions equal that fit's staged predictions at ``n_rounds`` exactly and every
    round-dependent default matches the CV folds. The truncated model pickles, and
    its parameters report ``n_rounds``. Works through a Pipeline and a
    TransformedTargetRegressor.

    Args:
        model: Fitted XGBoost, LightGBM or CatBoost estimator, or a wrapper of one.
        n_rounds: Round count to keep.
    """
    est = _final_estimator(model)
    if isinstance(n_rounds, bool) or not isinstance(n_rounds, (int, np.integer)):
        raise ValueError(f"Round count must be a positive integer, got {n_rounds!r}")
    k = int(n_rounds)
    configured = booster_max_rounds(est)
    if not 1 <= k <= configured:
        # Validated before anything is mutated. A LightGBM model may hold FEWER trees
        # than configured (no split improved the loss); that is allowed.
        raise ValueError(f"Round count {k} is outside 1..{configured} (configured rounds)")
    if isinstance(est, XGBOOST_MODELS):
        booster = est.get_booster()
        if booster.num_boosted_rounds() > k:
            est._Booster = booster[:k]
        est.set_params(n_estimators=k)
    elif isinstance(est, LIGHTGBM_MODELS):
        import lightgbm

        if est.booster_.current_iteration() > k:
            est._Booster = lightgbm.Booster(model_str=est.booster_.model_to_string(num_iteration=k))
        updates = {a: k for a in _lgbm_round_aliases(est.get_params())}
        updates["n_estimators"] = k
        est.set_params(**updates)
    elif CATBOOST_AVAILABLE and isinstance(est, CATBOOST_MODELS):
        if est.tree_count_ > k:
            est.shrink(ntree_end=k)
        # CatBoost refuses set_params on a fitted model; record the count it now holds.
        params = est.get_params()
        key = next((a for a in _CATBOOST_ROUND_KEYS if params.get(a) is not None), "iterations")
        est._init_params[key] = k
    else:
        raise TypeError(f"{type(est).__name__} is not a supported boosting model")


def sanitize_booster(model):
    """A clone of ``model`` with eval-only settings removed when it is a booster.

    Boosters are never given an eval_set, whether or not round selection runs, so
    settings that need one (early stopping) would make the fit fail.
    """
    if not is_boosting_model(_final_estimator(model)):
        return model
    model = clone(model)
    strip_eval_only_params(model)
    return model


def round_truncation_from_row(row) -> Optional[tuple]:
    """``(fit_rounds, selected_rounds)`` for a results row whose booster was fitted at
    ``fit_rounds`` and truncated to ``selected_rounds``, else None.

    Rebuilds reproduce such a row by fitting at ``fit_rounds`` and calling
    :func:`truncate_booster` with ``selected_rounds``; the row's ``Params`` carry the
    selected count (a direct fit at that count is identical unless a round-dependent
    default is in play, as with CatBoost's automatic learning rate).
    """
    if row is None:
        return None
    # Mapping / Series rows have .get; DataFrame.itertuples() rows are namedtuples.
    getter = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    if not _parse_bool(getter("round_selection_truncated", None)):
        return None
    fit_rounds = _parse_count(getter("n_estimators_fit", None))
    selected = _parse_count(getter("n_estimators_selected", None))
    if fit_rounds is None or selected is None or selected > fit_rounds:
        if fit_rounds is not None or selected is not None:
            logger.warning(
                "Ignoring inconsistent round-truncation metadata: fit=%r selected=%r",
                getter("n_estimators_fit", None),
                getter("n_estimators_selected", None),
            )
        return None
    return fit_rounds, selected


def parse_bool_cell(value) -> bool:
    """Public form of the row-flag parser (also used by the code generator)."""
    return _parse_bool(value)


def parse_count_cell(value) -> Optional[int]:
    """Public form of the round-count parser (also used by the code generator)."""
    return _parse_count(value)


def _parse_bool(value) -> bool:
    """Row flag as bool: True/1/"True"/"1"/"yes"; NaN, None, "" and anything else False."""
    if value is None or isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value == 1)  # integral floats (1.0) from numeric storage count too
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return False


def _parse_count(value) -> Optional[int]:
    """Positive integer round count from a row cell (int, integral float or digit string)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number) or number < 1 or number != np.floor(number):
        return None
    return int(number)


class _RoundTruncatedBoosterBase(BaseEstimator):
    """Clone-safe "fit at ``fit_rounds``, then keep the first ``n_rounds``" booster.

    A results row with round selection describes the scored configuration fitted at
    its maximum round count and truncated (:func:`truncate_booster`). Code that clones
    and refits base models (ensembles, per-fold refits) would otherwise refit the
    stored count directly, which differs whenever a default depends on the round count
    (CatBoost's automatic learning rate). Each ``fit`` here reproduces the procedure.

    Args:
        estimator: Unfitted booster (or Pipeline ending in one) with the row's params.
        fit_rounds: Rounds to fit (the row's ``n_estimators_fit``).
        n_rounds: Rounds to keep (the row's ``n_estimators_selected``).
    """

    def __init__(self, estimator=None, fit_rounds=None, n_rounds=None):
        self.estimator = estimator
        self.fit_rounds = fit_rounds
        self.n_rounds = n_rounds

    def fit(self, X, y, **fit_params):
        est = sanitize_booster(clone(self.estimator))
        set_booster_rounds(est, self.fit_rounds)
        est.fit(X, y, **fit_params)
        truncate_booster(est, self.n_rounds)
        self.estimator_ = est
        if hasattr(est, "classes_"):
            self.classes_ = est.classes_
        if hasattr(est, "n_features_in_"):
            self.n_features_in_ = est.n_features_in_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class RoundTruncatedRegressor(RegressorMixin, _RoundTruncatedBoosterBase):
    """Regressor form of :class:`_RoundTruncatedBoosterBase`."""


class RoundTruncatedClassifier(ClassifierMixin, _RoundTruncatedBoosterBase):
    """Classifier form of :class:`_RoundTruncatedBoosterBase`."""

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def round_truncated_from_row(estimator, row, task_type: str):
    """Wrap ``estimator`` so every fit reproduces a truncated results row; else return it.

    Args:
        estimator: Unfitted booster built from the row's Params.
        row: Results row (mapping / Series).
        task_type: ``'regression'`` or ``'classification'``.
    """
    truncation = round_truncation_from_row(row)
    if truncation is None or not is_boosting_model(_final_estimator(estimator)):
        return estimator
    cls = RoundTruncatedClassifier if task_type == "classification" else RoundTruncatedRegressor
    return cls(estimator=estimator, fit_rounds=truncation[0], n_rounds=truncation[1])


def strip_eval_only_params(model) -> None:
    """Remove settings that need an eval_set (none is ever passed), in place.

    XGBoost: constructor ``early_stopping_rounds`` and ``EarlyStopping`` callbacks
    (other callbacks are kept). LightGBM: every ``early_stopping_round`` alias.
    CatBoost: overfitting-detector settings and ``use_best_model``.

    Args:
        model: A booster, or a Pipeline / TransformedTargetRegressor wrapping one
            (anything else is left alone).
    """
    est = _final_estimator(model)
    params = est.get_params()
    updates: Dict[str, Any] = {}
    if isinstance(est, XGBOOST_MODELS):
        if params.get("early_stopping_rounds") is not None:
            updates["early_stopping_rounds"] = None
        callbacks = params.get("callbacks")
        if callbacks:
            from xgboost.callback import EarlyStopping

            kept = [cb for cb in callbacks if not isinstance(cb, EarlyStopping)]
            if len(kept) != len(callbacks):
                updates["callbacks"] = kept or None
    elif isinstance(est, LIGHTGBM_MODELS):
        updates = {k: None for k in _LGBM_EARLY_STOP_KEYS if params.get(k) is not None}
    elif CATBOOST_AVAILABLE and isinstance(est, CATBOOST_MODELS):
        # Remove the keys: CatBoost rejects od_type=None at fit, and its get_params
        # drops None values, so a nulled key would break a later sklearn clone.
        for key in _CATBOOST_EVAL_KEYS:
            est._init_params.pop(key, None)
        if params.get("use_best_model"):
            updates["use_best_model"] = False
    if updates:
        est.set_params(**updates)


def round_selection_unsupported_reason(model) -> Optional[str]:
    """Why prefix-based round selection would be invalid for this booster, or None.

    Selection reads the k-round model off a maximum-round fit, which is only
    valid when later rounds never change earlier contributions.

    Args:
        model: A booster or a Pipeline ending in one.

    Returns:
        A user-facing reason, or None when round selection is valid.
    """
    est = _final_estimator(model)
    if not is_boosting_model(est):
        return None
    params = est.get_params()
    if isinstance(est, XGBOOST_MODELS):
        booster = str(params.get("booster") or "gbtree").lower()
        if booster == "gblinear":
            return (
                "XGBoost booster='gblinear' has no per-round predictions "
                "(iteration_range is ignored)"
            )
        if booster == "dart":
            return (
                "XGBoost booster='dart' re-weights earlier trees as rounds are added, so "
                "the first k rounds of a longer fit are not the k-round model"
            )
        if (params.get("rate_drop") or 0) > 0 or params.get("one_drop"):
            return (
                "XGBoost tree dropout (rate_drop / one_drop) re-weights earlier trees as "
                "rounds are added, so the first k rounds of a longer fit are not the "
                "k-round model"
            )
    elif isinstance(est, LIGHTGBM_MODELS):
        for key in _LGBM_BOOSTING_KEYS:
            if str(params.get(key) or "").lower() == "dart":
                return (
                    "LightGBM boosting='dart' re-weights earlier trees as rounds are "
                    "added, so the first k rounds of a longer fit are not the k-round model"
                )
    elif CATBOOST_AVAILABLE and isinstance(est, CATBOOST_MODELS):
        if params.get("model_shrink_rate"):
            return (
                "CatBoost model_shrink_rate shrinks earlier trees as rounds are added, so "
                "the first k rounds of a longer fit are not the k-round model"
            )
        if params.get("posterior_sampling"):
            return (
                "CatBoost posterior_sampling shrinks earlier trees as rounds are added, so "
                "the first k rounds of a longer fit are not the k-round model"
            )
    return None


def _catboost_staged_raw(model, X: np.ndarray) -> np.ndarray:
    """Cumulative raw scores of a fitted CatBoost model, shape (n_rounds, n_samples, dim).

    One leaf-index call plus a cumulative sum over trees, instead of one predict
    call per round (staged_predict costs ~1 ms per round).
    """
    leaf_idx = np.asarray(model.calc_leaf_indexes(X), dtype=np.int64)
    values = np.asarray(model.get_leaf_values(), dtype=float)
    counts = np.asarray(model.get_tree_leaf_counts(), dtype=np.int64)
    dim = values.size // int(counts.sum())
    values = values.reshape(-1, dim)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    contrib = values[offsets[None, :] + leaf_idx]  # (n_samples, n_trees, dim)
    scale, bias = model.get_scale_and_bias()
    raw = np.cumsum(contrib, axis=1) * scale + np.asarray(bias, dtype=float).reshape(1, 1, -1)
    return raw.transpose(1, 0, 2)


def _catboost_staged(model, X: np.ndarray, is_clf: bool) -> np.ndarray:
    """Staged CatBoost predictions (regression) or class probabilities (classification).

    Every candidate is accepted only when its last round equals the model's own
    ``predict`` / ``predict_proba``, so a loss with a link function (Poisson,
    Tweedie: exponent) is staged on the prediction scale, never as raw margins.

    Raises:
        ValueError: No staging matches the native predictions (unsupported loss).
    """
    reference = model.predict_proba(X) if is_clf else np.ravel(model.predict(X))

    def _matches(staged: np.ndarray) -> bool:
        return staged.shape[1:] == reference.shape and bool(
            np.allclose(staged[-1], reference, rtol=1e-6, atol=1e-9)
        )

    try:
        raw = _catboost_staged_raw(model, X)
        if is_clf:
            if raw.shape[2] == 1:
                p1 = 1.0 / (1.0 + np.exp(-raw[:, :, 0]))
                staged = np.stack([1.0 - p1, p1], axis=-1)
            else:
                shifted = raw - raw.max(axis=2, keepdims=True)
                expd = np.exp(shifted)
                staged = expd / expd.sum(axis=2, keepdims=True)
            if _matches(staged):
                return staged
        elif raw.shape[2] == 1:
            for staged in (raw[:, :, 0], np.exp(raw[:, :, 0])):
                if _matches(staged):
                    return staged
    except Exception as exc:  # unusual tree layout: fall back to the slow path
        logger.debug("CatBoost leaf-based staging unavailable (%s); using staged_predict", exc)
    if is_clf:
        staged = np.stack(list(model.staged_predict_proba(X, eval_period=1)))
        if _matches(staged):
            return staged
    else:
        for prediction_type in ("RawFormulaVal", "Exponent"):
            try:
                staged = np.stack(
                    [
                        np.ravel(p)
                        for p in model.staged_predict(
                            X, prediction_type=prediction_type, eval_period=1
                        )
                    ]
                )
            except Exception:  # prediction type not valid for this loss
                continue
            if _matches(staged):
                return staged
    raise ValueError(
        f"Cannot stage CatBoost predictions for loss {model.get_params().get('loss_function')!r}"
    )


def booster_staged_predict(model, X: np.ndarray, n_rounds: Optional[int] = None) -> np.ndarray:
    """Predict with a fitted booster at every round count 1..R.

    Regressors return shape ``(R, n_samples)``. Classifiers return class
    probabilities of shape ``(R, n_samples, n_classes)`` in ``model.classes_``
    order; the predicted label at round r is ``classes_[argmax]`` of row r.

    Args:
        model: A fitted XGBoost, LightGBM or CatBoost estimator (not a Pipeline).
        X: Feature matrix already transformed by any preprocessing steps.
        n_rounds: If given and the booster built fewer rounds (LightGBM stops when no
            split improves the training loss), the last row is repeated up to this
            length. A refit with that many rounds would stop at the same tree.

    Returns:
        Staged predictions as described above.
    """
    is_clf = _model_is_classifier(model)
    if isinstance(model, XGBOOST_MODELS):
        built = int(model.get_booster().num_boosted_rounds())
        fn = model.predict_proba if is_clf else model.predict
        staged = np.stack([np.asarray(fn(X, iteration_range=(0, k))) for k in range(1, built + 1)])
    elif isinstance(model, LIGHTGBM_MODELS):
        built = int(model.booster_.current_iteration())
        fn = model.predict_proba if is_clf else model.predict
        staged = np.stack([np.asarray(fn(X, num_iteration=k)) for k in range(1, built + 1)])
    elif CATBOOST_AVAILABLE and isinstance(model, CATBOOST_MODELS):
        staged = _catboost_staged(model, X, is_clf)
    else:
        raise TypeError(f"{type(model).__name__} is not a supported boosting model")
    if not is_clf and staged.ndim == 3:
        staged = staged.reshape(staged.shape[0], -1)
    if n_rounds is not None and staged.shape[0] < n_rounds:
        pad = np.repeat(staged[-1:], n_rounds - staged.shape[0], axis=0)
        staged = np.concatenate([staged, pad], axis=0)
    elif n_rounds is not None and staged.shape[0] > n_rounds:
        staged = staged[:n_rounds]
    return staged


def booster_predict_at(model, X: np.ndarray, n_rounds: int, method: str = "predict") -> np.ndarray:
    """Predict with a fitted booster using only its first ``n_rounds`` rounds.

    Args:
        model: A fitted XGBoost, LightGBM or CatBoost estimator (not a Pipeline).
        X: Feature matrix already transformed by any preprocessing steps.
        n_rounds: Number of rounds to use.
        method: ``'predict'`` or ``'predict_proba'``.

    Returns:
        Predictions (1-D) or class probabilities (2-D).
    """
    fn = getattr(model, method)
    if isinstance(model, XGBOOST_MODELS):
        out = fn(X, iteration_range=(0, int(n_rounds)))
    elif isinstance(model, LIGHTGBM_MODELS):
        out = fn(X, num_iteration=int(n_rounds))
    elif CATBOOST_AVAILABLE and isinstance(model, CATBOOST_MODELS):
        out = fn(X, ntree_end=int(min(n_rounds, model.tree_count_)))
    else:
        raise TypeError(f"{type(model).__name__} is not a supported boosting model")
    out = np.asarray(out)
    return out if method == "predict_proba" else np.ravel(out)


def staged_labels(staged_proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Turn staged class probabilities ``(R, n, C)`` into staged labels ``(R, n)``."""
    return np.asarray(classes)[np.argmax(staged_proba, axis=2)]


def _align_proba(proba: np.ndarray, fold_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map probability columns of a fold model onto the global class order."""
    fold_classes = np.asarray(fold_classes)
    if fold_classes.shape == classes.shape and np.array_equal(fold_classes, classes):
        return proba
    out = np.zeros(proba.shape[:-1] + (len(classes),))
    out[..., np.searchsorted(classes, fold_classes)] = proba
    return out


def _stage_fold(final, X_test: np.ndarray, max_rounds: int, classes, fitted_tt=None):
    """Staged test predictions of one fitted fold booster.

    Returns:
        ``(staged, proba)``: regression ``(R, n_test)`` values and None; classification
        ``(R, n_test)`` labels and ``(R, n_test, C)`` probabilities in ``classes`` order.
    """
    staged = booster_staged_predict(final, X_test, n_rounds=max_rounds)
    if fitted_tt is not None:
        staged = np.asarray(fitted_tt.inverse_transform(staged.reshape(-1, 1))).reshape(
            staged.shape
        )
    if classes is None:
        return staged, None
    proba = _align_proba(staged, final.classes_, classes)
    return staged_labels(proba, classes), proba


def pooled_round_curve(
    fold_staged: List[np.ndarray],
    fold_test_idx: List[np.ndarray],
    y: np.ndarray,
    task_type: str,
) -> np.ndarray:
    """Pooled CV figure of merit at every round count.

    Regression: RMSE of the per-sample predictions (averaged over repeats under
    repeated CV). Classification: accuracy of the per-sample labels, reduced over
    repeats by the same majority vote as the reported predictions
    (:func:`_majority_label`: most votes; a tie goes to the label voted first, in
    fold order). Each sample appears once per repeat.

    Args:
        fold_staged: Per fold, staged predictions ``(R, n_test)``. For classification
            these are predicted labels (see :func:`staged_labels`).
        fold_test_idx: Per fold, the test-row indices into ``y``.
        y: Targets for all samples.
        task_type: ``'regression'`` or ``'classification'``.

    Returns:
        Array of length R.
    """
    y = np.asarray(y)
    n = y.shape[0]
    n_rounds = fold_staged[0].shape[0]
    counts = np.zeros(n)
    for test_idx in fold_test_idx:
        counts[test_idx] += 1
    mask = counts > 0
    if task_type == "regression":
        sums = np.zeros((n_rounds, n))
        for staged, test_idx in zip(fold_staged, fold_test_idx):
            sums[:, test_idx] += staged
        pred = sums[:, mask] / counts[mask]
        return np.sqrt(np.mean((pred - y[mask].astype(float)) ** 2, axis=1))
    classes = np.unique(y)
    n_folds = len(fold_staged)
    votes = np.zeros((n_rounds, n, len(classes)))
    first_vote = np.full((n_rounds, n, len(classes)), float(n_folds))
    rows = np.arange(n_rounds)[:, None]
    for fold_i, (staged, test_idx) in enumerate(zip(fold_staged, fold_test_idx)):
        cls_idx = np.clip(np.searchsorted(classes, staged), 0, len(classes) - 1)
        where = (rows, np.asarray(test_idx)[None, :], cls_idx)
        votes[where] += 1.0  # one vote per (round, sample) within a fold
        first_vote[where] = np.minimum(first_vote[where], fold_i)
    # Most votes; ties go to the label voted first (Counter.most_common semantics).
    key = votes * (n_folds + 1) - first_vote
    pred = classes[np.argmax(key[:, mask, :], axis=2)]
    return np.mean(pred == y[mask][None, :], axis=1)


def pooled_round_logloss(
    fold_probas: List[np.ndarray],
    fold_test_idx: List[np.ndarray],
    y: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Pooled CV log-loss at every round count (probabilities averaged over repeats).

    Args:
        fold_probas: Per fold, staged probabilities ``(R, n_test, C)`` in ``classes`` order.
        fold_test_idx: Per fold, the test-row indices into ``y``.
        y: Class labels for all samples.
        classes: Sorted class labels.

    Returns:
        Array of length R.
    """
    y = np.asarray(y)
    n = y.shape[0]
    n_rounds = fold_probas[0].shape[0]
    sums = np.zeros((n_rounds, n, len(classes)))
    counts = np.zeros(n)
    for proba, test_idx in zip(fold_probas, fold_test_idx):
        sums[:, test_idx, :] += proba
        counts[test_idx] += 1
    mask = counts > 0
    proba = sums[:, mask, :] / counts[mask][None, :, None]
    y_idx = np.searchsorted(classes, y[mask])
    p_true = proba[:, np.arange(int(mask.sum())), y_idx]
    return -np.mean(np.log(np.clip(p_true, 1e-15, 1.0)), axis=1)


def select_n_rounds(
    curve: np.ndarray,
    patience: Optional[int],
    higher_is_better: bool,
    tiebreak: Optional[np.ndarray] = None,
) -> int:
    """Pick one round count from a pooled CV curve.

    Scans from round 1 and keeps the best value seen; a later round must be
    strictly better to replace it. With ``patience`` > 0 the scan stops once that
    many consecutive rounds bring no improvement (the semantics of
    ``early_stopping_rounds`` in XGBoost/LightGBM/CatBoost, applied to the pooled
    curve instead of a single fold). NaN values never count as improvements.
    When every value is NaN the full curve length is returned.

    Args:
        curve: Figure of merit per round count (index 0 = 1 round).
        patience: Rounds without improvement before the scan stops; 0/None scans all.
        higher_is_better: True for accuracy-like curves, False for RMSE-like curves.
        tiebreak: Optional lower-is-better curve consulted only when ``curve`` ties
            exactly (pooled log-loss for classifiers, whose accuracy curve is a step
            function: without it the earliest round of every plateau would win).

    Returns:
        The selected round count (1-based).

    Raises:
        ValueError: Empty curve.
    """
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        raise ValueError("Cannot select a round count from an empty curve")
    tb = None if tiebreak is None else np.asarray(tiebreak, dtype=float)
    best_i: Optional[int] = None
    best = -np.inf if higher_is_better else np.inf
    best_tb = np.inf
    for i, value in enumerate(curve):
        if np.isnan(value):
            better = False
        elif value != best:
            better = (value > best) if higher_is_better else (value < best)
        else:
            better = tb is not None and not np.isnan(tb[i]) and tb[i] < best_tb
        if better:
            best_i, best = i, value
            best_tb = tb[i] if tb is not None and not np.isnan(tb[i]) else np.inf
        elif best_i is not None and patience and i - best_i >= patience:
            break
    return (best_i if best_i is not None else len(curve) - 1) + 1


def _select_from_staged(
    staged_out: List[np.ndarray],
    proba_out: Optional[List[np.ndarray]],
    test_indices: List[np.ndarray],
    y: np.ndarray,
    classes,
    patience: Optional[int],
):
    """Pooled curve and the selected round count for a set of staged folds.

    Returns:
        ``(n_rounds, curve)``.
    """
    is_clf = classes is not None
    curve = pooled_round_curve(
        staged_out, test_indices, y, "classification" if is_clf else "regression"
    )
    tiebreak = pooled_round_logloss(proba_out, test_indices, y, classes) if is_clf else None
    return select_n_rounds(curve, patience, higher_is_better=is_clf, tiebreak=tiebreak), curve


@dataclass
class BoostingRoundsCV:
    """Cross-validated booster predictions at one pooled-curve round count.

    Attributes:
        n_rounds: Selected round count (1-based).
        max_rounds: Round count each fold model was fitted with.
        curve: Pooled figure of merit per round count (RMSE or accuracy).
        test_indices: Per fold, the test-row indices.
        train_indices: Per fold, the training-row indices.
        fold_predictions: Per fold, predictions (labels for classifiers) at ``n_rounds``.
        fold_probas: Per fold, class probabilities at ``n_rounds`` (classifiers only),
            columns in ``classes`` order.
        classes: Sorted class labels (classifiers only).
        fold_models: Per fold, the fitted (pipeline) clone when ``keep_models`` was set;
            its booster holds ``max_rounds`` rounds, use :func:`booster_predict_at`.
        fold_target_transformers: Per fold, the fitted target transformer (or None)
            when ``keep_models`` was set; the fold model predicts in its space.
        fit_times: Per fold fit time in seconds.
    """

    n_rounds: int
    max_rounds: int
    curve: np.ndarray
    test_indices: List[np.ndarray]
    train_indices: List[np.ndarray]
    fold_predictions: List[np.ndarray]
    fold_probas: Optional[List[np.ndarray]] = None
    classes: Optional[np.ndarray] = None
    fold_models: Optional[list] = None
    fit_times: List[float] = field(default_factory=list)
    fold_target_transformers: Optional[list] = None


def _fit_fold_full_rounds(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sample_weight_train: Optional[np.ndarray],
    balanced_sample_weight: bool,
    target_transformer=None,
):
    """Fit one CV fold with the maximum round count. Never sees the test fold's y.

    Pipelines: preprocessing steps are fitted on the training rows only; samplers
    (``fit_resample``) touch training rows only and are skipped for the test rows.
    When ``sample_weight_train`` is given and a sampler changed the row count, the
    weights are recomputed as balanced weights on the resampled y (unchanged
    behaviour of the previous helper). ``balanced_sample_weight`` computes balanced
    class weights from the (post-resampling) training y alone. ``target_transformer``
    (regression only) is cloned, fitted on the training y and applied BEFORE the
    pipeline steps (samplers included), exactly as TransformedTargetRegressor fits
    the final model.

    Returns:
        (fitted pipeline clone, fitted final estimator, transformed X_test,
        fitted target transformer or None)
    """
    model_clone = clone(model)
    final = _get_model_from_pipeline(model_clone)
    strip_eval_only_params(final)
    Xt_train, Xt_test, yt_train = X_train, X_test, y_train
    fitted_tt = None
    if target_transformer is not None:
        fitted_tt = clone(target_transformer)
        yt_train = np.ravel(fitted_tt.fit_transform(np.asarray(yt_train).reshape(-1, 1)))
    sw = sample_weight_train
    if hasattr(model_clone, "steps"):
        for _, step in model_clone.steps[:-1]:
            if step is None or step == "passthrough":
                continue
            if hasattr(step, "fit_resample"):
                Xt_train, yt_train = step.fit_resample(Xt_train, yt_train)
                if sw is not None:
                    from sklearn.utils.class_weight import compute_sample_weight

                    sw = compute_sample_weight("balanced", yt_train)
            elif hasattr(step, "transform"):
                step.fit(Xt_train, yt_train)
                Xt_train = step.transform(Xt_train)
                Xt_test = step.transform(Xt_test)
    if balanced_sample_weight:
        from sklearn.utils.class_weight import compute_sample_weight

        sw = compute_sample_weight("balanced", yt_train)
    fit_kwargs = {"sample_weight": sw} if sw is not None else {}
    final.fit(Xt_train, yt_train, **fit_kwargs)
    return model_clone, final, Xt_test, fitted_tt


def cross_val_boosting_rounds(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    *,
    patience: Optional[int],
    sample_weight: Optional[np.ndarray] = None,
    balanced_sample_weight: bool = False,
    keep_models: bool = False,
    target_transformer=None,
) -> BoostingRoundsCV:
    """Cross-validate a booster and choose ONE round count from the pooled CV curve.

    Each fold is fitted once with the configured maximum round count and no
    eval_set; its test rows are predicted at every round count. The pooled
    curve (RMSECV for regressors; accuracy for classifiers, exact ties broken by
    pooled log-loss) picks the round count with :func:`select_n_rounds`, and
    every fold's predictions are reported at that count. Each fold chooses its own
    automatic defaults (e.g. CatBoost's learning rate) from its own training data;
    nothing is shared between folds. Changing only a test fold's labels can change
    the selected count (it is a CV statistic, like the PLS LV count) but never the
    fold's fitted model. The final model: fit the same configuration on all data,
    then :func:`truncate_booster` to the selected count.

    Args:
        model: Booster or Pipeline ending in one.
        X: Feature matrix.
        y: Targets.
        cv: Any sklearn splitter (repeated CV and LOO included).
        patience: ``early_stopping_rounds`` semantics on the pooled curve.
        sample_weight: Optional weights for all samples, sliced per training fold. Do
            not pass weights derived from all of y (e.g. balanced class weights): a
            fold's fit would then depend on its test labels. Use
            ``balanced_sample_weight`` for those.
        balanced_sample_weight: Balanced class weights computed from each training
            fold's (post-resampling) y.
        keep_models: Keep the fitted fold clones on the result.
        target_transformer: Regression only. A y transformer (cloned per fold, fitted
            on the training y); staged predictions are inverse-transformed before the
            curve is computed, so RMSECV is on the original y scale.

    Returns:
        A :class:`BoostingRoundsCV`.

    Raises:
        ValueError: The booster's configuration makes round selection invalid (see
            :func:`round_selection_unsupported_reason`).
    """
    import time

    X = np.asarray(X) if not hasattr(X, "iloc") else X.to_numpy()
    y = np.asarray(y)
    model, wrapped_tt = _split_target_wrapper(model)
    if wrapped_tt is not None:
        if target_transformer is not None:
            raise ValueError("Pass a target-transform wrapper or target_transformer, not both")
        target_transformer = wrapped_tt
    final_proto = _get_model_from_pipeline(model)
    if not is_boosting_model(final_proto):
        raise TypeError(f"{type(final_proto).__name__} is not a supported boosting model")
    reason = round_selection_unsupported_reason(final_proto)
    if reason is not None:
        raise ValueError(f"Boosting-round selection is not valid here: {reason}")
    is_clf = _model_is_classifier(final_proto)
    max_rounds = booster_max_rounds(final_proto)
    classes = np.unique(y) if is_clf else None

    staged_out: List[np.ndarray] = []
    proba_out: List[np.ndarray] = []
    test_indices: List[np.ndarray] = []
    train_indices: List[np.ndarray] = []
    models: list = []
    fold_tts: list = []
    fit_times: List[float] = []
    for train_idx, test_idx in cv.split(X, y):
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        start = time.time()
        fitted, final, Xt_test, fitted_tt = _fit_fold_full_rounds(
            model,
            X[train_idx],
            y[train_idx],
            X[test_idx],
            sw_train,
            balanced_sample_weight,
            target_transformer=None if is_clf else target_transformer,
        )
        fit_times.append(time.time() - start)
        staged, proba = _stage_fold(final, Xt_test, max_rounds, classes, fitted_tt)
        staged_out.append(staged)
        if is_clf:
            proba_out.append(proba)
        test_indices.append(np.asarray(test_idx))
        train_indices.append(np.asarray(train_idx))
        if keep_models:
            models.append(fitted)
            fold_tts.append(fitted_tt)

    n_rounds, curve = _select_from_staged(
        staged_out, proba_out if is_clf else None, test_indices, y, classes, patience
    )
    return BoostingRoundsCV(
        n_rounds=n_rounds,
        max_rounds=max_rounds,
        curve=curve,
        test_indices=test_indices,
        train_indices=train_indices,
        fold_predictions=[s[n_rounds - 1] for s in staged_out],
        fold_probas=[p[n_rounds - 1] for p in proba_out] if is_clf else None,
        classes=classes,
        fold_models=models if keep_models else None,
        fit_times=fit_times,
        fold_target_transformers=fold_tts if keep_models else None,
    )


def pool_boosting_predictions(
    res: BoostingRoundsCV, n_samples: int, method: str = "predict", y_dtype=None
) -> np.ndarray:
    """Per-sample CV predictions from a :class:`BoostingRoundsCV`.

    Same reduction as :func:`cross_val_predict_pooled`: one prediction per sample
    under plain K-fold/LOO; under repeated CV, regression and probabilities are
    averaged and classifier labels are majority-voted (:func:`_majority_vote`, the
    rule :func:`pooled_round_curve` selects with).
    """
    is_proba = method == "predict_proba"
    is_clf_labels = (not is_proba) and res.classes is not None
    folds = res.fold_probas if is_proba else res.fold_predictions
    counts = np.zeros(n_samples)
    for test_idx in res.test_indices:
        counts[test_idx] += 1
    repeated = bool(np.any(counts > 1))
    if is_clf_labels:
        if repeated:
            votes: List[list] = [[] for _ in range(n_samples)]
            for preds, test_idx in zip(folds, res.test_indices):
                for i, sample_idx in enumerate(test_idx):
                    votes[sample_idx].append(preds[i])
            return _majority_vote(
                votes, dtype=y_dtype if y_dtype is not None else res.classes.dtype
            )
        out = np.zeros(n_samples, dtype=y_dtype if y_dtype is not None else res.classes.dtype)
        for preds, test_idx in zip(folds, res.test_indices):
            out[test_idx] = preds
        return out
    shape = (n_samples, len(res.classes)) if is_proba else (n_samples,)
    out = np.zeros(shape)
    for preds, test_idx in zip(folds, res.test_indices):
        out[test_idx] += preds
    mask = counts > 0
    if repeated:
        out[mask] = out[mask] / (counts[mask][:, None] if is_proba else counts[mask])
    return out


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
    if scoring == "neg_root_mean_squared_error":
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    elif scoring == "neg_mean_squared_error":
        return -mean_squared_error(y_true, y_pred)
    elif scoring == "r2":
        return r2_score(y_true, y_pred)
    elif scoring == "neg_mean_absolute_error":
        return -mean_absolute_error(y_true, y_pred)
    elif scoring == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif scoring == "f1":
        return f1_score(y_true, y_pred, average="binary", zero_division=0)
    elif scoring == "f1_weighted":
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    elif scoring == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    elif scoring == "precision":
        return precision_score(y_true, y_pred, average="binary", zero_division=0)
    elif scoring == "recall":
        return recall_score(y_true, y_pred, average="binary", zero_division=0)
    else:
        raise ValueError(f"Unsupported scoring method: {scoring}")


def uses_round_selection(model, early_stopping_rounds: Optional[int], warn: bool = True) -> bool:
    """True when ``model`` is a booster, a positive patience was requested and the
    booster's configuration allows prefix-based round selection.

    A booster whose configuration does not allow it (XGBoost gblinear/DART,
    LightGBM DART, CatBoost shrinkage; see :func:`round_selection_unsupported_reason`)
    is fitted at its configured round count instead, with a warning.

    Args:
        model: Estimator or Pipeline.
        early_stopping_rounds: Requested patience.
        warn: Emit the fallback warning.
    """
    est = _final_estimator(model)
    if not (
        is_boosting_model(est) and early_stopping_rounds is not None and early_stopping_rounds > 0
    ):
        return False
    reason = round_selection_unsupported_reason(est)
    if reason is not None:
        if warn:
            warnings.warn(
                f"Boosting-round selection skipped: {reason}. The configured round count "
                "is fitted as is.",
                UserWarning,
                stacklevel=3,
            )
        return False
    return True


def _predict_with_fold_model(
    fitted, X: np.ndarray, n_rounds: int, target_transformer=None
) -> np.ndarray:
    """Predict with a fitted fold clone (pipeline or booster) at ``n_rounds`` rounds.

    ``target_transformer`` is that fold's fitted y transformer: predictions are
    inverse-transformed to the original y scale.
    """
    final = _get_model_from_pipeline(fitted)
    Xt = X
    if hasattr(fitted, "steps"):
        for _, step in fitted.steps[:-1]:
            if step is None or step == "passthrough" or hasattr(step, "fit_resample"):
                continue
            Xt = step.transform(Xt)
    pred = booster_predict_at(final, Xt, n_rounds)
    if target_transformer is not None:
        pred = np.ravel(target_transformer.inverse_transform(np.asarray(pred).reshape(-1, 1)))
    return pred


def _weight_param_name(model) -> str:
    """Fit kwarg that routes sample_weight to the final estimator."""
    if hasattr(model, "steps"):
        return f"{model.steps[-1][0]}__sample_weight"
    return "sample_weight"


def _cross_validate_balanced(
    model, X, y, cv, scoring_dict: Dict[str, str], return_train_score: bool
) -> Dict[str, np.ndarray]:
    """cross_validate with balanced class weights computed from each training fold."""
    import time

    from sklearn.utils.class_weight import compute_sample_weight

    X_arr = np.asarray(X) if not hasattr(X, "iloc") else X.to_numpy()
    y_arr = np.asarray(y)
    key = _weight_param_name(model)
    results: Dict[str, Any] = {"fit_time": [], "score_time": []}
    for name in scoring_dict:
        results[f"test_{name}"] = []
        if return_train_score:
            results[f"train_{name}"] = []
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        fold_model = clone(model)
        start = time.time()
        fold_model.fit(
            X_arr[train_idx],
            y_arr[train_idx],
            **{key: compute_sample_weight("balanced", y_arr[train_idx])},
        )
        results["fit_time"].append(time.time() - start)
        start = time.time()
        y_pred = np.ravel(fold_model.predict(X_arr[test_idx]))
        for name, scorer in scoring_dict.items():
            results[f"test_{name}"].append(_compute_score(y_arr[test_idx], y_pred, scorer))
        if return_train_score:
            y_train_pred = np.ravel(fold_model.predict(X_arr[train_idx]))
            for name, scorer in scoring_dict.items():
                results[f"train_{name}"].append(
                    _compute_score(y_arr[train_idx], y_train_pred, scorer)
                )
        results["score_time"].append(time.time() - start)
    return {k: np.asarray(v) for k, v in results.items()}


def cross_validate_with_early_stopping(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: Union[str, Dict[str, str]] = "neg_root_mean_squared_error",
    early_stopping_rounds: int = 40,
    n_jobs: int = 1,
    return_train_score: bool = False,
    return_estimator: bool = False,
    sample_weight: Optional[np.ndarray] = None,
    balanced_sample_weight: bool = False,
) -> Dict[str, np.ndarray]:
    """Cross-validate, choosing a booster's round count from the pooled CV curve.

    For boosting models (XGBoost, CatBoost, LightGBM) with ``early_stopping_rounds``
    > 0 this runs :func:`cross_val_boosting_rounds`: every fold is fitted without an
    eval_set, ONE round count is chosen from the pooled CV curve, and each fold is
    scored at that count. Other models fall back to sklearn's ``cross_validate``.

    Args:
        model: Estimator or Pipeline.
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        cv: CV splitter (LOO and repeated CV included).
        scoring: Scoring name or dict of names.
        early_stopping_rounds: Patience on the pooled curve; 0/None fits the full
            configured round count.
        n_jobs: Parallel jobs for the sklearn fallback only.
        return_train_score: Also score the training rows (at the selected count).
        return_estimator: Return the fitted estimators (not supported for boosters with
            round selection, whose fold models hold the maximum round count).
        sample_weight: Optional per-sample weights (must not be derived from all of y).
        balanced_sample_weight: Balanced class weights computed from each training
            fold's y (the class_weight path for sample_weight-only classifiers).

    Returns:
        Dict with ``test_<name>`` (or ``test_score``) arrays, ``fit_time``,
        ``score_time`` and, for boosters with round selection, ``n_rounds_selected``.
    """
    import time

    scoring_dict = {"score": scoring} if isinstance(scoring, str) else dict(scoring)
    model = sanitize_booster(model)
    if not uses_round_selection(model, early_stopping_rounds):
        if balanced_sample_weight:
            if return_estimator:
                raise NotImplementedError(
                    "return_estimator is not supported with balanced_sample_weight"
                )
            return _cross_validate_balanced(model, X, y, cv, scoring_dict, return_train_score)
        cv_kwargs: Dict[str, Any] = dict(
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            error_score="raise",
        )
        if sample_weight is not None:
            import sklearn

            # Clone first so set_fit_request doesn't mutate the caller's
            # estimator instance (the routing-state setter is sticky and
            # would persist past the config_context exit, polluting the
            # original model for any downstream caller).
            model_for_routing = clone(model)
            _inner = _get_model_from_pipeline(model_for_routing)
            if hasattr(_inner, "set_fit_request"):
                _inner.set_fit_request(sample_weight=True)
            with sklearn.config_context(enable_metadata_routing=True):
                return cross_validate(
                    model_for_routing,
                    X,
                    y,
                    params={"sample_weight": sample_weight},
                    **cv_kwargs,
                )
        return cross_validate(model, X, y, **cv_kwargs)

    if return_estimator:
        raise NotImplementedError(
            "return_estimator is not supported with boosting-round selection: the fold "
            "models hold the maximum round count. Use cross_val_boosting_rounds(..., "
            "keep_models=True) and booster_predict_at instead."
        )

    X_arr = np.asarray(X) if not hasattr(X, "iloc") else X.to_numpy()
    y_arr = np.asarray(y)
    res = cross_val_boosting_rounds(
        model,
        X_arr,
        y_arr,
        cv,
        patience=early_stopping_rounds,
        sample_weight=sample_weight,
        balanced_sample_weight=balanced_sample_weight,
        keep_models=return_train_score,
    )

    results: Dict[str, Any] = {"fit_time": np.asarray(res.fit_times), "score_time": []}
    for name in scoring_dict:
        results[f"test_{name}"] = []
        if return_train_score:
            results[f"train_{name}"] = []
    for fold_i, (test_idx, y_pred) in enumerate(zip(res.test_indices, res.fold_predictions)):
        start = time.time()
        for name, scorer in scoring_dict.items():
            results[f"test_{name}"].append(_compute_score(y_arr[test_idx], y_pred, scorer))
        if return_train_score:
            train_idx = res.train_indices[fold_i]
            y_train_pred = _predict_with_fold_model(
                res.fold_models[fold_i],
                X_arr[train_idx],
                res.n_rounds,
                target_transformer=res.fold_target_transformers[fold_i],
            )
            for name, scorer in scoring_dict.items():
                results[f"train_{name}"].append(
                    _compute_score(y_arr[train_idx], y_train_pred, scorer)
                )
        results["score_time"].append(time.time() - start)

    for key in list(results):
        results[key] = np.asarray(results[key])
    results["n_rounds_selected"] = res.n_rounds
    return results


def cross_val_predict_with_early_stopping(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    early_stopping_rounds: int = 40,
    method: str = "predict",
    sample_weight: Optional[np.ndarray] = None,
    return_n_rounds: bool = False,
    balanced_sample_weight: bool = False,
):
    """Cross-validated predictions, choosing a booster's round count from the pooled curve.

    Boosters with ``early_stopping_rounds`` > 0 go through
    :func:`cross_val_boosting_rounds` (no eval_set; one round count for all folds).
    Everything else goes through :func:`cross_val_predict_pooled`.

    Args:
        model: Estimator or Pipeline (terminal step named ``model`` when
            weights are used with a non-booster).
        X: Feature matrix.
        y: Target vector.
        cv: CV splitter (LOO and repeated CV included).
        early_stopping_rounds: Patience on the pooled curve; 0/None disables selection.
        method: ``'predict'`` or ``'predict_proba'``.
        sample_weight: Per-sample weights of length n_samples, sliced per training
            fold (must not be derived from all of y).
        return_n_rounds: Also return the selected round count (None when no
            selection ran).
        balanced_sample_weight: Balanced class weights computed from each training
            fold's y.

    Returns:
        Per-sample predictions, or ``(predictions, n_rounds)`` with ``return_n_rounds``.
    """
    model = sanitize_booster(model)
    if not uses_round_selection(model, early_stopping_rounds):
        preds = cross_val_predict_pooled(
            model,
            X,
            y,
            cv=cv,
            method=method,
            fit_params=(
                {"model__sample_weight": sample_weight} if sample_weight is not None else None
            ),
            balanced_weight_param=(_weight_param_name(model) if balanced_sample_weight else None),
        )
        return (preds, None) if return_n_rounds else preds

    y_arr = np.asarray(y)
    res = cross_val_boosting_rounds(
        model,
        X,
        y_arr,
        cv,
        patience=early_stopping_rounds,
        sample_weight=sample_weight,
        balanced_sample_weight=balanced_sample_weight,
    )
    preds = pool_boosting_predictions(res, y_arr.shape[0], method=method, y_dtype=y_arr.dtype)
    return (preds, res.n_rounds) if return_n_rounds else preds


def cross_val_score_with_early_stopping(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: str = "neg_root_mean_squared_error",
    early_stopping_rounds: int = 40,
    n_jobs: int = 1,
    sample_weight: Optional[np.ndarray] = None,
    balanced_sample_weight: bool = False,
) -> np.ndarray:
    """Per-fold CV scores; boosters are scored at one pooled-curve round count.

    Simplified interface to :func:`cross_validate_with_early_stopping`.

    Args:
        model: Estimator or Pipeline.
        X: Feature matrix.
        y: Target vector.
        cv: CV splitter.
        scoring: Scoring name.
        early_stopping_rounds: Patience on the pooled curve (boosters only).
        n_jobs: Parallel jobs (non-boosters only).
        sample_weight: Optional per-sample weights (must not be derived from all of y).
        balanced_sample_weight: Balanced class weights computed from each training
            fold's y.

    Returns:
        Array of per-fold scores.
    """
    results = cross_validate_with_early_stopping(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        sample_weight=sample_weight,
        balanced_sample_weight=balanced_sample_weight,
    )
    return results["test_score"]
