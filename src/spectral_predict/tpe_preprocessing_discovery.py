"""
TPE Preprocessing Discovery — Optuna-based preprocessing search.

Replaces the exhaustive basic-preprocessing-discovery and GA paths with a
single TPE (Tree-structured Parzen Estimator) search over a richer 5-D
space:  preproc × window × autoscale × baseline × smoothing.

Architecture: model-agnostic surrogate (LightGBM proxy with PLS fallback).
The TPE study evaluates preprocessing quality using a fast proxy model,
then returns top-N diverse configs.  The caller (search loop) fits ALL
enabled models against each config, preserving model × preprocessing diversity.

This preserves the output contract of ``discover_preprocessing`` — the
search loop does not need to know whether configs came from exhaustive
or TPE search.

References
----------
- Bergstra et al. (2011). "Algorithms for Hyper-Parameter Optimization."
  NeurIPS.  Introduces TPE.
- Akiba et al. (2019). "Optuna."  Proc. ACM SIGKDD.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Callable, Optional, Any

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from .constants import RANDOM_STATE
from .preprocessing_discovery import (
    PREPROCESSING_CANDIDATES,
    apply_preprocessing,
    get_edge_zone,
    select_diverse_configs,
)

# Derivative-aware window ranges (ported from ga_preprocessing.py:75-80).
# Higher derivative orders require larger windows for numerical stability.
DERIVATIVE_WINDOW_RANGES = {
    'deriv1': [5, 7, 9, 11, 13, 15, 17, 19, 21],
    'deriv2': [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    'deriv3': [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35],
    'deriv4': [15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 51],
}

BASELINE_METHODS = [None, 'als', 'polynomial', 'rubber_band', 'airpls']

_BASELINE_DEFAULT_PARAMS = {
    'polynomial': {'degree': 2},
    'als': {'lam': 1e5, 'p': 0.01},
    'rubber_band': {},
    'airpls': {'lam': 1e5},
}


# =============================================================================
# Model-family-aware proxy routing (2026-05-08)
# =============================================================================
#
# The TPE proxy used to be hardcoded LightGBM. On the user's NIR chemometrics
# workflow at n≈49 with PLS downstream, that produced two distinct failure
# modes (PROJECT_STATUS.md 2026-05-08): (1) min_child_samples=20 default
# blocked every split → mean-prediction RMSE for all configs; (2) when the
# proxy was informative, it ranked tree-family preprocessings that PLS
# downstream didn't want (SPXY 20% A/B: R²pred 0.9722 → 0.9405).
#
# Fix: pick the proxy family from the user's enabled-models list. Tree-family
# downstream gets a tree-family proxy with adaptive min_child_samples; linear/
# PLS-family downstream gets a PLS proxy. Mixed → linear (chemometrics-canonical;
# the diversity selector spreads preprocessings across types so the main grid
# still evaluates every type with the user's actual model).

TREE_FAMILY_MODELS = frozenset({
    'LightGBM', 'XGBoost', 'CatBoost', 'RandomForest',
})
LINEAR_FAMILY_MODELS = frozenset({
    'PLS', 'PLS-DA', 'Ridge', 'Lasso', 'ElasticNet',
})
# SVM/SVR/MLP/NeuralBoosted are intentionally uncategorized — they're slow as
# proxies and fall under the resolver's mixed/unknown rule below (→ 'linear').

VALID_PROXY_FAMILIES = frozenset({'tree', 'linear'})


def resolve_tpe_proxy_family(models_to_test) -> str:
    """Pick the TPE proxy family from the user's enabled-models list.

    Returns 'tree' iff the only enabled models are in TREE_FAMILY_MODELS;
    otherwise returns 'linear' (the chemometrics-canonical PLS / LogReg path).

    Unknown model names are silently routed to 'linear' for forward
    compatibility — a typo or future-model in a saved-CSV reload path
    shouldn't crash TPE.
    Empty/None inputs route to 'linear' (no information → safe default).
    """
    if not models_to_test:
        return 'linear'
    has_tree = any(m in TREE_FAMILY_MODELS for m in models_to_test)
    has_linear = any(m in LINEAR_FAMILY_MODELS for m in models_to_test)
    if has_tree and not has_linear:
        return 'tree'
    return 'linear'


# RESERVED — per-trial window filtering would break Optuna's ask/tell contract.
# See SESSION_LOG.md 2026-05-01 GLM 5.1 review for details.
# The current "suggest from union, return -inf on invalid" approach is correct.
def _resolve_window_choices(preproc_name: str) -> List[int]:
    """Return derivative-aware window list for a preprocessing name."""
    for deriv_key in ('deriv4', 'deriv3', 'deriv2', 'deriv1'):
        if deriv_key in preproc_name:
            return DERIVATIVE_WINDOW_RANGES.get(deriv_key, [17])
    return [17]


def _apply_full_preprocessing(
    X: np.ndarray,
    preproc_name: str,
    window: Optional[int],
    autoscale: bool,
    baseline_method: Optional[str],
    smoothing: bool,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
) -> np.ndarray:
    """Apply the full preprocessing chain for a TPE trial.

    Order matches ``build_preprocessing_pipeline``: baseline → smoothing →
    core (SNV/deriv) → autoscale.
    """
    from .baseline import (
        BaselineALS,
        BaselineAirPLS,
        BaselinePolynomial,
        BaselineRubberBand,
    )
    from .preprocess import SNV, SavgolDerivative, SavgolSmooth

    X = np.asarray(X, dtype=np.float64)

    # 1. Baseline correction
    if baseline_method is not None:
        params = _BASELINE_DEFAULT_PARAMS.get(baseline_method, {})
        if baseline_method == 'polynomial':
            X = BaselinePolynomial(degree=params.get('degree', 2)).fit_transform(X)
        elif baseline_method == 'als':
            X = BaselineALS(lambda_=params.get('lam', 1e5), p=params.get('p', 0.01)).fit_transform(X)
        elif baseline_method == 'rubber_band':
            X = BaselineRubberBand().fit_transform(X)
        elif baseline_method == 'airpls':
            X = BaselineAirPLS(lam=params.get('lam', 1e5)).fit_transform(X)

    # 2. Smoothing (Savitzky-Golay).
    # T-37 fix (post-merge review): use the user-selected smoothing_window /
    # smoothing_polyorder rather than hardcoded 17/2 — TPE used to score one
    # smoothing chain and grid then evaluated a different one whenever the user
    # picked non-default smoothing settings.
    if smoothing:
        X = SavgolSmooth(window_length=smoothing_window, polyorder=smoothing_polyorder).fit_transform(X)

    # 3. Core preprocessing (SNV / derivatives)
    if preproc_name == 'raw':
        pass
    elif preproc_name == 'snv':
        X = SNV().fit_transform(X)
    else:
        X = apply_preprocessing(X, preproc_name, window)

    # 4. Autoscale (UV scaling — StandardScaler per column)
    if autoscale:
        X = StandardScaler().fit_transform(X)

    return X


def _quick_evaluate_tree(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
    LGBMRegressor,
    LGBMClassifier,
) -> float:
    """Tree-family LightGBM proxy (non-seeded CV).

    Caller passes in LGBM classes after confirming the import succeeded.

    Uses adaptive ``min_child_samples = max(2, n_train_per_fold // 5)`` so
    splits are legal at chemometrics-sized n. Default LGBM ``min_child_samples=20``
    blocks every split when n_train_per_fold < 40 → mean-prediction collapse;
    the adaptive formula scales with fold size and was validated on the user's
    BoneCollagen + LightGBM workflow at SPXY 20% (PROJECT_STATUS.md 2026-05-08).
    """
    import warnings
    n_samples = X.shape[0]
    n_train_per_fold = n_samples - (n_samples // cv_folds)
    adaptive_mcs = max(2, n_train_per_fold // 5)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        if task_type == 'one_class':
            n_outliers = int(np.sum(y == -1))
            if n_outliers < 2:
                return 0.0
            n_splits = min(cv_folds, n_outliers)
            model = LGBMClassifier(
                class_weight='balanced',
                n_estimators=50,
                max_depth=3,
                min_child_samples=adaptive_mcs,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            )
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy')
            return scores.mean()
        elif task_type == 'classification':
            model = LGBMClassifier(
                n_estimators=50,
                max_depth=4,
                min_child_samples=adaptive_mcs,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            )
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            return scores.mean()
        else:
            model = LGBMRegressor(
                n_estimators=50,
                max_depth=4,
                min_child_samples=adaptive_mcs,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            )
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
            return scores.mean()


def _quick_evaluate_oneclass_iforest(
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """IsolationForest one-class fallback (non-seeded).

    Mirrors ``_evaluate_with_seed_oneclass_iforest`` but uses the global
    ``RANDOM_STATE`` instead of a caller-supplied seed. Used when LightGBM is
    unavailable or crashes on one_class data.
    """
    try:
        from sklearn.ensemble import IsolationForest

        n_outliers = int(np.sum(y == -1))
        if n_outliers < 2:
            return 0.0
        X_inlier = X[y != -1]
        if len(X_inlier) < 5:
            return 0.0
        clf = IsolationForest(
            contamination='auto',
            random_state=RANDOM_STATE,
            n_estimators=50,
            n_jobs=1,
        )
        clf.fit(X_inlier)
        preds = clf.predict(X)
        inlier_mask = y != -1
        outlier_mask = y == -1
        if inlier_mask.sum() == 0 or outlier_mask.sum() == 0:
            return 0.0
        inlier_recall = (preds[inlier_mask] == 1).mean()
        outlier_recall = (preds[outlier_mask] == -1).mean()
        return float((inlier_recall + outlier_recall) / 2)
    except Exception:
        return -np.inf


def _quick_evaluate_linear(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
) -> float:
    """Linear-family proxy (PLS regression / LogReg+StandardScaler classification).

    Handles regression and classification only. Caller routes one_class
    separately (to ``_quick_evaluate_oneclass_iforest``).
    """
    if task_type == 'regression':
        n_components = min(10, X.shape[1] // 10, X.shape[0] // 2)
        n_components = max(2, n_components)
        pls = PLSRegression(n_components=n_components, scale=False)
        scores = cross_val_score(pls, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
        return scores.mean()
    elif task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=1),
        )
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy')
        return scores.mean()
    else:
        raise ValueError(
            f"task_type={task_type!r} is not handled by _quick_evaluate_linear; "
            "caller must route one_class separately"
        )


def _quick_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
    *,
    proxy_family: str = 'linear',
) -> float:
    """Cross-validated proxy evaluation, model-family-aware (2026-05-08).

    The proxy model family is picked by the caller via ``proxy_family``:

    - ``'linear'`` (default): PLS regression / LogReg+StandardScaler
      classification. Chemometrics-canonical, fast, and aligned with PLS-family
      downstream models. Default because mixed/unknown enabled-models lists
      should pick the field-canonical proxy, not the ML-canonical one.
    - ``'tree'``: LightGBM with adaptive ``min_child_samples = max(2, n_train_per_fold // 5)``.
      Used only when ``resolve_tpe_proxy_family`` resolves to tree (i.e. the
      user enabled tree-family models exclusively). The adaptive ``mcs``
      formula scales with fold size so trees can grow on n<50 chemometrics
      datasets.

    One-class is family-independent: routes to LGBM-supervised-on-y_oc when
    LGBM is available, falls back to IsolationForest otherwise. Q3 of the
    plan: there is no PLS one-class variant and tree-family one-class is
    well-served by IF.

    Returns
    -------
    score : float
        For regression: negative RMSE (higher is better — Optuna maximises).
        For classification/one-class: balanced accuracy (higher is better).
    """
    if proxy_family not in VALID_PROXY_FAMILIES:
        raise ValueError(
            f"unknown proxy_family={proxy_family!r}; "
            f"expected one of {sorted(VALID_PROXY_FAMILIES)}"
        )

    n_samples = X.shape[0]
    cv_folds = min(cv_folds, n_samples // 2)
    cv_folds = max(2, cv_folds)

    if task_type == 'one_class':
        # Family-independent: LGBM-supervised-on-y_oc when available, IF otherwise.
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError:
            return _quick_evaluate_oneclass_iforest(X, y)
        try:
            return _quick_evaluate_tree(
                X, y, 'one_class', cv_folds, LGBMRegressor, LGBMClassifier
            )
        except Exception:
            # LGBM installed but crashed (OOM, numerical, etc.) — fall back to IF
            # so the proxy still produces an informative score for one_class
            # configs (closes the H1 failure mode for the single-start path too).
            return _quick_evaluate_oneclass_iforest(X, y)

    if proxy_family == 'tree':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError:
            # Tree requested but LGBM not installed — fall back to linear so
            # the proxy still produces an informative score. Logged once.
            return _quick_evaluate_linear(X, y, task_type, cv_folds)
        try:
            return _quick_evaluate_tree(
                X, y, task_type, cv_folds, LGBMRegressor, LGBMClassifier
            )
        except Exception:
            # LGBM installed but CV crashed — return -inf for THIS trial rather
            # than silently switching to a different family. Mixing objectives
            # is dishonest; missing one trial is fine (TPE handles it).
            return float('-inf')

    return _quick_evaluate_linear(X, y, task_type, cv_folds)


def _evaluate_with_seed_tree(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
    random_state: int,
    LGBMRegressor,
    LGBMClassifier,
) -> float:
    """Tree-family LightGBM proxy with seeded CV.

    Caller passes in LGBM classes after confirming the import succeeded.
    All CV splitters are constructed with ``shuffle=True`` and the supplied
    ``random_state`` so per-seed evaluation actually varies (closes
    DeepSeek STRONG D1).

    Uses adaptive ``min_child_samples = max(2, n_train_per_fold // 5)`` to
    keep splits legal at chemometrics-sized n. Same formula as the
    non-seeded ``_quick_evaluate_tree``.
    """
    import warnings
    n_samples = X.shape[0]
    n_train_per_fold = n_samples - (n_samples // cv_folds)
    adaptive_mcs = max(2, n_train_per_fold // 5)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        if task_type == 'one_class':
            n_outliers = int(np.sum(y == -1))
            if n_outliers < 2:
                return 0.0
            n_splits = min(cv_folds, n_outliers)
            model = LGBMClassifier(
                class_weight='balanced',
                n_estimators=50,
                max_depth=3,
                min_child_samples=adaptive_mcs,
                random_state=random_state,
                verbose=-1,
                n_jobs=1,
            )
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy')
            return scores.mean()
        elif task_type == 'classification':
            model = LGBMClassifier(
                n_estimators=50,
                max_depth=4,
                min_child_samples=adaptive_mcs,
                random_state=random_state,
                verbose=-1,
                n_jobs=1,
            )
            cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            return scores.mean()
        else:
            model = LGBMRegressor(
                n_estimators=50,
                max_depth=4,
                min_child_samples=adaptive_mcs,
                random_state=random_state,
                verbose=-1,
                n_jobs=1,
            )
            cv = KFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            scores = cross_val_score(
                model, X, y, cv=cv, scoring='neg_root_mean_squared_error'
            )
            return scores.mean()


def _evaluate_with_seed_linear(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
    random_state: int,
) -> float:
    """Linear-family fallback (PLS / LogReg+StandardScaler) with seeded CV.

    Handles regression and classification only. Caller routes one_class to
    ``_evaluate_with_seed_oneclass_iforest`` separately.
    """
    if task_type == 'regression':
        n_components = min(10, X.shape[1] // 10, X.shape[0] // 2)
        n_components = max(2, n_components)
        pls = PLSRegression(n_components=n_components, scale=False)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pls, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        return scores.mean()
    elif task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=1, random_state=random_state),
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        return scores.mean()
    else:
        raise ValueError(
            f"task_type={task_type!r} is not handled by _evaluate_with_seed_linear; "
            "caller must route one_class separately"
        )


def _evaluate_with_seed_oneclass_iforest(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> float:
    """IsolationForest one-class fallback with seeded random_state.

    Closes DeepSeek H1 from the post-Phase-4 review: when LightGBM is
    unavailable or crashes on one_class, this path keeps the multi-seed
    rescore informative instead of degenerating to all-inf.
    """
    try:
        from sklearn.ensemble import IsolationForest

        n_outliers = int(np.sum(y == -1))
        if n_outliers < 2:
            return 0.0
        # IF is itself stochastic: random_state controls per-tree
        # subsampling, so seeded re-evaluation produces seed-varying
        # scores even on small datasets.
        X_inlier = X[y != -1]
        if len(X_inlier) < 5:
            return 0.0
        clf = IsolationForest(
            contamination='auto',
            random_state=random_state,
            n_estimators=50,
            n_jobs=1,
        )
        clf.fit(X_inlier)
        # Score on full data: predict -1 for outliers, 1 for inliers.
        # Compare to ground-truth label encoding (y == -1 means outlier).
        preds = clf.predict(X)
        # Balanced accuracy = average of inlier-recall and outlier-recall
        inlier_mask = y != -1
        outlier_mask = y == -1
        if inlier_mask.sum() == 0 or outlier_mask.sum() == 0:
            return 0.0
        inlier_recall = (preds[inlier_mask] == 1).mean()
        outlier_recall = (preds[outlier_mask] == -1).mean()
        return float((inlier_recall + outlier_recall) / 2)
    except Exception:
        return -np.inf


def evaluate_config_with_seed(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
    random_state: int,
    *,
    proxy_family: str = 'linear',
) -> float:
    """Single-seed CV evaluation, model-family-aware (2026-05-08).

    Differs from ``_quick_evaluate`` in that the supplied ``random_state``
    flows into both the model constructor AND a shuffled CV splitter so
    per-seed evaluation actually varies (closes DeepSeek STRONG D1).

    Used ONLY by the multi-seed rescore in the multistart wrapper
    (``run_tpe_multistart_preprocessing_discovery``). The single-start TPE
    path uses ``_quick_evaluate``.

    See ``_quick_evaluate`` for ``proxy_family`` semantics. One-class is
    family-independent (LGBM-supervised when available, IF otherwise).

    Returns
    -------
    float
        For regression: negative RMSE (higher is better).
        For classification / one-class: balanced accuracy (higher is better).
    """
    if proxy_family not in VALID_PROXY_FAMILIES:
        raise ValueError(
            f"unknown proxy_family={proxy_family!r}; "
            f"expected one of {sorted(VALID_PROXY_FAMILIES)}"
        )

    n_samples = X.shape[0]
    cv_folds = min(cv_folds, n_samples // 2)
    cv_folds = max(2, cv_folds)

    if task_type == 'one_class':
        # Family-independent: LGBM-supervised when available, IF otherwise.
        # Preserves DeepSeek H1 (one_class IF fallback) and DeepSeek MED-1
        # (split ImportError / runtime-crash handling).
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            _lgbm_available = True
        except ImportError:
            _lgbm_available = False

        if _lgbm_available:
            try:
                return _evaluate_with_seed_tree(
                    X, y, 'one_class', cv_folds, random_state,
                    LGBMRegressor, LGBMClassifier,
                )
            except Exception:
                # LGBM installed but crashed on one_class data — fall through to IF
                # (DeepSeek H1: keeps the multi-seed rescore informative instead
                # of degenerating to all-inf).
                pass
        return _evaluate_with_seed_oneclass_iforest(X, y, random_state)

    if proxy_family == 'tree':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError:
            # Tree requested but LGBM not installed — fall back to linear.
            return _evaluate_with_seed_linear(X, y, task_type, cv_folds, random_state)
        try:
            return _evaluate_with_seed_tree(
                X, y, task_type, cv_folds, random_state,
                LGBMRegressor, LGBMClassifier,
            )
        except Exception:
            # LGBM installed but THIS seed's CV crashed — return -inf for the
            # seed rather than silently switching to a different family
            # (DeepSeek MED-1: mixing objectives across seeds is dishonest).
            return float('-inf')

    return _evaluate_with_seed_linear(X, y, task_type, cv_folds, random_state)


def run_tpe_preprocessing_discovery(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    n_trials: int = 75,
    n_startup_trials: int = 20,
    n_top: int = 10,
    cv_folds: int = 5,
    enable_autoscale: bool = True,
    enable_baseline: bool = True,
    enable_smoothing: bool = True,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    progress_callback: Optional[Callable] = None,
    random_state: int = RANDOM_STATE,
    skip_diversity: bool = False,
    *,
    proxy_family: str = 'linear',
) -> List[Dict[str, Any]]:
    """TPE-based preprocessing discovery.

    Uses Optuna's TPESampler with ``multivariate=True`` to search a 5-D
    space (preproc × window × autoscale × baseline × smoothing) and return
    top-N diverse preprocessing configurations.

    The output contract is identical to ``discover_preprocessing`` — the
    search loop tests ALL enabled models against each returned config.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features).
    y : np.ndarray
        Target values.
    task_type : str
        'regression', 'classification', or 'one_class'.
    n_trials : int
        Total TPE trials (default 75).
    n_startup_trials : int
        Random startup before TPE kicks in (default 20).
    n_top : int
        Number of diverse top-N configs to return (default 10).
    cv_folds : int
        Cross-validation folds for fitness evaluation (default 5).
    enable_autoscale : bool
        Include autoscale dimension (default True).
    enable_baseline : bool
        Include baseline dimension (default True).
    enable_smoothing : bool
        Include smoothing dimension (default True).
    progress_callback : callable, optional
        ``progress_callback(current, total, message)``.
    random_state : int
        Random seed (default RANDOM_STATE = 42).

    Returns
    -------
    configs : list of dict
        Top-N preprocessing configurations with the same keys as
        ``discover_preprocessing`` output.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    print(f"\n{'='*70}")
    print("TPE PREPROCESSING DISCOVERY")
    print(f"{'='*70}")
    print(f"  Task type: {task_type}")
    print(f"  Proxy family: {proxy_family}")
    print(f"  Data shape: {X.shape}")
    print(f"  Trials: {n_trials} (startup: {n_startup_trials})")
    print(f"  Dimensions: preproc(14) × window × autoscale({enable_autoscale}) × baseline({enable_baseline}) × smoothing({enable_smoothing})")
    print(f"  Looking for top {n_top} diverse configurations")
    print(f"{'='*70}\n")

    preproc_names = [p[0] for p in PREPROCESSING_CANDIDATES]
    preproc_requires_window = {p[0]: p[2] for p in PREPROCESSING_CANDIDATES}

    all_window_choices = sorted({
        w
        for ranges in DERIVATIVE_WINDOW_RANGES.values()
        for w in ranges
    })

    effective_baseline_methods = BASELINE_METHODS if enable_baseline else [None]

    def _objective(trial: optuna.Trial) -> float:
        preproc_name = trial.suggest_categorical('preproc', preproc_names)

        window = trial.suggest_categorical('window', all_window_choices)
        if not preproc_requires_window[preproc_name]:
            window = None

        if enable_autoscale:
            autoscale = trial.suggest_categorical('autoscale', [False, True])
        else:
            autoscale = False

        if enable_baseline and len(effective_baseline_methods) > 1:
            baseline = trial.suggest_categorical(
                'baseline',
                [str(m) for m in effective_baseline_methods],
            )
            baseline_method = None if baseline == 'None' else baseline
        else:
            baseline_method = None

        if enable_smoothing:
            smooth = trial.suggest_categorical('smoothing', [False, True])
        else:
            smooth = False

        # T-37 fix (post-merge review): reject derivative+window combos that
        # don't satisfy SavgolDerivative's window_length >= polyorder + 2 rule
        # up front. The previous code path silently auto-adjusted the window
        # inside apply_preprocessing, but the original (invalid) value is what
        # Optuna stored in trial.params and what the grid search downstream
        # then rebuilt with — causing build_preprocessing_pipeline to crash on
        # the invalid window. Returning -inf here teaches TPE to avoid them.
        if window is not None and preproc_requires_window.get(preproc_name, False):
            valid_windows = _resolve_window_choices(preproc_name)
            if window not in valid_windows:
                return -np.inf

        try:
            X_prep = _apply_full_preprocessing(
                X, preproc_name, window, autoscale, baseline_method, smooth,
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder,
            )

            if not np.isfinite(X_prep).all():
                return -np.inf

            edge_zone = get_edge_zone(preproc_name, window)
            if edge_zone > 0 and X_prep.shape[1] > 2 * edge_zone:
                X_eval = X_prep[:, edge_zone:-edge_zone]
            else:
                X_eval = X_prep

            if np.any(np.std(X_eval, axis=0) < 1e-10):
                return -np.inf

            score = _quick_evaluate(X_eval, y, task_type, cv_folds, proxy_family=proxy_family)
            return score
        except Exception:
            return -np.inf

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        sampler = TPESampler(
            multivariate=True,
            n_startup_trials=n_startup_trials,
            seed=random_state,
        )
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for i in range(n_trials):
        trial = study.ask()
        value = _objective(trial)
        study.tell(trial, value)
        if progress_callback:
            progress_callback(i + 1, n_trials, f"TPE trial {i+1}/{n_trials}")

    # Collect all completed (non-failed) trials
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)
    ]

    if not completed_trials:
        print("WARNING: No valid TPE trials completed!")
        return []

    print(f"  Completed trials: {len(completed_trials)}/{n_trials}")
    if completed_trials:
        best = max(completed_trials, key=lambda t: t.value)
        print(f"  Best trial score: {best.value:.4f}")

    # Build config dicts from completed trials
    all_configs = []
    for t in completed_trials:
        params = t.params
        preproc_name = params['preproc']
        window = params.get('window')
        autoscale = params.get('autoscale', False)
        baseline_str = params.get('baseline', 'None')
        baseline_method = None if baseline_str == 'None' else baseline_str
        smooth = params.get('smoothing', False)

        # T-37 fix (post-merge review): canonicalize the stored window so that
        # raw / snv (which ignore window in the objective) emit window=None and
        # don't survive the dedupe loop below as multiple "different" configs
        # keyed off the bogus suggested-but-unused integer.
        if not preproc_requires_window.get(preproc_name, False):
            window = None

        deriv_order = None
        for d in [4, 3, 2, 1]:
            if f'deriv{d}' in preproc_name:
                deriv_order = d
                break

        display_name = preproc_name
        if baseline_method:
            display_name = f"{baseline_method}+{display_name}"
        if smooth:
            display_name = f"sg0+{display_name}"
        if autoscale:
            display_name = f"{display_name}+autoscale"

        # Audit-trail proxy_model_name reflects what _quick_evaluate actually
        # used (one_class is family-independent — always 'lightgbm' or
        # 'isolation_forest' depending on availability; here we report the
        # nominal family pick).
        if task_type == 'one_class':
            proxy_model_name = 'lightgbm'  # one_class default; IF only fires on LGBM failure
        elif proxy_family == 'tree':
            proxy_model_name = 'lightgbm'
        else:
            proxy_model_name = 'pls' if task_type == 'regression' else 'logreg'

        all_configs.append({
            'preprocessing': preproc_name,
            'window': int(window) if window is not None else None,
            'deriv': deriv_order,
            'polyorder': deriv_order + 1 if deriv_order else None,
            'score': -t.value if task_type == 'regression' else t.value,
            'n_wavelengths': X.shape[1],
            'importance_method': proxy_model_name,
            'model_name': None,
            '_tpe_trial_number': t.number,
            '_tpe_autoscale': autoscale,
            '_tpe_baseline_method': baseline_method,
            '_tpe_baseline_params': _BASELINE_DEFAULT_PARAMS.get(baseline_method, {}) if baseline_method else None,
            '_tpe_smoothing': smooth,
            '_tpe_proxy_family': proxy_family,
            '_tpe_proxy_model_name': proxy_model_name,
        })

    # Deduplicate by (preproc, window, autoscale, baseline, smoothing) tuple
    seen = set()
    unique_configs = []
    for cfg in all_configs:
        key = (
            cfg['preprocessing'],
            cfg['window'],
            cfg['_tpe_autoscale'],
            cfg['_tpe_baseline_method'],
            cfg['_tpe_smoothing'],
        )
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)

    # Select top-N. By default applies select_diverse_configs to spread across
    # preprocessing types. When called from the multistart wrapper with
    # skip_diversity=True, returns the raw top-N by score so the
    # cross-study union doesn't lose configs that would have been exiled
    # by a per-study diversity slot. The multistart wrapper applies its
    # own diversity (keyed on (preproc, autoscale)) at the post-rescore
    # stage. Closes DeepSeek H2 from the post-Phase-4 review.
    if skip_diversity:
        # Sort by score (desc for classification/one_class, asc for regression)
        if task_type == 'regression':
            ranked = sorted(unique_configs, key=lambda c: c.get('score', float('inf')))
        else:
            ranked = sorted(
                unique_configs, key=lambda c: c.get('score', float('-inf')), reverse=True
            )
        top_configs = ranked[:n_top]
    else:
        top_configs = select_diverse_configs(unique_configs, n_top, task_type)

    # Detect proxy collapse: when n_train_per_fold < ~40, LightGBM's default
    # min_child_samples=20 blocks every split and the proxy returns mean-
    # prediction RMSE for every preprocessing — independent of X. Showing the
    # per-config "RMSE=X.XXXX" line in this regime is actively misleading
    # because all values are identical and reflect y-distribution noise, not
    # preprocessing quality. Detect and replace with an honest banner. See
    # PROJECT_STATUS.md 2026-05-08 (latest) entry for full diagnosis.
    _completed_values = np.array([float(t.value) for t in completed_trials])
    proxy_uninformative = bool(
        len(_completed_values) >= 2
        and float(np.std(_completed_values)) < 1e-9
    )

    print(f"\n=== TPE Top {len(top_configs)} Configurations ===")
    if proxy_uninformative:
        # Banner is family-aware: tree path collapse usually means
        # n_train_per_fold is too small even for the adaptive
        # min_child_samples; linear path collapse is unusual (PLS doesn't
        # have the mean-prediction failure mode) and usually indicates
        # near-constant X across configs or a numerical edge.
        if proxy_family == 'tree':
            collapse_diagnosis = (
                "tree proxy (LightGBM) returned identical scores — even with adaptive\n"
                "  min_child_samples, n_train_per_fold may be too small for splits to grow"
            )
        else:
            collapse_diagnosis = (
                "linear proxy (PLS / LogReg) returned identical scores — unusual,\n"
                "  may indicate near-constant X across configs or a numerical edge"
            )
        msg_lines = [
            f"  NOTE: {collapse_diagnosis}.",
            "  At this data size, TPE provides no optimization signal; the "
            "configs below were",
            "  selected by random+diverse sampling across preprocessing types. "
            "They will be",
            "  evaluated by your actual model in the main grid search with "
            "proper CV.",
        ]
        for line in msg_lines:
            print(line)
    for i, cfg in enumerate(top_configs):
        window_str = f"w={cfg['window']}" if cfg['window'] else ""
        extras = []
        if cfg['_tpe_baseline_method']:
            extras.append(cfg['_tpe_baseline_method'])
        if cfg['_tpe_smoothing']:
            extras.append('sg0')
        if cfg['_tpe_autoscale']:
            extras.append('autoscale')
        extras_str = '+'.join(extras)
        full_name = f"{cfg['preprocessing']} {window_str}"
        if extras_str:
            full_name += f" [{extras_str}]"
        if proxy_uninformative:
            print(f"  {i+1}. {full_name}")
        else:
            if task_type == 'regression':
                score_str = f"RMSE={cfg['score']:.4f}"
            else:
                score_str = f"Acc={cfg['score']:.4f}"
            print(f"  {i+1}. {full_name}: {score_str}")

    if progress_callback:
        if proxy_uninformative:
            progress_callback(n_trials, n_trials,
                              "=== TPE Preprocessing Discovery (proxy uninformative) ===")
            if proxy_family == 'tree':
                progress_callback(n_trials, n_trials,
                                  f"  Tree proxy (LightGBM) returned identical scores for all {len(_completed_values)} trials")
                progress_callback(n_trials, n_trials,
                                  "  (n_train_per_fold too small for splits even with adaptive min_child_samples).")
            else:
                progress_callback(n_trials, n_trials,
                                  f"  Linear proxy (PLS/LogReg) returned identical scores for all {len(_completed_values)} trials")
                progress_callback(n_trials, n_trials,
                                  "  (unusual — may indicate near-constant X across configs).")
            progress_callback(n_trials, n_trials,
                              "  Configs below selected by random+diverse sampling, not by RMSE ranking;")
            progress_callback(n_trials, n_trials,
                              "  your actual model will evaluate each one in the main grid search.")
        else:
            progress_callback(n_trials, n_trials, "=== TPE Top Preprocessing Ranking ===")
        for i, cfg in enumerate(top_configs[:10]):
            window_str = f"w={cfg['window']}" if cfg['window'] else ""
            extras = []
            if cfg['_tpe_baseline_method']:
                extras.append(cfg['_tpe_baseline_method'])
            if cfg['_tpe_smoothing']:
                extras.append('sg0')
            if cfg['_tpe_autoscale']:
                extras.append('autoscale')
            extras_str = '+'.join(extras)
            full_name = f"{cfg['preprocessing']} {window_str}"
            if extras_str:
                full_name += f" [{extras_str}]"
            if proxy_uninformative:
                progress_callback(n_trials, n_trials, f"  {i+1}. {full_name}")
            else:
                if task_type == 'regression':
                    score_str = f"RMSE={cfg['score']:.4f}"
                else:
                    score_str = f"Acc={cfg['score']:.4f}"
                progress_callback(n_trials, n_trials, f"  {i+1}. {full_name}: {score_str}")

    return top_configs


# =============================================================================
# Phase 4: Multi-start TPE + multi-seed phase-2 rescore (2026-05-06)
# =============================================================================

# Available start seeds for the multistart wrapper. n_starts in {3, 5, 7}
# selects a prefix of this list. Stable across runs so different invocations
# explore the same regions when n_starts is the same.
_MULTISTART_SEEDS = [42, 0, 7, 100, 31, 17, 88]


def _multistart_config_key(cfg: Dict[str, Any]) -> tuple:
    """Discrete fingerprint of a TPE config for cross-study deduplication.

    Excludes ``n_components`` / continuous params on purpose: the downstream
    grid search reads its own model hyperparameters and would discard those
    anyway. Including them would create false-distinctness across seeds and
    defeat the union-dedup. (DeepSeek plan-review v3 confirmed this is the
    right granularity.)
    """
    return (
        cfg.get('preprocessing'),
        cfg.get('window'),
        cfg.get('_tpe_autoscale', False),
        cfg.get('_tpe_baseline_method'),
        cfg.get('_tpe_smoothing', False),
    )


def run_tpe_multistart_preprocessing_discovery(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    n_trials: int = 75,
    n_top: int = 10,
    cv_folds: int = 5,
    enable_autoscale: bool = True,
    enable_baseline: bool = True,
    enable_smoothing: bool = True,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    n_starts: int = 5,
    per_start_pool: int = 7,
    n_seeds: int = 5,
    progress_callback: Optional[Callable] = None,
    controller=None,
    *,
    proxy_family: str = 'linear',
) -> List[Dict[str, Any]]:
    """Multi-start TPE + phase-2 multi-seed rescore.

    Closes the TPE drift problem documented in
    ``tools/bayesian_topk_stability.py``: pairwise top-K Jaccard ≈ 0
    across 3 seeded ``run_unified_bayesian`` runs because TPE's
    ``n_startup_trials=20`` random phase sees different categorical
    configs per seed and the GMM/KDE models diverge from there. Multi-
    start runs M independent TPE studies with different random_states,
    unions their per-study top candidates, then rescores the union with
    multi-seed CV via ``phase2_adaptive_rescore``.

    Parameters
    ----------
    X, y : np.ndarray
        Spectra and target.
    task_type : str
        ``"regression"``, ``"classification"``, or ``"one_class"``.
    n_trials : int
        Trials per individual TPE study (default 75 — same as
        single-start ``run_tpe_preprocessing_discovery``).
    n_top : int
        Final top-N to return after multi-seed rescore.
    n_starts : int
        Number of independent TPE studies (default 5; recommended floor
        per the empirical exhaustive-seed comparison). Options [3, 5, 7]
        in the GUI.
    per_start_pool : int
        Top-K' per study to feed into the union (default 7).
    n_seeds : int
        Multi-seed rescore breadth (default 5; helper's ``DEFAULT_SEEDS``
        prefix).
    progress_callback : callable, optional
        ``progress_callback(current, total, message)`` style.

    Returns
    -------
    List[Dict[str, Any]]
        Top-N configs in the same shape as
        ``run_tpe_preprocessing_discovery`` output, with rescored mean
        in the ``score`` field. Each dict additionally carries
        ``_tpe_multistart_halt_reason`` for downstream visibility.
    """
    from .phase2_rescore import phase2_adaptive_rescore

    if n_starts > len(_MULTISTART_SEEDS):
        raise ValueError(
            f"n_starts={n_starts} exceeds available start seeds "
            f"({len(_MULTISTART_SEEDS)})"
        )
    seeds_for_starts = _MULTISTART_SEEDS[:n_starts]

    print(f"\n{'='*70}")
    print("TPE MULTI-START PREPROCESSING DISCOVERY")
    print(f"{'='*70}")
    print(f"  n_starts={n_starts} (seeds={seeds_for_starts})")
    print(f"  n_trials per start={n_trials}")
    print(f"  per-start pool={per_start_pool}")
    print(f"  rescore n_seeds={n_seeds}")
    print(f"  task_type={task_type}, n_top={n_top}")
    print(f"{'='*70}\n")

    # Phase 1: run M independent TPE studies and collect per-study top-K'.
    union: list[Dict[str, Any]] = []
    seen_keys: set[tuple] = set()
    for i, seed in enumerate(seeds_for_starts):
        if progress_callback:
            progress_callback(
                i, n_starts, f"TPE start {i + 1}/{n_starts} (seed={seed})"
            )
        per_start_top = run_tpe_preprocessing_discovery(
            X, y,
            task_type=task_type,
            n_trials=n_trials,
            n_top=per_start_pool,
            cv_folds=cv_folds,
            enable_autoscale=enable_autoscale,
            enable_baseline=enable_baseline,
            enable_smoothing=enable_smoothing,
            smoothing_window=smoothing_window,
            smoothing_polyorder=smoothing_polyorder,
            random_state=seed,
            progress_callback=None,  # outer multistart owns the progress reporting
            # Phase 4 fix (DeepSeek H2): skip per-study diversity. The
            # multistart union applies its own diversity (preproc,
            # autoscale) at the post-rescore stage; per-study diversity
            # would exile configs that lose a slot in one study but
            # would survive cross-study rescore.
            skip_diversity=True,
            proxy_family=proxy_family,
        )
        for cfg in per_start_top:
            key = _multistart_config_key(cfg)
            if key not in seen_keys:
                seen_keys.add(key)
                union.append(cfg)

    print(
        f"  Union of {n_starts} studies: {len(union)} unique discrete configs "
        f"(out of {n_starts * per_start_pool} per-study top-K' candidates)"
    )

    if not union:
        print("WARNING: TPE multistart produced empty union")
        return []

    # Phase 2: rescore the union via the shared helper in DEGENERATE mode
    # (single iteration over the entire union — multi-start did the
    # exploration, helper just denoises scoring).
    def _eval_fn(cfg: Dict[str, Any], rs: int) -> float:
        # Apply the full preprocessing chain that produced the original
        # TPE score, then evaluate with the supplied random_state.
        try:
            X_prep = _apply_full_preprocessing(
                X,
                cfg['preprocessing'],
                cfg.get('window'),
                cfg.get('_tpe_autoscale', False),
                cfg.get('_tpe_baseline_method'),
                cfg.get('_tpe_smoothing', False),
                smoothing_window=smoothing_window,
                smoothing_polyorder=smoothing_polyorder,
            )
            if not np.isfinite(X_prep).all():
                return -np.inf
            edge_zone = get_edge_zone(cfg['preprocessing'], cfg.get('window'))
            if edge_zone > 0 and X_prep.shape[1] > 2 * edge_zone:
                X_eval = X_prep[:, edge_zone:-edge_zone]
            else:
                X_eval = X_prep
            return evaluate_config_with_seed(
                X_eval, y, task_type, cv_folds, rs,
                proxy_family=proxy_family,
            )
        except Exception:
            return -np.inf

    rescored, halt_metadata = phase2_adaptive_rescore(
        candidates=union,
        eval_fn=_eval_fn,
        key_fn=_multistart_config_key,
        score_direction="maximize",
        pool_size_progression=[len(union)],  # degenerate: single iteration
        max_pool_multiplier=999,  # cap effectively disabled in degenerate mode
        top_n=n_top,
        n_seeds=n_seeds,
        diversity_key_fn=lambda c: (
            c.get('preprocessing'),
            c.get('_tpe_autoscale', False),
        ),
    )

    halt_reason = halt_metadata['halt_reason']
    winner_scores = halt_metadata.get('winner_scores', [])
    print(
        f"  Phase 2 rescore halted: {halt_reason} "
        f"(evaluated {halt_metadata['candidates_evaluated']} candidates)"
    )

    # Annotate each returned config with:
    # - The rescored multi-seed mean (replaces the original single-seed score
    #   from per-study TPE; closes Codex LOW from post-Phase-4 review).
    # - The multistart halt reason for downstream visibility (closes the
    #   docstring promise; search.py surfaces this in result CSV rows).
    result_configs: list[Dict[str, Any]] = []
    for i, cfg in enumerate(rescored):
        annotated = dict(cfg)
        annotated['_tpe_multistart_halt_reason'] = halt_reason
        if i < len(winner_scores):
            mean_score, std_score = winner_scores[i]
            # eval_fn returns higher-is-better; convert to user-facing
            # convention: regression score = +RMSE, classification score
            # = accuracy. The original cfg['score'] follows the same
            # convention, so for regression we negate (the helper's mean is
            # -RMSE since score_direction='maximize').
            if task_type == 'regression':
                annotated['score'] = -mean_score
            else:
                annotated['score'] = mean_score
            annotated['_tpe_multistart_rescored_std'] = std_score
        result_configs.append(annotated)

    # If the user clicked Stop while Phase 2 rescore was running, Phase 2
    # still ran to completion (it has no cancel hook). Prefix the post-stop
    # output so the user knows these are completion artifacts of work
    # already in flight, not new work happening after they stopped.
    stopped = bool(controller is not None and getattr(controller, "is_ended", False))
    prefix = "[POST-STOP] " if stopped else ""

    header = f"=== TPE Multistart Top {len(result_configs)} Configurations ==="
    print(f"\n{prefix}{header}" if stopped else f"\n{header}")
    if progress_callback:
        progress_callback(n_starts, n_starts, f"{prefix}{header}")
    for i, cfg in enumerate(result_configs):
        window_str = f"w={cfg['window']}" if cfg.get('window') else ""
        extras = []
        if cfg.get('_tpe_baseline_method'):
            extras.append(cfg['_tpe_baseline_method'])
        if cfg.get('_tpe_smoothing'):
            extras.append('sg0')
        if cfg.get('_tpe_autoscale'):
            extras.append('autoscale')
        extras_str = '+'.join(extras)
        full_name = f"{cfg['preprocessing']} {window_str}"
        if extras_str:
            full_name += f" [{extras_str}]"
        # Score sign + format follows the convention established at the rescore
        # block above and in run_tpe_preprocessing_discovery: +RMSE for
        # regression, accuracy / balanced-accuracy otherwise.
        score_val = cfg.get('score')
        std_val = cfg.get('_tpe_multistart_rescored_std')
        if score_val is not None:
            if task_type == 'regression':
                score_str = f"RMSE={score_val:.4f}"
            else:
                score_str = f"score={score_val:.4f}"
            if std_val is not None:
                score_str += f" ±{std_val:.4f}"
            line = f"{prefix}  {i + 1}. {full_name}: {score_str}"
        else:
            line = f"{prefix}  {i + 1}. {full_name}"
        print(line)
        if progress_callback:
            progress_callback(n_starts, n_starts, line)

    return result_configs
