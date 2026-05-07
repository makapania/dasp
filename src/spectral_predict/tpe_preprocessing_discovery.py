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


def _quick_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
) -> float:
    """Cross-validated evaluation using LightGBM proxy (PLS fallback).

    Returns
    -------
    score : float
        For regression: negative RMSE (higher is better — Optuna maximises).
        For classification/one-class: balanced accuracy (higher is better).
    """
    import warnings
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier

        n_samples = X.shape[0]
        cv_folds = min(cv_folds, n_samples // 2)
        cv_folds = max(2, cv_folds)

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
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    n_jobs=1,
                )
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
                return scores.mean()

    except Exception:
        # T-37 fix (post-merge review): the classification fallback used to score
        # PLSRegression with `scoring='accuracy'`, which produces NaN folds because
        # PLSRegression returns continuous values. NaN propagates to scores.mean()
        # and Optuna then filters all those trials out as failed — silently turning
        # the LightGBM-unavailable case into a no-op for classification. Use a
        # proper classifier (LogisticRegression on autoscaled inputs) instead.
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
        else:  # one_class — failed trial, don't pollute TPE with garbage
            return -np.inf


def evaluate_config_with_seed(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
    random_state: int,
) -> float:
    """Single-seed CV evaluation that ACTUALLY varies with ``random_state``.

    Differs from ``_quick_evaluate`` in two load-bearing ways (closes
    DeepSeek STRONG D1 from plan-review v3):

    1. The model constructor receives the supplied ``random_state``
       (LightGBM is stochastic via per-feature subsampling, so this
       changes per-seed model fits).
    2. The CV splitter is constructed with ``shuffle=True`` and the
       supplied ``random_state`` (so the train/test partitions vary per
       seed). ``_quick_evaluate`` uses ``cv=cv_folds`` (int), which
       sklearn resolves to non-shuffled KFold for classification and
       regression — deterministic regardless of any passed seed.

    Used ONLY by the multi-seed rescore in the multistart wrapper
    (``run_tpe_multistart_preprocessing_discovery``). The existing
    single-start TPE path continues to call ``_quick_evaluate`` so its
    behavior is bit-exact preserved (no regression in
    ``test_t44_autoscale_wiring`` etc.).

    Returns
    -------
    float
        For regression: negative RMSE (higher is better).
        For classification / one-class: balanced accuracy (higher is better).
    """
    import warnings

    n_samples = X.shape[0]
    cv_folds = min(cv_folds, n_samples // 2)
    cv_folds = max(2, cv_folds)

    # Closes DeepSeek MED #1 (post-Phase-4 review): split the try/except so
    # ImportError (LightGBM not installed) routes to the sklearn fallback for
    # ALL seeds of a given config, while runtime errors during CV (OOM,
    # numerical edge) return -inf for THAT SEED only — without falling through
    # to a different model. Pre-fix, a LightGBM crash mid-run for one seed
    # would silently switch that seed's evaluation to PLS/LogReg, mixing
    # objectives in the multi-seed mean and producing misleading scores.
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier
        _lgbm_available = True
    except ImportError:
        _lgbm_available = False

    if _lgbm_available:
        try:
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
        except Exception:
            # LightGBM is installed but the CV run crashed (OOM, numerical
            # issue, etc.). Behavior depends on task_type:
            #
            # - regression / classification: return -inf for THIS seed
            #   instead of silently falling through to PLS/LogReg. Mixing
            #   objectives across seeds is dishonest; missing one seed's
            #   contribution is fine.
            #
            # - one_class: fall through to the IsolationForest fallback
            #   below. The IF path is the designed alternative for
            #   one_class (added in Fix #2 / DeepSeek H1) — it's not a
            #   foreign objective, it's the documented fallback. Without
            #   this fallthrough, a globally-broken LightGBM (installed
            #   but failing on one_class data) would produce all-inf
            #   union members and the rescore would degenerate to
            #   diversity-only ranking, exactly the H1 failure mode.
            if task_type != 'one_class':
                return float('-inf')
            # else: fall through to the sklearn-only fallback path

    # LightGBM not installed → use the sklearn-only fallback path. Reached
    # for every seed in this branch (consistent objective across seeds).
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
        # one_class fallback: IsolationForest on inlier-only training,
        # score by detection rate on the outlier subset (closes DeepSeek
        # H1 from the post-Phase-4 review). Without this branch the
        # one_class multi-seed rescore degraded to a no-op when LightGBM
        # was unavailable: every config tied at -inf and only diversity
        # selection ranked them.
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

            score = _quick_evaluate(X_eval, y, task_type, cv_folds)
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

        all_configs.append({
            'preprocessing': preproc_name,
            'window': int(window) if window is not None else None,
            'deriv': deriv_order,
            'polyorder': deriv_order + 1 if deriv_order else None,
            'score': -t.value if task_type == 'regression' else t.value,
            'n_wavelengths': X.shape[1],
            'importance_method': 'lightgbm',
            'model_name': None,
            '_tpe_trial_number': t.number,
            '_tpe_autoscale': autoscale,
            '_tpe_baseline_method': baseline_method,
            '_tpe_baseline_params': _BASELINE_DEFAULT_PARAMS.get(baseline_method, {}) if baseline_method else None,
            '_tpe_smoothing': smooth,
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

    # Print summary
    print(f"\n=== TPE Top {len(top_configs)} Configurations ===")
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
        if task_type == 'regression':
            score_str = f"RMSE={cfg['score']:.4f}"
        else:
            score_str = f"Acc={cfg['score']:.4f}"
        print(f"  {i+1}. {full_name}: {score_str}")

    if progress_callback:
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
                X_eval, y, task_type, cv_folds, rs
            )
        except Exception:
            return -np.inf

    rescored, halt_metadata = phase2_adaptive_rescore(
        candidates=union,
        eval_fn=_eval_fn,
        key_fn=_multistart_config_key,
        score_direction="maximize",
        initial_pool_size=len(union),
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

    print(f"\n=== TPE Multistart Top {len(result_configs)} Configurations ===")
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
        print(f"  {i + 1}. {full_name}")

    return result_configs
