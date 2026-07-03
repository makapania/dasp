"""T-17 multi-target grid orchestrator: preprocessing × varsel × model × hp.

New orchestration layer ABOVE the F2 seed evaluator. Reuses pure primitives
from search.py / preprocess.py / models.py / variable_selection.py / ga_pls.py /
multi_y.py and the F2 cell helper from multitarget_search.py. Never touches
run_search's single-Y path. Grid engine ONLY.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

_SG_SPEC = {"sg1": (1, 2), "sg2": (2, 3), "sg3": (3, 4), "sg4": (4, 5)}


def _base_config(name: str, deriv, window, polyorder, *, interference_to_add,
                 baseline_method, baseline_params, smoothing,
                 smoothing_window, smoothing_polyorder) -> dict[str, Any]:
    return {
        "name": name, "deriv": deriv, "window": window, "polyorder": polyorder,
        "interference": interference_to_add,
        "baseline_method": baseline_method, "baseline_params": baseline_params,
        "smoothing": smoothing, "smoothing_window": smoothing_window,
        "smoothing_polyorder": smoothing_polyorder,
    }


def build_multitarget_preprocess_configs(
    preprocessing_methods: dict[str, bool],
    *,
    window_sizes: Optional[Sequence[int]] = None,
    autoscale: bool = False,
    baseline_method: Optional[str] = None,
    baseline_params: Optional[dict] = None,
    smoothing: bool = False,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    interference_to_add: Any = None,
) -> list[dict[str, Any]]:
    """Enumerate the basic preprocessing grid (mirrors search.py:2288-2616)."""
    pm = preprocessing_methods or {}
    if window_sizes is None:
        window_sizes = [11]
    common = dict(
        interference_to_add=interference_to_add,
        baseline_method=baseline_method, baseline_params=baseline_params,
        smoothing=smoothing, smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder,
    )
    configs: list[dict[str, Any]] = []
    if pm.get("raw", False):
        configs.append(_base_config("raw", None, None, None, **common))
    if pm.get("snv", False):
        configs.append(_base_config("snv", None, None, None, **common))
    for sg, (deriv, poly) in _SG_SPEC.items():
        if not pm.get(sg, False):
            continue
        for w in window_sizes:
            configs.append(_base_config("deriv", deriv, w, poly, **common))
            if pm.get("snv", False):
                configs.append(_base_config("snv_deriv", deriv, w, poly, **common))
            if pm.get("deriv_snv", False):
                configs.append(_base_config("deriv_snv", deriv, w, poly, **common))
    if not configs:
        configs.append(_base_config("raw", None, None, None, **common))

    # Baseline doubling (search.py:2565).
    if baseline_method is not None and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["baseline_method"] = None; no["baseline_params"] = None
            without.append(no)
            bl = dict(cfg); bl["base_name"] = cfg.get("base_name", cfg["name"])
            bl["name"] = f"{baseline_method}+{cfg['name']}"
            with_.append(bl)
        configs = without + with_

    # Smoothing doubling (search.py:2582).
    if smoothing and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["smoothing"] = False
            without.append(no)
            sm = dict(cfg); sm["base_name"] = cfg.get("base_name", cfg["name"])
            nm = cfg["name"]
            sm["name"] = (f"{nm.split('+', 1)[0]}+sg0+{nm.split('+', 1)[1]}"
                          if "+" in nm else f"sg0+{nm}")
            with_.append(sm)
        configs = without + with_

    # Autoscale doubling (search.py:2603).
    if autoscale and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["autoscale"] = False
            without.append(no)
            sc = dict(cfg); sc["autoscale"] = True
            sc["base_name"] = cfg.get("base_name", cfg["name"])
            sc["name"] = cfg["name"] + "+autoscale"
            with_.append(sc)
        configs = without + with_

    return configs


_INTERVAL_METHODS = {"ipls_forward", "ipls_backward", "mc_sipls", "mwpls"}

LINEAR_MODELS = {"PLS", "Ridge", "Lasso", "ElasticNet"}
_IMPORTANCE_METHODS = {"spa", "ga", "importance", "fipls_spa"}


def _parse_ipls_subset_limit(limit: str) -> Optional[int]:
    """'Top N' -> N; 'All' (or anything non-numeric) -> None (no truncation)."""
    if not limit or limit.strip().lower() == "all":
        return None
    digits = "".join(ch for ch in str(limit) if ch.isdigit())
    return int(digits) if digits else None


def _interval_subset_adapter(
    method: str, X_pp: np.ndarray, Y: np.ndarray, wavelengths: np.ndarray,
    *, ipls_subset_limit: str = "Top 10",
) -> list[dict[str, Any]]:
    """Run a multi-Y-safe interval varsel method; return truncated subset dicts."""
    from .variable_selection import ipls_backward, ipls_forward, mc_sipls, mwpls

    fn = {"ipls_forward": ipls_forward, "ipls_backward": ipls_backward,
          "mc_sipls": mc_sipls, "mwpls": mwpls}[method]
    raw = fn(X_pp, Y, wavelengths)  # wavelengths is a required positional
    # Best RMSECV first (finite before non-finite).
    ordered = sorted(raw, key=lambda d: (0 if np.isfinite(d.get("rmsecv", np.inf)) else 1,
                                         d.get("rmsecv", np.inf)))
    limit = _parse_ipls_subset_limit(ipls_subset_limit)
    if limit is not None:
        ordered = ordered[:limit]
    return [
        {"indices": np.asarray(d["indices"]), "tag": d.get("tag", method),
         "method": method}
        for d in ordered
    ]


def verify_spa_multi_y_safe(X: np.ndarray, Y: np.ndarray, *, n_features: int = 5) -> bool:
    """Return True iff spa_selection produces a sane 2-D-Y importance array.

    Gates BOTH ``spa`` and ``fipls_spa`` (which internally calls spa). If SPA's
    R2 criterion degenerates on 2-D Y this returns False and the caller demotes
    both methods to skip-with-notice.
    """
    from .variable_selection import spa_selection

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] < 2:
        return True  # single-Y path is already verified elsewhere
    try:
        imp = np.asarray(spa_selection(X, Y, n_features=n_features), dtype=float)
    except Exception:
        return False
    return bool(
        imp.ndim == 1
        and imp.shape[0] == X.shape[1]
        and np.all(np.isfinite(imp))
        and np.count_nonzero(imp) >= 1
    )


def _importances_to_subsets(
    importances: np.ndarray, method: str, *, variable_counts, n_features_sub: int,
) -> list[dict[str, Any]]:
    """Top-N subsets from an importance array (mirrors search.py:3695/3766)."""
    imp = np.asarray(importances, dtype=float)
    counts = variable_counts or [10, 20, 50, 100, 250, 500, 1000]
    valid = [n for n in counts if n < n_features_sub]
    subsets: list[dict[str, Any]] = []
    for n_top in valid:
        top = np.argsort(imp, kind="stable")[-n_top:][::-1]
        subsets.append({"indices": np.asarray(top), "tag": f"{method}_top{n_top}",
                        "method": method})
    return subsets


def _model_independent_importances(method: str, X_pp: np.ndarray, Y: np.ndarray):
    """spa / ga importance arrays (model-independent, cached per preprocess)."""
    if method == "spa":
        from .variable_selection import spa_selection
        return np.asarray(spa_selection(X_pp, Y, n_features=max(5, X_pp.shape[1] // 10)),
                          dtype=float)
    if method == "ga":
        from .ga_pls import ga_pls_selection
        return np.asarray(ga_pls_selection(X_pp, Y, task_type="regression", verbose=0),
                          dtype=float)
    return None


def _importance_reference_fit(
    model_name: str, X_pp: np.ndarray, Y: np.ndarray, min_fold_train: int,
) -> np.ndarray:
    """Model-specific importances: fit a reference estimator on full X_pp/Y,
    extract a (n_features, n_targets) matrix, aggregate to per-feature scores."""
    from sklearn.multioutput import MultiOutputRegressor

    from .models import get_feature_importances
    from .multi_y import aggregate_importance
    from .multitarget_search import build_multitarget_estimator, resolve_multitarget_strategy

    Y = np.asarray(Y, dtype=float)
    strategy = resolve_multitarget_strategy(model_name)
    est = build_multitarget_estimator(strategy, {}, min_fold_train, X_pp.shape[1])
    est.fit(X_pp, Y)  # importances are rank-only; raw-Y fit is adequate
    n_features = X_pp.shape[1]

    if isinstance(est, MultiOutputRegressor):
        cols = []
        for i, sub in enumerate(est.estimators_):
            imp = get_feature_importances(sub, model_name, X_pp, Y[:, i])
            cols.append(np.asarray(imp, dtype=float).ravel())
        matrix = np.column_stack(cols)  # (n_features, n_targets)
    elif hasattr(est, "feature_importances_"):
        matrix = np.asarray(est.feature_importances_, dtype=float).reshape(n_features, -1)
    elif hasattr(est, "coef_"):
        coef = np.abs(np.asarray(est.coef_, dtype=float))
        matrix = coef.T if coef.ndim == 2 else coef.reshape(n_features, -1)
    else:
        matrix = np.ones((n_features, 1))
    return aggregate_importance(matrix, rule="mean")
