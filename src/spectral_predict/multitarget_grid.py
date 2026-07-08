"""T-17 multi-target grid orchestrator: preprocessing × varsel × model × hp.

New orchestration layer ABOVE the F2 seed evaluator. Reuses pure primitives
from search.py / preprocess.py / models.py / variable_selection.py / ga_pls.py /
multi_y.py and the F2 cell helper from multitarget_search.py. Never touches
run_search's single-Y path. Grid engine ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

_SG_SPEC = {"sg1": (1, 2), "sg2": (2, 3), "sg3": (3, 4), "sg4": (4, 5)}


def _is_joint_capable(model_name: str) -> bool:
    """True iff the model has a joint multi-target variant (resolving it in joint
    mode does not raise). Unknown models are treated as not joint-capable."""
    from .multitarget_search import resolve_multitarget_strategy

    try:
        resolve_multitarget_strategy(model_name, mode="joint")
        return True
    except ValueError:
        # NoJointVariantError (no joint variant) and unknown-model ValueError both
        # mean "cannot run joint here".
        return False


def _expand_model_modes(model_name: str, coupling_mode: str):
    """Map a coupling_mode to the coupling mode(s) to emit for one model.

    Returns (modes, skip_notice) where modes is a list of "independent"/"joint"
    strings and skip_notice is None or a user-facing skip string.

    - "independent": every model -> ["independent"].
    - "joint": joint-capable -> ["joint"]; else [] + skip-with-notice.
    - "both": joint-capable -> ["independent", "joint"]; else ["independent"]
      (a non-joint-capable model still runs its single INDEPENDENT cell).
    """
    cm = (coupling_mode or "independent").lower()
    if cm == "independent":
        return ["independent"], None
    if cm == "joint":
        if _is_joint_capable(model_name):
            return ["joint"], None
        return [], f"{model_name} has no joint variant — skipped in Joint mode"
    if cm == "both":
        if _is_joint_capable(model_name):
            return ["independent", "joint"], None
        return ["independent"], None
    raise ValueError(
        f"Unknown coupling_mode {coupling_mode!r}; expected "
        "'independent', 'joint', or 'both'."
    )


def _preprocess_fingerprint(pc: dict) -> tuple:
    """Fully-discriminating hashable key for a preprocess config.

    ``build_multitarget_preprocess_configs`` emits MANY distinct configs that
    share the same ``name`` string ("deriv", "snv_deriv", "raw") while differing
    in deriv/window/polyorder/baseline_method/smoothing/autoscale/interference.
    Keying the per-preprocess varsel cache by ``pc["name"]`` therefore collides
    two different X blocks on one cache entry (cache bleed -> wrong variables).

    This helper collapses EVERY dimension that changes the produced X block into
    a hashable tuple, so two configs share a fingerprint iff they are identical
    as preprocessors. ``pc.get(...)`` with defaults keeps it crash-safe when keys
    are absent (e.g. ``autoscale`` is only set once autoscale-doubling runs).
    """
    def _freeze(v):
        if isinstance(v, dict):
            return tuple(sorted((k, _freeze(val)) for k, val in v.items()))
        if isinstance(v, (list, tuple)):
            return tuple(_freeze(x) for x in v)
        if isinstance(v, np.ndarray):
            return (tuple(v.shape), tuple(np.asarray(v).ravel().tolist()))
        return v

    return (
        _freeze(pc.get("name")),
        _freeze(pc.get("deriv")),
        _freeze(pc.get("window")),
        _freeze(pc.get("polyorder")),
        _freeze(pc.get("baseline_method")),
        _freeze(pc.get("baseline_params")),
        _freeze(pc.get("smoothing", False)),
        _freeze(pc.get("smoothing_window")),
        _freeze(pc.get("smoothing_polyorder")),
        _freeze(pc.get("autoscale", False)),
        _freeze(pc.get("interference")),
    )


def _describe_preprocess_config(pc: dict) -> str:
    """Compact human-readable preprocessing label including SG deriv/window/polyorder.

    ``pc['name']`` alone (e.g. ``"snv_deriv"``) omits the derivative order,
    Savitzky-Golay window, and polyorder that distinguish otherwise same-named
    cells, so the leaderboard/CSV cannot tell ``snv_deriv d1 w11`` from
    ``snv_deriv d2 w17``. This threads those numeric params onto the composed
    ``name`` (which already encodes baseline / smoothing / autoscale), e.g.
    ``"snv_deriv d2 w17 p3"`` or ``"raw"``.
    """
    name = pc.get("name", "raw")
    parts = [name]
    deriv = pc.get("deriv")
    window = pc.get("window")
    poly = pc.get("polyorder")
    if deriv:
        parts.append(f"d{int(deriv)}")
    if window:
        parts.append(f"w{int(window)}")
    if poly is not None:
        parts.append(f"p{int(poly)}")
    return " ".join(parts)


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
_IMPORTANCE_METHODS = {
    "spa", "ga", "importance", "fipls_spa",
    # T-17 UVE/CARS family, now multi-Y-safe (PLS-2 coefficient tensor / joint
    # criterion). cars-tree / uve_cars_tree remain skipped (per-target LightGBM).
    "uve", "uve_spa", "cars", "uve_cars", "uve_cars_spa", "fipls_cars",
    # T-17 legacy iPLS importance path: multi-Y via native multi-output r2.
    "ipls",
}

# Methods whose internal chain runs SPA on 2-D Y -- gated on the same spa_ok
# verification as bare ``spa`` (if SPA's 2-D criterion degenerates, skip these).
_SPA_DEPENDENT_METHODS = {"spa", "fipls_spa", "uve_spa", "uve_cars_spa"}


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
    # NaN-sink discipline: argsort places NaN LAST (treats it as the largest
    # value), so a raw argsort would hand NaN/inf features back as the "top" set.
    # If nothing is finite there is no meaningful subset; otherwise push every
    # non-finite entry to -inf so it sinks to the bottom instead of the top.
    if not np.any(np.isfinite(imp)):
        return []
    imp = np.where(np.isfinite(imp), imp, -np.inf)
    counts = variable_counts or [10, 20, 50, 100, 250, 500, 1000]
    valid = [n for n in counts if n < n_features_sub]
    subsets: list[dict[str, Any]] = []
    for n_top in valid:
        top = np.argsort(imp, kind="stable")[-n_top:][::-1]
        subsets.append({"indices": np.asarray(top), "tag": f"{method}_top{n_top}",
                        "method": method})
    return subsets


def _model_independent_importances(
    method: str, X_pp: np.ndarray, Y: np.ndarray, wavelengths: Optional[np.ndarray] = None,
    *,
    uve_cutoff_multiplier: float = 1.0, uve_n_components: Optional[int] = None,
):
    """spa / ga / fipls_spa importance arrays (model-independent, cached per preprocess).

    ``wavelengths`` is required by the ``fipls_spa`` branch (forward-iPLS needs
    wavelength positions to build intervals); it is otherwise unused. Kept
    optional/keyword so existing 3-positional-arg callers keep working.

    ``uve_cutoff_multiplier`` / ``uve_n_components`` are forwarded to the UVE
    family (T-17 W2). Note the param-name asymmetry: ``uve_selection`` takes
    ``n_components``; ``uve_spa_selection`` / ``uve_cars_selection`` /
    ``uve_cars_spa_selection`` take ``uve_n_components``.
    """
    if method == "spa":
        from .variable_selection import spa_selection
        return np.asarray(spa_selection(X_pp, Y, n_features=max(5, X_pp.shape[1] // 10)),
                          dtype=float)
    if method == "ga":
        # T-17 W7a: the multi-target grid intentionally routes linear GA only
        # (ga_pls_selection). ga_lightgbm_selection's multi-Y fitness branch is
        # exercised solely by the single-Y search.py path; it is not wired here.
        from .ga_pls import ga_pls_selection
        return np.asarray(ga_pls_selection(X_pp, Y, task_type="regression", verbose=0),
                          dtype=float)
    if method == "fipls_spa":
        from .variable_selection import fipls_spa_selection

        if wavelengths is None:
            raise ValueError("fipls_spa requires wavelengths")
        imp = np.asarray(
            fipls_spa_selection(X_pp, Y, wavelengths), dtype=float,
        )
        if imp.shape != (X_pp.shape[1],):
            raise ValueError(
                f"fipls_spa_selection returned importance with shape {imp.shape}; "
                f"expected ({X_pp.shape[1]},)"
            )
        return imp
    # T-17 UVE/CARS family (multi-Y-safe). All return a full-length (n_features,)
    # importance array, ranked into top-N subsets by _importances_to_subsets.
    if method == "ipls":
        from .variable_selection import ipls_selection
        return np.asarray(ipls_selection(X_pp, Y), dtype=float)
    if method == "uve":
        from .variable_selection import uve_selection
        return np.asarray(
            uve_selection(
                X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier,
                n_components=uve_n_components,
            ), dtype=float,
        )
    if method == "cars":
        from .variable_selection import cars_selection
        return np.asarray(cars_selection(X_pp, Y), dtype=float)
    if method == "uve_cars":
        from .variable_selection import uve_cars_selection
        return np.asarray(
            uve_cars_selection(
                X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier,
                uve_n_components=uve_n_components,
            ), dtype=float,
        )
    if method in ("uve_spa", "uve_cars_spa"):
        n_target = max(5, X_pp.shape[1] // 10)
        if method == "uve_spa":
            from .variable_selection import uve_spa_selection
            return np.asarray(
                uve_spa_selection(
                    X_pp, Y, n_features=n_target,
                    cutoff_multiplier=uve_cutoff_multiplier,
                    uve_n_components=uve_n_components,
                ), dtype=float,
            )
        from .variable_selection import uve_cars_spa_selection
        return np.asarray(
            uve_cars_spa_selection(
                X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier,
                uve_n_components=uve_n_components, spa_n_features=n_target,
            ), dtype=float,
        )
    if method == "fipls_cars":
        from .variable_selection import fipls_cars_selection

        if wavelengths is None:
            raise ValueError("fipls_cars requires wavelengths")
        imp = np.asarray(fipls_cars_selection(X_pp, Y, wavelengths), dtype=float)
        if imp.shape != (X_pp.shape[1],):
            raise ValueError(
                f"fipls_cars_selection returned importance with shape {imp.shape}; "
                f"expected ({X_pp.shape[1]},)"
            )
        return imp
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
        # Native models lacking both feature_importances_ and coef_ (e.g. sklearn
        # MLPRegressor, which exposes coefs_ as a list) must NOT collapse to a
        # uniform np.ones vector -- that yields meaningless top-N varsel subsets.
        # Delegate to the shared single-Y importance extractor (|first-layer
        # weights| for MLP) so multi-target parity holds with single-Y behavior.
        imp = np.asarray(
            get_feature_importances(est, model_name, X_pp, Y), dtype=float
        ).ravel()
        matrix = imp.reshape(n_features, -1)
    return aggregate_importance(matrix, rule="mean")


# T-17 W6a: still skip-with-notice on 2-D Y. cars-tree / uve_cars_tree run a
# per-target LightGBM (wall-clock explosion). vcpa-iriv's leaf has a multi-Y
# PLS criterion but is NOT wired into the interval/importance routes here --
# the grid never reaches it, so it stays a skip-with-notice (NOT because it is
# "single-Y-only" -- the leaf does support 2-D PLS mode).
SKIP_WITH_NOTICE = {
    "cars-tree", "uve_cars_tree", "vcpa-iriv",
}


def classify_varsel_method(method: str, *, enabled_models, spa_ok: bool) -> str:
    """Route a GUI varsel method string to 'subset' / 'importance' / 'skip'."""
    if method in SKIP_WITH_NOTICE:
        return "skip"
    if method in _INTERVAL_METHODS:
        return "subset"
    if method in _SPA_DEPENDENT_METHODS:
        return "importance" if spa_ok else "skip"
    if method == "ga":
        has_linear = any(m in LINEAR_MODELS for m in (enabled_models or []))
        return "importance" if has_linear else "skip"
    if method in _IMPORTANCE_METHODS:
        return "importance"
    return "skip"


def build_multitarget_varsel_subsets(
    methods, X_pp, Y, wavelengths, *, enabled_models, variable_counts,
    ipls_subset_limit, spa_ok, cache, preprocess_id,
    apply_uve_prefilter: bool = False,
    uve_cutoff_multiplier: float = 1.0, uve_n_components: Optional[int] = None,
):
    """Return (subsets_incl_full, skipped_notices), caching per (preprocess, method).

    ``uve_cutoff_multiplier`` / ``uve_n_components`` are forwarded to the UVE
    family (T-17 W2): the prefilter ``get_uve_threshold`` (via ``n_components``)
    and the UVE importance producers in ``_model_independent_importances``.
    """
    n_features_sub = int(X_pp.shape[1])
    subsets: list[dict[str, Any]] = [
        {"indices": np.arange(n_features_sub), "tag": "full", "method": "full"}
    ]
    skipped: list[str] = []
    if apply_uve_prefilter:
        # T-17: UVE is now multi-Y-safe, so the prefilter contributes its
        # UVE-threshold-selected variable set as a standalone candidate subset
        # (rather than being skipped). Cached per preprocess like other methods.
        key = (preprocess_id, "apply_uve_prefilter")
        if key not in cache:
            try:
                from .variable_selection import get_uve_threshold

                _imp, _thr, mask = get_uve_threshold(
                    X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier,
                    n_components=uve_n_components,
                )
                idx = np.where(np.asarray(mask, dtype=bool))[0]
                # Only useful if UVE actually eliminated something (a keep-all or
                # keep-none mask adds nothing over the always-present full subset).
                if 0 < idx.size < n_features_sub:
                    cache[key] = [{"indices": idx, "tag": "uve_prefilter",
                                   "method": "apply_uve_prefilter"}]
                else:
                    cache[key] = []
            except Exception:
                cache[key] = []
                skipped.append("apply_uve_prefilter")
        subsets.extend(
            {**d, "indices": np.array(d["indices"], copy=True)} for d in cache[key]
        )
    for method in (methods or []):
        kind = classify_varsel_method(method, enabled_models=enabled_models, spa_ok=spa_ok)
        if kind == "skip":
            skipped.append(method)
            continue
        key = (preprocess_id, method)
        if kind == "subset":
            if key not in cache:
                # First-pass resilience (mirrors the per-cell NaN-sink discipline):
                # one raising (preprocess, method) must NOT abort the whole search.
                try:
                    cache[key] = _interval_subset_adapter(
                        method, X_pp, Y, wavelengths, ipls_subset_limit=ipls_subset_limit
                    )
                except Exception:
                    if method not in skipped:
                        skipped.append(method)
                    continue
            # Copy-on-return: hand out fresh dicts/arrays so a caller mutating a
            # returned subset's indices cannot poison the cached entry.
            subsets.extend(
                {**d, "indices": np.array(d["indices"], copy=True)} for d in cache[key]
            )
        elif kind == "importance":
            if method == "importance":
                # model-specific: orchestrator resolves per (preprocess, model).
                subsets.append({"method": "importance", "model_specific": True})
                continue
            if key not in cache:
                try:
                    imp = _model_independent_importances(
                        method, X_pp, Y, wavelengths=wavelengths,
                        uve_cutoff_multiplier=uve_cutoff_multiplier,
                        uve_n_components=uve_n_components,
                    )
                    cache[key] = _importances_to_subsets(
                        imp, method, variable_counts=variable_counts,
                        n_features_sub=n_features_sub,
                    )
                except Exception:
                    if method not in skipped:
                        skipped.append(method)
                    continue
            # Copy-on-return (see the subset branch above).
            subsets.extend(
                {**d, "indices": np.array(d["indices"], copy=True)} for d in cache[key]
            )
    return subsets, skipped


def _dedup_model_configs(model_grids: dict) -> list[dict[str, Any]]:
    """Flatten get_model_grids output to [{'model_name','params'}], deduped by the
    FULL consumed keyset (builders now consume all keys, so this is a safety net).
    """
    out, seen = [], set()
    for model_name, configs in model_grids.items():
        for _estimator, params in configs:
            key = (model_name, frozenset(
                (k, tuple(v) if isinstance(v, (list, np.ndarray)) else v)
                for k, v in params.items()
            ))
            if key in seen:
                continue
            seen.add(key)
            out.append({"model_name": model_name, "params": dict(params)})
    return out


def _cap_and_dedup_pls_for_subset(
    model_cfgs: list[dict[str, Any]], n_features_sub: int, min_fold_train: int,
) -> list[dict[str, Any]]:
    """Cap PLS n_components to the per-subset cap (cap_components) BEFORE dedup,
    so narrow subsets do not spawn duplicate effective cells with overstated
    n_components. Non-PLS configs pass through unchanged. Returns configs whose
    params hold the EFFECTIVE post-cap values.
    """
    from .multi_y import cap_components

    out, seen = [], set()
    for mc in model_cfgs:
        params = dict(mc["params"])
        if mc["model_name"] == "PLS" and "n_components" in params:
            params["n_components"] = cap_components(
                min_fold_train, n_features_sub, int(params["n_components"])
            )
        key = (mc["model_name"], frozenset(
            (k, tuple(v) if isinstance(v, (list, np.ndarray)) else v)
            for k, v in params.items()
        ))
        if key in seen:
            continue
        seen.add(key)
        out.append({"model_name": mc["model_name"], "params": params})
    return out


def _apply_preprocess_and_restrict(
    pc: dict, X_arr: np.ndarray, Y_arr: np.ndarray, wl: np.ndarray, *,
    wl_min, wl_max, wl_regions, restriction_active: bool,
):
    """Build one cell's preprocessed X block (the deterministic first-pass step).

    Reproduces the exact preprocessing + wavelength-restriction / edge-mask
    transform that :func:`run_multitarget_grid_search` applies per preprocess
    config, and additionally returns the FITTED preprocessing ``Pipeline`` (or
    ``None`` for raw). The grid's search loop discards the pipeline (it only
    needs ``X_pp``); :func:`refit_multitarget_final` keeps it to persist a
    reload-capable ``.dasp``. Single source of truth so the refit block can
    never drift from what the search evaluated.

    Returns ``(X_pp, wl_pp, preprocessor)``.
    """
    from sklearn.pipeline import Pipeline

    from .preprocess import build_preprocessing_pipeline
    from .search import _apply_edge_mask_to_data

    name = pc.get("base_name", pc["name"])
    steps = build_preprocessing_pipeline(
        name, pc["deriv"], pc["window"], pc["polyorder"],
        task_type="regression", interference=pc.get("interference"), wavelengths=wl,
        baseline_method=pc.get("baseline_method"), baseline_params=pc.get("baseline_params"),
        smoothing=pc.get("smoothing", False), smoothing_window=pc.get("smoothing_window", 17),
        smoothing_polyorder=pc.get("smoothing_polyorder", 2), autoscale=pc.get("autoscale", False),
    )
    preprocessor = None
    X_pp = X_arr.copy()
    if steps:
        preprocessor = Pipeline(steps)
        X_pp = preprocessor.fit_transform(X_pp, Y_arr)
    wl_pp = wl
    # Wavelength restriction (search.py:2777) then edge-mask if unrestricted (search.py:2876).
    if restriction_active:
        if wl_regions:
            mask = np.zeros(wl_pp.shape[0], dtype=bool)
            for lo, hi in wl_regions:
                mask |= (wl_pp >= lo) & (wl_pp <= hi)
        else:
            mask = np.ones(wl_pp.shape[0], dtype=bool)
            if wl_min is not None:
                mask &= wl_pp >= wl_min
            if wl_max is not None:
                mask &= wl_pp <= wl_max
        X_pp, wl_pp = X_pp[:, mask], wl_pp[mask]
        if X_pp.shape[1] == 0:
            raise ValueError(
                "Wavelength restriction excludes all wavelengths; "
                "no spectral columns remain."
            )
    elif pc.get("deriv") and pc.get("window"):
        X_pp, wl_pp, _ = _apply_edge_mask_to_data(X_pp, wl_pp, pc)
    return X_pp, wl_pp, preprocessor


def run_multitarget_grid_search(
    X, Y, *, model_names, target_names, wavelengths,
    preprocessing_methods, autoscale,
    baseline_method=None, baseline_params=None,
    smoothing=False, smoothing_window=17, smoothing_polyorder=2,
    interference_to_add=None, wavelength_restriction=None,
    variable_selection_methods, variable_counts=None,
    apply_uve_prefilter: bool = False,
    uve_cutoff_multiplier: float = 1.0, uve_n_components: Optional[int] = None,
    ipls_subset_limit="Top 10", tier="standard", model_grid_overrides=None,
    max_n_components=10, max_iter=500, window_sizes=None,
    cv="kfold", n_folds=5, n_repeats=5, random_state=42,
    optimization_method="grid", controller=None, progress_callback=None, n_jobs=-1,
    coupling_mode: str = "independent",
):
    from .cv_utils import build_cv_splitter
    from .models import get_model_grids
    from .multi_y import inter_target_correlation
    from .multitarget_search import MultiTargetSearchOutput, _evaluate_multitarget_cell

    if optimization_method != "grid":
        raise ValueError(
            f"Multi-target grid search is Grid-engine ONLY; got "
            f"optimization_method={optimization_method!r}. Bayesian/NSGA-II are 1-D-only."
        )

    coupling_mode = (coupling_mode or "independent").lower()
    if coupling_mode not in ("independent", "joint", "both"):
        raise ValueError(
            f"coupling_mode must be 'independent', 'joint', or 'both'; got "
            f"{coupling_mode!r}."
        )

    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    wl = np.asarray(wavelengths, dtype=float)
    target_names = list(target_names)
    correlation = inter_target_correlation(Y_arr)

    # Grid-only gate for spa/fipls_spa inclusion (verified once on the raw block).
    # T-17 W1: only run the SPA preflight when the user actually selected a
    # SPA-dependent method; otherwise short-circuit to spa_ok=True (the check is
    # expensive and irrelevant when no SPA chain is in play).
    requested_methods = set(variable_selection_methods or [])
    spa_ok = (
        verify_spa_multi_y_safe(X_arr, Y_arr)
        if requested_methods & _SPA_DEPENDENT_METHODS
        else True
    )

    preprocess_configs = build_multitarget_preprocess_configs(
        preprocessing_methods, window_sizes=window_sizes, autoscale=autoscale,
        baseline_method=baseline_method, baseline_params=baseline_params,
        smoothing=smoothing, smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder, interference_to_add=interference_to_add,
    )

    wl_min = wl_max = wl_regions = None
    if wavelength_restriction:
        wl_min = wavelength_restriction.get("min")
        wl_max = wavelength_restriction.get("max")
        wl_regions = wavelength_restriction.get("regions")
    restriction_active = bool(wl_regions or wl_min is not None or wl_max is not None)

    splitter = build_cv_splitter(
        cv, n_folds, "regression", n_repeats=n_repeats, random_state=random_state, y=None,
    ) if isinstance(cv, str) else cv
    min_fold_train = min(len(tr) for tr, _ in splitter.split(X_arr, Y_arr))

    results = []
    skipped_all: list[str] = []
    varsel_cache: dict = {}
    best_finite = None
    # First pass builds the full cell list so 'total' is honest.
    cells = []  # (preprocess_cfg, X_sub, wl_sub, subset_dict, model_cfg, coupling_mode)

    def _emit_cells(pc_, X_, tag_, method_, mc_):
        modes, notice = _expand_model_modes(mc_["model_name"], coupling_mode)
        if notice is not None and notice not in skipped_all:
            skipped_all.append(notice)
        for _mode in modes:
            cells.append((pc_, X_, tag_, method_, mc_, _mode))

    for pc in preprocess_configs:
        if controller is not None and not controller.check_and_wait():
            break
        name = pc.get("base_name", pc["name"])
        # T-17 W1-2: the first/varsel pass (this loop) is the expensive part on
        # real configs (UVE/CARS/iPLS dominate wall-clock), but it emitted NO
        # progress, so a consumer saw "nothing happens" before the modeling
        # pass. Emit through the SAME callback + payload shape the modeling pass
        # uses below (message/current/total/best_model). current/total are 0/0
        # because the honest cell total is unknown until this pass finishes.
        # Placed AFTER check_and_wait so a stop suppresses the emission and a
        # pause blocks before it -- no new pause point, no resume disruption.
        if progress_callback is not None:
            progress_callback({
                "message": f"Variable selection: preprocessing '{name}' ...",
                "current": 0, "total": 0, "best_model": None,
            })
        X_pp, wl_pp, _ = _apply_preprocess_and_restrict(
            pc, X_arr, Y_arr, wl,
            wl_min=wl_min, wl_max=wl_max, wl_regions=wl_regions,
            restriction_active=restriction_active,
        )

        subsets, skipped = build_multitarget_varsel_subsets(
            variable_selection_methods, X_pp, Y_arr, wl_pp,
            enabled_models=model_names, variable_counts=variable_counts,
            ipls_subset_limit=ipls_subset_limit, spa_ok=spa_ok,
            cache=varsel_cache,
            preprocess_id=_preprocess_fingerprint(pc),
            apply_uve_prefilter=apply_uve_prefilter,
            uve_cutoff_multiplier=uve_cutoff_multiplier,
            uve_n_components=uve_n_components,
        )
        for s in skipped:
            if s not in skipped_all:
                skipped_all.append(s)

        clamped = min(min_fold_train - 1, max_n_components)
        model_grids = get_model_grids(
            task_type="regression", n_features=X_pp.shape[1], tier=tier,
            enabled_models=list(model_names), max_n_components=max(1, clamped),
            max_iter=max_iter, **(model_grid_overrides or {}),
        )
        model_cfgs = _dedup_model_configs(model_grids)

        for s in subsets:
            if s.get("model_specific"):
                # 'importance' resolved per model below.
                for mc in model_cfgs:
                    # First-pass resilience: one model's importance-fit failure
                    # must NOT abort the search; record method+model and continue.
                    try:
                        imp = _importance_reference_fit(
                            mc["model_name"], X_pp, Y_arr, min_fold_train
                        )
                    except Exception:
                        notice = f"importance:{mc['model_name']}"
                        if notice not in skipped_all:
                            skipped_all.append(notice)
                        continue
                    for sub in _importances_to_subsets(
                        imp, "importance", variable_counts=variable_counts,
                        n_features_sub=X_pp.shape[1],
                    ):
                        X_ss = X_pp[:, sub["indices"]]
                        for cmc in _cap_and_dedup_pls_for_subset(
                            [mc], X_ss.shape[1], min_fold_train
                        ):
                            _emit_cells(pc, X_ss, sub["tag"], "importance", cmc)
                continue
            idx = s["indices"]
            X_sub = X_pp[:, idx]
            for mc in _cap_and_dedup_pls_for_subset(
                model_cfgs, X_sub.shape[1], min_fold_train
            ):
                _emit_cells(pc, X_sub, s["tag"], s["method"], mc)

    total = len(cells)
    for i, (pc, X_sub, tag, method, mc, mode) in enumerate(cells):
        if controller is not None and not controller.check_and_wait():
            break
        res = _evaluate_multitarget_cell(
            X_sub, Y_arr, mc["model_name"], mc["params"], splitter, min_fold_train,
            X_sub.shape[1], target_names, n_folds=n_folds, n_repeats=n_repeats,
            random_state=random_state, preprocessing=_describe_preprocess_config(pc),
            varsel_method=method, varsel_tag=tag, mode=mode,
        )
        results.append(res)
        if np.isfinite(res.joint_q2) and (best_finite is None or res.joint_q2 > best_finite.joint_q2):
            best_finite = res
        if progress_callback is not None:
            bm = None
            if best_finite is not None:
                bm = {
                    "Model": best_finite.model_name, "Preprocess": best_finite.preprocessing,
                    "Deriv": None, "RMSEcv": float(np.mean(best_finite.metrics.get("rmse", [np.nan]))),
                    "R2cv": float(best_finite.joint_q2),
                    "top_vars": str(best_finite.n_variables),
                }
            progress_callback({
                "message": f"Multi-target cell {i + 1}/{total}",
                "current": i + 1, "total": total, "best_model": bm,
            })

    results.sort(
        key=lambda r: (np.isfinite(r.joint_q2),
                       r.joint_q2 if np.isfinite(r.joint_q2) else float("-inf")),
        reverse=True,
    )
    return MultiTargetSearchOutput(
        results=results, target_names=target_names, correlation=correlation,
        n_targets=Y_arr.shape[1], skipped=skipped_all,
    )


@dataclass
class RefitMultiTargetModel:
    """A fully-fitted multi-target model reconstructed from a search result.

    Holds everything :mod:`spectral_predict.model_io` needs to persist a
    reload-and-predict ``.dasp`` for a multi-target leaderboard row. Produced by
    :func:`refit_multitarget_final`.

    Attributes:
        estimator: The fitted estimator (native 2-D JOINT model, native 2-D
            Ridge, or a fitted ``MultiOutputRegressor`` for INDEPENDENT models).
            For a JOINT model it was fit on Y-scaled targets, so its raw
            ``predict`` output is in scaled units — use :meth:`predict` (or the
            persisted ``y_scaler``) to recover RAW target units.
        preprocessor: The fitted preprocessing ``Pipeline`` (or ``None`` for a
            raw-spectra model). Persisted so a reloaded model reproduces the
            preprocessed block from raw spectra.
        y_scaler: The full-calibration :class:`~spectral_predict.multi_y.FoldYScaler`
            for a JOINT model, or ``None`` for INDEPENDENT / single-target.
        variable_indices: Column indices (into the preprocessed block) selected
            by variable selection for this cell; ``arange`` when full-spectrum.
        X_final: The exact preprocessed + subset block the estimator was fit on,
            shape ``(n_samples, n_selected)``.
        mode: ``"JOINT"`` or ``"INDEPENDENT"`` (the honest coupling label).
        model_name: Canonical model name.
        params: The (post-cap) hyperparameters used.
        target_names: Per-target labels, in prediction order.
        preprocessing: The compact preprocessing label
            (``_describe_preprocess_config`` output).
        varsel_method / varsel_tag: The variable-selection method + subset tag.
        subset_wavelengths: Wavelength values of the selected columns
            (``wl_pp[variable_indices]``) — the predict-time required set.
        full_wavelengths: The full (raw) wavelength axis — the predict-time
            full-spectrum reference used to re-subset after preprocessing.
        per_target_metrics: The result's per-target RAW-unit metrics, in order.
    """

    estimator: Any
    preprocessor: Any
    y_scaler: Optional[Any]
    variable_indices: np.ndarray
    X_final: np.ndarray
    mode: str
    model_name: str
    params: dict
    target_names: list[str]
    preprocessing: str
    varsel_method: str
    varsel_tag: str
    subset_wavelengths: list[float]
    full_wavelengths: list[float]
    per_target_metrics: list[dict] = field(default_factory=list)

    def predict(self, X_block: Any) -> np.ndarray:
        """Predict RAW target units from an already-preprocessed+subset block.

        ``X_block`` must be the preprocessed, variable-selected block (i.e. the
        same shape/order as :attr:`X_final`). For a JOINT model the persisted
        Y-scaler is applied so the returned ``(n, n_targets)`` array is in RAW
        target units; INDEPENDENT models already predict raw units.
        """
        pred = np.asarray(self.estimator.predict(X_block), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if self.y_scaler is not None:
            pred = self.y_scaler.inverse_transform(pred)
        return pred

    def build_metadata(self, *, performance: Optional[dict] = None) -> dict:
        """Assemble the :func:`spectral_predict.model_io.save_model` metadata dict.

        Uses full-spectrum-preprocessing bookkeeping (``full_wavelengths`` +
        subset ``wavelengths``) so a reloaded model re-applies preprocessing to
        raw spectra then re-selects the chosen columns by wavelength value —
        the same convention the single-Y save path uses for preprocessed+subset
        models.
        """
        if performance is None:
            q2s = [m.get("q2") for m in self.per_target_metrics if m.get("q2") is not None]
            performance = {"joint_q2": float(np.mean(q2s))} if q2s else {}
        return {
            "model_name": self.model_name,
            "task_type": "regression",
            "preprocessing": self.preprocessing,
            "wavelengths": list(self.subset_wavelengths),
            "n_vars": len(self.subset_wavelengths),
            "n_samples": int(self.X_final.shape[0]),
            "performance": performance,
            "params": dict(self.params),
            "multitarget_mode": self.mode,
            "n_targets": len(self.target_names),
            "target_names": list(self.target_names),
            "prediction_columns": [f"{t}_pred" for t in self.target_names],
            "per_target_metrics": list(self.per_target_metrics),
            "use_full_spectrum_preprocessing": True,
            "full_wavelengths": list(self.full_wavelengths),
        }

    def save(self, filepath, *, performance: Optional[dict] = None) -> None:
        """Persist this refit as a ``.dasp`` via :func:`model_io.save_model`.

        Passes the JOINT ``y_scaler`` through so reloaded predictions come back
        in RAW target units. Single-Y save/load is untouched (this only ever
        writes multi-target metadata keys).
        """
        from .model_io import save_model

        save_model(
            model=self.estimator,
            preprocessor=self.preprocessor,
            metadata=self.build_metadata(performance=performance),
            filepath=filepath,
            y_scaler=self.y_scaler,
        )


def refit_multitarget_final(
    result: Any,
    X: Any,
    Y: Any,
    target_names: Sequence[str],
    *,
    wavelengths: Any,
    model_names: Optional[Sequence[str]] = None,
    preprocessing_methods: dict,
    autoscale: bool = False,
    baseline_method=None,
    baseline_params=None,
    smoothing: bool = False,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    interference_to_add: Any = None,
    wavelength_restriction=None,
    variable_selection_methods: Optional[Sequence[str]] = None,
    variable_counts=None,
    apply_uve_prefilter: bool = False,
    uve_cutoff_multiplier: float = 1.0,
    uve_n_components: Optional[int] = None,
    ipls_subset_limit: str = "Top 10",
    window_sizes=None,
    cv: Any = "kfold",
    n_folds: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    **_ignored,
) -> RefitMultiTargetModel:
    """Reconstruct a fully-fitted multi-target model from a ``MultiTargetResult``.

    A search :class:`~spectral_predict.multitarget_search.MultiTargetResult`
    retains the metadata needed to *identify* a cell (``preprocessing`` label,
    ``varsel_method`` / ``varsel_tag``, ``params``, ``mode``) but NOT the fitted
    estimator or the selected variable indices. This helper REPLAYS the exact
    deterministic cell pipeline on the FULL calibration set to reproduce the
    cell, then fits the final model:

    1. **Preprocessing** — rebuild the preprocess-config grid from the run
       config, find the config whose compact label matches
       ``result.preprocessing``, and apply it via the shared
       :func:`_apply_preprocess_and_restrict` (same code the search runs), which
       also yields the fitted preprocessing ``Pipeline`` to persist.
    2. **Variable selection** — this project selects variables on the full
       calibration set (not per fold), so re-running the exact varsel method
       (via the shared :func:`build_multitarget_varsel_subsets` /
       :func:`_importance_reference_fit` primitives) is deterministic and
       reproduces the cell's column subset. The subset whose ``method``/``tag``
       matches the result is selected.
    3. **Fitting** — resolve the identical strategy
       (``resolve_multitarget_strategy(model_name, mode)``), build the estimator
       with :func:`build_multitarget_estimator`, and fit exactly as
       :func:`~spectral_predict.multi_y.multi_y_cv_pool` does: JOINT models fit
       on Y scaled by a full-calibration ``FoldYScaler`` (persisted);
       INDEPENDENT models fit on RAW per-target Y (no scaler). No fitting logic
       is forked — the same builders/strategy resolver are reused.

    Args:
        result: The :class:`MultiTargetResult` leaderboard row to refit.
        X: Raw feature matrix ``(n_samples, n_features)`` (pre-preprocessing).
        Y: Target block ``(n_samples,)`` or ``(n_samples, n_targets)``.
        target_names: Per-target labels, in order.
        wavelengths: Full raw wavelength axis (length ``n_features``).
        model_names: Enabled models for the run (needed to reproduce GA-gated
            varsel routing); defaults to ``[result.model_name]``.
        The remaining keyword args mirror :func:`run_multitarget_grid_search`
        and MUST match the values the search was run with so reconstruction is
        exact. Unrecognized kwargs (e.g. ``coupling_mode``, ``optimization_method``)
        are ignored so a full run-config dict can be splatted in.

    Returns:
        A :class:`RefitMultiTargetModel` holding the fitted estimator, the
        variable indices, the final preprocessed+subset block, the JOINT
        ``y_scaler`` (or ``None``), and the metadata needed to save/predict.

    Raises:
        ValueError: If the result's preprocessing label matches no config, if
            the varsel subset cannot be reconstructed, or if the reconstructed
            strategy's coupling mode/scaling disagrees with the stored result
            (a loud drift guard).
    """
    from .cv_utils import build_cv_splitter
    from .multi_y import FoldYScaler
    from .multitarget_search import (
        build_multitarget_estimator,
        resolve_multitarget_strategy,
    )

    if getattr(result, "error", None):
        raise ValueError(
            f"Cannot refit a failed search result (error={result.error!r}); "
            "it has no reproducible model."
        )

    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    wl = np.asarray(wavelengths, dtype=float)
    target_names = list(target_names)
    enabled_models = list(model_names) if model_names else [result.model_name]

    # --- 1. Rebuild the preprocess config that produced result.preprocessing ---
    preprocess_configs = build_multitarget_preprocess_configs(
        preprocessing_methods, window_sizes=window_sizes, autoscale=autoscale,
        baseline_method=baseline_method, baseline_params=baseline_params,
        smoothing=smoothing, smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder, interference_to_add=interference_to_add,
    )
    matches = [
        pc for pc in preprocess_configs
        if _describe_preprocess_config(pc) == result.preprocessing
    ]
    if not matches:
        raise ValueError(
            f"No preprocessing config reproduces label {result.preprocessing!r}; "
            "the run config passed to refit_multitarget_final does not match the "
            "search that produced this result."
        )
    pc = matches[0]

    wl_min = wl_max = wl_regions = None
    if wavelength_restriction:
        wl_min = wavelength_restriction.get("min")
        wl_max = wavelength_restriction.get("max")
        wl_regions = wavelength_restriction.get("regions")
    restriction_active = bool(wl_regions or wl_min is not None or wl_max is not None)

    X_pp, wl_pp, preprocessor = _apply_preprocess_and_restrict(
        pc, X_arr, Y_arr, wl,
        wl_min=wl_min, wl_max=wl_max, wl_regions=wl_regions,
        restriction_active=restriction_active,
    )

    # --- 2. Reconstruct the variable-selection subset (deterministic) ---
    splitter = build_cv_splitter(
        cv, n_folds, "regression", n_repeats=n_repeats, random_state=random_state, y=None,
    ) if isinstance(cv, str) else cv
    min_fold_train = min(len(tr) for tr, _ in splitter.split(X_arr, Y_arr))

    requested_methods = set(variable_selection_methods or [])
    spa_ok = (
        verify_spa_multi_y_safe(X_arr, Y_arr)
        if requested_methods & _SPA_DEPENDENT_METHODS
        else True
    )

    idx = None
    if result.varsel_method == "importance":
        # Model-specific importances (not emitted by build_multitarget_varsel_subsets).
        imp = _importance_reference_fit(result.model_name, X_pp, Y_arr, min_fold_train)
        for sub in _importances_to_subsets(
            imp, "importance", variable_counts=variable_counts,
            n_features_sub=X_pp.shape[1],
        ):
            if sub["tag"] == result.varsel_tag:
                idx = np.asarray(sub["indices"])
                break
    else:
        subsets, _skipped = build_multitarget_varsel_subsets(
            variable_selection_methods, X_pp, Y_arr, wl_pp,
            enabled_models=enabled_models, variable_counts=variable_counts,
            ipls_subset_limit=ipls_subset_limit, spa_ok=spa_ok, cache={},
            preprocess_id=_preprocess_fingerprint(pc),
            apply_uve_prefilter=apply_uve_prefilter,
            uve_cutoff_multiplier=uve_cutoff_multiplier, uve_n_components=uve_n_components,
        )
        for s in subsets:
            if s.get("model_specific"):
                continue
            if s.get("method") == result.varsel_method and s.get("tag") == result.varsel_tag:
                idx = np.asarray(s["indices"])
                break
    if idx is None:
        raise ValueError(
            f"Could not reconstruct varsel subset (method={result.varsel_method!r}, "
            f"tag={result.varsel_tag!r}) for preprocessing {result.preprocessing!r}. "
            "The run config passed to refit_multitarget_final must match the search."
        )

    X_sub = X_pp[:, idx]

    # --- 3. Fit the final model (identical strategy/scaling as the cell) ---
    strategy = resolve_multitarget_strategy(result.model_name, mode=result.mode.lower())
    if strategy.mode != result.mode or strategy.scale_y != result.scale_y:
        raise ValueError(
            "Reconstructed strategy drifted from the stored result "
            f"(mode {strategy.mode!r} vs {result.mode!r}, scale_y "
            f"{strategy.scale_y} vs {result.scale_y}); refusing to save a model "
            "whose coupling/scaling would not reproduce the search."
        )
    estimator = build_multitarget_estimator(
        strategy, result.params, X_sub.shape[0], X_sub.shape[1]
    )
    if strategy.scale_y:
        y_scaler = FoldYScaler().fit(Y_arr)
        estimator.fit(X_sub, y_scaler.transform(Y_arr))
    else:
        y_scaler = None
        estimator.fit(X_sub, Y_arr)

    return RefitMultiTargetModel(
        estimator=estimator,
        preprocessor=preprocessor,
        y_scaler=y_scaler,
        variable_indices=np.asarray(idx),
        X_final=X_sub,
        mode=result.mode,
        model_name=result.model_name,
        params=dict(result.params),
        target_names=target_names,
        preprocessing=result.preprocessing,
        varsel_method=result.varsel_method,
        varsel_tag=result.varsel_tag,
        subset_wavelengths=[float(v) for v in np.asarray(wl_pp)[idx]],
        full_wavelengths=[float(v) for v in wl],
        per_target_metrics=list(result.metrics.get("per_target", [])),
    )
