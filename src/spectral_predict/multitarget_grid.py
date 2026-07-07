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
    from sklearn.pipeline import Pipeline

    from .cv_utils import build_cv_splitter
    from .models import get_model_grids
    from .multi_y import inter_target_correlation
    from .multitarget_search import MultiTargetSearchOutput, _evaluate_multitarget_cell
    from .preprocess import build_preprocessing_pipeline
    from .search import _apply_edge_mask_to_data

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
        steps = build_preprocessing_pipeline(
            name, pc["deriv"], pc["window"], pc["polyorder"],
            task_type="regression", interference=pc.get("interference"), wavelengths=wl,
            baseline_method=pc.get("baseline_method"), baseline_params=pc.get("baseline_params"),
            smoothing=pc.get("smoothing", False), smoothing_window=pc.get("smoothing_window", 17),
            smoothing_polyorder=pc.get("smoothing_polyorder", 2), autoscale=pc.get("autoscale", False),
        )
        X_pp = X_arr.copy()
        if steps:
            X_pp = Pipeline(steps).fit_transform(X_pp, Y_arr)
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
