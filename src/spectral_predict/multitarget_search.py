"""Model-agnostic multi-target (multi-Y) search orchestrator for T-17.

This is the Grid-engine-only orchestrator for the multi-target (``n_targets >= 1``)
regression path. It is a **superset** of the legacy single-Y search: single-target
is the one-column case, proven by the consolidation pin (per-target Q2 at
``n_targets == 1`` equals the legacy ``r2_score(all_y_test, all_y_pred)``).

Design invariants (from the T-17 plan):

* **Grid engine ONLY.** Bayesian (``unified``) and NSGA-II are 1-D-only engines
  and are deliberately excluded; :func:`run_multitarget_search` refuses any
  ``optimization_method`` other than ``'grid'`` so a stale radio value cannot
  silently route 2-D Y into a 1-D engine.
* **Honest JOINT/INDEPENDENT labeling.** :func:`resolve_multitarget_strategy`
  returns the exact coupling mode for every model. JOINT models genuinely
  couple targets (shared latent variables / impurity / hidden layers / joint
  loss); INDEPENDENT models are separate per-target estimators under one shared
  searched configuration and carry :data:`INDEPENDENT_PRECISE_NOTE` verbatim so
  batched breadth is never mistaken for coupling. An UNKNOWN model name RAISES
  (fails loud, never silently mislabels coupling).
* **Fold Y-scaling is JOINT-only.** JOINT strategies fit on fold-scaled Y
  (``scale_y=True``); INDEPENDENT strategies fit on RAW per-target Y
  (``scale_y=False``) to preserve the "separate per-target models" guarantee.
* **Per-target metrics on RAW units** via :mod:`spectral_predict.multi_y`.

JOINT models (F3): PLS-2 (flagship), RandomForest (native multi-output
impurity), MLP (shared hidden layers), CatBoost (``loss_function='MultiRMSE'``),
and XGBoost (``multi_strategy='multi_output_tree'``) -- each fitting the
fold-scaled Y block natively and predicting ``(n, n_targets)``. Booster
early-stopping is DISABLED under multi-Y for v1 (no ``eval_set`` / ``od_*`` /
``early_stopping_rounds``). The optional JOINT MultiTask linear models
(``MultiTaskLasso`` / ``MultiTaskElasticNet``, shared L21 row-support) also fit
on fold-scaled Y.

INDEPENDENT models (F4): Ridge (native per-column 2-D solve) plus SVR, LightGBM,
plain Lasso/ElasticNet and NeuralBoosted (wrapped in
:class:`sklearn.multioutput.MultiOutputRegressor`). These fit on RAW per-target
Y -- at a FIXED config an INDEPENDENT model is bit-identical to N separate
single-target fits, so fold Y-scaling is NEVER applied (scale-sensitive
``alpha`` / ``epsilon`` would break that guarantee). They carry
:data:`INDEPENDENT_PRECISE_NOTE` so batched breadth is never mistaken for
coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .cv_utils import build_cv_splitter
from .multi_y import (
    cap_components,
    inter_target_correlation,
    multi_y_cv_pool,
    multi_y_metrics,
)

__all__ = [
    "INDEPENDENT_PRECISE_NOTE",
    "MultiTargetStrategy",
    "MultiTargetResult",
    "MultiTargetSearchOutput",
    "resolve_multitarget_strategy",
    "build_multitarget_estimator",
    "_evaluate_multitarget_cell",
    "run_multitarget_search",
]


# The precise honest-labeling note carried by every INDEPENDENT result. Pinned
# verbatim by tests so it can never drift into overstating coupling. (Codex: the
# looser "== separate models" wording overstates equivalence under a
# shared-config search.)
INDEPENDENT_PRECISE_NOTE = (
    "separate per-target estimators under one shared searched configuration "
    "— not a coupled model, no correlation benefit; for exact equivalence "
    "to N independent searches, run separate single-target searches."
)

_JOINT = "JOINT"
_INDEPENDENT = "INDEPENDENT"


@dataclass(frozen=True)
class MultiTargetStrategy:
    """How a model couples (or does not couple) multiple targets.

    Attributes:
        model_name: Canonical regression model name.
        mode: ``"JOINT"`` (genuinely couples targets) or ``"INDEPENDENT"``
            (separate per-target estimators under one shared config).
        mechanism: Human-readable description of the coupling mechanism.
        scale_y: True for JOINT (fold Y-scaler used for fitting); False for
            INDEPENDENT (fit on RAW per-target Y).
        native: True if the estimator natively accepts a 2-D ``Y`` block; False
            if it must be wrapped in ``MultiOutputRegressor``.
        precise_note: Empty for JOINT; :data:`INDEPENDENT_PRECISE_NOTE` for
            INDEPENDENT.
        joint_params: Extra estimator kwargs required to activate native joint
            multi-output (e.g. CatBoost ``loss_function='MultiRMSE'``,
            XGBoost ``multi_strategy='multi_output_tree'``).
        supports_optional_joint: True for Lasso/ElasticNet, which default to
            INDEPENDENT but offer an optional JOINT ``MultiTask`` variant.
    """

    model_name: str
    mode: str
    mechanism: str
    scale_y: bool
    native: bool
    precise_note: str
    joint_params: dict[str, Any] = field(default_factory=dict)
    supports_optional_joint: bool = False


def _joint(model_name: str, mechanism: str, *, native: bool = True,
           joint_params: Optional[dict[str, Any]] = None) -> MultiTargetStrategy:
    return MultiTargetStrategy(
        model_name=model_name,
        mode=_JOINT,
        mechanism=mechanism,
        scale_y=True,
        native=native,
        precise_note="",
        joint_params=dict(joint_params or {}),
    )


def _independent(model_name: str, mechanism: str, *, native: bool,
                 supports_optional_joint: bool = False) -> MultiTargetStrategy:
    return MultiTargetStrategy(
        model_name=model_name,
        mode=_INDEPENDENT,
        mechanism=mechanism,
        scale_y=False,
        native=native,
        precise_note=INDEPENDENT_PRECISE_NOTE,
        supports_optional_joint=supports_optional_joint,
    )


# Per-model multi-output strategy table (the core of the T-17 scope). Keyed by
# canonical regression model name. JOINT models genuinely couple targets;
# INDEPENDENT models are per-target-separable (batched).
_STRATEGY_TABLE: dict[str, MultiTargetStrategy] = {
    # --- JOINT (genuine coupling) ---
    "PLS": _joint("PLS", "shared latent variables (sklearn native 2-D Y)"),
    "RandomForest": _joint(
        "RandomForest", "impurity averaged across targets (sklearn native)"
    ),
    "MLP": _joint("MLP", "shared hidden layers (sklearn native)"),
    "CatBoost": _joint(
        "CatBoost",
        "loss_function='MultiRMSE'",
        joint_params={"loss_function": "MultiRMSE"},
    ),
    "XGBoost": _joint(
        "XGBoost",
        "multi_strategy='multi_output_tree'",
        joint_params={"multi_strategy": "multi_output_tree"},
    ),
    # --- INDEPENDENT (per-target-separable, batched) ---
    "Ridge": _independent(
        "Ridge", "per-target single 2-D solve (sklearn native)", native=True
    ),
    "Lasso": _independent(
        "Lasso",
        "MultiOutputRegressor (optional JOINT MultiTaskLasso)",
        native=False,
        supports_optional_joint=True,
    ),
    "ElasticNet": _independent(
        "ElasticNet",
        "MultiOutputRegressor (optional JOINT MultiTaskElasticNet)",
        native=False,
        supports_optional_joint=True,
    ),
    "SVR": _independent("SVR", "MultiOutputRegressor", native=False),
    "LightGBM": _independent(
        "LightGBM", "MultiOutputRegressor (no native multi-output)", native=False
    ),
    "NeuralBoosted": _independent(
        "NeuralBoosted", "MultiOutputRegressor (1-D only estimator)", native=False
    ),
    # --- Optional JOINT variants of the sparse linear models ---
    # MultiTaskLasso/ElasticNet impose a shared L21 row-support across targets:
    # their Frobenius fit + joint sparsity are non-separable, so they are JOINT
    # and fit on fold-scaled Y (an unscaled high-variance target would dominate
    # the shared sparsity pattern). Reached via ``supports_optional_joint`` on
    # plain Lasso/ElasticNet.
    "MultiTaskLasso": _joint(
        "MultiTaskLasso", "shared L21 row-support (joint sparsity)"
    ),
    "MultiTaskElasticNet": _joint(
        "MultiTaskElasticNet", "shared L21 row-support + L2 (joint sparsity)"
    ),
}

# Aliases accepted for convenience; SVM is the classification-side name for SVR.
_MODEL_ALIASES = {"SVM": "SVR"}


def resolve_multitarget_strategy(model_name: str) -> MultiTargetStrategy:
    """Resolve a model's multi-target coupling strategy.

    Pure function. Returns the JOINT/INDEPENDENT mode, the fold-Y-scaling rule,
    the build mechanism, and (for INDEPENDENT) the exact honest-labeling note.

    Args:
        model_name: Canonical regression model name (e.g. ``"PLS"``,
            ``"Ridge"``, ``"LightGBM"``). ``"SVM"`` is accepted as an alias for
            ``"SVR"``.

    Returns:
        The :class:`MultiTargetStrategy` for the model.

    Raises:
        ValueError: If ``model_name`` is unknown. Failing loud here is a
            statistical-integrity guardrail: a silently-defaulted mode could
            mislabel a batched model as coupling.
    """
    if not isinstance(model_name, str):
        raise ValueError(f"model_name must be a str, got {type(model_name).__name__}.")
    key = _MODEL_ALIASES.get(model_name, model_name)
    strategy = _STRATEGY_TABLE.get(key)
    if strategy is None:
        raise ValueError(
            f"Unknown multi-target model {model_name!r}. Known models: "
            f"{sorted(_STRATEGY_TABLE)}. Refusing to guess a coupling mode "
            "(would risk mislabeling batched breadth as coupling)."
        )
    return strategy


def build_multitarget_estimator(
    strategy: MultiTargetStrategy,
    params: Optional[dict[str, Any]],
    n_samples: int,
    n_features: int,
):
    """Build an unfitted estimator for a resolved multi-target strategy.

    F3 wires all **JOINT** models: PLS, RandomForest, MLP, CatBoost
    (``loss_function='MultiRMSE'``), and XGBoost
    (``multi_strategy='multi_output_tree'``). Each natively accepts the 2-D
    fold-scaled ``Y`` block and predicts ``(n, n_targets)``. Booster
    early-stopping is disabled under multi-Y for v1 (the builder sets no
    ``od_*`` / ``early_stopping_rounds`` and :func:`multi_y_cv_pool` passes no
    ``eval_set``). PLS latent components are capped at
    ``A <= min(n_samples - 1, n_features)`` via
    :func:`spectral_predict.multi_y.cap_components`; ``scale=False`` matches the
    legacy single-Y ``get_model("PLS")`` builder so single-target parity holds.

    F4 wires the INDEPENDENT models: Ridge is native (one 2-D solve, per-column);
    SVR, LightGBM, plain Lasso/ElasticNet and NeuralBoosted are wrapped in
    :class:`sklearn.multioutput.MultiOutputRegressor` (one cloned base estimator
    per target, fit on that target's RAW column). The optional JOINT
    ``MultiTaskLasso`` / ``MultiTaskElasticNet`` variants are native 2-D
    estimators fit on fold-scaled Y. INDEPENDENT builders' defaults mirror the
    legacy ``build_model_from_params`` regression builders.

    Args:
        strategy: A resolved :class:`MultiTargetStrategy`.
        params: Hyperparameters for the config (e.g. ``{"n_components": 8}``).
        n_samples: Sample count used for component capping. Callers inside a CV
            search MUST pass the **minimum fold training size**, not the full
            sample count: the estimator is cloned inside each (smaller) train
            fold, so a component count valid for the full N can exceed a fold's
            upper bound and raise a sklearn ``ValueError`` inside CV.
        n_features: Feature count (for component capping).

    Returns:
        An unfitted sklearn-compatible estimator that accepts a 2-D ``Y`` block.

    Raises:
        NotImplementedError: For a resolved strategy whose model name has no
            builder wired (should not happen for any table entry).
    """
    params = dict(params or {})
    name = strategy.model_name

    if name == "PLS":
        from sklearn.cross_decomposition import PLSRegression

        requested = int(params.get("n_components", 10))
        n_components = cap_components(n_samples, n_features, requested)
        pls_kwargs: dict[str, Any] = dict(n_components=n_components, scale=False)
        if "max_iter" in params:
            pls_kwargs["max_iter"] = int(params["max_iter"])
        if "tol" in params:
            pls_kwargs["tol"] = float(params["tol"])
        return PLSRegression(**pls_kwargs)

    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", 1.0),
            bootstrap=bool(params.get("bootstrap", True)),
            max_leaf_nodes=params.get("max_leaf_nodes", None),
            min_impurity_decrease=float(params.get("min_impurity_decrease", 0.0)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
        )

    if name == "MLP":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64,)),
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            alpha=float(params.get("alpha", 1e-3)),
            batch_size=params.get("batch_size", "auto"),
            learning_rate=params.get("learning_rate", "constant"),
            learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
            momentum=float(params.get("momentum", 0.9)),
            max_iter=int(params.get("max_iter", 500)),
            early_stopping=bool(params.get("early_stopping", True)),
            random_state=int(params.get("random_state", 42)),
        )

    if name == "CatBoost":
        from catboost import CatBoostRegressor

        kwargs: dict[str, Any] = dict(
            iterations=int(params.get("iterations", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            depth=int(params.get("depth", 5)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 4.0)),
            min_data_in_leaf=int(params.get("min_data_in_leaf", 1)),
            random_state=int(params.get("random_state", 42)),
            verbose=False,
        )
        if "border_count" in params:
            kwargs["border_count"] = int(params["border_count"])
        if "random_strength" in params:
            kwargs["random_strength"] = float(params["random_strength"])
        if "bootstrap_type" in params:
            kwargs["bootstrap_type"] = params["bootstrap_type"]
        if "bagging_temperature" in params and params.get("bootstrap_type", "Bayesian") == "Bayesian":
            kwargs["bagging_temperature"] = float(params["bagging_temperature"])
        kwargs.update(strategy.joint_params)  # loss_function='MultiRMSE'
        return CatBoostRegressor(**kwargs)

    if name == "XGBoost":
        # JOINT via multi_strategy='multi_output_tree' (from strategy.joint_params,
        # xgboost >= 2.0): each tree grows vector-valued leaves so targets share
        # split structure. Booster early-stopping is DISABLED for multi-Y v1 -- no
        # early_stopping_rounds is set and multi_y_cv_pool passes no eval_set.
        from xgboost import XGBRegressor

        kwargs = dict(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 6)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.6)),
            reg_alpha=float(params.get("reg_alpha", 0.2)),
            reg_lambda=float(params.get("reg_lambda", 1.5)),
            min_child_weight=float(params.get("min_child_weight", 1)),
            gamma=float(params.get("gamma", 0.0)),
            tree_method="hist",
            random_state=int(params.get("random_state", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
            verbosity=0,
        )
        kwargs.update(strategy.joint_params)  # multi_strategy='multi_output_tree'
        return XGBRegressor(**kwargs)

    # --- JOINT optional MultiTask linear models (fit on fold-scaled Y) ---
    # MultiTaskLasso/ElasticNet natively accept a 2-D Y block and impose a
    # shared L21 row-support across targets (genuine joint sparsity). They
    # require n_targets >= 2 (the single-Y path never routes here).
    if name == "MultiTaskLasso":
        from sklearn.linear_model import MultiTaskLasso

        p = dict(params)
        p.setdefault("alpha", 1.0)
        p.setdefault("max_iter", 500)
        p.setdefault("random_state", 42)
        return MultiTaskLasso(**p)

    if name == "MultiTaskElasticNet":
        from sklearn.linear_model import MultiTaskElasticNet

        p = dict(params)
        p.setdefault("alpha", 1.0)
        p.setdefault("l1_ratio", 0.5)
        p.setdefault("max_iter", 500)
        p.setdefault("random_state", 42)
        return MultiTaskElasticNet(**p)

    # --- INDEPENDENT models (per-target-separable) fit on RAW per-target Y ---
    # Ridge is native (sklearn solves the 2-D Y system per column in one call).
    # The rest have no native multi-output, so they are wrapped in
    # MultiOutputRegressor, which clones the base estimator once per target and
    # fits each on that target's RAW column -- bit-identical to N separate
    # single-target fits at a FIXED config. Defaults mirror the legacy
    # ``build_model_from_params`` regression builders so single-target parity is
    # preserved. No fold Y-scaling is applied to these models (scale-sensitive
    # ``alpha`` / ``epsilon`` at fixed hyperparameters would otherwise break the
    # "separate per-target models" guarantee).
    if name == "Ridge":
        from sklearn.linear_model import Ridge

        p = dict(params)
        p.setdefault("random_state", 42)
        return Ridge(**p)

    base = _build_independent_base(name, params)
    if base is not None:
        from sklearn.multioutput import MultiOutputRegressor

        return MultiOutputRegressor(base)

    raise NotImplementedError(
        f"Multi-target estimator for {strategy.model_name!r} ({strategy.mode}) "
        "has no builder wired."
    )


def _build_independent_base(name: str, params: dict[str, Any]):
    """Build the per-target BASE estimator for an INDEPENDENT MOR-wrapped model.

    Returns an unfitted single-target estimator whose defaults mirror the legacy
    ``build_model_from_params`` regression builders, or ``None`` if ``name`` is
    not an MOR-wrapped INDEPENDENT model. The caller wraps the result in
    :class:`sklearn.multioutput.MultiOutputRegressor`.
    """
    p = dict(params)
    if name == "Lasso":
        from sklearn.linear_model import Lasso

        p.setdefault("max_iter", 500)
        p.setdefault("random_state", 42)
        return Lasso(**p)
    if name == "ElasticNet":
        from sklearn.linear_model import ElasticNet

        p.setdefault("l1_ratio", 0.5)
        p.setdefault("max_iter", 500)
        p.setdefault("random_state", 42)
        return ElasticNet(**p)
    if name == "SVR":
        from sklearn.svm import SVR

        return SVR(**p)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        p.setdefault("random_state", 42)
        p.setdefault("n_jobs", -1)
        p.setdefault("verbosity", -1)
        return LGBMRegressor(**p)
    if name == "NeuralBoosted":
        from .neural_boosted import NeuralBoostedRegressor

        p.setdefault("random_state", 42)
        p.setdefault("verbose", 0)
        return NeuralBoostedRegressor(**p)
    return None


@dataclass
class MultiTargetResult:
    """One (model, hyperparameter) config's cross-validated multi-target result.

    Attributes:
        model_name: Canonical model name.
        mode: ``"JOINT"`` or ``"INDEPENDENT"`` (the honest coupling label).
        params: The hyperparameters used.
        joint_q2: Joint selection criterion = mean per-target raw-unit Q2.
        metrics: Full :func:`spectral_predict.multi_y.multi_y_metrics` dict
            (per-target R2/RMSE/Q2/RPD/RER/CCC/Bias in raw units).
        precise_note: Empty for JOINT; the INDEPENDENT honest-labeling note.
        scale_y: Whether fold Y-scaling was used (JOINT only).
        mechanism: Human-readable coupling mechanism description.
        y_true_pooled: Pooled RAW-unit truths ``(n_tested, n_targets)`` used for
            scoring (feeds per-target plots/export). At ``n_targets == 1`` this
            is byte-identical to the legacy raw-Y path.
        y_pred_pooled: Pooled RAW-unit predictions ``(n_tested, n_targets)``,
            aligned row-for-row with ``y_true_pooled``. At ``n_targets == 1``
            this is ``np.array_equal`` to legacy ``cross_val_predict_pooled``.
    """

    model_name: str
    mode: str
    params: dict[str, Any]
    joint_q2: float
    metrics: dict[str, Any]
    precise_note: str
    scale_y: bool
    mechanism: str
    y_true_pooled: Optional[np.ndarray] = None
    y_pred_pooled: Optional[np.ndarray] = None
    preprocessing: str = "raw"
    varsel_method: str = "full"
    varsel_tag: str = "full"
    n_variables: Optional[int] = None
    error: Optional[str] = None


@dataclass
class MultiTargetSearchOutput:
    """Result of a multi-target grid search.

    Attributes:
        results: Config results, ranked by ``joint_q2`` descending.
        target_names: Target column labels, in order.
        correlation: :func:`spectral_predict.multi_y.inter_target_correlation`
            guardrail dict (``corr_matrix``, ``mean_abs_corr``, ``is_weak``).
        n_targets: Number of targets.
    """

    results: list[MultiTargetResult]
    target_names: list[str]
    correlation: dict[str, Any]
    n_targets: int
    skipped: list[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[MultiTargetResult]:
        """The top-ranked result (highest joint Q2), or None if empty."""
        return self.results[0] if self.results else None


def _evaluate_multitarget_cell(
    X_sub: Any,
    Y: Any,
    model_name: str,
    params: dict[str, Any],
    splitter: Any,
    min_fold_train: int,
    n_features_sub: int,
    target_names: list[str],
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    preprocessing: str = "raw",
    varsel_method: str = "full",
    varsel_tag: str = "full",
) -> MultiTargetResult:
    """Evaluate one (preprocess, varsel-subset, model, hp) cell; NaN-sink on failure.

    ``n_features_sub`` MUST be ``X_sub.shape[1]`` (the subset feature count) so PLS
    component capping cannot over-request. Any exception (degenerate subset,
    non-finite fold, sklearn ValueError) returns ``joint_q2=np.nan`` + ``error`` —
    never a finite 0.0 — so the NaN-safe rank sinks it.
    """
    strategy: Optional[MultiTargetStrategy] = None
    try:
        strategy = resolve_multitarget_strategy(model_name)
        estimator = build_multitarget_estimator(
            strategy, params, min_fold_train, n_features_sub
        )
        y_true, y_pred = multi_y_cv_pool(
            estimator, X_sub, Y, splitter,
            scale_y=strategy.scale_y, n_folds=n_folds,
            n_repeats=n_repeats, random_state=random_state,
        )
        metrics = multi_y_metrics(y_true, y_pred, target_names=target_names)
        return MultiTargetResult(
            model_name=model_name, mode=strategy.mode, params=dict(params),
            joint_q2=metrics["joint_q2"], metrics=metrics,
            precise_note=strategy.precise_note, scale_y=strategy.scale_y,
            mechanism=strategy.mechanism, y_true_pooled=y_true, y_pred_pooled=y_pred,
            preprocessing=preprocessing, varsel_method=varsel_method,
            varsel_tag=varsel_tag, n_variables=int(n_features_sub), error=None,
        )
    except Exception as exc:  # NaN-sink: never a finite 0.0
        # strategy may be None if resolve_multitarget_strategy raised (e.g. an
        # UNKNOWN model name). Guarded fallbacks keep this cell's NaN sink
        # honest without aborting the whole search.
        return MultiTargetResult(
            model_name=model_name,
            mode=(strategy.mode if strategy is not None else "UNKNOWN"),
            params=dict(params),
            joint_q2=np.nan, metrics={},
            precise_note=(strategy.precise_note if strategy is not None else ""),
            scale_y=(strategy.scale_y if strategy is not None else False),
            mechanism=(strategy.mechanism if strategy is not None else ""),
            y_true_pooled=None, y_pred_pooled=None,
            preprocessing=preprocessing, varsel_method=varsel_method,
            varsel_tag=varsel_tag, n_variables=int(n_features_sub), error=str(exc),
        )


def run_multitarget_search(
    X: Any,
    Y: Any,
    model_configs: Sequence[dict[str, Any]],
    *,
    cv: Any = "kfold",
    n_folds: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    target_names: Optional[list[str]] = None,
    optimization_method: str = "grid",
    weak_corr_threshold: float = 0.35,
) -> MultiTargetSearchOutput:
    """Run a Grid-engine multi-target search over model/hyperparameter configs.

    Superset of the legacy single-Y search: ``Y`` may be 1-D (one target) or
    2-D ``(n_samples, n_targets)``. Each config is cross-validated with
    :func:`spectral_predict.multi_y.multi_y_cv_pool` (fold Y-scaling for JOINT,
    raw Y for INDEPENDENT), scored per-target on RAW units by
    :func:`spectral_predict.multi_y.multi_y_metrics`, and ranked by the joint-Q2
    criterion (mean per-target Q2). Results carry the honest JOINT/INDEPENDENT
    label + precise note.

    Args:
        X: Feature matrix ``(n_samples, n_features)`` (already preprocessed).
        Y: Target block ``(n_samples,)`` or ``(n_samples, n_targets)``.
        model_configs: Sequence of ``{"model_name": str, "params": dict}``.
        cv: CV splitter object or strategy string (passed to
            :func:`multi_y_cv_pool`).
        n_folds: Fold count when ``cv`` is a strategy string.
        n_repeats: Repeat count when ``cv`` is a repeated strategy string.
        random_state: RNG seed for the splitter when ``cv`` is a string.
        target_names: Per-target labels; defaults to ``target_0`` etc.
        optimization_method: MUST be ``'grid'``. Bayesian/NSGA-II are 1-D-only
            engines and are refused here.
        weak_corr_threshold: Mean-abs-correlation below which the correlation
            guardrail flags the target block as weakly correlated.

    Returns:
        A :class:`MultiTargetSearchOutput` with ranked results, target names,
        and the correlation guardrail.

    Raises:
        ValueError: If ``optimization_method`` is not ``'grid'``, or if
            ``model_configs`` is empty.
    """
    if optimization_method != "grid":
        raise ValueError(
            f"Multi-target search is Grid-engine ONLY; got "
            f"optimization_method={optimization_method!r}. Bayesian (unified) and "
            "NSGA-II are 1-D-only engines and are deliberately excluded — the "
            "multi-target dispatcher forces optimization_method='grid'."
        )
    if not model_configs:
        raise ValueError("run_multitarget_search requires at least one model config.")

    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    n_samples, n_features = X_arr.shape
    n_targets = Y_arr.shape[1]

    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]
    elif len(target_names) != n_targets:
        raise ValueError(
            f"target_names has {len(target_names)} entries but Y has "
            f"{n_targets} targets."
        )

    correlation = inter_target_correlation(Y_arr, weak_threshold=weak_corr_threshold)

    # Build the CV splitter ONCE so component capping sees the real fold train
    # sizes. Estimators (e.g. PLS n_components) must be capped against the
    # SMALLEST fold training set, not the full sample count: the estimator is
    # cloned inside every (smaller) train fold, and a component count valid for
    # the full N can exceed a fold's upper bound and raise inside CV.
    if isinstance(cv, str):
        splitter = build_cv_splitter(
            cv, n_folds, "regression", n_repeats=n_repeats,
            random_state=random_state, y=None,
        )
    else:
        splitter = cv
    min_fold_train = min(len(train_idx) for train_idx, _ in splitter.split(X_arr, Y_arr))

    results: list[MultiTargetResult] = []
    for config in model_configs:
        model_name = config["model_name"]
        params = config.get("params", {})
        results.append(
            _evaluate_multitarget_cell(
                X_arr, Y_arr, model_name, params, splitter, min_fold_train,
                X_arr.shape[1], target_names,
                n_folds=n_folds, n_repeats=n_repeats, random_state=random_state,
            )
        )

    # NaN-safe ranking: a model whose pooled predictions go non-finite (e.g. PLS
    # coef overflow on a collinear fold, MLP divergence) yields joint_q2 == NaN.
    # Plain sort on a NaN key is undefined per CPython docs and could float a
    # broken model to rank #1 / best. Push non-finite scores to the bottom.
    results.sort(
        key=lambda r: (
            np.isfinite(r.joint_q2),
            r.joint_q2 if np.isfinite(r.joint_q2) else float("-inf"),
        ),
        reverse=True,
    )
    return MultiTargetSearchOutput(
        results=results,
        target_names=list(target_names),
        correlation=correlation,
        n_targets=n_targets,
    )
