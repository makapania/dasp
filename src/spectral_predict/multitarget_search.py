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

F3 scope: all JOINT models are wired -- PLS-2 (flagship), RandomForest (native
multi-output impurity), MLP (shared hidden layers), CatBoost
(``loss_function='MultiRMSE'``), and XGBoost
(``multi_strategy='multi_output_tree'``) -- each fitting the fold-scaled Y block
natively and predicting ``(n, n_targets)``. Booster early-stopping is DISABLED
under multi-Y for v1 (no ``eval_set`` / ``od_*`` / ``early_stopping_rounds``).
The INDEPENDENT models (Ridge, Lasso/ElasticNet, SVR, LightGBM, NeuralBoosted)
are resolved by :func:`resolve_multitarget_strategy` but *built* in F4; calling
the builder for a not-yet-wired model raises :class:`NotImplementedError`.
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
    The INDEPENDENT models are built in F4.

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
        NotImplementedError: For models resolved but not yet wired in F2.
    """
    params = dict(params or {})
    name = strategy.model_name

    if name == "PLS":
        from sklearn.cross_decomposition import PLSRegression

        requested = int(params.get("n_components", 10))
        n_components = cap_components(n_samples, n_features, requested)
        return PLSRegression(n_components=n_components, scale=False)

    if name == "RandomForest":
        # sklearn native multi-output: impurity is averaged across targets, so a
        # single forest genuinely couples the fold-scaled target block. Defaults
        # mirror the legacy get_model("RandomForest") builder.
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            max_features=params.get("max_features", 1.0),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
        )

    if name == "MLP":
        # sklearn native multi-output: one network with shared hidden layers and
        # a multi-unit output head. early_stopping is an internal validation
        # split (NOT booster early-stopping) and is safe under multi-Y.
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64,)),
            alpha=float(params.get("alpha", 1e-3)),
            learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
            max_iter=int(params.get("max_iter", 500)),
            early_stopping=bool(params.get("early_stopping", True)),
            random_state=int(params.get("random_state", 42)),
        )

    if name == "CatBoost":
        # JOINT via loss_function='MultiRMSE' (from strategy.joint_params): one
        # ensemble optimizes the summed per-target RMSE on the fold-scaled block.
        # Booster early-stopping is DISABLED for multi-Y v1 -- we build with no
        # od_* params and multi_y_cv_pool never passes an eval_set, so fitting is
        # a plain full-iteration fit.
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

    raise NotImplementedError(
        f"Multi-target estimator for {strategy.model_name!r} ({strategy.mode}) "
        "is resolved but not yet wired. F3 implements the JOINT models (PLS, "
        "RandomForest, MLP, CatBoost, XGBoost); the INDEPENDENT models (Ridge, "
        "Lasso/ElasticNet, SVR, LightGBM, NeuralBoosted) are built in F4."
    )


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

    @property
    def best(self) -> Optional[MultiTargetResult]:
        """The top-ranked result (highest joint Q2), or None if empty."""
        return self.results[0] if self.results else None


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
        strategy = resolve_multitarget_strategy(model_name)
        estimator = build_multitarget_estimator(
            strategy, params, min_fold_train, n_features
        )
        y_true, y_pred = multi_y_cv_pool(
            estimator,
            X_arr,
            Y_arr,
            splitter,
            scale_y=strategy.scale_y,
            n_folds=n_folds,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        metrics = multi_y_metrics(y_true, y_pred, target_names=target_names)
        results.append(
            MultiTargetResult(
                model_name=model_name,
                mode=strategy.mode,
                params=dict(params),
                joint_q2=metrics["joint_q2"],
                metrics=metrics,
                precise_note=strategy.precise_note,
                scale_y=strategy.scale_y,
                mechanism=strategy.mechanism,
                y_true_pooled=y_true,
                y_pred_pooled=y_pred,
            )
        )

    results.sort(key=lambda r: r.joint_q2, reverse=True)
    return MultiTargetSearchOutput(
        results=results,
        target_names=list(target_names),
        correlation=correlation,
        n_targets=n_targets,
    )
