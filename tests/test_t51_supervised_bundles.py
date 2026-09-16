"""T-51 PR B: curated supervised extra-axis bundles.

Test ids follow docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md section 6
(T8a, T8b, T9). The default path (no bundles enabled) is pinned separately by T2/T2b/T3
in ``tests/test_t51_extra_axes_mechanism.py``; nothing here re-blesses those pins.

The round-trip tests build the search-time model the way the Bayesian objective does
(``build_model`` plus the objective's pipeline wrap) and capture its ``Params`` with
the objective's capture function. ``test_t9_real_run_rows_match_search_time_model``
proves that construction reproduces real search rows exactly, so the constructed rows
stand in for search rows with chosen, non-default bundle values.
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

import t51_baseline_recipe as recipe  # noqa: E402
from spectral_predict import search_spaces as ss  # noqa: E402
from spectral_predict import unified_bayesian as ub  # noqa: E402
from spectral_predict.code_generator import CodeGenerator, ExportOptions  # noqa: E402
from spectral_predict.models import CATBOOST_RUNTIME_PARAMS, build_model  # noqa: E402
from spectral_predict.search import _rebuild_model_from_row  # noqa: E402
from spectral_predict.search_spaces import (  # noqa: E402
    BUNDLES,
    AxisSpec,
    BundleSpec,
    ExtraAxesConfigError,
    apply_extra_axes,
    discover_derived_keys,
    discover_suggested_names,
    resolve_bundles,
)

SUPERVISED = frozenset({"regression", "classification"})

# Plan section 3.1, transcribed independently of the registry so drift is caught.
# axis tuple: (key, kind, low, high, log, choices, applies_when_id)
PLAN_TABLE: dict[str, dict[str, Any]] = {
    "rf_features": {
        "families": {"RandomForest"},
        "tasks": SUPERVISED,
        "axes": [
            (
                "max_features",
                "categorical",
                None,
                None,
                False,
                ("sqrt", "log2", 0.1, 0.3, 0.5, 1.0),
                None,
            ),
        ],
        "constants": {},
    },
    "xgb_regularization": {
        "families": {"XGBoost"},
        "tasks": SUPERVISED,
        "axes": [
            ("reg_alpha", "float", 1e-4, 10.0, True, None, None),
            ("reg_lambda", "float", 1e-3, 100.0, True, None, None),
        ],
        "constants": {},
    },
    "xgb_child": {
        "families": {"XGBoost"},
        "tasks": SUPERVISED,
        "axes": [
            ("min_child_weight", "float", 0.5, 20.0, True, None, None),
            ("gamma", "float", 1e-4, 5.0, True, None, None),
        ],
        "constants": {},
    },
    "xgb_sampling": {
        "families": {"XGBoost"},
        "tasks": SUPERVISED,
        "axes": [
            ("colsample_bytree", "float", 0.3, 1.0, False, None, None),
            ("colsample_bylevel", "float", 0.3, 1.0, False, None, None),
        ],
        "constants": {},
    },
    "lgbm_regularization": {
        "families": {"LightGBM"},
        "tasks": SUPERVISED,
        "axes": [
            ("reg_alpha", "float", 1e-4, 10.0, True, None, None),
            ("reg_lambda", "float", 1e-3, 100.0, True, None, None),
        ],
        "constants": {},
    },
    "lgbm_sampling": {
        "families": {"LightGBM"},
        "tasks": SUPERVISED,
        "axes": [
            ("subsample", "float", 0.5, 1.0, False, None, None),
            ("colsample_bytree", "float", 0.3, 1.0, False, None, None),
        ],
        "constants": {},
    },
    "lgbm_child": {
        "families": {"LightGBM"},
        "tasks": SUPERVISED,
        "axes": [
            ("min_child_samples", "int", 2, 50, False, None, None),
            ("min_split_gain", "float", 1e-4, 1.0, True, None, None),
        ],
        "constants": {},
    },
    "catboost_sampling": {
        "families": {"CatBoost"},
        "tasks": SUPERVISED,
        "axes": [
            ("subsample", "float", 0.5, 1.0, False, None, None),
            ("rsm", "float", 0.1, 1.0, False, None, None),
        ],
        "constants": {"bootstrap_type": "Bernoulli"},
    },
    "svm_gamma": {
        "families": {"SVM", "SVR"},
        "tasks": SUPERVISED,
        "axes": [("gamma", "float", 1e-5, 10.0, True, None, "kernel_is_rbf")],
        "constants": {},
    },
    "mlp_activation": {
        "families": {"MLP"},
        "tasks": SUPERVISED,
        "axes": [
            ("activation", "categorical", None, None, False, ("relu", "tanh", "logistic"), None),
        ],
        "constants": {},
    },
    "plsda_head": {
        "families": {"PLS-DA"},
        "tasks": frozenset({"classification"}),
        "axes": [("lr_C", "float", 1e-3, 1e3, True, None, None)],
        "constants": {},
    },
}

# SVM is the classifier and SVR the regressor: build_model has no SVM regressor or SVR
# classifier, so those registry cross-products are never real runs.
_NO_ESTIMATOR = {("SVM", "regression"), ("SVR", "classification")}

# PR C added one-class bundles to the same registry; this file covers the supervised
# ones (tests/test_t51_one_class_bundles.py covers the rest).
SUPERVISED_BUNDLES = {
    bid: bundle
    for bid, bundle in BUNDLES.items()
    if set(bundle.task_types) <= SUPERVISED
}

CASES = [
    pytest.param(bid, family, task, id=f"{bid}-{family}-{task}")
    for bid, bundle in sorted(SUPERVISED_BUNDLES.items())
    for family in sorted(bundle.families)
    for task in sorted(bundle.task_types)
    if (family, task) not in _NO_ESTIMATOR
]

SCALE_SENSITIVE = {"SVM", "SVR", "MLP"}


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        yield


def _reserved(model: str, task: str, n_features: int = 80) -> frozenset[str]:
    sampler = lambda t: ub.suggest_model_params(t, model, n_features, task)  # noqa: E731
    return discover_suggested_names(sampler) | discover_derived_keys(sampler)


def search_time_pipeline(model_name: str, task: str, params: dict[str, Any]) -> Pipeline:
    """The pipeline the unified objective fits for these params (no imbalance/autoscale)."""
    model = build_model(model_name, params, task_type=task)
    if task == "classification" and model_name == "PLS-DA":
        head = LogisticRegression(
            C=params.get("lr_C", 1.0),
            solver=params.get("lr_solver", "lbfgs"),
            max_iter=params.get("lr_max_iter", 1000),
            random_state=42,
        )
        return Pipeline([("pls", model), ("scaler", StandardScaler()), ("lr", head)])
    if model_name in SCALE_SENSITIVE:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return Pipeline([("model", model)])


# --- registry ------------------------------------------------------------------------


def test_registry_matches_plan_table() -> None:
    assert set(SUPERVISED_BUNDLES) == set(PLAN_TABLE)
    for bid, spec in PLAN_TABLE.items():
        bundle = BUNDLES[bid]
        assert bundle.id == bid
        assert set(bundle.families) == spec["families"], bid
        assert set(bundle.task_types) == set(spec["tasks"]), bid
        assert dict(bundle.constants) == spec["constants"], bid
        assert bundle.revision == 1
        assert bundle.label and bundle.help, bid
        got = [
            (a.key, a.kind, a.low, a.high, a.log, a.choices, a.applies_when_id) for a in bundle.axes
        ]
        assert got == spec["axes"], bid
        assert all(a.param_name is None and a.step is None for a in bundle.axes), bid


def test_supervised_and_one_class_bundles_stay_disjoint() -> None:
    """A bundle is supervised or one-class, never both: the samplers differ."""
    for bundle in BUNDLES.values():
        tasks = set(bundle.task_types)
        assert tasks <= SUPERVISED or tasks == {"one_class"}, bundle.id
    assert set(BUNDLES) - set(SUPERVISED_BUNDLES), "PR C's one-class bundles are registered"


def test_every_bundle_is_off_by_default() -> None:
    for bundle in SUPERVISED_BUNDLES.values():
        for family in bundle.families:
            for task in bundle.task_types:
                assert resolve_bundles(family, task, (), None) == ()


@pytest.mark.parametrize(("bid", "family", "task"), CASES)
def test_preflight_against_real_sampler(bid: str, family: str, task: str) -> None:
    resolved = resolve_bundles(family, task, (bid,), None, base_param_names=_reserved(family, task))
    assert resolved == (BUNDLES[bid],)


@pytest.mark.parametrize(
    ("family", "task"),
    sorted(
        {
            (f, t)
            for b in SUPERVISED_BUNDLES.values()
            for f in b.families
            for t in b.task_types
        }
        - _NO_ESTIMATOR
    ),
)
def test_all_registry_bundles_enabled_together_resolve(family: str, task: str) -> None:
    """One shared selection of every id resolves per model without cross-bundle clashes."""
    resolved = resolve_bundles(
        family, task, tuple(BUNDLES), None, base_param_names=_reserved(family, task)
    )
    expected = sorted(
        b.id for b in BUNDLES.values() if family in b.families and task in b.task_types
    )
    assert expected and all(bid in SUPERVISED_BUNDLES for bid in expected)
    assert [b.id for b in resolved] == expected


def test_enabling_a_bundle_changes_the_space_identity() -> None:
    ids = {
        ss.canonical_space_identity((bundle,), search_space_given=False)
        for bundle in BUNDLES.values()
    }
    assert None not in ids
    assert len(ids) == len(BUNDLES)


# --- the `already` guard in apply_extra_axes ------------------------------------------


class _RecordTrial:
    """Minimal trial: records suggestions and returns each distribution's low/first choice."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def suggest_float(self, name: str, low: float, high: float, **_: Any) -> float:
        self.params[name] = low
        return low

    def suggest_int(self, name: str, low: int, high: int, **_: Any) -> int:
        self.params[name] = low
        return low

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        self.params[name] = choices[0]
        return choices[0]


def _spec(bid: str, *axes: AxisSpec, **constants: Any) -> BundleSpec:
    return BundleSpec(
        id=bid,
        families=frozenset({"PLS"}),
        task_types=frozenset({"regression"}),
        axes=axes,
        constants=constants,
    )


def test_apply_catches_duplicate_axis_name_when_resolve_is_bypassed() -> None:
    a = _spec("a", AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True))
    b = _spec("b", AxisSpec(key="tol", kind="float", low=1e-8, high=1e-6, log=True))
    with pytest.raises(ExtraAxesConfigError, match="already suggested"):
        apply_extra_axes(_RecordTrial(), {"n_components": 3}, (a, b))


def test_apply_catches_key_reused_under_an_alias_when_resolve_is_bypassed() -> None:
    a = _spec("a", AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True))
    alias = _spec(
        "b",
        AxisSpec(key="tol", kind="float", low=1e-8, high=1e-6, log=True, param_name="tol_alias"),
    )
    with pytest.raises(ExtraAxesConfigError, match="already suggested"):
        apply_extra_axes(_RecordTrial(), {}, (a, alias))


def test_apply_catches_constant_that_an_axis_writes_when_resolve_is_bypassed() -> None:
    axis = _spec("a", AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True))
    const = _spec("c", AxisSpec(key="max_iter", kind="int", low=400, high=600), tol=1e-6)
    with pytest.raises(ExtraAxesConfigError, match="constant 'tol'"):
        apply_extra_axes(_RecordTrial(), {}, (axis, const))


@pytest.mark.parametrize(
    ("family", "task"),
    [
        ("XGBoost", "regression"),
        ("LightGBM", "classification"),
        ("CatBoost", "classification"),
    ],
)
def test_already_guard_does_not_misfire_on_registry_bundles(family: str, task: str) -> None:
    """Every applicable registry bundle together, constants included, applies cleanly."""
    resolved = resolve_bundles(
        family, task, tuple(BUNDLES), None, base_param_names=_reserved(family, task)
    )
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3)).ask()
    params = ub.suggest_model_params(trial, family, 80, task)
    out = apply_extra_axes(trial, params, resolved)
    for bundle in resolved:
        assert all(axis.key in out for axis in bundle.axes)
        assert all(out[k] == v for k, v in bundle.constants.items())


# --- T8a: suggest + apply + build -----------------------------------------------------


@pytest.mark.parametrize(("bid", "family", "task"), CASES)
def test_t8a_suggest_apply_build(bid: str, family: str, task: str) -> None:
    bundle = BUNDLES[bid]
    resolved = resolve_bundles(family, task, (bid,), None, base_param_names=_reserved(family, task))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11))
    for _ in range(6):  # several seeds so SVM sees both kernels
        trial = study.ask()
        base = ub.suggest_model_params(trial, family, 80, task)
        base_names = set(trial.params)
        out = apply_extra_axes(trial, base, resolved)
        assert {a.optuna_name for a in bundle.axes} <= set(trial.params) - base_names
        for axis in bundle.axes:
            gated = axis.applies_when_id is not None and not ss.PREDICATES[axis.applies_when_id](
                base
            )
            if gated:
                assert out.get(axis.key) == base.get(axis.key)
            else:
                assert out[axis.key] == trial.params[axis.optuna_name]
        assert {k: out[k] for k in bundle.constants} == dict(bundle.constants)
        build_model(family, out, task_type=task)
        study.tell(trial, 0.0)


def test_t8a_svm_gamma_written_only_under_rbf() -> None:
    resolved = (BUNDLES["svm_gamma"],)
    for kernel in ("rbf", "linear"):
        trial = optuna.trial.FixedTrial({"kernel": kernel, "C": 1.0, "gamma": 0.02})
        out = apply_extra_axes(
            trial, ub.suggest_model_params(trial, "SVM", 80, "classification"), resolved
        )
        assert out.get("gamma") == (0.02 if kernel == "rbf" else None)


# --- T8b: real fits at the distribution edges -----------------------------------------


def _tiny(task: str, n_samples: int = 36, n_features: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((n_samples, n_features))
    signal = X[:, 0] - 0.5 * X[:, 3]
    if task == "regression":
        return X, signal + 0.1 * rng.standard_normal(n_samples)
    if task == "binary":
        return X, (signal > np.median(signal)).astype(int)
    return X, np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))


def _edge_configs(bundle: BundleSpec) -> list[dict[str, Any]]:
    n = max([2] + [len(a.choices) for a in bundle.axes if a.kind == "categorical"])
    configs = []
    for i in range(n):
        config = {}
        for axis in bundle.axes:
            if axis.kind == "categorical":
                config[axis.key] = axis.choices[i % len(axis.choices)]
            else:
                config[axis.key] = axis.low if i % 2 == 0 else axis.high
        configs.append(config)
    return configs


_FAST_BASE = {"n_estimators": 20, "iterations": 20, "max_iter": 300}

T8B_CASES = [
    pytest.param(bid, family, fit_task, id=f"{bid}-{family}-{fit_task}")
    for bid, family, task in [p.values for p in CASES]
    for fit_task in (["regression"] if task == "regression" else ["binary", "multiclass"])
]


@pytest.mark.parametrize(("bid", "family", "fit_task"), T8B_CASES)
def test_t8b_fit_at_edges(bid: str, family: str, fit_task: str) -> None:
    task = "regression" if fit_task == "regression" else "classification"
    bundle = BUNDLES[bid]
    X, y = _tiny(fit_task)
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=5)).ask()
    base = ub.suggest_model_params(trial, family, X.shape[1], task)
    if family in ("SVM", "SVR"):
        base["kernel"] = "rbf"
        base["gamma"] = "scale"
    for key, value in _FAST_BASE.items():
        if key in base:
            base[key] = value
    base = apply_extra_axes(trial, base, (bundle,))
    for config in _edge_configs(bundle):
        params = {**base, **config}
        model = search_time_pipeline(family, task, params)
        model.fit(X, y)
        predictions = model.predict(X)
        assert np.all(np.isfinite(np.asarray(predictions, dtype=float)))
        if fit_task == "multiclass":
            assert model.predict_proba(X).shape == (len(y), 3)


# --- T9: short real runs ---------------------------------------------------------------

# One real task per bundle keeps the run count to one per bundle; T8b covers the others.
T9_RUNS = [
    ("rf_features", "RandomForest", "regression"),
    ("xgb_regularization", "XGBoost", "regression"),
    ("xgb_child", "XGBoost", "classification"),
    ("xgb_sampling", "XGBoost", "regression"),
    ("lgbm_regularization", "LightGBM", "classification"),
    ("lgbm_sampling", "LightGBM", "regression"),
    ("lgbm_child", "LightGBM", "regression"),
    ("catboost_sampling", "CatBoost", "classification"),
    ("svm_gamma", "SVR", "regression"),
    ("mlp_activation", "MLP", "classification"),
    ("plsda_head", "PLS-DA", "classification"),
]


def _real_run(family: str, task: str, bundles: tuple[str, ...], n_trials: int = 3):
    X, y, wl = (
        recipe.classification_data() if task == "classification" else recipe.regression_data()
    )
    return ub.run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=wl,
        model_name=family,
        task_type=task,
        n_trials=n_trials,
        enabled_extra_axes=bundles,
        **recipe.COMMON_KWARGS,
    )


def _stored_key(family: str, key: str) -> str:
    if family == "PLS-DA":
        return {"lr_C": "lr__C"}.get(key, f"pls__{key}")
    return f"model__{key}"


@pytest.mark.parametrize(("bid", "family", "task"), T9_RUNS, ids=[r[0] for r in T9_RUNS])
def test_t9_real_run_rows_match_search_time_model(bid: str, family: str, task: str) -> None:
    bundle = BUNDLES[bid]
    df, study = _real_run(family, task, (bid,))
    assert study.user_attrs[ub.EXTRA_AXES_BUNDLES_ATTR] == [f"{bid}@r1"]
    assert ub.EXTRA_AXES_SPACE_ATTR in study.user_attrs
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert completed and all(t.value < 1e9 for t in completed), [t.value for t in completed]
    assert len(df) > 0
    by_number = {t.number: t for t in completed}
    for _, row in df.iterrows():
        trial = by_number[int(row["trial_number"])]
        stored = ast.literal_eval(row["Params"])
        replay = optuna.trial.FixedTrial(trial.params)
        base = ub.suggest_model_params(replay, family, 80, task)
        params = apply_extra_axes(replay, base, (bundle,))
        if "n_components_actual" in trial.user_attrs:  # PLS clamps to the subset size
            params["n_components"] = trial.user_attrs["n_components_actual"]
        # The constructed search-time model reproduces the stored row exactly.
        constructed = ub._capture_serializable_params(search_time_pipeline(family, task, params))
        assert constructed == stored
        for axis in bundle.axes:
            if axis.key in params:
                assert stored[_stored_key(family, axis.key)] == trial.params[axis.optuna_name]
        for key, value in bundle.constants.items():
            assert stored[_stored_key(family, key)] == value
        assert not any(k.rsplit("__", 1)[-1] in CATBOOST_RUNTIME_PARAMS for k in stored)


def test_t9_bundles_give_the_run_its_own_study_name() -> None:
    _, default = _real_run("LightGBM", "regression", (), n_trials=1)
    _, one = _real_run("LightGBM", "regression", ("lgbm_child",), n_trials=1)
    _, two = _real_run("LightGBM", "regression", ("lgbm_sampling", "lgbm_child"), n_trials=1)
    names = {recipe.study_base_name(s.study_name) for s in (default, one, two)}
    assert len(names) == 3
    assert ub.EXTRA_AXES_SPACE_ATTR not in default.user_attrs


def test_t9_svm_gamma_written_only_on_rbf_trials() -> None:
    _, study = _real_run("SVM", "classification", ("svm_gamma",), n_trials=8)
    kernels = set()
    for trial in study.trials:
        stored = ast.literal_eval(trial.user_attrs["model_params"])
        kernels.add(trial.params["kernel"])
        assert "gamma" in trial.params  # suggested uniformly
        if trial.params["kernel"] == "rbf":
            assert stored["model__gamma"] == trial.params["gamma"]
        else:
            # Not written: SVC keeps its own default, not the sampled value.
            assert stored["model__gamma"] == "scale"
    assert kernels == {"rbf", "linear"}


def test_plsda_head_search_time_c_reaches_logistic_regression() -> None:
    _, study = _real_run("PLS-DA", "classification", ("plsda_head",), n_trials=4)
    cs = []
    for trial in study.trials:
        stored = ast.literal_eval(trial.user_attrs["model_params"])
        assert stored["lr__C"] == trial.params["lr_C"]
        assert "lr_C" not in stored and "pls__lr_C" not in stored
        cs.append(trial.params["lr_C"])
    assert len(set(cs)) == len(cs)


# --- T9: round trips (rebuild, save/load, export) --------------------------------------

N_FEATURES = 30


def _round_trip_data(task: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(91)
    X = rng.standard_normal((60, N_FEATURES))
    signal = X[:, 0] - 0.7 * X[:, 3] + 0.4 * X[:, 7] + 0.3 * rng.standard_normal(60)
    if task == "regression":
        y = signal
    elif task == "binary":
        y = (signal > np.median(signal)).astype(int)
    else:
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    X_test = rng.standard_normal((15, N_FEATURES))
    columns = [f"{1000 + 2 * i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=columns), y, X_test


# (family, data task, bundle ids, base Optuna values, bundle Optuna values)
ROUND_TRIPS: dict[str, tuple[str, str, tuple[str, ...], dict[str, Any], dict[str, Any]]] = {
    "RandomForest": (
        "RandomForest",
        "regression",
        ("rf_features",),
        {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
        {"max_features": 0.3},
    ),
    "XGBoost": (
        "XGBoost",
        "regression",
        ("xgb_child", "xgb_regularization", "xgb_sampling"),
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8},
        {
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "min_child_weight": 2.0,
            "gamma": 0.05,
            "colsample_bytree": 0.5,
            "colsample_bylevel": 0.6,
        },
    ),
    "LightGBM": (
        "LightGBM",
        "regression",
        ("lgbm_child", "lgbm_regularization", "lgbm_sampling"),
        {"max_depth": -1, "num_leaves": 15, "n_estimators": 50, "learning_rate": 0.1},
        {
            "min_child_samples": 8,
            "min_split_gain": 0.001,
            "reg_alpha": 0.05,
            "reg_lambda": 3.0,
            "subsample": 0.6,
            "colsample_bytree": 0.5,
        },
    ),
    "CatBoost": (
        "CatBoost",
        "multiclass",
        ("catboost_sampling",),
        {"iterations": 50, "learning_rate": 0.1, "depth": 4, "l2_leaf_reg": 3.0},
        {"subsample": 0.7, "rsm": 0.5},
    ),
    "SVM": (
        "SVM",
        "binary",
        ("svm_gamma",),
        {"kernel": "rbf", "C": 10.0},
        {"gamma": 0.01},
    ),
    "MLP": (
        "MLP",
        "regression",
        ("mlp_activation",),
        {"hidden_size": 32, "n_layers": 1, "alpha": 1e-3, "learning_rate_init": 1e-3},
        {"activation": "tanh"},
    ),
    "PLS-DA": (
        "PLS-DA",
        "binary",
        ("plsda_head",),
        {"n_components": 3},
        {"lr_C": 0.02},
    ),
}


def round_trip_case(name: str) -> dict[str, Any]:
    """Search-time model and its stored results row for one representative per family."""
    family, data_task, bids, base_values, bundle_values = ROUND_TRIPS[name]
    task = "regression" if data_task == "regression" else "classification"
    X, y, X_test = _round_trip_data(data_task)
    trial = optuna.trial.FixedTrial({**base_values, **bundle_values})
    resolved = resolve_bundles(
        family, task, bids, None, base_param_names=_reserved(family, task, N_FEATURES)
    )
    params = apply_extra_axes(
        trial, ub.suggest_model_params(trial, family, N_FEATURES, task), resolved
    )
    for key, value in bundle_values.items():
        assert params[key] == value
    reference = search_time_pipeline(family, task, params)
    row_params = ub._capture_serializable_params(reference)
    for key, value in bundle_values.items():
        assert row_params[_stored_key(family, key)] == value, key
    reference.fit(X.values, y)
    row = {
        "Model": family,
        "Task": task,
        "Params": str(row_params),
        "LVs": params.get("n_components", np.nan),
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
    }
    return {
        "family": family,
        "task": task,
        "X": X,
        "y": y,
        "X_test": X_test,
        "params": params,
        "row_params": row_params,
        "row": row,
        "reference": reference,
        "bundle_values": bundle_values,
        "constants": {k: v for bid in bids for k, v in BUNDLES[bid].constants.items()},
    }


def predictions(model: Any, X: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        return model.predict_proba(X)
    return np.asarray(model.predict(X)).ravel()


def assert_estimator_carries(model: Any, case: dict[str, Any]) -> None:
    params = model.get_params()
    for key, value in {**case["bundle_values"], **case["constants"]}.items():
        stored = _stored_key(case["family"], key)
        candidates = [stored, stored.split("__", 1)[1], f"model__{key}", key]
        found = [params[c] for c in candidates if c in params]
        assert found and found[0] == value, (key, found)


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_t9_rebuild_from_row_reproduces_search_time_model(name: str) -> None:
    case = round_trip_case(name)
    rebuilt = _rebuild_model_from_row(pd.Series(case["row"]), case["task"])
    assert_estimator_carries(rebuilt, case)
    rebuilt.fit(case["X"].values, case["y"])
    np.testing.assert_allclose(
        predictions(rebuilt, case["X_test"], case["task"]),
        predictions(case["reference"], case["X_test"], case["task"]),
        rtol=1e-6,
        atol=1e-8,
    )


def test_t9_lightgbm_rebuild_keeps_bagging_active() -> None:
    """The row's base bagging_freq=1 must survive, or subsample silently does nothing."""
    case = round_trip_case("LightGBM")
    assert case["row_params"]["model__bagging_freq"] == 1
    rebuilt = _rebuild_model_from_row(pd.Series(case["row"]), case["task"])
    assert rebuilt.get_params()["bagging_freq"] == 1
    rebuilt.fit(case["X"].values, case["y"])
    no_bagging = _rebuild_model_from_row(pd.Series(case["row"]), case["task"])
    no_bagging.set_params(bagging_freq=0, subsample_freq=0)
    no_bagging.fit(case["X"].values, case["y"])
    assert not np.allclose(rebuilt.predict(case["X_test"]), no_bagging.predict(case["X_test"]))


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_t9_export_keeps_bundle_params(name: str) -> None:
    case = round_trip_case(name)
    config = {
        "model_name": case["family"],
        "preprocessing": "raw",
        "task_type": case["task"],
        "target_name": "target",
        "params": case["row_params"],
        "metrics": {"R2": 0.0},
        "variable_indices": None,
        "wavelengths": list(range(N_FEATURES)),
        "cv_folds": 3,
        "imbalance_method": None,
    }
    generator = CodeGenerator(config, ExportOptions(format="script"))
    if case["family"] == "PLS-DA":
        _, exported = generator._split_pls_da_params(case["row_params"])
        assert exported["C"] == case["bundle_values"]["lr_C"]
        return
    exported = generator._normalize_model_params(case["row_params"])
    for key, value in {**case["bundle_values"], **case["constants"]}.items():
        assert exported[key] == value, key


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_t9_save_load_and_exported_script_parity(name: str, tmp_path: Path) -> None:
    """Rebuilt model -> save_model/load_model -> exported script: identical predictions."""
    from tests.test_t20_saved_model_export_parity import _run_parity

    case = round_trip_case(name)
    rebuilt = _rebuild_model_from_row(pd.Series(case["row"]), case["task"])
    _run_parity(
        model=rebuilt,
        model_name=case["family"],
        task_type=case["task"],
        X_train=case["X"].values,
        y_train=case["y"],
        X_test=case["X_test"],
        params=case["row_params"],
        imbalance_method=None,
        tmp_path=tmp_path,
    )
    # _run_parity fitted `rebuilt` in place; it must still equal the search-time model.
    np.testing.assert_allclose(
        predictions(rebuilt, case["X_test"], case["task"]),
        predictions(case["reference"], case["X_test"], case["task"]),
        rtol=1e-6,
        atol=1e-8,
    )
