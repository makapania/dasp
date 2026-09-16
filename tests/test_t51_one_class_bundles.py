"""T-51 PR C: curated one-class extra-axis bundles.

Plan: docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md section 5 ("PR C").
The mechanism itself (resolution, identity, gating, the untouched default path) is
covered by tests/test_t51_extra_axes_mechanism.py; this file pins the three one-class
bundles, that their keys reach the sklearn estimators, and that a real one-class
Bayesian run records them.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from spectral_predict import unified_bayesian as ub
from spectral_predict.contamination import build_one_class_model
from spectral_predict.search_spaces import (
    BUNDLES,
    ExtraAxesConfigError,
    apply_extra_axes,
    discover_derived_keys,
    discover_suggested_names,
    resolve_bundles,
)

ONE_CLASS = frozenset({"one_class"})

# Plan section 5, transcribed independently of the registry so drift is caught.
# axis tuple: (key, kind, low, high, choices, applies_when_id)
PLAN_TABLE: dict[str, dict[str, Any]] = {
    "if_max_samples": {
        "families": {"IsolationForest"},
        "axes": [("max_samples", "categorical", None, None, ("auto", 0.5, 0.8, 1.0), None)],
    },
    "lof_metric": {
        "families": {"LOF"},
        "axes": [
            ("metric", "categorical", None, None, ("euclidean", "manhattan", "cosine"), None)
        ],
    },
    "ocsvm_poly": {
        "families": {"OneClassSVM"},
        "axes": [
            ("degree", "int", 2, 3, None, "oc_kernel_is_poly"),
            ("coef0", "float", -1.0, 1.0, None, "oc_kernel_poly_or_sigmoid"),
        ],
    },
}


@pytest.fixture
def one_class_data():
    rng = np.random.RandomState(42)
    n_features = 20
    X = np.vstack([rng.randn(30, n_features) * 0.3, rng.randn(8, n_features) + 3.0])
    y = np.array(["clean"] * 30 + ["contaminated"] * 8)
    wavelengths = np.array([400.0 + i * 10 for i in range(n_features)])
    return pd.DataFrame(X, columns=[f"{w:.1f}" for w in wavelengths]), pd.Series(y), wavelengths


@pytest.mark.parametrize("bundle_id", sorted(PLAN_TABLE))
def test_registry_matches_the_plan(bundle_id):
    spec = BUNDLES[bundle_id]
    expected = PLAN_TABLE[bundle_id]
    assert set(spec.families) == expected["families"]
    assert set(spec.task_types) == ONE_CLASS
    assert spec.label and spec.help
    assert [
        (a.key, a.kind, a.low, a.high, a.choices, a.applies_when_id) for a in spec.axes
    ] == expected["axes"]


@pytest.mark.parametrize("bundle_id", sorted(PLAN_TABLE))
def test_axes_open_only_params_the_base_sampler_leaves_alone(bundle_id):
    """The rule every bundle must obey: never re-suggest a tuned or derived param."""
    spec = BUNDLES[bundle_id]
    (family,) = spec.families

    def sampler(trial):
        return ub.suggest_one_class_params(trial, family)

    base = discover_suggested_names(sampler) | discover_derived_keys(sampler)
    for axis in spec.axes:
        assert axis.key not in base, f"{axis.key} is already set by {family}'s base sampler"


@pytest.mark.parametrize("bundle_id", sorted(PLAN_TABLE))
def test_supervised_tasks_and_other_families_do_not_resolve(bundle_id):
    spec = BUNDLES[bundle_id]
    (family,) = spec.families
    for task in ("regression", "classification"):
        assert resolve_bundles(family, task, [bundle_id], None, base_param_names=frozenset()) == ()
    other = "LOF" if family != "LOF" else "IsolationForest"
    assert resolve_bundles(other, "one_class", [bundle_id], None, base_param_names=frozenset()) == ()


def test_unknown_bundle_id_still_raises():
    with pytest.raises(ExtraAxesConfigError):
        resolve_bundles(
            "LOF", "one_class", ["lof_metrics"], None, base_param_names=frozenset()
        )


def _row_params(raw):
    """Results rows store Params as a repr string (or a dict in memory)."""
    if isinstance(raw, dict):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}


def _params_for(model_name, bundle_id, fixed):
    """Run one trial of the real base sampler plus the bundle, with fixed suggestions."""
    resolved = resolve_bundles(
        model_name,
        "one_class",
        [bundle_id],
        None,
        base_param_names=(
            discover_suggested_names(lambda t: ub.suggest_one_class_params(t, model_name))
            | discover_derived_keys(lambda t: ub.suggest_one_class_params(t, model_name))
        ),
    )
    assert resolved, "bundle must resolve for its own family/task"
    trial = optuna.trial.FixedTrial(fixed)
    params = ub.suggest_one_class_params(trial, model_name)
    return apply_extra_axes(trial, params, resolved)


def test_if_max_samples_is_written_and_builds():
    params = _params_for(
        "IsolationForest",
        "if_max_samples",
        {"n_estimators": 100, "contamination": 0.05, "max_features": 1.0, "max_samples": 0.8},
    )
    assert params["max_samples"] == 0.8
    model = build_one_class_model("IsolationForest", params)
    assert model.max_samples == 0.8


def test_lof_metric_is_written_and_builds():
    params = _params_for(
        "LOF", "lof_metric", {"n_neighbors": 10, "contamination": 0.05, "metric": "cosine"}
    )
    assert params["metric"] == "cosine"
    model = build_one_class_model("LOF", params)
    assert model.metric == "cosine"


@pytest.mark.parametrize(
    "kernel, expect_degree, expect_coef0",
    [("poly", True, True), ("sigmoid", False, True), ("rbf", False, False)],
)
def test_ocsvm_poly_writes_only_what_the_kernel_uses(kernel, expect_degree, expect_coef0):
    """Suggested on every trial, written only where it changes the fit (plan 3.0)."""
    params = _params_for(
        "OneClassSVM",
        "ocsvm_poly",
        {"nu": 0.05, "kernel": kernel, "gamma": "scale", "degree": 2, "coef0": 0.5},
    )
    assert ("degree" in params) is expect_degree
    assert ("coef0" in params) is expect_coef0
    model = build_one_class_model("OneClassSVM", params)
    assert model.degree == (2 if expect_degree else 3)  # sklearn default is 3
    assert model.coef0 == (0.5 if expect_coef0 else 0.0)


FIT_MATRIX = (
    [
        pytest.param("IsolationForest", "if_max_samples", {"max_samples": value},
                     id=f"IsolationForest-max_samples-{value}")
        for value in ("auto", 0.5, 0.8, 1.0)
    ]
    + [
        pytest.param("LOF", "lof_metric", {"metric": value}, id=f"LOF-metric-{value}")
        for value in ("euclidean", "manhattan", "cosine")
    ]
    + [
        pytest.param("OneClassSVM", "ocsvm_poly",
                     {"kernel": kernel, "degree": degree, "coef0": coef0},
                     id=f"OneClassSVM-{kernel}-d{degree}-c{coef0}")
        for kernel in ("rbf", "poly", "sigmoid")
        for degree, coef0 in ((2, -1.0), (3, 1.0))
    ]
)


@pytest.mark.parametrize(("model_name", "bundle_id", "chosen"), FIT_MATRIX)
def test_every_bundle_value_really_fits(model_name, bundle_id, chosen, one_class_data):
    """Every offered value must fit and predict, not just construct (PR B's T8b)."""
    X, y, _ = one_class_data
    base = {
        "IsolationForest": {"n_estimators": 50, "contamination": 0.05, "max_features": 1.0},
        "LOF": {"n_neighbors": 5, "contamination": 0.05},
        "OneClassSVM": {"nu": 0.05, "gamma": "scale"},
    }[model_name]
    params = _params_for(model_name, bundle_id, {**base, **chosen})
    model = build_one_class_model(model_name, params)
    inliers = X.values[(y == "clean").values]
    model.fit(inliers)
    predictions = model.predict(X.values)
    assert set(np.unique(predictions)) <= {-1, 1}
    assert len(predictions) == len(X)


@pytest.mark.parametrize("model_name", ["IsolationForest", "LOF", "OneClassSVM"])
def test_one_shared_selection_of_every_id_resolves(model_name):
    """The PR D pattern: hand every id to each model; only its own must resolve."""
    base = discover_suggested_names(
        lambda t: ub.suggest_one_class_params(t, model_name)
    ) | discover_derived_keys(lambda t: ub.suggest_one_class_params(t, model_name))
    resolved = resolve_bundles(
        model_name, "one_class", tuple(BUNDLES), None, base_param_names=base
    )
    expected = sorted(
        b.id
        for b in BUNDLES.values()
        if model_name in b.families and "one_class" in b.task_types
    )
    assert [b.id for b in resolved] == expected
    assert expected, "each one-class family owns a bundle"


def test_one_class_bundle_values_reach_a_real_run(one_class_data):
    """End to end: a real one-class Bayesian run records the opened axis in its rows."""
    X, y, wavelengths = one_class_data
    results_df, study = ub.run_unified_bayesian(
        X=X.values,
        y=y.values,
        wavelengths=wavelengths,
        model_name="LOF",
        task_type="one_class",
        n_trials=4,
        cv_folds=3,
        random_state=42,
        verbose=False,
        inlier_class_label="clean",
        enabled_extra_axes=("lof_metric",),
    )
    assert results_df is not None and len(results_df) > 0
    assert all("metric" in t.params for t in study.trials)
    assert {t.params["metric"] for t in study.trials} <= {"euclidean", "manhattan", "cosine"}
    recorded = [_row_params(row) for row in results_df["Params"]]
    assert any("metric" in params for params in recorded), recorded[:2]


def test_enabled_bundle_gives_the_run_its_own_study_name(one_class_data):
    """A bundle run must never resume or pollute a default-space study."""
    X, y, wavelengths = one_class_data
    names = []
    for extra in ((), ("lof_metric",)):
        _, study = ub.run_unified_bayesian(
            X=X.values,
            y=y.values,
            wavelengths=wavelengths,
            model_name="LOF",
            task_type="one_class",
            n_trials=2,
            cv_folds=3,
            random_state=42,
            verbose=False,
            inlier_class_label="clean",
            enabled_extra_axes=extra,
        )
        names.append(study.study_name)
    assert names[0] != names[1]


# --- Codex review of 05223ca ---------------------------------------------------------


@pytest.mark.parametrize(
    ("n_rows", "expected"),
    [
        # sklearn: 'auto' is min(256, n); a fraction is int(fraction * n).
        (24, {"auto": 24, 0.5: 12, 0.8: 19, 1.0: 24}),
        (300, {"auto": 256, 0.5: 150, 0.8: 240, 1.0: 300}),
    ],
)
def test_max_samples_choices_are_distinct_above_256(n_rows, expected):
    """1.0 duplicates 'auto' only below 257 rows; above it, it is the only full sample.

    Dropping 1.0 (as an earlier commit did) would have removed full-sample exploration
    for larger sets on the false premise that 0.5/0.8 exceed 'auto' there: at 300 rows
    0.8 is 240, still under 'auto''s 256.
    """
    from sklearn.ensemble import IsolationForest

    X = np.random.RandomState(0).randn(n_rows, 5)
    resolved = {
        choice: IsolationForest(max_samples=choice, random_state=42, n_jobs=1)
        .fit(X)
        .max_samples_
        for choice in BUNDLES["if_max_samples"].axes[0].choices
    }
    assert resolved == expected
    if n_rows > 256:
        assert len(set(resolved.values())) == len(resolved), "all four are distinct"


def test_a_bundle_run_on_ordinary_data_completes(one_class_data):
    """30 inliers, 3 folds: the ordinary case an enabled bundle must handle."""
    X, y, wavelengths = one_class_data
    results_df, _ = ub.run_unified_bayesian(
        X=X.values,
        y=y.values,
        wavelengths=wavelengths,
        model_name="IsolationForest",
        task_type="one_class",
        n_trials=2,
        cv_folds=3,
        random_state=42,
        verbose=False,
        inlier_class_label="clean",
        enabled_extra_axes=("if_max_samples",),
    )
    assert results_df is not None and len(results_df) > 0


def test_registry_keys_are_exactly_the_two_plan_tables():
    """Restores the exact-membership check that scoping the PR B tests narrowed."""
    from tests.test_t51_supervised_bundles import PLAN_TABLE as SUPERVISED_PLAN

    assert set(BUNDLES) == set(SUPERVISED_PLAN) | set(PLAN_TABLE)


def test_every_bundle_in_the_registry_is_off_by_default():
    for bundle in BUNDLES.values():
        for family in bundle.families:
            for task in bundle.task_types:
                assert resolve_bundles(family, task, (), None) == ()


def test_numeric_labels_with_a_text_inlier_label_still_run():
    """Integer labels with a text inlier_class_label: the objective compares them as
    strings, so this is a supported combination an enabled bundle must not disturb."""
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(30, 8) * 0.3, rng.randn(8, 8) + 3.0])
    y = np.array([1] * 30 + [0] * 8)  # integer labels, text label below
    results_df, _ = ub.run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=np.arange(8, dtype=float),
        model_name="IsolationForest",
        task_type="one_class",
        n_trials=2,
        cv_folds=3,
        random_state=42,
        verbose=False,
        inlier_class_label="1",
        enabled_extra_axes=("if_max_samples",),
    )
    assert results_df is not None and len(results_df) > 0


def test_tiny_one_class_data_behaves_as_documented():
    """Pins the documented limitation (Codex review of 7dafc19).

    Three inliers under repeated 2-fold CV leaves one-row training folds, which the
    fractions cannot fit. Half the folds are enough to score a trial, so a fractional
    row CAN still reach the leaderboard - scored from the successful folds alone, with
    nothing marking it as partial. That is why the help text warns instead of promising
    the trials are merely wasted.
    """
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(3, 10) * 0.3, rng.randn(6, 10) + 3.0])
    y = np.array(["clean"] * 3 + ["contam"] * 6)
    results_df, study = ub.run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=np.arange(400.0, 500.0, 10.0),
        model_name="IsolationForest",
        task_type="one_class",
        n_trials=6,
        cv_folds=2,
        cv_strategy="repeated_kfold",
        cv_n_repeats=2,
        random_state=42,
        verbose=False,
        inlier_class_label="clean",
        enabled_extra_axes=("if_max_samples",),
    )
    scored = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    fractional = [
        t for t in scored if isinstance(t.params.get("max_samples"), float)
    ]
    assert fractional, (
        "documented behaviour: a fractional trial can still be scored here, from the "
        "folds that succeeded"
    )
    if results_df is not None and len(results_df):
        assert "partial_cv" not in results_df.columns, (
            "if a partial-CV marker is ever added, relax the documented warning"
        )
