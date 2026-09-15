"""Post-merge review fixes for the unified Bayesian search and its extra-axes spaces.

Each test reproduces a bug verified on main after T-51 PR B (#74):

- ``convert_study_to_dataframe`` raised ``NameError`` on any baseline trial.
- ``svm_gamma`` resolved for SVM+regression and SVR+classification, pairs with no
  estimator, so every trial silently became a penalty.
- ``model_name='pls-da'`` was not normalised, so ``plsda_head`` never resolved.
- Categorical choices Optuna conflates (``1`` and ``1.0``) passed validation.
- NumPy scalar constants/choices reached the ``Params`` string, which
  ``ast.literal_eval`` cannot parse.
"""
from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import t51_baseline_recipe as recipe  # noqa: E402
from spectral_predict import unified_bayesian as ub  # noqa: E402
from spectral_predict.search_spaces import (  # noqa: E402
    BUNDLES,
    AxisSpec,
    BundleSpec,
    ExtraAxesConfigError,
    canonical_space_identity,
    resolve_bundles,
)


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _run(model: str, task: str, n_trials: int, **kwargs: Any):
    data_fn = {
        "regression": recipe.regression_data,
        "classification": recipe.classification_data,
        "one_class": recipe.one_class_data,
    }[task]
    X, y, wl = data_fn()
    call = {**recipe.COMMON_KWARGS, **kwargs}
    return ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name=model, task_type=task, n_trials=n_trials, **call
    )


# --- baseline_params in convert_study_to_dataframe ------------------------------------


def _baseline_study(apply_baseline: bool) -> optuna.Study:
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    for key, value in {
        "preprocessing": "raw", "n_vars": 8, "subset_tag": "full",
        "apply_baseline": apply_baseline, "apply_smoothing": False, "apply_autoscale": False,
        "window": 0, "deriv": 0, "poly": 0, "all_wavelengths": "1000,2000",
        "full_vars_masked": 8, "model_params": "{'n_components': 2}",
    }.items():
        trial.set_user_attr(key, value)
    study.tell(trial, 0.42)
    return study


def test_convert_study_with_baseline_trial_records_baseline_params() -> None:
    df = ub.convert_study_to_dataframe(
        _baseline_study(apply_baseline=True), model_name="PLS", task_type="regression",
        wavelengths=np.array([1000.0, 2000.0]), n_features=8, cv_folds=3,
        baseline_method="als", baseline_params={"lam": 1e4, "p": 0.05},
    )
    assert df.iloc[0]["baseline_method"] == "als"
    assert df.iloc[0]["baseline_params"] == {"lam": 1e4, "p": 0.05}


def test_convert_study_without_baseline_trial_has_no_baseline_params() -> None:
    df = ub.convert_study_to_dataframe(
        _baseline_study(apply_baseline=False), model_name="PLS", task_type="regression",
        wavelengths=np.array([1000.0, 2000.0]), n_features=8, cv_folds=3,
        baseline_method="als", baseline_params={"lam": 1e4, "p": 0.05},
    )
    assert df.iloc[0]["baseline_params"] is None


@pytest.mark.parametrize("task", ["regression", "classification", "one_class"])
def test_empty_study_dataframe_has_baseline_columns(task: str) -> None:
    df = ub.convert_study_to_dataframe(
        optuna.create_study(), model_name="PLS", task_type=task,
        wavelengths=np.array([1000.0, 2000.0]), n_features=8, cv_folds=3,
    )
    assert df.empty
    assert {"baseline_method", "baseline_params"} <= set(df.columns)


def test_bayesian_run_with_baseline_persists_params_for_validation_rebuild() -> None:
    df, study = _run(
        "PLS", "regression", n_trials=8,
        baseline_method="polynomial", baseline_params={"degree": 3},
    )
    applied = [t for t in study.trials if t.user_attrs.get("apply_baseline")]
    assert applied, "setup: no trial applied baseline correction"
    rows = df[df["baseline_method"].notna()]
    assert len(rows) > 0
    assert all(p == {"degree": 3} for p in rows["baseline_params"])
    assert all(p is None for p in df[df["baseline_method"].isna()]["baseline_params"])


# --- svm_gamma family/task pairs ------------------------------------------------------


@pytest.mark.parametrize(("model", "task"), [("SVM", "regression"), ("SVR", "classification")])
def test_svm_gamma_rejects_pairs_without_an_estimator(model: str, task: str) -> None:
    with pytest.raises(ExtraAxesConfigError, match="does not support"):
        resolve_bundles(model, task, ("svm_gamma",), None)


def test_svm_gamma_rejection_happens_before_any_trial() -> None:
    with pytest.raises(ExtraAxesConfigError, match="does not support"):
        _run("SVM", "regression", n_trials=2, enabled_extra_axes=("svm_gamma",))


@pytest.mark.parametrize(("model", "task"), [("SVM", "classification"), ("SVR", "regression")])
def test_svm_gamma_still_resolves_supported_pairs_with_unchanged_identity(
    model: str, task: str
) -> None:
    (resolved,) = resolve_bundles(model, task, ("svm_gamma",), None)
    unrestricted = BundleSpec(
        id=resolved.id, families=resolved.families, task_types=resolved.task_types,
        axes=resolved.axes, constants=resolved.constants, revision=resolved.revision,
    )
    assert canonical_space_identity((resolved,), False) == canonical_space_identity(
        (unrestricted,), False
    )


@pytest.mark.parametrize("bid", sorted(BUNDLES))
def test_curated_bundles_are_hashable(bid: str) -> None:
    bundle = BUNDLES[bid]
    assert hash(bundle) == hash(_rebuilt_with_fresh_dicts(bundle))
    assert {bundle}


def _rebuilt_with_fresh_dicts(bundle: BundleSpec) -> BundleSpec:
    return BundleSpec(
        id=bundle.id, families=bundle.families, task_types=bundle.task_types,
        axes=bundle.axes, constants=dict(bundle.constants), label=bundle.label,
        help=bundle.help, revision=bundle.revision,
        family_task_types=(
            None if bundle.family_task_types is None else dict(bundle.family_task_types)
        ),
    )


def test_family_task_types_must_be_a_subset() -> None:
    bad = BundleSpec(
        id="b", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True),),
        family_task_types={"PLS": frozenset({"classification"})},
    )
    with pytest.raises(ExtraAxesConfigError, match="family_task_types"):
        resolve_bundles("PLS", "regression", ("b",), {"b": bad})


def test_unrelated_model_still_skips_svm_gamma_silently() -> None:
    assert resolve_bundles("PLS", "regression", ("svm_gamma",), None) == ()
    assert "svm_gamma" in BUNDLES


# --- 'pls-da' spelling ----------------------------------------------------------------


def test_lowercase_plsda_resolves_plsda_head() -> None:
    messages: list[dict] = []
    df, study = _run(
        "pls-da", "classification", n_trials=3, enabled_extra_axes=("plsda_head",),
        progress_callback=messages.append,
    )
    assert all("lr_C" in t.params for t in study.trials), "plsda_head did not resolve"
    assert not any("None of the enabled" in str(m.get("message")) for m in messages)
    assert (df["Model"] == "PLS-DA").all()


# --- categorical choices Optuna conflates ---------------------------------------------


@pytest.mark.parametrize(
    "choices",
    [(1, 1.0), ("sqrt", True, 1), (0.0, -0.0), (np.float64(0.5), 0.5), (None, None)],
    ids=["int-float", "bool-int", "signed-zero", "numpy-float", "none"],
)
def test_categorical_choices_equal_under_optuna_are_rejected(choices: tuple) -> None:
    bundle = BundleSpec(
        id="rf", families=frozenset({"RandomForest"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="max_features", kind="categorical", choices=choices),),
    )
    with pytest.raises(ExtraAxesConfigError, match="duplicate categorical choices"):
        resolve_bundles("RandomForest", "regression", ("rf",), {"rf": bundle})


@pytest.mark.parametrize("nan", [float("nan"), np.float64("nan")], ids=["float", "numpy"])
def test_nan_categorical_choices_are_rejected(nan: float) -> None:
    """``nan != nan`` would slip past the equality check; NaN is rejected outright."""
    bundle = BundleSpec(
        id="rf", families=frozenset({"RandomForest"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="max_features", kind="categorical", choices=(nan, nan)),),
    )
    with pytest.raises(ExtraAxesConfigError, match="non-finite"):
        resolve_bundles("RandomForest", "regression", ("rf",), {"rf": bundle})


def test_distinct_categorical_choices_still_accepted() -> None:
    bundle = BundleSpec(
        id="rf", families=frozenset({"RandomForest"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="max_features", kind="categorical", choices=("1", 1, 0.5)),),
    )
    assert resolve_bundles("RandomForest", "regression", ("rf",), {"rf": bundle})


# --- NumPy scalar literals --------------------------------------------------------------


def _lof_constant(value: Any) -> BundleSpec:
    return BundleSpec(
        id="lof_metric", families=frozenset({"LOF"}), task_types=frozenset({"one_class"}),
        axes=(), constants={"metric": value},
    )


def test_numpy_constants_and_choices_are_normalised_to_builtins() -> None:
    bundle = BundleSpec(
        id="n", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="algorithm", kind="categorical",
                       choices=(np.str_("nipals"), np.int64(2), np.bool_(False))),),
        constants={"tol": np.float64(1e-6), "max_iter": np.int64(500), "flag": np.bool_(True)},
    )
    (resolved,) = resolve_bundles("PLS", "regression", ("n",), {"n": bundle})
    assert [type(c) for c in resolved.axes[0].choices] == [str, int, bool]
    assert {k: type(v) for k, v in resolved.constants.items()} == {
        "tol": float, "max_iter": int, "flag": bool,
    }
    builtin = BundleSpec(
        id="n", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="algorithm", kind="categorical", choices=("nipals", 2, False)),),
        constants={"tol": 1e-6, "max_iter": 500, "flag": True},
    )
    assert canonical_space_identity((resolved,), False) == canonical_space_identity(
        (builtin,), False
    )


def test_numpy_constant_round_trips_through_one_class_params() -> None:
    space = {"lof_metric": _lof_constant(np.str_("manhattan"))}
    df, study = _run(
        "LOF", "one_class", n_trials=2, inlier_class_label=1,
        enabled_extra_axes=("lof_metric",), search_space=space,
    )
    assert len(df) > 0
    for raw in df["Params"]:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        assert parsed["metric"] == "manhattan"
        assert type(parsed["metric"]) is str
