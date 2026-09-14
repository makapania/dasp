"""T-51 PR A: opt-in extra-axes mechanism.

Test ids follow docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md section 6.
The prime directive is that with no bundles enabled the default Bayesian search and its
study names are unchanged; T2b and T3 pin that against a baseline captured on main
after PR #67 (tests/fixtures/t51_default_path_baseline.json).
"""
from __future__ import annotations

import importlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pytest
from optuna.storages import InMemoryStorage

sys.path.insert(0, str(Path(__file__).parent))

import t51_baseline_recipe as recipe  # noqa: E402
from spectral_predict import search_spaces as ss  # noqa: E402
from spectral_predict import unified_bayesian as ub  # noqa: E402
from spectral_predict.search_spaces import (  # noqa: E402
    AxisSpec,
    BundleSpec,
    ExtraAxesConfigError,
    apply_extra_axes,
    canonical_space_identity,
    discover_suggested_names,
    resolve_bundles,
)

FIXTURE = Path(__file__).parent / "fixtures" / "t51_default_path_baseline.json"

# PLS pins tol/max_iter as constants, so a bundle may open them additively.
PLS_TOL = BundleSpec(
    id="probe_pls_tol",
    families=frozenset({"PLS"}),
    task_types=frozenset({"regression"}),
    axes=(AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True),),
)
PLS_ITER = BundleSpec(
    id="probe_pls_iter",
    families=frozenset({"PLS"}),
    task_types=frozenset({"regression"}),
    axes=(AxisSpec(key="max_iter", kind="int", low=400, high=600),),
)
RIDGE_ONLY = BundleSpec(
    id="probe_ridge_only",
    families=frozenset({"Ridge"}),
    task_types=frozenset({"regression"}),
    axes=(AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True),),
)
SPACE = {b.id: b for b in (PLS_TOL, PLS_ITER, RIDGE_ONLY)}


@pytest.fixture(autouse=True)
def _quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture()
def baseline() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture()
def sqlite_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route run_unified_bayesian's storage URL to a temp SQLite file."""
    db = tmp_path / "t51.sqlite3"
    url = f"sqlite:///{db.as_posix()}?check_same_thread=False&timeout=30"
    # Resolve the module at call time: test_t41_* pops and re-imports run_state, so a
    # reference captured at import could be a stale module object.
    monkeypatch.setattr(importlib.import_module("spectral_predict.run_state"),
                        "get_storage_url", lambda: url)
    return db


def _run(model: str = "PLS", task: str = "regression", n_trials: int = 3, **kwargs: Any):
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


class _ExplodingTrial:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"apply_extra_axes touched trial.{name} with nothing enabled")


class _FixedTrial:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self.params: dict[str, Any] = {}

    def _take(self, name: str, *_: Any, **__: Any) -> Any:
        self.params[name] = self._values[name]
        return self._values[name]

    suggest_int = suggest_float = _take

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        return self._take(name)


# --- T1 ------------------------------------------------------------------------------


def test_t1_apply_is_a_no_op_without_bundles() -> None:
    params = {"n_components": 5}
    assert apply_extra_axes(_ExplodingTrial(), params, ()) is params


def test_t1_resolve_is_a_no_op_without_bundles() -> None:
    assert resolve_bundles("PLS", "regression", (), None) == ()
    assert canonical_space_identity((), search_space_given=False) is None


# --- T2 / T2b: default study names ---------------------------------------------------


@pytest.mark.parametrize(("model", "task", "data_fn", "extra"), recipe.NAME_CASES)
def test_t2b_default_study_base_names_are_pinned(baseline, model, task, data_fn, extra) -> None:
    """Literal pin, no __version__ interpolation: a version bump fails this on purpose."""
    _, study = _run(model, task, n_trials=1, **extra)
    expected = baseline["default_study_base_names"][f"{model}|{task}"]
    assert recipe.study_base_name(study.study_name) == expected


def test_t2_default_study_name_env_suffix() -> None:
    _, study = _run(n_trials=1)
    digest = ub._environment_digest(ub._numerical_environment())
    assert study.study_name.endswith(f"_{ub.ENV_FINGERPRINT_VERSION}_{digest}")
    assert ub.EXTRA_AXES_SPACE_ATTR not in study.user_attrs
    assert ub.N_STARTUP_TRIALS_REQUESTED_ATTR not in study.user_attrs


def test_t2_empty_selection_and_default_startup_keep_default_name(baseline) -> None:
    _, study = _run(n_trials=1, enabled_extra_axes=(), n_startup_trials=None)
    assert (
        recipe.study_base_name(study.study_name)
        == baseline["default_study_base_names"]["PLS|regression"]
    )


# --- T3: default TPE trajectory ------------------------------------------------------


def _trace(study: optuna.Study) -> list[dict[str, Any]]:
    return json.loads(
        json.dumps(
            [{"number": t.number, "params": t.params, "value": t.value} for t in study.trials],
            sort_keys=True,
            default=str,
        )
    )


@pytest.mark.parametrize("kwargs", [{}, {"enabled_extra_axes": ()}], ids=["omitted", "empty"])
def test_t3_default_trajectory_matches_main(baseline, kwargs) -> None:
    if ub._environment_digest(ub._numerical_environment()) != baseline["environment_digest"]:
        pytest.skip(
            "t51_default_path_baseline.json was captured in a different numerical "
            "environment; re-bless it in a reviewed commit (plan section 6, T3)"
        )
    _, study = _run(n_trials=recipe.TRACE_TRIALS, **kwargs)
    assert len(study.trials) == recipe.TRACE_TRIALS > ub.DEFAULT_N_STARTUP_TRIALS
    got, expected = _trace(study), baseline["pls_regression_trace"]
    # Params exactly; values to rel 1e-9, because the env digest does not capture BLAS.
    assert [(t["number"], t["params"]) for t in got] == [
        (t["number"], t["params"]) for t in expected
    ]
    assert [t["value"] for t in got] == pytest.approx([t["value"] for t in expected], rel=1e-9)


def _function_source(source: str, name: str, following: str) -> str:
    match = re.search(rf"^def {name}\(.*?(?=^def {following}\()", source, re.S | re.M)
    assert match, f"could not locate {name}"
    return match.group(0).rstrip()


@pytest.mark.parametrize(
    ("name", "following"),
    [("suggest_model_params", "suggest_one_class_params"),
     ("suggest_one_class_params", "compute_importances")],
)
def test_base_sampler_bodies_unchanged_from_main(baseline, name, following) -> None:
    """Prime directive: bundles are additive; the default samplers are never edited.

    PR F is the one planned exception (plan section 5) and must re-bless this pin.
    """
    import hashlib

    source = Path(ub.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    digest = hashlib.sha256(_function_source(source, name, following).encode("utf-8")).hexdigest()
    assert digest == baseline["sampler_source_sha256"][name]


# --- T4 / T10: startup trials on all three sampler paths -----------------------------


@pytest.fixture()
def sampler_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    real = ub.TPESampler

    def spy(*args: Any, **kwargs: Any):
        seen.append(kwargs["n_startup_trials"])
        return real(*args, **kwargs)

    monkeypatch.setattr(ub, "TPESampler", spy)
    return seen


@pytest.mark.parametrize(("startup", "expected"), [(None, 20), (5, 5)])
def test_t4_t10_startup_trials_survive_auto_migration(
    sqlite_storage, sampler_spy, monkeypatch, startup, expected
) -> None:
    monkeypatch.setattr(ub, "_AUTO_THRESHOLD_S", 0.0)
    _, study = _run(n_trials=ub._AUTO_WARMUP + 2, enable_sqlite_persistence="auto",
                    n_startup_trials=startup)
    assert not isinstance(study._storage, InMemoryStorage), "auto-migration did not happen"
    assert sqlite_storage.exists()
    assert len(study.trials) == ub._AUTO_WARMUP + 2
    assert len(sampler_spy) >= 2  # initial in-memory sampler + migrated sampler
    assert set(sampler_spy) == {expected}


@pytest.mark.parametrize(("startup", "expected"), [(None, 20), (5, 5)])
def test_t4_t10_startup_trials_on_always_reattach(
    sqlite_storage, sampler_spy, startup, expected
) -> None:
    _, study = _run(n_trials=2, enable_sqlite_persistence="always", n_startup_trials=startup)
    assert not isinstance(study._storage, InMemoryStorage)
    assert study.sampler._n_startup_trials == expected
    assert set(sampler_spy) == {expected}


def test_t10_resume_reattaches_with_the_new_startup_value(sqlite_storage) -> None:
    _, first = _run(n_trials=2, enable_sqlite_persistence="always", n_startup_trials=5)
    _, resumed = _run(n_trials=3, enable_sqlite_persistence="always", n_startup_trials=9)
    assert resumed.study_name == first.study_name
    assert resumed.sampler._n_startup_trials == 9
    assert resumed.user_attrs[ub.N_STARTUP_TRIALS_REQUESTED_ATTR] == 9


def test_t10_startup_attr_only_when_passed() -> None:
    _, study = _run(n_trials=1, n_startup_trials=7)
    assert study.user_attrs[ub.N_STARTUP_TRIALS_REQUESTED_ATTR] == 7
    assert study.sampler._n_startup_trials == 7


def test_invalid_startup_trials_rejected() -> None:
    with pytest.raises(ValueError, match="n_startup_trials"):
        _run(n_trials=1, n_startup_trials=0)


# --- T5 / T15: fail fast before storage ----------------------------------------------


def test_t5_unknown_bundle_raises_before_storage(sqlite_storage) -> None:
    with pytest.raises(ExtraAxesConfigError, match="Unknown extra-axes bundle"):
        _run(n_trials=1, enable_sqlite_persistence="always", enabled_extra_axes=("nope",))
    assert not sqlite_storage.exists()


def test_t15_collision_with_base_sampler_raises_before_storage(sqlite_storage) -> None:
    clash = BundleSpec(
        id="clash",
        families=frozenset({"PLS"}),
        task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="n_components", kind="int", low=2, high=30),),
    )
    with pytest.raises(ExtraAxesConfigError, match="collides"):
        _run(n_trials=1, enable_sqlite_persistence="always",
             enabled_extra_axes=("clash",), search_space={"clash": clash})
    assert not sqlite_storage.exists()


def test_t15_collision_with_objective_name_raises() -> None:
    clash = BundleSpec(
        id="clash", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="n_vars", kind="int", low=2, high=30),),
    )
    with pytest.raises(ExtraAxesConfigError, match="collides"):
        resolve_bundles("PLS", "regression", ("clash",), {"clash": clash})


def test_t15_objective_reraises_config_error_instead_of_penalty() -> None:
    """Runtime guard: a collision that slips past pre-flight aborts the study."""
    X, y, wl = recipe.regression_data()
    clash = BundleSpec(
        id="clash", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="n_components", kind="int", low=2, high=30),),
    )
    objective = ub.create_unified_objective(
        X, y, wl, "PLS", cv_folds=3, resolved_extra_axes=(clash,)
    )
    study = optuna.create_study(direction="minimize")
    with pytest.raises(ExtraAxesConfigError):
        study.optimize(objective, n_trials=1)


def test_objective_reserved_names_match_source() -> None:
    source = Path(ub.__file__).read_text(encoding="utf-8")
    samplers = re.search(
        r"^def suggest_model_params\(.*?^def create_unified_objective\(", source, re.S | re.M
    ).group(0)
    names = set(re.findall(r"suggest_(?:int|float|categorical)\(\s*['\"]([^'\"]+)", source))
    sampler_names = set(
        re.findall(r"suggest_(?:int|float|categorical)\(\s*['\"]([^'\"]+)", samplers)
    )
    # Names suggested only outside the model samplers must all be reserved.
    assert names - sampler_names <= ss.OBJECTIVE_RESERVED_NAMES
    assert ss.OBJECTIVE_RESERVED_NAMES <= names


def test_discover_follows_categorical_branches() -> None:
    lgbm = discover_suggested_names(
        lambda t: ub.suggest_model_params(t, "LightGBM", 500, "regression")
    )
    assert {"max_depth", "num_leaves", "n_estimators", "learning_rate"} <= lgbm
    svr = discover_suggested_names(lambda t: ub.suggest_model_params(t, "SVR", 500, "regression"))
    assert {"kernel", "C", "epsilon"} <= svr
    oc = discover_suggested_names(lambda t: ub.suggest_one_class_params(t, "OneClassSVM"))
    assert oc == {"nu", "kernel", "gamma"}


# --- T6: canonicalisation ------------------------------------------------------------


def test_t6_order_and_duplicates_do_not_change_name_or_trajectory() -> None:
    _, a = _run(n_trials=4, enabled_extra_axes=("probe_pls_tol", "probe_pls_iter"),
                search_space=SPACE)
    _, b = _run(n_trials=4, enabled_extra_axes=("probe_pls_iter", "probe_pls_tol", "probe_pls_tol"),
                search_space=SPACE)
    assert a.study_name == b.study_name
    assert _trace(a) == _trace(b)
    assert {"tol", "max_iter"} <= set(a.trials[0].params)


def test_t6_different_applicable_sets_give_different_names() -> None:
    _, one = _run(n_trials=1, enabled_extra_axes=("probe_pls_tol",), search_space=SPACE)
    _, two = _run(n_trials=1, enabled_extra_axes=("probe_pls_tol", "probe_pls_iter"),
                  search_space=SPACE)
    assert one.study_name != two.study_name


def test_t6_non_applicable_only_keeps_default_name_and_warns_once(
    baseline, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(ss, "BUNDLES", {"probe_ridge_only": RIDGE_ONLY})
    messages: list[str] = []
    with caplog.at_level(logging.WARNING, logger=ub.logger.name):
        _, study = _run(n_trials=1, enabled_extra_axes=("probe_ridge_only",),
                        progress_callback=lambda e: messages.append(e.get("message", "")))
    assert (
        recipe.study_base_name(study.study_name)
        == baseline["default_study_base_names"]["PLS|regression"]
    )
    warned = [r for r in caplog.records if "apply to PLS" in r.getMessage()]
    assert len(warned) == 1
    assert sum("apply to PLS" in m for m in messages) == 1
    assert ub.EXTRA_AXES_SPACE_ATTR not in study.user_attrs


def test_t6_registry_bundles_resolve_via_default_registry(monkeypatch) -> None:
    monkeypatch.setattr(ss, "BUNDLES", dict(SPACE))
    _, study = _run(n_trials=2, enabled_extra_axes=("probe_pls_tol",))
    assert study.user_attrs[ub.EXTRA_AXES_BUNDLES_ATTR] == ["probe_pls_tol@r1"]
    assert "tol" in study.trials[0].params


# --- T7: fingerprint exactness -------------------------------------------------------


def _fingerprint(params: dict[str, Any]) -> tuple:
    return ub._build_fit_fingerprint(
        preprocess_config={"name": "raw"}, subset_type="full", subset_tag="full", n_vars=10,
        top_indices=None, model_name="SVM", task_type="classification", model_params=params,
        imbalance_method=None, imbalance_params=None,
        use_sample_weight_for_classification=False, resolved_class_weight=(),
        tail_lr_random_state=None, early_stopping_rounds=None, use_early_stopping=False,
        baseline_method=None, baseline_params=None, smoothing_window=17, smoothing_polyorder=2,
    )


GATED = BundleSpec(
    id="gated_gamma", families=frozenset({"SVM"}), task_types=frozenset({"classification"}),
    axes=(AxisSpec(key="gamma", kind="float", low=1e-5, high=10.0, log=True,
                   applies_when_id="kernel_is_rbf"),),
)


def test_t7_unwritten_values_do_not_split_fingerprints() -> None:
    base = {"C": 1.0, "kernel": "linear"}
    p1 = apply_extra_axes(_FixedTrial({"gamma": 0.1}), dict(base), (GATED,))
    p2 = apply_extra_axes(_FixedTrial({"gamma": 3.0}), dict(base), (GATED,))
    assert "gamma" not in p1
    assert _fingerprint(p1) == _fingerprint(p2)


def test_t7_written_values_split_fingerprints_and_suggest_is_uniform() -> None:
    base = {"C": 1.0, "kernel": "rbf"}
    t1, t2 = _FixedTrial({"gamma": 0.1}), _FixedTrial({"gamma": 3.0})
    p1 = apply_extra_axes(t1, dict(base), (GATED,))
    p2 = apply_extra_axes(t2, dict(base), (GATED,))
    assert (p1["gamma"], p2["gamma"]) == (0.1, 3.0)
    assert _fingerprint(p1) != _fingerprint(p2)
    assert _fingerprint(p1) == _fingerprint(dict(p1))
    linear_trial = _FixedTrial({"gamma": 0.1})
    apply_extra_axes(linear_trial, {"kernel": "linear"}, (GATED,))
    assert "gamma" in linear_trial.params  # suggested even when not written


def test_constants_are_written() -> None:
    bundle = BundleSpec(
        id="c", families=frozenset({"X"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="a", kind="int", low=1, high=2),), constants={"mode": "fixed"},
    )
    out = apply_extra_axes(_FixedTrial({"a": 2}), {"z": 0}, (bundle,))
    assert out == {"z": 0, "a": 2, "mode": "fixed"}


# --- T12: resume ---------------------------------------------------------------------


def test_t12_resume_identity_and_attrs(sqlite_storage, baseline) -> None:
    kw = {"enable_sqlite_persistence": "always", "search_space": SPACE}
    _, first = _run(n_trials=2, enabled_extra_axes=("probe_pls_tol",), **kw)
    _, again = _run(n_trials=4, enabled_extra_axes=("probe_pls_tol",), **kw)
    assert again.study_name == first.study_name
    assert len(again.trials) == 4  # n_trials is a total target

    _, other = _run(n_trials=1, enabled_extra_axes=("probe_pls_iter",), **kw)
    assert other.study_name != first.study_name

    _, default = _run(n_trials=1, enable_sqlite_persistence="always")
    assert (
        recipe.study_base_name(default.study_name)
        == baseline["default_study_base_names"]["PLS|regression"]
    )

    url = f"sqlite:///{sqlite_storage.as_posix()}"
    reopened = optuna.load_study(study_name=first.study_name, storage=url)
    assert reopened.user_attrs[ub.EXTRA_AXES_BUNDLES_ATTR] == ["probe_pls_tol@r1"]
    assert reopened.user_attrs[ub.EXTRA_AXES_SPACE_ATTR]
    assert len(reopened.trials) == 4


def test_t12_space_attrs_survive_auto_migration(sqlite_storage, monkeypatch) -> None:
    monkeypatch.setattr(ub, "_AUTO_THRESHOLD_S", 0.0)
    _, study = _run(n_trials=ub._AUTO_WARMUP + 1, enable_sqlite_persistence="auto",
                    enabled_extra_axes=("probe_pls_tol",), search_space=SPACE)
    assert not isinstance(study._storage, InMemoryStorage)
    url = f"sqlite:///{sqlite_storage.as_posix()}"
    reopened = optuna.load_study(study_name=study.study_name, storage=url)
    assert reopened.user_attrs[ub.EXTRA_AXES_BUNDLES_ATTR] == ["probe_pls_tol@r1"]


# --- T13: identity serialisation -----------------------------------------------------


def _variant(**axis_changes: Any) -> BundleSpec:
    axis = AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True)
    fields = {**axis.__dict__, **axis_changes}
    return BundleSpec(id="v", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                      axes=(AxisSpec(**fields),))


def test_t13_identity_covers_every_semantic_field() -> None:
    base = canonical_space_identity((_variant(),), True)
    variants = [
        _variant(param_name="tol_alias"),
        _variant(applies_when_id="kernel_is_rbf"),
        _variant(high=1e-4),
        _variant(log=False),
        BundleSpec(id="v", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                   axes=_variant().axes, revision=2),
        BundleSpec(id="v", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                   axes=_variant().axes, constants={"k": 1}),
    ]
    digests = {canonical_space_identity((v,), True) for v in variants}
    assert base not in digests and len(digests) == len(variants)


def test_t13_choice_types_are_distinct() -> None:
    def cat(choices: tuple[Any, ...]) -> str | None:
        bundle = BundleSpec(id="c", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                            axes=(AxisSpec(key="k", kind="categorical", choices=choices),))
        return canonical_space_identity((bundle,), True)

    assert len({cat((1,)), cat((1.0,)), cat((True,)), cat(("1",))}) == 4


def test_t13_custom_space_always_has_identity() -> None:
    assert canonical_space_identity((), search_space_given=True) is not None


def test_t13_unknown_predicate_rejected() -> None:
    bad = BundleSpec(id="bad", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                     axes=(AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5,
                                    applies_when_id="no_such_predicate"),))
    with pytest.raises(ExtraAxesConfigError, match="applies_when_id"):
        resolve_bundles("PLS", "regression", ("bad",), {"bad": bad})


def test_t13_search_space_replaces_registry(monkeypatch) -> None:
    monkeypatch.setattr(ss, "BUNDLES", {"probe_pls_tol": PLS_TOL})
    with pytest.raises(ExtraAxesConfigError, match="Unknown"):
        resolve_bundles("PLS", "regression", ("probe_pls_tol",), {"probe_pls_iter": PLS_ITER})
    assert resolve_bundles("PLS", "regression", ("probe_pls_tol",), None) == (PLS_TOL,)


def test_resolved_bundles_are_snapshots() -> None:
    space = {"probe_pls_tol": PLS_TOL}
    resolved = resolve_bundles("PLS", "regression", ("probe_pls_tol",), space)
    assert resolved == (PLS_TOL,) and resolved[0] is not PLS_TOL


def _one_axis(**axis: Any) -> dict[str, BundleSpec]:
    fields = {"key": "tol", "kind": "float", "low": 1e-7, "high": 1e-5, **axis}
    return {"b": BundleSpec(id="b", families=frozenset({"PLS"}),
                            task_types=frozenset({"regression"}), axes=(AxisSpec(**fields),))}


@pytest.mark.parametrize(
    ("axis", "match"),
    [
        ({"kind": "integer"}, "unknown kind"),
        ({"low": 1e-5, "high": 1e-7}, "low"),
        ({"kind": "categorical", "choices": ()}, "non-empty choices"),
        ({"kind": "categorical", "choices": ({"a": 1},)}, "not allowed"),
        ({"log": True, "low": 0.0}, "log scale"),
        ({"kind": "int", "low": 1.5, "high": 3}, "bounds"),
        ({"kind": "int", "low": 1, "high": 9, "step": 2, "log": True}, "step"),
    ],
)
def test_malformed_axes_rejected_before_optimisation(axis, match) -> None:
    with pytest.raises(ExtraAxesConfigError, match=match):
        resolve_bundles("PLS", "regression", ("b",), _one_axis(**axis))


def test_constants_cannot_override_suggested_params() -> None:
    base = discover_suggested_names(lambda t: ub.suggest_model_params(t, "PLS", 80, "regression"))
    bad = {"b": BundleSpec(id="b", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                           axes=PLS_TOL.axes, constants={"n_components": 3})}
    with pytest.raises(ExtraAxesConfigError, match="override"):
        resolve_bundles("PLS", "regression", ("b",), bad, base_param_names=base)


def test_constants_must_be_literals_and_unique_across_bundles() -> None:
    odd = {"b": BundleSpec(id="b", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                           axes=PLS_TOL.axes, constants={"flags": frozenset({1})})}
    with pytest.raises(ExtraAxesConfigError, match="not allowed"):
        resolve_bundles("PLS", "regression", ("b",), odd)
    writes_tol = BundleSpec(id="c", families=frozenset({"PLS"}),
                            task_types=frozenset({"regression"}), axes=PLS_ITER.axes,
                            constants={"tol": 1e-6})
    with pytest.raises(ExtraAxesConfigError, match="both write"):
        resolve_bundles("PLS", "regression", ("b", "c"), {"b": PLS_TOL, "c": writes_tol})


def test_same_key_under_different_optuna_names_rejected() -> None:
    alias = BundleSpec(id="alias", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                       axes=(AxisSpec(key="tol", kind="float", low=1e-8, high=1e-6, log=True,
                                      param_name="tol_alias"),))
    with pytest.raises(ExtraAxesConfigError, match="both write"):
        resolve_bundles("PLS", "regression", ("probe_pls_tol", "alias"),
                        {"probe_pls_tol": PLS_TOL, "alias": alias})


def test_duplicate_axis_across_bundles_rejected() -> None:
    twin = BundleSpec(id="twin", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                      axes=PLS_TOL.axes)
    with pytest.raises(ExtraAxesConfigError, match="both suggest"):
        resolve_bundles("PLS", "regression", ("probe_pls_tol", "twin"),
                        {"probe_pls_tol": PLS_TOL, "twin": twin})


def test_one_class_insertion_point(monkeypatch) -> None:
    lof_metric = BundleSpec(
        id="probe_lof_metric", families=frozenset({"LOF"}), task_types=frozenset({"one_class"}),
        axes=(AxisSpec(key="metric", kind="categorical", choices=("euclidean", "manhattan")),),
    )
    df, study = _run("LOF", "one_class", n_trials=3, inlier_class_label=1,
                     enabled_extra_axes=("probe_lof_metric",),
                     search_space={"probe_lof_metric": lof_metric})
    assert all("metric" in t.params for t in study.trials)
    assert any("metric" in str(p) for p in df["Params"])
