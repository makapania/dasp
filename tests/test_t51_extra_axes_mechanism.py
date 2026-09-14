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
import math
import os
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
        # Set T51_REQUIRE_BASELINE=1 on the machine that blessed the fixture so the
        # prime-directive pin can never silently turn into a skip there.
        if os.environ.get("T51_REQUIRE_BASELINE") == "1":
            pytest.fail("numerical environment differs from t51_default_path_baseline.json")
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
     ("suggest_one_class_params", "compute_importances"),
     ("_build_fit_fingerprint", "_register_or_replay_fingerprint")],
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
    with pytest.raises(ExtraAxesConfigError, match="collides|already suggests"):
        _run(n_trials=1, enable_sqlite_persistence="always",
             enabled_extra_axes=("clash",), search_space={"clash": clash})
    assert not sqlite_storage.exists()


def test_t15_collision_with_objective_name_raises() -> None:
    clash = BundleSpec(
        id="clash", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
        axes=(AxisSpec(key="n_vars", kind="int", low=2, high=30),),
    )
    with pytest.raises(ExtraAxesConfigError, match="collides|already suggests"):
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
    # The pin below only sees literal names; a computed name would evade it silently.
    calls = re.findall(r"\.suggest_(?:int|float|categorical)\(\s*([^,\)]*)", source)
    non_literal = [c for c in calls if not re.fullmatch(r"\s*['\"][^'\"]+['\"]\s*", c)]
    assert non_literal == [], f"suggest_* with non-literal names: {non_literal}"
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


def _svm_gated_study() -> tuple[optuna.Study, Any, dict[str, Any]]:
    X, y, wl = recipe.classification_data()
    objective = ub.create_unified_objective(
        X, y, wl, "SVM", task_type="classification", cv_folds=3, random_state=42,
        y_original=y, seen_fingerprints={}, resolved_extra_axes=(GATED,),
    )
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(0))
    study.optimize(objective, n_trials=1)
    return study, objective, dict(study.trials[0].params)


@pytest.mark.parametrize(("kernel", "replays"), [("linear", True), ("rbf", False)])
def test_t7_dedup_replay_through_objective(kernel, replays) -> None:
    """Unwritten bundle values replay a prior fit; written ones are fitted afresh."""
    study, objective, template = _svm_gated_study()
    for gamma in (0.1, 3.0):
        study.enqueue_trial({**template, "kernel": kernel, "gamma": gamma})
    study.optimize(objective, n_trials=2)
    first, second = study.trials[1], study.trials[2]
    assert first.params["gamma"] != second.params["gamma"]
    assert first.user_attrs.get(ub.DUPLICATE_OF_TRIAL_ATTR) is None
    duplicate_of = second.user_attrs.get(ub.DUPLICATE_OF_TRIAL_ATTR)
    if replays:
        assert duplicate_of == first.number
        assert second.value == first.value
    else:
        assert duplicate_of is None


def test_registry_key_must_match_bundle_id() -> None:
    with pytest.raises(ExtraAxesConfigError, match="Registry key"):
        resolve_bundles("PLS", "regression", ("x",), {"x": PLS_TOL})


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
    assert reopened.user_attrs[ub.EXTRA_AXES_SPACE_ATTR] == study.user_attrs[ub.EXTRA_AXES_SPACE_ATTR]
    assert len(reopened.trials) == ub._AUTO_WARMUP + 1
    assert all("tol" in t.params for t in reopened.trials)


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


def test_t13_identity_depends_only_on_effective_space(baseline) -> None:
    assert canonical_space_identity((), search_space_given=True) is None
    registry_digest = canonical_space_identity((PLS_TOL,), search_space_given=False)
    assert canonical_space_identity((PLS_TOL,), search_space_given=True) == registry_digest
    _, study = _run(n_trials=1, search_space=SPACE)  # custom space, nothing selected
    assert (
        recipe.study_base_name(study.study_name)
        == baseline["default_study_base_names"]["PLS|regression"]
    )


def test_numpy_scalar_bounds_accepted_and_hash_like_python() -> None:
    plain = BundleSpec(id="n", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                       axes=(AxisSpec(key="max_iter", kind="int", low=400, high=600, step=10),))
    numpy_ = BundleSpec(id="n", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                        axes=(AxisSpec(key="max_iter", kind="int", low=np.int64(400),
                                       high=np.int64(600), step=np.int64(10)),))
    assert resolve_bundles("PLS", "regression", ("n",), {"n": numpy_})
    assert canonical_space_identity((plain,), False) == canonical_space_identity((numpy_,), False)


def test_bare_string_selection_rejected() -> None:
    with pytest.raises(ExtraAxesConfigError, match="sequence of bundle ids"):
        resolve_bundles("PLS", "regression", "probe_pls_tol", SPACE)


def test_empty_bundle_rejected() -> None:
    empty = {"e": BundleSpec(id="e", families=frozenset({"PLS"}),
                             task_types=frozenset({"regression"}), axes=())}
    with pytest.raises(ExtraAxesConfigError, match="no axes"):
        resolve_bundles("PLS", "regression", ("e",), empty)


def test_malformed_non_applicable_bundle_fails_on_every_model() -> None:
    """A shared multi-model selection must fail on the first model, not mid-batch."""
    broken_ridge = {"r": BundleSpec(id="r", families=frozenset({"Ridge"}),
                                    task_types=frozenset({"regression"}),
                                    axes=(AxisSpec(key="tol", kind="float", low=1.0, high=0.1),))}
    with pytest.raises(ExtraAxesConfigError, match="low"):
        resolve_bundles("PLS", "regression", ("r",), broken_ridge)


@pytest.mark.parametrize("value", [5.9, "5", True])
def test_non_integer_startup_trials_rejected(value) -> None:
    with pytest.raises(ValueError, match="integer"):
        _run(n_trials=1, n_startup_trials=value)


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
        ({"kind": "categorical", "choices": ()}, "choices only"),
        ({"kind": "categorical", "low": None, "high": None, "choices": ()}, "non-empty choices"),
        ({"kind": "categorical", "low": object(), "high": None, "choices": ("a",)}, "choices only"),
        ({"choices": (1.0,)}, "do not take choices"),
        ({"low": float("nan")}, "finite"),
        ({"high": float("inf")}, "finite"),
        ({"kind": "int", "low": 1, "high": 9, "step": 1.5}, "step"),
        ({"kind": "int", "low": 1, "high": 9, "step": True}, "step"),
        ({"log": 1}, "log must be a bool"),
        ({"kind": "categorical", "low": None, "high": None, "choices": (float("nan"),)}, "non-finite"),
        ({"key": ""}, "non-empty string"),
        ({"param_name": 3}, "non-empty string"),
        ({"kind": "categorical", "low": None, "high": None, "choices": ({"a": 1},)}, "not allowed"),
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


def test_axis_alias_cannot_override_base_sampled_key() -> None:
    base = discover_suggested_names(lambda t: ub.suggest_model_params(t, "PLS", 80, "regression"))
    alias = {"a": BundleSpec(id="a", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                             axes=(AxisSpec(key="n_components", kind="int", low=2, high=9,
                                            param_name="alternate"),))}
    with pytest.raises(ExtraAxesConfigError, match="alias"):
        resolve_bundles("PLS", "regression", ("a",), alias, base_param_names=base)


def test_optuna_name_alias_colliding_with_base_is_rejected() -> None:
    """Key is new, but the Optuna name shadows a base suggestion: Optuna 5 would reuse it."""
    base = discover_suggested_names(lambda t: ub.suggest_model_params(t, "PLS", 80, "regression"))
    shadow = {"s": BundleSpec(id="s", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                              axes=(AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, log=True,
                                             param_name="n_components"),))}
    with pytest.raises(ExtraAxesConfigError, match="collides"):
        resolve_bundles("PLS", "regression", ("s",), shadow, base_param_names=base)


def _reserved(model: str, task: str) -> frozenset[str]:
    if task == "one_class":
        sampler = lambda t: ub.suggest_one_class_params(t, model)  # noqa: E731
    else:
        sampler = lambda t: ub.suggest_model_params(t, model, 500, task)  # noqa: E731
    return discover_suggested_names(sampler) | ss.discover_derived_keys(sampler)


def _bundle(bid: str, family: str, task: str, *axes: AxisSpec, **constants: Any) -> BundleSpec:
    return BundleSpec(id=bid, families=frozenset({family}), task_types=frozenset({task}),
                      axes=axes, constants=constants)


def test_derived_keys_are_reserved_but_pinned_constants_are_not() -> None:
    mlp = _reserved("MLP", "regression")
    assert "hidden_layer_sizes" in mlp  # derived from hidden_size and n_layers
    assert "max_iter" not in mlp  # genuinely pinned
    svm = _reserved("SVM", "classification")
    assert "gamma" not in svm  # 'scale' whenever present: a pinned constant
    lgbm = _reserved("LightGBM", "regression")
    assert {"num_leaves", "max_depth"} <= lgbm
    assert not {"reg_alpha", "subsample", "min_child_samples"} & lgbm


@pytest.mark.parametrize("as_constant", [False, True], ids=["axis", "constant"])
def test_derived_key_override_rejected(as_constant) -> None:
    if as_constant:
        bundle = _bundle("d", "MLP", "regression",
                         AxisSpec(key="activation", kind="categorical", choices=("relu",)),
                         hidden_layer_sizes=5)
    else:
        bundle = _bundle("d", "MLP", "regression",
                         AxisSpec(key="hidden_layer_sizes", kind="int", low=2, high=9,
                                  param_name="hls_alias"))
    with pytest.raises(ExtraAxesConfigError, match="derives"):
        resolve_bundles("MLP", "regression", ("d",), {"d": bundle},
                        base_param_names=_reserved("MLP", "regression"))


def test_derived_key_override_rejected_via_entry_point(sqlite_storage) -> None:
    """run_unified_bayesian must feed derived keys into pre-flight, not only names."""
    bundle = _bundle("derived_override", "MLP", "regression",
                     AxisSpec(key="hidden_layer_sizes", kind="int", low=2, high=9,
                              param_name="hls_alias"))
    with pytest.raises(ExtraAxesConfigError, match="derives"):
        _run("MLP", "regression", n_trials=1, enable_sqlite_persistence="always",
             enabled_extra_axes=("derived_override",),
             search_space={"derived_override": bundle})
    assert not sqlite_storage.exists()


PLANNED_BUNDLES = [
    ("RandomForest", "regression", _bundle("rf_features", "RandomForest", "regression",
        AxisSpec(key="max_features", kind="categorical", choices=("sqrt", "log2", 0.1, 0.3, 0.5, 1.0)))),
    ("XGBoost", "regression", _bundle("xgb_regularization", "XGBoost", "regression",
        AxisSpec(key="reg_alpha", kind="float", low=1e-4, high=10.0, log=True),
        AxisSpec(key="reg_lambda", kind="float", low=1e-3, high=100.0, log=True))),
    ("XGBoost", "regression", _bundle("xgb_child", "XGBoost", "regression",
        AxisSpec(key="min_child_weight", kind="float", low=0.5, high=20.0, log=True),
        AxisSpec(key="gamma", kind="float", low=1e-4, high=5.0, log=True))),
    ("XGBoost", "regression", _bundle("xgb_sampling", "XGBoost", "regression",
        AxisSpec(key="colsample_bytree", kind="float", low=0.3, high=1.0),
        AxisSpec(key="colsample_bylevel", kind="float", low=0.3, high=1.0))),
    ("LightGBM", "regression", _bundle("lgbm_sampling", "LightGBM", "regression",
        AxisSpec(key="subsample", kind="float", low=0.5, high=1.0),
        AxisSpec(key="colsample_bytree", kind="float", low=0.3, high=1.0))),
    ("LightGBM", "regression", _bundle("lgbm_child", "LightGBM", "regression",
        AxisSpec(key="min_child_samples", kind="int", low=2, high=50),
        AxisSpec(key="min_split_gain", kind="float", low=1e-4, high=1.0, log=True))),
    ("CatBoost", "classification", _bundle("catboost_sampling", "CatBoost", "classification",
        AxisSpec(key="subsample", kind="float", low=0.5, high=1.0),
        AxisSpec(key="rsm", kind="float", low=0.1, high=1.0), bootstrap_type="Bernoulli")),
    ("SVM", "classification", _bundle("svm_gamma", "SVM", "classification",
        AxisSpec(key="gamma", kind="float", low=1e-5, high=10.0, log=True,
                 applies_when_id="kernel_is_rbf"))),
    ("MLP", "regression", _bundle("mlp_activation", "MLP", "regression",
        AxisSpec(key="activation", kind="categorical", choices=("relu", "tanh", "logistic")))),
    ("PLS-DA", "classification", _bundle("plsda_head", "PLS-DA", "classification",
        AxisSpec(key="lr_C", kind="float", low=1e-3, high=1e3, log=True))),
    ("IsolationForest", "one_class", _bundle("if_max_samples", "IsolationForest", "one_class",
        AxisSpec(key="max_samples", kind="categorical", choices=("auto", 0.5, 0.8, 1.0)))),
    ("LOF", "one_class", _bundle("lof_metric", "LOF", "one_class",
        AxisSpec(key="metric", kind="categorical", choices=("euclidean", "manhattan", "cosine")))),
    ("OneClassSVM", "one_class", _bundle("ocsvm_poly", "OneClassSVM", "one_class",
        AxisSpec(key="degree", kind="int", low=2, high=3, applies_when_id="oc_kernel_is_poly"),
        AxisSpec(key="coef0", kind="float", low=-1.0, high=1.0,
                 applies_when_id="oc_kernel_poly_or_sigmoid"))),
]


PLANNED_BUNDLES.append(("LightGBM", "regression", _bundle(
    "lgbm_regularization", "LightGBM", "regression",
    AxisSpec(key="reg_alpha", kind="float", low=1e-4, high=10.0, log=True),
    AxisSpec(key="reg_lambda", kind="float", low=1e-3, high=100.0, log=True))))

# Plan section 3.1 bundles that apply to both task types, and SVR for svm_gamma.
_OTHER_TASK = {
    "rf_features": ("RandomForest", "classification"),
    "xgb_regularization": ("XGBoost", "classification"),
    "xgb_child": ("XGBoost", "classification"),
    "xgb_sampling": ("XGBoost", "classification"),
    "lgbm_sampling": ("LightGBM", "classification"),
    "lgbm_child": ("LightGBM", "classification"),
    "lgbm_regularization": ("LightGBM", "classification"),
    "catboost_sampling": ("CatBoost", "regression"),
    "svm_gamma": ("SVR", "regression"),
    "mlp_activation": ("MLP", "classification"),
}
PLANNED_BUNDLES.extend(
    (model, task, BundleSpec(id=b.id, families=frozenset({model}), task_types=frozenset({task}),
                             axes=b.axes, constants=b.constants))
    for _, _, b in list(PLANNED_BUNDLES) if b.id in _OTHER_TASK
    for model, task in [_OTHER_TASK[b.id]]
)


@pytest.mark.parametrize(("model", "task", "bundle"), PLANNED_BUNDLES,
                         ids=[f"{b.id}-{t}" for _, t, b in PLANNED_BUNDLES])
def test_planned_bundles_pass_preflight(model, task, bundle) -> None:
    """Guard for PR B/C: validation must not over-reject the bundles the plan specifies."""
    resolved = resolve_bundles(model, task, (bundle.id,), {bundle.id: bundle},
                               base_param_names=_reserved(model, task))
    assert resolved == (bundle,)


def test_owner_tokens_cannot_be_forged_by_keys() -> None:
    forged = _bundle("f", "PLS", "regression",
                     AxisSpec(key="const:tol", kind="float", low=1e-7, high=1e-5, param_name="tol"),
                     tol=1e-6)
    with pytest.raises(ExtraAxesConfigError, match="writes as a key|suggests as an Optuna"):
        resolve_bundles("PLS", "regression", ("f",), {"f": forged})


def test_derived_key_probes_cover_midpoint_and_cancelling_inputs() -> None:
    def midpoint_only(trial):
        x = trial.suggest_int("x", 0, 10)
        return {"x": x, "flag": "on" if 3 <= x <= 7 else "off"}

    def cancelling(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_float("y", 0.0, 1.0)
        return {"x": x, "y": y, "diff": x - y, "fixed": 0.5}

    def present_only_mid_range(trial):
        x = trial.suggest_int("x", 0, 10)
        return {"x": x, **({"derived": x} if 3 <= x <= 7 else {})}

    assert "flag" in ss.discover_derived_keys(midpoint_only)
    derived = ss.discover_derived_keys(cancelling)
    assert "diff" in derived and "fixed" not in derived
    assert "derived" in ss.discover_derived_keys(present_only_mid_range)


def test_probe_values_stay_on_grid_and_finite() -> None:
    assert ss._probe_value(50, 500, 0.5, True, 50, False) % 50 == 0
    assert 50 <= ss._probe_value(50, 500, 0.5, True, 50, False) <= 500
    assert math.isfinite(ss._probe_value(1e308, 1.1e308, 0.5, False, None, True))
    assert math.isfinite(ss._probe_value(-1e308, 1e308, 0.25, False, None, False))
    assert ss._probe_value(1e-4, 100.0, 0.5, False, None, True) == pytest.approx(0.1)


def test_generator_selection_rejected() -> None:
    with pytest.raises(ExtraAxesConfigError, match="sequence of bundle ids"):
        resolve_bundles("PLS", "regression", (b for b in ("probe_pls_tol",)), SPACE)


@pytest.mark.parametrize(
    ("bundle_kwargs", "match"),
    [
        ({"constants": ["x"]}, "mapping"),
        ({"axes": AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5)}, "tuple of AxisSpec"),
        ({"revision": 1.5}, "revision"),
        ({"revision": 0}, "revision"),
        ({"families": frozenset()}, "families and task_types"),
    ],
)
def test_malformed_bundle_shape_rejected(bundle_kwargs, match) -> None:
    fields = {"id": "m", "families": frozenset({"PLS"}), "task_types": frozenset({"regression"}),
              "axes": PLS_TOL.axes, **bundle_kwargs}
    with pytest.raises(ExtraAxesConfigError, match=match):
        resolve_bundles("PLS", "regression", ("m",), {"m": BundleSpec(**fields)})


def test_step_must_align_with_bounds_and_choices_must_be_unique() -> None:
    with pytest.raises(ExtraAxesConfigError, match="multiple of step"):
        resolve_bundles("PLS", "regression", ("b",),
                        _one_axis(kind="int", low=1, high=8, step=2))
    with pytest.raises(ExtraAxesConfigError, match="duplicate"):
        resolve_bundles("PLS", "regression", ("b",),
                        _one_axis(kind="categorical", low=None, high=None, choices=("a", "a")))


@pytest.mark.parametrize("selection", [5, {"probe_pls_tol": True}, b"probe_pls_tol"])
def test_non_sequence_selection_rejected(selection) -> None:
    with pytest.raises(ExtraAxesConfigError, match="sequence of bundle ids"):
        resolve_bundles("PLS", "regression", selection, SPACE)


@pytest.mark.parametrize(
    ("low_a", "low_b", "step_a", "step_b"),
    [(0, 0.0, None, None), (0.0, -0.0, None, None)],
)
def test_equivalent_float_bounds_share_identity(low_a, low_b, step_a, step_b) -> None:
    a = _bundle("e", "PLS", "regression", AxisSpec(key="tol", kind="float", low=low_a, high=1.0))
    b = _bundle("e", "PLS", "regression", AxisSpec(key="tol", kind="float", low=low_b, high=1.0))
    assert canonical_space_identity((a,), False) == canonical_space_identity((b,), False)


def test_int_step_none_equals_step_one_and_np_str_equals_str() -> None:
    a = _bundle("e", "PLS", "regression", AxisSpec(key="max_iter", kind="int", low=1, high=9))
    b = _bundle("e", "PLS", "regression",
                AxisSpec(key="max_iter", kind="int", low=1, high=9, step=1))
    assert canonical_space_identity((a,), False) == canonical_space_identity((b,), False)
    c = _bundle("c", "LOF", "one_class", AxisSpec(key="metric", kind="categorical", choices=("x",)))
    d = _bundle("c", "LOF", "one_class",
                AxisSpec(key="metric", kind="categorical", choices=(np.str_("x"),)))
    assert canonical_space_identity((c,), False) == canonical_space_identity((d,), False)


@pytest.mark.parametrize(
    ("axis", "match"),
    [({"low": -1e308, "high": 1e308}, "span"),
     ({"kind": "int", "low": 1, "high": 10**400}, "2\\*\\*53")],
)
def test_unrepresentable_bounds_rejected(axis, match) -> None:
    with pytest.raises(ExtraAxesConfigError, match=match):
        resolve_bundles("PLS", "regression", ("b",), _one_axis(**axis))


def test_string_families_rejected() -> None:
    bad = BundleSpec(id="s", families="PLS-DA", task_types=frozenset({"classification"}),
                     axes=PLS_TOL.axes)
    with pytest.raises(ExtraAxesConfigError, match="set of strings"):
        resolve_bundles("PLS", "classification", ("s",), {"s": bad})


def test_non_string_ids_rejected() -> None:
    with pytest.raises(ExtraAxesConfigError, match="must be strings"):
        resolve_bundles("PLS", "regression", ("probe_pls_tol", 3), SPACE)


def test_optuna_name_of_one_axis_cannot_be_key_of_another() -> None:
    writes_k = _bundle("a", "PLS", "regression",
                       AxisSpec(key="tol", kind="float", low=1e-7, high=1e-5, param_name="tol_n"))
    names_k = _bundle("b", "PLS", "regression",
                      AxisSpec(key="max_iter", kind="int", low=400, high=600, param_name="tol"))
    with pytest.raises(ExtraAxesConfigError, match="writes as a key|suggests as an Optuna"):
        resolve_bundles("PLS", "regression", ("a", "b"), {"a": writes_k, "b": names_k})


def test_runtime_guard_checks_written_keys_too() -> None:
    alias = _bundle("x", "PLS", "regression",
                    AxisSpec(key="n_components", kind="int", low=2, high=9, param_name="alt"))
    trial = _FixedTrial({"alt": 3})
    trial.params["n_components"] = 5
    with pytest.raises(ExtraAxesConfigError, match="clashes"):
        apply_extra_axes(trial, {"n_components": 5}, (alias,))


def test_non_string_constant_keys_rejected() -> None:
    bad = {"b": BundleSpec(id="b", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                           axes=PLS_TOL.axes, constants={1: 2})}
    with pytest.raises(ExtraAxesConfigError, match="constant key"):
        resolve_bundles("PLS", "regression", ("b",), bad)


def test_malformed_axis_fails_before_storage_via_entry_point(sqlite_storage) -> None:
    with pytest.raises(ExtraAxesConfigError, match="finite"):
        _run(n_trials=2, enable_sqlite_persistence="always", enabled_extra_axes=("b",),
             search_space=_one_axis(low=float("nan")))
    assert not sqlite_storage.exists()


def _raise_on_trial(monkeypatch: pytest.MonkeyPatch, from_trial: int) -> None:
    real = ub.apply_extra_axes

    def failing(trial, params, resolved):
        if trial.number >= from_trial:
            raise ExtraAxesConfigError("injected runtime configuration error")
        return real(trial, params, resolved)

    monkeypatch.setattr(ub, "apply_extra_axes", failing)


def test_config_error_propagates_after_migration_restart(sqlite_storage, monkeypatch) -> None:
    monkeypatch.setattr(ub, "_AUTO_THRESHOLD_S", 0.0)
    _raise_on_trial(monkeypatch, from_trial=ub._AUTO_WARMUP + 1)
    with pytest.raises(ExtraAxesConfigError, match="injected"):
        _run(n_trials=ub._AUTO_WARMUP + 4, enable_sqlite_persistence="auto",
             enabled_extra_axes=("probe_pls_tol",), search_space=SPACE)
    # Documented decision: completed trials are retained, never deleted, on runtime errors.
    url = f"sqlite:///{sqlite_storage.as_posix()}"
    (name,) = optuna.study.get_all_study_names(storage=url)
    stored = optuna.load_study(study_name=name, storage=url)
    completed = [t for t in stored.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == ub._AUTO_WARMUP + 1


def test_one_class_config_error_propagates(monkeypatch) -> None:
    _raise_on_trial(monkeypatch, from_trial=0)
    lof_metric = BundleSpec(
        id="probe_lof_metric", families=frozenset({"LOF"}), task_types=frozenset({"one_class"}),
        axes=(AxisSpec(key="metric", kind="categorical", choices=("euclidean", "manhattan")),),
    )
    with pytest.raises(ExtraAxesConfigError, match="injected"):
        _run("LOF", "one_class", n_trials=2, inlier_class_label=1,
             enabled_extra_axes=("probe_lof_metric",),
             search_space={"probe_lof_metric": lof_metric})


def test_constants_must_be_literals_and_unique_across_bundles() -> None:
    odd = {"b": BundleSpec(id="b", families=frozenset({"PLS"}), task_types=frozenset({"regression"}),
                           axes=PLS_TOL.axes, constants={"flags": frozenset({1})})}
    with pytest.raises(ExtraAxesConfigError, match="not allowed"):
        resolve_bundles("PLS", "regression", ("b",), odd)
    writes_tol = BundleSpec(id="c", families=frozenset({"PLS"}),
                            task_types=frozenset({"regression"}), axes=PLS_ITER.axes,
                            constants={"tol": 1e-6})
    with pytest.raises(ExtraAxesConfigError, match="both write"):
        resolve_bundles("PLS", "regression", ("probe_pls_tol", "c"),
                        {"probe_pls_tol": PLS_TOL, "c": writes_tol})


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
