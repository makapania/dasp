"""QW1 thread budget: assert the configuration the policy produces, never wall-clock."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict import parallel_policy
from spectral_predict.parallel_policy import (
    CVPlan,
    estimator_threads,
    limit_estimator_threads,
    plan_cv,
)


@pytest.fixture
def cores(monkeypatch):
    """Pin the core count so plans are machine-independent."""

    def set_cores(n: int) -> None:
        monkeypatch.setattr(parallel_policy, "physical_cores", lambda: n)

    set_cores(24)
    return set_cores


@pytest.fixture
def not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)


# --------------------------------------------------------------------------- plan_cv


def test_parallel_plan_splits_cores_between_pool_and_fits(cores, not_frozen):
    plan = plan_cv(5, 49, 2151)
    assert plan == CVPlan(n_jobs=5, backend="loky", model_threads=4)
    assert plan.n_jobs * plan.model_threads <= 24


def test_pool_is_bounded_by_cores_and_fits_go_single_threaded(cores, not_frozen):
    cores(8)
    plan = plan_cv(49, 49, 2151)  # LOO-sized
    assert plan.n_jobs == 8
    assert plan.model_threads == 1


@pytest.mark.parametrize("n_splits", [2, 3, 5, 10, 24, 100])
@pytest.mark.parametrize("n_cores", [1, 2, 4, 6, 24])
def test_plan_never_oversubscribes(cores, not_frozen, n_splits, n_cores):
    cores(n_cores)
    plan = plan_cv(n_splits, 100, 1000)
    assert plan.n_jobs <= min(n_splits, n_cores)
    if plan.parallel:
        assert plan.n_jobs * plan.model_threads <= n_cores
    else:
        assert plan.backend == "sequential"


def test_requested_n_jobs_bounds_the_pool(cores, not_frozen):
    assert plan_cv(10, 100, 1000, requested_n_jobs=3).n_jobs == 3


def test_tiny_job_runs_serially_single_threaded(cores, not_frozen):
    plan = plan_cv(5, 10, 100)  # 5k cells
    assert plan == CVPlan(n_jobs=1, backend="sequential", model_threads=1)
    assert not plan.parallel


def test_single_split_runs_serially(cores, not_frozen):
    # One big fit: serial, and free to use every core (not forced to one thread).
    assert plan_cv(1, 1000, 1000) == CVPlan(1, "sequential", None)


def test_requested_n_jobs_zero_is_rejected(cores, not_frozen):
    with pytest.raises(ValueError):
        plan_cv(5, 100, 1000, requested_n_jobs=0)


def test_negative_requested_n_jobs_counts_back_from_cpu_count(cores, not_frozen, monkeypatch):
    import joblib

    monkeypatch.setattr(joblib, "effective_n_jobs", lambda n: 3)
    assert plan_cv(10, 100, 1000, requested_n_jobs=-2).n_jobs == 3


def test_catboost_folds_are_serial_in_threaded_pools(cores, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert plan_cv(5, 49, 2151, model_name="CatBoost") == CVPlan(1, "sequential", None)


def test_catboost_folds_use_the_process_pool_when_not_frozen(cores, not_frozen):
    assert plan_cv(5, 49, 2151, model_name="CatBoost") == CVPlan(5, "loky", 4)


@pytest.mark.parametrize("model", sorted(parallel_policy.MODELS_PREFER_SERIAL_CV))
def test_serial_preferred_models_keep_their_threading(cores, not_frozen, model):
    assert plan_cv(5, 100, 2000, model_name=model) == CVPlan(1, "sequential", None)


def test_explicit_serial_request_leaves_model_untouched(cores, not_frozen):
    assert plan_cv(5, 100, 2000, requested_n_jobs=1) == CVPlan(1, "sequential", None)


def test_frozen_bundle_uses_threading_backend(cores, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    plan = plan_cv(5, 49, 2151)
    assert plan.backend == "threading"
    assert plan.n_jobs == 5 and plan.model_threads == 4


def test_search_frozen_helper_delegates_to_policy(monkeypatch):
    from spectral_predict.search import _frozen_needs_threading_fallback

    monkeypatch.delattr(sys, "frozen", raising=False)
    assert _frozen_needs_threading_fallback() is False
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert _frozen_needs_threading_fallback() is True


def test_backend_context_routes_sklearn_pools(cores, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    from joblib.parallel import get_active_backend

    plan = plan_cv(5, 49, 2151)
    with plan.backend_context():
        backend, _ = get_active_backend()
    assert type(backend).__name__ == "ThreadingBackend"


def test_pool_model_threads(cores):
    cores(24)
    assert parallel_policy.pool_model_threads(6) == 4
    assert parallel_policy.pool_model_threads(48) == 1


def test_task_pool_plan_serialises_catboost_in_threaded_pools(cores, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert parallel_policy.task_pool_plan(-1, model_name="CatBoost") == CVPlan(
        1, "sequential", None
    )
    lgbm = parallel_policy.task_pool_plan(6, model_name="LightGBM")
    assert lgbm == CVPlan(6, "threading", 4)


def test_task_pool_plan_keeps_catboost_parallel_with_processes(cores, not_frozen):
    assert parallel_policy.task_pool_plan(6, model_name="CatBoost") == CVPlan(6, "loky", 4)


def test_ga_candidate_pool_routes_through_policy(cores, monkeypatch):
    from spectral_predict.ga_preprocessing import _candidate_pool_plan

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert not _candidate_pool_plan(-1, {"name": "CatBoost", "params": {}}).parallel
    assert _candidate_pool_plan(-1, {"name": "XGBoost", "params": {}}).backend == "threading"
    assert _candidate_pool_plan(-1, None).parallel


def test_contains_catboost_sees_wrapped_and_non_terminal_models():
    catboost = pytest.importorskip("catboost")
    from sklearn.model_selection import GridSearchCV

    cb = catboost.CatBoostRegressor(verbose=False, allow_writing_files=False)
    assert parallel_policy.contains_catboost(cb)
    assert parallel_policy.contains_catboost(GridSearchCV(cb, {"depth": [3]}))
    assert parallel_policy.contains_catboost(
        Pipeline([("model", cb), ("scaler", StandardScaler())])
    )
    assert not parallel_policy.contains_catboost(
        Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor())])
    )


def test_spa_seed_pool_is_physical_cores_with_capped_blas(cores):
    from spectral_predict.variable_selection import _spa_seed_plan

    cores(24)
    plan = _spa_seed_plan()
    assert plan.backend == "threading" and plan.n_jobs == 8 and plan.model_threads == 3
    cores(4)
    plan = _spa_seed_plan()
    assert plan.n_jobs == 4 and plan.n_jobs * plan.model_threads <= 4


def test_caller_sized_pools_are_capped_at_physical_cores(cores, monkeypatch):
    import joblib

    cores(8)
    monkeypatch.setattr(joblib, "effective_n_jobs", lambda n: 16)  # 8c/16t machine, n_jobs=-1
    assert parallel_policy.pool_workers(-1) == 8
    assert parallel_policy.pool_workers(-1) * parallel_policy.pool_model_threads(-1) <= 8


def test_threading_plan_caps_shared_blas_pool_in_workers(cores, monkeypatch):
    """Effective limits, measured from a raised baseline (the suite's env pins 1)."""
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    if not _pool_threads("blas"):
        pytest.skip("no BLAS runtime visible to threadpoolctl")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    plan = plan_cv(5, 49, 2151)  # 24 cores pinned -> 5 workers x 4 threads
    assert plan.backend == "threading" and plan.model_threads == 4
    with threadpool_limits(limits=8, user_api="blas"):
        assert _pool_threads("blas") == {8}
        with plan.backend_context():
            seen = Parallel(n_jobs=plan.n_jobs, backend=plan.backend)(
                delayed(_pool_threads)("blas") for _ in range(5)
            )
        assert all(s == {4} for s in seen)
        assert _pool_threads("blas") == {8}


def test_loky_plan_leaves_blas_to_joblib(cores, not_frozen):
    from threadpoolctl import threadpool_limits

    if not _pool_threads("blas"):
        pytest.skip("no BLAS runtime visible to threadpoolctl")
    plan = plan_cv(5, 49, 2151)
    with threadpool_limits(limits=8, user_api="blas"):
        with plan.backend_context():
            assert _pool_threads("blas") == {8}  # loky workers get cores//workers via env


# ------------------------------------------------------------- limit_estimator_threads


def _boosters():
    lightgbm = pytest.importorskip("lightgbm")
    xgboost = pytest.importorskip("xgboost")
    catboost = pytest.importorskip("catboost")
    return [
        lightgbm.LGBMRegressor(n_jobs=-1, verbosity=-1),
        xgboost.XGBRegressor(n_jobs=-1),
        RandomForestRegressor(n_jobs=-1),
        catboost.CatBoostRegressor(verbose=False, allow_writing_files=False),
    ]


@pytest.mark.parametrize("idx", range(4))
def test_limit_caps_bare_and_pipelined_models_without_mutating(idx):
    model = _boosters()[idx]
    before = model.get_params()
    capped = limit_estimator_threads(model, 1)
    assert set(estimator_threads(capped).values()) == {1}
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    capped_pipe = limit_estimator_threads(pipe, 3)
    assert estimator_threads(capped_pipe) == {"model": 3}
    assert model.get_params() == before  # original untouched -> captured params unchanged


def test_limit_none_returns_same_object():
    model = RandomForestRegressor(n_jobs=-1)
    assert limit_estimator_threads(model, None) is model


def test_limit_skips_imblearn_resamplers():
    imblearn = pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipe = ImbPipeline([("imbalance", SMOTE()), ("model", RandomForestRegressor(n_jobs=-1))])
    capped = limit_estimator_threads(pipe, 1)
    assert capped.named_steps["model"].n_jobs == 1
    assert capped.named_steps["imbalance"].get_params() == SMOTE().get_params()
    assert imblearn  # imported for the skip


def test_limit_reaches_models_held_by_wrappers():
    from sklearn.ensemble import BaggingRegressor

    wrapper = BaggingRegressor(estimator=RandomForestRegressor(n_jobs=-1), n_jobs=-1)
    capped = limit_estimator_threads(wrapper, 2)
    assert capped.n_jobs == 2 and capped.estimator.n_jobs == 2
    assert wrapper.estimator.n_jobs == -1


# ------------------------------------------------------------------- OpenMP limiter


def _openmp_threads() -> set[int]:
    from threadpoolctl import threadpool_info

    return {i["num_threads"] for i in threadpool_info() if i["user_api"] == "openmp"}


def _pool_threads(user_api: str) -> set[int]:
    from threadpoolctl import threadpool_info

    return {i["num_threads"] for i in threadpool_info() if i["user_api"] == user_api}


@pytest.fixture
def raised_openmp():
    """OpenMP at 3 threads, so a cap to 1 and its restore are observable."""
    import sklearn.neighbors  # noqa: F401 -- loads sklearn's OpenMP runtime
    from threadpoolctl import threadpool_limits

    if not _openmp_threads():
        pytest.skip("no OpenMP runtime loaded")
    with threadpool_limits(limits=3, user_api="openmp"):
        assert _openmp_threads() == {3}
        yield


def test_openmp_single_threaded_limits_and_restores(raised_openmp):
    with parallel_policy.openmp_single_threaded():
        assert _openmp_threads() == {1}
    assert _openmp_threads() == {3}


def test_overlapping_caps_from_two_threads_restore_the_original(raised_openmp):
    """Non-LIFO overlap: A enters, B enters, A exits, B exits -> back to 3, not stuck at 1."""
    import threading

    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def thread_a():
        with parallel_policy.openmp_single_threaded():
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def thread_b():
        a_in.wait(5)
        with parallel_policy.openmp_single_threaded():
            b_in.set()
            a_out.wait(5)
            seen["after_a_left"] = _openmp_threads()

    workers = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(10)
    assert seen["after_a_left"] == {1}  # B's cap survives A's exit
    assert _openmp_threads() == {3}  # originals back once both left


def test_nested_stricter_cap_applies_and_outer_restores(raised_openmp):
    with parallel_policy.native_thread_limit(2, "openmp"):
        assert _openmp_threads() == {2}
        with parallel_policy.native_thread_limit(1, "openmp"):
            assert _openmp_threads() == {1}
        assert _openmp_threads() == {2}  # the outer budget is back once the inner left
    assert _openmp_threads() == {3}


def test_exception_in_inner_cap_unwinds_to_outer_then_original(raised_openmp):
    with parallel_policy.native_thread_limit(2, "openmp"):
        with pytest.raises(RuntimeError):
            with parallel_policy.native_thread_limit(1, "openmp"):
                raise RuntimeError("fit failed")
        assert _openmp_threads() == {2}
    assert _openmp_threads() == {3}
    assert parallel_policy._ACTIVE_LIMITS == {}


@pytest.fixture
def raised_blas(raised_openmp):
    from threadpoolctl import threadpool_limits

    if not _pool_threads("blas"):
        pytest.skip("no BLAS runtime visible to threadpoolctl")
    with threadpool_limits(limits=8, user_api="blas"):
        yield


def test_cross_api_overlap_from_two_threads_never_clobbers(raised_blas):
    """A caps BLAS, B caps OpenMP, A exits first: B's OpenMP cap must survive, and
    once both left BLAS is back to 8 (not leaked at 2) and OpenMP back to 3."""
    import threading

    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def thread_a():
        with parallel_policy.native_thread_limit(2, "blas"):
            a_in.set()
            b_in.wait(5)
            seen["openmp_while_both"] = _openmp_threads()
        a_out.set()

    def thread_b():
        a_in.wait(5)
        with parallel_policy.native_thread_limit(1, "openmp"):
            b_in.set()
            a_out.wait(5)
            seen["openmp_after_a_left"] = _openmp_threads()
            seen["blas_after_a_left"] = _pool_threads("blas")

    workers = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(10)
    assert seen["openmp_while_both"] == {1}
    assert seen["openmp_after_a_left"] == {1}
    assert seen["blas_after_a_left"] == {8}
    assert _pool_threads("blas") == {8}
    assert _openmp_threads() == {3}


def test_cap_below_one_is_rejected():
    with pytest.raises(ValueError):
        with parallel_policy.native_thread_limit(0, "blas"):
            pass


def test_one_class_cv_runs_under_openmp_cap(monkeypatch, raised_openmp):
    from spectral_predict import contamination

    seen = []
    real = contamination.build_one_class_model

    def spy(name, params):
        seen.append(_openmp_threads())
        return real(name, params)

    monkeypatch.setattr(contamination, "build_one_class_model", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 30))
    y_oc = np.r_[np.ones(30), -np.ones(10)].astype(int)
    contamination.run_one_class_cv(X, y_oc, "LOF", {"n_neighbors": 5}, n_folds=3)
    assert seen and all(s == {1} for s in seen)
    assert _openmp_threads() == {3}


def test_simca_cross_fit_null_is_capped():
    from spectral_predict.simca import MultiClassClassModel

    assert hasattr(MultiClassClassModel._cross_fit_null, "__wrapped__")


# ------------------------------------------------------------------- call sites


class _RecordingParallel:
    """Stand-in for joblib.Parallel: records the pool it was asked for, runs inline."""

    calls: list[tuple[int, str]] = []

    def __init__(self, n_jobs=None, backend=None, **_kw):
        type(self).calls.append((n_jobs, backend))

    def __call__(self, tasks):
        return [fn(*a, **kw) for fn, a, kw in tasks]


def _regression_data(n=40, p=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"{1000 + i}" for i in range(p)])
    y = pd.Series(X.iloc[:, :5].sum(axis=1) + rng.normal(size=n) * 0.1)
    return X, y


def test_grid_search_folds_are_capped_and_refit_params_unchanged(cores, not_frozen, monkeypatch):
    from spectral_predict import search

    seen_threads = []
    real_fold = search._run_single_fold

    def spy_fold(pipe, *args, **kwargs):
        seen_threads.append(estimator_threads(pipe))
        return real_fold(pipe, *args, **kwargs)

    _RecordingParallel.calls = []
    monkeypatch.setattr(search, "Parallel", _RecordingParallel)
    monkeypatch.setattr(search, "_run_single_fold", spy_fold)
    cores(4)

    X, y = _regression_data()
    df, _ = search.run_search(
        X,
        y,
        "regression",
        models_to_test=["RandomForest"],
        folds=5,
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        rf_n_trees_list=[10],
        rf_max_depth_list=[None],
    )
    assert _RecordingParallel.calls and all(c == (4, "loky") for c in _RecordingParallel.calls)
    assert seen_threads and all(t == {"model": 1} for t in seen_threads)
    params = str(df.iloc[0]["Params"])
    assert "'n_jobs': -1" in params  # the refit / captured params keep the model as built


def test_grid_search_frozen_uses_threading_pool(cores, monkeypatch):
    from spectral_predict import search

    _RecordingParallel.calls = []
    monkeypatch.setattr(search, "Parallel", _RecordingParallel)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    cores(4)
    X, y = _regression_data()
    search.run_search(
        X,
        y,
        "regression",
        models_to_test=["RandomForest"],
        folds=5,
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        rf_n_trees_list=[10],
        rf_max_depth_list=[None],
    )
    assert _RecordingParallel.calls and all(b == "threading" for _, b in _RecordingParallel.calls)


def test_bayesian_folds_are_capped(cores, not_frozen, monkeypatch):
    optuna = pytest.importorskip("optuna")
    from spectral_predict import unified_bayesian

    seen = []
    real = unified_bayesian.cross_val_predict_pooled

    def spy(model, X, y, cv, n_jobs=1, **kwargs):
        seen.append((n_jobs, estimator_threads(model)))
        return real(model, X, y, cv, n_jobs=1, **kwargs)

    monkeypatch.setattr(unified_bayesian, "cross_val_predict_pooled", spy)
    # Trials may pick small variable subsets; make every one large enough for a pool.
    monkeypatch.setattr(parallel_policy, "TINY_JOB_CELLS", 0)
    cores(4)
    X, y = _regression_data()
    wl = np.array([float(c) for c in X.columns])
    _, study = unified_bayesian.run_unified_bayesian(
        X=X.to_numpy(),
        y=y.to_numpy(),
        wavelengths=wl,
        model_name="RandomForest",
        task_type="regression",
        n_trials=2,
        cv_folds=4,
        random_state=42,
        verbose=False,
        enable_sqlite_persistence="never",
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert completed
    parallel_calls = [s for s in seen if s[0] > 1]
    assert parallel_calls, seen
    assert all(n == 4 and threads == {"model": 1} for n, threads in parallel_calls)


def test_ipls_interval_cv_is_serial(monkeypatch):
    from spectral_predict import variable_selection

    seen = []
    real = variable_selection.cross_val_score

    def spy(*args, n_jobs=None, **kwargs):
        seen.append(n_jobs)
        return real(*args, n_jobs=1, **kwargs)

    monkeypatch.setattr(variable_selection, "cross_val_score", spy)
    X, y = _regression_data(p=200)
    variable_selection.ipls_selection(X.to_numpy(), y.to_numpy(), n_intervals=5, cv_folds=3)
    assert seen and set(seen) == {1}


def test_ga_preprocessing_fitness_model_takes_thread_cap(monkeypatch):
    from spectral_predict import ga_preprocessing

    seen = []
    real = ga_preprocessing.cross_val_predict

    def spy(model, *args, **kwargs):
        seen.append(estimator_threads(model))
        return real(model, *args, **kwargs)

    monkeypatch.setattr(ga_preprocessing, "cross_val_predict", spy)
    X, y = _regression_data(p=120)
    ga_preprocessing.evaluate_fitness(
        np.array([0, 0], dtype=np.int32),
        X.to_numpy(),
        y.to_numpy(),
        cv_folds=3,
        model_config={"name": "RandomForest", "params": {"n_estimators": 5}},
        model_threads=1,
    )
    ga_preprocessing.evaluate_fitness(
        np.array([0, 0], dtype=np.int32),
        X.to_numpy(),
        y.to_numpy(),
        cv_folds=3,
        fitness_model="lightgbm",
        model_threads=2,
    )
    assert seen == [{"estimator": 1}, {"estimator": 2}]


@pytest.mark.parametrize("model_type", ["RandomForest", "XGBoost", "LightGBM", "CatBoost"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_nsga2_models_are_single_threaded(model_type, task):
    pytest.importorskip("pymoo")
    from spectral_predict.nsga2_search import _build_model

    model = _build_model(model_type, 1, task, 42)
    assert set(estimator_threads(model).values()) == {1}


def test_learning_curve_estimator_is_capped(cores, not_frozen, monkeypatch):
    import sklearn.model_selection as ms

    from spectral_predict.diagnostics import compute_learning_curve

    seen = []
    real = ms.learning_curve

    def spy(estimator, *args, n_jobs=None, **kwargs):
        seen.append((n_jobs, estimator_threads(estimator)))
        return real(estimator, *args, n_jobs=1, **kwargs)

    monkeypatch.setattr(ms, "learning_curve", spy)
    cores(4)
    X, y = _regression_data()
    compute_learning_curve(
        RandomForestRegressor(n_estimators=5, n_jobs=-1),
        X.to_numpy(),
        y.to_numpy(),
        cv=5,
        train_sizes=np.linspace(0.5, 1.0, 2),
    )
    assert seen == [(4, {"estimator": 1})]
