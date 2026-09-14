"""CatBoost must never write a ``catboost_info/`` training directory.

By default CatBoost creates ``catboost_info/`` in the current working directory on
every fit. When the cwd is unwritable (the installer puts the app under Program
Files) every CatBoost fit raised ``Can't create train working dir: catboost_info``,
and concurrent fits sharing the directory could race (seen in CI on Windows).

The blocked-cwd fixture reproduces the failure portably: a regular *file* named
``catboost_info`` in the cwd makes the directory impossible to create.

The fix must stay out of model identity: ``allow_writing_files`` is a
construction-time runtime kwarg and must not appear in captured result-row params.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from spectral_predict.models import (  # noqa: E402
    CATBOOST_RUNTIME_PARAMS,
    build_model,
    get_model,
    get_model_grids,
    strip_runtime_params,
)

TASKS = ["regression", "classification"]


@pytest.fixture()
def blocked_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir into a directory where CatBoost cannot create ``catboost_info/``."""
    (tmp_path / "catboost_info").write_text("not a directory", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def clean_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Chdir into an empty directory and assert no ``catboost_info/`` was left behind."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    assert not (tmp_path / "catboost_info").exists()


def _data(task: str, n_samples: int = 30, n_features: int = 20):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n_samples)
    if task == "classification":
        y = (y > np.median(y)).astype(int)
    return X, y


_SMALL = {"iterations": 5, "depth": 2, "learning_rate": 0.1}


@pytest.mark.parametrize("task", TASKS)
def test_get_model_fits_in_blocked_cwd(blocked_cwd, task) -> None:
    X, y = _data(task)
    model = get_model("CatBoost", task_type=task)
    model.set_params(iterations=5)
    model.fit(X, y)


@pytest.mark.parametrize("task", TASKS)
def test_build_model_fits_in_blocked_cwd(blocked_cwd, task) -> None:
    X, y = _data(task)
    build_model("CatBoost", dict(_SMALL), task_type=task).fit(X, y)


@pytest.mark.parametrize("task", TASKS)
def test_build_model_accepts_stored_runtime_param(blocked_cwd, task) -> None:
    """A params dict that already carries the runtime kwarg must not raise a duplicate."""
    X, y = _data(task)
    params = {**_SMALL, **CATBOOST_RUNTIME_PARAMS}
    build_model("CatBoost", params, task_type=task).fit(X, y)


@pytest.mark.parametrize("task", TASKS)
def test_grid_models_fit_in_blocked_cwd_and_params_dict_is_unchanged(blocked_cwd, task) -> None:
    X, y = _data(task)
    grids = get_model_grids(
        task,
        X.shape[1],
        tier="quick",
        enabled_models=["CatBoost"],
        catboost_iterations_list=[5],
        catboost_depths=[2],
    )
    configs = grids["CatBoost"]
    assert configs
    for model, params_dict in configs[:2]:
        assert "allow_writing_files" not in params_dict
        model.fit(X, y)


@pytest.mark.parametrize("task", TASKS)
def test_normal_fit_leaves_no_catboost_info(clean_cwd, task) -> None:
    X, y = _data(task)
    build_model("CatBoost", dict(_SMALL), task_type=task).fit(X, y)
    get_model("CatBoost", task_type=task).set_params(iterations=5).fit(X, y)


@pytest.mark.parametrize("task", TASKS)
def test_nsga2_builder_fits_in_blocked_cwd(blocked_cwd, task) -> None:
    from spectral_predict import nsga2_search

    X, y = _data(task)
    model = nsga2_search._build_model("CatBoost", 0, task, random_state=0)
    model.set_params(iterations=5)
    model.fit(X, y)


@pytest.mark.parametrize("task", TASKS)
def test_nsga2_decoded_params_exclude_runtime_param(task) -> None:
    from spectral_predict import nsga2_search

    n_wl = 20
    chromosome = np.zeros(13 + n_wl, dtype=int)  # 13 structural genes, then the mask
    chromosome[13:] = 1
    solution = nsga2_search.decode_solution(
        chromosome, n_wl, model_types=["CatBoost"], task_type=task
    )
    params = ast.literal_eval(solution["model_params"])
    assert params  # captured from get_model(...).get_params()
    assert "allow_writing_files" not in params


@pytest.mark.parametrize("task", TASKS)
def test_preprocessing_discovery_importance_in_blocked_cwd(blocked_cwd, task, capsys) -> None:
    from spectral_predict.preprocessing_discovery import compute_importance

    X, y = _data(task)
    importance = compute_importance(X, y, method="model_specific", model_name="CatBoost",
                                    task_type=task)
    assert importance.shape == (X.shape[1],)
    # A failed CatBoost fit is swallowed and silently replaced by LightGBM importance.
    assert "Tree importance failed" not in capsys.readouterr().out


@pytest.mark.parametrize("task", TASKS)
def test_diagnostics_validation_curve_in_blocked_cwd(blocked_cwd, task) -> None:
    from sklearn.model_selection import KFold, StratifiedKFold

    from spectral_predict.diagnostics import compute_ensemble_validation_curve

    X, y = _data(task)
    cv = StratifiedKFold(3) if task == "classification" else KFold(3)
    result = compute_ensemble_validation_curve(
        "CatBoost", X, y, {"iterations": 10, "depth": 2}, cv, task=task, n_points=2
    )
    assert np.all(np.isfinite(result["cv_scores"]))


def test_strip_runtime_params_handles_pipeline_prefixes() -> None:
    params = {"depth": 4, "allow_writing_files": False, "model__allow_writing_files": False,
              "model__depth": 4}
    assert strip_runtime_params(params) == {"depth": 4, "model__depth": 4}


def test_run_search_catboost_in_blocked_cwd(blocked_cwd) -> None:
    from spectral_predict.search import run_search

    X, y = _data("regression", n_samples=30, n_features=20)
    X_df = pd.DataFrame(X, columns=np.linspace(1000.0, 1100.0, X.shape[1]))
    df_ranked, _ = run_search(
        X_df,
        pd.Series(y),
        "regression",
        folds=3,
        tier="quick",
        models_to_test=["CatBoost"],
        enabled_models=["CatBoost"],  # models_to_test only filters the tier's grid
        preprocessing_methods={"raw": True},
        catboost_iterations_list=[5],
        catboost_depths=[2],
        catboost_learning_rates=[0.1],
        catboost_l2_leaf_reg_list=[3.0],
        catboost_border_count_list=[32],
        catboost_bagging_temperature_list=[1.0],
        catboost_random_strength_list=[1.0],
        enable_variable_subsets=False,
        enable_region_subsets=False,
    )
    assert len(df_ranked) > 0
    assert (df_ranked["Model"] == "CatBoost").all()
    assert np.isfinite(df_ranked["RMSE"].astype(float)).all()
    for params_str in df_ranked["Params"]:
        params = ast.literal_eval(params_str) if isinstance(params_str, str) else params_str
        assert not any(k.endswith("allow_writing_files") for k in params)


def test_bayesian_catboost_in_blocked_cwd_keeps_params_clean(blocked_cwd) -> None:
    import optuna

    from spectral_predict.unified_bayesian import run_unified_bayesian

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = _data("regression", n_samples=30, n_features=20)
    wl = np.linspace(1000.0, 1100.0, X.shape[1])
    _, study = run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="CatBoost", task_type="regression", n_trials=1,
        cv_folds=3, random_state=42, verbose=False, enable_sqlite_persistence="never",
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert completed, [t.user_attrs.get("error") for t in study.trials]
    for trial in completed:
        assert np.isfinite(trial.value) and trial.value < 1e9  # 1e10 is the failed-fit penalty
        assert "allow_writing_files" not in trial.user_attrs.get("model_params", "")


@pytest.mark.parametrize("task", TASKS)
def test_exported_script_disables_catboost_writing_files(task) -> None:
    from spectral_predict.code_generator import CodeGenerator

    config = {
        "model_name": "CatBoost",
        "preprocessing": "raw",
        "task_type": task,
        "params": {"iterations": 50, "depth": 4, "learning_rate": 0.1},
    }
    script = CodeGenerator(config).generate_script()
    assert "'allow_writing_files': False" in script
