"""When a Bayesian run's resume record is released, and when it must be kept.

Reviews of PR #79 (DeepSeek, Codex, GLM):
- Every Bayesian branch releases its record right after the search finishes, before
  CSV/report/ensemble I/O. This includes one-class, which used to return without
  releasing it, so the user was asked "Resume previous run?" on every launch.
- The record stays resumable when any model's search raised, or the user pressed
  Stop.
- The search runs on the data the worker verified, even if ``self.X``/``self.y``
  change while it sets up.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui_module
from spectral_predict.search_controller import SearchController

_VARS = (
    "optimization_method", "task_type", "n_unified_trials", "folds", "output_dir",
    "bayesian_persistence_mode", "use_pls", "use_ridge",
    "use_ocsvm", "use_isolation_forest", "use_elliptic_envelope", "use_lof", "use_pca_simca",
)


class _StopSearch(BaseException):
    """Escapes the worker's ``except Exception`` handlers."""


def _regression_data(seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.uniform(0, 10, 30))
    X = pd.DataFrame(
        np.outer(y, np.linspace(0.5, 1.5, 20)) + rng.normal(0, 0.2, (30, 20)),
        columns=[str(1000 + 10 * i) for i in range(20)],
    )
    return X, y


@pytest.fixture
def worker_env(gui_app, tmp_path, monkeypatch, reimport_modules):
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()

    def run_now(_ms, func=None, *args):
        if func is not None:
            func(*args)

    monkeypatch.setattr(gui_app.root, "after", run_now)
    saved = {name: getattr(gui_app, name).get() for name in _VARS}
    saved_indices = (gui_app.active_indices, gui_app.excluded_spectra)
    gui_app.optimization_method.set("unified")
    gui_app.task_type.set("regression")
    gui_app.n_unified_trials.set(2)
    gui_app.folds.set(3)
    gui_app.output_dir.set(str(tmp_path / "out"))
    gui_app.bayesian_persistence_mode.set("always")
    gui_app.active_indices = None
    gui_app.search_controller = SearchController()
    monkeypatch.chdir(tmp_path)  # reports/ is written relative to cwd

    yield rs

    for name, value in saved.items():
        getattr(gui_app, name).set(value)
    gui_app.active_indices, gui_app.excluded_spectra = saved_indices
    rs._reset_for_tests()


def _fake_bayesian(behaviour):
    """A run_unified_bayesian stand-in; ``behaviour(model_name, X, y)`` may raise."""
    calls = []

    def fake(X, y, wavelengths, model_name, **kwargs):
        calls.append((model_name, np.array(X, copy=True), np.array(y, copy=True)))
        behaviour(model_name, X, y)
        return pd.DataFrame(), None

    return fake, calls


def _report_raises(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("report disk full")

    monkeypatch.setattr("spectral_predict.report.write_markdown_report", fail)


def _run_worker(gui_app, models):
    try:
        gui_app._run_analysis_thread(models, "quick")
    except _StopSearch:
        pass


def test_one_class_bayesian_run_releases_record(gui_app, worker_env, monkeypatch):
    """A real one-class unified run under 'always' must not leave a resume prompt."""
    rs = worker_env
    X, _ = _regression_data()
    y = pd.Series(["a"] * 22 + ["b"] * 8)
    gui_app.X, gui_app.y = X, y
    gui_app.task_type.set("one_class")
    for name, var in gui_app.one_class_model_checkboxes.items():
        var.set(name == "PCA-SIMCA")

    logged: list[str] = []
    real_log = gui_app._log_progress
    monkeypatch.setattr(gui_app, "_log_progress", lambda m: (logged.append(m), real_log(m)))
    gui_app._run_analysis_thread(["PCA-SIMCA"], "quick", resolved_inlier_label="a")

    assert not any("Error in PCA-SIMCA" in m or "[X] Error" in m for m in logged), logged[-5:]
    assert rs.find_incomplete_run() is None, "finished one-class run left its sidecar"
    stores = list(rs._sidecar_path().parent.glob("*.sqlite3"))
    assert len(stores) == 1, "the 'always' store was created"
    import optuna

    url = f"sqlite:///{stores[0].as_posix()}"
    (name,) = optuna.study.get_all_study_names(storage=url)
    assert len(optuna.load_study(study_name=name, storage=url).trials) == 2
    with patch("tkinter.messagebox.askyesnocancel") as ask:
        gui_app._check_for_incomplete_run()
    assert not ask.called


def test_record_released_before_post_search_io_fails(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    fake, _ = _fake_bayesian(lambda *a: None)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    def report_fails(*args, **kwargs):
        assert rs.find_incomplete_run() is None, "released before the report step"
        raise OSError("report disk full")

    monkeypatch.setattr("spectral_predict.report.write_markdown_report", report_fails)
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS"], "quick")

    assert rs.find_incomplete_run() is None


def test_one_failing_model_keeps_run_resumable(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()

    def ridge_fails(model_name, X, y):
        if model_name == "Ridge":
            raise RuntimeError("database is locked")

    fake, calls = _fake_bayesian(ridge_fails)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    _report_raises(monkeypatch)  # same setup that releases the record when nothing fails
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS", "Ridge"], "quick")

    assert [c[0] for c in calls] == ["PLS", "Ridge"]
    assert rs.find_incomplete_run() is not None, "unfinished run must stay resumable"


def test_user_stop_keeps_run_resumable(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    fake, _ = _fake_bayesian(lambda *a: gui_app.search_controller.stop())
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    _report_raises(monkeypatch)
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS"], "quick")

    assert rs.find_incomplete_run() is not None


def test_search_uses_the_data_the_worker_verified(gui_app, worker_env, monkeypatch):
    """Codex review: changing self.X/self.y during worker setup must not reach the search."""
    rs = worker_env
    X_a, y_a = _regression_data(3)
    X_b, y_b = _regression_data(4)
    gui_app.X, gui_app.y = X_a, y_a

    real_start_run = rs.start_run

    def start_run_then_user_loads_other_data(*args, **kwargs):
        meta = real_start_run(*args, **kwargs)
        gui_app.X, gui_app.y = X_b, y_b  # the user acts while the worker sets up
        return meta

    monkeypatch.setattr(rs, "start_run", start_run_then_user_loads_other_data)

    def stop(*_a):
        raise _StopSearch

    fake, calls = _fake_bayesian(stop)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    _run_worker(gui_app, ["PLS"])

    _, X_used, y_used = calls[0]
    np.testing.assert_allclose(X_used, X_a.to_numpy())
    np.testing.assert_allclose(y_used, y_a.to_numpy())
    assert rs.find_incomplete_run().dataset_fingerprint == rs.fingerprint_dataset(X_a, y_a)
    gui_app.X, gui_app.y = X_a, y_a


def test_data_changed_between_click_and_worker_stops_a_resume(gui_app, worker_env, monkeypatch):
    """The main thread verified data A; the worker binds B and must stop, deleting nothing."""
    rs = worker_env
    X_a, y_a = _regression_data(3)
    meta = rs.start_run(label="t", dataset_fingerprint=rs.fingerprint_dataset(X_a, y_a),
                        bayesian_persistence_mode="always")
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()
    assert rs.resume_run(meta.run_id) is not None

    gui_app.X, gui_app.y = X_a, y_a
    assert gui_app._confirm_resume_before_launch() is True
    gui_app.X, gui_app.y = _regression_data(4)  # changed before the worker starts

    fake, calls = _fake_bayesian(lambda *a: None)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    gui_app._run_analysis_thread(["PLS"], "quick")

    assert not calls, "no search on unverified data"
    assert rs.is_resuming() and rs.find_incomplete_run().run_id == meta.run_id
    assert Path(meta.storage_path).exists()
    gui_app.X, gui_app.y = X_a, y_a
