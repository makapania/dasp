"""T-51 PR D: the Bayesian extra-axes card, its launch wiring and resume behaviour.

Settings-module and advisory tests without Tk are in ``tests/test_t51_pr_d_settings.py``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui_module
from spectral_predict.run_gui_settings import (
    EXTRA_AXES_VAR_PREFIX,
    N_STARTUP_TRIALS_SETTING,
    extra_axes_settings,
)
from spectral_predict.search_spaces import BUNDLES, ExtraAxesConfigError

from tests.gui.test_resume_data_mismatch_keeps_run import _FakeThread  # noqa: F401
from tests.gui.test_resume_round9 import fake_thread, paused_with_settings  # noqa: F401
from tests.gui.test_resume_run_completion import (  # noqa: F401 -- fixtures
    _regression_data,
    _report_raises,
    _successful_results_row,
    worker_env,
)


def _axis(bundle_id: str) -> str:
    return f"{EXTRA_AXES_VAR_PREFIX}{bundle_id}"


@pytest.fixture
def models(gui_app):
    """Select supervised models by name; restores every model checkbox after."""
    names = gui_app._STANDARD_MODEL_VARS
    saved = {var: getattr(gui_app, var).get() for var in names.values()}

    def select(*wanted):
        for model, var in names.items():
            getattr(gui_app, var).set(model in wanted)

    yield select
    for var, value in saved.items():
        getattr(gui_app, var).set(value)


def _recording(fail=None):
    calls = []

    def fake(X, y, wavelengths, model_name, **kwargs):
        calls.append(
            (model_name, kwargs.get("enabled_extra_axes"), kwargs.get("n_startup_trials"))
        )
        if fail is not None:
            fail(model_name)
        return _successful_results_row(), None

    return fake, calls


def _click_and_run(gui_app):
    """Launch through the real gate and run the worker it would have started."""
    before = len(_FakeThread.created)
    with patch("tkinter.messagebox.showerror") as err, patch("tkinter.messagebox.showwarning"):
        gui_app._run_analysis()
        launched = len(_FakeThread.created) > before
        if launched:
            worker = _FakeThread.created[-1]
            worker.target(*worker.args, **worker.kwargs)
    return launched, err


# --- registry coverage --------------------------------------------------------------


def test_every_bundle_has_a_var_a_checkbox_and_both_settings_lists(gui_app):
    for name in extra_axes_settings():
        assert hasattr(gui_app, name), f"{name} not created at construction"
        assert name in gui_module.BAYESIAN_REQUIRED_SETTINGS
    assert set(gui_app._extra_axes_checkbuttons) == set(BUNDLES)


# --- wiring (tests 1-5) -------------------------------------------------------------


def test_ticked_bundles_reach_the_supervised_call(gui_app, worker_env, fake_thread, models, monkeypatch):
    gui_app.X, gui_app.y = _regression_data()
    models("PLS", "XGBoost")
    getattr(gui_app, _axis("xgb_sampling")).set(True)
    getattr(gui_app, _axis("lof_metric")).set(True)  # one-class only: not sent
    getattr(gui_app, N_STARTUP_TRIALS_SETTING).set(" 30")
    fake, calls = _recording()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    launched, _ = _click_and_run(gui_app)

    assert launched
    assert calls == [("PLS", ("xgb_sampling",), 30), ("XGBoost", ("xgb_sampling",), 30)]


def test_nothing_ticked_sends_defaults(gui_app, worker_env, fake_thread, models, monkeypatch):
    gui_app.X, gui_app.y = _regression_data()
    models("PLS")
    fake, calls = _recording()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    _click_and_run(gui_app)

    assert calls == [("PLS", (), None)]


def test_auto_task_filters_by_the_task_inferred_from_frozen_data(
    gui_app, worker_env, fake_thread, models, monkeypatch
):
    gui_app.X, gui_app.y = _regression_data()
    gui_app.task_type.set("auto")
    models("XGBoost")
    getattr(gui_app, _axis("xgb_sampling")).set(True)
    getattr(gui_app, _axis("if_max_samples")).set(True)
    fake, calls = _recording()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    _click_and_run(gui_app)

    assert calls == [("XGBoost", ("xgb_sampling",), None)]


def test_one_class_call_gets_one_class_bundles(gui_app, worker_env, fake_thread, monkeypatch):
    X, _ = _regression_data()
    gui_app.X, gui_app.y = X, pd.Series(["a"] * 22 + ["b"] * 8)
    gui_app.task_type.set("one_class")
    getattr(gui_app, _axis("if_max_samples")).set(True)
    getattr(gui_app, _axis("xgb_sampling")).set(True)  # supervised only: not sent
    fake, calls = _recording()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["IsolationForest"], "quick", resolved_inlier_label="a")

    assert calls == [("IsolationForest", ("if_max_samples",), None)]


def test_bundles_and_startup_are_frozen_at_the_click(
    gui_app, worker_env, fake_thread, models, monkeypatch
):
    gui_app.X, gui_app.y = _regression_data()
    models("PLS", "XGBoost")
    getattr(gui_app, _axis("xgb_sampling")).set(True)
    getattr(gui_app, N_STARTUP_TRIALS_SETTING).set("25")

    def change_mid_run(model_name):
        getattr(gui_app, _axis("xgb_sampling")).set(False)
        getattr(gui_app, _axis("xgb_child")).set(True)
        getattr(gui_app, N_STARTUP_TRIALS_SETTING).set("40")

    fake, calls = _recording(fail=change_mid_run)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    _click_and_run(gui_app)

    assert calls == [("PLS", ("xgb_sampling",), 25), ("XGBoost", ("xgb_sampling",), 25)]


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "20.5"])
def test_bad_startup_value_blocks_a_bayesian_launch(gui_app, worker_env, fake_thread, models, bad):
    gui_app.X, gui_app.y = _regression_data()
    models("PLS")
    getattr(gui_app, N_STARTUP_TRIALS_SETTING).set(bad)

    launched, err = _click_and_run(gui_app)

    assert not launched
    assert err.called and "Startup trials" in err.call_args[0][1]


def test_grid_launch_ignores_the_startup_box(gui_app, worker_env, fake_thread, models):
    gui_app.X, gui_app.y = _regression_data()
    models("PLS")
    gui_app.optimization_method.set("grid")
    getattr(gui_app, N_STARTUP_TRIALS_SETTING).set("abc")
    before = len(_FakeThread.created)
    with patch("tkinter.messagebox.showerror") as err:
        gui_app._run_analysis()
    assert len(_FakeThread.created) == before + 1, "grid search launched"
    assert not err.called


def test_bundle_config_error_counts_as_a_failed_model(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()

    def ridge_rejects(model_name):
        if model_name == "Ridge":
            raise ExtraAxesConfigError("Bundle 'x' axis collides with a base parameter")

    fake, calls = _recording(fail=ridge_rejects)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    _report_raises(monkeypatch)
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS", "Ridge"], "quick")

    assert [c[0] for c in calls] == ["PLS", "Ridge"]
    assert rs.find_incomplete_run() is not None, "a config error keeps the run resumable"


# --- resume (tests 6-9) -------------------------------------------------------------
# paused_with_settings saved {"folds": 5, "n_unified_trials": 7, "use_pls": True}:
# a pre-PR-D snapshot, with no extra-axes keys.

_SETTINGS_DIALOG = "Settings differ from the interrupted run"


def _dialog_titles(ask):
    return [call[0][0] for call in ask.call_args_list]


def test_old_run_with_default_controls_resumes_without_a_settings_dialog(
    gui_app, paused_with_settings
):
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask, \
         patch("tkinter.messagebox.askyesno", return_value=True):
        gui_app._confirm_resume_before_launch(["PLS"], "quick")
    assert _SETTINGS_DIALOG not in _dialog_titles(ask)


def test_old_run_with_a_ticked_bundle_shows_the_settings_dialog(gui_app, paused_with_settings):
    rs, meta, _ = paused_with_settings
    getattr(gui_app, _axis("rf_features")).set(True)
    with patch("tkinter.messagebox.askyesnocancel", side_effect=[True, None]) as ask:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert ask.call_args[0][0] == _SETTINGS_DIALOG
    assert _axis("rf_features") in ask.call_args[0][1]
    assert getattr(gui_app, _axis("rf_features")).get() is True, "Cancel changes nothing"
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_old_run_with_a_changed_startup_shows_the_settings_dialog(gui_app, paused_with_settings):
    getattr(gui_app, N_STARTUP_TRIALS_SETTING).set("40")
    with patch("tkinter.messagebox.askyesnocancel", side_effect=[True, True]) as ask:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert ask.call_args[0][0] == _SETTINGS_DIALOG
    assert getattr(gui_app, N_STARTUP_TRIALS_SETTING).get() == "", "Yes restores the old value"


@pytest.fixture
def crashed_old_run(gui_app, tmp_path, monkeypatch, reimport_modules):
    """A crashed run whose snapshot predates PR D, found at app startup."""
    import sys

    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()
    meta = rs.start_run(
        label="t", bayesian_persistence_mode="always", gui_settings={"folds": 5},
    )
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()
    saved_folds = gui_app.folds.get()
    yield rs
    gui_app.folds.set(saved_folds)
    rs._reset_for_tests()


def test_startup_resume_of_old_run_resets_new_controls(gui_app, crashed_old_run):
    getattr(gui_app, _axis("xgb_sampling")).set(True)
    getattr(gui_app, N_STARTUP_TRIALS_SETTING).set("40")
    with patch("tkinter.messagebox.askyesnocancel", return_value=True):
        gui_app._check_for_incomplete_run()

    assert getattr(gui_app, _axis("xgb_sampling")).get() is False
    assert getattr(gui_app, N_STARTUP_TRIALS_SETTING).get() == ""
    assert gui_app.folds.get() == 5


def test_startup_decide_later_leaves_controls_alone(gui_app, crashed_old_run):
    getattr(gui_app, _axis("xgb_sampling")).set(True)
    with patch("tkinter.messagebox.askyesnocancel", return_value=None):
        gui_app._check_for_incomplete_run()

    assert getattr(gui_app, _axis("xgb_sampling")).get() is True


# --- greying (test 12) --------------------------------------------------------------


def _disabled(gui_app, bundle_id):
    return gui_app._extra_axes_checkbuttons[bundle_id].instate(["disabled"])


def test_greying_follows_the_task_type(gui_app):
    saved = gui_app.task_type.get()
    try:
        gui_app.task_type.set("regression")
        gui_app._on_task_type_changed()
        assert not _disabled(gui_app, "xgb_sampling")
        assert _disabled(gui_app, "if_max_samples")
        assert _disabled(gui_app, "plsda_head"), "classification only"

        gui_app.task_type.set("one_class")
        gui_app._on_task_type_changed()
        assert _disabled(gui_app, "xgb_sampling")
        assert not _disabled(gui_app, "if_max_samples")

        gui_app.y = None  # auto with no data takes the early return
        gui_app.task_type.set("auto")
        gui_app._on_task_type_changed()
        assert not any(_disabled(gui_app, b) for b in BUNDLES), "unknown task: all enabled"
    finally:
        gui_app.task_type.set(saved)
        gui_app._on_task_type_changed()


def test_advisory_caption_updates(gui_app, models):
    saved = gui_app.task_type.get()
    try:
        gui_app.task_type.set("regression")
        models("XGBoost")
        getattr(gui_app, _axis("xgb_sampling")).set(True)
        gui_app._refresh_extra_axes_advisory()
        text = gui_app._extra_axes_advisory.get()
        assert "Extra axes apply to XGBoost" in text
        assert "search dimensions (XGBoost)" in text
    finally:
        gui_app.task_type.set(saved)


# --- study identity (test 13) -------------------------------------------------------


def test_gui_launched_bundle_run_has_the_python_study_name(
    gui_app, worker_env, fake_thread, models, monkeypatch
):
    """A real (tiny) XGBoost run from the GUI gets the same study as the direct call."""
    pytest.importorskip("xgboost")
    from spectral_predict.unified_bayesian import run_unified_bayesian

    names, passed = [], []

    def real(*args, **kwargs):
        results, study = run_unified_bayesian(*args, **kwargs)
        names.append(study.study_name)
        passed.append((args, kwargs))
        return results, study

    gui_app.X, gui_app.y = _regression_data()
    models("XGBoost")
    gui_app.bayesian_persistence_mode.set("never")
    getattr(gui_app, _axis("xgb_sampling")).set(True)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", real)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    _click_and_run(gui_app)
    assert len(names) == 1
    args, kwargs = passed[0]
    assert kwargs["enabled_extra_axes"] == ("xgb_sampling",)

    # Same inputs as the GUI's call, from Python: the bundle id alone decides identity.
    direct_kwargs = {k: v for k, v in kwargs.items() if k not in ("progress_callback", "controller")}
    _, direct = run_unified_bayesian(*args, **direct_kwargs)
    _, plain = run_unified_bayesian(*args, **dict(direct_kwargs, enabled_extra_axes=()))
    assert names[0] == direct.study_name
    assert names[0] != plain.study_name, "the bundle gives the run its own study"
