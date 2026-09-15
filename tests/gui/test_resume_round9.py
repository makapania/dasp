"""Round 9 fixes for PR #79 (Codex review of round 8, 8f4cf54).

1. A one-class Bayesian resume runs the models frozen at the click, never the
   one-class checkboxes as they are when the worker gets there.
2. A model-list / trial-count override frozen for a resume never leaks into the
   fresh run that replaces it.
3. A damaged saved-run record is reported and offered (move aside / leave it),
   never silently moved or replaced by the next fresh run.
4. If the worker thread can't start, the run the gate claimed is released.
5. A finished run whose record can't be unlinked still releases its claim.
6. A resume uses the run's own trial count, and shows analysis settings that
   differ from the run's (restore / delete and start fresh / decide later).
Plus end-to-end hand-off tests through the real ``_run_analysis``: gate, then
the thread it would start, then the worker.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui_module

from tests.gui.test_resume_data_mismatch_keeps_run import _FakeThread
from tests.gui.test_resume_round8 import paused_with_models  # noqa: F401 -- fixture
from tests.gui.test_resume_run_completion import (
    _StopSearch,
    _regression_data,
    _successful_results_row,
    worker_env,  # noqa: F401 -- fixture
)


def _recording_bayesian(fail_models=()):
    """A run_unified_bayesian stand-in recording (model_name, n_trials, n_rows)."""
    calls: list[tuple[str, int, int]] = []

    def fake(X, y, wavelengths, model_name, **kwargs):
        calls.append((model_name, kwargs.get("n_trials"), len(X)))
        if model_name in fail_models:
            raise RuntimeError("database is locked")
        return _successful_results_row(), None

    return fake, calls


@pytest.fixture
def fake_thread(monkeypatch):
    _FakeThread.created = []
    monkeypatch.setattr(gui_module.threading, "Thread", _FakeThread)
    return _FakeThread


def _click(gui_app):
    before = len(_FakeThread.created)
    gui_app._run_analysis()
    return _FakeThread.created[-1] if len(_FakeThread.created) > before else None


def _select_models(gui_app, names):
    for attr in ("use_pls", "use_ridge"):
        getattr(gui_app, attr).set(False)
    for name in names:
        getattr(gui_app, {"PLS": "use_pls", "Ridge": "use_ridge"}[name]).set(True)


# ---------------------------------------------------------------------------
# Item 1: one-class resume uses the frozen model list
# ---------------------------------------------------------------------------


def test_one_class_worker_runs_frozen_models_not_live_checkboxes(gui_app, worker_env, monkeypatch):
    X, _ = _regression_data()
    gui_app.X, gui_app.y = X, pd.Series(["a"] * 22 + ["b"] * 8)
    gui_app.task_type.set("one_class")
    for name, var in gui_app.one_class_model_checkboxes.items():
        var.set(name == "IsolationForest")  # changed after the click
    fake, calls = _recording_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PCA-SIMCA"], "quick", resolved_inlier_label="a")

    assert [c[0] for c in calls] == ["PCA-SIMCA"]


# ---------------------------------------------------------------------------
# Item 2: no stale override after a delete / start fresh
# ---------------------------------------------------------------------------


def test_mismatch_start_fresh_drops_resume_overrides(gui_app, paused_with_models):
    rs, meta, _ = paused_with_models
    gui_app.X, gui_app.y = _regression_data(99)  # not the run's data
    gui_app.n_unified_trials.set(3)
    with patch("tkinter.messagebox.askyesnocancel", return_value=True), \
         patch("tkinter.messagebox.askyesno", return_value=True):
        launch = gui_app._confirm_resume_before_launch(["Ridge"], "quick")

    assert launch is True
    assert gui_app._pending_bayesian_models is None
    fresh = rs.find_incomplete_run()
    assert fresh.run_id != meta.run_id and fresh.model_names == ["Ridge"]
    assert gui_app._pending_bayesian_n_trials == 3 == fresh.n_trials_per_model


# ---------------------------------------------------------------------------
# Item 3: damaged record
# ---------------------------------------------------------------------------


def _damage(rs):
    path = rs._sidecar_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ damaged", encoding="utf-8")
    return path


def test_gate_leaves_damaged_record_when_user_says_no(gui_app, worker_env):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    record = _damage(rs)
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert ask.call_args[0][0] == "Saved run record is damaged"
    assert record.read_text(encoding="utf-8") == "{ damaged"
    assert rs.get_active_run_id() is None


def test_gate_moves_damaged_record_aside_then_starts_fresh(gui_app, worker_env):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    record = _damage(rs)
    with patch("tkinter.messagebox.askyesno", return_value=True):
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is True
    kept = list(record.parent.glob("active_run.corrupt-*.json"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == "{ damaged"
    assert rs.find_incomplete_run().run_id == gui_app._pending_bayesian_run_id


def test_startup_offers_damaged_record_and_leaves_it_on_no(gui_app, worker_env):
    rs = worker_env
    record = _damage(rs)
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask, \
         patch("tkinter.messagebox.askyesnocancel") as resume_prompt:
        gui_app._check_for_incomplete_run()

    assert ask.call_args[0][0] == "Saved run record is damaged"
    assert not resume_prompt.called
    assert record.read_text(encoding="utf-8") == "{ damaged"


# ---------------------------------------------------------------------------
# Item 4: thread start failure releases the claim
# ---------------------------------------------------------------------------


def test_thread_start_failure_releases_claim(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    _select_models(gui_app, ["PLS"])

    class _NoThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

        def is_alive(self):
            return False

    monkeypatch.setattr(gui_module.threading, "Thread", _NoThread)
    with patch("tkinter.messagebox.showerror") as err:
        gui_app._run_analysis()

    assert err.call_args[0][0] == "Could not start analysis"
    assert rs.get_active_run_id() is None, "the claim did not leak"


# ---------------------------------------------------------------------------
# Item 5: mark_complete unlink failure
# ---------------------------------------------------------------------------


def test_mark_complete_failure_still_releases_claim(gui_app, worker_env, monkeypatch):
    rs = worker_env
    meta = rs.start_run(label="t", bayesian_persistence_mode="always")

    def locked():
        raise PermissionError("sidecar locked by antivirus")

    monkeypatch.setattr(rs, "mark_complete", locked)
    gui_app._complete_run_state_after_search(meta.run_id)

    assert rs.get_active_run_id() is None
    assert rs.find_incomplete_run().run_id == meta.run_id, "the record stays on disk"


# ---------------------------------------------------------------------------
# Item 6: trial count and settings
# ---------------------------------------------------------------------------


@pytest.fixture
def paused_with_settings(gui_app, tmp_path, monkeypatch, reimport_modules):
    """A Stopped run of PLS with 7 trials and folds=5, data (X, y)."""
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()
    names = ("optimization_method", "task_type", "n_unified_trials", "folds", "use_pls",
             "use_ridge", "bayesian_persistence_mode", "output_dir")
    saved = {n: getattr(gui_app, n).get() for n in names}
    saved_indices = gui_app.active_indices

    X, y = _regression_data(15)
    meta = rs.start_run(
        label="t", dataset_fingerprint=rs.fingerprint_dataset(X, y), model_names=["PLS"],
        n_trials_per_model=7, bayesian_persistence_mode="always",
        gui_settings={"folds": 5, "n_unified_trials": 7, "use_pls": True},
    )
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()

    def run_now(_ms, func=None, *args):
        if func is not None:
            func(*args)

    monkeypatch.setattr(gui_app.root, "after", run_now)
    gui_app.optimization_method.set("unified")
    gui_app.task_type.set("regression")
    gui_app.bayesian_persistence_mode.set("always")
    gui_app.output_dir.set(str(tmp_path / "out"))
    gui_app.active_indices = None
    _select_models(gui_app, ["PLS"])
    gui_app.folds.set(5)
    gui_app.n_unified_trials.set(2)
    gui_app.X, gui_app.y = X, y
    from spectral_predict.search_controller import SearchController
    gui_app.search_controller = SearchController()
    monkeypatch.chdir(tmp_path)

    yield rs, meta, (X, y)

    for n, v in saved.items():
        getattr(gui_app, n).set(v)
    gui_app.active_indices = saved_indices
    rs._reset_for_tests()


def test_resume_asks_about_trial_count_and_uses_the_runs_own(gui_app, paused_with_settings):
    rs, meta, _ = paused_with_settings
    with patch("tkinter.messagebox.askyesnocancel", return_value=True), \
         patch("tkinter.messagebox.askyesno", return_value=True) as ask:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is True
    assert ask.call_args[0][0] == "Resuming a different trial count"
    assert "run: 7; set now: 2" in ask.call_args[0][1]
    assert gui_app._pending_bayesian_n_trials == 7


def test_differing_settings_restore_runs_nothing(gui_app, paused_with_settings):
    rs, meta, _ = paused_with_settings
    gui_app.folds.set(3)
    with patch("tkinter.messagebox.askyesnocancel", side_effect=[True, True]) as ask:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert ask.call_args[0][0] == "Settings differ from the interrupted run"
    assert "folds: 5 in the run, 3 now" in ask.call_args[0][1]
    assert gui_app.folds.get() == 5
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_differing_settings_cancel_changes_nothing(gui_app, paused_with_settings):
    rs, meta, _ = paused_with_settings
    gui_app.folds.set(3)
    with patch("tkinter.messagebox.askyesnocancel", side_effect=[True, None]):
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert gui_app.folds.get() == 3
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_differing_settings_no_deletes_and_starts_fresh(gui_app, paused_with_settings):
    rs, meta, _ = paused_with_settings
    gui_app.folds.set(3)
    with patch("tkinter.messagebox.askyesnocancel", side_effect=[True, False]):
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is True
    assert not Path(meta.storage_path).exists()
    fresh = rs.find_incomplete_run()
    assert fresh.run_id != meta.run_id and fresh.gui_settings["folds"] == 3
    assert gui_app._pending_bayesian_n_trials == 2


def test_persistence_radio_is_not_a_settings_difference(gui_app, paused_with_settings):
    rs, meta, _ = paused_with_settings
    meta_settings = dict(meta.gui_settings)
    assert "bayesian_persistence_mode" not in meta_settings
    from spectral_predict.run_gui_settings import diff_gui_settings

    assert diff_gui_settings(
        {"bayesian_persistence_mode": "always", "folds": 5},
        {"bayesian_persistence_mode": "auto", "folds": 5},
    ) == []


# ---------------------------------------------------------------------------
# End to end through _run_analysis: gate -> thread -> worker
# ---------------------------------------------------------------------------


def test_e2e_resume_runs_saved_trials_on_click_time_data(
    gui_app, paused_with_settings, fake_thread, monkeypatch
):
    rs, meta, (X, y) = paused_with_settings
    fake, calls = _recording_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    with patch("tkinter.messagebox.askyesnocancel", return_value=True), \
         patch("tkinter.messagebox.askyesno", return_value=True), \
         patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        worker = _click(gui_app)
        assert worker is not None
        gui_app.X, gui_app.y = _regression_data(42)[0].iloc[:10], _regression_data(42)[1][:10]
        worker.target(*worker.args, **worker.kwargs)

    assert calls == [("PLS", 7, len(X))], "saved trial count, data bound at the click"
    assert rs.find_incomplete_run() is None, "finished resume released its record"


def test_e2e_one_failing_model_keeps_run_resumable(gui_app, worker_env, fake_thread, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    _select_models(gui_app, ["PLS", "Ridge"])
    fake, calls = _recording_bayesian(fail_models=("Ridge",))
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        worker = _click(gui_app)
        run_id = worker.kwargs["analysis_run_id"]
        worker.target(*worker.args, **worker.kwargs)

    assert [c[0] for c in calls] == ["PLS", "Ridge"]
    assert rs.find_incomplete_run().run_id == run_id
    assert rs.get_active_run_id() is None, "claim released, record kept"


def test_e2e_setup_exception_releases_claim(gui_app, worker_env, fake_thread, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    _select_models(gui_app, ["PLS"])

    def setup_fails(*a, **k):
        raise RuntimeError("setup failed")

    fake, calls = _recording_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        worker = _click(gui_app)
        run_id = worker.kwargs["analysis_run_id"]
        assert rs.get_active_run_id() == run_id, "the gate claimed the run"
        # Uncaught during worker setup, after the claim and before any search.
        monkeypatch.setattr(gui_app, "_get_imbalance_params", setup_fails)
        try:
            worker.target(*worker.args, **worker.kwargs)
        except RuntimeError:
            pass

    assert calls == [], "the search never started"
    assert rs.get_active_run_id() is None
