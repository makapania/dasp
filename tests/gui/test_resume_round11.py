"""Round 11 fixes for PR #79 (Codex review of round 10, dbb64ce).

1. The Bayesian branches read their settings from the snapshot taken at the
   click, so a change during a multi-model run can't give later models a
   different study than the one approved.
2. A delete that removed the store but not the record releases the resume, so
   a later click can't "resume" a run with no saved trials.
3. A store path that can't be resolved (OSError) keeps the record for a retry.
4. After a damaged record is moved aside, the claim is released even if the
   re-read fails.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import spectral_predict_gui_optimized as gui_module

from tests.gui.test_resume_data_mismatch_keeps_run import (  # noqa: F401 -- fixtures
    _FakeThread,
    _dataset,
    resumed,
)
from tests.gui.test_resume_run_completion import (  # noqa: F401 -- fixtures
    _regression_data,
    _successful_results_row,
    worker_env,
)
from tests.gui.test_resume_round9 import _select_models, fake_thread  # noqa: F401


def test_settings_changed_mid_run_do_not_reach_later_models(
    gui_app, worker_env, fake_thread, monkeypatch
):
    gui_app.X, gui_app.y = _regression_data()
    _select_models(gui_app, ["PLS", "Ridge"])
    gui_app.bayes_enable_autoscale.set(False)
    seen = []

    def fake(X, y, wavelengths, model_name, **kwargs):
        seen.append((model_name, kwargs["enable_autoscale"], kwargs["cv_folds"]))
        gui_app.bayes_enable_autoscale.set(True)  # user changes it while PLS runs
        gui_app.folds.set(7)
        return _successful_results_row(), None

    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)
    try:
        with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
            gui_app._run_analysis()
            assert _FakeThread.created, "the gate launched"
            worker = _FakeThread.created[-1]
            worker.target(*worker.args, **worker.kwargs)
    finally:
        gui_app.bayes_enable_autoscale.set(True)

    assert seen == [("PLS", False, 3), ("Ridge", False, 3)]


def test_store_deleted_but_record_locked_releases_resume(gui_app, resumed, monkeypatch):
    rs, meta, store, started, (X, y) = resumed
    record = rs._sidecar_path()
    real_unlink = Path.unlink

    def record_locked(self, *a, **k):
        if self == record:
            raise PermissionError("record locked")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", record_locked)
    gui_app.X, gui_app.y = _dataset(2)  # mismatch -> start fresh
    with patch("tkinter.messagebox.askyesno", return_value=True), \
         patch("tkinter.messagebox.showerror") as err:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False and err.called
    assert not store.exists()
    assert not rs.is_resuming(), "a run with no store is no longer being resumed"

    monkeypatch.setattr(Path, "unlink", real_unlink)
    gui_app.X, gui_app.y = X, y  # even the matching data can't resume it now
    with patch("tkinter.messagebox.askyesnocancel") as prompt:
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not prompt.called
    assert gui_app._pending_bayesian_run_id != meta.run_id


def test_unresolvable_store_path_keeps_record(worker_env, monkeypatch):
    rs = worker_env
    meta = rs.start_run(label="x", model_names=["m"], bayesian_persistence_mode="always")
    Path(meta.storage_path).touch()
    rs._reset_for_tests()
    real_resolve = Path.resolve

    def flaky_resolve(self, *a, **k):
        if str(self).endswith(".sqlite3"):
            raise OSError("drive unavailable")
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)
    result = rs.discard_incomplete_run(meta.run_id)
    assert not result.fully_succeeded and not result.sidecar_deleted
    monkeypatch.setattr(Path, "resolve", real_resolve)
    assert rs.discard_incomplete_run(meta.run_id).fully_succeeded


def test_reread_failure_after_move_aside_still_releases_claim(gui_app, resumed, monkeypatch):
    rs, meta, store, started, (X, y) = resumed
    rs._sidecar_path().write_text("{ damaged", encoding="utf-8")
    real_find = rs.find_incomplete_run
    calls = []

    def find_then_fail():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("share went away")
        return real_find()

    monkeypatch.setattr(rs, "find_incomplete_run", find_then_fail)
    gui_app.X, gui_app.y = X, y
    with patch("tkinter.messagebox.askyesno", return_value=True):
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert not rs.is_resuming()

    monkeypatch.setattr(rs, "find_incomplete_run", real_find)
    assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert rs.find_incomplete_run().run_id == gui_app._pending_bayesian_run_id
