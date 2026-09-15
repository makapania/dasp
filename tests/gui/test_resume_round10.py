"""Round 10 fixes for PR #79 (Codex review of round 9, 4c028c5).

1. The worker dispatches on the optimization method and task type read at the
   click, so switching Grid to Bayesian while it starts can't reach a resumed
   run's storage without the gate's checks.
2. Bayesian autoscale and the imbalance parameters are captured and compared,
   like every other input of the Bayesian study name.
3. A delete whose store is locked keeps the record, so retrying after the lock
   clears succeeds instead of failing forever.
4. A record that becomes damaged after a resume was claimed can be moved aside
   from Run Analysis.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import spectral_predict_gui_optimized as gui_module

from tests.gui.test_resume_data_mismatch_keeps_run import (  # noqa: F401 -- fixtures
    _FakeThread,
    _dataset,
    resumed,
)
from tests.gui.test_resume_round9 import (  # noqa: F401 -- fixtures
    _recording_bayesian,
    paused_with_settings,
)
from tests.gui.test_resume_run_completion import _StopSearch


def test_worker_dispatches_on_mode_frozen_at_click(gui_app, resumed, monkeypatch):
    rs, meta, store, started, _ = resumed
    gui_app.optimization_method.set("grid")
    gui_app.bayesian_persistence_mode.set("always")
    fake, calls = _recording_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    grid_calls = []

    def grid_search(*a, **k):
        grid_calls.append(1)
        raise _StopSearch

    monkeypatch.setattr("spectral_predict.search.run_search", grid_search)
    gui_app.X, gui_app.y = _dataset(2)  # not the resumed run's data
    try:
        with patch("tkinter.messagebox.askyesno") as ask, \
             patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
            before = len(_FakeThread.created)
            gui_app._run_analysis()
            worker = _FakeThread.created[-1]
            assert len(_FakeThread.created) == before + 1 and not ask.called
            gui_app.optimization_method.set("unified")  # switched while starting
            try:
                worker.target(*worker.args, **worker.kwargs)
            except _StopSearch:
                pass
    finally:
        gui_app.optimization_method.set("unified")

    assert calls == [], "no Bayesian search on unchecked data"
    assert grid_calls == [1]
    assert rs.is_resuming() and store.exists()


def test_bayes_autoscale_and_imbalance_params_are_compared(gui_app):
    from spectral_predict.run_gui_settings import capture_gui_settings, diff_gui_settings

    current = capture_gui_settings(gui_app)
    for key in ("bayes_enable_autoscale", "k_neighbors", "n_bins", "boost_factor"):
        assert key in current
    saved = dict(current, bayes_enable_autoscale=not current["bayes_enable_autoscale"])
    assert [d[0] for d in diff_gui_settings(saved, current)] == ["bayes_enable_autoscale"]


def test_locked_store_delete_can_be_retried(gui_app, paused_with_settings, monkeypatch):
    rs, meta, _ = paused_with_settings
    gui_app.folds.set(3)
    store = Path(meta.storage_path)
    real_unlink = Path.unlink

    def store_locked(self, *a, **k):
        if self == store:
            raise PermissionError("sharing violation")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", store_locked)
    with patch("tkinter.messagebox.askyesnocancel", side_effect=[True, False]), \
         patch("tkinter.messagebox.showerror") as err:
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert err.called
    assert rs.is_resuming() and rs.find_incomplete_run().run_id == meta.run_id

    monkeypatch.setattr(Path, "unlink", real_unlink)  # the lock clears
    with patch("tkinter.messagebox.askyesnocancel", return_value=False):
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not store.exists()
    assert rs.find_incomplete_run().run_id != meta.run_id


@pytest.mark.parametrize("move", [True, False])
def test_record_damaged_during_resume_can_be_moved_aside(gui_app, resumed, move):
    rs, meta, store, started, (X, y) = resumed
    record = rs._sidecar_path()
    record.write_text("{ damaged", encoding="utf-8")
    gui_app.X, gui_app.y = X, y
    with patch("tkinter.messagebox.askyesno", return_value=move) as ask:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert ask.call_args_list[0][0][0] == "Saved run record is damaged"
    assert launch is move
    if move:
        assert not rs.is_resuming()
        assert list(record.parent.glob("active_run.corrupt-*.json"))
        assert rs.find_incomplete_run().run_id == gui_app._pending_bayesian_run_id
    else:
        assert rs.is_resuming()
        assert record.read_text(encoding="utf-8") == "{ damaged"
