"""Round-7 fixes for PR #79 (Codex + DeepSeek review, plus the user's binding
decision that a saved paused/failed/crashed Bayesian run must always be asked
about — offered to resume or delete — until it is resumed to completion or
deleted).

Covers:
1. Stop -> Run race: a worker must decide "was I stopped?" using its OWN
   captured controller, never ``self.search_controller`` (which a second
   Run Analysis click may have already replaced).
2. A second Run Analysis click cannot start while a previous worker thread
   is still alive.
3. A model whose search returns no usable (non-penalty) results counts as
   failed, same as a raised exception.
4. A paused (Stop) or failed run is not silently reused nor silently
   orphaned by the next Run Analysis click: the click asks Resume / Delete /
   Decide-later, matching the startup dialog.
5. HAS_UNIFIED_BAYESIAN False registers no run.
6. A Stop'd run is offered for resume at the next app launch too.
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

from tests.gui.test_resume_run_completion import (
    _fake_bayesian,
    _fake_bayesian_empty_results,
    _regression_data,
    worker_env,  # noqa: F401 -- re-exported fixture
)


def test_all_fits_failed_keeps_run_resumable(gui_app, worker_env, monkeypatch):
    """Codex review of #79 round 7: every trial for a model failing inside the
    objective returns an empty frame, not an exception. That must count as
    a failed model, same as a raised exception, so the run stays resumable.
    """
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    fake, calls = _fake_bayesian_empty_results(lambda *a: None)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS"], "quick")

    assert calls, "the model was actually attempted"
    assert rs.find_incomplete_run() is not None, (
        "an all-fits-failed model must keep the run resumable, "
        "not be treated as a clean finish"
    )


def test_classification_bayesian_success_releases_record(gui_app, worker_env, monkeypatch):
    """A classification Bayesian success must release the record too, not
    just regression (round 7 test list)."""
    rs = worker_env
    X, y = _regression_data()
    y = pd.Series(np.where(y > y.median(), "high", "low"), index=y.index)
    gui_app.X, gui_app.y = X, y
    gui_app.task_type.set("classification")
    fake, calls = _fake_bayesian(lambda *a: None)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS"], "quick")

    assert calls
    assert rs.find_incomplete_run() is None, "a clean classification run must release its record"


def test_has_unified_bayesian_false_registers_no_run(gui_app, worker_env, monkeypatch):
    """DeepSeek review of #79 round 7: ``_uses_bayesian_run_state`` must also
    check ``HAS_UNIFIED_BAYESIAN`` — otherwise a build where the module
    failed to import registers a run and then bails out before it can ever
    be completed, leaving a permanent phantom resume prompt."""
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    monkeypatch.setattr(gui_module, "HAS_UNIFIED_BAYESIAN", False)

    assert gui_app._uses_bayesian_run_state() is False
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS"], "quick")

    assert rs.find_incomplete_run() is None, "no run should have been registered at all"
    assert rs.get_active_run_id() is None


def test_stop_then_run_race_does_not_release_old_run(gui_app, worker_env, monkeypatch):
    """Codex P2 block: if the user presses Stop and clicks Run Analysis again
    before the old worker exits, ``self.search_controller`` is replaced with
    the NEW run's (unstopped) controller. The old worker must still decide
    "was I stopped?" from its OWN captured controller — not the live
    ``self.search_controller`` — or it wrongly releases the paused run.

    Driven deterministically and synchronously (Tk variables can't be read
    off the main thread without a running mainloop, so this can't use a
    real background worker thread): the fake search itself performs the
    "Stop, then a second click swaps in a fresh controller" side effect
    while the (single, real) worker call is still in flight, exactly as a
    second Run Analysis click would between trials.
    """
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    old_controller = gui_app.search_controller

    def race(model_name, X, y):
        # User presses Stop on the run in flight...
        old_controller.stop()
        # ...and clicks Run Analysis again before the old worker has
        # noticed. _run_analysis would create a brand-new controller here,
        # replacing self.search_controller (but NOT the old worker's own
        # captured `controller` argument).
        gui_app.search_controller = SearchController()

    fake, calls = _fake_bayesian(race)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(["PLS"], "quick", None, old_controller)

    assert calls
    assert rs.find_incomplete_run() is not None, (
        "the old run must stay resumable — it must not be released by "
        "reading the NEW controller's (unstopped) state"
    )


def test_second_run_analysis_refused_while_worker_alive(gui_app, monkeypatch):
    """Codex P2 block: a second Run Analysis click must not start a second
    worker while the previous one is still alive."""
    fake_thread = type("FakeAliveThread", (), {"is_alive": lambda self: True})()
    prior_thread = getattr(gui_app, "analysis_thread", None)
    gui_app.analysis_thread = fake_thread
    try:
        called = {"n": 0}
        monkeypatch.setattr(
            gui_module.threading, "Thread",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        with patch("tkinter.messagebox.showwarning") as warn:
            gui_app._run_analysis()

        assert called["n"] == 0, "no second worker thread may be constructed"
        assert warn.called
        assert gui_app.analysis_thread is fake_thread
    finally:
        # This fixture reuses one session-scoped app; a fake "always alive"
        # thread must not leak into every later test's Run Analysis click.
        gui_app.analysis_thread = prior_thread


# ---------------------------------------------------------------------------
# User decision: keep asking about a saved run (Resume/Delete/Decide-later)
# ---------------------------------------------------------------------------


@pytest.fixture
def paused_no_resume_flag(gui_app, tmp_path, monkeypatch, reimport_modules):
    """A run that was Stopped earlier this session: its sidecar/store are on
    disk, but (per the round-7 fix) run_state's in-process claim on it has
    already been released, so ``is_resuming()`` is False — exactly the state
    ``_complete_run_state_after_search`` leaves behind on Stop/failure.
    """
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()

    X, y = _regression_data(5)
    meta = rs.start_run(
        label="t", dataset_fingerprint=rs.fingerprint_dataset(X, y),
        bayesian_persistence_mode="always",
    )
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    # Simulate what _complete_run_state_after_search now does on Stop/failure:
    # the in-process claim is released, but the sidecar + store are left
    # exactly as they are. (Not calling clear_resume_state() itself here:
    # its real _cleanup_empty_sqlite would open this fake single-header
    # file, see "no such table: trials" i.e. zero trials, and delete it --
    # same reason test_resume_data_mismatch_keeps_run.py uses
    # _reset_for_tests() rather than a real trial-counting store.)
    rs._reset_for_tests()
    assert not rs.is_resuming()
    assert rs.find_incomplete_run() is not None

    def run_now(_ms, func=None, *args):
        if func is not None:
            func(*args)

    monkeypatch.setattr(gui_app.root, "after", run_now)
    saved = (gui_app.optimization_method.get(), gui_app.task_type.get())
    gui_app.optimization_method.set("unified")
    gui_app.task_type.set("regression")
    gui_app.search_controller = SearchController()  # a fresh one for this "click"

    yield rs, meta, (X, y)

    gui_app.optimization_method.set(saved[0])
    gui_app.task_type.set(saved[1])
    rs._reset_for_tests()


def test_click_run_with_matching_data_offers_resume_and_resumes(gui_app, paused_no_resume_flag):
    rs, meta, (X, y) = paused_no_resume_flag
    gui_app.X, gui_app.y = X, y
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
        launch = gui_app._confirm_resume_before_launch()

    assert ask.called and ask.call_args[0][0] == "Resume interrupted run?"
    assert launch is True
    assert rs.is_resuming() and rs.get_active_run_id() == meta.run_id
    assert Path(meta.storage_path).exists()


def test_click_run_with_different_data_offers_resume_then_mismatch_dialog(gui_app, paused_no_resume_flag):
    """Choosing "resume" on the pending-run prompt, then loading data that
    doesn't match, must fall through to the existing mismatch dialog rather
    than launching on unverified data."""
    rs, meta, (X, y) = paused_no_resume_flag
    other_X, other_y = _regression_data(6)
    gui_app.X, gui_app.y = other_X, other_y

    with patch("tkinter.messagebox.askyesnocancel", return_value=True), \
         patch("tkinter.messagebox.askyesno", return_value=False) as ask_mismatch:
        launch = gui_app._confirm_resume_before_launch()

    assert ask_mismatch.called
    assert ask_mismatch.call_args[0][0] == "Different data than the interrupted run"
    assert launch is False
    assert rs.is_resuming() and rs.find_incomplete_run().run_id == meta.run_id
    assert Path(meta.storage_path).exists(), "the saved run's store must survive a mismatch"


def test_click_run_delete_choice_removes_saved_run_and_starts_fresh(gui_app, paused_no_resume_flag):
    rs, meta, (X, y) = paused_no_resume_flag
    gui_app.X, gui_app.y = _regression_data(7)  # any data — deletion doesn't check it
    with patch("tkinter.messagebox.askyesnocancel", return_value=False) as ask:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert ask.called
    assert launch is True
    assert not rs.is_resuming()
    assert not Path(meta.storage_path).exists(), "the old SQLite file must be deleted"
    # Round 8: _confirm_resume_before_launch now registers the fresh run
    # itself (moved from the worker), so a NEW run replaces the deleted one
    # rather than leaving nothing registered at all.
    new_meta = rs.find_incomplete_run()
    assert new_meta is not None and new_meta.run_id != meta.run_id
    assert gui_app._pending_bayesian_run_id == new_meta.run_id


def test_click_run_cancel_choice_launches_nothing_and_keeps_saved_run(gui_app, paused_no_resume_flag):
    rs, meta, (X, y) = paused_no_resume_flag
    gui_app.X, gui_app.y = _regression_data(8)
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        launch = gui_app._confirm_resume_before_launch()

    assert ask.called
    assert launch is False
    assert not rs.is_resuming(), "cancel must not silently start resuming either"
    assert rs.find_incomplete_run().run_id == meta.run_id, "nothing was touched"
    assert Path(meta.storage_path).exists()


def test_a_new_different_data_run_cannot_silently_reuse_storage(gui_app, paused_no_resume_flag, monkeypatch):
    """End-to-end: with a pending-but-undecided saved run, clicking Run
    Analysis with different data and choosing to delete must run the new
    analysis on a BRAND NEW run id/storage, never the old one."""
    rs, meta, (X, y) = paused_no_resume_flag
    new_X, new_y = _regression_data(9)
    gui_app.X, gui_app.y = new_X, new_y

    fake, calls = _fake_bayesian(lambda *a: None)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    with patch("tkinter.messagebox.askyesnocancel", return_value=False), \
         patch("tkinter.messagebox.showerror"):
        assert gui_app._confirm_resume_before_launch() is True
        gui_app._run_analysis_thread(["PLS"], "quick")

    assert calls and np.allclose(calls[0][1], new_X.to_numpy())
    new_meta = rs.find_incomplete_run()
    # A clean run releases its own record; either way it must not be the old one.
    assert new_meta is None or new_meta.run_id != meta.run_id


def test_stopped_run_offered_for_resume_at_next_launch(gui_app, paused_no_resume_flag):
    """The startup check must find a Stop'd run's sidecar exactly like a
    crashed one — ``_complete_run_state_after_search`` releasing its
    in-process claim must not touch the on-disk sidecar."""
    rs, meta, (X, y) = paused_no_resume_flag
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        gui_app._check_for_incomplete_run()

    assert ask.called
    assert ask.call_args[0][0] == "Resume previous run?"
    assert rs.find_incomplete_run().run_id == meta.run_id, "cancel must keep asking next time"


def test_startup_yes_resumes(gui_app, paused_no_resume_flag):
    """Startup "Resume previous run?": Yes resumes it (the third of the three
    startup choices, alongside the Delete and Cancel tests below)."""
    rs, meta, (X, y) = paused_no_resume_flag
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
        gui_app._check_for_incomplete_run()

    assert ask.called and ask.call_args[0][0] == "Resume previous run?"
    assert rs.is_resuming() and rs.get_active_run_id() == meta.run_id
    assert Path(meta.storage_path).exists()


def test_startup_no_deletes_saved_run(gui_app, paused_no_resume_flag):
    """Startup "Resume previous run?": No permanently deletes the sidecar and
    the SQLite store — the gap the coordinator flagged: this choice existed
    in the dialog but had no GUI-level test of its own (only unit tests of
    ``run_state.discard_incomplete_run`` in isolation)."""
    rs, meta, (X, y) = paused_no_resume_flag
    with patch("tkinter.messagebox.askyesnocancel", return_value=False) as ask:
        gui_app._check_for_incomplete_run()

    assert ask.called and ask.call_args[0][0] == "Resume previous run?"
    assert rs.find_incomplete_run() is None, "the saved run must be deleted, not left dangling"
    assert not Path(meta.storage_path).exists()
    assert not rs.is_resuming()
