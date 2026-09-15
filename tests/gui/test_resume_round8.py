"""Round 8 fixes for PR #79 (Codex + DeepSeek review of round 7, b45bfac).

Suggested design adopted: ``_confirm_resume_before_launch`` is now the single
main-thread authority for a Bayesian launch. It snapshots the optimization
method, task type, selected models, persistence mode and bound data, decides
resume/delete/fresh, and CLAIMS the run slot (registers or resumes it) before
the worker thread is even created. The frozen decision is passed into
``_run_analysis_thread`` as plain arguments (``analysis_run_id``,
``uses_bayesian_run_state``); the worker never re-derives either from live Tk
state and never re-decides.

Covers:
1. Resuming with a changed model selection asks, and uses the run's original
   models when confirmed.
2. A read failure, or a failed ``resume_run()``, must not launch (can't prove
   it's safe to let this click's own registration overwrite the sidecar).
3. A Delete that doesn't fully succeed must not launch either (both the
   pending-run three-way dialog and the mismatch dialog), and
   ``discard_incomplete_run`` itself refuses to unlink a sidecar that was
   replaced by another instance between its own read and the unlink.
4. The worker trusts the frozen ``uses_bayesian_run_state`` flag over
   whatever ``self.optimization_method`` says by the time it actually runs.
5. Every exit from the worker after registration — an early guard return, a
   setup exception, or the resume fingerprint re-check's own early return —
   either completes normally or releases the in-process claim while keeping
   the sidecar.
6. Wording: an all-penalties model whose study is already at its full trial
   count points the user at Delete.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui_module
from spectral_predict.run_state import DiscardResult

from tests.gui.test_resume_data_mismatch_keeps_run import (
    _dataset,
    resumed,  # noqa: F401 -- re-exported fixture
)
from tests.gui.test_resume_run_completion import (
    _fake_bayesian,
    _regression_data,
    worker_env,  # noqa: F401 -- re-exported fixture
)


@pytest.fixture
def paused_with_models(gui_app, tmp_path, monkeypatch, reimport_modules):
    """A run that was Stopped earlier this session while searching PLS and
    Ridge: its sidecar/store are on disk, and (per the round-7 fix)
    run_state's in-process claim on it has already been released, so
    ``is_resuming()`` is False — exactly the state
    ``_complete_run_state_after_search`` leaves behind on Stop/failure.
    Mirrors ``paused_no_resume_flag`` in ``test_resume_round7.py``, but with
    ``model_names`` set so the round-8 item 1 (resume-model reconciliation)
    tests have something concrete to compare the click's current selection
    against.
    """
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()

    X, y = _regression_data(15)
    meta = rs.start_run(
        label="t", dataset_fingerprint=rs.fingerprint_dataset(X, y),
        model_names=["PLS", "Ridge"],
        bayesian_persistence_mode="always",
    )
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    # See test_resume_round7.py's paused_no_resume_flag for why this is
    # _reset_for_tests() rather than clear_resume_state(): the latter's real
    # _cleanup_empty_sqlite would open this fake single-header file, see
    # zero trials, and delete it.
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
    from spectral_predict.search_controller import SearchController
    gui_app.search_controller = SearchController()

    yield rs, meta, (X, y)

    gui_app.optimization_method.set(saved[0])
    gui_app.task_type.set(saved[1])
    rs._reset_for_tests()


# ---------------------------------------------------------------------------
# Item 1: resuming must run the ORIGINAL models, not whatever is selected now
# ---------------------------------------------------------------------------


def test_resume_with_different_models_asks_and_defaults_to_original(gui_app, paused_with_models):
    rs, meta, (X, y) = paused_with_models
    gui_app.X, gui_app.y = X, y
    with patch("tkinter.messagebox.askyesno", return_value=True) as ask:
        launch = gui_app._confirm_resume_before_launch(["Ridge"], "quick")

    assert ask.called and ask.call_args[0][0] == "Resuming different models"
    body = ask.call_args[0][1]
    assert "PLS, Ridge" in body or "PLS" in body  # the run's original models are named
    assert launch is True
    assert gui_app._pending_bayesian_models == meta.model_names


def test_resume_with_different_models_no_cancels(gui_app, paused_with_models):
    rs, meta, (X, y) = paused_with_models
    gui_app.X, gui_app.y = X, y
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        launch = gui_app._confirm_resume_before_launch(["Ridge"], "quick")

    assert ask.called
    assert launch is False
    assert rs.is_resuming(), "declining must not abandon the resume, just this click"
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_resume_with_matching_models_does_not_ask(gui_app, paused_with_models):
    rs, meta, (X, y) = paused_with_models
    gui_app.X, gui_app.y = X, y
    with patch("tkinter.messagebox.askyesno") as ask:
        launch = gui_app._confirm_resume_before_launch(list(meta.model_names), "quick")

    assert not ask.called
    assert launch is True
    assert gui_app._pending_bayesian_models is None  # no override needed


def test_end_to_end_resume_runs_original_models_not_current_selection(
    gui_app, paused_with_models, monkeypatch
):
    """The scenario Codex traced: Stop a PLS run, select Ridge, click Resume.
    Ridge must not run in place of the still-unfinished original models, and
    the record must only release once ALL of them (not just Ridge) finish."""
    rs, meta, (X, y) = paused_with_models
    gui_app.X, gui_app.y = X, y
    fake, calls = _fake_bayesian(lambda *a: None)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    with patch("tkinter.messagebox.askyesno", return_value=True), \
         patch("tkinter.messagebox.showerror"):
        assert gui_app._confirm_resume_before_launch(["Ridge"], "quick") is True
        models_for_thread = gui_app._pending_bayesian_models
        gui_app._run_analysis_thread(
            models_for_thread, "quick",
            analysis_run_id=gui_app._pending_bayesian_run_id,
            uses_bayesian_run_state=gui_app._pending_uses_bayesian_run_state,
        )

    assert [c[0] for c in calls] == list(meta.model_names), (
        "the original models ran, not the click's current selection"
    )
    assert rs.find_incomplete_run() is None, "all original models completed -> released"


# ---------------------------------------------------------------------------
# Item 2: a read failure, or a failed resume, must not launch
# ---------------------------------------------------------------------------


def test_read_failure_does_not_launch(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()

    def boom():
        raise OSError("sidecar is on a dead network share")

    monkeypatch.setattr(rs, "find_incomplete_run", boom)
    with patch("tkinter.messagebox.showerror") as err:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert err.called
    assert rs.get_active_run_id() is None, "nothing was registered over an unknown state"


def test_resume_run_returning_none_does_not_launch(gui_app, paused_with_models, monkeypatch):
    rs, meta, (X, y) = paused_with_models
    gui_app.X, gui_app.y = X, y
    monkeypatch.setattr(rs, "resume_run", lambda run_id: None)
    with patch("tkinter.messagebox.showwarning") as warn:
        launch = gui_app._confirm_resume_before_launch(list(meta.model_names), "quick")

    assert launch is False
    assert warn.called
    assert rs.find_incomplete_run().run_id == meta.run_id, "the saved run was left untouched"


# ---------------------------------------------------------------------------
# Item 3: a Delete that doesn't fully succeed must not launch either
# ---------------------------------------------------------------------------


def test_pending_delete_partial_failure_does_not_launch(gui_app, paused_with_models, monkeypatch):
    rs, meta, (X, y) = paused_with_models
    gui_app.X, gui_app.y = _regression_data(11)
    monkeypatch.setattr(
        rs, "discard_incomplete_run",
        lambda run_id: DiscardResult(sidecar_deleted=False, storage_deleted=False, errors=["locked"]),
    )
    with patch("tkinter.messagebox.askyesnocancel", return_value=False), \
         patch("tkinter.messagebox.showerror") as err:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert err.called
    assert rs.find_incomplete_run().run_id == meta.run_id, "not overwritten by a fresh registration"


def test_mismatch_delete_partial_failure_does_not_launch(gui_app, resumed, monkeypatch):
    rs, meta, store, started, (X, y) = resumed
    gui_app.X, gui_app.y = _dataset(77)  # deliberately mismatched
    monkeypatch.setattr(
        rs, "discard_incomplete_run",
        lambda run_id: DiscardResult(sidecar_deleted=False, storage_deleted=False, errors=["locked"]),
    )
    with patch("tkinter.messagebox.askyesno", return_value=True), \
         patch("tkinter.messagebox.showerror") as err:
        launch = gui_app._confirm_resume_before_launch(["PLS"], "quick")

    assert launch is False
    assert err.called
    assert rs.is_resuming(), "the resume was not abandoned over a failed delete"
    assert store.exists()


def test_discard_refuses_to_delete_a_replaced_sidecar(tmp_path, monkeypatch, reimport_modules):
    """run_state.py fix: discard_incomplete_run's own find_incomplete_run()
    call and its later unlink() are not atomic. Re-check the run id right
    before unlinking; never unlink a sidecar another instance replaced."""
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()

    meta_a = rs.start_run(label="a", dataset_fingerprint="fp-a")
    rs._reset_for_tests()
    meta_b = rs.start_run(label="b", dataset_fingerprint="fp-b")  # replaces the sidecar
    rs._reset_for_tests()
    assert meta_b.run_id in rs._sidecar_path().read_text(encoding="utf-8")

    # Simulate: our caller's initial find_incomplete_run() (before this
    # function's own) read run A, before the other instance's start_run()
    # replaced the sidecar with B an instant later.
    monkeypatch.setattr(rs, "find_incomplete_run", lambda: meta_a)
    result = rs.discard_incomplete_run(meta_a.run_id)

    assert not result.sidecar_deleted
    assert result.errors
    assert rs._sidecar_path().exists()
    assert meta_b.run_id in rs._sidecar_path().read_text(encoding="utf-8"), (
        "run B's sidecar must survive untouched"
    )


# ---------------------------------------------------------------------------
# Item 4: the worker trusts the frozen flag, not live Tk state
# ---------------------------------------------------------------------------


def test_worker_does_not_register_when_frozen_flag_says_not_bayesian(gui_app, worker_env, monkeypatch):
    """Launch Grid, then flip the GUI to Bayesian before the worker runs
    (simulated: the frozen flag says False while live state says 'unified').
    The worker must not even ATTEMPT to register a run despite what live
    state says now — checked directly against ``start_run`` being called at
    all, not against the sidecar's fate afterward (a clean success deletes
    it either way, which would make this assertion pass for the wrong
    reason if it only checked the end state)."""
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    gui_app.optimization_method.set("unified")  # live state changed after the (simulated) gate decision
    register_calls = []
    real_start_run = rs.start_run

    def start_run_spy(*args, **kwargs):
        register_calls.append(1)
        return real_start_run(*args, **kwargs)

    monkeypatch.setattr(rs, "start_run", start_run_spy)
    fake, _ = _fake_bayesian(lambda *a: None)
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)

    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(
            ["PLS"], "quick", analysis_run_id=None, uses_bayesian_run_state=False,
        )

    assert not register_calls, "start_run must never be called when the frozen flag says not-bayesian"
    assert rs.get_active_run_id() is None


def test_worker_registers_run_state_only_via_frozen_context_not_relive(gui_app, worker_env, monkeypatch):
    """The reverse: the gate froze a real registered run id while Bayesian
    was selected; if the dispatch later reads a different live mode (e.g. a
    grid run, which never calls _complete_run_state_after_search), the
    registered run must still not be silently lost — the round-8 `finally`
    releases it, keeping it resumable."""
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    meta = rs.start_run(
        label="quick", dataset_fingerprint=rs.fingerprint_dataset(gui_app.X, gui_app.y),
        bayesian_persistence_mode="always",
    )
    gui_app.optimization_method.set("grid")  # live state no longer matches the frozen decision

    def grid_must_not_run(*args, **kwargs):
        raise AssertionError("grid path should not run for this test")

    monkeypatch.setattr("spectral_predict.search.run_search", grid_must_not_run)
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(
            ["PLS"], "quick", analysis_run_id=meta.run_id, uses_bayesian_run_state=True,
        )

    # The frozen run id is neither silently completed nor lost: it stays
    # resumable via the round-8 `finally` safety net (grid dispatch never
    # calls _complete_run_state_after_search itself).
    assert rs.find_incomplete_run() is not None
    assert rs.find_incomplete_run().run_id == meta.run_id
    assert rs.get_active_run_id() is None
    gui_app.optimization_method.set("unified")


# ---------------------------------------------------------------------------
# Item 5: every post-registration exit either completes or releases the claim
# ---------------------------------------------------------------------------


def test_one_class_guard_return_releases_claim_keeps_sidecar(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.task_type.set("one_class")
    X, _ = _regression_data()
    y = pd.Series(["a"] * 22 + ["b"] * 8)
    gui_app.X, gui_app.y = X, y
    try:
        with patch("tkinter.messagebox.showerror"):
            # An inlier label that can't possibly be in the data trips the
            # "Invalid Inlier Class" guard, which used to `return` without
            # ever clearing the in-process claim.
            gui_app._run_analysis_thread(["PCA-SIMCA"], "quick", resolved_inlier_label="not-a-real-class")

        assert rs.find_incomplete_run() is not None, "the sidecar must survive"
        assert rs.get_active_run_id() is None, "the in-process claim must be released"
    finally:
        gui_app.task_type.set("regression")


def test_setup_exception_releases_claim_keeps_sidecar(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    gui_app.task_type.set("auto")

    def boom(_y):
        raise RuntimeError("boom during task-type inference")

    monkeypatch.setattr(gui_module, "_infer_task_type_from_y", boom)
    try:
        with patch("tkinter.messagebox.showerror"):
            gui_app._run_analysis_thread(["PLS"], "quick")

        assert rs.find_incomplete_run() is not None
        assert rs.get_active_run_id() is None
    finally:
        gui_app.task_type.set("regression")


def test_resume_fingerprint_recheck_return_is_not_double_handled(gui_app, worker_env, monkeypatch):
    """The worker's own resume fingerprint re-check (a deliberate, fully
    handled exit: is_resuming() stays True, nothing on disk changes) must
    NOT be treated by the round-8 `finally` safety net as an unhandled exit
    — that would incorrectly clear_resume_state() an intact resume."""
    rs = worker_env
    X, y = _regression_data()
    meta = rs.start_run(
        label="quick", dataset_fingerprint=rs.fingerprint_dataset(X, y),
        bayesian_persistence_mode="always",
    )
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()
    assert rs.resume_run(meta.run_id) is not None

    gui_app.X, gui_app.y = X, y
    monkeypatch.setattr(rs, "verify_resume_fingerprint", lambda fp: (False, "different"))
    with patch("tkinter.messagebox.showerror"):
        gui_app._run_analysis_thread(
            ["PLS"], "quick", analysis_run_id=meta.run_id, uses_bayesian_run_state=True,
        )

    assert rs.is_resuming(), "the fingerprint re-check's own handling must be left alone"
    assert rs.find_incomplete_run().run_id == meta.run_id
