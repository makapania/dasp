"""Loaded data that does not match the resumed run must never delete the saved run.

User decision on PR #79: "leave it and say it does not match so they can change".
Run Analysis checks a pending resume on the Tk main thread before the worker starts:
- "No" (default) keeps the saved run. Nothing runs and the UI returns to idle.
- "Yes" deliberately starts fresh. The old SQLite file stays on disk and can no
  longer be resumed.
A resume that cannot be verified is treated the same way; it never counts as a match.
"""
from __future__ import annotations

import inspect
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui_module


class _StopBeforeSearch(BaseException):
    """Raised from the start_run spy; BaseException escapes the worker's handlers."""


class _FakeThread:
    created: list["_FakeThread"] = []

    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args
        _FakeThread.created.append(self)

    def start(self):
        pass

    def is_alive(self):
        return False


def _dataset(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(12, 6)), columns=[f"{w}" for w in range(1000, 1006)])
    return X, pd.Series(rng.normal(size=12), index=X.index)


@pytest.fixture
def resumed(gui_app, tmp_path, monkeypatch, reimport_modules):
    """A crashed 'auto' run with a saved store, resumed at startup."""
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()

    X, y = _dataset(1)
    meta = rs.start_run(
        label="t", dataset_fingerprint=rs.fingerprint_dataset(X, y),
        bayesian_persistence_mode="auto",
    )
    store = Path(meta.storage_path)
    store.write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()  # new app session
    assert rs.resume_run(meta.run_id) is not None

    def run_now(_ms, func=None, *args):
        if func is not None:
            func(*args)

    monkeypatch.setattr(gui_app.root, "after", run_now)
    saved = (gui_app.optimization_method.get(), gui_app.task_type.get(), gui_app.use_pls.get())
    gui_app.optimization_method.set("unified")
    gui_app.task_type.set("regression")
    gui_app.use_pls.set(True)
    gui_app._pending_validation_indices = [0, 1]

    _FakeThread.created = []
    monkeypatch.setattr(gui_module.threading, "Thread", _FakeThread)
    started: list = []
    real_start_run = rs.start_run

    def start_run_spy(*args, **kwargs):
        started.append(real_start_run(*args, **kwargs))
        raise _StopBeforeSearch

    monkeypatch.setattr(rs, "start_run", start_run_spy)
    monkeypatch.setattr(
        "spectral_predict.search.run_search",
        Mock(side_effect=AssertionError("the search must not start")),
    )

    yield rs, meta, store, started, (X, y)

    gui_app.optimization_method.set(saved[0])
    gui_app.task_type.set(saved[1])
    gui_app.use_pls.set(saved[2])
    gui_app._pending_validation_indices = None
    rs._reset_for_tests()


def _click_run(gui_app, X, y) -> _FakeThread | None:
    """Click Run Analysis; return the worker it would have launched, if any."""
    gui_app.X, gui_app.y = X, y
    before = len(_FakeThread.created)
    gui_app._run_analysis()
    return _FakeThread.created[-1] if len(_FakeThread.created) > before else None


def _run_worker(worker: _FakeThread) -> bool:
    """Run the worker body synchronously; True if it reached start_run."""
    try:
        worker.target(*worker.args)
    except _StopBeforeSearch:
        return True
    return False


def _assert_idle_and_kept(gui_app, rs, meta, store):
    assert store.exists() and rs._sidecar_path().exists()
    assert rs.find_incomplete_run().run_id == meta.run_id
    assert rs.is_resuming() and rs.get_storage_url() == meta.storage_url
    assert gui_app._pending_validation_indices == [0, 1]
    assert "Starting analysis" not in gui_app.progress_info.cget("text")
    assert gui_app.best_model_info.cget("text") == "(none)"
    assert gui_app.time_estimate_label.cget("text") == ""
    assert gui_app.search_controller.check_and_wait() is False, "controller ended"
    assert str(gui_app.stop_btn.cget("state")) == "disabled"


def test_mismatch_no_keeps_saved_run_and_returns_to_idle(gui_app, resumed):
    rs, meta, store, started, _ = resumed
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        worker = _click_run(gui_app, *_dataset(2))

    assert ask.call_count == 1
    assert ask.call_args[0][0] == "Different data than the interrupted run"
    body = ask.call_args[0][1]
    assert body.startswith("The data you loaded is not the same as the data")
    assert "nothing was deleted" in body
    assert body.rstrip().splitlines()[-1].startswith("Details:"), "technical detail last"
    assert ask.call_args.kwargs.get("default") == "no"
    assert worker is None, "no worker may start on mismatched data"
    assert not started
    _assert_idle_and_kept(gui_app, rs, meta, store)
    assert "Resume kept" in gui_app.progress_info.cget("text")


def test_dialog_runs_on_main_thread_and_a_slow_answer_is_honoured(gui_app, resumed):
    """No cross-thread wait or timeout: the answer, however late, decides."""
    rs, meta, store, started, _ = resumed
    seen = {}

    def slow_yes(*args, **kwargs):
        seen["main"] = threading.current_thread() is threading.main_thread()
        time.sleep(0.5)
        return True

    with patch("tkinter.messagebox.askyesno", side_effect=slow_yes):
        worker = _click_run(gui_app, *_dataset(2))

    assert seen["main"]
    assert worker is not None and not rs.is_resuming()


def test_retry_with_matching_data_resumes(gui_app, resumed):
    rs, meta, store, started, (X, y) = resumed
    with patch("tkinter.messagebox.askyesno", return_value=False):
        assert _click_run(gui_app, *_dataset(2)) is None

    with patch("tkinter.messagebox.askyesno") as ask:
        worker = _click_run(gui_app, X, y)
        assert _run_worker(worker)

    assert not ask.called
    assert started[0].run_id == meta.run_id, "start_run returned the resumed run"
    assert rs.is_resuming() and rs.get_storage_url() == meta.storage_url


def test_start_fresh_keeps_file_and_cannot_resume_old_store(gui_app, resumed):
    rs, meta, store, started, _ = resumed
    with patch("tkinter.messagebox.askyesno", return_value=True):
        worker = _click_run(gui_app, *_dataset(2))
    assert worker is not None
    assert _run_worker(worker), "a deliberate fresh start proceeds"

    new = started[0]
    assert new.run_id != meta.run_id and new.storage_path != meta.storage_path
    assert not rs.is_resuming() and rs.get_storage_url() != meta.storage_url
    assert store.exists(), "the old SQLite file stays for retention cleanup"
    assert rs.find_incomplete_run().run_id == new.run_id
    assert gui_app._pending_validation_indices is None


def test_verification_exception_stops_and_asks(gui_app, resumed, monkeypatch):
    rs, meta, store, started, (X, y) = resumed

    def boom(_fp):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(rs, "verify_resume_fingerprint", boom)
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        worker = _click_run(gui_app, X, y)  # even with matching data

    assert worker is None and not started
    assert ask.call_args[0][0] == "Can't check the interrupted run"
    body = ask.call_args[0][1]
    assert body.startswith("dasp could not read the record")
    assert body.rstrip().splitlines()[-1].startswith("Details:") and "disk on fire" in body
    _assert_idle_and_kept(gui_app, rs, meta, store)


def test_unreadable_sidecar_is_not_a_match(gui_app, resumed):
    rs, meta, store, started, (X, y) = resumed
    rs._sidecar_path().write_text("{corrupt", encoding="utf-8")
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        worker = _click_run(gui_app, X, y)

    assert worker is None and not started
    assert ask.call_args[0][0] == "Can't check the interrupted run"
    assert store.exists() and rs.is_resuming()


def test_worker_recheck_failure_stops_before_start_run(gui_app, resumed, monkeypatch):
    """If the record breaks between the click and the worker, the worker stops too."""
    rs, meta, store, started, (X, y) = resumed
    with patch("tkinter.messagebox.askyesno") as ask:
        worker = _click_run(gui_app, X, y)
    assert worker is not None and not ask.called

    def boom(_fp):
        raise RuntimeError("changed underneath")

    monkeypatch.setattr(rs, "verify_resume_fingerprint", boom)
    assert not _run_worker(worker)
    assert not started
    assert store.exists() and rs.is_resuming()
    assert "Resume kept" in gui_app.progress_info.cget("text")


def test_non_bayesian_completion_keeps_pending_resume(gui_app, resumed):
    """GLM review: a run with no registered id (grid/NSGA-II/SIMCA) must not complete
    the pending Bayesian resume, which would delete its sidecar."""
    rs, meta, store, started, _ = resumed
    from spectral_predict.search_controller import SearchController

    gui_app.search_controller = SearchController()
    gui_app._complete_run_state_after_search(None)
    gui_app._complete_run_state_after_search("some_other_run")
    assert rs.find_incomplete_run().run_id == meta.run_id
    assert rs.is_resuming()

    gui_app._complete_run_state_after_search(meta.run_id)  # the resumed run finishing
    assert rs.find_incomplete_run() is None
    assert store.exists()


def test_multiclass_simca_does_not_use_bayesian_run_state(gui_app, resumed):
    gui_app.task_type.set("multiclass_simca")
    try:
        assert gui_app._uses_bayesian_run_state() is False
        with patch("tkinter.messagebox.askyesno") as ask:
            assert gui_app._confirm_resume_before_launch() is True
        assert not ask.called
    finally:
        gui_app.task_type.set("regression")


def test_worker_has_no_end_of_run_mark_complete():
    source = inspect.getsource(gui_module.SpectralPredictApp._run_analysis_thread)
    assert "mark_complete()" not in source
    assert source.count("self._complete_run_state_after_search(analysis_run_id") == 2
    assert source.count("analysis_run_id = meta.run_id") == 1
