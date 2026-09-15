"""Loaded data that does not match the resumed run must never delete the saved run.

User decision on PR #79: "leave it and say it does not match so they can change".
Run Analysis now asks: keep the saved run (nothing runs; load the matching data and
click Run again) or start a fresh analysis on purpose (the old SQLite file stays on
disk and can no longer be resumed by accident). These tests drive the real
``_run_analysis_thread`` up to ``start_run`` and stop there.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


class _StopBeforeSearch(BaseException):
    """Raised from the start_run spy; BaseException escapes the thread's handlers."""


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
    sidecar = rs._sidecar_path()

    def run_now(_ms, func=None, *args):
        if func is not None:
            func(*args)

    monkeypatch.setattr(gui_app.root, "after", run_now)
    method = gui_app.optimization_method.get()
    gui_app.optimization_method.set("unified")
    gui_app._pending_validation_indices = [0, 1]

    started: list = []
    real_start_run = rs.start_run

    def start_run_spy(*args, **kwargs):
        started.append(real_start_run(*args, **kwargs))
        raise _StopBeforeSearch

    monkeypatch.setattr(rs, "start_run", start_run_spy)
    search = Mock(side_effect=AssertionError("the search must not start"))
    monkeypatch.setattr("spectral_predict.search.run_search", search)

    yield rs, meta, store, sidecar, started, (X, y)

    gui_app.optimization_method.set(method)
    gui_app._pending_validation_indices = None
    rs._reset_for_tests()


def _run_analysis(gui_app, X, y) -> bool:
    """Run the worker body synchronously; True if it reached start_run."""
    gui_app.X, gui_app.y = X, y
    try:
        gui_app._run_analysis_thread(["PLS"], "quick")
    except _StopBeforeSearch:
        return True
    return False


def test_mismatch_keeps_saved_run_and_does_not_start(gui_app, resumed):
    rs, meta, store, sidecar, started, _ = resumed
    other_X, other_y = _dataset(2)

    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        reached = _run_analysis(gui_app, other_X, other_y)

    assert ask.call_count == 1
    assert "does not match" in ask.call_args[0][0]
    assert "nothing was deleted" in ask.call_args[0][1]
    assert not reached and not started, "nothing may run on mismatched data"
    assert store.exists() and sidecar.exists()
    assert rs.find_incomplete_run().run_id == meta.run_id
    assert rs.is_resuming() and rs.get_storage_url() == meta.storage_url
    assert gui_app._pending_validation_indices == [0, 1]
    assert "Resume kept" in gui_app.progress_status.cget("text")


def test_retry_with_matching_data_resumes(gui_app, resumed):
    rs, meta, store, sidecar, started, (X, y) = resumed
    other_X, other_y = _dataset(2)
    with patch("tkinter.messagebox.askyesno", return_value=False):
        assert not _run_analysis(gui_app, other_X, other_y)

    with patch("tkinter.messagebox.askyesno") as ask:
        reached = _run_analysis(gui_app, X, y)

    assert not ask.called
    assert reached
    assert started[0].run_id == meta.run_id, "start_run returned the resumed run"
    assert rs.is_resuming() and rs.get_storage_url() == meta.storage_url


def test_start_fresh_keeps_file_and_cannot_resume_old_store(gui_app, resumed):
    rs, meta, store, sidecar, started, _ = resumed
    other_X, other_y = _dataset(2)

    with patch("tkinter.messagebox.askyesno", return_value=True):
        reached = _run_analysis(gui_app, other_X, other_y)

    assert reached, "a deliberate fresh start proceeds"
    new = started[0]
    assert new.run_id != meta.run_id
    assert new.storage_path != meta.storage_path
    assert not rs.is_resuming()
    assert rs.get_storage_url() != meta.storage_url
    assert store.exists(), "the old SQLite file stays for retention cleanup"
    assert rs.find_incomplete_run().run_id == new.run_id, "sidecar now names the new run"
    assert gui_app._pending_validation_indices is None
