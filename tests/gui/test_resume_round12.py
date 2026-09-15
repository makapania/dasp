"""Round 12 fixes for PR #79 (Codex review of round 11, 3f70023).

1. Calibration-row selection (validation holdout, excluded spectra, active
   subset) and the CV folds used for class filtering are frozen at the click.
2. A validation holdout that differs from the resumed run's is shown before
   resuming; the run's own split is restored on the main thread.
3. A Bayesian setting that can't be read stops the launch, and with a launch
   snapshot the worker never falls back to a live control.
4. A delete that removed the record (even if the store stayed) releases the resume.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import spectral_predict_gui_optimized as gui_module
from spectral_predict.run_state import DiscardResult

from tests.gui.test_resume_data_mismatch_keeps_run import (  # noqa: F401 -- fixtures
    _FakeThread,
    _dataset,
    resumed,
)
from tests.gui.test_resume_round9 import (  # noqa: F401 -- fixtures
    _recording_bayesian,
    _select_models,
    fake_thread,
)
from tests.gui.test_resume_run_completion import (  # noqa: F401 -- fixtures
    _regression_data,
    worker_env,
)


@pytest.fixture
def validation_state(gui_app):
    saved = (gui_app.validation_enabled.get(), gui_app.validation_indices,
             gui_app.validation_X, gui_app.validation_y, gui_app.excluded_spectra)
    yield
    (enabled, gui_app.validation_indices, gui_app.validation_X, gui_app.validation_y,
     gui_app.excluded_spectra) = saved
    gui_app.validation_enabled.set(enabled)


def test_rows_frozen_at_click(gui_app, worker_env, fake_thread, validation_state, monkeypatch):
    X, y = _regression_data()
    gui_app.X, gui_app.y = X, y
    _select_models(gui_app, ["PLS"])
    gui_app.validation_enabled.set(True)
    gui_app.validation_indices = set(X.index[:6])
    fake, calls = _recording_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)

    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        gui_app._run_analysis()
        worker = _FakeThread.created[-1]
        gui_app.validation_enabled.set(False)  # changed after the click
        gui_app.validation_indices = set()
        gui_app.excluded_spectra = set(X.index[6:10])
        worker.target(*worker.args, **worker.kwargs)

    assert calls[0][2] == len(X) - 6, "trained on the rows approved at the click"


def test_blank_required_setting_stops_launch(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    real = gui_module._launch_settings_snapshot

    def without_folds(app):
        snap = real(app)
        snap.pop("folds")  # what capture does when the IntVar box is blank
        return snap

    monkeypatch.setattr(gui_module, "_launch_settings_snapshot", without_folds)
    with patch("tkinter.messagebox.showerror") as err:
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert err.call_args[0][0] == "Invalid settings" and "folds" in err.call_args[0][1]
    assert rs.get_active_run_id() is None


def test_worker_with_snapshot_never_reads_live_controls(gui_app, worker_env, monkeypatch):
    gui_app.X, gui_app.y = _regression_data()
    fake, calls = _recording_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    snapshot = gui_module._launch_settings_snapshot(gui_app)
    snapshot.pop("folds")
    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        try:
            gui_app._run_analysis_thread(
                ["PLS"], "quick", analysis_run_id=None, uses_bayesian_run_state=True,
                analysis_settings=snapshot,
            )
        except KeyError:
            pass
    assert calls == [], "no search with a setting that wasn't approved"


@pytest.fixture
def resumed_with_split(gui_app, tmp_path, monkeypatch, reimport_modules, validation_state):
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()
    X, y = _dataset(1)
    meta = rs.start_run(
        label="t", dataset_fingerprint=rs.fingerprint_dataset(X, y),
        bayesian_persistence_mode="always", validation_indices=list(X.index[:4]),
    )
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()
    assert rs.resume_run(meta.run_id) is not None
    gui_app._pending_validation_indices = list(X.index[:4])
    saved = gui_app.optimization_method.get()
    gui_app.optimization_method.set("unified")
    gui_app.X, gui_app.y = X, y
    gui_app.validation_enabled.set(True)
    gui_app.validation_X = None
    yield rs, meta, X
    gui_app.optimization_method.set(saved)
    gui_app._pending_validation_indices = None
    rs._reset_for_tests()


def test_different_holdout_asks_and_yes_restores_runs_split(gui_app, resumed_with_split):
    rs, meta, X = resumed_with_split
    gui_app.validation_indices = set(X.index[4:8])
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
        assert gui_app._confirm_resume_before_launch(None, "quick") is True
    assert ask.call_args[0][0] == "Validation set differs from the interrupted run"
    assert gui_app.validation_indices == set(X.index[:4])


def test_different_holdout_cancel_keeps_everything(gui_app, resumed_with_split):
    rs, meta, X = resumed_with_split
    gui_app.validation_indices = set(X.index[4:8])
    with patch("tkinter.messagebox.askyesnocancel", return_value=None):
        assert gui_app._confirm_resume_before_launch(None, "quick") is False
    assert gui_app.validation_indices == set(X.index[4:8])
    assert rs.is_resuming()


def test_empty_holdout_restores_runs_split_without_asking(gui_app, resumed_with_split):
    rs, meta, X = resumed_with_split
    gui_app.validation_indices = set()
    with patch("tkinter.messagebox.askyesnocancel") as ask:
        assert gui_app._confirm_resume_before_launch(None, "quick") is True
    assert not ask.called
    assert gui_app.validation_indices == set(X.index[:4])


def test_record_deleted_store_kept_releases_resume(gui_app, resumed, monkeypatch):
    rs, meta, store, started, (X, y) = resumed
    monkeypatch.setattr(
        rs, "discard_incomplete_run",
        lambda run_id: DiscardResult(sidecar_deleted=True, storage_deleted=False,
                                     errors=["storage path unusable"]),
    )
    gui_app.X, gui_app.y = _dataset(2)  # mismatch -> start fresh
    with patch("tkinter.messagebox.askyesno", return_value=True), \
         patch("tkinter.messagebox.showerror") as err:
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert "no longer be resumed" in err.call_args[0][1]
    assert not rs.is_resuming()
