"""The "Resume previous run?" flow must not change the user's persistence radio.

It used to force ``bayesian_persistence_mode`` to 'always'. 'auto' now reloads a
matching saved study by itself (tests/test_resume_under_auto_after_crash.py), so
the resume flow keeps the restored/user value and says nothing about Always-on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def run_state_tmp(tmp_path, monkeypatch, reimport_modules):
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()
    yield rs
    rs._reset_for_tests()


def _crashed_run(rs, mode: str, gui_mode: str | None, with_store: bool):
    settings = {"bayesian_persistence_mode": gui_mode} if gui_mode else None
    meta = rs.start_run(label="t", bayesian_persistence_mode=mode, gui_settings=settings)
    if with_store:
        Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()  # simulate a new app session; the sidecar stays on disk
    return meta


@pytest.mark.parametrize("user_mode", ["auto", "always"])
def test_resume_does_not_change_persistence_mode(gui_app, run_state_tmp, user_mode):
    rs = run_state_tmp
    gui_app.bayesian_persistence_mode.set(user_mode)
    meta = _crashed_run(rs, "auto", gui_mode=None, with_store=True)

    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask, \
         patch("tkinter.messagebox.showwarning") as warn:
        gui_app._check_for_incomplete_run()

    assert ask.called, "a run with a saved store must be offered for resume"
    assert not warn.called, "no 'set it manually' fallback warning"
    assert rs.is_resuming() and rs.get_storage_url() == meta.storage_url
    assert gui_app.bayesian_persistence_mode.get() == user_mode
    banner = gui_app.progress_status.cget("text")
    assert "Resuming previous run" in banner
    assert "Always-on" not in banner


def test_resume_restores_captured_mode_without_override(gui_app, run_state_tmp):
    rs = run_state_tmp
    gui_app.bayesian_persistence_mode.set("never")
    _crashed_run(rs, "auto", gui_mode="auto", with_store=True)

    with patch("tkinter.messagebox.askyesnocancel", return_value=True):
        gui_app._check_for_incomplete_run()

    assert gui_app.bayesian_persistence_mode.get() == "auto"


@pytest.mark.parametrize("mode", ["never", "auto"])
def test_no_prompt_when_nothing_was_saved(gui_app, run_state_tmp, mode):
    """'never' (and an 'auto' crash during in-memory warmup) has no SQLite store."""
    rs = run_state_tmp
    gui_app.bayesian_persistence_mode.set(mode)
    _crashed_run(rs, mode, gui_mode=mode, with_store=False)

    with patch("tkinter.messagebox.askyesnocancel") as ask, \
         patch("tkinter.messagebox.showwarning") as warn:
        gui_app._check_for_incomplete_run()

    assert not ask.called
    assert not warn.called
    assert not rs.is_resuming()
    assert gui_app.bayesian_persistence_mode.get() == mode
    # The sidecar is kept (no cross-process-unsafe delete); next start_run replaces it.
    assert rs.find_incomplete_run() is not None


def test_banner_does_not_promise_unconditional_reuse(gui_app, run_state_tmp):
    _crashed_run(run_state_tmp, "auto", gui_mode=None, with_store=True)
    with patch("tkinter.messagebox.askyesnocancel", return_value=True):
        gui_app._check_for_incomplete_run()
    banner = gui_app.progress_status.cget("text")
    assert "reused when the data" in banner


@pytest.fixture
def immediate_after(gui_app, monkeypatch):
    """Run root.after callbacks synchronously so worker-thread UI updates are observable."""
    def run_now(_ms, func=None, *args):
        if func is not None:
            func(*args)
    monkeypatch.setattr(gui_app.root, "after", run_now)
    logged: list[str] = []
    real_log = gui_app._log_progress
    monkeypatch.setattr(gui_app, "_log_progress", lambda m: (logged.append(m), real_log(m)))
    return logged


def test_declined_resume_is_surfaced(gui_app, run_state_tmp, immediate_after):
    """A resumed run whose saved study is not reused must tell the user."""
    rs = run_state_tmp
    meta = _crashed_run(rs, "auto", gui_mode=None, with_store=True)
    assert rs.resume_run(meta.run_id) is not None
    event = {
        "stage": "unified_bayesian",
        "message": "[T-41] A persisted study named for this PLS configuration exists but "
                   "was run on different or unrecorded data, so it is NOT resumed.",
        "t41_decision": "auto_existing_study_data_mismatch",
        "resume_declined": True,
    }

    with patch("tkinter.messagebox.showwarning") as warn:
        gui_app._progress_callback(event)
        gui_app._progress_callback(dict(event))  # second model: no second dialog

    assert warn.call_count == 1
    assert "not reused" in warn.call_args[0][0].lower()
    assert any(m.startswith("[RUN] Resume: saved trials were NOT reused") for m in immediate_after)
    assert "could not be reused" in gui_app.progress_status.cget("text")


@pytest.mark.parametrize(
    "key, title, log_prefix, status_fragment",
    [
        ("data_mismatch_resume", "Resumed on different data",
         "[RUN] Resume: saved trials were reused, but they were run on DIFFERENT data",
         "reused but the data differs"),
        ("data_unverified_resume", "Resumed data can't be verified",
         "[RUN] Resume: saved trials were reused, but their data can't be verified",
         "can't be verified"),
        ("resume_check_failed", "Saved trials could not be checked",
         "[RUN] Resume: could not check this model's saved trials",
         "could not be checked"),
    ],
)
def test_resume_issue_notices_are_worded_per_kind(
    gui_app, run_state_tmp, immediate_after, key, title, log_prefix, status_fragment
):
    """An explicit-'always' resume that replays trials from different/unverifiable data
    is 'resumed, but ...', not 'declined'; all kinds share the one-per-run dialog."""
    rs = run_state_tmp
    meta = _crashed_run(rs, "always", gui_mode=None, with_store=True)
    assert rs.resume_run(meta.run_id) is not None

    with patch("tkinter.messagebox.showwarning") as warn:
        gui_app._progress_callback({"message": "[T-41] WARNING: detail", key: True})
        assert status_fragment in gui_app.progress_status.cget("text")
        gui_app._progress_callback({"message": "second", "resume_declined": True})

    assert warn.call_count == 1, "one dialog per resumed run across all kinds"
    assert warn.call_args[0][0] == title
    assert any(m.startswith(log_prefix) for m in immediate_after)


def test_declined_event_outside_resume_is_only_logged(gui_app, run_state_tmp, immediate_after):
    event = {"message": "Previous Bayesian results ...", "environment_changed": True,
             "resume_declined": True}
    with patch("tkinter.messagebox.showwarning") as warn:
        gui_app._progress_callback(event)
    assert not warn.called
    assert not any(m.startswith("[RUN] Resume:") for m in immediate_after)
