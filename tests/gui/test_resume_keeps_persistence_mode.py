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
