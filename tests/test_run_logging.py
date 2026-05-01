"""T-11 A regression tests for per-run disk-mirrored logging."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_logging(tmp_path, monkeypatch):
    """Force the run-logging module into a clean state with a tmp log dir.

    Each test gets a fresh module + a private LOCALAPPDATA / XDG_DATA_HOME so
    the tests don't write into the user's real ~/AppData or ~/.local.
    """
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    # Drop the cached module + reimport so module-level state is fresh.
    sys.modules.pop("spectral_predict.run_logging", None)
    sys.modules.pop("spectral_predict.resource_paths", None)
    rp = importlib.import_module("spectral_predict.resource_paths")
    rl = importlib.import_module("spectral_predict.run_logging")

    yield rl, rp, tmp_path

    # Clean up: detach any handlers + restore stdout/stderr if the test
    # installed the tee proxy.
    logger = logging.getLogger("dasp.run")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    if rl._original_stdout is not None:
        sys.stdout = rl._original_stdout
        sys.stderr = rl._original_stderr


def test_setup_run_logger_creates_log_file(fresh_logging):
    rl, _, _ = fresh_logging

    logger, path = rl.setup_run_logger(capture_stdout=False)

    assert path.exists()
    assert path.suffix == ".log"
    assert path.stat().st_size > 0  # Banner already written
    logger.info("test message")
    for h in logger.handlers:
        h.flush()
    contents = path.read_text(encoding="utf-8")
    assert "DASP run log" in contents
    assert "test message" in contents


def test_setup_run_logger_is_idempotent(fresh_logging):
    rl, _, _ = fresh_logging

    _, path1 = rl.setup_run_logger(capture_stdout=False)
    _, path2 = rl.setup_run_logger(capture_stdout=False)

    assert path1 == path2  # Same file across multiple calls in one process


def test_log_event_no_op_before_setup(fresh_logging):
    rl, _, _ = fresh_logging

    # Must not raise if called before setup_run_logger
    rl.log_event("orphan event")
    assert rl.get_active_log_path() is None


def test_log_event_writes_after_setup(fresh_logging):
    rl, _, _ = fresh_logging

    _, path = rl.setup_run_logger(capture_stdout=False)
    rl.log_event("event after setup")
    for h in logging.getLogger("dasp.run").handlers:
        h.flush()
    contents = path.read_text(encoding="utf-8")
    assert "event after setup" in contents


def test_label_appears_in_filename(fresh_logging):
    rl, _, _ = fresh_logging

    _, path = rl.setup_run_logger(label="standard", capture_stdout=False)
    assert "_standard" in path.name


def test_capture_stdout_tees_print_to_log(fresh_logging):
    rl, _, _ = fresh_logging

    _, path = rl.setup_run_logger(capture_stdout=True)
    print("backend message via stdout")
    sys.stdout.flush()
    for h in logging.getLogger("dasp.run").handlers:
        h.flush()
    contents = path.read_text(encoding="utf-8")
    assert "backend message via stdout" in contents


def test_user_data_dir_under_localappdata(fresh_logging):
    _, rp, tmp_path = fresh_logging

    user_dir = rp.get_user_data_dir()
    log_dir = rp.get_user_log_dir()
    optuna_dir = rp.get_user_optuna_dir()

    assert user_dir.is_relative_to(tmp_path)
    assert user_dir.name == "dasp"
    assert log_dir == user_dir / "logs"
    assert optuna_dir == user_dir / "optuna"
    assert log_dir.exists()
    assert optuna_dir.exists()
