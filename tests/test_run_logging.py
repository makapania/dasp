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
    # installed the tee proxy. T-45 added the spectral_predict logger
    # handler too — clean both so tests don't leak state into siblings.
    for logger_name in ("dasp.run", "spectral_predict"):
        sp_logger = logging.getLogger(logger_name)
        for handler in list(sp_logger.handlers):
            handler.close()
            sp_logger.removeHandler(handler)
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


# ----------------------------------------------------------------------
# Tests added in response to PR #6 multi-agent review (Codex meta-review)
# ----------------------------------------------------------------------


def test_tee_preserves_blank_and_whitespace_lines(fresh_logging):
    """MEDIUM #4 regression: write/flush must mirror stdout faithfully,
    including blank and whitespace-only lines (backend table separators
    and progress sections lose meaning when stripped).
    """
    rl, _, _ = fresh_logging

    captured: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    logger = logging.getLogger("dasp.run.tee_blank_test")
    logger.handlers.clear()
    logger.addHandler(Capture())
    logger.setLevel(logging.INFO)
    logger.propagate = False

    import io
    tee = rl._TeeStream(io.StringIO(), logger, logging.INFO)
    tee.write("header\n\n   \nbody\n")
    tee.flush()

    assert captured == ["header", "", "   ", "body"]


def test_tee_byte_threshold_emits_no_newline_string(fresh_logging):
    """B3 regression: when the byte threshold trips on a buffer with no
    newlines at all, the tail must be EMITTED (not re-buffered verbatim).
    Re-buffering defeats the threshold's stated purpose of capping memory
    on long no-newline strings.
    """
    rl, _, _ = fresh_logging

    captured: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    logger = logging.getLogger("dasp.run.tee_threshold_test")
    logger.handlers.clear()
    logger.addHandler(Capture())
    logger.setLevel(logging.INFO)
    logger.propagate = False

    import io
    tee = rl._TeeStream(io.StringIO(), logger, logging.INFO)
    big_string = "x" * (rl._BUFFER_BYTE_THRESHOLD + 1024)  # No newline
    tee.write(big_string)

    assert captured, "buffer threshold should have flushed at least one record"
    assert captured[0] == big_string
    # Buffer must be drained — bytes counter back to zero.
    assert tee._buffer_bytes == 0
    assert tee._buffer == []


def test_tee_logger_failure_falls_back_to_original(fresh_logging):
    """B1 regression: when the underlying logger raises (broken file
    handler / disk full / rotation lock), _TeeStream falls back to the
    original stream rather than propagating the exception out of every
    print() call site in worker threads.
    """
    rl, _, _ = fresh_logging

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            raise OSError("simulated handler death (e.g. disk full)")

    logger = logging.getLogger("dasp.run.tee_broken_test")
    logger.handlers.clear()
    logger.addHandler(BrokenHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False

    import io
    fallback_stream = io.StringIO()
    tee = rl._TeeStream(fallback_stream, logger, logging.INFO)
    # Must not raise even though logger.emit raises.
    tee.write("worker thread message\n")
    tee.flush()
    # Fallback stream got the original write (above the lock-protected
    # path) AND the recovery write (below). The first write was not
    # guarded by the logger; both messages should appear in the fallback.
    fallback_text = fallback_stream.getvalue()
    assert "worker thread message" in fallback_text


def test_rotating_handler_rolls_over_at_threshold(fresh_logging, monkeypatch):
    """DeepSeek "cannot verify #1" regression: force the rolling threshold
    to a tiny value, write enough lines to trigger multiple rotations,
    and verify the backup chain populates without raising.
    """
    rl, _, _ = fresh_logging

    monkeypatch.setattr(rl, "_LOG_MAX_BYTES", 1024)  # 1 KB threshold
    monkeypatch.setattr(rl, "_LOG_BACKUP_COUNT", 3)

    logger, path = rl.setup_run_logger(capture_stdout=False)

    # Each iteration writes ~80 bytes (timestamp + level + message). To get
    # past 1 KB threshold and force multiple rotations, write 200 records.
    for i in range(200):
        logger.info("rollover-pressure record %d with some padding text", i)
    for h in logger.handlers:
        h.flush()

    # Active log file should exist and be under the threshold (rotation
    # truncates as it rolls).
    assert path.exists()
    # Backup files appear with `.1`, `.2`, ... suffixes; at least one must
    # exist after this volume. Use glob to handle path differences.
    backups = list(path.parent.glob(f"{path.name}.*"))
    assert backups, f"expected at least one backup, got none in {list(path.parent.iterdir())}"


# ---------------------------------------------------------------------------
# T-45: app-lifetime logger handler
# ---------------------------------------------------------------------------


def test_setup_app_logger_creates_dasp_log_at_user_data_dir(fresh_logging):
    """T-45: setup_app_logger writes to <user_data_dir>/dasp.log so module
    warnings (sidecar corruption, WAL rejection, capture-time Tk failures)
    survive in the bundled GUI where stderr is /dev/null."""
    rl, _, tmp_path = fresh_logging

    path = rl.setup_app_logger()
    assert path is not None
    assert path.name == "dasp.log"
    assert path.parent.name == "dasp"

    # Module loggers (run_state, run_gui_settings, unified_bayesian) all
    # use logger = logging.getLogger(__name__), so they propagate up to
    # the spectral_predict logger where the T-45 handler attached.
    sp_logger = logging.getLogger("spectral_predict")
    assert any(
        h.__class__.__name__ == "_SafeRotatingFileHandler"
        for h in sp_logger.handlers
    )


def test_setup_app_logger_captures_module_warnings(fresh_logging):
    """End-to-end: a logger.warning from a child logger (the same
    'spectral_predict.run_state' shape used in production) lands in
    dasp.log."""
    rl, _, _ = fresh_logging
    path = rl.setup_app_logger()
    assert path is not None

    child = logging.getLogger("spectral_predict.run_state")
    child.warning("T-45 smoke test: corrupted sidecar coerced to never")

    for h in logging.getLogger("spectral_predict").handlers:
        h.flush()

    contents = path.read_text(encoding="utf-8")
    assert "T-45 smoke test" in contents
    assert "WARNING" in contents
    assert "spectral_predict.run_state" in contents


def test_setup_app_logger_idempotent(fresh_logging):
    """Idempotent: a second call returns the same path AND doesn't add a
    duplicate handler (which would double-log every warning)."""
    rl, _, _ = fresh_logging

    path1 = rl.setup_app_logger()
    path2 = rl.setup_app_logger()
    assert path1 == path2

    sp_logger = logging.getLogger("spectral_predict")
    file_handlers = [
        h for h in sp_logger.handlers
        if h.__class__.__name__ == "_SafeRotatingFileHandler"
    ]
    assert len(file_handlers) == 1, (
        f"setup_app_logger must not double-attach; got {len(file_handlers)}"
    )


def test_setup_app_logger_swallows_setup_failure(fresh_logging, monkeypatch):
    """Logger setup is best-effort: if get_user_data_dir() raises
    (read-only filesystem, missing env var on a misconfigured host), the
    function returns None and the caller continues. App startup must not
    hang on a logger config failure."""
    rl, rp, _ = fresh_logging

    def _explode():
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr(rp, "get_user_data_dir", _explode)
    # Also patch where setup_app_logger imports from.
    monkeypatch.setattr(
        "spectral_predict.resource_paths.get_user_data_dir", _explode
    )

    path = rl.setup_app_logger()
    assert path is None, "setup must swallow the exception and return None"
