"""
Per-run disk-mirrored logging for the GUI.

Without this, a multi-hour search that crashes leaves no post-mortem trail —
the Tkinter progress widget is capped at 2000 lines and dies with the
process. This module sets up a `logging.FileHandler` writing to
`<user_data_dir>/dasp/logs/run_<timestamp>.log` so any GUI progress event,
backend logger call, or routed `print()` survives the run.

The setup also installs a tee-style stdout/stderr proxy so backend `print()`
calls in `search.py` / `unified_bayesian.py` / `nsga2_search.py` (which go
to `/dev/null` in the bundled app where `console=False`) also land in the
log file. Original streams are preserved on dev runs.

Public surface:
    setup_run_logger(label: str | None = None, capture_stdout: bool = True)
        -> tuple[Logger, Path]
        Configure a per-run logger and return (logger, log_path).
        Idempotent — multiple calls in the same process reuse the same file
        for the lifetime of the run, so background workers and the GUI see
        the same log.

    get_active_log_path() -> Path | None
        Read-only accessor for "where is the active log file" (used by GUI
        to surface a "View log" button).
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from datetime import datetime
from pathlib import Path

from spectral_predict.resource_paths import get_user_log_dir

_LOGGER_NAME = "dasp.run"
_active_log_path: Path | None = None
_original_stdout = None
_original_stderr = None

# Guards the idempotency check + setup block in setup_run_logger so two
# concurrent callers can't both pass the None-check and double-attach the
# RotatingFileHandler / TeeStream. The current single-caller pattern in
# the GUI thread is safe, but the docstring promises multi-caller use
# ("background workers and the GUI see the same log"), so the contract
# needs the lock to hold under future call-sites.
_setup_lock = threading.RLock()

# Codex MEDIUM #10: cap log file growth. A 12-hour verbose run can produce
# multi-GB logs without rotation. RotatingFileHandler caps each file at
# this size and keeps a small backup chain.
_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
_LOG_BACKUP_COUNT = 3                # plus the active file = 200 MB ceiling

# Codex MEDIUM #10: also flush oversized newline-free buffers to avoid
# pathological memory growth if a backend module writes a huge string
# without ever emitting a newline.
_BUFFER_BYTE_THRESHOLD = 64 * 1024


class _TeeStream:
    """File-like object that writes to both an original stream and a logger.

    Backend `print()` -> sys.stdout.write -> tee writes to file logger AND
    original stdout (which is the dev console or /dev/null in the bundle).

    Thread-safe: the buffer is guarded by a per-stream lock (Codex MEDIUM #2).
    `setup_run_logger` replaces process-global `sys.stdout` / `sys.stderr`,
    so worker threads + library threads can write concurrently. Without the
    lock, partial-line interleaving would corrupt log entries and could drop
    tail content during concurrent flushes.
    """

    def __init__(self, original, logger: logging.Logger, level: int = logging.INFO):
        self._original = original
        self._logger = logger
        self._level = level
        self._buffer: list[str] = []
        self._buffer_bytes = 0
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        # Write through to the original stream first so dev-console behavior
        # is preserved. Uses no lock because the original stream's
        # thread-safety is its own concern (file objects are; sys.stdout is).
        try:
            if self._original is not None:
                self._original.write(text)
        except Exception:
            pass

        if not text:
            return 0

        # Kimi MINOR #4: `splitlines()` splits on BOTH '\r' and '\n'. tqdm
        # progress bars emit `\r<bar>\r<bar>...\n` and would generate one
        # log line per intermediate bar update — pure spam. Strip carriage
        # returns so only newlines split, and the final post-`\r` line state
        # ends up logged once.
        text_for_buffer = text.replace("\r", "")
        if not text_for_buffer:
            return len(text)

        emit_lines: list[str] = []
        with self._lock:
            self._buffer.append(text_for_buffer)
            self._buffer_bytes += len(text_for_buffer)
            should_flush = (
                "\n" in text_for_buffer
                or self._buffer_bytes >= _BUFFER_BYTE_THRESHOLD
            )
            if should_flush:
                joined = "".join(self._buffer)
                self._buffer = []
                self._buffer_bytes = 0
                # Use split('\n') (newline-only) since we already stripped \r.
                if joined.endswith("\n"):
                    emit_lines = joined.split("\n")[:-1]  # drop trailing ""
                    tail = ""
                else:
                    parts = joined.split("\n")
                    if parts:
                        # Last fragment has no trailing newline yet — keep
                        # it buffered so we don't fragment a partial line.
                        emit_lines = parts[:-1]
                        tail = parts[-1]
                    else:
                        tail = ""
                if tail:
                    self._buffer.append(tail)
                    self._buffer_bytes = len(tail)

        # Emit outside the lock to minimize contention; logger is itself
        # thread-safe so concurrent emits are fine. Preserve whitespace-only
        # and empty lines: the log file is meant to mirror stdout faithfully
        # for crash post-mortem, so backend-formatted blank-line separators
        # in tables / progress sections must survive.
        for line in emit_lines:
            self._logger.log(self._level, line)
        return len(text)

    def flush(self) -> None:
        emit_lines: list[str] = []
        with self._lock:
            if self._buffer:
                joined = "".join(self._buffer)
                self._buffer = []
                self._buffer_bytes = 0
                # Same \n-only split as write() — already \r-stripped.
                # Preserve whitespace-only and empty lines for log fidelity
                # (mirrors the same change in write() above).
                emit_lines = joined.split("\n")
                # Drop only the trailing "" produced when joined ends in "\n"
                # — that artifact is a split-side-effect, not a real line.
                if emit_lines and emit_lines[-1] == "" and joined.endswith("\n"):
                    emit_lines = emit_lines[:-1]
        for line in emit_lines:
            self._logger.log(self._level, line)
        try:
            if self._original is not None:
                self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._original and self._original.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        if self._original is not None:
            return self._original.fileno()
        raise OSError("Tee stream has no underlying file descriptor")


def setup_run_logger(
    label: str | None = None, capture_stdout: bool = True
) -> tuple[logging.Logger, Path]:
    """Configure the per-run logger. Returns (logger, log_path).

    The first call in a process creates the log file, attaches a
    FileHandler, and (if `capture_stdout=True`) installs a tee-style proxy
    over `sys.stdout` and `sys.stderr` so backend `print()` calls land in
    the file. Subsequent calls return the same logger + path so background
    workers and the GUI share one file.
    """
    global _active_log_path, _original_stdout, _original_stderr

    with _setup_lock:
        logger = logging.getLogger(_LOGGER_NAME)
        if _active_log_path is not None:
            return logger, _active_log_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        log_path = get_user_log_dir() / f"run_{timestamp}{suffix}.log"

        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't double-emit to root

        # Codex MEDIUM #10: rotate to cap total disk usage on long verbose runs.
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=False,
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

        logger.info("=== DASP run log ===")
        logger.info("Path: %s", log_path)
        logger.info("Python: %s", sys.version.split()[0])
        logger.info("Platform: %s", sys.platform)

        if capture_stdout and _original_stdout is None:
            _original_stdout = sys.stdout
            _original_stderr = sys.stderr
            sys.stdout = _TeeStream(_original_stdout, logger, logging.INFO)
            sys.stderr = _TeeStream(_original_stderr, logger, logging.WARNING)

        _active_log_path = log_path
        return logger, log_path


def get_active_log_path() -> Path | None:
    """Return the active log file path, or None if no run logger has been set up."""
    return _active_log_path


def log_event(message: str) -> None:
    """Convenience: log an INFO event to the run logger if one is active.

    Safe to call before `setup_run_logger` — silently no-ops in that case so
    early-startup callers don't have to guard.
    """
    if _active_log_path is None:
        return
    logging.getLogger(_LOGGER_NAME).info(message)
