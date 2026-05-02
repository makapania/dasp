"""
Per-run disk-mirrored logging for the GUI.

Without this, a multi-hour search that crashes leaves no post-mortem trail —
the Tkinter progress widget is capped at 2000 lines and dies with the
process. This module sets up a `logging.handlers.RotatingFileHandler` writing to
`<user_data_dir>/dasp/logs/run_<timestamp>.log` (rotated at 50 MB, 3
backups kept) so any GUI progress event,
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


class _SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler with a non-recursive error path.

    Codex meta-review B2: the default `handleError` writes to `sys.stderr`,
    but `setup_run_logger` REPLACES `sys.stderr` with a `_TeeStream` that
    forwards to this very handler. A rollover failure (Windows file lock
    on `.log` -> `.log.1` rename, AV holding the inode) would then trigger
    handleError -> sys.stderr.write -> tee -> handler.emit -> rollover
    fails again -> feedback loop bloating the buffer. Override to write
    error notices to the *original* stderr captured at setup time, which
    is never the tee.
    """

    def handleError(self, record: logging.LogRecord) -> None:
        if logging.raiseExceptions:
            target = _original_stderr if _original_stderr is not None else sys.__stderr__
            try:
                import traceback
                target.write("--- _SafeRotatingFileHandler error ---\n")
                traceback.print_exc(file=target)
                target.write(f"Logged from file {record.filename}, line {record.lineno}\n")
                target.write("--- end handler error ---\n")
                target.flush()
            except Exception:
                pass  # last resort — there's nowhere safe left to log to

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
            newline_present = "\n" in text_for_buffer
            byte_threshold_hit = self._buffer_bytes >= _BUFFER_BYTE_THRESHOLD
            if newline_present or byte_threshold_hit:
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
                        # Last fragment has no trailing newline yet.
                        # Normally we keep it buffered so a partial line
                        # isn't fragmented in the log. BUT if the byte
                        # threshold tripped without ANY newline in the
                        # whole buffer (parts == [joined]), keeping the
                        # tail would just re-buffer everything — defeating
                        # the threshold's stated purpose of capping memory
                        # on long no-newline strings (Codex meta-review B3).
                        # Emit the whole tail as one record in that case.
                        if byte_threshold_hit and not newline_present and len(parts) == 1:
                            emit_lines = parts
                            tail = ""
                        else:
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
        #
        # Codex meta-review B1: each `_logger.log` call is wrapped because
        # `sys.stdout` has been replaced with this object — if the logger's
        # file handler is broken (disk full, rollover lock), an unprotected
        # raise would propagate out of any `print()` call site in a worker
        # thread and crash it. On logger failure we fall back to the
        # original stream so the message is at least visible there.
        for line in emit_lines:
            try:
                self._logger.log(self._level, line)
            except Exception:
                try:
                    if self._original is not None:
                        self._original.write(line + "\n")
                except Exception:
                    pass  # both paths broken — last resort, drop the line
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
        # Codex meta-review B1: same protection as in write(). If the file
        # handler is broken, fall back to the original stream rather than
        # crashing the caller (sys.stdout is this object).
        for line in emit_lines:
            try:
                self._logger.log(self._level, line)
            except Exception:
                try:
                    if self._original is not None:
                        self._original.write(line + "\n")
                except Exception:
                    pass
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
    _SafeRotatingFileHandler (50 MB per file, 3 backups), and (if
    `capture_stdout=True`) installs a tee-style proxy
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
        # Codex meta-review B2: use _SafeRotatingFileHandler so rollover
        # errors don't recurse through the tee back into this handler.
        fh = _SafeRotatingFileHandler(
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


_app_log_path: Path | None = None
_app_logger_lock = threading.RLock()
_APP_LOG_MAX_BYTES = 1_000_000  # 1 MB per file — module warnings only, not run output
_APP_LOG_BACKUP_COUNT = 3       # 4 MB total ceiling


def setup_app_logger() -> Path | None:
    """Wire a RotatingFileHandler to the ``spectral_predict`` logger.

    Module-level ``logger.warning`` calls in run_state, run_gui_settings,
    and unified_bayesian propagate up to the ``spectral_predict`` logger.
    Without a handler attached there, those warnings vanish in the bundled
    PyInstaller GUI (no console attached → stderr is /dev/null).

    Distinct from ``setup_run_logger``: that wires a per-run log keyed to
    Run Analysis click time. This wires an app-lifetime log so warnings
    from app startup (e.g. corrupted resume sidecar) and from module-level
    paths that aren't behind the dasp.run logger still land somewhere
    diagnosable. Both can coexist — the run logger is in
    ``logs/run_<timestamp>.log``; this one is at the top level
    ``<user_data_dir>/dasp.log``.

    Idempotent: a second call returns the same path without re-adding the
    handler. Returns the path on success, ``None`` if setup failed (the
    GUI keeps starting either way — logger visibility is best-effort).
    """
    global _app_log_path
    with _app_logger_lock:
        if _app_log_path is not None:
            return _app_log_path

        try:
            from spectral_predict.resource_paths import get_user_data_dir

            log_path = get_user_data_dir() / "dasp.log"
            handler = _SafeRotatingFileHandler(
                log_path,
                maxBytes=_APP_LOG_MAX_BYTES,
                backupCount=_APP_LOG_BACKUP_COUNT,
                encoding="utf-8",
                delay=False,
            )
            handler.setLevel(logging.WARNING)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            sp_logger = logging.getLogger("spectral_predict")
            sp_logger.addHandler(handler)
            if sp_logger.level == logging.NOTSET or sp_logger.level > logging.WARNING:
                sp_logger.setLevel(logging.WARNING)

            _app_log_path = log_path
            return log_path
        except Exception:
            return None


def get_app_log_path() -> Path | None:
    """Return the app-lifetime log file path, or None if setup hasn't run."""
    return _app_log_path


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
