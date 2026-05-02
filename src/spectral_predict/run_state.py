"""
Per-run state + Optuna SQLite storage management (T-11 D).

When a Bayesian search starts, a `run_state.start_run()` call generates a
unique run-id, creates a SQLite store at
`<user_data_dir>/dasp/optuna/<run_id>.sqlite3`, and writes a sidecar JSON at
`<user_data_dir>/dasp/optuna/active_run.json` describing the run's
configuration. While the run is active, `optuna.create_study` calls in
`unified_bayesian.py` pull the storage URL via `get_storage_url()` and pass
`load_if_exists=True` so re-running with the same name picks up where it
left off. On successful completion the sidecar is removed and the storage
file is left behind (user can delete it, or we can clean up old ones on a
schedule — both are out of scope for V1).

If the app crashes or is force-quit mid-run, the sidecar persists. On the
next app launch, `find_incomplete_run()` returns the sidecar metadata and
the GUI shows a "Found incomplete run — resume?" dialog. If the user
picks Resume, `resume_run()` sets the active storage URL to the previous
run's SQLite, and the user's next "Run Analysis" click re-uses the
existing studies — Optuna skips already-completed trials and continues
from the cutoff.

Public surface:
    start_run(label, dataset_fingerprint, model_names, n_trials_per_model)
        -> RunMetadata
    mark_complete()
    get_storage_url() -> str | None
    is_resuming() -> bool
    find_incomplete_run() -> RunMetadata | None
    resume_run(run_id)
    discard_incomplete_run(run_id)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from spectral_predict.resource_paths import get_user_optuna_dir

logger = logging.getLogger(__name__)

# Closed set of legal persistence-mode values, shared by RunMetadata and
# run_unified_bayesian's enable_sqlite_persistence parameter so the two stay
# in sync. Invalid values raise ValueError at the call boundary instead of
# silently falling through to the in-memory branch.
PersistenceMode = Literal["auto", "always", "never"]
_VALID_PERSISTENCE_MODES = ("auto", "always", "never")


def _validate_persistence_mode(value: str) -> str:
    """Raise ValueError if `value` isn't one of the three legal modes."""
    if value not in _VALID_PERSISTENCE_MODES:
        raise ValueError(
            f"persistence mode must be one of {_VALID_PERSISTENCE_MODES}, got {value!r}"
        )
    return value


_lock = threading.Lock()
_active_storage_url: str | None = None
_active_run_id: str | None = None
# Cached metadata from the original `start_run` call. Codex+type-design-analyzer
# meta-review Cluster C: prior `start_run` idempotent-return path synthesized a
# fresh RunMetadata with the *new caller's* args while reusing the original
# run_id/storage_url, producing inconsistent state vs. what the sidecar held on
# disk. Caching the original metadata fixes the contract: subsequent calls
# return the SAME object the first call returned.
_active_metadata: "RunMetadata | None" = None
_is_resuming: bool = False
_SIDECAR_NAME = "active_run.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON to `path` atomically.

    Codex HIGH #5: a plain `write_text()` can leave partial JSON if the
    process dies mid-write, and two app instances can race on the same
    sidecar. Write to a temp file in the same directory, fsync, then
    `os.replace()` into place — `replace` is atomic on POSIX and Windows
    (since Python 3.3). This eliminates partial-state corruption and
    narrows the multi-instance race window to "whoever finishes second
    wins," which is acceptable for our single-user GUI scenario.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
            fp.flush()
            try:
                os.fsync(fp.fileno())
            except OSError:
                # Some filesystems (network shares, ramdisks) don't support
                # fsync. `os.replace` still gives crash-atomicity on the
                # destination side, so this isn't fatal.
                pass
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the temp file if replace failed.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


@dataclasses.dataclass
class RunMetadata:
    run_id: str
    storage_path: str
    storage_url: str
    label: str | None
    dataset_fingerprint: str | None
    model_names: list[str]
    n_trials_per_model: int | None
    started_iso: str
    bayesian_persistence_mode: PersistenceMode = "never"  # T-41
    # T-43: snapshot of GUI settings at start_run time. None when no settings
    # were captured (older sidecars, headless callers). Stored as a flat
    # dict[str, JSON-serializable] so future GUI additions auto-flow through
    # without schema migration; restore tolerates missing/unknown keys.
    gui_settings: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_persistence_mode(self.bayesian_persistence_mode)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        # Older sidecars predate the field; default to 'never'. Unknown values
        # (e.g. corrupted sidecar) coerce to 'never' rather than crashing the
        # resume flow — but log a warning so the issue isn't invisible.
        mode = data.get("bayesian_persistence_mode", "never")
        if mode not in _VALID_PERSISTENCE_MODES:
            logger.warning(
                "T-41: sidecar has invalid bayesian_persistence_mode=%r; coercing to 'never'",
                mode,
            )
            mode = "never"
        data["bayesian_persistence_mode"] = mode

        # T-43: ignore unknown fields so a future schema addition can land
        # without breaking older Python builds that lack the field. Without
        # this, `cls(**data)` would TypeError on the unknown kwarg.
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclasses.dataclass
class DiscardResult:
    """Outcome of `discard_incomplete_run`.

    Codex meta-review Cluster A3: the prior `discard_incomplete_run` always
    returned `True` even when both `unlink()` calls failed. The GUI surfaced
    "Discarding stale sidecar + SQLite" while neither file was actually
    removed, leading to silently-orphaned SQLite files (storage-only failure)
    or repeated resume prompts (sidecar-failure). Callers now get per-file
    success and a list of human-readable error strings to surface.
    """
    sidecar_deleted: bool
    storage_deleted: bool
    errors: list[str]

    @property
    def fully_succeeded(self) -> bool:
        return self.sidecar_deleted and self.storage_deleted and not self.errors


def _sidecar_path() -> Path:
    return get_user_optuna_dir() / _SIDECAR_NAME


def _cleanup_empty_sqlite(meta: "RunMetadata") -> None:
    """T-41: delete the SQLite file if it has no trial rows.

    When all models stay in-memory, Optuna either never creates the file or
    creates an essentially-empty shell. We delete it so the next session's
    ``find_incomplete_run()`` doesn't offer a phantom "Resume?".

    Trial-count gate (not file size): a tiny one-class run can produce a
    real <32 KB SQLite, so the prior size threshold could destroy the only
    successful trial's record. Open the DB, count rows in `trials`, delete
    only when count == 0. Lock-failures (AV, in-use) surface at WARNING so
    silent disk-leaks don't accumulate.
    """
    if not meta.storage_path:
        return
    sqlite_path = Path(meta.storage_path)
    if not sqlite_path.exists():
        return  # never created — nothing to clean up

    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=2.0)
        try:
            row = conn.execute("SELECT COUNT(*) FROM trials").fetchone()
            trial_count = int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            # Schema not initialized — file exists but Optuna never wrote.
            trial_count = 0
        else:
            logger.warning(
                "T-41: stale-SQLite cleanup skipped (lock/AV/permission?): %s", exc
            )
            return
    except sqlite3.DatabaseError as exc:
        logger.warning("T-41: stale-SQLite cleanup skipped (corrupt file?): %s", exc)
        return

    if trial_count > 0:
        return  # has real data; keep it

    try:
        sqlite_path.unlink(missing_ok=True)
        logger.debug("T-41: removed empty SQLite file %s", sqlite_path)
    except FileNotFoundError:
        pass  # raced with another instance
    except OSError as exc:
        logger.warning(
            "T-41: could not remove empty SQLite file %s: %s", sqlite_path, exc
        )


def fingerprint_dataset(X, y) -> str:
    """Compute a short, deterministic fingerprint of an (X, y) dataset.

    Used to warn the user on resume if they've loaded different data than
    the original run was running on. Hashes shape + a small slice of values
    rather than the full array (the full hash would be costly on large
    spectra, and we only need to detect "is this clearly different?", not
    cryptographic equivalence).
    """
    try:
        import numpy as np
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        h = hashlib.sha256()
        h.update(str(X_arr.shape).encode("utf-8"))
        h.update(str(y_arr.shape).encode("utf-8"))
        # Take a few elements from start, middle, end of each — enough to
        # distinguish unrelated datasets without hashing GB of data.
        if X_arr.size > 0:
            for idx in (0, X_arr.size // 2, X_arr.size - 1):
                try:
                    h.update(str(X_arr.flat[idx]).encode("utf-8"))
                except Exception:
                    break
        if y_arr.size > 0:
            for idx in (0, y_arr.size // 2, y_arr.size - 1):
                try:
                    h.update(str(y_arr.flat[idx]).encode("utf-8"))
                except Exception:
                    break
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def start_run(
    label: str | None = None,
    dataset_fingerprint: str | None = None,
    model_names: list[str] | None = None,
    n_trials_per_model: int | None = None,
    bayesian_persistence_mode: PersistenceMode = "never",
    gui_settings: dict[str, Any] | None = None,
) -> RunMetadata:
    """Begin a new Optuna-persisted run. Idempotent within one search.

    The first call generates a UUID, picks the storage path, and writes the
    sidecar. Subsequent calls in the same search return the existing
    metadata so all `create_study` callers within one Run Analysis click
    share one SQLite file.

    T-41: when ``bayesian_persistence_mode='never'``, no SQLite URL is
    generated (``get_storage_url()`` returns ``None``). This saves I/O and
    avoids orphaned ``.sqlite3`` sidecars for all-in-memory sessions.
    """
    _validate_persistence_mode(bayesian_persistence_mode)
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming
    with _lock:
        # Cluster C fix: idempotent path returns the cached original metadata,
        # NOT a synthesized one. This ensures callers see the same fingerprint,
        # label, model_names, and started_iso the FIRST `start_run` recorded —
        # what's actually on disk in the sidecar — rather than whatever args
        # the second caller happened to pass.
        if _active_metadata is not None:
            return _active_metadata

        run_id = uuid.uuid4().hex[:12]

        # T-41: skip SQLite URL entirely for 'never' mode — saves I/O and
        # avoids the stale-sidecar problem (no SQLite file → no orphan).
        if bayesian_persistence_mode == "never":
            storage_path = get_user_optuna_dir() / f"{run_id}.sqlite3"
            storage_url = None  # type: ignore[assignment]
        else:
            storage_path = get_user_optuna_dir() / f"{run_id}.sqlite3"
            # Optuna's SQLite URL needs forward slashes even on Windows.
            # Kimi MINOR #7: extend the busy timeout. Default SQLite lock-wait
            # is short; Windows + concurrent dasp instances can hit "database
            # is locked" errors mid-trial. 30s gives the contended writer
            # plenty of time to finish without false-failing the optimization.
            storage_url = (
                f"sqlite:///{storage_path.as_posix()}?check_same_thread=False&timeout=30"
            )

        meta = RunMetadata(
            run_id=run_id,
            storage_path=str(storage_path),
            storage_url=storage_url or "",  # empty string when 'never'
            label=label,
            dataset_fingerprint=dataset_fingerprint,
            model_names=list(model_names or []),
            n_trials_per_model=n_trials_per_model,
            started_iso=datetime.now().isoformat(),
            bayesian_persistence_mode=bayesian_persistence_mode,
            gui_settings=dict(gui_settings) if gui_settings else None,
        )
        _atomic_write_json(_sidecar_path(), meta.to_dict())
        _active_storage_url = storage_url
        _active_run_id = run_id
        _active_metadata = meta
        _is_resuming = False
        return meta


def mark_complete() -> None:
    """Mark the active run as cleanly finished. Removes the sidecar.

    Codex meta-review NEW BUG #1: prior implementation deleted the sidecar
    UNCONDITIONALLY. The GUI calls `mark_complete()` after every successful
    analysis (Bayesian / grid / NSGA), so a user with a paused Bayesian run
    who clicks "Decide later" and then completes a fresh grid search would
    have their prior resume sidecar silently destroyed. Fix: only unlink
    the sidecar if its `run_id` matches `_active_run_id` — i.e. only delete
    OUR sidecar.

    Codex meta-review A1: prior implementation also cleared `_active_run_id`
    even when the unlink failed (Windows file lock, AV, permission). This
    diverged in-memory state from disk and silenced the failure. Fix: on
    OSError, leave in-memory state alone and re-raise so the caller's
    handler can surface the failure.

    Leaves the SQLite file in place so the user can inspect Optuna study
    contents post-hoc. Old SQLite files accumulate in
    `<user_data_dir>/dasp/optuna/`; manual cleanup or a future scheduled
    cleanup is the user's responsibility (out of T-11 D scope).
    """
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming
    with _lock:
        sidecar = _sidecar_path()
        sidecar_belongs_to_active_run = False
        if sidecar.exists() and _active_run_id is not None:
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar_belongs_to_active_run = (
                    data.get("run_id") == _active_run_id
                )
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                # Unreadable sidecar — conservatively don't unlink in case
                # it belongs to a different run or another instance owns it.
                sidecar_belongs_to_active_run = False

        if sidecar_belongs_to_active_run:
            try:
                sidecar.unlink()
            except OSError:
                # Cleanup failed; preserve in-memory state so the user can
                # retry and so the next launch can still find the sidecar.
                # Re-raise — the GUI handler at the call site already wraps
                # mark_complete() in try/except and surfaces the failure.
                raise

        # T-41 stale-sidecar cleanup: if the SQLite file was never written to
        # (all models stayed in-memory) the file either doesn't exist or is
        # essentially empty. Delete it so the next session's find_incomplete_run()
        # doesn't offer a phantom "Resume?" for a run with nothing to resume.
        if _active_metadata is not None:
            _cleanup_empty_sqlite(_active_metadata)

        # Sidecar either didn't exist, didn't belong to us, or was deleted.
        # Either way, our run is done — clear in-memory state.
        _active_storage_url = None
        _active_run_id = None
        _active_metadata = None
        _is_resuming = False


def get_storage_url() -> str | None:
    """Return the active Optuna storage URL, or None if no run is active.

    Used by `unified_bayesian.create_study` to decide whether to pass
    `storage=...` (persistent) or fall back to in-memory.
    """
    return _active_storage_url


def is_resuming() -> bool:
    """True if the active run was loaded from a prior crashed run."""
    return _is_resuming


def verify_resume_fingerprint(current_fingerprint: str) -> tuple[bool, str | None]:
    """Compare the current dataset fingerprint to the resumed run's stored value.

    Codex HIGH #7: the fingerprint was being stored at run start but never
    enforced at resume time, so a user could click Resume on a stale
    sidecar, load different data, and Optuna would silently pick up the
    old run's trials with new (incompatible) objective values. This
    function gives the GUI a place to gate that.

    Returns (matches, stored_fingerprint). `matches` is True if:
        - we're not currently in a resumed state (nothing to verify), OR
        - the stored fingerprint is unknown/empty (older sidecars), OR
        - the current and stored fingerprints are identical.
    Otherwise returns (False, stored_fingerprint) and the caller should
    refuse to proceed — typically by calling `clear_resume_state()` and
    surfacing an error to the user.
    """
    if not _is_resuming:
        return True, None
    if not _active_run_id:
        return True, None

    sidecar = _sidecar_path()
    if not sidecar.exists():
        # Sidecar was deleted between resume_run() and now — treat as
        # "nothing to verify" since the resume metadata is gone anyway.
        return True, None

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True, None

    # Kimi MAJOR #2: a second app instance could have overwritten the sidecar
    # between resume_run() and now. If the sidecar's run_id no longer
    # matches what we resumed, the on-disk fingerprint is for a DIFFERENT
    # run — comparing against it is meaningless. Refuse the resume so the
    # GUI can quarantine the stale sidecar.
    if data.get("run_id") != _active_run_id:
        return False, None

    stored = data.get("dataset_fingerprint")
    if not stored or stored == "unknown":
        # Older sidecar without a fingerprint — accept silently.
        return True, None

    return stored == current_fingerprint, stored


def clear_resume_state() -> None:
    """Drop the resume flag without deleting the sidecar / SQLite.

    Fallback path used when the GUI cannot determine the rejected run_id
    (e.g. import-time / partial-init failure); in normal operation,
    fingerprint mismatches go through `discard_incomplete_run` instead
    (Kimi MAJOR #3b). The sidecar persists; future launches will re-offer
    it for inspection.

    T-41: also cleans up empty SQLite files from all-in-memory sessions so
    the next launch doesn't offer a phantom "Resume?" with nothing to resume.
    """
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming
    with _lock:
        if _active_metadata is not None:
            _cleanup_empty_sqlite(_active_metadata)
        _active_storage_url = None
        _active_run_id = None
        _active_metadata = None
        _is_resuming = False


def find_incomplete_run() -> RunMetadata | None:
    """Look for a sidecar from a previously crashed/aborted run.

    Returns the metadata if one exists, else None. Does NOT modify state —
    the GUI calls this on startup to decide whether to show the resume
    dialog. The actual resume happens via `resume_run(run_id)`.

    Codex meta-review A2: prior implementation caught only
    `(JSONDecodeError, TypeError, KeyError)` and silently DELETED corrupt
    sidecars. Two issues fixed here:
      1. `OSError` / `PermissionError` / `UnicodeDecodeError` from
         `read_text()` now bubble up — a locked or unreadable sidecar is
         a caller-visible decision (start fresh? abort? retry?), not
         something the library should silently swallow.
      2. Unreadable-but-existing sidecars are quarantined (renamed to
         `.corrupt`) rather than deleted — a downgrade from a future
         schema looks identical to a corruption, and we want recovery
         to remain possible.
    """
    sidecar = _sidecar_path()
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return RunMetadata.from_dict(data)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, TypeError, KeyError, UnicodeDecodeError):
        try:
            sidecar.rename(sidecar.with_suffix(".corrupt"))
        except OSError:
            # Quarantine failed — fall back to deletion as last resort
            # so we don't keep prompting on a sidecar we can't parse.
            try:
                sidecar.unlink()
            except OSError:
                pass
        return None
    # OSError (permission, locked file, dead network share) intentionally
    # escapes — the GUI startup wraps this in its own handler and surfaces
    # a warning to the user.


def resume_run(run_id: str) -> RunMetadata | None:
    """Activate a previously-incomplete run by run_id.

    The next `optuna.create_study` call in the search will pass `storage=`
    pointing at the existing SQLite, plus `load_if_exists=True`, so the
    same study_name will pick up where it left off. Returns the metadata
    on success, or None if the sidecar's run_id doesn't match or the
    on-disk SQLite is missing or untrusted.

    Code-reviewer + Codex meta-review: the sidecar's `storage_path` is
    untrusted JSON content. Validate it resolves under the project's
    user-optuna directory before trusting it as the Optuna URL — a tampered
    sidecar (e.g. via Dropbox/OneDrive sync conflict) could otherwise
    point Optuna at an arbitrary path on disk.
    """
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming

    meta = find_incomplete_run()
    if meta is None or meta.run_id != run_id:
        return None

    optuna_dir = get_user_optuna_dir().resolve()
    try:
        storage_path = Path(meta.storage_path).resolve()
    except OSError:
        return None
    if not storage_path.is_relative_to(optuna_dir):
        # Tampered sidecar — refuse to use the path or the URL derived
        # from it. Don't auto-discard; let the GUI surface the situation.
        return None
    if not storage_path.exists():
        # SQLite file went missing — nothing to resume from. Discard the
        # orphaned sidecar so we don't keep prompting next launch.
        discard_incomplete_run(run_id)
        return None

    with _lock:
        _active_storage_url = meta.storage_url
        _active_run_id = meta.run_id
        _active_metadata = meta
        _is_resuming = True
    return meta


def discard_incomplete_run(run_id: str) -> DiscardResult:
    """Delete the sidecar + SQLite for an incomplete run.

    Returns a `DiscardResult` describing per-file success and any errors.
    Codex meta-review A3: prior implementation swallowed `OSError` on both
    unlink calls and returned bare `True` even when nothing was removed —
    the GUI's "Discarding stale sidecar + SQLite" message lied about
    success when the files were locked. Callers can now surface the
    actual outcome.

    Code-reviewer: also path-validates `storage_path` against the project's
    user-optuna directory before unlinking, refusing to follow a tampered
    sidecar that points outside.
    """
    meta = find_incomplete_run()
    if meta is None or meta.run_id != run_id:
        return DiscardResult(sidecar_deleted=False, storage_deleted=False, errors=[])

    sidecar = _sidecar_path()
    optuna_dir = get_user_optuna_dir().resolve()
    errors: list[str] = []

    sidecar_deleted = False
    try:
        if sidecar.exists():
            sidecar.unlink()
            sidecar_deleted = True
    except OSError as e:
        errors.append(f"sidecar unlink failed: {e}")

    storage_deleted = False
    try:
        storage_path = Path(meta.storage_path).resolve()
    except OSError as e:
        errors.append(f"storage_path resolve failed: {e}")
        return DiscardResult(
            sidecar_deleted=sidecar_deleted,
            storage_deleted=False,
            errors=errors,
        )
    if not storage_path.is_relative_to(optuna_dir):
        errors.append(
            f"storage_path outside optuna dir, refusing to unlink: {storage_path}"
        )
    else:
        try:
            if storage_path.exists():
                storage_path.unlink()
                storage_deleted = True
        except OSError as e:
            errors.append(f"storage unlink failed: {e}")

    return DiscardResult(
        sidecar_deleted=sidecar_deleted,
        storage_deleted=storage_deleted,
        errors=errors,
    )


def _reset_for_tests() -> None:
    """Test-only reset of module-level state. Do not call from production code."""
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming
    with _lock:
        _active_storage_url = None
        _active_run_id = None
        _active_metadata = None
        _is_resuming = False
