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
import os
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

from spectral_predict.resource_paths import get_user_optuna_dir


_lock = threading.Lock()
_active_storage_url: str | None = None
_active_run_id: str | None = None
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

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        return cls(**data)


def _sidecar_path() -> Path:
    return get_user_optuna_dir() / _SIDECAR_NAME


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
) -> RunMetadata:
    """Begin a new Optuna-persisted run. Idempotent within one search.

    The first call generates a UUID, picks the storage path, and writes the
    sidecar. Subsequent calls in the same search return the existing
    metadata so all `create_study` callers within one Run Analysis click
    share one SQLite file.
    """
    global _active_storage_url, _active_run_id, _is_resuming
    with _lock:
        if _active_storage_url is not None and _active_run_id is not None:
            existing_path = get_user_optuna_dir() / f"{_active_run_id}.sqlite3"
            return RunMetadata(
                run_id=_active_run_id,
                storage_path=str(existing_path),
                storage_url=_active_storage_url,
                label=label,
                dataset_fingerprint=dataset_fingerprint,
                model_names=model_names or [],
                n_trials_per_model=n_trials_per_model,
                started_iso=datetime.now().isoformat(),
            )

        run_id = uuid.uuid4().hex[:12]
        storage_path = get_user_optuna_dir() / f"{run_id}.sqlite3"
        # Optuna's SQLite URL needs forward slashes even on Windows.
        # Kimi MINOR #7: extend the busy timeout. Default SQLite lock-wait is
        # short; Windows + concurrent dasp instances can hit "database is
        # locked" errors mid-trial. 30s gives the contended writer plenty of
        # time to finish without false-failing the optimization.
        storage_url = (
            f"sqlite:///{storage_path.as_posix()}?check_same_thread=False&timeout=30"
        )

        meta = RunMetadata(
            run_id=run_id,
            storage_path=str(storage_path),
            storage_url=storage_url,
            label=label,
            dataset_fingerprint=dataset_fingerprint,
            model_names=list(model_names or []),
            n_trials_per_model=n_trials_per_model,
            started_iso=datetime.now().isoformat(),
        )
        _atomic_write_json(_sidecar_path(), meta.to_dict())
        _active_storage_url = storage_url
        _active_run_id = run_id
        _is_resuming = False
        return meta


def mark_complete() -> None:
    """Mark the active run as cleanly finished. Removes the sidecar.

    Leaves the SQLite file in place so the user can inspect Optuna study
    contents post-hoc if they want; cleanup of old SQLite files is a
    separate concern (out of T-11 D scope).
    """
    global _active_storage_url, _active_run_id, _is_resuming
    with _lock:
        sidecar = _sidecar_path()
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
        _active_storage_url = None
        _active_run_id = None
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

    Used after a fingerprint mismatch when the user wants to fall back to
    a fresh run but keep the previous sidecar around for inspection. The
    sidecar persists; future launches will re-offer it.
    """
    global _active_storage_url, _active_run_id, _is_resuming
    with _lock:
        _active_storage_url = None
        _active_run_id = None
        _is_resuming = False


def find_incomplete_run() -> RunMetadata | None:
    """Look for a sidecar from a previously crashed/aborted run.

    Returns the metadata if one exists, else None. Does NOT modify state —
    the GUI calls this on startup to decide whether to show the resume
    dialog. The actual resume happens via `resume_run(run_id)`.
    """
    sidecar = _sidecar_path()
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return RunMetadata.from_dict(data)
    except (json.JSONDecodeError, TypeError, KeyError):
        # Corrupt sidecar — discard it.
        try:
            sidecar.unlink()
        except OSError:
            pass
        return None


def resume_run(run_id: str) -> RunMetadata | None:
    """Activate a previously-incomplete run by run_id.

    The next `optuna.create_study` call in the search will pass `storage=`
    pointing at the existing SQLite, plus `load_if_exists=True`, so the
    same study_name will pick up where it left off. Returns the metadata
    on success, or None if the sidecar's run_id doesn't match.
    """
    global _active_storage_url, _active_run_id, _is_resuming

    meta = find_incomplete_run()
    if meta is None or meta.run_id != run_id:
        return None
    storage_path = Path(meta.storage_path)
    if not storage_path.exists():
        # SQLite file went missing — nothing to resume from.
        discard_incomplete_run(run_id)
        return None

    with _lock:
        _active_storage_url = meta.storage_url
        _active_run_id = meta.run_id
        _is_resuming = True
    return meta


def discard_incomplete_run(run_id: str) -> bool:
    """Delete the sidecar + SQLite for an incomplete run. Returns True on success."""
    meta = find_incomplete_run()
    if meta is None or meta.run_id != run_id:
        return False
    sidecar = _sidecar_path()
    storage_path = Path(meta.storage_path)
    try:
        if sidecar.exists():
            sidecar.unlink()
    except OSError:
        pass
    try:
        if storage_path.exists():
            storage_path.unlink()
    except OSError:
        pass
    return True


def _reset_for_tests() -> None:
    """Test-only reset of module-level state. Do not call from production code."""
    global _active_storage_url, _active_run_id, _is_resuming
    with _lock:
        _active_storage_url = None
        _active_run_id = None
        _is_resuming = False
