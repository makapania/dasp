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
    find_incomplete_run() -> RunMetadata | None   (raises CorruptRunRecordError)
    set_aside_corrupt_run_record() -> Path | None
    has_resumable_store(meta) -> bool
    get_active_run_id() -> str | None
    get_resumed_run() -> RunMetadata | None
    abandon_resume()
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
import stat
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

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


class ResumeVerificationError(RuntimeError):
    """The loaded data could not be checked against the run being resumed."""


class CorruptRunRecordError(RuntimeError):
    """The saved-run record exists but is not a valid run description.

    Raised by `find_incomplete_run` instead of silently moving the record aside,
    so the caller can tell the user and let them choose (#79 round 9). The file
    is left untouched; `set_aside_corrupt_run_record` moves it out of the way.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"the saved-run record {path} is damaged: {reason}")
        self.path = path
        self.reason = reason


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

# T-50: stale-SQLite cleanup policy. Old completed runs accumulate in
# `<user_data_dir>/dasp/optuna/` because mark_complete() deliberately leaves
# trial archives in place for post-hoc inspection (see its docstring).
_KEEP_LAST_N_RUNS = 5
_DELETE_AFTER_DAYS = 30
# A `*.sqlite3-wal` sibling whose mtime is within this window means another
# dasp instance is actively writing — never touch the underlying .sqlite3.
_WAL_SAFETY_WINDOW_HOURS = 1


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
    bayesian_persistence_mode: PersistenceMode = "auto"
    # Snapshot of GUI settings at start_run time. None when no settings
    # were captured (older sidecars, headless callers). Stored as a flat
    # dict[str, JSON-serializable] so future GUI additions auto-flow
    # through without schema migration; restore tolerates missing/unknown
    # keys.
    gui_settings: dict[str, Any] | None = None
    # External validation set indices (DataFrame index labels — can be int
    # or string depending on how the user loaded the data). Used with .loc
    # to re-slice the same calibration / validation partition the resumed
    # trials trained on, which prevents silent leakage when the original
    # algorithm was non-deterministic (Random) or hand-picked (Manual).
    # Deterministic algorithms (SPXY / Kennard-Stone / Stratified) would
    # reproduce the same indices from data + algorithm + percentage, but
    # persisting the indices is cheaper and removes an entire class of
    # "user forgets to click Create Validation Set on resume" footguns.
    validation_indices: list[Any] | None = None
    # The calibration-row choice (``calibration_rows_record``): versioned
    # ``canonical_label`` keys of the excluded samples, the Analysis Subset
    # ("active" None = all samples) and the holdout. Reloading the same file
    # clears exclusions; the resume gate uses these keys to offer restoring them.
    calibration_rows: dict[str, Any] | None = None
    # How sample labels were normalised when the run started
    # (``LABEL_NORMALIZATION``), written by the GUI, which also records
    # ``calibration_rows``. None: a legacy or headless record, saved before
    # repeated IDs got collision-free suffixes, so the same file can now give a
    # saved label to other rows.
    label_normalization: int | None = None
    # ``calibration_identity`` digest of the exact calibration and holdout rows
    # the run trained and scored on. The resume gate recomputes it and resumes
    # only on a match. None with the two fields above also None: a legacy record.
    calibration_identity: dict[str, Any] | None = None

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

        # Ignore unknown fields so a future schema addition can land without
        # breaking older Python builds that lack the field. Without this,
        # `cls(**data)` would TypeError on the unknown kwarg.
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}

        # Type guard: a corrupted sidecar storing `gui_settings: "malformed"`
        # (string instead of dict) would pass the field filter and crash
        # later in restore_gui_settings when `.items()` is called on a
        # string. Coerce to None with a warning so the resume flow degrades
        # to "no auto-restore" rather than surfacing a generic outer-handler
        # exception.
        gs = filtered.get("gui_settings")
        if gs is not None and not isinstance(gs, dict):
            logger.warning(
                "sidecar gui_settings has unexpected type %s; coercing to "
                "None (auto-restore disabled for this resume)",
                type(gs).__name__,
            )
            filtered["gui_settings"] = None

        # Same defense for validation_indices: must be a list of JSON
        # scalars (int or str — DataFrame index labels can be either).
        # A non-list or a list with non-scalar entries means corrupted
        # state — degrade to "no validation restore" rather than blindly
        # slicing with garbage labels.
        vi = filtered.get("validation_indices")
        if vi is not None:
            if not isinstance(vi, list) or not all(
                isinstance(x, (int, str)) and not isinstance(x, bool)
                for x in vi
            ):
                logger.warning(
                    "sidecar validation_indices has unexpected shape %s; "
                    "coercing to None (validation restore disabled for "
                    "this resume)",
                    type(vi).__name__,
                )
                filtered["validation_indices"] = None
        # calibration_rows / calibration_identity / label_normalization are kept
        # as stored: the resume gate treats an unrecognised value as "can't
        # verify" and asks, rather than silently resuming as if it were absent.
        return cls(**filtered)


# Version of the sample-label normalisation recorded with a run. 1: repeated
# IDs get collision-free suffixes ("A", "A.1", "A" -> "A", "A.1", "A.2"; the
# old scheme gave "A.1" twice) and the GUI suffixes any repeats left at install.
LABEL_NORMALIZATION = 1
# Version of the ``calibration_rows`` key encoding.
CALIBRATION_RECORD_VERSION = 1
# Version of the ``calibration_identity`` digest. 2: every label and section is
# length-prefixed, row counts are hashed, integer targets are hashed losslessly.
CALIBRATION_IDENTITY_VERSION = 2


class UnsupportedLabelError(ValueError):
    """A sample label has no deterministic, type-preserving encoding."""


def canonical_label(value: Any) -> str:
    """A type-tagged string for a sample label that round-trips through JSON.

    Supported: int (numpy ints too), float (numpy floats too, NaN allowed), str,
    bool, and tuples of these. Equal labels give equal keys and labels of
    different types never collide (``5``, ``5.0`` and ``"5"`` all differ).

    Raises:
        UnsupportedLabelError: any other label type. Its encoding would not be
            guaranteed to be distinct or stable across sessions.
    """
    if isinstance(value, np.generic):
        value = value.item()  # numpy scalar -> Python scalar
    if isinstance(value, bool):
        return f"b:{value}"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{value!r}"
    if isinstance(value, str):
        return f"s:{value}"
    if isinstance(value, tuple):
        return "t:" + json.dumps([canonical_label(v) for v in value])
    raise UnsupportedLabelError(f"unsupported sample label type {type(value).__name__}")


def _valid_key_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


_ROWS_KEYS = frozenset({"version", "excluded", "active", "holdout"})
_IDENTITY_KEYS = frozenset({"version", "calibration", "holdout", "n_calibration", "n_holdout"})


def valid_calibration_rows(rows: Any) -> bool:
    """True if ``rows`` is a ``calibration_rows_record`` of the current version.

    Every key must be present with the right type, so a record that passes can
    be decoded without further checks.
    """
    return (
        isinstance(rows, dict)
        and set(rows) == _ROWS_KEYS
        and rows["version"] == CALIBRATION_RECORD_VERSION
        and _valid_key_list(rows["excluded"])
        and _valid_key_list(rows["holdout"])
        and (rows["active"] is None or _valid_key_list(rows["active"]))
    )


def valid_calibration_identity(identity: Any) -> bool:
    """True if ``identity`` is a ``calibration_identity`` of the current version."""
    return (
        isinstance(identity, dict)
        and set(identity) == _IDENTITY_KEYS
        and identity["version"] == CALIBRATION_IDENTITY_VERSION
        and isinstance(identity["calibration"], str)
        and isinstance(identity["holdout"], str)
        and all(
            isinstance(identity[k], int) and not isinstance(identity[k], bool)
            for k in ("n_calibration", "n_holdout")
        )
    )


def calibration_rows_record(excluded, active, holdout=()) -> dict[str, Any]:
    """The calibration-row choice in a form the run record can store.

    Used to offer restoring a run's exclusions and holdout on resume; the
    ``calibration_identity`` digest is what decides whether a resume matches.

    Args:
        excluded: Labels the user excluded from the analysis.
        active: Labels of the Analysis Subset, or None for all samples.
        holdout: Labels of the validation holdout.

    Returns:
        ``{"version", "excluded", "active", "holdout"}`` with sorted
        ``canonical_label`` keys.

    Raises:
        UnsupportedLabelError: a label has no deterministic encoding.
    """
    return {
        "version": CALIBRATION_RECORD_VERSION,
        "excluded": sorted({canonical_label(v) for v in (excluded or ())}),
        "active": None if active is None else sorted({canonical_label(v) for v in active}),
        "holdout": sorted({canonical_label(v) for v in (holdout or ())}),
    }


def _put(h, tag: bytes, payload: bytes) -> None:
    """Feed one tagged, length-prefixed section, so sections can't run together."""
    h.update(tag)
    h.update(len(payload).to_bytes(8, "little"))
    h.update(payload)


def _target_bytes(y) -> tuple[bytes, bytes]:
    """``(tag, bytes)`` for a target column, lossless and container-independent.

    Integers are hashed as int64, or uint64 above the int64 range (float64 would
    merge values above 2**53). An object column whose values are all numbers is
    hashed like the numeric column it equals, so the same targets in a different
    container give the same digest. Integer columns that fit neither int64 nor
    uint64 (values beyond 2**64, or negatives mixed with values above the int64
    range) fall back to a per-value repr encoding.
    """
    import pandas as pd

    def _numeric(values) -> tuple[bytes, bytes] | None:
        if pd.api.types.is_bool_dtype(values):
            return b"yb", np.ascontiguousarray(values.to_numpy(dtype=np.uint8)).tobytes()
        if (
            pd.api.types.is_unsigned_integer_dtype(values)
            and not values.isna().any()
            and len(values)
            and int(values.max()) > np.iinfo(np.int64).max
        ):
            # Only values beyond int64 need their own encoding: casting them to
            # int64 would wrap them onto negatives.
            return b"yu", np.ascontiguousarray(values.to_numpy(dtype=np.uint64)).tobytes()
        if pd.api.types.is_integer_dtype(values) and not values.isna().any():
            return b"yi", np.ascontiguousarray(values.to_numpy(dtype=np.int64)).tobytes()
        if pd.api.types.is_numeric_dtype(values):
            return b"yf", np.ascontiguousarray(values.to_numpy(dtype=np.float64)).tobytes()
        return None

    encoded = _numeric(y)
    if encoded is not None:
        return encoded
    items = [v.item() if isinstance(v, np.generic) else v for v in y]
    numbers = [v for v in items if not (isinstance(v, float) and np.isnan(v))]
    if numbers and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in numbers):
        # Python numbers only (missing values as NaN): hash as the numeric column
        # they equal, provided the conversion loses nothing.
        if any(isinstance(v, float) for v in items):
            dtypes = ("float64",)
        else:
            dtypes = ("int64", "uint64")  # uint64 for non-negative ints above int64
        for dtype in dtypes:
            try:
                converted = pd.Series(items, dtype=dtype)
            except (OverflowError, ValueError, TypeError):
                continue  # out of this dtype's range
            if all(a == b or (a != a and b != b) for a, b in zip(items, converted.tolist())):
                return _numeric(converted)
    parts = []
    for value in items:
        text = repr(value).encode("utf-8")
        parts.append(len(text).to_bytes(8, "little") + text)
    return b"yo", b"".join(parts)


def _rows_digest(X, y) -> str:
    """blake2b over the ordered labels, wavelengths, spectra and targets of ``X``."""
    h = hashlib.blake2b(digest_size=16)
    h.update(f"calibration-identity-v{CALIBRATION_IDENTITY_VERSION}".encode("ascii"))
    n_rows = 0 if X is None else len(X)
    n_cols = 0 if X is None else X.shape[1]
    _put(h, b"shape", f"{n_rows}x{n_cols}".encode("ascii"))
    if X is None or n_rows == 0:
        return h.hexdigest()
    for tag, labels in ((b"col", X.columns), (b"row", X.index)):
        for label in labels:
            _put(h, tag, canonical_label(label).encode("utf-8"))
    _put(h, b"X", np.ascontiguousarray(X.to_numpy(dtype=np.float64)).tobytes())
    if y is None:
        _put(h, b"ynone", b"")
    else:
        y_aligned = y if y.index.equals(X.index) else y.reindex(X.index)
        tag, payload = _target_bytes(y_aligned)
        _put(h, tag, payload)
    return h.hexdigest()


def calibration_identity(X_cal, y_cal, X_holdout, y_holdout) -> dict[str, Any]:
    """Digest of exactly the calibration and holdout rows a run trains and scores on.

    Covers each row's label and position, the wavelength axis, every spectral
    value, the target and the row counts, so a resume on relabelled, reordered
    or edited data, or on a different sample selection, gives a different
    identity.

    Raises:
        UnsupportedLabelError: a label has no deterministic encoding.
    """
    return {
        "version": CALIBRATION_IDENTITY_VERSION,
        "calibration": _rows_digest(X_cal, y_cal),
        "holdout": _rows_digest(X_holdout, y_holdout),
        "n_calibration": 0 if X_cal is None else int(len(X_cal)),
        "n_holdout": 0 if X_holdout is None else int(len(X_holdout)),
    }


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


def _coerce_validation_indices(raw) -> list[Any] | None:
    """Normalize validation index labels for sidecar persistence.

    Returns a deterministic list of ints (sorted) when every label is
    int-typed; otherwise preserves insertion order. Filters non-scalar
    entries silently — caller is the GUI which already constrains input
    to DataFrame index labels.
    """
    cleaned = [
        x for x in raw
        if isinstance(x, (int, str)) and not isinstance(x, bool)
    ]
    if not cleaned:
        return None
    if all(isinstance(x, int) for x in cleaned):
        return sorted(cleaned)
    return cleaned


def start_run(
    label: str | None = None,
    dataset_fingerprint: str | None = None,
    model_names: list[str] | None = None,
    n_trials_per_model: int | None = None,
    bayesian_persistence_mode: PersistenceMode = "auto",
    gui_settings: dict[str, Any] | None = None,
    validation_indices: list[Any] | None = None,
    calibration_rows: dict[str, Any] | None = None,
    label_normalization: int | None = None,
    calibration_identity: dict[str, Any] | None = None,
) -> RunMetadata:
    """Begin a new Optuna-persisted run. Idempotent within one search.

    The first call generates a UUID, picks the storage path, and writes the
    sidecar. Subsequent calls in the same search return the existing
    metadata so all `create_study` callers within one Run Analysis click
    share one SQLite file.

    T-41: when ``bayesian_persistence_mode='never'``, no SQLite URL is
    generated (``get_storage_url()`` returns ``None``). This saves I/O and
    avoids orphaned ``.sqlite3`` sidecars for all-in-memory sessions.

    ``calibration_rows`` / ``label_normalization`` / ``calibration_identity``:
    the GUI passes the run's exclusions, Analysis Subset and holdout
    (``calibration_rows_record``), ``LABEL_NORMALIZATION`` and the digest of
    the rows it trains on (``calibration_identity``), so a resume can verify
    that it continues on the same calibration set.
    """
    _validate_persistence_mode(bayesian_persistence_mode)
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming
    if _active_metadata is None:
        # Never write over a damaged record: keep its contents under a new name
        # (#79 round 9). A rename failure raises OSError, so nothing is replaced.
        set_aside_corrupt_run_record()
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
            validation_indices=(
                # Preserve label type (int or str) — DataFrame .loc slicing
                # is type-sensitive. Sort only when all labels are int; for
                # mixed/str labels keep insertion order.
                _coerce_validation_indices(validation_indices)
                if validation_indices else None
            ),
            calibration_rows=_recordable(
                calibration_rows, valid_calibration_rows, "calibration_rows"
            ),
            label_normalization=label_normalization,
            calibration_identity=_recordable(
                calibration_identity, valid_calibration_identity, "calibration_identity"
            ),
        )
        _atomic_write_json(_sidecar_path(), meta.to_dict())
        _active_storage_url = storage_url
        _active_run_id = run_id
        _active_metadata = meta
        _is_resuming = False
        return meta


def _recordable(value: Any, is_valid, name: str) -> Any:
    if value is None:
        return None
    if not is_valid(value):
        logger.warning(
            "start_run: %s has an unexpected shape and was not recorded; a resume "
            "of this run can't verify its calibration rows",
            name,
        )
        return None
    return value


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
    refuse to proceed and tell the user.

    Raises:
        ResumeVerificationError: while resuming, if the sidecar is missing,
            unreadable or not a JSON object. A resume that cannot be verified
            must stop the run, never count as a match (Codex review of #79).
    """
    if not _is_resuming:
        return True, None
    if not _active_run_id:
        raise ResumeVerificationError("resume is active but has no run id")

    sidecar = _sidecar_path()
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResumeVerificationError(
            f"the resume record {sidecar} no longer exists"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ResumeVerificationError(
            f"the resume record {sidecar} could not be read: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ResumeVerificationError(f"the resume record {sidecar} is not a JSON object")

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

    The sidecar persists; future launches will re-offer it for inspection.
    A data-fingerprint mismatch no longer uses this: the GUI keeps the resume
    pending or, on the user's choice, calls `abandon_resume`.

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


def abandon_resume() -> None:
    """Stop resuming the active run without touching any file.

    Used when the user deliberately starts a fresh analysis instead of resuming
    (e.g. the loaded data does not match the interrupted run). Nothing is deleted:
    the old SQLite store stays on disk for normal retention cleanup. The next
    `start_run` generates a new run id and storage path and overwrites the sidecar,
    so the old store can no longer be resumed by accident. If no run is started,
    the sidecar still names the old run and the next launch offers it again.
    """
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming
    with _lock:
        _active_storage_url = None
        _active_run_id = None
        _active_metadata = None
        _is_resuming = False


def get_active_run_id() -> str | None:
    """Run id of the active (started or resumed) run, or ``None``."""
    return _active_run_id


def get_resumed_run() -> RunMetadata | None:
    """Metadata of the run being resumed, or ``None`` when not resuming."""
    return _active_metadata if _is_resuming else None


def find_incomplete_run() -> RunMetadata | None:
    """Look for a sidecar from a previously crashed/aborted run.

    Returns the metadata if one exists, else None. Does NOT modify state —
    the GUI calls this on startup to decide whether to show the resume
    dialog. The actual resume happens via `resume_run(run_id)`.

    Codex meta-review A2: `OSError` / `PermissionError` from `read_text()`
    bubble up — a locked or unreadable sidecar is a caller-visible decision
    (start fresh? abort? retry?), not something the library should swallow.

    Raises:
        CorruptRunRecordError: the sidecar exists but is not valid JSON, not
            a JSON object, or lacks/mistypes a required field. The file is
            NOT touched. It used to be renamed to `.corrupt` (or deleted when
            the rename failed) and reported as "no run", so the caller's next
            fresh run silently replaced it (Codex review of #79 round 8).
    """
    sidecar = _sidecar_path()
    try:
        raw = sidecar.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise CorruptRunRecordError(sidecar, f"not UTF-8 text ({exc})") from exc
    return _parse_run_record(sidecar, raw)
    # OSError (permission, locked file, dead network share) intentionally
    # escapes — the GUI wraps this in its own handler and tells the user.


def _parse_run_record(path: Path, raw: str) -> RunMetadata:
    """Parse sidecar text into metadata, or raise `CorruptRunRecordError`."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptRunRecordError(path, f"not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise CorruptRunRecordError(path, "not a JSON object")
    for key in ("run_id", "storage_path", "storage_url", "started_iso"):
        if not isinstance(data.get(key), str):
            raise CorruptRunRecordError(path, f"'{key}' is missing or not text")
    if not data["run_id"]:
        raise CorruptRunRecordError(path, "'run_id' is empty")
    names = data.get("model_names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise CorruptRunRecordError(path, "'model_names' is missing or not a list of names")
    n_trials = data.get("n_trials_per_model")
    if n_trials is not None and (not isinstance(n_trials, int) or isinstance(n_trials, bool)):
        raise CorruptRunRecordError(path, "'n_trials_per_model' is not a whole number")
    try:
        return RunMetadata.from_dict(data)
    except (TypeError, KeyError, ValueError) as exc:
        raise CorruptRunRecordError(path, str(exc)) from exc


def set_aside_corrupt_run_record() -> Path | None:
    """Move a damaged saved-run record out of the way, keeping its contents.

    Renames the sidecar to a new, unique ``active_run.corrupt-<timestamp>.json``
    next to it (never overwriting an earlier one), so nothing is lost and a
    new run can be recorded. Only acts if the record is STILL damaged when
    re-read: another dasp window may have replaced it with a valid record in
    the meantime, which must survive.

    Returns:
        The new path, or None when there was nothing damaged to move (no
        record, or it is now valid).

    Raises:
        OSError: the record could not be read or renamed.
    """
    sidecar = _sidecar_path()
    with _lock:
        try:
            raw = sidecar.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError:
            pass  # damaged — move it
        else:
            try:
                _parse_run_record(sidecar, raw)
                return None
            except CorruptRunRecordError:
                pass
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for attempt in range(100):
            suffix = f"-{attempt}" if attempt else ""
            target = sidecar.with_name(f"active_run.corrupt-{stamp}{suffix}.json")
            if target.exists():
                continue
            sidecar.rename(target)
            logger.warning("Moved damaged saved-run record aside to %s", target)
            return target
        raise OSError(f"no free name to set aside {sidecar}")


def has_resumable_store(meta: RunMetadata) -> bool:
    """Whether an incomplete run left a SQLite store that could hold trials.

    False for a run started under 'never' (no storage URL) and for an 'auto' run
    that crashed during its in-memory warmup (the file is only created when a study
    migrates). The GUI uses this to skip the "Resume previous run?" prompt when there
    is nothing to resume. It does not validate the path; ``resume_run`` still does.
    The sidecar is deliberately left alone, never deleted here: the next Bayesian
    run's ``start_run`` replaces it. A check-then-delete cannot be made safe across
    processes, because another window may write its own sidecar in between (Codex
    review of #79).
    """
    if not meta.storage_url or not meta.storage_path:
        return False
    try:
        st = Path(meta.storage_path).stat()
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        # Unknown is not "absent": prompt, and let resume_run decide and report.
        return True
    return stat.S_ISREG(st.st_mode) and st.st_size > 0


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

    try:
        meta = find_incomplete_run()
    except CorruptRunRecordError as exc:
        logger.warning("Resume refused: %s", exc)
        return None
    if meta is None or meta.run_id != run_id:
        return None

    optuna_dir = get_user_optuna_dir().resolve()
    try:
        storage_path = Path(meta.storage_path).resolve()
    except (OSError, ValueError) as exc:
        # ValueError: e.g. an embedded NUL from a corrupted/tampered sidecar.
        logger.warning("Resume refused: sidecar storage path is unusable: %s", exc)
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

    Codex review of #79 round 8: the initial `find_incomplete_run()` call
    above and the `sidecar.unlink()` below are not atomic. Another dasp
    instance can replace the sidecar with its OWN run's between the two —
    the same class of cross-process race already fixed for the startup
    cleanup path (round 2: no automatic delete there at all, because a
    check-then-delete can't be made safe). Here the delete is the whole
    point, so instead the run id is re-read right before unlinking, and the
    unlink is refused if it no longer matches — the replacement sidecar
    (and the run it names) survives untouched.
    """
    try:
        meta = find_incomplete_run()
    except CorruptRunRecordError as exc:
        return DiscardResult(sidecar_deleted=False, storage_deleted=False, errors=[str(exc)])
    if meta is None or meta.run_id != run_id:
        return DiscardResult(sidecar_deleted=False, storage_deleted=False, errors=[])

    sidecar = _sidecar_path()
    optuna_dir = get_user_optuna_dir().resolve()
    errors: list[str] = []

    # Codex review of #79 round 9: the store goes first. Deleting the record
    # first and then failing on a locked store left a store no retry could find
    # (the record naming it was gone). A store already absent counts as deleted,
    # so a retry after a half-finished delete can complete.
    storage_deleted = False
    storage_retryable_failure = False
    try:
        storage_path = Path(meta.storage_path).resolve()
    except OSError as e:
        # Transient (e.g. an unavailable drive): keep the record for a retry.
        errors.append(f"storage_path resolve failed: {e}")
        storage_retryable_failure = True
    except ValueError as e:  # e.g. an embedded NUL: retrying can't help
        errors.append(f"storage path unusable: {e}")
    else:
        if not storage_path.is_relative_to(optuna_dir):
            errors.append(
                f"storage_path outside optuna dir, refusing to unlink: {storage_path}"
            )
        else:
            try:
                storage_path.unlink(missing_ok=True)
                storage_deleted = True
            except OSError as e:
                errors.append(f"storage unlink failed: {e}")
                storage_retryable_failure = True
            except ValueError as e:  # e.g. an embedded NUL: retrying can't help
                errors.append(f"storage path unusable: {e}")

    sidecar_deleted = False
    if storage_retryable_failure:
        # Keep the record so Delete can be retried once the lock clears.
        return DiscardResult(sidecar_deleted=False, storage_deleted=False, errors=errors)
    try:
        if sidecar.exists():
            try:
                current = json.loads(sidecar.read_text(encoding="utf-8"))
                current_run_id = current.get("run_id") if isinstance(current, dict) else None
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                current_run_id = None
            if current_run_id != run_id:
                errors.append(
                    "sidecar no longer names this run (replaced by another "
                    "dasp instance); refusing to delete it"
                )
            else:
                sidecar.unlink()
                sidecar_deleted = True
    except OSError as e:
        errors.append(f"sidecar unlink failed: {e}")

    return DiscardResult(
        sidecar_deleted=sidecar_deleted,
        storage_deleted=storage_deleted,
        errors=errors,
    )


def cleanup_old_sqlite_files() -> tuple[int, int]:
    """Best-effort cleanup of stale Optuna SQLite trial archives.

    Old `<run_id>.sqlite3` files accumulate in ``<user_data_dir>/dasp/optuna/``
    because :func:`mark_complete` deliberately leaves them for post-hoc
    inspection. ~50 MB/month under normal use; this function bounds the leak.

    Policy:
      * Keep the ``_KEEP_LAST_N_RUNS`` newest by mtime.
      * Additionally delete anything older than ``_DELETE_AFTER_DAYS`` (overrides
        keep-N — a single 31-day-old file is fair game).
      * Never touch the active run's storage_path (read from active_run.json).
      * Never touch a file whose ``-wal`` sibling was modified within the last
        ``_WAL_SAFETY_WINDOW_HOURS`` (concurrent dasp instance safeguard).
      * For each deleted ``.sqlite3``, also unlink matching ``-shm`` / ``-wal``.

    Returns
    -------
    tuple[int, int]
        ``(files_deleted, bytes_freed)``. Per-file failures (file locked,
        permission denied) are logged at WARNING and the count under-reports;
        the function never blocks app startup.
    """
    try:
        optuna_dir = get_user_optuna_dir()
    except OSError as exc:
        logger.warning("T-50: cannot resolve optuna dir: %s", exc)
        return (0, 0)

    if not optuna_dir.exists():
        return (0, 0)

    active_storage = _read_active_storage_path_safely(optuna_dir)
    now = datetime.now()
    cutoff_age = now - timedelta(days=_DELETE_AFTER_DAYS)
    wal_safety = now - timedelta(hours=_WAL_SAFETY_WINDOW_HOURS)

    active_key = (
        os.path.normcase(str(active_storage)) if active_storage is not None else None
    )
    candidates: list[tuple[Path, datetime]] = []
    for path in optuna_dir.glob("*.sqlite3"):
        if active_key is not None:
            try:
                path_key = os.path.normcase(str(path.resolve()))
            except OSError:
                path_key = os.path.normcase(str(path))
            if path_key == active_key:
                continue
        wal = path.with_suffix(".sqlite3-wal")
        if wal.exists():
            try:
                wal_mtime = datetime.fromtimestamp(wal.stat().st_mtime)
            except OSError:
                continue  # can't stat sibling — be conservative, skip
            if wal_mtime > wal_safety:
                continue  # likely active in another dasp instance
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        candidates.append((path, mtime))

    # Newest first so the keep-set is the K most recent.
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    keep = {p for p, _ in candidates[:_KEEP_LAST_N_RUNS]}
    to_delete = [
        (p, m) for (p, m) in candidates
        if p not in keep or m < cutoff_age
    ]

    deleted = 0
    bytes_freed = 0
    for path, _ in to_delete:
        try:
            size = path.stat().st_size
        except OSError as exc:
            logger.warning("T-50: cannot stat %s: %s", path, exc)
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("T-50: could not delete stale SQLite %s: %s", path, exc)
            continue
        deleted += 1
        bytes_freed += size
        for sibling in (
            path.with_suffix(".sqlite3-shm"),
            path.with_suffix(".sqlite3-wal"),
            # GLM M2 (T-50 fix-of-fixes): when WAL is rejected (network share,
            # AV-shimmed path — see _apply_wal_pragmas in unified_bayesian.py),
            # SQLite falls back to rollback-journal mode, leaving orphan
            # `<id>.sqlite3-journal` files. Cleanup the parent without these
            # would leak the journal forever.
            path.with_suffix(".sqlite3-journal"),
        ):
            try:
                sibling_size = sibling.stat().st_size
                sibling.unlink()
                bytes_freed += sibling_size
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "T-50: could not delete sibling %s: %s", sibling, exc
                )
    return (deleted, bytes_freed)


def _read_active_storage_path_safely(optuna_dir: Path) -> Path | None:
    """Read ``storage_path`` from the active-run sidecar without raising.

    Returns the resolved Path of the SQLite file the active run owns, or None
    if there is no sidecar / the sidecar is malformed / the path can't be
    resolved / the resolved path escapes ``optuna_dir`` (tampered sidecar).
    Cleanup callers use this to skip the active file.
    """
    sidecar = optuna_dir / _SIDECAR_NAME
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    raw = data.get("storage_path") if isinstance(data, dict) else None
    if not isinstance(raw, str) or not raw:
        return None
    try:
        resolved = Path(raw).resolve()
    except OSError:
        return None
    # DeepSeek M2 / GLM L3 (T-50 fix-of-fixes): match the defense-in-depth
    # posture of resume_run / discard_incomplete_run, which both refuse to
    # trust a sidecar storage_path that resolves outside the optuna dir.
    # A tampered sidecar pointing at /etc/passwd wouldn't match any glob
    # candidate anyway (no functional gap today), but the inconsistency is
    # a refactor hazard.
    try:
        if not resolved.is_relative_to(optuna_dir.resolve()):
            return None
    except OSError:
        return None
    return resolved


def _reset_for_tests() -> None:
    """Test-only reset of module-level state. Do not call from production code."""
    global _active_storage_url, _active_run_id, _active_metadata, _is_resuming
    with _lock:
        _active_storage_url = None
        _active_run_id = None
        _active_metadata = None
        _is_resuming = False
