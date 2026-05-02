# T-46: `_apply_wal_pragmas` return value discarded at both call sites

**Status:** PLANNED — surgical fix, ~10 LOC.
**Filed:** 2026-05-02.
**Source:** pr-review-toolkit silent-failure-hunter MEDIUM finding during the consolidated T-43+T-42+T-38 review (also noted by DeepSeek in the T-41 review trail; deferred at the time).
**Priority:** MEDIUM — affects users on filesystems that reject WAL (network shares, AV-shimmed paths). Rarely fires but completely silent when it does.

## Problem

`src/spectral_predict/unified_bayesian.py:_apply_wal_pragmas(storage_url) -> bool` explicitly returns `False` when the SQLite filesystem rejects WAL mode. The function's docstring says:

> "Caller should surface that — performance silently halves under journal mode DELETE."

Both call sites discard the return value:

- `unified_bayesian.py:2108` — `_apply_wal_pragmas(storage_url)` (no assignment, in `run_unified_bayesian`).
- `unified_bayesian.py:1723` — `_apply_wal_pragmas(sqlite_url)` (no assignment, inside `_migrate_study_to_sqlite`).

The author of the helper documented that callers must surface the False case. They don't.

## Failure mode hidden today

User's `<user_data_dir>` is on a OneDrive- or Dropbox-synced location. SQLite refuses to apply `PRAGMA journal_mode=WAL` (network filesystems often don't support file locking semantics WAL needs); journal_mode silently falls back to DELETE. Bayesian runs are slower (each commit is full fsync rather than batched WAL append), and concurrent dasp instances are more likely to hit "database is locked" errors mid-trial.

Without surfacing the WAL rejection, the user sees:

- Slower-than-expected Bayesian search.
- Sporadic database-locked errors on multi-instance use.
- No clue why — the auto-calculator picked SQLite, the persistence radio is "always," everything looks fine.

## Fix

Capture the return at both call sites and surface via the progress callback:

```python
# unified_bayesian.py:2108 (run_unified_bayesian)
if not _apply_wal_pragmas(storage_url):
    if progress_callback is not None:
        progress_callback({
            "type": "warning",
            "message": (
                "WAL journal mode rejected by the filesystem — Bayesian "
                "search will run with the slower DELETE journal mode. "
                "Move the dasp data directory to a local (non-synced) "
                "drive for best performance."
            ),
        })
```

```python
# unified_bayesian.py:1723 (_migrate_study_to_sqlite)
if not _apply_wal_pragmas(sqlite_url):
    logger.warning("...")  # complement T-45 once that ships
```

The `_migrate_study_to_sqlite` site doesn't have a progress_callback in scope — log to the logger and let T-45's file handler surface it.

## Verification

1. Mock `_apply_wal_pragmas` to return `False`.
2. Run a tiny Bayesian search with a `progress_callback` capturing all events.
3. Assert at least one event has `type == 'warning'` and the message mentions WAL rejection.

Add the test to `tests/test_t41_bayesian_sqlite_auto_calculator.py`.

## Out of scope

- Auto-detecting OneDrive/Dropbox paths and warning at app launch (a different ticket — would need path-introspection heuristics).
- Falling back to a known-good local-only data directory if the user's pick doesn't support WAL (would change persistence semantics for the user; probably unwanted).
