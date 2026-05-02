# T-50: Auto-cleanup old `*.sqlite3` files in `<user_data_dir>/dasp/optuna/`

**Status:** PLANNED — small infrastructure ticket, ~30 LOC.
**Filed:** 2026-05-02 (during overnight queue session, after user inspected their optuna dir and saw 6 SQLite files accumulated from runs Apr 30 → May 2).
**Source:** User question during T-44 manual-test session: "is there a mechanism to delete these once runs are completed?" Answer was no — `mark_complete()` in `run_state.py:420-441` deliberately preserves the SQLite for post-hoc inspection, with the docstring explicitly punting cleanup as "the user's responsibility (out of T-11 D scope)."
**Priority:** LOW-MEDIUM — silent disk leak, real but slow. ~14 MB after 3 days of normal use; would grow to hundreds of MB over months of regular sessions.

## Problem

`run_state.mark_complete()` unlinks the sidecar (`active_run.json`) on clean completion but **deliberately leaves the SQLite trial-archive file in place**:

```python
# run_state.py:437-440
"""
Leaves the SQLite file in place so the user can inspect Optuna study
contents post-hoc. Old SQLite files accumulate in
`<user_data_dir>/dasp/optuna/`; manual cleanup or a future scheduled
cleanup is the user's responsibility (out of T-11 D scope).
"""
```

Each completed Bayesian/grid run drops a `<run_id>.sqlite3` (1–2 MB typical, larger for verbose multi-model runs). With ~2–3 runs per session and weekly use, the dir grows by ~50 MB/month. There is no UI to surface this — the user has to know about the dir to find it.

The "post-hoc inspection" justification is real but rare: most users never open Optuna dashboard against an old run. The cost of the leak is paid by every user; the benefit accrues to the few who do post-hoc analysis.

## Concrete numbers from a real user dir (2026-05-02 snapshot)

```
1.7 MB  91e62c03bf65.sqlite3       (May 2)
1.7 MB  94e77a265d38.sqlite3       (May 1)
1.7 MB  9e1d4260faac.sqlite3       (Apr 30)
0.7 MB  a0032db1fb65.sqlite3       (May 2 — currently active)
4.0 MB  a0032db1fb65.sqlite3-wal   (active run's WAL)
1.7 MB  b6902a7b82e2.sqlite3       (May 1)
2.0 MB  eb872b9aa9f4.sqlite3       (May 1 — abandoned, has WAL)
4.1 MB  eb872b9aa9f4.sqlite3-wal   (abandoned run's WAL)
```

~17 MB total in 3 days of light use. The two abandoned `*-wal` files are particularly noteworthy — they're orphan WAL journals from interrupted runs that even Optuna's own resume path won't pick up.

## Fix

Add a `cleanup_old_sqlite_files()` function in `run_state.py` that runs at app startup (called from the GUI's `main()` and `cli.main()`, same wiring shape as T-45's `setup_app_logger`).

### Policy

- Keep last **K = 5** completed runs (most-recent by mtime).
- Additionally delete anything older than **N = 30 days** regardless of count.
- **Never delete files associated with the active run** — `active_run.json`, if present, names the active SQLite via its `storage_path` field; that file plus its `*-shm` / `*-wal` siblings are off-limits.
- **Never delete a file with a `*-wal` sibling whose mtime is within the last hour** — concurrent dasp instance safeguard. WAL files mid-rollover should not be touched.
- For each deleted `<run_id>.sqlite3`, also delete the matching `<run_id>.sqlite3-shm` and `<run_id>.sqlite3-wal` if present.

### Implementation sketch

```python
# run_state.py

_KEEP_LAST_N_RUNS = 5
_DELETE_AFTER_DAYS = 30
_WAL_SAFETY_WINDOW_HOURS = 1


def cleanup_old_sqlite_files() -> tuple[int, int]:
    """Best-effort cleanup of stale Optuna SQLite trial archives.

    Returns (files_deleted, bytes_freed). Failures are logged at WARNING
    and the count under-reports — the cleanup never blocks app startup.

    Policy: keep last K completed runs by mtime, plus delete anything
    older than N days. Skip the currently-active run's storage_path and
    skip any file whose -wal sibling was modified in the last hour
    (concurrent-instance safeguard).
    """
    optuna_dir = get_user_optuna_dir()
    active_storage = _read_active_storage_path_safely(optuna_dir)
    cutoff_age = datetime.now() - timedelta(days=_DELETE_AFTER_DAYS)
    wal_safety = datetime.now() - timedelta(hours=_WAL_SAFETY_WINDOW_HOURS)

    candidates = []
    for path in optuna_dir.glob("*.sqlite3"):
        if path == active_storage:
            continue
        wal = path.with_suffix(".sqlite3-wal")
        if wal.exists():
            wal_mtime = datetime.fromtimestamp(wal.stat().st_mtime)
            if wal_mtime > wal_safety:
                continue  # active in another instance
        candidates.append((path, datetime.fromtimestamp(path.stat().st_mtime)))

    # Sort newest first; everything past the keep-N or older than cutoff is fair game.
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    keep = set(p for p, _ in candidates[:_KEEP_LAST_N_RUNS])
    to_delete = [
        (p, m) for (p, m) in candidates
        if p not in keep or m < cutoff_age
    ]

    deleted = 0
    bytes_freed = 0
    for path, _ in to_delete:
        try:
            size = path.stat().st_size
            path.unlink()
            for sibling in (
                path.with_suffix(".sqlite3-shm"),
                path.with_suffix(".sqlite3-wal"),
            ):
                if sibling.exists():
                    bytes_freed += sibling.stat().st_size
                    sibling.unlink()
            bytes_freed += size
            deleted += 1
        except OSError as exc:
            logger.warning("T-50: could not delete stale SQLite %s: %s", path, exc)
    return deleted, bytes_freed
```

### Wiring

```python
# spectral_predict_gui_optimized.py main() — after setup_app_logger
try:
    from spectral_predict.run_state import cleanup_old_sqlite_files
    deleted, freed = cleanup_old_sqlite_files()
    if deleted:
        logger.info("T-50: cleaned up %d stale Optuna SQLite files (%.1f MB)",
                    deleted, freed / 1_048_576)
except Exception:
    pass  # best-effort
```

Same call from `cli.main()` per T-45's three-startup-path lesson.

## Tests

In `tests/test_run_state.py`:

1. **`test_cleanup_keeps_last_k`** — populate the optuna dir with 8 fake `*.sqlite3` files with monotonically increasing mtimes; call `cleanup_old_sqlite_files()`; assert 3 oldest were deleted, 5 newest survive.
2. **`test_cleanup_skips_active_run_storage`** — write an `active_run.json` pointing at one specific `.sqlite3`; call cleanup with that file as the oldest; assert it survives.
3. **`test_cleanup_skips_recent_wal_sibling`** — populate two old `.sqlite3` files, give one of them a fresh `-wal` sibling (mtime within the safety window); assert that one survives, the other is deleted.
4. **`test_cleanup_unlinks_shm_and_wal_siblings`** — for each deleted `.sqlite3`, confirm matching `-shm` and `-wal` siblings are also gone if they existed.
5. **`test_cleanup_age_cutoff_overrides_keep_count`** — populate 3 files, all older than 30 days; assert all 3 deleted regardless of K=5.
6. **`test_cleanup_swallows_unlink_failure`** — monkeypatch `Path.unlink` to raise OSError on one file; assert the function still returns and logs a warning, and the OTHER files still get deleted.

## Out of scope

- A "Clear cached run history" button in the GUI. Could be added in a follow-up; for now the auto-cleanup covers the silent-leak case.
- Deleting the corresponding `outputs/` files (results CSVs, reports) — those are user-owned; the user already manages them manually.
- Surfacing the cleanup count in the UI status bar. Logger.info covers it for users who care; most don't.
- Compressing old SQLite files instead of deleting. Disk savings are real but the post-hoc-inspection use case is rare enough that compression complexity isn't worth it.

## Acceptance criteria

- After running 8 Bayesian searches in a row, the optuna dir contains exactly 5 `*.sqlite3` files (the most recent 5).
- Active run's SQLite is never touched mid-run.
- Concurrent dasp instances don't step on each other (the WAL-mtime safeguard fires).
- Cleanup runs in <100 ms even with 100+ files in the dir (glob + stat is the bottleneck).

## Dependencies

None. Branches off latest main. Independent of any in-flight work.
