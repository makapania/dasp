# T-41: Bayesian SQLite Auto-Calculator + WAL Mode — Validation Note

**Date:** 2026-05-01
**Branch:** `fix/T41-bayesian-sqlite-auto-calculator`
**Ticket:** T-41
**Plan:** `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md`

---

## Problem

T-11 introduced Optuna SQLite persistence for crash-resume capability. The SQLite
overhead is roughly constant at ~200ms/trial regardless of model. For fast models like
PLS (~30ms/fit), this produces a 8-28x slowdown on a per-trial basis. User reported
"Bayesian is at least 5x slower than it used to be."

## Root cause

Pre-T-41: `start_run()` always generated a SQLite URL; `run_unified_bayesian()` passed
it to `optuna.create_study()` unconditionally. Every trial incurred ~200ms of SQLite
write overhead even for models where that represents 8-10x the actual fit time.

## Empirical benchmark (collected 2026-05-01, Win11, .venv312)

### Per-model overhead (n=10 trials each)

| Model | In-memory | SQLite WAL | WAL ratio | Overhead/trial |
|---|---|---|---|---|
| PLS | 30 ms | 244 ms | **8.13x** | +214 ms |
| Ridge | 21 ms | 215 ms | **10.16x** | +193 ms |
| LightGBM | 276 ms | 520 ms | **1.89x** | +244 ms |
| XGBoost | 534 ms | 728 ms | **1.36x** | +193 ms |

### Storage comparison (n=20 trials, PLS)

| Storage | Wall-clock | Ratio |
|---|---|---|
| In-memory | 0.43s | 1.00x |
| SQLite default | 12.25s | 28.19x |
| SQLite WAL | 3.63s | 8.36x |
| Optuna JournalStorage (file) | 7.04s | 16.19x |

**JournalStorage rejected** — worse than SQLite WAL on Windows due to NTFS cross-process
file locking overhead. SQLite+WAL is the better Windows backend.

## Fix

### Auto-calculator logic

Each `run_unified_bayesian` call decides independently:
- Default `enable_sqlite_persistence='never'`: pure in-memory, zero overhead.
- `'auto'`: first 10 trials in-memory; at trial 10 compute median fit time.
  - If median > 1.0s: migrate to SQLite+WAL (overhead ~1.2x at threshold).
  - If median <= 1.0s: stay in-memory (fast models like PLS).
  - Fallback: if fewer than 3 trials completed, default SQLite ON (conservative).
- `'always'`: SQLite+WAL from trial 0 for all models.

### Threshold justification

At median fit = 1.0s, overhead ratio = `1 + 200ms/1000ms = 1.2x`. User considers <2x
acceptable. The threshold gives comfortable margin while ensuring heavy models (XGBoost,
RandomForest on large datasets) get persistence by default in `'auto'` mode.

### SQLite migration (mid-run)

Uses `optuna.copy_study` (data copy) + `optuna.load_study(sampler=TPESampler(...))`.

**Critical trap**: `optuna.create_study(load_if_exists=True, sampler=...)` SILENTLY
IGNORES the sampler kwarg on existing studies (Optuna 4.8, empirically verified by
DeepSeek V4 Pro Max 2026-05-01). The migration helper explicitly uses `load_study(sampler=...)`
to attach TPE correctly. TPE rebuilds its internal state from copied trials on the next
`optimize()` call.

### WAL mode

Applied at migration time (NOT in `start_run`). `start_run` doesn't create the SQLite
file — Optuna does so lazily on first access. If all models stay in-memory, the file
never exists and applying WAL would be a no-op or error. Migration triggers the file
creation, so WAL pragmas are applied immediately after `copy_study`.

WAL + `synchronous=NORMAL` prevents database corruption but may lose trials between
checkpoints on crash. `wal_autocheckpoint=50` bounds loss to ~50 trials per crash.
Note: `wal_autocheckpoint` is a per-connection setting (not file-persistent);
`journal_mode=WAL` IS file-persistent (set once, applies to all subsequent connections).

### Default value override (user decision)

Plan originally proposed `default='auto'`. User overrode to `default='never'` with
rationale: "99.99% of the time this would not be used" — for their workflow, in-memory
is right almost always; they want zero overhead by default and will opt into `'auto'`
or `'always'` only when they specifically want crash-resume.

Two concrete code changes from the plan:
1. `run_unified_bayesian(enable_sqlite_persistence: str = 'never')` (was `'auto'`)
2. `tk.StringVar(value='never')` in GUI init (was `'auto'`)

## Performance with default `'never'`

With `default='never'`, the T-41 code path short-circuits at the very first line of the
storage-decision block. No warmup timer, no migration, no SQLite file. PLS Bayesian runs
at the same speed as pre-T-11 in-memory (ratio ~1.0x). Verified by
`test_t41_bayesian_sqlite_auto_calculator.py::TestNeverModeNoSQLite`.

## Test results

**45 T-41 + run_state tests** all passing:

```
tests/test_t41_bayesian_sqlite_auto_calculator.py  (16 tests) - all PASSED
tests/test_run_state.py  (29 tests) - all PASSED
```

**91-test regression sweep** all passing:
- `test_unified_bayesian_baseline.py` (regression + classification models)
- `test_t41_bayesian_sqlite_auto_calculator.py`
- `test_cv_pls_clamp.py`
- `test_autoscale_bayesian.py`

## Files changed

| File | Change |
|---|---|
| `src/spectral_predict/unified_bayesian.py` | Added `_apply_wal_pragmas`, `_make_tpe_sampler`, `_migrate_study_to_sqlite` helpers; `enable_sqlite_persistence` parameter; auto-calculator logic in `run_unified_bayesian` |
| `src/spectral_predict/run_state.py` | Added `logging`; `bayesian_persistence_mode` field to `RunMetadata`; updated `start_run` to accept mode + skip SQLite URL for `'never'`; added `_cleanup_empty_sqlite` helper; stale-sidecar cleanup in `mark_complete` and `clear_resume_state` |
| `spectral_predict_gui_optimized.py` | Added `bayesian_persistence_mode` StringVar (default `'never'`); 3-way radio buttons (Auto/Always on/Always off) + tooltip; both `run_unified_bayesian` call sites updated; `start_run` call updated |
| `tests/test_t41_bayesian_sqlite_auto_calculator.py` | 16 new tests (plan called for 11+; extras cover run_state 'never' URL contract) |
| `tests/test_run_state.py` | 3 pre-existing tests updated to pass `bayesian_persistence_mode='auto'` where they assert `storage_url` is non-empty (correct update — those tests test the persistence machinery, so they should opt in) |
| `docs/bugfix_validation/T41_bayesian_sqlite_auto_calculator.md` | This file |
| `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md` | Updated to record default-off override + user rationale |
| `docs/PROJECT_STATUS.md` | Updated |
| `docs/SESSION_LOG.md` | Entry added |
