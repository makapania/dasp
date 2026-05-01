# T-41: Bayesian SQLite Auto-Calculator + WAL Mode

> **For agentic workers:** Implementation work; not a refactor. Ship as a single-PR ticket on a branch off post-T-36 main.

**Goal:** Eliminate the user-visible Bayesian slowdown introduced by T-11's Optuna SQLite persistence, by making storage choice automatic per-model, defaulting to in-memory for fast models and SQLite (with WAL mode) for heavy models. Provide a 3-way manual override (`Auto / Always on / Always off`).

**Tech Stack:** Optuna 4.8 (`optuna.copy_study`, `JournalStorage`/`RDBStorage`), sqlite3 (`PRAGMA journal_mode=WAL`), existing dasp `run_state` machinery.

**Source:** Conversation 2026-05-01 — user reported "Bayesian is at least 5× slower than it used to be" (PLS specifically — actually 28× slower). Codex investigation traced to T-11's per-trial SQLite writes. Benchmark (`tests/_bench_bayesian_per_model.py`) showed overhead is roughly constant ~200ms/trial regardless of model — fast models pay 8-10×, heavy models pay 1.36-1.89×.

---

## Background

### Empirical data (collected 2026-05-01 on Win11, .venv312)

`tests/_bench_bayesian_per_model.py`, n_trials=10 per condition:

| Model | In-memory | SQLite default | SQLite WAL | WAL ratio | Overhead/trial |
|---|---|---|---|---|---|
| PLS | 30 ms | 600 ms | 244 ms | **8.13×** | +214 ms |
| Ridge | 21 ms | n/a | 215 ms | **10.16×** | +193 ms |
| LightGBM | 276 ms | n/a | 520 ms | **1.89×** | +244 ms |
| XGBoost | 534 ms | n/a | 728 ms | **1.36×** | +193 ms |
| RandomForest | 1490 ms | n/a | 573 ms | 0.38× (anomaly) | n/a (noise dominates) |

`tests/_bench_bayesian_sqlite.py`, n_trials=20 PLS:

| Storage | Wall-clock | Ratio |
|---|---|---|
| In-memory | 0.43s | 1.00× |
| SQLite default | 12.25s | 28.19× |
| SQLite WAL | 3.63s | 8.36× |
| Optuna JournalStorage (file) | 7.04s | 16.19× |

**Two findings:**

1. SQLite overhead is roughly constant ~200ms/trial regardless of model. Slowdown ratio = `1 + (200ms / fit_time_ms)`. Below ~1000ms fit time, ratio exceeds 1.2× and starts to bite.
2. WAL mode is 3.4× faster than default SQLite on Windows for this workload. Free win — should always be used for the SQLite path.

**Mixed-model runs are already per-model.** The GUI Bayesian loop (`spectral_predict_gui_optimized.py:27317-27346`) iterates `for model_name in selected_models` and calls `run_unified_bayesian` once per model. Each call is its own Optuna study. So per-model decisions are the natural unit; no cross-model coordination needed.

### Why JournalStorage was worse than SQLite WAL

JournalFileBackend uses cross-process file locking which is expensive on NTFS. RDB SQLite + WAL is the better Windows backend. Surprising but the data is clear.

### What T-11's user-attr writes look like

Each Bayesian trial calls `set_user_attr` ~25 times for metadata persistence (preprocessing fields, model params, n_vars, subset_tag, all_wavelengths, etc.). Per-call overhead ~5-10ms with default SQLite, ~2ms with WAL. Plus the trial-state COMMIT. Total per-trial finalize cost ~200ms with WAL. This irreducible overhead is what the auto-calculator works around.

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| Per-model storage decision (each `run_unified_bayesian` call decides independently) | The GUI already loops over models. Mixed-model runs (PLS + XGBoost) get the right behavior per model automatically — no shared global state to reason about. |
| Auto-calculator with first-5-trial timing window | Real fit times depend on dataset size (n_samples, n_features) and CV folds, not just model name. Measuring first 5 trials is more accurate than a hardcoded model→speed table. |
| In-memory for first 5 trials regardless of decision | (a) too short to bother setting up SQLite, (b) we don't know the decision yet. Crash window = ~5 trials worth of work, acceptable. |
| `optuna.copy_study(in_mem → sqlite)` for migration | Optuna's built-in primitive. Preserves all trial state, params, user_attrs, and lets the same TPE sampler continue. |
| WAL mode always when SQLite is active | 3.4× faster than default. No correctness tradeoff — WAL is a journaling mode, not a relaxed durability mode. (`PRAGMA synchronous=NORMAL` is the normal companion; still crash-safe under WAL semantics.) |
| 3-way GUI override `Auto / Always on / Always off` | Most users want auto. Power users with long heavy-model runs may want always-on for safety. Speed-sensitive users may want always-off. Three radio buttons, one tooltip. |
| Threshold = mean fit time > 1000ms → SQLite ON | At 1000ms fit time, ratio is `1 + 200/1000 = 1.2×`. User considers <2× acceptable; this is well within. |
| Decision logged to progress callback | Surface the auto-calc decision so users see "Auto-enabled SQLite (mean trial = 4.2s)" or "Auto-disabled SQLite (mean trial = 0.04s, would slow run by 6×)". |
| `run_state` machinery becomes the *default* for the session | The auto-calculator can override per-call. `run_state.get_storage_url()` still works for callers (one-class Bayesian, regression Bayesian) but the auto-calculator decides whether to honor it. |

---

## User-facing behavior (concrete examples)

| Scenario | Behavior |
|---|---|
| User runs PLS Bayesian, 100 trials, default `Auto` | First 5 trials in-memory (~150ms). Decision: in-memory (mean fit ~30ms < 1000ms). Remaining 95 trials in-memory. Total ~3s. **No resume capability** but matches pre-T-11 speed. |
| User runs XGBoost Bayesian, 100 trials, default `Auto` | First 5 trials in-memory (~3s). Decision: SQLite ON (mean fit ~600ms; ratio with overhead = 1.33×). Migrate study to SQLite + WAL. Remaining 95 trials with persistence. Total ~75s vs ~55s in-mem-only. **Resume works.** |
| User selects PLS + XGBoost, default `Auto` | PLS run: in-memory throughout (~3s). XGBoost run: SQLite ON after auto-decision. Each model's persistence decision is independent. |
| User explicitly picks `Always on` | All trials in SQLite + WAL from trial 1, even fast models. PLS will be 8× slower, user accepts. |
| User explicitly picks `Always off` | All trials in-memory always. No resume even for heavy models. |
| Crash mid-run, `Auto` mode picked SQLite | Resume works as today — Optuna re-opens the SQLite study, picks up where left off. |
| Crash mid-run, `Auto` mode stayed in-memory | No resume. User loses the run. (Acceptable because in-mem is reserved for fast models where re-run is cheap.) |
| Crash during first 5 trials (before decision) | No resume regardless of mode. (Acceptable; few seconds of work lost.) |

---

## Tasks

### Task 1: `unified_bayesian.run_unified_bayesian` — accept new parameter

- [ ] Add `enable_sqlite_persistence: str = 'auto'` to signature (after `enable_autoscale`).
  - Values: `'auto'`, `'always'`, `'never'`.
- [ ] Plumb to `create_unified_objective` (no behavior change there yet — just parameter pass-through).
- [ ] Update docstring.

### Task 2: `unified_bayesian` — auto-decision logic

- [ ] After 5 completed trials in `study.optimize` callback, compute `mean_trial_time = mean(t.duration.total_seconds() for t in last 5 completed trials)`.
- [ ] If `enable_sqlite_persistence == 'auto'`:
  - If mean > 1.0s AND `run_state.get_storage_url()` is None: skip migration (user disabled persistence at session level — respect that).
  - If mean > 1.0s AND storage URL is set: migrate the in-memory study to SQLite via `optuna.copy_study`, recreate the sampler with the existing trials, continue.
  - If mean ≤ 1.0s: stay in-memory for the rest of the run; **release the run_state SQLite handle so the file isn't held open uselessly**.
- [ ] If `enable_sqlite_persistence == 'always'`: skip the timing window; create the SQLite study from trial 0.
- [ ] If `enable_sqlite_persistence == 'never'`: stay in-memory; ignore `run_state.get_storage_url()`.
- [ ] Log the decision to `progress_callback` so users see what's happening.

### Task 3: `run_state` — WAL mode when creating the SQLite file

- [ ] In `start_run` (or wherever the .db file is first opened), after creating the file but before Optuna touches it:
  ```python
  import sqlite3
  conn = sqlite3.connect(str(db_path))
  conn.execute("PRAGMA journal_mode = WAL")
  conn.execute("PRAGMA synchronous = NORMAL")
  conn.close()
  ```
- [ ] WAL setting persists with the file — no need to repeat on resume.
- [ ] Document in the docstring that WAL is always used for new files; resume of a non-WAL file from a pre-T-41 dasp will continue using whatever journal mode the file was created with (don't fight existing files).

### Task 4: Migration helper

- [ ] Helper `_migrate_study_to_sqlite(study, sqlite_url, study_name) -> optuna.Study`:
  - `optuna.copy_study(from_study_name=study.study_name, to_study_name=study_name, from_storage=study._storage, to_storage=sqlite_url)`
  - Re-load the study from the new storage and return it.
  - Recreate the TPE sampler if needed (Optuna typically does this transparently, but verify).
- [ ] Test that the migrated study continues optimizing correctly (TPE state preserved across the migration).

### Task 5: GUI — 3-way radio button

- [ ] In Bayesian config section of `_create_tab4a_basic_settings` (or wherever the Bayesian-specific options live — verify location), add:
  ```python
  self.bayesian_persistence_mode = tk.StringVar(value='auto')
  ttk.Label(parent, text="Crash-resume persistence:", style='Subheading.TLabel').grid(...)
  for value, text in [('auto', 'Auto (recommended)'), ('always', 'Always on'), ('never', 'Always off')]:
      ttk.Radiobutton(parent, text=text, variable=self.bayesian_persistence_mode, value=value).grid(...)
  ```
- [ ] Tooltip explains the auto-calculator math (~200ms/trial overhead, threshold at 1s mean fit time, mention that fast models like PLS run 8× slower with persistence so 'always on' is rarely the right choice).
- [ ] Both `run_unified_bayesian` call sites in the GUI (one-class @ ~27003 and regression/cls @ ~27345) get the new kwarg: `enable_sqlite_persistence=self.bayesian_persistence_mode.get()`.

### Task 6: Persist the user's choice in `run_state` start_run + .dasp metadata

- [ ] When the user picks `Always off`, `start_run` should not create a SQLite file at all (saves I/O and avoids confusion if the user inspects the run folder).
- [ ] Add `bayesian_persistence_mode: str` to `start_run` metadata so it's logged for audit.
- [ ] No need to add to saved-model metadata (T-36 lesson — this is a per-run setting, not a per-model property).

### Task 7: Tests

- [ ] `test_t41_auto_calculator_picks_in_memory_for_pls`:
  - Run a tiny PLS Bayesian study with `enable_sqlite_persistence='auto'`.
  - Assert that the study's storage is `InMemoryStorage` after completion.
  - Assert that no `.db` file was created in the run dir.
- [ ] `test_t41_auto_calculator_picks_sqlite_for_slow_model`:
  - Run a tiny RandomForest Bayesian (or mock the trial duration to >1s).
  - Assert that the study migrates to SQLite by trial 6.
  - Assert that the .db file exists and contains all completed trials.
- [ ] `test_t41_always_on_creates_sqlite_from_trial_zero`:
  - Run a tiny PLS Bayesian with `'always'`.
  - Assert SQLite is used from the start (verify via `study._storage` type).
- [ ] `test_t41_always_off_never_creates_sqlite`:
  - Run a tiny RandomForest Bayesian with `'never'`.
  - Assert no `.db` file is created.
- [ ] `test_t41_wal_mode_set_on_new_file`:
  - Use `start_run` to create a SQLite file; query `PRAGMA journal_mode` — assert returns `'wal'`.
- [ ] `test_t41_migration_preserves_trial_count`:
  - Manually migrate a 5-trial in-memory study to SQLite.
  - Reload from SQLite, assert all 5 trials present with same params.
- [ ] `test_t41_resumable_after_migration`:
  - Migrate mid-run; close the session; reopen via `load_if_exists`; verify trial count and best value.
- [ ] Benchmark regression: run `tests/_bench_bayesian_per_model.py` with the auto-calculator enabled; assert PLS ratio < 1.2× and XGBoost ratio < 2.0× (loose thresholds for CI noise tolerance).

### Task 8: Validation note

- [ ] `docs/bugfix_validation/T41_bayesian_sqlite_auto_calculator.md`:
  - Re-run the benchmark with the auto-calculator and document the new ratios.
  - Document the auto-decision threshold choice (1.0s mean fit time) and the math justifying it.
  - Note that JournalStorage was tested and rejected (16× ratio on Windows due to NTFS file locking).

### Task 9: PR

- [ ] Branch `fix/T41-bayesian-sqlite-auto-calculator`, branched from main (after T-36 lands) — NOT from `feature/T36-autoscale-toggle`.
- [ ] Targeted test sweep: `pytest tests/test_unified_bayesian*.py tests/test_t41_*.py tests/test_cv_pls_clamp.py -v`.
- [ ] Multi-pass review: DeepSeek per-phase + Codex final, mirror T-36 review pattern.
- [ ] Open PR with summary referencing this plan + benchmark numbers.

---

## Verification checklist

- [ ] PLS Bayesian on `Auto` runs at full in-memory speed (≤ 1.2× of pre-T-11).
- [ ] XGBoost Bayesian on `Auto` migrates to SQLite + WAL after first 5 trials and is resumable on crash.
- [ ] LightGBM/CatBoost on `Auto` makes the right call based on actual fit time on the user's dataset (not a hardcoded table).
- [ ] `Always on` works for all models (pays the speed cost; gains universal resume).
- [ ] `Always off` works for all models (gives up resume; gains speed even on heavy models — useful when user is sure they won't crash).
- [ ] Mixed-model run (PLS + XGBoost in same Bayesian session) gets the right per-model decision.
- [ ] WAL mode is set on every new SQLite file; existing files inherit their journal mode.
- [ ] Resume after crash works for both pre-migration (in-memory) and post-migration (SQLite) states.
- [ ] No new code paths affect grid-search performance (T-41 is Bayesian-only).

## Out-of-scope (file as separate tickets if needed)

- Periodic-snapshot architecture (run fully in-memory, snapshot to disk every N seconds in background thread). Was the leading candidate before benchmark data showed the simple per-model auto-calc is sufficient. File as T-42 if the auto-calc proves inadequate in practice.
- Optuna's `JournalStorage` as primary backend. Tested and rejected — slower than SQLite WAL on Windows.
- Cross-platform tuning (Linux, macOS) — likely faster fsync on those platforms could shift the threshold. Test if dasp ever expands beyond Windows.
- One-class Bayesian deserves the same treatment but is small enough that current usage may be fine. Verify in benchmark and bundle if cheap.

## Known follow-ups

- T-42 (periodic snapshot) if the per-model auto-calc isn't enough.
- WAL benchmark on production-shaped FTIR datasets (the synthetic benchmark used 200 features; real FTIR has ~1000+, where in-memory PLS times scale up and the threshold may shift).
- Heuristic refinement: the 1.0s threshold is an educated guess. After T-41 ships, gather telemetry from real user runs to verify it's well-calibrated.

---

## Effort estimate

~180 LOC + tests. Single-day ticket if no surprises in the migration path. The trickiest piece is verifying that `optuna.copy_study` mid-run doesn't break the TPE sampler's internal state — needs careful testing. If `copy_study` turns out to be incompatible with mid-run migration, fall back to: "decide before trial 0 based on a hardcoded model→speed lookup, never migrate" — degrades the heuristic but keeps the architecture simple.
