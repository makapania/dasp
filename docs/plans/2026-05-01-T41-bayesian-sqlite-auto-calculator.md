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
| Auto-calculator with **first-10-trial timing window** using **median** fit time | Real fit times depend on dataset size (n_samples, n_features) and CV folds, not just model name. 10 trials gives a stable enough estimate; median is robust to TPE's startup-phase outliers (e.g., random `n_components=1` PLS draws skewing the mean low). [Revised per DeepSeek review #3.] |
| In-memory for first 10 trials regardless of decision | (a) too short to bother setting up SQLite, (b) we don't know the decision yet. Crash window = ~10 trials worth of work, acceptable. **Fallback rule:** if completed trials < 3 (e.g., one-class with most folds skipping), default SQLite ON conservatively — don't trust an empty-distribution decision. |
| `optuna.copy_study(in_mem → sqlite)` for migration; **then** `optuna.load_study(storage=..., sampler=TPESampler(seed=..., n_startup_trials=20, n_ei_candidates=32, multivariate=True, consider_endpoints=True))` to attach a fresh sampler that rebuilds TPE state from the copied trials | DeepSeek empirically verified (Optuna 4.8) that this two-step pattern works: `copy_study` is the data move; `load_study(sampler=...)` is the only way to attach a sampler to an existing study. **`create_study(load_if_exists=True, sampler=...)` SILENTLY IGNORES the sampler kwarg on existing studies — do NOT use that.** TPE's `n_startup_trials` counter is preserved across migration, and trials past startup show clear TPE-guided exploitation post-migration (5+5 test sequence: best=3.93 random startup → best=0.043 trial 9 post-migration). |
| WAL mode applied **at migration time** (not in `start_run`) | Plan originally proposed WAL in `start_run`, but `start_run` doesn't actually create the SQLite file (Optuna does so lazily on first access). If all models stay in-memory, the file never exists and the PRAGMA call would target a non-existent file. Instead: when migrating, `copy_study` creates the SQLite file, then immediately apply `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL` before the migration's first commit completes. [Revised per DeepSeek review #5.] |
| WAL durability tradeoff (honest version) | WAL+sync=NORMAL prevents database corruption on crash but **may lose uncheckpointed trials**. Default WAL auto-checkpoint is 1000 pages (~4MB, ~800 trials at ~5KB/trial). On crash, last few trials past the most recent checkpoint are lost. Acceptable for our use case. **Set `PRAGMA wal_autocheckpoint = 50`** to bound loss to ~50 trials per crash event. [Revised per DeepSeek review #4.] |
| 3-way GUI override `Auto / Always on / Always off` | Most users want auto. Power users with long heavy-model runs may want always-on for safety. Speed-sensitive users may want always-off. Three radio buttons, one tooltip. |
| Threshold = median fit time > 1000ms → SQLite ON | At 1000ms fit time, ratio is `1 + 200/1000 = 1.2×`. User considers <2× acceptable; this is well within. |
| Decision logged to progress callback | Surface the auto-calc decision so users see "Auto-enabled SQLite (median trial = 4.2s)" or "Auto-disabled SQLite (median trial = 0.04s, would slow run by 6×)". |
| `run_state` machinery becomes the *default* for the session | The auto-calculator can override per-call by simply NOT passing `storage` to `optuna.create_study` (the `_active_storage_url` global stays set for subsequent models — no "release" needed; the file is created lazily on first SQLite access). [Clarified per DeepSeek review #2.] |

---

## User-facing behavior (concrete examples)

| Scenario | Behavior |
|---|---|
| User runs PLS Bayesian, 100 trials, default `Auto` | First 10 trials in-memory (~300ms). Decision: in-memory (median fit ~30ms < 1000ms). Remaining 90 trials in-memory. Total ~3s. **No resume capability** but matches pre-T-11 speed. |
| User runs XGBoost Bayesian, 100 trials, default `Auto` | First 10 trials in-memory (~6s). Decision: SQLite ON (median fit ~600ms; ratio with overhead = 1.33×). Migrate study to SQLite + WAL at trial 11. Remaining 90 trials with persistence. Total ~75s vs ~55s in-mem-only. **Resume works.** |
| User selects PLS + XGBoost, default `Auto` | PLS run: in-memory throughout (~3s). XGBoost run: SQLite ON after auto-decision. Each model's persistence decision is independent. |
| User explicitly picks `Always on` | All trials in SQLite + WAL from trial 1, even fast models. PLS will be 8× slower, user accepts. |
| User explicitly picks `Always off` | All trials in-memory always. No resume even for heavy models. |
| Crash mid-run, `Auto` mode picked SQLite | Resume works as today — Optuna re-opens the SQLite study, picks up where left off. |
| Crash mid-run, `Auto` mode stayed in-memory | No resume. User loses the run. (Acceptable because in-mem is reserved for fast models where re-run is cheap.) |
| Crash during first 10 trials (before decision) | No resume regardless of mode. (Acceptable; few seconds of work lost.) |

---

## Tasks

### Task 1: `unified_bayesian.run_unified_bayesian` — accept new parameter

- [ ] Add `enable_sqlite_persistence: str = 'auto'` to signature (after `enable_autoscale`).
  - Values: `'auto'`, `'always'`, `'never'`.
- [ ] Plumb to `create_unified_objective` (no behavior change there yet — just parameter pass-through).
- [ ] Update docstring.

### Task 2: `unified_bayesian` — auto-decision logic

- [ ] After 10 completed trials in `study.optimize` callback, compute `median_trial_time = median(t.duration.total_seconds() for t in completed trials)`. (Revised from 5/mean per DeepSeek review #3 — TPE's startup-phase random draws can skew mean low.)
- [ ] **Fallback for low completion count:** if completed trials < 3, default to SQLite ON (conservative — don't trust an empty-distribution decision). Common in one-class with many fold-skip failures.
- [ ] If `enable_sqlite_persistence == 'auto'`:
  - If median > 1.0s AND `run_state.get_storage_url()` is None: skip migration (user disabled persistence at session level — respect that).
  - If median > 1.0s AND storage URL is set: migrate per Task 4 (copy_study + load_study + WAL).
  - If median ≤ 1.0s: stay in-memory for the rest of the run; **do not pass `storage` kwarg to `optuna.create_study` for this model**. The `_active_storage_url` global stays set for the next model in the loop. The SQLite file is created lazily by Optuna on first access — if no model in this session ever migrates, the file is never created (no cleanup needed). [Clarified per DeepSeek review #2.]
- [ ] If `enable_sqlite_persistence == 'always'`: skip the timing window; create the SQLite study from trial 0 with `optuna.create_study(storage=run_state.get_storage_url(), sampler=TPESampler(...), load_if_exists=True)`.
- [ ] If `enable_sqlite_persistence == 'never'`: stay in-memory; ignore `run_state.get_storage_url()`.
- [ ] Log the decision to `progress_callback` so users see e.g. "Auto-enabled SQLite (median trial = 4.2s, ratio 1.05×)" or "Auto-disabled SQLite (median trial = 0.04s, would slow run by 6×)".

### Task 3: WAL mode at migration time (NOT in `start_run`)

[Revised per DeepSeek review #5: `start_run` doesn't actually create the SQLite file — Optuna does so lazily when first accessed. If all models stay in-memory, the file never exists. So WAL setup belongs in the migration helper, not `start_run`.]

- [ ] In the migration helper (Task 4), after `copy_study` creates the target SQLite file but before further Optuna writes:
  ```python
  import sqlite3
  # copy_study has now created the file; apply WAL pragmas before we make more writes
  db_path = sqlite_url.replace("sqlite:///", "")
  conn = sqlite3.connect(db_path)
  conn.execute("PRAGMA journal_mode = WAL")
  conn.execute("PRAGMA synchronous = NORMAL")
  conn.execute("PRAGMA wal_autocheckpoint = 50")  # bound loss to ~50 trials on crash
  conn.close()
  ```
- [ ] For the `Always on` path (Task 2 — SQLite from trial 0): the pragmas need to be applied immediately after `optuna.create_study(storage=url)` first creates the file. Wrap that in a helper.
- [ ] WAL setting persists with the file — no need to repeat on resume.
- [ ] Resume of a pre-T-41 file (created without WAL) continues with the original journal mode. Don't fight existing files.
- [ ] If the user picks `Always off`, **`start_run` should not generate a SQLite URL at all** — saves I/O and avoids the stale-sidecar problem (DeepSeek review #5). Document this branch.

### Task 4: Migration helper

[Revised per DeepSeek review #1: explicitly prescribe `load_study(sampler=...)` because `create_study(load_if_exists=True, sampler=...)` SILENTLY IGNORES the sampler kwarg on existing studies — easy footgun.]

- [ ] Helper `_migrate_study_to_sqlite(study, sqlite_url, study_name, random_state) -> optuna.Study`:
  ```python
  import optuna
  from optuna.samplers import TPESampler

  # Step 1: copy all trials, params, user_attrs, directions to SQLite
  optuna.copy_study(
      from_study_name=study.study_name,
      to_study_name=study_name,
      from_storage=study._storage,
      to_storage=sqlite_url,
  )

  # Step 2: apply WAL pragmas to the newly-created SQLite file (Task 3)
  _apply_wal_pragmas(sqlite_url)

  # Step 3: load the study with a fresh TPE sampler. The sampler rebuilds its
  # internal state from the copied trials on the next study.optimize() call.
  # CRITICAL: load_study(sampler=...) is the ONLY way to attach a sampler to
  # an existing study. create_study(load_if_exists=True, sampler=...) silently
  # ignores the sampler kwarg on existing studies — DO NOT use that pattern.
  return optuna.load_study(
      study_name=study_name,
      storage=sqlite_url,
      sampler=TPESampler(
          seed=random_state,
          n_startup_trials=20,
          n_ei_candidates=32,
          multivariate=True,
          consider_endpoints=True,
      ),
  )
  ```
- [ ] After migration, the **returned** study object MUST replace the in-memory study at `unified_bayesian.py:~2089` so `convert_study_to_dataframe` reads from SQLite, not the disconnected in-memory study (DeepSeek architectural note).
- [ ] Test that the migrated study continues optimizing with TPE (not random) — assert trial post-migration uses TPE-suggested params, not uniform-random.
- [ ] Test that `n_startup_trials` counter is preserved: 5 pre-migration + 15 post-migration random + remaining TPE-guided.

### Task 5: GUI — 3-way radio button

- [ ] In Bayesian config section of `_create_tab4a_basic_settings` (or wherever the Bayesian-specific options live — verify location), add:
  ```python
  self.bayesian_persistence_mode = tk.StringVar(value='auto')
  ttk.Label(parent, text="Crash-resume persistence:", style='Subheading.TLabel').grid(...)
  for value, text in [('auto', 'Auto (recommended)'), ('always', 'Always on'), ('never', 'Always off')]:
      ttk.Radiobutton(parent, text=text, variable=self.bayesian_persistence_mode, value=value).grid(...)
  ```
- [ ] Tooltip explains the auto-calculator math (~200ms/trial overhead, threshold at 1s median fit time, mention that fast models like PLS run 8× slower with persistence so 'always on' is rarely the right choice).
- [ ] Both `run_unified_bayesian` call sites in the GUI (one-class @ ~27003 and regression/cls @ ~27345) get the new kwarg: `enable_sqlite_persistence=self.bayesian_persistence_mode.get()`.

### Task 6: Persist the user's choice in `run_state` start_run + handle stale sidecars

- [ ] When the user picks `Always off`, `start_run` should NOT generate a SQLite URL at all (saves I/O and avoids the stale-sidecar problem).
- [ ] Add `bayesian_persistence_mode: str` to `start_run` metadata so it's logged for audit.
- [ ] No need to add to saved-model metadata (T-36 lesson — this is a per-run setting, not a per-model property).
- [ ] **Stale-sidecar cleanup (per DeepSeek review #5):** in `mark_complete()`, after deleting the `active_run.json` sidecar, also check whether the SQLite file exists / has any trials. If the file doesn't exist (every model stayed in-memory) OR is empty (Optuna created it but no trials wrote), delete it too. Prevents the next session's `find_incomplete_run()` from offering a phantom "Resume?" for a session that has nothing to resume.
- [ ] Add the same cleanup to `clear_resume_state()` for the user-rejects-resume path.

### Task 7: Tests

- [ ] `test_t41_auto_calculator_picks_in_memory_for_pls`:
  - Run a tiny PLS Bayesian study with `enable_sqlite_persistence='auto'`.
  - Assert that the study's storage is `InMemoryStorage` after completion.
  - Assert that no `.db` file was created in the run dir.
- [ ] `test_t41_auto_calculator_picks_sqlite_for_slow_model`:
  - Run a tiny RandomForest Bayesian (or mock the trial duration to >1s).
  - Assert that the study migrates to SQLite by trial 11 (10 trials warmup + decision).
  - Assert that the .db file exists and contains all completed trials.
- [ ] `test_t41_always_on_creates_sqlite_from_trial_zero`:
  - Run a tiny PLS Bayesian with `'always'`.
  - Assert SQLite is used from the start (verify via `study._storage` type).
- [ ] `test_t41_always_off_never_creates_sqlite`:
  - Run a tiny RandomForest Bayesian with `'never'`.
  - Assert no `.db` file is created.
- [ ] `test_t41_wal_mode_set_on_migrated_file` (revised — was "on new file"):
  - Force a migration; query `PRAGMA journal_mode` on the resulting file — assert returns `'wal'`.
  - Also verify `PRAGMA wal_autocheckpoint` returns 50.
- [ ] `test_t41_migration_preserves_trial_count`:
  - Manually migrate a 10-trial in-memory study to SQLite.
  - Reload from SQLite, assert all 10 trials present with same params.
- [ ] `test_t41_resumable_after_migration`:
  - Migrate mid-run; close the session; reopen via `load_if_exists`; verify trial count and best value.
- [ ] **`test_t41_tpe_continues_learning_across_migration`** (NEW — DeepSeek review #8.1):
  - Run a 20-trial study where median fit is forced > 1.0s (mock the model).
  - Assert that the BEST value strictly improves between pre-migration trial 10 and post-migration trial 20.
  - Critical: catches the silently-ignored-sampler bug if anyone changes the migration helper to use `create_study(load_if_exists=True, sampler=...)`.
- [ ] **`test_t41_mixed_model_independent_decisions`** (NEW — DeepSeek review #8.3):
  - Simulate a GUI loop: call `run_unified_bayesian` with `model_name='PLS'` then with `model_name='XGBoost'` in sequence.
  - Assert the PLS study uses InMemoryStorage; the XGBoost study migrates to SQLite.
- [ ] **`test_t41_auto_decision_surfaced_in_progress_callback`** (NEW — DeepSeek review #8.4):
  - Intercept `progress_callback` calls; assert one of them carries an "Auto-enabled SQLite" or "Auto-disabled SQLite" message with the median fit time.
- [ ] **`test_t41_one_class_uses_same_path`** (NEW — DeepSeek review #7):
  - Verify the one-class Bayesian path (`run_unified_bayesian(task_type='one_class')`) honors `enable_sqlite_persistence` the same way the regression/classification path does.
- [ ] **`test_t41_low_completion_count_defaults_sqlite_on`** (NEW — fallback rule):
  - Mock first 10 trials so most fail (`value=inf`, completed < 3).
  - Assert the auto-calc defaults to SQLite ON (conservative path).
- [ ] **`test_t41_stale_sidecar_cleanup`** (NEW — DeepSeek review #5):
  - Run an all-in-memory session (PLS only, `'auto'` mode → no migration).
  - Call `mark_complete()`.
  - Assert no leftover `.sqlite3` file in the run dir AND no `active_run.json` sidecar.
- [ ] Benchmark regression: run `tests/_bench_bayesian_per_model.py` with the auto-calculator enabled; assert PLS ratio < 1.2× and XGBoost ratio < 2.0× (loose thresholds for CI noise tolerance).

### Task 8: Validation note

- [ ] `docs/bugfix_validation/T41_bayesian_sqlite_auto_calculator.md`:
  - Re-run the benchmark with the auto-calculator and document the new ratios.
  - Document the auto-decision threshold choice (1.0s median fit time) and the math justifying it.
  - Note that JournalStorage was tested and rejected (16× ratio on Windows due to NTFS file locking).

### Task 9: PR

- [ ] Branch `fix/T41-bayesian-sqlite-auto-calculator`, branched from main (after T-36 lands) — NOT from `feature/T36-autoscale-toggle`.
- [ ] Targeted test sweep: `pytest tests/test_unified_bayesian*.py tests/test_t41_*.py tests/test_cv_pls_clamp.py -v`.
- [ ] Multi-pass review: DeepSeek per-phase + Codex final, mirror T-36 review pattern.
- [ ] Open PR with summary referencing this plan + benchmark numbers.

---

## Verification checklist

- [ ] PLS Bayesian on `Auto` runs at full in-memory speed (≤ 1.2× of pre-T-11).
- [ ] XGBoost Bayesian on `Auto` migrates to SQLite + WAL after first 10 completed trials (i.e., at trial 11) and is resumable on crash.
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

~180 LOC + tests. Single-day ticket. DeepSeek's pre-implementation review (2026-05-01, see "Review trail" below) empirically verified the trickiest piece — `optuna.copy_study` + `optuna.load_study(sampler=...)` mid-run migration preserves TPE state correctly on Optuna 4.8 — so the fallback architecture (hardcoded model→speed lookup, never migrate) is unlikely to be needed.

## Review trail

| Pass | Reviewer | Verdict | Findings |
|---|---|---|---|
| Pre-implementation | DeepSeek V4 Pro Max thinking (direct API, 2026-05-01) | READY_WITH_PLAN_REVISIONS | 2 HIGH (Task 4 sampler-attach mechanism, Task 2 "release handle" ambiguity), 2 MEDIUM (warmup window too short, WAL durability claim overstated), 4 LOW (stale sidecar, threshold edge cases, one-class deferral, missing tests). All revisions applied to this plan; ready to implement. Empirical Optuna 4.8 tests confirmed `copy_study` + `load_study(sampler=TPESampler)` works for mid-run migration. |
| Implementation | _pending_ | _pending_ | _pending_ |
| Codex final | _pending_ | _pending_ | _pending_ |
