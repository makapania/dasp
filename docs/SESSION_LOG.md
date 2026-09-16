# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

Older entries are in [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md) — grep it for historical context; batch 7 on 2026-09-15 moved every entry dated before 2026-09-14 plus the full round-by-round PR #79 crash-resume history (condensed here into the 2026-09-15 "PR #79 crash-resume (merged a5f9a70)" entry); batch 6 moved entries dated before 2026-09-01.

## 2026-09-14 - T-CI-2 RESOLVED: the Linux Xvfb GUI "hang" was never a hang — pytest-timeout budget vs. a legitimately huge test

**Evidence (run 34885012314, job 104113389892, `gui-linux`):** pytest-timeout fired on
`test_xgboost_via_gui` at 180s. The `--timeout-method=thread` stack dump showed the
**MainThread inside XGBoost training, not blocked**:
`harness.run_analysis_direct` (harness.py:542) → `run_search` (search.py:3353,
`pipe.fit`) → `xgboost/sklearn.py:1430 fit` → `training.py:200 bst.update`. The only
other threads were idle joblib/loky waiters. Captured stdout reached **[33/256] grid
configs in the 180s window** (~5.5 s/config → the full sweep needs ~25+ min on the
2-core-class runner). Same test, same run, **windows-latest: 19:30:44 → 20:30:41,
~60 minutes, PASSED** — because the Windows job sets no per-test timeout.

**Conclusion:** the T-CI-2 hypothesis ("a test deadlocks under Xvfb") is false. The
heavy `TestAllModelsViaGUI` model tests (xgboost ~60 min, lightgbm ~7 min on Windows)
simply exceed the informational job's `--timeout=180` on any runner. No OMP/n_jobs
limit fits 256 configs × 5×5 CV of 100-tree boosts into 180s, and a one-test skip
would just move the timeout to LightGBM next run. Per user decision (no Linux GUI
support wanted, fix must be small), the `gui-linux` job was removed from ci.yml and
the T-CI-2 comments rewritten; `--ignore=tests/gui` stays on both Linux jobs and the
Windows leg keeps native GUI coverage. pytest-timeout stays in the lock (harmless).
Killing the job is lossless: with `--timeout-method=thread` pytest-timeout kills the
whole process at first timeout, so the job could never report more than one heavy
test anyway.

---

## 2026-09-14 - REPRODUCED: re-running an 'auto' Bayesian analysis deletes the earlier run's saved study

**Severity: silent data loss, in the default persistence mode.**

**Reproduction:** `tests/test_t41_auto_rerun_preserves_study.py`, which fails on `main`
@ `2860d17`. Run 1 of a slow PLS analysis in 'auto' migrates to SQLite and persists 11
trials. Run 2 of the same configuration:

1. **Starts over in memory.** 'auto' always creates a fresh in-memory study and never
   looks for the one already in SQLite, so the warmup trials are repeated for nothing.
2. **The migration fails.** After warmup, `_migrate_study_to_sqlite` calls
   `optuna.copy_study` into the same configuration-derived `study_name`, which raises
   "Another study with study_name=... already exists".
3. **The cleanup deletes the earlier study.** The except-branch runs
   `optuna.delete_study(study_name, storage_url)`. It was meant to remove a
   half-created copy, but it deletes the earlier run's study. In the test the SQLite
   file ends with **zero studies**.

**Scope, narrower than first assumed.** Data loss needs two 'auto' runs with the same
configuration sharing one storage URL.
- **GUI, normal use:** protected by accident. `run_state.start_run` gives every
  analysis a new `<run_id>.sqlite3`, and GUI crash recovery forces 'always'.
- **Exposed:** callers that keep one storage URL across runs (scripts, custom
  `run_state` use), or the same model and configuration run twice inside one active run.
- **Origin:** it predates T-51. It was surfaced by the T-51 PR A reviews (DeepSeek,
  Codex) and then reproduced.

**Fix** (branch `fix/T41-auto-resume-data-loss`, revised after GLM + DeepSeek review):
- **Resume in 'auto'** only when the SQLite file exists (checked without opening it;
  zero-byte files count as absent), holds this exact `study_name`, **and** its stored
  `data_fingerprint` matches the current (X, y, wavelengths).
  - The study name encodes configuration and environment but **no data identity**.
    Without the data gate, a script reusing one storage for two same-shape datasets
    would resume the wrong study and replay its cached scores (GLM finding).
  - Every study now records `data_fingerprint`, a full SHA-256 of the arrays. The
    existing `run_state.fingerprint_dataset` hashes only three values, too weak for
    this gate.
  - Older studies have no fingerprint, so they are not resumed. They are left intact.
- **Never delete after `DuplicatedStudyError`.** `copy_study` raises it before writing
  anything, so the name belongs to someone else. This closes the check-then-copy race.
- **An unanswerable existence check never authorises deletion** (DeepSeek HIGH: the
  first version treated a listing error as "absent" and would have deleted).
  `_study_exists` returns `None` when unsure.
- **A partial study created by the failing attempt is still deleted.**
  *Correction (PR #78):* no longer true. The failed-migration delete was removed, and a
  failed migration now only warns. See the post-merge Bayesian/search-space fixes
  entry.
- **Review round 2** (DeepSeek ready, GLM merge-with-fixes), all applied:
  - **The fingerprint is written ONLY on a study with no trials.** Writing it onto a
    legacy study resumed via 'always' would claim that study's old trials came from
    the current data, and the 'auto' gate would then wrongly resume it. GLM had called
    that backfill a bonus; DeepSeek was right.
  - **A data mismatch in 'auto' switches that run to 'never'.** The name is taken, so a
    migration could never succeed and would only raise a "migration failed" alarm.
  - **'always' resume on different recorded data now warns** through `progress_callback`
    (flag `data_mismatch_resume`). It still resumes, because crash recovery and legacy
    studies depend on resume by name. The check reuses the single name listing that
    `test_bayesian_study_lookup` pins to one call; a second listing broke that contract.
  - **Fingerprinting is skipped in 'never' mode and when there is no storage.** It hashes
    through a buffer view with no copy, and is documented as SHA-256 truncated to 64 bits.
  - **`_sqlite_file_exists` swallows `OSError`** from a stat race.
- **Six guard mutations, all caught:** no duplicate guard, fail-open check, no data
  gate, no resume, fingerprint written onto existing studies, mismatch staying 'auto'.
- **Accepted residual risk (documented, not fixed):** a multi-process window where our
  check says "absent", our migration fails before writing anything, and another process
  creates the same name before our delete.

## 2026-09-14 - T-51 PR A (extra-axes mechanism): gotchas hit while implementing

**1. `tests/test_bayesian_study_lookup.py` fails when run after `test_t41_*` in one process
(pre-existing on `main`, not fixed).**
- **Cause:** the T-41 `fresh_run_state` fixture pops `spectral_predict.run_state` from
  `sys.modules` and re-imports it. The lookup tests then patch `get_storage_url` on their
  import-time module object. `run_unified_bayesian` does
  `from spectral_predict.run_state import get_storage_url` at call time, which now gets the
  new module, so the patch is never seen and 4 tests fail.
- **Why the full suite hides it:** xdist distributes the two files to different workers.
- **Reproduce:** `pytest -p no:xdist tests/test_t41_bayesian_sqlite_auto_calculator.py tests/test_bayesian_study_lookup.py`.
- **Fix pattern** (used in the new T-51 tests): resolve the module with
  `importlib.import_module("spectral_predict.run_state")` inside the fixture.

**2. Base-sampler parameter names depend on categorical branches.** For example, LightGBM
`num_leaves` is suggested only for some `max_depth` values. The pre-flight collision check
therefore explores every categorical path with a recording stub
(`search_spaces.discover_suggested_names`) instead of running the sampler once.
`search_spaces` receives the sampler as a callable, which avoids an import cycle with
`unified_bayesian`.

**3. `Path.write_text` translates `\n` to CRLF on Windows.** Mutation scripts that read and
write sources in text mode keep CRLF working copies consistent (git autocrlf). Check line
endings before committing anything a script rewrote.

**3a. Optuna 5 does NOT raise on a duplicate `suggest_*` name (Fable probe, corrects the
2026-08-30 entry below).** Calling `suggest_float('a', 0, 2)` after
`suggest_float('a', 0, 1)` in the same trial emits an "Inconsistent parameter values"
warning and returns the first value. An *identical* duplicate returns silently with no
warning. Only a different-kind duplicate (`suggest_int` after `suggest_float`) raises
`ValueError`. Re-probed 2026-09-14 after reviewers disagreed. A resumed study also accepts a different distribution for the same name
across trials. A bundle that collides with a base-suggested name would therefore do
nothing, silently, rather than crash. That is why `resolve_bundles` rejects collisions
before the run, and why `apply_extra_axes` also checks `trial.params` at run time.

**3b. Pre-existing (GLM review, not fixed): 'auto' persistence never resumes a study that
an earlier 'auto' run migrated to SQLite.** In 'auto' mode `run_unified_bayesian` always
creates a fresh **in-memory** study (`create_study(..., sampler=sampler)`); only 'always'
touches storage before warmup. A second 'auto' run therefore starts from zero under the
same name, and when it migrates again it copies into the existing SQLite study. If that
re-migration fails, the cleanup `optuna.delete_study(study_name, storage_url)` could
remove the **earlier run's** trials (DeepSeek round-2 note). This is unchanged from
`main`; it needs a T-41 follow-up ticket.

**4. Default-path baseline captured on `main` @ `2860d17`** (post-#67). Two independent
captures were byte-identical, so the 30-trial PLS TPE trace is deterministic under
`enable_sqlite_persistence='never'`. Fixture: `tests/fixtures/t51_default_path_baseline.json`.

---

### 2026-09-14 — CI red-cause investigation (GitHub failure emails)

- **Not timeouts.** Every CI test job ran to completion (~2h is the serial suite:
  ~3,000 tests, no xdist). CI red = five deterministic failures, all test drift, none
  product bugs:
  - `test_cv_strategy::test_classification_metrics_template_has_no_nameerror` execs
    `get_cross_validation_template()` standalone; since #52 the template calls
    `_fit_fold` / `EARLY_STOPPING_ROUNDS`, which CodeGenerator always emits first.
    The template is only used via CodeGenerator.
  - `test_t19` ×2: one pinned the pre-`_fit_fold` fold-fit string; the other asserted
    bare substring `"fit_kwargs" not in script`, tripped by the helper's `**_fit_kwargs`
    and by an unconditional XGBoost comment in the CV block (comment now emitted only
    with the XGBoost sample-weight block).
  - `tests/gui/test_multiclass_gui::test_tab9_rejects_multiclass_primary` patched
    `src.spectral_predict.model_io.load_model` — a *different module object* from the
    `spectral_predict.model_io` the GUI imports, so the patch never applied. The
    auxiliary twin passed by accident (a FileNotFoundError also calls showerror).
  - `tests/gui/test_comprehensive::test_catboost_via_gui`: harness hard-coded CatBoost
    as `comprehensive`; `model_config.MODEL_TIERS` has it only in `experimental`.
    Harness now derives the tier from `get_tier_models`.
- **Local-only** `test_export_code` ×2 failures: tests ran exported scripts with bare
  `python` (PATH = system Python without sklearn). Use `sys.executable`.
- **`[skip ci]` is honoured**; the extra runs were PR-branch commits and merge commits.
- black/flake8 CI steps were never reached while tests failed; tree needs ~212 files
  reformatted and has ~1.8k flake8 issues.
- **Gotcha - check a redundancy claim at BOTH ends of the range.** Plan section 5 lists
  `max_samples` cat[auto, 0.5, 0.8, 1.0] for IsolationForest. `'auto'` is
  `min(256, n_samples)` and a fraction is `int(fraction * n)`, so under 257 inliers 1.0
  duplicates 'auto' (identical fits, distinct fit fingerprints, a quarter of the TPE mass
  on a duplicate). GLM 5.3 caught that and I dropped 1.0 - wrongly. Codex then showed the
  justification was false: at n=300 'auto' is 256 but 0.8 is only 240, so 1.0 is the ONLY
  full-sample choice above 256 and dropping it removed real capability. 1.0 is restored
  with the coincidence documented. Unlike `minkowski(p=2)`, which always aliases
  euclidean, this redundancy is data-dependent - probe the whole range before calling a
  choice degenerate, and distrust a tidy-sounding justification (including your own).
- **Fractional `max_samples` needs >= 2 rows in the smallest training fold.**
  `int(0.5 * 1)` is 0 and sklearn raises "Sample weights must contain at least one
  non-zero number", so with 2-3 inliers under 2-fold CV every trial failed into a 1e10
  penalty while 'auto' completed - it reads as "this model is bad" rather than "this
  bundle cannot run on this data". New `BundleSpec.min_train_fold_rows` (validated, part
  of bundle equality, NOT part of the space identity) plus a check in
  `run_unified_bayesian` that raises `ExtraAxesConfigError` before the study is created.
- **Two traps when adding a field to `BundleSpec`:** `test_curated_bundles_are_hashable`
  rebuilds a bundle field by field via `_rebuilt_with_fresh_dicts`, so a new field must be
  added there or the hash comparison fails; and
  `test_base_sampler_bodies_unchanged_from_main` pins sampler source by slicing between
  two `def`s, so a new module-level helper placed between `suggest_model_params` and
  `suggest_one_class_params` breaks the pin even though neither sampler changed.
- **Adding to `BUNDLES` breaks registry-wide assertions in the PR B tests.**
  `tests/test_t51_supervised_bundles.py` iterated all of `BUNDLES` (fitting supervised
  models) and carried an explicit `test_no_one_class_bundles_in_pr_b` guard. Scoped those to
  a `SUPERVISED_BUNDLES` subset and turned the guard into a supervised/one-class
  disjointness check. PR D/E/F will hit the same tests.
- `optuna.trial.FixedTrial.params` only contains names suggested so far, so seeding an axis
  name in the fixed dict does not defeat `apply_extra_axes`'s clash guard — the tests
  exercise the real path.
- Mixed-type categorical choices (`"auto"` with floats) are fine for Optuna and for the
  identity hash; `_validate_axis` rejects choices that compare equal, not ones of mixed type.
