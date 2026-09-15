# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

Older entries are in [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md); batch 5 on 2026-09-12 moved entries before 2026-07-12, following the two-month retention rule.

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

## 2026-09-13 - Lockfile changes never reached existing venvs (why jcamp stayed 1.2.2)

Root cause, confirmed by GLM 5.3 Flash and DeepSeek 4.1 Flash reviews plus git
history: `requirements-lock.txt` was created with `jcamp==1.2.2` (`0cad9e5`,
09-10) and moved to `==1.3.2` in `be7963c` (09-12), the same commit that switched
`io.py` to `jcamp.readfile`. Venvs built from the older lock kept 1.2.2. Nothing
re-applied the lock: `RUN_SPECTRAL_PREDICT.bat` only reinstalled when three
modules were *missing*, not when versions were *wrong*; the migration steps read
as create-time only; `pip check` compares against pyproject floors, not exact pins.
The manual step 3b (`pip install jcamp==1.3.2`) fixed one package and would miss
the next bump the same way.

Fix: `scripts/check_env_lock.py` (stdlib only, ~120 ms) compares installed
distributions to every `==` pin. Launchers run it and reinstall from the lock on
drift; a failed install still stops the launch, as
`tests/test_build_and_launcher_safety.py` requires (an early warn-and-launch
version broke those tests, caught in GLM's PR 66 review). Exit 2 (unreadable
lock) warns and launches without reinstalling, and the parser skips lines it
cannot judge (options, URLs, markers) so a lock change cannot cause a
reinstall on every launch. The installer build
refuses a drifted build venv, since the bundle would ship the stale package
(`DASP_ALLOW_LOCK_DRIFT=1` for 3.12 rollback builds). The check runs before the
previous `dist/` bundle is deleted, so a refused build keeps the old one. `read_jcamp_file` raises an
actionable `ImportError` instead of `AttributeError`; the frozen self-test checks
`jcamp.readfile`. Gotcha: a lockfile regenerated with PowerShell `>` gets a BOM
(UTF-8 or UTF-16), so the script decodes by BOM. Verified end to end: downgraded
`.venv314` to jcamp 1.2.2 → check exit 1 → lock reinstall → exit 0, jcamp 1.3.2.

---

## 2026-09-13 - FIXED: Model Development CV row raised TclError "isn't packed" on default kfold

`_on_refine_cv_strategy_changed` packed widgets `before=` siblings that the
previous state had hidden with `pack_forget()`. The tab starts on kfold, which
hides the Repeats label; any later `refine_cv_strategy.set('kfold')`
(`_load_default_parameters`, loading a saved model's training config) then packed
the Folds spinbox before that hidden label and Tk raised. Also broken:
loo -> repeated_kfold. No user action needed. Tkinter reports callback
exceptions to stderr and keeps running, so it showed only in the console log
during a first 3.14 GUI launch. Fix: hide all four widgets, then re-pack the
visible ones before the always-packed hint label. Tests:
`tests/gui/test_refine_cv_strategy_widgets.py` (8, fail before / pass after).
`tests/gui` in `.venv314`: 132 passed, 2 failed (both baseline), no teardown crash
this time.

---

## 2026-09-13 - Local .venv314 full suite: baseline-clean, but Python crashes at session teardown

Primary Windows machine, `.venv314` (Python 3.14.7), `main` at `8231feb`,
`pytest -q -p no:randomly --tb=no -rf`: all 3,042 collected tests ran, and the
only failures were the five known baseline node IDs listed in
`reviews/2026-09-12-pr65-installation-validation.md`. Zero new failures.

**But the process died with `Windows fatal exception: access violation` after the
last test**, in the `session_app` fixture teardown (`tests/gui/conftest.py:86`,
`root.destroy()` in tkinter). Exit code 5 (from 0xC0000005), and pytest's final
summary, `-rf` list and cache were never written. It looks like a failed run
but is not. Seen once; not yet known whether it reproduces or happens on CI.

**Recovering the failure list without the summary:** with `-p no:randomly` the
order is fixed, so each `F` in the progress output maps to the test at that
position in `pytest --collect-only -q -p no:randomly`. Check that the progress
character count equals the collected count first. Or run with `-v` so each
result is written as it happens.

---

## 2026-09-13 - jcamp: .venv314 must be upgraded to 1.3.2 (JCAMP import is broken on 1.2.2)

**Install jcamp 1.3.2 in `.venv314` on every machine.** Since `be7963c`, `io.py`
calls `jcamp.readfile(...)`, which exists only in jcamp >=1.3.0, so JCAMP-DX
import fails on 1.2.2. As of 2026-09-13, JCAMP did not work in `.venv314` on
other machines.

```bash
.venv314\Scripts\python -m pip install "jcamp==1.3.2"
.venv314\Scripts\python -c "import importlib.metadata as m; print(m.version('jcamp'))"   # expect 1.3.2
```

Confirm with `pip show jcamp` or `importlib.metadata`, not `jcamp.__version__`:
upstream never updated that string, so 1.3.2 still prints `1.2.2`.

Verified on the primary machine (`.venv314`, Python 3.14.7) with 1.3.2: a
JCAMP-DX write/read round trip through `write_jcamp_file` / `read_jcamp_file`
recovered all 2151 wavelengths. `.venv312` stays on 1.2.2 as the rollback venv.

---

## 2026-09-12 - Python 3.14 migration: what the analysis got wrong, and what only building could tell us

**An analysis document that was never executed had nine errors in it.**
`docs/PYTHON_UPGRADE_DECISION.md` was careful, cited file:line throughout, and
was still wrong in ways that would each have cost a turn. Its own Appendix B
admits nothing was installed, built or run. Recorded because the failure mode
generalizes: *verified by inspection* and *verified* are different claims.

- It reported the installed jcamp as 1.2.1 drifting from a 1.2.2 lockfile pin.
  It had read `jcamp.__version__`, which the package ships stale. pip metadata
  said 1.2.2 and matched. **Reading a `__version__` attribute is not a version
  check** - use `importlib.metadata`.
- It said the build path hardcodes 3.12 "in four places". It was ~25 across
  three files, several user-visible.
- It called source-only `jcamp==1.2.2` the one blocker for 3.14. It builds from
  source on 3.14 without complaint.
- It carried the Python 3.12 float `sum()` change in as a migration risk. That
  landed in 3.11->3.12; the project already started at 3.12.
- Both build files still deferred to a "production 3.11 spec" and a
  `build_installer.py` that no longer exist.

**The pandas TOC collision fires on essentially every build.** The post-COLLECT
repair in `build_installer_py312.py` triggered on all three 3.14 builds. It is
load-bearing, not a historical workaround awaiting cleanup.

**PyInstaller's manual DLL globs never matched OpenBLAS.** The spec collects
`*/lib/*.dll`, `*/*.libs/*.dll`, `*/.libs/*.dll`, `*/libs/*.dll` - every pattern
requires a parent directory. But `numpy.libs/`, `scipy.libs/`, `pandas.libs/`
and `llvmlite.libs/` sit at the TOP level of site-packages. The bundle works
only because PyInstaller's hooks collect them. **That manual list is a false
safety net**; if hook behavior changes, it will not catch the fall.

**A build step that could not fail was failing.** Every path in
`run_inno_setup()` returned True and `main()` discarded the result, so a build
producing no installer printed "Build Complete" and exited 0. Reproduced
directly. Compounding it, `find_inno_setup()` never checked
`%LOCALAPPDATA%\Programs`, which is where winget installs Inno Setup - so a
machine with it correctly installed reported "not found". For a project whose
only distribution channel is that installer, this was the most expensive
possible thing to fail quietly.

**Aggregate metrics hide per-sample changes.** The first baseline compared only
ranked tables. Adding per-sample out-of-fold predictions changed what the
comparison could see: after numpy 2.4.4->2.5.3, per-sample predictions were
byte-identical while every aggregate metric column moved by 1e-13 to 1e-15.
Comparing only one of the two would have given a misleading answer either way.

**Ranking ties are decided by floating-point noise.** That same numpy bump
swapped ranks 225/226 between two PLS models whose `CompositeScore` agreed to 15
significant figures. Harmless here, but near-tied rows can reorder for reasons
unrelated to model quality.

**Upgrade ordering can be forced by a cap you cannot see.** `numba 0.66`
required `numpy<2.5`, so numpy could not advance until numba did - the resolver
does not explain this, it just refuses. Conversely `alive-progress 3.3.0` pins
`about-time` and `graphemeu` to exact versions, so pip installs the newer one
and *then* reports the conflict, leaving the environment broken.
`scripts/upgrade_check.py` now detects both classes before you try.

**Do not modify a venv while its test suite is running.** Upgrading a package
mid-run invalidated a 1-hour suite and it had to be redone.

---

## 2026-09-10 - Repo hygiene: a .gitignore rule that never matched, and AGENTS.md drift

**1. `git status` output is not valid `.gitignore` syntax.** The rule added to
suppress the mangled Windows tempfile names was written as
`C\357\200\272Users*` - copied straight out of `git status`, which C-quotes
non-ASCII bytes in paths. Git does **not** decode those octal escapes in a
`.gitignore` pattern, so the rule matched nothing and all five files kept showing up
as untracked for months. Replaced with `C*Users*AppData*Temp*`, verified with
`git check-ignore -v`. Lesson: always confirm a new ignore rule with
`git check-ignore -v <path>` rather than assuming a pasted path works as a pattern.

**2. `AGENTS.md` was an untracked, stale copy of `CLAUDE.md`.** It was 17 lines
behind - missing the "there is no CLI" section and the mandate to read
`docs/AGENT_COMPOSITION.md` - so Codex was reading a guide that still implied a CLI
existed, while Claude read the current one. Because it was untracked it also existed
on only one machine. Replaced with a short **pointer** to `CLAUDE.md` and committed.
Do not re-copy the contents: a copy is what drifted. One guide, one file.

**3. ~40 untracked scratch files were masking the real answer to "is everything
committed?"** `.pytest-tmp*/` trees, `tools/_*` A/B JSONs and repro scripts,
`*_fails.txt`, `merge_gate_diff.json`, `live_gui_*`, timestamped `example/colab_*.ipynb`.
All now ignored, with `!tools/_autoscale_bayes_compare_full.json` negated because it
is tracked on purpose. A noisy `git status` hid one genuinely unpushed branch
(`feat/T16-phase2-permutation`, local-only, now pushed).

---

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

## 2026-09-13 - T-51 step 1 (SVM scaler) and step 0 (caller sweep): what the ticket got wrong

**1. The SVM scaler bug reached further than the ticket listed, and exported code was
already correct.** `'SVM'` (the registered classification family) was missing from
`search.py` `SCALE_SENSITIVE_MODELS` (grid pipeline and `_rebuild_model_from_row`), from the
local set in `unified_bayesian.py`, and from **five** GUI tuples (Tab 7 param stripping,
three refit pipelines, and the full-spectrum save path). `code_generator._needs_standard_scaler`
already listed `'SVM'`, so exported scripts scaled the model while the app did not. The fix
also restores app/export parity. On the pinned fixture in `tests/test_svm_scaler_family.py`,
accuracy is 1.0 scaled vs 0.883 unscaled.

**2. NSGA-II was never affected.** It encodes classification SVM as `model_type='SVR'`
(`nsga2_search._build_model` returns `SVC` for classification), and `'SVR'` was already
in its set. The ticket's "omits both" was wrong.

**3. Found, NOT fixed: the Bayesian importance proxy is unscaled for every scale-sensitive
family.** `unified_bayesian.compute_importances` (`method='importance'`) calls
`build_model(...)` then `model.fit(X, y)` on a bare estimator. SVR, Ridge, Lasso,
ElasticNet, MLP and (now) SVM all compute variable-selection importances on unscaled
spectra, even though their CV fits are scaled. It is independent of the `'SVC'` string
bug, and fixing it changes importance rankings for every scale-sensitive model, so it
needs its own ticket and approval. The new spy test excludes that caller explicitly.

**3b. Also found, NOT fixed (DeepSeek review):**
- **NSGA-II display metrics are unscaled for every scale-sensitive family.** The helpers
  `_compute_solution_r2`, `_compute_display_rmse`, `_compute_nir_metrics`,
  `_compute_classification_cv_metrics` and `_compute_calibration_metrics` call
  `_build_model` bare, and only wrap it when an imbalance step exists. Fitness itself is
  scaled (`nsga2_search.py:~1465`). The NSGA-II leaderboard's F1/AUC/R2 columns therefore
  describe a different model than the one ranked. So the earlier "NSGA-II was never
  affected" holds only for the fitness path.
- **NSGA-II cannot evaluate classification SVM chosen as 'SVM' (GLM review).** The GUI
  appends `'SVM'` to `selected_models` (`gui:~24084` → `models=` `~:29153`), but
  `nsga2_search._build_model` has no `'SVM'` branch (it returns `None` at `~:1062`). Every
  such chromosome scores the 1e10 penalty, and no SVM row can come out. Only the `'SVR'`
  encoding builds SVC.
- **`model_registry.MODELS_WITH_FEATURE_IMPORTANCE` lacks `'SVM'` (GLM review).** Grid
  classification SVM rows show `top_vars = "N/A"`, even though `get_feature_importances`
  handles `'SVM'`. It is not a one-word fix: `MODELS_WITH_SUBSET_SUPPORT` is the same list,
  so adding `'SVM'` also switches on subset search for SVM (more configs, longer runs).
  That needs its own decision.
- **GUI refit double-scales under autoscale.** The preprocessing pipeline appends an
  `autoscale` StandardScaler, and the scale-sensitive branches (`~:39628/39756/39811`)
  append another. This affects SVR, Ridge, MLP and others, not just SVM. It is
  near-identity, but it diverges from T-36 backend behaviour.

**4. There are three TPE sampler paths, not two.** `_make_tpe_sampler(random_state)`
hardcodes `n_startup_trials=20`. It is used by `_migrate_study_to_sqlite` (the T-41 'auto'
in-memory→SQLite migration) and the 'always' reattach, in addition to the inline sampler.
Threading a new `n_startup_trials` only to the two sites named in the ticket would
silently reset it after auto-migration. Recorded in
`docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md`.

**5. Step-0 caller sweep:** 55 real calls to `run_unified_bayesian` /
`create_unified_objective`, none via `**kwargs`. `tools/bench_baseline_compare.py`
runs a `WORKER_SCRIPT` string against an *old checkout*, so it must never gain new kwargs.

**6. The version bump touches four files.** `test_t14b_pyinstaller_and_gui_version_drift.py`
pins `pyproject.toml`, `installer/spectral_predict_py312.iss` and `version_info.txt`
(strings and the `filevers` tuple) to `__version__`.

**Tooling:** dispatching opencode (GLM) in write mode with `--dangerously-skip-permissions`
is blocked by Claude Code's auto-mode permission check ("Create Unsafe Agents"). Read-only
opencode reviews are unaffected.

---

## 2026-08-30 - T-51 design: two non-obvious constraints on widening the Bayesian search space

**Context**: a downstream contamination project asked for a way to widen DASP's Optuna
search space (`HANDOFF_2026-08-29_DASP_SEARCH_SPACE.md`). Design work only - no code.
Ticket at `docs/plans/2026-08-30-T51-bayesian-opt-in-search-axes.md`.

**1. You cannot additively widen an axis the base sampler already suggests.**
Optuna raises if the same parameter name is suggested twice in one trial with a
different distribution. So an "add axes after the untouched sampler" design can ONLY
open hyperparameters that `suggest_model_params` pins as constants. Audit:
- Openable (pinned constants): XGBoost `reg_alpha`/`reg_lambda`/`colsample_bytree`
  (`unified_bayesian.py:765-767`), `gamma`/`min_child_weight` (absent); LightGBM
  `min_child_samples`/`subsample`/`colsample_bytree`/`reg_alpha`/`reg_lambda`
  (`:748-753`); RandomForest `max_features` (`:724`); SVM `gamma='scale'` - assigned,
  not suggested (`:788`); PLS-DA logistic head `C` (absent).
- NOT openable (already suggested): Ridge/Lasso/ElasticNet `alpha` (`:698`, `:706`,
  `:713`); MLP `alpha` (`:804`); OneClassSVM `gamma` - `suggest_categorical` (`:843`);
  PCA-SIMCA `n_components` (`:837`).
This killed a planned `linear_alpha_wide` bundle. Found by DeepSeek in peer review.

**2. Clamping a value AFTER `trial.suggest_*` does not change what TPE learns.**
Optuna stores whatever `suggest_*` returned in `trial.params`. Mutating the value
afterwards changes the fingerprint and the reported value, but TPE's KDE is still built
from the original suggestion. So the one-class `n_components` fix splits in two:
- clamp-before-fingerprint + record resolved value -> buys dedup and an honest `LVs`
  column, but does NOT change the search trajectory;
- deriving the ceiling and passing it INTO `suggest_int` is the only thing that changes
  what TPE sees, and that requires editing the sampler body.
An earlier draft of the ticket asserted the first would fix the trajectory. It would not.

**3. `_dasp_version` is already in the Optuna study-identity hash** (`:2563`), so a
version bump already orphans persisted studies. Two reviewers independently proposed
adding a new unconditional schema version for resume safety; unnecessary. The real
requirement is just that behaviour-changing PRs bump `__version__`.

**Also confirmed (not yet fixed)**: `SCALE_SENSITIVE_MODELS` contains `'SVC'` but the
registered classifier family is `'SVM'` (`models.py:281,498`; `model_registry.py:32`),
so classification SVM is fit with no StandardScaler. Sites: `search.py:156` (used
`:464`, `:4962`), `unified_bayesian.py:1534` (used `:1605`), `nsga2_search.py:1388`
(omits both), GUI `:40481`. Prerequisite for any SVM `gamma` tuning.

**Tooling note**: `gpt-5.6` is not reachable from a ChatGPT-account Codex login - it
hard-errors "not supported when using Codex with a ChatGPT account". `gpt-5.5` is the
only model that auth mode can reach. Switching would require API-key auth.

---

## 2026-07-30 - Second-round review of feat/agent-composition-guide: doc examples were wrong

**Context**: the branch's own commit log claimed a Codex + GLM 5.2 round had already
passed. A fresh independent round (Codex + GLM 5.2 via the Alibaba Token Plan route)
found the *guide itself* - the branch's headline deliverable - contained examples that
fail on first use. Do not treat a prior in-branch review claim as sufficient.

**Real, introduced by the branch** (both fixed here):
- `docs/AGENT_COMPOSITION.md` listed `get_uve_threshold` in the score-array family,
  whose documented return is a single `(n_features,)` array. It actually returns a
  3-tuple `(importances, threshold, selected_mask)` (`variable_selection.py:306`).
- `AGENT_COMPOSITION.md` section 8 saved the `MultiClassClassModel` bound in section 6
  under `"model_name": "PLS", "task_type": "regression"`, and never defined `X_new`.
  A false schema loads fine and then predicts down the wrong dispatch path.

**Real but PRE-EXISTING on main** (fixed opportunistically, not regressions):
- `README.md` `SubsetTag` claimed a fixed enum `all, top-20, top-5, top-3`. Actual
  tags are method-dependent: `full`, `top{n}_{method}`, `{method}_top{n}`, interval
  tags. Match by prefix, not equality.
- `docs/MACHINE_LEARNING_MODELS.md` had `get_model('NeuralBoosted', learning_rate=0.2)`
  - `get_model` takes no per-model hyperparameters; this raises `TypeError`.
- `README.md` clone/issues/citation URLs said `yourusername/deepspec`; repo is
  `makapania/dasp`.

**Review-method note**: Codex labelled 8 findings MERGE BLOCKER including items it
itself filed under MEDIUM/LOW, and flagged two pre-existing README/doc errors as
blockers introduced by the PR. GLM declared `AGENT_COMPOSITION.md` clean and missed
both real bugs in it, and flagged its own pre-existing find as a blocker without
checking it against main. **Always diff a claimed blocker against `origin/main`
before accepting it as introduced by the branch.** Both reviewers independently
confirmed the genuinely load-bearing facts: no live reference to the deleted CLI in
installer/.spec/CI/GUI, and the `run_search` 2-tuple docstring is true
(`search.py:4405`).

**Verification**: both fixed examples executed end-to-end against the real 49-sample
bone collagen dataset (UVE kept 142/2151 vars; save/load round-tripped 49
predictions). `tests/test_agent_composition_api.py` 53 passed.

## 2026-07-30 - GOTCHA: verifying from a git worktree silently tests main's src

`.venv312` has an editable install of `spectral-predict` that resolves the package
to a FIXED path: `C:\Users\mspon\git\dasp\src`. Running `python script.py` from a
git worktree therefore imports the MAIN checkout's source, not the worktree's -
silently, with no error.

This produced a false negative during the agent-composition review: a check that
exported a bundle and scanned it for a placeholder URL "failed" after the fix was
applied, because it was exercising main's `export_bundle.py`, not the worktree's.

Confusingly, `pytest` run from the worktree DOES pick up the worktree source (the
rootdir/conftest path insertion puts `src/` first), so tests can pass against branch
code while a plain script in the same directory silently tests main. Do not infer
from "pytest passed" that an ad-hoc script tested the same tree.

**When verifying branch code from a worktree, do one of:**
- `sys.path.insert(0, os.path.join(os.getcwd(), "src"))` at the top of the script, and
  assert the resolved module path contains the worktree directory before trusting the
  result - e.g. `assert "wt-name" in spectral_predict.__file__`
- or set `PYTHONPATH` to the worktree `src/`
- or re-run `pip install -e .` from the worktree (pollutes the shared venv - avoid)

Cheap habit that would have caught it immediately: print `module.__file__` and assert
on it as the first line of any verification script.

---

## 2026-07-30 - Agent-composition branch reviewed, fixed, and MERGED to main (`763c4ed`)

**What happened this session.** Reviewed `feat/agent-composition-guide` (CLI retirement +
`docs/AGENT_COMPOSITION.md` + public `multiclass_varsel_mask`), found and fixed real
defects, merged to `main`, pushed.

**Sequence:**
1. Independent second-round review dispatched: Codex, GLM 5.2, Qwen 3.8. The branch's own
   commit log already claimed a passing Codex + GLM round - it was not sufficient.
   GLM's first dispatch stalled (opencode-go subscription at limit) and was surgically
   re-routed to `glm-alibaba` (Alibaba Token Plan); Qwen ran on the same plan.
2. Fixed 2 real regressions the branch introduced, both in the new guide (`29adaf3`).
3. Fixed 2 Qwen nits (`2a170bb`) - one of which was under-rated as cosmetic and was
   actually consumer-visible (the `"preprocessing"` metadata key).
4. Fixed pre-existing placeholder repo URLs incl. live code in `export_bundle.py`
   (`e36a579`), closing the tracked QUICK_WINS P2.
5. Recorded the worktree/editable-install gotcha (`2c52a48`).
6. Merged `--no-ff` to `main`, pushed, re-ran `pip install -e .` to clear the stale shim.

**CLI: abandoned, user decision 2026-07-30.** Codex recommended keeping `cli.py` for one
release as a deprecation stub printing a migration path; the user declined outright
("we are abandoning cli for now"). Clean removal stands. Do not re-add a console script.

**ACTION REQUIRED ON EVERY OTHER MACHINE** on first pull of `763c4ed` or later: run
`pip install -e .`. `git pull` deletes `cli.py` but leaves
`.venv312/Scripts/spectral-predict.exe` behind, which then raises
`ModuleNotFoundError: No module named 'spectral_predict.cli'`. Full first-pull checklist
is now at the TOP of `docs/PROJECT_STATUS.md`.

**Merge-safety basis.** `main`'s CI is red and has been since ~June 2026 (T-CI-1 rot), so
a green check was not available. Merged on failure-set-diff instead: ran
`test_export_code.py` on the branch AND on the untouched `main` checkout and got an
IDENTICAL 2-failure set. Targeted suites 75 passed. Both corrected doc examples executed
end-to-end against the 49-sample bone collagen dataset; export bundle generated and
scanned for the placeholder URL (zero hits).

**Post-merge check that caught my own sloppiness:** I first reported a matplotlib leak on
`import spectral_predict`. Wrong - my check had also imported `spectral_predict.search`,
which legitimately pulls matplotlib. The bare-import guarantee is intact. Test the exact
claim, not a superset of it.
## 2026-07-29 — Agent-facing API: CLI retired, composition guide added

**Context:** user asked whether the repo is in a good state for AI agents to run analyses headlessly. Investigation reframed the question twice; the plan that survived was collapsed on the user's own argument plus a Codex review.

**The `spectral-predict` CLI was dead, with FOUR independent defects — not the "3-line fix" it looked like.** Verified by running it: (1) `read_csv_spectra`/`read_asd_dir` return `(df, metadata)` tuples while `cli.py:190,196` treated the result as a DataFrame → `AttributeError`; (2) `import sys` inside an `except` block shadowed the module-level import, so the top-level handler raised `UnboundLocalError` and **masked every real error** — the single most turn-wasting defect, hit on three unrelated failures; (3) `lambda_penalty=` passed to `run_search`, which takes `variable_penalty` and has no `**kwargs` → `TypeError` (found by Codex); (4) `run_search` returns a **2-tuple** `(df_ranked, label_encoder)` but the CLI assigned it to `df_ranked` and called `.to_csv()`. Only `--help`/`--version` ever worked, which is exactly what `tests/test_cli_help.py` covered — hence years of silent rot. **Retired rather than repaired**, on the user's argument that a CLI can only encode a fixed analysis shape and *there is no such thing as a standard analysis*.

**Agents deliberately bypass the orchestrators, and that is correct.** An active agent-driven research program (528-combination DD-SIMCA search, Bayesian PLS-DA, 32-page manuscript validated on 948 external objects) never imports `run_search`/`run_one_class_search`. It imports primitives (`simca.MultiClassClassModel`, `contamination.PCASIMCA`, `variable_selection.cars_selection`, `preprocess.build_preprocessing_pipeline`) and writes its own orchestration + `StratifiedGroupKFold` CV. This is the scikit-learn pattern and it is right for research: each project owns its sampling design, splitter, ranking objective, and reporting. **Conclusion: do NOT build an orchestrator or a config-file runner.** An earlier draft plan proposed declaring `__all__` across all modules + `__init__.py` re-exports + a full private-function audit; the user and Codex both rejected it as work that would not change what agents can do. Codex additionally noted top-level re-exports would pull heavy optional deps into `import spectral_predict` and erode its headless-safety.

**Root cause of the fragility that DID need fixing: no declared public API.** `search.py` is 5 public / 24 private; only `model_io.py` (plus `readers/` and `templates/` package inits) declared `__all__`; `__init__.py` exported only `__version__`. So composing a novel analysis *required* reaching into internals — the external pipeline imported the private `search._multiclass_varsel_mask`, which a rename here would have broken silently on another machine. Fix: promoted to public `multiclass_varsel_mask` with `_multiclass_varsel_mask` retained as a delegating alias, plus `__all__` on `search.py` only (selective, not a sweep) and a repo-local contract test.

**Undocumented traps found by actually executing the doc examples (5 of 10 first drafts failed).** Do not write API docs from inspection: (a) `run_search`'s `preprocessing_methods` is a **dict of bools** (`{"raw": True}`), not a list of strings — a list raises `AttributeError: 'list' object has no attribute 'get'`; (b) valid `build_preprocessing_pipeline` names are only `raw|snv|deriv|snv_deriv|deriv_snv` — `snv_deriv1` raises; (c) variable selectors return **importance score arrays** of shape `(n_features,)`, not boolean masks; (d) `save_model` requires metadata keys `model_name`/`task_type`/`wavelengths`/**`n_vars`**; (e) `run_search`'s docstring claimed a bare DataFrame while returning a 2-tuple — docstring corrected. Also: `example/` holds **49** ASD files, not the 37 the README claimed.

**Grouped CV remains a real backend gap (T-15), deliberately not closed.** No search entry point accepts `groups`, and `cv_utils.py` raises `NotImplementedError` for `group_kfold`/`leave_one_group_out`. Agents compose around it with their own splitter, so it is documented in `AGENT_COMPOSITION.md` rather than plumbed — closing it would make `run_search` usable for grouped designs but buys no flexibility that scripts don't already have.

**`interactive.py` / `interactive_gui.py` are now orphaned** — they were imported only by the retired `cli.py`. Left in place (deleting needs explicit permission); their docstrings now say so, since they otherwise read as live API.

**Review lesson — executing examples and reading source catch DIFFERENT classes of doc error. Do both.** Every example in `AGENT_COMPOSITION.md` was executed (17/17 green) and two reviewers still found real inaccuracies that execution structurally could not catch:
- **Codex:** the guide said "all selectors take `(X, y, ...)` and return an `(n_features,)` importance array", then listed `ipls_forward`/`ipls_backward`/`mc_sipls`/`mwpls` among them. Those are a *different family* — `wavelengths` is a required THIRD POSITIONAL arg and they return a **list of subset dicts**. Cause: I verified the four selectors I exercised and generalised the claim to the ones I had only checked were importable. Codex found it by reading `variable_selection.py`.
- **GLM 5.2:** the guide promised `n_select` could be omitted while the signature made it a required positional — following the doc raised `TypeError`. Fixed by making the signature match the doc (`n_select=None`); the body already handled `None`.
- **GLM 5.2 (subtler):** the guide listed `rank` as an always-present key on interval-subset dicts. It is absent on `ipls_forward`'s *combined-interval* entries (`variable_selection.py` ~1855-1862) — my test run stopped early and produced only single-interval entries, so **execution reported the key as always present**. Also `tag` was undocumented. Only source-reading catches a conditional key that a given run happens not to exercise.
- **I also reintroduced the very trap the guide exists to prevent:** my rewrite of `docs/MACHINE_LEARNING_MODELS.md` wrote `df = run_search(...)` three times, right after discovering `run_search` returns a 2-tuple. Caught by Codex.

**Reviewers disagreed on the back-compat alias.** GLM wanted a `DeprecationWarning` wrapper on `_multiclass_varsel_mask`; Codex explicitly argued against (would break warning-strict callers, adds noise). Kept it a silent alias — its whole purpose is to not disturb a live off-repo research pipeline. Revisit only if the private name is actually being retired.

**Pre-existing unrelated failure:** `tests/test_cv_strategy.py::TestPostMergeReviewFixes::test_classification_metrics_template_has_no_nameerror` fails with `NameError: name '_fit_fold' is not defined` — verified identical on `main`, so not from this work.

**GUI tests spawn Tk windows and closing them kills the run.** A full `pytest tests/` background run died at 36% with exit 127 when the user manually closed stuck analysis windows. Run `pytest tests/ --ignore=tests/gui` for background/unattended verification (this is also what the repo's Linux CI does); run `tests/gui` only when someone is expecting windows to appear. Also: don't run the GUI suite at all for a change that touches no GUI code — that was needless here and cost the user manual cleanup.

**"We didn't touch the GUI" is a claim to CHECK, not assume — the GUI imports PRIVATE backend names.** `spectral_predict_gui_optimized.py:30550` does `from spectral_predict.search import _WOLD_METHODS, _multiclass_preprocess_matrix, _multiclass_varsel_mask, build_multiclass_decision_view` and calls `_multiclass_varsel_mask` at `gui:30573` in the decision-view rebuild. A hard rename of that private function would have broken the GUI silently — the back-compat alias added for the off-repo research pipeline is what saved it. Before claiming a backend change is GUI-safe, grep the GUI for the symbol; it reaches past the public surface.

## 2026-09-12 - Both bugs fixed, and two verification failures worth remembering

Implemented the Codex review below. Notes on the parts that were not obvious.

**An unreadable version must be fatal, not a placeholder.** The Optuna fingerprint
raises `EnvironmentFingerprintError` when a tracked distribution's version cannot
be read. Degrading to `"unknown"` would make two DIFFERENT broken environments
hash identically and therefore resume-compatible - the exact bug the fingerprint
exists to prevent. A genuinely ABSENT package is different: that is a definite
fact, recorded as `"absent"` and hashed.

**Enumerating Optuna studies CREATES the SQLite file.** The 'previous results were
computed elsewhere' notice originally ran whenever a storage URL existed, which
broke the 'never' and 'auto'-warmup promise of staying purely in memory. Two T-41
tests caught it (`test_auto_picks_in_memory_no_db_file`,
`test_never_mode_in_memory_no_db`). Now gated on `always`, which is also the only
mode that actually resumes before a trial runs.

**A revert that silently does not apply produces a fake proof.** Verifying that the
new MultiGroupEPO tests actually FAIL against the old code, the revert was written
with `
` line endings against a CRLF file, so `str.replace` matched nothing. The
tests 'passed against the buggy code' because the buggy code was never restored.
**Assert the mutation happened** (count occurrences before/after) rather than
trusting a replace, and do byte-level edits on CRLF files.

**A test can pass for the wrong reason.** `test_group_labels_are_sorted` used the
probe's forward group order, which happened to ALREADY be sorted - so it passed
with the bug present. It only distinguishes sorted-vs-insertion order when the
input is deliberately unsorted.

**`run_unified_bayesian`'s third positional is `wavelengths`, not `task_type`.**
Passing `'regression'` there makes every trial fail with `IndexError: too many
indices for array: array is 0-dimensional`, and the run returns an empty
leaderboard rather than raising. That looks exactly like an upgrade regression.
Signature: `run_unified_bayesian(X, y, wavelengths, model_name, task_type=...)`.

**Not fixed, deliberately:** the GUI 'Apply EPO' path builds `EstimatedEPO`
(GUI:58521) with `random_state=None` (`contaminant_analysis.py:462`) and is still
nondeterministic. The full cross-version SQLite replay matrix is also not built;
current coverage is the digest's sensitivity plus the existing T-41 storage tests.

---

## 2026-09-12 - Codex review: numerical-environment resume and MultiGroupEPO seeds

Evaluation only; no application source edits. Both pre-existing bugs are real.
Recommendation: FIX NOW for both, but Optuna needs explicit old-cache retirement,
not merely an extra version string silently changing the study name.

- Optuna persistence defaults to auto in both GUI and backend; after 10 trials it
  migrates if median completed-trial duration exceeds 1 second (or fewer than 3
  completed). GUI crash recovery restores the storage URL and forces always mode.
  A bare backend call without active run_state has no disk persistence.
- Reproduced on a synthetic temporary SQLite: Python 3.12.10 / numpy 2.4.4 /
  sklearn 1.8.0 / Optuna 4.8.0 created one completed PLS trial; Python 3.14.7 /
  numpy 2.5.3 / sklearn 1.9.1 / Optuna 5.0.0 reopened the same named study and
  replayed its exact score as trial 1 with ZERO CV calls. This also affects TPE
  history, trial-budget accounting, and old leaderboard rows, so changing only
  trial fingerprints is insufficient. Source: unified_bayesian.py:2562, 2685,
  1672, 2912, 3074; GUI:23902.
- Small fix design: keep the existing config-only name as a base; append a stable
  environment digest, persist the unhashed environment in study.user_attrs before
  trials, and warn through logging plus progress_callback if resuming instead
  starts a fresh environment-specific study. Never reuse legacy scores with
  unknown provenance. Leave old study rows/databases intact. Old studies cannot
  continue under the new identity even when their actual environment happens to
  match: this is intentional one-time cache invalidation, not database corruption.
- MultiGroupEPO's hash(label) at contaminant_analysis.py:2323 drives the library,
  SVD, projection at :2376, and transform return at :2405. Two subprocesses with
  identical synthetic input and PYTHONHASHSEED=1/2 differed by max abs 1.976 in
  transformed values. An in-memory stable blake2b replacement produced identical
  library/projection/transformed data across those processes. Reversing dict
  insertion order still produced 5.3e-15 transform differences; sort string keys
  during validation to fix numeric assembly order too. No second hash() call was
  found in src/spectral_predict.
- Blast-radius nuance: analyze_multiple_contaminants returns this transformer
  under results['epo'], but computes combined_influence/exclusion_regions through
  a SEPARATE MultiContaminantAnalyzer(random_state=42), at :2721. The GUI displays
  that combined result (:57677) and Apply EPO uses EstimatedEPO (:58531), not the
  MultiGroupEPO object. EstimatedEPO defaults random_state=None (:469), including
  that GUI call, and remains a separate randomness issue after the hash fix.
- Existing focused suite: 110 passed (test_bayesian_dedup,
  test_t41_bayesian_sqlite_auto_calculator, test_t42_write_path_plumbing,
  test_contaminant_analysis). Existing MultiGroupEPO tests mostly assert shapes;
  add cross-process transformed-output and dict-order tests. Add end-to-end
  SQLite resume tests for same environment, changed environment, legacy study,
  and auto-migration preserving environment metadata.

## 2026-09-12 - PR #65 performance/safety review (Codex)

Review target is e13393f against main 8de7445; Python 3.14.7 in .venv314
passes pip check. The 125 focused environment/EPO/dedup/SQLite/contaminant tests
pass, as do 9 JCAMP tests (1 skipped).

Two non-obvious review findings (subsequently verified and quantified below):
- The new compatibility notice asks Optuna for every study SUMMARY but only uses
  names. Installed Optuna 5.0.0 study/study.py:1594 loads ALL trials for EACH
  study, including their arrays and user attributes. This adds a whole-database
  scan before every always-persistent model search, including unrelated models.
  get_all_study_names avoids loading trial history entirely.
- _query_build_python splits stdout on all whitespace, truncating a site-packages
  path containing a space. The pandas post-build repair then silently skips
  because its source path does not exist, and falsely reports a match. The
  existing log records this repair as necessary on all three 3.14 builds.

PR #65 verification update:
- Build helper replay with `C:\Users\Jane Doe\dasp\.venv314\Lib\site-packages`
  returned `C:\Users\Jane`. This confirms the whitespace parsing regression.
- Three-study SQLite with 240 realistic payload-bearing trials (11.4 MiB):
  median study-summary scan 0.205 s versus name listing 0.0197 s (3 repetitions).
- Real PLS SQLite calls: the same numerical environment resumed the same study;
  mocking numpy's installed version created a different study and emitted the
  incompatibility notice. No prior database was modified by these probes.
- An attempted pybaselines fingerprint probe was NOT valid evidence: Bayesian
  apply_preprocessing only implements polynomial/als/rubber_band/airpls and
  silently ignores `asls`. Missing pybaselines is not a proven fingerprint defect
  on this execution path. Do not promote an unexercised dependency to a finding.
- The existing dist bundle predates dcde845: bundled unified_bayesian.py contains
  no ENV_FINGERPRINT definitions and its contaminant_analysis.py also differs
  from HEAD. Its executable timestamp is 15:46; the bug-fix commit is 17:23.
  A --test run on that artifact cannot validate the final PR's two bug fixes.

PR #65 review complete: three P2 findings, no application edits. The third is
RUN_SPECTRAL_PREDICT.bat:25-27: a failed lock install is masked by a successful
editable --no-deps install. A temporary copy of the actual launcher, with only
Python calls replaced by a controlled batch stub, reached gui_launched and
returned 0 after the lock install returned 1. install.bat already handles this
correctly. Fix both command error checks in the launcher.

Two fixed-workload timing rounds (3.12 then 3.14, then reversed), six measurements
per environment after warmup, show RandomForest 2.633 -> 2.014 seconds (24% faster)
and XGBoost 1.887 -> 2.160 seconds (14% slower). Single-thread 240 x 512 synthetic
regression, 3-fold CV; all five tested models' RMSEs agree across stacks. These
are combined-stack workload measurements, not general GUI performance claims.
Full evidence, boundaries and next steps are in
[the review](reviews/2026-09-12-pr65-performance-safety.md).

## 2026-09-12 - PR #65 name-only lookup audit and Fable review

The summaries really do populate _CachedStorage's trial cache, but for the URL
string passed here Optuna constructs a NEW temporary storage instance. That cache
is discarded after the summary call; _existing retains only strings. Actual
resume uses separately constructed create_study/load_study storage objects, and
rehydrates fingerprints, sampler history, result rows and arrays from study.trials.
Both enumeration APIs use the same RDBStorage constructor and get_all_studies;
keep the existing always-mode gate because both can initialize a new database.

Requested Fable review completed through the read-only wrapper; modelUsage
confirms claude-fable-5-1. Fable independently confirms the replacement is safe
and all three prior findings are real; it characterizes the scan as avoidable
cost rather than a correctness bug. Fable did not rerun the timing or SQLite
probes, and requests a regression check for the incompatible-environment notice.

Probe gotcha: Optuna 5.0 removes Study.set_system_attr (not just deprecates it).
The first preservation probe stopped at fixture setup for that reason. Use the
public RDBStorage system-attribute methods for fixture metadata. The subsequent
Windows cleanup error was an open SQLite handle, not evidence of data loss.

Name-only lookup verification PASSED using the exact replacement compiled only
in memory (no application source edit):
- Complete SQLite dumps identical before/after either enumeration. SQL trace:
  summaries read two studies' trial tables; names read none; neither wrote data.
- Original and proposed functions resumed the same 24-trial study with all
  stored fields unchanged, including arrays, parameters, distributions, dates,
  user/system attrs and 23 fingerprints; identical 23-row leaderboards.
- Continued original/proposed database copies to 26 trials: identical new TPE
  suggestions, scores, attrs, arrays, leaderboards and progress callbacks.
- Same warning for a retained legacy study; its original rows/metadata preserved.
- auto warmup and never mode performed no lookup and created no SQLite database.

Fable's full returned opinion is saved verbatim in
[the Fable review](reviews/2026-09-12-pr65-fable.md). Codex agrees with its safety
conclusion. Its claim that the benchmark differences are specifically library
version effects is too strong: our benchmark changed Python and dependencies
together, so it cannot isolate those causes. All checks used disposable SQLite
fixtures; no existing studies or environments were changed. The proposed source
change remains unapplied, consistent with this verification/review request.

## 2026-09-12 - Implementing the three PR #65 review fixes

Added ten focused cases before touching application code: five failed against
HEAD, reproducing the whitespace-path pandas corruption, missing repair inputs
being called a successful verification, lock-install failure reaching the GUI,
and resume using the full-summary API. The other five protect existing behavior.

First launcher attempt added a nested failure/exit block immediately after the
lock command. It prevented GUI startup but the controlled cmd.exe batch probe
still returned 0 on that first failure path. Replaced both pip error paths with
a shared failure label outside the conditional, matching install.bat's pattern;
keep the exit-code assertion, not just the 'GUI did not start' assertion.

All ten new cases now pass, and the combined focused suite is 77 passed
(environment fingerprint, Bayesian dedup, T-41/T-42 persistence and build version
checks included). Black with target py314 and flake8 pass for the new test files.
The production fix keeps the always-mode gate and uses only study-name lookup;
resume still reloads the selected study. Build-path lines are split with
splitlines(), missing repair source or bundle now fails verification, and both
launcher pip commands branch to the shared nonzero failure exit.

A fresh standalone/installer build is running to replace the artifact that
predated dcde845. Existing installed application and environments are untouched.

The fresh PyInstaller/Inno build completed successfully and actually repaired
the pandas.util TOC collision. All 75 bundled project Python files match the
working source byte-for-byte, and the repaired pandas module matches .venv314.
The hidden executable --test returned 0 after 16.84 seconds. The capture helper
then hit a cp1252 UnicodeEncodeError while printing the saved log, after the
executable had already completed; inspect the saved output with UTF-8 console
encoding instead of rerunning the completed test.

Saved smoke-test output confirms ALL TESTS PASSED: 42/42 imports, functional
XGBoost/LightGBM/CatBoost fits, active frozen threading fallback and a completed
99-row PLS/LightGBM cross-validated search. Installer size is 230,106,755 bytes
(219.4 MiB), SHA256
171742e9f918ee776416d12cc25998a5a64d1b4f86e597ecc17d70039f021c9a.
No clean install, installed-app upgrade or uninstall was performed. The full
suite was not repeated; the 77 focused tests and new bundle smoke test passed.

## 2026-09-12 - Final PR #65 review and installation gates

User authorized the final Fable review, installation/upgrade checks and merge if
they pass. A second, separate read-only Fable opinion is running on the final
89f9479 implementation and its regression tests. The GitHub connector works even
though gh CLI authentication fails: PR is open/mergeable, CodeRabbit passed, and
the current Actions run failed; inspect job evidence before treating it as the
documented baseline.

The status guide's statement that this machine has an existing installed copy
is stale. Elevated read-only inventory found no Spectral Predict uninstall entry
in HKCU/HKLM (including WOW6432Node) and no Program Files installation. The desktop
entry is a shortcut to RUN_SPECTRAL_PREDICT.bat. Windows is Home and has no Windows
Sandbox. Use an isolated test install and a verified prior release/build as the
upgrade baseline; do not describe a source checkout as an installed-app upgrade.

Fable's final review found no merge blocker in 89f9479. It noted a low-severity
Optuna floor mismatch (name-only API starts in 3.4; pyproject allows 3.0), plus
informational launcher newline and test-isolation observations. Its full output
is preserved in reviews/2026-09-12-pr65-fable-final.md. The substantive reviewer
was claude-fable-5-1; modelUsage also records a 20-token internal Haiku call.

The real installer completed a clean per-user install in the isolated workspace
with exit 0 and no reboot. The @oai/sky native pipe was unavailable on two tries;
use the installed runtime and packaged GUI code for an automated integration
check if recovery also fails, and distinguish this from desktop click testing.

Computer Use recovery after a kernel reset also failed (native pipe missing).
The installed executable's unchanged --test passed (42/42 imports and all
functional checks). A separate installed-runtime GUI probe reached the real Run
Analysis callback but failed importing logging.handlers from run_logging. This
is not yet attributed to the package versus the custom test host: inspect the
executable's PYZ, base-library zip and loose modules before deciding. Adding a
logging package search path alone cannot supply a file absent from the bundle.

Confirmed the GUI failure is a packaging defect: logging.handlers is absent from
the executable PYZ and loose modules, and PyInstaller's warning file lists
spectral_predict.run_logging (and many other backend modules) as unresolved.
The spec gives Analysis only the repository root despite the src layout. Runtime
sys.path insertion permits backend imports but cannot retroactively discover
their dependencies during freezing. Add src to Analysis.pathex and extend the
real --test with run_logging/run_state imports and model save/load/prediction.
The clean-test install was uninstalled successfully (exit 0). The baseline main
build on retained .venv312 also completed, including its installer.

The original installer reproduced the overlay defect on an actual baseline
installation: it retained 3,309 obsolete runtime files, python312.dll and both
numpy dist-info versions. The upgraded runtime reported numpy.__version__=2.5.3
but importlib.metadata.version('numpy')=2.4.4. This is a concrete cache-identity
safety issue, not just excess disk usage. The saved legacy model and user-note
sentinels survived. Add InstallDelete only for the app-owned {app}\_internal
directory, preserve the rest of {app}, and add runtime-versus-metadata checks to
--test. Validate both upgrade and uninstall preserve those user-file sentinels.

Live CI comparison is now exact: on Windows the PR and base main have the same
five failing node IDs; Linux has the same three non-GUI failures. The optional
dependency job has those same three. The informational Linux GUI job times out
on the pre-existing XGBoost GUI case on both commits. All ten new tests passed
in the PR's Windows CI (3,002 passed total versus base 2,977); package build passed.

Updated focused suite: 81 passed. The extended source --test passes 44/44 imports,
all runtime/metadata comparisons, booster fits, search and the model round trip.
Executing that exact extended test function against the original upgraded
installation correctly fails eight checks: GUI logging plus seven mismatched
numerical-package metadata versions. This validates that the added gates detect
both reproduced packaging defects. The corrected bundle/installer build is in
progress. Only this project's editable metadata was refreshed in .venv314 (no
dependency installation or version change); pip check remains clean.

The additional concurrent review commit f60cfa5 was explicitly reviewed after it
was noticed in the latest history. It is a parent of db975e2 and was included in
the rebuilt artifact and 81-test run. Its five EPO determinism tests also pass;
six Git Bash launcher control-flow probes pass with stubbed Python (no pip).

The corrected in-place upgrade passes a full SHA256 comparison of all 20,384
runtime files against the build, with zero missing, extra, or different files.
The legacy model hash remains identical. Installed --test passes 44/44 plus
metadata, fits, search, and model round trip. The installed Tk callback harness
completes PLS analysis and GUI model loading, and reproduces all 30 predictions
from the baseline 3.12 saved PLS model (1e-12 tolerance). Scikit-learn emits its
expected cross-version warning; this is evidence for this model, not a general
promise of pickle compatibility. Uninstall exits 0 and removes the executable,
Python DLL, and registration while retaining the model and note. The first note
check falsely failed because PowerShell compared the existing CRLF fixture to
an LF literal; byte inspection confirmed the expected text and CRLF.

The corrected fresh installation also matches all 20,384 runtime files by SHA256,
passes the actual executable --test and installed Tk analysis/model loader, and
uninstalls successfully. Both model and note hashes match before/after the fresh
uninstall; application registration and runtime are removed. All 75 loose bundled
backend source files match the checkout. Final installer: 230,295,124 bytes, SHA256
3ef6ed77e422a8ee0792c21b3a88c24ba3365469fbd10703969707921f321913.
The final PR description is prepared, but the GitHub connector refused its
update with HTTP 403 Resource not accessible by integration. The first response
was mistakenly summarized without checking isError; a read-back showed the old
body, and the full retry response confirmed the permission failure. Check an
existing local/browser authentication route for the authorized update and merge.
The latest application CI (34735172751, db975e2) is still running; build passes
and its informational GUI timeout matches base. Leave this CI run undisturbed
while finishing the report; final documentation will be committed and pushed.

The existing local GitHub CLI login works outside the restricted process; the
connector itself lacks PR-write access. gh pr edit succeeded, and gh pr view
confirmed the updated body and unchanged db975e2 head. No new login or credential
was needed. The test-only baseline .venv312 junction was removed without recursion;
the real rollback Python executable hash was verified unchanged. Only generated
baseline build/bundle copies were cleaned up; the baseline installer, test logs,
small evidence files, current deliverable and real .venv312 are retained.

Installed-GUI harness gotcha: setting APPDATA/LOCALAPPDATA isolates logs/state,
but the GUI output_dir still defaults to cwd/outputs. The two successful harness
runs exported three-row, 120-variable PLS tables there at 21:32:05 and 21:39:07.
Those exact test-only CSVs were verified and moved into the isolated evidence
directory. Future GUI probes should also set app.output_dir to a fixture path.

At 05:06 UTC the three full CI jobs were still running. An attempt to fetch
the in-progress Windows job log returned GitHub BlobNotFound (404), so its
failure list is not available yet. This is log availability, not a test result.
The monitor continues checking both PR head and base while awaiting completion.

Latest optional-dependency CI completed at 05:12:52 UTC: 3 failed, 2,880 passed,
33 skipped. Its three failing node IDs exactly match base; zero new failures.
Windows and primary Linux jobs are still running. The user requested that the
completed work be committed now. Commit/push the validation documents separately
from the already-pushed application db975e2; documentation-only changes do not
require another two-hour application test run. The final merge remains pending
the two remaining failure-set comparisons.

Final application CI completed at approximately 05:14 UTC. Windows: 5 failed,
3,008 passed, 29 skipped; Linux and optional dependencies: 3 failed, 2,880 passed,
33 skipped each. Every failing node ID exactly matches base main; no collection
errors and zero new failures. Build passes and informational Xvfb timeout is the
same baseline XGBoost GUI case. All authorized merge gates now pass.

The first documentation commit attempt stopped on a transient .git/index.lock.
The immediate inspection found no remaining lock or Git process; no lock file
was deleted, and the four staged documentation files remained intact. Retry the
commit with these completed CI results, then push and perform the authorized
merge using an expected-head guard.

Validation documents were committed and pushed as fbe8e95. Another active
session immediately added ecad576 on top, which explains the initial remote-head
verification mismatch. Its sole change is a 14-line PROJECT_STATUS review note;
reviewed it and verified db975e2..ecad576 changes documentation only. The final
PR description was updated with the completed zero-new-failures CI comparison.

The attempted ordinary merge of PR 65, guarded to ecad576, was rejected BEFORE
execution by automatic approval review. Stated reason: merging into the default
branch is consequential and explicit authorization for that exact merge side
effect was not found. No alternate merge route was attempted. All code, tests
and documentation are complete; the PR remains open pending explicit user
confirmation to merge PR 65 into main. The shared checkout stays on the feature
branch for the other active session.

### 2026-09-13 — PR #65 merged

User explicitly authorized the merge. PR #65 merged into main as 6b956c8 (merge
commit, --match-head-commit 8bd3f8f; 8bd3f8f was docs-only on top of the CI-tested
db975e2). CI failure set at db975e2 identical to main 8de7445 (same five tests;
zero new). feat/python-314-upgrade branch retained. Deferred from review:
fingerprint failure still aborts never-mode runs (deliberate); consider_endpoints
deprecation needs a numerical A/B before removal. Gotcha: gh pr merge
--match-head-commit rejects short SHAs ("Could not coerce value to GitObjectID").

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
- **Gotcha:** `pyproject.toml` sets `pythonpath = ["src"]`, which is prepended ahead
  of `PYTHONPATH`. Pointing `PYTHONPATH` at another source tree does NOT test that
  tree — swap the file in the worktree instead.
- **Gotcha (agents):** opencode read-only mode rejects reads outside the repo root,
  including scratchpad worktrees and temp files. Tell it to read branch code inline
  (`git show origin/<branch>:<path>`) and run tests yourself. Git Bash also strips
  backslashes from Windows paths in prompts — use forward slashes.

### 2026-09-14 — Test-order leak root cause (PR #70)

`test_bayesian_study_lookup.py` binds `run_state` at import and monkeypatches
`run_state.get_storage_url`; `unified_bayesian` imports `run_state` lazily from
`sys.modules`. Fixtures that `sys.modules.pop()` + re-import without restoring left
the fresh copy registered, so the patch hit a module nobody used. Fix: shared
`reimport_modules` fixture in `tests/conftest.py` that registers each entry and the
package attribute with monkeypatch.

### 2026-09-14 — T-51 PR B0: PLS-DA head-param plumbing (branch fix/T51-b0-plsda-head-params)

Shared `models.split_plsda_params(params) -> (transformer_params, head_params)` plus
`models.PLSDA_HEAD_DEFAULTS` now feed build_model, `_rebuild_model_from_row`, Tab 7
refit, ensemble reconstruction and the exporter. Gotchas found while verifying the
plan's hop table on main (plan §3.2 was partly wrong):
- **`PLSTransformer.set_params` never raises.** It overrides BaseEstimator's and
  `setattr`s any key, so legacy `lr_C` pushed into it becomes a junk attribute, not a
  ValueError. Only the constructor (`build_model`) raises TypeError. Tests detect leaks
  via `vars(pipe.named_steps['pls'])`.
- **Tab 7 canonical rows were already right.** `_apply_pipeline_params_to_pipe`
  re-applies every `lr__*` / `pls__*` key found in `pipe.get_params(deep=True)` after
  the pipeline is built (all three build paths). Only legacy `lr_*` rows lost
  C/solver/max_iter there. Worth a ticket: that re-apply also clobbers a UI override of
  `n_components` and applies a stored `lr__class_weight`.
- **Real grid/Bayesian rows are canonical.** Grid `params` is replaced by the fitted
  pipeline's full `get_params()` (search.py ~5340), so legacy `lr_*` rows only come from
  a failed capture or old CSVs. Validation rebuild was the consumer that dropped C for
  current rows.
- **Fifth consumer not in the plan:** GUI `_reconstruct_models_from_results` (ensemble
  training) hard-coded `LogisticRegression(max_iter=1000)` and applied only
  `n_components`. Fixed with the same helper.
- Search-time construction (search.py ~4952, unified_bayesian.py ~1724, nsga2,
  ga_preprocessing) reads legacy `lr_C` from the grid/suggest dict and is correct;
  untouched.

### 2026-09-14 — opencode DeepSeek route hangs

Two `opencode run --agent readonly -m deepseek/deepseek-flash` reviews (PR #71, #72)
printed only the header and hung 10-20 min with no tool calls; GLM 5.3 Flash and Kimi
K2.6 ran equivalent prompts fine in the same window. Killed the two processes and
re-ran those reviews on Kimi. Until re-checked, don't rely on DeepSeek via opencode;
pre-fetch refs and forbid `gh`/`git fetch` in agent prompts to rule out prompts/network.

### 2026-09-14 — CatBoost `catboost_info/` (fix/catboost-no-train-dir)

- **Root cause:** no production CatBoost construction passed `allow_writing_files=False`,
  so every fit created `catboost_info/` in the cwd. Unwritable cwd (Program Files
  install) or a concurrent-fit race makes the fit raise. Portable repro: a regular
  *file* named `catboost_info` in the cwd (error then reads `Can't create train tmp
  dir: tmp`).
- **Gotcha: CatBoost `get_params()` echoes every explicitly passed constructor kwarg**
  (it returns `_init_params`, not all defaults). A runtime kwarg set at construction
  therefore leaks into every `get_params()` capture: grid `Params` (search.py
  `_run_single_config`), Bayesian `_capture_serializable_params`, NSGA-II
  `decode_solution`. Fix: `models.CATBOOST_RUNTIME_PARAMS` + `strip_runtime_params` at
  those three capture sites; `with_catboost_runtime_params` merges (explicit wins) so a
  stored dict carrying the key never raises a duplicate-kwarg TypeError.
- **The failure was silently swallowed in most paths:** preprocessing-discovery tree
  importance falls back to LightGBM, diagnostics validation curve yields NaN scores,
  Bayesian returns the 1e10 penalty, `run_search` drops the configs.
- **`run_search(models_to_test=[...])` only filters the tier's grid.** CatBoost is not
  in the `quick` tier, so `models_to_test=["CatBoost"]` raises "No valid models found"
  unless `enabled_models=["CatBoost"]` is also passed. AGENT_COMPOSITION §7 says
  `models_to_test` "overrides tier", which is misleading.

### 2026-09-14 — T-51 PR B: supervised bundles (branch feat/T51-pr-b-supervised-bundles)

- **A bundle that breaks a fit does not fail a real run.** The objective's broad handler
  turns any fit error into a `1e10` penalty trial, so `run_unified_bayesian` "succeeds"
  with every trial penalised. Bundle tests must assert `trial.value < 1e9` on every
  completed trial (as `test_t9_real_run_rows_match_search_time_model` does), not just
  that a dataframe came back.
- **`families` x `task_types` is a cartesian product.** `svm_gamma` (families `{SVM,SVR}`,
  both tasks) therefore also resolves for `SVM`+regression and `SVR`+classification.
  Harmless: `build_model` has no estimator for either, so such a run fails regardless.
  Tests skip those two combos (`_NO_ESTIMATOR`).
- **`plsda_head` applies to `PLS-DA` only, not `PLS`+classification**, although the
  objective builds the same PLS-DA pipeline for both spellings. The rebuild and Tab 7
  paths key on the literal `'PLS-DA'`, so opening the head for `'PLS'` would store an
  `lr__C` that `_rebuild_model_from_row` ignores. Left as the plan specifies.
- **`lgbm_child` on small data can flatten the search.** Recipe data (45 rows, 3-fold,
  30 train rows per fold) with `min_child_samples=34` gave three identical trial values:
  no tree can split. The bundle's help text warns about it; it is not a bug.
- **Constructing the search-time model in tests.** `optuna.trial.FixedTrial(trial.params)`
  replays `suggest_model_params` + `apply_extra_axes` for a stored trial, and
  `build_model` plus the objective's pipeline wrap (scaler for SVM/SVR/MLP, the
  `pls/scaler/lr` pipeline for PLS-DA) reproduces `trial.user_attrs['model_params']`
  exactly. The round-trip tests rely on that, and it is asserted against real runs.
- **Pre-existing, not traced: Tab 7 refit of a Bayesian XGBoost row logs
  `Parameters: { "model__colsample_bytree", ... } are not used`.** Some Tab 7 step hands
  `model__`-prefixed keys to the bare `XGBRegressor`, whose `set_params` accepts unknown
  keys into `kwargs` and forwards them to the booster. Predictions still match the
  search-time model, because the unprefixed values are applied as well. It happens with
  or without bundles.

### 2026-09-14 — Post-merge fixes: GUI ensemble reconstruction, CatBoost refit, NameErrors (branch fix/post-merge-gui-ensemble)

- **Two `Params` spellings, three consumers.** Grid rows capture `pipe.named_steps['model']`
  (bare keys; search.py ~5252); Bayesian rows capture the whole Pipeline (`model__*`,
  `scaler__*`; unified_bayesian `_capture_serializable_params`); PLS-DA rows are always
  `pls__*` / `lr__*`. The GUI ensemble reconstruction (`_reconstruct_models_from_results`)
  *filtered out* `model__*`, so every ensemble base model built from a Bayesian row used
  defaults. `models.estimator_params_from_row` is now shared with `_rebuild_model_from_row`.
- **`get_model` clips `n_components` to `max_n_components` (default 10).** The ensemble path
  passed `n_components` only to `get_model`, so PLS rows with >10 LVs were silently clipped.
  It is now applied via `set_params` after construction.
- **Rows record the PLS-DA head seed and weighting** (`lr__random_state`, `lr__class_weight`
  come from the fitted pipeline's `get_params`). `split_plsda_params` deliberately drops them;
  use `models.plsda_head_kwargs` to build the head. Grid search always seeds 42
  (`RANDOM_STATE`); only Bayesian runs with a non-default `random_state` differ.
- **Ensemble classes refit clones, not the loaded model.** `clone()` of a fitted CatBoost
  works (via get_params) and `set_params` is legal on the unfitted clone, so the runtime kwarg
  is applied in `ensemble._clone_for_refit`, which walks `get_params(deep=True)` for nested
  CatBoost steps.
- **The GUI module had no `logger`** although four Tab 7 branches call `logger.warning`
  (flake8 F821). Use `logging.getLogger("spectral_predict.gui")`: `__name__` is `__main__`
  when run as a script, and `setup_app_logger` attaches its file handler to
  `spectral_predict` only.
- **Testing a Tk deferred callback:** monkeypatch `app.root.after` to record the function and
  call it after the method returns. Tk's real loop would swallow the NameError via
  `report_callback_exception`, so `root.update()` alone does not fail the test.
- **Not fixed (questions):** XGBoost classification `sample_weight` is fit-time and not in
  `Params`, so ensembles don't re-apply it (GUI ensembles are regression-only, so moot
  today). `models.py` has a pre-existing, unreachable F821 (`return model` at the end of
  `get_model`).

#### PR #77 review round (Codex block, GLM, DeepSeek)

- **The GUI ensemble rebuild ignored most preprocessing, not just Autoscale.** It matched
  literal `Preprocess` names only: `snv`/`snv_deriv`/`deriv_snv` built steps, `raw`/`snv`
  went to a GA wrapper, and **`deriv` (the grid/Bayesian/NSGA-II name for plain
  derivatives) and every `+`-affixed display name fell through with no preprocessing**.
  Bayesian and NSGA-II rows also store the *normalised* name (`deriv`, not `deriv1`), so
  the `NSGA_PREPROCESS_TYPES` numbered list never matched current rows. The validation
  rebuild's inline row parsing is now `preprocess.preprocessing_config_from_row`, shared by
  both paths.
- **Changing wrapper types changes ensemble CV mode.** `_is_wrapped_model` keys on the
  class name; any wrapped base model switches `create_ensemble(refit_base_models=False)`
  for the whole ensemble. `raw`/`snv` rows are now plain Pipelines, so ensembles of them
  refit per fold (the default). Subsets use `FunctionTransformer(np.take, indices, axis=1)`
  after the preprocessing steps: clonable and picklable, positional like the validation
  path.
- **The validation rebuild read `smoothing` as `bool(cell)` before its float check**, so a
  NaN cell (mixed grid + NSGA-II table) meant smoothing ON. The shared helper treats NaN
  as off.
- **GUI wrappers' `get_params(deep=True)` is shallow**, so walking `get_params(deep=True)`
  misses estimators inside them. `ensemble._iter_nested_estimators` walks shallow params,
  `__dict__`, and list/tuple/dict containers, with an id() visited set.
- **Test recipe for Bayesian preprocessing parity:** `unified_bayesian.apply_preprocessing`
  steps are per-spectrum (stateless) except autoscale, so preprocess train+test together
  with `apply_autoscale=False`, then fit a `StandardScaler` on the train rows.
- Legacy `sg1`/`sg2`, `deriv1`-style and GA names keep the old wrapper path.

#### PR #77 review round 3 (Codex block on ae15e64, GLM)

- **Exhaustive-search rows have an unbuildable `PreprocessBase`** (`snv_deriv1_w11`, the
  chromosome name). Preferring `PreprocessBase` made `build_preprocessing_pipeline` raise,
  and the GUI's per-row `except` silently dropped the model from the ensemble. The rebuild
  must check `preprocess_chromosome` first, as validation does. The search-time closure
  (`chromosome_to_transform`) is not clonable. `ga_preprocessing._spectrum_steps` now feeds
  both the closure and `chromosome_to_steps` (Pipeline steps), verified bit-identical over
  all 14 types x 17 windows x {2,3}-gene chromosomes, including identical ValueErrors for
  illegal window/polyorder pairs.
- **The GUI rebuild's per-row `except` turns any rebuild error into a silently shorter
  ensemble.** Regressions there show up only as "Failed to reconstruct" in the progress
  log, so tests must assert that one model came back.
- **Missing `Window` on a derivative row:** on main the validation rebuild passed
  `window=None` to `SavgolDerivative` and failed in `transform` (`None % 2`), while the old
  GUI defaulted deriv=1/window=15. The shared helper now applies 1/15 for derivative names.
- **GLM's round-1 claim was wrong: NSGA-II writes `str(dict)` Params** (`decode_solution`,
  nsga2_search.py ~2453), so NSGA-II rows never rebuilt with defaults. Dict cells are real
  only for in-memory result rows (scoring.py ~308). `parse_row_params` keeps dict support
  as defence; the docs no longer claim NSGA-II writes dicts. Verify reviewer claims about
  row shapes against the writer before building on them.
- `smoothing` string cells: `bool('False')` is True; parse strings like `Autoscale`.
- `chromosome_from_row` keeps validation's `ga_genes` fallback (pre-rename CSVs). The GUI
  Refine tab uses `ga_genes` for GA-PLS wavelength indices, so such a column in a results
  table would be decoded as a preprocessing chromosome by both paths (pre-existing risk).

#### PR #77 review round 4 (Codex block on c60ff80, DeepSeek, GLM)

- **dtype is part of preprocessing parity.** `chromosome_to_transform` and the validation
  rebuild both `np.asarray(X, dtype=np.float64)` before SNV/Savitzky-Golay. The SPC reader
  returns float32, and float32 counts around 1e6 give derivatives that differ enough to
  move RMSEP (0.941 -> 0.949 in Codex's repro). Pipeline steps rebuilt from a chromosome
  now start with `FunctionTransformer(_to_float64)`, a module-level function so it pickles.
  Parity tests need float32 count-scale input: float64 standard-normal data hides this.
  Non-chromosome rows (`build_preprocessing_pipeline`) have no such step in either the
  validation or ensemble path, so they stay consistent with each other.
- **"Missing" is not just `None` in a results row.** Validation's legacy
  `preprocess_chromosome` -> `ga_genes` fallback used `is None`, but in a concatenated
  table a legacy row's `preprocess_chromosome` cell is NaN. That is pre-existing on main;
  `tests/test_ga_preprocessing.py` already pinned the NaN fallback in a mirror of the
  logic, not in the real reader.
- Chromosome genes are bounds-checked (`_checked_genes`): an out-of-range `WINDOW_SIZES`
  index used to raise `IndexError`, which the GUI's `except ValueError` did not catch.

#### PR #77 round 5 (merge-with-nits)

- **Malformed input raises more than ValueError.** `np.asarray([10**30, 3]).astype(int64)`
  raises `OverflowError`, `len()` of a 0-d array raises `TypeError`, and `ast.literal_eval`
  of a deeply nested string can raise `RecursionError`/`MemoryError`. Callers that recover
  on `ValueError` need the parser to normalise all of them (`_MALFORMED_CHROMOSOME_ERRORS`).
  Also build the error message from a guarded, truncated `repr`, because repr itself can
  recurse.
- **Widening one copy of a parser creates divergence.** Round 4 widened the truthy strings
  in `preprocess.py` only, and four sibling copies (GUI `_parse_autoscale_flag`, the GUI
  model loader, `CodeGenerator._autoscale_enabled`, the one-class validation rebuild)
  kept `{true, 1, yes}`. They now all call `preprocess.parse_bool_cell`, and a source-scan
  test rejects a reintroduced `in ('true', '1'...)` tuple in those modules.

User asked for Codex, DeepSeek Flash and GLM 5.3 on everything merged this session
(earlier reviews were Kimi/GLM only; DeepSeek had hung). DeepSeek via opencode worked
once prompts forbade `gh`/`git fetch`; one run died on a self-typoed absolute path
(`sporheim`) — tell it to read files only via `git show <sha>:<relpath>`.
Codex found the most. Verified real bugs (all pre-existing or edge cases, none caused
by the PRs' default paths):
- **Deletion guard fails open:** `_sqlite_file_exists` returns False on OSError, and the
  migration code treats False as "file absent" → `_target_absent=True` → a failed
  migration can `delete_study` a pre-existing study.
- **`convert_study_to_dataframe` NameError:** `baseline_params` undefined (flake8 F821)
  — crashes result conversion for any Bayesian trial with `apply_baseline=True`.
- **Ensemble reconstruction drops tuned params:** GUI `_reconstruct_models_from_results`
  filters out `model__*` keys instead of stripping the prefix; Bayesian rows store
  estimator params as `model__*`, so ensembles train with defaults. Also drops PLS-DA
  `class_weight` and forces `random_state=42` (validation rebuild too).
- **GUI NameErrors:** `logger` undefined in `_run_refined_model_thread` (Tab 7 task-type
  mismatch branch); deferred `lambda: ...(str(e))` after `except ... as e` (Py3 unbinds e).
- `svm_gamma` resolves for SVM+regression / SVR+classification (silent 1e10 penalty);
  `'pls-da'` not normalised; categorical `(1, 1.0)` conflated by Optuna; `np.str_`
  constants break `ast.literal_eval` of Params; legacy CatBoost refit in ensemble.py.
- **CI:** push runs share one concurrency group, so a newer merge cancels an older
  *pending* main run; `continue-on-error` flake8 hid the F821 bugs above.
Rejected: branch-protection "required checks" concerns (main is unprotected); GLM's
"FixedTrial.params is pre-populated" block (it starts empty; verified).
Lesson: black/flake8 non-blocking was fine for style, but pyflakes F821-class checks
should always block — they found two crash bugs no test covered.

### 2026-09-14 — Post-merge Bayesian/search-space fixes (branch fix/post-merge-bayesian-search-spaces)

- **The T-41 failed-migration cleanup delete was removed; it cannot be made safe.**
  It deleted by name. Three rounds of guards each failed open:
  - a locked file read as "absent" because `_sqlite_file_exists` returns False on
    `OSError`;
  - a tri-state fix that still used `Path.is_file()`, which swallows every `OSError`;
    its test was falsely green because it overrode `is_file` to raise (DeepSeek);
  - `sqlite:///file:x.db?uri=true` parsed as a missing path (Codex).

  Even perfect checks leave a race: another process creates the study between the
  check and `copy_study`, then the copy fails with a lock error rather than
  `DuplicatedStudyError`, and the delete removes that study (Codex). Decision
  (orchestrator): never delete. Warn with the study name and storage, and leave any
  partial copy. It carries the data fingerprint, so an 'auto' re-run resumes it.
  `tests/test_t41_auto_rerun_preserves_study.py` asserts every failure mode keeps
  earlier studies, and that `unified_bayesian` source contains no `delete_study`.
- **`Path.is_file()` / `exists()` never raise `OSError`.** On Python 3.14 `is_file` is
  `os.path.isfile`, which returns False on any error. Call `path.stat()` directly when
  the error matters. Inject such faults by patching `Path.stat` for one path, never by
  overriding the predicate under test.
- **Frozen dataclasses with dict fields are unhashable.** Every `BundleSpec` already
  raised on `hash()` before #78, because `constants` defaults to a dict. Fixed with
  `field(hash=False)` on the mapping fields. They still count for `__eq__`, so the
  hash stays consistent with equality.
- **NumPy scalars break dataclass eq/hash consistency.** `np.float32(0.1) == 0.1` is
  True, because NumPy 2 casts the Python float to float32, but the float32 hashes its
  true value, 0.10000000149. `AxisSpec.__post_init__` therefore converts `low`/`high`/
  `step`/`choices` with `.item()`. As a result, `np.float32(0.1)` and `0.1` are now
  unequal, which matches their already-different space identities (Codex review of
  #78).
- **`_sqlite_path_from_url` is not a drop-in for `_sqlite_file_exists`.** It uses
  `urlparse`, which turns a relative `sqlite:///rel.db` into `/rel.db` on POSIX, while
  SQLAlchemy treats that URL as relative. It was left unshared.
- **`convert_study_to_dataframe` raised NameError on `baseline_params`.** It is a
  run-level value, hashed into the study name and never stored as a trial attr, so it
  has to be passed in. No test ran a Bayesian search with `baseline_method` set, so
  flake8 F821 was the only warning.
- **The PR B "harmless" `svm_gamma` cross-product entry above was wrong in effect.**
  Those pairs used to raise, and after PR B they ran with every trial penalised. They
  are now rejected via `BundleSpec.family_task_types`. That field is left out of the
  identity, so revision 1 and existing study names hold.
- **Optuna matches categorical choices by `==` (`list.index`), not by type.** `1`, `1.0`
  and `True` are one choice. The type-tagged uniqueness check suits identity hashing
  but cannot catch them.
- **`np.float64` and `np.str_` subclass `float`/`str`**, so `isinstance` literal checks
  accepted them while rejecting `np.int64`. Their NumPy 2 `repr` (`np.str_('x')`)
  breaks `ast.literal_eval` of `Params`. `resolve_bundles` now converts them with
  `np.generic.item()`.
- **A text-mode Python read/write converts CRLF files to LF**, which shows as a
  whole-file diff. `unified_bayesian.py` and `models.py` are CRLF in the index,
  `search_spaces.py` is LF. Restore CRLF before committing.

### 2026-09-14 — Crash-resume no longer flips persistence to 'always' (branch fix/resume-no-persistence-flip)

- **Premise verified with a real crash.** A child process runs an 'auto' PLS search via
  `start_run`, migrates after the 10-trial warmup, and dies with `os._exit` at 12/14
  trials (no WAL checkpoint). `find_incomplete_run` → `resume_run` → 'auto' re-run hits
  `auto_resumed_existing_study`, keeps the 12 trials unchanged and finishes at 14 in
  the same study. The migrated study carries `data_fingerprint` because it is stamped
  on the in-memory study at creation and `copy_study` copies user attrs.
- **A crashed 'never' run still left a sidecar and a prompt.** `start_run` writes the
  sidecar for every mode, so the next launch asked "Resume?", and Yes/No both ended in
  `resume_run` finding no SQLite ("Resume failed"). The same happened for an 'auto' run
  that crashed during warmup (the file only appears at migration). Fixed: the GUI skips
  the prompt when `run_state.has_resumable_store(meta)` is False. The sidecar is not
  deleted; the next Bayesian `start_run` overwrites it. A `stat` error other than
  FileNotFoundError still prompts.
- **T-43 restore includes `bayesian_persistence_mode`.** Resume therefore sets the radio
  to the crashed run's own mode, which the user accepts in the prompt. Nothing else
  captured the forced 'always': `start_run` returns the resumed metadata unchanged, so
  the sidecar kept the original mode, and the GUI has no preferences file for it.
- **Behaviour differences from forced 'always', accepted:** under 'auto', a study with
  a different or missing data fingerprint is not resumed. That run stays in memory with
  no crash-resume for that model, and the stored study is untouched. A legacy
  unfingerprinted study needs a name match, and the name includes `__version__` and
  the environment hash. Only a 0.5.0b3 dev build from 2026-09-13/14 could leave one.
  Multi-model runs gate each study by name: migrated models resume, unmigrated ones
  start fresh.
- **Review round on #79 (Codex block, DeepSeek/GLM nits):**
  - *Codex:* after a Python or NumPy update the study name's environment suffix
    changes, so an accepted resume under 'auto' silently started over. The
    `environment_changed`/`legacy_study_format` notices were gated on 'always'. They
    now also run for 'auto' when `_sqlite_file_exists` is already True, so the
    'auto'/'never' promise of not touching absent storage holds. A test forbids
    `get_all_study_names` when the file is absent. The data-mismatch sub-warning stays
    'always'-only: under 'auto' that case is already the declined gate branch.
  - Declines carry `unified_bayesian.RESUME_DECLINED_KEY`. The GUI's
    `_progress_callback_impl` → `_notify_resume_declined` acts only while
    `is_resuming()`: a log line, a status line, and one warning per run (keyed by
    storage URL).
  - `Path.resolve()` raises `ValueError` on an embedded NUL. `resume_run` caught only
    `OSError`.
  - (Superseded in round 2) Round 1 cleared 'never' sidecars that had no store.
- **Review round 2 on #79:**
  - *Codex:* the cleanup read the sidecar, compared run_id, then unlinked it.
    `threading.Lock` is per process, so another window could write its own 'auto'
    sidecar between the check and the unlink, and cleanup deleted that window's
    recovery record (reproduced). **Decision: no automatic sidecar deletion.** A
    stale sidecar costs only a silent check per launch; `start_run` overwrites it.
    Same lesson as the T-41 study delete: a name/id check followed by a delete is
    never atomic across processes.
  - *Codex:* the environment/legacy notice fired for studies with zero trials, e.g.
    a crash right after `create_study`. `resume_declined` now requires at least one
    COMPLETE trial (`_has_completed_trials`, read-only, only on storage already
    listed; unreadable counts as non-empty).
  - A `_study_exists` of None (lock) under 'auto' with a file now emits
    `resume_check_failed` instead of restarting silently.
  - **User decision (2026-09-15): a resume data mismatch never deletes.** The old
    `_run_analysis_thread` path called `discard_incomplete_run`, which deleted the
    sidecar and the SQLite store, and then started fresh silently. It now asks
    through `_ask_on_main_thread` (the iPLS queue pattern; the worker must not open
    Tk dialogs):
    - **keep** (default, also on timeout or error): `_end_analysis_without_search`,
      and the resume state stays intact;
    - **start fresh**: `run_state.abandon_resume()` (in memory only), after which
      `start_run` writes a new id, store and sidecar.

    Gotcha: the check sits inside the run-state `try/except Exception`, which only
    logs and continues. An exception in the handler would therefore *resume on the
    mismatched data*, so the call site catches it and stops. The GUI tests stop the
    worker at `start_run` with a `BaseException` sentinel, which the thread's
    `except Exception` handlers don't catch.
  - GUI `_RESUME_ISSUE_NOTICES` maps `resume_declined`, `resume_check_failed`,
    `data_mismatch_resume` and `data_unverified_resume` to per-kind wording, with one
    dialog per resumed run.
