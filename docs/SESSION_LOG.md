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

### 2026-09-15 — PR #79 crash-resume (merged a5f9a70) — condensed

Reference for the Bayesian crash-resume machinery (`run_state`, the GUI launch gate,
`active_run.json`). The full round-by-round history (13 Codex review rounds) is in
SESSION_LOG_ARCHIVE.md, batch 7. Binding user decisions and the known limitations
(e.g. multi-window safety would need file locking) live in PROJECT_STATUS §1 and
CHANGELOG 0.5.0b3.

**Design rule — one main-thread launch gate.** `_confirm_resume_before_launch` is the
single authority for a Bayesian launch. On the Tk main thread, before the worker exists,
it decides resume/delete/fresh, claims the run slot (`start_run`/`resume_run`), and
freezes that decision into `_run_analysis_thread`'s arguments: the selected models
(`_pending_bayesian_models` — a resume always freezes the run's own saved `model_names`),
the trial count (`analysis_n_trials`; a resume uses the saved `n_trials_per_model`), the
loaded data (`analysis_data`) and the filtered rows (`analysis_rows`), the optimization
method and task type (`analysis_modes`), `analysis_run_id`/`uses_bayesian_run_state`, and
a `capture_gui_settings` snapshot (`analysis_settings`, read in the worker via
`_setting(name)`). **The worker never reads live Tk state.** Nearly every Codex block in
rounds 9-12 was a spot where the worker still re-read something the gate had already
decided — the mode, `self.X`/`self.y`, the trial count, per-model settings inside the
model loop, the row filters, the validation split. Grid, one-class grid, NSGA-II and
post-search validation/config still read live Tk state; they don't touch run_state, so
that is out of scope rather than fixed.

**Whitelist rule.** A new GUI input that feeds the Bayesian study-name hash must be added
to `CAPTURABLE_SETTINGS` *and* `BAYESIAN_REQUIRED_SETTINGS`. Miss it and a changed control
silently starts a *different* study, while the finished run releases the saved run's
record. (Found via `bayes_enable_autoscale` and the imbalance params `k_neighbors`,
`n_bins`, `boost_factor`. A blank IntVar whose `.get()` raises is why the required-settings
check exists: `_setting` must never fall back to the live var when a snapshot was given.)

**Gotchas worth keeping:**
- `Path.is_file()` on Python 3.14 swallows every OSError (→ False); use `stat()` when
  "unknown" must differ from "absent".
- There is exactly one resume record (`active_run.json`), and a fresh `start_run`
  overwrites it unconditionally — anything that keeps an old run "for later" while
  launching a new one orphans it.
- `discard_incomplete_run` deletes the SQLite store BEFORE the record, keeps the record on
  a retryable store failure so Delete can be retried, and counts a missing store as
  deleted. `unlink` raising `ValueError` for an embedded NUL is not retryable.
- A damaged record must be reported (`CorruptRunRecordError`), never silently renamed or
  deleted; the old `.corrupt` rename lost it on Windows when a `.corrupt` already existed.
- `run_state`'s `_lock` is not reentrant: the set-aside call must happen before
  `with _lock:` in `start_run`, not inside it.
- A batch-edit script once converted the whole 60k-line GUI file from CRLF to LF (a
  122k-line diff); check `git diff --stat` after any scripted edit.

### 2026-09-15 — T-51 PR C: one-class bundles (PR #80, branch feat/T51-pr-c-one-class-bundles)

- **The mechanism needed nothing.** PRs A/B built resolution, gating, identity hashing and
  the base-sampler collision check generically; PR C is three `BundleSpec`s plus their entry
  in `BUNDLES`. `suggest_one_class_params` and `contamination.build_one_class_model` are
  untouched (`build_one_class_model` does `Estimator(**params)`, so any opened key lands on
  the estimator as named). Check that first before planning work for PR E/F.
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

### 2026-09-16 — T-51 PR C merged (#80, f401c29): what the five review rounds were about

The three bundles were right after GLM 5.3's first pass. Every round after that was about a
tiny-fold guard I invented from a review comment and eventually deleted. Read this before
adding "one small safety check" to a data-only PR.

- **Kept (the actual feature):** `if_max_samples`, `lof_metric`, `ocsvm_poly` in
  `search_spaces.BUNDLES`. `search_spaces.py` is the only source file the PR touched; the
  PR A/B mechanism is byte-identical to before. `build_one_class_model` does
  `Estimator(**params)`, so an opened key lands on the estimator as named.
- **Redundancy is often data-dependent.** GLM found `max_samples=1.0` duplicates `'auto'`
  (`min(256, n)`) below 257 inliers and I dropped 1.0, writing a false justification
  ("0.5/0.8 exceed auto above 256") into the code, the PR body and this log. Codex showed a
  fraction is `int(fraction * n)`: at n=300, auto=256 but 0.8=240, so 1.0 is the ONLY
  full-sample choice above 256. Restored, coincidence documented. Two reviewers disagreeing
  was the signal I should have caught; `minkowski(p=2)` aliases euclidean at every size,
  which is what a genuinely degenerate choice looks like.
- **The guard: three implementations, three false refusals.** Codex asked for tiny-fold runs
  to be refused up front. Each version refused runs that work on `main`: (1) it compared raw
  label values while the objective compares strings, so integer labels with
  `inlier_class_label='1'` counted zero inliers; (2) it ran before rows with NaN targets are
  dropped; (3) after moving it below the drop, supervised resampling still *expands*
  training folds after it (`cv_utils.py:1052`). Codex's fourth round recommended dropping it
  and I did. **Root cause: it duplicated logic that already lives on the execution path
  (label matching, row cleaning, resampling) and every copy drifted.** A false refusal is
  worse than the penalty-storm it prevents. A real feasibility check has to run against the
  fitting pipeline, not predict it, and belongs in its own PR.
- **The documented limitation is narrower than "wasted trials".** `0.5`/`0.8` need >= 2
  inliers per training fold (`auto`/`1.0` fit one). Too few successful folds → `+inf`, a
  skip reason, no leaderboard row. But repeated CV needs only half the folds
  (`contamination.py:732`), so with 3 inliers under repeated 2-fold CV the fractional trial
  IS scored, from the successful folds alone, unmarked and possibly omitting inliers — an
  incomplete metric, not merely a wasted trial. Pinned by
  `test_tiny_one_class_data_behaves_as_documented`, which fails if a partial-CV marker is
  ever added so the warning gets relaxed instead of going stale.
- **Traps hit while editing:** `test_curated_bundles_are_hashable` rebuilds a `BundleSpec`
  field by field, so a new field must be added to `_rebuilt_with_fresh_dicts`;
  `test_base_sampler_bodies_unchanged_from_main` pins sampler source by slicing between two
  `def`s, so a helper placed between the two samplers breaks the pin; and scripted patching
  left an LF block in a CRLF file, which showed up as a phantom diff against main (check
  `git diff <base> --stat` after any scripted edit).

### 2026-09-23 — CARS top-N padding: requesting more vars than CARS kept pads with the longest wavelengths (reported from Border Cave NIRS project; VERIFIED in default paths 2026-09-23; fix on branch)

**Observed** (Border Cave analysis, `analysis/bayes_varsel/`, driving `unified_bayesian` from a script): a
frozen `topN_cars` pipeline asked for N = 500 wavelengths against a CARS selection of about 70. The fitted subset was the CARS
core plus about 430 wavelengths from the *long end* of the spectrum. An independent re-implementation of dasp's CARS reproduced the
core exactly (109/109 in one case), so the padding comes from the top-N step, not from CARS.

**Mechanism (read from code, not yet test-pinned):** `cars_selection` returns a sparse importance array (zeros for
unselected variables). Top-N is taken as `np.argsort(importances, kind='stable')[-n_vars:]`. When
`n_vars > count_nonzero(importances)`, the extra picks are zero-importance ties, and the stable sort breaks the ties by index,
so the tail is the highest indices, i.e. the longest wavelengths. The results are silently mislabelled
`top{N}_cars` and the models may fit on arbitrary long-wavelength (often noisy, >2400 nm) variables.

**Same idiom at:** `unified_bayesian.py` ~1398/1426/1602/1627; `search.py` ~3930, ~4052, ~5540, ~6914. Not yet checked
which paths can actually request N above the CARS count (grid search may cap via the method-optimal count; the Bayesian
subset-size suggestion may not). Affects any sparse selector, e.g. SPA, UVE-thresholded or CARS hybrids, not only CARS.

**Likely fix direction:** cap `n_vars` at `count_nonzero(importances)` for sparse selectors (or skip the trial), and record the
true selected count in the result row. Add a test that asks for N above the CARS count and asserts no zero-importance
variables are returned.

**Verdict (2026-09-23): real in default paths, not only when forced.** No top-N site caps N at
`count_nonzero(importances)`. Grid (`search.py` ~3859/3930): CARS-family methods run *every* user variable
count (GUI defaults 10/20/50/100/250) and then add the method-optimal count as one extra run; only that extra run
is exact. Bayesian (`unified_bayesian.py` ~1552/1627, one-class ~1336/1426): `n_vars` is sampled from
`SUBSET_SIZES` = 10..1000 regardless of subset_type, and 'cars' is always in `available_methods`. One-class grid
(`search.py` ~6914) is the same shape. Evidence, BoneCollagen (49 x 2151), `cars_selection` random_state=42:
raw kept 159 (15 iterations, the Bayesian setting) / 193 (50 iterations); SNV kept 238 / 193. So the default grid's top-250
pads with 12-91 zero-importance long-wavelength variables, and Bayesian N=500/1000 pads with 262-841. N <= 100 never pads here.

**Fix (branch `fix/sparse-selector-topn-cap`, Codex-reviewed plan: AGREE-WITH-CHANGES).** `variable_selection.SPARSE_SELECTOR_METHODS`
(CARS family incl. the multiclass API's `cars_tree` spelling, uve_*/fipls_* hybrids, spa, vcpa-iriv, ga) + `_cap_top_n` caps N at the non-zero count, by method name only.
Codex warned against keying on "any zeros": `_apply_edge_mask`, tree importances, Lasso and the uniform fallback all
produce legitimate zeros. The tag keeps the requested count (`top250_cars`); `n_vars` records the fitted count (user's
choice). Gotchas: (1) grid must dedupe on the *capped* count, including against the method-optimal run (which already
equals the non-zero count and is tagged plain `cars`); (2) the one-class grid wrote the *requested* `n_vars` to the row,
now `len(top_indices)`; (3) the Bayesian fingerprint includes `subset_tag`, so capped trials would never dedupe. They now
fingerprint with `fit_tag` = `top{fitted}_{method}`; (4) `multiclass_varsel_mask`'s `_mask_from_scores` had the same padding
(found by the DeepSeek review) and is now capped. GLM/DeepSeek review round 1 also added `ga` and fixed the one-class progress
counter on skipped counts. Known, accepted: when CARS raises inside Bayesian `compute_importances`, the 'importance' fallback
is returned under the name 'cars' and IS capped at its non-zero count (drops zero-importance proxy vars; defensible, untested).
`ipls` (zero = interval R2 <= 0, a score, not non-selection) is deliberately not capped. Tests: `tests/test_sparse_selector_topn_cap.py`.

**Round 2 (2026-09-26, Codex BLOCK + GLM 5.3 MERGE-WITH-CHANGES).** (5) An all-zero sparse score array used to return the
*requested* N, i.e. the old padding. It is reachable: the grid's all-zero uniform fallback runs *before* `_apply_edge_mask`, so
a CARS selection lying wholly in an SG derivative edge zone reached the top-N step all-zero (Codex repro: `top10_cars` fitted
1102-1111). `_cap_top_n` now returns 0 for that case and every caller skips (grid breaks out of the counts, one-class skips
with the progress bump, multiclass raises `MulticlassVarselUnsupported`, Bayesian returns the usual penalty). Never slice
`[-0:]`: it selects every column. (6) `run_multiclass_simca_search` swept every NSelect for mask paths, so capped counts gave
identical fits ranked side by side; it now skips a resolved mask already fitted for the same prep/engine/path/alpha/ncomp.
(7) A replayed Bayesian duplicate returns before `selected_wavelengths`/`n_vars` user_attrs are set; tests reading them must
skip replays. Forcing params in a Bayesian test: patch `unified_bayesian.TPESampler` (the in-memory study builds it
directly at ~2837), not `_make_tpe_sampler` (only the SQLite reattach uses that). Not done, by user decision: no
study-name/version marker for pre-fix studies (the fix was never published, so old studies are dev-only).
(8) **The PR broke `test_t3_default_trajectory_matches_main` and round 1 missed it.** That test skips unless the machine's
numerical-env digest matches the fixture's, so on a non-blessing machine "the suite is green but for the 4 fingerprint
cases" says nothing about T3. On the blessing machine (this one: `.venv314`, digest `322dc72485c2`) it failed: trials 4/6/8/13/14
are CARS requests of 50-1000 that now fit 6-25 vars (same params, new values), and TPE diverges from trial 20. Trace re-blessed
in `tests/fixtures/t51_default_path_baseline.json` (3 identical captures; names/env/sampler hashes untouched). Any change to
default Bayesian trial values must re-bless it; run the suite on the blessing machine before claiming green.

**Round 3 (2026-09-26, Codex MERGE-WITH-CHANGES, GLM 5.3 MERGE).** (9) Multiclass `n_components` may be a per-class
dict, which is unhashable: the mask-dedup key crashed the whole search with `TypeError`. The key now uses `repr(_alpha)`
and `repr(_ncomp)`. Any set/dict key built from a multiclass grid axis needs this. (10) The one-class skips advanced
`current_config` but not `skipped_configs` ("0 skipped" in the completion message); both do now, and an empty sparse
selection skips the whole method once, before the counts loop. `AGENT_COMPOSITION.md` §3a's top-N snippet now caps sparse
arrays. Accepted and left as is: a failed multiclass fit still marks its mask as tested (a later NSelect with the same mask
would fail the same way).

### 2026-09-26 — T-51 PR D (GUI card for extra axes): implementation gotchas

Plan: `docs/plans/2026-09-26-T51-PR-D-gui-plan.md` (revision 3; three plan-review rounds, Codex + GLM 5.3).
- **Legacy snapshots are normalised in two places only:** inside `diff_gui_settings`, and at the startup full-restore
  call site. Never inside `restore_gui_settings`. The settings-diff dialog passes it a *partial patch*; filling
  defaults there reset other keys and looped forever (Codex reproduced this on plan revision 1). Test:
  `test_partial_restore_writes_only_the_given_keys`.
- **The PR D Tk vars are created in `__init__`, not with the collapsible card.** `capture_gui_settings` skips missing
  attributes, and a missing required key blocks every Bayesian launch.
- **The launch gate checks only presence,** so the startup StringVar needs its own value check (`parse_n_startup_trials`).
  The worker parses the frozen string again, so a direct worker call with a bad value fails that model (counted, run
  stays resumable) instead of reaching the backend.
- **Advisory:** `subset_type`/`n_vars`/`region_id` are suggested inline in the objective (both branches), so
  `extra_axes_advisory.SHARED_SUBSET_AXES` is a constant, drift-guarded against real studies' params. The one-class
  objective returns `inf` *before* suggesting them when `y_oc` is None. A test that forgets `inlier_class_label` sees
  no subset params at all.
- **Tooling:** in this Git Bash, Python edit scripts fed through a heredoc sometimes turned a `\n` in the replacement
  into a real newline, leaving a broken f-string in the GUI file. Use the Edit tool for any text containing
  backslashes, and `ast.parse` the GUI file after scripted edits.
- **Reviews and results (2026-09-27):**
  - GLM 5.3 said MERGE. Codex said MERGE-WITH-CHANGES in rounds 1 and 2; everything is fixed in `51acba2`
    and `cf59dbc`. PR #82 is open and not merged.
  - Full non-GUI suite: 3496 passed / 26 skipped. Full GUI suite on the final commit: 290 passed / 7 skipped /
    1 failed, and that one (`test_multiclass_gui.py::test_run_analysis_accepts_multiclass_engine_selection`)
    **also fails on `main`**.
  - The full GUI suite takes about 38 min here. A `timeout 900` wrapper killed an earlier run silently: `| tail`
    still exits 0. Run it in the background with no timeout.
- **Shared-app leakage:** ticking model boxes flips `model_tier` to 'custom', and a later test's
  `_on_tier_changed()` then returns early. Setting `task_type` rewrites `imbalance_method`. Restore `task_type`
  first and `imbalance_method` last. `tests/gui/test_t51_pr_d_gui.py::_restore_shared_gui_state` does this.

## 2026-09-28 - MC-PLS stability selector (leaf_phys_nir) vs dasp CARS: seed-stability test — do NOT add as a selector
Candidate from the leaf_phys_nir project (`loop/selection.py::_rank_by_cars`, which is misnamed: it is Monte-Carlo PLS
stability selection, not CARS). Test: 4 targets (dry/wet × SLA/LMA), 30 selector seeds each. Each target used its
rank-1 leaf recipe, rebuilt to 1e-13 of the ledger R²; only the selector and its seed varied, and scores are on the
recipe's external holdout. Scratch outputs only (not in the repo).
- **Band identity:** MC top-k mean pairwise Jaccard 0.29-0.39, against a null of k/(2p-k) ≈ 0.04-0.07. Only 2-11 bands
  are picked by all 30 seeds. dasp CARS is lower (0.16-0.34).
- **Regions (25 nm) favour dasp CARS:** MC r 0.49-0.65, dasp CARS r 0.71-0.94.
- **Holdout R²:** dasp CARS is never worse. Dry/SLA ties (0.772 vs 0.772); dasp is higher on wet/SLA (0.864 vs 0.799),
  wet/LMA (0.830 vs 0.798) and dry/LMA (0.803 vs 0.785). MC's seed SD is 0.015-0.032; dasp's is 0.009-0.016.
- **The earlier single-seed "MC 0.794 vs dasp 0.772" was a lucky seed** (93rd percentile of MC's seed distribution).
- **Tie-break:** MC's counts have ~10 distinct values, so ties fill 66-72 top-k slots. Ranking ties by mean |coef| over
  surviving iterations raises Jaccard 0.37→0.48 at no R² cost.
- **Decision:** don't add it as a selector. The worthwhile generic feature is **seed-frequency reporting for any
  score-array selector**: run it over N seeds and report per-band and per-region selection frequency. dasp CARS with a
  30-seed frequency report gave the most reproducible region picture.
- **Gotcha:** dasp CARS fits PLS with `scale=True` and ranks by |coef| in original units; MC used unscaled PLS. This is
  the likely cause of their dry/SLA region disagreement (500-750 nm: MC ~15% of bands, dasp ~1%). Untested.

## 2026-09-28 - Whole-codebase adversarial review: 133 findings kept (129 confirmed); full list in docs/reviews/
Full list with IDs, failure scenarios and verifier reasoning: `docs/reviews/2026-09-28-adversarial-review.md`. Method:
a Claude workflow with 11 areas, each finding re-checked by a skeptic told to refute it (1 refuted). Severity is the
verifier's: 2 critical, 30 high, 61 medium, 40 low. Duplicate pairs: R009/R026, R010/R064, R014/R019. Themes, by harm:
1. **What is saved, exported or predicted is not what was validated.** Y-transform save paths lose or double-apply
   preprocessing (R001 critical, R020); early stopping + Y-transform trains the final model on untransformed y
   (R014/R019); a stale bias correction is saved into a new model (R010); Tab 7 maps wavelengths ±0.5 to the first hit
   while predict uses ±0.01 (R009); the export bundle preprocesses twice (R015); the code export mis-maps
   preprocessing (R058); `all_vars` is written with %g, so subset models are validated on the full spectrum (R031).
2. **CV scores inflated by leakage.** Booster early stopping uses the CV test fold as eval_set: one root cause in
   `cv_utils.py:634`, repeated in the Bayesian (R003) and NSGA-II (R022) paths, and TPE optimises the biased score.
   Ensemble R2CV uses base models trained on the validation fold (R002 critical; repro: true CV R² 0.23 vs reported
   0.875); ensemble weights are fitted in-sample (R021).
3. **GUI analyses the wrong rows.** A new dataset keeps the old exclusions and validation split (R004); validation
   snapshots go stale (R005); two exclusion paths exclude the wrong sample or none (R006, R007); Data Management
   bypasses X_original (R037).
4. **Interference and contaminant removal methods are mathematically wrong.** The default EstimatedEPO removes noise
   directions, not the contaminant (R024); OSC removes the y-PREDICTIVE direction (R025); the Interference tab's
   exclusion, OSC and DOSC always crash (R075); JYPLS-inv omits centering (R091); compute_leverage gives every sample
   1.0 (R092).
5. **Readers.** The OPUS reader returns the background single-channel, not absorbance (R017); `read_ascii_spectra` is
   defined twice in io.py and the later one breaks folder import (R062).
6. **Classification metrics:** labels other than {0,1} give NaN/crash (R029); LOO averages per-fold F1 over 1-sample
   folds (R030).
Fix order proposed: themes 1-2 first (they change reported and deployed numbers), then 5 (R017), 3 and 4.

## 2026-10-02 - User rule: chemometrics validation conventions, not ML "leakage" rules
The user (Unscrambler is the reference standard) ruled that these are NOT leakage and must not be "fixed": row-wise
preprocessing (SNV, SG derivatives) applied before CV; bands/regions chosen a priori from chemistry (wavelengths must
not change mid-analysis); a holdout fixed before modelling. Real leakage = a test fold's y, or fitting on test samples,
producing that fold's score (booster early stopping on the test fold R028/R003/R022; ensemble base models trained on
the scored fold R002/R021). Rule now in CLAUDE.md. Also: the 2026-09-28 reviews were Claude-only; Codex gpt-6-astra
and GLM 5.3 cross-checks of all Wave 1+2 items were launched 2026-10-02 before any fix starts.

## 2026-10-02 - GUI dataset state (R004-R007, R037, R038): one install path, validation from the run's own data

Branch `fix/gui-dataset-state`. Gotchas worth knowing before touching data loading:
- **Every loader goes through `_install_dataset(X_original, y, ref, metadata_df, mode=...)`.** Import
  (`_load_and_plot_data`), Data Management Use/Merge & Use/filter/trim and calibration-transfer replace.
  `mode="replace"` clears `excluded_spectra`, the validation split and the Quality Check report;
  `"append"`/`"update"` keep exclusions and the holdout (pruned to labels still present). Assigning
  `self.X`/`self.y` directly anywhere else re-opens R037: `_update_wavelengths` rebuilds X from
  `X_original`, and `_on_target_column_changed` reindexes y to `X_original` from `combined_metadata_df`.
- **Replace keeps the Validation checkbox while a crash-resume is pending** (`_pending_validation_indices`
  or `run_state.is_resuming()`). Startup "Resume" restores `validation_enabled=True`; unticking it on the
  data reload would make the launch gate's settings diff fire a restore dialog the user never caused.
  The split itself is still cleared; the gate restores the run's own split.
- **`_load_and_plot_data`'s readers assign `self.X_original/y/ref` piecemeal.** It now snapshots the
  previous dataset first (`_capture_dataset_state`) and restores it on any stop before the install
  (sub-integer axis, X/y misalignment, append failure, exception). Before, a sub-integer rejection left
  new y next to old X (R038).
- **`_wavelength_filtered` no longer rounds `X_original.columns` in place**; it returns a new frame.
  Callers that relied on the reader's frame being mutated would now see unrounded columns.
- **The worker builds its own validation set** (`_validation_snapshot(X_run, y_run, frozen validation
  rows, frozen exclusions)`) and all five metric call sites (grid/Bayesian/NSGA-II/one-class/multiclass)
  use it, never `self.validation_X`. `self.validation_X` is still refreshed (launch, wavelength filter,
  baseline replace, target switch) for Tab 7/ensembles, but it is a convenience copy, not the source.
  Samples excluded after the split are not scored. `search.check_validation_axes` makes the backend
  raise on train/val width mismatch; equal-width axis identity can only be checked in the GUI (labels).
- **Click exclusion:** plotted lines carry `_dasp_sample_label`/`_dasp_sample_pos` (`_tag_sample_artist`).
  Never parse the gid back to an int: combined-file IDs are strings ('5', '007', '-3').
- **Quality Check:** `generate_outlier_report` gets an array, so `Sample_Index` is a POSITION. The GUI
  adds `Sample_Label`, keys tree rows by `outlier_row_<pos>` and maps iid -> label. Mark/unmark refuse a
  report whose sample index differs from the current X.
- `tests/gui/test_multiclass_gui.py::test_run_analysis_accepts_multiclass_engine_selection` fails on main
  too (its fake Thread doesn't accept the `kwargs=` the launch has passed since #79). Not caused here.
## 2026-10-02 - GUI dataset state, review round 1 (Codex BLOCK / GLM merge-with-fixes) follow-ups

- **A crash-resume must check exclusions, not just the data fingerprint.** The fingerprint covers X/y,
  and reloading the same file (replace) clears exclusions, so the gate accepted a resume on a different
  calibration set. The run record now stores `calibration_rows` (`{"excluded": [...], "active": [...]|None}`,
  via `run_state.calibration_rows_record`); `_reconcile_resume_calibration_rows` runs before the split
  reconcile and asks (restore / start fresh / cancel). Records without the field resume with a log line.
- **Data Management and calibration-transfer data keep their exact wavelength axis** (`exact_axis=True`),
  preserving their behaviour before the install helper: Import's integer rounding would reject 0.48 cm^-1
  FTIR axes and change resume fingerprints. `_exact_wavelength_axis` remembers the rule for later wavelength
  Updates, appends and edits. Whether Import should keep rounding is still open.
- **Rejected loads restore units/data type too** (`_DATASET_STATE_ATTRS/VARS`), and browse-time combined-file
  detection writes `_combined_metadata_df_preview`, never `combined_metadata_df` (it ran before the snapshot).
- **Target switch and Analysis Subset columns merge metadata stores by sample** (`_metadata_column`):
  after Data Management data gets an appended combined file, `ref` and `combined_metadata_df` each hold
  their own samples' values. Use `.infer_objects()`: a NaN-padded store makes numeric targets object dtype.
- **Between-run validation consumers:** manual ensemble retrain uses the run's frozen holdout
  (`training_data_cache["validation"]`, `_last_run_validation`); Tab 7 refit builds its own snapshot from the
  same data as its calibration rows. Prediction-tab data loaded from the validation set keeps its own targets
  (`prediction_actuals`). `_check_validation_axis` is called where it can fire (it was tautological in the
  worker).
- **Duplicate IDs:** `io.rename_duplicate_ids` now never produces a duplicate (`["A","A.1","A"]` gave two
  "A.1"); `_install_dataset` also gives repeated labels a unique suffix. QC staleness uses a data
  fingerprint (index, columns, values), so a baseline replace or wavelength change also refuses a stale report.
## 2026-10-02 - GUI dataset state, review round 2 follow-ups

- **Resume gate: check every saved label is present BEFORE comparing.** Intersecting saved exclusions/subset
  with the loaded labels made a renamed sample vanish from both sides and the gate returned "ok". Now any
  missing saved label refuses (record kept).
- **`calibration_rows` stores `run_state.canonical_label` keys** (type-tagged strings: `i:5`, `f:5.0`,
  `s:5`, `t:[...]`), so float/tuple IDs are recorded and compare equal after the JSON round trip. A new
  record without rows asks (resume anyway / fresh / cancel) instead of resuming silently.
- **GUI records carry `label_normalization` (`run_state.LABEL_NORMALIZATION` = 1; `start_run` defaults to None, so headless and test records count as legacy).** Legacy records (None) were
  saved before repeated IDs got collision-free suffixes (old reader: `["A","A.1","A"]` -> "A.1" twice), so
  the same saved label can name other rows. `_labels_differ_from_legacy` is set at install when the reader's
  `duplicate_rename_mapping` differs from the old scheme or the install had to suffix repeats; the gate
  then refuses a legacy resume (keep/fresh). An exact migration would need the old row order, not stored.
- QC staleness fingerprint includes the targets; the Analysis Subset dialog and `_metadata_stores` use only
  the installed dataset's stores (the uninstalled DM merge never had `metadata_df` anyway); Comparison's
  validation load refreshes the snapshot; repeated NaN IDs become "nan", "nan.1".
## 2026-10-02 - GUI dataset state, review round 3: one identity check for crash-resume

- **`calibration_identity` is the final authority on resume.** At launch the GUI records a blake2b digest of
  the exact calibration rows the worker trains on (after subset, exclusions and holdout removal, in order:
  canonical labels, wavelength labels, float64 spectra, targets) and of the holdout rows
  (`run_state.calibration_identity`, built by `_calibration_identity_now`, which mirrors the worker's
  filtering). The gate recomputes it after any user-approved restore and resumes only on a match; otherwise
  one dialog: start fresh / keep (default keep). This replaces chasing label edge cases (swaps, order,
  normalization drift, provenance). The coarse `dataset_fingerprint` (shape + 3 cells) is still checked first.
- `calibration_rows` (versioned keys of excluded / active / holdout) only drives the restore offers;
  the holdout keys also fix float/tuple holdout labels that `validation_indices` drops.
- `canonical_label` supports int/float/str/bool (numpy too) and tuples of these, nothing else
  (`UnsupportedLabelError`): no repr fallback, which could collide or change across sessions.
- Record classes: legacy (all three new fields None) resumes with a log, unless the loaded labels look renamed
  (`_labels_look_renamed`: `"<id>.<k>"` next to `"<id>"`), computed from the loaded labels at the gate so it
  can't go stale after Data Management, merges, viewer edits or Revert. Anything else not exactly current
  (unknown `label_normalization`, malformed rows/identity, unsupported labels) asks: resume anyway / fresh /
  keep. `from_dict` no longer coerces malformed `calibration_rows` to None (that made it look legacy).
- `rename_duplicate_ids` tests missingness per label (`pd.isna` on a MultiIndex raises).
## 2026-10-02 - GUI dataset state, review round 4: transactional resume checks, worker-parity identity

- **Resume reconciliation is transactional.** `_reconcile_resume_selection` runs rows -> holdout -> identity;
  the gate snapshots the selection first (`_capture_calibration_selection`) and puts it back on any outcome
  but "ok" (`_restore_calibration_selection`), so an approved exclusion/holdout restore followed by "keep"
  at a mismatch leaves the GUI as it was at the click.
- **`_prepare_calibration` is the one definition of the rows a run trains on** (subset -> exclusions ->
  holdout -> mixed-type label normalisation -> rare-class drop). The worker and `_calibration_identity_now`
  both call it, so the digest never includes rows the worker later drops. Tests capture the X/y the worker
  actually passes to `run_unified_bayesian` and rebuild the identity independently, for both start_run sites.
- **Identity v2** (`CALIBRATION_IDENTITY_VERSION`): every label/section length-prefixed (NUL-joined labels
  collided), row counts hashed, integer targets as int64 (float64 merged ints above 2**53), object targets
  that are all Python/numpy numbers hashed like the numeric column they equal. Unknown versions ask.
- `valid_calibration_rows/identity` require every schema key and element type; "resume anyway" on an
  unverifiable record keeps the current selection and never decodes its holdout keys.
- Legacy record + labels that look renamed ("S1" and "S1.2") now asks resume anyway / fresh / keep: that
  spelling also occurs naturally. `rename_duplicate_ids` uses missing-aware keys (tuple IDs with NaN parts).