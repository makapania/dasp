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

## 2026-10-02 - Wavelength mapping contract (R009/R026/R112, R031/R078) and label encoder (R016), branch fix/wavelength-mapping
- **One contract:** `spectral_predict/wavelength_matching.py` (`match_wavelengths`, `resolve_wavelength_list`,
  `format_wavelength_list`, `WavelengthMatchError`; on the declared surface). Exact axis value first, else the single
  value within 0.01; missing, two-candidate or two-values-one-column all raise. Tab 7 (both mapping sites), model_io
  (every former `< 0.01` first-hit loop and `_select_wavelengths_from_dataframe`) and both validation rebuilds use it.
- **The GUI rounds axes to integers on load** (`_apply_wavelength_filter`, ~19670, refuses sub-integer data with a
  dialog). So the fine-grid R009 bug is reachable in the GUI only when that filter is skipped (X_original kept, e.g.
  after its error dialog) and from Python; R031/R078 are reachable in the GUI after nm<->cm-1 conversion (1e7/x
  columns are not rounded). Do not "simplify" the contract on the assumption that GUI axes are integers.
- **Legacy %g handling (revised after review round 1, Codex BLOCK):** a numeric window around the rounded value is
  wrong below powers of ten (`"10000"` would match 9999.97, which %g prints as `"9999.97"`) and for negatives. Match a
  legacy token to the axis columns whose OWN `f"{a:g}"` equals the token, and require exactly one. Telling new rows
  from old ones per row also failed: repr writes `10000.1`, which %g also writes. The writer now appends a `0` to any
  mantissa that lacks a trailing zero after the decimal point (`10000.10`, `1.0e-05`, `7407.4074074074080`); %g
  strips trailing zeros, so it never writes such a token, and classification is per token. Rows written by
  `1f3c671` (plain repr) never shipped.
- **Old models with %g-rounded metadata wavelengths:** the pre-fix refit stored the parsed `all_vars` floats. On a
  0.012 grid ~45% of them have both neighbours within ±0.01, so unstamped models are mapped with
  `match_wavelengths(..., legacy_g=True)` (the same %g-text rule) at predict time and in the load-time replay; if even
  that is ambiguous, load warns and predict raises "Retrain".
- **Ensembles are excluded from the retrain replay:** the GUI ensemble save stores `wavelengths == full_wavelengths`
  (the whole exact axis) for every member, so replaying the Tab 7 first-hit rule would flag every fine-grid ensemble
  falsely. Members carry `ensemble_parent`/`is_base_model`.
- **Old saved models:** `save_model` now stamps `metadata['wavelength_matching'] = 1`. On load, an unstamped Path A model
  (`use_full_spectrum_preprocessing` + `full_wavelengths`) replays the old first-hit-within-0.5 mapping against
  `full_wavelengths` (= the Tab 7 training axis) and warns if any feature differs (Tab 8 load shows a dialog).
  Prediction already read the named column before the fix, so old models predict exactly as before; only the warning
  is new. Not auto-remapped to the neighbour on purpose (brief: warn + retrain).
- **GUI `wavelength_indices` must be a list:** the one-class validation block tests `if wavelength_indices:`; a numpy
  array there raises "truth value of an array is ambiguous". The refit sites call `.tolist()`.
- **R016:** refinement trains on raw numeric labels (it encodes only text labels), so `self.label_encoder` (fitted on
  every Bayesian/NSGA classification search) never belongs to a refined model; the save fallback is gone.
  `save_model` also drops an encoder whose model `classes_` are not codes 0..n-1 (warning), and `predict_with_model`
  skips decoding for such legacy artifacts. Display-only fallbacks (`refined_label_encoder or self.label_encoder` at
  ~21176/21198/37715/38596) still mis-label numeric classes in tooltips/plots: not fixed here.
- **Validation failures are surfaced via `df.attrs["validation_failures"]`** (`{row index: reason}`) and
  `df.attrs["validation_attempted"]` from both `compute_validation_metrics_for_top_models` and the one-class twin;
  the GUI logs "computed for only X of top N" plus reasons, counting only this run's non-failed attempted rows.
  The one-class helper now clears val_* on attempted rows first (it used to keep an earlier run's numbers). A
  supervised row with no usable `all_vars` is validated on the full spectrum only if it is tagged full AND its
  `n_vars` equals the column count.
- **Review round 2 (Codex BLOCK on 820c039):**
  - The R016 class-count rule was wrong: an encoder fit on a,b,c with a model trained on codes for a,b only is a
    valid pair. Only provable staleness (model classes not codes 0..n-1 of the encoder) is rejected; ownership is
    enforced where the encoder is chosen (Tab 7 saves only `refined_label_encoder`).
  - Legacy `%g` interpretation at predict is limited to `_is_legacy_tab7_model` metadata (unstamped Tab 7 Path A).
    Ensembles/multiclass keep exact-first; `save_ensemble` stamps its top-level metadata too.
  - Legacy tokens are classified by their TEXT (`_looks_like_g_token`: <=6 significant digits, no trailing zero
    after the point, fixed notation only for 1e-4 <= |v| < 1e6), never by re-formatting the parsed value, which
    fails for subnormals (`float('1e-318')` prints `9.99999e-319`).
  - Success is recorded, not inferred: `attrs["validation_succeeded"]` from both helpers (try/else). One-class
    `val_BalancedAcc` is NaN by design on an inlier-only validation set.
  - The missing-`all_vars` fallback needs an affirmative `"full"` tag (`SubsetTag`, or `Subset` when SubsetTag is
    null) and matching `n_vars`; the rule is `wavelength_matching._full_spectrum_fallback_refusal`, shared by
    search validation, GUI ensemble rebuild and `ensemble.extract_preprocessor_config`.
  - Ensemble paths converted: GUI `_reconstruct_models_from_results` (`parse_wavelength_subset` now takes the row;
    failures raise and the row is excluded with "[!] Failed to reconstruct"), `preprocessing_wrapper.
    PreprocessorConfig` and `ensemble.extract_preprocessor_config`. The wrapper-level
    `_match_wavelengths_normalized` (WavelengthSubsetWrapper predict path) was converted at the merge of main
    (PR #84) in its new home `src/spectral_predict/model_wrappers.py`; no copy remains in the GUI or model_io. Test fixtures that rebuilt full rows without `all_vars` now carry
    `SubsetTag="full"` + `n_vars`, as real search rows do.
  - `compute_composite_score` re-keys validation attrs after `reset_index` (`scoring._remap_validation_attrs`).
- **Review round 3 (Codex BLOCK on dae6654):**
  - Encoder ownership is now RECORDED, not inferred: `save_model` writes `metadata['label_encoder_owned']=True`
    whenever it keeps an encoder (dropping only provably stale ones: bool or non-code classes). Prediction decodes
    only classification models, and only (a) stamped encoders whose codes are valid, or (b) for unstamped legacy
    files, encoders that provably fit (model classes == codes 0..n-1, same count). Everything else returns raw
    predictions with a warning. No class-count rule on new files (subset-class models stay valid); the count rule
    survives only as the legacy proof. Raw {0,1} next to a stale x,y,z is indistinguishable from a subset-trained
    model without the stamp, hence no decoding.
  - Model Development loading (`_load_model_for_refinement`) applies `_full_spectrum_fallback_refusal`; with no
    `all_vars`, `top_vars` is used only if it maps and its count equals `n_vars` (it keeps at most the top 30).
    Otherwise "# ERROR" in the wavelength box.
  - `_looks_like_g_token` now checks the exponent grammar exactly as %g writes it: two-digit padded exponent, more
    digits only when needed, and exponent form only for exp < -4 or >= 6. `1e+00`, `1e+03`, `1e+006`, `1.5e+05`
    are matched exactly.
  - Multi-class holdout val_* need no pre-clear: the supervised helper re-initialises every val column to NaN for
    ALL rows on entry (whole-column assignment). Pinned by a test instead of new code.
- **Review round 4 (Codex MERGE-WITH-FIXES):** files with absent/null `task_type` are classifiers when the fitted
  estimator is (`model_io._effective_task_type`: sklearn `is_classifier` or `classes_`), so an owned encoder still
  decodes them; explicit regression/one_class/multiclass_simca dispatch is unchanged. The decoding decision is one
  helper (`_decoding_encoder`) shared by prediction and `predict_with_uncertainty`, whose probability column names
  now follow `model.classes_` (decoded only through a qualifying encoder, never more names than columns). That
  fixes the Tab 8 `_display_uncertainty` IndexError with a superset encoder.
- **Review round 5:** `model_wrappers._match_wavelengths_normalized` partitions PER ITEM (numeric requests via
  `match_wavelengths`, non-numeric ones such as an ID column must be present literally, order restored); an
  all-or-nothing numeric parse broke legacy wrappers with `[1000.0, "id"]`. Tab 8 `_display_uncertainty` uses the
  union of all loaded models' class labels as headers and places each model's probabilities under its own labels
  (blank where absent); headers used to come from the first model only.
- **Follow-ups not done (review round 2, deliberately out of scope):** `scoring.py` ~112 substitutes the CV gap
  when validation is missing; `save_model` stamps `wavelength_matching` on any save, so an old fitted model
  merely re-saved without retraining would lose its retrain warning.

## 2026-10-02 - fix/contaminant-maths (QW3, QW6, R024, R025, R075, R113, R114): gotchas
- **EPO must not centre the nuisance library.** `EstimatedEPO`, `MultiGroupEPO` and `interference.EPO` all
  column-centred D before the SVD. When the rows share one contaminant shift (mean diff + noise copies,
  bootstrapped mean diffs, equal-dose groups, one shape at several levels) centring subtracts the contaminant and the
  SVD returns jitter. Now: SVD of the uncentred D, `P = I - VV^T`, `transform = X @ P` (a spectrum). `bootstrap` was
  removed outright: uncentred, its 2nd+ directions are sampling jitter = analyte variation.
- **Uncentred multi-group D needs a rank rule.** With k = n_groups, two groups sharing one contaminant give a 2nd
  direction that is the difference of their mean ANALYTE levels; projecting it out kept 0.35% of the analyte.
  `MultiGroupEPO` now keeps a direction only if S_k^2 > 9 x the expected sampling energy of the mean differences
  (`sum var_g/n_g + var_ref/n_ref`). A factor of 4 (2 SE) flagged a same-population group ~1 in 20 when its spread
  lies along one direction (the analyte), so 9 (3 SE).
- **`interference.EPO` padded its basis with null-space vectors** when the library rank < n_components (it computed
  `S_truncated` and never used it). Harmless-looking with the old centred library; with an uncentred rank-1 library it
  projected out arbitrary directions. Now capped at the library rank with a warning; tests that asked for 2-3
  components from the rank-1 fixture library now use a random library.
- **numpy 2.5 `np.linalg.pinv` default cutoff kept the ~1e-14 singular value left by mean-centring** (trace of
  `X X^+` was 79.05 for rank 79) and broke `T = X W` in DOSC at the 1e-4 level. Pass
  `rcond=max(shape)*eps` explicitly.
- **OSC/DOSC replaced** (Fearn 2000 / Westerhuis et al. 2001): removed scores satisfy t'y = 0 to ~1e-15; weights and
  loadings are stored and replayed; output is `X - T P'` on the original scale. Old OSC removed the first PLS loading
  (corr(t,y) 0.98 on the bench); old DOSC's replayed scores correlated 0.2 with y.
- **Corrected EPO spectra are projections, not "decontaminated" spectra.** A clean spectrum also loses its own
  projection on the contaminant direction (baseline overlap), so its level drops a few percent. Tests assert
  `X @ P` and "not centred", not "clean spectra unchanged".
- **GUI:** `self.X_train` / `self.wavelengths` are never assigned anywhere, so the Interference Application page's
  "Load from Import Tab" always said "no data". It now reads `self.X` / `self.y` (aligned by sample label). The same
  dead attributes are still read by the Diagnostics sub-tab (GUI ~57351, ~61000, ~61050); not fixed here.
- Repo line endings are mixed: the GUI and the test files are stored CRLF, `src/` modules LF. Python rewrites with
  default newline handling turned the whole GUI diff into 124k lines; write CRLF files back with `newline=''`.

## 2026-10-02 - fix/contaminant-maths round 2 (Codex BLOCK / GLM merge-with-fixes on 0205c32)
- **Pickles.** Old fitted EstimatedEPO/MultiGroupEPO (have `X_mean_`, no `fit_version_`) silently changed output;
  old OSC/DOSC raised NotFittedError. Now every fit stamps `fit_version_ = 2`; objects without it replay the old
  transform exactly (with a warning) so saved downstream models keep their predictions. Verified cross-process:
  fitted with f6a2287 code, unpickled with the branch, max |old - new| <= 9e-16 for all six classes and an
  OSC+PLS pipeline.
- **interference.EPO needs two library kinds.** Uncentred SVD is right only for difference/pure-interferent
  libraries. A library of whole spectra (one sample at several moisture levels) contains the analyte; uncentred,
  its first direction IS the analyte (kept 0.33% analyte, 99.7% moisture). `library_type='samples'` (default,
  = differences from the library mean, the old behaviour) vs `'differences'` (uncentred). GUI Application page
  asks which; preprocess.py passes `library_type` through.
- **MultiContaminantAnalyzer joint projection** took the numerical rank of the union of per-group directions, so
  two groups sharing one contaminant removed their analyte sampling difference too (analyte kept 0.02%). It now
  delegates to MultiGroupEPO (`joint_epo_`).
- **MultiGroupEPO automatic count replaced** (the 9x summed-energy rule was not a per-direction test and diluted
  a contaminant in one of k groups). Now: rows weighted by 1/sqrt(1/n_g + 1/n_ref); sequential test of the largest
  remaining squared singular value against a bootstrap of the POOLED within-group residuals (randomly signed,
  rescaled sqrt(n/(n-1))), with a small-sample factor F_{1-a}(1,df)/chi2_{1-a}(1), df = N - groups - 1; alpha 0.01,
  999 draws, seed 0. Two failed variants, for the record: label permutation across all spectra lost power when
  a real contaminant was present (it inflates the null; Codex blank-group case 0.01); a sign-flip of each
  group's OWN residuals was anti-conservative at n=5 (false positives 5-7%). Needs >= 2 spectra per group for
  auto, else `n_total_components`. Rates (200 runs, analyte swing ~1, constant dose, old 9x -> new):
  all-groups dose 0.2: n=5 0.05->0.04, n=10 0.01->0.01, n=40 0.91->1.00; dose 0.5: n=5 0.48->0.28, n=10 1->1;
  one of k groups: k=2 n=5 d=0.5 0.07->0.12, d=1 0.98->1.00; k=2 n=10 d=0.5 0.30->0.91; k=4 n=5 d=1 0.37->0.96;
  k=4 n=10 d=0.5 0.00->0.11; false positives 0-2% old, 0-1.5% new; Codex blank-group case 0.21->0.94. A
  contaminant smaller than the group means' sampling spread along their noisiest direction (the analyte, at
  small n) is not detectable by any data-driven direction test; the Apply page has an override.
- **GUI:** Restore/Apply now rebuild `validation_X` from `self.X` by the cached validation IDs (minimal, so
  fix/gui-dataset-state's `_install_dataset` can absorb it); Restore also checks a content fingerprint (an
  in-place edit kept object identity); EPO "directions to remove" (auto/1-5) and an unpaired-groups caution;
  "auto found nothing" is an information dialog with the override hint. Pairing is NOT offered: the loaders
  discard contaminated-group sample names and the combined import has no specimen-ID column.

## 2026-10-02 - fix/contaminant-maths round 3 (Codex BLOCK / GLM merge-with-fixes on a5b17f2)
- **The pooled residual bootstrap assumed one within-group covariance.** With identical means and no
  contaminant it removed a direction in 25% of runs (reference n=50 SD1 vs group n=5 SD4) and 34%
  (reference n=5 SD4 vs two groups n=50 SD1), 400 runs each. Replaced by a per-group bootstrap: each
  group's mean error is drawn from its OWN residuals (rescaled sqrt(n/(n-1)), random signs), one shared
  reference draw per replicate, and each group's error multiplied by sqrt(df/chi2_df), df = n_g - 1 (a
  Behrens-Fisher-type predictive; the global F/chi2 factor and a min-Welch-df factor were dropped - the
  latter made any 2-spectrum group veto everything). False positives at alpha 0.01, 400 runs: Codex
  heteroscedastic cases 0.5% / 1.5% / 0.25% (10/10/10, ref SD4) / 0.25% (equal SD 50 vs 5); GLM grid k=2-4,
  n=2-40: 0-1.25% (0% at n<=5: conservative). Pooled a5b17f2 on the same cells: 25% / 34% / 4.5% / 1.75%,
  grid 0.5-2.75%.
- **Cost: power at small n.** Detection (400 runs; among hits, median contaminant energy removed):
  all groups, dose 0.5: n=5 0.03 (75%), n=10 0.97 (97%); dose 0.2: n=40 0.99 (95%), n<=10 <=0.02. One of
  k groups: k=2 n=10 d=0.5 0.48 (94%); k=2 n=5 d=1 0.43 (97%); k=4 n=5 d=1 0.04; k=4 n=10 d=1 1.00 (99%).
  The pooled version found far more at n=5 (k=4 n=5 d=1: 0.96) but at the false-positive cost above. With
  groups of 2-3 spectra the test essentially never removes anything; a 2-spectrum blank group makes the
  max statistic too heavy-tailed to find a clear contaminant elsewhere (Codex blank case 0.94 -> 0.01).
  Accepted because the GUI count is now advisory and has a manual choice.
- Benchmark "hits" now report removal quality too: low-power cells' hits remove 13-40% of the contaminant
  (Codex saw 22% for n=5 dose 0.2), so a bare detection rate overstated usefulness.
- Other fixes: `__setstate__` migrates constructor params of old MultiGroupEPO (alpha/n_resamples/
  random_state) and interference.EPO (library_type = 'samples' iff old center=True, matching what the old
  code did) so clone/refit work without stamping fit_version_; Apply/Restore clear an unrebuildable holdout
  (`_reset_validation_set`); MultiContaminantAnalyzer warns and falls back to all group directions for
  singleton groups and no longer hides "removes nothing"; analyze_multiple_contaminants keeps partial
  results with a GUI-worded note; zero-capacity (one wavelength) no longer IndexErrors; failed fits leave no
  half-fitted state; bootstrap chunks sized to ~64 MB.
- 0205c32 (round 1) never left this branch, so pickles fitted by it (corrected method, no fit_version_)
  would wrongly take the legacy path; no action: only f6a2287-and-earlier pickles exist in the wild.

## 2026-10-02 - fix/contaminant-maths round 4 (final review)
- Known limit, documented rather than fixed: the per-group bootstrap randomly sign-flips residuals, which
  symmetrises them, so SKEWED groups with very different spreads are anti-conservative (Codex: lognormal
  null, n 50/10, SD 1/4 -> 9.4% false removals at alpha 0.01, each erasing the analyte; exponential 3.8%;
  `test_auto_rank_skewed_heteroscedastic_null_rate` reproduces 18/200 and bounds it at <= 30/200).
  Dialog, docstrings, UserGuide and the Apply caution now say the p-values are approximate, can be wrong
  both ways (one contaminated group of four at n=10, dose 0.5: ~6.5% detection), and that groups of 2-3
  need a manual count. The advisory dialog adds a warning when a group's residual scores along the
  suggested direction have |skewness| > 1 (n >= 8) - a hint, not a test.
- tests/gui/conftest.py `_suppress_dialogs` now also patches tkinter.simpledialog.askinteger/askstring/
  askfloat (return None); no existing test used the real dialogs.
## 2026-10-02 - Ensemble CV fix (branch fix/ensemble-cv; R002, R018, R021, R105)
- **R018 masked R002.** With string specimen IDs every GUI ensemble died on `y_filtered[train_idx]` (pandas 3 label
  lookup), so the inflated R2CV was only visible on RangeIndex data. Fixing the indexing alone would have published
  the leaky numbers. Example data (49 bone spectra, 5 base models incl. GA + CARS wrappers): before, string IDs ->
  all 5 methods `[X] failed`; RangeIndex -> R2CV 0.936-0.970 with best honest base 0.869. After: 0.847-0.881 for
  both index types.
- **The "~0.03 R2 loss from StandardScaler divergence" (commit 9c4d02c) was honest CV, not a bug.** Refitting a
  wrapper on 4/5 of the rows gives lower OOF R2 than predicting with the full-data fit; that drop is the point. The
  wrappers already cloned correctly (clone().fit(all rows) reproduces the original predictions). `_is_wrapped_model`
  and the GUI's `refit_base_models=not any_wrapped` are gone; `refit_base_models=False` still exists in the ensemble
  classes for API compatibility but warns that it learns in-sample.
- **WavelengthSubsetWrapper passed full-width numpy arrays straight through** ("assume columns are already matched"),
  so any numpy caller (validation arrays, numpy-X ensemble fits) fed 2151 columns to a 40-column model. It now takes
  `all_columns` and subsets arrays by position, or raises when the width matches neither.
- **Outer CV design:** `ensemble.cross_validate_ensembles` clones and refits every base model on each outer training
  fold (shared across ensemble types), then `create_ensemble` learns weights from inner OOF within that fold. The
  deployed ensemble is still fitted on all calibration rows. Cost is about that of the old loop with unwrapped
  members (plus one base fit per member per outer fold).
- **Failed OOF members are now removed from the ensemble** (`models`, `model_names`, preprocessor lists; recorded in
  `excluded_models_`), so saved metadata, weights and routing all agree. This mutates `models` in `fit` (sklearn
  convention says don't), chosen because `model_io.save_ensemble`, the GUI metadata and viz all read `.models` /
  `.model_names`; the caller's list is never mutated in place. All members failing raises ValueError.
- **Stacking R2cal can be far below R2cv** on weak data (meta-model trained on OOF features, applied to in-sample
  features): -0.75 vs -0.35 on noise. Not a bug in this fix; don't assert cal >= cv in tests.
- `create_auto_ensembles` (no production caller) had the same Series indexing and two full-data fallbacks; folds that
  cannot rebuild 2 models now give NaN metrics instead of calibration predictions, and `unique_model_count` counts only
  successful rebuilds (R106 in passing). The first commit missed its two `n < 2` calibration fallbacks; round 1 of
  review removed those too (NaN + warning).
- **Review round 1 (Codex BLOCK, GLM merge-with-fixes) added:** a caller's stacking `meta_model` is cloned per outer
  fold and inside every `StackingEnsemble.fit` (a warm_start learner carried fold state); `cross_validate_ensembles`
  rejects `preprocessors` / `preprocessor_configs` (members are refitted on raw rows, so a separate transform would
  apply at predict time only; Codex measured R2 -30 vs 0.998) and non-numeric y; dropping a failed member resolves
  every member's effective preprocessing first, because `_get_preprocessor` allows short lists and a survivor could
  inherit the dropped member's transform.
- **Saving any ensemble with a GA / Combined wrapper member raised PicklingError**: the wrappers cache a local
  closure in `_transform`. They now drop it in `__getstate__` (it is rebuilt from `preprocess_config`).
- **Saved ensemble uncertainty used in-sample residuals** labelled `cv_residuals` (`_save_selected_ensemble`
  re-predicted the training rows). It now saves the outer-CV OOF predictions kept in each `ensemble_results` entry, or
  no CV data at all. It also recorded `task_type='auto'` when the task radio was on auto, which dropped the residuals.
- **Review round 2:** `BaseEstimator.__getstate__` returns the LIVE `__dict__` (py3.14 / sklearn 1.9), so editing it
  cleared the cache of an object being pickled mid-prediction; copy it first. The six ensemble wrappers moved from
  the GUI script to `spectral_predict/model_wrappers.py` (the GUI re-exports the names): pickles made by the GUI named
  `__main__.<Class>` and could not load in a script or a frozen app with a different entry module. `model_io` loads
  every pickle through `_joblib_load`, which temporarily points missing `__main__` / `spectral_predict_gui_optimized`
  names at the backend classes (adds only, removes afterwards). Ensemble files with `task_type='auto'` load as
  regression. `create_auto_ensembles` now warns that its CV is optimistic: specialists come from search-time
  regional rankings over all rows; honest per-fold rankings not implemented (no production caller).
- **Review round 3: the round-2 load shim mutated global state** (temporary `__main__` attributes and a stub
  `spectral_predict_gui_optimized` in `sys.modules`): racy under concurrent loads, it shadowed a real GUI import made
  during a load, and an interrupt could leak it. Replaced with a per-load unpickler: `model_io._joblib_load`
  repeats `joblib.load`'s non-memmap path with a `NumpyUnpickler` subclass whose `find_class` maps the six
  (`__main__` | GUI module, wrapper) pairs via `model_wrappers.resolve_legacy_class`. joblib 1.6 has no public
  unpickler hook; the one private helper used (`numpy_pickle._validate_fileobject_and_memmap`, decompression) is
  imported inside a try with a plain `joblib.load` fallback. `model_wrappers.LegacyWrapperUnpickler` does the same
  for plain pickle (GUI raw `.pkl` prediction models).
- **Black with `--target-version py314` rewrites `except (A, B):` to `except A, B:`**, which is a SyntaxError on the
  3.12 rollback build. Run Black on this repo with `--target-version py312`.
- `load_model` also maps `task_type='auto'` to regression, but only for a regressor (not an sklearn classifier and
  no `classes_`); classifiers stay 'auto'.

## 2026-10-02 - fix/ct-honest-labels (QW2, R085, R091, R128): gotchas
- **Two CT build paths.** The Build button (GUI ~55034) calls `_build_transfer_model_new`; the older
  `_build_ct_transfer_model` (~46990) has no callers. The roadmap's "TSR uses KS" line was true only of the dead
  one; the live one took the first n rows. Fix the live path first; the dead one was only made honest.
- **JYPLS-inv enhanced-y path indexed twice.** It subset the paired arrays by `transfer_indices` and then passed the
  subset together with the same indices to `estimate_jypls_inv`, which indexes again (wrong rows or IndexError).
  Dormant (radio disabled), fixed anyway.
- **R091 maths.** sklearn `PLSRegression.transform` = `(X - mean) @ x_rotations_` for `scale=False`; `x_weights_`
  only projects deflated X. The transfer is `mean_primary + (c + T_sat M - mean(T_primary)) P^T`, applied as
  `X @ B + offset`. Without the primary-block mean the 2-block stacked mean leaves half the instrument offset.
  Old saved jypls models have no `'offset'` and are now refused rather than silently applied.
- **`black --line-ranges` is not hunk-local.** A range touching one entry of a multi-line statement reformats the
  whole statement, so editing three tooltips re-quoted the entire ~770-entry `TOOLTIP_CONTENT` dict. Drop
  black-only hunks afterwards (normalise quotes, `\'`, whitespace and trailing commas, compare) before committing.
- **Worktree Bash guard.** In an isolated worktree the Bash tool refuses heredocs/compound commands it cannot
  verify; write scripts with the Write tool and run them with PowerShell instead.
- **`cross_val_score(groups=...)` breaks under sklearn metadata routing.** With
  `config_context(enable_metadata_routing=True)` the `groups=` keyword raises; a broad `except ValueError`
  around it silently skipped CV (JYPLS chose 1 component, cv_rmse=inf). Materialise
  `list(GroupKFold(...).split(X, y, groups))` and pass `cv=splits`; that works with and without routing.
- **The GUI holdout does not use `sample_selection.kennard_stone`.** `_validation_kennard_stone` (GUI ~20627)
  is its own pdist/squareform implementation; R085 only affected CT standards and model_io representatives.
- **CT dead code (round 2).** The transfer-model registry UI was never built (no `ct_registry_tree`, no bindings),
  so its seven handlers and `transfer_model_registry` were deleted. The quality-plot SG window had a floor of 5,
  so an ROI of 1-4 wavelengths raised and the shared try/except hid every plot; `ct_derivative_window` now
  adapts (None below 3) and raw/scatter plots are drawn independently of the derivative tabs.

## 2026-10-02 - User decision: booster tree count = one value from the pooled CV curve (xgb.cv / lgb.cv style)
Replaces per-fold early stopping on the scored fold (R028/R003/R022). Each fold is fit once at max rounds; staged
predictions give a pooled CV curve; one round count is chosen (like PLS LVs from RMSECV in Unscrambler), CV metrics are
reported at it, and the final model is fit on all calibration data with that count. Rejected: inner 10% holdout per
fold (too noisy at n~40-50), and n_estimators as a plain grid axis. Accepted caveat: the mild optimism of choosing on
the same folds, as for PLS LV selection. Implemented on branch fix/booster-early-stopping.

## 2026-10-02 - QW7 DPI/fonts gotchas (branch feat/dpi-fonts)
- **Tk has no font fallback list.** `font=(('Segoe UI','Arial'),10)` becomes the Tcl string `{{Segoe UI} Arial} 10`,
  which Tk reads as ONE family called "Segoe UI Arial"; Windows substitutes Arial. `('TkDefaultFont', 10, 'bold')`
  has the same problem: inside a tuple the name is a family, not the named font, so it also renders Arial. Use
  `tkfont.Font` named fonts (`self.fonts[...]`, Tk names `Dasp*`). Tk deletes a named font when the Python object that
  created it is garbage-collected, which is why `_init_named_fonts` also keeps the owning objects on `root`.
- **What scales by itself and what does not, once DPI aware.** Point sizes follow `tk scaling`, which goes from 1.333
  to 1.667 at 125%. Embedded matplotlib canvases rescale from `tk scaling` (`_update_device_pixel_ratio`). Literal
  pixel values do NOT scale. Fixed `Toplevel.geometry("WxH")` is the one that breaks: the custom-range dialog hid its
  Apply/Cancel buttons at 125% (it was already 9 px short at 100%). Wrap such sizes in `_px_geometry`. Fixed Treeview
  column widths break too: the 80 px Results column cut `1.23456e-05` to `1.23456e-0` at 125%.
- **Treeview row height depends on the Tk version (round-1 correction, round-2 wording).** My first note said
  "Treeview row height follows the font". It does not follow it live on either version:
  - **Tk 9.0.4** (shipped with 3.14) sets `rowheight` once, at style init, from the row font (linespace + 2): 17 px at
    96 dpi, 22 px at 120 dpi. It does not re-sync if the font or scaling changes later.
  - **Tk 8.6.15** (in `.venv312`, used by the `DASP_BUILD_PYTHON=312` rollback build) leaves `rowheight` empty and uses
    a fixed 20 px. That is exactly the linespace at 125% and clips text above it.

  `_apply_theme` now sets the `Treeview` rowheight to the TkDefaultFont linespace + `_px(2)`, which gives the same rows
  on both versions (verified under .venv312 and .venv314). That is 1-2 px more than Tk 9's own value at 150% and 200%,
  which is harmless. Invariant, also commented at that line: Treeview tag fonts must not be taller than TkDefaultFont.
  The default build is still 3.14 (`BUILD_PYTHON_VERSION = os.environ.get("DASP_BUILD_PYTHON", "314")`).
- **Column widths: measure, don't multiply (round 2).** Scaling a 70 px column with `_px(70)` gave 88 px at 125%, but
  `1.23456e-05` needs 81 px of text plus Tk's cell padding (4 px per side at 96 dpi, scaled), i.e. 91 px. Text does not
  scale exactly linearly (hinting), so the Results table now sizes float columns from the `.6g` text measured in the
  row font (`_float_column_text_width`) plus padding, with the 96-dpi widths as minimums. `font.measure()` already
  returns pixels at the current scale, so never multiply its result by `_UI_SCALE`.
- **Widest-string search (round 3).**
  - Two shortcuts each missed wider values: taking the longest strings by character count (`1.23456e+10` is wider
    than `1.23456e-05` at the same length, because '+' is wider than '-'), and taking only the first 500 rows plus
    the extremes.
  - Measuring every distinct string costs about 100 us each inside Tk, and that is text layout, not Python-to-Tcl
    overhead: moving the loop into Tcl made it slower, about 1.1 s per 10k strings.
  - On Windows, `font.measure(s)` equals the sum of its per-character widths exactly, because GDI text extents
    apply no kerning. Checked over 12k `.6g` strings, Segoe UI 9/10 and Arial 9, at 100/125/200%.
  - `_float_column_text_width` therefore ranks every distinct string by summed character widths, measures the top
    50 exactly, and returns the larger of the two. That is exact on Windows, safe elsewhere, and takes about 15 ms
    per 10k values.
- **`tk scaling` set on a fresh root does change font measurement in that interpreter** (62 / 81 px for
  `1.23456e-05` at 100 / 125%). Use that to test real-scale text widths without a DPI-aware process. It does not
  reproduce a true 200% DPI-aware process exactly (102 px here versus Codex's 132 px measured at real 192 dpi), so
  the tests compare widths and text measured in the same interpreter.
- **Dialog placement (round 2).** Tk's `winfo_screenwidth/height` on Windows describe the *primary* monitor only.
  `_px_geometry(size, owner)` now asks Win32 for the work area of the owner's monitor (`MonitorFromWindow` +
  `GetMonitorInfoW` `rcWork`, in the process's own DPI coordinate space, which is also Tk's) when the dialog opens. It
  clamps the size, leaving room for the title bar, and centres the dialog on the owner inside that area. Without
  Win32 it falls back to the Tk screen size.
- **Test processes start DPI-unaware**, so the session app's `_UI_SCALE` is 1.0 and pixel assertions are unchanged.
  Only `main()` calls `_enable_windows_dpi_awareness()`. However, the GUI module calls `matplotlib.use('TkAgg')`, so
  the first pyplot figure in a test with no running Tk mainloop declares **per-monitor** DPI awareness
  (matplotlib's `Win32_SetProcessDpiAwareness_max`). In the GUI suite this first happens in
  `test_contaminant_tab.py::TestApplyCorrection::test_apply_correction_no_attribute_error`. After that, Tk font
  measurements in the same process come back in physical pixels, even for pixel-sized fonts. Measure text in a fresh
  subprocess, as `tests/test_gui_dpi_fonts.py::sci_text_px_96` does. The real app is unaffected: there the mainloop
  is running, so matplotlib skips the call.
- **Black 26 targets 3.14 and rewrites `except (A, B):` as PEP 758 `except A, B:`.** That is a SyntaxError on 3.12,
  which the `DASP_BUILD_PYTHON=312` rollback build still uses. Keep the parentheses in the GUI file.
- **A custom PyInstaller 6 `manifest=` REPLACES the built-in template rather than merging.** The spec therefore copies
  the template's compatibility/longPathAware block verbatim; PyInstaller still injects the execution level and
  Common-Controls v6.
- **`sed -i` from Git Bash rewrites `spectral_predict_gui_optimized.py` from CRLF to LF.** The index stores CRLF, so
  every line then shows as changed. Restore with a byte-level `\n` to `\r\n` pass, or use the Edit tool.
- **Screenshot capture:** in a DPI-unaware process, `ImageGrab.grab(window=hwnd)` returns the pre-stretch logical
  bitmap, which hides the blur. Grab the full screen, which comes back in physical pixels, and crop it by
  `full.width / winfo_screenwidth()`.

## 2026-10-02 - fix/readers: OPUS block priority (R017) and one ASCII reader (R062)
- **brukeropus gotchas.** `OPUSFile.__getattr__` returns None (not AttributeError) for any absent name, so
  `hasattr(opus_file, 'a')` is always True; trust `data_keys` (1-D blocks only; `series_keys` are 2-D). A non-OPUS file
  does not raise in `read_opus`: it returns `is_opus=False`, and `__getattr__` then recurses on `self.params`
  (RecursionError, which `hasattr` does not catch). The reader now checks `is_opus` first and takes the first usable
  block in `OPUS_BLOCK_PRIORITY` (a, t, r, other processed types, then sm, then rf, with a UserWarning for sm/rf);
  metadata `opus_block` records the key. No real OPUS fixture exists in the repo or example/; tests use fakes.
- **Wrapper merge order.** io.py's vendor wrappers built `{normalised keys..., **file_metadata}`, so reader keys won.
  OPUS data_type became the raw 'transmittance'/'reference'. PerkinElmer file_format became 'sp' and x_unit the
  non-canonical 'wavenumber_cm-1', which the GUI's `_apply_x_unit_metadata` treats as nm (read_sp_dir passed it
  through as well). Reader keys now go first, and the PerkinElmer reader emits 'cm-1'/'nm'.
- **ASCII.** The later `read_ascii_spectra` (pd.read_csv, header=0, no folders) shadowed the folder-aware one, and
  five tests asserted the lost first row (2001 -> 2000). There is now one implementation. The delimiter comes from the
  first fully numeric row (the old folder parser chose it from the first line, so a heading with more spaces than
  tabs over tab-separated data picked ' ' and then parsed nothing). Lines before that row are the header. Files are opened as utf-8-sig so a BOM does not turn row 1 into a
  header. The x unit is taken only from explicit unit tokens in the headings, because our own writer labels x
  "Wavelength" whatever its unit. `_parse_ascii_file` now returns `(df, info)` and raises. Unknown kwargs raise
  TypeError (they used to go to pd.read_csv; no caller passes any).
- **Review round 1 (Codex BLOCK, GLM merge-with-fixes).** The pipeline data_type decides whether the GUI offers a
  log: 'reflectance' gets A = log10(1/R). So every OPUS block that is already logged or linear in concentration
  (logr = -log R, KM, ATR, PAS, Raman, emission, aria) must map to 'absorbance'; mapping logr to 'reflectance'
  logged it twice. "4000,5,0,123" (decimal comma + comma delimiter) splits into four integers and silently gave
  x=4000, y=5; such files are now refused (decimal='.' overrides). pd.read_csv's header=0 path had tolerated text
  columns, inline '#' comments and `decimal=','`; the hand parser must keep all three. The GUI never shows
  warnings.warn or print output: reader problems the user must see go in `metadata['import_warnings']`, which
  `_show_import_warnings` puts in a dialog (OPUS/ASCII/PerkinElmer main import only). PerkinElmer .sp has no unit
  field in specio; ranges up to 3300 (the Lambda UV/Vis/NIR limit) are ambiguous and now default to nm at 40%
  with a warning. `read_sp_dir` globbed `*.sp` + `*.SP`, which on Windows lists every file twice.
- **Review round 2.** `data_type` is a physical ordinate type, not a "may log" flag: the GUI uses it for
  10**-x conversion, plot labels, the absorbance-only Auto Bone FTIR gate and saved-model compatibility. Readers
  now report a third type, 'other' (`io.OTHER_DATA_TYPE`), for Kubelka-Munk, photoacoustic, Raman, emission and
  raw single-channel spectra; `source_data_type` names it. Log-reflectance and ATR stay 'absorbance'
  (absorbance-equivalent). The GUI offers no conversion for 'other' (main tab, prediction, both CT modes), and
  saves `source_data_type`/`data_type_converted_from` with models. The prediction and CT import paths used to
  re-run `detect_spectral_data_type` on values and drop reader metadata (an OPUS logr spectrum became
  "reflectance, 100%"); `_resolve_loaded_data_type` now prefers the reader's type, and every active import path
  calls `_show_import_warnings`. CT conversions used the main tab's `source_data_type`/`data_value_scale`;
  `_convert_with_source` swaps in the data's own. ASCII: every delimiter x decimal reading is scored on how many
  lines give numeric x/y; ties must agree or the file is refused. The decimal-comma guard looks only at x/y and
  the next field, plus leading-zero tokens (thousands groups). Fields are split with the csv module (quotes).
- **Review round 3.** More loaders re-detected the type from values: contamination, Multi-Model Comparison
  (including a validation-set source, which must take the main tab's current type), the CT wizard's
  primary/satellite loads and CT's "use as working data" handoff (must keep Mode B's type after conversion).
  `_load_spectra_from_directory[_as_df]` return arrays only, so they leave the reader metadata in
  `self._last_dir_load_metadata`. Compatibility: `model_io.check_data_type_compatibility` compares
  `source_data_type` when the pipeline types agree (Raman vs KM are both 'other'); legacy models without a
  source type, and converted data, are compared on the type alone; CT prediction now runs the check too.
  `_convert_with_source` now takes and returns the per-dataset value scale (resetting it to 1.0 broke % round
  trips). Tie comparison must be NaN-aware. The comma-ambiguity guard is per row (one competing row refuses the
  file) and covers thousands groups. Ensemble saves carry the ordinate keys into every base model.
- **Review round 4 (final).** Coordinator decision: comma-delimited files default to the decimal-point reading
  (decimal commas inside comma-delimited data are not valid CSV). Rows that also fit a decimal-comma or
  thousands split only warn (import_warnings, so the GUI shows it); the file is refused only when that split
  explains an inconsistency (differing field counts, repeated x, leading zeros). Round 3's per-row refusal
  rejected real files (integer nm or Raman shift + integer counts + a float column). A point-decimal third
  field can never be half of a decimal-comma pair. Contaminant groups now keep their own type/source/scale;
  a group whose stated type differs from the clean data (or is 'other') is refused, and conversion refuses
  while any group mismatches. `_loaded_value_scale` honours a carried scale before looking at the type, so
  converted % reflectance converts back to %. Source labels are canonicalised (`io.canonical_source_data_type`;
  Omnic 'Log(1/R)' == OPUS 'log_reflectance'). PowerShell 5.1 mangles `"` inside native-command arguments:
  write commit messages to a file and use `git commit -F`.
- **Review round 5 (final).** Contaminant compatibility: the non-convertible policy runs before any equality
  check ('other' matches only 'other' with the same canonical source, so Kubelka-Munk vs Raman is refused); stored
  groups are re-validated when clean data loads (with an offer to remove offenders) and again before
  difference analysis / automated detection; combined-file groups get their own records with a scale decided
  from the whole file; empty groups are rejected and `_contam_convert_data_type` computes every array before
  committing. ASCII folder summaries now keep every number-format (decimal-comma) warning in full and summarise
  other kinds per category with all files named; exponent fragments ("4,123E+3,0,123") count as warning-only
  evidence; a leading zero refuses only when a competing split exists ("0400,0.5" loads, "1,000,0.123" refuses).
  Note: the GUI's transmittance and reflectance formulas are the same number (-log10 T == log10(1/T)), so
  passing the carried source into contamination conversion is bookkeeping, not a value change. Deferred by the
  coordinator (see PROJECT_STATUS Follow-Ups): CSV/reference implicit-index shift and ASD-text decimal-comma
  misreads, both pre-existing.
- **Tooling gotcha.** The Bash tool's heredocs turned `\b` and `\n` inside Python string literals into real
  control characters (a backspace ended up in a regex). Write code containing backslashes with the Write/Edit
  tools, not via heredoc.

## 2026-10-02 - Classification label convention + pooled CV metrics + regression FoM (R029/R030/QW5, branch fix/classification-metrics)
- **Label convention lives in the METRICS, not the fitted labels.** `scoring.classification_metrics` scores against the
  sorted class list: binary positive = second sorted label, Specificity = TNR of the first, multiclass macro. Used by
  grid folds / pooled CV / calibration / `compute_validation_metrics_for_top_models`, Bayesian and NSGA-II CV +
  calibration, Model Development (CV, calibration, holdout, confusion panel) and the class-specialist ensemble CV F1.
- **Gotcha (review round 1, Codex HIGH): do NOT re-code numeric labels before fitting.** The first commit encoded
  every target to 0..K-1 in run_search; PLS-DA regresses on the label VALUES, so {1,2,100} gave a different model
  from Model Development / the validation helper / saved models, which refit the raw labels (CV acc 0.78 vs 0.83).
  run_search again fits the user's labels (text labels label-encoded, as before); only metrics apply the convention.
- **Gotcha: sklearn `cross_val_predict(method="predict_proba")` label-encodes y before fitting.** For PLS-DA with
  unevenly spaced numeric labels this fits a DIFFERENT model from `method="predict"` (or from a manual fold loop on
  raw labels). Compare against a manual fold loop, not cross_val_predict proba, when labels are not 0..K-1.
- **Gotcha: `model_io.save_model` with a LabelEncoder fitted on NUMERIC labels fails** (`label_mapping` has np.int64
  keys -> json TypeError). Being fixed on fix/wavelength-mapping; not touched here.
- **R030:** headline CV classification metrics are always computed from pooled out-of-fold predictions. Folds now
  return `y_proba` aligned to the global class order (zeros for a class missing from the training fold), so pooled
  AUC/LogLoss work under LOO. Repeated CV: labels majority vote, probabilities per-sample mean (same policy as
  `cv_utils.cross_val_predict_pooled`); AUC/LogLoss were previously mean-of-folds there. NSGA-II classification CV
  now pools too. A zero-filled class column can dominate pooled LogLoss (matches sklearn).
- Calibration F1/Precision/Recall were support-WEIGHTED (grid, Bayesian, NSGA-II, Model Dev) while CV used
  binary/macro; now the same definition everywhere. Bayesian and Model Dev multiclass AUC: weighted -> macro.
- **QW5:** `scoring.regression_figures_of_merit`. Bellon-Maurel 2010 "SEP" (Eq. 1) = our RMSE/RMSEP; their SEPc (Eq. 4,
  /m); our SEP/SECV is the n-1 bias-corrected form. RPIQ = IQR/RMSE (§5.3, Table 2 caption, no equation number).
  RPD/RER unchanged (ddof=0, RMSE-based) except a perfect model now gives inf instead of 0.0 (`scoring.spread_ratio`,
  also used by Bayesian, NSGA-II and Model Development). New columns: SECV, RPIQ (CV; grid, Bayesian, NSGA-II); SEP,
  Biaspred, RPDpred, RPIQpred, RERpred, CCCpred, Slopepred, Interceptpred, Bias_p_pred, Slope_p_pred (external
  validation). Bias/slope t-tests are acceptance tests only for the holdout; CV/calibration = diagnostic.
- inf ratios: `json.dump` writes the non-standard token `Infinity` into saved-model metadata; dasp's `json.load` reads
  it back, strict JSON parsers outside dasp reject it. Documented (scoring.spread_ratio), model_io left alone.
- Black's py314/py315 target rewrites `except (A, B):` to the PEP 758 `except A, B:` form; do not accept that in src
  (the py312 spec build cannot parse it). Use `black --target-version py312`.
- Not covered by the convention: one-class / multi-class SIMCA metrics, the Predictions tab statistics panel
  (GUI ~45332, weighted), `cv_utils` named scorers.
- **Round 2 gotcha: repeated-CV probability accumulation broadcast narrower fold probabilities.** SMOTE-ENN on hard
  data can leave a fold model with ONE class; `cross_val_predict_pooled` added its (n,1) proba into every class
  column, rows summed to K and Bayesian LogLosscv read ~2e-16 for a failing model. Fold probabilities are now aligned
  via the fold model's `classes_` (cv_utils `_proba_in_class_order`, also in the early-stopping loop), and
  `classification_metrics` makes AUC/LogLoss NaN when proba rows do not sum to 1 (atol 1e-4).
- **Round 3 (user decision, option b): Bayesian and NSGA-II fit numeric labels as given, except XGBoost.** XGBoost
  rejects labels that are not 0..K-1 (`Invalid classes inferred ... got [1 2]`), so those engines fit it on codes and
  decode its predictions before scoring. Bayesian studies whose numeric labels are not 0..K-1 get a `|labels=raw1`
  study-name segment (after `|boost_rounds=` when both apply); other study names are unchanged. NSGA-II's returned
  `label_encoder` is now None for numeric labels (the GUI used it to re-code validation labels while training labels
  stayed raw).
- **Follow-up (c):** an XGBoost label-encoding wrapper in models.py usable by every engine. Grid and Model
  Development still cannot fit XGBoost on numeric labels that are not 0..K-1.
- Test gotcha: validation-rebuild tests need round wavelengths; `all_vars` is written with `%g` (R031, being fixed on
  fix/wavelength-mapping), so `np.linspace` wavelengths silently fail to map back.
- **Round 4: one label-policy helper, `scoring.classification_fit_labels`** (used by Bayesian, NSGA-II incl.
  `_compute_top_variables`, the validation rebuild and the GUI). Integer-valued numeric labels (bool included) are
  fitted raw; text and NON-INTEGER numeric labels ({0.1,0.2}, which sklearn reads as continuous so stratified CV
  refuses them) are label-encoded as before (user decision). Grid search previously failed outright on fractional
  labels (StratifiedKFold "continuous"); run_search now encodes them too and returns the encoder (trivial fix), and
  Model Development encodes them likewise.
- **Gotcha: the Bayesian data fingerprint is dtype-sensitive.** LabelEncoder used to turn int32 / float64 / bool
  {0,1} into int64; fitting them "raw" changed the fingerprint and broke `auto` resume of unchanged studies. Labels
  already 0..K-1 now return exactly that int64 array (policy "codes").
- Resuming a pre-raw1 {1,2,100} study now emits "Resume declined for <model>: label policy changed" with
  `label_policy_changed` + `resume_declined` (GUI shows the resume-issue dialog); the old study stays in the SQLite
  file, and the run record is released after that notice, as for the environment-changed case.
- GUI: Bayesian holdout rebuild decodes its temporary codes back to raw integer labels (`_holdout_labels_as_fitted`);
  NSGA-II holdout encodes training AND validation labels with the search encoder (`_encode_holdout_pair`; raw text
  training labels against coded validation labels scored 0); display encoders are None for raw numeric labels
  (`_display_label_encoder`), and the class legend decodes keys only when every key is a code of the encoder.
- **Round 5 gotcha: `_run_analysis_thread` ends with a shared `self.label_encoder = label_encoder` for EVERY
  optimization method.** Setting only `self.label_encoder` inside the Bayesian branch was wiped by that tail (text /
  fractional Bayesian runs then showed codes in the legend and saved codes). Set the branch's local `label_encoder`.
- XGBoost Bayesian studies no longer get `|labels=raw1` (their code fit is unchanged); trials resumed from before the
  policy carry code-keyed per-class metrics, which `convert_study_to_dataframe(label_classes=...)` decodes when the
  key set is exactly the code set and differs from the user-label set (unambiguous).
- The pre-policy resume notice is hedged ("a study with this configuration but different class labels exists"):
  study names carry no data identity, so a 0..K-1 study of other data looks the same.
- `_save_refined_model` saves only `refined_label_encoder` (no fallback to the global search encoder).
- Model Development repeated CV now reduces to one prediction per sample (vote / mean / mean proba) before headline
  metrics, plots and stored predictions, as the grid does; its comparison line now uses the row's Accuracycv.
- NSGA-II classification objective = 1 - pooled accuracy (fold accuracies weighted by test size), so Accuracycv
  matches the pooled definition.
