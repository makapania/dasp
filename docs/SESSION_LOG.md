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
(CARS family, uve_*/fipls_* hybrids, spa, vcpa-iriv) + `_cap_top_n` caps N at the non-zero count, by method name only.
Codex warned against keying on "any zeros": `_apply_edge_mask`, tree importances, Lasso and the uniform fallback all
produce legitimate zeros. The tag keeps the requested count (`top250_cars`); `n_vars` records the fitted count (user's
choice). Gotchas: (1) grid must dedupe on the *capped* count, including against the method-optimal run (which already
equals the non-zero count and is tagged plain `cars`); (2) the one-class grid wrote the *requested* `n_vars` to the row,
now `len(top_indices)`; (3) the Bayesian fingerprint includes `subset_tag`, so capped trials would never dedupe. They now
fingerprint with `fit_tag` = `top{fitted}_{method}`. Not covered: GA (selection-frequency zeros) and the 'importance'
fallback when CARS raises inside `compute_importances`, which stays uncapped. Tests: `tests/test_sparse_selector_topn_cap.py`.
