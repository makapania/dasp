# Changelog

All notable changes to Spectral Predict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: the entries between `0.1.0` (early 2025) and `0.5.0b1` (April 2026)
> were not maintained in this file. The git history under
> `git log v0.1.0..0.5.0b1` (or commit `bbf7766` for the b1 cut) is the
> authoritative record for that period.

## [0.5.0b3] - Unreleased

### Added

- Results-row rebuild helpers on the declared composition surface
  (`docs/AGENT_COMPOSITION.md` §8b): `models.parse_row_params`,
  `models.estimator_params_from_row`, `models.plsda_head_kwargs`,
  `preprocess.preprocessing_config_from_row`, `ga_preprocessing.chromosome_from_row` and
  `ga_preprocessing.chromosome_to_steps`.

- **T-51 PR A** — opt-in extra hyperparameter axes for the unified Bayesian search.
  `run_unified_bayesian` gains `enabled_extra_axes`, `search_space` and
  `n_startup_trials`. The new `spectral_predict.search_spaces` module provides
  `AxisSpec`, `BundleSpec`, `ExtraAxesConfigError` and the curated `BUNDLES` registry,
  which is empty until PR B/C. Bundles may open only hyperparameters the default space
  pins to a single value. Collisions and malformed bundles raise before any study is
  created. With nothing enabled, the search, its TPE trajectory and its study names are
  unchanged. `n_startup_trials` now also survives the T-41 in-memory→SQLite
  auto-migration. See `docs/AGENT_COMPOSITION.md` §7b.
- **T-51 PR B** — curated supervised bundles in `search_spaces.BUNDLES`, all off by
  default: `rf_features`, `xgb_regularization`, `xgb_child`, `xgb_sampling`,
  `lgbm_regularization`, `lgbm_sampling`, `lgbm_child` (including `min_split_gain`),
  `catboost_sampling` (also sets `bootstrap_type='Bernoulli'`, which `subsample` needs
  for multiclass), `svm_gamma` (written only on RBF-kernel trials), `mlp_activation` and
  `plsda_head` (the PLS-DA logistic head's `C`, stored as `lr__C`). Enable them from
  Python with `run_unified_bayesian(..., enabled_extra_axes=(...))`; there is no GUI
  control yet (PR D). Enabled bundles give the run its own study name. Sampled values are
  stored in `Params` as estimator parameters and survive rebuild, Tab 7 refit,
  save/load and export. With no bundle enabled, searches and study names are unchanged.
  `apply_extra_axes` now also rejects two axes that share an Optuna name or key when
  `resolve_bundles` is bypassed.
- **T-51 PR C** — three opt-in one-class bundles in `search_spaces.BUNDLES`, all off by
  default and resolving only for `task_type='one_class'`: `if_max_samples`
  (IsolationForest `max_samples` from {auto, 0.5, 0.8, 1.0}), `lof_metric` (LOF `metric`
  from {euclidean, manhattan, cosine}) and `ocsvm_poly` (One-Class SVM `degree` int 2-3 on
  poly trials, `coef0` -1 to 1 on poly and sigmoid trials — both suggested every trial but
  written only where the kernel uses them, so rbf trials are unchanged). Enable them from
  Python with `run_unified_bayesian(..., enabled_extra_axes=(...))`; there is no GUI
  control yet (PR D). Enabling any bundle gives the run its own study name, so a
  default-space study is never resumed or polluted. The benchmark showed no gain from
  floating these axes; they are for deliberate exploration. A bundle can declare a
  minimum training-fold size (`BundleSpec.min_train_fold_rows`); `if_max_samples` needs
  2, because a fraction of a one-row fold is zero samples and sklearn raises. Too small
  a fold now raises `ExtraAxesConfigError` before the study is created instead of
  failing every trial into a penalty.

### Fixed

- **Resuming a crashed Bayesian run no longer changes the persistence setting.**
  Accepting "Resume previous run?" used to force *Crash-resume persistence* to
  *Always on*, because 'auto' once ignored the saved study. 'auto' now reloads a saved
  study whose configuration and data match, so the setting is left as the user (or the
  restored run settings) had it, and the banner no longer mentions Always-on. The
  prompt is also skipped when the crashed run saved nothing: a run under *Always off*,
  or an 'auto' run that crashed during its in-memory warmup. Before, those prompted and
  then reported "Resume failed". The stale sidecar is kept (a cross-process
  check-then-delete is unsafe) and replaced by the next Bayesian run. New
  `run_state.has_resumable_store(meta)`.
- **A resumed run that cannot reuse saved trials now says so.** An 'auto' run whose
  SQLite file exists now reports "previous results computed in a different numerical
  environment". Before, only 'always' did, so after e.g. a NumPy update an accepted
  resume silently started over. Declines (environment change, older study format,
  different or unrecorded data) carry a `resume_declined` progress key, but only when
  the declined study has completed trials. A failed study check under 'auto' emits
  `resume_check_failed`. During a resume the GUI logs these, updates the status line
  and warns once per run. It does the same for 'always' resumes that replay trials from
  different (`data_mismatch_resume`) or unverifiable (`data_unverified_resume`) data,
  worded as "resumed, but …". The resume banner no longer promises unconditional reuse.
- **Behaviour change: data that doesn't match the resumed run no longer deletes it.**
  Clicking Run Analysis on a resumed run with different data used to discard the
  sidecar *and the SQLite store* and silently start fresh. Now nothing is deleted. The
  check runs on the main thread before the analysis starts, so there is no timeout.
  A resume that cannot be verified is not treated as a match: an unreadable or
  missing record now raises `run_state.ResumeVerificationError`. A dialog shows both
  data fingerprints, or why they couldn't be checked, and offers two choices:
  - **No (default):** keep the saved run. Nothing runs, so you can load the matching
    data and click Run again to resume.
  - **Yes:** delete the interrupted run and start fresh now (round 7: there is only
    one sidecar slot, so leaving the old run in place while a new one starts would
    silently overwrite its sidecar and orphan its SQLite store — it could never be
    offered again).

  Pending validation indices are kept while the resume is pending and cleared on a
  fresh start. Keeping the run returns the UI fully to idle and ends the search
  controller. New `run_state.get_resumed_run()`.
- **A grid, NSGA-II or multi-class SIMCA run no longer removes a pending Bayesian
  resume.** Every successful analysis called `run_state.mark_complete()`, which
  deleted the resumed run's sidecar. It is now called only for the run the analysis
  registered. Multi-class SIMCA registers none, because it does not use Optuna.
- **Finished one-class Bayesian runs no longer ask "Resume previous run?" on every
  launch.** The one-class branch returned without releasing its resume record. Every
  Bayesian branch now releases it right after the search, before CSV, report and
  ensemble steps, so a failure in those steps can no longer leave a finished run
  resumable.
- **Behaviour change: a Bayesian run stays resumable if any model's search raised, or
  if the user pressed Stop.** It used to be released whenever the analysis finished.
  The next launch offers to resume or discard it.
- The analysis worker binds the loaded data once and uses those arrays for both the
  resume check and the search. Loading other data or changing the target while it
  starts can no longer run the search on data that was not checked.
- New `run_state.get_active_run_id()` and `SearchController.is_end_requested()`.
- `run_state.resume_run` refuses a sidecar storage path that raises `ValueError`
  (e.g. an embedded NUL) instead of crashing the startup check.
- **Round 7 (reviews of #79): a paused or failed run is now offered again — resume,
  delete, or decide later — every time it could otherwise be silently reused or
  silently orphaned.** User decision: a saved paused/failed/crashed Bayesian run is
  asked about until it is resumed to completion or deleted.
  - **Behaviour change:** clicking Run Analysis while an un-decided resumable run sits
    on disk (paused via Stop or left over from a model failure earlier this session,
    or from a startup "decide later" answer) now shows the same three-way choice as
    the startup dialog — *Resume* / *Delete* / *Decide later* — before the click's own
    `start_run()` can silently overwrite the shared sidecar. *Resume* falls through to
    the existing data-fingerprint check; *Delete* removes it and starts fresh; *Decide
    later* cancels the click and touches nothing.
  - A Stop or a model failure used to leave the run's in-process state active but not
    flagged as resuming, so the *next* `start_run()` idempotently returned that stale
    metadata with no fingerprint check ever running (`_confirm_resume_before_launch`
    only checks while `is_resuming()`). `_complete_run_state_after_search` now releases
    the in-process claim (`run_state.clear_resume_state`) on Stop/failure — the sidecar
    and SQLite store are untouched, so the next click or launch still finds them.
  - **Stop → Run race:** if the user pressed Stop and clicked Run Analysis again
    before the old worker noticed, `self.search_controller` was already the new run's
    (unstopped) controller by the time the old worker checked whether it had been
    stopped, so it wrongly released the old run's record. Each worker now captures its
    own `SearchController` when `_run_analysis` starts it, and uses only that one for
    every stop decision and every search call.
  - **A second Run Analysis click is refused while a previous worker is still alive**,
    rather than racing two workers on `self.search_controller` and on run_state's
    single active run.
  - **A model whose search comes back with no usable results now counts as failed,
    same as a raised exception.** When every trial for a model fails inside the
    objective, the backend absorbs the exception into a 1e10 penalty per trial and
    returns an empty results frame rather than raising — which the GUI's "0 errors"
    count had been treating as a clean finish, releasing a run that produced nothing.
  - `_uses_bayesian_run_state` now also checks `HAS_UNIFIED_BAYESIAN`. A build where
    `unified_bayesian` failed to import used to register a run and then bail out
    before ever completing it, leaving a permanent phantom resume prompt.
  - The training-config `total_samples_original` metadata read `self.X_original` /
    `self.X` again after the worker bound `X_run`/`y_run` to a snapshot at start —
    changing data while the worker was still setting up could report a stale count
    against the wrong dataset. Both branches now use the worker's own bound snapshot.
  - The mismatch dialog's "Yes, start fresh" now deletes the interrupted run instead of
    just abandoning it in memory, for the same orphaning reason as above (see the
    updated behaviour-change entry further up).
- **Round 8 (Codex + DeepSeek review of round 7): `_confirm_resume_before_launch` is now
  the single main-thread authority for a Bayesian launch.** It snapshots the
  optimization method, task type, selected models, persistence mode and the loaded
  data, decides resume/delete/fresh, and claims the run slot (registers or resumes it)
  *before* the worker thread is even created. The frozen decision passes into
  `_run_analysis_thread` as plain arguments; the worker never re-derives any of it from
  live Tk state and never re-decides. Everything round 7 verified as working (the
  controller race fix, empty-frame-as-error, the `HAS_UNIFIED_BAYESIAN` gate, the bound
  `X`/`y` snapshots, explicit-deletion wording, grid/NSGA-II staying unaffected) is
  unchanged.
  - **Resuming a run whose model selection changed now asks, and defaults to the
    run's original models.** Stopping a PLS search, selecting Ridge, and clicking
    Resume used to run Ridge in Ridge's place while PLS's study stayed unfinished, and
    then release the record as if the whole run had completed. A mismatch between the
    resumed run's saved `model_names` and the current click's selection now asks; "Yes"
    substitutes the run's own models for this click, "No" cancels it.
  - **Behaviour change: a read failure, or a failed `resume_run()`, no longer starts a
    fresh analysis.** Both used to fall through to "launch anyway," letting this
    click's own registration silently overwrite a saved run's sidecar that was never
    actually confirmed absent. Both now refuse to launch and ask the user to retry.
  - **Behaviour change: a Delete that doesn't fully succeed no longer starts a fresh
    analysis either** (in both the pending-run three-way dialog and the mismatch
    dialog) — same reasoning: an unconfirmed delete must not be treated as "safe to
    overwrite."
  - `run_state.discard_incomplete_run` re-reads the sidecar's run id immediately before
    unlinking it, and refuses if it no longer matches — its own initial
    `find_incomplete_run()` read and the unlink were not atomic, so another dasp
    instance's `start_run()` in between could have replaced the sidecar with its own
    run, which the old code deleted anyway.
  - **Every exit from the worker after a run is registered/resumed now either completes
    normally or releases the in-process claim while keeping the sidecar.** A `try`/
    `finally` around the whole worker body closes this uniformly: the one-class
    inlier/guard checks' early `return`s, and any exception during setup, used to leave
    the claim dangling, so the next `start_run()` could idempotently reuse stale
    metadata with no fingerprint check ever running. The resume fingerprint re-check's
    own early return (a deliberately-handled exit: the resume stays exactly as it was)
    is explicitly excluded from this generic cleanup.
  - A model whose saved study already has its full trial count, but every trial is a
    penalty, now says so and points at Delete as the fix, instead of just "treated as
    failed" (which would otherwise re-prompt forever with no next step spelled out).
  - Wording: the "decide later" choice also says a new Bayesian analysis can't start
    until the saved run is resumed or deleted, and that Grid/NSGA-II searches aren't
    affected.
- **Round 9 (Codex review of round 8).**
  - **Behaviour change: a damaged saved-run record is reported, not silently moved.**
    `run_state.find_incomplete_run` used to rename an unparseable record to `.corrupt`
    (or delete it when the rename failed) and report "no saved run", so the next run
    replaced it. It now raises `run_state.CorruptRunRecordError` and leaves the file
    alone. Records with missing or mistyped required fields also count as damaged.
    At launch and on Run Analysis the GUI asks whether to move it aside; "No" leaves it
    and runs nothing. New `run_state.set_aside_corrupt_run_record()` renames it to a
    unique `active_run.corrupt-<timestamp>.json`, but only if it is still damaged.
    `start_run` sets a damaged record aside before writing and raises `OSError`
    rather than overwrite it.
  - **A resume uses the run's own trial count.** A resume always runs the saved
    `model_names` and `n_trials_per_model`. When either differs from the current
    controls, the GUI asks first.
  - **Behaviour change: analysis settings that differ from the resumed run are shown
    before anything runs.** Yes puts the run's settings back (nothing runs; click again
    to resume). No deletes the run and starts fresh with the current settings. Cancel
    changes nothing. The persistence radio, model checkboxes, trial count and
    display-only options are not compared (`run_gui_settings.diff_gui_settings`). Model
    hyperparameter ranges are not in the saved settings and are not checked.
  - A one-class Bayesian run uses the model list frozen at the click. It no longer
    re-reads the one-class checkboxes, which could run IsolationForest in place of a
    resumed PCA-SIMCA run.
  - The worker receives the data checked at the click (`analysis_data`) and the frozen
    trial count (`analysis_n_trials`), instead of binding them when the thread starts.
  - A model or trial override frozen for a resume is cleared when that run is deleted
    or replaced by a fresh one.
  - If the worker thread fails to start, the run the gate claimed is released and an
    error is shown.
  - If `mark_complete` can't delete a finished run's record (e.g. a Windows file lock),
    the in-process claim is still released. The record stays on disk and is offered at
    the next launch.
  - User-facing text no longer says "sidecar" or "saved-run slot".
  - **Round 10 (Codex review of round 9):**
    - The worker dispatches on the optimization method and task type read at the
      click. Switching Grid to Bayesian while a Grid run was starting could run a
      Bayesian search on a resumed run's storage with no data check.
    - `bayes_enable_autoscale`, `k_neighbors`, `n_bins` and `boost_factor` are now
      captured and compared on resume. They are part of the Bayesian study name, so
      changing one used to silently start a new study.
    - **Behaviour change:** `discard_incomplete_run` deletes the store before the
      record, and keeps the record when the store is locked. Before, a locked store
      left a deleted record and a store no retry could reach. A store that is
      already gone counts as deleted, so a retry after a half-finished delete
      completes.
    - A record that becomes damaged after a resume was claimed is offered for moving
      aside from Run Analysis. Before, only restarting dasp got out of that state.
  - **Round 11 (Codex review of round 10):**
    - The Bayesian branches read their settings (CV, baseline, smoothing, regions,
      UVE, autoscale, imbalance, persistence) from a snapshot taken at the click
      (`analysis_settings`). Changing one during a multi-model run used to give later
      models a different study. When that run finished, the record of the approved
      one was released while its studies were still unfinished.
      `_get_baseline_params_for_method` and `_get_imbalance_params` take an optional
      `get` reader.
    - If a delete removes the store but can't remove the record, the resume is
      released. A later click with matching data used to "resume" a run whose trials
      were gone.
    - `discard_incomplete_run` also keeps the record when the store path can't be
      resolved (`OSError`), so the delete can be retried.
    - After a damaged record is moved aside, the claim is released before the re-read,
      so a failing re-read can no longer leave the resume stuck.
  - **Round 12 (Codex review of round 11):**
    - Calibration rows (validation holdout, excluded spectra, active subset) and the
      CV folds used to drop too-small classes are frozen at the click
      (`analysis_rows`, snapshot settings). Changing them after the click used to
      change the rows the Bayesian search trained on.
    - **Behaviour change:** a resume checks the validation holdout. If the current
      holdout differs from the run's, a dialog offers to use the run's split, delete
      the run and start fresh, or cancel. With no holdout set, the run's split is
      restored without asking. The restore happens on the main thread before launch.
      It used to happen in the worker, which kept a different manual split.
    - **Behaviour change:** a Bayesian launch stops with "Invalid settings" when a
      setting the search needs can't be read (e.g. a blank number box). With a launch
      snapshot, the worker never falls back to a live control.
    - A delete that removed the record but not the store also releases the resume.
      The dialog says the run can no longer be resumed instead of offering a retry.
  - **Round 13 (Codex review of round 12):**
    - A resumed run that held nothing out is compared too. Creating a validation
      holdout before resuming changes the rows that run trained on, so the same
      three-way dialog now appears (use the run's split — here, no holdout — delete
      and start fresh, or cancel).
    - A failed validation restore changes nothing. It used to clear `validation_X`
      before checking the run's samples were present, which left the current
      validation set half-cleared and silently skipped validation metrics in a later
      run. Both replacement slices are built before anything is replaced.
    - A blank detail box belonging to a switched-off option (e.g. the smoothing
      window with Bayesian smoothing off) no longer blocks the launch.
    - The pending-run Delete message no longer offers a retry once the record or the
      store was deleted.
  - **Known limitations (not addressed):**
    - Two dasp windows share one `active_run.json` with no file lock. The last
      `start_run` wins, and a window may offer or delete a run another window just
      recorded. `discard_incomplete_run` and `set_aside_corrupt_run_record` re-read the
      record first, which narrows but does not close the window.
    - A finished run whose record couldn't be deleted is offered for resume again.
    - A run whose SQLite store is missing or empty is not offered. Its record stays
      until the next Bayesian run replaces it.
    - A resume restores only the whitelisted GUI settings, not per-model
      hyperparameter grids.
- **Ensembles trained from Bayesian results now use the tuned hyperparameters.**
  Ensemble model reconstruction discarded every `model__*` key in a row's `Params`, and
  Bayesian rows store all estimator params under that prefix, so each base model trained
  with defaults (e.g. RandomForest `n_estimators`/`max_features`, SVM `C`/`gamma`, MLP
  `activation`, Ridge `alpha`, boosting learning rates and regularisation). The prefix is
  now stripped through the new shared `models.estimator_params_from_row`, which the
  validation rebuild also uses. Grid-search rows (bare keys) are unaffected, except
  that PLS `n_components` above 10 is no longer clipped to 10 in ensembles.
  **Ensemble results built from Bayesian rows change.**
- **Ensemble models are rebuilt with the row's full preprocessing.** Only `snv`,
  `snv_deriv` and `deriv_snv` rows got preprocessing. `deriv` rows (no derivative),
  every `+`-affixed name (`raw+autoscale`, `als+snv`) and the `Autoscale`, baseline and
  smoothing columns were ignored, and the per-model scaler was added even when the
  search had autoscaled. Wavelength subsets of grid/Bayesian rows were taken *before*
  preprocessing. The ensemble rebuild now shares `preprocess.preprocessing_config_from_row`
  with the validation rebuild, skips the per-model scaler for autoscaled rows, and takes
  subsets after preprocessing, as the search does. `raw`/`snv` rows are now plain
  Pipelines rather than GUI preprocessing wrappers, so ensembles of them refit base
  models per CV fold (the ensemble default). Legacy `sg1`/`sg2`, `deriv1`-style and GA
  preprocessing names keep their old path. Validation rebuild: a NaN `smoothing` cell
  (mixed results tables) no longer turns smoothing on, and a NaN `PreprocessBase` falls
  back to `Preprocess`. **Ensemble results change.**
- Validation rebuild and ensemble reconstruction accept a `Params` cell holding a dict
  (in-memory result rows) as well as `str(dict)`, via the shared `models.parse_row_params`.
- **Exhaustive-preprocessing rows are rebuilt from their chromosome in ensembles**, as the
  validation rebuild already did (their `PreprocessBase`, e.g. `snv_deriv1_w11`, is not a
  pipeline name). New `ga_preprocessing.chromosome_from_row` / `chromosome_to_steps` share
  the parsing and the per-spectrum steps with the search-time transform, which is
  bit-identical after the refactor. The steps start with a float64 conversion, as the
  search transform does, so float32 spectra (SPC files) match the validation rebuild.
- Validation rebuild: a legacy row in a mixed results table (`preprocess_chromosome`
  NaN, `ga_genes` set) decodes its `ga_genes` chromosome again instead of falling back to
  the preprocessing name. A malformed or out-of-range chromosome raises `ValueError` and
  that row falls back to its name, instead of an `IndexError`.
- A derivative row with a missing `Deriv` / `Window` rebuilds with 1 / 15 (the GUI's old
  defaults) in both paths; validation used to fail on it.
- **One parser for `Autoscale` / `smoothing` flag cells.** New `preprocess.parse_bool_cell`
  (strings `'true'`, `'1'`, `'1.0'`, `'yes'`, `'on'`, case- and whitespace-insensitive;
  NaN/`None` give the default) is used by the validation rebuild, the ensemble rebuild,
  the GUI model loader, the code exporter and the one-class validation rebuild. They
  previously disagreed on `'on'` / `'1.0'`, and the one-class rebuild read a `smoothing`
  cell of `'False'` as on.
- Malformed preprocessing chromosomes (huge integers, 0-d arrays, deeply nested
  literals) always raise `ValueError`, so the row falls back to its name or is skipped
  instead of escaping the GUI's and validation's error handling. `'[]'` falls back to
  `ga_genes` like an empty list.
- `plsda_head_kwargs` coerces `lr__random_state=42.0` to `42` and `'None'` to `None`,
  and raises `ValueError` on values `LogisticRegression` would reject.
- **PLS-DA heads rebuilt from a row keep the search's seed and class weighting.**
  Validation rebuild and ensemble reconstruction forced `random_state=42`; ensemble
  reconstruction also dropped `class_weight`. Both now restore the row's
  `lr__random_state` and `lr__class_weight` (new `models.plsda_head_kwargs`), so a
  search run with another seed and a stochastic solver (`saga`) refits the same head.
  Rows without a recorded seed keep 42.
- **Ensemble refits of CatBoost models saved before the `catboost_info/` fix** no longer
  write that directory. Per-fold clones get `allow_writing_files=False`, found by walking
  params, attributes and step lists (the GUI wrappers' `get_params(deep=True)` is
  shallow), so CatBoost nested in a Pipeline, a GUI wrapper or a VotingRegressor is
  covered. Previously the refit failed in an
  unwritable cwd and the model silently got NaN out-of-fold predictions.
- **GUI NameErrors.** The GUI module had no `logger`, so Model Development refit crashed
  when the task radio disagreed with the saved result's Task (and in two other warning
  branches); it now logs to `spectral_predict.gui`, which reaches `dasp.log`. The
  learning-curve error callback referenced the except-bound `e` after the block ended
  and raised instead of showing the error.
- **Bayesian search and extra-axes post-merge fixes** (reviews of T-51 PR B / T-41):
  - **Behaviour change: a failed 'auto' SQLite migration no longer deletes anything.**
    The cleanup called `optuna.delete_study` by name, and nothing could prove this
    attempt had created that study. Three ways it deleted someone else's:
    - a locked or permission-denied file read as "no file" (`Path.is_file` swallows
      those errors);
    - a `sqlite:///file:x.db?uri=true` URL parsed as a missing path;
    - another process created the study between the check and the copy, and the copy
      then failed with a transient lock error.

    The migration failure is now logged as a warning that names the study and storage;
    a partial copy of the study may remain there. The copy carries this run's data
    fingerprint (Optuna 5.0 copies study attributes before trials), so a later 'auto'
    run on the same data resumes it, or discard it with
    the run's saved state. The resume-gating file check now calls `stat` directly and
    never treats a SQLite URI filename as an existing file.
  - An 'always' run that resumes a persisted study whose data fingerprint is missing
    (a legacy study) or unreadable now warns that the data it ran on can't be verified. The progress
    event carries `data_unverified_resume: True`. The study is still resumed.
  - `run_unified_bayesian` raised `NameError` while building its results table whenever
    `baseline_method` was set and any trial applied baseline correction.
    `convert_study_to_dataframe` gains a `baseline_params` keyword. Baseline rows now
    carry the run's `baseline_params`, so the validation rebuild uses non-default
    ALS/polynomial settings. An empty results frame now also has the
    `baseline_method` and `baseline_params` columns.
  - `svm_gamma` resolved for `SVM` + regression and `SVR` + classification, where every
    trial silently became a penalty. Those pairs now raise `ExtraAxesConfigError`.
    `BundleSpec` gains an optional `family_task_types`. The bundle's revision and space
    identity are unchanged, so existing `svm_gamma` studies for the supported
    `SVM` + classification and `SVR` + regression pairs still resume.
    `BundleSpec` is now hashable. Its `constants` and `family_task_types` mappings are
    left out of `__hash__` but still count for equality. `AxisSpec` converts NumPy
    bounds, steps and choices to Python builtins when it is constructed. Before this,
    `np.float32(0.1)` compared equal to `0.1` but hashed its true value. The curated
    bundles' space identities are unchanged.
  - `model_name="pls-da"` is normalised to `"PLS-DA"`. Before this, `plsda_head` never
    resolved and every trial failed to build. Lowercase callers now get the `PLS-DA`
    study name.
  - Categorical choices that compare equal (`1` and `1.0`, `True` and `1`) are
    rejected. Optuna treats them as one choice, so a trial could fit one value and
    record the other.
  - NumPy scalars in bundle `constants` and `choices` are converted to Python
    builtins. Before this, `np.str_` or `np.float64` reached the `Params` string, which
    `ast.literal_eval` cannot parse, and `np.int64` was rejected.
  - Two latent flake8 F821 names (`PersistenceMode` annotation, a dead `return model`
    in `models.get_model`).

- **CatBoost no longer writes `catboost_info/`.** Every CatBoost fit wrote a
  training-log directory into the current working directory, so fits failed with
  `Can't create train working dir: catboost_info` when the cwd was unwritable (an
  install under Program Files), concurrent fits could race on it (seen in CI), and it
  littered wherever the app ran. Every CatBoost construction (`get_model`,
  `build_model`, the grid, NSGA-II, preprocessing discovery, diagnostics validation
  curves) now passes `allow_writing_files=False` via `models.CATBOOST_RUNTIME_PARAMS`.
  It is a runtime kwarg, not model identity: result-row `Params`, Bayesian trial
  `model_params`, fit fingerprints and study names are unchanged. Exported Python
  scripts now include `'allow_writing_files': False` in the CatBoost `model_params`.
  Models saved before this fix still carry the default if refit after loading.

- **T-51 PR B0** — PLS-DA models rebuilt from a results row now keep the tuned
  LogisticRegression head (`C`, `solver`, `max_iter`) and the PLS transformer settings.
  Validation rebuild used `C=1.0` for every current PLS-DA row. Model Development
  refit and exported scripts lost the head for rows that spell it `lr_C` / `lr_solver` /
  `lr_max_iter`, and ensemble training ignored it for every row. Refits of grid searches
  whose `plsda_lr_C_list` differs from 1.0 therefore change, and now match the search's
  head params. Ensemble training re-applying `class_weight` and the head seed is covered
  by the PLS-DA head entry above.
  `build_model('PLS-DA', params)` no longer raises on `lr_*` or `pls__*` keys.
  Search-time scores, the default Bayesian search and study names are unchanged; no
  version bump.

- **T-51 step 1** — classification SVM is now StandardScaler-wrapped in grid search,
  Bayesian search, validation rebuild and Model Development refit. The scale-sensitive
  sets listed `'SVC'`, which no model name matches; the registered family is `'SVM'`, so
  every classification SVM had been fit on unscaled spectra. Exported scripts already
  scaled it. **This changes SVM classification results.** `__version__` is part of
  every Optuna study name, so persisted Bayesian studies for **all** models start fresh
  after upgrading. The old studies stay on disk.

## [0.5.0b2] - 2026-05-03

Second beta of the 0.5.0 cycle. Bug-fix-and-observability batch on top of
`0.5.0b1`, plus one user-visible behavior change (T-19 Auto mode).

### Added

- **T-19** — model-native imbalance handling exposed through the Search tab,
  including an `Auto` mode that resolves to a sensible per-model default at
  runtime (instead of forcing the user to pick one). Boosting paths thread
  `sample_weight` correctly across resamplers.

### Changed

- **T-47** — Bayesian persistence default flipped from `"never"` to `"auto"`.
  Searches are now resumable out of the box; users get the recovery path
  without having to opt in.
- **T-14 / T-14b** — every version-displaying surface (report footer,
  exported-script header, exported-notebook metadata, GUI title bar,
  in-canvas version label, PyInstaller `version_info.txt`, Inno Setup
  `MyAppVersion`, build script `VERSION`) now derives from the canonical
  `spectral_predict.__version__`. Bumping the version in one place updates
  every artefact in lockstep. Regression tests pin the contract.

### Fixed

- **T-06 / T-06b** — canonical Araújo-2001 SPA enumeration; parallelised seed
  loop via joblib threading.
- **T-21** — hides x-unit Convert button in cases that produced a non-uniform
  wavelength grid for Savitzky–Golay derivatives.
- **T-11** — pause/resume hardening, Optuna SQLite storage, on-disk run logs,
  study-name fingerprint completeness, narrowed import catches.
- **T-29** — replaced bare `except:` in scoring with `except Exception` and
  warning emission, so silent metric failures surface in the run log.
- **T-30** — removed leftover `[DEBUG]` and `[PLS-DA DEBUG]` `print()` calls
  from `search.py` (`calibration_transfer.py` and `nsga2_search.py` triage
  follow as T-30b).
- **T-32** — corrected `y_train_for_model` threading through resampler +
  `sample_weight` path (boosting models on imbalanced classification).
- **T-38** — deleted dead preprocessing modules and a dead GUI flag.
- **T-42 / T-43 / T-44** — sidecar metadata correctness: write-path plumbing,
  resume restore validation indices, n_trials variable typo fix,
  task_type sibling phantom hasattr.
- **T-45** — wired file handler so module `logger.warning` lands on disk;
  CLI bypass + reload dedup follow-ups closed.
- **T-46** — surfaced `_apply_wal_pragmas` return value at both call sites.
- **T-47** — fix-of-fixes for the `auto` default flip (DeepSeek MEDIUM + 2
  LOWs).
- **T-49** — persisted validation indices on resume (correctness blocker).
- **T-50** — auto-cleanup of stale Optuna SQLite trial archives at app
  startup; configurable retention is queued as T-50b.

## [0.1.0] - 2025-01-27

### Added

#### Core Features
- **CSV Input Support**
  - Wide format: first column = ID, remaining columns = wavelengths
  - Long format: automatic detection and pivoting for single-spectrum files
  - Validation for minimum 100 wavelengths and monotonic ordering

- **ASD File Support**
  - ASCII .sig file reader with robust numeric data detection
  - ASCII .asd file reader
  - Binary .asd detection with clear error messages
  - Support for multi-column formats (automatically selects last column as reflectance)
  - Header line skipping for files with metadata

- **Preprocessing Pipeline**
  - Standard Normal Variate (SNV) transformer
  - Savitzky-Golay derivative (1st and 2nd order)
  - Configurable window sizes (7, 19) and polynomial orders
  - Multiple preprocessing combinations: raw, snv, deriv, snv→deriv, deriv→snv

- **Model Ensemble**
  - **Regression**: PLS Regression, Random Forest, MLP
  - **Classification**: PLS-DA, Random Forest, MLP
  - Grid search over hyperparameters:
    - PLS: n_components [2, 4, 6, 8, 10, 12, 16, 20, 24]
    - Random Forest: n_estimators [200, 500], max_depth [None, 15, 30]
    - MLP: hidden layers [(64,), (128, 64)], alpha [1e-4, 1e-3], learning_rate [1e-3, 1e-2]

- **Feature Selection**
  - Variable Importance in Projection (VIP) for PLS models
  - Feature importances for Random Forest
  - Weight-based importances for MLP
  - Automated subset selection: top-20, top-5, top-3 variables

- **Cross-Validation & Metrics**
  - 5-fold CV (configurable)
  - Stratified K-fold for classification
  - Regression metrics: RMSE, R²
  - Classification metrics: Accuracy, ROC-AUC (binary and multiclass)

- **Intelligent Ranking**
  - Composite scoring with simplicity penalty
  - Configurable lambda penalty (default: 0.15)
  - Formula: z(metric) + λ × (LVs/25 + n_vars/full_vars)
  - Lower scores = better models

- **Output & Reporting**
  - CSV results table with all model runs
  - Markdown reports with top-5 models
  - Detailed configuration and performance metrics

#### CLI
- `spectral-predict` command-line interface
- `--spectra` mode for CSV input
- `--asd-dir` mode for ASD directory input
- `--reference` for target variable mapping
- `--target` for single-target prediction
- `--folds` for CV configuration
- `--lambda-penalty` for complexity penalty tuning
- `--outdir` for output directory configuration
- `--asd-reader` flag (auto/python/rs-prospectr/rs-asdreader)

#### Infrastructure
- Complete test suite (30 tests)
- CI/CD with GitHub Actions
  - Linux and Windows testing
  - Python 3.10, 3.11, 3.12 support
  - Black code formatting checks
  - Flake8 linting
  - Package build validation
- Development dependencies: pytest, black, flake8, build, twine
- Optional dependencies: specdal for binary ASD support

#### Documentation
- Comprehensive README with installation and usage examples
- Inline documentation for all functions
- Type hints for better IDE support
- Example commands for common use cases

### Planned (Future Releases)

#### Binary ASD Readers
- **Native Python reader** (stub in `readers/asd_native.py`)
  - Pure-Python binary ASD parser
  - No external dependencies

- **R Bridge** (stub in `readers/asd_r_bridge.py`)
  - Integration with R's asdreader package
  - Integration with R's prospectr package
  - Requires rpy2 and R installation

#### Future Enhancements
- Interactive mode for target selection
- CSV directory batch processing
- Model persistence and reloading
- Feature selection optimization
- Additional preprocessing methods
- Support for additional file formats

## [Unreleased]

### To Be Added
- SpecDAL integration for binary ASD files
- Native Python binary ASD reader
- R bridge implementation
- Interactive CLI mode
- Model export/import functionality
- Additional spectral file formats (SPC, OPUS, etc.)

---

## Version History

- **0.1.0** (2025-01-27) - Initial release with CSV and ASCII ASD support

[0.1.0]: https://github.com/makapania/dasp/releases/tag/v0.1.0
