# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

Older entries are in [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md) — grep it for historical context; batch 8 on 2026-10-03 moved every entry dated 2026-09-14 to 2026-09-28; batch 7 on 2026-09-15 moved every entry dated before 2026-09-14 plus the full round-by-round PR #79 crash-resume history (condensed here into the 2026-09-15 "PR #79 crash-resume (merged a5f9a70)" entry); batch 6 moved entries dated before 2026-09-01.

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


## 2026-10-02 - Tab 7 Y-transform save contract (R048/R001/R020/R014/R010), branch fix/ytransform-save
- **R048 was masking R001/R020.** Every TTR refit crashed on `pipe.steps` before reaching the TTR save branch, so no
  .dasp with a TTR could exist; a bare `hasattr` guard would have shipped silent corruption (lost prep_pipeline,
  scaler saved twice). Fix both together.
- **Save contract:** Tab 7 always runs Path A, so spectral preprocessing (`prep_pipeline`) is fitted OUTSIDE the TTR on
  the full spectrum; the TTR wraps only the post-subset `[imbalance?, scaler?, model]`. Save a TTR exactly like the
  untransformed case (preprocessor = prep_pipeline; model = scaler+model or bare model) and re-wrap that prediction
  model with `y_transform.replace_fitted_regressor` (copies the fitted TTR, swaps `regressor_`). Never split the
  TTR's inner steps out as the preprocessor.
- **'Box-Cox'.** `.lower().replace('-', '-')` was a no-op, so the combobox value never matched 'boxcox'; also
  `validate()` skipped the y>0 check for it. All entry points now go through `normalize_y_transform_method`.
- **Early stopping + transform:** CV transforms fold y by hand; the final fit is now TTR-wrapped (no ES on the final
  fit either way). If the booster ES agent changes the final fit (e.g. `set_params(model__n_estimators=...)`), the
  TTR needs `regressor__` prefixes.
- **Corrections:** each new `refined_model` gets a fresh `_refined_model_token`; corrections record the token they
  were computed under and `_correction_to_save()` only returns a matching one, regression only. `model_io` drops a
  correction for non-regression at save and ignores one at predict (legacy files).
- **Pre-existing, not fixed:** `_plot_wavelength_importance` applies `refined_preprocessor` (full-spectrum prep) to
  `refined_X_train` (already preprocessed + subset), so the residual-correlation overlay double-preprocesses.
- **Review round 1 (Codex + GLM, MERGE-WITH-FIXES):**
  - Token race: Compute runs on the Tk thread while the refit worker can swap the model. Capture the token BEFORE
    reading `refined_y_*`, keep the result only if the token is unchanged; the worker sets the token to None before
    the model swap and issues the new one only after the CV predictions are stored. Save/Compute are also disabled
    while a refit runs (GA abort paths now re-enable them).
  - Every consumer that rebuilds or inspects the model must handle a TTR: code export (now `YTransformRegressor` in
    the generated script; `_fit_fold` re-enters with transformed train/eval y for early stopping; export refuses
    Y-transform + imbalance), RF tree variance in `predict_with_uncertainty` (inverse-transform each tree), complexity
    curve (clones of `final_pipe`, params `regressor__model__<p>`). SHAP has no TreeExplainer for a TTR and falls
    back to KernelExplainer (works, slower, original units).
  - **Tree-count hook for fix/booster-early-stopping:** set any final-fit parameter on `final_pipe` at the
    "FINAL-FIT PARAMETER HOOK" comment (just after the Y-transform wrap); `_final_param_prefix` is `'regressor__'`
    when it is a TTR, so the booster tree count is `regressor__model__n_estimators`.
  - Saved `y_transform` is now the canonical name (`log`, `boxcox`, `none`); files from before this fix hold display
    names ('Log', 'None'), so readers must normalize.
  - Export parity gotcha: a Tab 7 XGBoost refit fills params missing from `Params` with GUI defaults (subsample 0.8,
    colsample_bytree 0.6, ...) that the code export does not know, so export CV differs unless the row is complete.
    Pre-existing, not Y-transform specific; test rows carry full params.
- **Review round 2 (Codex BLOCK, GLM MERGE):**
  - Disabled Run buttons are NOT a refit guard: the Model Development tab handler treats a disabled Run as
    "uninitialised" and re-enables it, and loading defaults or a Results row re-enables it too. `_run_refined_model`
    now refuses while `_refit_active`; the flag is cleared by `_end_refit(generation)`, queued in the worker's
    `finally` AFTER the run's own result callbacks. Save refuses while a refit is active, reads everything once via
    `_refined_state_snapshot()`, and aborts if the model token changed during the file dialog.
  - Error path: Save/Export stay enabled when the previous model is still complete (`refined_model` set AND token
    set); a failure mid-swap (token None) disables them.
  - Exported `YTransformRegressor` must be `(RegressorMixin, BaseEstimator)` (mixin first) or `is_regressor` is
    False and VotingRegressor rejects it; it also mirrors TTR's (n,)/(n,1) output shape.
  - Complexity grids must contain the fitted value (8-point grids around a base usually miss it).
- **Review round 3 (Codex BLOCK, GLM MERGE-WITH-FIXES):**
  - The refit worker must not write ANY `refined_*` field before success: the one-class path wrote
    `refined_full_wavelengths` before its no-inlier / too-few-folds exits, so a failed run left Save enabled with
    model A plus run B's axis. Both paths now build their fitted state locally and call
    `_publish_refined_state(dict)` once (token None during the setattr loop, fresh token after). `refined_ga_*`
    are run inputs (set by Results-row loading / GA), frozen into `refined_config`, so they stay as they are.
  - If `threading.Thread(...)`/`start()` raises after `_refit_active=True`, nothing ever clears it: the launch is
    wrapped and releases its generation on failure.
  - Export Code joins Save/Compute: disabled during a run, refused at method level, config + data from one snapshot.
    `_update_bias_correction_ui(from_run_completion=True)` is the only call allowed during a run (the completion
    callback runs before `_end_refit` clears the flag).
- **Review round 4 (Codex + GLM MERGE-WITH-FIXES):**
  - The refit result is now ONE frozen `RefinedState` (module level in the GUI) held in `app._refined_state` and
    replaced by a single assignment; `app.refined_<field>` and `app._refined_model_token` are properties onto it
    (assigning one replaces the whole object via `dataclasses.replace`). Unset fields raise AttributeError so the
    many `hasattr(self, 'refined_...')` checks still work. The worker queues publication on the Tk thread
    (`_publish_refined_state_on_tk_thread`), so Tk callbacks never straddle a swap; off-Tk or update()-calling
    consumers (learning-curve worker, SHAP) capture `self._refined_state` once.
  - `RefinedState.training` = the Results row (shallow copy) + autoscale flag captured at worker START; Save
    metadata and Export read params/Deriv/Poly/imbalance/early stopping/autoscale from it, never from the live
    selection. GA genes/config/model type are frozen per run (locals), not re-read at publish.
  - `_refined_state_snapshot()` returns None without a token/model; Save, Export entry and the open dialog's
    `do_export` refuse then. Loading a Results row is refused while a refit runs.
  - Tooling gotcha: passing `\\n` through the agent's Bash heredoc arrived as `\n` (escape collapsed), so string
    anchors containing backslashes silently failed to match; anchor on backslash-free text.
- **Review round 5 (Codex + GLM MERGE-WITH-FIXES):**
  - The refit worker read the live Results row (~24 sites: Params/Preprocess/Window/Deriv, hyperparams,
    one-class, imbalance, early stopping, `optuna_params` at publish) and live Tk vars/data. Now
    `_capture_refit_inputs()` freezes ALL of it on the Tk thread in `_run_refined_model` (or at the top of a
    direct `_run_refined_model_thread()` call) and the worker reads only `run_inputs`; grep confirms no
    `self.selected_model_config` / `self.<tkvar>.get()` / `self.X|y|validation_*` read remains in the worker.
    `_collect_refine_hyperparams` is pre-collected for the widget model and 'PLS' (the only remap target).
  - `_on_result_double_click` assigned `selected_model_config` before the guarded loader refused: guard at the
    very top now (both branches).
  - Save metadata data_type / x_unit / validation_* / inlier fallback come from `snap['training']`.
  - Plot click callbacks (regression scatter, residual, leverage) captured y_true/y_pred but read the CURRENT
    cv_indices/specimen_ids, so a click on A's plot after B published offered B's specimen. Each plot binds
    `plot_state = self._refined_state` for itself and its callbacks.
  - The Export dialog shows and exports the snapshot taken when it opened; `do_export` refuses if the current
    token is no longer that snapshot's.
- **Review round 6 (DeepSeek MERGE-WITH-FIXES):** helpers called FROM the worker count too:
  `_parse_wavelength_spec` read the live `_original_wavelength_order`; it now takes `original_order` (worker passes
  its frozen copy; None or [] = available order). Plot callbacks no longer fall back to the live `self.y.index`
  (no specimen IDs in the run = nothing offered). GA inputs are stored via `root.after` (`_store_ga_inputs`),
  not from the worker. Captured frames are `copy(deep=False)`: shares data under pandas CoW, freezes the object.
- **Merge / follow-up items (not fixed here):**
  - R015: the bundle export ships already-preprocessed `refined_X_train` but its script preprocesses again
    (pre-existing; still true with the Y-transform wrapper).
  - fix/booster-early-stopping: its export round/tree-count selector must look inside `YTransformRegressor`
    (`model.regressor`) in generated code, and inside the TTR (`regressor__model__...`) in the app.

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

## 2026-10-02 - GUI dataset state, review round 5 follow-ups

- Resume reconciliation restores the click-time selection in a `finally`, so an exception part-way (e.g. in
  the identity digest) also leaves exclusions, holdout and the validation status text as they were.
- `_prepare_calibration` aligns X and y by label (equal lengths only; unequal lengths stay a worker error)
  before the digest; the worker's own realignment after it is now a no-op.
- uint64 targets above the int64 range are hashed as uint64 (an int64 cast wrapped them onto negatives);
  smaller unsigned columns still hash like signed ones.
- `rename_duplicate_ids` and the install step decide "repeated" with missing-aware keys: pandas'
  `duplicated()` treats NaN and pd.NA as different IDs.

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

## 2026-10-02 - Booster early stopping replaced by ONE round count from the pooled CV curve (R028/R003/R022/R126)
Branch `fix/booster-early-stopping`. `cv_utils._fit_with_early_stopping` (eval_set = the scored test fold) is deleted.
New primitives in `cv_utils`: `cross_val_boosting_rounds` (fit each fold at max rounds, no eval_set; stage test
predictions; pick one count from pooled RMSECV, or pooled accuracy with pooled log-loss as exact-tie breaker),
`booster_staged_predict`, `select_n_rounds` (`early_stopping_rounds` = patience of the scan), `set_booster_rounds`.
Gotchas worth knowing:
- **Final review (Codex BLOCK / DeepSeek MERGE-WITH-FIXES on 5ec39e5):** (1) the grid final R->k refit fitted
  XGBoost without the balanced class weights CV used (calibration F1 0.000 vs 0.625 on rebuild); it now passes
  `compute_sample_weight('balanced', y)` and the weighted-grid test compares calibration metrics with the
  validation rebuild. (2) #91 left integer labels {1,2,5} un-encoded in the grid, and XGBoost refuses them:
  `_run_single_config` now fits codes (`classification_fit_labels(..., 'XGBoost')`) and decodes y_test/y_pred,
  calibration predictions and probability class order. (3) With a TTR, `cross_validate_with_early_stopping`
  train scores used the fold model's transformed-space predictions; `BoostingRoundsCV.fold_target_transformers`
  now keeps each fold's transformer and `_predict_with_fold_model` inverse-transforms. (5) `_fit_fold_full_rounds`
  applies the target transformer BEFORE in-fold samplers, as the final TransformedTargetRegressor does.
  Still open (pre-existing, not changed): regression sample weighters (`imbalance` step with `sample_weight_`)
  weight the CV folds, but neither the grid final refit nor the validation rebuild applies them.
- **Merge of main 1ca2de3 (#92-#95):** #94's thread budget wraps the booster folds: the grid runs `_run_single_fold`
  on `fold_pipe` (thread-capped copy of the sanitized pipe) with `y_fit` (XGBoost codes); the final refit keeps
  `pipe`. Bayesian round selection uses the capped `cv_model`; the plan goes sequential whenever round selection or
  per-fold balanced weights (`_cv_balanced_param`, which replaces #94's `_cv_fit_params`) force the manual loop.
- **Exported helper sources are user-visible text.** The export copies cv_utils functions with
  `inspect.getsource`, docstrings included: `_final_estimator`'s docstring named `YTransformRegressor`, so every
  export "contained" a Y-transform (`test_y_transform_consumers::test_export_without_transform_is_unchanged`).
  Keep export-only class names out of copied helpers' docstrings.
- **Tab 7 XGBoost on {1,2,5} crashed** (also on main since #91: integer labels fitted as given, XGBoost refuses
  them; the refit ends with no model). Tab 7 now label-encodes when `classification_fit_labels(y, model_name)`
  says `xgb_codes`; the codes equal the grid's, and the existing encoder plumbing decodes CV predictions,
  probability columns (sorted-label order) and the saved model's predictions. GUI test pins accuracy = grid row's.
- **Merge of main 6f63216 (PR #90 export helpers, PR #91 label policy / pooled metrics):** study names put
  `|labels=raw1` AFTER `|boost_rounds=`; the booster old-scoring notice matches the previous base both with and
  without the `|labels=` segment (a post-#91, pre-fix LightGBM/CatBoost study on raw labels would otherwise
  restart silently). #91's `label_policy_changed` notice goes through the same `_notes` loop as
  `booster_scoring_changed`, so it carries `resume_declined` and the GUI keeps the run record. `search._fold_metrics`
  (shared by plain folds and the pooled-round booster folds) now uses `scoring.classification_metrics` with
  probabilities aligned to the FULL class list and returns `y_proba` for #91's pooled AUC/log-loss.
- **Truncation identity holds** for XGBoost (`iteration_range`), LightGBM (`num_iteration`) and CatBoost (`ntree_end`),
  bagging included: a refit with `n_estimators=k` predicts exactly what the max-round fold model predicts at round k
  (diff 0.0 / 2e-16). So CV at the selected count = CV of the refit, and re-running the selection with max=k returns k
  (idempotent). Tab 7 and exports therefore reproduce a row whose Params already carry k with no special casing.
- **CatBoost `staged_predict` costs ~1-3 ms per round** (0.9 s for 300 rounds on 16 rows). `_catboost_staged_raw` uses
  `calc_leaf_indexes` + `get_leaf_values` + cumsum (exact, ~1 ms total) and falls back to `staged_predict` if the last
  round does not match `predict`/`predict_proba` (unusual losses / tree layouts).
- **LightGBM may build fewer trees than `n_estimators`** (stops when no split helps); staged arrays are padded with the
  last round, which is what a refit with more rounds would also do.
- **Pure accuracy as the classification curve picks round 1** on plateaus (strict improvement keeps the earliest round),
  giving nearly untrained boosters. Hence the pooled log-loss tie-break on exact accuracy ties.
- **Stratified splits move when labels move**: a "change only the test-fold labels" test must replay fixed splits.
- Bayesian study names gain `|boost_rounds=pooled_cv_curve_v1` only for booster studies with round selection on, so
  old biased trials never resume beside corrected ones; every other study name (and the T51 pinned names) is unchanged.
- Bayesian Params carry `model__n_estimators` (pipeline-prefixed), grid Params `n_estimators`/`iterations`.
- Export copies the cv_utils primitives' source with `inspect.getsource`, so in-app and exported selection cannot drift;
  the export runs a pre-pass `_choose_boosting_rounds` and then a plain CV loop at the selected count.
- Example data (BoneCollagen, snv, 5-fold, 200 rounds, patience 40): LightGBM RMSEcv 3.900 -> 3.949 (k=93),
  XGBoost 3.960 -> 4.062 (k=177). The bias is modest on real signal and large on noise (review repro: R2cv +0.02 vs -0.54).
- **Review round 1 (Codex BLOCK, GLM merge-with-fixes) gotchas:**
  - Balanced class weights computed from ALL of y and then sliced per fold (Bayesian and NSGA-II XGBoost
    class_weight paths) also leak test labels into a fold's fit. CV now uses `balanced_sample_weight=True` /
    `cross_val_predict_pooled(balanced_weight_param=...)`; all-of-y weights only for the full-data refit.
  - Prefix selection is invalid for XGBoost gblinear (iteration_range ignored: flat curve), XGBoost/LightGBM DART
    and CatBoost model_shrink_rate/posterior_sampling (later rounds rescale earlier trees).
    `round_selection_unsupported_reason` -> fitted at the configured count, row records no selection.
  - CatBoost with learning_rate=None picks its rate from the round count: `learning_rate_` of the first fold is pinned
    for the other folds and the refit (`BoostingRoundsCV.pinned_params`, `apply_round_selection`); exact (diff 0.0).
  - LightGBM `num_iterations` aliases override `n_estimators`; setting an alias to None crashes LightGBM, so
    `set_booster_rounds` sets every present alias to the same count. Early-stopping aliases CAN be set to None.
  - CatBoost Poisson/Tweedie: `predict` exponentiates, `staged_predict` defaults to raw; staging is accepted only when
    its last round equals native predict (raw, exp(raw), or staged_predict 'Exponent').
  - Repeated-CV vote ties: reported predictions use `Counter.most_common` (first-voted label wins ties); the selection
    curve now reproduces that exactly (`first_vote` array) - test checks curve[k] == accuracy of reported preds at k.
  - Changing a booster study's identity hid the old study from the env/legacy notice (it searched the new base only).
    The previous-policy base is recomputed (`previous_policy_study_base` study attr) and reported like an env change.
  - Tab 7's validation-curve diagnostic refits the model ~27 times on purpose; a fit-count test must exclude it.
- **Review round 2 (Codex BLOCK; GLM MERGE) - redesign of the final booster model:**
  - Pinning CatBoost's automatic learning rate from fold 1 leaked labels across folds when a resampler made fold
    sizes label-dependent, and missed other round-dependent defaults (leaf_estimation_iterations changes with
    iterations). Replaced by: every fold chooses its own defaults; the final model is the scored configuration
    fitted on all data at the maximum R and TRUNCATED to k (`cv_utils.truncate_booster`). Truncation that pickles:
    XGBoost `est._Booster = booster[:k]`; LightGBM `est._Booster = Booster(model_str=model_to_string(num_iteration=k))`;
    CatBoost `shrink(ntree_end=k)` - CatBoost refuses set_params on a fitted model, so the count is written to
    `_init_params`. All exact vs the R-fit's staged predictions at k (diff 0.0), also after pickling.
  - Params carry the selected count k; rows add `n_estimators_fit` (R) and `round_selection_truncated`. Params stay
    pure estimator params on purpose: Tab 7 and other consumers `set_params` every bare key, so flag keys there would
    break them. Rebuilds use `cv_utils.round_truncation_from_row` (validation rebuild, Tab 7, export). Ensembles
    (since round 3, faf0dba) wrap such boosters in `RoundTruncatedRegressor/Classifier`, so every ensemble fit and
    refit is also "fit at R, truncate to k".
  - A declined resume (old booster scoring, other environment/data) no longer completes the saved run when the
    replacement finishes: `_resume_not_continued_run_id` keeps the record (PROJECT_STATUS §1 binding decision).
  - NSGA-II XGBoost Params contained `'missing': nan` (not literal_eval-able), so the selected count was silently
    dropped; NaN defaults are now omitted and `_with_selected_rounds` raises instead of ignoring.
  - XGBoost supports dropout under gbtree (rate_drop / one_drop): prefix-unsafe like DART.
  - The export's regression final-model template prints CCC with `_lins_ccc`, which only the CV section defines, so
    a regression export without CV raises NameError (pre-existing, not fixed here).
- **Review round 3 (Codex BLOCK; GLM merge-with-fixes) gotchas:**
  - CatBoost `get_params` reports only the arguments that were set, so a row's Params omit automatic defaults.
    Rebuilding from `get_model("CatBoost")` (or the export's DEFAULT_PARAMS) and then `set_params` silently injects
    `learning_rate=0.1` etc. CatBoost rows are now built from their stored params alone
    (`models.catboost_from_row_params`): validation rebuild, Tab 7, export.
  - CatBoost drops None-valued keys when reconstructing, and `od_type=None` fails at fit: eval-only keys are now
    REMOVED from `_init_params` rather than nulled, or a later sklearn clone raises KeyError.
  - The export's final-model template computes calibration metrics right after `model.fit(...)`: truncation must be
    spliced in immediately after that line, not appended after the section.
  - `_resume_not_continued_run_id` must be scoped to one launch attempt (reset at launch and on every completion path).
  - all_vars is written with %g (R031, other branch): tests that round-trip a row through the validation rebuild need
    integer wavelengths, or columns are silently dropped.
- **Round 4 (DeepSeek, final) + merge of main f1fbf79 (PR #89 ytransform-save):** #89 wraps the Tab 7 final model in
  a TransformedTargetRegressor even when CV transformed each fold by hand, so the saved booster now trains on the
  transformed target; truncation runs right after that fit (`truncate_booster` unwraps TTR `regressor_`). The
  export's `YTransformRegressor` has no `transformer` attribute, so `_final_estimator` now unwraps any `regressor`
  (and `booster_max_rounds` / `set_booster_rounds` / `uses_round_selection` resolve through it);
  `cross_val_boosting_rounds` accepts a TTR and derives the matching target transformer. #89's leaky export
  `_fit_fold` / `_fit_with(eval_set)` path was dropped in the merge; the export reads staged predictions through the
  fold wrapper's `transformer_`.


## 2026-10-02 - GUI dataset state, round 7 follow-ups

- The combined CSV/Excel readers decide "repeated IDs" with the same missing-aware check as
  `rename_duplicate_ids` (`io._has_repeated_ids`), not `duplicated()`.
- Object-dtype targets with non-negative ints above int64 hash like the uint64 column they equal.
- The worker's X/y realignment after preparation is gone (it was dead after round 6);
  `_prepare_calibration` records `n_realigned` and the worker logs the same warning from it.

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
## 2026-10-02 - fix/preexisting-test-export: export metric helpers + multiclass fake Thread

- **Export NameErrors came from helpers defined inside the CV block.** `_lins_ccc` (regression) and
  `_one_class_metrics` (one-class) lived in the CV templates, yet the final-model block calls them too, so any
  script exported with `include_cross_validation=False` died at calibration metrics. One-class had the same bug
  as regression. Both now live in `templates/validation.py` (`LINS_CCC_HELPER`, `ONE_CLASS_METRICS_HELPER`,
  `get_metric_helpers_template`); `CodeGenerator` emits them once after the model block, in the script and in
  the notebook's model/CV cell. The imbalance-aware regression CV block (`_render_cross_validation_with_imbalance`)
  never set `ccc`, which the shared metrics block prints: a third NameError, fixed with one line.
  Classification exports have no shared helper and were fine. Separate, not fixed: imbalance regression with
  PLS fails because `PLSRegression.fit` takes no `sample_weight`.
- **`test_run_analysis_accepts_multiclass_engine_selection`: the test was wrong, not the code.** Its fake
  `threading.Thread` took only `(target, args, daemon)`; `_run_analysis` correctly passes `kwargs=`. The fake
  now mirrors Thread's signature and forwards args and kwargs.
- **Round 2 (Codex probes + GLM review of d4f608a).** A stale `imbalance_method` on a one-class config sent export
  down the *regression* imbalance CV/final-model path (`_lins_ccc` NameError; it would also have fitted the
  one-class model with y and sample weights). Imbalance does not apply to one-class (backend fits inliers only, GUI
  hides the card), so `CodeGenerator.__init__` now drops it for `one_class`. `generate_notebook` ignored
  `include_cross_validation` and always emitted the CV cell; it now honours the flag as `generate_script` does.
  With CV off, `include_visualization` emitted plots that read `y_pred_cv` / `all_y_true_arr`;
  `get_visualization_code(include_cv_plots=False)` now keeps only the spectra plot. The imbalance-regression final
  model now prints calibration RMSE/R2/CCC like the plain path, and imbalance classification no longer prints its CV
  metrics twice. Noted, not fixed: the regression pred-vs-actual title prints a literal `{rmse:.4f}` (the viz
  templates are never `.format`ed but use `{{ }}`), and the one-class "decision score" histogram plots +1/-1
  labels, not scores.
- **Round 3: the two plot bugs.** `templates/visualization.py` strings are emitted verbatim (never `.format`ed), so
  `{{rmse:.4f}}` inside the generated f-string printed literal braces; the template now uses single braces, like the
  spectra plot. The one-class histogram plotted `y_pred_cv` (the +1/-1 labels). The one-class CV template now also
  builds `cv_scores`, the out-of-fold `decision_function` (or `score_samples`) values aligned with
  `all_y_true_arr` (averaged per sample under Repeated K-Fold), plus `cv_scores_are_decision`. The histogram plots
  those and marks the threshold at score = 0, the same rule the CV block uses to predict. The executed-export test
  reads every figure title back through matplotlib and fails on any `{`/`}`, and checks the one-class scores have
  more than two distinct values.
- **Round 4 (DeepSeek review): tests now check what they claim.** Every executed-export case asserts
  `Cross-validation Results` appears once with CV on and never with CV off (a classification+imbalance case fails on
  d4f608a, which printed it twice). `test_exported_regression_metrics_match_independent_computation` recomputes CV
  and calibration RMSE/R2/MAE/CCC with sklearn + `scoring.lins_ccc` on the same data and KFold splits and matches
  the printed values to 2e-4 (also true on d4f608a, so CV numbers are unchanged). The one-class probe asserts
  `len(cv_scores) == len(all_y_true_arr) == len(y_pred_cv)` and that score >= 0 agrees with the reported label
  outside the 25% of samples nearest the threshold. Added OneClassSVM (scaling branch) and regression-imbalance
  notebook cases. Gotcha: under Repeated K-Fold `cv_scores` are per-sample means, `y_pred_cv` a majority vote, so
  they can disagree near 0 (now said in the template comments).

## 2026-10-02 - QW1 thread budget + QW10 test split (branch perf/thread-budget)
- **`n_jobs=1` inside the fold pool is NOT the right rule on a many-core box.** Warm loky pool, 5 folds, 24 cores:
  XGBoost 49x2151 was 617 ms with `n_jobs=1` vs 450 ms with 4 threads/fit (24//5) and 526 ms with the old nested
  `-1`; at 100x2151 `n_jobs=1` (2.1 s) was *slower* than the old nesting (1.4 s) because 5 single-threaded fits leave
  19 cores idle. `parallel_policy.plan_cv` therefore splits the cores: pool = min(n_splits, physical cores), each fit
  gets cores // pool (= 1 once folds >= cores, e.g. LOO). Never multiplied.
- **Tiny jobs:** a warm pool already beats a serial loop at ~20k cells (5 folds x 20 x 200); break-even ~5-10k.
  `TINY_JOB_CELLS = 10_000`, below which folds run serially with single-threaded fits. Serial fits that keep
  `n_jobs=-1` lose to `n_jobs=1` at every size below ~50k cells (thread wake-up).
- **Only the fold clone is capped** (`limit_estimator_threads` clones). The refit pipe/`model` keeps `n_jobs=-1`, so
  result-row Params, Bayesian fingerprints and study hashes are unchanged (asserted in tests/test_parallel_policy.py).
- **Bayesian:** `cross_val_predict_pooled` honours `n_jobs` only on its sklearn-delegate path; repeated CV,
  fit_params (class_weight sample weights) and early stopping run a serial loop, so the plan collapses to serial with
  untouched (all-core) models there. sklearn-owned pools need `joblib.parallel_config(backend=...)`
  (`CVPlan.backend_context()`), otherwise frozen bundles would get loky. Frozen Bayesian/diagnostic curves now use a
  threading pool instead of a serial loop (same backend grid search already used frozen).
- **`threadpool_limits()` costs ~8 ms per call** (it rescans loaded DLLs); one-class CV calls it per config. The policy
  caches a `ThreadpoolController` (0.006 ms per limit). OMP via env (`OMP_NUM_THREADS`) must be set before numpy loads
  and never caps an explicit booster `n_jobs`.
- **GUI session reset:** deepcopy of all launch attributes fails (`cannot pickle '_tkinter.tkapp'` - fonts, styles,
  sidebar hold Tk handles), so the snapshot keeps only plain-data launch attributes; objects are judged GUI-bound by
  their *direct* attributes only (recursing reaches the app itself via bound methods/args and marks everything GUI).
  The reset also drops data attributes created since launch (e.g. `X_before_contam_correction`), cancels pending
  `after` callbacks (a fresh app has none), removes test-added traces and Toplevels, and re-applies Tk values until
  traces stop rewriting them.
- **Order-dependent GUI tests found by the reset** (each failed when run alone on main f6a2287): 
  `test_multiclass_gui::test_run_analysis_accepts_multiclass_engine_selection` (its fake Thread lacked `kwargs=`), and
  `test_resume_round9::test_e2e_one_failing_model_keeps_run_resumable` + `test_resume_round11::
  test_settings_changed_mid_run_do_not_reach_later_models` (`_select_models` unticked only PLS/Ridge; the launch tier
  also ticks ElasticNet). Fixed in the tests. Default GUI selection now passes forward and reversed.
- **Suite timings (24-thread box at 100% from ~10 concurrent agents; ratios only):** non-GUI on main f6a2287 (no
  addopts, all 3522) 21.0 min; branch with `-o addopts=""` (3592 incl. 70 new) 17.7 min; branch default selection
  16.7 min. GUI default selection (253) ~80 s on both main and branch. So most of QW10's time saving must come from
  the 34 comprehensive GUI tests (not timed here; `test_xgboost_via_gui` alone ~60 min on CI). The biggest remaining
  default-run costs are not marked slow: `test_wavelength_filtering_integration.py::TestScenario8Consistency` (2
  tests, 130 s) and five 24 s `test_t41_auto_rerun_preserves_study.py` tests.
- **Visible "ASP ... (Not Responding)" windows during GUI tests:** withdrawing the root before
  `SpectralPredictApp(root)` is undone by the app's own `root.state('zoomed')` in `__init__`; dialogs also call
  `deiconify()` and every new Toplevel maps on creation. `tests/gui/conftest.py` now has a session autouse fixture
  (off with `--visible`) that makes `Wm.state('normal'/'zoomed'/'iconic')` and `deiconify` no-ops, withdraws each new
  Toplevel, tolerates `grab_set` on a withdrawn dialog, and re-withdraws after app startup and each reset. Verified by
  polling MainWindowTitle of the pytest process: main showed the window, the branch showed none over the full default
  GUI run (253 passed).
- **Review round 1 fixes (Codex + GLM, both MERGE-WITH-FIXES):** (1) Black running on 3.14 rewrote
  `except (A, B, C):` into the 3.14-only `except A, B, C:` - breaks the 3.12 rollback build. Run Black with
  `--target-version py312` on touched files. (2) OpenMP/BLAS caps are process-wide (vcomp: a cap set in one thread is
  seen by already-running threads), so overlapping `threadpoolctl` limit contexts from two threads left OpenMP stuck at
  1; `native_thread_limit` is now lock + refcount, originals restored at depth 0. (3) Threading-backend pools (frozen)
  share one BLAS pool: `CVPlan.backend_context()` caps BLAS at the per-fit budget (loky workers already get
  cores//workers via joblib's worker env). (4) CatBoost's predict and post-fit feature importance ignore the
  constructor `thread_count` (default -1 = all cores): CatBoost folds run serially in threading pools. (5)
  `root.after_cancel(id)` on a callback scheduled by a child widget deletes the child's Tcl command; the child's
  destroy then raises "can't delete Tcl command" (reproduced). Cancel the raw timer with
  `root.tk.call('after','cancel',id)`. (6) Caller-sized pools (`n_jobs=-1` = logical CPUs) are capped at physical
  cores (`pool_workers`). Nightly Linux leg added for the 52 non-GUI slow tests.
- **Review round 2 (GLM + DeepSeek):** threadpoolctl's `restore_original_limits()` restores EVERY library the
  controller holds, not just the user_api it limited - a BLAS context's exit un-capped a live OpenMP context and a BLAS
  cap leaked past all exits (reproduced by both reviewers). `native_thread_limit` now limits/restores through
  `controller.select(user_api=...)` and keeps a multiset of open limits per API, re-applying the strictest open one on
  every exit (so a stricter inner context no longer pins the outer at its value). GA candidate pools and the SPA seed
  loop now go through the policy (`task_pool_plan`; SPA = physical cores, max 8, BLAS capped; SPA output verified
  bit-identical on example data). Notes: `plan_cv(requested_n_jobs=0)` now raises; a single non-tiny split keeps the
  estimator's own n_jobs, so LightGBM/XGBoost may differ in the last bits from a 1-thread fit; the GUI fixture's raw
  `after cancel` leaves one Tcl command per cancelled callback registered until its widget dies.
- **Review round 3 (DeepSeek):** (1) a policy test that pins `physical_cores` must also pin
  `joblib.effective_n_jobs` - `pool_workers` reads the real logical count (4 on CI runners). (2) VotingRegressor /
  StackingRegressor keep their models in list-valued `estimators` params, which a "values with get_params" walk skips,
  so a wrapped CatBoost kept `thread_count=None`. `_set_threads` and `contains_catboost` now share one walker
  (`_sub_estimators` / `_walk_estimators`) that also reads `(name, estimator)` lists, tuples and dicts, with an
  id-visited set instead of a depth cap. (3) `native_thread_limit` rolls back its bookkeeping if applying the cap
  fails on enter, and rejects non-int / < 1 thread counts with ValueError.

## 2026-10-02 - QW4 holdout direction (`fix/holdout-direction`)

- **Root cause.** The GUI's `_validation_kennard_stone` / `_validation_spxy` returned the samples KS/SPXY
  pick first as the *holdout*. KS/SPXY pick the representative boundary samples, so the extremes went to
  validation and the model had to extrapolate. Now `sample_selection.split_calibration_holdout(X, n_holdout,
  method, y)` picks the `N - n_holdout` **calibration** samples and the rest validate; the GUI and the test
  harness (`tests/gui/harness.py::apply_spxy_holdout`, same bug) route through it.
- **SPXY** (Galvão 2005) now divides the X- and y-distance *matrices* by their maxima and adds them
  (`spxy_distance_matrix`). Both the GUI and backend versions used to min-max scale every spectral column,
  which gives a near-constant noise column the weight of a real band. **DUPLEX** (Snee 1977) now runs two
  interleaved max-min selections (cal seed = farthest pair, val seed = farthest remaining pair, then each set
  adds the sample farthest from its *own* members); the old version alternated one KS order, which sent an
  extreme to validation. DUPLEX is a new GUI option.
- **Kennard-Stone order is unchanged** (checked against the old implementation on 200 random cases, ties
  included), so calibration-transfer standards (`calibration_transfer`, GUI CT tab) and `model_io`
  representative sets are identical. CT semantics untouched: there the KS pick *is* the wanted set.
- **Saved holdouts are stable by construction.** Every persisted path stores holdout IDs, never the algorithm
  output: run_state `validation_indices` / `calibration_rows` (resume restores those labels in
  `_reconcile_resume_validation_split`), the refined-model config, the data-viewer revert state. Nothing
  re-runs a selector on load. The `run_gui_settings` comment that said the algorithm name was enough to
  rebuild a deterministic split was wrong and is rewritten. Test: `tests/gui/test_holdout_direction_gui.py`
  restores a pre-QW4 holdout exactly with every selector patched to raise.
- **Example numbers** (bundled bone collagen, N=49, PLS, LV by 5-fold CV on the calibration set, raw spectra;
  scratch script, not pinned): 9 held out (the GUI's 20%): KS old 3.57 (4 holdout samples outside the cal y
  range) vs new 1.30 (0); SPXY old 4.63 (2) vs new 1.62 (0); DUPLEX 2.06 (1); random splits mean 2.97 (sd
  1.05, 200 splits). 10 held out: KS 3.35 vs 1.29, SPXY 4.38 vs 1.56, DUPLEX 1.88, random 3.02. SNV gives
  the same picture (KS 4.69 vs 1.22, SPXY 5.26 vs 2.41). Note the corrected KS/SPXY RMSEP is *below* the
  random-split mean: an interior holdout is the easy case, which the UserGuide now says.
- **Not done (optional per the cross-check):** distances on preprocessed spectra / PCA scores (still raw X as
  loaded), stratified KS, and group-aware selection (F1).
- **GLM 5.3 review LOWs (fixed).** A one-sample DUPLEX set has no seed pair: a one-sample holdout now takes the
  sample farthest from the calibration seed (was: the lower end of the farthest remaining pair). KS tie rule
  (lowest row index) is pinned by a duplicated-rows test and a randomized check against a slow literal max-min
  reference in `tests/test_holdout_direction.py`.

## 2026-10-03 - Wave 1/2 review workflow: reviewer quotas, opencode gotchas, multi-agent merge lessons

Session of 2026-10-02/03 merged PRs #83-#95 (11 Wave 1/2 branches + 2 small fix branches) with Opus agents per
branch and external reviewers per round. Non-obvious lessons:
- **Reviewer quotas run out mid-wave.** Codex (gpt-6-astra) hit its limit (reset 2026-10-09); GLM 5.3 hit the z.ai
  5-hour cap and the opencode-go budget. User rule since: **Codex only for major checks**; routine confirms go to
  GLM 5.3 / DeepSeek (`deepseek/deepseek-flash`; Pro only when the user says "pro").
- **opencode read-only prompts must say**: no shell redirection (`>`, `tee`, `/tmp`) — GLM wrote `%TEMP%/diff.txt`
  then aborted when the readonly agent refused to read outside the repo; no reads outside the repo (DeepSeek died
  reading CPython's stdlib — use `python -c "import inspect; ..."`); "if a tool call is rejected, skip and continue;
  always end with a VERDICT". Paste prior findings into the prompt instead of pointing at scratchpad files.
- **Reviewers read code via `git show <sha>:path`** or existing agent worktrees; the main checkout may be on any
  branch. Line numbers they cite refer to the reviewed commit.
- **Shared scratchpad clobbering:** one agent's commit-message file overwrote another's (`git commit -F` from a
  common name). Use unique file names per agent.
- **Merge order matters for GUI branches.** Each merge into main forces the remaining branches to re-merge
  `spectral_predict_gui_optimized.py`; docs (PROJECT_STATUS/SESSION_LOG/CHANGELOG) conflict almost every time —
  resolve by keeping both sides (SESSION_LOG/CHANGELOG) or taking main's PROJECT_STATUS and re-adding the branch's
  item. Re-run tests after every main merge: auto-merged GUI code still broke (e.g. #89's export helpers that
  early-stopped on the held-out fold re-entered via merge and had to be dropped on fix/booster-early-stopping).
- **GUI test isolation:** the session-scoped app leaks state between files. #89's test replaced `threading.Thread`
  process-wide (loky queue feeder never started → hang) and inserted fixed Treeview iids; #90's fake Thread lacked
  `is_alive` and stayed as `app.analysis_thread`. Fixed in #93; `gui_app` now resets `analysis_thread`/`_refit_active`.
  Mixing `tests/gui` files after root-level files in one pytest call can give "fixture 'gui_app' not found" — run GUI
  and non-GUI suites in separate invocations.
- **Open follow-ups recorded elsewhere:** CSV/reference/ASD decimal-comma misreads (PROJECT_STATUS Follow-Ups);
  PLS + regression imbalance `sample_weight`; regression sample weighters weight CV folds but not the final refit;
  QW4 distance on preprocessed spectra/PCA scores; DeepSeek LOWs on #86 (7% vs 6.5% text; skew hint first direction).

