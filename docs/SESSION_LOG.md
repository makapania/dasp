# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

## 2026-04-10 — One-Class Prediction Disaster: Order-of-Operations Bug (Claude Opus 4.6)

### Bug: Every specimen labeled "Outlier" at predict time, including training data

**Symptom reported:** User trained one-class models via Bayesian optimization, loaded top results into Model Development, ran refinement, saved models, then predicted on novel specimens. *Every* specimen — including specimens that were literally in the training set — was labeled "Outlier" in the main Prediction Results table. All one-class model families affected (OneClassSVM/IF/LOF/EllipticEnvelope/PCA-SIMCA). Multi-class models on the same data worked correctly.

**Root cause:** Order-of-operations mismatch between training and predict. The one-class early-exit branch in `_run_refined_model_thread` (`spectral_predict_gui_optimized.py:34260-34283`) applies `prep_pipeline_oc.fit_transform(X_full)` to the **full** spectrum *then* subsets to selected wavelengths:

```python
X_full_preprocessed = prep_pipeline_oc.fit_transform(X_full)   # full-spectrum fit
X_work = X_full_preprocessed[:, wavelength_indices]            # subset AFTER
```

The `StandardScaler` inside `run_one_class_cv` is then fit on `X_work`, so it expects data in the (preprocess-then-subset) feature space. BUT `self.refined_config` at line 34342 hardcodes `'use_full_spectrum_preprocessing': False`, and `self.refined_full_wavelengths` is never assigned in the one-class branch. These propagate into model metadata at lines 35726-35727. At predict time, `model_io.py:640-677` sees both flags False/None and takes Mode B (subset-first → preprocess), which for SNV / SG derivatives / baseline correction produces mathematically *different* feature values than training. The scaler then receives distributionally alien inputs, `model.predict()` returns -1 across the board, including on samples the model was trained on.

**Why the 2026-04-10 "scaler persistence" fix did not catch this:** that fix ensured `cal_scaler` / `cal_pca_reducer` / `oc_score_stats` were persisted for grid-search results. Orthogonal to this bug. The scaler IS being saved correctly; it is just receiving inputs in the wrong feature space at predict time.

**Why the multi-class path works on the same data:** `spectral_predict_gui_optimized.py:33815` sets `use_full_spectrum_preprocessing = True` for regression/classification, and line 35515 assigns `self.refined_full_wavelengths`. The one-class early-exit branch was a copy-paste that missed both lines. This is the Nth instance of the pattern flagged in SESSION_LOG `2026-04-09` ("One-class early returns are the #1 bug pattern").

**Severity note on Savitzky–Golay:** for `deriv=1` or `deriv=2` preprocessing the subset-first path is not merely slightly different — the SG window at subset boundaries lands on indices outside the subset, producing artificial discontinuities and derivative values off by orders of magnitude. This is the dominant contributor when SG is enabled (flagged by qwen3-235B review).

**Fix:** Two lines in `_run_refined_model_thread`:
- Assign `self.refined_full_wavelengths = list(original_wavelengths)` right after `X_work` is built.
- Flip `'use_full_spectrum_preprocessing'` from `False` to `True` in `self.refined_config`.

The metadata save path at `:35726-35727` already reads these correctly — no save-side change needed.

### Secondary bug: Uncertainty-tab display always shows "Outlier"

`spectral_predict_gui_optimized.py:38245`:

```python
predictions = self.predictions_df[model_name].values   # STRINGS at this point
pred_label = "Inlier" if predictions[i] == 1 else "Outlier"   # str == int → False
```

Line 37770-37780 converts numeric +1/-1 to `"Inlier ({label})"` / `"Outlier"` strings before storing in `predictions_df`. The Uncertainty display then compares those strings to the integer `1`, which is unconditionally False, so every row shows "Outlier" regardless of the real prediction. Not what the user was seeing in *this* report (they confirmed they were on the Results tab), but would silently contradict the Results tab once the primary bug is fixed. Fix: replace with `isinstance(raw, str)` branch that accepts already-stringified predictions.

### Latent bug: Bayesian path never stores cal_scaler/cal_pca_reducer/oc_score_stats on trials

`src/spectral_predict/unified_bayesian.py:1009-1037` stores metrics and preprocessing config but never calls `trial.set_user_attr('cal_scaler', ...)`, `'cal_pca_reducer'`, or `'oc_score_stats'`. The 2026-04-10 fix only covered `search.py` (grid search). Not biting the user *today* because all saves flow through refinement (which re-runs CV), but a latent landmine for any future "save direct from results" code path. Close it, but audit `convert_study_to_dataframe` consumers first — storing sklearn objects as DataFrame columns could break `to_csv` / Treeview inserts / sort handlers.

### Architecture insight: test suite does not exercise save→load→predict-on-novel-specimens for one-class with preprocessing

All 44 existing contamination tests passed. None of them round-trip through `save_model` → `load_model` → `predict_with_model` against training data with an actual preprocessing pipeline (SNV/SG). This is exactly why the bug was invisible and PROJECT_STATUS listed prediction as working. Added `test_one_class_save_load_predict_roundtrip_with_preprocessing` to cover it.

### Architecture insight: `use_full_spectrum_preprocessing=False` is effectively dead for regression/classification

Searching the codebase, the flag is only ever set to `True` in the non-one-class paths (lines 28371, 33815). It is only ever set to `False` in the one-class early-exit branch (34342 — the bug) and in a couple of load/default contexts. If no path actually wants Mode B anymore, Step 1 here is a point fix but the underlying dichotomy in `model_io.py:640-677` could be simplified in a future cleanup. Out of scope for this bugfix.

---

## 2026-04-10 — Tooltip Gap Fixes for Metric Column Headers (Claude Opus 4.6)

### Gap: Results Treeview had no tooltip for Sensitivity, Sensitivitycv, AUC, AUCcv columns

**Root cause:** `TreeviewHeaderTooltip._get_tooltip_for_column()` at `spectral_predict_gui_optimized.py:354` does exact-match lookup against `TOOLTIP_CONTENT['metrics']`. The one-class Bayesian/grid result dict emits keys `Sensitivity`/`Sensitivitycv`/`AUC`/`AUCcv` (see `:34324-34330`), but those keys were never added to the tooltip dict — so hover silently returned `None`. Same gap existed for classification validation columns `val_Accuracy`/`val_Precision`/`val_Recall`/`val_F1` (populated at `:35832-35833`) and regression validation columns `RMSEP`/`R2pred`.

### Gotcha: `R²pred` superscript vs. `R2pred` ASCII

The main results Treeview uses the **ASCII** key `R2pred` (defined authoritatively in `src/spectral_predict/search.py:416, 647`). The separate ensemble comparison Treeview in `_populate_ensemble_comparison_table` around `:27977` uses the **superscript** key `R²pred`. The tooltip dict needs only `R2pred` because the ensemble comparison Treeview has no `TreeviewHeaderTooltip` attached at all — that's a separate pre-existing gap for a future fix.

### Gotcha: Some existing metric tooltips were too jargony for the target audience

`ROC_AUC`, `Kappa`, `MCC`, `BER`, `LogLoss` assumed ML background (e.g., "agreement beyond chance", "Balances all confusion matrix categories", "Standard metric in PLS-DA"). Rewrote all five with concrete, audience-appropriate wording plus "rule of thumb" numeric guidance where available. `ROC_AUCcv` was a one-liner placeholder — expanded to match.

### Gotcha: `LogLoss` is not a calibration metric

Initial rewrite said "measures how well-calibrated the model's probability estimates are". Code review flagged this: LogLoss captures both calibration AND discrimination. Changed to "measures how good the model's probability estimates are" to avoid the technically-incorrect calibration framing.

### `TOOLTIP_CONTENT` requires GUI restart

The dict is populated at import time (`:1253`), so tooltip edits only take effect after a full Python/GUI restart. Not a hot-reloadable change.

---

## 2026-04-10 — One-Class External Validation Parity (Claude Opus 4.6)

### Bug: One-class external validation only populated on top 10 models regardless of validation_top_n

**Root cause:** `spectral_predict_gui_optimized.py:25133` hardcoded `for idx in range(min(10, len(results_df))):` in the one-class validation block — ignored `self.validation_top_n` (default 700 that the classification/regression path honors). User running Bayesian one-class with 700-trial validation saw only the first 10 rows populated; rest were NaN.

### Bug: One-class external validation had 3 metrics (BalancedAcc/Sensitivity/Specificity) while cal/CV had 7

**Root cause:** Same inline GUI block hand-computed only `balanced_accuracy_score`/`recall_score` (for pos_label=-1 and +1). Never called `decision_function` to get scores for AUC, and never computed Precision/F1/Accuracy. Meanwhile `run_one_class_cv()` stored 7 metrics for cal AND CV via `one_class_metrics()` — so the Results tab showed the column mismatch.

**Fix:** Added `compute_validation_metrics_for_top_one_class_models()` in `contamination.py` — mirrors `compute_validation_metrics_for_top_models()` in `search.py` (the classification/regression template). Reuses existing `one_class_metrics()`, `build_one_class_model()`, and `build_preprocessing_pipeline()` rather than duplicating logic. Replaced the inline GUI block (spectral_predict_gui_optimized.py ~25115-25227) with a single call passing `self.validation_top_n.get()`.

Adds all 7 val_* columns in canonical order: val_Sensitivity, val_Specificity, val_Precision, val_F1, val_Accuracy, val_BalancedAcc, val_AUC. Preprocessing cache keyed by config so models sharing preprocessing only pay the transform cost once.

### Bug: Progress tab flooded with blank lines during one-class validation metrics step

**Root cause:** First cut of `compute_validation_metrics_for_top_one_class_models` called `progress_callback` every row with `{'stage': 'validation', 'current': ..., 'total': ...}` — no `'message'` key. `_progress_callback` at `spectral_predict_gui_optimized.py:25958` does `msg = info.get('message', '')` then `self._log_progress(msg)`, so every invocation appended an empty line. 80 rows → 80 blank lines.

**Fix:** Throttled callback to every 10 rows and included a `'message'` key, matching the classification helper's pattern at `search.py:738-744`.

### Bug: val_* columns appeared at far-right of Results tab instead of adjacent to cal/CV columns

**Root cause:** Helper only did `df_results[col] = np.nan` which appends new columns at the end of the DataFrame. Classification helper explicitly reorders at `search.py:748-789` to slot validation columns right after their calibration counterparts.

**Fix:** After populating metrics, helper now removes val_* columns and re-inserts them immediately after the last cv metric present in the frame (prefers `AUCcv`, falls back through the cv metric chain, then to `Imbalance`/`SubsetTag`). Final layout now reads: cal metrics → cv metrics → val metrics → top_vars/all_vars/CompositeScore/Rank.

**Verified:** Synthetic 25-model DataFrame — progress_callback invoked exactly 2 times (at i=10, i=20) with proper messages; val_Sensitivity lands at index `AUCcv + 1`; existing 44 contamination tests still pass.

### Bug: PCA-SIMCA silently produces 0 rows in Bayesian one-class runs on small training sets

**Symptom reported:** User selected 5 one-class models × 20 Bayesian trials and expected 100 rows in the Results tab. Got 80 — PCA-SIMCA was completely missing. Running again with 4 models INCLUDING SIMCA (implying a different split/inlier count) produced SIMCA results. User asked: "is there a reason small validation set would impact SIMCA more than other models?"

**Root cause:** `PCASIMCA.fit()` at `contamination.py:108-111` had a hardcoded guard:

```python
if n_samples < 10:
    raise ValueError(f"Need at least 10 clean samples to fit DD-SIMCA, got {n_samples}")
```

User's dataset had ~7 training inliers after the validation split. 5-fold CV produces training folds of size `4N/5` = 5–6 samples. Every single fold tripped the guard → `run_one_class_cv` skip guard (`contamination.py:599`) fired → Bayesian objective returned `+inf` → `convert_study_to_dataframe` dropped all 20 SIMCA trials (`unified_bayesian.py:1960-1961`) → 0 SIMCA rows in final DataFrame.

**Answer to the user's question: yes, SIMCA alone is vulnerable to small training sets.** The other 4 one-class models each have explicit small-sample fallbacks:

| Model | Small-sample accommodation |
|---|---|
| OneClassSVM | Kernel-based — fits 2+ points |
| IsolationForest | Random split trees — fits 1+ points |
| LOF | `n_neighbors` auto-clamped to `n_samples - 1` |
| EllipticEnvelope | PCA fallback at `contamination.py:568-572` when `n_features > n_samples` |
| PCA-SIMCA | **None — hard floor at 10** |

So the validation set size doesn't affect SIMCA "more" in some abstract sense; it's that carving any fixed amount off an already-small training set pushes SIMCA over its unique cliff while the others coast.

**Fix (`contamination.py:105-128`):** Replaced the 10-sample floor with the true mathematical minimum:

```python
if n_samples < 3:
    raise ValueError(f"Need at least 3 clean samples to fit DD-SIMCA, got {n_samples}")
# ...
max_components = min(n_samples - 1, n_features)
if max_components < 1:
    raise ValueError(...)
```

`n_components` was already clamped to `n_samples - 1` at line 114, and `_fit_chi2` already has a method-of-moments fallback (`:187-210`) that handles low-sample instability gracefully. The 10-sample floor was conservative, not mathematical.

**Skip-reason surfacing:** Augmented `run_one_class_cv` to collect `fold_errors` and return `skip_reason` in the result dict (`contamination.py:540-610`). The Bayesian objective now stores this as `trial.set_user_attr('skip_reason', ...)` when a trial bails (`unified_bayesian.py:1003-1009`), and the progress wrapper surfaces it in the GUI as `"Trial N/20 - SKIPPED (<reason>)"` instead of the previous useless `-inf` (`unified_bayesian.py:1810-1821`). Future silent-drop failure modes will now be visible.

**Verified:** Direct unit test with user's exact scenario (7 inliers, 30 features, 5-fold CV):
- 5 samples: fit succeeds (previously failed)
- 3 samples: fit succeeds (new floor)
- 2 samples: correctly rejected with new error message
- `run_one_class_cv` with 7 inliers / 5-fold: **all 5 folds succeed**, BalancedAcc=0.60, sensitivity=1.00
- 46 existing contamination tests still pass

**Follow-up requested by user (unclaimed):** Add Leave-One-Out CV as an optional cv_strategy for all task types. LOO uses every sample and eliminates the fold-too-small failure mode entirely for tiny datasets — noted in PROJECT_STATUS.md Follow-Ups section.

---

## 2026-04-09 — Pre-Merge Review Fixes (Claude Opus 4.6)

### Bug: Bayesian one-class results show NaN on Results tab
**Root cause:** `unified_bayesian.py:1000` explicitly set `compute_calibration=False`. Cal metrics never computed → trial user_attrs empty → DataFrame columns all NaN. Model Development worked because it recomputes metrics directly from trained model.
**Fix:** Changed to `compute_calibration=True`. Adds ~2x per trial but Bayesian runs 50-100 trials vs grid's thousands.

### Bug: Scaler/PCA not persisted in grid search result dicts
**Root cause (found by Codex gpt-5.4):** `run_one_class_search()` result dicts never stored `cal_scaler`, `cal_pca_reducer`, or `oc_score_stats` from `run_one_class_cv()`. Saved OneClassSVM/LOF/EllipticEnvelope models would predict on raw features instead of scaled space. Also caused batch-dependent uncertainty thresholds.
**Fix:** Added `scaler`, `pca_reducer`, `oc_score_stats` to both result dict blocks (full-spectrum and varsel).

### Bug: LightGBM label encoding in importance computation
**Root cause:** `compute_one_class_importances()` passed +1/-1 labels to `LGBMClassifier` which expects 0/1. Auto-remapped but semantically ambiguous and may fail in strict builds.
**Fix:** Convert `y_oc` to binary before fit: `(y_oc == -1).astype(int)`.

### Performance: Reduced IsolationForest/LOF grid bloat
- IF: replaced `n_estimators=200/300` entries with `n_estimators=100` + `max_features` variants (~50% faster)
- LOF: removed 2 redundant configs that only varied `contamination` with identical `n_neighbors=20` (~40% faster)
- Discovery: skipped unused importance computation during initial scan (~17% faster)

---

## 2026-04-09 — One-Class Bug Fixes (Claude Opus 4.6)

### Bug: One-class models "hang" on Model Development page
**Root cause:** `_run_refined_model_thread()` one-class path returns at ~line 34075 and ~line 33997, bypassing `_update_refined_results()` which re-enables buttons, resets cursor, enables save. Not a real hang — the model completes but the UI never updates.
**Fix:** Call `_update_refined_results(results_text)` before returning.

### Bug: Dancing man animation never stops for one-class
**Root cause:** `_run_analysis_thread()` one-class path returns at ~line 25087, bypassing cleanup code at lines 25808-25822 (stop animation, play chime, reset buttons). Same pattern as Model Dev — early return skips shared cleanup.
**Fix:** Added cleanup calls before all 4 one-class early returns. Also fixed 2 pre-existing issues: `_update_search_buttons('stopped')` (invalid state, should be `'idle'`) and non-one-class Bayesian missing-module error path.

### Bug: External validation checkbox unchecks for one-class
**Root cause:** One-class early return skips `self.last_training_config` storage (line 25716). When user double-clicks a result to load in Model Dev, `_load_model_for_refinement` sees no training config → clears `validation_enabled` (line 30349).
**Fix:** Store `last_training_config` before the one-class early return.

### Bug: No validation metrics for one-class
**Root cause:** One-class early return at ~line 25099 exits before the validation computation block at ~line 25256. `run_one_class_search` also has no validation parameters. Validation metrics simply never computed.
**Fix:** Added validation computation in both the Results path (rebuild full pipeline per result row) and Model Development path (use calibration model's scaler/PCA).

### Bug: Grid search inlier label type mismatch
**Root cause:** `search.py` line 4796 does `y_np == inlier_class_label` without string conversion. Bayesian path (line 831-832) converts both to strings. If y has strings and label is numeric (or vice versa), all samples marked as outliers.
**Fix:** Added string conversion to match Bayesian path.

### Bug: SHAP hangs for one-class models
**Root cause:** KernelExplainer is O(n_features × n_samples × n_background) model evaluations. With spectral data (hundreds of wavelengths), this takes hours, not seconds.
**Fix:** Replaced with permutation importance (fast, model-agnostic). TreeExplainer used only for IsolationForest without scaler/PCA.

### Bug: Wavelength importance silent failure for one-class
**Root cause:** `get_feature_importances()` in `models.py` raises ValueError for one-class model names. Exception caught silently at line 31393 — no plot, no error message.
**Fix:** Added one-class branch in `_plot_wavelength_importance()` that calls existing `compute_one_class_importances()` from `contamination.py`.

### Bug: Preprocessing discovery callback mismatch
**Root cause:** `discover_preprocessing` calls `progress_callback(current, total, msg)` with 3 positional args. GUI's `_progress_callback` expects a dict. Regular grid search had a wrapper (line 1204); one-class grid search passed the raw callback.
**Fix:** Added same wrapper pattern for one-class.

### Bug: Residual correlation zeroed out for ALL models
**Root cause:** Indentation error — `self.wavelength_residual_corr_data = np.zeros(len(wavelengths))` at line 31453 was at the `else` block level, not inside the `except` block. Always overwrote the successfully computed correlation.
**Fix:** Indented into the `except` block.

### Architecture insight: One-class early returns are the #1 bug pattern
The one-class pipeline was added as an early-exit branch (`if task_type == "one_class": ... return`) in both `_run_analysis_thread` and `_run_refined_model_thread`. Every piece of shared cleanup code after the return is skipped. There are 6+ locations where this caused bugs. When adding any new post-analysis feature, check that the one-class path also reaches it.

### Architecture insight: Preprocessing discovery dropdown is cosmetic for one-class
All importance methods (model_specific, lightgbm, cars_tree, vip) resolve to LightGBM for one-class. The per-model refinement at line 935 of `preprocessing_discovery.py` never triggers because `models_to_test` isn't passed from `run_one_class_search`.

### Performance: First-time one-class preprocessing discovery is slower than classification
User reported ~20-100x slower. Likely because one-class has fewer outliers → stratified CV with very small minority class → more variance → potentially more work. Needs profiling. Applies to grid search, not just discovery.

### Bug: Validation metrics fail with "Unknown preprocess: None"
**Root cause:** Grid search results use column names `'Preprocess'`/`'PreprocessBase'`/`'Deriv'`/`'Window'`/`'Poly'`. Validation code used `'Preprocessing'`/`'Derivative'` — wrong column names, so all values were None.
**Fix:** Use `row.get('PreprocessBase', row.get('Preprocess', row.get('preprocessing', 'raw')))` chain that handles both grid search and Bayesian column naming.
