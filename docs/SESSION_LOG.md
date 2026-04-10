# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

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

**Verified:** Synthetic 3-model DataFrame populates all 7 val_* columns for all rows; `top_n=2` correctly limits to top 2 by Rank. Existing 44 contamination tests pass.

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
