# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

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
User reported ~20-100x slower. Likely because one-class has fewer outliers → stratified CV with very small minority class → more variance → potentially more work. Needs profiling.
