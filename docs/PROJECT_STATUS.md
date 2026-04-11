# Project Status

> **Last updated:** 2026-04-11 by Claude (Opus 4.6) — final pre-merge cleanup pass on PR #3 (grid-search OC validation silent NaN + bundled minimum-to-merge fixes)

---

## Active Branch: `claude/contamination-detection-model-PrntN`

**Goal:** Add one-class contamination detection models as first-class citizens alongside regression and classification. One-class models learn from clean/inlier samples only and flag anything outside that distribution as suspect.

**Models:** OneClassSVM, IsolationForest, LOF (LocalOutlierFactor), EllipticEnvelope, PCA-SIMCA

**Prediction tab manual verification: PASSED 2026-04-11.** User confirmed one-class predictions correctly show inliers when they should after the 2026-04-10 order-of-operations fix. See SESSION_LOG.md 2026-04-10 entry for the original bug trace.

---

## What Works

- [x] One-class model implementations in `src/spectral_predict/contamination.py`
- [x] Bayesian optimization for one-class (`unified_bayesian.py` handles `task_type='one_class'`)
- [x] Grid search for one-class (`run_one_class_search` in `search.py`)
- [x] GUI task type selection: "One-Class" radio button shows one-class model checkboxes
- [x] Inlier class selection UI with auto-detection
- [x] Model save/load with scaler + PCA reducer persistence (`model_io.py`)
- [ ] **Prediction tab: AWAITING MANUAL VERIFICATION** — code fix + automated regression tests in place (see `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`). Do not re-mark as working until the user runs the real GUI save→load→predict flow on actual data and confirms training inliers come back as "Inlier (X)".
- [x] External validation: label mapping, confusion matrix, balanced accuracy/sensitivity/specificity
- [x] External validation metrics in Results tab — top N respects `validation_top_n` (default 700), matching classification/regression
- [x] External validation produces full 7-metric set (Sensitivity, Specificity, Precision, F1, Accuracy, BalancedAcc, AUC) via `compute_validation_metrics_for_top_one_class_models()` in `contamination.py` — parity with cal/CV
- [x] External validation metrics in Model Development results text
- [x] Validation checkbox preserved when loading results into Model Development
- [x] Dancing man animation stops on completion
- [x] Model Development: runs complete, buttons re-enable, cursor resets
- [x] Wavelength importance: surrogate LightGBM via `compute_one_class_importances()`
- [x] SHAP: permutation importance for most models, TreeExplainer for IsolationForest
- [x] Diagnostics: decision score distribution + sample classification plots
- [x] Residual/leverage diagnostics route to one-class-specific plots (not regression fallback)
- [x] Basic preprocessing discovery works for one-class grid search (callback wrapper fixed)
- [x] Variable selection: 'importance' method works for one-class
- [x] Results Treeview tooltips: all one-class metrics (Sensitivity/AUC/cv) + validation metrics (RMSEP/R2pred/val_*) now covered; jargon-heavy tooltips (ROC_AUC, Kappa, MCC, BER, LogLoss) rewritten for non-technical audience

## Known Issues

- [x] ~~**Bayesian one-class results NaN on Results tab**~~ — Fixed: enabled `compute_calibration=True` in Bayesian objective
- [x] ~~**Scaler/PCA not persisted in grid search results**~~ — Fixed: `cal_scaler`, `cal_pca_reducer`, `oc_score_stats` now stored in result dicts
- [x] ~~**LightGBM label encoding bug**~~ — Fixed: `compute_one_class_importances()` now converts +1/-1 to 0/1
- [x] ~~**"Contamination" naming**~~ — Fixed: comments/docstrings updated where "contamination" referred to task type
- [x] ~~**Performance: IsolationForest/LOF grid bloat**~~ — Fixed: reduced IF n_estimators (300→100), removed redundant LOF configs, skipped unused importance in discovery
- [x] ~~**PCA-SIMCA silently produces 0 rows on small training sets**~~ — Fixed 2026-04-10: `PCASIMCA.fit()` had a hardcoded `n_samples < 10` guard that tripped every CV fold when training inliers were scarce (e.g. ~7 inliers + 5-fold CV → 5-6 per training fold). All 5 folds failed → skip guard → `+inf` every trial → 0 rows in final DataFrame. Other one-class models (OCSVM/IF/LOF/EllipticEnvelope) all have explicit small-sample fallbacks. Relaxed SIMCA floor to `n_samples < 3` (the true mathematical minimum for PCA + chi² moment fit). Also added `skip_reason` surfacing through `run_one_class_cv` → `unified_bayesian` progress wrapper so any remaining skips show "SKIPPED (reason)" in the GUI progress tab instead of a silent `-inf`.
- [x] ~~**Every specimen labeled "Outlier" at predict time, including training specimens**~~ — Fixed 2026-04-10 (second session). Root cause: one-class refinement thread in `_run_refined_model_thread` trained with full-spectrum preprocessing first then subset to selected wavelengths, but hardcoded `'use_full_spectrum_preprocessing': False` in the saved metadata and never populated `full_wavelengths`. At predict time `model_io.py` took the subset-first branch, which for SNV / SG derivatives / baseline correction produces feature values mathematically different from training, so the StandardScaler fit inside `run_one_class_cv` received distributionally alien inputs and `model.predict()` returned -1 on everything. Fix: assign `self.refined_full_wavelengths = list(original_wavelengths)` and flip the flag to `True`, matching the regression/classification refinement path. Regression test added: `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`. See `docs/SESSION_LOG.md 2026-04-10 "One-Class Prediction Disaster"` for full trace.
- [x] ~~**Uncertainty subtab always shows "Outlier" for one-class**~~ — Fixed 2026-04-10 (second session). `_display_uncertainty()` at line ~38318 read already-stringified predictions (`"Inlier (…)"` / `"Outlier"`) from `predictions_df` but compared them to the integer `1`, which always evaluated False. Would have silently contradicted the Results tab once the primary bug above was fixed. Replaced with an `isinstance(raw, str)` branch.
- [x] ~~**Bayesian trials did not persist `cal_scaler`, `cal_pca_reducer`, `oc_score_stats`**~~ — Closed 2026-04-10 (second session). Latent bug: `unified_bayesian.py` stored metrics and preprocessing config but not the fitted calibration artifacts, so any future direct-save-from-results path would silently save `None`. Added `set_user_attr` calls mirroring `search.py:5171-5173` and propagated the three keys through `convert_study_to_dataframe`. Does not change current user-visible behavior because saves go through refinement, which re-runs CV from scratch.
- [x] ~~**Grid-search one-class validation columns silently empty**~~ — Fixed 2026-04-11. Grid search wrote display name (`'snv_deriv1_w11'`) in `Preprocess` but never set `PreprocessBase`, so the validation helper passed an unparseable name to `build_preprocessing_pipeline`, the resulting `ValueError("Unknown preprocess: …")` was swallowed by the row-level except, and val_* dropped to NaN for every derivative grid-search row. Bayesian was unaffected because it already wrote `PreprocessBase`. Two-layer fix: search.py now writes `PreprocessBase` (mirrors classification at line 4506), and `compute_validation_metrics_for_top_one_class_models` now normalizes display names as a fallback. Regression test: `tests/test_contamination_detection.py::TestGridSearchValidationMetricsParity`. See `docs/SESSION_LOG.md 2026-04-11`.
- [ ] **One-class grid search inherently slower than classification** — Expected due to two-phase CV+calibration design and LOF O(n²) complexity. Optimized but still slower by nature.
- [ ] **Preprocessing importance dropdown has no effect for one-class** — All methods resolve to LightGBM. Per-model refinement never triggers because `models_to_test` isn't passed.
- [ ] **Variable selection limited** — Only 'importance' method works. UVE/SPA/iPLS/CARS are PLS-specific and incompatible. This is by design but could use a UI hint.
- [ ] **Residual correlation overlay** — Disabled for one-class (no continuous residuals). Shows zeros.
- [ ] **One-class hyperparameter grids are NOT exposed in Model Config subtab** — violates the project rule "All hyperparameters are exposed and user-editable" (`CLAUDE.md`). Only four scalars are surfaced in `oc_hyperparams_frame` (`gui:6040`): `nu`, `contamination`, `alpha`, `n_components`. Everything else (kernel, gamma, n_estimators, max_samples, max_features, support_fraction, n_neighbors, metric, additional alpha/n_components values) lives in the hardcoded `ONE_CLASS_PARAM_GRIDS` constant in `src/spectral_predict/contamination.py`. Regression and classification have a per-model collapsible card each (`_create_tab4c_model_configuration`, lines 11726+) — one-class needs five equivalent cards (OCSVM, IF, EllipticEnvelope, LOF, PCA-SIMCA). See `SESSION_LOG.md 2026-04-11` for the full inventory.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Surrogate LightGBM for wavelength importance | Model-agnostic, fast, already implemented in `compute_one_class_importances()` |
| Permutation importance instead of SHAP KernelExplainer | KernelExplainer is impossibly slow for spectral data (hundreds of wavelengths × thousands of model evals per sample) |
| TreeExplainer only for IsolationForest without scaler/PCA | Only case where SHAP TreeExplainer works directly |
| Validation metrics: balanced accuracy, sensitivity, specificity | Standard one-class metrics. Sensitivity = outlier detection rate, Specificity = inlier retention rate |
| String comparison for inlier labels | Both Bayesian and grid search convert y and inlier_label to strings before comparison to handle numeric/text label mismatches |
| Early return pattern in GUI threads | One-class has separate pipeline from regression/classification. Every early return MUST include cleanup (animation stop, button reset, etc.) |

## Key Files

| File | Role |
|------|------|
| `spectral_predict_gui_optimized.py` | Main GUI (~45K lines). One-class paths scattered throughout. |
| `src/spectral_predict/contamination.py` | One-class model implementations, `run_one_class_cv()`, `compute_one_class_importances()` |
| `src/spectral_predict/search.py` | `run_one_class_search()` for grid search |
| `src/spectral_predict/unified_bayesian.py` | Bayesian optimization, handles `task_type='one_class'` |
| `src/spectral_predict/scoring.py` | Scoring functions with one-class metrics |
| `src/spectral_predict/model_io.py` | Save/load with scaler/PCA persistence |
| `src/spectral_predict/preprocessing_discovery.py` | Smart preprocessing, has one-class path at line ~680 |

## Next Steps

1. ~~End-to-end manual test of save→load→predict~~ ✅ done 2026-04-11
2. Re-run grid-search OC validation in the GUI to confirm `val_*` columns now populate (regression test passes; quick visual confirmation in the actual Results tab is the last thing keeping this from merge)
3. Consider adding UI hint when variable selection methods are incompatible with one-class
4. Merge to main
5. **Follow-up PR** (post-merge): expose all one-class hyperparameter grids in Model Config subtab — see Known Issues entry for scope

## Follow-Ups (unclaimed)

- **Add Leave-One-Out CV as a CV-strategy option for all task types (regression, classification, one-class).** Small training sets (especially one-class, where only inliers count toward `k`) hit K-fold edge cases hard — e.g. 7 inliers with 5-fold leaves 5-6 per training fold, which stresses any model with a minimum-samples floor. LOO uses every sample, eliminates the "fold-too-small" failure mode entirely, and gives a less biased estimate for tiny datasets. Would slot into `run_one_class_cv` (swap `KFold(n_splits=n_folds)` for `LeaveOneOut()` behind a `cv_strategy='loo'` branch) and the equivalent paths in `search.py` / `unified_bayesian.py`. UI needs a new control in the CV-folds area ("Leave-one-out" as an alternative to numeric folds).
