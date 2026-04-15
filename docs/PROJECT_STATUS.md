# Project Status

> **Last updated:** 2026-04-15 by Claude (Opus 4.6) — **PR #4 numeric parity validation complete. Verdict: PARTIAL → merge with release note.** Calibration metrics bit-identical across all four (model, task) combos. Classification CV metrics bit-identical (PLS-DA + LightGBM, 34/34 numeric keys each). Regression CV: `R2cv`, `MAEcv`, `Bias`, `CompositeScore`, per-quartile RMSE bit-identical — **only `RMSEcv` (and derived `RPD`/`RER`) differ by ~7.5%**. Root cause: deliberate pooling-strategy change at `search.py:4214` (main mean-of-fold) vs `search.py:4293` (branch pooled `sqrt(MSE)`), motivated by LOO degeneracy + chemometrics-convention alignment (Unscrambler/PLS_Toolbox/SIMCA). Predictions themselves are identical (proven by R2cv matching to 16dp). LOO sanity: regression sensible; classification degenerates to majority-class voting (data/model issue, not plumbing). Full report: `docs/pr4_parity_report.md`. Pending user review.

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
- [x] Prediction tab: save→load→predict round-trip verified manually 2026-04-11. Training inliers come back as "Inlier (X)" as expected. Regression test: `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`.
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
- [x] CV strategy support: LOO, Repeated K-Fold, and standard K-Fold via `build_cv_splitter()` factory in `cv_utils.py`. Pooled RMSEcv (regression) and pooled sensitivity/specificity (one-class). GUI controls in Analysis tab + Model Development. Cost estimator with LOO/Repeated warnings. training_config stores cv_strategy for model save/load. Post-review fixes (2026-04-12): RepeatedKFold crash via `cross_val_predict_pooled`, one-class Bayesian cv_strategy forwarding, GUI LOO folds metadata, LOO classification minority-class guard, differentiated mixed-regime warnings.

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
- [ ] **LOO / Repeated K-Fold results may not auto-populate the Results tab** — observed 2026-04-12 during manual testing of PR #4 (cv-strategy-overhaul). User ran LOO on a 49-sample regression dataset; analysis completed without errors, but the Results tab did not display the completed runs as expected. Repeated K-Fold uncertain — not explicitly tested for the same behavior. Save/load/predict round-trip from a refined LOO model works. Possible causes to investigate: (a) Results tab's Treeview populate path is gated on `Folds` matching a specific integer range or on `training_config['cv_strategy'] == 'kfold'`; (b) progress callback's final "completion" signal isn't wired for non-kfold strategies; (c) sort/filter logic in `_populate_results_treeview` (or equivalent) drops rows where `Folds == n_samples` (LOO case). Needs repro + trace before fixing.

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
2. ~~Re-run grid-search OC validation in the GUI to confirm `val_*` columns now populate~~ ✅ done 2026-04-11 — user confirmed "the validation stats now appear in grid search"
3. ~~Merge PR #3 to main~~ ✅ pending merge in this session
4. Follow-up PRs (post-merge): see "Follow-Ups (unclaimed)" section below

## Follow-Ups (unclaimed)

- **Expose all one-class hyperparameter grids in Model Config subtab.** Currently only four scalars (`nu`, `contamination`, `alpha`, `n_components`) are surfaced in `oc_hyperparams_frame` (`gui:6040`). Everything else (kernel, gamma, n_estimators, max_samples, max_features, support_fraction, n_neighbors, metric, additional alpha/n_components values) lives in the hardcoded `ONE_CLASS_PARAM_GRIDS` constant in `src/spectral_predict/contamination.py`. This violates the project rule from `CLAUDE.md`: *"All hyperparameters are exposed and user-editable."* Suggested scope: add 5 collapsible cards to `_create_tab4c_model_configuration` (one per OC model family — OCSVM, IF, EllipticEnvelope, LOF, PCA-SIMCA), thread the resulting per-model dicts through `run_one_class_search` and `run_unified_bayesian` instead of `ONE_CLASS_PARAM_GRIDS`, and gate them on Custom tier the same way classification/regression do. See `SESSION_LOG.md 2026-04-11` for the full per-model parameter inventory.

- **OC validation helper silently uses default params on `ast.literal_eval` failure.** `compute_validation_metrics_for_top_one_class_models` at `contamination.py:1010-1019` falls back to `params={}` if the row's `Params` string can't be parsed (malformed CSV reload, unexpected dict format, etc.). The helper then validates the *default* estimator and reports believable-but-wrong `val_*` metrics for that row. Fix: skip the row with `logger.warning("[OC Validation] Params parse failed for row %s, skipping", idx)` and `continue`, mirroring the no-inliers branch a few lines down. Codex flagged this as MAJOR in the pre-merge review; not a ship-blocker because the GUI-produced rows always have well-formed `Params`, but a silent-wrong-result hazard for any external CSV reload.

- **OC validation helper silently truncates / falls back to full-spectrum on wavelength-map miss.** `contamination.py:977-989` reads `all_vars`, parses to floats, looks up each in `wl_to_idx`, and quietly drops any wavelength that isn't found. If the lookup is partial → validates on a truncated feature set. If it's empty → silently validates on the full spectrum. Either way the saved `val_*` metrics don't correspond to the model the row claims to describe. Fix: require exact mapping (all model wavelengths present in `wl_to_idx`) or skip the row with a logged warning. Codex flagged this as MAJOR; same severity caveat as the Params parse fix.

- **`predict_with_uncertainty` swallows one-class decision_function failures silently.** `model_io.py:854-860`: after `predict_with_model()` returns, the bare `except Exception: decision_scores = None` catches any score-extraction error and returns predictions with empty `uncertainty`/`applicability_domain` payloads. The GUI has no way to surface "predictions worked, uncertainty broke" — exactly the failure mode this PR has already hit once with the order-of-operations bug. Fix: log the exception with `logger.warning(...)` at minimum; ideally return a structured `decision_score_error` flag that the GUI can display alongside the predictions.

- **OC validation helper can't recover the right `polyorder` for 2nd-derivative grid-search rows.** Grid search writes `polyorder=None` (delegating to `polyorder_map` inside `SavgolDerivative` which picks 3 for `deriv=2`), but the validation helper at `contamination.py:922-927` falls back to `min(2, window-1) if window > 2 else 0`, which gives `poly=2` for `deriv=2`. The pipeline then runs with the wrong polyorder and the resulting `val_*` metrics are slightly off vs. what training actually used. Fix: either store the resolved polyorder in the grid-search result dict (most direct), or import the same `polyorder_map` lookup into the validation helper, or have `_maybe_int` fall back to `polyorder_map[deriv]` when poly isn't set.

- **CV Strategy Phase 2: Propagate outer CV strategy into variable selection inner loops.** Currently inner loops (UVE, SPA, iPLS, CARS, GA-PLS) always use hardcoded 5-fold K-fold regardless of the outer CV strategy. A GUI warning is displayed when outer != kfold. Phase 2 would thread `cv_strategy`/`cv_n_repeats` into the variable selection internals.

- **CV Strategy Phase 2: Propagate CV strategy into predictor screening and smart preprocessing.** These internal CV sites are currently hardcoded 5-fold and don't respect the user's chosen strategy.

- **CV Strategy Phase 2: NSGA-II multi-objective search CV strategy support.** Currently falls back to K-fold with a logged warning when non-kfold strategy is selected. Needs native LOO/Repeated K-fold support.

- **One-class search: Add `training_config` to result rows.** Regression/classification grid search writes `training_config` (with `cv_strategy`, `folds`, `cv_n_repeats`) but one-class search does not. This means one-class saved models don't preserve the CV strategy used during training.
