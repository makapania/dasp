# Project Status

> **Last updated:** 2026-04-19 by GLM-5.1 — Auto Bone FTIR index extraction added to Explore-tab Peak Calculator (main branch, uncommitted). New backend functions `extract_bone_ftir_indices()` and `extract_bone_ftir_indices_all_samples()` in `peak_calculator.py`. GUI mode toggle (Manual / Auto Bone FTIR) in Peak Calculator dialog. Auto mode uses fixed Kontopoulos et al. 2018 method, now reads stored spectra directly (no transform-capable Explore "raw" path), validates cm-1 + absorbance. Explore and Peak Calculator now both support single-sample selection: sample names appear directly in the Peak Calculator scope dropdown, and the Explore Sets row has a Sample dropdown that acts as an explicit set-of-one override. Selecting a sample clears the active set selection, disables assign mode, labels the state as `Single sample override`, and now actually filters the Explore raw/derivative/baseline plots down to the selected spectrum; `All Samples` restores the normal full view. 24/24 new tests pass, 117/117 existing peak-calc/bone tests pass, implementation matches reference ftir_indices.py exactly. All changes uncommitted in worktree.

> **Previously:** 2026-04-19 by GLM-5.1 — Analysis Subset V1 implemented on branch `glm/analysis-subset-v1` (NOT yet merged). New pure-logic module `src/spectral_predict/analysis_subset.py` with 41 tests. Analysis tab has "Analysis Subset" card above Holdout Validation. Metadata-only dialog with categorical multi-select. Subset provenance stored in `last_training_config`. Mismatch warnings in Model Development. One-class guardrail blocks when subset removes all inlier samples. C2 missing-column safe handling via `_refresh_active_group_indices`. All changes uncommitted in worktree. 41/41 analysis_subset tests + 61/61 OC regression tests pass. See `docs/SESSION_LOG.md` 2026-04-19 entry for architecture decisions and gotchas.

---

## Repo-distribution install path (added 2026-04-17)

Goal: enable a small group to clone the repo and run the GUI without dealing with venv setup themselves. Bundled-app limitations (PyInstaller forces Python 3.11, multiprocessing falls back to threading-only) led to deciding to share the source for now.

**What's in place:**
- `install.bat` (Windows) / `install.sh` (mac/linux): detects Python 3.12, creates `.venv312`, runs `pip install -e . --upgrade`. Idempotent.
- `INSTALL.md`: GUI-focused walkthrough with troubleshooting for the three real failure modes (Python missing, proxy/firewall, missing C++ build tools).
- `pyproject.toml` deps audited via AST scan vs declared deps. Added: `Pillow>=10.0.0`, `shap>=0.44.0`. Re-enabled: `jcamp>=1.2.1` (the old "Python 3.11 bug" comment was stale; jcamp 1.2.x imports cleanly on 3.12). Floors bumped: `numpy>=2.0`, `pandas>=2.0`, `scikit-learn>=1.5`, `scipy>=1.11` (sklearn floor traces to PR #5 evidence).
- `run_gui.sh`: fixed stale `spectral_predict_gui.py` filename (was `spectral_predict_gui_optimized.py`).

**Intentionally NOT declared as required:**
- **`torch`** — `src/spectral_predict/learned_preprocessing.py` exists and imports torch, but is **not wired into the GUI**. The dead import block at `gui:165-170` was removed 2026-04-17. The module itself is preserved because there may be a future place for learned-preprocessing models. **If/when learned_preprocessing is wired into the GUI**, decide: (a) add torch to required deps (~800MB install cost), or (b) keep torch optional + show a clear "this feature requires torch — install with `pip install torch`" message in the GUI when the user tries to use it.
- **`agilent-ir-formats`** — no Python 3.12 wheel exists on PyPI. Stays in optional `[agilent]` extra. .seq file loading will raise a clear ImportError from `agilent_reader.py:62` until the upstream package ships a 3.12 wheel.

**Next:** Test the 3.12 bundle GUI with actual analysis (user-verified). The threading fallback means no multiprocessing parallelism in the bundle, but no crash either.

---

## 3.11 build retirement — pending real-world soak (2026-04-17)

The 3.12 PyInstaller bundle is now functional and superior on every dimension that matters (newer wheels, cleaner dep set, working installer, fork-bomb fix). The 3.11 bundle is now redundant *technically*. **Recommendation: keep the 3.11 build path in-repo for ~2 weeks of real-user soak before retiring.** The 3.12 path has only been verified by the maintainer on one machine; new edge cases may surface when the limited-share group exercises it. Once the 3.12 bundle has run cleanly for a few weeks across the audience:

1. Delete `spectral_predict.spec`, `spectral_predict_mac.spec`, `build_installer.py`, `installer/spectral_predict.iss`, `docs/BUNDLED_APP_BUILD_GUIDE.md`, `RUN_SPECTRAL_PREDICT.bat`, `run_v3.bat`
2. Rename `spectral_predict_py312.spec` → `spectral_predict.spec`, `build_installer_py312.py` → `build_installer.py`, etc. (drop the `_py312` suffix everywhere)
3. Adjust the spec/build script for the rename (the `APP_NAME` constant, the .iss `OutputBaseFilename` — current values keep `-py312`/`_py312` suffixes to coexist with the 3.11 path)
4. Update `docs/BUNDLED_APP_BUILD_GUIDE_PY312.md` → `docs/BUNDLED_APP_BUILD_GUIDE.md`
5. Delete `.venv311/` (or leave for occasional 3.11 reproduction if needed)

**Risk of retiring now (today):** if the 3.12 bundle has an undiscovered issue specific to a user's machine (different antivirus interaction, missing C++ runtime, etc.), there's no fallback. The 3.11 path is well-traveled.

**Cost of keeping the 3.11 path another 2 weeks:** essentially zero — the two paths are independent files, no maintenance overhead, no risk of cross-contamination.

---

## Tooltip coverage — RESOLVED 2026-04-17 (GLM 5.1 implementation)

Originally an audit-only finding (6 missing-tooltip items + 1 codebase discrepancy); GLM then implemented the fixes in the same session. **All 16 missing `CreateToolTip()` calls added** + new `TOOLTIP_CONTENT['one_class']` section (10 entries) + new `cross_validation['repeats']` key. Tooltips are research-backed (sklearn docs + chemometrics literature with citations: Bouckaert & Frank 2004, Schölkopf et al. 2001, Liu/Ting/Zhou 2008, Breunig et al. 2000, Rousseeuw & Van Driessen 1999, Wold 1976, Oliveri & Downey 2012).

**Coverage now includes:**
- CV repeats label + spinbox (main tab + refine tab)
- CV strategy combobox (main tab + refine tab)
- All 5 one-class model checkboxes (PCA-SIMCA, OneClassSVM, IsolationForest, EllipticEnvelope, LOF)
- Inlier class label + combobox
- All 4 one-class hyperparameter controls (`nu`, `contamination`, `alpha`, `n_components`)

**Verification: passed.** GUI imports cleanly, TOOLTIP_CONTENT['one_class'] dict has all 10 expected keys, no key shorter than 50 chars (sanity guard against placeholder text).

### `repeated_stratified_kfold` — clarified (was a documentation issue, not a bug)
Earlier audit said `RepeatedStratifiedKFold` doesn't appear in the codebase. **Refined finding:** the *string value* `'repeated_stratified_kfold'` is not in the dropdown — but the *class* `RepeatedStratifiedKFold` IS imported and used in `cv_utils.py`. The CV factory `build_cv_splitter` auto-selects between `RepeatedKFold` and `RepeatedStratifiedKFold` based on task type (classification → stratified, regression → unstratified) when the user selects `repeated_kfold`. So users see one dropdown option, get the right splitter under the hood. No user action needed; status doc PR #4 description was technically accurate, just misleading without that context.

---

## Recently resolved

### OC validation helper silent-wrong-result paths + PCA-SIMCA default — FIXED 2026-04-18 (9e9ca11 + follow-up)

Three silent-wrong-result paths in `compute_validation_metrics_for_top_one_class_models` (`src/spectral_predict/contamination.py`) eliminated — all now warn and skip instead of producing believable-but-wrong `val_*` metrics:
- Malformed / missing / empty `Params` string → skip (was: default to `{}`, validate with estimator defaults)
- Partial wavelength-map miss → skip (was: silently drop missing wavelengths, validate on truncated feature set)
- Removed fallback to `selected_wavelengths` user-attr (for regression/classification Bayesian, only first 50 are stored there; `all_vars` always holds the full trained subset for all OC code paths)
- Follow-up: missing/empty/NaN `all_vars` also now warns and skips (was: silently validated on full spectrum) — both OC Bayesian (`unified_bayesian.py:1056-1062`) and OC grid (`search.py:5332`) always populate it, so a missing value means row metadata is corrupt

Also bumped PCA-SIMCA `n_components` default from 5 to 10 at `gui:2892` (more sensible starting point for typical NIR datasets). Regression test suite: 6 new parametrized cases in `TestGridSearchValidationMetricsParity::test_corrupt_metadata_row_is_skipped_not_silently_wrong` covering all warn+skip paths. 54/54 `test_contamination_detection.py` pass. Closes three "Follow-Ups (unclaimed)" items flagged MAJOR by Codex in the pre-merge review.

### LOO + Repeated K-Fold cross-validation strategies — MERGED 2026-04-17 (PR #4)

Added `LeaveOneOut`, `RepeatedKFold`, and `RepeatedStratifiedKFold` as first-class CV options across regression / classification / one-class. Backend: `src/spectral_predict/cv_utils.py` — factory `build_cv_splitter`, `validate_cv_strategy_for_task`, `cross_val_predict_pooled`, `reduce_repeated_cv_predictions`. Pooled RMSEcv (regression — fixes the LOO degeneracy where mean-of-fold-RMSE = MAE); pooled sensitivity/specificity (one-class); BER consistency under repeated CV; AUC kept as mean-of-fold-AUCs (scores not comparable across independently-fit folds). GUI: Analysis-tab + Model-Development controls, cost estimator with LOO/Repeated warnings. `training_config` stores `cv_strategy` for save/load. Includes a code-export fallback in `code_generator.py:108-115` so exported scripts/notebooks honor the user's CV regime when callers pass `training_config` rather than top-level keys (caught in pre-merge codex review). 1456-line test suite at `tests/test_cv_strategy.py`, 5 prior review rounds visible in commit history.

### LightGBM "parameter capture" warning + NaN / crash on `.venv311` — FIXED 2026-04-16 (PR #5)

Fix: `clone(model)` at four pipe-construction sites in `search.py` (`:2191`, `:4161`, `:4163` for the regression bug, plus `:4139` PLS-DA defensive). Root cause: shared estimator instance being fit first on full preprocessed X (setting `n_features_in_=2151`), then reused in subset-feature fits; sklearn 1.5.2's pre-fit `_check_n_features(reset=False)` fires before the refit would reset the count, raising. sklearn 1.7.2's pre-fit check is more lenient and doesn't raise on the same pattern. The PLS-DA `:4139` site was added after Claude pr-reviewer flagged it as the same pattern; classification baseline run on `.venv311` with PLS-DA un-cloned proved PLS-DA doesn't actually hit the bug (importance-capture at `:2191` doesn't fire for PLS-DA), so the clone there is purely defensive/symmetric. (The separate sklearn 1.7.2 `"X does not have valid feature names"` UserWarning flood seen on `.venv312` comes from a DataFrame-fit / ndarray-predict transition in the pipeline; it's unrelated to the shared-state bug and is NOT changed by this fix — see COMPARISON.md.)

Verification: harness `scripts/verify_shared_model_fix.py` run with GUI defaults on BoneCollagen — 4-run before/after matrix on both venvs. Post-fix `.venv311` produces 9408 LightGBM rows with `best_cv_rmse=0.9702327793086989`, bit-identical to `.venv312`. See `docs/plans/2026-04-16-lightgbm-shared-model-fix.md` for the plan and `docs/plans/artifacts/2026-04-16/COMPARISON.md` for the metrics matrix.

## Known follow-ups (deferred from PR #4 reviews, non-blocking)

- **`run_bayesian_search()` does not call `validate_cv_strategy_for_task()` upfront.** Located in `src/spectral_predict/search.py:3096`. Unlike `run_search()` and `run_unified_bayesian()` which validate the CV-strategy / task-type / class-count combination before any trials run, `run_bayesian_search()` accepts `cv_strategy` / `cv_n_repeats` and proceeds directly. Result: a bad CV config (e.g. LOO requested with task='classification' and a class with <2 samples) would only fail mid-trial inside the Bayesian objective rather than producing a clean preflight error. Codex flagged this as non-blocking in the PR #4 review. Fix is a one-liner: add `validate_cv_strategy_for_task(...)` at the top of the function, mirroring the other two entry points.
- **Style nits in CV code**: `cv_utils.py` duplicates the `RepeatedKFold` import (module-level + inside `build_cv_splitter`); `templates/validation.py` prints `({cv_folds}-fold)` even for `loo`/`repeated_kfold` (misleading once strategy-specific exports are working); `_majority_vote` should default to NaN in the empty-votes else-branch for safety; LOO `splits = list(...)` materialization at `search.py:~4228` is O(n²) memory for very large datasets (fine for n<5000 spectral data but worth a comment).
- **Suppress sklearn 1.7.2 `"X does not have valid feature names"` UserWarning flood** on `.venv312` — cosmetic noise, ~12MB stderr per grid run, unrelated to any bug. Consider `warnings.filterwarnings(...)` in the GUI entry point.

---

## Active Branch context (secondary)

## Active Branch: `claude/contamination-detection-model-PrntN`

**Goal:** Add one-class contamination detection models as first-class citizens alongside regression and classification. One-class models learn from clean/inlier samples only and flag anything outside that distribution as suspect.

**Models:** OneClassSVM, IsolationForest, LOF (LocalOutlierFactor), EllipticEnvelope, PCA-SIMCA

**Prediction tab manual verification: PASSED 2026-04-11.** User confirmed one-class predictions correctly show inliers when they should after the 2026-04-10 order-of-operations fix. See SESSION_LOG.md 2026-04-10 entry for the original bug trace.

---

## What Works

- [x] **Analysis Subset V1** (branch `glm/analysis-subset-v1`, uncommitted): Pure-logic module `src/spectral_predict/analysis_subset.py` with 41 tests. Analysis tab card, metadata-only dialog with categorical multi-select, subset provenance in training config, mismatch warnings, one-class guardrail, C2 missing-column safe handling.
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
- [x] **One-class hyperparameter grids now exposed in Model Config subtab** — Fixed 2026-04-19. Five collapsible cards in Tab 4C (OneClassSVM, IsolationForest, EllipticEnvelope, LOF, PCA-SIMCA) using old-style BooleanVar + Checkbutton pattern. Untouched cards preserve curated default grids. Customized cards use per-model Cartesian product. Bayesian one-class path unchanged.
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

- **Expose remaining one-class hyperparameters (partial exposure shipped 2026-04-19).** The 2026-04-19 parity work added Tab 4C cards covering the parameters already in the curated `get_one_class_model_grids()` defaults — OCSVM `kernel`/`gamma`/`nu`/`degree`, IF `n_estimators`/`contamination`/`max_features`, EllipticEnvelope `contamination`, LOF `n_neighbors`/`contamination`, PCA-SIMCA `n_components`/`alpha`. Deferred parameters that sklearn supports but the GUI still does not surface: **LOF `metric`**, **IsolationForest `max_samples`**, **OneClassSVM `coef0`** (and additional kernels like `sigmoid`/`linear`), **EllipticEnvelope `support_fraction`**. Also not yet exposed: additional predefined values beyond the shipped curated grid (e.g. OCSVM `nu=0.2`, IF `n_estimators=200/500`, LOF `n_neighbors=50`, PCA-SIMCA `alpha=0.10`) — users can enter these as custom values today, but there's no predefined checkbox. Finally, the one-class **Bayesian** path was deliberately out of scope for 2026-04-19 and still uses the hardcoded parameter ranges in `unified_bayesian.py::suggest_one_class_params()` rather than anything from the new cards; any future work to honor user-customized grids in Bayesian search would need a separate design. Exposing the deferred parameters requires: (a) extending the curated default grid in `get_one_class_model_grids()`, (b) adding checkbox rows to the corresponding Tab 4C card in `spectral_predict_gui_optimized.py::_create_tab4c_model_configuration`, (c) extending the per-model builder in `_resolve_one_class_model_grids()` in `src/spectral_predict/search.py`, and (d) adding default-state comparison entries in `_collect_one_class_model_param_overrides()` so the curated-default preservation invariant still holds. The 2026-04-19 plan's "Non-Goals" section lists these explicitly as future follow-ups.

- **`predict_with_uncertainty` swallows one-class decision_function failures silently.** `model_io.py:854-860`: after `predict_with_model()` returns, the bare `except Exception: decision_scores = None` catches any score-extraction error and returns predictions with empty `uncertainty`/`applicability_domain` payloads. The GUI has no way to surface "predictions worked, uncertainty broke" — exactly the failure mode this PR has already hit once with the order-of-operations bug. Fix: log the exception with `logger.warning(...)` at minimum; ideally return a structured `decision_score_error` flag that the GUI can display alongside the predictions.

- **OC validation helper can't recover the right `polyorder` for 2nd-derivative grid-search rows.** Grid search writes `polyorder=None` (delegating to `polyorder_map` inside `SavgolDerivative` which picks 3 for `deriv=2`), but the validation helper at `contamination.py:922-927` falls back to `min(2, window-1) if window > 2 else 0`, which gives `poly=2` for `deriv=2`. The pipeline then runs with the wrong polyorder and the resulting `val_*` metrics are slightly off vs. what training actually used. Fix: either store the resolved polyorder in the grid-search result dict (most direct), or import the same `polyorder_map` lookup into the validation helper, or have `_maybe_int` fall back to `polyorder_map[deriv]` when poly isn't set.

- **CV Strategy Phase 2: Propagate outer CV strategy into variable selection inner loops.** Currently inner loops (UVE, SPA, iPLS, CARS, GA-PLS) always use hardcoded 5-fold K-fold regardless of the outer CV strategy. A GUI warning is displayed when outer != kfold. Phase 2 would thread `cv_strategy`/`cv_n_repeats` into the variable selection internals.

- **CV Strategy Phase 2: Propagate CV strategy into predictor screening and smart preprocessing.** These internal CV sites are currently hardcoded 5-fold and don't respect the user's chosen strategy.

- **CV Strategy Phase 2: NSGA-II multi-objective search CV strategy support.** Currently falls back to K-fold with a logged warning when non-kfold strategy is selected. Needs native LOO/Repeated K-fold support.

- **One-class search: Add `training_config` to result rows.** Regression/classification grid search writes `training_config` (with `cv_strategy`, `folds`, `cv_n_repeats`) but one-class search does not. This means one-class saved models don't preserve the CV strategy used during training.

## Analysis Subset V1 — Known Limitations / Risk (2026-04-19)

Blocking bugs (stale `active_indices` after dataset replacement or row deletion) fixed in this session. Remaining deferrals:

- **Integration-level GUI tests not yet written:** reload-with-subset, row-delete-with-subset, revert-after-column-delete, and mismatch-warning text. Pure-logic coverage (41 tests in `test_analysis_subset.py`) is solid, but no automated test drives the GUI through these multi-step scenarios.
- **Manual GUI verification checklist still pending:** end-to-end validation that the Analysis-tab card, data-viewer highlighting, provenance in training config, and one-class guardrail all behave correctly after each dataset-replacement path.
- **`_use_for_analysis()` does not carry `combined_metadata_df`** from data sources. If a user loads data via Data Management → Use for Analysis, metadata columns may be absent, causing the subset dialog to show no columns. This is a pre-existing limitation of the data-source path, not introduced by Analysis Subset V1.
