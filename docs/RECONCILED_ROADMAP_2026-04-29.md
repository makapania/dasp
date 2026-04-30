# Reconciled Roadmap — 2026-04-29

> **Source material:** `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` + six docs in `docs/analysis_vs_unscrambler/`.
>
> **Reviewers:** Claude (project read + code spot-checks), Codex (independent), DeepSeek V4 Pro (fresh eyes, briefed on chemometrics conventions and prior conclusions).
>
> **Source rule:** Per-spectrum preprocessing (SNV, SG derivatives, baseline) is NOT data leakage by chemometrics convention — see `~/.claude/projects/.../memory/feedback_chemometrics_conventions.md`.

This document deduplicates ~240 raw findings from seven analysis docs down to a small set of real tickets, plus a plain-language report explaining the choices.

---

## Status legend

- **P0** — Block real research (correctness or productivity).
- **P1** — Strongly recommended for publication-quality results.
- **P2** — Real but smaller; do in batches.
- **P3** — Drop / defer / out-of-scope.
- **PENDING** — Needs user decision before scoping.

---

## P0 — Science integrity (block real research)

### T-01: Run the per-fold variable-selection leakage audit
**Why it matters:** Three documents recommend the audit; nobody actually performed it. CARS, UVE, VIP, SPA, iPLS may be fitting on full calibration data, then passing a frozen variable list into CV — which produces optimistic CV estimates that won't survive external validation or peer review. DeepSeek points to `search.py:5400-5460` as a likely leakage site.
**Scope:** For each variable-selection method × each search path (grid / Bayesian / NSGA-II / one-class), trace whether it's called inside a sklearn Pipeline (refits per fold, OK) or outside (leaky).
**Effort:** 1-2 days for the audit; separate fix tickets per leak.
**Acceptance:** A short doc listing each varsel method × each search path with leak/clean status.

### T-02: Fix ensemble OOF preprocessor leakage
**Where:** `ensemble.py:315-316, 571-572, 818-819`
**Why:** OOF predictions used for ensemble-weight optimization apply a preprocessor fitted on full training data including the held-out fold. Inflates ensemble CV scores by ~0.01-0.03 R².
**Fix:** Refit preprocessor inside each fold via sklearn `Pipeline`, or document limitation explicitly.
**Effort:** Half day.
**Acceptance:** Ensemble CV scores match independent recomputation with per-fold preprocessing; regression test.

### T-03: Fix preprocessing-discovery full-data leakage
**Where:** `preprocessing_discovery.py:217, 235, 297`
**Why:** Importances used to rank preprocessing pipelines are computed on the full calibration set. The "discovered" preprocessing is selected on leaked information.
**Fix:** Compute importances within CV folds (nested CV) or document the optimism.
**Effort:** 1 day.
**Acceptance:** Synthetic dataset where leaked vs unleaked selection differ confirms unleaked path picks the genuinely-best pipeline.

### T-04: Fix one-class UVE prefilter using outlier-contaminated labels
**Where:** `search.py:5649-5655`
**Why:** UVE called with binary +1/-1 labels covering all samples (inliers + outliers). UVE selects features that distinguish outliers — opposite of one-class philosophy.
**Fix:** Fit UVE on inlier-only y, or skip UVE for one-class.
**Effort:** Half day.
**Acceptance:** PCA-SIMCA / OCSVM with UVE produces variable sets that rank by inlier-stability.

### T-05: Fix VIP formula
**Where:** `models.py:1738`
**Why:** Uses `np.var(y)` instead of per-component `y_loadings_`. Wavelength rankings via VIP are skewed.
**Fix:** Canonical Wold (2001) formula using `y_loadings_`.
**Effort:** 30 min.

### T-06: Fix SPA `n_random_starts` non-functionality
**Where:** `variable_selection.py:401-408`
**Why:** Loops `n_random_starts` times but every start picks `np.argmax(initial_corrs)` — same starting variable. `random_state` unused.
**Fix:** `rng.choice()` for random first-variable starts.
**Effort:** 30 min.

### T-07: Fix PDS even-window arithmetic
**Where:** `calibration_transfer.py:193-215`
**Why:** Even windows overflow `B` allocation.
**Fix:** Enforce odd window or correct arithmetic.
**Effort:** 30 min.

### T-08: Fix CARS tree-mode weight-update bias
**Where:** `variable_selection.py:1519-1522, 1549`
**Why:** Selected variables get tiny weights (~0.01); unselected stay at ~1.0 — sampling biased toward unselected, CARS oscillates instead of converging.
**Fix:** Softmax/exponential weighting consistent with standard CARS.
**Effort:** 1-2 hr.

### T-09: Fix SPC multi-subfile silent data loss
**Where:** `io.py:965`
**Why:** Files with multiple subfiles silently discard subfiles 2+. Thermo OMNIC stacks lose spectra.
**Fix:** Read all subfiles or warn explicitly.
**Effort:** 2-4 hr.

### T-10: Clamp PLS component grid by training-fold sample count
**Where:** `models.py:842-843`
**Why:** Grid uses `min(n_features, max_n_components)` but doesn't clamp by `n_samples_train_per_fold`. Small datasets hit silent fold failures.
**Fix:** `min(n_features, n_samples_train_per_fold, max_n_components)`.
**Effort:** 30 min.

---

### T-32: Fix existing sample_weight length-mismatch bug in resampling path
**Where:** `search.py:3869-3873` (compute) and `:3883` (use)
**Why:** Active bug found by Codex while reviewing T-19. The path computes `sample_weight_train` from `y_train` (pre-resampling) but then passes it to `final_model.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)` against `y_train` rather than `y_train_for_model` (the resampled y). When a resampler is active, the lengths mismatch and the fit either errors or — depending on estimator checks — silently uses wrong-length weights.
**Fix:** Compute `sample_weight_train` AFTER resampling using `y_train_for_model`, OR pass `y_train_for_model` to `fit()`. Verify both resampling-present and resampling-absent paths still work.
**Effort:** 1-2 hr.
**Acceptance:** Run model fit with both SMOTE and class_weight='balanced' enabled on a synthetic imbalanced dataset; verify the model fits without errors and produces sensible metrics.

---

## P0 — Productivity (block real long-running searches)

### T-11: Pause/resume hardening + Optuna SQLite persistence
**Where:** `search_controller.py`, `unified_bayesian.py:1826`
**Why:** User has lost hours on long CatBoost runs. Coarse pause checkpoint, UI lies about pause state, no thread-alive check on Resume, no persistence across crashes.
**Fix:** (a) Optuna `storage="sqlite:///..."` for crash-survivable resume. (b) "Pausing — waiting for current trial" intermediate state. (c) Resume thread-alive check. (d) Per-trial checkpoint inside long objectives.
**Effort:** 3-5 days.
**Acceptance:** Kill GUI mid-search, restart, resume from saved Optuna study.

### T-12: Disk-mirrored logging for long runs
**Where:** GUI `_log_progress` (~line 27770), backend `print()` paths
**Why:** Progress text widget-only with 2000-line cap; bundle has `console=False` so backend prints invisible.
**Fix:** `logging.FileHandler` writing to `outputs/logs/<run-timestamp>.log`. Replace `print()` in `unified_bayesian.py` (47), `search.py` (268+), `calibration_transfer.py` (~66), `io.py` (100+).
**Effort:** 1-2 days for plumbing; print-to-logger replacements incremental.

### T-13: Wire jackknife prediction intervals into GUI
**Where:** `diagnostics.py:143-230` (already implemented), Prediction tab
**Why:** Per-sample prediction intervals coded but never called. Critical for "should I keep this sample?" research questions.
**Effort:** 2-3 days.

### T-14: Hardcoded version-string mismatch
**Where:** `__init__.py:3` (`0.5.0b1`), `report.py:139` (`v0.4.0`), `code_generator.py:373` (`3.9.0`), GUI title
**Fix:** Single version constant.
**Effort:** 30 min.

---

## P1 — Scientific rigor for publication

### T-15: LeaveOneGroupOut / GroupKFold + group-column concept
**Where:** `cv_utils.py:241-258` (factory needs extension), metadata schema, GUI exposure
**Why:** Bone-FTIR data has site/instrument/prep-batch correlations; naive K-fold over-estimates transferability. Without LOGO, transfer claims aren't defensible.
**Fix:** (a) `LeaveOneGroupOut` and `GroupKFold` in splitter factory. (b) Designate metadata column as "group column." (c) Min-group-size filtering with warning that filtering induces conditioning bias. (d) GUI exposure on Analysis tab.
**Effort:** 3-5 days.

### T-16: Bootstrap CIs and paired permutation tests
**Where:** New module `inference.py`
**Why:** Paper claims "MCC gap +0.225, 95% CI excludes zero, p < 0.01" — no inferential infrastructure today.
**Fix:** Site-level block bootstrap (with stratification by site size — DeepSeek's edge case), specimen-level stratified bootstrap, paired permutation on metric gap. GUI workflow ingesting two prediction CSVs.
**Effort:** 4-7 days.

### T-17: **Multi-Y / PLS-2 workflow** *(elevated 2026-04-29 per user request — predicting 2-3 properties simultaneously is core to bone-FTIR research)*
**Where:** Pipeline-wide. `search.py:695` (current `np.ravel()`), scoring, results DataFrame schema, plot generation, prediction tab, code export templates.
**Why:** When Y variables are correlated (collagen yield + IRSF + C/P + Am/P + isotope ratios), joint PLS-2 modeling can be more accurate than separate single-Y models. sklearn's `PLSRegression` already supports multi-output natively — the gap is the workflow plumbing assuming 1D y.
**Scope (substantial):**
1. Multi-target column selector in GUI (Import & Preview / Analysis Configuration)
2. Pipeline accepts DataFrame Y (no `np.ravel`)
3. Per-Y metrics (RMSE, R², bias) AND joint metrics (multivariate Q²) reported in Results
4. Plot generation handles multi-Y (predicted-vs-observed grid, per-Y residuals)
5. Prediction tab outputs multi-column predictions
6. Code export templates handle multi-Y
7. Save/load preserves multi-Y target list
**Effort:** 2-3 weeks of concentrated work. High-leverage but big surface area.
**Acceptance:** Train PLS on (collagen-yield, IRSF, C/P) jointly; predict all three; verify per-Y RMSE matches separate single-Y baselines within tolerance.

### T-18: Stratified Kennard-Stone for cal/test split
**Where:** `sample_selection.py`, validation-split UI
**Why:** Class-imbalanced datasets need class stratification. Current KS doesn't.
**Fix:** ~30 LOC: KS within each class, recombine.
**Effort:** Half day.

### T-19: Model-native imbalance handling (loss reweighting) — full design
**Source design doc:** `docs/plans/2026-04-29-model-native-loss-reweighting.md` — read its REVISIONS section first (it captures the post-Codex corrections to the original body). This ticket is the implementation; both have been revised together against Codex's verification of the actual code state.

---

**Scope correction (verified 2026-04-29 by Codex audit across `src/`):**

PLS-DA inner LR has **~10 construction sites**. Five active sites are missing `class_weight` handling and need to be wired:

| Site | Path | Status |
|------|------|--------|
| `search.py:4257-4261` | main grid CV | ✅ already honors `imbalance_method='class_weight'` |
| `unified_bayesian.py:1219-1222` | unified Bayesian path | ✅ already honors |
| `nsga2_search.py:1417-1426` | `SpectralOptimizationProblem` | ✅ already honors |
| `nsga2_search.py:3077-3080` | NSGA-II classification CV metrics | ✅ already honors |
| `code_generator.py:880-884, 1248` | exported code injection | ✅ already injects `class_weight='balanced'` |
| `search.py:380` | `_rebuild_model_from_row()` | ❌ silent default — **fix** |
| `bayesian_utils.py:400` | importance/full-fit utility | ❌ silent default — **fix** |
| `nsga2_search.py:822` | `_build_model()` PLS-DA branch | ❌ silent default — **fix** |
| `nsga2_search.py:3453` | calibration metrics conversion | ❌ silent default — **fix** |
| `ga_preprocessing.py:428` | GA preprocessing fitness | ❌ silent default — **fix** |

The auxiliary sites (`search.py:380` rebuild-from-row, `bayesian_utils.py:400` importance utility, etc.) are not the main CV training path but they still produce results the user sees (rebuilt models, importance plots, NSGA-II calibration metrics). Silent inconsistency across paths is the failure mode where two of N configurations silently ignore the user's imbalance choice.

**Boosting models (none accept any imbalance kwarg today):** `models.py:285` (XGBoost), `:307` (LightGBM), `:327` (CatBoost) — new wiring needed.

**MLPClassifier:** sklearn `MLPClassifier.fit()` accepts `sample_weight` only as of **sklearn ≥1.7**. Current `pyproject.toml:29` floor is `>=1.5.0`. **Decision required before implementation:** test whether sklearn ≥1.7 builds cleanly in the PyInstaller bundle (`spectral_predict_py312.spec`). If yes, bump the floor and include MLP in balanced-loss support. If no — MLP is among the user's least-used models, so **skip MLP from the imbalance dropdown** as an acceptable tradeoff rather than block T-19 on a sklearn version bump. User's local venv is sklearn 1.8, so the design works locally regardless; the gating question is bundle compatibility on other machines.

**SVC:** already supports `class_weight`; needs UI exposure on a per-model card.

---

**UI design (4 modes per classifier card):**
1. `Equal sample weights (no correction)` — DEFAULT. Tooltip must explicitly say "this is the library default but is rarely the right choice for imbalanced data."
2. `Auto (decide at fit time)` — call `detect_class_imbalance(y, threshold=3.0)` per CV fold; apply balanced if ratio ≥ 3:1.
3. `Class-balanced loss reweighting` — always apply balanced regardless of ratio.
4. `scale_pos_weight (custom)` — binary boosting only (XGBoost / LightGBM / CatBoost binary Logloss). Numeric entry; greyed out for multiclass.

---

**Backend helper:** add `resolve_loss_reweighting(model_name, y_train, mode)` in `src/spectral_predict/imbalance.py` returning the right kwarg dict per model + per task. Centralizes per-library mapping.

---

**Auto-mode + resampling: REVISED design after Codex review**

**WARNING (Codex confirmed):** imblearn's `Pipeline._fit()` does NOT automatically resize or recompute `fit_params` after a resampler step (`imblearn/pipeline.py:435-455`). If you precompute `sample_weight` from the pre-resampling y and pass it via `fit_params` (e.g. `pipe.fit(X, y, classifier__sample_weight=sw)`), the resampler produces a different y' of different length, and the `sample_weight` array will mismatch. This invalidates the original "just route sample_weight via fit_params" plan in the source design doc.

**Mechanically what actually works (Codex verified each):**

| Library | Mechanism | Resampling-aware? | Why |
|---------|-----------|-------------------|-----|
| sklearn `LogisticRegression`, `RandomForestClassifier`, `SVC` | `class_weight='balanced'` | ✅ YES (lazy at fit) | `compute_class_weight` runs from y passed to fit |
| LightGBM `LGBMClassifier` | `class_weight='balanced'` | ✅ YES (lazy at fit) | `lightgbm/sklearn.py:969-976` recomputes from fit-time y |
| CatBoost `CatBoostClassifier` | `auto_class_weights='Balanced'` or `'SqrtBalanced'` | ✅ YES (computed on train pool) | `catboost/core.py:5020-5023` |
| XGBoost binary | `scale_pos_weight=N` (constructor) | ❌ NO (frozen at __init__) | `xgboost/sklearn.py:875` — set once, used as-is |
| XGBoost multiclass / MLP | `sample_weight` via fit() kwarg | ⚠ NEEDS CUSTOM WRAPPER | imblearn doesn't propagate fit_params through resampling |

**Implementation:**

1. **For sklearn / LightGBM / CatBoost** — use the native lazy kwargs. No custom wrapper needed. Automatically resampling-aware.

2. **For XGBoost binary** — use `scale_pos_weight` only when no resampler is in the pipeline. When resampling is active AND user wants balanced for XGBoost, fall back to the wrapper approach (#3).

3. **For XGBoost multiclass and MLP** — use a thin custom estimator wrapper that intercepts `fit(X, y)` and computes `sample_weight = compute_sample_weight('balanced', y)` from the y RECEIVED (post-resampling). The wrapper's `fit()` then calls `inner.fit(X, y, sample_weight=sample_weight)`. This bypasses the imblearn-fit_params problem.

**Existing bug Codex found while verifying this design:** `search.py:3883` in the resampling path computes `sample_weight_train` from pre-resampling `y_train` but passes it to `final_model.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)` against `y_train` (rather than `y_train_for_model`, the resampled y). When a resampler runs, this is an active length-mismatch bug TODAY. Tracked separately as **T-32**.

---

**Other things to wire:**
- **Auto-mode per-fold logging in Results tab** — surface "Imbalance handling: auto → balanced (ratio 4.2:1)" per fold/run.
- **GUI-side hint when both resampling and loss-reweighting are active** — small inline note at the dropdown explaining auto sees post-resampling balance and explicit "balanced" applies on top. Don't grey out; chemometricians legitimately combine (mild undersampling + class_weight on the remaining majority).
- **Code export updates** — `code_generator.py:_render_model_instantiation` (~line 860) currently handles `class_weight='balanced'` for RF/LR/SVC; extend to render boosting kwargs and the wrapper-based sample_weight path so exported scripts reproduce the imbalance handling.
- **Stacking ensemble limitation** — `StackingEnsemble.fit()` (`ensemble.py:796-825`) does NOT route `sample_weight` to base-model `fit()`. Auto-mode sample-weight will NOT propagate into stacking-internal models. Document as a known limitation; users who need balanced loss reweighting in stacked models should choose `class_weight`-style modes for base models that support them.

**Tests:** `tests/test_imbalance_loss_reweighting.py` — auto-applies-when-imbalanced; per-fold correctness under class drift; the wrapper recomputes `sample_weight` from the y it receives (post-resampling); `class_weight='balanced'` works through SMOTE without wrapper; save/load roundtrip with auto; code export includes correct kwarg per model; sklearn floor compatibility check.

**Effort:** **5-7 days** (revised up from earlier 3-5 day estimate after counting: 5 PLS-DA sites + wrapper design + sklearn floor bump + tests for the resampling × balanced paths + code export + GUI exposure).

**Out of scope (separate future ticket):** regression-target imbalance handling — `detect_regression_imbalance` exists at `imbalance.py:133`.

### T-20: Saved-model ↔ exported-script reproducibility test
**Where:** New test in `tests/`
**Why:** Code export is marketed as a strength but no test verifies exported script produces identical predictions to saved model.
**Fix:** Round-trip test covering regression, classification, one-class.
**Effort:** 1 day.

### T-21: Wavelength-spacing uniformity guard for SG derivatives
**Where:** `preprocess.py` SG branch
**Why:** Non-uniform grid produces artefactual peak shifts that look like chemical shifts in publications — silent scientific bug.
**Fix:** `np.diff(wavelengths).std() / mean()` < tolerance check; warn or refuse.
**Effort:** 1-2 hr.

### T-22: Multi-source consensus wavenumber selection
**Where:** New workflow on variable-selection card or post-hoc tool
**Why:** Paper's strict consensus (10 wavenumbers, perfect classification) defends against reviewer "cherry-picked channels" objection.
**Fix:** Run search N times with different seeds; aggregate selected wavenumbers with band tolerance and agreement threshold.
**Effort:** 3-4 days.

---

## P2 — Real but smaller

### T-23: Add ENLR / LDA-on-PLS / PCR / MLR as model cards
**Why:** All small additions. ENLR already constructed in `nsga2_search.py:856` — just needs UI wiring.
**Effort:** Half day for all four.

### T-24: Add Lin's CCC as a metric option
**Where:** `scoring.py`
**Effort:** ~10 LOC.

### T-25: Safe ZIP extraction + UX warning at model load
**Where:** `model_io.py:417, 1676`, GUI load paths (`gui:41606, 44070, 49038`)
**Why:** Zip Slip via `extractall()` without path validation. Per Codex: HMAC alone insufficient (key extractable). Better fix: validate ZIP entry names + UX warning at load.
**Effort:** 1 day.

### T-26: SNV near-zero std tolerance threshold
**Where:** `preprocess.py:43`
**Fix:** `stds[stds < 1e-12] = 1.0`. Log warning when triggered.
**Effort:** 15 min.

### T-27: Encoding handling on file I/O
**Where:** `io.py`, `report.py`, `calibration_transfer.py`, `library_search.py`, `data_management.py`, `instrument_profiles.py`
**Fix:** `encoding='utf-8'` to `open()` and `pd.read_csv()`; consider `chardet` fallback.
**Effort:** 1-2 hr.

### T-28: Lower `min_wavelengths` from 100 to 10
**Where:** `io.py:140, 1864`
**Effort:** 5 min.

### T-29: Replace bare `except:` in scoring.py with NaN + warning
**Where:** `scoring.py:509, 517, 535`
**Effort:** 30 min.

### T-30: Remove debug prints
**Where:** `search.py:2313, 2789, 2834, 3091-3100, 3991-3999`
**Effort:** 15 min.

---

## P1 additional — confirmed 2026-04-29

### T-31: Multi-class SIMCA (true class-modeling, not anomaly detection) — CONFIRMED

**User science context (why this is genuinely needed, not feature creep):**
- Fossil bone differs site-by-site — every site is "really its own thing." Site-classes don't generalize cleanly across new excavations, so a discriminant classifier trained on sites A/B/C and applied to a sample from site D will silently force it into one of A/B/C.
- Consolidants (preservation chemicals applied to specimens) are often of unknown identity. A specimen treated with an unknown consolidant has spectral features the model has never seen. PLS-DA / RF / XGBoost will produce a confident class prediction anyway — silently wrong.
- Both cases are textbook "none of the above" SIMCA use cases: a specimen can legitimately not match ANY trained class.

**Why discriminant classifiers (current dasp) are insufficient:**
- PLS-DA, RF, XGBoost learn boundaries between trained classes. Every test specimen is forced into the closest of A/B/C.
- No "unknown / none of the above" output is possible from the model itself.
- No "this specimen plausibly belongs to multiple classes" output, which matters for samples on a diagenetic continuum.

**What multi-class SIMCA gives that one-class screening + a discriminant classifier doesn't:**
- Per-class membership decisions are independent. A sample can be "in class A" AND "in class B" (continuum), or in neither (novel diagenesis / unknown consolidant). One-class screening only tells you "in/out of training distribution"; SIMCA tells you "in/out of EACH trained class."
- Class-specific applicability domain: which class's PCA model is this specimen consistent with, considering only THAT class's variation?

**Scope:**
- New module `src/spectral_predict/simca.py`: per-class PCA model + per-class T² + Q-residual thresholds + per-class membership predicate.
- New model card on the classification path. Outputs: per-sample × per-class membership decision matrix; auxiliary "any class?" / "multiple classes?" summary columns.
- GUI exposure: results display showing per-class scores AND a "novel/multiple/single" label. Code export.
- Tests: synthetic 3-class data + a "novel" 4th-class holdout; assert SIMCA labels novel-class samples as "no class" and discriminant classifier labels them as something.

**Effort:** 1-2 weeks for a correct implementation + GUI exposure + tests + code export.

**Acceptance:** Train on 3 known sites; predict on a 4th unseen site; SIMCA flags ≥X% of new-site samples as "no class" while PLS-DA forces them into trained classes.

---

## P3 — Drop / defer

| Item | Reason |
|------|--------|
| Internationalization / Japanese UI | Single-PI academic tool. |
| ASTM E1655 compliance templates | Regulated NIR; not your science. |
| Qt/PySide6 GUI migration | 3-6 month project; doesn't help any research outcome. |
| PDF/Word/HTML "regulatory" reports | Markdown + matplotlib + pandoc covers academic publication. |
| Multiblock PLS, MCR-ALS, ASCA, PLS-PM, MSPC | Industrial chemometrics; not your domain. |
| HMAC signing of `.dasp` | Threat model is academic sharing; T-25's warning is enough. |
| Drag-and-drop file loading | UX nicety, no scientific value. |
| Removing "BETA" cosmetic | It IS beta. |
| MATLAB `.mat` import/export | scipy.io ad-hoc when needed. |
| Norris-Williams gap derivatives | SG covers the use case. |
| FIPLS-CARS named hybrid | Paper-specific; chain manually. |
| "Random-split sensitivity test" workflow | 30-line notebook task; T-16 covers same scientific argument. |
| Bayesian regression with PyMC backend | Massive infrastructure. Punt until specific use case. |
| Per-fold full-spectrum normalization "fix" | Per-spectrum SNV/SG/baseline are not leakage by chemometrics convention. |
| 57K-line GUI decomposition as a project | Real tech debt but P3. Don't break working code without specific need. |
| 40% test coverage target | Coverage theatre. Add integration tests for workflows that broke in real use. |

---

# DELIVERABLE 2: Plain-language report

## The big picture

Seven analysis documents in two folders flagged roughly 240 issues. If you turned them all into tickets, you'd never finish. So the job was to cut that pile down to the things that actually matter for your work.

To do this honestly, three independent reviewers from different model families looked at the documents:

1. **Claude (me)** — read everything, pushed back on framing problems, verified key technical claims by reading the actual code.
2. **Codex** — separate model from a different family (OpenAI), given the same task with no context from my review.
3. **DeepSeek V4 Pro** — third model from yet another family (Chinese-trained), given the prior reviewers' conclusions and asked specifically to find what they missed.

Three reviews from three different model families is the strongest "second opinion" possible. When all three agreed, I trusted it. When they disagreed, I picked the strongest argument.

## The framing problem (most important thing to understand)

The seven documents had a hidden bias. They asked the wrong question.

The Unscrambler analysis implicitly assumed the goal is *"compete with a commercial product."* It flagged things like missing PDF reports, missing Japanese-language support, missing pharmaceutical regulatory templates, and a 57,000-line GUI as critical blockers. None of those matter for your work. You're an academic researcher publishing papers, not selling software to pharma companies in Japan.

The FTIR paper analysis implicitly assumed the goal is *"exactly reproduce one paper."* So it flagged everything that paper did but your tool doesn't. Some of that is genuinely valuable (LOGO cross-validation, bootstrap confidence intervals). Some is paper-specific quirks (Lin's CCC as "standard," FIPLS-CARS as a named hybrid).

Neither analysis asked the right question: *what does this researcher need to do their actual science faster, more rigorously, and with results that survive peer review?*

That's the question I used to filter 240 issues into 30 real tickets, plus one pending decision (multi-class SIMCA) and one user-driven addition (multi-Y / PLS-2 prediction).

## The three things I cut hardest

### 1. Commercial-readiness theatre
"Remove BETA from the title," "add Japanese language support," "implement ASTM E1655 templates," "migrate to Qt." Real items in some product's roadmap — not yours.

### 2. Feature-parity-for-parity's-sake
Multiblock PLS. ASCA. MCR-ALS. PLS path modeling. MSPC. Real chemometrics methods used in industries you don't work in. Came up because Unscrambler has them. Don't help bone-collagen science.

### 3. Inflated counts and false alarms
"509 except blocks" — when verified, actual count is roughly half. "Binary specificity bug" — not a bug; the reviewer flipped row/column conventions. "PLS-DA inner LR has no class weighting" — wrong; it does. "Fix it with HMAC signing" — technically insufficient because the key would ship inside the app.

I removed all of these. The tickets you see now survived three rounds of skepticism.

## What I kept, and why

### Bucket A: Real correctness bugs that affect your CV scores (T-01 through T-10)

Silent bugs where the tool produces *plausible* numbers that are *wrong*. The most dangerous kind. Your headline R² or MCC will look fine — but they bias your published results.

**T-01: the audit nobody did.** All three reviewers agreed the analyses kept saying "audit per-fold variable selection" but never did the audit. CARS, UVE, VIP, SPA can leak information if computed on full data before CV. By chemometrics convention, per-spectrum operations like SNV and Savitzky-Golay derivatives are NOT leakage (your domain rule, now in saved memory). But variable selection that uses your y-values is real leakage, and we don't know which paths in dasp do it correctly. T-01 finds out.

**T-02 through T-04** are leakage bugs we already know exist: ensemble OOF preprocessor saw the validation fold; preprocessing-discovery rankings computed on full data; one-class UVE prefilter fed both inliers and outliers (so it picks features that distinguish outliers — opposite of what one-class needs).

**T-05 through T-10** are sharper, smaller correctness bugs: VIP wrong formula, SPA random-starts not random, PDS even-window broken, CARS-tree oscillating, SPC files silently dropping subfiles, PLS components not clamped by fold size.

Not sexy. Half a day to a day each. But every one removes a quiet way for your published numbers to be wrong.

### Bucket B: Productivity (T-11 through T-14)

Your project status mentions a 19-hour CatBoost run where pause/resume silently no-op'd. Progress text isn't logged to disk. The bundled app can't show backend `print()` output. Version string is wrong in three places.

**T-11** (Optuna SQLite persistence) is the single biggest productivity win — your 19-hour run becomes resumable across crashes.
**T-12** (disk-mirrored logging) means you can actually debug what happened.

### Bucket C: Statistical machinery for publication (T-15 through T-22)

Items that turn dasp from "an automation tool" into "a tool that produces publication-defensible results."

**T-15: LOGO / GroupKFold by site.** Your bone-FTIR work pools spectra from multiple sites, instruments, sample-prep batches. Standard 5-fold cross-validation across that pool is optimistically biased — the model partly memorizes within-batch correlations rather than learning chemistry. LOGO holds out whole sites at a time, which is a much harder test and the one a reviewer will demand.

**T-16: Bootstrap CIs and paired permutation tests.** Your paper's headline finding is "the chemometric model beats the index baseline by MCC +0.225, 95% CI excludes zero, p < 0.01." Right now dasp can't compute that confidence interval or that p-value.

**T-17: Multi-Y / PLS-2 workflow.** You confirmed predicting 2-3 properties simultaneously is highly useful. sklearn's `PLSRegression` already supports multi-output natively — the gap is the surrounding pipeline assuming 1D y. This is the largest single-ticket effort on the list (2-3 weeks) but it gives you joint-modeling capability that single-Y workflows can't match when your Y variables are correlated.

**T-18 (stratified Kennard-Stone), T-19 (loss reweighting for boosting models — narrower than the original design doc), T-20 (saved-model reproducibility test), T-21 (wavelength-spacing check), T-22 (consensus channel selection)** all serve the same goal: your publication can survive peer review.

### Bucket D: Polish that's actually quick (T-23 through T-30)

5-minute to 1-day fixes. Some you've already noted in `QUICK_WINS.md`. Worth doing in batches when you need a context switch from larger work.

## What I dropped

The "drop / defer" table cuts about half the original findings. Three reasons each item is there:

1. **It's for a different product** (multiblock PLS for industrial process control; Japanese UI for pharma; ASTM E1655 for grain analyzers).
2. **It's already covered by a smaller fix** (renaming PCA-SIMCA to DD-SIMCA disambiguates the one-class detector; multi-class class-modeling SIMCA is now T-31 above as a real P1 ticket per user science context).
3. **It violates your domain conventions** (per-spectrum SNV / SG / baseline applied before splitting is not leakage by chemometrics community convention; saved to memory).

## The two questions I needed your input on

### Multi-class SIMCA (T-31 PENDING)

Your existing multi-class classifiers (PLS-DA, RF, XGBoost) are *discriminant* classifiers — they force every specimen into one of the trained classes. SIMCA is a *class-modeling* approach where a specimen can belong to one class, multiple classes, or none.

If your bone-FTIR work occasionally encounters specimens that genuinely don't fit any trained class (contaminated samples, unusual diagenesis, novel preservation states), multi-class SIMCA gives you a "this is unlike anything I was trained on" output that PLS-DA cannot. That's a real scientific capability.

If your specimens are reliably one of N known classes, your existing classifiers handle the work fine.

**Action:** confirm whether the "none of the above" / "could be multiple" output is useful for your science before T-31 becomes a tracked ticket.

### PLS-2 / multi-Y prediction (T-17, ELEVATED)

You confirmed this is highly useful. Predicting 2-3 properties simultaneously when those properties are correlated (e.g., collagen yield + IRSF + C/P + isotope ratios) is more accurate via joint PLS-2 than via separate single-Y models. The model exists in sklearn; the gap is the workflow surrounding it.

**Action:** elevated to P1 in the ticket list above. Substantial effort (2-3 weeks) but high-leverage.

## My recommended top-5 first

1. **T-01** (run the per-fold variable-selection audit) — until you know which paths leak, you can't trust your CV scores.
2. **T-15** (LeaveOneGroupOut by site) — your paper's transferability claim depends on it.
3. **T-11** (pause/resume + SQLite) — a 19-hour run shouldn't be lost to a crash.
4. **T-16** (bootstrap CIs and permutation tests) — your paper's inferential claims depend on these.
5. **T-12** (disk-mirrored logging) — debugging long runs is currently impossible.

Everything else can wait. These five, in some order, unblock both your immediate research and the paper you're trying to publish. T-17 (multi-Y) is the biggest follow-on and should slot in after T-01-T-04 are clean.

---

## Cross-references

- **Source analyses:** `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md`, `docs/analysis_vs_unscrambler/*.md`
- **Quick wins (smaller pre-existing list):** `docs/QUICK_WINS.md`
- **Loss reweighting design doc:** `docs/plans/2026-04-29-model-native-loss-reweighting.md` (scope needs revision per Codex finding — PLS-DA inner LR already wired; only boosting models need new work)
- **Pause/resume gaps:** see `Follow-Ups (unclaimed)` section in `docs/PROJECT_STATUS.md`
- **Chemometrics-conventions rule (auto-memory):** `feedback_chemometrics_conventions.md` — per-spectrum operations are not leakage
