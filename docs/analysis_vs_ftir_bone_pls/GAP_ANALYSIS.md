# dasp vs. FTIR Bone PLS Paper — Gap Analysis

**Date:** 2026-04-29
**Source paper directory:** `C:\Users\mspon\Desktop\_DeskSync\FTIR Bone PLS\` (Sponheimer et al., in prep, "Multivariate spectral methods for ATR-FTIR screening of archaeological bone collagen — a worked demonstration across independently collected datasets")
**Manuscript reviewed:** `paper/manuscript/paper_v5_locked_framing.md` + `paper_v5_locked_framing_v7_supplementary.md`
**Analysis scripts reviewed:** `paper/scripts/analysis/03q_bootstrap_inference.py` and supporting CV/holdout pipelines

---

## Purpose

Maps the statistical analyses in the FTIR Bone PLS paper against dasp's current capabilities. Identifies what dasp can already reproduce, what is partially supported, and what is genuinely missing. Intended as a planning reference for prioritizing future feature work.

The paper is methodologically representative of the modern chemometrics literature: it tests multivariate spectral classifiers against the standard Pal Chowdhury 10-index Random Forest baseline across multiple datasets, multiple CV strategies, and an external-validation holdout. Anything missing here is missing for *any* user who wants to follow current chemometrics best practice on bone-FTIR or comparable spectral classification problems — it is not specific to this paper's biological domain.

---

## Confirmed missing in dasp

These are verified absent from `src/spectral_predict/` by code inspection (grep) at the time of writing.

### 1. Leave-one-group-out (LOGO) cross-validation by site/group
**Paper use:** Pooled LOGO-CV by site with min-group-size filters (≥3, ≥5 specimens). Headline cell of the chemometric panel — pooled LOGO ≥5 sites at 1 wt% threshold gives MCC = 0.636 (LDA on PLS scores) vs index baseline 0.411.

**dasp status:** `cv_utils.py` ships `kfold` / `repeated_kfold` / `loo`. No `LeaveOneGroupOut` or `GroupKFold` is wired into the search/Bayesian path. Verified by grep — only `ga_pls.py` imports `GroupKFold`, and that's an internal use.

**Why it matters:** This is the largest single gap. The paper's transferability claim — "the model generalizes across independently collected datasets" — depends entirely on holding out whole sites/groups during CV. Standard 5-fold CV across pooled spectra is optimistically biased when there are within-site spectral correlations (same instrument, same operator, same sample-prep batch), which is the rule rather than the exception in this literature.

**What's needed:**
- `LeaveOneGroupOut` and `GroupKFold` as CV strategy options in `cv_utils.build_cv_splitter`
- A "group column" concept in metadata (the Analysis Subset V1 work reads metadata categoricals but doesn't pass them to CV as group keys)
- Min-group-size filtering in the data prep step
- Cost-estimator updates to handle variable-fold-count strategies
- GUI exposure on the Analysis tab + Model Development

### 2. Site-level block bootstrap CIs
**Paper use:** 1000 iterations resampling at the SITE level to build percentile CIs around metric *gaps* between two methods (chemometric vs index baseline). Implemented in `paper/scripts/analysis/03q_bootstrap_inference.py:120-160`.

**dasp status:** No bootstrap inference module anywhere in `src/`. Verified by grep — zero matches for `bootstrap_ci` or `block_bootstrap`.

**Why it matters:** Without CIs on the *gap* between two methods, you cannot make a defensible claim that one method is better than another. The paper's headline finding ("MCC gap +0.225, 95% CI excludes zero, p < 0.01") is entirely dependent on this machinery.

**What's needed:**
- A statistical-comparison module that takes two prediction CSVs (or two saved-model outputs) and produces gap CIs
- Both site-level (block) and specimen-level (stratified) bootstrap variants
- GUI exposure as a "Compare two models" workflow

### 3. Paired permutation tests
**Paper use:** 10,000 iterations swapping prediction labels per specimen between two methods to compute two-sided p-values for the metric gap. Same script as #2, function `paired_permutation` at line 180.

**dasp status:** No `permutation_test`, `paired_permutation`, or equivalent in `src/`. Verified by grep.

**Why it matters:** Same as #2 — the paper makes inferential claims about *whether* one method beats another, and that requires p-values on the gap. Bootstrap CIs and permutation p-values are complementary; the paper reports both.

**What's needed:** Same module as #2; the permutation test is one more function alongside the bootstrap.

### 4. Multi-source consensus wavenumber selection
**Paper use:** Six independent V=20 selections compared with band-tolerance matching at ±10 cm⁻¹. Identifies "strict consensus" (all 6 sources agree) — 10 wavenumbers, achieves perfect classification on the holdout. "Relaxed consensus" (≥4 of 6) — 13 wavenumbers, matches the index baseline.

**dasp status:** Each dasp run returns one selection. No mechanism aggregates selections across multiple runs/searches with positional tolerance.

**Why it matters:** This is the paper's primary stability/reproducibility argument — if a wavenumber survives across six independent variable-selection runs, it's almost certainly real signal rather than search noise. It also produces a much smaller, more interpretable model.

**What's needed:**
- A "Run search N times with different seeds, aggregate selected wavenumbers" workflow
- A consensus aggregator with configurable band tolerance (paper uses ±10 cm⁻¹) and configurable agreement threshold (strict = all sources, relaxed = ≥k of N)
- GUI exposure as an option on the variable-selection card or as a post-hoc tool that ingests multiple saved models

### 5. Random-split sensitivity test
**Paper use:** Fix a wavelength set chosen on the KS calibration set, refit the same hyperparameters across 5 random stratified 80/20 splits, report mean ± SD. Used as a stability check on the headline KS holdout result.

**dasp status:** External validation in dasp is single-shot (one train/test split). No "lock these channels, repeat-split N times, summarize" workflow.

**Why it matters:** Reviewers routinely ask "is this just a lucky split?" The paper's Table S4 has a `Random Acc (mean ± SD)` and `Random MCC (mean ± SD)` column for every configuration — that's the kind of evidence that disarms the lucky-split objection.

**What's needed:**
- "Refit fixed channels across N random splits" as a one-click feature on the External Validation tab
- Aggregated reporting (mean ± SD across splits)

### 6. LDA on PLS scores as a distinct classifier
**Paper use:** Best chemometric classifier in 4 of 12 panel cells, including the headline LOGO ≥5 / 1 wt% cell at MCC = 0.636.

**dasp status:** Has PLS-DA (PLS regression projected to one-hot Y, threshold) but not the explicit "fit PLS to extract scores → fit LDA on those scores" recipe as a model option.

**Why it matters:** Statistically these are different procedures. PLS-DA is a regression-based discriminant; LDA-on-PLS-scores is a generative discriminant on the latent factors. They can disagree, especially with class imbalance, and LDA-on-PLS-scores is the more classical chemometrics recipe (Brereton & Lloyd 2014).

**What's needed:** New model card in `models.py` — Pipeline of `PLSRegression(n_components=k)` → `LinearDiscriminantAnalysis()`. Mechanically a small addition.

### 7. Elastic-net logistic regression (ENLR) as a first-class classifier
**Paper use:** Best chemometric in 2 of 12 panel cells. The paper labels it "Logistic regression (EN, balanced)".

**dasp status:** dasp's `ElasticNet` model card uses `sklearn.linear_model.ElasticNet`, which is the **regression** flavor (squared-error loss). The classifier — `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=...)` — is constructed inside `nsga2_search.py` (lines 856, 3077) but is not exposed as a standalone model card the user can pick. The only direct LogisticRegression entry point for classification today is the post-PLS LR inside the PLS-DA pipeline, which uses default L2.

**Why it matters:** ENLR is a workhorse classifier in chemometrics. It gives you sparse coefficient vectors (interpretable) plus a probabilistic output (calibrated uncertainty). The paper's `enlr_coefficient_map.csv` and `enlr_band_coverage.csv` outputs trade on these properties.

**What's needed:** New model card in `models.py` calling `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=..., C=..., class_weight=...)` with `C` and `l1_ratio` exposed as hyperparameters.

### 8. Lin's Concordance Correlation Coefficient (CCC)
**Paper use:** Validates extracted FTIR indices (IRSF, C/P, Am/P, etc.) against published Kontopoulos / Hixon values using Pearson r, Lin's CCC, and RMSE.

**dasp status:** Reports R² and RMSE; CCC is not an option. Verified by grep — no matches for `lin.*ccc`, `concordance`, `lins_ccc`.

**Why it matters:** R² measures linear correlation; Lin's CCC additionally penalizes deviation from the y=x line. For instrument-transfer or method-comparison work where you want to demonstrate that two measurements are *the same*, not just *correlated*, CCC is the standard metric.

**What's needed:** Add `concordance_correlation_coefficient` to `scoring.py` as a metric option. ~10 lines.

### 9. Bayesian regression with uncertainty propagation
**Paper use:** Hixon et al. (2024) used a Bayesian log-linked regression on the A1/PO4 ratio for collagen yield, emphasizing uncertainty propagation rather than a point R². The paper references this as the regression baseline.

**dasp status:** Frequentist point-estimate regression only. No PyMC / Stan / NumPyro probabilistic regression backend.

**Why it matters:** For yield prediction where downstream sample-selection decisions hinge on quantitative thresholds, posterior predictive intervals are more useful than point estimates. This is a research-trend shift in the field, not just an idiosyncratic Hixon choice.

**What's needed:** Significant — would require a probabilistic-programming backend, posterior storage, posterior-predictive plotting infrastructure. Lower priority than the statistical-inference items above.

### 10. Model-native loss reweighting for class imbalance (boosting models + PLS-DA inner LR)
**Paper use:** "XGBoost (scale_pos_weight)", "LightGBM (balanced)", "Logistic regression (EN, balanced)" appear verbatim in Tables S1–S3.

**dasp status:** Imbalance handling exposes `class_weight='balanced'` only on `RandomForest` / `LogisticRegression` / `SVC` (per `code_generator.py:866`). XGBoost / LightGBM / CatBoost construction sites in `models.py` never receive any imbalance-aware kwarg. PLS-DA's inner LR is also stuck at default `class_weight=None`. Verified by grep — zero matches for `scale_pos_weight`, `is_unbalance`, `auto_class_weights`.

**Why it matters:** Resampling (SMOTE etc.) and loss reweighting are *different* statistical interventions. The chemometrics literature defaults to loss reweighting because it preserves real measurements (no synthetic samples, no discarded majority data) — particularly important for small spectroscopic datasets.

**What's needed:** See `docs/plans/2026-04-29-model-native-loss-reweighting.md` for the full design proposal. Per-model "Imbalance handling" dropdown with four options (Equal sample weights / Auto / Class-balanced loss reweighting / scale_pos_weight custom).

---

## Partially supported / needs adaptation

### 11. Stratified Kennard-Stone for cal/test split
**Paper use:** Stratified KS ensures the test set inherits the calibration class balance — critical when classes are imbalanced.

**dasp status:** Kennard-Stone exists (`sample_selection.py`) and is wired to validation-set creation as `_validation_kennard_stone`, but the unstratified variant. Class-stratified KS would need to layer KS-within-each-class on top.

**What's needed:** ~30 LOC addition: stratify by class, run KS within each class, recombine. Expose as a "Stratified Kennard-Stone" option on the validation-split UI.

### 12. Pooled cross-dataset analysis with site provenance preserved
**Paper use:** Two datasets pooled on a common wavenumber grid, with each specimen retaining a `site` / `group` identifier used for #1 and #2.

**dasp status:** Data Management can combine datasets, but a `site` / `group` metadata column is not a first-class CV concept. The Analysis Subset V1 work reads metadata categoricals but doesn't expose them downstream.

**What's needed:** Designate one metadata column as "group column" for CV/bootstrap purposes. Required to make #1 and #2 work end-to-end.

### 13. Per-fold variable selection
**Paper use:** Per-fold VIP scores from a 5-LV PLS regression, refit each CV fold. Per-fold standardization too. Critical for honest CV estimates.

**dasp status:** Variable selection methods often run pre-CV on the full calibration set, which gives mildly optimistic CV estimates. Some paths do correctly nest selection inside the Pipeline (so it refits per fold), others don't. Audit needed.

**What's needed:** Code audit — identify which dasp variable-selection paths leak through `Pipeline.fit` (good — refits per fold) and which compute on the full calibration set then pass selected variables to a downstream model fit (bad — optimistic). Document and fix.

### 14. FIPLS-CARS hybrid
**Paper use:** iPLS-window selection feeding into CARS (paper's `paper/results/within_dataset_cars.csv`, regression headline at R² = 0.72 with 50 wavelengths).

**dasp status:** iPLS and CARS exist as separate variable-selection methods; chaining them as a single hybrid is not a built-in recipe.

**What's needed:** Either expose a "chain selectors" option on the variable-selection card, or add an explicit `iPLS → CARS` hybrid as a named method.

### 15. Refit a saved wavelength set on a new split
**Paper use:** Lock the channels chosen on the KS calibration set, refit hyperparameters across 5 random splits.

**dasp status:** Can save a model and predict on a holdout, but "freeze just the channels, refit on a new calibration set" isn't a one-click workflow.

**What's needed:** "Lock variables" mode for saved models — when applying a saved model to a new dataset, refit the model with the same selected channels but recomputed coefficients on the new data.

---

## Already supported (paper analyses dasp can reproduce today)

For completeness — these are *not* gaps:

- **SG derivatives 1st–4th + SNV in either order.** Paper's headline preprocessing (SG-4thder + SNV w=41 p=5) is fully supported; verified in `preprocess.py:99` (`polyorder_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}`)
- **Classifier panel** (except #6 LDA-on-PLS and #7 ENLR): PLS-DA, SVC-RBF, RandomForest, XGBoost, LightGBM, CatBoost
- **Variable selection methods:** VIP, CARS, iPLS, UVE, SPA, GA-PLS, tree-native importance — all present
- **Standard CV strategies:** 5-fold stratified, K-fold, Repeated K-fold, LOO (note: LOGO is the missing one — see #1)
- **Bayesian hyperparameter search** (used in the paper to identify the headline preprocessing config)
- **Metrics:** MCC, balanced accuracy, F1, accuracy
- **Single-split external validation** holdout
- **Confusion matrix, sensitivity, specificity** for classification
- **PLS regression with R² / RMSE / cross-val R²**
- **Pal Chowdhury 10-index extraction.** dasp's `peak_calculator.py` ships a Bone FTIR preset with local linear baseline correction at user-specified troughs. IRSF, C/P, Am/P, OrgInorg, etc. are all computable in-tool.

---

## Recommended priority for closing gaps

Ranked by leverage on user analytical reach, not by implementation effort:

1. **#1 LOGO / GroupKFold CV + group-column concept.** Unlocks #1 and is a hard prerequisite for #2 and #12. Highest leverage by far — every transfer-learning paper in chemometrics needs this.
2. **#2 + #3 Bootstrap CIs + permutation tests for two-method comparison.** Probably the highest-leverage *new statistical infrastructure* — no other chemometrics tool surfaces these in a usable way. Could be a differentiator.
3. **#10 Model-native loss reweighting.** Already has a design doc (`docs/plans/2026-04-29-model-native-loss-reweighting.md`). Closes a real reproducibility gap with low risk.
4. **#11 Stratified KS + #5 random-split sensitivity + #15 lock-variables refit.** Together these complete the "external validation rigor" story.
5. **#4 Multi-source consensus wavenumber aggregator.** The most novel feature — would actually contribute new tooling to chemometrics, not just parity. Moderate effort.
6. **#6 LDA-on-PLS + #7 ENLR model cards.** Both small additions; close the classifier-panel parity gap.
7. **#13 Per-fold variable selection audit.** Quiet correctness fix; ship before any of the inferential items so reviewers can't object on optimistic-CV grounds.
8. **#8 Lin's CCC.** Tiny addition; do as a drive-by.
9. **#14 FIPLS-CARS hybrid + #12 group-column concept.** Lower priority, niche.
10. **#9 Bayesian regression.** Significant infrastructure investment; defer until the simpler gaps are closed.

---

## Cross-references

- **Loss reweighting design proposal:** `docs/plans/2026-04-29-model-native-loss-reweighting.md`
- **Comparable analysis vs. CAMO Unscrambler:** `docs/analysis_vs_unscrambler/MASTER_ANALYSIS.md` and sibling docs
- **Source paper repo:** `C:\Users\mspon\Desktop\_DeskSync\FTIR Bone PLS\` (manuscript v7 + supplementary, plus all `paper/scripts/analysis/*.py` and `paper/results/*.csv`)
