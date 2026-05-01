# T-16 findings: Model-vs-model comparison machinery (broadly reframed)

**Branch:** none yet — T-16 is **not implemented**. Investigation of main + competitor docs + literature.
**Status:** FINDINGS ONLY — verdict pending main agent
**Author:** Opus 4.7 (1M context), 2026-04-30

The user has reframed T-16 broadly. The original roadmap entry framed it
narrowly as "block bootstrap CIs + paired permutation tests" because the
FTIR Bone PLS paper used those specifically. The user now says: *"ways of
really comparing between models are warranted and have to be looked at
from a competitive framework. there was jackknifing and permutations and
i don't know what we should do, but what our competitors do is a head
start."* T-15 (LOGO) has been dropped, so T-16 is now standalone.

This document is a structured catalog: what dasp has, what competitors
expose, what the canonical chemometrics/ML literature recommends, what
the methodological options are, and what the smallest-defensible-shape
candidates look like. **No verdict, no recommendation.**

---

## TL;DR

**Competitor parity (one line each).** The big four chemometrics-GUI
competitors expose model-vs-model and model-validity inferential
machinery to varying degrees:

| Tool | Bootstrap CIs | Permutation / Y-randomization | Jackknife on coefficients | Two-model paired test |
|------|---------------|-------------------------------|---------------------------|----------------------|
| **The Unscrambler X / Aspen** | Not as a primary feature | Not surfaced | **YES — "Uncertainty Testing"** (Martens & Martens jackknife, native flagship feature) | NO |
| **PLS_Toolbox / Solo** (Eigenvector) | NO (boot CIs not a primary GUI feature) | **YES — "Permutation Test"** in Analysis GUI (Y-randomization for overfit detection) | Not as a coefficient-significance feature | NO |
| **SIMCA / Sartorius** | NO | **YES — permutation test for OPLS/PLS validity** (Q² distribution under permuted Y) + **CV-ANOVA** (analytical p-value on Q²) | NO | NO |
| **OPUS / Bruker QUANT** | NO | NO | NO | NO — only "Cross Validation" vs "Test Set Validation" |
| **caret (R)** | YES (in `rsample`) | NO (not its scope) | NO | **YES — `compare_models()` paired t-test, `diff.resamples()` t-test/Wilcoxon** |
| **mlr3 / mlr3benchmark (R)** | NO (in core; rsample-style is via mlr3resampling) | NO | NO | **YES — Friedman + Nemenyi post-hoc, Bonferroni-Dunn (Demšar 2006 critical-difference plots)** |
| **tidymodels (R)** | **YES — `int_pctl()`, `int_t()`, `int_bca()`** on `tune_results` for any metric | NO | NO | YES (via `tidyposterior`/Bayesian; not in core) |
| **mdatools (R)** | NO | NO | YES — coefficient jackknife (chemometrics-canonical) | NO |

**Reading.** Two distinct competitor traditions exist, and neither
covers what dasp's user is asking for:

1. **Chemometrics tools** (Unscrambler, SIMCA, PLS_Toolbox) ship
   *single-model* inferential machinery — Y-permutation for "is this
   model better than chance", jackknife for "are these coefficients
   non-zero". They do **not** ship two-model paired comparison features.
2. **General ML frameworks** (caret, mlr3, tidymodels) ship *two-model
   and N-model* paired comparison features — `compare_models()`,
   Friedman + Nemenyi, bootstrap CIs on metrics. They do **not** ship
   the chemometrics-canonical Y-permutation or coefficient jackknife.

**Canonical literature recommends** (with citations in §4):

- **Two classifiers, one dataset:** Wilcoxon signed-rank on per-fold
  errors (non-parametric; Demšar 2006). Or 5×2 CV paired t-test
  (Bouckaert & Frank 2004 measure replicability and find sorted-runs
  + t-test best). Or paired t-test on resampled metric differences
  (Hothorn et al. 2005 — caret's foundation).
- **Multiple classifiers, multiple datasets (or repeats):** Friedman
  test + Nemenyi post-hoc (Demšar 2006). This is what mlr3 and the
  ML community settled on as the canon.
- **Coefficient significance for PLS:** jackknife (Martens & Martens
  2000; Westad & Martens; the Unscrambler "Uncertainty Test" canon).
- **Model validity (single model is real, not chance):** Y-permutation
  test (the SIMCA / OPLS canon; PLS_Toolbox ships this).
- **Bootstrap CIs on metrics:** Efron & Tibshirani 1993; rsample
  `int_pctl()` / `int_bca()` is the modern reference implementation.
- **Block bootstrap for site/group structure:** Politis & Romano 1994
  (general); used in the user's FTIR Bone PLS paper for site-level
  resampling.
- **Westad & Marini 2015** (the canonical chemometrics-validation
  tutorial) recommends: external test set first (n>50); for CV-only
  small-data regime, segmented CV + permutation tests for validity +
  jackknife for parameter uncertainty.

**Smallest-defensible-shape candidates (no recommendation, just shapes
the verdict-writer can pick from):**

- **Shape A — single-model uncertainty only (chemometrics-conservative).**
  Add bootstrap CIs to existing per-row metrics in the Results table
  ("RMSEcv = 0.42 [95% CI: 0.38, 0.47]"). Add Y-permutation test as a
  one-click validity check. This matches what every chemometrics
  competitor ships and addresses 80% of "is my model real" questions.
  No new comparison UI needed. Effort: ~3–5 days.

- **Shape B — two-model paired comparison (caret-style).**
  Adds a "Compare two models" dialog launched from the Results table
  (right-click two rows → Compare). Runs paired t-test + Wilcoxon
  signed-rank on the per-fold metrics. Returns p-value + 95% CI on
  the metric difference. Matches caret/Hothorn 2005 conventions. Does
  NOT match what chemometrics competitors expose, but matches what
  the user's paper actually needs. Effort: ~3–5 days.

- **Shape C — paper-aligned (FTIR Bone PLS specific).**
  Site-level block bootstrap + paired permutation on metric *gaps*
  between two methods. This is what the user's paper uses
  (`paper/scripts/analysis/03q_bootstrap_inference.py`). Requires a
  group/site column concept (T-15 was dropped, but block bootstrap
  needs *some* group structure to bootstrap over). Most rigorous for
  publication-defensible transferability claims. Effort: ~5–7 days
  if T-15-style group infrastructure is added inline; ~3–4 days as a
  bare module if the user supplies the group vector externally.

- **Shape D — full Demšar-style multi-model panel.**
  When the user has selected N>2 rows in Results, run Friedman +
  Nemenyi post-hoc and produce a critical-difference plot. Matches
  mlr3benchmark. Most ambitious; provides functionality no commercial
  chemometrics tool offers. Effort: ~7–10 days incl. plotting.

- **Hybrid possibilities** between shapes are natural (e.g., A + B for
  "single-model CI in Results table + a Compare dialog with paired
  test"; A + C for "auto CI in Results + a paired-permutation
  workflow"). The verdict-writer is the one who picks.

---

## 1. Reality in dasp's codebase (Step 1)

### 1a. `scoring.py` — what's there today

**`src/spectral_predict/scoring.py:1-654`** — the whole file. Functions
exposed:

- `compute_composite_score()` — ranks model rows by performance score
  + variable penalty + gap penalty.
- `_compute_unified_complexity()` — internal scoring component (0-100
  complexity index).
- `compute_specificity()` — TN-rate metric.
- `lins_ccc()` — Lin's CCC (just merged via T-24).
- `create_results_dataframe()` — schema for the Results table:
  metric columns are `RMSE / R2 / RMSEcv / R2cv / MAEcv / RPD / Bias /
  RER / CCC / CCCcv` (regression), `Sensitivity / Specificity /
  Precision / F1 / Accuracy / BalancedAcc / AUC` (one-class), or
  `Accuracy / ROC_AUC / F1 / Precision / Recall / Specificity / Kappa /
  MCC / BalancedAcc / BER / LogLoss` (classification), each with a
  `cv` suffixed CV variant.
- `add_result()` — append a row.
- `compute_imbalance_metrics()` + `print_imbalance_metrics_report()` —
  imbalance-aware classification metrics.

**Crucially absent from `scoring.py`:** any CI columns, p-value
columns, bootstrap, permutation, jackknife, AIC, BIC, paired
comparison, or significance test. The schema is point-estimate only.

### 1b. Bootstrap / permutation / jackknife / paired test elsewhere in `src/`

Grepped the `bootstrap|jackknife|permutation|paired_t|wilcoxon|friedman|
nemenyi|aic|bic|delong|model_compare|compare_model|paired_perm` patterns
across `src/spectral_predict`. Hits:

- `src/spectral_predict/diagnostics.py:143-230` —
  `jackknife_prediction_intervals(model, X_train, y_train, X_test,
  confidence=0.95)`. **Implements the chemometrics-canonical jackknife
  prediction interval** for regression PLS-style models. Returns
  `(predictions, lower_bounds, upper_bounds, std_errors)`.
- `src/spectral_predict/search.py` — uses of `bootstrap` are RandomForest
  hyperparameter `bootstrap=True/False`, not statistical bootstrap.
- `src/spectral_predict/models.py`, `nsga2_search.py`,
  `neural_boosted.py`, `ga_pls.py`, `ga_preprocessing.py`,
  `learned_preprocessing.py`, `model_config.py`, `bayesian_config.py`,
  `contaminant_analysis.py` — all uses of `bootstrap` /
  `bootstrap_type` are RF/CatBoost/XGBoost hyperparameters (Bayesian
  bootstrap inside CatBoost, RF bootstrap-sampling toggle, etc.). Not
  inferential bootstrap.

**No permutation test, no paired t-test, no Wilcoxon, no Friedman, no
AIC/BIC, no DeLong test, no compare_models() function anywhere in
`src/`.** The codebase has zero inferential model-comparison
infrastructure beyond the already-existing jackknife prediction
intervals in diagnostics.py.

### 1c. The jackknife at `diagnostics.py:143` is GUI-orphaned

**`spectral_predict_gui_optimized.py`** — grepped for `jackknife`. Zero
matches in the GUI file (verified, confirmed by `Grep` against the
57k-line file). The function exists but no GUI button calls it.

This was already flagged in the roadmap as **T-13 "Wire jackknife
prediction intervals into GUI"** (KEEP, P0 productivity). T-13 is
*one* application of the existing jackknife code; T-16 could
potentially share infrastructure with T-13 but is a different feature
(per-sample prediction intervals vs. per-model metric uncertainty).

### 1d. The Results tab is point-estimate-only

**`spectral_predict_gui_optimized.py:14218-14400`** —
`_create_tab6_results()`. The Results tab is a sortable Treeview
showing one row per (model × preprocessing × variable-selection)
config. Columns are the metric columns from
`create_results_dataframe()` plus `Rank`, `CompositeScore`,
`ComplexityScore`, `top_vars`. Behaviors visible to the user:

- Sort by any column, multi-level sort (Shift+Click).
- Quartile-based highlighting (top models per Y-value region).
- Class-based highlighting (top models per class for classification).
- Overfit tag (gray text + overstrike for models where calibration ≫
  CV).
- Double-click → load result into Model Development tab for tuning.

**No CI columns, no p-value comparison, no "Compare two rows"
context-menu action.** When a user wants to know "is method A's
RMSEcv = 0.42 actually better than method B's RMSEcv = 0.45?", the
only signal is the sort order — there is no inferential answer.

### 1e. The Multi-Model Comparison tab is NOT statistical comparison

**`spectral_predict_gui_optimized.py:47694-48068`** —
`_create_tab9_multi_model_comparison()` exists but the name is
misleading. It is a **prediction-time multi-model dashboard**: load N
saved models, load a new spectrum (or directory of spectra), run
predictions through all models simultaneously, show side-by-side
predicted values + flagging rules ("if X > threshold, flag"). It is
designed for screening workflows like "primary model predicts
collagen yield; auxiliary models flag consolidant contamination".

It does NOT compute statistical comparison metrics (no CIs, no
p-values, no paired tests). It is a deployment / screening UI, not a
methodology UI.

The name overlap with the user's reframed T-16 is incidental but worth
flagging: **if T-16 ships, naming will need to disambiguate** (e.g.,
"Statistical Comparison" or "Method Significance" tab vs. the existing
"Multi-Model" prediction dashboard).

### 1f. Code-export templates do not include comparison statistics

**`src/spectral_predict/templates/*.py`** — `header.py`,
`preprocessing.py`, `variable_selection.py`, `models.py`,
`validation.py`, `visualization.py`. Grepped for `compare`, `Compare`
— zero matches. Templates produce single-model standalone Python
scripts; they do not produce comparison reports.

### 1g. Implications for T-16

The codebase has:

- A jackknife prediction-interval engine in `diagnostics.py:143-230`
  that's GUI-orphaned (would inform T-13 but probably not T-16
  directly — different statistical question).
- Zero metric-CI infrastructure.
- Zero pairwise-comparison infrastructure.
- A misleadingly-named "Multi-Model" tab that's actually a prediction
  dashboard (would need to rename or carefully scope T-16's UI).

T-16 is a green-field addition. Whatever shape ships will be the
codebase's first inferential machinery; nothing to preserve, nothing to
break.

---

## 2. Competitor coverage (Step 2)

The detailed catalog. Citations are URLs to the relevant docs.

### 2a. The Unscrambler X / Aspen Unscrambler (CAMO / Aspen)

**Flagship feature: "Uncertainty Testing"** — a jackknife on PLS
regression coefficients.

[Spectroscopy Europe — Uncertainty testing in PLS regression](https://www.spectroscopyeurope.com/td-column/uncertainty-testing-pls-regression):

> "The method was first called 'The Jacknife' but this terminology has
> been used elsewhere in statistics so it may be better to call it
> 'Uncertainty testing'."
>
> "[Uncertainty Testing] retaining estimates of the regression
> coefficients from cross-validation iterations to estimate the
> variance of the coefficients and hence test if they were
> significantly different from zero."

This is **single-model** coefficient-significance machinery, derived
from Martens & Martens 2000 (*Modified Jack-knife estimation of
parameter uncertainty in bilinear modelling by partial least squares
regression*, Food Quality and Preference 11). The output: per-channel
significance markers ("which wavelengths' coefficients are
non-zero?"), a coefficient CI, and a variable-selection workflow
(iteratively drop non-significant variables → smaller model).

**What Unscrambler does NOT expose:**
- No two-model paired comparison.
- No bootstrap CIs on metrics (RMSEP / R² / Accuracy).
- No Y-permutation test as a primary feature.
- No multi-model panel (Friedman / Nemenyi).

**Sources:**
- [Spectroscopy Europe — Uncertainty Testing](https://www.spectroscopyeurope.com/td-column/uncertainty-testing-pls-regression)
- [Martens & Martens 2000 — Modified Jack-knife (FQP)](https://www.sciencedirect.com/science/article/abs/pii/S0950329399000397)
- [Westad & Martens — Variable Selection in NIR based on significance testing in PLSR](https://www.researchgate.net/publication/244738666_Variable_selection_in_NIR_based_on_significance_testing_in_Partial_Least_Squares_Regression_PLSR)

### 2b. PLS_Toolbox / Solo (Eigenvector Research)

**Flagship feature: "Permutation Test"** — Y-randomization for overfit
detection.

[Eigenvector wiki — Tools: Permutation Test](https://wiki.eigenvector.com/index.php?title=Tools%3A_Permutation_Test)
(redirected to www.eigenvectordocs.com — direct fetch returns 403 but
search-result excerpts are quoted faithfully):

> "Permutation tests involve repeatedly and randomly reordering the
> y-block, rebuilding the model with the current modeling settings
> after each reordering. Permutation tests are a way to help identify
> an overfit model as well as provide a probability that the given
> model is significantly different from one built under the same
> conditions but on random data."
>
> "The probability table shows probabilities that the predictions for
> the original, unperturbed model could have come from random chance,
> or in other words, probabilities that the original model is not
> significantly different from one created from randomly shuffling the
> y-block."
>
> "Permutation test was added to the Analysis GUI to allow easy
> testing for over-fit regression models with randomized y-block. The
> test provides results using multiple statistical methods and can
> examine both self-prediction residuals and cross-validated
> residuals."

This is **single-model validity** machinery: "is my model better than
random chance, or am I just overfitting?" Output: p-values comparing
original-Y residuals to permuted-Y residuals.

**What PLS_Toolbox does NOT explicitly expose** (per available
documentation):

- No two-model paired comparison.
- No bootstrap CIs on metrics as a primary GUI feature.
- No coefficient jackknife (Unscrambler's territory).
- No Friedman / Nemenyi multi-model.

**Sources:**
- [Eigenvector — Permutation Test (search-result excerpts)](https://wiki.eigenvector.com/index.php?title=Tools%3A_Permutation_Test)

### 2c. SIMCA / Sartorius (Umetrics)

**Flagship features: Permutation test + CV-ANOVA** for model validity.

[Sartorius SIMCA app note — Multivariate Calibration in SIMCA of Spectroscopic Data](https://www.sartorius.com/download/545668/simca-appnote3-spectroscopydata-en-b-00061-sartorius-data.pdf)
+ [PLS/OPLS validation — RSC Mol. BioSyst. 2015](https://pubs.rsc.org/en/content/articlelanding/2015/mb/c4mb00414k):

> "SIMCA proposes many parameters or tests to assess the quality of
> the computed model (the number of significant components, R², Q²,
> pCV-ANOVA, and the permutation test)."
>
> "[The permutation test] consists in comparing the Q² obtained for
> the original dataset with the distribution of Q² values calculated
> when original Y values are randomly assigned to the individuals."
>
> "For model validation, cross-validation is applied to determine
> model dimensionality and permutation testing for further validation
> of the final model in SIMCA."

So SIMCA exposes:

1. **Permutation test (Y-randomization)** — same purpose as
   PLS_Toolbox's. Single-model validity. Q² distribution under null.
2. **CV-ANOVA** — analytical ANOVA-style p-value on Q² (not
   permutation-based). Single-model.

**Critical caveat from peer-reviewed literature** (RSC 2015):

> "A simple permutation of dataset rows can, in several cases, lead to
> contradictory conclusions about the significance of the models when
> a K-fold cross-validation is used. When Q² values lower than 0.5 are
> obtained, SIMCA users should at least verify that the quality
> parameters are stable towards permutation of the rows in their
> dataset."

So SIMCA's permutation test is documented but has known failure modes
that practitioners have published on. dasp implementing the same
feature would inherit the same caveats.

**What SIMCA does NOT expose:**
- No two-model paired comparison.
- No bootstrap CIs on metrics.
- No coefficient jackknife as a flagship feature.
- No Friedman / Nemenyi.

**Sources:**
- [Sartorius — Multivariate Calibration in SIMCA app note](https://www.sartorius.com/download/545668/simca-appnote3-spectroscopydata-en-b-00061-sartorius-data.pdf)
- [SIMCA 15 User Guide](https://www.sartorius.com/download/544940/simca-15-user-guide-en-b-00076-sartorius-data.pdf)
- [PLS/OPLS validation — Mol. BioSyst. 2015](https://pubs.rsc.org/en/content/articlelanding/2015/mb/c4mb00414k)

### 2d. OPUS / Bruker QUANT

**No inferential machinery.** Per [OPUS QUANT manual](https://files.nocnt.ru/hardware/science/senterra/opus65-doc-en/quant.pdf)
+ [Bruker Quantification page](https://www.bruker.com/en/products-and-solutions/infrared-and-raman/opus-spectroscopy-software/quantification.html):

Validation = "Cross Validation" or "Test Set Validation". RMSEP and R²
are reported as point estimates. **No CIs, no p-values, no model
comparison test, no permutation, no jackknife.** OPUS is an
instrument-vendor calibration-deployment tool; methodological
inference is out of scope by design.

### 2e. Galactic GRAMS / Thermo OMNIC

Inconclusive — vendor-tied, docs not freely searchable. Spectroscopy
software with PLS plugins (GRAMS IQ family). Same workflow class as
OPUS (instrument calibration deployment); unlikely to expose model
comparison methodology. **Not investigated further** given API budget.

### 2f. mdatools (R, Kucheryavskiy)

The open-source SIMCA reference for chemometrics. From `simca()` and
`crossval()` reference docs (cited in T-15 findings §3e):

- `cv = 'loo' / 'rand' / 'ven' / N` — segmented CV.
- Coefficient jackknife is exposed (chemometrics canonical).
- **No bootstrap CIs on metrics as a primary feature.**
- **No Y-permutation test.**
- **No paired model comparison.**

mdatools is intentionally minimalist re: inference; the community
relies on `pls` package's `jack.test()` ([rdrr.io/cran/pls/jack.test](https://search.r-project.org/CRAN/refmans/pls/html/jack.test.html))
for jackknife coefficient testing.

### 2g. caret (R)

The classical R ML framework, widely used by chemometricians for
non-PLS modeling.

[`diff.resamples()` and `compare_models()` reference](https://rdrr.io/cran/caret/man/diff.resamples.html):

> "Methods for making inferences about differences between models."
>
> "compare_models() is a shorthand function to compare two models
> using a single metric. It returns the results of t.test on the
> differences."
>
> "[diff.resamples] for each metric being compared, all pair-wise
> differences are computed and tested to assess if the difference is
> equal to zero. The output of this function should have scalar
> outputs called `estimate` and `p.value`."
>
> "The ideas and methods here are based on Hothorn et al. (2005) and
> Eugster et al. (2008)."

So caret exposes:

1. **`compare_models(modA, modB)`** — paired t-test on the per-fold
   metric differences between two trained models. Returns p-value +
   CI on the mean difference. ~3-line R call.
2. **`diff.resamples(resamples_object)`** — N-model pairwise
   differences, with multiple-comparison correction (Bonferroni
   default; `p.adjust()` integration).

caret does NOT expose Y-permutation or coefficient jackknife —
ML-framework conventions, not chemometrics ones.

### 2h. mlr3 / mlr3benchmark (R)

The modern R ML successor to mlr.

[mlr3 benchmark — Evaluation chapter](https://mlr3book.mlr-org.com/chapters/chapter3/evaluation_and_benchmarking.html):

> "The aggregated performances just give a quick impression which
> approaches work well and which approaches are probably
> underperforming. However, the aggregates do not account for variance
> and cannot replace a statistical test. ... Analysis of benchmark
> experiments, including statistical tests, is covered in more detail
> in Section 11.3."

[mlr3benchmark CRAN docs](https://cran.r-project.org/web/packages/mlr3benchmark/refman/mlr3benchmark.html)
+ [autoplot.BenchmarkAggr reference](https://mlr3benchmark.mlr-org.com/reference/autoplot.BenchmarkAggr.html):

> "Critical difference plots (Demsar, 2006) display learners on the
> x-axis according to their average rank with the best performing on
> the left and decreasing performance going right. Any learners not
> connected by a horizontal bar are significantly different in
> performance."
>
> "The 'fn' plot type plots post-hoc Friedman-Nemenyi by first calling
> BenchmarkAggr$friedman_posthoc and plotting significant pairs in
> coloured squares and leaving non-significant pairs blank, useful for
> simply visualising pair-wise comparisons."
>
> "For critical difference plots, critical differences are either
> computed between all learners (test = 'nemenyi'), or to a baseline
> (test = 'bd'). Bonferroni-Dunn usually yields higher power than
> Nemenyi as it only compares algorithms to one baseline."

So mlr3benchmark exposes:

1. **Global Friedman test** — "is any learner different from any
   other?" non-parametric.
2. **Nemenyi post-hoc** — pairwise comparison after Friedman is
   significant.
3. **Bonferroni-Dunn variant** — comparison vs. a single baseline (more
   power).
4. **Critical-difference plots** (Demšar 2006) — visualization.

mlr3benchmark explicitly cites Demšar 2006 as its methodological
foundation. This is the strongest "we ship Demšar's recipe out of the
box" treatment of any tool I surveyed.

### 2i. tidymodels (R)

The next-generation R ML framework. Bootstrap CIs are first-class.

[rsample Bootstrap CIs vignette](https://rsample.tidymodels.org/articles/Applications/Intervals.html)
+ [int_pctl reference](https://rsample.tidymodels.org/reference/int_pctl.html)
+ [tune::int_pctl.tune_results](https://tune.tidymodels.org/reference/int_pctl.tune_results.html):

`int_pctl()`, `int_t()`, `int_bca()` — three flavors of bootstrap CI:

- **Percentile** — standard, easy, needs many resamples.
- **Studentized t** — fewer resamples, needs variance estimate.
- **BCa (bias-corrected accelerated)** — most accurate, computationally
  expensive.

These are exposed at the `tune_results` level, so you can ask "give me
a 95% CI on RMSE for this fitted model" or "give me a 95% CI on the
metric difference between model A and model B" on the same
infrastructure.

For two-model comparison specifically, tidymodels recommends the
companion package `tidyposterior` (Bayesian comparison via Stan). But
the simpler `int_pctl()` + per-fold differences route is also
documented.

### 2j. Python ecosystem

**chemotools (Python):** Not investigated in detail; search returned
no specific model-comparison machinery. Likely point-estimate only.

**mlxtend (Python):** Has `mlxtend.evaluate.paired_ttest_5x2cv` —
implements the Bouckaert & Frank 2004 + Dietterich 1998 5×2 CV paired
t-test directly. Also `combined_ftest_5x2cv`. Not chemometrics
specific. Source: standard mlxtend documentation.

**scikit-learn (Python):** Does NOT ship a paired classifier
comparison test in core. Recommends the user roll their own from
`scipy.stats` (Wilcoxon, t-test) on per-fold scores from
`cross_val_score`. The "Compare cross validation strategies" pattern
is example-only, not a first-class API.

**scipy.stats (Python):** `wilcoxon`, `ttest_rel`, `friedmanchisquare`
— the building blocks. dasp would use these directly if shipping
shape B or D.

### 2k. Competitor summary table

Two distinct traditions surface:

| Capability | Chemometrics tools (Unscrambler, SIMCA, PLS_Toolbox) | ML frameworks (caret, mlr3, tidymodels) |
|------------|----------------------------------------------------|----------------------------------------|
| Bootstrap CI on metrics | NO | YES (tidymodels rsample) |
| Y-permutation test (single-model validity) | YES (PLS_Toolbox, SIMCA) | NO (not their convention) |
| Coefficient jackknife (PLS) | YES (Unscrambler "Uncertainty Test", mdatools, R `pls`) | NO (not their convention) |
| Paired t-test on per-fold metrics (two-model) | NO | YES (caret compare_models) |
| Wilcoxon signed-rank (two-model) | NO | YES (caret diff.resamples test='wilcox') |
| Friedman + Nemenyi (multi-model panel) | NO | YES (mlr3benchmark) |
| 5×2 CV paired t-test | NO | YES (mlxtend Python; Bouckaert & Frank 2004) |
| AIC / BIC | NO (irrelevant for non-nested PLS) | YES (lme4, base R) |
| DeLong test (AUC) | NO | NO (specialized; pROC R package only) |

**dasp's position:** Currently has **none** of these. Adding even one
catches dasp up to a tradition it doesn't currently participate in.
Adding both traditions (bootstrap CIs + Y-permutation + paired test)
puts dasp ahead of any single competitor on cross-tradition coverage.

---

## 3. Canonical chemometrics + ML literature (Step 3)

### 3a. Westad & Marini 2015 — chemometrics-validation tutorial

**Westad, F. & Marini, F. (2015).** *Validation of chemometric models —
A tutorial.* Anal. Chim. Acta 893:14–24.
[ScienceDirect DOI](https://www.sciencedirect.com/science/article/abs/pii/S0003267015008922)

Per the abstract / available extracts:

> "This tutorial focuses on validation from both a numerical and
> conceptual point of view. ... the commonly applied procedure of
> randomly dividing a dataset into calibration and test sets must be
> used with care and can only be justified when there is no systematic
> stratification of samples that would affect validated estimates."

The tutorial is *the* chemometrics-validation reference and
distinguishes between:

1. **Test set validation** (preferred for n>50; covered in T-15 findings §4a).
2. **Cross-validation** (small-data fallback).
3. **Permutation tests** (Y-randomization for model validity, the SIMCA
   / PLS_Toolbox canon).
4. **Jackknife uncertainty testing** (the Unscrambler / Martens &
   Martens canon for PLS coefficients).

Westad & Marini does NOT recommend a specific *two-model paired test*.
That is general ML literature territory.

### 3b. Demšar 2006 — the canonical ML-side reference

**Demšar, J. (2006).** *Statistical Comparisons of Classifiers over
Multiple Data Sets.* JMLR 7:1–30.
[JMLR open access](https://jmlr.org/papers/v7/demsar06a.html)

The paper's recommendations (verbatim from search-result excerpts and
the JMLR abstract):

> "[We recommend] a set of non-parametric tests for statistical
> comparisons of classifiers: the Wilcoxon signed ranks test for
> comparison of two classifiers and the Friedman test with the
> corresponding post-hoc tests for comparison of more classifiers over
> multiple data sets."
>
> "The Nemenyi test is similar to the Tukey test for ANOVA and is used
> when all classifiers are compared to each other."

**Demšar's hierarchy:**

1. Two classifiers, multiple datasets (or repeats): **Wilcoxon
   signed-rank**.
2. N classifiers, multiple datasets: **Friedman + Nemenyi (all-pairs)**
   or **Bonferroni-Dunn (vs. baseline)**.
3. Visualization: **Critical-difference plots** showing average ranks
   with significance bars.

This is what mlr3benchmark ships directly. It's what most ML papers
cite when they say "we performed statistical comparison."

### 3c. Hothorn et al. 2005 — caret's foundation

**Hothorn, T., Leisch, F., Zeileis, A., Hornik, K. (2005).** *The
design and analysis of benchmark experiments.* J. Comput. Graph. Stat.
14(3):675–699.

Per caret's documentation citation: this paper laid the framework that
caret implements via `diff.resamples()`. Key points (from secondary
sources):

- For paired comparison on resampled metrics, **paired t-test** on
  per-fold differences is the default; Wilcoxon is the non-parametric
  alternative.
- **Multiple-comparison correction** (Bonferroni) is recommended when
  pairwise-comparing N>2 models.
- The framework makes the (controversial) assumption that per-fold
  scores are sufficiently independent to license a t-test. Bouckaert
  & Frank 2004 and others have published on this assumption's
  fragility.

### 3d. Bouckaert & Frank 2004 — referenced in dasp's tooltips

**Bouckaert, R. R., & Frank, E. (2004).** *Evaluating the Replicability
of Significance Tests for Comparing Learning Algorithms.* PAKDD 2004.
[Author PDF](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf)

Per search-result excerpts:

> "The paper shows that 5x2 cv, resampling and 10 fold cv suffer from
> low replicability. The research indicates that the combination of a
> sorted runs sampling scheme with a t-test gives the most desirable
> performance judged on Type I and II error and replicability."

So Bouckaert & Frank find **none of the standard tests have great
replicability** but their preferred recipe is **sorted runs + paired
t-test**. This is more nuanced than "use 5×2 CV". The 5×2 CV paired
t-test (Dietterich 1998) is still widely used in ML practice and is
what mlxtend exposes; Bouckaert & Frank's analysis suggests it's
*one* defensible choice among several but not optimal.

dasp's tooltip cites this paper at `gui:1357`:

```
"Reference: Bouckaert and Frank (2004) on the replicability of
significance tests for comparing learning algorithms."
```

— in the context of recommending **3-5 repeats for exploratory work,
10+ for publication-quality estimates** (Repeated K-Fold CV). dasp
already nodes to the spirit of B&F (more repeats → more reliable
significance). T-16 would actualize that hint into actual significance
tests.

### 3e. Efron & Tibshirani 1993 — bootstrap canon

**Efron, B. & Tibshirani, R. J. (1993).** *An Introduction to the
Bootstrap.* CRC Press. Already cited in `diagnostics.py:14`.

Foundational reference for percentile bootstrap, bias-corrected
bootstrap (BC), and BCa. tidymodels' `int_pctl()` / `int_t()` /
`int_bca()` implementations follow this canon directly.

### 3f. Politis & Romano 1994 — block bootstrap canon

**Politis, D. N. & Romano, J. P. (1994).** *The Stationary Bootstrap.*
J. Amer. Statist. Assoc. 89(428):1303–1313.

The canonical reference for **block bootstrap** with stationary
sampling (random block lengths). Used when iid resampling is
inappropriate due to spatial / temporal / hierarchical correlation.

For dasp's user case (FTIR Bone PLS): the relevant variant is
**fixed-block bootstrap by group/site** — sample whole sites with
replacement to get a fold-level metric distribution. The user's paper
(`paper/scripts/analysis/03q_bootstrap_inference.py`) implements this
directly.

T-15 was dropped, so dasp doesn't currently have a "group" concept in
the GUI. **A site-level block bootstrap requires the user to supply a
group vector externally** (or T-15 to be re-implemented inline with
T-16). The user's reframe didn't address this constraint.

### 3g. Centner et al. 1996 — UVE (already covered in T-04 findings)

The chemometrics master rule lists Centner 1996 as canonical for UVE
variable selection. Centner's UVE is **not** a model-comparison
method; mentioned for completeness. Same for Wold 2001 (PLS), Brereton
multiple references. None of these chemometrics canonical refs
specify a two-model paired comparison test.

### 3h. The user's FTIR Bone PLS paper

**`paper/scripts/analysis/03q_bootstrap_inference.py`** (already
catalogued in `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` §2-3):

- **Site-level block bootstrap.** 1000 iterations resampling whole
  sites with replacement. Computes per-iteration metric difference
  (chemometric vs. index-baseline MCC). Reports percentile CI on the
  gap.
- **Paired permutation test.** 10,000 iterations swapping prediction
  labels per specimen between two methods. Two-sided p-value on the
  metric gap.

Both are implemented directly in the paper's analysis script. They
are *not* imports from a chemometrics convention — they are imports
from biostatistics / ML practice (the paper's authors brought the
methodology to bone-FTIR). The closest chemometrics-canonical
analogue is Westad & Marini 2015's "preferred validation pipeline"
which mentions test sets + permutation + jackknife but doesn't
specifically prescribe block bootstrap on multi-site CV.

So the user's paper is using methodology from **outside the
chemometrics convention** but the methods are widely-published
canon in ML / biostatistics. For T-16, this means the framing
"what does the chemometrics literature say" is necessary but not
sufficient — the user's actual use case requires bringing in
Demšar 2006-style and Politis & Romano 1994-style machinery that
chemometrics tools don't typically expose.

### 3i. Soneson, Gerster, Delorenzi 2014 — batch confounding

(Already covered in T-15 findings §4f.) Their PLOS One paper
demonstrates over-optimism of standard k-fold CV with confounded
batch effects. Recommends prevention at design time, but for already-
collected confounded data, the implication is **honest uncertainty
quantification** — bootstrap CIs that account for the group structure,
permutation tests that respect the group structure. This argues for
**block bootstrap on metrics** (Shape A or C), not against any
particular paired test.

### 3j. Field-alignment summary

| Question | Chemometrics canon | ML canon | Both share |
|----------|-------------------|----------|------------|
| Is my model better than chance? | Y-permutation / CV-ANOVA (SIMCA, PLS_Toolbox); jackknife on coefficients (Unscrambler) | Permutation test on labels | Both endorse some flavor of permutation |
| Are my coefficients real? | Jackknife (Unscrambler, Martens & Martens 2000; R `pls::jack.test`) | NA — not the question ML asks of black-box models | Chemometrics-only |
| Is method A better than method B (single dataset)? | **Not addressed canonically** | Wilcoxon signed-rank or paired t-test on per-fold errors (Demšar 2006, Hothorn 2005, caret) | ML-only |
| Is method A better than method B (multiple datasets)? | NA (chemometrics rarely runs same method on N datasets) | Wilcoxon, or 5×2 CV paired t (Bouckaert & Frank 2004) | ML-only |
| Are N methods different? | NA | Friedman + Nemenyi (Demšar 2006) | ML-only |
| Are my error bars right when I have group structure? | NA explicitly | Block bootstrap (Politis & Romano 1994) | Both implicitly endorse honest uncertainty |

The user's reframe (*"ways of really comparing between models"*) lives
**predominantly in the ML canon, not the chemometrics canon**. The
chemometrics canon answers "is my model real" but does not answer
"is method A better than method B". This is the gap the FTIR Bone PLS
paper had to fill by importing biostatistics methods (block bootstrap,
paired permutation), and dasp would face the same gap.

---

## 4. Methodological options dasp could ship (Step 4)

Each technique below: one-paragraph description + canonical citation +
implementation effort estimate.

### 4a. Bootstrap CIs on metrics (per-model)

**What it does.** Resample (with replacement) the held-out CV
predictions B times. Compute the metric on each resample. Report
percentile CI (or BCa). User sees "RMSEcv = 0.42 [95% CI: 0.38, 0.47]"
in the Results table.

**Block bootstrap variant.** When samples have structure (sites,
batches, instruments), sample whole groups instead of individual
samples. Politis & Romano 1994. Required for transferability claims
when within-group correlations exist.

**Citations.** Efron & Tibshirani 1993 (general); Politis & Romano
1994 (block); rsample `int_pctl()` (modern impl reference).

**Effort estimate (basic, no block, no group):**
~2 days. ~50 LOC for the core bootstrap function + integration into
`scoring.py` + 4 new columns (`RMSEcv_lo`, `RMSEcv_hi`, `R2cv_lo`,
`R2cv_hi`) in the Results dataframe + GUI tooltip explaining CI
column. Plus tests (~50 LOC).

**Effort estimate (block / group-aware):**
~3-4 days. Adds: group-vector parameter to bootstrap, group-aware
resampling logic, GUI dropdown for "group column" if T-15 isn't
already in. ~100 LOC plus tests.

### 4b. Paired permutation test (two models, single dataset)

**What it does.** H₀: methods A and B are equivalent on this dataset.
For each of P permutations (typical: 10,000), randomly swap the
identity of which method's prediction belongs to each sample. Compute
the metric difference. Two-sided p-value = fraction of permuted
differences ≥ observed |difference|.

**Implementation note.** The user's paper uses this directly (paired
permutation on prediction labels per specimen). It's a label-swapping
permutation, not Y-randomization.

**Citations.** Good 2000 (*Permutation, Parametric, and Bootstrap
Tests of Hypotheses*); the user's `03q_bootstrap_inference.py`
`paired_permutation()` function at line ~180.

**Effort estimate:** ~2-3 days. Requires a "Compare two models" UI
(see §5 below). ~80 LOC for the test function + ~150 LOC for the
GUI dialog + ~50 LOC tests.

### 4c. Wilcoxon signed-rank test (two models, per-fold metrics)

**What it does.** Take per-fold metric values for two models on the
same N folds. Compute paired differences. Wilcoxon signed-rank tests
H₀ that the median difference is zero. Non-parametric — robust to
non-normality of per-fold errors.

**Citations.** Wilcoxon 1945; Demšar 2006 §3.1 "Wilcoxon signed-ranks
test"; caret `diff.resamples(test='wilcox')`.

**Implementation note.** Per-fold metrics are already computed in
dasp's CV path (one metric per fold). The test is one `scipy.stats.wilcoxon` call.

**Effort estimate:** ~half-day. ~10 LOC for the function. Most
effort is the GUI plumbing (compare-dialog) which is shared with
4b/4d/4f.

### 4d. Paired t-test (two models, per-fold metrics)

**What it does.** Same setup as Wilcoxon but parametric. Assumes
per-fold metric differences are approximately normal. For typical 5-
or 10-fold CV with reasonable models, this assumption is borderline
defensible (Hothorn 2005 makes the case for it, B&F 2004 question its
replicability).

**Citations.** Hothorn et al. 2005; caret `compare_models()` and
`diff.resamples(test='t.test')` defaults.

**Implementation note.** `scipy.stats.ttest_rel`. One-line.

**Effort estimate:** ~half-day. Pairs naturally with Wilcoxon (same
GUI dialog, "test type" dropdown).

### 4e. Jackknife on metrics

**What it does.** Drop-one-sample CV-style estimation of metric
variance. For each of N samples, recompute the metric leaving that
sample out. The variance of these N leave-one-out metrics estimates
the metric's sampling variance.

**Citations.** Tukey 1958 (origin); Efron 1979 (modern theory).

**Implementation note.** Has the *worst* statistical properties of all
the methods listed here (notoriously biased downward for variance of
non-smooth statistics like AUC or MCC). Bootstrap is preferred.

**Effort estimate:** ~1 day. ~30 LOC. **Probably not worth shipping**
when bootstrap is just as easy and has better properties. Unless dasp
specifically wants to align with the user's paper's "jackknife"
terminology — but the user's paper uses block bootstrap, not jackknife
on metrics.

### 4f. 5×2 CV paired t-test (Bouckaert & Frank / Dietterich 1998)

**What it does.** Run 5 independent 2-fold CV splits. For each split,
compute the metric difference (model A − model B). The 5×2 = 10
differences, but with a corrected variance estimate that accounts for
within-split dependence. T-test on the corrected differences. Designed
to be robust to per-fold non-independence.

**Citations.** Dietterich 1998 (*Approximate Statistical Tests for
Comparing Supervised Classification Learning Algorithms*); Bouckaert
& Frank 2004.

**Implementation note.** mlxtend Python ships
`paired_ttest_5x2cv()` directly. Could be vendored or reimplemented.
Adds ~10x compute cost (10 fits per pair).

**Effort estimate:** ~1-2 days. ~50 LOC for the test function + integration
with the search infrastructure (rerun two-model search with 5×2
splits). Pairs naturally with the Compare dialog.

### 4g. Bayesian model comparison

**What it does.** Posterior probability that method A's metric > method
B's metric, given the data. Uses a Bayesian hierarchical model on
per-fold differences. tidyposterior in R is a reference impl (uses
Stan).

**Citations.** Benavoli, Corani, Demšar, Zaffalon 2017 (*Time for a
change: a tutorial for comparing multiple classifiers through
Bayesian analysis*) — the modern Bayesian alternative to Demšar 2006.

**Effort estimate:** ~2-3 weeks. Requires probabilistic-programming
backend (PyMC / NumPyro / Stan). **Out of scope for V1** per
chemometrics master rule's bundled-app constraint (PyInstaller +
PyMC is messy; the user is in non-technical bone-FTIR territory, not
Bayesian inference).

### 4h. AIC / BIC

**What it does.** Information-criterion-based model selection. AIC =
−2 log L + 2k; BIC = −2 log L + k log n.

**Citations.** Akaike 1973; Schwarz 1978.

**Implementation note.** Designed for **nested likelihood-based
models** (e.g., choosing PLS components). Not appropriate for
comparing PLS-DA vs. RF vs. XGBoost — these aren't nested and don't
have a comparable likelihood. AIC/BIC for "which model class wins" is
methodologically incorrect.

**Effort estimate:** Skip. Wrong tool for the question.

### 4i. DeLong test for AUC comparison

**What it does.** Specifically for binary classification AUC. Computes
the variance and covariance of two AUCs nonparametrically. Z-test on
the AUC difference.

**Citations.** DeLong, DeLong, Clarke-Pearson 1988.

**Implementation note.** R `pROC::roc.test()`; Python is via
`pROC.roc.test` wrapper. Limited applicability — binary
classification AUC only. The user's domain has multi-class
classification (bone provenance) and regression (collagen yield);
binary AUC is not the headline metric.

**Effort estimate:** ~half-day if specifically requested. ~30 LOC.
**Probably not worth shipping** as default V1 because the user's
domain isn't AUC-headlined.

### 4j. Y-permutation test (single-model validity)

**What it does.** Permute the y-vector P times, refit the model
each time, compute the CV metric. Original-Y metric is compared
against the distribution of permuted-Y metrics. p-value = fraction
of permuted metrics ≥ original. PLS_Toolbox and SIMCA both ship
this as their primary inferential feature.

**Citations.** Lindgren et al. 1996 (*Model validation by permutation
tests: applications to variable selection*); SIMCA's documented
implementation.

**Implementation note.** Different from §4b — this is a *single-model
validity* test, not a *two-model comparison* test. Both are
"permutation tests" but answer different questions.

**Effort estimate:** ~2-3 days. Refit the full model P times (typically
P = 100 to 1000); requires a progress bar (P × full-CV runtime is
significant). ~100 LOC.

### 4k. Coefficient jackknife (Unscrambler-style)

**What it does.** During CV, save the regression coefficient vector
from each fold's PLS fit. The variance of these CV coefficients across
folds estimates the coefficient sampling variance. Per-coefficient
significance test = "is the mean / SE > t-critical?"

**Citations.** Martens & Martens 2000; Westad & Martens; R `pls::jack.test`.

**Implementation note.** Specific to PLS regression (and PLS-DA where
the inner regression has coefficients). Doesn't generalize to
RF/XGBoost. Existing `diagnostics.py:143-230` has jackknife
infrastructure but for *prediction intervals*, not coefficient
variance — would need extension.

**Effort estimate:** ~3-4 days. ~150 LOC + GUI exposure on the PLS
coefficient plot tab. Pairs naturally with T-13 (jackknife prediction
intervals into GUI).

---

## 5. GUI surface options (Step 5)

dasp ships as a bundled Inno Setup desktop app to non-technical users.
GUI surface decisions matter as much as algorithm choice.

### 5a. Bootstrap CIs in the Results table (passive)

**Pattern:** auto-add `_lo` / `_hi` columns next to each existing
metric. Or render as `0.42 ± 0.05` in a single column. Or add
sparkline-like bars. No new UI element required; users see CIs alongside
their existing point estimates.

**Pros:**
- Zero clicks. Users get inferential signal automatically.
- Catches dasp up to tidymodels-style bootstrap-CI-on-metrics canon.
- Matches the chemometrics convention of "report uncertainty alongside
  point estimate".

**Cons:**
- Adds compute cost to every search (B × N_models runs of the metric
  function — small, but accumulates).
- Doubles the column count in the Results table — already cluttered.
- Doesn't answer "is method A better than method B" directly; user
  has to eyeball "do the CIs overlap?" which is a known fallacy
  (overlapping CIs don't imply non-significance; non-overlapping
  CIs imply more than significance).

**Best fit for:** Shape A / C above.

### 5b. "Compare Two Models" right-click on Results table

**Pattern:** User Ctrl-clicks two rows in the Results table → right-
click → "Compare these two models" → modal dialog opens. Dialog shows:

- Per-metric difference (model A − model B).
- Paired t-test p-value (default) or Wilcoxon signed-rank p-value
  (toggle).
- 95% CI on the difference (bootstrap or t-distribution-based).
- (Optional) paired permutation test p-value (slower; toggle).
- Effect-size measure (Cohen's d for t-test, rank-biserial for
  Wilcoxon).

Plus a verdict line: "Model A is significantly better than Model B
(p = 0.003, 95% CI: [0.012, 0.087])."

**Pros:**
- Explicit comparison; the user clicks "are these different" and gets
  a defensible answer.
- Caret's `compare_models()` analogue; matches familiar ML-framework
  convention.
- Pairs directly with the user's actual paper-publishing workflow.

**Cons:**
- Required scaffolding: GUI dialog, per-fold metric storage (currently
  only aggregate metrics are saved per row, NOT per-fold values —
  unverified, but `scoring.py` schema doesn't include per-fold arrays).
  Would need to extend the result row schema or rerun CV at compare
  time.
- "Two-model" only — doesn't scale to N>2 (need shape D for that).

**Best fit for:** Shape B.

### 5c. "Run Permutation Test" button on Model Development tab

**Pattern:** User loads a refined model in Model Development tab →
clicks "Validate via Permutation Test" → progress dialog (P=100 or
1000 permutations, configurable) → result page shows:

- Original metric value.
- Permuted-metric distribution (histogram).
- p-value with verdict ("Model is significant: p < 0.001").

Matches PLS_Toolbox / SIMCA's UX directly.

**Pros:**
- Catches dasp up to PLS_Toolbox / SIMCA on the "is this model real"
  question.
- Explicit, single-button, single-model validity check.

**Cons:**
- Slow (P × full CV runtime). Needs progress bar + cancel button.
- Doesn't answer two-model comparison.

**Best fit for:** addition to Shape A (single-model uncertainty +
permutation validity).

### 5d. Friedman + Nemenyi multi-model panel

**Pattern:** User multi-selects N>2 rows in Results → "Compare these N
methods" → critical-difference plot rendered (mlr3benchmark style).
With horizontal bars indicating non-significant pairs.

**Pros:**
- Matches state-of-the-art ML practice (Demšar 2006, mlr3benchmark).
- No commercial chemometrics tool offers this.

**Cons:**
- New plot type to implement (CD plot). matplotlib-feasible but
  novel.
- Requires per-fold metric storage (same as 5b).
- Most ambitious UI surface.

**Best fit for:** Shape D.

### 5e. Per-fold variability surface

**Pattern:** Each Results row stores per-fold metric arrays. Add a
"View per-fold metrics" expand-row option: click row → expansion
shows per-fold RMSE / R² / etc. with a small sparkline.

**Pros:**
- Cheap (per-fold metrics are already computed; just need to store).
- Foundation for 5b/5d (those tests need per-fold values).

**Cons:**
- UI noise; users may not know what to do with per-fold values.

**Best fit for:** infrastructure prerequisite for shapes B / D.

### 5f. Distribution-model check

dasp ships as a bundled Inno Setup desktop GUI for non-technical
bone-FTIR / paleoanthropology / archaeology users. Per the T-26
lesson, a backend-only T-16 (e.g., expose a `compare_two_models()`
function in `scoring.py` with no GUI) is dead code for the bundled-app
user base.

Whatever shape ships **must surface in the GUI** to deliver value.
The shapes catalogued in §5a-5d are GUI-first by design. The
Python-API-only path (similar to the current `lins_ccc` function)
would be an explicit non-goal.

The verdict-writer should specifically check that the chosen shape's
GUI surface is concrete and clickable, not "expose via the API for
expert users."

---

## 6. Smallest-defensible-shape candidates (Step 6)

The verdict-writer's job. Catalog of shapes the verdict can scope to
without inventing patterns the field doesn't use:

### 6a. Shape A — chemometrics-conservative (single-model uncertainty + validity)

**Includes:**
- Bootstrap CIs on per-row metrics in Results table (§4a basic).
- Y-permutation test on Model Development tab (§4j).

**Field-alignment:** Matches what every chemometrics competitor ships
(SIMCA + PLS_Toolbox have permutation; Unscrambler has jackknife —
this shape has both).

**Doesn't address:** "Is method A better than method B?" That
question is unanswered by chemometrics tools and would still be
unanswered by Shape A.

**Effort:** ~5-7 days (bootstrap impl + perm test + 2 GUI surfaces).

**Risk:** Possibly under-specifies what the user's paper actually
needs. The paper explicitly compares two methods — chemometric vs.
index-baseline — and reports gap CI + p-value. Shape A doesn't
deliver that.

### 6b. Shape B — caret-style two-model comparison

**Includes:**
- Per-fold metric storage in result rows (§5e infrastructure).
- "Compare Two Models" dialog (§5b).
- Paired t-test (§4d) + Wilcoxon signed-rank (§4c) as toggle.
- Optional: paired permutation test (§4b) as slower alternative.

**Field-alignment:** Matches caret / Hothorn 2005 / Demšar 2006
two-classifier convention. Does NOT match chemometrics-tool
convention (which doesn't have this feature).

**Doesn't address:** Single-model validity ("is my model real");
multi-model panel ("which of these 5 is best with significance?");
group/site structure.

**Effort:** ~5-7 days.

**Risk:** Skips the chemometrics-canonical Y-permutation feature
that domain users may expect from a chemometrics tool.

### 6c. Shape C — paper-aligned (FTIR Bone PLS replication)

**Includes:**
- Per-fold + per-sample metric storage.
- Site-level block bootstrap (§4a block variant) — requires user-
  supplied group vector or T-15-style group infrastructure.
- Paired permutation test on metric gaps (§4b).
- "Compare Two Models" dialog displaying gap CI + p-value (§5b style
  but with block resampling).

**Field-alignment:** Matches the user's actual paper-publishing
workflow directly. Doesn't match a single competitor exactly but
adopts the FTIR Bone PLS paper's published methodology (which is
itself imported from biostatistics, not chemometrics).

**Doesn't address:** Single-model validity (no Y-permutation); multi-
model panel.

**Effort:** ~5-7 days as a bare module (user supplies group vector via
a GUI dropdown bound to a metadata column). ~10-12 days if T-15
group infrastructure is rebuilt inline.

**Dependency consideration:** T-15 was dropped, so the group/site
column concept doesn't exist in the GUI. Block bootstrap fundamentally
needs a group structure. Either T-15 comes back (probably as a
hidden dependency of T-16 if shape C is chosen) or the user manually
loads a metadata column and points to it from the new dialog.

### 6d. Shape D — Demšar full multi-model panel

**Includes:**
- All of Shape B.
- Plus Friedman + Nemenyi post-hoc (§4 "Demšar 2006 multi-model").
- Plus critical-difference plot (mlr3benchmark style).

**Field-alignment:** Matches mlr3benchmark / Demšar 2006. Goes beyond
any commercial chemometrics tool. Most rigorous ML-side scoping.

**Effort:** ~10-14 days.

**Risk:** Probably too ambitious for V1. Dasp has 0 inferential
infrastructure today; jumping to "critical-difference plots" is a 10x
scope increase. Could be deferred as Shape B+ in a follow-up.

### 6e. Hybrid candidates

- **A + B:** chemometrics single-model validity + ML two-model
  comparison. Covers both traditions. ~10-12 days. Probably what a
  best-of-both verdict picks.
- **A + C:** chemometrics single-model validity + paper-aligned
  block bootstrap. Most paper-publication-ready. ~10-12 days.
  Requires T-15 dependency resolution.
- **A + B + Y-permutation in same dialog:** consolidated UI where
  the Compare dialog has a "validate single model" tab next to the
  "compare two models" tab. ~12-14 days.

### 6f. Already-in-dasp overlap

- **Lin's CCC (T-24, just merged):** A metric, not a comparison
  test. Doesn't overlap T-16 directly. Could be a candidate metric
  to compute CIs / paired tests on.
- **Jackknife prediction intervals (`diagnostics.py:143-230`):**
  Per-sample prediction intervals, not per-metric CI. Different
  statistical question. T-13 is the GUI-wiring ticket for that, not
  T-16. Some shared bootstrap-style infrastructure could be reused
  if the implementation is careful.
- **Bouckaert & Frank 2004 reference in tooltip:** Already nodes to
  the spirit of significance testing but doesn't actualize it. T-16
  Shape B+ closes that loop.

### 6g. Competitor-parity vs. competitor-exceeding

| Shape | dasp would meet | dasp would exceed |
|-------|----------------|-------------------|
| A | Catches up to PLS_Toolbox + SIMCA on single-model permutation; matches Unscrambler's flagship if jackknife on coefficients (§4k) is added | Tidymodels-style bootstrap CI on metrics |
| B | Catches up to caret/Hothorn 2005 on paired t-test + Wilcoxon | Goes beyond any chemometrics competitor in the "compare two methods" question they don't expose |
| C | Catches up to the FTIR Bone PLS paper's methodology directly | Goes beyond any commercial chemometrics tool in handling site-level block bootstrap |
| D | Catches up to mlr3benchmark on Friedman + Nemenyi | First chemometrics GUI tool to expose Demšar 2006 critical-difference plots |

The **competitor-parity floor** is Shape A (matches what chemometrics
tools have shipped since the 1990s but dasp lacks). The **competitor-
exceeding ceiling** is Shape D (no commercial chemometrics tool offers
multi-model significance panels).

### 6h. The user's reframe specifically asked

The user's reframe was: *"ways of really comparing between models are
warranted ... what our competitors do is a head start."*

Reading that literally:

- "Comparing between models" → **Shape B / C / D** territory (two-
  model or N-model comparison).
- "What our competitors do" → **Shape A** territory (Unscrambler /
  PLS_Toolbox / SIMCA single-model machinery).

These are not the same shape. The user's reframe implicitly contains
*both* a "compare two methods" goal and a "match what competitors
ship" sanity check, which point to different feature sets.

The verdict-writer should be explicit about which interpretation
takes priority:

- If "comparing between models" wins: Shape B/C/D, accepting that
  this goes beyond what any chemometrics competitor offers.
- If "what our competitors do" wins: Shape A, accepting that it
  doesn't directly answer the user's "comparing between models"
  question.
- If both must be served: Shape A + B (or A + C), accepting the
  larger scope.

---

## 7. Notes / uncertainties for the verdict-writer

1. **The user's reframe is internally tensioned.** "Comparing between
   models" is an ML-framework-canon question; "what our competitors
   do" is a chemometrics-tool-canon answer. The two canons have
   different feature sets. Verdict-writer needs to pick or merge.

2. **Per-fold metric storage is a hidden prerequisite.** Shapes
   B/C/D all need per-fold metric arrays per result row. dasp's
   current Results schema (`scoring.py:457-497`) stores aggregate
   metrics only. Per-fold storage would add ~K× memory per row (K =
   fold count). For typical 5-10 fold CV with 100-1000 result rows,
   this is small (~50-100 KB extra), but the schema change is
   non-trivial and would touch search.py, bayesian_utils.py, all
   exporters.

3. **The Compare Two Models UI may need to defer to "Refit and
   compare on user-supplied test set"** as a fallback, when per-fold
   metrics aren't available (e.g., Bayesian search results, which
   use Optuna trial.user_attrs and may not have full per-fold
   storage).

4. **T-15 dependency for Shape C.** Site-level block bootstrap
   needs a group vector. T-15 was dropped — but if Shape C wins,
   T-15-equivalent infrastructure (group-column dropdown bound to a
   metadata column) becomes a hidden dependency. This is implicit
   re-scoping of T-15.

5. **Misleading existing tab name.** "Multi-Model Comparison" tab
   (`gui:47694`) already exists — but it's a prediction dashboard,
   not statistical comparison. Naming the new feature must avoid
   user confusion. Suggestions: "Method Significance" / "Model
   Significance Tests" / "Statistical Comparison".

6. **Permutation tests have known failure modes for K-fold.** RSC
   Mol. BioSyst. 2015 specifically warns about Y-permutation under
   K-fold giving contradictory results. If Shape A includes
   Y-permutation, dasp must (a) document this caveat in the GUI,
   or (b) restrict to LOO/repeated CV (which has its own bias).
   This is non-trivial UX.

7. **mlxtend dependency option for 5×2 CV paired t-test.** If shape
   B includes 5×2 CV (Bouckaert & Frank), `mlxtend` provides a
   reference impl. Adding mlxtend is ~15 MB to the bundled installer.
   Trade-off: vendor a 200-LOC reimplementation vs. add the
   dependency.

8. **No need to worry about AIC/BIC.** Methodologically wrong tool
   for non-nested model class comparison. Verdict can drop entirely.

9. **DeLong test is binary-AUC-only.** If the user's domain shifts
   to binary classification headlines, reconsider. For now, drop.

10. **Real finding can warrant zero action (T-26 lesson) — possibly
    applicable here.** dasp currently has zero comparison machinery,
    but so does OPUS / Bruker QUANT, and OPUS users coexist with the
    chemometrics community. The "do nothing, users use external
    Python/R" path is defensible if the verdict-writer concludes
    bone-FTIR users can be guided to caret / mlxtend / scipy.stats
    for inferential needs. **This would be a valid disposition**
    even though dasp has the infrastructure to ship something.

11. **All four shapes preserve future extensibility.** Shape A doesn't
    block Shape B; Shape B doesn't block Shape D. The verdict-writer
    can pick the smallest shape and let later iterations expand.
    This argues for *starting* with the cheapest defensible option
    (Shape A — bootstrap CIs in Results table) and adding Shape B
    only after observing whether users actually use Shape A's CIs.

12. **The original T-16 framing (block bootstrap + paired permutation)
    is Shape C.** The user's reframe broadens the question, but if
    the goal is publishing-the-paper-with-dasp parity, Shape C is the
    most direct answer. Shape A or A+B is the "broader competitor
    parity" answer.

13. **dasp's existing diagnostics.py jackknife doesn't fit T-16.**
    Per-sample prediction intervals (T-13) ≠ per-model metric CIs
    (T-16). The infrastructure isn't directly reusable. Mentioning
    here for completeness — verdict-writer should not assume the
    diagnostics.py jackknife is a head-start for T-16.

14. **Validation gate filter check** — running T-16 through the five
    recurring false-alarm patterns:

   | Pattern | Applies to T-16? |
   |---------|------------------|
   | Sklearn-instinct false alarm | **No.** Multiple chemometrics-canonical features (Y-permutation, jackknife) are the baseline ask. Adding ML-canonical features (paired tests, CD plots) is an explicit user-desired extension. |
   | Defensive code in unreachable branch | **No.** No code exists yet. |
   | Display-economy / code-style | **No.** New columns / new dialogs deliver real inferential signal, not display fluff. |
   | dasp already matches the leading program | **No.** dasp has zero comparison machinery; competitors have one or more features. |
   | Real finding can warrant zero action | **Possibly.** Per §11 above, "users can use external tools" is a defensible disposition. |

---

## Sources

### Competitors

- [The Unscrambler — Uncertainty Testing](https://www.spectroscopyeurope.com/td-column/uncertainty-testing-pls-regression)
- [Martens & Martens 2000 — Modified Jack-knife (FQP 11)](https://www.sciencedirect.com/science/article/abs/pii/S0950329399000397)
- [Westad & Martens — Variable Selection in NIR via PLSR significance testing](https://www.researchgate.net/publication/244738666_Variable_selection_in_NIR_based_on_significance_testing_in_Partial_Least_Squares_Regression_PLSR)
- [Eigenvector — Tools: Permutation Test](https://wiki.eigenvector.com/index.php?title=Tools%3A_Permutation_Test)
- [Eigenvector — Using Cross-Validation](https://www.eigenvectordocs.com/index.php?title=Using_Cross-Validation)
- [Sartorius — Multivariate Calibration in SIMCA app note](https://www.sartorius.com/download/545668/simca-appnote3-spectroscopydata-en-b-00061-sartorius-data.pdf)
- [SIMCA 15 User Guide](https://www.sartorius.com/download/544940/simca-15-user-guide-en-b-00076-sartorius-data.pdf)
- [PLS/OPLS validation — RSC Mol. BioSyst. 2015](https://pubs.rsc.org/en/content/articlelanding/2015/mb/c4mb00414k)
- [OPUS QUANT manual — Bruker PDF](https://files.nocnt.ru/hardware/science/senterra/opus65-doc-en/quant.pdf)
- [Bruker — OPUS Quantification page](https://www.bruker.com/en/products-and-solutions/infrared-and-raman/opus-spectroscopy-software/quantification.html)
- [mdatools simca() reference (Kucheryavskiy)](https://rdrr.io/cran/mdatools/man/simca.html)
- [R `pls::jack.test`](https://search.r-project.org/CRAN/refmans/pls/html/jack.test.html)

### ML frameworks

- [caret diff.resamples / compare_models](https://rdrr.io/cran/caret/man/diff.resamples.html)
- [mlr3 benchmark — Evaluation chapter](https://mlr3book.mlr-org.com/chapters/chapter3/evaluation_and_benchmarking.html)
- [mlr3benchmark — autoplot.BenchmarkAggr (CD plots, Friedman-Nemenyi)](https://mlr3benchmark.mlr-org.com/reference/autoplot.BenchmarkAggr.html)
- [tidymodels rsample — Bootstrap CIs vignette](https://rsample.tidymodels.org/articles/Applications/Intervals.html)
- [tidymodels rsample — int_pctl reference](https://rsample.tidymodels.org/reference/int_pctl.html)
- [tidymodels tune — int_pctl.tune_results](https://tune.tidymodels.org/reference/int_pctl.tune_results.html)
- [tidymodels — Bootstrap-metrics tutorial](https://www.tidymodels.org/learn/models/bootstrap-metrics/)
- [Workflowsets — Tuning and comparing models](https://workflowsets.tidymodels.org/articles/tuning-and-comparing-models.html)

### Canonical literature

- [Demšar 2006 — Statistical Comparisons of Classifiers (JMLR 7)](https://jmlr.org/papers/v7/demsar06a.html)
- [Bouckaert & Frank 2004 — Replicability of Significance Tests (PAKDD 2004)](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf)
- [Westad & Marini 2015 — Validation of chemometric models tutorial (ACA 893)](https://www.sciencedirect.com/science/article/abs/pii/S0003267015008922)
- [Filzmoser et al. 2009 — Repeated double cross-validation (J. Chemom. 23)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.1225)
- [Krstajic et al. 2014 — Cross-validation pitfalls (J. Cheminf. 6:10)](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10)
- [Soneson, Gerster, Delorenzi 2014 — Batch Effect Confounding (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0100335)

### dasp internal references

- `C:\Users\mspon\git\dasp\src\spectral_predict\scoring.py` (full file, especially `create_results_dataframe()` lines 442-497 — schema with no CI/p-value columns)
- `C:\Users\mspon\git\dasp\src\spectral_predict\diagnostics.py:143-230` — existing `jackknife_prediction_intervals()`, GUI-orphaned
- `C:\Users\mspon\git\dasp\spectral_predict_gui_optimized.py:14218-14400` — Tab 6 Results (point-estimate-only Treeview)
- `C:\Users\mspon\git\dasp\spectral_predict_gui_optimized.py:47694-48068` — Tab 9 "Multi-Model Comparison" (prediction dashboard, NOT statistical comparison; naming clash for T-16 UI)
- `C:\Users\mspon\git\dasp\spectral_predict_gui_optimized.py:1357` — existing tooltip cite to Bouckaert & Frank 2004
- `C:\Users\mspon\git\dasp\src\spectral_predict\templates\*.py` — code-export templates have no comparison statistics
- `C:\Users\mspon\git\dasp\docs\analysis_vs_ftir_bone_pls\GAP_ANALYSIS.md` §2-3 — paper-aligned block-bootstrap + paired-permutation framing
- `C:\Users\mspon\git\dasp\docs\analysis_vs_unscrambler\TECHNICAL_GAPS.md:248-253` — already documents "Permutation testing / Y-randomization for model validity" as gap vs. Unscrambler
- `C:\Users\mspon\git\dasp\docs\bugfix_validation\T15_findings.md` — context on dropped T-15 (group infrastructure dependency for shape C)
