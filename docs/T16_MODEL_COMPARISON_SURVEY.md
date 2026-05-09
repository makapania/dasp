# T-16 — Competitive Model-Comparison Machinery Survey

**Date:** 2026-05-08
**Reframed from:** "block bootstrap CIs + paired permutation" (2026-04-30)
**Original user framing:** *"ways of really comparing between models are warranted and have to be looked at from a competitive framework. there was jackknifing and permutations and i don't know what we should do, but what our competitors do is a head start."*

**Headline finding:** the user's premise — that competitors expose pairwise model-comparison machinery dasp could copy — is **mostly wrong**. The commercial chemometrics packages answer a *different* question (model-vs-null) than the one dasp users actually ask (model-A-vs-model-B). The relevant prior art lives in the ML/stats canon (Dietterich 1998, paired bootstrap, McNemar), not in chemometrics commercial software.

---

## The two distinct "model comparison" questions

Before surveying what's available, the survey turned up a critical conceptual split:

| Question | What it answers | Statistical machinery |
|---|---|---|
| **Q1: Model vs. null** | "Is my model real, or is it overfitting / spurious?" | Permutation test on Y, CV-ANOVA, F-test on PRESS |
| **Q2: Model A vs. Model B** | "Is preprocessing X + model Y better than preprocessing X' + model Y'?" | Paired bootstrap CI on ΔRMSEP, paired permutation on signed differences, Wilcoxon signed-rank, McNemar (classification), Diebold-Mariano (forecasting) |

**The chemometrics commercial software answers Q1.** dasp users actually need Q2 — they have a leaderboard with R²pred 0.95 vs 0.94 and want to know if the difference is real or noise. Today they "pick the higher number," which is overfitting-prone.

---

## Survey: what each commercial package exposes

### Unscrambler X / Unscrambler 11 (CAMO, now Aspen Technology)

**Pairwise model comparison: NO.**

Has paired t-test, equal/unequal variance Student's t-test, F-test, Levene's test, Bartlett's test — but these are for *means and variances of variables*, not for prediction-error differences between two models. No bootstrap CIs or permutation tests for model-vs-model surfaced in the documentation.

Sources: [The Unscrambler — Wikipedia](https://en.wikipedia.org/wiki/The_Unscrambler), [CAMO — Unscrambler](https://www.camo.com/unscrambler/).

### SIMCA (Sartorius / Umetrics)

**Pairwise model comparison: NO. Q1 only.**

Exposes:
- **CV-ANOVA** (Eriksson et al. 2008): F-test on the cross-validated PRESS to ask whether a single PLS/OPLS model is significantly better than a model that always predicts the mean. This is Q1.
- **Permutation test on Y**: random Y-shuffling distribution of Q² compared against the original Q². Again Q1 — is this model real?

No machinery for "is model A's RMSEP significantly different from model B's RMSEP."

Sources: [CV-ANOVA paper](https://www.researchgate.net/publication/229570726_CV-ANOVA_for_significance_testing_of_PLS_and_OPLSR_models), [SIMCA 15 User Guide PDF](https://www.sartorius.com/download/544940/simca-15-user-guide-en-b-00076-sartorius-data.pdf), [PLS/OPLS permutation paper](https://pubmed.ncbi.nlm.nih.gov/25382277/).

### PLS_Toolbox / Solo (Eigenvector)

**Pairwise model comparison: visual only, no statistical test.**

- Cross-validation methods: Venetian Blinds, Contiguous Blocks, Random Subsets, etc.
- Plot: `RMSECV/RMSEC` ratio vs `RMSECV` — "best model in lower left corner with small overfit and prediction error." This is a visual heuristic, not a statistical test.
- Bootstrap CIs in the literature surfaced were for PLS-SEM (Structural Equation Modeling — *parameter* CIs) which is a different application than predictive model comparison.

Sources: [Eigenvector — PLS_Toolbox](https://eigenvector.com/software/pls-toolbox/), [Eigenvector docs — Cross-Validation](https://www.eigenvectordocs.com/index.php?title=Using_Cross-Validation), [Eigenvector — Hating on R-squared](https://eigenvector.com/%EF%BF%BCevaluating-models-hating-on-r-squared/).

### OPUS / OPUS QUANT / OPUS QUANT2 (Bruker)

**Pairwise model comparison: NO. Within-model component selection only.**

Mahalanobis-distance one-sided F-test on PRESS at α=0.05 to pick the smallest LV count whose PRESS is not significantly worse than the lowest. This is *within one PLS model* (component selection), not between two distinct models.

Sources: [Bruker — Quantification](https://www.bruker.com/en/products-and-solutions/infrared-and-raman/opus-spectroscopy-software/quantification.html), [OPUS QUANT manual PDF](https://files.nocnt.ru/hardware/science/senterra/opus65-doc-en/quant.pdf).

### mdatools (R, open-source)

**Pairwise model comparison: NO. Q1 only.**

`randtest()` is a permutation test on Y for selecting the optimal number of components within a single PLS model. Confirmed by direct code reading of `pls.R` on GitHub: no functions take two PLS model objects as arguments. No paired bootstrap, no paired permutation between two models, no McNemar.

Sources: [mdatools — randomization test](https://mda.tools/docs/pls--randomization-test.html), [GitHub — svkucheryavski/mdatools/pls.R](https://github.com/svkucheryavski/mdatools/blob/master/R/pls.R), [CRAN package PDF](https://cran.r-project.org/web/packages/mdatools/mdatools.pdf).

### Chemometrics literature

The two canonical validation references the user has cited (Westad & Marini 2015, Workman 2018) recommend external test set + permutation testing. Westad & Marini's permutation framing is again Q1 (compare against random labels), not Q2 (compare two real models).

Sources: [Westad & Marini 2015 — Validation of chemometric models tutorial](https://www.sciencedirect.com/science/article/abs/pii/S0003267015008922), [Anal Chim Acta PDF mirror](https://elearning.unito.it/scienzedellanatura/pluginfile.php/79077/mod_resource/content/1/ACA%202015%20-%20Marini%20-%20Validation%20of%20chemometric%20models%20%E2%80%93%20A%20tutorial.pdf), [Ezenarro et al. 2025 systematic review of NIRS validation](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.70036).

---

## Where Q2 machinery actually lives: the ML/stats canon

dasp's question (Q2) is canonical in **machine learning and statistics**, less so in chemometrics:

| Method | Reference | Use case | Compute cost |
|---|---|---|---|
| **Paired bootstrap CI on ΔRMSEP / ΔAccuracy** | Efron & Tibshirani 1993 | Regression OR classification; any metric | Cheap (resample existing predictions, no refit) |
| **Paired permutation on signed differences** | classical | Regression OR classification | Cheap (sign-flip on existing predictions) |
| **Wilcoxon signed-rank** | classical | When residual distributions are skewed | Cheap |
| **McNemar's test** | McNemar 1947 | Classification only (per-sample agreement) | Cheap |
| **5×2cv paired t-test** | Dietterich 1998 | Comparing learning *algorithms* | Expensive (10 refits per algorithm) |
| **5×2cv combined F-test** | Alpaydin 1999 | More robust than Dietterich's t-test | Expensive (same as above) |
| **Diebold-Mariano** | Diebold & Mariano 1995 | Forecasting; **NOT recommended for model selection** per Diebold himself | Cheap |

**The cheap methods (top 4) all share a key property:** they operate on **already-computed predictions** on a fixed held-out set. They don't require refitting models. dasp already produces those predictions via `compute_validation_metrics_for_top_models()` at `search.py:568` — the per-model held-out predictions are the input these tests need.

**The expensive methods (5×2cv variants)** require re-running the entire training procedure 10 times per model. Doesn't fit dasp's existing "search once, rank, then validate top-N" workflow without restructuring.

**Diebold-Mariano** is widely misused for model selection; Diebold himself wrote a paper warning against this exact misuse ([Penn arxiv mirror](https://www.sas.upenn.edu/~fdiebold/papers/paper113/Diebold_DM%20Test.pdf)).

Sources: [Dietterich 1998 — Approximate Statistical Tests](https://pubmed.ncbi.nlm.nih.gov/9744903/), [mlxtend — paired_ttest_5x2cv](https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/), [Raschka — Model evaluation tutorial](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html).

---

## Recommended dasp shape — smallest defensible implementation

Given the survey's headline finding, "copy what competitors do" leads to nothing useful for Q2. The smallest defensible shape draws from the ML/stats canon and reuses what dasp already has.

### Tier 1 — paired bootstrap CI (~1-2 days)

For any two leaderboard rows (the user picks two from the GUI), compute on the existing held-out partition:

- **Regression:** 95% CI on Δ = RMSEP_A − RMSEP_B via paired bootstrap on per-sample squared residuals (B=2000). Verdict: "A clearly better" / "B clearly better" / "indistinguishable" depending on whether 0 falls inside the CI.
- **Classification:** same on Δ = Accuracy_B − Accuracy_A via paired bootstrap on per-sample correctness. Or Δ-F1 for imbalanced.

**Why first:** it's cheap (no refits — just resampling existing predictions), it directly answers the user's "0.95 vs 0.94 — real or noise" question, and the CI presentation is more interpretable than a p-value.

### Tier 2 — paired permutation test (~half day, complementary to Tier 1)

Same input data, different test. Generates a p-value via random sign-flips on the per-sample difference vector (B=10000). Pair with Tier 1 because CI and p-value answer slightly different questions and chemometrics reviewers expect both.

### Tier 3 — McNemar's test for classification (~couple hours)

Per-sample agreement table (model A correct / wrong × model B correct / wrong). Standard in classification model comparison; cheap to add once Tier 1+2 are wired.

### NOT recommended

- **5×2cv (Dietterich, Alpaydin)** — would require restructuring dasp's search workflow. The 10× refit cost is real on chemometrics-sized leaderboards (1000+ rows).
- **Diebold-Mariano** — author warned against using it for model selection.
- **CV-ANOVA** — answers Q1, not Q2.
- **Within-model permutation test (à la SIMCA, mdatools `randtest`)** — answers Q1. dasp's per-model gap filter `|R²cv − R²pred| ≤ 0.10` already serves the "is this model overfit" purpose.

### GUI surface

Selection: GUI table → checkbox/highlight two rows → "Compare these models" button → modal dialog with Tier 1 CI, Tier 2 p-value, (Tier 3 McNemar table for classification) + a one-line plain-English verdict. CSV export of the comparison record (which models, which test set, which test, the result).

---

## Effort

- Tier 1: ~1-2 days (bootstrap utility + GUI surface + result-record CSV).
- Tier 2: ~half day (permutation utility + dialog row).
- Tier 3 classification add: ~couple hours.
- Tests: ~half day (analytical sanity checks + integration tests against the existing held-out partition).

Total: ~3-4 days for the full Tier 1-3 shape. Original handoff estimate of ~3-5 days post-survey is on track.

---

## Open decisions for the user

1. **Tier 3 in scope, or defer?** McNemar is small and complementary; could ship Tier 1+2+3 as one or hold Tier 3 for after Tier 1+2 see use.
2. **CSV export shape:** sidecar comparison-record CSV (one row per pairwise comparison, keyed on the two model fingerprints), or columns appended to the leaderboard? Sidecar is cleaner since pairwise comparisons are sparse.
3. **GUI placement:** in the Results tab next to the existing leaderboard, or as a dedicated "Compare" sub-tab? Results-tab inline is more discoverable; sub-tab gives more screen real estate for the dialog.
