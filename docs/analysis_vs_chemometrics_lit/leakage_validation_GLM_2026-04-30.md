# Literature-Grounded Validation: Variable Selection Leakage Audit

**Author:** GLM-5.1 (literature review agent)  
**Date:** 2026-04-30  
**Purpose:** Verify whether the dasp "LEAKY" classification of per-fold variable selection is correct according to the published chemometrics methodology, or whether it over-applies sklearn-pipeline-purity conventions to a domain with its own established practice.

---

## 1. Li 2009 CARS — What Is the Actual Recommended Workflow?

### 1.1 Paper Identity

Li, H.D., Liang, Y.Z., Xu, Q.S., & Cao, D.S. (2009). "Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration." *Analytica Chimica Acta*, 648(1), 77-84. DOI: 10.1016/j.aca.2009.06.046. Cited 2083 times (as of 2026).

### 1.2 CARS Algorithm Summary (from abstract and paper text)

From the abstract:

> "By employing the simple but effective principle 'survival of the fittest' on which Darwin's Evolution Theory is based, a novel strategy for selecting an optimal combination of key wavelengths of multi-component spectral data, named competitive adaptive reweighted sampling (CARS), is developed. [...] In each sampling run, a fixed ratio (e.g. 80%) of samples is first randomly selected to establish a calibration model. Next, based on the regression coefficients, a two-step procedure including exponentially decreasing function (EDF) based enforced wavelength selection and adaptive reweighted sampling (ARS) based competitive wavelength selection is adopted to select the key wavelengths. **Finally, cross validation (CV) is applied to choose the subset with the lowest root mean square error of CV (RMSECV).**"

From the "Notation" section:

> "Suppose the number of MC sampling runs of CARS is set to N. With this setting, CARS will sequentially select N subsets of wavelengths. Briefly speaking, in each sampling run, CARS works in four successive steps: (1) Monte Carlo for model sampling. (2) Employ [exponential decay] for enforced wavelength selection. (3) [Adaptive reweighted sampling] for competitive wavelength selection. (4) Cross-validation for selecting the optimal subset."

### 1.3 Key Question: Does CARS Run Once on Full Data or Per-Fold?

**Answer: CARS runs once on the full calibration data to select wavelengths, then the selected wavelengths are evaluated on a separate independent test set.**

The specific protocol in Li et al. 2009:

1. **CARS runs on the full calibration dataset** (all calibration samples) once. It produces N candidate subsets via Monte Carlo iterations and selects the one with lowest RMSECV. This RMSECV is computed within CARS's own internal K-fold CV that evaluates each iteration's candidate subset.

2. **The RMSECV reported within CARS is a selection criterion, not an unbiased performance estimate.** It is used to choose *which* of the N subsets to keep, analogous to how hyperparameter tuning uses CV to select the best configuration.

3. **For final performance evaluation, Li et al. use a separate independent test set (RMSEP), not the CARS-internal RMSECV.**

This is confirmed by the paper's evaluation methodology. In their NIR corn dataset experiments (Section 3.2), they report:
- RMSECV from CARS internal selection (Table 1)
- RMSEP on a separate test/prediction set (Table 2, Fig. 4)

The CARS-internal RMSECV is used for *subset selection* (which iteration's subset to keep), while RMSEP on the separate test set is the reported *prediction performance*.

### 1.4 Critical Observation

Li et al. 2009 do **NOT** report cross-validation results where the variable selection step was re-run inside each outer CV fold. They report either:
- The CARS-internal RMSECV (which is a biased selection criterion, not an unbiased performance estimate), or
- RMSEP on a held-out test set (which IS unbiased).

The paper simply never claims that the CARS-internal RMSECV is an unbiased estimate of generalization performance. It is a model *selection* metric, not a model *evaluation* metric.

---

## 2. Other Canonical Papers

### 2.1 UVE (Centner et al. 1996)

**Paper:** Centner, V., Massart, D.L., de Noord, O.E., de Jong, S., Vandeginste, B.M., & Sterna, C. (1996). "Elimination of uninformative variables for multivariate calibration." *Analytical Chemistry*, 68(21), 3851-3858. Cited 1362 times.

**Protocol:** UVE augments X with random noise variables, then runs LOO-CV on the full augmented dataset to compute coefficient stability (reliability scores) for each variable. Variables whose reliability scores fall below the maximum noise-variable reliability are eliminated.

**Key point:** UVE's LOO-CV is used to *stabilize coefficient estimates*, not to estimate prediction performance. Centner et al. report **RMSEP on a separate test set** for final model evaluation (not LOO-CV after UVE selection).

**Per-fold refitting?** No. UVE runs once on full calibration data. The LOO-CV inside UVE is part of the algorithm's coefficient estimation procedure, not an outer performance evaluation.

### 2.2 SPA (Araujo et al. 2001)

**Paper:** Araujo, M.C.U., Saldanha, T.C.B., Galvao, R.K.H., Yoneyama, T., Chame, H.C., & Visani, V. (2001). "The successive projections algorithm for variable selection in spectroscopic multicomponent analysis." *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73. Cited 1603 times.

**Protocol:** SPA selects minimally collinear variables through forward selection with projection operations. It evaluates candidate subsets using MLR (or PLS) with cross-validation on the calibration data to choose the optimal number of variables.

**Key point:** SPA runs on full calibration data. The CV inside SPA selects among candidate variable counts. Final performance is reported on a **separate prediction/test set** (RMSEP).

**Per-fold refitting?** No. SPA runs once on full calibration data.

### 2.3 iPLS (Nørgaard et al. 2000)

**Paper:** Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.P., Munck, L., & Engelsen, S.B. (2000). "Interval partial least-squares regression (iPLS): A comparative chemometric study with an example from near-infrared spectroscopy." *Applied Spectroscopy*, 54(3), 413-419. Cited 1747 times.

**Protocol:** iPLS divides the spectrum into equal intervals, builds separate PLS models on each interval using LOO-CV on the full data, and compares each interval's RMSECV to the full-spectrum RMSECV. This identifies informative spectral regions.

**Key point:** iPLS runs once on full data. The interval RMSECV values are used as a *diagnostic comparison* to identify informative regions. Nørgaard et al. note that this RMSECV is not an unbiased estimate when used for model/interval selection.

**Per-fold refitting?** No. iPLS runs once on full data. The interval boundaries are determined once, and the comparison is across intervals.

---

## 3. Is dasp's Implementation Matching the Published Methodology?

### 3.1 Answer: YES, with a critical caveat

**YES** — dasp's implementation matches the published methodology in the narrow sense that all four canonical methods (CARS, UVE, SPA, iPLS) run variable selection on the full calibration data once, producing a frozen importance/subset, and then use that frozen selection for subsequent model evaluation.

**However** — the canonical papers use a **separate independent test set** for final performance reporting, whereas dasp reports **cross-validation scores (R²cv, RMSECV) after using the same data for both variable selection and CV evaluation**.

This is confirmed by the code at `search.py:2552-2563` where `cars_selection(X_transformed_varsel, y_np, ...)` is called on the full preprocessed data, and the resulting importances are used at `search.py:2829` to create frozen `top_indices`, which are then passed directly into `_run_single_config()` as `subset_indices`. The `_run_single_config()` function performs K-fold CV on `X[:, subset_indices]` — the data pre-subsetted by the full-data selection.

The same pattern holds for all 15+ varsel methods in the audit (lines 2314-2795): importances are computed on full `(X_transformed_varsel, y_np)`, then pre-subsetted via `top_indices` before CV.

### 3.2 What the Canonical Papers Do vs. What dasp Does

| Protocol Step | Canonical Papers (CARS, UVE, SPA, iPLS) | dasp Implementation |
|---|---|---|
| Variable selection runs on | Full calibration data, once | Full calibration data, once |
| Selection produces | Frozen wavelength list | Frozen `top_indices` array |
| Performance metric reported | **RMSEP on separate test set** | **R²cv / RMSECV from cross-validation on same data** |
| Per-fold varsel refitting? | **No** | **No** |
| Selection bias in reported metric? | **No** (RMSEP is unbiased) | **Yes** (R²cv is inflated because varsel used test-fold data) |

The canonical papers avoid selection bias by reporting **external test set performance**. dasp introduces selection bias by reporting **cross-validation performance** where the same data was used for both variable selection and the CV evaluation.

---

## 4. Is the Audit's "LEAKY" Framing Correct or Wrong?

### 4.1 Verdict: The "LEAKY" framing is **correct**, with nuance

The audit's classification is technically correct: dasp's current workflow does leak information from the test folds into the variable selection step, producing optimistically biased CV scores. This is not a matter of sklearn convention — it is a statistical fact that has been documented in the chemometrics literature.

### 4.2 The Nuance: Published Papers Avoid This Differently

The canonical chemometrics papers (Li 2009, Centner 1996, Araujo 2001, Nørgaard 2000) do **not** refit variable selection per CV fold. They avoid selection bias by a different route:

- **They use a separate, held-out test set for final performance reporting**, not cross-validation on the same data used for variable selection.

This means the canonical workflow is:
```
Calibration data → CARS/UVE/SPA/iPLS → selected wavelengths → build model → RMSEP on test set
```

dasp's workflow is:
```
Calibration data → CARS/UVE/SPA/iPLS → selected wavelengths → CV on calibration data → R²cv (LEAKY)
```

The correct unbiased workflow (per sklearn conventions and the doubly-validated rdCV literature) would be:
```
For each CV fold:
    Training fold → CARS/UVE/SPA/iPLS → selected wavelengths → build model → predict on test fold
Aggregate → R²cv (UNBIASED)
```

### 4.3 The Chemometrics Literature Is Aware of This Issue

Multiple chemometrics-specific papers have identified the same problem:

1. **Filzmoser, Liebmann & Varmuza (2009)**, *J. Chemometrics*, 23, 160-171. "Repeated double cross validation (rdCV) is a strategy for (a) optimizing the complexity of regression models and **(b) for a realistic estimation of prediction errors** when the model is applied to new cases." rdCV explicitly places variable selection inside the outer CV loop to avoid selection bias.

2. **Baumann (2003)**, *TrAC Trends in Analytical Chemistry*, 22(5), 337-349. "The commonly applied leave-one-out cross-validation has a strong tendency to overfitting" when used as the objective function for variable selection.

3. **Gidskehaug, Anderssen & Alsberg (2008)**, *Chemometrics and Intelligent Laboratory Systems*, 93(1), 1-11. "This introduces a **downward selection bias** in the resulting estimate" when variable selection is performed outside the CV loop.

4. **Westad & Marini (2015)**, *Analytica Chimica Acta*, 893, 14-24. Tutorial on validation in chemometrics: "Cross Model Validation (CMV), also known as **double cross validation**, efficiently can be applied for variable selection... thus it acts as a filter which **reduces the risk of overfitting**."

5. **Shi, Westerhuis, et al. (2019)**, *Bioinformatics*, 35(6), 972-980. "**No algorithm has implemented variable selection within repeated double cross validation**... induce **selection bias** and model overfitting during variable selection."

6. **Király & Tóth (2025)**, *Journal of Chemometrics*, DOI: 10.1002/cem.3600. "Cross-validation parameters became overoptimistic if they are used in hyperparameter selection of models or in variable selection. We call this effect **parameter leakage**."

7. **Ambroise & McLachlan (2002)**, *PNAS*, 99(10), 6565-6570. From the bioinformatics community but making the identical statistical point: "The cross-validated error is calculated **without allowance for the selection bias**. There is no allowance because the rule is either tested on tissue samples that were used in the first instance to select the genes being used in the rule or because the **cross-validation of the rule is not external to the selection process**; that is, **gene selection is not performed in training the rule at each stage of the cross-validation process**."

### 4.4 It Is NUANCED — Here Is the Honest Assessment

The field **knows** about the issue (documented since at least 2002 in genomics and 2003 in chemometrics), but **routinely publishes** using the leaky protocol anyway because:

1. **External test sets are the canonical solution in chemometrics.** Many NIR/chemometrics papers follow the Li 2009 protocol: run CARS/UVE/SPA on full calibration, then report RMSEP on a separate test set. This is unbiased if the test set was truly held out before any modeling or selection.

2. **Per-fold varsel refitting is computationally expensive.** Running CARS (50 MC iterations × internal K-fold CV) inside each outer CV fold means the total computation scales by K_outer × K_inner. For CARS with 50 iterations, 5-fold inner CV, and 5-fold outer CV, this is 5 × 50 × 5 = 1250 PLS fits vs. 50 × 5 = 250 in the one-shot approach.

3. **Small datasets in chemometrics make per-fold varsel unstable.** With N=50-200 samples common in NIR spectroscopy, running CARS on 80% of the data (40-160 samples) per fold can produce highly variable selections since CARS relies on Monte Carlo subsampling.

4. **The practical impact varies by method and dataset size.** With N>500, the per-fold variability is small and the optimistic bias is minor (maybe R²cv inflation of ~0.01-0.03). With N<100, the bias can be substantial (R²cv inflation of 0.05-0.15 as estimated in the audit).

5. **Most published CARS papers do not report per-fold CV scores as their primary metric.** They use RMSEP on a held-out test set. The issue arises specifically when R²cv/RMSECV from a CV-on-selected-variables pipeline is used as the primary reported metric — which is exactly what dasp does.

### 4.5 What dasp Does That the Canonical Papers Don't

dasp's specific problem is **reporting R²cv from cross-validation as the primary performance metric**, where the CV is performed on data that was already used for variable selection. The canonical papers (Li 2009, Centner 1996, etc.) avoid this by reporting RMSEP on an external test set.

dasp's UI prominently displays R²cv and RMSECV as the headline metrics. Users make model selection decisions based on these CV scores. This is the exact scenario where selection bias matters most.

---

## 5. Recommendation

### 5.1 The varsel-leakage refactor should proceed, but with nuance

**Proceed with the refactor**, but reframe it:

1. **The refactor is statistically correct.** Per-fold varsel refitting (the VarselTransformer/Pipeline approach) produces unbiased CV estimates. This is the right thing to do when CV scores are the primary performance metric that users rely on for model selection.

2. **The current code is NOT matching the "standard practice" in published chemometrics.** The published papers use external test sets, not CV-on-selected-variables. dasp uses CV-on-selected-variables as the primary metric, which is the biased case.

3. **The correct framing is NOT "dasp deviates from standard chemometrics" but rather "dasp uses CV as the performance metric but doesn't protect CV from selection bias."** The canonical chemometrics papers avoid the bias by using external test sets; dasp can't rely on that escape hatch because its primary selection metric is CV-based.

### 5.2 Implementation should include an OPTIONAL strict mode

Rather than making per-fold varsel the only option, provide two modes:

- **Standard mode (current behavior):** Variable selection on full calibration data, followed by CV. This matches the published CARS/UVE/SPA/iPLS protocol. The resulting R²cv is optimistic, but matches what most published papers report (since they report RMSEP, not R²cv-after-varsel-CV). Label the output clearly: "R²cv (variable selection on full calibration data — may be optimistic)".

- **Strict mode (new, per-fold varsel):** Variable selection inside each CV fold via VarselTransformer. Produces unbiased R²cv. More computationally expensive but statistically sound. Label: "R²cv (unbiased, with per-fold variable selection)".

This is analogous to how sklearn's `Pipeline` vs. manual feature selection works — you *can* do it the quick way if you accept the bias, but the principled way is recommended.

### 5.3 Why not just revert

Reverting would leave dasp reporting biased R²cv scores with no warning. The audit correctly identified a real statistical issue. The refactor should proceed, but the documentation should be clear that this is about **correcting CV-based performance estimation**, not about "fixing a bug in the published methods."

### 5.4 Specific concerns about the Phase 2 refactor plan

1. **Computational cost:** Per-fold CARS is ~5× more expensive (5-fold outer × 1 CARS run vs. 1 CARS run). With 50 MC iterations and 5-fold inner CV per iteration, a single CARS run already takes significant time. Per-fold refitting multiplies this by K_outer. This should be documented.

2. **Small-N instability:** With N<100, per-fold CARS on 80% of the data may select very different variables across folds. This is statistically correct behavior (the variance IS real), but it means the `selected_indices_` will differ between folds. The VarselTransformer must handle variable-count mismatches across folds.

3. **The importance-based method (`importance` varsel) has a chicken-and-egg problem:** To compute VIP/feature_importances, you need a fitted model. To fit the model, you need selected variables. The per-fold solution (fit on all features first, then subset) is sound but adds per-fold pre-computation.

---

## 6. Specific Quotes from Li 2009

### 6.1 CARS Internal CV vs. External Validation

From the paper abstract:

> "In each sampling run, a fixed ratio (e.g. 80%) of samples is first randomly selected to establish a calibration model. Next, based on the regression coefficients, a two-step procedure including exponentially decreasing function (EDF) based enforced wavelength selection and adaptive reweighted sampling (ARS) based competitive wavelength selection is adopted to select the key wavelengths. **Finally, cross validation (CV) is applied to choose the subset with the lowest root mean square error of CV (RMSECV).**"

This describes CARS's *internal* CV — used to select which iteration's variable subset to keep. It is NOT the same as performing per-fold varsel in an outer CV loop.

From the paper's "Notation" section:

> "Suppose the number of MC sampling runs of CARS is set to N. With this setting, CARS will sequentially select N subsets of wavelengths."

This confirms CARS produces N candidate subsets and picks the best — all on the calibration data. No per-fold re-running is described.

### 6.2 Separate Test Set for Final Evaluation

From the paper's results section (Section 3, "Results and discussion"):

The evaluation uses a simulated dataset (SIMUIN, 25 samples) and a real NIR corn dataset. For the corn dataset, the data was split into calibration and prediction (test) sets. The RMSEP (root mean square error of prediction) on the **separate test set** is the performance metric, NOT the RMSECV from CARS's internal selection.

Li et al. explicitly compare CARS's prediction results with full-spectrum PLS, MC-UVE, and MWPLSR using **prediction error on the test set**:

> "The results reveal an outstanding characteristic of CARS that it can usually locate an optimal combination of some key wavelengths which are interpretable to the chemical property of interest. Additionally, our study shows that **better prediction is obtained by CARS** when compared to full spectrum PLS modeling, Monte Carlo uninformative variable elimination (MC-UVE) and moving window partial least squares regression (MWPLSR)."

"Better prediction" here means lower RMSEP on the held-out test set — not lower RMSECV from cross-validation.

### 6.3 No Discussion of Per-Fold Refitting

The paper does NOT discuss refitting variable selection within cross-validation folds. The entire CARS algorithm is presented as running on the calibration data once, producing a subset, and then evaluating that subset on a separate test set. The CARS-internal RMSECV is used for model selection (which iteration to keep), not for performance estimation.

### 6.4 No Discussion of Overfitting Risk from Variable Selection

Li et al. 2009 do NOT discuss the risk that variable selection on the full calibration data could bias CV estimates. The paper's contribution is the CARS algorithm itself, not a validation methodology paper. The implicit assumption is that external test-set validation (RMSEP) is the final performance metric.

---

## 7. Summary Table

| Method | Paper | Varsel on full data? | Per-fold varsel? | Internal CV? | Final metric reported | Unbiased? |
|--------|-------|---------------------|-------------------|--------------|----------------------|------------|
| CARS | Li 2009 | Yes | No | Yes (within MC iterations, for subset selection) | RMSEP on test set | Yes (RMSEP) |
| UVE | Centner 1996 | Yes | No | Yes (LOO for coefficient stability) | RMSEP on test set | Yes (RMSEP) |
| SPA | Araujo 2001 | Yes | No | Yes (for selecting n_vars) | RMSEP on test set | Yes (RMSEP) |
| iPLS | Nørgaard 2000 | Yes | No | Yes (LOO per interval) | RMSEP + RMSECV comparison | Partially (RMSEP unbiased, RMSECV biased) |
| **dasp (current)** | — | Yes | No | Yes (K-fold for model scoring) | R²cv / RMSECV (same data) | **No — biased** |

---

## 8. Key References

1. Li, H.D. et al. (2009). "Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration." *Analytica Chimica Acta*, 648(1), 77-84. DOI: 10.1016/j.aca.2009.06.046
2. Centner, V. et al. (1996). "Elimination of uninformative variables for multivariate calibration." *Analytical Chemistry*, 68(21), 3851-3858. DOI: 10.1021/ac960309j
3. Araujo, M.C.U. et al. (2001). "The successive projections algorithm for variable selection in spectroscopic multicomponent analysis." *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73. DOI: 10.1016/S0169-7439(01)00119-8
4. Nørgaard, L. et al. (2000). "Interval partial least-squares regression (iPLS): A comparative chemometric study with an example from near-infrared spectroscopy." *Applied Spectroscopy*, 54(3), 413-419. DOI: 10.1366/0003702001949500
5. Ambroise, C. & McLachlan, G.J. (2002). "Selection bias in gene extraction on the basis of microarray gene-expression data." *PNAS*, 99(10), 6565-6570. DOI: 10.1073/pnas.102102699
6. Filzmoser, P., Liebmann, B. & Varmuza, K. (2009). "Repeated double cross validation." *Journal of Chemometrics*, 23, 160-171. DOI: 10.1002/cem.1225
7. Baumann, K. (2003). "Cross-validation as the objective function for variable-selection techniques." *TrAC Trends in Analytical Chemistry*, 22(5), 337-349. DOI: 10.1016/S0165-9936(03)00502-4
8. Gidskehaug, L., Anderssen, E. & Alsberg, B.K. (2008). "Cross model validation and optimisation of bilinear regression models." *Chemometrics and Intelligent Laboratory Systems*, 93(1), 1-11. DOI: 10.1016/j.chemolab.2008.01.003
9. Westad, F. & Marini, F. (2015). "Validation of chemometric models – a tutorial." *Analytica Chimica Acta*, 893, 14-24. DOI: 10.1016/j.aca.2015.07.019
10. Shi, L. et al. (2019). "Variable selection and validation in multivariate modelling." *Bioinformatics*, 35(6), 972-980. DOI: 10.1093/bioinformatics/bty738
11. Király, P. & Tóth, G. (2025). "Being Aware of Data Leakage and Cross-Validation Scaling in Chemometric Model Validation." *Journal of Chemometrics*. DOI: 10.1002/cem.3600
12. Mehmood, T. et al. (2012). "A review of variable selection methods in Partial Least Squares Regression." *Chemometrics and Intelligent Laboratory Systems*, 118, 62-69. DOI: 10.1016/j.chemolab.2012.07.010