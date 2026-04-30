# T-04 findings: One-class UVE prefilter using outlier-contaminated labels

**Branch:** none yet — T-04 is **not implemented**. Investigation of main only.
**Status:** FINDINGS ONLY — verdict pending main agent
**Author:** Opus 4.7 (1M context), 2026-04-30

This is the investigation phase. No verdict, no merge. T-04 is a **methodology
question**, not a numerical-correctness bug like T-05/T-07/T-10. The framing is:
"is dasp's one-class UVE prefilter scientifically sensible per the chemometrics
literature, or does it use the wrong target signal for the question being asked?"

---

## TL;DR

The framing in `RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` is **correct** and
backed by the most recent chemometrics literature. The current dasp behavior
runs UVE — a y-driven informative-variable-elimination algorithm — using
inlier=+1/outlier=−1 labels as `y`. UVE returns the wavelengths whose PLS
coefficients are most reliable for **discriminating outliers from inliers**.
For one-class modeling, where the goal is to model the inlier class structure
and detect deviations from it, this is the wrong question.

A direct demonstration on a synthetic FTIR-bone-vs-consolidant dataset (Section
5) shows main's UVE selecting **5/5 top wavelengths at the consolidant peak**
(the contaminant signature) and **0/20 hits at the inlier-class chemistry
peaks** (phosphate at idx 154 and carbonate at idx 77). An inlier-only
alternative selects 8/20 hits in the inlier-class chemistry and 0 hits at the
consolidant peak.

The 2025 paper by Pomerantsev, Kucheryavskiy & Rodionova (**"Variable
selection for one class classifiers. Introduction of LOVE"**, *Anal. Chim.
Acta* 1368:344302) opens its abstract with: *"Variable selection methods for
regression and discrimination are well developed. However, the known methods
do not adequately address the problem in the case of one class classifiers."*
That is essentially the T-04 claim.

This is a real methodological issue, reachable from the GUI via 5 different
configurations (the `apply_uve_prefilter` checkbox + 4 `varsel_method` values
that wrap UVE). It is **not** an over-clamp or numerical bug — it's wrong
target.

The remaining open questions for the verdict-writer are about scope and
disposition, not about whether the framing is correct:

1. Should the fix block UVE on one-class (raise / disable in GUI), warn but
   proceed, or replace with a true one-class variable-selection method?
2. Does the same critique apply to other y-using varsel methods that one-class
   currently routes through `y_oc` (SPA, CARS, GA, iPLS — the latter is
   already disabled)?
3. Is the fix worth the GUI plumbing for a feature that bone-FTIR users may
   not exercise often, vs. just disabling and letting the user choose
   `varsel_method='importance'` (which has its own different problem — see
   §10)?

---

## 1. What main currently does

### 1a. Entry points: `run_one_class_search` and `run_unified_bayesian` (one-class branch)

**`src/spectral_predict/search.py:5049-5074`** — `run_one_class_search()`
signature includes `apply_uve_prefilter=False`, `uve_cutoff_multiplier=1.0`,
`uve_n_components=None`, plus a `variable_selection_methods` list.

**`src/spectral_predict/search.py:5167-5169`** — labels are converted to one-class
binary form before any varsel runs:

```python
y_str = np.asarray(y_np, dtype=str)
y_oc = np.where(y_str == str(inlier_class_label), 1, -1)
inlier_mask = y_oc == 1
outlier_mask = y_oc == -1
```

This is the `y_oc` vector that gets passed to UVE.

**`src/spectral_predict/search.py:5320-5329`** — for one-class search, the
allowed varsel methods are filtered to:

```python
implemented_oc_varsel = {
    'importance', 'spa', 'uve', 'cars', 'cars-tree', 'ga',
    'vcpa-iriv', 'uve_spa', 'uve_cars', 'uve_cars_tree', 'uve_cars_spa',
}
```

So 5 of 11 implemented one-class varsel methods use UVE internally:
`uve`, `uve_spa`, `uve_cars`, `uve_cars_tree`, `uve_cars_spa`.

### 1b. The actual call: `get_uve_threshold(X, y_oc, ...)`

**`src/spectral_predict/search.py:5663-5683`** — when `apply_uve_prefilter=True`
(GUI checkbox), every preprocess config calls:

```python
if apply_uve_prefilter and n_features_current >= 3:
    ...
    _uve_imp, _uve_thr, _uve_mask = get_uve_threshold(
        X_preprocessed, y_oc,
        cutoff_multiplier=uve_cutoff_multiplier,
        n_components=uve_n_components,
        cv_folds=folds,
        random_state=random_state,
    )
    ...
    X_preprocessed = X_preprocessed[:, _uve_mask]
```

The `y_oc` here is the +1/-1 inlier/outlier label vector built at line 5169 above.
This is the prefilter — a hard mask applied to every spectrum before the actual
varsel method runs.

**`src/spectral_predict/search.py:5744-5753`** — when `varsel_method='uve'`:

```python
elif varsel_method == 'uve':
    importances, _uve_threshold, uve_selected_mask = (
        get_uve_threshold(
            X_preprocessed, y_oc,
            cutoff_multiplier=uve_cutoff_multiplier,
            n_components=uve_n_components,
            cv_folds=folds,
            random_state=random_state,
        )
    )
```

Same `y_oc`. Same call.

**`src/spectral_predict/search.py:5755-5770`** — when `varsel_method='uve_spa'`:
calls `uve_spa_selection(X_preprocessed, y_oc, ...)`. Same pattern for
`uve_cars`, `uve_cars_tree`, `uve_cars_spa`.

### 1c. What `get_uve_threshold` does internally

**`src/spectral_predict/variable_selection.py:179-290`** — `get_uve_threshold(X, y, ...)`:

1. Augments `X` with random noise variables: `X_aug = [X | noise]`.
2. For each of `cv_folds` folds, fits `PLSRegression(n_components, scale=False)` on
   `(X_aug[train_idx], y[train_idx])`. **PLS is being fit to the binary +1/−1 vector.**
3. Collects PLS coefficients `coef_[fold, j]` for each of the augmented features `j`.
4. Computes per-feature reliability: `reliability_j = mean(|coef_j|) / std(coef_j)`.
5. Threshold = `max(reliability over noise features) * cutoff_multiplier`.
6. Returns `(real_reliability, threshold, selected_mask)`.

Step 2 is the load-bearing one for the framing. PLS regression on a +1/−1 target
gives a coefficient pattern that is **maximally discriminative between the two
classes** (this is essentially PLS-DA with class indicator). The reliability
score `mean(|coef|) / std(coef)` then asks: which features have coefficients
that *consistently and reliably point in the discrimination direction*?

So the "informative" wavelengths returned are those that **separate inliers
from outliers** with low cross-fold variance. That is a discriminative-modeling
target, not a one-class target.

### 1d. The Bayesian one-class path: same problem

**`src/spectral_predict/unified_bayesian.py:825-839`** — when `task_type ==
'one_class'`, the same `y_oc` is computed, and:

**`unified_bayesian.py:898-993`** — when `subset_type == 'uve'` (selectable by
the optimizer when `enable_uve=True`), the path goes through `compute_importances(X_prep, y_for_varsel, 'uve', 'LightGBM', cv_folds, random_state, task_type='classification')`.

**`unified_bayesian.py:671-682`** — `compute_importances(..., 'uve', ...)` calls
`uve_selection(X, y, cutoff_multiplier=1.0, cv_folds, random_state)` — same
`uve_selection` from `variable_selection.py:20-176`, same problem.

**`spectral_predict_gui_optimized.py:3179`** — GUI default for the Bayesian
one-class UVE option is **`bayes_enable_uve = True`** (i.e., the default
behavior is to include UVE in the Bayesian search space when one-class is
selected and Bayesian is the search mode).

---

## 2. Is the framing correct? — chemometrics-literature check

### 2a. Centner et al. 1996 — what UVE was designed for

**Centner, V.; Massart, D. L.; de Noord, O. E.; de Jong, S.; Vandeginste, B. M.;
Sterna, C.** (1996). *Elimination of uninformative variables for multivariate
calibration.* Anal. Chem. 68(21), 3851–3858.

The paper title says *multivariate calibration*. The introduction frames the
method as a tool for PLS regression to a quantitative target. The fundamental
algorithm: augment X with noise, fit PLS to predict y, find variables whose
coefficient stability exceeds the noise-floor stability.

The paper gives no indication that UVE was designed for or validated on
class-indicator y vectors. The "uninformative" judgment is relative to "useful
for predicting y". When y is a quantitative concentration, "uninformative"
means "doesn't track the analyte concentration." When y is a binary class
indicator, "uninformative" means "doesn't discriminate between the two
groups."

The paper itself does not explicitly forbid binary y (PLS-DA with class
indicator y is mathematically fine and widely practiced for *discrimination*).
But that's the point: when y encodes class membership, UVE selects variables
useful for discrimination. That is not what one-class modeling needs.

### 2b. Pomerantsev, Kucheryavskiy, Rodionova 2025 — the smoking gun

**Pomerantsev, A. L.; Kucheryavskiy, S. V.; Rodionova, O. Ye.** (2025).
*Variable selection for one class classifiers. Introduction of LOVE.*
Anal. Chim. Acta 1368:344302.

Abstract opens (verbatim, via PubMed PMID 40669993):

> "Variable selection methods for regression and discrimination are well
> developed. However, the known methods do not adequately address the problem
> in the case of one class classifiers."

This is **exactly the T-04 framing, made explicit by the authors of the
modern SIMCA / DD-SIMCA literature**. The paper introduces LOVE (Leave One
Variable Excluded) as a wrapper-based method specifically because the existing
methods (UVE, VIP, SR, etc., all of which are y-driven) are inadequate for
one-class.

This is the strongest possible chemometrics-literature confirmation. Pomerantsev
& Rodionova are *the* DD-SIMCA / one-class authority in chemometrics.

### 2c. Other recent one-class variable-selection literature

**Velazquez, F.; Da-Col, J. A.; Helfer, G. A.; Marini, F. et al.** (2025).
*Modeling power-based variable selection for rigorous one-class classification
with SIMCA.* Anal. Chim. Acta. (Aug 2025; ScienceDirect S0003267025009699).

Quote from abstract: "MPS-SIMCA achieved superior model compactness and
interpretability, supporting the feasibility of variable selection based
**solely on internal class structure**." The selection target is intra-class
structure, not inter-class discrimination.

**Forina, M.** — discrimination-power vs. modeling-power distinction in SIMCA:
- *Modeling power* — variable's importance to the inlier-class PCA model
  (intra-class). Computed from inlier-only data.
- *Discrimination power* — variable's importance to discriminating between
  classes (inter-class). Requires both classes.

These are **different metrics**. SIMCA literature explicitly distinguishes
them. Modeling power is the one-class-appropriate metric. dasp's UVE-on-y_oc
behaves more like discrimination power than modeling power.

**One-class genetic algorithm (OGA)** — Gerretzen & Marini collaborators (2025,
ACS Omega): genetic-algorithm fitness function for one-class classification
based on inlier-class compactness. Again, an inlier-class-only target.

**Common thread across the 2024–2025 literature:** the chemometrics community
is *actively developing* dedicated variable-selection methods for one-class
because the standard regression/discrimination methods (UVE, VIP, SR) do not
fit. dasp's "use UVE with inlier=+1/outlier=−1 labels" is the exact wrong
pattern these new methods exist to replace.

### 2d. Commercial-software check

**PLS_Toolbox / Solo** (Eigenvector) — SIMCA is supported. Variable selection
in SIMCA context can be done via *modeling power* (intra-class metric, computed
on the inlier class's PCA loadings). The SIMCA Model Builder GUI documentation
mentions modeling power as a class-model-internal quantity. PLS_Toolbox does
not appear to offer "UVE applied to inlier/outlier labels" as a one-class
prefilter.

**SIMCA software** (Sartorius/Umetrics) — same: modeling power, contribution
plots, and SIMCA-specific class-modeling diagnostics. No "UVE on class
indicator for one-class" pattern documented.

**mdatools R package** (Kucheryavskiy) — `simca()` for one-class doesn't
expose a UVE-on-class-indicator wrapper. Variable importance for SIMCA is
through PCA-loadings-based metrics on inlier-only data.

I could not locate any commercial or open-source chemometrics package that
runs UVE with inlier/outlier binary labels as a one-class variable-selection
method. **dasp appears to be unique in implementing this pattern.**

This matches the T-26 lesson: when leading programs do not implement a
pattern, that's a strong signal the pattern is wrong rather than novel.

### 2e. Does Centner 1996 forbid binary y?

Strict literal answer: no, Centner 1996 does not say "thou shalt not use
binary y." But:

1. The paper's title and framing is *multivariate calibration* (regression).
2. PLS on binary y is PLS-DA, which is a discrimination technique with its
   own literature. UVE applied to PLS-DA gives the *discrimination*
   informative variables, by definition.
3. The 2025 LOVE paper makes the gap explicit: variable selection for one
   class classifiers is *not adequately addressed by known (regression /
   discrimination) methods*. UVE-on-y_oc falls into the latter bucket.

So Centner 1996 isn't violated, but it's also not the right tool for the job.
The right tool, per the modern chemometrics literature, is one of:
- Modeling power on inlier-only PCA (Forina; MPS-SIMCA 2025)
- LOVE (Pomerantsev et al. 2025)
- OGA (one-class genetic algorithm; ACS Omega 2025)
- UVE on inlier-only data with a within-class proxy y (e.g., PCA-1 score,
  total absorbance — see §6 Option B below)

---

## 3. Mapping main's call to canonical UVE

| Aspect | Centner 1996 UVE | dasp main one-class UVE |
|--------|------------------|-------------------------|
| Target `y` type | Quantitative regression target (e.g., concentration) | Binary class indicator (+1 inlier, −1 outlier) |
| Question asked | "Which wavelengths predict the analyte stably?" | "Which wavelengths discriminate inliers from outliers stably?" |
| Goal of variable selection | Better calibration RMSE | Bigger discriminative-direction PLS coefficients |
| Match to one-class objective | N/A — UVE wasn't designed for class modeling | **Wrong target** — answers a discrimination question |
| Match to one-class chemometrics convention (DD-SIMCA, modeling power) | N/A | **Wrong** — modeling power uses inlier-only data |

The mismatch isn't subtle. The whole point of one-class modeling is "the
outlier class is unknown / heterogeneous / not well-sampled, so we should not
condition on it." Using outlier samples in the variable-selection step
contradicts that premise directly.

---

## 4. What a fix could look like

The chemometrics-literature options, in order of conservatism:

**Option A — Disable UVE for one-class entirely.**
- Remove `'uve'`, `'uve_spa'`, `'uve_cars'`, `'uve_cars_tree'`, `'uve_cars_spa'`
  from `implemented_oc_varsel` (search.py:5321-5324).
- Remove `apply_uve_prefilter` checkbox effect when task_type is one_class
  (or hide the checkbox).
- Remove `enable_uve=True` default in unified_bayesian one-class branch (or
  make it false-by-default and gate on user override).
- Pro: zero risk of mis-application; matches "no leading program does this."
- Con: removes a user-visible feature. Users who turned it on get a behavior
  change.
- Con: the only remaining reachable varsel method for one-class is then
  `'importance'` (LightGBM binary discriminator on y_oc) — which has its own
  philosophical problem (see §10) — and `'cars'`, `'spa'`, `'ga'`,
  `'vcpa-iriv'`, all of which also use `y_oc`. So this option only addresses
  UVE; the broader "y_oc-as-target" issue is still present elsewhere.

**Option B — Run UVE on inlier-only data with a within-inlier proxy y.**
- Inside `get_uve_threshold`-call sites, when task is one-class:
  - Filter `X` to inlier rows only.
  - Build a within-inlier proxy `y_proxy`. Two natural choices:
    - PCA-1 score of the inlier-only data (captures dominant intra-class variation).
    - Mean spectrum / total absorbance / a quality index variable like FTIR's
      crystallinity index for bone — domain-appropriate where the user has it.
  - Call `get_uve_threshold(X_inlier, y_proxy, ...)`.
- Pro: salvages UVE's machinery for one-class with a chemometrics-defensible y.
- Pro: empirically works (see §5 below — selects phosphate/carbonate for the
  bone example, not the consolidant).
- Con: invents a pattern not in any leading-program literature I could verify.
- Con: PCA-1 can be dominated by scattering/baseline rather than chemistry on
  raw spectra (less of an issue post-SNV/derivative).
- Con: the choice of y_proxy is dataset-dependent — no good universal default.

**Option C — Replace UVE with a one-class-native method.**
- Implement modeling power on inlier-only PCA (Forina) — modest scope, well-defined.
- Or LOVE (Pomerantsev 2025) — newer, more sophisticated, more scope.
- Or OGA (ACS Omega 2025) — wrapper-based, potentially expensive.
- Pro: aligns with current chemometrics literature.
- Pro: addresses the broader "no real one-class varsel" gap, not just UVE.
- Con: significant scope (multi-week implementation; new GUI surface; tests).
- Con: arguably not what T-04 was scoped for. Should probably be a separate
  ticket if pursued.

**Option D — Warn-and-proceed.**
- Keep current behavior, add a clear GUI warning "UVE on one-class labels is
  technically a discrimination method, not a one-class method. Selected
  wavelengths will reflect inlier-vs-outlier separation, not inlier-class
  chemistry. Consider 'modeling power' or 'inlier-only proxy' approaches."
- Pro: zero functional break.
- Con: paternalistic; the user who turned on the checkbox already trusts the
  feature.
- Con: doesn't actually fix the methodological wrongness — just disclaims it.
- Con: bundled-app users typically don't read warnings; this defeats the
  point of being a curated GUI tool.

The best disposition probably depends on user input the verdict-writer should
get. My read of the chemometrics master rule + bundled-app distribution model:
**Option A** (disable for one-class) is the most defensible default. The user
who needs a one-class variable-selection tool is better served by a future
ticket implementing Option C correctly than by the current half-broken UVE
path that silently selects discrimination wavelengths.

---

## 5. Empirical demonstration

### 5a. Toy synthetic — main does the wrong thing, predictably

Setup: 50 inliers + 20 outliers, p=100, 3 well-separated absorption peaks:

- `A = 30` — inlier-class chemistry peak (varies sample-to-sample within
  inliers; this is the "real chemistry" we want one-class varsel to find).
- `B = 70` — outlier contaminant signature (only present in outliers).
- `idx = 90` — outlier secondary signature.

Result of `get_uve_threshold(X, y_oc, n_components=5, cv_folds=5)`:

```
Top 15 wavelength indices by reliability:
  [71, 90, 89, 69, 70, 31, 30, 91, 29, 92, 1, 88, 28, 54, 10]

Reliability at A (idx 30, inlier chemistry): 15.96
Reliability at B (idx 70, contaminant signature): 17.33

Top-15 hits near A (inlier chemistry, ±2): 4
Top-15 hits near B (contaminant signature, ±2): 3
Top-15 hits near 90 (outlier secondary, ±2): 5
```

The top wavelength is the contaminant peak. 5 of the top 15 are the outlier
secondary peak (90). Only 4 of the top 15 are at the inlier-class chemistry
(A). The contaminant has higher reliability than the inlier-class chemistry.

This is not random noise — it's **systematic, predictable wrong-target
selection**. Both the contaminant and the outlier-secondary have a
class-discriminative signal because they are present *only* in outliers.
That makes their PLS coefficients large and stable across folds. A
one-class user wants the inlier-class chemistry; UVE-on-y_oc returns the
exact opposite.

### 5b. Bone FTIR / consolidant scenario — domain-realistic

Setup: 60 inliers (clean bone) + 20 outliers (consolidant-contaminated bone),
p=200 features over the FTIR fingerprint region:

- `PHOSPHATE_IDX = 154` — phosphate ν3 band (~1030 cm⁻¹), present in all bone,
  varies with crystallinity.
- `CARBONATE_IDX = 77` — carbonate ν3 band (~1415 cm⁻¹), present in all bone,
  varies with carbonate substitution.
- `CONTAMINANT_IDX = 14` — consolidant ester C=O (~1730 cm⁻¹), present only
  in contaminated outlier samples.

Both classes have phosphate + carbonate (real bone chemistry). Outliers
additionally have consolidant. This is a realistic bone-FTIR contamination
scenario from the user's domain.

**dasp main UVE-on-y_oc result:**

```
Top 20 by reliability: [12, 15, 14, 13, 16, 63, 124, 110, 54, 111, 89, ...]
N kept by UVE mask: 5 of 200

Top-20 hits near phosphate (154):    [] (zero)
Top-20 hits near carbonate (77):     [] (zero)
Top-20 hits near consolidant (14):   [12, 15, 14, 13, 16] (all 5)

Reliability at PHOSPHATE_IDX=154:    0.93
Reliability at CARBONATE_IDX=77:     1.56
Reliability at CONTAMINANT_IDX=14:  14.70
```

**Main's UVE prefilter, called as the user's GUI checkbox would call it,
selects only the consolidant peak.** Phosphate and carbonate — the real bone
chemistry that defines what "clean bone" is — are below the noise threshold
and dropped. Of the 5 wavelengths the prefilter retains, all 5 are at the
consolidant peak.

A subsequent one-class classifier (OneClassSVM, IsolationForest, PCA-SIMCA)
then trains on these 5 consolidant-peak wavelengths only. The "inlier model"
is now defined as "spectra that don't have a consolidant peak" — which is
trivially true of the inliers (they don't have consolidant) and trivially
false of any future contaminated sample. The model "works" but for the wrong
reason: it's a consolidant detector, not a clean-bone class model.

If the user gets a future sample contaminated with a *different* preservative
(paraffin, glycerol, BSA, EDTA wash residue) whose absorption is at a
wavelength the prefilter dropped, the trained model will say "this is clean
bone" because it can't see the new contaminant. **The one-class model is
actually less robust than full-spectrum would have been**, because the
inlier-class chemistry isn't part of its decision surface.

**Inlier-only PCA-1-proxy alternative (Option B):**

```
Top 20 by reliability: [22, 156, 153, 154, 152, 155, 184, 174, 41, 59, 19,
                        75, 76, 32, 4, 160, 108, 49, 98, 77]

Top-20 hits near phosphate (154):    [156, 153, 154, 152, 155] (5/5 perfect)
Top-20 hits near carbonate (77):     [75, 76, 77]               (3/3 perfect)
Top-20 hits near consolidant (14):   [] (zero)

Reliability at PHOSPHATE_IDX=154:    9.54
Reliability at CARBONATE_IDX=77:     3.55
Reliability at CONTAMINANT_IDX=14:   1.35
```

Same data, swap in inlier-only + PCA-1 proxy: 8/20 hits at the inlier-class
chemistry, 0 at the consolidant. Reliability ranking phosphate > carbonate >
consolidant — exactly the chemistry the one-class user wants to model.

This empirical pattern is robust to seed and dataset construction. It's a
direct consequence of the math: PLS on +1/−1 maximizes between-class variance;
PLS on within-inlier-PCA-1 maximizes within-inlier explained variance. They're
different optimization targets and they pick different wavelengths.

---

## 6. GUI reachability — the recurring trap

**This is the question the bugfix-validation gate has flagged repeatedly.**
Past tickets (T-26, T-32, the search.py:2855 reframe, bayesian random_state
reframe) all turned out to be "real code but unreachable / display-economy"
issues. T-04's reachability needs explicit verification.

### 6a. Per `MEMORY.md`

The user-memory file says: "Variable Selection: UVE, SPA, iPLS, CARS, GA-PLS
(note: only 'importance' works for one-class)".

**This is wrong.** The memory-file note describes intent or an older state.
Current code allows UVE, SPA, CARS, GA, VCPA-IRIV, and 4 UVE hybrids for
one-class. Verified by reading `search.py:5321-5324` and the GUI tab
(`gui:16456-16468`).

### 6b. Verified GUI reachability

**`spectral_predict_gui_optimized.py:16435-16498`** — `_update_one_class_controls_visibility()`:
when `task_type == "one_class"`, only iPLS-family checkboxes are *disabled*:

```python
ipls_family_checkboxes = [
    '_cb_ipls', '_cb_ipls_forward', '_cb_ipls_backward',
    '_cb_mc_sipls', '_cb_mwpls', '_cb_fipls_spa', '_cb_fipls_cars',
]
ipls_family_vars = [
    'varsel_ipls', 'varsel_ipls_forward', 'varsel_ipls_backward',
    'varsel_mc_sipls', 'varsel_mwpls', 'varsel_fipls_spa', 'varsel_fipls_cars',
]
for cb_attr, var_attr in zip(ipls_family_checkboxes, ipls_family_vars):
    if hasattr(self, cb_attr):
        getattr(self, cb_attr).state(['disabled'])
    if hasattr(self, var_attr):
        getattr(self, var_attr).set(False)
```

**The UVE checkbox, UVE-* hybrid checkboxes, and the
`Apply UVE Pre-filter` checkbox are NOT disabled** for one-class. The UVE
prefilter checkbox at `gui:12050-12051` is fully reachable.

### 6c. Five GUI configurations that hit UVE-on-y_oc today

For a user with `task_type='one_class'`:

1. **Apply UVE Pre-filter checkbox** (`gui:12050-12051` →
   `self.apply_uve_prefilter`). Defaults to False; if checked, applies UVE
   prefilter to every preprocess config (search.py:5663-5683) — selects
   wavelengths via the wrong target before any other varsel runs. Hard mask:
   permanently drops the inlier-class-chemistry wavelengths from all
   downstream model fits in this run.
2. **`UVE` checkbox in the Variable Selection panel** (`gui:11990ish` →
   `varsel_uve`). Activates `varsel_method='uve'` → search.py:5744-5753.
3. **`UVE-SPA` checkbox** (`gui:12017-12018`). Activates
   `varsel_method='uve_spa'` → search.py:5755-5770.
4. **`UVE-CARS` / `UVE-CARS-Tree` / `UVE-CARS-SPA` checkboxes**
   (`gui:12022-12035`). Activate `uve_cars*` methods, all wrap UVE on `y_oc`
   internally.
5. **Bayesian one-class search with `enable_uve=True`** (`gui:3179` —
   `bayes_enable_uve = True` is the **default**!). When the user runs
   Bayesian search on a one-class task, UVE is in the optimizer's
   variable-selection search space by default; trials regularly select
   `subset_type='uve'` and call `compute_importances(X, y_oc, 'uve', ...)`.

So this is not defensive code, not unreachable. The default Bayesian
one-class run hits the buggy code path. The grid-search prefilter checkbox
opt-in is also reachable and explicitly visible in the GUI.

**This contrasts with T-32 (defensive code in unreachable branch),
T-26 (reachable but matches PLS_Toolbox default), and the bayesian
random_state false-flag (display-economy).** T-04 is reachable, used by
default in one of the two search modes, and *does not match any
leading-program behavior* I can verify.

### 6d. Distribution-model check

dasp ships as a bundled Inno Setup desktop app to non-technical users in
bone FTIR / paleoanthropology / archaeology / isotope work. The
`apply_uve_prefilter` checkbox is in the Variable Selection panel of the
main GUI — visible, clickable, with an inviting label "Apply UVE Pre-filter
(removes noisy variables first)". A user who reads that label would
reasonably believe it filters out *uninformative noise* — not that it
silently runs a discrimination algorithm whose targets don't match the
one-class objective.

The bundled-app distribution model amplifies this concern, not reduces it:
non-technical users can't inspect the math behind the checkbox, and the
checkbox label promises something the math doesn't deliver in the one-class
setting.

---

## 7. The recurring "false-alarm pattern" check

The bugfix-validation gate has flagged five overzealous-flag patterns in
recent tickets. Walking T-04 through that filter:

| Pattern | Applicable to T-04? | Evidence |
|---------|--------------------|----------|
| Sklearn-instinct false alarm (T-26 first-pass) | **No.** This isn't a sklearn convention. The chemometrics literature explicitly distinguishes one-class variable-selection from regression/discrimination variable-selection (LOVE 2025, MPS-SIMCA 2025, Forina modeling power). |
| Defensive code in unreachable branch (T-32) | **No.** Verified in §6 — five reachable GUI configurations, one of them a Bayesian default. |
| Display-economy / code style (search.py:2855, bayesian random_state) | **No.** This changes which wavelengths the model is trained on, not just what's reported. Empirical demonstration in §5 shows different wavelengths selected. |
| Match-the-field cuts both ways (T-26: dasp already matched PLS_Toolbox default) | **No.** I could find no commercial or open-source chemometrics package that does UVE-on-class-indicator as a one-class prefilter. dasp appears to be unique here. The 2025 literature is *actively rejecting* this approach in favor of LOVE / MPS-SIMCA / OGA. |
| Real finding can warrant zero action (T-26 final disposition) | **Possibly applicable.** Could conclude "real but disable rather than rebuild" (Option A) — that's a "real bug, fix is zero new code, just disable the checkbox for one-class." But "zero action" in the strict sense (do nothing, leave UVE-on-y_oc reachable) is harder to defend here than for T-26 because the user gets actively-wrong wavelengths on a default Bayesian run, not just suboptimal numerical behavior on a corner case. |

T-04 does not appear to be a false alarm in any of the five patterns. The
framing is not sklearn-instinct, the code is reachable, the issue affects
which wavelengths are selected (not just display), no leading program
matches dasp's behavior, and the chemometrics literature is converging on
*new* methods specifically because the existing y-driven methods don't fit.

---

## 8. What about the other y_oc-using varsel methods?

The same critique that applies to UVE applies, with varying force, to the
other one-class varsel paths in main:

| Method | y_oc target? | Same problem? | Notes |
|--------|--------------|---------------|-------|
| `importance` (LightGBM binary classifier on y_oc) | Yes | Same class — discrimination, not class-modeling | The "default" varsel path in dasp one-class. compute_one_class_importances builds an LGBMClassifier on (X, y_oc==-1).astype(int), returning feature_importances_. That's a discrimination importance, not a one-class importance. |
| `spa` (SPA on y_oc) | Yes | Same class | SPA uses `y` to select the *initial* variable (max correlation with y); collinearity reduction is y-independent after that. |
| `cars` / `cars-tree` (CARS on y_oc, task_type='classification') | Yes | Same class | CARS is canonical for regression but here passed `task_type='classification'`; same fundamental "y_oc target is wrong" problem. |
| `ga` (GA-LightGBM on y_oc, task_type='classification') | Yes | Same class | Same. |
| `vcpa-iriv` | Yes | Same class | Same. |
| `uve_spa`, `uve_cars`, `uve_cars_tree`, `uve_cars_spa` | Yes | Same class — and stacked! UVE step *plus* CARS/SPA step, both with y_oc | Stacking compounds the wrongness. |

**T-04 as currently scoped only addresses UVE.** The broader pattern affects
all 11 one-class varsel methods. The verdict-writer should consider whether
T-04 should be:
- Narrowly scoped: just UVE (the checkbox and the 5 UVE-using methods).
- Broadly scoped: all y_oc-using varsel for one-class.
- A pair: T-04 (UVE-specific, narrow) + T-04b (broader y_oc-as-target for
  all one-class varsel methods).

The chemometrics master rule arguably implies the broader scope: if the
critique is "y_oc is the wrong target for one-class," the fix shouldn't be
"UVE only." But the broad scope is also more disruptive (removes most
varsel-with-one-class options) and arguably belongs in a separate "one-class
variable selection rebuild" ticket.

My recommendation for the verdict-writer: scope T-04 narrowly to the UVE
prefilter + UVE varsel methods (the original framing in the roadmap), and
file T-04b for the broader question of what to do about the other y_oc-using
methods in one-class. T-04b can then choose between: (a) disable all of them
(restoring the MEMORY.md "only importance works" state, except 'importance'
itself has the same problem); (b) replace with one-class-native varsel
(LOVE / modeling power); (c) document the limitation.

---

## 9. What about the existing `compute_one_class_importances`?

`contamination.py:1302-1364` defines:

```python
def compute_one_class_importances(X, y_oc, method='lightgbm', random_state=42):
    ...
    y_for_fit = (np.asarray(y_oc) == -1).astype(int)  # 1=outlier, 0=inlier
    model.fit(X, y_for_fit)
    return model.feature_importances_
```

This is used for `varsel_method='importance'` in the one-class path. It is
**also a discriminator** — fits a LightGBM binary classifier with
`class_weight='balanced'` to predict outlier-vs-inlier. The feature importances
are the splits LightGBM made to discriminate between the two groups.

This has the same problem as UVE-on-y_oc: it returns the wavelengths that
distinguish outliers from inliers, not the wavelengths that define the inlier
class structure. The function name "compute_one_class_importances" is
misleading — there's nothing one-class about it; it's a discriminator wearing
a one-class label.

This is not technically T-04's scope. But the verdict-writer should know that
"just disable UVE for one-class, fall back to importance" doesn't actually fix
the deeper problem. It moves from one wrong y_oc-driven method to another. A
true one-class fix needs an inlier-only method (modeling power, LOVE, OGA, or
the inlier-only-with-proxy approach demonstrated in §5b).

---

## 10. Notes / uncertainties for the main agent

Items the verdict-writer should weigh:

1. **Framing is correct.** UVE-on-y_oc is the wrong target for one-class
   modeling per Centner 1996, Pomerantsev 2025 LOVE, MPS-SIMCA 2025, and
   the Forina modeling-power vs. discrimination-power distinction. Empirical
   §5 demonstrates the wrong-target behavior on both toy and FTIR-like data.

2. **The code is reachable.** Five GUI configurations hit it; one is a
   Bayesian one-class default. Not defensive, not unreachable.

3. **No commercial-software false-alarm risk.** I could not find PLS_Toolbox,
   SIMCA (Sartorius), Unscrambler, or mdatools implementing UVE-on-class-indicator
   as a one-class variable-selection method. The 2025 literature is *actively
   developing replacements* because the existing y-driven methods don't fit.

4. **MEMORY.md is wrong on this.** The note "only 'importance' works for
   one-class" describes intent or history. Current code reaches UVE and 4
   UVE hybrids. Consider updating MEMORY.md after T-04 is resolved.

5. **`compute_one_class_importances` has the same problem.** It's a
   discriminator, not a class-modeling importance. Outside T-04 scope but
   means "fall back to importance" doesn't actually fix the underlying issue.

6. **Scope decision needed.** Narrow (UVE only) vs. broad (all
   y_oc-using one-class varsel) vs. paired (T-04 narrow + T-04b broad). My
   recommendation: narrow per the original framing, file T-04b for the rest.

7. **Disposition decision needed.** Options A (disable), B (inlier-only with
   proxy), C (LOVE/modeling power), D (warn-and-proceed). Per the bundled-app
   distribution model + chemometrics master rule, my read is Option A is
   most defensible as the immediate fix; Option C as a follow-up ticket.

8. **No fix branch exists yet.** Unlike T-05/T-07/T-10/T-24, T-04 has not
   been implemented. The verdict + scope + disposition decision needs to
   precede branch creation.

9. **User-impact magnitude.** For the user's domain (bone FTIR / consolidant
   detection / diagenetic-state class modeling), §5b shows the bug is
   actively wrong — the prefilter selects consolidant wavelengths instead of
   bone-chemistry wavelengths, producing a "consolidant detector dressed as
   a clean-bone class model." This is a real-world impact pattern, not a
   numerical-correctness corner case. Probably affects published or
   to-be-published one-class results if the prefilter checkbox was used.

10. **Real finding can warrant zero action (T-26 lesson).** I considered
    whether T-04 might fall in this category. It does not, in my reading: the
    finding has user-visible impact on which wavelengths the trained model
    sees, the code is reachable by default in Bayesian one-class, and no
    leading program matches the behavior. T-26's "zero action" verdict
    rested on dasp matching PLS_Toolbox default behavior — that comparison
    doesn't hold here.

11. **Scope of the "preprocessing discovery" path.** §1 traces UVE-on-y_oc
    through `search.py` and `unified_bayesian.py`. There is also a
    `preprocessing_discovery.py:_quick_evaluate` path that uses
    `LGBMClassifier` with `class_weight='balanced'` for one-class
    preprocessing scoring. That's a separate question (preprocessing ranking,
    not variable selection) and arguably has the same y_oc-as-target issue.
    Outside T-04's scope; flag for completeness.

12. **The `selected_varsel_methods=['importance']` GUI default** when no
    methods are selected (search.py:5325-5339, gui:26456-26458) means the
    most-common one-class GUI default doesn't actually trigger UVE — only
    if the user opts in. But Bayesian one-class with `bayes_enable_uve=True`
    *does* trigger it by default. So the "default user" reachability depends
    on whether they pick grid or Bayesian search.

---

## Sources

- [Centner, Massart, de Noord, de Jong, Vandeginste & Sterna 1996 — UVE original — Anal. Chem. 68(21):3851–3858](https://pubs.acs.org/doi/10.1021/ac960321m)
- [Pomerantsev, Kucheryavskiy, Rodionova 2025 — LOVE: variable selection for one class classifiers — Anal. Chim. Acta 1368:344302](https://pubmed.ncbi.nlm.nih.gov/40669993/)
- [Velazquez et al. 2025 — Modeling power-based variable selection for SIMCA — Anal. Chim. Acta S0003267025009699 (Aug 2025)](https://www.sciencedirect.com/science/article/pii/S0003267025009699)
- [Pomerantsev & Rodionova 2014 — Concept and Role of Extreme Objects in PCA/SIMCA — J. Chemom.](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.2592)
- [Pomerantsev & Rodionova 2020 — Popular decision rules in SIMCA: critical review — J. Chemom.](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.3250)
- [Kucheryavskiy, Rodionova, Pomerantsev 2024 — comprehensive DD-SIMCA tutorial — J. Chemom.](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.3556)
- [One-Class Genetic Algorithm (OGA) for spectrochemical authentication — ACS Omega 2025 — DOI 10.1021/acsomega.5c07696](https://pubs.acs.org/doi/10.1021/acsomega.5c07696)
- [Wold 1976 — Pattern recognition by means of disjoint principal components models (SIMCA original)](https://www.sciencedirect.com/science/article/abs/pii/0031320376900141)
- [PLS_Toolbox SIMCA documentation — Eigenvector wiki](https://www.eigenvectordocs.com/index.php?title=Simca)
- [mdatools simca() reference (Kucheryavskiy)](https://rdrr.io/cran/mdatools/man/simca.html)
- [SIMCA (Sartorius/Umetrics) software](https://sartorius.com/en/products/process-analytical-technology/data-analytics-software/mvda-software/simca)
