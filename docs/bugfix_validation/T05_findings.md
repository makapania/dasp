# T-05 findings: VIP formula correctness

**Branch:** `fix/T05-vip-formula-fix` (HEAD `3269475`)
**Status:** FINDINGS ONLY — verdict pending main agent
**Author:** Opus 4.7 (1M context), 2026-04-30
**Worktree:** `.worktrees/T05-vip-formula-fix/`

This is the investigation phase. No verdict, no merge.

---

## 1. What the branch changes

Branch source-only diff vs main (other diff noise is just docs that didn't exist when the branch was cut):

| File                                          | Change                                          |
|-----------------------------------------------|-------------------------------------------------|
| `src/spectral_predict/models.py`              | Rewrites `compute_vip()` (lines ~1709-1751 → ~1710-1780) |
| `tests/test_vip_formula.py`                   | New file: 7 tests (164 lines)                   |

### 1a. The actual code change

**Main (buggy):**

```python
def compute_vip(pls_model, X, y):
    if isinstance(pls_model, PLSTransformer):
        pls_model = pls_model.pls_

    W = pls_model.x_weights_   # (n_features, n_components)
    T = pls_model.x_scores_    # (n_samples, n_components)

    # SSY: sum of squares of y explained by each component
    y = np.asarray(y).reshape(-1, 1)
    ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)   # <-- np.var(y) is per-COMPONENT shouldn't be a constant
    ssy_total = np.sum(ssy_comp)

    n_features = W.shape[0]
    weight = np.sum((W ** 2) * ssy_comp, axis=1)
    vip_scores = np.sqrt(n_features * weight / ssy_total)
    return vip_scores
```

The defining bug is on the `ssy_comp` line: `np.var(y, axis=0)` is a *single scalar* — the global Y variance — multiplied identically into every component's `T_a' T_a`. Per the canonical formula (next section), this term must be *per component* and proportional to the squared Y-loading `q_a` of that component.

**Branch (canonical):**

```python
W = np.asarray(pls_model.x_weights_)
T = np.asarray(pls_model.x_scores_)
Q = np.asarray(pls_model.y_loadings_)   # sklearn shape: (n_targets, n_components)
q = Q if Q.ndim == 1 else Q[0, :]       # univariate Y: row 0

n_features = W.shape[0]

ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)        # SSY_a = q_a^2 * (T_a' T_a)
ssy_total = float(np.sum(ssy_comp))

if ssy_total <= 0.0:
    return np.zeros(n_features, dtype=float)

col_norm_sq = np.sum(W ** 2, axis=0)
col_norm_sq = np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)
w_norm_sq = (W ** 2) / col_norm_sq                  # defensive: sklearn weights are already unit-norm

vip_scores = np.sqrt(n_features * (w_norm_sq @ ssy_comp) / ssy_total)
```

Differences vs main:
- `ssy_comp = (q^2) * (T_a' T_a)` instead of `var(y) * (T_a' T_a)` — uses **per-component** y-loading, not global y-variance.
- Adds `||W_a||^2` defensive normalization. (sklearn already returns unit-norm column weights, so this is a no-op for `PLSRegression`, but it makes the formula correct for non-sklearn callers.)
- Adds `ssy_total <= 0` short-circuit returning zeros (degenerate case where Y has no PLS-recoverable variance).

### 1b. Tests added (`tests/test_vip_formula.py`)

| Test                                                | Class           | Discriminator? | Purpose                                                   |
|-----------------------------------------------------|-----------------|----------------|-----------------------------------------------------------|
| `test_old_and_new_formulas_disagree_on_this_problem`| Canonical       | sanity         | Verifies fixture is sensitive (max\|diff\| > 1e-3)        |
| `test_compute_vip_matches_canonical_reference`      | Canonical       | YES            | Compares dasp output to inline canonical reference impl   |
| `test_compute_vip_does_not_match_old_buggy_formula` | Canonical       | YES            | Asserts dasp output != old buggy formula (catches no-op)  |
| `test_canonical_invariant_average_squared_vip_is_one`| Canonical      | sanity         | mean(VIP^2) == 1 — both old and new satisfy this          |
| `test_output_shape_and_nonnegativity`               | Canonical       | sanity         | Shape + non-negativity + finite                           |
| `test_compute_vip_n_components_1`                   | EdgeCases       | sanity         | Single component case (collapse case)                     |
| `test_compute_vip_zero_y_loadings_returns_zeros`    | EdgeCases       | YES            | Degenerate `y_loadings_ = 0` → expects all zeros          |

The test file documents which tests discriminate the bug vs which are sanity guards. Codex flagged the invariant test as a sanity check (both formulas pass it) — the test file explicitly labels it `[SANITY]` in the docstring.

### 1c. Out-of-scope sites (acknowledged in plan, deferred to T-05a)

The plan (`docs/plans/2026-04-29-T05-vip-formula-fix.md`) explicitly lists these as out-of-scope:

| File                                                      | Line | Note                                              |
|-----------------------------------------------------------|------|---------------------------------------------------|
| `src/spectral_predict/templates/variable_selection.py`    | 54   | `VIP_TEMPLATE` for exported standalone scripts    |
| `src/spectral_predict/nsga2_search.py`                    | ~627 | NSGA-II PLS feature-importance branch (per Codex) |

I checked these in main:
- `templates/variable_selection.py:54`: confirmed — `ssy_comp = np.sum(T**2, axis=0) * np.var(y_reshaped, axis=0)` (same bug).
- `nsga2_search.py:627`: confirmed — has its own inline buggy VIP. Code at line 627: `ssy_comp = np.sum(T**2, axis=0) * np.var(y_arr, axis=0)`. Same bug as `models.py:1738`. So NSGA-II has TWO VIP paths: an inline one (lines ~620-636, buggy) and a `get_feature_importances` call site at line 3316 (which goes through `models.py:compute_vip` and would benefit from T-05). Codex was correct.
- `contaminant_analysis.py:_compute_vip` (lines 971-1009): already uses the canonical formula in main — uses `Q[i, 0]**2 * (T[:, i].T @ T[:, i])`. So this site is *not* buggy. (Internal inconsistency: dasp had two VIP implementations on main, one canonical, one buggy.)

---

## 2. Canonical Wold 2001 VIP formula

**Primary reference:**
Wold, S.; Sjöström, M.; Eriksson, L. (2001). "PLS-regression: a basic tool of chemometrics." *Chemometrics and Intelligent Laboratory Systems* 58:109-130.

The canonical formula, as restated in the literature and in **every** open-source chemometrics package I could verify (see Section 6):

> For PLS dimension `a`:
> - Per-component explained Y sum-of-squares: `SSY_a = q_a^2 * (T_a' T_a)`
>   where `q_a` is the y-loading for component `a` and `T_a` is the X-scores vector for component `a`.
> - Total explained Y sum-of-squares: `SSY_cum = sum_a SSY_a`
> - VIP for variable `j`:
>   `VIP_j = sqrt( p * sum_a [ (W_{j,a} / ||W_a||)^2 * SSY_a ] / SSY_cum )`
>   where `p` is the number of features and `W` is the matrix of X-weights.

Equivalent restatement (Mehmood et al. 2012, Eq. 1):
The Mehmood, Liland, Snipen, Sæbø (2012) review paper "A review of variable selection methods in Partial Least Squares Regression" (*Chemometrics and Intelligent Laboratory Systems* 118:62-69) presents the same formula as Eq. 1, citing Wold 2001.

Chong & Jun (2005) "Performance of some variable selection methods when multicollinearity is present" (*Chemometrics and Intelligent Laboratory Systems* 78:103-112) is the reference cited by Eigenvector PLS_Toolbox and by `mevik/pls`'s `VIP.R`. Chong & Jun's formulation is algebraically identical to Wold 2001.

**Key invariant:** `mean_j(VIP_j^2) = 1`. This holds whenever `||W_a|| = 1` for every component (which is true by construction in sklearn's `PLSRegression` and in PLS_Toolbox).

---

## 3. What main currently computes

```python
ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)     # main, models.py:1738
```

Mapping to canonical:
- Canonical wants `SSY_a = q_a^2 * (T_a' T_a)`.
- Main computes `pseudo_SSY_a = var(y) * (T_a' T_a)`.
- The factor that varies per component (`q_a^2`) is replaced by a constant (`var(y)`).

Effect: `var(y)` factors out of both the numerator and denominator of the VIP ratio, leaving:

```
main_VIP_j = sqrt( p * sum_a [W_{j,a}^2 * (T_a' T_a)] / sum_a [(T_a' T_a)] )
```

This is a "weights weighted by X-score energy" formula, NOT a "weights weighted by Y-explained variance" formula. **It loses the connection to Y entirely.** Two PLS components with the same `||T_a||^2` but very different `q_a^2` get equal weight — even if one component contributes 100x more to the Y prediction than the other.

This is **wrong relative to the canonical formula**, not equivalent-but-rearranged.

### Distribution-of-error context
- Main's formula does still satisfy the `mean(VIP^2) = 1` invariant, because that invariant is enforced by `||W_a|| = 1`, independent of how `SSY_a` is defined. So sanity checks like "VIP > 1 means important" still produce a coherent ranking — it's just ranking by a *different* quantity than VIP is supposed to be.
- Main's formula will agree with canonical when `q_a` happens to be roughly equal across all components (rare in real PLS — Y-loadings typically decay as components capture less Y variance).
- Main's formula will diverge sharply from canonical when X has high-variance latent structure that does NOT predict Y (interferents, scattering, baseline drift) — see Section 5.

---

## 4. What the fix changes

```python
ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)            # branch, models.py:1769
```

Mapping to canonical:
- Canonical wants `SSY_a = q_a^2 * (T_a' T_a)`.
- Branch computes `SSY_a = q_a^2 * (T_a' T_a)`.
- **Identical.**

The branch also adds `||W_a||^2` normalization:

```python
col_norm_sq = np.sum(W ** 2, axis=0)
col_norm_sq = np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)
w_norm_sq = (W ** 2) / col_norm_sq
```

For sklearn's `PLSRegression`, `||W_a||^2 = 1` for all `a` (verified empirically — see Section 5; sklearn weights are unit-norm). So this division is a no-op in practice, but it makes the formula numerically robust if `compute_vip` is ever called with a non-sklearn PLS object whose weights are not unit-norm. Defensive but correct.

Verdict on the formula: **the branch implements the canonical Wold 2001 formula exactly, with a defensive (and harmless under sklearn) `||W_a||` normalization.**

---

## 5. Empirical demonstration

I ran both formulas on three datasets to show the magnitude of the difference.

### 5a. The fixture used by the tests (synthetic, n=60, p=20, 2 latent factors)

Setup:
- Factor 1: blocks features 2-5, drives Y with coefficient 5.0
- Factor 2: blocks features 12-15, drives Y with coefficient 0.5 (10x weaker)
- Both factors have comparable X-energy

Output:
```
sklearn x_weights_ shape: (20, 2)
sklearn y_loadings_ values: [[2.4584, 0.2786]]
||W_a||^2 per component: [1.0, 1.0]    <-- sklearn confirms unit-norm

OLD VIP top-4 features: [12, 14, 15, 13]  (factor 2 — the WEAK Y signal)
NEW VIP top-4 features: [4, 3, 5, 2]      (factor 1 — the STRONG Y signal)

Per-component SSY:
  canonical (q^2 * ||T_a||^2): [1177.26,   18.60]   ratio 63x
  OLD (var(y) * ||T_a||^2):    [3885.26, 4779.53]   ratio 0.81x  <-- OLD says factor 2 has MORE Y-explained variance than factor 1, the opposite of truth

max |OLD - NEW| = 1.39
```

The OLD formula ranks factor 2 (weak Y predictor) above factor 1 (strong Y predictor). The NEW formula ranks factor 1 above factor 2. This is exactly the bug Mehmood 2012 / Wold 2001 warn about: ignoring `q_a` makes VIP confuse "high X-energy" with "high Y-explained-variance."

### 5b. Realistic FTIR-style spectra (n=30, p=200, 3 Gaussian bands)

Setup:
- Three smooth Gaussian "bands" at different wavenumbers
- Y dominated by band 1 (coefficient 4.0)
- Bands 2 and 3 contribute weakly to Y but have comparable X-energy

Output (n_components=3):
```
y-loadings: [1.09, 0.22, 0.15]
max |OLD - NEW| = 1.57
OLD top-5 features: ALL band 1
NEW top-5 features: ALL band 1
```

When one band cleanly dominates Y, **both formulas agree on top features**, even though their absolute VIP values differ by ~1.5. This is reassuring: in the easy case where the user's spectroscopy gives a clean dominant Y-correlated band, the bug does not flip the top-N selection. The user-impact pattern is "bug bites only when there's structured non-Y X-variation."

### 5c. Realistic NIR-style with scattering interferent (n=50, p=100)

Setup:
- Signal band: Gaussian centered at index 24, X-score variance ~ 1.0, drives Y with coefficient 5.0
- Noise band: Gaussian centered at index 71, X-score variance ~ 9.0 (3x larger), does NOT predict Y
- Y depends only on signal score

This pattern is realistic for NIR/FTIR: scattering effects, water bands, or interferents create high-variance X-features that vary sample-to-sample but don't track the target chemistry.

Output (n_components=3):
```
y-loadings: [1.42, 0.27, -2.68]
max |OLD - NEW| = 1.70

OLD top-5 features: [71, 70, 72, 69, 73]   bands: NOISE, NOISE, NOISE, NOISE, NOISE  <-- WRONG
NEW top-5 features: [24, 23, 25, 22, 26]   bands: SIGNAL, SIGNAL, SIGNAL, SIGNAL, SIGNAL  <-- CORRECT

OLD VIP at signal max (24): 1.43       NEW VIP at signal max (24): 3.13
OLD VIP at noise max (71): 3.15        NEW VIP at noise max (71): 1.47
```

**This is the case where the bug demonstrably picks the wrong wavelengths.** The old formula's variable-importance ranking flips the signal and noise bands. If a user runs `varsel_method='importance'` on a spectrum with scattering or strong interferent, main's `compute_vip` selects the interferent wavelengths and drops the chemistry wavelengths.

This is exactly the kind of pattern where the user's domain (bone FTIR with collagen/apatite/lipid bands and varying sample matrix effects, NIR with water/scattering) is most likely to be affected.

### 5d. Edge-case behavior

- `n_components=1`: both formulas reduce to identical formulas (only one component, both formulas have a single term over the same denominator). Confirmed by the edge-case test passing on both branches.
- `y_loadings_ = 0`: NEW returns all zeros (deliberate design choice in branch); OLD returns nonzero values (`var(y)` is still > 0 even when the model can't predict Y). The branch's behavior is more defensible — if the PLS model has no Y-explanatory power per component, VIP shouldn't claim any variable is important.

---

## 6. Commercial / open-source software sanity check

The chemometrics master rule asks: do leading programs and well-known open-source implementations use the canonical (`q_a^2 * T_a' T_a`) formula or main's (`var(y) * T_a' T_a`) formula? Strong consensus on canonical:

| Source                                                  | Formula                                                | Evidence                                                                |
|---------------------------------------------------------|--------------------------------------------------------|-------------------------------------------------------------------------|
| Eigenvector **PLS_Toolbox** (`vip` function)            | Cites Chong & Jun 2005 (canonical)                     | Eigenvector docs Vip page; references Chong & Jun 2005                  |
| **SIMCA** (Sartorius/Umetrics)                          | Wold 2001-style; "weights = % Y-variance per component"| User Guide v15; OPLS-VIP paper (Galindo-Prieto et al. 2014)             |
| R **`plsVarSel`** (Mehmood/Liland's own package)        | Canonical: `c(Yloadings)^2 * colSums(scores^2)`        | filters.R source: `SS <- c(object$Yloadings)^2 * colSums(object$scores^2)` |
| R **`simplerspec`** (Philipp Baumann, NIR domain)       | Canonical: `Yloadings^2 * colSums(scores^2)`           | pls-vip.R: `SS <- c(object$Yloadings)^2 * colSums(object$scores^2)`     |
| R **`chillR`** (mostly copied from mevik/pls)           | Canonical (cites Chong & Jun 2005)                     | chillR docs; mevik/pls reference impl                                   |
| R **`mevik/pls`** `VIP.R`                               | Canonical: `Q2 * diag(crossprod(TT))`                  | source: `Q2TT <- Q2[1:opt.comp] * diag(crossprod(TT))[1:opt.comp]`      |
| R **`rchemo`**                                          | Canonical (cites Mehmood 2012)                         | rchemo docs reference Mehmood et al.                                    |
| R **`mdatools`** (Sergey Kucheryavskiy)                 | Canonical (cites Chong & Jun 2005)                     | mdatools docs reference                                                 |
| R **`mixOmics`**                                        | Canonical (Tenenhaus 1998)                             | mixOmics docs                                                           |
| **scikit-learn issue #7050** reference Python impl      | Canonical: uses `model.y_loadings_`                    | `s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)` where `q = y_loadings_` |
| **scikit-learn PR #13492** (proposed `vip_` attribute)  | Canonical                                              | `self.vip_ = np.sqrt(w0 * np.sum(s * w_norm.T ** 2, axis=1) / s_sum)`   |

I did not find a single open-source or commercial implementation that uses main's `var(y) * (T_a' T_a)` formula. **The chemometrics field is unanimous: VIP uses per-component y-loadings, not total y-variance.**

dasp's contaminant_analysis.py:_compute_vip already uses the canonical formula in main (lines 971-1009: `q_val**2 * (T[:, i].T @ T[:, i])`). So even within dasp, main is internally inconsistent — the contaminant-analysis VIP is canonical, the global compute_vip is not.

---

## 7. Distribution-model check (where is VIP consumed?)

VIP is **not** purely a display metric. It directly drives variable selection and feature ranking through `get_feature_importances()`. Call sites in main:

| File                                          | Line  | Effect                                                              |
|-----------------------------------------------|-------|---------------------------------------------------------------------|
| `models.py:1786`                              | 1786  | `get_feature_importances` for PLS / PLS-DA returns `compute_vip(...)` — central feature-importance path |
| `bayesian_utils.py:432`                       | 432   | Bayesian search varsel branch: `if varsel_method == 'importance': importances = get_feature_importances(...)` — drops bottom-N variables, refits |
| `search.py:2477`                              | 2477  | Grid search (run_search) varsel branch: same pattern as bayesian    |
| `search.py:4823`                              | 4823  | Final-model importance computation (display + report)               |
| `unified_bayesian.py:645`                     | 645   | Unified bayesian objective: `if method == 'importance': return get_feature_importances(...)` |
| `nsga2_search.py:3316`                        | 3316  | NSGA-II: `importances = get_feature_importances(model, model_type)` for top-N wavelength selection |
| `preprocessing_discovery.py:228`              | 228   | Smart preprocessing: `compute_vip(pls, X, y)` for PLS-driven preprocessing recommendation |

**Implications:**
1. VIP feeds the `varsel_method='importance'` path for PLS / PLS-DA. Wrong VIP → wrong variables retained → wrong CV scores → potentially wrong selected model + wrong selected wavelengths.
2. NSGA-II uses VIP for its top-N wavelength sweep on PLS-included objectives.
3. Preprocessing discovery (the "smart preprocessing" feature) uses VIP-derived feature importance to rank preprocessing methods. Wrong VIP could push toward suboptimal preprocessing.
4. The variable-importance plot in the GUI (final-model report) shows VIP-based importance for PLS / PLS-DA models. Users looking at this plot to interpret which wavelengths drive a PLS model are seeing wrong rankings on data with structured X-noise (Section 5c).

The fix is backend-only and reachable by every GUI user who builds a PLS / PLS-DA model. There is no opt-in, no hidden parameter, no power-user-only knob. **This satisfies the bundled-app distribution requirement: every GUI user gets the corrected behavior automatically.**

---

## 8. Test results

### 8a. Branch (.worktrees/T05-vip-formula-fix/)

```
$ pytest tests/test_vip_formula.py -v

tests/test_vip_formula.py::TestVIPCanonicalFormula::test_old_and_new_formulas_disagree_on_this_problem PASSED
tests/test_vip_formula.py::TestVIPCanonicalFormula::test_compute_vip_matches_canonical_reference PASSED
tests/test_vip_formula.py::TestVIPCanonicalFormula::test_compute_vip_does_not_match_old_buggy_formula PASSED
tests/test_vip_formula.py::TestVIPCanonicalFormula::test_canonical_invariant_average_squared_vip_is_one PASSED
tests/test_vip_formula.py::TestVIPCanonicalFormula::test_output_shape_and_nonnegativity PASSED
tests/test_vip_formula.py::TestVIPEdgeCases::test_compute_vip_n_components_1 PASSED
tests/test_vip_formula.py::TestVIPEdgeCases::test_compute_vip_zero_y_loadings_returns_zeros PASSED

============================== 7 passed in 1.02s ==============================
```

All 7 tests pass on the branch.

### 8b. Main (running the same test file against main's source via PYTHONPATH)

```
FAILED ::TestVIPCanonicalFormula::test_compute_vip_matches_canonical_reference
   AssertionError: Not equal to tolerance rtol=1e-10
   Mismatched elements: 20 / 20 (100%)
   Max absolute difference among violations: 1.38694975

FAILED ::TestVIPCanonicalFormula::test_compute_vip_does_not_match_old_buggy_formula
   AssertionError: compute_vip() still matches the old (buggy) formula — fix not applied.
   assert 0.0 > 0.001

FAILED ::TestVIPEdgeCases::test_compute_vip_zero_y_loadings_returns_zeros
   AssertionError: assert np.False_ (got nonzero values for zero y_loadings)

========================= 3 failed, 4 passed in 2.26s =========================
```

3 failed on main, 7 pass on the branch — exactly the predicted state from the plan ("3 failed, 4 passed pre-fix"). The 4 that pass on main are: the disagreement-fixture sanity check (independent of `compute_vip`), the average-squared-VIP invariant (satisfied by both formulas), the shape/nonnegativity check, and the n_components=1 case (where both formulas collapse).

The bug is real and reproducible on main; the test file successfully discriminates it.

---

## 9. Codex review reference (pre-master-rule)

Two pre-master-rule Codex review files exist (`codex_reviews/T05_vip_formula.txt` v1 and `T05_vip_formula_v2.txt`). The v2 review's final verdict was **APPROVE_WITH_CHANGES** after three blockers were resolved in the plan:

1. **Resolved.** Documentation said sklearn's univariate `y_loadings_` should be indexed `[:, 0]`; corrected to `[0, :]` (sklearn shape is `(n_targets, n_components)` so row 0 is the per-component vector). Code in branch correctly uses `Q[0, :]`.
2. **Resolved.** Test plan originally claimed the average-squared-VIP invariant test would catch the bug; Codex correctly observed both old and new satisfy this when `||W_a|| = 1`. Plan and test docstring updated to label the test `[SANITY]` rather than `[DISCRIMINATING]`.
3. **Resolved (deferred).** Codex noted the same buggy formula exists in `templates/variable_selection.py:54` and (per Codex) `nsga2_search.py:627`. Plan explicitly defers these to follow-up T-05a; branch does NOT touch them.

Codex's substantive cross-check on the formula itself (v2 file, line ~4322): "*The formula itself, once the indexing text is corrected, matches the canonical Wold/Mehmood form: `SSY_a = q_a^2 * t_a' t_a`, normalized squared weights by component, divided by total explained Y SS. External cross-check: rchemo documents VIP as based on Y-variance explained per component following Mehmood et al. 2012*."

Codex independently ran the test fixture and reported `max_abs_diff = 1.3869` between old and new — matches my measurement (`1.39`).

Two non-blocking notes from Codex v2 worth flagging for the main agent:
- Minor wording inconsistency in the plan around edge-case test pre-fix expected behavior (plan says zero-`y_loadings_` test passes pre-fix in one place and fails pre-fix in another). I confirmed by running: pre-fix it FAILS (3 failed total, including this one). Plan wording quibble only, no code impact.
- Plan claim that the all-zeros guard "matches existing pattern in compute_vip" — the existing main `compute_vip` does NOT have this guard. Codex notes the comment is misleading but not blocking. Branch code is correct; only the comment needs minor edit.

---

## 10. Notes / uncertainties for the main agent

Items the verdict-writer should double-check before drafting the verdict:

1. **`nsga2_search.py:627` is real and buggy.** Confirmed inline buggy VIP block at lines ~620-636 (`ssy_comp = np.sum(T**2, axis=0) * np.var(y_arr, axis=0)`). T-05 does not touch it; it remains buggy after T-05 merges. Whether to fold this into T-05 or push to T-05a is a scope-judgment call for the verdict-writer. The plan explicitly defers it; Codex flagged it; it is real.

2. **`templates/variable_selection.py` is a real out-of-scope site.** This is the script-export template that ships generated standalone Python scripts to the user. If a user exports a script from dasp and runs it outside the GUI, they will get wrong VIP rankings. This is a real, user-facing miss for the export-to-script workflow. Whether to push T-05a as a follow-up or rope it into T-05's merge is a judgment call.

3. **Real-world impact magnitude is data-dependent.** Section 5b shows that for clean dominant-band data, the bug doesn't flip top-N selection. Section 5c shows it does flip top-N when X has high-variance non-Y-explanatory structure (interferents, scattering, baseline effects). For the user's domain (bone FTIR with mineral/collagen/lipid bands and varying sample matrix; NIR with water bands and scattering), Section 5c is the more representative pattern, suggesting the bug has been actively wrong on real user data.

4. **Internal inconsistency in main is significant.** dasp main has two PLS VIP implementations: `models.py:compute_vip` (buggy) and `contaminant_analysis.py:_compute_vip` (canonical). The contaminant-analysis path uses `Q[i, 0]**2 * (T[:, i].T @ T[:, i])` already. So one half of dasp's VIP is correct and the other half is wrong — the fix brings them into agreement.

5. **No commercial-software false-alarm risk.** Unlike T-26 (where main matched PLS_Toolbox default behavior), every chemometrics package I could verify uses the canonical formula. There is no chemometrics tradition of "global y-variance VIP." The bug appears to have originated as a coding shortcut, not as alignment with any field convention.

6. **Defensive `||W_a||` normalization.** The branch adds `(W^2) / col_norm_sq` defensively. For sklearn's `PLSRegression` this is a no-op (verified: `||W_a||^2 = [1.0, 1.0]` in my tests). Left in, this protects future use with non-sklearn PLS objects (e.g., custom PLS variants). Doesn't change current behavior; cheap insurance.

7. **Unused `X` and `y` parameters.** The fixed `compute_vip` no longer uses its `X` or `y` arguments — VIP is fully determined by the fitted PLS object's `x_weights_`, `x_scores_`, `y_loadings_`. The branch keeps the signature for API compatibility (callers in `search.py`, `bayesian_utils.py`, etc., still pass `X, y`). This is the right call to avoid breaking call sites; the branch documents the args as "kept for API compatibility."

---

## Sources

- Wold, S.; Sjöström, M.; Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems* 58:109-130. https://doi.org/10.1016/S0169-7439(01)00155-1
- Chong, I.-G.; Jun, C.-H. (2005). Performance of some variable selection methods when multicollinearity is present. *Chemometrics and Intelligent Laboratory Systems* 78:103-112.
- Mehmood, T.; Liland, K. H.; Snipen, L.; Sæbø, S. (2012). A review of variable selection methods in Partial Least Squares Regression. *Chemometrics and Intelligent Laboratory Systems* 118:62-69.
- Galindo-Prieto, B.; Eriksson, L.; Trygg, J. (2014). Variable influence on projection (VIP) for orthogonal projections to latent structures (OPLS). *Journal of Chemometrics* 28:623-632.
- [Eigenvector PLS_Toolbox `vip` function documentation](https://www.eigenvectordocs.com/index.php?title=Vip)
- [SIMCA 15 User Guide — Sartorius/Umetrics](https://www.sartorius.com/download/544940/simca-15-user-guide-en-b-00076-sartorius-data.pdf)
- [scikit-learn issue #7050 — VIP score calculation reference impl](https://github.com/scikit-learn/scikit-learn/issues/7050)
- [scikit-learn PR #13492 — proposed `vip_` attribute](https://github.com/scikit-learn/scikit-learn/pull/13492)
- [plsVarSel R package source (Mehmood/Liland)](https://github.com/cran/plsVarSel/blob/master/R/filters.R)
- [simplerspec R package pls-vip.R (Philipp Baumann, NIR)](https://github.com/philipp-baumann/simplerspec/blob/master/R/pls-vip.R)
- [chillR R package VIP function docs](https://search.r-project.org/CRAN/refmans/chillR/html/VIP.html)
- [rchemo R package vip function docs](https://search.r-project.org/CRAN/refmans/rchemo/html/vip.html)
- [mdatools R package vipscores function docs](https://rdrr.io/cran/mdatools/man/vipscores.html)
- [mevik/pls reference VIP.R](http://mevik.net/work/software/pls.html)
