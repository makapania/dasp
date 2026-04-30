# T-05 validation: VIP formula correctness

**Branch:** `fix/T05-vip-formula-fix` (HEAD `3269475`)
**Status:** APPROVED for merge (after rebase + strict-test pass)
**Author:** Opus 4.7, 2026-04-30
**Verdict:** REAL BUG. Real user impact in the user's domain. Field consensus is unambiguous. Fix
is bit-identical to canonical Wold 2001 formula. Strongest-evidence verdict in the validation
gate so far.

Investigation findings: [T05_findings.md](T05_findings.md) (produced by parallel investigation agent).

---

## TL;DR

`models.py:compute_vip` in main computes `SSY_a = var(y) * (T_a' T_a)`. The canonical Wold 2001
VIP formula is `SSY_a = q_a^2 * (T_a' T_a)` where `q_a` is the per-component y-loading. The
factor that should vary per component (`q_a^2`) has been replaced by a constant (`var(y)`), so
main's VIP loses the connection to Y entirely — it ranks variables by X-energy weighted by
component, not by Y-explained variance.

In the user's domain (bone FTIR with collagen/apatite/lipid bands plus variable matrix /
scattering effects, or NIR with water bands and particle-size effects), this means main's VIP
can rank an interferent / scattering wavelength above the chemistry wavelength when X has
high-variance non-Y-explanatory structure. The investigation agent's Section 5c shows main
picking the noise band's top 5 wavelengths over the signal band's top 5. The fix corrects this
to the canonical Wold 2001 formula.

Field check: every commercial chemometrics tool (PLS_Toolbox, SIMCA) and every open-source
chemometrics package the agent could verify (R: plsVarSel, simplerspec, chillR, mevik/pls,
rchemo, mdatools, mixOmics; sklearn issue #7050 reference impl, sklearn PR #13492) uses the
canonical formula. **No tool uses main's `var(y)` formula.** Unlike T-26 where dasp main
happened to match PLS_Toolbox's default, here dasp is the outlier and the fix brings it into
line with universal practice.

Also notable: dasp main is internally inconsistent. `contaminant_analysis.py:_compute_vip`
already uses the canonical formula (`q_val**2 * (T[:, i].T @ T[:, i])` at lines 971-1009).
Only `models.py:compute_vip` is buggy. The fix brings the two paths into agreement.

## 1. Canonical reference (universal field consensus)

**Wold, S.; Sjöström, M.; Eriksson, L.** (2001). _PLS-regression: a basic tool of chemometrics._
Chemometrics and Intelligent Laboratory Systems 58:109–130.

For PLS dimension `a`:
- Per-component explained Y sum-of-squares: `SSY_a = q_a^2 * (T_a' T_a)`
- Total: `SSY_cum = Σ_a SSY_a`
- VIP for variable j: `VIP_j = sqrt( p · Σ_a [(W_{j,a} / ||W_a||)^2 · SSY_a] / SSY_cum )`

where `p` = number of features, `W` = X-weights, `T` = X-scores, `q_a` = y-loading for
component a.

Equivalent restatements:
- Mehmood, Liland, Snipen, Sæbø (2012), Eq. 1 — review article specifically about PLS
  variable selection.
- Chong & Jun (2005) — cited reference in PLS_Toolbox and `mevik/pls`. Algebraically
  identical.

Key invariant: `mean_j(VIP_j^2) = 1` (when `||W_a|| = 1`, true in sklearn `PLSRegression`).

## 2. Main's bug, exactly

`src/spectral_predict/models.py` (line ~1738 in main):

```python
ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)   # <-- bug
```

`np.var(y, axis=0)` is a single scalar (the global Y variance) multiplied identically into
every component's `T_a' T_a`. Per the canonical formula, this term must be per-component and
proportional to `q_a^2`. The factor that should distinguish components has been collapsed to a
constant.

Effect: `var(y)` factors out of both numerator and denominator of the VIP ratio, leaving
`main_VIP_j = sqrt( p · Σ_a [W_{j,a}^2 · (T_a' T_a)] / Σ_a (T_a' T_a) )`.

This is "weights weighted by X-score energy," NOT "weights weighted by Y-explained variance."
**It loses the connection to Y entirely.**

Two PLS components with the same `||T_a||^2` but very different `q_a^2` get equal weight in
main, even if one component contributes 100x more to the Y prediction than the other. That is
exactly the bug Mehmood 2012 / Wold 2001 warn the field about.

## 3. The fix, exactly

Branch `src/spectral_predict/models.py` (~line 1769):

```python
W = np.asarray(pls_model.x_weights_)
T = np.asarray(pls_model.x_scores_)
Q = np.asarray(pls_model.y_loadings_)         # sklearn shape: (n_targets, n_components)
q = Q if Q.ndim == 1 else Q[0, :]             # univariate Y: row 0

ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)  # canonical SSY_a = q_a^2 · (T_a' T_a)
ssy_total = float(np.sum(ssy_comp))

if ssy_total <= 0.0:
    return np.zeros(n_features, dtype=float)

col_norm_sq = np.sum(W ** 2, axis=0)
col_norm_sq = np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)
w_norm_sq = (W ** 2) / col_norm_sq            # ||W_a||^2 normalization
                                              # (no-op for sklearn PLSRegression
                                              #  since weights are unit-norm)
vip_scores = np.sqrt(n_features * (w_norm_sq @ ssy_comp) / ssy_total)
```

Maps exactly to canonical. The `||W_a||^2` normalization is defensive (a no-op for sklearn but
correct for non-sklearn PLS objects). The `ssy_total <= 0` guard returns zeros for the
degenerate case of a PLS model with no Y-explanatory power.

## 4. Empirical demonstration of user-visible impact

The agent ran three fixtures (full detail in `T05_findings.md` Section 5):

**5a. Synthetic 2-factor, n=60, p=20, factor 1 strong-Y, factor 2 weak-Y, comparable X-energy:**

```
OLD VIP top-4 features: [12, 14, 15, 13]   <-- ranks the WEAK-Y factor's wavelengths
NEW VIP top-4 features: [4, 3, 5, 2]       <-- ranks the STRONG-Y factor's wavelengths
max |OLD - NEW| = 1.39
```

**5b. Realistic FTIR-style, n=30, p=200, dominant Gaussian band:**

When one band cleanly dominates Y, both formulas agree on top features (the bug doesn't flip
top-N). Reassuring: the bug is data-dependent, not always-wrong.

**5c. Realistic NIR-style with high-variance non-Y interferent, n=50, p=100:**

```
Signal band index 24 (Y-coef = 5.0, X-score var ~ 1.0)
Noise band  index 71 (Y-coef = 0.0, X-score var ~ 9.0)

OLD VIP top-5: [71, 70, 72, 69, 73]   <-- all NOISE band, WRONG
NEW VIP top-5: [24, 23, 25, 22, 26]   <-- all SIGNAL band, CORRECT
OLD VIP at signal max: 1.43           NEW: 3.13
OLD VIP at noise max:  3.15           NEW: 1.47
```

This is the user's domain pattern: NIR / FTIR with structured X-variation that doesn't track
the chemistry. Main's formula confidently selects the wrong wavelengths.

## 5. Where VIP feeds back into dasp's behavior

VIP is **not** purely a display metric. It directly drives selection logic:

| File                              | Line | What VIP affects                                              |
|-----------------------------------|------|--------------------------------------------------------------|
| `models.py`                       | 1786 | `get_feature_importances` — central PLS / PLS-DA importance  |
| `bayesian_utils.py`               | 432  | Bayesian search `varsel_method='importance'` — drops bottom-N|
| `search.py`                       | 2477 | Grid search `varsel_method='importance'` — same pattern      |
| `search.py`                       | 4823 | Final-model importance computation (display + report)        |
| `unified_bayesian.py`             | 645  | Unified bayesian objective `method='importance'`             |
| `nsga2_search.py`                 | 3316 | NSGA-II top-N wavelength sweep on PLS-included objectives    |
| `preprocessing_discovery.py`      | 228  | "Smart preprocessing" PLS-driven recommendation              |

So a wrong VIP propagates into:
1. Wrong variables retained when `varsel_method='importance'`
2. Wrong variable-importance plot for the user interpreting which wavelengths drive their PLS model
3. Wrong PLS-driven preprocessing recommendation
4. Wrong NSGA-II wavelength selection on PLS objectives

Backend-only fix; every GUI user automatically benefits. **Distribution-model check passes
unambiguously** — no opt-in, no hidden parameter, no power-user-only knob.

## 6. Field-alignment check (chemometrics master rule)

Universal consensus on canonical formula across:

| Source                                                  | Formula                                                  |
|---------------------------------------------------------|----------------------------------------------------------|
| Eigenvector **PLS_Toolbox** `vip`                       | Cites Chong & Jun 2005 (canonical)                       |
| **SIMCA** (Sartorius/Umetrics)                          | Wold 2001-style, Y-variance-per-component                |
| R **`plsVarSel`** (Mehmood/Liland's own package)        | `c(Yloadings)^2 * colSums(scores^2)`                     |
| R **`simplerspec`** (Baumann, NIR domain)               | `Yloadings^2 * colSums(scores^2)`                        |
| R **`chillR`**                                          | Canonical (cites Chong & Jun 2005)                       |
| R **`mevik/pls`** `VIP.R`                               | `Q2 * diag(crossprod(TT))`                               |
| R **`rchemo`**                                          | Canonical (cites Mehmood 2012)                           |
| R **`mdatools`** (Kucheryavskiy)                        | Canonical (cites Chong & Jun 2005)                       |
| R **`mixOmics`**                                        | Canonical (Tenenhaus 1998)                               |
| **sklearn issue #7050** reference impl                  | `q = y_loadings_`, canonical                             |
| **sklearn PR #13492**                                   | `s = q^2 * ||T_a||^2`, canonical                         |

The investigation found **zero** tools using main's `var(y)` formulation. Unlike T-26 (where
dasp main matched PLS_Toolbox's default behavior), T-05 main is the outlier. Fix brings dasp
into universal compliance.

## 7. Internal consistency check

dasp main has TWO PLS VIP implementations:
- `src/spectral_predict/models.py:compute_vip` — **buggy** (`var(y)` formulation)
- `src/spectral_predict/contaminant_analysis.py:_compute_vip` — **canonical** (`q_val**2` at
  lines 971-1009)

So one half of dasp's VIP is correct, the other half is wrong. T-05 brings them into
agreement.

## 8. Out-of-scope sites (deferred to T-05a)

The plan explicitly defers these. The investigation confirmed both are real:

| Site                                              | Status                                                  |
|---------------------------------------------------|---------------------------------------------------------|
| `templates/variable_selection.py:54`              | Same `var(y) * (T'T)` bug. Affects exported standalone  |
|                                                   | scripts that users run outside the GUI.                 |
| `nsga2_search.py:~627` (inline VIP block 620-636) | Same bug. Affects NSGA-II's inline PLS-importance path. |
|                                                   | NSGA-II's other VIP path at line 3316 goes through      |
|                                                   | `models.py:compute_vip` and IS fixed by T-05.           |

These are real out-of-scope sites the user should be aware of when scoping T-05a.
**Recommendation:** merge T-05 now (fixes the dominant user-facing path), file T-05a as a
follow-up to address the two remaining sites. Don't block T-05 on T-05a — the central path
is the user impact.

## 9. Test results

**Branch (`pytest tests/test_vip_formula.py -v`):**

```
7 passed in 1.02s
```

All 7 tests pass on the branch, including:
- `test_compute_vip_matches_canonical_reference` (numerical check vs inline canonical reference)
- `test_compute_vip_does_not_match_old_buggy_formula` (catches no-op fix)
- `test_compute_vip_zero_y_loadings_returns_zeros` (degenerate-case behavior)

**Main (same tests against main's source):**

```
3 failed, 4 passed in 2.26s
```

3 tests fail on main, exactly as the plan predicted:
1. `test_compute_vip_matches_canonical_reference` — max abs diff = 1.387 vs canonical
2. `test_compute_vip_does_not_match_old_buggy_formula` — exact match to old formula (assert 0.0 > 0.001)
3. `test_compute_vip_zero_y_loadings_returns_zeros` — main returns nonzero when q=0

The 4 that pass on main are: the disagreement-fixture sanity check (independent of
`compute_vip`), the average-squared-VIP invariant (satisfied by both formulas), the
shape/nonnegativity check, and the n_components=1 case (where both formulas collapse to the
same thing).

**The bug is real, reproducible on main, and the test file successfully discriminates it.**

## 10. Verdict

**APPROVE for merge** (after rebase + strict-test pass following T-10's pattern).

Reasoning:

1. **Universally agreed canonical formula.** Wold 2001 + Mehmood 2012 + Chong & Jun 2005 +
   Tenenhaus 1998 + every commercial tool + every open-source chemometrics package.
2. **Main is the outlier.** `var(y)` formulation appears in zero tools the agent could find.
3. **User-visible bug in the user's domain.** Section 5c shows main picking interferent
   wavelengths over signal wavelengths in NIR/FTIR-realistic patterns. The user's bone FTIR
   work has matrix/scattering effects exactly fitting this pattern.
4. **Backend-only, every GUI user benefits.** No knob, no opt-in, distribution-model check
   passes (unlike T-26).
5. **Internal consistency improvement.** dasp's contaminant_analysis already uses canonical;
   fix brings models.py into agreement.
6. **Tests discriminate the bug correctly.** 3 fail on main, 7 pass on branch.
7. **Defensive `||W_a||` normalization.** No-op for sklearn, robust for future non-sklearn use.
8. **Out-of-scope sites are real but acknowledged.** T-05a deferred; doesn't block T-05's
   central-path fix.

This is the strongest-evidence verdict in the validation gate so far. There is no
field-convention reason to keep main's behavior, no defensive-implementation argument for
preserving the bug, no distribution-model concern. The fix is correct.

## 11. Pre-merge checklist (same pattern as T-10)

- [ ] User reads + approves this validation note
- [ ] Rebase `fix/T05-vip-formula-fix` onto current main (resolves the docs-deletion noise
      from the pre-reframing branch base)
- [ ] Re-run `pytest tests/test_vip_formula.py -v` post-rebase (expect 7 passed)
- [ ] Run regression sweep on T-05-relevant tests:
      `tests/test_search_comprehensive.py tests/test_bayesian_utils.py
       tests/test_unified_bayesian_baseline.py tests/test_nsga2_search.py
       tests/test_plsda_importance.py tests/test_ga_pls.py
       tests/test_end_to_end_workflow.py tests/test_end_to_end_workflows.py
       tests/smoke/test_imports.py tests/test_model_io.py`
- [ ] If all green, fast-forward merge into main
- [ ] Push
- [ ] Update `docs/bugfix_validation/README.md` status table to MERGED
- [ ] File T-05a follow-up ticket for `templates/variable_selection.py:54` and
      `nsga2_search.py:~627`
