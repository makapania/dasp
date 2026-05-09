# T-16 Phase 1 — CV-ANOVA p-value column for PLS regression rows

**Branch:** `feat/T16-phase1-cv-anova` (off `main` at `98e6e98`)
**Status:** plan v2 (Codex review folded in 2026-05-08), not started
**Effort:** ~half day source + tests
**Codex review:** 1 BLOCKER + 3 MEDIUM + 1 LOW — all addressed below; v1 of this file at `ae00868` is the pre-review state.

---

## 1. Goal

Add a `cv_anova_pvalue` column to the result CSV, populated for PLS regression rows only, that answers Q1 ("is this model real or could it have arisen by chance against a mean-prediction null?") using the canonical CV-ANOVA F-test from Eriksson et al. 2008.

Out of scope for Phase 1:
- PLS-DA (classification) — math extends but Y dummy-coding adds complexity; defer to Phase 1b if user wants
- Non-PLS rows (LightGBM, RF, SVM, MLP) — CV-ANOVA is mathematically defined for PLS/OPLS only; non-PLS rows get blank `cv_anova_pvalue`
- F-test LV-selection rule (the OPUS variant) — adds a kwarg to `run_search`, dropped to keep the search loop pristine
- Phase 2 permutation test — separate plan, separate branch

---

## 2. Math + reference

**Reference:** Eriksson, L., Trygg, J., & Wold, S. (2008). *CV-ANOVA for significance testing of PLS and OPLS® models*. Journal of Chemometrics, 22(11-12), 594-600.

**Formula** (regression, single Y):

```
N           = number of training samples (post-NaN-drop)
A           = number of PLS latent variables (n_components from params)
PRESS       = sum of squared CV residuals = N * RMSEcv**2
SSY         = total sum of squares of y = sum((y - mean(y))**2)
F           = ((SSY - PRESS) / A) / (PRESS / (N - A - 1))
df1         = A
df2         = N - A - 1
p_value     = scipy.stats.f.sf(F, df1, df2)   # use sf, not 1 - cdf
```

**Codex MEDIUM-1 fold-in:** use `scipy.stats.f.sf(F, df1, df2)` rather than `1 - scipy.stats.f.cdf(F, df1, df2)`. `sf` is the survival function — numerically stable in the extreme tail; `1 - cdf` rounds tiny but non-zero p-values to zero.

**PRESS = N × RMSEcv² is correct in dasp's grid + Bayesian regression paths** because both use pooled cross-validated predictions (`cross_val_predict_pooled` at `unified_bayesian.py:1694` / equivalent at `search.py:4792`) and compute RMSE as `sqrt(mean_squared_error(y, y_pred_cv))` over the concatenated holdout vector. So `N × RMSEcv² = sum((y - y_pred_cv)²) = PRESS` exactly.

**Codex MEDIUM-3 fold-in (repeated-CV semantics):** dasp's repeated K-fold reduces repeated predictions to a single per-sample average before RMSEcv is computed (`cv_utils.py:363-373, 465-477`). So `PRESS = N × RMSEcv²` is the PRESS of the averaged-prediction vector, not the sum of all repeated-fold residuals. This is a sensible (if non-standard) interpretation — "is the averaged-CV prediction better than mean prediction?" — and the plan **accepts and documents this** rather than blanking out repeated-CV rows. The docstring of `compute_cv_anova_pvalue()` will say so explicitly.

**Edge cases (Codex MEDIUM-2 fold-in — expanded):**

| Input condition | Action |
|---|---|
| `n_components < 1` | return `np.nan` (no LVs is degenerate) |
| `N - A - 1 <= 0` | return `np.nan` (over-parametrised; df2 invalid) |
| `N < 2` | return `np.nan` (no variance to test) |
| `y_true` non-1D OR multi-output | return `np.nan` (Phase 1 is single-Y only) |
| `y_true` contains non-finite values | return `np.nan` (defensive — upstream should have filtered) |
| `rmsecv` non-finite OR `<= 0` | return `np.nan` |
| `SSY <= 0` (zero-variance y) | return `np.nan` (null comparison undefined) |
| `PRESS >= SSY` (model worse than mean) | clip F to 0 → p_value = 1.0 (not significant) |

---

## 3. Files to touch

### 3.1 New helper in `src/spectral_predict/scoring.py`

Add a pure function `compute_cv_anova_pvalue()` near the existing scoring helpers. Pure scipy + numpy, no dasp-internal dependencies.

```python
def compute_cv_anova_pvalue(
    y_true: np.ndarray,
    rmsecv: float,
    n_components: int,
) -> float:
    """CV-ANOVA F-test p-value (Eriksson, Trygg & Wold 2008).

    Returns p-value for the null hypothesis that the PLS regression
    model's cross-validated PRESS is no better than mean-prediction
    PRESS. Only defined for single-Y PLS regression with pooled
    cross-validated predictions; returns nan on degenerate inputs
    (see edge-case table in the Phase 1 plan doc).

    For repeated K-fold CV, dasp reduces repeated predictions to a
    single per-sample average before computing RMSEcv. The p-value
    here therefore tests the averaged-CV-prediction vector against
    mean prediction — a sensible extension of the original Eriksson
    formulation, but not what the 2008 paper literally describes.
    """
    # ... implementation per §2 math + edge-case table ...
```

LOC: ~35 source + ~12 docstring.

### 3.2 Insertion in `_run_single_config()` — grid path (`search.py:4488`)

After the regression-metrics block at `search.py:5256-5269` (where `result["RMSEcv"] = mean_rmse` lands). Reuse the already-normalized `lvs` extraction at `search.py:5129-5133`:

```python
# CV-ANOVA p-value for PLS regression rows (Eriksson 2008)
if model_name == "PLS" and task_type == "regression" and lvs is not None:
    result["cv_anova_pvalue"] = compute_cv_anova_pvalue(
        y_true=y, rmsecv=mean_rmse, n_components=lvs,
    )
else:
    result["cv_anova_pvalue"] = np.nan
```

**Codex LOW fold-in:** uses the existing `lvs` (already defined at `search.py:5133` from `params.get("n_components") or params.get("pls__n_components")`) instead of re-implementing the extraction.

LOC: ~6.

### 3.3 Insertion in unified Bayesian regression path (`unified_bayesian.py`)

**Codex BLOCKER fold-in.** The Bayesian/TPE path computes RMSEcv independently and converts trials to result rows without going through `_run_single_config`. Without this insertion, every Bayesian/TPE PLS-regression result CSV would silently lack the new column.

**Site A — objective function** at `unified_bayesian.py:1698-1699` (right after `rmse` is computed):

```python
rmse = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
r2 = r2_score(y, y_pred_cv)

# CV-ANOVA p-value — only for PLS regression
if model_name == "PLS":
    n_lv_for_anova = params.get("n_components") or params.get("pls__n_components")
    if n_lv_for_anova is not None:
        cv_anova_p = compute_cv_anova_pvalue(
            y_true=y, rmsecv=rmse, n_components=int(n_lv_for_anova)
        )
        trial.set_user_attr('cv_anova_pvalue', cv_anova_p)
```

LOC: ~7.

**Site B — convert_study_to_dataframe** at `unified_bayesian.py:3131-3137` (right after `RMSEcv` is read into the row dict):

```python
row['RMSEcv'] = trial.user_attrs.get('RMSEcv', trial.value)
row['R2cv'] = trial.user_attrs.get('R2cv', np.nan)
row['CCCcv'] = trial.user_attrs.get('CCCcv', np.nan)
row['cv_anova_pvalue'] = trial.user_attrs.get('cv_anova_pvalue', np.nan)
```

LOC: ~1.

### 3.4 Imports

- `src/spectral_predict/search.py`: import `compute_cv_anova_pvalue` from `.scoring`
- `src/spectral_predict/unified_bayesian.py`: import `compute_cv_anova_pvalue` from `.scoring`

### 3.5 No GUI change for Phase 1

The column lands in the CSV automatically. The Results tab leaderboard auto-displays non-internal columns (verified by Codex at `spectral_predict_gui_optimized.py:29507-29524`). A dedicated visual treatment (e.g., asterisk on `cv_anova_pvalue < 0.05`) is deferred to Phase 1d polish.

### 3.6 Result-CSV schema

`cv_anova_pvalue` joins the existing per-row scalar columns. `create_results_dataframe()` at `scoring.py:498-521` does not predeclare every column; rows added via `add_result()` accept the new key. Old result CSVs lacking this column continue to load (pandas yields `nan`); old code reading new CSVs ignores the column. Backwards-compatible.

---

## 4. Tests

New file `tests/test_cv_anova.py`. Approximately ~120 LOC, ~9 cases:

1. **High-signal case**: synthetic `y = X[:, 5] + 0.01*noise`; PLS with 2 LVs on N=50; expect `p < 0.001`.
2. **No-signal case**: synthetic `y = noise` (uncorrelated with X); expect `p > 0.5`.
3. **Degenerate** `PRESS >= SSY`: model worse than mean; expect `p == 1.0` exactly (clipped).
4. **Edge** `N - A - 1 <= 0`: over-parametrised; expect `np.nan`.
5. **Edge** `n_components < 1`: expect `np.nan`.
6. **Edge** zero-variance y (`SSY = 0`): expect `np.nan` (Codex MEDIUM-2).
7. **Edge** non-finite RMSEcv / non-finite y: expect `np.nan` (Codex MEDIUM-2).
8. **Edge** multi-output y_true: expect `np.nan` (Codex MEDIUM-2).
9. **Reference value pin**: hand-computed F and p on a 10-sample fixture, asserted within `1e-6` of `scipy.stats.f.sf(F, A, N-A-1)`.

Plus two integration tests:
10. **End-to-end through `run_search`**: load BoneCollagen subset, run `run_search(models_to_test=['PLS'])`, assert `cv_anova_pvalue` column present in result DataFrame, all PLS rows non-nan and in `[0, 1]`.
11. **End-to-end through `run_unified_bayesian`** (Codex BLOCKER fold-in): same test but via Bayesian path (`run_unified_bayesian(models_to_test=['PLS'])`), assert column lands in the converted DataFrame, values are sensible.

Verification commands:
```
.venv312/Scripts/python.exe -m py_compile src/spectral_predict/scoring.py src/spectral_predict/search.py src/spectral_predict/unified_bayesian.py
.venv312/Scripts/python.exe -m pytest tests/test_cv_anova.py -v
```

---

## 5. Commit shape

Single commit, branch `feat/T16-phase1-cv-anova`:

```
feat(T-16): CV-ANOVA p-value column for PLS regression rows

Adds cv_anova_pvalue per Eriksson, Trygg & Wold 2008 to the result CSV
for PLS regression rows. Computed at row finalization in BOTH the
grid path (_run_single_config) and the unified Bayesian path
(via trial user_attr). Pure additive — no change to which rows get
computed, which row wins, or any search-loop behavior. Non-PLS rows
and PLS-DA rows get nan.

Numerical implementation uses scipy.stats.f.sf for tail stability.
Edge cases (zero-variance y, multi-output, non-finite, over-
parametrised, model-worse-than-mean) all return nan or clip to p=1.0.

Phase 1 of T-16 (model-comparison machinery survey at
docs/T16_MODEL_COMPARISON_SURVEY.md). Phase 2 (on-demand permutation
test) is a separate ticket.

Files:
- src/spectral_predict/scoring.py: new compute_cv_anova_pvalue() helper
- src/spectral_predict/search.py: insertion in _run_single_config
- src/spectral_predict/unified_bayesian.py: insertions in objective + dataframe converter
- tests/test_cv_anova.py: 9 unit + 2 integration tests
```

LOC estimate: ~50 source + ~120 tests = ~170 total (up from v1's ~115 due to BLOCKER + MEDIUM fold-ins).

---

## 6. Verification battery

Before commit:
- `py_compile` on all three modified source files — no syntax errors
- `pytest tests/test_cv_anova.py -v` — 11/11 pass
- Smoke test via real `run_search` AND `run_unified_bayesian` on BoneCollagen with `models_to_test=['PLS']` — confirm `cv_anova_pvalue` lands in both result paths with sensible values

After commit, before push:
- `git diff --stat main..feat/T16-phase1-cv-anova` — confirm only 4 files changed (3 source + 1 test), no whitespace drift
- Spot-check a fresh result CSV from a real GUI run — confirm column appears

---

## 7. Out of scope (defer to follow-up tickets if user wants)

- **Phase 1b — PLS-DA CV-ANOVA**: extend to classification by computing PRESS on continuous PLS predictions of dummy-coded class labels. Math is straightforward; complexity is in extracting per-fold continuous predictions vs the threshold-applied class labels currently surfaced. Estimate ~half day.
- **Phase 1c — F-test LV selection rule (OPUS variant)**: opt-in alternative to min-RMSEcv selection. Dropped from Phase 1 to keep `run_search` signature pristine. Estimate ~half day if revived.
- **Phase 1d — GUI asterisk / color coding** for `cv_anova_pvalue < 0.05`: cosmetic polish. Estimate ~couple hours.
- **Phase 2 — on-demand permutation test for arbitrary leaderboard rows**: separate plan, separate branch. Estimate ~2-3 days.

---

## 8. Codex review verdict + fold-in summary

| Severity | Finding | Resolution |
|---|---|---|
| **BLOCKER** | Bayesian PLS rows bypass `_run_single_config` → silent column drop on TPE/Bayesian CSVs | §3.3 added — insertion at `unified_bayesian.py` objective + dataframe converter; integration test 11 covers it |
| **MEDIUM-1** | Use `f.sf` not `1 - f.cdf` for tail stability | §2 formula updated |
| **MEDIUM-2** | Missing degenerate-input guards (zero-variance y, multi-output, non-finite) | §2 edge-case table expanded; tests 6/7/8 added |
| **MEDIUM-3** | Repeated-CV PRESS semantics differ from Eriksson 2008 | §2 documents the averaged-prediction interpretation; helper docstring will state it |
| **LOW** | Re-extracting `n_components` ignores the existing `lvs` normalization at `search.py:5133` | §3.2 reuses `lvs` |

All five findings folded in. No new open questions for the reviewer post-fold-in.

---

## 9. Open questions for the user (decisions before implementation)

1. **PLS-DA scope:** ship Phase 1 as PLS-regression-only (current plan) or bundle PLS-DA in the same commit? **Default:** PLS-regression first; PLS-DA as Phase 1b.
2. **Column naming:** `cv_anova_pvalue` (current — verbose, explicit) vs `Q2_pvalue` (chemometrics-canonical) vs `pCV-ANOVA` (matches SIMCA literally)? **Default:** `cv_anova_pvalue`.
3. **Greenlight:** with the v2 plan above, ready to implement, or any further changes?
