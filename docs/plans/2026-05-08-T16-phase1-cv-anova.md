# T-16 Phase 1 — CV-ANOVA p-value column for PLS regression rows

**Branch:** `feat/T16-phase1-cv-anova` (off `main` at `98e6e98`)
**Status:** plan, not started
**Effort:** ~half day source + tests

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
p_value     = 1 - scipy.stats.f.cdf(F, df1, df2)
```

**Edge cases:**
- `N - A - 1 <= 0` (over-parametrised — too many LVs for the data): return `np.nan`. Should not happen in practice because dasp's PLS clamps `n_components <= n_train // 2 - 1`, but defensive.
- `PRESS >= SSY` (model worse than mean prediction — degenerate): F becomes negative; clip F to 0 so p_value clamps to 1.0 (correctly indicating "not significant").
- `A < 1`: shouldn't happen; defensive `np.nan`.

**Pre-conditions to populate the column:**
- `model_name in {"PLS", "PLSRegression"}` (regression PLS variants only — exclude PLS-DA)
- `task_type == "regression"`
- `RMSEcv is not None and not np.isnan(RMSEcv)`
- `n_components` extractable from `params`

If any pre-condition fails: column gets `np.nan`. Non-PLS rows always get `np.nan`.

---

## 3. Files to touch

### 3.1 New helper in `src/spectral_predict/scoring.py`

Add a pure function `compute_cv_anova_pvalue()` near the existing scoring helpers. Pure scipy + numpy, no dasp-internal dependencies. Keep it testable in isolation.

```python
def compute_cv_anova_pvalue(
    y_true: np.ndarray,
    rmsecv: float,
    n_components: int,
) -> float:
    """CV-ANOVA F-test p-value (Eriksson, Trygg & Wold 2008).

    Returns p-value for the null hypothesis that the PLS model's CV
    PRESS is no better than mean-prediction. Only defined for PLS
    regression with single Y. Returns nan on degenerate inputs.
    """
    # ... implementation per §2 math ...
```

LOC: ~25 source + ~8 docstring.

### 3.2 Insertion point in `_run_single_config()` (`search.py:4488`)

After the regression-metrics block at `search.py:5256-5269` (where `result["RMSEcv"] = mean_rmse` lands), add the call:

```python
# CV-ANOVA p-value for PLS regression rows (Eriksson 2008)
if model_name in ("PLS", "PLSRegression"):
    n_lv = params.get("n_components")
    if n_lv is not None and not np.isnan(mean_rmse):
        result["cv_anova_pvalue"] = compute_cv_anova_pvalue(
            y_true=y, rmsecv=mean_rmse, n_components=int(n_lv)
        )
    else:
        result["cv_anova_pvalue"] = np.nan
else:
    result["cv_anova_pvalue"] = np.nan
```

LOC: ~10. Inserted immediately after line 5269. Variable `y` is already in scope (parameter at `4490`); `mean_rmse` is the RMSEcv at `5262`; `params` is in scope at `4494`; `model_name` at `4493`.

### 3.3 Import in `search.py`

Add `compute_cv_anova_pvalue` to the existing scoring imports at the top of `search.py`. Single-line change.

### 3.4 No GUI change for Phase 1

The column lands in the CSV automatically. The Results tab leaderboard already auto-populates from result-CSV columns. If we want a dedicated GUI display (e.g., asterisk on rows where `cv_anova_pvalue < 0.05`), defer to a Phase 1.5 polish ticket — not in scope here.

### 3.5 Result-CSV schema

`cv_anova_pvalue` joins the existing per-row scalar columns. Old result CSVs lacking this column continue to load (pandas yields `nan` for missing columns); old code reading new CSVs ignores the column. Backwards-compatible.

---

## 4. Tests

New file `tests/test_cv_anova.py`. Approximate ~80 LOC, ~5 test cases:

1. **High-signal case**: synthetic `y = X[:, 5] + 0.01*noise`; PLS with 2 LVs on N=50; expect `p < 0.001`.
2. **No-signal case**: synthetic `y = noise` (uncorrelated with X); expect `p > 0.5`.
3. **Degenerate case** `PRESS >= SSY`: model worse than mean; expect `p == 1.0` exactly (clipped).
4. **Edge case** `N - A - 1 <= 0`: over-parametrised; expect `np.nan`.
5. **Reference value pin**: hand-computed F and p on a 10-sample fixture, asserted within `1e-6` of `scipy.stats.f.sf(F, A, N-A-1)`.

Plus one integration test extending the existing search-pipeline tests:
6. **End-to-end through `run_search`**: load BoneCollagen subset (or the existing test fixture), run `run_search(models_to_test=['PLS'])`, assert `cv_anova_pvalue` column present, all PLS rows have non-nan values, all values are in `[0, 1]`.

Verification command: `.venv312/Scripts/python.exe -m pytest tests/test_cv_anova.py -v` plus a `py_compile` of the modified `search.py` and `scoring.py`.

---

## 5. Commit shape

Single commit, branch `feat/T16-phase1-cv-anova`:

```
feat(T-16): CV-ANOVA p-value column for PLS regression rows

Adds cv_anova_pvalue per Eriksson, Trygg & Wold 2008 to the result CSV
for PLS regression rows. Computed at row finalization in
_run_single_config from already-existing RMSEcv, n_components, and
y_true. Pure additive — no change to which rows get computed, which
row wins, or any search-loop behavior. Non-PLS rows and PLS-DA rows
get nan.

Phase 1 of T-16 (model-comparison machinery survey at
docs/T16_MODEL_COMPARISON_SURVEY.md). Phase 2 (on-demand permutation
test) is a separate ticket.

Files:
- src/spectral_predict/scoring.py: new compute_cv_anova_pvalue() helper
- src/spectral_predict/search.py: insertion point in _run_single_config
- tests/test_cv_anova.py: 6 unit + integration tests
```

LOC estimate: ~35 source + ~80 tests = ~115 total.

---

## 6. Verification battery

Before commit:
- `py_compile src/spectral_predict/scoring.py src/spectral_predict/search.py` — no syntax errors
- `pytest tests/test_cv_anova.py -v` — 6/6 pass
- Smoke test: `.venv312/Scripts/python.exe -c "from spectral_predict.search import run_search; ..."` running PLS on BoneCollagen and asserting `cv_anova_pvalue` lands in the resulting DataFrame (one-shot script, not committed)

After commit, before push:
- `git diff --stat main..feat/T16-phase1-cv-anova` — confirm only 3 files changed, no whitespace drift
- Spot-check a fresh result CSV from a real GUI run on BoneCollagen — confirm the column appears with sensible values (PLS rows: small p; non-PLS rows: nan)

---

## 7. Out of scope (defer to follow-up tickets if user wants)

- **Phase 1b — PLS-DA CV-ANOVA**: extend to classification by computing PRESS on continuous PLS predictions of dummy-coded class labels. Math is straightforward; complexity is in extracting per-fold continuous predictions vs the threshold-applied class labels currently surfaced. Estimate ~half day.
- **Phase 1c — F-test LV selection rule (OPUS variant)**: opt-in alternative to min-RMSEcv selection. Dropped from Phase 1 to keep `run_search` signature pristine. Estimate ~half day if revived.
- **Phase 1d — GUI asterisk / color coding** for `cv_anova_pvalue < 0.05`: cosmetic polish. Estimate ~couple hours.
- **Phase 2 — on-demand permutation test for arbitrary leaderboard rows**: separate plan, separate branch. Estimate ~2-3 days.

---

## 8. Open questions for the reviewer

1. **PLS-DA scope:** ship Phase 1 as PLS-regression-only, or bundle PLS-DA in the same commit? The math extends but adds a per-fold-continuous-prediction wiring requirement. My read: PLS-regression first, PLS-DA as Phase 1b.
2. **Column naming:** `cv_anova_pvalue` (verbose, explicit) vs `Q2_pvalue` (chemometrics-canonical naming) vs `pCV-ANOVA` (matches SIMCA literally)? My read: `cv_anova_pvalue` because dasp's column-naming convention is snake_case English.
3. **Imports placement:** import `scipy.stats` lazily inside `compute_cv_anova_pvalue()` (avoid top-level import cost) or at top of `scoring.py` (already imported elsewhere)? My read: top-of-file since scipy is already a dasp dependency and scoring.py likely uses it.
