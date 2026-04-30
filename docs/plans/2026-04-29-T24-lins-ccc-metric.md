# T-24: Lin's Concordance Correlation Coefficient Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Add Lin's CCC as an additional regression metric in the result dict, distinguishing scale/bias bias from correlation.

**Architecture:** Always-computed regression metric (not a user-selectable toggle). Implementation lives in `src/spectral_predict/scoring.py` as a new pure-numpy helper `lins_ccc(y_true, y_pred)`. The helper is invoked from the existing regression metric block in `src/spectral_predict/search.py` (lines 4347-4395) where pooled CV predictions are aggregated, producing one new column `CCCcv`. A second invocation at the calibration-metric site (lines 4517-4519) produces a `CCC` column for calibration. Both columns are added to the regression `metric_cols` schema in `scoring.create_results_dataframe()` so they appear consistently in saved CSVs and the GUI results table. Composite scoring (`compute_composite_score`) is **not** modified — CCC is informational, sitting alongside RMSE/R²/RPD/RER/Bias.

**Tech Stack:** numpy, pytest. No new dependencies.

**Source:** roadmap T-24 (`docs/RECONCILED_ROADMAP_2026-04-29.md`).

---

## Background

**Why CCC matters scientifically:** Pearson r and R² are insensitive to systematic bias and scale shift — a model that predicts `ŷ = y + 5` (constant +5 bias) has Pearson r = 1.0 and a degenerate R² that can still look acceptable depending on framing, but it is unambiguously a worse predictor than `ŷ = y`. Lin's Concordance Correlation Coefficient (CCC) penalizes both correlation departures AND bias/scale departures from the 1:1 line. The FTIR Bone PLS paper that anchors `docs/RECONCILED_ROADMAP_2026-04-29.md` reports CCC alongside RMSE/R² as the standard concordance metric in chemometrics for predicted-vs-observed agreement.

**Definition (Lin 1989):** Given paired observations `(y_true_i, y_pred_i)` with means `μ_x, μ_y`, variances `σ_x², σ_y²`, and Pearson correlation `ρ`:

```
CCC = (2 * ρ * σ_x * σ_y) / (σ_x² + σ_y² + (μ_x - μ_y)²)
```

Equivalently, with population covariance `cov(x, y) = ρ * σ_x * σ_y`:

```
CCC = (2 * cov(x, y)) / (σ_x² + σ_y² + (μ_x - μ_y)²)
```

Range: `[-1, 1]`. CCC = 1 only when perfect agreement (predictions fall exactly on 1:1 line). CCC = -1 only on perfect anti-concordance.

**Citation:** Lin, L. I. (1989). "A concordance correlation coefficient to evaluate reproducibility." *Biometrics*, 45(1), 255-268. (Replicate the citation in the helper's docstring.)

**Decision: where surfaced.** Always computed for regression — emitted as `CCC` (calibration) and `CCCcv` (cross-validation) columns in the result dict, parallel to how `RMSE`/`RMSEcv`, `R2`/`R2cv`, `RPD`, `RER`, `Bias` are already always-on. This matches the existing metric philosophy (compute everything cheap, let the user sort/filter in the GUI). No GUI checkbox, no CLI flag. The user gets it for free on every regression run. (Rationale: CCC is ~10 floating-point ops on the existing pooled `(all_y_test, all_y_pred)` arrays — there is no performance cost worth gating it behind a UI toggle.)

**Why this is "~10 LOC" per the roadmap:** the formula itself is roughly 5 lines of numpy. Wiring is two single-line additions to `result["..."] = ...` blocks plus two entries in the `metric_cols` schema. The line count outside tests is ~10-15.

---

## Fix sites (pre-verified against current `main`)

| Location | What goes there |
|---|---|
| `src/spectral_predict/scoring.py` (new top-level function below `compute_specificity`, before `create_results_dataframe`) | `def lins_ccc(y_true, y_pred) -> float` — pure helper |
| `src/spectral_predict/scoring.py:400` (inside `create_results_dataframe`, regression `metric_cols` list) | Add `"CCC", "CCCcv"` after `"Bias", "RER"` |
| `src/spectral_predict/search.py:4519` (calibration regression branch, after `cal_r2 = r2_score(y, y_pred_cal)`) | `cal_ccc = lins_ccc(y, y_pred_cal)` |
| `src/spectral_predict/search.py:4360` (CV regression block, after `bias_cv = float(np.mean(all_y_pred - all_y_test))`) | `ccc_cv = lins_ccc(all_y_test, all_y_pred)` |
| `src/spectral_predict/search.py:4748` (regression result-dict population, after `result["RER"] = rer`) | `result["CCC"] = cal_ccc if cal_ccc is not None else np.nan` and `result["CCCcv"] = ccc_cv` |
| `src/spectral_predict/search.py:4481-4493` (cal_* initializers) | Add `cal_ccc = None` alongside `cal_rmse = None` |
| `tests/test_scoring.py` (or new `tests/test_lins_ccc.py`) | New test class `TestLinsCCC` |
| `spectral_predict_gui_optimized.py:1437-1448` (metric tooltip dict) | Add `'CCC'` and `'CCCcv'` tooltip entries (same dict block as RMSE/Bias) |
| `spectral_predict_gui_optimized.py:27964-27967` (`higher_is_better_cols` set in `_sort_results_by_column`) | Add `'CCC'` and `'CCCcv'` to the set |

Imports: `numpy as np` is already imported in `scoring.py` (line 5). The new helper needs nothing else. `search.py` needs `from spectral_predict.scoring import lins_ccc` added near the top imports if not already present (verify in Task 5 — there is currently no scoring import there because the regression metric block builds the result dict inline; the new import is required).

---

## Verification matrix

The verification harness is the unit-test suite. The four key TDD assertions (each is its own test):

| # | Scenario | Inputs | Expected CCC | Why |
|---|---|---|---|---|
| 1 | Perfect prediction | `y_pred == y_true` (e.g. arange) | `1.0` (within 1e-12) | Both numerator and denominator collapse to `2σ²`; ratio = 1 |
| 2 | Perfect anti-correlation | `y_pred = -y_true` (centered around 0) | `-1.0` (within 1e-12) | Pearson r = -1, σ_x = σ_y, μ_x = μ_y, so CCC = -1 |
| 3 | Bias-only shift | `y_pred = y_true + 5`, where `y_true` has nonzero variance | `0 < CCC < 1` AND CCC strictly less than Pearson r (which is exactly 1) | Penalizes systematic offset that R² is blind to |
| 4 | Scale-only shift | `y_pred = 2 * y_true`, with `y_true` zero-mean | `0 < CCC < 1` AND CCC strictly less than Pearson r (which is exactly 1) | Penalizes scale mismatch |

Plus auxiliary tests:
- Range check: CCC always in `[-1, 1]` for random arrays.
- NaN propagation: passing arrays with NaN → returns NaN (not crash).
- Length mismatch: passing mismatched-length arrays → raises `ValueError`.
- Constant predictions (zero variance): `y_pred = constant` → returns `0.0` (not NaN, not divide-by-zero).
- Constant truth (zero variance): `y_true = constant`, `y_pred` varying → returns `0.0`.
- Both constant and equal: `y_pred == y_true == constant` → returns `1.0` (degenerate but defined; document this convention).
- Symmetry: `lins_ccc(a, b) == lins_ccc(b, a)` — CCC is symmetric in its arguments.

---

## Task 1: Create fresh branch off main

**Files:** none modified yet.

**Step 1:** Verify clean baseline.
```bash
git status
git log --oneline -3
```
Expected: on `main`, no unstaged changes (other than `.claude/settings.local.json`, which is fine).

**Step 2:** Create and switch to new branch.
```bash
git checkout -b feature/t24-lins-ccc-metric
```
Expected: "Switched to a new branch 'feature/t24-lins-ccc-metric'".

No commit yet.

---

## Task 2: TDD — write failing tests first

**Files:**
- Modify: `tests/test_scoring.py`

**Step 1:** Append a new test class `TestLinsCCC` to `tests/test_scoring.py`.

The test class must import `lins_ccc` from `spectral_predict.scoring` (this import will fail at collection time until Task 3 lands — that's the TDD pattern).

Write these tests verbatim (one assertion per test where reasonable):

```python
class TestLinsCCC:
    """Tests for Lin's Concordance Correlation Coefficient.

    Reference: Lin, L. I. (1989). "A concordance correlation coefficient
    to evaluate reproducibility." Biometrics, 45(1), 255-268.
    """

    def test_perfect_prediction_returns_one(self):
        """Perfect prediction (ŷ = y) should give CCC = 1."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(0.0, 10.0, 50)
        y_pred = y_true.copy()
        assert abs(lins_ccc(y_true, y_pred) - 1.0) < 1e-12

    def test_perfect_anticorrelation_returns_minus_one(self):
        """Perfectly anti-correlated, mean-zero predictions give CCC = -1."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(-5.0, 5.0, 50)  # mean = 0
        y_pred = -y_true                       # mean = 0, σ_y = σ_x, ρ = -1
        assert abs(lins_ccc(y_true, y_pred) - (-1.0)) < 1e-12

    def test_bias_only_below_one_even_when_pearson_is_one(self):
        """ŷ = y + 5: Pearson r = 1 but CCC < 1 (penalizes systematic bias)."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(0.0, 10.0, 50)
        y_pred = y_true + 5.0
        ccc = lins_ccc(y_true, y_pred)
        # Pearson would be exactly 1.0 here — CCC must be strictly less.
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        assert abs(pearson - 1.0) < 1e-12, "sanity: Pearson should be 1 for pure bias"
        assert ccc < 1.0
        assert ccc > 0.0  # still concordant in trend

    def test_scale_only_below_one_even_when_pearson_is_one(self):
        """ŷ = 2y (zero-mean y): Pearson r = 1 but CCC < 1 (penalizes scale mismatch)."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(-5.0, 5.0, 50)  # mean = 0 → no bias term
        y_pred = 2.0 * y_true                # ρ = 1, σ_y = 2σ_x
        ccc = lins_ccc(y_true, y_pred)
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        assert abs(pearson - 1.0) < 1e-12
        assert ccc < 1.0
        # Closed-form: CCC = 2 * 1 * σ * 2σ / (σ² + 4σ² + 0) = 4σ² / 5σ² = 0.8
        assert abs(ccc - 0.8) < 1e-12

    def test_known_closed_form_scale_only(self):
        """Closed-form check (redundant with above but explicit): scale = 2 → CCC = 4/5."""
        from spectral_predict.scoring import lins_ccc
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(1000)
        y_true -= y_true.mean()  # exact zero mean
        y_pred = 2.0 * y_true
        assert abs(lins_ccc(y_true, y_pred) - 0.8) < 1e-2  # sample noise tolerance

    def test_range_within_bounds_random_inputs(self):
        """Random paired arrays produce CCC in [-1, 1]."""
        from spectral_predict.scoring import lins_ccc
        rng = np.random.default_rng(42)
        for _ in range(20):
            y_true = rng.standard_normal(100)
            y_pred = rng.standard_normal(100)
            ccc = lins_ccc(y_true, y_pred)
            assert -1.0 <= ccc <= 1.0

    def test_symmetry_in_arguments(self):
        """CCC(a, b) = CCC(b, a)."""
        from spectral_predict.scoring import lins_ccc
        rng = np.random.default_rng(1)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        assert abs(lins_ccc(a, b) - lins_ccc(b, a)) < 1e-12

    def test_nan_in_inputs_returns_nan(self):
        """NaN in either array yields NaN (not crash, not silent zero)."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.array([1.0, 2.0, np.nan, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(lins_ccc(y_true, y_pred))

    def test_length_mismatch_raises(self):
        """Different-length arrays raise ValueError."""
        from spectral_predict.scoring import lins_ccc
        with pytest.raises(ValueError):
            lins_ccc(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_constant_predictions_returns_zero(self):
        """Zero-variance predictions (degenerate model): return 0.0, not NaN/inf."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(0.0, 10.0, 20)
        y_pred = np.full(20, 5.0)
        result = lins_ccc(y_true, y_pred)
        assert result == 0.0  # degenerate but defined by convention

    def test_constant_truth_returns_zero(self):
        """Zero-variance truth: return 0.0, not NaN/inf."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.full(20, 7.0)
        y_pred = np.linspace(0.0, 10.0, 20)
        assert lins_ccc(y_true, y_pred) == 0.0

    def test_both_constant_and_equal_returns_one(self):
        """Both arrays identical constants: degenerate but in agreement → 1.0."""
        from spectral_predict.scoring import lins_ccc
        y_true = np.full(20, 7.0)
        y_pred = np.full(20, 7.0)
        assert lins_ccc(y_true, y_pred) == 1.0

    def test_accepts_lists_and_pandas_series(self):
        """Helper coerces list / pandas Series to numpy float arrays."""
        from spectral_predict.scoring import lins_ccc
        import pandas as pd
        y_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(lins_ccc(y_list, y_series) - 1.0) < 1e-12
```

**Step 2:** Verify the tests fail with the expected error (function doesn't exist yet).

```bash
.venv311/Scripts/python.exe -m pytest tests/test_scoring.py::TestLinsCCC -v
```

Expected: every test fails at the `from spectral_predict.scoring import lins_ccc` line with `ImportError`. This is the TDD red state — the next task makes them green.

**Step 3:** Commit the failing tests.

```bash
git add tests/test_scoring.py
git commit -m "test: add failing TDD tests for Lin's CCC metric (T-24)"
```

---

## Task 3: Implement `lins_ccc` helper in `scoring.py`

**Files:**
- Modify: `src/spectral_predict/scoring.py`

**Step 1:** Add the helper function. Insert it immediately after `compute_specificity()` (which ends at line ~366) and before `create_results_dataframe()` (currently at line 369). Do not touch any other code yet.

The function:

```python
def lins_ccc(y_true, y_pred) -> float:
    """Compute Lin's Concordance Correlation Coefficient.

    CCC measures agreement between paired observations along the 1:1 line.
    Unlike Pearson r (which is invariant to scale and bias) or R² (which
    can be inflated under bias), CCC penalizes BOTH correlation departures
    AND systematic shift / scale-change of the predictions away from the
    identity line. Range: [-1, 1].

    Formula (Lin 1989):
        CCC = 2·ρ·σ_x·σ_y / (σ_x² + σ_y² + (μ_x − μ_y)²)

    where ρ is the Pearson correlation between x = y_true and y = y_pred.

    Parameters
    ----------
    y_true : array-like
        Observed reference values.
    y_pred : array-like
        Predicted values, same length as y_true.

    Returns
    -------
    ccc : float
        Concordance correlation coefficient in [-1, 1].
        Returns 0.0 when either input has zero variance and the two
        arrays are not identical (degenerate-but-defined convention).
        Returns 1.0 when both inputs are equal constants.
        Returns NaN if either input contains NaN.

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.

    References
    ----------
    Lin, L. I. (1989). A concordance correlation coefficient to evaluate
    reproducibility. Biometrics, 45(1), 255-268.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )

    # NaN propagation — if any NaN in either array, return NaN.
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return float("nan")

    # Population (biased) variances and covariance — Lin (1989) formulation.
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()       # ddof=0
    var_pred = y_pred.var()       # ddof=0
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    # Degenerate cases (zero-variance inputs):
    #   - Both constants AND equal → perfect agreement → 1.0
    #   - Either zero-variance and they differ → no concordance information → 0.0
    if denominator == 0.0:
        # Both inputs constant; if they are equal constants, return 1.0,
        # otherwise the data are still "in agreement on a single point"
        # — but if mean_true != mean_pred the (μ_x − μ_y)² term would
        # have been nonzero, so reaching here implies equal constants.
        return 1.0
    if var_true == 0.0 or var_pred == 0.0:
        # One side has zero variance, the other does not. Pearson is
        # undefined; CCC convention used here returns 0.0 (no concordance
        # information). This matches the test_constant_predictions_returns_zero
        # / test_constant_truth_returns_zero contract.
        return 0.0

    return float(2.0 * cov / denominator)
```

**Step 2:** Verify the tests now pass.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_scoring.py::TestLinsCCC -v
```

Expected: all 13 tests pass (green). If any fail, debug — do NOT lower the tolerance and do NOT change the expected values.

**Step 3:** Verify the rest of the scoring suite still passes (regression check).

```bash
.venv311/Scripts/python.exe -m pytest tests/test_scoring.py -v
```

Expected: all pre-existing tests still pass.

**Step 4:** Commit.

```bash
git add src/spectral_predict/scoring.py
git commit -m "feat: add lins_ccc helper to scoring (T-24)

Implements Lin's Concordance Correlation Coefficient per Lin (1989),
'A concordance correlation coefficient to evaluate reproducibility',
Biometrics 45(1), 255-268. CCC penalizes both correlation departures
and bias/scale shift away from the 1:1 line — useful in chemometrics
for assessing predicted-vs-observed agreement where Pearson r and R²
can be misleadingly optimistic under systematic bias."
```

---

## Task 4: Add `CCC` / `CCCcv` to the regression results schema

**Files:**
- Modify: `src/spectral_predict/scoring.py` (line 400)

**Step 1:** Find the `create_results_dataframe` regression branch:

```python
    if task_type == "regression":
        # Calibration metrics first, then CV metrics, then NIR-specific metrics
        metric_cols = ["RMSE", "R2", "RMSEcv", "R2cv", "MAEcv", "RPD", "Bias", "RER"]
```

Replace with:

```python
    if task_type == "regression":
        # Calibration metrics first, then CV metrics, then NIR-specific metrics
        metric_cols = ["RMSE", "R2", "RMSEcv", "R2cv", "MAEcv", "RPD", "Bias", "RER", "CCC", "CCCcv"]
```

**Step 2:** Sanity-check the dataframe builder still works. There are no existing tests for `create_results_dataframe()`'s regression branch directly, but `test_scoring.py::TestCompositeScoring` constructs result rows manually — those tests must still pass because they don't reference CCC columns.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_scoring.py -v
```

Expected: all pass.

**Step 3:** Commit.

```bash
git add src/spectral_predict/scoring.py
git commit -m "feat: include CCC/CCCcv in regression results schema (T-24)"
```

---

## Task 5: Wire CCC into `search.py` regression metric block

**Files:**
- Modify: `src/spectral_predict/search.py` (lines 4347-4360, 4481-4493, 4517-4519, 4737-4748)

**Step 1:** Add the import. Find the existing imports near the top of `search.py` (the file already imports a handful of helpers from `spectral_predict.scoring` if any — verify with Grep). If there's no existing scoring import, add at the appropriate location among intra-package imports:

```python
from spectral_predict.scoring import lins_ccc
```

If a `from spectral_predict.scoring import ...` line already exists, just add `lins_ccc` to it.

**Step 2:** Initialize `cal_ccc = None` alongside the other `cal_*` initializers (lines 4481-4493). Add it after `cal_r2 = None`:

```python
    cal_rmse = None
    cal_r2 = None
    cal_ccc = None
```

**Step 3:** Compute calibration CCC. In the regression calibration branch (currently lines 4517-4519):

```python
        if task_type == "regression":
            cal_rmse = np.sqrt(mean_squared_error(y, y_pred_cal))
            cal_r2 = r2_score(y, y_pred_cal)
```

Add:

```python
        if task_type == "regression":
            cal_rmse = np.sqrt(mean_squared_error(y, y_pred_cal))
            cal_r2 = r2_score(y, y_pred_cal)
            cal_ccc = lins_ccc(y, y_pred_cal)
```

**Step 4:** Compute CV CCC from pooled predictions. In the regression CV-metric block (currently around lines 4358-4361):

```python
        # Bias: Mean prediction error (positive = systematic overprediction)
        bias_cv = float(np.mean(all_y_pred - all_y_test))
```

Add immediately after:

```python
        # CCC: Lin's Concordance Correlation Coefficient on pooled CV predictions.
        # Penalizes both correlation departures and systematic bias/scale shift —
        # complement to R²cv, which is bias-blind under linear shifts.
        ccc_cv = lins_ccc(all_y_test, all_y_pred)
```

**Step 5:** Initialize `ccc_cv` defensively. Look near the top of the regression aggregation block where `mean_rmse`, `mean_r2`, etc. are computed (around line 4351). If there's a fallback path where these can be set to `np.nan` (search for `mean_rmse = np.nan` or similar), set `ccc_cv = np.nan` in the same fallback. If no such fallback exists, then `ccc_cv` is unconditionally set inside the regression branch — leave it alone.

(Concretely: the regression branch from `if task_type == "regression":` at ~4337 to its else at ~4395 always executes the CCC line if it's placed after `bias_cv`, so no defensive init is needed there. The only risk is if `all_y_test`/`all_y_pred` could be empty — verify this is guaranteed nonempty by upstream filtering.)

**Step 6:** Populate the result dict. In the result-dict population block (lines 4737-4748):

```python
    if task_type == "regression":
        # Calibration metrics (training data)
        result["RMSE"] = cal_rmse if cal_rmse is not None else np.nan
        result["R2"] = cal_r2 if cal_r2 is not None else np.nan
        # Cross-validation metrics (test fold averages)
        result["RMSEcv"] = mean_rmse
        result["R2cv"] = mean_r2
        # NIR-specific metrics (computed from aggregated CV predictions)
        result["MAEcv"] = mae_cv
        result["RPD"] = rpd
        result["Bias"] = bias_cv
        result["RER"] = rer
```

Replace with (CCC additions inserted in the natural place — calibration CCC next to R²/RMSE, CV CCC next to R²cv):

```python
    if task_type == "regression":
        # Calibration metrics (training data)
        result["RMSE"] = cal_rmse if cal_rmse is not None else np.nan
        result["R2"] = cal_r2 if cal_r2 is not None else np.nan
        result["CCC"] = cal_ccc if cal_ccc is not None else np.nan
        # Cross-validation metrics (test fold averages)
        result["RMSEcv"] = mean_rmse
        result["R2cv"] = mean_r2
        result["CCCcv"] = ccc_cv
        # NIR-specific metrics (computed from aggregated CV predictions)
        result["MAEcv"] = mae_cv
        result["RPD"] = rpd
        result["Bias"] = bias_cv
        result["RER"] = rer
```

**Step 7:** Smoke-test by importing.

```bash
.venv311/Scripts/python.exe -c "from spectral_predict import search; print('ok')"
```

Expected: `ok`.

**Step 8:** Confirm there are no other regression-result-dict population sites that need the same wiring. Run a grep:

Use Grep tool for `result\["RPD"\]` in `src/spectral_predict/`.

Expected: only the one site at `search.py:4746`. If any others appear (e.g., one_class branch, alternate result-builder), evaluate each — CCC is regression-only, so one-class and classification branches stay untouched.

Also grep for parallel result-construction sites that might emit regression metrics under a different code path:

Use Grep tool for `"RMSEcv"` in `src/spectral_predict/`.

Expected matches: `search.py` (the canonical site), `nsga2_search.py` (NSGA-II metric block), `unified_bayesian.py` (Bayesian metric reporting), `bayesian_utils.py`. **For each match in those auxiliary files, evaluate whether they construct full regression result rows that flow into the user-visible results dataframe.** If they do, repeat the same wiring (call `lins_ccc(y_pooled_test, y_pooled_pred)` and emit `CCC` / `CCCcv`). If they only print or log a subset of metrics, leave them.

(This step is the most likely to surface a "missed site." It is the moral equivalent of the "search.py:380 silent default" finding in T-19. Be thorough.)

**Step 9:** Commit.

```bash
git add src/spectral_predict/search.py
git commit -m "feat: emit CCC and CCCcv in regression result dict (T-24)"
```

---

## Task 6: Wire CCC into auxiliary result-construction sites (if needed)

**Files (conditional on Task 5 Step 8 findings):**
- `src/spectral_predict/nsga2_search.py`
- `src/spectral_predict/unified_bayesian.py`
- `src/spectral_predict/bayesian_utils.py`

**Step 1:** For each file flagged in Task 5 Step 8 as constructing a regression result row, locate the equivalent of the `search.py:4737-4748` block (the place where `result["RMSEcv"]` and `result["R2cv"]` are assigned). Add the same CCC pair:

```python
result["CCC"] = ...      # from cal y / cal y_pred via lins_ccc
result["CCCcv"] = ...    # from pooled CV y / pooled CV y_pred via lins_ccc
```

If pooled CV predictions aren't readily available in that path (e.g. NSGA-II reports per-fold means rather than pooling), set `result["CCCcv"] = np.nan` and add a comment noting that pooled CCC is not computed in that path. **Do NOT** average per-fold CCCs — that's mathematically wrong for the same reason averaging per-fold R² is wrong (different fold variances).

**Step 2:** If Task 5 Step 8 surfaced no auxiliary sites with full regression result rows, skip this task entirely and note that in the commit message.

**Step 3:** Run the full test suite:

```bash
.venv311/Scripts/python.exe -m pytest tests/ -v
```

Expected: all pass.

**Step 4:** Commit (only if changes were made).

```bash
git add src/spectral_predict/
git commit -m "feat: emit CCC in NSGA-II/Bayesian regression result paths (T-24)"
```

---

## Task 7: GUI tooltip and sort-direction wiring

**Files:**
- Modify: `spectral_predict_gui_optimized.py` (lines 1437-1448, 27964-27967)

**Step 1:** Add tooltip entries. In the `'metrics': { ... }` dict around line 1438, add entries adjacent to the existing `'Bias'` entry:

```python
        'CCC': "CCC (Lin's Concordance Correlation Coefficient)\n\n"
               "Agreement with the 1:1 line. Range: -1 to 1 (1 = perfect).\n"
               "Unlike R², CCC penalizes systematic bias and scale shift.\n"
               "Reference: Lin (1989), Biometrics 45(1).",
        'CCCcv': "CCCcv (CCC Cross-Validation)\n\n"
                 "Lin's CCC computed on pooled out-of-fold predictions.\n"
                 "More reliable than calibration CCC for judging real-world fit.",
```

**Step 2:** Add sort-direction. In `_sort_results_by_column` around line 27964:

```python
        higher_is_better_cols = {
            'R2', 'R2cv', 'R²', 'Accuracy', 'Accuracycv',
            'ROC_AUC', 'F1', 'F1cv', 'ROC_AUCcv', 'RPD', 'RER',
        }
```

Replace with:

```python
        higher_is_better_cols = {
            'R2', 'R2cv', 'R²', 'Accuracy', 'Accuracycv',
            'ROC_AUC', 'F1', 'F1cv', 'ROC_AUCcv', 'RPD', 'RER',
            'CCC', 'CCCcv',
        }
```

**Step 3:** Smoke-test the GUI module imports.

```bash
.venv311/Scripts/python.exe -c "import importlib.util, pathlib; spec = importlib.util.spec_from_file_location('gui', pathlib.Path('spectral_predict_gui_optimized.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"
```

This is heavy because it loads the whole GUI; if it's too slow or crashes on `tkinter` import in a headless context, fall back to a syntax-only check:

```bash
.venv311/Scripts/python.exe -m py_compile spectral_predict_gui_optimized.py && echo ok
```

Expected: `ok`.

**Step 4:** Commit.

```bash
git add spectral_predict_gui_optimized.py
git commit -m "feat: GUI tooltip + sort direction for CCC/CCCcv (T-24)"
```

---

## Task 8: End-to-end smoke test on real data

**Files:**
- No new files — uses existing test data.

**Step 1:** Run a tiny `run_search` against `example/` BoneCollagen and confirm both `CCC` and `CCCcv` columns appear in the result dataframe with finite values.

```bash
.venv311/Scripts/python.exe -c "
import sys, pathlib
sys.path.insert(0, 'src')
import pandas as pd
from spectral_predict.io import read_asd_dir
from spectral_predict.search import run_search

example_dir = pathlib.Path('example')
spectra, _ = read_asd_dir(example_dir)
ref = pd.read_csv(example_dir / 'BoneCollagen.csv', encoding='utf-8-sig')
ref['__key__'] = ref['File Number'].str.replace(' ', '', regex=False)
spectra.index = spectra.index.str.replace('.asd', '', regex=False)
joined = spectra.join(ref.set_index('__key__')[['%Collagen']], how='inner').dropna(subset=['%Collagen'])
y = joined['%Collagen'].astype(float)
X = joined.drop(columns=['%Collagen']).astype(float)

df, _ = run_search(
    X, y, task_type='regression',
    folds=5, models_to_test=['PLS'],
    preprocessing_methods={'raw': True},
    enable_variable_subsets=False, enable_region_subsets=False,
)

assert 'CCC' in df.columns, f'CCC missing! columns: {list(df.columns)}'
assert 'CCCcv' in df.columns, f'CCCcv missing! columns: {list(df.columns)}'
finite_ccc = df['CCC'].dropna()
finite_ccccv = df['CCCcv'].dropna()
print(f'Rows: {len(df)}, finite CCC: {len(finite_ccc)}, finite CCCcv: {len(finite_ccccv)}')
print(f'CCC range: [{finite_ccc.min():.4f}, {finite_ccc.max():.4f}]')
print(f'CCCcv range: [{finite_ccccv.min():.4f}, {finite_ccccv.max():.4f}]')
assert (finite_ccc >= -1.0).all() and (finite_ccc <= 1.0).all(), 'CCC out of [-1, 1]'
assert (finite_ccccv >= -1.0).all() and (finite_ccccv <= 1.0).all(), 'CCCcv out of [-1, 1]'
# Sanity: for a working PLS model on real data we expect CCC >= 0.
assert finite_ccc.max() > 0.5, 'expected at least one PLS run with CCC > 0.5'
print('OK')
"
```

Expected: `OK` printed; finite ranges within `[-1, 1]`.

**Step 2:** If the smoke test fails because of an unrelated issue (e.g., a separate broken kwarg), narrow the kwargs further or fall back to a synthetic dataset:

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n, p = 60, 200
X = pd.DataFrame(rng.standard_normal((n, p)))
y = pd.Series(X.iloc[:, :5].sum(axis=1) + 0.1 * rng.standard_normal(n))
```

Then call `run_search(X, y, task_type='regression', folds=5, models_to_test=['PLS'], preprocessing_methods={'raw': True}, enable_variable_subsets=False, enable_region_subsets=False)`.

**Step 3:** Do NOT commit smoke-test scripts. The test was just for verification.

---

## Task 9: Documentation + roadmap update

**Files:**
- Modify: `docs/RECONCILED_ROADMAP_2026-04-29.md` (T-24 entry)
- Modify: `docs/PROJECT_STATUS.md`
- Append to: `docs/SESSION_LOG.md`

**Step 1:** Mark T-24 as completed in the roadmap. Locate the T-24 entry (around line 268):

```markdown
### T-24: Add Lin's CCC as a metric option
**Where:** `scoring.py`
**Effort:** ~10 LOC.
```

Replace with:

```markdown
### T-24: Add Lin's CCC as a metric option ✅ DONE
**Where:** `scoring.py` (helper), `search.py` (wiring), `spectral_predict_gui_optimized.py` (tooltip + sort).
**Result:** `CCC` (calibration) and `CCCcv` (cross-validation) always-on regression metrics. Lin (1989) formula. Tests in `tests/test_scoring.py::TestLinsCCC`. Implementation: ~30 LOC including tests.
**Effort:** ~10 LOC core + 100 LOC tests + ~10 LOC GUI/wiring (matches roadmap estimate for the core).
```

**Step 2:** Add a one-liner to `docs/PROJECT_STATUS.md` under "What Works" or "Recently Resolved":

```markdown
- Lin's CCC metric (T-24): `CCC` and `CCCcv` columns in regression results. Penalizes bias/scale shift that R² ignores. Lin (1989) formula.
```

**Step 3:** Append to `docs/SESSION_LOG.md` under today's date:

```markdown
### T-24: Lin's CCC metric implemented

Added `lins_ccc(y_true, y_pred)` helper in `scoring.py` per Lin (1989). Wired into `search.py` regression metric block — `CCC` from calibration predictions, `CCCcv` from pooled CV predictions. Schema updated in `create_results_dataframe()`. GUI tooltip + sort direction added.

Design choices:
- Always computed for regression (no GUI toggle) — same philosophy as RPD/RER/Bias.
- Pooled CCCcv from `(all_y_test, all_y_pred)`, not averaged per-fold (averaging per-fold concordance is mathematically wrong, same reason averaging per-fold R² is wrong).
- Population variances (ddof=0), per the Lin (1989) original formulation.
- Degenerate cases: zero-variance predictions / truth → 0.0; both equal constants → 1.0; NaN inputs → NaN.

Tests cover: perfect prediction → 1, perfect anti-correlation → -1, bias-only → CCC < 1 even though Pearson r = 1, scale-only → CCC = 0.8 (closed form for ŷ = 2y), range [-1, 1], symmetry, NaN propagation, length-mismatch raises, degenerate constants, list/Series inputs.
```

**Step 4:** Commit.

```bash
git add docs/RECONCILED_ROADMAP_2026-04-29.md docs/PROJECT_STATUS.md docs/SESSION_LOG.md
git commit -m "docs: close T-24 — Lin's CCC metric"
```

---

## Task 10: Final verification

**Step 1:** Run the full test suite end-to-end.

```bash
.venv311/Scripts/python.exe -m pytest tests/ -v
```

Expected: all green. Any pre-existing failures unrelated to this work are flagged but not blocking.

**Step 2:** Show final git log on the branch.

```bash
git log --oneline main..HEAD
```

Expected: 5-7 commits matching the task structure.

**Step 3:** Push the branch (if requested by the user — do NOT auto-push).

---

## Non-goals / explicitly out of scope

- **CCC for classification.** CCC is regression-specific. Do NOT emit it for classification or one-class result rows.
- **Composite-score integration.** CCC is informational. `compute_composite_score` continues to rank by `R2cv` (regression). Adding CCC into the ranking would change user-facing rankings retroactively for every saved CSV — out of scope.
- **CI for CCC (Lin's confidence interval per Lin 1989 §3).** The variance-of-CCC formulas in Lin (1989) are useful but not requested. T-16 (bootstrap CIs) covers this need separately.
- **Per-fold CCC.** Pooled-CV CCC matches how RMSEcv/R²cv are computed in this codebase. Per-fold CCC averaging is mathematically wrong; do not add it.
- **CCC for one-class detectors.** One-class CV pools binary decisions, not continuous regression predictions. CCC doesn't apply.
- **Renaming existing columns or backward-compat handling for old saved CSVs that lack CCC columns.** Old CSVs simply load without CCC columns; this is fine because pandas handles missing columns gracefully in the GUI display layer.

---

## Open questions for reviewer

1. **Auxiliary result-construction sites.** Task 5 Step 8 asks to grep `unified_bayesian.py` and `nsga2_search.py` for full regression result rows. Are those paths active in the user-facing results table, or are they only used internally for trial selection? If only internal, skipping CCC there is fine. The roadmap's "~10 LOC" estimate suggests the user thinks of this as a single-site change.
2. **NaN convention for degenerate inputs.** Tests assume `lins_ccc(constant, varying) == 0.0`. An alternative convention is to return NaN. The 0.0 convention matches "no concordance information" and lets sorting/aggregation work without special-casing; the NaN convention matches Pearson-r practice. I picked 0.0 for robustness in the GUI display path. Reviewer can flip this.
3. **Pooled-CV CCC under repeated CV.** Codebase uses `reduce_repeated_cv_predictions` to majority-pool per-sample under repeated K-fold. The pooled `(all_y_test, all_y_pred)` arrays are exactly what we want for CCCcv — verified by re-reading `search.py:4340-4351`. No special-casing needed.
4. **Population vs sample variance (ddof).** Lin (1989) uses population variances. sklearn's `r2_score` and `mean_squared_error` are also "n" denominators (not "n-1"). Using `ddof=0` (np default) matches both. If a reviewer cites a chemometrics tradition of `ddof=1`, the change is one keyword everywhere — but the closed-form scale-only test (CCC = 0.8 for ŷ = 2y) only holds exactly under `ddof=0`. Sticking with `ddof=0`.
