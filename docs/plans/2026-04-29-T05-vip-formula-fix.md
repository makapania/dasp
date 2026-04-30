# T-05: Fix VIP Formula Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incorrect VIP computation in `compute_vip()` with the canonical Wold (2001) formula using per-component `y_loadings_`.

**Architecture:** Single-function fix in `src/spectral_predict/models.py:compute_vip`. Parametrized regression test verifies new formula matches independent computation; old formula does not.

**Tech Stack:** numpy, scikit-learn (PLSRegression), pytest, existing project test infrastructure.

**Source:** roadmap T-05 in `docs/RECONCILED_ROADMAP_2026-04-29.md`.

---

## Background

### The bug

`compute_vip()` (`src/spectral_predict/models.py:1709-1751`) currently computes per-component explained Y variance as:

```python
ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)   # WRONG
```

This multiplies the sum of squared X-scores for each component by a single scalar (`np.var(y)`), so every component gets weighted by the **same** value. The component-axis spread is therefore driven entirely by `sum(T_a**2)` (the X-score energy), independent of how much **Y** each component actually explains. VIP collapses to a re-weighting of `W**2` by X-score energy, which is *not* VIP.

### The canonical formula

The correct definition (Wold, 2001; Mehmood et al., 2012) is:

> Let `T` be the X-scores `(n_samples, A)`, `q` be the y-loadings `(A,)` (or `y_loadings_[:, 0]` for univariate Y in scikit-learn's `PLSRegression`), and `W` be the X-weights `(p, A)`.
>
> The Y variance explained by component `a` is:
>
>     SSY_a = q_a**2 * (T_a.T @ T_a)
>
> and the total explained Y variance is `SSY_total = sum_a SSY_a`.
>
> Then for variable `j`:
>
>     VIP_j = sqrt( p * sum_a [ SSY_a * (W_{j,a} / ||W_a||)**2 ] / SSY_total )

Notes:

- The `(W_{j,a} / ||W_a||)**2` term is the *normalized* squared X-weight for variable `j` on component `a`. With `sklearn.cross_decomposition.PLSRegression`, `x_weights_` columns are already unit-norm (`||W_a||==1`), so the normalization is a no-op for the sklearn case but we keep the explicit `/ ||W_a||**2` for correctness against any other PLS implementation that may pass through this function.
- `sum_j VIP_j**2 / p == 1` is the canonical sanity invariant ("average squared VIP equals 1") — used as a test.

### Why fix it now

VIP is exposed in the GUI's wavelength-importance ranking (`get_feature_importances` -> `compute_vip` for `model_name in ("PLS","PLS-DA")`) and feeds the `importance` variable-selection method. A skewed VIP shifts which wavelengths get selected, which propagates through the entire model search. Roadmap effort estimate: **30 min**.

### References

- Wold, S., Sjöström, M., Eriksson, L. (2001). "PLS-regression: a basic tool of chemometrics." *Chemometrics and Intelligent Laboratory Systems*, **58**(2), 109–130. https://doi.org/10.1016/S0169-7439(01)00155-1
- Mehmood, T., Liland, K. H., Snipen, L., Sæbø, S. (2012). "A review of variable selection methods in Partial Least Squares Regression." *Chemometrics and Intelligent Laboratory Systems*, **118**, 62–69. https://doi.org/10.1016/j.chemolab.2012.07.010 — see Eq. (1) for the VIP formula stated above.

---

## Files touched by this plan

| File | Change |
|---|---|
| `tests/test_vip_formula.py` | **NEW** — parametrized regression test exercising old vs new formula |
| `src/spectral_predict/models.py` | Modify `compute_vip()` body only (lines ~1709-1751); function signature and docstring unchanged except a small docstring update citing Wold (2001) |

No other files require changes. `get_feature_importances` calls `compute_vip` with the same signature; the fix is internal.

---

## Pre-flight: Verify clean baseline

- [ ] **Step 0.1:** Verify on `main` with no unrelated staged changes.
  ```bash
  git status
  git log --oneline -3
  ```
  Expected: on `main`, only `.claude/settings.local.json` may show as modified (ignored). No other unstaged or staged files.

- [ ] **Step 0.2:** Confirm the buggy line is exactly where we think it is.
  ```bash
  sed -n '1735,1740p' src/spectral_predict/models.py
  ```
  Expected output (literal):
  ```
      # Get explained variance by each component
      # SSY: sum of squares of y explained by each component
      y = np.asarray(y).reshape(-1, 1)
      ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)

      # Total SSY
  ```
  If this does not match, **STOP** — the file has shifted. Re-locate `compute_vip` via `grep -n "def compute_vip" src/spectral_predict/models.py` and update line references in subsequent tasks.

---

## Task 1: Write failing regression test

**Goal:** Establish a test that fails on the *current* (buggy) implementation and will pass on the corrected one. The test exercises the numerical difference between the two formulas on a small synthetic PLS problem.

**Files:**

- Create: `tests/test_vip_formula.py`

- [ ] **Step 1.1:** Write the test file with the following exact contents.

  ```python
  """
  Regression test for the canonical VIP formula in compute_vip().

  Bug (pre-T-05): compute_vip() used np.var(y) as a per-component scalar weight
  for all components, so the per-component Y-explained-variance term
  collapsed to a constant times sum(T_a**2). This skewed VIP rankings on any
  PLS problem where Y-explained variance was not proportional to X-score
  energy (i.e. effectively all real PLS problems with > 1 component).

  Canonical formula (Wold 2001; Mehmood et al. 2012, Eq. 1):

      SSY_a = q_a**2 * (T_a.T @ T_a)               # q = y_loadings_
      VIP_j = sqrt( p * sum_a [ SSY_a * (W_{j,a} / ||W_a||)**2 ] / SSY_total )

  This test:
    1. Builds a small synthetic PLS regression problem where the OLD and NEW
       formulas disagree by > 1e-3 on at least one variable (proves the test
       is sensitive to the bug).
    2. Compares compute_vip() output to an independent reference
       implementation of the canonical formula.
    3. Verifies the canonical invariant: sum(VIP**2) / p ~= 1.
  """

  from __future__ import annotations

  import numpy as np
  import pytest
  from sklearn.cross_decomposition import PLSRegression

  from spectral_predict.models import compute_vip


  # ---------- helpers ----------

  def _reference_vip_canonical(pls, X, y):
      """Independent VIP reference using y_loadings_ (Wold 2001)."""
      W = np.asarray(pls.x_weights_)        # (p, A)
      T = np.asarray(pls.x_scores_)         # (n, A)
      Q = np.asarray(pls.y_loadings_)       # (n_targets, A) for sklearn >=1.1
      # univariate Y -> q is shape (1, A); squeeze to (A,)
      q = Q.ravel() if Q.ndim == 1 else Q[0, :]
      p = W.shape[0]

      # Per-component Y SS
      ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)   # (A,)
      ssy_total = ssy_comp.sum()

      # Normalized squared weights (sklearn x_weights_ cols are unit-norm,
      # but normalize defensively for any other caller).
      w_norm_sq = (W ** 2) / (np.sum(W ** 2, axis=0, keepdims=True) + 1e-300)

      vip = np.sqrt(p * (w_norm_sq @ ssy_comp) / (ssy_total + 1e-300))
      return vip


  def _reference_vip_OLD_BUGGY(pls, X, y):
      """The pre-T-05 formula. Used only to assert the test would catch it."""
      W = np.asarray(pls.x_weights_)
      T = np.asarray(pls.x_scores_)
      y_arr = np.asarray(y).reshape(-1, 1)
      ssy_comp = np.sum(T ** 2, axis=0) * np.var(y_arr, axis=0)
      ssy_total = np.sum(ssy_comp)
      p = W.shape[0]
      weight = np.sum((W ** 2) * ssy_comp, axis=1)
      return np.sqrt(p * weight / ssy_total)


  # ---------- fixtures ----------

  @pytest.fixture
  def synthetic_pls_problem():
      """
      Two latent factors with very different Y-loading magnitudes:
      - factor 1 contributes strongly to Y (q1 = 5.0)
      - factor 2 contributes weakly to Y (q2 = 0.5) but has comparable X energy
      The old (constant np.var(y)) formula will badly mis-weight factor 2.
      """
      rng = np.random.default_rng(0)
      n_samples = 60
      p = 20

      # Two orthogonal latent score vectors with similar X-energy
      t1 = rng.standard_normal(n_samples)
      t2 = rng.standard_normal(n_samples)
      # Two distinct loading patterns (peaks in different variables)
      w1 = np.zeros(p); w1[2:6] = 1.0
      w2 = np.zeros(p); w2[12:16] = 1.0
      X = np.outer(t1, w1) + np.outer(t2, w2) + 0.05 * rng.standard_normal((n_samples, p))

      # Y depends ~10x more on factor 1 than factor 2
      y = 5.0 * t1 + 0.5 * t2 + 0.05 * rng.standard_normal(n_samples)
      return X, y


  @pytest.fixture
  def fitted_pls(synthetic_pls_problem):
      X, y = synthetic_pls_problem
      pls = PLSRegression(n_components=2, scale=False)
      pls.fit(X, y)
      return pls, X, y


  # ---------- tests ----------

  class TestVIPCanonicalFormula:
      def test_old_and_new_formulas_disagree_on_this_problem(self, fitted_pls):
          """
          Sanity check that this synthetic problem is sensitive: the old and
          new formulas must produce materially different VIPs, otherwise the
          regression test below would be vacuous.
          """
          pls, X, y = fitted_pls
          old = _reference_vip_OLD_BUGGY(pls, X, y)
          new = _reference_vip_canonical(pls, X, y)
          max_abs_diff = float(np.max(np.abs(old - new)))
          assert max_abs_diff > 1e-3, (
              "Synthetic problem is not sensitive enough — old and new VIP "
              f"agree to {max_abs_diff:.2e}. Adjust the fixture."
          )

      def test_compute_vip_matches_canonical_reference(self, fitted_pls):
          """compute_vip() output equals an independent canonical implementation."""
          pls, X, y = fitted_pls
          got = compute_vip(pls, X, y)
          want = _reference_vip_canonical(pls, X, y)
          np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

      def test_compute_vip_does_not_match_old_buggy_formula(self, fitted_pls):
          """compute_vip() must NOT reproduce the pre-T-05 buggy output."""
          pls, X, y = fitted_pls
          got = compute_vip(pls, X, y)
          old = _reference_vip_OLD_BUGGY(pls, X, y)
          max_abs_diff = float(np.max(np.abs(got - old)))
          assert max_abs_diff > 1e-3, (
              "compute_vip() still matches the old (buggy) formula — fix not applied."
          )

      def test_canonical_invariant_average_squared_vip_is_one(self, fitted_pls):
          """sum(VIP**2) / p == 1 is the textbook invariant (Wold 2001)."""
          pls, X, y = fitted_pls
          vip = compute_vip(pls, X, y)
          p = pls.x_weights_.shape[0]
          mean_sq = float(np.sum(vip ** 2) / p)
          assert mean_sq == pytest.approx(1.0, rel=1e-6, abs=1e-8)

      def test_output_shape_and_nonnegativity(self, fitted_pls):
          pls, X, y = fitted_pls
          vip = compute_vip(pls, X, y)
          assert vip.shape == (pls.x_weights_.shape[0],)
          assert np.all(vip >= 0)
          assert np.all(np.isfinite(vip))
  ```

- [ ] **Step 1.2:** Run the test against the **current (buggy)** implementation. Confirm the failures we expect.
  ```bash
  pytest tests/test_vip_formula.py -v
  ```
  Expected outcome:
  - `test_old_and_new_formulas_disagree_on_this_problem` — **PASS** (old vs new differ; this is a fixture sanity check, independent of `compute_vip`).
  - `test_compute_vip_matches_canonical_reference` — **FAIL** with an `AssertionError: Not equal to tolerance ...` message because `compute_vip` returns the buggy result.
  - `test_compute_vip_does_not_match_old_buggy_formula` — **FAIL** with `compute_vip() still matches the old (buggy) formula — fix not applied.`.
  - `test_canonical_invariant_average_squared_vip_is_one` — **FAIL** (the buggy formula does not satisfy this invariant).
  - `test_output_shape_and_nonnegativity` — **PASS** (shape and non-negativity are unaffected by the bug).
  
  Net: **3 failed, 2 passed**. If you see anything else (e.g. import error), fix that first before continuing.

- [ ] **Step 1.3:** Commit the failing test.
  ```bash
  git add tests/test_vip_formula.py
  git commit -m "test: add failing regression test for canonical VIP formula (T-05)"
  ```

---

## Task 2: Implement the canonical VIP formula

**Goal:** Replace the body of `compute_vip()` with the canonical Wold (2001) formula. Keep the function signature, name, and contract identical. Update the docstring with the citation.

**Files:**

- Modify: `src/spectral_predict/models.py` (function `compute_vip`, currently lines 1709-1751)

- [ ] **Step 2.1:** Replace the entire `compute_vip` function (from `def compute_vip(...)` through the final `return vip_scores`) with the version below. Use `Edit` to swap the exact block.

  **`old_string` (current buggy implementation):**
  ```python
  def compute_vip(pls_model, X, y):
      """
      Compute Variable Importance in Projection (VIP) scores for a fitted PLS model.

      Parameters
      ----------
      pls_model : PLSRegression or PLSTransformer
          Fitted PLS model
      X : array-like
          Training data
      y : array-like
          Target values

      Returns
      -------
      vip_scores : ndarray
          VIP score for each variable
      """
      # Handle PLSTransformer wrapper
      if isinstance(pls_model, PLSTransformer):
          pls_model = pls_model.pls_

      # Get PLS components
      W = pls_model.x_weights_  # (n_features, n_components)
      T = pls_model.x_scores_  # (n_samples, n_components)

      # Get explained variance by each component
      # SSY: sum of squares of y explained by each component
      y = np.asarray(y).reshape(-1, 1)
      ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)

      # Total SSY
      ssy_total = np.sum(ssy_comp)

      # VIP calculation (vectorized for performance)
      n_features = W.shape[0]
      n_components = W.shape[1]

      # Vectorized version: same math, but uses broadcasting instead of loop
      weight = np.sum((W ** 2) * ssy_comp, axis=1)  # Sum over components for each feature
      vip_scores = np.sqrt(n_features * weight / ssy_total)

      return vip_scores
  ```

  **`new_string` (canonical implementation):**
  ```python
  def compute_vip(pls_model, X, y):
      """
      Compute Variable Importance in Projection (VIP) scores for a fitted PLS model.

      Uses the canonical formula from Wold et al. (2001), as restated in
      Mehmood et al. (2012, Eq. 1):

          SSY_a   = q_a**2 * (T_a.T @ T_a)
          VIP_j   = sqrt( p * sum_a [ SSY_a * (W_{j,a} / ||W_a||)**2 ] / SSY_total )

      where ``W = x_weights_``, ``T = x_scores_``, ``q = y_loadings_[:, 0]``
      (univariate Y), and ``p = n_features``. The invariant
      ``mean(VIP**2) == 1`` follows from this definition.

      Parameters
      ----------
      pls_model : PLSRegression or PLSTransformer
          Fitted PLS model exposing ``x_weights_``, ``x_scores_``,
          ``y_loadings_``.
      X : array-like
          Training data. Currently unused (kept for API compatibility); VIP
          is fully determined by the fitted model's weights/scores/loadings.
      y : array-like
          Target values. Currently unused (kept for API compatibility); the
          per-component Y variance enters via ``y_loadings_``, not ``y``.

      Returns
      -------
      vip_scores : ndarray of shape (n_features,)
          VIP score for each variable.

      References
      ----------
      Wold, S., Sjöström, M., Eriksson, L. (2001). PLS-regression: a basic
      tool of chemometrics. *Chemometrics and Intelligent Laboratory
      Systems*, 58(2), 109-130.

      Mehmood, T., Liland, K. H., Snipen, L., Sæbø, S. (2012). A review of
      variable selection methods in Partial Least Squares Regression.
      *Chemometrics and Intelligent Laboratory Systems*, 118, 62-69.
      """
      # Handle PLSTransformer wrapper
      if isinstance(pls_model, PLSTransformer):
          pls_model = pls_model.pls_

      # Get PLS components
      W = np.asarray(pls_model.x_weights_)   # (n_features, n_components)
      T = np.asarray(pls_model.x_scores_)    # (n_samples, n_components)
      Q = np.asarray(pls_model.y_loadings_)  # (n_targets, n_components) for sklearn

      # Univariate Y: collapse Q to shape (n_components,)
      if Q.ndim == 1:
          q = Q
      else:
          q = Q[0, :]

      n_features = W.shape[0]

      # Per-component Y sum-of-squares: SSY_a = q_a**2 * sum_n T[n,a]**2
      ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)
      ssy_total = float(np.sum(ssy_comp))

      # Guard against degenerate fits (zero Y variance captured)
      if ssy_total <= 0.0:
          return np.zeros(n_features, dtype=float)

      # Normalized squared X-weights: (W_{j,a} / ||W_a||)**2
      # sklearn's PLSRegression already returns unit-norm columns of W, but
      # normalize defensively so the function is correct for any caller.
      col_norm_sq = np.sum(W ** 2, axis=0)
      col_norm_sq = np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)
      w_norm_sq = (W ** 2) / col_norm_sq  # broadcasts (p, A) / (A,) -> (p, A)

      # VIP_j = sqrt( p * sum_a [ SSY_a * w_norm_sq[j, a] ] / SSY_total )
      vip_scores = np.sqrt(n_features * (w_norm_sq @ ssy_comp) / ssy_total)

      return vip_scores
  ```

- [ ] **Step 2.2:** Verify the file still imports cleanly.
  ```bash
  python -c "from spectral_predict.models import compute_vip; print('ok')"
  ```
  Expected: `ok`. Any `ImportError`/`SyntaxError` means the edit landed wrong — re-read the function and fix.

- [ ] **Step 2.3:** Run the new VIP test file. All five tests should pass.
  ```bash
  pytest tests/test_vip_formula.py -v
  ```
  Expected: **5 passed**. Specifically:
  - `test_old_and_new_formulas_disagree_on_this_problem` — PASS
  - `test_compute_vip_matches_canonical_reference` — PASS
  - `test_compute_vip_does_not_match_old_buggy_formula` — PASS
  - `test_canonical_invariant_average_squared_vip_is_one` — PASS
  - `test_output_shape_and_nonnegativity` — PASS

- [ ] **Step 2.4:** Run the existing PLS-DA importance tests to confirm no collateral damage. Those tests assert that `get_feature_importances` for a PLS-DA pipeline matches `compute_vip` on the underlying PLS step — both sides change together, so they should still pass.
  ```bash
  pytest tests/test_plsda_importance.py -v
  ```
  Expected: **5 passed**. (`test_importance_from_pipeline`, `test_importance_matches_pls_component`, `test_importance_not_from_lr`, `test_importance_from_bare_pls_transformer`, `test_pipeline_with_scaler_model_steps`.)

  If `test_importance_matches_pls_component` fails, that means there are two separate VIP code paths and we missed one. Search:
  ```bash
  grep -n "x_weights_\|y_loadings_" src/spectral_predict/models.py
  ```
  and reconcile.

- [ ] **Step 2.5:** Commit the implementation.
  ```bash
  git add src/spectral_predict/models.py
  git commit -m "fix(T-05): use canonical Wold (2001) VIP formula in compute_vip

  Pre-fix, compute_vip() weighted every PLS component by a single
  scalar np.var(y), so per-component Y-explained variance collapsed
  to sum(T_a**2) * const. This skewed VIP rankings on any problem
  with > 1 component, propagating into the GUI's wavelength
  importance display and the 'importance' variable-selection method.

  Replace with SSY_a = q_a**2 * (T_a.T @ T_a) using y_loadings_, per
  Wold et al. (2001) and Mehmood et al. (2012, Eq. 1). Verified by
  tests/test_vip_formula.py: matches an independent canonical
  reference, fails the old-formula match, and satisfies the textbook
  mean(VIP**2) == 1 invariant. Existing PLS-DA importance tests
  still pass."
  ```

---

## Task 3: Broader regression check

**Goal:** Confirm no other test in the suite relied on the old buggy VIP numbers. We expect zero regressions because the only consumer is `get_feature_importances` -> ranking, and no ranking-stability test pins exact VIP magnitudes from the old formula.

- [ ] **Step 3.1:** Run any tests likely to touch VIP.
  ```bash
  pytest tests/ -v -k "vip or importance or pls" --no-header
  ```
  Expected: all collected tests pass. If a test fails because it pinned VIP values from the buggy formula, that test is itself wrong — flag it in the commit message and update the pinned values to match the canonical formula. Do **not** revert the implementation.

- [ ] **Step 3.2:** Optional fast smoke (skip if it takes > 2 min):
  ```bash
  pytest tests/test_plsda_importance.py tests/test_vip_formula.py -v
  ```
  Expected: **10 passed** (5 + 5).

- [ ] **Step 3.3:** If Step 3.1 surfaced any test that needed updating, commit those updates separately:
  ```bash
  git add tests/<the_file>.py
  git commit -m "test: update VIP-magnitude expectations after T-05 canonical-formula fix"
  ```
  If nothing needed updating, skip this step.

---

## Task 4: Living-doc updates

**Files:**

- Modify: `docs/PROJECT_STATUS.md` — add a one-liner under "Recently resolved" or equivalent.
- Modify: `docs/SESSION_LOG.md` — append a 2026-04-29 entry.

- [ ] **Step 4.1:** Append to `docs/SESSION_LOG.md` (under a `## 2026-04-29` heading, creating it if absent):
  ```markdown
  ### T-05: VIP formula fix
  Replaced `np.var(y)` per-component weight with canonical
  `q_a**2 * sum(T_a**2)` (Wold 2001, Mehmood et al. 2012 Eq. 1) in
  `src/spectral_predict/models.py:compute_vip`. Old formula collapsed all
  components to the same Y-weighting scalar, skewing VIP rankings whenever
  components had similar X-score energy but different Y-loading. New tests
  in `tests/test_vip_formula.py` lock the formula in. Existing PLS-DA
  importance tests pass unchanged. See `docs/plans/2026-04-29-T05-vip-formula-fix.md`.
  ```

- [ ] **Step 4.2:** Edit `docs/PROJECT_STATUS.md` — under the relevant "What works" / "Recently resolved" section (look for the most recent dated entry and add below it):
  ```markdown
  - **2026-04-29:** T-05 VIP formula corrected to canonical Wold (2001) form
    (`src/spectral_predict/models.py:compute_vip`). Wavelength rankings via VIP
    are no longer skewed by the per-component Y-variance term collapsing to a
    constant. Test: `tests/test_vip_formula.py`.
  ```
  If the file has no obvious slot, place it under the section heading that lists recent fixes; do not invent new top-level sections.

- [ ] **Step 4.3:** Commit doc updates.
  ```bash
  git add docs/PROJECT_STATUS.md docs/SESSION_LOG.md
  git commit -m "docs: log T-05 VIP formula fix in SESSION_LOG and PROJECT_STATUS"
  ```

---

## Non-goals / explicitly out of scope

- Changing the `compute_vip` signature (still takes `pls_model, X, y`). `X` and `y` become unused — documented in the docstring; not removed to keep the call sites in `get_feature_importances` and any test code unchanged.
- Multivariate-Y VIP. The canonical formula generalizes (sum over Y columns of `q_{k,a}**2`); this fix preserves the existing univariate-Y assumption (`Q[0, :]`) because every PLS call site in `spectral_predict/` passes a single response. If multi-Y support is added later, revisit.
- Re-running the full nightly grid search. VIP feeds the `importance` variable-selection path; downstream rankings *will* shift on real data, but that is the intended fix, not a regression. No baseline metrics are pinned anywhere in the suite that we need to update.
- Re-deriving or sanity-checking the VIP formula for non-sklearn PLS implementations beyond the defensive `||W_a||` normalization already added.

---

## Open questions for reviewer

1. Are there any other call sites that consume `compute_vip` output and pin numerical magnitudes (e.g. golden-standard fixtures, frozen result CSVs)? `tests/gold_standards/` was not exhaustively scanned — if any of those embed VIP numbers from the old formula, they will need refreshed values.
2. Should `compute_vip` drop the unused `X`, `y` parameters (signature change) or keep them for backward compatibility? This plan keeps them for compatibility; a follow-up can deprecate.
3. The plan does not run on the full `tests/` suite, only the targeted subset. Worth a full `pytest tests/ -x` before merging? (Probably yes, but it's outside this 30-min ticket's effort budget.)
