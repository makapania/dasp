# T-10: Clamp PLS Component Grid by Training Fold Sample Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Prevent silent fold failures when PLS n_components grid contains values larger than the per-fold training sample count.

**Architecture:** CV-strategy-aware clamp at grid-construction time inside `run_search` and `run_bayesian_search` (single source of truth: a new pure helper `compute_min_train_fold_size(cv_strategy, n_samples, folds)` in `cv_utils.py`). Replace the existing K-fold-only formula `n_samples * (folds - 1) // folds` (search.py:1109 and search.py:3312) with the helper, which knows that LOO has train-fold size n-1 and that K-fold/RepeatedKFold use the conservative floor `n_samples * (folds - 1) // folds`. The clamp is applied to `safe_max_components` BEFORE `get_model_grids()` is called, so `models.py:842-843` continues to receive an already-clamped `max_n_components`. `models.py` itself is left as a defense-in-depth guard with a doc-comment update; the authoritative correction lives at the call sites where CV strategy is known.

**Tech Stack:** numpy, scikit-learn (PLSRegression), pytest.

**Source:** roadmap T-10.

---

## Background — what is broken today

- `src/spectral_predict/search.py:1109` and `src/spectral_predict/search.py:3312` both compute `min_train_samples = n_samples * (folds - 1) // folds`. This is correct for `cv_strategy='kfold'` and `cv_strategy='repeated_kfold'` (RepeatedKFold uses the same partition geometry as KFold; only the number of repeats varies). It is **incorrect for `cv_strategy='loo'`** where train-fold size is `n_samples - 1`, not `n_samples * 4 / 5`.
- The LOO miscalculation is conservative (under-estimates train size, never over-estimates) so it does not cause silent failures by itself. **However:**
- The clamp is applied to `safe_max_components`, which is forwarded to `get_model_grids(..., max_n_components=safe_max_components, ...)`. The grid in `models.py:842-843` is then `min(n_features, max_n_components)` — i.e. `models.py` does NOT independently clamp by `n_samples_train_per_fold`. If any caller of `get_model_grids` ever passed an unclamped `max_n_components` (e.g. a future entry point that forgets the clamp, a test, a refactor), `PLSRegression` would silently fail or warn at fit time inside CV folds. sklearn's `PLSRegression.fit` raises `ValueError` only when `n_components > min(X.shape)`; for a 20-sample dataset with k=5 → train fold = 16, asking for n_components=18 fits but produces unstable scores.
- The roadmap fix says "min(n_features, n_samples_train_per_fold, max_n_components)". The decision below is whether to put that defense in `models.py` (grid construction) or at the call site (where CV strategy is known).

---

## Design decision — clamp at the call site, defense-in-depth in models.py

**Chosen:** clamp at the call site (`search.py`). Reasons:

1. **CV strategy lives at the call site, not in `models.py`.** `get_model_grids` does not currently take a `cv_strategy` argument and adding one would ripple through ~12 call sites that all pass a single `safe_max_components` integer today. The simplest correct fix is to compute the right `min_train_samples` at the call site (where `cv_strategy` is already in scope) and keep `models.py` as the integer-receiver.
2. **All four current entry points to `get_model_grids` already do clamping.** `run_search`, `run_bayesian_search` (and the Bayesian search forwards via `models.py` line 343-345). The two production-relevant entry points have the `min_train_samples` computation right above the `get_model_grids` call. The fix is local: replace the formula.
3. **Per-fold clamping inside the model is a separate failure mode.** Even with the grid clamped correctly, a per-fold fit with an unusual splitter (e.g. GroupKFold with one giant group) might still receive too few train samples. That is out of scope for T-10 — silently clamping `n_components` inside a per-fold fit hides the user's intent. The right behavior under that scenario is the existing sklearn ValueError, surfaced.
4. **`models.py` defense-in-depth is cheap and worth adding.** Update the docstring of `get_model_grids` to state that `max_n_components` MUST already be clamped by `n_samples_train_per_fold`, and add an `assert max_n_components >= 1` guard. Do NOT silently re-clamp inside `models.py` because that would mask a caller that forgot the clamp.

**Rejected alternatives:**

- *Per-fold silent clamp inside a model wrapper.* Hides the bug. A user asking for `n_components=15` on a dataset where some fold has 14 train rows wants to know that, not get a silently-different model.
- *Pass `cv_strategy` and `cv_n_repeats` into `get_model_grids`.* Larger refactor, ripples through every test that constructs grids in isolation, no real win — the call site already has the info.
- *Compute the clamp inside `models.py:840-843` from `n_samples` (not `n_samples_train_per_fold`).* Over-permissive: would let `n_components = n_samples` through, which fails inside K-fold every time.

---

## Where the new helper lives

`src/spectral_predict/cv_utils.py` — new pure function:

```python
def compute_min_train_fold_size(
    cv_strategy: str,
    n_samples: int,
    n_folds: int,
) -> int:
    """Conservative lower bound on the smallest training-fold size.

    PLS regression requires n_components <= min(n_features, n_samples_train_fold).
    The grid for `n_components` must be clamped using a value that is no greater
    than the smallest training-fold size any CV split will produce, otherwise
    sklearn raises silently inside the fold or returns NaN metrics that are
    swallowed by the search aggregator.

    Strategy semantics:
      - 'kfold': train fold size = n_samples - ceil(n_samples / n_folds).
                 Conservative floor used here: n_samples * (n_folds - 1) // n_folds.
                 This matches what shuffled KFold actually produces for evenly-
                 divisible n; for n not evenly divisible by n_folds, the formula
                 under-counts by at most 1 (safe in the conservative direction).
      - 'repeated_kfold': identical to kfold (RepeatedKFold reuses KFold
                          partitions across repeats; per-fold geometry is the
                          same). n_repeats does NOT affect train-fold size.
      - 'loo': train fold size = n_samples - 1.

    Group splitters (GroupKFold, LeaveOneGroupOut) are NOT covered here and
    will raise NotImplementedError; T-15 will plumb group-aware sizing through
    a separate path.

    Parameters
    ----------
    cv_strategy : str
        One of 'kfold', 'repeated_kfold', 'loo'.
    n_samples : int
        Total samples in the calibration set.
    n_folds : int
        Number of folds. Ignored when cv_strategy == 'loo'.

    Returns
    -------
    int
        Conservative lower bound on the smallest training-fold size, >= 1.

    Raises
    ------
    ValueError
        If n_samples < 2 or n_folds < 2 (kfold/repeated_kfold).
    NotImplementedError
        If cv_strategy is 'group_kfold' or 'leave_one_group_out' (T-15 scope).
    """
    if n_samples < 2:
        raise ValueError(
            f"PLS clamp requires n_samples >= 2 (got {n_samples})."
        )
    if cv_strategy == 'loo':
        return n_samples - 1
    if cv_strategy in ('kfold', 'repeated_kfold'):
        if n_folds < 2:
            raise ValueError(
                f"K-fold CV requires n_folds >= 2 (got {n_folds})."
            )
        return max(1, n_samples * (n_folds - 1) // n_folds)
    if cv_strategy in ('group_kfold', 'leave_one_group_out'):
        raise NotImplementedError(
            f"compute_min_train_fold_size: {cv_strategy!r} not supported yet "
            "(T-15 will add group-aware sizing)."
        )
    raise ValueError(
        f"Unknown cv_strategy: {cv_strategy!r}. "
        "Expected 'kfold', 'repeated_kfold', or 'loo'."
    )
```

---

## Tasks

Tasks are sized so each one corresponds to a single Edit + a single test run + a single commit. GLM 5.1 should pause after every task and verify the test output matches the expected line before moving on.

### Task 1: Add the helper + tests (TDD red→green)

**Files:**
- Create: `tests/test_cv_pls_clamp.py`
- Modify: `src/spectral_predict/cv_utils.py`

**Step 1 (RED).** Write tests first. Create `tests/test_cv_pls_clamp.py`:

```python
"""Tests for PLS-component clamping by training-fold size (T-10).

Covers:
- compute_min_train_fold_size for kfold / repeated_kfold / loo
- Edge cases (n_samples=2, n_folds=2, n_folds=n_samples)
- Group-strategy NotImplementedError
- Unknown strategy ValueError
"""
from __future__ import annotations

import pytest

from spectral_predict.cv_utils import compute_min_train_fold_size


class TestComputeMinTrainFoldSize:
    """Pure-function tests for the new helper."""

    def test_kfold_n10_k5_returns_8(self):
        # 10 samples, 5 folds → test fold = 2, train fold = 8
        assert compute_min_train_fold_size('kfold', 10, 5) == 8

    def test_kfold_n20_k5_returns_16(self):
        assert compute_min_train_fold_size('kfold', 20, 5) == 16

    def test_kfold_n100_k5_returns_80(self):
        assert compute_min_train_fold_size('kfold', 100, 5) == 80

    def test_kfold_n9_k5_returns_7_conservative(self):
        # 9 // 5 * 4 = 7 (conservative; actual smallest train fold is 7)
        assert compute_min_train_fold_size('kfold', 9, 5) == 7

    def test_kfold_n7_k3_returns_4(self):
        # 7 * 2 // 3 = 4 (conservative; actual smallest train fold is 4 or 5
        # depending on how sklearn distributes the remainder)
        assert compute_min_train_fold_size('kfold', 7, 3) == 4

    def test_repeated_kfold_matches_kfold(self):
        # RepeatedKFold reuses KFold geometry; n_repeats does not change train size.
        assert (
            compute_min_train_fold_size('repeated_kfold', 20, 5)
            == compute_min_train_fold_size('kfold', 20, 5)
        )

    def test_loo_n20_returns_19(self):
        # LOO: train fold size = n - 1 regardless of n_folds.
        assert compute_min_train_fold_size('loo', 20, 5) == 19

    def test_loo_n10_returns_9(self):
        assert compute_min_train_fold_size('loo', 10, 5) == 9

    def test_loo_ignores_n_folds(self):
        # n_folds is meaningless for LOO; helper should not error if it's 0/None-equivalent.
        assert compute_min_train_fold_size('loo', 50, 99) == 49
        # n_folds=0 would otherwise cause a divide-by-zero — confirm LOO short-circuits.
        assert compute_min_train_fold_size('loo', 50, 0) == 49

    def test_kfold_minimum_n2_k2(self):
        # Smallest valid case: n=2, k=2 → train fold = 1.
        assert compute_min_train_fold_size('kfold', 2, 2) == 1

    def test_kfold_n_samples_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_samples >= 2"):
            compute_min_train_fold_size('kfold', 1, 5)

    def test_kfold_n_folds_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_folds >= 2"):
            compute_min_train_fold_size('kfold', 20, 1)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown cv_strategy"):
            compute_min_train_fold_size('bogus', 20, 5)

    def test_group_strategies_not_implemented(self):
        with pytest.raises(NotImplementedError, match="T-15"):
            compute_min_train_fold_size('group_kfold', 20, 5)
        with pytest.raises(NotImplementedError, match="T-15"):
            compute_min_train_fold_size('leave_one_group_out', 20, 5)
```

**Step 2.** Run the tests — they should fail with `ImportError: cannot import name 'compute_min_train_fold_size'`.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_cv_pls_clamp.py -x -v
```

Expected: `ImportError` or `AttributeError`, all 14 tests collected red.

**Step 3 (GREEN).** Add `compute_min_train_fold_size` to `src/spectral_predict/cv_utils.py`. Insert the function between `validate_cv_strategy_for_task` (ends ~line 150) and `estimate_total_cv_fits` (starts ~line 153). Use the docstring + body shown in the "Where the new helper lives" section above verbatim.

**Step 4.** Re-run the tests.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_cv_pls_clamp.py -x -v
```

Expected: 14/14 pass.

**Step 5.** Commit.

```bash
git add src/spectral_predict/cv_utils.py tests/test_cv_pls_clamp.py
git commit -m "test+feat: compute_min_train_fold_size helper for PLS clamp (T-10)

Pure-function helper that returns a conservative lower bound on the
smallest training-fold size for kfold / repeated_kfold / loo. Future
T-15 work will plumb group-aware sizing through this function; for
now group strategies raise NotImplementedError.

14 tests cover normal cases, edge cases (n=2, k=2; n=k; LOO ignoring
n_folds), and error paths."
```

---

### Task 2: Add a grid-construction integration test (RED)

**Files:**
- Modify: `tests/test_cv_pls_clamp.py` (append a new TestClass)

**Step 1.** Append the following test class to `tests/test_cv_pls_clamp.py`:

```python
class TestRunSearchPLSGridClamping:
    """Black-box: run_search must clamp the PLS grid for small datasets.

    These tests do NOT exercise the full pipeline — they call run_search with
    a single PLS model and inspect the result DataFrame to confirm that no
    grid row used n_components > min_train_fold_size.

    Uses synthetic regression data sized so the bug would show up if the
    clamp is missing. Keeps run_search invocations small (3 folds, no
    variable subsets) so each test runs in <30 s.
    """

    @pytest.fixture
    def tiny_regression_data(self):
        """N=10 samples, 50 features. K=5 → train fold = 8."""
        import numpy as np
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 50))
        # y as linear combo of first 3 features + small noise
        y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.05 * rng.standard_normal(10)
        return X, y

    @pytest.fixture
    def normal_regression_data(self):
        """N=80 samples, 50 features. K=5 → train fold = 64. Grid should NOT be clamped by samples."""
        import numpy as np
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 50))
        y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.05 * rng.standard_normal(80)
        return X, y

    def test_n10_kfold_clamps_to_8_components(self, tiny_regression_data):
        """N=10, k=5 → max grid n_components must be 8, NOT 20 (max_n_components arg)."""
        from spectral_predict.search import run_search
        X, y = tiny_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='kfold',
            max_n_components=20,  # deliberately too high
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        # Grid should have produced rows with n_components in {1, ..., 8}.
        # The 'Params' column or 'Model_Params' / 'LVs' column carries the n_components value.
        # Identify the column name (search.py writes 'LVs' for PLS in some result schemas
        # and 'Params' as a JSON string in others — check both).
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 8, (
            f"PLS grid for N=10 k=5 produced n_components={max(n_components_seen)}, "
            f"expected max 8. Clamp is broken. Seen: {sorted(n_components_seen)}"
        )
        # Sanity: at least n_components=1 should be present (the floor).
        assert 1 in n_components_seen

    def test_n10_loo_clamps_to_9_components(self, tiny_regression_data):
        """N=10, LOO → max grid n_components must be 9 (n-1), NOT 8 (the K-fold floor)."""
        from spectral_predict.search import run_search
        X, y = tiny_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,  # ignored under LOO
            cv_strategy='loo',
            max_n_components=20,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 9, (
            f"PLS grid for N=10 LOO produced n_components={max(n_components_seen)}, "
            f"expected max 9 (n-1). Seen: {sorted(n_components_seen)}"
        )
        # The fix should produce 9 (LOO bound), strictly more than 8 (K-fold bound).
        # If the value is still 8, the LOO branch wasn't applied.
        assert max(n_components_seen) == 9, (
            f"LOO clamp expected n_components_max == 9, got {max(n_components_seen)}. "
            "If this is 8, the fix is using the K-fold formula instead of n-1."
        )

    def test_n80_kfold_uses_full_grid_default_max(self, normal_regression_data):
        """N=80, k=5, max_n_components=10 (default) → all 10 components present.

        Confirms the clamp does NOT artificially shrink grids on larger datasets:
        train fold = 64 >> 10, so the bind is max_n_components, not n_samples.
        """
        from spectral_predict.search import run_search
        X, y = normal_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='kfold',
            max_n_components=10,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) == 10, (
            f"PLS grid for N=80 k=5 with max_n_components=10 should reach 10, "
            f"got {max(n_components_seen)}. Clamp may be over-aggressive."
        )

    def test_n10_repeated_kfold_matches_kfold(self, tiny_regression_data):
        """RepeatedKFold should produce the same n_components ceiling as KFold."""
        from spectral_predict.search import run_search
        X, y = tiny_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='repeated_kfold',
            cv_n_repeats=2,
            max_n_components=20,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 8, (
            f"RepeatedKFold N=10 k=5 produced max n_components={max(n_components_seen)}, "
            f"expected 8 (same as kfold). Seen: {sorted(n_components_seen)}"
        )


def _extract_n_components_seen(df) -> set[int]:
    """Pull the unique set of n_components values from a result DataFrame.

    Handles two known result-row schemas:
    1. Direct column 'LVs' (or 'n_components') containing an int per row.
    2. 'Params' or 'Model_Params' column containing a JSON string with
       {"n_components": N, ...}.
    """
    import json
    seen: set[int] = set()
    if df is None or len(df) == 0:
        return seen

    # Schema 1: direct integer column
    for col in ('LVs', 'n_components', 'NumComponents'):
        if col in df.columns:
            for v in df[col].dropna().tolist():
                try:
                    seen.add(int(v))
                except (TypeError, ValueError):
                    continue
            if seen:
                return seen

    # Schema 2: JSON string in Params / Model_Params
    for col in ('Params', 'Model_Params'):
        if col in df.columns:
            for v in df[col].dropna().tolist():
                if isinstance(v, dict):
                    if 'n_components' in v:
                        seen.add(int(v['n_components']))
                elif isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(parsed, dict) and 'n_components' in parsed:
                        seen.add(int(parsed['n_components']))
            if seen:
                return seen

    return seen
```

**Step 2.** Run the new test class.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_cv_pls_clamp.py::TestRunSearchPLSGridClamping -x -v
```

Expected behavior:
- `test_n10_kfold_clamps_to_8_components`: should PASS with the existing K-fold formula (search.py:1109 already produces 8 for n=10 k=5). This is a baseline guard.
- `test_n10_loo_clamps_to_9_components`: should **FAIL** today, because the existing formula at search.py:1109 produces `10 * 4 // 5 = 8`, not 9. Failure message will be "expected n_components_max == 9, got 8".
- `test_n80_kfold_uses_full_grid_default_max`: should PASS today.
- `test_n10_repeated_kfold_matches_kfold`: should PASS today (RepeatedKFold geometry = KFold geometry).

The one failing test is the RED that drives the fix in Task 3.

**Step 3.** Do NOT commit yet. The test will be committed alongside the fix in Task 3.

---

### Task 3: Replace the K-fold-only formula in `run_search`

**Files:**
- Modify: `src/spectral_predict/search.py`

**Step 1.** Open `src/spectral_predict/search.py`. Find the block at lines 1104–1128:

```python
    # Adjust max_n_components based on CV training fold size
    # For REGRESSION: PLS requires n_components <= min(n_features, n_samples_in_training_fold)
    # For CLASSIFICATION: PLS-DA uses PLS as dimensionality reduction before LR classifier,
    #                     so we can be less strict (LR can handle more components than samples)
    # Use TRAINING fold size (not test fold) since PLS is fit on training data
    min_train_samples = n_samples * (folds - 1) // folds

    if task_type == "regression":
        # Strict constraint for PLS regression: n_components <= min(n_samples_train, n_features)
        safe_max_components = min(max_n_components, min_train_samples, n_features)
    else:
        # More relaxed constraint for PLS-DA classification
        # PLS transforms to latent space, then LR classifies
        # Allow more components since LR can handle high-dimensional input
        safe_max_components = min(max_n_components, n_features)
        # Still warn if components exceed training fold size (not recommended but allowed)
        if max_n_components > min_train_samples:
            print(f"Note: Using {max_n_components} PLS components with min_train_size~{min_train_samples}. " +
                  f"This is acceptable for PLS-DA (classification) but may cause instability.")

    if safe_max_components < max_n_components:
        print(f"Note: Reducing max components from {max_n_components} to {safe_max_components} " +
              f"due to dataset constraints (n_samples={n_samples}, n_features={n_features}, " +
              f"min_train_size~{min_train_samples}, task={task_type})")
```

**Step 2.** Replace the `min_train_samples = ...` line with a call to the new helper. The replacement uses an Edit with enough context to be unique:

Find:
```python
    # Adjust max_n_components based on CV training fold size
    # For REGRESSION: PLS requires n_components <= min(n_features, n_samples_in_training_fold)
    # For CLASSIFICATION: PLS-DA uses PLS as dimensionality reduction before LR classifier,
    #                     so we can be less strict (LR can handle more components than samples)
    # Use TRAINING fold size (not test fold) since PLS is fit on training data
    min_train_samples = n_samples * (folds - 1) // folds
```

Replace with:
```python
    # Adjust max_n_components based on CV training fold size.
    # T-10: cv_strategy-aware. K-fold/RepeatedKFold use the conservative
    # floor `n_samples * (folds - 1) // folds`. LOO has train-fold size n-1.
    # Group splitters are not yet supported (T-15 follow-up).
    # For REGRESSION: PLS requires n_components <= min(n_features, n_samples_train_fold)
    # For CLASSIFICATION: PLS-DA uses PLS as dimensionality reduction before LR classifier,
    #                     so we can be less strict (LR can handle more components than samples)
    from .cv_utils import compute_min_train_fold_size
    min_train_samples = compute_min_train_fold_size(
        cv_strategy=cv_strategy,
        n_samples=n_samples,
        n_folds=folds,
    )
```

The `from .cv_utils import compute_min_train_fold_size` is added inline rather than at the module top because `cv_utils` is already imported elsewhere in this module via `from .cv_utils import validate_cv_strategy_for_task` (search.py:1052). Adding it next to the use-site keeps the diff small and matches the existing inline-import idiom used in this file.

**Step 3.** Run the integration test.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_cv_pls_clamp.py::TestRunSearchPLSGridClamping -x -v
```

Expected: all 4 tests pass. The previously-failing `test_n10_loo_clamps_to_9_components` should now produce `n_components_max == 9`.

**Step 4.** Run the full CV-strategy test suite to confirm no regression.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_cv_strategy.py -x -v
```

Expected: all tests pass (1456-line suite shipped with PR #4). If any test fails, STOP and investigate — the regression must come from the formula change, not from the new helper (which is purely additive).

**Step 5.** Commit.

```bash
git add src/spectral_predict/search.py tests/test_cv_pls_clamp.py
git commit -m "fix: clamp PLS components by CV-aware training fold size in run_search (T-10)

Replace n_samples * (folds - 1) // folds at search.py:1109 with the
new cv_strategy-aware compute_min_train_fold_size helper. LOO now
gets the correct n-1 bound instead of the K-fold floor (4n/5),
unlocking 1-2 extra grid components for small LOO runs.

K-fold and RepeatedKFold paths produce identical numerics to the old
formula (helper preserves the exact * (n-1) // n geometry).

Adds 4 integration tests on tiny synthetic data (N=10) covering:
- N=10 k=5: max n_components clamped to 8 (regression test for fix)
- N=10 LOO: max n_components reaches 9 (proves CV-aware logic works)
- N=80 k=5: full grid still produced when train fold >> max_n_components
- N=10 repeated_kfold: matches kfold geometry (n_repeats irrelevant)"
```

---

### Task 4: Replace the K-fold-only formula in `run_bayesian_search`

**Files:**
- Modify: `src/spectral_predict/search.py` (the Bayesian path at lines 3309–3324)

**Step 1.** Find at lines 3309–3324:

```python
    # Adjust max_n_components based on data constraints (same logic as run_search)
    # For small wavelength subsets, PLS n_components must be capped
    # Use TRAINING fold size (not test fold) since PLS is fit on training data
    min_train_samples = n_samples * (folds - 1) // folds

    if task_type == "regression":
        # Strict constraint for PLS regression: n_components <= min(n_samples_train, n_features)
        safe_max_components = min(max_n_components, min_train_samples, n_features)
    else:
        # More relaxed for PLS-DA classification
        safe_max_components = min(max_n_components, n_features)

    if safe_max_components < max_n_components:
        print(f"Note: Bayesian search reducing max PLS components from {max_n_components} to {safe_max_components} "
              f"(n_features={n_features}, min_train_size~{min_train_samples})")
        max_n_components = safe_max_components
```

**Step 2.** Replace the `min_train_samples = ...` line. Apply the same edit pattern as Task 3:

Find:
```python
    # Adjust max_n_components based on data constraints (same logic as run_search)
    # For small wavelength subsets, PLS n_components must be capped
    # Use TRAINING fold size (not test fold) since PLS is fit on training data
    min_train_samples = n_samples * (folds - 1) // folds
```

Replace with:
```python
    # Adjust max_n_components based on data constraints (same logic as run_search).
    # T-10: cv_strategy-aware. See compute_min_train_fold_size for semantics.
    from .cv_utils import compute_min_train_fold_size
    min_train_samples = compute_min_train_fold_size(
        cv_strategy=cv_strategy,
        n_samples=n_samples,
        n_folds=folds,
    )
```

**Step 3.** Add a Bayesian-path integration test. Append to `tests/test_cv_pls_clamp.py` (still inside `TestRunSearchPLSGridClamping` is fine, or create `TestRunBayesianSearchPLSGridClamping` for clarity):

```python
class TestRunBayesianSearchPLSGridClamping:
    """Black-box: run_bayesian_search must clamp the PLS LV upper bound for small datasets.

    Bayesian search uses an Optuna IntDistribution for n_components rather than
    an enumerated grid, but the upper bound (max_n_components) flows through the
    same min_train_samples clamp as run_search. We confirm by inspecting the
    'LVs' / 'n_components' column of the returned DataFrame and asserting no
    trial exceeded the CV-aware bound.
    """

    @pytest.fixture
    def tiny_regression_data(self):
        import numpy as np
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 50))
        y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.05 * rng.standard_normal(10)
        return X, y

    def test_n10_loo_bayesian_caps_at_9(self, tiny_regression_data):
        from spectral_predict.search import run_bayesian_search
        X, y = tiny_regression_data
        # Use a tiny n_trials to keep this test fast.
        df, _ = run_bayesian_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='loo',
            n_trials=8,
            max_n_components=20,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 9, (
            f"Bayesian PLS for N=10 LOO produced n_components={max(n_components_seen)}, "
            f"expected max 9 (n-1). Clamp is broken in run_bayesian_search."
        )
```

**Step 4.** Run the Bayesian test.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_cv_pls_clamp.py::TestRunBayesianSearchPLSGridClamping -x -v
```

Expected: 1/1 pass.

**Step 5.** Run the broader test suite for any Bayesian regressions.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_bayesian_utils.py tests/test_cv_strategy.py -x
```

Expected: all pass.

**Step 6.** Commit.

```bash
git add src/spectral_predict/search.py tests/test_cv_pls_clamp.py
git commit -m "fix: clamp PLS components by CV-aware training fold size in run_bayesian_search (T-10)

Mirror the run_search fix at search.py:3312. Bayesian search's
n_components upper bound now uses compute_min_train_fold_size,
so LOO Bayesian runs on small datasets get n-1 components instead
of the K-fold floor.

Adds 1 integration test (N=10 LOO Bayesian with 8 trials)."
```

---

### Task 5: Defense-in-depth in `models.py` (docstring + assert)

**Files:**
- Modify: `src/spectral_predict/models.py` (lines 582–584 docstring, line 842 grid line)

**Step 1.** Update the `max_n_components` parameter docstring at lines 582–584. Find:

```python
    max_n_components : int, default=10
        Maximum number of PLS components to test
```

Replace with:

```python
    max_n_components : int, default=10
        Maximum number of PLS components to test. CALLER MUST CLAMP this by
        the smallest training-fold size (use cv_utils.compute_min_train_fold_size).
        models.py does NOT independently clamp by n_samples — that is the
        caller's responsibility, because the CV strategy lives at the call site.
        See run_search / run_bayesian_search for the canonical pattern.
```

**Step 2.** Update the grid construction comment at lines 840–843. Find:

```python
    # PLS components grid - test ALL integer values from 1 to max allowed
    # Max is limited by both n_features and max_n_components (which is adjusted for CV fold size)
    pls_max = min(n_features, max_n_components)
    pls_components = list(range(1, pls_max + 1)) if pls_max >= 1 else [1]
```

Replace with:

```python
    # PLS components grid - test ALL integer values from 1 to max allowed.
    # T-10: max_n_components MUST be pre-clamped by min_train_fold_size by the
    # caller (search.py uses cv_utils.compute_min_train_fold_size). models.py
    # only clamps by n_features here as a final guard against feature-only
    # caps; we deliberately do NOT silently re-clamp by n_samples because
    # we don't have CV strategy at this scope.
    pls_max = min(n_features, max_n_components)
    pls_components = list(range(1, pls_max + 1)) if pls_max >= 1 else [1]
```

**Step 3.** No new tests in this task — the helper tests already pin behavior, and adding an assert inside `get_model_grids` would break the many tests that pass `max_n_components` directly without going through `run_search`. A docstring-only update is the right level of defense.

**Step 4.** Run the broader model-grid test suite to confirm no behavior change.

```bash
.venv312/Scripts/python.exe -m pytest tests/test_search_comprehensive.py tests/test_tab7_model_development.py tests/test_contamination_detection.py tests/test_tiers_with_examples.py tests/test_model_integration_fix.py -x
```

Expected: all pass.

**Step 5.** Commit.

```bash
git add src/spectral_predict/models.py
git commit -m "docs: clarify max_n_components contract in get_model_grids (T-10)

Document that callers MUST clamp max_n_components by the smallest
training-fold size (use cv_utils.compute_min_train_fold_size).
models.py does not silently re-clamp by n_samples because it does
not see cv_strategy. run_search / run_bayesian_search are the
canonical clamp sites."
```

---

### Task 6: Update PROJECT_STATUS and SESSION_LOG

**Files:**
- Modify: `docs/PROJECT_STATUS.md` (move T-10 from open roadmap to "Recently resolved")
- Modify: `docs/SESSION_LOG.md` (append a 2026-04-29 entry)

**Step 1.** In `docs/PROJECT_STATUS.md`, find the leading note that mentions T-10 (or the reconciled roadmap status block) and add a "T-10 done" line. Add to the "Recently resolved" section a paragraph:

```markdown
### T-10: PLS component grid CV-strategy-aware clamp — FIXED 2026-04-29

`search.py:1109` and `search.py:3312` previously used `n_samples * (folds - 1) // folds`
to bound the PLS `n_components` grid. Correct for K-fold/RepeatedKFold, under-counts
for LOO (true train-fold size is `n - 1`, not `4n/5`). Added
`cv_utils.compute_min_train_fold_size(cv_strategy, n_samples, n_folds)` and routed
both grid clamps through it. LOO grids now reach the proper `n - 1` ceiling.
Group splitters (`group_kfold`, `leave_one_group_out`) deliberately raise
`NotImplementedError` here; T-15 will plumb group-aware sizing.

Test coverage: `tests/test_cv_pls_clamp.py` — 14 unit tests on the helper +
4 grid integration tests on N=10 / N=80 synthetic data + 1 Bayesian integration test.
```

**Step 2.** Append to `docs/SESSION_LOG.md` (date 2026-04-29 if a section exists, or create one):

```markdown
## 2026-04-29 — T-10 PLS component clamp

**What broke today:** roadmap T-10 surfaced that `min_train_samples = n_samples * (folds - 1) // folds`
at `search.py:1109` and `:3312` was K-fold-only. Under LOO the formula under-counts
the train-fold size by 1 (`4n/5` vs `n-1`). Conservative, so not a silent failure
producer in itself, but it shrinks the LOO PLS grid by 1-2 components on small datasets,
and any future caller who forgot to clamp would crash inside CV.

**Design call:** clamp at the call site (where `cv_strategy` is in scope) rather than
inside `models.py`. `models.py` does NOT silently re-clamp by `n_samples` because that
would mask a caller bug — it asserts the contract via docstring instead.

**Files touched:**
- `src/spectral_predict/cv_utils.py` (new helper `compute_min_train_fold_size`)
- `src/spectral_predict/search.py` (two call sites: 1109, 3312)
- `src/spectral_predict/models.py` (docstring + comment)
- `tests/test_cv_pls_clamp.py` (new, 19 tests)

**Group-splitter handling:** `group_kfold` and `leave_one_group_out` raise
`NotImplementedError` from the helper. T-15 will route these through a separate
group-aware sizing path (smallest group size = max possible test fold).
```

**Step 3.** Commit.

```bash
git add docs/PROJECT_STATUS.md docs/SESSION_LOG.md
git commit -m "docs: T-10 PLS component clamp resolution + session log

Move T-10 from open roadmap to 'Recently resolved'. Document the
design call (clamp at call site, not in models.py) and the
group-splitter deferral to T-15."
```

---

## Verification matrix

After all tasks complete, run the full suite and confirm:

```bash
.venv312/Scripts/python.exe -m pytest tests/test_cv_pls_clamp.py tests/test_cv_strategy.py tests/test_search_comprehensive.py tests/test_bayesian_utils.py tests/test_tab7_model_development.py tests/test_contamination_detection.py tests/test_tiers_with_examples.py tests/test_model_integration_fix.py -v
```

Expected: all pass. The `tests/test_cv_pls_clamp.py` block adds 19 new tests (14 helper + 4 grid + 1 Bayesian), no existing tests should regress.

| Scenario | Before fix | After fix |
|---|---|---|
| N=10, k=5, KFold | max grid n_components = 8 | max grid n_components = 8 (unchanged) |
| N=10, LOO | max grid n_components = 8 (under-bound) | max grid n_components = 9 (correct) |
| N=10, RepeatedKFold k=5 | max grid n_components = 8 | max grid n_components = 8 (unchanged) |
| N=80, k=5, max_n_components=10 | max grid n_components = 10 | max grid n_components = 10 (unchanged) |
| N=10, group_kfold | NA — falls through k=5 formula | helper raises NotImplementedError; T-15 |
| Bayesian N=10 LOO | upper bound 8 | upper bound 9 |

---

## Non-goals / explicitly out of scope

- **Group splitters (`group_kfold`, `leave_one_group_out`).** Helper raises `NotImplementedError`. T-15 will add a separate path that uses the smallest-group size as the bound.
- **Per-fold silent clamping inside the model.** Hides the user's intent. If a per-fold fit ever sees `n_components > train_fold_size`, sklearn should raise; the search aggregator should surface that, not swallow it. Out of scope for T-10 — touch `_run_single_config`'s exception swallowing in a separate ticket.
- **NSGA-II / GA-PLS clamping.** `nsga2_search.py:751` (`_get_constrained_pls_components`) already does its own per-trial clamp by both `n_features` and `n_samples`. Some NSGA-II evaluation paths already compute `min_train_samples` from cv_folds. The deferred ticket should be an **AUDIT** of NSGA-II decode/result-row/reporting consistency (not an assumed identical bug) — verify whether `_get_constrained_pls_components` already receives the correct train-fold size or only `n_samples`, and whether the per-trial decode, the result-row writeback, and the reporting step are all consistent. Do NOT change in this PR. T-10 scope is the two grid-construction sites called out in the roadmap.
- **One-class search.** Does not use PLS. No change needed.
- **Bayesian Optuna IntDistribution.** The upper bound for `n_components` is fed via `max_n_components`, which now flows through the same clamp. No change to the Optuna search space construction itself.

---

## Open questions for reviewer

1. Should the helper also accept `'stratified_kfold'` and `'repeated_stratified_kfold'` as aliases? `cv_utils.build_cv_splitter` resolves stratification internally and the user-facing strategy strings are only `'kfold'`, `'repeated_kfold'`, `'loo'`, but a code-export script or a unit test might pass the explicit stratified name. Current plan: accept only the three user-facing strings, raise on others. Alternative: alias `'stratified_kfold' → 'kfold'` and `'repeated_stratified_kfold' → 'repeated_kfold'`.
2. The `test_n80_kfold_uses_full_grid_default_max` test imports `run_search` which triggers the full preprocessing+CV pipeline. That makes the test ~10-30 s per case. Acceptable for a `tests/` slow-tier test, or should it be marked `@pytest.mark.slow`?
3. NSGA-II's `_get_constrained_pls_components(model_param, n_features, n_samples)` at `nsga2_search.py:751` clamps by `n_samples` (not `n_samples_train_fold`). Some NSGA-II evaluation paths already compute `min_train_samples` from cv_folds. The deferred ticket should be an AUDIT of decode/result-row/reporting consistency rather than an assumed identical bug. Filed as a follow-up, not in T-10 scope.

---

## Uncertainty / things to confirm before merge

- **Result-row schema for `Params` / `LVs` / `n_components`.** The `_extract_n_components_seen` helper inspects multiple possible column names because `search.py` has two known schemas (direct `LVs` integer column, JSON-string `Params` column). If neither schema is present in the test output, the helper returns `set()` and the test fails with "No PLS rows produced". Before merging: run `tests/test_cv_pls_clamp.py::TestRunSearchPLSGridClamping::test_n10_kfold_clamps_to_8_components` once, print `df.columns.tolist()` and the dtype of the `n_components`-bearing column, and confirm the helper finds it. If neither schema matches, the helper needs a third branch. Estimated likelihood: low — both schemas have shipped — but worth a 30-second sanity print.
- **`folds=0` under LOO.** The helper returns `n_samples - 1` regardless of `n_folds`, which is correct, but the call site at `search.py:1109` still passes `folds=5` (the user's UI choice) even when `cv_strategy='loo'`. We rely on `compute_min_train_fold_size` ignoring `n_folds` for LOO. Test `test_loo_ignores_n_folds` pins this behavior.
- **`PROJECT_STATUS.md` exact location for the resolved entry.** The doc has multiple "Previously" stanzas; place the T-10 resolved note inside the existing "Recently resolved" section, not in the leading status block.
