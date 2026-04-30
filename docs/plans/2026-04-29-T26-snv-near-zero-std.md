# T-26: SNV Near-Zero Std Tolerance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Prevent NaN in SNV output when a spectrum has near-zero standard deviation; warn the user.

**Architecture:** Numerical floor on the divisor in `snv()` at preprocess.py:43, plus a warning surfaced via the project's logger.

**Tech Stack:** numpy, logging, pytest.

**Source:** roadmap T-26.

---

## Background

`SNV.transform` at `src/spectral_predict/preprocess.py:24-45` computes `(X - means) / stds` per row. The current guard at line 43 is:

```python
stds[stds == 0] = 1.0
```

This catches **exact** zero (true constant spectra), but a spectrum with std on the order of `1e-15` (e.g. a near-constant spectrum from a saturated detector or a fully-blanked region after baseline subtraction) sails past the guard and produces values like `(x - mean)/1e-15` — astronomically large numerical garbage that downstream NaN-propagates through Savgol/PCA/PLS.

The fix is to replace `== 0` with a tolerance threshold (`< 1e-12`) and emit a single `logger.warning(...)` per `transform()` call summarizing how many rows were flagged. This makes the failure mode visible without spamming the console one warning per row on a 50-row dataset where every row trips the threshold.

The roadmap entry says only "log warning when triggered" — this plan picks **warn-once-per-call with a flagged-row count** as the sensible interpretation. Rationale: per-row warnings are noisy in CV folds (each fold's `transform` would log N warnings, and there are typically 5-10 folds × multiple preprocessing combos = thousands of identical warnings); per-call with a count gives the same diagnostic information without log flooding.

---

## Decisions (resolved upfront, no placeholders)

1. **Threshold value:** `1e-12`. Rationale:
   - `np.std` of finite float64 values that are mathematically equal but accumulated through arithmetic typically lands in the `1e-16`–`1e-14` range; `1e-12` is comfortably above that floor while still being eight orders of magnitude below typical real spectral noise (`~1e-4` for absorbance).
   - This is the value specified in the roadmap entry.
2. **Warning style:** **warn once per `transform()` call**, including the count of flagged rows.
   - Format: `f"SNV: {n_flagged} of {n_total} spectra had near-zero std (< 1e-12); divisor floored to 1.0 for those rows. This usually indicates constant or fully-blanked spectra."`
   - Issued only when `n_flagged > 0`.
3. **Logger:** module-level `logger = logging.getLogger(__name__)` declared at the top of `preprocess.py`, matching the pattern used in `baseline_advanced.py:16`, `contamination.py:55`, `library_search.py:31`, etc.
4. **Behavior on normal spectra:** **bit-identical to current behavior**. The threshold only changes behavior for rows with std `< 1e-12`. Rows with std ≥ `1e-12` follow exactly the same code path as today (no log emission, no division change).
5. **Existing tests:** the gold-standard test `tests/numerical/test_preprocessing_correctness.py::test_snv_constant_values` expects constant rows → all-zero output. That still holds: `(x - mean)/1.0 = 0` whether the divisor was floored from `0.0` or from `1e-15`. The existing extended test `tests/test_preprocess_extended.py::test_snv_constant_spectrum` only asserts shape preservation and no-NaN, so it remains green too.
6. **Backward-compatibility surface:** No public API change. `SNV` keeps the same `fit`/`transform` signature, no new constructor parameters. The threshold is a module-level constant `_SNV_STD_FLOOR = 1e-12` (private, underscore-prefixed) so future tuning is a one-line edit.

---

## Files to modify

| File | Purpose |
|---|---|
| `src/spectral_predict/preprocess.py` | Add `import logging`, module logger, `_SNV_STD_FLOOR` constant, update `SNV.transform`. |
| `tests/test_preprocess_extended.py` | Add new `TestSNVNearZeroStd` class with TDD tests for the new behavior. |

**Files NOT touched:**
- `tests/numerical/test_preprocessing_correctness.py` — gold-standard tests; existing constant-row test stays green and is the regression guard for "no behavior change on existing edge case".
- `tests/test_preprocessing_fix_comprehensive.py` — integration tests (`run_search` end-to-end) — should pass unchanged.
- `src/spectral_predict/preprocessing_discovery.py` — uses `SNV` but doesn't depend on internals.

---

## Task 1: Verify clean baseline

**Files:** none modified.

**Step 1:** Confirm git state and that the working tree is on `main` with no relevant unstaged changes to preprocess.py.

```bash
git status
git log --oneline -3
```

Expected: HEAD on `main`, no modifications to `src/spectral_predict/preprocess.py` or `tests/test_preprocess_extended.py`.

**Step 2:** Run the existing SNV tests to confirm green baseline.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_preprocess_extended.py::TestSNVEdgeCases tests/numerical/test_preprocessing_correctness.py::TestSNVCorrectness -v
```

Expected: all SNV tests pass. Capture the exit code (must be 0) before proceeding. If any test is already red, **STOP** and report — the baseline is broken before the change.

No commit yet.

---

## Task 2: Write failing tests (TDD red phase)

**Files:**
- Modify: `tests/test_preprocess_extended.py`

**Step 1:** Add a new test class `TestSNVNearZeroStd` after `TestSNVEdgeCases` (i.e. after the existing `test_snv_preserves_shape` method, before `class TestSavgolEdgeCases`). The class must contain exactly the four tests below.

The tests use `caplog` (pytest's built-in log capture fixture) to assert on `logger.warning` emissions from `spectral_predict.preprocess`.

```python
@pytest.mark.numerical
class TestSNVNearZeroStd:
    """Test SNV's near-zero-std tolerance threshold (T-26)."""

    def test_normal_spectra_no_warning_and_unchanged(self, caplog):
        """Normal spectra produce no warning and bit-identical output to pre-T26 SNV."""
        import logging
        np.random.seed(42)
        X = np.random.randn(20, 200) * 0.5 + 1.0  # std ~0.5, well above 1e-12

        # Reference output — manually computed using the old formula (no floor needed)
        means_ref = X.mean(axis=1, keepdims=True)
        stds_ref = X.std(axis=1, keepdims=True)
        X_expected = (X - means_ref) / stds_ref

        snv = SNV()
        with caplog.at_level(logging.WARNING, logger="spectral_predict.preprocess"):
            X_actual = snv.fit_transform(X)

        np.testing.assert_allclose(
            X_actual, X_expected,
            rtol=1e-12, atol=1e-12,
            err_msg="Normal-spectrum SNV output changed after T-26 floor",
        )
        assert not any(
            "near-zero std" in rec.message for rec in caplog.records
        ), "Should not warn on normal spectra"

    def test_constant_spectrum_no_nan_unit_divided_with_warning(self, caplog):
        """Constant spectrum (std == 0) produces no NaN, gets unit-divided, and warns once."""
        import logging
        X = np.ones((3, 50)) * 7.5  # exactly constant -> std = 0

        snv = SNV()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="spectral_predict.preprocess"):
            X_snv = snv.fit_transform(X)

        # No NaN/inf
        assert np.all(np.isfinite(X_snv)), "Output must be finite"
        # Constant input -> (x - mean) = 0, divided by floored std=1.0 -> all zeros
        assert np.allclose(X_snv, 0.0), "Constant spectrum should yield all-zero SNV output"

        warn_records = [r for r in caplog.records if "near-zero std" in r.message]
        assert len(warn_records) == 1, (
            f"Should emit exactly one warning per transform() call, got {len(warn_records)}"
        )
        assert "3 of 3" in warn_records[0].message, (
            f"Warning should report 3 of 3 flagged rows; got: {warn_records[0].message}"
        )

    def test_near_constant_spectrum_unit_divided_with_warning(self, caplog):
        """Near-constant spectrum (std ~ 1e-15) is unit-divided and warns; no numerical blowup."""
        import logging
        # Build a row whose std is below 1e-12 but not exactly 0.
        # np.std of [a, a, a + 1e-15, a, ...] is ~ 1e-16 to 1e-15 depending on N.
        rng = np.random.default_rng(0)
        base = 1.234
        X = np.full((2, 100), base)
        # Add tiny perturbations on the order of 1e-15
        X[0, 0] += 1e-15
        X[0, 50] -= 1e-15
        X[1, 10] += 2e-15

        # Sanity: std must actually be below the 1e-12 floor for this test to be meaningful.
        assert (X.std(axis=1) < 1e-12).all(), (
            f"Test setup error: row stds {X.std(axis=1)} are not below 1e-12"
        )

        snv = SNV()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="spectral_predict.preprocess"):
            X_snv = snv.fit_transform(X)

        # Must be finite — pre-fix this would produce ~1e15 magnitude or NaN.
        assert np.all(np.isfinite(X_snv)), "Output must be finite (no NaN/inf from tiny divisor)"
        # Magnitude check: with floor = 1.0 divisor, (x - mean) is at most ~2e-15, so output magnitude
        # must be on the order of 1e-15, not 1e15.
        assert np.max(np.abs(X_snv)) < 1e-10, (
            f"Output magnitude {np.max(np.abs(X_snv))} suggests divisor was NOT floored"
        )

        warn_records = [r for r in caplog.records if "near-zero std" in r.message]
        assert len(warn_records) == 1, (
            f"Should emit exactly one warning per transform() call, got {len(warn_records)}"
        )
        assert "2 of 2" in warn_records[0].message

    def test_mixed_normal_and_flagged_warns_with_correct_count(self, caplog):
        """Dataset with mix of normal + constant rows: warning counts only flagged rows."""
        import logging
        np.random.seed(1)
        X_normal = np.random.randn(7, 80) * 0.3 + 2.0  # 7 normal rows
        X_const = np.full((3, 80), 4.2)  # 3 constant rows
        X = np.vstack([X_normal, X_const])

        snv = SNV()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="spectral_predict.preprocess"):
            X_snv = snv.fit_transform(X)

        # Normal rows: SNV yields mean=0 std=1
        for i in range(7):
            assert np.isclose(X_snv[i].mean(), 0.0, atol=1e-10), f"Normal row {i} mean != 0"
            assert np.isclose(X_snv[i].std(), 1.0, atol=1e-10), f"Normal row {i} std != 1"
        # Constant rows: all zeros
        for i in range(7, 10):
            assert np.allclose(X_snv[i], 0.0), f"Constant row {i} should be all zeros"

        warn_records = [r for r in caplog.records if "near-zero std" in r.message]
        assert len(warn_records) == 1, "Should warn exactly once per transform() call"
        assert "3 of 10" in warn_records[0].message, (
            f"Warning should report 3 of 10 flagged; got: {warn_records[0].message}"
        )
```

**Step 2:** Run only the new tests to confirm they fail in the expected way.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_preprocess_extended.py::TestSNVNearZeroStd -v
```

Expected:
- `test_normal_spectra_no_warning_and_unchanged` — **PASS** even before the fix (the new behavior matches the old behavior on normal spectra; test exists as a regression guard).
- `test_constant_spectrum_no_nan_unit_divided_with_warning` — **FAIL on warning assertion** (no warning is emitted today; numerical part already passes because `stds == 0` is already caught).
- `test_near_constant_spectrum_unit_divided_with_warning` — **FAIL on numerical assertion** (`np.max(np.abs(X_snv)) < 1e-10` fails — current code divides by `1e-15`, producing huge numbers; warning assertion also fails).
- `test_mixed_normal_and_flagged_warns_with_correct_count` — **FAIL on warning assertion**.

If any of those red-phase failures don't appear, **STOP** and inspect — either the test setup is wrong (e.g. `np.std` returned a value above `1e-12` unexpectedly) or the production code already has the fix.

**Step 3:** Commit the failing tests as the TDD red checkpoint.

```bash
git add tests/test_preprocess_extended.py
git commit -m "test: add T-26 failing tests for SNV near-zero std threshold"
```

---

## Task 3: Implement the fix (TDD green phase)

**Files:**
- Modify: `src/spectral_predict/preprocess.py`

**Step 1:** Add `import logging` and a module-level logger near the top of the file. Currently `preprocess.py:1-7` is:

```python
"""Preprocessing transformers for spectral data."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
```

Replace with:

```python
"""Preprocessing transformers for spectral data."""

import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Floor for SNV per-row standard deviation. Rows with std below this are treated
# as constant: divisor is replaced with 1.0 to avoid numerical blowup / NaN.
# See roadmap T-26.
_SNV_STD_FLOOR = 1e-12
```

**Step 2:** Update `SNV.transform` (currently `preprocess.py:24-45`). Replace the body from line 38 onward (`X = np.asarray(X)` through `return (X - means) / stds`) with:

```python
        X = np.asarray(X)
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)

        # Tolerance floor: rows with std < _SNV_STD_FLOOR are effectively constant.
        # Replace divisor with 1.0 to avoid NaN / numerical blowup, and emit a
        # single warning per transform() call summarising the flagged-row count.
        flagged_mask = stds < _SNV_STD_FLOOR
        n_flagged = int(flagged_mask.sum())
        if n_flagged > 0:
            stds = np.where(flagged_mask, 1.0, stds)
            logger.warning(
                "SNV: %d of %d spectra had near-zero std (< %g); divisor floored "
                "to 1.0 for those rows. This usually indicates constant or fully-"
                "blanked spectra.",
                n_flagged,
                X.shape[0],
                _SNV_STD_FLOOR,
            )

        return (X - means) / stds
```

Note: the exact-zero guard `stds[stds == 0] = 1.0` on the old line 43 is **removed** — the new floor catches `stds == 0` as a strict subset of `stds < 1e-12`, so it's redundant.

**Step 3:** Sanity-check that the module still imports.

```bash
.venv311/Scripts/python.exe -c "from spectral_predict.preprocess import SNV; print('ok')"
```

Expected: `ok`.

**Step 4:** Run the new tests — they should now pass.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_preprocess_extended.py::TestSNVNearZeroStd -v
```

Expected: all four tests PASS.

If any fail, do NOT modify the test assertions to make them pass. Diagnose the implementation. The most likely failure modes:
- Logger name mismatch — verify `caplog.at_level(..., logger="spectral_predict.preprocess")` matches the actual `__name__` (it should: the module is `spectral_predict/preprocess.py` imported as `spectral_predict.preprocess`).
- Mask shape — `stds` has shape `(n_samples, 1)` after `keepdims=True`; `flagged_mask.sum()` correctly counts flagged rows because each row contributes one True/False, but if you forget `keepdims=True` upstream, the shape will break the broadcast in `np.where`. Re-verify the diff matches Step 2 exactly.

**Step 5:** Run the full SNV test surface — confirm nothing regressed.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_preprocess_extended.py::TestSNVEdgeCases tests/test_preprocess_extended.py::TestSNVNearZeroStd tests/numerical/test_preprocessing_correctness.py::TestSNVCorrectness -v
```

Expected: all green. The existing `TestSNVCorrectness::test_snv_constant_values` (gold-standard "constant input → all zeros") must still pass — it does, because `(x - mean)/1.0 = 0` whether the floor was triggered by `==0` (old) or `<1e-12` (new).

**Step 6:** Run the broader integration test that hits SNV in `run_search` — confirm no regressions in end-to-end.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_preprocessing_fix_comprehensive.py -v -x
```

Expected: same pass/fail status as before the change (compare to baseline from Task 1 if any tests were skipped/red there). If a previously-green test newly fails, **STOP** and investigate; don't proceed to commit.

**Step 7:** Commit the implementation.

```bash
git add src/spectral_predict/preprocess.py
git commit -m "fix: SNV near-zero std tolerance threshold (T-26)

Replace the exact-zero guard at preprocess.py:43 with a 1e-12 floor
on per-row std. Rows below the floor get divisor=1.0 and trigger a
single per-call warning summarising how many rows were flagged.

Prevents numerical blowup / NaN propagation when a spectrum has
near-constant values (e.g. saturated detector, fully-blanked region).
The previous guard only caught exact-zero std and silently passed
through ~1e-15 stds, producing ~1e15 magnitude output.

No behaviour change on normal spectra (regression-tested). Existing
gold-standard 'constant input -> all zeros' test still passes."
```

---

## Task 4: Update living docs

**Files:**
- Modify: `docs/PROJECT_STATUS.md` — add a one-line entry under "What works" or the most recent "Recently resolved" section noting T-26 is done.
- Modify: `docs/SESSION_LOG.md` — append a 2026-04-29 entry summarising the fix and pointing to the commit SHA.
- Modify: `docs/ROADMAP.md` (or the equivalent reconciled-roadmap doc — locate via `git log --oneline -- docs | grep -i roadmap | head -3` if the filename is uncertain) — mark T-26 as done.

**Step 1:** PROJECT_STATUS.md — add under whichever heading tracks completed-this-week items:

```text
- T-26 (2026-04-29): SNV now floors near-zero per-row std (<1e-12) and warns once per transform() call. Prevents NaN propagation from saturated/blanked spectra. See commit <SHA>.
```

**Step 2:** SESSION_LOG.md — append:

```text
## 2026-04-29 — T-26 SNV near-zero std floor

`SNV.transform` previously guarded only `stds == 0`. Spectra with std on the
order of 1e-15 (saturated detector, fully-blanked baseline-subtracted regions)
produced output of magnitude ~1e15, NaN-propagating through downstream Savgol
/ PCA / PLS. Replaced with a `1e-12` tolerance floor and a single per-call
`logger.warning` reporting flagged-row count. Module-level constant
`_SNV_STD_FLOOR` allows future tuning without touching transform logic.

Tests: `tests/test_preprocess_extended.py::TestSNVNearZeroStd` (4 cases:
normal, constant, near-constant 1e-15, mixed). Existing gold-standard
constant-row test still passes (semantics unchanged: constant input still
yields all-zero output).

Commit: <SHA>.
```

**Step 3:** ROADMAP — locate the T-26 entry and mark it complete (e.g. strikethrough, "✅", or move to a "Done" section per the file's existing convention; mirror the convention used for previously-closed tickets).

**Step 4:** Commit the docs.

```bash
git add docs/PROJECT_STATUS.md docs/SESSION_LOG.md docs/ROADMAP.md
git commit -m "docs: close T-26 SNV near-zero std floor"
```

(If the roadmap filename differs, adjust the `git add` accordingly. Do NOT add `docs/plans/2026-04-29-T26-snv-near-zero-std.md` — that was committed up-front before execution started.)

---

## Verification checklist (before declaring done)

- [ ] `pytest tests/test_preprocess_extended.py::TestSNVNearZeroStd -v` — all 4 tests green.
- [ ] `pytest tests/test_preprocess_extended.py::TestSNVEdgeCases -v` — all green (regression guard).
- [ ] `pytest tests/numerical/test_preprocessing_correctness.py::TestSNVCorrectness -v` — all green (gold-standard regression guard).
- [ ] `pytest tests/test_preprocessing_fix_comprehensive.py -v` — same status as Task 1 baseline.
- [ ] `git log --oneline -5` — three commits visible: failing-tests commit, fix commit, docs commit. (Plus the upfront plan commit.)
- [ ] `git status` — clean working tree.
- [ ] `python -c "from spectral_predict.preprocess import SNV, _SNV_STD_FLOOR; print(_SNV_STD_FLOOR)"` — prints `1e-12`.

---

## Non-goals / explicitly out of scope

- Rejecting / dropping near-constant spectra upstream (that's a data-curation concern; SNV's job is just to not produce NaN).
- Making the threshold user-configurable via GUI — `_SNV_STD_FLOOR` is a module constant; if a future user reports it's wrong, that's a separate ticket.
- Changing behavior of `SavgolDerivative`, `SavgolSmooth`, or `build_preprocessing_pipeline` — out of scope.
- Auditing other preprocessors (baselines, smoothing) for analogous near-zero hazards — separate ticket if/when the need arises.

---

## Open questions for reviewer

1. Is `1e-12` the right floor for spectra in absorbance units (typical `~1e-4` noise)? For very-low-signal Raman or fluorescence, the absolute std might legitimately be smaller. Roadmap specified `1e-12`; flagging in case the reviewer has a calibrated value from real datasets.
2. Should the warning include the row indices (e.g. `"rows [3, 17, 42]"`) instead of just a count? Tradeoff: more diagnostic value vs. potentially hundreds of indices for pathological inputs. Current plan: count-only, since the user can easily identify constant rows themselves with `X.std(axis=1) < 1e-12`.
3. Is `logger.warning` the right severity, or should this be `logger.info`? Argument for `warning`: a near-constant spectrum is almost always a data-quality problem the user should investigate. Argument for `info`: in CV with many folds, the same flagged spectra will trigger the warning per fold — could feel spammy. Plan went with `warning` (one warning per `transform()` call is acceptable spam, and severity should match diagnostic importance).
