# T-07: Fix PDS Even-Window Arithmetic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the B-allocation overflow in `estimate_pds` (`src/spectral_predict/calibration_transfer.py:170-222`) and the matching ambiguity in `apply_pds` (`:225-258`) when `window` is even.

**Architecture:** Reject even windows at the API boundary with an explicit `ValueError`, and harden `apply_pds` to read `B.shape[1]` instead of the redundant `window` argument so legacy/loaded `B` arrays drive their own slicing. Single-file change in `src/spectral_predict/calibration_transfer.py`. New unit tests live in `tests/test_pds_window_arithmetic.py`. Reference implementation: Wang, Veltkamp, & Kowalski (1991), "Multivariate Instrument Standardization," *Anal. Chem.* 63(23), 2750-2756 — defines the PDS local-window regression as a *centered, odd-width* neighborhood around each wavelength channel.

**Tech Stack:** numpy, pytest. No new dependencies.

**Source:** RECONCILED_ROADMAP_2026-04-29.md ticket T-07. Originally surfaced in `docs/analysis_vs_unscrambler/LESSER_FINDINGS.md` §1.6.

---

## Background

### The bug

`estimate_pds` allocates `B = np.zeros((p, window))` (line 195) but uses
```python
half_window = window // 2                          # line 193
start = max(0, i - half_window)                    # line 199
end   = min(p, i + half_window + 1)                # line 200
```
to slice `X_satellite[:, start:end]`. The slice has `2 * half_window + 1` columns. For **odd** `window` (e.g., 11), `2 * (11//2) + 1 = 11 == window`. For **even** `window` (e.g., 10 or 4), `2 * (window//2) + 1 = window + 1 ≠ window`:

| `window` | `half_window` | slice columns | `B.shape[1]` | overflow? |
|---|---|---|---|---|
| 11 (odd)  | 5 | 11 | 11 | no |
| 5 (odd)   | 2 | 5  | 5  | no |
| 10 (even) | 5 | 11 | 10 | **yes** |
| 4 (even)  | 2 | 5  | 4  | **yes** |

For an interior wavelength `i` (where `start = i - half_window` and `end = i + half_window + 1` are both unclamped), the assignment at line 215
```python
offset = start - (i - half_window)   # = 0 for interior
B[i, offset:offset + len(b)] = b     # B[i, 0:5] = b   ← b has length 5, B[i] has length 4
```
fails with `ValueError: could not broadcast input array from shape (5,) into shape (4,)`. For boundary wavelengths the slice length differs and the failure mode is more subtle (silent corruption of the next row of `B` via numpy fancy-slicing semantics is **not** what happens — the basic slice raises). Net effect: PDS estimation crashes the moment the loop reaches the first interior column when `window` is even.

`apply_pds` at line 253 (`b = B[i, offset:offset + X_window.shape[1]]`) inherits the same arithmetic — for an even-window `B` loaded from disk, the slice length still mismatches and the matrix multiplication at line 256 is silently wrong (or raises depending on shapes).

### Why the literature says "odd"

Wang et al. (1991) derive PDS as a piecewise local regression with a centered window of width `2k+1` around each wavelength `i` (i.e., the window is symmetric: `i-k, ..., i, ..., i+k`). Even widths break the symmetric-center assumption — there is no canonical convention for which side gets the extra channel — and downstream consumers (e.g., MATLAB's PLS_Toolbox, the Python `pyspectra` package) typically reject or coerce.

### Design decision

**Reject even windows with a clear `ValueError`** in `estimate_pds`. Rationale:

1. **Matches the literature.** Wang et al. (1991) defines `2k+1` widths only.
2. **Avoids silent contract drift.** Silent coercion (e.g., `window += 1`) would mean the returned `B.shape[1]` no longer equals the user's stated `window`, so any downstream code that re-reads the requested value (e.g., a saved JSON sidecar where `window` is stored as a scalar) would have to inspect `B.shape[1]` instead. Easier to refuse at the door.
3. **Cheaper than fixing arithmetic.** Fixing arithmetic for even `window` requires picking an asymmetric-center convention (left-biased: `[i-k+1 ... i+k]` or right-biased: `[i-k ... i+k-1]`); both are valid, neither is canonical, and either would surprise users who compare against MATLAB or `pyspectra`.
4. **30-min ticket.** Don't over-engineer.

A small *defensive* hardening goes in alongside: `apply_pds` derives the per-column slice from `B.shape[1]` rather than the redundant `window` arg, so a hand-edited or legacy even-window `B` can't trigger silent shape errors. We **deprecate** the `window` argument in `apply_pds` (warn but tolerate) — the array carries its own shape.

**Not done:** modifying `_run_single_config`, the GUI window picker, or the `TransferModel` schema. UI-side validation can be a follow-up if QA reports user confusion.

---

## Fix sites (pre-verified by reading file)

| Location | Change |
|---|---|
| `src/spectral_predict/calibration_transfer.py:170-194` (top of `estimate_pds`) | Add `if window % 2 == 0: raise ValueError(...)` immediately after the docstring, before `n_samples, p = ...`. Mention Wang et al. (1991) in the message. |
| `src/spectral_predict/calibration_transfer.py:225-258` (`apply_pds`) | Replace `half_window = window // 2` (line 239) with `window_eff = B.shape[1]` and `half_window = window_eff // 2`. Issue `DeprecationWarning` if caller passed a `window` arg that disagrees with `B.shape[1]`. Update the loop accordingly. |
| `src/spectral_predict/calibration_transfer.py:170-194` (docstring) | Add `Raises ValueError if window is even.` and cite Wang et al. (1991). |

---

## Test matrix

| # | window | scenario | expected behavior |
|---|---|---|---|
| 1 | 5 (odd, baseline) | `estimate_pds` + `apply_pds` round-trip on synthetic spectra | runs clean; `B.shape == (p, 5)`; `apply_pds` recovers primary within tolerance |
| 2 | 4 (even, **boundary case**) | `estimate_pds` | raises `ValueError` mentioning "odd" and "Wang" |
| 3 | 10 (even, GUI-plausible default) | `estimate_pds` | raises `ValueError` |
| 4 | 1 (degenerate odd, sanity) | `estimate_pds` | runs (window of 1 = identity-like) |
| 5 | 5, then `apply_pds` called with stale `window=11` | mismatch between caller's `window` and `B.shape[1]=5` | issues `DeprecationWarning`; uses `B.shape[1]`; result correct |
| 6 | 5 (interior column overflow regression) | `estimate_pds` with `p = 50, window = 5` | no `IndexError`/broadcast error on `B[i, ...]` assignment |

---

## Task 1: Create fresh branch off main

**Files:** none modified yet.

**Step 1:** Verify clean baseline.
```bash
git status
git log --oneline -3
```
Expected: on `main`, working tree clean (or only `.claude/settings.local.json` unstaged).

**Step 2:** Create and switch to new branch.
```bash
git checkout -b fix/t07-pds-even-window
```
Expected: "Switched to a new branch 'fix/t07-pds-even-window'".

No commit yet.

---

## Task 2: Write failing test for even-window rejection (RED)

**Files:**
- Create: `tests/test_pds_window_arithmetic.py`

**Step 1:** Write the test file. This is the full content (no placeholders):

```python
"""
Unit tests for PDS window-arithmetic robustness.

Roadmap ticket T-07 — verify that estimate_pds rejects even window
sizes (which would otherwise overflow the B-coefficient allocation),
and that apply_pds derives its slice geometry from B.shape[1] rather
than a redundant `window` argument.

Reference: Wang, Veltkamp, & Kowalski (1991), "Multivariate Instrument
Standardization," Anal. Chem. 63(23), 2750-2756 — PDS is defined on a
centered, odd-width local window of size 2k+1.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from spectral_predict.calibration_transfer import estimate_pds, apply_pds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_pair():
    """Two synthetic instrument spectra on the same grid.

    20 wavelengths is enough that interior columns exist for window<=5
    and we can detect overflow for even widths.
    """
    rng = np.random.default_rng(seed=42)
    n_samples, p = 12, 20
    X_primary = rng.normal(size=(n_samples, p))
    # Satellite = primary plus a small wavelength-local distortion
    X_satellite = X_primary + 0.05 * rng.normal(size=(n_samples, p))
    return X_primary, X_satellite


# ---------------------------------------------------------------------------
# Test 1 — odd window baseline (sanity)
# ---------------------------------------------------------------------------

def test_estimate_pds_odd_window_baseline(synthetic_pair):
    """window=5 (odd) — runs clean, B has expected shape, transform reduces error."""
    X_primary, X_satellite = synthetic_pair
    B = estimate_pds(X_primary, X_satellite, window=5)
    assert B.shape == (X_satellite.shape[1], 5)
    X_transformed = apply_pds(X_satellite, B, window=5)
    pre = np.mean(np.abs(X_primary - X_satellite))
    post = np.mean(np.abs(X_primary - X_transformed))
    assert post < pre, f"PDS should reduce error: pre={pre:.4f}, post={post:.4f}"


# ---------------------------------------------------------------------------
# Test 2 — even window rejection (THE BUG)
# ---------------------------------------------------------------------------

def test_estimate_pds_rejects_even_window_4(synthetic_pair):
    """window=4 (even, boundary case) — must raise ValueError, not crash with broadcast error."""
    X_primary, X_satellite = synthetic_pair
    with pytest.raises(ValueError, match="(?i)odd"):
        estimate_pds(X_primary, X_satellite, window=4)


def test_estimate_pds_rejects_even_window_10(synthetic_pair):
    """window=10 (even, GUI-plausible) — must raise ValueError."""
    X_primary, X_satellite = synthetic_pair
    with pytest.raises(ValueError, match="(?i)odd"):
        estimate_pds(X_primary, X_satellite, window=10)


def test_estimate_pds_error_message_cites_wang(synthetic_pair):
    """The ValueError message should mention Wang or the 1991 reference."""
    X_primary, X_satellite = synthetic_pair
    with pytest.raises(ValueError) as excinfo:
        estimate_pds(X_primary, X_satellite, window=4)
    msg = str(excinfo.value)
    assert "1991" in msg or "Wang" in msg, (
        f"Error should reference Wang et al. (1991); got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — degenerate odd window
# ---------------------------------------------------------------------------

def test_estimate_pds_window_1_is_identity_like(synthetic_pair):
    """window=1 — degenerate but valid odd width; B should be (p, 1)."""
    X_primary, X_satellite = synthetic_pair
    B = estimate_pds(X_primary, X_satellite, window=1)
    assert B.shape == (X_satellite.shape[1], 1)


# ---------------------------------------------------------------------------
# Test 4 — interior-column overflow regression (the actual buggy assignment)
# ---------------------------------------------------------------------------

def test_estimate_pds_no_broadcast_error_at_interior(synthetic_pair):
    """Regression: with the original buggy arithmetic, an even window
    raises broadcast error at the first interior column. Verify that
    odd windows do NOT regress to the same failure (sanity)."""
    X_primary, X_satellite = synthetic_pair
    # window=5, p=20 — i=2 is the first interior column. Just call it.
    B = estimate_pds(X_primary, X_satellite, window=5)
    # If we got here without exception, B[2, :] is fully populated.
    assert np.any(B[2, :] != 0), "Interior B row should not be all zeros for non-degenerate input"


# ---------------------------------------------------------------------------
# Test 5 — apply_pds reads B.shape[1], not the `window` arg
# ---------------------------------------------------------------------------

def test_apply_pds_uses_B_shape_when_window_arg_disagrees(synthetic_pair):
    """A caller passing a stale `window` arg that disagrees with B.shape[1]
    should still get correct geometry (driven by the array), and a warning."""
    X_primary, X_satellite = synthetic_pair
    B = estimate_pds(X_primary, X_satellite, window=5)
    assert B.shape[1] == 5

    # Caller passes a wrong window (11) — apply_pds should warn and use B.shape[1]=5.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_a = apply_pds(X_satellite, B, window=11)
    deprec = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprec, "Expected DeprecationWarning when window arg disagrees with B.shape[1]"

    # And the result should match what apply_pds returns when called consistently.
    X_b = apply_pds(X_satellite, B, window=5)
    np.testing.assert_allclose(X_a, X_b, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 6 — round-trip determinism on a wider odd window
# ---------------------------------------------------------------------------

def test_estimate_apply_pds_roundtrip_window_11():
    """The default GUI window (11) — full round-trip should be stable."""
    rng = np.random.default_rng(seed=7)
    n_samples, p = 30, 50
    X_primary = rng.normal(size=(n_samples, p))
    X_satellite = X_primary + 0.02 * rng.normal(size=(n_samples, p))
    B = estimate_pds(X_primary, X_satellite, window=11)
    assert B.shape == (p, 11)
    X_t = apply_pds(X_satellite, B, window=11)
    assert X_t.shape == X_satellite.shape
    pre = np.mean(np.abs(X_primary - X_satellite))
    post = np.mean(np.abs(X_primary - X_t))
    assert post < pre
```

**Step 2:** Run the new test file. All even-window tests should fail (because the current implementation does not raise). The odd-window tests may pass or fail depending on whether the broadcast error fires at p=20 — this is not the gate for this task. Just record what fails.

```bash
pytest tests/test_pds_window_arithmetic.py -v
```
Expected (pre-fix):
- `test_estimate_pds_odd_window_baseline` — PASS (current code works for odd)
- `test_estimate_pds_rejects_even_window_4` — **FAIL** (current code raises broadcast error or `ValueError` from numpy, not our `ValueError`; or worse, succeeds with corrupted B)
- `test_estimate_pds_rejects_even_window_10` — **FAIL**
- `test_estimate_pds_error_message_cites_wang` — **FAIL**
- `test_estimate_pds_window_1_is_identity_like` — PASS
- `test_estimate_pds_no_broadcast_error_at_interior` — PASS
- `test_apply_pds_uses_B_shape_when_window_arg_disagrees` — **FAIL** (current `apply_pds` does not warn)
- `test_estimate_apply_pds_roundtrip_window_11` — PASS

**Step 3:** Commit the failing tests on their own.
```bash
git add tests/test_pds_window_arithmetic.py
git commit -m "test: add failing tests for PDS even-window rejection (T-07)"
```

---

## Task 3: Apply the minimal fix (GREEN)

**Files:**
- Modify: `src/spectral_predict/calibration_transfer.py`

**Step 1:** Add the even-window guard at the top of `estimate_pds`, immediately after the docstring (before `n_samples, p = X_satellite.shape` on line 192).

Insert this block right after the closing `"""` of the docstring:
```python
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(
            f"window must be a positive integer (got {window!r}). "
            "PDS uses a centered local-regression window of width 2k+1."
        )
    if window % 2 == 0:
        raise ValueError(
            f"window must be odd (got {window}). PDS is defined on a "
            "centered, odd-width local window of size 2k+1 — see "
            "Wang, Veltkamp, & Kowalski (1991), 'Multivariate Instrument "
            "Standardization,' Anal. Chem. 63(23), 2750-2756."
        )
```

**Step 2:** Update the docstring `Parameters` block in `estimate_pds` to read:
```
    window : int
        Window size (odd integer >= 1) for local regression around each
        wavelength. Even values raise ValueError. See Wang et al. (1991).
```
And add a `Raises` section:
```
    Raises
    ------
    ValueError
        If `window` is not a positive odd integer.
```

**Step 3:** Harden `apply_pds`. Replace the body's geometry-from-`window` with geometry-from-`B.shape[1]`:

Old (lines 238-256):
```python
    n_samples, p = X_satellite_new.shape
    half_window = window // 2

    X_transformed = np.zeros_like(X_satellite_new)

    for i in range(p):
        # Determine window boundaries
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        # Extract window from satellite spectra
        X_window = X_satellite_new[:, start:end]

        # Get coefficients for this wavelength
        offset = start - (i - half_window)
        b = B[i, offset:offset + X_window.shape[1]]

        # Apply transformation
        X_transformed[:, i] = X_window @ b

    return X_transformed
```

New:
```python
    import warnings as _warnings

    n_samples, p = X_satellite_new.shape
    window_eff = int(B.shape[1])
    if window_eff % 2 == 0:
        raise ValueError(
            f"B has even second dimension ({window_eff}). PDS coefficients "
            "must come from an odd-width window (see Wang et al. 1991). "
            "Re-estimate B with an odd window."
        )
    if window != window_eff:
        _warnings.warn(
            f"apply_pds: `window` arg ({window}) disagrees with "
            f"B.shape[1] ({window_eff}); using B.shape[1]. The `window` "
            "argument is redundant and may be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    half_window = window_eff // 2

    X_transformed = np.zeros_like(X_satellite_new)

    for i in range(p):
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        X_window = X_satellite_new[:, start:end]

        offset = start - (i - half_window)
        b = B[i, offset:offset + X_window.shape[1]]

        X_transformed[:, i] = X_window @ b

    return X_transformed
```

**Step 4:** Verify the change compiles / imports.
```bash
python -c "from spectral_predict.calibration_transfer import estimate_pds, apply_pds; print('ok')"
```
Expected: `ok`.

**Step 5:** Run the new test file.
```bash
pytest tests/test_pds_window_arithmetic.py -v
```
Expected: all 7 tests pass.

**Step 6:** If any test fails, investigate. Common issues:
- `DeprecationWarning` filter blocking the warning in test 5 → ensure `warnings.simplefilter("always")` is in place (it is).
- Error-message regex too strict → match is `(?i)odd` and the message contains "must be odd" — should match. If not, recheck the actual error string.

**Step 7:** Commit the fix.
```bash
git add src/spectral_predict/calibration_transfer.py
git commit -m "fix: reject even windows in estimate_pds; harden apply_pds geometry (T-07)

estimate_pds(X_primary, X_satellite, window=W) allocated B with shape
(p, W) but sliced X_satellite[:, i-W//2 : i+W//2+1], which has W+1
columns when W is even. The assignment B[i, 0:W+1] = b raised a
broadcast ValueError at the first interior column, breaking PDS for
any even window.

Reject even windows with a clear ValueError citing Wang, Veltkamp, &
Kowalski (1991), 'Multivariate Instrument Standardization,' Anal.
Chem. 63(23), 2750-2756, which defines PDS on a centered, odd-width
local window of size 2k+1.

Also harden apply_pds: derive window geometry from B.shape[1] instead
of the redundant `window` argument; warn (DeprecationWarning) when the
caller passes a `window` that disagrees with B.shape[1]. Defends
against legacy/loaded B arrays carrying authoritative shape."
```

---

## Task 4: Verify nothing else broke

**Files:** none modified.

**Step 1:** Run the existing PDS test that uses `window=11` (odd, baseline).
```bash
pytest tests/gui/test_comprehensive.py::TestCalibrationTransferModule::test_pds_transfer -v
pytest tests/test_end_to_end_workflows.py -v -k "pds"
```
Expected: pass. If they fail, inspect — they shouldn't, since `window=11` is the existing default and unchanged.

**Step 2:** Run the new test file end-to-end one more time.
```bash
pytest tests/test_pds_window_arithmetic.py -v
```
Expected: 7 passed.

**Step 3:** If both green, no commit needed (no files changed).

If either fails, **STOP** and investigate. Likely culprits: a downstream call site somewhere in the GUI or `_run_single_config` that hardcodes an even window — search:
```bash
```
(use Grep tool: pattern `estimate_pds\(.*window\s*=\s*\d+`, type `py`).

If a caller passes `window=10` it will now hit the new `ValueError`. If found, decide per-site whether to fix the caller (preferred) or to coerce in a thin wrapper. Update this plan and revisit.

---

## Task 5: Update living docs

**Files:**
- Modify: `docs/PROJECT_STATUS.md` — append a one-liner under "Recently resolved" pointing to the fix commit and ticket T-07.
- Modify: `docs/SESSION_LOG.md` — append a 2026-04-29 entry summarizing root cause (even `window` ⇒ slice has `window+1` columns ⇒ broadcast error), the design choice (reject vs coerce), and the citation (Wang et al. 1991).
- Modify: `docs/RECONCILED_ROADMAP_2026-04-29.md` — mark T-07 as complete with the fix commit SHA.

**Step 1:** PROJECT_STATUS.md — add under the most recent "Recently resolved" section:
```
- 2026-04-29 — T-07: PDS even-window arithmetic fixed. estimate_pds now
  raises ValueError on even windows (cites Wang et al. 1991); apply_pds
  derives geometry from B.shape[1]. New tests in
  tests/test_pds_window_arithmetic.py.
```

**Step 2:** SESSION_LOG.md — append:
```
## 2026-04-29 — T-07 PDS even-window arithmetic

**Symptom:** estimate_pds(window=10) raised
`ValueError: could not broadcast input array from shape (11,) into shape (10,)`
at the first interior wavelength.

**Root cause:** B allocated as (p, window) but slice spans 2*(window//2)+1
columns. For odd window, that = window. For even window, = window+1.
The assignment B[i, 0:window+1] = b overflows.

**Fix:** Reject even windows with ValueError citing Wang, Veltkamp, &
Kowalski (1991), Anal. Chem. 63(23), 2750-2756. Also harden apply_pds
to derive geometry from B.shape[1] and warn on stale `window` arg.

**Why reject vs coerce:** Wang et al. define PDS on centered odd-width
windows; coercing silently would create a contract drift between
requested `window` and returned `B.shape[1]`. Cheap to enforce at the
boundary.

Commit: <SHA from Task 3>.
```

**Step 3:** RECONCILED_ROADMAP_2026-04-29.md — under T-07, append:
```
**Status:** Fixed 2026-04-29 (commit <SHA>). See
docs/plans/2026-04-29-T07-pds-even-window-fix.md and
tests/test_pds_window_arithmetic.py.
```

**Step 4:** Commit.
```bash
git add docs/PROJECT_STATUS.md docs/SESSION_LOG.md docs/RECONCILED_ROADMAP_2026-04-29.md
git commit -m "docs: close T-07 PDS even-window fix"
```

---

## Task 6: Final verification

**Step 1:** Run the full test suite to ensure no collateral regressions.
```bash
pytest tests/ -v --timeout=120
```
Expected: pass rate identical to pre-fix baseline (or +7 from the new tests). Any new failures must be investigated.

**Step 2:** If clean, the branch is ready for merge. Do not push or open a PR — the next agent / human handles integration per project conventions.

---

## Non-goals / explicitly out of scope

- GUI-side window picker validation. The Tkinter GUI's PDS window control may still let users select even values; the new `ValueError` will surface at run time. UX hardening (greying out evens, picker increments of 2) is a separate ticket.
- Migration of saved `TransferModel` artifacts that happen to have even-window `B` arrays. `apply_pds` now raises if `B.shape[1]` is even; users with legacy bad models must re-estimate. Acceptable per the "this was always broken" framing.
- Refactoring the redundant `window` argument out of `apply_pds`'s signature. Deprecation warning issued; removal deferred to a major version bump.
- Performance: the inner Python loop in `estimate_pds`/`apply_pds` is O(p · n_samples · window²); leaving as-is.

---

## Open questions for reviewer

1. Are there any callers in the GUI (`spectral_predict_gui_optimized.py`) or elsewhere that pass a hard-coded even `window`? A quick `grep` of `estimate_pds\(` and `apply_pds\(` should be definitive — if so, they'll start raising and need their own fix.
2. Is there a saved-model migration concern — i.e., have any production users persisted `TransferModel` instances with even-`window` `B` arrays? Likely no (because the bug means estimation crashed), but worth a sanity check on the `models/` examples.
3. Should `apply_pds`'s new `DeprecationWarning` be promoted to a `FutureWarning` or even `UserWarning`? `DeprecationWarning` is silent by default outside `__main__`, which means user code may never see it. Pick what fits project convention.

---

## Effort estimate

- Task 1 (branch): 1 min
- Task 2 (failing tests): 10 min
- Task 3 (fix + commit): 8 min
- Task 4 (regression check): 5 min
- Task 5 (docs): 5 min
- Task 6 (full test sweep): 3-10 min (depends on test suite size)

**Total: ~35-45 min.** Slightly over the roadmap's 30-min estimate due to the apply_pds hardening + doc updates, both of which the reviewer can drop if they want strict-30.
