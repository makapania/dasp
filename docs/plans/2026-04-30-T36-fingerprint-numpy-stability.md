# T-36: `fingerprint_dataset` numpy 2.x repr stability

**Status:** PLANNED — deferred from T-11 PR #6 review.
**Filed:** 2026-04-30.
**Source:** code-reviewer specialist agent during T-11 PR #6 review.
**Priority:** LOW — fires only on numpy upgrade across a resume boundary, which is rare.

## Problem

`src/spectral_predict/run_state.py:fingerprint_dataset` uses
`str(X_arr.flat[idx])` to format a few sample values into the SHA-256
hash:

```python
for idx in (0, X_arr.size // 2, X_arr.size - 1):
    h.update(str(X_arr.flat[idx]).encode("utf-8"))
```

NumPy 2.x changed the default scalar repr semantics ("legacy printing"
was removed, default repr now varies by dtype precision and platform).
Two key user-visible cases:

1. **`np.float64(1.234567890123)`** — NumPy 1.x prints `"1.234567890123"`,
   NumPy 2.x prints `"np.float64(1.234567890123)"` for some dtypes /
   contexts.
2. **`np.float32`** — precision-truncation differs across versions for
   high-precision inputs.

If a user's resume run was fingerprinted on NumPy 1.x and they upgrade
the dasp bundle (which ships its own NumPy via PyInstaller — but a dev
running from `.venv312` could upgrade independently), the same
byte-identical `(X, y)` produces a different fingerprint, triggering a
spurious "data has changed" rejection at `verify_resume_fingerprint`.

User-visible failure mode: "I just upgraded dasp and now it refuses to
resume my paused 12-hour run, claiming the data is different — but I
didn't change anything."

## Fix

Replace `str(val)` with a version-stable formatter:

```python
def _stable_format(val) -> str:
    """Format a numpy scalar in a way that's stable across numpy versions."""
    return f"{float(val):.10g}"
```

`float(val)` always returns a Python float; `:.10g` is a fixed format
specifier whose behavior hasn't changed since Python 3.0. 10 digits of
precision is more than enough for a fingerprint; we're hashing the
*format* of the value, not its full precision.

**Edge cases to handle:**

- **`val` is NaN or infinity**: `float(nan)` → `nan`; `f"{float('nan'):.10g}"`
  → `'nan'`. Stable.
- **`val` is integer (X is int dtype)**: `float(1)` → `1.0`,
  `f"{1.0:.10g}"` → `'1'`. Stable.
- **`val` is complex**: dasp doesn't accept complex spectra; should not
  arise. If it does, `float(complex_val)` raises `TypeError` and the
  outer `except Exception: break` catches it gracefully (current
  behavior).
- **`val` is `None` or non-numeric**: should not arise from a numpy
  array, but the existing `except Exception: break` catches it.

## Files touched

- `src/spectral_predict/run_state.py` — replace `str(...)` in the two
  flat-iteration blocks at the X and y loops.
- `tests/test_run_state.py` — add a regression test that asserts
  identical input arrays produce identical fingerprints regardless of
  numpy version (use `np.float64` and `np.float32` separately to
  exercise the cases that changed).

## Test plan

```python
def test_fingerprint_stable_across_dtypes():
    """Same numerical values in different dtypes produce the same hash."""
    rs = ...  # fresh module
    X64 = np.array([[1.234567890123, 2.0], [3.0, 4.0]], dtype=np.float64)
    X32 = np.array([[1.234567890123, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([0.0, 1.0])
    fp64 = rs.fingerprint_dataset(X64, y)
    fp32 = rs.fingerprint_dataset(X32, y)
    # Note: float32 truncates 1.234567890123 → 1.234568, so fp32 ≠ fp64
    # is the EXPECTED behavior. The stability test is "fp64 today ==
    # fp64 tomorrow" — which we exercise by hashing the bytes directly:
    assert fp64 == hashlib.sha256(...).hexdigest()[:16]  # baked-in expected value
```

The "stability across numpy versions" assertion is hard to test in CI
unless we pin two numpy versions and run twice. The practical version:
hash the format-output bytes directly with a known-good baseline so a
future numpy change that breaks our format function fails this test.

## Open questions

1. **Should `fingerprint_dataset` use a richer hash basis** — e.g. add
   `dtype.str`, `byteorder`, full `tobytes()` of a sample slice? A
   richer basis would catch dtype changes the user might consider
   "the same data" (float32 vs float64 of the same values). But the
   current spirit is "is this clearly different?" — sloppy is fine.
   Defer this question.
2. **Bundle compatibility**: PyInstaller bundles the numpy version that
   was active when the bundle was built. So bundled-app users can't
   silently upgrade numpy across a resume — this bug only fires on dev
   runs from `.venv312` or after a bundle re-build. Lower priority for
   the user base, but fixing is still cheap.
3. **Is `:.10g` enough precision?** For chemometrics spectral
   intensities (typically 0.01-1.0 absorbance, 4-6 sig figs of
   meaningful data), 10 sig figs is overkill. Fine.

## Success criteria

- Resume across a numpy major-version bump no longer triggers spurious
  "data has changed" rejection on byte-identical input.
- Existing fingerprint determinism test (`test_fingerprint_determinism`
  in `tests/test_run_state.py`) still passes — same array produces
  same hash within one numpy version.

## Estimated effort

~20 minutes including test.
