# T-21: Savitzky-Golay Wavelength Uniformity Guard Implementation Plan

> **⚠️ PLAN UNDER REVIEW (2026-04-30) — see [`docs/bugfix_validation/T21_findings.md`](../bugfix_validation/T21_findings.md) before implementing.**
>
> An investigation under the chemometrics master rule flagged three issues with this plan as written:
>
> 1. **Citation error.** `OpenSpecy.is_evenly_spaced()` does not exist. OpenSpecy actually proactively auto-resamples non-uniform spectra via `conform_spec()` rather than warn. The plan's tolerance defense leans on this fabricated citation.
> 2. **Citation unverified.** PLS_Toolbox `gridcheck` at 1e-2 tolerance could not be verified in any public Eigenvector documentation. The 1% CV threshold appears to be dasp inventing a precedent rather than citing one.
> 3. **Scope gap.** The plan claims `wavelengths` is already plumbed at `search.py:612`, but verification shows it isn't. Plus 30+ direct `SavgolDerivative()` instantiations across the GUI / ensemble / GA / interactive paths bypass the guard entirely. As-planned, the guard fires on roughly half the SG paths.
>
> Additional reachability finding (2026-04-30 user verification): the GUI's x-unit RADIO BUTTON is relabel-only and does NOT create non-uniform data (verified: `_on_x_unit_override` in `spectral_predict_gui_optimized.py:19007-19029` does not call `convert_x_axis`). Only the explicit "Convert" button (`_convert_x_unit_and_replot:19031+`) introduces non-uniformity, and that button is rarely used in routine analysis. The bug surface is much narrower than this plan assumes.
>
> **Pending verdict:** DROP / DEFER / APPROVE_MINIMAL (guard at the convert-button site only, not at every SG path). See `T21_findings.md` for the full analysis. Do not implement this plan as-written without first reading the findings + getting an updated disposition decision.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Guard against silent SG-derivative artefacts when wavelength spacing is non-uniform.

**Architecture:** Reusable `_validate_uniform_grid(wavelengths, *, tolerance=0.01, strict=False, context="Savitzky-Golay")` helper at module scope in `src/spectral_predict/preprocess.py`. Helper computes the coefficient of variation `cv = np.diff(wavelengths).std() / abs(np.diff(wavelengths).mean())` and either issues a `UserWarning` with a recommended `scipy.interpolate.interp1d` resampling snippet (default, `strict=False`) or raises `ValueError` (`strict=True`). The helper is plugged into `SavgolDerivative` and `SavgolSmooth` at the top of `transform()`, but only when `self.wavelengths is not None` — backward compatibility is preserved by keeping `wavelengths` as an optional constructor argument that defaults to `None`. Helper lives at module scope so future preprocessing methods (baseline, GLSW, etc.) can reuse it.

**Tech Stack:** numpy, scipy.signal (`savgol_filter`), scipy.interpolate (referenced in warning text only — no new runtime dependency), pytest, warnings (stdlib).

**Source:** roadmap T-21 (`docs/PROJECT_STATUS.md` reconciled roadmap).

---

## Background

Savitzky-Golay derivatives (Savitzky & Golay, *Anal. Chem.* **36**(8):1627, 1964) compute weighted polynomial coefficients on a fixed-stride window. The weights derived in the original paper assume **uniform sampling** — each point is at coordinate `x_0, x_0 + Δ, x_0 + 2Δ, …`. `scipy.signal.savgol_filter` carries that same assumption: it returns *index-space* derivatives (units: 1/sample) and silently produces them regardless of the actual `x` spacing. When users feed a non-uniform grid (e.g. NIR ASD data 350–2500 nm @ 1 nm concatenated with MIR OPUS data 4000–400 cm⁻¹ @ 4 cm⁻¹ resolution after `cm⁻¹↔nm` conversion, or ASD splice corrections at 1000 nm and 1800 nm), the filter still runs to completion but the output is no longer a true derivative — peak positions shift, peak heights distort, and visual inspection is needed to detect anything is wrong.

The downstream risk is high because:

1. The bug is **silent** — no exception, no warning. CV metrics still compute. PLS still fits.
2. The output is **plausible-looking** — derivative spectra still oscillate around zero with reasonable amplitudes.
3. The artefacts mimic **real chemical effects** — apparent peak shifts can be misinterpreted as red/blue shifts from H-bonding or other chemistry. This has been a recurring source of error in published chemometrics work.
4. Spectral Predict explicitly supports **multi-instrument workflows** (ASD + OPUS + SPC + JCAMP, see `MEMORY.md` "X-Axis Unit System" section). Any user who concatenates files from instruments with different sampling resolutions hits this silently.

**Existing Spectral Predict context:**
- `SavgolDerivative` and `SavgolSmooth` live at `src/spectral_predict/preprocess.py:48-220`. Neither receives wavelengths today — they operate purely on `X`.
- `build_preprocessing_pipeline()` at `preprocess.py:279` already accepts a `wavelengths=None` kwarg (currently used only for `WavelengthExcluder` at `:355`). Threading it into the SG constructors is a one-line change in the three SG branches (`:534, :538, :543`).
- `wavelengths` is already plumbed from `search.py` callers (e.g. `search.py:612, 1958`). They pass `X.columns.values` when the input is a DataFrame.
- Existing tests at `tests/test_preprocess_extended.py` and `tests/numerical/test_preprocessing_correctness.py` exercise SG transformers without wavelengths. None test grid uniformity.

**Why now:** roadmap T-21 flagged this as a "silent scientific bug." 1–2 hr scope. No user-visible behavior change for uniform grids (which is the dominant case).

---

## Design Decisions

### Decision 1: Tolerance threshold = `0.01` (1 % coefficient of variation)

`cv = np.diff(wavelengths).std() / abs(np.diff(wavelengths).mean())` is the dimensionless coefficient of variation of consecutive sample spacings.

- **`< 0.001`** would reject legitimate datasets — float arithmetic on regenerated wavelength axes (e.g., `np.linspace(350, 2500, 2151)` round-tripped through `cm⁻¹↔nm`) routinely produces `cv ≈ 1e-7` to `1e-5`, but visual inspection of stored axis vectors in real ASD/OPUS files shows `cv` up to `~3e-4` due to instrument-side rounding.
- **`< 0.01`** is the conventional chemometrics threshold for "effectively uniform" — admits the small float / instrument noise above while rejecting any deliberate concatenation. An ASD 1 nm grid spliced to an OPUS 4 cm⁻¹ grid (~ 1 nm at 1500 nm but ~ 16 nm at 4000 nm in nm-space) gives `cv > 0.5`, well above either threshold.
- **`< 0.05`** is too permissive — admits 5 % spacing variation, at which point a true derivative is already significantly distorted.

**Rationale for 0.01:** balances false-positive rate (rejecting clean data) against false-negative rate (admitting harmful data). Aligns with common chemometrics practice (e.g., the OpenSpecy `is_evenly_spaced()` helper uses 1e-2 by default; PLS_Toolbox's `gridcheck` warns at ~1e-2). Documented in helper docstring with citation hint.

### Decision 2: Warn-and-proceed (not raise)

The helper issues a `UserWarning` (via `warnings.warn`, category `UserWarning`, stacklevel=3 so the warning points at the user's `fit_transform` call site, not at the helper itself) and proceeds with the computation. It does **not** raise.

**Rationale:**
- **Scientific publications need defensible behavior.** A loud warning at the analysis stage is defensible — the user sees it, can either fix the grid or document the deviation, and the model still trains so calibration metrics are still produced for diagnostic comparison.
- **Hard-raising would break the GUI workflow.** Users routinely run `run_search` with 50+ preprocessing combinations overnight; one ValueError in `SavgolDerivative.transform()` would crash the whole grid search and waste compute. A warning lets the run finish, and the user inspects the warnings tab in the morning.
- **Resampling is a user decision, not the library's.** Different downstream tasks tolerate different interpolators (linear vs cubic vs spline; with vs without endpoint padding). Force-resampling silently would be just as much a "silent scientific bug" as the original problem.
- **The warning text includes a copy-pasteable resampling snippet** so the user can act on it without external research.
- A `strict=True` keyword on the helper allows callers (e.g., a future "publication mode" CLI flag) to opt into hard-raising. Not exposed in the GUI in this ticket — left as a future follow-up.

The warning text:

```
Non-uniform wavelength grid detected for {context} (cv={cv:.4f}, tolerance=0.01).
Savitzky-Golay derivatives assume uniform sampling; non-uniform grids produce
artefactual peak shifts and amplitude distortion. To fix, resample to a uniform
grid before applying SG, e.g.:
    from scipy.interpolate import interp1d
    target_wl = np.linspace(wavelengths.min(), wavelengths.max(), len(wavelengths))
    X_resampled = interp1d(wavelengths, X, axis=1, kind='cubic')(target_wl)
Proceeding with the original grid; results may not represent true derivatives.
```

(The warning is issued exactly once per `(SavgolDerivative, SavgolSmooth)` instance because Python's default warning filter dedupes by location. We do **not** override this — repeated warnings during CV folds would spam the GUI log.)

### Decision 3: Helper at module scope, called from both SG transformers

`_validate_uniform_grid(wavelengths, *, tolerance=0.01, strict=False, context="Savitzky-Golay")` is a **module-level private function** in `preprocess.py`. Called from:

- `SavgolDerivative.transform()` — after the existing window-validation logic, before `savgol_filter`.
- `SavgolSmooth.transform()` — same insertion point.

`SavgolDerivative.__init__` and `SavgolSmooth.__init__` gain a new `wavelengths=None` keyword. `build_preprocessing_pipeline()` threads its existing `wavelengths` arg into all three SG branches (`:534, :538, :543`).

**Why a helper, not inline:**
- Reusable for future preprocessing methods (baseline, GLSW, EPO, calibration transfer DS/PDS — all have spacing assumptions).
- Single source of truth for the tolerance constant.
- Testable in isolation without instantiating an SG transformer.

**Why `wavelengths` at the constructor (not `transform`):**
- Matches sklearn convention — transformers are stateful objects whose configuration is captured at `__init__`. Passing `wavelengths` to `transform()` would break sklearn's `.fit(X).transform(X)` call signature and trip `Pipeline` integration.
- `wavelengths` is metadata, not data. Keeping it on the estimator survives serialization, pickling, and the existing `model_io.py` save/load logic.

**Backward compatibility:** all existing call sites that don't pass `wavelengths` get `wavelengths=None`, which short-circuits the check (no warning, no exception). Existing tests pass unchanged. The 30+ direct `SavgolDerivative(...)` instantiations in `ensemble_preprocessing.py`, `ga_preprocessing.py`, `interactive.py`, `nsga2_search.py`, etc. continue working — they just don't get the new guard until a follow-up ticket plumbs `wavelengths` into them. That's intentional scope-limiting for T-21.

---

## Test Cases (TDD red-green order)

| # | Grid | Expected `cv` | Expected behavior | Notes |
|---|---|---|---|---|
| (a) | NIR ASD 400–2500 nm step 1 nm (2101 points) | `~0.0` (≤ 1e-15) | passes silently, no warning | Standard ASD instrument grid |
| (b) | ASD+OPUS concatenated: 400–2500 nm @ 1 nm joined to 4000–400 cm⁻¹ @ 4 cm⁻¹ converted to nm | `> 0.4` | issues `UserWarning` matching "Non-uniform wavelength grid" | Mixed-instrument splice; primary motivating case |
| (c) | MIR cm⁻¹ grid 4000–400 step 4 (901 points) | `~0.0` (≤ 1e-15) | passes silently, no warning | Standard OPUS instrument grid |
| (d) | `np.linspace(350, 2500, 2151)` round-tripped `nm → cm⁻¹ → nm` via `1e7/x` | `> 0` but `< 0.01` | passes silently, no warning | Float-precision noise on a clean grid; must NOT trigger |
| (e) | ASD with 1 missing point (e.g., 400, 401, 402, 404, 405, …) — single dropout | `cv ≈ 0.7` | issues warning | Models a NaN-removed wavelength column |
| (f) | Helper called with `strict=True` on a non-uniform grid | n/a | raises `ValueError` matching "Non-uniform wavelength grid" | Validates the strict path even though it's not wired in by default |
| (g) | Helper called with `wavelengths=None` | n/a | returns immediately, no warning | Backward compat |
| (h) | Helper called with `len(wavelengths) < 2` | n/a | returns immediately, no warning (cannot compute diff) | Edge case: single-wavelength input |
| (i) | `SavgolDerivative(wavelengths=non_uniform).fit_transform(X)` | `cv > 0.01` | issues warning AND returns same `X_deriv` numerics as without `wavelengths` | The guard does NOT modify the output, only warns |
| (j) | `SavgolSmooth(wavelengths=non_uniform).fit_transform(X)` | `cv > 0.01` | issues warning AND returns same `X_smooth` numerics as without `wavelengths` | Same as (i) for smoothing |
| (k) | `build_preprocessing_pipeline("deriv", wavelengths=non_uniform_wl, deriv=1, window=7, polyorder=2)` then `pipe.fit_transform(X)` | `cv > 0.01` | warning fires once during fit | End-to-end pipeline integration |
| (l) | Re-fit on same data twice with same `SavgolDerivative` instance | n/a | warning fires once total (Python warnings dedup) | Avoid GUI log spam during CV |

---

## Task 0: Pre-flight — environment + working state

**Files:** none modified.

**Step 1:** Confirm clean tree on a fresh feature branch off `main`.

```bash
git -C C:/Users/sponheim/git/dasp status --short
git -C C:/Users/sponheim/git/dasp log --oneline -3
```

Expected: clean (or only `.claude/settings.local.json`). HEAD on `main`.

**Step 2:** Create branch.

```bash
git -C C:/Users/sponheim/git/dasp checkout -b feat/t21-sg-wavelength-uniformity-guard
```

Expected: "Switched to a new branch …".

**Step 3:** Verify `pytest` runs the existing SG suite green so we have a known-good baseline.

```bash
cd C:/Users/sponheim/git/dasp
.venv311/Scripts/python.exe -m pytest tests/test_preprocess_extended.py tests/numerical/test_preprocessing_correctness.py -v --tb=short
```

Expected: all pass. Record the exit code in the task notes — if anything is already red, **stop** and report; do not proceed with new tests on a broken baseline.

No commit yet.

---

## Task 1: Write failing tests (RED phase)

**Files:**
- Create: `tests/test_sg_wavelength_uniformity.py`

**Step 1:** Write the full test file. All tests should fail (or error on missing import) until Task 2 ships the implementation.

```python
"""Tests for Savitzky-Golay wavelength-uniformity guard (T-21).

Verifies that SavgolDerivative and SavgolSmooth warn (or raise, with strict=True)
when wavelength spacing is non-uniform. Uniform grids (within float-precision
tolerance) must pass silently to avoid warning spam during routine CV runs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from spectral_predict.preprocess import (
    SavgolDerivative,
    SavgolSmooth,
    _validate_uniform_grid,
    build_preprocessing_pipeline,
)


# ---------- Fixtures: representative wavelength grids ----------

@pytest.fixture
def asd_grid_nm():
    """ASD VNIR/SWIR: 400-2500 nm @ 1 nm. Uniform."""
    return np.arange(400.0, 2501.0, 1.0)  # 2101 points


@pytest.fixture
def opus_grid_cm():
    """OPUS MIR: 4000-400 cm-1 @ 4 cm-1. Uniform (descending)."""
    return np.arange(4000.0, 399.0, -4.0)  # 901 points


@pytest.fixture
def asd_opus_concat_nm(asd_grid_nm, opus_grid_cm):
    """ASD nm concatenated with OPUS cm-1 converted to nm. Strongly non-uniform."""
    opus_in_nm = 1e7 / opus_grid_cm  # 2500-25000 nm
    return np.sort(np.concatenate([asd_grid_nm, opus_in_nm]))


@pytest.fixture
def roundtrip_noisy_grid_nm():
    """Clean nm grid round-tripped nm->cm-1->nm. Tiny float noise but cv < 0.01."""
    nm = np.linspace(350.0, 2500.0, 2151)
    return 1e7 / (1e7 / nm)


@pytest.fixture
def asd_with_dropout(asd_grid_nm):
    """ASD grid with one missing wavelength (e.g., dead pixel removed)."""
    return np.delete(asd_grid_nm, 100)


def _make_X(n_features, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((10, n_features))


# ---------- Helper-level tests ----------

class TestValidateUniformGrid:
    def test_uniform_asd_grid_passes_silently(self, asd_grid_nm):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            _validate_uniform_grid(asd_grid_nm)  # must not warn or raise

    def test_uniform_opus_grid_passes_silently(self, opus_grid_cm):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_uniform_grid(opus_grid_cm)

    def test_concat_grid_warns(self, asd_opus_concat_nm):
        with pytest.warns(UserWarning, match="Non-uniform wavelength grid"):
            _validate_uniform_grid(asd_opus_concat_nm)

    def test_dropout_grid_warns(self, asd_with_dropout):
        with pytest.warns(UserWarning, match="Non-uniform wavelength grid"):
            _validate_uniform_grid(asd_with_dropout)

    def test_roundtrip_float_noise_passes(self, roundtrip_noisy_grid_nm):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_uniform_grid(roundtrip_noisy_grid_nm)

    def test_strict_mode_raises_on_nonuniform(self, asd_opus_concat_nm):
        with pytest.raises(ValueError, match="Non-uniform wavelength grid"):
            _validate_uniform_grid(asd_opus_concat_nm, strict=True)

    def test_strict_mode_passes_on_uniform(self, asd_grid_nm):
        # Should not raise.
        _validate_uniform_grid(asd_grid_nm, strict=True)

    def test_none_wavelengths_returns_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_uniform_grid(None)

    def test_too_few_points_returns_silently(self):
        # Cannot compute np.diff on fewer than 2 points.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_uniform_grid(np.array([500.0]))
            _validate_uniform_grid(np.array([]))

    def test_warning_text_contains_resampling_hint(self, asd_opus_concat_nm):
        with pytest.warns(UserWarning) as record:
            _validate_uniform_grid(asd_opus_concat_nm)
        assert any("interp1d" in str(w.message) for w in record)
        assert any("cv=" in str(w.message) for w in record)

    def test_context_appears_in_warning(self, asd_opus_concat_nm):
        with pytest.warns(UserWarning, match="custom-context"):
            _validate_uniform_grid(asd_opus_concat_nm, context="custom-context")

    def test_descending_grid_treated_same_as_ascending(self, opus_grid_cm):
        # cv uses .std()/.mean(), and mean() on a descending grid is negative.
        # Helper must take abs() of the mean to handle both directions.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_uniform_grid(opus_grid_cm)  # descending, uniform


# ---------- SavgolDerivative integration tests ----------

class TestSavgolDerivativeUniformityGuard:
    def test_uniform_grid_no_warning(self, asd_grid_nm):
        X = _make_X(len(asd_grid_nm))
        sg = SavgolDerivative(deriv=1, window=7, polyorder=2, wavelengths=asd_grid_nm)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sg.fit_transform(X)

    def test_nonuniform_grid_warns(self, asd_opus_concat_nm):
        X = _make_X(len(asd_opus_concat_nm))
        sg = SavgolDerivative(deriv=1, window=7, polyorder=2,
                              wavelengths=asd_opus_concat_nm)
        with pytest.warns(UserWarning, match="Non-uniform wavelength grid"):
            sg.fit_transform(X)

    def test_no_wavelengths_no_warning(self, asd_opus_concat_nm):
        # Backward compat: if wavelengths is not provided, no guard fires.
        X = _make_X(len(asd_opus_concat_nm))
        sg = SavgolDerivative(deriv=1, window=7, polyorder=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sg.fit_transform(X)

    def test_guard_does_not_alter_numerics(self, asd_opus_concat_nm):
        X = _make_X(len(asd_opus_concat_nm))
        sg_with = SavgolDerivative(deriv=1, window=7, polyorder=2,
                                   wavelengths=asd_opus_concat_nm)
        sg_without = SavgolDerivative(deriv=1, window=7, polyorder=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress the expected warning
            out_with = sg_with.fit_transform(X)
        out_without = sg_without.fit_transform(X)
        np.testing.assert_allclose(out_with, out_without, rtol=1e-12, atol=1e-12)

    def test_warning_fires_once_per_instance(self, asd_opus_concat_nm):
        # Default Python warning filter dedupes by location - calling transform
        # twice on the same instance should produce <=1 warning.
        X = _make_X(len(asd_opus_concat_nm))
        sg = SavgolDerivative(deriv=1, window=7, polyorder=2,
                              wavelengths=asd_opus_concat_nm)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("default")
            sg.fit_transform(X)
            sg.transform(X)
        nonuniform_warnings = [w for w in record
                               if "Non-uniform wavelength grid" in str(w.message)]
        assert len(nonuniform_warnings) == 1


# ---------- SavgolSmooth integration tests ----------

class TestSavgolSmoothUniformityGuard:
    def test_uniform_grid_no_warning(self, asd_grid_nm):
        X = _make_X(len(asd_grid_nm))
        sm = SavgolSmooth(window_length=11, polyorder=2, wavelengths=asd_grid_nm)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sm.fit_transform(X)

    def test_nonuniform_grid_warns(self, asd_opus_concat_nm):
        X = _make_X(len(asd_opus_concat_nm))
        sm = SavgolSmooth(window_length=11, polyorder=2,
                          wavelengths=asd_opus_concat_nm)
        with pytest.warns(UserWarning, match="Non-uniform wavelength grid"):
            sm.fit_transform(X)

    def test_guard_does_not_alter_numerics(self, asd_opus_concat_nm):
        X = _make_X(len(asd_opus_concat_nm))
        sm_with = SavgolSmooth(window_length=11, polyorder=2,
                               wavelengths=asd_opus_concat_nm)
        sm_without = SavgolSmooth(window_length=11, polyorder=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_with = sm_with.fit_transform(X)
        out_without = sm_without.fit_transform(X)
        np.testing.assert_allclose(out_with, out_without, rtol=1e-12, atol=1e-12)


# ---------- build_preprocessing_pipeline integration ----------

class TestPipelineIntegration:
    def test_pipeline_threads_wavelengths_to_savgol(self, asd_opus_concat_nm):
        from sklearn.pipeline import Pipeline
        X = _make_X(len(asd_opus_concat_nm))
        steps = build_preprocessing_pipeline(
            "deriv", deriv=1, window=7, polyorder=2,
            wavelengths=asd_opus_concat_nm,
        )
        pipe = Pipeline(steps)
        with pytest.warns(UserWarning, match="Non-uniform wavelength grid"):
            pipe.fit_transform(X)

    def test_pipeline_uniform_grid_silent(self, asd_grid_nm):
        from sklearn.pipeline import Pipeline
        X = _make_X(len(asd_grid_nm))
        steps = build_preprocessing_pipeline(
            "deriv", deriv=1, window=7, polyorder=2,
            wavelengths=asd_grid_nm,
        )
        pipe = Pipeline(steps)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipe.fit_transform(X)

    def test_pipeline_snv_deriv_warns(self, asd_opus_concat_nm):
        from sklearn.pipeline import Pipeline
        X = _make_X(len(asd_opus_concat_nm))
        steps = build_preprocessing_pipeline(
            "snv_deriv", deriv=2, window=9, polyorder=3,
            wavelengths=asd_opus_concat_nm,
        )
        pipe = Pipeline(steps)
        with pytest.warns(UserWarning, match="Non-uniform wavelength grid"):
            pipe.fit_transform(X)

    def test_pipeline_no_wavelengths_silent(self, asd_opus_concat_nm):
        # If caller doesn't pass wavelengths, we cannot validate, so must not warn.
        from sklearn.pipeline import Pipeline
        X = _make_X(len(asd_opus_concat_nm))
        steps = build_preprocessing_pipeline(
            "deriv", deriv=1, window=7, polyorder=2,
            # wavelengths omitted on purpose
        )
        pipe = Pipeline(steps)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipe.fit_transform(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

**Step 2:** Run the new test file. Expect ImportError on `_validate_uniform_grid` (the helper doesn't exist yet) and AttributeError / TypeError on the new `wavelengths` keyword.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_sg_wavelength_uniformity.py -v --tb=short
```

Expected: collection error or every test fails. This is the RED phase.

**Step 3:** Commit the failing tests.

```bash
git -C C:/Users/sponheim/git/dasp add tests/test_sg_wavelength_uniformity.py
git -C C:/Users/sponheim/git/dasp commit -m "test: add failing tests for SG wavelength uniformity guard (T-21)"
```

---

## Task 2: Implement the helper and wire into SG transformers (GREEN phase)

**Files:**
- Modify: `src/spectral_predict/preprocess.py`

**Step 1:** Add the `warnings` import at the top.

Edit at the existing import block (`preprocess.py:1-6`):

```python
"""Preprocessing transformers for spectral data."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
```

→

```python
"""Preprocessing transformers for spectral data."""

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
```

**Step 2:** Add the module-level helper just before `class SNV` (i.e., insert at line ~9, before line 9's `class SNV(...)`).

```python
# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Coefficient-of-variation tolerance for wavelength spacing. Spacings whose
# CV (std/|mean|) exceeds this value are treated as non-uniform. Set to 1e-2
# to admit normal float-precision noise on resampled / round-tripped axes
# (typically <= 1e-4) while rejecting deliberate concatenation of grids from
# different instruments (typically >= 1e-1). Aligns with chemometrics
# convention (e.g. OpenSpecy is_evenly_spaced default).
_UNIFORM_GRID_TOLERANCE = 0.01


def _validate_uniform_grid(wavelengths, *, tolerance=_UNIFORM_GRID_TOLERANCE,
                           strict=False, context="Savitzky-Golay"):
    """Warn (or raise) if a wavelength grid is not approximately uniformly spaced.

    Savitzky-Golay derivatives (Savitzky & Golay, Anal. Chem. 36(8):1627, 1964)
    assume uniform sampling. ``scipy.signal.savgol_filter`` carries the same
    assumption: outputs are index-space derivatives, computed identically
    regardless of the actual ``x`` spacing. A non-uniform grid therefore
    produces an artefactual signal that no longer represents a true derivative
    -- peak positions shift, peak amplitudes distort -- without raising any
    error. This helper guards against that silent failure mode.

    Parameters
    ----------
    wavelengths : array-like or None
        1-D array of wavelength (or wavenumber) values, ascending or descending.
        Must be the same length as the feature axis being transformed. ``None``
        and length-< 2 inputs return immediately without warning.
    tolerance : float, default 0.01
        Threshold on the coefficient of variation
        ``np.diff(wavelengths).std() / abs(np.diff(wavelengths).mean())``.
        Spacings whose CV exceeds this value are flagged.
    strict : bool, default False
        If True, raise ``ValueError`` on non-uniform grids instead of warning.
        Use for "publication mode" callers that need a hard guarantee.
    context : str, default "Savitzky-Golay"
        Free-text label included in the warning / error message so callers
        from different preprocessing methods produce distinguishable diagnostics.

    Returns
    -------
    None
        The helper has no return value -- it side-effects via warnings or
        exceptions.

    Raises
    ------
    ValueError
        If ``strict=True`` and the grid is non-uniform.

    Notes
    -----
    The CV is dimensionless, so the helper works for both nm and cm-1 grids.
    ``abs(mean)`` is used so descending grids (e.g. OPUS 4000 -> 400 cm-1) are
    treated identically to ascending grids.
    """
    if wavelengths is None:
        return
    wl = np.asarray(wavelengths, dtype=float).ravel()
    if wl.size < 2:
        return
    diffs = np.diff(wl)
    mean_diff = abs(diffs.mean())
    if mean_diff == 0:
        return  # all wavelengths identical -- pathological but not our problem
    cv = diffs.std() / mean_diff
    if cv <= tolerance:
        return

    msg = (
        f"Non-uniform wavelength grid detected for {context} "
        f"(cv={cv:.4f}, tolerance={tolerance}).\n"
        f"Savitzky-Golay derivatives assume uniform sampling; non-uniform grids "
        f"produce artefactual peak shifts and amplitude distortion.\n"
        f"To fix, resample to a uniform grid before applying SG, e.g.:\n"
        f"    from scipy.interpolate import interp1d\n"
        f"    target_wl = np.linspace(wavelengths.min(), wavelengths.max(), len(wavelengths))\n"
        f"    X_resampled = interp1d(wavelengths, X, axis=1, kind='cubic')(target_wl)\n"
        f"Proceeding with the original grid; results may not represent true derivatives."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)
```

**Step 3:** Add `wavelengths` to `SavgolDerivative.__init__`.

Edit in `preprocess.py` `class SavgolDerivative` (current `__init__` at line 62):

```python
    def __init__(self, deriv=1, window=7, polyorder=None):
        self.deriv = deriv
        self.window = window
        self.polyorder = polyorder
```

→

```python
    def __init__(self, deriv=1, window=7, polyorder=None, wavelengths=None):
        self.deriv = deriv
        self.window = window
        self.polyorder = polyorder
        self.wavelengths = wavelengths
```

**Step 4:** Add the guard call at the top of `SavgolDerivative.transform`.

The current method body starts (line 89):

```python
    def transform(self, X):
        """..."""
        X = np.asarray(X)

        # Ensure odd window
        window = self.window
```

Replace with:

```python
    def transform(self, X):
        """..."""
        # Wavelength-uniformity guard (T-21). Silent for None / uniform grids.
        _validate_uniform_grid(self.wavelengths,
                               context=f"Savitzky-Golay derivative (deriv={self.deriv})")

        X = np.asarray(X)

        # Ensure odd window
        window = self.window
```

(Keep the docstring exactly as-is; just insert the `_validate_uniform_grid` call between the docstring and `X = np.asarray(X)`.)

**Step 5:** Repeat for `SavgolSmooth`.

`__init__` (current at line 152):

```python
    def __init__(self, window_length=17, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
```

→

```python
    def __init__(self, window_length=17, polyorder=2, wavelengths=None):
        self.window_length = window_length
        self.polyorder = polyorder
        self.wavelengths = wavelengths
```

`transform` (current at line 164):

```python
    def transform(self, X):
        """..."""
        X = np.asarray(X)

        # Ensure odd window
        window = self.window_length
```

→

```python
    def transform(self, X):
        """..."""
        # Wavelength-uniformity guard (T-21). Silent for None / uniform grids.
        _validate_uniform_grid(self.wavelengths,
                               context="Savitzky-Golay smoothing")

        X = np.asarray(X)

        # Ensure odd window
        window = self.window_length
```

**Step 6:** Thread `wavelengths` from `build_preprocessing_pipeline` into the SG branches.

In the existing function body, the SG branches at lines 533–545 currently are:

```python
    elif preprocess_name == "deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        steps.append(("savgol", savgol))

    elif preprocess_name == "snv_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        steps.append(("snv", SNV()))
        steps.append(("savgol", savgol))

    elif preprocess_name == "deriv_snv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        steps.append(("savgol", savgol))
        steps.append(("snv", SNV()))
```

Update all three to pass `wavelengths=wavelengths`:

```python
    elif preprocess_name == "deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder,
                                  wavelengths=wavelengths)
        steps.append(("savgol", savgol))

    elif preprocess_name == "snv_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder,
                                  wavelengths=wavelengths)
        steps.append(("snv", SNV()))
        steps.append(("savgol", savgol))

    elif preprocess_name == "deriv_snv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder,
                                  wavelengths=wavelengths)
        steps.append(("savgol", savgol))
        steps.append(("snv", SNV()))
```

**Step 7:** Also thread `wavelengths` into the `SavgolSmooth` step at line 521 (the `if smoothing:` block):

```python
    if smoothing:
        steps.append(("smooth", SavgolSmooth(
            window_length=smoothing_window,
            polyorder=smoothing_polyorder
        )))
```

→

```python
    if smoothing:
        steps.append(("smooth", SavgolSmooth(
            window_length=smoothing_window,
            polyorder=smoothing_polyorder,
            wavelengths=wavelengths,
        )))
```

**Step 8:** Run the new test file. All tests should pass.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_sg_wavelength_uniformity.py -v --tb=short
```

Expected: every test green.

If any test fails:
- `test_strict_mode_raises_on_nonuniform` failing → check that the `strict=True` branch raises *after* the message is built.
- `test_warning_fires_once_per_instance` failing → check that `warnings.warn` is being called via the default filter (don't override). If the test still produces 2 warnings, look for re-instantiation in `transform()` (we want the same message location each time).
- `test_descending_grid_treated_same_as_ascending` failing → verify `abs(diffs.mean())` is used, not `diffs.mean()`.

**Step 9:** Run the full preprocessing test suite to confirm no regression.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_preprocess_extended.py tests/numerical/test_preprocessing_correctness.py tests/test_sg_wavelength_uniformity.py -v --tb=short
```

Expected: all green. The pre-existing tests don't pass `wavelengths` to SG, so they exercise the backward-compat path.

**Step 10:** Run a wider net to check for accidental side effects in callers of `SavgolDerivative` / `SavgolSmooth` that use positional args.

```bash
.venv311/Scripts/python.exe -m pytest tests/test_ga_preprocessing.py tests/test_ensemble_preprocessing.py tests/test_preprocessing_discovery.py -v --tb=short
```

Expected: all green. These call sites use `SavgolDerivative(deriv=..., window=..., polyorder=...)` with keyword args (verified by grep in the planning step), so adding a new optional kwarg at the end is safe.

**Step 11:** Commit the implementation.

```bash
git -C C:/Users/sponheim/git/dasp add src/spectral_predict/preprocess.py
git -C C:/Users/sponheim/git/dasp commit -m "feat: guard SG derivatives against non-uniform wavelength grids (T-21)

scipy.signal.savgol_filter assumes uniform sampling; on non-uniform grids
it silently produces artefactual derivatives that mimic real chemical
peak shifts. Add module-level _validate_uniform_grid() helper and an
optional wavelengths= kwarg on SavgolDerivative and SavgolSmooth.

Tolerance is 1% coefficient of variation on np.diff(wavelengths) -- admits
float-precision noise on resampled axes while rejecting multi-instrument
concatenation. Default behavior issues UserWarning with a copy-pasteable
scipy.interpolate.interp1d resampling snippet; strict=True raises ValueError
for future publication-mode callers.

build_preprocessing_pipeline() threads its existing wavelengths kwarg into
the three SG branches and the smoothing step. Backward compatible: callers
that don't pass wavelengths get the silent (None) path. See Savitzky &
Golay, Anal. Chem. 36(8):1627, 1964 for the original derivation requiring
uniform sampling."
```

---

## Task 3: Verify on real example data

**Files:** none modified — sanity check only.

**Step 1:** Confirm BoneCollagen ASD example data passes silently (uniform grid).

```bash
.venv311/Scripts/python.exe -c "
import sys
sys.path.insert(0, 'src')
import warnings, numpy as np
from spectral_predict.io import read_asd_dir
from spectral_predict.preprocess import SavgolDerivative

X, meta = read_asd_dir('example/')
wl = X.columns.values.astype(float)
diffs = np.diff(wl)
print(f'ASD grid: n={len(wl)}, range={wl.min()}-{wl.max()}, '
      f'cv={diffs.std()/abs(diffs.mean()):.6f}')
with warnings.catch_warnings():
    warnings.simplefilter('error')
    sg = SavgolDerivative(deriv=1, window=7, polyorder=2, wavelengths=wl)
    out = sg.fit_transform(X.values)
    print(f'OK: SG ran with cv guard active, output shape={out.shape}')
"
```

Expected:
- `cv` should print as something like `0.000000` (ASD grid is exactly 1 nm steps).
- `OK: SG ran with cv guard active, output shape=(N, M)` — no warning, no exception.

**Step 2:** Negative control — synthetically corrupt the grid and confirm the warning fires.

```bash
.venv311/Scripts/python.exe -c "
import sys, warnings
sys.path.insert(0, 'src')
import numpy as np
from spectral_predict.preprocess import SavgolDerivative

# Build a corrupted grid: ASD nm 400-2500 + OPUS-converted 2500-25000 nm
asd = np.arange(400.0, 2501.0, 1.0)
opus = 1e7 / np.arange(4000.0, 399.0, -4.0)
wl = np.sort(np.concatenate([asd, opus]))
X = np.random.default_rng(0).standard_normal((5, len(wl)))

with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter('always')
    sg = SavgolDerivative(deriv=1, window=7, polyorder=2, wavelengths=wl)
    sg.fit_transform(X)
    msgs = [str(w.message) for w in record if 'Non-uniform' in str(w.message)]
    assert msgs, 'Expected non-uniform warning, got none'
    print('OK: warning fired —')
    print(msgs[0])
"
```

Expected: prints "OK: warning fired" followed by the full warning text (including `cv=...`, `interp1d`, and the resampling snippet).

**Step 3:** No commit needed for verification step — but record both invocations' stdout in the next task's session log entry.

---

## Task 4: Update living docs (PROJECT_STATUS, SESSION_LOG, roadmap)

**Files:**
- Modify: `docs/PROJECT_STATUS.md` — flip the T-21 status to "shipped" with the fix SHA.
- Modify: `docs/SESSION_LOG.md` — append an entry dated `2026-04-29` with the verification stdout from Task 3 and the design-decision summary.

**Step 1:** Find the T-21 line in `docs/PROJECT_STATUS.md` (likely under the reconciled roadmap section). Update it to reference the commit SHA from Task 2 and mark as resolved with a one-line summary: tolerance 0.01, warn-and-proceed, helper at module scope.

**Step 2:** Append to `docs/SESSION_LOG.md`:

```markdown
## 2026-04-29 — T-21 SG wavelength-uniformity guard shipped

**What:** Added `_validate_uniform_grid` helper at module scope in
`src/spectral_predict/preprocess.py`. Plumbed optional `wavelengths=` kwarg
into `SavgolDerivative` and `SavgolSmooth`. `build_preprocessing_pipeline`
threads its existing `wavelengths` kwarg through the three SG branches
and the smoothing step.

**Tolerance:** CV (`np.diff(wavelengths).std() / abs(mean)`) > 0.01 triggers
the guard. Picked over 0.05 (too permissive — admits 5% spacing variation
that already distorts derivatives) and 0.001 (too tight — flags float-precision
noise on resampled axes).

**Warn vs raise:** warn-and-proceed by default (`UserWarning` with a
copy-pasteable `scipy.interpolate.interp1d` resampling snippet). `strict=True`
helper kwarg raises `ValueError` for future publication-mode callers; not
exposed in the GUI in this ticket.

**Backward compat:** all existing call sites that don't pass `wavelengths`
hit the `None` short-circuit → no warning, no exception, identical numerics.
30+ direct `SavgolDerivative(...)` instantiations in `ensemble_preprocessing.py`,
`ga_preprocessing.py`, `interactive.py`, `nsga2_search.py`, `unified_bayesian.py`
unchanged — they don't get the guard until a follow-up plumbs `wavelengths`
into them. Intentional T-21 scope limit.

**Verification on BoneCollagen example data:** ASD grid `cv ≈ 0` (uniform 1 nm),
guard is silent, derivative runs as before. Synthetic ASD+OPUS concat with
`cv ≈ 0.5` triggers the warning with the expected message text.

**Commit:** <SHA from Task 2>.
```

**Step 3:** Commit doc updates.

```bash
git -C C:/Users/sponheim/git/dasp add docs/PROJECT_STATUS.md docs/SESSION_LOG.md
git -C C:/Users/sponheim/git/dasp commit -m "docs: T-21 SG wavelength uniformity guard shipped — update status + log"
```

---

## Task 5: Final verification + branch ready for review

**Files:** none modified.

**Step 1:** Run the entire test suite to confirm zero regressions.

```bash
.venv311/Scripts/python.exe -m pytest tests/ -v --tb=short -x --ignore=tests/gui
```

Expected: all green. (`--ignore=tests/gui` skips the Tkinter integration tests, which are slow and orthogonal to this change. If a non-gui test fails, **stop** and investigate before declaring done.)

**Step 2:** Run black formatting on the touched file (project rule: line length 100).

```bash
.venv311/Scripts/python.exe -m black --line-length 100 src/spectral_predict/preprocess.py tests/test_sg_wavelength_uniformity.py
```

Expected: "1 file left unchanged" or "2 files reformatted" — if any reformat happened, commit it as a follow-up `style:` commit and re-run the test suite.

**Step 3:** Print branch + log for handoff.

```bash
git -C C:/Users/sponheim/git/dasp log --oneline main..HEAD
```

Expected: 2 or 3 commits (test, feat, docs, optional style).

**Step 4:** Do **not** push. Plan ends here; the user will decide whether to open a PR or merge directly.

---

## Non-goals / explicitly out of scope

- **Plumbing `wavelengths` into the 30+ direct `SavgolDerivative(...)` callsites in `ensemble_preprocessing.py`, `ga_preprocessing.py`, `interactive.py`, `interactive_gui.py`, `nsga2_search.py`, `unified_bayesian.py`, `preprocessing_wrapper.py`, `preprocessing_discovery.py`, `contamination.py`** — that's a follow-up ticket. Each call site needs its own analysis (does it have access to the wavelength axis at that point in the call graph?). Including it here would balloon T-21 from 1–2 hr to ~6 hr.
- **GUI exposure of `strict=True` "publication mode"** — left for a future ticket. Today, the helper supports it for programmatic callers but no UI surfaces it.
- **Auto-resampling fallback** — explicitly rejected per Decision 2. Resampling is a user choice, not a library default.
- **Tolerance configurability via env var or config file** — overkill for v1. The constant lives at module top so a future patch can expose it if real users hit a false positive.
- **Integration into baseline correction, GLSW, EPO, calibration transfer** — those have their own spacing assumptions, but each is its own analysis. The helper is built for reuse and they'll be follow-ups.
- **Detection of `wavelengths` from `X.columns` automatically** — too magical; would couple the transformer to pandas. Caller is responsible for passing wavelengths explicitly.

---

## Open questions for reviewer

1. Is `0.01` the right tolerance, or should we go tighter (e.g., `0.005`)? My case for `0.01`: float round-trip noise on `1e7/x` produces `cv ~ 1e-7`, instrument-side rounding in stored ASD/OPUS axis vectors goes up to `~3e-4`, single-pixel dropouts give `cv ~ 0.7`. Real concat cases are `>= 0.1`. So `0.005` would also work. Anything below `0.001` would false-positive on round-tripped grids.
2. Is `UserWarning` the right category, or should it be a domain-specific subclass (e.g., `class SpectralPredictWarning(UserWarning)`)? The latter would let users selectively filter spectral-predict warnings without filtering all `UserWarning`s. Current plan uses `UserWarning` for simplicity; a custom subclass is a clean follow-up if anyone asks.
3. Should we also guard against **monotonicity** (i.e., reject grids that aren't strictly ascending or strictly descending)? scipy's `savgol_filter` doesn't care about monotonicity (it only operates on indices), but a non-monotonic grid is *almost certainly* a data-loading bug rather than a real measurement. Punted to follow-up — it's a different failure mode (data integrity) than this ticket (sampling assumption).
4. Is the warning text too long for the GUI log tab? It's ~7 lines. If it spams the UI, consider truncating or providing a link to a docs anchor instead.
