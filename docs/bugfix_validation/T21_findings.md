# T-21 findings: Savitzky-Golay wavelength uniformity guard

**Branch:** none — no implementation exists. Only the plan at
`docs/plans/2026-04-29-T21-sg-wavelength-uniformity-guard.md` (committed `d8c46f0`).
**Status:** FINDINGS ONLY — verdict pending main agent.
**Author:** Opus 4.7 (1M context), 2026-04-30.

T-21 is structurally different from T-05/T-07/T-10/T-26 — there is no `fix/T21` branch.
The original Codex review wedged for an hour and never produced output, so what we have to
validate is **the plan itself**, not a code diff.

The validation question is therefore:

1. Is non-uniform spacing actually a real chemometrics correctness issue (vs. a sklearn-style
   theoretical-concerns-only flag)?
2. Does the proposed plan match how the field actually handles this, or does it invent
   something more aggressive / less aggressive than precedent?
3. Is the bug actually reachable in the user's typical workflow (ASD NIR + OPUS FTIR for
   bone collagen / paleoanthropology), or is it solving a theoretical problem nobody hits?
4. Is the plan's literature framing accurate? (Two specific claims need verification —
   PLS_Toolbox `gridcheck` and OpenSpecy `is_evenly_spaced()`. Both turned out to be more
   tenuous than the plan implies.)

---

## 1. What `main` currently does for SG

### 1a. The two SG transformers (preprocess.py:48-220)

`SavgolDerivative` (preprocess.py:48-134) and `SavgolSmooth` (preprocess.py:137-220) are
thin wrappers around `scipy.signal.savgol_filter` with auto-window-clamp logic. **Neither
class accepts `wavelengths`. Neither validates uniform spacing. There is no guard
anywhere in the codebase.**

`SavgolDerivative.transform()` (excerpt, preprocess.py:75-134):

```python
def transform(self, X):
    X = np.asarray(X)

    # Ensure odd window
    window = self.window
    if window % 2 == 0:
        window = window + 1

    # Default polyorder ...
    # Auto-adjust window if too large for feature count ...
    # Validate polyorder constraint ...

    X_deriv = savgol_filter(
        X, window_length=window, polyorder=polyorder, deriv=self.deriv, axis=1
    )
    return X_deriv
```

`scipy.signal.savgol_filter` is called with the default `delta=1.0` (no axis information at
all — it operates purely on indices). On non-uniform input, scipy silently returns
*index-space* derivatives; whether those are correct depends on whether the caller treats
them as derivative-with-respect-to-index (always correct) or derivative-with-respect-to-
wavelength (only correct on uniform grids).

Within dasp the output is consumed by downstream models and wavelength-aware
plotting/peak-attribution code, which interprets the values as derivatives in
wavelength/wavenumber space. So on a non-uniform grid the values are **misinterpreted**
relative to their true definition — the silent-bug claim is real.

### 1b. No existing uniformity check anywhere

`Grep` for `uniform`, `evenly_spaced`, `gridcheck`, `np.diff.*std` across the entire codebase
turns up no spacing-validation logic. No reader resamples to uniform. No user-warning is
emitted today on non-uniform input.

---

## 2. Where SG gets called from

### 2a. Call-site count

`SavgolDerivative(...)` and `SavgolSmooth(...)` are instantiated **400 times across 22
files**. Among production code (excluding tests/docs/codex_reviews):

| File | Count |
|------|-------|
| `spectral_predict_gui_optimized.py` | 36 (e.g. `_build_transform_from_config` at lines 1908, 1941–1971) |
| `src/spectral_predict/ensemble_preprocessing.py` | 28 |
| `src/spectral_predict/ga_preprocessing.py` | 12 |
| `src/spectral_predict/preprocess.py` (transformer + pipeline builder) | 6 |
| `src/spectral_predict/preprocessing_discovery.py` | 3 |
| `src/spectral_predict/unified_bayesian.py` | 2 |
| `src/spectral_predict/interactive.py` | 2 |
| `src/spectral_predict/interactive_gui.py` | 2 |
| `src/spectral_predict/preprocessing_wrapper.py` | 1 |
| `src/spectral_predict/nsga2_search.py` | 1 |

**None of these 87+ production call sites pass `wavelengths`** today (the constructors
don't accept the kwarg yet, by design).

### 2b. The pipeline path that *does* take `wavelengths`

`build_preprocessing_pipeline()` at `preprocess.py:279` already accepts `wavelengths=None`
(currently used only for the `WavelengthExcluder` / interference-removal branch at
`preprocess.py:355`). The SG branches (`:534, :538, :543`) and the smoothing step at
`:521` do **not** thread `wavelengths` to the SG constructors today.

### 2c. The plan's claim about `search.py` plumbing

The plan states (in "Existing Spectral Predict context"):

> `wavelengths` is already plumbed from `search.py` callers (e.g., `search.py:612, 1958`).
> They pass `X.columns.values` when the input is a DataFrame.

**This is partly false.** I verified all 5 `build_preprocessing_pipeline` call sites in
`search.py`:

| Line | Passes wavelengths? | Path |
|------|---------------------|------|
| `search.py:612` | **NO** | Per-fold preprocessing inside CV — the main hot path |
| `search.py:1968` | YES | Pre-CV full-spectrum preprocessing (Phase 3 interference) |
| `search.py:3552` | YES | Same pattern as 1968 (alt code path) |
| `search.py:4200` | YES | Same pattern (alt code path) |
| `search.py:5397, 5613` | **NO** | One-class CV preprocessing |

So the plan's claim that `wavelengths` is already plumbed at `:612` is wrong — that's the
main per-fold path, where wavelengths is **not** passed today. T-21 would need to fix
`:612, :5397, :5613` as part of its scope, or accept that the guard simply doesn't fire on
the main hot path.

### 2d. The plan's "non-goal" — 30+ unguarded callsites

The plan explicitly carves out the 30+ direct `SavgolDerivative(...)` instantiations in
`ensemble_preprocessing.py`, `ga_preprocessing.py`, `interactive.py`, `interactive_gui.py`,
`nsga2_search.py`, `unified_bayesian.py`, `preprocessing_wrapper.py`,
`preprocessing_discovery.py`, `contamination.py` as a **follow-up ticket**. So in the
T-21-as-planned end state, **the guard fires only on the `build_preprocessing_pipeline()`
path (and only when the caller passes `wavelengths`)** — i.e., exactly 3 of 5 search.py call
sites + the GUI's preview/interactive paths if they happen to use the pipeline builder.

A user who runs an Ensemble or GA preprocessing search (which goes through
`ensemble_preprocessing.py` / `ga_preprocessing.py`) gets **no guard at all**. A user who
clicks any preprocessing preview button in the GUI (which goes through
`_build_transform_from_config` at `gui:1893`) also gets **no guard at all** — because that
helper instantiates SG directly and never sees wavelengths.

This is the largest gap between the plan's framing (silent bug fixed) and the plan's
delivery (silent bug fixed only on a subset of paths).

### 2e. GUI choice of SG window/polyorder is wavelength-agnostic

The GUI exposes `Window` and `Poly` spinboxes (gui:1486-1488). The chosen values are passed
straight to `SavgolDerivative(window=...)` without any consideration of the data's actual
spacing. Window of 7 means "7 sample indices wide", which is 7 nm on uniform 1-nm ASD data,
~28 cm⁻¹ on uniform 4-cm⁻¹ OPUS data, and **mathematically meaningless** on a
post-conversion non-uniform grid.

---

## 3. What the plan actually proposes

### 3a. Implementation sketch (preprocess.py, +1 helper, +2 kwargs, ~50 LOC)

- Module-level `_validate_uniform_grid(wavelengths, tolerance=0.01, strict=False, context=...)`
  helper computing `cv = np.diff(wl).std() / abs(np.diff(wl).mean())` and warning (or
  raising in strict mode) if `cv > 0.01`.
- New `wavelengths=None` kwarg on `SavgolDerivative.__init__` and `SavgolSmooth.__init__`,
  stored on `self`. Guard called at the top of each `transform()`.
- `build_preprocessing_pipeline()` threads its existing `wavelengths` arg into the SG and
  smoothing constructors.

### 3b. Tolerance: 0.01 (1% CV)

Plan defends 0.01 as "the conventional chemometrics threshold for 'effectively uniform'"
and cites OpenSpecy and PLS_Toolbox as precedents. **Both citations are weak — see
section 5 below.** The empirical bracketing in the plan (0.001 too tight, 0.05 too
permissive, 0.01 admits float-precision noise on round-tripped axes) is technically
sensible and consistent with what one would compute, but it is dasp inventing the
tolerance, not citing it.

### 3c. Behavior: warn-and-proceed (UserWarning), not raise

This matches the field's general precedent — neither prospectr nor scipy raise. PLS_Toolbox
and Unscrambler haven't been verifiable on this point (see section 5).

### 3d. Strict-mode opt-in

`strict=True` raises `ValueError`. **Not exposed in the GUI in this ticket.** Per the T-26
post-mortem rule ("backend-only knobs are dead code in the bundled-app distribution model"),
this part of the API delivers zero value to actual users today. It's a defensible
forward-compat hook, but it is dead code by the same standard that killed T-26.

---

## 4. Canonical reference: does Savitzky-Golay 1964 require uniform sampling?

### 4a. Savitzky & Golay 1964 — uniform-spacing assumption is canonical

Savitzky, A.; Golay, M. J. E. (1964). _Smoothing and Differentiation of Data by
Simplified Least Squares Procedures._ Anal. Chem. **36**(8):1627–1639.

The full text PDF (`https://agora.cs.wcu.edu/~huffman/figures/sgpaper1964.pdf`) was not
machine-extractable in this session, but the uniform-spacing assumption is universally
acknowledged in secondary sources. Wikipedia summarizes the derivation as:

> "When the data points are equally spaced, an analytical solution to the least-squares
> equations can be found, in the form of a single set of 'convolution coefficients' that
> can be applied to all data sub-sets, to give estimates of the smoothed signal."

This is the key property that makes S-G computationally efficient: a single fixed-stride
kernel is precomputed once and convolved across the spectrum. The kernel coefficients
**are functions of the polynomial order, window length, and derivative order — but
NOT of the actual x-spacing**. So the operator is implicitly "derivative with respect to
index", and only equals "derivative with respect to wavelength" if the grid is uniform.

### 4b. Steinier, Termonia & Deltour 1972 — same assumption

Steinier, J.; Termonia, Y.; Deltour, J. (1972). _Smoothing and Differentiation of Data by
Simplified Least Squares Procedure._ Anal. Chem. **44**(11):1906–1909. Provides corrections
to S-G 1964's tables of convolution coefficients. Same uniform-spacing assumption.

### 4c. Gorry 1990 — generalizes to any polynomial order, still uniform

Gorry, P. A. (1990). _General Least-Squares Smoothing and Differentiation by the
Convolution (Savitzky-Golay) Method._ Anal. Chem. **62**(6):570–573. Generalizes the
convolution-coefficient computation to arbitrary polynomial / derivative / window
combinations via Gram-polynomial recursion, but still assumes uniformly spaced input.

### 4d. Extensions for non-uniform spacing exist, but are not canonical

Modern works (e.g., Pisarski's `savitzky_golay` implementation; ResearchGate Q&A on
unequally-spaced SG) provide explicit non-uniform-grid generalizations. SciPy issue #19812
(2024) is an open feature request to add such support. **None of these is the canonical
S-G** — they are explicit extensions, and any chemometrics paper / software citing
"Savitzky-Golay" without further qualification means the 1964 uniform-grid version.

**Verdict on canonical reference:** the plan's framing is correct. S-G 1964 assumes
uniform sampling, scipy carries that assumption, and applying scipy's `savgol_filter` to a
non-uniform x-axis silently produces something that is no longer the named operator from
the literature.

---

## 5. Commercial-software / open-source sanity check

This is the section where T-26 was caught the first time around, so I dug specifically into
the plan's two specific load-bearing claims (PLS_Toolbox `gridcheck`, OpenSpecy
`is_evenly_spaced()`).

### 5a. PLS_Toolbox `gridcheck` — the plan's claim is unverified

The plan says (Decision 1, tolerance defense):

> "PLS_Toolbox's `gridcheck` warns at ~1e-2."

I could not verify this. Eigenvector documentation pages I attempted (the `Savgol` wiki
page, the PLS_Toolbox topic index, the Advanced Preprocessing page) either redirected,
returned 403, or contained no `gridcheck` reference. The PLS_Toolbox 7.0 quick-reference
card and the Eigenvector Savgol PDF both describe SG without mentioning a uniform-spacing
guard.

`gridcheck` may exist as an internal utility — Eigenvector's MATLAB function namespace
includes hundreds of helpers — but I have no documented evidence of its tolerance value or
its triggering behavior. **The plan's "1e-2 default" claim is uncited and may be made up.**

What is documented for PLS_Toolbox SG (Eigenvector wiki "Savgol"):

> "The Savitzky-Golay (SavGol) algorithm essentially fits individual polynomials to windows
> around each point in the spectrum, and these polynomials are then used to smooth the
> data."

No mention of uniformity validation in the user-visible documentation.

### 5b. OpenSpecy `is_evenly_spaced()` — the plan's claim is *false*

The plan says (Decision 1, tolerance defense):

> "the OpenSpecy `is_evenly_spaced()` helper uses 1e-2 by default"

**There is no `is_evenly_spaced()` function in OpenSpecy.** I fetched the GitHub repo
(`wincowgerDEV/OpenSpecy-package`), listed `R/`, fetched `conform_spec.R` and
`smooth_intens.R`, and found:

- OpenSpecy's strategy is **proactive resampling**, not validation. `conform_spec()`
  interpolates (or rolls / mean-aggregates) input spectra onto a user-specified uniform
  wavenumber grid via `stats::approx()`. The default in the wrapper `process_spec()` is
  `conform_spec = TRUE` with `res = 5` cm⁻¹, so the ~standard usage path always produces
  uniform input downstream.
- `smooth_intens()` (the SG / Whittaker-Henderson wrapper) does **not** itself validate
  spacing — it relies on `conform_spec()` having been run.
- The function name `is_evenly_spaced` does not appear in the source.

**This means the plan's headline tolerance precedent is fabricated.** The plan's
underlying argument (1e-2 admits float noise, rejects multi-instrument splices) is still
internally defensible on numerical grounds, but the OpenSpecy attribution is not.

### 5c. prospectr (R, NIR-focused) — documents requirement, no enforcement

prospectr `savitzkyGolay()` documentation says explicitly:

> "It requires evenly spaced data points."

The function takes a `delta.wav` argument used for "scaling and getting numerically correct
derivatives", but it does **not validate** that the input is evenly spaced — it trusts the
user. (Verified by WebFetch on the prospectr documentation page.) This is the
chemometrics-Python convention: document the requirement, trust the user.

### 5d. scipy.signal.savgol_filter — assumption documented in issues, not in the function

scipy's main `savgol_filter` documentation does not warn about non-uniform input. The open
GitHub issue #19812 ("ENH: Savitsky-Golay filter for non uniformly spaced data") confirms
in the discussion that the current implementation **assumes uniform spacing** by design
("limited to uniformly spaced independent data... reduces to a single set of coefficients
and very efficient folding operation"). scipy therefore does not warn on non-uniform
input — same as prospectr.

### 5e. Unscrambler / OPUS / SIMCA — undocumented in public sources

I could not find authoritative documentation that any of Unscrambler (CAMO/Aspen), OPUS
(Bruker), or SIMCA (Sartorius) validates uniform spacing for SG. The OPUS reference manual
mentions derivative as a preprocessing step but doesn't discuss spacing. Eigenvector's
Savgol white paper similarly discusses parameter selection (window, polyorder) but not
uniform-spacing validation.

The most likely real-world pattern across commercial software: **the format readers write
uniform grids by construction** (FTIR instruments produce uniform-cm⁻¹ output natively;
NIR instruments produce uniform-nm), so a uniformity check has rarely been needed because
the input is rarely non-uniform — until the user does an axis conversion or splices
multi-instrument data.

### 5f. Summary: precedent for warn-and-proceed exists, not for raise

| Tool | Documents requirement? | Validates? | Warn / raise? |
|------|------------------------|------------|---------------|
| Savitzky & Golay 1964 | Yes (implicit) | n/a | n/a |
| scipy.signal.savgol_filter | Mentioned in issue #19812 only | No | Silent |
| prospectr (R) | Yes ("requires evenly spaced") | No | Silent |
| OpenSpecy (R) | n/a — auto-resamples to uniform | n/a | Proactive `conform_spec()` |
| PLS_Toolbox | Possibly via `gridcheck` (unverified) | Possibly | Possibly warn |
| Unscrambler | Undocumented in public sources | Unknown | Unknown |
| OPUS | Undocumented in public sources | Unknown | Unknown |

**Consensus precedent:** trust the user (scipy, prospectr) OR proactively resample
(OpenSpecy). **No tool in the verified list "warns and proceeds with a UserWarning at
1% CV"** the way the plan proposes. The behavior is internally defensible on chemometrics
grounds (S-G's uniform-spacing assumption is real), but it is dasp inventing a warning
pattern, not aligning with documented commercial precedent.

---

## 6. Codebase reality check: how often is dasp's input non-uniform?

### 6a. File-format readers all produce uniform-by-construction output

| Reader | Spacing | Verified |
|--------|---------|----------|
| `read_asd_dir` (ASD .asd binary, `io.py:645`) | Uniform 1 nm @ 350-2500 nm | Plan + ASD spec |
| `read_opus_file` / `read_opus_dir` (`readers/opus_reader.py`) | Uniform cm⁻¹ (instrument-determined: 1, 2, 4, 8, or 16 cm⁻¹ per Bruker resolution settings) | `opus_reader.py:299` sorts ascending wavenumber, line 310 checks strictly increasing |
| `read_jcamp_dir` / `read_jcamp_file` | Uniform if file uses `##DELTAX=`, else as-stored | JCAMP-DX standard supports both uniform and non-uniform |
| `read_spc_dir` / `read_spc_file` | Uniform if file is `XEvenSpacing` (most common), else explicit x-vector | Galactic SPC standard |
| `read_csv_spectra` / `read_excel_spectra` / `read_combined_csv` | Whatever the user-supplied columns are | User-controlled |

**dasp does not resample to uniform anywhere.** No `interp1d`, no `np.linspace`-redefinition
in any reader. The grid that comes out of the reader is the grid that goes into SG.

### 6b. Explore tab does NOT show non-uniform-spacing warnings today

`Grep` for spacing-warning logic in the Explore tab and the Diagnostics PCA path returns
nothing. Today a user can load a non-uniform CSV, run SG derivative previews, and never see
any indication of the problem.

### 6c. The unit-conversion case — the strongest argument for the guard

Per `MEMORY.md` and gui:19016 (`_convert_x_unit_and_replot`):

> "This converts column headers (wavelength values) using `1e7/x`, re-sorts columns
> (since the conversion inverts order), swaps wavelength filter min/max..."

`io.py:31` `convert_x_axis(columns, from_unit, to_unit)`:

```python
if {from_unit, to_unit} == {'nm', 'cm-1'}:
    return 1e7 / columns
```

This is a **value transformation, not a resampling**. After conversion, the grid is **non-
uniform in the new unit by construction** (because `1e7/x` is a non-linear map):

- A uniform 1 nm grid 400–2500 nm becomes cm⁻¹ values `1e7/400=25000`, `1e7/401=24938.9`,
  `1e7/402=24875.6`, …, `1e7/2500=4000`. Spacing varies from ~0.0006 cm⁻¹ at the high-nm end
  to ~62 cm⁻¹ at the low-nm end. cv ≈ 1.0+ — extreme non-uniformity.
- A uniform 4 cm⁻¹ grid 4000–400 cm⁻¹ becomes nm values `1e7/4000=2500`, `1e7/3996=2502.5`,
  …, `1e7/400=25000`. Similar story.

So the moment a user clicks "Convert x-unit" in the Import tab and then runs any SG
preprocessing (preprocess preview, predictor screening derivative subtab, model search with
deriv preprocessing), **the SG derivative is computed against a strongly non-uniform grid
and is silently wrong** in the new unit. The derivatives still look reasonable visually
(oscillation around zero, peak shape), so the user has no signal that anything is wrong.

This is the strongest single justification for the guard. It is reachable from the GUI in
exactly two clicks (Convert x-unit, then Run Search with `deriv1` preprocess).

### 6d. The mixed-instrument-splice case — possible but rarer

A user who concatenates ASD nm + OPUS cm⁻¹-converted-to-nm in a combined CSV would get a
strongly non-uniform grid. Possible but probably not the typical bone-FTIR / paleo-isotope
workflow.

### 6e. The ASD-direct case — uniform, no issue

The user's typical bone-FTIR workflow (per `MEMORY.md` and `docs/PROJECT_STATUS.md`) is
likely OPUS FTIR in cm⁻¹ or ASD VNIR/SWIR in nm, both **uniform by construction**. No
guard would fire on that path.

---

## 7. User-distribution-model check (T-26 lesson)

dasp ships as a bundled Inno Setup desktop GUI to non-technical users (paleoanthropology,
bone diagenesis, isotope work). Per `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md`
caveat 5:

> "dasp ships as a bundled Inno Setup desktop app to non-technical users — there is no
> Python REPL. Backend-only knobs not reachable from the GUI are dead code for the actual
> user base."

### 7a. The warn-and-proceed default would surface to the GUI user

Python's default warning filter writes `UserWarning`s to stderr, which dasp's GUI
log-redirect (search for `sys.stderr` redirection) routes to the run log. So a user running
a search with non-uniform spacing **would see** something like:

```
UserWarning: Non-uniform wavelength grid detected for Savitzky-Golay derivative ...
```

This is reachable user value — assuming the warning text is read.

### 7b. The strict=True hook is dead code by the T-26 standard

The plan adds `strict=True` for "future publication-mode CLI flag". There is no GUI
checkbox for it. Per the T-26 lesson: **users cannot reach `strict=True` from the bundled
desktop app**. It would be GUI-plumbing follow-up work that nobody has filed yet.

This isn't a fatal flaw — the warn-default path is the value-delivering path, and the
strict hook is a 2-line addition that doesn't cost anything to leave in. But the plan's
description of `strict=True` as "left for a future ticket" is mildly euphemistic; per the
T-26 rule, it's dead-code-with-a-future-ticket.

### 7c. User recourse if the warning fires

Plan offers: a copy-pasteable `scipy.interpolate.interp1d` resampling snippet in the
warning text. **For the bundled GUI user, this snippet is not actionable** — the user
cannot paste Python code anywhere. There is no GUI button to "resample to uniform grid",
no plumbing for `scipy.interpolate.interp1d` exposed through the UI.

So the warning's actionability for the actual user base is roughly: "your data is
non-uniform; the math is broken; we cannot help you fix it from this UI." That's not
nothing — at least the user knows. But it's much less than the warning text implies.

A natural T-21 follow-up that would deliver real GUI value: a "Resample to uniform grid"
button on the Import tab, plumbed to `scipy.interpolate.interp1d` with default cubic /
linear options. That would make the guard's warning into an actionable signal. **The
current plan does not include this.**

---

## 8. Distinct cases worth flagging separately

| Case | Reachable today? | Guard fires? | Severity |
|------|------------------|--------------|----------|
| ASD direct (nm, uniform) | Yes (typical) | No (cv ≈ 0) | None — guard is silent |
| OPUS direct (cm⁻¹, uniform) | Yes (typical) | No (cv ≈ 0) | None — guard is silent |
| OPUS direct + SG (no conversion) | Yes (typical) | No | None |
| OPUS converted to nm + SG | **Yes — 2 GUI clicks** | Yes (cv ≈ 1.0) | **HIGH — silently-wrong derivatives** |
| ASD converted to cm⁻¹ + SG | **Yes — 2 GUI clicks** | Yes (cv ≈ 1.0) | **HIGH — silently-wrong derivatives** |
| ASD + OPUS concatenated CSV | Possible | Yes (cv > 0.4) | Medium — silently-wrong derivatives |
| JCAMP file with non-uniform native spacing | Possible (rare) | Yes | Medium |
| CSV from non-spectrometer source | Possible (rare) | Yes | Medium |
| Float-noise round-tripped grid | Possible | No (cv < 1e-4) | None — guard correctly silent |
| Single-pixel dropout (NaN-filtered column) | Possible | Yes (cv ≈ 0.7) | Medium |

The "OPUS converted to nm + SG" and "ASD converted to cm⁻¹ + SG" cases are the strongest
argument for the guard. They are reachable in 2 clicks from the GUI's main Import tab.
Per `MEMORY.md`, the unit-conversion feature was added in 2026-02-15 and the ~37 hardcoded
xlabels were updated to use it, suggesting it's a real workflow the user values. If the
user has converted units even once and run SG once, they have hit this bug — and they
have no warning today.

---

## 9. Is there a real-world dataset where this bug bites the user today?

Best honest answer: **probably yes, but I cannot verify which datasets without more
session context**. The user's typical data is OPUS FTIR (uniform cm⁻¹) — direct, no
problem. The user's other typical data is ASD NIR (uniform nm) — direct, no problem.

The bug bites if and only if:

1. The user has used the unit-conversion feature (`_convert_x_unit_and_replot`,
   added 2026-02-15) on real data, AND
2. After conversion, ran SG-derivative preprocessing through any path.

`docs/SESSION_LOG.md` and `MEMORY.md` mention the unit-conversion feature as a substantial
2026-02-15 addition, but I see no log entry confirming the user has actually used it on
real bone-FTIR data, nor any session entry about derivative artefacts. The recurring
failure mode (per `MEMORY.md` user preferences) is "agents repeatedly false-flag chemometrics
standards as leakage" — i.e., theoretical concerns about reachable-but-unhit code paths.

The honest version: this is real chemometrics correctness (S-G assumption is well-defined,
violation produces wrong values), the guard is reachable from the GUI (unit conversion +
SG), but I cannot prove it has bitten this specific user on a specific dataset. It's
probabilistic-real, not session-log-confirmed-real.

---

## 10. The plan's own internal weaknesses

Independent of merit:

### 10a. "wavelengths plumbed at search.py:612" is wrong

(Section 2c). Three of the five `build_preprocessing_pipeline` call sites in `search.py`
do pass `wavelengths`, but the per-fold and one-class call sites (`:612, :5397, :5613`) do
not. T-21 as planned would not guard those paths. Either fix as part of T-21 or accept the
gap explicitly.

### 10b. "30+ direct callsites are a follow-up" is large gap

(Section 2d). `ensemble_preprocessing.py`, `ga_preprocessing.py`, `interactive.py`,
`interactive_gui.py`, the GUI's `_build_transform_from_config`, and the search.py per-fold
path all bypass the guard. If the goal is "no silent SG bug in dasp", this scope is
insufficient. If the goal is "guard the most common paths and file follow-ups", that's
fine but the framing should change.

### 10c. The OpenSpecy `is_evenly_spaced()` precedent is fabricated

(Section 5b). That function does not exist. OpenSpecy's strategy is proactive interpolation
via `conform_spec()`, not validation. The plan's tolerance defense leans on this citation
without basis.

### 10d. The PLS_Toolbox `gridcheck` precedent is unverified

(Section 5a). May exist; not documented publicly with a tolerance value. The plan's
specific 1e-2 attribution is uncited.

### 10e. Warn-and-proceed is internally defensible on chemometrics grounds

(Section 4). S-G 1964's uniform-spacing assumption is real and load-bearing. So the
underlying concern is not invented — it's the published behavior of the named operator.
That's the strongest part of the plan and is independent of the weak commercial-precedent
citations.

### 10f. The strict=True hook is dead code in the bundled-app distribution

(Section 7b). Cheap to leave in. Worth acknowledging that nobody can reach it from the GUI.

### 10g. The warning text's resampling snippet is not actionable from the GUI

(Section 7c). Real follow-up value would be a GUI "Resample to uniform" button. Not in
this ticket.

### 10h. The plan describes the bug as "silent scientific bug"

The framing is accurate per the chemometrics literature. But the calibration of "silent
scientific bug" with respect to **reachable user impact** is what matters. T-26 was also
"silent numerical bug" but at a corner case the user has never hit. T-21's reachable
scenario (unit-converted data + SG) is more plausible than T-26's, but I cannot prove the
user has actually hit it. The main agent should weight this when deciding APPROVED vs.
DEFER.

---

## 11. Notes for the verdict

I am explicitly NOT writing the verdict. But the questions the main agent will face are
roughly:

1. **Is the underlying concern real per the literature?** YES. S-G 1964's
   uniform-spacing assumption is canonical; scipy preserves it; non-uniform input produces
   wrong derivatives. This is not a sklearn-instinct issue.

2. **Is the bug reachable in the user's GUI workflow?** YES, via unit conversion + SG.
   Not reachable in the typical OPUS-direct or ASD-direct workflow.

3. **Does the plan's design match the field?** PARTIALLY. The warn-and-proceed default at
   ~1% CV is internally defensible but not directly cited from any verified commercial /
   open-source implementation (OpenSpecy claim is wrong; PLS_Toolbox claim is unverified;
   prospectr/scipy don't validate at all). This may be acceptable — dasp can lead the
   field on a scientifically-correct guard nobody else has — or it may be inventing a
   pattern (the T-26 failure mode).

4. **Does the plan deliver real GUI user value?** PARTIALLY. The warn-default surfaces in
   stderr → run log, which is reachable to the user. The `strict=True` hook is dead code
   per the T-26 rule. The resampling-snippet warning text is not GUI-actionable. A natural
   follow-up (GUI "Resample to uniform" button) is not in scope.

5. **Are the plan's specific factual claims accurate?**
   - S-G 1964 uniform-spacing assumption: yes.
   - PLS_Toolbox `gridcheck` at 1e-2: unverified.
   - OpenSpecy `is_evenly_spaced()` at 1e-2: **false** (function does not exist).
   - "wavelengths is already plumbed from search.py:612": **false** (it isn't).
   - "30+ callsites follow-up": correct — and a real coverage gap.

6. **What's the recommendation tree?**
   - APPROVE (warn-only, scope as-planned): accept that the guard fires on roughly half
     the call sites; the half it fires on includes the main `run_search` path on the
     happy path; the user gets a warning when they hit unit-converted data + SG.
   - APPROVE WITH SCOPE EXPANSION: also fix the per-fold (`:612`) and one-class
     (`:5397, :5613`) call sites to pass `wavelengths`. ~30 min of additional work; closes
     the largest gap. Also worth touching `gui:1908,1941-1971`'s
     `_build_transform_from_config` for the preview path.
   - DEFER pending GUI "Resample to uniform" follow-up: ship the guard and the resample
     button together so the warning is actionable. Aligns with T-26-rule
     ("backend-only knobs are dead code"). ~3-4 hr total.
   - REJECT_AS_IS: weak — the underlying concern is real, the field doesn't have a
     better convention, and the warn-and-proceed default doesn't break anything.
   - DROP: hard to defend given the canonical-reference status of S-G 1964 and the
     reachability via unit conversion. This isn't T-26.

---

## 12. Sources

- Savitzky, A.; Golay, M. J. E. (1964). Anal. Chem. 36(8):1627–1639. — canonical reference
- Steinier, Termonia & Deltour (1972). Anal. Chem. 44(11):1906–1909. — corrections, same assumption
- Gorry, P. A. (1990). Anal. Chem. 62(6):570–573. — generalization, same assumption
- [scipy.signal.savgol_filter docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)
- [scipy issue #19812: ENH non-uniform SG](https://github.com/scipy/scipy/issues/19812)
- [prospectr R `savitzkyGolay`](https://search.r-project.org/CRAN/refmans/prospectr/html/savitzkyGolay.html)
- [OpenSpecy `conform_spec.R` source](https://raw.githubusercontent.com/wincowgerDEV/OpenSpecy-package/main/R/conform_spec.R)
- [OpenSpecy `smooth_intens.R` source](https://raw.githubusercontent.com/wincowgerDEV/OpenSpecy-package/main/R/smooth_intens.R)
- [Eigenvector Savgol page (PLS_Toolbox)](https://www.eigenvectordocs.com/index.php?title=Savgol)
- [Eigenvector PLS_Toolbox topic index](https://www.software.eigenvector.com/docarchive/v75/pls_toolbox_topics.html)
- [Wikipedia: Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)
- [NIRPY Research: Savitzky-Golay smoothing method](https://nirpyresearch.com/savitzky-golay-smoothing-method/)
- dasp source files cited inline (`src/spectral_predict/preprocess.py`, `src/spectral_predict/search.py`,
  `src/spectral_predict/io.py`, `src/spectral_predict/readers/opus_reader.py`,
  `spectral_predict_gui_optimized.py`).
