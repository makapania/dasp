# T-26 validation: SNV near-zero std tolerance threshold

**Branch:** `fix/T26-snv-near-zero-std` (HEAD `1fd9d2c`)
**Status:** REJECT_AS_IS — the underlying concern is real but the fix design does not match
how leading commercial chemometrics software handles this case
**Author:** Opus 4.7, 2026-04-30
**Verdict:** REJECT for merge in current form. Recommend rescope to match PLS_Toolbox /
SIMCA design pattern.

---

## TL;DR / what changed in this validation

A first-pass version of this note recommended APPROVED. After the user pushed back ("make
sure that is not the way leading programs do it though"), web research into Eigenvector's
PLS_Toolbox documentation, SIMCA application notes, and the chemometrics literature
established that **T-26's design pattern is not what leading programs do**. The right
approach exists in the field already and is different from what T-26 implements.

This is exactly the kind of correction the chemometrics master rule exists to catch.

## 1. What the leading programs actually do

**PLS_Toolbox / Solo (Eigenvector Research)** — `snv()` is implemented as `auto()` applied
to the transpose. From the `auto()` documentation:

> "The optional input offset is added to the standard deviations before scaling and can be
> used to suppress low-level variables that would otherwise have standard deviations near
> zero."
>
> "Autoscaling includes an adjustable parameter 'offset' which is added to each column's
> standard deviation prior to scaling. When offset is non-zero, the diagonal of the matrix
> S is equal to (s + offset)^-1 where s is the vector of standard deviations for each column
> of X."
>
> Default `offset = 0`.
>
> "A setting near the expected noise level (in the variables' units) is a good
> approximation."

So the formula is:

```
divisor = std + offset                # additive, NOT a threshold
output  = (X - mean(X)) / divisor
```

PLS_Toolbox also exposes a separate `badreplacement` parameter for the **exact-zero std**
case (default 0 = "any value in given variable is set to zero. Variable is effectively
excluded").

**SIMCA / Sartorius** — application notes describe the same offset concept:
"A user-definable offset can be used to avoid over-normalizing samples which have near zero
standard deviation." Same continuous additive offset, user-controlled.

**The R `prospectr`, `rnirs`, `rchemo`, `spectacles`, `inspectr` packages** — all expose SNV
without near-zero protection by default; degenerate spectra are the user's responsibility
to handle upstream. `spectacles` and `inspectr` additionally implement **RNV (Robust Normal
Variate; Guo, Wu & Massart, 1999)** as a separate preprocessing function.

**RNV (Guo et al., 1999, Anal. Chim. Acta 382:87–103)** — the literature-defined robust
alternative for SNV's closure / near-constant-spectrum problems. Uses the 10th percentile
in the numerator and the interquartile range in the denominator instead of mean and std.
This is a methodology-level fix, not a numerical guard.

## 2. What T-26 actually does (and how it differs)

**T-26 design:**

```python
_SNV_STD_FLOOR = 1e-12               # hardcoded
flagged_mask = stds < _SNV_STD_FLOOR
stds = np.where(flagged_mask, 1.0, stds)   # discrete replacement
```

**Differences from leading-program practice:**

| dimension                | PLS_Toolbox / SIMCA               | T-26 as implemented              |
|--------------------------|-----------------------------------|----------------------------------|
| Threshold mechanism      | Continuous additive offset        | Discrete threshold + replace     |
| Formula                  | `(X − mean) / (std + offset)`     | `(X − mean) / 1.0` if flagged    |
| User control             | `offset` parameter (default 0)    | Hardcoded `_SNV_STD_FLOOR`       |
| Default behavior         | Pure Barnes 1989 (offset=0)       | Slightly modified Barnes 1989    |
| Exact-zero std handling  | `badreplacement` param (default 0)| Same code path as near-zero     |
| Continuity               | Smooth in `std`                   | Discontinuous at threshold       |
| Documentation guidance   | "Set offset to expected noise"    | None — user has no knob          |

T-26 is dasp inventing its own pattern that does not exist in PLS_Toolbox, SIMCA, or any
leading R/Python chemometrics package I could verify. **This violates the chemometrics
master rule's explicit corollary: "If we are doing things that software costing tens of
thousands of dollars does not do, it should be a red flag."**

## 3. Was the underlying concern even real?

The concern T-26 set out to address was: rows with `std ∈ (0, 1e-12)` get divided by tiny
std and produce unit-normalized round-off noise that looks like real spectra to downstream
models.

This is empirically true on main (verified — see Section 4 below). But:

- **PLS_Toolbox at default `offset = 0`** has the same behavior as dasp main on near-zero
  std. PLS_Toolbox does *not* automatically protect against this. Users are expected to set
  `offset = noise_level` if they have such data.
- **SIMCA** also exposes the parameter; default is unprotected.
- **The R chemometrics ecosystem** also doesn't protect by default.

So leading programs treat "what to do with near-constant spectra" as a **user
responsibility**, not a library responsibility. The library's job is to provide the knob.
The user's job is to know whether their data needs the knob turned on.

T-26's "automatic protection at hardcoded threshold" is paternalistic relative to that
convention. It silently changes behavior on near-zero data without giving the user control.

## 4. Empirical demonstration of the bug on main (still valid)

Setup:

```python
X = np.full((2, 100), 1.234)
X[0, 0] += 1e-15
X[0, 50] -= 1e-15
X[1, 10] += 2e-15
out = SNV().fit_transform(X)
```

Main (dasp HEAD) produces:

```
input row stds: [2.72e-16  3.13e-16]
output max abs: 7.09
output mean per row: [0.82, 0.77]
output std  per row: [0.58, 0.63]
```

I.e. round-off scaled to look like a real spectrum. **This matches PLS_Toolbox at default
`offset = 0`.** It is not a bug per the field's convention — it is the documented default
behavior, and the field's documented remedy is "set offset to your noise level."

## 5. Canonical reference

**Barnes, R. J.; Dhanoa, M. S.; Lister, S. J.** (1989). _Standard Normal Variate
Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra._ Applied
Spectroscopy 43(5), 772–777. — defines `(x − mean) / std`, no behavior for `std ≈ 0`.

**Guo, Q.; Wu, W.; Massart, D. L.** (1999). _The robust normal variate transform for
pattern recognition with near-infrared data._ Analytica Chimica Acta 382:87–103. —
literature-defined robust alternative using percentile/IQR.

**Eigenvector Research, PLS_Toolbox documentation** — `auto()` and `snv()` functions, with
`offset` parameter and `badreplacement` parameter.

## 6. Verdict

**REJECT for merge in current form.**

Not because the underlying concern is fake — main does produce unit-normalized round-off
on near-constant spectra. But because:

1. **The fix design does not match leading-program practice.** PLS_Toolbox / SIMCA / R
   chemometrics packages all use a continuous user-controlled `offset` parameter, not a
   hardcoded threshold.
2. **The fix removes user control where leading programs provide it.** Users cannot tune
   `_SNV_STD_FLOOR` per-dataset based on their actual noise floor.
3. **The default behavior change is paternalistic relative to field convention.** Leading
   programs default to no protection and trust the user to enable it when needed.
4. **There is a literature-defined methodology alternative (RNV; Guo 1999) for the same
   problem class** that does not appear in dasp at all and would be the more
   chemometrics-native solution if defensive behavior is wanted.

## 7. Recommended path forward

Three options, in order of master-rule alignment:

**Option A — Match PLS_Toolbox `offset` parameter (preferred).** Replace T-26 with:

```python
class SNV(BaseEstimator, TransformerMixin):
    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)

    def transform(self, X):
        X = np.asarray(X)
        means = X.mean(axis=1, keepdims=True)
        stds  = X.std(axis=1, keepdims=True)
        # Eigenvector PLS_Toolbox auto() formula: divisor = std + offset
        return (X - means) / (stds + self.offset)
```

- Default `offset = 0` exactly preserves current main behavior (Barnes 1989, no behavior
  change for any user not opting in).
- Users with noisy / occasionally-flat data set `offset` to their absorbance noise floor
  (e.g., 1e-4) and get bounded output for near-flat rows.
- Matches PLS_Toolbox / SIMCA / Eigenvector convention exactly.
- GUI exposure: one optional float field on the SNV preprocessing block.
- For the exact-zero-std case, optionally add a `badreplacement` parameter mirroring
  PLS_Toolbox's, or keep dasp main's existing `stds[stds == 0] = 1.0` (which produces
  zeros — matches PLS_Toolbox `badreplacement=0` default of "exclude the row").

**Option B — Add RNV as a separate preprocessing option (orthogonal to A).** Implement
Guo 1999 RNV as `RNV` transformer alongside `SNV`. This is the literature-defined robust
alternative for closure / near-constant-spectrum problems. Independent of A — they could
both be added.

**Option C — Drop T-26 entirely with no replacement.** Defensible. Most leading-program
users work around degenerate spectra via QC at import time, not preprocessing-layer guards.
dasp main's existing exact-zero handling already matches PLS_Toolbox's `badreplacement=0`
default.

## 8. Tickets to file (proposed)

If the user accepts this rescope:

1. **T-26 (rescoped):** Add `offset: float = 0.0` parameter to SNV, formula `(X − mean) /
   (std + offset)`, default preserves Barnes 1989. GUI field. Documentation note about
   recommended values. (Closes T-26 with field-aligned design.)
2. **T-26b (new, optional):** Implement RNV (Guo 1999) as separate `RNV` preprocessing
   transformer. Adds a literature-defined robust alternative.
3. **Drop the existing fix/T26-snv-near-zero-std branch.** The hardcoded `_SNV_STD_FLOOR`
   approach should not be merged. The branch ref can be kept for historical reference but
   the design needs to be replaced, not iterated.

## 9. What this validation taught me

The first-pass version of this note approved T-26 by appealing to "universal numerical-
computing practice" (scipy/numpy/sklearn pattern) without verifying that the field's
leading programs actually adopt that pattern. They don't. They use a different, more
principled, user-controlled approach. This is exactly the failure mode the user warned
about and the master rule exists to prevent: importing generic ML / scientific-computing
instincts into a domain that has its own conventions.

The reviewer (the user) caught it by asking the right question — "is that the way leading
programs do it?" — and the answer was no. Future bugfix validations should verify
leading-program behavior **before** drafting the verdict, not after.

## Sources

- [Eigenvector PLS_Toolbox `snv()` documentation](https://www.eigenvectordocs.com/index.php?title=Snv)
- [Eigenvector `auto()` function documentation (offset and badreplacement parameters)](https://www.software.eigenvector.com/docarchive/v42/auto.html)
- [Eigenvector Advanced Preprocessing: Sample Normalization](https://www.eigenvectordocs.com/index.php?title=Advanced_Preprocessing%3A_Sample_Normalization)
- [Sartorius/SIMCA preprocessing app note](https://www.sartorius.com/download/545668/simca-appnote3-spectroscopydata-en-b-00061-sartorius-data.pdf)
- [Guo, Wu & Massart 1999 — RNV original paper](https://www.sciencedirect.com/science/article/abs/pii/S0003267098007375)
- [R `spectacles` package — SNV and RNV](https://search.r-project.org/CRAN/refmans/spectacles/html/snv.html)
- [Barnes, Dhanoa & Lister 1989 — original SNV paper](https://journals.sagepub.com/doi/10.1366/0003702894202201)
