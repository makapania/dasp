# T-07 validation: PDS even-window arithmetic + apply_pds geometry hardening

**Branch:** `fix/T07-pds-even-window` (HEAD `d7cc3c0`)
**Status:** APPROVED for merge (after rebase + strict-test pass)
**Author:** Opus 4.7, 2026-04-30
**Verdict:** REAL BUG. Smaller class than T-05 (loud crash, not silent wrong output) but
field-aligned, internally consistent with dasp's existing GUI/docstring contract, and
delivers value to the un-validated GUI build path.

Investigation findings: [T07_findings.md](T07_findings.md).

---

## TL;DR

`estimate_pds(window=even)` in main crashes with a cryptic numpy `could not broadcast`
traceback (verified for windows 2/4/6/8). `apply_pds(X, B, window=W)` when W disagrees with
`B.shape[1]` crashes with a numpy matmul dim-mismatch. The fix replaces both with clear
`ValueError`/`FutureWarning` messages citing Wang 1991 and derives apply_pds geometry from
`B.shape[1]` (the saved-model's authoritative source).

Bug class is "improve error message + add API robustness" — not "fix wrong numerical
output" like T-05. But the fix is field-aligned, the bug has zero precedent in commercial
chemometrics tools surveyed, and dasp's GUI tooltip + docstring + GUI build path #1
already say "must be odd" — only the backend itself was missing the check. The fix
brings the backend into line with what dasp already documents.

Real-world user impact: GUI build path #1 (`_build_transfer_model`, line 42344-42349)
already enforces odd via messagebox before backend is called, so the dominant user path
is protected today. The fix matters for GUI build path #2 (`_build_transfer_model_new`,
line 44625) which is un-validated, and for any programmatic caller. Lower urgency than
T-05/T-10 but a clean correctness improvement with no downside.

## 1. Canonical reference

**Wang, Y.; Veltkamp, D. J.; Kowalski, B. R.** (1991). _Multivariate Instrument
Standardization._ Analytical Chemistry 63(23), 2750–2756.

PDS construction: for each primary-instrument wavelength `i`, fit local regression
predicting `X_primary[:, i]` from a centered window of satellite channels `i-k` through
`i+k`. The window has full width **2k+1 (always odd)**.

Restated identically in:
- Bouveresse & Massart 1996 (PDS improvements)
- Feudale et al. 2002 (calibration transfer review)
- RNIR R tutorial (half-window MWsize, full = 2[MWsize-1]+1)
- specProc R package `pds.R` (parameter `win` is half-window)
- PLS_Toolbox / Solo (full-window parameter, must be odd)

## 2. Field-alignment check (universal odd-only consensus)

| Source                                  | Even windows allowed?                                  |
|-----------------------------------------|--------------------------------------------------------|
| Wang 1991                               | Not defined; `2k+1` formulation makes even impossible  |
| Bouveresse & Massart 1996               | Same                                                   |
| Feudale 2002 review                     | Same                                                   |
| RNIR R tutorial (Guifh)                 | NO — half-window `MWsize`                              |
| specProc R package (Goueguel)           | NO — half-window `win`, full = `2*win+1`              |
| PLS_Toolbox / Solo (Eigenvector)        | NO — full-window param, odd-required                   |
| **dasp main**                           | **NO per docstring + GUI tooltip + GUI path #1, but backend NOT enforced — bug** |

The investigation found **zero** implementations that accept even windows. Every tool
either makes even impossible by parameterization (half-window) or enforces odd on a
full-window parameter. dasp main's combination of "documented as odd" + "no backend
enforcement" has no precedent.

The branch's enforcement of odd-only at the backend matches the field universally.

## 3. What main does (the bug)

`src/spectral_predict/calibration_transfer.py:170-222`:

```python
def estimate_pds(X_primary, X_satellite, window: int = 11) -> np.ndarray:
    n_samples, p = X_satellite.shape
    half_window = window // 2
    B = np.zeros((p, window))           # allocates `window` columns

    for i in range(p):
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)
        X_window = X_satellite[:, start:end]      # interior width = 2*half+1
        b = np.linalg.lstsq(X_window, X_primary[:, i], rcond=None)[0]
        B[i, offset:offset + len(b)] = b           # writes 2*half+1 elements
```

For odd `window`: `2*half+1 == window`, slice fits. Correct.

For even `window`: `2*half+1 == window+1`, slice is one element wider than B's row.
`ValueError: could not broadcast input array from shape (window+1,) into shape (window,)`.

Empirical (verified on main):

```
window=2: ValueError: could not broadcast (2,) into (1,)
window=4: ValueError: could not broadcast (3,) into (2,)
window=6: ValueError: could not broadcast (4,) into (3,)
window=8: ValueError: could not broadcast (5,) into (4,)
```

Hard crash with cryptic message. User has to understand PDS internals to interpret it.

## 4. What the fix changes

**`estimate_pds`:** Adds two upfront guards. Downstream arithmetic is byte-identical to
main.

```python
if not isinstance(window, (int, np.integer)) or window < 1:
    raise ValueError(
        f"window must be a positive integer (got {window!r}). "
        "PDS uses a centered local-regression window of width 2k+1."
    )
if window % 2 == 0:
    raise ValueError(
        f"window must be odd (got {window}). This implementation uses a "
        "centered, odd-width local window of size 2k+1 — see "
        "Wang, Veltkamp, & Kowalski (1991), 'Multivariate Instrument "
        "Standardization,' Anal. Chem. 63(23), 2750-2756."
    )
```

**`apply_pds`:**

- Signature: `window: int = 11` → `window: int | None = None`.
- Geometry source: derives `half_window = (B.shape[1]) // 2` from the saved-model matrix
  instead of from the caller's `window` argument.
- Validates `B.shape[1]` is odd (rejects stale even-width B from disk).
- Validates `B.shape[0] == X.shape[1]` (rejects mismatched feature count).
- If caller passed a `window` that disagrees with `B.shape[1]`, emits `FutureWarning` and
  uses `B.shape[1]` (more permissive than main, which crashes).

This is a small API hardening adjacent to the even-window fix. The "use saved model's
geometry, not caller's hint" principle is universal in serialization-aware chemometrics
(PLS_Toolbox saved transfer models carry their own geometry; you don't pass a window arg
when applying a saved PDS model).

## 5. Distribution-model check (where is the bug user-reachable?)

PDS is reachable in dasp's Calibration Transfer GUI tab. Two GUI build paths:

| Path                                              | Validates window?                              |
|---------------------------------------------------|------------------------------------------------|
| `_build_transfer_model` (line 42334-42352)        | YES — messagebox "must be odd" before backend  |
| `_build_transfer_model_new` (line 44624-44627)    | NO — passes window straight to backend         |

The dominant user path (`_build_transfer_model`) is already protected. The
under-validated path (`_build_transfer_model_new`) is where the fix delivers value to
GUI users. Anyone hitting that path with an even window today gets a numpy traceback in
the console + crashed model build; post-fix they get a clear messagebox saying "must be
odd" with a Wang 1991 citation.

Also delivers value to programmatic callers (rare in the bundled-app distribution but
relevant for the export-script workflow) who would hit the matmul mismatch on
`apply_pds(X, B, window=W)` when `W ≠ B.shape[1]`.

PDS's actual usage is "occasional, advanced" rather than "every analysis." Bone FTIR
users do calibration transfer between labs/instruments, but it's not in every workflow.

**Severity ranking:** lower than T-05 (VIP, every PLS importance ranking), lower than
T-10 (PLS clamp, every LOO + spinbox-folds search). But it's a real bug fix with no
downside — backend-only, every GUI user benefits automatically when they hit either
path.

## 6. `apply_pds` signature change (worth flagging in the verdict)

The branch changes `apply_pds(X, B, window: int = 11)` → `apply_pds(X, B, window: int | None = None)`.

For default-arg callers (`apply_pds(X, B)`): no behavior change. Branch derives geometry
from B; main's default `window=11` would have crashed if B.shape[1] != 11.

For callers passing explicit `window` matching B's shape: no behavior change.

For callers passing explicit `window` that disagrees with B's shape: branch emits
`FutureWarning` and uses B's geometry. Main crashes with matmul mismatch.

This is a *behavior change* (more permissive than main), not just a guard. Per the plan
and Codex's review, this was scoped together with the even-window fix. The change
matches universal chemometrics convention ("trust the saved model's geometry, not the
caller's hint"). No downside; documented; FutureWarning gives a path for callers to
notice and update.

Worth noting in the verdict but not a blocker.

## 7. Empirical demonstration

**Even windows on main** (`window ∈ {2, 4, 6, 8}`):

All four sizes raise `ValueError: could not broadcast input array from shape (k+1,) into
shape (k,)`. Hard crash, deep numpy internals.

**Even windows on branch:**

All four sizes raise `ValueError: window must be odd (got K). This implementation uses
a centered, odd-width local window of size 2k+1 — see Wang, Veltkamp, & Kowalski (1991)
...`. Clear message + actionable + citation.

**Odd windows (legitimate path):**

window=5: main pre-error 0.0811 → post-error 0.0734. Branch: identical.
window=11 (GUI default): main pre 0.0811 → post 0.0646. Branch: identical.

**No behavior change for legitimate input.**

**Stale `apply_pds(X, B5, window=11)` with `B5.shape=(50, 5)`:**

Main: `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0`.

Branch: `FutureWarning: apply_pds: caller passed window=11, but B has width 5. Ignoring
caller arg; using B-derived width.` Then returns correctly transformed spectra using
window=5.

## 8. Test results

**Branch:** `pytest tests/test_pds_window_arithmetic.py -v`
```
10 passed in 0.07s
```

**Main:** Same tests against main's source via PYTHONPATH:
```
6 failed, 4 passed in ~3s
```

The 4 that pass on main are the odd-window/identity-case tests; the 6 that fail are the
even-window rejection tests, the apply_pds-stale-window test, the apply_pds-shape-validation
test, and the apply_pds-rejects-even-B test.

Discrimination is correct.

## 9. Internal consistency

Notable: dasp main is internally inconsistent on PDS window enforcement.

| Place                                           | Says window must be odd? |
|-------------------------------------------------|---------------------------|
| `estimate_pds` docstring                        | YES — "(odd integer)"     |
| GUI tooltip (line 1071)                         | YES — "Must be odd number"|
| GUI build path #1 (line 42334-42352)            | YES — messagebox enforces |
| GUI build path #2 (line 44624-44627)            | NO — silent passthrough   |
| Backend `estimate_pds` (line 170-222)           | NO — silent passthrough   |
| Backend `apply_pds` (line 225-258)              | NO — silent passthrough   |

The fix brings the two backend functions and GUI build path #2 into line with what dasp
already documents. Internal-consistency improvement, not a new constraint.

## 10. Verdict

**APPROVE for merge.**

Reasoning:

1. **Universal field consensus on odd-only.** Wang 1991, Bouveresse & Massart 1996,
   Feudale 2002, RNIR, specProc, PLS_Toolbox — every implementation surveyed uses odd
   windows by parameterization or enforcement. dasp main's "documented as odd, not
   enforced" stance has no precedent.
2. **Internal consistency improvement.** dasp's docstring + tooltip + GUI build path #1
   already say "must be odd." Only the backend lacked enforcement. Fix completes the
   contract.
3. **Real but smaller-class bug.** Main crashes loudly with cryptic numpy traceback; fix
   replaces with clear `ValueError` citing Wang 1991. Not a silent-wrong-output bug like
   T-05 — severity is lower.
4. **`apply_pds` hardening is field-aligned.** "Trust the saved model's geometry, not
   the caller's hint" is universal in serialization-aware chemometrics. The
   `FutureWarning` on disagreement is more permissive than main's crash.
5. **Distribution-model check passes.** Backend-only fix, every GUI user automatically
   benefits when they hit GUI build path #2 or programmatic callers. PDS is reachable in
   the bundled app via the Calibration Transfer tab.
6. **Tests discriminate the bug.** 6/10 fail on main, 10/10 pass on branch.
7. **Codex review's 5 suggestions all incorporated.** Verified by the agent's audit of
   the branch source against the Codex review file.

The branch is ready to merge. Real-world urgency is lower than T-05 / T-10 because:
- GUI path #1 already enforces odd
- Most users go through GUI path #1
- PDS itself is an advanced/occasional feature, not every-analysis

But there's no downside. Backend correctness improvement, no breaking changes for
legitimate input, no GUI changes needed, no opt-in.

## 11. Pre-merge checklist (same pattern as T-05/T-10)

- [ ] User reads + approves this validation note
- [ ] Rebase `fix/T07-pds-even-window` onto current main (resolves the docs-deletion noise
      from the pre-reframing branch base — same situation T-05/T-10 had)
- [ ] Re-run `pytest tests/test_pds_window_arithmetic.py -v` post-rebase (expect 10 passed)
- [ ] Run regression sweep on T-07-relevant tests: smoke imports + cv tests +
      search_comprehensive + bayesian + nsga2 + models + plsda + ga_pls + end_to_end +
      `test_calibration_transfer*.py` (this is the most relevant file for T-07)
- [ ] If all green, fast-forward merge into main
- [ ] Push
- [ ] Update `docs/bugfix_validation/README.md` status table to MERGED
- [ ] Confirm the "opencode draft (post-task snapshot)" commit at HEAD is acceptable to
      keep in the merged history, or interactively-rebase it down before merging
