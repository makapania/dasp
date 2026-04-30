# T-04 validation: one-class UVE prefilter using outlier-contaminated labels

**Status:** APPROVED + IMPLEMENTED 2026-04-30 (Option A — disable UVE family in one-class mode)
**Author:** Opus 4.7, 2026-04-30
**Verdict:** REAL methodological issue with user-visible impact. Fix: GUI-level disable for
one-class mode, matching the existing iPLS-family pattern.

Investigation findings: [T04_findings.md](T04_findings.md).

---

## TL;DR

UVE in one-class mode runs `get_uve_threshold(X, y_oc, ...)` where `y_oc` is the binary
+1/−1 inlier/outlier label vector. UVE was designed by Centner et al. 1996 for quantitative
regression — applied to a class indicator it returns wavelengths that **discriminate**
outliers from inliers, not wavelengths that define the **inlier class structure** (which is
what one-class modeling needs).

Pomerantsev, Kucheryavskiy & Rodionova (2025), _Variable selection for one class
classifiers. Introduction of LOVE_, Anal. Chim. Acta 1368:344302, opens its abstract with
exactly the T-04 framing:

> "Variable selection methods for regression and discrimination are well developed. However,
> the known methods do not adequately address the problem in the case of one class
> classifiers."

This is the chemometrics field's leading authority on DD-SIMCA / one-class confirming
that y-driven methods like UVE don't fit one-class modeling.

The empirical demonstration on a bone-FTIR-realistic synthetic dataset (60 inliers + 20
consolidant-contaminated outliers) shows main's UVE prefilter selecting **5/5 top
wavelengths at the consolidant peak** and **0/20 hits at phosphate or carbonate** (the
actual bone chemistry). The trained one-class model is "a consolidant detector dressed as
a clean-bone class model" — and would silently fail on samples contaminated with a
different preservative whose absorption is at a wavelength the prefilter dropped.

## What changed

The fix matches the existing iPLS-family GUI grey-out pattern dasp already uses for
methods that don't fit a particular task type. When `task_type == 'one_class'`:

1. The UVE checkbox + four UVE-hybrid checkboxes (`UVE-SPA`, `UVE-CARS`, `UVE-CARS-Tree`,
   `UVE-CARS-SPA`) get `state(['disabled'])` and their underlying `varsel_*` BooleanVars
   are forced to False.
2. The `Apply UVE Pre-filter` checkbox gets the same treatment (so the hard-mask prefilter
   path is also unreachable from the GUI).
3. The Bayesian-mode `UVE Variable Selection` checkbox gets disabled and its
   `bayes_enable_uve` BooleanVar is forced to False (so the default Bayesian one-class run
   no longer hits UVE).

All checkboxes get re-enabled when the user switches back to regression or classification.

## Files touched

| File                                          | Change                                            |
|-----------------------------------------------|---------------------------------------------------|
| `spectral_predict_gui_optimized.py`           | 6 checkbox creations now assign to `self._cb_*` attributes (matches iPLS pattern). The `_update_one_class_controls_visibility()` disable-list grew from 7 → 14 items. |

No backend code changed. The user can no longer reach UVE-on-`y_oc` from the GUI; if anyone
calls `run_one_class_search` or `run_unified_bayesian` programmatically with
`varsel_method='uve'` (or with `apply_uve_prefilter=True`), the buggy code path is still
present but no longer reachable from the bundled-app distribution. This is the same
defense level dasp uses for iPLS in one-class.

## Why GUI-only and not backend block too

Per the user's explicit instruction ("we could just grey it out when in one-class mode.
this is exactly what happens with regression and classification"). The iPLS family is
also handled via GUI-only grey-out, not a backend block — keeping the same pattern
preserves codebase consistency.

If a future ticket wants to add backend defense-in-depth (raise on
`run_one_class_search(varsel_method='uve')` programmatic call), it can be filed as T-04c.
Out of scope for this ticket.

## Field-alignment check

The agent's investigation could not locate any commercial chemometrics package (PLS_Toolbox,
SIMCA, Unscrambler, mdatools) that implements UVE-on-class-indicator as a one-class
variable-selection method. The 2024–2025 literature (LOVE, MPS-SIMCA, OGA) is **actively
developing** dedicated one-class variable-selection methods *because* the existing y-driven
methods don't fit. Disabling UVE for one-class brings dasp in line with the field's
direction. See [T04_findings.md](T04_findings.md) §2 + §6 for the full literature survey.

## Out-of-scope items (filed for follow-up)

The investigation flagged three related issues that are **not** addressed by this fix:

1. **The other y_oc-using one-class varsel methods** (`importance`, `spa`, `cars`,
   `cars-tree`, `ga`, `vcpa-iriv`) have varying degrees of the same fundamental problem.
   `compute_one_class_importances` is a LightGBM binary classifier on `y_oc` — also a
   discriminator wearing a one-class label. Disposition: **T-04b** (broader scope, separate
   ticket). Per the agent: "The chemometrics master rule arguably implies the broader
   scope: if the critique is 'y_oc is the wrong target for one-class,' the fix shouldn't
   be 'UVE only.'"

2. **A proper one-class-native variable-selection method.** The chemometrics field has
   converged on options: modeling power on inlier-only PCA (Forina), LOVE (Pomerantsev
   2025), OGA (Anal. Chem. Acta 2025). Disposition: **T-04c** (multi-week scope, separate
   ticket).

3. **The `preprocessing_discovery.py:_quick_evaluate` one-class path** also uses
   LGBMClassifier with class_weight='balanced' on `y_oc`. Same fundamental issue; out of
   T-04 scope. Disposition: re-evaluate alongside T-04b.

The narrow T-04 fix delivered today — GUI grey-out — is the immediate user-protection
move. The deeper architectural question of "what is the right one-class variable-selection
in dasp?" is correctly deferred to T-04b/c when the user wants to invest in that direction.

## Reachability summary (the recurring trap, checked explicitly)

This is NOT an overzealous flag like T-26 / T-32 / the two re-evaluation flags:

| Check | Result |
|-------|--------|
| Is the buggy code path reachable in the GUI? | YES — 5 configurations: prefilter checkbox + 4 UVE varsel methods + Bayesian one-class default (`bayes_enable_uve=True`) |
| Does the bug have user-visible impact (not just display)? | YES — selects different wavelengths than chemometrics-correct alternative; on bone-FTIR demo, picked 0/20 phosphate/carbonate vs 5/5 consolidant peak |
| Does dasp match leading-program behavior? | NO — no commercial package implements UVE-on-class-indicator for one-class |
| Is this a sklearn-instinct false alarm? | NO — Pomerantsev 2025 LOVE paper explicitly says regression/discrimination methods don't fit one-class |
| Does fixing it break working code? | NO — GUI-only change, backend code untouched, iPLS pattern reused exactly |

## Tests

- `tests/smoke/test_imports.py`: 6 passed (verifies the GUI module imports after the
  checkbox-attribute additions and disable-list expansion)
- `tests/test_contamination_detection.py`: 69 passed (the one-class detection test suite,
  unaffected by GUI-level changes since it tests the backend directly)

Total: 75 tests pass post-fix.

## MEMORY.md update needed

The agent flagged that `MEMORY.md` says "Variable Selection: UVE, SPA, iPLS, CARS, GA-PLS
(note: only 'importance' works for one-class)." This was wrong on main pre-T-04 (current
code reached UVE + 4 UVE hybrids in one-class). After T-04, the note becomes _correct_:
only `importance` and the non-UVE methods (`spa`, `cars`, `ga`, `vcpa-iriv`) are reachable
in one-class mode. The MEMORY.md note will be updated as part of the session checkpoint.

## Verdict

**APPROVED + IMPLEMENTED.** GUI grey-out matches dasp's existing iPLS-family pattern,
delivers immediate protection to bundled-app users, doesn't touch backend code, and
preserves the code path for programmatic callers if they want defense-in-depth later.
Three follow-up tickets (T-04b broader y_oc-as-target audit, T-04c LOVE/modeling-power
implementation, preprocessing_discovery one-class y_oc reuse) are filed as deferred
ahead-of-the-immediate-fix.
