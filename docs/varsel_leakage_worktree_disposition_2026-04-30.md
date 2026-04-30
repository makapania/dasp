# Varsel-Leakage Worktree Disposition — 2026-04-30

> **Worktree:** `.worktrees/varsel-leakage-fix` on branch `fix/varsel-leakage`, HEAD `ac4aae1`
>
> **Re-evaluator:** opencode (mimo-v2.5-pro), 2026-04-30
>
> **Context:** The worktree was created under the "per-fold varsel is the correct fix" framing. That framing has been reconsidered: dasp's varsel-on-full-calibration is the standard chemometrics workflow (Li 2009, Centner 1996, Araujo 2001, Nørgaard 2000, Wold 2001). Per-fold varsel is NOT the correct default. However, some artifacts have value under the new framing.

---

## Artifact-by-artifact disposition

### 1. `src/spectral_predict/varsel_transformer.py` (~500 LOC, 4 classes)

**Verdict:** MERGE_AS_OPTIONAL_TOOL

**What it is:** sklearn-compatible transformers wrapping dasp's varsel methods:
- `VarselTransformer` — generic wrapper for any varsel method (SPA, UVE, iPLS, CARS, etc.)
- `ModelImportanceVarselTransformer` — wraps model-native feature importance (VIP, LightGBM, etc.)
- `SubsetVarselTransformer` — wraps subset-returning methods (iPLS forward/backward)
- `OneClassVarselTransformer` — wraps one-class importance (LightGBM surrogate)

**Why keep as optional tool:** Under the master rule, per-fold varsel is NOT the correct default workflow. But the code has three legitimate uses:

1. **T-22 stability diagnostic (bootstrap varsel frequency):** Run `VarselTransformer` inside a bootstrap loop N times, report which wavelengths are selected how often. This is the *right diagnostic* for "is this wavelength real chemistry?" — stable selection across resamples = evidence of real absorption bands. The transformer's `fit/transform` API makes it trivially insertable into a bootstrap loop.

2. **Expert mode for users with external test sets:** A user who has a proper external test set (the canonical chemometrics workflow) might want to run per-fold varsel as a *diagnostic comparison* — "does my R²cv change if I do per-fold varsel?" This is a legitimate question, not the default.

3. **Reproducing specific papers that use per-fold varsel:** Filzmoser 2009 (rdCV), Shi 2019, and Király & Tóth 2025 all describe per-fold varsel workflows. A researcher trying to reproduce one of these papers needs the capability.

**Recommended treatment:** Add the module to `src/spectral_predict/` but do NOT wire it into the default search paths. Document it as a "diagnostic / expert-mode tool" in the user guide. Wire it into a future "Bootstrap Stability" GUI card (T-22 reframe).

### 2. `tests/test_varsel_transformer.py` (~243 LOC, 32 tests)

**Verdict:** MERGE (with the module)

**Why:** 32 passing tests covering the sklearn contract (fit/transform/clone/get_support), planted-signal correctness, Pipeline integration, and cross-validation behavior. These tests validate the transformer infrastructure and should ship with it.

### 3. `docs/T01_VARSEL_LEAKAGE_AUDIT.md` (330 lines)

**Verdict:** CHERRY_PICK — retain as a renamed reference doc

**What's valuable:** The method × path matrix (lines 88-265) is a **technically correct description of where varsel runs** in every search path. The file:line references are accurate and useful for anyone tracing the code. The architectural note (lines 312-330) correctly describes the two-phase varsel-then-CV structure.

**What's wrong:** The "LEAKY" labels and the framing that per-fold varsel is the fix. The banner at lines 1-27 already corrects this.

**Recommended treatment:** Rename to `docs/VARSEL_ARCHITECTURE_AUDIT.md`. Keep the banner (updated to remove "LEAKY" from the body). Replace all "LEAKY" labels with "varsel-on-full" throughout the body. Remove the "Recommended fix" section (lines 61-71). Add a new section: "Implications for performance reporting" explaining that R²cv is biased upward and the canonical solution is external test sets (RMSEP), not per-fold varsel.

### 4. `docs/plans/2026-04-29-varsel-leakage-fix.md` (~2900 lines)

**Verdict:** DROP

**Why:** The entire plan is predicated on the wrong framing (per-fold varsel as the fix). The 7-phase plan, the call-site refactors, the VarselTransformer wiring into grid/Bayesian/NSGA-II paths — all represent a worldview that conflicts with chemometrics convention. The plan's architecture decisions (pipeline approach, manual per-fold loop, hybrid for Bayesian) are technically sound engineering but serve the wrong goal.

**What's worth extracting:** The computational cost analysis (per-fold CARS is ~5× more expensive) is useful context for any future per-fold-varsel-as-diagnostic documentation. But this is a paragraph, not a 2900-line plan.

### 5. Codex review files (`codex_review_*.txt`)

**Verdict:** DROP

**Why:** These reviews were conducted under the same wrong framing. The code-level observations are correct but the strategic conclusions are wrong. Not worth preserving.

### 6. Literature validation files (already copied to main repo)

**Verdict:** MERGE (already done)

**Why:** `docs/analysis_vs_chemometrics_lit/leakage_validation_GLM_2026-04-30.md` and `leakage_validation_codex_2026-04-30.txt` are the literature-validation passes that correctly identified the framing error. They are reference documents that justify the re-evaluation. Already copied to main repo.

### 7. The 4 plan-revision commits + 5 Phase-1 implementation commits

**Verdict:** CHERRY_PICK — only the `varsel_transformer.py` + test file are worth keeping

**Commits `70738c9` → `ac4aae1` contain:**
- Plan revisions (DROP — see #4 above)
- `varsel_transformer.py` implementation (MERGE_AS_OPTIONAL_TOOL — see #1)
- `test_varsel_transformer.py` (MERGE — see #2)
- Audit doc updates (CHERRY_PICK — see #3)
- Codex review files (DROP — see #5)
- Literature validation files (already merged — see #6)

**Recommended treatment:** Do NOT merge the branch as-is. Instead, cherry-pick `varsel_transformer.py` and `test_varsel_transformer.py` into a new branch (e.g., `feature/varsel-stability-diagnostic`), add documentation framing them as a diagnostic tool, and merge that.

---

## Summary

| Artifact | Verdict | Action |
|----------|---------|--------|
| `varsel_transformer.py` | MERGE_AS_OPTIONAL_TOOL | Cherry-pick to new branch, document as diagnostic |
| `test_varsel_transformer.py` | MERGE | Ship with the module |
| `T01_VARSEL_LEAKAGE_AUDIT.md` | CHERRY_PICK | Rename to `VARSEL_ARCHITECTURE_AUDIT.md`, fix labels |
| `plans/2026-04-29-varsel-leakage-fix.md` | DROP | Delete |
| Codex review files | DROP | Delete |
| Literature validation files | MERGE | Already in main repo |
| Plan-revision commits | CHERRY_PICK | Only varsel_transformer + tests |
| Phase-1 implementation commits | CHERRY_PICK | Only varsel_transformer + tests |

---

## Key question: Is the `VarselTransformer` family dead code under the master rule?

**No.** It is dead code *as a default pipeline component* but has three legitimate uses as an optional diagnostic tool:

1. **Bootstrap stability diagnostic (T-22 reframe):** The primary use. Run varsel N times, report wavelength frequency. This is the correct answer to "is this wavelength real chemistry?"

2. **Expert-mode per-fold varsel for diagnostic comparison:** A user with an external test set can compare R²cv from varsel-on-full vs per-fold-varsel as a sensitivity analysis.

3. **Paper reproduction:** Researchers reproducing rdCV (Filzmoser 2009) or similar workflows need per-fold varsel capability.

The code is well-written (500 LOC, 4 classes, 32 passing tests, clean sklearn contract). Dropping it entirely would waste good engineering. Keeping it as a non-default diagnostic preserves the investment without conflicting with chemometrics convention.
