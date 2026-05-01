# Bugfix Validation

Per-branch validation notes for the `fix/T*` bugfix branches, written under the chemometrics master rule:

> DASP implements chemometrics methods from the literature. The literature is the specification.
> Every methodology decision must be validated against (1) chemometrics literature and (2) the
> applied domain (bone FTIR / paleoanthropology / isotopes / diagenesis). If sklearn / generic ML /
> genomics ML conventions conflict with chemometrics, chemometrics wins unless the user explicitly
> says otherwise. Commercial software behavior (Unscrambler, OPUS, GRAMS, SIMCA, PLS_Toolbox) is a
> useful sanity check.

## Purpose

Before any `fix/T*` branch merges into main, a one-page validation note must be authored here that
answers:

1. **Canonical reference.** Which paper / textbook defines the correct behavior? Cite page or
   equation number.
2. **Side-by-side.** Current code vs. canonical formula or expected behavior.
3. **Commercial-software sanity check.** What does Unscrambler / SIMCA / OPUS / PLS_Toolbox /
   open-source equivalent (e.g. `pls` R package) do? Would those tools call the current code
   buggy?
4. **Test that fails on old code, passes on new code.** Concrete pytest invocation + expected
   output.
5. **Verdict.** Real bug per the literature, or sklearn-instinct false alarm?

## Status

| Branch                          | Validation note                          | Verdict        |
|---------------------------------|------------------------------------------|----------------|
| `fix/T05-vip-formula-fix`       | [T05_vip_formula.md](T05_vip_formula.md) ([investigation](T05_findings.md)) | MERGED 2026-04-30 — fast-forward into main at `2c068cd` after rebase + 290 tests pass |
| `fix/T07-pds-even-window`       | [T07_pds_even_window.md](T07_pds_even_window.md) ([investigation](T07_findings.md)) | MERGED 2026-04-30 — fast-forward into main at `1b91d93` after rebase + 327 tests pass |
| `fix/T10-pls-components-clamp`  | [T10_pls_components_clamp.md](T10_pls_components_clamp.md) | MERGED 2026-04-30 — fast-forward into main at `fbeb50c` after rebase + 290+ tests pass |
| `fix/T24-lins-ccc`              | [T24_lins_ccc.md](T24_lins_ccc.md) ([investigation](T24_findings.md)) | MERGED 2026-04-30 — fast-forward into main at `0087cad` after rebase + GUI plumbing + 342 tests pass |
| `fix/T26-snv-near-zero-std`     | [T26_snv_near_zero_std.md](T26_snv_near_zero_std.md) | DROP / WONT_FIX — current dasp behavior matches PLS_Toolbox default; bundled-app distribution makes a backend-only knob useless |

## Lessons from T-26

The T-26 validation went through three verdicts before settling: APPROVED → REJECT_AS_IS →
DROP. Each correction came from a question I should have asked myself first:

1. **APPROVED → REJECT_AS_IS.** Initial verdict appealed to "universal numerical-computing
   practice" (scipy/numpy/sklearn pattern of flooring near-zero divisors). Wrong frame —
   generic ML/scientific-computing instinct, not chemometrics. After the user pushed back,
   actual documentation lookup showed PLS_Toolbox / SIMCA use a continuous user-controlled
   `offset` parameter, not a hardcoded threshold. dasp would have been inventing its own
   pattern.

2. **REJECT_AS_IS → DROP.** Second verdict recommended rescoping to match PLS_Toolbox's
   `offset` parameter. The user pointed out that dasp ships as a bundled Inno Setup
   desktop app — there is no Python REPL, and "power users can set the parameter
   programmatically" describes nobody in the user base. Adding a backend-only parameter
   would deliver zero value; adding a GUI knob is 1–2 hours of plumbing for a corner
   case the user has never hit in thousands of analyses.

**Validation rules learned:**

1. **Verify leading-program behavior with actual documentation lookup before drafting the
   verdict.** Section 4 of the note template ("Commercial-software sanity check") must
   cite specific documentation pages, not generic claims like "universal numerical
   practice."
2. **Distribution model matters.** dasp is a bundled GUI app for non-technical users in
   bone FTIR / paleoanthropology / archaeology / isotope work. A "fix" that's only
   reachable from the Python API is not a fix — it's dead code. Validate that the proposed
   change actually delivers value to the GUI user base before recommending merge.
3. **Match-the-field cuts both ways.** The chemometrics master rule says don't invent
   patterns the field doesn't use. It also implies: if the field's *default* behavior is
   already what dasp does, the field-alignment gap may already be zero.
4. **A real finding can warrant zero action.** The original T-26 ticket described a real
   numerical behavior. The validation gate is allowed to conclude "yes this happens, no
   we shouldn't fix it" if leading programs accept the same behavior at their defaults
   and no real-world dataset has ever triggered it.

## Follow-up tickets

| Branch / Site                                   | Validation note                          | Status         |
|-------------------------------------------------|------------------------------------------|----------------|
| T-05a duplicate VIP formulas (templates + nsga2) | [T05a_vip_duplicates.md](T05a_vip_duplicates.md) | MERGED 2026-04-30 — applied same canonical Wold 2001 fix to both sites; 162 regression tests pass |
| T-04 one-class UVE prefilter | [T04_one_class_uve.md](T04_one_class_uve.md) ([investigation](T04_findings.md)) | MERGED 2026-04-30 — GUI grey-out matching iPLS pattern; 75 tests pass. T-04b/c deferred (broader y_oc audit + LOVE-style native one-class varsel) |
| T-21 SG wavelength uniformity guard | [T21_findings.md](T21_findings.md) | RESOLVED 2026-04-30 by hiding the "Convert to other unit" button (the only path that creates the non-uniform-grid bug surface). Radio-button relabel path was already safe. Function + widget preserved in code (un-comment one `.pack()` line) for a future resample-on-convert fix. ~20-60% derivative error magnitude verified empirically. |
| T-32 sample_weight length mismatch | (analyzed inline in this session) | DEFERRED to T-19 — current code path is unreachable; bug would fire only after T-19 makes class_weight + resampler combinations possible |
| T-06 SPA canonical Araújo 2001 enumeration | [T06_spa_canonical_seeds.md](T06_spa_canonical_seeds.md) ([investigation](T06_findings.md)) | MERGED 2026-04-30 — replaced non-functional `n_random_starts` knob with canonical Araújo 2001 deterministic enumeration over all J seeds. Cross-reviewed by Codex + Kimi K2.6. 5 new T-06 tests pass; full sweep 226+ tests green. T-06b parallelization follow-up MERGED same session via `joblib.Parallel(backend='threading')`, 7-8× speedup. |
| T-08 CARS tree-mode weight-update bias | [T08_cars_tree_no_action.md](T08_cars_tree_no_action.md) ([investigation](T08_findings.md)) | DROPPED 2026-04-30 — false alarm on framing's specific claims. Cited line numbers reference PLS-mode branch, not tree-mode. Empirical reproducer disproved all three bug claims (oscillation, bias, persistent tiny weights). CARS-Tree converges with std 0.0007-0.0038 over last 10 iterations and recovers 5/5 informative wavelengths in synthetic test. CARS-Tree confirmed dasp-invented for tree models that lack PLS coefficients. T-26 precedent applies. |
| T-15 LeaveOneGroupOut / GroupKFold | [T15_dropped.md](T15_dropped.md) ([investigation](T15_findings.md)) | DROPPED 2026-04-30 — user decision after gate investigation. Chemometrics canonical practice is external test sets (Westad & Marini 2015 / Workman 2018), NOT LOGO. Competitor parity mixed (only PLS_Toolbox exposes group-aware CV). User's data regime (5-100 N per site, 20× ratio) makes LOGO a footgun without T-16 uncertainty bands. User's own paper-archive notes document the failure mode (held-out severely-degraded sites lose extreme-degradation training examples). |
| T-11 Pause/resume + Optuna SQLite + disk logging | [T11_pause_resume_hardening.md](T11_pause_resume_hardening.md) ([investigation](T11_findings.md)) ([full review trail](../reviews/deepseek_v4pro_24h_review_2026-04-30.md)) | **MERGED 2026-05-01 via PR #6** at `50057af` (rebase merge, linear history). 7-pass review: Codex initial (4 HIGH + 3 MEDIUM all fixed pre-PR) → Kimi K2.6 initial (2 MAJOR + 4 MINOR all fixed pre-PR) → DeepSeek V4 Pro pass-1 24h (2 HIGH + 3 MEDIUM + 9 LOW/INFO closed) → DeepSeek pass-2 recheck (2 NEW HIGH closed) → 5 specialist agents in parallel (13 findings: 12 closed + 1 deferred to T-34) → Codex meta-review (1 NEW critical NEW BUG #1 closed; 1 deferred NEW BUG #2 → T-34) → DeepSeek pass-3 high-effort (READY_TO_MERGE, 0 blockers, 1 cosmetic accepted). Closes T-12. New tickets filed for deferrals: T-34 / T-35 / T-36 / T-37. 47 T-11 tests passing. |

## Codex review archive

`codex_reviews/` holds the original per-ticket Codex reviews captured during implementation. These
were written before the chemometrics master rule was established, so their framing may be biased
toward sklearn-style pipeline purity. Use them as one input among several — not as authoritative
sign-off.

## Validation note template

```markdown
# T-XX validation: <ticket title>

**Branch:** `fix/T-XX-...`
**Status:** [DRAFT | APPROVED | REJECTED]

## 1. Canonical reference

<paper citation, page/equation>

## 2. Current behavior vs. canonical

<side-by-side code excerpt + formula>

## 3. Commercial software sanity check

<what does Unscrambler / SIMCA / PLS_Toolbox do?>

## 4. Test

```bash
pytest <path> -v
```

Expected: failing on `main`, passing on `fix/T-XX-...`.

## 5. Verdict

[REAL BUG per the literature | FALSE ALARM | INCONCLUSIVE]

Reasoning: ...
```
