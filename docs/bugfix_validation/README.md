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
| `fix/T05-vip-formula-fix`       | _pending_                                | _pending_      |
| `fix/T07-pds-even-window`       | _pending_                                | _pending_      |
| `fix/T10-pls-components-clamp`  | [T10_pls_components_clamp.md](T10_pls_components_clamp.md) | APPROVED — real bug (LOO over-clamp), small impact, clean fix, complete coverage |
| `fix/T24-lins-ccc`              | _pending_                                | _pending_      |
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
