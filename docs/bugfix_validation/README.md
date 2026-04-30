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

| Branch                          | Validation note          | Verdict          |
|---------------------------------|--------------------------|------------------|
| `fix/T05-vip-formula-fix`       | _pending_                | _pending_        |
| `fix/T07-pds-even-window`       | _pending_                | _pending_        |
| `fix/T10-pls-components-clamp`  | _pending_                | _pending_        |
| `fix/T24-lins-ccc`              | _pending_                | _pending_        |
| `fix/T26-snv-near-zero-std`     | _pending_                | _pending_        |

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
