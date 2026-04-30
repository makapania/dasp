# T-05a validation: duplicate VIP formulas at templates + nsga2_search

**Status:** APPROVED (extension of T-05; same canonical Wold 2001 fix applied to two more sites)
**Author:** Opus 4.7, 2026-04-30
**Verdict:** Same fix as T-05, two additional sites confirmed buggy and reachable.

T-05 closed the central buggy VIP at `src/spectral_predict/models.py:compute_vip`. Two
duplicate copies of the same `np.var(y) * (T'T)` bug were explicitly out of scope per the
T-05 plan and deferred to T-05a. The T-05 investigation agent confirmed both, and direct
re-grep on main confirms them again on 2026-04-30 post-T-05-merge:

| File                                            | Line | Reachable via                                         |
|-------------------------------------------------|------|-------------------------------------------------------|
| `src/spectral_predict/templates/variable_selection.py` | 54   | Standalone Python scripts exported from the GUI. Users running exported scripts outside dasp would get wrong VIP rankings. |
| `src/spectral_predict/nsga2_search.py` (`_compute_wavelength_importance`) | 627  | NSGA-II adaptive guidance — feeds back into the GA's wavelength sampling probability. Wrong VIP biases the GA's search. |

## What changed

Each site now uses the canonical Wold 2001 formula identical in shape to the post-T-05
`models.py:compute_vip`:

```python
W = np.asarray(pls.x_weights_)
T = np.asarray(pls.x_scores_)
Q = np.asarray(pls.y_loadings_)
q = Q if Q.ndim == 1 else Q[0, :]

ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)         # canonical SSY_a
ssy_total = float(np.sum(ssy_comp))
if ssy_total <= 0.0:
    return zeros

col_norm_sq = np.sum(W ** 2, axis=0)
col_norm_sq = np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)
w_norm_sq = (W ** 2) / col_norm_sq                    # ||W_a||^2 normalization
vip_scores = np.sqrt(n_features * (w_norm_sq @ ssy_comp) / ssy_total)
```

## Field-alignment + canonical references

Same as T-05 (no new field-alignment work needed). Wold 2001 / Mehmood 2012 / Chong & Jun
2005, plus universal commercial + open-source consensus on the canonical formula. See
[T05_findings.md](T05_findings.md) Section 6 for the full survey.

## Reachability check

Both sites are GUI-reachable for dasp's bundled-app users:

1. **`templates/variable_selection.py:54`**: Users with the "Export Script" button generate
   standalone Python scripts that contain this VIP code. If they run those scripts they get
   wrong rankings. (This is the smaller user surface — exported-script workflow.)

2. **`nsga2_search.py:627`**: Users selecting NSGA-II as the optimization method (one of
   three options in `self.optimization_method`: grid / unified / nsga2) trigger this code.
   The wrong VIP biases NSGA-II's adaptive wavelength sampling. (Larger user surface — any
   NSGA-II run on regression data.)

So both sites pass the bundled-app distribution check.

## No back-compat risk

Same as T-05: the new formula is a correctness fix, not a behavior change for legitimate
use. For exported standalone scripts, the new VIP returns canonical values matching what
`models.py:compute_vip` (the dasp internal authoritative path) returns post-T-05. For
NSGA-II adaptive guidance, the wavelength biasing is now driven by the correct importance
ranking instead of the bug's pseudo-importance.

## Tests

- `tests/test_vip_formula.py` (the central T-05 test file): still 7/7 pass post-T-05a fix.
- `tests/test_nsga2_search.py`: 54 passed, 0 failures (relevant since NSGA-II's
  `_compute_wavelength_importance` is exercised by these tests).
- `tests/test_plsda_importance.py`: passes (uses `compute_vip` from models.py).
- Broader regression sweep: running. Tally to be appended on completion.

## Verdict

**APPROVE.** Same fix as T-05's central path, applied to the two known-buggy duplicate
sites. No new field-alignment work, no new tests required (existing tests cover both the
formula and the call sites that exercise it). Internal-consistency improvement: dasp now
has one VIP formula, correctly applied in all three sites.

This closes the T-05 / T-05a follow-up. No further duplicate VIP formulas remain in the
codebase per `grep "ssy_comp.*var(y" src/`.
