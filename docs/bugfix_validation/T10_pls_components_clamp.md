# T-10 validation: CV-aware PLS components clamp

**Branch:** `fix/T10-pls-components-clamp` (HEAD `bc348d2`)
**Status:** APPROVED for merge (after rebase onto current main)
**Author:** Opus 4.7, 2026-04-30
**Verdict:** REAL BUG, small real-world impact, clean fix, complete coverage by design.

---

## TL;DR

The bug is a real but minor over-clamping issue in dasp's automated PLS-component grid
construction when `cv_strategy='loo'` is used. Main's formula
`n_samples * (folds - 1) // folds` is correct for k-fold and repeated_kfold but produces
the wrong value for LOO when the GUI passes the spinbox `folds` value (typically 5)
instead of `n_samples`. T-10 extracts the formula into a `cv_strategy`-dispatched helper
that returns `n - 1` for LOO regardless of the `folds` argument.

Real-world impact: minimal. The bug never causes crashes or wrong predictions — it just
restricts the PLS-component grid to `(n × 4 // 5)` candidates instead of `(n − 1)` for
LOO users, losing roughly 19 candidate component values for `n=100`. Optimal PLS components
in chemometrics are typically 5–15, well below either clamp, so this rarely changes
results in the user's domain (bone FTIR / NIR / collagen).

But the fix is cheap, well-tested (21 tests pass), and the bug is genuinely incorrect, so
APPROVE.

## 1. What is the actual bug?

**dasp main, `search.py:1109` (and similarly `:3312` for Bayesian search):**

```python
min_train_samples = n_samples * (folds - 1) // folds
```

This formula is correct for `cv_strategy='kfold'` (and `'repeated_kfold'`, which has
identical fold geometry). It is wrong for `cv_strategy='loo'` when `folds ≠ n_samples`.

**The GUI inconsistency that triggers the bug** (`spectral_predict_gui_optimized.py:27367`):

```python
results_df, label_encoder = run_search(
    X_filtered,
    y_filtered,
    task_type=task_type,
    folds=self.folds.get(),               # spinbox value, typically 5
    cv_strategy=self.cv_strategy.get(),   # may be 'loo'
    ...
)
```

The GUI passes `folds=5` even when `cv_strategy='loo'` (the spinbox isn't auto-updated
when LOO is selected). At line `26801` the GUI knows the right value (`_eff_folds = len(X_filtered) if cv_strategy == 'loo' else self.folds.get()`) — it just doesn't use that
value at the `run_search` call site.

**Result:** for an LOO user with `n=100, folds=5` (default spinbox):

| value                   | computed     | actual LOO max  |
|-------------------------|--------------|-----------------|
| `min_train_samples`     | 100×4//5 = 80 | 99             |
| `safe_max_components`   | 80           | 99              |

The PLS-components grid is built from `1..safe_max_components`. Main searches
`1..80`; T-10 searches `1..99`. Neither produces wrong predictions — main just
tests a smaller grid.

## 2. Mapping main's clamp behavior across all scenarios

```
scenario                                                  main_clamp   actual_max verdict
---------------------------------------------------------------------------------------------
kfold n=20 folds=5                                                16           16 OK (same)
kfold n=100 folds=5                                               80           80 OK (same)
kfold n=50 folds=10                                               45           45 OK (same)
LOO n=20 folds=5 (BUGGY: spinbox not updated)                     16           19 OVER-CLAMP by 3
LOO n=100 folds=5 (BUGGY: spinbox not updated)                    80           99 OVER-CLAMP by 19
LOO n=20 folds=20 (correct: GUI substituted)                      19           19 OK (same)
```

Critically: **main never over-shoots.** It only over-clamps. So no crashes, no `ValueError`
from sklearn, no wrong outputs — just a smaller-than-physically-possible PLS grid in the
LOO+default-spinbox case.

This is verified empirically:

```
sklearn 1.8.0 PLSRegression behavior:
- n_components ≤ min(features, train_samples): fit OK
- n_components > train_samples: ValueError "n_components upper bound is N. Got M."
- silent truncation: no, it raises.
```

Since main's clamp is conservative (always ≤ actual), it never produces an `n_components`
value that violates sklearn's check. The only effect is "tests fewer values than
physically possible."

## 3. What T-10 changes

**New helper** in `cv_utils.py`:

```python
def compute_min_train_fold_size(cv_strategy, n_samples, n_folds) -> int:
    if cv_strategy == 'loo':
        return n_samples - 1
    if cv_strategy in ('kfold', 'repeated_kfold'):
        ...
        return max(1, n_samples * (n_folds - 1) // n_folds)
    if cv_strategy in ('group_kfold', 'leave_one_group_out'):
        raise NotImplementedError(...)  # T-15 territory
    raise ValueError(f"Unknown cv_strategy: {cv_strategy!r}")
```

Key properties:

- LOO returns `n - 1` regardless of the (now-meaningless) `n_folds` argument
- k-fold / repeated_kfold preserves the existing formula (no behavior change)
- group splitters explicitly raise `NotImplementedError` — preserves the option for T-15
  to add group-aware sizing without back-compat concerns
- Validation: rejects `n_samples < 2`, `n_folds < 2`, `n_folds > n_samples`

**Call sites updated** in `search.py:1109` (run_search) and `:3312` (run_bayesian_search).
Plus `models.py:580` docstring clarifies that callers must pre-clamp.

## 4. Coverage check — what about NSGA-II?

`grep` finds five additional occurrences of the buggy formula in
`src/spectral_predict/nsga2_search.py` (lines 1366, 1410, 2564, 2711, 2855). T-10 does
**not** modify these, which initially looked like an incomplete fix.

**On inspection these are not bugs** — `nsga2_search.py:1784-1790` explicitly forces
`cv_strategy='kfold'`:

```python
if cv_strategy not in ('kfold', None):
    print(f"Warning: NSGA-II search does not support {cv_strategy} CV; falling back to K-fold.")
    cv_strategy = 'kfold'
```

So those five occurrences only ever run in k-fold context, where the formula is correct.
T-10's coverage is complete by design.

When T-15 (LeaveOneGroupOut / GroupKFold) lands, NSGA-II will need to be revisited
anyway — at that point the helper can be extended to support group splitters, and those
five sites can switch to the helper. That is T-15's scope, not T-10's.

## 5. Field-alignment check (chemometrics master rule)

This bug is internal to dasp's automated PLS-component grid construction. It is not a
methodology question — it is dasp computing the rank constraint slightly wrong in one
sub-case of one strategy.

**Do leading commercial programs do automated PLS-component selection like dasp?**

Largely no, in the form dasp does it. PLS_Toolbox / SIMCA / The Unscrambler typically
present the user an RMSECV (or RMSEP) curve and let them pick `n_components` manually,
following the rule of thumb "pick the elbow / first local minimum / smallest model
within 1 SE of the minimum." Automated grid search over PLS components is more of a
sklearn-pattern feature dasp adopted as an automation layer.

So for this fix, "what does PLS_Toolbox do for the n_components rank-clamp during
automated search?" doesn't really apply — they don't auto-search. Within dasp's
auto-search, the rank constraint must be computed correctly, and T-10 fixes a real
sub-case of that computation.

The mathematical bound `n_components ≤ min(rank(X), n_samples_train)` is universal and
agreed across the literature (Wold 2001, Centner & Massart 1998). sklearn enforces
strictly. Both before and after T-10, dasp respects the bound — T-10 just uses the
correct value of `n_samples_train` for LOO instead of the wrong-but-always-conservative
fallback.

## 6. Distribution-model check (the lesson from T-26)

T-10 is purely backend. No GUI changes needed. Users automatically get the fix when they
update dasp. There is no "power users can opt into this programmatically" caveat — every
user who selects LOO from the GUI dropdown gets the corrected clamp.

The bundled-app distribution model is satisfied: this fix delivers value to the actual
GUI user base (LOO users) without requiring any user action or configuration knob.

## 7. Real-world impact

For the user's domain (bone FTIR, paleoanthropology, isotopes):

- Typical sample counts: 20–200
- Typical optimal PLS components: 2–8 (occasionally up to 15 for complex matrices)
- LOO usage: present but uncommon
- Bug fires for: LOO + spinbox-default-folds (5)
- Over-clamp difference for `n=100`: 80 vs 99 — irrelevant if optimal is 2–8

So the bug almost certainly never affected published results in the user's domain. But
the fix is correct, the tests pass, and leaving the bug in place would surface as a
correctness gap to anyone running automated tests against the codebase.

A user who runs LOO + saves a model from before this fix vs. after this fix will get
identical or near-identical results for typical n_components, but might see a slightly
different selection if their data happens to have an optimum in the (n×4//5, n−1) range.
That's an improvement, not a regression — the new search space includes everything the
old one did, plus more valid candidates.

## 8. Test results

Branch test suite `tests/test_cv_pls_clamp.py`:

```
21 passed, 196 warnings in 2.51s
```

The 196 warnings are all from sklearn (`y residual is constant` and
`UndefinedMetricWarning` for tiny synthetic datasets the tests use to exercise edge
cases). They are inherent to the small test fixtures, not introduced by T-10.

Test coverage:

- Helper unit tests for kfold (5 cases), repeated_kfold (1), LOO (3), edge cases (4),
  group splitters (1), unknown strategy (1) — 16 tests
- Integration tests in `run_search` (4 cases) and `run_bayesian_search` (1 case) — 5 tests

## 9. Verdict

**APPROVE for merge.**

Reasoning:

1. **Real bug.** Main over-clamps PLS components for LOO + default-folds-spinbox.
2. **Small but real impact.** Users searching on LOO get fewer candidates than
   physically possible. Doesn't affect typical chemometrics workloads (optimal
   n_components is well below either clamp).
3. **Clean fix.** Extract formula into helper, dispatch by cv_strategy, raise on
   group splitters (forward-compat with T-15).
4. **Complete coverage.** NSGA-II's apparently-buggy formula occurrences are
   unreachable for LOO because NSGA-II force-falls-back to k-fold.
5. **No field-alignment issue.** This is dasp-specific automation correctness, not a
   methodology choice. Mathematical rank constraint is universally agreed.
6. **No distribution-model issue.** Backend-only fix; users automatically benefit.

## 10. Pre-merge checklist

- [ ] User reads + approves this validation note
- [ ] Rebase `fix/T10-pls-components-clamp` onto current main (resolves the docs-deletion
      noise from the pre-reframing branch base — same situation as T-26 had)
- [ ] Re-run `pytest tests/test_cv_pls_clamp.py -v` post-rebase
- [ ] Merge into main (fast-forward if rebase was clean; otherwise merge commit)
- [ ] Push
- [ ] Update `docs/bugfix_validation/README.md` status table to APPROVED
- [ ] Optional: keep `fix/T10-pls-components-clamp` branch ref for revert safety, or
      delete after a successful run cycle
