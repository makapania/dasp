## T-06 SPA `n_random_starts` non-functionality — canonical Araújo 2001 enumeration

**Branch:** `fix/T06-spa-canonical-seeds`
**Status:** APPROVED (pending merge)
**Date:** 2026-04-30

## 1. Canonical reference

Araújo, M. C. U., Saldanha, T. C. B., Galvão, R. K. H., Yoneyama, T., Chame,
H. C., & Visani, V. (2001). The successive projections algorithm for variable
selection in spectroscopic multicomponent analysis. *Chemometrics and
Intelligent Laboratory Systems*, 57(2), 65–73.

Canonical SPA is **deterministic enumeration over every variable as candidate
first variable.** The algorithm is:

```
For initials k(0) = 1 to J (every variable in turn):
  For N = N_min to N_max:
    Run forward-projection chain from k(0), selecting N variables
    Score the chain via CV
Return the (k*(0), N*) with the best CV criterion.
```

There is no random initialization in canonical SPA. The "starts" are an
enumeration, not a sample.

Modern open-source reference implementations confirm this:
- `auswahl` (Python) — explicit "every variable as candidate seed", no
  `n_random_starts`, no `random_state`.
- Galvão 2012 SPA-GUI (Araújo's group) — same deterministic enumeration.
- PMC 5573288 review of SPA — quotes the canonical algorithm verbatim.

No verified chemometrics implementation exposes a "random restart count"
parameter for SPA.

## 2. Current behavior vs. canonical

**Pre-T-06 (`main`):**

```python
def spa_selection(X, y, n_features, n_random_starts=10, cv_folds=5, random_state=42):
    ...
    for start_idx in range(n_random_starts):    # ← 10 iterations of identical work
        ...
        first_var = np.argmax(initial_corrs)    # ← always the same variable
        ...
```

The function's own docstring conceded the bug:
> "currently SPA is deterministic, but this parameter is included for
>  API consistency and future enhancements"

But the GUI exposed `SPA Random Starts: 10` (Spinbox 1–50) as a user-facing
control, and the value plumbed through to a function that ignored it.
Empirical run on `np.random.default_rng(0)` 30×15 noise: with
`n_random_starts ∈ {1, 5, 10}` and `random_state ∈ {42, 123}` the function
returned byte-identical importances. Wall-clock cost scaled linearly with the
knob value for zero benefit.

**Post-T-06:**

```python
def spa_selection(X, y, n_features, cv_folds=5):
    ...
    for first_var in range(n_vars):    # ← every variable enumerated as seed
        selected_indices = [first_var]
        ...
```

The signature drops both `n_random_starts` and `random_state`. SPA is
documented as deterministic and matches canonical Araújo 2001 enumeration.

## 3. Commercial / open-source software sanity check

| Implementation | Initial variable strategy | Random restarts? | Configurable count? |
|---|---|---|---|
| Araújo 2001 canonical | Loop over all J variables as seed | No (deterministic) | No (always J) |
| Galvão 2012 SPA-GUI | Loop over all J variables as seed | No | No |
| `auswahl` (Python) | Every variable as seed | No | No |
| `prospectr` (R) | Does not implement SPA | n/a | n/a |
| dasp pre-T-06 | argmax-correlation only, repeated N times | No (despite name) | Yes — but did nothing |
| **dasp post-T-06** | **Loop over all J variables as seed** | **No (canonical)** | **No** |

The pre-fix dasp pattern matched no field implementation — it was neither
random restarts nor canonical enumeration, just a single argmax chain repeated
N times. The fix moves dasp into alignment with canonical Araújo 2001 +
auswahl.

## 4. GUI reachability

Pre-fix: `gui:3725` (`IntVar(value=10)`) → `gui:12084-12086` (Spinbox 1-50,
default 10, label "SPA Random Starts:") → `gui:26794, 27486` (passed to
`run_search` and `run_one_class_search`) → `search.py:887, 5061` (signature
parameters) → `search.py:2497, 2528, 2611, 2627, 5727, 5755, 5815`
(`spa_selection` and 3 hybrid call sites) → noop loop in
`variable_selection.py:401`.

Post-fix: every site removed. The `params_frame` grid in the Variable
Selection panel renumbered to close the gap (rows 3–8 → 2–7, rows 9–11 →
8–10).

The fix is reachable to the bundled-app user without any new GUI plumbing —
the existing parameter pane simply has one fewer (non-functional) row.

## 5. Test

```bash
.venv312/Scripts/python.exe -m pytest \
    tests/test_variable_selection.py \
    tests/test_wavelength_selection.py \
    tests/test_golden_standard_performance.py \
    tests/test_nspfce.py \
    tests/smoke/test_imports.py \
    tests/test_cv_strategy.py \
    tests/test_search_comprehensive.py \
    tests/test_vip_formula.py \
    tests/test_pds_window_arithmetic.py \
    tests/test_cv_pls_clamp.py \
    -v
```

Expected: 226 passed (after adding 5 new T-06 tests + Codex/Kimi review fixes).

The 5 new T-06 tests in `tests/test_variable_selection.py`:
1. **`test_spa_deterministic`** — same input → same output across calls.
2. **`test_spa_explores_multiple_seeds`** — pinned-data fixture
   (`default_rng(0)`, 30×15) where canonical SPA picks chain `[0, 5, 6, 7, 10]`
   from seed 6, while argmax-only would pick `[3, 5, 8, 10, 11]` from seed 11.
   Asserts both selections explicitly.
3. **`test_spa_evaluates_all_j_seeds`** — patches `cross_val_score` to
   count calls; asserts exactly J calls happen (catches "loop accidentally
   narrowed to range(10)" regression).
4. **`test_spa_rejects_legacy_n_random_starts`** — TypeError if kwarg passed.
5. **`test_spa_rejects_legacy_random_state`** — TypeError if kwarg passed.

## 6. Verdict

**REAL BUG per the literature (canonical Araújo 2001 enumeration not implemented)
+ MISLEADING USER-FACING SURFACE (GUI knob doing nothing).**

Reasoning:
- The `n_random_starts` knob was non-functional (verified empirically:
  byte-identical output for any value of the parameter).
- The proposed fix in the original roadmap (`rng.choice()` for random first
  variable) was the master-rule failure mode: an sklearn-style stochastic
  restart pattern with no chemometrics-field precedent. No verified
  implementation in any chemometrics literature or tooling exposes random
  restarts for SPA.
- Canonical Araújo 2001 is **deterministic enumeration over all J seeds**.
  Both `auswahl` (modern open-source Python) and Galvão 2012 SPA-GUI follow
  this pattern. The fix moves dasp into alignment with the field default.
- The bundled-app distribution model (T-26 lesson) is satisfied: the fix is
  fully reachable from the GUI without new plumbing, and the misleading knob
  is removed from the user surface.

## 7. Cross-family review

Both Codex (US-trained) and Kimi K2.6 (Moonshot, Chinese-trained) reviewed
the diff against the validation-gate methodology. Both confirmed:
- Canonical Araújo 2001 inner forward-projection math is unchanged from
  pre-fix; only the outer seed loop differs.
- No off-by-one or seed-set bugs in the new `for first_var in range(n_vars)`
  enumeration.
- No state leakage between seeds.

Both flagged the same MAJOR test-weakness on the original soft
`test_spa_explores_multiple_seeds`. Resolved by replacing it with a
deterministic pinned-data assertion + adding a complementary call-count
invariant test (Kimi's suggestion).

Codex caught a missed call-site (`tests/gui/test_comprehensive.py:387`) —
fixed.

Codex caught template ↔ in-app divergence on small-sample CV-fold reduction
and failure-fallback semantics — fixed (template now mirrors production's
small-sample fold adjustment and uniform-fallback behavior).

Kimi caught dead `y_norm` computation in production (leftover from the
argmax-correlation seed path) — removed.

Kimi caught Python-loop projection in template (~2-3 orders of magnitude
slower than production's vectorized matmul on FTIR-scale J=2000) — vectorized.

## 8. Performance note (deferred follow-up)

Canonical enumeration is O(J seeds) × O(N×J inner forward chain) × O(folds ×
PLS CV). For typical bone-FTIR data after preprocessing (J = 200-1500
wavelengths), this is **100×–1500× the prior single-chain-repeated-10-times
work**. Pre-fix typical SPA runtime: ~1-5 sec; post-fix: ~30-750 sec on the
high end of J.

This is canonical-correct but visible. Deferred follow-up options:
1. Parallelize the seed loop with joblib (independent CV evaluations);
   constrain inner `cross_val_score(n_jobs=1)` to avoid PyInstaller bundle
   issues per `_frozen_needs_threading_fallback`.
2. Vectorize the inner forward chain across seeds (heavier rewrite).
3. Top-K seeds only (bounded enumeration) — but this would be option D from
   the original triage and was rejected as non-canonical.

If users report visible slowdowns in real bone-FTIR workflows, option 1 is
the smallest change. Tracked as informal "T-06 perf follow-up" in
PROJECT_STATUS.

## 9. Related observations (out of scope)

- `bayesian_utils.py:444, 481` previously hardcoded `spa_n_random_starts =
  10`, ignoring any (non-existent) GUI plumbing into the Bayesian path. Both
  hardcodes removed by this commit. Future Bayesian-path GUI plumbing for
  other varsel knobs should use the same pattern (read from GUI, pass
  through).
- `src/spectral_predict/wavelength_selection.py` is a parallel,
  unused-in-production SPA implementation (only imported by tests and its
  own docstring examples). Possibly historical scaffolding from an earlier
  refactor plan. Out of T-06 scope; flagged for a future cleanup ticket
  (call it T-06b: "remove or merge dead-code parallel SPA module").
- `cross_val_score(scoring='r2')` in `spa_selection` is internally
  inconsistent with iPLS / CARS / GA-PLS (which optimize RMSECV) per Kimi's
  observation. For fixed `y`, R² maximization is monotonically equivalent to
  MSE minimization, so this is a code-style nit not a correctness bug.
  Pre-existing; not introduced by T-06.
