# T-06 SPA n_random_starts non-functionality — investigation findings

**Status:** INVESTIGATION COMPLETE — verdict pending synthesis by calling agent
**Date:** 2026-04-30
**Author:** Opus 4.7 (1M context)

---

## TL;DR

The roadmap framing is **substantially correct** on the code-reading: `variable_selection.py:293-486`'s `spa_selection` is fully deterministic. Every iteration of the `for start_idx in range(n_random_starts)` loop produces the *exact same* selection because the first variable is hardcoded to `np.argmax(initial_corrs)` (line 407) and nothing else inside the loop varies across iterations. `random_state` is accepted in the signature (line 293) and never used inside the function body. The function's own docstring at line 322-324 already concedes this: *"currently SPA is deterministic, but this parameter is included for API consistency and future enhancements."*

Empirical confirmation (Step 4 below): with `n_random_starts ∈ {1, 5, 10}` and `random_state ∈ {42, 123}`, the function returns **byte-identical importances** in every case. The `n_random_starts` GUI knob at `gui:12085` (Spinbox 1–50, default 10) burns CPU on N redundant identical computations and produces zero benefit.

The fix the roadmap proposes (`rng.choice()` for random first-variable starts) is, however, **not what canonical Araujo 2001 SPA actually does**. The canonical algorithm iterates *deterministically over every variable as a candidate starting variable* and keeps the chain with the best CV criterion — not random starts. So the surface-level bug ("`n_random_starts` does nothing") is real, but the proposed fix would invent a non-canonical pattern. A field-aligned fix would either (a) remove the knob and document SPA as deterministic per Araujo 2001 (matching the docstring), or (b) replace the loop with a deterministic loop over candidate seeds (what Araujo 2001 actually specifies).

The user-facing GUI control exists (Spinbox 1-50, default 10) — so this is not dead code. A user who reads "SPA Random Starts: 10" reasonably believes the algorithm explores 10 starts, but it explores only 1 (10x).

---

## Step 1 — Reality in the codebase

### 1a. Function signature and docstring acknowledgment

`src/spectral_predict/variable_selection.py:293`:

```python
def spa_selection(X, y, n_features, n_random_starts=10, cv_folds=5, random_state=42):
```

Docstring at line 322-324:

```
random_state : int, default=42
    Random seed for reproducibility (currently SPA is deterministic, but
    this parameter is included for API consistency and future enhancements)
```

So the function's own author already documented the bug and labelled `random_state` as "future-enhancement parking" rather than wired-up. This contradicts the docstring at line 318-319 that describes `n_random_starts` as "Number of random initializations" — there are no random initializations.

### 1b. The loop body — verbatim quote, lines 398-470

```python
398:    print(f"Running SPA with {n_random_starts} random starts...")
399:
400:    # Step 2: Run multiple random starts
401:    for start_idx in range(n_random_starts):
402:        # Initialize: select variable with max correlation with y
403:        selected_indices = []
404:        available_indices = set(range(n_vars))
405:
406:        # First variable: highest correlation with y
407:        first_var = np.argmax(initial_corrs)
408:        selected_indices.append(first_var)
409:        available_indices.remove(first_var)
410:
411:        # Iteratively select remaining variables (n_features - 1 more)
412:        for step in range(1, n_features):
413:            # Compute projections for all available variables at once
414:            ...
422:            avail_idx = np.array(sorted(available_indices))
423:            # X_avail_norm has shape (n_samples, len(avail_idx))
424:            X_avail_norm = X_norm[:, avail_idx]
425:            # Matrix multiply ...
426:            ...
427:            corr_matrix = (X_avail_norm.T @ X_selected_norm) / n_samples
428:            # Projection for each available var = sum of squared correlations with selected set
429:            proj_values = np.sum(corr_matrix ** 2, axis=1)
430:
431:            # Select variable with MINIMUM projection (least correlated with selected set)
432:            min_proj_var = avail_idx[np.argmin(proj_values)]
433:
434:            selected_indices.append(min_proj_var)
435:            available_indices.remove(min_proj_var)
436:
437:        # Step 3: Evaluate this selection using PLS with cross-validation
438:        try:
439:            ...
449:            cv_scores = cross_val_score(
450:                pls, X_selected, y,
451:                cv=cv_folds,
452:                scoring='r2',
453:                n_jobs=1
454:            )
455:            mean_score = np.mean(cv_scores)
456:
457:            # Track best selection ...
458:            if not np.isnan(mean_score) and not np.isinf(mean_score):
459:                if mean_score > best_score:
460:                    best_score = mean_score
461:                    best_selection = selected_indices.copy()
462:                    print(f"  Start {start_idx+1}/{n_random_starts}: R² = {mean_score:.4f} (new best)")
463:                else:
464:                    print(f"  Start {start_idx+1}/{n_random_starts}: R² = {mean_score:.4f}")
465:            ...
```

### 1c. Why every iteration is identical

Every variable that enters the loop body (line 401) — `initial_corrs`, `X_norm`, `n_samples`, `n_features`, `n_vars`, `cv_folds`, `cv_scores` (deterministic given X, y, cv_folds because `cross_val_score` with default cv=int just uses sequential KFold) — is fixed before the loop starts. There is **no source of randomness inside the loop**: `np.argmax`, `set(range(...))`, the deterministic projection arithmetic, and `cross_val_score` with an integer `cv` argument are all reproducible. `random_state` is never referenced anywhere in the function body. So `for start_idx in range(n_random_starts)` is effectively `for start_idx in range(n_random_starts): redo_the_same_work_again()`.

The roadmap's line-number claim ("variable_selection.py:401-408") is correct on the loop start and the argmax line; the loop actually runs to ~470. The verb "every start picks `np.argmax(initial_corrs)` — the same starting variable" is accurate.

---

## Step 2 — GUI reachability

### 2a. The GUI control exists

`spectral_predict_gui_optimized.py:3725` — Tk variable:

```python
self.spa_n_random_starts = tk.IntVar(value=10)  # Random starts for SPA
```

`spectral_predict_gui_optimized.py:12084-12086` — Spinbox widget:

```python
ttk.Label(params_frame, text="SPA Random Starts:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=5)
ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.spa_n_random_starts, width=8).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
ttk.Label(params_frame, text="(default: 10)", style='Caption.TLabel').grid(row=2, column=2, sticky=tk.W, padx=10)
```

Range 1–50, default 10. This appears in the Variable Selection panel (params_frame), labelled "SPA Random Starts". It is fully reachable from the GUI to a non-technical user.

### 2b. Plumbing from GUI → search → spa_selection

The user's spinbox value flows through:

- `gui:26799`: `spa_n_random_starts=self.spa_n_random_starts.get()` — passed to the run-search call (regression / one-class grid search).
- `gui:27492`: same arg, passed to a second run-search entry point (Bayesian/optimization branch).
- `search.py:887`: `run_search(... spa_n_random_starts=10, ...)` signature accepts the user value.
- `search.py:1093`: `if spa_n_random_starts != 10:` — non-default values are tagged in the result label, so the value flows downstream.
- `search.py:2500`: `spa_selection(X, y, n_features, n_random_starts=spa_n_random_starts, ...)` (regression grid search per-fold path).
- `search.py:2528`: `uve_spa_selection(... spa_n_random_starts=spa_n_random_starts ...)` (UVE+SPA hybrid).
- `search.py:2615`, `:2631`: regional/extra paths.
- `search.py:5070`: `run_one_class_search(... spa_n_random_starts=10, ...)` accepts the value.
- `search.py:5739`: `spa_selection(... n_random_starts=spa_n_random_starts ...)` (one-class direct SPA).
- `search.py:5767`, `:5827`: one-class hybrid SPA paths.
- `bayesian_utils.py:444, 481`: hardcoded `spa_n_random_starts = 10` (NOT plumbed from GUI; uses defaults). The "Old Bayesian" path uses defaults only.

So the user's GUI value reaches `spa_selection` via every regression and one-class grid-search SPA call path. It does not reach the legacy Bayesian path (`bayesian_utils.py`), but does reach the unified Bayesian path (which uses the same `compute_importances` infrastructure).

### 2c. Non-default values are user-visible in result labels

`search.py:1093` tags non-default `spa_n_random_starts` in result labels:

```python
if spa_n_random_starts != 10:
    ...  # add to subset tag
```

So a user who sets `n_random_starts=20` sees the tag in their results table, reinforcing the (false) impression that the value affects the search.

### 2d. GUI label is misleading

The label "SPA Random Starts:" with caption "(default: 10)" leads a user to reasonably believe SPA is doing 10 random initializations. The function does 1 initialization repeated 10 times. A user comparing 10 starts vs 50 starts will see no result difference but will see SPA take 5x longer.

---

## Step 3 — Field alignment with documentation lookup

### 3a. Araujo 2001 SPA — what the canonical paper actually specifies

The Araujo 2001 paper (*Chemom Intell Lab Syst* 57:65-73) is the canonical specification dasp's docstring cites at variable_selection.py:350-352. Web research (the paper's PDF is not directly fetchable but its algorithmic structure is well-documented in secondary sources):

> "The iterative scheme of the algorithm makes the initially selected variable a degree of freedom; therefore, **SPA considers every variable as candidate seed and subsequently selects the variable set with a maximum CV performance**."

(Multiple sources echo this — quoted from the PMC SPA-as-initialization paper at https://pmc.ncbi.nlm.nih.gov/articles/PMC5573288/ which references Araujo's algorithm; same description appears in the auswahl Python library docs at https://auswahl.readthedocs.io/en/latest/point_selection.html).

A direct paraphrase from a SPA description deriving from the original paper, re-quoted via a community implementation:

> "the first wavelength k(0) and the number N are given… For initials 1 until J do… For N s Nmin to Nmax do… Using Steps 0–5 above, select N wavelengths starting from k(0) = initial. … k*(0) = arg min r(N), initial = 1, …, J"

That is: **the canonical Araujo 2001 algorithm iterates DETERMINISTICALLY OVER EVERY VARIABLE as a candidate first variable** (initial = 1 to J, where J is the number of variables), runs the full forward-projection chain from each, and keeps the one with the best validation criterion. There is **no random restart** in the canonical algorithm — the loop is fully enumerative over candidate seeds.

So the dasp `n_random_starts` knob is misnamed in two ways:

1. There are no "random" starts in the canonical SPA algorithm (it's a deterministic enumeration).
2. The number of starts is canonically `J = n_variables` (every variable tried as seed), not a user-configurable count.

dasp's current behavior — argmax-correlation seed only, repeated `n_random_starts` times — is *neither* random restarts *nor* canonical Araujo 2001. It's a single deterministic chain repeated `n_random_starts` times for no reason.

### 3b. PLS_Toolbox / Solo (Eigenvector)

I could not locate explicit Eigenvector documentation for SPA implementation details. The Eigenvector "Variable Selection" wiki page (https://www.eigenvectordocs.com/index.php?title=Variable_Selection) was 403-blocked from this session. Per the chemometrics literature, MATLAB SPA toolboxes that derive from Araujo's group (e.g. the "spa_gui" GUI described in Galvão et al. 2012, *Chemom Intell Lab Syst*) implement the deterministic-over-J-seeds approach. Eigenvector's PLS_Toolbox is widely understood to follow the same pattern.

I am NOT claiming PLS_Toolbox specifically — I could not verify it from documentation in this session. I am only claiming the canonical algorithm structure that Eigenvector-style implementations follow.

### 3c. auswahl (Python) — modern open-source SPA

The `auswahl` Python library (https://auswahl.readthedocs.io/en/latest/point_selection.html) explicitly states:

> "The iterative scheme of the algorithm makes the initially selected variable a degree of freedom. Therefore, SPA considers every variable as candidate seed and subsequently selects the variable set with a maximum CV performance."

API signature: `SPA(n_features_to_select, n_cv_folds, pls, ...)`. **No `n_random_starts` parameter.** **No `random_state`.** Auswahl is explicitly enumerating-over-seeds, deterministic, with no randomness exposed.

### 3d. prospectr (R)

`prospectr` does NOT implement SPA. Its sample-selection functions are `kenStone`, `naes`, `duplex`, `shenkWest`, `puchwein`, `honigs`. (Verified via the package's GitHub README.) Not a relevant comparison.

### 3e. Other implementations checked

- The Springer 2017 Journal of Applied Spectroscopy paper "SPA-MLR Classifier for Detection of Contaminants on Chicken Carcasses in Hyperspectral Images" describes SPA as a forward-selection-with-deterministic-seed-enumeration, no random restarts.
- Wiley 2020 "Moving-Window-Improved MC-UVE Combining SPA for NIRS" same description.
- Jiang 2010 IEEE paper on UVE-SPA: deterministic enumeration over seeds.

The 2001 Araujo paper, the 2012 Galvão Chemom Intell Lab Syst paper, and modern open-source replicas (auswahl) all describe the same enumerate-over-J-seeds structure. I find **no chemometrics implementation in any verified source that exposes a "random restart count" parameter for SPA.**

### 3f. Summary of field alignment

| Implementation | Initial variable strategy | Random restarts? | Configurable count? |
|----------------|---------------------------|------------------|---------------------|
| Araujo 2001 canonical | Loop over all J variables as seed | No (deterministic) | No (always J) |
| Galvão 2012 SPA-GUI | Loop over all J variables as seed | No | No |
| auswahl (Python) | Per docs: every variable as seed | No | No |
| prospectr (R) | Does not implement SPA | n/a | n/a |
| dasp `spa_selection` | argmax-correlation, repeated N times (always identical) | No (despite name) | Yes — but does nothing |

The dasp pattern matches no known field implementation. The closest field analog would be a single-seed argmax-correlation initialization (which is what Centner-style PLS-coefficient ranking would give as the "best" first variable), repeated only once. The `n_random_starts` knob is a dasp-invented surface that does nothing useful.

---

## Step 4 — Empirical demonstration

### 4a. Script

`tests/_t06_empirical.py` (created during this investigation; `_` prefix prevents pytest auto-collection):

```python
# Synthesize 20×100 dataset, 3 informative wavelengths
X, y, informative_idx = synthesize_dataset(seed=0)

# Test cases:
# 1. random_state=42, n_random_starts=5 (call A)
# 2. random_state=42, n_random_starts=5 (call A repeat — sanity check determinism)
# 3. random_state=123, n_random_starts=5 (different RNG seed)
# 4. random_state=42, n_random_starts=1 (single iteration)
# 5. random_state=42, n_random_starts=10 (more iterations)
```

### 4b. Output (verbatim, .venv312)

```
Dataset: X.shape=(20, 100), informative wavelengths at [7, 52, 92]

--- Single call with n_random_starts=5, random_state=42 ---
Running SPA with 5 random starts...
  Start 1/5: R² = -7.1281 (new best)
  Start 2/5: R² = -7.1281
  Start 3/5: R² = -7.1281
  Start 4/5: R² = -7.1281
  Start 5/5: R² = -7.1281
Top 10 selected (random_state=42, n_random_starts=5): [52, 37, 16, 58, 13, 89, 43, 40, 90, 79]

--- Re-call with the same args (random_state=42, n_random_starts=5) ---
[exact same trace]
Top 10 selected (repeat): [52, 37, 16, 58, 13, 89, 43, 40, 90, 79]
Identical to first run? True

--- Call with different random_state=123, n_random_starts=5 ---
[exact same trace]
Top 10 selected (random_state=123): [52, 37, 16, 58, 13, 89, 43, 40, 90, 79]
Identical to random_state=42 run? True

--- Call with n_random_starts=1 ---
Running SPA with 1 random starts...
  Start 1/1: R² = -7.1281 (new best)
Top 10 selected (n_random_starts=1): [52, 37, 16, 58, 13, 89, 43, 40, 90, 79]
Identical to n_random_starts=5 run? True

--- Call with n_random_starts=10 ---
Running SPA with 10 random starts...
  [10 identical R² = -7.1281 lines]
Top 10 selected (n_random_starts=10): [52, 37, 16, 58, 13, 89, 43, 40, 90, 79]
Identical to n_random_starts=1 run? True

CONCLUSIONS
random_state=42 vs 123 produce identical output: True
n_random_starts=1 vs 5 produce identical output: True
n_random_starts=1 vs 10 produce identical output: True
```

### 4c. Conclusions from empirical run

1. **Every iteration of the `n_random_starts` loop produces the exact same R² and the exact same selection.** All 10 lines in the n_random_starts=10 trace report `R² = -7.1281` to 4 decimal places. Only the first one is tagged "(new best)" because subsequent ties don't strictly exceed `best_score`.
2. **`random_state=42` vs `random_state=123` produce byte-identical `importances` arrays.** The parameter is fully unused.
3. **`n_random_starts=1` vs `5` vs `10` produce byte-identical `importances` arrays.** The parameter has zero effect on output.
4. **Wall-clock cost scales linearly with `n_random_starts`.** A user setting the GUI knob to 50 pays 50x the SPA cost for zero benefit.
5. **The selection misses the informative wavelengths.** Informative wavelengths are at indices [7, 52, 92]; only 52 appears in the top-10 selection. SPA's primary objective is collinearity reduction, not informativeness — so this isn't a SPA bug per se, but it does illustrate that the algorithm being deterministic-from-argmax means a user cannot escape this single chain even with the `n_random_starts` knob cranked up.
6. **The R² is negative (-7.1281)**, which means the cross-validated PLS R² across the SPA-selected wavelengths is far worse than predicting the mean. With a 20-sample dataset and 3 informative wavelengths buried in 97 noise wavelengths, this is expected. The point is that 10 redundant identical runs don't help find a better selection.

---

## Step 5 — Test sweep readiness

### 5a. Tests that touch `variable_selection.spa_selection` directly

`pytest tests/ -k spa --collect-only` finds 38 SPA-related collected tests. Specifically relevant for spa_selection in `variable_selection.py`:

- `tests/test_variable_selection.py::TestVariableSelection::test_spa_basic` (line 232)
- `tests/test_variable_selection.py::TestVariableSelection::test_spa_reduces_collinearity` (line 261)
- `tests/test_variable_selection.py::TestVariableSelection::test_uve_spa_basic` (line 311)
- `tests/test_variable_selection.py::TestVariableSelection::test_uve_spa_hybrid` (line 348)
- additional SPA-touching tests at lines 405-509 within the same file (e.g. small-dataset edge case at 408, hybrid coverage at 421, 462, 493, 509, 512)

### 5b. Tests that touch `wavelength_selection.spa` (a DIFFERENT module)

`tests/test_wavelength_selection.py` imports `from spectral_predict.wavelength_selection import spa, cars, vcpa_iriv` — this is a **separate, parallel SPA implementation** in `src/spectral_predict/wavelength_selection.py`. The relevant tests:

- `TestSPA::test_spa_basic_functionality`
- `TestSPA::test_spa_selects_informative_wavelengths`
- `TestSPA::test_spa_handles_collinearity`
- `TestSPA::test_spa_different_n_vars`
- `TestSPA::test_spa_invalid_inputs`
- `TestSPA::test_spa_deterministic` (line 171) — **explicitly asserts SPA is deterministic** (`np.testing.assert_array_equal(result1['selected_indices'], result2['selected_indices'])`)

Note: `wavelength_selection.spa` is not imported by any production code (only by tests, per Grep — `from spectral_predict.wavelength_selection` appears only in `test_wavelength_selection.py` and within `wavelength_selection.py`'s own docstring examples). So this is a separate, unused module — possibly historical scaffolding from an earlier refactor plan (per `docs/plans/2026-02-07-asp-implementation-plan.md`). **A T-06 fix to `variable_selection.spa_selection` would not affect these tests, and they would not catch a regression in the actual production path.**

### 5c. Integration test

`tests/test_golden_standard_performance.py::test_variable_selection_spa_correctness` (line 150) runs a full `run_search` pipeline with `variable_selection_methods=["spa"]`. This exercises `variable_selection.spa_selection` end-to-end via the search dispatch.

### 5d. Recommended test sweep invocation

```bash
.venv312/Scripts/python.exe -m pytest \
    tests/test_variable_selection.py \
    tests/test_wavelength_selection.py \
    tests/test_golden_standard_performance.py \
    tests/test_nspfce.py \
    -v
```

Total: 47 tests collected across the three primary files (per the `pytest --collect-only` run during this investigation), plus `test_nspfce.py::TestNSPFCEBasic::test_nspfce_with_wavelength_selection_spa`.

I did not actually run the test sweep — that's the calling agent's job in the verdict step. Just identified the right invocation.

---

## Step 6 — Distribution-model check

### 6a. The fix is NOT dead-code-without-GUI-work

Unlike T-26 (where a backend-only `offset` parameter would have been unreachable from the bundled app), T-06 has a fully reachable GUI control:

- `spectral_predict_gui_optimized.py:12085`: `ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.spa_n_random_starts, width=8)` is rendered in the Variable Selection panel of the main GUI.
- Default value `10` (`gui:3725`).
- Tooltip caption `"(default: 10)"`.
- Label text `"SPA Random Starts:"`.
- The user can change the value through the GUI today; the value reaches `spa_selection` in `variable_selection.py`; the function ignores it and does the same work N times.

So the fix surface is reachable to the bundled-app user **without any new GUI plumbing**. Whatever code change is made will deliver behavior that the existing GUI knob already advertises.

### 6b. Three options and their distribution-model implications

**Option A — Wire `random_state` to randomize the first variable.** This is the roadmap's proposed fix (`rng.choice()` for random first-variable starts).

- Pro: matches the GUI label literally — `n_random_starts` will become "number of random starts" in fact.
- Pro: zero new GUI plumbing.
- Con: **invents a non-canonical pattern not in any verified field implementation.** Araujo 2001 does not specify random restarts.
- Con: random initialization may produce a worse selection than argmax-correlation on small / noisy datasets (because pure-random first variable is uninformed about y; argmax-correlation at least uses y).
- Con: would break the existing `tests/test_wavelength_selection.py::TestSPA::test_spa_deterministic` test if applied to the wrong module — but T-06 is about `variable_selection.spa_selection`, so it would not (since the test imports from `wavelength_selection`).

**Option B — Loop deterministically over candidate first variables (canonical Araujo 2001).** Replace `for start_idx in range(n_random_starts)` with `for first_var in range(n_vars)` (or top-K-by-correlation as a tunable hybrid). Drop or repurpose the `n_random_starts` knob.

- Pro: matches Araujo 2001 exactly.
- Pro: matches auswahl.
- Con: O(J) seeds × O(J) inner forward selection × O(folds) CV ≈ J²×folds operations, which for FTIR with J=1000 wavelengths gets expensive (10M ops at 5 folds — manageable but visible).
- Con: requires GUI changes — the existing "SPA Random Starts" Spinbox no longer makes sense; would need to relabel, change semantics, or remove.
- Con: existing `test_spa_deterministic` would still pass (canonical SPA *is* deterministic), but other tests asserting specific selections may break depending on whether they used the current `argmax`-only chain.

**Option C — Document SPA as deterministic, remove the `n_random_starts` knob.**

- Pro: matches the function's own current behavior + docstring.
- Pro: removes the misleading GUI label.
- Pro: simplest possible diff; no algorithmic risk.
- Con: removes a user-facing knob the user may have relied on (even though it does nothing) — values >10 in saved configurations would need migration handling.
- Con: less "fix" and more "admit the truth"; would need GUI label change or removal of the spinbox.

**Option D — Top-K-by-correlation seeds, deterministic.** Loop over the top K initial-correlation candidates (where K is the user's `n_random_starts` value), repurposing the knob from "random restarts" to "deterministic top-K seeds" — like a partial Araujo enumeration with the knob controlling depth.

- Pro: salvages the `n_random_starts` knob meaningfully (now controls deterministic diversity).
- Pro: bridges current behavior (argmax = top-1) to canonical Araujo (top-J = full enumeration).
- Pro: cheap — 10 seeds × forward chain × CV scales linearly with the existing knob.
- Con: invents a "partial Araujo" pattern. Defensible but not directly cited in the field.
- Con: requires GUI label change ("SPA Random Starts" → "SPA Seed Candidates" or similar).

### 6c. The user's GUI experience under each option

- **Today:** "SPA Random Starts: 10" — no effect; user pays 10x time for nothing; selection identical at 1, 10, 50.
- **Option A:** "SPA Random Starts: 10" — actually random; results vary across runs unless `random_state` is fixed; chemometrics literature does not support this pattern.
- **Option B:** Spinbox removed or repurposed; SPA always runs J seeds; runtime ~J× current; selections better but slower; matches Araujo 2001.
- **Option C:** Spinbox removed; runtime ~10× faster; behavior unchanged; user-facing simplification.
- **Option D:** "SPA Seed Candidates: 10" — top-10 most-correlated wavelengths each tried as seed; deterministic; runtime ~equal to current; selection diversity gained.

### 6d. Recommendation framing for verdict-writer

A defensible chemometrics-aligned fix would be **C or D, not A**:

- **C** (drop the knob) is the most conservative — admits the bug exists, removes the misleading knob, leaves SPA as the documented argmax-only deterministic chain. Honest about field convention being "deterministic SPA". Cost ~30 min of GUI cleanup.
- **D** (top-K seeds, deterministic) salvages the knob in a chemometrics-aligned way — partial Araujo enumeration controlled by user. Cost ~1-2 hours.
- **B** (full Araujo enumeration) is most field-aligned but most expensive computationally and diverges most from today's behavior.
- **A** (the roadmap's proposed `rng.choice()` random restart fix) is the **least field-aligned** option — invents a pattern no canonical reference uses.

This is precisely the kind of "sklearn-instinct fix that the field doesn't actually use" pattern the master rule warns about. The roadmap framing identifies a real bug but proposes a fix that follows ML-style stochastic-restart conventions rather than chemometrics-style deterministic-enumeration conventions.

---

## Open questions for the calling agent

1. **Scope vs. invented-pattern tradeoff.** The roadmap's `rng.choice()` fix would make the code match the current GUI label literally, but at the cost of adopting a stochastic-restart pattern with no field precedent. Is that acceptable? Or should the verdict steer toward C/D instead?

2. **`wavelength_selection.spa` (the parallel module).** This unused-in-production module also has its own SPA; its `test_spa_deterministic` test would not be touched by a T-06 fix to `variable_selection.spa_selection`. Should T-06's scope expand to the parallel module, or should that be a separate cleanup ticket noting the dead-code parallel implementation?

3. **The `random_state` parameter.** Once the bug is acknowledged, should the parameter be removed entirely (Option C) or kept-but-actually-wired (Options A/B/D)? Removing has a backward-compat surface (saved configs may include it).

4. **Performance of canonical Araujo.** For high-dimensional FTIR data (J = 800-3000 wavelengths), full canonical Araujo enumeration (Option B) could add measurable wall-clock time. Top-K (Option D) is a sensible compromise. Should the verdict pick a numerical default for K?

5. **Existing `test_spa_deterministic` in `wavelength_selection`.** That test asserts SPA produces identical output across two calls. Under Option A (random restarts), `wavelength_selection.spa` would still be deterministic (different module), but `variable_selection.spa_selection` would not be. Should there be a new test that *requires* `n_random_starts` to actually vary the output?

6. **Default `n_random_starts` value.** Today the spinbox defaults to 10. Under Option A, that would mean every SPA run does 10 PLS-CV evaluations (slow). Under Option D, default 10 means top-10 candidates (acceptable). Under Option C, the parameter is removed. The default choice has UX implications.

7. **Interaction with leakage audit (T-01).** `docs/T01_VARSEL_LEAKAGE_AUDIT.md` identifies SPA as leaky in the grid-search and Bayesian paths because it uses full `y` for its internal MLR fitness. A T-06 fix that adds randomness to first-variable selection does not address that leakage; it's an orthogonal issue. The verdict-writer may want to note this.

---

## Sources

- [Araújo et al. 2001 - The successive projections algorithm for variable selection in spectroscopic multicomponent analysis - Chemom Intell Lab Syst 57:65-73 (paywalled)](https://www.sciencedirect.com/science/article/abs/pii/S0169743901001198)
- [auswahl Python library - Wavelength Point Selection (SPA reference)](https://auswahl.readthedocs.io/en/latest/point_selection.html)
- [auswahl SPA basic example](https://auswahl.readthedocs.io/en/latest/auto_examples/plot_spa_features.html)
- [Galvão et al. 2012 - A graphical user interface for variable selection employing SPA - Chemom Intell Lab Syst (paywalled)](https://www.sciencedirect.com/science/article/abs/pii/S0169743912001347)
- [PMC 5573288 - SPA as initialization method, with summary of canonical algorithm](https://pmc.ncbi.nlm.nih.gov/articles/PMC5573288/)
- [Soares Jiang 2010 - UVE+SPA hybrid (IEEE; description matches deterministic-seed-enumeration)](https://ieeexplore.ieee.org/document/5303610/)
- [prospectr R package - functions list (does not include SPA)](https://github.com/l-ramirez-lopez/prospectr)
- [Eigenvector Variable Selection wiki (403 in this session, referenced for completeness)](https://www.eigenvectordocs.com/index.php?title=Variable_Selection)
- dasp source: `src/spectral_predict/variable_selection.py:293-486`, `src/spectral_predict/search.py:887, 1093, 2500, 2528, 5070, 5739`, `spectral_predict_gui_optimized.py:3725, 12084-12086, 26799, 27492`, `tests/test_variable_selection.py`, `tests/test_wavelength_selection.py:171`, `tests/test_golden_standard_performance.py:150`
- Empirical script: `tests/_t06_empirical.py`
