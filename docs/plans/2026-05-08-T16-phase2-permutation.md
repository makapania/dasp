# T-16 Phase 2 — On-demand permutation test for arbitrary leaderboard rows

**Branch:** `feat/T16-phase2-permutation` (off `main`)
**Status:** plan v1, awaiting Codex review
**Effort estimate:** ~2-3 days source + tests + GUI
**Depends on:** none in code; Phase 1 (PR #60) is independent — Phase 2 doesn't import or rely on Phase 1's CV-ANOVA code.

---

## 1. Goal

User can right-click any leaderboard row and ask "is this model real, or did its CV score arise by chance under shuffled labels?" The dialog runs N permutations (default 100; chemometrics convention; matches SIMCA), shows a progress bar, and reports a p-value with a null-distribution histogram.

Where Phase 1 stops short — non-PLS-regression rows (LightGBM, RF, SVM, MLP, **including dasp's PLS-DA hybrid**) — Phase 2 picks up. The permutation test is **model-agnostic by design**: it shuffles Y, refits the pipeline, and asks how often shuffled-Y scores match or beat the observed score. No closed-form math, no PLS-on-Y assumption.

Out of scope for Phase 2:
- Q2 paired model-vs-model tests (Phase 3 — separate ticket per the T-16 survey doc)
- Bootstrap CI on individual model parameters
- Bayesian posterior credible intervals

---

## 2. Statistical design

**The null hypothesis.** "The CV metric (R²cv for regression, Accuracycv / BalancedAcccv / F1cv for classification) achieved by this row is no better than what would arise if Y were random with respect to X."

**Procedure.** For `n_permutations` (default 100) iterations:
1. Permute Y (random shuffle of the calibration labels).
2. Refit the pipeline on (X, Y_permuted) using the same CV strategy that produced the observed metric.
3. Record the resulting null-CV-metric.

**p-value:** `(1 + count(null_metric ≥ observed_metric)) / (1 + n_permutations)`. The +1 in numerator and denominator is the standard Monte-Carlo p-value adjustment (Phipson & Smyth 2010): never returns exactly 0, conservatively bounded.

**Reference:** Westad & Marini 2015 (cited extensively in dasp docs already). Also implicit in SIMCA's permutation test on Q², mdatools' `randtest()` (single-model variant), and the broader chemometrics literature.

---

## 3. Design decisions

### 3.1 What gets permuted: model-conditional vs full-pipeline

**Choice — ship "model-conditional" as Phase 2a default; offer "full-pipeline" as a future flag.**

| Mode | What's refit per permutation | Cost per permutation | Interpretation |
|---|---|---|---|
| **Model-conditional** (Phase 2a default) | Just the model, on already-preprocessed and variable-selected X | seconds for PLS, ~10-60s for LightGBM/RF | "Given the feature set we ended up using, does the model find real signal in Y?" |
| **Full-pipeline** (Phase 2b — flag, not default) | Preprocessing → variable selection → model | minutes for varsel rows (CARS, GA-PLS, UVE, SPA, iPLS each cost seconds-to-minutes per fit) | "Would this whole search pipeline produce a score this good by chance?" |

Why model-conditional is the right default:
- Matches SIMCA's permutation-on-Q² convention (SIMCA has no varsel)
- Order-of-magnitude faster — usable on demand without long waits
- Honest if the dialog labels it correctly: "this test holds the variable selection fixed"
- Full-pipeline mode can be added later as a checkbox in the dialog with a "this will take a while" warning

The dialog will explicitly say "model-conditional permutation test (variable selection held fixed)" so users know what they're getting.

### 3.2 Default N

**N = 100.** Matches SIMCA, matches Westad & Marini's recommendation. Smallest p-value distinguishable: 1/101 ≈ 0.0099. For typical chemometrics regimes that's enough — if a row's null is very tight, the user already knows it's overfit from the gap filter. Optional N=500 / N=1000 for tail precision; surfaces in dialog as a slider.

### 3.3 Pipeline-factory reconstruction from a result row

dasp's existing `_rebuild_model_from_row()` at `search.py:333` rebuilds the model object only. For Phase 2 we additionally need the **preprocessed-and-subsetted X** for that row.

Two inputs we have access to:
- The original `X_train` (in scope when the user opens the leaderboard)
- The row's metadata: `Preprocess`, `Subset`, `n_vars`, `top_vars` columns (stored in the result CSV)

**Plan:** factor out a helper `build_pipeline_input_from_row(row, X_train, y_train, wavelengths)` that:
1. Applies the row's preprocessing config to `X_train` → `X_pre`
2. Reads the row's selected-variable indices (from `top_vars` column or recomputes from `Subset` config) → `selected_idx`
3. Returns `(X_pre[:, selected_idx], model_factory)` where `model_factory` is a no-arg callable returning a fresh unfitted model with the row's hyperparameters

Then the permutation kernel just does `for _ in range(N): cv_score(model_factory(), X_processed, shuffle(y))`.

**Reuse, don't duplicate:** `compute_validation_metrics_for_top_models` at `search.py:568` already does this preprocessing-and-subsetting. Refactor a helper out of it rather than reimplementing.

### 3.4 GUI dialog

Tk Toplevel modal. Components:
- **Progress bar** + "X / N permutations done" label
- **Cancel button** — must reliably interrupt a mid-permutation refit (cooperative via a `threading.Event` checked between permutations)
- **N selector** — default 100; dropdown 100 / 500 / 1000
- **Run button** kicks off background thread
- **On completion:** matplotlib histogram of null-CV-metric distribution + vertical line at observed metric + p-value text + verdict ("Significantly better than chance at α=0.05" / "Not significantly better than chance")
- **Save Result button** writes the per-permutation null distribution to a sidecar CSV `<result>_permutation.csv`; updates the main result CSV's `permutation_pvalue` column for that row's index

**Threading:** the permutation work runs in a `threading.Thread` daemon. Per the project's `feedback_claude_code_concurrency.md` and the Tk worker-thread hardening lessons in `SESSION_LOG.md` (2026-05-08 batch), all Tk widget updates from the worker must go through `root.after(0, ...)` not direct calls. The progress callback will queue updates this way.

### 3.5 Result-CSV column

`permutation_pvalue` column appears in the result CSV — but populated lazily, only for rows the user has explicitly tested. Default value `nan`.

---

## 4. Files to create / touch

### 4.1 New module: `src/spectral_predict/significance.py` (~200 LOC)

- `compute_permutation_pvalue(model_factory, X, y, cv, metric_fn, n_permutations=100, random_state=42, progress_callback=None, cancel_event=None) -> tuple[float, np.ndarray]`
- Pure function. No GUI dependency. Returns `(p_value, null_distribution)`.
- Calls `model_factory()` per permutation to get a fresh unfitted estimator, then `cross_val_predict` (or `cross_val_score`) on shuffled Y.
- Uses dasp's existing CV utilities to ensure same CV strategy as the original search.
- Cooperative cancel via `cancel_event.is_set()` check at the top of each iteration.
- Phipson & Smyth 2010 p-value adjustment: `(1 + n_at_or_above) / (1 + n_permutations)`.

### 4.2 Helper: factor out preprocessing-and-subset from `compute_validation_metrics_for_top_models`

In `src/spectral_predict/search.py`. Extract the per-row preprocessing-and-subset reconstruction into a helper `_build_row_input_for_refit(row, X_train, y_train, wavelengths)` returning `(X_processed_subset, model_factory_callable)`. Both `compute_validation_metrics_for_top_models` and the new permutation path call it.

This is a refactor; behavior unchanged.

### 4.3 GUI dialog: `spectral_predict_gui_optimized.py`

- New method `_open_permutation_test_dialog(self, row_index)` opens the modal.
- New right-click context menu item / button on the leaderboard table: "Test model significance…"
- New Toplevel class `PermutationTestDialog` with the components in §3.4.
- Updates `permutation_pvalue` column on the in-memory leaderboard DataFrame and writes the sidecar CSV on user request.

### 4.4 Tests: `tests/test_permutation_significance.py` (~150 LOC)

1. **High-signal regression**: synthetic data where `y = X[:, 5] + small_noise`; PLS pipeline; expect `p < 0.05` with N=100.
2. **No-signal regression**: `y = noise`; expect `p > 0.5` with N=100.
3. **High-signal classification (PLS-DA)**: synthetic with class-discriminating feature; expect `p < 0.05`. **This is the PLS-DA Q1 coverage Phase 1 deferred** — proves Phase 2 closes that gap.
4. **No-signal classification**: random labels; expect `p > 0.5`.
5. **Phipson-Smyth bound**: zero null hits → p = 1/(N+1), not 0. Tests adjustment is applied.
6. **Cancel mid-run**: set the cancel event after 5 permutations; verify the kernel returns early with partial null distribution and an `aborted=True` flag (or raises a documented exception — design choice for review).
7. **Progress callback fires**: mock callback, assert it's called N times.
8. **Reproducibility**: same `random_state` → same null distribution → same p-value.
9. **Integration via `_build_row_input_for_refit`**: end-to-end on a synthetic result row, confirms the helper feeds the kernel correctly.

GUI tests are deferred to manual smoke; the project pattern (per CLAUDE.md and recent T-19 / T-46 work) is to validate GUI work by actually running the dialog rather than via test harness.

---

## 5. Commit shape

Multi-commit on `feat/T16-phase2-permutation`:

1. `refactor(search): factor _build_row_input_for_refit out of compute_validation_metrics_for_top_models` — pure refactor, behavior preserved by existing tests
2. `feat(T-16): permutation significance kernel + tests` — new module + 9 tests
3. `feat(T-16 GUI): permutation test dialog with progress + histogram` — GUI surface
4. `docs(t-16): Phase 2 ship + SESSION_LOG note`

Or one squashed feature commit if the refactor fits cleanly with the new code. Commit shape decision deferred to implementation.

---

## 6. Verification battery

Before commit:
- `py_compile` on all touched files
- `pytest tests/test_permutation_significance.py -v` — 9/9 pass
- Targeted regression on `test_t44_autoscale_wiring`, `test_bayesian_dedup`, `test_cv_pls_clamp`, `test_one_class_varsel_filtering` — confirm refactor doesn't break adjacent paths
- Manual smoke: launch GUI, run a small search on BoneCollagen, right-click a PLS row → "Test model significance…" → confirm dialog opens, runs to completion, histogram looks sensible, p-value updates the leaderboard column

After commit:
- `git diff --stat main..feat/T16-phase2-permutation` — confirm files match plan
- Spot-check on a real PLS-DA row (the Phase 1 gap): permutation produces a meaningful p-value

---

## 7. Out of scope (future tickets)

- **Phase 2b — full-pipeline permutation** (refits varsel + preprocessing per permutation): can be added as a checkbox in the dialog. Costly. Estimate ~half day source if Phase 2a's kernel is generic enough.
- **Phase 3 — Q2 paired model-vs-model tests** (paired bootstrap CI on ΔRMSEP, paired permutation, McNemar): separate plan, separate branch. The Phase 2 kernel may be reusable for the bootstrap variant.
- **Persistence of `permutation_pvalue`** across sessions: needs a workflow decision (always recompute on result-load? cache to a sidecar? rely on user re-running?). Defer until users complain.
- **Multi-row batch permutation** ("test top 10 rows in one batch"): would dramatically speed up the use case where a user wants Q1 on the whole leaderboard, but adds GUI complexity. Defer until users ask.

---

## 8. Open questions for the reviewer

1. **Cancel semantics:** kernel returns `(p_value, null_dist, aborted)` with partial data when cancelled, OR raises a documented exception? My read: return-with-flag is friendlier for the GUI (can show partial histogram), but the partial p-value is not statistically valid. Verdict request from reviewer.

2. **Variable-selection refits:** the plan ships model-conditional only. Is that enough for the user's typical use case, or is full-pipeline permutation actually needed at Phase 2a? Specifically for varsel-using rows (CARS, GA-PLS, etc.) — model-conditional is somewhat optimistic because the variables were chosen under the original Y. Reviewer judgment requested.

3. **CV strategy reconstruction:** the row's CV strategy (kfold / repeated / GroupKFold) lives in the search-time config but may not be in the row itself. Confirm that the leaderboard or a sidecar metadata blob preserves enough info to reconstruct the same CV split for permutation, OR document that Phase 2 always uses default 5-fold KFold and may give slightly different observed-metric than what's in the row.

4. **`permutation_pvalue` column consistency:** when a row is tested and gets a p-value, do we also update the in-memory leaderboard, the GUI display, AND the on-disk result CSV? Or just the dialog? Persistence patterns in dasp are inconsistent — review how the existing validation-metrics flow handles this for guidance.

5. **GUI thread safety:** the `feedback` memory pin about Tk worker threads + the deferred bare-Tk-AST-sweep ticket suggest there's a known landmine. Want a sister-site grep to catch any worker→Tk-direct calls before shipping?

6. **Refactor safety of `_build_row_input_for_refit`:** Phase 1's `cv_anova_pvalue` recently landed; does the proposed refactor of `compute_validation_metrics_for_top_models` interact badly with anything?

7. **Anything else missing:** Phase 1 had a real BLOCKER (Bayesian-path bypass) the reviewer caught. Looking for the equivalent here — what's the dispatcher-level concern I'm not seeing?
