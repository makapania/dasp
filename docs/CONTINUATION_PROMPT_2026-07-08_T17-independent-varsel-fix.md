# T-17 fix ticket — INDEPENDENT-mode variable selection is secretly COUPLED

**Created:** 2026-07-07 (for pickup 2026-07-08)
**Status:** T-17 branch (`feat/T17-multitarget-regression`) is **NOT merge-ready** — blocked on this bug.
**Priority:** BLOCKER for T-17 merge. This is the root cause of "multi-target does much worse per target; why would anyone use it" (user, 2026-07-07).

---

## The bug (confirmed empirically + in source)

In **INDEPENDENT** coupling mode, multi-target **variable selection is still coupled**. CARS / UVE / importance / iPLS / SPA all run **once on the stacked multi-column `Y`**, producing ONE variable subset that is then shared by **every** target's (separately-fit) model. So a strong target is forced to use wavelengths chosen to also satisfy the other target(s).

This violates the INDEPENDENT honesty contract (`INDEPENDENT_PRECISE_NOTE` in `multitarget_search.py`): the *models* are per-target-separate, but the *variable subset* is not — it's jointly selected.

### Root cause (worktree `src/spectral_predict/multitarget_grid.py`)
- **`_model_independent_importances` (~line 305–402):** every varsel method receives the full 2-D `Y`. Confirmed:
  - `cars_selection(X_pp, Y)` at **:360–362**
  - `uve_selection(X_pp, Y, ...)` at **:352–359**
  - `ipls_selection(X_pp, Y)` at **:349–351**; fipls_spa at :333–346; uve_cars / uve_spa / uve_cars_spa families likewise on the joint `Y`.
- **`build_multitarget_varsel_subsets` (~:471):** called with the full `Y_arr` (from the grid at ~:777–786); returns ONE subset list per (preprocess, method), shared across targets.
- **Structural (the hard part):** a search "cell" = `(preprocess, varsel-subset, model, hp, mode)` (built ~:745–752; evaluated by `_evaluate_multitarget_cell` ~:832–841, which scores **all** targets on the single `X_sub`). There is **no seam for a per-target subset** — even under `coupling_mode="independent"`, N and ADF cannot each get their own subset.

## Evidence (real backend, on the user's data)

Data: `Norway Single leaf dry 2023 data(in).csv` (129×2151 spectra, targets N/ADF/NDF). Config: SNV+SG1+SG2+deriv_snv, PLS, kfold 5, INDEPENDENT. Matched preprocessing (`deriv_snv d2`), same CV, same top-100 cardinality:

| Variables selected on | N CV R² |
|---|---|
| **N alone** (single-Y path does this) | **0.859** |
| **joint (N,ADF)** (multi-target does this) | **0.712** ← byte-matches the real `run_multitarget_grid_search` best deriv_snv d2 row (0.7118) |

- Preprocessing is **not** the cause (both paths reach deriv_snv d2).
- Joint-mean ranking is **not** the cause (the best-joint row *is* N's best row).
- The **shared variable subset is the entire ~0.147 R² gap.** n_components is a symptom (joint-CARS forced to LVs=10 vs N-only LVs=5–6).
- Single-Y N (real `run_search`, CARS-on-N) = **0.851** — honest, because single-Y selects on the lone `y`.

**Note:** the true GUI default varsel is **importance-only** (`varsel_cars` default False, `varsel_importance` default True, GUI ~:3847/:3856) — but importance is ALSO computed on the joint `Y`, so the default multi-target experience is affected too.

## The fix

Under **INDEPENDENT** coupling, run variable selection **per target** (CARS-on-N → N's model; CARS-on-ADF → ADF's model), not once on the stacked `Y`. Under **JOINT** coupling, keep joint varsel (correct there — deliberately coupled).

Expected result: multi-target INDEPENDENT N per-target R² ≈ single-Y N (~0.85); ADF unaffected; JOINT unchanged.

### Threading required (this is the real work — medium/large)
A cell currently carries ONE subset for all targets. Per-target subsets means threading **per-target variable indices** through:
1. **Subset build** — under independent mode, compute a subset per target (loop the varsel over each `Y[:, t]`), or a `{target: indices}` map.
2. **Cell model** — INDEPENDENT already fits per-target (MultiOutputRegressor / per-column); each per-target estimator must be fit on its own `X[:, idx_t]`. Today the cell fits all targets on one `X_sub`.
3. **Evaluation** — `_evaluate_multitarget_cell` must score each target on its own subset.
4. **Refit / save** — `refit_multitarget_final` + `RefitMultiTargetModel` must carry per-target `variable_indices` and predict each target on its own columns; `model_io` metadata + `predict_with_model` must handle per-target subsets.
5. **GUI display** — the leaderboard shows one `#Vars`/`Varsel` per row; decide how to present per-target subsets (e.g. per-target nvars in the per-target columns, or a combined tag). Detail dialog + export likewise.

### Design decisions to settle first
- Does INDEPENDENT-per-target-varsel become a NEW cell shape, or do we expand each varsel cell into a per-target-subset variant? (Affects grid size + leaderboard semantics.)
- How to keep JOINT varsel unchanged while INDEPENDENT goes per-target (branch on `strategy.mode`).
- Leaderboard/`.dasp`/export representation of per-target subsets.

## Acceptance criteria
- On the Norway data, INDEPENDENT multi-target N per-target R² recovers to ~0.85 (≈ single-Y), ADF unchanged; verified via the real `run_multitarget_grid_search`.
- JOINT-mode results byte-identical to today.
- Single-Y `run_search` path byte-identical (gold-snapshot guard).
- `refit_multitarget_final` save→reload→predict still exact for per-target-subset models.
- Regression tests: a multi-Y test asserting that under INDEPENDENT mode, varsel is per-target (e.g. two targets driven by disjoint synthetic bands each recover their own band).

## Reproduction recipe (recreate tomorrow)
- Real backend on worktree src (`sys.path.insert(0, ".../dasp-t17-uve-cars/src")`), venv `.venv312`.
- Load the CSV; X = integer-named cols (350–2500); targets N, ADF (complete-case).
- Decompose: for `deriv_snv d2` preprocessing, PLS LV sweep 1–10, compare `cars_selection(X, y_N)` (top-100) vs `cars_selection(X, Y_joint)` (top-100) → N CV R². The ~0.86 vs ~0.71 split is the bug.
- Today's scratch scripts (may not persist): `scratchpad/norway_diag.py`, `norway_pp.py`, `norway_varsel.py`, plus the agent's `repro_multitarget.py` / `decompose.py`.

## Review cadence for the fix (per project convention)
Codex per phase + a cross-family pass (Kimi/GLM) on the varsel-threading diff + pr-review-toolkit + merge-gate vs origin/main + re-run the Norway validation. Chemometrics-gate: confirm per-target selection actually recovers per-target-appropriate bands (the check that was skipped when this WIP first landed).
