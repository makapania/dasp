# Implementation Plan: VIP-CARS Variable Selection

**Date:** 2026-07-07
**Status:** Plan — not yet implemented
**Implements:** `docs/METHOD_DESIGN_vip_cars.md` (binding design note, 2026-07-07)
**Scope:** New, separately-named variable-selection method that drives the existing CARS
adaptive-reweighted-sampling (ARS) loop with PLS **VIP scores** instead of `|PLS coef|`.

---

## 0. Binding constraints (restated from the design note — non-negotiable)

1. **Additive only.** Canonical CARS (PLS1 `|coef|`, Li 2009) and CARS-Tree must be
   **byte-for-byte unchanged**. VIP-CARS is a new selectable method, never a silent
   modification. Guard tests proving unchanged outputs of the existing methods are part
   of the deliverable and run from Phase 0 onward.
2. Eligible on **PLS1 (regression), PLS-DA (classification), and PLS2 (multi-target
   regression)** — presented everywhere as dasp's own named method, never as "CARS".
3. VIP comes from the **already-fitted** PLS inside the loop, routed through the existing
   `get_feature_importances()` / `compute_vip()` machinery (`src/spectral_predict/models.py:1790`,
   `:1717`) where practical; the PLS2 case reuses the multi-Y `cars_scaled_coef` aggregation
   (`multi_y.py:446` on the T-17 worktree).
4. **A/B validation including wall-clock** before any default/recommendation claim
   (project rule: "neutral must include wall-clock"). VIP-CARS should be ~flat wall-clock
   because it reuses the PLS fit; a tie-and-slower result is a regression.

---

## 1. SPEC / Definition of Done  ← merge is gated on every box below

The user's stated gate: *merge only if the feature set is complete and working to spec.*
"Complete and working to spec" means ALL of the following, each independently checkable:

### S1 — Functional behavior (core)
- [ ] `cars_selection()` (`src/spectral_predict/variable_selection.py:1224`) accepts a new
      keyword `importance_signal: str = "coef"` with values `"coef"` (default) and `"vip"`.
      With `"vip"`, the PLS-path weight write-back at `:1531` uses VIP scores from the
      **same** `pls.fit(X_subset, y)` at `:1526` — zero additional model fits per iteration.
- [ ] `importance_signal="vip"` combined with a tree `model_type` raises `ValueError`
      (VIP is undefined for trees; CARS-Tree already covers them).
- [ ] VIP scores are non-negative by construction and are written back un-negated;
      the degenerate all-zero VIP case (SSY_total <= 0 in `compute_vip`, `models.py:1778`)
      is floored at `1e-6` before write-back so `rng.choice(p=...)` never receives an
      all-zero probability vector (same rationale as the tree floor at `:1504`).
- [ ] RMSECV selection of the winning iteration (`:1523`, `:1556`) is **unchanged** —
      RMSECV still comes from the same PLS CV. (VIP is the PLS fit, so there is no
      select-by-X/score-by-PLS coupling mismatch — spell this out in the docstring.)
- [ ] Determinism: two calls with identical inputs + `random_state` return identical arrays.

### S2 — All three PLS paths work
- [ ] **PLS1** regression: `cars_selection(X, y, importance_signal="vip")` returns a valid
      importance vector (shape `(n_features,)`, non-negative, non-zero winners) on real
      example data.
- [ ] **PLS-DA** classification: same call with encoded integer labels
      (the existing CARS classification convention — `search.py:3670` passes
      `task_type`, y is label-encoded upstream, and the PLS path fits `PLSRegression`
      on the encoded y exactly as canonical CARS does today). VIP on that fit is
      well-defined; `compute_vip` needs no DA-specific change for this path (it reads
      `x_weights_/x_scores_/y_loadings_` of the fitted `PLSRegression`). Verified by test.
- [ ] **PLS2** multi-target (on the T-17 branch — see Phase 3): multi-Y VIP computed from
      the single joint PLS2 fit on column-scaled Y, per-target VIP columns aggregated with
      `aggregate_importance(rule="cars_scaled_coef")` — the same rule the T-17 `|coef|`
      path uses (`variable_selection.py:1481` in worktree `C:/Users/mspon/dasp-t17-uve-cars`).
- [ ] One-class task filtering: `vip-cars` mirrors `cars` exactly (CARS is currently
      permitted in one-class mode; the T-04 ban covers only the UVE + iPLS families —
      `tests/test_one_class_varsel_filtering.py:74`). Test added.

### S3 — Additive-only guards (canonical methods untouched)
- [ ] Golden-fixture byte-identity tests pass: `cars` (PLS path, regression), `cars`
      (classification), `cars-tree` (hybrid), `cars-aware` outputs on fixed-seed data are
      `np.array_equal` to `.npy` fixtures generated from pre-change `main` HEAD (Phase 0).
- [ ] Default-signature guard: calling `cars_selection` with only the pre-existing
      positional/keyword arguments produces the canonical `|coef|` path (i.e. the new
      parameter defaults to `"coef"` and every existing call site is untouched).
- [ ] `git diff` on the dispatch blocks shows canonical branches only *extended by new
      elif arms*, never edited: `search.py:3638-3671`, `:6799-6802`, `:7429-7436`.

### S4 — Search/grid integration (every dispatch surface `cars`/`cars-tree` has today)
- [ ] `run_search` registry `search.py:1752-1772`: `"vip-cars"` added.
- [ ] Standard grid dispatch `search.py:3638`: `vip-cars` branch calls `cars_selection`
      with `importance_signal="vip"`, `model_type=None`.
- [ ] Second (deep-dive/refinement) dispatch: registry `search.py:6295-6296` and branch
      `search.py:6799-6802` extended identically.
- [ ] Varsel cache key `search.py:2909-2946`: `vip-cars` is model-independent (falls into
      the else-branch at `:2944-2946`); distinct method string ⇒ distinct cache entry. Test.
- [ ] Multiclass SIMCA path: `_MULTICLASS_IMPORTANCE_METHODS` `search.py:7268-7283` gains
      `"vip_cars"` (underscore convention, mirroring `cars_tree`); dispatch at `:7429`
      extended; category note at `:2919`/`:2937` untouched (vip-cars is not
      model-category-dependent).
- [ ] Subset/downstream method list `search.py:3960-3965` (methods whose importances feed
      subset construction) includes `"vip-cars"`.
- [ ] Results leaderboard rows show `vip-cars` in the VarSel column and survive the GUI's
      method-string round-trips (`spectral_predict_gui_optimized.py:30195`, `:30601` treat
      the stamp generically — verify by integration test, no code change expected).

### S5 — GUI
- [ ] New checkbox "VIP-CARS (PLS VIP signal)" in the varsel frame next to CARS
      (`spectral_predict_gui_optimized.py:12338-12346`), caption:
      "dasp method — CARS loop driven by PLS VIP; for PLS/PLS-DA models".
- [ ] `tk.BooleanVar` `self.varsel_vip_cars` initialized near `:3915-3916`, default False.
- [ ] Collection block `:28067-28070`: appends `'vip-cars'`.
- [ ] Multiclass tab group list `:3173-3180`: `("vip_cars", "VIP-CARS")` added to the
      "Discrimination-based" group.
- [ ] Settings persistence: `"varsel_vip_cars"` added to `src/spectral_predict/run_gui_settings.py:133-138`
      so save/restore round-trips (T-43 resume too — see `tests/test_t43_resume_auto_restore.py:100`).
- [ ] Method guide text (speed table at `:12272`) gains a VIP-CARS row ("Slow | PLS,
      PLS-DA | dasp method").
- [ ] One-class enable/disable checkbox lists (`:17125`, `:17202`): VIP-CARS mirrors CARS
      (i.e. NOT added to the disabled lists).

### S6 — Exports / repro-script codegen / model IO
- [ ] `src/spectral_predict/templates/variable_selection.py`: new `VIP_CARS_TEMPLATE`
      (self-contained function, CARS loop + inline VIP computation, no dasp imports —
      same style as `CARS_TEMPLATE` at `:292`); registry at `:411-418` gains both
      `'vip-cars'` and `'vip_cars'` keys (the lookup at `:420` lowercases but does not
      normalize separators).
- [ ] `src/spectral_predict/templates/__init__.py:19-51`: `VIP_CARS_TEMPLATE` exported.
- [ ] `code_generator.py:743-757` renders the template when
      `config['variable_selection_method']` is `vip-cars`; generated script executes
      standalone and reproduces a top-N selection (test).
- [ ] `model_io.py`: no method-string coupling exists (saved models persist selected
      wavelength indices, not method names) — verified by a save/load round-trip test on
      a vip-cars model; **no code change expected**, but the verification is part of DoD.
- [ ] Docstrings/citations: VIP-CARS docstring cites Li 2009 (ARS loop), Wold 2001 +
      Mehmood 2012 (VIP), and Zheng 2012 SCARS as the swap-the-signal precedent, and
      states explicitly: "dasp method; not the published CARS of Li et al. 2009."

### S7 — Tests green
- [ ] All new unit/integration tests pass (Phase list §4).
- [ ] Failure-set diff vs `origin/main` is empty (per project rule: main is red; the PR
      must add zero NEW failures — `gh pr view --json statusCheckRollup` + local diff of
      failing-test sets).

### S8 — A/B validation done and written up
- [ ] Harness run (Phase 7): canonical CARS vs VIP-CARS, PLS/PLS-DA, ≥2 real datasets,
      ≥3 seeds, reporting RMSEP/R²/joint-Q² (regression), macro-F1 (classification),
      selected-variable count, **and wall-clock per arm**.
- [ ] Pass criterion: quality parity-or-better AND wall-clock within ±10% of canonical
      CARS. Result recorded in this doc's §8 + `docs/SESSION_LOG.md`.
- [ ] If VIP-CARS loses on quality it still merges as an opt-in method (it is additive and
      honestly labeled), but the GUI caption/docs must not claim superiority — only "dasp
      method, alternative signal". If it is materially slower, fix or do not merge.

### S9 — Docs / naming
- [ ] `docs/PROJECT_STATUS.md` + `docs/SESSION_LOG.md` updated (session protocol).
- [ ] CLAUDE.md "Key Features" varsel list mentions VIP-CARS.
- [ ] Naming used consistently everywhere per §2 (no surface ever labels it "CARS").

---

## 2. Naming decision (recommended)

**Method name: `VIP-CARS`** ("VIP-driven CARS"). Rationale: leads with the signal (like
SCARS leads with Stability), reads naturally next to CARS and CARS-Tree in the GUI, and
cannot be mistaken for canonical CARS.

Exact strings — these deliberately mirror the CARS-Tree convention, which uses a hyphen
token in `run_search` but an underscore token in the multiclass registry:

| Surface | String | Mirrors |
|---|---|---|
| `run_search` / deep-dive registry + dispatch (`search.py:1762`, `:3638`, `:6295`, `:6799`) | `'vip-cars'` | `'cars-tree'` |
| Multiclass registry + dispatch (`search.py:7273-7283`, `:7429`) | `'vip_cars'` | `'cars_tree'` |
| GUI checkbox label | `VIP-CARS (PLS VIP signal)` | `CARS-Tree (Hybrid Importance)` |
| GUI multiclass tuple (`gui:3173`) | `("vip_cars", "VIP-CARS")` | `("cars", "CARS")` |
| GUI BooleanVar / settings key | `varsel_vip_cars` | `varsel_cars_tree` |
| `cars_selection` parameter | `importance_signal="vip"` (default `"coef"`) | new |
| Codegen template keys (`templates/variable_selection.py:411`) | `'vip-cars'` AND `'vip_cars'` | `'cars'` |
| Results VarSel column value | `vip-cars` (flows from run_search token) | `cars-tree` |
| Docstring/report citation line | "VIP-CARS (dasp method): CARS ARS loop (Li et al. 2009) driven by PLS VIP (Wold 2001; Mehmood 2012); signal-swap precedent: SCARS (Zheng et al. 2012)" | — |

Rejected: `cars-vip` (buries the differentiator), `vipcars` (unreadable), reusing plain
`cars` with a flag in the UI (violates constraint 1's "never a silent modification").

---

## 3. Architecture of the change (core signal path)

`cars_selection` PLS branch today (`variable_selection.py:1509-1531`):

```
pls.fit(X_subset, y)          # :1526  — full-subset fit
coef = pls.coef_.ravel()      # :1530
weights[selected_vars] = np.abs(coef)   # :1531  — THE signal write-back
```

Change (additive branch, canonical line untouched):

```
pls.fit(X_subset, y)                                  # unchanged, same fit
if importance_signal == "vip":
    from .models import get_feature_importances
    imp = get_feature_importances(pls, "PLS", X_subset, y)   # -> compute_vip
    weights[selected_vars] = np.maximum(imp, 1e-6)
else:
    coef = pls.coef_.ravel()
    weights[selected_vars] = np.abs(coef)             # canonical, byte-identical
```

Notes:
- `get_feature_importances(model, "PLS", ...)` (`models.py:1790`) unwraps Pipelines and
  calls `compute_vip` (`models.py:1717`) — the bare `PLSRegression` here passes through
  untouched. This honors constraint 3 (route through the dispatcher, don't reimplement).
- `compute_vip` is O(n_features x n_components) on already-computed matrices — negligible
  vs the CV fits, hence the ~flat wall-clock expectation.
- Validation up front in `cars_selection`: `importance_signal not in ("coef", "vip")` →
  `ValueError`; `importance_signal == "vip" and use_tree_model` → `ValueError`.
- Normalization at `:1533-1534` (`weights /= weights.sum()`) applies to both signals
  unchanged.
- The `1e-6` floor for VIP guards two edge cases: `compute_vip`'s all-zeros return when
  `ssy_total <= 0` (`models.py:1778-1779`) and near-zero VIPs on noise channels — same
  unrecoverability argument as the tree floor comment at `:1501-1504`.

---

## 4. Phased task breakdown

Ordered so the additive-only guard exists **before** any production code changes.

### Phase 0 — Golden fixtures + guard tests (BEFORE touching production code)
**Effort: ~0.5 day.** Files:
- NEW `tests/fixtures/cars_golden/` — `.npy` fixtures generated at pre-change `main` HEAD:
  `cars_pls_regression.npy`, `cars_pls_classification.npy`, `cars_tree_hybrid.npy`,
  `cars_aware.npy` (fixed synthetic data: seeded `RandomState(42)`, 60x120 X, plus a
  deterministic slice of `example/` data; `n_iterations=50`, defaults otherwise).
- NEW `tests/test_vip_cars.py` — `TestCanonicalCarsUnchanged`: re-runs the same calls and
  asserts `np.array_equal` against the fixtures; also asserts
  `inspect.signature(cars_selection)` still has `importance_signal='coef'` default once
  Phase 1 lands (parametrize to tolerate the parameter not existing yet, so the guard is
  runnable at Phase 0).
- NEW small generator script `tools/gen_cars_golden_fixtures.py` (committed, so fixtures
  are regenerable and auditable).

**Acceptance:** guard tests pass on unmodified main. Commit before Phase 1 (project rule:
commit before yielding control).

### Phase 1 — Core signal path (PLS1)
**Effort: ~0.5–1 day.** Files:
- `src/spectral_predict/variable_selection.py:1224` — add `importance_signal="coef"`
  keyword; docstring (signal semantics, dasp-method statement, citations); validation;
  the elif branch of §3 at `:1526-1531`.
**Acceptance:** Phase 0 guards still pass byte-identically; new unit tests: vip path
returns valid vector, differs from coef path on collinear synthetic data, deterministic,
tree+vip raises, all-zero-VIP floor works (mock `compute_vip` → zeros).

### Phase 2 — PLS-DA (classification) path
**Effort: ~0.5 day.** Files:
- `src/spectral_predict/variable_selection.py` — no structural change expected: the PLS
  branch already fits `PLSRegression` on encoded labels for classification (canonical
  CARS behavior); VIP rides the same fit. Confirm and document in the docstring that
  the DA case is PLS1-on-encoded-y VIP (binary) / ordinal-encoded (multiclass), exactly
  the y canonical CARS uses today — no new modeling decision introduced.
- `search.py:7429-7436` — extend the `("cars", "cars_tree")` arm's sibling: new
  `if method == "vip_cars":` arm calling `cars_selection(..., importance_signal="vip",
  task_type=task_type)`; add `"vip_cars"` to `_MULTICLASS_IMPORTANCE_METHODS` (`:7268`).
**Acceptance:** unit test on 3-class synthetic labels via `_multiclass_varsel_mask`
returns a valid boolean mask; guard tests still pass.

### Phase 3 — PLS2 (multi-target) — executes in the T-17 worktree
**Effort: ~1 day.** Branch: `feat/T17-multitarget-regression`
(worktree `C:/Users/mspon/dasp-t17-uve-cars`, base commit `bc750d9`). PLS2 machinery
(`_cars_multi_cell` at worktree `variable_selection.py:1393`, `multi_y.py:446`,
`multitarget_grid.py:218/:344`) does **not exist on main**, so this phase lands as a
commit on the T-17 branch (rebase onto main after Phases 1–2 merge, or fold into the
T-17 review). The merge gate treats the feature as complete when both are green.

Files (worktree paths):
- `src/spectral_predict/variable_selection.py:1477-1481` (`_cars_multi_cell` PLS branch) —
  additive: when `importance_signal == "vip"`, compute per-target VIP columns from the
  single joint `pls_full.fit(X_subset, y_scaled)` (Y column-scaled exactly as the coef
  path — scaling stays load-bearing), then
  `feature_weights = aggregate_importance(vip_matrix, rule="cars_scaled_coef")`
  (constraint 3). Per-target VIP_j^(s) uses `q_{s,a}²` in place of the summed
  `Σ_s q_{s,a}²`; implement as a small `compute_vip_multi(pls) -> (n_features, n_targets)`
  helper in `models.py` beside `compute_vip` (its docstring at `models.py:1730-1732`
  already states the multi-Y form — note in the new docstring that l2-aggregating
  per-target VIP columns is the dasp PLS2 rule chosen for consistency with the
  `cars_scaled_coef` coef path, and that the RMSECV criterion — pooled
  `sqrt(mean(1-Q2))`, worktree `:1445-1453` — is unchanged).
- `src/spectral_predict/variable_selection.py` (worktree `cars_selection` multi-Y dispatch
  at `:1684-1693`) — thread `importance_signal` through to `_cars_multi_cell`.
- `src/spectral_predict/multitarget_grid.py:218` — add `"vip-cars"` to the multi-Y
  allow-list; dispatch beside `:344-346` (`cars_selection(X_pp, Y, importance_signal="vip")`).
  Tree-mode `cars-tree` stays in the deferred set (`:418`) — VIP-CARS is PLS-only, so no
  new deferral needed.
**Acceptance:** T-17 branch tests: multi-Y vip-cars returns valid vector on 2-target
synthetic data; differs from target-0-only and target-1-only single-Y runs (mirroring
`tests/test_multitarget_varsel.py:216-267` pattern); T-17's existing single-Y identity
guards (`test_multitarget_varsel.py:31` etc.) still pass; main-repo golden guards pass
after rebase.

### Phase 4 — GUI
**Effort: ~0.5 day.** Files (all `spectral_predict_gui_optimized.py` unless noted):
- `:3915-3916` BooleanVar; `:12338-12346` checkbox row + caption (insert after CARS-Tree
  row; renumber grid rows below or append at the family's end); `:28067-28070` collection
  (`'vip-cars'`); `:3173-3180` multiclass tuple; `:12272` speed-table row;
  `src/spectral_predict/run_gui_settings.py:133-138` persistence key.
- Watch the CRLF/LF drift landmine on this file (memory: check `git diff --stat` after
  each Edit; binary-mode rewrite fallback).
**Acceptance:** `py_compile` clean; targeted GUI tests (`tests/gui/test_multiclass_gui_parity.py:209`
extended so `vip_cars` appears in offered keys); settings save/load round-trip test;
manual smoke: checkbox visible, run starts, leaderboard row shows `vip-cars`.

### Phase 5 — Exports / repro codegen
**Effort: ~0.5 day.** Files:
- `src/spectral_predict/templates/variable_selection.py` — `VIP_CARS_TEMPLATE`
  (self-contained: ARS loop + inline VIP formula; header comment with the three-citation
  line and "dasp method" statement); registry `:411-418` gains `'vip-cars'` + `'vip_cars'`.
- `src/spectral_predict/templates/__init__.py:19-51` — export.
- `code_generator.py` — no change expected (`:745-757` is generic); verify only.
**Acceptance:** generated script for a vip-cars result compiles (`py_compile`) and, when
executed against `example/` data, selects a plausible subset; export/codegen test added.
Save/load round-trip via `model_io` on a vip-cars model (no code change expected — S6).

### Phase 6 — Integration wiring + full test pass
**Effort: ~1 day.** Files:
- `search.py:1762` registry; `:3638` dispatch arm (`elif varsel_method == "vip-cars":
  ... cars_selection(..., model_type=None, use_hybrid_importance=False,
  importance_signal="vip", task_type=task_type)`); `:6295` + `:6799` second dispatch;
  `:3960-3965` subset list; docstring at `:5929`.
- Cache-correctness: extend `tests/test_varsel_caching_correctness.py:62-75` pattern —
  `vip-cars` caches deterministically AND under a key distinct from `cars`.
- End-to-end smoke (project rule: e2e after search-machinery change): regression +
  classification through `run_search` with `variable_selection_methods=['cars','vip-cars']`
  on `example/` data; both rows appear, `cars` row metrics identical to a cars-only run.
**Acceptance:** all new tests + Phase 0 guards green; failing-set diff vs `origin/main`
empty. **Non-goal explicitly:** `unified_bayesian.VAR_METHODS` (`unified_bayesian.py:374`)
is NOT extended in v1 (see §7).

### Phase 7 — A/B validation (merge-gate evidence)
**Effort: ~0.5 day build + overnight runs.** Files:
- NEW `tools/ab_vip_cars.py` (pattern: `tools/_tpe_fix_ab_arm_*.csv` /
  `_preprocessing_refactor_ab_*.json` harnesses): arms = {`cars`, `vip-cars`}, PLS (+
  PLS-DA dataset), ≥3 seeds, fixed preprocessing set; outputs JSON with per-arm RMSEP/R²
  (or macro-F1), joint-Q² where multi-target (T-17 branch run), n selected variables,
  wall-clock per arm.
- Datasets: `example/` shipped data + the user's real NIR sets used in prior A/Bs
  (ask user to point at the same ones used for the 2026-05-07 TPE A/B).
**Acceptance / pass criteria:** S8. Record verdict in this file §8 and SESSION_LOG.

**Total estimate: ~4.5–5 dev-days** (Phases 0–2 and 4–6 on main-repo branch
`feat/vip-cars`; Phase 3 on the T-17 branch; Phase 7 spans both).

Review cadence (per project convention): Codex review at end of Phase 2 (core), end of
Phase 6 (integration), cross-family (GLM/Kimi via opencode-call) sister-site sweep after
Phase 6, pr-review-toolkit + merge-gate checklist before merge.

---

## 5. Test plan (consolidated)

| # | Test | Location | Phase |
|---|---|---|---|
| T1 | Golden byte-identity: `cars` PLS regression | `tests/test_vip_cars.py` + fixtures | 0 |
| T2 | Golden byte-identity: `cars` classification | same | 0 |
| T3 | Golden byte-identity: `cars-tree`, `cars-aware` | same | 0 |
| T4 | Default-arg guard: no-new-kwargs call == canonical path | same | 1 |
| T5 | VIP path: valid vector, differs from coef on collinear data, deterministic | same | 1 |
| T6 | `vip` + tree model_type raises ValueError; bad signal string raises | same | 1 |
| T7 | All-zero-VIP floor (mocked `compute_vip` → zeros): no crash, sampling proceeds | same | 1 |
| T8 | PLS-DA: `_multiclass_varsel_mask(..., "vip_cars", ...)` returns valid mask, 3 classes | same | 2 |
| T9 | PLS2: multi-Y vip-cars valid + not identical to either single-target run | T-17 `tests/test_multitarget_varsel.py` | 3 |
| T10 | PLS2: T-17 single-Y identity guards still green | existing T-17 tests | 3 |
| T11 | Cache: deterministic + key distinct from `cars` | extend `tests/test_varsel_caching_correctness.py` | 6 |
| T12 | Integration: `run_search` with `['cars','vip-cars']` → both rows; cars row invariant | `tests/test_vip_cars.py` | 6 |
| T13 | Codegen: template renders for both spellings, generated script `py_compile`s + runs | `tests/test_vip_cars.py` | 5 |
| T14 | model_io round-trip on vip-cars model | extend `tests/test_model_io_comprehensive.py` | 5 |
| T15 | GUI: multiclass parity offers `vip_cars`; settings persistence round-trip | `tests/gui/test_multiclass_gui_parity.py`, settings test | 4 |
| T16 | One-class filtering: `vip-cars` treated exactly like `cars` | extend `tests/test_one_class_varsel_filtering.py` | 6 |
| T17 | A/B harness (quality + wall-clock) — evidence, not pytest | `tools/ab_vip_cars.py` | 7 |

Run targeted tests per phase (project rule: no full-suite runs for small changes); full
failing-set diff vs `origin/main` once, at Phase 6 end.

---

## 6. Merge-gate checklist (go/no-go)

- [ ] **G1** All S3 guard tests green (canonical CARS/CARS-Tree byte-identical). *Hard gate.*
- [ ] **G2** S1/S2 functional: PLS1 + PLS-DA green on main-repo branch; PLS2 green on T-17
      branch (or post-T-17-merge rebase). *Hard gate — feature set incomplete without all three.*
- [ ] **G3** S4 all six dispatch surfaces wired (`run_search` registry, standard dispatch,
      deep-dive registry+dispatch, cache key verified, multiclass registry+dispatch, subset list).
- [ ] **G4** S5 GUI complete incl. settings persistence; manual smoke screenshot attached to PR.
- [ ] **G5** S6 exports/codegen/model-io verified (T13, T14).
- [ ] **G6** S7 tests green; zero new failures vs `origin/main`; CI checked via
      `gh pr view --json statusCheckRollup` before merge.
- [ ] **G7** S8 A/B done: quality parity-or-better, wall-clock within ±10%. Slower-and-tied = no-go.
- [ ] **G8** S9 naming audit: grep the diff for the strings `CARS`/`cars` — every new
      user-visible surface says VIP-CARS / vip-cars / vip_cars; docstring carries the
      dasp-method + citations line.
- [ ] **G9** Reviews done per cadence (§4); docs updated (PROJECT_STATUS, SESSION_LOG, CLAUDE.md).

---

## 7. Risks & non-goals

**Risks**
- *Touching canonical CARS by accident.* Mitigated by Phase-0 fixtures committed before any
  production edit, and by writing the change as a new elif arm rather than editing `:1531`.
- *PLS-DA VIP edge cases.* Multiclass ordinal-encoded y is a known crudeness — but it is
  the exact y canonical CARS already uses (`search.py:7429-7435`), so VIP-CARS introduces
  no NEW modeling decision. Documented, not "fixed", in v1 (a one-hot-Y PLS2-DA VIP
  variant is a possible v2, gated on user confirmation — methodology change).
- *Degenerate VIP (all-zero / SSY≤0).* Floored at 1e-6 (T7). Distinct from the tree
  sparsity case but same failure mode in `rng.choice`.
- *RMSECV coupling.* None for VIP (VIP *is* the PLS fit — design note §4.1); state it in
  the docstring so future signal swaps re-check it.
- *Wall-clock.* `compute_vip` is matrix-algebra on fitted attributes; if the A/B shows
  >10% regression, suspect the dispatcher import/unwrap overhead in the hot loop — hoist
  the import, or call `compute_vip` directly (still "through the existing machinery").
- *T-17 dependency for PLS2.* T-17 is unreviewed WIP. If T-17 stalls, Phases 0–2, 4–6 are
  independently mergeable; the merge gate then explicitly re-scopes with the user (G2
  says all three paths — only the user can waive PLS2).
- *GUI file line-ending drift* (memory landmine): check `git diff --stat` after every Edit
  to `spectral_predict_gui_optimized.py`.
- *Two token spellings* (`vip-cars` / `vip_cars`) mirror the existing cars-tree wart. Do
  NOT unify the existing cars-tree spellings in this PR (scope creep + guard risk).

**Non-goals (v1)**
- No change to canonical CARS, CARS-Tree, `cars-aware`, or any UVE hybrid.
- No `uve_vip_cars` hybrid; no VIP signal for the tree path.
- No Bayesian search-space extension (`unified_bayesian.py:374 VAR_METHODS`) — opt-in
  grid/multiclass surfaces only; revisit after A/B evidence.
- No NSPFCE/calibration-transfer selector (`calibration_transfer.py:1028`), no NSGA-II
  guidance change (`nsga2_search.py`), no preprocessing-discovery change (it already has
  a separate `'vip'` importance option — unrelated feature, `preprocessing_discovery.py:118`).
- No multi-target *classification* (rejected in the design note §3).
- No novelty/publication claim until Scopus/WoS due diligence (design note §7.3).

---

## 8. A/B results (to be filled at Phase 7)

| Dataset | Task | Arm | Metric | Wall-clock | Verdict |
|---|---|---|---|---|---|
| _tbd_ | | | | | |
