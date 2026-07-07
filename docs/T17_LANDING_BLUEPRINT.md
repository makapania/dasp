# T-17 Landing Blueprint — UVE/CARS Multi-Y "Ship, Not Revert"

**Worktree:** `C:/Users/mspon/dasp-t17-uve-cars` — branch `feat/T17-multitarget-regression` — HEAD `bc750d9`.
**Venv (tests):** `C:/Users/mspon/git/dasp/.venv312/Scripts/python.exe`.
**Binding decision:** SHIP UVE/CARS multi-Y. Everywhere the review says "ship OR revert," the answer is SHIP: flip stale reject/skip tests to positive support, wire GUI params end-to-end, update docs. Later feature VIP-CARS depends on this machinery.

**Constraint reminder for executing agents:** READ/py_compile/pytest are fine. `spectral_predict_gui_optimized.py` is 60,544 lines with known CRLF/LF drift — **after every Edit run `git diff --stat` and abort/redo if phantom whitespace hunks appear** (per `feedback_edit_tool_line_endings.md`).

---

## Verified failure baseline (measured at HEAD `bc750d9`)

Ran the venv against the named files. Confirmed:

- `tests/test_variable_selection.py` — **5 fail** (all in `TestMultiYRejectGuards`): `test_uve_rejects_multi_y`, `test_cars_rejects_multi_y`, `test_uve_spa_rejects_multi_y`, `test_uve_cars_rejects_multi_y`, `test_uve_cars_spa_rejects_multi_y`. These now raise nothing because the functions support 2-D Y — the `pytest.raises(NotImplementedError)` fails.
- `tests/test_multitarget_varsel.py` — **4 fail**: `test_ipls_selection_rejects_2d_y`, `test_vcpa_iriv_rejects_2d_y`, `test_classify_varsel_method` (asserts uve/cars/ipls/fipls_cars → "skip"), `test_build_varsel_subsets_full_plus_interval_and_skips` (asserts `skipped == {"uve","cars"}`).
- `tests/test_multitarget_integration.py` — **1 fail**: `test_multitarget_grid_end_to_end_real_data` (asserts `"JOINT" in modes` but default coupling is now Independent-only → only INDEPENDENT present).
- **Also found (NOT in Codex's list but same file group):** `tests/test_multitarget_grid.py` — **2 fail**: `test_grid_search_end_to_end_ranks_and_skips` (asserts `"uve" in out.skipped`), `test_grid_search_apply_uve_prefilter_surfaces_skip` (asserts `"apply_uve_prefilter" in out.skipped`). These must be reconciled too or the gate stays red.

Total: **12 failures** across those 4 files. `tests/test_io_*` all pass at HEAD — the "1 failed io" Codex saw did not reproduce; noted as not-verified/likely-transient.

---

## Line-number reconciliation vs Codex (checked against HEAD)

| Codex claim | Verified at HEAD | Status |
|---|---|---|
| `multitarget_grid.py:609` unconditional SPA preflight | `spa_ok = verify_spa_multi_y_safe(X_arr, Y_arr)` at **line 610** | CONFIRMED (off by 1) |
| GUI `uve_cutoff_multiplier`/`uve_n_components` at `:12347` | Entry widgets at **12349/12353**; Tk vars declared **3866/3867** | CONFIRMED (near) |
| `_collect_multitarget_config` at `:15785` | `def` at **15707**; return dict spans 15771–15798; `:15785` is the `variable_selection_methods` line | CONFIRMED (def is at 15707, not 15785) |
| `run_multitarget_grid_search` at `:565` | `def` at **line 565** | CONFIRMED exact |
| prefilter `get_uve_threshold(X_pp, Y)` at `:458` | **line 458** | CONFIRMED exact |
| `_IMPORTANCE_METHODS` at `:214` | **line 214** | CONFIRMED exact |
| `_model_independent_importances` UVE-family calls | lines **305–371** | CONFIRMED |
| `test_variable_selection.py:896` TestMultiYRejectGuards | class at **line 895** | CONFIRMED (off by 1) |
| `test_multitarget_varsel.py:284` classify→skip | `test_classify_varsel_method` at **272**; `test_build_varsel_subsets...` at **289** | CONFIRMED (near) |
| `test_multitarget_integration.py:89` JOINT+INDEPENDENT | assert `"JOINT" in modes` at **line ~99** (test def at 64) | CONFIRMED behavior; line drifted |
| vcpa-iriv leaf `wavelength_selection.py:466-493` | `_prep_varsel_y` at **466**, tree-mode reject at **475**, multi-Y PLS criterion **496** | CONFIRMED |
| router `multitarget_grid.py:415-417` SKIP_WITH_NOTICE | comment **415-416**, set **417-419** (`"cars-tree","uve_cars_tree","vcpa-iriv"`) | CONFIRMED (off by ~2) |
| `ga_lightgbm.py:226-254` multi-Y fitness | `_fitness_function_lgbm` multi-Y branch at **228-258** | CONFIRMED |
| cost-notice comments `variable_selection.py:1408` and `:1686` | **1408** ("surfaced honestly in the T-17 cost notice") and **1686** ("cost notice. See _cars_multi_cell.") | CONFIRMED exact |

---

## Investigation findings that drive the two open decisions

### vcpa-iriv — **DECISION: REVERT THE LEAF (re-assert single-Y-only).**
The leaf at `wavelength_selection.py:358` (`vcpa_iriv`) got a multi-Y PLS-2 joint criterion (`_multi_y` branch, `_evaluate_interval_pls_multi` at line 496) but **still rejects tree-mode 2-D Y** (`_reject_multi_y` at 475). The router `multitarget_grid.py:417` lists `vcpa-iriv` in `SKIP_WITH_NOTICE` with a contradictory comment ("single-Y-only IRIV criterion") that no longer matches the half-upgraded leaf. **vcpa-iriv is never called by the multi-target grid** — the grid only reaches interval methods (`_INTERVAL_METHODS`) and importance methods (`_IMPORTANCE_METHODS`); `vcpa-iriv` is in neither, so it always classifies "skip." The multi-Y PLS branch is dead code reachable only by a direct `vcpa_iriv(X, 2dY)` call. Wiring it into the grid is out of scope for T-17 landing (VIP-CARS does not depend on it) and adds a new method surface + tests. **Cleanest ship: keep `vcpa-iriv` in SKIP_WITH_NOTICE (it stays a skip-with-notice), fix the contradictory comment, and leave the leaf's multi-Y branch but correct the docstring** so the code doesn't claim single-Y-only while supporting 2-D. The stale test `test_vcpa_iriv_rejects_2d_y` asserts the WHOLE function raises on 2-D — that is now false (only tree-mode raises). Flip that test to assert: PLS-mode 2-D returns a result dict; tree-mode 2-D still raises `NotImplementedError`.

### ga_lightgbm multi-Y — **DECISION: DROP (document as intentionally-unrouted), do not wire.**
`ga_lightgbm.py` has a complete multi-Y capability: entry `ga_lightgbm_selection` (line 553) preserves 2-D via `_prep_varsel_y`, and `_fitness_function_lgbm` (228-258) has a per-target MultiOutputRegressor joint-RMSECV branch. But the multi-target grid's `ga` importance path (`_model_independent_importances`, line 318) calls **`ga_pls_selection` only** — never `ga_lightgbm_selection`. And `ga` classifies to "importance" ONLY when a linear model is enabled (`classify_varsel_method`, line 430: `has_linear = any(m in LINEAR_MODELS ...)`). So the LightGBM GA fitness is genuinely unreachable from the multi-target flow. Wiring it means adding a tree-GA route + method string + tests — new surface, not needed by VIP-CARS. **Ship action: leave the code (it is exercised by the single-Y `search.py` path at 3614/6655 and is not dead there), but add a one-line comment at the multi-target `ga` branch noting the grid intentionally routes linear-GA only.** No behavior change; kills the "silent dead code" concern.

### Dangling "T-17 cost notice" — **DECISION: REMOVE the phantom references.**
Comments at `variable_selection.py:1408` ("surfaced honestly in the T-17 cost notice") and `:1686` ("cost notice. See _cars_multi_cell") reference a user-facing cost notice that was never written. No such notice exists in the GUI or backend (grep for "cost notice" finds only these two comments). **Reword both to describe the actual cost (per-target LightGBM = n_targets × single-Y cost) without promising a notice that doesn't exist.**

---

## Work items

### W1 — Gate the SPA preflight
- **File/anchor:** `src/spectral_predict/multitarget_grid.py:610` (inside `run_multitarget_grid_search`).
- **Exact change:** Replace the unconditional `spa_ok = verify_spa_multi_y_safe(X_arr, Y_arr)` with a guarded version: only call `verify_spa_multi_y_safe` when the requested `variable_selection_methods` intersect `_SPA_DEPENDENT_METHODS` (`{"spa","fipls_spa","uve_spa","uve_cars_spa"}`, already defined at line 225); otherwise set `spa_ok = True`. Concretely:
  ```python
  requested = set(variable_selection_methods or [])
  spa_ok = (
      verify_spa_multi_y_safe(X_arr, Y_arr)
      if requested & _SPA_DEPENDENT_METHODS
      else True
  )
  ```
- **Acceptance check:** New test in `tests/test_multitarget_grid.py`: `test_no_spa_preflight_when_no_spa_method` — monkeypatch `multitarget_grid.verify_spa_multi_y_safe` with a spy that records calls; run `run_multitarget_grid_search(..., variable_selection_methods=[])` and `variable_selection_methods=["ipls_forward"]`; assert spy call-count == 0. Add a positive companion `test_spa_preflight_runs_when_spa_selected` with `variable_selection_methods=["spa"]` asserting call-count == 1.
  `…/.venv312/Scripts/python.exe -m pytest tests/test_multitarget_grid.py -q`
- **Dependencies:** Touches `multitarget_grid.py`. **Serialize with W2, W3b, W6a, W7a** (same file). INDEPENDENT of all GUI, variable_selection.py, wavelength_selection.py, ga_lightgbm.py, and docs items.
- **Effort:** S.

### W2 — Thread UVE params through the backend grid
- **Files/anchors:** `src/spectral_predict/multitarget_grid.py` — `run_multitarget_grid_search` signature (565), `build_multitarget_varsel_subsets` signature (438) + call site (681), `_model_independent_importances` signature (305) + UVE-family calls (341-358), prefilter `get_uve_threshold` call (458).
- **Exact change:** Add two params `uve_cutoff_multiplier: float = 1.0, uve_n_components: int | None = None` to:
  1. `run_multitarget_grid_search(...)` signature (add near the other varsel kwargs, keyword-only region).
  2. Pass them into the `build_multitarget_varsel_subsets(...)` call at 681.
  3. `build_multitarget_varsel_subsets(...)` signature (438) — add the two params; forward to `_model_independent_importances(...)` (call at 503) and to the prefilter `get_uve_threshold` (458): `get_uve_threshold(X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier, n_components=uve_n_components)`.
  4. `_model_independent_importances(...)` (305) — add the two params; thread to each UVE-family call:
     - `uve_selection(X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier, n_components=uve_n_components)` (341)
     - `uve_spa_selection(X_pp, Y, n_features=n_target, cutoff_multiplier=uve_cutoff_multiplier, uve_n_components=uve_n_components)` (353)
     - `uve_cars_selection(X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier, uve_n_components=uve_n_components)` (349)
     - `uve_cars_spa_selection(X_pp, Y, cutoff_multiplier=uve_cutoff_multiplier, uve_n_components=uve_n_components, spa_n_features=n_target)` (355-357)
  - NOTE the parameter-name asymmetry (verified in source): `uve_selection`/`get_uve_threshold` take `n_components`; `uve_spa_selection`/`uve_cars_selection`/`uve_cars_spa_selection` take `uve_n_components`. Map correctly per call.
- **Acceptance check:** New tests in `tests/test_multitarget_varsel.py`:
  - `test_uve_params_reach_get_uve_threshold` — monkeypatch `variable_selection.get_uve_threshold` with a spy; call `build_multitarget_varsel_subsets(..., apply_uve_prefilter=True, uve_cutoff_multiplier=1.3, uve_n_components=7)`; assert spy received `cutoff_multiplier==1.3, n_components==7`.
  - `test_uve_params_reach_uve_family` — spy on `variable_selection.uve_selection`; call with `variable_selection_methods=["uve"]` + non-default params; assert forwarded.
  `…/.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -q`
- **Dependencies:** Touches `multitarget_grid.py`. **Serialize with W1, W3b, W6a, W7a.** Depends on nothing else. **W5 (GUI) depends on W2's new signature** — W5 must run AFTER W2.
- **Effort:** M.

### W3 — Flip stale reject/skip tests to positive support + add positive coverage
Split into file-disjoint sub-items so they can parallelize.

- **W3a — `tests/test_variable_selection.py`:** Rewrite `TestMultiYRejectGuards` (class at 895). Remove the 5 `pytest.raises(NotImplementedError)` guard tests (`test_uve_rejects_multi_y`, `test_cars_rejects_multi_y`, `test_uve_spa_rejects_multi_y`, `test_uve_cars_rejects_multi_y`, `test_uve_cars_spa_rejects_multi_y`). Replace with positive tests asserting each returns a full-width `(n_features,)` finite importance/selection on a `(40,3)` Y: e.g. `imp = uve_selection(X, Y); assert imp.shape == (X.shape[1],) and np.all(np.isfinite(imp))`. **KEEP** `test_reject_helper_passes_single_y` (the `_reject_multi_y` helper itself is unchanged and still correctly rejects 2-D — it is just no longer called by UVE/CARS). Rename the class to `TestMultiYSupport`.
  - **Acceptance:** `…/.venv312/Scripts/python.exe -m pytest tests/test_variable_selection.py -q`
  - **Depends on:** nothing (test-only, own file). INDEPENDENT.
  - **Effort:** M.

- **W3b — `tests/test_multitarget_varsel.py` (classify + build):** Fix `test_classify_varsel_method` (272): the loop asserting `["uve","cars","cars-tree","uve_cars","ipls","vcpa-iriv","fipls_cars"]` all → "skip" is wrong. Per `_IMPORTANCE_METHODS` at HEAD, **`uve`, `cars`, `uve_cars`, `fipls_cars`, `ipls` now classify "importance"**; only `cars-tree`, `uve_cars_tree`, `vcpa-iriv` remain "skip". Rewrite: assert `uve`/`cars`/`uve_cars`/`fipls_cars`/`ipls`/`uve_spa`/`uve_cars_spa` → "importance" (note `uve_spa`/`uve_cars_spa` require `spa_ok=True`), and `cars-tree`/`uve_cars_tree`/`vcpa-iriv` → "skip". Fix `test_build_varsel_subsets_full_plus_interval_and_skips` (289): with `["ipls_forward","uve","cars"]` the `skipped` set should now be **empty** (uve/cars produce importance subsets); assert `any(s["method"]=="uve" ...)` and `any(s["method"]=="cars" ...)` appear in subsets and `skipped == set()` (or does not contain uve/cars).
  - **Acceptance:** `…/.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -q`
  - **Depends on:** nothing behaviorally, but **edits the same test file as W2's new tests and W3d** — serialize the test-file edits (W2 tests, W3b, W3d all write `test_multitarget_varsel.py`). Assign all `test_multitarget_varsel.py` edits to ONE agent.
  - **Effort:** M.

- **W3c — `tests/test_multitarget_grid.py` (the 2 extra failures):** Fix `test_grid_search_end_to_end_ranks_and_skips` (108): drop `assert "uve" in out.skipped` — uve now produces subsets, so it should NOT be skipped. Replace with `assert "uve" not in out.skipped` and `assert any(r.varsel_method == "uve" for r in out.results)`. Fix `test_grid_search_apply_uve_prefilter_surfaces_skip` (130): the prefilter is no longer a skip — UVE-on-Y is now multi-Y-safe and contributes a `uve_prefilter` subset (see `build_multitarget_varsel_subsets` lines 449-472). Rewrite as `test_grid_search_apply_uve_prefilter_contributes_subset`: assert `"apply_uve_prefilter" not in out.skipped` and that a `uve_prefilter`-tagged cell may appear (guard for the keep-all/keep-none case: assert the run completes + ranks). Update the docstring accordingly. Leave `test_grid_search_no_uve_prefilter_no_notice` (negative control) unchanged.
  - **Acceptance:** `…/.venv312/Scripts/python.exe -m pytest tests/test_multitarget_grid.py -q`
  - **Depends on:** nothing; own file. INDEPENDENT (but see W1/W7 which ADD tests to this same file — assign all `test_multitarget_grid.py` edits to ONE agent).
  - **Effort:** M.

- **W3d — New positive multi-Y varsel coverage:** Add to `tests/test_multitarget_varsel.py` a `TestMultiYUveCarsSupport` block: `test_uve_multi_y_produces_subsets`, `test_cars_multi_y_produces_subsets`, `test_uve_cars_hybrid_multi_y`, `test_fipls_cars_multi_y` — each drives `build_multitarget_varsel_subsets` with the method and asserts ≥1 non-full subset with finite indices `< n_features`.
  - **Acceptance:** same pytest as W3b.
  - **Depends on:** shares `test_multitarget_varsel.py` with W2/W3b — same-agent serialize.
  - **Effort:** M.

### W4 — Integration test: Independent-only default + `both` companion
- **File/anchor:** `tests/test_multitarget_integration.py` — `test_multitarget_grid_end_to_end_real_data` (def 64, JOINT assertion ~99).
- **Exact change:** Change `assert "JOINT" in modes and "INDEPENDENT" in modes` to `assert modes == {"INDEPENDENT"}` (default is Independent-only, user-confirmed 2026-07-04). Add companion `test_multitarget_grid_both_coupling_yields_both_modes`: same call with `coupling_mode="both"` and `model_names=["PLS","Ridge"]`; assert `"JOINT" in modes and "INDEPENDENT" in modes` (PLS is joint-capable → emits both; Ridge → independent only).
- **Acceptance check:** `…/.venv312/Scripts/python.exe -m pytest tests/test_multitarget_integration.py -q`
- **Dependencies:** Own file. INDEPENDENT.
- **Effort:** S.

### W5 — Thread UVE GUI params into `_collect_multitarget_config`
- **File/anchor:** `spectral_predict_gui_optimized.py` — `_collect_multitarget_config` return dict (15771–15798). Tk vars already exist: `self.uve_cutoff_multiplier` (3866, DoubleVar), `self.uve_n_components` (3867, StringVar). The single-Y path already parses `uve_n_comp` at 29303-29307 (empty→None, int-parse-with-warning).
- **Exact change:** In the return dict (after `"apply_uve_prefilter": ...` at 15787), add:
  ```python
  "uve_cutoff_multiplier": self.uve_cutoff_multiplier.get(),
  "uve_n_components": <parsed int or None>,
  ```
  Parse `uve_n_components` with the same empty→None/int-parse guard used at 29303-29307 (extract to a tiny local, or inline the `.strip()`+`int()`+`except ValueError→None` pattern). Do NOT reference `uve_n_comp` from the other method — recompute locally inside `_collect_multitarget_config`.
- **GUI-FILE WARNING:** 60k-line god-class with CRLF/LF drift. **Run `git diff --stat` immediately after the edit**; expect exactly 1 file changed with a small +line count and no whitespace-only hunks. If phantom hunks appear, revert and redo via binary-mode Python rewrite (per `feedback_edit_tool_line_endings.md`).
- **Acceptance check:** Extend `tests/gui/test_multitarget_tab.py` (has `_collect_multitarget_config` coverage at TestMultiTargetGridDispatch, ~178): add `test_collect_config_forwards_uve_params` — set `gui_app.uve_cutoff_multiplier.set(1.3)`, `gui_app.uve_n_components.set("7")`, call `_collect_multitarget_config()`, assert `cfg["uve_cutoff_multiplier"]==1.3` and `cfg["uve_n_components"]==7`; and an empty-string→None case.
  `…/.venv312/Scripts/python.exe -m pytest tests/gui/test_multitarget_tab.py -q`
- **Dependencies:** **Depends on W2** (backend signature must accept the two new kwargs, else `_run_multitarget_search_thread` `**cfg` unpacking raises TypeError). Touches the GUI file — the ONLY item touching the GUI source, so no GUI-source conflicts. Run in a wave after W2.
- **Effort:** M.

### W6 — vcpa-iriv leaf reconciliation (DECISION: keep skip, fix comments/docstring; flip stale test)
- **W6a — router comment:** `src/spectral_predict/multitarget_grid.py:415-419`. Keep `vcpa-iriv` in `SKIP_WITH_NOTICE`. Fix the contradictory comment at 415-416 to state accurately: vcpa-iriv is skip-with-notice in the multi-target grid because it is not wired into the interval/importance routes (its multi-Y PLS criterion exists at the leaf but is unrouted); NOT because it is "single-Y-only."
  - **Depends on:** touches `multitarget_grid.py` — **serialize with W1/W2/W3b-not(test)/W7a.** Comment-only, tiny.
  - **Effort:** S.
- **W6b — leaf docstring + stale test:** `src/spectral_predict/wavelength_selection.py` `vcpa_iriv` docstring (369+, "Target values (required)" / single-Y framing at 391-392) — note PLS-mode supports 2-D Y (pooled joint RMSECV) while tree-mode rejects it. Flip `tests/test_multitarget_varsel.py::test_vcpa_iriv_rejects_2d_y` (42): assert PLS-mode 2-D returns a dict with `selected_indices`; tree-mode 2-D (`model_type="RandomForest"`) still raises `NotImplementedError`. ALSO flip `tests/test_multitarget_varsel.py::test_ipls_selection_rejects_2d_y` (23): `ipls_selection` now supports 2-D (it is in `_IMPORTANCE_METHODS` via the `ipls` route, `_model_independent_importances` line 338) — assert it returns a finite `(n_features,)` importance; keep `test_ipls_selection_single_y_still_works` (31) as-is.
  - **Depends on:** the `test_multitarget_varsel.py` edits share a file with W2-tests/W3b/W3d — SAME AGENT. The `wavelength_selection.py` docstring edit is file-disjoint from everything else and can go with any wave.
  - **Effort:** S.

### W7 — ga_lightgbm (DECISION: document unrouted) + cost-notice cleanup
- **W7a — ga route comment:** `src/spectral_predict/multitarget_grid.py:318` (the `ga` branch of `_model_independent_importances`, which calls `ga_pls_selection`). Add one comment line: the multi-target grid intentionally routes linear GA (`ga_pls_selection`) only; `ga_lightgbm_selection`'s multi-Y fitness is exercised only by the single-Y `search.py` path.
  - **Depends on:** touches `multitarget_grid.py` — serialize with W1/W2/W6a.
  - **Effort:** S.
- **W7b — cost-notice comments:** `src/spectral_predict/variable_selection.py:1408` and `:1686`. Reword both to drop the phantom "T-17 cost notice" promise; state the actual cost plainly (per-target LightGBM = n_targets × single-Y CV cost).
  - **Depends on:** touches `variable_selection.py` — file-disjoint from the multitarget_grid cluster; can run in Wave 1. (Note: `variable_selection.py` source is NOT edited by any other work item — W3a only edits its TEST file.)
  - **Effort:** S.

### W8 — Docs: reflect shipped state
- **Files:** `docs/PROJECT_STATUS.md` (T-17 active-direction block, lines 3-21; the "UVE/CARS ... raise NotImplementedError ... v1.1 deferred" and "UVE/CARS (approved deferral)" statements at lines ~5/18), `docs/SESSION_LOG.md` (add a dated 2026-07-07 entry).
- **Exact change:** Update PROJECT_STATUS to state UVE/CARS/hybrids + ipls-legacy are now multi-Y-shipped (importance route), that vcpa-iriv/cars-tree/uve_cars_tree remain skip-with-notice, that UVE GUI params are wired end-to-end, and record the new test gate. Add a SESSION_LOG entry summarizing the landing (findings verified, decisions: vcpa-iriv keep-skip, ga_lightgbm document-unrouted, cost-notice removed).
- **Acceptance check:** N/A (docs). Grep to confirm no remaining "NotImplementedError on 2-D Y"/"v1.1 deferred"/"approved deferral" claims about UVE/CARS.
- **Dependencies:** Own files. INDEPENDENT of all source. Do LAST so it describes the final state.
- **Effort:** S.

---

## Parallelization map (file-disjoint waves)

Conflict groups by file:
- **G-grid** (`multitarget_grid.py` source): W1, W2, W6a, W7a — **one agent, serialized in this order.**
- **G-tv** (`tests/test_multitarget_varsel.py`): W2-tests, W3b, W3d, W6b-test — **one agent, serialized.**
- **G-tgrid** (`tests/test_multitarget_grid.py`): W1-tests, W3c, W7a-test — **one agent** (W1's SPA-spy test + W3c edits + any W7 test live here). Assign W1's test authoring to the same agent that owns G-grid OR coordinate: simplest is G-grid agent writes both the source AND its tests in `test_multitarget_grid.py`, and also folds in W3c. So **G-grid + G-tgrid = one agent.**
- File-disjoint singletons: W3a (`test_variable_selection.py`), W4 (`test_multitarget_integration.py`), W5 (GUI source — only GUI toucher), W6b-doc (`wavelength_selection.py`), W7b (`variable_selection.py` source), W8 (docs).

**Wave 1 (all concurrent — no shared files):**
- Agent A = G-grid+G-tgrid: **W1 → W2 → W6a → W7a** in `multitarget_grid.py`, plus their tests + **W3c** in `test_multitarget_grid.py`.
- Agent B = G-tv: **W3b + W3d + W6b-test** in `test_multitarget_varsel.py` (plus W2's forwarding tests — see note).
- Agent C: **W3a** (`test_variable_selection.py`).
- Agent D: **W4** (`test_multitarget_integration.py`).
- Agent E: **W7b** (`variable_selection.py` cost-notice) + **W6b-doc** (`wavelength_selection.py` docstring) — two disjoint source files, one agent fine.

**Note on W2↔Agent B:** W2's *source* change lives in `multitarget_grid.py` (Agent A). W2's *tests* live in `test_multitarget_varsel.py` (Agent B). Agent B's UVE-param tests will FAIL until Agent A lands the signature. Either (a) Agent A also writes W2's tests in `test_multitarget_varsel.py` (then B must not touch that file — collapse B into A), or (b) run Agent A first, then B. **Recommended: make Agent A own ALL of `multitarget_grid.py` + `test_multitarget_grid.py` + `test_multitarget_varsel.py`** (grid backend + both its test files), and let B/C/D/E take the disjoint files. That removes the cross-agent ordering hazard entirely. Revised minimal fan-out:

- **Agent A:** W1, W2, W3b, W3c, W3d, W6a, W6b-test, W7a (all `multitarget_grid.py` + `test_multitarget_grid.py` + `test_multitarget_varsel.py`).
- **Agent B:** W3a (`test_variable_selection.py`).
- **Agent C:** W4 (`test_multitarget_integration.py`).
- **Agent D:** W7b + W6b-doc (`variable_selection.py` + `wavelength_selection.py` sources).
- **Agent E:** — reserved for W5 (GUI), Wave 2.

**Wave 2 (after Agent A lands W2's signature):**
- **W5** (GUI `_collect_multitarget_config` + `test_multitarget_tab.py`). Depends on W2. Only GUI-source toucher → no conflict once Wave 1 done.

**Wave 3 (after all code green):**
- **W8** (docs) — describes final state.

---

## Completion gate

All must be fully green (run from worktree root with the venv):

```
C:/Users/mspon/git/dasp/.venv312/Scripts/python.exe -m pytest \
  tests/test_variable_selection.py \
  tests/test_multitarget_varsel.py \
  tests/test_multitarget_integration.py \
  tests/test_multitarget_grid.py \
  tests/test_multitarget_search.py \
  tests/test_multitarget_export.py \
  tests/test_multitarget_save_load_roundtrip.py \
  tests/test_multi_y.py \
  tests/test_wavelength_selection.py \
  tests/gui/test_multitarget_tab.py \
  -q
```

(These are the 10 files Codex named plus the two adjacent files that also carry stale contracts: `test_multitarget_grid.py` and `test_wavelength_selection.py` — confirmed necessary because `test_multitarget_grid.py` has 2 of the 12 real failures. If a stricter "exactly Codex's 10" gate is wanted, ensure the list contains: `test_variable_selection`, `test_multitarget_varsel`, `test_multitarget_integration`, `test_multitarget_grid`, `test_multitarget_search`, `test_multitarget_export`, `test_multitarget_save_load_roundtrip`, `test_multi_y`, `test_wavelength_selection`, `test_multitarget_tab`.)

**Pre-existing main-red set (do NOT block on):** per `feedback_check_ci_before_merge.md`, main has been red since 2025-10-27 with `test_cv_strategy` nameerror, `test_export_code` ×2, `test_t19_class_weight_per_library` ×2. Confirm the landing adds **zero new failures vs `origin/main`** by diffing the full-suite failure set — do not chase these 5.

**GUI smoke (manual, per `feedback_e2e_smoke_after_refactor.md`):** launch `python spectral_predict_gui_optimized.py`, load `example/` ASD + `example/BoneCollagen.csv`, select 2 targets, enable UVE + CARS varsel, set a non-default `uve_cutoff_multiplier`, run a multi-target grid, confirm UVE/CARS cells appear on the leaderboard (not skip-notices) and the run completes. Unit tests do not catch GUI-integration breakage.
