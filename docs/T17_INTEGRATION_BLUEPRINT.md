# T-17 Multi-Target Tab — Integration Blueprint

**Branch:** `feat/T17-multitarget-regression` @ `9222e63` (worktree `C:/Users/mspon/dasp-t17-uve-cars`)
**Verdict being addressed:** "The multi-target feature is not well integrated into the rest of the program."
**Audience:** build team (GLM 5.2 workers for mechanical items, Opus for judgment items).

---

## 0. Governing constraints (from user — non-negotiable)

1. **Integration into the main flow is PREFERRED (revised steer).** The user finds it
   strange/undesirable that this one feature works differently from everything else.
   Own-tab/own-sub-tabs is **acceptable but a FALLBACK** — used only where safe
   integration into the shared single-Y machinery is not feasible. Each integration
   point is therefore decided by **feasibility + safety, not aesthetics**: can it join
   the shared machinery **additively and with guard tests, without risking the PLS1
   regression or classification flows?** YES → INTEGRATE (preferred). NO → STANDALONE
   fallback, with the reason safe integration was infeasible spelled out.
2. **ZERO regression to single-Y PLS1 regression and classification flows** — unchanged,
   and it is now the literal test that decides integrate-vs-standalone per gap. Every
   item must be **additive**. Where integration needs a shared component to change, the
   item carries a byte-identity / guard test proving single-Y regression + classification
   behavior is unchanged, and is flagged **HIGHER-RISK**. Precedent guardrails exist and
   must stay green:
   - `tests/test_multitarget_save_load_roundtrip.py:208-278` — single-Y `.dasp` metadata
     gold snapshot (`TestSingleYMetadataGoldSnapshot`) + byte-identical predict.
   - `tests/test_multitarget_export.py:170-189` — single-target config never activates the
     multi-target code generator.
3. **Do not re-report sortable columns** — fixed in `9222e63`
   (`_sort_multitarget_tree`, gui:16092).

### The load-bearing distinction: shared BACKEND vs shared GUI WIDGETS

Investigation shows the answer splits cleanly along one line, and this line drives every
per-gap verdict below:

- **Shared BACKEND ecosystem is schema-neutral and CAN be integrated safely** — it is
  already gated on metadata keys / config shape that single-Y saves never carry:
  `model_io.save_model`/`load_model`/`predict_with_model` (gated on `y_scaler` /
  `multitarget_mode` / `n_targets`), `CodeGenerator` (auto-activates on
  `len(target_names)>1`), the progress-callback payload (already shaped to match
  `_progress_callback_impl`). Making multi-target artifacts first-class **here** is the
  integration that actually matters to the user ("saves/exports/predicts like everything
  else"). All INTEGRATE-verdict.
- **Shared GUI WIDGETS that are single-Y-SCHEMA-BOUND cannot be extended safely** —
  `results_tree` + `_populate_results_table_inner` (gui:31172) hard-branch on the single-Y
  column schema (class/regional rankings, `filter_*` vars, RMSEP-gap, expert choices,
  header tooltips, quartile tags), and `_load_model_for_refinement` (gui:35488) + Model
  Dev assume scalar `self.y`, 1-D RMSE/R², refine-CV restore, and 1-D prediction plots.
  A 6×n_targets + joint-Q² + Mode schema would require branching every one of these hot
  paths that serve PLS1 + classification — exactly the risky surgery the constraint
  forbids. STANDALONE-fallback verdict, but wired to the shared backend so the artifacts
  produced are first-class.

---

## 1. Current architecture map (verified file:line)

### Multi-target flow (all in `spectral_predict_gui_optimized.py` unless noted)

| Piece | Location |
|---|---|
| Tab creation (sub-tab 4F of Analysis Config `config_notebook`) | `_create_tab4f_multitarget` gui:14528-14753 |
| Target listbox + refresh + selection sync | gui:14588-14591, 14773-14803 |
| Engine lock (greys Bayesian/NSGA-II when >1 target) | `_update_multitarget_engine_lock` gui:14805-14855 |
| Config collection (reads the SAME shared widgets as single-Y run) | `_collect_multitarget_config` gui:15709-15808 |
| Run dispatch (worker thread + queue) | `_run_multitarget_search` gui:15839-15920 |
| Worker body | `_run_multitarget_search_thread` gui:15931-15955 |
| Progress callback (worker → queue) | `_multitarget_progress` gui:15957-15964 |
| Main-thread poller | `_poll_multitarget_queue` gui:15966-16018 |
| Done/failed handlers | gui:16020-16037 |
| Cancel (own `SearchController`, gui:14726-14727) | `_cancel_multitarget_search` gui:16039-16046 |
| Results grid populate | `_populate_multitarget_results` gui:16048-16090 |
| Column sort (just fixed) | `_sort_multitarget_tree` gui:16092-16118 |
| CSV export (the ONLY export) | `_export_multitarget_csv` gui:16120-16157 |
| Backend search | `run_multitarget_grid_search` `src/spectral_predict/multitarget_grid.py:610-824` |
| Result dataclasses | `MultiTargetResult` / `MultiTargetSearchOutput` `multitarget_search.py:557-611` |
| Estimator builders (refit-capable) | `build_multitarget_estimator` `multitarget_search.py:316-510` |

### Single-Y reference flow (the conventions to match)

| Piece | Location |
|---|---|
| Run entry (validates, **switches to Analysis Progress tab**, clears log, starts animation/timer) | `_run_analysis` gui:25415-25493 (`self.notebook.select(6)` at 25478) |
| Analysis Progress tab: animated figure, "Best Model So Far", progress info + ETA, Pause/Resume/Stop, 2000-line capped log, status | `_create_tab5...` gui:16160-16221 |
| Progress callback: log line + `current/total` + elapsed/ETA + best-model panel (regression/classification/one-class variants) | `_progress_callback_impl` gui:30756-30843 |
| Disk-mirrored log (per-run file) | `_log_progress` gui:30845-30868 → `run_logging.log_event`; logger created in `_run_analysis_thread` gui:27394-27407 (`setup_run_logger(label=tier)`) |
| Pause/Resume/Stop handlers | gui:25023 / 25067 / 25083, driving `SearchController` (`src/spectral_predict/search_controller.py:6-114`, supports pause+resume+end + `is_actually_paused`) |
| Results tab: card, filter bar, sort hint, header tooltips (`TreeviewHeaderTooltip` + `TOOLTIP_CONTENT['metrics']`), quartile row tags, click + double-click bindings | `_create_tab6_results` gui:16223-16299 (bindings 16275-16277, tooltips 16280-16284) |
| Double-click → attach training/validation config → `_load_model_for_refinement` → jump to Model Dev tab 8 | `_on_result_double_click` gui:32713-32782 |
| Save `.dasp` (refined model, Model Dev) | `_save_refined_model` gui:41077, `save_model(...)` call gui:41346 |
| Export code (Colab notebook / bundle / py / R) dialog | `_export_for_publication` gui:41371-41605 (colab branch 41596-41605) |
| 300-DPI plot export helper | `_add_plot_export_button` gui:18267-18300 |
| Prediction tab: loads `.dasp`, one prediction **column per model** | gui:43218-43301 (`results[col_name] = predictions` at 43301) |

---

## 2. Backend-ready inventory (capability that exists but is UNWIRED in the GUI)

This is the cheapest work in the blueprint — tests already pin these behaviors.

| Capability | Where | Pinned by |
|---|---|---|
| `.dasp` save/load of multi-target models: `y_scaler` persistence (`y_scaler.npz`), `target_names` / `multitarget_mode` / `n_targets` / `prediction_columns` / `per_target_metrics` metadata, loud-guard "JOINT without y_scaler raises", backward-compat load | `model_io.save_model` (y_scaler param `model_io.py:98`, guard 367-375, write 377-382, 413-414), `load_model` (552-573), `predict_with_model` raw-unit inverse-transform (823-828) | `tests/test_multitarget_save_load_roundtrip.py` (all green on branch) |
| Code export (script AND notebook) for multi-target: `CodeGenerator.is_multitarget` (`code_generator.py:156`), `_generate_multitarget_script` (825), `_generate_multitarget_notebook` (878); exported script reproduces in-app pooled CV predictions bit-for-bit, honest JOINT/INDEPENDENT headers, final-model refit + `predict_raw()` in raw units | `code_generator.py` + `templates/` | `tests/test_multitarget_export.py` |
| Progress payload with `message`, `current`, `total`, `best_model` dict (same shape `_progress_callback_impl` consumes) | `multitarget_grid.py:802-814` | emitted today; GUI discards all but `message` |
| Pause/Resume/Stop: the multi-target worker already checks `controller.check_and_wait()` per preprocessing config (grid:700) and per cell (grid:791); `SearchController` supports pause/resume/end | `search_controller.py:39-89` | Cancel is wired (gui:16039); Pause/Resume just lacks buttons |
| Refit builders for a final model (JOINT native / MOR-wrapped INDEPENDENT, component capping, `FoldYScaler`) | `build_multitarget_estimator` `multitarget_search.py:316-510`; `FoldYScaler` in `multi_y.py` | roundtrip + export tests exercise them |
| Pooled CV predictions retained per result → per-target predicted-vs-actual plots need **zero recompute** | `MultiTargetResult.y_true_pooled` / `y_pred_pooled` `multitarget_search.py:586-587` | populated by `_evaluate_multitarget_cell` |

**Backend genuinely MISSING (not just unwired):**
- No single "refit this leaderboard row on full calibration data" helper. A search result
  retains `preprocessing` (string), `varsel_method`, `varsel_tag`, `n_variables`,
  `params`, `mode` (`multitarget_search.py:578-591`) but **NOT the fitted estimator and
  NOT the variable-selection indices**. Save-from-leaderboard therefore requires
  re-running preprocessing + varsel to reconstruct the column subset (deterministic,
  `random_state=42` throughout, and `build_multitarget_varsel_subsets` is cached/pure) —
  but that reconstruction helper does not exist yet. See item W3-1.
- Multi-target export tests pin `preprocessing='raw'` + full-spectrum only. The
  generator does route preprocessing + varsel application into multi-target scripts
  (`code_generator.py:837-850`), but a preprocessed/subset multi-Y export is
  **generated-yet-unpinned** — verify before shipping W2-2.

---

## 3. Gap-by-gap analysis

### GAP 1 — "The format is completely unlike all other methods"

**Verdict: PART real deficiency, PART justified.** The tab already uses the app's card
system (`_create_card` gui:14580, 14601, 14656, 14691), `Subheading.TLabel` /
`Caption.TLabel` styles, the accent run button (`_create_accent_button` gui:14675), and
`CreateToolTip` on model checkboxes (gui:14634). It is not un-themed. The real
divergences:

| # | Divergence | Single-Y convention | Multi-target today | Real gap? |
|---|---|---|---|---|
| 1a | Run entry: every other config sub-tab (4A gui:11374, 4B 12119, 4C 12417, 4D 14221, 4E 14375) has "▶ Run Analysis" driving the shared flow; 4F has its own "▶ Run Multi-Target Search" mid-page | shared "Run Analysis" entry | separate Run button + separate engine | **Partial INTEGRATE.** The separate controller/engine is correct (zero-regression), but the *entry point* can converge: keep the 4F Run button, but on completion route progress→tab 5 and give results a home in tab 6 (Gaps 2/3). Full convergence of the button itself (one "Run Analysis" that detects >1 target) is possible but adds a branch to `_run_analysis` gui:25415 — **defer unless user wants it** (D-decision). Placement: move the 4F Run/Cancel row to the TOP like siblings. S |
| 1b | Results grid lives inside a **config** sub-tab; every other config sub-tab is configuration-only, results go elsewhere | config and results separated | config + status + results stacked in one scrolling canvas | Per user steer: standalone is fine. The consistent fix is **sub-structure**: give 4F its own internal notebook (Setup / Progress / Results sub-tabs) or clearly separated cards with the Results card getting the Results-tab treatments (see 1d, Gap 3). M |
| 1c | `tk.Listbox` for target selection is a raw unthemed widget (gui:14588-14589) — no dark-mode colors, unlike themed ttk everywhere else | ttk themed widgets / checkbox lists | raw Listbox with default colors | **Real (cosmetic)**. Either style the Listbox from `self.colors` (bg/fg/selectbackground) or swap to the checkbox-list pattern. S |
| 1d | No header tooltips on the results tree | `TreeviewHeaderTooltip` + `TOOLTIP_CONTENT['metrics']` (gui:16280-16284) | none | **Real.** Add a `TreeviewHeaderTooltip` with multi-target metric explanations (Joint Q², Mode, per-target R²/RMSE/RPD/RER/CCCcv/Bias). Note: tooltip must key on the DYNAMIC per-target column ids built in `_populate_multitarget_results` (gui:16056-16075). S |
| 1e | No Rank column, no row-count/status line, no filter bar | Results tab has Rank, filter bar, sort hint (gui:16236-16254) | model/mode/joint_q2 + per-target cols only | **Partially real.** Rank column: yes, cheap. Filter bar/quartile tags: single-Y-specific machinery — **justified skip** for v1. S |
| 1f | Status feedback is a small inline `Caption` label (gui:14686-14688) | dedicated Progress surface | one-line label | Covered by Gap 2. |

**Scope/routing:** 1a+1c+1d+1e = **S, GLM-suitable** (mechanical, additive, one file).
1b = **M, needs-Opus for the layout decision** (sub-notebook vs cards), GLM for execution.
**Shared-code risk:** none — all inside `_create_tab4f_multitarget` and new methods.
One caution: gui:14750-14753 binds `<<NotebookTabChanged>>` on `config_notebook` with a
single lambda ("no other handler" invariant documented there). If 1b adds an inner
notebook, do NOT bind on `config_notebook` again — use `bind('...', add='+')` or the new
inner notebook.

### GAP 2 — "Nothing happens in progress"

**Verdict: REAL, and the single cheapest/highest-value fix in this blueprint.** The
backend already emits everything the single-Y progress surface consumes; the GUI throws
it away.

**Root cause (three layers):**
1. `_poll_multitarget_queue` (gui:15977-15979) uses ONLY `payload["message"]` and writes
   it to the one-line status label. The payload also carries `current`, `total`,
   `best_model` (`multitarget_grid.py:811-814` — deliberately shaped to match
   `_progress_callback_impl`, see docstring gui:15960-15962) — all discarded.
2. **The first pass is silent.** `run_multitarget_grid_search` builds the full cell list
   first (preprocessing × varsel × model grids, grid:699-787) and only starts emitting
   progress in the second-pass evaluation loop (grid:802). Variable selection (UVE/CARS/
   iPLS) is the expensive part, so for large configs the user stares at a static
   "Running…" (set once at gui:15912) for the entire varsel phase. This is the literal
   "nothing happens".
3. No disk log: single-Y calls `setup_run_logger` per run (gui:27407) and mirrors every
   line (`_log_progress` gui:30845); the multi-target worker never does either.
4. No Pause/Resume buttons — even though `_multitarget_controller` is a full
   `SearchController` (pause/resume/end all supported, `search_controller.py:39-59`) and
   the worker already honors `check_and_wait()` at both loop levels (grid:700, 791).

**RECOMMENDATION: INTEGRATE (preferred) — reuse the shared Analysis Progress tab (tab 5).**
This is safe: the progress payload is already shaped to match `_progress_callback_impl`
(deliberately, per docstring gui:15960-15962), and a single-Y run and a multi-target run
are **mutually exclusive in practice** (both are user-triggered; nothing starts two at
once). The only shared-widget hazard is the Pause/Resume/Stop buttons, which today call
`self.search_controller` directly (gui:25023/25067/25083) — that is resolved additively
(below) and guard-tested.

**What integrated looks like:**
- `_run_multitarget_search` switches to tab 5 (`self.notebook.select(6)`), clears the
  progress log, starts the running-figure animation, and sets `analysis_start_time` —
  exactly the single-Y preamble (gui:25476-25492).
- `_poll_multitarget_queue` feeds the SHARED widgets from the payload it already receives:
  `progress_info` / `time_estimate_label` (`current`/`total` + ETA via the gui:30764-30794
  arithmetic), `best_model_info` (`payload["best_model"]`), and `progress_text` via
  `_log_progress` (which also gives the disk mirror for free).
- **Shared-code change (additive, guard-tested):** add an "active controller" indirection
  so Pause/Resume/Stop drive whichever run is live. Cleanest: a
  `self._active_search_controller` attribute (defaults to `self.search_controller`);
  `_run_analysis` and `_run_multitarget_search` each set it to their controller at
  dispatch; `_pause_search`/`_resume_search`/`_stop_search` operate on it. Guard test:
  a single-Y run with no multi-target run present still pauses/stops identically (assert
  the same controller object is used). **This is the one HIGHER-RISK touch in Gap 2 —
  Opus reviews.**
- A concurrent-start guard already exists for multi-target (gui:15859-15863); add the
  symmetric guard so a multi-target run can't start while a single-Y run is active (and
  vice-versa) — cheap, and it is what makes the shared-tab reuse safe.
- **Backend (small, additive):** emit progress from the first pass too — a callback per
  preprocessing config / per varsel method inside grid:699-787
  (`{"message": f"Variable selection: {method} on {pp_name}…", "current": 0, "total": 0}`).
  Pure addition to `multitarget_grid.py`; no single-Y file touched. This kills the literal
  "nothing happens" (varsel dominates wall-clock).

**Scope: M. GLM-suitable** for the wiring; the active-controller indirection + concurrent-
start guard is the judgment point → **Opus reviews that diff.** If the active-controller
change is judged unsafe on review, the STANDALONE fallback (a Progress card inside tab 4F
with its own bar/log/pause buttons) is the documented alternative.
**Shared-code risk:** medium (tab-5 widgets + the three Pause/Stop handlers). Add a single-Y
progress-parity guard test; keep `test_multitarget_grid.py` green.

### GAP 3 — "Results stay in the same tab"

**RECOMMENDATION: STANDALONE FALLBACK — safe integration into the shared `results_tree`
is NOT feasible.** This is the one gap where I recommend against the preferred outcome,
and here is the concrete why (grounded in `_populate_results_table_inner` gui:31172-31289):

- The single-Y results grid is not schema-neutral. `_populate_results_table_inner`
  hard-branches on the single-Y column schema: `is_classification = 'per_class_metrics'
  in results_df.columns` (gui:31199), `is_one_class` (31200), `regional_rmse` regression
  ranking (31239), class rankings (31202), `_compute_expert_choices` (31274), the RMSEP-
  gap checkbox (31280-31288), and a fixed battery of filter vars (`filter_min_r2cv_var`,
  `filter_max_rmsecv_var`, `filter_max_rmsep_var`, … 31184-31191). The header tooltips key
  on `TOOLTIP_CONTENT['metrics']` (gui:16280), the quartile tags on single-Y Y-regions
  (16287-16290), and `_on_result_double_click` consumes a flat single-Y row dict
  (gui:32770-32773).
- A multi-target result is a fundamentally different shape: 6 metrics × n_targets (built
  dynamically, gui:16056-16075) + joint Q² + a JOINT/INDEPENDENT Mode column, with a
  variable column count per run. Injecting it into `results_tree` means adding a
  multi-target branch to **every one of the hot paths above** — the exact functions that
  render PLS1 regression and classification results. That is broad surgery on the
  highest-traffic shared widget, and the regression surface is enormous (filters, sort,
  quartile highlighting, expert choices, decision-view, Model Dev handoff all assume the
  single-Y schema). It fails the "additive + guard-testable without risking PLS1/
  classification" test.

**Fallback done right (bring the standalone grid to full parity + co-locate):** keep the
dedicated `multitarget_tree` but (a) give it the Results-tab visual treatments it is
missing — header tooltips, Rank column, results-count line, stable iids (Gap 1 items
1d/1e + Gap 4), and (b) reduce the "works differently" feeling by **placing it in the main
Results tab as a switched secondary view** (a small "Single-target / Multi-target" segmented
toggle at the top of tab 6 that swaps which tree is packed) rather than buried in a config
sub-tab. The toggle is additive (packs/unpacks frames; touches no single-Y render code) and
makes results feel like they live "in the same place" without merging the schemas. If even
the co-location toggle is judged too invasive, the current in-tab (4F) grid is the final
fallback. **Scope: M for the co-location toggle, GLM-suitable; S for the parity items.**
**This is DESIGN DECISION #1 for the user** (co-locate via toggle vs. keep in 4F).

### GAP 4 — "Can't double-click to run a model when done"

**RECOMMENDATION: STANDALONE FALLBACK (detail dialog) — safe integration into the shared
Model Dev refine path is NOT feasible; a standalone dialog fully satisfies the complaint.**

**Root cause:** `self.multitarget_tree` has zero event bindings (creation gui:16699-16716;
only header `command=` sorts). The single-Y double-click (`_on_result_double_click`
gui:32713) feeds `_load_model_for_refinement` (gui:35488) and jumps to Model Dev (tab 8).

**Why integration is infeasible (verified in `_load_model_for_refinement` gui:35488-35561):**
that handler and the Model Dev tab it drives are thoroughly single-Y-shaped — it restores
scalar validation indices / excluded spectra / refine-CV state (gui:35499-35531), branches
on a flat single-Y row dict (`RMSE`/`R2` vs `Accuracy`, gui:35542-35561), operates against
a scalar `self.y`, and every Model Dev plot assumes 1-D `refined_y_pred`
(`_plot_regression_predictions` gui:36165-36255). Extending all of it for Y-blocks is XL,
cross-cutting surgery on the exact code that serves PLS1 refinement — it fails the safety
test. So true "refine in Model Dev" is deferred (user decision #2).

**What "integrated" looks like (recommended, additive):** double-click opens a
**Multi-Target Model Detail dialog** (new `Toplevel`, styled like `_export_for_publication`'s
dialog gui:41391+ but themed):
- Per-target predicted-vs-actual scatter grid — **zero recompute**: every
  `MultiTargetResult` retains `y_true_pooled` / `y_pred_pooled`
  (`multitarget_search.py:586-587`); one subplot per target with per-target
  R²/RMSE/RPD/RER/CCCcv/Bias from `res.metrics["per_target"]`.
- Correlation context from `output.correlation` (`inter_target_correlation`,
  grid:653) — the "why joint?" panel.
- Buttons: **Save Model (.dasp)** (→ W3-1), **Export Code (script/notebook)** (→ W2-2),
  **Export Plot (300 DPI)** via the existing helper `_add_plot_export_button`
  (gui:18267 — takes any Figure; confirm it has no single-Y state coupling — from its
  signature `(parent_frame, figure, default_filename)` it does not).
- Row→result mapping: `_populate_multitarget_results` currently inserts rows without
  stable iids tied to `output.results` indices (gui:16077-16090) — insert with
  `iid=str(result_index)` so double-click and sorting stay consistent (sort moves items,
  `index()`-based lookup would break; the single-Y handler had exactly this bug class,
  gui:32730-32743).

**True "refine in Model Dev" for multi-target** = adapting `_load_model_for_refinement`,
Model Dev fitting, and all its plot panels for Y-blocks. Estimate **XL, needs-Opus,
touches heavily-shared code** — defer; listed as a user decision below.

**Scope:** detail dialog + binding = **M, GLM-suitable** once W2-2/W3-1 exist to wire the
buttons to (dialog can ship first with plots + CSV/plot export only).
**Dependencies:** gui file only; consumes `_multitarget_last_output` (gui:16029).

### GAP 5 — "Can the models be saved and exported as Colab files?"

**RECOMMENDATION: INTEGRATE — the shared model/export ECOSYSTEM (`model_io`,
`CodeGenerator`, the Prediction tab) is schema-neutral and already accepts multi-target
artifacts; make them first-class there.** Only the GUI *trigger buttons* are standalone
(they can't live in Model Dev, which is single-Y-shaped per Gap 4), but the artifacts
themselves save/load/export/predict through the **same** backend as every other model —
which is exactly the "works like everything else" the user wants. Today the tab's only
export is `_export_multitarget_csv` (gui:16120-16157) — a metrics table, not a model.

Split into three work items:

**(a) Export Code — script + Colab notebook. INTEGRATE. Backend READY, GUI trigger unwired.**
- `CodeGenerator` auto-activates the multi-target generator when
  `config['target_names']` has >1 entry (`code_generator.py:156`); notebook + script
  paths both pinned to reproduce in-app pooled CV predictions
  (`tests/test_multitarget_export.py:78-167`), honest JOINT/INDEPENDENT headers
  (:209-221), NeuralBoosted builder (:224-234).
- GUI work: an "Export Code…" action (control row + detail-dialog button) that builds
  `config = {model_name, params, preprocessing, task_type='regression', target_names,
  wavelengths, cv_folds, cv_strategy, cv_n_repeats}` from the selected
  `MultiTargetResult` + the run's config, and `ExportOptions(format='notebook'|'script',
  include_data=True, data_X=…, data_y=Y_block, wavelengths=…)`. Mirror the colab branch
  of `_export_for_publication` (gui:41596-41605) but as a NEW method — do not touch the
  single-Y dialog.
- **Caveats to verify (Opus review):** (i) exported preprocessing string must be the
  generator's expected key — search results carry `_describe_preprocess_config(pc)`
  output (grid:796); confirm it maps through `preproc_map` (`code_generator.py:757-769`);
  (ii) varsel subsets: embedded `data_X` should be the RAW spectra with the generator
  re-applying preprocessing+varsel, or pre-subset — follow whichever convention the
  single-Y embedded-data path uses (gui:41438+ `do_export`); (iii) add a pinning test for
  one preprocessed+varsel multi-Y export (tests currently pin raw/full only).
- **Scope: M.** Mechanical part GLM-suitable; the three caveats need an Opus pass.

**(b) Save `.dasp`. INTEGRATE. Backend save/load READY; the missing piece is the final refit.**
- `model_io.save_model` already accepts `y_scaler` and the full multi-target metadata
  contract, with the JOINT-without-scaler loud guard (model_io.py:367-375) and the
  single-Y gold-snapshot protection.
- Needed: `refit_multitarget_final(X_raw, Y, result, run_config) -> (pipeline_or_none,
  fitted_estimator, y_scaler_or_none, variable_indices)` in `multitarget_search.py` or
  `multitarget_grid.py` (backend, additive):
  1. rebuild preprocessing via `build_preprocessing_pipeline` from the result's
     preprocess config (grid:703-712);
  2. reconstruct the varsel column subset — re-run
     `build_multitarget_varsel_subsets` (grid:735-744) with identical inputs and select
     the subset whose `tag == result.varsel_tag` (deterministic: fixed seeds throughout);
     **risk:** the result stores only the tag string; if two subsets could share a tag,
     disambiguate — verify tag uniqueness per preprocess config;
  3. `build_multitarget_estimator(strategy, result.params, n_samples=N_full, …)`; for
     JOINT fit on `FoldYScaler().fit(Y).transform(Y)` and persist the scaler; for
     INDEPENDENT fit raw (exactly the exported-script final-fit semantics pinned by
     `test_exported_final_model_predicts_raw_units_for_joint`);
  4. `save_model(model=…, preprocessor=…, metadata={target_names, multitarget_mode,
     n_targets, prediction_columns, per_target_metrics, wavelengths(subset), …},
     y_scaler=…)`.
- GUI: "💾 Save Model" in control row + detail dialog; NEW method, no touch of
  `_save_refined_model`.
- **Scope: L. Needs-Opus** (refit correctness: scaled-vs-raw Y, varsel reconstruction,
  wavelength bookkeeping for the saved subset). Tests: extend the roundtrip suite with a
  "save from a search result, reload, predict" test; single-Y gold snapshot must stay
  green.

**(c) NEW GAP (found during investigation): INTEGRATE — the Prediction tab cannot consume a
multi-target `.dasp`.** This is the strongest "make it work like everything else" item:
a saved multi-target model should load and predict in the shared Prediction tab exactly
like a single-Y model. The predict loop assigns one column per model —
`results[col_name] = predictions` (gui:43301) — but `predict_with_model` on a JOINT/
INDEPENDENT multi-Y model returns `(n, k)` (model_io.py:823-828), which will raise or
mis-broadcast. The metadata already carries `prediction_columns` (pinned at
roundtrip test :175) for exactly this fan-out. Fix: in the prediction loop, if
`metadata.get('n_targets', 1) > 1`, fan predictions out to
`f"{col_name}::{target}"` columns using `metadata['prediction_columns']`; leave the
1-D path byte-identical (guard on the metadata key, which single-Y saves never have —
gold snapshot proves it). Uncertainty/AD panels for multi-Y: out of scope v1, show
"n/a". **Scope: M.** GLM-suitable with the guard spec, but this DOES touch the shared
prediction loop → **HIGHER-RISK flag; requires a single-Y prediction regression test**
(load an existing single-Y `.dasp`, assert identical results table).
Without (c), shipping (b) gives users a model file they can only use via exported code —
call this out in the user decisions.

---

## 4. Additional findings (not in the user's list)

1. **First-pass silence** (see Gap 2, root cause 2) — the biggest contributor to
   "nothing happens" on real workloads because varsel dominates wall-clock.
2. **No stable row iids** in `_populate_multitarget_results` (gui:16077-16090) —
   prerequisite for double-click (Gap 4) to survive the new column sorting.
3. **Run-lifecycle button states:** during a run only the Run button is disabled
   (`_set_multitarget_running` gui:15922-15929); Export CSV stays clickable and will
   export the PREVIOUS run's `_multitarget_last_output` mid-run. Minor; disable exports
   while running. S, GLM.
4. **`effective_n_notice` — VERIFIED NOT A BUG.** It is popped before dispatch
   (`cfg.pop("effective_n_notice", "")` gui:15869) and surfaced in the status label
   (gui:15911-15912). Also verified: the run path shows an honest pre-run cell-count
   lower bound (gui:15889-15912). No action.

---

## 5. DESIGN DECISIONS FOR USER

Per-gap integration recommendations are made on feasibility+safety grounds; these are the
residual forks only the user should settle.

1. **Results co-location (Gap 3):** I recommend STANDALONE grid (safe integration into
   `results_tree` is infeasible — see Gap 3) but propose reducing the "works differently"
   feeling by hosting the multi-target grid in the **main Results tab via a
   Single/Multi-target toggle**. **Decide:** co-locate via the toggle (preferred, still
   additive), or leave the grid in the 4F sub-tab (least churn)? — Merging schemas into
   one tree is off the table on safety grounds; confirm you accept that.
2. **Double-click semantics (Gap 4):** recommendation = Multi-Target Model Detail dialog
   (per-target plots + save + export; zero recompute). True refinement in Model Dev is XL
   and reshapes single-Y machinery (infeasible safely). **Decide:** is the detail dialog
   sufficient, or is Model Dev refinement a hard requirement (→ separate Opus-led ticket)?
3. **Run entry convergence (Gap 1a):** keep the dedicated 4F Run button (progress/results
   still route to the shared tabs), or converge on ONE "Run Analysis" that detects >1
   target and dispatches the multi-target engine (adds a guarded branch to `_run_analysis`
   gui:25415)? Recommendation: keep the dedicated button for v1; converge later if desired.
4. **Prediction-tab support for multi-Y `.dasp` (item 5c):** ship with Save (recommended —
   a savable-but-unusable model is a half-feature and this is the clearest "works like
   everything else" win), or defer? Note 5c touches the shared single-Y predict loop
   (guard-tested).
5. **Pause/Resume for multi-target via the shared tab-5 buttons (Gap 2):** include the
   active-controller indirection (recommended, guard-tested) or keep multi-target
   Cancel-only? Recommendation: include.

---

## 6. Work items, waves, and serialization

**File-conflict rule:** `spectral_predict_gui_optimized.py` is one 60.6k-line file — all
GUI items MUST serialize (one worker at a time on that file). Backend items
(`multitarget_grid.py`, `multitarget_search.py`, `code_generator.py`, tests) can run in
parallel with each other and with GUI work on disjoint files.
**CRLF risk:** the repo has mixed line-ending history; after every Edit to the GUI file
run `git diff --stat` and reject phantom whitespace hunks (known Edit-tool drift issue).

| ID | Item | Verdict | Files | Size | Route | Notes |
|---|---|---|---|---|---|---|
| W1-1 | Progress INTEGRATE: route multi-target run to shared Analysis Progress tab (bar+ETA+best-model+log), + active-controller indirection for Pause/Resume/Stop + concurrent-start guard + single-Y progress-parity guard test | INTEGRATE | gui | M | GLM build, **Opus reviews the active-controller + run_logging concurrency diff** | serialize on gui; HIGHER-RISK (tab-5 widgets + 3 pause/stop handlers) |
| W1-2 | First-pass progress emission | INTEGRATE (backend) | multitarget_grid.py + test | S | GLM | parallel with W1-1 (different file) |
| W1-3 | Disk log via `setup_run_logger(label="multitarget")` + `log_event` | INTEGRATE | gui (+run_logging if named-logger needed) | S | GLM; Opus if run_logging changes | fold into W1-1's gui slot |
| W2-1 | Grid parity + co-location: run-row placement, themed listbox, header tooltips, Rank column, stable iids, export-disable-while-running; Single/Multi toggle in tab 6 (D-decision #1) | STANDALONE (parity) | gui | M | GLM | after W1-1 (same file) |
| W2-2 | Export Code (script+notebook) trigger + preprocessing/varsel mapping + new pinning test | INTEGRATE (backend) | gui + tests (+code_generator only if mapping gap) | M | GLM build, **Opus for the 3 caveats** | test file parallel; gui part serialized |
| W2-3 | Detail dialog + double-click binding (per-target plots, metrics, correlation, plot export) | STANDALONE (Model Dev infeasible) | gui | M | GLM with this spec | after W2-1; save/export buttons stub until W2-2/W3-1 |
| W3-1 | `refit_multitarget_final` helper + roundtrip test + Save-Model GUI trigger | INTEGRATE (backend) | multitarget_search.py/grid + tests + gui | L | **Opus** (refit correctness) | backend first (parallel), gui wiring last |
| W3-2 | Prediction-tab multi-Y fan-out via `prediction_columns` + single-Y predict guard test | INTEGRATE | gui + tests | M | **Opus** (shared predict loop — HIGHER-RISK) | user decision #4 |
| W3-3 | (Optional, user decision #2) Model Dev multi-target refinement | INTEGRATE (deferred — infeasible now) | gui, broad | XL | Opus, separate ticket | not in this pass |

**Wave order:** Wave 1 (W1-1‖W1-2, W1-3) → Wave 2 (W2-1 → W2-3 on gui; W2-2 test
work parallel) → Wave 3 (W3-1 backend ‖ W3-2 pending decision; gui wiring serialized).
Every wave ends with: `py_compile` on gui, targeted multitarget test files, the two
byte-identity guard suites, and a `git diff --stat` CRLF check.
