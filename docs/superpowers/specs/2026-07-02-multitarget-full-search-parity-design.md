# Spec — T-17 Multi-Target "Full Search Parity"

**Status:** DRAFT v3 — folds in three v2 cross-family re-evals (Codex GPT-5.6 high, MiMo 2.5, GLM 5.2), all repo-grounded, plus a verified 18-method varsel inventory. Decisions D1/D2/D3 + the HP-inheritance fork resolved. Awaiting final user OK before `writing-plans`.
**Branch:** `feat/T17-multitarget-regression` (PR #63, open, growing — do NOT merge foundation first).
**Author:** fresh agent, 2026-07-02. Design USER-APPROVED (handoff `docs/CONTINUATION_PROMPT_2026-07-02_T17-full-parity.md`).

---

## 1. Problem

The shipped Multi-Target tab runs each selected model **once, default hyperparameters, full
spectra, one preprocessing state — no search** (`spectral_predict_gui_optimized.py:14759` builds
`params={}` and calls F2 `run_multitarget_search` on `self.X`), **synchronously on the Tk main
thread**.

## 2. Goal

Full parity with the single-Y search: a grid over **preprocessing × variable-selection × model ×
hyperparameters**, joint multi-Y CV, ranked by joint Q², shown as a real leaderboard. Grid engine
ONLY. The Multi-Target tab keeps only target multi-select + model checklist + leaderboard, and
**inherits every other search setting** from the config the normal Run button assembles — including
the **full per-model hyperparameter grids** (HP decision (a), §6).

## 3. Non-goals / v1.1 deferrals

- Bayesian / NSGA-II multi-target (grid-only guard stays).
- Any change to `run_search`'s single-Y path (guardrail #1 — byte-identity).
- Override controls on the Multi-Target tab (YAGNI).
- Cell-loop parallelism (D1 — sequential v1; measured follow-up).
- UVE/CARS varsel family (8 methods) on 2-D Y → **skip-with-notice** (user-approved v1.1 deferral, handoff §2.3).
- `ipls` (legacy) + `vcpa-iriv` multi-Y support → skip for v1 (genuinely broken, not cheaply fixable — §6 full-parity boundary).

---

## 4. Approach

A **new orchestration layer above the F2 seed evaluator**, reusing existing building blocks;
`run_search` untouched.

```
for preprocess_config in preprocess_configs:            # basic SNV/SG/deriv/baseline/smoothing (± autoscale) grid
    pipe = build_preprocessing_pipeline(cfg)            # reuse existing pipeline builder (safe primitive)
    X_pp = pipe.fit_transform(X_full)                   # full-X, chemometrics convention (guardrail #3)
    X_pp, wl_pp = apply wavelength restriction + edge-mask exactly as run_search (skip edge-mask if restricted)
    subsets = [full] + varsel_subsets(X_pp, Y, wl_pp)   # cached per (preprocess, method); two adapters; skip-with-notice
    for subset in subsets:
        X_sub = X_pp[:, subset.indices]
        for (model_name, params) in dedup(model_hp_configs):   # get_model_grids(enabled + FULL override kwargs)
            result = _evaluate_multitarget_cell(X_sub, Y, model_name, params, ...)  # per-subset cap; exception->NaN
rank all results by joint_q2 (NaN-safe, F2 sort key multitarget_search.py:641)
```

### 4.1 Reuse map (verified; corrected per review)

| Piece | Source | Role |
|---|---|---|
| `resolve_multitarget_strategy`, `build_multitarget_estimator` (extended, §6), `run_multitarget_search` (refactored to share the cell helper), NaN-safe sort `:641`, grid guard `:562` | `multitarget_search.py` | strategy/estimator + F2 seed evaluator |
| `multi_y_cv_pool` (SERIAL fold loop `:221`), `multi_y_metrics`, `inter_target_correlation`, `cap_components` `:474`, `aggregate_importance` | `multi_y.py` | per-cell CV + RAW metrics + multi-Y importance aggregation |
| `build_preprocessing_pipeline` | `preprocess.py:286` | transform for one preprocess config (reuse; pure builder) |
| `get_model_grids(task_type, n_features, tier, enabled_models, max_n_components, max_iter, **override_kwargs)` → `{name: [(estimator, params_dict), …]}` | `models.py:542` | orchestrator uses the **`params_dict`** list; forwards the user's FULL override kwargs (§5.1.5) |
| interval varsel (subset-returning): `ipls_forward/backward`, `mc_sipls`, `mwpls` | `variable_selection.py` | return `list[{'indices','tag',…}]`; take `wavelengths=` |
| importance varsel: `spa_selection` (`:466` array), `ga_pls_selection` (`ga_pls.py:574` freq array, linear only), `get_feature_importances` (`models.py:1798`, multi-Y via `aggregate_importance` `:1836`) | | return importance arrays → top-N via `variable_counts` |
| `_reject_multi_y` (`variable_selection.py:34`) + new guards on the 3 silent-flatteners | | skip-with-notice trigger; fail-loud hardening |
| `SearchController` (`search_controller.py:6`) | | cooperative Cancel |
| single-Y config assembly + worker/progress/cancel | GUI: `_run_analysis` `:24172`, `_run_analysis_thread` `:26146` (override locals ~`26414-29226`), `run_search(` `:29135`, `_progress_callback_impl` `:29513` | config vars to inherit + thread/progress/cancel pattern to mirror |

### 4.2 What `run_search` actually does (corrects handoff §2.6b)

Cell loop is **sequential** (`search.py:2933`/`2949`, `check_and_wait()` `:2935`/`:2951`); joblib
`Parallel` only over CV folds inside `_run_single_config` (`:4751`). `multi_y_cv_pool` is serial
(`multi_y.py:221`). Heavy models parallelize across cores via their own `n_jobs=-1`
(`models.py:991,1246,1291,1412`) inside each cell — the model of parallelism v1 adopts (D1).

---

## 5. Component design

### 5.1 Backend orchestrator — `multitarget_search.py`

**New `run_multitarget_grid_search`** (F2 `run_multitarget_search` stays as the single-preprocessing
seed API). Signature carries the inherited config explicitly:

```python
def run_multitarget_grid_search(
    X, Y, *,
    model_names, target_names, wavelengths,
    preprocessing_methods, autoscale,                    # basic preproc grid inputs
    baseline_method=None, baseline_params=None,
    smoothing=False, smoothing_window=None, smoothing_polyorder=None,
    interference_to_add=None, wavelength_restriction=None,
    variable_selection_methods, variable_counts=None,
    ipls_subset_limit="Top 10",                          # STRING ("Top 5"/"Top 10"/"Top 20"/"All"), NOT int
    tier, model_grid_overrides=None,                     # full get_model_grids kwargs (5.1.5)
    max_n_components=10, max_iter=500,
    cv="kfold", n_folds=5, n_repeats=5, random_state=42,
    optimization_method="grid",                          # asserted == 'grid'
    controller=None, progress_callback=None, n_jobs=-1,
) -> MultiTargetSearchOutput
```

**Steps:**

1. **Assert grid-only** — raise `ValueError` if `optimization_method != "grid"` (message family of
   F2 `multitarget_search.py:562`).

2. **Enumerate preprocess configs** — new `build_multitarget_preprocess_configs(...)` mirroring
   `run_search`'s basic enumeration (`search.py:2288-2616`): raw/SNV/SG1-4/deriv variants, then
   **baseline doubling (`:2565`), smoothing doubling (`:2582`), autoscale doubling (`:2603`)**.
   `autoscale` flows INTO this builder (not directly into `build_preprocessing_pipeline`). Reuses
   `build_preprocessing_pipeline`; does not touch `run_search`. **Test-pinned** against the single-Y
   list (D3). 1-D discovery modes enabled in the inherited config are ignored here (D2).

3. **Per preprocess config:** `X_pp = pipe.fit_transform(X_full)`; then apply the **same wavelength
   restriction + derivative edge-mask** as `run_search` (`:2777-2883`) to `X_pp` and `wl_pp` —
   preserving the **"skip edge-mask when a restriction is active"** condition. Full-X preprocessing
   is chemometrics convention (guardrail #3); autoscale is `StandardScaler` **column** scaling
   (`preprocess.py:565`) applied before CV — matches single-Y (`search.py:2767`), documented as
   distinct from per-spectrum SNV/SG (not conflated). GLM confirmed this does NOT interact with the
   per-fold JOINT `FoldYScaler` (different matrix: X vs Y) — no leakage/double-scaling.

4. **Build subsets** — always `{'indices': all, 'tag':'full', 'method':'full'}`, plus per enabled
   varsel method **routed by the verified inventory below**. Results are **cached per
   `(preprocess_config, method)`** (importances/subsets are model/hp-independent — mirrors the
   single-Y cache `search.py:3304`) to avoid re-running expensive methods (GA/SPA) per cell.

   **Verified varsel routing (all 18 GUI method strings):**

   | Adapter | Methods | Evidence / handling |
   |---|---|---|
   | **subset-adapter** | `ipls_forward`, `ipls_backward`, `mc_sipls`, `mwpls` | multi-Y-safe via `_prep_varsel_y` → `_evaluate_interval_pls_multi` (`variable_selection.py:1738`); return `{'indices','tag'}`; pass `wavelengths=wl_pp`; truncate by `ipls_subset_limit` string |
   | **importance-adapter** | `spa`, `fipls_spa`, `ga` (LINEAR models only), `importance` | importance array → top-N via `variable_counts` filtered to `n < n_features_sub` (LOW-1). **`spa`** (⚠ verify 2-D SPA-CV criterion with a test — load-bearing, see below); **`fipls_spa`** = de-ravel it (remove the pre-`ravel()` at `variable_selection.py:1156`; its internals already call the multi-safe `ipls_forward` + `spa_selection`) — **contingent on the same `spa` verification**; `ga` linear-only (`ga_pls` joint-PLS fitness) — `ga`+tree flattens → skip; `importance` is model-specific → per-`(preprocess, model)` reference fit on full `X_pp` then `aggregate_importance` (`models.py:1836`) |
   | **skip-with-notice** | `uve`, `uve_spa`, `cars`, `cars-tree`, `uve_cars`, `uve_cars_tree`, `uve_cars_spa`, `fipls_cars` (8 clean rejecters — **user-approved v1.1 deferral**, handoff §2.3); `ga` when tree models; `apply_uve_prefilter` | appended to `skipped`, never called, never raised |
   | **skip-with-notice (broken, need real multi-Y work — OPEN, see §6 "full-parity boundary")** | `ipls` (legacy importance path), `vcpa-iriv` | genuinely single-Y-only (`ipls` flattens at `variable_selection.py:630`; `vcpa-iriv` yields degenerate `inf` RMSECV on 2-D Y at `wavelength_selection.py:497`); not cheaply fixable |

   **`spa` verification is now load-bearing** for full parity: it gates BOTH `spa` and `fipls_spa`.
   The build sequence must confirm `spa_selection` produces sane non-degenerate importances on 2-D Y
   (its `_prep_varsel_y` preserves 2-D, but the internal `_evaluate_spa_seed` R² criterion is
   unverified). If it verifies safe → both `spa` and de-raveled `fipls_spa` are IN; if not → both
   demote to skip-with-notice.

   **Defensive hardening (fail-loud):** add a `_reject_multi_y` guard to the two genuinely-broken
   flatteners `ipls` (`variable_selection.py:630`) and `vcpa_iriv` (`wavelength_selection.py:352`)
   so a scripted 2-D-Y caller raises instead of producing garbage — belt-and-braces with the skip
   list (guardrail #4 spirit). Single-Y behavior unchanged (guard fires only on 2-D Y). (`fipls_spa`
   is NOT guarded — it is being de-raveled into a supported method instead.)

5. **Build model×hp configs** — `get_model_grids(task_type='regression',
   n_features=X_full.shape[1], tier=tier, enabled_models=model_names,
   max_n_components=min(min_fold_train-1, max_n_components), max_iter=max_iter,
   **(model_grid_overrides or {}))`. **`model_grid_overrides` must carry the EXACT
   `get_model_grids` kwargs** the GUI forwards to `run_search`, including the non-`*_list` names
   (`learning_rates`, `neuralboosted_hidden_sizes`, `xgb_learning_rates`, `xgb_max_depths`,
   `xgb_subsample`, `elasticnet_l1_ratios`, `catboost_learning_rates`, `catboost_depths`,
   `svr_kernels`, …) — built by the GUI mirroring its own `run_search` call (`gui:29159-29227`), NOT
   a lossy re-mapping. Keep each model's **`params_dict`** list; **dedup** identical cells (now that
   builders consume the FULL param set — HP decision (a) — dedup is mostly a no-op but retained as a
   safety net) with a regression pin that **`dedup_keyset == consumed_keyset`** so no real config is
   lost from the ranking (GLM MEDIUM-1). `max_n_components` is pre-clamped by `min_fold_train-1`
   per the `models.py:583` caller contract (LOW-2).

6. **Evaluate each cell** via shared `_evaluate_multitarget_cell(X_sub, Y, model_name, params,
   splitter, min_fold_train, n_features_sub, target_names)`:
   - `resolve_multitarget_strategy` → `build_multitarget_estimator(strategy, params, min_fold_train,
     n_features_sub = X_sub.shape[1])` — **per-subset** feature count so `cap_components` can't
     over-request (single most likely refactor bug). Degenerate/empty subsets fall through to the
     NaN sink (below), not a crash.
   - `multi_y_cv_pool(scale_y=strategy.scale_y)` → `multi_y_metrics`.
   - **Exception path:** wrap the cell; on ANY failure return `MultiTargetResult(..., joint_q2=np.nan,
     error=…)` — **never a finite 0.0** — so the NaN-safe sort sinks it. `multi_y_metrics` may emit
     non-finite per-target values directly; the sort handles NaN `joint_q2`.
   - **F2 refactor:** `run_multitarget_search` is refactored to call this helper — this **changes F2's
     error behavior from exception→abort to exception→NaN-row**; existing F2 tests that assert
     propagation must be updated (L1).

7. **Rank** all cells NaN-safe (F2 sort key `multitarget_search.py:641`).

8. **Progress + cancel:** sequential loop; `controller.check_and_wait()` before each cell; emit a
   progress info dict shaped for the GUI consumer (§5.2).

**Data model** (dataclasses, not frozen; append-only):

```python
# MultiTargetResult gains (defaults):
preprocessing: str = "raw";  varsel_method: str = "full";  varsel_tag: str = "full"
n_variables: int | None = None;  error: str | None = None
# params holds the EFFECTIVE (post-cap) hyperparameters.
# MultiTargetSearchOutput gains:
skipped: list[str] = field(default_factory=list)   # default_factory REQUIRED
```

### 5.2 GUI wiring — `spectral_predict_gui_optimized.py`

`_run_multitarget_search` (14759) reworked to:
1. **Read the inherited config** — same state `_run_analysis_thread` reads: `preprocessing_methods`,
   `use_autoscale`, baseline/smoothing/interference, `selected_varsel_methods` (`:28125`),
   `variable_counts`, `ipls_subset_limit` (string), `model_tier`, CV vars, the **full per-model HP
   override kwargs** (built exactly as the `run_search` call does at `:29159-29227`), and any active
   wavelength restriction. Models from the tab's own `multitarget_model_vars`.
2. **Sample filtering** — keep existing active/excluded/held-out mask (14802-14818).
3. **Worker thread** — `threading.Thread(target=self._run_multitarget_search_thread, daemon=True)`;
   a **separate `self._multitarget_controller = SearchController()`** (never reuse
   `self.search_controller`); **Cancel** button (in-flight only) → `.stop()`.
4. **Progress marshalling** — convert each "best so far" `MultiTargetResult` into the **dict shape
   `_progress_callback_impl` expects** (`:29558`: `RMSEcv`/`R2cv`/`Preprocess`/…) or a dedicated
   multi-target renderer; all Tk writes via `root.after(0, …)`; worker exceptions marshalled to a
   messagebox; explicit completion/error/cancel state ownership.
5. **Pre-run heads-up** — honest cell-count estimate `n_preprocess × (1 + Σ varsel_subset_counts) ×
   Σ model_n_configs` before starting. No cap.
6. **Leaderboard** (14849) leading columns: **preprocessing, varsel method, #variables, key
   hyperparameters (per-model renderer), mode, joint Q²**, then per-target block. Scroll fixed
   (`52c814b`).
7. **Skip-with-notice** in status + footnote: e.g. *"UVE, CARS, iPLS(legacy), VCPA skipped — not
   multi-target-safe. Smart/TPE/Exhaustive-GA preprocessing is 1-D-only; ran the basic preprocessing
   grid instead."* (fiPLS-SPA is supported once `spa` verifies 2-D-safe.)
8. **CSV export** (14882) gains the new leading columns.

---

## 6. Decisions — RESOLVED

- **D1 Parallelism → SEQUENTIAL v1** (user). Sequential cell loop + `SearchController` checkpoints +
  progress; heavy models keep in-cell `n_jobs=-1`. Cell-loop parallelism deferred (measured follow-up).
- **D2 1-D preprocessing-discovery → SKIP-WITH-NOTICE + basic-grid fallback.** Notice says "ran the
  basic grid instead."
- **D3 Preprocess enumerator → OWN**, `build_preprocessing_pipeline`-reusing, **test-pinned** against
  full single-Y semantics (baseline/smoothing/autoscale doubling, restriction, edge-mask).
- **HP inheritance → (a) EXTEND BUILDERS TO FULL PARITY** (user, "full parity").
  `build_multitarget_estimator` consumes the full inherited hyperparameter set per model (PLS
  `max_iter`/`tol`; RF `bootstrap`/`min_samples_split`/`max_leaf_nodes`/`min_impurity_decrease`; MLP
  `activation`/`solver`/`batch_size`/`learning_rate`/`momentum`; CatBoost `border_count`/
  `random_strength`/`bootstrap_type`/`bagging_temperature`; XGBoost full; INDEPENDENT bases likewise).
  Dedup retained as a no-op safety net + `dedup_keyset==consumed_keyset` pin.

- **Full-parity boundary (varsel) — RESOLVED.** Applying "full parity" to the varsel axis: bring in
  every method that is multi-Y-safe or *cheaply* made safe → the 4 interval methods, `ga`-linear,
  `importance`, and `spa` + `fipls_spa` (contingent on the `spa` 2-D verification). The boundary —
  what "full parity" does NOT reach in v1 — is (i) **UVE/CARS**, an explicit user-approved v1.1
  deferral, and (ii) **`ipls`-legacy + `vcpa-iriv`**, genuinely single-Y-only and needing real
  multi-Y reimplementation. **User decision 2026-07-02: (ii) stays skip-with-notice for v1** (least-
  used methods, non-trivial new work; confirmed-safe set already covers intervals + SPA + GA +
  importance).

---

## 7. Guardrails honored

1. **Single-Y byte-identity** — separate path; `run_search` untouched; only pure
   `build_preprocessing_pipeline` shared. `_reject_multi_y` hardening fires only on 2-D Y (single-Y
   unaffected). `MultiTargetResult`/`…Output` fields append-only. Gold fixtures stay green.
2. **Grid engine ONLY** — GUI grey-out + force-to-grid + new orchestrator asserts grid-only.
3. **Chemometrics conventions NOT bugs** — full-X per-spectrum ops + varsel-on-full-calibration are
   convention; autoscale-before-CV matches single-Y, documented as column scaling.
4. **NaN-sink discipline** — shared cell helper returns NaN on failure (never finite 0.0); NaN-safe
   sort. `mean_nmse>0 else 0.0` idiom confirmed only at `variable_selection.py:80` + `ga_pls.py:223`,
   both finiteness-guarded; no new finite-sentinel path.
5. No new deps; never touch `.env`; never commit data files; commit ASAP; re-check branch first.
6. Methodology changes need user confirmation — D2 + HP(a) accepted.

---

## 8. Testing strategy

- **Unit (orchestrator):** preprocessing per cell; **subset-adapter AND importance-adapter** produce
  subsets with `n_variables < full`; **variable_counts filtered to `< n_features_sub`**; all 3
  varsel families exercised; **skip-with-notice** for every rejecter + the 3 silent-flatteners +
  `ga`-tree + `apply_uve_prefilter` (reported, not raised); **`_reject_multi_y` hardening** raises on
  2-D Y for `ipls`/`fipls_spa`/`vcpa-iriv` while single-Y stays byte-identical; **cell-exception →
  NaN row, sunk not aborted**; **per-subset PLS cap** (subset < `min_fold_train-1` must not raise);
  **HP full-consumption** (a tuned `rf_bootstrap`/`mlp_activation` changes the fitted estimator) +
  **`dedup_keyset == consumed_keyset`** per model; grid-only assertion; **varsel cache** hit per
  `(preprocess, method)`.
- **`spa` 2-D verification pin:** confirm `spa_selection` on 2-D Y produces sane, non-degenerate
  importances (the one unconfirmed importance-adapter method) — else demote `spa` to skip-with-notice.
- **Enumerator pin (D3):** `build_multitarget_preprocess_configs` matches single-Y basic enumeration.
- **Integration:** end-to-end grid on real ASD spectra (`read_asd_dir('example')` → (49,2151) joined
  to `BoneCollagen.csv` on `File Number` + a constructed 2nd target, smoke-only); leaderboard with a
  JOINT, an INDEPENDENT, and a varsel cell; save→reload→predict round-trip (F7).
- **GUI:** worker doesn't block Tk; **separate controller** Cancel stops it; the tab passes the
  **inherited** config incl. full HP override kwargs (spy/mock); progress callback receives a
  GUI-shaped dict (no `MultiTargetResult` type leak).
- **Regression:** single-Y suites green + byte-identical; **F2 tests updated** for the exception→NaN
  behavior change; full suite + merge-gate at the end.

## 9. Review cadence (per [[feedback_handoffs_include_review_cadence]])

- **Per phase (A backend / B GUI / C polish):** Codex review of that phase's diff.
- **Whole-diff cross-family** at integration: `glm` + `kimi27` (by alias).
- **Full pr-review-toolkit** before merge-readiness.
- **Merge gate:** local diff-failure-set, Windows Py3.12 `.venv312`, `--ignore=tests/gui`, expect the
  5 known pre-existing failures, assert **zero new** vs `origin/main` ([[feedback_check_ci_before_merge]]).
  Do NOT auto-merge PR #63 — explicit greenlight only.
- **Implementation delegation:** coding subtasks may route to GLM 5.2 (write-mode `opencode-call`) in
  place of a Sonnet subagent ([[feedback_glm52_substitutes_sonnet_impl]]); Opus keeps orchestration
  + chemometrics-gated judgment.

## 10. Build sequence (plan will refine)

- **A. Backend** — extend `build_multitarget_estimator` to full HP consumption; `_reject_multi_y`
  hardening on the 3 flatteners; `_evaluate_multitarget_cell` (+ F2 refactor, exception→NaN);
  `build_multitarget_preprocess_configs` (+ pin); the two varsel adapters + cache + skip logic +
  `importance` reference-fit + `spa` 2-D pin; `run_multitarget_grid_search` (grid assert, full
  override forwarding, per-subset cap, dedup + keyset pin, ranking, sequential loop + progress/cancel);
  `MultiTargetResult`/`…Output` fields.
- **B. GUI** — rework `_run_multitarget_search` for full config inheritance (incl. HP override
  kwargs) + worker thread + separate SearchController + dict-shaped progress + Cancel + heads-up;
  leaderboard columns + per-model HP renderer; skip-notice surfacing.
- **C. Polish** — CSV columns; honest progress/heads-up copy.
- **D. Tests + per-phase reviews + cross-family + toolkit + merge-gate.**

## 11. Review fold-in log

v1: Codex + Kimi K2.7. v2 re-eval: Codex GPT-5.6 high (NOT-sound → varsel taxonomy + HP kwargs),
MiMo 2.5 (GO-with-conditions → C1/C2 silent-flatten, H1/H2 contracts, M1 caching), GLM 5.2
(SOUND-with-fixes → MEDIUM-1 HP consumption/dedup-lockstep; all four multi_y math focus points
verified sound). Verified 18-method varsel inventory (Explore) → 4 subset-safe, 3 importance-safe
(spa⚠/ga-linear/importance-reffit), 11 skip. Decisions: D1 sequential, D2 skip+basic, D3 own+pin,
HP(a) full-consumption. All convergent findings folded above.
