# Agent Handoff — T-17 Multi-Target "Full Search Parity"

> **You are a fresh agent picking this up cold. Brainstorming is DONE and the design below is USER-APPROVED. Do NOT re-brainstorm.** Your job: (1) write the spec, (2) get the user's review of it, (3) invoke `superpowers:writing-plans`, (4) implement per `superpowers:executing-plans` with review + merge-gate at the end.
>
> **First actions:** read `docs/PROJECT_STATUS.md` (top block) and `CLAUDE.md` (Session Protocol), then this file in full. Work only on branch `feat/T17-multitarget-regression` (PR #63 is open and GROWING — not merged). Use `.venv312` (Python 3.12).

---

## 1. Mission

The T-17 **Multi-Target regression** feature shipped its foundation (F1–F7, PR #63) but the GUI tab currently runs **each selected model ONCE, with default hyperparameters, on full spectra, at one preprocessing state — no search at all**. `_run_multitarget_search` literally builds `configs = [{"model_name": name, "params": {}} ...]` and calls `run_multitarget_search` on `self.X`. The result is uninformative ("one model per type, tells you almost nothing") and is "the case you already know will be bad."

**Goal: give the multi-target path full search parity with the single-Y search** — a grid over **preprocessing × variable-selection × model × hyperparameters**, evaluated in joint multi-Y CV, ranked by joint Q², shown as a real leaderboard. Grid engine ONLY (Bayesian/NSGA-II are 1-D).

## 2. USER-APPROVED design decisions (do not relitigate)

1. **New multi-target orchestrator that REUSES the existing building blocks.** Do NOT extend/edit `run_search` (it is ~3000 lines of 1-D assumptions; the single-Y byte-identity guardrail must be protected). Build the preprocessing × varsel × model × hyperparameter loop as a new orchestration layer that grows the existing `run_multitarget_search` (F2) upward, calling the multi-Y CV/metrics primitives per cell.
2. **Inherit the user's existing search configuration** — the SAME config the normal Run button assembles: preprocessing mode, enabled variable-selection methods, models to test, hyperparameter grids, CV strategy/folds/repeats. The Multi-Target tab keeps ONLY: target multi-select + model checklist + leaderboard. **Overrides only if a genuine need appears — start with ZERO override controls (YAGNI).**
3. **UVE and CARS: skip-with-notice.** F5 built multi-Y iPLS/SPA/GA-PLS/MWPLS, but UVE/CARS raise `NotImplementedError` on 2-D Y (discrimination methods, v1.1 deferral). If the inherited config enables them, SKIP them and report it in status/results ("UVE, CARS skipped — not supported for multi-target"); do not error out, do not hard-stop.
4. **Real leaderboard results** — one row per (preprocessing × varsel × model × hyperparameter) cell, columns: preprocessing, varsel method, #variables, key hyperparameters (e.g. PLS components), mode (JOINT/INDEPENDENT), joint Q², then per-target metrics (R²/RMSE/RPD/RER/CCCcv/Bias, raw units). Ranked by joint Q². (Horizontal scroll on this table was already fixed — see §7.)
5. **Grow on the PR #63 branch** — do NOT merge the foundation first. Re-run review + merge-gate at the end.
6. **Runtime stance: no cap, no "multi-target tier."** The grid does NOT grow with target count (targets don't multiply cells). Per-cell cost: JOINT models (PLS-2/RF/MLP/CatBoost/XGBoost) ≈ single-Y; INDEPENDENT models (Ridge/Lasso/EN/SVR/LightGBM/NeuralBoosted via `MultiOutputRegressor`) ≈ N_targets×; varsel (esp. GA-PLS) is the heavy part as it already is. The real work is **UX + parallelism**, not limiting: (a) run on a **worker thread** with a **Cancel** button (the current `_run_multitarget_search` runs synchronously on the Tk main thread and would freeze the UI — code-reviewer flagged this); (b) reuse the same **joblib parallelism** `run_search` uses for the cell loop; (c) **progress feedback** + an honest pre-run heads-up when the inherited config is heavy.

## 3. Codebase reuse map (verified this session)

- `src/spectral_predict/multitarget_search.py` — `run_multitarget_search` (F2, seed per-config evaluator; grid-only guard, joint-Q² ranking, `MultiTargetResult`/`MultiTargetSearchOutput`), `resolve_multitarget_strategy`, `build_multitarget_estimator`. Extend the orchestration ABOVE this.
- `src/spectral_predict/multi_y.py` — `multi_y_cv_pool` (per-fold JOINT Y-scaling + raw-unit pooling), `multi_y_metrics` (per-target RAW metrics + joint_q2), `FoldYScaler`, `extract_pls_multi_y`, `cap_components`.
- `src/spectral_predict/search.py` — `run_search` is the single-Y reference (DO NOT EDIT its single-Y path). Study its structure to mirror: preprocessing-config construction (~lines 1824–2032, uses `build_preprocessing_pipeline`), `get_model_grids(task_type, n_features, ...)` for hyperparameter grids, the main `for model_name, model_configs in model_grids.items()` loops (~2649/2933), `_run_single_config` (~4494) as the per-cell pattern, and how varsel subsets feed `_run_single_config` (subset_type, ~3099–4017). Find the joblib parallelism it uses and mirror it.
- `src/spectral_predict/variable_selection.py` + `ga_pls.py` — F5 multi-Y varsel: `ipls_forward`, `ipls_backward`, `mc_sipls`, `mwpls`, `spa_selection`, `ga_pls_selection` all accept 2-D Y. `_reject_multi_y` raises for `uve*`/`cars*` on 2-D Y (the skip-with-notice trigger). `_evaluate_interval_pls_multi` is the multi-Y interval scorer.
- GUI `spectral_predict_gui_optimized.py` — `_create_tab4f_multitarget` (~14523), `_run_multitarget_search` (~14746, REWORK this), `_populate_multitarget_results` (~14836), `_MULTITARGET_METRIC_KEYS`. The tab must read the same config vars the main Run assembles and dispatch on a worker thread. Find the normal search's worker-thread + progress-bar + cancel pattern and reuse it. `_export_multitarget_csv` already exists for CSV export.
- Real-data smoke pattern: `from spectral_predict.io import read_asd_dir; read_asd_dir('example')` → (49, 2151); join to `example/BoneCollagen.csv` on `File Number` (strip the space: "Spectrum 00001" → "Spectrum00001"); only one continuous target (`%Collagen`), so a 2-target smoke needs a constructed 2nd target (label it smoke-only).

## 4. HARD guardrails (do not violate)

1. **Single-Y byte-identity.** Do not edit `run_search`'s single-Y path. The new orchestrator is a SEPARATE path. Gold fixtures: `tests/gold_standards/varsel_single_y.npz` + `TestSingleYByteIdentityGold`, `tests/test_vip_formula.py`.
2. **Grid engine ONLY** for multi-target (Bayesian/NSGA-II are 1-D). Keep the existing 3-layer guard (GUI grey-out + force-to-grid + `run_multitarget_search` `ValueError`).
3. **Chemometrics conventions are NOT bugs:** per-spectrum SNV/SG derivatives/baseline on full data, and varsel-on-full-calibration, are community convention — do not "fix" them as leakage.
4. **NaN-sink pattern:** three sites were fixed this session (`run_multitarget_search` sort; `_evaluate_interval_pls_multi`; `ga_pls._fitness_function`). If you add any varsel/CV evaluator, grep the repo for the idiom `mean_nmse > 0.0 else 0.0` and guard finiteness (a non-finite per-target q2 must sink to the WORST sentinel, never a fake-perfect 0.0). See SESSION_LOG 2026-07-02.
5. No new deps without updating `pyproject.toml`. Never touch `.env`. Never commit data files. Commit ASAP after each working piece (parallel sessions share this repo). Re-check `git branch --show-current` before any git op; never force-push/reset on shared refs.
6. Chemometrics/methodology changes need user confirmation; pure engineering + bug fixes are fine.

## 5. Suggested build sequence (the plan will refine this)

- **A. Backend orchestrator** — new multi-target grid function: build preprocess_configs (reuse builder) → for each, apply preprocessing → run each enabled multi-Y varsel method (+ a "full spectra" baseline) to get feature subsets → for each model × hyperparameter config, evaluate the cell via `multi_y_cv_pool`+`multi_y_metrics` → collect `MultiTargetResult` rows with full metadata → rank by joint_q2. Parallelize the cell loop (joblib, mirror `run_search`). UVE/CARS skipped-with-notice.
- **B. GUI wiring** — rework `_run_multitarget_search` to read the inherited config, dispatch the orchestrator on a worker thread with progress + Cancel, and populate the leaderboard (preprocessing/varsel/#vars/hyperparams/mode/joint-Q²/per-target). Pre-run heads-up for heavy configs.
- **C. Polish** — skip-with-notice surfacing, honest progress/heads-up, CSV export columns.
- **D. Tests + review + gate.**

## 6. Testing strategy

- **Unit:** orchestrator loop (preprocessing applied per cell; varsel applied and reduces #vars; cells produced for every preprocessing×varsel×model×hp combo); UVE/CARS skip-with-notice (enabled → skipped + reported, not raised); ranking by joint_q2; parallel determinism (same results serial vs parallel). Reuse the NaN-sink discriminating-pin style.
- **Integration:** end-to-end multi-target grid on real spectra (read_asd_dir('example') + a 2-target Y) reproducing an expected leaderboard; a JOINT cell, an INDEPENDENT cell, a varsel cell (n_vars < full); save→reload→predict round-trip (F7).
- **GUI:** worker-thread run doesn't block the main loop; Cancel stops it; the tab passes the inherited config (spy/mock the orchestrator and assert it received the config the main Run would).
- **Regression:** single-Y suites stay green and byte-identical (`run_search` untouched). Full suite + merge-gate diff-failure-set vs `origin/main` at the end (see PROJECT_STATUS "MERGE GATE" note for the protocol: Windows Py3.12, `--ignore=tests/gui`, expect the 5 known pre-existing failures, zero new).

## 7. Review checkpoints (MANDATORY — this project reviews at every gate)

This codebase is reviewed heavily and the user expects it. Mirror how the F1–F7 foundation was built: **per-phase review gate → fold NEEDS-CHANGES → proceed**, then a **whole-diff cross-family pass + pr-review-toolkit before merge**, then the **merge gate**. Give every reviewer the by-design list (single-Y byte-identity via additive branches; grid-only; chemometrics conventions are not leakage; UVE/CARS skip-with-notice) so they don't waste findings.

- **Per-phase gate (after each of A / B / C):** run a **Codex** review (`codex-reviewer` agent, or the codex-review skill) on that phase's diff — Codex earns its slot on exactly this kind of cross-file orchestration/dispatcher work. Fold NEEDS-CHANGES before moving on. Add a second orthogonal reviewer (Kimi/GLM via `opencode-call`, full repo access) when the phase is large or touches CV/varsel math.
- **Whole-diff before merge:** cross-family pass — **Codex + Kimi + GLM** with FULL repo access. For multi-file code review, dispatch **parallel `opencode-call` agents** (NOT the peer-review skill, which is for self-contained targets). Then run the **pr-review-toolkit** (Claude-family specialists) — `silent-failure-hunter` is the single highest-value one here given the NaN-sink history; also `code-reviewer` + `pr-test-analyzer`. The foundation's toolkit pass caught a HIGH the four cross-family passes missed, so do BOTH layers — they are orthogonal.
- **Merge gate:** local diff-failure-set vs `origin/main` (Windows Py3.12 `.venv312`, `--ignore=tests/gui`; expect the 5 known pre-existing failures — `test_cv_strategy` nameerror, `test_export_code` ×2, `test_t19_class_weight` ×2 — and require ZERO new). Do NOT auto-merge — wait for explicit user greenlight.
- **Model routing (current, verified 2026-07-02) — dispatch by ALIAS, never a hardcoded version:** `glm` → GLM 5.2 (default); use **`kimi27`** (Kimi K2.7-Code) for coding/review tasks, NOT `kimi` (K2.6, general). GLM bills the z.ai subscription; opencode-go covers Kimi/DeepSeek/etc. When the user says "codex", use Codex — never substitute another model. See memory `feedback_model_version_aliases`, `feedback_codex`, `feedback_review_method_signal`, `feedback_glm_routing`, `feedback_check_ci_before_merge`.

## 8. Already done this session (don't redo)

- **GUI bug fixes committed on the branch (`52c814b`):** (a) Multi-Target results table horizontal scroll — the tab's scroll-canvas inner frame is now pinned to viewport width and the results tree has both x/y scrollbars; (b) **pre-existing** Refine-tab `_on_refine_cv_strategy_changed` crash (`pack(before=<unpacked>)` → `TclError`) — now forgets toggleable widgets and re-packs anchored to the always-packed hint. The Refine fix is unrelated to T-17 and could be split to a separate fix against `main` if PR-cleanliness matters.
- **Review + gate on the foundation:** F1–F7 passed 4 cross-family review passes (Codex/Kimi K2.6/GLM 5.1/GLM 5.2) + full pr-review-toolkit (5 Claude-family specialists). Three MEDIUM NaN-sinks fixed + pinned. Merge gate: 0 new failures vs `origin/main`. See PROJECT_STATUS + SESSION_LOG (2026-07-02 entries).
- **Deferred review items (non-blocking, in PR #63 comments):** type enforcement-altitude (`MultiTargetStrategy.__post_init__`, frozen-dict sharing, `MultiTargetResult`/`FoldYScaler` guards); `inter_target_correlation` constant-target nan advisory; interval-subset Returns multi-Y docstring note. Fold any that are cheap while you're in the relevant files.

## 9. Process

Follow the superpowers flow: write the spec to `docs/superpowers/specs/2026-07-02-multitarget-full-search-parity-design.md`, self-review (placeholders/consistency/scope/ambiguity), **ask the user to review it**, then invoke `superpowers:writing-plans` for the implementation plan, then `superpowers:executing-plans`. Update `docs/PROJECT_STATUS.md` + `docs/SESSION_LOG.md` and commit per the Session Protocol. Do NOT auto-merge PR #63 — wait for explicit user greenlight.
