# T-31 Multi-Class SIMCA — Continuation Prompt (Phase D + merge-gate)

You are continuing the **T-31 multi-class SIMCA / class-modeling** build. **Phases A, B, and C are complete, each multi-family-gated, and pushed** on `feat/T31-multiclass-simca` (NOT merged). Your job is **Phase D (GUI + code export)**, then the **merge-readiness gate**. Start a **fresh Opus session** (prior sessions ran long). Read §0 first.

## 0. READ FIRST (mandatory, in order)
1. `docs/PROJECT_STATUS.md` — top block: Phases A+B+C done; Phase D next.
2. `docs/superpowers/specs/2026-07-04-T31-multiclass-class-modeling.md` — §7 (GUI + export requirements), §2 decision #2 (5th task type), §6 (persistence/predict schema).
3. `docs/plans/2026-07-04-T31-multiclass-simca-implementation.md` — **Phase D = D1 (radio+controls), D2 (results view + Wold plots), D3 (code export).**
4. `docs/SESSION_LOG.md` — the two `2026-07-04`/`2026-07-05` T-31 entries (esp. the **Decision D novelty fix** and the deferred fold-in list).
5. Skim the backend you're wiring a UI onto: `src/spectral_predict/simca.py` (`MultiClassClassModel`, `wold_diagnostic_plot_data`), `src/spectral_predict/search.py` (`run_multiclass_simca_search`), `src/spectral_predict/scoring.py` (`create_results_dataframe("multiclass_simca")`), `src/spectral_predict/model_io.py` (multiclass save/load + `predict_with_model` schema), `src/spectral_predict/model_registry.py` (`MULTICLASS_ENGINES`).

## 1. Environment / branch
- Branch **`feat/T31-multiclass-simca`**, tip **`cbab8b9`**, pushed. **`git branch --show-current` must equal it before ANY git op** (shared HEAD; re-check before reset/branch-move; isolated worktree for multi-step work).
- Python **`.venv312/Scripts/python.exe`** only. Windows, Git Bash. No new deps.
- Tests: `.venv312/Scripts/python.exe -m pytest tests/test_multiclass_search.py tests/test_simca.py -q` (green at Phase-C end). GUI import smoke: `.venv312/Scripts/python.exe -c "import spectral_predict_gui_optimized"`.

## 2. Execution model (proven A1→C fold-in — keep it)
**Opus (you) orchestrate; a fresh Opus subagent or GLM-5.2 worker implements; you review + commit per task.** Per task: write contract tests / a verification plan FIRST; delegate implementation (HALT-OR-BLOCK; leave uncommitted; trust `git diff` not the wrapper's "(no changes)"); review the diff yourself (esp. that the single-Y GUI paths are untouched); commit per task with explicit `git add <paths>` (NEVER `git add -A`) + push. Trailer:
```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: <your session url>
```
**GUI caveat:** Tkinter is hard to unit-test. Use the **`run` and `screenshot` skills** to actually launch the GUI and drive the new tab (select the radio, run on synthetic/real data, view the decision matrix + Wold plots) — do NOT trust import-smoke alone. `feedback_e2e_smoke_after_refactor`: run a real classification/regression through the GUI too, to confirm you didn't break the existing task paths. Watch for bare-`Tk`-in-worker-thread bugs (use queue + `root.after` poller, per prior GUI hardening).

## 3. Backend API you are wiring (all landed + tested)
- **Search:** `run_multiclass_simca_search(X, y, wavelengths=None, engines=None, preprocess_configs=None, preprocessing_methods=None, window_sizes=None, alpha=0.05, varsel_paths=None, variable_selection_n_select=None, n_components=0.99, min_class_samples=10, cv_splits=5, variable_penalty=0, gap_penalty=0, baseline_method=None, baseline_params=None, enable_smoothing=False, smoothing_window=17, smoothing_polyorder=2, progress_callback=None, controller=None)` → results DataFrame. `progress_callback` receives `{stage, message, current, total}`; `controller.check_and_wait()` for pause/stop (mirror the one-class dispatch).
- **Engines:** `model_registry.MULTICLASS_ENGINES = ['pca-simca','ocsvm','isolation-forest','lof','elliptic-envelope']`; `get_supported_models("multiclass_simca")` returns it.
- **n_components control:** **float in (0,1) = per-class variance fraction (default 0.99, novelty-oriented — THE recommended default);** int / `{class:int}` dict / `"per_class_cv"` (discrimination-oriented, under-detects novelty) also valid. Expose 0.99 as the GUI default; let advanced users set an int or per_class_cv.
- **varsel_paths:** subset of `["none","wold_modeling","wold_discriminating","wold_balanced","importance"]` (+ a precomputed bool mask is accepted by the model directly, if the GUI ever computes one). `variable_selection_n_select` = top-N.
- **Results schema:** `create_results_dataframe("multiclass_simca")` columns = common (`Task,Model,Params,Preprocess,Deriv,Window,Poly,LVs,n_vars,full_vars,SubsetTag,Imbalance`) + `NoveltyAUC,Efficiency,NoveltyRate,NoClassRate,AmbiguityRate,ExactSetRate,MeanSensitivity,MeanSpecificity,Alpha,MinClassN,n_classes,engine_family,varsel_path,unmodelable_classes,reason` + `top_vars,all_vars,CompositeScore,Rank`. Ranked by NoveltyAUC (lexicographic, NaN last).
- **Per-sample prediction (for the decision-matrix view):** `MultiClassClassModel.decision_matrix(X) -> (P (n,K) p-values, A (n,K) bool accept)`; `.predict(X) -> labels` (class-name / `"multiple"` / `"novel"`); `.classes_` = column order.
- **Wold diagnostic plots:** `wold_diagnostic_plot_data(X, y, n_components, scaling, wavelengths) -> {classes, variables, modeling_power (K,p), discriminating_power (K,p), modeling_power_agg (p,), discriminating_power_agg (p,)}`.
- **Persistence:** `model_io.save_model(model, None, metadata, path)` with `metadata` incl. `task_type="multiclass_simca"`, `class_names`, `engine_family`, `alpha`, `scaling`; `load_model` + `predict_with_model(m, X, validate_wavelengths=...)` returns `{p_values, decision_matrix, summary_label, accepted_classes, ...}`. The current build raises `NotImplementedError` on an unknown task_type (A8 gate).

## 4. Phase D tasks (spec §7; plan Phase D). **Re-grep every GUI symbol at edit time** — line numbers drift.
- **D1 — 5th task radio + controls.** Add a `value="multiclass_simca"` Radiobutton next to the existing task radios (**`spectral_predict_gui_optimized.py:6462-6465`**, after the One-Class one; text e.g. "Multi-Class Class Modeling"). Extend **`_on_task_type_changed` (:16773)** to show a multiclass control panel (engine multi-select from `MULTICLASS_ENGINES`; α spinbox default 0.05; n_components control default **0.99** with an advanced int/per_class_cv option; varsel-path picker; min_class_samples) and hide classification/one-class-only widgets. **Disambiguate from the existing one-class PCA-SIMCA entry point** (labels/help). Consider whether the Refine tab radios (:15170-15172) need it (likely defer). The GUI gates on `self.task_type.get()`; the search dispatch that calls `run_one_class_search` is near **:28044/:28080** — add a `multiclass_simca` branch there calling `run_multiclass_simca_search` on a worker thread (queue + `root.after` poller; forward `progress_callback`/`controller`).
- **D2 — results view (decision matrix + Wold plots).** Render the leaderboard (the multiclass columns) + a per-sample **decision matrix** (P/A → single-class / "multiple" / "novel" labels) + **Wold MPOW/DPOW diagnostic plots** (from `wold_diagnostic_plot_data`). Pattern the dispatch on the one-class detection results branch (re-grep — the plan's old `~29535` pointer is stale; find the one-class results renderer). Worker thread, not the Tk main thread. CSV export includes the decision matrix.
- **D3 — code export.** `code_generator` script + notebook templates gated on `task_type=="multiclass_simca"` reproducing the in-app decision matrix bit-identically (mirror the existing one-class export path). Test: exported script/notebook decision matrix == in-app.

## 5. Deferred fold-ins to CLOSE during Phase D (from the A/B/C gates)
- **`predict_with_uncertainty` has NO `multiclass_simca` branch** — it would return the dict where an ndarray is expected. Add a proper branch (return the §6 decision-matrix schema) or an explicit `NotImplementedError`. **Needed before the GUI predict path.**
- **`_cross_fit_null` uses `get_one_class_model` with no PCA wrapper for EllipticEnvelope** (simca.py) — EE folds fail when n_features>n_samples (real spectra are 2151-wide). Add a fold-failure warning / PCA-reduce for EE (surfaces if a user picks the EE engine).
- **Remaining task-type sites** (`model_config.py:158-181`, `cv_utils.py:275-340`, `code_generator.py`, `templates/validation.py`) — thread `multiclass_simca` (D3 covers code_generator/templates; check model_config/cv_utils are only hit by the model-registry path, else branch them). Extend the C1 fall-through audit if you touch them.
- **LOCO double-CV perf** (`_multiclass_loco_novelty_auc` runs `cross_validate` twice per config) — at real-data scale (757×2151, 10 classes) pass the already-computed OOF P into the helper to halve CV cost. Optional but recommended before a full real-data GUI run.

## 6. Phase-D gate + MERGE-READINESS
- **Phase-D gate:** GUI smoke — `import spectral_predict_gui_optimized` clean; launch via the `run`/`screenshot` skills; the multiclass tab renders; a search **completes on synthetic AND on the real data** (`Contaminated Samples Raw_ORAU Added.xlsx`, `Site` classes — hold out a site, confirm the decision-matrix view shows novels, aggregate metrics only, don't dump raw spectra); save→load→predict round-trips in-app; export reproduces the decision matrix. Confirm existing regression/classification/one-class GUI paths still work.
- **MERGE-READINESS (after D):** **full multi-family whole-diff pass** — Codex 5.5 (high) + ≥2 orthogonal families from {Kimi K2.7, MiniMax M3} (**rotate off whoever wrote the code; DeepSeek must NOT go via opencode-go — Kimi+MiniMax is the safe pair, as Phases B/C used**) — THEN the full **pr-review-toolkit** (code-reviewer, silent-failure-hunter, pr-test-analyzer, type-design-analyzer). Verify findings before agreeing (Phase-B/C panels each produced empirically-wrong HIGHs a probe refuted). **Merge gate = local diff-failure-set vs `origin/main`** (main red on cloud CI since 2025-10-27, so compare failure SETS; the PR must add ZERO new failures — run the full suite ex-GUI on HEAD and on `origin/main`, diff the sets). **Do NOT auto-merge — await explicit user greenlight.**

## 7. Guardrails / decisions LOCKED (do not re-litigate)
- **Never edit the single-Y GUI/search paths**; keep regression/classification/one_class byte-identical.
- **α global; IsolationForest `score_samples` NOT negated; `scaling="per_class"` default.**
- **Decision D (locked this session):** `n_components=0.99` (per-class variance fraction) is the novelty-oriented DEFAULT (held-out-site novelty 17%→100% on real data); `per_class_cv` is discrimination-oriented and under-detects novelty — expose but don't default to it. **novelty_rate is class-balanced.** **LOCO NoveltyAUC is an optimistic within-dataset proxy** (surface in GUI/report help). Composite ranking is single-objective on NoveltyAUC (Decision C; `·Efficiency^0.5` alternative deferred — user undecided).
- **min_class_samples LAYERED** (hard-block n<10; non-SIMCA n<20 unmodelable+warn; SIMCA warns at n<max(20,5·nc)). **Chemometric leakage standards** (per-spectrum ops outside folds not leakage; autoscale/calibration/varsel train-only inside folds).

## 8. Session protocol
Per `CLAUDE.md`: append non-obvious findings to `SESSION_LOG.md` as you go; update `PROJECT_STATUS.md` after each task/phase; commit + push docs. Do NOT ask the user to remind you.

Start by reading §0, then begin **Phase D / task D1** (re-grep the GUI symbols first; use the `run`/`screenshot` skills to verify the tab live, not just import-smoke).
