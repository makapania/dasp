# T-31 Multi-Class SIMCA — Continuation Prompt (Phase C onward)

You are continuing the **T-31 multi-class SIMCA / class-modeling** build. Phases **A and B are complete, reviewed by multi-family gates, and pushed**; **Phase C is in progress — C1 is done**. Your job is **C2**, then **C3**, then the Phase-C gate + real-data e2e, then **Phase D**. One task at a time, TDD, per-task commit, multi-family gate at each phase boundary.

Start a **fresh Opus session** for this (the prior session ran long). Read the files in §0 first.

## 0. READ FIRST (mandatory, in this order)
1. `docs/PROJECT_STATUS.md` — top block is the T-31 active direction (Phase A + B done + gated; **C1 done**; C2 next).
2. `docs/superpowers/specs/2026-07-04-T31-multiclass-class-modeling.md` — design spec (source of truth). §5.4 (CV design), §7 (search + metrics + ranking), §8 (edge cases).
3. `docs/plans/2026-07-04-T31-multiclass-simca-implementation.md` — the TDD task plan. **Phase C = C1 (done), C2, C3.**
4. `docs/SESSION_LOG.md` — the two `2026-07-04` entries (Phase A findings + Phase B findings/gate). Grep, don't read whole.
5. Skim `src/spectral_predict/simca.py` (the module: `MultiClassClassModel` + `cross_validate` + `evaluate_novelty` + `multiclass_simca_metrics`/`wilson_ci`/`novelty_tradeoff_auc` + Wold varsel), and the C1 additions in `src/spectral_predict/scoring.py` + `src/spectral_predict/model_registry.py`, and `tests/test_multiclass_search.py`.

## 1. Environment / branch
- Branch: **`feat/T31-multiclass-simca`** (off `origin/main`), pushed, tip `831fa6e`. **`git branch --show-current` must equal `feat/T31-multiclass-simca` before ANY git op** (a parallel session shares this repo's HEAD; re-check before any reset/branch-move; use an isolated worktree for multi-step work).
- Python: **`.venv312/Scripts/python.exe`** only. Windows, Git Bash available. No new deps without `pyproject.toml`.
- Run: `.venv312/Scripts/python.exe -m pytest tests/test_simca.py tests/test_multiclass_search.py -q` (56 + 5 = 61 pass at C1 end).

## 2. Execution model (proven across A1–B3 + C1 — keep it)
**Opus (you) orchestrate; GLM-5.2 write-mode workers implement; you review + commit per task.** Per task:
1. **Write the contract tests yourself** (Opus), run them, **confirm they FAIL (red)** — never tautological. **Before pinning a numeric threshold, verify it's empirically achievable with a throwaway probe** (this caught the DPOW-RMS-vs-std issue and the novelty-guard tolerance in Phase B).
2. **Delegate implementation** to a GLM-5.2 worker via the `opencode-call` agent (alias `glm`, WRITE mode, **HALT-OR-BLOCK**: "if a test looks wrong, STOP and report — do NOT edit tests or guess"). Tell it the exact file(s), run the suite, leave uncommitted. The wrapper often prints a false "(no changes detected)" — trust `git diff`. (Trivial glue you may implement directly, as B2/C1 were.)
3. **Review the diff yourself** — leakage (anything fit on data must be train-only inside folds), correctness, no regression on adjacent suites.
4. **Commit per task** with explicit `git add <paths>` (NEVER `git add -A` — many untracked `tools/` scratch files) + push. Trailer:
   ```
   Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
   Claude-Session: <your session url>
   ```

## 3. Phase C tasks (spec §7; plan "Phase C")

### C1 — DONE (commit `69a4750`). What it established (C2 MUST emit matching column keys):
`scoring.create_results_dataframe("multiclass_simca")` schema =
`common_cols` (`Task, Model, Params, Preprocess, Deriv, Window, Poly, LVs, n_vars, full_vars, SubsetTag, Imbalance`) +
metric cols `["NoveltyAUC","Efficiency","NoveltyRate","NoClassRate","AmbiguityRate","ExactSetRate","MeanSensitivity","MeanSpecificity","Alpha","MinClassN","n_classes","engine_family","varsel_path"]` +
`["top_vars","all_vars","CompositeScore","Rank"]`.
`compute_composite_score(df,"multiclass_simca")` ranks by **-NoveltyAUC** (higher AUC = better), tie-break toward larger **MinClassN** (`-1e-9*MinClassN`); gap penalty is a no-op; variable penalty uses `n_vars/full_vars`. `model_registry.MULTICLASS_ENGINES = ['pca-simca','ocsvm','isolation-forest','lof','elliptic-envelope']`, `get_supported_models("multiclass_simca")` returns it. **Remaining C1 sites (`model_config`/`cv_utils`/`code_generator`/`templates`) are Phase-D/export — thread them in D, not now.**

### C2 — `run_multiclass_simca_search` (the big one)
Pattern on `run_one_class_search` (`search.py:5594`). **Never** build a G^K product.
- **Grid = preprocessing configs × engines** (`MULTICLASS_ENGINES`, or a user subset) × `varsel_path` options (none / wold_* / supervised-importance / precomputed). Per-row **per-class `n_components` is auto-tuned INSIDE the row** via `MultiClassClassModel(n_components="per_class_cv", …)` — do not grid it.
- **Metrics via Phase-A harness:** for each config fit `MultiClassClassModel`, get OOF metrics from `cross_validate` (feeds `multiclass_simca_metrics` on the OOF decision matrix) and novelty from `evaluate_novelty` (LOCO for in-data novel-class estimate). **Ranking metric = `novelty_tradeoff_auc` → the `NoveltyAUC` column** (spec §7: α-sweep AUC of (novelty_rate, 1−false_rejection_rate); NOT aggregate novelty at fixed α — that cheats on small classes). NaN-safe throughout.
- **Rows keyed** `(preprocessing, engine, varsel_path)`. **Unmodelable class → row flagged (reason recorded), NEVER silently drop a decision-matrix column** (spec §8). Populate `MinClassN` (smallest modeled class n), `n_classes`, `Alpha`, `engine_family`, `varsel_path` on every row.
- **Leakage:** per-spectrum SNV/SG/baseline outside folds (chemometrics convention, NOT leakage); column-autoscale / per-class calibration / varsel fit train-only inside folds (the `MultiClassClassModel` already enforces this — do not re-fit varsel outside its `fit`). A.1 per-class n_components tuning is nested inside A.2 (already inside `cross_validate`).
- **Tests (TDD):** grid runs on synthetic K=3 (+ a held-out novel class); per-class n_components tuned inside each row (assert no G^K blowup — row count == n_preproc × n_engine × n_varsel); rows keyed correctly; ranked by NoveltyAUC NaN-safe; unmodelable class → row flagged not dropped. Reuse the `_graded` / `_novel_split` synthetic helpers from `tests/test_simca.py` (copy or import).

### C3 — result-row population + composite wiring
- **Test:** the emitted result dataframe has the full C1 column set with **`engine_family` and `varsel_path` correct on every row**; `compute_composite_score` ranks by the §7 metric (already wired in C1 — C2/C3 just populate `NoveltyAUC`/`MinClassN`).
- **Impl:** C2 builds each row dict with the exact C1 keys; verify `create_results_dataframe("multiclass_simca")` + `add_result` round-trips.

## 4. Deferred items to fold in during C/D (from the A + B gates)
- **`predict_with_uncertainty` has no `multiclass_simca` branch** — needed before Phase D GUI (add a branch or explicit `NotImplementedError`). `predict_with_model` already has the A8 branch + `_SUPPORTED_TASK_TYPES` gate.
- **`_cross_fit_null` uses `get_one_class_model` with no PCA wrapper for EllipticEnvelope** — EE folds can fail when n_features>n_samples; add a fold-failure warning / PCA-reduce for EE (surfaces in C2 when EE engine runs on wide spectra).
- **Tuning-scaler leakage** (minor: per_class scaler fit on all rows before inner tuning CV — affects only the discrete n_components choice). Fix if cheap.
- **`novelty_tradeoff_auc` threshold count** is O(unique p-values) — downsample to ~500 before production scale (matters at real-data 757×2151).
- **Remaining task-type sites** (`model_config:158-181`, `cv_utils:275-340`, `code_generator`, `templates/validation.py`) — thread `multiclass_simca` in Phase D (export/GUI), with the fall-through audit extended to them.

## 5. Phase-C multi-family gate (MANDATORY at the phase boundary)
After C2+C3 committed, run a **multi-family panel** on the Phase-C diff: **Codex 5.5 (high) + ≥2 orthogonal families** from {Kimi K2.7, MiniMax M3, DeepSeek V4 Pro} — **rotate so GLM-5.2 (which writes the code) does NOT review its own diff**. **Routing (from memory — DO NOT violate):** Codex via `codex-reviewer` agent; Kimi/MiniMax via `opencode-call` (aliases `kimi27`/`minimax`), read-only. **DeepSeek must NOT go through opencode-go/opencode-call** — DeepSeek-API-only method (`feedback_deepseek_routing`); simplest is to use **Kimi + MiniMax** as the two orthogonal families (as Phase B did). Consolidate findings, **verify before agreeing** (`receiving-code-review` skill — the Phase-B panel produced 2 HIGHs that were empirically wrong; a probe refuted them), fold real ones with discriminating tests (revert-and-confirm-FAIL), surface methodology to the user.
- **Plus end-to-end smoke (Phase-C gate, plan):** real multi-class set through `run_multiclass_simca_search` + a genuinely held-out novel class; save→load→predict round-trip reproduces the decision matrix at max|diff|=0. Use the **real-data validation set** (§6). Aggregate metrics only — don't dump raw spectra.

## 6. Real-data validation set (for the C2/C3 e2e)
`C:\Users\mspon\Desktop\_DeskSync\contamination\Contaminated Samples Raw_ORAU Added.xlsx`, sheet `All Samples` (757×2151 FTIR; metadata cols `Specimen, Collagen, Site, contamination, Consolidant`; spectral cols are the integer-named 350–2500). **`Site`=10 classes is the flagship novelty case**; `contamination`=6, `Consolidant`=2. Phase-A smoke: SIMCA flagged 53–86% of held-out-site samples novel vs 100%-forced discriminant baseline. Use for the Phase-C real-data e2e: **LOCO on held-out sites** through `run_multiclass_simca_search`; compare NoveltyAUC / novelty rate across engines + preprocessing; confirm preprocessing sharpens it vs the raw-spectra Phase-A smoke.

## 7. Guardrails / decisions already LOCKED (do not re-litigate)
- **Never edit `search.py`'s single-Y path**; keep existing regression/classification/one_class paths byte-identical (diff/gold guard).
- **Chemometric standards, not ML standards, on leakage** (per-spectrum SNV/SG/baseline outside folds is NOT leakage; autoscale/calibration/varsel train-only inside folds).
- **α is GLOBAL** (never per-class). **IsolationForest `score_samples` NOT negated** (higher=more-normal). `scaling="per_class"` default.
- **Phase-B methodology resolved with user (2026-07-04):** keep **DPOW RMS-about-zero**; `balanced` = **min-max-normalized** MPOW·DPOW product; MPOW `ddof=0` upward-bias documented (dof-refit deferred); supervised-varsel adversarial-novelty limitation documented; **model-layer supervised ships `importance` only** — the **precomputed boolean-mask hook** (`variable_selection=<bool ndarray>`, `varsel_path_="precomputed"`) is how C2 wires any other supervised method (compute the mask externally on fold-train, pass it in). So C2's `varsel_path` enumeration = {none, wold_modeling/discriminating/balanced, supervised-importance, and optionally precomputed-mask from spa/cars/ga computed in the search loop}.
- **min_class_samples = LAYERED** (hard-block n<10; non-SIMCA n<20 unmodelable+warn; SIMCA warns at n<max(20,5·n_comp) but models). Varsel uses only classes ≥ max(min_class_samples, n_components+1) (Phase-B gate fix).

## 8. Merge-readiness (after Phase D — do NOT do earlier, do NOT auto-merge)
Full multi-family whole-diff pass (Codex + ≥2 orthogonal families) + the complete **pr-review-toolkit** (code-reviewer, silent-failure-hunter, pr-test-analyzer, type-design-analyzer). **Merge gate = local diff-failure-set vs `origin/main`** (main red on cloud CI since 2025-10-27, so compare failure SETS; the PR must add zero new failures). **Await explicit user greenlight.**

## 9. Session protocol
Per `CLAUDE.md`: append non-obvious findings to `SESSION_LOG.md` as you go; update `PROJECT_STATUS.md` after each task/phase; commit + push docs. Do NOT ask the user to remind you.

Start by reading §0, then begin **Phase C / task C2** (write its contract tests first, probe any numeric threshold before pinning).
