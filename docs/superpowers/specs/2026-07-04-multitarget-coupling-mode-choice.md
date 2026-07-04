# Orchestrator Brief — Multi-Target Coupling-Mode Choice (Independent / Joint / Both)

**For:** an Opus orchestrator driving **GLM 5.2** write-mode workers (`opencode-call`, `glm` alias, HALT-OR-BLOCK). Repo `C:\Users\mspon\git\dasp`, branch `feat/T17-multitarget-regression`, Python 3.12 (`.venv312`). Do NOT push/merge — leave commits on the branch for the main session to review.

**USER-APPROVED design (do not relitigate).** This is a follow-up to the shipped T-17 multi-target feature. Build it as additive changes; keep `run_search`'s single-Y path byte-identical.

---

## 1. Problem / motivation

Today the multi-target coupling mode is a **fixed property of each model** (`_STRATEGY_TABLE` in `src/spectral_predict/multitarget_search.py`): PLS/RandomForest/MLP/CatBoost/XGBoost are **JOINT-only**; Ridge/Lasso/ElasticNet/SVR/LightGBM/NeuralBoosted are **INDEPENDENT-only**. So a user **cannot** run PLS-1-per-target (independent PLS), RF-per-target, etc. — the flagship models are forced into joint multi-output.

That's backwards from common chemometrics practice, where **separate single-target models per property is the default** and joint (PLS-2) is the *specialized* choice for correlated targets. It also makes the key scientific comparison — "does coupling actually help on my data?" — impossible, because you can't put joint PLS-2 and independent PLS-1 on the same leaderboard.

Note: INDEPENDENT here = "one shared searched configuration, fit separately per target, ranked by mean per-target Q²" (the existing `INDEPENDENT_PRECISE_NOTE` semantics — "shared-config B"). This brief does NOT add per-target-optimal search (running the single-Y search N times — "C"); that stays a possible separate future feature.

## 2. Approved design

Make coupling a **user choice**, not a fixed model attribute. A GUI **coupling selector: Independent / Joint / Both**, default **Independent**.

**Mode → cells mapping** (per selected model, per config):
- **Independent:** every selected model → 1 INDEPENDENT cell (per-column / `MultiOutputRegressor`, `scale_y=False`, carries `INDEPENDENT_PRECISE_NOTE`).
- **Joint:** only joint-capable models → 1 JOINT cell. Joint-capable = PLS(→PLS-2), RandomForest, MLP, CatBoost, XGBoost, and Lasso/ElasticNet via their `MultiTaskLasso`/`MultiTaskElasticNet` variants (`supports_optional_joint`). A selected model with **no** joint variant (Ridge, SVR, LightGBM, NeuralBoosted) → **skip-with-notice** (e.g. "Ridge has no joint variant — skipped in Joint mode"), never silently run independent.
- **Both:** joint-capable models → **2 cells** (INDEPENDENT + JOINT) so the leaderboard shows e.g. `PLS [INDEPENDENT]` vs `PLS [JOINT]`; non-joint-capable models → 1 INDEPENDENT cell.

The `MultiTargetResult.mode` field (JOINT/INDEPENDENT) already distinguishes the rows on the leaderboard; "Both" simply emits both.

## 3. Code surface + required changes

Verify each anchor against the real code (line numbers may have drifted from the recent UX-fix commits).

### 3a. `src/spectral_predict/multitarget_search.py`
- **`_STRATEGY_TABLE` / `resolve_multitarget_strategy`** currently return ONE fixed strategy per model. Make resolution **mode-aware**: `resolve_multitarget_strategy(model_name, mode="independent"|"joint")`. Keep backward-compat (default preserves current behavior / the single-Y consolidation pin). For a joint-capable model + `mode="joint"` return the JOINT strategy; for any model + `mode="independent"` return an INDEPENDENT strategy (build one for the currently-JOINT-only models — PLS/RF/MLP/CatBoost/XGBoost — mirroring the existing `_independent(...)` shape, `scale_y=False`, `precise_note=INDEPENDENT_PRECISE_NOTE`). For Lasso/ElasticNet + `mode="joint"` → the MultiTask variant. For a model with no joint variant + `mode="joint"` → raise a clear, catchable error (the orchestrator turns it into skip-with-notice).
- **`build_multitarget_estimator`** already builds JOINT (native/`joint_params`) and INDEPENDENT (`MultiOutputRegressor` / native Ridge) estimators. Ensure it can build the **INDEPENDENT** form of the joint-capable models: PLS-1 per target = `MultiOutputRegressor(PLSRegression(scale=False, n_components=cap))`; RF/MLP/CatBoost/XGBoost independent = `MultiOutputRegressor(<single-target estimator with the same params>)`. Component/param capping still per-subset. INDEPENDENT never applies fold Y-scaling.
- **Honest labeling preserved:** INDEPENDENT cells carry `INDEPENDENT_PRECISE_NOTE`; JOINT cells do not. An INDEPENDENT PLS-1's mechanism string should read like "separate PLS-1 per target".

### 3b. `src/spectral_predict/multitarget_grid.py`
- `run_multitarget_grid_search` gets a `coupling_mode: str = "independent"` param ("independent"|"joint"|"both"). When expanding model×hp configs into cells, apply the §2 mapping: for each model, emit the requested mode(s), resolving strategy per (model, mode). Skip-with-notice (append to `output.skipped`) for joint-requested models with no joint variant. Each emitted cell evaluates via the existing `_evaluate_multitarget_cell` (which already records `mode`). Ranking unchanged (NaN-safe joint_q2).
- Dedup must key on **(model, mode, params, preprocessing, varsel)** so an independent and a joint cell of the same model are NOT collapsed.

### 3c. GUI `spectral_predict_gui_optimized.py`
- In `_create_tab4f_multitarget`, add a **coupling selector** (3 radios: Independent / Joint / Both; `self.multitarget_coupling = tk.StringVar(value="independent")`) near the model checklist. Update the model-checklist tags: instead of a fixed `[JOINT]`/`[INDEPENDENT]` per model, show each model's **capability** (e.g. "PLS — joint or independent", "Ridge — independent only") so the user understands what Joint/Both will do.
- `_collect_multitarget_config` forwards `coupling_mode=self.multitarget_coupling.get()` to `run_multitarget_grid_search`.
- The leaderboard already has a **Mode** column — with "Both" it now shows JOINT and INDEPENDENT rows for capable models; no column change needed. Surface the skip-with-notice ("… has no joint variant") in the existing skip line.
- Keep the pre-run cell-count heads-up honest (Both roughly doubles capable-model cells).

## 4. Tests (discriminating)
- INDEPENDENT PLS on 2-D Y = PLS-1 per target and is **numerically different** from JOINT PLS-2 on a correlated block (assert the two modes give different pooled preds / different joint_q2), proving both are reachable and distinct.
- A joint-capable model (e.g. RandomForest) runs in **independent** mode (MultiOutputRegressor, one forest per target) and produces a finite result labeled INDEPENDENT.
- `coupling_mode="both"` on a capable model produces **two** leaderboard rows (one JOINT, one INDEPENDENT) for the same config; dedup does not collapse them.
- `coupling_mode="joint"` with a no-joint-variant model (Ridge/SVR/LightGBM/NeuralBoosted) → that model in `output.skipped`, not raised, and the run completes with the others.
- Lasso/ElasticNet in joint mode use the MultiTask variant (JOINT, no `INDEPENDENT_PRECISE_NOTE`).
- **Single-Y byte-identity preserved:** the `n_targets == 1` consolidation pin + gold fixtures stay green; `run_search`/`search.py` 0-diff.
- Default `coupling_mode="independent"` — a GUI test asserts the tab defaults to Independent and forwards it.

## 5. Guardrails
- `run_search` single-Y path byte-identical (never edit `search.py`). Grid-engine-only (the coupling change is orthogonal to the engine guard). NaN-sink discipline (cell failure → `joint_q2=np.nan`, never finite 0.0). Dataclass fields append-only. No new deps. Never edit `.env`. `.venv312` only. Before each commit `git branch --show-current` == feat/T17-multitarget-regression; use explicit `git add` paths (NEVER `git add -A` — repo has many untracked scratch files); CRLF `git diff --stat` check after GUI edits. Do NOT push/merge.
- Chemometrics conventions are not bugs (per-spectrum ops, varsel-on-full-calibration, full-X autoscale). Both INDEPENDENT and JOINT are legitimate; do not "correct" one toward the other.

## 6. Process
Drive GLM 5.2 workers per component (3a backend strategy/builder → 3b orchestrator mode-expansion → 3c GUI selector → tests), each a TDD cycle; the orchestrator independently confirms each test FAILS on un-fixed code (reject tautological tests). Commit per component with explicit paths. After the build, run the multi-target + single-Y-gold suites green + `import spectral_predict_gui_optimized`, then report commit hashes + a short summary + any design ambiguity encountered (flag back rather than guess). The main session will Codex-review + cross-family-review the diff and handle the push.
