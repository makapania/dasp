# Continuation Prompt — T-17 Multi-Target Regression (2026-07-02)

## Status: BUILD COMPLETE (F1–F7), NOT MERGED

Branch: `feat/T17-multitarget-regression` (repo `C:/Users/mspon/git/dasp`). Tip commit `8c66f1c`.
Pushed to remote this session with `-u` (first time the branch exists on origin). Not yet merged to `main`.

Spec: canonical at `~/.claude/plans/i-want-you-to-humble-crystal.md`; repo copy `docs/plans/2026-07-01-T17-multitarget-regression.md`.

Goal delivered: joint multi-output regression as a **Grid-only** superset path (n_targets ≥ 1) where the legacy single-Y path stays **BYTE-IDENTICAL** (proven by gold fixtures). JOINT vs INDEPENDENT modes are honestly labeled per model.

---

## Done + tested + reviewed (each committed and tagged `t17-F{n}-*`)

| Feature | Commit(s) | Codex | DeepSeek | What |
|---|---|---|---|---|
| F1-foundation | `95f1437` + review `ebb31b3` | NEEDS-CHANGES (folded) | SOUND | `src/spectral_predict/multi_y.py`: `FoldYScaler`, `multi_y_cv_pool`, `multi_y_metrics`, `extract_pls_multi_y`, `cap_components`. 22 tests. |
| F2-orchestrator-pls2 | `c1de0f3` + review `7298f46` | NEEDS-CHANGES (folded) | SOUND | `src/spectral_predict/multitarget_search.py`: `resolve_multitarget_strategy`, `build_multitarget_estimator` (PLS-2), `run_multitarget_search` (joint-Q² ranking + grid-only guard). |
| F3-joint-models | `c272d96` | n/a | SOUND | RF/MLP/CatBoost(MultiRMSE)/XGBoost(multi_output_tree) native JOINT. Booster early-stopping OFF under multi-Y (v1). |
| F4-independent-models | `c431849` | SOUND | SOUND | Ridge native 2-D; SVR/LightGBM/Lasso/EN/NeuralBoosted via `MultiOutputRegressor`; optional JOINT MultiTaskLasso/EN. |
| F5-varsel-vip | `5a3f637` + review `13cc81e` | NEEDS-CHANGES (folded) | NEEDS-CHANGES (folded) | iPLS fwd/bwd + MC-siPLS + MWPLS + SPA + GA-PLS + `compute_vip` multi-Y via additive `if y.ndim==1`/`else`. Gold-fixture byte-identity. UVE/CARS+hybrids raise on 2-D Y. |
| F6-gui | `02ca8f1` + review `10c1db2` | NEEDS-CHANGES (folded) | NEEDS-CHANGES (folded) | "Multi-Target" sub-tab: EXTENDED-select targets, JOINT/INDEPENDENT tags, results Treeview + CSV export. Bayesian/NSGA-II greyed + forced to grid when >1 target. |
| F7-export-modelio | `c338bea` + review `8c66f1c` | NEEDS-CHANGES (folded) | SOUND | Multi-Y export templates + isolated `_generate_multitarget_script/_notebook`; `save_model(y_scaler=)` → `y_scaler.npz`; `predict_with_model` inverse-transforms JOINT preds to RAW. Exports reproduce in-app preds at max\|diff\|==0.0. |

### Test verification (this session, `.venv312`, Python 3.12)
- `pytest tests/test_multi_y.py tests/test_multitarget_search.py tests/test_variable_selection.py tests/test_multitarget_export.py tests/test_multitarget_save_load_roundtrip.py` → **138 passed**.
- `pytest tests/gui/test_multitarget_tab.py tests/test_vip_formula.py tests/test_plsda_importance.py` → **18 passed** (GUI tab + single-Y VIP byte-identity pins).
- Known-unrelated: `test_cv_strategy.py` failure is main-red (confirmed pre-existing at F1 with changes stashed); two `test_export_code` subprocess tests fail only because bare `python` on PATH lacks sklearn (not `.venv312`) — documented T-CI-3/T-CI-4.

---

## Next feature / next action

**No F8 in the plan — the build is feature-complete.** The next action is **merge readiness**, not more building:

1. **Open a PR** `feat/T17-multitarget-regression` → `main`. Do NOT auto-merge. Per project process (`feedback_check_ci_before_merge.md`): main has been red since 2025-10-27, so verify via local diff-failure-set (PR branch adds zero new failures vs `origin/main`), not the cloud CI rollup. Wait for explicit user greenlight before `gh pr merge`.
2. **Recommended pre-merge whole-diff cross-family review.** The Codex/DeepSeek reviews were per-feature gates scoped to one feature's diff — they did not see cross-feature interactions. A single pass over the full `main..feat/T17-multitarget-regression` diff (peer-review skill: DeepSeek + GLM + Kimi) is advisable before merge.
3. **End-to-end smoke** per `feedback_e2e_smoke_after_refactor.md`: run an actual 2-target regression through `run_multitarget_search` on real data (e.g. `example/BoneCollagen.csv` with two numeric targets) AND through the GUI Multi-Target sub-tab, then export + reload + predict, to catch integration breakage unit tests miss.

---

## Parked blockers / deferrals (v1.1, explicitly out of v1 scope)

- **UVE/CARS variable selection on multi-Y** — raises `NotImplementedError` via `_reject_multi_y` (function-level mirror of the GUI grey-out). UVE-on-y is a discrimination method; multi-Y adaptation is a separate design task.
- **Bayesian + NSGA-II multi-target** — 1-D-only engines. Multi-target is Grid ONLY, enforced at three layers (GUI grey-out + GUI force-to-grid + `run_multitarget_search` guard). Extending TPE/NSGA-II to 2-D Y is a v1.1 design task.
- **F6 per-model hyperparameters** — the Multi-Target sub-tab uses default hyperparameters per model (no full grid UI). Within F6 scope but a natural follow-up if users want tuning.
- **Booster early-stopping under multi-Y** — disabled for v1 (JOINT CatBoost/XGBoost fit full iterations, no `eval_set`). Fine for correctness; a perf/quality follow-up.

## HARD GUARDRAILS (unchanged — do not violate)

1. Single-Y path stays **BYTE-IDENTICAL**: every varsel/VIP change is an additive `if y.ndim==1` branch, proven by `tests/gold_standards/varsel_single_y.npz` + `TestSingleYByteIdentityGold`. Regenerate gold ONLY from pre-change HEAD.
2. Chemometrics conventions win: per-spectrum SNV/SG/baseline on full data and varsel-on-full-calibration are NOT leakage — do not "fix" them.
3. Multi-target = Grid engine ONLY (grey out + force `optimization_method='grid'`).
4. Fold Y-scaler = JOINT fitting only; per-target metrics on RAW units.
5. Work ONLY on `feat/T17-multitarget-regression`; never main, never force-push, never skip hooks, never destructive resets (parallel sessions may share refs).
6. No new deps without `pyproject.toml`; never touch `.env`; never commit data files.
