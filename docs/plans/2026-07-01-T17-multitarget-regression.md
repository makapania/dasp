# Plan: Multi-target regression search (PLS-2 + general multi-output) — T-17

## Context

**Why.** dasp models one target at a time. Bone-FTIR/paleodiet research needs several *correlated* numeric properties at once (collagen yield, IRSF, carbonate/phosphate, amide/phosphate, C/N isotopes). Joint multi-target modeling lets correlated targets share structure and can beat N separate single-target models. Ticket **T-17**, user-endorsed. The gap is that dasp's pipeline force-flattens y to 1-D in ~40 places.

**Scope (user-directed 2026-07-01): the WHOLE model menu, not just PLS.** Multi-target is offered for every regression model — using each model's **native joint** multi-output where it genuinely couples targets, and **batched-independent** (`MultiOutputRegressor`) where it doesn't — with **prominent, honest JOINT-vs-INDEPENDENT labeling** everywhere so batched is never mistaken for coupling.

**"PLS-2" ≠ 2 targets.** It's the multi-response PLS variant vs PLS-1. Any number of targets (1, 2, 3, 5, …); Y is an (n_samples × n_targets) block throughout.

**Design validated by cross-family review** (Codex repo-grounded + DeepSeek V4 Pro direct API); both NEEDS-CHANGES, findings folded in.

## Direction (user refinements)

1. **Original stays intact.** Legacy single-Y search untouched, remains default.
2. **Superset (n_targets ≥ 1).** The new path treats single-target as the 1-column case, so it could *eventually replace* the legacy path — a **consolidation seam**, not v1 work; proven by pinning single-target output to legacy numeric parity.
3. **Built with deferred methods in mind + shared foundation now.** Anything the deferred UVE/CARS multi-Y work shares with the core is built once (see Foundation); deferred work becomes "add an aggregation rule," not "re-plumb."

## Per-model multi-output strategy (the core of the scope)

Installed/pinned versions verified: sklearn 1.8, xgboost 3.2 (pin ≥2.0), catboost 1.2.10 (pin ≥1.2), lightgbm 4.6.

| Model | Mode | Mechanism |
|---|---|---|
| **PLS** | **JOINT** (flagship) | shared latent variables (sklearn native 2-D Y) |
| **RandomForest** | **JOINT** | impurity averaged across targets (sklearn native) |
| **MLP** | **JOINT** | shared hidden layers (sklearn native) |
| **CatBoost** | **JOINT** | `loss_function='MultiRMSE'` |
| **XGBoost** | **JOINT** | `multi_strategy='multi_output_tree'` |
| **Ridge** | **INDEPENDENT** (native 2-D solve) | per-target, single solve |
| **Lasso / ElasticNet** | **INDEPENDENT** default; optional **JOINT** | plain = `MultiOutputRegressor`; `MultiTaskLasso/ElasticNet` = joint sparsity |
| **SVM (SVR)** | **INDEPENDENT** | `MultiOutputRegressor` |
| **LightGBM** | **INDEPENDENT** | `MultiOutputRegressor` (no native multi-output) |
| **NeuralBoosted** | **INDEPENDENT** | `MultiOutputRegressor` (1-D only per `neural_boosted.py`); or defer/disable in UI |

A `resolve_multitarget_strategy(model_name)` returns the mode + how to build the estimator. **INDEPENDENT mode = separate per-target estimators.** Precise claim (Codex): for a *fixed* preprocessing/wavelength/hyperparameter config, INDEPENDENT is bit-identical to N separate single-target fits. But in a *search* that selects one shared config by mean per-target score, adding/removing a target can change the selected config for all targets — so it is NOT identical to N independent searches. The label must state this exactly (below).

## Honest-labeling requirement (statistical-integrity guardrail — non-negotiable)

Every surface tags each model **JOINT** or **INDEPENDENT**: model picker, results-table column (`MultiTarget Mode`), CSV, exported code header, and plot captions. INDEPENDENT rows carry the precise note: **"separate per-target estimators under one shared searched configuration — not a coupled model, no correlation benefit; for exact equivalence to N independent searches, run separate single-target searches."** (Codex: the looser "== separate models" wording overstates equivalence under a shared-config search.) Without this, batched breadth reads as coupling it doesn't have.

## Shared model-agnostic foundation (build now) — `src/spectral_predict/multi_y.py`

Estimator-agnostic; consumed by the orchestrator, all models, v1 varsel, VIP, and (later, unchanged) UVE/CARS:

- **`FoldYScaler`** — per-column train-fold mean/std; transform train/val; inverse-transform preds.
- **`multi_y_cv_pool(...)`** — shape-aware CV preds `(n, n_targets)`. Replaces 1-D `cross_val_predict_pooled` (its repeated-CV branch allocates `pred_sum=np.zeros(n_samples)` and ravels at `cv_utils.py:539/552` — unusable). Reuses splitter `build_cv_splitter` (`cv_utils.py:321`).
- **`multi_y_metrics(...)`** — **per-target metrics on RAW units** (ultramode fix). Inverse-transform each fold's JOINT predictions to raw via that fold's `FoldYScaler`, pool raw `(n, n_targets)` preds/truths, compute per-target `Q2_s = 1 − PRESS_s/SSY_s` in raw units (SSY_s referenced to the single grand/full-training mean = pooled-test mean under complete K-fold/LOO), then **average across targets → joint Q²**. Consequences: joint Q² == mean per-target Q²; removes cross-fold bias from dividing residuals by differing per-fold stds (material at n≈20–60); and per-target Q² == legacy `r2_score(all_y_test, all_y_pred)` at n_targets=1 (secures the consolidation pin). Report per-target **R²/RMSE in raw units** (R² is scale-invariant, RMSE is NOT — R² alone can't catch a units bug), plus per-target **RPD, RER, CCCcv, Bias** for single-Y reporting parity (RPD=std(y)/RMSEcv, RER=range(y)/RMSEcv, matching `search.py:4820-4825`; Williams RPD tiers are what a NIR/FTIR reader checks first). These are per-target REPORTING metrics only, distinct from the joint-Q² SELECTION criterion.
- **`reduce_multi_y_score(per_target, rule="mean")`** — per-target CV error → scalar (mean is correct because per-target Q² is scale-invariant on raw units). Used by performance-based varsel + hyperparameter/component selection.
- **`extract_pls_multi_y(pls)`** — exposes the canonical chemometrics B-matrix as `(n_features, n_targets)`, but **implement as `pls.coef_.T`**: sklearn ≥1.3 (verified on pinned 1.8) exposes `PLSRegression.coef_` as `(n_targets, n_features)` — the transpose of the pre-1.1 layout (single-target → `(n_features, 1)`). Add a `tests/test_multi_y.py` orientation assertion. **Seam** for VIP now (VIP is unaffected — it reads `x_weights_`/`x_scores_`/`y_loadings_`, not `coef_`), UVE/CARS later.
- **`aggregate_importance(matrix, rule=...)`** — one home for aggregation rules. v1 = VIP loading-sum; v1.1 adds `uve_stability`, `cars_scaled_coef` with no re-plumbing.
- **`inter_target_correlation(...)` / `cap_components(...)`** — correlation guardrail + `A ≤ min(N−1, p)` cap.

**Task-agnostic seam (for future classification).** Orchestration, `multi_y_cv_pool`, JOINT/INDEPENDENT labeling, and the correlation guardrail are task-agnostic. Only the **metric layer** (`multi_y_metrics`) and **Y-scaling** are regression-specific. Structure them as a swappable layer so multi-label *classification* (accuracy/F1 metrics, native multi-output classifiers, no Y-scaling) can slot in later as a task mode of THIS module — not a rewrite.

## Statistical core (both reviewers ranked top)

- **Fold-wise Y autoscaling — mode-dependent (Codex correction + ultramode):**
  - **JOINT models** (PLS covariance, RF multi-output impurity, MLP summed-MSE, CatBoost MultiRMSE, **XGBoost multi_output_tree**, **MultiTaskLasso/ElasticNet** — their Frobenius fit + shared L21 row-support are non-separable, so unscaled Y lets the highest-variance target dominate the shared sparsity) **fit on fold-scaled Y** — otherwise a high-variance target dominates the shared criterion.
  - **INDEPENDENT models fit on RAW per-target Y** (NOT scaled). This rule applies ONLY to per-target-**separable** estimators (plain Lasso/EN via `MultiOutputRegressor`, SVR, LightGBM, NeuralBoosted): SVR `epsilon`, Lasso/EN `alpha` are scale-sensitive at fixed hyperparameters, so fitting on scaled Y would break the "== separate per-target models" guarantee.
  - **Metric, NOT scaled (ultramode fix):** the fold Y-scaler is used **only for JOINT-model fitting**, never for the metric. JOINT predictions are inverse-transformed to raw before scoring; per-target Q²/RMSE are computed on **raw units** (per-target Q² is scale-invariant so equal weighting is automatic — no scaled-unit pooling, which would bias PRESS across folds and break the n_targets=1 legacy parity). See `multi_y_metrics`.
  - **Final deployed model:** Y-scaler (for JOINT/MultiTask modes) fits on the **entire training set**, not fold-wise (fold scaling is CV-metric-only) — DeepSeek. (PLS fit `scale=False` at `models.py:132/399`; X autoscale unchanged.)
- **Hyperparameter / component selection** by the joint multivariate criterion (max joint Q² = mean per-target raw-unit Q²), never off one target; PLS components capped `A ≤ min(N−1, p)`.
- **Weak-correlation guardrail:** mean abs pairwise Pearson among scaled targets < ~0.35 → warn separate PLS-1 models likely better (no-op at n_targets=1). Especially relevant for JOINT models.

## Variable selection scope

**Architecture (Codex correction).** dasp's varsel methods have PLS hardwired as their *internal evaluator* (`_evaluate_interval_pls`, GA-PLS fitness, SPA/CARS PLS CV) — they are NOT model-agnostic today. But that's fine, because dasp already uses varsel as **PLS-proxy selection that feeds any downstream model**: varsel picks a wavelength subset, then the chosen model (RF, LightGBM, …) trains on it. So multi-target varsel stays **multi-Y-PLS-evaluated** (extend the internal PLS evaluator to 2-D Y via `multi_y_cv_pool` + `reduce_multi_y_score`) and outputs a subset consumed by **any** multi-target model. We do NOT refactor each method to use the downstream model as its evaluator in v1.

**v1 (multi-Y-PLS-evaluated, consume the foundation):**
- **Performance-based** — iPLS fwd/bwd, MC-siPLS, MWPLS, GA-PLS, SPA, performance hybrids: internal PLS evaluator extended to 2-D Y (mean joint Q²/RMSECV via the reducer).
- **VIP** — canonical multi-Y (`Σ_a (Σ_targets y_loadings[a]²)(w_ja/‖w_a‖)²`) via the seam; fix `models.py compute_vip` — **must keep the exact `Q[0,:]` result for single-target so single-Y VIP stays byte-identical** (`:1768`, docstring recipe already at `:1730`).
- Tree models: only **JOINT** tree models (RandomForest, XGBoost multi_output_tree, CatBoost MultiRMSE) expose a genuinely multi-target-aware scalar `feature_importances_`. **INDEPENDENT** (MOR-wrapped) tree models — in v1 just **LightGBM** — have NO `feature_importances_` on the wrapper (only `.estimators_`); `get_feature_importances` (`models.py:1866-1868`) would raise `AttributeError`. The display path must `isinstance`-check `MultiOutputRegressor` and aggregate per-target importances across `mor.estimators_` (mean), routed through `aggregate_importance` (its natural first non-PLS consumer).

**v1 — grey out in multi-target (seams pre-built):** UVE, CARS (PLS-mode L2 unsupported; tree-mode is LightGBM, not PLS), UVE/CARS hybrids → v1.1 aggregation rules.

**Deferred:** model-as-evaluator varsel (each method scored by the *actual* downstream model via an injected-evaluator API) — a larger refactor, not needed for v1.

## Files & touch-points

**New:** `src/spectral_predict/multi_y.py` (foundation); `src/spectral_predict/multitarget_search.py` (model-agnostic orchestrator, n_targets ≥ 1, per-model strategy resolver); `tests/test_multi_y.py`, `tests/test_multitarget_search.py`.

**Modify (additive; single-Y path byte-identical):**
- `models.py` — `resolve_multitarget_strategy` + estimator builders (CatBoost MultiRMSE, XGBoost multi_strategy, MultiOutputRegressor wrap, MultiTask linear); fix `compute_vip`.
- `variable_selection.py` + `ga_pls.py` — per-method `if y.ndim==1` (exact current code) / `else` (multi via foundation) branches for performance-based methods; downstream arrays shape-aware (entry ravel removal alone insufficient — Codex); grey-out UVE/CARS/hybrids.
- GUI `spectral_predict_gui_optimized.py` — multi-select target widget (extend the `Combobox` at `6446`; add `self.selected_targets` beside `self.target_column` so single-Y UI is untouched); a **new "Multi-Target" tab/sub-tab** (placement below); model picker + results grid + CSV + export all carry the JOINT/INDEPENDENT tag, per-target columns (R²/RMSE/RPD/RER/CCCcv/Bias in raw units), and the joint-Q² column.
- Model save/load + code export — persist target names, per-target Y-scaler stats, multi-target mode, prediction columns, per-target metrics.
- `scoring.py:503-513` — leave CV-ANOVA 2-D guard (stays single-Y-only).

**Search-ENGINE scope — v1 = Exhaustive/Grid ONLY (ultramode, high-value isolation gap).** The GUI drives all three engines from ONE shared `self.optimization_method` radio (`grid`/`unified`/`nsga2`, at `gui:3083` + `12454-12456`). Multi-target runs through the new `multitarget_search.py` (grid) orchestrator ONLY. `unified_bayesian.py` (Bayesian/TPE — passes y through with no ndim handling, calls `cross_val_predict_pooled` + `compute_cv_anova_pvalue`) and `nsga2_search.py` (assumes 1-D via `Q[0,:]` + `y_pred.ravel()`) are **DELIBERATELY-EXCLUDED 1-D engines**. When >1 target is selected the multi-target dispatcher MUST (1) **grey out** the "Bayesian Optimization" (`value=unified`) and "NSGA-II" (`value=nsga2`) radios — mirroring the UVE/CARS grey-out pattern — AND (2) **force `optimization_method='grid'` at run dispatch**, so a stale radio value cannot silently route 2-D Y into a 1-D engine. This completes the isolation claim (previously scoped only to `search.py` flatten sites).

**Other shared 1-D surfaces that need multi-Y handling (Codex):**
- **Code export** (`code_generator.py:1957`, `templates/validation.py:105`, `templates/header.py:47`) — 1-D/ravel; need multi-Y export templates so exported scripts reproduce multi-target predictions.
- **Booster early-stopping CV** (`cv_utils.py:1026/1085`) — 1-D. For JOINT boosters (LightGBM/XGBoost/CatBoost) with early stopping under multi-Y: either add a multi-Y early-stopping path OR **disable early stopping for multi-target boosters in v1** (simplest safe default).
- **`model_io.py`** (`:112/:602/:1016`) — uncertainty/docs assume a single regression vector; make per-target, and update the docstrings that assert single-vector shapes (`cv_residuals '(n_cv_samples,)'` at `:112`, `predict → '(n_samples,)'` at `:602`) to document the `(n_samples, n_targets)` case.

**GUI placement:** a dedicated **Multi-Target sub-tab under Analysis Configuration** (tab4) for v1 — keeps isolation while sitting beside the normal search. (Because the path is a superset, a later consolidation could fold it into the main search as an "N targets" mode; not v1.)

## Build order (one feature, staged for safety)
Foundation → PLS-2 (JOINT, flagship, prove machinery + single-target parity) → other JOINT models (RF, MLP, CatBoost, XGBoost; disable booster early-stopping in multi-Y for v1) → INDEPENDENT models (SVR, LightGBM, plain Lasso/EN, NeuralBoosted) fit on **raw Y** with labeling → multi-Y-PLS-evaluated varsel + VIP → multi-Y export templates + per-target model_io. Cross-family re-review before merge.

## Verification (end-to-end)
1. **Single-Y varsel byte-identity (committed gold fixture, ultramode).** Not a one-time "vs `main`" check — add `tests/gold_standards/varsel_single_y.npz` (matching the existing `*.npz` gold pattern) capturing selected indices + score vectors for each modified method (iPLS, SPA, MC-siPLS, MWPLS, GA-PLS) on a fixed 1-D synthetic case generated FROM `main`, with **explicit RNG seeds** (GA-PLS especially — any change in RNG draw count breaks byte-identity). Assert `np.array_equal` on indices / `assert_allclose` on scores after the refactor. (Existing varsel tests are property/range-only, so the line-85 change could silently perturb selected wavelengths otherwise.) Existing suites green.
2. **Consolidation pin:** `multitarget_search` at n_targets=1 matches legacy single-Y CV metrics; specifically assert **per-target Q² == legacy `r2_score(all_y_test, all_y_pred)`** within tolerance (secured by the raw-unit metric).
3. **JOINT ≠ INDEPENDENT + label truthfulness:** on a correlated target pair, a JOINT model's per-target result differs from the INDEPENDENT result; and **INDEPENDENT at a FIXED config** matches two separate single-target fits bit-for-bit (fit on raw Y), while noting a shared-config *search* need not equal N separate searches (proves the label's precise wording).
4. **Y-scaling + raw-unit metric pin:** fold-wise scaling matches a hand-built sklearn `Pipeline` within 1e-6; high-variance target no longer dominates; AND run a 2-target JOINT PLS with **deliberately divergent variances** (std ≈0.1 vs ≈100) and assert reported per-target RMSE equals RMSE recomputed from **inverse-transformed raw** predictions and is NOT the scaled-unit value (pins that inverse-transform precedes per-target reporting).
5. **Native multi-output smoke:** PLS, RF, MLP, CatBoost(MultiRMSE), XGBoost(multi_output_tree) each fit a 3-target block on scaled Y and return `(n,3)` preds; LightGBM/SVR/NeuralBoosted via MultiOutputRegressor fit on **raw Y** and return `(n,3)`.
5b. **Export + early-stopping:** exported multi-Y script reproduces in-app multi-target predictions; confirm booster early-stopping is disabled (or multi-Y-correct) under multi-target.
5c. **model_io round-trip (distinct from 5b's export path):** train a 3-target JOINT PLS, `save_model()`→`.dasp`→`load_model()`, assert (a) reloaded `predict()` bit-identical to pre-save; (b) per-target Y-scaler mean/std reload so inverse-transform still yields raw-unit per-target preds; (c) target names + per-target metrics survive in order. Mirror `tests/test_autoscale_save_load_roundtrip.py`.
6. **Labeling guardrail (automated, ultramode).** Table-driven pure-function test of `resolve_multitarget_strategy`: exact mode for every model in the per-model table; an **unknown model name RAISES** (fails loud, never silently mislabels coupling); and INDEPENDENT results carry the exact precise-note substring from the honest-labeling section (pins the string against drift).
6b. **Metrics/VIP/correlation-warning** as specified; multi-Y VIP matches manual formula, single-Y VIP unchanged; `coef_.T` orientation asserted.
7. **Run GUI** (`python spectral_predict_gui_optimized.py`, `.venv312`): multi-select 3 numeric targets, run across a JOINT and an INDEPENDENT model, confirm per-target + joint results, correct mode labels, Bayesian/NSGA-II radios greyed out, CSV/export; single-target mode still works.
8. Cross-family re-review (Codex + DeepSeek) before merge.

## Explicitly deferred
- **UVE/CARS multi-Y** (v1.1 via the `aggregate_importance` seam).
- **Multi-label CLASSIFICATION** — a future *task mode* of THIS module (reuses the task-agnostic foundation; swaps the metric layer + uses models' native multi-output classification). NOT its own module.
- **Multi-class SIMCA (T-31)** — explicitly SEPARATE: a class-modeling method (one model per class, "none/multiple" membership), not multi-target prediction. Stays its own ticket/module; does not merge here.
- Consolidation of the legacy single-Y orchestrator onto the superset; categorical/mixed targets in the joint regression model; multi-Y CV-ANOVA.
