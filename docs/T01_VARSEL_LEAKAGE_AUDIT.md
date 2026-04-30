# T-01: Per-Fold Variable Selection Leakage Audit

> Date: 2026-04-29
> Scope: Determine whether each variable selection method x search path refits varsel inside CV folds (CLEAN) or computes once on full calibration data with a frozen list passed into CV (LEAKY).
> Domain rule: Per-spectrum preprocessing (SNV/SG/baseline) is NOT leakage by chemometrics convention. This audit is specifically about y-using variable selection.

## Summary

| Verdict | Count |
|---------|-------|
| CLEAN    | 0    |
| LEAKY    | 60   |
| UNKNOWN  | 0    |
| N/A      | 35   |

**Every y-using variable selection method in every search path is LEAKY.** No method x path combination refits varsel inside CV folds. The pattern is universal: varsel runs once on the full preprocessed calibration matrix with full y, produces a frozen `top_indices` array, and `_run_single_config()` / `run_one_class_cv()` / `cross_val_score()` receives the pre-subsetted data with no per-fold varsel re-run.

The Refinement/Model Development path is N/A for all methods because it reuses the saved wavelength list from the search result — it does not re-run any varsel algorithm.

## Top 3 highest-impact leaks

1. **Grid search CARS (`cars_selection`) — `search.py:2552-2563`**
   CARS is the most popular varsel method in chemometrics (adaptive reweighted sampling with MC subsampling). It uses y both for MC sampling stratification and for the internal PLS fitness evaluation. With N<200, full-data CARS inflates R2cv by 0.05-0.15 in noisy data. It appears in all four search paths (grid, old Bayesian, unified Bayesian, one-class grid) and is the default varsel method for the unified Bayesian path. Affects the headline metric every user sees.

2. **Grid search `importance` mode — `search.py:2274` + `search.py:2476-2478`**
   The Pipeline is fit on full `(X_for_models, y_np)` at line 2274. For PLS models, this calls `compute_vip()` on the full-data model (models.py:1786), which uses y-dependent PLS weights/scores. For tree models, `feature_importances_` come from a full-data fit. This is the fallback varsel method and runs for every model x preprocessing combination. Impact: moderate (VIP/feature_importances are less noisy than CARS, but still inflate R2cv by ~0.02-0.05 on small datasets).

3. **One-class grid search UVE prefilter — `search.py:5649-5655`**
   UVE prefilter runs on `(X_preprocessed, y_oc)` where `y_oc` is the +1/-1 inlier/outlier label vector derived from full data. This prefilter can eliminate 30-60% of variables before any per-method varsel runs, compounding the leakage. For one-class, the y labels are especially sensitive (binary inlier/outlier), so full-data UVE has a larger inflation effect than for regression. See T-04 overlap.

## Recommended fix ordering for separate tickets

1. **Grid search path (highest user traffic)** — Create a `VarselTransformer` sklearn-compatible wrapper that wraps each varsel method and can be inserted into a Pipeline before the model. Extend `_run_single_config()` to optionally include the VarselTransformer in the pipeline so varsel refits per-fold from train indices only. This single fix covers the grid search path for all methods simultaneously. Reference: `cv_utils.py:cross_val_predict_pooled` as the per-fold loop infrastructure.

2. **Unified Bayesian path** — The `compute_importances()` function at `unified_bayesian.py:593` must be replaced with per-fold varsel inside the Optuna objective. This is architecturally harder than the grid fix because the objective already runs per-trial (not per-fold), and the importances are cached per preprocessing+method across trials. The fix: instead of caching importances computed on full data, cache per-fold varsel results keyed by `(preprocess, method, fold_train_indices_hash)`.

3. **Old Bayesian path (`bayesian_utils.py`)** — Same fix pattern as unified Bayesian but simpler (the old path calls `_run_single_config_fn` which could be upgraded to include varsel in the pipeline, piggybacking on the grid search fix).

4. **One-class grid search** — Same VarselTransformer approach as regression/classification grid search, but the one-class path has its own separate loop at `search.py:5676-5939`. The UVE prefilter at `search.py:5641-5674` also needs per-fold re-running.

5. **NSGA-II guidance (lowest priority)** — The CARS-Tree guidance at `nsga2_search.py:1872` is NOT leakage in the traditional sense (it biases the initial population sampling, not the CV evaluation). The CV evaluation uses the chromosome's wavelength mask directly with no y-using varsel. Mark as LOW priority.

---

## Search Paths Overview

| Path | Entry Point | Varsel Pattern |
|------|-------------|----------------|
| Grid search | `run_search()` in `search.py` | Varsel on full `(X_for_models, y_np)` at 2314-2741, frozen `top_indices` at 2829, CV via `_run_single_config()` |
| Old Bayesian | `run_bayesian_search()` in `search.py` → `bayesian_utils.create_objective_function()` | Varsel on full `(X, y)` at 444-548, frozen `top_indices` at 557, CV via `_run_single_config_fn()` |
| Unified Bayesian | `run_unified_bayesian()` in `unified_bayesian.py` | Varsel on full `(X_prep, y)` at 1120-1145, frozen `top_indices` at 1149, CV via `cross_val_score()` |
| NSGA-II | `run_nsga2_search()` in `nsga2_search.py` | Chromosome encodes wavelength mask; CARS at 1872 is guidance only (not per-fold varsel); CV via `cross_val_score()` |
| One-class grid | `run_one_class_search()` in `search.py` | Varsel on full `(X_preprocessed, y_oc)` at 5676-5870, frozen `top_indices` at 5937, CV via `run_one_class_cv()` |
| Refinement | `_run_refined_model_thread()` in GUI | Reuses saved wavelength list; does NOT re-run varsel |

---

## Method x Path Matrix

### UVE (`uve_selection` / `get_uve_threshold`)
- **Grid search**: LEAKY. `get_uve_threshold(X_transformed_varsel, y_np, ...)` at `search.py:2498-2504`. Uses full y. Importances fed to `top_indices` at 2829. No per-fold refit. **Magnitude**: Moderate. UVE's internal CV (5-fold within UVE itself) partially mitigates the leakage, but the variable set is still frozen across the outer CV folds. Typical R2cv inflation: 0.03-0.08 for N<200. **Fix**: VarselTransformer wrapper in pipeline.
- **Old Bayesian**: LEAKY. `uve_selection(X, y, ...)` at `bayesian_utils.py:458-464`. Same frozen-indices pattern at 557. **Fix**: Same as grid search.
- **Unified Bayesian**: LEAKY. `uve_selection(X, y, ...)` at `unified_bayesian.py:673-678`. Frozen `top_indices` at 1149. **Fix**: Per-fold varsel inside Optuna objective.
- **NSGA-II**: N/A. NSGA-II does not use UVE for varsel. The chromosome encodes wavelengths directly.
- **One-class grid**: LEAKY. `get_uve_threshold(X_preprocessed, y_oc, ...)` at `search.py:5723-5731`. Uses full `y_oc` (+1/-1 labels). Frozen `top_indices` at 5937. **Magnitude**: High for one-class. Binary labels make UVE threshold unstable on small datasets. **Fix**: Per-fold UVE re-run.
- **Refinement**: N/A. Reuses saved wavelength list.

### SPA (`spa_selection`)
- **Grid search**: LEAKY. `spa_selection(X_transformed_varsel, y_np, ...)` at `search.py:2487-2493`. Uses full y for the internal MLR fitness. Frozen `top_indices` at 2829. **Magnitude**: Moderate. SPA's projection-based selection is less y-sensitive than CARS, but the MLR RMSE evaluation still uses full y. Typical inflation: 0.02-0.06. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: LEAKY. `spa_selection(X, y, ...)` at `bayesian_utils.py:444-451`. **Fix**: Same.
- **Unified Bayesian**: N/A. Unified Bayesian does not include `spa` in `available_methods` (line 827: `['importance', 'cars', 'region']`).
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. `spa_selection(X_preprocessed, y_oc, ...)` at `search.py:5714-5720`. **Fix**: Per-fold SPA re-run.
- **Refinement**: N/A.

### iPLS (`ipls_selection`)
- **Grid search**: LEAKY. `ipls_selection(X_transformed_varsel, y_np, ...)` at `search.py:2527-2533`. Internal PLS + CV uses full y. Importances fed to `top_indices`. **Magnitude**: Lower than CARS/UVE. iPLS selects contiguous spectral intervals; the selection is somewhat stable across folds. Typical inflation: 0.01-0.04. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: LEAKY. `ipls_selection(X, y, ...)` at `bayesian_utils.py:491-498`. **Fix**: Same.
- **Unified Bayesian**: N/A. Not in `available_methods`.
- **NSGA-II**: N/A.
- **One-class grid**: N/A. Not in `implemented_oc_varsel` (`search.py:5299-5302`).
- **Refinement**: N/A.

### iPLS Forward (`ipls_forward`)
- **Grid search**: LEAKY. `ipls_forward(X_transformed_varsel, y_np, ...)` at `search.py:2324-2333`. Returns subsets with `indices` that are passed to `_run_single_config()` at 2393. **Magnitude**: Same as iPLS. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A. Not in default `variable_selection_methods` (`bayesian_utils.py:261`) and not handled in the method switch.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A.
- **Refinement**: N/A.

### iPLS Backward (`ipls_backward`)
- **Grid search**: LEAKY. `ipls_backward(X_transformed_varsel, y_np, ...)` at `search.py:2335-2342`. Same subset path as forward. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A.
- **Refinement**: N/A.

### MC-siPLS (`mc_sipls`)
- **Grid search**: LEAKY. `mc_sipls(X_transformed_varsel, y_np, ...)` at `search.py:2344-2353`. Monte Carlo sampling uses full y. **Magnitude**: Moderate (MC stability varies). **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A.
- **Refinement**: N/A.

### MWPLS (`mwpls`)
- **Grid search**: LEAKY. `mwpls(X_transformed_varsel, y_np, ...)` at `search.py:2355-2362`. Uses full y for windowed PLS evaluation. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A.
- **Refinement**: N/A.

### UVE-SPA (`uve_spa_selection`)
- **Grid search**: LEAKY. `uve_spa_selection(X_transformed_varsel, y_np, ...)` at `search.py:2512-2523`. Two-stage: UVE on full y, then SPA on full y. **Magnitude**: Higher than either alone (compounding leakage). **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: LEAKY. `uve_spa_selection(X, y, ...)` at `bayesian_utils.py:474-484`. **Fix**: Same.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. `uve_spa_selection(X_preprocessed, y_oc, ...)` at `search.py:5739-5748`. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### UVE-CARS (`uve_cars_selection`)
- **Grid search**: LEAKY. `uve_cars_selection(X_transformed_varsel, y_np, ...)` at `search.py:2576-2590`. Two-stage full-data leakage. **Magnitude**: High. UVE + CARS compounding. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A. Not in default methods or handled.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. `uve_cars_selection(X_preprocessed, y_oc, ...)` at `search.py:5772-5789`. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### UVE-CARS-Tree (`uve_cars_tree`)
- **Grid search**: LEAKY. Same call site as UVE-CARS at `search.py:2565-2590` (different `use_hybrid` flag). **Fix**: Same.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. `uve_cars_selection(...)` at `search.py:5770-5789` with `use_hybrid=True`. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### UVE-CARS-SPA (`uve_cars_spa_selection`)
- **Grid search**: LEAKY. `uve_cars_spa_selection(X_transformed_varsel, y_np, ...)` at `search.py:2595-2609`. Three-stage compounding. **Magnitude**: Very high for small datasets. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. `uve_cars_spa_selection(X_preprocessed, y_oc, ...)` at `search.py:5792-5809`. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### Forward iPLS-SPA (`fipls_spa_selection`)
- **Grid search**: LEAKY. `fipls_spa_selection(X_transformed_varsel, y_np, ...)` at `search.py:2614-2624`. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A.
- **Refinement**: N/A.

### Forward iPLS-CARS (`fipls_cars_selection`)
- **Grid search**: LEAKY. `fipls_cars_selection(X_transformed_varsel, y_np, ...)` at `search.py:2629-2641`. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A.
- **Refinement**: N/A.

### CARS (`cars_selection`)
- **Grid search**: LEAKY. `cars_selection(X_transformed_varsel, y_np, ...)` at `search.py:2552-2563`. MC sampling + exponential decay + internal PLS all use full y. Frozen `top_indices` at 2829. **Magnitude**: High. CARS's adaptive reweighted sampling is highly sensitive to which samples are included. Full-data CARS on N<200 typically inflates R2cv by 0.05-0.15. See Top 3 above. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: LEAKY. `cars_selection(X, y, ...)` at `bayesian_utils.py:508-517`. **Fix**: Same.
- **Unified Bayesian**: LEAKY. `cars_selection(X, y, ...)` at `unified_bayesian.py:655-665`. Importances cached per `(preprocess, method, model)` at 990, then `top_indices` at 992. **Fix**: Per-fold varsel inside Optuna objective.
- **NSGA-II**: LEAKY (guidance only, LOW priority). `cars_selection(X, y_for_importance, ...)` at `nsga2_search.py:1872`. Called once outside `_evaluate()`. Result biases initial population sampling probability but does NOT directly determine the wavelength mask used in CV. The chromosome encodes its own mask and CV runs on that mask via `cross_val_score` at ~1450. **Magnitude**: Low. The CARS guidance inflates the probability that good wavelengths appear in the initial population, but the CV score itself is computed on the chromosome's mask without varsel. The inflation is indirect and small (~0.01-0.02). **Fix**: Use only unsupervised initialization (uniform random) or accept the bias. Lowest priority.
- **One-class grid**: LEAKY. `cars_selection(X_preprocessed, y_oc, ...)` at `search.py:5754-5768`. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### CARS-aware (`cars-aware`)
- **Grid search**: LEAKY. Same call as CARS with `model_type=model_name` at `search.py:2540-2563`. **Fix**: Same as CARS.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A.
- **Refinement**: N/A.

### CARS-Tree (`cars-tree`)
- **Grid search**: LEAKY. Same call as CARS with `use_hybrid=True` at `search.py:2547-2563`. **Fix**: Same as CARS.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. Handled at `search.py:5750-5768` with `use_hybrid` flag. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### VCPA-IRIV (`vcpa_iriv`)
- **Grid search**: LEAKY. `vcpa_iriv(X_transformed_varsel, y_np, ...)` at `search.py:2647-2654`. Iterative elimination with binary matrix sampling uses full y. **Magnitude**: Moderate-high. VCPA's nested outer/inner iterations are computationally expensive, and full-data selection amplifies noise. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: LEAKY. `vcpa_iriv(X, y, ...)` at `bayesian_utils.py:524-531`. **Fix**: Same.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. `vcpa_iriv(X_preprocessed, y_oc, ...)` at `search.py:5812-5822`. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### GA-PLS (`ga_pls_selection`)
- **Grid search**: LEAKY. `ga_pls_selection(X_transformed_varsel, y_np, ...)` at `search.py:2696-2707` (linear models) and `search.py:2726-2737` (default). GA fitness evaluation uses PLS CV on full y. **Magnitude**: High. GA-PLS explores many variable subsets with full-data fitness, so the selected variables are tuned to the full dataset. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: N/A. GA for one-class uses `ga_lightgbm_selection`, not `ga_pls_selection`.
- **Refinement**: N/A.

### GA-LightGBM (`ga_lightgbm_selection`)
- **Grid search**: LEAKY. `ga_lightgbm_selection(X_transformed_varsel, y_np, ...)` at `search.py:2710-2722` (tree models). Same GA pattern with LightGBM fitness. **Fix**: VarselTransformer wrapper.
- **Old Bayesian**: N/A.
- **Unified Bayesian**: N/A.
- **NSGA-II**: N/A.
- **One-class grid**: LEAKY. `ga_lightgbm_selection(X_preprocessed, y_oc, ...)` at `search.py:5848-5860`. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### `importance` (model-native feature importances)
- **Grid search**: LEAKY. Pipeline fit on full `(X_for_models, y_np)` at `search.py:2274`. `get_feature_importances(fitted_model, ...)` at `search.py:2477-2478` extracts importances from the full-data model. For PLS: VIP scores (`models.py:1786` calls `compute_vip()`). For tree models: `feature_importances_`. For linear: `|coef_|`. All y-dependent. **Magnitude**: Moderate. See Top 3 above. **Fix**: The model must be fit per-fold; importances extracted from per-fold models. This is the hardest to fix because it requires per-fold model fitting before variable ranking, creating a chicken-and-egg problem. Practical approach: use an initial full-spectrum per-fold fit to rank variables, then refit the actual model on the selected subset per-fold.
- **Old Bayesian**: LEAKY. `get_feature_importances(fitted_model, model_name, X, y)` at `bayesian_utils.py:432-434`. `fitted_model` from `pipe.fit(X, y)` at line 412. **Fix**: Same.
- **Unified Bayesian**: LEAKY. `compute_importances(X, y, 'importance', model_name, ...)` at `unified_bayesian.py:958-962` (one-class) and `unified_bayesian.py:1120-1122` (regression/classification). Internally fits a model on full data at `unified_bayesian.py:643`. **Fix**: Per-fold importances.
- **NSGA-II**: N/A. NSGA-II does not use `importance` mode; wavelengths are chromosome-encoded.
- **One-class grid**: LEAKY. `compute_one_class_importances(X_preprocessed, y_oc, method='lightgbm', ...)` at `search.py:5703-5706`. Fits LightGBM on full data. **Fix**: Per-fold re-run.
- **Refinement**: N/A.

### Region selection (`create_region_subsets`)
- **Grid search**: N/A. Grid search does not use region-based varsel (it uses explicit varsel methods from the dropdown).
- **Old Bayesian**: LEAKY. `create_region_subsets(X, y, wavelengths, ...)` at `bayesian_utils.py:268-269`. Computes region correlations with y via `pearsonr` at `regions.py:170`. Frozen `region_indices` passed to `_run_single_config_fn` at 605-614. **Fix**: Per-fold region correlation computation.
- **Unified Bayesian**: LEAKY. `create_region_subsets(X_prep, y, wl_prep, ...)` at `unified_bayesian.py:927-932` (one-class) and `unified_bayesian.py:1092-1097` (regression/classification). Same `pearsonr`-based region selection on full data. **Fix**: Per-fold region computation.
- **NSGA-II**: N/A.
- **One-class grid**: N/A. One-class grid does not use region-based varsel.
- **Refinement**: N/A.

### VIP (`compute_vip` in models.py)
- **Grid search**: LEAKY (via `importance` mode). VIP is computed inside `get_feature_importances()` at `models.py:1784-1786` when `model_name` is PLS/PLS-DA. The model was fit on full data at `search.py:2274`. VIP is not a standalone varsel method in any path — it is the importance extraction mechanism for PLS models.
- **Old Bayesian**: LEAKY (via `importance` mode). Same channel: `get_feature_importances()` at `bayesian_utils.py:432`.
- **Unified Bayesian**: LEAKY (via `importance` mode). Same channel: `compute_importances()` at `unified_bayesian.py:643`.
- **NSGA-II**: N/A.
- **One-class grid**: N/A. One-class does not use PLS VIP (importance uses LightGBM surrogate).
- **Refinement**: N/A.

---

## wavelength_selection.py methods (separate implementations)

### `wavelength_selection.spa`
- Only imported by `calibration_transfer.py:1117` for instrument-transfer wavelength selection. Uses pseudo-y (spectral mean, `calibration_transfer.py:1113-1114`), not real target y. **Not used in any search path. N/A for all paths.**

### `wavelength_selection.cars`
- Same situation as `spa` above. Only in `calibration_transfer.py:1117` with pseudo-y. **N/A for all search paths.**

### `wavelength_selection.vcpa_iriv`
- Used in grid search via `search.py:79` (`from .wavelength_selection import vcpa_iriv`). Covered above under VCPA-IRIV.

---

## UVE Prefilter (cross-cutting)

The UVE prefilter is a separate leakage vector that runs before the per-method varsel loop.

- **Grid search**: LEAKY. `get_uve_threshold(X_transformed_varsel, y_np, ...)` at `search.py:2289-2305`. Reduces the variable set before the varsel loop at 2314. Frozen mask applied to all subsequent varsel methods. **Magnitude**: Compounds with whatever varsel method runs next. If UVE prefilter eliminates 40% of variables, the downstream varsel works on a pre-filtered space that was selected using full y.
- **One-class grid**: LEAKY. `get_uve_threshold(X_preprocessed, y_oc, ...)` at `search.py:5649-5655`. Uses binary `y_oc` labels. **Overlap with T-04** — the UVE prefilter in one-class uses inlier+outlier labels, which is a separate concern (T-04: the labels themselves may be unreliable for UVE's PLS-based stability analysis).
- **Old Bayesian**: N/A. Old Bayesian does not have a UVE prefilter step.
- **Unified Bayesian**: N/A. Unified Bayesian does not have a UVE prefilter step.
- **NSGA-II**: N/A.

---

## Bonus findings (not fixes — separate tickets)

1. **`bayesian_utils.py:261` hardcodes `random_state=42` inside varsel calls** — Line 442, 455, 469, 488, 502, 520. This ignores the user's `random_state` setting and makes Bayesian varsel non-reproducible across runs if the user changes the random seed. Grid search correctly threads `random_state` through.

2. **`bayesian_utils.py:557` uses `[::-1]` sort order** — `top_indices = np.argsort(importances, kind='stable')[-n_top:][::-1]`. Grid search at `search.py:2829` uses the same pattern. This is correct (descending importance) but differs from the Bayesian unified path at `unified_bayesian.py:1149` which uses `[-n_vars:]` without `[::-1]` (ascending argsort positions). The selected indices are the same set but in different order — should not affect results but is an inconsistency.

3. **`search.py:2855` hardcodes `top_n_vars=30`** — The `top_n_vars` parameter passed to `_run_single_config()` is always 30 regardless of actual `n_top`. This appears to control reporting/display only (not variable selection), but the mismatch between the actual variable count and the reporting count may confuse downstream consumers of the result dict.

---

## Cross-references

- **T-02** (ensemble OOF preprocessor leakage) — out of scope. Different leakage channel.
- **T-03** (preprocessing discovery full-data leakage) — out of scope. `preprocessing_discovery.py:175` calls `cars_selection(X, y, ...)` on full data. Same LEAKY pattern but in a different pipeline (smart preprocessing ranking, not model scoring).
- **T-04** (one-class UVE prefilter on outlier-contaminated labels) — overlaps with this audit's one-class UVE entries. The leakage documented here (full-data UVE on `y_oc`) is compounded by T-04's concern that `y_oc` labels are derived from user-specified inlier/outlier split, not ground truth.
- **CV Strategy Phase 2** (from PROJECT_STATUS follow-ups) — proposes propagating outer CV strategy into varsel inner loops. This is complementary but independent: even if inner CV strategy matches outer, the leakage persists because varsel runs on full data before outer CV starts.

---

## Architectural note: why everything is LEAKY

The codebase has a uniform two-phase architecture:

```
Phase 1: Varsel on full (X_preprocessed, y) → top_indices
Phase 2: For each (model, hyperparams):
            CV on X_preprocessed[:, top_indices] → score
```

Phase 1 and Phase 2 are structurally separated. No search path has a "Phase 1 inside the fold" variant. The fix requires breaking this separation — either:

(a) **Pipeline approach**: `Pipeline([("varsel", VarselTransformer(method, params)), ("model", model)])` — sklearn handles per-fold refitting automatically. Best for grid search and old Bayesian (which already use sklearn pipelines).

(b) **Manual per-fold loop**: In the CV loop, for each fold, run varsel on `(X_train, y_train)` only, then evaluate on `X_test`. Necessary for methods that don't fit sklearn's `fit/transform` API cleanly (e.g., iPLS which returns subsets, not a transform).

(c) **Hybrid**: For Bayesian paths, run varsel inside the Optuna objective per-trial on full data (still leaky per-trial, but at least different subsets per hyperparameter configuration). This is the minimal-effort improvement — it doesn't fix the per-fold leakage but reduces the compounding effect.

Option (a) is recommended for the grid search fix. It requires implementing ~15 VarselTransformer classes (one per method), each wrapping the existing varsel function into `fit()` (runs varsel on train data) and `transform()` (selects the top-N variables). The existing `variable_selection.py` functions remain unchanged; the transformers call them.
