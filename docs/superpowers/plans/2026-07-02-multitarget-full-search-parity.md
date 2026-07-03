# Multi-Target Full Search Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the T-17 Multi-Target tab full parity with the single-Y search — a grid over preprocessing × variable-selection × model × hyperparameters, joint multi-Y CV, ranked by joint Q², shown as a real leaderboard — by adding a NEW orchestration layer above the F2 seed evaluator, never touching `run_search`.

**Architecture:** A new grid orchestrator (`multitarget_grid.py`) enumerates a basic preprocessing grid (mirroring `search.py`), builds variable-selection subsets via two adapters (interval-subset + importance-top-N) with a skip-with-notice list for methods that are not multi-Y-safe, expands `get_model_grids` with the user's FULL per-model hyperparameter overrides, and evaluates every (preprocess × varsel × model × hp) cell through a shared `_evaluate_multitarget_cell` helper that wraps `multi_y_cv_pool` + `multi_y_metrics` and sinks any failure to `joint_q2=np.nan`. The GUI tab is reworked to inherit the normal Run config, dispatch on a worker thread with a separate `SearchController` + Cancel, and render the leaderboard. `run_search`'s single-Y path is byte-identical throughout.

**Tech Stack:** Python 3.12 (`.venv312`), numpy, pandas, scikit-learn, catboost, xgboost, lightgbm, Tkinter, pytest.

## Global Constraints

Every task's requirements implicitly include this section. Copy values verbatim.

- **Python 3.12 / `.venv312` only.** Run all tests as `.venv312/Scripts/python.exe -m pytest ...`. Never use `.venv311` (removed).
- **`run_search` single-Y byte-identity is SACRED.** Never edit `run_search` or its single-Y path in `src/spectral_predict/search.py`. Reuse only pure primitives from it by import (`build_preprocessing_pipeline`, `_apply_edge_mask_to_data`, `_get_edge_zone_size`). Gold fixtures `tests/gold_standards/varsel_single_y.npz` + `TestSingleYByteIdentityGold` and `tests/test_vip_formula.py` must stay green.
- **Grid engine ONLY.** The orchestrator asserts `optimization_method == "grid"` (Bayesian/NSGA-II are 1-D-only). Keep the GUI grey-out + force-to-grid.
- **NaN-sink discipline.** A failed cell returns `MultiTargetResult(..., joint_q2=np.nan, error=<str>)` — NEVER a finite `0.0`. Ranking is NaN-safe (push non-finite to the bottom). Never introduce a finite sentinel; the only allowed `mean_nmse > 0.0 else 0.0` idiom sites are `variable_selection.py:80` and `ga_pls.py:223`, both already finiteness-guarded.
- **Chemometrics conventions are NOT bugs.** Full-X per-spectrum SNV/SG/baseline, varsel-on-full-calibration, and autoscale-before-CV (column StandardScaler on X, distinct from the per-fold JOINT `FoldYScaler` on Y) are community convention. Do not "fix" them as leakage.
- **`_reject_multi_y` hardening fires ONLY on 2-D Y** (`arr.ndim == 2 and arr.shape[1] > 1`). Single-Y and single-column-2-D callers are unaffected.
- **Dataclass fields are append-only.** Add new `MultiTargetResult` / `MultiTargetSearchOutput` fields with defaults at the END; never reorder or remove existing fields.
- **No new dependencies.** Never edit `.env`. Never commit data files.
- **Commit after each task.** Before ANY git operation re-run `git branch --show-current` and confirm it prints `feat/T17-multitarget-regression`. Never force-push/reset shared refs. Do NOT merge PR #63.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/spectral_predict/multitarget_search.py` | Modify | F2 seed evaluator. Task 1 (dataclass fields), Task 2 (full-HP `build_multitarget_estimator`), Task 4 (`_evaluate_multitarget_cell` + F2 refactor). |
| `src/spectral_predict/variable_selection.py` | Modify | Task 3: add `_reject_multi_y` guard to the single-Y flattener `ipls_selection` (line 630). |
| `src/spectral_predict/wavelength_selection.py` | Modify | Task 3: add `_reject_multi_y`-style guard to `vcpa_iriv` (line 352). |
| `src/spectral_predict/multitarget_grid.py` | **Create** | NEW orchestration layer: preprocess enumerator (Task 5), varsel adapters + classifier + cache + spa gate (Tasks 6–10), `run_multitarget_grid_search` (Task 11). Imports pure primitives from `search.py`, `preprocess.py`, `models.py`, `variable_selection.py`, `ga_pls.py`, `multi_y.py`, and F2 helpers from `multitarget_search.py`. |
| `spectral_predict_gui_optimized.py` | Modify | Task 12 (rework `_run_multitarget_search` → inherited config + worker thread + separate controller + Cancel + progress + heads-up), Task 13 (leaderboard columns + skip-notice + CSV export). |
| `tests/test_multitarget_search.py` | Modify | Tasks 1, 2, 4 tests (extend existing file). |
| `tests/test_multitarget_varsel.py` | **Create** | Tasks 3, 6, 7, 8, 9, 10 tests. |
| `tests/test_multitarget_grid.py` | **Create** | Tasks 5, 11 tests. |
| `tests/gui/test_multitarget_tab.py` | Modify | Tasks 12, 13 tests (extend existing file). |
| `tests/test_multitarget_integration.py` | **Create** | Task 14 end-to-end + save/reload/predict round-trip. |

**Verified anchors (confirmed against real code this session):**
- `build_multitarget_estimator(strategy, params, n_samples, n_features)` — `multitarget_search.py:236`. JOINT builders (PLS `:283`, RF `:290`, MLP `:305`, CatBoost `:320`, XGBoost `:340`) use explicit `params.get(...)` and currently DROP keys; INDEPENDENT builders (`Ridge :398`, `_build_independent_base :417`) already pass params through with `**p`.
- `MultiTargetResult` — `multitarget_search.py:459` (dataclass, NOT frozen). `MultiTargetSearchOutput` — `:493`. F2 NaN-safe sort — `:641`. Grid guard — `:562`.
- `run_multitarget_search(...)` F2 seed API — `multitarget_search.py:516`, builds splitter once + `min_fold_train = min(len(train_idx) ...)` at `:601`, per-config loop `:604`.
- `multi_y_cv_pool(estimator, X, Y, cv, *, scale_y=True, fit_params=None, task_type="regression", n_folds=5, n_repeats=5, random_state=42)` — `multi_y.py:136`. `multi_y_metrics(Y_true, Y_pred, target_names=None)` — `:248`. `cap_components(n_samples, n_features, requested=None)` — `:474`. `aggregate_importance(matrix, rule="mean")` — `:408`.
- `get_model_grids(task_type, n_features, max_n_components=10, max_iter=500, ...tier='standard', enabled_models=None, n_jobs=-1)` — `models.py:542`; per-model `params_dict` keys (verified): PLS `{n_components,max_iter,tol}`; RF `{n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features,bootstrap,max_leaf_nodes,min_impurity_decrease}`; MLP `{hidden_layer_sizes,alpha,learning_rate_init,activation,solver,batch_size,learning_rate}` + `momentum` only when `solver=='sgd'`; CatBoost `{iterations,learning_rate,depth,l2_leaf_reg,border_count,random_strength,min_data_in_leaf,bootstrap_type}` + `bagging_temperature` only when `bootstrap_type=='Bayesian'`; XGBoost `{n_estimators,learning_rate,max_depth,subsample,colsample_bytree,reg_alpha,reg_lambda,min_child_weight,gamma,tree_method}`. `get_feature_importances(model, model_name, X, y)` — `models.py:1798` (fitted model → 1-D importances).
- Varsel (all in `variable_selection.py` unless noted): `_reject_multi_y(y, method)` `:34` (raises `NotImplementedError` only on 2-D Y with >1 col); `_prep_varsel_y(y)` `:17` (ravels 1-D / single-col, preserves 2-D); interval methods (multi-Y-safe, `wavelengths` is a REQUIRED positional): `ipls_forward(X, y, wavelengths, n_intervals=20, max_combine=5, cv_folds=5, random_state=42)` `:1806`, `ipls_backward(X, y, wavelengths, n_intervals=20, cv_folds=5, random_state=42, min_intervals=1)` `:1994`, `mc_sipls(X, y, wavelengths, n_intervals=20, n_combinations=500, max_combine=4, cv_folds=5, random_state=42)` `:2198`, `mwpls(X, y, wavelengths, window_sizes=None, step_size=None, cv_folds=5)` `:2321` — each returns `list[dict]` keyed `{'indices','tag','rmsecv','r2',...}`; `spa_selection(X, y, n_features, cv_folds=5)` `:430` (calls `_prep_varsel_y`, returns importance array shape `(n_features_in,)`); `fipls_spa_selection(X, y, wavelengths, ...)` `:1119` ravels Y at `:1156`; `ipls_selection(X, y, ...)` `:569` (LEGACY single-Y) ravels Y unconditionally at `:630`. `ga_pls_selection(X, y, ..., task_type='regression', ...)` — `ga_pls.py:499`, returns per-wavelength frequency array (2-D-safe via `multi_y`, PLS-fitness/linear-only, NO `wavelengths`/`models` param). `vcpa_iriv(X, y, ..., model_type=None, random_state=None)` — `wavelength_selection.py:352` (single-Y-only; `compute_rmsecv` at `:469` returns `np.inf` on 2-D Y via the bare-except at `:501`).
- `search.py`: preprocess enumeration `2288–2616` (list `preprocess_configs`; config-dict keys `name,deriv,window,polyorder,interference,baseline_method,baseline_params,smoothing,smoothing_window,smoothing_polyorder`; baseline doubling `:2565`, smoothing doubling `:2582`, autoscale doubling `:2603`); `build_preprocessing_pipeline` call `:2745` (returns step list, wrapped in `Pipeline` at `:2766`); wavelength restriction `2777–2855`, `wavelength_restriction_active` flag `:2864`, skip-edge-mask-when-restricted `:2876`; edge-mask helpers `_apply_edge_mask_to_data(X, wavelengths, preprocess_cfg)` `:227`, `_get_edge_zone_size(preprocess_cfg)` `:200`; varsel top-N `variable_counts` (LIST OF INTS, default `[10,20,50,100,250,500,1000]`) filtered `n < n_features` `:3695`, stable-argsort top-N `:3766`. `build_preprocessing_pipeline(preprocess_name, deriv, window, polyorder, ..., baseline_method, baseline_params, smoothing, smoothing_window, smoothing_polyorder, autoscale)` — `preprocess.py:286`.
- GUI: `_MULTITARGET_METRIC_KEYS` `:14513` (`(("r2","R²"),("rmse","RMSE"),("rpd","RPD"),("rer","RER"),("ccc","CCCcv"),("bias","Bias"))`); `_run_multitarget_search` `:14759` (currently synchronous, `configs=[{"model_name":name,"params":{}}]`); sample mask `:14802–14818`; `_populate_multitarget_results` `:14849`; `_export_multitarget_csv` `:14882`. Single-Y `run_search(...)` grid call + full HP override kwargs `:29135–29300`; `selected_varsel_methods` `:28125–28166`; `variable_counts` list build `:26359`; `ipls_subset_limit` StringVar `:3870` (values "Top 5"/"Top 10"/"Top 20"/"All"); worker spawn `:24401` (`threading.Thread(target=self._run_analysis_thread, daemon=True)`), `self.search_controller = SearchController()` `:24263`, `_stop_search` `:23840` (`self.search_controller.stop()`); `_progress_callback_impl` `:29513` expects `info={'message','current','total','best_model'}` with `best_model={'Model','Preprocess','Deriv','RMSEcv','R2cv','top_vars'}` for the regression path; Tk writes via `self.root.after(0, ...)`.

---

### Task 1: Extend result dataclasses (append-only fields)

**Files:**
- Modify: `src/spectral_predict/multitarget_search.py:459-513` (add fields to `MultiTargetResult` and `MultiTargetSearchOutput`)
- Test: `tests/test_multitarget_search.py` (append)

**Interfaces:**
- Produces: `MultiTargetResult` gains `preprocessing: str = "raw"`, `varsel_method: str = "full"`, `varsel_tag: str = "full"`, `n_variables: int | None = None`, `error: str | None = None`. `MultiTargetSearchOutput` gains `skipped: list[str] = field(default_factory=list)`. (`field` is already imported at `:47`.) All later tasks depend on these names.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_search.py`:

```python
def test_multitarget_result_has_appendonly_fields():
    from spectral_predict.multitarget_search import MultiTargetResult

    r = MultiTargetResult(
        model_name="PLS", mode="JOINT", params={}, joint_q2=0.5, metrics={},
        precise_note="", scale_y=True, mechanism="x",
    )
    # New append-only fields default sanely.
    assert r.preprocessing == "raw"
    assert r.varsel_method == "full"
    assert r.varsel_tag == "full"
    assert r.n_variables is None
    assert r.error is None


def test_multitarget_output_skipped_defaults_empty_list():
    from spectral_predict.multitarget_search import MultiTargetSearchOutput

    o1 = MultiTargetSearchOutput(results=[], target_names=["a"], correlation={}, n_targets=1)
    o2 = MultiTargetSearchOutput(results=[], target_names=["b"], correlation={}, n_targets=1)
    assert o1.skipped == []
    o1.skipped.append("uve")
    assert o2.skipped == []  # default_factory => not a shared mutable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_search.py::test_multitarget_result_has_appendonly_fields tests/test_multitarget_search.py::test_multitarget_output_skipped_defaults_empty_list -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword` is NOT raised because we don't pass them, but `AttributeError: 'MultiTargetResult' object has no attribute 'preprocessing'`).

- [ ] **Step 3: Write minimal implementation**

In `MultiTargetResult` (after `y_pred_pooled: Optional[np.ndarray] = None` at `:490`):

```python
    preprocessing: str = "raw"
    varsel_method: str = "full"
    varsel_tag: str = "full"
    n_variables: Optional[int] = None
    error: Optional[str] = None
```

In `MultiTargetSearchOutput` (after `n_targets: int` at `:508`, BEFORE the `best` property):

```python
    skipped: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_search.py::test_multitarget_result_has_appendonly_fields tests/test_multitarget_search.py::test_multitarget_output_skipped_defaults_empty_list -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must print feat/T17-multitarget-regression
git add src/spectral_predict/multitarget_search.py tests/test_multitarget_search.py
git commit -m "feat(T17): append-only fields on MultiTargetResult/SearchOutput"
```

---

### Task 2: Full hyperparameter consumption in `build_multitarget_estimator`

**Files:**
- Modify: `src/spectral_predict/multitarget_search.py:283-363` (PLS, RandomForest, MLP, CatBoost, XGBoost branches)
- Test: `tests/test_multitarget_search.py` (append)

**Interfaces:**
- Consumes: `build_multitarget_estimator(strategy, params, n_samples, n_features)` (signature unchanged).
- Produces: the JOINT builders now consume the FULL `params_dict` `get_model_grids` emits (per the verified key lists). XGBoost already consumes all its keys — leave it. INDEPENDENT builders already pass params through `**p` — leave them.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_search.py`:

```python
def test_pls_consumes_max_iter_and_tol():
    strat = resolve_multitarget_strategy("PLS")
    est = build_multitarget_estimator(
        strat, {"n_components": 3, "max_iter": 321, "tol": 1e-4}, n_samples=40, n_features=25
    )
    p = est.get_params()
    assert p["max_iter"] == 321
    assert p["tol"] == 1e-4


def test_rf_consumes_full_hp_set():
    strat = resolve_multitarget_strategy("RandomForest")
    est = build_multitarget_estimator(
        strat,
        {"n_estimators": 50, "min_samples_split": 5, "bootstrap": False,
         "max_leaf_nodes": 7, "min_impurity_decrease": 0.01},
        n_samples=40, n_features=12,
    )
    p = est.get_params()
    assert p["min_samples_split"] == 5
    assert p["bootstrap"] is False
    assert p["max_leaf_nodes"] == 7
    assert p["min_impurity_decrease"] == 0.01


def test_mlp_consumes_activation_solver_batch_and_momentum():
    strat = resolve_multitarget_strategy("MLP")
    est = build_multitarget_estimator(
        strat,
        {"activation": "tanh", "solver": "sgd", "batch_size": 16,
         "learning_rate": "adaptive", "momentum": 0.5},
        n_samples=40, n_features=12,
    )
    p = est.get_params()
    assert p["activation"] == "tanh"
    assert p["solver"] == "sgd"
    assert p["batch_size"] == 16
    assert p["learning_rate"] == "adaptive"
    assert p["momentum"] == 0.5


def test_catboost_consumes_border_random_strength_bootstrap_type():
    strat = resolve_multitarget_strategy("CatBoost")
    est = build_multitarget_estimator(
        strat,
        {"border_count": 32, "random_strength": 2.0, "bootstrap_type": "Bernoulli"},
        n_samples=40, n_features=12,
    )
    p = est.get_params()
    assert p["border_count"] == 32
    assert p["random_strength"] == 2.0
    assert p["bootstrap_type"] == "Bernoulli"
    # bagging_temperature is INCOMPATIBLE with a non-Bayesian bootstrap_type: must not be set.
    assert p.get("bagging_temperature") in (None, 1.0)


def test_catboost_bagging_temperature_only_with_bayesian():
    strat = resolve_multitarget_strategy("CatBoost")
    est = build_multitarget_estimator(
        strat, {"bootstrap_type": "Bayesian", "bagging_temperature": 3.0},
        n_samples=40, n_features=12,
    )
    assert est.get_params()["bagging_temperature"] == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_search.py -k "consumes or bagging" -v`
Expected: FAIL (`assert p["min_samples_split"] == 5` fails — RF ignores it; MLP/CatBoost likewise).

- [ ] **Step 3: Write minimal implementation**

Replace the PLS branch (`:283-288`):

```python
    if name == "PLS":
        from sklearn.cross_decomposition import PLSRegression

        requested = int(params.get("n_components", 10))
        n_components = cap_components(n_samples, n_features, requested)
        pls_kwargs: dict[str, Any] = dict(n_components=n_components, scale=False)
        if "max_iter" in params:
            pls_kwargs["max_iter"] = int(params["max_iter"])
        if "tol" in params:
            pls_kwargs["tol"] = float(params["tol"])
        return PLSRegression(**pls_kwargs)
```

Replace the RandomForest branch (`:290-303`):

```python
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", 1.0),
            bootstrap=bool(params.get("bootstrap", True)),
            max_leaf_nodes=params.get("max_leaf_nodes", None),
            min_impurity_decrease=float(params.get("min_impurity_decrease", 0.0)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
        )
```

Replace the MLP branch (`:305-318`):

```python
    if name == "MLP":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64,)),
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            alpha=float(params.get("alpha", 1e-3)),
            batch_size=params.get("batch_size", "auto"),
            learning_rate=params.get("learning_rate", "constant"),
            learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
            momentum=float(params.get("momentum", 0.9)),
            max_iter=int(params.get("max_iter", 500)),
            early_stopping=bool(params.get("early_stopping", True)),
            random_state=int(params.get("random_state", 42)),
        )
```

Replace the CatBoost branch (`:320-338`). Note: `bagging_temperature` is only valid with `bootstrap_type='Bayesian'`; pass it ONLY when present in params (the grid emits it only for Bayesian):

```python
    if name == "CatBoost":
        from catboost import CatBoostRegressor

        kwargs: dict[str, Any] = dict(
            iterations=int(params.get("iterations", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            depth=int(params.get("depth", 5)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 4.0)),
            min_data_in_leaf=int(params.get("min_data_in_leaf", 1)),
            random_state=int(params.get("random_state", 42)),
            verbose=False,
        )
        if "border_count" in params:
            kwargs["border_count"] = int(params["border_count"])
        if "random_strength" in params:
            kwargs["random_strength"] = float(params["random_strength"])
        if "bootstrap_type" in params:
            kwargs["bootstrap_type"] = params["bootstrap_type"]
        if "bagging_temperature" in params:
            kwargs["bagging_temperature"] = float(params["bagging_temperature"])
        kwargs.update(strategy.joint_params)  # loss_function='MultiRMSE'
        return CatBoostRegressor(**kwargs)
```

Leave the XGBoost branch (`:340-363`) unchanged — it already consumes all keys.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_search.py -k "consumes or bagging or build_pls or build_joint" -v`
Expected: PASS (existing `test_build_pls_estimator_scale_false_and_capped`, `test_build_joint_estimator_fits_scaled_3target_block`, and the new tests all pass).

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_search.py tests/test_multitarget_search.py
git commit -m "feat(T17): full per-model HP consumption in build_multitarget_estimator"
```

---

### Task 3: Fail-loud `_reject_multi_y` hardening on the two genuine flatteners

**Files:**
- Modify: `src/spectral_predict/variable_selection.py:628-630` (guard `ipls_selection`)
- Modify: `src/spectral_predict/wavelength_selection.py` (guard `vcpa_iriv` near its input validation `~:455`)
- Test: `tests/test_multitarget_varsel.py` (create)

**Interfaces:**
- Consumes: `_reject_multi_y(y, method)` (`variable_selection.py:34`).
- Produces: `ipls_selection` and `vcpa_iriv` raise `NotImplementedError` on genuine 2-D Y (`ndim==2 and shape[1]>1`) instead of silently flattening; single-Y and single-column-2-D behavior byte-identical.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multitarget_varsel.py`:

```python
"""Tests for the T-17 multi-target varsel adapters, guards, and grid orchestration."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20260702)


@pytest.fixture
def xy_multi(rng):
    n, p = 50, 40
    X = rng.standard_normal((n, p))
    base = X[:, :4] @ rng.standard_normal((4, 3))
    Y = base + 0.05 * rng.standard_normal((n, 3))
    wl = np.linspace(1000.0, 2000.0, p)
    return X, Y, wl


def test_ipls_selection_rejects_2d_y(xy_multi):
    from spectral_predict.variable_selection import ipls_selection

    X, Y, _wl = xy_multi
    with pytest.raises(NotImplementedError):
        ipls_selection(X, Y)


def test_ipls_selection_single_y_still_works(xy_multi):
    from spectral_predict.variable_selection import ipls_selection

    X, Y, _wl = xy_multi
    # single-column 2-D and 1-D must NOT raise (guard fires only on >1 column).
    out_1d = ipls_selection(X, Y[:, 0])
    out_col = ipls_selection(X, Y[:, [0]])
    assert out_1d is not None
    assert out_col is not None


def test_vcpa_iriv_rejects_2d_y(xy_multi):
    from spectral_predict.wavelength_selection import vcpa_iriv

    X, Y, _wl = xy_multi
    with pytest.raises(NotImplementedError):
        vcpa_iriv(X, Y, n_outer_iterations=1, n_inner_iterations=2, binary_matrix_samples=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "rejects_2d or single_y_still" -v`
Expected: FAIL (`ipls_selection`/`vcpa_iriv` silently ravel; no `NotImplementedError`).

- [ ] **Step 3: Write minimal implementation**

In `variable_selection.py`, replace the flatten at `:628-630`:

```python
    # Convert inputs to numpy arrays and ensure proper shapes
    X = np.asarray(X)
    _reject_multi_y(y, "ipls (legacy importance path)")
    y = np.asarray(y).ravel()
```

In `wavelength_selection.py`, add an import near the top (with the other `from .` imports) and a guard in `vcpa_iriv` right after the existing sample-count check (`if X.shape[0] != y.shape[0]: ...` at `~:455`):

```python
from .variable_selection import _reject_multi_y
```

```python
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        _reject_multi_y(y, "vcpa-iriv")
```

If a top-level import of `variable_selection` would create a circular import, instead do a local import inside `vcpa_iriv`: `from .variable_selection import _reject_multi_y` on the line before the guard.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "rejects_2d or single_y_still" -v`
Expected: PASS

Then confirm single-Y varsel gold identity is intact:
Run: `.venv312/Scripts/python.exe -m pytest tests/ -k "SingleYByteIdentityGold or vip_formula" -v`
Expected: PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/variable_selection.py src/spectral_predict/wavelength_selection.py tests/test_multitarget_varsel.py
git commit -m "fix(T17): fail-loud _reject_multi_y guard on ipls_selection + vcpa_iriv"
```

---

### Task 4: Shared `_evaluate_multitarget_cell` helper + F2 refactor (exception→NaN)

**Files:**
- Modify: `src/spectral_predict/multitarget_search.py` (add `_evaluate_multitarget_cell`; refactor `run_multitarget_search` loop `:604-635` to call it; export the helper in `__all__` `:60`)
- Test: `tests/test_multitarget_search.py` (append; update the F2 abort-behavior expectation)

**Interfaces:**
- Consumes: `resolve_multitarget_strategy`, `build_multitarget_estimator`, `multi_y_cv_pool`, `multi_y_metrics`.
- Produces: `_evaluate_multitarget_cell(X_sub, Y, model_name, params, splitter, min_fold_train, n_features_sub, target_names, *, n_folds=5, n_repeats=5, random_state=42, preprocessing="raw", varsel_method="full", varsel_tag="full") -> MultiTargetResult`. On ANY exception returns a result with `joint_q2=np.nan` and `error=str(exc)` (never a finite 0.0).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_search.py`:

```python
def test_evaluate_cell_returns_nan_on_failure():
    from spectral_predict.multitarget_search import _evaluate_multitarget_cell

    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 8))
    Y = rng.standard_normal((30, 2))
    splitter = build_cv_splitter("kfold", 3, "regression", n_repeats=1, random_state=42, y=None)
    # An impossible param forces the estimator to raise inside CV -> NaN sink.
    res = _evaluate_multitarget_cell(
        X, Y, "SVR", {"C": -5.0}, splitter, min_fold_train=20,
        n_features_sub=8, target_names=["a", "b"],
    )
    assert np.isnan(res.joint_q2)
    assert res.error is not None
    assert res.model_name == "SVR"


def test_evaluate_cell_success_carries_metadata():
    from spectral_predict.multitarget_search import _evaluate_multitarget_cell

    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 10))
    base = X[:, :3] @ rng.standard_normal((3, 2))
    Y = base + 0.05 * rng.standard_normal((40, 2))
    splitter = build_cv_splitter("kfold", 3, "regression", n_repeats=1, random_state=42, y=None)
    res = _evaluate_multitarget_cell(
        X, Y, "PLS", {"n_components": 4}, splitter, min_fold_train=25,
        n_features_sub=10, target_names=["a", "b"],
        preprocessing="snv", varsel_method="ipls_forward", varsel_tag="fwd_iPLS_1000-1100nm",
    )
    assert res.error is None
    assert res.preprocessing == "snv"
    assert res.varsel_method == "ipls_forward"
    assert res.varsel_tag == "fwd_iPLS_1000-1100nm"
    assert res.n_variables == 10
    assert res.metrics["q2"].shape == (2,)
```

Also update the F2 error-behavior expectation. Add:

```python
def test_run_multitarget_search_sinks_bad_config_to_nan_not_abort():
    """F2 refactor: a failing cell no longer aborts the whole search; it sinks to NaN."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 10))
    base = X[:, :3] @ rng.standard_normal((3, 2))
    Y = base + 0.05 * rng.standard_normal((40, 2))
    out = run_multitarget_search(
        X, Y,
        [{"model_name": "SVR", "params": {"C": -1.0}},   # bad -> NaN
         {"model_name": "PLS", "params": {"n_components": 3}}],  # good
        cv="kfold", n_folds=3, target_names=["a", "b"],
    )
    assert len(out.results) == 2
    # Good config ranks first; bad config sunk to the bottom with NaN + error.
    assert out.best.model_name == "PLS"
    assert np.isnan(out.results[-1].joint_q2)
    assert out.results[-1].error is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_search.py -k "evaluate_cell or sinks_bad_config" -v`
Expected: FAIL (`_evaluate_multitarget_cell` does not exist; current F2 loop raises out of the search on the bad SVR config).

- [ ] **Step 3: Write minimal implementation**

Add `"_evaluate_multitarget_cell"` to `__all__` at `:60`. Add the helper above `run_multitarget_search` (after `MultiTargetSearchOutput`):

```python
def _evaluate_multitarget_cell(
    X_sub: Any,
    Y: Any,
    model_name: str,
    params: dict[str, Any],
    splitter: Any,
    min_fold_train: int,
    n_features_sub: int,
    target_names: list[str],
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    preprocessing: str = "raw",
    varsel_method: str = "full",
    varsel_tag: str = "full",
) -> MultiTargetResult:
    """Evaluate one (preprocess, varsel-subset, model, hp) cell; NaN-sink on failure.

    ``n_features_sub`` MUST be ``X_sub.shape[1]`` (the subset feature count) so PLS
    component capping cannot over-request. Any exception (degenerate subset,
    non-finite fold, sklearn ValueError) returns ``joint_q2=np.nan`` + ``error`` —
    never a finite 0.0 — so the NaN-safe rank sinks it.
    """
    strategy = resolve_multitarget_strategy(model_name)
    try:
        estimator = build_multitarget_estimator(
            strategy, params, min_fold_train, n_features_sub
        )
        y_true, y_pred = multi_y_cv_pool(
            estimator, X_sub, Y, splitter,
            scale_y=strategy.scale_y, n_folds=n_folds,
            n_repeats=n_repeats, random_state=random_state,
        )
        metrics = multi_y_metrics(y_true, y_pred, target_names=target_names)
        return MultiTargetResult(
            model_name=model_name, mode=strategy.mode, params=dict(params),
            joint_q2=metrics["joint_q2"], metrics=metrics,
            precise_note=strategy.precise_note, scale_y=strategy.scale_y,
            mechanism=strategy.mechanism, y_true_pooled=y_true, y_pred_pooled=y_pred,
            preprocessing=preprocessing, varsel_method=varsel_method,
            varsel_tag=varsel_tag, n_variables=int(n_features_sub), error=None,
        )
    except Exception as exc:  # NaN-sink: never a finite 0.0
        return MultiTargetResult(
            model_name=model_name, mode=strategy.mode, params=dict(params),
            joint_q2=np.nan, metrics={}, precise_note=strategy.precise_note,
            scale_y=strategy.scale_y, mechanism=strategy.mechanism,
            y_true_pooled=None, y_pred_pooled=None,
            preprocessing=preprocessing, varsel_method=varsel_method,
            varsel_tag=varsel_tag, n_variables=int(n_features_sub), error=str(exc),
        )
```

Refactor the F2 loop `:604-635` to call it (replace the body of the `for config in model_configs:` loop):

```python
    results: list[MultiTargetResult] = []
    for config in model_configs:
        model_name = config["model_name"]
        params = config.get("params", {})
        results.append(
            _evaluate_multitarget_cell(
                X_arr, Y_arr, model_name, params, splitter, min_fold_train,
                X_arr.shape[1], target_names,
                n_folds=n_folds, n_repeats=n_repeats, random_state=random_state,
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_search.py -v`
Expected: PASS (all F2 tests + the new ones; the consolidation pin and JOINT/INDEPENDENT smokes still pass because the success path is unchanged).

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_search.py tests/test_multitarget_search.py
git commit -m "refactor(T17): shared _evaluate_multitarget_cell + F2 exception->NaN sink"
```

---

### Task 5: `build_multitarget_preprocess_configs` (mirrors search.py enumeration)

**Files:**
- Create: `src/spectral_predict/multitarget_grid.py`
- Test: `tests/test_multitarget_grid.py` (create)

**Interfaces:**
- Produces: `build_multitarget_preprocess_configs(preprocessing_methods, *, window_sizes=None, autoscale=False, baseline_method=None, baseline_params=None, smoothing=False, smoothing_window=17, smoothing_polyorder=2, interference_to_add=None) -> list[dict]`. Each config is a dict with keys `name, deriv, window, polyorder, interference, baseline_method, baseline_params, smoothing, smoothing_window, smoothing_polyorder` (+ optional `base_name`, `autoscale`), matching `search.py:2288-2616` semantics: raw/snv/sg1-4 derivative blocks then baseline/smoothing/autoscale doubling.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multitarget_grid.py`:

```python
"""Tests for the T-17 multi-target grid orchestrator (multitarget_grid.py)."""
from __future__ import annotations

import numpy as np
import pytest


def test_preprocess_configs_raw_and_snv():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({"raw": True, "snv": True})
    names = [c["name"] for c in cfgs]
    assert "raw" in names
    assert "snv" in names
    raw = next(c for c in cfgs if c["name"] == "raw")
    assert raw["deriv"] is None and raw["window"] is None and raw["polyorder"] is None


def test_preprocess_configs_sg_polyorder_pairing():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs(
        {"sg1": True, "sg2": True}, window_sizes=[11]
    )
    derivs = {(c["deriv"], c["polyorder"]) for c in cfgs if c["name"] == "deriv"}
    assert (1, 2) in derivs  # sg1 -> deriv 1 / poly 2
    assert (2, 3) in derivs  # sg2 -> deriv 2 / poly 3


def test_preprocess_configs_autoscale_doubling():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({"raw": True}, autoscale=True)
    names = [c["name"] for c in cfgs]
    assert "raw" in names                 # without-autoscale copy
    assert "raw+autoscale" in names       # with-autoscale copy
    assert any(c.get("autoscale") is True for c in cfgs)
    assert any(c.get("autoscale") is False for c in cfgs)


def test_preprocess_configs_baseline_doubling():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs(
        {"raw": True}, baseline_method="als", baseline_params={"lam": 1e5}
    )
    names = [c["name"] for c in cfgs]
    assert "raw" in names
    assert "als+raw" in names
    without = next(c for c in cfgs if c["name"] == "raw")
    assert without["baseline_method"] is None


def test_preprocess_configs_empty_falls_back_to_raw():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({})
    assert [c["name"] for c in cfgs] == ["raw"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_grid.py -k preprocess_configs -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'spectral_predict.multitarget_grid'`).

- [ ] **Step 3: Write minimal implementation**

Create `src/spectral_predict/multitarget_grid.py` with the module docstring and this function (the SG blocks mirror `search.py` deriv/poly pairing sg1→1/2, sg2→2/3, sg3→3/4, sg4→4/5):

```python
"""T-17 multi-target grid orchestrator: preprocessing × varsel × model × hp.

New orchestration layer ABOVE the F2 seed evaluator. Reuses pure primitives
from search.py / preprocess.py / models.py / variable_selection.py / ga_pls.py /
multi_y.py and the F2 cell helper from multitarget_search.py. Never touches
run_search's single-Y path. Grid engine ONLY.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

_SG_SPEC = {"sg1": (1, 2), "sg2": (2, 3), "sg3": (3, 4), "sg4": (4, 5)}


def _base_config(name: str, deriv, window, polyorder, *, interference_to_add,
                 baseline_method, baseline_params, smoothing,
                 smoothing_window, smoothing_polyorder) -> dict[str, Any]:
    return {
        "name": name, "deriv": deriv, "window": window, "polyorder": polyorder,
        "interference": interference_to_add,
        "baseline_method": baseline_method, "baseline_params": baseline_params,
        "smoothing": smoothing, "smoothing_window": smoothing_window,
        "smoothing_polyorder": smoothing_polyorder,
    }


def build_multitarget_preprocess_configs(
    preprocessing_methods: dict[str, bool],
    *,
    window_sizes: Optional[Sequence[int]] = None,
    autoscale: bool = False,
    baseline_method: Optional[str] = None,
    baseline_params: Optional[dict] = None,
    smoothing: bool = False,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    interference_to_add: Any = None,
) -> list[dict[str, Any]]:
    """Enumerate the basic preprocessing grid (mirrors search.py:2288-2616)."""
    pm = preprocessing_methods or {}
    if window_sizes is None:
        window_sizes = [11]
    common = dict(
        interference_to_add=interference_to_add,
        baseline_method=baseline_method, baseline_params=baseline_params,
        smoothing=smoothing, smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder,
    )
    configs: list[dict[str, Any]] = []
    if pm.get("raw", False):
        configs.append(_base_config("raw", None, None, None, **common))
    if pm.get("snv", False):
        configs.append(_base_config("snv", None, None, None, **common))
    for sg, (deriv, poly) in _SG_SPEC.items():
        if not pm.get(sg, False):
            continue
        for w in window_sizes:
            configs.append(_base_config("deriv", deriv, w, poly, **common))
            if pm.get("snv", False):
                configs.append(_base_config("snv_deriv", deriv, w, poly, **common))
            if pm.get("deriv_snv", False):
                configs.append(_base_config("deriv_snv", deriv, w, poly, **common))
    if not configs:
        configs.append(_base_config("raw", None, None, None, **common))

    # Baseline doubling (search.py:2565).
    if baseline_method is not None and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["baseline_method"] = None; no["baseline_params"] = None
            without.append(no)
            bl = dict(cfg); bl["base_name"] = cfg.get("base_name", cfg["name"])
            bl["name"] = f"{baseline_method}+{cfg['name']}"
            with_.append(bl)
        configs = without + with_

    # Smoothing doubling (search.py:2582).
    if smoothing and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["smoothing"] = False
            without.append(no)
            sm = dict(cfg); sm["base_name"] = cfg.get("base_name", cfg["name"])
            nm = cfg["name"]
            sm["name"] = (f"{nm.split('+', 1)[0]}+sg0+{nm.split('+', 1)[1]}"
                          if "+" in nm else f"sg0+{nm}")
            with_.append(sm)
        configs = without + with_

    # Autoscale doubling (search.py:2603).
    if autoscale and configs:
        without, with_ = [], []
        for cfg in configs:
            no = dict(cfg); no["autoscale"] = False
            without.append(no)
            sc = dict(cfg); sc["autoscale"] = True
            sc["base_name"] = cfg.get("base_name", cfg["name"])
            sc["name"] = cfg["name"] + "+autoscale"
            with_.append(sc)
        configs = without + with_

    return configs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_grid.py -k preprocess_configs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_grid.py tests/test_multitarget_grid.py
git commit -m "feat(T17): build_multitarget_preprocess_configs mirroring search.py grid"
```

---

### Task 6: Interval subset-adapter (ipls_forward/backward/mc_sipls/mwpls)

**Files:**
- Modify: `src/spectral_predict/multitarget_grid.py` (add `_interval_subset_adapter` + `_parse_ipls_subset_limit`)
- Test: `tests/test_multitarget_varsel.py` (append)

**Interfaces:**
- Consumes: `ipls_forward`, `ipls_backward`, `mc_sipls`, `mwpls` (`variable_selection.py`; `wavelengths` is a REQUIRED positional; each returns `list[dict]` keyed `{'indices','tag','rmsecv',...}`).
- Produces: `_parse_ipls_subset_limit(limit: str) -> int | None` ("Top 5"→5, "Top 10"→10, "Top 20"→20, "All"→None). `_interval_subset_adapter(method, X_pp, Y, wavelengths, *, ipls_subset_limit="Top 10") -> list[dict]` returning subset dicts `{'indices': np.ndarray, 'tag': str, 'method': str}`, best-RMSECV first, truncated by the limit.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_varsel.py`:

```python
def test_parse_ipls_subset_limit():
    from spectral_predict.multitarget_grid import _parse_ipls_subset_limit

    assert _parse_ipls_subset_limit("Top 5") == 5
    assert _parse_ipls_subset_limit("Top 20") == 20
    assert _parse_ipls_subset_limit("All") is None


def test_interval_subset_adapter_returns_truncated_subsets(xy_multi):
    from spectral_predict.multitarget_grid import _interval_subset_adapter

    X, Y, wl = xy_multi
    subs = _interval_subset_adapter(
        "ipls_forward", X, Y, wl, ipls_subset_limit="Top 5"
    )
    assert 1 <= len(subs) <= 5
    for s in subs:
        assert s["method"] == "ipls_forward"
        assert isinstance(s["indices"], np.ndarray)
        assert s["indices"].size >= 1
        assert s["indices"].size < X.shape[1] or "iPLS" in s["tag"]
        assert isinstance(s["tag"], str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "subset_limit or interval_subset_adapter" -v`
Expected: FAIL (functions do not exist).

- [ ] **Step 3: Write minimal implementation**

Append to `src/spectral_predict/multitarget_grid.py`:

```python
_INTERVAL_METHODS = {"ipls_forward", "ipls_backward", "mc_sipls", "mwpls"}


def _parse_ipls_subset_limit(limit: str) -> Optional[int]:
    """'Top N' -> N; 'All' (or anything non-numeric) -> None (no truncation)."""
    if not limit or limit.strip().lower() == "all":
        return None
    digits = "".join(ch for ch in str(limit) if ch.isdigit())
    return int(digits) if digits else None


def _interval_subset_adapter(
    method: str, X_pp: np.ndarray, Y: np.ndarray, wavelengths: np.ndarray,
    *, ipls_subset_limit: str = "Top 10",
) -> list[dict[str, Any]]:
    """Run a multi-Y-safe interval varsel method; return truncated subset dicts."""
    from .variable_selection import ipls_backward, ipls_forward, mc_sipls, mwpls

    fn = {"ipls_forward": ipls_forward, "ipls_backward": ipls_backward,
          "mc_sipls": mc_sipls, "mwpls": mwpls}[method]
    raw = fn(X_pp, Y, wavelengths)  # wavelengths is a required positional
    # Best RMSECV first (finite before non-finite).
    ordered = sorted(raw, key=lambda d: (0 if np.isfinite(d.get("rmsecv", np.inf)) else 1,
                                         d.get("rmsecv", np.inf)))
    limit = _parse_ipls_subset_limit(ipls_subset_limit)
    if limit is not None:
        ordered = ordered[:limit]
    return [
        {"indices": np.asarray(d["indices"]), "tag": d.get("tag", method),
         "method": method}
        for d in ordered
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "subset_limit or interval_subset_adapter" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_grid.py tests/test_multitarget_varsel.py
git commit -m "feat(T17): interval subset-adapter for multi-Y varsel"
```

---

### Task 7: `spa` 2-D verification gate (load-bearing)

**Files:**
- Modify: `src/spectral_predict/multitarget_grid.py` (add `verify_spa_multi_y_safe`)
- Test: `tests/test_multitarget_varsel.py` (append)

**Interfaces:**
- Consumes: `spa_selection(X, y, n_features, cv_folds=5)` (`variable_selection.py:430`, calls `_prep_varsel_y` so 2-D Y is preserved).
- Produces: `verify_spa_multi_y_safe(X, Y, *, n_features=5) -> bool` — True iff `spa_selection` on 2-D Y returns a finite, non-degenerate importance array of length `X.shape[1]` with at least one non-zero entry. This boolean GATES both `spa` and `fipls_spa` in later tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_varsel.py`:

```python
def test_verify_spa_multi_y_safe_true_on_informative_block(xy_multi):
    from spectral_predict.multitarget_grid import verify_spa_multi_y_safe

    X, Y, _wl = xy_multi
    assert verify_spa_multi_y_safe(X, Y, n_features=5) is True


def test_verify_spa_multi_y_safe_shape_and_finiteness():
    from spectral_predict.variable_selection import spa_selection

    rng = np.random.default_rng(9)
    X = rng.standard_normal((40, 20))
    Y = X[:, :3] @ rng.standard_normal((3, 2)) + 0.05 * rng.standard_normal((40, 2))
    imp = np.asarray(spa_selection(X, Y, n_features=5), dtype=float)
    assert imp.shape == (20,)
    assert np.all(np.isfinite(imp))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "verify_spa" -v`
Expected: FAIL (`verify_spa_multi_y_safe` does not exist; the second test also empirically confirms the SPA-on-2-D contract).

- [ ] **Step 3: Write minimal implementation**

Append to `src/spectral_predict/multitarget_grid.py`:

```python
def verify_spa_multi_y_safe(X: np.ndarray, Y: np.ndarray, *, n_features: int = 5) -> bool:
    """Return True iff spa_selection produces a sane 2-D-Y importance array.

    Gates BOTH ``spa`` and ``fipls_spa`` (which internally calls spa). If SPA's
    R2 criterion degenerates on 2-D Y this returns False and the caller demotes
    both methods to skip-with-notice.
    """
    from .variable_selection import spa_selection

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] < 2:
        return True  # single-Y path is already verified elsewhere
    try:
        imp = np.asarray(spa_selection(X, Y, n_features=n_features), dtype=float)
    except Exception:
        return False
    return (
        imp.ndim == 1
        and imp.shape[0] == X.shape[1]
        and np.all(np.isfinite(imp))
        and np.count_nonzero(imp) >= 1
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "verify_spa" -v`
Expected: PASS. (If the second test empirically FAILS — SPA returns non-finite/degenerate on 2-D Y — STOP: per spec §4 the correct action is to demote `spa` and `fipls_spa` to skip-with-notice in Tasks 8/9/10 instead of including them. Record the finding in `docs/SESSION_LOG.md`.)

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_grid.py tests/test_multitarget_varsel.py
git commit -m "feat(T17): spa 2-D verification gate for spa/fipls_spa inclusion"
```

---

### Task 8: Importance-adapter (spa / ga-linear / importance) + top-N via variable_counts

**Files:**
- Modify: `src/spectral_predict/multitarget_grid.py` (add `_importances_to_subsets`, `_model_independent_importances`, `_importance_reference_fit`)
- Test: `tests/test_multitarget_varsel.py` (append)

**Interfaces:**
- Consumes: `spa_selection`, `ga_pls_selection` (`ga_pls.py:499`, PLS-fitness/linear-only, 2-D-safe), `build_multitarget_estimator` + `get_feature_importances` (`models.py:1798`), `aggregate_importance` (`multi_y.py:408`), `resolve_multitarget_strategy`.
- Produces: `_importances_to_subsets(importances, method, *, variable_counts, n_features_sub) -> list[dict]` (top-N via `variable_counts` filtered to `n < n_features_sub`, stable-argsort mirroring `search.py:3766`); `_model_independent_importances(method, X_pp, Y) -> np.ndarray | None` (for `spa`, `ga`; None if method not handled); `_importance_reference_fit(model_name, X_pp, Y, min_fold_train) -> np.ndarray` (for the model-specific `importance` method).
- `LINEAR_MODELS = {"PLS", "Ridge", "Lasso", "ElasticNet"}` module constant (used to gate `ga`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_varsel.py`:

```python
def test_importances_to_subsets_top_n_filtered(xy_multi):
    from spectral_predict.multitarget_grid import _importances_to_subsets

    X, _Y, _wl = xy_multi
    imp = np.arange(X.shape[1], dtype=float)  # 40 features, strictly increasing
    subs = _importances_to_subsets(
        imp, "spa", variable_counts=[5, 10, 999], n_features_sub=X.shape[1]
    )
    # 999 >= 40 is filtered out; 5 and 10 remain.
    sizes = sorted(s["indices"].size for s in subs)
    assert sizes == [5, 10]
    top5 = next(s for s in subs if s["indices"].size == 5)
    assert set(top5["indices"].tolist()) == {35, 36, 37, 38, 39}  # highest importances
    assert top5["method"] == "spa"
    assert "top5" in top5["tag"]


def test_model_independent_importances_spa_and_ga(xy_multi):
    from spectral_predict.multitarget_grid import _model_independent_importances

    X, Y, _wl = xy_multi
    imp_spa = _model_independent_importances("spa", X, Y)
    assert imp_spa is not None and imp_spa.shape == (X.shape[1],)
    imp_ga = _model_independent_importances("ga", X, Y)
    assert imp_ga is not None and imp_ga.shape == (X.shape[1],)


def test_importance_reference_fit_tree_model(xy_multi):
    from spectral_predict.multitarget_grid import _importance_reference_fit

    X, Y, _wl = xy_multi
    imp = _importance_reference_fit("RandomForest", X, Y, min_fold_train=X.shape[0] - 1)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "importances_to_subsets or model_independent or reference_fit" -v`
Expected: FAIL (functions do not exist).

- [ ] **Step 3: Write minimal implementation**

Append to `src/spectral_predict/multitarget_grid.py`:

```python
LINEAR_MODELS = {"PLS", "Ridge", "Lasso", "ElasticNet"}
_IMPORTANCE_METHODS = {"spa", "ga", "importance", "fipls_spa"}


def _importances_to_subsets(
    importances: np.ndarray, method: str, *, variable_counts, n_features_sub: int,
) -> list[dict[str, Any]]:
    """Top-N subsets from an importance array (mirrors search.py:3695/3766)."""
    imp = np.asarray(importances, dtype=float)
    counts = variable_counts or [10, 20, 50, 100, 250, 500, 1000]
    valid = [n for n in counts if n < n_features_sub]
    subsets: list[dict[str, Any]] = []
    for n_top in valid:
        top = np.argsort(imp, kind="stable")[-n_top:][::-1]
        subsets.append({"indices": np.asarray(top), "tag": f"{method}_top{n_top}",
                        "method": method})
    return subsets


def _model_independent_importances(method: str, X_pp: np.ndarray, Y: np.ndarray):
    """spa / ga importance arrays (model-independent, cached per preprocess)."""
    if method == "spa":
        from .variable_selection import spa_selection
        return np.asarray(spa_selection(X_pp, Y, n_features=max(5, X_pp.shape[1] // 10)),
                          dtype=float)
    if method == "ga":
        from .ga_pls import ga_pls_selection
        return np.asarray(ga_pls_selection(X_pp, Y, task_type="regression", verbose=0),
                          dtype=float)
    return None


def _importance_reference_fit(
    model_name: str, X_pp: np.ndarray, Y: np.ndarray, min_fold_train: int,
) -> np.ndarray:
    """Model-specific importances: fit a reference estimator on full X_pp/Y,
    extract a (n_features, n_targets) matrix, aggregate to per-feature scores."""
    from sklearn.multioutput import MultiOutputRegressor

    from .models import get_feature_importances
    from .multi_y import aggregate_importance
    from .multitarget_search import build_multitarget_estimator, resolve_multitarget_strategy

    Y = np.asarray(Y, dtype=float)
    strategy = resolve_multitarget_strategy(model_name)
    est = build_multitarget_estimator(strategy, {}, min_fold_train, X_pp.shape[1])
    est.fit(X_pp, Y)  # importances are rank-only; raw-Y fit is adequate
    n_features = X_pp.shape[1]

    if isinstance(est, MultiOutputRegressor):
        cols = []
        for i, sub in enumerate(est.estimators_):
            imp = get_feature_importances(sub, model_name, X_pp, Y[:, i])
            cols.append(np.asarray(imp, dtype=float).ravel())
        matrix = np.column_stack(cols)  # (n_features, n_targets)
    elif hasattr(est, "feature_importances_"):
        matrix = np.asarray(est.feature_importances_, dtype=float).reshape(n_features, -1)
    elif hasattr(est, "coef_"):
        coef = np.abs(np.asarray(est.coef_, dtype=float))
        matrix = coef.T if coef.ndim == 2 else coef.reshape(n_features, -1)
    else:
        matrix = np.ones((n_features, 1))
    return aggregate_importance(matrix, rule="mean")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "importances_to_subsets or model_independent or reference_fit" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_grid.py tests/test_multitarget_varsel.py
git commit -m "feat(T17): importance-adapter (spa/ga/importance) top-N subsets"
```

---

### Task 9: De-ravel `fipls_spa` into a multi-Y-safe method (gated by Task 7)

**Files:**
- Modify: `src/spectral_predict/variable_selection.py:1155-1156` (replace the unconditional `y.ravel()` with `_prep_varsel_y`)
- Test: `tests/test_multitarget_varsel.py` (append)

**Interfaces:**
- Consumes: `_prep_varsel_y` (`variable_selection.py:17`), `verify_spa_multi_y_safe` (Task 7). `fipls_spa_selection` internals already call the multi-safe `ipls_forward` + `spa_selection`.
- Produces: `fipls_spa_selection(X, Y, wavelengths, ...)` preserves 2-D Y (no longer flattens), returning an importance array of length `X.shape[1]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_varsel.py`:

```python
def test_fipls_spa_preserves_2d_y_when_spa_safe(xy_multi):
    from spectral_predict.multitarget_grid import verify_spa_multi_y_safe
    from spectral_predict.variable_selection import fipls_spa_selection

    X, Y, wl = xy_multi
    if not verify_spa_multi_y_safe(X, Y, n_features=5):
        pytest.skip("SPA not 2-D-safe in this environment; fipls_spa stays skipped")
    imp = np.asarray(fipls_spa_selection(X, Y, wl), dtype=float)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "fipls_spa_preserves" -v`
Expected: FAIL (`fipls_spa_selection` ravels Y at `:1156`, so it fits on a flattened `(n*n_targets,)` target and its output shape/finiteness breaks).

- [ ] **Step 3: Write minimal implementation**

In `variable_selection.py`, replace the flatten at `:1155-1156`:

```python
    X = np.asarray(X)
    y = _prep_varsel_y(y)
```

(`_prep_varsel_y` ravels 1-D / single-column-2-D and preserves genuine 2-D Y, so single-Y behavior is byte-identical.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "fipls_spa_preserves" -v`
Expected: PASS
Then re-confirm single-Y gold identity:
Run: `.venv312/Scripts/python.exe -m pytest tests/ -k "SingleYByteIdentityGold" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/variable_selection.py tests/test_multitarget_varsel.py
git commit -m "feat(T17): de-ravel fipls_spa to a multi-Y-safe importance method"
```

---

### Task 10: Varsel method classifier + subset dispatcher + cache

**Files:**
- Modify: `src/spectral_predict/multitarget_grid.py` (add `classify_varsel_method`, `SKIP_WITH_NOTICE`, `build_multitarget_varsel_subsets`)
- Test: `tests/test_multitarget_varsel.py` (append)

**Interfaces:**
- Consumes: Task 6 `_interval_subset_adapter`, Task 8 `_model_independent_importances` / `_importances_to_subsets` / `_importance_reference_fit` / `LINEAR_MODELS`, Task 7 `verify_spa_multi_y_safe`.
- Produces:
  - `SKIP_WITH_NOTICE = {"uve","uve_spa","cars","cars-tree","uve_cars","uve_cars_tree","uve_cars_spa","fipls_cars","ipls","vcpa-iriv"}` module constant.
  - `classify_varsel_method(method, *, enabled_models, spa_ok) -> str` returning `"subset"`, `"importance"`, or `"skip"`.
  - `build_multitarget_varsel_subsets(methods, X_pp, Y, wavelengths, *, enabled_models, variable_counts, ipls_subset_limit, spa_ok, min_fold_train, cache, apply_uve_prefilter=False) -> tuple[list[dict], list[str]]` returning `(subsets_including_full, skipped_notices)`. `subsets` always leads with `{"indices": full, "tag": "full", "method": "full"}`. Model-independent results are memoized in `cache` keyed by `(preprocess_id, method)`; the model-specific `importance` method is left to the orchestrator (returns a marker `{"method": "importance", "model_specific": True}`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_varsel.py`:

```python
def test_classify_varsel_method():
    from spectral_predict.multitarget_grid import classify_varsel_method

    assert classify_varsel_method("ipls_forward", enabled_models=["PLS"], spa_ok=True) == "subset"
    assert classify_varsel_method("mwpls", enabled_models=["RandomForest"], spa_ok=True) == "subset"
    assert classify_varsel_method("spa", enabled_models=["PLS"], spa_ok=True) == "importance"
    assert classify_varsel_method("spa", enabled_models=["PLS"], spa_ok=False) == "skip"
    assert classify_varsel_method("fipls_spa", enabled_models=["PLS"], spa_ok=False) == "skip"
    assert classify_varsel_method("importance", enabled_models=["RandomForest"], spa_ok=True) == "importance"
    # ga is linear-only: present linear model -> importance; tree-only -> skip.
    assert classify_varsel_method("ga", enabled_models=["PLS"], spa_ok=True) == "importance"
    assert classify_varsel_method("ga", enabled_models=["RandomForest"], spa_ok=True) == "skip"
    # UVE/CARS/legacy always skip.
    for m in ["uve", "cars", "cars-tree", "uve_cars", "ipls", "vcpa-iriv", "fipls_cars"]:
        assert classify_varsel_method(m, enabled_models=["PLS"], spa_ok=True) == "skip"


def test_build_varsel_subsets_full_plus_interval_and_skips(xy_multi):
    from spectral_predict.multitarget_grid import build_multitarget_varsel_subsets

    X, Y, wl = xy_multi
    cache = {}
    subs, skipped = build_multitarget_varsel_subsets(
        ["ipls_forward", "uve", "cars"], X, Y, wl,
        enabled_models=["PLS"], variable_counts=[5, 10],
        ipls_subset_limit="Top 5", spa_ok=True,
        min_fold_train=X.shape[0] - 1, cache=cache,
        preprocess_id="raw",
    )
    tags = [s["tag"] for s in subs]
    assert "full" in tags
    assert subs[0]["method"] == "full"
    assert any(s["method"] == "ipls_forward" for s in subs)
    assert set(skipped) == {"uve", "cars"}


def test_build_varsel_subsets_cache_hit(xy_multi):
    from spectral_predict.multitarget_grid import build_multitarget_varsel_subsets

    X, Y, wl = xy_multi
    cache = {}
    build_multitarget_varsel_subsets(
        ["spa"], X, Y, wl, enabled_models=["PLS"], variable_counts=[5],
        ipls_subset_limit="All", spa_ok=True, min_fold_train=X.shape[0] - 1,
        cache=cache, preprocess_id="raw",
    )
    assert ("raw", "spa") in cache  # importance memoized per (preprocess, method)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "classify_varsel or build_varsel_subsets" -v`
Expected: FAIL (functions do not exist).

- [ ] **Step 3: Write minimal implementation**

Append to `src/spectral_predict/multitarget_grid.py`:

```python
SKIP_WITH_NOTICE = {
    "uve", "uve_spa", "cars", "cars-tree", "uve_cars", "uve_cars_tree",
    "uve_cars_spa", "fipls_cars", "ipls", "vcpa-iriv",
}


def classify_varsel_method(method: str, *, enabled_models, spa_ok: bool) -> str:
    """Route a GUI varsel method string to 'subset' / 'importance' / 'skip'."""
    if method in SKIP_WITH_NOTICE:
        return "skip"
    if method in _INTERVAL_METHODS:
        return "subset"
    if method in ("spa", "fipls_spa"):
        return "importance" if spa_ok else "skip"
    if method == "ga":
        has_linear = any(m in LINEAR_MODELS for m in (enabled_models or []))
        return "importance" if has_linear else "skip"
    if method == "importance":
        return "importance"
    return "skip"


def build_multitarget_varsel_subsets(
    methods, X_pp, Y, wavelengths, *, enabled_models, variable_counts,
    ipls_subset_limit, spa_ok, min_fold_train, cache, preprocess_id,
    apply_uve_prefilter: bool = False,
):
    """Return (subsets_incl_full, skipped_notices), caching per (preprocess, method)."""
    n_features_sub = int(X_pp.shape[1])
    subsets: list[dict[str, Any]] = [
        {"indices": np.arange(n_features_sub), "tag": "full", "method": "full"}
    ]
    skipped: list[str] = []
    if apply_uve_prefilter:
        skipped.append("apply_uve_prefilter")
    for method in (methods or []):
        kind = classify_varsel_method(method, enabled_models=enabled_models, spa_ok=spa_ok)
        if kind == "skip":
            skipped.append(method)
            continue
        key = (preprocess_id, method)
        if kind == "subset":
            if key not in cache:
                cache[key] = _interval_subset_adapter(
                    method, X_pp, Y, wavelengths, ipls_subset_limit=ipls_subset_limit
                )
            subsets.extend(cache[key])
        elif kind == "importance":
            if method == "importance":
                # model-specific: orchestrator resolves per (preprocess, model).
                subsets.append({"method": "importance", "model_specific": True})
                continue
            if key not in cache:
                imp = _model_independent_importances(method, X_pp, Y)
                cache[key] = _importances_to_subsets(
                    imp, method, variable_counts=variable_counts,
                    n_features_sub=n_features_sub,
                )
            subsets.extend(cache[key])
    return subsets, skipped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_varsel.py -k "classify_varsel or build_varsel_subsets" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_grid.py tests/test_multitarget_varsel.py
git commit -m "feat(T17): varsel classifier + subset dispatcher + per-preprocess cache"
```

---

### Task 11: `run_multitarget_grid_search` orchestrator

**Files:**
- Modify: `src/spectral_predict/multitarget_grid.py` (add `run_multitarget_grid_search` + `_dedup_model_configs`)
- Test: `tests/test_multitarget_grid.py` (append)

**Interfaces:**
- Consumes: `build_multitarget_preprocess_configs` (Task 5), `build_multitarget_varsel_subsets` (Task 10), `verify_spa_multi_y_safe` (Task 7), `_importance_reference_fit`/`_importances_to_subsets` (Task 8), `build_preprocessing_pipeline` (`preprocess.py:286`), `_apply_edge_mask_to_data` (`search.py:227`), `get_model_grids` (`models.py:542`), `_evaluate_multitarget_cell` (`multitarget_search.py`), `inter_target_correlation`+`build_cv_splitter`+`cap_components` (imported), `MultiTargetSearchOutput`.
- Produces: `run_multitarget_grid_search(X, Y, *, model_names, target_names, wavelengths, preprocessing_methods, autoscale, baseline_method=None, baseline_params=None, smoothing=False, smoothing_window=17, smoothing_polyorder=2, interference_to_add=None, wavelength_restriction=None, variable_selection_methods, variable_counts=None, ipls_subset_limit="Top 10", tier="standard", model_grid_overrides=None, max_n_components=10, max_iter=500, window_sizes=None, cv="kfold", n_folds=5, n_repeats=5, random_state=42, optimization_method="grid", controller=None, progress_callback=None, n_jobs=-1) -> MultiTargetSearchOutput`. Sequential cell loop; `controller.check_and_wait()` before each cell; `progress_callback(info_dict)` with GUI-shaped dict; NaN-safe rank; `output.skipped` carries the notices.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multitarget_grid.py`:

```python
@pytest.fixture
def rng():
    return np.random.default_rng(4242)


@pytest.fixture
def grid_xy(rng):
    n, p = 45, 30
    X = rng.standard_normal((n, p))
    base = X[:, :4] @ rng.standard_normal((4, 2))
    Y = base + 0.05 * rng.standard_normal((n, 2))
    wl = np.linspace(1000.0, 2000.0, p)
    return X, Y, wl


def test_grid_search_grid_only_assert(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    with pytest.raises(ValueError):
        run_multitarget_grid_search(
            X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
            preprocessing_methods={"raw": True}, autoscale=False,
            variable_selection_methods=[], tier="quick",
            cv="kfold", n_folds=3, n_repeats=1, optimization_method="unified",
        )


def test_grid_search_end_to_end_ranks_and_skips(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True, "snv": True}, autoscale=False,
        variable_selection_methods=["ipls_forward", "uve"], variable_counts=[5, 10],
        ipls_subset_limit="Top 3", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, random_state=42,
    )
    # Ranked, NaN-safe: best has a finite joint_q2.
    assert len(out.results) >= 4
    assert np.isfinite(out.results[0].joint_q2)
    # Both preprocessing states and >1 varsel tag appear.
    assert {r.preprocessing for r in out.results} >= {"raw", "snv"}
    assert any(r.varsel_method == "ipls_forward" for r in out.results)
    assert any(r.varsel_method == "full" for r in out.results)
    # UVE skip surfaced.
    assert "uve" in out.skipped


def test_grid_search_progress_callback_dict_shape(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    seen = []
    run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=[], tier="quick",
        cv="kfold", n_folds=3, n_repeats=1,
        progress_callback=lambda info: seen.append(info),
    )
    assert seen
    last = seen[-1]
    assert set(["message", "current", "total"]).issubset(last.keys())
    assert "best_model" in last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_grid.py -k "grid_search" -v`
Expected: FAIL (`run_multitarget_grid_search` does not exist).

- [ ] **Step 3: Write minimal implementation**

Append to `src/spectral_predict/multitarget_grid.py`:

```python
def _dedup_model_configs(model_grids: dict) -> list[dict[str, Any]]:
    """Flatten get_model_grids output to [{'model_name','params'}], deduped by the
    FULL consumed keyset (builders now consume all keys, so this is a safety net).
    """
    out, seen = [], set()
    for model_name, configs in model_grids.items():
        for _estimator, params in configs:
            key = (model_name, frozenset(
                (k, tuple(v) if isinstance(v, (list, np.ndarray)) else v)
                for k, v in params.items()
            ))
            if key in seen:
                continue
            seen.add(key)
            out.append({"model_name": model_name, "params": dict(params)})
    return out


def run_multitarget_grid_search(
    X, Y, *, model_names, target_names, wavelengths,
    preprocessing_methods, autoscale,
    baseline_method=None, baseline_params=None,
    smoothing=False, smoothing_window=17, smoothing_polyorder=2,
    interference_to_add=None, wavelength_restriction=None,
    variable_selection_methods, variable_counts=None,
    ipls_subset_limit="Top 10", tier="standard", model_grid_overrides=None,
    max_n_components=10, max_iter=500, window_sizes=None,
    cv="kfold", n_folds=5, n_repeats=5, random_state=42,
    optimization_method="grid", controller=None, progress_callback=None, n_jobs=-1,
):
    from sklearn.pipeline import Pipeline

    from .cv_utils import build_cv_splitter
    from .models import get_model_grids
    from .multi_y import inter_target_correlation
    from .multitarget_search import MultiTargetSearchOutput, _evaluate_multitarget_cell
    from .preprocess import build_preprocessing_pipeline
    from .search import _apply_edge_mask_to_data

    if optimization_method != "grid":
        raise ValueError(
            f"Multi-target grid search is Grid-engine ONLY; got "
            f"optimization_method={optimization_method!r}. Bayesian/NSGA-II are 1-D-only."
        )

    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    wl = np.asarray(wavelengths, dtype=float)
    target_names = list(target_names)
    correlation = inter_target_correlation(Y_arr)

    # Grid-only gate for spa/fipls_spa inclusion (verified once on the raw block).
    spa_ok = verify_spa_multi_y_safe(X_arr, Y_arr)

    preprocess_configs = build_multitarget_preprocess_configs(
        preprocessing_methods, window_sizes=window_sizes, autoscale=autoscale,
        baseline_method=baseline_method, baseline_params=baseline_params,
        smoothing=smoothing, smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder, interference_to_add=interference_to_add,
    )

    wl_min = wl_max = wl_regions = None
    if wavelength_restriction:
        wl_min = wavelength_restriction.get("min")
        wl_max = wavelength_restriction.get("max")
        wl_regions = wavelength_restriction.get("regions")
    restriction_active = bool(wl_regions or wl_min is not None or wl_max is not None)

    splitter = build_cv_splitter(
        cv, n_folds, "regression", n_repeats=n_repeats, random_state=random_state, y=None,
    ) if isinstance(cv, str) else cv
    min_fold_train = min(len(tr) for tr, _ in splitter.split(X_arr, Y_arr))

    results = []
    skipped_all: list[str] = []
    varsel_cache: dict = {}
    best_finite = None
    # First pass builds the full cell list so 'total' is honest.
    cells = []  # (preprocess_cfg, X_sub, wl_sub, subset_dict, model_cfg)

    for pc in preprocess_configs:
        name = pc.get("base_name", pc["name"])
        steps = build_preprocessing_pipeline(
            name, pc["deriv"], pc["window"], pc["polyorder"],
            task_type="regression", interference=pc.get("interference"), wavelengths=wl,
            baseline_method=pc.get("baseline_method"), baseline_params=pc.get("baseline_params"),
            smoothing=pc.get("smoothing", False), smoothing_window=pc.get("smoothing_window", 17),
            smoothing_polyorder=pc.get("smoothing_polyorder", 2), autoscale=pc.get("autoscale", False),
        )
        X_pp = X_arr.copy()
        if steps:
            X_pp = Pipeline(steps).fit_transform(X_pp, Y_arr)
        wl_pp = wl
        # Wavelength restriction (search.py:2777) then edge-mask if unrestricted (search.py:2876).
        if restriction_active:
            if wl_regions:
                mask = np.zeros(wl_pp.shape[0], dtype=bool)
                for lo, hi in wl_regions:
                    mask |= (wl_pp >= lo) & (wl_pp <= hi)
            else:
                mask = np.ones(wl_pp.shape[0], dtype=bool)
                if wl_min is not None:
                    mask &= wl_pp >= wl_min
                if wl_max is not None:
                    mask &= wl_pp <= wl_max
            X_pp, wl_pp = X_pp[:, mask], wl_pp[mask]
        elif pc.get("deriv") and pc.get("window"):
            X_pp, wl_pp, _ = _apply_edge_mask_to_data(X_pp, wl_pp, pc)

        subsets, skipped = build_multitarget_varsel_subsets(
            variable_selection_methods, X_pp, Y_arr, wl_pp,
            enabled_models=model_names, variable_counts=variable_counts,
            ipls_subset_limit=ipls_subset_limit, spa_ok=spa_ok,
            min_fold_train=min_fold_train, cache=varsel_cache, preprocess_id=pc["name"],
        )
        for s in skipped:
            if s not in skipped_all:
                skipped_all.append(s)

        clamped = min(min_fold_train - 1, max_n_components)
        model_grids = get_model_grids(
            task_type="regression", n_features=X_pp.shape[1], tier=tier,
            enabled_models=list(model_names), max_n_components=max(1, clamped),
            max_iter=max_iter, **(model_grid_overrides or {}),
        )
        model_cfgs = _dedup_model_configs(model_grids)

        for s in subsets:
            if s.get("model_specific"):
                # 'importance' resolved per model below.
                for mc in model_cfgs:
                    imp = _importance_reference_fit(mc["model_name"], X_pp, Y_arr, min_fold_train)
                    for sub in _importances_to_subsets(
                        imp, "importance", variable_counts=variable_counts,
                        n_features_sub=X_pp.shape[1],
                    ):
                        cells.append((pc, X_pp[:, sub["indices"]], sub["tag"],
                                      "importance", sub["indices"].size, mc))
                continue
            idx = s["indices"]
            X_sub = X_pp[:, idx]
            for mc in model_cfgs:
                cells.append((pc, X_sub, s["tag"], s["method"], X_sub.shape[1], mc))

    total = len(cells)
    for i, (pc, X_sub, tag, method, n_vars, mc) in enumerate(cells):
        if controller is not None:
            controller.check_and_wait()
        res = _evaluate_multitarget_cell(
            X_sub, Y_arr, mc["model_name"], mc["params"], splitter, min_fold_train,
            X_sub.shape[1], target_names, n_folds=n_folds, n_repeats=n_repeats,
            random_state=random_state, preprocessing=pc["name"],
            varsel_method=method, varsel_tag=tag,
        )
        results.append(res)
        if np.isfinite(res.joint_q2) and (best_finite is None or res.joint_q2 > best_finite.joint_q2):
            best_finite = res
        if progress_callback is not None:
            bm = None
            if best_finite is not None:
                bm = {
                    "Model": best_finite.model_name, "Preprocess": best_finite.preprocessing,
                    "Deriv": None, "RMSEcv": float(np.mean(best_finite.metrics.get("rmse", [np.nan]))),
                    "R2cv": float(best_finite.joint_q2),
                    "top_vars": str(best_finite.n_variables),
                }
            progress_callback({
                "message": f"Multi-target cell {i + 1}/{total}",
                "current": i + 1, "total": total, "best_model": bm,
            })

    results.sort(
        key=lambda r: (np.isfinite(r.joint_q2),
                       r.joint_q2 if np.isfinite(r.joint_q2) else float("-inf")),
        reverse=True,
    )
    return MultiTargetSearchOutput(
        results=results, target_names=target_names, correlation=correlation,
        n_targets=Y_arr.shape[1], skipped=skipped_all,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_grid.py -v`
Expected: PASS

Add a dedup pin (append and run):

```python
def test_dedup_keyset_equals_consumed_no_config_lost():
    from spectral_predict.multitarget_grid import _dedup_model_configs

    # Two RF configs differing only in a consumed key (bootstrap) must NOT collapse.
    class _E: pass
    grids = {"RandomForest": [
        (_E(), {"n_estimators": 50, "bootstrap": True}),
        (_E(), {"n_estimators": 50, "bootstrap": False}),
        (_E(), {"n_estimators": 50, "bootstrap": True}),  # exact dup -> collapses
    ]}
    out = _dedup_model_configs(grids)
    assert len(out) == 2
    assert {c["params"]["bootstrap"] for c in out} == {True, False}
```

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_grid.py::test_dedup_keyset_equals_consumed_no_config_lost -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/spectral_predict/multitarget_grid.py tests/test_multitarget_grid.py
git commit -m "feat(T17): run_multitarget_grid_search orchestrator (grid-only, NaN-safe, progress)"
```

---

### Task 12: GUI — rework `_run_multitarget_search` (inherited config + worker + Cancel)

**Files:**
- Modify: `spectral_predict_gui_optimized.py:14759-14847` (`_run_multitarget_search`; add `_run_multitarget_search_thread`, `_cancel_multitarget_search`, `_multitarget_progress`)
- Test: `tests/gui/test_multitarget_tab.py` (append)

**Interfaces:**
- Consumes: `run_multitarget_grid_search` (Task 11), `SearchController` (`search_controller.py`), the SAME config state `_run_analysis_thread` reads (see GUI anchors: `selected_varsel_methods` `:28125`, `variable_counts` `:26359`, `ipls_subset_limit` `:3870`, the full per-model HP override locals mirroring the `run_search` call `:29159-29227`, `self.use_autoscale`, `self.enable_smoothing`, baseline vars, `self.cv_strategy`, `self.folds`, `self.cv_n_repeats`, `self.max_iter`, tier, wavelength restriction `analysis_wl_*`).
- Produces: `self._multitarget_controller` (a NEW `SearchController`, never `self.search_controller`); a daemon worker thread; a `model_grid_overrides` dict built from the GUI HP locals; all Tk writes via `self.root.after(0, ...)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/test_multitarget_tab.py`:

```python
@pytest.mark.gui
class TestMultiTargetGridDispatch:
    def test_dispatch_passes_inherited_config_and_uses_separate_controller(self, gui_app, monkeypatch):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()
        gui_app.multitarget_model_vars["PLS"].set(True)

        captured = {}

        def _fake_grid(X, Y, **kwargs):
            captured.update(kwargs)
            from spectral_predict.multitarget_search import MultiTargetSearchOutput
            return MultiTargetSearchOutput(results=[], target_names=kwargs["target_names"],
                                           correlation={}, n_targets=Y.shape[1], skipped=["uve"])

        import spectral_predict.multitarget_grid as mg
        monkeypatch.setattr(mg, "run_multitarget_grid_search", _fake_grid)
        # Run the worker body synchronously for the test.
        gui_app._run_multitarget_search_thread(
            gui_app._collect_multitarget_config()
        )
        assert "PLS" in captured["model_names"]
        assert "preprocessing_methods" in captured
        assert "model_grid_overrides" in captured
        assert captured["optimization_method"] == "grid"
        # A SEPARATE controller instance is used, not the single-Y one.
        assert gui_app._multitarget_controller is not gui_app.search_controller

    def test_cancel_stops_multitarget_controller(self, gui_app):
        from spectral_predict.search_controller import SearchController
        gui_app._multitarget_controller = SearchController()
        gui_app._cancel_multitarget_search()
        assert gui_app._multitarget_controller.is_ended() or gui_app._multitarget_controller._stop_requested
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/gui/test_multitarget_tab.py -k "GridDispatch" -v -m gui`
Expected: FAIL (`_run_multitarget_search_thread`, `_collect_multitarget_config`, `_cancel_multitarget_search`, `_multitarget_controller` do not exist).

- [ ] **Step 3: Write minimal implementation**

Replace `_run_multitarget_search` (`:14759-14847`) and add helpers. `_collect_multitarget_config` gathers the inherited config into a dict; `_run_multitarget_search_thread` runs the orchestrator; `_cancel_multitarget_search` stops the separate controller. Build `model_grid_overrides` from the SAME HP local-builders `_run_analysis_thread` uses (factor them if needed, or read the `self.*` vars directly). Concretely:

```python
    def _collect_multitarget_config(self):
        """Assemble the inherited search config (same state the normal Run reads)."""
        import threading

        from spectral_predict.search_controller import SearchController

        targets = list(self.selected_targets)
        if len(targets) < 2:
            messagebox.showwarning("Multi-Target",
                "Select at least 2 numeric targets. For a single target, use the normal search.")
            return None
        source = None
        if getattr(self, "combined_metadata_df", None) is not None:
            source = self.combined_metadata_df
        elif getattr(self, "ref", None) is not None:
            source = self.ref
        if source is None or self.X is None:
            messagebox.showerror("Multi-Target", "Load spectral data and a reference table first.")
            return None
        model_names = [n for n, v in self.multitarget_model_vars.items() if v.get()]
        if not model_names:
            messagebox.showwarning("Multi-Target", "Select at least one model.")
            return None

        X_df = self.X
        Ydf = source[targets].reindex(X_df.index)
        keep = pd.Series(True, index=X_df.index)
        if self.active_indices is not None:
            keep &= X_df.index.isin(self.active_indices)
        if self.excluded_spectra:
            keep &= ~X_df.index.isin(self.excluded_spectra)
        if self.validation_enabled.get() and self.validation_indices:
            keep &= ~X_df.index.isin(self.validation_indices)
        mask = keep & Ydf.notna().all(axis=1) & X_df.notna().all(axis=1)
        if int(mask.sum()) < 3:
            messagebox.showerror("Multi-Target", "Not enough complete samples to cross-validate.")
            return None

        overrides = self._build_model_grid_overrides()  # mirrors run_search HP kwargs :29159-29227
        selected_varsel = self._collect_selected_varsel_methods()  # mirrors :28125
        variable_counts = self._collect_variable_counts()          # mirrors :26359
        wl_restriction = self._collect_wavelength_restriction()    # analysis_wl_* -> dict or None
        try:
            wavelengths = np.asarray(self.X.columns, dtype=float)  # numeric spectral columns
        except (ValueError, TypeError):
            wavelengths = np.arange(self.X.shape[1], dtype=float)  # synthetic/non-numeric columns

        return {
            "X": X_df.loc[mask].to_numpy(dtype=float),
            "Y": Ydf.loc[mask].to_numpy(dtype=float),
            "model_names": model_names, "target_names": targets,
            "wavelengths": wavelengths,
            "preprocessing_methods": self._collect_preprocessing_methods(),
            "autoscale": self.use_autoscale.get(),
            "baseline_method": getattr(self, "_baseline_method_value", None),
            "baseline_params": getattr(self, "_baseline_params_value", None),
            "smoothing": self.enable_smoothing.get(),
            "smoothing_window": self.smoothing_window.get(),
            "smoothing_polyorder": self.smoothing_polyorder.get(),
            "wavelength_restriction": wl_restriction,
            "variable_selection_methods": selected_varsel,
            "variable_counts": variable_counts or None,
            "ipls_subset_limit": self.ipls_subset_limit.get(),
            "tier": self._resolve_tier(), "model_grid_overrides": overrides,
            "max_n_components": self.max_n_components.get() if hasattr(self, "max_n_components") else 10,
            "max_iter": self.max_iter.get(),
            "cv": self.cv_strategy.get() or "kfold",
            "n_folds": int(self.folds.get() or 5),
            "n_repeats": int(self.cv_n_repeats.get() or 5),
        }

    def _run_multitarget_search(self):
        import threading

        from spectral_predict.search_controller import SearchController

        cfg = self._collect_multitarget_config()
        if cfg is None:
            return
        self.optimization_method.set("grid")
        self._multitarget_controller = SearchController()
        # Honest pre-run heads-up: n_preprocess x n_model_configs is a LOWER bound
        # (varsel subsets multiply it further). Cheap — no varsel is run here.
        try:
            from spectral_predict.models import get_model_grids
            from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs
            n_pp = len(build_multitarget_preprocess_configs(
                cfg["preprocessing_methods"], autoscale=cfg["autoscale"],
                baseline_method=cfg["baseline_method"], smoothing=cfg["smoothing"]))
            grids = get_model_grids(task_type="regression", n_features=cfg["X"].shape[1],
                                    tier=cfg["tier"], enabled_models=cfg["model_names"],
                                    **(cfg["model_grid_overrides"] or {}))
            n_cfg = sum(len(v) for v in grids.values())
            hint = f"  (≥ {n_pp * n_cfg} cells before variable selection — may take a while)"
        except Exception:
            hint = "  (running the full inherited grid — may take a while)"
        self.multitarget_status_label.config(text="Running…" + hint)
        self._multitarget_thread = threading.Thread(
            target=self._run_multitarget_search_thread, args=(cfg,), daemon=True)
        self._multitarget_thread.start()

    def _run_multitarget_search_thread(self, cfg):
        from spectral_predict.multitarget_grid import run_multitarget_grid_search

        X = cfg.pop("X"); Y = cfg.pop("Y")
        try:
            output = run_multitarget_grid_search(
                X, Y, optimization_method="grid",
                controller=self._multitarget_controller,
                progress_callback=self._multitarget_progress, **cfg)
        except Exception as exc:
            self.root.after(0, lambda: self._multitarget_failed(str(exc)))
            return
        self.root.after(0, lambda: self._multitarget_done(output))

    def _multitarget_progress(self, info):
        self.root.after(0, lambda: self.multitarget_status_label.config(
            text=f"{info.get('message', 'Running…')}"))

    def _multitarget_failed(self, msg):
        self.multitarget_status_label.config(text="Failed.")
        messagebox.showerror("Multi-Target Search Failed", msg)

    def _multitarget_done(self, output):
        self._multitarget_last_output = output
        self._populate_multitarget_results(output)
        note = ("  Skipped: " + ", ".join(output.skipped)) if output.skipped else ""
        self.multitarget_status_label.config(
            text=f"Done: {len(output.results)} cell(s), {output.n_targets} targets.{note}")

    def _cancel_multitarget_search(self):
        ctrl = getattr(self, "_multitarget_controller", None)
        if ctrl is not None:
            ctrl.stop()
        self.multitarget_status_label.config(text="Cancelling…")
```

Implement the small collector helpers (`_build_model_grid_overrides`, `_collect_selected_varsel_methods`, `_collect_variable_counts`, `_collect_wavelength_restriction`, `_collect_preprocessing_methods`, `_resolve_tier`) by copying the exact local-variable logic already present in `_run_analysis_thread` (varsel `:28125-28166`, variable_counts `:26359`, HP list locals feeding `:29159-29227`, tier resolution, `analysis_wl_*`). `_build_model_grid_overrides` returns a dict whose keys are EXACTLY the non-`X`/`y` HP kwargs of `get_model_grids` (e.g. `rf_bootstrap_list`, `mlp_activation_list`, `catboost_border_count_list`, `xgb_max_depths`, ...). Add a Cancel button wired to `self._cancel_multitarget_search` in `_create_tab4f_multitarget` next to the Run button.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/gui/test_multitarget_tab.py -k "GridDispatch or TestMultiTargetTab" -v -m gui`
Expected: PASS (new dispatch tests + existing tab tests).

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add spectral_predict_gui_optimized.py tests/gui/test_multitarget_tab.py
git commit -m "feat(T17-gui): multi-target grid dispatch on worker thread + separate controller + Cancel"
```

---

### Task 13: GUI — leaderboard columns, skip-notice, CSV export

**Files:**
- Modify: `spectral_predict_gui_optimized.py:14849-14914` (`_populate_multitarget_results`, `_export_multitarget_csv`)
- Test: `tests/gui/test_multitarget_tab.py` (append)

**Interfaces:**
- Consumes: `MultiTargetResult` new fields (`preprocessing`, `varsel_method`, `n_variables`, `params`), `MultiTargetSearchOutput.skipped`.
- Produces: leaderboard leading columns `preprocessing, varsel, #vars, key_hp, model, mode, joint_q2` then the per-target metric block; CSV gains `Preprocessing, Varsel_Method, N_Variables, Key_Hyperparameters` columns.

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/test_multitarget_tab.py`:

```python
@pytest.mark.gui
def test_leaderboard_shows_preprocess_varsel_nvars_columns(gui_app):
    from spectral_predict.multitarget_search import MultiTargetResult, MultiTargetSearchOutput

    res = MultiTargetResult(
        model_name="PLS", mode="JOINT", params={"n_components": 5}, joint_q2=0.8,
        metrics={"per_target": [{"target": "a", "r2": 0.8, "rmse": 1.0, "rpd": 2.0,
                                 "rer": 3.0, "ccc": 0.9, "bias": 0.0}],
                 "q2": np.array([0.8])},
        precise_note="", scale_y=True, mechanism="x",
        preprocessing="snv", varsel_method="ipls_forward", varsel_tag="fwd", n_variables=25,
    )
    out = MultiTargetSearchOutput(results=[res], target_names=["a"], correlation={},
                                  n_targets=1, skipped=["uve"])
    gui_app._populate_multitarget_results(out)
    cols = list(gui_app.multitarget_tree["columns"])
    assert "preprocessing" in cols
    assert "varsel" in cols
    assert "nvars" in cols
    row = gui_app.multitarget_tree.get_children()[0]
    values = gui_app.multitarget_tree.item(row, "values")
    assert "snv" in values
    assert "ipls_forward" in values
    assert "25" in values
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python.exe -m pytest tests/gui/test_multitarget_tab.py -k "leaderboard_shows" -v -m gui`
Expected: FAIL (columns `preprocessing`/`varsel`/`nvars` absent).

- [ ] **Step 3: Write minimal implementation**

In `_populate_multitarget_results` (`:14855`), extend the leading columns and row values:

```python
        columns = ["preprocessing", "varsel", "nvars", "key_hp", "model", "mode", "joint_q2"]
        for t in target_names:
            for key, _lbl in self._MULTITARGET_METRIC_KEYS:
                columns.append(f"{t}__{key}")
        tree.configure(columns=columns)
        for cid, text, w, anchor in [
            ("preprocessing", "Preprocess", 110, tk.W), ("varsel", "Varsel", 110, tk.W),
            ("nvars", "#Vars", 60, tk.E), ("key_hp", "Key HP", 140, tk.W),
            ("model", "Model", 110, tk.W), ("mode", "Mode", 100, tk.CENTER),
            ("joint_q2", "Joint Q²", 80, tk.E),
        ]:
            tree.heading(cid, text=text)
            tree.column(cid, width=w, anchor=anchor)
        for t in target_names:
            for key, lbl in self._MULTITARGET_METRIC_KEYS:
                col = f"{t}__{key}"
                tree.heading(col, text=f"{t} {lbl}")
                tree.column(col, width=90, anchor=tk.E)

        for res in output.results:
            per = {d["target"]: d for d in res.metrics.get("per_target", [])}
            key_hp = ", ".join(f"{k}={v}" for k, v in list(res.params.items())[:3])
            jq = f"{res.joint_q2:.4f}" if np.isfinite(res.joint_q2) else "NaN"
            values = [res.preprocessing, res.varsel_method,
                      "" if res.n_variables is None else str(res.n_variables),
                      key_hp, res.model_name, res.mode, jq]
            for t in target_names:
                d = per.get(t, {})
                for key, _lbl in self._MULTITARGET_METRIC_KEYS:
                    val = d.get(key)
                    values.append(f"{val:.4f}" if isinstance(val, (int, float)) else "")
            tree.insert("", tk.END, values=values)
```

In `_export_multitarget_csv` (`:14897`), extend the record:

```python
            record = {
                "Preprocessing": res.preprocessing,
                "Varsel_Method": res.varsel_method,
                "Varsel_Tag": res.varsel_tag,
                "N_Variables": res.n_variables,
                "Key_Hyperparameters": ", ".join(f"{k}={v}" for k, v in res.params.items()),
                "Model": res.model_name,
                "MultiTarget Mode": res.mode,
                "Joint_Q2": res.joint_q2,
                "Coupling_Mechanism": res.mechanism,
                "Precise_Note": res.precise_note,
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/gui/test_multitarget_tab.py -k "leaderboard_shows" -v -m gui`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add spectral_predict_gui_optimized.py tests/gui/test_multitarget_tab.py
git commit -m "feat(T17-gui): leaderboard preprocess/varsel/#vars/key-HP columns + skip-notice + CSV"
```

---

### Task 14: End-to-end integration + save/reload/predict round-trip

**Files:**
- Create: `tests/test_multitarget_integration.py`
- Test: itself

**Interfaces:**
- Consumes: `read_asd_dir` (`spectral_predict.io`), `run_multitarget_grid_search` (Task 11), `save_model`/`load_model`/`predict_with_model` (`model_io.py:87/:424/:641`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_multitarget_integration.py`:

```python
"""T-17 end-to-end integration: real ASD spectra + a constructed 2nd target."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def _load_example():
    from spectral_predict.io import read_asd_dir

    X, _meta = read_asd_dir("example")    # returns (df, metadata) tuple; df is (49, 2151)
    ref = pd.read_csv("example/BoneCollagen.csv")
    ref["_key"] = ref["File Number"].astype(str).str.replace(" ", "", regex=False)
    idx_key = [str(i).replace(" ", "") for i in X.index]
    y0 = pd.Series(ref.set_index("_key")["%Collagen"]).reindex(idx_key).to_numpy(float)
    mask = np.isfinite(y0)
    Xv = X.to_numpy(float)[mask]
    y0 = y0[mask]
    y1 = y0 * 1.7 + 3.0                    # constructed 2nd target (smoke-only)
    Y = np.column_stack([y0, y1])
    wl = np.asarray(X.columns, dtype=float)
    return Xv, Y, wl


@pytest.mark.skipif(not os.path.isdir("example"), reason="example data unavailable")
def test_multitarget_grid_end_to_end_real_data():
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = _load_example()
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["collagen", "synthetic"],
        wavelengths=wl, preprocessing_methods={"raw": True, "snv": True}, autoscale=False,
        variable_selection_methods=["ipls_forward"], variable_counts=[10],
        ipls_subset_limit="Top 3", tier="quick", cv="kfold", n_folds=3, n_repeats=1,
    )
    assert out.results
    assert np.isfinite(out.results[0].joint_q2)
    # A JOINT (PLS), an INDEPENDENT (Ridge), and a varsel cell all appear.
    modes = {r.mode for r in out.results}
    assert "JOINT" in modes and "INDEPENDENT" in modes
    assert any(r.varsel_method == "ipls_forward" for r in out.results)


@pytest.mark.skipif(not os.path.isdir("example"), reason="example data unavailable")
def test_multitarget_save_reload_predict_roundtrip(tmp_path):
    from sklearn.cross_decomposition import PLSRegression

    from spectral_predict.model_io import load_model, predict_with_model, save_model

    X, Y, _wl = _load_example()
    est = PLSRegression(n_components=5, scale=False).fit(X, Y)
    path = tmp_path / "mt_model.joblib"
    save_model(est, str(path), metadata={"multitarget_mode": "JOINT",
                                          "target_names": ["collagen", "synthetic"]})
    loaded = load_model(str(path))
    preds = predict_with_model(loaded, X[:5])
    preds = np.asarray(preds)
    assert preds.shape[0] == 5
    assert preds.ndim == 2 and preds.shape[1] == 2
```

- [ ] **Step 2: Run test to verify it fails (or is red for the right reason)**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_integration.py -v`
Expected: FAIL first only if an API mismatch exists (e.g. `save_model`/`predict_with_model` signature differs) — if so, adjust the call to the real `model_io` signatures (`save_model` `:87`, `predict_with_model` `:641`) without changing the round-trip intent. Once aligned, the grid test should PASS.

- [ ] **Step 3: Fix any API alignment**

If `save_model`/`predict_with_model` reject the shown kwargs, read `model_io.py:87` and `:641` and pass exactly the parameters they require (keep the JOINT metadata + 2-target prediction assertions).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python.exe -m pytest tests/test_multitarget_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add tests/test_multitarget_integration.py
git commit -m "test(T17): end-to-end multi-target grid + save/reload/predict round-trip"
```

---

## Review cadence (per spec §9)

Run these gates as you finish each phase — do NOT skip them; this project reviews at every gate. Give every reviewer the by-design list (single-Y byte-identity via additive branches; grid-only; chemometrics conventions are not leakage; UVE/CARS/iPLS-legacy/VCPA skip-with-notice; NaN-sink discipline).

- **After Phase A (Tasks 1–11):** Codex review of the backend diff (`codex-reviewer` agent or the codex-review skill). Add a second orthogonal reviewer (`opencode-call` on `glm` and/or `kimi27`, full repo access) because this phase touches CV/varsel math. Fold NEEDS-CHANGES before Phase B.
- **After Phase B (Tasks 12–13):** Codex review of the GUI diff (thread/controller/progress correctness).
- **After Phase C (Task 14):** Codex review of the integration diff.
- **Whole-diff before merge:** cross-family pass via parallel `opencode-call` agents on `glm` + `kimi27` (NOT the peer-review skill — that is for self-contained targets). Then run the full `pr-review-toolkit` (Claude-family specialists) — `silent-failure-hunter` is the highest-value one here given the NaN-sink history; also `code-reviewer` + `pr-test-analyzer`. Do BOTH layers; they are orthogonal.
- **Merge gate:** local diff-failure-set vs `origin/main` (Windows Py3.12 `.venv312`, `--ignore=tests/gui`; expect the 5 known pre-existing failures — `test_cv_strategy` nameerror, `test_export_code` ×2, `test_t19_class_weight` ×2 — and require ZERO new). Do NOT auto-merge PR #63 — wait for explicit user greenlight. Route implementation subtasks to GLM 5.2 (`opencode-call` write-mode) where useful; Opus keeps orchestration + chemometrics-gated judgment.
