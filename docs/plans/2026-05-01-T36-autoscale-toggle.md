# T-36: Autoscale (UV Scaling) Preprocessing Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add UV scaling (mean-center + unit variance per wavelength) as a user-selectable preprocessing option that doubles the config grid — when enabled, every selected preprocessing combination is tested both with and without autoscale, mirroring how baseline correction and smoothing toggles work today.

**Architecture:** Single config field `autoscale: bool` doubles `preprocess_configs` after they are built (same pattern as baseline at `search.py:1813-1828` and smoothing at `search.py:1830-1849`). When `autoscale=True`, a `StandardScaler` is appended at the end of the spectral preprocessing block in `build_preprocessing_pipeline()` (after SNV/derivatives, before imbalance). Per-model `StandardScaler` is skipped for `SCALE_SENSITIVE_MODELS` when autoscale is on (to avoid redundant scaling); PLS-DA's internal scaler stays (it operates on PLS scores, not features). The `apply_autoscale` toggle becomes a per-trial Optuna parameter in the Bayesian path.

**Tech Stack:** scikit-learn `StandardScaler`, existing dasp pipeline machinery. No new dependencies.

**Source:** Conversation 2026-04-30 — user empirically observed autoscale sometimes improves models even when SNV/derivatives are already applied. Field-aligned: SIMCA defaults to UV scaling for PLS; PLS_Toolbox and Unscrambler offer it as an option.

---

## Background

### Why autoscale is a real option (not redundant with existing scalers)

- dasp already runs `StandardScaler` per-model for `SCALE_SENSITIVE_MODELS` ({SVC, SVR, MLP, NeuralBoosted, Ridge, Lasso, ElasticNet}) at `search.py:4272`. But that scaler runs **after variable selection and imbalance handling** — variable selection sees un-autoscaled features, so importance rankings are dominated by high-variance wavelengths.
- True autoscale runs **before** variable selection, so VIP / CARS / SPA / iPLS see normalized features and select on chemistry-relevant rather than variance-dominant grounds.
- For tree models (RF/XGBoost/LightGBM/CatBoost), autoscale is mathematically a no-op (scale-invariant). Doubling configs for tree models is wasted compute — addressed in Phase 1.5 (out of scope for this ticket).
- For PLS / PLS-DA, autoscale is genuinely new behavior — PLS does not internally scale features. This is the model where the user is most likely seeing improvement.

### Pre-existing bugs uncovered during plan review

The original plan (DeepSeek V4 Pro draft) had three implementation bugs that would have shipped non-functional autoscale in part of the search surface. They are pinned here so they get fixed:

1. **`apply_preprocessing()` early-return bug** (`unified_bayesian.py:288-376`). Function returns early in every named-preprocessing branch (`raw`, `snv`, `'deriv' in name`). A trailing `if config.get('apply_autoscale')` block at the bottom would be unreachable. **Fix:** restructure each branch to assign `X = ...` rather than `return ...`, then apply autoscale, then `return X`.

2. **`unified_bayesian.py` preprocessing cache key omits autoscale** (`:857-864`). With `apply_autoscale` as a per-trial Optuna toggle, two trials differing only in autoscale would collide on the cache and the second trial would receive the first's `X_prep`. **Fix:** add `preprocess_config.get('apply_autoscale', False)` to the cache_key tuple.

3. **`contamination.py` validation cache key omits autoscale** (`:1109-1112`). Same pattern as `search.py:599` regression-validation key — must mirror the fix in plan step L. **Fix:** append `autoscale` to the cache_key tuple.

### Adjacent gap to bundle

`run_one_class_search` writes result rows at `search.py:5487-5535` and `~6014` that omit `baseline_method`, `smoothing`, `smoothing_window`, `smoothing_polyorder` columns. `compute_validation_metrics_for_top_one_class_models` (`contamination.py:1098-1107`) reads these from `row.get(...)` and falls back to defaults — meaning one-class validation today silently uses default smoothing/baseline regardless of what the search ran. Bundle the metadata-write fix into this ticket since we are touching the same code surface for `Autoscale`.

### Out-of-scope decisions (recorded for clarity)

| Item | Reason |
|---|---|
| NSGA-II support (`nsga2_search.py`) | Separate code path with its own `build_preprocessing_pipeline()` calls; follow-up |
| `run_bayesian_search` (`search.py:3153`) | Test-only path (no GUI caller); add autoscale support if/when it becomes user-facing |
| Tree-model skip in doubling loop | Phase 1.5 refinement — tree models are scale-invariant so autoscale doubling is wasted compute, but skipping needs a cross-cut check that the doubled config still flows correctly through varsel + imbalance. Defer. |
| Smart-preprocessing-discovery integration | Subsumed by T-37 (TPE quick-discovery). Don't bolt onto basic discovery. |

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| Grid doubling via config field `autoscale: bool` | Same pattern as smoothing (`search.py:1830-1849`) and baseline (`:1813-1828`). Predictable for users, predictable for code review. |
| Autoscale step appended after SNV/derivatives, before imbalance | Order: Interference → Baseline → Smoothing → SNV/SG-deriv → **Autoscale** → Imbalance. Imbalance must see normalized features to compute SMOTE neighbors etc. correctly. |
| Per-model `StandardScaler` skipped when `autoscale=True` | At `search.py:4272` and `unified_bayesian.py:1224-1227`. Prevents redundant scaling. PLS-DA scaler stays (operates on PLS scores, not features — different scaler). |
| One-class double-scaling tolerated | `contamination.py`'s per-fold StandardScaler at lines 650-654, 797-800, 1217-1220 is mathematically a no-op on already-scaled data. Threading a flag through three conditioning blocks adds risk for zero performance benefit. |
| `.dasp` boolean roundtrip: no code needed | `pd.to_csv()` writes `True`/`False`; `pd.read_csv()` infers bool. Same pattern as existing `smoothing` column. |
| Name suffix `+autoscale` in `Preprocess` column | Lets users filter result rows by `Autoscale` column AND by display name. `PreprocessBase` stays clean (e.g. `snv`) so `build_preprocessing_pipeline` validation rebuild still works. |

---

## Tasks

### Task 1: `preprocess.py` — add autoscale parameter and step

- [ ] Add `autoscale: bool = False` to `build_preprocessing_pipeline()` signature (line 283) after `smoothing_polyorder=2`.
- [ ] Add docstring entry before `Returns`.
- [ ] After the SNV/deriv `if/elif` block (after line 548 `raise ValueError`, before line 550 imbalance block), insert:
  ```python
  if autoscale:
      from sklearn.preprocessing import StandardScaler
      steps.append(("autoscale", StandardScaler()))
  ```
- [ ] Update the "Notes / Processing order" section of the docstring to include Autoscale.

### Task 2: `tests/test_preprocess_extended.py` — autoscale unit tests

- [ ] `test_autoscale_step_present` — `build_preprocessing_pipeline("snv", autoscale=True)` produces a step named `"autoscale"`.
- [ ] `test_autoscale_step_order` — autoscale comes after SNV / SG-deriv when both present.
- [ ] `test_autoscale_false_by_default` — no autoscale step when flag omitted.
- [ ] `test_autoscale_after_imbalance_not_present` — autoscale step is BEFORE imbalance step when both enabled.

### Task 3: `search.py` — `run_search` signature + grid-doubling block

- [ ] Add `autoscale=False` to `run_search()` signature after `smoothing_polyorder=2` (line 914).
- [ ] After the smoothing-doubling block (line 1849), before CV splitter creation (line 1851), insert the autoscale-doubling block:
  ```python
  # --- Autoscale toggle: when enabled, test both WITH and WITHOUT autoscaling ---
  if autoscale and preprocess_configs:
      configs_without = []
      configs_with = []
      for cfg in preprocess_configs:
          cfg_no = dict(cfg)
          cfg_no["autoscale"] = False
          configs_without.append(cfg_no)
          cfg_sc = dict(cfg)
          cfg_sc["autoscale"] = True
          cfg_sc["base_name"] = cfg.get("base_name", cfg["name"])
          cfg_sc["name"] = cfg["name"] + "+autoscale"
          configs_with.append(cfg_sc)
      preprocess_configs = configs_without + configs_with
  ```

### Task 4: `search.py` — propagate to `build_preprocessing_pipeline()` calls

- [ ] At line 1964 (global preprocessing call inside `run_search` main loop), add `autoscale=preprocess_cfg.get("autoscale", False)` after `smoothing_polyorder=...`.
- [ ] At line 4191 (`_run_single_config`'s pipeline build), same addition.
- [ ] **Decide and document:** `search.py:3543` (inside `run_bayesian_search`, line 3153). User-facing? If yes, add autoscale propagation. If no, add a comment noting this path is test-only. (Initial verdict from code review: test-only — no GUI caller. Document and skip.)

### Task 5: `search.py` — skip per-model `StandardScaler` when autoscale active

- [ ] At `search.py:4272`, change:
  ```python
  elif model_name in SCALE_SENSITIVE_MODELS:
      pipe_steps.append(("scaler", StandardScaler()))
      pipe_steps.append(("model", clone(model)))
  ```
  to:
  ```python
  elif model_name in SCALE_SENSITIVE_MODELS:
      if not preprocess_cfg.get("autoscale", False):
          pipe_steps.append(("scaler", StandardScaler()))
      pipe_steps.append(("model", clone(model)))
  ```
- [ ] **Do not** modify the PLS-DA branch at `:4248-4269` — its scaler is on PLS scores, not features.

### Task 6: `search.py` — results column + varsel cache key

- [ ] At line 4685 result dict, add `"Autoscale": preprocess_cfg.get("autoscale", False),`.
- [ ] At `_varsel_cache_key`'s `prep_parts` tuple (`:1915-1922`), add `preprocess_cfg.get('autoscale', False)`.

### Task 7: `search.py` — `compute_validation_metrics_for_top_models` (regression/classification)

- [ ] After line 529 (`smoothing_polyorder` extraction), add `autoscale = bool(row.get('Autoscale', False))`.
- [ ] At cache_key tuple (line 599), append `autoscale` as the 9th element.
- [ ] At `build_preprocessing_pipeline()` call (line 612-621), add `autoscale=autoscale,`.

### Task 8: `search.py` — `run_one_class_search` signature + doubling

- [ ] Add `autoscale=False` to `run_one_class_search()` signature after `ga_n_runs=5` (line 5063).
- [ ] After the smart/manual config builders (after line 5296), before the CV splitter, insert the doubling block (mirroring Task 3 but using `method` field — one-class configs have `method` not `base_name`).

### Task 9: `search.py` — `run_one_class_search` propagate + result columns

- [ ] Propagate `autoscale=preprocess_cfg.get("autoscale", False)` to both `build_preprocessing_pipeline()` calls (lines 5385-5395 full-spectrum, 5601-5611 varsel fallback).
- [ ] Add `"Autoscale": preprocess_cfg.get("autoscale", False),` to both one-class result dicts (line 5487-5535 full-spectrum, 6014-... varsel).
- [ ] **Bundle adjacent fix:** Add `"baseline_method"`, `"smoothing"`, `"smoothing_window"`, `"smoothing_polyorder"` to both one-class result dicts so validation rebuild reads actual values rather than defaults. (Pre-existing bug — see Background.)

### Task 10: `contamination.py` — validation metrics path

- [ ] After line 1107 (`smoothing_polyorder` extraction), add `autoscale = bool(row.get('Autoscale', False))`.
- [ ] At cache_key (line 1109-1112), append `autoscale`.
- [ ] At `build_preprocessing_pipeline()` call (line 1118), add `autoscale=autoscale,`.

### Task 11: `unified_bayesian.py` — `suggest_preprocessing` and `apply_preprocessing`

- [ ] Add `enable_autoscale: bool = False` parameter to `suggest_preprocessing()` (line 212).
- [ ] At end of `suggest_preprocessing()` (before `return config`, ~line 285), add:
  ```python
  if enable_autoscale:
      config['apply_autoscale'] = trial.suggest_categorical('apply_autoscale', [True, False])
  else:
      config['apply_autoscale'] = False
  ```
- [ ] **Restructure `apply_preprocessing()` to fix early-return bug (see Background bug #1).** Change each branch from `return ...` to `X = ...; ` and add a single trailing `if config.get('apply_autoscale'): X = StandardScaler().fit_transform(X)` followed by `return X`.

### Task 12: `unified_bayesian.py` — cache key fix + per-trial pipeline

- [ ] **Add `apply_autoscale` to preprocessing_cache key tuple (line 857-864).** This is bug #2 from Background. Add `preprocess_config.get('apply_autoscale', False)` as the 7th element.
- [ ] At `apply_preprocessing()` call site (line 870-876), pass through the new behavior (no signature change — `apply_autoscale` is in `config`).
- [ ] At `:1224-1226` (SCALE_SENSITIVE_MODELS branch), guard the StandardScaler with `if not preprocess_config.get('apply_autoscale', False):` — mirror search.py task 5.
- [ ] At `set_user_attr` blocks (~line 1052 and ~line 1416), add `trial.set_user_attr('autoscale', preprocess_config.get('apply_autoscale', False))`.
- [ ] At `convert_study_to_dataframe()` (~line 2100), add `'Autoscale': trial.user_attrs.get('autoscale', False),` to the row dict.

### Task 13: `unified_bayesian.py` — `run_unified_bayesian` + `create_unified_objective` signatures

- [ ] Add `enable_autoscale: bool = False` to `run_unified_bayesian()` signature (line ~1599) and `create_unified_objective()` (line 718).
- [ ] Thread through to the `suggest_preprocessing()` call at line 850.
- [ ] Thread through to `create_unified_objective()` call at line 1782.

### Task 14: `contamination.py` — cache key fix

- [ ] **Add `autoscale` to `cache_key` tuple at lines 1109-1112.** This is bug #3 from Background.

### Task 15: `spectral_predict_gui_optimized.py` — UI wiring

- [ ] Add `self.use_autoscale = tk.BooleanVar(value=False)` (after `self.use_deriv_snv` at line 3131).
- [ ] Add checkbox widget after deriv_snv checkbox (~line 11587):
  ```python
  self.autoscale_checkbox = ttk.Checkbutton(
      preprocess_frame,
      text="Autoscale (UV scaling: mean-center + unit variance per wavelength)",
      variable=self.use_autoscale
  )
  self.autoscale_checkbox.grid(row=7, column=0, sticky=tk.W, pady=5)
  ```
- [ ] Add tooltip explaining: when enabled, every selected preprocessing combination is tested both with and without autoscale; useful when wavelength variances differ substantially after SNV/derivatives.
- [ ] Pass `autoscale=self.use_autoscale.get()` to `run_search()` (~line 27623).
- [ ] Pass `autoscale=self.use_autoscale.get()` to `run_one_class_search()` (~line 26985).
- [ ] Pass `enable_autoscale=self.use_autoscale.get()` to both `run_unified_bayesian()` calls (~line 26899 and ~line 27234).

### Task 16: Integration tests

- [ ] `test_autoscale_grid_doubling.py` — `run_search` with autoscale=True produces 2× the result rows of autoscale=False, half with `Autoscale=True`.
- [ ] `test_autoscale_bayesian_cache.py` — manually trigger a Bayesian trial with `apply_autoscale=True` and another with `apply_autoscale=False` having same other params; verify `X_prep` differs (cache key fix verification).
- [ ] `test_autoscale_one_class_validation.py` — one-class top-N with mixed Autoscale values rebuilds correctly.

### Task 17: Validation note

- [ ] Write `docs/bugfix_validation/T36_autoscale_toggle.md` documenting the field-alignment check (SIMCA default vs PLS_Toolbox/Unscrambler optional), the empirical motivation, and the test sweep results.

### Task 18: PR

- [ ] Run targeted tests: `pytest tests/test_preprocess_extended.py tests/test_autoscale_*.py -v`.
- [ ] Run full sweep before merge.
- [ ] Open PR with summary referencing this plan and the bug-fix list.

---

## Verification checklist

- [ ] Grid path: enabling autoscale doubles result-row count; `Autoscale` column populated correctly.
- [ ] Bayesian path: trials with same preprocessing but different `apply_autoscale` produce different metrics (cache key works).
- [ ] One-class path: validation rebuild for an Autoscale=True top-N row produces same metrics as the original CV.
- [ ] Per-model scaler: `SCALE_SENSITIVE_MODELS` pipe has only one StandardScaler, not two, when autoscale=True.
- [ ] PLS-DA: pipe still has its internal scaler regardless of autoscale (verify pipe length).
- [ ] `.dasp` save/load: `Autoscale` column roundtrips via `pd.to_csv` / `pd.read_csv`.
- [ ] Old `.dasp` files without `Autoscale` column still load (defaults to False).

## Known follow-ups (out of scope, file as separate tickets if needed)

- Tree-model skip in doubling loop (Phase 1.5)
- NSGA-II support
- T-37 TPE quick-discovery (subsumes this as one dimension)
