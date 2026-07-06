# Multi-Class SIMCA UX Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multi-class SIMCA configure, sweep, report, validate, and present results like its sibling methods (especially one-class SIMCA), instead of via a bespoke import-page panel with single-value knobs and an auto-popup.

**Architecture:** Two layers. (1) Backend: `run_multiclass_simca_search` becomes list-capable (alpha × n_components × varsel-size grid axes), the results schema gains `NComponents`, the full variable-selection set is wired through the existing precomputed-mask hook on `MultiClassClassModel`, and `compute_validation_metrics_for_top_models` gains a `multiclass_simca` branch. (2) GUI: the config panel moves off the Import page into the 4A Model Config + 4B Variable Selection subtabs (mirroring one-class), alpha/n_components/variable-count become swept lists reusing one-class's collector/parser, the leaderboard reports the swept knobs, the Validation tab is wired in, and the run-completion auto-popup is removed.

**Tech Stack:** Python 3.12, numpy/pandas/scikit-learn, Tkinter GUI (`spectral_predict_gui_optimized.py`), pytest. Backend in `src/spectral_predict/` (`search.py`, `simca.py`, `scoring.py`, `unified_bayesian.py`).

## Global Constraints

- **Python 3.12 only.** Install/test in `.venv312`.
- **Black, line-length 100.** Type hints on all new functions; `from __future__ import annotations` where forward refs are needed; built-in generics (`list[str]`).
- **Byte-identical existing paths.** Single-Y (regression/classification/one-class) search/scoring/validation paths must be unchanged. Multi-class scalar callers (tests, saved-model reload, `_run_selected_multiclass_result`, `_fit_and_save_multiclass_model`) must produce identical results when a single value is passed — enforced by scalar-or-list normalization.
- **No new dependencies.**
- **Chemometrics honesty copy is required, not optional** (novelty caveat on discrimination-based varsel; known-class-only caveat on holdout validation). Copy text is specified in the relevant tasks verbatim.
- **Don't run the full test suite for small changes** — use targeted `pytest` per task (`py_compile` for GUI-only edits that have no headless test).
- **Spec:** `docs/superpowers/specs/2026-07-06-multiclass-simca-parity-design.md` (sections A–J). Each task cites its section.

---

## File Structure

- `src/spectral_predict/search.py` — `run_multiclass_simca_search` list-capable grid (F); `compute_validation_metrics_for_top_models` multiclass branch + new helper `_multiclass_holdout_metrics` (I).
- `src/spectral_predict/scoring.py` — multiclass schema gains `NComponents`; `display_cols` gains swept knobs (H).
- `src/spectral_predict/simca.py` — extend `MultiClassClassModel.variable_selection` string dispatch to accept the full supervised method set by computing an external mask (E); or add a search-layer mask builder (chosen: search-layer, keeps the model layer thin).
- `spectral_predict_gui_optimized.py` — new sweep state vars; relocate panels to 4A/4B; varsel group UI; run-handler list wiring; validation wiring; remove auto-popup; reporting columns + tooltips + decision-view header.
- `tests/test_multiclass_search.py` — backend grid/normalization/reporting/varsel/validation tests.
- `tests/gui/test_multiclass_gui_parity.py` (new) — headless GUI-state tests.

---

## Phase 1 — Backend: list-capable search grid + reporting schema

### Task 1: `NComponents` column in the multiclass results schema

**Files:**
- Modify: `src/spectral_predict/scoring.py:646-654` (multiclass `metric_cols`)
- Test: `tests/test_multiclass_search.py`

**Interfaces:**
- Produces: multiclass results schema includes a `"NComponents"` column between `"Alpha"` and `"MinClassN"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiclass_search.py
from spectral_predict.scoring import create_results_dataframe

def test_multiclass_schema_has_ncomponents_column():
    df = create_results_dataframe(task_type="multiclass_simca")
    assert "NComponents" in df.columns
    # Ordered right after Alpha for readability
    cols = list(df.columns)
    assert cols.index("NComponents") == cols.index("Alpha") + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py::test_multiclass_schema_has_ncomponents_column -v`
Expected: FAIL — `"NComponents" not in df.columns`.

- [ ] **Step 3: Add the column**

In `scoring.py`, the multiclass `metric_cols` list, insert `"NComponents"` immediately after `"Alpha"`:

```python
        metric_cols = [
            "NoveltyAUC", "Efficiency", "NoveltyRate", "NoClassRate",
            "AmbiguityRate", "ExactSetRate", "MeanSensitivity", "MeanSpecificity",
            "Alpha", "NComponents", "MinClassN", "n_classes",
            "engine_family", "varsel_path",
            "unmodelable_classes", "reason",
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py::test_multiclass_schema_has_ncomponents_column -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_predict/scoring.py tests/test_multiclass_search.py
git commit -m "feat(T31): add NComponents column to multiclass results schema"
```

---

### Task 2: scalar-or-list normalization + grid expansion in `run_multiclass_simca_search`

**Files:**
- Modify: `src/spectral_predict/search.py:7379-7403` (signature + grid build)
- Test: `tests/test_multiclass_search.py`

**Interfaces:**
- Consumes: existing `run_multiclass_simca_search` internals (per-row fit, OOF metrics, LOCO AUC).
- Produces: `alpha`, `n_components`, `variable_selection_n_select` accept a scalar OR a list. Grid = `preprocessing × engines × varsel_paths × sizes × alphas × n_components`. Each row emits its own `Alpha` and `NComponents`. A single scalar for each reproduces the pre-change row set exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multiclass_search.py
import numpy as np
from spectral_predict.search import run_multiclass_simca_search

def _toy():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 40)
    y = np.array(["A", "B", "C"] * 20)
    return X, y

def test_scalar_alpha_ncomp_matches_single_row_group():
    X, y = _toy()
    df = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=0.05, n_components=0.99, varsel_paths=["none"],
    )
    assert (df["Alpha"] == 0.05).all()
    assert (df["NComponents"].astype(str) == "0.99").all()

def test_list_alpha_expands_grid():
    X, y = _toy()
    df1 = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=0.05, n_components=0.99, varsel_paths=["none"],
    )
    df2 = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=[0.01, 0.05], n_components=[0.95, 0.99], varsel_paths=["none"],
    )
    # 2 alphas x 2 n_components = 4x the single-value row count
    assert len(df2) == 4 * len(df1)
    assert set(df2["Alpha"].unique()) == {0.01, 0.05}
    assert set(df2["NComponents"].astype(str).unique()) == {"0.95", "0.99"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py -k "scalar_alpha or list_alpha" -v`
Expected: `test_list_alpha_expands_grid` FAILS (grid does not multiply; `alpha` list unhashable in current scalar use). `test_scalar_alpha...` may pass or fail depending on current NComponents emission — if `NComponents` isn't emitted yet it FAILS on the `.all()`.

- [ ] **Step 3: Normalize inputs and expand the grid**

At the top of `run_multiclass_simca_search` body (after the docstring), add:

```python
    def _as_list(v):
        if v is None:
            return [None]
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    alphas = _as_list(alpha)
    n_components_list = _as_list(n_components)
    n_select_list = _as_list(variable_selection_n_select)
```

Then wrap the existing per-row construction loop so it iterates the added axes. Where the code currently builds one row per `(preprocess_cfg, engine, varsel_path)`, nest:

```python
    for _alpha in alphas:
        for _ncomp in n_components_list:
            for _n_select in n_select_list:
                # ... existing body, substituting:
                #   alpha        -> _alpha
                #   n_components -> _ncomp
                #   variable_selection_n_select -> _n_select
                # and set on each emitted row dict:
                #   row["Alpha"] = _alpha
                #   row["NComponents"] = _ncomp
```

(Every place that previously read the scalar `alpha` / `n_components` / `variable_selection_n_select` now reads the loop variable. Grep the function body for those three names and replace within the loop.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py -k "scalar_alpha or list_alpha" -v`
Expected: PASS both.

- [ ] **Step 5: Regression — existing multiclass search tests unchanged**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py tests/test_simca.py -q`
Expected: PASS (pre-existing tests green; scalar path byte-identical).

- [ ] **Step 6: Commit**

```bash
git add src/spectral_predict/search.py tests/test_multiclass_search.py
git commit -m "feat(T31): list-capable alpha/n_components/size grid in multiclass search"
```

---

### Task 3: emit `NComponents` per row + surface swept knobs in `display_cols`

**Files:**
- Modify: `src/spectral_predict/scoring.py:233-236` (multiclass `display_cols`)
- Test: `tests/test_multiclass_search.py`

**Interfaces:**
- Consumes: rows carrying `Alpha`, `NComponents`, `engine_family`, `varsel_path`, `n_vars` (Task 2 + existing).
- Produces: multiclass `display_cols` shows `Alpha`, `NComponents`, `Engine` (from `engine_family`), `VarSelMethod` (from `varsel_path`) alongside `n_vars`.

- [ ] **Step 1: Write the failing test**

```python
def test_multiclass_display_cols_show_swept_knobs():
    from spectral_predict.scoring import rank_and_score_results  # or the fn owning display_cols
    X, y = _toy()
    df = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=[0.01, 0.05], n_components=[0.95, 0.99], varsel_paths=["none"],
    )
    # The ranked display view must expose the swept dimensions
    for col in ("Alpha", "NComponents"):
        assert col in df.columns
        assert df[col].nunique() >= 2
```

- [ ] **Step 2: Run test to verify current state**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py::test_multiclass_display_cols_show_swept_knobs -v`
Expected: PASS on column presence (from Task 2) — this test pins the data. The *display* wiring is asserted in the GUI phase; here we ensure the columns exist and vary.

- [ ] **Step 3: Update `display_cols`**

In `scoring.py`, replace the multiclass `display_cols` block:

```python
        elif task_type == "multiclass_simca":
            display_cols = ["Rank", "Model", "NoveltyAUC", "Efficiency",
                           "Alpha", "NComponents", "engine_family", "varsel_path",
                           "MinClassN", "n_vars", "PerformanceScore",
                           "VarPenalty", "GapPenalty", "CompositeScore"]
```

(The GUI renames `engine_family`→`Engine` and `varsel_path`→`VarSelMethod` at display time in Phase 8; the backend keeps the canonical column names.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py::test_multiclass_display_cols_show_swept_knobs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_predict/scoring.py tests/test_multiclass_search.py
git commit -m "feat(T31): surface swept alpha/n_components/engine/varsel in multiclass leaderboard"
```

---

## Phase 2 — Backend: wire the full variable-selection set (spec E)

> **Methodology (decided with user, 2026-07-06):** variable selection is
> computed **once on the calibration set**, per chemometrics convention — this
> is NOT an ML leakage bug. Major packages (PLS_Toolbox GA, mdatools iPLS,
> CARS) select wavelengths from the full calibration data using each method's
> own internal CV, then validate. Honest performance comes from the external
> **Validation tab** (Task 11), not from refitting selection inside every CV
> fold. Wold modes stay on the model's native per-class path (intrinsic SIMCA
> modeling power, not discrimination selection). `compute_importances` only
> implements importance/cars/uve, so the full set is wired by calling
> `variable_selection.py`'s real implementations directly.

### Task 4: full-set search-layer varsel-mask dispatch

**Files:**
- Modify: `src/spectral_predict/search.py` (helper `_multiclass_varsel_mask`; call site in the grid row builder). Builds on commit `8b82c73` (the narrow importance/cars/uve version) — broaden it to the full set.
- Test: `tests/test_multiclass_search.py`

**Interfaces:**
- Consumes: `variable_selection.py`'s real selectors (already imported into `search.py` at lines 91–104): `uve_selection`, `spa_selection`, `cars_selection`, `ipls_selection`, `ipls_forward`, `ipls_backward`, `mc_sipls`, `mwpls`, `uve_spa_selection`, `uve_cars_selection`, `uve_cars_spa_selection`, `fipls_spa_selection`, `fipls_cars_selection`. Reference the STANDARD dispatch at `search.py:3138–3510` for each function's exact call signature (which need `wavelengths`, which return an importance array vs. interval `subsets`, which take `task_type=`).
- Produces: `_multiclass_varsel_mask(X, y, wavelengths, method, n_select, task_type="classification") -> np.ndarray[bool] | str | None`. Returns `None` for `"none"`; the method **string** for Wold modes (model-native); a boolean mask of shape `(n_features,)` for every discrimination-based method by calling that method's real implementation on the calibration matrix and normalizing its output (top-`n_select` of an importance array, or the selected indices of the best interval subset) to a mask. Raises `MulticlassVarselUnsupported` (caught by caller → skip with log) only when a method genuinely errors on the >2-class label.

- [ ] **Step 1: Write the failing tests**

```python
def _toy():
    import numpy as np
    rng = np.random.RandomState(0)
    return rng.rand(60, 40), np.array(["A", "B", "C"] * 20)

def test_varsel_mask_none_and_wold():
    from spectral_predict.search import _multiclass_varsel_mask
    X, y = _toy(); wl = None
    assert _multiclass_varsel_mask(X, y, wl, "none", 10) is None
    assert _multiclass_varsel_mask(X, y, wl, "wold_modeling", 10) == "wold_modeling"

def test_varsel_mask_importance_style_methods_return_masks():
    import numpy as np
    from spectral_predict.search import _multiclass_varsel_mask
    X, y = _toy(); wl = np.arange(40)
    for method in ("importance", "cars", "uve", "spa"):
        m = _multiclass_varsel_mask(X, y, wl, method, 10)
        assert isinstance(m, np.ndarray) and m.dtype == bool and m.shape == (40,)
        assert 1 <= m.sum() <= 40

def test_varsel_mask_interval_method_returns_mask():
    import numpy as np
    from spectral_predict.search import _multiclass_varsel_mask
    X, y = _toy(); wl = np.arange(40)
    m = _multiclass_varsel_mask(X, y, wl, "ipls_forward", 10)
    assert isinstance(m, np.ndarray) and m.dtype == bool and m.shape == (40,) and m.sum() >= 1

def test_varsel_mask_unsupported_skips_cleanly():
    from spectral_predict.search import _multiclass_varsel_mask, MulticlassVarselUnsupported
    X, y = _toy(); import numpy as np; wl = np.arange(40)
    try:
        m = _multiclass_varsel_mask(X, y, wl, "definitely_not_a_method", 10)
        assert m is None or getattr(m, "dtype", None) == bool
    except MulticlassVarselUnsupported:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py -k varsel_mask -v`
Expected: FAIL — spa/ipls_forward not yet handled (current helper raises `MulticlassVarselUnsupported` for them).

- [ ] **Step 3: Broaden the helper**

Replace the `_MULTICLASS_MASK_METHODS = frozenset({"importance","cars","uve"})` gate and the single `compute_importances` call with a dispatch that mirrors `search.py:3138–3510`. Two output shapes to normalize:

- **Importance-array methods** (`importance`→`compute_importances`; `spa`→`spa_selection(X,y,n_features=n_select)`; `uve`→`uve_selection`; `cars`→`cars_selection(...,task_type=)`; `uve_spa`/`uve_cars`/`uve_cars_spa`; `ipls`→`ipls_selection`; `fipls_spa`/`fipls_cars`): call the function exactly as the standard path does, take the returned per-feature score array, keep the top-`n_select` indices → boolean mask.
- **Interval-subset methods** (`ipls_forward`/`ipls_backward`/`mc_sipls`/`mwpls`): call with `wavelengths=`; the return is a list of interval index groups — select the best/first group's indices → boolean mask (these define their own count; `n_select` does not apply, and that's expected).
- `cars_tree` → `cars_selection(..., use_hybrid_importance=True)`; `vcpa`/`ga` → if no real implementation is reachable in `variable_selection.py`, `raise MulticlassVarselUnsupported` (caught → skip) and note it in the report so Task 8's GUI list can drop the genuinely-absent ones.

Wrap each call in `try/except Exception` → `raise MulticlassVarselUnsupported(...)` so any per-method failure on the >2-class label degrades to a clean skip. Keep the label int-encoding the current helper already added. Selection runs **once on the passed calibration matrix** (chemometrics convention — document this in the docstring).

- [ ] **Step 4: Update the caller to pass `wavelengths`**

At the call site, pass the run's `wavelengths` through and keep the existing `try/except MulticlassVarselUnsupported: continue` skip-with-warning. Confirm `varsel_path` is still tagged with the method name and that a genuinely-absent method skips rather than crashes the run.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py -k varsel_mask -v`
Expected: PASS.

- [ ] **Step 6: Regression**

Run: `.venv312/Scripts/python -m pytest tests/test_simca.py tests/test_multiclass_search.py -q`
Expected: PASS; Wold-family results unchanged. Record in the report which method names genuinely resolve vs. skip (feeds Task 8's GUI group list — do not offer a method the backend can only skip).

- [ ] **Step 7: Commit**

```bash
git add src/spectral_predict/search.py tests/test_multiclass_search.py
git commit -m "feat(T31): wire full varsel set into multiclass search via variable_selection.py (select-on-calibration)"
```

---

## Phase 3 — Backend: holdout validation for multiclass (spec I)

### Task 5: `multiclass_simca` branch in `compute_validation_metrics_for_top_models`

**Files:**
- Modify: `src/spectral_predict/search.py:574-693+` (init cols + per-row loop) and add helper `_multiclass_holdout_metrics`
- Test: `tests/test_multiclass_search.py`

**Interfaces:**
- Consumes: `MultiClassClassModel` (fit on calibration split, `decision_matrix`/predict on holdout); `multiclass_simca_metrics` from `simca.py`.
- Produces: for `task_type == "multiclass_simca"`, `df_results` gains `val_MeanSensitivity`, `val_MeanSpecificity`, `val_NoveltyRate`, `val_AmbiguityRate`, `val_ExactSetRate` columns, computed on the holdout for the top-N rows.

- [ ] **Step 1: Write the failing test**

```python
def test_multiclass_holdout_metrics_populate():
    from spectral_predict.search import (
        run_multiclass_simca_search, compute_validation_metrics_for_top_models,
    )
    import numpy as np
    rng = np.random.RandomState(1)
    X = rng.rand(90, 40); y = np.array(["A", "B", "C"] * 30)
    idx = rng.permutation(90); tr, va = idx[:70], idx[70:]
    df = run_multiclass_simca_search(
        X[tr], y[tr], engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=0.05, n_components=0.99, varsel_paths=["none"],
    )
    out = compute_validation_metrics_for_top_models(
        df, X[tr], y[tr], X[va], y[va],
        task_type="multiclass_simca", wavelengths=np.arange(40), top_n=5,
    )
    assert "val_MeanSensitivity" in out.columns
    assert out["val_MeanSensitivity"].notna().any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py::test_multiclass_holdout_metrics_populate -v`
Expected: FAIL — no `val_MeanSensitivity` column (falls into the `else` classification branch and errors or NaNs).

- [ ] **Step 3: Add the init-columns branch**

In the "Initialize columns" block, add before the `else`:

```python
    if task_type == "regression":
        df_results["RMSEP"] = np.nan
        df_results["R2pred"] = np.nan
    elif task_type == "multiclass_simca":
        for c in ("val_MeanSensitivity", "val_MeanSpecificity", "val_NoveltyRate",
                  "val_AmbiguityRate", "val_ExactSetRate"):
            df_results[c] = np.nan
    else:
        df_results["val_Accuracy"] = np.nan
        ...
```

- [ ] **Step 4: Add the per-row multiclass compute + helper**

Add a helper and call it in the top-N loop when `task_type == "multiclass_simca"` (mirroring how the regression/classification branches rebuild per row). The helper rebuilds the row's exact config from its `Alpha`/`NComponents`/`varsel_path`/`engine_family`/preprocess columns (same logic `_run_selected_multiclass_result` already uses), fits on `(X_train, y_train)`, scores the holdout:

```python
def _multiclass_holdout_metrics(row, X_train, y_train, X_val, y_val, wavelengths):
    from spectral_predict.simca import MultiClassClassModel, multiclass_simca_metrics
    # Rebuild config from the row (reuse the same builder the GUI double-click uses).
    cfg = _multiclass_row_to_config(row)  # extract helper shared with GUI path
    model = MultiClassClassModel(**cfg["model_kwargs"])
    model.fit(X_train_pp, y_train)          # X_train_pp: preprocess per cfg, subset to wavelengths
    decision = model.decision_matrix(X_val_pp)
    return multiclass_simca_metrics(decision, y_val, model.classes_)
```

Populate the `val_*` columns from the returned metrics dict. (Extract `_multiclass_row_to_config` from the existing `_run_selected_multiclass_result` GUI code so both share one builder — DRY.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py::test_multiclass_holdout_metrics_populate -v`
Expected: PASS.

- [ ] **Step 6: Regression**

Run: `.venv312/Scripts/python -m pytest tests/test_multiclass_search.py tests/test_simca.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/spectral_predict/search.py tests/test_multiclass_search.py
git commit -m "feat(T31): holdout decision-matrix metrics for multiclass validation set"
```

---

## Phase 4 — GUI: sweep state vars (spec B/C)

### Task 6: add multi-class sweep state variables + parser reuse

**Files:**
- Modify: `spectral_predict_gui_optimized.py:3124-3138` (state var block)
- Test: `tests/gui/test_multiclass_gui_parity.py` (new)

**Interfaces:**
- Produces: new tk vars mirroring one-class SIMCA: `mc_alpha_001`, `mc_alpha_005` (BooleanVar), `mc_alpha_custom` (StringVar); `mc_ncomp_3`, `mc_ncomp_5`, `mc_ncomp_7`, `mc_ncomp_095`, `mc_ncomp_099` (BooleanVar), `mc_ncomp_custom` (StringVar); `mc_ncomp_per_class_cv` (BooleanVar, the unique toggle). A collector `_collect_mc_alpha_list()` / `_collect_mc_ncomp_list()` returning lists, reusing `_parse_oc_n_components_list` and `_parse_oc_float_list`.

- [ ] **Step 1: Write the failing test**

```python
# tests/gui/test_multiclass_gui_parity.py
import pytest
tk = pytest.importorskip("tkinter")
from spectral_predict_gui_optimized import SpectralPredictApp

@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    a = SpectralPredictApp(root)
    yield a
    root.destroy()

def test_mc_sweep_collectors_return_defaults(app):
    app.mc_ncomp_099.set(True)
    assert app._collect_mc_ncomp_list() == [0.99]
    app.mc_alpha_005.set(True)
    assert app._collect_mc_alpha_list() == [0.05]

def test_mc_ncomp_collector_mixes_int_and_fraction(app):
    for v in (app.mc_ncomp_099, app.mc_ncomp_005 if hasattr(app, "mc_ncomp_005") else app.mc_ncomp_099):
        pass
    app.mc_ncomp_5.set(True); app.mc_ncomp_095.set(True)
    got = sorted(app._collect_mc_ncomp_list(), key=lambda x: (isinstance(x, float), x))
    assert 5 in got and 0.95 in got
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py -k sweep_collectors -v`
Expected: FAIL — vars/collectors undefined.

- [ ] **Step 3: Add the state vars**

Replace the single-value multi-class vars (keep `mc_min_class_samples`, `mc_engine_vars`, `mc_varsel_vars`; remove `mc_alpha`, `mc_n_components`, `mc_varsel_n_select` after Phase 7 rewires their readers) by adding:

```python
        # Multi-class SIMCA sweep vars (mirror one-class _collect_simca_overrides)
        self.mc_alpha_001 = tk.BooleanVar(value=False)
        self.mc_alpha_005 = tk.BooleanVar(value=True)
        self.mc_alpha_custom = tk.StringVar(value="")
        self.mc_ncomp_3 = tk.BooleanVar(value=False)
        self.mc_ncomp_5 = tk.BooleanVar(value=False)
        self.mc_ncomp_7 = tk.BooleanVar(value=False)
        self.mc_ncomp_095 = tk.BooleanVar(value=False)
        self.mc_ncomp_099 = tk.BooleanVar(value=True)   # novelty-oriented default
        self.mc_ncomp_custom = tk.StringVar(value="")
        self.mc_ncomp_per_class_cv = tk.BooleanVar(value=False)  # unique toggle
```

- [ ] **Step 4: Add the collectors**

```python
    def _collect_mc_alpha_list(self):
        vals = []
        if self.mc_alpha_001.get(): vals.append(0.01)
        if self.mc_alpha_005.get(): vals.append(0.05)
        custom = self.mc_alpha_custom.get().strip()
        if custom:
            parsed, _errs = self._parse_oc_float_list(custom, 0.0, 1.0)
            for v in parsed:
                if v not in vals: vals.append(v)
        return vals or [0.05]

    def _collect_mc_ncomp_list(self):
        vals = []
        for flag, v in ((self.mc_ncomp_3, 3), (self.mc_ncomp_5, 5),
                        (self.mc_ncomp_7, 7), (self.mc_ncomp_095, 0.95),
                        (self.mc_ncomp_099, 0.99)):
            if flag.get(): vals.append(v)
        custom = self.mc_ncomp_custom.get().strip()
        if custom:
            parsed, _errs = self._parse_oc_n_components_list(custom)
            for v in parsed:
                if v not in vals: vals.append(v)
        if self.mc_ncomp_per_class_cv.get():
            vals.append("per_class_cv")
        return vals or [0.99]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py -k sweep_collectors -v`
Expected: PASS (or SKIP if no display — then verify via `py_compile`).

- [ ] **Step 6: Compile check + commit**

```bash
.venv312/Scripts/python -m py_compile spectral_predict_gui_optimized.py
git add spectral_predict_gui_optimized.py tests/gui/test_multiclass_gui_parity.py
git commit -m "feat(T31): multi-class alpha/n_components sweep state vars + collectors"
```

---

## Phase 5 — GUI: relocate config panels to 4A/4B (spec A/D/G)

### Task 7: build the multi-class Model Config card in tab 4A; remove the import-page panel

**Files:**
- Modify: `spectral_predict_gui_optimized.py` — remove `mc_hyperparams_frame` construction at `6545-6601`; add a `mc_model_config_frame` built alongside the one-class SIMCA override card in the Model Config subtab (near where `_collect_simca_overrides`'s widgets live / the models card ~12760-12800); update `_on_task_type_changed` (`17010-17033`) to show/hide the new card instead of the import panel.
- Test: `tests/gui/test_multiclass_gui_parity.py`

**Interfaces:**
- Consumes: sweep vars from Task 6.
- Produces: `self.mc_model_config_frame` (holds alpha checkboxes, n_components checkboxes + custom + per_class_cv toggle, min-class-n spinbox) living in the 4A subtab; hidden unless task type is `multiclass_simca`. Import page no longer contains any multi-class widgets.

- [ ] **Step 1: Write the failing test**

```python
def test_import_page_has_no_mc_panel(app):
    # The old import-page frame must be gone
    assert not hasattr(app, "mc_hyperparams_frame")

def test_mc_model_config_card_exists(app):
    assert hasattr(app, "mc_model_config_frame")

def test_task_type_toggles_mc_card(app):
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()
    assert app.mc_model_config_frame.winfo_manager() != ""   # mapped
    app.task_type.set("regression")
    app._on_task_type_changed()
    assert app.mc_model_config_frame.winfo_manager() == ""    # unmapped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py -k "import_page or model_config_card or toggles_mc" -v`
Expected: FAIL — `mc_hyperparams_frame` still exists; `mc_model_config_frame` undefined.

- [ ] **Step 3: Remove the import-page panel**

Delete the block at `6545-6601` (`self.mc_hyperparams_frame = ...` through its `grid_remove()` and `cfg_row += 1`). Leave the task-type radio at `6490` intact.

- [ ] **Step 4: Build the 4A card**

In the Model Config subtab builder (where the one-class model config container is created), add a sibling frame `self.mc_model_config_frame` containing:
- alpha row: `Checkbutton`s bound to `mc_alpha_001` (`0.01`), `mc_alpha_005` (`0.05 ⭐`) + `Entry` bound to `mc_alpha_custom` labelled `Custom:`.
- n_components row: `Checkbutton`s bound to `mc_ncomp_3/5/7/095/099` (label `0.99 ⭐`) + `Entry` bound to `mc_ncomp_custom` + a separate `Checkbutton` bound to `mc_ncomp_per_class_cv` labelled `per_class_cv (auto; discrimination-oriented)`.
- min-class-n: `Spinbox` bound to `mc_min_class_samples` (`from_=3, to=1000`).
- Reuse the existing `CreateToolTip` copy for n_components from the old panel (`6563-6570`).

Build it hidden (`grid_remove()`), mirroring `oc_model_config_container`.

- [ ] **Step 5: Update `_on_task_type_changed`**

In the `elif task_type == "multiclass_simca":` branch (`17010`), replace `self.mc_hyperparams_frame.grid()` with `self.mc_model_config_frame.grid()`; in the other branches' hide logic (`17019`, `17036`) replace `mc_hyperparams_frame` with `mc_model_config_frame`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py -k "import_page or model_config_card or toggles_mc" -v`
Expected: PASS.

- [ ] **Step 7: Compile + commit**

```bash
.venv312/Scripts/python -m py_compile spectral_predict_gui_optimized.py
git add spectral_predict_gui_optimized.py tests/gui/test_multiclass_gui_parity.py
git commit -m "feat(T31): move multi-class hyperparameters from Import page to Model Config subtab"
```

---

### Task 8: multi-class variable selection in tab 4B — reuse Top-N + grouped method list (spec D/E)

**Files:**
- Modify: `spectral_predict_gui_optimized.py` — `_on_task_type_changed` multiclass branch (`17028-17029`) to STOP hiding `varsel_card_outer`; add a multi-class method-group frame in the 4B card; keep the shared "Top-N Variable Counts" checkboxes visible.
- Test: `tests/gui/test_multiclass_gui_parity.py`

**Interfaces:**
- Consumes: existing `var_10..var_1000` Top-N vars; `mc_varsel_vars` dict (extend to full method set).
- Produces: in multiclass mode the 4B card is visible, shows two labeled groups (SIMCA-native / discrimination-based), and the shared Top-N row drives the size sweep. Method vars: extend `mc_varsel_vars` to include `importance`, `cars`, `cars_tree`, `spa`, `uve`, `ipls`, `ipls_forward`, `ipls_backward`, `mc_sipls`, `mwpls`, `ga`, `vcpa`, `uve_spa`, `uve_cars`, `uve_cars_tree`, `uve_cars_spa` alongside the existing Wold entries.

- [ ] **Step 1: Write the failing test**

```python
def test_mc_varsel_card_visible_and_grouped(app):
    app.task_type.set("multiclass_simca"); app._on_task_type_changed()
    assert app.varsel_card_outer.winfo_manager() != ""       # 4B card shown
    assert hasattr(app, "mc_varsel_group_frame")
    # full method set present
    for k in ("importance", "cars", "spa", "uve", "ga", "wold_modeling"):
        assert k in app.mc_varsel_vars

def test_mc_reuses_topn_counts(app):
    # The shared Top-N vars exist and are the size-sweep source for multiclass
    for v in ("var_10", "var_50", "var_100"):
        assert hasattr(app, v)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py -k "varsel_card_visible or reuses_topn" -v`
Expected: FAIL — card hidden in mc mode; `mc_varsel_group_frame` undefined; method set incomplete.

- [ ] **Step 3: Extend `mc_varsel_vars`**

Where `mc_varsel_vars` is declared, add BooleanVars for the full set (default all False except keep existing Wold defaults). Group metadata:

```python
        self.mc_varsel_groups = {
            "SIMCA-native (novelty-safe)": [
                ("wold_modeling", "Wold modeling"),
                ("wold_discriminating", "Wold discriminating"),
                ("wold_balanced", "Wold balanced"),
            ],
            "Discrimination-based (confirm novelty on a true external class)": [
                ("importance", "Importance"), ("cars", "CARS"),
                ("cars_tree", "CARS-Tree"), ("spa", "SPA"), ("uve", "UVE"),
                ("ipls", "iPLS"), ("ipls_forward", "Forward iPLS"),
                ("ipls_backward", "Backward iPLS"), ("mc_sipls", "MC-siPLS"),
                ("mwpls", "MWPLS"), ("ga", "GA"), ("vcpa", "VCPA-IRIV"),
                ("uve_spa", "UVE-SPA"), ("uve_cars", "UVE-CARS"),
                ("uve_cars_tree", "UVE-CARS-Tree"), ("uve_cars_spa", "UVE-CARS-SPA"),
            ],
        }
        self.mc_varsel_vars = {
            key: tk.BooleanVar(value=(key == "wold_modeling"))
            for group in self.mc_varsel_groups.values() for key, _lbl in group
        }
```

- [ ] **Step 4: Build the grouped frame in the 4B card**

In `_create_tab4b_variable_selection`, after the shared varsel frame, add `self.mc_varsel_group_frame` (built hidden) that renders each group as a labeled section of `Checkbutton`s bound to `mc_varsel_vars`, with the group-title label doubling as the honesty caption for the discrimination-based group.

- [ ] **Step 5: Show it in multiclass mode**

In `_on_task_type_changed`, multiclass branch: remove the `varsel_card_outer.grid_remove()` line; instead `self.varsel_card_outer.grid()`, hide the *standard* method frame (`varsel_frame`) and Top-N stays visible, and `self.mc_varsel_group_frame.grid()`. In the non-multiclass branches, `self.mc_varsel_group_frame.grid_remove()` and restore the standard `varsel_frame`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py -k "varsel_card_visible or reuses_topn" -v`
Expected: PASS.

- [ ] **Step 7: Compile + commit**

```bash
.venv312/Scripts/python -m py_compile spectral_predict_gui_optimized.py
git add spectral_predict_gui_optimized.py tests/gui/test_multiclass_gui_parity.py
git commit -m "feat(T31): multi-class variable selection in 4B subtab (grouped set + shared Top-N sweep)"
```

---

## Phase 6 — GUI: run-handler wiring, validation, no auto-popup (spec F/I/J)

### Task 9: feed swept lists + Top-N sizes + varsel methods into the search call

**Files:**
- Modify: `spectral_predict_gui_optimized.py:28479-28538` (the `run_multiclass_simca_search(...)` call + `_mc_run_config`)
- Test: manual/e2e (headless smoke below)

**Interfaces:**
- Consumes: `_collect_mc_alpha_list()`, `_collect_mc_ncomp_list()`, the checked `var_10..var_1000` Top-N sizes, checked `mc_varsel_vars` keys.
- Produces: the search call passes `alpha=<list>`, `n_components=<list>`, `variable_selection_n_select=<list of checked Top-N>`, `varsel_paths=<checked mc methods>`.

- [ ] **Step 1: Build the collectors for sizes + methods**

Add near the run handler:

```python
        def _collect_mc_sizes(self):
            sizes = []
            for flag, n in ((self.var_10, 10), (self.var_20, 20), (self.var_50, 50),
                            (self.var_100, 100), (self.var_250, 250),
                            (self.var_500, 500), (self.var_1000, 1000)):
                if flag.get(): sizes.append(n)
            return sizes or [100]

        def _collect_mc_varsel_paths(self):
            paths = [k for k, v in self.mc_varsel_vars.items() if v.get()]
            return paths or ["none"]
```

- [ ] **Step 2: Rewrite the search call**

Replace the scalar args at `28485-28488` with:

```python
                        alpha=self._collect_mc_alpha_list(),
                        n_components=self._collect_mc_ncomp_list(),
                        varsel_paths=self._collect_mc_varsel_paths(),
                        variable_selection_n_select=self._collect_mc_sizes(),
                        min_class_samples=self.mc_min_class_samples.get(),
```

Update `_mc_run_config` (`28528-28538`) to store the lists (used by the double-click rebuild + validation): keep per-row rebuild reading from the row itself, but store `"alpha": self._collect_mc_alpha_list()` etc. for reference. Remove the now-stale `mc_ncomp` scalar references above the call (the log line at `28475` should print the lists).

- [ ] **Step 3: Headless smoke**

Run a scripted search with multiple alphas/sizes to confirm the leaderboard multiplies:

```bash
.venv312/Scripts/python - <<'PY'
import numpy as np
from spectral_predict.search import run_multiclass_simca_search
X = np.random.RandomState(0).rand(60,40); y = np.array(["A","B","C"]*20)
df = run_multiclass_simca_search(X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
    alpha=[0.01,0.05], n_components=[0.95,0.99], varsel_paths=["none","importance"],
    variable_selection_n_select=[10,50])
print(len(df), sorted(df["Alpha"].unique()), sorted(df["NComponents"].astype(str).unique()))
PY
```
Expected: row count = 2 alphas × 2 ncomps × 2 paths × 2 sizes (minus any skipped), Alpha `{0.01,0.05}`, NComponents `{0.95,0.99}`.

- [ ] **Step 4: Commit**

```bash
git add spectral_predict_gui_optimized.py
git commit -m "feat(T31): feed swept alpha/n_components/sizes + full varsel methods into multiclass run"
```

---

### Task 10: remove the run-completion auto-popup (spec J)

**Files:**
- Modify: `spectral_predict_gui_optimized.py:28560-28567`
- Test: `tests/gui/test_multiclass_gui_parity.py`

**Interfaces:**
- Produces: a completed multi-class run populates the leaderboard only; no decision-view window opens automatically. The silent CSV export (`28551-28556`) stays.

- [ ] **Step 1: Write the failing test (spy on the opener)**

```python
def test_no_auto_decision_popup(app, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(app, "_show_multiclass_decision_view", lambda *a, **k: called.__setitem__("n", called["n"]+1))
    # Simulate the tail of a successful run with a decision view present
    app._mc_decision_view = {"decision_matrix": [], "classes": []}
    app._finalize_multiclass_run_ui()   # small extracted method wrapping 28539-28575
    assert called["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py::test_no_auto_decision_popup -v`
Expected: FAIL — opener invoked once (or `_finalize_multiclass_run_ui` undefined).

- [ ] **Step 3: Remove the auto-open**

Delete the `if self._mc_decision_view and not ...: self.root.after(0, lambda v=...: self._show_multiclass_decision_view(v))` block at `28560-28563`. Keep the `elif ...reason` and `else` log-warning branches (they don't open a window). Extract the UI-finalize tail into `_finalize_multiclass_run_ui()` for testability.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py::test_no_auto_decision_popup -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spectral_predict_gui_optimized.py tests/gui/test_multiclass_gui_parity.py
git commit -m "feat(T31): stop auto-opening multi-class decision view on run completion"
```

---

### Task 11: wire multi-class into the Validation tab (spec I)

**Files:**
- Modify: `spectral_predict_gui_optimized.py` — the multi-class run handler (`28466-28471` note + call `compute_validation_metrics_for_top_models`); `_on_validation_algorithm_changed` / algorithm radios (`14514-14520`, `20500-20507`) to disable SPXY for multiclass.
- Test: `tests/gui/test_multiclass_gui_parity.py`

**Interfaces:**
- Consumes: `compute_validation_metrics_for_top_models(..., task_type="multiclass_simca")` (Task 5); `self.validation_X/_y/_indices`.
- Produces: when a validation set is enabled and task type is multiclass, the run computes `val_*` decision-matrix metrics on the holdout and they appear in the results table; SPXY is unavailable for multiclass.

- [ ] **Step 1: Write the failing test**

```python
def test_spxy_disabled_for_multiclass(app):
    app.task_type.set("multiclass_simca"); app._on_task_type_changed()
    assert app._validation_algo_allowed("SPXY") is False
    assert app._validation_algo_allowed("Kennard-Stone") is True
    assert app._validation_algo_allowed("Random") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py::test_spxy_disabled_for_multiclass -v`
Expected: FAIL — `_validation_algo_allowed` undefined.

- [ ] **Step 3: Add the guard + disable SPXY**

Add:

```python
    def _validation_algo_allowed(self, algo):
        if self.task_type.get() == "multiclass_simca" and algo == "SPXY":
            return False   # d_y is undefined for a categorical class label
        return True
```

In `_on_task_type_changed` multiclass branch, if `self.validation_algorithm.get() == "SPXY"`, reset it to `"Kennard-Stone"` and disable the SPXY radio; re-enable it in the non-multiclass branches.

- [ ] **Step 4: Replace the "does not apply" note with real wiring**

In the run handler, replace the `28466-28471` note block: when `self.validation_enabled.get() and self.validation_indices`, after the search returns `results_df`, call:

```python
                    if self.validation_enabled.get() and self.validation_indices:
                        from spectral_predict.search import compute_validation_metrics_for_top_models
                        results_df = compute_validation_metrics_for_top_models(
                            results_df,
                            X_train=X_filtered, y_train=y_filtered,
                            X_val=self.validation_X.values, y_val=self.validation_y.values,
                            task_type="multiclass_simca",
                            wavelengths=wavelengths, top_n=100,
                        )
                        self._log_progress(
                            "  Holdout validation: known-class val_* metrics added. "
                            "NOTE: a same-class holdout does not test novelty — "
                            "confirm 'none of the above' on a held-out class or true external contaminant."
                        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py::test_spxy_disabled_for_multiclass -v`
Expected: PASS.

- [ ] **Step 6: Compile + commit**

```bash
.venv312/Scripts/python -m py_compile spectral_predict_gui_optimized.py
git add spectral_predict_gui_optimized.py tests/gui/test_multiclass_gui_parity.py
git commit -m "feat(T31): wire multi-class into Validation tab (KS/Random/Stratified; SPXY disabled; known-class caveat)"
```

---

## Phase 7 — GUI: reporting polish (spec H)

### Task 12: rename swept columns for display + add tooltips + decision-view config header

**Files:**
- Modify: `spectral_predict_gui_optimized.py` — results-table column rename map (where `_populate_results_table` maps backend→display names); `TOOLTIP_CONTENT` column-help dict (`~1485-1544`); `_show_multiclass_decision_view` (`29879+`) to prepend a config header.
- Test: `tests/gui/test_multiclass_gui_parity.py`

**Interfaces:**
- Consumes: `engine_family`, `varsel_path`, `NComponents`, `Alpha` columns.
- Produces: leaderboard headers show `Engine`, `VarSelMethod`, `NComponents`, `Alpha`; each has a hover tooltip; the decision-view window shows a one-line config header (engine, alpha, n_components, varsel method + size).

- [ ] **Step 1: Write the failing test**

```python
def test_decision_view_header_states_config(app):
    view = {"decision_matrix": [], "classes": ["A"], "config_summary": None}
    header = app._multiclass_decision_header({
        "engine_family": "pca-simca", "Alpha": 0.05,
        "NComponents": 0.99, "varsel_path": "importance", "n_vars": 50,
    })
    assert "pca-simca" in header and "0.05" in header and "0.99" in header
    assert "importance" in header and "50" in header
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py::test_decision_view_header_states_config -v`
Expected: FAIL — `_multiclass_decision_header` undefined.

- [ ] **Step 3: Add the header builder + tooltips + rename map**

```python
    def _multiclass_decision_header(self, row):
        return (f"Engine: {row.get('engine_family')}  |  alpha: {row.get('Alpha')}  |  "
                f"n_components: {row.get('NComponents')}  |  "
                f"varsel: {row.get('varsel_path')} (top-{row.get('n_vars')})")
```

Call it in `_show_multiclass_decision_view` to render a label at the top of the window. Add `TOOLTIP_CONTENT` entries for `NComponents`, `VarSelMethod`, `Engine` (copy: `NComponents` → "Per-class PCA size for this row. Float in (0,1)=variance fraction, int=fixed count, per_class_cv=auto."). In the results-table display-name map, add `engine_family→"Engine"`, `varsel_path→"VarSelMethod"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312/Scripts/python -m pytest tests/gui/test_multiclass_gui_parity.py::test_decision_view_header_states_config -v`
Expected: PASS.

- [ ] **Step 5: Compile + commit**

```bash
.venv312/Scripts/python -m py_compile spectral_predict_gui_optimized.py
git add spectral_predict_gui_optimized.py tests/gui/test_multiclass_gui_parity.py
git commit -m "feat(T31): self-describing multi-class reporting (display names, tooltips, decision-view header)"
```

---

## Phase 8 — Verification

### Task 13: full regression + real-data e2e + docs

**Files:**
- Modify: `docs/PROJECT_STATUS.md`, `docs/SESSION_LOG.md`
- Test: existing suites + live GUI

- [ ] **Step 1: Backend + GUI regression**

Run: `.venv312/Scripts/python -m pytest tests/test_simca.py tests/test_multiclass_search.py tests/test_model_io.py tests/test_contamination.py tests/gui/test_multiclass_gui_parity.py -q`
Expected: PASS; zero regression vs pre-change on `test_simca`/`contamination`/`model_io`.

- [ ] **Step 2: Single-Y byte-identical smoke**

Run a regression (PLS) and classification (PLS-DA) search through `run_search`, and a one-class search through `run_one_class_search`, before/after this branch; diff the ranked leaderboards. Expected: identical.

- [ ] **Step 3: Real-data e2e (live GUI)**

Launch the GUI (`.venv312/Scripts/python spectral_predict_gui_optimized.py`), load `Contaminated Samples Raw_ORAU Added.xlsx` (`Site` target), select Multi-Class, in 4A check alphas `0.01,0.05` and n_components `0.95,0.99`, in 4B check `Wold modeling` + `importance` + Top-N `50,100`, enable a Kennard-Stone validation set, Run. Confirm: no popup on completion; leaderboard multiplies and shows Alpha/NComponents/Engine/VarSelMethod/n_vars/val_* columns; double-click a row opens the decision matrix with a config header; select a non-top row → Save Model → reload → predict reproduces its decision matrix.

- [ ] **Step 4: Update docs + commit**

Update `docs/PROJECT_STATUS.md` (ACTIVE DIRECTION → parity work done) and append a `docs/SESSION_LOG.md` entry (design + gotchas: SPXY-undefined-on-categorical-y, holdout-validates-known-class-not-novelty, mask-hook reuse).

```bash
git add docs/PROJECT_STATUS.md docs/SESSION_LOG.md
git commit -m "docs(T31): record multi-class UX parity work (config relocation, sweeps, validation, reporting)"
```

- [ ] **Step 5: Merge gate (do NOT auto-merge)**

Per project protocol: whole-diff cross-family review + pr-review-toolkit + local diff-failure-set vs `origin/main`; user greenlight only.

---

## Self-Review Notes

- **Spec coverage:** A→Task 7; B/C→Task 6; D→Task 8; E→Task 4 (backend) + Task 8 (GUI); F→Task 2 + Task 9; G→Task 7 (min-class-n spinbox); H→Task 1/3 (backend) + Task 12 (GUI); I→Task 5 (backend) + Task 11 (GUI); J→Task 10. All ten sections mapped.
- **Shared builder:** Task 5 and the GUI double-click path both need `_multiclass_row_to_config`; Task 5 Step 4 extracts it once (DRY) — the GUI's `_run_selected_multiclass_result` must be refactored to call the same helper when Task 5 lands (note added there).
- **Removal ordering:** `mc_alpha`/`mc_n_components`/`mc_varsel_n_select` scalar vars are read by the run handler until Task 9; delete them only after Task 9 rewires the readers (Task 6 adds the new vars but leaves the old ones until then).
