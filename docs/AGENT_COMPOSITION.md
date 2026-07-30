# Composing analyses from DASP primitives

**Audience:** anyone driving DASP from Python instead of the GUI — including AI
agents. Everything here is runnable from a clean clone against the repo's own
`example/` data.

---

## DASP is not an orchestrator

DASP supplies **chemometric building blocks**. Your script owns the research design:

| DASP provides | Your script owns |
|---|---|
| Readers for 10+ spectral formats | Which samples are in/out |
| Preprocessing (SNV, derivatives, baselines, MSC/OSC) | The sampling design and grouping |
| Variable selection (CARS, SPA, UVE, iPLS, Wold, …) | The cross-validation splitter |
| Models (PLS, RF, XGBoost, SVM, …) and class models (DD-SIMCA) | The ranking objective |
| Scoring metrics and model persistence | Reporting and figures |

There is deliberately **no command-line runner**. A command line can only encode a
fixed analysis shape, and there is no standard spectral analysis — every dataset
needs its own preprocessing, model set, and variable selection. The
`spectral-predict` console script was retired for exactly this reason.

If the built-in grid *is* what you want, call `run_search` (§7). If your design is
non-standard — grouped replicates, a locked holdout, external validation
collections, a custom ranking rule — compose the primitives and write your own loop.
That is the intended and supported path, not a workaround.

---

## Read this first: five things that will cost you a turn

1. **Readers return `(DataFrame, dict)` tuples, not DataFrames.**
   `X, meta = read_spectra(...)`. Forgetting the unpack gives
   `AttributeError: 'tuple' object has no attribute 'shape'`.

2. **`run_search` also returns a 2-tuple** — `(df_ranked, label_encoder)`.
   Unpack it: `df_ranked, label_encoder = run_search(...)`.

3. **Variable selectors return importance *score arrays*, not boolean masks.**
   Shape `(n_features,)`. You threshold or Top-N them yourself — or use
   `multiclass_varsel_mask` (§5), which does the normalisation for you.

4. **`run_search`'s `preprocessing_methods` is a dict of bools, not a list of
   strings.** `{"raw": True, "snv": True}`, not `["raw", "snv"]`. Passing a list
   raises `AttributeError: 'list' object has no attribute 'get'`.

5. **The score-penalty keyword is `variable_penalty`, not `lambda_penalty`.**

Two smaller ones: `read_csv_spectra` enforces a **minimum of 100 wavelengths**
(`io.py`), and `save_model` requires the metadata keys `model_name`, `task_type`,
`wavelengths`, and `n_vars`.

---

## 1. Load and align data

`read_spectra` auto-detects format from a file or a directory. Format-specific
readers (`read_asd_dir`, `read_csv_spectra`, …) are available when you want to be
explicit.

```python
from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy

X, meta = read_asd_dir("example/")                       # (DataFrame, dict)
ref = read_reference_csv("example/BoneCollagen.csv", "File Number")
X_aligned, y = align_xy(X, ref, "File Number", "%Collagen")

print(X_aligned.shape, y.shape, meta["data_type"])
# (49, 2151) (49,) reflectance
```

`align_xy` does flexible ID matching (extensions, spaces, case) and drops rows with
a missing target. Columns of `X` are float wavelengths; the index is the sample ID.

---

## 2. Preprocess

`build_preprocessing_pipeline` returns a **list of sklearn steps**, so you drop it
straight into a `Pipeline` and keep full control of fitting.

```python
from sklearn.pipeline import Pipeline
from spectral_predict.preprocess import build_preprocessing_pipeline

steps = build_preprocessing_pipeline("snv_deriv", deriv=1, window=17, polyorder=2)
X_pp = Pipeline(steps).fit_transform(X_aligned.to_numpy(dtype=float))
```

Valid `preprocess_name` values: `"raw"`, `"snv"`, `"deriv"`, `"snv_deriv"`,
`"deriv_snv"`. Order matters — `snv_deriv` is SNV then derivative; `deriv_snv` is the
reverse. Baseline correction, smoothing, and autoscaling are keyword arguments on
the same call.

> **Fit preprocessing inside your CV folds** if you want a leakage-free performance
> estimate. Variable *selection* is conventionally run once on the full calibration
> set (see §5).

---

## 3. Select variables

All selectors take `(X, y, ...)` and return an importance array of shape
`(n_features,)`. Higher is more important; CARS returns a sparse array where
non-zero entries are the selected variables.

```python
import numpy as np
from spectral_predict.variable_selection import (
    cars_selection, spa_selection, uve_selection, ipls_selection,
)

importances = cars_selection(X_pp, y.to_numpy(dtype=float),
                             n_iterations=50, pls_components=5, cv_folds=5)

top_n = 100
keep = np.argsort(importances)[::-1][:top_n]      # you choose the cutoff
X_sel = X_pp[:, np.sort(keep)]
```

Also available: `uve_spa_selection`, `uve_cars_selection`, `ipls_forward`,
`ipls_backward`, `fipls_spa_selection`, `fipls_cars_selection`, `mc_sipls`, `mwpls`,
`get_uve_threshold`.

**Why selection runs once on the full calibration set, not per fold:** stability of
the chosen wavelengths across resamples is itself the evidence that the selection
found real chemistry rather than fold-specific noise. Re-selecting inside each fold
makes "the model uses these wavelengths" scientifically uninterpretable. This is the
chemometrics convention (PLS_Toolbox GA, mdatools iPLS, CARS). The honest check is an
external/held-out set, not nested selection.

---

## 4. Your own cross-validation, including grouped CV

**There is no `groups` parameter on any search entry point.** `run_search`,
`run_one_class_search`, and `run_multiclass_simca_search` do not accept one, and
`cv_utils` raises `NotImplementedError` for `group_kfold` / `leave_one_group_out`.

If your samples contain replicates of the same specimen, random k-fold will leak
replicates across the split and inflate your metrics. Own the splitter:

```python
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

y_vals = y.to_numpy(dtype=float)
groups = ...          # one group id per row: specimen / object / site

rmses = []
for train_idx, test_idx in GroupKFold(n_splits=4).split(X_pp, y_vals, groups=groups):
    model = PLSRegression(n_components=5).fit(X_pp[train_idx], y_vals[train_idx])
    pred = model.predict(X_pp[test_idx]).ravel()
    rmses.append(np.sqrt(mean_squared_error(y_vals[test_idx], pred)))

print(f"grouped 4-fold RMSE = {np.mean(rmses):.3f}")
```

Use `StratifiedGroupKFold` for classification to keep class balance across folds.
For a locked holdout, split **by group** once, set the holdout aside, and never let
it influence preprocessing choice, selection, or ranking. Aggregating replicates
(e.g. one median spectrum per specimen) before fitting is often preferable to
carrying them as separate rows.

---

## 5. Multi-class variable-selection masks

`multiclass_varsel_mask` resolves a selection-method *name* into something you can
apply directly. It is the normalisation layer over §3, for genuine multi-class
labels.

```python
from spectral_predict.search import multiclass_varsel_mask

mask = multiclass_varsel_mask(X_mc, y_mc, wavelengths, "importance", n_select=15)
# -> ndarray of bool, shape (n_features,), exactly 15 True
```

Three possible returns:

| `method` | returns |
|---|---|
| `"none"` | `None` — no selection |
| a Wold method (`wold_*`) | the method string unchanged, for `MultiClassClassModel` to handle per-fold |
| any other supported method | a boolean mask, shape `(n_features,)` |

Unsupported methods raise `MulticlassVarselUnsupported` — it fails loudly rather
than silently returning nothing. If `n_select` is omitted it defaults to
`min(100, n_features)`.

---

## 6. Class models (DD-SIMCA and one-class)

`engine="pca-simca"` **is** DASP's DD-SIMCA implementation; there is no separate
"classical PCA-SIMCA" engine.

```python
from spectral_predict.simca import MultiClassClassModel

model = MultiClassClassModel(
    engine="pca-simca", alpha=0.05, n_components=3, min_class_samples=5,
)
model.fit(X_mc, y_mc)
predictions = model.predict(X_mc)      # ndarray, shape (n_samples,)
print(model.classes_)
```

`n_components` accepts an int, a per-class dict, or `"per_class_cv"`. Other engines:
`ocsvm`, `isolation-forest`, `lof`, `elliptic-envelope`.

For a single-class membership model:

```python
from spectral_predict.contamination import PCASIMCA

inliers = X_mc[y_mc == "clean"]
oc = PCASIMCA(n_components=3, alpha=0.05).fit(inliers)
labels = oc.predict(X_mc)              # +1 = in-class, -1 = outlier
```

A class model answers "is this sample consistent with class C?" — independently per
class. A sample can match several classes or none. Do **not** read a "no class"
outcome as an identification.

---

## 7. When the built-in grid is what you want

`run_search` sweeps preprocessing × models × hyperparameters and ranks the results.
Use it when the standard grid genuinely fits your question.

```python
from spectral_predict.search import run_search

df_ranked, label_encoder = run_search(       # NOTE: 2-tuple
    X_aligned, y, "regression",
    tier="quick",                            # quick | standard | comprehensive | experimental
    folds=3,
    preprocessing_methods={"raw": True, "snv": True},   # dict of bools, not a list
)
print(df_ranked.sort_values("Rank").head())
```

`models_to_test=["PLS", "XGBoost"]` overrides `tier`. Around 140 further keyword
arguments expose every hyperparameter grid, variable-selection option, and
optimization mode — see the `run_search` docstring.

Sibling entry points: `run_one_class_search` (contamination screening, returns a
DataFrame) and `run_multiclass_simca_search` (returns a DataFrame).

**Ranking caveat:** variable-subset rows can outrank full-spectrum rows simply by
having fewer variables. Filter by `SubsetTag` before comparing if that is not what
you want.

---

## 8. Save and reuse a model

```python
from spectral_predict.model_io import save_model, load_model, predict_with_model

metadata = {
    "model_name": "PLS",
    "task_type": "regression",
    "wavelengths": [float(c) for c in X_aligned.columns],
    "n_vars": X_aligned.shape[1],          # required
    "target_name": "%Collagen",
    "preprocess": "raw",
}
save_model(model, None, metadata, "my_model.dasp")

loaded = load_model("my_model.dasp")
predictions = predict_with_model(loaded, X_new)
```

`predict_with_model` validates that `X_new` carries the model's wavelengths and
re-applies the stored preprocessor, scaler, and PCA. For multi-class models it
returns a dict (`p_values`, `decision_matrix`, …) rather than an array.

---

## Declared stable surface

These are contract-tested in `tests/test_agent_composition_api.py`. Anything **not**
listed is an internal implementation detail that may change without notice.

| Module | Primitives |
|---|---|
| `io` | `read_spectra`, `read_asd_dir`, `read_csv_spectra`, `read_reference_csv`, `align_xy` |
| `preprocess` | `build_preprocessing_pipeline` |
| `unified_bayesian` | `apply_preprocessing`, `run_unified_bayesian` |
| `variable_selection` | `cars_selection`, `ipls_selection`, `spa_selection`, `uve_selection` |
| `simca` | `MultiClassClassModel` |
| `contamination` | `PCASIMCA` |
| `models` | `PLSTransformer` |
| `model_io` | `save_model`, `load_model`, `predict_with_model` |
| `search` | `run_search`, `run_one_class_search`, `run_multiclass_simca_search`, `multiclass_varsel_mask` |

Import from the module, not the package root — `import spectral_predict` is kept
free of matplotlib and tkinter so it stays usable headlessly, and top-level
re-exports would undo that.

**If you need something not on this list**, say so rather than reaching for an
underscore-prefixed name. A private function can be renamed without warning, which
silently breaks callers outside this repo — the reason `multiclass_varsel_mask`
became public.
