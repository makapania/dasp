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

Two smaller ones: the **readers enforce a minimum of 100 wavelengths** (every reader
in `io.py`, not just the CSV one), and `save_model` requires the metadata keys
`model_name`, `task_type`, `wavelengths`, and `n_vars`.

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
> set (see §3).

To fit preprocessing per fold, put it in the same `Pipeline` as the model so
`fit` only ever sees training rows:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline

def make_pipeline():
    steps = build_preprocessing_pipeline("snv_deriv", deriv=1, window=17, polyorder=2)
    return Pipeline(steps + [("model", PLSRegression(n_components=5))])

# inside each CV fold, on the RAW matrix (not the pre-transformed X_pp):
X_raw = X_aligned.to_numpy(dtype=float)
pipe = make_pipeline().fit(X_raw[train_idx], y_vals[train_idx])
pred = pipe.predict(X_raw[test_idx]).ravel()
```

The §4 example below uses a pre-transformed `X_pp` for brevity; prefer this form
when you are reporting a performance estimate.

---

## 3. Select variables

There are **two families with different signatures and different return types**.
Mixing them up is easy — check which family you are calling.

### 3a. Score-array selectors

Signature `(X, y, ...)`; return an importance array of shape `(n_features,)`, higher
= more important. You choose the cutoff. CARS returns a sparse array where the
non-zero entries are its selection.

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

Also in this family: `uve_spa_selection`, `uve_cars_selection`,
`fipls_spa_selection`, and `fipls_cars_selection`.

`get_uve_threshold` is **not** in this family despite the name — it returns a
**3-tuple** `(importances, threshold, selected_mask)`, not a bare score array. It
does the cutoff for you, so unpack it rather than calling `np.argsort` on the
result:

```python
from spectral_predict.variable_selection import get_uve_threshold

importances, threshold, mask = get_uve_threshold(
    X_pp, y.to_numpy(dtype=float), cutoff_multiplier=1.0, cv_folds=5,
)
X_sel = X_pp[:, mask]          # mask is already boolean, shape (n_features,)
print(f"kept {mask.sum()} of {mask.size} variables (threshold {threshold:.3f})")
```

### 3b. Interval-subset selectors

`ipls_forward`, `ipls_backward`, `mc_sipls`, and `mwpls` take **`wavelengths` as a
required third positional argument** and return a **list of candidate-subset dicts**
— *not* an importance array. You pick a subset, typically the lowest `rmsecv`.

Keys present on **every** entry: `indices`, `tag`, `interval_ids`, `n_intervals`,
`rmsecv`, `r2`. Conditional keys — do not rely on them: `rank` appears on
`mc_sipls`/`mwpls` entries and on `ipls_forward`'s single-interval entries but
**not** its combined-interval entries; `is_optimal` appears only on
`ipls_backward`. Selecting on `rmsecv` (as below) is always safe.

```python
import numpy as np
from spectral_predict.variable_selection import ipls_forward

wavelengths = np.asarray([float(c) for c in X_aligned.columns])
subsets = ipls_forward(X_pp, y.to_numpy(dtype=float), wavelengths,
                       n_intervals=20, cv_folds=5)

best = min(subsets, key=lambda s: s["rmsecv"])
X_sel = X_pp[:, np.sort(best["indices"])]
```

If you would rather not handle both shapes yourself, `multiclass_varsel_mask` (§5)
normalises either family into a single boolean mask.

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
than silently returning nothing. `n_select` is optional: omitted, `None`, or `NaN`
all fall back to `min(100, n_features)`.

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
`ocsvm`, `isolation-forest`, `lof`, `elliptic-envelope`. Note `min_class_samples`
defaults to **10** — the `5` above is only to keep this example small, not a floor.

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

This example is **self-contained** — it fits its own regression model rather than
reusing the class model from §6. The `metadata` you write must describe the object
you actually pass to `save_model`; saving a `MultiClassClassModel` under
`"task_type": "regression"` will load fine and then predict down the wrong dispatch
path.

```python
from sklearn.cross_decomposition import PLSRegression
from spectral_predict.model_io import save_model, load_model, predict_with_model

X_raw = X_aligned.to_numpy(dtype=float)
model = PLSRegression(n_components=5).fit(X_raw, y.to_numpy(dtype=float))

metadata = {
    "model_name": "PLS",
    "task_type": "regression",
    "wavelengths": [float(c) for c in X_aligned.columns],
    "n_vars": X_aligned.shape[1],          # required
    "target_name": "%Collagen",
    "preprocess": "raw",
}
# preprocessor=None matches "preprocess": "raw" above. If you fit a preprocessing
# Pipeline (§2), pass that fitted pipeline here instead of None and name it in
# "preprocess" — predict_with_model re-applies whatever you stored.
save_model(model, None, metadata, "my_model.dasp")

loaded = load_model("my_model.dasp")
X_new = X_aligned                          # stand-in: same wavelength columns
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
| `variable_selection` | score-array: `cars_selection`, `ipls_selection`, `spa_selection`, `uve_selection`; interval-subset: `ipls_forward`, `ipls_backward`, `mc_sipls`, `mwpls` |
| `simca` | `MultiClassClassModel` |
| `contamination` | `PCASIMCA` |
| `models` | `PLSTransformer` |
| `model_io` | `save_model`, `load_model`, `predict_with_model` |
| `search` | `run_search`, `run_one_class_search`, `run_multiclass_simca_search`, `multiclass_varsel_mask`, `build_multiclass_decision_view`, `compute_validation_metrics_for_top_models`, `MulticlassVarselUnsupported` |

**This table is the contract.** Only `search` declares an `__all__` enforcing it (it
matches the row above exactly); the other modules do not, so the absence of an
`__all__` elsewhere does not mean everything in them is public.

Import from the module, not the package root — `import spectral_predict` is kept
free of matplotlib and tkinter so it stays usable headlessly, and top-level
re-exports would undo that.

**If you need something not on this list**, say so rather than reaching for an
underscore-prefixed name. A private function can be renamed without warning, which
silently breaks callers outside this repo — the reason `multiclass_varsel_mask`
became public.
