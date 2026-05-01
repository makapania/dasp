# T-33: Gaussian Process Regression — rough plan

**Status:** ROUGH PLAN — awaiting user prioritization. Not yet implemented.
**Filed:** 2026-04-30 by user request.
**Ticket allocation:** Next available T-N after T-32. Add to roadmap if/when prioritized.

## Motivation

Add Gaussian Process Regression (GP) as a model card alongside the existing
PLS / Ridge / Lasso / ElasticNet / RandomForest / LightGBM / XGBoost /
CatBoost / SVR / MLP / NeuralBoosted set. GP is non-parametric Bayesian and
gives well-calibrated predictive uncertainties — relevant for the bone-FTIR /
isotopes / paleoanthropology workflows where uncertainty bands feed into
downstream interpretive claims more directly than for typical industrial
spectroscopy.

Chemometrics-literature precedent: GP regression appears in NIR / Raman /
FTIR work (e.g. Chen et al. 2007 *J. Pharm. Biomed. Anal.* on tablet potency,
Ferrari et al. 2018 on biomedical IR), typically with RBF or Matern kernels
on PCA-scored or full-spectrum input. The combination "preprocessing +
varsel + GP" is supported by SciKit-learn `GaussianProcessRegressor` out of
the box, no new dependencies.

Why now: explicit user request. Adds a Bayesian non-parametric option to the
model menu. Cost is moderate (sklearn-only, no new deps, no PyInstaller
bundle changes), but training is **O(n³)** in calibration sample count so
the model needs guarding against being run on Quick / Standard tier where
users expect fast iteration.

## Implementation plan (verbatim from user, treat as draft)

### File 1: `src/spectral_predict/model_registry.py`

Line 23 — Add `'GP'` to `REGRESSION_MODELS`:

```python
'CatBoost',
'GP',  # Gaussian Process Regression (non-parametric Bayesian)
```

### File 2: `src/spectral_predict/model_config.py`

Lines 39-44 — Add `'GP'` to `comprehensive` and `experimental` tiers
(deliberately NOT in `quick` / `standard` — `O(n³)`):

```python
'comprehensive': {
    'models': ['PLS', 'Ridge', 'ElasticNet', 'RandomForest', 'LightGBM', 'XGBoost', 'SVR', 'GP'],
},
'experimental': {
    'models': [..., 'CatBoost', 'SVR', 'MLP', 'NeuralBoosted', 'GP'],
}
```

After line 348 — Add GP hyperparameter defaults in `get_hyperparameters()`:

```python
'GP': {
    'standard': {
        'kernel': ['rbf', 'matern'],
        'alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1],
        'n_restarts_optimizer': [0, 2, 5],
    },
},
```

### File 3: `src/spectral_predict/search.py`

Line 113 — Add `'GP'` to `SCALE_SENSITIVE_MODELS`:

```python
SCALE_SENSITIVE_MODELS = {'SVC', 'SVR', 'MLP', 'NeuralBoosted', 'Ridge', 'Lasso', 'ElasticNet', 'GP'}
```

Line 119 — Add to `MODELS_PREFER_SERIAL_CV` (GP has internal threading + `n³` kernel inversion):

```python
MODELS_PREFER_SERIAL_CV = {'SVM', 'PLS', 'PLS-DA', 'Ridge', 'Lasso', 'ElasticNet', 'GP'}
```

### File 4: `src/spectral_predict/models.py`

Lines 177-178 (after SVR block in `get_model()`) — Add:

```python
elif model_name == "GP":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, Matern
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2,
        normalize_y=True,
        random_state=42,
    )
```

After ~line 450 (in `build_model()`) — Add:

```python
elif model_name == "GP":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, Matern
    kernel_name = params.get("kernel", "matern")
    if kernel_name == "rbf":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=0.1)
    else:
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5) + WhiteKernel(noise_level=0.1)
    alpha = params.get("alpha", 1e-4)
    n_restarts = params.get("n_restarts_optimizer", 2)
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=42,
    )
```

After ~line 1310 (in `get_model_grids()`, after LightGBM block) — Add GP grid:

```python
if 'GP' in enabled_models:
    gp_configs = []
    for kernel_name in ['rbf', 'matern']:
        for alpha in [1e-8, 1e-6, 1e-4, 1e-2, 0.1]:
            for n_restarts in [0, 2, 5]:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, Matern
                if kernel_name == 'rbf':
                    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=0.1)
                else:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5) + WhiteKernel(noise_level=0.1)
                gp_configs.append(
                    (GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts, normalize_y=True, random_state=42),
                     {"kernel": kernel_name, "alpha": alpha, "n_restarts_optimizer": n_restarts})
                )
    grids["GP"] = gp_configs
```

Line 1866 (in `get_feature_importances()`) — GP has no native feature
importances. Skip it (the else clause already handles unknown models).

### File 5: `src/spectral_predict/nsga2_search.py`

Line 104 — Add `'GP'` to `MODEL_TYPES`:

```python
MODEL_TYPES = [
    'PLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest',
    'LightGBM', 'XGBoost', 'CatBoost', 'SVR', 'MLP', 'NeuralBoosted', 'GP'
]
```

Line 121 — Add GP gene mapping (uses kernel parameter = gene 11 gamma slot,
alpha = gene 5):

```python
'GP': [5, 11],  # alpha (reg_alpha), kernel_gamma
```

### File 6: `spectral_predict_gui_optimized.py`

5 spots, all following the LightGBM pattern:

1. ~line 3028 — Add `self.use_gp = tk.BooleanVar(value=False)` (default OFF — it's slow)
2. ~line 3105 — Add `'GP': self.use_gp` to `self.model_checkboxes`
3. ~line 12349 — Create checkbox widget:
   ```python
   self.gp_checkbox = ttk.Checkbutton(models_frame, text="GP", variable=self.use_gp)
   self.gp_checkbox.grid(row=12, column=0, sticky=tk.W, pady=5)
   ttk.Label(models_frame, text="Gaussian Process — Bayesian, non-parametric, O(n³)").grid(row=12, column=1, sticky=tk.W)
   ```
4. ~line 12366 — Add `'GP': self.gp_checkbox` to `self.model_checkbox_widgets`
5. ~line 23159 — Add `if self.use_gp.get(): selected_models.append("GP")`
6. ~line 14818 — Add `'GP'` to `refine_model_combo['values']`

## Open questions / concerns to address at implementation time

1. **30-config grid is heavy for an `O(n³)` model.** 2 kernels × 5 alphas
   × 3 restart counts = 30 GP fits per CV fold. On a 5-fold × 10-repeat CV
   with n=200 calibration spectra, that's 30 × 50 = 1500 fits, each taking
   GP-O(n³) work. Realistic wall-clock could be hours. Suggest reducing
   the default grid (e.g. drop the alpha=1e-8 and 0.1 extremes, drop
   n_restarts=5) and let users opt in to the full grid via the per-model
   config card if needed.

2. **NSGA-II gene-11 (gamma slot) repurposed for kernel name.** The kernel
   parameter is categorical (`'rbf'` vs `'matern'`), but the gene slot is
   typically a float. Verify the NSGA-II decode handles string-vs-float
   correctly for gene 11 when `model_type == 'GP'` — may need a special
   case in the decoder, or use gene 11 as a continuous value mapped to
   `{0: 'rbf', 1: 'matern'}` via threshold.

3. **`n_restarts_optimizer` is sklearn 1.0+** — fine, project floor is
   `scikit-learn>=1.5` per `pyproject.toml`. No bump needed.

4. **WhiteKernel handles homoscedastic noise only.** If the bone-FTIR data
   has heteroscedastic noise (variance changes with wavenumber / sample
   group), `WhiteKernel` is wrong. Could switch to a heteroscedastic kernel
   later, but homoscedastic is the reasonable default for V1.

5. **`get_feature_importances()` skip path needs verification.** The user
   says "the else clause already handles unknown models" — verify that
   running a one-class subset selection or any varsel-importance-driven
   path doesn't crash when the active model is GP. If it does, add an
   explicit "GP has no feature importances" early-return rather than
   relying on `else`.

6. **Bundling: GaussianProcessRegressor is in sklearn already** — no
   PyInstaller bundle update needed. But verify the bundled
   `sklearn.gaussian_process` module is included; the PyInstaller `.spec`
   may need a hidden-import declaration if it's lazy-loaded only here.

7. **Test plan.** New tests should at minimum cover:
   - GP appears in `REGRESSION_MODELS` and the `comprehensive` /
     `experimental` tier model lists
   - GP is in `SCALE_SENSITIVE_MODELS` (regression test for missing scaler
     would surface as numerical instability)
   - `build_model("GP", {"kernel": "rbf", ...})` returns a configured
     `GaussianProcessRegressor`
   - `get_model_grids({'GP'})` returns 30 configs (or revised count)
   - End-to-end smoke: tier=comprehensive on a ≤50-sample synthetic
     dataset includes GP without crashing
   - GUI: GP checkbox toggles `selected_models` on Run Analysis click

8. **Documentation.** Add GP to `docs/MACHINE_LEARNING_MODELS.md` and the
   user guide (section 3 — Models). Cite sklearn docs + at least one
   chemometrics-literature reference (Chen 2007 or Ferrari 2018).

## Success criteria

- All 6 file edits applied without breaking existing tests.
- `tier=comprehensive` and `tier=experimental` runs include GP and produce
  a result row with `model='GP'` + sensible RMSEcv.
- GP can be enabled / disabled via the GUI checkbox; default state is OFF.
- Refinement-tab dropdown lists `'GP'` as an available refine target.
- Documentation updated.

## Out of scope

- Heteroscedastic kernels (homoscedastic-only for V1).
- Sparse / inducing-point GP variants for large `n` (dasp's typical
  spectroscopy `n` is ≤ 500; full-rank GP is feasible).
- GP-specific predictive-uncertainty UI (the `predict_proba` analog).
  Could land later if user demand surfaces.
