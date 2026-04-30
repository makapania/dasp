# Technical Gaps: Detailed Findings

> **Date**: 2026-04-29
> **Companion to**: `MASTER_ANALYSIS.md`

---

## 1. Preprocessing

### Methods Inventory

| Category | Methods | File |
|----------|---------|------|
| Scatter correction | SNV, MSC | `preprocess.py:9`, `interference.py:217` |
| Derivatives | SG 1st-4th order | `preprocess.py:48` |
| Smoothing | SG, Moving Average, Gaussian | `preprocess.py:137, 223, 256` |
| Baseline | Polynomial, ALS, airPLS, Rubber Band | `baseline.py:9-401` |
| Advanced baseline | 15+ via pybaselines (arPLS, SNIP, ModPoly...) | `baseline_advanced.py:30-325` |
| Signal correction | OSC, EPO, DOSC, GLSW | `interference.py:356-1445` |
| Wavelength | Exclusion ranges | `interference.py:75` |

### Critical Gap: EMSC (Extended Multiplicative Scatter Correction)
**Status**: Completely absent. No file references found.
**Why critical**: EMSC is the gold standard for separating chemical from physical information in NIR. Unlike MSC (only additive + multiplicative), EMSC models scatter as a function of wavelength using known reference spectra. Essential for pharmaceutical tablets, agricultural samples, food powders.
**Current workaround**: Chain MSC + EPO, which is not equivalent.
**Fix**: New sklearn transformer, ~200 lines. `interference.py` or new `emsc.py`.

### Important Gaps

- **Norris-Williams gap derivatives**: Unscrambler offers these; only SG derivatives available
- **Spectral normalization**: No Min-Max, Area, or Vector normalization as standalone steps
- **Mean centering**: Not exposed as a user-controllable step (happens implicitly in PLS/OSC)
- **Deconvolution / peak fitting**: Not implemented (low priority for quantitative NIR)

### Implementation Quality Issues

| Issue | Location | Detail |
|-------|----------|--------|
| Duplicate EPO class | `interference.py:565 vs 820` | Stub at line 565 raises `NotImplementedError`; full implementation at 820 silently overwrites it. Delete the stub. |
| ALS default p mismatch | `baseline.py:139` vs `preprocess.py:483` | Class default `p=0.001`, pipeline builder uses `p=0.01`. 10x difference affects baseline quality. |
| OSC is simplified | `interference.py:496` | Uses loading as orthogonal direction instead of proper Wold (1998) projection. Works but may remove some Y-relevant variation. DOSC (`interference.py:1158`) is actually more rigorous. |
| GLSW is diagonal-only | `interference.py:711` | Uses per-wavelength variance instead of full covariance matrix. Ignores inter-wavelength noise correlations. |
| NaN not guarded | `preprocess.py`, `baseline.py` | No NaN/Inf guards on input. Bare `np.asarray()` silently propagates NaN. |
| Preprocessing discovery limited | `preprocessing_discovery.py:29-49` | Only tests raw, SNV, and SG derivatives. Never discovers baseline + SNV or other combinations. |
| Fixed pipeline order | `preprocess.py:325` | Baseline -> Smoothing -> SNV/Derivs always. GUI Explore tab allows reordering, but automated search does not. |

### What Exceeds Unscrambler
- 20+ baseline methods (Unscrambler has 2-3)
- Real-time auto-refresh preview (Unscrambler requires Apply)
- 3rd and 4th derivative support
- EPO, DOSC, GLSW (not in Unscrambler)

---

## 2. Model Implementations

### Models Offered

**Regression**: PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, SVR, MLP, NeuralBoosted

**Classification**: PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP, NeuralBoosted

**One-Class**: PCA-SIMCA, OneClassSVM, IsolationForest, EllipticEnvelope, LOF

### Critical Gaps

#### PCR (Principal Component Regression) -- NOT IMPLEMENTED
Grep for "PCR", "PrincipalComponentRegression" returned zero results. PCR is a fundamental chemometrics model (PCA + OLS on scores) that Unscrambler has had since the 1990s. Essential for method comparison studies, teaching, and regulatory compliance where specific models are mandated.
**Fix**: `Pipeline([PCA(n_components), LinearRegression()])` -- trivially implemented.

#### PLS-2 (Multi-Y) -- NOT SUPPORTED
The entire search pipeline assumes 1D y (`search.py:4086+`). sklearn's `PLSRegression` handles multi-output natively, but the pipeline doesn't accept multi-column Y.
**Fix**: Extend pipeline to accept DataFrame Y, add multi-target column selector in GUI.

#### MLR (Multiple Linear Regression) -- NOT IMPLEMENTED
No `LinearRegression` found in model registry. Ridge is a regularized version, but plain MLR is the baseline all others are compared to. Important after variable selection reduces to a small feature set.
**Fix**: Add `LinearRegression` as `MLR` model. Trivial.

#### k-NN -- NOT IMPLEMENTED
Not a chemometrics workhorse, but expected in a comprehensive tool. Useful for quick non-linear baselines.

### PLS Implementation Quality

- Uses `sklearn.cross_decomposition.PLSRegression(n_components=nc, scale=False)` (`models.py:132`)
- `scale=False` is **correct** for spectroscopy (all variables same units)
- Component selection tests ALL integers from 1 to max (`models.py:840-843`)
- Loadings/scores/weights accessible via `__getattr__` forwarding (`models.py:90-99`)

#### VIP Bug (Minor)
`models.py:1738`: `ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)` uses global `var(y)` instead of PLS Y-loadings. The canonical Wold (2001) formula uses `y_loadings_`. Works reasonably for single-Y PLS but deviates from the published formula.

### Hyperparameter Grid Issues

| Model | Issue | Location |
|-------|-------|----------|
| SVR | Grid too narrow: C=[1,10], gamma=['scale'], epsilon=[0.1] | `model_config.py:312-322` |
| MLP | Grid too narrow: alpha=[0.001], lr=[0.001], activation=['relu'] only | `model_config.py:324-335` |
| RandomForest | n_estimators=[100] only, max_features=['sqrt'] only | `model_config.py:247-258` |
| PLS | Excellent: tests all component counts | `models.py:840-843` |
| XGBoost | Well-tuned for spectroscopy | `model_config.py:259-276` |
| LightGBM | Good with explicit depth limits | `model_config.py:277-293` |
| CatBoost | Adequate after recent improvement | `model_config.py:294-310` |

### Cross-Validation

**Correctly implemented**:
- K-Fold, Stratified K-Fold, Repeated K-Fold, LOO via `cv_utils.py`
- Pooled RMSEcv (not mean-of-fold-RMSEs) -- matches chemometrics convention (`search.py:4347`)
- Proper stratification for classification (`cv_utils.py:227`)

**Missing**: Venetian Blinds (Unscrambler-specific round-robin splitter -- culturally important for Unscrambler users)

### Model Diagnostics

**Implemented and matching Unscrambler**: Leverage (`diagnostics.py:58`), Hotelling T-squared (`model_io.py:1076`), Q-residuals (`model_io.py:1101`), prediction intervals (`diagnostics.py:143`), residual plots.

**Key issue**: Jackknife prediction intervals at `diagnostics.py:143-230` are fully implemented but **never called** from the prediction workflow or GUI. Leverage is computed but not integrated into validation metrics.

---

## 3. Variable Selection

### 19 Methods Offered

| Method | Location | Unscrambler Has? |
|--------|----------|:----------------:|
| importance (VIP/tree/coeff) | `models.py:1754` | Partial |
| SPA | `variable_selection.py:293` | No |
| UVE | `variable_selection.py:20` | No |
| iPLS (basic) | `variable_selection.py:489` | Yes |
| iPLS forward | `variable_selection.py:1727` | Partial |
| iPLS backward (biPLS) | `variable_selection.py:1915` | No |
| MC-siPLS | `variable_selection.py:2119` | No |
| MWPLS | `variable_selection.py:2242` | No |
| CARS | `variable_selection.py:1239` | No |
| CARS-aware | `variable_selection.py:1239` | No |
| CARS-tree | `variable_selection.py:1239` | No |
| VCPA-IRIV | `wavelength_selection.py:352` | No |
| GA-PLS / GA-LightGBM | `ga_pls.py:476`, `ga_lightgbm.py:522` | No |
| UVE + CARS | `variable_selection.py:812` | No |
| UVE + CARS-tree | `variable_selection.py:812` | No |
| UVE + CARS + SPA | `variable_selection.py:919` | No |
| Forward iPLS + SPA | `variable_selection.py:1044` | No |
| Forward iPLS + CARS | `variable_selection.py:1139` | No |
| UVE + SPA | `variable_selection.py:677` | No |

### Bugs and Issues

| Issue | Location | Detail |
|-------|----------|--------|
| UVE docstring reversed | `variable_selection.py:44` | Says "Values > 1.0 make filtering more conservative" but actually > 1.0 eliminates more variables (more aggressive). Code at line 160 confirms. |
| SPA deterministic | `variable_selection.py:407` | `n_random_starts=10` parameter implies randomness, but always selects first variable by max correlation with y. Docstring acknowledges: "currently SPA is deterministic." |
| Duplicate UVE code | `variable_selection.py:20-176` and `179-290` | `uve_selection()` and `get_uve_threshold()` duplicate ~80 lines of core logic. |
| Duplicate SPA | `variable_selection.py:293` vs `wavelength_selection.py:33` | Two independent implementations: correlation-based (used) vs QR-based (dead code from main pipeline perspective). |
| Duplicate CARS | `variable_selection.py:1239` vs `wavelength_selection.py:162` | Same situation. |
| Intervals use feature indices | `variable_selection.py:1620` | `interval_size = n_features // n_intervals` uses feature count, not wavelength range. Misleading for non-linear wavelength spacing. |
| GA-PLS classification | `ga_pls.py:207-217` | Uses median threshold for binary classification. Multi-class will silently produce wrong results. |

### What Unscrambler Has That SP Doesn't
- Interactive graphical wavelength selection from spectral plots
- PCA loadings-guided selection
- Regression coefficient click-to-select

---

## 4. File I/O

### Format Support

| Format | Read | Write | Multi-sample |
|--------|:----:|:-----:|:------------:|
| CSV (wide/long) | Yes | Yes | Yes |
| Excel | Yes | Yes | Yes |
| Combined CSV/Excel | Yes | -- | Yes |
| ASD (ASCII/binary) | Yes | -- | Yes |
| SPC (GRAMS) | Yes | Yes | Read: multi, Write: **single only** (`io.py:3648`) |
| JCAMP-DX | Yes | Yes | Read: multi, Write: **single only** (`io.py:3695`) |
| Bruker OPUS | Yes | -- | Yes |
| PerkinElmer | Yes | -- | Yes |
| Thermo Omnic | Yes | -- | Yes |
| Agilent | Yes (3.10 only) | -- | Yes |
| ASCII text | Yes | Yes | Read: multi, Write: **single only** (`io.py:3739`) |
| MATLAB .mat | **No** | **No** | -- |
| NetCDF/HDF5 | **No** | **No** | -- |

### Critical Gaps
- **No MATLAB .mat support** -- the lingua franca of scientific data exchange
- **Single-spectrum export limitation** for SPC, JCAMP, ASCII formats
- **No recursive directory scanning** -- all readers use flat glob
- **No parallel file reading** -- sequential for-loops on 1000+ files
- **In-memory only** -- no out-of-core or chunked reading for large datasets
- **SPC multi-subfile loses data** (`io.py:965`) -- only first subfile used

### Validation Gaps
- No wavelength spacing uniformity check (SG derivatives require uniform spacing)
- No spectral value range validation (physically impossible values pass silently)
- No spectral artifact detection (cosmic rays, saturation, detector overflow)
- No encoding detection for CSV files (international instruments may use Latin-1)

### Model Save/Load
- Well-structured `.dasp` ZIP archives with metadata.json + joblib pickles
- Version tracking, ensemble support, fast metadata inspection
- **Security risk**: No integrity verification on pickle files (see CODE_QUALITY.md)
- **Cross-version fragility**: Joblib pkl files depend on Python and sklearn versions

---

## 5. Calibration Transfer

### 6 Methods Implemented

| Method | Location | Unscrambler Has? |
|--------|----------|:----------------:|
| DS (Direct Standardization) | `calibration_transfer.py:114` | Yes |
| PDS (Piecewise DS) | `calibration_transfer.py:170` | Yes |
| TSR (Shenk-Westerhaus) | `calibration_transfer.py:386` | Yes |
| CTAI (Affine Invariance) | `calibration_transfer.py:641` | **No** |
| NS-PFCE (Non-supervised) | `calibration_transfer.py:971` | **No** |
| JYPLS-inv (Joint-Y PLS) | `calibration_transfer.py:1376` | **No** |

### Missing vs Unscrambler
- SST (Standardization via Signal Transformation) -- the one standard method not implemented

### Issues
- CTAI docstring says "need not be same samples" but code requires paired (`calibration_transfer.py:803-810`)
- NS-PFCE has verbose `print()` flood (`calibration_transfer.py:755-898`)
- NS-PFCE objective capped at 100 samples (`calibration_transfer.py:1184`)

---

## 6. Prediction and Validation

### Prediction Workflow
Well-designed with auto-preprocessing, multi-model batch prediction, consensus computation, and applicability domain.

### Applicability Domain -- Excellent
- 4-zone classification: within_domain / influential / new_features / outside_domain (`model_io.py:1117`)
- Per-sample reliability scores 0-95% (`model_io.py:1131`)
- Hotelling T-squared with F-distribution threshold (`model_io.py:1076`)
- Q-residuals with chi-squared threshold (`model_io.py:1101`)

### Ensemble Methods -- Excellent
6 types (SimpleAverage, RegionAware, MixtureOfExperts, Stacking, RegionSpecialist, ClassSpecialist) in `ensemble.py`. Out-of-fold predictions prevent data leakage. Auto-ensemble generation from regional specialists.

### Bias Correction -- Exceeds Unscrambler
Linear + nonlinear (polynomial/spline) correction in `bias_correction.py`. Unscrambler only offers linear.

### Missing vs Unscrambler
- Per-sample prediction intervals (jackknife code exists at `diagnostics.py:143` but **never wired into GUI**)
- Permutation testing / Y-randomization for model validity
- Durbin-Watson residual autocorrelation test
- X-Y correlation loading plots
- Multi-block data analysis
