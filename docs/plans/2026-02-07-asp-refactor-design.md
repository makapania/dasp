# ASP — Spectral Predict Refactor Design

> **Date**: 2026-02-07
> **Repo**: `C:\Users\sponheim\git\asp`
> **Status**: Design approved, ready for implementation planning
> **Origin**: Refactor of `dasp` (`C:\Users\sponheim\git\dasp`)

---

## 1. Goals

1. **Modular architecture** — easy for humans and agents to navigate, modify, and test
2. **Full feature parity** — every button, checkbox, and option from the current 15-tab GUI
3. **Modern UI** — PySide6 (Qt) with qt-material theme, solving Tkinter's scrolling limitations
4. **Clean start** — new repo, no legacy baggage, old repo stays as-is on GitHub

## 2. Constraints

- **Preprocessing-first**: Preprocess full spectrum once, then all models/subsets use that result. No data leakage.
- **"No preprocessing" is a valid choice**: Models can receive raw data if the user selects no preprocessing.
- **Every tab feature retained**: No "minor" features dropped. Cross-reference against GUI audit.
- **Reliability is non-negotiable**: This software produces insights for climate change and health research.
- **Backend knows nothing about the GUI**: Clean separation so the GUI is just a skin.

## 3. Project Structure

```
asp/
├── pyproject.toml
├── src/
│   └── spectral_predict/
│       ├── core/
│       ├── readers/
│       ├── preprocessing/
│       ├── models/
│       ├── search/
│       ├── selection/
│       ├── ensemble/
│       ├── transfer/
│       ├── analysis/
│       ├── library/
│       ├── export/
│       ├── utils/
│       ├── data_management/
│       └── gui/
├── tests/
└── example/
```

**Guiding principles:**
- One level of folder grouping, no deeper
- Each file under 500 lines
- Each function under 50 lines
- No god classes

---

## 4. Backend Modules

### 4.1 core/ — Foundation

| File | Contents |
|------|----------|
| `types.py` | `SpectralData` (spectra, target, wavelengths, sample_ids, metadata, data_type), `SearchResults`, `TransferModel` dataclass |
| `config.py` | `SearchConfig`, `PreprocessConfig`, `ExportOptions` — validated settings with dataclasses |
| `exceptions.py` | `SpectralPredictError` hierarchy: `DataLoadError`, `AlignmentError`, `PreprocessingError`, `SearchError`, `ConfigurationError`, `ValidationError` |
| `constants.py` | `RANDOM_STATE=42`, model group sets (`PLS_MODELS`, `TREE_MODELS`, `SCALE_SENSITIVE_MODELS`, etc.), tier names |

### 4.2 readers/ — File I/O

| File | Contents | Current source |
|------|----------|---------------|
| `base.py` | `BaseReader` interface (read_file, read_directory, extensions) | new |
| `csv.py` | read/write CSV (single file + directory + combined format) | `io.py` |
| `excel.py` | read/write Excel (single file + combined format) | `io.py` |
| `asd.py` | ASD native binary reader + R-bridge (asdreader, prospectr) | `io.py`, `readers/asd_native.py`, `readers/asd_r_bridge.py` |
| `opus.py` | Bruker OPUS reader + wavenumber-wavelength conversion | `readers/opus_reader.py` |
| `spc.py` | Thermo SPC read/write | `io.py` |
| `jcamp.py` | JCAMP-DX read/write | `io.py` |
| `ascii.py` | Generic ASCII read/write | `io.py` |
| `perkinelmer.py` | PerkinElmer .sp reader | `readers/perkinelmer_reader.py` |
| `agilent.py` | Agilent reader (.seq, .dmt, .asp, directory) | `readers/agilent_reader.py` |
| `alignment.py` | `align_xy()`, `normalize_filename()`, duplicate ID handling | `io.py` |
| `detection.py` | Auto-detect format, detect reflectance/absorbance, identify wavelength columns, auto-detect specimen ID and Y columns | `io.py` |

### 4.3 preprocessing/ — Spectral Transforms

| File | Contents | Current source |
|------|----------|---------------|
| `base.py` | `BasePreprocessor` (fit/transform interface) | new |
| `snv.py` | `SNV`, `SNV-Detrend` | `preprocess.py` |
| `derivatives.py` | `SavgolDerivative` (1st-4th order), `SavgolSmooth`, window auto-adjustment for constrained ranges | `preprocess.py` |
| `baseline.py` | `BaselinePolynomial`, `BaselineALS`, `BaselineAirPLS`, `rubber_band_baseline()` (Andrew's convex hull). **All available as analysis preprocessing options.** | `baseline.py`, `ensemble_preprocessing.py` imports |
| `msc.py` | `MSC` (Multiplicative Scatter Correction) | `interference.py` |
| `transforms.py` | Absorbance-reflectance conversion, scale inference | `io.py` |
| `pipeline.py` | Chain preprocessors, `PreprocessorConfig` (reconstruction without storing fitted objects) | `preprocessing_wrapper.py` |
| `discovery.py` | Smart preprocessing selection, exhaustive search, GA optimization, smart exhaustive with robust validation. Constants: `PREPROC_TYPES`, `WINDOW_SIZES`, `MODEL_TO_PROXY`. | `preprocessing_discovery.py`, `ga_preprocessing.py` |
| `learned.py` | `LearnedSpectralPreprocessing` (PyTorch Conv1d), `SpectralPreprocessorWithRegressor`, `LearnedPreprocessor`. Graceful fallback if PyTorch unavailable. | `learned_preprocessing.py` |

### 4.4 models/ — Model Management

| File | Contents | Current source |
|------|----------|---------------|
| `base.py` | `BaseModel` wrapper interface | new |
| `pls.py` | `PLSTransformer` (2D output fix), PLS-DA (PLS + LogisticRegression pipeline) | `models.py` |
| `linear.py` | Ridge, Lasso, ElasticNet | `models.py` |
| `trees.py` | RandomForest, LightGBM, XGBoost, CatBoost | `models.py` |
| `svm.py` | SVR, SVC | `models.py` |
| `neural.py` | MLP (regressor/classifier), `NeuralBoostedRegressor`, `NeuralBoostedClassifier` (stagewise MLP ensembles, Huber loss) | `models.py`, `neural_boosted.py` |
| `grids.py` | Hyperparameter grids per tier (Quick/Standard/Comprehensive/Experimental), `get_feature_importances()`, VIP computation | `models.py`, `model_config.py` |
| `registry.py` | Model registry: regression models, classification models, feature importance support, `get_supported_models()` | `model_registry.py`, `model_config.py` |

### 4.5 search/ — Search Strategies

| File | Contents | Current source |
|------|----------|---------------|
| `orchestrator.py` | `run_search()` — main coordinator. Preprocess-first loop. Validation metrics for top models. Edge masking. Imbalance/resampling pipeline integration. | `search.py` |
| `grid.py` | Grid search with CV, single-fold/single-config helpers | `search.py` |
| `bayesian.py` | Unified Bayesian optimization (Optuna TPE). `suggest_preprocessing()`, `suggest_model_params()`, `create_unified_objective()`, `run_unified_bayesian()`, `convert_study_to_dataframe()`. Search space definitions per model. | `unified_bayesian.py`, `bayesian_config.py`, `bayesian_utils.py` |
| `nsga2.py` | NSGA-II multi-objective. `SmartMutation`, `ImportanceTracker`, `SeededWavelengthSampling`, `SpectralOptimizationProblem`. `find_knee_point()`, `run_nsga2_search()`. Guided mode. | `nsga2_search.py` |
| `phases.py` | Full spectrum, variable selection, region analysis, iPLS phases | `search.py` (extracted from `run_search()`) |
| `results.py` | `compute_composite_score()` (multi-metric ranking, 0-10 penalty), result DataFrame creation, imbalance metrics, specificity | `scoring.py` |
| `cv.py` | `cross_validate_with_early_stopping()`, `cross_val_predict_with_early_stopping()`, `cross_val_score_with_early_stopping()`. Boosting model detection. | `cv_utils.py` |
| `controller.py` | `SearchController` (pause/resume/stop via threading events), progress tracking | `search_controller.py`, `progress_monitor.py` |

### 4.6 selection/ — Variable & Region Selection

| File | Contents | Current source |
|------|----------|---------------|
| `base.py` | `BaseSelector` interface | new |
| `importance.py` | Feature importance-based selection. Multiple methods: VIP, tree, coefficient, neural, SVM, CARS-tree, LightGBM, model-specific. | `preprocessing_discovery.py` (`compute_importance`), `variable_selection.py` |
| `uve.py` | `uve_selection()`, `get_uve_threshold()`, `uve_spa_selection()` (hybrid) | `variable_selection.py` |
| `spa.py` | `spa_selection()` (minimally correlated variables via iterative projection) | `variable_selection.py` |
| `cars.py` | `cars_selection()` (Monte Carlo + PLS/LightGBM), CARS-Tree variant | `variable_selection.py` |
| `vcpa.py` | `vcpa_iriv()` (Mann-Whitney U test variable classification) | `wavelength_selection.py` |
| `ipls.py` | `ipls_selection()` (interval evaluation), `ipls_forward()` (iterative combine), `ipls_backward()` (iterative remove). Helpers: `_create_intervals()`, `_evaluate_interval_pls()`, `_get_combined_indices()`, `_get_wavelength_ranges()` | `variable_selection.py` |
| `regions.py` | `compute_region_correlations()`, `get_top_regions()`, `get_region_variable_indices()`, `create_region_subsets()`, `format_region_report()`. Supports individual + pairwise combos. | `regions.py` |
| `ga.py` | `GAPLSSelector` (binary GA + PLS fitness), `GALightGBMSelector` (binary GA + LightGBM), `FitnessCache` (thread-safe) | `ga_pls.py`, `ga_lightgbm.py` |

### 4.7 ensemble/ — Ensemble Methods

| File | Contents | Current source |
|------|----------|---------------|
| `methods.py` | `SimpleAverageEnsemble`, `RegionAwareWeightedEnsemble`, `MixtureOfExpertsEnsemble`, `StackingEnsemble`, `RegionSpecialistEnsemble`, `ClassSpecialistEnsemble`. All support per-model preprocessing via `PreprocessorConfig`. | `ensemble.py` |
| `preprocessing.py` | `StackedPreprocessingRegressor`, `StackedPreprocessingClassifier`, `create_standard_preprocessing_ensemble()`. Meta-models: RidgeCV / LogisticRegressionCV. | `ensemble_preprocessing.py` |
| `auto.py` | `create_auto_ensembles()`, `compute_regional_rankings()`, `compute_class_rankings()`, `select_top_models_per_region()`, `select_top_models_quartile_flat()`, `extract_preprocessor_config()` | `ensemble.py` |
| `viz.py` | `plot_regional_performance()`, `plot_ensemble_weights()`, `plot_model_specialization_profile()`, `plot_prediction_comparison()`, `create_ensemble_report()` | `ensemble_viz.py` |

### 4.8 transfer/ — Calibration Transfer

| File | Contents | Current source |
|------|----------|---------------|
| `methods.py` | `estimate_ds()`, `apply_ds()`, `estimate_pds()`, `apply_pds()`, `estimate_tsr()`, `apply_tsr()`, `estimate_ctai()`, `apply_ctai()`, `estimate_nspfce()`, `apply_nspfce()`, `estimate_jypls_inv()`, `apply_jypls_inv()`, `apply_transfer_dispatch()` | `calibration_transfer.py` |
| `io.py` | `TransferModel` dataclass, `save_transfer_model()`, `load_transfer_model()` | `calibration_transfer.py` |
| `utils.py` | `resample_to_grid()`, `clip_wavelengths_to_region()` | `calibration_transfer.py` |

### 4.9 analysis/ — Problem-Specific Analysis

| File | Contents | Current source |
|------|----------|---------------|
| `outliers.py` | `run_pca_outlier_detection()`, `compute_pca_outlier_scores()` (T² + Q-residuals), `identify_outliers()`, `mahalanobis_distance()` | `outlier_detection.py` |
| `interference.py` | `WavelengthExcluder`, `MSC` (reference in preprocessing/), `OSC`, `EPO`, `GLSW`, `DOSC` | `interference.py` |
| `contaminant.py` | `DifferenceAnalyzer`, `EstimatedEPO` (pca_diff/mean_diff/bootstrap), `ContaminantOPLSDA`, `ContaminantGLSW`, `RegionExcluder` (backward iPLS), `MultiContaminantAnalyzer`, `MultiGroupEPO`, `MultiContaminantGLSW`, `analyze_contaminant_influence()` | `contaminant_analysis.py` |
| `imbalance.py` | **Classification**: `ClassificationResampler` — SMOTE, ADASYN, BorderlineSMOTE, RandomUnderSampler, TomekLinks, SMOTETomek, SMOTEENN. **Regression**: `RegressionUndersampler` (bin-based), `RegressionResampler` (oversample/smogn/smotetomek), `RegressionSampleWeighter` (binning/rare_boost/balanced). **Utilities**: `detect_class_imbalance()`, `detect_regression_imbalance()`, `build_imbalance_transformer()`, `get_available_methods()`, `recommend_imbalance_method()`, `validate_classification_config()`, `validate_imbalance_with_features()` | `imbalance.py` |
| `sample_selection.py` | `kennard_stone()`, `duplex()`, `spxy()`, `random_selection()`, `compare_selection_methods()` | `sample_selection.py` |
| `similarity.py` | `hit_quality_index()`, `spectral_angle_mapper()`, `sam_to_similarity()`, `euclidean_distance()`, `euclidean_to_similarity()`, `cosine_similarity()`, `first_derivative_correlation()`, `second_derivative_correlation()`, `spectral_information_divergence()`, `sid_to_similarity()`, `compute_similarity()`, `compute_batch_similarity()`, `METRICS` registry | `similarity_metrics.py` |
| `diagnostics.py` | `compute_residuals()`, `compute_leverage()`, `qq_plot_data()`, `jackknife_prediction_intervals()`, `compute_pls_complexity_curve()`, `compute_sklearn_validation_curve()`, `compute_ensemble_validation_curve()`, `compute_regularization_validation_curve()`, `compute_learning_curve()` | `diagnostics.py` |

### 4.10 library/ — Spectral Library

| File | Contents | Current source |
|------|----------|---------------|
| `library.py` | `SpectralLibrary` (persistent storage, auto-save, duplicate detection via fingerprinting), `LibraryEntry` dataclass, `get_library()`, `add_to_library()` | `library_search.py` |
| `search.py` | `search_library()`, batch similarity, category filtering | `library_search.py` |

### 4.11 export/ — Code Generation & Model I/O

| File | Contents | Current source |
|------|----------|---------------|
| `model_io.py` | `save_model()`, `load_model()`, `predict_with_model()`, `predict_with_uncertainty()`, `get_model_info()`, `save_ensemble()`, `load_ensemble()`. ZIP-based .dasp format. | `model_io.py` |
| `code_gen.py` | `CodeGenerator` (Python scripts + Jupyter notebooks), `ExportOptions`, embedded data encoding, Colab support | `code_generator.py` |
| `r_gen.py` | `RCodeGenerator` (R wrapper via reticulate, base64+gzip encoded Python) | `r_code_generator.py` |
| `bundle.py` | `ExportBundle` (ZIP: python/, r/, data/, docs/, requirements.txt, install_packages.R) | `export_bundle.py` |
| `report.py` | `write_markdown_report()` (top-5 models) | `report.py` |
| `templates/` | Code generation templates: `header.py`, `preprocessing.py` (SNV, SavGol, MSC), `models.py` (17 model templates + imports + defaults), `selection.py` (VIP, SPA, UVE, CARS), `validation.py` (CV, metrics, final model, prediction), `visualization.py` (pred vs actual, residuals, spectra, confusion matrix) | `templates/` |

### 4.12 utils/ — Shared Utilities

| File | Contents | Current source |
|------|----------|---------------|
| `instruments.py` | `InstrumentProfile` dataclass, `characterize_instrument()`, `compute_wavelength_spacing()`, `compute_roughness()`, `detect_interpolation()`, `analyze_peaks()`. Equalization: `choose_common_grid()`, `build_equalization_mapping_for_instrument()`, `equalize_dataset()` | `instrument_profiles.py`, `equalization.py` |
| `progress.py` | Progress tracking (Qt-based, replaces Tkinter `ProgressMonitor`) | `progress_monitor.py` |

### 4.13 data_management/ — Multi-Source Data

| File | Contents | Current source |
|------|----------|---------------|
| `manager.py` | `DataSource` dataclass, `MergedDataset` dataclass, `DataSourceManager` (add/remove/merge sources, wavelength alignment strategies: intersection/union/interpolation, duplicate handling: error/keep_first/keep_last/rename, sample filtering: regex/value_range/sample_list, wavelength trimming, export, save/load config) | `data_management.py` |

---

## 5. GUI Architecture (PySide6 + qt-material)

### 5.1 Structure

```
gui/
├── app.py                    # QMainWindow, sidebar nav, tab container
├── state.py                  # Central app state
│
├── tabs/
│   ├── base.py               # BaseTab interface
│   ├── data_management.py    # Tab 0:  4 subtabs
│   ├── import_tab.py         # Tab 1:  2 subtabs
│   ├── explore.py            # Tab 2:  9 subtabs
│   ├── data_viewer.py        # Tab 3:  spreadsheet editor
│   ├── quality_check.py      # Tab 4:  PCA outlier detection
│   ├── analysis_config.py    # Tab 5:  5 subtabs
│   ├── analysis_progress.py  # Tab 6:  live progress
│   ├── results.py            # Tab 7:  ranked results table
│   ├── model_dev.py          # Tab 8:  4 subtabs
│   ├── prediction.py         # Tab 9:  2 subtabs
│   ├── multi_model.py        # Tab 10: multi-model comparison
│   ├── calibration.py        # Tab 11: cal transfer + refl/abs conversion
│   ├── interference.py       # Tab 12: 4 subtabs
│   ├── spectral_library.py   # Tab 13: 2 subtabs
│   └── contaminant.py        # Tab 14: 4 subtabs
│
├── widgets/
│   ├── spectra_plot.py       # Matplotlib canvas (reusable)
│   ├── data_table.py         # Spreadsheet widget (replaces tksheet)
│   ├── collapsible.py        # Collapsible section
│   ├── file_browser.py       # File/dir picker with format detection
│   ├── card.py               # Card UI pattern
│   └── progress_bar.py       # Progress with ETA
│
└── services/
    ├── search_service.py     # Background search thread + signals
    ├── file_service.py       # Reader dispatch + format detection
    └── export_service.py     # Code gen, model I/O, bundles
```

### 5.2 Tab Feature Inventory

Each tab replicates every control from the current GUI. Full audit reference below.

**Tab 0 — Data Management (4 subtabs)**
- 0A Import Sources: TreeView of sources, Add/Remove/Refresh/Save Config/Load Config/Clear All, Browse spectral + reference, Load Source, Use for Analysis
- 0B Merge & Combine: Source checkboxes, Wavelength alignment (intersection/union/interpolation), Duplicate handling (error/keep first/keep last/rename), Preview/Execute/Merge & Use
- 0C Data Manipulation: Sample filtering (regex/value range/sample list), Wavelength trimming (min/max), Spectral conversion (reflectance/absorbance + save CSV/Excel), Export
- 0D View Merged Data: Spreadsheet widget, Load/Delete Column/Save Changes, right-click column add

**Tab 1 — Import & Preview (2 subtabs)**
- 1A Data: Spectral file/dir browse, Reference CSV/Excel browse, Combined data browse, Load Data button, Append checkbox, Clear All. Advanced: skip rows, custom headers, transpose, first row as wavelengths, wavelength column, wavelength range restriction, combine targets, target column, data type (reflectance/absorbance/auto-detect)
- 1B Plots: Dynamic plot tabs (populated after data load)

**Tab 2 — Explore (9 subtabs)**
- Raw Spectra, 1st Derivative, 2nd Derivative (SG window spinbox), Target Distribution, Predictor Screening (method combobox: VIP/Random Forest, Top N spinbox, wavelength range, Run button)
- Rubber Band BL, Polynomial BL, ALS Baseline, Manual Baseline (each with Save Corrected / Replace Working Data)
- Color-by dropdown (None / Y Value / metadata), bin method, # groups

**Tab 3 — Data Viewer**
- Spreadsheet with virtual scrolling, Show excluded checkbox, Export CSV, Save Changes, Undo, Add/Delete Column, Delete Rows, Exclude/Include Selected

**Tab 4 — Data Quality Check**
- PCA Components spinbox, Y Range (min/max), Run Outlier Detection, Export Report, visualization tabs

**Tab 5 — Analysis Configuration (5 subtabs)**
- 5A Basic Settings: Target variable, CV folds, variable/complexity penalties, wavelength restriction (custom regions, presets: UV/VIS/NIR/MIR). Preprocessing checkboxes (Raw, SNV, SG1-SG4, deriv_snv). Window sizes (7/11/17/23/31 + custom). Pre-processing steps (baseline correction: polynomial/asls/rubber_band/airpls, smoothing: window/polyorder). Basic preprocessing discovery (importance method, top configs). GA/exhaustive preprocessing (search method, population, generations).
- 5B Variable Selection: Top-N analysis enable, Region analysis enable + depth (Shallow/Medium/Deep/Thorough), test all regions individually, test pairwise, Top-N counts (10/20/50/100/250/500/1000). Methods: Feature Importance, SPA, UVE, UVE-SPA, iPLS, CARS, CARS-Tree, VCPA-IRIV, GA. Method-specific parameters.
- 5C Model Config: Tier radios (Quick/Standard/Comprehensive/Experimental/Custom). Optimization method (Grid/Bayesian/NSGA-II) with trials/pop/gens/selection/mode. Model checkboxes: PLS, PLS-DA, Ridge, Lasso, ElasticNet, RandomForest, MLP, SVR, SVM, XGBoost, LightGBM, CatBoost, NeuralBoosted.
- 5D Ensemble Methods: Ensemble configuration
- 5E Validation: Validation settings and metrics

**Tab 6 — Analysis Progress**
- Animated indicator, best model display, progress info, time estimate, Pause/Resume/Stop buttons, monospace output text (Consolas, colored)

**Tab 7 — Results**
- Treeview with quartile/class highlighting, overfit indicator, header tooltips, double-click loads into Model Dev

**Tab 8 — Model Development (4 subtabs)**
- 8A Selection: Mode control, model config display, Run Model button, Wavelength selection (All/NIR/Visible/Custom + preview)
- 8B Features: (currently stub)
- 8C Configuration: Full parameter editing
- 8D Results & Diagnostics: Visualization and statistics

**Tab 9 — Model Prediction (2 subtabs)**
- 9A Setup: Load Model File(s), Clear All, data source (Directory/CSV-Excel/Validation Set), Browse, Load Data, Run All Models, progress bar
- 9B Results: Prediction results and statistics

**Tab 10 — Multi-Model**
- Multi-model comparison interface

**Tab 11 — Calibration Transfer**
- DS, PDS, TSR, CTAI, NSPFCE, JYPLS-INV methods. Reflectance-absorbance conversion. Region of interest selection. Save/load transfer models.

**Tab 12 — Interference Removal (4 subtabs)**
- 12A Library Management
- 12B Method Configuration (OSC, EPO, GLSW, DOSC, WavelengthExcluder)
- 12C Application
- 12D Diagnostics

**Tab 13 — Spectral Library (2 subtabs)**
- 13A Library Management (add/remove/clear, batch add, categories, export)
- 13B Similarity Search (HQI, SAM, euclidean, cosine, 1st/2nd deriv correlation, SID)

**Tab 14 — Contaminant Analysis (4 subtabs)**
- 14A Load & Define Groups
- 14B Difference Analysis
- 14C Automated Detection (EstimatedEPO, OPLS-DA, GLSW, multi-contaminant)
- 14D Apply & Validate

### 5.3 Theming

- qt-material package for instant modern look
- Dark or light theme selectable
- One-line application: `apply_stylesheet(app, theme='dark_teal.xml')`
- Can customize or replace later without touching tab code

### 5.4 State Management

Central `AppState` dataclass replaces 100+ `self.` variables:

```python
@dataclass
class AppState:
    data: SpectralData | None          # Current working data
    data_original: SpectralData | None # Before wavelength filtering
    sources: list[DataSource]          # Multi-source management
    search_config: SearchConfig | None
    search_results: pd.DataFrame | None
    selected_model: dict | None        # Model loaded into dev tab
    loaded_models: list[dict]          # Models loaded for prediction
    transfer_models: list[TransferModel]
    library: SpectralLibrary | None
    is_searching: bool
    excluded_samples: set[str]
```

Tabs read from and write to this state. Qt signals notify other tabs when state changes.

---

## 6. Implementation Approach

### Phase 1: Skeleton
- Create repo at `C:\Users\sponheim\git\asp`
- Set up pyproject.toml, directory structure, dependencies
- Create core/ module (types, config, exceptions, constants)

### Phase 2: Backend port (module by module)
- Port each backend module from dasp, splitting god files as specified
- Work from actual source code, NOT old documentation
- Each module gets basic tests

### Phase 3: GUI shell
- PySide6 app with sidebar navigation and empty tab frames
- qt-material theme applied
- Shared widgets built (spectra_plot, data_table, collapsible, card)

### Phase 4: Wire tabs (one at a time)
- Start with Import tab (most foundational — data must load first)
- Then Explore, Data Viewer, Analysis Config, Progress, Results
- Then Model Dev, Prediction, Multi-Model
- Then advanced tabs: Cal Transfer, Interference, Library, Contaminant
- Data Management tab last (depends on everything else working)

### Phase 5: Test and validate
- Feature-by-feature comparison against current GUI
- Same data in both apps should produce same results

---

## 7. Dependencies

```
PySide6
qt-material
numpy
pandas
scipy
scikit-learn
matplotlib
openpyxl
xlsxwriter
optuna
pymoo
imbalanced-learn
lightgbm
xgboost
catboost (optional)
torch (optional, for learned preprocessing)
```

---

## 8. What's NOT Changing

- The science (algorithms, math, statistical methods)
- The search architecture (preprocess-first, then models x subsets)
- The .dasp model file format
- The code generation templates (Python/R/notebook export)
- File format support (every reader retained)
- Every user-facing option and control

## 9. What IS Changing

- 45K-line monolith GUI -> 15 tab files + shared widgets
- God backend files (3000-4000 lines) -> focused modules (<500 lines)
- Tkinter -> PySide6 (Qt) with qt-material theme
- 100+ self.variables -> centralized AppState
- Scrolling limitations -> native Qt scroll areas, splitters, collapsible sections
- Baseline methods (rubber band, AirPLS) added as analysis preprocessing options
