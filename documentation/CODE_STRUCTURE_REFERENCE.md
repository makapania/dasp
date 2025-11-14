# Code Structure Reference - Quick Navigation

## File Organization

```
/home/user/dasp/
├── spectral_predict_gui_optimized.py    [18,000 lines] - Main GUI application
│   ├── Tab 1: Import & Preview          - Data loading + spectral visualization
│   ├── Tab 2: Data Viewer               - Excel-like spreadsheet view
│   ├── Tab 3: Data Quality Check        - Outlier detection
│   ├── Tab 4: Analysis Configuration    - Model setup
│   ├── Tab 5: Analysis Progress         - Live progress monitor
│   ├── Tab 6: Results                   - Ranked models table
│   ├── Tab 7: Model Development         - Interactive refinement
│   ├── Tab 8: Model Prediction          - Apply saved models
│   └── Tab 10: Calibration Transfer     - Inter-instrument alignment
│
└── src/spectral_predict/
    ├── SPECTRA LOADING & I/O
    ├──────────────────────────
    │   ├── io.py                        [3,200 lines]
    │   │   ├── read_csv_spectra()       - CSV wide/long format
    │   │   ├── read_reference_csv()     - Target variable CSV
    │   │   ├── align_xy()               - Match spectra to targets
    │   │   ├── read_asd_dir()           - Load ASD files
    │   │   ├── read_spc_dir()           - Load SPC files
    │   │   ├── read_combined_csv()      - Single-file format
    │   │   ├── detect_spectral_data_type() - Reflectance vs Absorbance
    │   │   └── [Write functions for CSV, Excel, JCAMP, SPC]
    │   │
    │   └── readers/
    │       ├── opus_reader.py           - Bruker OPUS (.0, .1, etc.)
    │       ├── perkinelmer_reader.py    - PerkinElmer (.sp)
    │       ├── agilent_reader.py        - Agilent files
    │       ├── asd_native.py            - ASD ASCII native
    │       └── asd_r_bridge.py          - Binary ASD via specdal
    │
    ├── SPECTRA PREPROCESSING & OPERATIONS
    ├──────────────────────────────────────
    │   ├── preprocess.py
    │   │   ├── SNV class                - Standard Normal Variate
    │   │   ├── SavgolDerivative class   - Smooth differentiation
    │   │   └── build_preprocessing_pipeline()
    │   │
    │   ├── calibration_transfer.py      [1,500 lines]
    │   │   ├── TransferModel class      - Encapsulates transfer mapping
    │   │   ├── resample_to_grid()       - Wavelength interpolation
    │   │   ├── estimate_ds()            - Direct Standardization
    │   │   ├── estimate_pds()           - Piecewise DS
    │   │   ├── estimate_tsr()           - Transfer Sample Regression
    │   │   ├── estimate_ctai()          - Affine Invariance
    │   │   ├── estimate_nspfce()        - Nonlinear correction
    │   │   ├── estimate_jypls_inv()     - Joint inversion
    │   │   └── [Apply functions for each method]
    │   │
    │   ├── equalization.py              [Skeleton framework]
    │   │   ├── choose_common_grid()     - Multi-instrument harmonization
    │   │   └── build_equalization_mapping_for_instrument()
    │   │
    │   ├── regions.py
    │   │   ├── compute_region_correlations() - Divide spectrum into regions
    │   │   ├── get_top_regions()        - Top N by correlation
    │   │   └── create_region_subsets()  - Extract region wavelengths
    │   │
    │   ├── wavelength_selection.py      [Advanced wavelength subsetting]
    │   │
    │   ├── variable_selection.py        [Multiple algorithms]
    │   │   ├── SPA                      - Successive Projections Algorithm
    │   │   ├── UVE                      - Uninformative Variable Elimination
    │   │   ├── UVE-SPA Hybrid           - Combined method
    │   │   └── iPLS                     - Interval PLS
    │   │
    │   └── sample_selection.py
    │       ├── kennard_stone()          - Diversity-based selection
    │       ├── duplex()                 - Cal/val split
    │       ├── spxy()                   - Joint X-Y diversity
    │       └── random_selection()
    │
    ├── SPECTRA ANALYSIS & VISUALIZATION
    ├──────────────────────────────────────
    │   ├── interactive_gui.py           [15,000+ lines]
    │   │   ├── InteractiveLoadingGUI class
    │   │   │   ├── _create_raw_spectra_tab()      - Line plot visualization
    │   │   │   ├── _create_derivative1_tab()      - 1st deriv plot
    │   │   │   ├── _create_derivative2_tab()      - 2nd deriv plot
    │   │   │   └── _create_screening_tab()        - Correlation screening
    │   │   └── run_interactive_loading_gui()
    │   │
    │   ├── interactive.py
    │   │   ├── plot_spectra_overview()  - Generate PNG plots
    │   │   ├── show_data_preview()      - ASCII table
    │   │   ├── compute_predictor_screening()  - Correlation ranking
    │   │   └── reflectance_to_absorbance()
    │   │
    │   ├── ensemble_viz.py              [12,000 lines]
    │   │   ├── plot_regional_performance() - Heatmap by region
    │   │   ├── plot_ensemble_weights()  - Bar charts
    │   │   └── [Various diagnostic plots]
    │   │
    │   ├── diagnostics.py
    │   │   ├── compute_residuals()      - y_true - y_pred
    │   │   ├── compute_leverage()       - Hat values
    │   │   ├── qq_plot_data()           - Normality check
    │   │   └── jackknife_prediction_intervals()
    │   │
    │   └── outlier_detection.py         [20,000+ lines]
    │       ├── run_pca_outlier_detection() - PCA T² statistics
    │       ├── compute_q_residuals()    - SPE distances
    │       ├── compute_mahalanobis_distance()
    │       ├── check_y_data_consistency() - Target outliers
    │       └── generate_outlier_report() - Comprehensive report
    │
    ├── MODEL BUILDING & ENSEMBLE
    ├────────────────────────────
    │   ├── search.py                    [50,000+ lines]
    │   │   ├── run_search()             - Main grid search engine
    │   │   ├── [Model × Preprocessing × Variable selection combinations]
    │   │   └── Cross-validation loop
    │   │
    │   ├── models.py                    [78,000+ lines]
    │   │   ├── Model implementations    - PLS, RF, MLP, etc.
    │   │   └── Classification variants
    │   │
    │   ├── neural_boosted.py            [35,000+ lines]
    │   │   └── NeuralBoosted class      - Ensemble model
    │   │
    │   ├── ensemble.py                  [19,000+ lines]
    │   │   ├── EnsembleRegressor        - Weighted ensemble
    │   │   └── EnsembleClassifier
    │   │
    │   ├── model_registry.py            - Model availability
    │   ├── model_config.py              - Model hyperparameters
    │   ├── model_io.py                  - Save/load .dasp models
    │   ├── instrument_profiles.py       - Instrument characterization
    │   ├── scoring.py                   - Ranking functions
    │   └── progress_monitor.py          - Progress tracking
    │
    └── UTILITIES
        ├── report.py                    - Markdown report generation
        └── cli.py                       - Command-line interface
```

## Key Module Functions by Purpose

### 1. DATA LOADING & ALIGNMENT
```python
# Load spectra
X, metadata = read_asd_dir('/path/to/asd/files')
# Load targets
ref = read_reference_csv('reference.csv', 'sample_id_column')
# Align
X_aligned, y_aligned = align_xy(X, ref, 'sample_id_column', 'target_column')
```

### 2. VISUALIZATION
```python
# Interactive preview
run_interactive_loading_gui(X, y, 'sample_id', 'target_name')

# Generate static plots
plot_spectra_overview(X, output_dir='plots/')
plot_predictor_screening(results, output_dir='plots/')

# GUI-based (in spectral_predict_gui_optimized.py)
# - _generate_plots() for spectral tabs
# - _plot_pca_scores() for outlier visualization
# - _plot_regression_predictions() for model results
```

### 3. PREPROCESSING
```python
# Build pipeline
steps = build_preprocessing_pipeline('snv_deriv', deriv=1, window=7)
pipe = Pipeline(steps)
X_processed = pipe.transform(X)

# Or individual transformers
snv = SNV()
X_snv = snv.transform(X)

deriv = SavgolDerivative(deriv=1, window=7)
X_deriv = deriv.transform(X)
```

### 4. CALIBRATION TRANSFER
```python
# Estimate transfer model
A = estimate_ds(X_master, X_slave_paired)
transfer_model = TransferModel(
    master_id='master_1',
    slave_id='slave_1',
    method='ds',
    wavelengths_common=common_wl,
    params={'A': A}
)

# Apply to new data
X_slave_new_transferred = apply_ds(X_slave_new, A)

# Resample to common grid
X_resampled = resample_to_grid(X, wl_src, wl_target)
```

### 5. OUTLIER DETECTION
```python
# Run comprehensive detection
report = generate_outlier_report(X, y, n_pca_components=5)

# Individual methods
pca_results = run_pca_outlier_detection(X, y, n_components=5)
q_resid = compute_q_residuals(X, pca_results['pca_model'])
mahal = compute_mahalanobis_distance(pca_results['scores'])
```

### 6. VARIABLE SELECTION
```python
# Select top wavelengths
selected_indices = spa(X, y, n_features=20)
selected_indices = uve(X, y, n_features=20)
selected_indices = ipls(X, y, n_intervals=10)
```

### 7. SAMPLE SELECTION
```python
# For calibration transfer
selected_idx = kennard_stone(X, n_samples=20)
selected_idx = spxy(X, y, n_samples=20)
```

### 8. SPECTRAL REGION ANALYSIS
```python
# Divide into regions
regions = compute_region_correlations(X, y, wavelengths, region_size=50)
# Get top regions
top_regions = get_top_regions(regions, n_top=5)
```

## Dependencies by Module

| Module | Key Dependencies |
|--------|------------------|
| io.py | pandas, numpy, scipy |
| preprocess.py | numpy, scipy.signal |
| calibration_transfer.py | numpy, scipy.interpolate, scipy.linalg |
| models.py | scikit-learn, numpy, pandas |
| neural_boosted.py | scikit-learn, numpy |
| outlier_detection.py | scikit-learn, numpy, scipy.stats |
| interactive_gui.py | matplotlib, tkinter |
| spectral_predict_gui_optimized.py | tkinter, matplotlib, tksheet, PIL |
| ensemble.py | numpy, scikit-learn |

## Performance Characteristics

| Operation | Complexity | Dataset Size Tested |
|-----------|-----------|-------------------|
| Load ASD files | O(n×p) | 50-100 files |
| PCA outlier detection | O(n×p²) or O(p³) | n<1000, p<5000 |
| Kennard-Stone selection | O(n²×p) | n<5000 |
| Calibration transfer DS | O(n×p²) or O(p³) | Multiple instruments |
| Neural Boosted search | O(n_iter × n_cv × complexity) | 5-100 sample configs |

## Key Data Structures

### Spectral Data
```python
X = pd.DataFrame(
    data=[[...], [...], ...],  # Spectral values
    columns=[400.5, 401.2, ..., 2495.3],  # Wavelengths as floats
    index=['sample_1', 'sample_2', ...]  # Sample IDs
)
```

### Target Data
```python
y = pd.Series(
    data=[12.5, 15.3, ...],  # Property values
    index=['sample_1', 'sample_2', ...],  # Aligned sample IDs
    name='%Collagen'
)
```

### Calibration Transfer Model
```python
transfer_model = TransferModel(
    master_id='master_1',
    slave_id='slave_1',
    method='ds',  # or 'pds', 'tsr', 'ctai', 'nspfce', 'jypls-inv'
    wavelengths_common=np.array([400.5, 401.2, ..., 2495.3]),
    params={'A': matrix_or_params_dict},
    meta={'resolution_ratio': 1.2, 'rmse': 0.005}
)
```

### Outlier Report
```python
report = {
    'pca_outliers': [idx1, idx2, ...],
    'leverage_outliers': [idx3, idx4, ...],
    'y_outliers': [idx5, ...],
    'all_outliers': {idx: [methods_detecting_it]},
    'excluded_samples': [final list],
    'n_excluded': int
}
```

---

## GUI Tab Structure in spectral_predict_gui_optimized.py

### Tab 1: Import & Preview (Lines ~1711-1900)
- Subtab 1A: Data loading
  - Directory/file selection
  - Format detection
  - Reference file matching
  
- Subtab 1B: Spectral plots
  - Dynamic matplotlib canvas
  - Raw, 1st deriv, 2nd deriv tabs
  - Click-to-exclude samples

### Tab 2: Data Viewer (Lines ~1922-1990)
- tksheet-based spreadsheet
- Virtual scrolling for large datasets

### Tab 3: Data Quality Check (Lines ~1992-2150)
- PCA T², Q-residuals, Mahalanobis scatter plots
- Y-value distribution histogram
- Sample exclusion checkboxes

### Tab 4: Analysis Configuration (Lines ~2149-3618)
- 4A: Basic settings (preprocessing, models)
- 4B: Variable selection method
- 4C: Model hyperparameters
- 4D: Ensemble methods
- 4E: Cross-validation settings

### Tab 5: Progress (Lines ~3618-3671)
- Live progress bar
- Iteration counter
- Status messages

### Tab 6: Results (Lines ~3671-3779)
- Sortable results table
- Model metrics (RMSE, R², Accuracy, AUC)
- Click row to refine model

### Tab 7: Model Development (Lines ~3779-4250)
- 7A: Load model from results
- 7B: Feature engineering
- 7C: Hyperparameter tuning
- 7D: Diagnostics (residuals, leverage, predictions)

### Tab 8: Model Prediction (Lines ~4250-13000)
- Load .dasp model files
- Specify new spectral data
- Make predictions
- Export results

### Tab 10: Calibration Transfer (Lines ~13000-17300)
- Master/slave spectra pairing
- Method selection (DS, PDS, TSR, CTAI, NSPFCE, JYPLS-inv)
- Transfer quality visualization
- Apply to new data

