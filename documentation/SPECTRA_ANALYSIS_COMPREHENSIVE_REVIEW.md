# Spectral Predict - Spectra Viewing & Handling Analysis

## 1. VISUALIZATION CAPABILITIES

### Current Visualization Methods
**Location:** `spectral_predict_gui_optimized.py` (main GUI, 18,000+ lines)

#### 1.1 Interactive Spectral Plots
- **Raw Spectra Tab**: Plots all spectra as overlaid line charts with transparency
- **1st Derivative Tab**: Savitzky-Golay 1st derivative visualization
- **2nd Derivative Tab**: Savitzky-Golay 2nd derivative visualization
- **Features:**
  - Click-to-toggle individual spectra (only visible in raw spectra tab)
  - Automatic sampling for large datasets (>50 spectra)
  - Adjustable alpha transparency
  - Grid overlay with zero-line reference for derivatives
  - matplotlib + tkinter with FigureCanvasTkAgg backend

#### 1.2 Interactive Loading GUI
**Location:** `src/spectral_predict/interactive_gui.py`
- Tab-based interface with matplotlib embeddings:
  - Data preview (treeview table)
  - Raw spectra plot
  - 1st derivative plot
  - 2nd derivative plot
  - Screening results plot
- Real-time absorbance conversion capability

#### 1.3 Advanced Diagnostics Plots
**Location:** `spectral_predict_gui_optimized.py` (Tab 5-7)
- **Outlier Detection Visualizations:**
  - Hotelling T² scores (scatter plot)
  - Q-residuals / SPE (scatter plot)
  - Mahalanobis distance (scatter plot)
  - Y-value distribution (histogram/boxplot)
  - Sample exclusion marking on spectra

- **Prediction Results Plots:**
  - Predicted vs. Actual scatter plot (regression)
  - Prediction residuals plot
  - Residual diagnostics (Q-Q plot, residual distribution)
  - Leverage plot (hat values)
  - Classification confusion matrix (heatmap)
  - Multi-class ROC curves (One-vs-Rest)
  - Prediction confidence intervals

- **Model Performance Plots:**
  - PCA scores (2D/3D scatter)
  - Feature importance (bar charts)
  - Regional performance heatmaps (model specialization)

#### 1.4 Static Plot Generation
**Location:** `src/spectral_predict/interactive.py`
- `plot_spectra_overview()`: Generates PNG files for:
  - Raw spectra
  - 1st derivative
  - 2nd derivative
- `plot_predictor_screening()`: Correlation screening visualization

### Visualization Limitations
- No 3D spectral surface plots
- No heatmap visualization of spectral matrix
- No contour/density plots for spectral populations
- Limited spectra comparison tools
- No animated/time-series spectra visualization
- No spectral fingerprinting/matching visualization

---

## 2. SPECTRA ARCHITECTURE & DATA STRUCTURES

### Core Data Representation
**Data Format:** pandas.DataFrame (wide format)
- **Rows:** Sample IDs (specimen/spectrum identifiers)
- **Columns:** Float wavelengths (nm) - column names are wavelengths
- **Values:** Spectral intensities (reflectance, absorbance, transmittance)

### Example:
```
             400.5  401.2  402.1  ...  2495.3  2496.1
sample_1   0.450  0.455  0.460  ...   0.125   0.128
sample_2   0.380  0.385  0.390  ...   0.105   0.108
sample_3   0.520  0.525  0.530  ...   0.145   0.148
```

### Key Design Principles
1. **Wide Format Only**: No native support for long format spectra representation
2. **Single-Spectrum per Row**: Each row = one measurement
3. **Immutable Wavelength Grid**: Once loaded, wavelengths are fixed
4. **Metadata Sparse**: Limited metadata capture (data_type, wavelength_range)

### Associated Data
- **Target Variable (y):** pd.Series indexed by sample ID
- **Reference Data:** Separate DataFrame for targets/properties
- **Sample IDs:** DataFrame index (can be filenames, numerical IDs, etc.)

---

## 3. FILE FORMAT & LOADING SUPPORT

### Supported Formats
**Location:** `src/spectral_predict/io.py` (3,200+ lines) & `src/spectral_predict/readers/`

| Format | Extensions | Read | Write | Native Support |
|--------|-----------|------|-------|-----------------|
| **CSV** | .csv | ✅ | ✅ | Built-in pandas |
| **Excel** | .xlsx, .xls | ✅ | ✅ | openpyxl |
| **ASD (ASCII)** | .asd, .sig | ✅ | ❌ | Built-in text parsing |
| **ASD (Binary)** | .asd | ✅ | ❌ | specdal (optional) |
| **SPC** | .spc | ✅ | ✅ | spc-io |
| **JCAMP-DX** | .jdx, .dx, .jcm | ✅ | ✅ | jcamp |
| **ASCII Text** | .txt, .dat | ✅ | ✅ | Built-in |
| **Bruker OPUS** | .0, .1, .2, etc. | ✅ | ❌ | brukeropus (optional) |
| **PerkinElmer** | .sp | ✅ | ❌ | specio (optional) |
| **Agilent** | .seq, .dmt, .asp, .bsw | ✅ | ❌ | agilent-ir-formats (optional) |

### Smart Data Detection
- **Auto-detect format** from file/directory structure
- **Data type detection**: Reflectance vs Absorbance (ML-based, ~80-90% confidence)
- **Column inference**: Automatic wavelength and specimen ID detection
- **Flexible matching**: Handles files with/without extensions, spaces, case variations

### Combined Format Support
- **Single-file datasets** with specimen ID + wavelengths + targets in one CSV/Excel
- **Multi-file datasets** with separate spectra directory + reference file
- **Fuzzy ID matching** for aligning spectra to reference data

---

## 4. OPERATIONS ON SPECTRA (EXISTING)

### 4.1 Preprocessing Transformers
**Location:** `src/spectral_predict/preprocess.py`

#### SNV (Standard Normal Variate)
```python
def transform(X):
    X_snv = (X - X.mean(axis=1)) / X.std(axis=1)
    return X_snv
```
- Per-spectrum normalization

#### Savitzky-Golay Derivatives
```python
def transform(X, deriv=1, window=7, polyorder=2):
    # 1st derivative (default polyorder=2)
    # 2nd derivative (default polyorder=3)
    return savgol_filter(X, window, polyorder, deriv=deriv, axis=1)
```
- Smooth differentiation along wavelength axis
- Configurable window size and polynomial order

#### Pipeline Construction
- Supports chains: `raw`, `snv`, `deriv`, `snv_deriv`, `deriv_snv`

### 4.2 Wavelength-Based Operations
**Location:** `src/spectral_predict/regions.py`

#### Region Analysis
```python
def compute_region_correlations(X, y, wavelengths, region_size=50, overlap=25):
    # Divide spectrum into overlapping regions
    # Compute correlation with target for each region
    # Return top regions by mean/max correlation
```

#### Region Subsets
- Extract top-N wavelength regions
- Filter spectra to specific wavelength ranges
- Regional performance analysis

### 4.3 Sample Selection Operations
**Location:** `src/spectral_predict/sample_selection.py`

#### Kennard-Stone Algorithm
- Select maximally diverse spectra from dataset
- O(n²) complexity

#### DUPLEX Algorithm
- Split spectra into calibration/validation sets
- Maintains diversity in both sets

#### SPXY Algorithm
- Joint X-Y space diversity (considers both spectra and targets)

#### Random Selection
- Baseline for comparison

### 4.4 Variable Selection (Wavelength Selection)
**Location:** `src/spectral_predict/variable_selection.py`

#### Methods:
1. **SPA** (Successive Projections Algorithm)
   - Minimizes collinearity among wavelengths
   
2. **UVE** (Uninformative Variable Elimination)
   - Filters noise-like wavelengths
   
3. **UVE-SPA Hybrid**
   - Combines both methods
   
4. **iPLS** (Interval PLS)
   - Region-based selection across spectrum

### 4.5 Outlier Detection & Filtering
**Location:** `src/spectral_predict/outlier_detection.py`

#### Methods:
- **PCA-based Hotelling T²**: Multivariate distance in PC space
- **Q-Residuals (SPE)**: Reconstruction error from PCA
- **Mahalanobis Distance**: Covariance-weighted distance
- **Y-value Checks**: Statistical outliers in target variable
- **Sample Exclusion**: Mark/exclude samples in analysis

### 4.6 Calibration Transfer (Inter-Instrument)
**Location:** `src/spectral_predict/calibration_transfer.py` (1,500+ lines)

#### Methods:
1. **DS** (Direct Standardization)
   - Linear matrix transformation: X_slave @ A = X_master
   
2. **PDS** (Piecewise Direct Standardization)
   - Windowed DS transformation along wavelengths
   
3. **TSR** (Transfer Sample Regression / Shenk-Westerhaus)
   - Calibration transfer using reference samples
   
4. **CTAI** (Calibration Transfer based on Affine Invariance)
   - Robust method for temperature/humidity drift
   
5. **NSPFCE** (Nonlinear Standardization for Particle Size and Field-effect Compensation)
   - Handles nonlinear instrument differences
   
6. **JYPLS-inv** (Joint Inversion for PLS)
   - Advanced PLS-based transfer

#### Resampling
```python
def resample_to_grid(X, wl_src, wl_target):
    # 1D linear interpolation to common wavelength grid
    # Handles wavelength misalignment between instruments
```

### 4.7 Instrument Equalization
**Location:** `src/spectral_predict/equalization.py`

- **Multi-instrument harmonization**: Combine spectra from multiple instruments
- **Common wavelength grid selection**: Robust algorithm for overlapping ranges
- **Smooth resolution compensation**: Account for different instrument resolutions

### 4.8 Diagnostic Statistics
**Location:** `src/spectral_predict/diagnostics.py`

- **Leverage computation**: Hat values (high-leverage samples)
- **Residual analysis**: Standardized residuals
- **Prediction intervals**: Jackknife-based confidence intervals
- **Q-Q plots**: Normality diagnostics

### 4.9 Spectral Screening
**Location:** `src/spectral_predict/interactive.py`

```python
def compute_predictor_screening(X, y, n_top=20):
    # Compute correlation of each wavelength with target
    # Rank by absolute correlation
    # Return top N wavelengths
```

---

## 5. MISSING FEATURES FOR ADVANCED SPECTRA ANALYSIS

### 5.1 Spectra Grouping & Classification
**Not Implemented:**
- ❌ Spectra clustering (K-means, hierarchical, DBSCAN on spectra)
- ❌ Spectra similarity/distance calculations (Euclidean, DTW, Spectral Angle Mapper)
- ❌ Spectral fingerprinting/identification
- ❌ Spectral library/reference matching
- ❌ Spectra grouping by user-defined categories
- ❌ Spectra KNN/similarity search
- ❌ Sample group averaging/mean spectra computation
- ❌ Sample group statistics (std, quantiles per group)

### 5.2 Spectral Operations & Arithmetic
**Not Implemented:**
- ❌ Spectral addition/subtraction (e.g., for baseline correction)
- ❌ Spectral multiplication/division (scaling)
- ❌ Spectral blending/interpolation between spectra
- ❌ Mean/median spectra across groups
- ❌ Spectral difference maps
- ❌ Spectral weighted combinations

### 5.3 Advanced Visualization
**Not Implemented:**
- ❌ 3D spectral surface plots
- ❌ Spectral heatmap/matrix visualization (wavelength × sample)
- ❌ Contour plots of spectral intensity
- ❌ Spectral density plots
- ❌ Waterfall plots (sample-by-sample stacked spectra)
- ❌ Interactive zoom/pan with linked axes
- ❌ Spectral overlay comparison with difference highlighting
- ❌ Animated spectral transitions
- ❌ Spectral movies (time-series visualization)
- ❌ Peak labeling/annotation on spectra
- ❌ Interactive peak picking/marking

### 5.4 Spectral Peak/Feature Analysis
**Not Implemented:**
- ❌ Automatic peak detection
- ❌ Peak picking and labeling
- ❌ Peak wavelength tracking across samples
- ❌ Peak area/height extraction
- ❌ Peak resolution metrics
- ❌ Spectral inflection point detection
- ❌ Band analysis (absorption bands, transmission windows)
- ❌ Peak fitting (Lorentzian, Gaussian, Voigt)

### 5.5 Spectral Similarity & Matching
**Not Implemented:**
- ❌ Spectral similarity metrics (SAM, SID, etc.)
- ❌ Spectral library search/matching
- ❌ Spectral angle mapper (SAM)
- ❌ Spectral information divergence (SID)
- ❌ Spectral correlation mapper
- ❌ Spectral feature matching algorithms
- ❌ Spectral unmixing (endmember extraction, linear mixing model)

### 5.6 Batch Operations on Spectra Collections
**Not Implemented:**
- ❌ Batch resample across multiple spectra
- ❌ Batch preprocessing with tracking
- ❌ Batch spectral smoothing with adaptive parameters
- ❌ Multi-spectrum baseline correction
- ❌ Spectral co-registration across collections
- ❌ Spectral mosaic/composite generation

### 5.7 Data Transformation & Representation
**Not Implemented:**
- ❌ Long format spectra support
- ❌ Spectral matrix export (wavelength × sample format)
- ❌ HDF5/NetCDF format support
- ❌ Spectral data compression/encoding
- ❌ Spectral time-series handling
- ❌ Multi-beam/multi-channel spectra support
- ❌ Wavelength alignment/co-registration between datasets

### 5.8 Interactive Spectral Analysis
**Not Implemented:**
- ❌ Real-time spectral transformations (user sees changes immediately)
- ❌ Interactive baseline subtraction tool
- ❌ Interactive spectral range selection with preview
- ❌ Spectral smoothing slider with live preview
- ❌ Click-to-identify features on spectrum
- ❌ Brush selection of spectral regions
- ❌ Linked brushing across multiple plots

### 5.9 Quality Metrics for Spectra
**Not Implemented:**
- ❌ Signal-to-noise ratio (SNR) per spectrum
- ❌ Spectral smoothness metric
- ❌ Dynamic range metrics
- ❌ Saturation detection
- ❌ Baseline slope/trend detection
- ❌ Spectral quality scores
- ❌ Automated spectral quality flags

### 5.10 Spectral Noise & Artifact Handling
**Not Implemented:**
- ❌ Noise estimation per spectrum
- ❌ Spike removal/cosmic ray correction
- ❌ Spectral smoothing (Savitzky-Golay, median, spline)
- ❌ Wavelet denoising
- ❌ Spectral clipping/normalization by wavelength
- ❌ Baseline correction (polynomial, asymmetric least squares)
- ❌ Spectral background subtraction

### 5.11 Statistical Spectral Analysis
**Not Implemented:**
- ❌ Principal Component Analysis (PCA) on spectral space
- ❌ Independent Component Analysis (ICA)
- ❌ Spectral entropy/complexity metrics
- ❌ Spectral variance analysis by wavelength
- ❌ Spectral skewness/kurtosis
- ❌ Spectral range analysis (min, max, dynamic range)

### 5.12 Export & Reporting for Spectra
**Not Implemented:**
- ❌ Spectral data export with different format options
- ❌ Spectral summary statistics reports
- ❌ Spectral quality control reports
- ❌ Spectral comparison reports
- ❌ Spectral feature extraction reports
- ❌ Custom spectra visualization export (SVG, PDF)

### 5.13 Spectra File Handling
**Not Implemented:**
- ❌ Spectral data compression/archiving
- ❌ Spectral data validation/integrity checking
- ❌ Batch spectral file conversion
- ❌ Spectral metadata preservation
- ❌ Spectral file tagging/categorization
- ❌ Spectral data versioning

---

## 6. ARCHITECTURE SUMMARY

### Strengths
1. ✅ **Robust multi-format loading**: 10+ formats with auto-detection
2. ✅ **Comprehensive preprocessing**: SNV, derivatives with multiple options
3. ✅ **Advanced calibration transfer**: 6 different inter-instrument methods
4. ✅ **Variable selection**: 4 different wavelength selection algorithms
5. ✅ **Outlier detection**: Multiple statistical methods
6. ✅ **Interactive GUI**: 10 tabs with matplotlib embeddings
7. ✅ **Sample selection**: Kennard-Stone, DUPLEX, SPXY algorithms

### Gaps
1. ❌ **Limited spectra comparison**: No similarity metrics or fingerprinting
2. ❌ **No spectral operations**: Can't add/subtract/average spectra
3. ❌ **No clustering**: Can't group similar spectra automatically
4. ❌ **No advanced visualization**: Limited to 2D line plots
5. ❌ **No batch spectra ops**: Limited multi-spectrum operations
6. ❌ **No peak analysis**: No automatic peak detection/extraction
7. ❌ **Limited metadata**: Minimal spectral metadata tracking

### Data Flow
```
Load Spectra (10+ formats)
        ↓
Auto-detect data type & format
        ↓
Align with target variable
        ↓
Outlier detection & exclusion
        ↓
Preprocessing (SNV, derivatives, etc.)
        ↓
Variable selection (SPA, UVE, iPLS)
        ↓
Model building & cross-validation
        ↓
Calibration transfer (inter-instrument)
        ↓
Prediction on new spectra
```

---

## 7. RECOMMENDATIONS FOR ADVANCED SPECTRA ANALYSIS

### High Priority (Most Useful)
1. **Spectra averaging by group** - Essential for sample comparison
2. **Spectral similarity metrics** - For fingerprinting/identification
3. **Automatic peak detection** - Common chemometric need
4. **3D/heatmap visualization** - Better visual understanding
5. **Baseline correction** - Essential preprocessing step

### Medium Priority
6. **Spectral arithmetic** (addition, subtraction, scaling)
7. **Interactive spectral tools** (click to identify features)
8. **Signal quality metrics** (SNR, saturation detection)
9. **Spectral library matching**
10. **Batch processing improvements**

### Nice to Have
11. Spectral unmixing
12. Animated spectral transitions
13. Spectral compression/archiving
14. Advanced noise handling (wavelets, etc.)
15. Spectral time-series support

