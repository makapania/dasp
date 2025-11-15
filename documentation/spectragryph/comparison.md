# SpectralGryph Feature Comparison & Implementation Strategy

## Executive Summary

**SpectralGryph** is a Windows-based optical spectroscopy software focused on **visualization, processing, and exploratory analysis** of spectra. Your current software (DASP) is a **machine learning and calibration transfer platform** optimized for predictive modeling.

**The Gap:** SpectralGryph excels at interactive visualization, spectral arithmetic, and comparative analysis - areas where DASP is currently weak.

**Recommendation:** **Enhance DASP** with SpectralGryph-inspired features rather than wholesale replacement. The two tools serve complementary purposes, and DASP's ML/calibration capabilities are unique and valuable.

---

## Feature Comparison Matrix

| Feature Category | SpectralGryph | DASP (Current) | Priority |
|-----------------|---------------|----------------|----------|
| **File Format Support** | 77 formats | 10+ formats | Medium |
| **Spectral Visualization** | Overlay, stack, zoom, reverse axis, peak labels | Basic 2D line plots, click-toggle | **HIGH** |
| **Grouping & Organization** | Datasets, subsets, visual differentiation | None | **HIGH** |
| **Spectral Arithmetic** | Add, subtract, multiply, divide spectra | None | **HIGH** |
| **Averaging Spectra** | Average selection with rules | None | **HIGH** |
| **Baseline Correction** | Polynomial fit subtraction, selective | None | **HIGH** |
| **Smoothing** | 4 algorithms (MA, SG 2nd-5th order, percentile, baseline-selective) | SG only (1st-2nd derivatives) | Medium |
| **Derivatives** | 1st-4th order with optional smoothing | 1st-2nd order (SG) | Low |
| **Normalization** | Peak, area, value-based | SNV only | Medium |
| **Peak Detection/Analysis** | Automatic peak finding, FWHM, area | None | **HIGH** |
| **Spectral Library Matching** | Database search & matching | None | Medium |
| **Automation/Batch Processing** | Sequence automation, parallel processing | Limited | Medium |
| **Machine Learning** | None | **6 algorithms, AutoML** | - |
| **Calibration Transfer** | None | **6 methods (DS, PDS, TSR, CTAI, etc.)** | - |
| **Wavelength Selection** | None | **4 algorithms (SPA, UVE, iPLS)** | - |
| **Outlier Detection** | None | **PCA Hotelling, Q-residuals, Mahalanobis** | - |

**Legend:**
- **HIGH** = Critical gap to address
- Medium = Nice to have
- Low = Already sufficient or low value
- `-` = DASP already superior

---

## SpectralGryph Key Features (77 Identified)

### 1. File I/O & Formats
- **77 file formats** supported (vs. DASP's 10+)
- Formats: UV/VIS, NIR, FTIR, Raman, LIBS, XRF, fluorescence
- Windows-only software

### 2. Visualization & Display
- **Overlay multiple spectra** with color/style differentiation
- **Stack spectra** vertically for comparison
- **Reverse wavelength axis**
- **Peak wavelength labels** on plots
- **Zoom/pan** interactive tools
- **EEM (Excitation Emission Matrix)** for fluorescence
- Export plots to publication-quality graphics

### 3. Spectral Grouping & Organization
- **Datasets and subsets** for organizing spectra
- **Visual differentiation** by group (colors, line styles)
- **Compare multiple groups** side-by-side
- **Selection rules** for filtering spectra

### 4. Mathematical Operations on Spectra
- **Addition** of spectra
- **Subtraction** of spectra (e.g., background removal)
- **Multiplication** of spectra
- **Division** of spectra
- **Averaging** of selected spectra with rules

### 5. Preprocessing & Processing
- **Baseline Correction:**
  - Polynomial fit subtraction
  - Baseline-selective smoothing (vertical gradient)
- **Smoothing (4 algorithms):**
  - Moving average
  - Savitzky-Golay (interval size, polynomial order 2nd-5th)
  - Percentile filter
  - Baseline-selective option
- **Derivatives:**
  - 1st-4th order
  - Optional internal smoothing
- **Normalization:**
  - Peak normalization (highest peak = 1)
  - Area normalization
  - Value-based normalization

### 6. Peak Analysis
- **Automatic peak detection**
- **Peak fitting**
- **Peak area integration**
- **FWHM (Full Width Half Maximum)** calculation
- **Peak tracking** across samples

### 7. Spectral Library & Database
- **Spectral database search** and matching
- **Library matching** for unknown sample identification
- **Similarity scoring**

### 8. Automation & Batch Processing
- **Processing sequences** (100% reproducible workflows)
- **Parallel processing** of multiple files
- **Automate repetitive tasks**
- Data reduction pipelines

### 9. Other Features
- **Chemometric preprocessing** functions
- **Merging spectra**
- **Data reduction** workflows
- Free for private and academic use
- Windows 7-11 compatible

---

## DASP Strengths (SpectralGryph Doesn't Have)

### 1. Machine Learning & Predictive Modeling
- **6 regression algorithms** with automatic model selection
- **Classification support** (NeuralBoosted, LightGBM, etc.)
- **AutoML** with hyperparameter tuning
- **Cross-validation** and performance metrics

### 2. Calibration Transfer (Inter-Instrument Equalization)
- **6 transfer methods:** DS, PDS, TSR, CTAI, NSPFCE, JYPLS-inv
- **Wavelength resampling** and alignment
- Multi-instrument calibration framework

### 3. Wavelength/Variable Selection
- **SPA, UVE, UVE-SPA, iPLS** algorithms
- **Region-based analysis**
- **Correlation screening**

### 4. Outlier Detection & Quality Control
- **PCA Hotelling T²**
- **Q-residuals**
- **Mahalanobis distance**
- **Y-value statistical tests**
- Multi-method comprehensive reports

### 5. Sample Selection Algorithms
- **Kennard-Stone, DUPLEX, SPXY**
- Diversity sampling for calibration

### 6. Production-Ready Architecture
- **250,000+ lines** of tested Python code
- **Extensive unit tests** (12+ test modules)
- **Real example datasets** (50 ASD files)
- **Well-documented API**

---

## Critical Gaps in DASP (Your Main Concerns)

### 1. **Viewing Different Groups of Spectra Differently** ❌
**Current State:** All spectra plotted with same styling; click-to-toggle only
**SpectralGryph Has:** Group-based colors/styles, datasets/subsets, visual differentiation
**Impact:** **Critical gap** - can't visually compare experimental groups

### 2. **Spectral Arithmetic (Add/Subtract/Average)** ❌
**Current State:** No spectral arithmetic operations
**SpectralGryph Has:** Add, subtract, multiply, divide, average with selection rules
**Impact:** **Critical gap** - can't do baseline subtraction, reference correction, signal averaging

### 3. **Baseline Correction** ❌
**Current State:** No baseline removal (essential preprocessing)
**SpectralGryph Has:** Polynomial fit subtraction, baseline-selective smoothing
**Impact:** **Critical gap** - many spectra need baseline correction before analysis

### 4. **Peak Detection/Analysis** ❌
**Current State:** No automatic peak finding
**SpectralGryph Has:** Automatic detection, fitting, FWHM, area integration
**Impact:** **High** - important for identifying spectral features

### 5. **Interactive Visualization** ⚠️
**Current State:** Basic 2D plots, limited interaction
**SpectralGryph Has:** Zoom, pan, overlay, stack, reverse axis, annotations
**Impact:** **High** - limits exploratory data analysis

---

## Implementation Strategy: Three Approaches

### **Option A: Enhance DASP (Recommended)**
**Approach:** Add SpectralGryph-inspired features to DASP as new modules

**Advantages:**
- ✅ Preserves unique ML/calibration capabilities
- ✅ Builds on 250,000 lines of tested code
- ✅ Leverages existing file I/O and preprocessing
- ✅ Incremental development (low risk)
- ✅ Unified Python ecosystem

**Disadvantages:**
- ⚠️ Requires GUI redesign for visualization
- ⚠️ Development time for new features

**Estimated Effort:** 4-6 weeks for core features

**Priority Features to Add:**
1. **Group-based visualization** (2-3 days)
   - Color/style by metadata groups
   - Interactive legend with group toggle

2. **Spectral arithmetic** (2-3 days)
   - Add, subtract, multiply, divide operations
   - Average by group with std dev bands

3. **Baseline correction** (3-4 days)
   - Polynomial baseline subtraction
   - Asymmetric least squares (ALS)
   - Airpls, SNIP algorithms

4. **Peak detection** (4-5 days)
   - Scipy find_peaks integration
   - FWHM, area, height extraction
   - Peak annotation on plots

5. **Interactive visualization** (5-7 days)
   - Matplotlib interactive mode
   - Zoom, pan, annotation tools
   - Plotly integration for web-like interactivity

---

### **Option B: Separate Companion Tool**
**Approach:** Build a lightweight SpectralGryph-like viewer that works alongside DASP

**Advantages:**
- ✅ Focused on visualization/exploration only
- ✅ Can use modern web tech (Plotly Dash, Streamlit)
- ✅ Doesn't complicate DASP's ML focus
- ✅ Can share file I/O with DASP

**Disadvantages:**
- ⚠️ Context switching between tools
- ⚠️ Duplicate file loading
- ⚠️ Integration complexity

**Estimated Effort:** 3-4 weeks for standalone tool

**Architecture:**
```
DASP (ML/Calibration) ←→ Shared Data ←→ SpectraViewer (Visualization/Exploration)
```

---

### **Option C: Wholesale Replacement (Not Recommended)**
**Approach:** Rewrite DASP from scratch with SpectralGryph-like features + ML

**Advantages:**
- ✅ Clean slate architecture
- ✅ Modern UI/UX design

**Disadvantages:**
- ❌ **Loses 250,000 lines of tested code**
- ❌ **Loses unique calibration transfer capabilities**
- ❌ **6-12 months development time**
- ❌ **High risk, uncertain outcome**
- ❌ Must re-implement ML, outlier detection, variable selection, etc.

**Verdict:** **Not recommended** - DASP's ML/calibration capabilities are too valuable to discard

---

## Recommended Implementation Plan (Option A)

### Phase 1: Critical Visualization Gaps (2 weeks)
**Goal:** Enable group-based viewing and spectral operations

1. **Group-Based Visualization Module** (`src/spectral_predict/visualization.py`)
   - Add `plot_spectra_by_group()` function
   - Color/linestyle mapping by metadata column
   - Interactive legend with group toggle
   - Matplotlib styling enhancements

   ```python
   from spectral_predict.visualization import plot_spectra_by_group

   plot_spectra_by_group(
       X,  # Spectra DataFrame
       groups=metadata['treatment'],  # Group labels
       title="Treatment Comparison",
       colors={'control': 'blue', 'treated': 'red'}
   )
   ```

2. **Spectral Arithmetic Module** (`src/spectral_predict/operations.py`)
   - `add_spectra()`, `subtract_spectra()`, `multiply_spectra()`, `divide_spectra()`
   - `average_spectra_by_group()` with std dev bands
   - `normalize_to_peak()`, `normalize_to_area()`

   ```python
   from spectral_predict.operations import subtract_spectra, average_spectra_by_group

   # Background subtraction
   corrected = subtract_spectra(sample_spectra, background_spectrum)

   # Group averaging
   group_means = average_spectra_by_group(X, groups=metadata['sample_type'])
   ```

3. **GUI Integration** (Update `spectral_predict_gui_optimized.py`)
   - Add "Spectra Viewer" tab with group selector
   - Add "Operations" tab for arithmetic
   - Update existing visualization to use new modules

### Phase 2: Baseline Correction (1 week)
**Goal:** Essential preprocessing for many spectra types

4. **Baseline Correction Module** (`src/spectral_predict/baseline.py`)
   - Polynomial baseline (`fit_polynomial_baseline()`)
   - Asymmetric Least Squares (ALS)
   - Airpls algorithm
   - SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)

   ```python
   from spectral_predict.baseline import baseline_als, baseline_polynomial

   # Polynomial baseline
   X_corrected = baseline_polynomial(X, degree=3)

   # Asymmetric least squares
   X_corrected = baseline_als(X, lambda_=1e5, p=0.001)
   ```

### Phase 3: Peak Analysis (1 week)
**Goal:** Automatic feature extraction

5. **Peak Detection Module** (`src/spectral_predict/peaks.py`)
   - `detect_peaks()` using scipy.signal.find_peaks
   - `calculate_fwhm()` for peak widths
   - `integrate_peak_area()` for quantification
   - `annotate_peaks_on_plot()` for visualization

   ```python
   from spectral_predict.peaks import detect_peaks, annotate_peaks_on_plot

   peaks = detect_peaks(spectrum, prominence=0.1, distance=10)
   # Returns: {wavelength: [500, 650, 800], height: [...], fwhm: [...]}

   fig = plot_spectra(X)
   annotate_peaks_on_plot(fig, peaks)
   ```

### Phase 4: Interactive Visualization (1-2 weeks)
**Goal:** Modern exploratory tools

6. **Plotly Integration** (Optional but recommended)
   - Add Plotly-based interactive viewer
   - Hover tooltips with wavelength/intensity
   - Zoom/pan/box-select
   - Export to HTML for sharing

   ```python
   from spectral_predict.interactive import interactive_spectra_viewer

   # Launch interactive viewer
   fig = interactive_spectra_viewer(X, groups=metadata['treatment'])
   fig.show()  # Opens in browser with full interactivity
   ```

---

## Quick Start: Minimal Viable Enhancement (3-5 days)

If you want the **biggest impact with minimal effort**, implement just these three functions:

### 1. Group-Based Plotting (1 day)
```python
# File: src/spectral_predict/visualization.py

import matplotlib.pyplot as plt
import numpy as np

def plot_spectra_by_group(X, groups, colors=None, title="Spectra by Group"):
    """
    Plot spectra with different colors/styles per group.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra (rows=samples, cols=wavelengths)
    groups : pd.Series or array-like
        Group labels for each sample
    colors : dict, optional
        {group_name: color} mapping
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    wavelengths = X.columns.astype(float)

    unique_groups = groups.unique()
    if colors is None:
        colors = {g: f'C{i}' for i, g in enumerate(unique_groups)}

    for group in unique_groups:
        mask = groups == group
        X_group = X[mask]

        # Plot mean with std band
        mean = X_group.mean()
        std = X_group.std()

        ax.plot(wavelengths, mean, label=group, color=colors[group], linewidth=2)
        ax.fill_between(wavelengths, mean - std, mean + std,
                        alpha=0.2, color=colors[group])

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    return fig
```

### 2. Spectral Arithmetic (1 day)
```python
# File: src/spectral_predict/operations.py

import pandas as pd

def subtract_spectra(X, reference):
    """
    Subtract a reference spectrum from all spectra.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra to correct
    reference : pd.Series or pd.DataFrame (single row)
        Reference spectrum to subtract

    Returns:
    --------
    pd.DataFrame
        Corrected spectra
    """
    if isinstance(reference, pd.DataFrame):
        reference = reference.iloc[0]

    return X.subtract(reference, axis=1)


def average_spectra_by_group(X, groups):
    """
    Compute mean spectrum for each group.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra
    groups : pd.Series or array-like
        Group labels

    Returns:
    --------
    pd.DataFrame
        Mean spectra (rows=groups, cols=wavelengths)
    """
    X_with_groups = X.copy()
    X_with_groups['_group'] = groups

    return X_with_groups.groupby('_group').mean()
```

### 3. Baseline Correction (2 days)
```python
# File: src/spectral_predict/baseline.py

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(X, lambda_=1e5, p=0.001, niter=10):
    """
    Asymmetric Least Squares baseline correction.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra to correct
    lambda_ : float
        Smoothness parameter (larger = smoother baseline)
    p : float
        Asymmetry parameter (0.001 - 0.1)
    niter : int
        Number of iterations

    Returns:
    --------
    pd.DataFrame
        Baseline-corrected spectra
    """
    X_corrected = X.copy()

    for idx, row in X.iterrows():
        y = row.values
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)

        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lambda_ * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)

        X_corrected.loc[idx] = y - z

    return X_corrected
```

**Usage Example:**
```python
# Load data
from spectral_predict.io import read_asd_dir, align_xy
X = read_asd_dir('data/spectra/')
metadata = pd.read_csv('data/metadata.csv')

# Baseline correction
from spectral_predict.baseline import baseline_als
X_corrected = baseline_als(X, lambda_=1e5)

# Plot by group
from spectral_predict.visualization import plot_spectra_by_group
plot_spectra_by_group(X_corrected, groups=metadata['treatment'])
```

---

## File Format Gap Analysis

**SpectralGryph:** 77 formats
**DASP:** 10+ formats

**DASP Currently Supports:**
- ASD (.asd)
- SPC (.spc)
- CSV (.csv)
- Excel (.xlsx, .xls)
- JCAMP-DX (.jdx, .dx)
- OPUS (.0, .1, .2, etc.)
- PerkinElmer (.sp)
- Agilent (.seq, .spa)
- ASCII text (.txt)

**Missing (SpectralGryph Has):**
SpectralGryph likely supports additional vendors and legacy formats. However, DASP already covers the most common formats. **Priority: Low** - current coverage is sufficient for most users.

---

## Technology Recommendations

### For Enhanced Visualization
**Option 1: Plotly** (Recommended)
- Modern, interactive
- Web-based export (shareable HTML)
- Built-in zoom/pan/hover
- Easy integration with pandas

**Option 2: Matplotlib + mplcursors**
- Stays consistent with current stack
- Add interactivity to existing plots
- Lower learning curve

**Option 3: Bokeh**
- Similar to Plotly
- Better for large datasets
- Steeper learning curve

### For GUI Redesign
**Current:** tkinter (functional but dated)

**Upgrade Options:**
1. **PyQt5/PyQt6** - Professional desktop app
2. **Streamlit** - Quick web-based dashboards
3. **Plotly Dash** - Interactive web apps
4. **Keep tkinter** - Just enhance visualization panels

**Recommendation:** Keep tkinter for now, add Plotly for visualization panels. Upgrade GUI later if needed.

---

## Risk Assessment

### Risks of Enhancing DASP (Option A)
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Code complexity increases | High | Medium | Modular design, separate files for new features |
| GUI becomes cluttered | Medium | Medium | Add new tabs, don't modify existing ones |
| Performance degradation | Low | Low | Optimize plotting for large datasets |
| Breaking existing functionality | Low | High | Comprehensive unit tests, version control |

### Risks of Wholesale Replacement (Option C)
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Project never completes | High | **Critical** | Don't pursue this option |
| Lose calibration transfer | Certain | **Critical** | N/A |
| Users reject new version | High | High | N/A |

---

## Conclusion & Next Steps

### **Recommended Approach: Enhance DASP (Option A)**

**Rationale:**
1. DASP has **unique, valuable capabilities** (calibration transfer, AutoML) that SpectralGryph lacks
2. SpectralGryph's strengths (visualization, grouping, arithmetic) can be **added incrementally** to DASP
3. **Low risk, high value** approach
4. Preserves 250,000 lines of tested code

### **Immediate Action Plan:**

**Week 1-2: Core Enhancements**
1. Create `src/spectral_predict/visualization.py` with group-based plotting
2. Create `src/spectral_predict/operations.py` with spectral arithmetic
3. Add "Spectra Viewer" tab to GUI

**Week 3: Baseline Correction**
4. Create `src/spectral_predict/baseline.py` with ALS algorithm
5. Add "Baseline Correction" preprocessing option to GUI

**Week 4: Peak Analysis**
6. Create `src/spectral_predict/peaks.py` with detection/annotation
7. Add peak detection to "Spectra Viewer" tab

**Week 5-6: Polish & Testing**
8. Add Plotly interactive viewer (optional)
9. Write unit tests for all new modules
10. Update documentation and examples

### **Success Metrics:**
- ✅ Can view different groups with distinct colors/styles
- ✅ Can average, add, subtract spectra
- ✅ Can perform baseline correction
- ✅ Can detect and annotate peaks
- ✅ Existing ML/calibration features still work
- ✅ Users can complete full workflow: Load → Visualize → Group → Preprocess → Model → Transfer

---

## Appendix: Code Structure for New Modules

```
dasp/
├── src/
│   └── spectral_predict/
│       ├── visualization.py      # NEW: Group-based plotting
│       ├── operations.py          # NEW: Spectral arithmetic
│       ├── baseline.py            # NEW: Baseline correction
│       ├── peaks.py               # NEW: Peak detection
│       └── interactive.py         # NEW: Plotly viewer (optional)
│
├── tests/
│   ├── test_visualization.py     # NEW
│   ├── test_operations.py        # NEW
│   ├── test_baseline.py          # NEW
│   └── test_peaks.py             # NEW
│
├── spectral_predict_gui_optimized.py  # UPDATE: Add new tabs
│
└── documentation/
    ├── SPECTRAGRYPH_COMPARISON.md (this file)
    ├── VISUALIZATION_GUIDE.md     # NEW: Usage examples
    └── SPECTRA_OPERATIONS_GUIDE.md # NEW: Arithmetic guide
```

---

## References

- SpectralGryph Official: https://www.effemm2.de/spectragryph/
- DASP Documentation: `/home/user/dasp/documentation/`
- Baseline Correction Algorithms: Eilers & Boelens (2005), Zhang et al. (2010)
- Peak Detection: scipy.signal.find_peaks documentation

---

**Document Version:** 1.0
**Created:** 2025-11-14
**Status:** Ready for review and implementation planning
