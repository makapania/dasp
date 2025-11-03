# Outlier Detection & Ridge/Lasso Implementation Handoff

**Date:** 2025-11-03
**Project:** DASP - Spectral Analysis System
**Status:** 🎯 Ready for Implementation
**Priority:** High (Quality of Life + Model Quality)

---

## 📋 Executive Summary

This document outlines two major enhancements to the DASP spectral analysis system:

1. **Ridge & Lasso Models**: Add these regularized regression models to the main analysis workflow
2. **Pre-Modeling Outlier Detection**: Implement comprehensive outlier screening before model creation

Both features will significantly improve model quality and user control over the analysis pipeline.

---

## 🎯 Part 1: Add Ridge & Lasso to Main Analysis

### Current State

**Ridge and Lasso are partially implemented:**
- ✅ Model definitions exist in `src/spectral_predict/models.py` (lines 40-44)
- ✅ Available in Custom Model Development tab dropdown (line 703 in GUI)
- ✅ Can be manually tested via the refinement workflow
- ❌ **NOT included in automated hyperparameter search** (`get_model_grids()`)
- ❌ **NO checkboxes in main Analysis Configuration tab**

### Why Add Them?

**Scientific Justification:**
- **Fast computation**: Much faster than RF/MLP/NeuralBoosted
- **Excellent baselines**: Provide comparison points for complex models
- **Handle collinearity**: Ridge especially good for high-dimensional spectral data
- **Regularization**: Built-in protection against overfitting
- **Interpretability**: Linear models with clear coefficients

**Technical Justification:**
- Code already exists - minimal implementation effort
- Already tested in Custom Model Development tab
- Standard sklearn models - well-documented and reliable

### Implementation Tasks

#### Task 1.1: Add GUI Checkboxes
**File:** `spectral_predict_gui_optimized.py`
**Location:** Lines 446-456 (Model Selection section)

**Current code:**
```python
ttk.Checkbutton(models_frame, text="✓ PLS (Partial Least Squares)", variable=self.use_pls).grid(row=0, column=0, sticky=tk.W, pady=5)
ttk.Label(models_frame, text="Linear, fast, interpretable", style='Caption.TLabel').grid(row=0, column=1, sticky=tk.W, padx=15)

ttk.Checkbutton(models_frame, text="✓ Random Forest", variable=self.use_randomforest).grid(row=1, column=0, sticky=tk.W, pady=5)
ttk.Label(models_frame, text="Nonlinear, robust", style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, padx=15)

ttk.Checkbutton(models_frame, text="✓ MLP (Multi-Layer Perceptron)", variable=self.use_mlp).grid(row=2, column=0, sticky=tk.W, pady=5)
ttk.Label(models_frame, text="Deep learning", style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)

ttk.Checkbutton(models_frame, text="✓ Neural Boosted", variable=self.use_neuralboosted).grid(row=3, column=0, sticky=tk.W, pady=5)
ttk.Label(models_frame, text="Gradient boosting with NNs", style='Caption.TLabel').grid(row=3, column=1, sticky=tk.W, padx=15)
```

**Add after PLS (insert between row 0 and 1):**
```python
ttk.Checkbutton(models_frame, text="✓ Ridge Regression", variable=self.use_ridge).grid(row=1, column=0, sticky=tk.W, pady=5)
ttk.Label(models_frame, text="L2 regularized linear, fast baseline", style='Caption.TLabel').grid(row=1, column=1, sticky=tk.W, padx=15)

ttk.Checkbutton(models_frame, text="✓ Lasso Regression", variable=self.use_lasso).grid(row=2, column=0, sticky=tk.W, pady=5)
ttk.Label(models_frame, text="L1 regularized linear, sparse solutions", style='Caption.TLabel').grid(row=2, column=1, sticky=tk.W, padx=15)
```

**Then update row numbers for existing models:**
- Random Forest: row=3
- MLP: row=4
- Neural Boosted: row=5

#### Task 1.2: Initialize Variables
**File:** `spectral_predict_gui_optimized.py`
**Location:** Around line 200-210 (in `__init__` method)

**Add after `self.use_pls = tk.BooleanVar(value=True)`:**
```python
self.use_ridge = tk.BooleanVar(value=False)  # Default off (baseline model)
self.use_lasso = tk.BooleanVar(value=False)  # Default off (baseline model)
```

#### Task 1.3: Update Model Collection Logic
**File:** `spectral_predict_gui_optimized.py`
**Location:** Around line 1060 (`_run_analysis` method)

**Current code:**
```python
selected_models = []
if self.use_pls.get():
    selected_models.append('PLS')
if self.use_randomforest.get():
    selected_models.append('RandomForest')
if self.use_mlp.get():
    selected_models.append('MLP')
if self.use_neuralboosted.get():
    selected_models.append('NeuralBoosted')
```

**Add:**
```python
if self.use_ridge.get():
    selected_models.append('Ridge')
if self.use_lasso.get():
    selected_models.append('Lasso')
```

#### Task 1.4: Implement Hyperparameter Grids
**File:** `src/spectral_predict/models.py`
**Location:** In `get_model_grids()` function, after PLS grid (around line 151)

**Add for regression:**
```python
# Ridge Regression
ridge_configs = []
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge_configs.append(
        (
            Ridge(alpha=alpha, random_state=42),
            {"alpha": alpha}
        )
    )
grids["Ridge"] = ridge_configs

# Lasso Regression
lasso_configs = []
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    lasso_configs.append(
        (
            Lasso(alpha=alpha, random_state=42, max_iter=max_iter),
            {"alpha": alpha}
        )
    )
grids["Lasso"] = lasso_configs
```

**Grid sizes:**
- Ridge: 7 alpha values = 7 configurations per preprocessing method
- Lasso: 6 alpha values = 6 configurations per preprocessing method

**Note:** These are fast models, so testing 6-7 configurations won't significantly impact runtime.

#### Task 1.5: Update Feature Importance Extraction
**File:** `src/spectral_predict/search.py`
**Location:** Around line 627 (feature importance extraction)

**Current code:**
```python
if model_name in ["PLS", "PLS-DA", "RandomForest", "MLP", "NeuralBoosted"]:
```

**Update to:**
```python
if model_name in ["PLS", "PLS-DA", "Ridge", "Lasso", "RandomForest", "MLP", "NeuralBoosted"]:
```

**Then add Ridge/Lasso coefficient extraction:**
```python
elif model_name in ["Ridge", "Lasso"]:
    # Get coefficients (linear models)
    coefs = np.abs(fitted_model.coef_)
    if len(coefs.shape) > 1:
        coefs = coefs[0]  # Handle multi-output case

    # Sort by absolute coefficient value
    importance_indices = np.argsort(coefs)[::-1]

    # Top N most important features
    n_top = min(n_top_vars, len(importance_indices))
    top_indices = importance_indices[:n_top]
    top_vars = [wavelengths[i] for i in top_indices]
```

#### Task 1.6: Testing Checklist

- [ ] Ridge checkbox appears in GUI
- [ ] Lasso checkbox appears in GUI
- [ ] Both default to unchecked
- [ ] Checking Ridge adds it to analysis
- [ ] Checking Lasso adds it to analysis
- [ ] Ridge tests 7 alpha values per preprocessing method
- [ ] Lasso tests 6 alpha values per preprocessing method
- [ ] Results appear in Results tab
- [ ] Feature importance shows top wavelengths
- [ ] Can double-click result and load in Custom Model Development
- [ ] Models run fast (should be fastest of all models)

**Estimated Time:** 2-3 hours

---

## 🔍 Part 2: Pre-Modeling Outlier Detection

### Overview

**Goal:** Implement comprehensive outlier screening **before** model creation to identify samples that may degrade model performance.

**Timing:** After data upload, before running main analysis

**Location:** New tab or section in existing workflow

### Scientific Background

**Why outlier detection matters:**
- Instrument errors (noise, drift, contamination)
- Sample preparation issues
- Mislabeled samples
- Samples outside typical chemical/physical space
- Data entry errors in reference values

**When to remove outliers:**
- ✅ Compelling technical justification (documented)
- ✅ Instrument malfunction confirmed
- ✅ Sample preparation error identified
- ✅ Y value outside chemically reasonable range
- ❌ Never just because model performs better without them

### Proposed UI/UX Design

#### Option A: New "Data Quality" Tab
**Location:** Between "Data Upload" and "Analysis Configuration"

**Benefits:**
- Clear workflow: Upload → QC → Configure → Run
- Dedicated space for diagnostics
- Can revisit QC without reloading data

#### Option B: Button in Data Upload Tab
**Location:** After data is loaded, before switching tabs

**Benefits:**
- Keeps workflow simple (fewer tabs)
- Optional step - users can skip if confident

**Recommendation:** Option A (dedicated tab) - better UX for comprehensive QC

### Implementation Plan

#### Component 2.1: PCA-Based Outlier Detection

**Method:** Principal Component Analysis on spectral data only (X)

**Visualizations:**
1. **Score Plot**: PC1 vs PC2 (scatter plot)
   - Color by Y value
   - Size by Hotelling T²
   - Interactive: hover for sample ID

2. **Hotelling T² Chart**: Bar chart or scatter
   - X-axis: Sample index
   - Y-axis: T² statistic
   - Horizontal line: 95% confidence threshold
   - Flag samples above threshold

3. **Scree Plot**: Variance explained by each PC
   - Helps determine how many PCs needed

**Implementation:**
```python
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats

def run_pca_outlier_detection(X, y=None, n_components=5):
    """
    Perform PCA-based outlier detection on spectral data.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Spectral data (samples × wavelengths)
    y : pd.Series or np.ndarray, optional
        Reference values for overlay
    n_components : int
        Number of principal components to compute

    Returns
    -------
    results : dict
        {
            'pca_model': fitted PCA object,
            'scores': PC scores (samples × n_components),
            'loadings': PC loadings (wavelengths × n_components),
            'variance_explained': variance explained by each PC,
            'hotelling_t2': Hotelling T² for each sample,
            't2_threshold': 95% confidence threshold,
            'outlier_flags': boolean array (True = outlier)
        }
    """
    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    # Compute Hotelling T²
    # T² = score · inv(cov) · score.T
    cov_matrix = np.cov(scores.T)
    inv_cov = np.linalg.inv(cov_matrix)

    t2_values = []
    for score in scores:
        t2 = score @ inv_cov @ score.T
        t2_values.append(t2)

    t2_values = np.array(t2_values)

    # Compute 95% threshold using F-distribution
    n_samples = X.shape[0]
    alpha = 0.05
    t2_threshold = (n_components * (n_samples - 1) / (n_samples - n_components) *
                    stats.f.ppf(1 - alpha, n_components, n_samples - n_components))

    outlier_flags = t2_values > t2_threshold

    return {
        'pca_model': pca,
        'scores': scores,
        'loadings': pca.components_.T,
        'variance_explained': pca.explained_variance_ratio_,
        'hotelling_t2': t2_values,
        't2_threshold': t2_threshold,
        'outlier_flags': outlier_flags,
        'n_outliers': np.sum(outlier_flags),
        'outlier_indices': np.where(outlier_flags)[0]
    }
```

#### Component 2.2: Q-Residuals (DModX)

**Method:** Distance of each sample from PCA model

**Visualization:** Q-residuals chart
- X-axis: Sample index
- Y-axis: Q-residual value
- Horizontal line: 95th percentile threshold

**Implementation:**
```python
def compute_q_residuals(X, pca_model, n_components=None):
    """
    Compute Q-residuals (SPE - Squared Prediction Error) for outlier detection.

    Parameters
    ----------
    X : np.ndarray
        Original data
    pca_model : PCA object
        Fitted PCA model
    n_components : int, optional
        Number of components to use. If None, uses all from model.

    Returns
    -------
    results : dict
        {
            'q_residuals': Q-residual for each sample,
            'q_threshold': 95th percentile threshold,
            'outlier_flags': boolean array,
            'n_outliers': count,
            'outlier_indices': array of indices
        }
    """
    if n_components is None:
        n_components = pca_model.n_components_

    # Project data to PC space and back
    scores = pca_model.transform(X)[:, :n_components]
    X_reconstructed = scores @ pca_model.components_[:n_components, :]

    # Compute reconstruction error
    residuals = X - X_reconstructed
    q_residuals = np.sum(residuals ** 2, axis=1)

    # 95th percentile threshold
    q_threshold = np.percentile(q_residuals, 95)

    outlier_flags = q_residuals > q_threshold

    return {
        'q_residuals': q_residuals,
        'q_threshold': q_threshold,
        'outlier_flags': outlier_flags,
        'n_outliers': np.sum(outlier_flags),
        'outlier_indices': np.where(outlier_flags)[0]
    }
```

#### Component 2.3: Mahalanobis Distance

**Method:** Multivariate distance in PCA space

**Visualization:**
- Bar chart of distances
- Threshold line (e.g., 3× median absolute deviation)

**Implementation:**
```python
def compute_mahalanobis_distance(scores):
    """
    Compute Mahalanobis distance for each sample in PCA space.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores (samples × n_components)

    Returns
    -------
    results : dict
        {
            'distances': Mahalanobis distance for each sample,
            'threshold': 3× MAD threshold,
            'outlier_flags': boolean array,
            'n_outliers': count,
            'outlier_indices': array of indices
        }
    """
    # Compute covariance and inverse
    cov_matrix = np.cov(scores.T)
    inv_cov = np.linalg.inv(cov_matrix)

    # Center of the distribution
    mean = np.mean(scores, axis=0)

    # Mahalanobis distance for each sample
    distances = []
    for score in scores:
        diff = score - mean
        distance = np.sqrt(diff @ inv_cov @ diff.T)
        distances.append(distance)

    distances = np.array(distances)

    # Threshold: 3× median absolute deviation
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    threshold = median + 3 * mad

    outlier_flags = distances > threshold

    return {
        'distances': distances,
        'median': median,
        'mad': mad,
        'threshold': threshold,
        'outlier_flags': outlier_flags,
        'n_outliers': np.sum(outlier_flags),
        'outlier_indices': np.where(outlier_flags)[0]
    }
```

#### Component 2.4: Y Data Consistency Checks

**Method:** Statistical checks on reference values

**Checks:**
1. Values outside ±3 standard deviations
2. Values outside chemically reasonable range (user-specified)
3. Histogram for visual inspection
4. Box plot for outlier detection

**Implementation:**
```python
def check_y_data_consistency(y, lower_bound=None, upper_bound=None):
    """
    Check reference data for outliers and inconsistencies.

    Parameters
    ----------
    y : np.ndarray or pd.Series
        Reference values
    lower_bound : float, optional
        Minimum chemically reasonable value
    upper_bound : float, optional
        Maximum chemically reasonable value

    Returns
    -------
    results : dict
        {
            'mean': mean value,
            'std': standard deviation,
            'z_scores': z-score for each sample,
            'z_outliers': samples with |z| > 3,
            'range_outliers': samples outside [lower, upper],
            'all_outliers': combined outlier flags,
            'n_outliers': count,
            'outlier_indices': array of indices
        }
    """
    y_array = np.array(y)

    # Compute statistics
    mean = np.mean(y_array)
    std = np.std(y_array)

    # Z-scores
    z_scores = (y_array - mean) / std
    z_outliers = np.abs(z_scores) > 3

    # Range check
    range_outliers = np.zeros(len(y_array), dtype=bool)
    if lower_bound is not None:
        range_outliers |= y_array < lower_bound
    if upper_bound is not None:
        range_outliers |= y_array > upper_bound

    # Combine
    all_outliers = z_outliers | range_outliers

    return {
        'mean': mean,
        'std': std,
        'median': np.median(y_array),
        'min': np.min(y_array),
        'max': np.max(y_array),
        'z_scores': z_scores,
        'z_outliers': z_outliers,
        'range_outliers': range_outliers,
        'all_outliers': all_outliers,
        'n_outliers': np.sum(all_outliers),
        'outlier_indices': np.where(all_outliers)[0]
    }
```

#### Component 2.5: Combined Outlier Report

**Function:** Aggregate all outlier detection methods

**Implementation:**
```python
def generate_outlier_report(X, y, n_pca_components=5,
                           y_lower_bound=None, y_upper_bound=None):
    """
    Comprehensive outlier detection report combining all methods.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Spectral data
    y : pd.Series or np.ndarray
        Reference values
    n_pca_components : int
        Number of PCs for analysis
    y_lower_bound : float, optional
        Minimum reasonable Y value
    y_upper_bound : float, optional
        Maximum reasonable Y value

    Returns
    -------
    report : dict
        {
            'pca': PCA results,
            'q_residuals': Q-residuals results,
            'mahalanobis': Mahalanobis distance results,
            'y_consistency': Y data check results,
            'combined_flags': combined outlier flags,
            'outlier_summary': DataFrame with all flags per sample
        }
    """
    # Run all detection methods
    pca_results = run_pca_outlier_detection(X, y, n_pca_components)
    q_results = compute_q_residuals(X, pca_results['pca_model'], n_pca_components)
    maha_results = compute_mahalanobis_distance(pca_results['scores'])
    y_results = check_y_data_consistency(y, y_lower_bound, y_upper_bound)

    # Create summary DataFrame
    import pandas as pd

    n_samples = X.shape[0]
    summary = pd.DataFrame({
        'Sample_Index': range(n_samples),
        'Y_Value': y if isinstance(y, np.ndarray) else y.values,
        'Hotelling_T2': pca_results['hotelling_t2'],
        'T2_Outlier': pca_results['outlier_flags'],
        'Q_Residual': q_results['q_residuals'],
        'Q_Outlier': q_results['outlier_flags'],
        'Mahalanobis_Distance': maha_results['distances'],
        'Maha_Outlier': maha_results['outlier_flags'],
        'Y_ZScore': y_results['z_scores'],
        'Y_Outlier': y_results['all_outliers'],
        'Total_Flags': (pca_results['outlier_flags'].astype(int) +
                       q_results['outlier_flags'].astype(int) +
                       maha_results['outlier_flags'].astype(int) +
                       y_results['all_outliers'].astype(int))
    })

    # Combined flags: flagged by 2+ methods
    combined_flags = summary['Total_Flags'] >= 2

    return {
        'pca': pca_results,
        'q_residuals': q_results,
        'mahalanobis': maha_results,
        'y_consistency': y_results,
        'combined_flags': combined_flags,
        'outlier_summary': summary,
        'high_confidence_outliers': summary[summary['Total_Flags'] >= 3],
        'moderate_confidence_outliers': summary[summary['Total_Flags'] == 2],
        'low_confidence_outliers': summary[summary['Total_Flags'] == 1]
    }
```

### GUI Implementation

#### New Tab: "Data Quality Check"

**File:** Create new file `src/spectral_predict/outlier_detection.py` with above functions

**GUI Location:** `spectral_predict_gui_optimized.py` - add new tab

**Tab Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  Data Quality Check                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Run Outlier Detection]  [Reset]  [Export Report]         │
│                                                             │
│  PCA Settings:                                              │
│    Number of components: [5    ▼]                          │
│    □ Auto-select based on variance (95%)                   │
│                                                             │
│  Y Value Range (optional):                                  │
│    Min: [________]  Max: [________]                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Results                                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  [Score Plot (PC1 vs PC2)]                           │  │
│  │                                                       │  │
│  │  Interactive matplotlib plot with:                   │  │
│  │  - Scatter points colored by Y value                 │  │
│  │  - Size by Hotelling T²                              │  │
│  │  - Hover for sample ID                               │  │
│  │  - 95% confidence ellipse                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  [Outlier Summary Table]                             │  │
│  │                                                       │  │
│  │  Sample | Y Value | T² | Q | Maha | Y | Total       │  │
│  │  ──────────────────────────────────────────────────  │  │
│  │  12     | 15.2   | ✓  |   | ✓   |   | 2 ⚠️         │  │
│  │  45     | 22.1   | ✓  | ✓ | ✓   | ✓ | 4 ⛔         │  │
│  │  ...                                                 │  │
│  │                                                       │  │
│  │  Legend: ✓ = Flagged | ⚠️ = 2-3 flags | ⛔ = 4 flags │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  [☐ Select All Flagged] [☑ Samples with 3+ flags]         │
│  [☐ Samples with 2+ flags]                                 │
│                                                             │
│  Selected: 5 samples                                        │
│  [Mark for Exclusion] [Keep All] [Review Individual]       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Workflow Integration

**Step 1:** User loads data in "Data Upload" tab

**Step 2:** System shows notification:
```
✓ Data loaded successfully (142 samples, 800 wavelengths)
→ Recommended: Run Data Quality Check before analysis
[Go to Quality Check]  [Skip and Continue]
```

**Step 3:** User runs quality check
- See visualizations
- Review flagged samples
- Select samples to exclude
- Document reasons for exclusion

**Step 4:** System creates filtered dataset
- Original data preserved
- Filtered data used for analysis
- Report includes exclusion log

**Step 5:** Proceed to Analysis Configuration with clean data

### Documentation Requirements

**For each excluded sample, record:**
1. Sample ID / index
2. Which methods flagged it (T², Q, Mahalanobis, Y)
3. Reason for exclusion (dropdown + free text)
   - Instrument error
   - Sample prep issue
   - Mislabeled
   - Out of range
   - Other (specify)
4. Timestamp
5. User who made decision

**Export format:** CSV file with exclusion log

### Testing Checklist

- [ ] PCA runs on spectral data
- [ ] Score plot displays correctly
- [ ] Hotelling T² computed and plotted
- [ ] Q-residuals computed and plotted
- [ ] Mahalanobis distance computed
- [ ] Y value checks work
- [ ] Combined report aggregates all methods
- [ ] Table shows all outlier flags clearly
- [ ] Can select/deselect samples
- [ ] Exclusion reasons can be documented
- [ ] Filtered dataset used in subsequent analysis
- [ ] Original data preserved
- [ ] Export report generates CSV
- [ ] Visual plots are interactive (hover, zoom)

**Estimated Time:** 8-12 hours for full implementation

---

## 📁 File Structure

```
dasp/
├── src/
│   └── spectral_predict/
│       ├── models.py              # Add Ridge/Lasso grids
│       ├── search.py              # Update feature importance
│       ├── outlier_detection.py  # NEW: All outlier detection code
│       └── ...
├── spectral_predict_gui_optimized.py  # Add checkboxes, new tab
└── OUTLIER_DETECTION_AND_RIDGE_LASSO_HANDOFF.md  # This file
```

---

## 🎯 Priority & Timeline

### Phase 1: Ridge & Lasso (Quick Win)
**Priority:** High
**Effort:** 2-3 hours
**Impact:** Immediate - better baselines, faster models

**Tasks:**
1. Add GUI checkboxes (30 min)
2. Implement hyperparameter grids (45 min)
3. Update feature importance (30 min)
4. Testing (1 hour)

### Phase 2: Outlier Detection (High Value)
**Priority:** High
**Effort:** 8-12 hours
**Impact:** Significant - better data quality, more reliable models

**Tasks:**
1. Implement outlier detection functions (3 hours)
2. Create GUI tab and visualizations (4 hours)
3. Integrate with workflow (2 hours)
4. Testing and documentation (3 hours)

**Recommended approach:** Implement Phase 1 first (quick win), then Phase 2

---

## 📚 References

### Ridge & Lasso
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

### Outlier Detection
- Hotelling, H. (1931). The generalization of Student's ratio. *The Annals of Mathematical Statistics*, 2(3), 360-378.
- Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals associated with principal component analysis. *Technometrics*, 21(3), 341-349.
- De Maesschalck, R., Jouan-Rimbaud, D., & Massart, D. L. (2000). The Mahalanobis distance. *Chemometrics and Intelligent Laboratory Systems*, 50(1), 1-18.

### Spectroscopy-Specific
- Workman Jr, J., & Weyer, L. (2012). *Practical guide to interpretive near-infrared spectroscopy*. CRC press.
- Esbensen, K. H., et al. (2002). *Multivariate Data Analysis – in Practice*. CAMO Software AS.

---

## 💡 Additional Considerations

### Ridge & Lasso
- **When to use Ridge:** High collinearity (very common in NIR spectroscopy)
- **When to use Lasso:** Feature selection desired (sparse solutions)
- **Comparison to PLS:** Similar performance often, but Ridge/Lasso are simpler
- **Preprocessing:** Both sensitive to scaling - SNV/normalization recommended

### Outlier Detection
- **False positives:** Some "outliers" may be valuable edge cases
- **Iterative approach:** Run detection, exclude, re-run detection
- **Validation:** Check if model improves after exclusion
- **Documentation:** Critical for regulatory/quality assurance
- **Reversibility:** Always keep original data

---

## ✅ Acceptance Criteria

### Ridge & Lasso
- [ ] Checkboxes appear in GUI
- [ ] Models run when selected
- [ ] Results appear in Results tab
- [ ] Feature importance extracts coefficients
- [ ] Performance comparable to literature benchmarks
- [ ] Faster than RF/MLP/NeuralBoosted

### Outlier Detection
- [ ] All detection methods implemented correctly
- [ ] Visualizations are clear and informative
- [ ] User can select/deselect samples
- [ ] Exclusion reasons are documented
- [ ] Filtered data used in analysis
- [ ] Original data preserved
- [ ] Export generates comprehensive report
- [ ] No crashes with edge cases (single outlier, no outliers, etc.)

---

## 🤝 Handoff Checklist

- [x] Document created with full implementation details
- [x] Scientific justification provided
- [x] Code examples included
- [x] Testing checklists defined
- [x] File locations specified
- [x] Timeline estimated
- [x] References cited
- [x] Acceptance criteria clear

**Ready for implementation!** 🚀

---

**Questions or clarifications?** Contact the development team or refer to:
- Main codebase: `spectral_predict_gui_optimized.py`
- Model definitions: `src/spectral_predict/models.py`
- Search logic: `src/spectral_predict/search.py`
- Documentation: Project README files
