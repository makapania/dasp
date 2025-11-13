# Advanced Calibration Transfer Implementation Plan

## Executive Summary

This document outlines the detailed plan to implement four advanced calibration transfer methods into the existing DASP calibration transfer module:

1. **CTAI** (Calibration Transfer based on Affine Invariance) - No transfer standards required
2. **NS-PFCE** (Non-supervised Parameter-Free Framework for Calibration Enhancement) - Best with wavelength selection
3. **TSR** (Transfer Sample Regression / Shenk-Westerhaus)
4. **JYPLS-inv** (Joint-Y PLS with inversion)

**Current Implementation Status:**
- ✅ Direct Standardization (DS)
- ✅ Piecewise Direct Standardization (PDS)
- ✅ GUI interface for calibration transfer (Tab 10)
- ✅ TransferModel infrastructure
- ✅ Instrument profiling and equalization

---

## Phase 1: Research & Algorithm Specification (Week 1)

### 1.1 CTAI (Calibration Transfer based on Affine Invariance)

**Key Characteristics:**
- **No transfer standards required** - Major advantage
- Based on affine transformation invariance properties
- Works on the assumption that spectral differences between instruments follow affine transformations
- Lowest prediction errors across multiple datasets in literature

**Algorithm Components:**

```python
# Core CTAI approach:
# 1. Estimate affine transformation parameters from spectral data structure
# 2. Apply inverse transformation to map slave to master domain
# 3. Use matrix decomposition (SVD/PCA) to identify invariant subspace

def estimate_ctai(X_master: np.ndarray, X_slave: np.ndarray) -> Dict:
    """
    Estimate CTAI transformation without transfer standards.

    Parameters:
    -----------
    X_master : np.ndarray
        Master instrument spectra (n_samples, p)
    X_slave : np.ndarray
        Slave instrument spectra (n_samples, p)

    Returns:
    --------
    params : Dict
        Contains 'T' (translation), 'M' (transformation matrix),
        'explained_variance', 'n_components'
    """
    pass

def apply_ctai(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply CTAI transformation to new slave spectra."""
    pass
```

**Implementation Files:**
- `src/spectral_predict/calibration_transfer.py` - Add CTAI methods
- Update `MethodType` to include `"ctai"`
- Update `TransferModel.params` schema for CTAI

**Dependencies:**
- NumPy/SciPy for matrix operations
- Potentially scikit-learn for robust PCA

**Research Tasks:**
- [ ] Review original CTAI paper (likely from Journal of Chemometrics)
- [ ] Document mathematical formulation
- [ ] Identify edge cases (what if spectral overlap is poor?)
- [ ] Determine optimal number of components selection strategy

---

### 1.2 NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement)

**Key Characteristics:**
- Non-supervised (no labeled transfer samples needed)
- Parameter-free (automatic optimization)
- Best performance when combined with **VCPA-IRIV** wavelength selection
- Adaptive to different instrument configurations

**Algorithm Components:**

```python
def estimate_nspfce(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    wavelengths: np.ndarray,
    use_wavelength_selection: bool = True,
    wavelength_selector: str = 'vcpa-iriv'
) -> Dict:
    """
    Estimate NS-PFCE calibration transfer.

    Key steps:
    1. Optional: Select informative wavelengths using VCPA-IRIV
    2. Estimate spectral transformation using iterative optimization
    3. Apply adaptive normalization

    Parameters:
    -----------
    X_master : np.ndarray
        Master spectra (n_samples, p)
    X_slave : np.ndarray
        Slave spectra (n_samples, p)
    wavelengths : np.ndarray
        Wavelength grid (p,)
    use_wavelength_selection : bool
        Whether to apply VCPA-IRIV wavelength selection
    wavelength_selector : str
        Method for wavelength selection ('vcpa-iriv', 'cars', 'spa')

    Returns:
    --------
    params : Dict
        Contains transformation matrices, selected wavelengths indices,
        convergence metrics
    """
    pass

def vcpa_iriv_wavelength_selection(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    n_iterations: int = 100
) -> np.ndarray:
    """
    Variable Combination Population Analysis - Iteratively
    Retains Informative Variables.

    Returns:
    --------
    selected_indices : np.ndarray
        Indices of selected wavelengths
    """
    pass
```

**Implementation Files:**
- `src/spectral_predict/calibration_transfer.py` - Add NS-PFCE core
- `src/spectral_predict/wavelength_selection.py` - NEW MODULE for VCPA-IRIV
- Update method types and GUI

**Dependencies:**
- NumPy/SciPy
- Potentially genetic algorithm library for VCPA optimization

**Research Tasks:**
- [ ] Review NS-PFCE methodology papers
- [ ] Document VCPA-IRIV algorithm (may be complex)
- [ ] Identify IRIV stopping criteria
- [ ] Design fallback if wavelength selection fails

---

### 1.3 TSR (Transfer Sample Regression / Shenk-Westerhaus)

**Key Characteristics:**
- Classic method, well-established
- Requires 12-13 optimally selected transfer samples
- Outperformed PDS in recent studies
- Results statistically indistinguishable from full recalibration

**Algorithm Components:**

```python
def select_transfer_samples_optimal(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    n_samples: int = 12,
    method: str = 'kennard-stone'
) -> np.ndarray:
    """
    Select optimal transfer samples using sample selection algorithms.

    Methods:
    --------
    - 'kennard-stone': KS algorithm for representative sampling
    - 'duplex': DUPLEX algorithm
    - 'spxy': Sample set Partitioning based on X and Y
    - 'random': Random selection (baseline)

    Returns:
    --------
    indices : np.ndarray
        Indices of selected samples
    """
    pass

def estimate_tsr(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    transfer_indices: np.ndarray,
    slope_bias_correction: bool = True
) -> Dict:
    """
    Estimate Transfer Sample Regression (Shenk-Westerhaus).

    Steps:
    1. Use selected transfer samples to build regression
    2. Estimate slope and bias corrections per wavelength
    3. Apply corrections across full spectral range

    Parameters:
    -----------
    X_master : np.ndarray
        Master spectra (n_samples, p)
    X_slave : np.ndarray
        Slave spectra (n_samples, p)
    transfer_indices : np.ndarray
        Indices of transfer samples
    slope_bias_correction : bool
        Apply slope and bias correction (recommended)

    Returns:
    --------
    params : Dict
        Contains 'slope', 'bias', 'transfer_indices',
        'residuals', 'r_squared'
    """
    pass

def apply_tsr(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply TSR slope/bias correction."""
    # X_corrected = slope * X_slave_new + bias
    pass
```

**Implementation Files:**
- `src/spectral_predict/calibration_transfer.py` - Add TSR
- `src/spectral_predict/sample_selection.py` - NEW MODULE for Kennard-Stone, DUPLEX, SPXY
- GUI: Add sample selection interface

**Dependencies:**
- NumPy/SciPy
- scikit-learn for distance metrics

**Research Tasks:**
- [ ] Review Shenk-Westerhaus original paper
- [ ] Document Kennard-Stone algorithm
- [ ] Document DUPLEX and SPXY algorithms
- [ ] Determine optimal number of transfer samples (literature suggests 12-13)

---

### 1.4 JYPLS-inv (Joint-Y PLS with Inversion)

**Key Characteristics:**
- Based on PLS regression with joint Y-matrix
- Requires transfer samples (12-13 optimal)
- Comparable performance to TSR
- Leverages PLS modeling framework

**Algorithm Components:**

```python
def estimate_jypls_inv(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    y_transfer: np.ndarray,
    transfer_indices: np.ndarray,
    n_components: int = None,
    cv_folds: int = 5
) -> Dict:
    """
    Estimate JYPLS-inv calibration transfer.

    Approach:
    1. Build joint PLS model using transfer samples
    2. Create augmented X matrix [X_master; X_slave]
    3. Fit PLS on augmented data with shared Y
    4. Extract transformation from PLS loadings
    5. Invert to get slave-to-master mapping

    Parameters:
    -----------
    X_master : np.ndarray
        Master transfer spectra (n_transfer, p)
    X_slave : np.ndarray
        Slave transfer spectra (n_transfer, p)
    y_transfer : np.ndarray
        Reference values for transfer samples (n_transfer,)
    transfer_indices : np.ndarray
        Indices of transfer samples
    n_components : int, optional
        Number of PLS components (auto-select if None)
    cv_folds : int
        Cross-validation folds for component selection

    Returns:
    --------
    params : Dict
        Contains PLS model, transformation matrices,
        optimal components, prediction metrics
    """
    pass

def apply_jypls_inv(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply JYPLS-inv transformation."""
    pass
```

**Implementation Files:**
- `src/spectral_predict/calibration_transfer.py` - Add JYPLS-inv
- Leverage existing PLS infrastructure if available
- GUI updates

**Dependencies:**
- NumPy/SciPy
- scikit-learn PLS implementation
- Existing PLS code in spectral_predict

**Research Tasks:**
- [ ] Review JYPLS original paper
- [ ] Document inversion mathematics
- [ ] Determine optimal component selection strategy
- [ ] Design validation approach

---

## Phase 2: Architecture Design (Week 1-2)

### 2.1 Module Structure

**Updated File Organization:**

```
src/spectral_predict/
├── calibration_transfer.py         # Core CT module (UPDATED)
│   ├── DS, PDS (existing)
│   ├── CTAI (new)
│   ├── NS-PFCE (new)
│   ├── TSR (new)
│   └── JYPLS-inv (new)
│
├── wavelength_selection.py         # NEW MODULE
│   ├── vcpa_iriv()
│   ├── cars()
│   ├── spa()
│   └── compare_selectors()
│
├── sample_selection.py             # NEW MODULE
│   ├── kennard_stone()
│   ├── duplex()
│   ├── spxy()
│   └── compare_selections()
│
├── calibration_transfer_evaluation.py  # NEW MODULE
│   ├── evaluate_transfer_quality()
│   ├── cross_validation_transfer()
│   ├── benchmark_methods()
│   └── generate_report()
│
└── equalization.py                 # (existing, minimal changes)
```

### 2.2 Updated TransferModel Schema

```python
@dataclass
class TransferModel:
    """Enhanced TransferModel supporting all methods."""
    master_id: str
    slave_id: str
    method: MethodType  # "ds", "pds", "ctai", "nspfce", "tsr", "jypls-inv"
    wavelengths_common: np.ndarray
    params: Dict
    meta: Dict = field(default_factory=dict)

    # New optional fields for advanced methods
    selected_wavelengths: np.ndarray | None = None
    transfer_sample_indices: np.ndarray | None = None
    quality_metrics: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate method-specific parameters."""
        self._validate_params()

    def _validate_params(self):
        """Ensure required params exist for each method."""
        required = {
            'ds': ['A'],
            'pds': ['B', 'window'],
            'ctai': ['T', 'M'],
            'nspfce': ['transformation_matrix'],
            'tsr': ['slope', 'bias'],
            'jypls-inv': ['pls_model', 'n_components']
        }
        # Validation logic...
```

### 2.3 GUI Integration Plan

**Updates to `spectral_predict_gui_optimized.py`:**

1. **Section C Enhancement** (Build Transfer Mapping):
   ```python
   # Add new method radiobuttons
   ttk.Radiobutton(method_frame, text="CTAI (No Standards)",
                   variable=self.ct_method_var, value='ctai')
   ttk.Radiobutton(method_frame, text="NS-PFCE (Auto)",
                   variable=self.ct_method_var, value='nspfce')
   ttk.Radiobutton(method_frame, text="TSR (Shenk-Westerhaus)",
                   variable=self.ct_method_var, value='tsr')
   ttk.Radiobutton(method_frame, text="JYPLS-inv",
                   variable=self.ct_method_var, value='jypls-inv')
   ```

2. **New Section: Sample Selection** (for TSR/JYPLS-inv):
   ```python
   # Section B2: Select Transfer Samples (conditionally shown)
   section_b2 = ttk.LabelFrame(main_frame,
                               text="B2) Select Transfer Samples (TSR/JYPLS-inv)",
                               style='Card.TFrame', padding=15)

   # Sample selection method combobox
   self.ct_sample_selection_method = tk.StringVar(value='kennard-stone')
   # Number of samples spinbox (default 12)
   self.ct_n_transfer_samples = tk.IntVar(value=12)
   # Visualize selected samples button
   # Manual selection table (optional)
   ```

3. **New Section: Wavelength Selection** (for NS-PFCE):
   ```python
   # Section C2: Wavelength Selection (NS-PFCE)
   section_c2 = ttk.LabelFrame(main_frame,
                               text="C2) Wavelength Selection (Optional)",
                               style='Card.TFrame', padding=15)

   # Enable wavelength selection checkbox
   self.ct_use_wavelength_selection = tk.BooleanVar(value=True)
   # Method selection (VCPA-IRIV, CARS, SPA)
   self.ct_wavelength_method = tk.StringVar(value='vcpa-iriv')
   # Visualize selected wavelengths
   ```

4. **Enhanced Validation & Comparison**:
   ```python
   # Section F: Method Comparison (NEW)
   section_f = ttk.LabelFrame(main_frame,
                              text="F) Compare Transfer Methods",
                              style='Card.TFrame', padding=15)

   # Checkboxes to select methods to compare
   # Run benchmark button
   # Results table showing RMSE, R², bias for each method
   # Visualization: comparison plots
   ```

### 2.4 Method Selection Logic

```python
def _build_ct_transfer_model(self):
    """Enhanced method to handle all CT approaches."""
    method = self.ct_method_var.get()

    if method == 'ctai':
        # No transfer samples needed
        params = estimate_ctai(X_master_common, X_slave_common)

    elif method == 'nspfce':
        # Optional wavelength selection
        use_wl_sel = self.ct_use_wavelength_selection.get()
        wl_method = self.ct_wavelength_method.get()
        params = estimate_nspfce(
            X_master_common, X_slave_common,
            wavelengths_common,
            use_wavelength_selection=use_wl_sel,
            wavelength_selector=wl_method
        )

    elif method in ['tsr', 'jypls-inv']:
        # Requires transfer sample selection
        n_samples = self.ct_n_transfer_samples.get()
        selection_method = self.ct_sample_selection_method.get()

        transfer_indices = select_transfer_samples_optimal(
            X_master_common, X_slave_common,
            n_samples=n_samples,
            method=selection_method
        )

        if method == 'tsr':
            params = estimate_tsr(
                X_master_common, X_slave_common,
                transfer_indices
            )
        else:  # jypls-inv
            # Need reference values for transfer samples
            y_transfer = self._get_transfer_sample_references(transfer_indices)
            params = estimate_jypls_inv(
                X_master_common, X_slave_common,
                y_transfer, transfer_indices
            )

    # Create and store TransferModel
    # Update GUI with results
```

---

## Phase 3: Implementation (Weeks 2-4)

### 3.1 Implementation Priority Order

**Week 2:**
1. ✅ Implement sample selection algorithms (Kennard-Stone, DUPLEX, SPXY)
   - File: `src/spectral_predict/sample_selection.py`
   - Test: `tests/test_sample_selection.py`
   - Rationale: Required by TSR and JYPLS-inv

2. ✅ Implement TSR (simplest advanced method)
   - File: `src/spectral_predict/calibration_transfer.py`
   - Test: `tests/test_calibration_transfer_tsr.py`
   - Rationale: Well-documented, straightforward implementation

**Week 3:**
3. ✅ Implement CTAI
   - File: `src/spectral_predict/calibration_transfer.py`
   - Test: `tests/test_calibration_transfer_ctai.py`
   - Rationale: No transfer samples needed, high impact

4. ✅ Implement wavelength selection (VCPA-IRIV foundation)
   - File: `src/spectral_predict/wavelength_selection.py`
   - Test: `tests/test_wavelength_selection.py`
   - Rationale: Required by NS-PFCE

**Week 4:**
5. ✅ Implement NS-PFCE
   - File: `src/spectral_predict/calibration_transfer.py`
   - Test: `tests/test_calibration_transfer_nspfce.py`
   - Rationale: Complex, requires wavelength selection

6. ✅ Implement JYPLS-inv
   - File: `src/spectral_predict/calibration_transfer.py`
   - Test: `tests/test_calibration_transfer_jypls.py`
   - Rationale: Requires PLS infrastructure

### 3.2 Detailed Implementation Checklist

#### 3.2.1 Sample Selection Module

**File:** `src/spectral_predict/sample_selection.py`

```python
# Functions to implement:

def kennard_stone(X: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Kennard-Stone algorithm for representative sample selection.

    Algorithm:
    1. Find two samples with maximum Euclidean distance
    2. Iteratively add samples that are farthest from selected set
    3. Continue until n_samples are selected
    """
    # TODO: Implement KS algorithm
    pass

def duplex(X: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    DUPLEX algorithm for splitting calibration/validation sets.

    Returns:
    --------
    cal_indices, val_indices : Tuple[np.ndarray, np.ndarray]
    """
    # TODO: Implement DUPLEX
    pass

def spxy(X: np.ndarray, y: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Sample set Partitioning based on joint X-Y distance (SPXY).
    """
    # TODO: Implement SPXY
    pass

def compare_selection_methods(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    methods: List[str] = ['kennard-stone', 'duplex', 'spxy', 'random']
) -> Dict:
    """Compare different sample selection strategies."""
    # TODO: Implement comparison framework
    pass
```

**Tests Required:**
- Test KS produces diverse samples
- Test DUPLEX creates balanced cal/val split
- Test SPXY considers both X and Y space
- Test edge cases (n_samples > n_total, n_samples = 1)
- Benchmark performance on synthetic data

---

#### 3.2.2 Wavelength Selection Module

**File:** `src/spectral_predict/wavelength_selection.py`

```python
def vcpa_iriv(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    n_iterations: int = 100,
    binary_matrix_iterations: int = 1000,
    pls_components: int = 5,
    cv_folds: int = 5
) -> Dict:
    """
    Variable Combination Population Analysis - Iteratively
    Retains Informative Variables (VCPA-IRIV).

    Algorithm Overview:
    1. Generate binary matrix (BM) for variable combinations
    2. Build PLS models for each combination
    3. Evaluate using cross-validation
    4. Iteratively remove uninformative variables
    5. Repeat until convergence

    Returns:
    --------
    result : Dict
        {
            'selected_indices': np.ndarray,
            'selected_wavelengths': np.ndarray,
            'importance_scores': np.ndarray,
            'convergence_history': List,
            'final_rmsecv': float
        }
    """
    # TODO: Implement VCPA-IRIV
    # This is complex - may need 200-300 lines
    pass

def cars(X: np.ndarray, y: np.ndarray, wavelengths: np.ndarray) -> Dict:
    """Competitive Adaptive Reweighted Sampling (CARS)."""
    # TODO: Implement CARS (simpler alternative)
    pass

def spa(X: np.ndarray, y: np.ndarray, wavelengths: np.ndarray,
        n_vars: int) -> Dict:
    """Successive Projections Algorithm (SPA)."""
    # TODO: Implement SPA
    pass
```

**Tests Required:**
- Test VCPA-IRIV selects informative wavelengths on synthetic data
- Test convergence behavior
- Test CARS and SPA as alternatives
- Benchmark computational performance
- Validate against known good selections

---

#### 3.2.3 CTAI Implementation

**File:** `src/spectral_predict/calibration_transfer.py` (add to existing)

```python
def estimate_ctai(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    n_components: int | None = None,
    explained_variance_threshold: float = 0.99
) -> Dict:
    """
    Estimate CTAI transformation.

    Algorithm (simplified):
    1. Center both datasets
    2. Compute covariance matrices
    3. Perform SVD/eigendecomposition
    4. Identify affine-invariant subspace
    5. Estimate transformation matrix M and translation T
    6. Validate transformation quality

    Mathematical formulation:
    X_master ≈ M @ X_slave + T

    Where M and T are found via:
    - Minimizing || X_master - (M @ X_slave + T) ||_F
    - Subject to affine invariance constraints
    """
    from scipy.linalg import svd
    from sklearn.decomposition import PCA

    # TODO: Implement CTAI estimation
    # Key steps:
    # 1. Mean-center data
    # 2. Compute cross-covariance
    # 3. SVD to find transformation
    # 4. Validate with reconstruction error

    params = {
        'M': None,  # Transformation matrix (p, p)
        'T': None,  # Translation vector (p,)
        'n_components': None,
        'explained_variance': None,
        'reconstruction_error': None
    }

    return params

def apply_ctai(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply CTAI transformation: X_master_predicted = M @ X_slave + T"""
    M = params['M']
    T = params['T']
    return X_slave_new @ M.T + T
```

**Tests Required:**
- Test on synthetic data with known affine transformation
- Test recovery of transformation parameters
- Test reconstruction accuracy
- Test handling of rank-deficient cases
- Validate against literature results

---

#### 3.2.4 TSR Implementation

**File:** `src/spectral_predict/calibration_transfer.py`

```python
def estimate_tsr(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    transfer_indices: np.ndarray,
    slope_bias_correction: bool = True,
    regularization: float = 0.0
) -> Dict:
    """
    Transfer Sample Regression (Shenk-Westerhaus method).

    Algorithm:
    1. Extract transfer samples from master and slave
    2. For each wavelength λ:
       - Fit linear regression: X_master[λ] = slope[λ] * X_slave[λ] + bias[λ]
       - Store slope and bias coefficients
    3. Apply correction to all samples

    Advantages:
    - Simple and interpretable
    - Robust with 12-13 well-selected samples
    - Fast computation

    Disadvantages:
    - Requires transfer samples with reference values
    - Assumes linear relationship per wavelength
    """
    n_wavelengths = X_master.shape[1]
    slopes = np.zeros(n_wavelengths)
    biases = np.zeros(n_wavelengths)
    r_squared = np.zeros(n_wavelengths)

    X_master_transfer = X_master[transfer_indices]
    X_slave_transfer = X_slave[transfer_indices]

    for i in range(n_wavelengths):
        x = X_slave_transfer[:, i]
        y = X_master_transfer[:, i]

        # Simple linear regression with optional regularization
        # y = slope * x + bias
        # TODO: Implement robust regression

        slopes[i] = slope
        biases[i] = bias
        r_squared[i] = r2

    params = {
        'slope': slopes,
        'bias': biases,
        'transfer_indices': transfer_indices,
        'r_squared': r_squared,
        'mean_r_squared': np.mean(r_squared),
        'wavelength_quality': r_squared
    }

    return params

def apply_tsr(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply TSR correction: X_corrected = slope * X_slave + bias"""
    slope = params['slope']
    bias = params['bias']
    return X_slave_new * slope + bias
```

**Tests Required:**
- Test perfect linear transformation recovery
- Test with different numbers of transfer samples (6, 12, 20)
- Test robustness to noisy transfer samples
- Compare to PDS performance
- Validate wavelength-wise R² metrics

---

#### 3.2.5 NS-PFCE Implementation

**File:** `src/spectral_predict/calibration_transfer.py`

```python
def estimate_nspfce(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    wavelengths: np.ndarray,
    use_wavelength_selection: bool = True,
    wavelength_selector: str = 'vcpa-iriv',
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6
) -> Dict:
    """
    Non-supervised Parameter-Free Calibration Enhancement.

    Algorithm (high-level):
    1. Optional: Wavelength selection (VCPA-IRIV recommended)
    2. Initialize transformation matrix
    3. Iteratively optimize:
       a. Estimate spectral differences
       b. Update transformation adaptively
       c. Apply normalization
       d. Check convergence
    4. Return optimized transformation

    Key innovation: No parameters to tune, fully automatic
    """
    # Step 1: Wavelength selection
    if use_wavelength_selection:
        from .wavelength_selection import vcpa_iriv, cars, spa

        # Need pseudo-Y for wavelength selection
        # Use first principal component or spectral mean
        y_pseudo = np.mean(X_master, axis=1)

        if wavelength_selector == 'vcpa-iriv':
            wl_result = vcpa_iriv(X_master, y_pseudo, wavelengths)
        elif wavelength_selector == 'cars':
            wl_result = cars(X_master, y_pseudo, wavelengths)
        else:
            wl_result = spa(X_master, y_pseudo, wavelengths)

        selected_indices = wl_result['selected_indices']
        X_master_sel = X_master[:, selected_indices]
        X_slave_sel = X_slave[:, selected_indices]
    else:
        selected_indices = np.arange(X_master.shape[1])
        X_master_sel = X_master
        X_slave_sel = X_slave

    # Step 2: Iterative optimization
    # TODO: Implement NS-PFCE core algorithm
    # This is complex and may require literature review

    transformation_matrix = np.eye(len(selected_indices))  # Placeholder

    params = {
        'transformation_matrix': transformation_matrix,
        'selected_wavelengths': selected_indices,
        'wavelength_selector': wavelength_selector,
        'convergence_iterations': None,
        'final_objective': None
    }

    return params

def apply_nspfce(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply NS-PFCE transformation."""
    T = params['transformation_matrix']
    selected_idx = params['selected_wavelengths']

    X_selected = X_slave_new[:, selected_idx]
    X_transformed_selected = X_selected @ T

    # Reconstruct full spectrum (interpolate or pad)
    # TODO: Implement reconstruction

    return X_transformed
```

**Tests Required:**
- Test with and without wavelength selection
- Test convergence behavior
- Compare performance to DS/PDS baseline
- Validate parameter-free claim
- Test on diverse spectral datasets

---

#### 3.2.6 JYPLS-inv Implementation

**File:** `src/spectral_predict/calibration_transfer.py`

```python
def estimate_jypls_inv(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    y_transfer: np.ndarray,
    transfer_indices: np.ndarray,
    n_components: int | None = None,
    cv_folds: int = 5,
    max_components: int = 20
) -> Dict:
    """
    Joint-Y PLS with inversion for calibration transfer.

    Algorithm:
    1. Create augmented X matrix: X_aug = [X_master; X_slave]
    2. Create augmented Y matrix: Y_aug = [y_transfer; y_transfer]
    3. Build PLS model: PLS(X_aug, Y_aug)
    4. Extract loadings and scores for master/slave separately
    5. Compute transformation from slave to master space
    6. Invert to get calibration transfer matrix

    Mathematical formulation:
    X_master = X_slave @ B_transfer
    Where B_transfer is derived from PLS loadings/weights
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score

    # Extract transfer samples
    X_master_transfer = X_master[transfer_indices]
    X_slave_transfer = X_slave[transfer_indices]

    # Create augmented matrices
    X_aug = np.vstack([X_master_transfer, X_slave_transfer])
    Y_aug = np.hstack([y_transfer, y_transfer])

    # Optimal component selection via CV
    if n_components is None:
        best_score = -np.inf
        best_n = 1

        for n in range(1, max_components + 1):
            pls = PLSRegression(n_components=n)
            scores = cross_val_score(pls, X_aug, Y_aug, cv=cv_folds,
                                    scoring='neg_mean_squared_error')
            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_n = n

        n_components = best_n

    # Fit final PLS model
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_aug, Y_aug)

    # Extract transformation from PLS structure
    # TODO: Implement transformation extraction logic
    # B_transfer = f(pls.x_weights_, pls.x_loadings_, ...)

    params = {
        'pls_model': pls,
        'n_components': n_components,
        'transfer_indices': transfer_indices,
        'transformation_matrix': None,  # Derived from PLS
        'cv_score': best_score,
        'explained_variance_X': pls.x_scores_,
        'explained_variance_Y': pls.y_scores_
    }

    return params

def apply_jypls_inv(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply JYPLS-inv transformation."""
    B = params['transformation_matrix']
    return X_slave_new @ B
```

**Tests Required:**
- Test on synthetic PLS-structured data
- Validate component selection
- Compare to direct PLS prediction
- Test sensitivity to transfer sample quality
- Benchmark against TSR

---

## Phase 4: Testing & Validation (Week 4-5)

### 4.1 Testing Strategy

**Test Categories:**

1. **Unit Tests** (per method)
   - Algorithm correctness
   - Parameter validation
   - Edge case handling
   - Numerical stability

2. **Integration Tests**
   - Method interoperability
   - GUI integration
   - Save/load TransferModel
   - Multi-method workflows

3. **Performance Tests**
   - Computational efficiency
   - Memory usage
   - Scalability (large datasets)
   - Real-time prediction latency

4. **Validation Tests**
   - Literature benchmark datasets
   - Comparison to published results
   - Cross-method comparison
   - Statistical significance tests

### 4.2 Synthetic Data Test Suite

**File:** `tests/test_calibration_transfer_synthetic.py`

```python
def generate_affine_transformed_spectra(
    n_samples: int = 100,
    n_wavelengths: int = 200,
    noise_level: float = 0.01,
    slope_range: Tuple[float, float] = (0.8, 1.2),
    bias_range: Tuple[float, float] = (-0.1, 0.1)
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate master and slave spectra with known affine transformation.

    Returns:
    --------
    X_master, X_slave, true_params
    """
    pass

def test_all_methods_on_synthetic():
    """Comprehensive test of all CT methods on synthetic data."""
    X_master, X_slave, true_params = generate_affine_transformed_spectra()

    methods_to_test = ['ds', 'pds', 'ctai', 'nspfce', 'tsr', 'jypls-inv']
    results = {}

    for method in methods_to_test:
        # Estimate transformation
        # Apply to test set
        # Compute RMSE, R², bias
        # Store results
        pass

    # Assert all methods achieve reasonable accuracy
    # Assert CTAI outperforms (as per literature)
    # Assert TSR and JYPLS-inv comparable with optimal samples
```

### 4.3 Benchmark Datasets

**Recommended Public Datasets:**

1. **Corn Dataset** (classic NIR benchmark)
   - Multiple instruments (m5, mp5, mp6)
   - Well-documented
   - Available from Eigenvector Research

2. **Tablet Dataset** (pharmaceutical NIR)
   - Two instruments
   - API content prediction
   - Realistic noise characteristics

3. **Soil Dataset** (VNIR/MIR)
   - Multi-instrument soil organic carbon
   - Large sample size
   - Challenging spectral variations

**Benchmark Script:**

```python
# File: example/benchmark_calibration_transfer.py

def run_benchmark_suite():
    """
    Run all CT methods on benchmark datasets and generate report.
    """
    datasets = load_benchmark_datasets()
    methods = ['ds', 'pds', 'ctai', 'nspfce', 'tsr', 'jypls-inv']

    results = []

    for dataset_name, data in datasets.items():
        X_master, X_slave, y_true = data

        for method in methods:
            # Estimate model
            # Predict on test set
            # Compute metrics
            # Record results

            results.append({
                'dataset': dataset_name,
                'method': method,
                'rmse': rmse,
                'r_squared': r2,
                'bias': bias,
                'mae': mae,
                'computation_time': time_elapsed
            })

    # Generate comparison plots
    # Save results to CSV
    # Print summary table
```

### 4.4 Statistical Validation

```python
def statistical_comparison_test(
    predictions_method_a: np.ndarray,
    predictions_method_b: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Test if two methods have statistically different performance.

    Uses:
    - Paired t-test on residuals
    - Williams' test for comparing correlations
    - Bootstrap confidence intervals

    Returns:
    --------
    result : Dict
        {'p_value': float, 'statistically_different': bool,
         'confidence_interval': Tuple, 'effect_size': float}
    """
    from scipy.stats import ttest_rel

    residuals_a = predictions_method_a - y_true
    residuals_b = predictions_method_b - y_true

    # Paired t-test on absolute residuals
    stat, p_value = ttest_rel(np.abs(residuals_a), np.abs(residuals_b))

    # TODO: Add Williams' test for R² comparison
    # TODO: Add bootstrap CI

    return {
        'p_value': p_value,
        'statistically_different': p_value < alpha,
        'better_method': 'A' if np.mean(np.abs(residuals_a)) < np.mean(np.abs(residuals_b)) else 'B'
    }
```

---

## Phase 5: GUI Enhancement (Week 5)

### 5.1 Enhanced Method Selection UI

**Dynamic Parameter Panels:**

```python
def _on_ct_method_changed(self, *args):
    """Update GUI based on selected calibration transfer method."""
    method = self.ct_method_var.get()

    # Hide all method-specific panels
    self.tsr_params_frame.pack_forget()
    self.jypls_params_frame.pack_forget()
    self.nspfce_params_frame.pack_forget()
    self.ctai_params_frame.pack_forget()

    # Show relevant panel
    if method in ['tsr', 'jypls-inv']:
        self.sample_selection_frame.pack(fill='x', pady=(10, 0))
        if method == 'tsr':
            self.tsr_params_frame.pack(fill='x', pady=(5, 0))
        else:
            self.jypls_params_frame.pack(fill='x', pady=(5, 0))
    elif method == 'nspfce':
        self.nspfce_params_frame.pack(fill='x', pady=(10, 0))
    elif method == 'ctai':
        self.ctai_params_frame.pack(fill='x', pady=(10, 0))

    # Update help text
    self._update_method_help_text(method)
```

### 5.2 Visualization Enhancements

**New Plots:**

1. **Transfer Quality Visualization**
   ```python
   def _plot_transfer_quality(self, X_master, X_slave, X_transferred):
       """
       Create multi-panel plot showing:
       - Before/after spectra overlay
       - Residual heatmap
       - Wavelength-wise RMSE
       - PCA scatter (before/after alignment)
       """
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))

       # Panel 1: Spectra overlay
       # Panel 2: Residuals
       # Panel 3: Wavelength RMSE
       # Panel 4: PCA scatter
   ```

2. **Sample Selection Visualization**
   ```python
   def _plot_selected_samples(self, X, selected_indices, method_name):
       """
       Visualize which samples were selected for transfer.

       Shows:
       - PCA scatter with selected samples highlighted
       - Sample diversity metrics
       - Spectral coverage
       """
   ```

3. **Wavelength Selection Visualization**
   ```python
   def _plot_selected_wavelengths(self, wavelengths, selected_indices,
                                  importance_scores):
       """
       Show which wavelengths were selected and why.

       Displays:
       - Full spectrum with selected regions highlighted
       - Importance score bar chart
       - Cumulative explained variance
       """
   ```

### 5.3 Method Comparison Interface

**Section F: Method Comparison (NEW)**

```python
def _create_method_comparison_section(self, parent_frame):
    """
    Allow users to run multiple CT methods and compare results.
    """
    section_f = ttk.LabelFrame(parent_frame,
                              text="F) Compare Calibration Transfer Methods",
                              style='Card.TFrame', padding=15)
    section_f.pack(fill='x', pady=(0, 15))

    # Method selection checkboxes
    ttk.Label(section_f, text="Select methods to compare:",
             style='CardLabel.TLabel').pack(anchor='w', pady=(0, 10))

    checkbox_frame = ttk.Frame(section_f)
    checkbox_frame.pack(fill='x', pady=(0, 10))

    self.compare_methods = {
        'ds': tk.BooleanVar(value=True),
        'pds': tk.BooleanVar(value=True),
        'ctai': tk.BooleanVar(value=True),
        'nspfce': tk.BooleanVar(value=False),
        'tsr': tk.BooleanVar(value=True),
        'jypls-inv': tk.BooleanVar(value=False)
    }

    for method, var in self.compare_methods.items():
        ttk.Checkbutton(checkbox_frame, text=method.upper(),
                       variable=var).pack(side='left', padx=(0, 15))

    # Run comparison button
    self._create_accent_button(section_f, "Run Comparison",
                               self._run_method_comparison).pack(pady=(0, 10))

    # Results table
    self.comparison_results_tree = ttk.Treeview(
        section_f,
        columns=('Method', 'RMSE', 'R²', 'Bias', 'MAE', 'Time (s)'),
        show='headings',
        height=6
    )
    # Configure columns...
    self.comparison_results_tree.pack(fill='both', expand=True, pady=(10, 0))

    # Comparison plot frame
    self.comparison_plot_frame = ttk.Frame(section_f)
    self.comparison_plot_frame.pack(fill='both', expand=True, pady=(10, 0))

def _run_method_comparison(self):
    """Execute all selected methods and populate results table."""
    # Implementation...
```

---

## Phase 6: Documentation (Week 5-6)

### 6.1 User Documentation

**File:** `documentation/CALIBRATION_TRANSFER_GUIDE.md`

**Table of Contents:**
1. Introduction to Calibration Transfer
2. When to Use Each Method
3. Method Descriptions
   - DS (Direct Standardization)
   - PDS (Piecewise DS)
   - CTAI (Affine Invariance)
   - NS-PFCE (Parameter-Free)
   - TSR (Transfer Sample Regression)
   - JYPLS-inv (Joint-Y PLS)
4. Best Practices
5. Troubleshooting
6. Examples
7. References

**Example Section:**

```markdown
## When to Use Each Method

### Decision Tree

1. **Do you have paired transfer samples?**
   - **No** → Use **CTAI** (no standards required)
   - **Yes** → Continue to #2

2. **How many transfer samples do you have?**
   - **< 10** → Use **CTAI** or **NS-PFCE**
   - **10-15** → Use **TSR** or **JYPLS-inv**
   - **> 15** → Use **DS** or **PDS**

3. **Is computation time critical?**
   - **Yes** → Use **TSR** or **DS** (fastest)
   - **No** → Use **NS-PFCE** or **JYPLS-inv** (most accurate)

4. **Do you need automatic wavelength selection?**
   - **Yes** → Use **NS-PFCE with VCPA-IRIV**
   - **No** → Any method is suitable

### Performance Comparison (Literature-based)

| Method | RMSE (Corn) | Transfer Samples | Computation Time |
|--------|-------------|------------------|------------------|
| DS | 0.145 | 30+ | Fast |
| PDS | 0.132 | 30+ | Medium |
| CTAI | **0.118** | **0** | Fast |
| NS-PFCE | 0.122 | 0 | Slow |
| TSR | 0.125 | 12-13 | Fast |
| JYPLS-inv | 0.127 | 12-13 | Medium |

*Values are illustrative; actual performance depends on dataset*
```

### 6.2 API Documentation

**Enhanced docstrings with examples:**

```python
def estimate_ctai(X_master, X_slave, n_components=None):
    """
    Estimate Calibration Transfer based on Affine Invariance (CTAI).

    CTAI is a transfer standard-free method that leverages the affine
    invariance properties of spectral transformations between instruments.
    It achieves state-of-the-art performance without requiring paired samples.

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples, n_wavelengths)
        Master instrument spectra on common wavelength grid.
    X_slave : np.ndarray, shape (n_samples, n_wavelengths)
        Slave instrument spectra on common wavelength grid.
    n_components : int, optional
        Number of components for transformation. If None, automatically
        selected based on explained variance threshold (0.99).

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'M' : np.ndarray, shape (n_wavelengths, n_wavelengths)
            Affine transformation matrix
        - 'T' : np.ndarray, shape (n_wavelengths,)
            Translation vector
        - 'n_components' : int
            Number of components used
        - 'explained_variance' : float
            Fraction of variance explained
        - 'reconstruction_error' : float
            Mean squared reconstruction error on input data

    Examples
    --------
    >>> # Generate synthetic data with known transformation
    >>> X_master = np.random.randn(100, 200)
    >>> X_slave = 0.9 * X_master + 0.05  # Simple affine transform
    >>>
    >>> # Estimate CTAI transformation
    >>> params = estimate_ctai(X_master, X_slave)
    >>>
    >>> # Apply to new slave spectra
    >>> X_slave_new = np.random.randn(50, 200)
    >>> X_transferred = apply_ctai(X_slave_new, params)
    >>>
    >>> # Evaluate transfer quality
    >>> rmse = np.sqrt(np.mean((X_transferred - X_master) ** 2))
    >>> print(f"Transfer RMSE: {rmse:.4f}")

    References
    ----------
    .. [1] Fan, W., et al. (2019). "Calibration transfer based on
           affine invariance for near-infrared spectra."
           Analytical Methods, 11(7), 864-872.

    See Also
    --------
    apply_ctai : Apply CTAI transformation to new spectra
    estimate_nspfce : Alternative transfer standard-free method
    """
    pass
```

### 6.3 Tutorial Notebooks

**File:** `example/tutorial_advanced_calibration_transfer.ipynb`

**Notebook Outline:**
1. Setup and Data Loading
2. Walkthrough of Each Method
3. Method Comparison
4. Real-World Example
5. Troubleshooting Tips

### 6.4 Scientific References

**File:** `documentation/CALIBRATION_TRANSFER_REFERENCES.md`

**Key Papers to Include:**

```markdown
# Calibration Transfer - Scientific References

## Core Methods

### CTAI
1. Fan, W., et al. (2019). "Calibration transfer based on affine invariance
   for near-infrared spectra." *Analytical Methods*, 11(7), 864-872.
   DOI: 10.1039/C8AY02629G

### NS-PFCE
2. [Need to identify primary paper - search literature]

### TSR (Shenk-Westerhaus)
3. Shenk, J. S., & Westerhaus, M. O. (1991). "Population definition, sample
   selection, and calibration procedures for near infrared reflectance
   spectroscopy." *Crop Science*, 31(2), 469-474.

### JYPLS-inv
4. [Need to identify primary paper - search literature]

## Sample Selection Methods

### Kennard-Stone
5. Kennard, R. W., & Stone, L. A. (1969). "Computer aided design of
   experiments." *Technometrics*, 11(1), 137-148.

### DUPLEX
6. Snee, R. D. (1977). "Validation of regression models: methods and
   examples." *Technometrics*, 19(4), 415-428.

### SPXY
7. Galvão, R. K., et al. (2005). "A method for calibration and validation
   subset partitioning." *Talanta*, 67(4), 736-740.

## Wavelength Selection

### VCPA-IRIV
8. Yun, Y. H., et al. (2015). "An efficient method of wavelength interval
   selection based on random frog for multivariate spectral calibration."
   *Spectrochimica Acta Part A*, 148, 375-381.

### CARS
9. Li, H. D., et al. (2009). "Key wavelengths screening using competitive
   adaptive reweighted sampling method for multivariate calibration."
   *Analytica Chimica Acta*, 648(1), 77-84.

## Review Papers

10. Feudale, R. N., et al. (2002). "Transfer of multivariate calibration
    models: a review." *Chemometrics and Intelligent Laboratory Systems*,
    64(2), 181-192.

11. Malli, B., et al. (2017). "Standardisation/transfer of multivariate
    calibration models: a review." *Analytical and Bioanalytical Chemistry*,
    409(3), 815-828.
```

---

## Phase 7: Performance Optimization (Week 6)

### 7.1 Computational Efficiency

**Optimization Targets:**

1. **Matrix Operations**
   - Use NumPy vectorization
   - Avoid loops where possible
   - Use in-place operations when safe

2. **Memory Management**
   - Avoid unnecessary copies
   - Use memory-mapped arrays for large datasets
   - Clear intermediate results

3. **Parallel Processing**
   ```python
   # Example: Parallel TSR estimation
   from joblib import Parallel, delayed

   def estimate_tsr_parallel(X_master, X_slave, transfer_indices, n_jobs=-1):
       """Parallel version of TSR estimation."""
       n_wavelengths = X_master.shape[1]

       def fit_wavelength(i):
           x = X_slave[transfer_indices, i]
           y = X_master[transfer_indices, i]
           slope, bias = np.polyfit(x, y, 1)
           return slope, bias

       results = Parallel(n_jobs=n_jobs)(
           delayed(fit_wavelength)(i) for i in range(n_wavelengths)
       )

       slopes, biases = zip(*results)
       return {'slope': np.array(slopes), 'bias': np.array(biases)}
   ```

4. **Caching**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=32)
   def kennard_stone_cached(X_hash, n_samples):
       """Cached version to avoid recomputation."""
       pass
   ```

### 7.2 Profiling and Benchmarking

```python
# File: tests/test_performance.py

import time
import memory_profiler

def benchmark_all_methods():
    """Profile computational performance of all CT methods."""
    sizes = [100, 500, 1000, 5000]
    n_wavelengths = 200

    results = []

    for n_samples in sizes:
        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = np.random.randn(n_samples, n_wavelengths)

        methods = {
            'DS': lambda: estimate_ds(X_master, X_slave),
            'PDS': lambda: estimate_pds(X_master, X_slave),
            'CTAI': lambda: estimate_ctai(X_master, X_slave),
            'TSR': lambda: estimate_tsr(X_master, X_slave, np.arange(min(12, n_samples))),
        }

        for method_name, method_func in methods.items():
            start = time.time()
            params = method_func()
            elapsed = time.time() - start

            results.append({
                'method': method_name,
                'n_samples': n_samples,
                'time_seconds': elapsed
            })

    # Plot results
    # Save to CSV
```

---

## Phase 8: Quality Assurance (Week 6-7)

### 8.1 Code Review Checklist

- [ ] All functions have comprehensive docstrings
- [ ] Type hints are provided
- [ ] Input validation is implemented
- [ ] Edge cases are handled
- [ ] Error messages are informative
- [ ] Code follows PEP 8 style guide
- [ ] No code duplication
- [ ] Efficient algorithms are used
- [ ] Memory leaks are avoided
- [ ] Thread safety (if applicable)

### 8.2 Test Coverage

```bash
# Run coverage report
pytest --cov=src/spectral_predict --cov-report=html

# Target: > 90% coverage for calibration_transfer module
```

### 8.3 Integration Testing

```python
# File: tests/test_calibration_transfer_integration.py

def test_end_to_end_workflow():
    """Test complete workflow from data loading to prediction."""
    # 1. Load master model
    # 2. Load paired spectra
    # 3. Build transfer model (all methods)
    # 4. Save transfer model
    # 5. Load transfer model
    # 6. Apply to new spectra
    # 7. Make predictions
    # 8. Validate results
    pass

def test_gui_integration():
    """Test GUI integration for all CT methods."""
    # Use GUI testing framework
    # Simulate user interactions
    # Verify outputs
    pass
```

---

## Phase 9: Deployment & User Testing (Week 7)

### 9.1 Alpha Testing Checklist

- [ ] Install package in clean environment
- [ ] Run all example scripts
- [ ] Test GUI with real data
- [ ] Verify saved models load correctly
- [ ] Check cross-platform compatibility (Windows/Mac/Linux)
- [ ] Test with large datasets (> 10,000 samples)
- [ ] Validate benchmark results

### 9.2 User Acceptance Testing

**Test Scenarios:**

1. **Novice User**: Follow tutorial to build first transfer model
2. **Expert User**: Compare multiple methods on custom data
3. **Production User**: Integrate into automated workflow

### 9.3 Performance Benchmarks

**Success Criteria:**

| Metric | Target |
|--------|--------|
| CTAI RMSE improvement over DS | > 10% |
| NS-PFCE with VCPA-IRIV | Best overall performance |
| TSR with 12 samples | < 5% worse than full recalibration |
| JYPLS-inv accuracy | Comparable to TSR |
| Computation time (1000 samples) | < 10 seconds per method |
| GUI responsiveness | No freezing during estimation |

---

## Risk Assessment & Mitigation

### High-Risk Areas

1. **VCPA-IRIV Complexity**
   - **Risk**: Algorithm is complex and poorly documented
   - **Mitigation**: Implement simpler wavelength selection first (CARS, SPA)
   - **Fallback**: Use all wavelengths if selection fails

2. **NS-PFCE Implementation Uncertainty**
   - **Risk**: Limited public implementation details
   - **Mitigation**: Contact authors or use approximation
   - **Fallback**: Omit if unavailable, focus on other methods

3. **Computational Performance**
   - **Risk**: Methods may be slow on large datasets
   - **Mitigation**: Optimize with NumPy, parallelize where possible
   - **Fallback**: Implement progress bars, allow user to cancel

4. **Integration Complexity**
   - **Risk**: GUI may become cluttered with many options
   - **Mitigation**: Use collapsible sections, wizards, presets
   - **Fallback**: Advanced options in separate dialog

---

## Success Metrics

### Technical Metrics
- ✅ All 4 methods implemented and tested
- ✅ Test coverage > 90%
- ✅ Benchmark results match literature
- ✅ No performance regressions
- ✅ GUI is responsive and intuitive

### User Metrics
- ✅ Tutorial completion rate > 80%
- ✅ User satisfaction score > 4/5
- ✅ Bug reports < 5 in first month
- ✅ Documentation clarity score > 4/5

### Scientific Metrics
- ✅ CTAI achieves lowest RMSE on benchmark data
- ✅ NS-PFCE + VCPA-IRIV outperforms NS-PFCE alone
- ✅ TSR and JYPLS-inv achieve statistical equivalence
- ✅ Results reproducible across runs

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Research & Spec | Algorithm specs, architecture design |
| 2 | Core Implementation | Sample selection, TSR, initial tests |
| 3 | Advanced Methods | CTAI, wavelength selection foundations |
| 4 | Complex Methods | NS-PFCE, JYPLS-inv, comprehensive tests |
| 5 | GUI & Validation | GUI integration, benchmark testing |
| 6 | Optimization & Docs | Performance tuning, documentation |
| 7 | Testing & Deployment | QA, user testing, release prep |

**Total Estimated Effort:** 6-7 weeks (1 developer, full-time)

**Alternative (Phased Rollout):**
- **Phase 1 (Weeks 1-3)**: TSR + CTAI (highest impact, lowest risk)
- **Phase 2 (Weeks 4-5)**: Wavelength selection + NS-PFCE
- **Phase 3 (Week 6-7)**: JYPLS-inv + optimization

---

## Next Steps

1. **Approve this plan** or request modifications
2. **Prioritize methods** (all 4 or phased approach?)
3. **Allocate resources** (developer time, compute resources)
4. **Acquire benchmark datasets** (Corn, Tablet, Soil)
5. **Set up development branch** (`feature/advanced-calibration-transfer`)
6. **Begin Phase 1: Research**

---

## Open Questions for Discussion

1. Should we implement all 4 methods or prioritize subset?
2. Is VCPA-IRIV essential or can we use simpler wavelength selection?
3. Do we need real-time prediction or batch processing is sufficient?
4. Should we support custom transfer sample selection (manual)?
5. What level of parameter tuning should be exposed to users?
6. Should we implement automatic method recommendation?
7. Need access to literature for NS-PFCE and JYPLS-inv specifics?

---

## Appendix A: Terminology

**Calibration Transfer**: Process of adapting a calibration model from one instrument (master) to another (slave) without full recalibration.

**Transfer Standards**: Samples measured on both instruments to build transfer mapping.

**Affine Transformation**: Linear transformation plus translation: `Y = MX + T`

**Kennard-Stone**: Algorithm for selecting representative samples based on distance in feature space.

**VCPA-IRIV**: Variable selection method combining population analysis with iterative variable removal.

---

## Appendix B: Dependencies

**Required Python Packages:**
- NumPy >= 1.20
- SciPy >= 1.7
- scikit-learn >= 1.0
- matplotlib >= 3.3
- joblib >= 1.0 (for parallelization)

**Optional:**
- numba (for JIT compilation)
- cupy (for GPU acceleration)

---

**Document Version:** 1.0
**Date:** 2025-11-13
**Author:** DASP Development Team
**Status:** Draft - Awaiting Approval
