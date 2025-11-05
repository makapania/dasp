# Implementation Guide: Model Diagnostics Features for Spectral Predict GUI

**Target Application:** Spectral Predict GUI (Chemometrics Software)
**Estimated Effort:** 25-34 hours (4-5 days)
**Complexity:** Medium
**Purpose:** Add industry-standard model diagnostics to bring software to parity with commercial packages

---

## Executive Summary

This guide provides complete instructions for implementing 4 critical diagnostic features:

1. **Residual Plots** (3 plot types) - Assess model fit quality
2. **Leverage Plot** - Identify influential samples
3. **Prediction Intervals** - Quantify prediction uncertainty (jack-knife method)
4. **MSC Preprocessing** - Multiplicative Scatter Correction

All features integrate into **Tab 6 (Custom Model Development)** of the GUI.

---

## Current Application Context

### File Structure
```
dasp/
├── spectral_predict_gui_optimized.py    # Main GUI (4,800 lines)
├── src/spectral_predict/
│   ├── search.py                        # Search engine
│   ├── models.py                        # Model definitions
│   ├── preprocess.py                    # Preprocessing (151 lines)
│   ├── variable_selection.py            # Variable selection
│   └── outlier_detection.py             # Outlier detection
└── tests/                               # Unit tests
```

### Key Information
- **GUI Framework:** tkinter with ttk styling
- **Plotting:** matplotlib with TkAgg backend (`FigureCanvasTkAgg`)
- **Models:** PLS, Ridge, Lasso, Random Forest, MLP, Neural Boosted
- **Current Tab 6 Location:** Lines 998-1187 in `spectral_predict_gui_optimized.py`
- **Model Run Thread:** `_run_refined_model_thread()` at line 3233
- **Dependencies:** All required (numpy, scipy, matplotlib, sklearn)

### Critical Design Patterns
1. **Background threading:** Model runs in separate thread to keep UI responsive
2. **Pipeline architecture:** Preprocessing + model in sklearn Pipeline
3. **Instance variables:** Store results in `self.refined_*` attributes
4. **Plot embedding:** Use `FigureCanvasTkAgg` to embed matplotlib in tkinter

---

## FEATURE 1: RESIDUAL DIAGNOSTIC PLOTS

### Objective
Add 3 residual plots to assess model fit quality for regression models.

### 1.1 Create Diagnostics Module

**Create new file:** `src/spectral_predict/diagnostics.py`

```python
"""
Model diagnostics utilities for spectral analysis.

Provides:
- Residual computation (raw and standardized)
- Leverage calculation (hat values for influential samples)
- Q-Q plot data for normality assessment
- Prediction intervals (jack-knife method)
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def compute_residuals(y_true, y_pred):
    """
    Compute residuals for regression models.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values
    y_pred : array-like, shape (n_samples,)
        Predicted values

    Returns
    -------
    residuals : ndarray
        y_true - y_pred
    standardized_residuals : ndarray
        Residuals divided by their standard deviation
    """
    residuals = np.array(y_true) - np.array(y_pred)
    std_resid = residuals / np.std(residuals) if np.std(residuals) > 1e-10 else residuals
    return residuals, std_resid


def compute_leverage(X, return_threshold=True):
    """
    Compute leverage (hat values) for samples.

    Leverage h_ii = diag(X(X'X)^-1X')
    High leverage points have h_ii > 2p/n or 3p/n (thresholds)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (preprocessed data used for model fitting)
    return_threshold : bool, default=True
        If True, also return leverage thresholds

    Returns
    -------
    leverage : ndarray, shape (n_samples,)
        Hat values for each sample
    threshold_2p : float (optional)
        2p/n threshold for moderate leverage
    threshold_3p : float (optional)
        3p/n threshold for high leverage

    Notes
    -----
    For large n_features, uses SVD-based approach for numerical stability.
    """
    X = np.asarray(X)
    n, p = X.shape

    # Use SVD for numerical stability when p is large
    if p > 100:
        # H = U @ U.T where U comes from SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        leverage = np.sum(U**2, axis=1)
    else:
        # Standard formula: H = X(X'X)^-1X'
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            H = X @ XtX_inv @ X.T
            leverage = np.diag(H)
        except np.linalg.LinAlgError:
            # Fallback to SVD if matrix is singular
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            leverage = np.sum(U**2, axis=1)

    if return_threshold:
        threshold_2p = 2 * p / n
        threshold_3p = 3 * p / n
        return leverage, threshold_2p, threshold_3p

    return leverage


def qq_plot_data(residuals):
    """
    Compute Q-Q plot coordinates for normality assessment.

    Parameters
    ----------
    residuals : array-like
        Model residuals

    Returns
    -------
    theoretical_quantiles : ndarray
        Expected quantiles from normal distribution
    sample_quantiles : ndarray
        Observed quantiles from residuals (sorted)
    """
    from scipy import stats

    residuals = np.asarray(residuals)
    sample_quantiles = np.sort(residuals)

    # Compute theoretical quantiles
    n = len(residuals)
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(1/(n+1), n/(n+1), n)
    )

    return theoretical_quantiles, sample_quantiles


def jackknife_prediction_intervals(model, X_train, y_train, X_test, confidence=0.95):
    """
    Compute prediction intervals using jack-knife (leave-one-out) resampling.

    Faster than bootstrap for small-to-moderate sample sizes.
    Suitable for PLS regression models.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model (e.g., PLSRegression)
    X_train : array-like, shape (n_train, n_features)
        Training features
    y_train : array-like, shape (n_train,)
        Training targets
    X_test : array-like, shape (n_test, n_features)
        Test features for prediction
    confidence : float, default=0.95
        Confidence level (0.95 = 95% interval)

    Returns
    -------
    predictions : ndarray, shape (n_test,)
        Point predictions for X_test
    lower_bounds : ndarray, shape (n_test,)
        Lower confidence bounds
    upper_bounds : ndarray, shape (n_test,)
        Upper confidence bounds
    std_errors : ndarray, shape (n_test,)
        Standard errors of predictions

    Notes
    -----
    Uses delete-1 jackknife:
    1. For each training sample i, fit model on data excluding sample i
    2. Predict on X_test with this model
    3. Compute variance across jackknife replications
    4. Construct intervals using t-distribution

    Computational cost: O(n_train * fit_time)
    WARNING: Can be slow for n_train > 200
    """
    from scipy import stats
    from sklearn.base import clone

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Get point predictions from full model
    predictions = model.predict(X_test).flatten()

    # Jackknife resampling: leave-one-out predictions
    jackknife_preds = np.zeros((n_train, n_test))

    for i in range(n_train):
        # Create leave-one-out dataset
        mask = np.ones(n_train, dtype=bool)
        mask[i] = False

        X_loo = X_train[mask]
        y_loo = y_train[mask]

        # Clone and fit model
        model_loo = clone(model)
        model_loo.fit(X_loo, y_loo)

        # Predict on test set
        jackknife_preds[i, :] = model_loo.predict(X_test).flatten()

    # Compute jackknife variance
    # Variance = (n-1)/n * sum((theta_i - theta_mean)^2)
    mean_preds = np.mean(jackknife_preds, axis=0)
    jackknife_var = ((n_train - 1) / n_train) * np.sum(
        (jackknife_preds - mean_preds)**2, axis=0
    )
    std_errors = np.sqrt(jackknife_var)

    # Construct confidence intervals using t-distribution
    # df = n_train - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n_train - 1)

    lower_bounds = predictions - t_critical * std_errors
    upper_bounds = predictions + t_critical * std_errors

    return predictions, lower_bounds, upper_bounds, std_errors
```

**Save this file as:** `src/spectral_predict/diagnostics.py`

---

### 1.2 Add Residual Plots to GUI

**File to modify:** `spectral_predict_gui_optimized.py`

#### Step 1: Add UI Frame for Residual Plots

**Location:** Line 1182 (after `self.refine_plot_frame` creation)

**Add this code:**

```python
        # Residual Diagnostics (regression only)
        ttk.Label(content_frame, text="Residual Diagnostics", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        diagnostics_frame = ttk.LabelFrame(content_frame, text="Residual Analysis", padding="20")
        diagnostics_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.residual_diagnostics_frame = ttk.Frame(diagnostics_frame)
        self.residual_diagnostics_frame.pack(fill='both', expand=True)
```

#### Step 2: Create Plotting Method

**Location:** After line 3223 (after `_plot_refined_predictions` method)

**Add this method:**

```python
    def _plot_residual_diagnostics(self):
        """
        Plot three residual diagnostic plots in Tab 6.

        Creates:
        1. Residuals vs Fitted Values (detect heteroscedasticity)
        2. Residuals vs Sample Index (detect patterns/trends)
        3. Q-Q Plot (assess normality assumption)

        Only shown for regression tasks after model run completes.
        """
        if not HAS_MATPLOTLIB:
            return

        # Only for regression
        if not hasattr(self, 'refined_config') or self.refined_config.get('task_type') != 'regression':
            return

        if not hasattr(self, 'refined_y_true') or not hasattr(self, 'refined_y_pred'):
            return

        from spectral_predict.diagnostics import compute_residuals, qq_plot_data

        # Clear existing plot
        for widget in self.residual_diagnostics_frame.winfo_children():
            widget.destroy()

        y_true = self.refined_y_true
        y_pred = self.refined_y_pred

        # Compute residuals
        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # Create 1x3 subplot figure
        fig = Figure(figsize=(18, 5))

        # Plot 1: Residuals vs Fitted
        ax1 = fig.add_subplot(131)
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=40)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Fitted Values', fontsize=10)
        ax1.set_ylabel('Residuals', fontsize=10)
        ax1.set_title('Residuals vs Fitted', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals vs Index
        ax2 = fig.add_subplot(132)
        indices = np.arange(len(residuals))
        ax2.scatter(indices, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=40)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Sample Index', fontsize=10)
        ax2.set_ylabel('Residuals', fontsize=10)
        ax2.set_title('Residuals vs Index', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Q-Q Plot
        ax3 = fig.add_subplot(133)
        theoretical_q, sample_q = qq_plot_data(residuals)
        ax3.scatter(theoretical_q, sample_q, alpha=0.6, edgecolors='black', linewidths=0.5, s=40)

        # Add reference line
        min_q = min(theoretical_q.min(), sample_q.min())
        max_q = max(theoretical_q.max(), sample_q.max())
        ax3.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=2)

        ax3.set_xlabel('Theoretical Quantiles', fontsize=10)
        ax3.set_ylabel('Sample Quantiles', fontsize=10)
        ax3.set_title('Q-Q Plot (Normality)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.residual_diagnostics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
```

#### Step 3: Call from Update Results

**Location:** Line 3706 (in `_update_refined_results` method)

**Modify existing line:**

```python
            # Plot the predictions
            self._plot_refined_predictions()
            self._plot_residual_diagnostics()  # ADD THIS LINE
            messagebox.showinfo("Success", "Refined model analysis complete!")
```

---

## FEATURE 2: LEVERAGE PLOT

### Objective
Add leverage plot to identify influential samples (only for linear models).

### 2.1 Add UI Frame for Leverage Plot

**File:** `spectral_predict_gui_optimized.py`
**Location:** After residual diagnostics frame creation (after the code you added in 1.2 Step 1)

**Add:**

```python
        # Leverage Diagnostics (linear models only)
        ttk.Label(content_frame, text="Leverage Analysis", style='Heading.TLabel').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(25, 15))
        row += 1

        leverage_frame = ttk.LabelFrame(content_frame, text="Influential Samples (Hat Values)", padding="20")
        leverage_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.leverage_plot_frame = ttk.Frame(leverage_frame)
        self.leverage_plot_frame.pack(fill='both', expand=True)
```

### 2.2 Store X Data for Leverage Calculation

**Location:** Line 3670 (in `_run_refined_model_thread`, after storing `refined_y_pred`)

**Add:**

```python
            # Store predictions for plotting
            self.refined_y_true = np.array(all_y_true)
            self.refined_y_pred = np.array(all_y_pred)

            # Store X data for leverage calculation (ADD THIS)
            self.refined_X_cv = X_raw.copy()
```

### 2.3 Create Leverage Plotting Method

**Location:** After `_plot_residual_diagnostics` method

**Add:**

```python
    def _plot_leverage_diagnostics(self):
        """
        Plot leverage (hat values) to identify influential samples.

        Shows:
        - Hat values vs sample index
        - Threshold lines for moderate (2p/n) and high (3p/n) leverage
        - Labels for high-leverage samples

        Only shown for linear models (PLS, Ridge, Lasso).
        """
        if not HAS_MATPLOTLIB:
            return

        # Only for regression with linear/PLS models
        if not hasattr(self, 'refined_config'):
            return

        task_type = self.refined_config.get('task_type')
        model_name = self.refined_config.get('model_name')

        # Leverage only meaningful for linear models (PLS, Ridge, Lasso)
        if task_type != 'regression' or model_name not in ['PLS', 'Ridge', 'Lasso']:
            return

        if not hasattr(self, 'refined_X_cv') or self.refined_X_cv is None:
            return  # Need X data for leverage calculation

        from spectral_predict.diagnostics import compute_leverage

        # Clear existing plot
        for widget in self.leverage_plot_frame.winfo_children():
            widget.destroy()

        # Compute leverage on the CV data
        X_data = self.refined_X_cv
        leverage, threshold_2p, threshold_3p = compute_leverage(X_data)

        # Create figure
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # Determine colors based on leverage thresholds
        colors = []
        for h in leverage:
            if h > threshold_3p:
                colors.append('red')  # High leverage
            elif h > threshold_2p:
                colors.append('orange')  # Moderate leverage
            else:
                colors.append('steelblue')  # Normal

        indices = np.arange(len(leverage))
        ax.scatter(indices, leverage, c=colors, alpha=0.7, edgecolors='black', linewidths=0.5, s=60)

        # Add threshold lines
        ax.axhline(y=threshold_2p, color='orange', linestyle='--', linewidth=2,
                   label=f'Moderate Leverage (2p/n = {threshold_2p:.3f})')
        ax.axhline(y=threshold_3p, color='red', linestyle='--', linewidth=2,
                   label=f'High Leverage (3p/n = {threshold_3p:.3f})')

        # Label high-leverage points
        high_leverage_indices = np.where(leverage > threshold_3p)[0]
        for idx in high_leverage_indices:
            ax.annotate(f'{idx}', (idx, leverage[idx]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Leverage (Hat Values)', fontsize=11)
        ax.set_title('Leverage Plot - Influential Samples', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Add info text
        n_high = np.sum(leverage > threshold_3p)
        n_moderate = np.sum((leverage > threshold_2p) & (leverage <= threshold_3p))
        info_text = f'High leverage: {n_high} samples\nModerate leverage: {n_moderate} samples'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.leverage_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
```

### 2.4 Call Leverage Plot

**Location:** Line 3706 (after residual diagnostics call)

**Modify:**

```python
            # Plot the predictions
            self._plot_refined_predictions()
            self._plot_residual_diagnostics()
            self._plot_leverage_diagnostics()  # ADD THIS LINE
            messagebox.showinfo("Success", "Refined model analysis complete!")
```

---

## FEATURE 3: MSC PREPROCESSING

### Objective
Add Multiplicative Scatter Correction as a preprocessing option.

### 3.1 Add MSC Class to Preprocessing Module

**File:** `src/spectral_predict/preprocess.py`
**Location:** After SNV class (around line 42)

**Add this class:**

```python
class MSC(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction (MSC).

    Corrects for light scattering effects by fitting each spectrum to a reference spectrum
    using a linear regression model: spectrum_i = a + b * reference + noise
    The corrected spectrum is: (spectrum_i - a) / b

    Parameters
    ----------
    reference : {'mean', 'median'} or array-like, default='mean'
        Reference spectrum to use for correction.
        - 'mean': Use mean spectrum of calibration set
        - 'median': Use median spectrum of calibration set
        - array: Use provided spectrum as reference
    """

    def __init__(self, reference='mean'):
        self.reference = reference
        self.reference_spectrum_ = None

    def fit(self, X, y=None):
        """
        Fit MSC by computing reference spectrum.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data
        y : ignored

        Returns
        -------
        self
        """
        X = np.asarray(X)

        if isinstance(self.reference, str):
            if self.reference == 'mean':
                self.reference_spectrum_ = np.mean(X, axis=0)
            elif self.reference == 'median':
                self.reference_spectrum_ = np.median(X, axis=0)
            else:
                raise ValueError(f"Unknown reference type: {self.reference}")
        else:
            # User-provided reference
            self.reference_spectrum_ = np.asarray(self.reference)
            if self.reference_spectrum_.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Reference spectrum has {self.reference_spectrum_.shape[0]} features "
                    f"but X has {X.shape[1]} features"
                )

        return self

    def transform(self, X):
        """
        Apply MSC transformation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_msc : ndarray, shape (n_samples, n_features)
            MSC-corrected spectra
        """
        X = np.asarray(X)

        if self.reference_spectrum_ is None:
            raise ValueError("MSC must be fitted before transform")

        X_msc = np.zeros_like(X)

        for i in range(X.shape[0]):
            # Fit linear model: spectrum_i = a + b * reference
            # Use numpy polyfit for speed
            coeffs = np.polyfit(self.reference_spectrum_, X[i, :], 1)
            b, a = coeffs  # polyfit returns [slope, intercept]

            # Protect against division by zero
            if abs(b) < 1e-6:
                b = 1.0

            # Correct: (spectrum - a) / b
            X_msc[i, :] = (X[i, :] - a) / b

        return X_msc
```

### 3.2 Update Pipeline Builder

**File:** `src/spectral_predict/preprocess.py`
**Location:** Line 109 (`build_preprocessing_pipeline` function)

**Replace the entire function with:**

```python
def build_preprocessing_pipeline(preprocess_name, deriv=None, window=None, polyorder=None):
    """
    Build a preprocessing pipeline from a configuration.

    Parameters
    ----------
    preprocess_name : str
        One of: 'raw', 'snv', 'msc', 'deriv', 'snv_deriv', 'msc_deriv', 'deriv_snv', 'deriv_msc'
    deriv : int, optional
        Derivative order (for deriv-based pipelines)
    window : int, optional
        Window size (for deriv-based pipelines)
    polyorder : int, optional
        Polynomial order (for deriv-based pipelines)

    Returns
    -------
    steps : list
        List of (name, transformer) tuples
    """
    from sklearn.pipeline import Pipeline

    if preprocess_name == "raw":
        return []

    elif preprocess_name == "snv":
        return [("snv", SNV())]

    elif preprocess_name == "msc":
        return [("msc", MSC())]

    elif preprocess_name == "deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol)]

    elif preprocess_name == "snv_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("snv", SNV()), ("savgol", savgol)]

    elif preprocess_name == "msc_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("msc", MSC()), ("savgol", savgol)]

    elif preprocess_name == "deriv_snv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol), ("snv", SNV())]

    elif preprocess_name == "deriv_msc":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol), ("msc", MSC())]

    else:
        raise ValueError(f"Unknown preprocess: {preprocess_name}")
```

### 3.3 Update GUI Preprocessing Dropdown

**File:** `spectral_predict_gui_optimized.py`
**Location:** Line 1133

**Replace:**

```python
preprocess_combo['values'] = ['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']
```

**With:**

```python
preprocess_combo['values'] = ['raw', 'snv', 'msc', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'msc_sg1', 'msc_sg2', 'deriv_snv', 'deriv_msc']
```

### 3.4 Update Preprocessing Mappings

**File:** `spectral_predict_gui_optimized.py`
**Location:** Line 3332 (in `_run_refined_model_thread` method)

**Find these three dictionaries and update them:**

```python
            # Map GUI preprocessing names to search.py format
            preprocess_name_map = {
                'raw': 'raw',
                'snv': 'snv',
                'msc': 'msc',  # ADD
                'sg1': 'deriv',
                'sg2': 'deriv',
                'snv_sg1': 'snv_deriv',
                'snv_sg2': 'snv_deriv',
                'msc_sg1': 'msc_deriv',  # ADD
                'msc_sg2': 'msc_deriv',  # ADD
                'deriv_snv': 'deriv_snv',
                'deriv_msc': 'deriv_msc'  # ADD
            }

            deriv_map = {
                'raw': 0,
                'snv': 0,
                'msc': 0,  # ADD
                'sg1': 1,
                'sg2': 2,
                'snv_sg1': 1,
                'snv_sg2': 2,
                'msc_sg1': 1,  # ADD
                'msc_sg2': 2,  # ADD
                'deriv_snv': 1,
                'deriv_msc': 1  # ADD
            }

            polyorder_map = {
                'raw': 2,
                'snv': 2,
                'msc': 2,  # ADD
                'sg1': 2,
                'sg2': 3,
                'snv_sg1': 2,
                'snv_sg2': 3,
                'msc_sg1': 2,  # ADD
                'msc_sg2': 3,  # ADD
                'deriv_snv': 2,
                'deriv_msc': 2  # ADD
            }
```

### 3.5 Update Derivative Detection

**Location:** Line 3391 (same method)

**Find:**

```python
is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']
```

**Replace with:**

```python
is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'msc_sg1', 'msc_sg2', 'deriv_snv', 'deriv_msc']
```

---

## FEATURE 4: PREDICTION INTERVALS (JACK-KNIFE)

### Objective
Add prediction intervals for PLS models using jack-knife resampling.

**Note:** This is the most complex feature. The jack-knife function is already in the diagnostics module (created in Feature 1).

### 4.1 Compute Intervals During CV

**File:** `spectral_predict_gui_optimized.py`
**Location:** Line 3670 (in `_run_refined_model_thread`, after storing predictions)

**Add this block:**

```python
            # Store predictions for plotting
            self.refined_y_true = np.array(all_y_true)
            self.refined_y_pred = np.array(all_y_pred)

            # Store X data for leverage calculation
            self.refined_X_cv = X_raw.copy()

            # Compute prediction intervals for PLS models (ADD THIS ENTIRE BLOCK)
            if model_name == 'PLS' and task_type == 'regression' and len(X_raw) < 300:
                # Only compute for PLS regression with reasonable sample size
                # Skip if n > 300 (too slow)
                from spectral_predict.diagnostics import jackknife_prediction_intervals

                print("DEBUG: Computing prediction intervals using jack-knife method...")
                all_intervals = []

                # Recompute CV to get intervals for each fold
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_raw, y_array)):
                    X_train, X_test = X_raw[train_idx], X_raw[test_idx]
                    y_train, y_test = y_array[train_idx], y_array[test_idx]

                    # Clone and fit pipeline
                    pipe_fold = clone(pipe)
                    pipe_fold.fit(X_train, y_train)

                    # Extract model from pipeline (last step)
                    model_fold = pipe_fold.named_steps['model']

                    # Compute intervals (this is the slow part)
                    try:
                        _, lower, upper, std_err = jackknife_prediction_intervals(
                            model_fold, X_train, y_train, X_test, confidence=0.95
                        )

                        all_intervals.append({
                            'test_idx': test_idx,
                            'lower': lower,
                            'upper': upper,
                            'std_err': std_err
                        })
                    except Exception as e:
                        print(f"WARNING: Failed to compute intervals for fold {fold_idx}: {e}")
                        all_intervals = None
                        break

                self.refined_prediction_intervals = all_intervals
                print("DEBUG: Prediction intervals computed successfully.")
            else:
                self.refined_prediction_intervals = None
```

### 4.2 Display Intervals in Results Text

**Location:** Line 3599 (in results_text formatting, after MAE line)

**Find the regression results block and add interval info:**

```python
                results_text = f"""Refined Model Results:

Cross-Validation Performance ({self.refine_folds.get()} folds):
  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}
  R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}
  MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}
"""

                # ADD THIS BLOCK FOR PREDICTION INTERVALS
                if hasattr(self, 'refined_prediction_intervals') and self.refined_prediction_intervals is not None:
                    # Compute average interval width
                    all_widths = []
                    for fold_data in self.refined_prediction_intervals:
                        widths = fold_data['upper'] - fold_data['lower']
                        all_widths.extend(widths)
                    avg_width = np.mean(all_widths)

                    results_text += f"""
Prediction Intervals (95% Confidence):
  Average Interval Width: ±{avg_width/2:.4f}
  Method: Jackknife (leave-one-out)
  Note: Intervals shown as error bars in prediction plot
"""

                # Rest of results_text continues here...
                results_text += f"""
COMPARISON TO LOADED MODEL:
  Original R² (from Results tab): {loaded_r2}
  ...
```

### 4.3 Add Error Bars to Prediction Plot

**Location:** Line 3194 (in `_plot_refined_predictions` method)

**Find the scatter plot line and add error bars after it:**

```python
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidths=0.5, s=50)

        # ADD ERROR BARS IF INTERVALS AVAILABLE
        if hasattr(self, 'refined_prediction_intervals') and self.refined_prediction_intervals is not None:
            # Reconstruct full interval arrays from CV folds
            n_samples = len(y_true)
            lower_full = np.zeros(n_samples)
            upper_full = np.zeros(n_samples)

            for fold_data in self.refined_prediction_intervals:
                test_idx = fold_data['test_idx']
                lower_full[test_idx] = fold_data['lower']
                upper_full[test_idx] = fold_data['upper']

            # Add error bars to plot
            ax.errorbar(y_true, y_pred,
                        yerr=[y_pred - lower_full, upper_full - y_pred],
                        fmt='none', ecolor='gray', alpha=0.3, linewidth=0.5, capsize=2)

        # 1:1 line (rest of plot continues)
        min_val = min(y_true.min(), y_pred.min())
```

---

## TESTING & VALIDATION

### Unit Tests

**Create:** `tests/test_diagnostics.py`

```python
import numpy as np
import pytest
from src.spectral_predict.diagnostics import (
    compute_residuals,
    compute_leverage,
    qq_plot_data,
)


def test_compute_residuals():
    """Test residual computation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    residuals, std_resid = compute_residuals(y_true, y_pred)

    assert len(residuals) == 5
    assert np.allclose(residuals, y_true - y_pred)
    assert np.abs(np.std(std_resid) - 1.0) < 0.01  # Standardized should have std ≈ 1


def test_compute_leverage_small_p():
    """Test leverage with p < n."""
    X = np.random.randn(50, 10)
    leverage, thresh_2p, thresh_3p = compute_leverage(X)

    assert len(leverage) == 50
    assert thresh_2p == 20/50
    assert thresh_3p == 30/50
    assert np.all(leverage >= 0)
    assert np.all(leverage <= 1)


def test_compute_leverage_large_p():
    """Test leverage with large p (uses SVD)."""
    X = np.random.randn(50, 150)
    leverage, thresh_2p, thresh_3p = compute_leverage(X)

    assert len(leverage) == 50
    # Should work even with p > n


def test_qq_plot_data():
    """Test Q-Q plot data generation."""
    residuals = np.random.randn(100)
    theoretical_q, sample_q = qq_plot_data(residuals)

    assert len(theoretical_q) == 100
    assert len(sample_q) == 100
    # Sample quantiles should be sorted
    assert np.all(sample_q[:-1] <= sample_q[1:])
```

**Create:** `tests/test_msc.py`

```python
import numpy as np
from src.spectral_predict.preprocess import MSC


def test_msc_transform():
    """Test MSC preprocessing."""
    X = np.random.randn(20, 100) + 5  # Add offset

    msc = MSC(reference='mean')
    msc.fit(X)
    X_corrected = msc.transform(X)

    assert X_corrected.shape == X.shape
    # Mean spectrum should be approximately zero-centered after correction
    mean_corrected = np.mean(X_corrected, axis=0)
    assert np.abs(np.mean(mean_corrected)) < 0.5


def test_msc_with_median():
    """Test MSC with median reference."""
    X = np.random.randn(20, 100)

    msc = MSC(reference='median')
    msc.fit(X)
    X_corrected = msc.transform(X)

    assert X_corrected.shape == X.shape
```

### Manual Testing Checklist

**After implementing all features, test the following scenarios:**

#### Residual Plots
- [ ] Load data and run Model Development with PLS model
- [ ] Verify 3 residual plots appear
- [ ] Check Q-Q plot shows normality (or lack thereof)
- [ ] Residual vs Fitted shows no obvious pattern (good model)
- [ ] Plots do NOT appear for classification tasks

#### Leverage Plot
- [ ] Run PLS model - leverage plot should appear
- [ ] Run Ridge model - leverage plot should appear
- [ ] Run Random Forest model - leverage plot should NOT appear (non-linear)
- [ ] Verify high-leverage points are labeled
- [ ] Check threshold lines are visible

#### MSC Preprocessing
- [ ] Select 'msc' from dropdown
- [ ] Run model - should complete without errors
- [ ] Try 'msc_sg1' (MSC + 1st derivative) - should work
- [ ] Try 'msc_sg2' (MSC + 2nd derivative) - should work
- [ ] Try 'deriv_msc' (derivative then MSC) - should work

#### Prediction Intervals
- [ ] Run PLS model with n < 300 samples
- [ ] Verify "Computing prediction intervals..." appears in console
- [ ] Check results text shows "Prediction Intervals (95% Confidence)"
- [ ] Verify error bars appear on prediction plot
- [ ] Test with n > 300 - intervals should be skipped (too slow)
- [ ] Verify Ridge/Lasso models do NOT compute intervals

#### Integration Testing
- [ ] Run analysis with excluded samples - plots should reflect excluded data
- [ ] Run with validation set enabled - all features should work
- [ ] Switch between different model types - appropriate plots shown/hidden
- [ ] Verify all plots update when re-running model
- [ ] Check that plots don't interfere with model saving

---

## TROUBLESHOOTING

### Common Issues

**Issue:** "Module 'spectral_predict.diagnostics' not found"
**Solution:** Ensure `diagnostics.py` is in `src/spectral_predict/` directory

**Issue:** Plots not appearing
**Solution:** Check that `HAS_MATPLOTLIB = True` at top of GUI file (line 36)

**Issue:** Leverage calculation fails (singular matrix)
**Solution:** SVD fallback should handle this automatically; check that code path

**Issue:** Jack-knife taking too long
**Solution:** Verify the n < 300 check is working (line 3673 in added code)

**Issue:** MSC division by zero
**Solution:** Check the `if abs(b) < 1e-6` protection is in place (MSC transform method)

**Issue:** Error bars not showing on plot
**Solution:** Verify `refined_prediction_intervals` is being stored correctly

---

## COMPLETION CHECKLIST

After implementation, verify ALL items:

### Code Files
- [ ] Created `src/spectral_predict/diagnostics.py` (~250 lines)
- [ ] Modified `src/spectral_predict/preprocess.py` (added MSC class)
- [ ] Modified `spectral_predict_gui_optimized.py` (added 4 methods, 2 UI frames)
- [ ] Created `tests/test_diagnostics.py`
- [ ] Created `tests/test_msc.py`

### Features Working
- [ ] Residual plots display for regression
- [ ] Leverage plot displays for PLS/Ridge/Lasso
- [ ] MSC preprocessing option available and functional
- [ ] Prediction intervals compute for PLS (n < 300)
- [ ] Error bars show on prediction plot when available

### UI/UX
- [ ] All new plots embedded cleanly in Tab 6
- [ ] Plots scroll properly in tab
- [ ] No performance degradation in UI responsiveness
- [ ] Status messages appear during interval computation

### Testing
- [ ] All unit tests pass
- [ ] Manual testing completed for all scenarios
- [ ] Edge cases handled (small n, large p, non-linear models)

---

## ESTIMATED TIMELINE

- **Day 1 (6-8 hours):** Create diagnostics module + residual plots
- **Day 2 (4-5 hours):** Add leverage plot to GUI
- **Day 3 (2-3 hours):** Implement MSC preprocessing
- **Day 4 (4-5 hours):** Add prediction interval computation
- **Day 5 (4-6 hours):** Testing, bug fixes, and documentation

**Total: 20-27 hours (3-5 days)**

---

## FINAL NOTES

### Architecture Principles Maintained
- Background threading for long computations ✓
- sklearn Pipeline architecture ✓
- Instance variable storage pattern ✓
- matplotlib FigureCanvasTkAgg embedding ✓

### Industry Standards Achieved
After implementation, the application will have:
- Comprehensive residual diagnostics (like Unscrambler X)
- Leverage analysis (like PLS_Toolbox)
- Prediction uncertainty quantification (like SIMCA)
- MSC preprocessing (standard in all packages)

### Performance Considerations
- Leverage: Fast (O(n²p) or O(np²), cached SVD)
- Residuals: Very fast (O(n))
- Jack-knife: Slow (O(n² * fit_time)), limited to n < 300
- MSC: Fast (O(np))

### Future Enhancements
After this implementation, consider:
1. Cook's Distance plot (combines leverage + residuals)
2. Analytical prediction intervals for PLS (faster than jack-knife)
3. Bootstrap intervals as alternative to jack-knife
4. Residual distribution histogram

---

## CONTACT & SUPPORT

If you encounter issues during implementation:
1. Check the TROUBLESHOOTING section
2. Verify all line numbers match (they may shift during editing)
3. Ensure all imports are present at top of files
4. Run unit tests to isolate problems

**Good luck with the implementation! This will bring your spectral analysis software to professional-grade status.** 🚀
