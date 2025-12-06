"""
spectral_predict_v3.core.calibration_transfer
==============================================

Calibration transfer methods for transferring calibration models between instruments.

Supported methods:
- DS (Direct Standardization)
- PDS (Piecewise Direct Standardization)
- TSR (Transfer Sample Regression / Shenk-Westerhaus)
- CTAI (Calibration Transfer based on Affine Invariance)
- NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement)
- JYPLS-inv (Joint-Y PLS with Inversion)

Enhanced with VCPA-IRIV wavelength selection for NS-PFCE method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np


MethodType = Literal["ds", "pds", "tsr", "ctai", "nspfce", "jypls-inv"]


@dataclass
class TransferModel:
    """
    Encapsulates a calibration transfer mapping from a slave instrument
    to a master instrument on a common wavelength grid.
    """
    master_id: str
    slave_id: str
    method: MethodType
    wavelengths_common: np.ndarray
    params: Dict
    meta: Dict = field(default_factory=dict)


def resample_to_grid(
    X: np.ndarray,
    wl_src: np.ndarray,
    wl_target: np.ndarray,
) -> np.ndarray:
    """
    Resample spectra from wl_src grid to wl_target grid using 1D interpolation.

    Parameters
    ----------
    X : np.ndarray
        Spectra of shape (n_samples, n_src_wavelengths).
    wl_src : np.ndarray
        Source wavelengths, shape (n_src_wavelengths,).
    wl_target : np.ndarray
        Target wavelengths, shape (n_target_wavelengths,).

    Returns
    -------
    np.ndarray
        Resampled spectra of shape (n_samples, n_target_wavelengths).
    """
    from scipy.interpolate import interp1d

    n_samples = X.shape[0]
    n_target = wl_target.shape[0]
    X_resampled = np.zeros((n_samples, n_target))

    for i in range(n_samples):
        interpolator = interp1d(wl_src, X[i, :],
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        X_resampled[i, :] = interpolator(wl_target)

    return X_resampled


# ==============================================================================
# DS (Direct Standardization)
# ==============================================================================

def estimate_ds(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    lam: float = 0.0,
) -> np.ndarray:
    """
    Estimate a Direct Standardization (DS) matrix A such that:
        X_slave @ A ≈ X_master

    Parameters
    ----------
    X_master : np.ndarray
        Master instrument spectra on common grid, shape (n_samples, p).
    X_slave : np.ndarray
        Slave instrument spectra on common grid, shape (n_samples, p).
    lam : float
        Optional ridge regularization parameter.

    Returns
    -------
    np.ndarray
        DS matrix A of shape (p, p).
    """
    p = X_slave.shape[1]
    XtX = X_slave.T @ X_slave

    if lam > 0:
        XtX += lam * np.eye(p)

    XtY = X_slave.T @ X_master
    A = np.linalg.solve(XtX, XtY)

    return A


def apply_ds(X_slave_new: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Apply a previously estimated DS matrix A to new slave spectra.

    Returns
    -------
    np.ndarray
        Transformed spectra in master instrument domain.
    """
    return X_slave_new @ A


# ==============================================================================
# PDS (Piecewise Direct Standardization)
# ==============================================================================

def estimate_pds(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    window: int = 11,
) -> np.ndarray:
    """
    Estimate Piecewise Direct Standardization (PDS) coefficients B.

    Parameters
    ----------
    X_master : np.ndarray
        Master spectra on common grid, shape (n_samples, p).
    X_slave : np.ndarray
        Slave spectra on common grid, shape (n_samples, p).
    window : int
        Window size (odd integer) for local regression around each wavelength.

    Returns
    -------
    np.ndarray
        PDS coefficient array B of shape (p, window).
    """
    n_samples, p = X_slave.shape
    half_window = window // 2

    B = np.zeros((p, window))

    for i in range(p):
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        X_window = X_slave[:, start:end]
        y_target = X_master[:, i]

        try:
            b = np.linalg.lstsq(X_window, y_target, rcond=None)[0]
            offset = start - (i - half_window)
            B[i, offset:offset + len(b)] = b
        except np.linalg.LinAlgError:
            center = half_window
            if 0 <= center < window:
                B[i, center] = 1.0

    return B


def apply_pds(
    X_slave_new: np.ndarray,
    B: np.ndarray,
    window: int = 11,
) -> np.ndarray:
    """
    Apply previously estimated PDS coefficients B to new slave spectra.

    Returns
    -------
    np.ndarray
        Transformed spectra in master instrument domain.
    """
    n_samples, p = X_slave_new.shape
    half_window = window // 2

    X_transformed = np.zeros_like(X_slave_new)

    for i in range(p):
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        X_window = X_slave_new[:, start:end]
        offset = start - (i - half_window)
        b = B[i, offset:offset + X_window.shape[1]]

        X_transformed[:, i] = X_window @ b

    return X_transformed


# ==============================================================================
# TSR (Transfer Sample Regression / Shenk-Westerhaus)
# ==============================================================================

def estimate_tsr(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    transfer_indices: np.ndarray,
    slope_bias_correction: bool = True,
    regularization: float = 0.0,
) -> Dict:
    """
    Estimate Transfer Sample Regression (TSR / Shenk-Westerhaus method).

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples, n_wavelengths)
        Master instrument spectra on common wavelength grid.
    X_slave : np.ndarray, shape (n_samples, n_wavelengths)
        Slave instrument spectra on common wavelength grid.
    transfer_indices : np.ndarray, shape (n_transfer,)
        Indices of samples to use for building transfer mapping.
    slope_bias_correction : bool, default=True
        If True, apply full slope + bias correction.
    regularization : float, default=0.0
        Ridge regularization parameter for regression.

    Returns
    -------
    params : dict
        Dictionary containing slope, bias, r_squared, and other metrics.
    """
    n_samples, n_wavelengths = X_master.shape

    if X_slave.shape != X_master.shape:
        raise ValueError(
            f"X_master and X_slave must have same shape: "
            f"{X_master.shape} vs {X_slave.shape}"
        )

    if len(transfer_indices) < 2:
        raise ValueError(f"Need at least 2 transfer samples, got {len(transfer_indices)}")

    if transfer_indices.max() >= n_samples:
        raise ValueError(
            f"transfer_indices contains index {transfer_indices.max()} "
            f"but only {n_samples} samples available"
        )

    X_master_transfer = X_master[transfer_indices]
    X_slave_transfer = X_slave[transfer_indices]
    n_transfer = len(transfer_indices)

    slopes = np.ones(n_wavelengths) if not slope_bias_correction else np.zeros(n_wavelengths)
    biases = np.zeros(n_wavelengths)
    r_squared = np.zeros(n_wavelengths)

    for i in range(n_wavelengths):
        x = X_slave_transfer[:, i]
        y = X_master_transfer[:, i]

        if slope_bias_correction:
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2) + regularization

            if denominator > 1e-10:
                slope = numerator / denominator
                bias = y_mean - slope * x_mean
            else:
                slope = 1.0
                bias = y_mean - x_mean

            slopes[i] = slope
            biases[i] = bias

            y_pred = slope * x + bias
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)

            if ss_tot > 1e-10:
                r_squared[i] = 1 - (ss_res / ss_tot)
            else:
                r_squared[i] = 1.0

        else:
            slopes[i] = 1.0
            biases[i] = np.mean(y - x)

            y_pred = x + biases[i]
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot > 1e-10:
                r_squared[i] = 1 - (ss_res / ss_tot)
            else:
                r_squared[i] = 1.0

    params = {
        'slope': slopes,
        'bias': biases,
        'transfer_indices': transfer_indices,
        'r_squared': r_squared,
        'mean_r_squared': np.mean(r_squared),
        'wavelength_quality': r_squared,
        'n_transfer_samples': n_transfer,
        'slope_bias_correction': slope_bias_correction
    }

    return params


def apply_tsr(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply TSR calibration transfer to new slave instrument spectra.

    Parameters
    ----------
    X_slave_new : np.ndarray, shape (n_samples, n_wavelengths)
        New slave instrument spectra to transform.
    params : dict
        TSR parameters from estimate_tsr.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in master instrument domain.
    """
    slope = params['slope']
    bias = params['bias']

    n_wavelengths = len(slope)
    if X_slave_new.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_slave_new has {X_slave_new.shape[1]} wavelengths "
            f"but model expects {n_wavelengths}"
        )

    X_transferred = X_slave_new * slope + bias

    return X_transferred


# ==============================================================================
# CTAI (Calibration Transfer based on Affine Invariance)
# ==============================================================================

def estimate_ctai(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    n_components: int | None = None,
    explained_variance_threshold: float = 0.99,
) -> Dict:
    """
    Estimate CTAI (Calibration Transfer based on Affine Invariance).

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples, n_wavelengths)
        Master instrument spectra on common wavelength grid.
    X_slave : np.ndarray, shape (n_samples, n_wavelengths)
        Slave instrument spectra on common wavelength grid.
    n_components : int, optional
        Number of principal components to use.
    explained_variance_threshold : float, default=0.99
        Fraction of variance to retain when auto-selecting n_components.

    Returns
    -------
    params : dict
        Dictionary containing transformation matrix M, translation T, and metrics.
    """
    from scipy.linalg import svd

    n_samples_master, n_wavelengths = X_master.shape
    n_samples_slave = X_slave.shape[0]

    print("\n=== CTAI Debug Information ===")
    print(f"  Input shapes: Master {X_master.shape}, Slave {X_slave.shape}")

    if X_slave.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_master and X_slave must have same number of wavelengths: "
            f"{n_wavelengths} vs {X_slave.shape[1]}"
        )

    if n_samples_master < 2 or n_samples_slave < 2:
        raise ValueError("Need at least 2 samples in both master and slave datasets")

    if np.any(np.isnan(X_master)):
        raise ValueError(f"X_master contains {np.sum(np.isnan(X_master))} NaN values")
    if np.any(np.isinf(X_master)):
        raise ValueError(f"X_master contains {np.sum(np.isinf(X_master))} infinite values")
    if np.any(np.isnan(X_slave)):
        raise ValueError(f"X_slave contains {np.sum(np.isnan(X_slave))} NaN values")
    if np.any(np.isinf(X_slave)):
        raise ValueError(f"X_slave contains {np.sum(np.isinf(X_slave))} infinite values")

    print(f"  Data validation: PASSED")
    print(f"  Master data range: [{np.min(X_master):.6f}, {np.max(X_master):.6f}]")
    print(f"  Slave data range: [{np.min(X_slave):.6f}, {np.max(X_slave):.6f}]")

    master_mean = np.mean(X_master, axis=0)
    slave_mean = np.mean(X_slave, axis=0)

    X_master_centered = X_master - master_mean
    X_slave_centered = X_slave - slave_mean

    print(f"  Step 1: Mean centering complete")

    have_paired_samples = (n_samples_master == n_samples_slave)

    if not have_paired_samples:
        raise ValueError(
            f"CTAI requires paired samples (same samples on both instruments).\n"
            f"Got {n_samples_master} master samples and {n_samples_slave} slave samples.\n"
            f"For unpaired samples, use TSR, DS, or PDS instead."
        )

    print(f"  Detected paired samples ({n_samples_master} samples on both instruments)")
    print(f"  Step 2: Computing SVD of slave data...")

    try:
        U_slave, S_slave, Vt_slave = svd(X_slave_centered, full_matrices=False)
        print(f"    SVD successful: U{U_slave.shape}, S{S_slave.shape}, Vt{Vt_slave.shape}")
        print(f"    Singular values range: [{np.min(S_slave):.6e}, {np.max(S_slave):.6e}]")
        if S_slave[-1] > 0:
            print(f"    Condition number: {S_slave[0]/S_slave[-1]:.2e}")
    except np.linalg.LinAlgError as e:
        raise ValueError(f"SVD failed: {e}")

    if n_components is None:
        explained_var_cumsum = np.cumsum(S_slave**2) / np.sum(S_slave**2)
        n_components = np.searchsorted(explained_var_cumsum, explained_variance_threshold) + 1
        n_components = min(n_components, min(n_samples_slave, n_wavelengths))
        print(f"  Step 3: Auto-selected {n_components} components (threshold={explained_variance_threshold})")
    else:
        n_components = min(n_components, len(S_slave))
        print(f"  Step 3: Using {n_components} components (user-specified)")

    V_truncated = Vt_slave[:n_components, :].T
    S_truncated = S_slave[:n_components]

    X_slave_projected = X_slave_centered @ V_truncated
    X_master_projected = X_master_centered @ V_truncated

    print(f"  Step 4: Computing transformation in {n_components}-D PC space...")

    M_reduced = np.linalg.lstsq(X_slave_projected, X_master_projected, rcond=None)[0]
    print(f"    M_reduced shape: {M_reduced.shape}")

    M = V_truncated @ M_reduced @ V_truncated.T

    print(f"    M shape: {M.shape}")
    print(f"    M range: [{np.min(M):.6f}, {np.max(M):.6f}]")
    print(f"    M diagonal mean: {np.mean(np.diag(M)):.6f}")

    if np.any(np.isnan(M)):
        raise ValueError("Transformation matrix M contains NaN values!")
    if np.any(np.isinf(M)):
        raise ValueError("Transformation matrix M contains infinite values!")

    T = master_mean - slave_mean @ M

    print(f"    T (translation) range: [{np.min(T):.6f}, {np.max(T):.6f}]")

    print(f"  Step 5: Validating transformation quality...")
    X_slave_transformed = X_slave @ M + T

    if np.any(np.isnan(X_slave_transformed)):
        raise ValueError("Transformed data contains NaN values!")
    if np.any(np.isinf(X_slave_transformed)):
        raise ValueError("Transformed data contains infinite values!")

    print(f"    Transformed data range: [{np.min(X_slave_transformed):.6f}, {np.max(X_slave_transformed):.6f}]")

    X_master_sample = X_master[:min(n_samples_master, n_samples_slave)]
    X_slave_sample = X_slave_transformed[:min(n_samples_master, n_samples_slave)]

    reconstruction_error = np.sqrt(np.mean((X_master_sample - X_slave_sample) ** 2))
    explained_variance = np.sum(S_truncated**2) / np.sum(S_slave**2) if len(S_slave) > 0 else 1.0

    print(f"\n  === CTAI Results ===")
    print(f"  Components: {n_components}")
    print(f"  Explained Variance: {explained_variance:.4f}")
    print(f"  Reconstruction RMSE: {reconstruction_error:.6f}")
    print(f"  ==================\n")

    params = {
        'M': M,
        'T': T,
        'n_components': n_components,
        'explained_variance': explained_variance,
        'reconstruction_error': reconstruction_error,
        'master_mean': master_mean,
        'slave_mean': slave_mean,
        'eigenvalues': S_truncated,
    }

    return params


def apply_ctai(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply CTAI calibration transfer to new slave instrument spectra.

    Parameters
    ----------
    X_slave_new : np.ndarray, shape (n_samples, n_wavelengths)
        New slave instrument spectra to transform.
    params : dict
        CTAI parameters from estimate_ctai.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in master instrument domain.
    """
    M = params['M']
    T = params['T']

    n_wavelengths = M.shape[0]
    if X_slave_new.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_slave_new has {X_slave_new.shape[1]} wavelengths "
            f"but model expects {n_wavelengths}"
        )

    X_transferred = X_slave_new @ M + T

    return X_transferred


# ==============================================================================
# VCPA-IRIV (Variable Combination Population Analysis - IRIV)
# ==============================================================================

def vcpa_iriv(
    X: np.ndarray,
    y: np.ndarray,
    n_outer_iterations: int = 10,
    n_inner_iterations: int = 50,
    pls_components: int = 5,
    cv_folds: int = 5,
    importance_threshold: float = 0.5,
    random_state: int | None = None
) -> Dict:
    """
    Variable Combination Population Analysis - Iteratively Retains
    Informative Variables (VCPA-IRIV).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)
        Spectral data matrix.
    y : np.ndarray, shape (n_samples,)
        Target values.
    n_outer_iterations : int, default=10
        Number of IRIV outer iterations.
    n_inner_iterations : int, default=50
        Number of BM sampling iterations per outer iteration.
    pls_components : int, default=5
        Number of PLS components.
    cv_folds : int, default=5
        Cross-validation folds.
    importance_threshold : float, default=0.5
        Threshold for removing low-importance variables.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with selected_indices, importance_scores, convergence_history.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_wavelengths = X.shape

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    active_indices = np.arange(n_wavelengths)
    convergence_history = []
    n_vars_history = []

    for outer_iter in range(n_outer_iterations):
        n_active = len(active_indices)
        n_vars_history.append(n_active)

        if n_active <= pls_components:
            print(f"  VCPA-IRIV: Stopped at iteration {outer_iter} (too few variables)")
            break

        importance_scores = np.zeros(n_active)

        for inner_iter in range(n_inner_iterations):
            inclusion_prob = 0.9 * (1 - outer_iter / n_outer_iterations) + 0.2

            binary_vector = np.random.rand(n_active) < inclusion_prob
            n_selected = np.sum(binary_vector)

            if n_selected <= pls_components:
                continue

            selected_vars = active_indices[binary_vector]
            X_subset = X[:, selected_vars]

            try:
                pls = PLSRegression(n_components=min(pls_components, n_selected-1))

                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_errors = []

                for train_idx, val_idx in kf.split(X_subset):
                    X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    pls.fit(X_train, y_train)
                    y_pred = pls.predict(X_val)
                    mse = np.mean((y_val - y_pred.ravel()) ** 2)
                    cv_errors.append(mse)

                rmsecv = np.sqrt(np.mean(cv_errors))

                if rmsecv < np.inf:
                    weight = 1.0 / (rmsecv + 1e-10)
                    importance_scores[binary_vector] += weight

            except Exception:
                continue

        if importance_scores.sum() > 0:
            importance_scores = importance_scores / importance_scores.sum()

        current_rmsecv = 1.0 / (importance_scores.mean() + 1e-10) if importance_scores.mean() > 0 else np.inf
        convergence_history.append(current_rmsecv)

        threshold_value = importance_threshold * importance_scores.max()
        keep_mask = importance_scores >= threshold_value

        if np.sum(keep_mask) <= pls_components:
            print(f"  VCPA-IRIV: Stopped removal at iteration {outer_iter}")
            break

        active_indices = active_indices[keep_mask]

        if np.sum(keep_mask) == n_active:
            print(f"  VCPA-IRIV: Converged at iteration {outer_iter}")
            break

    X_final = X[:, active_indices]

    try:
        pls_final = PLSRegression(n_components=min(pls_components, len(active_indices)-1))

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_errors = []

        for train_idx, val_idx in kf.split(X_final):
            X_train, X_val = X_final[train_idx], X_final[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pls_final.fit(X_train, y_train)
            y_pred = pls_final.predict(X_val)
            mse = np.mean((y_val - y_pred.ravel()) ** 2)
            cv_errors.append(mse)

        final_rmsecv = np.sqrt(np.mean(cv_errors))

    except Exception:
        final_rmsecv = np.inf

    result = {
        'selected_indices': active_indices,
        'importance_scores': importance_scores if len(importance_scores) == len(active_indices) else np.ones(len(active_indices)),
        'convergence_history': convergence_history,
        'n_vars_history': n_vars_history,
        'final_rmsecv': final_rmsecv,
        'n_selected': len(active_indices)
    }

    return result


# ==============================================================================
# NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement)
# ==============================================================================

def estimate_nspfce(
    X_master: np.ndarray,
    X_slave: np.ndarray,
    wavelengths: np.ndarray,
    use_wavelength_selection: bool = True,
    wavelength_selector: str = 'vcpa-iriv',
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    normalize: bool = True
) -> Dict:
    """
    Non-supervised Parameter-Free Calibration Enhancement (NS-PFCE).

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples_master, n_wavelengths)
        Master instrument spectra.
    X_slave : np.ndarray, shape (n_samples_slave, n_wavelengths)
        Slave instrument spectra.
    wavelengths : np.ndarray, shape (n_wavelengths,)
        Wavelength grid.
    use_wavelength_selection : bool, default=True
        Whether to apply wavelength selection.
    wavelength_selector : str, default='vcpa-iriv'
        Method for wavelength selection: 'vcpa-iriv', 'cars', or 'spa'.
    max_iterations : int, default=100
        Maximum iterations for optimization.
    convergence_threshold : float, default=1e-6
        Convergence criterion.
    normalize : bool, default=True
        Apply adaptive normalization.

    Returns
    -------
    params : dict
        Dictionary with transformation_matrix, selected_wavelengths, and metrics.
    """
    n_samples_master, n_wavelengths = X_master.shape
    n_samples_slave = X_slave.shape[0]

    if X_slave.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_master and X_slave must have same number of wavelengths: "
            f"{n_wavelengths} vs {X_slave.shape[1]}"
        )

    if wavelengths.shape[0] != n_wavelengths:
        raise ValueError(
            f"wavelengths must have same length as spectral dimension: "
            f"{wavelengths.shape[0]} vs {n_wavelengths}"
        )

    selected_wavelengths = None

    if use_wavelength_selection:
        print(f"  NS-PFCE: Performing wavelength selection using {wavelength_selector}...")

        y_pseudo_master = np.mean(X_master, axis=1)

        try:
            if wavelength_selector == 'vcpa-iriv':
                wl_result = vcpa_iriv(
                    X_master, y_pseudo_master,
                    n_outer_iterations=5,
                    n_inner_iterations=30,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown wavelength_selector: {wavelength_selector}")

            selected_wavelengths = wl_result['selected_indices']
            print(f"  NS-PFCE: Selected {len(selected_wavelengths)}/{n_wavelengths} wavelengths")

            X_master_sel = X_master[:, selected_wavelengths]
            X_slave_sel = X_slave[:, selected_wavelengths]
            n_selected = len(selected_wavelengths)

        except Exception as e:
            print(f"  NS-PFCE: Wavelength selection failed ({str(e)}), using all wavelengths")
            X_master_sel = X_master
            X_slave_sel = X_slave
            n_selected = n_wavelengths
            selected_wavelengths = np.arange(n_wavelengths)

    else:
        X_master_sel = X_master
        X_slave_sel = X_slave
        n_selected = n_wavelengths
        selected_wavelengths = np.arange(n_wavelengths)

    master_mean = np.mean(X_master_sel, axis=0)
    slave_mean = np.mean(X_slave_sel, axis=0)
    master_std = np.std(X_master_sel, axis=0) + 1e-10
    slave_std = np.std(X_slave_sel, axis=0) + 1e-10

    scale_factors = master_std / slave_std
    T = np.diag(scale_factors)
    offset = master_mean - slave_mean * scale_factors

    convergence_iterations = 0
    objective_history = []

    for iteration in range(max_iterations):
        X_slave_transformed = X_slave_sel @ T + offset
        n_compare = min(n_samples_master, n_samples_slave, 100)
        obj = np.mean((X_master_sel[:n_compare] - X_slave_transformed[:n_compare]) ** 2)
        objective_history.append(obj)

        if iteration > 0:
            obj_change = abs(objective_history[-1] - objective_history[-2])
            if obj_change < convergence_threshold:
                convergence_iterations = iteration
                break

        reg_param = 1e-6
        XtX = X_slave_sel.T @ X_slave_sel + reg_param * np.eye(n_selected)
        XtY = X_slave_sel.T @ X_master_sel

        try:
            T_new = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            T_new = np.linalg.pinv(X_slave_sel) @ X_master_sel

        damping = 0.5
        T = damping * T_new + (1 - damping) * T

        X_slave_transformed = X_slave_sel @ T
        offset = np.mean(X_master_sel - X_slave_transformed, axis=0)

        if normalize and iteration % 10 == 0:
            scale = np.diag(T)
            scale_mean = np.mean(scale)
            if scale_mean > 0:
                T = T / scale_mean
                offset = offset * scale_mean

    if convergence_iterations == 0:
        convergence_iterations = max_iterations

    X_slave_transformed = X_slave_sel @ T + offset
    n_compare = min(n_samples_master, n_samples_slave)
    final_objective = np.sqrt(np.mean((X_master_sel[:n_compare] - X_slave_transformed[:n_compare]) ** 2))

    print(f"  NS-PFCE: Converged in {convergence_iterations} iterations")
    print(f"  NS-PFCE: Final RMSE: {final_objective:.6f}")

    params = {
        'transformation_matrix': T,
        'T': T,
        'offset': offset,
        'selected_wavelengths': selected_wavelengths,
        'wavelength_selector': wavelength_selector if use_wavelength_selection else None,
        'use_wavelength_selection': use_wavelength_selection,
        'n_selected_wavelengths': n_selected,
        'convergence_iterations': convergence_iterations,
        'n_iterations': convergence_iterations,
        'converged': (convergence_iterations < max_iterations),
        'final_objective': final_objective,
        'objective_history': objective_history,
        'convergence_history': objective_history
    }

    return params


def apply_nspfce(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply NS-PFCE calibration transfer to new slave instrument spectra.

    Parameters
    ----------
    X_slave_new : np.ndarray, shape (n_samples, n_wavelengths)
        New slave instrument spectra to transform.
    params : dict
        NS-PFCE parameters from estimate_nspfce.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in master instrument domain.
    """
    T = params['transformation_matrix']
    offset = params['offset']
    selected_wavelengths = params['selected_wavelengths']
    use_wl_selection = params['use_wavelength_selection']

    if use_wl_selection and selected_wavelengths is not None:
        X_slave_selected = X_slave_new[:, selected_wavelengths]
        X_transformed_selected = X_slave_selected @ T + offset

        X_transferred = X_slave_new.copy()
        X_transferred[:, selected_wavelengths] = X_transformed_selected

    else:
        n_wavelengths_expected = T.shape[0]
        if X_slave_new.shape[1] != n_wavelengths_expected:
            raise ValueError(
                f"X_slave_new has {X_slave_new.shape[1]} wavelengths "
                f"but model expects {n_wavelengths_expected}"
            )

        X_transferred = X_slave_new @ T + offset

    return X_transferred


# ==============================================================================
# JYPLS-inv (Joint-Y PLS with Inversion)
# ==============================================================================

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
    Estimate JYPLS-inv (Joint-Y PLS with inversion) calibration transfer.

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples, n_wavelengths)
        Master instrument spectra.
    X_slave : np.ndarray, shape (n_samples, n_wavelengths)
        Slave instrument spectra.
    y_transfer : np.ndarray, shape (n_transfer,)
        Reference values for transfer samples.
    transfer_indices : np.ndarray, shape (n_transfer,)
        Indices of transfer samples.
    n_components : int | None, optional
        Number of PLS components.
    cv_folds : int, optional
        Number of cross-validation folds.
    max_components : int, optional
        Maximum number of components to try.

    Returns
    -------
    params : dict
        Dictionary with transformation_matrix and PLS metrics.
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError("scikit-learn is required for JYPLS-inv")

    if X_master.shape != X_slave.shape:
        raise ValueError(f"X_master and X_slave must have same shape")

    if len(transfer_indices) != len(y_transfer):
        raise ValueError(f"Number of transfer_indices must match y_transfer length")

    if len(transfer_indices) < 2:
        raise ValueError("Need at least 2 transfer samples for JYPLS-inv")

    n_samples, n_wavelengths = X_master.shape

    X_master_transfer = X_master[transfer_indices]
    X_slave_transfer = X_slave[transfer_indices]

    X_aug = np.vstack([X_master_transfer, X_slave_transfer])
    Y_aug = np.concatenate([y_transfer, y_transfer]).reshape(-1, 1)

    if n_components is None:
        best_rmse = np.inf
        best_n = 1

        max_comp = min(max_components, len(transfer_indices) - 1, n_wavelengths)

        for n in range(1, max_comp + 1):
            pls = PLSRegression(n_components=n, scale=False)

            try:
                scores = cross_val_score(
                    pls, X_aug, Y_aug, cv=min(cv_folds, len(transfer_indices)),
                    scoring='neg_root_mean_squared_error'
                )
                avg_rmse = -np.mean(scores)

                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_n = n
            except Exception:
                break

        n_components = best_n
        cv_rmse = best_rmse
    else:
        cv_rmse = None

    max_comp = min(len(transfer_indices) - 1, n_wavelengths)
    if n_components > max_comp:
        n_components = max_comp

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_aug, Y_aug)

    T_all = pls.transform(X_aug)
    n_transfer = len(transfer_indices)
    T_master = T_all[:n_transfer, :]
    T_slave = T_all[n_transfer:, :]

    W = pls.x_weights_
    P = pls.x_loadings_

    T_slave_T = T_slave.T
    T_slave_cov = T_slave_T @ T_slave + 1e-6 * np.eye(n_components)
    T_slave_cov_inv = np.linalg.inv(T_slave_cov)
    M_scores = T_slave_cov_inv @ (T_slave_T @ T_master)

    transformation_matrix = W @ M_scores @ P.T

    X_var = np.var(X_aug, axis=0).sum()
    X_reconstructed = pls.inverse_transform(T_all)
    X_residual_var = np.var(X_aug - X_reconstructed, axis=0).sum()
    explained_variance_ratio = 1.0 - (X_residual_var / X_var)

    params = {
        'transformation_matrix': transformation_matrix,
        'n_components': n_components,
        'cv_rmse': cv_rmse if cv_rmse is not None else 0.0,
        'transfer_indices': transfer_indices,
        'pls_x_weights': W,
        'pls_x_loadings': P,
        'pls_scores_master': T_master,
        'pls_scores_slave': T_slave,
        'score_transformation': M_scores,
        'explained_variance_ratio': explained_variance_ratio,
        'n_transfer_samples': len(transfer_indices)
    }

    return params


def apply_jypls_inv(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply JYPLS-inv transformation to new slave spectra.

    Parameters
    ----------
    X_slave_new : np.ndarray, shape (n_samples, n_wavelengths)
        New slave spectra to transfer.
    params : dict
        Parameters from estimate_jypls_inv.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transferred spectra in master domain.
    """
    B = params['transformation_matrix']

    if X_slave_new.shape[1] != B.shape[0]:
        raise ValueError(
            f"X_slave_new has {X_slave_new.shape[1]} wavelengths, "
            f"but transformation matrix expects {B.shape[0]}"
        )

    X_transferred = X_slave_new @ B

    return X_transferred


# ==============================================================================
# Unified Dispatcher
# ==============================================================================

def apply_transfer_dispatch(X_slave: np.ndarray, transfer_model: TransferModel) -> np.ndarray:
    """
    Unified dispatcher for applying any transfer model type.

    Parameters
    ----------
    X_slave : np.ndarray
        Slave instrument spectra to transform.
    transfer_model : TransferModel
        Transfer model object.

    Returns
    -------
    np.ndarray
        Transformed spectra in master instrument space.
    """
    method = transfer_model.method.lower()
    params = transfer_model.params

    if method == 'ds':
        return apply_ds(X_slave, params['A'])
    elif method == 'pds':
        return apply_pds(X_slave, params['B'], params['window'])
    elif method == 'tsr':
        return apply_tsr(X_slave, params)
    elif method == 'ctai':
        return apply_ctai(X_slave, params)
    elif method == 'ns-pfce' or method == 'nspfce':
        return apply_nspfce(X_slave, params)
    elif method == 'jypls-inv':
        return apply_jypls_inv(X_slave, params)
    else:
        raise ValueError(
            f"Unknown transfer method: {method}. "
            f"Supported methods are: ds, pds, tsr, ctai, ns-pfce, jypls-inv"
        )


# ==============================================================================
# Save/Load
# ==============================================================================

def save_transfer_model(
    transfer_model: TransferModel,
    directory: Path | str,
    name: str | None = None,
) -> Path:
    """
    Save a TransferModel to disk using JSON for metadata and NPZ for arrays.

    Parameters
    ----------
    transfer_model : TransferModel
        The model to save.
    directory : Path or str
        Target directory.
    name : str, optional
        Optional base filename.

    Returns
    -------
    Path
        Path prefix for the saved model.
    """
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"{transfer_model.master_id}_from_{transfer_model.slave_id}_{transfer_model.method}"

    path_prefix = directory / name

    metadata = {
        "master_id": transfer_model.master_id,
        "slave_id": transfer_model.slave_id,
        "method": transfer_model.method,
        "meta": transfer_model.meta,
    }

    arrays_to_save = {
        "wavelengths_common": transfer_model.wavelengths_common,
    }

    for key, value in transfer_model.params.items():
        if isinstance(value, np.ndarray):
            arrays_to_save[f"param_{key}"] = value
        elif isinstance(value, (int, float)):
            metadata[f"param_{key}"] = value

    np.savez(f"{path_prefix}.npz", **arrays_to_save)

    with open(f"{path_prefix}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return path_prefix


def load_transfer_model(path_prefix: Path | str) -> TransferModel:
    """
    Load a TransferModel previously saved.

    Parameters
    ----------
    path_prefix : Path or str
        Path prefix (without extension).

    Returns
    -------
    TransferModel
    """
    import json

    path_prefix = Path(path_prefix)

    with open(f"{path_prefix}.json", "r") as f:
        metadata = json.load(f)

    arrays = np.load(f"{path_prefix}.npz")

    wavelengths_common = arrays["wavelengths_common"]

    params = {}
    for key in arrays.keys():
        if key.startswith("param_"):
            param_name = key[6:]
            params[param_name] = arrays[key]

    for key, value in metadata.items():
        if key.startswith("param_"):
            param_name = key[6:]
            params[param_name] = value

    return TransferModel(
        master_id=metadata["master_id"],
        slave_id=metadata["slave_id"],
        method=metadata["method"],
        wavelengths_common=wavelengths_common,
        params=params,
        meta=metadata.get("meta", {}),
    )
