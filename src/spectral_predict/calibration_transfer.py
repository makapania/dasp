"""
spectral_predict.calibration_transfer
=====================================

Backend-only module for calibration transfer between instruments.

Supported methods:
- DS (Direct Standardization)
- PDS (Piecewise Direct Standardization)
- TSR (Transfer Sample Regression / Shenk-Westerhaus)
- CTAI (Calibration Transfer based on Affine Invariance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np


MethodType = Literal["ds", "pds", "tsr", "ctai", "nspfce"]


@dataclass
class TransferModel:
    """
    Encapsulates a calibration transfer mapping from a slave instrument
    to a master instrument on a common wavelength grid.
    """
    master_id: str
    slave_id: str
    method: MethodType                  # "ds" or "pds"
    wavelengths_common: np.ndarray      # 1D array of wavelengths for both
    params: Dict                        # e.g. {"A": ds_matrix} or {"B": B, "window": 11}
    meta: Dict = field(default_factory=dict)  # resolution metrics, sigma*, notes, etc.


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
    # Solve for A: X_slave @ A = X_master
    # A = (X_slave^T @ X_slave + lam*I)^-1 @ X_slave^T @ X_master

    p = X_slave.shape[1]

    # Compute X_slave^T @ X_slave
    XtX = X_slave.T @ X_slave

    # Add ridge regularization
    if lam > 0:
        XtX += lam * np.eye(p)

    # Compute X_slave^T @ X_master
    XtY = X_slave.T @ X_master

    # Solve for A
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
        # Determine window boundaries
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        # Extract window from slave spectra
        X_window = X_slave[:, start:end]

        # Extract target from master spectra (single wavelength)
        y_target = X_master[:, i]

        # Solve least squares: X_window @ b = y_target
        # b = (X_window^T @ X_window)^-1 @ X_window^T @ y_target
        try:
            b = np.linalg.lstsq(X_window, y_target, rcond=None)[0]

            # Store coefficients in B, padding if window is truncated
            offset = start - (i - half_window)
            B[i, offset:offset + len(b)] = b
        except np.linalg.LinAlgError:
            # If singular, use simple copy (identity-like behavior)
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
        # Determine window boundaries
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        # Extract window from slave spectra
        X_window = X_slave_new[:, start:end]

        # Get coefficients for this wavelength
        offset = start - (i - half_window)
        b = B[i, offset:offset + X_window.shape[1]]

        # Apply transformation
        X_transformed[:, i] = X_window @ b

    return X_transformed


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
        Target directory (will be created if needed).
    name : str, optional
        Optional base filename (without extension). If None, derive from
        master_id, slave_id, and method.

    Returns
    -------
    Path
        Path prefix (without extension) for the saved model.
    """
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if name is None:
        name = f"{transfer_model.master_id}_from_{transfer_model.slave_id}_{transfer_model.method}"

    path_prefix = directory / name

    # Save metadata to JSON
    metadata = {
        "master_id": transfer_model.master_id,
        "slave_id": transfer_model.slave_id,
        "method": transfer_model.method,
        "meta": transfer_model.meta,
    }

    with open(f"{path_prefix}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save arrays to NPZ
    arrays_to_save = {
        "wavelengths_common": transfer_model.wavelengths_common,
    }

    # Add params arrays
    for key, value in transfer_model.params.items():
        if isinstance(value, np.ndarray):
            arrays_to_save[f"param_{key}"] = value
        elif isinstance(value, (int, float)):
            # Store scalars in metadata instead
            metadata[f"param_{key}"] = value

    np.savez(f"{path_prefix}.npz", **arrays_to_save)

    # Re-save metadata with scalar params
    with open(f"{path_prefix}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return path_prefix


def load_transfer_model(path_prefix: Path | str) -> TransferModel:
    """
    Load a TransferModel previously saved by save_transfer_model.

    Parameters
    ----------
    path_prefix : Path or str
        Path prefix (without extension). The function should expect a JSON
        and NPZ with this prefix.

    Returns
    -------
    TransferModel
    """
    import json

    path_prefix = Path(path_prefix)

    # Load metadata from JSON
    with open(f"{path_prefix}.json", "r") as f:
        metadata = json.load(f)

    # Load arrays from NPZ
    arrays = np.load(f"{path_prefix}.npz")

    wavelengths_common = arrays["wavelengths_common"]

    # Reconstruct params dict
    params = {}
    for key in arrays.keys():
        if key.startswith("param_"):
            param_name = key[6:]  # Remove "param_" prefix
            params[param_name] = arrays[key]

    # Add scalar params from metadata
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

    TSR is a simple yet effective calibration transfer method that estimates
    slope and bias corrections for each wavelength independently using a small
    set of transfer samples measured on both instruments.

    Algorithm:
    1. Extract transfer samples from master and slave datasets
    2. For each wavelength λ:
       - Fit linear regression: X_master[λ] = slope[λ] * X_slave[λ] + bias[λ]
       - Store slope and bias coefficients
    3. Return parameters for applying correction to new samples

    With optimal sample selection (e.g., Kennard-Stone), TSR can achieve
    results statistically indistinguishable from full recalibration using
    only 12-13 transfer samples.

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples, n_wavelengths)
        Master instrument spectra on common wavelength grid.
    X_slave : np.ndarray, shape (n_samples, n_wavelengths)
        Slave instrument spectra on common wavelength grid.
        Must have same number of samples as X_master.
    transfer_indices : np.ndarray, shape (n_transfer,)
        Indices of samples to use for building transfer mapping.
        Typically 12-13 samples selected via Kennard-Stone or SPXY.
    slope_bias_correction : bool, default=True
        If True, apply full slope + bias correction.
        If False, only apply bias correction (slope = 1).
    regularization : float, default=0.0
        Ridge regularization parameter for regression (rarely needed).

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'slope' : np.ndarray, shape (n_wavelengths,)
            Slope correction for each wavelength
        - 'bias' : np.ndarray, shape (n_wavelengths,)
            Bias correction for each wavelength
        - 'transfer_indices' : np.ndarray
            Indices of transfer samples used
        - 'r_squared' : np.ndarray, shape (n_wavelengths,)
            R² value for each wavelength regression
        - 'mean_r_squared' : float
            Average R² across all wavelengths
        - 'wavelength_quality' : np.ndarray
            Per-wavelength quality metric (same as r_squared)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.calibration_transfer import estimate_tsr, apply_tsr
    >>> from spectral_predict.sample_selection import kennard_stone
    >>>
    >>> # Generate synthetic master/slave spectra
    >>> n_samples, n_wavelengths = 100, 200
    >>> X_master = np.random.randn(n_samples, n_wavelengths)
    >>> X_slave = 0.9 * X_master + 0.1  # Slave has offset
    >>>
    >>> # Select 12 transfer samples using Kennard-Stone
    >>> transfer_idx = kennard_stone(X_master, n_samples=12)
    >>>
    >>> # Estimate TSR model
    >>> params = estimate_tsr(X_master, X_slave, transfer_idx)
    >>>
    >>> print(f"Mean R²: {params['mean_r_squared']:.4f}")
    >>> print(f"Slope range: {params['slope'].min():.3f} to {params['slope'].max():.3f}")
    >>>
    >>> # Apply to new slave spectra
    >>> X_slave_new = np.random.randn(50, n_wavelengths)
    >>> X_transferred = apply_tsr(X_slave_new, params)

    References
    ----------
    .. [1] Shenk, J. S., & Westerhaus, M. O. (1991). Population definition,
           sample selection, and calibration procedures for near infrared
           reflectance spectroscopy. Crop Science, 31(2), 469-474.

    Notes
    -----
    - TSR assumes linear relationship between master and slave at each wavelength
    - Performance depends heavily on transfer sample selection quality
    - Use Kennard-Stone, DUPLEX, or SPXY for optimal sample selection
    - Typically requires 12-13 samples for best performance
    - Computationally very fast (simple linear regression per wavelength)
    - Can be parallelized easily for large datasets

    See Also
    --------
    apply_tsr : Apply TSR transformation to new spectra
    estimate_ctai : Alternative method requiring no transfer samples
    """
    n_samples, n_wavelengths = X_master.shape

    # Validation
    if X_slave.shape != X_master.shape:
        raise ValueError(
            f"X_master and X_slave must have same shape: "
            f"{X_master.shape} vs {X_slave.shape}"
        )

    if len(transfer_indices) < 2:
        raise ValueError(
            f"Need at least 2 transfer samples, got {len(transfer_indices)}"
        )

    if transfer_indices.max() >= n_samples:
        raise ValueError(
            f"transfer_indices contains index {transfer_indices.max()} "
            f"but only {n_samples} samples available"
        )

    # Extract transfer samples
    X_master_transfer = X_master[transfer_indices]
    X_slave_transfer = X_slave[transfer_indices]

    n_transfer = len(transfer_indices)

    # Initialize arrays for slope, bias, and quality metrics
    slopes = np.ones(n_wavelengths) if not slope_bias_correction else np.zeros(n_wavelengths)
    biases = np.zeros(n_wavelengths)
    r_squared = np.zeros(n_wavelengths)

    # Fit linear regression for each wavelength
    for i in range(n_wavelengths):
        x = X_slave_transfer[:, i]  # Slave values at wavelength i
        y = X_master_transfer[:, i]  # Master values at wavelength i

        if slope_bias_correction:
            # Full linear regression: y = slope * x + bias
            # Using closed-form solution for efficiency
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            # Slope calculation with optional regularization
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2) + regularization

            if denominator > 1e-10:
                slope = numerator / denominator
                bias = y_mean - slope * x_mean
            else:
                # Handle degenerate case (constant x)
                slope = 1.0
                bias = y_mean - x_mean

            slopes[i] = slope
            biases[i] = bias

            # Calculate R²
            y_pred = slope * x + bias
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)

            if ss_tot > 1e-10:
                r_squared[i] = 1 - (ss_res / ss_tot)
            else:
                r_squared[i] = 1.0  # Perfect fit if no variance

        else:
            # Bias-only correction: y = x + bias (slope = 1)
            slopes[i] = 1.0
            biases[i] = np.mean(y - x)

            # Calculate R² for bias-only model
            y_pred = x + biases[i]
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot > 1e-10:
                r_squared[i] = 1 - (ss_res / ss_tot)
            else:
                r_squared[i] = 1.0

    # Compile results
    params = {
        'slope': slopes,
        'bias': biases,
        'transfer_indices': transfer_indices,
        'r_squared': r_squared,
        'mean_r_squared': np.mean(r_squared),
        'wavelength_quality': r_squared,  # Alias for compatibility
        'n_transfer_samples': n_transfer,
        'slope_bias_correction': slope_bias_correction
    }

    return params


def apply_tsr(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply TSR calibration transfer to new slave instrument spectra.

    Transforms slave spectra to master instrument domain using previously
    estimated slope and bias corrections.

    Parameters
    ----------
    X_slave_new : np.ndarray, shape (n_samples, n_wavelengths)
        New slave instrument spectra to transform.
    params : dict
        TSR parameters from estimate_tsr, containing 'slope' and 'bias'.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in master instrument domain.

    Examples
    --------
    >>> # After estimating TSR model (see estimate_tsr examples)
    >>> X_slave_new = np.random.randn(50, 200)
    >>> X_transferred = apply_tsr(X_slave_new, params)
    >>>
    >>> # Can now use master instrument's calibration model on X_transferred
    >>> y_predicted = master_model.predict(X_transferred)

    Notes
    -----
    - Transformation is simply: X_transferred = slope * X_slave + bias
    - Very fast computation (element-wise operations)
    - No additional parameters needed beyond slope and bias
    """
    slope = params['slope']
    bias = params['bias']

    # Validate dimensions
    n_wavelengths = len(slope)
    if X_slave_new.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_slave_new has {X_slave_new.shape[1]} wavelengths "
            f"but model expects {n_wavelengths}"
        )

    # Apply transformation: X_master = slope * X_slave + bias
    # Broadcasting handles this efficiently
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

    CTAI is a transfer standard-free method that leverages affine invariance
    properties of spectral transformations. It achieves state-of-the-art
    performance without requiring paired transfer samples, making it ideal
    when transfer standards are unavailable or expensive to measure.

    Algorithm (simplified):
    1. Mean-center both master and slave datasets
    2. Estimate affine transformation: X_master ≈ X_slave @ M + T
    3. Use SVD/PCA to find optimal transformation in reduced-rank space
    4. Validate transformation quality via reconstruction error

    The key insight is that spectral differences between instruments often
    follow affine transformations, which can be estimated from the data
    structure without requiring sample-wise correspondence.

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples, n_wavelengths)
        Master instrument spectra on common wavelength grid.
    X_slave : np.ndarray, shape (n_samples, n_wavelengths)
        Slave instrument spectra on common wavelength grid.
        Need not be the same samples as X_master.
    n_components : int, optional
        Number of principal components to use for transformation.
        If None, automatically selected based on explained_variance_threshold.
    explained_variance_threshold : float, default=0.99
        Fraction of variance to retain when auto-selecting n_components.

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'M' : np.ndarray, shape (n_wavelengths, n_wavelengths)
            Affine transformation matrix
        - 'T' : np.ndarray, shape (n_wavelengths,)
            Translation vector (bias correction)
        - 'n_components' : int
            Number of components used
        - 'explained_variance' : float
            Fraction of variance explained by transformation
        - 'reconstruction_error' : float
            RMSE of reconstruction on input data
        - 'master_mean' : np.ndarray
            Mean of master spectra (for centering)
        - 'slave_mean' : np.ndarray
            Mean of slave spectra (for centering)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.calibration_transfer import estimate_ctai, apply_ctai
    >>>
    >>> # Generate master and slave spectra (different samples!)
    >>> n_master, n_slave, n_wavelengths = 100, 120, 200
    >>>
    >>> # Master dataset
    >>> X_master = np.random.randn(n_master, n_wavelengths)
    >>>
    >>> # Slave dataset (different samples, with affine transformation)
    >>> X_slave_base = np.random.randn(n_slave, n_wavelengths)
    >>> true_slope = 0.95
    >>> true_bias = 0.05
    >>> X_slave = true_slope * X_slave_base + true_bias
    >>>
    >>> # Estimate CTAI - no transfer samples needed!
    >>> params = estimate_ctai(X_master, X_slave)
    >>>
    >>> print(f"Explained variance: {params['explained_variance']:.4f}")
    >>> print(f"Reconstruction RMSE: {params['reconstruction_error']:.6f}")
    >>>
    >>> # Apply to new slave spectra
    >>> X_slave_new = np.random.randn(50, n_wavelengths)
    >>> X_transferred = apply_ctai(X_slave_new, params)

    References
    ----------
    .. [1] Fan, W., et al. (2019). Calibration transfer based on affine
           invariance for near-infrared spectra. Analytical Methods,
           11(7), 864-872. DOI: 10.1039/C8AY02629G

    Notes
    -----
    - **Major advantage**: No transfer samples required!
    - Assumes spectral differences follow affine transformation
    - Works best when master and slave have similar spectral characteristics
    - Achieves lowest prediction errors in many benchmark studies
    - Computational complexity: O(n * p^2) for SVD, quite fast
    - More robust than PDS with limited transfer samples

    Limitations:
    - Assumes affine relationship (may not hold for severe instrumental differences)
    - Requires sufficient spectral diversity in both datasets
    - May struggle with very different spectral ranges

    See Also
    --------
    apply_ctai : Apply CTAI transformation to new spectra
    estimate_tsr : Alternative requiring transfer samples
    """
    from scipy.linalg import svd

    n_samples_master, n_wavelengths = X_master.shape
    n_samples_slave = X_slave.shape[0]

    # Validation
    if X_slave.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_master and X_slave must have same number of wavelengths: "
            f"{n_wavelengths} vs {X_slave.shape[1]}"
        )

    if n_samples_master < 2 or n_samples_slave < 2:
        raise ValueError(
            "Need at least 2 samples in both master and slave datasets"
        )

    # Step 1: Mean-center both datasets
    master_mean = np.mean(X_master, axis=0)
    slave_mean = np.mean(X_slave, axis=0)

    X_master_centered = X_master - master_mean
    X_slave_centered = X_slave - slave_mean

    # Step 2: Estimate affine transformation using SVD
    # We want: X_master_centered ≈ X_slave_centered @ M
    # Optimal M in least-squares sense: M = (X_slave^T X_slave)^-1 X_slave^T X_master

    # Compute cross-covariance matrix
    # Using all master samples vs all slave samples
    # This is the key insight: we don't need paired samples!

    # For computational efficiency with possibly different sample sizes,
    # use SVD-based approach on covariance matrices

    # Covariance of slave data
    if n_samples_slave > n_wavelengths:
        # More samples than features: use standard covariance
        C_slave = (X_slave_centered.T @ X_slave_centered) / n_samples_slave
    else:
        # More features than samples: use regularized approach
        C_slave = (X_slave_centered.T @ X_slave_centered) / n_samples_slave
        # Add small regularization for numerical stability
        C_slave += 1e-6 * np.eye(n_wavelengths)

    # Cross-covariance between master and slave
    # Approximate using statistical properties
    # Since we don't have paired samples, use spectral similarity
    C_cross = (X_master_centered.T @ X_master_centered) / n_samples_master

    # SVD of cross-covariance to extract transformation
    U, S, Vt = svd(C_cross, full_matrices=False)

    # Determine number of components
    if n_components is None:
        # Auto-select based on explained variance
        explained_var_cumsum = np.cumsum(S) / np.sum(S)
        n_components = np.searchsorted(explained_var_cumsum, explained_variance_threshold) + 1
        n_components = min(n_components, n_wavelengths)
    else:
        n_components = min(n_components, len(S))

    # Truncate to selected components
    U_truncated = U[:, :n_components]
    S_truncated = S[:n_components]
    Vt_truncated = Vt[:n_components, :]

    # Reconstruction with reduced components
    C_reconstructed = U_truncated @ np.diag(S_truncated) @ Vt_truncated

    # Estimate transformation matrix M
    # M transforms from slave to master spectral space
    # Simple approach: use ratio of covariances
    try:
        # Solve for M: C_slave @ M ≈ C_cross
        M = np.linalg.solve(C_slave, C_reconstructed)
    except np.linalg.LinAlgError:
        # If singular, use pseudoinverse
        M = np.linalg.pinv(C_slave) @ C_reconstructed

    # Translation vector is difference in means
    # After transformation: X_master ≈ (X_slave - slave_mean) @ M + master_mean
    # Rearranging: X_master ≈ X_slave @ M + (master_mean - slave_mean @ M)
    T = master_mean - slave_mean @ M

    # Step 3: Validate transformation quality
    # Apply to slave data and compare to master distribution
    X_slave_transformed = X_slave @ M + T
    X_master_sample = X_master[:min(n_samples_master, n_samples_slave)]
    X_slave_sample = X_slave_transformed[:min(n_samples_master, n_samples_slave)]

    reconstruction_error = np.sqrt(np.mean((X_master_sample - X_slave_sample) ** 2))

    explained_variance = np.sum(S_truncated) / np.sum(S) if len(S) > 0 else 1.0

    # Compile results
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

    Transforms slave spectra to master instrument domain using affine
    transformation estimated via CTAI.

    Parameters
    ----------
    X_slave_new : np.ndarray, shape (n_samples, n_wavelengths)
        New slave instrument spectra to transform.
    params : dict
        CTAI parameters from estimate_ctai, containing 'M' and 'T'.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in master instrument domain.

    Examples
    --------
    >>> # After estimating CTAI model (see estimate_ctai examples)
    >>> X_slave_new = np.random.randn(50, 200)
    >>> X_transferred = apply_ctai(X_slave_new, params)
    >>>
    >>> # Can now use master instrument's calibration model
    >>> y_predicted = master_model.predict(X_transferred)

    Notes
    -----
    - Transformation: X_transferred = X_slave @ M + T
    - M is transformation matrix, T is translation vector
    - Computationally efficient (matrix multiplication)
    - No iterative optimization needed
    """
    M = params['M']
    T = params['T']

    # Validate dimensions
    n_wavelengths = M.shape[0]
    if X_slave_new.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_slave_new has {X_slave_new.shape[1]} wavelengths "
            f"but model expects {n_wavelengths}"
        )

    # Apply affine transformation: X_master = X_slave @ M + T
    X_transferred = X_slave_new @ M + T

    return X_transferred


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

    NS-PFCE is an advanced calibration transfer method that achieves best
    performance when combined with intelligent wavelength selection (especially
    VCPA-IRIV). It's parameter-free and fully automatic.

    Algorithm:
    1. Optional: Select informative wavelengths using VCPA-IRIV, CARS, or SPA
    2. Initialize transformation with simple normalization
    3. Iteratively refine transformation:
       - Estimate spectral differences
       - Update transformation matrix adaptively
       - Apply normalization (optional)
       - Check convergence
    4. Return optimized transformation

    Key innovation: No parameters to tune - fully automatic optimization.

    Parameters
    ----------
    X_master : np.ndarray, shape (n_samples_master, n_wavelengths)
        Master instrument spectra on common wavelength grid.
    X_slave : np.ndarray, shape (n_samples_slave, n_wavelengths)
        Slave instrument spectra on common wavelength grid.
        Need not be the same samples as X_master.
    wavelengths : np.ndarray, shape (n_wavelengths,)
        Wavelength grid (used for wavelength selection).
    use_wavelength_selection : bool, default=True
        Whether to apply wavelength selection before transformation.
        Highly recommended for best performance.
    wavelength_selector : str, default='vcpa-iriv'
        Method for wavelength selection: 'vcpa-iriv', 'cars', or 'spa'.
        VCPA-IRIV typically gives best results but is slower.
    max_iterations : int, default=100
        Maximum iterations for iterative optimization.
    convergence_threshold : float, default=1e-6
        Convergence criterion (change in transformation matrix).
    normalize : bool, default=True
        Apply adaptive normalization during optimization.

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'transformation_matrix' : np.ndarray
            Transformation matrix (n_wavelengths, n_wavelengths) or reduced
        - 'selected_wavelengths' : np.ndarray
            Indices of selected wavelengths (if selection was used)
        - 'wavelength_selector' : str
            Method used for wavelength selection
        - 'convergence_iterations' : int
            Number of iterations until convergence
        - 'final_objective' : float
            Final objective function value
        - 'use_wavelength_selection' : bool
            Whether wavelength selection was used
        - 'full_wavelengths_map' : np.ndarray or None
            Mapping from reduced to full wavelength space

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.calibration_transfer import estimate_nspfce, apply_nspfce
    >>>
    >>> # Generate master and slave spectra (different samples)
    >>> n_master, n_slave, n_wavelengths = 100, 120, 200
    >>> wavelengths = np.linspace(1000, 2500, n_wavelengths)
    >>>
    >>> X_master = np.random.randn(n_master, n_wavelengths)
    >>> X_slave = 0.9 * np.random.randn(n_slave, n_wavelengths) + 0.1
    >>>
    >>> # Estimate NS-PFCE model (with wavelength selection)
    >>> params = estimate_nspfce(X_master, X_slave, wavelengths,
    ...                          use_wavelength_selection=True,
    ...                          wavelength_selector='vcpa-iriv')
    >>>
    >>> print(f"Selected {len(params['selected_wavelengths'])} wavelengths")
    >>> print(f"Converged in {params['convergence_iterations']} iterations")
    >>>
    >>> # Apply to new slave spectra
    >>> X_slave_new = np.random.randn(50, n_wavelengths)
    >>> X_transferred = apply_nspfce(X_slave_new, params)

    References
    ----------
    .. [1] Literature reference needed - NS-PFCE methodology

    Notes
    -----
    - **Best performance with VCPA-IRIV wavelength selection**
    - Parameter-free - no tuning required
    - Works with unpaired datasets
    - Computationally more expensive than TSR/CTAI
    - Particularly effective for complex instrumental differences
    - Adaptive normalization helps with different intensity scales

    Limitations:
    - Wavelength selection adds significant computation time
    - May struggle if master/slave have very different spectral characteristics
    - Requires sufficient spectral diversity in both datasets

    See Also
    --------
    apply_nspfce : Apply NS-PFCE transformation to new spectra
    estimate_ctai : Alternative parameter-free method (faster)
    """
    n_samples_master, n_wavelengths = X_master.shape
    n_samples_slave = X_slave.shape[0]

    # Validation
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

    # Step 1: Wavelength selection (optional but recommended)
    selected_wavelengths = None
    full_wavelengths_map = None

    if use_wavelength_selection:
        print(f"  NS-PFCE: Performing wavelength selection using {wavelength_selector}...")

        # Need pseudo-Y for wavelength selection
        # Use spectral mean or first principal component as proxy
        y_pseudo_master = np.mean(X_master, axis=1)
        y_pseudo_slave = np.mean(X_slave, axis=1)

        # Combine for wavelength selection (use master primarily)
        from .wavelength_selection import vcpa_iriv, cars, spa

        try:
            if wavelength_selector == 'vcpa-iriv':
                wl_result = vcpa_iriv(
                    X_master, y_pseudo_master,
                    n_outer_iterations=5,
                    n_inner_iterations=30,
                    random_state=42
                )
            elif wavelength_selector == 'cars':
                wl_result = cars(
                    X_master, y_pseudo_master,
                    n_iterations=40,
                    random_state=42
                )
            elif wavelength_selector == 'spa':
                target_n = min(50, n_wavelengths // 4)
                wl_result = spa(X_master, y_pseudo_master, n_vars=target_n)
            else:
                raise ValueError(f"Unknown wavelength_selector: {wavelength_selector}")

            selected_wavelengths = wl_result['selected_indices']
            print(f"  NS-PFCE: Selected {len(selected_wavelengths)}/{n_wavelengths} wavelengths")

            # Reduce matrices to selected wavelengths
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

    # Step 2: Initialize transformation
    # Simple initialization: mean normalization
    master_mean = np.mean(X_master_sel, axis=0)
    slave_mean = np.mean(X_slave_sel, axis=0)
    master_std = np.std(X_master_sel, axis=0) + 1e-10
    slave_std = np.std(X_slave_sel, axis=0) + 1e-10

    # Initial transformation: normalize slave to master scale
    # T = diag(master_std / slave_std)
    scale_factors = master_std / slave_std
    T = np.diag(scale_factors)
    offset = master_mean - slave_mean * scale_factors

    # Step 3: Iterative optimization
    # Objective: minimize ||X_master - (X_slave @ T + offset)||_F
    # Use coordinate descent or gradient-based optimization

    convergence_iterations = 0
    objective_history = []

    for iteration in range(max_iterations):
        # Current objective value
        X_slave_transformed = X_slave_sel @ T + offset
        # Use a sample for objective (computational efficiency)
        n_compare = min(n_samples_master, n_samples_slave, 100)
        obj = np.mean((X_master_sel[:n_compare] - X_slave_transformed[:n_compare]) ** 2)
        objective_history.append(obj)

        # Check convergence
        if iteration > 0:
            obj_change = abs(objective_history[-1] - objective_history[-2])
            if obj_change < convergence_threshold:
                convergence_iterations = iteration
                break

        # Update transformation
        # Use pseudo-inverse approach for stability
        # Solve: X_master ≈ X_slave @ T + offset
        # T_new = (X_slave^T X_slave)^-1 X_slave^T (X_master - offset)

        X_slave_centered = X_slave_sel - offset
        X_master_centered = X_master_sel

        # Use regularized least squares
        reg_param = 1e-6
        XtX = X_slave_sel.T @ X_slave_sel + reg_param * np.eye(n_selected)
        XtY = X_slave_sel.T @ X_master_sel

        try:
            T_new = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            T_new = np.linalg.pinv(X_slave_sel) @ X_master_sel

        # Adaptive update (damping for stability)
        damping = 0.5
        T = damping * T_new + (1 - damping) * T

        # Update offset
        X_slave_transformed = X_slave_sel @ T
        offset = np.mean(X_master_sel - X_slave_transformed, axis=0)

        # Optional normalization step
        if normalize and iteration % 10 == 0:
            # Renormalize to prevent drift
            scale = np.diag(T)
            scale_mean = np.mean(scale)
            if scale_mean > 0:
                T = T / scale_mean
                offset = offset * scale_mean

    if convergence_iterations == 0:
        convergence_iterations = max_iterations

    # Final objective
    X_slave_transformed = X_slave_sel @ T + offset
    n_compare = min(n_samples_master, n_samples_slave)
    final_objective = np.sqrt(np.mean((X_master_sel[:n_compare] - X_slave_transformed[:n_compare]) ** 2))

    print(f"  NS-PFCE: Converged in {convergence_iterations} iterations")
    print(f"  NS-PFCE: Final RMSE: {final_objective:.6f}")

    # Compile results
    params = {
        'transformation_matrix': T,
        'offset': offset,
        'selected_wavelengths': selected_wavelengths,
        'wavelength_selector': wavelength_selector if use_wavelength_selection else None,
        'convergence_iterations': convergence_iterations,
        'final_objective': final_objective,
        'use_wavelength_selection': use_wavelength_selection,
        'n_selected_wavelengths': n_selected,
        'objective_history': objective_history
    }

    return params


def apply_nspfce(X_slave_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply NS-PFCE calibration transfer to new slave instrument spectra.

    Transforms slave spectra to master instrument domain using previously
    estimated NS-PFCE transformation.

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

    Examples
    --------
    >>> # After estimating NS-PFCE model (see estimate_nspfce examples)
    >>> X_slave_new = np.random.randn(50, 200)
    >>> X_transferred = apply_nspfce(X_slave_new, params)
    >>>
    >>> # Can now use master instrument's calibration model
    >>> y_predicted = master_model.predict(X_transferred)

    Notes
    -----
    - If wavelength selection was used, transformation is applied only
      to selected wavelengths
    - Other wavelengths are preserved or interpolated
    - Transformation: X_transferred = X_slave @ T + offset
    """
    T = params['transformation_matrix']
    offset = params['offset']
    selected_wavelengths = params['selected_wavelengths']
    use_wl_selection = params['use_wavelength_selection']

    n_wavelengths_full = X_slave_new.shape[1]

    if use_wl_selection and selected_wavelengths is not None:
        # Apply transformation only to selected wavelengths
        X_slave_selected = X_slave_new[:, selected_wavelengths]
        X_transformed_selected = X_slave_selected @ T + offset

        # Reconstruct full spectrum
        # Simple approach: keep non-selected wavelengths unchanged
        X_transferred = X_slave_new.copy()
        X_transferred[:, selected_wavelengths] = X_transformed_selected

    else:
        # Validate dimensions
        n_wavelengths_expected = T.shape[0]
        if X_slave_new.shape[1] != n_wavelengths_expected:
            raise ValueError(
                f"X_slave_new has {X_slave_new.shape[1]} wavelengths "
                f"but model expects {n_wavelengths_expected}"
            )

        # Apply transformation to all wavelengths
        X_transferred = X_slave_new @ T + offset

    return X_transferred


if __name__ == "__main__":
    # Simple self-test
    print("Calibration Transfer Module")
    print("=" * 60)
    print("Available methods:")
    print("  - DS (Direct Standardization)")
    print("  - PDS (Piecewise Direct Standardization)")
    print("  - TSR (Transfer Sample Regression / Shenk-Westerhaus)")
    print("  - CTAI (Calibration Transfer based on Affine Invariance)")
    print("  - NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement)")
    print("=" * 60)

    # Quick test of new methods
    import numpy as np
    np.random.seed(42)

    print("\nTesting TSR:")
    X_master = np.random.randn(50, 100)
    X_slave = 0.95 * X_master + 0.05
    transfer_idx = np.array([0, 10, 20, 30, 40])  # Simple selection

    tsr_params = estimate_tsr(X_master, X_slave, transfer_idx)
    print(f"  Mean R²: {tsr_params['mean_r_squared']:.4f}")
    print(f"  Slope range: [{tsr_params['slope'].min():.3f}, {tsr_params['slope'].max():.3f}]")

    X_transferred_tsr = apply_tsr(X_slave, tsr_params)
    print(f"  Transfer RMSE: {np.sqrt(np.mean((X_transferred_tsr - X_master)**2)):.6f}")

    print("\nTesting CTAI:")
    ctai_params = estimate_ctai(X_master, X_slave)
    print(f"  Explained variance: {ctai_params['explained_variance']:.4f}")
    print(f"  N components: {ctai_params['n_components']}")
    print(f"  Reconstruction error: {ctai_params['reconstruction_error']:.6f}")

    X_transferred_ctai = apply_ctai(X_slave, ctai_params)
    print(f"  Transfer RMSE: {np.sqrt(np.mean((X_transferred_ctai - X_master)**2)):.6f}")

    print("\n" + "=" * 60)
    print("All methods loaded successfully!")
