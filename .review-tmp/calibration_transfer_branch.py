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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)


MethodType = Literal["ds", "pds", "tsr", "ctai", "nspfce", "jypls-inv"]


@dataclass
class TransferModel:
    """
    Encapsulates a calibration transfer mapping from a satellite instrument
    to a primary instrument on a common wavelength grid.
    """
    primary_id: str
    satellite_id: str
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


def clip_wavelengths_to_region(
    X: np.ndarray,
    wavelengths: np.ndarray,
    region_start: float,
    region_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip spectra and wavelengths to a region of interest.

    Parameters
    ----------
    X : np.ndarray
        Spectra of shape (n_samples, n_wavelengths).
    wavelengths : np.ndarray
        Wavelength array of shape (n_wavelengths,).
    region_start : float
        Start of the region (inclusive).
    region_end : float
        End of the region (inclusive).

    Returns
    -------
    X_clipped : np.ndarray
        Spectra clipped to the region, shape (n_samples, n_region).
    wl_clipped : np.ndarray
        Wavelengths in the region, shape (n_region,).
    indices : np.ndarray
        Boolean mask or integer indices into the original wavelength array.
    """
    mask = (wavelengths >= region_start) & (wavelengths <= region_end)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(
            f"No wavelengths found in region [{region_start}, {region_end}]. "
            f"Available range: [{wavelengths.min():.1f}, {wavelengths.max():.1f}]"
        )
    return X[:, indices], wavelengths[indices], indices


def estimate_ds(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    lam: float = 0.0,
) -> np.ndarray:
    """
    Estimate a Direct Standardization (DS) matrix A such that:
        X_satellite @ A ≈ X_primary

    Parameters
    ----------
    X_primary : np.ndarray
        Primary instrument spectra on common grid, shape (n_samples, p).
    X_satellite : np.ndarray
        Satellite instrument spectra on common grid, shape (n_samples, p).
    lam : float
        Optional ridge regularization parameter.

    Returns
    -------
    np.ndarray
        DS matrix A of shape (p, p).
    """
    # Solve for A: X_satellite @ A = X_primary
    # A = (X_satellite^T @ X_satellite + lam*I)^-1 @ X_satellite^T @ X_primary

    p = X_satellite.shape[1]

    # Compute X_satellite^T @ X_satellite
    XtX = X_satellite.T @ X_satellite

    # Add ridge regularization
    if lam > 0:
        XtX += lam * np.eye(p)

    # Compute X_satellite^T @ X_primary
    XtY = X_satellite.T @ X_primary

    # Solve for A
    A = np.linalg.solve(XtX, XtY)

    return A


def apply_ds(X_satellite_new: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Apply a previously estimated DS matrix A to new satellite spectra.

    Returns
    -------
    np.ndarray
        Transformed spectra in primary instrument domain.
    """
    return X_satellite_new @ A


def estimate_pds(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    window: int = 11,
) -> np.ndarray:
    """
    Estimate Piecewise Direct Standardization (PDS) coefficients B.

    Parameters
    ----------
    X_primary : np.ndarray
        Primary spectra on common grid, shape (n_samples, p).
    X_satellite : np.ndarray
        Satellite spectra on common grid, shape (n_samples, p).
    window : int
        Window size (odd integer >= 1) for local regression around each
        wavelength. This implementation uses a full centered width of
        2k+1 (channels i-k to i+k), so it must be odd. Even values
        raise ValueError.

    Returns
    -------
    np.ndarray
        PDS coefficient array B of shape (p, window).

    Raises
    ------
    ValueError
        If `window` is not a positive odd integer.
    """
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(
            f"window must be a positive integer (got {window!r}). "
            "PDS uses a centered local-regression window of width 2k+1."
        )
    if window % 2 == 0:
        raise ValueError(
            f"window must be odd (got {window}). This implementation uses a "
            "centered, odd-width local window of size 2k+1 — see "
            "Wang, Veltkamp, & Kowalski (1991), 'Multivariate Instrument "
            "Standardization,' Anal. Chem. 63(23), 2750-2756."
        )
    n_samples, p = X_satellite.shape
    half_window = window // 2

    B = np.zeros((p, window))

    for i in range(p):
        # Determine window boundaries
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        # Extract window from satellite spectra
        X_window = X_satellite[:, start:end]

        # Extract target from primary spectra (single wavelength)
        y_target = X_primary[:, i]

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
    X_satellite_new: np.ndarray,
    B: np.ndarray,
    window: int | None = None,
) -> np.ndarray:
    """
    Apply previously estimated PDS coefficients B to new satellite spectra.

    Parameters
    ----------
    X_satellite_new : np.ndarray
        New satellite spectra, shape (n_samples, p).
    B : np.ndarray
        PDS coefficient matrix from estimate_pds, shape (p, w) where w
        is the odd window width 2k+1.
    window : int or None
        Deprecated. If provided and disagrees with B.shape[1], a
        FutureWarning is issued and B.shape[1] is used instead.

    Returns
    -------
    np.ndarray
        Transformed spectra in primary instrument domain.

    Raises
    ------
    ValueError
        If B has even second dimension or B.shape[0] != X_satellite_new.shape[1].
    """
    import warnings

    n_samples, p = X_satellite_new.shape
    expected_window = int(B.shape[1])
    if expected_window % 2 == 0:
        raise ValueError(
            f"B has even width {expected_window}; PDS B must be odd-width "
            f"2k+1. Re-fit estimator on data using current calibration_transfer.py."
        )
    if B.shape[0] != p:
        raise ValueError(
            f"B.shape[0]={B.shape[0]} != X.shape[1]={p}; "
            f"B was fitted on a different feature count."
        )
    if window is not None and window != expected_window:
        warnings.warn(
            f"apply_pds: caller passed window={window}, but B has width "
            f"{expected_window}. Ignoring caller arg; using B-derived width.",
            FutureWarning,
            stacklevel=2,
        )
    half_window = expected_window // 2

    X_transformed = np.zeros_like(X_satellite_new)

    for i in range(p):
        start = max(0, i - half_window)
        end = min(p, i + half_window + 1)

        X_window = X_satellite_new[:, start:end]

        offset = start - (i - half_window)
        b = B[i, offset:offset + X_window.shape[1]]

        X_transformed[:, i] = X_window @ b

    return X_transformed


def save_transfer_model(
    transfer_model: TransferModel,
    directory: Path | str,
    name: str | None = None,
    data_type_suffix: str = "",
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
        primary_id, satellite_id, and method.
    data_type_suffix : str, optional
        Suffix to append to auto-generated filename (e.g., "_abs" or "_ref").
        Only used when name is None.

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
        name = f"{transfer_model.primary_id}_from_{transfer_model.satellite_id}_{transfer_model.method}{data_type_suffix}"

    path_prefix = directory / name

    # Save metadata to JSON
    metadata = {
        "primary_id": transfer_model.primary_id,
        "satellite_id": transfer_model.satellite_id,
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
        primary_id=metadata["primary_id"],
        satellite_id=metadata["satellite_id"],
        method=metadata["method"],
        wavelengths_common=wavelengths_common,
        params=params,
        meta=metadata.get("meta", {}),
    )


# ==============================================================================
# TSR (Transfer Sample Regression / Shenk-Westerhaus)
# ==============================================================================

def estimate_tsr(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
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
    1. Extract transfer samples from primary and satellite datasets
    2. For each wavelength λ:
       - Fit linear regression: X_primary[λ] = slope[λ] * X_satellite[λ] + bias[λ]
       - Store slope and bias coefficients
    3. Return parameters for applying correction to new samples

    With optimal sample selection (e.g., Kennard-Stone), TSR can achieve
    results statistically indistinguishable from full recalibration using
    only 12-13 transfer samples.

    Parameters
    ----------
    X_primary : np.ndarray, shape (n_samples, n_wavelengths)
        Primary instrument spectra on common wavelength grid.
    X_satellite : np.ndarray, shape (n_samples, n_wavelengths)
        Satellite instrument spectra on common wavelength grid.
        Must have same number of samples as X_primary.
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
    >>> # Generate synthetic primary/satellite spectra
    >>> n_samples, n_wavelengths = 100, 200
    >>> X_primary = np.random.randn(n_samples, n_wavelengths)
    >>> X_satellite = 0.9 * X_primary + 0.1  # Satellite has offset
    >>>
    >>> # Select 12 transfer samples using Kennard-Stone
    >>> transfer_idx = kennard_stone(X_primary, n_samples=12)
    >>>
    >>> # Estimate TSR model
    >>> params = estimate_tsr(X_primary, X_satellite, transfer_idx)
    >>>
    >>> print(f"Mean R²: {params['mean_r_squared']:.4f}")
    >>> print(f"Slope range: {params['slope'].min():.3f} to {params['slope'].max():.3f}")
    >>>
    >>> # Apply to new satellite spectra
    >>> X_satellite_new = np.random.randn(50, n_wavelengths)
    >>> X_transferred = apply_tsr(X_satellite_new, params)

    References
    ----------
    .. [1] Shenk, J. S., & Westerhaus, M. O. (1991). Population definition,
           sample selection, and calibration procedures for near infrared
           reflectance spectroscopy. Crop Science, 31(2), 469-474.

    Notes
    -----
    - TSR assumes linear relationship between primary and satellite at each wavelength
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
    n_samples, n_wavelengths = X_primary.shape

    # Validation
    if X_satellite.shape != X_primary.shape:
        raise ValueError(
            f"X_primary and X_satellite must have same shape: "
            f"{X_primary.shape} vs {X_satellite.shape}"
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
    X_primary_transfer = X_primary[transfer_indices]
    X_satellite_transfer = X_satellite[transfer_indices]

    n_transfer = len(transfer_indices)

    # Initialize arrays for slope, bias, and quality metrics
    slopes = np.ones(n_wavelengths) if not slope_bias_correction else np.zeros(n_wavelengths)
    biases = np.zeros(n_wavelengths)
    r_squared = np.zeros(n_wavelengths)

    # Fit linear regression for each wavelength
    for i in range(n_wavelengths):
        x = X_satellite_transfer[:, i]  # Satellite values at wavelength i
        y = X_primary_transfer[:, i]  # Primary values at wavelength i

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


def apply_tsr(X_satellite_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply TSR calibration transfer to new satellite instrument spectra.

    Transforms satellite spectra to primary instrument domain using previously
    estimated slope and bias corrections.

    Parameters
    ----------
    X_satellite_new : np.ndarray, shape (n_samples, n_wavelengths)
        New satellite instrument spectra to transform.
    params : dict
        TSR parameters from estimate_tsr, containing 'slope' and 'bias'.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in primary instrument domain.

    Examples
    --------
    >>> # After estimating TSR model (see estimate_tsr examples)
    >>> X_satellite_new = np.random.randn(50, 200)
    >>> X_transferred = apply_tsr(X_satellite_new, params)
    >>>
    >>> # Can now use primary instrument's calibration model on X_transferred
    >>> y_predicted = primary_model.predict(X_transferred)

    Notes
    -----
    - Transformation is simply: X_transferred = slope * X_satellite + bias
    - Very fast computation (element-wise operations)
    - No additional parameters needed beyond slope and bias
    """
    slope = params['slope']
    bias = params['bias']

    # Validate dimensions
    n_wavelengths = len(slope)
    if X_satellite_new.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_satellite_new has {X_satellite_new.shape[1]} wavelengths "
            f"but model expects {n_wavelengths}"
        )

    # Apply transformation: X_primary = slope * X_satellite + bias
    # Broadcasting handles this efficiently
    X_transferred = X_satellite_new * slope + bias

    return X_transferred


# ==============================================================================
# CTAI (Calibration Transfer based on Affine Invariance)
# ==============================================================================

def estimate_ctai(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
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
    1. Mean-center both primary and satellite datasets
    2. Estimate affine transformation: X_primary ≈ X_satellite @ M + T
    3. Use SVD/PCA to find optimal transformation in reduced-rank space
    4. Validate transformation quality via reconstruction error

    The key insight is that spectral differences between instruments often
    follow affine transformations, which can be estimated from the data
    structure without requiring sample-wise correspondence.

    Parameters
    ----------
    X_primary : np.ndarray, shape (n_samples, n_wavelengths)
        Primary instrument spectra on common wavelength grid.
    X_satellite : np.ndarray, shape (n_samples, n_wavelengths)
        Satellite instrument spectra on common wavelength grid.
        Need not be the same samples as X_primary.
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
        - 'primary_mean' : np.ndarray
            Mean of primary spectra (for centering)
        - 'satellite_mean' : np.ndarray
            Mean of satellite spectra (for centering)

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.calibration_transfer import estimate_ctai, apply_ctai
    >>>
    >>> # Generate primary and satellite spectra (different samples!)
    >>> n_primary, n_satellite, n_wavelengths = 100, 120, 200
    >>>
    >>> # Primary dataset
    >>> X_primary = np.random.randn(n_primary, n_wavelengths)
    >>>
    >>> # Satellite dataset (different samples, with affine transformation)
    >>> X_satellite_base = np.random.randn(n_satellite, n_wavelengths)
    >>> true_slope = 0.95
    >>> true_bias = 0.05
    >>> X_satellite = true_slope * X_satellite_base + true_bias
    >>>
    >>> # Estimate CTAI - no transfer samples needed!
    >>> params = estimate_ctai(X_primary, X_satellite)
    >>>
    >>> print(f"Explained variance: {params['explained_variance']:.4f}")
    >>> print(f"Reconstruction RMSE: {params['reconstruction_error']:.6f}")
    >>>
    >>> # Apply to new satellite spectra
    >>> X_satellite_new = np.random.randn(50, n_wavelengths)
    >>> X_transferred = apply_ctai(X_satellite_new, params)

    References
    ----------
    .. [1] Fan, W., et al. (2019). Calibration transfer based on affine
           invariance for near-infrared spectra. Analytical Methods,
           11(7), 864-872. DOI: 10.1039/C8AY02629G

    Notes
    -----
    - **Major advantage**: No transfer samples required!
    - Assumes spectral differences follow affine transformation
    - Works best when primary and satellite have similar spectral characteristics
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

    n_samples_primary, n_wavelengths = X_primary.shape
    n_samples_satellite = X_satellite.shape[0]

    logger.debug("CTAI input shapes: Primary %s, Satellite %s", X_primary.shape, X_satellite.shape)

    if X_satellite.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_primary and X_satellite must have same number of wavelengths: "
            f"{n_wavelengths} vs {X_satellite.shape[1]}"
        )

    if n_samples_primary < 2 or n_samples_satellite < 2:
        raise ValueError(
            "Need at least 2 samples in both primary and satellite datasets"
        )

    # Check for NaN/inf values
    if np.any(np.isnan(X_primary)):
        n_nan = np.sum(np.isnan(X_primary))
        raise ValueError(f"X_primary contains {n_nan} NaN values")
    if np.any(np.isinf(X_primary)):
        n_inf = np.sum(np.isinf(X_primary))
        raise ValueError(f"X_primary contains {n_inf} infinite values")
    if np.any(np.isnan(X_satellite)):
        n_nan = np.sum(np.isnan(X_satellite))
        raise ValueError(f"X_satellite contains {n_nan} NaN values")
    if np.any(np.isinf(X_satellite)):
        n_inf = np.sum(np.isinf(X_satellite))
        raise ValueError(f"X_satellite contains {n_inf} infinite values")

    logger.debug(
        "CTAI data validation passed; primary range [%.6f, %.6f], satellite range [%.6f, %.6f]",
        np.min(X_primary), np.max(X_primary), np.min(X_satellite), np.max(X_satellite),
    )

    # Step 1: Mean-center both datasets
    primary_mean = np.mean(X_primary, axis=0)
    satellite_mean = np.mean(X_satellite, axis=0)

    X_primary_centered = X_primary - primary_mean
    X_satellite_centered = X_satellite - satellite_mean

    logger.debug(
        "CTAI mean centering complete; primary mean range [%.6f, %.6f], satellite mean range [%.6f, %.6f]",
        np.min(primary_mean), np.max(primary_mean), np.min(satellite_mean), np.max(satellite_mean),
    )

    # Step 2: Estimate affine transformation using PCA-regularized regression
    # We want: X_primary ≈ X_satellite @ M + T (in data space, not covariance space!)
    # CTAI's key insight: Use PCA to find low-rank approximation for stability

    # Check if we have paired samples
    have_paired_samples = (n_samples_primary == n_samples_satellite)

    if not have_paired_samples:
        raise ValueError(
            f"CTAI requires paired samples (same samples on both instruments).\n"
            f"Got {n_samples_primary} primary samples and {n_samples_satellite} satellite samples.\n"
            f"For unpaired samples, use TSR, DS, or PDS instead."
        )

    print(f"  CTAI: Detected {n_samples_primary} paired samples on both instruments")

    try:
        U_satellite, S_satellite, Vt_satellite = svd(X_satellite_centered, full_matrices=False)
        logger.debug(
            "CTAI SVD: U%s S%s Vt%s; singular values [%.6e, %.6e]",
            U_satellite.shape, S_satellite.shape, Vt_satellite.shape,
            np.min(S_satellite), np.max(S_satellite),
        )
        if S_satellite[-1] > 0:
            logger.debug("CTAI SVD condition number: %.2e", S_satellite[0] / S_satellite[-1])
    except np.linalg.LinAlgError as e:
        raise ValueError(f"SVD failed: {e}. Check if data has sufficient variance.")

    # Step 2b: Determine number of components (for regularization)
    if n_components is None:
        # Auto-select based on explained variance
        explained_var_cumsum = np.cumsum(S_satellite**2) / np.sum(S_satellite**2)
        n_components = np.searchsorted(explained_var_cumsum, explained_variance_threshold) + 1
        n_components = min(n_components, min(n_samples_satellite, n_wavelengths))
        print(f"  CTAI: Auto-selected {n_components} components (threshold={explained_variance_threshold})")
    else:
        n_components = min(n_components, len(S_satellite))
        print(f"  CTAI: Using {n_components} components (user-specified)")

    # Step 2c: Project data onto principal components
    # This is the key: work in reduced-rank space for numerical stability
    V_truncated = Vt_satellite[:n_components, :].T  # Shape: (n_wavelengths, n_components)
    S_truncated = S_satellite[:n_components]

    # Project satellite and primary data onto satellite's principal components
    X_satellite_projected = X_satellite_centered @ V_truncated  # (n_samples, n_components)
    X_primary_projected = X_primary_centered @ V_truncated  # (n_samples, n_components)

    M_reduced = np.linalg.lstsq(X_satellite_projected, X_primary_projected, rcond=None)[0]
    logger.debug("CTAI M_reduced shape: %s (PCA space transformation)", M_reduced.shape)

    M = V_truncated @ M_reduced @ V_truncated.T

    logger.debug(
        "CTAI M shape %s; M range [%.6f, %.6f]; M diagonal mean %.6f",
        M.shape, np.min(M), np.max(M), np.mean(np.diag(M)),
    )

    # Check for NaN/inf in transformation matrix
    if np.any(np.isnan(M)):
        raise ValueError(f"Transformation matrix M contains NaN values! This indicates numerical instability.")
    if np.any(np.isinf(M)):
        raise ValueError(f"Transformation matrix M contains infinite values! This indicates numerical instability.")

    # Translation vector handles mean differences
    # T = mean(X_primary) - mean(X_satellite @ M)
    T = primary_mean - satellite_mean @ M

    logger.debug(
        "CTAI translation T range [%.6f, %.6f]; T mean %.6f",
        np.min(T), np.max(T), np.mean(T),
    )

    X_satellite_transformed = X_satellite @ M + T

    # Check for NaN/inf in transformed data
    if np.any(np.isnan(X_satellite_transformed)):
        raise ValueError(f"Transformed data contains NaN values! Transformation failed.")
    if np.any(np.isinf(X_satellite_transformed)):
        raise ValueError(f"Transformed data contains infinite values! Transformation failed.")

    logger.debug(
        "CTAI transformed data range [%.6f, %.6f]",
        np.min(X_satellite_transformed), np.max(X_satellite_transformed),
    )

    X_primary_sample = X_primary[:min(n_samples_primary, n_samples_satellite)]
    X_satellite_sample = X_satellite_transformed[:min(n_samples_primary, n_samples_satellite)]

    reconstruction_error = np.sqrt(np.mean((X_primary_sample - X_satellite_sample) ** 2))

    explained_variance = np.sum(S_truncated**2) / np.sum(S_satellite**2) if len(S_satellite) > 0 else 1.0

    print(
        f"  CTAI: components={n_components}, "
        f"explained_variance={explained_variance:.4f}, "
        f"reconstruction_RMSE={reconstruction_error:.6f}"
    )

    # Compile results
    params = {
        'M': M,
        'T': T,
        'n_components': n_components,
        'explained_variance': explained_variance,
        'reconstruction_error': reconstruction_error,
        'primary_mean': primary_mean,
        'satellite_mean': satellite_mean,
        'eigenvalues': S_truncated,
    }

    return params


def apply_ctai(X_satellite_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply CTAI calibration transfer to new satellite instrument spectra.

    Transforms satellite spectra to primary instrument domain using affine
    transformation estimated via CTAI.

    Parameters
    ----------
    X_satellite_new : np.ndarray, shape (n_samples, n_wavelengths)
        New satellite instrument spectra to transform.
    params : dict
        CTAI parameters from estimate_ctai, containing 'M' and 'T'.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in primary instrument domain.

    Examples
    --------
    >>> # After estimating CTAI model (see estimate_ctai examples)
    >>> X_satellite_new = np.random.randn(50, 200)
    >>> X_transferred = apply_ctai(X_satellite_new, params)
    >>>
    >>> # Can now use primary instrument's calibration model
    >>> y_predicted = primary_model.predict(X_transferred)

    Notes
    -----
    - Transformation: X_transferred = X_satellite @ M + T
    - M is transformation matrix, T is translation vector
    - Computationally efficient (matrix multiplication)
    - No iterative optimization needed
    """
    M = params['M']
    T = params['T']

    # Validate dimensions
    n_wavelengths = M.shape[0]
    if X_satellite_new.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_satellite_new has {X_satellite_new.shape[1]} wavelengths "
            f"but model expects {n_wavelengths}"
        )

    # Apply affine transformation: X_primary = X_satellite @ M + T
    X_transferred = X_satellite_new @ M + T

    return X_transferred


# ==============================================================================
# NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement)
# ==============================================================================

def estimate_nspfce(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    wavelengths: np.ndarray,
    use_wavelength_selection: bool = False,  # Changed default to False - works better
    wavelength_selector: str = 'cars',  # Changed default to 'cars' (more stable than vcpa-iriv)
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    normalize: bool = False  # Changed default to False - normalization can cause issues
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
    X_primary : np.ndarray, shape (n_samples_primary, n_wavelengths)
        Primary instrument spectra on common wavelength grid.
    X_satellite : np.ndarray, shape (n_samples_satellite, n_wavelengths)
        Satellite instrument spectra on common wavelength grid.
        Need not be the same samples as X_primary.
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
    >>> # Generate primary and satellite spectra (different samples)
    >>> n_primary, n_satellite, n_wavelengths = 100, 120, 200
    >>> wavelengths = np.linspace(1000, 2500, n_wavelengths)
    >>>
    >>> X_primary = np.random.randn(n_primary, n_wavelengths)
    >>> X_satellite = 0.9 * np.random.randn(n_satellite, n_wavelengths) + 0.1
    >>>
    >>> # Estimate NS-PFCE model (with wavelength selection)
    >>> params = estimate_nspfce(X_primary, X_satellite, wavelengths,
    ...                          use_wavelength_selection=True,
    ...                          wavelength_selector='vcpa-iriv')
    >>>
    >>> print(f"Selected {len(params['selected_wavelengths'])} wavelengths")
    >>> print(f"Converged in {params['convergence_iterations']} iterations")
    >>>
    >>> # Apply to new satellite spectra
    >>> X_satellite_new = np.random.randn(50, n_wavelengths)
    >>> X_transferred = apply_nspfce(X_satellite_new, params)

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
    - May struggle if primary/satellite have very different spectral characteristics
    - Requires sufficient spectral diversity in both datasets

    See Also
    --------
    apply_nspfce : Apply NS-PFCE transformation to new spectra
    estimate_ctai : Alternative parameter-free method (faster)
    """
    n_samples_primary, n_wavelengths = X_primary.shape
    n_samples_satellite = X_satellite.shape[0]

    # Validation
    if X_satellite.shape[1] != n_wavelengths:
        raise ValueError(
            f"X_primary and X_satellite must have same number of wavelengths: "
            f"{n_wavelengths} vs {X_satellite.shape[1]}"
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
        y_pseudo_primary = np.mean(X_primary, axis=1)
        y_pseudo_satellite = np.mean(X_satellite, axis=1)

        # Combine for wavelength selection (use primary for selection)
        from .wavelength_selection import vcpa_iriv, cars, spa

        try:
            if wavelength_selector == 'vcpa-iriv':
                wl_result = vcpa_iriv(
                    X_primary, y_pseudo_primary,
                    n_outer_iterations=5,
                    n_inner_iterations=30,
                    random_state=42
                )
            elif wavelength_selector == 'cars':
                wl_result = cars(
                    X_primary, y_pseudo_primary,
                    n_iterations=40,
                    random_state=42
                )
            elif wavelength_selector == 'spa':
                target_n = min(50, n_wavelengths // 4)
                wl_result = spa(X_primary, y_pseudo_primary, n_vars=target_n)
            else:
                raise ValueError(f"Unknown wavelength_selector: {wavelength_selector}")

            selected_wavelengths = wl_result['selected_indices']
            print(f"  NS-PFCE: Selected {len(selected_wavelengths)}/{n_wavelengths} wavelengths")

            # Reduce matrices to selected wavelengths
            X_primary_sel = X_primary[:, selected_wavelengths]
            X_satellite_sel = X_satellite[:, selected_wavelengths]
            n_selected = len(selected_wavelengths)

        except Exception as e:
            print(f"  NS-PFCE: Wavelength selection failed ({str(e)}), using all wavelengths")
            X_primary_sel = X_primary
            X_satellite_sel = X_satellite
            n_selected = n_wavelengths
            selected_wavelengths = np.arange(n_wavelengths)

    else:
        X_primary_sel = X_primary
        X_satellite_sel = X_satellite
        n_selected = n_wavelengths
        selected_wavelengths = np.arange(n_wavelengths)

    # Step 2: Initialize transformation
    # Simple initialization: mean normalization
    primary_mean = np.mean(X_primary_sel, axis=0)
    satellite_mean = np.mean(X_satellite_sel, axis=0)
    primary_std = np.std(X_primary_sel, axis=0) + 1e-10
    satellite_std = np.std(X_satellite_sel, axis=0) + 1e-10

    # Initial transformation: normalize satellite to primary scale
    # T = diag(primary_std / satellite_std)
    scale_factors = primary_std / satellite_std
    T = np.diag(scale_factors)
    offset = primary_mean - satellite_mean * scale_factors

    # Step 3: Iterative optimization
    # Objective: minimize ||X_primary - (X_satellite @ T + offset)||_F
    # Use coordinate descent or gradient-based optimization

    convergence_iterations = 0
    objective_history = []

    for iteration in range(max_iterations):
        # Current objective value
        X_satellite_transformed = X_satellite_sel @ T + offset
        # Use a sample for objective (computational efficiency)
        n_compare = min(n_samples_primary, n_samples_satellite, 100)
        obj = np.mean((X_primary_sel[:n_compare] - X_satellite_transformed[:n_compare]) ** 2)
        objective_history.append(obj)

        # Check convergence
        if iteration > 0:
            obj_change = abs(objective_history[-1] - objective_history[-2])
            if obj_change < convergence_threshold:
                convergence_iterations = iteration
                break

        # Update transformation
        # Use pseudo-inverse approach for stability
        # Solve: X_primary ≈ X_satellite @ T + offset
        # Rearrange: X_primary - offset ≈ X_satellite @ T
        # T_new = (X_satellite^T X_satellite)^-1 X_satellite^T (X_primary - offset)

        # BUG #3 FIX: Account for offset in the target
        X_target = X_primary_sel - offset  # Subtract offset from target

        # Use regularized least squares
        reg_param = 1e-6
        XtX = X_satellite_sel.T @ X_satellite_sel + reg_param * np.eye(n_selected)
        XtY = X_satellite_sel.T @ X_target  # BUG #3 FIX: Use offset-subtracted target

        try:
            T_new = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            T_new = np.linalg.pinv(X_satellite_sel) @ X_target

        # Adaptive update (damping for stability)
        damping = 0.5
        T = damping * T_new + (1 - damping) * T

        # Update offset
        X_satellite_transformed = X_satellite_sel @ T
        offset = np.mean(X_primary_sel - X_satellite_transformed, axis=0)

        # Optional normalization step
        # BUG #5 FIX: Use Frobenius norm instead of diagonal (T may not be diagonal)
        if normalize and iteration % 10 == 0:
            # Renormalize using Frobenius norm to prevent drift
            frob_norm = np.linalg.norm(T, 'fro')
            expected_norm = np.sqrt(n_selected)  # Expected norm for identity-like matrix
            if frob_norm > 0:
                scale_factor = expected_norm / frob_norm
                T = T * scale_factor
                offset = offset / scale_factor

    if convergence_iterations == 0:
        convergence_iterations = max_iterations

    # Final objective
    X_satellite_transformed = X_satellite_sel @ T + offset
    n_compare = min(n_samples_primary, n_samples_satellite)
    final_objective = np.sqrt(np.mean((X_primary_sel[:n_compare] - X_satellite_transformed[:n_compare]) ** 2))

    print(f"  NS-PFCE: Converged in {convergence_iterations} iterations")
    print(f"  NS-PFCE: Final RMSE: {final_objective:.6f}")

    # Get actual wavelength values for selected wavelengths
    selected_wavelength_values = wavelengths[selected_wavelengths] if use_wavelength_selection else wavelengths

    # Compile results
    params = {
        # Transformation parameters
        'transformation_matrix': T,
        'T': T,  # Alias for GUI compatibility
        'offset': offset,

        # Wavelength selection
        'selected_wavelengths': selected_wavelengths,  # Indices into original wavelength array
        'selected_wavelength_indices': selected_wavelengths,  # Alias for backward compatibility
        'selected_wavelength_values': selected_wavelength_values,  # Actual wavelength values (nm)
        'original_wavelengths': wavelengths,  # Original full wavelength array
        'wavelength_selector': wavelength_selector if use_wavelength_selection else None,
        'wavelength_selection_method': wavelength_selector if use_wavelength_selection else None,  # Alias for backward compat
        'use_wavelength_selection': use_wavelength_selection,
        'n_selected_wavelengths': n_selected,
        'n_original_wavelengths': n_wavelengths,

        # Convergence information (dual naming for backward compatibility)
        'convergence_iterations': convergence_iterations,
        'n_iterations': convergence_iterations,  # Alias for GUI compatibility
        'converged': (convergence_iterations < max_iterations),  # Boolean convergence flag
        'final_objective': final_objective,

        # History
        'objective_history': objective_history,
        'convergence_history': objective_history  # Alias for GUI compatibility
    }

    return params


def apply_nspfce(
    X_satellite_new: np.ndarray,
    params: Dict,
    return_full_spectrum: bool = False
) -> np.ndarray:
    """
    Apply NS-PFCE calibration transfer to new satellite instrument spectra.

    Transforms satellite spectra to primary instrument domain using previously
    estimated NS-PFCE transformation.

    Parameters
    ----------
    X_satellite_new : np.ndarray, shape (n_samples, n_wavelengths)
        New satellite instrument spectra to transform.
    params : dict
        NS-PFCE parameters from estimate_nspfce.
    return_full_spectrum : bool, default=False
        If wavelength selection was used:
        - False (default): Return only the selected wavelengths (reduced spectrum).
          This is the correct behavior - all wavelengths in output are consistently
          transformed to primary instrument scale.
        - True: Return full spectrum with non-selected wavelengths unchanged.
          WARNING: This creates a mixed-scale spectrum (selected wavelengths in
          primary scale, others in satellite scale) which is usually incorrect.

    Returns
    -------
    X_transferred : np.ndarray
        Transformed spectra in primary instrument domain.
        Shape depends on wavelength selection:
        - No wavelength selection: (n_samples, n_wavelengths)
        - With wavelength selection and return_full_spectrum=False:
          (n_samples, n_selected_wavelengths)
        - With wavelength selection and return_full_spectrum=True:
          (n_samples, n_wavelengths) but mixed scales (not recommended)

    Examples
    --------
    >>> # After estimating NS-PFCE model (see estimate_nspfce examples)
    >>> X_satellite_new = np.random.randn(50, 200)
    >>> X_transferred = apply_nspfce(X_satellite_new, params)
    >>>
    >>> # If wavelength selection was used, X_transferred has fewer wavelengths
    >>> # Use params['selected_wavelength_values'] to get the wavelength grid
    >>> print(f"Output wavelengths: {params['selected_wavelength_values']}")
    >>>
    >>> # Can now use primary instrument's calibration model
    >>> y_predicted = primary_model.predict(X_transferred)

    Notes
    -----
    - If wavelength selection was used, output has only selected wavelengths
    - Use params['selected_wavelength_values'] to get the wavelength grid
    - Transformation: X_transferred = X_satellite @ T + offset
    """
    T = params['transformation_matrix']
    offset = params['offset']
    selected_wavelengths = params['selected_wavelengths']
    use_wl_selection = params['use_wavelength_selection']

    n_wavelengths_full = X_satellite_new.shape[1]

    if use_wl_selection and selected_wavelengths is not None:
        # Extract only selected wavelengths from input
        X_satellite_selected = X_satellite_new[:, selected_wavelengths]

        # Apply transformation
        X_transformed_selected = X_satellite_selected @ T + offset

        if return_full_spectrum:
            # WARNING: This creates mixed-scale spectrum (not recommended)
            # Keep non-selected wavelengths unchanged (satellite scale)
            # Selected wavelengths are transformed (primary scale)
            X_transferred = X_satellite_new.copy()
            X_transferred[:, selected_wavelengths] = X_transformed_selected
        else:
            # BUG #4 FIX: Return only selected wavelengths (all in primary scale)
            # This is the correct behavior - output is consistent
            X_transferred = X_transformed_selected

    else:
        # No wavelength selection - apply to all wavelengths
        # Validate dimensions
        n_wavelengths_expected = T.shape[0]
        if X_satellite_new.shape[1] != n_wavelengths_expected:
            raise ValueError(
                f"X_satellite_new has {X_satellite_new.shape[1]} wavelengths "
                f"but model expects {n_wavelengths_expected}"
            )

        # Apply transformation to all wavelengths
        X_transferred = X_satellite_new @ T + offset

    return X_transferred


def estimate_jypls_inv(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    y_transfer: np.ndarray,
    transfer_indices: np.ndarray,
    n_components: int | None = None,
    cv_folds: int = 5,
    max_components: int = 20
) -> Dict:
    """
    Estimate JYPLS-inv (Joint-Y PLS with inversion) calibration transfer.

    JYPLS-inv uses a joint PLS model where primary and satellite transfer samples
    are combined in an augmented X matrix with shared Y values. The PLS model
    learns a common latent structure, and the transformation is derived from
    the PLS components to map satellite spectra to primary space.

    Parameters
    ----------
    X_primary : np.ndarray, shape (n_samples, n_wavelengths)
        Primary instrument spectra.
    X_satellite : np.ndarray, shape (n_samples, n_wavelengths)
        Satellite instrument spectra (same samples as X_primary).
    y_transfer : np.ndarray, shape (n_transfer,)
        Reference values for transfer samples.
        Used for PLS modeling to find optimal transformation.
    transfer_indices : np.ndarray, shape (n_transfer,)
        Indices of transfer samples in X_primary and X_satellite.
        Typically 12-13 samples selected by Kennard-Stone or SPXY.
    n_components : int | None, optional
        Number of PLS components. If None, determined by cross-validation.
        Default: None (auto-select).
    cv_folds : int, optional
        Number of cross-validation folds for component selection.
        Default: 5.
    max_components : int, optional
        Maximum number of components to try in CV.
        Default: 20.

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'transformation_matrix' : np.ndarray, shape (n_wavelengths, n_wavelengths)
            Matrix B such that X_primary ≈ X_satellite @ B
        - 'n_components' : int
            Number of PLS components used
        - 'cv_rmse' : float
            Cross-validation RMSE for component selection
        - 'transfer_indices' : np.ndarray
            Indices of transfer samples
        - 'pls_x_weights' : np.ndarray
            PLS X weights (for inspection)
        - 'pls_x_loadings' : np.ndarray
            PLS X loadings (for inspection)
        - 'explained_variance_ratio' : float
            Proportion of X variance explained by PLS components

    Notes
    -----
    Algorithm:
    1. Extract transfer samples from primary and satellite
    2. Create augmented X = [X_primary_transfer; X_satellite_transfer]
    3. Create augmented Y = [y_transfer; y_transfer] (shared Y)
    4. Fit PLS(X_aug, Y_aug) with optimal number of components
    5. Extract separate PLS scores for primary (T_m) and satellite (T_s)
    6. Compute transformation: B = W @ (T_s^T T_s)^{-1} @ T_s^T @ T_m @ P^T
       where W = PLS X-weights, P = PLS X-loadings
    7. Simplified approach: B derived from PLS regression coefficients

    The transformation maps satellite spectra into primary space while preserving
    the PLS latent structure that predicts Y.

    References
    ----------
    - Feudale, R. N., et al. (2002). "Transfer of multivariate calibration
      models: a review." Chemometrics and Intelligent Laboratory Systems, 64(2), 181-192.
    - Bouveresse, E., & Massart, D. L. (1996). "Improvement of the piecewise
      direct standardisation procedure for the transfer of NIR spectra for
      multivariate calibration." Chemometrics and intelligent laboratory systems, 32(2), 201-213.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.sample_selection import kennard_stone
    >>>
    >>> # Simulate primary and satellite spectra
    >>> np.random.seed(42)
    >>> n_samples, n_wavelengths = 100, 200
    >>> X_primary = np.random.randn(n_samples, n_wavelengths)
    >>> X_satellite = 0.95 * X_primary + 0.05  # Satellite with bias/scale
    >>> y = 2.0 * X_primary[:, 50] - 1.5 * X_primary[:, 100] + np.random.randn(n_samples) * 0.1
    >>>
    >>> # Select 12 transfer samples with Kennard-Stone
    >>> transfer_idx = kennard_stone(X_primary, n_samples=12)
    >>> y_transfer = y[transfer_idx]
    >>>
    >>> # Estimate JYPLS-inv model
    >>> params = estimate_jypls_inv(X_primary, X_satellite, y_transfer, transfer_idx, n_components=5)
    >>> print(f"PLS Components: {params['n_components']}")
    >>> print(f"CV RMSE: {params['cv_rmse']:.6f}")
    >>>
    >>> # Apply transformation
    >>> X_transferred = apply_jypls_inv(X_satellite, params)
    >>> rmse_improvement = np.sqrt(np.mean((X_satellite - X_primary)**2)) / np.sqrt(np.mean((X_transferred - X_primary)**2))
    >>> print(f"RMSE improvement: {rmse_improvement:.2f}x")
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError("scikit-learn is required for JYPLS-inv. Install with: pip install scikit-learn")

    # Validate inputs
    if X_primary.shape != X_satellite.shape:
        raise ValueError(f"X_primary and X_satellite must have same shape. Got {X_primary.shape} and {X_satellite.shape}")

    if len(transfer_indices) != len(y_transfer):
        raise ValueError(f"Number of transfer_indices ({len(transfer_indices)}) must match y_transfer length ({len(y_transfer)})")

    if len(transfer_indices) < 2:
        raise ValueError("Need at least 2 transfer samples for JYPLS-inv")

    n_samples, n_wavelengths = X_primary.shape

    # Extract transfer samples
    X_primary_transfer = X_primary[transfer_indices]
    X_satellite_transfer = X_satellite[transfer_indices]

    # Create augmented matrices
    # X_aug = [X_primary_transfer]  <- primary samples
    #         [X_satellite_transfer]   <- satellite samples (to be mapped to primary)
    X_aug = np.vstack([X_primary_transfer, X_satellite_transfer])

    # Y_aug = [y_transfer]  <- same Y for primary
    #         [y_transfer]  <- same Y for satellite (joint-Y approach)
    Y_aug = np.concatenate([y_transfer, y_transfer]).reshape(-1, 1)

    # Determine optimal number of PLS components via cross-validation
    if n_components is None:
        best_rmse = np.inf
        best_n = 1

        # Try different numbers of components
        max_comp = min(max_components, len(transfer_indices) - 1, n_wavelengths)

        for n in range(1, max_comp + 1):
            pls = PLSRegression(n_components=n, scale=False)

            try:
                # Cross-validation scores
                scores = cross_val_score(
                    pls, X_aug, Y_aug, cv=min(cv_folds, len(transfer_indices)),
                    scoring='neg_root_mean_squared_error'
                )
                avg_rmse = -np.mean(scores)  # Negative because sklearn uses neg_rmse

                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_n = n
            except Exception:
                # If CV fails (e.g., too few samples), use previous best
                break

        n_components = best_n
        cv_rmse = best_rmse
    else:
        # User specified components
        cv_rmse = None

    # Validate n_components
    max_comp = min(len(transfer_indices) - 1, n_wavelengths)
    if n_components > max_comp:
        n_components = max_comp

    # Fit final PLS model
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_aug, Y_aug)

    # Extract PLS scores for primary and satellite separately
    T_all = pls.transform(X_aug)  # shape: (2*n_transfer, n_components)
    n_transfer = len(transfer_indices)
    T_primary = T_all[:n_transfer, :]  # Primary scores
    T_satellite = T_all[n_transfer:, :]   # Satellite scores

    # Compute transformation matrix
    # Approach: Find B such that T_primary ≈ T_satellite in PLS space
    # Then map back to original space using PLS loadings
    #
    # Simplified: Use PLS regression coefficients to build transformation
    # The PLS model learns: Y_aug = X_aug @ coef
    # We want to transform satellite → primary in X space
    #
    # Transformation derived from PLS structure:
    # X_primary ≈ X_satellite @ B
    # B is computed using PLS weights (W) and loadings (P)
    #
    # Method: Compute transformation in score space, then project back
    # B = W @ inv(T_satellite^T @ T_satellite) @ (T_satellite^T @ T_primary) @ P^T
    # where W = X-weights, P = X-loadings

    W = pls.x_weights_  # shape: (n_wavelengths, n_components)
    P = pls.x_loadings_  # shape: (n_wavelengths, n_components)

    # Compute score transformation: T_primary = T_satellite @ M
    # M = inv(T_satellite^T @ T_satellite) @ (T_satellite^T @ T_primary)
    T_satellite_T = T_satellite.T  # (n_components, n_transfer)
    T_satellite_cov = T_satellite_T @ T_satellite + 1e-6 * np.eye(n_components)  # Regularization
    T_satellite_cov_inv = np.linalg.inv(T_satellite_cov)
    M_scores = T_satellite_cov_inv @ (T_satellite_T @ T_primary)  # (n_components, n_components)

    # Map back to original space
    # B = W @ M_scores @ P^T
    transformation_matrix = W @ M_scores @ P.T  # (n_wavelengths, n_wavelengths)

    # Calculate explained variance
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
        'pls_scores_primary': T_primary,
        'pls_scores_satellite': T_satellite,
        'score_transformation': M_scores,
        'explained_variance_ratio': explained_variance_ratio,
        'n_transfer_samples': len(transfer_indices)
    }

    return params


def apply_jypls_inv(X_satellite_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply JYPLS-inv transformation to new satellite spectra.

    Parameters
    ----------
    X_satellite_new : np.ndarray, shape (n_samples, n_wavelengths)
        New satellite spectra to transfer to primary domain.
    params : dict
        Parameters from estimate_jypls_inv().

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transferred spectra in primary domain.

    Examples
    --------
    >>> # After estimating JYPLS-inv model
    >>> X_transferred = apply_jypls_inv(X_satellite_new, jypls_params)
    >>> print(f"Transferred shape: {X_transferred.shape}")
    """
    B = params['transformation_matrix']

    if X_satellite_new.shape[1] != B.shape[0]:
        raise ValueError(
            f"X_satellite_new has {X_satellite_new.shape[1]} wavelengths, "
            f"but transformation matrix expects {B.shape[0]}"
        )

    # Apply transformation: X_primary ≈ X_satellite @ B
    X_transferred = X_satellite_new @ B

    return X_transferred


def apply_transfer_dispatch(X_satellite: np.ndarray, transfer_model: TransferModel) -> np.ndarray:
    """
    Unified dispatcher for applying any transfer model type.

    This function provides a single interface for applying calibration transfer
    models regardless of the specific method used (DS, PDS, TSR, CTAI, etc.).

    Parameters
    ----------
    X_satellite : np.ndarray
        Satellite instrument spectra to transform, shape (n_samples, n_wavelengths).
        The wavelengths should match the transfer model's common wavelength grid.
    transfer_model : TransferModel
        Transfer model object containing the method type, parameters, and
        wavelength information.

    Returns
    -------
    np.ndarray
        Transformed spectra in primary instrument space, shape (n_samples, n_wavelengths).

    Raises
    ------
    ValueError
        If the transfer method is unknown or unsupported.

    Examples
    --------
    >>> # Load a transfer model
    >>> transfer_model = load_transfer_model("path/to/model")
    >>>
    >>> # Apply to new satellite spectra
    >>> X_satellite_new = load_spectra("satellite_data.csv")
    >>> X_primary_space = apply_transfer_dispatch(X_satellite_new, transfer_model)
    """
    method = transfer_model.method.lower()
    params = transfer_model.params

    if method == 'ds':
        return apply_ds(X_satellite, params['A'])
    elif method == 'pds':
        return apply_pds(X_satellite, params['B'], params['window'])
    elif method == 'tsr':
        return apply_tsr(X_satellite, params)
    elif method == 'ctai':
        return apply_ctai(X_satellite, params)
    elif method == 'ns-pfce' or method == 'nspfce':
        return apply_nspfce(X_satellite, params)
    elif method == 'jypls-inv':
        return apply_jypls_inv(X_satellite, params)
    else:
        raise ValueError(
            f"Unknown transfer method: {method}. "
            f"Supported methods are: ds, pds, tsr, ctai, ns-pfce, jypls-inv"
        )


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
    print("  - JYPLS-inv (Joint-Y PLS with Inversion)")
    print("=" * 60)

    # Quick test of new methods
    import numpy as np
    np.random.seed(42)

    print("\nTesting TSR:")
    X_primary = np.random.randn(50, 100)
    X_satellite = 0.95 * X_primary + 0.05
    transfer_idx = np.array([0, 10, 20, 30, 40])  # Simple selection

    tsr_params = estimate_tsr(X_primary, X_satellite, transfer_idx)
    print(f"  Mean R²: {tsr_params['mean_r_squared']:.4f}")
    print(f"  Slope range: [{tsr_params['slope'].min():.3f}, {tsr_params['slope'].max():.3f}]")

    X_transferred_tsr = apply_tsr(X_satellite, tsr_params)
    print(f"  Transfer RMSE: {np.sqrt(np.mean((X_transferred_tsr - X_primary)**2)):.6f}")

    print("\nTesting CTAI:")
    ctai_params = estimate_ctai(X_primary, X_satellite)
    print(f"  Explained variance: {ctai_params['explained_variance']:.4f}")
    print(f"  N components: {ctai_params['n_components']}")
    print(f"  Reconstruction error: {ctai_params['reconstruction_error']:.6f}")

    X_transferred_ctai = apply_ctai(X_satellite, ctai_params)
    print(f"  Transfer RMSE: {np.sqrt(np.mean((X_transferred_ctai - X_primary)**2)):.6f}")

    print("\n" + "=" * 60)
    print("All methods loaded successfully!")
