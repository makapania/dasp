"""
spectral_predict.calibration_transfer
=====================================

Backend-only module for calibration transfer between instruments using
Direct Standardization (DS) and Piecewise Direct Standardization (PDS).

This file is a skeleton: function bodies are stubs (`pass`) for another
agent to fill in with real implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np


MethodType = Literal["ds", "pds"]


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


if __name__ == "__main__":
    # Optional: simple self-test or placeholder for future examples.
    print("calibration_transfer skeleton loaded.")
