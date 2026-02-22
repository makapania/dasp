"""Bias/slope correction for spectral predictions (ASTM E1655).

Provides linear and nonlinear post-hoc correction that can be saved
with a model and auto-applied during prediction.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional


def compute_bias_slope(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute linear bias/slope correction: y_true = bias + slope * y_pred.

    Parameters
    ----------
    y_true : array-like
        Reference (actual) values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    dict
        Correction data with keys:
        - method: 'linear'
        - bias: float intercept
        - slope: float slope
        - metrics_original: dict with R2, RMSE, MAE before correction
        - metrics_corrected: dict with R2, RMSE, MAE after correction
        - prediction_scale: 'original' (documents that correction is in original y scale)
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Original metrics
    metrics_original = {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
    }

    # Fit: y_true = bias + slope * y_pred
    coeffs = np.polyfit(y_pred, y_true, 1)
    slope, bias = coeffs

    # Apply correction to get corrected predictions
    y_corrected = bias + slope * y_pred

    metrics_corrected = {
        'R2': float(r2_score(y_true, y_corrected)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_corrected))),
        'MAE': float(mean_absolute_error(y_true, y_corrected)),
    }

    return {
        'method': 'linear',
        'bias': float(bias),
        'slope': float(slope),
        'metrics_original': metrics_original,
        'metrics_corrected': metrics_corrected,
        'prediction_scale': 'original',
    }


def compute_nonlinear_correction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "polynomial",
    degree: int = 2,
) -> dict:
    """Compute nonlinear correction: y_true = f(y_pred).

    Parameters
    ----------
    y_true : array-like
        Reference (actual) values.
    y_pred : array-like
        Predicted values.
    method : str
        'polynomial' or 'spline'. Spline is approximated as a high-degree
        polynomial for JSON serialization.
    degree : int
        Polynomial degree (2-4 recommended).

    Returns
    -------
    dict
        Correction data with keys:
        - method: 'nonlinear'
        - degree: int
        - coefficients: list of floats (highest degree first)
        - metrics_original, metrics_corrected: dicts
        - prediction_scale: 'original'
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    n = len(y_true)
    if n < degree + 1:
        raise ValueError(f"Insufficient data (n={n}) for degree-{degree} correction")

    metrics_original = {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
    }

    if method == "spline":
        if n < 4:
            raise ValueError("Insufficient data for spline (n < 4)")
        from scipy.interpolate import UnivariateSpline

        order = np.argsort(y_pred)
        xs, ys = y_pred[order], y_true[order]
        spline = UnivariateSpline(xs, ys, s=None, k=min(3, n - 1))
        # Approximate spline as polynomial for serialization
        x_range = np.linspace(xs.min(), xs.max(), 500)
        y_range = spline(x_range)
        coeffs = np.polyfit(x_range, y_range, min(degree, 6))
    else:
        coeffs = np.polyfit(y_pred, y_true, degree)

    poly = np.poly1d(coeffs)
    y_corrected = poly(y_pred)

    metrics_corrected = {
        'R2': float(r2_score(y_true, y_corrected)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_corrected))),
        'MAE': float(mean_absolute_error(y_true, y_corrected)),
    }

    return {
        'method': 'nonlinear',
        'degree': int(degree),
        'coefficients': [float(c) for c in coeffs],
        'metrics_original': metrics_original,
        'metrics_corrected': metrics_corrected,
        'prediction_scale': 'original',
    }


def apply_correction(y_pred: np.ndarray, correction_data: dict) -> np.ndarray:
    """Apply a saved correction to predictions.

    Parameters
    ----------
    y_pred : array-like
        Raw predicted values from the model.
    correction_data : dict
        Correction data from compute_bias_slope() or compute_nonlinear_correction().

    Returns
    -------
    np.ndarray
        Corrected predictions.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    method = correction_data.get('method', 'linear')

    if method == 'linear':
        bias = correction_data['bias']
        slope = correction_data['slope']
        return bias + slope * y_pred
    elif method == 'nonlinear':
        coeffs = correction_data['coefficients']
        poly = np.poly1d(coeffs)
        return poly(y_pred)
    else:
        raise ValueError(f"Unknown correction method: {method}")
