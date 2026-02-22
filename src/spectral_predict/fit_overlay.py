"""Fit line overlay computation for pred-vs-actual plots.

Supports linear, polynomial, and smooth (kernel regression) fits with edge case guards.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def compute_fit_line(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "linear",
    poly_degree: int = 2,
    direction: str = "prediction",
    n_points: int = 200,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, float]:
    """Compute a fit line through pred-vs-actual scatter data.

    Parameters
    ----------
    y_true : array-like
        Reference (actual) values.
    y_pred : array-like
        Predicted values.
    method : str
        One of 'linear', 'poly2', 'poly3', 'poly4', 'smooth'.
    poly_degree : int
        Polynomial degree (used only when method starts with 'poly').
    direction : str
        'prediction' — fit y_pred = f(y_true).  Natural scatter reading.
        'correction' — fit y_true = f(y_pred).  Used for bias/slope correction.
    n_points : int
        Number of points in the returned fit curve.

    Returns
    -------
    x_fit : ndarray or None
        x-coordinates of the fit curve.
    y_fit : ndarray or None
        y-coordinates of the fit curve.
    equation_str : str
        Human-readable equation or error message.
    r2_fit : float
        R² of the fit (nan on error).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Remove NaN pairs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)
    if n < 2:
        return None, None, "Insufficient data (n < 2)", float("nan")

    # Set up x/y based on direction
    if direction == "correction":
        x, y = y_pred, y_true  # y_true = f(y_pred)
    else:
        x, y = y_true, y_pred  # y_pred = f(y_true)

    # Guard: constant x values
    if np.ptp(x) < 1e-12:
        return None, None, "Constant x values", float("nan")

    # Determine degree from method string
    method_lower = method.lower().replace(" ", "")
    if method_lower == "linear":
        degree = 1
    elif method_lower in ("poly2", "polynomial(2)"):
        degree = 2
    elif method_lower in ("poly3", "polynomial(3)"):
        degree = 3
    elif method_lower in ("poly4", "polynomial(4)"):
        degree = 4
    elif method_lower == "smooth":
        degree = None  # handled separately
    else:
        degree = poly_degree

    # --- Smooth (Gaussian kernel regression) path ---
    if degree is None:
        if n < 4:
            return None, None, "Insufficient data for smooth (n < 4)", float("nan")
        try:
            order = np.argsort(x)
            xs, ys = x[order], y[order]

            # Gaussian kernel regression (Nadaraya-Watson estimator)
            x_fit = np.linspace(xs.min(), xs.max(), n_points)
            bandwidth = (xs.max() - xs.min()) * 0.15  # 15% of data range

            diff = xs[np.newaxis, :] - x_fit[:, np.newaxis]  # (n_points, n)
            weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
            y_fit = (weights @ ys) / weights.sum(axis=1)

            # R² using numpy interpolation back to original x positions
            y_hat = np.interp(x, x_fit, y_fit)
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            return x_fit, y_fit, "Smooth", r2
        except Exception as e:
            return None, None, f"Smooth error: {e}", float("nan")

    # --- Polynomial / linear path ---
    if n < degree + 1:
        return None, None, f"Insufficient data (n={n} < degree+1={degree + 1})", float("nan")

    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(x.min(), x.max(), n_points)
        y_fit = poly(x_fit)

        # R²
        y_hat = poly(x)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Build equation string
        eq = _format_polynomial(coeffs)
        return x_fit, y_fit, eq, r2

    except Exception as e:
        return None, None, f"Fit error: {e}", float("nan")


def _format_polynomial(coeffs: np.ndarray) -> str:
    """Format polynomial coefficients as a human-readable equation string."""
    degree = len(coeffs) - 1
    if degree == 1:
        slope, intercept = coeffs
        sign = "+" if intercept >= 0 else "-"
        return f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"

    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{c:+.4f}")
        elif power == 1:
            terms.append(f"{c:+.4f}x")
        else:
            terms.append(f"{c:+.4f}x^{power}")

    eq = " ".join(terms) if terms else "0"
    # Clean up leading +
    if eq.startswith("+"):
        eq = eq[1:]
    return f"y = {eq.strip()}"
