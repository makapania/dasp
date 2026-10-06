"""
spectral_predict.calibration_transfer
=====================================

Backend-only module for calibration transfer between instruments.

Supported methods (internal key -> what the code does). The keys are stored in
saved transfer models, so they stay as they are even where the historical name is
misleading; user-facing text should come from ``method_display_name``.

- ``'ds'``: Direct Standardization (ridge-regularised full matrix, or the centred
  dual form from ``estimate_ds_dual``).
- ``'pds'``: Piecewise Direct Standardization (legacy uncentred, or centred low
  rank from ``estimate_pds_lowrank``).
- ``'tsr'``: per-wavelength slope/bias standardization. Not trimmed scores
  regression (Folch-Fortuny et al. 2017), which is also abbreviated TSR.
- ``'ctai'``: paired regression in the satellite PCA space ("PC-DS"). Not the
  published standard-free CTAI (Zhao et al. 2019, Molecules 24(9):1802); it needs
  the same standards measured on both instruments.
- ``'nspfce'``: iterative ridge DS, a dasp heuristic. Not PFCE/NS-PFCE.
- ``'jypls-inv'``: experimental PLS score mapping; disabled in the GUI.

Every method here is fitted on paired rows: row i of the primary matrix and row i
of the satellite matrix must be the same physical standard. To check whether a
transfer works, and to compare methods, use ``transfer_evaluation.evaluate_transfer``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)


MethodType = Literal["ds", "pds", "tsr", "ctai", "nspfce", "jypls-inv"]

DEFAULT_METHOD: str = "tsr"
"""Method key the GUI selects by default (per-wavelength slope/bias)."""

METHOD_SHORT_LABELS: dict[str, str] = {
    "ds": "DS",
    "pds": "PDS",
    "tsr": "Slope/bias per wavelength",
    "ctai": "PC-DS",
    "nspfce": "Iterative ridge DS",
    "ns-pfce": "Iterative ridge DS",
    "jypls-inv": "JYPLS-inv (experimental)",
}
"""Short user-facing labels (radio buttons, status lines), keyed by method key."""

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "ds": "Direct Standardization (DS)",
    "pds": "Piecewise Direct Standardization (PDS)",
    "tsr": "Per-wavelength slope/bias standardization",
    "ctai": "Paired regression in satellite PCA space (PC-DS)",
    "nspfce": "Iterative ridge DS (dasp heuristic)",
    "ns-pfce": "Iterative ridge DS (dasp heuristic)",
    "jypls-inv": "JYPLS-inv score mapping (experimental)",
}
"""Full user-facing names, keyed by method key."""


def method_display_name(method: str, short: bool = False) -> str:
    """Return the user-facing name for a stored transfer-method key.

    Args:
        method: Internal key as stored in ``TransferModel.method`` (e.g. ``'ctai'``).
        short: Return the short label instead of the full name.

    Returns:
        The display name, or ``method.upper()`` for an unknown key.
    """
    table = METHOD_SHORT_LABELS if short else METHOD_DISPLAY_NAMES
    return table.get(str(method).lower(), str(method).upper())


def select_transfer_standards(X_primary: np.ndarray, n_standards: int | None = None) -> np.ndarray:
    """Choose which paired standards a transfer is fitted on.

    Args:
        X_primary: Primary-instrument spectra of the loaded paired standards,
            shape (n_pairs, n_wavelengths).
        n_standards: How many to use. ``None`` (or n_pairs) uses every loaded pair.
            A smaller value picks that many by Kennard-Stone on the primary spectra.

    Returns:
        Row indices into ``X_primary`` (and the matching satellite rows).

    Raises:
        ValueError: If fewer than 2 standards are requested or available, or more
            are requested than are loaded.
    """
    from .sample_selection import kennard_stone

    n_pairs = X_primary.shape[0]
    if n_pairs < 2:
        raise ValueError(f"Need at least 2 paired standards, got {n_pairs}")
    if n_standards is None or n_standards == n_pairs:
        return np.arange(n_pairs)
    if n_standards > n_pairs:
        raise ValueError(
            f"Asked for {n_standards} transfer standards but only {n_pairs} pairs are loaded"
        )
    if n_standards < 2:
        raise ValueError(f"Need at least 2 transfer standards, got {n_standards}")
    return kennard_stone(np.asarray(X_primary, dtype=float), n_samples=int(n_standards))


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


# ==============================================================================
# Centred low-rank PDS and dual-form DS (2026-10, F2/CT1-CT2)
#
# New-form parameters use keys the pre-2026-10 apply code does not read
# ('B_centred', 'ds_form'), so an older dasp build fails with a KeyError instead of
# applying the map without its offset.
# ==============================================================================

TRANSFER_FORMAT_VERSION: int = 2
"""``meta['format_version']`` written for centred PDS and dual-form DS models."""


def estimate_pds_lowrank(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    window: int = 11,
    rank: int | None = None,
) -> Dict:
    """Estimate centred, low-rank Piecewise Direct Standardization.

    For each wavelength i the satellite window (channels i-k..i+k) and the primary
    channel i are mean-centred over the standards, and the local coefficients are
    fitted by truncated SVD (principal components regression) of the centred
    window. An offset per wavelength restores the means, so the map is affine:
    ``x_primary[i] ≈ x_satellite[window] @ b_i + offset[i]``. Capping the rank below
    the number of standards keeps the fit from chasing noise when there are few
    standards (Wang, Veltkamp & Kowalski 1991, *Anal Chem* 63(23):2750-2756).

    Args:
        X_primary: Primary spectra of the paired standards, (n, p).
        X_satellite: Satellite spectra of the same standards, same row order.
        window: Odd window width 2k+1. Edge windows are truncated, as in
            ``estimate_pds``.
        rank: Components per window. ``None`` uses the most the data allow,
            ``n - 1`` (one degree of freedom goes to centring). It is also capped per
            window by the window's width and its numerical rank.

    Returns:
        Params for ``TransferModel(method='pds')``: ``B_centred`` (p, window),
        ``offset`` (p,), ``window``, ``rank`` (the resolved integer cap) and
        ``centred`` (True). Apply with ``apply_pds_centred`` or
        ``apply_transfer_dispatch``.

    Raises:
        ValueError: Bad window or rank, mismatched shapes, fewer than 2 standards,
            or non-finite values.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    Xp = np.asarray(X_primary, dtype=np.float64)
    Xs = np.asarray(X_satellite, dtype=np.float64)
    if Xp.ndim != 2 or Xp.shape != Xs.shape:
        raise ValueError(
            f"X_primary and X_satellite must be 2-D with the same shape, got "
            f"{Xp.shape} and {Xs.shape}"
        )
    if not isinstance(window, (int, np.integer)) or window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window!r}")
    n, p = Xs.shape
    if n < 2:
        raise ValueError(f"Centred PDS needs at least 2 standards, got {n}")
    if not (np.isfinite(Xp).all() and np.isfinite(Xs).all()):
        raise ValueError("Standard spectra contain NaN or infinite values")
    if rank is not None and (not isinstance(rank, (int, np.integer)) or rank < 1):
        raise ValueError(f"rank must be a positive integer or None, got {rank!r}")
    window = int(window)
    rank_cap = min(n - 1, window) if rank is None else min(int(rank), n - 1, window)

    half = window // 2
    mean_s = Xs.mean(axis=0)
    mean_p = Xp.mean(axis=0)
    # Zero-padding the centred satellite data gives every wavelength a full-width
    # window; padded columns carry no variance, so they get no coefficient.
    padded = np.pad(Xs - mean_s, ((0, 0), (half, half)))
    windows = np.moveaxis(sliding_window_view(padded, window, axis=1), 1, 0)  # (p, n, w)
    U, s, Vt = np.linalg.svd(windows, full_matrices=False)
    k = s.shape[1]
    # Numerical-rank cut on the scale of the whole data set, not of each window: a
    # window with no satellite variation keeps only centring round-off, which a
    # per-window relative cut would invert into large coefficients.
    data_scale = max(float(s.max(initial=0.0)), float(np.sqrt(n) * np.abs(Xs).max()))
    tol = max(n, window) * np.finfo(np.float64).eps * data_scale
    keep = (s > tol) & (np.arange(k) < rank_cap)
    uty = np.einsum("pnk,np->pk", U, Xp - mean_p)
    scaled = np.where(keep, uty / np.where(keep, s, 1.0), 0.0)
    B = np.einsum("pk,pkw->pw", scaled, Vt)

    pos = np.arange(p)[:, None] - half + np.arange(window)[None, :]
    inside = (pos >= 0) & (pos < p)
    B[~inside] = 0.0
    mean_s_windows = np.where(inside, mean_s[np.clip(pos, 0, p - 1)], 0.0)
    offset = mean_p - np.einsum("pw,pw->p", B, mean_s_windows)

    return {
        "B_centred": B,
        "offset": offset,
        "window": window,
        "rank": int(rank_cap),
        "centred": True,
    }


def apply_pds_centred(X_satellite_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply centred PDS params from ``estimate_pds_lowrank``.

    Args:
        X_satellite_new: Satellite spectra, (m, p).
        params: Must hold ``B_centred`` and ``offset``.

    Returns:
        Spectra in the primary instrument's domain.
    """
    offset = np.asarray(params["offset"], dtype=np.float64)
    return apply_pds(np.asarray(X_satellite_new, dtype=np.float64), params["B_centred"]) + offset


def estimate_ds_dual(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    lam_rel: float = 1e-2,
    center: bool = True,
) -> Dict:
    """Estimate ridge Direct Standardization in dual (n × n) form.

    With the centred standards Xc (satellite) and Pc (primary), the ridge DS map
    ``A = (Xcᵀ Xc + λI)⁻¹ Xcᵀ Pc`` equals ``Xcᵀ (Xc Xcᵀ + λI)⁻¹ Pc``. Solving the
    n × n system instead of the p × p one takes milliseconds, and storing ``Xc``
    and ``W = (Xc Xcᵀ + λI)⁻¹ Pc`` (each n × p) replaces the p × p matrix.

    Args:
        X_primary: Primary spectra of the paired standards, (n, p).
        X_satellite: Satellite spectra of the same standards, same row order.
        lam_rel: Ridge strength relative to the mean squared norm of the centred
            satellite standards, ``λ = lam_rel · trace(Xc Xcᵀ) / n``, so it does not
            depend on the spectral scale. Must be > 0.
        center: Centre both instruments' standards and add the means back (an
            intercept). ``False`` reproduces uncentred ridge DS.

    Returns:
        Params for ``TransferModel(method='ds')``: ``ds_form='dual'``, ``basis``,
        ``W``, ``mean_satellite``, ``mean_primary``, ``lam``, ``lam_rel`` and
        ``centred``. Apply with ``apply_ds_dual`` or ``apply_transfer_dispatch``.

    Raises:
        ValueError: Mismatched shapes, ``lam_rel <= 0``, fewer than 2 standards,
            non-finite values, or (centred form) satellite standards with no
            variation.
    """
    Xp = np.asarray(X_primary, dtype=np.float64)
    Xs = np.asarray(X_satellite, dtype=np.float64)
    if Xp.ndim != 2 or Xp.shape != Xs.shape:
        raise ValueError(
            f"X_primary and X_satellite must be 2-D with the same shape, got "
            f"{Xp.shape} and {Xs.shape}"
        )
    if not (np.isfinite(lam_rel) and lam_rel > 0):
        raise ValueError(f"lam_rel must be a finite value > 0, got {lam_rel!r}")
    n, p = Xs.shape
    if n < 2:
        raise ValueError(f"DS needs at least 2 standards, got {n}")
    if not (np.isfinite(Xp).all() and np.isfinite(Xs).all()):
        raise ValueError("Standard spectra contain NaN or infinite values")
    mean_s = Xs.mean(axis=0) if center else np.zeros(p)
    mean_p = Xp.mean(axis=0) if center else np.zeros(p)
    Xc = Xs - mean_s
    Pc = Xp - mean_p
    K = Xc @ Xc.T
    scale = float(np.trace(K)) / n
    # Identical rows rarely centre to exact zeros; compare with the data's own size.
    raw_scale = float(np.mean(np.sum(Xs**2, axis=1)))
    if not scale > (np.finfo(np.float64).eps * max(n, p)) ** 2 * max(raw_scale, 1e-300):
        raise ValueError("The satellite standards do not vary; DS cannot be fitted")
    lam = float(lam_rel) * scale
    W = np.linalg.solve(K + lam * np.eye(n), Pc)
    return {
        "ds_form": "dual",
        "basis": Xc,
        "W": W,
        "mean_satellite": mean_s,
        "mean_primary": mean_p,
        "lam": lam,
        "lam_rel": float(lam_rel),
        "centred": bool(center),
    }


def apply_ds_dual(X_satellite_new: np.ndarray, params: Dict) -> np.ndarray:
    """Apply dual-form DS params from ``estimate_ds_dual``.

    Args:
        X_satellite_new: Satellite spectra, (m, p).
        params: From ``estimate_ds_dual``.

    Returns:
        Spectra in the primary instrument's domain.

    Raises:
        ValueError: The spectra do not have the model's wavelength count.
    """
    X = np.asarray(X_satellite_new, dtype=np.float64)
    basis = np.asarray(params["basis"], dtype=np.float64)
    if X.shape[1] != basis.shape[1]:
        raise ValueError(
            f"Spectra have {X.shape[1]} wavelengths but the DS model expects {basis.shape[1]}"
        )
    centred = X - np.asarray(params["mean_satellite"], dtype=np.float64)
    return np.asarray(params["mean_primary"], dtype=np.float64) + (centred @ basis.T) @ np.asarray(
        params["W"], dtype=np.float64
    )


def estimate_prediction_correction(
    y_ref: np.ndarray,
    y_pred_satellite: np.ndarray,
    fit_slope: bool = True,
    y_range_reference: tuple[float, float] | None = None,
) -> Dict:
    """Slope/bias correction of predictions, fitted on satellite standards.

    Fits ``y_ref = bias + slope · ŷ_satellite`` (or bias only, slope 1) on standards
    whose reference values are known and whose satellite spectra the model
    predicted (Bouveresse et al. 1996, *Anal Chem* 68(6):982-990). The result has the
    ``bias_correction`` format, so ``apply_prediction_correction`` and saved
    models apply it. It holds no fit metrics: a correction's fit to its own standards
    is not a validation (use ``transfer_evaluation.evaluate_transfer``).

    Args:
        y_ref: Reference values of the standards.
        y_pred_satellite: The model's predictions from the satellite spectra.
        fit_slope: Fit slope and bias (needs 3+ standards); ``False`` fits bias only.
        y_range_reference: (min, max) of the model's calibration y, for a warning
            when the standards span little of it.

    Returns:
        ``{'method': 'linear', 'bias', 'slope', 'prediction_scale': 'original',
        'source': 'satellite_standards', 'fit': 'slope_bias' | 'bias',
        'n_standards', 'warnings'}``.

    Raises:
        ValueError: Mismatched lengths, non-finite values, too few standards, or
            predictions with no spread when fitting a slope.
    """
    y = np.asarray(y_ref, dtype=np.float64).ravel()
    yhat = np.asarray(y_pred_satellite, dtype=np.float64).ravel()
    if y.shape != yhat.shape:
        raise ValueError(f"y_ref has {y.size} values and y_pred_satellite {yhat.size}")
    if not (np.isfinite(y).all() and np.isfinite(yhat).all()):
        raise ValueError("Reference values or predictions contain NaN or infinite values")
    n = int(y.size)
    need = 3 if fit_slope else 1
    if n < need:
        raise ValueError(
            f"A {'slope/bias' if fit_slope else 'bias'} correction needs at least {need} "
            f"standards, got {n}"
        )
    warnings_out: list[str] = []
    if fit_slope:
        if np.ptp(yhat) <= 0:
            raise ValueError("The satellite predictions do not vary; a slope cannot be fitted")
        slope, bias = (float(v) for v in np.polyfit(yhat, y, 1))
    else:
        slope, bias = 1.0, float(np.mean(y - yhat))
    if n < 5:
        warnings_out.append(f"Correction fitted on only {n} standards.")
    if y_range_reference is not None:
        lo, hi = (float(v) for v in y_range_reference)
        if fit_slope and hi > lo and np.ptp(y) < 0.3 * (hi - lo):
            warnings_out.append(
                "The standards span less than 30% of the calibration y range; the slope "
                "extrapolates outside it."
            )
    return {
        "method": "linear",
        "bias": bias,
        "slope": slope,
        "prediction_scale": "original",
        "source": "satellite_standards",
        "fit": "slope_bias" if fit_slope else "bias",
        "n_standards": n,
        "warnings": warnings_out,
    }


def apply_prediction_correction(y_pred: np.ndarray, correction: Dict) -> np.ndarray:
    """Apply a correction from ``estimate_prediction_correction`` to predictions.

    Args:
        y_pred: The model's predictions from (untransferred) satellite spectra.
        correction: The correction dict.

    Returns:
        Corrected predictions, ``bias + slope · y_pred``.
    """
    from .bias_correction import apply_correction

    return apply_correction(y_pred, correction)


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
        elif isinstance(value, (int, float, str, np.integer, np.floating, np.bool_)):
            # Store scalars (numpy ones as Python values, which json can write) and
            # short strings (e.g. 'standard_selection') in metadata
            # .item() keeps np.longdouble as longdouble, which json can't write
            if isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.generic):
                value = value.item()
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
# Per-wavelength slope/bias standardization (method key 'tsr')
# ==============================================================================

def estimate_tsr(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    transfer_indices: np.ndarray,
    slope_bias_correction: bool = True,
    regularization: float = 0.0,
) -> Dict:
    """
    Estimate a per-wavelength slope/bias standardization (method key ``'tsr'``).

    For each wavelength separately, regress the primary value on the satellite value
    over the chosen paired standards: ``X_primary[:, k] = slope[k] * X_satellite[:, k]
    + bias[k]``. Wavelengths are fitted independently; no information is shared
    between neighbouring channels. With ``slope_bias_correction=False`` only the
    bias is fitted (slope fixed at 1).

    This is slope/bias standardization in the spirit of Shenk & Westerhaus (1991);
    equivalence to their published procedure has not been checked. The key
    ``'tsr'`` is historical and is kept because saved transfer models store it: this
    is **not** trimmed scores regression (Folch-Fortuny et al. 2017), which is also
    abbreviated TSR.

    Rows of ``X_primary`` and ``X_satellite`` must be the same physical standards
    (paired). Two parameters are fitted per wavelength, so the fit is possible from
    few standards, but the per-wavelength R² it reports is computed on the fitting
    standards and is not a validation.

    Parameters
    ----------
    X_primary : np.ndarray, shape (n_samples, n_wavelengths)
        Primary instrument spectra on common wavelength grid.
    X_satellite : np.ndarray, shape (n_samples, n_wavelengths)
        Satellite instrument spectra on common wavelength grid.
        Must have same number of samples as X_primary.
    transfer_indices : np.ndarray, shape (n_transfer,)
        Indices of the paired standards to fit on (e.g. from
        ``select_transfer_standards``).
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
    .. [1] Shenk, J. S., & Westerhaus, M. O. (1991). New standardization and
           calibration procedures for NIRS analytical systems. Crop Science,
           31(6), 1694-1696. doi:10.2135/cropsci1991.0011183X003100060064x
           (cited as the slope/bias standardization idea; this code is not
           verified to reproduce their procedure).
    .. [2] Folch-Fortuny, A., et al. (2017). Calibration transfer between NIR
           spectrometers: new proposals and a comparative study. Journal of
           Chemometrics, doi:10.1002/cem.2874 (the *other* "TSR": trimmed scores
           regression, not implemented here).

    Notes
    -----
    - Assumes a linear relationship between instruments at each wavelength.
    - Wavelength shifts and bandwidth differences mix neighbouring channels and
      cannot be corrected by an independent per-channel fit; use PDS for those.
    - ``select_transfer_standards`` chooses the standards (all loaded pairs, or a
      Kennard-Stone subset).

    See Also
    --------
    apply_tsr : Apply the slope/bias correction to new spectra
    select_transfer_standards : Choose the standards to fit on
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
    Apply per-wavelength slope/bias standardization (key ``'tsr'``) to new spectra.

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
# Paired regression in satellite PCA space ("PC-DS", historical key 'ctai')
# ==============================================================================

def estimate_ctai(
    X_primary: np.ndarray,
    X_satellite: np.ndarray,
    n_components: int | None = None,
    explained_variance_threshold: float = 0.99,
) -> Dict:
    """
    Estimate a paired regression in the satellite PCA space ("PC-DS", key ``'ctai'``).

    Despite the historical key, this is **not** the published CTAI (calibration
    transfer based on affine invariance; Zhao et al. 2019, Molecules 24(9):1802),
    which is standard-free and corrects predictions of a master PLS model. This
    function needs the same standards measured on both instruments, row for row.

    Algorithm:
    1. Mean-centre the primary and satellite standards separately.
    2. Take the satellite principal components V (n_components of them, chosen by
       ``explained_variance_threshold`` when not given).
    3. Project **both** centred matrices onto V and regress the primary scores on
       the satellite scores (least squares), giving M_reduced.
    4. Map back: ``M = V @ M_reduced @ V.T`` and ``T = mean_primary - mean_satellite @ M``.

    Because the output is also projected onto the satellite PCA basis, any primary
    variation outside the satellite's leading components is lost; this differs from
    plain truncated-SVD DS, which leaves the output unrestricted.

    Parameters
    ----------
    X_primary : np.ndarray, shape (n_samples, n_wavelengths)
        Primary instrument spectra of the paired standards, on the common grid.
    X_satellite : np.ndarray, shape (n_samples, n_wavelengths)
        Satellite instrument spectra of the **same** standards, in the same row
        order and on the same grid (equal column counts are required).
    n_components : int, optional
        Number of satellite principal components to keep.
        If None, automatically selected based on explained_variance_threshold.
    explained_variance_threshold : float, default=0.99
        Fraction of satellite variance to retain when auto-selecting n_components.

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'M' : np.ndarray, shape (n_wavelengths, n_wavelengths)
            Transformation matrix
        - 'T' : np.ndarray, shape (n_wavelengths,)
            Translation vector
        - 'n_components' : int
            Number of components used
        - 'explained_variance' : float
            Fraction of satellite variance in the kept components
        - 'reconstruction_error' : float
            RMSE on the fitting standards (resubstitution; not a validation)
        - 'primary_mean' : np.ndarray
            Mean of primary spectra
        - 'satellite_mean' : np.ndarray
            Mean of satellite spectra

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.calibration_transfer import estimate_ctai, apply_ctai
    >>> rng = np.random.default_rng(0)
    >>> X_primary = rng.standard_normal((30, 200))          # 30 paired standards
    >>> X_satellite = 0.95 * X_primary + 0.05               # same standards, row for row
    >>> params = estimate_ctai(X_primary, X_satellite, n_components=10)
    >>> X_transferred = apply_ctai(X_satellite, params)

    Notes
    -----
    - Requires paired standards; raises if the row counts differ.
    - With few standards the fit can reproduce the standards closely and still do
      worse on new samples; check it on standards not used for fitting.

    See Also
    --------
    apply_ctai : Apply the transformation to new spectra
    estimate_ds : Direct Standardization
    """
    from scipy.linalg import svd

    n_samples_primary, n_wavelengths = X_primary.shape
    n_samples_satellite = X_satellite.shape[0]

    logger.debug("PC-DS input shapes: Primary %s, Satellite %s", X_primary.shape, X_satellite.shape)

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
        "PC-DS data validation passed; primary range [%.6f, %.6f], satellite range [%.6f, %.6f]",
        np.min(X_primary),
        np.max(X_primary),
        np.min(X_satellite),
        np.max(X_satellite),
    )

    # Step 1: Mean-center both datasets
    primary_mean = np.mean(X_primary, axis=0)
    satellite_mean = np.mean(X_satellite, axis=0)

    X_primary_centered = X_primary - primary_mean
    X_satellite_centered = X_satellite - satellite_mean

    logger.debug(
        "PC-DS mean centering complete; primary mean range [%.6f, %.6f], satellite mean range [%.6f, %.6f]",
        np.min(primary_mean),
        np.max(primary_mean),
        np.min(satellite_mean),
        np.max(satellite_mean),
    )

    # Step 2: Estimate affine transformation using PCA-regularized regression
    # We want: X_primary ≈ X_satellite @ M + T (in data space, not covariance space!)

    # Check if we have paired samples
    have_paired_samples = (n_samples_primary == n_samples_satellite)

    if not have_paired_samples:
        raise ValueError(
            f"PC-DS requires paired samples (same samples on both instruments).\n"
            f"Got {n_samples_primary} primary samples and {n_samples_satellite} satellite samples.\n"
            f"Every transfer method here needs paired standards."
        )

    print(f"  PC-DS: Detected {n_samples_primary} paired samples on both instruments")

    try:
        U_satellite, S_satellite, Vt_satellite = svd(X_satellite_centered, full_matrices=False)
        logger.debug(
            "PC-DS SVD: U%s S%s Vt%s; singular values [%.6e, %.6e]",
            U_satellite.shape,
            S_satellite.shape,
            Vt_satellite.shape,
            np.min(S_satellite),
            np.max(S_satellite),
        )
        if S_satellite[-1] > 0:
            logger.debug("PC-DS SVD condition number: %.2e", S_satellite[0] / S_satellite[-1])
    except np.linalg.LinAlgError as e:
        raise ValueError(f"SVD failed: {e}. Check if data has sufficient variance.")

    # Step 2b: Determine number of components (for regularization)
    if n_components is None:
        # Auto-select based on explained variance
        explained_var_cumsum = np.cumsum(S_satellite**2) / np.sum(S_satellite**2)
        n_components = np.searchsorted(explained_var_cumsum, explained_variance_threshold) + 1
        n_components = min(n_components, min(n_samples_satellite, n_wavelengths))
        print(
            f"  PC-DS: Auto-selected {n_components} components (threshold={explained_variance_threshold})"
        )
    else:
        n_components = min(n_components, len(S_satellite))
        print(f"  PC-DS: Using {n_components} components (user-specified)")

    # Step 2c: Project data onto principal components
    # This is the key: work in reduced-rank space for numerical stability
    V_truncated = Vt_satellite[:n_components, :].T  # Shape: (n_wavelengths, n_components)
    S_truncated = S_satellite[:n_components]

    # Project satellite and primary data onto satellite's principal components
    X_satellite_projected = X_satellite_centered @ V_truncated  # (n_samples, n_components)
    X_primary_projected = X_primary_centered @ V_truncated  # (n_samples, n_components)

    M_reduced = np.linalg.lstsq(X_satellite_projected, X_primary_projected, rcond=None)[0]
    logger.debug("PC-DS M_reduced shape: %s (PCA space transformation)", M_reduced.shape)

    M = V_truncated @ M_reduced @ V_truncated.T

    logger.debug(
        "PC-DS M shape %s; M range [%.6f, %.6f]; M diagonal mean %.6f",
        M.shape,
        np.min(M),
        np.max(M),
        np.mean(np.diag(M)),
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
        "PC-DS translation T range [%.6f, %.6f]; T mean %.6f",
        np.min(T),
        np.max(T),
        np.mean(T),
    )

    X_satellite_transformed = X_satellite @ M + T

    # Check for NaN/inf in transformed data
    if np.any(np.isnan(X_satellite_transformed)):
        raise ValueError(f"Transformed data contains NaN values! Transformation failed.")
    if np.any(np.isinf(X_satellite_transformed)):
        raise ValueError(f"Transformed data contains infinite values! Transformation failed.")

    logger.debug(
        "PC-DS transformed data range [%.6f, %.6f]",
        np.min(X_satellite_transformed),
        np.max(X_satellite_transformed),
    )

    X_primary_sample = X_primary[:min(n_samples_primary, n_samples_satellite)]
    X_satellite_sample = X_satellite_transformed[:min(n_samples_primary, n_samples_satellite)]

    reconstruction_error = np.sqrt(np.mean((X_primary_sample - X_satellite_sample) ** 2))

    explained_variance = np.sum(S_truncated**2) / np.sum(S_satellite**2) if len(S_satellite) > 0 else 1.0

    print(
        f"  PC-DS: components={n_components}, "
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
    Apply a PC-DS transfer (key ``'ctai'``) to new satellite instrument spectra.

    Transforms satellite spectra to the primary instrument domain with the
    affine map ``X @ M + T`` estimated by ``estimate_ctai``.

    Parameters
    ----------
    X_satellite_new : np.ndarray, shape (n_samples, n_wavelengths)
        New satellite instrument spectra to transform.
    params : dict
        Parameters from estimate_ctai, containing 'M' and 'T'.

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        Transformed spectra in primary instrument domain.

    Examples
    --------
    >>> # After estimate_ctai (see its example)
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
# Iterative ridge DS (historical key 'nspfce'; a dasp heuristic, not PFCE)
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
    Iterative ridge DS (historical key ``'nspfce'``): a dasp heuristic.

    This is **not** PFCE or NS-PFCE (parameter-free calibration enhancement;
    Zhang et al. 2021, Anal. Chim. Acta 1142:169-178), which constrains the slave
    model's regression coefficients to correlate with the master model's. No
    published reference exists for this
    function; treat it as an in-house variant of Direct Standardization.

    Algorithm (as implemented):
    1. Optionally select wavelengths (CARS, SPA or VCPA-IRIV run against the
       spectral mean as a pseudo-target).
    2. Initialise ``T = diag(std_primary / std_satellite)`` and a matching offset.
    3. Repeat: solve the ridge problem (lambda = 1e-6)
       ``T_new = argmin ||(X_primary - offset) - X_satellite @ T||``, damp the
       update (``T = 0.5 * T_new + 0.5 * T``), and reset the offset to the mean
       residual. Stop when the change in mean squared error falls below
       ``convergence_threshold`` or after ``max_iterations``.

    Rows of ``X_primary`` and ``X_satellite`` must be the same physical standards
    (paired); the least-squares step uses them row for row. With few standards the
    fitted full matrix can reproduce the standards almost exactly and still do
    worse than no correction on new samples.

    Parameters
    ----------
    X_primary : np.ndarray, shape (n_samples_primary, n_wavelengths)
        Primary instrument spectra on common wavelength grid.
    X_satellite : np.ndarray, shape (n_samples, n_wavelengths)
        Satellite instrument spectra of the **same** standards, in the same row
        order, on the common wavelength grid.
    wavelengths : np.ndarray, shape (n_wavelengths,)
        Wavelength grid (used for wavelength selection).
    use_wavelength_selection : bool, default=False
        Whether to apply wavelength selection before transformation. When True the
        output contains only the selected wavelengths.
    wavelength_selector : str, default='cars'
        Method for wavelength selection: 'vcpa-iriv', 'cars', or 'spa'.
    max_iterations : int, default=100
        Maximum iterations for iterative optimization.
    convergence_threshold : float, default=1e-6
        Convergence criterion (change in transformation matrix).
    normalize : bool, default=False
        Rescale T to a fixed Frobenius norm every 10 iterations.

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
    >>> rng = np.random.default_rng(0)
    >>> wavelengths = np.linspace(1000, 2500, 200)
    >>> X_primary = rng.standard_normal((40, 200))           # 40 paired standards
    >>> X_satellite = 0.9 * X_primary + 0.1                   # same standards, row for row
    >>> params = estimate_nspfce(X_primary, X_satellite, wavelengths)
    >>> X_transferred = apply_nspfce(X_satellite, params)

    References
    ----------
    None. This is a dasp heuristic (see above); it is not the published PFCE.

    See Also
    --------
    apply_nspfce : Apply the transformation to new spectra
    estimate_ds : Direct Standardization
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
        print(
            f"  Iterative ridge DS: Performing wavelength selection using {wavelength_selector}..."
        )

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
            print(
                f"  Iterative ridge DS: Selected {len(selected_wavelengths)}/{n_wavelengths} wavelengths"
            )

            # Reduce matrices to selected wavelengths
            X_primary_sel = X_primary[:, selected_wavelengths]
            X_satellite_sel = X_satellite[:, selected_wavelengths]
            n_selected = len(selected_wavelengths)

        except Exception as e:
            print(
                f"  Iterative ridge DS: Wavelength selection failed ({str(e)}), using all wavelengths"
            )
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

    print(f"  Iterative ridge DS: Converged in {convergence_iterations} iterations")
    print(f"  Iterative ridge DS: Final RMSE: {final_objective:.6f}")

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
    Apply an iterative ridge DS transfer (key ``'nspfce'``) to new satellite spectra.

    Transforms satellite spectra to primary instrument domain using previously
    estimated transformation.

    Parameters
    ----------
    X_satellite_new : np.ndarray, shape (n_samples, n_wavelengths)
        New satellite instrument spectra to transform.
    params : dict
        Parameters from estimate_nspfce.
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
    >>> # After estimate_nspfce (see its example)
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
    max_components: int = 20,
) -> Dict:
    """
    Estimate an experimental joint-Y PLS score mapping ("JYPLS-inv").

    The paired standards measured on both instruments are stacked into one X block
    that shares the reference values y, and one PLS model is fitted to it. Each
    spectrum is projected to PLS scores with the model's own centring and
    ``x_rotations_`` (so undeflated X is projected correctly). An affine map with an
    intercept is fitted from the satellite scores to the primary scores of the same
    standards, and a satellite spectrum is transferred by mapping its scores and
    reconstructing with the X loadings around the primary standards' mean:

        X_transferred = mean_primary + (c + T_sat @ M - mean(T_primary)) @ P.T

    which is applied as the affine map ``X_sat @ B + offset``.

    The output lies in the primary mean plus the span of the PLS loadings, so
    spectral variation outside the k components is not carried over. This is a
    score-mapping implementation in the spirit of joint-Y PLS inversion; it has not
    been checked against a published JYPLS-inv algorithm (which fits separate
    loadings per instrument block). It is disabled in the GUI.

    Parameters
    ----------
    X_primary : np.ndarray, shape (n_samples, n_wavelengths)
        Primary instrument spectra.
    X_satellite : np.ndarray, shape (n_samples, n_wavelengths)
        Satellite instrument spectra of the same samples, row for row.
    y_transfer : np.ndarray, shape (n_transfer,)
        Measured reference values for the transfer standards. Every value must be
        finite; missing targets are rejected, never substituted.
    transfer_indices : np.ndarray, shape (n_transfer,)
        Rows of X_primary / X_satellite that are the transfer standards.
    n_components : int | None, optional
        Number of PLS components. If None, chosen by cross-validated y error with
        both spectra of a standard kept in the same fold.
    cv_folds : int, optional
        Number of cross-validation folds for component selection. Default 5.
    max_components : int, optional
        Maximum number of components to try in CV. Default 20.

    Returns
    -------
    params : dict
        - 'transformation_matrix' : (n_wavelengths, n_wavelengths) B
        - 'offset' : (n_wavelengths,) so that X_transferred = X_sat @ B + offset
        - 'n_components', 'cv_rmse' (RMSECV of y for the joint PLS when
          auto-selected, else 0.0), 'transfer_indices', 'explained_variance_ratio'
        - 'x_mean', 'pls_x_rotations', 'pls_x_weights', 'pls_x_loadings'
        - 'pls_scores_primary', 'pls_scores_satellite'
        - 'score_intercept', 'score_transformation' (c and M above)
        - 'primary_mean' (mean primary spectrum of the standards)

    Raises
    ------
    ValueError
        On shape mismatches, fewer than 2 standards, or non-finite y values.
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
    except ImportError:
        raise ImportError(
            "scikit-learn is required for JYPLS-inv. Install with: pip install scikit-learn"
        )

    if X_primary.shape != X_satellite.shape:
        raise ValueError(
            f"X_primary and X_satellite must have same shape. "
            f"Got {X_primary.shape} and {X_satellite.shape}"
        )

    transfer_indices = np.asarray(transfer_indices)
    y_transfer = np.asarray(y_transfer, dtype=float).ravel()

    if len(transfer_indices) != len(y_transfer):
        raise ValueError(
            f"Number of transfer_indices ({len(transfer_indices)}) must match "
            f"y_transfer length ({len(y_transfer)})"
        )

    if len(transfer_indices) < 2:
        raise ValueError("Need at least 2 transfer samples for JYPLS-inv")

    if not np.all(np.isfinite(y_transfer)):
        n_bad = int(np.sum(~np.isfinite(y_transfer)))
        raise ValueError(
            f"JYPLS-inv needs a measured reference value for every transfer standard; "
            f"{n_bad} of {len(y_transfer)} are missing or non-finite"
        )

    n_wavelengths = X_primary.shape[1]
    n_transfer = len(transfer_indices)

    X_primary_transfer = X_primary[transfer_indices]
    X_satellite_transfer = X_satellite[transfer_indices]

    # Stack both instruments' spectra of the same standards; they share y.
    X_aug = np.vstack([X_primary_transfer, X_satellite_transfer])
    Y_aug = np.concatenate([y_transfer, y_transfer]).reshape(-1, 1)
    # Both spectra of one standard carry the same y, so they must share a fold.
    groups = np.tile(np.arange(n_transfer), 2)

    cv_rmse = None
    if n_components is None:
        n_splits = min(cv_folds, n_transfer)
        # Materialise the grouped splits so no `groups=` keyword is passed to
        # cross_val_score: under sklearn metadata routing that keyword raises.
        splits = list(GroupKFold(n_splits=n_splits).split(X_aug, Y_aug, groups))
        smallest_train = min(len(train) for train, _ in splits)
        max_comp = min(max_components, n_transfer - 1, n_wavelengths, smallest_train - 1)

        rmse_by_n: dict[int, float] = {}
        for n in range(1, max_comp + 1):
            scores = cross_val_score(
                PLSRegression(n_components=n, scale=False),
                X_aug,
                Y_aug,
                cv=splits,
                scoring="neg_root_mean_squared_error",
                error_score="raise",
            )
            avg_rmse = float(-np.mean(scores))
            if np.isfinite(avg_rmse):
                rmse_by_n[n] = avg_rmse

        if not rmse_by_n:
            raise ValueError(
                "JYPLS-inv could not choose the number of PLS components: cross-validation "
                f"gave no finite error for 1..{max_comp} components. Pass n_components."
            )
        n_components = min(rmse_by_n, key=rmse_by_n.get)
        cv_rmse = rmse_by_n[n_components]

    max_comp = min(n_transfer - 1, n_wavelengths)
    n_components = max(min(int(n_components), max_comp), 1)

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_aug, Y_aug)

    # scale=False, so the model's preprocessing is centring on the stacked mean.
    x_mean = X_aug.mean(axis=0)
    R = pls.x_rotations_  # projects undeflated, centred X to scores
    P = pls.x_loadings_
    W = pls.x_weights_

    T_primary = (X_primary_transfer - x_mean) @ R
    T_satellite = (X_satellite_transfer - x_mean) @ R

    # Affine score map with intercept: T_primary ~ c + T_satellite @ M
    design = np.hstack([np.ones((n_transfer, 1)), T_satellite])
    coef = np.linalg.lstsq(design, T_primary, rcond=None)[0]
    score_intercept = coef[0]
    M_scores = coef[1:]

    primary_mean = X_primary_transfer.mean(axis=0)
    primary_score_mean = T_primary.mean(axis=0)

    transformation_matrix = R @ M_scores @ P.T
    offset = primary_mean + (score_intercept - primary_score_mean - x_mean @ R @ M_scores) @ P.T

    X_reconstructed = (X_aug - x_mean) @ R @ P.T + x_mean
    X_var = np.var(X_aug, axis=0).sum()
    X_residual_var = np.var(X_aug - X_reconstructed, axis=0).sum()
    explained_variance_ratio = 1.0 - (X_residual_var / X_var) if X_var > 0 else 1.0

    return {
        "transformation_matrix": transformation_matrix,
        "offset": offset,
        "n_components": n_components,
        "cv_rmse": cv_rmse if cv_rmse is not None else 0.0,
        "transfer_indices": transfer_indices,
        "x_mean": x_mean,
        "pls_x_rotations": R,
        "pls_x_weights": W,
        "pls_x_loadings": P,
        "pls_scores_primary": T_primary,
        "pls_scores_satellite": T_satellite,
        "score_intercept": score_intercept,
        "score_transformation": M_scores,
        "primary_mean": primary_mean,
        "explained_variance_ratio": explained_variance_ratio,
        "n_transfer_samples": n_transfer,
    }


def apply_jypls_inv(X_satellite_new: np.ndarray, params: Dict) -> np.ndarray:
    """
    Apply a JYPLS-inv score mapping to new satellite spectra.

    Parameters
    ----------
    X_satellite_new : np.ndarray, shape (n_samples, n_wavelengths)
        New satellite spectra to transfer to the primary domain.
    params : dict
        Parameters from estimate_jypls_inv().

    Returns
    -------
    X_transferred : np.ndarray, shape (n_samples, n_wavelengths)
        ``X_satellite_new @ B + offset``.

    Raises
    ------
    ValueError
        If the wavelength count does not match, or if ``params`` come from the
        pre-fix implementation (no ``'offset'``), whose output dropped the PLS
        centring and mean spectrum and is not a primary-domain spectrum.
    """
    B = params["transformation_matrix"]

    if "offset" not in params:
        raise ValueError(
            "This JYPLS-inv transfer model was built by an older dasp version whose "
            "transfer maths dropped the PLS centring and mean spectrum, so its output "
            "is not a primary-instrument spectrum. Rebuild the transfer model."
        )

    if X_satellite_new.shape[1] != B.shape[0]:
        raise ValueError(
            f"X_satellite_new has {X_satellite_new.shape[1]} wavelengths, "
            f"but transformation matrix expects {B.shape[0]}"
        )

    return X_satellite_new @ B + np.asarray(params["offset"])


def apply_transfer_dispatch(X_satellite: np.ndarray, transfer_model: TransferModel) -> np.ndarray:
    """
    Unified dispatcher for applying any transfer model type.

    This function provides a single interface for applying calibration transfer
    models regardless of the method key stored in the model.

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
        if params.get('ds_form') == 'dual':
            return apply_ds_dual(X_satellite, params)
        return apply_ds(X_satellite, params['A'])
    elif method == 'pds':
        if 'B_centred' in params:
            return apply_pds_centred(X_satellite, params)
        return apply_pds(X_satellite, params['B'], params.get('window'))
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
    for _key in ("ds", "pds", "tsr", "ctai", "nspfce", "jypls-inv"):
        print(f"  - {_key}: {method_display_name(_key)}")
    print("=" * 60)

    # Quick test of new methods
    import numpy as np
    np.random.seed(42)

    print("\nTesting slope/bias per wavelength (key tsr):")
    X_primary = np.random.randn(50, 100)
    X_satellite = 0.95 * X_primary + 0.05
    transfer_idx = np.array([0, 10, 20, 30, 40])  # Simple selection

    tsr_params = estimate_tsr(X_primary, X_satellite, transfer_idx)
    print(f"  Mean R²: {tsr_params['mean_r_squared']:.4f}")
    print(f"  Slope range: [{tsr_params['slope'].min():.3f}, {tsr_params['slope'].max():.3f}]")

    X_transferred_tsr = apply_tsr(X_satellite, tsr_params)
    print(f"  Transfer RMSE: {np.sqrt(np.mean((X_transferred_tsr - X_primary)**2)):.6f}")

    print("\nTesting PC-DS (key ctai):")
    ctai_params = estimate_ctai(X_primary, X_satellite)
    print(f"  Explained variance: {ctai_params['explained_variance']:.4f}")
    print(f"  N components: {ctai_params['n_components']}")
    print(f"  Reconstruction error: {ctai_params['reconstruction_error']:.6f}")

    X_transferred_ctai = apply_ctai(X_satellite, ctai_params)
    print(f"  Transfer RMSE: {np.sqrt(np.mean((X_transferred_ctai - X_primary)**2)):.6f}")

    print("\n" + "=" * 60)
    print("All methods loaded successfully!")
