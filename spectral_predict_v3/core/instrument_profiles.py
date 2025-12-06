"""
spectral_predict_v3.core.instrument_profiles
============================================

Instrument characterization and profile management for calibration transfer.

This module provides tools for:
- Creating instrument profiles from spectral data
- Computing data-driven resolution metrics
- Comparing instruments
- Saving/loading instrument profiles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
from scipy import signal


@dataclass
class InstrumentProfile:
    """
    Representation of a spectroscopy instrument with data-driven
    characterization of its wavelength grid and effective resolution.
    """
    instrument_id: str
    vendor: Optional[str] = None
    model: Optional[str] = None
    description: Optional[str] = None

    wavelengths: Optional[np.ndarray] = None

    delta_lambda_med: Optional[float] = None
    roughness_R: Optional[float] = None
    detail_score: Optional[float] = None

    peak_count: Optional[int] = None
    avg_peak_fwhm: Optional[float] = None
    avg_peak_sharpness: Optional[float] = None

    is_interpolated: Optional[bool] = None

    extra: Dict = field(default_factory=dict)


def compute_wavelength_spacing(wavelengths: np.ndarray) -> Dict:
    """
    Compute basic spacing statistics for a wavelength grid.

    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of monotonically increasing wavelength values.

    Returns
    -------
    Dict
        Dictionary with delta_lambda_med, delta_lambda_min, delta_lambda_max.
    """
    deltas = np.diff(wavelengths)
    return {
        "delta_lambda_med": float(np.median(deltas)),
        "delta_lambda_min": float(np.min(deltas)),
        "delta_lambda_max": float(np.max(deltas)),
    }


def compute_roughness(X: np.ndarray, wavelengths: np.ndarray) -> float:
    """
    Compute a scalar spectral roughness metric from spectra and wavelengths.

    Uses second derivatives along the wavelength axis.

    Parameters
    ----------
    X : np.ndarray
        Spectra array of shape (n_samples, n_wavelengths).
    wavelengths : np.ndarray
        1D wavelength array aligned with columns of X.

    Returns
    -------
    float
        Scalar roughness metric (higher = more high-frequency detail).
    """
    second_deriv = np.diff(X, n=2, axis=1)
    roughness = float(np.sqrt(np.mean(second_deriv ** 2)))

    return roughness


def detect_interpolation(wavelengths: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Detect if wavelength grid appears to be uniformly interpolated.

    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of wavelength values.
    tolerance : float
        Maximum relative variation in spacing to consider uniform.

    Returns
    -------
    bool
        True if wavelengths appear uniformly interpolated.
    """
    if len(wavelengths) < 3:
        return False

    deltas = np.diff(wavelengths)

    mean_delta = np.mean(deltas)
    if mean_delta == 0:
        return False

    relative_variation = np.std(deltas) / mean_delta

    return relative_variation < tolerance


def compute_peak_fwhm(spectrum: np.ndarray, wavelengths: np.ndarray,
                      peak_idx: int) -> Tuple[float, float]:
    """
    Compute FWHM (Full Width at Half Maximum) for a single peak.

    Parameters
    ----------
    spectrum : np.ndarray
        1D spectrum array.
    wavelengths : np.ndarray
        1D wavelength array.
    peak_idx : int
        Index of the peak.

    Returns
    -------
    Tuple[float, float]
        (FWHM in nm, peak sharpness = height/FWHM)
    """
    peak_height = spectrum[peak_idx]
    half_max = peak_height / 2.0

    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_max:
        left_idx -= 1

    if left_idx < peak_idx and spectrum[left_idx] < half_max < spectrum[left_idx + 1]:
        frac = (half_max - spectrum[left_idx]) / (spectrum[left_idx + 1] - spectrum[left_idx])
        left_wavelength = wavelengths[left_idx] + frac * (wavelengths[left_idx + 1] - wavelengths[left_idx])
    else:
        left_wavelength = wavelengths[left_idx]

    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
        right_idx += 1

    if right_idx > peak_idx and spectrum[right_idx] < half_max < spectrum[right_idx - 1]:
        frac = (half_max - spectrum[right_idx]) / (spectrum[right_idx - 1] - spectrum[right_idx])
        right_wavelength = wavelengths[right_idx] + frac * (wavelengths[right_idx - 1] - wavelengths[right_idx])
    else:
        right_wavelength = wavelengths[right_idx]

    fwhm = abs(right_wavelength - left_wavelength)
    sharpness = peak_height / fwhm if fwhm > 0 else 0.0

    return fwhm, sharpness


def analyze_peaks(X: np.ndarray, wavelengths: np.ndarray) -> Dict:
    """
    Detect peaks and compute peak-based resolution metrics.

    Parameters
    ----------
    X : np.ndarray
        Spectra array of shape (n_samples, n_wavelengths).
    wavelengths : np.ndarray
        1D wavelength array.

    Returns
    -------
    Dict
        Dictionary with peak_count, avg_peak_fwhm, avg_peak_sharpness.
    """
    all_fwhm = []
    all_sharpness = []
    all_peak_counts = []

    for i in range(X.shape[0]):
        spectrum = X[i, :]

        spec_min = np.min(spectrum)
        spec_max = np.max(spectrum)
        if spec_max > spec_min:
            spectrum_norm = (spectrum - spec_min) / (spec_max - spec_min)
        else:
            continue

        peaks, properties = signal.find_peaks(
            spectrum_norm,
            prominence=0.05,
            distance=5,
            width=2
        )

        all_peak_counts.append(len(peaks))

        for peak_idx in peaks:
            try:
                fwhm, sharpness = compute_peak_fwhm(spectrum_norm, wavelengths, peak_idx)
                if fwhm > 0:
                    all_fwhm.append(fwhm)
                    all_sharpness.append(sharpness)
            except Exception:
                continue

    avg_peak_count = int(np.mean(all_peak_counts)) if all_peak_counts else 0
    avg_fwhm = float(np.mean(all_fwhm)) if all_fwhm else 0.0
    avg_sharpness = float(np.mean(all_sharpness)) if all_sharpness else 0.0

    return {
        "peak_count": avg_peak_count,
        "avg_peak_fwhm": avg_fwhm,
        "avg_peak_sharpness": avg_sharpness,
    }


def characterize_instrument(
    instrument_id: str,
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    vendor: Optional[str] = None,
    model: Optional[str] = None,
    description: Optional[str] = None,
) -> InstrumentProfile:
    """
    Build an InstrumentProfile and populate its data-driven metrics.

    Returns
    -------
    InstrumentProfile
    """
    spacing_stats = compute_wavelength_spacing(wavelengths)
    delta_lambda_med = spacing_stats["delta_lambda_med"]

    roughness_R = compute_roughness(spectra, wavelengths)

    detail_score = roughness_R / delta_lambda_med if delta_lambda_med > 0 else 0.0

    is_interpolated = detect_interpolation(wavelengths)

    peak_stats = analyze_peaks(spectra, wavelengths)

    return InstrumentProfile(
        instrument_id=instrument_id,
        vendor=vendor,
        model=model,
        description=description,
        wavelengths=wavelengths.copy(),
        delta_lambda_med=delta_lambda_med,
        roughness_R=roughness_R,
        detail_score=detail_score,
        peak_count=peak_stats["peak_count"],
        avg_peak_fwhm=peak_stats["avg_peak_fwhm"],
        avg_peak_sharpness=peak_stats["avg_peak_sharpness"],
        is_interpolated=is_interpolated,
        extra={},
    )


def save_instrument_profiles(
    profiles: Dict[str, InstrumentProfile],
    path: Path | str,
) -> None:
    """
    Serialize instrument profiles to a JSON file.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_python_type(val):
        """Convert numpy types to Python native types for JSON serialization."""
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, np.bool_):
            return bool(val)
        return val

    data = {}
    for inst_id, profile in profiles.items():
        data[inst_id] = {
            "instrument_id": profile.instrument_id,
            "vendor": profile.vendor,
            "model": profile.model,
            "description": profile.description,
            "wavelengths": profile.wavelengths.tolist() if profile.wavelengths is not None else None,
            "delta_lambda_med": to_python_type(profile.delta_lambda_med),
            "roughness_R": to_python_type(profile.roughness_R),
            "detail_score": to_python_type(profile.detail_score),
            "peak_count": to_python_type(profile.peak_count),
            "avg_peak_fwhm": to_python_type(profile.avg_peak_fwhm),
            "avg_peak_sharpness": to_python_type(profile.avg_peak_sharpness),
            "is_interpolated": to_python_type(profile.is_interpolated),
            "extra": profile.extra,
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_instrument_profiles(path: Path | str) -> Dict[str, InstrumentProfile]:
    """
    Load instrument profiles from a JSON file.
    """
    import json

    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    profiles = {}
    for inst_id, profile_data in data.items():
        profiles[inst_id] = InstrumentProfile(
            instrument_id=profile_data["instrument_id"],
            vendor=profile_data.get("vendor"),
            model=profile_data.get("model"),
            description=profile_data.get("description"),
            wavelengths=np.array(profile_data["wavelengths"]) if profile_data.get("wavelengths") is not None else None,
            delta_lambda_med=profile_data.get("delta_lambda_med"),
            roughness_R=profile_data.get("roughness_R"),
            detail_score=profile_data.get("detail_score"),
            peak_count=profile_data.get("peak_count"),
            avg_peak_fwhm=profile_data.get("avg_peak_fwhm"),
            avg_peak_sharpness=profile_data.get("avg_peak_sharpness"),
            is_interpolated=profile_data.get("is_interpolated"),
            extra=profile_data.get("extra", {}),
        )

    return profiles


def rank_instruments_by_detail(
    profiles: Dict[str, InstrumentProfile]
) -> List[str]:
    """
    Return instrument_ids sorted by descending detail_score.
    """
    sorted_items = sorted(
        profiles.items(),
        key=lambda x: x[1].detail_score if x[1].detail_score is not None else 0.0,
        reverse=True
    )
    return [inst_id for inst_id, _ in sorted_items]


def estimate_smoothing_between_instruments(
    wavelengths_high: np.ndarray,
    X_high: np.ndarray,
    wavelengths_low: np.ndarray,
    X_low: np.ndarray,
    sigma_candidates: List[float],
) -> float:
    """
    Estimate optimal Gaussian smoothing width to map high-resolution
    instrument to low-resolution instrument.

    Parameters
    ----------
    wavelengths_high : np.ndarray
        High-resolution wavelengths.
    X_high : np.ndarray
        High-resolution spectra.
    wavelengths_low : np.ndarray
        Low-resolution wavelengths.
    X_low : np.ndarray
        Low-resolution spectra.
    sigma_candidates : List[float]
        List of sigma values to try.

    Returns
    -------
    float
        Optimal sigma value.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d

    best_sigma = sigma_candidates[0]
    best_mse = float('inf')

    for sigma in sigma_candidates:
        if sigma > 0:
            X_high_smoothed = gaussian_filter1d(X_high, sigma=sigma, axis=1)
        else:
            X_high_smoothed = X_high.copy()

        X_high_resampled = np.zeros((X_high_smoothed.shape[0], wavelengths_low.shape[0]))
        for i in range(X_high_smoothed.shape[0]):
            interpolator = interp1d(wavelengths_high, X_high_smoothed[i, :],
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            X_high_resampled[i, :] = interpolator(wavelengths_low)

        mse = np.mean((X_high_resampled - X_low) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_sigma = sigma

    return best_sigma
