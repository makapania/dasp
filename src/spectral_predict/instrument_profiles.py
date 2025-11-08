"""
spectral_predict.instrument_profiles
====================================

Backend-only module for instrument characterization and registry management.
This file is a skeleton: function bodies are stubs (`pass`) for another agent
to fill in with real implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np


@dataclass
class InstrumentProfile:
    """
    Representation of a spectroscopy instrument with basic, data-driven
    characterization of its wavelength grid and effective resolution.
    """
    instrument_id: str
    vendor: Optional[str] = None
    model: Optional[str] = None
    description: Optional[str] = None

    wavelengths: Optional[np.ndarray] = None  # canonical wavelength grid

    # Data-driven metrics
    delta_lambda_med: Optional[float] = None  # median wavelength spacing
    roughness_R: Optional[float] = None       # spectral roughness metric
    detail_score: Optional[float] = None      # e.g., roughness_R / delta_lambda_med

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
        Dictionary with at least keys:
        - "delta_lambda_med": median spacing
        - "delta_lambda_min": minimum spacing
        - "delta_lambda_max": maximum spacing
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

    Typically this uses second derivatives along the wavelength axis.

    Parameters
    ----------
    X : np.ndarray
        Spectra array of shape (n_samples, n_wavelengths).
    wavelengths : np.ndarray
        1D wavelength array aligned with columns of X.

    Returns
    -------
    float
        Scalar roughness metric (higher ~ more high-frequency detail).
    """
    # Compute second derivative along wavelength axis for each spectrum
    # Using central differences: d2f/dx2 ≈ (f(i+1) - 2*f(i) + f(i-1)) / dx^2
    second_deriv = np.diff(X, n=2, axis=1)

    # Use RMS of second derivatives as roughness metric
    roughness = float(np.sqrt(np.mean(second_deriv ** 2)))

    return roughness


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

    This function should:
    - Call compute_wavelength_spacing
    - Call compute_roughness
    - Derive detail_score (e.g., roughness / median spacing)

    Returns
    -------
    InstrumentProfile
    """
    # Compute spacing statistics
    spacing_stats = compute_wavelength_spacing(wavelengths)
    delta_lambda_med = spacing_stats["delta_lambda_med"]

    # Compute roughness metric
    roughness_R = compute_roughness(spectra, wavelengths)

    # Compute detail score (roughness normalized by spacing)
    # Higher detail_score indicates higher effective resolution
    detail_score = roughness_R / delta_lambda_med if delta_lambda_med > 0 else 0.0

    return InstrumentProfile(
        instrument_id=instrument_id,
        vendor=vendor,
        model=model,
        description=description,
        wavelengths=wavelengths.copy(),
        delta_lambda_med=delta_lambda_med,
        roughness_R=roughness_R,
        detail_score=detail_score,
        extra={},
    )


def save_instrument_profiles(
    profiles: Dict[str, InstrumentProfile],
    path: Path | str,
) -> None:
    """
    Serialize instrument profiles (including wavelength arrays and metrics)
    to a JSON file.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert profiles to JSON-serializable format
    data = {}
    for inst_id, profile in profiles.items():
        data[inst_id] = {
            "instrument_id": profile.instrument_id,
            "vendor": profile.vendor,
            "model": profile.model,
            "description": profile.description,
            "wavelengths": profile.wavelengths.tolist() if profile.wavelengths is not None else None,
            "delta_lambda_med": profile.delta_lambda_med,
            "roughness_R": profile.roughness_R,
            "detail_score": profile.detail_score,
            "extra": profile.extra,
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_instrument_profiles(path: Path | str) -> Dict[str, InstrumentProfile]:
    """
    Load instrument profiles from a JSON file previously produced by
    save_instrument_profiles.
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
            extra=profile_data.get("extra", {}),
        )

    return profiles


def rank_instruments_by_detail(
    profiles: Dict[str, InstrumentProfile]
) -> List[str]:
    """
    Return instrument_ids sorted by descending detail_score.
    """
    # Sort by detail_score in descending order
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
    Given paired spectra for the same samples on a high-detail instrument
    (wavelengths_high, X_high) and a low-detail instrument
    (wavelengths_low, X_low), search over sigma_candidates (e.g. Gaussian
    smoothing widths) and return the sigma that best maps high to low,
    according to a suitable error metric (e.g. mean squared error).
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d

    best_sigma = sigma_candidates[0]
    best_mse = float('inf')

    for sigma in sigma_candidates:
        # Smooth high-res spectra if sigma > 0
        if sigma > 0:
            X_high_smoothed = gaussian_filter1d(X_high, sigma=sigma, axis=1)
        else:
            X_high_smoothed = X_high.copy()

        # Resample smoothed high-res spectra to low-res wavelength grid
        X_high_resampled = np.zeros((X_high_smoothed.shape[0], wavelengths_low.shape[0]))
        for i in range(X_high_smoothed.shape[0]):
            interpolator = interp1d(wavelengths_high, X_high_smoothed[i, :],
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            X_high_resampled[i, :] = interpolator(wavelengths_low)

        # Compute MSE between resampled high-res and original low-res
        mse = np.mean((X_high_resampled - X_low) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_sigma = sigma

    return best_sigma


if __name__ == "__main__":
    # Optional: simple self-test or placeholder for future examples.
    print("instrument_profiles skeleton loaded.")
