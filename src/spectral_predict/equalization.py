"""
spectral_predict.equalization
=============================

Backend-only module for multi-instrument spectral equalization to a
common domain. This sits on top of instrument_profiles and
calibration_transfer.

This file is a skeleton: function bodies are stubs (`pass`) for another
agent to fill in with real implementations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .instrument_profiles import InstrumentProfile
from .calibration_transfer import TransferModel


def choose_common_grid(
    profiles: Dict[str, InstrumentProfile],
    instrument_ids: List[str],
) -> np.ndarray:
    """
    Decide a common wavelength grid to use for equalization across
    multiple instruments.

    Typically this will:
    - Look at the intersection (or suitable union) of wavelength ranges.
    - Use the coarsest spacing (largest median delta_lambda) as a guide.

    Returns
    -------
    np.ndarray
        1D array of common wavelengths.
    """
    # Find overlapping wavelength range
    min_wl = max(profiles[inst_id].wavelengths.min() for inst_id in instrument_ids)
    max_wl = min(profiles[inst_id].wavelengths.max() for inst_id in instrument_ids)

    # Use the coarsest spacing (largest delta_lambda_med) to avoid over-sampling
    coarsest_spacing = max(profiles[inst_id].delta_lambda_med for inst_id in instrument_ids)

    # Generate common grid
    common_wl = np.arange(min_wl, max_wl + coarsest_spacing, coarsest_spacing)

    return common_wl


def build_equalization_mapping_for_instrument(
    instrument_profile: InstrumentProfile,
    wavelengths_common: np.ndarray,
    reference_profile: InstrumentProfile | None = None,
    transfer_model: TransferModel | None = None,
):
    """
    Build and return a callable that maps spectra from a given instrument
    into the common domain.

    This can combine:
    - Resampling to wavelengths_common
    - Optional smoothing (based on resolution estimates)
    - Optional calibration transfer (DS/PDS) if transfer_model is provided

    Returns
    -------
    callable
        A function f(X, wl_src) -> X_common
    """
    from .calibration_transfer import resample_to_grid, apply_ds, apply_pds

    def mapping_func(X: np.ndarray, wl_src: np.ndarray) -> np.ndarray:
        """Map spectra from source instrument to common grid."""
        # Step 1: Resample to common grid
        X_common = resample_to_grid(X, wl_src, wavelengths_common)

        # Step 2: Apply calibration transfer if provided
        if transfer_model is not None:
            if transfer_model.method == "ds":
                A = transfer_model.params["A"]
                X_common = apply_ds(X_common, A)
            elif transfer_model.method == "pds":
                B = transfer_model.params["B"]
                window = transfer_model.params.get("window", 11)
                X_common = apply_pds(X_common, B, window)

        # Step 3: Optional smoothing could be added here based on reference_profile
        # For now, we skip this step

        return X_common

    return mapping_func


def equalize_dataset(
    spectra_by_instrument: Dict[str, Tuple[np.ndarray, np.ndarray]],
    profiles: Dict[str, InstrumentProfile],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equalize spectra from multiple instruments into a common domain.

    Parameters
    ----------
    spectra_by_instrument : dict
        Mapping from instrument_id -> (wavelengths, X) where X is of
        shape (n_samples_i, n_wavelengths_i).
    profiles : dict
        Mapping from instrument_id -> InstrumentProfile.

    Returns
    -------
    (wavelengths_common, X_common) : Tuple[np.ndarray, np.ndarray]
        Common wavelength grid and stacked spectra from all instruments,
        after equalization.
    """
    # Choose common wavelength grid
    instrument_ids = list(spectra_by_instrument.keys())
    wavelengths_common = choose_common_grid(profiles, instrument_ids)

    # Equalize each instrument's spectra
    equalized_spectra = []

    for inst_id, (wavelengths, X) in spectra_by_instrument.items():
        profile = profiles[inst_id]

        # Build mapping function for this instrument
        mapping_func = build_equalization_mapping_for_instrument(
            instrument_profile=profile,
            wavelengths_common=wavelengths_common,
        )

        # Apply mapping
        X_common = mapping_func(X, wavelengths)
        equalized_spectra.append(X_common)

    # Stack all equalized spectra
    X_common = np.vstack(equalized_spectra)

    return wavelengths_common, X_common


if __name__ == "__main__":
    # Optional: simple self-test or placeholder for future examples.
    print("equalization skeleton loaded.")
