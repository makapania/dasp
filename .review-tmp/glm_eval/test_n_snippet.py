import numpy as np


def normalize_to_unit_area(spectra: np.ndarray) -> np.ndarray:
    """Normalize each spectrum so it has unit area (sum = 1).

    Args:
        spectra: 2D array of shape (n_samples, n_wavelengths).

    Returns:
        2D array of the same shape with each row summing to 1.
    """
    areas = spectra.sum(axis=1)
    return spectra / areas


def signal_to_noise(spectra: np.ndarray, noise_region: slice) -> np.ndarray:
    """Estimate signal-to-noise ratio for each spectrum.

    Args:
        spectra: 2D array (n_samples, n_wavelengths).
        noise_region: slice indicating wavelengths considered noise.

    Returns:
        1D array of length n_samples with S/N estimates.
    """
    signal = spectra.max(axis=1)
    noise = spectra[:, noise_region].std(axis=1)
    return signal / noise
