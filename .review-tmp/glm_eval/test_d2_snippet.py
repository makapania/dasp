import numpy as np
from scipy.signal import savgol_filter


def smooth_spectra(
    X_train: np.ndarray,
    X_test: np.ndarray,
    window: int = 11,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Savitzky-Golay smoothing to all spectra."""
    X_all = np.vstack([X_train, X_test])
    X_smoothed = savgol_filter(X_all, window_length=window, polyorder=polyorder, axis=1)
    n_train = len(X_train)
    return X_smoothed[:n_train], X_smoothed[n_train:]


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each wavelength feature to zero mean / unit std."""
    X_all = np.vstack([X_train, X_test])
    X_norm = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
    n_train = len(X_train)
    return X_norm[:n_train], X_norm[n_train:]
