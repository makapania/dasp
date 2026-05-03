import numpy as np
from scipy.signal import detrend


def per_spectrum_minmax(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalize each spectrum to [0, 1]."""
    X_all = np.vstack([X_train, X_test])
    X_min = X_all.min(axis=1, keepdims=True)
    X_max = X_all.max(axis=1, keepdims=True)
    X_norm = (X_all - X_min) / (X_max - X_min)
    n_train = len(X_train)
    return X_norm[:n_train], X_norm[n_train:]


def per_spectrum_l2(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalize each spectrum (unit norm per row)."""
    X_all = np.vstack([X_train, X_test])
    norms = np.linalg.norm(X_all, axis=1, keepdims=True)
    X_norm = X_all / norms
    n_train = len(X_train)
    return X_norm[:n_train], X_norm[n_train:]


def detrend_per_spectrum(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly detrend each spectrum independently."""
    X_all = np.vstack([X_train, X_test])
    X_detrended = detrend(X_all, axis=1, type="linear")
    n_train = len(X_train)
    return X_detrended[:n_train], X_detrended[n_train:]


def baseline_subtract_per_spectrum(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each spectrum's own mean of the first 10 wavelengths as baseline."""
    X_all = np.vstack([X_train, X_test])
    baselines = X_all[:, :10].mean(axis=1, keepdims=True)
    X_corrected = X_all - baselines
    n_train = len(X_train)
    return X_corrected[:n_train], X_corrected[n_train:]
