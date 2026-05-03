import numpy as np
from sklearn.decomposition import PCA


def reduce_dimensions(
    X_train: np.ndarray, X_test: np.ndarray, n_components: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dimensionality of train and test spectra using PCA."""
    X_all = np.vstack([X_train, X_test])
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_all)
    n_train = len(X_train)
    return X_reduced[:n_train], X_reduced[n_train:]


def msc_correction(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Multiplicative Scatter Correction (MSC) to spectra.

    Uses the mean spectrum of each input set as the reference.
    """
    X_all = np.vstack([X_train, X_test])
    reference = X_all.mean(axis=0)
    corrected = np.zeros_like(X_all)
    for i in range(X_all.shape[0]):
        slope, intercept = np.polyfit(reference, X_all[i, :], 1)
        corrected[i, :] = (X_all[i, :] - intercept) / slope
    n_train = len(X_train)
    return corrected[:n_train], corrected[n_train:]
