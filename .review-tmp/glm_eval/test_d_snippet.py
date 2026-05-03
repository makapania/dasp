import numpy as np

def preprocess_spectra(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SNV preprocessing to train and test spectra together."""
    X_all = np.vstack([X_train, X_test])
    X_all_snv = (X_all - X_all.mean(axis=1, keepdims=True)) / X_all.std(axis=1, keepdims=True)
    n_train = len(X_train)
    return X_all_snv[:n_train], X_all_snv[n_train:]


def baseline_correct(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply asymmetric least squares baseline correction to all spectra."""
    X_all = np.vstack([X_train, X_test])
    baseline = np.median(X_all, axis=0)
    X_corrected = X_all - baseline
    n_train = len(X_train)
    return X_corrected[:n_train], X_corrected[n_train:]
