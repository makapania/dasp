"""Preprocessing transformers for spectral data."""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SNV(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) transformation.

    Normalizes each spectrum (row) by subtracting its mean and dividing by its standard deviation.
    """

    def fit(self, X, y=None):
        """Fit transformer (no-op for SNV)."""
        return self

    def transform(self, X):
        """
        Apply SNV transformation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_snv : ndarray, shape (n_samples, n_features)
            SNV-transformed spectra
        """
        X = np.asarray(X)
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)

        # Avoid division by zero
        stds[stds == 0] = 1.0

        return (X - means) / stds


class MSC(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction (MSC) transformation.

    Corrects for additive and multiplicative scatter effects by fitting each spectrum
    to a reference spectrum using a linear model: spectrum_i = a + b * reference,
    then returning (spectrum - a) / b.

    Parameters
    ----------
    reference : str or array-like, default='mean'
        Reference spectrum to use. Options:
        - 'mean': Use mean of all spectra (computed during fit)
        - 'median': Use median of all spectra (computed during fit)
        - array-like: Use provided custom reference spectrum
    """

    def __init__(self, reference='mean'):
        self.reference = reference

    def fit(self, X, y=None):
        """
        Compute reference spectrum.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data
        y : None
            Ignored

        Returns
        -------
        self
        """
        X = np.asarray(X)

        if isinstance(self.reference, str):
            if self.reference == 'mean':
                self.reference_ = X.mean(axis=0)
            elif self.reference == 'median':
                self.reference_ = np.median(X, axis=0)
            else:
                raise ValueError(f"Unknown reference type: {self.reference}")
        else:
            self.reference_ = np.asarray(self.reference)

        return self

    def transform(self, X):
        """
        Apply MSC transformation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_msc : ndarray, shape (n_samples, n_features)
            MSC-corrected spectra
        """
        X = np.asarray(X)
        X_msc = np.zeros_like(X)

        for i in range(X.shape[0]):
            # Fit linear model: spectrum_i = a + b * reference
            # Using least squares: [a, b] = (A^T A)^-1 A^T y
            # where A = [[1, ref[0]], [1, ref[1]], ..., [1, ref[n]]]
            ref = self.reference_
            spectrum = X[i, :]

            # Build design matrix
            A = np.vstack([np.ones(len(ref)), ref]).T

            # Solve least squares
            coeffs = np.linalg.lstsq(A, spectrum, rcond=None)[0]
            a, b = coeffs

            # Handle division by zero
            if abs(b) < 1e-6:
                b = 1.0

            # Apply correction
            X_msc[i, :] = (spectrum - a) / b

        return X_msc


class SavgolDerivative(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay derivative transformation.

    Parameters
    ----------
    deriv : int, default=1
        Derivative order (1 or 2)
    window : int, default=7
        Window length (must be odd; if even, will be incremented by 1)
    polyorder : int, optional
        Polynomial order. If None, defaults to 2 for deriv=1, 3 for deriv=2
    """

    def __init__(self, deriv=1, window=7, polyorder=None):
        self.deriv = deriv
        self.window = window
        self.polyorder = polyorder

    def fit(self, X, y=None):
        """Fit transformer (no-op for Savgol)."""
        return self

    def transform(self, X):
        """
        Apply Savitzky-Golay derivative.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_deriv : ndarray, shape (n_samples, n_features)
            Derivative spectra
        """
        X = np.asarray(X)

        # Ensure odd window
        window = self.window
        if window % 2 == 0:
            window = window + 1

        # Default polyorder
        polyorder = self.polyorder
        if polyorder is None:
            polyorder = 2 if self.deriv == 1 else 3

        # Validate
        if window < polyorder + 2:
            raise ValueError(f"Window length ({window}) must be >= polyorder ({polyorder}) + 2")

        if window > X.shape[1]:
            raise ValueError(
                f"Window length ({window}) must be <= number of features ({X.shape[1]})"
            )

        # Apply along axis=1 (features)
        X_deriv = savgol_filter(
            X, window_length=window, polyorder=polyorder, deriv=self.deriv, axis=1
        )

        return X_deriv


def build_preprocessing_pipeline(preprocess_name, deriv=None, window=None, polyorder=None):
    """
    Build a preprocessing pipeline from a configuration.

    Parameters
    ----------
    preprocess_name : str
        One of: 'raw', 'snv', 'deriv', 'snv_deriv', 'deriv_snv', 'msc', 'msc_deriv', 'deriv_msc'
    deriv : int, optional
        Derivative order (for deriv-based pipelines)
    window : int, optional
        Window size (for deriv-based pipelines)
    polyorder : int, optional
        Polynomial order (for deriv-based pipelines)

    Returns
    -------
    steps : list
        List of (name, transformer) tuples
    """
    from sklearn.pipeline import Pipeline

    if preprocess_name == "raw":
        return []

    elif preprocess_name == "snv":
        return [("snv", SNV())]

    elif preprocess_name == "msc":
        return [("msc", MSC())]

    elif preprocess_name == "deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol)]

    elif preprocess_name == "snv_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("snv", SNV()), ("savgol", savgol)]

    elif preprocess_name == "deriv_snv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol), ("snv", SNV())]

    elif preprocess_name == "msc_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("msc", MSC()), ("savgol", savgol)]

    elif preprocess_name == "deriv_msc":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol), ("msc", MSC())]

    else:
        raise ValueError(f"Unknown preprocess: {preprocess_name}")
