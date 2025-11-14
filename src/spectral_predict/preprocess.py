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

    def get_feature_names_out(self, input_features=None):
        """Pass through feature names unchanged.

        Parameters
        ----------
        input_features : array-like of str or None
            Input feature names

        Returns
        -------
        feature_names_out : ndarray of str
            Unchanged feature names
        """
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)

    def __sklearn_is_fitted__(self):
        """SNV requires no fitting - always ready to transform."""
        return True


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

    def get_feature_names_out(self, input_features=None):
        """Pass through feature names unchanged.

        Parameters
        ----------
        input_features : array-like of str or None
            Input feature names

        Returns
        -------
        feature_names_out : ndarray of str
            Unchanged feature names
        """
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)

    def __sklearn_is_fitted__(self):
        """Savgol derivative requires no learned parameters - always ready to transform."""
        return True


def build_preprocessing_pipeline(preprocess_name, deriv=None, window=None, polyorder=None):
    """
    Build a preprocessing pipeline from a configuration.

    Parameters
    ----------
    preprocess_name : str
        One of: 'raw', 'snv', 'deriv', 'snv_deriv', 'deriv_snv'
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

    elif preprocess_name == "deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol)]

    elif preprocess_name == "snv_deriv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("snv", SNV()), ("savgol", savgol)]

    elif preprocess_name == "deriv_snv":
        savgol = SavgolDerivative(deriv=deriv, window=window, polyorder=polyorder)
        return [("savgol", savgol), ("snv", SNV())]

    else:
        raise ValueError(f"Unknown preprocess: {preprocess_name}")
