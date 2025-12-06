"""
Preprocessing transformers for spectral data (v3 standalone).

Forked from v1 - simplified for v3's numpy-first approach.

Includes:
- SNV (Standard Normal Variate)
- SavgolDerivative (Savitzky-Golay derivatives)
- MSC (Multiplicative Scatter Correction)
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


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


class MSC(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction (MSC).

    Removes multiplicative scatter effects and baseline offset by fitting each
    spectrum to a reference spectrum (typically the mean of the calibration set).
    Similar to SNV but uses a common reference rather than per-spectrum normalization.

    For each spectrum s_i:
        s_i_corrected = (s_i - a_i) / b_i
    where a_i and b_i are obtained by linear regression: s_i = a_i + b_i * s_ref

    Parameters
    ----------
    reference : {'mean', 'median'} or array-like, default='mean'
        Reference spectrum to use:
        - 'mean': Use mean spectrum of training set
        - 'median': Use median spectrum of training set
        - array: Use provided spectrum as reference

    Attributes
    ----------
    reference_ : array, shape (n_wavelengths,)
        Reference spectrum used for correction

    n_features_in_ : int
        Number of wavelengths

    Examples
    --------
    >>> from spectral_predict_v3.core.preprocess import MSC
    >>> msc = MSC(reference='mean')
    >>> X_corrected = msc.fit_transform(X_train)
    >>> X_test_corrected = msc.transform(X_test)

    References
    ----------
    Geladi et al. (1985). "Linearization and scatter-correction for near-infrared
    reflectance spectra of meat." Applied Spectroscopy, 39(3), 491-500.
    """

    def __init__(self, reference='mean'):
        self.reference = reference

    def fit(self, X, y=None):
        """
        Compute reference spectrum from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Training spectral data
        y : Ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        self : object
            Fitted transformer
        """
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        if isinstance(self.reference, str):
            if self.reference == 'mean':
                self.reference_ = np.mean(X, axis=0)
            elif self.reference == 'median':
                self.reference_ = np.median(X, axis=0)
            else:
                raise ValueError(f"reference must be 'mean', 'median', or array-like, got {self.reference}")
        else:
            self.reference_ = np.asarray(self.reference)
            if len(self.reference_) != self.n_features_in_:
                raise ValueError(
                    f"Reference spectrum length ({len(self.reference_)}) must match "
                    f"number of features ({self.n_features_in_})"
                )

        return self

    def transform(self, X):
        """
        Apply MSC to spectral data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to correct

        Returns
        -------
        X_corrected : array, shape (n_samples, n_wavelengths)
            Scatter-corrected spectra
        """
        X = np.asarray(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but MSC was fitted with {self.n_features_in_} features"
            )

        # Check if reference has near-zero variance
        if np.std(self.reference_) < 1e-12:
            warnings.warn(
                "Reference spectrum has near-zero variance. MSC correction skipped, returning data unchanged.",
                UserWarning
            )
            return X.copy()

        X_corrected = np.zeros_like(X)

        for i in range(X.shape[0]):
            # Check if spectrum has near-zero variance
            if np.std(X[i, :]) < 1e-12:
                X_corrected[i, :] = X[i, :]
                continue

            # Fit: s_i = a + b * s_ref
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    fit = np.polyfit(self.reference_, X[i, :], 1)
            except (np.RankWarning, np.linalg.LinAlgError):
                # Spectrum or reference is constant/degenerate - return unchanged
                X_corrected[i, :] = X[i, :]
                continue

            # Avoid division by near-zero slope
            if abs(fit[0]) < 1e-10:
                X_corrected[i, :] = X[i, :]
                continue

            # Correct: s_corrected = (s_i - a) / b
            X_corrected[i, :] = (X[i, :] - fit[1]) / fit[0]

        return X_corrected
