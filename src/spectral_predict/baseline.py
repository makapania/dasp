"""Baseline correction transformers for spectral data."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin


class BaselineALS(BaseEstimator, TransformerMixin):
    """
    Asymmetric Least Squares (ALS) baseline correction.

    This is one of the most important preprocessing steps for spectroscopy.
    The ALS algorithm fits a smooth baseline underneath peaks and subtracts it.

    Parameters
    ----------
    lambda_ : float, default=1e5
        Smoothness parameter (larger = smoother baseline)
        Typical range: 1e2 to 1e9
        - 1e5-1e6: Good for most spectra
        - 1e2-1e4: Less smooth, follows signal more closely
        - 1e7-1e9: Very smooth, good for broad baselines

    p : float, default=0.001
        Asymmetry parameter (0.001 - 0.1)
        - 0.001-0.01: Strong asymmetry (baseline stays under peaks)
        - 0.1: Less asymmetry (baseline can rise above signal)

    niter : int, default=10
        Number of iterations (typically 10-20)

    Reference
    ---------
    Eilers, P. H. C., & Boelens, H. F. M. (2005).
    Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 1(1), 5.

    Examples
    --------
    >>> from spectral_predict.baseline import BaselineALS
    >>> # Standard baseline correction
    >>> baseline = BaselineALS(lambda_=1e5, p=0.001)
    >>> X_corrected = baseline.fit_transform(X)
    >>>
    >>> # Very smooth baseline for broad background
    >>> baseline = BaselineALS(lambda_=1e7, p=0.001)
    >>> X_corrected = baseline.fit_transform(X)
    """

    def __init__(self, lambda_=1e5, p=0.001, niter=10):
        self.lambda_ = lambda_
        self.p = p
        self.niter = niter

    def fit(self, X, y=None):
        """Fit transformer (no-op for ALS baseline correction)."""
        return self

    def transform(self, X):
        """
        Apply ALS baseline correction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_features)
            Baseline-corrected spectra
        """
        X = np.asarray(X)
        X_corrected = np.zeros_like(X)

        for i in range(X.shape[0]):
            y = X[i, :]
            L = len(y)

            # Create difference matrix for smoothness
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))

            # Initialize weights
            w = np.ones(L)

            # Iterative reweighting
            for _ in range(self.niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + self.lambda_ * D.dot(D.transpose())
                # Convert to CSC format for efficient solving
                Z = Z.tocsc()
                z = spsolve(Z, w * y)

                # Update weights (asymmetric)
                w = self.p * (y > z) + (1 - self.p) * (y < z)

            # Subtract baseline
            X_corrected[i, :] = y - z

        return X_corrected


class BaselinePolynomial(BaseEstimator, TransformerMixin):
    """
    Polynomial baseline correction.

    Fits a polynomial to each spectrum and subtracts it.
    Simpler than ALS but less flexible.

    Parameters
    ----------
    degree : int, default=3
        Polynomial degree (1-5 typical)
        - 1: Linear baseline
        - 2-3: Gentle curved baseline (most common)
        - 4-5: Complex curved baseline

    Examples
    --------
    >>> from spectral_predict.baseline import BaselinePolynomial
    >>> # Linear baseline
    >>> baseline = BaselinePolynomial(degree=1)
    >>> X_corrected = baseline.fit_transform(X)
    >>>
    >>> # Quadratic baseline (more flexible)
    >>> baseline = BaselinePolynomial(degree=2)
    >>> X_corrected = baseline.fit_transform(X)
    """

    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, X, y=None):
        """Fit transformer (no-op for polynomial baseline correction)."""
        return self

    def transform(self, X):
        """
        Apply polynomial baseline correction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_features)
            Baseline-corrected spectra
        """
        X = np.asarray(X)
        X_corrected = np.zeros_like(X)

        # Wavelength indices (0 to n_features-1)
        wavelengths = np.arange(X.shape[1])

        for i in range(X.shape[0]):
            y = X[i, :]

            # Fit polynomial
            coeffs = np.polyfit(wavelengths, y, self.degree)
            baseline = np.polyval(coeffs, wavelengths)

            # Subtract baseline
            X_corrected[i, :] = y - baseline

        return X_corrected
