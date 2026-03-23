"""Baseline correction transformers for spectral data."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin


def rubber_band_baseline(y: np.ndarray) -> np.ndarray:
    """Compute rubber band (convex hull) baseline for a single spectrum.

    Uses Andrew's monotone chain algorithm to find the lower convex hull,
    which represents the baseline a rubber band would form if stretched
    underneath the spectrum.

    Parameters
    ----------
    y : array, shape (n_wavelengths,)
        Single spectrum intensity values.

    Returns
    -------
    baseline : array, shape (n_wavelengths,)
        Estimated rubber band baseline.
    """
    n = len(y)
    if n < 3:
        return np.zeros(n)

    # Andrew's monotone chain: compute lower convex hull.
    # Points are (i, y[i]) already sorted by x (i = 0..n-1).
    # We keep only vertices where the path makes right turns (clockwise).
    hull_indices = np.empty(n, dtype=np.intp)
    hull_size = 0

    for i in range(n):
        while hull_size >= 2:
            # Cross product of vectors (hull[-2] → hull[-1]) and (hull[-2] → i)
            o = hull_indices[hull_size - 2]
            a = hull_indices[hull_size - 1]
            cross = (a - o) * (y[i] - y[o]) - (y[a] - y[o]) * (i - o)
            if cross <= 0:
                hull_size -= 1
            else:
                break
        hull_indices[hull_size] = i
        hull_size += 1

    hi = hull_indices[:hull_size]
    baseline = np.interp(np.arange(n), hi, y[hi])

    return baseline


class BaselineRubberBand(BaseEstimator, TransformerMixin):
    """Rubber band (convex hull) baseline correction.

    Stretches a virtual rubber band underneath each spectrum using the lower
    convex hull, then subtracts the resulting baseline. Fully automatic — no
    tuneable parameters.

    Examples
    --------
    >>> from spectral_predict.baseline import BaselineRubberBand
    >>> baseline = BaselineRubberBand()
    >>> X_corrected = baseline.fit_transform(X)
    """

    def fit(self, X, y=None):
        """Fit transformer (stores number of features)."""
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X):
        """Apply rubber band baseline correction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data.

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_features)
            Baseline-corrected spectra.
        """
        X = np.asarray(X, dtype=np.float64)
        X_corrected = np.zeros_like(X)
        for i in range(X.shape[0]):
            bl = rubber_band_baseline(X[i, :])
            X_corrected[i, :] = X[i, :] - bl
        return X_corrected


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
        # Set fitted attributes for sklearn compatibility (required for Pipeline serialization)
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
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
        # Set fitted attributes for sklearn compatibility (required for Pipeline serialization)
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
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


class BaselineAirPLS(BaseEstimator, TransformerMixin):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).

    An improved version of ALS that adaptively determines weights based on
    the difference between data and fitted baseline, providing better
    convergence and more robust baseline estimation.

    Parameters
    ----------
    lam : float, default=1e5
        Smoothness parameter. Higher = smoother baseline.
        Typical range: 1e2 to 1e9

    max_iter : int, default=15
        Maximum iterations.

    tol : float, default=1e-3
        Convergence tolerance for weight changes.

    Examples
    --------
    >>> from spectral_predict.baseline import BaselineAirPLS
    >>> # Standard airPLS baseline correction
    >>> baseline = BaselineAirPLS(lam=1e6)
    >>> X_corrected = baseline.fit_transform(X_raman)

    Notes
    -----
    airPLS improves on ALS by:
    - Adaptive weight calculation based on fitting residuals
    - Automatic handling of different peak intensities
    - Generally faster convergence

    Particularly effective for Raman spectroscopy and NIR with fluorescence.

    References
    ----------
    Zhang, Z. M., Chen, S., & Liang, Y. Z. (2010). Baseline correction
    using adaptive iteratively reweighted penalized least squares.
    Analyst, 135(5), 1138-1146.
    """

    def __init__(self, lam=1e5, max_iter=15, tol=1e-3):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """Fit transformer (stores number of features)."""
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def _baseline_airpls_single(self, y):
        """
        Compute airPLS baseline for a single spectrum.

        Parameters
        ----------
        y : array, shape (n_wavelengths,)
            Single spectrum

        Returns
        -------
        baseline : array, shape (n_wavelengths,)
            Estimated baseline
        """
        L = len(y)

        # Second derivative matrix
        diags = np.array([1, -2, 1])
        D = sparse.diags(diags, [0, 1, 2], shape=(L - 2, L), format='csc')
        DTD = self.lam * D.T @ D

        w = np.ones(L)

        for iteration in range(self.max_iter):
            W = sparse.diags(w, 0, format='csc')
            Z = W + DTD

            try:
                z = spsolve(Z, w * y)
            except Exception:
                return np.zeros(L)

            # Compute residuals
            d = y - z

            # airPLS weight update: exponential decrease for positive residuals
            # (points above baseline)
            d_neg = d[d < 0]
            if len(d_neg) > 0:
                m = np.abs(d_neg).mean()
            else:
                m = 1.0

            # Weight function: exp(-|d|/m) for d > 0, 1 for d <= 0
            w_new = np.zeros(L)
            w_new[d <= 0] = 1.0
            w_new[d > 0] = np.exp(-np.abs(d[d > 0]) / (2 * m + 1e-10))

            # Add small regularization to avoid zero weights
            w_new = np.maximum(w_new, 1e-6)

            # Check convergence
            if np.sum(np.abs(w_new - w)) / np.sum(w) < self.tol:
                break

            w = w_new

        return z

    def transform(self, X):
        """
        Apply airPLS baseline correction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_features)
            Baseline-corrected spectra
        """
        X = np.asarray(X, dtype=np.float64)
        X_corrected = np.zeros_like(X)

        for i in range(X.shape[0]):
            baseline = self._baseline_airpls_single(X[i, :])
            X_corrected[i, :] = X[i, :] - baseline

        return X_corrected
