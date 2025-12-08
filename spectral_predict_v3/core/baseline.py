"""
Baseline correction methods for spectral data.

Provides algorithms for removing baseline drift and fluorescence background
from spectral data, particularly useful for Raman spectroscopy.

Includes:
- BaselinePolynomial: Polynomial fitting to spectrum minima
- BaselineAsLS: Asymmetric Least Squares (Whittaker smoother)
- BaselineAirPLS: Adaptive iteratively reweighted Penalized Least Squares
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin


class BaselinePolynomial(BaseEstimator, TransformerMixin):
    """
    Polynomial baseline fitting and subtraction.

    Fits a polynomial to local minima of the spectrum and subtracts it.
    Works well for smooth, slowly-varying baselines.

    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial to fit (1=linear, 2=quadratic, etc.)
        Higher degrees fit more complex baselines but risk overfitting.

    n_segments : int, default=20
        Number of segments to divide spectrum into for finding local minima.
        More segments = more points for fitting = potentially better fit.

    percentile : float, default=10
        Percentile within each segment to use as the local minimum.
        Lower values select the actual minimum, higher values are more robust
        to noise.

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths in fitted data

    Examples
    --------
    >>> from spectral_predict_v3.core.baseline import BaselinePolynomial
    >>> bl = BaselinePolynomial(degree=3, n_segments=30)
    >>> X_corrected = bl.fit_transform(X)

    Notes
    -----
    This method works best for:
    - Smooth baseline drift
    - Instrument baseline offset
    - Linear or slowly-curving backgrounds

    For fluorescence backgrounds (asymmetric), use BaselineAsLS instead.

    References
    ----------
    Lieber, C. A., & Mahadevan-Jansen, A. (2003). Automated method for
    subtraction of fluorescence from biological Raman spectra.
    Applied spectroscopy, 57(11), 1363-1367.
    """

    def __init__(self, degree=2, n_segments=20, percentile=10):
        self.degree = degree
        self.n_segments = n_segments
        self.percentile = percentile

    def fit(self, X, y=None):
        """
        Fit transformer (stores number of features).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data
        y : Ignored

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
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
        X = np.asarray(X, dtype=np.float64)
        X_corrected = np.zeros_like(X)
        n_wavelengths = X.shape[1]

        # Create x-axis for polynomial fitting
        x = np.arange(n_wavelengths)

        # Segment boundaries
        segment_size = max(1, n_wavelengths // self.n_segments)

        for i in range(X.shape[0]):
            spectrum = X[i, :]

            # Find local minima in each segment
            min_x = []
            min_y = []

            for j in range(self.n_segments):
                start = j * segment_size
                end = min((j + 1) * segment_size, n_wavelengths)

                if start >= n_wavelengths:
                    break

                segment = spectrum[start:end]
                if len(segment) > 0:
                    # Use percentile to be robust to noise
                    local_min = np.percentile(segment, self.percentile)
                    # Find index closest to this percentile value
                    idx = np.argmin(np.abs(segment - local_min))
                    min_x.append(start + idx)
                    min_y.append(segment[idx])

            if len(min_x) < self.degree + 1:
                # Not enough points - fall back to just endpoints
                min_x = [0, n_wavelengths - 1]
                min_y = [spectrum[0], spectrum[-1]]

            # Fit polynomial to minima
            try:
                coeffs = np.polyfit(min_x, min_y, self.degree)
                baseline = np.polyval(coeffs, x)
            except (np.linalg.LinAlgError, np.RankWarning):
                # Fitting failed - return unchanged
                baseline = np.zeros(n_wavelengths)

            # Subtract baseline
            X_corrected[i, :] = spectrum - baseline

        return X_corrected


class BaselineAsLS(BaseEstimator, TransformerMixin):
    """
    Asymmetric Least Squares (AsLS) baseline correction.

    Uses Whittaker smoothing with asymmetric weights to estimate baseline.
    Particularly effective for Raman spectroscopy with fluorescence background
    where peaks are asymmetric (sharp peaks on smooth background).

    Parameters
    ----------
    lam : float, default=1e5
        Smoothness parameter (lambda). Higher values = smoother baseline.
        Typical range: 1e2 to 1e9
        - 1e2-1e4: More flexible, follows data closely
        - 1e5-1e6: Moderate smoothness (good default)
        - 1e7-1e9: Very smooth, slow-varying baseline

    p : float, default=0.01
        Asymmetry parameter. Typical range: 0.001 to 0.1
        - Lower p: Baseline fits more tightly under peaks (for positive peaks)
        - Higher p: More symmetric fitting
        For Raman (positive peaks on baseline): use p=0.001-0.05
        For absorption spectra: may need p=0.5-0.99

    max_iter : int, default=10
        Maximum iterations for asymmetric weighting convergence.

    tol : float, default=1e-3
        Convergence tolerance (relative change in weights).

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths in fitted data

    Examples
    --------
    >>> from spectral_predict_v3.core.baseline import BaselineAsLS
    >>> bl = BaselineAsLS(lam=1e6, p=0.01)
    >>> X_corrected = bl.fit_transform(X_raman)

    Notes
    -----
    The algorithm iteratively:
    1. Fits a smooth baseline using Whittaker smoother
    2. Assigns weights: low weight where data > baseline (peaks)
    3. Repeats until convergence

    This is the classic "asymmetric least squares" method. For faster
    convergence or specific applications, consider BaselineAirPLS.

    References
    ----------
    Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with
    asymmetric least squares smoothing. Leiden University Medical Centre
    Report, 1(1), 5.
    """

    def __init__(self, lam=1e5, p=0.01, max_iter=10, tol=1e-3):
        self.lam = lam
        self.p = p
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """
        Fit transformer (stores number of features).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data
        y : Ignored

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def _baseline_als_single(self, y):
        """
        Compute AsLS baseline for a single spectrum.

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

        # Construct second derivative matrix D
        # D is (L-2, L) such that D @ y gives second differences
        diags = np.array([1, -2, 1])
        D = sparse.diags(diags, [0, 1, 2], shape=(L - 2, L), format='csc')

        # Whittaker smoother: (W + lam * D'D) z = W y
        # W is diagonal weight matrix
        # Start with uniform weights
        w = np.ones(L)
        W = sparse.diags(w, 0, format='csc')

        DTD = self.lam * D.T @ D

        for iteration in range(self.max_iter):
            W = sparse.diags(w, 0, format='csc')
            Z = W + DTD

            try:
                z = spsolve(Z, w * y)
            except Exception:
                # Solver failed - return zeros
                return np.zeros(L)

            # Update weights asymmetrically
            # Points above baseline get weight p (small)
            # Points below baseline get weight 1-p (large)
            w_new = self.p * (y > z) + (1 - self.p) * (y <= z)

            # Check convergence
            if np.sum(np.abs(w_new - w)) / np.sum(w) < self.tol:
                break

            w = w_new

        return z

    def transform(self, X):
        """
        Apply AsLS baseline correction.

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
            baseline = self._baseline_als_single(X[i, :])
            X_corrected[i, :] = X[i, :] - baseline

        return X_corrected


class BaselineAirPLS(BaseEstimator, TransformerMixin):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).

    An improved version of AsLS that adaptively determines weights based on
    the difference between data and fitted baseline, providing better
    convergence and more robust baseline estimation.

    Parameters
    ----------
    lam : float, default=1e5
        Smoothness parameter. Higher = smoother baseline.

    max_iter : int, default=15
        Maximum iterations.

    tol : float, default=1e-3
        Convergence tolerance for weight changes.

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths in fitted data

    Examples
    --------
    >>> from spectral_predict_v3.core.baseline import BaselineAirPLS
    >>> bl = BaselineAirPLS(lam=1e6)
    >>> X_corrected = bl.fit_transform(X_raman)

    Notes
    -----
    airPLS improves on AsLS by:
    - Adaptive weight calculation based on fitting residuals
    - Automatic handling of different peak intensities
    - Generally faster convergence

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
        """Fit transformer."""
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
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
        W = sparse.diags(w, 0, format='csc')

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
        """Apply airPLS baseline correction."""
        X = np.asarray(X, dtype=np.float64)
        X_corrected = np.zeros_like(X)

        for i in range(X.shape[0]):
            baseline = self._baseline_airpls_single(X[i, :])
            X_corrected[i, :] = X[i, :] - baseline

        return X_corrected


class SavgolSmooth(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay smoothing filter (without differentiation).

    Applies a low-pass filter that preserves peak shape and height
    while removing high-frequency noise. Unlike SavgolDerivative,
    this does not compute derivatives (deriv=0).

    Parameters
    ----------
    window_length : int, default=11
        Length of filter window (must be odd and > polyorder).
        Larger windows = more smoothing but may blur peaks.

    polyorder : int, default=2
        Order of polynomial used for fitting.
        Higher order preserves sharper features.

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths in fitted data

    Examples
    --------
    >>> from spectral_predict_v3.core.baseline import SavgolSmooth
    >>> smoother = SavgolSmooth(window_length=15, polyorder=3)
    >>> X_smooth = smoother.fit_transform(X)

    Notes
    -----
    The Savitzky-Golay filter convolves the data with coefficients
    derived from fitting a polynomial of degree `polyorder` to
    `window_length` points. This preserves moments of the original
    data better than simple moving average smoothing.

    For noisy spectra:
    - Increase window_length for more smoothing
    - Keep polyorder >= 2 to preserve peak shapes

    References
    ----------
    Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation
    of data by simplified least squares procedures. Analytical chemistry,
    36(8), 1627-1639.
    """

    def __init__(self, window_length=11, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X, y=None):
        """Fit transformer."""
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Apply Savitzky-Golay smoothing.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data

        Returns
        -------
        X_smooth : ndarray, shape (n_samples, n_features)
            Smoothed spectra
        """
        from scipy.signal import savgol_filter

        X = np.asarray(X, dtype=np.float64)

        # Ensure odd window
        window = self.window_length
        if window % 2 == 0:
            window = window + 1

        # Validate parameters
        if window > X.shape[1]:
            raise ValueError(
                f"Window length ({window}) must be <= number of features ({X.shape[1]})"
            )

        if window <= self.polyorder:
            raise ValueError(
                f"Window length ({window}) must be > polyorder ({self.polyorder})"
            )

        # Apply smoothing (deriv=0)
        X_smooth = savgol_filter(
            X,
            window_length=window,
            polyorder=self.polyorder,
            deriv=0,
            axis=1
        )

        return X_smooth


# Convenience functions for direct use
def polynomial_baseline(X, degree=2, n_segments=20, percentile=10):
    """
    Apply polynomial baseline correction.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Spectral data
    degree : int
        Polynomial degree
    n_segments : int
        Number of segments for finding minima
    percentile : float
        Percentile for local minimum selection

    Returns
    -------
    X_corrected : ndarray
        Baseline-corrected spectra
    """
    bl = BaselinePolynomial(degree=degree, n_segments=n_segments, percentile=percentile)
    return bl.fit_transform(X)


def als_baseline(X, lam=1e5, p=0.01, max_iter=10):
    """
    Apply AsLS baseline correction.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Spectral data
    lam : float
        Smoothness parameter
    p : float
        Asymmetry parameter
    max_iter : int
        Maximum iterations

    Returns
    -------
    X_corrected : ndarray
        Baseline-corrected spectra
    """
    bl = BaselineAsLS(lam=lam, p=p, max_iter=max_iter)
    return bl.fit_transform(X)


def airpls_baseline(X, lam=1e5, max_iter=15):
    """
    Apply airPLS baseline correction.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Spectral data
    lam : float
        Smoothness parameter
    max_iter : int
        Maximum iterations

    Returns
    -------
    X_corrected : ndarray
        Baseline-corrected spectra
    """
    bl = BaselineAirPLS(lam=lam, max_iter=max_iter)
    return bl.fit_transform(X)


def savgol_smooth(X, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay smoothing.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Spectral data
    window_length : int
        Filter window length
    polyorder : int
        Polynomial order

    Returns
    -------
    X_smooth : ndarray
        Smoothed spectra
    """
    sm = SavgolSmooth(window_length=window_length, polyorder=polyorder)
    return sm.fit_transform(X)
