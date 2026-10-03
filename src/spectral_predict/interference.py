"""
Interference removal transformers for spectral data.

This module provides methods for removing systematic interference (moisture,
temperature, particle size effects, etc.) from NIR/spectral data. All transformers
follow sklearn's BaseEstimator and TransformerMixin pattern for pipeline compatibility.

Methods implemented:
- WavelengthExcluder: Remove specified wavelength ranges (e.g., moisture bands)
- MSC (Multiplicative Scatter Correction): Alternative to SNV for scatter correction
- OSC (Orthogonal Signal Correction): Remove Y-orthogonal systematic variation
- EPO (External Parameter Orthogonalization): Remove specific interferents using reference library
- GLSW (Generalized Least Squares Weighting): Optimal wavelength weighting for heteroscedastic noise
- DOSC (Direct Orthogonal Signal Correction): Simplified OSC variant

Literature References:
------------------
OSC:
    Wold et al. (1998). "Orthogonal signal correction of near-infrared spectra."
    Chemometrics and Intelligent Laboratory Systems, 44(1-2), 175-185.

EPO:
    Roger et al. (2003). "EPO-PLS external parameter orthogonalisation of PLS
    application to temperature-independent measurement of sugar content of intact fruits."
    Chemometrics and Intelligent Laboratory Systems, 66(2), 191-204.

GLSW:
    Seasholtz & Kowalski (1993). "The parsimony principle applied to multivariate calibration."
    Analytica Chimica Acta, 277(2), 165-177.

MSC:
    Geladi et al. (1985). "Linearization and scatter-correction for near-infrared
    reflectance spectra of meat." Applied Spectroscopy, 39(3), 491-500.

Usage Examples:
--------------
Basic wavelength exclusion:
    >>> from spectral_predict.interference import WavelengthExcluder
    >>> # Exclude common moisture absorption bands
    >>> excluder = WavelengthExcluder(wavelengths, exclude_ranges=[(1400, 1500), (1900, 2000)])
    >>> X_filtered = excluder.fit_transform(X)

Simple moisture/temperature removal (OSC):
    >>> from spectral_predict.interference import OSC
    >>> osc = OSC(n_components=1)
    >>> X_corrected = osc.fit_transform(X_train, y_train)
    >>> X_test_corrected = osc.transform(X_test)

Advanced interferent removal (EPO):
    >>> from spectral_predict.interference import EPO
    >>> # Load interferent library (e.g., moisture spectra at different levels)
    >>> X_moisture = load_moisture_library()  # Shape: (n_interferent_samples, n_wavelengths)
    >>> epo = EPO(n_components=3)
    >>> epo.fit(X_train, y_train, X_interferents=X_moisture)
    >>> X_corrected = epo.transform(X_train)

Pipeline integration:
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> pipeline = Pipeline([
    ...     ('wavelength_exclude', WavelengthExcluder(wavelengths, exclude_ranges=[(1400, 1500)])),
    ...     ('osc', OSC(n_components=2)),
    ...     ('pls', PLSRegression(n_components=10))
    ... ])
    >>> pipeline.fit(X_train, y_train)
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import warnings


class WavelengthExcluder(BaseEstimator, TransformerMixin):
    """
    Exclude specified wavelength ranges from spectral data.

    Useful for removing regions dominated by noise or interference (e.g., strong
    water absorption bands at 1400-1500 nm and 1900-2000 nm in NIR spectroscopy).

    Parameters
    ----------
    wavelengths : array-like, shape (n_wavelengths,)
        Wavelength values corresponding to spectral channels

    exclude_ranges : list of tuples, optional
        List of (min, max) wavelength ranges to exclude.
        Default: [(1400, 1500), (1900, 2000)] (common NIR moisture bands)

    invert : bool, default=False
        If True, KEEP only the specified ranges (exclude everything else)

    Attributes
    ----------
    mask_ : array, shape (n_wavelengths,)
        Boolean mask indicating which wavelengths to keep (True) or exclude (False)

    n_features_in_ : int
        Number of features (wavelengths) before exclusion

    n_features_out_ : int
        Number of features (wavelengths) after exclusion

    wavelengths_out_ : array, shape (n_features_out_,)
        Wavelength values after exclusion

    Examples
    --------
    >>> wavelengths = np.arange(1000, 2501)  # 1000-2500 nm
    >>> X = np.random.randn(100, len(wavelengths))
    >>>
    >>> # Exclude moisture bands
    >>> excluder = WavelengthExcluder(wavelengths, exclude_ranges=[(1400, 1500), (1900, 2000)])
    >>> X_filtered = excluder.fit_transform(X)
    >>> print(f"Original: {X.shape[1]} wavelengths, Filtered: {X_filtered.shape[1]} wavelengths")
    >>>
    >>> # Custom exclusion
    >>> excluder = WavelengthExcluder(wavelengths, exclude_ranges=[(2300, 2400)])  # CO2 band
    >>> X_filtered = excluder.fit_transform(X)
    """

    def __init__(self, wavelengths, exclude_ranges=None, invert=False):
        self.wavelengths = wavelengths
        self.exclude_ranges = exclude_ranges if exclude_ranges is not None else [(1400, 1500), (1900, 2000)]
        self.invert = invert

    def fit(self, X, y=None):
        """
        Compute wavelength exclusion mask.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
        y : Ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        self : object
            Fitted transformer
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        self.n_features_in_ = X.shape[1]

        if len(self.wavelengths) != self.n_features_in_:
            raise ValueError(
                f"Wavelength array length ({len(self.wavelengths)}) must match "
                f"number of features in X ({self.n_features_in_})"
            )

        # Create mask: True = keep, False = exclude
        if self.invert:
            # Invert mode: start with all excluded, then include specified ranges
            self.mask_ = np.zeros(self.n_features_in_, dtype=bool)
            for wl_min, wl_max in self.exclude_ranges:
                in_range = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
                self.mask_[in_range] = True  # Keep this range
        else:
            # Normal mode: start with all included, then exclude specified ranges
            self.mask_ = np.ones(self.n_features_in_, dtype=bool)
            for wl_min, wl_max in self.exclude_ranges:
                in_range = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
                self.mask_[in_range] = False  # Exclude this range

        self.n_features_out_ = np.sum(self.mask_)
        self.wavelengths_out_ = self.wavelengths[self.mask_]

        if self.n_features_out_ == 0:
            warnings.warn(
                "All wavelengths excluded! Check exclude_ranges parameter.",
                UserWarning
            )

        return self

    def transform(self, X):
        """
        Apply wavelength exclusion to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data

        Returns
        -------
        X_filtered : array, shape (n_samples, n_features_out_)
            Spectral data with excluded wavelengths removed
        """
        check_is_fitted(self, ['mask_', 'n_features_out_'])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but WavelengthExcluder was fitted with "
                f"{self.n_features_in_} features"
            )

        return X[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        """
        Get wavelength values after exclusion.

        Returns
        -------
        wavelengths_out : array, shape (n_features_out_,)
            Remaining wavelengths after exclusion
        """
        check_is_fitted(self, 'wavelengths_out_')
        return self.wavelengths_out_


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
    >>> from spectral_predict.interference import MSC
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
        X = check_array(X, accept_sparse=False, dtype=np.float64)
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
        check_is_fitted(self, ['reference_', 'n_features_in_'])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

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


def _as_2d_target(y) -> np.ndarray:
    y = check_array(y, accept_sparse=False, dtype=np.float64, ensure_2d=False)
    return y.reshape(-1, 1) if y.ndim == 1 else y


# Fitted-state versions. OSC/DOSC objects pickled before 2026-10 (e.g. inside a
# saved preprocessing pipeline) have no ``fit_version_``; their transform replays
# the old output exactly so the saved downstream model still gets its inputs.
_OSC_FIT_VERSION = 2
_DOSC_FIT_VERSION = 2


def _warn_legacy(name):
    warnings.warn(
        f"{name} was fitted by an older dasp whose removed scores were not orthogonal "
        "to y. Replaying its original output so a saved model's predictions are "
        "unchanged. Refit the preprocessing to use the corrected method.",
        UserWarning,
    )


class OSC(BaseEstimator, TransformerMixin):
    """
    Orthogonal Signal Correction (OSC), Fearn (2000) formulation.

    Removes the largest systematic variation in X (spectra) whose scores are
    orthogonal to y. Each component k is found as follows (X and y mean-centred,
    X already deflated by components 1..k-1):

    1. Project the weight space away from the y-covariance directions:
       ``Z = X (I - Q Q^T)``, where Q is an orthonormal basis of ``X^T y``.
    2. The weight ``w`` is the first right singular vector of Z (the
       largest-variance direction among those with ``(X w)^T y = 0``).
    3. Score ``t = X w`` (so ``t^T y = 0`` exactly), loading
       ``p = X^T t / (t^T t)``, and deflate ``X <- X - t p^T``.

    The weights W and loadings P are stored and replayed on new spectra in
    ``transform``, which needs no y.

    Parameters
    ----------
    n_components : int, default=1
        Number of y-orthogonal components to remove (typically 1-3).

    tol : float, default=1e-10
        Relative singular-value threshold below which no further component is
        extracted (nothing systematic is left that is orthogonal to y).

    Attributes
    ----------
    weights_ : ndarray, shape (n_wavelengths, n_components_)
        Weight vectors w; scores of new data are ``(X - X_mean_) @ w``.

    loadings_ : ndarray, shape (n_wavelengths, n_components_)
        Loading vectors p used to deflate.

    P_osc_ : ndarray, shape (n_wavelengths, n_components_)
        Alias of ``weights_`` (kept for existing callers).

    n_components_ : int
        Number of components actually removed.

    variance_removed_ : ndarray, shape (n_components_,)
        Share of the total (centred) X sum of squares removed by each component.

    X_mean_ : ndarray, shape (n_wavelengths,)
        Training mean used to centre new data before computing scores.

    References
    ----------
    Fearn, T. (2000). On orthogonal signal correction. Chemometrics and
    Intelligent Laboratory Systems 50(1):47-52.
    Wold, S., Antti, H., Lindgren, F. & Ohman, J. (1998). Orthogonal signal
    correction of near-infrared spectra. Chemometrics and Intelligent
    Laboratory Systems 44(1-2):175-185.

    Notes
    -----
    OSC must be fitted with y. Fit it on training data only; within
    cross-validation it belongs inside the fold.

    The output is on the original spectral scale: ``X - T P^T`` with the scores T
    computed from mean-centred X. The training mean is not subtracted.

    The previous implementation removed the first PLS loading, i.e. the
    y-PREDICTIVE direction (finding R025).
    """

    def __init__(self, n_components=1, tol=1e-10):
        self.n_components = n_components
        self.tol = tol

    def fit(self, X, y):
        """
        Compute OSC weights and loadings from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Training spectral data
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target variable(s)

        Returns
        -------
        self : object
            Fitted transformer
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        y = _as_2d_target(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        max_components = max(0, min(n_samples - 1, self.n_features_in_))
        if self.n_components > max_components:
            warnings.warn(
                f"n_components={self.n_components} is greater than the maximum possible "
                f"({max_components}). Using maximum instead.",
                UserWarning,
            )
        n_wanted = max(0, min(int(self.n_components), max_components))

        self.X_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        Xd = X - self.X_mean_
        yc = y - self.y_mean_

        total_ss = float(np.sum(Xd**2))
        scale = np.sqrt(total_ss) if total_ss > 0 else 1.0

        weights: list[np.ndarray] = []
        loadings: list[np.ndarray] = []
        variance_removed: list[float] = []

        cov = Xd.T @ yc  # (n_features, n_targets); unchanged by OSC deflation
        U_c, s_c, _ = np.linalg.svd(cov, full_matrices=False)
        if s_c.size == 0 or s_c[0] <= 1e-12 * scale * max(1.0, float(np.linalg.norm(yc))):
            if n_wanted > 0:
                warnings.warn(
                    "y has no covariance with X (constant y or y orthogonal to X); "
                    "OSC cannot tell y-related from y-orthogonal variation and removes "
                    "nothing.",
                    UserWarning,
                )
            n_wanted = 0
        else:
            Q = U_c[:, s_c > 1e-12 * s_c[0]]

        for _ in range(n_wanted):
            Z = Xd - (Xd @ Q) @ Q.T
            _, S, Vt = np.linalg.svd(Z, full_matrices=False)
            if S.size == 0 or S[0] <= self.tol * scale:
                break
            w = Vt[0]
            # Exact orthogonality to Q (guards against round-off in the SVD).
            w = w - Q @ (Q.T @ w)
            w /= np.linalg.norm(w)
            t = Xd @ w
            tt = float(t @ t)
            if tt <= 0:
                break
            p = Xd.T @ t / tt
            Xd = Xd - np.outer(t, p)
            weights.append(w)
            loadings.append(p)
            variance_removed.append(tt * float(p @ p) / total_ss if total_ss > 0 else 0.0)

        if len(weights) < n_wanted:
            warnings.warn(
                f"OSC found only {len(weights)} y-orthogonal component(s) above tol; "
                f"{n_wanted} were requested.",
                UserWarning,
            )

        n_feat = self.n_features_in_
        self.weights_ = np.column_stack(weights) if weights else np.zeros((n_feat, 0))
        self.loadings_ = np.column_stack(loadings) if loadings else np.zeros((n_feat, 0))
        self.P_osc_ = self.weights_
        self.n_components_ = self.weights_.shape[1]
        self.variance_removed_ = np.array(variance_removed)
        self.fit_version_ = _OSC_FIT_VERSION
        return self

    def transform(self, X):
        """
        Remove the fitted y-orthogonal components from spectra.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to transform

        Returns
        -------
        X_osc : array, shape (n_samples, n_wavelengths)
            ``X - T P^T`` on the original scale, where the scores T are computed
            from X centred with the training mean.
        """
        if not hasattr(self, "fit_version_") and hasattr(self, "P_osc_"):
            return self._legacy_transform(X)
        check_is_fitted(self, ["weights_", "loadings_", "n_features_in_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OSC was fitted with {self.n_features_in_} features"
            )

        Xd = X - self.X_mean_
        for k in range(self.weights_.shape[1]):
            t = Xd @ self.weights_[:, k]
            Xd = Xd - np.outer(t, self.loadings_[:, k])
        return Xd + self.X_mean_

    def _legacy_transform(self, X):
        """Replay an OSC pickled by dasp before 2026-10 (exact old output).

        That version removed the first PLS loading (the y-PREDICTIVE direction,
        finding R025) and returned centred data. A model saved downstream of it was
        trained on exactly that output, so it is reproduced for prediction parity.
        """
        _warn_legacy("OSC")
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OSC was fitted with {self.n_features_in_} features"
            )
        if self.P_osc_.shape[1] == 0:
            return X
        X_centered = X - self.X_mean_
        for i in range(self.P_osc_.shape[1]):
            w_ortho = self.P_osc_[:, i:i + 1]
            X_centered = X_centered - (X_centered @ w_ortho) @ w_ortho.T
        return X_centered


# Placeholder classes for EPO, GLSW, DOSC (to be implemented in Phase 2)
class EPO(BaseEstimator, TransformerMixin):
    """
    External Parameter Orthogonalization (EPO).

    Removes specific interference (moisture, temperature, particle size) using
    a reference library of interferent spectra.

    TO BE IMPLEMENTED IN PHASE 2 (Day 6-8)

    Parameters
    ----------
    n_components : int, optional
        Number of PLS components for interferent model. If None, auto-select via CV.

    Examples
    --------
    >>> # Load interferent library (e.g., moisture spectra at 5%, 10%, 15%, 20%)
    >>> X_moisture = np.loadtxt('moisture_library.csv', delimiter=',')
    >>> epo = EPO(n_components=3)
    >>> epo.fit(X_train, y_train, X_interferents=X_moisture)
    >>> X_corrected = epo.transform(X_train)

    References
    ----------
    Roger et al. (2003). "EPO-PLS external parameter orthogonalisation of PLS
    application to temperature-independent measurement of sugar content of intact fruits."
    Chemometrics and Intelligent Laboratory Systems, 66(2), 191-204.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y, X_interferents=None):
        raise NotImplementedError("EPO will be implemented in Phase 2 (Day 6-8)")

    def transform(self, X):
        raise NotImplementedError("EPO will be implemented in Phase 2 (Day 6-8)")


class GLSW(BaseEstimator, TransformerMixin):
    """
    Generalized Least Squares Weighting (GLSW).

    Down-weights wavelength regions dominated by interference while preserving
    analyte information. Computes optimal weighting matrix from spectral covariance.
    This provides heteroscedastic variance weighting for improved regression.

    The method computes a diagonal weighting matrix W where wavelengths with high
    noise/interference receive lower weights, while informative wavelengths receive
    higher weights. This is particularly useful when different spectral regions have
    different noise levels (e.g., water absorption bands are noisier).

    Parameters
    ----------
    method : {'covariance', 'residual'}, default='covariance'
        Method for computing weight matrix:
        - 'covariance': Use inverse of spectral covariance (assumes noise ~ covariance)
        - 'residual': Use inverse of residual variance from PLS model (more sophisticated)

    regularization : float, default=1e-6
        Regularization parameter added to diagonal to avoid singularity.
        Increase if you get numerical instability warnings.

    n_components : int, optional
        Number of PLS components for 'residual' method. If None, uses min(10, n_samples-1).
        Only used when method='residual'.

    Attributes
    ----------
    W_ : array, shape (n_wavelengths, n_wavelengths)
        Diagonal weighting matrix (only diagonal elements stored for efficiency)

    n_features_in_ : int
        Number of wavelengths

    feature_weights_ : array, shape (n_wavelengths,)
        Weight for each wavelength (diagonal of W_)

    Examples
    --------
    >>> from spectral_predict.interference import GLSW
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> # Weight wavelengths by inverse noise variance
    >>> glsw = GLSW(method='covariance')
    >>> X_weighted = glsw.fit_transform(X_train)
    >>>
    >>> # Use in pipeline with Ridge regression
    >>> pipeline = Pipeline([
    ...     ('glsw', GLSW(method='covariance')),
    ...     ('ridge', Ridge(alpha=1.0))
    ... ])
    >>> pipeline.fit(X_train, y_train)

    References
    ----------
    Seasholtz, M. B., & Kowalski, B. R. (1993). "The parsimony principle applied
    to multivariate calibration." Analytica Chimica Acta, 277(2), 165-177.

    Brown, C. D. (2001). "Robust calibration with respect to background variation."
    Applied Spectroscopy, 55(5), 563-567.

    Notes
    -----
    GLSW is particularly effective when:
    - Different wavelengths have different noise levels (heteroscedastic noise)
    - Certain spectral regions have high interference (e.g., water bands)
    - You want to optimally weight information across the spectrum

    The transformation applies the square root of the weighting matrix:
    X_weighted = X @ sqrt(W)

    This is equivalent to weighted least squares: min ||W^(1/2) (Xβ - y)||²
    """

    def __init__(self, method='covariance', regularization=1e-6, n_components=None):
        self.method = method
        self.regularization = regularization
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Compute GLSW weighting matrix from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Training spectral data
        y : array-like, shape (n_samples,), optional
            Target variable. Required for method='residual', ignored for method='covariance'.

        Returns
        -------
        self : object
            Fitted transformer
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        if self.method == 'covariance':
            # Method 1: Weight by inverse of spectral covariance
            # Assumes variance in X is proportional to measurement noise

            # Compute covariance matrix (or just variances for diagonal approximation)
            # For computational efficiency, we use diagonal weighting (per-wavelength variance)
            variances = np.var(X, axis=0)

            # Add regularization to avoid division by zero
            variances = variances + self.regularization

            # Weights are inverse of variance (high variance → low weight)
            self.feature_weights_ = 1.0 / variances

            # Normalize weights to have mean=1 (preserves scale)
            self.feature_weights_ = self.feature_weights_ / np.mean(self.feature_weights_)

        elif self.method == 'residual':
            # Method 2: Weight by inverse of residual variance from PLS model
            # More sophisticated - weights based on prediction residuals

            if y is None:
                raise ValueError("GLSW with method='residual' requires y for fitting")

            y = check_array(y, accept_sparse=False, dtype=np.float64, ensure_2d=False)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")

            # Determine number of PLS components
            if self.n_components is None:
                n_comp = min(10, n_samples - 1, self.n_features_in_)
            else:
                n_comp = min(self.n_components, n_samples - 1, self.n_features_in_)

            # Build PLS model to get residuals
            from sklearn.cross_decomposition import PLSRegression
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, y)

            # Compute residuals for each wavelength
            # Back-project to get residuals in X-space
            X_pred = pls.predict(X)  # This is in Y-space
            # We want residuals in X-space per wavelength

            # Alternative: compute variance of X not explained by PLS
            X_scores = pls.transform(X)  # PLS scores
            X_reconstructed = X_scores @ pls.x_loadings_.T  # Reconstruct X from PLS
            residuals = X - X_reconstructed

            # Variance of residuals per wavelength
            residual_variances = np.var(residuals, axis=0)

            # Add regularization
            residual_variances = residual_variances + self.regularization

            # Weights are inverse of residual variance
            self.feature_weights_ = 1.0 / residual_variances

            # Normalize
            self.feature_weights_ = self.feature_weights_ / np.mean(self.feature_weights_)

        else:
            raise ValueError(f"method must be 'covariance' or 'residual', got '{self.method}'")

        # Store square root of weights for transformation
        # This is because we apply W^(1/2) to X
        self.W_sqrt_ = np.sqrt(self.feature_weights_)

        return self

    def transform(self, X):
        """
        Apply GLSW weighting to spectral data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to weight

        Returns
        -------
        X_weighted : array, shape (n_samples, n_wavelengths)
            Weighted spectral data
        """
        check_is_fitted(self, ['W_sqrt_', 'n_features_in_'])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but GLSW was fitted with {self.n_features_in_} features"
            )

        # Apply weighting: multiply each wavelength by its weight
        # This is equivalent to X @ diag(W_sqrt)
        X_weighted = X * self.W_sqrt_

        return X_weighted

    def get_feature_weights(self):
        """
        Get the weight assigned to each wavelength.

        Returns
        -------
        weights : array, shape (n_wavelengths,)
            Weight for each wavelength (higher = more important)
        """
        check_is_fitted(self, 'feature_weights_')
        return self.feature_weights_.copy()


class EPO(BaseEstimator, TransformerMixin):
    """
    External Parameter Orthogonalization (EPO).

    Removes specific interference (moisture, temperature, particle size) using
    a reference library of interferent spectra. EPO builds an interferent subspace
    from the reference library and projects data orthogonal to this subspace,
    removing interferent effects while preserving analyte signal.

    This is particularly useful when you have a library of pure interferent spectra
    (e.g., moisture at different levels, temperature variations) that you want to
    remove from your measurements.

    Parameters
    ----------
    n_components : int, default=2
        Number of interferent principal components to remove.
        Typically 1-5 components. Too many components risk removing analyte signal.

        WARNING: Start with 1-3 components and increase cautiously.

    center : bool, default=True
        Whether to subtract the training mean from X before projecting.
        - True: transform returns ``(X - X_mean_) @ P_orth_`` (centred features,
          suitable inside a modelling pipeline).
        - False: transform returns ``X @ P_orth_``, a spectrum on the original
          scale with the interferent removed.

    library_type : {'samples', 'differences'}, default='samples'
        What the rows of ``X_interferents`` are. EPO removes the subspace of the
        nuisance-difference matrix D (Roger et al. 2003), so D is built
        accordingly:
        - 'samples': whole spectra of the same material(s) measured at different
          interferent levels (e.g. one sample at several moisture contents). The
          rows also contain the analyte, so D is the rows minus their mean
          spectrum: the mean is the reference condition, and only the variation
          between rows (the interferent) is removed. A constant shift shared by
          every row cannot be told apart from the analyte and is kept.
        - 'differences': pure interferent spectra, or difference spectra
          (spectrum at a condition minus the same specimen at the reference
          condition). These contain no analyte, so D is the rows themselves,
          uncentred; centring them would cancel an interferent that every row
          shares (finding R024).

    svd_tol : float, default=1e-8
        Tolerance for SVD truncation. Singular values below this threshold
        are treated as zero to improve numerical stability.

    Attributes
    ----------
    n_features_in_ : int
        Number of features (wavelengths) in training data.

    n_components_ : int
        Actual number of components used (may differ from n_components
        if auto-reduced due to insufficient interferent samples).

    X_mean_ : ndarray, shape (n_features_in_,)
        Mean of training data (used for centering in transform).

    interferent_mean_ : ndarray, shape (n_features_in_,)
        Mean of interferent library (informational; not subtracted).

    P_orth_ : ndarray, shape (n_features_in_, n_features_in_)
        Orthogonal projection matrix for removing interferent signal.
        P_orth = I - V @ V.T, where V contains interferent principal components.

    interferent_components_ : ndarray, shape (n_features_in_, n_components_)
        Principal components of interferent subspace (V matrix from SVD).

    explained_variance_ : ndarray, shape (n_components_,)
        Amount of interferent variance explained by each component.

    Examples
    --------
    Basic usage with moisture interferent library:

    >>> from spectral_predict.interference import EPO
    >>> import numpy as np
    >>> # Simulated data: 100 samples, 50 wavelengths
    >>> X_train = np.random.randn(100, 50)
    >>> y_train = np.random.randn(100)
    >>> # Interferent library: 10 moisture spectra at different levels
    >>> X_moisture = np.random.randn(10, 50)
    >>> epo = EPO(n_components=2)
    >>> epo.fit(X_train, y_train, X_interferents=X_moisture)
    >>> X_corrected = epo.transform(X_train)

    Pipeline integration:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> pipeline = Pipeline([
    ...     ('epo', EPO(n_components=2)),
    ...     ('pls', PLSRegression(n_components=10))
    ... ])
    >>> # Note: X_interferents must be passed to fit
    >>> pipeline.fit(X_train, y_train, epo__X_interferents=X_moisture)

    References
    ----------
    Roger et al. (2003). "EPO-PLS external parameter orthogonalisation of PLS
    application to temperature-independent measurement of sugar content of intact fruits."
    Chemometrics and Intelligent Laboratory Systems, 66(2), 191-204.
    """

    def __init__(self, n_components=2, center=True, svd_tol=1e-8, library_type='samples'):
        self.n_components = n_components
        self.center = center
        self.svd_tol = svd_tol
        self.library_type = library_type

    def __setstate__(self, state):
        """Give EPO objects pickled before 2026-10 the ``library_type`` they lack.

        The old code centred the library exactly when ``center`` was True, so that
        maps to 'samples' (centred) or 'differences' (uncentred); a refit of an old
        object therefore builds the same kind of library it was fitted with. The
        fitted projection itself is not touched.
        """
        super().__setstate__(state)
        if not hasattr(self, "library_type"):
            self.library_type = 'samples' if getattr(self, "center", True) else 'differences'

    def fit(self, X, y=None, X_interferents=None):
        """
        Fit EPO transformer using interferent reference library.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Training spectral data (used only for validation and centering)

        y : Ignored
            Not used, present for sklearn API compatibility

        X_interferents : array-like, shape (n_interferent_samples, n_wavelengths)
            Reference library of interferent spectra (e.g., moisture at different levels).
            REQUIRED - EPO cannot function without this.

            Example: If measuring plant nitrogen but moisture interferes, provide
            spectra of samples with varying moisture content (library_type='samples'),
            or moisture difference spectra (library_type='differences').

        Returns
        -------
        self : object
            Fitted transformer
        """
        # Validate X
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Validate n_components parameter
        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError(
                f"n_components must be an integer, got {type(self.n_components).__name__}"
            )

        if self.n_components <= 0:
            raise ValueError(
                f"n_components must be a positive integer, got {self.n_components}"
            )

        # Validate center parameter
        if not isinstance(self.center, bool):
            raise TypeError(
                f"center must be True or False, got {type(self.center).__name__}"
            )

        # Validate svd_tol parameter
        if not isinstance(self.svd_tol, (int, float, np.number)):
            raise TypeError(
                f"svd_tol must be a number, got {type(self.svd_tol).__name__}"
            )

        if self.svd_tol < 0:
            raise ValueError(
                f"svd_tol must be non-negative, got {self.svd_tol}"
            )

        if self.library_type not in ('samples', 'differences'):
            raise ValueError(
                f"library_type must be 'samples' or 'differences', got {self.library_type!r}"
            )

        # ✅ CRITICAL FIX #1: Validate X_interferents is provided
        if X_interferents is None:
            raise ValueError(
                "X_interferents is required for EPO. "
                "Provide a reference library of interferent spectra. "
                "Example: epo.fit(X_train, y_train, X_interferents=moisture_library)"
            )

        # Validate X_interferents
        X_interferents = check_array(X_interferents, accept_sparse=False, dtype=np.float64)

        # ✅ CRITICAL FIX #1: Validate feature dimensions match
        if X.shape[1] != X_interferents.shape[1]:
            raise ValueError(
                f"X and X_interferents must have same number of features (wavelengths). "
                f"Got X: {X.shape[1]}, X_interferents: {X_interferents.shape[1]}"
            )

        # ✅ CRITICAL FIX #2: Validate interferent library has sufficient samples
        n_interferent_samples = X_interferents.shape[0]

        if n_interferent_samples == 0:
            raise ValueError(
                "X_interferents is empty (0 samples). "
                "Provide at least one interferent spectrum."
            )

        # `center` only affects X in transform. The nuisance matrix D depends on
        # what the library rows are (see library_type).
        if self.center:
            self.X_mean_ = np.mean(X, axis=0)
        else:
            self.X_mean_ = np.zeros(self.n_features_in_)
        self.interferent_mean_ = np.mean(X_interferents, axis=0)

        if self.library_type == 'samples':
            # Whole spectra (analyte + interferent): differences from the library
            # mean isolate the interferent variation. Using the rows uncentred
            # would remove the shared analyte spectrum instead.
            nuisance = X_interferents - self.interferent_mean_
            per_wavelength = np.std(nuisance, axis=0)
            if np.all(per_wavelength < 1e-12):
                raise ValueError(
                    "X_interferents has near-zero variance across all wavelengths, so a "
                    "library of whole sample spectra holds no interferent variation. If "
                    "the rows are pure interferent or difference spectra, use "
                    "library_type='differences'."
                )
            n_flat = int(np.sum(per_wavelength < 1e-12))
            flat_msg = "have zero variance in the interferent library"
        else:
            # Pure interferent or difference spectra: used as they are, uncentred.
            nuisance = X_interferents
            per_wavelength = np.max(np.abs(X_interferents), axis=0)
            if np.all(per_wavelength < 1e-12):
                raise ValueError(
                    "X_interferents is (near) zero at every wavelength. "
                    "Cannot build interferent subspace from empty spectra."
                )
            n_flat = int(np.sum(per_wavelength < 1e-12))
            flat_msg = "are zero in every interferent spectrum"

        if n_flat > 0:
            warnings.warn(
                f"{n_flat}/{X_interferents.shape[1]} wavelengths {flat_msg}. These "
                f"wavelengths will not contribute to interferent subspace.",
                UserWarning
            )

        # ✅ CRITICAL FIX #2: Reduce n_components if insufficient samples
        if n_interferent_samples < self.n_components:
            warnings.warn(
                f"X_interferents has only {n_interferent_samples} samples but "
                f"n_components={self.n_components}. Reducing to {n_interferent_samples} components.",
                UserWarning
            )
            effective_components = n_interferent_samples
        else:
            effective_components = self.n_components

        # Recommended warning for excessive components
        max_reasonable = min(10, n_interferent_samples - 1)
        if effective_components > max_reasonable:
            warnings.warn(
                f"n_components={effective_components} is very large. "
                f"Risk of removing analyte signal! Consider using ≤ {max_reasonable} components.",
                UserWarning
            )

        self.n_components_ = effective_components

        # Cap n_components at maximum mathematically possible
        max_possible = min(n_interferent_samples, self.n_features_in_)
        if self.n_components_ > max_possible:
            warnings.warn(
                f"n_components={self.n_components_} exceeds maximum possible ({max_possible}). "
                f"Reducing to {max_possible}.",
                UserWarning
            )
            self.n_components_ = max_possible

        # Build interferent subspace using SVD
        # SVD: X_interferents = U @ S @ Vt
        # We want the first n_components_ right singular vectors (rows of Vt)
        try:
            U, S, Vt = np.linalg.svd(nuisance, full_matrices=False)
        except np.linalg.LinAlgError:
            raise ValueError(
                "SVD failed on interferent library. This may indicate numerical issues. "
                "Check for NaN/Inf values or extreme outliers in X_interferents."
            )

        # Keep only directions the library actually spans. Without this, a library
        # of rank r < n_components contributed arbitrary null-space vectors, and
        # projecting those out removed random (possibly analyte) directions.
        cutoff = max(self.svd_tol, S[0] * max(nuisance.shape) * np.finfo(np.float64).eps)
        n_valid = int(np.sum(S > cutoff))
        if n_valid < self.n_components_:
            warnings.warn(
                f"Interferent library spans only {n_valid} direction(s); "
                f"using {n_valid} instead of {self.n_components_} components.",
                UserWarning
            )
            self.n_components_ = n_valid

        # Get interferent principal components (first n_components_ columns of V)
        # Note: Vt is (n_components, n_features), we want V = Vt.T
        V = Vt.T  # Shape: (n_features, n_components)
        self.interferent_components_ = V[:, :self.n_components_]

        # Store explained variance
        total_variance = np.sum(S ** 2)
        if total_variance > 0:
            self.explained_variance_ = (S[:self.n_components_] ** 2) / total_variance
        else:
            self.explained_variance_ = np.zeros(self.n_components_)

        # Build orthogonal projection matrix
        # P_orth = I - V @ V.T
        # This projects data orthogonal to the interferent subspace
        V_comp = self.interferent_components_
        self.P_orth_ = np.eye(self.n_features_in_) - V_comp @ V_comp.T

        return self

    def transform(self, X):
        """
        Apply EPO transformation to remove interferent signal.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to transform

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_wavelengths)
            ``(X - X_mean_) @ P_orth_``. With ``center=True`` (default) X_mean_
            is the training mean, so the output is centred features; with
            ``center=False`` X_mean_ is zero and the output is a spectrum on the
            original scale.
        """
        check_is_fitted(self, ['P_orth_', 'X_mean_'])

        # Validate X
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # ✅ CRITICAL FIX: Validate feature count matches training
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but EPO was fitted with "
                f"{self.n_features_in_} features."
            )

        # Center using TRAINING mean (prevent data leakage)
        X_centered = X - self.X_mean_

        # Apply orthogonal projection to remove interferent signal
        X_corrected = X_centered @ self.P_orth_

        return X_corrected

    def get_interferent_components(self):
        """
        Get the interferent principal components.

        Returns
        -------
        components : ndarray, shape (n_wavelengths, n_components_)
            Interferent subspace basis vectors
        """
        check_is_fitted(self, 'interferent_components_')
        return self.interferent_components_.copy()

    def get_explained_variance(self):
        """
        Get the variance explained by each interferent component.

        Returns
        -------
        explained_variance : ndarray, shape (n_components_,)
            Fraction of interferent variance explained by each component
        """
        check_is_fitted(self, 'explained_variance_')
        return self.explained_variance_.copy()


class DOSC(BaseEstimator, TransformerMixin):
    """
    Direct Orthogonal Signal Correction (DOSC).

    Non-iterative OSC (Westerhuis, de Jong & Smilde 2001). y is first replaced by
    its least-squares projection Y_hat onto the column space of X; X is deflated
    by Y_hat in sample space; the leading principal-component scores T of that
    residual are the removed scores. T lies in the column space of X and is
    orthogonal to Y_hat, so ``T^T y = 0``. Weights ``W = X^+ T`` replay the
    scores on new spectra (``T_new = (X_new - X_mean_) W``), and loadings
    ``P = X^T T (T^T T)^-1`` give ``X_corrected = X - T P^T``.

    When there are more wavelengths than samples, ``X^+`` is the minimum-norm
    pseudo-inverse and W can amplify noise on new spectra (a known property of
    DOSC). Check held-out predictions before relying on it.

    Parameters
    ----------
    n_components : int, default=1
        Number of Y-orthogonal components to remove.
        Typically 1-3 components. Too many components can remove Y-related signal.

    center : bool, default=True
        Whether to mean-center X and y before DOSC.
        Recommended to keep True for most applications.

    n_pls_components : int or 'auto', default='auto'
        Ignored. The earlier implementation used a PLS model here, and its removed
        scores were not orthogonal to y. The parameter is still validated and
        accepted so saved settings and pipelines that pass it keep constructing.

    Attributes
    ----------
    n_features_in_ : int
        Number of features (wavelengths) seen during fit.

    n_components_ : int
        Actual number of components used (may be less than n_components
        if insufficient samples or features).

    X_mean_ : ndarray, shape (n_features_in_,)
        Mean of X (used for centering during transform).

    y_mean_ : ndarray, shape (n_targets,)
        Mean of y (stored for reference).

    weights_ : ndarray, shape (n_features_in_, n_components_)
        W, maps centred spectra to removed scores.

    loadings_ : ndarray, shape (n_features_in_, n_components_)
        P, the spectral shapes that are subtracted.

    P_orth_ : ndarray, shape (n_features_in_, n_features_in_)
        Linear part of the correction, ``I - W P^T`` (not an orthogonal projector).

    dosc_components_ : ndarray, shape (n_features_in_, n_components_)
        Unit-norm loading directions of the removed components.

    explained_variance_ : ndarray, shape (n_components_,)
        Variance in X explained by each Y-orthogonal component.

    Examples
    --------
    Basic usage:

    >>> from spectral_predict.interference import DOSC
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> dosc = DOSC(n_components=2)
    >>> dosc.fit(X, y)
    >>> X_corrected = dosc.transform(X)

    Pipeline integration:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> pipeline = Pipeline([
    ...     ('dosc', DOSC(n_components=2)),
    ...     ('pls', PLSRegression(n_components=10))
    ... ])
    >>> pipeline.fit(X, y)

    References
    ----------
    Westerhuis, J. A., de Jong, S., & Smilde, A. K. (2001).
    "Direct orthogonal signal correction."
    Chemometrics and Intelligent Laboratory Systems, 56(1), 13-25.
    """

    def __init__(self, n_components=1, center=True, n_pls_components='auto'):
        self.n_components = n_components
        self.center = center
        self.n_pls_components = n_pls_components

    def fit(self, X, y):
        """
        Fit DOSC transformer to find Y-orthogonal variation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training spectral data

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values. Required for DOSC (unlike unsupervised methods).

        Returns
        -------
        self : object
            Fitted transformer
        """
        # Validate inputs
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Validate n_components
        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError(
                f"n_components must be an integer, got {type(self.n_components).__name__}"
            )

        if self.n_components <= 0:
            raise ValueError(
                f"n_components must be a positive integer, got {self.n_components}"
            )

        # Validate center parameter
        if not isinstance(self.center, bool):
            raise TypeError(
                f"center must be True or False, got {type(self.center).__name__}"
            )

        # Validate n_pls_components parameter
        if isinstance(self.n_pls_components, str):
            if self.n_pls_components != 'auto':
                raise ValueError(
                    f"n_pls_components must be 'auto' or an integer, got '{self.n_pls_components}'"
                )
        elif isinstance(self.n_pls_components, (int, np.integer)):
            if self.n_pls_components <= 0:
                raise ValueError(
                    f"n_pls_components must be positive, got {self.n_pls_components}"
                )
        else:
            raise TypeError(
                f"n_pls_components must be 'auto' or an integer, got {type(self.n_pls_components).__name__}"
            )

        # Center data
        if self.center:
            self.X_mean_ = np.mean(X, axis=0)
            self.y_mean_ = np.mean(y, axis=0)
            X_centered = X - self.X_mean_
            y_centered = y - self.y_mean_
        else:
            self.X_mean_ = np.zeros(n_features)
            self.y_mean_ = np.zeros(y.shape[1])
            X_centered = X.copy()
            y_centered = y.copy()

        # Determine effective number of components
        max_components = min(n_samples - 1, n_features)
        if self.n_components > max_components:
            warnings.warn(
                f"n_components={self.n_components} exceeds maximum ({max_components}). "
                f"Reducing to {max_components}.",
                UserWarning
            )
            effective_components = max_components
        else:
            effective_components = self.n_components

        # DOSC, Westerhuis, de Jong & Smilde (2001):
        # 1. Y_hat: least-squares projection of y onto the column space of X.
        # 2. A_y = X deflated by Y_hat (sample space): (I - P_Yhat) X.
        # 3. T = leading principal-component scores of A_y. T lies in col(X) and is
        #    orthogonal to Y_hat, hence orthogonal to y itself (y - Y_hat is
        #    orthogonal to col(X)).
        # 4. Weights W = X^+ T, so new spectra give scores X_new W; loadings
        #    P = X^T T (T^T T)^-1; X_corrected = X - T P^T.
        # The previous implementation projected X onto principal directions of the
        # PLS X-residual. Those directions do not give scores orthogonal to y.
        coef, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
        y_hat = X_centered @ coef
        y_scale = max(1.0, float(np.abs(y_centered).max()))
        if not np.any(np.abs(y_hat) > 1e-12 * y_scale):
            warnings.warn(
                "y has no projection on X (constant y or y orthogonal to X); DOSC "
                "cannot tell y-related from y-orthogonal variation and removes nothing.",
                UserWarning,
            )
            effective_components = 0
        A_y = X_centered - y_hat @ (np.linalg.pinv(y_hat) @ X_centered)

        try:
            U, S, _ = np.linalg.svd(A_y, full_matrices=False)
        except np.linalg.LinAlgError:
            raise ValueError(
                "SVD failed on Y-orthogonal residuals. Check for NaN/Inf in data."
            )

        total_ss = float(np.sum(X_centered**2))
        scale = np.sqrt(total_ss) if total_ss > 0 else 1.0
        n_valid = int(np.sum(S > 1e-10 * scale))
        if n_valid < effective_components:
            warnings.warn(
                f"Only {n_valid} y-orthogonal component(s) found; using {n_valid} "
                f"instead of {effective_components}.",
                UserWarning,
            )
            effective_components = n_valid
        self.n_components_ = effective_components

        T = U[:, :effective_components] * S[:effective_components]
        if effective_components > 0:
            # Explicit cutoff: centring leaves one ~1e-14 singular value, and the
            # library default keeps and inverts it, which breaks T = X W (and with
            # it the orthogonality of the replayed scores to y) at the 1e-4 level.
            rcond = max(X_centered.shape) * np.finfo(np.float64).eps
            W = np.linalg.pinv(X_centered, rcond=rcond) @ T
            T = X_centered @ W  # identical up to round-off; keeps fit == transform
            P = X_centered.T @ T @ np.linalg.pinv(T.T @ T, rcond=rcond)
        else:
            W = np.zeros((n_features, 0))
            P = np.zeros((n_features, 0))

        self.weights_ = W
        self.loadings_ = P
        # Unit-norm loading directions, for inspection and influence plots.
        norms = np.linalg.norm(P, axis=0)
        norms[norms == 0] = 1.0
        self.dosc_components_ = P / norms

        # Share of the X sum of squares removed by each component.
        if total_ss > 0 and effective_components > 0:
            self.explained_variance_ = (
                np.sum(T**2, axis=0) * np.sum(P**2, axis=0) / total_ss
            )
        else:
            self.explained_variance_ = np.zeros(effective_components)

        # Linear part of the correction: x_corrected = x - (x - mean) W P^T.
        self.P_orth_ = np.eye(n_features) - W @ P.T
        self.fit_version_ = _DOSC_FIT_VERSION

        return self

    def transform(self, X):
        """
        Apply DOSC transformation to remove Y-orthogonal variation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Spectral data to transform

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_features)
            ``X - T P^T`` on the original scale, with scores
            ``T = (X - X_mean_) W``. The training mean is not subtracted.
        """
        legacy = not hasattr(self, "fit_version_") and hasattr(self, "P_orth_")
        if not legacy:
            check_is_fitted(self, ['weights_', 'loadings_', 'X_mean_'])

        # Validate X
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Check feature count
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but DOSC was fitted with "
                f"{self.n_features_in_} features."
            )

        if legacy:
            # Pickled before 2026-10: replay the old (X - X_mean_) @ P_orth_ exactly.
            _warn_legacy("DOSC")
            return (X - self.X_mean_) @ self.P_orth_

        T = (X - self.X_mean_) @ self.weights_
        return X - T @ self.loadings_.T

    def get_dosc_components(self):
        """
        Get the Y-orthogonal components.

        Returns
        -------
        components : ndarray, shape (n_features_in_, n_components_)
            Y-orthogonal subspace basis vectors
        """
        check_is_fitted(self, 'dosc_components_')
        return self.dosc_components_.copy()

    def get_explained_variance(self):
        """
        Get the variance explained by each Y-orthogonal component.

        Returns
        -------
        explained_variance : ndarray, shape (n_components_,)
            Fraction of Y-orthogonal variance explained by each component
        """
        check_is_fitted(self, 'explained_variance_')
        return self.explained_variance_.copy()
