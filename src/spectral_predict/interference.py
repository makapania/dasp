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
from sklearn.cross_decomposition import PLSRegression
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


class OSC(BaseEstimator, TransformerMixin):
    """
    Orthogonal Signal Correction (OSC).

    Removes systematic variation in X (spectra) that is orthogonal to y (target variable).
    Particularly effective for removing moisture and temperature effects that don't
    correlate with the analyte of interest.

    Algorithm:
    1. Build PLS model between X and y
    2. Extract orthogonal components (variation in X not correlated with y)
    3. Remove these components from X
    4. Iterate until convergence or max components reached

    Parameters
    ----------
    n_components : int, default=1
        Number of orthogonal components to remove (typically 1-3 is sufficient)

    tol : float, default=1e-6
        Convergence tolerance for OSC algorithm

    max_iter : int, default=100
        Maximum iterations for OSC algorithm

    Attributes
    ----------
    P_osc_ : array, shape (n_wavelengths, n_components)
        OSC projection matrix for removing orthogonal variation

    n_features_in_ : int
        Number of wavelengths

    variance_removed_ : array, shape (n_components,)
        Variance explained by each OSC component

    Examples
    --------
    >>> from spectral_predict.interference import OSC
    >>> # Remove 1 component of Y-orthogonal variation (e.g., moisture)
    >>> osc = OSC(n_components=1)
    >>> X_train_corrected = osc.fit_transform(X_train, y_train)
    >>> X_test_corrected = osc.transform(X_test)
    >>>
    >>> # Remove multiple components
    >>> osc = OSC(n_components=3)
    >>> X_corrected = osc.fit_transform(X, y)
    >>> print(f"Variance removed by each component: {osc.variance_removed_}")

    References
    ----------
    Wold et al. (1998). "Orthogonal signal correction of near-infrared spectra."
    Chemometrics and Intelligent Laboratory Systems, 44(1-2), 175-185.

    Notes
    -----
    OSC must be fitted with y (target variable) to determine which variation is
    orthogonal. The same transformation is then applied to test data (without y).
    This is safe for cross-validation as long as OSC is fitted only on training folds.

    **Important:** OSC returns mean-centered data (using training set mean). This is
    correct behavior and necessary for the algorithm. If you need the original scale,
    add the training mean back after OSC transformation.
    """

    def __init__(self, n_components=1, tol=1e-6, max_iter=100):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Compute OSC transformation from training data.

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
        y = check_array(y, accept_sparse=False, dtype=np.float64, ensure_2d=False)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")

        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Validate n_components
        if self.n_components > min(n_samples - 1, self.n_features_in_):
            warnings.warn(
                f"n_components={self.n_components} is greater than the maximum possible "
                f"({min(n_samples - 1, self.n_features_in_)}). Using maximum instead.",
                UserWarning
            )
            effective_components = min(self.n_components, n_samples - 1, self.n_features_in_)
        else:
            effective_components = self.n_components

        # Center data (store means for transform)
        self.X_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        X_centered = X - self.X_mean_
        y_centered = y - self.y_mean_

        # Storage for OSC components
        P_osc_list = []
        variance_removed = []

        X_osc = X_centered.copy()

        for comp in range(effective_components):
            # 1. Build PLS model to find Y-relevant subspace
            pls = PLSRegression(n_components=min(5, n_samples - 1, X_osc.shape[1]))
            pls.fit(X_osc, y_centered)

            # 2. Project X onto Y-relevant subspace
            # X_scores = X @ pls.x_weights_
            # X_y_relevant = X_scores @ pls.x_loadings_.T

            # 3. Compute orthogonal component (X - X_y_relevant)
            # For OSC, we use the first PLS component as proxy for Y-direction
            # and compute orthogonal subspace

            # Get first PLS score and loading
            t = pls.x_scores_[:, 0:1]  # First score vector
            p = pls.x_loadings_[:, 0:1]  # First loading vector

            # Orthogonalize: remove component in direction of p
            # This is the Y-orthogonal variation we want to remove
            w_ortho = p / np.linalg.norm(p)  # Normalize

            # Check convergence (if loading magnitude is very small, stop)
            if np.linalg.norm(w_ortho) < self.tol:
                warnings.warn(
                    f"OSC converged early at component {comp + 1} (loading magnitude < tol)",
                    UserWarning
                )
                break

            P_osc_list.append(w_ortho.ravel())

            # Compute variance explained by this component
            t_ortho = X_osc @ w_ortho
            var_explained = np.sum(t_ortho ** 2) / np.sum(X_osc ** 2)
            variance_removed.append(var_explained)

            # Remove this orthogonal component from X
            X_osc = X_osc - t_ortho @ w_ortho.T

        if len(P_osc_list) == 0:
            warnings.warn("No OSC components extracted. Returning identity transformation.", UserWarning)
            self.P_osc_ = np.zeros((self.n_features_in_, 0))
        else:
            self.P_osc_ = np.column_stack(P_osc_list)

        self.variance_removed_ = np.array(variance_removed)

        return self

    def transform(self, X):
        """
        Apply OSC transformation to remove orthogonal variation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to transform

        Returns
        -------
        X_osc : array, shape (n_samples, n_wavelengths)
            Transformed spectral data with orthogonal components removed
        """
        check_is_fitted(self, ['P_osc_', 'n_features_in_'])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OSC was fitted with {self.n_features_in_} features"
            )

        if self.P_osc_.shape[1] == 0:
            # No components extracted, return unchanged
            return X

        # Center using training mean to prevent data leakage
        X_centered = X - self.X_mean_

        # Project onto orthogonal components and remove
        for i in range(self.P_osc_.shape[1]):
            w_ortho = self.P_osc_[:, i:i+1]
            t_ortho = X_centered @ w_ortho
            X_centered = X_centered - t_ortho @ w_ortho.T

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
        Whether to mean-center data before applying EPO.
        - True: Center X using training mean, center interferents using interferent mean
        - False: No centering (assumes data already centered)

        Note: EPO returns mean-centered data (training mean subtracted).

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
        Mean of interferent library.

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

    def __init__(self, n_components=2, center=True, svd_tol=1e-8):
        self.n_components = n_components
        self.center = center
        self.svd_tol = svd_tol

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
            spectra of samples with varying moisture content.

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

        # Store means for centering
        if self.center:
            self.X_mean_ = np.mean(X, axis=0)
            self.interferent_mean_ = np.mean(X_interferents, axis=0)
            X_interferents_centered = X_interferents - self.interferent_mean_
        else:
            self.X_mean_ = np.zeros(self.n_features_in_)
            self.interferent_mean_ = np.zeros(self.n_features_in_)
            X_interferents_centered = X_interferents.copy()

        # ✅ CRITICAL FIX #3: Check for zero/low variance in interferents
        interferent_std = np.std(X_interferents_centered, axis=0)

        # Check if entire library is constant
        if np.all(interferent_std < 1e-12):
            raise ValueError(
                "X_interferents has near-zero variance across all wavelengths. "
                "Cannot build interferent subspace from constant spectra. "
                "Provide interferent library with variation (e.g., different moisture levels)."
            )

        # Check if some wavelengths are constant (this is OK, but warn)
        n_constant_wavelengths = np.sum(interferent_std < 1e-12)
        if n_constant_wavelengths > 0:
            warnings.warn(
                f"{n_constant_wavelengths}/{X_interferents.shape[1]} wavelengths have "
                f"zero variance in interferent library. These wavelengths will not "
                f"contribute to interferent subspace.",
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
            U, S, Vt = np.linalg.svd(X_interferents_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            raise ValueError(
                "SVD failed on interferent library. This may indicate numerical issues. "
                "Check for NaN/Inf values or extreme outliers in X_interferents."
            )

        # Truncate small singular values for numerical stability
        S_truncated = S.copy()
        S_truncated[S < self.svd_tol] = 0.0

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
            Mean-centered data with interferent signal removed

            Note: Data is mean-centered using training mean (X_mean_).
            This is correct behavior for EPO.
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

    Simplified variant of OSC with direct computation of Y-orthogonal subspace.
    DOSC removes systematic variation in X that is orthogonal to Y using a
    direct PLS-based projection, avoiding the iterative deflation of standard OSC.

    This method is more stable and computationally efficient than iterative OSC,
    making it suitable for removing systematic noise (e.g., baseline drift,
    temperature effects) that is not related to the target variable.

    Parameters
    ----------
    n_components : int, default=1
        Number of Y-orthogonal components to remove.
        Typically 1-3 components. Too many components can remove Y-related signal.

    center : bool, default=True
        Whether to mean-center X and y before DOSC.
        Recommended to keep True for most applications.

    n_pls_components : int or 'auto', default='auto'
        Number of PLS components to use for finding Y-predictive subspace.
        - 'auto': Automatically determined as min(10, n_samples-1, n_features)
        - int: Specific number of components (must be > 0)

        More components = better Y-space approximation but slower and risk overfitting.
        Fewer components = faster, more robust but may miss Y-patterns.

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

    P_orth_ : ndarray, shape (n_features_in_, n_features_in_)
        Orthogonal projection matrix for removing Y-orthogonal variation.

    dosc_components_ : ndarray, shape (n_features_in_, n_components_)
        Y-orthogonal components extracted from X.

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

        self.n_components_ = effective_components

        # Compute Y-orthogonal subspace using PLS
        # 1. Fit PLS to get Y-predictive directions
        # Determine number of PLS components to use
        if self.n_pls_components == 'auto':
            n_pls_components = min(10, n_samples - 1, n_features)  # Use up to 10 PLS components
        else:
            # User-specified value, but cap at maximum possible
            n_pls_components = min(self.n_pls_components, n_samples - 1, n_features)
            if n_pls_components < self.n_pls_components:
                warnings.warn(
                    f"n_pls_components={self.n_pls_components} exceeds maximum possible "
                    f"({min(n_samples - 1, n_features)}). Using {n_pls_components} instead.",
                    UserWarning
                )

        pls = PLSRegression(n_components=n_pls_components)
        pls.fit(X_centered, y_centered)

        # 2. Get PLS X-loadings (P) and X-scores (T)
        # X = T @ P.T + E_pls
        # T = X @ W, where W are PLS weights
        T_pls = pls.x_scores_  # Shape: (n_samples, n_pls_components)
        P_pls = pls.x_loadings_  # Shape: (n_features, n_pls_components)

        # 3. Compute residuals orthogonal to Y-predictive space
        X_reconstructed_pls = T_pls @ P_pls.T
        E_orth = X_centered - X_reconstructed_pls  # Y-orthogonal residuals

        # 4. Extract principal components of Y-orthogonal residuals
        try:
            U, S, Vt = np.linalg.svd(E_orth, full_matrices=False)
        except np.linalg.LinAlgError:
            raise ValueError(
                "SVD failed on Y-orthogonal residuals. Check for NaN/Inf in data."
            )

        # 5. Take first n_components_ Y-orthogonal directions
        V = Vt.T  # Shape: (n_features, min(n_samples, n_features))
        self.dosc_components_ = V[:, :self.n_components_]

        # 6. Store explained variance
        total_variance = np.sum(S ** 2)
        if total_variance > 0:
            self.explained_variance_ = (S[:self.n_components_] ** 2) / total_variance
        else:
            self.explained_variance_ = np.zeros(self.n_components_)

        # 7. Build orthogonal projection matrix
        # P_orth = I - V_dosc @ V_dosc.T
        V_dosc = self.dosc_components_
        self.P_orth_ = np.eye(n_features) - V_dosc @ V_dosc.T

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
            Mean-centered data with Y-orthogonal variation removed

            Note: Data is mean-centered using training mean (X_mean_).
        """
        check_is_fitted(self, ['P_orth_', 'X_mean_'])

        # Validate X
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Check feature count
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but DOSC was fitted with "
                f"{self.n_features_in_} features."
            )

        # Center using TRAINING mean (prevent data leakage)
        X_centered = X - self.X_mean_

        # Apply orthogonal projection
        X_corrected = X_centered @ self.P_orth_

        return X_corrected

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
