"""
Contaminant-aware spectral region analysis for spectral data.

This module provides methods for identifying and removing contaminant influence from
spectral data when you have contaminated and uncontaminated sample groups but cannot
measure pure contaminant spectra (e.g., due to probe geometry constraints).

Methods implemented:
- DifferenceAnalyzer: Calculate and visualize difference spectra between groups
- EstimatedEPO: EPO using estimated interferent library from group differences
- ContaminantOPLSDA: OPLS-DA for identifying contaminant-influenced wavelengths
- ContaminantGLSW: Contaminant-aware GLSW weighting
- RegionExcluder: Backward iPLS-based region exclusion

Use Case:
---------
Bone samples with and without consolidants (like Glyptal). User wants to predict
% collagen accurately but cannot scan pure contaminant due to probe geometry.

Literature References:
---------------------
EPO-PLS:
    Roger et al. (2003). "EPO-PLS external parameter orthogonalisation of PLS
    application to temperature-independent measurement of sugar content of intact fruits."
    Chemometrics and Intelligent Laboratory Systems, 66(2), 191-204.

Interval PLS:
    Norgaard et al. (2000). "Interval partial least-squares regression (iPLS):
    A comparative chemometric study."
    Applied Spectroscopy 54(3): 413-419.

OPLS-DA:
    Bylesjo et al. (2006). "OPLS discriminant analysis: combining the strengths
    of PLS-DA and SIMCA classification."
    Journal of Chemometrics, 20(8-10), 341-351.

Usage Examples:
--------------
Difference spectrum analysis (exploratory):
    >>> from spectral_predict.contaminant_analysis import DifferenceAnalyzer
    >>> analyzer = DifferenceAnalyzer()
    >>> analyzer.fit(X_contaminated, X_uncontaminated)
    >>> diff_spectrum = analyzer.get_difference_spectrum()
    >>> peak_regions = analyzer.identify_peak_regions(threshold=0.5)

Estimated EPO (no Y values needed):
    >>> from spectral_predict.contaminant_analysis import EstimatedEPO
    >>> epo = EstimatedEPO(n_components=3)
    >>> epo.fit_groups(X_contaminated, X_uncontaminated)
    >>> X_all_corrected = epo.transform(np.vstack([X_contaminated, X_uncontaminated]))

OPLS-DA region identification:
    >>> from spectral_predict.contaminant_analysis import ContaminantOPLSDA
    >>> oplsda = ContaminantOPLSDA()
    >>> oplsda.fit(X_contaminated, X_uncontaminated)
    >>> wavelength_influence = oplsda.get_wavelength_influence()
    >>> exclusion_regions = oplsda.get_exclusion_regions(wavelengths, threshold=0.7)

Backward iPLS region exclusion (requires Y):
    >>> from spectral_predict.contaminant_analysis import RegionExcluder
    >>> excluder = RegionExcluder(n_intervals=20)
    >>> excluder.fit(X_uncontaminated, y_uncontaminated, wavelengths)
    >>> X_optimized = excluder.transform(X_contaminated)
"""

from __future__ import annotations

import numpy as np
import warnings
from typing import Optional, Tuple, List, Dict, Any, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score


class DifferenceAnalyzer(BaseEstimator, TransformerMixin):
    """
    Analyze difference spectra between contaminated and uncontaminated groups.

    This is an exploratory tool that calculates and visualizes the difference
    between contaminated and uncontaminated sample groups. The difference spectrum
    reveals where the contaminant has spectral influence.

    No Y values (target variable) are required - this is purely unsupervised.

    Parameters
    ----------
    normalize : bool, default=True
        Whether to normalize spectra before computing difference.
        - True: Use SNV normalization (remove scale differences)
        - False: Use raw spectra

    method : {'mean', 'median', 'pca'}, default='mean'
        Method for computing group representative spectrum:
        - 'mean': Use mean spectrum of each group
        - 'median': Use median spectrum (more robust to outliers)
        - 'pca': Use first principal component direction

    Attributes
    ----------
    contaminated_representative_ : ndarray, shape (n_wavelengths,)
        Representative spectrum of contaminated group

    uncontaminated_representative_ : ndarray, shape (n_wavelengths,)
        Representative spectrum of uncontaminated group

    difference_spectrum_ : ndarray, shape (n_wavelengths,)
        Difference: contaminated - uncontaminated

    n_features_in_ : int
        Number of wavelengths

    contaminated_std_ : ndarray, shape (n_wavelengths,)
        Standard deviation in contaminated group (for confidence intervals)

    uncontaminated_std_ : ndarray, shape (n_wavelengths,)
        Standard deviation in uncontaminated group

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import DifferenceAnalyzer
    >>>
    >>> # Sample data
    >>> X_contaminated = np.random.randn(30, 100)
    >>> X_uncontaminated = np.random.randn(40, 100)
    >>>
    >>> # Analyze difference
    >>> analyzer = DifferenceAnalyzer()
    >>> analyzer.fit(X_contaminated, X_uncontaminated)
    >>>
    >>> # Get difference spectrum
    >>> diff = analyzer.get_difference_spectrum()
    >>>
    >>> # Identify peak regions (potential contaminant influence)
    >>> wavelengths = np.arange(1000, 1100)
    >>> peaks = analyzer.identify_peak_regions(wavelengths, threshold=0.5)
    """

    def __init__(self, normalize: bool = True, method: str = 'mean'):
        self.normalize = normalize
        self.method = method

    def fit(self, X_contaminated: np.ndarray, X_uncontaminated: np.ndarray) -> 'DifferenceAnalyzer':
        """
        Compute difference spectrum between groups.

        Parameters
        ----------
        X_contaminated : array-like, shape (n_contaminated, n_wavelengths)
            Spectral data from contaminated samples

        X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
            Spectral data from uncontaminated samples

        Returns
        -------
        self : DifferenceAnalyzer
            Fitted analyzer
        """
        # Validate inputs
        X_contaminated = check_array(X_contaminated, dtype=np.float64)
        X_uncontaminated = check_array(X_uncontaminated, dtype=np.float64)

        if X_contaminated.shape[1] != X_uncontaminated.shape[1]:
            raise ValueError(
                f"X_contaminated and X_uncontaminated must have same number of wavelengths. "
                f"Got {X_contaminated.shape[1]} and {X_uncontaminated.shape[1]}"
            )

        self.n_features_in_ = X_contaminated.shape[1]

        # Optionally normalize spectra
        if self.normalize:
            X_contaminated = self._snv_normalize(X_contaminated)
            X_uncontaminated = self._snv_normalize(X_uncontaminated)

        # Compute representative spectra based on method
        if self.method == 'mean':
            self.contaminated_representative_ = np.mean(X_contaminated, axis=0)
            self.uncontaminated_representative_ = np.mean(X_uncontaminated, axis=0)
        elif self.method == 'median':
            self.contaminated_representative_ = np.median(X_contaminated, axis=0)
            self.uncontaminated_representative_ = np.median(X_uncontaminated, axis=0)
        elif self.method == 'pca':
            self.contaminated_representative_ = self._pca_representative(X_contaminated)
            self.uncontaminated_representative_ = self._pca_representative(X_uncontaminated)
        else:
            raise ValueError(f"method must be 'mean', 'median', or 'pca', got '{self.method}'")

        # Compute difference spectrum
        self.difference_spectrum_ = self.contaminated_representative_ - self.uncontaminated_representative_

        # Store standard deviations for confidence intervals
        self.contaminated_std_ = np.std(X_contaminated, axis=0)
        self.uncontaminated_std_ = np.std(X_uncontaminated, axis=0)

        # Store sample sizes for SE calculation
        self._n_contaminated = X_contaminated.shape[0]
        self._n_uncontaminated = X_uncontaminated.shape[0]

        return self

    def _snv_normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply SNV (Standard Normal Variate) normalization."""
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        return (X - mean) / std

    def _pca_representative(self, X: np.ndarray) -> np.ndarray:
        """Get first PC direction as representative spectrum."""
        X_centered = X - np.mean(X, axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Return mean projected onto first PC direction
        return np.mean(X, axis=0) + Vt[0] * np.std(X @ Vt[0].T)

    def get_difference_spectrum(self) -> np.ndarray:
        """
        Get the difference spectrum (contaminated - uncontaminated).

        Returns
        -------
        difference : ndarray, shape (n_wavelengths,)
            Difference spectrum. Positive values indicate higher absorbance
            in contaminated samples, negative indicates lower.
        """
        check_is_fitted(self, 'difference_spectrum_')
        return self.difference_spectrum_.copy()

    def get_absolute_difference(self) -> np.ndarray:
        """
        Get absolute difference spectrum (magnitude of change).

        Returns
        -------
        abs_difference : ndarray, shape (n_wavelengths,)
            Absolute value of difference spectrum
        """
        check_is_fitted(self, 'difference_spectrum_')
        return np.abs(self.difference_spectrum_)

    def get_normalized_influence(self) -> np.ndarray:
        """
        Get contaminant influence normalized to 0-1 range.

        Returns
        -------
        influence : ndarray, shape (n_wavelengths,)
            Normalized influence score (0 = no influence, 1 = maximum influence)
        """
        check_is_fitted(self, 'difference_spectrum_')
        abs_diff = np.abs(self.difference_spectrum_)
        max_diff = np.max(abs_diff)
        if max_diff < 1e-10:
            return np.zeros_like(abs_diff)
        return abs_diff / max_diff

    def identify_peak_regions(
        self,
        wavelengths: np.ndarray,
        threshold: float = 0.5,
        min_width: int = 3
    ) -> List[Tuple[float, float, float]]:
        """
        Identify wavelength regions with significant contaminant influence.

        Parameters
        ----------
        wavelengths : array-like, shape (n_wavelengths,)
            Wavelength values

        threshold : float, default=0.5
            Threshold for normalized influence (0-1). Regions with influence
            above this threshold are identified as contaminant-affected.

        min_width : int, default=3
            Minimum number of consecutive wavelengths to form a region

        Returns
        -------
        regions : list of (start_wl, end_wl, peak_influence)
            List of identified regions with their wavelength bounds and
            maximum influence score within the region
        """
        check_is_fitted(self, 'difference_spectrum_')
        wavelengths = np.asarray(wavelengths)

        if len(wavelengths) != self.n_features_in_:
            raise ValueError(
                f"wavelengths length ({len(wavelengths)}) must match "
                f"n_features_in_ ({self.n_features_in_})"
            )

        influence = self.get_normalized_influence()
        above_threshold = influence >= threshold

        # Find contiguous regions
        regions = []
        in_region = False
        start_idx = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_region:
                # Start of new region
                in_region = True
                start_idx = i
            elif not above_threshold[i] and in_region:
                # End of region
                in_region = False
                if i - start_idx >= min_width:
                    peak_influence = np.max(influence[start_idx:i])
                    regions.append((
                        wavelengths[start_idx],
                        wavelengths[i-1],
                        peak_influence
                    ))

        # Handle region at end
        if in_region and len(above_threshold) - start_idx >= min_width:
            peak_influence = np.max(influence[start_idx:])
            regions.append((
                wavelengths[start_idx],
                wavelengths[-1],
                peak_influence
            ))

        return regions

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence interval for the difference spectrum.

        Uses standard error of difference between means.

        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level

        Returns
        -------
        lower : ndarray, shape (n_wavelengths,)
            Lower bound of confidence interval
        upper : ndarray, shape (n_wavelengths,)
            Upper bound of confidence interval
        """
        check_is_fitted(self, ['difference_spectrum_', 'contaminated_std_', 'uncontaminated_std_'])

        from scipy import stats

        # Standard error of difference between means
        se_diff = np.sqrt(
            self.contaminated_std_**2 / self._n_contaminated +
            self.uncontaminated_std_**2 / self._n_uncontaminated
        )

        # Critical value for t-distribution
        df = self._n_contaminated + self._n_uncontaminated - 2
        t_crit = stats.t.ppf((1 + confidence) / 2, df)

        lower = self.difference_spectrum_ - t_crit * se_diff
        upper = self.difference_spectrum_ + t_crit * se_diff

        return lower, upper

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        DifferenceAnalyzer doesn't transform data - raises informative error.

        For actual data transformation, use EstimatedEPO or ContaminantGLSW.
        """
        raise NotImplementedError(
            "DifferenceAnalyzer is for exploratory analysis only. "
            "Use EstimatedEPO or ContaminantGLSW for data transformation."
        )


class EstimatedEPO(BaseEstimator, TransformerMixin):
    """
    External Parameter Orthogonalization using estimated interferent library.

    This is an adaptation of EPO (Roger et al., 2003) for cases where pure
    contaminant spectra are not available. Instead, the interferent library
    is estimated from the differences between contaminated and uncontaminated
    sample groups.

    No Y values (target variable) are required - this is unsupervised.

    Algorithm:
    1. Compute difference vectors between groups (multiple methods)
    2. Build "pseudo-interferent" library from these differences
    3. Extract principal components of the interferent library
    4. Project all spectra orthogonal to this interferent subspace

    Parameters
    ----------
    n_components : int, default=2
        Number of interferent components to remove. Start small (1-3) and
        increase cautiously to avoid removing analyte signal.

    estimation_method : {'mean_diff', 'pca_diff', 'bootstrap'}, default='pca_diff'
        Method for building interferent library:
        - 'mean_diff': Single difference between group means
        - 'pca_diff': PCA on concatenated groups to find discriminating directions
        - 'bootstrap': Bootstrap sampling to build multiple difference vectors

    n_bootstrap : int, default=50
        Number of bootstrap samples for 'bootstrap' method

    center : bool, default=True
        Whether to mean-center data before EPO

    svd_tol : float, default=1e-8
        Tolerance for SVD truncation

    random_state : int or None, default=None
        Random seed for bootstrap sampling

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths

    interferent_library_ : ndarray, shape (n_interferents, n_wavelengths)
        Estimated interferent spectra library

    interferent_components_ : ndarray, shape (n_wavelengths, n_components_)
        Principal components of interferent subspace

    P_orth_ : ndarray, shape (n_wavelengths, n_wavelengths)
        Orthogonal projection matrix

    X_mean_ : ndarray, shape (n_wavelengths,)
        Mean spectrum for centering

    explained_variance_ : ndarray, shape (n_components_,)
        Variance explained by each interferent component

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import EstimatedEPO
    >>>
    >>> # Sample data
    >>> X_contaminated = np.random.randn(30, 100) + 0.5  # With contaminant
    >>> X_uncontaminated = np.random.randn(40, 100)      # Clean
    >>>
    >>> # Fit EPO on group differences
    >>> epo = EstimatedEPO(n_components=2)
    >>> epo.fit_groups(X_contaminated, X_uncontaminated)
    >>>
    >>> # Transform all spectra
    >>> X_all = np.vstack([X_contaminated, X_uncontaminated])
    >>> X_corrected = epo.transform(X_all)
    """

    def __init__(
        self,
        n_components: int = 2,
        estimation_method: str = 'pca_diff',
        n_bootstrap: int = 50,
        center: bool = True,
        svd_tol: float = 1e-8,
        random_state: Optional[int] = None
    ):
        self.n_components = n_components
        self.estimation_method = estimation_method
        self.n_bootstrap = n_bootstrap
        self.center = center
        self.svd_tol = svd_tol
        self.random_state = random_state

    def fit_groups(
        self,
        X_contaminated: np.ndarray,
        X_uncontaminated: np.ndarray
    ) -> 'EstimatedEPO':
        """
        Fit EPO using estimated interferent library from group differences.

        Parameters
        ----------
        X_contaminated : array-like, shape (n_contaminated, n_wavelengths)
            Spectral data from contaminated samples

        X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
            Spectral data from uncontaminated samples

        Returns
        -------
        self : EstimatedEPO
            Fitted transformer
        """
        # Validate inputs
        X_contaminated = check_array(X_contaminated, dtype=np.float64)
        X_uncontaminated = check_array(X_uncontaminated, dtype=np.float64)

        if X_contaminated.shape[1] != X_uncontaminated.shape[1]:
            raise ValueError(
                f"Groups must have same number of wavelengths. "
                f"Got {X_contaminated.shape[1]} and {X_uncontaminated.shape[1]}"
            )

        self.n_features_in_ = X_contaminated.shape[1]

        # Combine groups for centering
        X_all = np.vstack([X_contaminated, X_uncontaminated])

        # Store mean for centering
        if self.center:
            self.X_mean_ = np.mean(X_all, axis=0)
        else:
            self.X_mean_ = np.zeros(self.n_features_in_)

        # Build interferent library based on estimation method
        if self.estimation_method == 'mean_diff':
            self.interferent_library_ = self._estimate_mean_diff(
                X_contaminated, X_uncontaminated
            )
        elif self.estimation_method == 'pca_diff':
            self.interferent_library_ = self._estimate_pca_diff(
                X_contaminated, X_uncontaminated
            )
        elif self.estimation_method == 'bootstrap':
            self.interferent_library_ = self._estimate_bootstrap(
                X_contaminated, X_uncontaminated
            )
        else:
            raise ValueError(
                f"estimation_method must be 'mean_diff', 'pca_diff', or 'bootstrap', "
                f"got '{self.estimation_method}'"
            )

        # Build EPO projection matrix from interferent library
        self._build_projection_matrix()

        return self

    def fit(self, X: np.ndarray, y=None, X_interferents: Optional[np.ndarray] = None) -> 'EstimatedEPO':
        """
        Fit EPO with explicit interferent library (for sklearn compatibility).

        For the typical use case with two groups, use fit_groups() instead.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Training spectral data (used only for centering)

        y : Ignored
            Not used

        X_interferents : array-like, shape (n_interferents, n_wavelengths)
            Explicit interferent library. If None, raises error.

        Returns
        -------
        self : EstimatedEPO
        """
        X = check_array(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        if X_interferents is None:
            raise ValueError(
                "X_interferents is required for EPO. "
                "For group-based estimation, use fit_groups() instead."
            )

        X_interferents = check_array(X_interferents, dtype=np.float64)

        if X_interferents.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_interferents must have same number of wavelengths as X. "
                f"Got {X_interferents.shape[1]} and {self.n_features_in_}"
            )

        if self.center:
            self.X_mean_ = np.mean(X, axis=0)
        else:
            self.X_mean_ = np.zeros(self.n_features_in_)

        self.interferent_library_ = X_interferents
        self._build_projection_matrix()

        return self

    def _estimate_mean_diff(
        self,
        X_contaminated: np.ndarray,
        X_uncontaminated: np.ndarray
    ) -> np.ndarray:
        """Build library from mean difference (single vector)."""
        mean_cont = np.mean(X_contaminated, axis=0)
        mean_uncont = np.mean(X_uncontaminated, axis=0)
        diff = mean_cont - mean_uncont

        # Return as 2D array
        return diff.reshape(1, -1)

    def _estimate_pca_diff(
        self,
        X_contaminated: np.ndarray,
        X_uncontaminated: np.ndarray
    ) -> np.ndarray:
        """
        Build library from PCA on discriminating directions.

        Algorithm:
        1. Compute within-group covariance matrices
        2. Compute between-group covariance (group mean difference)
        3. Use LDA-like approach to find discriminating directions
        """
        n_cont = X_contaminated.shape[0]
        n_uncont = X_uncontaminated.shape[0]

        # Group means
        mean_cont = np.mean(X_contaminated, axis=0)
        mean_uncont = np.mean(X_uncontaminated, axis=0)
        overall_mean = (n_cont * mean_cont + n_uncont * mean_uncont) / (n_cont + n_uncont)

        # Between-group scatter (difference direction)
        diff = mean_cont - mean_uncont

        # Create multiple "pseudo-interferent" spectra by adding noise
        # This helps build a more robust interferent subspace
        rng = np.random.RandomState(self.random_state)

        n_pseudo = max(self.n_components * 3, 10)  # At least 10 pseudo-spectra
        pseudo_library = []

        # Add mean difference
        pseudo_library.append(diff)

        # Add variations around mean difference
        diff_std = np.std(diff) if np.std(diff) > 1e-10 else 1.0
        for _ in range(n_pseudo - 1):
            noise = rng.randn(len(diff)) * diff_std * 0.1
            pseudo_library.append(diff + noise)

        return np.array(pseudo_library)

    def _estimate_bootstrap(
        self,
        X_contaminated: np.ndarray,
        X_uncontaminated: np.ndarray
    ) -> np.ndarray:
        """Build library from bootstrap sampling of group differences."""
        rng = np.random.RandomState(self.random_state)

        n_cont = X_contaminated.shape[0]
        n_uncont = X_uncontaminated.shape[0]

        pseudo_library = []

        for _ in range(self.n_bootstrap):
            # Sample with replacement from each group
            cont_idx = rng.choice(n_cont, size=n_cont, replace=True)
            uncont_idx = rng.choice(n_uncont, size=n_uncont, replace=True)

            # Compute means of bootstrap samples
            mean_cont = np.mean(X_contaminated[cont_idx], axis=0)
            mean_uncont = np.mean(X_uncontaminated[uncont_idx], axis=0)

            # Store difference
            pseudo_library.append(mean_cont - mean_uncont)

        return np.array(pseudo_library)

    def _build_projection_matrix(self):
        """Build orthogonal projection matrix from interferent library."""
        # Center interferent library
        lib_mean = np.mean(self.interferent_library_, axis=0)
        lib_centered = self.interferent_library_ - lib_mean

        # Handle case of single interferent spectrum
        if self.interferent_library_.shape[0] == 1:
            # Use the single spectrum directly
            v = self.interferent_library_[0]
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            self.interferent_components_ = v_norm.reshape(-1, 1)
            self.explained_variance_ = np.array([1.0])
            self.n_components_ = 1
        else:
            # SVD to get principal components
            try:
                U, S, Vt = np.linalg.svd(lib_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "SVD failed on interferent library. Check for NaN/Inf values."
                )

            # Truncate small singular values
            S_valid = S > self.svd_tol
            n_valid = np.sum(S_valid)

            if n_valid == 0:
                warnings.warn(
                    "Interferent library has no significant variation. "
                    "EPO will have no effect.",
                    UserWarning
                )
                self.interferent_components_ = np.zeros((self.n_features_in_, 1))
                self.explained_variance_ = np.array([0.0])
                self.n_components_ = 0
                self.P_orth_ = np.eye(self.n_features_in_)
                return

            # Effective number of components
            effective_n = min(self.n_components, n_valid)
            if effective_n < self.n_components:
                warnings.warn(
                    f"Only {n_valid} significant components in interferent library. "
                    f"Using {effective_n} instead of {self.n_components}.",
                    UserWarning
                )

            self.n_components_ = effective_n

            # Store components (columns of V)
            V = Vt.T
            self.interferent_components_ = V[:, :self.n_components_]

            # Store explained variance
            total_var = np.sum(S**2)
            if total_var > 0:
                self.explained_variance_ = (S[:self.n_components_]**2) / total_var
            else:
                self.explained_variance_ = np.zeros(self.n_components_)

        # Build orthogonal projection matrix: P_orth = I - V @ V.T
        V_comp = self.interferent_components_
        self.P_orth_ = np.eye(self.n_features_in_) - V_comp @ V_comp.T

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply EPO transformation to remove estimated interferent signal.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to transform

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_wavelengths)
            Transformed data with interferent signal removed.
            Note: Data is mean-centered using training mean.
        """
        check_is_fitted(self, ['P_orth_', 'X_mean_'])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but EstimatedEPO was fitted with "
                f"{self.n_features_in_} features."
            )

        # Center and project
        X_centered = X - self.X_mean_
        X_corrected = X_centered @ self.P_orth_

        return X_corrected

    def get_interferent_components(self) -> np.ndarray:
        """Get the estimated interferent principal components."""
        check_is_fitted(self, 'interferent_components_')
        return self.interferent_components_.copy()

    def get_explained_variance(self) -> np.ndarray:
        """Get variance explained by each interferent component."""
        check_is_fitted(self, 'explained_variance_')
        return self.explained_variance_.copy()

    def get_wavelength_influence(self) -> np.ndarray:
        """
        Get relative influence of interferent at each wavelength.

        Returns
        -------
        influence : ndarray, shape (n_wavelengths,)
            Influence score at each wavelength (0-1, higher = more interferent influence)
        """
        check_is_fitted(self, 'interferent_components_')

        # Sum of squared loadings across components
        influence = np.sum(self.interferent_components_**2, axis=1)

        # Normalize to 0-1
        max_inf = np.max(influence)
        if max_inf > 1e-10:
            influence = influence / max_inf

        return influence


class ContaminantOPLSDA(BaseEstimator, TransformerMixin):
    """
    OPLS-DA for identifying contaminant-influenced wavelengths.

    This implements a simplified OPLS-DA (Orthogonal Projections to Latent Structures
    Discriminant Analysis) for binary classification: contaminated vs uncontaminated.

    No Y values (target variable like % collagen) are required - the binary group
    membership (contaminated/uncontaminated) is the target.

    The method identifies wavelengths that discriminate between groups, which
    indicates where the contaminant has spectral influence.

    Parameters
    ----------
    n_components : int, default=2
        Number of predictive components

    n_orthogonal : int, default=1
        Number of orthogonal components (Y-orthogonal variation within groups)

    scale : bool, default=True
        Whether to scale features to unit variance

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths

    predictive_loadings_ : ndarray, shape (n_wavelengths, n_components)
        Loadings for predictive (discriminant) components

    orthogonal_loadings_ : ndarray, shape (n_wavelengths, n_orthogonal)
        Loadings for orthogonal (within-group) components

    coef_ : ndarray, shape (n_wavelengths,)
        Discriminant coefficients (higher = more discriminating power)

    vip_scores_ : ndarray, shape (n_wavelengths,)
        VIP (Variable Importance in Projection) scores

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import ContaminantOPLSDA
    >>>
    >>> # Sample data
    >>> X_contaminated = np.random.randn(30, 100) + 0.5
    >>> X_uncontaminated = np.random.randn(40, 100)
    >>> wavelengths = np.arange(1000, 1100)
    >>>
    >>> # Fit OPLS-DA
    >>> oplsda = ContaminantOPLSDA()
    >>> oplsda.fit(X_contaminated, X_uncontaminated)
    >>>
    >>> # Get wavelength influence
    >>> influence = oplsda.get_wavelength_influence()
    >>>
    >>> # Get exclusion regions
    >>> regions = oplsda.get_exclusion_regions(wavelengths, threshold=0.7)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_orthogonal: int = 1,
        scale: bool = True
    ):
        self.n_components = n_components
        self.n_orthogonal = n_orthogonal
        self.scale = scale

    def fit(
        self,
        X_contaminated: np.ndarray,
        X_uncontaminated: np.ndarray
    ) -> 'ContaminantOPLSDA':
        """
        Fit OPLS-DA model to discriminate between groups.

        Parameters
        ----------
        X_contaminated : array-like, shape (n_contaminated, n_wavelengths)
            Spectral data from contaminated samples

        X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
            Spectral data from uncontaminated samples

        Returns
        -------
        self : ContaminantOPLSDA
        """
        # Validate inputs
        X_contaminated = check_array(X_contaminated, dtype=np.float64)
        X_uncontaminated = check_array(X_uncontaminated, dtype=np.float64)

        if X_contaminated.shape[1] != X_uncontaminated.shape[1]:
            raise ValueError(
                f"Groups must have same number of wavelengths. "
                f"Got {X_contaminated.shape[1]} and {X_uncontaminated.shape[1]}"
            )

        self.n_features_in_ = X_contaminated.shape[1]

        # Create combined X and binary y
        X = np.vstack([X_contaminated, X_uncontaminated])
        y = np.concatenate([
            np.ones(X_contaminated.shape[0]),
            np.zeros(X_uncontaminated.shape[0])
        ])

        n_samples = X.shape[0]

        # Scale if requested
        if self.scale:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X

        # Store mean for centering
        self.X_mean_ = np.mean(X_scaled, axis=0)
        self.y_mean_ = np.mean(y)

        X_centered = X_scaled - self.X_mean_
        y_centered = y - self.y_mean_

        # Determine effective number of components
        max_comp = min(n_samples - 1, self.n_features_in_, 10)
        effective_n_comp = min(self.n_components, max_comp)

        if effective_n_comp < self.n_components:
            warnings.warn(
                f"Reducing n_components from {self.n_components} to {effective_n_comp}",
                UserWarning
            )

        # Fit standard PLS first to get predictive subspace
        pls = PLSRegression(n_components=effective_n_comp)
        pls.fit(X_centered, y_centered)

        # Store predictive components
        self.predictive_loadings_ = pls.x_loadings_
        self.predictive_weights_ = pls.x_weights_

        # Get PLS coefficients (discriminant direction)
        self.coef_ = pls.coef_.ravel()

        # Compute VIP scores (Variable Importance in Projection)
        self.vip_scores_ = self._compute_vip(pls, X_centered, y_centered)

        # Compute orthogonal components (simplified approach)
        # These are the variations in X that don't predict y
        X_pred_scores = pls.transform(X_centered)
        X_reconstructed = X_pred_scores @ self.predictive_loadings_.T
        X_orthogonal = X_centered - X_reconstructed

        # PCA on orthogonal part
        if self.n_orthogonal > 0 and X_orthogonal.shape[0] > 1:
            _, S_orth, Vt_orth = np.linalg.svd(X_orthogonal, full_matrices=False)
            n_orth = min(self.n_orthogonal, len(S_orth))
            self.orthogonal_loadings_ = Vt_orth[:n_orth].T
        else:
            self.orthogonal_loadings_ = np.zeros((self.n_features_in_, 0))

        # Store the underlying PLS model
        self._pls = pls

        return self

    def _compute_vip(
        self,
        pls: PLSRegression,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Compute VIP (Variable Importance in Projection) scores."""
        n_features = X.shape[1]
        n_components = pls.n_components

        # Get weights and scores
        W = pls.x_weights_
        T = pls.x_scores_
        Q = pls.y_loadings_

        # Handle Q shape: sklearn may return (n_components, n_targets) or transpose
        # For 1D y, Q should be (n_components, 1) but let's handle both cases
        if Q.shape[0] == 1 and Q.shape[1] == n_components:
            # Q is transposed: (1, n_components) -> use Q[0, i]
            Q = Q.T  # Now (n_components, 1)

        # Compute SS for each component
        ss = np.zeros(n_components)
        for i in range(n_components):
            # Q[i, 0] is the y-loading for component i
            q_val = Q[i, 0] if Q.shape[1] > 0 else Q[i] if Q.ndim == 1 else 1.0
            ss[i] = (T[:, i].T @ T[:, i]) * (q_val**2)

        total_ss = np.sum(ss)

        # Compute VIP for each variable
        vip = np.zeros(n_features)
        for j in range(n_features):
            s = 0
            for i in range(n_components):
                s += (W[j, i]**2) * ss[i]
            vip[j] = np.sqrt(n_features * s / (total_ss + 1e-10))

        return vip

    def get_wavelength_influence(self) -> np.ndarray:
        """
        Get contaminant influence at each wavelength.

        Uses VIP scores as the measure of influence.

        Returns
        -------
        influence : ndarray, shape (n_wavelengths,)
            Influence score (higher = more contaminant influence)
        """
        check_is_fitted(self, 'vip_scores_')

        # Normalize VIP scores to 0-1 range
        vip = self.vip_scores_
        max_vip = np.max(vip)
        if max_vip > 1e-10:
            return vip / max_vip
        return np.zeros_like(vip)

    def get_discriminant_direction(self) -> np.ndarray:
        """
        Get the discriminant direction (coefficients).

        Returns
        -------
        direction : ndarray, shape (n_wavelengths,)
            Discriminant coefficients. Positive values indicate wavelengths
            where contaminated samples have higher response.
        """
        check_is_fitted(self, 'coef_')
        return self.coef_.copy()

    def get_exclusion_regions(
        self,
        wavelengths: np.ndarray,
        threshold: float = 0.7,
        min_width: int = 3
    ) -> List[Tuple[float, float]]:
        """
        Get wavelength regions to exclude based on contaminant influence.

        Parameters
        ----------
        wavelengths : array-like, shape (n_wavelengths,)
            Wavelength values

        threshold : float, default=0.7
            Influence threshold (0-1). Wavelengths above this are excluded.

        min_width : int, default=3
            Minimum region width in wavelengths

        Returns
        -------
        regions : list of (start_wl, end_wl)
            Wavelength regions to exclude
        """
        check_is_fitted(self, 'vip_scores_')
        wavelengths = np.asarray(wavelengths)

        if len(wavelengths) != self.n_features_in_:
            raise ValueError(
                f"wavelengths length ({len(wavelengths)}) must match "
                f"n_features_in_ ({self.n_features_in_})"
            )

        influence = self.get_wavelength_influence()
        above_threshold = influence >= threshold

        # Find contiguous regions
        regions = []
        in_region = False
        start_idx = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_region:
                in_region = True
                start_idx = i
            elif not above_threshold[i] and in_region:
                in_region = False
                if i - start_idx >= min_width:
                    regions.append((wavelengths[start_idx], wavelengths[i-1]))

        # Handle region at end
        if in_region and len(above_threshold) - start_idx >= min_width:
            regions.append((wavelengths[start_idx], wavelengths[-1]))

        return regions

    def get_splot_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for S-plot visualization.

        S-plot shows covariance (correlation with y) vs correlation (reliability).
        Variables in corners have both high magnitude and high reliability.

        Returns
        -------
        p_corr : ndarray, shape (n_wavelengths,)
            Correlation of each wavelength with y (reliability)
        p_cov : ndarray, shape (n_wavelengths,)
            Covariance of each wavelength with predictive scores (magnitude)
        """
        check_is_fitted(self, ['_pls', 'coef_'])

        # p(corr) = correlation with first predictive component
        p_corr = self.predictive_loadings_[:, 0] / (
            np.linalg.norm(self.predictive_loadings_[:, 0]) + 1e-10
        )

        # p(cov) = covariance (scaled coefficients)
        p_cov = self.coef_ / (np.std(self.coef_) + 1e-10)

        return p_corr, p_cov

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X by removing orthogonal (within-group) variation.

        This returns data with only the predictive variation retained.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_wavelengths)
            Transformed data
        """
        check_is_fitted(self, ['orthogonal_loadings_', 'X_mean_'])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Scale if fitted with scaling
        if self.scale:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        X_centered = X_scaled - self.X_mean_

        # Remove orthogonal components
        if self.orthogonal_loadings_.shape[1] > 0:
            t_orth = X_centered @ self.orthogonal_loadings_
            X_transformed = X_centered - t_orth @ self.orthogonal_loadings_.T
        else:
            X_transformed = X_centered

        return X_transformed


class ContaminantGLSW(BaseEstimator, TransformerMixin):
    """
    Contaminant-aware Generalized Least Squares Weighting.

    This adapts GLSW to down-weight wavelengths where contaminant influence is
    detected. It's a "soft" approach that doesn't hard-exclude wavelengths but
    reduces their influence on subsequent modeling.

    No Y values are required - weighting is based on group differences.

    Parameters
    ----------
    regularization : float, default=1e-6
        Regularization for numerical stability

    influence_power : float, default=1.0
        Power for influence-based weighting. Higher values = more aggressive
        down-weighting of contaminant regions.
        - 1.0: Linear inverse weighting
        - 2.0: Quadratic inverse weighting (more aggressive)

    min_weight : float, default=0.1
        Minimum weight to assign (prevents complete elimination)

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths

    feature_weights_ : ndarray, shape (n_wavelengths,)
        Weight for each wavelength (higher = more trusted)

    W_sqrt_ : ndarray, shape (n_wavelengths,)
        Square root of weights for transformation

    contamination_influence_ : ndarray, shape (n_wavelengths,)
        Estimated contamination influence at each wavelength

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import ContaminantGLSW
    >>>
    >>> # Sample data
    >>> X_contaminated = np.random.randn(30, 100) + 0.5
    >>> X_uncontaminated = np.random.randn(40, 100)
    >>>
    >>> # Fit GLSW
    >>> glsw = ContaminantGLSW()
    >>> glsw.fit_groups(X_contaminated, X_uncontaminated)
    >>>
    >>> # Transform all data
    >>> X_all = np.vstack([X_contaminated, X_uncontaminated])
    >>> X_weighted = glsw.transform(X_all)
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        influence_power: float = 1.0,
        min_weight: float = 0.1
    ):
        self.regularization = regularization
        self.influence_power = influence_power
        self.min_weight = min_weight

    def fit_groups(
        self,
        X_contaminated: np.ndarray,
        X_uncontaminated: np.ndarray
    ) -> 'ContaminantGLSW':
        """
        Fit GLSW weights based on contaminant influence from group differences.

        Wavelengths with high variance in contaminated group but low variance
        in uncontaminated group are down-weighted.

        Parameters
        ----------
        X_contaminated : array-like, shape (n_contaminated, n_wavelengths)
            Spectral data from contaminated samples

        X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
            Spectral data from uncontaminated samples

        Returns
        -------
        self : ContaminantGLSW
        """
        # Validate inputs
        X_contaminated = check_array(X_contaminated, dtype=np.float64)
        X_uncontaminated = check_array(X_uncontaminated, dtype=np.float64)

        if X_contaminated.shape[1] != X_uncontaminated.shape[1]:
            raise ValueError(
                f"Groups must have same number of wavelengths. "
                f"Got {X_contaminated.shape[1]} and {X_uncontaminated.shape[1]}"
            )

        self.n_features_in_ = X_contaminated.shape[1]

        # Compute variance in each group
        var_contaminated = np.var(X_contaminated, axis=0)
        var_uncontaminated = np.var(X_uncontaminated, axis=0)

        # Compute mean difference (absolute)
        mean_cont = np.mean(X_contaminated, axis=0)
        mean_uncont = np.mean(X_uncontaminated, axis=0)
        abs_diff = np.abs(mean_cont - mean_uncont)

        # Contamination influence metric:
        # High variance ratio (contaminated/uncontaminated) + high mean difference
        # indicates contaminant influence
        var_ratio = (var_contaminated + self.regularization) / (var_uncontaminated + self.regularization)

        # Normalize components
        var_ratio_norm = var_ratio / (np.max(var_ratio) + 1e-10)
        abs_diff_norm = abs_diff / (np.max(abs_diff) + 1e-10)

        # Combined influence (average of normalized components)
        self.contamination_influence_ = 0.5 * var_ratio_norm + 0.5 * abs_diff_norm

        # Compute weights: inverse of influence
        # Higher influence = lower weight
        weights = 1.0 / (self.contamination_influence_**self.influence_power + self.regularization)

        # Clip to minimum weight
        weights = np.maximum(weights, self.min_weight)

        # Normalize weights to have mean = 1
        self.feature_weights_ = weights / np.mean(weights)

        # Square root for transformation
        self.W_sqrt_ = np.sqrt(self.feature_weights_)

        return self

    def fit(self, X: np.ndarray, y=None) -> 'ContaminantGLSW':
        """
        Fit GLSW using variance-based weighting (standard approach).

        For contaminant-aware weighting, use fit_groups() instead.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data

        y : Ignored

        Returns
        -------
        self : ContaminantGLSW
        """
        X = check_array(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Standard variance-based weighting
        variances = np.var(X, axis=0) + self.regularization

        self.contamination_influence_ = variances / np.max(variances)

        weights = 1.0 / (variances**self.influence_power)
        weights = np.maximum(weights, self.min_weight)
        self.feature_weights_ = weights / np.mean(weights)
        self.W_sqrt_ = np.sqrt(self.feature_weights_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply GLSW weighting to spectral data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data

        Returns
        -------
        X_weighted : ndarray, shape (n_samples, n_wavelengths)
            Weighted spectral data
        """
        check_is_fitted(self, ['W_sqrt_', 'n_features_in_'])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        return X * self.W_sqrt_

    def get_feature_weights(self) -> np.ndarray:
        """Get weight for each wavelength."""
        check_is_fitted(self, 'feature_weights_')
        return self.feature_weights_.copy()

    def get_contamination_influence(self) -> np.ndarray:
        """Get estimated contamination influence at each wavelength."""
        check_is_fitted(self, 'contamination_influence_')
        return self.contamination_influence_.copy()


class RegionExcluder(BaseEstimator, TransformerMixin):
    """
    Backward iPLS-based region exclusion for contaminant removal.

    This method uses backward interval PLS on uncontaminated samples (with Y values)
    to identify spectral regions that hurt prediction performance. These regions
    likely contain contaminant interference.

    **REQUIRES Y values** (target variable) for the uncontaminated samples.

    Algorithm:
    1. Divide spectrum into intervals
    2. Use backward iPLS to iteratively remove intervals that improve prediction
    3. The removed intervals are the "bad" regions (contaminant + noise)
    4. Apply the final interval selection to all data

    Parameters
    ----------
    n_intervals : int, default=20
        Number of intervals to divide spectrum into

    cv_folds : int, default=5
        Number of cross-validation folds

    min_intervals : int, default=5
        Minimum number of intervals to keep (prevents over-exclusion)

    wavelengths : array-like or None, default=None
        Wavelength values. If None, uses indices.

    random_state : int or None, default=None
        Random seed for CV

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths

    selected_indices_ : ndarray
        Indices of selected (kept) wavelengths

    excluded_indices_ : ndarray
        Indices of excluded wavelengths

    selected_intervals_ : list
        List of selected interval indices

    exclusion_history_ : list
        History of interval exclusions with RMSECV at each step

    best_rmsecv_ : float
        RMSECV of final selected intervals

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import RegionExcluder
    >>>
    >>> # Uncontaminated data with Y values (required)
    >>> X_uncontaminated = np.random.randn(50, 200)
    >>> y_uncontaminated = np.random.randn(50)
    >>> wavelengths = np.linspace(1000, 2500, 200)
    >>>
    >>> # Fit on uncontaminated data
    >>> excluder = RegionExcluder(n_intervals=20)
    >>> excluder.fit(X_uncontaminated, y_uncontaminated, wavelengths)
    >>>
    >>> # Transform contaminated data
    >>> X_contaminated = np.random.randn(30, 200)
    >>> X_optimized = excluder.transform(X_contaminated)
    """

    def __init__(
        self,
        n_intervals: int = 20,
        cv_folds: int = 5,
        min_intervals: int = 5,
        wavelengths: Optional[np.ndarray] = None,
        random_state: Optional[int] = None
    ):
        self.n_intervals = n_intervals
        self.cv_folds = cv_folds
        self.min_intervals = min_intervals
        self.wavelengths = wavelengths
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: Optional[np.ndarray] = None
    ) -> 'RegionExcluder':
        """
        Fit region excluder using backward iPLS.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data (should be uncontaminated samples)

        y : array-like, shape (n_samples,)
            Target values (e.g., % collagen)

        wavelengths : array-like or None, optional
            Wavelength values. Overrides constructor wavelengths if provided.

        Returns
        -------
        self : RegionExcluder
        """
        # Validate inputs
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {len(y)}"
            )

        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Handle wavelengths
        if wavelengths is not None:
            self.wavelengths_ = np.asarray(wavelengths)
        elif self.wavelengths is not None:
            self.wavelengths_ = np.asarray(self.wavelengths)
        else:
            self.wavelengths_ = np.arange(self.n_features_in_)

        if len(self.wavelengths_) != self.n_features_in_:
            raise ValueError(
                f"wavelengths length ({len(self.wavelengths_)}) must match "
                f"n_features ({self.n_features_in_})"
            )

        # Adjust cv_folds if needed
        effective_cv = min(self.cv_folds, n_samples)
        if effective_cv < self.cv_folds:
            warnings.warn(
                f"Reducing cv_folds from {self.cv_folds} to {effective_cv}",
                UserWarning
            )

        # Create intervals
        self.intervals_ = self._create_intervals()

        # Start with all intervals
        remaining_intervals = list(range(len(self.intervals_)))

        # Evaluate full model
        full_indices = self._get_combined_indices(remaining_intervals)
        best_rmsecv = self._evaluate_pls(X[:, full_indices], y, effective_cv)

        self.exclusion_history_ = [{
            'n_intervals': len(remaining_intervals),
            'rmsecv': best_rmsecv,
            'removed': None
        }]

        # Backward elimination
        while len(remaining_intervals) > self.min_intervals:
            best_removal = None
            best_new_rmsecv = best_rmsecv

            for interval_id in remaining_intervals:
                # Try removing this interval
                test_intervals = [i for i in remaining_intervals if i != interval_id]
                test_indices = self._get_combined_indices(test_intervals)

                if len(test_indices) < 3:
                    continue

                rmsecv = self._evaluate_pls(X[:, test_indices], y, effective_cv)

                if rmsecv < best_new_rmsecv:
                    best_new_rmsecv = rmsecv
                    best_removal = {
                        'removed_id': interval_id,
                        'remaining': test_intervals,
                        'rmsecv': rmsecv
                    }

            if best_removal is None:
                # No improvement
                break

            # Apply removal
            remaining_intervals = best_removal['remaining']
            best_rmsecv = best_removal['rmsecv']

            self.exclusion_history_.append({
                'n_intervals': len(remaining_intervals),
                'rmsecv': best_rmsecv,
                'removed': best_removal['removed_id']
            })

        # Store final selection
        self.selected_intervals_ = remaining_intervals
        self.selected_indices_ = self._get_combined_indices(remaining_intervals)

        # Compute excluded indices
        all_indices = set(range(self.n_features_in_))
        self.excluded_indices_ = np.array(sorted(
            all_indices - set(self.selected_indices_)
        ))

        self.best_rmsecv_ = best_rmsecv

        return self

    def _create_intervals(self) -> List[Tuple[int, int, float, float]]:
        """Create equal-width intervals."""
        interval_size = self.n_features_in_ // self.n_intervals

        intervals = []
        for i in range(self.n_intervals):
            start_idx = i * interval_size
            if i == self.n_intervals - 1:
                end_idx = self.n_features_in_
            else:
                end_idx = (i + 1) * interval_size

            if end_idx > start_idx:
                start_wl = self.wavelengths_[start_idx]
                end_wl = self.wavelengths_[end_idx - 1]
                intervals.append((start_idx, end_idx, start_wl, end_wl))

        return intervals

    def _get_combined_indices(self, interval_ids: List[int]) -> np.ndarray:
        """Get combined indices for selected intervals."""
        indices = []
        for interval_id in sorted(interval_ids):
            start_idx, end_idx, _, _ = self.intervals_[interval_id]
            indices.extend(range(start_idx, end_idx))
        return np.array(indices, dtype=int)

    def _evaluate_pls(self, X: np.ndarray, y: np.ndarray, cv_folds: int) -> float:
        """Evaluate PLS model using cross-validation."""
        n_samples, n_features = X.shape
        n_components = min(10, n_features // 2, n_samples // 2)
        n_components = max(1, n_components)

        try:
            pls = PLSRegression(n_components=n_components, scale=False)
            y_pred = cross_val_predict(pls, X, y, cv=cv_folds)
            rmsecv = np.sqrt(mean_squared_error(y, y_pred))
            return rmsecv
        except Exception:
            return np.inf

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply region exclusion to spectral data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to transform

        Returns
        -------
        X_selected : ndarray, shape (n_samples, n_selected_wavelengths)
            Spectral data with excluded regions removed
        """
        check_is_fitted(self, ['selected_indices_', 'n_features_in_'])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        return X[:, self.selected_indices_]

    def get_selected_wavelengths(self) -> np.ndarray:
        """Get wavelengths that were kept (not excluded)."""
        check_is_fitted(self, ['selected_indices_', 'wavelengths_'])
        return self.wavelengths_[self.selected_indices_]

    def get_excluded_wavelengths(self) -> np.ndarray:
        """Get wavelengths that were excluded."""
        check_is_fitted(self, ['excluded_indices_', 'wavelengths_'])
        if len(self.excluded_indices_) == 0:
            return np.array([])
        return self.wavelengths_[self.excluded_indices_]

    def get_exclusion_ranges(self) -> List[Tuple[float, float]]:
        """
        Get wavelength ranges that were excluded.

        Returns
        -------
        ranges : list of (start_wl, end_wl)
            Excluded wavelength ranges
        """
        check_is_fitted(self, ['intervals_', 'selected_intervals_'])

        excluded_interval_ids = set(range(len(self.intervals_))) - set(self.selected_intervals_)

        ranges = []
        for interval_id in sorted(excluded_interval_ids):
            _, _, start_wl, end_wl = self.intervals_[interval_id]
            ranges.append((start_wl, end_wl))

        return ranges


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_contaminant_influence(
    X_uncontaminated: np.ndarray,
    X_contaminated: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    method: str = 'all',
    n_components: int = 3,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive contaminant influence analysis.

    Runs multiple analysis methods and returns combined results.

    Parameters
    ----------
    X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
        Spectral data from uncontaminated (clean) samples

    X_contaminated : array-like, shape (n_contaminated, n_wavelengths)
        Spectral data from contaminated samples

    wavelengths : array-like, shape (n_wavelengths,), optional
        Wavelength values. If None, uses indices.

    method : str, default='all'
        Analysis method(s) to run:
        - 'difference': Difference spectrum analysis only
        - 'epo' or 'estimated_epo': Estimated EPO analysis only
        - 'oplsda' or 'opls_da': OPLS-DA analysis only
        - 'glsw': GLSW analysis only
        - 'all': Run all methods

    n_components : int, default=3
        Number of components for EPO and OPLS-DA methods

    threshold : float, default=0.5
        Threshold for identifying exclusion regions (0-1)

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'wavelengths': input wavelengths
        - 'difference': DifferenceAnalyzer results (if run)
        - 'epo': EstimatedEPO results (if run)
        - 'oplsda': ContaminantOPLSDA results (if run)
        - 'glsw': ContaminantGLSW results (if run)
        - 'combined_influence': Average influence across all methods
        - 'exclusion_regions': Wavelength regions recommended for exclusion
    """
    X_contaminated = np.asarray(X_contaminated)
    X_uncontaminated = np.asarray(X_uncontaminated)

    # Handle optional wavelengths
    if wavelengths is None:
        wavelengths = np.arange(X_contaminated.shape[1])
    else:
        wavelengths = np.asarray(wavelengths)

    # Normalize method name
    method = method.lower().replace('-', '_').replace(' ', '_')
    if method == 'estimated_epo':
        method = 'epo'
    elif method == 'opls_da':
        method = 'oplsda'

    results = {'wavelengths': wavelengths}
    influences = []

    # Difference analysis
    if method in ['difference', 'all']:
        analyzer = DifferenceAnalyzer()
        analyzer.fit(X_contaminated, X_uncontaminated)

        results['difference'] = {
            'analyzer': analyzer,
            'spectrum': analyzer.get_difference_spectrum(),
            'influence': analyzer.get_normalized_influence(),
            'peak_regions': analyzer.identify_peak_regions(wavelengths, threshold=threshold)
        }
        influences.append(results['difference']['influence'])

    # EPO analysis
    if method in ['epo', 'all']:
        epo = EstimatedEPO(n_components=n_components, random_state=42)
        epo.fit_groups(X_contaminated, X_uncontaminated)

        results['epo'] = {
            'transformer': epo,
            'influence': epo.get_wavelength_influence(),
            'explained_variance': epo.get_explained_variance()
        }
        influences.append(results['epo']['influence'])

    # OPLS-DA analysis
    if method in ['oplsda', 'all']:
        oplsda = ContaminantOPLSDA(n_components=n_components)
        oplsda.fit(X_contaminated, X_uncontaminated)

        results['oplsda'] = {
            'model': oplsda,
            'influence': oplsda.get_wavelength_influence(),
            'vip_scores': oplsda.vip_scores_,
            'exclusion_regions': oplsda.get_exclusion_regions(wavelengths, threshold=threshold)
        }
        influences.append(results['oplsda']['influence'])

    # GLSW analysis
    if method in ['glsw', 'all']:
        glsw = ContaminantGLSW()
        glsw.fit_groups(X_contaminated, X_uncontaminated)

        results['glsw'] = {
            'transformer': glsw,
            'influence': glsw.get_contamination_influence(),
            'weights': glsw.get_feature_weights()
        }
        influences.append(results['glsw']['influence'])

    # Combine influences
    if len(influences) > 0:
        results['combined_influence'] = np.mean(np.vstack(influences), axis=0)

        # Find recommended exclusion regions using provided threshold
        above_threshold = results['combined_influence'] >= threshold

        regions = []
        in_region = False
        start_idx = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_region:
                in_region = True
                start_idx = i
            elif not above_threshold[i] and in_region:
                in_region = False
                if i - start_idx >= 3:
                    regions.append((wavelengths[start_idx], wavelengths[i-1]))

        if in_region and len(wavelengths) - start_idx >= 3:
            regions.append((wavelengths[start_idx], wavelengths[-1]))

        results['exclusion_regions'] = regions

    return results


# =============================================================================
# Multi-Contaminant Support
# =============================================================================

class MultiContaminantAnalyzer(BaseEstimator, TransformerMixin):
    """
    Analyze multiple contaminant types simultaneously.

    This class extends the basic contaminant analysis to handle cases with
    multiple different contaminant types, e.g.:
    - Clean bones
    - Bones with Glyptal
    - Bones with Paraloid B-72
    - Bones with both consolidants

    Each contaminant group is analyzed against the uncontaminated reference,
    and the influences are combined to identify regions affected by any contaminant.

    No Y values are required - this is purely unsupervised group analysis.

    Parameters
    ----------
    n_epo_components : int, default=2
        Number of EPO components per contaminant type

    estimation_method : str, default='pca_diff'
        Method for EPO estimation ('mean_diff', 'pca_diff', 'bootstrap')

    aggregation : {'max', 'mean', 'sum'}, default='max'
        How to combine influences from multiple contaminants:
        - 'max': Take maximum influence across contaminants (conservative)
        - 'mean': Average influence (balanced)
        - 'sum': Sum influences (aggressive)

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths

    contaminant_labels_ : list of str
        Labels for each contaminant type

    per_contaminant_influence_ : dict
        Wavelength influence for each contaminant type

    combined_influence_ : ndarray, shape (n_wavelengths,)
        Combined influence across all contaminants

    epo_transformers_ : dict
        EstimatedEPO transformer for each contaminant type

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import MultiContaminantAnalyzer
    >>>
    >>> # Sample data with multiple contaminant types
    >>> X_clean = np.random.randn(40, 100)
    >>> contaminant_groups = {
    ...     'Glyptal': np.random.randn(20, 100) + 0.3,
    ...     'Paraloid': np.random.randn(15, 100) + np.array([0]*50 + [0.5]*50),
    ...     'Both': np.random.randn(10, 100) + 0.5
    ... }
    >>> wavelengths = np.arange(1000, 1100)
    >>>
    >>> # Analyze all contaminants
    >>> analyzer = MultiContaminantAnalyzer()
    >>> analyzer.fit(X_clean, contaminant_groups)
    >>>
    >>> # Get combined influence
    >>> combined = analyzer.get_combined_influence()
    >>>
    >>> # Get per-contaminant breakdown
    >>> per_contam = analyzer.get_per_contaminant_influence()
    >>>
    >>> # Get exclusion regions considering all contaminants
    >>> regions = analyzer.get_exclusion_regions(wavelengths, threshold=0.5)
    """

    def __init__(
        self,
        n_epo_components: int = 2,
        estimation_method: str = 'pca_diff',
        aggregation: str = 'max',
        random_state: int = 42
    ):
        self.n_epo_components = n_epo_components
        self.estimation_method = estimation_method
        self.aggregation = aggregation
        self.random_state = random_state

    def fit(
        self,
        X_uncontaminated: np.ndarray,
        contaminant_groups: Dict[str, np.ndarray]
    ) -> 'MultiContaminantAnalyzer':
        """
        Fit analyzer on multiple contaminant groups.

        Parameters
        ----------
        X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
            Spectral data from uncontaminated (clean) samples.
            This is the reference group.

        contaminant_groups : dict of {str: array-like}
            Dictionary mapping contaminant labels to spectral data arrays.
            Each array should have shape (n_samples_i, n_wavelengths).

            Example:
            {
                'Glyptal': X_glyptal,         # (20, 100)
                'Paraloid': X_paraloid,       # (15, 100)
                'Both': X_both_contaminants   # (10, 100)
            }

        Returns
        -------
        self : MultiContaminantAnalyzer
        """
        # Validate inputs
        X_uncontaminated = check_array(X_uncontaminated, dtype=np.float64)
        self.n_features_in_ = X_uncontaminated.shape[1]

        if not contaminant_groups:
            raise ValueError("contaminant_groups cannot be empty")

        # Validate all contaminant groups
        validated_groups = {}
        for label, X_group in contaminant_groups.items():
            X_group = check_array(X_group, dtype=np.float64)
            if X_group.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Contaminant group '{label}' has {X_group.shape[1]} features, "
                    f"expected {self.n_features_in_}"
                )
            validated_groups[label] = X_group

        self.contaminant_labels_ = list(validated_groups.keys())
        self.X_uncontaminated_ = X_uncontaminated

        # Analyze each contaminant type
        self.per_contaminant_influence_ = {}
        self.epo_transformers_ = {}
        self.difference_spectra_ = {}

        for label, X_contaminated in validated_groups.items():
            # Run EPO-based analysis
            epo = EstimatedEPO(
                n_components=self.n_epo_components,
                estimation_method=self.estimation_method,
                random_state=self.random_state
            )
            epo.fit_groups(X_contaminated, X_uncontaminated)
            self.epo_transformers_[label] = epo

            # Get wavelength influence
            self.per_contaminant_influence_[label] = epo.get_wavelength_influence()

            # Also store difference spectrum for visualization
            diff_analyzer = DifferenceAnalyzer()
            diff_analyzer.fit(X_contaminated, X_uncontaminated)
            self.difference_spectra_[label] = diff_analyzer.get_difference_spectrum()

        # Combine influences based on aggregation method
        influence_matrix = np.vstack([
            self.per_contaminant_influence_[label]
            for label in self.contaminant_labels_
        ])

        if self.aggregation == 'max':
            self.combined_influence_ = np.max(influence_matrix, axis=0)
        elif self.aggregation == 'mean':
            self.combined_influence_ = np.mean(influence_matrix, axis=0)
        elif self.aggregation == 'sum':
            raw_sum = np.sum(influence_matrix, axis=0)
            self.combined_influence_ = raw_sum / (np.max(raw_sum) + 1e-10)
        else:
            raise ValueError(
                f"aggregation must be 'max', 'mean', or 'sum', got '{self.aggregation}'"
            )

        return self

    def get_combined_influence(self) -> np.ndarray:
        """
        Get combined contaminant influence across all types.

        Returns
        -------
        influence : ndarray, shape (n_wavelengths,)
            Combined influence score (0-1)
        """
        check_is_fitted(self, 'combined_influence_')
        return self.combined_influence_.copy()

    def get_per_contaminant_influence(self) -> Dict[str, np.ndarray]:
        """
        Get influence breakdown by contaminant type.

        Returns
        -------
        influences : dict of {str: ndarray}
            Influence array for each contaminant type
        """
        check_is_fitted(self, 'per_contaminant_influence_')
        return {k: v.copy() for k, v in self.per_contaminant_influence_.items()}

    def get_difference_spectra(self) -> Dict[str, np.ndarray]:
        """
        Get difference spectra for each contaminant type.

        Returns
        -------
        spectra : dict of {str: ndarray}
            Difference spectrum for each contaminant type
        """
        check_is_fitted(self, 'difference_spectra_')
        return {k: v.copy() for k, v in self.difference_spectra_.items()}

    def get_exclusion_regions(
        self,
        wavelengths: np.ndarray,
        threshold: float = 0.5,
        min_width: int = 3
    ) -> List[Tuple[float, float, List[str]]]:
        """
        Get wavelength regions to exclude with contributing contaminants.

        Parameters
        ----------
        wavelengths : array-like, shape (n_wavelengths,)
            Wavelength values

        threshold : float, default=0.5
            Combined influence threshold for exclusion

        min_width : int, default=3
            Minimum region width

        Returns
        -------
        regions : list of (start_wl, end_wl, contaminants)
            Excluded regions with list of contributing contaminant types
        """
        check_is_fitted(self, 'combined_influence_')
        wavelengths = np.asarray(wavelengths)

        if len(wavelengths) != self.n_features_in_:
            raise ValueError(
                f"wavelengths length ({len(wavelengths)}) must match "
                f"n_features_in_ ({self.n_features_in_})"
            )

        above_threshold = self.combined_influence_ >= threshold

        regions = []
        in_region = False
        start_idx = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_region:
                in_region = True
                start_idx = i
            elif not above_threshold[i] and in_region:
                in_region = False
                if i - start_idx >= min_width:
                    # Identify which contaminants contribute to this region
                    contributors = self._identify_contributors(
                        start_idx, i, threshold * 0.5
                    )
                    regions.append((
                        wavelengths[start_idx],
                        wavelengths[i-1],
                        contributors
                    ))

        if in_region and len(wavelengths) - start_idx >= min_width:
            contributors = self._identify_contributors(
                start_idx, len(wavelengths), threshold * 0.5
            )
            regions.append((
                wavelengths[start_idx],
                wavelengths[-1],
                contributors
            ))

        return regions

    def _identify_contributors(
        self,
        start_idx: int,
        end_idx: int,
        threshold: float
    ) -> List[str]:
        """Identify which contaminants contribute to a region."""
        contributors = []
        for label in self.contaminant_labels_:
            region_influence = self.per_contaminant_influence_[label][start_idx:end_idx]
            if np.max(region_influence) >= threshold:
                contributors.append(label)
        return contributors

    def transform(self, X: np.ndarray, remove_all: bool = True) -> np.ndarray:
        """
        Transform data by removing all estimated contaminant signals.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to transform

        remove_all : bool, default=True
            If True, sequentially apply all EPO transformers.
            If False, only return mean-centered data.

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_wavelengths)
            Transformed data with contaminant signals removed
        """
        check_is_fitted(self, 'epo_transformers_')
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        if not remove_all:
            return X - np.mean(self.X_uncontaminated_, axis=0)

        # Sequentially apply all EPO transformers
        X_corrected = X.copy()
        for label in self.contaminant_labels_:
            # Note: Each transformer centers the data, so we need to handle this
            epo = self.epo_transformers_[label]
            X_corrected = epo.transform(X_corrected + epo.X_mean_)

        return X_corrected


class MultiGroupEPO(BaseEstimator, TransformerMixin):
    """
    EPO for removing multiple interferent types simultaneously.

    Builds a combined interferent subspace from multiple contaminant groups
    and removes them in a single transformation step.

    This is more mathematically rigorous than sequential application of
    individual EPO transformers.

    Parameters
    ----------
    n_components_per_group : int, default=2
        Number of EPO components to extract per contaminant group

    n_total_components : int or None, default=None
        Total number of components for final combined EPO.
        If None, uses n_components_per_group * n_groups.

    center : bool, default=True
        Whether to mean-center data

    svd_tol : float, default=1e-8
        Tolerance for SVD truncation

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths

    combined_interferent_library_ : ndarray
        Combined interferent library from all groups

    interferent_components_ : ndarray
        Principal components of combined interferent subspace

    P_orth_ : ndarray
        Orthogonal projection matrix

    group_labels_ : list of str
        Labels for each contaminant group

    per_group_variance_ : dict
        Variance contribution from each group

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import MultiGroupEPO
    >>>
    >>> # Sample data
    >>> X_clean = np.random.randn(40, 100)
    >>> contaminant_groups = {
    ...     'Glyptal': np.random.randn(20, 100) + 0.3,
    ...     'Paraloid': np.random.randn(15, 100) + np.array([0]*50 + [0.5]*50)
    ... }
    >>>
    >>> # Fit combined EPO
    >>> epo = MultiGroupEPO(n_components_per_group=2)
    >>> epo.fit(X_clean, contaminant_groups)
    >>>
    >>> # Transform all data at once
    >>> X_all = np.vstack([X_clean] + list(contaminant_groups.values()))
    >>> X_corrected = epo.transform(X_all)
    """

    def __init__(
        self,
        n_components_per_group: int = 2,
        n_total_components: Optional[int] = None,
        center: bool = True,
        svd_tol: float = 1e-8
    ):
        self.n_components_per_group = n_components_per_group
        self.n_total_components = n_total_components
        self.center = center
        self.svd_tol = svd_tol

    def fit(
        self,
        X_uncontaminated: np.ndarray,
        contaminant_groups: Dict[str, np.ndarray]
    ) -> 'MultiGroupEPO':
        """
        Fit EPO using combined interferent library from all groups.

        Parameters
        ----------
        X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
            Reference (clean) samples

        contaminant_groups : dict of {str: array-like}
            Dictionary mapping contaminant labels to spectral data

        Returns
        -------
        self : MultiGroupEPO
        """
        # Validate inputs
        X_uncontaminated = check_array(X_uncontaminated, dtype=np.float64)
        self.n_features_in_ = X_uncontaminated.shape[1]

        if not contaminant_groups:
            raise ValueError("contaminant_groups cannot be empty")

        # Validate all groups
        validated_groups = {}
        for label, X_group in contaminant_groups.items():
            X_group = check_array(X_group, dtype=np.float64)
            if X_group.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Group '{label}' has {X_group.shape[1]} features, "
                    f"expected {self.n_features_in_}"
                )
            validated_groups[label] = X_group

        self.group_labels_ = list(validated_groups.keys())
        n_groups = len(self.group_labels_)

        # Compute overall mean for centering
        all_samples = [X_uncontaminated] + list(validated_groups.values())
        X_all = np.vstack(all_samples)

        if self.center:
            self.X_mean_ = np.mean(X_all, axis=0)
        else:
            self.X_mean_ = np.zeros(self.n_features_in_)

        # Reference mean
        ref_mean = np.mean(X_uncontaminated, axis=0)

        # Build combined interferent library
        # Each group contributes difference vectors
        interferent_library_parts = []
        self.per_group_variance_ = {}

        for label, X_contaminated in validated_groups.items():
            # Compute mean difference
            cont_mean = np.mean(X_contaminated, axis=0)
            diff = cont_mean - ref_mean

            # Build pseudo-interferent spectra for this group
            # Use bootstrap-like approach
            n_pseudo = max(self.n_components_per_group * 2, 5)
            diff_std = np.std(diff) if np.std(diff) > 1e-10 else 1.0

            group_library = [diff]
            rng = np.random.RandomState(hash(label) % 2**31)
            for _ in range(n_pseudo - 1):
                noise = rng.randn(len(diff)) * diff_std * 0.1
                group_library.append(diff + noise)

            group_array = np.array(group_library)
            interferent_library_parts.append(group_array)

            # Track variance contribution
            self.per_group_variance_[label] = np.var(diff)

        # Combine all interferent libraries
        self.combined_interferent_library_ = np.vstack(interferent_library_parts)

        # Center the combined library
        lib_mean = np.mean(self.combined_interferent_library_, axis=0)
        lib_centered = self.combined_interferent_library_ - lib_mean

        # SVD to get combined interferent subspace
        try:
            U, S, Vt = np.linalg.svd(lib_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            raise ValueError("SVD failed on combined interferent library")

        # Determine number of components
        n_valid = np.sum(S > self.svd_tol)
        if self.n_total_components is None:
            n_components = min(
                self.n_components_per_group * n_groups,
                n_valid,
                self.n_features_in_ - 1
            )
        else:
            n_components = min(self.n_total_components, n_valid, self.n_features_in_ - 1)

        if n_components < 1:
            n_components = 1

        self.n_components_ = n_components

        # Store components
        V = Vt.T
        self.interferent_components_ = V[:, :n_components]

        # Store explained variance
        total_var = np.sum(S**2)
        if total_var > 0:
            self.explained_variance_ = (S[:n_components]**2) / total_var
        else:
            self.explained_variance_ = np.zeros(n_components)

        # Build projection matrix
        V_comp = self.interferent_components_
        self.P_orth_ = np.eye(self.n_features_in_) - V_comp @ V_comp.T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply multi-group EPO transformation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data to transform

        Returns
        -------
        X_corrected : ndarray, shape (n_samples, n_wavelengths)
            Transformed data with all interferent signals removed
        """
        check_is_fitted(self, ['P_orth_', 'X_mean_'])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        X_centered = X - self.X_mean_
        X_corrected = X_centered @ self.P_orth_

        return X_corrected

    def get_wavelength_influence(self) -> np.ndarray:
        """
        Get combined interferent influence at each wavelength.

        Returns
        -------
        influence : ndarray, shape (n_wavelengths,)
            Influence score (0-1)
        """
        check_is_fitted(self, 'interferent_components_')

        influence = np.sum(self.interferent_components_**2, axis=1)
        max_inf = np.max(influence)
        if max_inf > 1e-10:
            influence = influence / max_inf

        return influence

    def get_explained_variance(self) -> np.ndarray:
        """Get variance explained by each component."""
        check_is_fitted(self, 'explained_variance_')
        return self.explained_variance_.copy()


class MultiContaminantGLSW(BaseEstimator, TransformerMixin):
    """
    GLSW weighting accounting for multiple contaminant types.

    Computes wavelength weights that down-weight regions affected by
    ANY of the specified contaminants.

    Parameters
    ----------
    regularization : float, default=1e-6
        Regularization for numerical stability

    influence_power : float, default=1.0
        Power for influence-based weighting

    min_weight : float, default=0.1
        Minimum weight to assign

    aggregation : {'max', 'mean'}, default='max'
        How to combine influences:
        - 'max': Most conservative (any contaminant causes down-weighting)
        - 'mean': Balanced approach

    Attributes
    ----------
    n_features_in_ : int
        Number of wavelengths

    feature_weights_ : ndarray, shape (n_wavelengths,)
        Combined weights

    per_contaminant_influence_ : dict
        Influence for each contaminant type

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_predict.contaminant_analysis import MultiContaminantGLSW
    >>>
    >>> X_clean = np.random.randn(40, 100)
    >>> contaminant_groups = {
    ...     'Glyptal': np.random.randn(20, 100) + 0.3,
    ...     'Paraloid': np.random.randn(15, 100) + np.array([0]*50 + [0.5]*50)
    ... }
    >>>
    >>> glsw = MultiContaminantGLSW()
    >>> glsw.fit(X_clean, contaminant_groups)
    >>>
    >>> X_all = np.vstack([X_clean] + list(contaminant_groups.values()))
    >>> X_weighted = glsw.transform(X_all)
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        influence_power: float = 1.0,
        min_weight: float = 0.1,
        aggregation: str = 'max'
    ):
        self.regularization = regularization
        self.influence_power = influence_power
        self.min_weight = min_weight
        self.aggregation = aggregation

    def fit(
        self,
        X_uncontaminated: np.ndarray,
        contaminant_groups: Dict[str, np.ndarray]
    ) -> 'MultiContaminantGLSW':
        """
        Fit GLSW weights for multiple contaminant types.

        Parameters
        ----------
        X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
            Reference (clean) samples

        contaminant_groups : dict of {str: array-like}
            Dictionary mapping contaminant labels to spectral data

        Returns
        -------
        self : MultiContaminantGLSW
        """
        X_uncontaminated = check_array(X_uncontaminated, dtype=np.float64)
        self.n_features_in_ = X_uncontaminated.shape[1]

        if not contaminant_groups:
            raise ValueError("contaminant_groups cannot be empty")

        # Reference statistics
        var_ref = np.var(X_uncontaminated, axis=0)
        mean_ref = np.mean(X_uncontaminated, axis=0)

        # Compute influence for each contaminant
        self.per_contaminant_influence_ = {}
        self.contaminant_labels_ = []

        for label, X_contaminated in contaminant_groups.items():
            X_contaminated = check_array(X_contaminated, dtype=np.float64)
            if X_contaminated.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Group '{label}' has {X_contaminated.shape[1]} features, "
                    f"expected {self.n_features_in_}"
                )

            self.contaminant_labels_.append(label)

            # Compute influence for this contaminant
            var_cont = np.var(X_contaminated, axis=0)
            mean_cont = np.mean(X_contaminated, axis=0)

            var_ratio = (var_cont + self.regularization) / (var_ref + self.regularization)
            abs_diff = np.abs(mean_cont - mean_ref)

            var_ratio_norm = var_ratio / (np.max(var_ratio) + 1e-10)
            abs_diff_norm = abs_diff / (np.max(abs_diff) + 1e-10)

            influence = 0.5 * var_ratio_norm + 0.5 * abs_diff_norm
            self.per_contaminant_influence_[label] = influence

        # Combine influences
        influence_matrix = np.vstack([
            self.per_contaminant_influence_[label]
            for label in self.contaminant_labels_
        ])

        if self.aggregation == 'max':
            combined_influence = np.max(influence_matrix, axis=0)
        else:  # mean
            combined_influence = np.mean(influence_matrix, axis=0)

        self.combined_influence_ = combined_influence

        # Compute weights
        weights = 1.0 / (combined_influence**self.influence_power + self.regularization)
        weights = np.maximum(weights, self.min_weight)
        self.feature_weights_ = weights / np.mean(weights)
        self.W_sqrt_ = np.sqrt(self.feature_weights_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply GLSW weighting."""
        check_is_fitted(self, ['W_sqrt_', 'n_features_in_'])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        return X * self.W_sqrt_

    def get_feature_weights(self) -> np.ndarray:
        """Get combined feature weights."""
        check_is_fitted(self, 'feature_weights_')
        return self.feature_weights_.copy()

    def get_per_contaminant_influence(self) -> Dict[str, np.ndarray]:
        """Get influence breakdown by contaminant type."""
        check_is_fitted(self, 'per_contaminant_influence_')
        return {k: v.copy() for k, v in self.per_contaminant_influence_.items()}


# =============================================================================
# Multi-Contaminant Convenience Function
# =============================================================================

def analyze_multiple_contaminants(
    X_uncontaminated: np.ndarray,
    contaminant_groups: Dict[str, np.ndarray],
    wavelengths: Optional[np.ndarray] = None,
    method: str = 'all',
    n_components: int = 2,
    threshold: float = 0.5,
    aggregation: str = 'max'
) -> Dict[str, Any]:
    """
    Comprehensive analysis of multiple contaminant types.

    Parameters
    ----------
    X_uncontaminated : array-like, shape (n_uncontaminated, n_wavelengths)
        Reference (clean) samples

    contaminant_groups : dict of {str: array-like}
        Dictionary mapping contaminant labels to spectral data.

        Example:
        {
            'Glyptal': X_glyptal,
            'Paraloid B-72': X_paraloid,
            'Epoxy': X_epoxy
        }

    wavelengths : array-like, shape (n_wavelengths,), optional
        Wavelength values. If None, uses indices.

    method : str, default='all'
        Method to run. Can be a single method or 'all':
        - 'difference': Difference spectra for each contaminant
        - 'epo' or 'estimated_epo': Multi-group EPO analysis
        - 'oplsda' or 'opls_da': OPLS-DA analysis
        - 'glsw': Multi-contaminant GLSW
        - 'all': Run all methods

    n_components : int, default=2
        Number of components for EPO and OPLS-DA methods

    threshold : float, default=0.5
        Threshold for identifying exclusion regions (0-1)

    aggregation : {'max', 'mean', 'sum'}, default='max'
        How to combine influences from multiple contaminants

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'wavelengths': input wavelengths
        - 'contaminant_labels': list of contaminant names
        - 'difference': per-contaminant difference spectra
        - 'epo': MultiGroupEPO results
        - 'glsw': MultiContaminantGLSW results
        - 'combined_influence': overall combined influence
        - 'per_contaminant_influence': influence breakdown
        - 'exclusion_regions': regions with contributing contaminants
    """
    X_uncontaminated = np.asarray(X_uncontaminated)

    # Handle optional wavelengths
    n_features = X_uncontaminated.shape[1]
    if wavelengths is None:
        wavelengths = np.arange(n_features)
    else:
        wavelengths = np.asarray(wavelengths)

    # Normalize method name
    method = method.lower().replace('-', '_').replace(' ', '_')
    if method == 'estimated_epo':
        method = 'epo'
    elif method == 'opls_da':
        method = 'oplsda'

    # Convert single method to list
    if method == 'all':
        methods = ['difference', 'epo', 'glsw']
    else:
        methods = [method]

    results = {
        'wavelengths': wavelengths,
        'contaminant_labels': list(contaminant_groups.keys())
    }

    # Difference spectra
    if 'difference' in methods:
        diff_spectra = {}
        for label, X_cont in contaminant_groups.items():
            analyzer = DifferenceAnalyzer()
            analyzer.fit(X_cont, X_uncontaminated)
            diff_spectra[label] = {
                'spectrum': analyzer.get_difference_spectrum(),
                'influence': analyzer.get_normalized_influence()
            }
        results['difference'] = diff_spectra

    # Multi-group EPO
    if 'epo' in methods:
        epo = MultiGroupEPO(n_components_per_group=n_components)
        epo.fit(X_uncontaminated, contaminant_groups)
        results['epo'] = {
            'transformer': epo,
            'influence': epo.get_wavelength_influence(),
            'explained_variance': epo.get_explained_variance(),
            'per_group_variance': epo.per_group_variance_
        }

    # Multi-contaminant GLSW (supports aggregation parameter)
    if 'glsw' in methods:
        glsw = MultiContaminantGLSW(aggregation=aggregation)
        glsw.fit(X_uncontaminated, contaminant_groups)
        results['glsw'] = {
            'transformer': glsw,
            'weights': glsw.get_feature_weights(),
            'per_contaminant_influence': glsw.get_per_contaminant_influence()
        }

    # Combined analysis (supports aggregation parameter)
    analyzer = MultiContaminantAnalyzer(n_epo_components=n_components, aggregation=aggregation, random_state=42)
    analyzer.fit(X_uncontaminated, contaminant_groups)

    results['combined_influence'] = analyzer.get_combined_influence()
    results['per_contaminant_influence'] = analyzer.get_per_contaminant_influence()
    results['exclusion_regions'] = analyzer.get_exclusion_regions(
        wavelengths, threshold=threshold
    )

    return results
