"""
Outlier Detection for Spectral Predict v3.

This module provides comprehensive outlier detection methods for spectral data
including PCA-based detection, Q-residuals, Mahalanobis distance, and reference
value consistency checks. Results are returned as dataclasses for type safety.

Methods based on standard chemometric outlier detection approaches:
- Hotelling T²: Distance in principal component space
- Q-residuals (SPE): Distance from PCA model (reconstruction error)
- Mahalanobis distance: Multivariate distance with covariance weighting
- Y-value checks: Statistical and range-based outlier detection

References
----------
- Hotelling, H. (1931). The generalization of Student's ratio.
- Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals
  associated with principal component analysis.
- De Maesschalck, R., et al. (2000). The Mahalanobis distance.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from sklearn.decomposition import PCA
from scipy import stats


# ============================================================================
# RESULT DATACLASSES
# ============================================================================

@dataclass
class PCAOutlierResult:
    """
    Result of PCA-based outlier detection.

    Attributes
    ----------
    pca_model : PCA
        Fitted sklearn PCA model
    scores : np.ndarray
        PC scores (samples × n_components)
    loadings : np.ndarray
        PC loadings (wavelengths × n_components)
    variance_explained : np.ndarray
        Fraction of variance explained by each PC
    hotelling_t2 : np.ndarray
        Hotelling T² statistic for each sample
    t2_threshold : float
        95% confidence threshold for T²
    outlier_flags : np.ndarray
        Boolean array (True = outlier)
    n_outliers : int
        Count of outliers detected
    outlier_indices : np.ndarray
        Array indices of outlier samples
    """
    pca_model: Any  # sklearn PCA object
    scores: np.ndarray
    loadings: np.ndarray
    variance_explained: np.ndarray
    hotelling_t2: np.ndarray
    t2_threshold: float
    outlier_flags: np.ndarray
    n_outliers: int
    outlier_indices: np.ndarray


@dataclass
class QResidualResult:
    """
    Result of Q-residuals (SPE) outlier detection.

    Attributes
    ----------
    q_residuals : np.ndarray
        Q-residual (SPE) for each sample
    q_threshold : float
        95th percentile threshold
    outlier_flags : np.ndarray
        Boolean array (True = outlier)
    n_outliers : int
        Count of outliers detected
    outlier_indices : np.ndarray
        Array indices of outlier samples
    """
    q_residuals: np.ndarray
    q_threshold: float
    outlier_flags: np.ndarray
    n_outliers: int
    outlier_indices: np.ndarray


@dataclass
class MahalanobisResult:
    """
    Result of Mahalanobis distance outlier detection.

    Attributes
    ----------
    distances : np.ndarray
        Mahalanobis distance for each sample
    median : float
        Median distance
    mad : float
        Median absolute deviation
    threshold : float
        3× MAD threshold (median + 3*MAD)
    outlier_flags : np.ndarray
        Boolean array (True = outlier)
    n_outliers : int
        Count of outliers detected
    outlier_indices : np.ndarray
        Array indices of outlier samples
    """
    distances: np.ndarray
    median: float
    mad: float
    threshold: float
    outlier_flags: np.ndarray
    n_outliers: int
    outlier_indices: np.ndarray


@dataclass
class YConsistencyResult:
    """
    Result of Y-value consistency checks.

    Attributes
    ----------
    mean : float or None
        Mean of reference values (None for categorical)
    std : float or None
        Standard deviation (None for categorical)
    median : float or None
        Median value (None for categorical)
    min : float or None
        Minimum value (None for categorical)
    max : float or None
        Maximum value (None for categorical)
    z_scores : np.ndarray
        Z-score for each sample
    z_outliers : np.ndarray
        Boolean array for samples with |z| > 3
    range_outliers : np.ndarray
        Boolean array for samples outside bounds
    all_outliers : np.ndarray
        Boolean array combining z_outliers and range_outliers
    n_outliers : int
        Total count of outliers
    outlier_indices : np.ndarray
        Array indices of outlier samples
    is_categorical : bool
        True for categorical data
    unique_values : list or None
        List of unique class labels (categorical only)
    value_counts : list or None
        Count of samples per class (categorical only)
    frequencies : list or None
        Frequency proportion of each class (categorical only)
    """
    mean: Optional[float]
    std: Optional[float]
    median: Optional[float]
    min: Optional[float]
    max: Optional[float]
    z_scores: np.ndarray
    z_outliers: np.ndarray
    range_outliers: np.ndarray
    all_outliers: np.ndarray
    n_outliers: int
    outlier_indices: np.ndarray
    is_categorical: bool = False
    unique_values: Optional[List[Any]] = None
    value_counts: Optional[List[int]] = None
    frequencies: Optional[List[float]] = None


@dataclass
class OutlierReport:
    """
    Comprehensive outlier detection report combining all methods.

    Attributes
    ----------
    pca : PCAOutlierResult
        PCA outlier detection results
    q_residuals : QResidualResult
        Q-residuals outlier detection results
    mahalanobis : MahalanobisResult
        Mahalanobis distance results
    y_consistency : YConsistencyResult
        Y data consistency check results
    combined_flags : np.ndarray
        Boolean array for high-confidence outliers (2+ methods)
    total_flags_per_sample : np.ndarray
        Number of methods flagging each sample
    high_confidence_indices : np.ndarray
        Samples flagged by 3+ methods
    moderate_confidence_indices : np.ndarray
        Samples flagged by exactly 2 methods
    low_confidence_indices : np.ndarray
        Samples flagged by exactly 1 method
    """
    pca: PCAOutlierResult
    q_residuals: QResidualResult
    mahalanobis: MahalanobisResult
    y_consistency: YConsistencyResult
    combined_flags: np.ndarray
    total_flags_per_sample: np.ndarray
    high_confidence_indices: np.ndarray
    moderate_confidence_indices: np.ndarray
    low_confidence_indices: np.ndarray


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def run_pca_outlier_detection(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_components: int = 5
) -> PCAOutlierResult:
    """
    Perform PCA-based outlier detection on spectral data.

    Computes principal component scores and Hotelling T² statistic for each
    sample. The T² statistic measures the distance of each sample from the
    center of the principal component space, accounting for variance in each
    direction.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (samples × wavelengths)
    y : np.ndarray, optional
        Reference values (not used in detection, for plotting only)
    n_components : int, default=5
        Number of principal components to compute

    Returns
    -------
    PCAOutlierResult
        Dataclass containing PCA outlier metrics

    Notes
    -----
    The Hotelling T² statistic is computed as:
        T² = score · inv(cov) · score.T

    The 95% threshold is based on the F-distribution:
        T²_threshold = (p(n-1)/(n-p)) * F(α, p, n-p)
    where p = n_components, n = n_samples, α = 0.05

    Edge cases handled:
    - If covariance matrix is singular, regularization is applied
    - If n_components >= n_samples, it is clipped to n_samples - 1
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    # Clip n_components to valid range
    n_components = min(n_components, n_samples - 1, n_features)

    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    # Compute Hotelling T²
    if n_components == 1:
        # Special case for single component
        cov_matrix = np.var(scores)
        if cov_matrix < 1e-10:
            cov_matrix = 1e-10
        inv_cov = 1.0 / cov_matrix
    else:
        cov_matrix = np.cov(scores.T)

        # Handle singular covariance matrix
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Add small regularization to diagonal
            cov_matrix = cov_matrix + np.eye(n_components) * 1e-6
            inv_cov = np.linalg.inv(cov_matrix)

    t2_values = []
    if n_components == 1:
        # For single component, T² = (score - 0)² / variance
        for score in scores:
            t2 = (score[0] ** 2) * inv_cov
            t2_values.append(t2)
    else:
        for score in scores:
            t2 = score @ inv_cov @ score.T
            t2_values.append(t2)

    t2_values = np.array(t2_values)

    # Compute 95% threshold using F-distribution
    alpha = 0.05
    if n_samples > n_components:
        t2_threshold = (n_components * (n_samples - 1) / (n_samples - n_components) *
                        stats.f.ppf(1 - alpha, n_components, n_samples - n_components))
    else:
        # If n_samples <= n_components, use chi-squared approximation
        t2_threshold = stats.chi2.ppf(1 - alpha, n_components)

    outlier_flags = t2_values > t2_threshold

    return PCAOutlierResult(
        pca_model=pca,
        scores=scores,
        loadings=pca.components_.T,
        variance_explained=pca.explained_variance_ratio_,
        hotelling_t2=t2_values,
        t2_threshold=t2_threshold,
        outlier_flags=outlier_flags,
        n_outliers=int(np.sum(outlier_flags)),
        outlier_indices=np.where(outlier_flags)[0]
    )


def compute_q_residuals(
    X: np.ndarray,
    pca_model: Any,
    n_components: Optional[int] = None
) -> QResidualResult:
    """
    Compute Q-residuals (SPE - Squared Prediction Error) for outlier detection.

    Q-residuals measure the reconstruction error when projecting data into the
    principal component space and back. High Q-residuals indicate samples that
    are poorly represented by the PCA model.

    Parameters
    ----------
    X : np.ndarray
        Original spectral data
    pca_model : PCA
        Fitted PCA model
    n_components : int, optional
        Number of components to use for reconstruction. If None, uses all
        components from the fitted model.

    Returns
    -------
    QResidualResult
        Dataclass containing Q-residual metrics

    Notes
    -----
    Q-residual is computed as:
        Q = sum((X - X_reconstructed)²)

    The threshold uses the 95th percentile of the Q-residual distribution.
    """
    X = np.asarray(X)

    if n_components is None:
        n_components = pca_model.n_components_
    else:
        # Clip to available components
        n_components = min(n_components, pca_model.n_components_)

    # Project data to PC space and back
    scores = pca_model.transform(X)[:, :n_components]
    X_reconstructed = scores @ pca_model.components_[:n_components, :]

    # Add back the mean (PCA centers the data)
    X_reconstructed += pca_model.mean_

    # Compute reconstruction error
    residuals = X - X_reconstructed
    q_residuals = np.sum(residuals ** 2, axis=1)

    # 95th percentile threshold
    q_threshold = np.percentile(q_residuals, 95)

    outlier_flags = q_residuals > q_threshold

    return QResidualResult(
        q_residuals=q_residuals,
        q_threshold=q_threshold,
        outlier_flags=outlier_flags,
        n_outliers=int(np.sum(outlier_flags)),
        outlier_indices=np.where(outlier_flags)[0]
    )


def compute_mahalanobis_distance(scores: np.ndarray) -> MahalanobisResult:
    """
    Compute Mahalanobis distance for each sample in PCA space.

    The Mahalanobis distance is a multivariate measure of how far each sample
    is from the center of the distribution, accounting for correlations between
    variables and their variances.

    Parameters
    ----------
    scores : np.ndarray
        PCA scores (samples × n_components)

    Returns
    -------
    MahalanobisResult
        Dataclass containing Mahalanobis distance metrics

    Notes
    -----
    Mahalanobis distance is computed as:
        D = sqrt((x - μ)' Σ⁻¹ (x - μ))
    where μ is the mean and Σ is the covariance matrix.

    The threshold uses 3× median absolute deviation (MAD), which is robust
    to outliers in the distance distribution itself.
    """
    # Ensure scores is 2D
    scores = np.asarray(scores)
    if len(scores.shape) == 1:
        scores = scores.reshape(-1, 1)

    # Compute covariance and inverse
    cov_matrix = np.cov(scores.T)

    # Handle singular covariance (e.g., single component or perfectly correlated)
    if scores.shape[1] == 1:
        # For single component, use variance directly
        inv_cov = np.array([[1.0 / (cov_matrix + 1e-10)]])
    else:
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Add small regularization to diagonal
            cov_matrix += np.eye(scores.shape[1]) * 1e-6
            inv_cov = np.linalg.inv(cov_matrix)

    # Center of the distribution
    mean = np.mean(scores, axis=0)

    # Mahalanobis distance for each sample
    distances = []
    for score in scores:
        diff = score - mean
        distance = np.sqrt(diff @ inv_cov @ diff.T)
        distances.append(distance)

    distances = np.array(distances)

    # Threshold: 3× median absolute deviation (MAD)
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))

    # Avoid division by zero
    if mad < 1e-10:
        mad = 1e-10

    threshold = median + 3 * mad

    outlier_flags = distances > threshold

    return MahalanobisResult(
        distances=distances,
        median=median,
        mad=mad,
        threshold=threshold,
        outlier_flags=outlier_flags,
        n_outliers=int(np.sum(outlier_flags)),
        outlier_indices=np.where(outlier_flags)[0]
    )


def check_y_data_consistency(
    y: np.ndarray,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None
) -> YConsistencyResult:
    """
    Check reference data for outliers and inconsistencies.

    Performs statistical checks on reference values to identify potential
    data entry errors, mislabeled samples, or values outside chemically
    reasonable ranges. For categorical data, returns class distribution
    instead of outlier detection.

    Parameters
    ----------
    y : np.ndarray
        Reference values (numeric or categorical)
    lower_bound : float, optional
        Minimum chemically reasonable value (ignored for categorical)
    upper_bound : float, optional
        Maximum chemically reasonable value (ignored for categorical)

    Returns
    -------
    YConsistencyResult
        Dataclass containing consistency check results

    Notes
    -----
    Categorical data detection:
    - Data is considered categorical if dtype is object or non-numeric
    - For categorical data, no outliers are flagged
    - Class distribution statistics are provided instead

    Numeric data outlier detection:
    - Z-score detection uses the ±3σ rule
    - Range checks are optional based on domain knowledge
    """
    y = np.asarray(y)

    # Check if data is categorical (non-numeric)
    is_categorical = (y.dtype == object or
                     not np.issubdtype(y.dtype, np.number))

    if is_categorical:
        # For categorical data, return class distribution
        unique_values, counts = np.unique(y, return_counts=True)

        # Calculate class frequencies
        total_samples = len(y)
        frequencies = counts / total_samples

        return YConsistencyResult(
            mean=None,
            std=None,
            median=None,
            min=None,
            max=None,
            z_scores=np.zeros(len(y), dtype=float),
            z_outliers=np.zeros(len(y), dtype=bool),
            range_outliers=np.zeros(len(y), dtype=bool),
            all_outliers=np.zeros(len(y), dtype=bool),
            n_outliers=0,
            outlier_indices=np.array([], dtype=int),
            is_categorical=True,
            unique_values=unique_values.tolist(),
            value_counts=counts.tolist(),
            frequencies=frequencies.tolist()
        )

    # Compute statistics
    mean = np.mean(y)
    std = np.std(y)
    median = np.median(y)
    min_val = np.min(y)
    max_val = np.max(y)

    # Z-scores (handle zero std)
    if std < 1e-10:
        z_scores = np.zeros_like(y)
    else:
        z_scores = (y - mean) / std

    z_outliers = np.abs(z_scores) > 3

    # Range check
    range_outliers = np.zeros(len(y), dtype=bool)
    if lower_bound is not None:
        range_outliers |= y < lower_bound
    if upper_bound is not None:
        range_outliers |= y > upper_bound

    # Combine
    all_outliers = z_outliers | range_outliers

    return YConsistencyResult(
        mean=float(mean),
        std=float(std),
        median=float(median),
        min=float(min_val),
        max=float(max_val),
        z_scores=z_scores,
        z_outliers=z_outliers,
        range_outliers=range_outliers,
        all_outliers=all_outliers,
        n_outliers=int(np.sum(all_outliers)),
        outlier_indices=np.where(all_outliers)[0]
    )


def generate_outlier_report(
    X: np.ndarray,
    y: np.ndarray,
    n_pca_components: int = 5,
    y_lower_bound: Optional[float] = None,
    y_upper_bound: Optional[float] = None
) -> OutlierReport:
    """
    Comprehensive outlier detection report combining all methods.

    Runs all outlier detection methods (PCA/Hotelling T², Q-residuals,
    Mahalanobis distance, Y-value checks) and aggregates results into
    a comprehensive report with confidence levels.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (samples × wavelengths)
    y : np.ndarray
        Reference values
    n_pca_components : int, default=5
        Number of principal components for PCA-based methods
    y_lower_bound : float, optional
        Minimum chemically reasonable Y value
    y_upper_bound : float, optional
        Maximum chemically reasonable Y value

    Returns
    -------
    OutlierReport
        Dataclass containing comprehensive outlier detection results

    Notes
    -----
    Confidence levels:
    - High (3+ flags): Strong evidence of outlier, recommend review
    - Moderate (2 flags): Possible outlier, investigate further
    - Low (1 flag): Borderline case, likely not a concern

    The combined_flags uses 2+ methods as the threshold for outlier
    classification, which balances sensitivity and specificity.
    """
    # Run all detection methods
    pca_results = run_pca_outlier_detection(X, y, n_pca_components)
    q_results = compute_q_residuals(X, pca_results.pca_model, n_pca_components)
    maha_results = compute_mahalanobis_distance(pca_results.scores)
    y_results = check_y_data_consistency(y, y_lower_bound, y_upper_bound)

    # Compute combined flags
    total_flags = (
        pca_results.outlier_flags.astype(int) +
        q_results.outlier_flags.astype(int) +
        maha_results.outlier_flags.astype(int) +
        y_results.all_outliers.astype(int)
    )

    combined_flags = total_flags >= 2

    # Separate by confidence level
    high_confidence = np.where(total_flags >= 3)[0]
    moderate_confidence = np.where(total_flags == 2)[0]
    low_confidence = np.where(total_flags == 1)[0]

    return OutlierReport(
        pca=pca_results,
        q_residuals=q_results,
        mahalanobis=maha_results,
        y_consistency=y_results,
        combined_flags=combined_flags,
        total_flags_per_sample=total_flags,
        high_confidence_indices=high_confidence,
        moderate_confidence_indices=moderate_confidence,
        low_confidence_indices=low_confidence
    )
