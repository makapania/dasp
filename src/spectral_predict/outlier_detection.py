"""Outlier detection functions for spectral analysis.

This module provides comprehensive outlier detection methods for spectral data
including PCA-based detection, Q-residuals, Mahalanobis distance, and reference
value consistency checks.

Methods are based on standard chemometric outlier detection approaches:
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
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats


def run_pca_outlier_detection(X, y=None, n_components=5):
    """
    Perform PCA-based outlier detection on spectral data.

    Computes principal component scores and Hotelling T² statistic for each
    sample. The T² statistic measures the distance of each sample from the
    center of the principal component space, accounting for variance in each
    direction.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Spectral data (samples × wavelengths)
    y : pd.Series or np.ndarray, optional
        Reference values for overlay in visualizations
    n_components : int, default=5
        Number of principal components to compute

    Returns
    -------
    results : dict
        Dictionary containing:
        - pca_model : sklearn.decomposition.PCA
            Fitted PCA object
        - scores : np.ndarray
            PC scores (samples × n_components)
        - loadings : np.ndarray
            PC loadings (wavelengths × n_components)
        - variance_explained : np.ndarray
            Fraction of variance explained by each PC
        - hotelling_t2 : np.ndarray
            Hotelling T² statistic for each sample
        - t2_threshold : float
            95% confidence threshold using F-distribution
        - outlier_flags : np.ndarray
            Boolean array (True = outlier)
        - n_outliers : int
            Count of outliers detected
        - outlier_indices : np.ndarray
            Array indices of outlier samples

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
    # Convert to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.array(X)

    n_samples, n_features = X_array.shape

    # Clip n_components to valid range
    n_components = min(n_components, n_samples - 1, n_features)

    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_array)

    # Compute Hotelling T²
    # T² = score · inv(cov) · score.T
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
    # T²_threshold = (p(n-1)/(n-p)) * F(α, p, n-p)
    alpha = 0.05
    if n_samples > n_components:
        t2_threshold = (n_components * (n_samples - 1) / (n_samples - n_components) *
                        stats.f.ppf(1 - alpha, n_components, n_samples - n_components))
    else:
        # If n_samples <= n_components, use chi-squared approximation
        t2_threshold = stats.chi2.ppf(1 - alpha, n_components)

    outlier_flags = t2_values > t2_threshold

    return {
        'pca_model': pca,
        'scores': scores,
        'loadings': pca.components_.T,
        'variance_explained': pca.explained_variance_ratio_,
        'hotelling_t2': t2_values,
        't2_threshold': t2_threshold,
        'outlier_flags': outlier_flags,
        'n_outliers': int(np.sum(outlier_flags)),
        'outlier_indices': np.where(outlier_flags)[0]
    }


def compute_q_residuals(X, pca_model, n_components=None):
    """
    Compute Q-residuals (SPE - Squared Prediction Error) for outlier detection.

    Q-residuals measure the reconstruction error when projecting data into the
    principal component space and back. High Q-residuals indicate samples that
    are poorly represented by the PCA model.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Original spectral data
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    n_components : int, optional
        Number of components to use for reconstruction. If None, uses all
        components from the fitted model.

    Returns
    -------
    results : dict
        Dictionary containing:
        - q_residuals : np.ndarray
            Q-residual (SPE) for each sample
        - q_threshold : float
            95th percentile threshold
        - outlier_flags : np.ndarray
            Boolean array (True = outlier)
        - n_outliers : int
            Count of outliers detected
        - outlier_indices : np.ndarray
            Array indices of outlier samples

    Notes
    -----
    Q-residual is computed as:
        Q = sum((X - X_reconstructed)²)

    The threshold uses the 95th percentile of the Q-residual distribution.

    Edge cases handled:
    - If X is a DataFrame, it is converted to numpy array
    - If n_components > available components, uses all available
    """
    # Convert to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.array(X)

    if n_components is None:
        n_components = pca_model.n_components_
    else:
        # Clip to available components
        n_components = min(n_components, pca_model.n_components_)

    # Project data to PC space and back
    scores = pca_model.transform(X_array)[:, :n_components]
    X_reconstructed = scores @ pca_model.components_[:n_components, :]

    # Add back the mean (PCA centers the data)
    X_reconstructed += pca_model.mean_

    # Compute reconstruction error
    residuals = X_array - X_reconstructed
    q_residuals = np.sum(residuals ** 2, axis=1)

    # 95th percentile threshold
    q_threshold = np.percentile(q_residuals, 95)

    outlier_flags = q_residuals > q_threshold

    return {
        'q_residuals': q_residuals,
        'q_threshold': q_threshold,
        'outlier_flags': outlier_flags,
        'n_outliers': int(np.sum(outlier_flags)),
        'outlier_indices': np.where(outlier_flags)[0]
    }


def compute_mahalanobis_distance(scores):
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
    results : dict
        Dictionary containing:
        - distances : np.ndarray
            Mahalanobis distance for each sample
        - median : float
            Median distance
        - mad : float
            Median absolute deviation
        - threshold : float
            3× MAD threshold (median + 3*MAD)
        - outlier_flags : np.ndarray
            Boolean array (True = outlier)
        - n_outliers : int
            Count of outliers detected
        - outlier_indices : np.ndarray
            Array indices of outlier samples

    Notes
    -----
    Mahalanobis distance is computed as:
        D = sqrt((x - μ)' Σ⁻¹ (x - μ))
    where μ is the mean and Σ is the covariance matrix.

    The threshold uses 3× median absolute deviation (MAD), which is robust
    to outliers in the distance distribution itself.

    Edge cases handled:
    - If covariance matrix is singular, regularization is applied
    - If scores is 1D, it is reshaped to 2D
    """
    # Ensure scores is 2D
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

    return {
        'distances': distances,
        'median': median,
        'mad': mad,
        'threshold': threshold,
        'outlier_flags': outlier_flags,
        'n_outliers': int(np.sum(outlier_flags)),
        'outlier_indices': np.where(outlier_flags)[0]
    }


def check_y_data_consistency(y, lower_bound=None, upper_bound=None):
    """
    Check reference data for outliers and inconsistencies.

    Performs statistical checks on reference values to identify potential
    data entry errors, mislabeled samples, or values outside chemically
    reasonable ranges.

    Parameters
    ----------
    y : np.ndarray or pd.Series
        Reference values
    lower_bound : float, optional
        Minimum chemically reasonable value
    upper_bound : float, optional
        Maximum chemically reasonable value

    Returns
    -------
    results : dict
        Dictionary containing:
        - mean : float
            Mean of reference values
        - std : float
            Standard deviation
        - median : float
            Median value
        - min : float
            Minimum value
        - max : float
            Maximum value
        - z_scores : np.ndarray
            Z-score for each sample
        - z_outliers : np.ndarray
            Boolean array for samples with |z| > 3
        - range_outliers : np.ndarray
            Boolean array for samples outside [lower_bound, upper_bound]
        - all_outliers : np.ndarray
            Boolean array combining z_outliers and range_outliers
        - n_outliers : int
            Total count of outliers
        - outlier_indices : np.ndarray
            Array indices of outlier samples

    Notes
    -----
    Z-score outlier detection uses the ±3σ rule, which flags approximately
    0.3% of samples from a normal distribution.

    Range checks are optional and should be based on domain knowledge
    (e.g., protein content cannot exceed 100%, pH must be 0-14, etc.)

    Edge cases handled:
    - If std is zero (all values identical), z_scores are set to zero
    - If bounds are None, range_outliers are all False
    """
    # Convert to numpy array
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)

    # Compute statistics
    mean = np.mean(y_array)
    std = np.std(y_array)
    median = np.median(y_array)
    min_val = np.min(y_array)
    max_val = np.max(y_array)

    # Z-scores (handle zero std)
    if std < 1e-10:
        z_scores = np.zeros_like(y_array)
    else:
        z_scores = (y_array - mean) / std

    z_outliers = np.abs(z_scores) > 3

    # Range check
    range_outliers = np.zeros(len(y_array), dtype=bool)
    if lower_bound is not None:
        range_outliers |= y_array < lower_bound
    if upper_bound is not None:
        range_outliers |= y_array > upper_bound

    # Combine
    all_outliers = z_outliers | range_outliers

    return {
        'mean': float(mean),
        'std': float(std),
        'median': float(median),
        'min': float(min_val),
        'max': float(max_val),
        'z_scores': z_scores,
        'z_outliers': z_outliers,
        'range_outliers': range_outliers,
        'all_outliers': all_outliers,
        'n_outliers': int(np.sum(all_outliers)),
        'outlier_indices': np.where(all_outliers)[0]
    }


def generate_outlier_report(X, y, n_pca_components=5,
                           y_lower_bound=None, y_upper_bound=None):
    """
    Comprehensive outlier detection report combining all methods.

    Runs all outlier detection methods (PCA/Hotelling T², Q-residuals,
    Mahalanobis distance, Y-value checks) and aggregates results into
    a comprehensive report with confidence levels.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Spectral data (samples × wavelengths)
    y : pd.Series or np.ndarray
        Reference values
    n_pca_components : int, default=5
        Number of principal components for PCA-based methods
    y_lower_bound : float, optional
        Minimum chemically reasonable Y value
    y_upper_bound : float, optional
        Maximum chemically reasonable Y value

    Returns
    -------
    report : dict
        Dictionary containing:
        - pca : dict
            PCA outlier detection results
        - q_residuals : dict
            Q-residuals outlier detection results
        - mahalanobis : dict
            Mahalanobis distance outlier detection results
        - y_consistency : dict
            Y data consistency check results
        - combined_flags : np.ndarray
            Boolean array for high-confidence outliers (2+ methods)
        - outlier_summary : pd.DataFrame
            DataFrame with all flags per sample, columns:
            - Sample_Index: Sample index
            - Y_Value: Reference value
            - Hotelling_T2: T² statistic
            - T2_Outlier: Flagged by T²
            - Q_Residual: Q-residual value
            - Q_Outlier: Flagged by Q-residuals
            - Mahalanobis_Distance: Distance value
            - Maha_Outlier: Flagged by Mahalanobis
            - Y_ZScore: Z-score of Y value
            - Y_Outlier: Flagged by Y checks
            - Total_Flags: Sum of all flags
        - high_confidence_outliers : pd.DataFrame
            Samples flagged by 3+ methods
        - moderate_confidence_outliers : pd.DataFrame
            Samples flagged by exactly 2 methods
        - low_confidence_outliers : pd.DataFrame
            Samples flagged by exactly 1 method

    Notes
    -----
    Confidence levels:
    - High (3+ flags): Strong evidence of outlier, recommend review
    - Moderate (2 flags): Possible outlier, investigate further
    - Low (1 flag): Borderline case, likely not a concern

    The combined_flags uses 2+ methods as the threshold for outlier
    classification, which balances sensitivity and specificity.

    Edge cases handled:
    - All detection methods handle their own edge cases
    - If no outliers detected by any method, DataFrames will be empty
    - If y is None, Y consistency checks are skipped
    """
    # Run all detection methods
    pca_results = run_pca_outlier_detection(X, y, n_pca_components)
    q_results = compute_q_residuals(X, pca_results['pca_model'], n_pca_components)
    maha_results = compute_mahalanobis_distance(pca_results['scores'])
    y_results = check_y_data_consistency(y, y_lower_bound, y_upper_bound)

    # Convert y to array if needed
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)

    # Create summary DataFrame
    n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)

    summary = pd.DataFrame({
        'Sample_Index': range(n_samples),
        'Y_Value': y_array,
        'Hotelling_T2': pca_results['hotelling_t2'],
        'T2_Outlier': pca_results['outlier_flags'],
        'Q_Residual': q_results['q_residuals'],
        'Q_Outlier': q_results['outlier_flags'],
        'Mahalanobis_Distance': maha_results['distances'],
        'Maha_Outlier': maha_results['outlier_flags'],
        'Y_ZScore': y_results['z_scores'],
        'Y_Outlier': y_results['all_outliers'],
        'Total_Flags': (pca_results['outlier_flags'].astype(int) +
                       q_results['outlier_flags'].astype(int) +
                       maha_results['outlier_flags'].astype(int) +
                       y_results['all_outliers'].astype(int))
    })

    # Combined flags: flagged by 2+ methods (high confidence)
    combined_flags = summary['Total_Flags'] >= 2

    # Separate by confidence level
    high_confidence = summary[summary['Total_Flags'] >= 3].copy()
    moderate_confidence = summary[summary['Total_Flags'] == 2].copy()
    low_confidence = summary[summary['Total_Flags'] == 1].copy()

    return {
        'pca': pca_results,
        'q_residuals': q_results,
        'mahalanobis': maha_results,
        'y_consistency': y_results,
        'combined_flags': combined_flags,
        'outlier_summary': summary,
        'high_confidence_outliers': high_confidence,
        'moderate_confidence_outliers': moderate_confidence,
        'low_confidence_outliers': low_confidence
    }
