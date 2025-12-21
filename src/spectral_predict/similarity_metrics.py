"""
Spectral Similarity Metrics for Library Search.

This module provides various similarity/distance metrics for comparing spectra,
including industry-standard Hit Quality Index (HQI) and Spectral Angle Mapper (SAM).
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Union, Optional
import warnings


def hit_quality_index(query: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate Hit Quality Index (HQI) between two spectra.

    HQI is the squared Pearson correlation coefficient, the industry standard
    for spectral library matching. Returns a value between 0 and 1, where
    1 indicates perfect match.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array)
    reference : np.ndarray
        Reference spectrum (1D array, same length as query)

    Returns
    -------
    float
        HQI score between 0 (no match) and 1 (perfect match)

    Notes
    -----
    HQI = r² where r is the Pearson correlation coefficient.
    Sensitive to baseline shifts - consider using derivative correlation
    for spectra with baseline issues.
    """
    if len(query) != len(reference):
        raise ValueError(f"Spectra must have same length: {len(query)} vs {len(reference)}")

    # Handle constant spectra (zero variance)
    if np.std(query) == 0 or np.std(reference) == 0:
        return 0.0

    r = np.corrcoef(query, reference)[0, 1]

    # Handle NaN from numerical issues
    if np.isnan(r):
        return 0.0

    return float(r ** 2)


def spectral_angle_mapper(query: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate Spectral Angle Mapper (SAM) between two spectra.

    SAM measures the angle between two spectra treated as vectors in
    n-dimensional space. Insensitive to intensity/illumination variations.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array)
    reference : np.ndarray
        Reference spectrum (1D array, same length as query)

    Returns
    -------
    float
        Angle in radians (0 = identical, π/2 = orthogonal)
        Lower values indicate better match.

    Notes
    -----
    SAM = arccos(dot(q, r) / (||q|| * ||r||))
    Insensitive to multiplicative scaling, good for uncalibrated spectra.
    """
    if len(query) != len(reference):
        raise ValueError(f"Spectra must have same length: {len(query)} vs {len(reference)}")

    # Handle zero-norm vectors
    norm_q = np.linalg.norm(query)
    norm_r = np.linalg.norm(reference)

    if norm_q == 0 or norm_r == 0:
        return np.pi / 2  # Maximum dissimilarity for zero vectors

    cos_theta = np.dot(query, reference) / (norm_q * norm_r)

    # Clamp to [-1, 1] to handle numerical precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return float(np.arccos(cos_theta))


def sam_to_similarity(sam_angle: float) -> float:
    """
    Convert SAM angle (radians) to similarity score (0-1).

    Parameters
    ----------
    sam_angle : float
        SAM angle in radians

    Returns
    -------
    float
        Similarity score between 0 (orthogonal) and 1 (identical)
    """
    # Normalize: 0 radians -> 1.0, π/2 radians -> 0.0
    return float(1.0 - (2.0 * sam_angle / np.pi))


def euclidean_distance(query: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two spectra.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array)
    reference : np.ndarray
        Reference spectrum (1D array, same length as query)

    Returns
    -------
    float
        Euclidean distance (lower = more similar)

    Notes
    -----
    D = sqrt(sum((q - r)²))
    Sensitive to scaling and baseline - normalize spectra first for best results.
    """
    if len(query) != len(reference):
        raise ValueError(f"Spectra must have same length: {len(query)} vs {len(reference)}")

    return float(np.linalg.norm(query - reference))


def euclidean_to_similarity(distance: float, scale: float = 1.0) -> float:
    """
    Convert Euclidean distance to similarity score (0-1).

    Parameters
    ----------
    distance : float
        Euclidean distance
    scale : float
        Scaling factor for distance normalization

    Returns
    -------
    float
        Similarity score between 0 (very distant) and 1 (identical)
    """
    return float(1.0 / (1.0 + distance / scale))


def cosine_similarity(query: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate cosine similarity between two spectra.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array)
    reference : np.ndarray
        Reference spectrum (1D array, same length as query)

    Returns
    -------
    float
        Cosine similarity between -1 and 1 (1 = identical direction)

    Notes
    -----
    This is the cosine of the spectral angle: cos(SAM).
    Unlike SAM, higher values indicate better match.
    """
    if len(query) != len(reference):
        raise ValueError(f"Spectra must have same length: {len(query)} vs {len(reference)}")

    norm_q = np.linalg.norm(query)
    norm_r = np.linalg.norm(reference)

    if norm_q == 0 or norm_r == 0:
        return 0.0

    return float(np.dot(query, reference) / (norm_q * norm_r))


def first_derivative_correlation(
    query: np.ndarray,
    reference: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2
) -> float:
    """
    Calculate correlation of first derivatives (baseline-insensitive HQI).

    Applies Savitzky-Golay first derivative to both spectra before
    computing HQI. This removes baseline effects and emphasizes
    spectral features.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array)
    reference : np.ndarray
        Reference spectrum (1D array, same length as query)
    window_length : int
        Savitzky-Golay window length (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter

    Returns
    -------
    float
        HQI of first derivatives, between 0 and 1

    Notes
    -----
    More robust to baseline shifts than standard HQI, but amplifies noise.
    """
    if len(query) != len(reference):
        raise ValueError(f"Spectra must have same length: {len(query)} vs {len(reference)}")

    if len(query) < window_length:
        warnings.warn(f"Spectrum too short for window_length={window_length}, using length-1")
        window_length = len(query) - 1 if len(query) % 2 == 0 else len(query) - 2
        if window_length < 3:
            return 0.0

    # Ensure odd window length
    if window_length % 2 == 0:
        window_length += 1

    # Compute first derivatives
    deriv_q = savgol_filter(query, window_length, polyorder, deriv=1)
    deriv_r = savgol_filter(reference, window_length, polyorder, deriv=1)

    return hit_quality_index(deriv_q, deriv_r)


def second_derivative_correlation(
    query: np.ndarray,
    reference: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> float:
    """
    Calculate correlation of second derivatives.

    Applies Savitzky-Golay second derivative to both spectra before
    computing HQI. More aggressive baseline removal than first derivative.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array)
    reference : np.ndarray
        Reference spectrum (1D array, same length as query)
    window_length : int
        Savitzky-Golay window length (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter

    Returns
    -------
    float
        HQI of second derivatives, between 0 and 1
    """
    if len(query) != len(reference):
        raise ValueError(f"Spectra must have same length: {len(query)} vs {len(reference)}")

    if len(query) < window_length:
        window_length = len(query) - 1 if len(query) % 2 == 0 else len(query) - 2
        if window_length < 5:
            return 0.0

    if window_length % 2 == 0:
        window_length += 1

    if polyorder >= window_length:
        polyorder = window_length - 1

    deriv_q = savgol_filter(query, window_length, polyorder, deriv=2)
    deriv_r = savgol_filter(reference, window_length, polyorder, deriv=2)

    return hit_quality_index(deriv_q, deriv_r)


def spectral_information_divergence(query: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate Spectral Information Divergence (SID) between two spectra.

    SID measures the discrepancy between probability distributions of
    spectral signatures using Kullback-Leibler divergence.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array, positive values)
    reference : np.ndarray
        Reference spectrum (1D array, positive values)

    Returns
    -------
    float
        SID value (lower = more similar, 0 = identical)

    Notes
    -----
    Spectra are normalized to probability distributions before computing SID.
    Negative values are clipped to a small positive number.
    """
    if len(query) != len(reference):
        raise ValueError(f"Spectra must have same length: {len(query)} vs {len(reference)}")

    # Ensure positive values (clip negatives to small positive)
    eps = 1e-10
    q = np.clip(query, eps, None)
    r = np.clip(reference, eps, None)

    # Normalize to probability distributions
    p = q / np.sum(q)
    q_dist = r / np.sum(r)

    # Compute KL divergences
    kl_pq = np.sum(p * np.log(p / q_dist))
    kl_qp = np.sum(q_dist * np.log(q_dist / p))

    return float(kl_pq + kl_qp)


def sid_to_similarity(sid: float, scale: float = 0.1) -> float:
    """
    Convert SID to similarity score (0-1).

    Parameters
    ----------
    sid : float
        SID value
    scale : float
        Scaling factor

    Returns
    -------
    float
        Similarity score between 0 and 1
    """
    return float(1.0 / (1.0 + sid / scale))


# Metric registry for easy access
METRICS = {
    'hqi': {
        'func': hit_quality_index,
        'name': 'Hit Quality Index',
        'higher_is_better': True,
        'range': (0, 1),
    },
    'sam': {
        'func': spectral_angle_mapper,
        'name': 'Spectral Angle Mapper',
        'higher_is_better': False,
        'range': (0, np.pi / 2),
        'to_similarity': sam_to_similarity,
    },
    'euclidean': {
        'func': euclidean_distance,
        'name': 'Euclidean Distance',
        'higher_is_better': False,
        'range': (0, np.inf),
        'to_similarity': euclidean_to_similarity,
    },
    'cosine': {
        'func': cosine_similarity,
        'name': 'Cosine Similarity',
        'higher_is_better': True,
        'range': (-1, 1),
    },
    'deriv1_corr': {
        'func': first_derivative_correlation,
        'name': '1st Derivative Correlation',
        'higher_is_better': True,
        'range': (0, 1),
    },
    'deriv2_corr': {
        'func': second_derivative_correlation,
        'name': '2nd Derivative Correlation',
        'higher_is_better': True,
        'range': (0, 1),
    },
    'sid': {
        'func': spectral_information_divergence,
        'name': 'Spectral Information Divergence',
        'higher_is_better': False,
        'range': (0, np.inf),
        'to_similarity': sid_to_similarity,
    },
}


def compute_similarity(
    query: np.ndarray,
    reference: np.ndarray,
    metric: str = 'hqi',
    normalize: bool = True,
    **kwargs
) -> float:
    """
    Compute similarity between two spectra using specified metric.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum
    reference : np.ndarray
        Reference spectrum
    metric : str
        Metric name: 'hqi', 'sam', 'euclidean', 'cosine', 'deriv1_corr',
        'deriv2_corr', 'sid'
    normalize : bool
        If True, convert distance metrics to similarity scores (0-1 scale)
    **kwargs
        Additional arguments passed to metric function

    Returns
    -------
    float
        Similarity score (higher = more similar when normalize=True)
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(METRICS.keys())}")

    metric_info = METRICS[metric]
    score = metric_info['func'](query, reference, **kwargs)

    if normalize and not metric_info['higher_is_better']:
        if 'to_similarity' in metric_info:
            score = metric_info['to_similarity'](score)
        else:
            # Default normalization for distance metrics
            score = 1.0 / (1.0 + score)

    return score


def compute_batch_similarity(
    query: np.ndarray,
    references: np.ndarray,
    metric: str = 'hqi',
    normalize: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Compute similarity between query and multiple reference spectra.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum (1D array)
    references : np.ndarray
        Reference spectra (2D array, rows are spectra)
    metric : str
        Metric name
    normalize : bool
        If True, convert to similarity scores
    **kwargs
        Additional arguments passed to metric function

    Returns
    -------
    np.ndarray
        Array of similarity scores
    """
    n_refs = references.shape[0]
    scores = np.zeros(n_refs)

    for i in range(n_refs):
        scores[i] = compute_similarity(query, references[i], metric, normalize, **kwargs)

    return scores
