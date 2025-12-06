"""
Class Imbalance Detection for Spectral Predict v3.

This module provides imbalance detection methods for both classification
and regression tasks in spectroscopy. Results are returned as dataclasses
for type safety and clean integration with v3.

CLASSIFICATION METHODS:
- Class imbalance detection with severity levels
- Multi-class imbalance detection
- Threshold-based warnings

REGRESSION METHODS:
- Target distribution imbalance detection
- Skewed distribution detection
- Range-based coverage analysis

Example:
    >>> from spectral_predict_v3.core.imbalance import detect_class_imbalance
    >>> result = detect_class_imbalance(y_train)
    >>> if result.is_imbalanced:
    ...     print(f"Imbalance detected: {result.severity}")
    ...     print(f"Ratio: {result.imbalance_ratio:.1f}:1")
    ...     print(f"Recommendation: {result.recommendation}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import Counter


# ============================================================================
# RESULT DATACLASSES
# ============================================================================

@dataclass
class ClassImbalanceResult:
    """
    Result of classification imbalance detection.

    Attributes
    ----------
    is_imbalanced : bool
        Whether imbalance exceeds threshold
    imbalance_ratio : float
        Ratio of majority to minority class (majority_count / minority_count)
    class_counts : dict
        Count of samples per class {class: count}
    majority_class : any
        Label of the majority class
    minority_class : any
        Label of the minority class
    severity : str
        Imbalance severity: 'none', 'moderate', 'severe', 'extreme'
    recommendation : str
        Suggested method for handling imbalance
    """
    is_imbalanced: bool
    imbalance_ratio: float
    class_counts: Dict[Any, int]
    majority_class: Any
    minority_class: Any
    severity: str
    recommendation: str


@dataclass
class RegressionImbalanceResult:
    """
    Result of regression target imbalance detection.

    Attributes
    ----------
    is_imbalanced : bool
        Whether distribution is imbalanced
    bin_counts : np.ndarray
        Number of samples per bin
    bin_edges : np.ndarray
        Bin boundaries
    sparse_bins : list
        Indices of bins with insufficient samples
    coverage : float
        Ratio of minimum bin count to mean bin count
    severity : str
        Imbalance severity: 'none', 'moderate', 'severe'
    recommendation : str
        Suggested method for handling imbalance
    n_samples : int
        Total number of samples
    target_range : tuple
        (min_value, max_value) of target
    """
    is_imbalanced: bool
    bin_counts: np.ndarray
    bin_edges: np.ndarray
    sparse_bins: List[int]
    coverage: float
    severity: str
    recommendation: str
    n_samples: int
    target_range: tuple


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_class_imbalance(y: np.ndarray, threshold: float = 3.0) -> ClassImbalanceResult:
    """
    Detect class imbalance in classification targets.

    Parameters
    ----------
    y : array-like
        Target labels (classification)
    threshold : float, default=3.0
        Imbalance ratio threshold (majority:minority) above which to flag

    Returns
    -------
    ClassImbalanceResult
        Dataclass containing imbalance metrics and recommendations

    Examples
    --------
    >>> # Balanced dataset
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> result = detect_class_imbalance(y)
    >>> result.is_imbalanced
    False
    >>> result.severity
    'none'

    >>> # Imbalanced dataset (90/10 split)
    >>> y = np.array([0]*90 + [1]*10)
    >>> result = detect_class_imbalance(y)
    >>> result.is_imbalanced
    True
    >>> result.imbalance_ratio
    9.0
    >>> result.severity
    'severe'
    """
    y = np.asarray(y)
    class_counts = Counter(y)

    if len(class_counts) < 2:
        return ClassImbalanceResult(
            is_imbalanced=False,
            imbalance_ratio=1.0,
            class_counts=dict(class_counts),
            majority_class=None,
            minority_class=None,
            severity='none',
            recommendation='No imbalance detected (single class)'
        )

    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]

    imbalance_ratio = majority_count / minority_count

    # Determine severity
    if imbalance_ratio < threshold:
        severity = 'none'
        is_imbalanced = False
        recommendation = 'No imbalance handling needed'
    elif imbalance_ratio < 5.0:
        severity = 'moderate'
        is_imbalanced = True
        recommendation = 'Use class_weight="balanced" or light SMOTE'
    elif imbalance_ratio < 10.0:
        severity = 'severe'
        is_imbalanced = True
        recommendation = 'Use SMOTE or ADASYN for oversampling'
    else:
        severity = 'extreme'
        is_imbalanced = True
        recommendation = 'Combine SMOTE with undersampling (SMOTETomek)'

    return ClassImbalanceResult(
        is_imbalanced=is_imbalanced,
        imbalance_ratio=imbalance_ratio,
        class_counts=dict(class_counts),
        majority_class=majority_class,
        minority_class=minority_class,
        severity=severity,
        recommendation=recommendation
    )


def detect_regression_imbalance(
    y: np.ndarray,
    n_bins: int = 10,
    coverage_threshold: float = 0.2
) -> RegressionImbalanceResult:
    """
    Detect target imbalance in regression (uneven distribution across range).

    Parameters
    ----------
    y : array-like
        Target values (regression)
    n_bins : int, default=10
        Number of bins to divide target range
    coverage_threshold : float, default=0.2
        Minimum fraction of samples per bin for balanced distribution

    Returns
    -------
    RegressionImbalanceResult
        Dataclass containing imbalance metrics and recommendations

    Examples
    --------
    >>> # Balanced distribution
    >>> y = np.linspace(0, 10, 100)
    >>> result = detect_regression_imbalance(y)
    >>> result.is_imbalanced
    False

    >>> # Skewed distribution (many zeros, few high values)
    >>> y = np.concatenate([np.zeros(80), np.random.uniform(5, 10, 20)])
    >>> result = detect_regression_imbalance(y)
    >>> result.is_imbalanced
    True
    >>> result.severity in ['moderate', 'severe']
    True
    """
    y = np.asarray(y)

    # Create bins across target range
    bin_counts, bin_edges = np.histogram(y, bins=n_bins)
    mean_count = bin_counts.mean()
    min_count = bin_counts.min()

    # Find sparse bins
    threshold_count = coverage_threshold * len(y) / n_bins
    sparse_bins = np.where(bin_counts < threshold_count)[0].tolist()

    coverage = min_count / mean_count if mean_count > 0 else 0

    # Determine severity
    if coverage > 0.5:  # Min bin has >50% of mean
        severity = 'none'
        is_imbalanced = False
        recommendation = 'Target distribution is relatively balanced'
    elif coverage > 0.2:
        severity = 'moderate'
        is_imbalanced = True
        recommendation = 'Use target binning with sample weights'
    else:
        severity = 'severe'
        is_imbalanced = True
        recommendation = 'Use rare-value boosting or consider data collection'

    return RegressionImbalanceResult(
        is_imbalanced=is_imbalanced,
        bin_counts=bin_counts,
        bin_edges=bin_edges,
        sparse_bins=sparse_bins,
        coverage=coverage,
        severity=severity,
        recommendation=recommendation,
        n_samples=len(y),
        target_range=(float(y.min()), float(y.max()))
    )


# ============================================================================
# MULTI-CLASS IMBALANCE
# ============================================================================

@dataclass
class MultiClassImbalanceResult:
    """
    Result of multi-class imbalance detection.

    Attributes
    ----------
    is_imbalanced : bool
        Whether any class pair exceeds threshold
    class_counts : dict
        Count of samples per class
    max_ratio : float
        Maximum imbalance ratio across all class pairs
    min_class : any
        Class with minimum samples
    max_class : any
        Class with maximum samples
    severity : str
        Overall imbalance severity
    recommendation : str
        Suggested handling method
    pairwise_ratios : dict
        Ratios between all class pairs {(class1, class2): ratio}
    """
    is_imbalanced: bool
    class_counts: Dict[Any, int]
    max_ratio: float
    min_class: Any
    max_class: Any
    severity: str
    recommendation: str
    pairwise_ratios: Dict[tuple, float] = field(default_factory=dict)


def detect_multiclass_imbalance(y: np.ndarray, threshold: float = 3.0) -> MultiClassImbalanceResult:
    """
    Detect imbalance in multi-class classification (3+ classes).

    This function checks all pairwise class ratios to identify the worst
    imbalance and provide appropriate recommendations.

    Parameters
    ----------
    y : array-like
        Target labels (classification with 3+ classes)
    threshold : float, default=3.0
        Imbalance ratio threshold

    Returns
    -------
    MultiClassImbalanceResult
        Dataclass with multi-class imbalance metrics

    Examples
    --------
    >>> # Balanced 3-class
    >>> y = np.array([0]*30 + [1]*30 + [2]*30)
    >>> result = detect_multiclass_imbalance(y)
    >>> result.is_imbalanced
    False

    >>> # Imbalanced 3-class (60/30/10)
    >>> y = np.array([0]*60 + [1]*30 + [2]*10)
    >>> result = detect_multiclass_imbalance(y)
    >>> result.is_imbalanced
    True
    >>> result.max_ratio
    6.0
    """
    y = np.asarray(y)
    class_counts = Counter(y)

    if len(class_counts) < 2:
        return MultiClassImbalanceResult(
            is_imbalanced=False,
            class_counts=dict(class_counts),
            max_ratio=1.0,
            min_class=None,
            max_class=None,
            severity='none',
            recommendation='Single class detected'
        )

    # Find max and min classes
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)
    max_ratio = class_counts[max_class] / class_counts[min_class]

    # Compute all pairwise ratios
    pairwise_ratios = {}
    classes = sorted(class_counts.keys())
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            count1, count2 = class_counts[c1], class_counts[c2]
            ratio = max(count1, count2) / min(count1, count2)
            pairwise_ratios[(c1, c2)] = ratio

    # Determine severity
    if max_ratio < threshold:
        severity = 'none'
        is_imbalanced = False
        recommendation = 'No imbalance handling needed'
    elif max_ratio < 5.0:
        severity = 'moderate'
        is_imbalanced = True
        recommendation = 'Use class_weight="balanced"'
    elif max_ratio < 10.0:
        severity = 'severe'
        is_imbalanced = True
        recommendation = 'Use SMOTE with multi-class support'
    else:
        severity = 'extreme'
        is_imbalanced = True
        recommendation = 'Consider combining classes or collecting more data'

    return MultiClassImbalanceResult(
        is_imbalanced=is_imbalanced,
        class_counts=dict(class_counts),
        max_ratio=max_ratio,
        min_class=min_class,
        max_class=max_class,
        severity=severity,
        recommendation=recommendation,
        pairwise_ratios=pairwise_ratios
    )


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_imbalance_warning(result: ClassImbalanceResult) -> str:
    """
    Format a class imbalance result as a user-friendly warning message.

    Parameters
    ----------
    result : ClassImbalanceResult
        The imbalance detection result

    Returns
    -------
    str
        Formatted warning message

    Examples
    --------
    >>> y = np.array([0]*90 + [1]*10)
    >>> result = detect_class_imbalance(y)
    >>> msg = format_imbalance_warning(result)
    >>> print(msg)
    CLASS IMBALANCE DETECTED (SEVERE)
    ...
    """
    if not result.is_imbalanced:
        return "No class imbalance detected"

    lines = [
        f"CLASS IMBALANCE DETECTED ({result.severity.upper()})",
        f"",
        f"Imbalance Ratio: {result.imbalance_ratio:.1f}:1",
        f"Majority Class: {result.majority_class} ({result.class_counts[result.majority_class]} samples)",
        f"Minority Class: {result.minority_class} ({result.class_counts[result.minority_class]} samples)",
        f"",
        f"Class Distribution:",
    ]

    for cls, count in sorted(result.class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / sum(result.class_counts.values())
        lines.append(f"  {cls}: {count} samples ({pct:.1f}%)")

    lines.extend([
        f"",
        f"Recommendation: {result.recommendation}"
    ])

    return "\n".join(lines)


def format_regression_imbalance_warning(result: RegressionImbalanceResult) -> str:
    """
    Format a regression imbalance result as a user-friendly warning message.

    Parameters
    ----------
    result : RegressionImbalanceResult
        The imbalance detection result

    Returns
    -------
    str
        Formatted warning message
    """
    if not result.is_imbalanced:
        return "No target imbalance detected"

    lines = [
        f"TARGET IMBALANCE DETECTED ({result.severity.upper()})",
        f"",
        f"Target Range: {result.target_range[0]:.2f} to {result.target_range[1]:.2f}",
        f"Total Samples: {result.n_samples}",
        f"Coverage Ratio: {result.coverage:.2f}",
        f"",
        f"Sparse Bins: {len(result.sparse_bins)} of {len(result.bin_counts)} bins have insufficient samples",
        f"",
        f"Recommendation: {result.recommendation}"
    ]

    return "\n".join(lines)
