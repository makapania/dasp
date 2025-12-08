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
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import compute_sample_weight

# Check for imbalanced-learn availability
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    SMOTE = ADASYN = BorderlineSMOTE = None
    RandomUnderSampler = TomekLinks = None
    SMOTETomek = SMOTEENN = None


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


@dataclass
class ImbalanceRecommendation:
    """
    Recommendation for handling imbalance based on data characteristics.

    Attributes
    ----------
    recommended_method : str or None
        Recommended imbalance handling method, or None if not needed
    reason : str
        Explanation for the recommendation
    alternative : str or None
        Alternative method to consider
    warnings : list of str
        Potential issues or caveats with the data
    """
    recommended_method: Optional[str]
    reason: str
    alternative: Optional[str]
    warnings: List[str] = field(default_factory=list)


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


# ============================================================================
# CLASSIFICATION RESAMPLING
# ============================================================================

class ClassificationResampler(BaseEstimator):
    """
    Wrapper for imbalanced-learn resampling methods that works in pipelines.

    This transformer applies resampling using fit_resample() for use with
    imblearn Pipeline.

    Parameters
    ----------
    method : str or object
        Resampling method name ('smote', 'adasyn', etc.) or imblearn object
    random_state : int, optional
        Random seed for reproducibility. CRITICAL for scientific publications.
    **params : dict
        Parameters to pass to the resampling method

    Example
    -------
    >>> resampler = ClassificationResampler('smote', random_state=42, k_neighbors=5)
    >>> X_res, y_res = resampler.fit_resample(X_train, y_train)

    Note
    ----
    This class should NOT inherit from TransformerMixin because it implements
    fit_resample() for use with imblearn Pipeline. TransformerMixin would add
    a transform() method that conflicts with fit_resample() semantics.
    """

    def __init__(self, method='smote', random_state=None, **params):
        if not HAS_IMBLEARN:
            raise ImportError(
                "imbalanced-learn (imblearn) is required for resampling methods. "
                "Install with: pip install imbalanced-learn"
            )

        self.method = method
        self.random_state = random_state
        self.params = params
        self.resampler_ = None
        self.X_resampled_ = None
        self.y_resampled_ = None

    def fit(self, X, y=None):
        """Fit the resampler (creates internal resampler object)."""
        # Create resampler based on method name
        if isinstance(self.method, str):
            method_map = {
                'smote': SMOTE,
                'adasyn': ADASYN,
                'borderline_smote': BorderlineSMOTE,
                'random_undersampler': RandomUnderSampler,
                'tomek_links': TomekLinks,
                'smote_tomek': SMOTETomek,
                'smote_enn': SMOTEENN
            }
            method_lower = self.method.lower().replace('-', '_')
            if method_lower not in method_map:
                raise ValueError(
                    f"Unknown resampling method: {self.method}. "
                    f"Available: {list(method_map.keys())}"
                )
            resampler_class = method_map[method_lower]
            # Pass random_state for reproducibility (CRITICAL for scientific work)
            resampler_params = dict(self.params)
            if self.random_state is not None:
                resampler_params['random_state'] = self.random_state
            self.resampler_ = resampler_class(**resampler_params)
        else:
            # Allow passing custom imblearn object
            self.resampler_ = self.method

        return self

    def fit_resample(self, X, y):
        """
        Fit and resample the data.

        This is the main method called during pipeline training.
        """
        self.fit(X, y)

        original_size = len(y)
        original_class_counts = Counter(y)

        # Validate minimum samples for SMOTE-based methods
        if isinstance(self.resampler_, (SMOTE, ADASYN, BorderlineSMOTE, SMOTETomek, SMOTEENN)):
            k = self.params.get('k_neighbors', 5)
            min_samples_per_class = Counter(y)
            if min(min_samples_per_class.values()) <= k:
                warnings.warn(
                    f"Some classes have ≤{k} samples. SMOTE requires k_neighbors+1 samples. "
                    f"Skipping resampling for this fold.",
                    UserWarning
                )
                return X, y

        try:
            X_res, y_res = self.resampler_.fit_resample(X, y)
            self.X_resampled_ = X_res
            self.y_resampled_ = y_res

            resampled_size = len(y_res)
            change_pct = 100 * (resampled_size - original_size) / original_size

            if resampled_size > original_size:
                print(f"  {self.method.upper()}: {original_size} -> {resampled_size} samples "
                      f"(+{change_pct:.1f}% oversampling)")
            elif resampled_size < original_size:
                print(f"  {self.method.upper()}: {original_size} -> {resampled_size} samples "
                      f"({change_pct:.1f}% undersampling)")
            else:
                print(f"  {self.method.upper()}: {original_size} samples (balanced)")

            return X_res, y_res
        except Exception as e:
            warnings.warn(
                f"Resampling failed: {e}. Proceeding without resampling.",
                UserWarning
            )
            return X, y


# ============================================================================
# REGRESSION IMBALANCE TRANSFORMERS
# ============================================================================

class RegressionUndersampler(BaseEstimator):
    """
    Undersample over-represented target ranges for regression.

    This is ideal for datasets with many zeros or heavily skewed distributions
    (e.g., collagen % with many zeros and sparse high values). It randomly
    removes samples from over-represented bins to create a more balanced
    target distribution.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins to divide target range
    sampling_strategy : str or float, default='auto'
        How to determine target samples per bin:
        - 'auto': Undersample to median bin count
        - 'mean': Undersample to mean bin count
        - float (0-1): Keep this fraction of samples in over-represented bins
    random_state : int, default=42
        Random seed for reproducibility

    Example
    -------
    >>> # Dataset with many zeros (e.g., collagen % from 0-19%)
    >>> undersampler = RegressionUndersampler(n_bins=10, sampling_strategy='auto')
    >>> X_res, y_res = undersampler.fit_resample(X, y)
    >>> print(f"Original: {len(y)} samples")
    >>> print(f"Resampled: {len(y_res)} samples")

    Note
    ----
    This class should NOT inherit from TransformerMixin because it implements
    fit_resample() for use with imblearn Pipeline. TransformerMixin would add
    a transform() method that conflicts with fit_resample() semantics.
    """

    def __init__(self, n_bins=10, sampling_strategy='auto', random_state=42):
        self.n_bins = n_bins
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the undersampler."""
        return self

    def fit_resample(self, X, y):
        """
        Undersample over-represented target ranges.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        original_size = len(y)

        # Create bins
        bin_edges = np.linspace(y.min(), y.max(), self.n_bins + 1)
        bin_indices = np.digitize(y, bins=bin_edges[:-1], right=False) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Count samples per bin
        unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
        bin_count_dict = dict(zip(unique_bins, bin_counts))

        # Determine target count per bin
        if self.sampling_strategy == 'auto':
            target_count = int(np.median(bin_counts))
        elif self.sampling_strategy == 'mean':
            target_count = int(np.mean(bin_counts))
        elif isinstance(self.sampling_strategy, float):
            # Keep this fraction of samples in over-represented bins
            target_count = int(max(bin_counts) * self.sampling_strategy)
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

        # Undersample over-represented bins
        # Use RandomState for thread-safe, reproducible random sampling
        rng = np.random.RandomState(self.random_state)
        indices_to_keep = []

        for bin_idx in range(self.n_bins):
            bin_mask = bin_indices == bin_idx
            bin_sample_indices = np.where(bin_mask)[0]
            n_samples_in_bin = len(bin_sample_indices)

            if n_samples_in_bin > target_count:
                # Randomly select target_count samples
                selected = rng.choice(bin_sample_indices, size=target_count, replace=False)
                indices_to_keep.extend(selected)
            else:
                # Keep all samples in this bin
                indices_to_keep.extend(bin_sample_indices)

        indices_to_keep = np.array(sorted(indices_to_keep))

        resampled_size = len(indices_to_keep)
        reduction_pct = 100 * (original_size - resampled_size) / original_size

        print(f"  Undersampling: {original_size} -> {resampled_size} samples "
              f"({reduction_pct:.1f}% reduction, target range: {y.min():.2f}-{y.max():.2f})")

        return X[indices_to_keep], y[indices_to_keep]


class RegressionSampleWeighter(BaseEstimator, TransformerMixin):
    """
    Compute sample weights for regression based on target distribution.

    This transformer computes weights during fit() and stores them for use
    by downstream models that support sample_weight.

    Parameters
    ----------
    strategy : str, default='binning'
        Weighting strategy:
        - 'binning': Bin targets and weight inversely by bin frequency
        - 'rare_boost': Exponentially boost rare target values
        - 'balanced': Simple inverse frequency weighting
    n_bins : int, default=5
        Number of bins for 'binning' strategy
    boost_factor : float, default=2.0
        Boost multiplier for 'rare_boost' strategy

    Attributes
    ----------
    sample_weight_ : array
        Computed sample weights (stored after fit)

    Example
    -------
    >>> weighter = RegressionSampleWeighter(strategy='binning', n_bins=5)
    >>> weighter.fit(X_train, y_train)
    >>> # Access weights: weighter.sample_weight_
    >>> model.fit(X_train, y_train, sample_weight=weighter.sample_weight_)
    """

    def __init__(self, strategy='binning', n_bins=5, boost_factor=2.0):
        self.strategy = strategy
        self.n_bins = n_bins
        self.boost_factor = boost_factor
        self.sample_weight_ = None
        self.bin_edges_ = None

    def fit(self, X, y):
        """Compute sample weights based on target distribution."""
        y = np.asarray(y).ravel()

        if self.strategy == 'binning':
            # Bin targets and weight by inverse bin frequency
            bin_indices = np.digitize(y, bins=np.linspace(y.min(), y.max(), self.n_bins + 1))
            bin_counts = Counter(bin_indices)
            total_samples = len(y)

            weights = np.array([
                total_samples / (self.n_bins * bin_counts[bin_idx])
                for bin_idx in bin_indices
            ])

            self.bin_edges_ = np.linspace(y.min(), y.max(), self.n_bins + 1)

        elif self.strategy == 'rare_boost':
            # Exponentially boost samples far from the median
            median = np.median(y)
            std = np.std(y)
            if std == 0:
                weights = np.ones(len(y))
            else:
                distances = np.abs(y - median) / std
                weights = 1.0 + (self.boost_factor - 1.0) * (distances / distances.max())

        elif self.strategy == 'balanced':
            # Simple inverse frequency weighting (treat as discrete values)
            weights = compute_sample_weight('balanced', y)

        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Use 'binning', 'rare_boost', or 'balanced'."
            )

        # Normalize weights to mean=1
        self.sample_weight_ = weights / weights.mean()

        return self

    def transform(self, X):
        """Pass through unchanged (weights are stored in sample_weight_)."""
        return X

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        self.fit(X, y)
        return X

    def get_sample_weight(self):
        """Retrieve computed sample weights."""
        if self.sample_weight_ is None:
            raise RuntimeError("Must call fit() before get_sample_weight()")
        return self.sample_weight_


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def build_imbalance_transformer(method, task_type='classification', random_state=None, **params):
    """
    Factory function to create imbalance handling transformers.

    Parameters
    ----------
    method : str
        Imbalance handling method name

        Classification methods:
        - 'smote': Synthetic Minority Over-sampling Technique
        - 'adasyn': Adaptive Synthetic Sampling
        - 'borderline_smote': BorderlineSMOTE variant
        - 'random_undersampler': Random undersampling of majority class
        - 'tomek_links': Remove Tomek links
        - 'smote_tomek': Combined SMOTE + Tomek Links
        - 'smote_enn': Combined SMOTE + Edited Nearest Neighbors

        Regression methods:
        - 'binning': Target binning with sample weights
        - 'rare_boost': Rare-value boosting
        - 'balanced': Inverse frequency weighting

    task_type : str, default='classification'
        'classification' or 'regression'

    random_state : int, optional
        Random seed for reproducibility. CRITICAL for scientific publications.
        Ensures resampling produces identical results across runs.

    **params : dict
        Method-specific parameters

    Returns
    -------
    transformer : BaseEstimator
        sklearn-compatible transformer

    Example
    -------
    >>> # Classification with SMOTE (reproducible)
    >>> transformer = build_imbalance_transformer(
    ...     'smote', task_type='classification', random_state=42, k_neighbors=5
    ... )
    >>>
    >>> # Regression with binning
    >>> transformer = build_imbalance_transformer(
    ...     'binning', task_type='regression', random_state=42, n_bins=5
    ... )
    """
    if task_type == 'classification':
        return ClassificationResampler(method=method, random_state=random_state, **params)

    elif task_type == 'regression':
        if method == 'undersample':
            # RegressionUndersampler already accepts random_state in params
            if random_state is not None and 'random_state' not in params:
                params['random_state'] = random_state
            return RegressionUndersampler(**params)
        elif method in ['binning', 'rare_boost', 'balanced']:
            return RegressionSampleWeighter(strategy=method, **params)
        else:
            raise ValueError(
                f"Unknown regression method: {method}. "
                f"Use 'undersample', 'binning', 'rare_boost', or 'balanced'."
            )

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


# ============================================================================
# METHOD INFORMATION
# ============================================================================

# Comprehensive method descriptions for UI display and user guidance
CLASSIFICATION_METHOD_INFO = {
    'smote': {
        'name': 'SMOTE',
        'short': 'SMOTE - Synthetic oversampling (standard)',
        'description': (
            'Synthetic Minority Over-sampling Technique. Creates synthetic minority class '
            'samples by interpolating between existing minority samples and their k-nearest '
            'neighbors. For each minority sample, it randomly selects one of its k neighbors '
            'and creates a new sample along the line connecting them.'
        ),
        'when_to_use': (
            'Best for moderate to severe imbalance (3:1 to 10:1 ratio) when you have '
            'enough minority samples (>10) and want to increase minority representation '
            'without losing majority class information.'
        ),
        'pros': [
            'Creates diverse synthetic samples (not just duplicates)',
            'Preserves all majority class samples',
            'Works well with most classifiers'
        ],
        'cons': [
            'Can create noisy samples if classes overlap',
            'Requires enough minority samples for k-neighbors',
            'May not work well in very high dimensions'
        ],
        'key_params': {'k_neighbors': 'Number of nearest neighbors (default: 5)'}
    },
    'adasyn': {
        'name': 'ADASYN',
        'short': 'ADASYN - Adaptive synthetic sampling',
        'description': (
            'Adaptive Synthetic Sampling. Like SMOTE, but generates more synthetic samples '
            'in regions where the minority class is harder to learn (near the decision '
            'boundary). It adaptively shifts focus to difficult examples.'
        ),
        'when_to_use': (
            'When minority class samples are not uniformly distributed and some regions '
            'are harder to classify than others. Good when you want the model to focus '
            'on difficult boundary cases.'
        ),
        'pros': [
            'Focuses on hard-to-learn regions',
            'Reduces bias from easy examples',
            'Often improves classifier performance on difficult cases'
        ],
        'cons': [
            'Can over-generate in noisy regions',
            'May increase overfitting on outliers',
            'Slightly more computationally expensive than SMOTE'
        ],
        'key_params': {'n_neighbors': 'Number of nearest neighbors (default: 5)'}
    },
    'borderline_smote': {
        'name': 'BorderlineSMOTE',
        'short': 'BorderlineSMOTE - Focus on borderline cases',
        'description': (
            'A SMOTE variant that only generates synthetic samples from minority class '
            'samples that are near the decision boundary (borderline samples). '
            'Borderline samples are those whose neighbors include both majority and '
            'minority class samples.'
        ),
        'when_to_use': (
            'When you want to strengthen the decision boundary without adding samples '
            'in already-pure regions. Useful when classes have clear separation in '
            'some areas but overlap in others.'
        ),
        'pros': [
            'More targeted than regular SMOTE',
            'Strengthens weak boundary regions',
            'Less likely to create redundant samples'
        ],
        'cons': [
            'May not help if entire minority class is borderline',
            'Fewer samples generated than regular SMOTE',
            'Requires careful tuning of borderline detection'
        ],
        'key_params': {'k_neighbors': 'Neighbors for SMOTE (default: 5)',
                       'm_neighbors': 'Neighbors for borderline detection (default: 10)'}
    },
    'random_undersampler': {
        'name': 'Random Undersampling',
        'short': 'Random undersampling of majority class',
        'description': (
            'Randomly removes samples from the majority class until class balance is '
            'achieved. Simple and fast, but discards potentially useful information.'
        ),
        'when_to_use': (
            'When you have a very large dataset and can afford to lose majority samples. '
            'Good for extreme imbalance (>10:1) with abundant data, or when training '
            'time is a concern.'
        ),
        'pros': [
            'Simple and very fast',
            'Reduces training time significantly',
            'No synthetic data creation (uses only real samples)',
            'Works with any sample size'
        ],
        'cons': [
            'Discards potentially useful majority samples',
            'May lose important patterns in majority class',
            'Results can vary between runs (random selection)'
        ],
        'key_params': {'sampling_strategy': 'Target ratio (default: auto = equal classes)'}
    },
    'tomek_links': {
        'name': 'Tomek Links',
        'short': 'Tomek Links - Remove boundary noise',
        'description': (
            'Identifies and removes Tomek links from the dataset. A Tomek link is a pair '
            'of samples from different classes that are each other\'s nearest neighbor. '
            'Removing the majority class sample from each link cleans the decision boundary.'
        ),
        'when_to_use': (
            'When classes overlap at the boundary and you want to clean up noise without '
            'aggressive undersampling. Often used as a post-processing step after SMOTE.'
        ),
        'pros': [
            'Cleans noisy decision boundaries',
            'Removes only ambiguous samples',
            'Improves classifier decision boundary'
        ],
        'cons': [
            'Removes relatively few samples',
            'May not significantly reduce imbalance',
            'Computationally expensive for large datasets'
        ],
        'key_params': {}
    },
    'smote_tomek': {
        'name': 'SMOTETomek',
        'short': 'SMOTETomek - Combined over/undersampling',
        'description': (
            'Combines SMOTE oversampling with Tomek links cleaning. First applies SMOTE '
            'to increase minority samples, then removes Tomek links to clean up the '
            'decision boundary. Provides both balance and boundary clarity.'
        ),
        'when_to_use': (
            'For severe to extreme imbalance when you want both increased minority '
            'representation AND a cleaner decision boundary. Best when classes have '
            'overlapping regions that confuse the classifier.'
        ),
        'pros': [
            'Best of both worlds: more minority samples + cleaner boundary',
            'Reduces noise introduced by SMOTE',
            'Often provides best overall performance'
        ],
        'cons': [
            'More computationally expensive',
            'Two-step process with more parameters',
            'May remove too many samples if classes overlap significantly'
        ],
        'key_params': {'k_neighbors': 'SMOTE neighbors (default: 5)'}
    },
    'class_weight': {
        'name': 'Class Weights',
        'short': 'Class weights - No resampling, weight loss function',
        'description': (
            'Does not modify the dataset. Instead, adjusts the loss function to penalize '
            'misclassification of minority class samples more heavily. The weight is '
            'typically inversely proportional to class frequency.'
        ),
        'when_to_use': (
            'When you have very few minority samples (<10) and resampling would be '
            'unreliable, or when you want to preserve the original data distribution. '
            'Also good for very small datasets where any data loss is costly.'
        ),
        'pros': [
            'No data modification - preserves original distribution',
            'Works with any sample size (even 1 per class)',
            'No risk of creating unrealistic synthetic samples',
            'Fast - no preprocessing step needed'
        ],
        'cons': [
            'Model must support sample_weight or class_weight',
            'May not be as effective as resampling for severe imbalance',
            'Does not address overlapping classes'
        ],
        'key_params': {}
    }
}

REGRESSION_METHOD_INFO = {
    'undersample': {
        'name': 'Undersampling',
        'short': 'Undersample over-represented ranges (e.g., many zeros)',
        'description': (
            'Divides the target range into bins and randomly removes samples from '
            'over-represented bins until a more balanced distribution is achieved. '
            'Ideal for datasets with many zeros or heavily skewed distributions.'
        ),
        'when_to_use': (
            'When you have many samples concentrated in one target range (e.g., many '
            'zeros in collagen %) and sparse samples in other ranges. Works best when '
            'you have enough total samples to afford losing some.'
        ),
        'pros': [
            'Uses only real samples (no synthetic data)',
            'Balances target distribution across range',
            'Reduces training time'
        ],
        'cons': [
            'Discards potentially useful samples',
            'May lose rare combinations of features',
            'Less effective with very small datasets'
        ],
        'key_params': {'n_bins': 'Number of target bins (default: 10)',
                       'sampling_strategy': 'Target count method (auto/mean/float)'}
    },
    'binning': {
        'name': 'Target Binning',
        'short': 'Target binning - Weight by target frequency',
        'description': (
            'Divides targets into bins and computes sample weights inversely proportional '
            'to bin frequency. Samples from sparse target ranges get higher weights, '
            'causing the model to pay more attention to them during training.'
        ),
        'when_to_use': (
            'When you want to preserve all samples but emphasize rare target values. '
            'Good when you cannot afford to lose any data but want the model to '
            'perform better on under-represented target ranges.'
        ),
        'pros': [
            'Preserves all samples',
            'Emphasizes rare target values',
            'Simple and interpretable'
        ],
        'cons': [
            'Requires model that supports sample_weight',
            'May cause overfitting on rare samples',
            'Bin edges can affect results'
        ],
        'key_params': {'n_bins': 'Number of target bins (default: 5)'}
    },
    'rare_boost': {
        'name': 'Rare-Value Boost',
        'short': 'Rare-value boost - Emphasize uncommon targets',
        'description': (
            'Computes sample weights based on how far each target value is from the '
            'median. Samples with target values far from the median (rare values) '
            'receive exponentially higher weights.'
        ),
        'when_to_use': (
            'For severe target imbalance where extreme values are rare but important. '
            'Useful when predicting unusual conditions (very high or very low values) '
            'is more critical than predicting common values accurately.'
        ),
        'pros': [
            'Strongly emphasizes tail values',
            'Continuous weighting (no binning artifacts)',
            'Preserves all samples'
        ],
        'cons': [
            'May cause overfitting on extreme values',
            'Less interpretable than binning',
            'Boost factor requires tuning'
        ],
        'key_params': {'boost_factor': 'Weight multiplier for rare values (default: 2.0)'}
    },
    'balanced': {
        'name': 'Balanced Weights',
        'short': 'Balanced - Simple inverse frequency',
        'description': (
            'Treats target values as discrete categories and applies inverse frequency '
            'weighting (similar to classification class_weight="balanced"). Simple '
            'approach that works when target values cluster into natural groups.'
        ),
        'when_to_use': (
            'When target values naturally cluster into groups (e.g., discrete '
            'concentration levels) rather than being continuous. Also useful as a '
            'simple baseline before trying more complex methods.'
        ),
        'pros': [
            'Simple and fast',
            'Automatic weight calculation',
            'Works well for clustered targets'
        ],
        'cons': [
            'Treats continuous values as discrete',
            'May not work well for truly continuous targets',
            'Can be unstable with many unique values'
        ],
        'key_params': {}
    }
}


def get_method_info(method: str, task_type: str = 'classification') -> dict:
    """
    Get detailed information about a specific imbalance handling method.

    Parameters
    ----------
    method : str
        Method name (e.g., 'smote', 'adasyn', 'binning')
    task_type : str
        'classification' or 'regression'

    Returns
    -------
    dict
        Dictionary with keys: name, short, description, when_to_use, pros, cons, key_params

    Example
    -------
    >>> info = get_method_info('smote', 'classification')
    >>> print(info['description'])
    >>> print(info['when_to_use'])
    >>> print(info['pros'])
    """
    if task_type == 'classification':
        info_dict = CLASSIFICATION_METHOD_INFO
    else:
        info_dict = REGRESSION_METHOD_INFO

    method_key = method.lower().replace('-', '_')
    if method_key in info_dict:
        return info_dict[method_key]
    else:
        return {
            'name': method,
            'short': method,
            'description': 'No detailed information available for this method.',
            'when_to_use': '',
            'pros': [],
            'cons': [],
            'key_params': {}
        }


def get_all_method_info(task_type: str = 'classification') -> dict:
    """
    Get detailed information about all available methods for a task type.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'

    Returns
    -------
    dict
        Dictionary mapping method names to their info dictionaries
    """
    if task_type == 'classification':
        return CLASSIFICATION_METHOD_INFO.copy()
    else:
        return REGRESSION_METHOD_INFO.copy()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_methods(task_type='classification'):
    """
    Get list of available imbalance handling methods.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'

    Returns
    -------
    list of tuples
        (method_name, description) pairs
    """
    if task_type == 'classification':
        if not HAS_IMBLEARN:
            return [('class_weight', 'Model-based class weighting (no resampling)')]

        return [
            ('smote', 'SMOTE - Synthetic oversampling (standard)'),
            ('adasyn', 'ADASYN - Adaptive synthetic sampling'),
            ('borderline_smote', 'BorderlineSMOTE - Focus on borderline cases'),
            ('random_undersampler', 'Random undersampling of majority class'),
            ('tomek_links', 'Tomek Links - Remove boundary noise'),
            ('smote_tomek', 'SMOTETomek - Combined over/undersampling'),
            ('class_weight', 'Class weights - No resampling, weight loss function')
        ]

    elif task_type == 'regression':
        return [
            ('undersample', 'Undersample over-represented ranges (e.g., many zeros)'),
            ('binning', 'Target binning - Weight by target frequency'),
            ('rare_boost', 'Rare-value boost - Emphasize uncommon targets'),
            ('balanced', 'Balanced - Simple inverse frequency')
        ]

    else:
        return []


def recommend_imbalance_method(y, task_type='classification') -> ImbalanceRecommendation:
    """
    Intelligently recommend imbalance handling method based on data characteristics.

    This function considers:
    - Imbalance severity (ratio of majority to minority)
    - Absolute sample counts (determines if undersampling is viable)
    - Number of samples in minority class (determines if SMOTE is viable)

    Parameters
    ----------
    y : array-like
        Target values
    task_type : str
        'classification' or 'regression'

    Returns
    -------
    ImbalanceRecommendation
        Dataclass with recommended_method, reason, alternative, and warnings
    """
    warnings_list = []

    if task_type == 'classification':
        info = detect_class_imbalance(y)

        if info.severity == 'none':
            return ImbalanceRecommendation(
                recommended_method=None,
                reason='Data is balanced, no imbalance handling needed',
                alternative=None,
                warnings=[]
            )

        # Get sample counts
        class_counts = info.class_counts
        minority_count = class_counts[info.minority_class]
        majority_count = class_counts[info.majority_class]
        total_samples = len(y)

        # Decision logic based on sample availability

        # Case 1: Plenty of samples (>500 total, minority >100)
        if total_samples > 500 and minority_count > 100:
            if info.severity == 'moderate':
                return ImbalanceRecommendation(
                    recommended_method='random_undersampler',
                    reason=f'Plenty of samples ({majority_count} majority). Undersampling is efficient and preserves real data.',
                    alternative='class_weight',
                    warnings=warnings_list
                )
            else:  # severe or extreme
                return ImbalanceRecommendation(
                    recommended_method='smote_tomek',
                    reason=f'Large imbalance but sufficient samples. Combined method balances data while removing noise.',
                    alternative='smote',
                    warnings=warnings_list
                )

        # Case 2: Moderate samples (200-500 total OR minority 50-100)
        elif total_samples > 200 or minority_count >= 50:
            if minority_count < 10:
                warnings_list.append(f'Very few minority samples ({minority_count}). SMOTE may create unrealistic synthetic data.')
                return ImbalanceRecommendation(
                    recommended_method='class_weight',
                    reason=f'Too few minority samples ({minority_count}) for reliable oversampling. Class weights are safer.',
                    alternative=None,
                    warnings=warnings_list
                )
            else:
                return ImbalanceRecommendation(
                    recommended_method='smote',
                    reason=f'Moderate dataset. SMOTE creates synthetic minority samples without losing majority data.',
                    alternative='adasyn',
                    warnings=warnings_list
                )

        # Case 3: Small dataset (<200 total OR minority <50)
        else:
            if minority_count < 10:
                warnings_list.append(f'Only {minority_count} minority samples. All resampling methods may be unreliable.')
                warnings_list.append('Consider collecting more data if possible.')
                return ImbalanceRecommendation(
                    recommended_method='class_weight',
                    reason=f'Very small dataset ({total_samples} samples). Class weights avoid data manipulation.',
                    alternative=None,
                    warnings=warnings_list
                )
            else:
                if majority_count > 3 * minority_count:
                    warnings_list.append(f'Small dataset with imbalance. Results may be unstable.')
                return ImbalanceRecommendation(
                    recommended_method='class_weight',
                    reason=f'Small dataset. Class weights are most reliable without duplicating limited data.',
                    alternative='smote',
                    warnings=warnings_list
                )

    elif task_type == 'regression':
        info = detect_regression_imbalance(y)

        if info.severity == 'none':
            return ImbalanceRecommendation(
                recommended_method=None,
                reason='Target distribution is balanced',
                alternative=None,
                warnings=[]
            )

        n_samples = info.n_samples

        if n_samples < 100:
            warnings_list.append(f'Small dataset ({n_samples} samples). Imbalance handling may have limited effect.')

        if info.severity == 'moderate':
            return ImbalanceRecommendation(
                recommended_method='binning',
                reason='Moderate target imbalance - binning with weights emphasizes rare ranges',
                alternative='rare_boost',
                warnings=warnings_list
            )
        else:  # severe
            return ImbalanceRecommendation(
                recommended_method='rare_boost',
                reason='Severe target imbalance - exponentially boost rare values',
                alternative='binning',
                warnings=warnings_list
            )

    return ImbalanceRecommendation(
        recommended_method=None,
        reason='Unknown task type',
        alternative=None,
        warnings=['Unknown task type specified']
    )


# ============================================================================
# UPFRONT VALIDATION
# ============================================================================

def validate_classification_config(y, imbalance_method, imbalance_params=None, n_folds=5):
    """
    Validate that the imbalance method is compatible with the data.

    Call this BEFORE starting training to give immediate feedback to the user.
    This prevents wasting time on long training runs that will fail partway through.

    Parameters
    ----------
    y : array-like
        Target labels (classification)
    imbalance_method : str or None
        The imbalance handling method selected (e.g., 'smote', 'adasyn', etc.)
    imbalance_params : dict, optional
        Parameters for the imbalance method (e.g., {'k_neighbors': 5})
    n_folds : int, default=5
        Number of cross-validation folds

    Returns
    -------
    bool
        True if configuration is valid

    Raises
    ------
    ValueError
        If configuration is invalid, with clear message and suggestions

    Example
    -------
    >>> # Before starting training, validate the configuration
    >>> validate_classification_config(y_train, 'smote', {'k_neighbors': 5}, n_folds=5)
    True

    >>> # This will raise a helpful error:
    >>> validate_classification_config(y_with_3_samples_in_minority, 'smote', {'k_neighbors': 5})
    ValueError: SMOTE with k_neighbors=5 requires at least 6 samples per class,
                but class 'minority_class' has only 3.
                Options: set k_neighbors<=2, use random_undersampler, or use class_weight.
    """
    if imbalance_method is None or imbalance_method == 'class_weight':
        return True  # No resampling validation needed

    if imbalance_params is None:
        imbalance_params = {}

    y = np.asarray(y)
    class_counts = Counter(y)

    if len(class_counts) < 2:
        raise ValueError(
            f"Classification requires at least 2 classes, but only 1 class found in data. "
            f"Check your target variable."
        )

    min_class_count = min(class_counts.values())
    min_class_name = min(class_counts, key=class_counts.get)

    # Check 1: Enough samples for CV folds (each class needs at least n_folds samples)
    if min_class_count < n_folds:
        raise ValueError(
            f"Class '{min_class_name}' has only {min_class_count} samples, "
            f"but {n_folds}-fold stratified CV requires at least {n_folds} samples per class.\n\n"
            f"Options:\n"
            f"  1. Reduce CV folds to {min_class_count} or fewer\n"
            f"  2. Use class_weight='balanced' instead of resampling\n"
            f"  3. Collect more samples for the minority class"
        )

    # Check 2: SMOTE-based methods require k_neighbors + 1 samples per class
    smote_methods = ('smote', 'adasyn', 'borderline_smote', 'smote_tomek', 'smote_enn')
    if imbalance_method.lower().replace('-', '_') in smote_methods:
        k_neighbors = imbalance_params.get('k_neighbors', 5)
        required_samples = k_neighbors + 1

        if min_class_count < required_samples:
            max_valid_k = min_class_count - 1
            raise ValueError(
                f"SMOTE with k_neighbors={k_neighbors} requires at least {required_samples} samples "
                f"per class, but class '{min_class_name}' has only {min_class_count}.\n\n"
                f"Options:\n"
                f"  1. Set k_neighbors<={max_valid_k} (current data supports k_neighbors up to {max_valid_k})\n"
                f"  2. Use 'random_undersampler' instead (works with any class size)\n"
                f"  3. Use 'class_weight' instead (no resampling, adjusts loss function)\n"
                f"  4. Collect more samples for the minority class"
            )

    # Check 3: Warn about potential issues with very small classes in CV folds
    # With stratified CV, samples are distributed across folds
    # A class with exactly n_folds samples will have only 1 sample per training fold minority
    samples_per_fold_train = (min_class_count * (n_folds - 1)) // n_folds
    if samples_per_fold_train < 3:
        warnings.warn(
            f"Class '{min_class_name}' has {min_class_count} samples. "
            f"With {n_folds}-fold CV, some training folds may have only ~{samples_per_fold_train} "
            f"samples of this class, which may produce unstable results. "
            f"Consider using fewer folds or class_weight method.",
            UserWarning
        )

    return True


def validate_imbalance_with_features(X, y, imbalance_method, imbalance_params=None, n_folds=5):
    """
    Extended validation that also checks feature dimensions.

    This should be called when feature matrix X is available, to warn about
    high-dimensional data issues with SMOTE.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like
        Target labels
    imbalance_method : str or None
        The imbalance handling method
    imbalance_params : dict, optional
        Parameters for the method
    n_folds : int, default=5
        Number of CV folds

    Returns
    -------
    bool
        True if configuration is valid

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    # First run basic validation
    validate_classification_config(y, imbalance_method, imbalance_params, n_folds)

    if imbalance_method is None or imbalance_method == 'class_weight':
        return True

    X = np.asarray(X)
    n_samples, n_features = X.shape

    # Check for high-dimensional data with SMOTE
    smote_methods = ('smote', 'adasyn', 'borderline_smote', 'smote_tomek', 'smote_enn')
    if imbalance_method.lower().replace('-', '_') in smote_methods:
        if n_features > 500:
            warnings.warn(
                f"Using SMOTE with {n_features} features (high-dimensional data). "
                f"SMOTE uses k-nearest neighbors, which can be unreliable in high dimensions "
                f"due to the 'curse of dimensionality'. Consider:\n"
                f"  1. Applying dimensionality reduction (PCA, variable selection) before SMOTE\n"
                f"  2. Using 'class_weight' instead, which doesn't depend on distance metrics\n"
                f"  3. Using 'random_undersampler' which randomly samples without distances",
                UserWarning
            )

    return True
