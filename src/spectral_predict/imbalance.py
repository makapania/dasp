"""
Data Imbalance Handling for Spectral Prediction

This module provides comprehensive imbalance handling methods for both
classification and regression tasks in spectroscopy:

CLASSIFICATION METHODS:
- SMOTE: Synthetic Minority Over-sampling Technique
- ADASYN: Adaptive Synthetic Sampling
- BorderlineSMOTE: SMOTE variant focusing on borderline samples
- RandomUnderSampler: Random majority class undersampling
- TomekLinks: Remove Tomek links (noise at class boundaries)
- SMOTETomek: Combined over/undersampling

REGRESSION METHODS:
- Target binning with sample weights
- Rare-value boosting (emphasize underrepresented target ranges)

All methods are sklearn-compatible and integrate seamlessly with pipelines.

Example:
    >>> from spectral_predict.imbalance import build_imbalance_transformer
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> # Classification with SMOTE
    >>> pipe = Pipeline([
    ...     ('imbalance', build_imbalance_transformer('smote', k_neighbors=5)),
    ...     ('model', RandomForestClassifier())
    ... ])
    >>> pipe.fit(X_train, y_train)

    >>> # Regression with target binning
    >>> pipe = Pipeline([
    ...     ('imbalance', build_imbalance_transformer('binning', n_bins=5)),
    ...     ('model', PLSRegression())
    ... ])
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import compute_sample_weight
from collections import Counter
import warnings

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
# DETECTION & ANALYSIS
# ============================================================================

def detect_class_imbalance(y, threshold=3.0):
    """
    Detect class imbalance in classification targets.

    Parameters:
    -----------
    y : array-like
        Target labels (classification)
    threshold : float, default=3.0
        Imbalance ratio threshold (majority:minority) above which to flag

    Returns:
    --------
    dict with keys:
        - 'is_imbalanced': bool
        - 'imbalance_ratio': float (majority_count / minority_count)
        - 'class_counts': dict {class: count}
        - 'majority_class': class label
        - 'minority_class': class label
        - 'severity': str ('none', 'moderate', 'severe', 'extreme')
        - 'recommendation': str (suggested method)
    """
    y = np.asarray(y)
    class_counts = Counter(y)

    if len(class_counts) < 2:
        return {
            'is_imbalanced': False,
            'imbalance_ratio': 1.0,
            'class_counts': dict(class_counts),
            'majority_class': None,
            'minority_class': None,
            'severity': 'none',
            'recommendation': 'No imbalance detected (single class)'
        }

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

    return {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': imbalance_ratio,
        'class_counts': dict(class_counts),
        'majority_class': majority_class,
        'minority_class': minority_class,
        'severity': severity,
        'recommendation': recommendation
    }


def detect_regression_imbalance(y, n_bins=10, coverage_threshold=0.2):
    """
    Detect target imbalance in regression (uneven distribution across range).

    Parameters:
    -----------
    y : array-like
        Target values (regression)
    n_bins : int, default=10
        Number of bins to divide target range
    coverage_threshold : float, default=0.2
        Minimum fraction of samples per bin for balanced distribution

    Returns:
    --------
    dict with keys:
        - 'is_imbalanced': bool
        - 'bin_counts': array (samples per bin)
        - 'bin_edges': array (bin boundaries)
        - 'sparse_bins': list (bin indices with <coverage_threshold samples)
        - 'coverage': float (min_bin_count / mean_bin_count)
        - 'severity': str ('none', 'moderate', 'severe')
        - 'recommendation': str (suggested method)
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

    return {
        'is_imbalanced': is_imbalanced,
        'bin_counts': bin_counts,
        'bin_edges': bin_edges,
        'sparse_bins': sparse_bins,
        'coverage': coverage,
        'severity': severity,
        'recommendation': recommendation,
        'n_samples': len(y),
        'target_range': (float(y.min()), float(y.max()))
    }


# ============================================================================
# CLASSIFICATION IMBALANCE TRANSFORMERS
# ============================================================================

class ClassificationResampler(BaseEstimator):
    """
    Wrapper for imbalanced-learn resampling methods that works in pipelines.

    This transformer applies resampling using fit_resample() for use with
    imblearn Pipeline.

    Parameters:
    -----------
    method : str or object
        Resampling method name ('smote', 'adasyn', etc.) or imblearn object
    **params : dict
        Parameters to pass to the resampling method

    Example:
    --------
    >>> resampler = ClassificationResampler('smote', k_neighbors=5)
    >>> X_res, y_res = resampler.fit_resample(X_train, y_train)

    Note:
    -----
    This class should NOT inherit from TransformerMixin because it implements
    fit_resample() for use with imblearn Pipeline. TransformerMixin would add
    a transform() method that conflicts with fit_resample() semantics.
    """

    def __init__(self, method='smote', **params):
        if not HAS_IMBLEARN:
            raise ImportError(
                "imbalanced-learn (imblearn) is required for resampling methods. "
                "Install with: pip install imbalanced-learn"
            )

        self.method = method
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
            self.resampler_ = resampler_class(**self.params)
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

    Parameters:
    -----------
    n_bins : int, default=10
        Number of bins to divide target range
    sampling_strategy : str or float, default='auto'
        How to determine target samples per bin:
        - 'auto': Undersample to median bin count
        - 'mean': Undersample to mean bin count
        - float (0-1): Keep this fraction of samples in over-represented bins
    random_state : int, default=42
        Random seed for reproducibility

    Example:
    --------
    >>> # Dataset with many zeros (e.g., collagen % from 0-19%)
    >>> undersampler = RegressionUndersampler(n_bins=10, sampling_strategy='auto')
    >>> X_res, y_res = undersampler.fit_resample(X, y)
    >>> print(f"Original: {len(y)} samples")
    >>> print(f"Resampled: {len(y_res)} samples")

    Note:
    -----
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
        np.random.seed(self.random_state)
        indices_to_keep = []

        for bin_idx in range(self.n_bins):
            bin_mask = bin_indices == bin_idx
            bin_sample_indices = np.where(bin_mask)[0]
            n_samples_in_bin = len(bin_sample_indices)

            if n_samples_in_bin > target_count:
                # Randomly select target_count samples
                selected = np.random.choice(bin_sample_indices, size=target_count, replace=False)
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

    Parameters:
    -----------
    strategy : str, default='binning'
        Weighting strategy:
        - 'binning': Bin targets and weight inversely by bin frequency
        - 'rare_boost': Exponentially boost rare target values
        - 'balanced': Simple inverse frequency weighting
    n_bins : int, default=5
        Number of bins for 'binning' strategy
    boost_factor : float, default=2.0
        Boost multiplier for 'rare_boost' strategy

    Attributes:
    -----------
    sample_weight_ : array
        Computed sample weights (stored after fit)

    Example:
    --------
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

def build_imbalance_transformer(method, task_type='classification', **params):
    """
    Factory function to create imbalance handling transformers.

    Parameters:
    -----------
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

    **params : dict
        Method-specific parameters

    Returns:
    --------
    transformer : BaseEstimator
        sklearn-compatible transformer

    Example:
    --------
    >>> # Classification with SMOTE
    >>> transformer = build_imbalance_transformer(
    ...     'smote', task_type='classification', k_neighbors=5
    ... )
    >>>
    >>> # Regression with binning
    >>> transformer = build_imbalance_transformer(
    ...     'binning', task_type='regression', n_bins=5
    ... )
    """
    if task_type == 'classification':
        return ClassificationResampler(method=method, **params)

    elif task_type == 'regression':
        if method == 'undersample':
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
# UTILITY FUNCTIONS
# ============================================================================

def get_available_methods(task_type='classification'):
    """
    Get list of available imbalance handling methods.

    Parameters:
    -----------
    task_type : str
        'classification' or 'regression'

    Returns:
    --------
    list of tuples: (method_name, description)
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


def recommend_imbalance_method(y, task_type='classification'):
    """
    Intelligently recommend imbalance handling method based on data characteristics.

    This function considers:
    - Imbalance severity (ratio of majority to minority)
    - Absolute sample counts (determines if undersampling is viable)
    - Number of samples in minority class (determines if SMOTE is viable)

    Parameters:
    -----------
    y : array-like
        Target values
    task_type : str
        'classification' or 'regression'

    Returns:
    --------
    dict with keys:
        - 'recommended_method': str
        - 'reason': str
        - 'alternative': str
        - 'warnings': list of str (potential issues with data)
    """
    warnings_list = []

    if task_type == 'classification':
        info = detect_class_imbalance(y)

        if info['severity'] == 'none':
            return {
                'recommended_method': None,
                'reason': 'Data is balanced, no imbalance handling needed',
                'alternative': None,
                'warnings': []
            }

        # Get sample counts
        class_counts = info['class_counts']
        minority_count = class_counts[info['minority_class']]
        majority_count = class_counts[info['majority_class']]
        total_samples = len(y)

        # Decision logic based on sample availability

        # Case 1: Plenty of samples (>500 total, minority >100)
        if total_samples > 500 and minority_count > 100:
            if info['severity'] == 'moderate':
                return {
                    'recommended_method': 'random_undersampler',
                    'reason': f'Plenty of samples ({majority_count} majority). Undersampling is efficient and preserves real data.',
                    'alternative': 'class_weight',
                    'warnings': warnings_list
                }
            else:  # severe or extreme
                return {
                    'recommended_method': 'smote_tomek',
                    'reason': f'Large imbalance but sufficient samples. Combined method balances data while removing noise.',
                    'alternative': 'smote',
                    'warnings': warnings_list
                }

        # Case 2: Moderate samples (200-500 total OR minority 50-100)
        elif total_samples > 200 or minority_count >= 50:
            if minority_count < 10:
                warnings_list.append(f'Very few minority samples ({minority_count}). SMOTE may create unrealistic synthetic data.')
                return {
                    'recommended_method': 'class_weight',
                    'reason': f'Too few minority samples ({minority_count}) for reliable oversampling. Class weights are safer.',
                    'alternative': None,
                    'warnings': warnings_list
                }
            else:
                return {
                    'recommended_method': 'smote',
                    'reason': f'Moderate dataset. SMOTE creates synthetic minority samples without losing majority data.',
                    'alternative': 'adasyn',
                    'warnings': warnings_list
                }

        # Case 3: Small dataset (<200 total OR minority <50)
        else:
            if minority_count < 10:
                warnings_list.append(f'Only {minority_count} minority samples. All resampling methods may be unreliable.')
                warnings_list.append('Consider collecting more data if possible.')
                return {
                    'recommended_method': 'class_weight',
                    'reason': f'Very small dataset ({total_samples} samples). Class weights avoid data manipulation.',
                    'alternative': None,
                    'warnings': warnings_list
                }
            else:
                if majority_count > 3 * minority_count:
                    warnings_list.append(f'Small dataset with imbalance. Results may be unstable.')
                return {
                    'recommended_method': 'class_weight',
                    'reason': f'Small dataset. Class weights are most reliable without duplicating limited data.',
                    'alternative': 'smote',
                    'warnings': warnings_list
                }

    elif task_type == 'regression':
        info = detect_regression_imbalance(y)

        if info['severity'] == 'none':
            return {
                'recommended_method': None,
                'reason': 'Target distribution is balanced',
                'alternative': None,
                'warnings': []
            }

        n_samples = info['n_samples']

        if n_samples < 100:
            warnings_list.append(f'Small dataset ({n_samples} samples). Imbalance handling may have limited effect.')

        if info['severity'] == 'moderate':
            return {
                'recommended_method': 'binning',
                'reason': 'Moderate target imbalance - binning with weights emphasizes rare ranges',
                'alternative': 'rare_boost',
                'warnings': warnings_list
            }
        else:  # severe
            return {
                'recommended_method': 'rare_boost',
                'reason': 'Severe target imbalance - exponentially boost rare values',
                'alternative': 'binning',
                'warnings': warnings_list
            }

    return {
        'recommended_method': None,
        'reason': 'Unknown task type',
        'alternative': None,
        'warnings': ['Unknown task type specified']
    }
