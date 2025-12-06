"""
Tests for class imbalance detection in Spectral Predict v3.

This module provides comprehensive tests for:
- Balanced dataset detection (should return no warning)
- Imbalanced dataset detection (90/10 split)
- Extreme imbalance detection (99/1 split)
- Regression imbalance with skewed distributions
- Multi-class classification (3+ classes)
- Threshold sensitivity
"""

import pytest
import numpy as np
from spectral_predict_v3.core.imbalance import (
    detect_class_imbalance,
    detect_regression_imbalance,
    detect_multiclass_imbalance,
    format_imbalance_warning,
    format_regression_imbalance_warning,
    ClassImbalanceResult,
    RegressionImbalanceResult,
    MultiClassImbalanceResult
)


# ============================================================================
# CLASSIFICATION IMBALANCE TESTS
# ============================================================================

def test_balanced_binary_classification():
    """Test that balanced datasets are not flagged as imbalanced."""
    # Perfect balance: 50/50
    y = np.array([0] * 50 + [1] * 50)
    result = detect_class_imbalance(y)

    assert isinstance(result, ClassImbalanceResult)
    assert not result.is_imbalanced
    assert result.severity == 'none'
    assert result.imbalance_ratio == 1.0
    assert result.class_counts[0] == 50
    assert result.class_counts[1] == 50


def test_slightly_imbalanced_binary():
    """Test detection with slight imbalance (2:1 ratio, below default threshold)."""
    # 2:1 ratio (below default threshold of 3.0)
    y = np.array([0] * 66 + [1] * 33)
    result = detect_class_imbalance(y)

    assert not result.is_imbalanced
    assert result.severity == 'none'
    assert result.imbalance_ratio == pytest.approx(2.0, rel=0.01)


def test_moderate_imbalance():
    """Test detection of moderate imbalance (4:1 ratio)."""
    # 4:1 ratio - moderate imbalance
    y = np.array([0] * 80 + [1] * 20)
    result = detect_class_imbalance(y)

    assert result.is_imbalanced
    assert result.severity == 'moderate'
    assert result.imbalance_ratio == pytest.approx(4.0, rel=0.01)
    assert result.majority_class == 0
    assert result.minority_class == 1
    assert 'class_weight' in result.recommendation.lower() or 'smote' in result.recommendation.lower()


def test_severe_imbalance_90_10():
    """Test detection of severe imbalance (90/10 split = 9:1 ratio)."""
    # 9:1 ratio - severe imbalance
    y = np.array([0] * 90 + [1] * 10)
    result = detect_class_imbalance(y)

    assert result.is_imbalanced
    assert result.severity == 'severe'
    assert result.imbalance_ratio == pytest.approx(9.0, rel=0.01)
    assert result.n_outliers is None or hasattr(result, 'class_counts')  # Just checking structure
    assert 'smote' in result.recommendation.lower() or 'adasyn' in result.recommendation.lower()


def test_extreme_imbalance_99_1():
    """Test detection of extreme imbalance (99/1 split = 99:1 ratio)."""
    # 99:1 ratio - extreme imbalance
    y = np.array([0] * 99 + [1] * 1)
    result = detect_class_imbalance(y)

    assert result.is_imbalanced
    assert result.severity == 'extreme'
    assert result.imbalance_ratio == pytest.approx(99.0, rel=0.01)
    assert 'undersampling' in result.recommendation.lower() or 'smote' in result.recommendation.lower()


def test_single_class():
    """Test handling of single-class data (edge case)."""
    y = np.array([0] * 100)
    result = detect_class_imbalance(y)

    assert not result.is_imbalanced
    assert result.severity == 'none'
    assert result.imbalance_ratio == 1.0
    assert result.majority_class is None
    assert result.minority_class is None


def test_custom_threshold():
    """Test threshold sensitivity."""
    y = np.array([0] * 60 + [1] * 40)  # 1.5:1 ratio

    # With default threshold (3.0), should not be imbalanced
    result_default = detect_class_imbalance(y, threshold=3.0)
    assert not result_default.is_imbalanced

    # With lower threshold (1.2), should be imbalanced
    result_low = detect_class_imbalance(y, threshold=1.2)
    assert result_low.is_imbalanced


# ============================================================================
# MULTI-CLASS IMBALANCE TESTS
# ============================================================================

def test_balanced_multiclass():
    """Test balanced 3-class dataset."""
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    result = detect_multiclass_imbalance(y)

    assert isinstance(result, MultiClassImbalanceResult)
    assert not result.is_imbalanced
    assert result.severity == 'none'
    assert result.max_ratio == pytest.approx(1.0, rel=0.01)


def test_imbalanced_multiclass_60_30_10():
    """Test imbalanced 3-class dataset (60/30/10 split)."""
    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    result = detect_multiclass_imbalance(y)

    assert result.is_imbalanced
    assert result.max_ratio == pytest.approx(6.0, rel=0.01)
    assert result.severity in ['severe', 'moderate']
    assert result.min_class == 2
    assert result.max_class == 0
    assert result.class_counts[0] == 60
    assert result.class_counts[1] == 30
    assert result.class_counts[2] == 10


def test_multiclass_pairwise_ratios():
    """Test that pairwise ratios are computed correctly."""
    y = np.array([0] * 40 + [1] * 20 + [2] * 10)
    result = detect_multiclass_imbalance(y)

    # Should have 3 pairwise comparisons for 3 classes
    assert len(result.pairwise_ratios) == 3
    # Check specific ratio
    assert result.pairwise_ratios[(0, 2)] == pytest.approx(4.0, rel=0.01)


def test_multiclass_4_classes():
    """Test with 4 classes."""
    y = np.array([0] * 100 + [1] * 50 + [2] * 25 + [3] * 5)
    result = detect_multiclass_imbalance(y)

    assert result.is_imbalanced
    assert result.max_ratio == pytest.approx(20.0, rel=0.01)  # 100/5
    assert result.severity == 'extreme'
    assert len(result.pairwise_ratios) == 6  # C(4,2) = 6 pairs


# ============================================================================
# REGRESSION IMBALANCE TESTS
# ============================================================================

def test_balanced_regression():
    """Test balanced regression target distribution."""
    # Uniform distribution
    y = np.linspace(0, 10, 100)
    result = detect_regression_imbalance(y)

    assert isinstance(result, RegressionImbalanceResult)
    assert not result.is_imbalanced
    assert result.severity == 'none'
    assert result.coverage > 0.5  # Good coverage


def test_regression_skewed_many_zeros():
    """Test regression with many zeros (common in spectroscopy)."""
    # 80% zeros, 20% distributed from 5-10
    y = np.concatenate([np.zeros(80), np.random.uniform(5, 10, 20)])
    result = detect_regression_imbalance(y, n_bins=10)

    assert result.is_imbalanced
    assert result.severity in ['moderate', 'severe']
    assert len(result.sparse_bins) > 0
    assert result.target_range[0] == pytest.approx(0.0, abs=0.01)


def test_regression_normal_distribution():
    """Test with normal distribution (should be balanced)."""
    np.random.seed(42)
    y = np.random.normal(loc=5.0, scale=1.0, size=1000)
    result = detect_regression_imbalance(y, n_bins=10)

    # Normal distribution should be relatively balanced
    assert result.coverage > 0.3  # Reasonable coverage


def test_regression_bin_count_parameter():
    """Test effect of n_bins parameter."""
    y = np.concatenate([np.zeros(80), np.random.uniform(5, 10, 20)])

    # Fewer bins - less likely to detect imbalance
    result_few = detect_regression_imbalance(y, n_bins=3)
    # More bins - more likely to detect imbalance
    result_many = detect_regression_imbalance(y, n_bins=20)

    # With more bins, sparse bins should be more numerous
    assert len(result_many.sparse_bins) >= len(result_few.sparse_bins)


def test_regression_coverage_threshold():
    """Test coverage threshold sensitivity."""
    y = np.concatenate([np.zeros(80), np.random.uniform(5, 10, 20)])

    # Strict threshold
    result_strict = detect_regression_imbalance(y, coverage_threshold=0.3)
    # Lenient threshold
    result_lenient = detect_regression_imbalance(y, coverage_threshold=0.05)

    # Strict threshold should flag more sparse bins
    assert len(result_strict.sparse_bins) >= len(result_lenient.sparse_bins)


# ============================================================================
# FORMATTING TESTS
# ============================================================================

def test_format_imbalance_warning_balanced():
    """Test formatting of balanced dataset warning."""
    y = np.array([0] * 50 + [1] * 50)
    result = detect_class_imbalance(y)
    msg = format_imbalance_warning(result)

    assert "No class imbalance detected" in msg


def test_format_imbalance_warning_imbalanced():
    """Test formatting of imbalanced dataset warning."""
    y = np.array([0] * 90 + [1] * 10)
    result = detect_class_imbalance(y)
    msg = format_imbalance_warning(result)

    assert "IMBALANCE DETECTED" in msg
    assert "SEVERE" in msg
    assert "9.0:1" in msg or "9:1" in msg
    assert "Recommendation" in msg


def test_format_regression_warning():
    """Test formatting of regression imbalance warning."""
    y = np.concatenate([np.zeros(80), np.random.uniform(5, 10, 20)])
    result = detect_regression_imbalance(y)
    msg = format_regression_imbalance_warning(result)

    if result.is_imbalanced:
        assert "TARGET IMBALANCE" in msg or "IMBALANCE DETECTED" in msg
        assert "Recommendation" in msg
    else:
        assert "No target imbalance detected" in msg


# ============================================================================
# EDGE CASES
# ============================================================================

def test_empty_array():
    """Test handling of empty array."""
    y = np.array([])

    # Should handle gracefully without crashing
    with pytest.raises(Exception):
        detect_class_imbalance(y)


def test_single_sample():
    """Test handling of single sample."""
    y = np.array([0])
    result = detect_class_imbalance(y)

    assert not result.is_imbalanced
    assert result.severity == 'none'


def test_two_samples_different_classes():
    """Test handling of two samples with different classes."""
    y = np.array([0, 1])
    result = detect_class_imbalance(y)

    assert not result.is_imbalanced
    assert result.imbalance_ratio == pytest.approx(1.0, rel=0.01)


def test_string_labels():
    """Test with string class labels."""
    y = np.array(['cat'] * 80 + ['dog'] * 20)
    result = detect_class_imbalance(y)

    assert result.is_imbalanced
    assert result.imbalance_ratio == pytest.approx(4.0, rel=0.01)
    assert result.majority_class == 'cat'
    assert result.minority_class == 'dog'


def test_nan_values_in_regression():
    """Test handling of NaN values in regression targets."""
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    # Should handle NaNs gracefully
    result = detect_regression_imbalance(y)
    # NaN handling depends on numpy's histogram, which should work


def test_identical_regression_values():
    """Test regression with all identical values."""
    y = np.ones(100) * 5.0
    result = detect_regression_imbalance(y)

    # All bins except one will be empty
    assert result.coverage == 0.0  # Min bin has 0, mean has ~10


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_complete_workflow_classification():
    """Test complete workflow: detect -> format -> analyze."""
    y = np.array([0] * 85 + [1] * 15)

    # Detect
    result = detect_class_imbalance(y)
    assert result.is_imbalanced

    # Format
    msg = format_imbalance_warning(result)
    assert len(msg) > 0

    # Analyze
    assert result.class_counts[0] == 85
    assert result.class_counts[1] == 15
    assert result.severity in ['moderate', 'severe', 'extreme']


def test_complete_workflow_multiclass():
    """Test complete workflow for multi-class."""
    y = np.array([0] * 50 + [1] * 30 + [2] * 15 + [3] * 5)

    result = detect_multiclass_imbalance(y)
    assert result.is_imbalanced
    assert result.max_ratio == pytest.approx(10.0, rel=0.01)
    assert len(result.class_counts) == 4


def test_complete_workflow_regression():
    """Test complete workflow for regression."""
    np.random.seed(42)
    y = np.concatenate([
        np.random.uniform(0, 2, 80),  # Many low values
        np.random.uniform(8, 10, 5)   # Few high values
    ])

    result = detect_regression_imbalance(y)
    msg = format_regression_imbalance_warning(result)

    assert result.n_samples == 85
    assert result.target_range[0] < 2.0
    assert result.target_range[1] > 8.0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_large_dataset_performance():
    """Test performance with large dataset (10000+ samples)."""
    import time

    # 10000 samples
    y = np.array([0] * 9000 + [1] * 1000)

    start = time.time()
    result = detect_class_imbalance(y)
    elapsed = time.time() - start

    # Should complete in < 0.1 seconds
    assert elapsed < 0.1
    assert result.is_imbalanced
    assert result.imbalance_ratio == pytest.approx(9.0, rel=0.01)


def test_regression_large_dataset():
    """Test regression detection with large dataset."""
    import time

    np.random.seed(42)
    y = np.random.exponential(scale=2.0, size=10000)

    start = time.time()
    result = detect_regression_imbalance(y)
    elapsed = time.time() - start

    # Should complete quickly
    assert elapsed < 0.5
    assert result.n_samples == 10000
