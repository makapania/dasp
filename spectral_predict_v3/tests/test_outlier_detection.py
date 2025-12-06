"""
Tests for outlier detection in Spectral Predict v3.

This module provides comprehensive tests for:
- Hotelling T² with known outliers
- Q-residuals detection accuracy
- Mahalanobis distance on multivariate data
- PCA score plots generation
- Varying n_components
- Edge cases: single sample, identical samples, NaN handling
- Performance on large datasets (10000+ samples)
"""

import pytest
import numpy as np
from sklearn.decomposition import PCA
from spectral_predict_v3.core.outlier_detection import (
    run_pca_outlier_detection,
    compute_q_residuals,
    compute_mahalanobis_distance,
    check_y_data_consistency,
    generate_outlier_report,
    PCAOutlierResult,
    QResidualResult,
    MahalanobisResult,
    YConsistencyResult,
    OutlierReport
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def normal_spectra():
    """Generate normal spectral data without outliers."""
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 200

    # Generate realistic spectra with Gaussian profiles
    wavelengths = np.linspace(400, 2400, n_wavelengths)
    X = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        # Random Gaussian peak
        center = np.random.uniform(800, 1800)
        width = np.random.uniform(100, 300)
        amplitude = np.random.uniform(0.5, 1.5)
        X[i] = amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))

        # Add noise
        X[i] += np.random.normal(0, 0.01, n_wavelengths)

    return X


@pytest.fixture
def spectra_with_outliers():
    """Generate spectral data with known outliers."""
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 200

    wavelengths = np.linspace(400, 2400, n_wavelengths)
    X = np.zeros((n_samples, n_wavelengths))

    # Normal samples
    for i in range(95):
        center = np.random.uniform(800, 1800)
        width = np.random.uniform(100, 300)
        amplitude = np.random.uniform(0.5, 1.5)
        X[i] = amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
        X[i] += np.random.normal(0, 0.01, n_wavelengths)

    # Outliers: samples 95-99
    # Outlier 1: Very high values
    X[95] = np.ones(n_wavelengths) * 10.0
    # Outlier 2: Very low values
    X[96] = np.ones(n_wavelengths) * -5.0
    # Outlier 3: Spike
    X[97] = np.zeros(n_wavelengths)
    X[97][100] = 20.0
    # Outlier 4: Different shape
    X[98] = np.linspace(0, 5, n_wavelengths)
    # Outlier 5: Random noise
    X[99] = np.random.normal(0, 2.0, n_wavelengths)

    return X


@pytest.fixture
def reference_values_normal():
    """Generate normal reference values."""
    np.random.seed(42)
    return np.random.normal(loc=10.0, scale=2.0, size=100)


@pytest.fixture
def reference_values_with_outliers():
    """Generate reference values with outliers."""
    np.random.seed(42)
    y = np.random.normal(loc=10.0, scale=2.0, size=100)
    # Add outliers
    y[95] = 50.0  # Z-score outlier
    y[96] = -20.0  # Z-score outlier
    return y


# ============================================================================
# PCA OUTLIER DETECTION TESTS
# ============================================================================

def test_pca_outlier_detection_normal_data(normal_spectra):
    """Test PCA outlier detection on normal data (should find few/no outliers)."""
    result = run_pca_outlier_detection(normal_spectra, n_components=5)

    assert isinstance(result, PCAOutlierResult)
    assert result.scores.shape == (100, 5)
    assert result.loadings.shape[1] == 5
    assert len(result.variance_explained) == 5
    assert len(result.hotelling_t2) == 100
    assert result.t2_threshold > 0

    # Should find few outliers in normal data (~5% expected)
    assert result.n_outliers < 10


def test_pca_outlier_detection_with_outliers(spectra_with_outliers):
    """Test PCA outlier detection with known outliers."""
    result = run_pca_outlier_detection(spectra_with_outliers, n_components=5)

    assert result.n_outliers > 0

    # Check that at least some known outliers (95-99) are detected
    known_outlier_indices = [95, 96, 97, 98, 99]
    detected_outliers = set(result.outlier_indices)

    overlap = len(set(known_outlier_indices) & detected_outliers)
    # Should detect at least 2 of the 5 known outliers
    assert overlap >= 2


def test_pca_n_components_parameter(normal_spectra):
    """Test varying n_components parameter."""
    # Test with different numbers of components
    for n_comp in [1, 3, 5, 10]:
        result = run_pca_outlier_detection(normal_spectra, n_components=n_comp)
        expected_comp = min(n_comp, 99, 200)  # min(n_comp, n_samples-1, n_features)
        assert result.scores.shape[1] == expected_comp


def test_pca_single_component(normal_spectra):
    """Test PCA with single component (edge case)."""
    result = run_pca_outlier_detection(normal_spectra, n_components=1)

    assert result.scores.shape == (100, 1)
    assert len(result.hotelling_t2) == 100
    assert result.t2_threshold > 0


def test_pca_variance_explained(normal_spectra):
    """Test that variance explained sums to reasonable value."""
    result = run_pca_outlier_detection(normal_spectra, n_components=5)

    assert np.all(result.variance_explained >= 0)
    assert np.all(result.variance_explained <= 1.0)
    assert np.sum(result.variance_explained) <= 1.0


# ============================================================================
# Q-RESIDUALS TESTS
# ============================================================================

def test_q_residuals_normal_data(normal_spectra):
    """Test Q-residuals on normal data."""
    # First run PCA
    pca_result = run_pca_outlier_detection(normal_spectra, n_components=5)

    # Compute Q-residuals
    q_result = compute_q_residuals(normal_spectra, pca_result.pca_model, n_components=5)

    assert isinstance(q_result, QResidualResult)
    assert len(q_result.q_residuals) == 100
    assert q_result.q_threshold > 0
    assert len(q_result.outlier_flags) == 100

    # Should find ~5% outliers (95th percentile threshold)
    assert q_result.n_outliers <= 10


def test_q_residuals_with_outliers(spectra_with_outliers):
    """Test Q-residuals with known outliers."""
    pca_result = run_pca_outlier_detection(spectra_with_outliers, n_components=5)
    q_result = compute_q_residuals(spectra_with_outliers, pca_result.pca_model)

    assert q_result.n_outliers > 0

    # Known outliers should have high Q-residuals
    known_outlier_indices = [95, 96, 97, 98, 99]
    detected_outliers = set(q_result.outlier_indices)

    overlap = len(set(known_outlier_indices) & detected_outliers)
    # Should detect at least some known outliers
    assert overlap >= 2


def test_q_residuals_varying_components(normal_spectra):
    """Test Q-residuals with varying number of components."""
    pca_model = PCA(n_components=10).fit(normal_spectra)

    # Test with different reconstruction components
    for n_comp in [1, 3, 5, 10]:
        q_result = compute_q_residuals(normal_spectra, pca_model, n_components=n_comp)
        assert len(q_result.q_residuals) == 100

    # More components = better reconstruction = lower residuals
    q_1comp = compute_q_residuals(normal_spectra, pca_model, n_components=1)
    q_10comp = compute_q_residuals(normal_spectra, pca_model, n_components=10)

    assert np.mean(q_1comp.q_residuals) > np.mean(q_10comp.q_residuals)


# ============================================================================
# MAHALANOBIS DISTANCE TESTS
# ============================================================================

def test_mahalanobis_normal_scores():
    """Test Mahalanobis distance on normal PCA scores."""
    np.random.seed(42)
    # Generate normal scores
    scores = np.random.normal(0, 1, size=(100, 5))

    result = compute_mahalanobis_distance(scores)

    assert isinstance(result, MahalanobisResult)
    assert len(result.distances) == 100
    assert result.median > 0
    assert result.mad >= 0
    assert result.threshold > 0

    # Should find few outliers with 3*MAD threshold
    assert result.n_outliers < 10


def test_mahalanobis_with_outliers():
    """Test Mahalanobis distance with known outliers."""
    np.random.seed(42)
    # Normal scores
    scores = np.random.normal(0, 1, size=(100, 5))

    # Add outliers
    scores[95] = np.ones(5) * 10.0  # Far from center
    scores[96] = np.ones(5) * -10.0  # Far from center

    result = compute_mahalanobis_distance(scores)

    assert result.n_outliers > 0
    # Outliers 95 and 96 should be detected
    assert 95 in result.outlier_indices or 96 in result.outlier_indices


def test_mahalanobis_single_dimension():
    """Test Mahalanobis distance with single dimension (edge case)."""
    np.random.seed(42)
    scores = np.random.normal(0, 1, size=(100, 1))

    result = compute_mahalanobis_distance(scores)

    assert len(result.distances) == 100
    assert result.threshold > 0


def test_mahalanobis_threshold_robustness():
    """Test that MAD-based threshold is robust to outliers."""
    np.random.seed(42)
    # Mostly normal with extreme outliers
    scores = np.random.normal(0, 1, size=(100, 3))
    scores[95:] = 100.0  # 5 extreme outliers

    result = compute_mahalanobis_distance(scores)

    # Threshold should still be reasonable (not inflated by outliers)
    # MAD is robust, so threshold should be < 50
    assert result.threshold < 50


# ============================================================================
# Y-VALUE CONSISTENCY TESTS
# ============================================================================

def test_y_consistency_normal_values(reference_values_normal):
    """Test Y-value consistency on normal distribution."""
    result = check_y_data_consistency(reference_values_normal)

    assert isinstance(result, YConsistencyResult)
    assert not result.is_categorical
    assert result.mean == pytest.approx(10.0, abs=0.5)
    assert result.std == pytest.approx(2.0, abs=0.5)
    assert len(result.z_scores) == 100

    # Should find few Z-score outliers (±3σ rule)
    assert result.n_outliers < 5


def test_y_consistency_with_outliers(reference_values_with_outliers):
    """Test Y-value consistency with known outliers."""
    result = check_y_data_consistency(reference_values_with_outliers)

    assert result.n_outliers > 0
    # Known outliers at indices 95, 96
    assert 95 in result.outlier_indices or 96 in result.outlier_indices


def test_y_consistency_range_bounds():
    """Test Y-value consistency with range bounds."""
    y = np.array([1, 2, 3, 4, 5, 100, -50])

    result = check_y_data_consistency(y, lower_bound=0, upper_bound=10)

    # Should detect 100 and -50 as range outliers
    assert result.n_outliers >= 2
    assert 5 in result.outlier_indices  # 100 is out of range
    assert 6 in result.outlier_indices  # -50 is out of range


def test_y_consistency_categorical():
    """Test Y-value consistency with categorical data."""
    y = np.array(['A'] * 40 + ['B'] * 35 + ['C'] * 25)

    result = check_y_data_consistency(y)

    assert result.is_categorical
    assert result.mean is None
    assert result.std is None
    assert result.n_outliers == 0  # No outliers for categorical
    assert len(result.unique_values) == 3
    assert result.value_counts == [40, 35, 25] or result.value_counts == [25, 35, 40]


def test_y_consistency_identical_values():
    """Test Y-value consistency with all identical values."""
    y = np.ones(100) * 5.0

    result = check_y_data_consistency(y)

    assert result.mean == 5.0
    assert result.std == 0.0
    assert result.n_outliers == 0  # No outliers when all identical


# ============================================================================
# COMPREHENSIVE OUTLIER REPORT TESTS
# ============================================================================

def test_outlier_report_normal_data(normal_spectra, reference_values_normal):
    """Test comprehensive outlier report on normal data."""
    report = generate_outlier_report(
        normal_spectra,
        reference_values_normal,
        n_pca_components=5
    )

    assert isinstance(report, OutlierReport)
    assert isinstance(report.pca, PCAOutlierResult)
    assert isinstance(report.q_residuals, QResidualResult)
    assert isinstance(report.mahalanobis, MahalanobisResult)
    assert isinstance(report.y_consistency, YConsistencyResult)

    assert len(report.combined_flags) == 100
    assert len(report.total_flags_per_sample) == 100

    # Should find few high-confidence outliers in normal data
    assert len(report.high_confidence_indices) < 5


def test_outlier_report_with_outliers(spectra_with_outliers, reference_values_with_outliers):
    """Test comprehensive outlier report with known outliers."""
    report = generate_outlier_report(
        spectra_with_outliers,
        reference_values_with_outliers,
        n_pca_components=5
    )

    # Should detect some outliers
    total_outliers = len(report.high_confidence_indices) + len(report.moderate_confidence_indices)
    assert total_outliers > 0

    # At least one known outlier should be in moderate or high confidence
    known_outliers = set([95, 96, 97, 98, 99])
    detected_moderate_high = set(report.high_confidence_indices) | set(report.moderate_confidence_indices)

    overlap = len(known_outliers & detected_moderate_high)
    assert overlap >= 2


def test_outlier_report_confidence_levels(spectra_with_outliers, reference_values_normal):
    """Test that confidence levels are properly separated."""
    report = generate_outlier_report(
        spectra_with_outliers,
        reference_values_normal,
        n_pca_components=5
    )

    # Check that flags add up correctly
    for i in range(len(report.total_flags_per_sample)):
        flags = report.total_flags_per_sample[i]

        if flags >= 3:
            assert i in report.high_confidence_indices
        elif flags == 2:
            assert i in report.moderate_confidence_indices
        elif flags == 1:
            assert i in report.low_confidence_indices


def test_outlier_report_combined_flags(normal_spectra, reference_values_normal):
    """Test that combined flags use 2+ methods threshold."""
    report = generate_outlier_report(
        normal_spectra,
        reference_values_normal,
        n_pca_components=5
    )

    for i in range(len(report.combined_flags)):
        if report.combined_flags[i]:
            # Combined flag should be True only if 2+ methods flagged it
            assert report.total_flags_per_sample[i] >= 2


# ============================================================================
# EDGE CASES
# ============================================================================

def test_single_sample_edge_case():
    """Test handling of single sample (edge case)."""
    X = np.random.normal(0, 1, size=(1, 100))
    y = np.array([5.0])

    # Should handle gracefully, though results may not be meaningful
    with pytest.raises(Exception):
        # PCA requires at least 2 samples
        run_pca_outlier_detection(X, n_components=1)


def test_two_samples():
    """Test handling of two samples."""
    X = np.random.normal(0, 1, size=(2, 100))
    y = np.array([5.0, 6.0])

    result = run_pca_outlier_detection(X, n_components=1)

    # Should work with 1 component
    assert result.scores.shape == (2, 1)


def test_identical_samples():
    """Test handling of identical samples."""
    X = np.ones((100, 200))  # All identical
    y = np.ones(100) * 5.0

    result = run_pca_outlier_detection(X, n_components=5)

    # All identical -> zero variance -> may have issues
    # Should handle gracefully
    assert result.scores.shape[0] == 100


def test_nan_handling_in_spectra():
    """Test handling of NaN values in spectra."""
    X = np.random.normal(0, 1, size=(100, 200))
    X[50, 100] = np.nan  # Insert a NaN

    # PCA should handle or raise appropriate error
    try:
        result = run_pca_outlier_detection(X, n_components=5)
        # If it doesn't raise, check result is valid
        assert result.scores.shape[0] == 100
    except (ValueError, np.linalg.LinAlgError):
        # Expected if NaN handling isn't built in
        pass


def test_high_dimensional_data():
    """Test with more wavelengths than samples."""
    X = np.random.normal(0, 1, size=(50, 500))  # n_wavelengths > n_samples
    y = np.random.normal(10, 2, size=50)

    result = run_pca_outlier_detection(X, n_components=10)

    # n_components should be clipped to min(10, 49, 500) = 10
    assert result.scores.shape[1] == 10


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_large_dataset_performance():
    """Test performance on large dataset (10000+ samples)."""
    import time

    np.random.seed(42)
    X = np.random.normal(0, 1, size=(10000, 200))
    y = np.random.normal(10, 2, size=10000)

    start = time.time()
    report = generate_outlier_report(X, y, n_pca_components=5)
    elapsed = time.time() - start

    # Should complete in reasonable time (< 10 seconds)
    assert elapsed < 10.0
    assert len(report.combined_flags) == 10000


def test_high_wavelength_performance():
    """Test performance with high wavelength count."""
    import time

    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 2000))  # 2000 wavelengths
    y = np.random.normal(10, 2, size=100)

    start = time.time()
    result = run_pca_outlier_detection(X, n_components=10)
    elapsed = time.time() - start

    # Should complete quickly (< 2 seconds)
    assert elapsed < 2.0
    assert result.scores.shape == (100, 10)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_complete_workflow():
    """Test complete outlier detection workflow."""
    np.random.seed(42)

    # Generate data with known outliers
    X = np.random.normal(0, 1, size=(100, 200))
    X[95:] = 10.0  # 5 outliers

    y = np.random.normal(10, 2, size=100)
    y[96] = 100.0  # Y outlier

    # Run complete analysis
    report = generate_outlier_report(X, y, n_pca_components=5)

    # Check all components are present
    assert report.pca.n_outliers >= 0
    assert report.q_residuals.n_outliers >= 0
    assert report.mahalanobis.n_outliers >= 0
    assert report.y_consistency.n_outliers >= 0

    # Should detect some outliers
    assert np.sum(report.combined_flags) > 0


def test_outlier_detection_consistency():
    """Test that multiple runs produce consistent results."""
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 200))
    y = np.random.normal(10, 2, size=100)

    # Run twice
    report1 = generate_outlier_report(X, y, n_pca_components=5)
    report2 = generate_outlier_report(X, y, n_pca_components=5)

    # Results should be identical
    assert np.array_equal(report1.combined_flags, report2.combined_flags)
    assert np.array_equal(report1.pca.outlier_flags, report2.pca.outlier_flags)
