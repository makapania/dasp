"""
Comprehensive tests for outlier detection module.

Tests cover:
- PCA-based outlier detection (Hotelling T²)
- Q-residuals (SPE) for reconstruction error
- Mahalanobis distance in PC space
- Y-data consistency checks (z-scores, range checks, categorical handling)
- Comprehensive outlier reporting

Each test validates mathematical properties, edge cases, and correct detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from spectral_predict.outlier_detection import (
    check_y_data_consistency,
    compute_mahalanobis_distance,
    compute_q_residuals,
    generate_outlier_report,
    run_pca_outlier_detection,
)


# =============================================================================
# PCA Outlier Detection Tests
# =============================================================================


@pytest.mark.numerical
class TestPCAOutlierDetection:
    """Test PCA-based outlier detection using Hotelling T² statistic."""

    def test_detects_leverage_outliers(self, outlier_data):
        """Verify that known outliers are detected by PCA T² statistic."""
        X, y, true_outliers = outlier_data

        results = run_pca_outlier_detection(X, y, n_components=5)

        # Check that we detect at least 3 out of 5 known outliers
        detected = results["outlier_indices"]
        overlap = len(set(detected) & set(true_outliers))

        assert overlap >= 3, f"Expected to detect at least 3/5 outliers, got {overlap}"
        assert results["n_outliers"] == len(detected)

    def test_hotelling_t2_threshold(self, synthetic_spectra_medium):
        """Verify that ~5% of samples exceed T² threshold at alpha=0.05."""
        X, y = synthetic_spectra_medium

        results = run_pca_outlier_detection(X, y, n_components=5)

        # With clean data, expect ~5% flagged as outliers (false positive rate)
        outlier_fraction = results["n_outliers"] / len(X)

        # Allow some tolerance: 0-10% flagged is reasonable
        assert 0 <= outlier_fraction <= 0.10, (
            f"Expected ~5% outliers (α=0.05), got {outlier_fraction:.1%}"
        )

    def test_handles_singular_covariance(self):
        """Verify that regularization works when covariance is singular."""
        # Create data with perfect collinearity (singular covariance)
        X = pd.DataFrame({
            "w1": [1, 2, 3, 4, 5],
            "w2": [2, 4, 6, 8, 10],  # Perfectly correlated with w1
            "w3": [1, 1, 1, 1, 1],  # Zero variance
        })
        y = pd.Series([1, 2, 3, 4, 5])

        # Should not crash due to singular matrix
        results = run_pca_outlier_detection(X, y, n_components=2)

        assert results["hotelling_t2"] is not None
        assert len(results["hotelling_t2"]) == len(X)
        assert not np.any(np.isnan(results["hotelling_t2"]))

    def test_clips_n_components(self, synthetic_spectra_small):
        """Verify that n_components is auto-clipped to valid range."""
        X, y = synthetic_spectra_small
        n_samples = len(X)

        # Request more components than possible
        results = run_pca_outlier_detection(X, y, n_components=n_samples + 10)

        # Should clip to n_samples - 1
        actual_components = results["scores"].shape[1]
        assert actual_components == min(n_samples - 1, X.shape[1])

    def test_returns_required_fields(self, synthetic_spectra_small):
        """Verify that all required fields are returned."""
        X, y = synthetic_spectra_small

        results = run_pca_outlier_detection(X, y, n_components=5)

        # Check all required fields
        required_fields = [
            "pca_model",
            "scores",
            "loadings",
            "variance_explained",
            "hotelling_t2",
            "t2_threshold",
            "outlier_flags",
            "n_outliers",
            "outlier_indices",
        ]

        for field in required_fields:
            assert field in results, f"Missing required field: {field}"

        # Verify shapes
        assert results["scores"].shape[0] == len(X)
        assert results["scores"].shape[1] == 5
        assert results["loadings"].shape[1] == 5
        assert len(results["hotelling_t2"]) == len(X)
        assert len(results["outlier_flags"]) == len(X)

    def test_single_component_case(self, synthetic_spectra_small):
        """Verify that single component case is handled correctly."""
        X, y = synthetic_spectra_small

        results = run_pca_outlier_detection(X, y, n_components=1)

        assert results["scores"].shape[1] == 1
        assert len(results["hotelling_t2"]) == len(X)
        assert not np.any(np.isnan(results["hotelling_t2"]))

    def test_threshold_follows_f_distribution(self, synthetic_spectra_small):
        """Verify that T² threshold matches F-distribution formula."""
        X, y = synthetic_spectra_small
        n_components = 5

        results = run_pca_outlier_detection(X, y, n_components=n_components)

        n_samples = len(X)
        alpha = 0.05

        # Compute expected threshold
        expected_threshold = (
            n_components
            * (n_samples - 1)
            / (n_samples - n_components)
            * stats.f.ppf(1 - alpha, n_components, n_samples - n_components)
        )

        # Allow small numerical tolerance
        assert np.isclose(
            results["t2_threshold"], expected_threshold, rtol=1e-6
        ), f"Expected {expected_threshold:.6f}, got {results['t2_threshold']:.6f}"


# =============================================================================
# Q-Residuals Tests
# =============================================================================


@pytest.mark.numerical
class TestQResiduals:
    """Test Q-residuals (SPE) for PCA reconstruction error."""

    def test_high_q_for_novel_samples(self, synthetic_spectra_small):
        """Verify that novel samples have high Q-residuals."""
        X, y = synthetic_spectra_small

        # Fit PCA on clean data
        pca_results = run_pca_outlier_detection(X, y, n_components=5)

        # Create novel sample (very different from training data)
        X_novel = X.copy()
        X_novel.iloc[0, :] = X.mean() + 5 * X.std()  # Extreme sample

        q_results = compute_q_residuals(X_novel, pca_results["pca_model"])

        # First sample should have higher Q-residual than typical samples
        assert q_results["q_residuals"][0] > np.median(q_results["q_residuals"])

    def test_q_threshold_is_95th_percentile(self, synthetic_spectra_medium):
        """Verify that Q threshold is at the 95th percentile."""
        X, y = synthetic_spectra_medium

        pca_results = run_pca_outlier_detection(X, y, n_components=5)
        q_results = compute_q_residuals(X, pca_results["pca_model"])

        # Threshold should be 95th percentile
        expected_threshold = np.percentile(q_results["q_residuals"], 95)

        assert np.isclose(q_results["q_threshold"], expected_threshold, rtol=1e-6)

        # Approximately 5% should exceed threshold
        outlier_fraction = q_results["n_outliers"] / len(X)
        assert 0.03 <= outlier_fraction <= 0.07  # Allow 3-7% for statistical variation

    def test_shape_preserved(self, synthetic_spectra_small):
        """Verify that output shapes match input."""
        X, y = synthetic_spectra_small

        pca_results = run_pca_outlier_detection(X, y, n_components=5)
        q_results = compute_q_residuals(X, pca_results["pca_model"])

        assert len(q_results["q_residuals"]) == len(X)
        assert len(q_results["outlier_flags"]) == len(X)
        assert q_results["outlier_flags"].dtype == bool

    def test_fewer_components_higher_q(self, synthetic_spectra_small):
        """Verify that fewer components lead to higher Q-residuals."""
        X, y = synthetic_spectra_small

        pca_results = run_pca_outlier_detection(X, y, n_components=10)

        # Compute Q with different numbers of components
        q_results_2 = compute_q_residuals(X, pca_results["pca_model"], n_components=2)
        q_results_5 = compute_q_residuals(X, pca_results["pca_model"], n_components=5)

        # Fewer components = worse reconstruction = higher Q
        assert (
            q_results_2["q_residuals"].mean() > q_results_5["q_residuals"].mean()
        ), "Fewer components should have higher Q-residuals"


# =============================================================================
# Mahalanobis Distance Tests
# =============================================================================


@pytest.mark.numerical
class TestMahalanobisDistance:
    """Test Mahalanobis distance computation in PC space."""

    def test_identifies_multivariate_outliers(self, outlier_data):
        """Verify that multivariate outliers are identified."""
        X, y, true_outliers = outlier_data

        pca_results = run_pca_outlier_detection(X, y, n_components=5)
        maha_results = compute_mahalanobis_distance(pca_results["scores"])

        # Should detect some of the known outliers
        detected = maha_results["outlier_indices"]
        overlap = len(set(detected) & set(true_outliers))

        assert overlap >= 2, f"Expected to detect at least 2/5 outliers, got {overlap}"

    def test_handles_1d_scores(self):
        """Verify that 1D score arrays are handled correctly."""
        # Create 1D scores (single PC)
        scores_1d = np.array([1, 2, 3, 4, 5, 100])  # Last is outlier

        results = compute_mahalanobis_distance(scores_1d)

        assert len(results["distances"]) == len(scores_1d)
        assert results["outlier_flags"][-1] == True  # Should flag the outlier

    def test_mad_robustness(self):
        """Verify that MAD-based threshold is robust to outliers."""
        # Create scores with one extreme outlier
        scores = np.random.randn(100, 2)
        scores[0] = [100, 100]  # Extreme outlier

        results = compute_mahalanobis_distance(scores)

        # MAD should be robust to the outlier
        # Median should be close to 0
        assert abs(results["median"]) < 2, "Median should be robust to outlier"

        # MAD should be reasonable (not inflated by outlier)
        assert results["mad"] < 10, "MAD should be robust to outlier"

        # Outlier should be detected
        assert results["outlier_flags"][0] == True

    def test_center_samples_have_low_distance(self, synthetic_spectra_small):
        """Verify that central samples have low Mahalanobis distance."""
        X, y = synthetic_spectra_small

        pca_results = run_pca_outlier_detection(X, y, n_components=5)
        maha_results = compute_mahalanobis_distance(pca_results["scores"])

        # Most samples should be below threshold
        below_threshold = maha_results["distances"] < maha_results["threshold"]
        assert np.mean(below_threshold) > 0.9  # At least 90% below threshold


# =============================================================================
# Y-Data Consistency Tests
# =============================================================================


@pytest.mark.unit
class TestYDataConsistency:
    """Test Y-value consistency checks for outlier detection."""

    def test_detects_z_score_outliers(self):
        """Verify that |z| > 3 values are flagged as outliers."""
        # Create data with known z-score outliers
        # Use normal distribution with known extreme value
        np.random.seed(42)
        y_normal = np.random.randn(100)  # Mean=0, std~1
        y = np.concatenate([y_normal, [10.0]])  # Add extreme outlier (z=10)

        results = check_y_data_consistency(y)

        # Last sample should be flagged (z-score >> 3)
        assert results["z_outliers"][-1] == True
        assert results["n_outliers"] >= 1

    def test_range_check_with_bounds(self):
        """Verify that range checks flag values outside bounds."""
        y = np.array([0, 5, 10, 15, 20, 25, 30])

        results = check_y_data_consistency(y, lower_bound=5, upper_bound=25)

        # First sample (0) and last sample (30) should be flagged
        assert results["range_outliers"][0] == True  # Below lower bound
        assert results["range_outliers"][-1] == True  # Above upper bound
        assert results["n_outliers"] == 2

    def test_handles_categorical_data(self):
        """Verify that categorical data is handled correctly."""
        y = np.array(["Low", "Medium", "High", "Low", "Medium", "High"])

        results = check_y_data_consistency(y)

        # Should recognize as categorical
        assert results["is_categorical"] == True
        assert results["n_outliers"] == 0  # No outliers for categorical
        assert len(results["unique_values"]) == 3
        assert sum(results["value_counts"]) == len(y)

        # Numeric stats should be None
        assert results["mean"] is None
        assert results["std"] is None

    def test_zero_std_handling(self):
        """Verify that zero standard deviation is handled correctly."""
        # All identical values
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        results = check_y_data_consistency(y)

        # Z-scores should all be zero
        assert np.all(results["z_scores"] == 0)
        assert results["n_outliers"] == 0
        assert results["std"] == 0.0

    def test_combined_outliers(self):
        """Verify that z-score and range outliers are combined."""
        y = np.array([0, 5, 10, 15, 20, 100])  # 0 is range outlier, 100 is z-score

        results = check_y_data_consistency(y, lower_bound=3, upper_bound=50)

        # Both should be flagged in all_outliers
        assert results["all_outliers"][0] == True  # Range outlier
        assert results["all_outliers"][-1] == True  # Z-score outlier
        assert results["n_outliers"] == 2

    def test_categorical_class_distribution(self):
        """Verify that categorical class distribution is computed correctly."""
        y = np.array(["A", "A", "A", "B", "B", "C"])

        results = check_y_data_consistency(y)

        # Check class distribution
        assert results["is_categorical"] == True
        assert set(results["unique_values"]) == {"A", "B", "C"}

        # Check frequencies sum to 1
        assert np.isclose(sum(results["frequencies"]), 1.0)

    def test_categorical_with_nan_values(self):
        """Verify that categorical data with NaN values does not crash.

        Regression test for: np.unique() on mixed str/NaN raises
        TypeError: '<' not supported between instances of 'str' and 'float'.
        NaN should be silently excluded from class distribution.
        """
        y = pd.Series(["Upland_Tundra", "Valley", "Upland_Tundra", np.nan, "Valley", np.nan])

        results = check_y_data_consistency(y)

        assert results["is_categorical"] is True
        assert results["n_outliers"] == 0
        assert set(results["unique_values"]) == {"Upland_Tundra", "Valley"}
        assert sum(results["value_counts"]) == 4  # 2 + 2, NaN excluded

    def test_categorical_all_nan(self):
        """Verify that all-NaN series (float64 by pandas inference) doesn't crash."""
        y = pd.Series([np.nan, np.nan, np.nan])

        results = check_y_data_consistency(y)

        # pandas infers float64 for all-NaN, so numeric path is taken
        assert results["n_outliers"] == 0


# =============================================================================
# Comprehensive Outlier Report Tests
# =============================================================================


@pytest.mark.integration
class TestGenerateOutlierReport:
    """Test comprehensive outlier detection report combining all methods."""

    def test_combines_all_methods(self, outlier_data):
        """Verify that all detection methods are included in report."""
        X, y, true_outliers = outlier_data

        report = generate_outlier_report(X, y, n_pca_components=5)

        # Check all methods are present
        assert "pca" in report
        assert "q_residuals" in report
        assert "mahalanobis" in report
        assert "y_consistency" in report
        assert "combined_flags" in report
        assert "outlier_summary" in report

    def test_confidence_levels_correct(self, outlier_data):
        """Verify that confidence levels are assigned correctly."""
        X, y, true_outliers = outlier_data

        report = generate_outlier_report(X, y, n_pca_components=5)

        summary = report["outlier_summary"]

        # Verify Total_Flags column
        assert "Total_Flags" in summary.columns

        # High confidence: 3+ flags
        high_conf = report["high_confidence_outliers"]
        if len(high_conf) > 0:
            assert np.all(high_conf["Total_Flags"] >= 3)

        # Moderate confidence: exactly 2 flags
        mod_conf = report["moderate_confidence_outliers"]
        if len(mod_conf) > 0:
            assert np.all(mod_conf["Total_Flags"] == 2)

        # Low confidence: exactly 1 flag
        low_conf = report["low_confidence_outliers"]
        if len(low_conf) > 0:
            assert np.all(low_conf["Total_Flags"] == 1)

    def test_summary_dataframe_structure(self, synthetic_spectra_small):
        """Verify that summary DataFrame has correct structure."""
        X, y = synthetic_spectra_small

        report = generate_outlier_report(X, y, n_pca_components=5)

        summary = report["outlier_summary"]

        # Check required columns
        required_columns = [
            "Sample_Index",
            "Y_Value",
            "Hotelling_T2",
            "T2_Outlier",
            "Q_Residual",
            "Q_Outlier",
            "Mahalanobis_Distance",
            "Maha_Outlier",
            "Y_ZScore",
            "Y_Outlier",
            "Total_Flags",
        ]

        for col in required_columns:
            assert col in summary.columns, f"Missing column: {col}"

        # Check length matches input
        assert len(summary) == len(X)

    def test_combined_flags_threshold(self, synthetic_spectra_medium):
        """Verify that combined flags use 2+ methods as threshold."""
        X, y = synthetic_spectra_medium

        report = generate_outlier_report(X, y, n_pca_components=5)

        summary = report["outlier_summary"]
        combined = report["combined_flags"]

        # Combined flags should be True when Total_Flags >= 2
        expected_combined = summary["Total_Flags"] >= 2

        assert np.all(combined == expected_combined.values)

    def test_handles_y_bounds(self, synthetic_spectra_small):
        """Verify that Y-value bounds are passed to consistency check."""
        X, y = synthetic_spectra_small

        # Set bounds that will flag some values
        y_min = float(y.min())
        y_max = float(y.max())
        y_range = y_max - y_min

        # Narrow bounds to flag outliers
        lower_bound = y_min + 0.1 * y_range
        upper_bound = y_max - 0.1 * y_range

        report = generate_outlier_report(
            X, y, n_pca_components=5, y_lower_bound=lower_bound, y_upper_bound=upper_bound
        )

        y_consistency = report["y_consistency"]

        # Should have some range outliers
        assert np.any(y_consistency["range_outliers"])

    def test_detects_known_outliers(self, outlier_data):
        """Verify that known outliers appear in high/moderate confidence."""
        X, y, true_outliers = outlier_data

        report = generate_outlier_report(X, y, n_pca_components=5)

        # Get all flagged samples (high + moderate confidence)
        high_conf_indices = report["high_confidence_outliers"]["Sample_Index"].values
        mod_conf_indices = report["moderate_confidence_outliers"]["Sample_Index"].values
        all_flagged = set(high_conf_indices) | set(mod_conf_indices)

        # Should detect at least 3 of the 5 known outliers
        overlap = len(all_flagged & set(true_outliers))
        assert overlap >= 3, f"Expected to detect at least 3/5 outliers, got {overlap}"


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and robustness of outlier detection."""

    def test_single_sample(self):
        """Verify that single sample is handled gracefully."""
        X = pd.DataFrame([[1, 2, 3, 4, 5]])
        y = pd.Series([1])

        # Should not crash
        results = run_pca_outlier_detection(X, y, n_components=1)

        assert len(results["hotelling_t2"]) == 1

    def test_all_outliers_scenario(self):
        """Verify behavior when all samples are outliers."""
        # Create highly variable data (all could be considered outliers)
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(20, 10) * 100)  # High variance
        y = pd.Series(np.random.randn(20) * 100)

        report = generate_outlier_report(X, y, n_pca_components=3)

        # Should still produce valid report
        assert "outlier_summary" in report
        assert len(report["outlier_summary"]) == len(X)

    def test_zero_variance_features(self):
        """Verify handling of zero-variance features."""
        X = pd.DataFrame({
            "w1": [1, 2, 3, 4, 5],
            "w2": [0, 0, 0, 0, 0],  # Zero variance
            "w3": [5, 4, 3, 2, 1],
        })
        y = pd.Series([1, 2, 3, 4, 5])

        # Should handle gracefully
        results = run_pca_outlier_detection(X, y, n_components=2)

        assert results is not None
        assert len(results["hotelling_t2"]) == len(X)

    def test_perfect_reconstruction(self):
        """Verify Q-residuals when PCA uses all components."""
        X = pd.DataFrame(np.random.randn(20, 5))
        y = pd.Series(np.random.randn(20))

        pca_results = run_pca_outlier_detection(X, y, n_components=5)

        # Use all 5 components for reconstruction
        q_results = compute_q_residuals(X, pca_results["pca_model"], n_components=5)

        # Q-residuals should be very small (near-perfect reconstruction)
        assert np.all(q_results["q_residuals"] < 1e-10)
