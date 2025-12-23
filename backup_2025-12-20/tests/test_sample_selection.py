"""
Tests for sample_selection module.

Tests all sample selection algorithms:
- Kennard-Stone
- DUPLEX
- SPXY
- Random selection
- Comparison utilities
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from spectral_predict.sample_selection import (
    kennard_stone,
    duplex,
    spxy,
    random_selection,
    compare_selection_methods
)


class TestKennardStone:
    """Test Kennard-Stone algorithm."""

    def test_basic_selection(self):
        """Test basic KS selection returns correct number of samples."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        n_select = 15

        indices = kennard_stone(X, n_samples=n_select)

        assert len(indices) == n_select
        assert len(np.unique(indices)) == n_select  # All unique
        assert indices.min() >= 0
        assert indices.max() < 100

    def test_deterministic(self):
        """Test that KS is deterministic."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        indices1 = kennard_stone(X, n_samples=10)
        indices2 = kennard_stone(X, n_samples=10)

        assert_array_equal(indices1, indices2)

    def test_two_samples(self):
        """Test selecting minimum (2) samples."""
        X = np.random.randn(20, 5)
        indices = kennard_stone(X, n_samples=2)

        assert len(indices) == 2
        # The two samples should be maximally distant
        dist = np.linalg.norm(X[indices[0]] - X[indices[1]])
        assert dist > 0

    def test_all_samples(self):
        """Test selecting all samples."""
        np.random.seed(42)
        X = np.random.randn(30, 10)
        indices = kennard_stone(X, n_samples=30)

        assert len(indices) == 30
        assert set(indices) == set(range(30))

    def test_diversity(self):
        """Test that KS selects diverse samples."""
        # Create dataset with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(30, 2) + np.array([0, 0])
        cluster2 = np.random.randn(30, 2) + np.array([10, 10])
        cluster3 = np.random.randn(30, 2) + np.array([0, 10])

        X = np.vstack([cluster1, cluster2, cluster3])

        indices = kennard_stone(X, n_samples=15)

        # Selected samples should span multiple clusters
        # Check that selected samples have high variance
        X_selected = X[indices]
        variance = np.var(X_selected, axis=0).sum()

        # Compare to random selection
        random_indices = np.random.choice(90, 15, replace=False)
        X_random = X[random_indices]
        variance_random = np.var(X_random, axis=0).sum()

        # KS should generally have higher variance (more diverse)
        # This is probabilistic but should hold with our seed
        assert variance >= variance_random * 0.8  # Allow some tolerance

    def test_different_metrics(self):
        """Test KS with different distance metrics."""
        X = np.random.randn(50, 10)

        indices_euclidean = kennard_stone(X, n_samples=10, metric='euclidean')
        indices_manhattan = kennard_stone(X, n_samples=10, metric='cityblock')

        assert len(indices_euclidean) == 10
        assert len(indices_manhattan) == 10
        # Different metrics may select different samples
        # Just check they're valid

    def test_error_too_many_samples(self):
        """Test error when requesting more samples than available."""
        X = np.random.randn(20, 5)

        with pytest.raises(ValueError, match="Cannot select 30 samples"):
            kennard_stone(X, n_samples=30)

    def test_error_too_few_samples(self):
        """Test error when requesting < 2 samples."""
        X = np.random.randn(20, 5)

        with pytest.raises(ValueError, match="Must select at least 2 samples"):
            kennard_stone(X, n_samples=1)


class TestDUPLEX:
    """Test DUPLEX algorithm."""

    def test_basic_split(self):
        """Test basic DUPLEX split."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cal_idx, val_idx = duplex(X, y, cal_ratio=0.75)

        assert len(cal_idx) == 75
        assert len(val_idx) == 25
        assert len(set(cal_idx) & set(val_idx)) == 0  # No overlap
        assert set(cal_idx) | set(val_idx) == set(range(100))  # All samples used

    def test_different_ratios(self):
        """Test DUPLEX with different calibration ratios."""
        X = np.random.randn(100, 10)

        cal_idx1, val_idx1 = duplex(X, cal_ratio=0.6)
        assert len(cal_idx1) == 60
        assert len(val_idx1) == 40

        cal_idx2, val_idx2 = duplex(X, cal_ratio=0.8)
        assert len(cal_idx2) == 80
        assert len(val_idx2) == 20

    def test_explicit_n_cal(self):
        """Test DUPLEX with explicit n_cal parameter."""
        X = np.random.randn(100, 10)

        cal_idx, val_idx = duplex(X, n_cal=70)

        assert len(cal_idx) == 70
        assert len(val_idx) == 30

    def test_deterministic(self):
        """Test that DUPLEX is deterministic."""
        X = np.random.randn(50, 10)

        cal_idx1, val_idx1 = duplex(X, cal_ratio=0.7)
        cal_idx2, val_idx2 = duplex(X, cal_ratio=0.7)

        assert_array_equal(sorted(cal_idx1), sorted(cal_idx2))
        assert_array_equal(sorted(val_idx1), sorted(val_idx2))

    def test_both_sets_representative(self):
        """Test that both cal and val sets span the feature space."""
        # Create dataset with wide range
        np.random.seed(42)
        X = np.random.uniform(-10, 10, size=(100, 5))

        cal_idx, val_idx = duplex(X, cal_ratio=0.75)

        X_cal = X[cal_idx]
        X_val = X[val_idx]

        # Both sets should span similar ranges
        for i in range(X.shape[1]):
            range_cal = X_cal[:, i].max() - X_cal[:, i].min()
            range_val = X_val[:, i].max() - X_val[:, i].min()
            range_total = X[:, i].max() - X[:, i].min()

            # Both should cover at least 50% of total range
            assert range_cal > 0.5 * range_total
            assert range_val > 0.4 * range_total  # Val set is smaller, allow more tolerance

    def test_error_invalid_split(self):
        """Test error on invalid split ratios."""
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Invalid split"):
            duplex(X, cal_ratio=0.99)  # Would give 0 validation samples


class TestSPXY:
    """Test SPXY algorithm."""

    def test_basic_selection(self):
        """Test basic SPXY selection."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        indices = spxy(X, y, n_samples=15)

        assert len(indices) == 15
        assert len(np.unique(indices)) == 15
        assert indices.min() >= 0
        assert indices.max() < 100

    def test_spans_y_range(self):
        """Test that SPXY selects samples spanning Y range."""
        np.random.seed(42)
        # Create data with strong X-Y relationship
        X = np.random.randn(100, 10)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

        indices = spxy(X, y, n_samples=20)

        y_selected = y[indices]

        # Selected samples should span most of Y range
        y_range_selected = y_selected.max() - y_selected.min()
        y_range_total = y.max() - y.min()

        assert y_range_selected > 0.8 * y_range_total

    def test_better_than_random_for_y_coverage(self):
        """Test that SPXY covers Y space better than random selection."""
        np.random.seed(42)
        X = np.random.randn(200, 15)
        y = np.random.randn(200)

        # SPXY selection
        spxy_indices = spxy(X, y, n_samples=30)
        y_spxy = y[spxy_indices]
        y_range_spxy = y_spxy.max() - y_spxy.min()

        # Random selection (average over multiple trials)
        y_ranges_random = []
        for seed in range(10):
            np.random.seed(seed)
            random_indices = np.random.choice(200, 30, replace=False)
            y_random = y[random_indices]
            y_ranges_random.append(y_random.max() - y_random.min())

        avg_random_range = np.mean(y_ranges_random)

        # SPXY should generally cover more Y range than random
        assert y_range_spxy >= avg_random_range * 0.9

    def test_multivariate_y(self):
        """Test SPXY with multi-dimensional Y."""
        np.random.seed(42)
        X = np.random.randn(80, 10)
        y = np.random.randn(80, 3)  # 3 target variables

        indices = spxy(X, y, n_samples=20)

        assert len(indices) == 20
        assert len(np.unique(indices)) == 20

    def test_deterministic(self):
        """Test that SPXY is deterministic."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        indices1 = spxy(X, y, n_samples=15)
        indices2 = spxy(X, y, n_samples=15)

        assert_array_equal(indices1, indices2)

    def test_error_dimension_mismatch(self):
        """Test error when X and y have different lengths."""
        X = np.random.randn(100, 10)
        y = np.random.randn(80)  # Wrong length

        with pytest.raises(ValueError, match="same number of samples"):
            spxy(X, y, n_samples=20)

    def test_error_too_many_samples(self):
        """Test error when requesting too many samples."""
        X = np.random.randn(30, 5)
        y = np.random.randn(30)

        with pytest.raises(ValueError, match="Cannot select 50 samples"):
            spxy(X, y, n_samples=50)


class TestRandomSelection:
    """Test random selection baseline."""

    def test_basic_random(self):
        """Test basic random selection."""
        indices = random_selection(100, 20, random_state=42)

        assert len(indices) == 20
        assert len(np.unique(indices)) == 20  # No duplicates
        assert indices.min() >= 0
        assert indices.max() < 100

    def test_reproducible(self):
        """Test that random selection is reproducible with seed."""
        indices1 = random_selection(100, 20, random_state=42)
        indices2 = random_selection(100, 20, random_state=42)

        assert_array_equal(indices1, indices2)

    def test_different_seeds(self):
        """Test that different seeds give different results."""
        indices1 = random_selection(100, 20, random_state=42)
        indices2 = random_selection(100, 20, random_state=99)

        # Should be different (extremely unlikely to be same)
        assert not np.array_equal(indices1, indices2)

    def test_error_too_many(self):
        """Test error when requesting too many samples."""
        with pytest.raises(ValueError, match="Cannot select 150 samples"):
            random_selection(100, 150)


class TestCompareSelectionMethods:
    """Test method comparison functionality."""

    def test_basic_comparison(self):
        """Test basic method comparison."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        results = compare_selection_methods(X, y, n_samples=15)

        # Should have results for all methods
        assert 'kennard-stone' in results
        assert 'random' in results
        assert 'duplex' in results
        assert 'spxy' in results

        # Each should have required metrics
        for method, metrics in results.items():
            assert 'indices' in metrics
            assert 'mean_distance' in metrics
            assert 'min_distance' in metrics
            assert 'coverage' in metrics
            assert 'n_samples' in metrics

            assert len(metrics['indices']) == 15
            assert metrics['mean_distance'] > 0
            assert metrics['min_distance'] >= 0
            assert 0 <= metrics['coverage'] <= 1

    def test_comparison_without_y(self):
        """Test comparison when y is not provided."""
        np.random.seed(42)
        X = np.random.randn(100, 20)

        results = compare_selection_methods(X, y=None, n_samples=15)

        # Should only have methods that don't need y
        assert 'kennard-stone' in results
        assert 'random' in results
        assert 'duplex' not in results
        assert 'spxy' not in results

    def test_custom_methods_list(self):
        """Test comparison with custom methods list."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        results = compare_selection_methods(
            X, y, n_samples=15,
            methods=['kennard-stone', 'random']
        )

        assert 'kennard-stone' in results
        assert 'random' in results
        assert 'duplex' not in results
        assert 'spxy' not in results

    def test_ks_better_coverage_than_random(self):
        """Test that KS generally has better coverage than random."""
        np.random.seed(42)
        X = np.random.randn(200, 30)

        results = compare_selection_methods(X, n_samples=40, methods=['kennard-stone', 'random'])

        ks_coverage = results['kennard-stone']['coverage']
        random_coverage = results['random']['coverage']

        # KS should have equal or better coverage
        assert ks_coverage >= random_coverage * 0.95  # Allow small tolerance


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_ks_for_transfer_samples(self):
        """Test using KS to select transfer samples for calibration transfer."""
        np.random.seed(42)

        # Simulate master and slave spectra
        n_samples = 200
        n_wavelengths = 150

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = 0.95 * X_master + 0.05  # Slight offset

        # Select 12 transfer samples (optimal for TSR according to literature)
        transfer_indices = kennard_stone(X_master, n_samples=12)

        # Extract transfer samples
        X_master_transfer = X_master[transfer_indices]
        X_slave_transfer = X_slave[transfer_indices]

        assert X_master_transfer.shape == (12, n_wavelengths)
        assert X_slave_transfer.shape == (12, n_wavelengths)

        # Verify they're diverse
        from scipy.spatial.distance import pdist
        distances = pdist(X_master_transfer)
        assert np.min(distances) > 0  # No duplicates
        assert np.std(distances) > 0  # Varied distances

    def test_spxy_for_calibration_split(self):
        """Test using SPXY for calibration/validation split."""
        np.random.seed(42)

        # Simulate spectral dataset with reference values
        X = np.random.randn(150, 100)
        y = np.random.randn(150)

        # Select 100 samples for calibration (diverse in both X and Y)
        cal_indices = spxy(X, y, n_samples=100)
        val_indices = np.array([i for i in range(150) if i not in cal_indices])

        X_cal, y_cal = X[cal_indices], y[cal_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Both sets should span Y range reasonably
        y_range_total = y.max() - y.min()
        y_range_cal = y_cal.max() - y_cal.min()
        y_range_val = y_val.max() - y_val.min()

        assert y_range_cal > 0.7 * y_range_total
        assert y_range_val > 0.5 * y_range_total  # Smaller set, lower requirement

    def test_workflow_compare_then_select(self):
        """Test workflow: compare methods, then use best one."""
        np.random.seed(42)

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Compare methods
        results = compare_selection_methods(X, y, n_samples=20)

        # Find method with best coverage
        best_method = max(results, key=lambda k: results[k]['coverage'])

        # Use best method's indices
        best_indices = results[best_method]['indices']

        assert len(best_indices) == 20
        assert best_method in ['kennard-stone', 'duplex', 'spxy', 'random']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(5, 3)

        indices = kennard_stone(X, n_samples=3)
        assert len(indices) == 3

    def test_high_dimensional(self):
        """Test with high-dimensional data."""
        X = np.random.randn(50, 500)  # More features than samples

        indices = kennard_stone(X, n_samples=10)
        assert len(indices) == 10

    def test_zero_variance_features(self):
        """Test with features that have zero variance."""
        X = np.random.randn(50, 10)
        X[:, 3] = 5.0  # Constant feature

        indices = kennard_stone(X, n_samples=15)
        assert len(indices) == 15

    def test_identical_samples(self):
        """Test with some identical samples."""
        X = np.random.randn(50, 10)
        X[10] = X[5]  # Make sample 10 identical to sample 5

        indices = kennard_stone(X, n_samples=15)
        # Should still work, might select one of the identical samples
        assert len(indices) == 15


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
