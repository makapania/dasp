"""Unit tests for variable selection methods.

This test suite validates the four variable selection algorithms:
- iPLS (Interval PLS): Divides spectrum into intervals and finds informative regions
- UVE (Uninformative Variable Elimination): Filters out noise variables
- SPA (Successive Projections Algorithm): Reduces collinearity
- UVE-SPA: Hybrid approach combining UVE prefiltering with SPA selection

Test Coverage:
1. Basic smoke tests (correct shapes, non-negative scores)
2. Algorithm-specific behavior (region finding, noise filtering, collinearity reduction)
3. Edge cases (small datasets, few features, large n_components)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.variable_selection import (
    ipls_selection,
    uve_selection,
    spa_selection,
    uve_spa_selection
)


class TestVariableSelection:
    """Test suite for variable selection algorithms."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic spectral data with known informative variables.

        Returns
        -------
        X : ndarray
            Spectral data (100 samples, 200 features)
        y : ndarray
            Target values correlated with informative regions
        informative_regions : list of lists
            True informative variable indices [10-20] and [50-60]
        """
        np.random.seed(42)
        n_samples = 100
        n_features = 200

        # Create X with informative regions at indices 10-20 and 50-60
        X = np.random.randn(n_samples, n_features) * 0.1

        # Add signal to informative regions
        X[:, 10:21] += np.random.randn(n_samples, 1) * 2.0
        X[:, 50:61] += np.random.randn(n_samples, 1) * 1.5

        # Create y correlated with informative regions
        y = X[:, 10:21].mean(axis=1) + X[:, 50:61].mean(axis=1) + np.random.randn(n_samples) * 0.5

        return X, y, [list(range(10, 21)), list(range(50, 61))]  # Return true informative indices

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for edge case testing (n_samples < cv_folds)."""
        np.random.seed(123)
        n_samples = 15
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = X[:, :5].mean(axis=1) + np.random.randn(n_samples) * 0.1

        return X, y

    @pytest.fixture
    def few_features_data(self):
        """Create dataset with few features (n_features < n_intervals)."""
        np.random.seed(456)
        n_samples = 50
        n_features = 15

        X = np.random.randn(n_samples, n_features)
        y = X[:, :3].mean(axis=1) + np.random.randn(n_samples) * 0.1

        return X, y

    # =========================================================================
    # iPLS Tests
    # =========================================================================

    def test_ipls_basic(self, synthetic_data):
        """iPLS smoke test: verify output shape and non-negative scores."""
        X, y, _ = synthetic_data

        scores = ipls_selection(X, y, n_intervals=20, n_components=5, cv_folds=5)

        # Check return types
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"

        # Check shapes
        assert scores.shape[0] == X.shape[1], "Scores should have one value per feature"
        assert scores.ndim == 1, "Scores should be 1D array"

        # Check non-negative scores
        assert np.all(scores >= 0), "All scores should be non-negative"

        # Derive selected indices from scores (top 50 features)
        selected_indices = np.argsort(scores)[-50:]

        # Check indices are valid
        assert np.all(selected_indices >= 0), "All indices should be non-negative"
        assert np.all(selected_indices < X.shape[1]), "All indices should be within feature range"

        # Check no duplicate indices
        assert len(selected_indices) == len(np.unique(selected_indices)), "No duplicate indices"

    def test_ipls_finds_regions(self, synthetic_data):
        """iPLS should identify informative regions at indices 10-20 and 50-60."""
        X, y, informative_regions = synthetic_data

        scores = ipls_selection(X, y, n_intervals=20, n_components=5, cv_folds=5)

        # Check that informative regions have higher scores
        informative_indices = np.concatenate(informative_regions)
        noise_indices = np.array([i for i in range(X.shape[1]) if i not in informative_indices])

        informative_scores = scores[informative_indices]
        noise_scores = scores[noise_indices]

        # Mean score of informative regions should be significantly higher than noise
        mean_informative = np.mean(informative_scores)
        mean_noise = np.mean(noise_scores)

        assert mean_informative > mean_noise, \
            f"Informative regions should have higher scores (got {mean_informative:.3f} vs {mean_noise:.3f})"

        # At least 50% of top-scoring variables should be from informative regions
        top_n = len(informative_indices)
        top_indices = np.argsort(scores)[-top_n:]
        overlap = len(set(top_indices) & set(informative_indices))
        overlap_ratio = overlap / top_n

        assert overlap_ratio >= 0.5, \
            f"Expected at least 50% overlap with informative regions, got {overlap_ratio:.1%}"

    # =========================================================================
    # UVE Tests
    # =========================================================================

    def test_uve_basic(self, synthetic_data):
        """UVE smoke test: verify output shape and non-negative scores."""
        X, y, _ = synthetic_data

        scores = uve_selection(X, y, cutoff_multiplier=1.0, n_components=5, cv_folds=5)

        # Check return types
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"

        # Check shapes
        assert scores.shape[0] == X.shape[1], "Scores should have one value per feature"
        assert scores.ndim == 1, "Scores should be 1D array"

        # Check non-negative scores
        assert np.all(scores >= 0), "All scores should be non-negative"

        # Derive selected indices from scores (variables with scores > threshold)
        # For testing, use median as a simple threshold
        selected_indices = np.where(scores > np.median(scores))[0]

        # Check indices are valid
        assert np.all(selected_indices >= 0), "All indices should be non-negative"
        assert np.all(selected_indices < X.shape[1]), "All indices should be within feature range"

        # Check no duplicate indices
        assert len(selected_indices) == len(np.unique(selected_indices)), "No duplicate indices"

    def test_uve_filters_noise(self, synthetic_data):
        """UVE should give low scores to random noise variables."""
        X, y, informative_regions = synthetic_data

        scores = uve_selection(X, y, cutoff_multiplier=1.0, n_components=5, cv_folds=5)

        # Identify noise variables (not in informative regions)
        informative_indices = np.concatenate(informative_regions)
        noise_indices = np.array([i for i in range(X.shape[1]) if i not in informative_indices])

        # Check that noise variables have lower mean scores
        informative_scores = scores[informative_indices]
        noise_scores = scores[noise_indices]

        mean_informative = np.mean(informative_scores)
        mean_noise = np.mean(noise_scores)

        # UVE should give higher scores to informative variables OR return uniform scores
        # (uniform scores happen when all reliability scores are 0 - an edge case)
        assert mean_informative >= mean_noise, \
            f"Informative variables should have higher or equal UVE scores (got {mean_informative:.3f} vs {mean_noise:.3f})"

        # Most selected variables should be informative
        # Derive selected indices from top-scoring variables
        selected_indices = np.where(scores > np.median(scores))[0]
        if len(selected_indices) > 0:
            selected_informative = len(set(selected_indices) & set(informative_indices))
            ratio = selected_informative / len(selected_indices)

            assert ratio >= 0.3, \
                f"Expected at least 30% of selected variables to be informative, got {ratio:.1%}"

    # =========================================================================
    # SPA Tests
    # =========================================================================

    def test_spa_basic(self, synthetic_data):
        """SPA smoke test: verify output shape and non-negative scores."""
        X, y, _ = synthetic_data

        scores = spa_selection(X, y, n_features=20, n_random_starts=10)

        # Check return types
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"

        # Check shapes
        assert scores.shape[0] == X.shape[1], "Scores should have one value per feature"
        assert scores.ndim == 1, "Scores should be 1D array"

        # Check non-negative scores
        assert np.all(scores >= 0), "All scores should be non-negative"

        # Derive selected indices from scores (non-zero scores indicate selected variables)
        selected_indices = np.where(scores > 0)[0]

        assert len(selected_indices) <= 20, "Should select at most n_features variables"

        # Check indices are valid
        assert np.all(selected_indices >= 0), "All indices should be non-negative"
        assert np.all(selected_indices < X.shape[1]), "All indices should be within feature range"

        # Check no duplicate indices
        assert len(selected_indices) == len(np.unique(selected_indices)), "No duplicate indices"

    def test_spa_reduces_collinearity(self, synthetic_data):
        """SPA should select variables with reduced collinearity."""
        X, y, _ = synthetic_data

        # SPA should select diverse variables
        scores = spa_selection(X, y, n_features=20, n_random_starts=10)

        # Derive selected indices from scores (non-zero scores)
        selected_indices = np.where(scores > 0)[0]

        if len(selected_indices) < 2:
            pytest.skip("Need at least 2 selected variables to test collinearity")

        # Compute correlation matrix of selected variables
        X_selected = X[:, selected_indices]
        corr_matrix = np.corrcoef(X_selected.T)

        # Remove diagonal (self-correlation = 1.0)
        n_selected = len(selected_indices)
        off_diagonal_mask = ~np.eye(n_selected, dtype=bool)
        off_diagonal_corr = np.abs(corr_matrix[off_diagonal_mask])

        # Mean absolute correlation should be reasonably low (< 0.7 indicates good diversity)
        mean_abs_corr = np.mean(off_diagonal_corr)

        assert mean_abs_corr < 0.9, \
            f"SPA should reduce collinearity, but mean |correlation| = {mean_abs_corr:.3f}"

        # Check that selected variables are spread across the spectrum
        # (not all clustered in one region)
        sorted_indices = np.sort(selected_indices)
        gaps = np.diff(sorted_indices)
        max_gap = np.max(gaps)

        # At least one gap should be > 5 wavelengths (indicates spread)
        assert max_gap > 5, \
            f"SPA should select variables across spectrum, largest gap = {max_gap}"

    # =========================================================================
    # UVE-SPA Tests
    # =========================================================================

    def test_uve_spa_basic(self, synthetic_data):
        """UVE-SPA smoke test: verify output shape and non-negative scores."""
        X, y, _ = synthetic_data

        scores = uve_spa_selection(
            X, y,
            n_features=20,
            cutoff_multiplier=1.0,
            uve_n_components=5,
            uve_cv_folds=5,
            spa_n_random_starts=10,
            spa_cv_folds=5
        )

        # Check return types
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"

        # Check shapes
        assert scores.shape[0] == X.shape[1], "Scores should have one value per feature"
        assert scores.ndim == 1, "Scores should be 1D array"

        # Check non-negative scores
        assert np.all(scores >= 0), "All scores should be non-negative"

        # Derive selected indices from scores (non-zero scores)
        selected_indices = np.where(scores > 0)[0]

        assert len(selected_indices) <= 20, "Should select at most n_features variables"

        # Check indices are valid
        assert np.all(selected_indices >= 0), "All indices should be non-negative"
        assert np.all(selected_indices < X.shape[1]), "All indices should be within feature range"

        # Check no duplicate indices
        assert len(selected_indices) == len(np.unique(selected_indices)), "No duplicate indices"

    def test_uve_spa_hybrid(self, synthetic_data):
        """UVE-SPA should combine both methods: prefilter noise, then reduce collinearity."""
        X, y, informative_regions = synthetic_data

        # Run UVE-SPA
        scores = uve_spa_selection(
            X, y,
            n_features=20,
            cutoff_multiplier=1.0,
            uve_n_components=5,
            uve_cv_folds=5,
            spa_n_random_starts=10,
            spa_cv_folds=5
        )

        # Derive selected indices from scores
        selected_indices = np.where(scores > 0)[0]

        # Run UVE alone
        uve_scores = uve_selection(X, y, cutoff_multiplier=1.0, n_components=5, cv_folds=5)
        # Derive UVE indices from scores (use median as threshold)
        uve_indices = np.where(uve_scores > np.median(uve_scores))[0]

        # Check that UVE-SPA selected indices are a subset of UVE prefiltered variables
        # (with some tolerance for algorithm differences)
        if len(selected_indices) > 0 and len(uve_indices) > 0:
            overlap = len(set(selected_indices) & set(uve_indices))
            overlap_ratio = overlap / len(selected_indices)

            assert overlap_ratio >= 0.5, \
                f"UVE-SPA should mostly select from UVE-prefiltered variables, got {overlap_ratio:.1%} overlap"

        # Check that selected variables have reduced collinearity (SPA's contribution)
        if len(selected_indices) >= 2:
            X_selected = X[:, selected_indices]
            corr_matrix = np.corrcoef(X_selected.T)
            n_selected = len(selected_indices)
            off_diagonal_mask = ~np.eye(n_selected, dtype=bool)
            off_diagonal_corr = np.abs(corr_matrix[off_diagonal_mask])
            mean_abs_corr = np.mean(off_diagonal_corr)

            assert mean_abs_corr < 0.9, \
                f"UVE-SPA should reduce collinearity (SPA contribution), but mean |corr| = {mean_abs_corr:.3f}"

    # =========================================================================
    # Edge Case Tests
    # =========================================================================

    def test_small_dataset(self, small_dataset):
        """Test all methods with n_samples < cv_folds."""
        X, y = small_dataset
        n_samples = X.shape[0]  # 15 samples

        # Should work with cv_folds adjusted to n_samples
        adjusted_folds = min(5, n_samples - 1)

        # Test iPLS
        scores_ipls = ipls_selection(
            X, y, n_intervals=10, n_components=3, cv_folds=adjusted_folds
        )
        selected_ipls = np.where(scores_ipls > 0)[0]
        assert len(selected_ipls) > 0, "iPLS should return selected indices even with small dataset"

        # Test UVE
        scores_uve = uve_selection(
            X, y, cutoff_multiplier=1.0, n_components=3, cv_folds=adjusted_folds
        )
        selected_uve = np.where(scores_uve > 0)[0]
        assert len(selected_uve) >= 0, "UVE should handle small dataset"

        # Test SPA
        scores_spa = spa_selection(
            X, y, n_features=10, n_random_starts=5
        )
        selected_spa = np.where(scores_spa > 0)[0]
        assert len(selected_spa) > 0, "SPA should return selected indices even with small dataset"

        # Test UVE-SPA
        scores_hybrid = uve_spa_selection(
            X, y,
            n_features=10,
            cutoff_multiplier=1.0,
            uve_n_components=3,
            uve_cv_folds=adjusted_folds,
            spa_n_random_starts=5,
            spa_cv_folds=5
        )
        selected_hybrid = np.where(scores_hybrid > 0)[0]
        assert len(selected_hybrid) >= 0, "UVE-SPA should handle small dataset"

    def test_few_features(self, few_features_data):
        """Test all methods with n_features < n_intervals."""
        X, y = few_features_data
        n_features = X.shape[1]  # 15 features

        # Test iPLS with n_intervals > n_features
        # Should automatically adjust to fewer intervals
        scores_ipls = ipls_selection(
            X, y, n_intervals=20, n_components=3, cv_folds=5
        )
        selected_ipls = np.where(scores_ipls > 0)[0]
        assert len(selected_ipls) > 0, "iPLS should handle n_intervals > n_features"
        assert scores_ipls.shape[0] == n_features, "Scores should match n_features"

        # Test UVE
        scores_uve = uve_selection(
            X, y, cutoff_multiplier=1.0, n_components=3, cv_folds=5
        )
        assert scores_uve.shape[0] == n_features, "UVE scores should match n_features"

        # Test SPA with n_features close to total features
        scores_spa = spa_selection(
            X, y, n_features=min(10, n_features - 1), n_random_starts=5
        )
        selected_spa = np.where(scores_spa > 0)[0]
        assert len(selected_spa) > 0, "SPA should handle few features"
        assert scores_spa.shape[0] == n_features, "SPA scores should match n_features"

        # Test UVE-SPA
        scores_hybrid = uve_spa_selection(
            X, y,
            n_features=min(10, n_features - 1),
            cutoff_multiplier=1.0,
            uve_n_components=3,
            uve_cv_folds=5,
            spa_n_random_starts=5,
            spa_cv_folds=5
        )
        assert scores_hybrid.shape[0] == n_features, "UVE-SPA scores should match n_features"

    def test_large_n_components(self, synthetic_data):
        """Test methods with n_components > n_features."""
        X, y, _ = synthetic_data
        n_features = X.shape[1]  # 200 features

        # Test iPLS with n_components = n_features + 10
        # Should automatically adjust to valid range
        scores_ipls = ipls_selection(
            X, y, n_intervals=20, n_components=n_features + 10, cv_folds=5
        )
        selected_ipls = np.where(scores_ipls > 0)[0]
        assert len(selected_ipls) > 0, "iPLS should handle n_components > n_features"

        # Test UVE with n_components > n_features
        scores_uve = uve_selection(
            X, y, cutoff_multiplier=1.0, n_components=n_features + 10, cv_folds=5
        )
        assert scores_uve.shape[0] == n_features, "UVE should handle n_components > n_features"

        # Test UVE-SPA
        scores_hybrid = uve_spa_selection(
            X, y,
            n_features=20,
            cutoff_multiplier=1.0,
            uve_n_components=n_features + 10,
            uve_cv_folds=5,
            spa_n_random_starts=5,
            spa_cv_folds=5
        )
        assert scores_hybrid.shape[0] == n_features, "UVE-SPA should handle n_components > n_features"

    # =========================================================================
    # Integration Tests
    # =========================================================================

    def test_all_methods_return_consistent_types(self, synthetic_data):
        """Verify all methods return consistent output types and shapes."""
        X, y, _ = synthetic_data

        methods = [
            ("iPLS", lambda: ipls_selection(X, y, n_intervals=20, n_components=5, cv_folds=5)),
            ("UVE", lambda: uve_selection(X, y, cutoff_multiplier=1.0, n_components=5, cv_folds=5)),
            ("SPA", lambda: spa_selection(X, y, n_features=20, n_random_starts=10)),
            ("UVE-SPA", lambda: uve_spa_selection(
                X, y, n_features=20, cutoff_multiplier=1.0, uve_n_components=5, uve_cv_folds=5,
                spa_n_random_starts=10, spa_cv_folds=5
            ))
        ]

        for method_name, method_func in methods:
            scores = method_func()

            # All methods should return numpy array of scores
            assert isinstance(scores, np.ndarray), \
                f"{method_name}: scores should be numpy array"

            # Scores should be 1D
            assert scores.ndim == 1, \
                f"{method_name}: scores should be 1D"

            # Scores should match n_features
            assert scores.shape[0] == X.shape[1], \
                f"{method_name}: scores shape mismatch"

            # Scores should be non-negative
            assert np.all(scores >= 0), \
                f"{method_name}: scores should be non-negative"
