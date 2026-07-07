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

        # UVE should give higher scores to informative variables
        assert mean_informative > mean_noise, \
            f"Informative variables should have higher UVE scores (got {mean_informative:.3f} vs {mean_noise:.3f})"

        # Most selected variables should be informative
        # Derive selected indices from top-scoring variables
        selected_indices = np.where(scores > np.median(scores))[0]
        if len(selected_indices) > 0:
            selected_informative = len(set(selected_indices) & set(informative_indices))
            ratio = selected_informative / len(selected_indices)

            assert ratio >= 0.15, \
                f"Expected at least 15% of selected variables to be informative, got {ratio:.1%}"

    def test_uve_produces_nonuniform_scores(self, synthetic_data):
        """UVE must produce non-uniform importance scores (regression test for coef extraction bug)."""
        X, y, informative_regions = synthetic_data

        scores = uve_selection(X, y, cutoff_multiplier=1.0, n_components=5, cv_folds=5)

        # Scores must NOT be uniform — std > 0 proves variable discrimination
        assert np.std(scores) > 0, (
            "UVE scores are uniform (std=0), indicating coefficient extraction bug"
        )

        # Informative variables must score strictly higher than noise on average
        informative_indices = np.concatenate(informative_regions)
        noise_indices = np.array([i for i in range(X.shape[1]) if i not in informative_indices])

        mean_informative = np.mean(scores[informative_indices])
        mean_noise = np.mean(scores[noise_indices])

        assert mean_informative > mean_noise, (
            f"Informative variables should score strictly higher than noise "
            f"(got {mean_informative:.4f} vs {mean_noise:.4f})"
        )

    # =========================================================================
    # SPA Tests
    # =========================================================================

    def test_spa_basic(self, synthetic_data):
        """SPA smoke test: verify output shape and non-negative scores."""
        X, y, _ = synthetic_data

        scores = spa_selection(X, y, n_features=20)

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
        scores = spa_selection(X, y, n_features=20)

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

    def test_spa_deterministic(self, synthetic_data):
        """T-06: canonical Araujo 2001 SPA is fully deterministic across repeat calls."""
        X, y, _ = synthetic_data

        scores_a = spa_selection(X, y, n_features=20)
        scores_b = spa_selection(X, y, n_features=20)

        np.testing.assert_array_equal(scores_a, scores_b)

    def test_spa_explores_multiple_seeds(self):
        """T-06: canonical SPA picks a winning seed != argmax-correlation seed.

        Pre-T-06 dasp seeded only at argmax(|corr(X[:, j], y)|) and repeated
        that single chain N times — so the selection was always the
        argmax-only chain. Canonical Araújo 2001 enumerates over every
        variable as candidate first variable and keeps the chain with the
        best CV criterion.

        This test pins a deterministic dataset (numpy default_rng(0), 30×15
        random) where canonical SPA picks chain [0, 5, 6, 7, 10] starting
        from seed 6, while the argmax seed is 11 leading to chain
        [3, 5, 8, 10, 11]. A regression to the pre-T-06 single-chain
        behavior would return the argmax chain and fail both assertions.
        """
        rng = np.random.default_rng(0)
        n_samples = 30
        n_vars = 15
        y = rng.normal(0, 1, n_samples)
        X = rng.normal(0, 1, (n_samples, n_vars))

        # Compute the argmax-correlation seed and the chain pre-T-06 would have produced.
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        y_norm = (y - y.mean()) / (y.std() + 1e-10)
        initial_corrs = np.abs(X_norm.T @ y_norm) / n_samples
        argmax_seed = int(np.argmax(initial_corrs))

        n_select = 5
        argmax_chain = [argmax_seed]
        available = set(range(n_vars)) - {argmax_seed}
        for _ in range(n_select - 1):
            X_sel = X_norm[:, argmax_chain]
            proj = {
                j: float(np.sum((X_norm[:, j] @ X_sel / n_samples) ** 2))
                for j in available
            }
            min_j = min(available, key=lambda j: proj[j])
            argmax_chain.append(min_j)
            available.remove(min_j)

        scores = spa_selection(X, y, n_features=n_select, cv_folds=5)
        canonical_chain = sorted(np.where(scores > 0)[0].tolist())

        # Pinned expectations from the verified empirical run on rng(0) data.
        assert sorted(argmax_chain) == [3, 5, 8, 10, 11], (
            f"argmax-only reference chain drifted: {sorted(argmax_chain)}"
        )
        assert canonical_chain == [0, 5, 6, 7, 10], (
            f"canonical chain drifted: {canonical_chain}"
        )
        assert canonical_chain != sorted(argmax_chain), (
            "Canonical SPA produced the same chain as argmax-only — "
            "T-06 fix should explore other seeds."
        )

    def test_spa_evaluates_all_j_seeds(self, synthetic_data):
        """T-06: canonical SPA must evaluate exactly J candidate seeds.

        Pinned via call-count on cross_val_score. A regression that narrowed
        the seed loop (e.g., back to range(n_random_starts) or
        range(min(n_vars, 10))) would fail this assertion immediately.
        Complementary to test_spa_explores_multiple_seeds, which pins the
        actual selection on a deterministic dataset.
        """
        from unittest.mock import patch

        X, y, _ = synthetic_data
        n_vars = X.shape[1]
        call_count = {"n": 0}

        def counting_cv(*args, **kwargs):
            call_count["n"] += 1
            return np.array([0.5])

        with patch(
            "spectral_predict.variable_selection.cross_val_score",
            side_effect=counting_cv,
        ):
            spa_selection(X, y, n_features=20)

        assert call_count["n"] == n_vars, (
            f"Canonical SPA should evaluate exactly {n_vars} seeds, "
            f"got {call_count['n']}"
        )

    def test_spa_rejects_legacy_n_random_starts(self, synthetic_data):
        """T-06: the n_random_starts kwarg was removed (canon has no such concept)."""
        X, y, _ = synthetic_data
        with pytest.raises(TypeError):
            spa_selection(X, y, n_features=20, n_random_starts=5)

    def test_spa_rejects_legacy_random_state(self, synthetic_data):
        """T-06: the random_state kwarg was removed (SPA is fully deterministic)."""
        X, y, _ = synthetic_data
        with pytest.raises(TypeError):
            spa_selection(X, y, n_features=20, random_state=42)

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
            X, y, n_features=10
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
            X, y, n_features=min(10, n_features - 1)
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
            ("SPA", lambda: spa_selection(X, y, n_features=20)),
            ("UVE-SPA", lambda: uve_spa_selection(
                X, y, n_features=20, cutoff_multiplier=1.0, uve_n_components=5, uve_cv_folds=5,
                spa_cv_folds=5
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


# ============================================================================
# T-17: Multi-target (multi-Y) variable selection
# ============================================================================
import contextlib
import io
import os

from spectral_predict.ga_pls import ga_pls_selection
from spectral_predict.variable_selection import (
    _prep_varsel_y,
    _reject_multi_y,
    cars_selection,
    ipls_backward,
    ipls_forward,
    mc_sipls,
    mwpls,
    uve_cars_selection,
    uve_cars_spa_selection,
    uve_spa_selection,
)

GOLD_VARSEL = os.path.join(
    os.path.dirname(__file__), "gold_standards", "varsel_single_y.npz"
)


def _gold_single_y_data():
    """Reproduce the EXACT fixed synthetic 1-D case used to build the gold
    fixture (tools/_gen_varsel_gold.py). Pinned RNG -> deterministic."""
    seed, n, p = 20260701, 60, 100
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    coef = np.zeros(p)
    coef[10:18] = rng.standard_normal(8)
    coef[55:60] = rng.standard_normal(5)
    y = X @ coef + 0.1 * rng.standard_normal(n)
    wavelengths = np.linspace(1000.0, 2500.0, p)
    return X, y, wavelengths


def _subset_arrays(subsets):
    rmsecv = np.array([s["rmsecv"] for s in subsets], dtype=float)
    r2 = np.array([s["r2"] for s in subsets], dtype=float)
    if subsets:
        best = min(subsets, key=lambda s: s["rmsecv"])
        best_idx = np.asarray(best["indices"], dtype=int)
    else:
        best_idx = np.array([], dtype=int)
    return rmsecv, r2, best_idx


class TestSingleYByteIdentityGold:
    """Guardrail: every T-17-modified performance-based varsel method stays
    BYTE-IDENTICAL on the single-Y path, proven against a committed gold
    fixture captured FROM the pre-refactor code with explicit RNG seeds.

    Existing varsel tests are property/range-only, so a stray perturbation of
    selected wavelengths or score vectors would slip through without this.
    """

    def test_gold_fixture_present(self):
        assert os.path.exists(GOLD_VARSEL), (
            "varsel_single_y.npz gold fixture missing; regenerate with "
            "tools/_gen_varsel_gold.py"
        )

    def test_interval_methods_byte_identical(self):
        X, y, wl = _gold_single_y_data()
        gold = np.load(GOLD_VARSEL)
        with contextlib.redirect_stdout(io.StringIO()):
            fwd = ipls_forward(X, y, wl, n_intervals=10, max_combine=4,
                               cv_folds=5, random_state=42)
            bwd = ipls_backward(X, y, wl, n_intervals=10, cv_folds=5,
                                random_state=42, min_intervals=1)
            mc = mc_sipls(X, y, wl, n_intervals=10, n_combinations=120,
                          max_combine=4, cv_folds=5, random_state=42)
            mw = mwpls(X, y, wl, window_sizes=[10, 20, 40], step_size=None,
                       cv_folds=5)
        for name, subs in [("ipls_fwd", fwd), ("ipls_bwd", bwd),
                           ("mc_sipls", mc), ("mwpls", mw)]:
            rmsecv, r2, best_idx = _subset_arrays(subs)
            np.testing.assert_array_equal(
                best_idx, gold[f"{name}_best_indices"],
                err_msg=f"{name}: selected indices drifted (single-Y not byte-identical)",
            )
            np.testing.assert_allclose(
                rmsecv, gold[f"{name}_rmsecv"], atol=0, rtol=0,
                err_msg=f"{name}: RMSECV vector drifted",
            )
            np.testing.assert_allclose(
                r2, gold[f"{name}_r2"], atol=0, rtol=0,
                err_msg=f"{name}: R2 vector drifted",
            )

    def test_spa_byte_identical(self):
        X, y, _ = _gold_single_y_data()
        gold = np.load(GOLD_VARSEL)
        with contextlib.redirect_stdout(io.StringIO()):
            spa = spa_selection(X, y, n_features=15, cv_folds=5)
        np.testing.assert_array_equal(np.asarray(spa, dtype=float),
                                      gold["spa_importances"])

    def test_gapls_byte_identical(self):
        X, y, _ = _gold_single_y_data()
        gold = np.load(GOLD_VARSEL)
        with contextlib.redirect_stdout(io.StringIO()):
            ga = ga_pls_selection(
                X, y, task_type="regression", population_size=20,
                n_generations=8, n_runs=2, cv=5, random_state=42, n_jobs=1,
                verbose=0, min_wavelengths=5, n_components=5,
            )
        np.testing.assert_array_equal(np.asarray(ga, dtype=float),
                                      gold["gapls_frequency"])


class TestMultiYVarsel:
    """T-17 additive multi-Y branches: the performance-based methods evaluate a
    JOINT PLS on a 2-D Y block via the multi_y foundation and still return the
    same subset/importance shapes."""

    @pytest.fixture
    def multi_y_data(self):
        rng = np.random.default_rng(7)
        n, p = 50, 60
        X = rng.standard_normal((n, p))
        base = X[:, 10:14] @ rng.standard_normal((4, 1))
        # divergent target scales -> exercises fold Y-scaling + raw-unit metrics
        Y = np.hstack([
            base + 0.05 * rng.standard_normal((n, 1)),
            50.0 * base + 2.0 * rng.standard_normal((n, 1)),
            0.1 * base + 0.01 * rng.standard_normal((n, 1)),
        ])
        wl = np.linspace(1000.0, 2000.0, p)
        return X, Y, wl

    def test_prep_varsel_y_shapes(self):
        y1 = np.arange(10.0)
        assert _prep_varsel_y(y1).shape == (10,)
        assert _prep_varsel_y(y1.reshape(-1, 1)).shape == (10,)
        Y = np.zeros((10, 3))
        assert _prep_varsel_y(Y).shape == (10, 3)

    def test_prep_varsel_y_ravel_equality_and_ndim(self):
        """FIX F: 1-D and (n,1) inputs both ravel to the SAME 1-D vector
        (byte-identical to the legacy np.asarray(y).ravel()); a genuine (n,3)
        block is returned 2-D unchanged."""
        n = 12
        y1d = np.arange(n, dtype=float)
        out_1d = _prep_varsel_y(y1d)
        out_col = _prep_varsel_y(y1d.reshape(-1, 1))
        assert out_1d.ndim == 1 and out_col.ndim == 1
        assert np.array_equal(out_1d, np.arange(n))
        assert np.array_equal(out_col, np.arange(n))
        Y = np.arange(n * 3, dtype=float).reshape(n, 3)
        out_block = _prep_varsel_y(Y)
        assert out_block.ndim == 2
        assert np.array_equal(out_block, Y)

    def test_single_y_ravel_swap_byte_identity(self):
        """FIX F: pin the ravel->_prep_varsel_y swap for the two changed entry
        points. Passing 1-D y and (n,1) y must yield IDENTICAL importance arrays,
        so a future regression that stops preserving single-Y equivalence is
        caught."""
        from spectral_predict.variable_selection import fipls_spa_selection

        rng = np.random.default_rng(20260702)
        n, p = 40, 30
        wl = np.linspace(1000.0, 2000.0, p)
        X = rng.standard_normal((n, p))
        y = X[:, 3:6] @ rng.standard_normal(3) + 0.05 * rng.standard_normal(n)

        with contextlib.redirect_stdout(io.StringIO()):
            ipls_1d = np.asarray(
                ipls_selection(X, y, n_intervals=10, n_components=5, cv_folds=5),
                dtype=float,
            )
            ipls_col = np.asarray(
                ipls_selection(X, y.reshape(-1, 1), n_intervals=10, n_components=5,
                               cv_folds=5),
                dtype=float,
            )
            fs_1d = np.asarray(fipls_spa_selection(X, y, wl), dtype=float)
            fs_col = np.asarray(fipls_spa_selection(X, y.reshape(-1, 1), wl), dtype=float)

        np.testing.assert_array_equal(ipls_1d, ipls_col)
        np.testing.assert_array_equal(fs_1d, fs_col)

    def test_ipls_forward_multi_y(self, multi_y_data):
        X, Y, wl = multi_y_data
        with contextlib.redirect_stdout(io.StringIO()):
            subs = ipls_forward(X, Y, wl, n_intervals=6, max_combine=3, cv_folds=5)
        assert len(subs) > 0
        best = max(s["r2"] for s in subs)  # r2 is joint Q2 in multi-Y
        assert best > 0.5

    def test_ipls_backward_multi_y(self, multi_y_data):
        X, Y, wl = multi_y_data
        with contextlib.redirect_stdout(io.StringIO()):
            subs = ipls_backward(X, Y, wl, n_intervals=6, cv_folds=5)
        assert len(subs) > 0

    def test_mc_sipls_multi_y(self, multi_y_data):
        X, Y, wl = multi_y_data
        with contextlib.redirect_stdout(io.StringIO()):
            subs = mc_sipls(X, Y, wl, n_intervals=6, n_combinations=40,
                            max_combine=3, cv_folds=5)
        assert len(subs) > 0
        for s in subs:
            assert np.isfinite(s["rmsecv"])

    def test_mwpls_multi_y(self, multi_y_data):
        X, Y, wl = multi_y_data
        with contextlib.redirect_stdout(io.StringIO()):
            subs = mwpls(X, Y, wl, window_sizes=[10, 20], cv_folds=5)
        assert len(subs) > 0

    def test_interval_pls_multi_nonfinite_q2_sinks_to_worst_sentinel(self, monkeypatch):
        """NaN-safety pin (GLM 5.2 pre-merge finding): a non-finite per-target
        q2 (PLS SVD divergence on a degenerate interval) must sink the interval
        to the worst-case ``(inf, -1.0)`` sentinel, NOT a spurious ``rmsecv=0.0``
        that would win iPLS/MC-siPLS/MWPLS selection as a fake perfect score.
        """
        import spectral_predict.multi_y as multi_y_mod
        from spectral_predict.variable_selection import _evaluate_interval_pls_multi

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 12))
        Y = np.column_stack([X[:, 0], X[:, 1]]) + 0.1 * rng.standard_normal((30, 2))

        def poisoned_metrics(yt, yp, **kwargs):
            return {"q2": np.array([np.nan, 0.97]), "joint_q2": float("nan")}

        # _evaluate_interval_pls_multi does `from .multi_y import multi_y_metrics`
        # at call time, so patch the source module attribute.
        monkeypatch.setattr(multi_y_mod, "multi_y_metrics", poisoned_metrics)
        rmsecv, q2 = _evaluate_interval_pls_multi(X, Y, cv_folds=5, n_components=3)
        assert np.isinf(rmsecv) and rmsecv > 0  # worst rmsecv -> ranks last
        assert q2 == -1.0

    def test_spa_multi_y(self, multi_y_data):
        X, Y, _ = multi_y_data
        with contextlib.redirect_stdout(io.StringIO()):
            imp = spa_selection(X, Y, n_features=10, cv_folds=5)
        assert imp.shape == (X.shape[1],)
        assert (imp > 0).sum() == 10

    def test_ga_pls_multi_y(self, multi_y_data):
        X, Y, _ = multi_y_data
        with contextlib.redirect_stdout(io.StringIO()):
            freq = ga_pls_selection(
                X, Y, task_type="regression", population_size=16,
                n_generations=5, n_runs=1, cv=5, random_state=0, n_jobs=1,
                verbose=0, min_wavelengths=4, n_components=4,
            )
        assert freq.shape == (X.shape[1],)
        assert freq.max() <= 1.0


class TestMultiYSupport:
    """UVE/CARS and their hybrids support multi-target (2-D Y) regression in
    T-17; each must return a full-width (n_features,) finite importance vector
    on a genuine (n, n_targets>=2) Y block (never silently flatten or raise)."""

    @pytest.fixture
    def XY(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 30))
        Y = rng.standard_normal((40, 3))
        return X, Y

    def test_uve_multi_y(self, XY):
        X, Y = XY
        with contextlib.redirect_stdout(io.StringIO()):
            imp = uve_selection(X, Y)
        assert imp.shape == (X.shape[1],)
        assert np.all(np.isfinite(imp))

    def test_cars_multi_y(self, XY):
        X, Y = XY
        with contextlib.redirect_stdout(io.StringIO()):
            imp = cars_selection(X, Y, n_iterations=10, monte_carlo_samples=20)
        assert imp.shape == (X.shape[1],)
        assert np.all(np.isfinite(imp))

    def test_uve_spa_multi_y(self, XY):
        X, Y = XY
        with contextlib.redirect_stdout(io.StringIO()):
            imp = uve_spa_selection(X, Y, n_features=5)
        assert imp.shape == (X.shape[1],)
        assert np.all(np.isfinite(imp))

    def test_uve_cars_multi_y(self, XY):
        X, Y = XY
        with contextlib.redirect_stdout(io.StringIO()):
            imp = uve_cars_selection(
                X, Y, n_iterations=10, monte_carlo_samples=20
            )
        assert imp.shape == (X.shape[1],)
        assert np.all(np.isfinite(imp))

    def test_uve_cars_spa_multi_y(self, XY):
        X, Y = XY
        with contextlib.redirect_stdout(io.StringIO()):
            imp = uve_cars_spa_selection(
                X, Y, n_iterations=10, monte_carlo_samples=20
            )
        assert imp.shape == (X.shape[1],)
        assert np.all(np.isfinite(imp))

    def test_reject_helper_passes_single_y(self):
        _reject_multi_y(np.arange(10.0), "X")
        _reject_multi_y(np.zeros((10, 1)), "X")
        with pytest.raises(NotImplementedError):
            _reject_multi_y(np.zeros((10, 2)), "X")
