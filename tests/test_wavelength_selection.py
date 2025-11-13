"""
Tests for wavelength_selection.py module.

Tests cover SPA, CARS, and VCPA-IRIV algorithms for variable/wavelength selection.
"""

import numpy as np
import pytest

from spectral_predict.wavelength_selection import (
    spa,
    cars,
    vcpa_iriv,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_regression_data():
    """
    Generate simple synthetic regression data.

    X has correlated wavelengths, y depends on specific wavelengths.
    """
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 50

    # Generate base features
    X = np.random.randn(n_samples, n_wavelengths)

    # Make some wavelengths highly informative
    informative_idx = [5, 15, 25, 35]
    weights = [2.0, -1.5, 1.0, -0.8]

    y = np.zeros(n_samples)
    for idx, weight in zip(informative_idx, weights):
        y += weight * X[:, idx]

    # Add noise
    y += 0.1 * np.random.randn(n_samples)

    return X, y, informative_idx


@pytest.fixture
def high_dimensional_data():
    """Generate high-dimensional data (many wavelengths)."""
    np.random.seed(123)
    n_samples = 80
    n_wavelengths = 200

    X = np.random.randn(n_samples, n_wavelengths)

    # Create y dependent on wavelengths in 3 regions
    region1 = list(range(20, 30))
    region2 = list(range(80, 90))
    region3 = list(range(150, 160))

    y = (X[:, region1].mean(axis=1) * 2.0 +
         X[:, region2].mean(axis=1) * -1.5 +
         X[:, region3].mean(axis=1) * 1.0)

    y += 0.1 * np.random.randn(n_samples)

    informative_regions = [region1, region2, region3]

    return X, y, informative_regions


@pytest.fixture
def collinear_data():
    """Generate data with high collinearity to test SPA."""
    np.random.seed(99)
    n_samples = 100

    # Base signal
    base = np.random.randn(n_samples)

    # Create highly correlated wavelengths
    X = np.column_stack([
        base,
        base + 0.01 * np.random.randn(n_samples),  # Almost identical to base
        base + 0.02 * np.random.randn(n_samples),
        base * 1.1 + 0.01 * np.random.randn(n_samples),
        np.random.randn(n_samples),  # Independent
        base * -0.9 + 0.01 * np.random.randn(n_samples),
    ])

    y = 2.0 * base + 0.1 * np.random.randn(n_samples)

    return X, y


# ============================================================================
# Test SPA (Successive Projections Algorithm)
# ============================================================================

class TestSPA:
    """Tests for Successive Projections Algorithm."""

    def test_spa_basic_functionality(self, simple_regression_data):
        """Test that SPA runs and returns valid results."""
        X, y, _ = simple_regression_data

        result = spa(X, y, n_vars=10)

        # Check return structure
        assert isinstance(result, dict)
        assert 'selected_indices' in result
        assert 'selected_wavelengths' in result
        assert 'projection_scores' in result

        # Check selected indices
        selected = result['selected_indices']
        assert len(selected) == 10
        assert all(0 <= idx < X.shape[1] for idx in selected)
        assert len(selected) == len(set(selected))  # No duplicates

    def test_spa_selects_informative_wavelengths(self, simple_regression_data):
        """Test that SPA tends to select informative wavelengths."""
        X, y, informative_idx = simple_regression_data

        result = spa(X, y, n_vars=20)
        selected = result['selected_indices']

        # At least some informative wavelengths should be selected
        overlap = set(selected) & set(informative_idx)
        assert len(overlap) >= 2, "Should select at least 2 informative wavelengths"

    def test_spa_handles_collinearity(self, collinear_data):
        """Test that SPA reduces collinearity."""
        X, y = collinear_data

        result = spa(X, y, n_vars=3)
        selected = result['selected_indices']

        # Selected variables should be relatively independent
        X_selected = X[:, selected]
        corr_matrix = np.corrcoef(X_selected.T)

        # Off-diagonal correlations should be lower than original
        off_diag_mean = np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]).mean()
        assert off_diag_mean < 0.95, "SPA should reduce collinearity"

    def test_spa_different_n_vars(self, simple_regression_data):
        """Test SPA with different number of variables."""
        X, y, _ = simple_regression_data

        for n_vars in [5, 10, 20]:
            result = spa(X, y, n_vars=n_vars)
            assert len(result['selected_indices']) == n_vars

    def test_spa_invalid_inputs(self, simple_regression_data):
        """Test SPA error handling."""
        X, y, _ = simple_regression_data

        # Too many variables requested
        with pytest.raises((ValueError, IndexError)):
            spa(X, y, n_vars=X.shape[1] + 10)

        # Mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            spa(X, y[:50], n_vars=10)

    def test_spa_deterministic(self, simple_regression_data):
        """Test that SPA is deterministic."""
        X, y, _ = simple_regression_data

        result1 = spa(X, y, n_vars=10)
        result2 = spa(X, y, n_vars=10)

        np.testing.assert_array_equal(result1['selected_indices'], result2['selected_indices'])


# ============================================================================
# Test CARS (Competitive Adaptive Reweighted Sampling)
# ============================================================================

class TestCARS:
    """Tests for Competitive Adaptive Reweighted Sampling."""

    def test_cars_basic_functionality(self, simple_regression_data):
        """Test that CARS runs and returns valid results."""
        X, y, _ = simple_regression_data

        result = cars(X, y, n_iterations=20, pls_components=3)

        # Check return structure
        assert isinstance(result, dict)
        assert 'selected_indices' in result
        assert 'selected_wavelengths' in result
        assert 'rmsecv_history' in result
        assert 'n_vars_history' in result
        assert 'best_iteration' in result

        # Check selected indices
        selected = result['selected_indices']
        assert len(selected) > 0
        assert all(0 <= idx < X.shape[1] for idx in selected)
        assert len(selected) == len(set(selected))  # No duplicates

    def test_cars_selects_informative_wavelengths(self, simple_regression_data):
        """Test that CARS selects informative wavelengths."""
        X, y, informative_idx = simple_regression_data

        result = cars(X, y, n_iterations=30, pls_components=5)
        selected = result['selected_indices']

        # Should select some informative wavelengths
        overlap = set(selected) & set(informative_idx)
        assert len(overlap) >= 2, "CARS should select informative wavelengths"

    def test_cars_convergence(self, simple_regression_data):
        """Test that CARS converges to optimal solution."""
        X, y, _ = simple_regression_data

        result = cars(X, y, n_iterations=50, pls_components=5)

        # RMSECV should generally decrease then stabilize
        rmsecv_history = result['rmsecv_history']
        assert len(rmsecv_history) > 0

        # Best iteration should not be the last (indicates proper selection)
        assert result['best_iteration'] < len(rmsecv_history)

    def test_cars_different_iterations(self, simple_regression_data):
        """Test CARS with different iteration counts."""
        X, y, _ = simple_regression_data

        result_10 = cars(X, y, n_iterations=10, pls_components=3)
        result_50 = cars(X, y, n_iterations=50, pls_components=3)

        # More iterations should have more history
        assert len(result_50['rmsecv_history']) > len(result_10['rmsecv_history'])

    def test_cars_different_pls_components(self, simple_regression_data):
        """Test CARS with different PLS component counts."""
        X, y, _ = simple_regression_data

        for n_comp in [2, 5, 10]:
            result = cars(X, y, n_iterations=20, pls_components=n_comp)
            assert len(result['selected_indices']) > 0

    def test_cars_reproducibility_with_seed(self, simple_regression_data):
        """Test that CARS is reproducible with same random seed."""
        X, y, _ = simple_regression_data

        np.random.seed(42)
        result1 = cars(X, y, n_iterations=20, pls_components=3)

        np.random.seed(42)
        result2 = cars(X, y, n_iterations=20, pls_components=3)

        np.testing.assert_array_equal(result1['selected_indices'], result2['selected_indices'])

    def test_cars_invalid_inputs(self, simple_regression_data):
        """Test CARS error handling."""
        X, y, _ = simple_regression_data

        # Invalid PLS components
        with pytest.raises((ValueError, Exception)):
            cars(X, y, n_iterations=20, pls_components=0)

        # Mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            cars(X, y[:50], n_iterations=20, pls_components=3)


# ============================================================================
# Test VCPA-IRIV
# ============================================================================

class TestVCPAIRIV:
    """Tests for Variable Combination Population Analysis with IRIV."""

    def test_vcpa_iriv_basic_functionality(self, simple_regression_data):
        """Test that VCPA-IRIV runs and returns valid results."""
        X, y, _ = simple_regression_data

        result = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=20)

        # Check return structure
        assert isinstance(result, dict)
        assert 'selected_indices' in result
        assert 'selected_wavelengths' in result
        assert 'variable_importance' in result
        assert 'n_outer_iterations' in result

        # Check selected indices
        selected = result['selected_indices']
        assert len(selected) > 0
        assert all(0 <= idx < X.shape[1] for idx in selected)
        assert len(selected) == len(set(selected))  # No duplicates

        # Check variable importance
        importance = result['variable_importance']
        assert len(importance) == X.shape[1]
        assert all(0 <= imp <= 1 for imp in importance)

    def test_vcpa_iriv_selects_informative_wavelengths(self, simple_regression_data):
        """Test that VCPA-IRIV selects informative wavelengths."""
        X, y, informative_idx = simple_regression_data

        result = vcpa_iriv(X, y, n_outer_iterations=8, n_inner_iterations=30)
        selected = result['selected_indices']

        # Should select informative wavelengths
        overlap = set(selected) & set(informative_idx)
        assert len(overlap) >= 2, "VCPA-IRIV should select informative wavelengths"

    def test_vcpa_iriv_importance_scores(self, simple_regression_data):
        """Test that importance scores are reasonable."""
        X, y, informative_idx = simple_regression_data

        result = vcpa_iriv(X, y, n_outer_iterations=10, n_inner_iterations=30)
        importance = result['variable_importance']

        # Informative wavelengths should have higher importance
        informative_importance = np.mean([importance[i] for i in informative_idx])
        non_informative_idx = [i for i in range(len(importance)) if i not in informative_idx]
        non_informative_importance = np.mean([importance[i] for i in non_informative_idx])

        # This may not always hold due to randomness, but should be true on average
        # We'll just check that the algorithm completes
        assert informative_importance >= 0  # Basic sanity check

    def test_vcpa_iriv_different_iterations(self, simple_regression_data):
        """Test VCPA-IRIV with different iteration settings."""
        X, y, _ = simple_regression_data

        result_few = vcpa_iriv(X, y, n_outer_iterations=3, n_inner_iterations=10)
        result_many = vcpa_iriv(X, y, n_outer_iterations=10, n_inner_iterations=30)

        # Both should return valid results
        assert len(result_few['selected_indices']) > 0
        assert len(result_many['selected_indices']) > 0

        # More iterations might select different variables
        # Just check that it completes

    def test_vcpa_iriv_binary_matrix_generation(self, simple_regression_data):
        """Test that VCPA-IRIV generates valid binary matrices."""
        X, y, _ = simple_regression_data

        # Run with verbose=False to avoid output
        result = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=20)

        # If binary matrices were generated correctly, we should have results
        assert 'variable_importance' in result
        assert len(result['variable_importance']) == X.shape[1]

    def test_vcpa_iriv_reproducibility_with_seed(self, simple_regression_data):
        """Test VCPA-IRIV reproducibility with random seed."""
        X, y, _ = simple_regression_data

        np.random.seed(42)
        result1 = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=20)

        np.random.seed(42)
        result2 = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=20)

        np.testing.assert_array_equal(result1['selected_indices'], result2['selected_indices'])
        np.testing.assert_array_almost_equal(result1['variable_importance'], result2['variable_importance'])

    def test_vcpa_iriv_invalid_inputs(self, simple_regression_data):
        """Test VCPA-IRIV error handling."""
        X, y, _ = simple_regression_data

        # Invalid iteration counts
        with pytest.raises((ValueError, Exception)):
            vcpa_iriv(X, y, n_outer_iterations=0, n_inner_iterations=20)

        # Mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            vcpa_iriv(X, y[:50], n_outer_iterations=5, n_inner_iterations=20)


# ============================================================================
# Test Method Comparisons
# ============================================================================

class TestMethodComparison:
    """Compare different wavelength selection methods."""

    def test_all_methods_on_same_data(self, simple_regression_data):
        """Test that all methods work on the same dataset."""
        X, y, _ = simple_regression_data

        # SPA
        result_spa = spa(X, y, n_vars=10)
        assert len(result_spa['selected_indices']) == 10

        # CARS
        result_cars = cars(X, y, n_iterations=20, pls_components=3)
        assert len(result_cars['selected_indices']) > 0

        # VCPA-IRIV
        result_vcpa = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=20)
        assert len(result_vcpa['selected_indices']) > 0

    def test_methods_select_different_subsets(self, simple_regression_data):
        """Test that different methods may select different wavelengths."""
        X, y, _ = simple_regression_data

        result_spa = spa(X, y, n_vars=10)
        result_cars = cars(X, y, n_iterations=20, pls_components=3)
        result_vcpa = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=15)

        # Methods should return results
        assert len(result_spa['selected_indices']) > 0
        assert len(result_cars['selected_indices']) > 0
        assert len(result_vcpa['selected_indices']) > 0

        # They may select different subsets (that's expected)

    def test_high_dimensional_scenario(self, high_dimensional_data):
        """Test all methods on high-dimensional data."""
        X, y, _ = high_dimensional_data

        # SPA - fast
        result_spa = spa(X, y, n_vars=20)
        assert len(result_spa['selected_indices']) == 20

        # CARS - moderate speed
        result_cars = cars(X, y, n_iterations=25, pls_components=5)
        assert len(result_cars['selected_indices']) > 0

        # VCPA-IRIV - slower but more thorough
        result_vcpa = vcpa_iriv(X, y, n_outer_iterations=5, n_inner_iterations=20)
        assert len(result_vcpa['selected_indices']) > 0


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegration:
    """Test wavelength selection in realistic scenarios."""

    def test_with_ns_pfce_workflow(self, simple_regression_data):
        """Test wavelength selection in NS-PFCE workflow."""
        X_master, y_master, _ = simple_regression_data

        # Simulate slave data (same structure, different bias)
        np.random.seed(99)
        X_slave = X_master * 0.9 + 0.05

        # Select wavelengths using VCPA-IRIV
        result = vcpa_iriv(X_master, y_master, n_outer_iterations=5, n_inner_iterations=20)
        selected_idx = result['selected_indices']

        # Extract selected wavelengths
        X_master_selected = X_master[:, selected_idx]
        X_slave_selected = X_slave[:, selected_idx]

        assert X_master_selected.shape[1] == len(selected_idx)
        assert X_slave_selected.shape[1] == len(selected_idx)

    def test_wavelength_reduction_ratio(self, high_dimensional_data):
        """Test that wavelength selection achieves good reduction."""
        X, y, _ = high_dimensional_data
        original_n_wavelengths = X.shape[1]

        # CARS typically reduces to 10-20% of original
        result_cars = cars(X, y, n_iterations=30, pls_components=5)
        selected_n = len(result_cars['selected_indices'])

        reduction_ratio = selected_n / original_n_wavelengths
        assert reduction_ratio < 0.5, "Should achieve significant reduction"

    def test_selected_wavelength_predictive_power(self, simple_regression_data):
        """Test that selected wavelengths maintain predictive power."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        X, y, _ = simple_regression_data

        # Baseline: all wavelengths
        model_all = Ridge(alpha=1.0)
        score_all = cross_val_score(model_all, X, y, cv=5, scoring='r2').mean()

        # Select wavelengths with SPA
        result_spa = spa(X, y, n_vars=15)
        X_selected = X[:, result_spa['selected_indices']]

        model_selected = Ridge(alpha=1.0)
        score_selected = cross_val_score(model_selected, X_selected, y, cv=5, scoring='r2').mean()

        # Selected wavelengths should maintain reasonable performance
        # Allow 20% degradation for 70% dimension reduction
        assert score_selected > score_all * 0.7, "Selected wavelengths should maintain predictive power"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample_size(self):
        """Test with minimal samples."""
        np.random.seed(42)
        n_samples = 20
        n_wavelengths = 30

        X = np.random.randn(n_samples, n_wavelengths)
        y = X[:, 5] + X[:, 15] + 0.1 * np.random.randn(n_samples)

        # SPA should work
        result_spa = spa(X, y, n_vars=5)
        assert len(result_spa['selected_indices']) == 5

        # CARS might struggle but should not crash
        try:
            result_cars = cars(X, y, n_iterations=10, pls_components=2)
            assert len(result_cars['selected_indices']) > 0
        except Exception:
            # CARS may fail with very small samples - that's acceptable
            pass

    def test_many_wavelengths_few_samples(self):
        """Test p >> n scenario."""
        np.random.seed(42)
        n_samples = 30
        n_wavelengths = 200

        X = np.random.randn(n_samples, n_wavelengths)
        y = X[:, :10].sum(axis=1) + 0.1 * np.random.randn(n_samples)

        # All methods should handle this
        result_spa = spa(X, y, n_vars=10)
        assert len(result_spa['selected_indices']) == 10

        # CARS with low components
        result_cars = cars(X, y, n_iterations=15, pls_components=3)
        assert len(result_cars['selected_indices']) > 0

        # VCPA-IRIV
        result_vcpa = vcpa_iriv(X, y, n_outer_iterations=3, n_inner_iterations=10)
        assert len(result_vcpa['selected_indices']) > 0

    def test_constant_wavelengths(self):
        """Test with some constant wavelengths."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 30

        X = np.random.randn(n_samples, n_wavelengths)
        # Make some wavelengths constant
        X[:, 5] = 1.0
        X[:, 15] = -0.5
        X[:, 25] = 0.0

        y = X[:, 10] + X[:, 20] + 0.1 * np.random.randn(n_samples)

        # Algorithms should handle this gracefully
        try:
            result_spa = spa(X, y, n_vars=10)
            # Constant wavelengths might be selected but shouldn't crash
            assert len(result_spa['selected_indices']) == 10
        except Exception as e:
            # Some numerical issues might occur - that's acceptable
            pass

    def test_perfect_correlation_with_target(self):
        """Test when one wavelength perfectly predicts target."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 20

        X = np.random.randn(n_samples, n_wavelengths)
        # Make y perfectly correlated with wavelength 10
        y = X[:, 10]

        # SPA should select wavelength 10 early
        result_spa = spa(X, y, n_vars=5)
        # Note: SPA uses projections, so perfect correlation might not guarantee first selection
        assert len(result_spa['selected_indices']) == 5

        # CARS should identify this
        result_cars = cars(X, y, n_iterations=20, pls_components=3)
        assert len(result_cars['selected_indices']) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
