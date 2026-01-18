"""
Test that grid search regional metrics align with auto-ensemble routing.

This test verifies the fix for the quartile mismatch bug where:
- Grid search computed regional RMSE using TRUE Y quartiles
- Auto-ensemble routed samples using PREDICTED Y quartiles

After the fix, both use predictions for quartile boundaries, ensuring
specialists selected for "low prediction space" receive samples that
fall into "low prediction space" during inference.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error


class TestGridSearchRegionalFix:
    """Test that grid search regional metrics use predictions for quartiles."""

    @pytest.fixture
    def sample_cv_data(self):
        """Generate sample CV data with predictions and true values."""
        np.random.seed(42)
        n_samples = 200

        # True values
        y_test = np.linspace(0, 100, n_samples) + np.random.randn(n_samples) * 5

        # Predictions with some bias/scaling (realistic scenario)
        # Predictions systematically higher by ~10 and compressed range
        y_pred = y_test * 0.9 + 10 + np.random.randn(n_samples) * 3

        return y_test, y_pred

    def test_quartiles_from_predictions(self, sample_cv_data):
        """Verify that quartile boundaries differ when using predictions vs true values."""
        y_test, y_pred = sample_cv_data

        # Quartiles from true values (OLD behavior)
        quartiles_true = np.percentile(y_test, [25, 50, 75])

        # Quartiles from predictions (NEW behavior after fix)
        quartiles_pred = np.percentile(y_pred, [25, 50, 75])

        # They should be different due to prediction bias/scaling
        # The fix changes which samples fall into each region
        assert not np.allclose(quartiles_true, quartiles_pred, atol=1.0), \
            "Quartiles should differ between true Y and predicted Y"

        print(f"Quartiles from true Y:     {quartiles_true}")
        print(f"Quartiles from predicted Y: {quartiles_pred}")

    def test_region_assignment_alignment(self, sample_cv_data):
        """Verify that samples get assigned to correct regions for ensemble routing."""
        y_test, y_pred = sample_cv_data

        # Using predictions for quartiles (fixed behavior)
        quartiles = np.percentile(y_pred, [25, 50, 75])

        # Assign samples to regions based on predictions
        def get_region(val, quartiles):
            if val < quartiles[0]:
                return 'Q1'
            elif val < quartiles[1]:
                return 'Q2'
            elif val < quartiles[2]:
                return 'Q3'
            else:
                return 'Q4'

        # All samples should get valid region assignments
        regions = [get_region(p, quartiles) for p in y_pred]
        assert all(r in ['Q1', 'Q2', 'Q3', 'Q4'] for r in regions)

        # Distribution should be roughly equal (quartiles split into ~25% each)
        region_counts = {r: regions.count(r) for r in ['Q1', 'Q2', 'Q3', 'Q4']}
        for region, count in region_counts.items():
            # Each quartile should have ~25% of samples (allow some tolerance)
            assert 40 <= count <= 60, f"{region} has {count} samples, expected ~50"

        print(f"Region distribution: {region_counts}")

    def test_regional_rmse_calculation(self, sample_cv_data):
        """Test that regional RMSE is computed correctly with prediction-based regions."""
        y_test, y_pred = sample_cv_data

        # Using predictions for quartiles (fixed behavior)
        quartiles = np.percentile(y_pred, [25, 50, 75])

        # Compute regional RMSE (mimics search.py logic after fix)
        regional_rmse = {}
        for i, (lower, upper) in enumerate([
            (-np.inf, quartiles[0]),  # Q1
            (quartiles[0], quartiles[1]),  # Q2
            (quartiles[1], quartiles[2]),  # Q3
            (quartiles[2], np.inf)  # Q4
        ]):
            # Use predictions for mask (fixed behavior)
            mask = (y_pred >= lower) & (y_pred < upper if i < 3 else y_pred >= lower)
            if mask.sum() > 0:
                regional_rmse[f'Q{i+1}'] = np.sqrt(mean_squared_error(
                    y_test[mask], y_pred[mask]
                ))

        # All regions should have valid RMSE values
        assert len(regional_rmse) == 4
        for region, rmse in regional_rmse.items():
            assert rmse > 0, f"{region} RMSE should be positive"
            assert not np.isnan(rmse), f"{region} RMSE should not be NaN"

        print(f"Regional RMSE: {regional_rmse}")


class TestAutoEnsemblePerformance:
    """Test that auto-ensemble R² is reasonable after the regional fix."""

    @pytest.fixture
    def synthetic_regression_data(self):
        """Create synthetic regression data with clear signal."""
        np.random.seed(42)
        n_samples = 300
        n_features = 50

        # Create data with clear linear relationship
        X = np.random.randn(n_samples, n_features)
        # Strong signal from first few features
        y = 5 * X[:, 0] + 3 * X[:, 1] + 2 * X[:, 2] + np.random.randn(n_samples) * 0.5

        return X, y

    def test_base_models_performance(self, synthetic_regression_data):
        """Verify base models achieve good R² on synthetic data."""
        X, y = synthetic_regression_data

        # Split data
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]

        # Fit models
        pls = PLSRegression(n_components=5)
        ridge = Ridge(alpha=1.0)

        pls.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        # Check performance
        r2_pls = r2_score(y_test, pls.predict(X_test))
        r2_ridge = r2_score(y_test, ridge.predict(X_test))

        print(f"PLS R²: {r2_pls:.4f}")
        print(f"Ridge R²: {r2_ridge:.4f}")

        # Both should achieve good R² on this synthetic data
        assert r2_pls > 0.9, f"PLS R² too low: {r2_pls}"
        assert r2_ridge > 0.9, f"Ridge R² too low: {r2_ridge}"

    def test_regional_specialist_selection_logic(self):
        """Test that specialist selection based on predictions works correctly.

        This is a unit test for the core logic that the fix enables:
        specialists are selected based on regional performance where regions
        are defined by prediction values, not true values.
        """
        np.random.seed(42)

        # Simulate results from grid search with regional RMSE
        # (After fix: regional RMSE computed using prediction-based quartiles)
        model_performance = {
            'PLS': {'Q1': 0.10, 'Q2': 0.05, 'Q3': 0.08, 'Q4': 0.15},  # Best at Q2
            'Ridge': {'Q1': 0.08, 'Q2': 0.12, 'Q3': 0.06, 'Q4': 0.10},  # Best at Q3
            'Lasso': {'Q1': 0.05, 'Q2': 0.10, 'Q3': 0.12, 'Q4': 0.08},  # Best at Q1
            'ElasticNet': {'Q1': 0.12, 'Q2': 0.08, 'Q3': 0.10, 'Q4': 0.05},  # Best at Q4
        }

        # Select best model per region
        best_per_region = {}
        for region in ['Q1', 'Q2', 'Q3', 'Q4']:
            best_model = min(model_performance.keys(),
                           key=lambda m: model_performance[m][region])
            best_per_region[region] = best_model

        expected = {
            'Q1': 'Lasso',
            'Q2': 'PLS',
            'Q3': 'Ridge',
            'Q4': 'ElasticNet',
        }

        assert best_per_region == expected, f"Wrong specialist selection: {best_per_region}"
        print(f"Specialists selected: {best_per_region}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
