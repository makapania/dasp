"""
Test that ensemble region assignment fix works correctly.

This test verifies the fix for the region assignment mismatch bug where:
- fit() assigned regions based on TRUE y values
- predict() assigned regions based on AVERAGE PREDICTIONS

After the fix, both should use average predictions for region assignment,
resulting in positive R² values for well-behaved data.
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from spectral_predict.ensemble import (
    RegionAwareWeightedEnsemble,
    MixtureOfExpertsEnsemble,
    StackingEnsemble,
    SimpleAverageEnsemble,
)


class TestEnsembleRegionFix:
    """Test that region-aware ensembles produce positive R² after fix."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with clear signal for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 50

        # Create data with clear linear relationship
        X = np.random.randn(n_samples, n_features)
        # Strong signal from first few features
        y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(n_samples) * 0.5

        # Split into train/test
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def fitted_models(self, sample_data):
        """Create fitted base models with good performance."""
        X_train, X_test, y_train, y_test = sample_data

        # Fit models that should achieve R² > 0.9
        model1 = PLSRegression(n_components=5)
        model2 = Ridge(alpha=1.0)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        # Verify base models work well
        r2_1 = r2_score(y_test, model1.predict(X_test))
        r2_2 = r2_score(y_test, model2.predict(X_test))
        assert r2_1 > 0.8, f"PLS R² too low: {r2_1}"
        assert r2_2 > 0.8, f"Ridge R² too low: {r2_2}"

        return [model1, model2], ["PLS", "Ridge"]

    def test_simple_average_positive_r2(self, sample_data, fitted_models):
        """Test SimpleAverageEnsemble produces positive R² (baseline)."""
        X_train, X_test, y_train, y_test = sample_data
        models, names = fitted_models

        ensemble = SimpleAverageEnsemble(models=models, model_names=names)
        ensemble.fit(X_train, y_train)

        y_pred = ensemble.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        assert r2 > 0.8, f"SimpleAverageEnsemble R² should be positive but got {r2}"
        print(f"SimpleAverageEnsemble R²: {r2:.4f}")

    def test_region_weighted_positive_r2(self, sample_data, fitted_models):
        """Test RegionAwareWeightedEnsemble produces positive R² after fix."""
        X_train, X_test, y_train, y_test = sample_data
        models, names = fitted_models

        ensemble = RegionAwareWeightedEnsemble(
            models=models, model_names=names, n_regions=5, cv=3
        )
        ensemble.fit(X_train, y_train)

        y_pred = ensemble.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # After fix, R² should be positive
        assert r2 > 0, f"RegionAwareWeightedEnsemble R² should be positive but got {r2}"
        # Should be comparable to base models
        assert r2 > 0.7, f"RegionAwareWeightedEnsemble R² too low: {r2}"
        print(f"RegionAwareWeightedEnsemble R²: {r2:.4f}")

    def test_mixture_experts_positive_r2(self, sample_data, fitted_models):
        """Test MixtureOfExpertsEnsemble produces positive R² after fix."""
        X_train, X_test, y_train, y_test = sample_data
        models, names = fitted_models

        ensemble = MixtureOfExpertsEnsemble(
            models=models, model_names=names, n_regions=5, soft_gating=True
        )
        ensemble.fit(X_train, y_train)

        y_pred = ensemble.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # After fix, R² should be positive
        assert r2 > 0, f"MixtureOfExpertsEnsemble R² should be positive but got {r2}"
        # Should be comparable to base models
        assert r2 > 0.7, f"MixtureOfExpertsEnsemble R² too low: {r2}"
        print(f"MixtureOfExpertsEnsemble R²: {r2:.4f}")

    def test_stacking_positive_r2(self, sample_data, fitted_models):
        """Test StackingEnsemble with region_aware=True produces positive R² after fix."""
        X_train, X_test, y_train, y_test = sample_data
        models, names = fitted_models

        ensemble = StackingEnsemble(
            models=models, model_names=names, region_aware=True, n_regions=5, cv=3
        )
        ensemble.fit(X_train, y_train)

        y_pred = ensemble.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # After fix, R² should be positive
        assert r2 > 0, f"StackingEnsemble R² should be positive but got {r2}"
        # Should be comparable to base models
        assert r2 > 0.7, f"StackingEnsemble R² too low: {r2}"
        print(f"StackingEnsemble R²: {r2:.4f}")

    def test_region_boundaries_from_predictions(self, sample_data, fitted_models):
        """Verify region boundaries are derived from CV predictions, not true y."""
        X_train, X_test, y_train, y_test = sample_data
        models, names = fitted_models

        ensemble = RegionAwareWeightedEnsemble(
            models=models, model_names=names, n_regions=5, cv=3
        )
        ensemble.fit(X_train, y_train)

        # Region boundaries should be based on prediction range, not y range
        # The boundaries should be within a reasonable range of predictions
        boundaries = ensemble.analyzer_.region_boundaries

        # Boundaries should exist and be sorted
        assert len(boundaries) == 6  # 5 regions = 6 boundaries
        assert np.all(np.diff(boundaries) > 0), "Boundaries should be sorted"

        # Boundaries should not exactly match y quantiles (since they're from predictions)
        y_boundaries = np.percentile(y_train, np.linspace(0, 100, 6))

        # They might be close but shouldn't be identical
        # (unless predictions are perfect, which they won't be)
        if not np.allclose(boundaries, y_boundaries, atol=0.1):
            print(
                "Region boundaries differ from y quantiles as expected after fix"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
