"""Unit tests for Neural Boosted Regression."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.neural_boosted import NeuralBoostedRegressor


class TestNeuralBoostedBasic:
    """Basic functionality tests."""

    def test_fit_predict(self):
        """Test basic fit and predict."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(
            n_estimators=10,
            learning_rate=0.1,
            hidden_layer_size=3,
            random_state=42,
            verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert len(model.estimators_) <= 10
        r2 = r2_score(y, predictions)
        assert r2 > 0.7, f"R² = {r2:.3f}, expected > 0.7"

    def test_perfect_fit(self):
        """Test that model can fit simple linear relationship."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        # Simple linear relationship: y = 2*x1 + 3*x2
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1]

        model = NeuralBoostedRegressor(
            n_estimators=50,
            learning_rate=0.1,
            hidden_layer_size=5,
            activation='identity',  # Linear for linear problem
            random_state=42,
            early_stopping=False,
            verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        r2 = r2_score(y, predictions)
        assert r2 > 0.95, f"Should fit linear relationship well, got R² = {r2:.3f}"

    def test_nonlinear_fit(self):
        """Test that model can capture nonlinearity."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        # Nonlinear relationship: y = x1² + sin(x2)
        y = X[:, 0]**2 + np.sin(X[:, 1] * 3) + np.random.randn(200) * 0.1

        model = NeuralBoostedRegressor(
            n_estimators=50,
            learning_rate=0.1,
            hidden_layer_size=5,
            activation='tanh',
            random_state=42,
            verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        r2 = r2_score(y, predictions)
        assert r2 > 0.6, f"Should capture nonlinearity, got R² = {r2:.3f}"


class TestEarlyStopping:
    """Early stopping tests."""

    def test_early_stopping_triggers(self):
        """Test that early stopping works."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(
            n_estimators=100,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.2,
            random_state=42,
            verbose=0
        )

        model.fit(X, y)

        # Should stop early, not use all 100 estimators
        assert model.n_estimators_ < 100, \
            f"Expected early stopping, but used {model.n_estimators_} estimators"
        assert len(model.validation_score_) > 0, "Should have validation scores"

    def test_no_early_stopping(self):
        """Test that model uses all estimators when early_stopping=False."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        n_est = 20
        model = NeuralBoostedRegressor(
            n_estimators=n_est,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=False,
            random_state=42,
            verbose=0
        )

        model.fit(X, y)

        assert model.n_estimators_ == n_est, \
            f"Expected {n_est} estimators, got {model.n_estimators_}"
        assert len(model.validation_score_) == 0, \
            "Should not have validation scores when early_stopping=False"

    def test_validation_improves_generalization(self):
        """Test that early stopping helps generalization."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Model with early stopping
        model_es = NeuralBoostedRegressor(
            n_estimators=100,
            learning_rate=0.2,
            hidden_layer_size=5,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
            verbose=0
        )
        model_es.fit(X_train, y_train)
        pred_es = model_es.predict(X_test)
        r2_es = r2_score(y_test, pred_es)

        # Model without early stopping (more prone to overfit)
        model_no_es = NeuralBoostedRegressor(
            n_estimators=100,
            learning_rate=0.2,
            hidden_layer_size=5,
            early_stopping=False,
            random_state=42,
            verbose=0
        )
        model_no_es.fit(X_train, y_train)
        pred_no_es = model_no_es.predict(X_test)
        r2_no_es = r2_score(y_test, pred_no_es)

        # Early stopping should help (or at least not hurt much)
        assert r2_es > r2_no_es - 0.1, \
            f"Early stopping R² = {r2_es:.3f}, No ES R² = {r2_no_es:.3f}"


class TestFeatureImportances:
    """Feature importance extraction tests."""

    def test_feature_importances_shape(self):
        """Test feature importance extraction."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(n_estimators=10, random_state=42, verbose=0)
        model.fit(X, y)

        importances = model.get_feature_importances()

        assert importances.shape == (20,), \
            f"Expected shape (20,), got {importances.shape}"
        assert np.all(importances >= 0), "Importances should be non-negative"
        assert np.sum(importances) > 0, "Importances should sum to positive value"

    def test_important_features_detected(self):
        """Test that model identifies truly important features."""
        np.random.seed(42)
        n_samples = 200
        n_features = 50

        # Create data where only features 5, 10, 15 matter
        X = np.random.randn(n_samples, n_features) * 0.1
        y = 5.0 * X[:, 5] + 3.0 * X[:, 10] - 2.0 * X[:, 15] + np.random.randn(n_samples) * 0.1

        model = NeuralBoostedRegressor(
            n_estimators=30,
            learning_rate=0.1,
            hidden_layer_size=5,
            random_state=42,
            verbose=0
        )
        model.fit(X, y)

        importances = model.get_feature_importances()
        top_5_indices = np.argsort(importances)[-5:][::-1]

        # At least 2 of the true important features should be in top 5
        important_features = {5, 10, 15}
        detected = important_features.intersection(set(top_5_indices))
        assert len(detected) >= 2, \
            f"Expected to detect 2+ important features, found {detected} in top 5: {top_5_indices}"


class TestLossFunctions:
    """Loss function tests."""

    def test_mse_loss(self):
        """Test MSE loss computation."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(
            loss='mse',
            n_estimators=20,
            random_state=42,
            verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        # Check that training loss decreases
        assert len(model.train_score_) > 0
        assert model.train_score_[0] > model.train_score_[-1], \
            "Training loss should decrease"

    def test_huber_loss(self):
        """Test Huber loss with outliers."""
        np.random.seed(42)
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        # Add outliers
        outlier_indices = [0, 1, 2]
        y[outlier_indices] = y[outlier_indices] + np.random.choice([-10, 10], size=len(outlier_indices))

        model_mse = NeuralBoostedRegressor(
            loss='mse',
            n_estimators=30,
            random_state=42,
            verbose=0
        )
        model_huber = NeuralBoostedRegressor(
            loss='huber',
            huber_delta=1.35,
            n_estimators=30,
            random_state=42,
            verbose=0
        )

        model_mse.fit(X, y)
        model_huber.fit(X, y)

        # Both should fit reasonably
        pred_mse = model_mse.predict(X)
        pred_huber = model_huber.predict(X)

        # Remove outliers for comparison
        mask = np.ones(len(y), dtype=bool)
        mask[outlier_indices] = False
        X_clean = X[mask]
        y_clean = y[mask]

        r2_mse = r2_score(y_clean, model_mse.predict(X_clean))
        r2_huber = r2_score(y_clean, model_huber.predict(X_clean))

        # Huber should be at least competitive on clean data
        assert r2_huber > 0.5, f"Huber R² = {r2_huber:.3f}"


class TestActivationFunctions:
    """Test different activation functions."""

    def test_tanh_activation(self):
        """Test tanh activation."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(
            activation='tanh',
            n_estimators=10,
            random_state=42,
            verbose=0
        )
        model.fit(X, y)
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        assert r2 > 0.5, f"tanh: R² = {r2:.3f}"

    def test_relu_activation(self):
        """Test ReLU activation."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(
            activation='relu',
            n_estimators=10,
            random_state=42,
            verbose=0
        )
        model.fit(X, y)
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        assert r2 > 0.5, f"relu: R² = {r2:.3f}"

    def test_identity_activation(self):
        """Test identity (linear) activation."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(
            activation='identity',
            n_estimators=10,
            random_state=42,
            verbose=0
        )
        model.fit(X, y)
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        assert r2 > 0.5, f"identity: R² = {r2:.3f}"

    def test_logistic_activation(self):
        """Test logistic (sigmoid) activation."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

        model = NeuralBoostedRegressor(
            activation='logistic',
            n_estimators=10,
            random_state=42,
            verbose=0
        )
        model.fit(X, y)
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        assert r2 > 0.4, f"logistic: R² = {r2:.3f}"


class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_learning_rate(self):
        """Test learning rate validation."""
        with pytest.raises(ValueError, match="learning_rate must be in"):
            NeuralBoostedRegressor(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be in"):
            NeuralBoostedRegressor(learning_rate=1.5)

    def test_invalid_hidden_layer_size(self):
        """Test hidden layer size validation."""
        with pytest.raises(ValueError, match="hidden_layer_size must be"):
            NeuralBoostedRegressor(hidden_layer_size=0)

    def test_large_hidden_layer_warning(self):
        """Test warning for large hidden layer."""
        with pytest.warns(UserWarning, match="large for weak learner"):
            NeuralBoostedRegressor(hidden_layer_size=15)

    def test_invalid_activation(self):
        """Test activation validation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            NeuralBoostedRegressor(activation='unknown')

    def test_invalid_loss(self):
        """Test loss validation."""
        with pytest.raises(ValueError, match="Unknown loss"):
            NeuralBoostedRegressor(loss='unknown')

    def test_invalid_validation_fraction(self):
        """Test validation fraction validation."""
        with pytest.raises(ValueError, match="validation_fraction must be"):
            NeuralBoostedRegressor(validation_fraction=0)

        with pytest.raises(ValueError, match="validation_fraction must be"):
            NeuralBoostedRegressor(validation_fraction=1.5)


class TestSpectralDataIntegration:
    """Test with spectral-like data."""

    def test_spectral_data_structure(self):
        """Test with synthetic spectral data."""
        np.random.seed(42)
        n_samples = 80
        n_wavelengths = 500

        # Simulate NIR spectrum (baseline around 1.0 with noise)
        X = np.random.randn(n_samples, n_wavelengths) * 0.05 + 1.0

        # Target based on specific wavelengths (simulate absorption peaks)
        # Simulate O-H peak at wavelength 100, C-H at 200, N-H at 300
        y = (2.0 * X[:, 100] + 1.5 * X[:, 200] - 0.8 * X[:, 300] +
             np.random.randn(n_samples) * 0.05)

        model = NeuralBoostedRegressor(
            n_estimators=30,
            learning_rate=0.1,
            hidden_layer_size=5,
            early_stopping=True,
            random_state=42,
            verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        # Check model performance
        r2 = r2_score(y, predictions)
        assert r2 > 0.7, f"R² = {r2:.3f}, expected > 0.7 for spectral-like data"

        # Check that it identifies important wavelengths
        importances = model.get_feature_importances()
        top_10_indices = np.argsort(importances)[-10:][::-1]

        # At least one of the true important wavelengths should be in top 10
        important_wavelengths = {100, 200, 300}
        detected = important_wavelengths.intersection(set(top_10_indices))
        assert len(detected) >= 1, \
            f"Expected to detect important wavelengths, found {detected} in top 10"

    def test_high_dimensional_spectral(self):
        """Test with high-dimensional spectral data (2000+ wavelengths)."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 2000

        X = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0
        # Only first 5 wavelengths matter
        y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

        model = NeuralBoostedRegressor(
            n_estimators=20,
            learning_rate=0.1,
            hidden_layer_size=3,
            early_stopping=True,
            random_state=42,
            verbose=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        r2 = r2_score(y, predictions)
        assert r2 > 0.5, f"R² = {r2:.3f}, should handle high-dimensional data"


class TestSklearnCompatibility:
    """Test sklearn compatibility."""

    def test_get_set_params(self):
        """Test get_params and set_params."""
        model = NeuralBoostedRegressor(
            n_estimators=50,
            learning_rate=0.1
        )

        params = model.get_params()
        assert params['n_estimators'] == 50
        assert params['learning_rate'] == 0.1

        model.set_params(n_estimators=100, learning_rate=0.2)
        assert model.n_estimators == 100
        assert model.learning_rate == 0.2

    def test_clone_compatibility(self):
        """Test that model can be cloned."""
        from sklearn.base import clone

        model = NeuralBoostedRegressor(
            n_estimators=50,
            learning_rate=0.1,
            hidden_layer_size=3
        )

        cloned = clone(model)
        assert cloned.n_estimators == 50
        assert cloned.learning_rate == 0.1
        assert cloned.hidden_layer_size == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
