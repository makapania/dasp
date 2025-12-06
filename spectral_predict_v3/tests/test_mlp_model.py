"""
Tests for MLP (Multi-Layer Perceptron) model in Spectral Predict v3.

Tests cover:
- MLP regression on synthetic data
- MLP classification on synthetic data
- Hyperparameter grid testing
- Convergence warnings handling
- Model save/load (via pickle)
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pickle
import tempfile
from pathlib import Path

from spectral_predict_v3.core.models import get_model
from spectral_predict_v3.core.model_config import get_hyperparameter_grid


class TestMLPRegression:
    """Test MLP for regression tasks."""

    def test_mlp_basic_regression(self):
        """Test basic MLP regression on synthetic data."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            noise=0.1,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create and train MLP model
        model = get_model('MLP', task_type='regression', random_state=42)
        assert model is not None, "MLP model should be created"

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # MLP should achieve reasonable R² on this simple problem
        assert r2 > 0.5, f"R² should be > 0.5, got {r2:.4f}"

        print(f"✓ MLP regression R² = {r2:.4f}")

    def test_mlp_hyperparameters(self):
        """Test MLP with different hyperparameters."""
        X, y = make_regression(
            n_samples=80,
            n_features=30,
            n_informative=5,
            noise=0.2,
            random_state=42
        )

        # Test different hidden layer configurations
        configs = [
            {'hidden_layer_sizes': (50,), 'activation': 'relu'},
            {'hidden_layer_sizes': (100,), 'activation': 'relu'},
            {'hidden_layer_sizes': (50, 50), 'activation': 'tanh'},
        ]

        for config in configs:
            model = get_model('MLP', task_type='regression', random_state=42, **config)
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            assert r2 > 0.3, f"R² should be > 0.3 for config {config}, got {r2:.4f}"
            print(f"✓ Config {config}: R² = {r2:.4f}")

    def test_mlp_hyperparameter_grid(self):
        """Test that hyperparameter grid is correctly defined."""
        grid = get_hyperparameter_grid('MLP')

        assert len(grid) > 0, "MLP should have hyperparameter grid"
        assert all('hidden_layer_sizes' in params for params in grid), \
            "All grid entries should have hidden_layer_sizes"
        assert all('activation' in params for params in grid), \
            "All grid entries should have activation"
        assert all('alpha' in params for params in grid), \
            "All grid entries should have alpha"

        print(f"✓ MLP hyperparameter grid has {len(grid)} configurations")

    def test_mlp_early_stopping(self):
        """Test that early stopping works."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            noise=0.1,
            random_state=42
        )

        model = get_model(
            'MLP',
            task_type='regression',
            random_state=42,
            max_iter=5000,
            early_stopping=True,
            n_iter_no_change=10
        )

        model.fit(X, y)

        # Early stopping should prevent full iterations
        assert model.n_iter_ < 5000, \
            f"Early stopping should stop before max_iter, got {model.n_iter_} iterations"

        print(f"✓ Early stopping worked: converged in {model.n_iter_} iterations")


class TestMLPClassification:
    """Test MLP for classification tasks."""

    def test_mlp_basic_classification(self):
        """Test basic MLP classification on synthetic data."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Create and train MLP classifier
        model = get_model('MLP', task_type='classification', random_state=42)
        assert model is not None, "MLP classifier should be created"

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # MLP should achieve reasonable accuracy on this simple problem
        assert accuracy > 0.6, f"Accuracy should be > 0.6, got {accuracy:.4f}"

        print(f"✓ MLP classification accuracy = {accuracy:.4f}")

    def test_mlp_binary_classification(self):
        """Test MLP on binary classification."""
        X, y = make_classification(
            n_samples=150,
            n_features=15,
            n_informative=8,
            n_classes=2,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = get_model(
            'MLP',
            task_type='classification',
            hidden_layer_sizes=(50,),
            activation='relu',
            alpha=0.001,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Binary classification should work well
        assert accuracy > 0.7, f"Binary classification accuracy should be > 0.7, got {accuracy:.4f}"

        print(f"✓ Binary classification accuracy = {accuracy:.4f}")

    def test_mlp_multiclass_classification(self):
        """Test MLP on multiclass classification."""
        X, y = make_classification(
            n_samples=200,
            n_features=25,
            n_informative=15,
            n_classes=5,
            random_state=42
        )

        model = get_model(
            'MLP',
            task_type='classification',
            hidden_layer_sizes=(100, 50),
            activation='relu',
            random_state=42
        )

        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        # Multiclass should work
        assert accuracy > 0.4, f"Multiclass accuracy should be > 0.4, got {accuracy:.4f}"

        print(f"✓ Multiclass (5 classes) accuracy = {accuracy:.4f}")


class TestMLPPersistence:
    """Test MLP model save/load."""

    def test_mlp_save_load_regression(self):
        """Test saving and loading MLP regression model."""
        X, y = make_regression(n_samples=50, n_features=10, random_state=42)

        # Train model
        model = get_model('MLP', task_type='regression', random_state=42)
        model.fit(X, y)
        y_pred_original = model.predict(X)

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            model_path = f.name

        try:
            # Load model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            # Test predictions match
            y_pred_loaded = loaded_model.predict(X)
            np.testing.assert_array_almost_equal(
                y_pred_original,
                y_pred_loaded,
                decimal=5,
                err_msg="Loaded model predictions should match original"
            )

            print("✓ MLP regression model save/load works")

        finally:
            # Cleanup
            Path(model_path).unlink(missing_ok=True)

    def test_mlp_save_load_classification(self):
        """Test saving and loading MLP classification model."""
        X, y = make_classification(n_samples=100, n_features=15, n_classes=3, random_state=42)

        # Train model
        model = get_model('MLP', task_type='classification', random_state=42)
        model.fit(X, y)
        y_pred_original = model.predict(X)

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            model_path = f.name

        try:
            # Load model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            # Test predictions match
            y_pred_loaded = loaded_model.predict(X)
            np.testing.assert_array_equal(
                y_pred_original,
                y_pred_loaded,
                err_msg="Loaded model predictions should match original"
            )

            print("✓ MLP classification model save/load works")

        finally:
            # Cleanup
            Path(model_path).unlink(missing_ok=True)


class TestMLPEdgeCases:
    """Test MLP edge cases and error handling."""

    def test_mlp_small_dataset(self):
        """Test MLP on very small dataset."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        model = get_model(
            'MLP',
            task_type='regression',
            hidden_layer_sizes=(10,),  # Small network
            max_iter=500,
            random_state=42
        )

        # Should train without error even on small dataset
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape, "Predictions should have correct shape"

        print("✓ MLP handles small dataset")

    def test_mlp_convergence_warning(self):
        """Test that MLP handles convergence warnings gracefully."""
        X, y = make_regression(n_samples=100, n_features=50, random_state=42)

        # Use very few iterations to trigger convergence warning
        model = get_model(
            'MLP',
            task_type='regression',
            max_iter=10,  # Very low
            random_state=42
        )

        # Should train but may not converge
        with pytest.warns(None):  # May or may not warn
            model.fit(X, y)

        # Should still produce predictions
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape, "Should produce predictions even without convergence"

        print("✓ MLP handles convergence warnings")

    def test_mlp_different_alphas(self):
        """Test MLP with different regularization strengths."""
        X, y = make_regression(n_samples=80, n_features=20, random_state=42)

        alphas = [0.0001, 0.001, 0.01, 0.1]

        for alpha in alphas:
            model = get_model(
                'MLP',
                task_type='regression',
                alpha=alpha,
                random_state=42
            )

            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            # All should train successfully
            assert not np.isnan(r2), f"R² should not be NaN for alpha={alpha}"

            print(f"✓ Alpha={alpha}: R² = {r2:.4f}")


def test_mlp_in_model_tiers():
    """Test that MLP is correctly included in model tiers."""
    from spectral_predict_v3.core.model_config import get_tier_models

    # MLP should be in comprehensive tier for both regression and classification
    reg_models = get_tier_models('comprehensive', 'regression')
    class_models = get_tier_models('comprehensive', 'classification')

    assert 'MLP' in reg_models, "MLP should be in comprehensive regression tier"
    assert 'MLP' in class_models, "MLP should be in comprehensive classification tier"

    print("✓ MLP correctly included in model tiers")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing MLP Model")
    print("=" * 60)

    # Regression tests
    print("\n--- Regression Tests ---")
    test_reg = TestMLPRegression()
    test_reg.test_mlp_basic_regression()
    test_reg.test_mlp_hyperparameters()
    test_reg.test_mlp_hyperparameter_grid()
    test_reg.test_mlp_early_stopping()

    # Classification tests
    print("\n--- Classification Tests ---")
    test_class = TestMLPClassification()
    test_class.test_mlp_basic_classification()
    test_class.test_mlp_binary_classification()
    test_class.test_mlp_multiclass_classification()

    # Persistence tests
    print("\n--- Persistence Tests ---")
    test_persist = TestMLPPersistence()
    test_persist.test_mlp_save_load_regression()
    test_persist.test_mlp_save_load_classification()

    # Edge case tests
    print("\n--- Edge Case Tests ---")
    test_edge = TestMLPEdgeCases()
    test_edge.test_mlp_small_dataset()
    test_edge.test_mlp_convergence_warning()
    test_edge.test_mlp_different_alphas()

    # Tier test
    print("\n--- Configuration Tests ---")
    test_mlp_in_model_tiers()

    print("\n" + "=" * 60)
    print("All MLP tests passed!")
    print("=" * 60)
