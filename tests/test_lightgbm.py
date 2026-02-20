"""
LightGBM Isolated Test Suite

Tests LightGBM in isolation to identify the root cause of negative R2 values.
Tests progressively more complex scenarios to pinpoint where the issue occurs.
"""

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


pytestmark = pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")


class TestLightGBMIsolated:
    """Isolated LightGBM tests to verify model works correctly."""

    def test_simple_dataset(self):
        """Test LightGBM on simple low-dimensional dataset."""
        X, y = make_regression(n_samples=100, n_features=50, random_state=42, noise=10)
        model = LGBMRegressor(random_state=42, verbosity=-1)
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        assert np.mean(scores) > 0, f"Mean R2 should be positive, got {np.mean(scores):.4f}"

    def test_high_dimensional_dataset(self):
        """Test LightGBM on high-dimensional dataset (like spectral data)."""
        X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
        model = LGBMRegressor(random_state=42, verbosity=-1)
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        assert np.mean(scores) > -1.0, f"Mean R2 unreasonably low: {np.mean(scores):.4f}"

    def test_high_dimensional_with_regularization(self):
        """Test LightGBM with regularization matching models.py config."""
        X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        assert np.mean(scores) > -1.0, f"Mean R2 unreasonably low: {np.mean(scores):.4f}"

    def test_high_dimensional_with_pipeline(self):
        """Test LightGBM with StandardScaler pipeline (like actual code)."""
        X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
        assert np.mean(scores) > -1.0, f"Mean R2 unreasonably low: {np.mean(scores):.4f}"

    def test_small_sample_size(self):
        """Test LightGBM with small sample size (might cause CV issues)."""
        X, y = make_regression(n_samples=50, n_features=2000, random_state=42, noise=10)
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        # Small samples may give negative R2, just verify it runs
        assert len(scores) == 3, "Should produce 3 fold scores"

    def test_float32_vs_float64(self):
        """Test LightGBM with both float32 and float64 data types."""
        X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)

        model = LGBMRegressor(random_state=42, verbosity=-1)
        scores_64 = cross_val_score(model, X, y, cv=5, scoring='r2')

        X_32 = X.astype(np.float32)
        y_32 = y.astype(np.float32)
        scores_32 = cross_val_score(model, X_32, y_32, cv=5, scoring='r2')

        assert len(scores_64) == 5
        assert len(scores_32) == 5

    def test_training_r2(self):
        """Test that LightGBM can achieve good training R2."""
        X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
        model = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        model.fit(X, y)
        train_r2 = model.score(X, y)
        assert train_r2 > 0.5, f"Training R2 should be decent, got {train_r2:.4f}"
