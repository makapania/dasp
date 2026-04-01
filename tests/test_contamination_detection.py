"""Tests for one-class contamination detection models.

Tests the contamination.py module including:
- PCA-SIMCA one-class classifier
- One-class model factory functions
- One-class metrics computation
- One-class cross-validation
- Model registry and config integration
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from spectral_predict.contamination import (
    PCASIMCA,
    get_one_class_model,
    build_one_class_model,
    get_one_class_model_grids,
    one_class_metrics,
    one_class_cv,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_nir_data():
    """Create synthetic NIR-like data with clean and contaminated samples.

    Clean samples cluster tightly (simulating pure material).
    Contaminated samples are scattered (simulating multiple contaminants).
    """
    rng = np.random.RandomState(42)
    n_features = 100  # wavelengths

    # Clean samples: tight cluster
    n_clean = 80
    clean_center = rng.randn(n_features) * 0.5
    X_clean = clean_center + rng.randn(n_clean, n_features) * 0.3

    # Contaminated samples: multiple distant clusters
    n_contam = 20
    contam_centers = rng.randn(3, n_features) * 2.0 + 3.0
    X_contam = np.vstack([
        contam_centers[0] + rng.randn(7, n_features) * 0.5,
        contam_centers[1] + rng.randn(7, n_features) * 0.5,
        contam_centers[2] + rng.randn(6, n_features) * 0.5,
    ])

    X = np.vstack([X_clean, X_contam])
    y = np.array([1] * n_clean + [-1] * n_contam)

    return X, y, n_clean, n_contam


@pytest.fixture
def clean_only_data():
    """Create data with only clean samples (no outliers)."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 100) * 0.3
    y = np.ones(50, dtype=int)
    return X, y


# ============================================================================
# PCA-SIMCA Tests
# ============================================================================

class TestPCASIMCA:
    def test_fit_predict_basic(self, synthetic_nir_data):
        X, y, n_clean, n_contam = synthetic_nir_data
        model = PCASIMCA(n_components=5)
        model.fit(X[:n_clean])  # Train on clean only

        predictions = model.predict(X)
        assert predictions.shape == (len(X),)
        assert set(np.unique(predictions)).issubset({-1, 1})

    def test_decision_function(self, synthetic_nir_data):
        X, y, n_clean, _ = synthetic_nir_data
        model = PCASIMCA(n_components=5)
        model.fit(X[:n_clean])

        scores = model.decision_function(X)
        assert scores.shape == (len(X),)
        # Clean samples should generally have higher scores
        clean_mean = scores[:n_clean].mean()
        contam_mean = scores[n_clean:].mean()
        assert clean_mean > contam_mean

    def test_score_samples_alias(self, synthetic_nir_data):
        X, y, n_clean, _ = synthetic_nir_data
        model = PCASIMCA(n_components=5)
        model.fit(X[:n_clean])
        np.testing.assert_array_equal(
            model.score_samples(X),
            model.decision_function(X)
        )

    def test_variance_explained_components(self, synthetic_nir_data):
        X, y, n_clean, _ = synthetic_nir_data
        model = PCASIMCA(n_components=0.95)  # 95% variance
        model.fit(X[:n_clean])
        assert model.n_components_ >= 1
        assert model.n_components_ <= min(n_clean - 1, X.shape[1])

    def test_get_set_params(self):
        model = PCASIMCA(n_components=7, alpha=0.01)
        params = model.get_params()
        assert params['n_components'] == 7
        assert params['alpha'] == 0.01

        model.set_params(n_components=3)
        assert model.n_components == 3

    def test_single_component(self, synthetic_nir_data):
        X, y, n_clean, _ = synthetic_nir_data
        model = PCASIMCA(n_components=1)
        model.fit(X[:n_clean])
        predictions = model.predict(X)
        assert predictions.shape == (len(X),)


# ============================================================================
# Model Factory Tests
# ============================================================================

class TestModelFactory:
    @pytest.mark.parametrize("model_name", [
        'OneClassSVM', 'IsolationForest', 'EllipticEnvelope', 'LOF', 'PCA-SIMCA'
    ])
    def test_get_one_class_model(self, model_name):
        model = get_one_class_model(model_name)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown one-class model"):
            get_one_class_model("InvalidModel")

    @pytest.mark.parametrize("model_name", [
        'OneClassSVM', 'IsolationForest', 'EllipticEnvelope', 'LOF', 'PCA-SIMCA'
    ])
    def test_build_one_class_model(self, model_name, synthetic_nir_data):
        X, y, n_clean, _ = synthetic_nir_data
        grids = get_one_class_model_grids()
        params = grids[model_name][0]

        model = build_one_class_model(model_name, params)
        model.fit(X[:n_clean])
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_grids_have_all_models(self):
        grids = get_one_class_model_grids()
        expected = {'OneClassSVM', 'IsolationForest', 'EllipticEnvelope', 'LOF', 'PCA-SIMCA'}
        assert set(grids.keys()) == expected

    def test_grids_have_multiple_configs(self):
        grids = get_one_class_model_grids()
        for model_name, configs in grids.items():
            assert len(configs) >= 3, f"{model_name} should have at least 3 configs"


# ============================================================================
# Metrics Tests
# ============================================================================

class TestOneClassMetrics:
    def test_perfect_detection(self):
        y_true = np.array([1, 1, 1, -1, -1])
        y_pred = np.array([1, 1, 1, -1, -1])
        m = one_class_metrics(y_true, y_pred)
        assert m['sensitivity'] == 1.0
        assert m['specificity'] == 1.0
        assert m['accuracy'] == 1.0
        assert m['balanced_accuracy'] == 1.0

    def test_all_missed(self):
        y_true = np.array([1, 1, 1, -1, -1])
        y_pred = np.array([1, 1, 1, 1, 1])  # Missed all outliers
        m = one_class_metrics(y_true, y_pred)
        assert m['sensitivity'] == 0.0
        assert m['specificity'] == 1.0

    def test_all_false_alarms(self):
        y_true = np.array([1, 1, 1, -1, -1])
        y_pred = np.array([-1, -1, -1, -1, -1])  # Everything flagged
        m = one_class_metrics(y_true, y_pred)
        assert m['sensitivity'] == 1.0
        assert m['specificity'] == 0.0

    def test_no_outliers(self):
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        m = one_class_metrics(y_true, y_pred)
        assert np.isnan(m['sensitivity'])  # No outliers to detect
        assert m['specificity'] == 1.0

    def test_auc_with_scores(self):
        y_true = np.array([1, 1, 1, -1, -1])
        y_pred = np.array([1, 1, 1, -1, -1])
        scores = np.array([0.5, 0.3, 0.4, -0.5, -0.3])
        m = one_class_metrics(y_true, y_pred, scores=scores)
        assert m['auc'] > 0.5


# ============================================================================
# Cross-Validation Tests
# ============================================================================

class TestOneClassCV:
    def test_cv_basic(self, synthetic_nir_data):
        X, y, _, _ = synthetic_nir_data
        model = get_one_class_model('IsolationForest')
        results = one_class_cv(model, X, y, folds=3)

        assert 'sensitivity' in results
        assert 'specificity' in results
        assert 'balanced_accuracy' in results
        assert results['n_folds'] == 3
        assert 0 <= results['balanced_accuracy'] <= 1

    def test_cv_too_few_inliers(self):
        X = np.random.randn(3, 10)
        y = np.array([1, 1, -1])
        model = get_one_class_model('IsolationForest')

        with pytest.raises(ValueError, match="Not enough inlier samples"):
            one_class_cv(model, X, y, folds=5)

    def test_cv_all_models(self, synthetic_nir_data):
        X, y, _, _ = synthetic_nir_data
        for model_name in ['OneClassSVM', 'IsolationForest', 'PCA-SIMCA']:
            model = get_one_class_model(model_name)
            results = one_class_cv(model, X, y, folds=3)
            assert results['balanced_accuracy'] >= 0, f"{model_name} failed CV"


# ============================================================================
# Model Registry Integration Tests
# ============================================================================

class TestRegistryIntegration:
    def test_one_class_models_in_registry(self):
        from spectral_predict.model_registry import ONE_CLASS_MODELS, get_supported_models
        assert 'PCA-SIMCA' in ONE_CLASS_MODELS
        assert 'OneClassSVM' in ONE_CLASS_MODELS
        assert 'IsolationForest' in ONE_CLASS_MODELS

        supported = get_supported_models('one_class')
        assert len(supported) == len(ONE_CLASS_MODELS)

    def test_one_class_tiers(self):
        from spectral_predict.model_config import get_tier_models
        quick = get_tier_models('quick', 'one_class')
        standard = get_tier_models('standard', 'one_class')
        comprehensive = get_tier_models('comprehensive', 'one_class')

        assert len(quick) <= len(standard) <= len(comprehensive)
        assert 'PCA-SIMCA' in quick  # Always available

    def test_scoring_one_class_columns(self):
        from spectral_predict.scoring import create_results_dataframe
        df = create_results_dataframe('one_class')
        assert 'Sensitivitycv' in df.columns
        assert 'Specificitycv' in df.columns
        assert 'BalancedAcccv' in df.columns
        assert 'AUCcv' in df.columns


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_models_get_model_one_class(self):
        """Test that models.py get_model works for one_class task type."""
        from spectral_predict.models import get_model
        model = get_model('IsolationForest', task_type='one_class')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_models_build_model_one_class(self):
        """Test that models.py build_model works for one_class task type."""
        from spectral_predict.models import build_model
        model = build_model('PCA-SIMCA', {'n_components': 3, 'alpha': 0.05}, task_type='one_class')
        assert hasattr(model, 'fit')

    def test_get_model_grids_one_class(self):
        """Test that get_model_grids returns one-class grids."""
        from spectral_predict.models import get_model_grids
        grids = get_model_grids(
            task_type='one_class',
            n_features=100,
            enabled_models=['PCA-SIMCA', 'IsolationForest'],
        )
        assert 'PCA-SIMCA' in grids
        assert 'IsolationForest' in grids
        assert len(grids['PCA-SIMCA']) > 0
