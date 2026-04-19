"""Tests for one-class detection models.

Tests the contamination.py module including:
- PCA-SIMCA one-class classifier (DD-SIMCA)
- One-class model factory functions
- One-class metrics computation
- Model registry and config integration
- Composite scoring for one_class task type
- Model save/load roundtrip
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from spectral_predict.contamination import (
    PCASIMCA,
    get_one_class_model,
    build_one_class_model,
    get_one_class_model_grids,
    one_class_metrics,
    run_one_class_cv,
)
from spectral_predict.scoring import compute_composite_score
from spectral_predict.model_io import save_model, load_model


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

    def test_get_model_grids_one_class_default_tier(self):
        """Test that get_model_grids returns one-class grids with default tier (no enabled_models)."""
        from spectral_predict.models import get_model_grids
        grids = get_model_grids(
            task_type='one_class',
            n_features=100,
        )
        assert len(grids) > 0, "Default tier should return one-class models"
        assert any(name in grids for name in ['PCA-SIMCA', 'IsolationForest', 'OneClassSVM'])

    def test_run_one_class_cv_clean_only(self, clean_only_data):
        """Test that run_one_class_cv handles clean-only data (no outliers)."""
        X, y = clean_only_data
        result = run_one_class_cv(X, y, 'PCA-SIMCA', {'n_components': 3, 'alpha': 0.05}, n_folds=3)
        assert 'mean_metrics' in result
        # With no outliers, sensitivity is NaN but specificity should be valid
        assert not np.isnan(result['mean_metrics'].get('specificity', np.nan))



# ============================================================================
# Search Pipeline Integration Tests
# ============================================================================

class TestSearchIntegration:
    def test_run_one_class_search(self):
        """Test the full run_one_class_search pipeline."""
        import pandas as pd
        from spectral_predict.search import run_one_class_search

        rng = np.random.RandomState(42)
        n_features = 50

        # Create synthetic spectral data
        clean_center = rng.randn(n_features) * 0.5
        X_clean = clean_center + rng.randn(40, n_features) * 0.3
        X_contam = rng.randn(10, n_features) * 2.0 + 3.0
        X = np.vstack([X_clean, X_contam])
        y = np.array(['clean'] * 40 + ['contaminated'] * 10)

        # Create DataFrame (run_one_class_search expects this)
        wavelengths = [f"{400 + i * 10:.1f}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=wavelengths)
        y_series = pd.Series(y)

        results = run_one_class_search(
            X=X_df,
            y=y_series,
            inlier_class_label='clean',
            folds=3,
            preprocessing_methods=['raw'],
            window_sizes=[17],
            enabled_models=['IsolationForest', 'PCA-SIMCA'],
        )

        assert len(results) > 0
        assert 'Rank' in results.columns
        assert 'Sensitivitycv' in results.columns
        assert 'Specificitycv' in results.columns
        assert 'BalancedAcccv' in results.columns
        assert 'CompositeScore' in results.columns

        # Best model should have reasonable performance
        best = results.iloc[0]
        assert best['BalancedAcccv'] >= 0.8

    def test_run_one_class_search_per_contaminant(self):
        """Test per-contaminant sensitivity reporting with multiple contaminant types."""
        import pandas as pd
        from spectral_predict.search import run_one_class_search

        rng = np.random.RandomState(42)
        n_features = 30

        # Clean samples
        X_clean = rng.randn(30, n_features) * 0.3
        # Two different contaminant types (spectrally distinct)
        X_contam_a = rng.randn(5, n_features) + 3.0
        X_contam_b = rng.randn(5, n_features) - 3.0
        X = np.vstack([X_clean, X_contam_a, X_contam_b])
        y = np.array(['clean'] * 30 + ['contam_A'] * 5 + ['contam_B'] * 5)

        wavelengths = [f"{400 + i * 10:.1f}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=wavelengths)
        y_series = pd.Series(y)

        results = run_one_class_search(
            X=X_df,
            y=y_series,
            inlier_class_label='clean',
            folds=3,
            preprocessing_methods=['raw'],
            window_sizes=[17],
            enabled_models=['IsolationForest'],
        )

        assert len(results) > 0
        # Should have per-contaminant columns
        has_per_contam = any(c.startswith('Cal_Sens_') for c in results.columns)
        assert has_per_contam, "Should have per-contaminant sensitivity columns"

    def test_run_one_class_search_with_preprocessing(self):
        """Test search pipeline with SNV and first-derivative preprocessing (6g)."""
        from spectral_predict.search import run_one_class_search

        rng = np.random.RandomState(42)
        n_features = 50

        clean_center = rng.randn(n_features) * 0.5
        X_clean = clean_center + rng.randn(40, n_features) * 0.3
        X_contam = rng.randn(10, n_features) * 2.0 + 3.0
        X = np.vstack([X_clean, X_contam])
        y = np.array(['clean'] * 40 + ['contaminated'] * 10)

        wavelengths = [f"{400 + i * 10:.1f}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=wavelengths)
        y_series = pd.Series(y)

        # run_one_class_search accepts high-level method names like 'snv', 'deriv1'
        # which it maps internally to build_preprocessing_pipeline-compatible names
        results = run_one_class_search(
            X=X_df,
            y=y_series,
            inlier_class_label='clean',
            folds=3,
            preprocessing_methods=['snv', 'deriv1'],
            window_sizes=[17],
            enabled_models=['IsolationForest', 'PCA-SIMCA'],
        )

        assert len(results) > 0
        assert 'Rank' in results.columns
        # Verify both preprocessing paths were evaluated
        unique_preprocess = results['Preprocess'].unique()
        assert len(unique_preprocess) >= 2, (
            f"Expected at least 2 preprocessing methods, got {list(unique_preprocess)}"
        )
        assert 'CompositeScore' in results.columns


# ============================================================================
# Composite Scoring Tests
# ============================================================================

class TestCompositeScoring:
    def test_compute_composite_score_one_class(self):
        """Verify composite scoring assigns Rank and non-NaN CompositeScore (6a)."""
        df = pd.DataFrame({
            'Model': ['IsolationForest', 'PCA-SIMCA'],
            'Preprocess': ['raw', 'snv'],
            'Deriv': [0, 0],
            'Window': [0, 0],
            'Poly': [0, 0],
            'LVs': [0, 0],
            'n_vars': [100, 100],
            'full_vars': [100, 100],
            'SubsetTag': ['', ''],
            'Imbalance': [0.2, 0.2],
            'Task': ['one_class', 'one_class'],
            'Params': ['{}', '{}'],
            'Sensitivity': [0.90, 0.85],
            'Specificity': [0.95, 0.92],
            'Precision': [0.80, 0.75],
            'F1': [0.85, 0.80],
            'Accuracy': [0.94, 0.91],
            'BalancedAcc': [0.925, 0.885],
            'AUC': [0.95, 0.90],
            'Sensitivitycv': [0.85, 0.80],
            'Specificitycv': [0.92, 0.90],
            'Precisioncv': [0.78, 0.72],
            'F1cv': [0.82, 0.76],
            'Accuracycv': [0.91, 0.88],
            'BalancedAcccv': [0.885, 0.850],
            'AUCcv': [0.92, 0.87],
            'top_vars': ['', ''],
            'all_vars': ['', ''],
        })

        result = compute_composite_score(df, 'one_class')
        assert 'Rank' in result.columns
        assert 'CompositeScore' in result.columns
        assert result['CompositeScore'].notna().all()
        assert result['Rank'].min() == 1

    def test_compute_composite_score_zero_outliers(self):
        """Verify scoring handles NaN BalancedAcccv without crashing (6i)."""
        df = pd.DataFrame({
            'Model': ['IsolationForest'],
            'Preprocess': ['raw'],
            'Deriv': [0],
            'Window': [0],
            'Poly': [0],
            'LVs': [0],
            'n_vars': [50],
            'full_vars': [50],
            'SubsetTag': [''],
            'Imbalance': [0.0],
            'Task': ['one_class'],
            'Params': ['{}'],
            'Sensitivity': [np.nan],
            'Specificity': [0.95],
            'Precision': [np.nan],
            'F1': [np.nan],
            'Accuracy': [0.95],
            'BalancedAcc': [np.nan],
            'AUC': [np.nan],
            'Sensitivitycv': [np.nan],
            'Specificitycv': [0.92],
            'Precisioncv': [np.nan],
            'F1cv': [np.nan],
            'Accuracycv': [0.92],
            'BalancedAcccv': [np.nan],
            'AUCcv': [np.nan],
            'top_vars': [''],
            'all_vars': [''],
        })

        result = compute_composite_score(df, 'one_class')
        assert 'Rank' in result.columns
        assert result['Rank'].iloc[0] == 1
        assert result['CompositeScore'].notna().all()
        assert np.isfinite(result['CompositeScore'].values).all()


# ============================================================================
# DD-SIMCA Statistical Properties Tests
# ============================================================================

class TestDDSIMCAProperties:
    def test_inlier_false_reject_rate(self):
        """DD-SIMCA with alpha=0.05 should classify >= 85% of clean data as inlier (6c).

        Uses simple isotropic data (200 samples, 20 features) where the
        chi-squared distribution fitting is well-conditioned.
        """
        rng = np.random.RandomState(42)
        X_clean = rng.randn(200, 20) * 0.3

        model = PCASIMCA(n_components=2, alpha=0.05)
        model.fit(X_clean)

        predictions = model.predict(X_clean)
        inlier_rate = np.mean(predictions == 1)
        assert inlier_rate >= 0.85, (
            f"Expected >= 85% inlier rate on training clean data, got {inlier_rate:.1%}"
        )


# ============================================================================
# AUC Sign Convention Test
# ============================================================================

class TestAUCSignConvention:
    def test_isolation_forest_auc_sign(self, synthetic_nir_data):
        """Verify AUC > 0.8 using negated decision_function scores (6d)."""
        X, y, n_clean, _ = synthetic_nir_data
        X_clean = X[:n_clean]

        model = IsolationForest(n_estimators=100, random_state=42)
        model.fit(X_clean)

        scores = model.decision_function(X)
        # IsolationForest: positive scores = inlier, negative = outlier
        # For AUC with y_true where 1=inlier, -1=outlier: negate scores so
        # outliers get higher values
        y_bin = (y == -1).astype(int)  # 1 = outlier, 0 = inlier
        auc = roc_auc_score(y_bin, -scores)
        assert auc > 0.8, f"Expected AUC > 0.8 for well-separated data, got {auc:.3f}"


# ============================================================================
# Model Save/Load Roundtrip Test
# ============================================================================

class TestModelSaveLoad:
    def test_pcasimca_save_load_roundtrip(self, synthetic_nir_data):
        """Save a fitted PCASIMCA model, load it back, verify predictions match (6h)."""
        X, y, n_clean, _ = synthetic_nir_data
        X_clean = X[:n_clean]

        model = PCASIMCA(n_components=5, alpha=0.05)
        model.fit(X_clean)
        original_preds = model.predict(X)
        original_scores = model.decision_function(X)

        wavelengths = [float(i) for i in range(X.shape[1])]
        metadata = {
            'model_name': 'PCA-SIMCA',
            'task_type': 'one_class',
            'wavelengths': wavelengths,
            'n_vars': len(wavelengths),
            'performance': {'BalancedAcc': 0.95},
            'n_components': 5,
            'alpha': 0.05,
        }

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_model(
                model=model,
                preprocessor=None,
                metadata=metadata,
                filepath=tmp_path,
            )

            loaded = load_model(tmp_path)
            assert loaded['metadata']['model_name'] == 'PCA-SIMCA'
            assert loaded['metadata']['task_type'] == 'one_class'

            loaded_model = loaded['model']
            loaded_preds = loaded_model.predict(X)
            loaded_scores = loaded_model.decision_function(X)

            np.testing.assert_array_equal(original_preds, loaded_preds)
            np.testing.assert_array_almost_equal(original_scores, loaded_scores)
        finally:
            import os
            os.unlink(tmp_path)


class TestRunOneClassCV:
    """Tests for the extracted run_one_class_cv function."""

    def test_run_one_class_cv_basic(self):
        """Basic run_one_class_cv with PCA-SIMCA returns expected dict structure."""
        rng = np.random.RandomState(42)
        X_inlier = rng.randn(50, 20)
        X_outlier = rng.randn(10, 20) + 3
        X = np.vstack([X_inlier, X_outlier])
        y_oc = np.array([1] * 50 + [-1] * 10)

        result = run_one_class_cv(
            X, y_oc, 'PCA-SIMCA', {'n_components': 5, 'alpha': 0.05},
            n_folds=3, random_state=42,
        )

        expected_keys = {
            'fold_metrics', 'mean_metrics', 'cal_model', 'cal_scaler',
            'cal_pca_reducer', 'cal_metrics', 'per_contaminant_sensitivity',
            'oc_score_stats', 'skipped',
        }
        assert set(result.keys()) == expected_keys
        assert result['skipped'] is False

        metric_keys = {'sensitivity', 'specificity', 'precision', 'f1',
                       'accuracy', 'balanced_accuracy', 'auc'}
        assert set(result['mean_metrics'].keys()) == metric_keys

        # PCA-SIMCA doesn't use scaler
        assert result['cal_scaler'] is None
        assert result['cal_model'] is not None

    def test_run_one_class_cv_with_scaling(self):
        """OneClassSVM path applies StandardScaler."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(40, 10), rng.randn(8, 10) + 3])
        y_oc = np.array([1] * 40 + [-1] * 8)

        result = run_one_class_cv(
            X, y_oc, 'OneClassSVM', {'nu': 0.1, 'kernel': 'rbf'},
            n_folds=3, random_state=42,
        )

        assert result['skipped'] is False
        assert result['cal_scaler'] is not None

    def test_run_one_class_cv_skipped(self):
        """Too few inliers for requested folds returns skipped=True."""
        rng = np.random.RandomState(42)
        # Only 4 inliers with n_folds=5 — KFold will raise, which should be caught
        X = np.vstack([rng.randn(4, 10), rng.randn(3, 10) + 3])
        y_oc = np.array([1] * 4 + [-1] * 3)

        # n_folds=5 but only 4 inliers — should handle gracefully
        result = run_one_class_cv(
            X, y_oc, 'PCA-SIMCA', {'n_components': 2, 'alpha': 0.05},
            n_folds=5, random_state=42,
        )

        assert result['skipped'] is True

    def test_run_one_class_cv_per_contaminant(self):
        """Per-contaminant sensitivity computed when y_original provided."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(40, 10), rng.randn(5, 10) + 3, rng.randn(5, 10) + 5])
        y_oc = np.array([1] * 40 + [-1] * 10)
        y_original = np.array(['clean'] * 40 + ['contam_A'] * 5 + ['contam_B'] * 5)

        result = run_one_class_cv(
            X, y_oc, 'IsolationForest', {'n_estimators': 50, 'contamination': 0.05},
            n_folds=3, random_state=42, y_original=y_original,
        )

        assert result['skipped'] is False
        assert 'contam_A' in result['per_contaminant_sensitivity']
        assert 'contam_B' in result['per_contaminant_sensitivity']


# ============================================================================
# Full-spectrum-first preprocessing + save/load/predict roundtrip
# ============================================================================

class TestOneClassRefinementRoundtrip:
    """Regression coverage for the 2026-04-10 "all outliers" bug.

    The one-class refinement thread in spectral_predict_gui_optimized.py
    trains with ``prep_pipeline.fit_transform(X_full)`` first and then
    subsets to selected wavelengths, but it used to hardcode
    ``'use_full_spectrum_preprocessing': False`` in the saved metadata and
    never populate ``full_wavelengths``. At predict time model_io.py took
    the "subset then preprocess" branch, which produces feature values
    that are mathematically different from training (especially for SG
    derivatives where the window lands on subset boundaries). The scaler
    inside run_one_class_cv was fit on the training-time feature space,
    so predictions on the model's own training data came back as -1 for
    every specimen.

    These tests lock in the contract at the model_io level: when the
    metadata accurately describes the training order (Mode A =
    full-spectrum preprocess then subset), the save->load->predict
    roundtrip must correctly label the training inliers. They also pin
    the characterization that the *broken* metadata (which the GUI
    used to emit) would produce the all-outlier symptom, so that a
    future maintainer can see the bug mechanism from the test alone.
    """

    @pytest.fixture
    def spectral_data_with_full_spectrum_preprocessing(self):
        """Build a synthetic spectral DataFrame, apply SNV + SG 1st derivative
        to the FULL spectrum, then subset to the middle wavelengths. Returns
        everything the refinement thread would hand to run_one_class_cv plus
        the raw training DataFrame needed for the predict-time roundtrip."""
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import build_preprocessing_pipeline

        rng = np.random.RandomState(2026)
        n_features = 200
        # Inliers: tight cluster centered at a smooth baseline
        n_inliers = 60
        baseline = np.linspace(0.2, 0.8, n_features) + 0.1 * np.sin(
            np.linspace(0, 4 * np.pi, n_features)
        )
        X_inliers = baseline + rng.randn(n_inliers, n_features) * 0.02
        # Outliers: distinctly shifted and noisier
        n_outliers = 15
        X_outliers = baseline + 0.5 + rng.randn(n_outliers, n_features) * 0.15

        X_full_raw = np.vstack([X_inliers, X_outliers])
        y_oc = np.array([1] * n_inliers + [-1] * n_outliers)

        # Wavelength columns as floats, matching GUI convention
        original_wavelengths = np.linspace(900.0, 1700.0, n_features)
        X_df = pd.DataFrame(
            X_full_raw,
            columns=[float(w) for w in original_wavelengths],
        )

        # Build preprocessing pipeline: SNV + SG 1st derivative
        # This is the case qwen3-235B flagged as catastrophic at subset boundaries
        prep_steps = build_preprocessing_pipeline(
            'snv_deriv', deriv=1, window=11, polyorder=2
        )
        assert isinstance(prep_steps, list) and len(prep_steps) > 0
        prep_pipeline = Pipeline(prep_steps)

        # Refinement thread order: FIT on full spectrum, THEN subset
        X_full_preprocessed = prep_pipeline.fit_transform(X_full_raw)

        # Subset to middle 100 wavelengths (simulates a selected band)
        selected_indices = list(range(50, 150))
        selected_wavelengths = [float(original_wavelengths[i]) for i in selected_indices]
        X_work = X_full_preprocessed[:, selected_indices]

        return {
            'X_df': X_df,                                # raw training DataFrame (for predict)
            'X_work': X_work,                            # preprocessed + subset (for run_one_class_cv)
            'y_oc': y_oc,
            'prep_pipeline': prep_pipeline,
            'original_wavelengths': list(original_wavelengths),
            'selected_wavelengths': selected_wavelengths,
            'n_inliers': n_inliers,
            'n_outliers': n_outliers,
        }

    def _train_and_build_metadata(
        self,
        data,
        *,
        use_full_spectrum_preprocessing: bool,
        full_wavelengths,
    ):
        """Train a OneClassSVM via run_one_class_cv and package a metadata
        dict matching what _run_refined_model_thread passes to save_model.

        Parameterized on the two metadata fields whose incorrect values
        caused the bug, so we can construct both the broken and fixed
        variants from the same training run."""
        result = run_one_class_cv(
            data['X_work'],
            data['y_oc'],
            'OneClassSVM',
            {'nu': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
            n_folds=3,
            random_state=42,
        )
        assert result['skipped'] is False
        assert result['cal_model'] is not None
        assert result['cal_scaler'] is not None

        metadata = {
            'model_name': 'OneClassSVM',
            'task_type': 'one_class',
            'wavelengths': data['selected_wavelengths'],
            'full_wavelengths': full_wavelengths,
            'use_full_spectrum_preprocessing': use_full_spectrum_preprocessing,
            'n_vars': len(data['selected_wavelengths']),
            'inlier_class_label': 'Inlier',
            'performance': {'BalancedAcc': 0.95},
            # These three are the artifacts the 2026-04-10 fix added for grid search,
            # and which _run_refined_model_thread already stores correctly today.
            'scaler': result['cal_scaler'],
            'pca_reducer': result['cal_pca_reducer'],
            'oc_score_stats': result['oc_score_stats'],
        }
        return result['cal_model'], metadata

    def _run_predict_roundtrip(self, model, preprocessor, metadata, X_df):
        """Save/load/predict helper. Returns the predicted labels."""
        from spectral_predict.model_io import predict_with_model

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_model(
                model=model,
                preprocessor=preprocessor,
                metadata=metadata,
                filepath=tmp_path,
            )
            loaded = load_model(tmp_path)
            return predict_with_model(loaded, X_df)
        finally:
            import os
            os.unlink(tmp_path)

    def test_correct_metadata_reproduces_training_predictions_exactly(
        self, spectral_data_with_full_spectrum_preprocessing
    ):
        """Contract: when metadata accurately describes full-spectrum-first
        training (use_full_spectrum_preprocessing=True + full_wavelengths
        populated), Mode A in model_io.py must reproduce the training-time
        decisions EXACTLY on the training data. This is a much stronger
        guarantee than a loose accuracy threshold: it proves that predict
        time is numerically indistinguishable from training time.

        OneClassSVM with nu=0.1 only classifies ~70% of noisy synthetic
        training data as inliers in its *own* fit, so asserting on an
        accuracy threshold would be model-dependent. Asserting exact
        equality is model-agnostic — whatever the model decided at
        training time, the roundtrip must reproduce it bit-for-bit."""
        data = spectral_data_with_full_spectrum_preprocessing

        model, metadata = self._train_and_build_metadata(
            data,
            use_full_spectrum_preprocessing=True,
            full_wavelengths=data['original_wavelengths'],
        )

        # Training-time ground truth: what the model says about its own
        # training data, computed the same way run_one_class_cv's calibration
        # block does (scaler.transform then model.predict).
        X_work_scaled = metadata['scaler'].transform(data['X_work'])
        training_predictions = model.predict(X_work_scaled)

        # Predict-time via the save->load->predict roundtrip.
        roundtrip_predictions = self._run_predict_roundtrip(
            model=model,
            preprocessor=data['prep_pipeline'],
            metadata=metadata,
            X_df=data['X_df'],
        )

        np.testing.assert_array_equal(
            roundtrip_predictions,
            training_predictions,
            err_msg=(
                "Predict-time labels did not match training-time labels "
                "exactly. Mode A in model_io.py is supposed to reproduce "
                "the training pipeline (preprocess full -> subset) bit-for-bit. "
                "If this assertion fails, either the wavelength subsetting "
                "introduced a rounding/ordering mismatch, the preprocessor "
                "is no longer stateless across fit/transform, or the scaler/PCA "
                "pickling lost precision. Investigate before relaxing this test."
            ),
        )

        # Sanity check: outliers should be cleanly separable on this fixture.
        # This is a model-behavior smoke test, not the primary regression guard.
        y_oc = data['y_oc']
        outlier_correct = (roundtrip_predictions[y_oc == -1] == -1).mean()
        assert outlier_correct > 0.8, (
            f"OneClassSVM on this fixture should detect >80% of outliers, "
            f"got {outlier_correct:.1%}. This is a fixture sanity check, not "
            f"the primary regression guard — if it fires, the synthetic data "
            f"generator likely drifted, not the save/load/predict path."
        )

    def test_broken_metadata_reproduces_all_outlier_bug(
        self, spectral_data_with_full_spectrum_preprocessing
    ):
        """Characterization: with the broken metadata the refinement thread
        used to emit (use_full_spectrum_preprocessing=False,
        full_wavelengths=None), the predict path takes Mode B (subset-first
        → preprocess) and the resulting feature values do not match the
        training-time scaler. A large majority of samples are mislabeled,
        including training inliers. This pins the bug mechanism so a
        future maintainer can see why the two-line fix in
        _run_refined_model_thread is load-bearing.

        Compares predict-time labels to training-time labels (the same
        ground truth the correct-metadata test uses) and asserts they
        DISAGREE on a large fraction of samples. The test is model- and
        fixture-agnostic: whatever the model decided at training, broken
        metadata must cause a large mismatch."""
        data = spectral_data_with_full_spectrum_preprocessing

        model, metadata = self._train_and_build_metadata(
            data,
            use_full_spectrum_preprocessing=False,
            full_wavelengths=None,
        )

        # Training-time ground truth (computed on X_work, which used the
        # full-spectrum-first preprocessing order).
        X_work_scaled = metadata['scaler'].transform(data['X_work'])
        training_predictions = model.predict(X_work_scaled)

        roundtrip_predictions = self._run_predict_roundtrip(
            model=model,
            preprocessor=data['prep_pipeline'],
            metadata=metadata,
            X_df=data['X_df'],
        )

        disagreement = (roundtrip_predictions != training_predictions).mean()
        assert disagreement > 0.3, (
            f"Expected broken metadata to mislabel >30% of samples relative "
            f"to training-time ground truth, got {disagreement:.1%}. If this "
            f"fails, either model_io.py's Mode B path has become more "
            f"forgiving of the flag mismatch, or the fixture's SNV+SG "
            f"preprocessing is no longer order-sensitive — investigate "
            f"before deleting this test."
        )


class TestGridSearchValidationMetricsParity:
    """Regression coverage for the grid-search one-class validation pipeline.

    The bug this guards against (caught during the merge review of PR #3):
    `compute_validation_metrics_for_top_one_class_models` rebuilds the
    preprocessing pipeline from the result-row's `PreprocessBase` column,
    falling back to `Preprocess` when absent. The grid search path in
    `run_one_class_search` only wrote `Preprocess` (a *display* name like
    `"snv_deriv1_w11"`), never `PreprocessBase`. The display name is not a
    valid `build_preprocessing_pipeline` key, so the helper raised
    `ValueError("Unknown preprocess: snv_deriv1_w11")`, the row-level
    `except Exception: continue` swallowed it, and every derivative-based
    grid-search row got NaN `val_*` columns. Bayesian was unaffected because
    `unified_bayesian.py` already wrote `PreprocessBase`.

    The classification/regression grid search at `search.py:4506-4507`
    writes both columns; this test enforces that the one-class grid search
    follows the same contract."""

    @pytest.fixture
    def grid_search_oc_results_row(self):
        """Build a result-row dict shaped exactly like one of grid search's
        derivative-based one-class entries (matches the literal dict at
        ``search.py:5136-5174`` after the fix). Uses ``IsolationForest`` so
        the test stays focused on the preprocessing-name plumbing rather
        than estimator-specific quirks."""
        # Integer-spaced wavelengths so rounded-string roundtrip (f"{w:.1f}")
        # matches wl_to_idx keys exactly. Real user data is almost always
        # integer nm (ASD) or clean decimals; np.linspace-produced floats would
        # alias and every wavelength would miss the dict lookup.
        wl_arr = np.arange(1100.0, 1150.0, 1.0)  # 50 clean wavelengths
        all_vars = ','.join(f"{w:.1f}" for w in wl_arr)
        return {
            'Task': 'one_class',
            'Model': 'IsolationForest',
            'Params': "{'n_estimators': 50, 'random_state': 42}",
            'Preprocess': 'snv_deriv1_w11',
            'PreprocessBase': 'snv_deriv',
            'Deriv': 1,
            'Window': 11,
            'Poly': 2,
            'LVs': None,
            'n_vars': 50,
            'full_vars': 50,
            'SubsetTag': 'full',
            'Imbalance': '—',
            'Sensitivity': 0.85,
            'Specificity': 0.90,
            'Precision': 0.80,
            'F1': 0.82,
            'Accuracy': 0.88,
            'BalancedAcc': 0.875,
            'AUC': 0.91,
            'Sensitivitycv': 0.80,
            'Specificitycv': 0.88,
            'Precisioncv': 0.78,
            'F1cv': 0.79,
            'Accuracycv': 0.85,
            'BalancedAcccv': 0.84,
            'AUCcv': 0.89,
            'n_inliers': 60,
            'n_outliers': 15,
            'inlier_class_label': 'Clean',
            'top_vars': 'N/A',
            'all_vars': all_vars,
            'per_contaminant_sensitivity': {},
            'Rank': 1,
        }

    @pytest.fixture
    def synthetic_train_val(self):
        """Train/validation split with inliers + outliers, sized to expose
        the build_preprocessing_pipeline path (50 wavelengths, derivative
        window of 11 fits comfortably)."""
        rng = np.random.RandomState(0)
        n_features = 50

        clean_center = rng.randn(n_features) * 0.4
        X_train_clean = clean_center + rng.randn(40, n_features) * 0.2
        outlier_centers = rng.randn(2, n_features) * 1.5 + 2.0
        X_train_out = np.vstack([
            outlier_centers[0] + rng.randn(8, n_features) * 0.4,
            outlier_centers[1] + rng.randn(7, n_features) * 0.4,
        ])
        X_train = np.vstack([X_train_clean, X_train_out])
        y_train = np.array(['Clean'] * 40 + ['Contam'] * 15)

        X_val_clean = clean_center + rng.randn(15, n_features) * 0.2
        X_val_out = outlier_centers[0] + rng.randn(8, n_features) * 0.4
        X_val = np.vstack([X_val_clean, X_val_out])
        y_val = np.array(['Clean'] * 15 + ['Contam'] * 8)

        wavelengths = np.arange(1100.0, 1100.0 + n_features, 1.0)
        return X_train, y_train, X_val, y_val, wavelengths

    def test_grid_search_derivative_row_populates_val_columns(
        self, grid_search_oc_results_row, synthetic_train_val
    ):
        """The validation helper must populate val_* columns for a grid-search
        result row that uses a derivative-based preprocessing config. This
        is the regression test for the silent NaN val_* bug — before the
        fix, every derivative grid-search row failed inside
        ``build_preprocessing_pipeline`` and the row-level except dropped
        them to NaN."""
        from spectral_predict.contamination import (
            compute_validation_metrics_for_top_one_class_models,
        )

        X_train, y_train, X_val, y_val, wavelengths = synthetic_train_val
        df = pd.DataFrame([grid_search_oc_results_row])

        result_df = compute_validation_metrics_for_top_one_class_models(
            df_results=df,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            inlier_label='Clean',
            wavelengths=wavelengths,
            top_n=10,
        )

        # Spot-check the canonical val_* columns. The exact metric values
        # depend on IsolationForest behavior on the synthetic fixture; the
        # important assertion is that they ARE finite — i.e. the helper
        # didn't silently bail inside its except clause.
        for col in ('val_Sensitivity', 'val_Specificity', 'val_BalancedAcc',
                    'val_Accuracy'):
            assert col in result_df.columns, f"{col} missing from result frame"
            value = result_df.loc[0, col]
            assert pd.notna(value), (
                f"{col} is NaN for a derivative grid-search row — the "
                f"validation helper silently bailed. Likely cause: "
                f"PreprocessBase missing from grid search result dict at "
                f"search.py:5136-5174 or :5640-5710, OR the helper can't "
                f"parse the Preprocess display name. See the docstring on "
                f"TestGridSearchValidationMetricsParity for context."
            )

    def test_grid_search_row_without_preprocess_base_still_works(
        self, grid_search_oc_results_row, synthetic_train_val
    ):
        """Defense-in-depth: even if a caller forgets to write
        ``PreprocessBase`` (e.g. an older results CSV reloaded from disk,
        or a third-party caller building a row directly), the helper must
        not silently drop val_* metrics. This forces either the helper to
        normalize the display name or the contract to require PreprocessBase
        — whichever the maintainer picks, the failure mode must not be
        'silent NaN'."""
        from spectral_predict.contamination import (
            compute_validation_metrics_for_top_one_class_models,
        )

        X_train, y_train, X_val, y_val, wavelengths = synthetic_train_val
        row = dict(grid_search_oc_results_row)
        row.pop('PreprocessBase')  # Simulate the pre-fix grid search output
        df = pd.DataFrame([row])

        result_df = compute_validation_metrics_for_top_one_class_models(
            df_results=df,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            inlier_label='Clean',
            wavelengths=wavelengths,
            top_n=10,
        )

        value = result_df.loc[0, 'val_BalancedAcc']
        assert pd.notna(value), (
            "Helper silently produced NaN for a grid-search row missing "
            "PreprocessBase. The contract should be either 'PreprocessBase "
            "is required' (raise loudly) or 'normalize Preprocess as a "
            "fallback' — silent NaN is a regression."
        )

    @pytest.mark.parametrize(
        'mutate,label',
        [
            (lambda r: r.update({'Params': "{'n_estimators': 50, 'random_state':"}), 'malformed_params'),
            (lambda r: r.pop('Params'), 'missing_params'),
            (lambda r: r.update({'Params': ''}), 'empty_params'),
            (lambda r: r.update({'all_vars': '9999.0,9998.0,9997.0'}), 'unknown_wavelengths'),
            (lambda r: r.update({'all_vars': ''}), 'empty_all_vars'),
            (lambda r: r.pop('all_vars'), 'missing_all_vars'),
        ],
        ids=['malformed_params', 'missing_params', 'empty_params',
             'unknown_wavelengths', 'empty_all_vars', 'missing_all_vars'],
    )
    def test_corrupt_metadata_row_is_skipped_not_silently_wrong(
        self, grid_search_oc_results_row, synthetic_train_val, mutate, label, caplog
    ):
        """Rows with corrupt metadata (malformed/missing Params, missing or
        unmappable wavelengths) must be skipped with a logged warning, never
        silently validated with default params / truncated / full-spectrum
        feature sets. Each of these silent-wrong paths was flagged MAJOR in
        the pre-merge Codex review of commit 9e9ca11."""
        import logging
        from spectral_predict.contamination import (
            compute_validation_metrics_for_top_one_class_models,
        )

        X_train, y_train, X_val, y_val, wavelengths = synthetic_train_val
        row = dict(grid_search_oc_results_row)
        mutate(row)
        df = pd.DataFrame([row])

        with caplog.at_level(logging.WARNING, logger='spectral_predict.contamination'):
            result_df = compute_validation_metrics_for_top_one_class_models(
                df_results=df,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                inlier_label='Clean',
                wavelengths=wavelengths,
                top_n=10,
            )

        # Skipped row ⇒ val_* columns absent or NaN (never a believable-but-
        # wrong number). And a warning must have been emitted so operators
        # can see the row was dropped.
        val_cols = [c for c in result_df.columns if c.startswith('val_')]
        for col in val_cols:
            assert pd.isna(result_df.loc[0, col]), (
                f"{col} populated for corrupt-metadata case '{label}' — "
                f"helper should have skipped this row, not validated it "
                f"with fallback defaults"
            )
        assert any('OC Validation' in rec.message for rec in caplog.records), (
            f"No [OC Validation] warning emitted for corrupt case '{label}'. "
            f"Silent skips are the hazard this test guards against."
        )


class TestOneClassModelConfigParity:
    """Tests for default-preservation and per-model override behavior
    in the one-class grid-search path (_resolve_one_class_model_grids)."""

    def test_default_resolution_returns_curated_grids(self):
        from spectral_predict.search import _resolve_one_class_model_grids
        from spectral_predict.contamination import get_one_class_model_grids

        all_models = list(get_one_class_model_grids().keys())
        result = _resolve_one_class_model_grids(all_models)
        expected = get_one_class_model_grids()

        assert set(result.keys()) == set(expected.keys())
        for model_name in expected:
            assert result[model_name] == expected[model_name], (
                f"{model_name}: default grid should match curated grid exactly"
            )

    def test_override_one_model_leaves_others_untouched(self):
        from spectral_predict.search import _resolve_one_class_model_grids
        from spectral_predict.contamination import get_one_class_model_grids

        all_models = list(get_one_class_model_grids().keys())
        overrides = {
            'OneClassSVM': {
                'kernel': ['rbf'],
                'gamma': ['scale'],
                'nu': [0.05],
                'degree': [2],
            },
        }
        result = _resolve_one_class_model_grids(all_models, oc_model_param_overrides=overrides)

        assert len(result['OneClassSVM']) == 1
        assert result['OneClassSVM'][0] == {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05}

        curated = get_one_class_model_grids()
        for model_name in ('IsolationForest', 'EllipticEnvelope', 'LOF', 'PCA-SIMCA'):
            assert result[model_name] == curated[model_name], (
                f"{model_name} should still use curated default"
            )

    def test_ocsvm_override_prunes_non_poly_degree(self):
        from spectral_predict.search import _resolve_one_class_model_grids

        overrides = {
            'OneClassSVM': {
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale'],
                'nu': [0.05],
                'degree': [2],
            },
        }
        result = _resolve_one_class_model_grids(
            ['OneClassSVM'], oc_model_param_overrides=overrides,
        )
        grids = result['OneClassSVM']
        rbf_combos = [g for g in grids if g['kernel'] == 'rbf']
        poly_combos = [g for g in grids if g['kernel'] == 'poly']

        assert len(rbf_combos) == 1
        assert 'degree' not in rbf_combos[0], (
            "rbf combos should not include degree key"
        )
        assert len(poly_combos) == 1
        assert poly_combos[0]['degree'] == 2

    def test_empty_row_falls_back_to_defaults(self):
        from spectral_predict.search import _resolve_one_class_model_grids
        from spectral_predict.contamination import get_one_class_model_grids

        curated = get_one_class_model_grids()
        default_nn = sorted(set(p['n_neighbors'] for p in curated['LOF']))

        overrides = {
            'LOF': {
                'n_neighbors': [],
                'contamination': [0.1],
            },
        }
        result = _resolve_one_class_model_grids(
            ['LOF'], oc_model_param_overrides=overrides,
        )
        grids = result['LOF']
        nn_values = sorted(set(g['n_neighbors'] for g in grids))
        assert nn_values == default_nn, (
            f"Empty n_neighbors should fall back to {default_nn}, got {nn_values}"
        )

    def test_no_override_returns_curated_end_to_end(self):
        from spectral_predict.search import _resolve_one_class_model_grids
        from spectral_predict.contamination import get_one_class_model_grids

        all_models = list(get_one_class_model_grids().keys())
        result_none = _resolve_one_class_model_grids(all_models)
        result_empty = _resolve_one_class_model_grids(
            all_models, oc_model_param_overrides=None,
        )
        curated = get_one_class_model_grids()

        assert result_none == curated
        assert result_empty == curated

    def test_reverted_model_uses_curated_grid_when_others_customized(self):
        """Acceptance criterion 6: a model restored to shipped-default widget
        state uses the curated grid again, even when OTHER models remain
        customized. The GUI collector drops default-matching models from the
        override dict; the backend must treat that omission as 'use curated'."""
        from spectral_predict.search import _resolve_one_class_model_grids
        from spectral_predict.contamination import get_one_class_model_grids

        all_models = list(get_one_class_model_grids().keys())
        curated = get_one_class_model_grids()

        overrides_both_custom = {
            'OneClassSVM': {'kernel': ['rbf'], 'gamma': ['scale'], 'nu': [0.5], 'degree': [2]},
            'IsolationForest': {'n_estimators': [500], 'contamination': [0.2], 'max_features': [0.3]},
        }
        both_custom = _resolve_one_class_model_grids(
            all_models, oc_model_param_overrides=overrides_both_custom,
        )
        assert both_custom['OneClassSVM'] != curated['OneClassSVM']
        assert both_custom['IsolationForest'] != curated['IsolationForest']

        overrides_if_reverted = {
            'OneClassSVM': {'kernel': ['rbf'], 'gamma': ['scale'], 'nu': [0.5], 'degree': [2]},
        }
        if_reverted = _resolve_one_class_model_grids(
            all_models, oc_model_param_overrides=overrides_if_reverted,
        )

        assert if_reverted['IsolationForest'] == curated['IsolationForest'], (
            "Reverted IsolationForest must use curated grid exactly"
        )
        assert if_reverted['OneClassSVM'] == both_custom['OneClassSVM'], (
            "Still-customized OCSVM must retain its custom grid"
        )
        for m in ('EllipticEnvelope', 'LOF', 'PCA-SIMCA'):
            assert if_reverted[m] == curated[m], (
                f"Untouched {m} must remain on curated grid"
            )

    def test_legacy_oc_hyperparams_still_works(self):
        from spectral_predict.search import _resolve_one_class_model_grids
        from spectral_predict.contamination import get_one_class_model_grids

        all_models = list(get_one_class_model_grids().keys())
        result = _resolve_one_class_model_grids(
            all_models,
            oc_hyperparams={'nu': 0.02, 'contamination': 0.03, 'alpha': 0.02, 'n_components': 8},
        )

        for p in result['OneClassSVM']:
            assert p['nu'] == 0.02
        for p in result['IsolationForest']:
            assert p['contamination'] == 0.03
        for p in result['EllipticEnvelope']:
            assert p['contamination'] == 0.03
        for p in result['LOF']:
            assert p['contamination'] == 0.03
        for p in result['PCA-SIMCA']:
            assert p['alpha'] == 0.02
            assert p['n_components'] == 8
