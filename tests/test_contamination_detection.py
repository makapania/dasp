"""Tests for one-class contamination detection models.

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
        has_per_contam = any(c.startswith('Sens_') for c in results.columns)
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
