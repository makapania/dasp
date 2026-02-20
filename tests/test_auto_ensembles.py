"""
Unit tests for automatic region-specialist ensemble generation.

Tests the RegionSpecialistEnsemble, ClassSpecialistEnsemble, and
create_auto_ensembles() functionality.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

# Import ensemble classes and functions
from spectral_predict.ensemble import (
    RegionSpecialistEnsemble,
    ClassSpecialistEnsemble,
    select_top_models_per_region,
    create_auto_ensembles,
    compute_regional_rankings,
    compute_class_rankings,
)


class TestRegionSpecialistEnsemble:
    """Test the RegionSpecialistEnsemble class for regression tasks."""

    def test_basic_creation(self):
        """Test that RegionSpecialistEnsemble can be created."""
        # Create simple models
        model1 = Ridge(alpha=1.0)
        model2 = Ridge(alpha=0.1)

        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)

        models_per_region = {
            'Q1': [(model1, None)],
            'Q2': [(model2, None)],
            'Q3': [(model1, None)],
            'Q4': [(model2, None)],
        }

        ensemble = RegionSpecialistEnsemble(
            models_per_region=models_per_region,
            region_boundaries=[0, 25, 50, 75, 100],
            model_names_per_region={
                'Q1': ['Ridge_1'],
                'Q2': ['Ridge_2'],
                'Q3': ['Ridge_1'],
                'Q4': ['Ridge_2'],
            }
        )

        assert ensemble is not None
        assert len(ensemble.models_per_region) == 4

    def test_fit_and_predict(self):
        """Test that RegionSpecialistEnsemble can fit and predict."""
        # Create simple models
        model1 = Ridge(alpha=1.0)
        model2 = Ridge(alpha=0.1)

        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)

        models_per_region = {
            'Q1': [(model1, None)],
            'Q2': [(model2, None)],
            'Q3': [(model1, None)],
            'Q4': [(model2, None)],
        }

        ensemble = RegionSpecialistEnsemble(
            models_per_region=models_per_region,
            region_boundaries=[0, 25, 50, 75, 100],
        )

        # Fit
        ensemble.fit(X, y)

        # Predict
        predictions = ensemble.predict(X)

        assert predictions is not None
        assert len(predictions) == len(X)
        assert not np.any(np.isnan(predictions))

    def test_get_specialist_info(self):
        """Test get_specialist_info returns correct information."""
        model1 = Ridge(alpha=1.0)
        model2 = Ridge(alpha=0.1)

        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)

        models_per_region = {
            'Q1': [(model1, None)],
            'Q2': [(model2, None)],
            'Q3': [(model1, None), (model2, None)],
            'Q4': [(model2, None)],
        }

        ensemble = RegionSpecialistEnsemble(
            models_per_region=models_per_region,
            region_boundaries=[0, 25, 50, 75, 100],
            model_names_per_region={
                'Q1': ['Ridge_1'],
                'Q2': ['Ridge_2'],
                'Q3': ['Ridge_1', 'Ridge_2'],
                'Q4': ['Ridge_2'],
            }
        )

        info = ensemble.get_specialist_info()

        assert 'Q1' in info
        assert 'Q3' in info
        assert info['Q1'] == ['Ridge_1']
        assert info['Q3'] == ['Ridge_1', 'Ridge_2']


class TestClassSpecialistEnsemble:
    """Test the ClassSpecialistEnsemble class for classification tasks."""

    def test_basic_creation(self):
        """Test that ClassSpecialistEnsemble can be created."""
        # Create simple models
        model1 = LogisticRegression(random_state=42)
        model2 = LogisticRegression(random_state=43)

        X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                                   n_informative=5, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)

        models_per_class = {
            0: [(model1, None)],
            1: [(model2, None)],
            2: [(model1, None), (model2, None)],
        }

        ensemble = ClassSpecialistEnsemble(
            models_per_class=models_per_class,
            model_names_per_class={
                0: ['LogReg_1'],
                1: ['LogReg_2'],
                2: ['LogReg_1', 'LogReg_2'],
            },
            classes=np.array([0, 1, 2])
        )

        assert ensemble is not None
        assert len(ensemble.models_per_class) == 3

    def test_fit_and_predict(self):
        """Test that ClassSpecialistEnsemble can fit and predict."""
        model1 = LogisticRegression(random_state=42)
        model2 = LogisticRegression(random_state=43)

        X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                                   n_informative=5, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)

        models_per_class = {
            0: [(model1, None)],
            1: [(model2, None)],
            2: [(model1, None)],
        }

        ensemble = ClassSpecialistEnsemble(
            models_per_class=models_per_class,
            classes=np.array([0, 1, 2])
        )

        # Fit
        ensemble.fit(X, y)

        # Predict
        predictions = ensemble.predict(X)

        assert predictions is not None
        assert len(predictions) == len(X)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_predict_proba(self):
        """Test that ClassSpecialistEnsemble returns valid probabilities."""
        model1 = LogisticRegression(random_state=42)
        model2 = LogisticRegression(random_state=43)

        X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                                   n_informative=5, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)

        models_per_class = {
            0: [(model1, None)],
            1: [(model2, None)],
            2: [(model1, None)],
        }

        ensemble = ClassSpecialistEnsemble(
            models_per_class=models_per_class,
            classes=np.array([0, 1, 2])
        )
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)

        assert proba.shape == (len(X), 3)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
        # All probabilities should be non-negative
        assert np.all(proba >= 0)


class TestComputeRegionalRankings:
    """Test the compute_regional_rankings function."""

    def test_with_regional_rmse_column(self):
        """Test ranking with valid regional_rmse data."""
        # Create test DataFrame with regional RMSE data
        results_df = pd.DataFrame({
            'Model': ['PLS', 'Ridge', 'Lasso'],
            'R2': [0.9, 0.85, 0.88],
            'RMSE': [0.1, 0.15, 0.12],
            'regional_rmse': [
                {'Q1': 0.05, 'Q2': 0.08, 'Q3': 0.12, 'Q4': 0.15},
                {'Q1': 0.10, 'Q2': 0.12, 'Q3': 0.14, 'Q4': 0.18},
                {'Q1': 0.08, 'Q2': 0.06, 'Q3': 0.10, 'Q4': 0.16},
            ],
        })

        result = compute_regional_rankings(results_df, top_n=2)

        assert 'rankings' in result
        assert 'top_models' in result
        assert len(result['rankings']) == 4  # Q1, Q2, Q3, Q4

    def test_without_regional_rmse_column(self):
        """Test that function handles missing regional_rmse gracefully."""
        results_df = pd.DataFrame({
            'Model': ['PLS', 'Ridge'],
            'R2': [0.9, 0.85],
        })

        result = compute_regional_rankings(results_df, top_n=2)

        assert 'rankings' in result
        assert len(result['top_models']) == 0


class TestComputeClassRankings:
    """Test the compute_class_rankings function."""

    def test_with_per_class_metrics_column(self):
        """Test ranking with valid per_class_metrics data."""
        results_df = pd.DataFrame({
            'Model': ['LogReg', 'SVM', 'RF'],
            'Accuracy': [0.9, 0.85, 0.88],
            'per_class_metrics': [
                {'0': {'F1': 0.85}, '1': {'F1': 0.90}, '2': {'F1': 0.88}},
                {'0': {'F1': 0.80}, '1': {'F1': 0.82}, '2': {'F1': 0.85}},
                {'0': {'F1': 0.88}, '1': {'F1': 0.85}, '2': {'F1': 0.86}},
            ],
        })

        result = compute_class_rankings(results_df, top_n=2)

        assert 'rankings' in result
        assert 'class_labels' in result
        assert result['class_labels'] == ['0', '1', '2']

    def test_without_per_class_metrics_column(self):
        """Test that function handles missing per_class_metrics gracefully."""
        results_df = pd.DataFrame({
            'Model': ['LogReg', 'SVM'],
            'Accuracy': [0.9, 0.85],
        })

        result = compute_class_rankings(results_df, top_n=2)

        assert 'rankings' in result
        assert result['class_labels'] is None


class TestCreateAutoEnsembles:
    """Test the create_auto_ensembles function."""

    def test_regression_auto_ensembles(self):
        """Test auto-ensemble creation for regression tasks."""
        # Create synthetic data
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'wl_{i}' for i in range(X.shape[1])])

        # Create mock results DataFrame with regional RMSE
        results_df = pd.DataFrame({
            'Model': ['PLS', 'Ridge', 'Lasso', 'ElasticNet'],
            'R2': [0.9, 0.85, 0.88, 0.87],
            'RMSE': [0.1, 0.15, 0.12, 0.13],
            'Preprocess': ['raw', 'snv', 'sg1', 'sg2'],
            'Deriv': [0, 0, 1, 2],
            'Window': [15, 15, 11, 11],
            'Poly': [2, 2, 2, 2],
            'Params': ['{}', '{}', '{}', '{}'],
            'all_vars': ['N/A', 'N/A', 'N/A', 'N/A'],
            'CompositeScore': [0.95, 0.90, 0.92, 0.91],
            'regional_rmse': [
                {'Q1': 0.05, 'Q2': 0.08, 'Q3': 0.12, 'Q4': 0.15},
                {'Q1': 0.10, 'Q2': 0.06, 'Q3': 0.14, 'Q4': 0.18},
                {'Q1': 0.08, 'Q2': 0.10, 'Q3': 0.08, 'Q4': 0.16},
                {'Q1': 0.09, 'Q2': 0.11, 'Q3': 0.10, 'Q4': 0.12},
            ],
        })

        # Create simple mock reconstruct function
        def mock_reconstruct(row, X_train, y_train):
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            return model, f"{row['Model']} ({row['Preprocess']})"

        # Test the function
        auto_ensembles = create_auto_ensembles(
            results_df=results_df,
            X_train=X_df,
            y_train=y,
            task_type='regression',
            reconstruct_func=mock_reconstruct,
            all_wavelengths=X_df.columns.tolist()
        )

        # Verify results
        assert isinstance(auto_ensembles, dict)
        # Should have at least some ensembles (depending on model diversity)
        if auto_ensembles:
            for name, info in auto_ensembles.items():
                assert 'ensemble' in info
                assert 'base_models' in info
                assert 'metrics' in info
                assert 'r2' in info['metrics']
                assert 'rmse' in info['metrics']

    def test_classification_auto_ensembles(self):
        """Test auto-ensemble creation for classification tasks."""
        # Create synthetic data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                                   n_informative=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'wl_{i}' for i in range(X.shape[1])])

        # Create mock results DataFrame with per-class metrics
        results_df = pd.DataFrame({
            'Model': ['LogReg', 'SVM', 'RF', 'XGB'],
            'Accuracy': [0.9, 0.85, 0.88, 0.87],
            'F1': [0.88, 0.83, 0.86, 0.85],
            'Preprocess': ['raw', 'snv', 'sg1', 'sg2'],
            'Deriv': [0, 0, 1, 2],
            'Window': [15, 15, 11, 11],
            'Poly': [2, 2, 2, 2],
            'Params': ['{}', '{}', '{}', '{}'],
            'all_vars': ['N/A', 'N/A', 'N/A', 'N/A'],
            'CompositeScore': [0.95, 0.90, 0.92, 0.91],
            'per_class_metrics': [
                {'0': {'F1': 0.85}, '1': {'F1': 0.90}, '2': {'F1': 0.88}},
                {'0': {'F1': 0.80}, '1': {'F1': 0.82}, '2': {'F1': 0.85}},
                {'0': {'F1': 0.88}, '1': {'F1': 0.85}, '2': {'F1': 0.86}},
                {'0': {'F1': 0.84}, '1': {'F1': 0.86}, '2': {'F1': 0.87}},
            ],
        })

        # Create simple mock reconstruct function
        def mock_reconstruct(row, X_train, y_train):
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            return model, f"{row['Model']} ({row['Preprocess']})"

        # Test the function
        auto_ensembles = create_auto_ensembles(
            results_df=results_df,
            X_train=X_df,
            y_train=y,
            task_type='classification',
            reconstruct_func=mock_reconstruct,
            all_wavelengths=X_df.columns.tolist()
        )

        # Verify results
        assert isinstance(auto_ensembles, dict)
        # Should have at least some ensembles (depending on model diversity)
        if auto_ensembles:
            for name, info in auto_ensembles.items():
                assert 'ensemble' in info
                assert 'base_models' in info
                assert 'metrics' in info
                assert 'accuracy' in info['metrics']
                assert 'f1' in info['metrics']

    def test_classification_with_numeric_labels(self):
        """Test auto-ensemble creation for classification with numeric labels (0, 1, 2).

        This tests the bug fix where numeric labels (not dtype=object) were incorrectly
        detected as regression because the inference logic checked `y_train.dtype == object`.
        The fix ensures task_type is passed explicitly through the closure.
        """
        # Create synthetic data with numeric integer labels (the bug case)
        X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                                   n_informative=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'wl_{i}' for i in range(X.shape[1])])

        # Verify y has numeric dtype (not object) - this is the bug trigger condition
        assert y.dtype != object, "Test setup: y should have numeric dtype to test the bug fix"
        assert len(np.unique(y)) == 3, "Test setup: should have 3 classes"

        # Create mock results DataFrame with per-class metrics
        results_df = pd.DataFrame({
            'Model': ['LogReg', 'SVM', 'RF', 'XGB'],
            'Accuracy': [0.9, 0.85, 0.88, 0.87],
            'F1': [0.88, 0.83, 0.86, 0.85],
            'Preprocess': ['raw', 'snv', 'sg1', 'sg2'],
            'Deriv': [0, 0, 1, 2],
            'Window': [15, 15, 11, 11],
            'Poly': [2, 2, 2, 2],
            'Params': ['{}', '{}', '{}', '{}'],
            'all_vars': ['N/A', 'N/A', 'N/A', 'N/A'],
            'CompositeScore': [0.95, 0.90, 0.92, 0.91],
            'per_class_metrics': [
                {'0': {'F1': 0.85}, '1': {'F1': 0.90}, '2': {'F1': 0.88}},
                {'0': {'F1': 0.80}, '1': {'F1': 0.82}, '2': {'F1': 0.85}},
                {'0': {'F1': 0.88}, '1': {'F1': 0.85}, '2': {'F1': 0.86}},
                {'0': {'F1': 0.84}, '1': {'F1': 0.86}, '2': {'F1': 0.87}},
            ],
        })

        # Create mock reconstruct function that returns classification models
        def mock_reconstruct(row, X_train, y_train):
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            return model, f"{row['Model']} ({row['Preprocess']})"

        # Test with task_type='classification' explicitly
        auto_ensembles = create_auto_ensembles(
            results_df=results_df,
            X_train=X_df,
            y_train=y,
            task_type='classification',
            reconstruct_func=mock_reconstruct,
            all_wavelengths=X_df.columns.tolist()
        )

        # Verify results use classification ensembles
        assert isinstance(auto_ensembles, dict)
        if auto_ensembles:
            for name, info in auto_ensembles.items():
                assert 'ensemble' in info
                ensemble = info['ensemble']
                # Should be ClassSpecialistEnsemble, not RegionSpecialistEnsemble
                assert isinstance(ensemble, ClassSpecialistEnsemble), \
                    f"Expected ClassSpecialistEnsemble but got {type(ensemble).__name__}"
                # Metrics should be classification metrics (accuracy, f1), not regression (r2, rmse)
                assert 'accuracy' in info['metrics'], "Should have accuracy metric for classification"
                assert 'f1' in info['metrics'], "Should have f1 metric for classification"
                assert 'r2' not in info['metrics'], "Should NOT have r2 metric for classification"
                assert 'rmse' not in info['metrics'], "Should NOT have rmse metric for classification"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
