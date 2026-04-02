"""Baseline tests for unified Bayesian optimization.

Tests regression and classification with multiple model types to ensure
no regressions from performance optimizations in the one-class path.
Uses small trial counts (10) for speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectral_predict.unified_bayesian import run_unified_bayesian


@pytest.fixture(scope="module")
def bone_data():
    """Load BoneCollagen example data and create synthetic spectral matrix."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'example', 'BoneCollagen.csv')
    df = pd.read_csv(csv_path)

    # Create synthetic spectral data (50 samples x 200 wavelengths)
    rng = np.random.RandomState(42)
    n_samples = len(df)
    n_wavelengths = 200
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Create spectra correlated with %Collagen
    y_reg = df['%Collagen'].values
    y_cls = df['CollagenCat'].values

    # Build X with signal + noise
    base_spectrum = np.sin(np.linspace(0, 4 * np.pi, n_wavelengths))
    X = np.outer(y_reg, base_spectrum) + rng.normal(0, 0.5, (n_samples, n_wavelengths))

    return X, y_reg, y_cls, wavelengths


N_TRIALS = 10  # Small for speed; enough to verify correctness
CV_FOLDS = 3   # Small for speed


class TestRegressionBaseline:
    """Baseline tests for regression models through unified Bayesian."""

    @pytest.mark.parametrize("model_name", ["PLS", "Ridge", "LightGBM"])
    def test_regression_model(self, bone_data, model_name):
        X, y_reg, _, wavelengths = bone_data
        results_df, study = run_unified_bayesian(
            X=X, y=y_reg, wavelengths=wavelengths,
            model_name=model_name,
            task_type='regression',
            n_trials=N_TRIALS,
            cv_folds=CV_FOLDS,
            random_state=42,
            verbose=False,
        )
        assert results_df is not None
        assert len(results_df) > 0
        assert 'RMSEcv' in results_df.columns
        # Verify we got actual numeric results
        assert results_df['RMSEcv'].notna().any()
        best_rmse = results_df['RMSEcv'].min()
        assert np.isfinite(best_rmse)
        assert best_rmse > 0
        print(f"  {model_name} regression: best RMSEcv = {best_rmse:.4f}")


class TestClassificationBaseline:
    """Baseline tests for classification models through unified Bayesian."""

    @pytest.mark.parametrize("model_name", ["PLS-DA", "LightGBM", "SVM"])
    def test_classification_model(self, bone_data, model_name):
        X, _, y_cls, wavelengths = bone_data

        # Encode labels to integers
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_cls)

        results_df, study = run_unified_bayesian(
            X=X, y=y_encoded, wavelengths=wavelengths,
            model_name=model_name,
            task_type='classification',
            n_trials=N_TRIALS,
            cv_folds=CV_FOLDS,
            random_state=42,
            verbose=False,
        )
        assert results_df is not None
        assert len(results_df) > 0
        assert 'Accuracycv' in results_df.columns
        assert results_df['Accuracycv'].notna().any()
        best_acc = results_df['Accuracycv'].max()
        assert np.isfinite(best_acc)
        assert 0 < best_acc <= 1.0
        print(f"  {model_name} classification: best Accuracycv = {best_acc:.4f}")


class TestResultsReproducibility:
    """Verify results are deterministic with same random seed."""

    def test_regression_reproducible(self, bone_data):
        X, y_reg, _, wavelengths = bone_data
        results = []
        for _ in range(2):
            df, _ = run_unified_bayesian(
                X=X, y=y_reg, wavelengths=wavelengths,
                model_name='PLS',
                task_type='regression',
                n_trials=5,
                cv_folds=CV_FOLDS,
                random_state=42,
                verbose=False,
            )
            results.append(df['RMSEcv'].min())
        assert abs(results[0] - results[1]) < 1e-10, \
            f"Results not reproducible: {results[0]} vs {results[1]}"

    def test_classification_reproducible(self, bone_data):
        X, _, y_cls, wavelengths = bone_data
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_cls)

        results = []
        for _ in range(2):
            df, _ = run_unified_bayesian(
                X=X, y=y_encoded, wavelengths=wavelengths,
                model_name='PLS-DA',
                task_type='classification',
                n_trials=5,
                cv_folds=CV_FOLDS,
                random_state=42,
                verbose=False,
            )
            results.append(df['Accuracycv'].max())
        assert abs(results[0] - results[1]) < 1e-10, \
            f"Results not reproducible: {results[0]} vs {results[1]}"


class TestOneClassBaseline:
    """Baseline tests for one-class models through unified Bayesian."""

    @pytest.mark.parametrize("model_name", ["PCA-SIMCA", "IsolationForest"])
    def test_one_class_model(self, bone_data, model_name):
        X, _, y_cls, wavelengths = bone_data

        # Use "High" as inlier class, rest as outliers
        import time
        start = time.perf_counter()

        results_df, study = run_unified_bayesian(
            X=X, y=y_cls, wavelengths=wavelengths,
            model_name=model_name,
            task_type='one_class',
            n_trials=N_TRIALS,
            cv_folds=CV_FOLDS,
            random_state=42,
            verbose=False,
            inlier_class_label='High',
        )
        elapsed = time.perf_counter() - start

        assert results_df is not None
        assert len(results_df) > 0
        print(f"  {model_name} one-class: {len(results_df)} results, took {elapsed:.2f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
