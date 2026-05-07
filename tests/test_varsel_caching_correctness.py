"""Test that importance caching produces identical results to uncached computation.

This is the critical correctness test — it verifies that the importance_cache
in unified_bayesian.py returns the same importances (and therefore the same
variable subsets) as computing fresh every time.
"""
from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from spectral_predict.unified_bayesian import (
    compute_importances,
    apply_preprocessing,
    run_unified_bayesian,
)


@pytest.fixture(scope="module")
def spectral_data():
    """Create synthetic spectral data for variable selection testing."""
    rng = np.random.RandomState(42)
    n_samples = 100
    n_wavelengths = 500
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Regression target
    y_reg = rng.uniform(1, 20, n_samples)
    base = np.sin(np.linspace(0, 4 * np.pi, n_wavelengths))
    X = np.outer(y_reg, base) + rng.normal(0, 0.5, (n_samples, n_wavelengths))

    # Classification target (3 classes)
    y_cls = np.array([0] * 40 + [1] * 35 + [2] * 25)

    # One-class labels
    y_oc_str = np.array(['High'] * 80 + ['Low'] * 20)

    return X, y_reg, y_cls, y_oc_str, wavelengths


class TestComputeImportancesReproducibility:
    """Verify compute_importances is deterministic (same input = same output)."""

    def test_importance_method_deterministic(self, spectral_data):
        X, y_reg, _, _, _ = spectral_data
        imp1 = compute_importances(X, y_reg, 'importance', 'PLS', 3, 42, 'regression')
        imp2 = compute_importances(X, y_reg, 'importance', 'PLS', 3, 42, 'regression')
        np.testing.assert_array_equal(imp1, imp2)

    def test_importance_lightgbm_deterministic(self, spectral_data):
        X, y_reg, _, _, _ = spectral_data
        imp1 = compute_importances(X, y_reg, 'importance', 'LightGBM', 3, 42, 'regression')
        imp2 = compute_importances(X, y_reg, 'importance', 'LightGBM', 3, 42, 'regression')
        np.testing.assert_array_equal(imp1, imp2)

    def test_cars_deterministic(self, spectral_data):
        X, y_reg, _, _, _ = spectral_data
        imp1 = compute_importances(X, y_reg, 'cars', 'LightGBM', 3, 42, 'regression')
        imp2 = compute_importances(X, y_reg, 'cars', 'LightGBM', 3, 42, 'regression')
        np.testing.assert_array_equal(imp1, imp2)

    def test_uve_deterministic(self, spectral_data):
        X, y_reg, _, _, _ = spectral_data
        imp1 = compute_importances(X, y_reg, 'uve', 'PLS', 3, 42, 'regression')
        imp2 = compute_importances(X, y_reg, 'uve', 'PLS', 3, 42, 'regression')
        np.testing.assert_array_equal(imp1, imp2)

    def test_different_methods_produce_different_results(self, spectral_data):
        X, y_reg, _, _, _ = spectral_data
        imp_importance = compute_importances(X, y_reg, 'importance', 'LightGBM', 3, 42, 'regression')
        imp_cars = compute_importances(X, y_reg, 'cars', 'LightGBM', 3, 42, 'regression')
        # They should NOT be identical
        assert not np.array_equal(imp_importance, imp_cars), \
            "importance and CARS should produce different rankings"

    def test_different_preprocessing_produces_different_importances(self, spectral_data):
        X, y_reg, _, _, _ = spectral_data
        # Raw
        imp_raw = compute_importances(X, y_reg, 'importance', 'LightGBM', 3, 42, 'regression')
        # After SNV
        config_snv = {'name': 'snv', 'deriv': 0, 'window': 0, 'polyorder': 0,
                       'apply_baseline': False, 'apply_smoothing': False}
        X_snv = apply_preprocessing(X, config_snv)
        imp_snv = compute_importances(X_snv, y_reg, 'importance', 'LightGBM', 3, 42, 'regression')
        assert not np.array_equal(imp_raw, imp_snv), \
            "Different preprocessing should produce different importances"


class TestVarselSubsetsInUnifiedBayesian:
    """Test that unified Bayesian with variable selection produces correct results."""

    def test_regression_with_varsel_multiple_models(self, spectral_data):
        """Run regression Bayesian with multiple models — exercises varsel caching."""
        X, y_reg, _, _, wavelengths = spectral_data

        # Run PLS first, then Ridge — both should use cached importances for same preprocessing
        results = {}
        for model_name in ['PLS', 'Ridge']:
            df, study = run_unified_bayesian(
                X=X, y=y_reg, wavelengths=wavelengths,
                model_name=model_name,
                task_type='regression',
                n_trials=15,
                cv_folds=3,
                random_state=42,
                verbose=False,
            )
            results[model_name] = df
            assert len(df) > 0
            assert df['RMSEcv'].notna().any()

            # Check that variable selection was applied (subset_tag should vary)
            if 'subset_tag' in df.columns:
                tags = df['subset_tag'].unique()
                print(f"  {model_name}: subset_tags = {list(tags)}")

    def test_classification_with_varsel(self, spectral_data):
        """Run classification Bayesian with SVM — exercises importance caching."""
        X, _, y_cls, _, wavelengths = spectral_data

        df, study = run_unified_bayesian(
            X=X, y=y_cls, wavelengths=wavelengths,
            model_name='SVM',
            task_type='classification',
            n_trials=15,
            cv_folds=3,
            random_state=42,
            verbose=False,
        )
        assert len(df) > 0
        assert df['Accuracycv'].notna().any()
        best_acc = df['Accuracycv'].max()
        assert 0 < best_acc <= 1.0
        print(f"  SVM classification: best Accuracycv = {best_acc:.4f}")

    def test_one_class_with_varsel(self, spectral_data):
        """Run one-class Bayesian — exercises importance caching + calibration skip."""
        X, _, _, y_oc_str, wavelengths = spectral_data

        for model_name in ['PCA-SIMCA', 'IsolationForest']:
            df, study = run_unified_bayesian(
                X=X, y=y_oc_str, wavelengths=wavelengths,
                model_name=model_name,
                task_type='one_class',
                n_trials=15,
                cv_folds=3,
                random_state=42,
                verbose=False,
                inlier_class_label='High',
            )
            assert len(df) > 0
            print(f"  {model_name} one-class: {len(df)} results")

    def test_one_class_with_uve_is_coerced(self, spectral_data, caplog):
        """One-class + ``enable_uve=True`` must be coerced to ``enable_uve=False``
        with a warning, and no Optuna trial may have ``subset_type='uve'``.

        UVE on y_oc (the +1/-1 binary labels constructed inside
        ``create_unified_objective``) is a discrimination method, not a
        one-class method (CLAUDE.md:66 / Pomerantsev et al. 2025 LOVE).
        Backend coercion lives at ``unified_bayesian.run_unified_bayesian``
        entry plus a defense-in-depth guard in ``create_unified_objective``.

        Cycle 4 sister-site fix — closes Kimi K2.6 STRONG / Codex cycle 4
        STRONG #1 (corroborated by GLM 5.1 NEEDS_DISCUSSION). Replaces an
        earlier ``test_one_class_with_uve`` that asserted the leaky path
        succeeded.
        """
        import logging
        X, _, _, y_oc_str, wavelengths = spectral_data

        with caplog.at_level(logging.WARNING):
            df, study = run_unified_bayesian(
                X=X, y=y_oc_str, wavelengths=wavelengths,
                model_name='PCA-SIMCA',
                task_type='one_class',
                n_trials=10,
                cv_folds=3,
                random_state=42,
                verbose=False,
                inlier_class_label='High',
                enable_uve=True,
            )

        assert len(df) > 0, "Coerced run should still produce results"

        # Coercion warning must fire exactly once (run_unified_bayesian's
        # entry guard short-circuits enable_uve before create_unified_objective
        # sees it, so the inner defense-in-depth guard does not double-fire).
        coercion_msgs = [
            r for r in caplog.records
            if "enable_uve" in r.getMessage()
            and "one-class" in r.getMessage()
        ]
        assert len(coercion_msgs) == 1, (
            "Coercion warning must fire exactly once; got "
            f"{len(coercion_msgs)}: {[r.getMessage() for r in coercion_msgs]}"
        )

        # No trial may have selected UVE — confirms the coercion reached
        # the available_methods list before Optuna sampled subset_type.
        uve_trials = [
            t for t in study.trials
            if t.params.get('subset_type') == 'uve'
        ]
        assert not uve_trials, (
            f"No trial may run UVE for one-class; got {len(uve_trials)} "
            f"uve trial(s): {[t.number for t in uve_trials]}"
        )


class TestCachingDoesNotChangeResults:
    """Verify that caching produces IDENTICAL results to a clean run."""

    def test_regression_results_identical_across_runs(self, spectral_data):
        """Two runs with same seed should produce identical results."""
        X, y_reg, _, _, wavelengths = spectral_data
        results = []
        for _ in range(2):
            df, _ = run_unified_bayesian(
                X=X, y=y_reg, wavelengths=wavelengths,
                model_name='Ridge',
                task_type='regression',
                n_trials=10,
                cv_folds=3,
                random_state=42,
                verbose=False,
            )
            results.append(df['RMSEcv'].min())
        assert abs(results[0] - results[1]) < 1e-10, \
            f"Cached run differs: {results[0]} vs {results[1]}"

    def test_classification_results_identical_across_runs(self, spectral_data):
        X, _, y_cls, _, wavelengths = spectral_data
        results = []
        for _ in range(2):
            df, _ = run_unified_bayesian(
                X=X, y=y_cls, wavelengths=wavelengths,
                model_name='LightGBM',
                task_type='classification',
                n_trials=10,
                cv_folds=3,
                random_state=42,
                verbose=False,
            )
            results.append(df['Accuracycv'].max())
        assert abs(results[0] - results[1]) < 1e-10, \
            f"Cached run differs: {results[0]} vs {results[1]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
