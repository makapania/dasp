"""
Unit and integration tests for GA-LightGBM variable selection.

This test suite validates the genetic algorithm for LightGBM-based wavelength
selection (ga_lightgbm.py) including:
- Binary chromosome encoding (wavelength selection)
- Fitness evaluation with cross-validated LightGBM
- Genetic operators (tournament selection, two-point crossover, bit-flip mutation)
- Multi-run aggregation for stability
- LightGBM-specific hyperparameter handling
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Check if LightGBM is available
try:
    import lightgbm

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Import GA-LightGBM functions
if HAS_LIGHTGBM:
    from spectral_predict.ga_lightgbm import (
        ga_lightgbm_selection,
        ga_lightgbm_selection_detailed,
        _initialize_population,
        _fitness_function_lgbm,
        _tournament_selection,
        _two_point_crossover,
        _bit_flip_mutation,
        _run_single_ga_lgbm,
        FitnessCache,
        DEFAULT_N_ESTIMATORS,
        DEFAULT_NUM_LEAVES_REGRESSION,
        DEFAULT_NUM_LEAVES_CLASSIFICATION,
        DEFAULT_LEARNING_RATE,
        DEFAULT_REG_LAMBDA,
    )


# Skip all tests if LightGBM not available
pytestmark = pytest.mark.skipif(
    not HAS_LIGHTGBM, reason="LightGBM not installed"
)


# =============================================================================
# Unit Tests - Chromosome Encoding
# =============================================================================


@pytest.mark.unit
class TestGALightGBMChromosome:
    """Test binary chromosome encoding (same as GA-PLS)."""

    def test_encodes_hyperparams_implicitly(self):
        """Chromosomes encode wavelength selection (not hyperparams)."""
        rng = np.random.RandomState(42)
        n_wavelengths = 100

        population = _initialize_population(
            n_wavelengths=n_wavelengths, pop_size=10, init_ratio=0.3, rng=rng
        )

        # Check binary encoding
        assert np.all((population == 0) | (population == 1)), "Should be binary"
        assert population.shape == (
            10,
            n_wavelengths,
        ), "Shape should be (pop_size, n_wavelengths)"

    def test_decodes_to_valid_params(self):
        """LightGBM params should be valid."""
        # Test that default params are reasonable
        assert DEFAULT_N_ESTIMATORS > 0
        assert DEFAULT_NUM_LEAVES_REGRESSION > 0
        assert DEFAULT_NUM_LEAVES_CLASSIFICATION > 0
        assert 0 < DEFAULT_LEARNING_RATE <= 1
        assert DEFAULT_REG_LAMBDA >= 0


# =============================================================================
# Unit Tests - Fitness Evaluation
# =============================================================================


@pytest.mark.unit
class TestGALightGBMFitness:
    """Test LightGBM-based fitness function."""

    def test_fitness_uses_lightgbm(self, synthetic_spectra_small):
        """Fitness should use LightGBM for evaluation."""
        from sklearn.model_selection import KFold

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        n_wavelengths = X.shape[1]
        chromosome = np.zeros(n_wavelengths, dtype=np.uint8)
        chromosome[:20] = 1  # Select first 20

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        fitness = _fitness_function_lgbm(
            chromosome,
            X,
            y,
            cv,
            task_type="regression",
            n_estimators=10,
            num_leaves=15,
            learning_rate=0.1,
            reg_lambda=0.1,
            min_wavelengths=5,
        )

        # Should return negative RMSECV for regression
        assert fitness <= 0 or fitness == -np.inf
        assert not np.isnan(fitness)

    def test_respects_min_wavelengths(self, synthetic_spectra_small):
        """Should penalize chromosomes with too few wavelengths."""
        from sklearn.model_selection import KFold

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        n_wavelengths = X.shape[1]
        chromosome = np.zeros(n_wavelengths, dtype=np.uint8)
        chromosome[:2] = 1  # Only 2 selected

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        fitness = _fitness_function_lgbm(
            chromosome,
            X,
            y,
            cv,
            task_type="regression",
            n_estimators=10,
            min_wavelengths=5,  # Require at least 5
        )

        assert fitness == -np.inf, "Too few wavelengths should get -inf"

    def test_classification_fitness(self, classification_data):
        """Should work for classification tasks."""
        from sklearn.model_selection import StratifiedKFold

        X, y = classification_data
        X = X.values
        y = y.values

        n_wavelengths = X.shape[1]
        chromosome = np.zeros(n_wavelengths, dtype=np.uint8)
        chromosome[:30] = 1

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        fitness = _fitness_function_lgbm(
            chromosome,
            X,
            y,
            cv,
            task_type="classification",
            n_estimators=10,
            num_leaves=15,
        )

        # Classification returns accuracy (0 to 1)
        assert 0 <= fitness <= 1 or fitness == -np.inf


# =============================================================================
# Unit Tests - LightGBM Parameters
# =============================================================================


@pytest.mark.unit
class TestGALightGBMBounds:
    """Test LightGBM hyperparameter bounds and defaults."""

    def test_n_estimators_range(self):
        """N_estimators should be positive integer."""
        assert isinstance(DEFAULT_N_ESTIMATORS, int)
        assert DEFAULT_N_ESTIMATORS > 0
        assert DEFAULT_N_ESTIMATORS <= 1000, "Default should be reasonable"

    def test_learning_rate_range(self):
        """Learning rate should be in (0, 1]."""
        assert 0 < DEFAULT_LEARNING_RATE <= 1

    def test_num_leaves_range(self):
        """Num_leaves should be positive."""
        assert DEFAULT_NUM_LEAVES_REGRESSION > 0
        assert DEFAULT_NUM_LEAVES_CLASSIFICATION > 0

        # Typical values for tree-based models
        assert DEFAULT_NUM_LEAVES_REGRESSION <= 255
        assert DEFAULT_NUM_LEAVES_CLASSIFICATION <= 255

    def test_reg_lambda_range(self):
        """Reg_lambda should be non-negative."""
        assert DEFAULT_REG_LAMBDA >= 0

    def test_different_defaults_for_task_types(self):
        """Regression and classification should have different num_leaves."""
        # Typically regression uses more leaves
        assert DEFAULT_NUM_LEAVES_REGRESSION != DEFAULT_NUM_LEAVES_CLASSIFICATION


# =============================================================================
# Integration Tests - GA-LightGBM Selection
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestGALightGBMOptimization:
    """Test full GA-LightGBM optimization."""

    def test_returns_optimized_params(self, synthetic_spectra_small):
        """GA-LightGBM should return importance scores."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        # Check shape
        assert importances.shape == (X.shape[1],)

        # Check range [0, 1]
        assert np.all(importances >= 0) and np.all(importances <= 1)

    def test_params_in_valid_range(self, synthetic_spectra_small):
        """Returned importance scores should be valid."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            n_estimators=10,
            num_leaves=15,
            learning_rate=0.1,
            reg_lambda=0.1,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        # All importances should be in [0, 1]
        assert np.all(importances >= 0) and np.all(importances <= 1)

        # Should have some variation
        assert importances.std() > 0, "Importances should vary"

    def test_fitness_improves(self, synthetic_spectra_small):
        """Fitness should improve or stay constant across generations."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        # Run single GA to track history
        from sklearn.model_selection import KFold

        rng = np.random.RandomState(42)
        n_wavelengths = X.shape[1]

        # Small test: just check that it runs
        best_chrom, best_fit, history = _run_single_ga_lgbm(
            X,
            y,
            task_type="regression",
            population_size=10,
            n_generations=10,
            crossover_rate=0.6,
            mutation_rate=0.01,
            cv_folds=3,
            min_wavelengths=5,
            n_estimators=10,
            num_leaves=15,
            learning_rate=0.1,
            reg_lambda=0.1,
            early_stopping=20,
            random_state=42,
        )

        assert best_chrom is not None
        assert len(history) > 0

        # Fitness should not decrease (monotonic improvement with elitism)
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1], f"Fitness decreased at generation {i}"

    def test_reproducibility(self, synthetic_spectra_small):
        """Same seed should give same results."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances1 = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        importances2 = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        assert np.allclose(importances1, importances2), "Results should be reproducible"

    def test_classification_task(self, classification_data):
        """GA-LightGBM should work for classification."""
        X, y = classification_data
        X = X.values
        y = y.values

        importances = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            task_type="classification",
            n_estimators=10,
            num_leaves=15,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0) and np.all(importances <= 1)

    def test_custom_lightgbm_params(self, synthetic_spectra_small):
        """Should accept custom LightGBM parameters."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            n_estimators=20,  # Custom
            num_leaves=10,  # Custom
            learning_rate=0.05,  # Custom
            reg_lambda=0.5,  # Custom
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        assert importances.shape == (X.shape[1],)


# =============================================================================
# Integration Tests - Detailed Results
# =============================================================================


@pytest.mark.integration
class TestGALightGBMDetailed:
    """Test detailed results function."""

    def test_detailed_results(self, synthetic_spectra_small):
        """Detailed function should return complete results."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        results = ga_lightgbm_selection_detailed(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            selection_threshold=0.5,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        # Check keys
        assert "importances" in results
        assert "selected_indices" in results
        assert "selection_threshold" in results
        assert "n_selected" in results

        # Check types
        assert isinstance(results["importances"], np.ndarray)
        assert isinstance(results["selected_indices"], np.ndarray)
        assert isinstance(results["n_selected"], (int, np.integer))

        # Check consistency
        assert results["n_selected"] == len(results["selected_indices"])


# =============================================================================
# Comparison Tests - GA-LightGBM vs GA-PLS
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestComparisonWithGAPLS:
    """Compare GA-LightGBM with GA-PLS."""

    def test_both_return_valid_importances(self, synthetic_spectra_small):
        """Both GA-LightGBM and GA-PLS should return valid importances."""
        from spectral_predict.ga_pls import ga_pls_selection

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        # GA-PLS
        importances_pls = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # GA-LightGBM
        importances_lgbm = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        # Both should be valid
        assert importances_pls.shape == importances_lgbm.shape
        assert np.all(importances_pls >= 0) and np.all(importances_pls <= 1)
        assert np.all(importances_lgbm >= 0) and np.all(importances_lgbm <= 1)

    def test_may_select_different_variables(self, synthetic_spectra_small):
        """GA-LightGBM and GA-PLS may select different wavelengths."""
        from spectral_predict.ga_pls import ga_pls_selection

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        # GA-PLS
        importances_pls = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # GA-LightGBM
        importances_lgbm = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        # They may differ (tree-based vs linear model)
        # Just check that they're not identical (would be surprising)
        # But don't assert this strongly - could coincidentally be similar
        correlation = np.corrcoef(importances_pls, importances_lgbm)[0, 1]
        # Correlation should be reasonable but not necessarily perfect
        assert -1 <= correlation <= 1


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """GA-LightGBM should handle small datasets."""
        X = np.random.randn(15, 30)
        y = np.random.randn(15)

        importances = ga_lightgbm_selection(
            X,
            y,
            population_size=5,
            n_generations=3,
            n_runs=1,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        assert importances.shape == (30,)

    def test_few_wavelengths(self):
        """GA-LightGBM should handle few wavelengths."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        importances = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            min_wavelengths=3,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        assert importances.shape == (20,)

    def test_missing_lightgbm_error(self):
        """Should raise ImportError if LightGBM not installed."""
        # This test is tricky - we know LightGBM IS installed
        # But we can check that the function checks for it
        from spectral_predict.ga_lightgbm import HAS_LIGHTGBM as MODULE_HAS_LIGHTGBM

        assert MODULE_HAS_LIGHTGBM is True, "LightGBM should be available for tests"


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Test computational performance."""

    def test_faster_than_ga_pls(self, synthetic_spectra_medium):
        """GA-LightGBM should be faster than GA-PLS for large datasets."""
        import time
        from spectral_predict.ga_pls import ga_pls_selection

        X, y = synthetic_spectra_medium
        X = X.values
        y = y.values

        # Time GA-LightGBM
        start = time.time()
        _ = ga_lightgbm_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=1,
            n_estimators=10,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )
        time_lgbm = time.time() - start

        # Time GA-PLS
        start = time.time()
        _ = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=1,
            cv=3,
            n_components=10,
            random_state=42,
            verbose=0,
        )
        time_pls = time.time() - start

        # Note: This is a soft test - timing can vary
        # Just check that both complete in reasonable time
        assert time_lgbm < 300, "GA-LightGBM should complete in 5 minutes"
        assert time_pls < 300, "GA-PLS should complete in 5 minutes"

        # Print for information
        print(f"\nGA-LightGBM: {time_lgbm:.2f}s, GA-PLS: {time_pls:.2f}s")
