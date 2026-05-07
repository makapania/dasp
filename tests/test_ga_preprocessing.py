"""
Unit and integration tests for exhaustive preprocessing optimization.

Originally written for the GA preprocessing path; the GA evolution loop and
its operators (tournament_selection, crossover, mutate) were removed in
2026-05-06 because parallel exhaustive enumeration finished the same
14 x 17 = 238 cell space in seconds. This module now validates:
- Chromosome encoding / decoding (still 2-gene; backward-compat with saved CSVs)
- Fitness evaluation
- Full exhaustive optimization workflow
- Convenience wrapper for integration with search.py
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from spectral_predict.ga_preprocessing import (
        random_chromosome,
        chromosome_to_transform,
        get_config_description,
        evaluate_fitness,
        optimize_preprocessing,
        get_optimized_preproc_config,
        PREPROC_TYPES,
        WINDOW_SIZES,
        DERIVATIVE_WINDOW_RANGES,
        N_GENES,
    )
    HAS_GA_PREPROCESSING = True
except (ImportError, ModuleNotFoundError):
    HAS_GA_PREPROCESSING = False

pytestmark = pytest.mark.skipif(
    not HAS_GA_PREPROCESSING, reason="ga_preprocessing imports failed"
)


# =============================================================================
# Unit Tests - Chromosome Encoding
# =============================================================================


@pytest.mark.unit
class TestChromosomeEncoding:
    """Test chromosome encoding and decoding."""

    def test_random_chromosome_shape(self):
        """Random chromosome should have correct number of genes."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        assert chrom.shape == (N_GENES,), f"Expected {N_GENES} genes"
        assert chrom.dtype == np.int32, "Genes should be integers"

    def test_random_chromosome_valid_ranges(self):
        """Random chromosome genes should be in valid ranges."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        assert 0 <= chrom[0] < len(PREPROC_TYPES), "Invalid preproc type index"
        assert 0 <= chrom[1] < len(WINDOW_SIZES), "Invalid window index"

    def test_encode_decode_roundtrip(self):
        """Encoding and decoding should preserve chromosome information."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        # Decode to transform
        name, transform = chromosome_to_transform(chrom)

        # Check name is string
        assert isinstance(name, str), "Name should be string"
        assert len(name) > 0, "Name should not be empty"

        # Check transform is callable or None
        assert transform is None or callable(transform), "Transform should be callable or None"

    def test_valid_preprocessing_decoded(self):
        """Decoded preprocessing should produce valid transformations."""
        # Test specific configurations (2-gene encoding: [preproc_type, window])
        test_configs = [
            np.array([0, 0], dtype=np.int32),   # raw
            np.array([1, 0], dtype=np.int32),   # snv
            np.array([2, 2], dtype=np.int32),   # deriv1 with window=9
            np.array([3, 3], dtype=np.int32),   # deriv2 with window=11
            np.array([6, 4], dtype=np.int32),   # snv_deriv1 with window=13
        ]

        X_test = np.random.randn(10, 50)

        for chrom in test_configs:
            name, transform = chromosome_to_transform(chrom)

            if transform is not None:
                X_transformed = transform(X_test)

                # Check output shape matches input
                assert X_transformed.shape == X_test.shape, (
                    f"Transform {name} changed shape"
                )

                # Check for finite values
                assert np.isfinite(X_transformed).all(), (
                    f"Transform {name} produced non-finite values"
                )

    def test_get_config_description(self):
        """Config description should be human-readable."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        desc = get_config_description(chrom)

        assert isinstance(desc, str), "Description should be string"
        assert len(desc) > 0, "Description should not be empty"
        assert "Preproc:" in desc, "Description should mention preprocessing"


# =============================================================================
# Unit Tests - Fitness Evaluation
# =============================================================================


@pytest.mark.unit
class TestFitnessEvaluation:
    """Test fitness evaluation function."""

    def test_fitness_is_cross_val_score(self, synthetic_spectra_small):
        """Fitness should be based on cross-validation score."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        # Evaluate fitness
        fitness = evaluate_fitness(chrom, X, y, cv_folds=3, n_components=5)

        # For regression, fitness is negative RMSECV (higher is better)
        assert fitness <= 0 or fitness == -np.inf, "Fitness should be negative or -inf"

    def test_handles_failed_preprocessing(self, synthetic_spectra_small):
        """Fitness should handle preprocessing failures gracefully."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        # Create chromosome that might cause issues
        # deriv4_snv (index 13) with smallest window (index 0 = 5)
        chrom = np.array([13, 0], dtype=np.int32)

        fitness = evaluate_fitness(chrom, X, y, cv_folds=3, n_components=5)

        # Should return finite value or -inf, but not crash
        assert fitness == -np.inf or np.isfinite(fitness), (
            "Fitness should be finite or -inf, not NaN"
        )

    def test_penalty_for_invalid(self):
        """Invalid chromosomes should get very low fitness."""
        # Create data that will cause preprocessing to fail
        X = np.ones((10, 20))  # Zero variance
        y = np.random.randn(10)

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        fitness = evaluate_fitness(chrom, X, y, cv_folds=3, n_components=5)

        # Should get penalty
        assert fitness == -np.inf, "Invalid preprocessing should get -inf fitness"

    def test_fitness_model_pls(self, synthetic_spectra_small):
        """Test fitness evaluation with PLS model."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        fitness = evaluate_fitness(
            chrom, X, y, cv_folds=3, n_components=5, fitness_model="pls"
        )

        assert np.isfinite(fitness) or fitness == -np.inf

    def test_fitness_model_lightgbm(self, synthetic_spectra_small):
        """Test fitness evaluation with LightGBM model."""
        try:
            import lightgbm
        except ImportError:
            pytest.skip("LightGBM not installed")

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        fitness = evaluate_fitness(
            chrom, X, y, cv_folds=3, n_components=5, fitness_model="lightgbm"
        )

        assert np.isfinite(fitness) or fitness == -np.inf


# =============================================================================
# Unit Tests - Genetic Operators
# =============================================================================


# =============================================================================
# Integration Tests - Exhaustive Optimization
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestExhaustivePreprocessingOptimization:
    """Test full exhaustive preprocessing optimization (replaces GA loop)."""

    def test_basic_optimization(self, synthetic_spectra_small):
        """Exhaustive should complete and return valid result."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            method="exhaustive",
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
            n_jobs=1,  # Sequential for test determinism
        )

        # Check result structure
        assert "best_genes" in result
        assert "best_name" in result
        assert "best_transform" in result
        assert "best_rmsecv" in result
        assert "best_config" in result
        assert "history" in result
        assert "configs" in result  # Top-N output

        # Check best_genes shape (chromosome encoding still 2-gene)
        assert result["best_genes"].shape == (N_GENES,)
        assert isinstance(result["best_name"], str)
        assert result["best_transform"] is None or callable(result["best_transform"])

    def test_best_preprocessing_identified(self, synthetic_spectra_small):
        """Exhaustive should identify a reasonable preprocessing configuration."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            method="exhaustive",
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
            n_jobs=1,
        )

        if result["best_transform"] is not None:
            X_preprocessed = result["best_transform"](X)
            assert X_preprocessed.shape == X.shape
            assert np.isfinite(X_preprocessed).all()

    def test_respects_search_space(self, synthetic_spectra_small):
        """Exhaustive should only produce valid preprocessing combinations."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            method="exhaustive",
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
            n_jobs=1,
        )

        genes = result["best_genes"]
        assert 0 <= genes[0] < len(PREPROC_TYPES)
        assert 0 <= genes[1] < len(WINDOW_SIZES)

    def test_reproducibility_with_seed(self, synthetic_spectra_small):
        """Exhaustive should be reproducible with same random seed."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result1 = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            random_state=42, verbose=0, n_jobs=1,
        )
        result2 = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            random_state=42, verbose=0, n_jobs=1,
        )

        assert np.array_equal(
            result1["best_genes"], result2["best_genes"]
        ), "Same seed should give same result"

    def test_classification_task(self, classification_data):
        """Exhaustive should work for classification tasks."""
        X, y = classification_data
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            task_type="classification", random_state=42, verbose=0, n_jobs=1,
        )

        assert "best_genes" in result
        assert "best_rmsecv" in result  # 1 - accuracy for classification
        assert result["task_type"] == "classification"

    def test_legacy_ga_method_raises(self, synthetic_spectra_small):
        """method='ga' was removed in 2026-05-06; passing it must raise."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        with pytest.raises(ValueError, match="'ga' mode was removed"):
            optimize_preprocessing(
                X, y, method="ga", cv_folds=3, random_state=42, verbose=0,
            )


# =============================================================================
# Integration Tests - Convenience Function
# =============================================================================


@pytest.mark.integration
class TestConvenienceFunction:
    """Test convenience function for integration with search.py."""

    def test_get_optimized_preproc_config_quick(self, synthetic_spectra_small):
        """Quick mode should return valid preprocessing config."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        name, transform = get_optimized_preproc_config(
            X, y, quick=True, random_state=42, verbose=0
        )

        assert isinstance(name, str)
        assert transform is None or callable(transform)

    def test_get_optimized_preproc_config_full(self, synthetic_spectra_small):
        """Full mode should return valid preprocessing config."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        name, transform = get_optimized_preproc_config(
            X, y, quick=False, random_state=42, verbose=0
        )

        assert isinstance(name, str)
        assert transform is None or callable(transform)

        # Test that transform works
        if transform is not None:
            X_transformed = transform(X)
            assert X_transformed.shape == X.shape
            assert np.isfinite(X_transformed).all()


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """Exhaustive should handle small datasets."""
        X = np.random.randn(10, 20)
        y = np.random.randn(10)

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
        )

        assert "best_genes" in result

    def test_high_dimensional_data(self):
        """Exhaustive should handle high-dimensional data."""
        X = np.random.randn(50, 1000)
        y = np.random.randn(50)

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            random_state=42, verbose=0, n_jobs=1,
        )

        assert "best_genes" in result
