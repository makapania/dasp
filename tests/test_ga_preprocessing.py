"""
Unit and integration tests for GA-based preprocessing optimization.

This test suite validates the genetic algorithm for preprocessing parameter
optimization (ga_preprocessing.py) including:
- Chromosome encoding and decoding
- Fitness evaluation
- Genetic operators (selection, crossover, mutation)
- Full optimization workflow
- Integration with preprocessing pipeline
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
        tournament_selection,
        crossover,
        mutate,
        optimize_preprocessing,
        get_optimized_preproc_config,
        PREPROC_TYPES,
        WINDOW_SIZES,
        BASELINE_METHODS,
        BASELINE_LAMBDAS,
        SMOOTHING_OPTIONS,
        SMOOTHING_WINDOWS,
        N_GENES,
    )
    HAS_CATBOOST = True
except (ImportError, ModuleNotFoundError):
    HAS_CATBOOST = False

pytestmark = pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")


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
        assert 0 <= chrom[2] < len(BASELINE_METHODS), "Invalid baseline method"
        assert 0 <= chrom[3] < len(BASELINE_LAMBDAS), "Invalid baseline lambda"
        assert 0 <= chrom[4] < len(SMOOTHING_OPTIONS), "Invalid smoothing option"
        assert 0 <= chrom[5] < len(SMOOTHING_WINDOWS), "Invalid smoothing window"

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
        # Test specific configurations
        test_configs = [
            np.array([0, 0, 0, 0, 0, 0], dtype=np.int32),  # raw, no processing
            np.array([1, 0, 0, 0, 0, 0], dtype=np.int32),  # snv only
            np.array([2, 1, 0, 0, 0, 0], dtype=np.int32),  # deriv1 with window
            np.array([0, 0, 1, 0, 0, 0], dtype=np.int32),  # raw + baseline
            np.array([0, 0, 0, 0, 1, 0], dtype=np.int32),  # raw + smoothing
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
        # (depends on implementation, but should not crash)
        chrom = np.array([9, 0, 3, 7, 1, 8], dtype=np.int32)  # deriv4 + airpls + smoothing

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


@pytest.mark.unit
class TestGeneticOperators:
    """Test genetic operators."""

    def test_tournament_selection_returns_parent(self):
        """Tournament selection should return a valid parent."""
        rng = np.random.RandomState(42)

        # Create population
        pop_size = 10
        population = np.array([random_chromosome(rng) for _ in range(pop_size)])
        fitness = np.random.randn(pop_size)

        # Select parent
        parent = tournament_selection(population, fitness, tournament_size=3, rng=rng)

        assert parent.shape == (N_GENES,), "Parent should have correct shape"
        assert parent.dtype == np.int32, "Parent should be integer array"

    def test_tournament_favors_high_fitness(self):
        """Tournament selection should favor high fitness individuals."""
        rng = np.random.RandomState(42)

        # Create population with known fitness
        pop_size = 10
        population = np.array([random_chromosome(rng) for _ in range(pop_size)])
        fitness = np.arange(pop_size, dtype=float)  # 0, 1, 2, ..., 9

        # Select many times
        selections = [
            tournament_selection(population, fitness, tournament_size=3, rng=rng)
            for _ in range(100)
        ]

        # Check that high-fitness individuals are selected more often
        # (statistical test - may occasionally fail)
        # Compare bytes to identify which individual was selected
        best_individual = population[-1]  # Highest fitness
        best_count = sum(1 for s in selections if np.array_equal(s, best_individual))

        # Best individual should be selected more than random chance (10%)
        assert best_count > 10, "High fitness individual should be selected frequently"

    def test_crossover_preserves_length(self):
        """Crossover should preserve chromosome length."""
        rng = np.random.RandomState(42)

        parent1 = random_chromosome(rng)
        parent2 = random_chromosome(rng)

        child1, child2 = crossover(parent1, parent2, crossover_rate=0.7, rng=rng)

        assert child1.shape == parent1.shape, "Child1 should have same shape as parent"
        assert child2.shape == parent2.shape, "Child2 should have same shape as parent"

    def test_crossover_with_zero_rate(self):
        """Crossover with rate=0 should return copies of parents."""
        rng = np.random.RandomState(42)

        parent1 = random_chromosome(rng)
        parent2 = random_chromosome(rng)

        child1, child2 = crossover(parent1, parent2, crossover_rate=0.0, rng=rng)

        assert np.array_equal(child1, parent1), "Child1 should equal parent1 with rate=0"
        assert np.array_equal(child2, parent2), "Child2 should equal parent2 with rate=0"

    def test_mutation_flips_genes(self):
        """Mutation should modify genes."""
        rng = np.random.RandomState(42)

        original = random_chromosome(rng)

        # High mutation rate to ensure changes
        mutated = mutate(original, mutation_rate=0.5, rng=rng)

        assert mutated.shape == original.shape, "Mutation should preserve shape"

        # With 50% mutation rate, should have some changes
        changes = np.sum(mutated != original)
        assert changes > 0, "Mutation should change at least one gene"

    def test_mutation_respects_ranges(self):
        """Mutated genes should stay in valid ranges."""
        rng = np.random.RandomState(42)

        original = random_chromosome(rng)
        mutated = mutate(original, mutation_rate=0.8, rng=rng)

        assert 0 <= mutated[0] < len(PREPROC_TYPES)
        assert 0 <= mutated[1] < len(WINDOW_SIZES)
        assert 0 <= mutated[2] < len(BASELINE_METHODS)
        assert 0 <= mutated[3] < len(BASELINE_LAMBDAS)
        assert 0 <= mutated[4] < len(SMOOTHING_OPTIONS)
        assert 0 <= mutated[5] < len(SMOOTHING_WINDOWS)


# =============================================================================
# Integration Tests - GA Optimization
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestGAPreprocessingOptimization:
    """Test full GA preprocessing optimization."""

    def test_basic_optimization(self, synthetic_spectra_small):
        """GA should complete and return valid result."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=5,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        # Check result structure
        assert "best_genes" in result
        assert "best_name" in result
        assert "best_transform" in result
        assert "best_rmsecv" in result
        assert "best_config" in result
        assert "history" in result

        # Check best_genes
        assert result["best_genes"].shape == (N_GENES,)
        assert isinstance(result["best_name"], str)
        assert result["best_transform"] is None or callable(result["best_transform"])

        # Check history
        assert len(result["history"]) == 6, "History should have 6 entries (gen 0-5)"

    def test_fitness_improves(self, synthetic_spectra_small):
        """Later generations should have better or equal fitness."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=10,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        history = result["history"]
        best_fitness_values = [gen["best_fitness"] for gen in history]

        # Check that fitness generally improves (monotonically non-decreasing)
        for i in range(1, len(best_fitness_values)):
            assert (
                best_fitness_values[i] >= best_fitness_values[i - 1]
            ), f"Fitness should not decrease from gen {i-1} to {i}"

    def test_best_preprocessing_identified(self, synthetic_spectra_small):
        """GA should identify a reasonable preprocessing configuration."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=10,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        # Apply best preprocessing
        if result["best_transform"] is not None:
            X_preprocessed = result["best_transform"](X)

            # Check output is valid
            assert X_preprocessed.shape == X.shape
            assert np.isfinite(X_preprocessed).all()

    def test_respects_search_space(self, synthetic_spectra_small):
        """GA should only produce valid preprocessing combinations."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=5,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        # Check best genes are in valid ranges
        genes = result["best_genes"]
        assert 0 <= genes[0] < len(PREPROC_TYPES)
        assert 0 <= genes[1] < len(WINDOW_SIZES)
        assert 0 <= genes[2] < len(BASELINE_METHODS)
        assert 0 <= genes[3] < len(BASELINE_LAMBDAS)
        assert 0 <= genes[4] < len(SMOOTHING_OPTIONS)
        assert 0 <= genes[5] < len(SMOOTHING_WINDOWS)

    def test_reproducibility_with_seed(self, synthetic_spectra_small):
        """GA should be reproducible with same random seed."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result1 = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=5,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        result2 = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=5,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        # Should get same best genes
        assert np.array_equal(
            result1["best_genes"], result2["best_genes"]
        ), "Same seed should give same result"

    def test_classification_task(self, classification_data):
        """GA should work for classification tasks."""
        X, y = classification_data
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=5,
            cv_folds=3,
            n_components=5,
            task_type="classification",
            random_state=42,
            verbose=0,
        )

        assert "best_genes" in result
        assert "best_rmsecv" in result  # Actually 1-accuracy for classification
        assert result["task_type"] == "classification"

    def test_elitism_preserves_best(self, synthetic_spectra_small):
        """Elitism should preserve best solutions across generations."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            population_size=10,
            n_generations=10,
            elitism=2,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        # With elitism, fitness should never decrease
        history = result["history"]
        best_fitness_values = [gen["best_fitness"] for gen in history]

        for i in range(1, len(best_fitness_values)):
            assert best_fitness_values[i] >= best_fitness_values[i - 1], (
                "Elitism should preserve best fitness"
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
        """GA should handle small datasets."""
        X = np.random.randn(10, 20)
        y = np.random.randn(10)

        result = optimize_preprocessing(
            X,
            y,
            population_size=5,
            n_generations=3,
            cv_folds=3,
            n_components=3,
            random_state=42,
            verbose=0,
        )

        assert "best_genes" in result

    def test_high_dimensional_data(self):
        """GA should handle high-dimensional data."""
        X = np.random.randn(50, 1000)
        y = np.random.randn(50)

        result = optimize_preprocessing(
            X,
            y,
            population_size=5,
            n_generations=3,
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
        )

        assert "best_genes" in result
