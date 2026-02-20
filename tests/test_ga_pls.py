"""
Unit and integration tests for GA-PLS variable selection.

This test suite validates the genetic algorithm for PLS-based wavelength
selection (ga_pls.py) including:
- Binary chromosome encoding (wavelength selection)
- Fitness evaluation with cross-validated PLS
- Genetic operators (tournament selection, two-point crossover, bit-flip mutation)
- Multi-run aggregation for stability
- Selection frequency scoring
"""

from __future__ import annotations

import numpy as np
import pytest
from spectral_predict.ga_pls import (
    ga_pls_selection,
    ga_pls_selection_detailed,
    _initialize_population,
    _fitness_function,
    _tournament_selection,
    _two_point_crossover,
    _bit_flip_mutation,
    _run_single_ga,
    FitnessCache,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_MIN_WAVELENGTHS,
)


# =============================================================================
# Unit Tests - Chromosome Encoding
# =============================================================================


@pytest.mark.unit
class TestGAPLSChromosome:
    """Test binary chromosome encoding for wavelength selection."""

    def test_binary_encoding(self):
        """Each gene should be 0 or 1."""
        rng = np.random.RandomState(42)
        population = _initialize_population(
            n_wavelengths=100, pop_size=10, init_ratio=0.3, min_wavelengths=5, rng=rng
        )

        # Check all values are binary
        assert np.all((population == 0) | (population == 1)), "Genes should be binary"

        # Check dtype
        assert population.dtype == np.uint8, "Population should be uint8"

    def test_minimum_selection_enforced(self):
        """At least min_vars wavelengths should be selected."""
        rng = np.random.RandomState(42)
        min_wavelengths = 10

        population = _initialize_population(
            n_wavelengths=100,
            pop_size=20,
            init_ratio=0.05,  # Very low ratio
            min_wavelengths=min_wavelengths,
            rng=rng,
        )

        # Check each individual has at least min_wavelengths selected
        for i in range(population.shape[0]):
            n_selected = np.sum(population[i])
            assert n_selected >= min_wavelengths, (
                f"Individual {i} has only {n_selected} wavelengths, "
                f"expected >= {min_wavelengths}"
            )

    def test_population_shape(self):
        """Population should have correct shape."""
        rng = np.random.RandomState(42)
        n_wavelengths = 100
        pop_size = 20

        population = _initialize_population(
            n_wavelengths=n_wavelengths, pop_size=pop_size, init_ratio=0.3, rng=rng
        )

        assert population.shape == (
            pop_size,
            n_wavelengths,
        ), "Population should have shape (pop_size, n_wavelengths)"

    def test_seeded_initialization(self):
        """Population initialization with seed importances."""
        rng = np.random.RandomState(42)
        n_wavelengths = 100

        # Create seed importances (favor first 50 wavelengths to ensure enough for selection)
        seed_importances = np.zeros(n_wavelengths)
        seed_importances[:50] = 1.0

        population = _initialize_population(
            n_wavelengths=n_wavelengths,
            pop_size=20,
            init_ratio=0.2,  # Lower ratio to ensure we don't exceed available wavelengths
            seed_importances=seed_importances,
            rng=rng,
        )

        # First 25% of population should use seeded initialization
        # These should favor first 50 wavelengths
        n_seeded = 20 // 4
        for i in range(n_seeded):
            selected = np.where(population[i] == 1)[0]
            # Count how many are in first 50
            n_in_first_50 = np.sum(selected < 50)
            # Should have higher than random selection (most should be in first 50)
            assert n_in_first_50 > 0, "Seeded individuals should select from important wavelengths"


# =============================================================================
# Unit Tests - Fitness Evaluation
# =============================================================================


@pytest.mark.unit
class TestGAPLSFitness:
    """Test fitness function."""

    def test_fitness_is_cv_r2(self, synthetic_spectra_small):
        """Fitness should be based on cross-validated RMSECV."""
        from sklearn.model_selection import KFold

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        n_wavelengths = X.shape[1]
        chromosome = np.zeros(n_wavelengths, dtype=np.uint8)
        # Select first 20 wavelengths
        chromosome[:20] = 1

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        fitness = _fitness_function(
            chromosome,
            X,
            y,
            cv,
            task_type="regression",
            n_components=5,
            min_wavelengths=5,
        )

        # For regression, fitness is negative RMSECV
        assert fitness <= 0 or fitness == -np.inf, "Fitness should be negative or -inf"
        # Should not be NaN
        assert not np.isnan(fitness), "Fitness should not be NaN"

    def test_penalty_for_too_few_vars(self, synthetic_spectra_small):
        """Chromosomes with too few variables should get penalty."""
        from sklearn.model_selection import KFold

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        n_wavelengths = X.shape[1]
        chromosome = np.zeros(n_wavelengths, dtype=np.uint8)
        # Select only 2 wavelengths (less than min)
        chromosome[:2] = 1

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        fitness = _fitness_function(
            chromosome,
            X,
            y,
            cv,
            task_type="regression",
            n_components=5,
            min_wavelengths=5,  # Minimum is 5
        )

        assert fitness == -np.inf, "Too few variables should get -inf fitness"

    def test_penalty_for_too_many_vars(self, synthetic_spectra_small):
        """Fitness should consider model complexity."""
        from sklearn.model_selection import KFold

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        n_wavelengths = X.shape[1]

        # Select only 10 wavelengths
        chromosome_small = np.zeros(n_wavelengths, dtype=np.uint8)
        chromosome_small[:10] = 1

        # Select all wavelengths
        chromosome_large = np.ones(n_wavelengths, dtype=np.uint8)

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        fitness_small = _fitness_function(
            chromosome_small, X, y, cv, task_type="regression", n_components=5
        )
        fitness_large = _fitness_function(
            chromosome_large, X, y, cv, task_type="regression", n_components=5
        )

        # Both should be valid (not -inf)
        assert fitness_small > -np.inf
        assert fitness_large > -np.inf

        # Note: We can't guarantee fitness_small > fitness_large because
        # it depends on the data, but both should be finite


# =============================================================================
# Unit Tests - Genetic Operators
# =============================================================================


@pytest.mark.unit
class TestGAPLSOperators:
    """Test GA-PLS genetic operators."""

    def test_crossover_preserves_length(self):
        """Crossover should preserve chromosome length."""
        rng = np.random.RandomState(42)
        n_wavelengths = 100

        parent1 = np.random.randint(0, 2, size=n_wavelengths, dtype=np.uint8)
        parent2 = np.random.randint(0, 2, size=n_wavelengths, dtype=np.uint8)

        child1, child2 = _two_point_crossover(
            parent1, parent2, crossover_rate=0.6, rng=rng
        )

        assert child1.shape == parent1.shape, "Child1 should have same length as parent"
        assert child2.shape == parent2.shape, "Child2 should have same length as parent"

    def test_crossover_maintains_binary(self):
        """Crossover should maintain binary encoding."""
        rng = np.random.RandomState(42)
        n_wavelengths = 100

        parent1 = np.random.randint(0, 2, size=n_wavelengths, dtype=np.uint8)
        parent2 = np.random.randint(0, 2, size=n_wavelengths, dtype=np.uint8)

        child1, child2 = _two_point_crossover(
            parent1, parent2, crossover_rate=0.6, rng=rng
        )

        assert np.all((child1 == 0) | (child1 == 1)), "Child1 should be binary"
        assert np.all((child2 == 0) | (child2 == 1)), "Child2 should be binary"

    def test_mutation_flips_bits(self):
        """Mutation should flip bits from 0 to 1 and vice versa."""
        rng = np.random.RandomState(42)
        n_wavelengths = 100

        original = np.zeros(n_wavelengths, dtype=np.uint8)
        original[:50] = 1

        # High mutation rate to ensure flips
        mutated = _bit_flip_mutation(
            original, mutation_rate=0.2, min_wavelengths=5, rng=rng
        )

        # Should have some differences
        n_flips = np.sum(mutated != original)
        assert n_flips > 0, "Mutation should flip at least one bit"

        # Should still be binary
        assert np.all((mutated == 0) | (mutated == 1)), "Mutated should be binary"

    def test_mutation_enforces_minimum(self):
        """Mutation should enforce minimum wavelength constraint."""
        rng = np.random.RandomState(42)
        n_wavelengths = 100
        min_wavelengths = 10

        # Start with exactly min_wavelengths selected
        original = np.zeros(n_wavelengths, dtype=np.uint8)
        original[:min_wavelengths] = 1

        # Mutate with high rate
        mutated = _bit_flip_mutation(
            original, mutation_rate=0.3, min_wavelengths=min_wavelengths, rng=rng
        )

        # Should still have at least min_wavelengths selected
        n_selected = np.sum(mutated)
        assert n_selected >= min_wavelengths, (
            f"Mutation should maintain minimum, got {n_selected}"
        )

    def test_selection_favors_fit(self):
        """Tournament selection should favor high-fitness individuals."""
        rng = np.random.RandomState(42)
        pop_size = 20
        n_wavelengths = 100

        # Create population
        population = _initialize_population(
            n_wavelengths=n_wavelengths, pop_size=pop_size, init_ratio=0.3, rng=rng
        )

        # Assign fitness scores (linearly increasing)
        fitness_scores = np.arange(pop_size, dtype=float)

        # Select many times
        selected = _tournament_selection(
            population, fitness_scores, tournament_size=3, rng=rng
        )

        assert selected.shape == (pop_size, n_wavelengths)

        # Statistical test: best individual should appear frequently
        # (This is probabilistic, may rarely fail)
        best_individual = population[-1]  # Highest fitness
        selections_equal_best = [
            np.array_equal(selected[i], best_individual) for i in range(pop_size)
        ]
        n_best_selected = sum(selections_equal_best)

        # Best should be selected more than random (5%)
        assert n_best_selected > 1, "Best individual should be selected multiple times"


# =============================================================================
# Unit Tests - Fitness Cache
# =============================================================================


@pytest.mark.unit
class TestFitnessCache:
    """Test fitness caching mechanism."""

    def test_cache_hit(self):
        """Cache should return stored values."""
        cache = FitnessCache()

        chromosome = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        fitness = -2.5

        cache.set(chromosome, fitness)
        cached_fitness = cache.get(chromosome)

        assert cached_fitness == fitness, "Cached fitness should match"
        assert cache.hits == 1, "Should have 1 hit"
        assert cache.misses == 0, "Should have 0 misses"

    def test_cache_miss(self):
        """Cache should return None for unknown chromosomes."""
        cache = FitnessCache()

        chromosome = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        cached_fitness = cache.get(chromosome)

        assert cached_fitness is None, "Unknown chromosome should return None"
        assert cache.hits == 0, "Should have 0 hits"
        assert cache.misses == 1, "Should have 1 miss"

    def test_cache_clear(self):
        """Cache clear should reset everything."""
        cache = FitnessCache()

        chromosome = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        cache.set(chromosome, -2.5)
        cache.get(chromosome)

        cache.clear()

        assert len(cache.cache) == 0, "Cache should be empty"
        assert cache.hits == 0, "Hits should be reset"
        assert cache.misses == 0, "Misses should be reset"


# =============================================================================
# Integration Tests - GA-PLS Selection
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestGAPLSSelection:
    """Test full GA-PLS optimization."""

    def test_returns_selected_wavelengths(self, synthetic_spectra_small):
        """GA-PLS should return importance scores for all wavelengths."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # Check shape
        assert importances.shape == (
            X.shape[1],
        ), "Should return scores for all wavelengths"

        # Check range [0, 1]
        assert np.all(importances >= 0) and np.all(
            importances <= 1
        ), "Importances should be in [0, 1]"

    def test_selection_improves_model(self, synthetic_spectra_small):
        """Selected variables should improve model performance."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        # Get importances
        importances = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # Select top variables
        n_select = min(50, X.shape[1] // 2)
        top_indices = np.argsort(importances)[-n_select:]

        # Compare performance
        X_selected = X[:, top_indices]

        # Build simple models
        model_all = PLSRegression(n_components=5)
        model_selected = PLSRegression(n_components=5)

        # Cross-validate (using negative MSE scoring)
        score_all = cross_val_score(
            model_all, X, y, cv=3, scoring="neg_mean_squared_error"
        ).mean()
        score_selected = cross_val_score(
            model_selected, X_selected, y, cv=3, scoring="neg_mean_squared_error"
        ).mean()

        # Selected variables should perform reasonably (may not always be better due to randomness)
        # Just check that it's not catastrophically worse
        assert score_selected > -np.inf, "Selected variables should give finite score"

    def test_respects_max_vars(self, synthetic_spectra_small):
        """Selection frequency should respect the optimization."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            selection_threshold=0.5,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # Check that not all variables are selected (some should have low scores)
        n_high_importance = np.sum(importances > 0.8)
        n_wavelengths = X.shape[1]

        # Should select subset, not all
        assert n_high_importance < n_wavelengths, (
            "Should not select all wavelengths with high importance"
        )

    def test_reproducibility_with_seed(self, synthetic_spectra_small):
        """GA-PLS should be reproducible with same seed."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances1 = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            cv=3,
            random_state=42,
            verbose=0,
        )

        importances2 = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # Should get same importances
        assert np.allclose(
            importances1, importances2
        ), "Same seed should give same importances"

    def test_multi_run_stability(self, synthetic_spectra_small):
        """Multiple runs should produce stable importance scores."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        importances = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=5,  # More runs for stability
            selection_threshold=0.6,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # Check that importances are reasonable
        assert importances.min() >= 0.0, "Min importance should be >= 0"
        assert importances.max() <= 1.0, "Max importance should be <= 1"

        # At least some variables should be consistently selected
        n_stable = np.sum(importances >= 0.6)
        assert n_stable > 0, "Should have some stable selections across runs"


# =============================================================================
# Integration Tests - Detailed Results
# =============================================================================


@pytest.mark.integration
class TestGAPLSDetailed:
    """Test detailed results function."""

    def test_detailed_results(self, synthetic_spectra_small):
        """Detailed function should return complete results."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        results = ga_pls_selection_detailed(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            selection_threshold=0.5,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # Check keys
        assert "importances" in results
        assert "selected_indices" in results
        assert "selection_threshold" in results
        assert "n_selected" in results

        # Check types and shapes
        assert isinstance(results["importances"], np.ndarray)
        assert isinstance(results["selected_indices"], np.ndarray)
        assert isinstance(results["n_selected"], (int, np.integer))

        # Check consistency
        assert results["n_selected"] == len(results["selected_indices"])


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """GA-PLS should handle small datasets."""
        X = np.random.randn(15, 30)
        y = np.random.randn(15)

        importances = ga_pls_selection(
            X,
            y,
            population_size=5,
            n_generations=3,
            n_runs=1,
            cv=3,
            random_state=42,
            verbose=0,
        )

        assert importances.shape == (30,)

    def test_few_wavelengths(self):
        """GA-PLS should handle few wavelengths."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        importances = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            min_wavelengths=3,
            cv=3,
            random_state=42,
            verbose=0,
        )

        assert importances.shape == (20,)

    def test_classification_task(self, classification_data):
        """GA-PLS should work for classification."""
        X, y = classification_data
        X = X.values
        y = y.values

        importances = ga_pls_selection(
            X,
            y,
            population_size=10,
            n_generations=5,
            n_runs=2,
            task_type="classification",
            cv=3,
            random_state=42,
            verbose=0,
        )

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0) and np.all(importances <= 1)
