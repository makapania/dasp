"""
Test to diagnose GA + CARS interaction issue.

When CARS and GA are run together, GA produces much worse results than when run alone.
This test attempts to reproduce and diagnose the issue.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression


def generate_spectral_data(n_samples=100, n_wavelengths=200, n_informative=20, noise=0.1, random_state=42):
    """Generate synthetic spectral data with known informative wavelengths."""
    rng = np.random.RandomState(random_state)

    # Generate base spectra
    X = rng.randn(n_samples, n_wavelengths)

    # Create target based on specific wavelengths
    informative_indices = np.linspace(10, n_wavelengths - 10, n_informative, dtype=int)
    coefficients = rng.randn(n_informative)
    y = X[:, informative_indices] @ coefficients + noise * rng.randn(n_samples)

    return X, y, informative_indices


class TestGACARSInteraction:
    """Test that GA produces consistent results regardless of CARS execution."""

    @pytest.fixture
    def spectral_data(self):
        """Generate test data."""
        return generate_spectral_data(n_samples=50, n_wavelengths=100, n_informative=10)

    def test_ga_results_are_reproducible(self, spectral_data):
        """Test that GA produces identical results when run twice."""
        from spectral_predict.ga_pls import ga_pls_selection

        X, y, _ = spectral_data

        # Run GA twice with same settings
        importances1 = ga_pls_selection(
            X, y,
            population_size=16,
            n_generations=25,
            n_runs=1,
            n_components=5,
            cv=3,
            random_state=42,
            verbose=0
        )

        importances2 = ga_pls_selection(
            X, y,
            population_size=16,
            n_generations=25,
            n_runs=1,
            n_components=5,
            cv=3,
            random_state=42,
            verbose=0
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            importances1, importances2,
            err_msg="GA should produce identical results with same random_state"
        )

    def test_compare_ga_alone_vs_after_cars(self, spectral_data):
        """Compare GA results when run alone vs after CARS."""
        from spectral_predict.ga_pls import ga_pls_selection
        from spectral_predict.variable_selection import cars_selection

        X, y, _ = spectral_data

        # Run GA alone
        print("\n=== GA ALONE ===")
        ga_alone = ga_pls_selection(
            X.copy(), y.copy(),  # Use copies to ensure isolation
            population_size=16,
            n_generations=25,
            n_runs=1,
            n_components=5,
            cv=3,
            random_state=42,
            verbose=1
        )

        # Run CARS then GA
        print("\n=== CARS FIRST ===")
        _ = cars_selection(
            X.copy(), y.copy(),
            n_iterations=30,
            pls_components=5,
            cv_folds=3,
            random_state=42
        )

        print("\n=== GA AFTER CARS ===")
        ga_after_cars = ga_pls_selection(
            X.copy(), y.copy(),
            population_size=16,
            n_generations=25,
            n_runs=1,
            n_components=5,
            cv=3,
            random_state=42,
            verbose=1
        )

        # Compare
        print("\n=== COMPARISON ===")
        print(f"GA alone - max importance: {np.max(ga_alone):.4f}")
        print(f"GA after CARS - max importance: {np.max(ga_after_cars):.4f}")

        # These should be very similar (not necessarily identical due to CV randomness)
        correlation = np.corrcoef(ga_alone, ga_after_cars)[0, 1]
        print(f"Correlation between results: {correlation:.4f}")

        # If GA is working correctly, correlation should be high
        assert correlation > 0.8, f"GA results should be highly correlated, got {correlation:.4f}"


class TestSearchLoopSimulation:
    """Simulate the actual search loop to find the bug."""

    def test_simulate_search_loop(self):
        """Simulate the search.py loop structure."""
        from spectral_predict.ga_pls import ga_pls_selection
        from spectral_predict.variable_selection import cars_selection
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import Pipeline

        # Generate data
        X, y, true_informative = generate_spectral_data(
            n_samples=50, n_wavelengths=100, n_informative=10
        )

        # Simulate search loop structure
        selected_methods = ['cars', 'ga']

        # Create model (as in search.py line 994)
        model = PLSRegression(n_components=5)

        # Build and fit pipeline (as in search.py lines 1084-1089)
        pipe_steps = [("model", model)]
        pipe = Pipeline(pipe_steps)
        pipe.fit(X, y)

        # Get fitted model (as in search.py line 1092)
        fitted_model = pipe.named_steps["model"]

        # X_transformed_varsel = X_for_models (as in search.py line 1095)
        X_transformed_varsel = X

        results = {}

        for varsel_method in selected_methods:
            print(f"\n--- Running {varsel_method} ---")

            if varsel_method == 'cars':
                importances = cars_selection(
                    X_transformed_varsel, y,
                    n_iterations=30,
                    pls_components=5,
                    cv_folds=3,
                    random_state=42
                )
            elif varsel_method == 'ga':
                # This is what happens in search.py - same data reference
                importances = ga_pls_selection(
                    X_transformed_varsel, y,
                    population_size=16,
                    n_generations=25,
                    n_runs=1,
                    n_components=5,
                    cv=3,
                    random_state=42,
                    verbose=1
                )

            results[varsel_method] = importances.copy()
            print(f"{varsel_method} max importance: {np.max(importances):.4f}")

        # Now run GA in isolation for comparison
        print("\n--- GA in complete isolation ---")
        X_fresh, y_fresh, _ = generate_spectral_data(
            n_samples=50, n_wavelengths=100, n_informative=10
        )

        ga_isolated = ga_pls_selection(
            X_fresh, y_fresh,
            population_size=16,
            n_generations=25,
            n_runs=1,
            n_components=5,
            cv=3,
            random_state=42,
            verbose=1
        )

        print(f"\nGA isolated max importance: {np.max(ga_isolated):.4f}")
        print(f"GA in loop max importance: {np.max(results['ga']):.4f}")

        # The data is generated fresh with same seed, so results should match
        # if GA is working correctly
        correlation = np.corrcoef(results['ga'], ga_isolated)[0, 1]
        print(f"Correlation: {correlation:.4f}")


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-s", "-k", "test_simulate"])
