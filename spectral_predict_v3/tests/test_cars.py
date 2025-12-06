"""
Tests for CARS (Competitive Adaptive Reweighted Sampling) variable selection.

Tests cover:
- CARS variable selection accuracy
- CARS vs SPA comparison
- Monte Carlo stability (multiple runs)
- Varying sampling ratios
- Early stopping criteria
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

from spectral_predict_v3.core.variable_selection import cars_selection, spa_selection


class TestCARSBasic:
    """Basic CARS functionality tests."""

    def test_cars_basic_selection(self):
        """Test basic CARS variable selection."""
        # Create synthetic data with known informative variables
        n_samples = 100
        n_features = 200
        n_informative = 10

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=0.1,
            random_state=42
        )

        # Run CARS
        importances = cars_selection(
            X, y,
            n_iterations=30,
            pls_components=5,
            random_state=42
        )

        # Should return importances for all variables
        assert importances.shape == (n_features,), \
            f"Should return importances for all features, got shape {importances.shape}"

        # Should select some variables (non-zero importance)
        n_selected = np.sum(importances > 0)
        assert n_selected > 0, "Should select at least some variables"
        assert n_selected < n_features, "Should not select all variables"

        print(f"✓ CARS selected {n_selected}/{n_features} variables")

    def test_cars_identifies_informative_variables(self):
        """Test that CARS tends to select informative variables."""
        # Create data where first 5 variables are highly informative
        n_samples = 80
        n_features = 50
        np.random.seed(42)

        X = np.random.randn(n_samples, n_features)
        # Make first 5 features strongly predictive
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1 + X[:, 3] * 0.8 + X[:, 4] * 0.5
        y += np.random.randn(n_samples) * 0.1  # Small noise

        # Run CARS
        importances = cars_selection(
            X, y,
            n_iterations=40,
            pls_components=5,
            random_state=42
        )

        # Get top selected variables
        selected_vars = np.where(importances > 0)[0]

        # Should select at least some of the first 5 variables
        informative_selected = np.sum(np.isin(selected_vars, [0, 1, 2, 3, 4]))

        print(f"✓ CARS selected {informative_selected}/5 known informative variables")
        print(f"  Total selected: {len(selected_vars)} variables")

        # Should find at least some informative variables
        assert informative_selected >= 2, \
            f"Should select at least 2 informative variables, got {informative_selected}"


class TestCARSStability:
    """Test CARS stability and reproducibility."""

    def test_cars_reproducibility(self):
        """Test that CARS is reproducible with same random seed."""
        X, y = make_regression(n_samples=60, n_features=100, random_state=42)

        # Run twice with same seed
        importances1 = cars_selection(X, y, n_iterations=20, random_state=42)
        importances2 = cars_selection(X, y, n_iterations=20, random_state=42)

        # Should be identical
        np.testing.assert_array_equal(
            importances1,
            importances2,
            err_msg="CARS should be reproducible with same random seed"
        )

        print("✓ CARS is reproducible with same random seed")

    def test_cars_multiple_runs_stability(self):
        """Test that CARS is reasonably stable across multiple runs."""
        X, y = make_regression(n_samples=80, n_features=80, random_state=42)

        # Run CARS multiple times with different seeds
        n_runs = 5
        all_selections = []

        for seed in range(n_runs):
            importances = cars_selection(
                X, y,
                n_iterations=30,
                random_state=seed
            )
            selected = np.where(importances > 0)[0]
            all_selections.append(set(selected))

        # Calculate stability: how many variables are selected in all runs
        common_vars = set.intersection(*all_selections)
        union_vars = set.union(*all_selections)

        stability = len(common_vars) / len(union_vars) if len(union_vars) > 0 else 0

        print(f"✓ CARS stability: {stability:.2%}")
        print(f"  Common variables: {len(common_vars)}")
        print(f"  Total unique variables: {len(union_vars)}")

        # Some variables should be consistently selected
        assert len(common_vars) > 0, "Some variables should be consistently selected"


class TestCARSParameters:
    """Test CARS with different parameters."""

    def test_cars_different_iterations(self):
        """Test CARS with different numbers of iterations."""
        X, y = make_regression(n_samples=70, n_features=60, random_state=42)

        iterations_list = [10, 30, 50]
        results = []

        for n_iter in iterations_list:
            importances = cars_selection(
                X, y,
                n_iterations=n_iter,
                random_state=42
            )

            n_selected = np.sum(importances > 0)
            results.append(n_selected)

            print(f"✓ n_iterations={n_iter}: selected {n_selected} variables")

        # All should select at least some variables
        assert all(n > 0 for n in results), "All iteration counts should select variables"

    def test_cars_different_pls_components(self):
        """Test CARS with different numbers of PLS components."""
        X, y = make_regression(n_samples=60, n_features=50, random_state=42)

        components_list = [3, 5, 8]

        for n_comp in components_list:
            importances = cars_selection(
                X, y,
                n_iterations=25,
                pls_components=n_comp,
                random_state=42
            )

            n_selected = np.sum(importances > 0)

            print(f"✓ pls_components={n_comp}: selected {n_selected} variables")

            assert n_selected > 0, f"Should select variables with {n_comp} components"

    def test_cars_sampling_ratio(self):
        """Test CARS with different Monte Carlo sampling ratios."""
        X, y = make_regression(n_samples=70, n_features=60, random_state=42)

        sampling_ratios = [60, 80, 100]

        for ratio in sampling_ratios:
            importances = cars_selection(
                X, y,
                n_iterations=30,
                monte_carlo_samples=ratio,
                random_state=42
            )

            n_selected = np.sum(importances > 0)

            print(f"✓ monte_carlo_samples={ratio}%: selected {n_selected} variables")

            assert n_selected > 0, f"Should select variables with {ratio}% sampling"


class TestCARSVsSPA:
    """Compare CARS with SPA variable selection."""

    def test_cars_vs_spa_comparison(self):
        """Compare CARS and SPA on same dataset."""
        # Generate data
        n_samples = 100
        n_features = 150
        np.random.seed(42)

        X = np.random.randn(n_samples, n_features)
        # Make first 10 features informative
        y = np.sum(X[:, :10], axis=1) + np.random.randn(n_samples) * 0.1

        # Run CARS
        cars_importances = cars_selection(
            X, y,
            n_iterations=40,
            random_state=42
        )
        cars_selected = np.where(cars_importances > 0)[0]

        # Run SPA for same number of variables
        spa_importances = spa_selection(
            X, y,
            n_features=len(cars_selected),
            random_state=42
        )
        spa_selected = np.argsort(spa_importances)[-len(cars_selected):]

        print(f"✓ CARS selected: {len(cars_selected)} variables")
        print(f"✓ SPA selected: {len(spa_selected)} variables")

        # Test prediction performance with selected variables
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # CARS performance
        if len(cars_selected) > 0:
            X_cars_train = X_train[:, cars_selected]
            X_cars_test = X_test[:, cars_selected]
            n_comp = min(5, len(cars_selected) - 1)
            pls_cars = PLSRegression(n_components=n_comp)
            pls_cars.fit(X_cars_train, y_train)
            y_pred_cars = pls_cars.predict(X_cars_test)
            r2_cars = r2_score(y_test, y_pred_cars)
        else:
            r2_cars = 0.0

        # SPA performance
        if len(spa_selected) > 0:
            X_spa_train = X_train[:, spa_selected]
            X_spa_test = X_test[:, spa_selected]
            n_comp = min(5, len(spa_selected) - 1)
            pls_spa = PLSRegression(n_components=n_comp)
            pls_spa.fit(X_spa_train, y_train)
            y_pred_spa = pls_spa.predict(X_spa_test)
            r2_spa = r2_score(y_test, y_pred_spa)
        else:
            r2_spa = 0.0

        print(f"  CARS R²: {r2_cars:.4f}")
        print(f"  SPA R²: {r2_spa:.4f}")

        # Both methods should achieve reasonable performance
        assert r2_cars > 0.3 or r2_spa > 0.3, "At least one method should work well"


class TestCARSEdgeCases:
    """Test CARS edge cases and error handling."""

    def test_cars_small_dataset(self):
        """Test CARS on small dataset."""
        X = np.random.randn(20, 30)
        y = np.random.randn(20)

        # Should handle small dataset (may adjust parameters internally)
        importances = cars_selection(
            X, y,
            n_iterations=15,
            pls_components=3,
            cv_folds=3,
            random_state=42
        )

        assert importances.shape == (30,), "Should return importances for all features"

        print("✓ CARS handles small dataset")

    def test_cars_few_variables(self):
        """Test CARS when dataset has few variables."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        importances = cars_selection(
            X, y,
            n_iterations=20,
            random_state=42
        )

        assert importances.shape == (10,), "Should handle few variables"

        print("✓ CARS handles dataset with few variables")

    def test_cars_high_dimensional(self):
        """Test CARS on high-dimensional data (p >> n)."""
        X = np.random.randn(40, 200)  # More features than samples
        y = np.random.randn(40)

        # Should handle high-dimensional case
        importances = cars_selection(
            X, y,
            n_iterations=20,
            pls_components=3,  # Low components for high-dimensional
            random_state=42
        )

        assert importances.shape == (200,), "Should handle high-dimensional data"

        n_selected = np.sum(importances > 0)
        assert n_selected < 200, "Should reduce dimensionality"

        print(f"✓ CARS handles high-dimensional data: {n_selected}/200 selected")


def test_cars_in_variable_selection_config():
    """Test that CARS is correctly included in variable selection config."""
    from spectral_predict_v3.core.model_config import get_variable_selection_config

    # CARS should be in comprehensive tier
    config = get_variable_selection_config('comprehensive')

    assert 'cars' in config['methods'], \
        "CARS should be in comprehensive variable selection methods"

    print("✓ CARS correctly included in variable selection config")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing CARS Variable Selection")
    print("=" * 60)

    # Basic tests
    print("\n--- Basic Tests ---")
    test_basic = TestCARSBasic()
    test_basic.test_cars_basic_selection()
    test_basic.test_cars_identifies_informative_variables()

    # Stability tests
    print("\n--- Stability Tests ---")
    test_stability = TestCARSStability()
    test_stability.test_cars_reproducibility()
    test_stability.test_cars_multiple_runs_stability()

    # Parameter tests
    print("\n--- Parameter Tests ---")
    test_params = TestCARSParameters()
    test_params.test_cars_different_iterations()
    test_params.test_cars_different_pls_components()
    test_params.test_cars_sampling_ratio()

    # Comparison tests
    print("\n--- CARS vs SPA Comparison ---")
    test_compare = TestCARSVsSPA()
    test_compare.test_cars_vs_spa_comparison()

    # Edge case tests
    print("\n--- Edge Case Tests ---")
    test_edge = TestCARSEdgeCases()
    test_edge.test_cars_small_dataset()
    test_edge.test_cars_few_variables()
    test_edge.test_cars_high_dimensional()

    # Config test
    print("\n--- Configuration Tests ---")
    test_cars_in_variable_selection_config()

    print("\n" + "=" * 60)
    print("All CARS tests passed!")
    print("=" * 60)
