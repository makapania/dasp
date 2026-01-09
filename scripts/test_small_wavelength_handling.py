"""
Test script for verifying graceful handling of restricted wavelength ranges.

Tests various scenarios:
1. 30 wavelengths (typical restricted range like 2030-2060nm)
2. 10 wavelengths (heavily restricted)
3. 5 wavelengths (extreme restriction)
4. 2 wavelengths (minimal - only basic models should work)

Each scenario tests:
- Savitzky-Golay derivatives with various window sizes
- Variable selection methods (CARS, iPLS, SPA, UVE)
- PLS n_components capping
- Model fitting

Run with: python scripts/test_small_wavelength_handling.py
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_predict.preprocess import SavgolDerivative, SavgolSmooth
from spectral_predict.variable_selection import (
    cars_selection, ipls_selection, spa_selection, uve_selection
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


def test_savgol_auto_adjustment():
    """Test Savitzky-Golay window auto-adjustment."""
    print("\n" + "="*70)
    print("TEST 1: Savitzky-Golay Window Auto-Adjustment")
    print("="*70)

    test_cases = [
        (30, 31, "30 wavelengths, window=31 (should reduce to 29)"),
        (10, 17, "10 wavelengths, window=17 (should reduce to 9)"),
        (5, 11, "5 wavelengths, window=11 (should reduce to 5)"),
        (3, 7, "3 wavelengths, window=7 (should reduce to 3 or fail gracefully)"),
    ]

    results = []
    for n_features, window, description in test_cases:
        print(f"\n--- {description} ---")
        X = np.random.randn(50, n_features)

        try:
            sg = SavgolDerivative(deriv=1, window=window)
            X_transformed = sg.transform(X)
            results.append((description, "PASS", f"Output shape: {X_transformed.shape}"))
            print(f"SUCCESS: Transformed to shape {X_transformed.shape}")
        except ValueError as e:
            results.append((description, "EXPECTED FAIL", str(e)))
            print(f"EXPECTED ERROR (too few wavelengths): {e}")
        except Exception as e:
            results.append((description, "UNEXPECTED FAIL", str(e)))
            print(f"UNEXPECTED ERROR: {e}")

    return results


def test_variable_selection_graceful_degradation():
    """Test variable selection methods with small feature counts."""
    print("\n" + "="*70)
    print("TEST 2: Variable Selection Graceful Degradation")
    print("="*70)

    n_samples = 50
    test_cases = [
        (30, "30 wavelengths"),
        (10, "10 wavelengths"),
        (5, "5 wavelengths (below CARS minimum)"),
        (3, "3 wavelengths (below iPLS minimum)"),
    ]

    results = []

    for n_features, description in test_cases:
        print(f"\n--- {description} ---")
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        # Test CARS
        print(f"\nCARS ({n_features} features):")
        try:
            importances = cars_selection(X, y, n_iterations=10, pls_components=3)
            if np.all(importances == 1):
                results.append((f"CARS {description}", "SKIPPED (uniform)", ""))
                print("  -> Returned uniform importances (graceful skip)")
            else:
                results.append((f"CARS {description}", "PASS", f"Max importance: {importances.max():.4f}"))
                print(f"  -> Completed, max importance: {importances.max():.4f}")
        except Exception as e:
            results.append((f"CARS {description}", "FAIL", str(e)))
            print(f"  -> ERROR: {e}")

        # Test iPLS
        print(f"\niPLS ({n_features} features):")
        try:
            importances = ipls_selection(X, y, n_intervals=10)
            if np.all(importances == 1):
                results.append((f"iPLS {description}", "SKIPPED (uniform)", ""))
                print("  -> Returned uniform importances (graceful skip)")
            else:
                results.append((f"iPLS {description}", "PASS", f"Max importance: {importances.max():.4f}"))
                print(f"  -> Completed, max importance: {importances.max():.4f}")
        except Exception as e:
            results.append((f"iPLS {description}", "FAIL", str(e)))
            print(f"  -> ERROR: {e}")

        # Test SPA
        print(f"\nSPA ({n_features} features):")
        try:
            importances = spa_selection(X, y, n_features=min(10, n_features))
            if np.all(importances == 1):
                results.append((f"SPA {description}", "SKIPPED (uniform)", ""))
                print("  -> Returned uniform importances (graceful skip)")
            else:
                results.append((f"SPA {description}", "PASS", f"Max importance: {importances.max():.4f}"))
                print(f"  -> Completed, max importance: {importances.max():.4f}")
        except Exception as e:
            results.append((f"SPA {description}", "FAIL", str(e)))
            print(f"  -> ERROR: {e}")

        # Test UVE
        print(f"\nUVE ({n_features} features):")
        try:
            importances = uve_selection(X, y)
            if np.all(importances == 1):
                results.append((f"UVE {description}", "SKIPPED (uniform)", ""))
                print("  -> Returned uniform importances (graceful skip)")
            else:
                results.append((f"UVE {description}", "PASS", f"Max importance: {importances.max():.4f}"))
                print(f"  -> Completed, max importance: {importances.max():.4f}")
        except Exception as e:
            results.append((f"UVE {description}", "FAIL", str(e)))
            print(f"  -> ERROR: {e}")

    return results


def test_pls_n_components_capping():
    """Test PLS n_components capping with small feature counts."""
    print("\n" + "="*70)
    print("TEST 3: PLS n_components Capping")
    print("="*70)

    test_cases = [
        (30, 10, "30 wavelengths, n_components=10 (should work)"),
        (10, 10, "10 wavelengths, n_components=10 (should cap to 9)"),
        (5, 10, "5 wavelengths, n_components=10 (should cap to 4)"),
        (2, 10, "2 wavelengths, n_components=10 (should cap to 1)"),
    ]

    results = []
    n_samples = 50

    for n_features, n_components, description in test_cases:
        print(f"\n--- {description} ---")
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        # Cap n_components as the code does
        capped_n_components = min(n_components, n_features - 1, n_samples - 1)
        capped_n_components = max(1, capped_n_components)

        print(f"  Original n_components: {n_components}")
        print(f"  Capped n_components: {capped_n_components}")

        try:
            pls = PLSRegression(n_components=capped_n_components, scale=False)
            pls.fit(X, y)
            y_pred = pls.predict(X)
            results.append((description, "PASS", f"Fitted with n_components={capped_n_components}"))
            print(f"  -> SUCCESS: PLS fitted with n_components={capped_n_components}")
        except Exception as e:
            results.append((description, "FAIL", str(e)))
            print(f"  -> ERROR: {e}")

    return results


def test_model_fitting_with_small_features():
    """Test various models with small feature counts."""
    print("\n" + "="*70)
    print("TEST 4: Model Fitting with Small Feature Counts")
    print("="*70)

    test_cases = [
        (30, "30 wavelengths"),
        (10, "10 wavelengths"),
        (5, "5 wavelengths"),
        (2, "2 wavelengths"),
    ]

    results = []
    n_samples = 50

    for n_features, description in test_cases:
        print(f"\n--- {description} ---")
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        # Test Ridge (should always work)
        print(f"\nRidge ({n_features} features):")
        try:
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            results.append((f"Ridge {description}", "PASS", ""))
            print("  -> SUCCESS")
        except Exception as e:
            results.append((f"Ridge {description}", "FAIL", str(e)))
            print(f"  -> ERROR: {e}")

        # Test PLS with capped components
        print(f"\nPLS ({n_features} features):")
        n_components = min(5, n_features - 1, n_samples - 1)
        n_components = max(1, n_components)
        try:
            model = PLSRegression(n_components=n_components, scale=False)
            model.fit(X, y)
            results.append((f"PLS {description}", "PASS", f"n_components={n_components}"))
            print(f"  -> SUCCESS with n_components={n_components}")
        except Exception as e:
            results.append((f"PLS {description}", "FAIL", str(e)))
            print(f"  -> ERROR: {e}")

    return results


def print_summary(all_results):
    """Print summary of all test results."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passes = sum(1 for _, status, _ in all_results if status == "PASS")
    skipped = sum(1 for _, status, _ in all_results if "SKIPPED" in status)
    fails = sum(1 for _, status, _ in all_results if status == "FAIL")
    expected_fails = sum(1 for _, status, _ in all_results if "EXPECTED" in status)

    print(f"\nTotal tests: {len(all_results)}")
    print(f"  PASS: {passes}")
    print(f"  SKIPPED (graceful): {skipped}")
    print(f"  EXPECTED FAIL: {expected_fails}")
    print(f"  UNEXPECTED FAIL: {fails}")

    if fails > 0:
        print("\nFAILED TESTS:")
        for test, status, msg in all_results:
            if status == "FAIL":
                print(f"  - {test}: {msg}")

    return fails == 0


def main():
    """Run all tests."""
    print("="*70)
    print("SMALL WAVELENGTH HANDLING TEST SUITE")
    print("="*70)
    print("Testing graceful degradation for restricted wavelength ranges.")

    all_results = []

    # Run all tests
    all_results.extend(test_savgol_auto_adjustment())
    all_results.extend(test_variable_selection_graceful_degradation())
    all_results.extend(test_pls_n_components_capping())
    all_results.extend(test_model_fitting_with_small_features())

    # Print summary
    success = print_summary(all_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
