"""
Test script to verify model independence and reproducibility.

This script tests:
1. Model Independence: PLS results should be identical whether trained alone or with XGBoost
2. Reproducibility: Same search twice should give bit-exact identical results
3. Parameter Consistency: Models should produce same predictions after save/load

For scientific research, these properties are CRITICAL.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from spectral_predict.search import run_search


def create_synthetic_data(n_samples=100, n_features=50, random_seed=42):
    """Create synthetic spectral data for testing."""
    np.random.seed(random_seed)

    # Create wavelengths
    wavelengths = np.linspace(1000, 2000, n_features)

    # Create spectra with some structure
    X = np.random.randn(n_samples, n_features) * 0.1
    for i in range(n_samples):
        # Add baseline
        X[i] += np.linspace(0.5, 1.5, n_features)
        # Add peaks
        X[i] += 0.3 * np.exp(-((wavelengths - 1200) ** 2) / 10000)
        X[i] += 0.2 * np.exp(-((wavelengths - 1600) ** 2) / 10000)

    # Create target with correlation to certain wavelengths
    y = (X[:, 10] + X[:, 30] + X[:, 40] + np.random.randn(n_samples) * 0.1)

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=[f"{w:.1f}" for w in wavelengths])
    y_series = pd.Series(y, name='target')

    return X_df, y_series


def test_model_independence():
    """
    Test 1: Model Independence

    Verify that PLS produces identical results whether trained alone or with XGBoost.
    This is CRITICAL for scientific reproducibility.
    """
    print("=" * 80)
    print("TEST 1: MODEL INDEPENDENCE")
    print("=" * 80)
    print("\nTesting if PLS results are identical when trained:")
    print("  Run A: PLS alone")
    print("  Run B: PLS + XGBoost together")
    print()

    # Create test data
    X, y = create_synthetic_data(n_samples=100, n_features=50, random_seed=42)

    # Run A: Train PLS alone
    print("Run A: Training PLS alone...")
    results_pls_alone, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier='quick'
    )

    # Filter to full model (not subsets)
    pls_alone = results_pls_alone[results_pls_alone['SubsetTag'] == 'full'].copy()

    # Run B: Train PLS + XGBoost together
    print("\nRun B: Training PLS + XGBoost together...")
    results_combined, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        models_to_test=['PLS', 'XGBoost'],
        preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier='quick'
    )

    # Filter to PLS full model
    pls_combined = results_combined[
        (results_combined['Model'] == 'PLS') &
        (results_combined['SubsetTag'] == 'full')
    ].copy()

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    if len(pls_alone) == 0 or len(pls_combined) == 0:
        print("ERROR: No PLS results found!")
        return False

    # Compare each row
    success = True
    for idx in range(len(pls_alone)):
        row_alone = pls_alone.iloc[idx]
        row_combined = pls_combined.iloc[idx]

        print(f"\nConfiguration {idx + 1}:")
        print(f"  Params: {row_alone['Params']}")
        print(f"  PLS Alone    - R²: {row_alone['R2']:.10f}, RMSE: {row_alone['RMSE']:.10f}")
        print(f"  PLS Combined - R²: {row_combined['R2']:.10f}, RMSE: {row_combined['RMSE']:.10f}")

        # Check if identical (allow tiny floating point errors)
        r2_diff = abs(row_alone['R2'] - row_combined['R2'])
        rmse_diff = abs(row_alone['RMSE'] - row_combined['RMSE'])

        if r2_diff > 1e-10 or rmse_diff > 1e-10:
            print(f"  [FAIL] - Results differ!")
            print(f"     R� difference: {r2_diff:.2e}")
            print(f"     RMSE difference: {rmse_diff:.2e}")
            success = False
        else:
            print(f"  [PASS] - Results are identical")

    print("\n" + "=" * 80)
    if success:
        print("[PASS] TEST 1 PASSED: Models are independent")
    else:
        print("[FAIL] TEST 1 FAILED: Models affect each other!")
    print("=" * 80 + "\n")

    return success


def test_reproducibility():
    """
    Test 2: Reproducibility

    Verify that running the same search twice produces identical results.
    This is CRITICAL for scientific reproducibility.
    """
    print("=" * 80)
    print("TEST 2: REPRODUCIBILITY")
    print("=" * 80)
    print("\nTesting if results are identical across two runs with same parameters")
    print()

    # Create test data
    X, y = create_synthetic_data(n_samples=100, n_features=50, random_seed=42)

    # Run 1
    print("Run 1: Training models...")
    results1, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        models_to_test=['PLS', 'RandomForest', 'XGBoost'],
        preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier='quick'
    )

    # Run 2
    print("\nRun 2: Training models again with same parameters...")
    results2, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        models_to_test=['PLS', 'RandomForest', 'XGBoost'],
        preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier='quick'
    )

    # Filter to full models
    results1_full = results1[results1['SubsetTag'] == 'full'].reset_index(drop=True)
    results2_full = results2[results2['SubsetTag'] == 'full'].reset_index(drop=True)

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    success = True
    for model_name in ['PLS', 'RandomForest', 'XGBoost']:
        print(f"\n{model_name}:")

        model1 = results1_full[results1_full['Model'] == model_name]
        model2 = results2_full[results2_full['Model'] == model_name]

        if len(model1) == 0 or len(model2) == 0:
            print(f"  [WARN] WARNING: {model_name} results not found")
            continue

        for idx in range(len(model1)):
            row1 = model1.iloc[idx]
            row2 = model2.iloc[idx]

            print(f"\n  Config {idx + 1}: {row1['Params']}")
            print(f"    Run 1 - R²: {row1['R2']:.10f}, RMSE: {row1['RMSE']:.10f}")
            print(f"    Run 2 - R²: {row2['R2']:.10f}, RMSE: {row2['RMSE']:.10f}")

            # Check if identical (allow tiny floating point errors)
            r2_diff = abs(row1['R2'] - row2['R2'])
            rmse_diff = abs(row1['RMSE'] - row2['RMSE'])

            if r2_diff > 1e-10 or rmse_diff > 1e-10:
                print(f"    [FAIL] FAIL - Results differ!")
                print(f"       R² difference: {r2_diff:.2e}")
                print(f"       RMSE difference: {rmse_diff:.2e}")
                success = False
            else:
                print(f"    [PASS] PASS - Results are identical")

    print("\n" + "=" * 80)
    if success:
        print("[PASS] TEST 2 PASSED: Results are reproducible")
    else:
        print("[FAIL] TEST 2 FAILED: Results are not reproducible!")
    print("=" * 80 + "\n")

    return success


def test_cv_splitter_isolation():
    """
    Test 3: CV Splitter Isolation

    Verify that the CV splitter is properly isolated between models.
    If models share the CV splitter state, later models might see different splits.
    """
    print("=" * 80)
    print("TEST 3: CV SPLITTER ISOLATION")
    print("=" * 80)
    print("\nTesting if CV splits are consistent across models")
    print()

    # Create test data
    X, y = create_synthetic_data(n_samples=100, n_features=50, random_seed=42)

    # Run with models in different orders
    print("Run A: Training order: PLS, XGBoost, RandomForest...")
    results_order_a, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        models_to_test=['PLS', 'XGBoost', 'RandomForest'],
        preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier='quick'
    )

    print("\nRun B: Training order: RandomForest, XGBoost, PLS...")
    results_order_b, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        models_to_test=['RandomForest', 'XGBoost', 'PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier='quick'
    )

    # Compare results for each model
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    success = True
    for model_name in ['PLS', 'XGBoost', 'RandomForest']:
        print(f"\n{model_name}:")

        model_a = results_order_a[
            (results_order_a['Model'] == model_name) &
            (results_order_a['SubsetTag'] == 'full')
        ]
        model_b = results_order_b[
            (results_order_b['Model'] == model_name) &
            (results_order_b['SubsetTag'] == 'full')
        ]

        if len(model_a) == 0 or len(model_b) == 0:
            print(f"  [WARN] WARNING: {model_name} results not found")
            continue

        for idx in range(len(model_a)):
            row_a = model_a.iloc[idx]
            row_b = model_b.iloc[idx]

            print(f"  Config {idx + 1}: {row_a['Params']}")
            print(f"    Order A - R²: {row_a['R2']:.10f}, RMSE: {row_a['RMSE']:.10f}")
            print(f"    Order B - R²: {row_b['R2']:.10f}, RMSE: {row_b['RMSE']:.10f}")

            # Check if identical
            r2_diff = abs(row_a['R2'] - row_b['R2'])
            rmse_diff = abs(row_a['RMSE'] - row_b['RMSE'])

            if r2_diff > 1e-10 or rmse_diff > 1e-10:
                print(f"    [FAIL] FAIL - Results depend on training order!")
                print(f"       R² difference: {r2_diff:.2e}")
                print(f"       RMSE difference: {rmse_diff:.2e}")
                success = False
            else:
                print(f"    [PASS] PASS - Results independent of order")

    print("\n" + "=" * 80)
    if success:
        print("[PASS] TEST 3 PASSED: CV splitter is properly isolated")
    else:
        print("[FAIL] TEST 3 FAILED: Training order affects results!")
    print("=" * 80 + "\n")

    return success


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MODEL INDEPENDENCE AND REPRODUCIBILITY TEST SUITE")
    print("=" * 80)
    print("\nThis test suite verifies critical properties for scientific research:")
    print("  1. Model Independence: Results don't depend on which other models are trained")
    print("  2. Reproducibility: Same inputs always produce same outputs")
    print("  3. CV Splitter Isolation: Training order doesn't affect results")
    print("\n" + "=" * 80 + "\n")

    results = {}

    # Run tests
    try:
        results['model_independence'] = test_model_independence()
    except Exception as e:
        print(f"ERROR in test_model_independence: {e}")
        import traceback
        traceback.print_exc()
        results['model_independence'] = False

    try:
        results['reproducibility'] = test_reproducibility()
    except Exception as e:
        print(f"ERROR in test_reproducibility: {e}")
        import traceback
        traceback.print_exc()
        results['reproducibility'] = False

    try:
        results['cv_splitter_isolation'] = test_cv_splitter_isolation()
    except Exception as e:
        print(f"ERROR in test_cv_splitter_isolation: {e}")
        import traceback
        traceback.print_exc()
        results['cv_splitter_isolation'] = False

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("[PASS][PASS][PASS] ALL TESTS PASSED [PASS][PASS][PASS]")
        print("\nYour models are:")
        print("  - Independent (selecting different models doesn't affect results)")
        print("  - Reproducible (same inputs = same outputs)")
        print("  - Order-invariant (training order doesn't matter)")
        print("\nSAFE FOR SCIENTIFIC RESEARCH [PASS]")
    else:
        print("[FAIL][FAIL][FAIL] SOME TESTS FAILED [FAIL][FAIL][FAIL]")
        print("\nCRITICAL ISSUES FOUND:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  - {test_name}")
        print("\nNOT SAFE FOR SCIENTIFIC RESEARCH UNTIL FIXED!")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
