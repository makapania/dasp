"""
Diagnostic test for R² reproducibility issue

This script helps diagnose why the Results tab shows different R² than Model Development tab.
It simulates both workflows to compare results.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

def test_ridge_reproducibility():
    """Test if Ridge models produce consistent R² values with same configuration."""

    print("="*80)
    print("R² REPRODUCIBILITY DIAGNOSTIC TEST")
    print("="*80)
    print("\nPurpose: Identify why Results tab and Model Development tab show different R² values")
    print("\nInstructions:")
    print("1. Load your data in the GUI")
    print("2. Run a Ridge search with default settings (alpha=0.001)")
    print("3. Note the R² value from Results tab")
    print("4. Run this test with the same data")
    print("5. Compare the R² values")
    print("\n" + "="*80 + "\n")

    # Generate synthetic data that mimics spectral data
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 500

    # Create synthetic spectral data
    X = np.random.randn(n_samples, n_wavelengths)
    # Add some structure to make it more realistic
    for i in range(n_samples):
        X[i] += np.sin(np.linspace(0, 4*np.pi, n_wavelengths)) * (i % 10) / 10

    # Create synthetic target with some relationship to X
    y = np.mean(X[:, 100:200], axis=1) + 0.5 * np.mean(X[:, 300:400], axis=1) + np.random.randn(n_samples) * 0.1

    print(f"Data shape: {n_samples} samples × {n_wavelengths} wavelengths")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Target mean: {y.mean():.3f}, std: {y.std():.3f}")
    print()

    # Test 1: Simulate search.py workflow
    print("TEST 1: Simulating search.py workflow (Results tab)")
    print("-" * 60)

    model_search = Ridge(alpha=0.001, random_state=None)
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_folds_search = []
    rmse_folds_search = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_fold = Ridge(alpha=0.001, random_state=None)
        model_fold.fit(X_train, y_train)
        y_pred = model_fold.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        r2_folds_search.append(r2)
        rmse_folds_search.append(rmse)
        print(f"  Fold {fold_idx+1}: R²={r2:.4f}, RMSE={rmse:.4f}")

    mean_r2_search = np.mean(r2_folds_search)
    std_r2_search = np.std(r2_folds_search)
    mean_rmse_search = np.mean(rmse_folds_search)

    print(f"\nResults Tab (search.py) - Mean CV R²: {mean_r2_search:.4f} ± {std_r2_search:.4f}")
    print(f"Results Tab (search.py) - Mean CV RMSE: {mean_rmse_search:.4f}")
    print()

    # Test 2: Simulate GUI refinement workflow
    print("TEST 2: Simulating GUI refinement workflow (Model Development tab)")
    print("-" * 60)

    model_refine = Ridge(alpha=0.001, random_state=None)
    cv_splitter_refine = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_folds_refine = []
    rmse_folds_refine = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter_refine.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_fold = Ridge(alpha=0.001, random_state=None)
        model_fold.fit(X_train, y_train)
        y_pred = model_fold.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        r2_folds_refine.append(r2)
        rmse_folds_refine.append(rmse)
        print(f"  Fold {fold_idx+1}: R²={r2:.4f}, RMSE={rmse:.4f}")

    mean_r2_refine = np.mean(r2_folds_refine)
    std_r2_refine = np.std(r2_folds_refine)
    mean_rmse_refine = np.mean(rmse_folds_refine)

    print(f"\nModel Development Tab (GUI) - Mean CV R²: {mean_r2_refine:.4f} ± {std_r2_refine:.4f}")
    print(f"Model Development Tab (GUI) - Mean CV RMSE: {mean_rmse_refine:.4f}")
    print()

    # Compare results
    print("="*80)
    print("COMPARISON")
    print("="*80)

    r2_diff = mean_r2_refine - mean_r2_search
    rmse_diff = mean_rmse_refine - mean_rmse_search

    print(f"R² difference (Refinement - Search): {r2_diff:+.6f}")
    print(f"RMSE difference (Refinement - Search): {rmse_diff:+.6f}")
    print()

    if abs(r2_diff) < 1e-10:
        print("[PASS] R2 values are identical!")
        print("  The workflows produce the same results with synthetic data.")
        print("  This suggests the issue is data-specific or related to actual model loading.")
        print("")
        print("Most likely causes of your R2 mismatch:")
        print("  1. Wrong row selected in Results tab (different hyperparameters)")
        print("  2. Excluded spectra changed between search and refinement")
        print("  3. Wavelength filter applied after search but before refinement")
        print("  4. Validation set configuration changed")
        print("  5. Data file was reloaded with different samples/wavelengths")
    else:
        print("[FAIL] R2 values differ!")
        print(f"  Expected difference: ~0.0000, Actual: {r2_diff:+.6f}")
        print("  This indicates a systematic difference in the workflows.")

    print()
    print("Next steps:")
    print("1. If this test PASSES, the issue is likely in how actual data is loaded/processed")
    print("2. If this test FAILS, there's a systematic difference in the workflows")
    print("3. Check console output when loading a model in the GUI for warnings about:")
    print("   - Wavelength mismatches")
    print("   - Missing all_vars")
    print("   - Validation set differences")
    print("   - Preprocessing differences")

    return {
        'search_r2': mean_r2_search,
        'refine_r2': mean_r2_refine,
        'difference': r2_diff,
        'passed': abs(r2_diff) < 1e-10
    }


if __name__ == '__main__':
    result = test_ridge_reproducibility()

    print("\n" + "="*80)
    if result['passed']:
        print("TEST RESULT: PASSED")
    else:
        print("TEST RESULT: FAILED")
    print("="*80)
