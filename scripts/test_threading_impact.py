"""
Test whether BLAS threading affects spectral model results.

This script tests whether reproducibility mode (single-threaded BLAS)
produces different results than default multi-threaded execution.

If results are identical, reproducibility mode provides no additional
value and could potentially be removed.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def set_single_thread():
    """Set BLAS/LAPACK to single-threaded mode."""
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[var] = '1'


def clear_thread_settings():
    """Clear BLAS thread environment variables (use system defaults)."""
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ.pop(var, None)


def generate_synthetic_spectral_data(n_samples=1000, n_wavelengths=2500, seed=42):
    """Generate synthetic spectral data for testing."""
    np.random.seed(seed)

    # Create wavelengths
    wavelengths = np.linspace(400, 2500, n_wavelengths)

    # Generate X with spectral-like structure
    X = np.zeros((n_samples, n_wavelengths))
    for i in range(n_samples):
        # Base spectrum
        base = np.sin(wavelengths / 100) + np.cos(wavelengths / 200)
        # Add noise
        noise = np.random.randn(n_wavelengths) * 0.1
        # Add sample-specific variation
        variation = np.random.randn() * 0.5 * np.sin(wavelengths / 50)
        X[i] = base + noise + variation

    # Generate y as function of specific wavelength regions
    region1 = X[:, 200:300].mean(axis=1)
    region2 = X[:, 500:600].mean(axis=1)
    y = 2 * region1 + 3 * region2 + np.random.randn(n_samples) * 0.5

    return X, y, wavelengths


def run_pls_cv(X, y, n_components=5, n_folds=5, n_jobs=1):
    """Run PLS with cross-validation."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score, KFold

    np.random.seed(42)  # Reset seed for reproducibility

    model = PLSRegression(n_components=n_components)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=n_jobs)
    return scores.mean(), scores.std()


def run_ridge_cv(X, y, alpha=1.0, n_folds=5, n_jobs=1):
    """Run Ridge with cross-validation."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold

    np.random.seed(42)

    model = Ridge(alpha=alpha, random_state=42)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=n_jobs)
    return scores.mean(), scores.std()


def run_lightgbm_cv(X, y, n_estimators=50, n_folds=5, n_jobs=1):
    """Run LightGBM with cross-validation."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None, None
    from sklearn.model_selection import cross_val_score, KFold

    np.random.seed(42)

    model = LGBMRegressor(n_estimators=n_estimators, random_state=42, verbose=-1, n_jobs=1)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=n_jobs)
    return scores.mean(), scores.std()


def main():
    print("=" * 70)
    print("THREADING IMPACT TEST")
    print("=" * 70)
    print("\nThis test compares results between:")
    print("  1. Single-threaded BLAS (reproducible mode)")
    print("  2. Multi-threaded BLAS (default mode)")
    print()

    # Generate data
    print("Generating synthetic spectral data...")
    X, y, wavelengths = generate_synthetic_spectral_data()
    print(f"  Shape: {X.shape}")
    print()

    results = {}

    # Test 1: Single-threaded BLAS
    print("=" * 40)
    print("TEST 1: Single-threaded BLAS (n_threads=1)")
    print("=" * 40)
    set_single_thread()

    # Need to reload numpy/scipy to pick up new thread settings
    # This is tricky - in practice, settings must be set BEFORE import
    # For this test, we'll rely on threadpoolctl
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            results['single_pls'] = run_pls_cv(X, y)
            results['single_ridge'] = run_ridge_cv(X, y)
            results['single_lgbm'] = run_lightgbm_cv(X, y)
            print(f"  PLS:      R² = {results['single_pls'][0]:.10f} ± {results['single_pls'][1]:.10f}")
            print(f"  Ridge:    R² = {results['single_ridge'][0]:.10f} ± {results['single_ridge'][1]:.10f}")
            if results['single_lgbm'][0] is not None:
                print(f"  LightGBM: R² = {results['single_lgbm'][0]:.10f} ± {results['single_lgbm'][1]:.10f}")
    except ImportError:
        print("  WARNING: threadpoolctl not available, using environment variables only")
        results['single_pls'] = run_pls_cv(X, y)
        results['single_ridge'] = run_ridge_cv(X, y)
        results['single_lgbm'] = run_lightgbm_cv(X, y)
        print(f"  PLS:      R² = {results['single_pls'][0]:.10f} ± {results['single_pls'][1]:.10f}")
        print(f"  Ridge:    R² = {results['single_ridge'][0]:.10f} ± {results['single_ridge'][1]:.10f}")
        if results['single_lgbm'][0] is not None:
            print(f"  LightGBM: R² = {results['single_lgbm'][0]:.10f} ± {results['single_lgbm'][1]:.10f}")
    print()

    # Test 2: Multi-threaded BLAS (default)
    print("=" * 40)
    print("TEST 2: Multi-threaded BLAS (default)")
    print("=" * 40)
    clear_thread_settings()

    try:
        from threadpoolctl import threadpool_limits
        # Don't limit - use system defaults
        results['multi_pls'] = run_pls_cv(X, y)
        results['multi_ridge'] = run_ridge_cv(X, y)
        results['multi_lgbm'] = run_lightgbm_cv(X, y)
    except ImportError:
        results['multi_pls'] = run_pls_cv(X, y)
        results['multi_ridge'] = run_ridge_cv(X, y)
        results['multi_lgbm'] = run_lightgbm_cv(X, y)

    print(f"  PLS:      R² = {results['multi_pls'][0]:.10f} ± {results['multi_pls'][1]:.10f}")
    print(f"  Ridge:    R² = {results['multi_ridge'][0]:.10f} ± {results['multi_ridge'][1]:.10f}")
    if results['multi_lgbm'][0] is not None:
        print(f"  LightGBM: R² = {results['multi_lgbm'][0]:.10f} ± {results['multi_lgbm'][1]:.10f}")
    print()

    # Test 3: Compare n_jobs=1 vs n_jobs=-1
    print("=" * 40)
    print("TEST 3: n_jobs=1 vs n_jobs=-1")
    print("=" * 40)
    results['serial_pls'] = run_pls_cv(X, y, n_jobs=1)
    results['parallel_pls'] = run_pls_cv(X, y, n_jobs=-1)
    print(f"  PLS (n_jobs=1):  R² = {results['serial_pls'][0]:.10f}")
    print(f"  PLS (n_jobs=-1): R² = {results['parallel_pls'][0]:.10f}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    diff_pls = abs(results['single_pls'][0] - results['multi_pls'][0])
    diff_ridge = abs(results['single_ridge'][0] - results['multi_ridge'][0])
    diff_njobs = abs(results['serial_pls'][0] - results['parallel_pls'][0])

    print(f"\nBLAS threading difference (single vs multi):")
    print(f"  PLS:   {diff_pls:.2e}")
    print(f"  Ridge: {diff_ridge:.2e}")
    if results['single_lgbm'][0] is not None and results['multi_lgbm'][0] is not None:
        diff_lgbm = abs(results['single_lgbm'][0] - results['multi_lgbm'][0])
        print(f"  LightGBM: {diff_lgbm:.2e}")

    print(f"\nn_jobs difference (1 vs -1):")
    print(f"  PLS: {diff_njobs:.2e}")

    # Conclusion
    threshold = 1e-10
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    all_identical = (diff_pls < threshold and diff_ridge < threshold and diff_njobs < threshold)

    if all_identical:
        print("\n[OK] Results are IDENTICAL (within 1e-10)")
        print("     Reproducibility mode provides NO additional value for this data size.")
        print("     Consider removing reproducibility mode toggle entirely.")
    else:
        print("\n[!] Results DIFFER between threading modes")
        print("    Reproducibility mode may still provide value.")
        print(f"    Maximum difference: {max(diff_pls, diff_ridge, diff_njobs):.2e}")
    print()


if __name__ == "__main__":
    main()
