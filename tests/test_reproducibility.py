"""
Comprehensive test suite for reproducibility functionality.

This module tests that the reproducible mode ensures bit-identical results
across runs, which is critical for scientific research applications.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression, make_classification

# Import the functions we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectral_predict.search import run_search
from spectral_predict.variable_selection import (
    uve_selection,
    spa_selection,
    ipls_selection,
    uve_spa_selection
)
from spectral_predict.reproducibility import set_blas_threads, check_reproducibility_status


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def regression_data():
    """Create synthetic regression data for testing."""
    np.random.seed(12345)
    X, y = make_regression(
        n_samples=50,
        n_features=100,
        n_informative=10,
        noise=0.5,
        random_state=12345
    )

    # Create wavelength column names (simulating spectral data)
    wavelengths = [f"{400 + i*2}" for i in range(100)]
    X_df = pd.DataFrame(X, columns=wavelengths)
    y_series = pd.Series(y, name='target')

    return X_df, y_series


@pytest.fixture
def classification_data():
    """Create synthetic classification data for testing."""
    np.random.seed(12345)
    X, y = make_classification(
        n_samples=50,
        n_features=100,
        n_informative=10,
        n_classes=2,
        random_state=12345
    )

    # Create wavelength column names
    wavelengths = [f"{400 + i*2}" for i in range(100)]
    X_df = pd.DataFrame(X, columns=wavelengths)
    y_series = pd.Series(y, name='target')

    return X_df, y_series


# ============================================================================
# TEST 1: BASIC REPRODUCIBILITY
# ============================================================================

def test_basic_reproducibility_regression(regression_data):
    """Test that reproducible mode gives identical results for regression."""
    X, y = regression_data

    # Run search twice with reproducible=True
    results1, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS', 'Ridge'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    results2, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS', 'Ridge'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    # Results should be EXACTLY identical (bit-for-bit)
    pd.testing.assert_frame_equal(results1, results2)

    # Also verify specific metrics
    assert np.allclose(results1['RMSE'].values, results2['RMSE'].values, rtol=0, atol=0), \
        "RMSE values must be bit-identical in reproducible mode"
    assert np.allclose(results1['R2'].values, results2['R2'].values, rtol=0, atol=0), \
        "R2 values must be bit-identical in reproducible mode"


def test_basic_reproducibility_classification(classification_data):
    """Test that reproducible mode gives identical results for classification."""
    X, y = classification_data

    # Run search twice with reproducible=True
    results1, _ = run_search(
        X, y,
        task_type='classification',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS', 'Ridge'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    results2, _ = run_search(
        X, y,
        task_type='classification',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS', 'Ridge'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    # Results should be EXACTLY identical
    pd.testing.assert_frame_equal(results1, results2)


# ============================================================================
# TEST 2: VARIABLE SELECTION REPRODUCIBILITY
# ============================================================================

def test_uve_reproducibility(regression_data):
    """Test that UVE variable selection is reproducible."""
    X, y = regression_data
    X_np = X.values
    y_np = y.values

    # Run UVE twice with same random_state
    imp1 = uve_selection(X_np, y_np, random_state=42)
    imp2 = uve_selection(X_np, y_np, random_state=42)

    # Should be identical
    np.testing.assert_array_equal(imp1, imp2,
        err_msg="UVE selection must be reproducible with same random_state")


def test_spa_reproducibility(regression_data):
    """Test that SPA variable selection is reproducible."""
    X, y = regression_data
    X_np = X.values
    y_np = y.values

    # Run SPA twice with same random_state
    imp1 = spa_selection(X_np, y_np, n_features=20, random_state=42)
    imp2 = spa_selection(X_np, y_np, n_features=20, random_state=42)

    # Should be identical
    np.testing.assert_array_equal(imp1, imp2,
        err_msg="SPA selection must be reproducible with same random_state")


def test_ipls_reproducibility(regression_data):
    """Test that iPLS variable selection is reproducible."""
    X, y = regression_data
    X_np = X.values
    y_np = y.values

    # Run iPLS twice with same random_state
    imp1 = ipls_selection(X_np, y_np, n_intervals=10, random_state=42)
    imp2 = ipls_selection(X_np, y_np, n_intervals=10, random_state=42)

    # Should be identical
    np.testing.assert_array_equal(imp1, imp2,
        err_msg="iPLS selection must be reproducible with same random_state")


def test_uve_spa_reproducibility(regression_data):
    """Test that UVE-SPA hybrid selection is reproducible."""
    X, y = regression_data
    X_np = X.values
    y_np = y.values

    # Run UVE-SPA twice with same random_state
    imp1 = uve_spa_selection(X_np, y_np, n_features=20, random_state=42)
    imp2 = uve_spa_selection(X_np, y_np, n_features=20, random_state=42)

    # Should be identical
    np.testing.assert_array_equal(imp1, imp2,
        err_msg="UVE-SPA selection must be reproducible with same random_state")


# ============================================================================
# TEST 3: RANDOM STATE VARIATION
# ============================================================================

def test_different_random_state_gives_different_results(regression_data):
    """Test that different random_state values produce different (but reproducible) results."""
    X, y = regression_data

    # Run with random_state=42
    results_42a, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    results_42b, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    # Run with random_state=123
    results_123, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=123,
        tier='quick'
    )

    # Same random_state should give identical results
    pd.testing.assert_frame_equal(results_42a, results_42b)

    # Different random_state should give different results
    # (CV splits will be different, leading to different metrics)
    assert not results_42a.equals(results_123), \
        "Different random_state should produce different results"


# ============================================================================
# TEST 4: MODEL-SPECIFIC REPRODUCIBILITY
# ============================================================================

@pytest.mark.parametrize("model_name", ['PLS', 'Ridge', 'Lasso'])
def test_model_reproducibility(model_name, regression_data):
    """Test reproducibility for each model type."""
    X, y = regression_data

    # Run twice
    results1, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=3,
        models_to_test=[model_name],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    results2, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=3,
        models_to_test=[model_name],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    # Should be identical
    pd.testing.assert_frame_equal(results1, results2)


# ============================================================================
# TEST 5: RANKING STABILITY FOR NEAR-TIES
# ============================================================================

def test_ranking_stability(regression_data):
    """Test that ranking is stable even when models have very similar performance."""
    X, y = regression_data

    # Run 3 times to ensure stability
    results_list = []
    for i in range(3):
        results, _ = run_search(
            X, y,
            task_type='regression',
            folds=3,
            max_n_components=8,  # More components = more similar models
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
            enable_variable_subsets=False,
            enable_region_subsets=False,
            reproducible=True,
            random_state=42,
            tier='quick'
        )
        results_list.append(results)

    # All rankings should be identical
    for i in range(1, len(results_list)):
        pd.testing.assert_frame_equal(results_list[0], results_list[i])

    # Verify that Rank column is consistent
    for i in range(1, len(results_list)):
        np.testing.assert_array_equal(
            results_list[0]['Rank'].values,
            results_list[i]['Rank'].values,
            err_msg="Ranking must be stable across runs"
        )


# ============================================================================
# TEST 6: BLAS THREAD CONTROL
# ============================================================================

def test_blas_thread_control():
    """Test that BLAS thread control works."""
    # Set to 1 thread
    set_blas_threads(1)

    # Check status
    status = check_reproducibility_status()

    # Verify environment variables are set
    assert status['blas_threads_env']['OMP_NUM_THREADS'] == '1'
    assert status['blas_threads_env']['MKL_NUM_THREADS'] == '1'
    assert status['blas_threads_env']['OPENBLAS_NUM_THREADS'] == '1'


# ============================================================================
# TEST 7: PENALTIES DEFAULT TO ZERO
# ============================================================================

def test_penalties_default_to_zero(regression_data):
    """Test that variable and complexity penalties default to 0."""
    X, y = regression_data

    # Run search without specifying penalties (should default to 0)
    results, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=5,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,
        random_state=42,
        tier='quick'
    )

    # With penalties=0, ranking should be by primary metric only (R² for regression)
    # Verify that results are sorted by R² descending
    r2_values = results['R2'].values
    assert np.all(r2_values[:-1] >= r2_values[1:]), \
        "With penalties=0, results should be ranked by R² (descending)"


# ============================================================================
# SUMMARY TEST
# ============================================================================

def test_full_reproducibility_workflow(regression_data):
    """
    End-to-end test of the complete reproducibility workflow.

    This test verifies that the reproducibility mode works correctly
    across multiple runs with various settings.
    """
    X, y = regression_data

    # Define test parameters
    test_params = {
        'task_type': 'regression',
        'folds': 3,
        'max_n_components': 5,
        'models_to_test': ['PLS', 'Ridge'],
        'preprocessing_methods': {'raw': True, 'snv': True, 'deriv': False, 'deriv_snv': False},
        'enable_variable_subsets': False,
        'enable_region_subsets': False,
        'reproducible': True,
        'random_state': 42,
        'tier': 'quick'
    }

    # Run 5 times to ensure complete stability
    results_list = []
    for i in range(5):
        results, _ = run_search(X, y, **test_params)
        results_list.append(results)

    # ALL runs should produce EXACTLY the same results
    for i in range(1, len(results_list)):
        pd.testing.assert_frame_equal(
            results_list[0],
            results_list[i],
            check_exact=True,
            check_dtype=True,
            obj=f"Run {i+1} must be identical to Run 1"
        )

    print("SUCCESS: All 5 runs produced bit-identical results!")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
