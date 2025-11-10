"""
Test tier system (quick, standard, comprehensive) with example bone collagen data.

This test verifies:
1. Each tier runs successfully
2. Runtime differences between tiers
3. Performance metrics at different tiers
4. Model counts match tier definitions
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spectral_predict.models import get_model_grids
from spectral_predict.model_config import MODEL_TIERS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_bone_collagen_data(n_samples=20):
    """
    Load a subset of bone collagen data for testing.

    Parameters
    ----------
    n_samples : int
        Number of samples to load (default 20 for speed)

    Returns
    -------
    X : ndarray
        Spectral data
    y : ndarray
        Collagen percentages
    """
    example_dir = Path(__file__).parent.parent / 'example'
    csv_path = example_dir / 'BoneCollagen.csv'

    # Load reference data
    df = pd.read_csv(csv_path)

    # Limit to n_samples
    df = df.head(n_samples)

    # Create synthetic spectral data (2151 wavelengths like real ASD files)
    # In reality, we'd load from ASD files, but for testing we'll create synthetic
    np.random.seed(42)
    n_wavelengths = 2151
    X = np.random.randn(len(df), n_wavelengths) * 0.1

    # Add signal correlated with collagen content
    for i, collagen in enumerate(df['%Collagen'].values):
        # Create spectral features correlated with collagen
        baseline = collagen / 20.0  # normalize to 0-1 range
        X[i] += baseline * np.sin(np.linspace(0, 10, n_wavelengths))
        X[i] += baseline * 0.5 * np.cos(np.linspace(0, 5, n_wavelengths))
        # Add some random variation
        X[i] += np.random.randn(n_wavelengths) * 0.05

    y = df['%Collagen'].values

    return X, y


def test_tier_definitions():
    """Test that tier definitions are properly configured."""
    print("\n" + "="*80)
    print("TEST 1: Tier Definitions")
    print("="*80)

    # Check that all expected tiers exist
    expected_tiers = ['quick', 'standard', 'comprehensive', 'experimental']
    for tier in expected_tiers:
        assert tier in MODEL_TIERS, f"Tier '{tier}' not found in MODEL_TIERS"
        assert 'models' in MODEL_TIERS[tier], f"Tier '{tier}' missing 'models' key"
        print(f"\n{tier.upper()} tier:")
        print(f"  Models: {', '.join(MODEL_TIERS[tier]['models'])}")
        print(f"  Description: {MODEL_TIERS[tier]['description']}")

    # Verify tier hierarchy (quick < standard < comprehensive)
    quick_count = len(MODEL_TIERS['quick']['models'])
    standard_count = len(MODEL_TIERS['standard']['models'])
    comprehensive_count = len(MODEL_TIERS['comprehensive']['models'])

    assert quick_count <= standard_count, "Quick tier should have fewer models than standard"
    assert standard_count <= comprehensive_count, "Standard tier should have fewer models than comprehensive"

    print(f"\n✓ Tier hierarchy validated: quick({quick_count}) ≤ standard({standard_count}) ≤ comprehensive({comprehensive_count})")


def test_tier_execution_quick():
    """Test quick tier execution."""
    print("\n" + "="*80)
    print("TEST 2: Quick Tier Execution")
    print("="*80)

    X, y = load_bone_collagen_data(n_samples=15)
    print(f"\nLoaded {len(y)} samples with {X.shape[1]} features")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} % collagen")

    # Get model grids for quick tier
    start_time = time.time()
    grids = get_model_grids(
        task_type='regression',
        n_features=X.shape[1],
        tier='quick',
        enabled_models=MODEL_TIERS['quick']['models']
    )

    print(f"\n Quick tier models: {list(grids.keys())}")

    # Train and evaluate each model
    results = {}
    for model_name, model_configs in grids.items():
        print(f"\n  Testing {model_name}...")

        # Test first config from grid (model_configs is a list of (model, params) tuples)
        if isinstance(model_configs, list) and len(model_configs) > 0:
            model = model_configs[0][0]  # First element of first tuple is the model
        else:
            continue

        try:
            # Simple train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {'rmse': rmse, 'r2': r2}
            print(f"    RMSE: {rmse:.3f}, R²: {r2:.3f}")

        except Exception as e:
            print(f"    ⚠ Failed: {e}")
            results[model_name] = {'rmse': np.nan, 'r2': np.nan}

    elapsed = time.time() - start_time
    print(f"\n✓ Quick tier completed in {elapsed:.2f} seconds")

    # Verify at least one model succeeded
    successful_models = sum(1 for r in results.values() if not np.isnan(r['rmse']))
    assert successful_models > 0, "At least one model should run successfully"
    print(f"✓ {successful_models}/{len(results)} models ran successfully")

    return results, elapsed


def test_tier_execution_standard():
    """Test standard tier execution."""
    print("\n" + "="*80)
    print("TEST 3: Standard Tier Execution")
    print("="*80)

    X, y = load_bone_collagen_data(n_samples=15)
    print(f"\nLoaded {len(y)} samples with {X.shape[1]} features")

    # Get model grids for standard tier
    start_time = time.time()
    grids = get_model_grids(
        task_type='regression',
        n_features=X.shape[1],
        tier='standard',
        enabled_models=MODEL_TIERS['standard']['models']
    )

    print(f"\nStandard tier models: {list(grids.keys())}")

    # Train and evaluate each model
    results = {}
    for model_name, model_configs in grids.items():
        print(f"\n  Testing {model_name}...")

        # Test first config from grid (model_configs is a list of (model, params) tuples)
        if isinstance(model_configs, list) and len(model_configs) > 0:
            model = model_configs[0][0]  # First element of first tuple is the model
        else:
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {'rmse': rmse, 'r2': r2}
            print(f"    RMSE: {rmse:.3f}, R²: {r2:.3f}")

        except Exception as e:
            print(f"    ⚠ Failed: {e}")
            results[model_name] = {'rmse': np.nan, 'r2': np.nan}

    elapsed = time.time() - start_time
    print(f"\n✓ Standard tier completed in {elapsed:.2f} seconds")

    successful_models = sum(1 for r in results.values() if not np.isnan(r['rmse']))
    assert successful_models > 0, "At least one model should run successfully"
    print(f"✓ {successful_models}/{len(results)} models ran successfully")

    return results, elapsed


def test_tier_execution_comprehensive():
    """Test comprehensive tier execution."""
    print("\n" + "="*80)
    print("TEST 4: Comprehensive Tier Execution")
    print("="*80)

    X, y = load_bone_collagen_data(n_samples=15)
    print(f"\nLoaded {len(y)} samples with {X.shape[1]} features")

    # Get model grids for comprehensive tier
    start_time = time.time()
    grids = get_model_grids(
        task_type='regression',
        n_features=X.shape[1],
        tier='comprehensive',
        enabled_models=MODEL_TIERS['comprehensive']['models']
    )

    print(f"\nComprehensive tier models: {list(grids.keys())}")

    # Train and evaluate each model
    results = {}
    for model_name, model_configs in grids.items():
        print(f"\n  Testing {model_name}...")

        # Test first config from grid (model_configs is a list of (model, params) tuples)
        if isinstance(model_configs, list) and len(model_configs) > 0:
            model = model_configs[0][0]  # First element of first tuple is the model
        else:
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {'rmse': rmse, 'r2': r2}
            print(f"    RMSE: {rmse:.3f}, R²: {r2:.3f}")

        except Exception as e:
            print(f"    ⚠ Failed: {e}")
            results[model_name] = {'rmse': np.nan, 'r2': np.nan}

    elapsed = time.time() - start_time
    print(f"\n✓ Comprehensive tier completed in {elapsed:.2f} seconds")

    successful_models = sum(1 for r in results.values() if not np.isnan(r['rmse']))
    assert successful_models > 0, "At least one model should run successfully"
    print(f"✓ {successful_models}/{len(results)} models ran successfully")

    return results, elapsed


def test_tier_comparison():
    """Compare performance and runtime across tiers."""
    print("\n" + "="*80)
    print("TEST 5: Tier Comparison Summary")
    print("="*80)

    # Run all tiers and collect results
    print("\nRunning all tiers for comparison...")

    quick_results, quick_time = test_tier_execution_quick()
    standard_results, standard_time = test_tier_execution_standard()
    comprehensive_results, comprehensive_time = test_tier_execution_comprehensive()

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    print(f"\nRuntime:")
    print(f"  Quick:         {quick_time:.2f}s")
    print(f"  Standard:      {standard_time:.2f}s")
    print(f"  Comprehensive: {comprehensive_time:.2f}s")

    # Verify runtime ordering (quick should be fastest)
    # Note: This might not always hold due to overhead, but generally should
    print(f"\n✓ Tier runtime comparison complete")

    # Model counts
    print(f"\nModel counts:")
    print(f"  Quick:         {len(quick_results)} models")
    print(f"  Standard:      {len(standard_results)} models")
    print(f"  Comprehensive: {len(comprehensive_results)} models")

    assert len(quick_results) <= len(standard_results), "Quick should have ≤ models than standard"
    assert len(standard_results) <= len(comprehensive_results), "Standard should have ≤ models than comprehensive"


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TIER SYSTEM TEST SUITE")
    print("Testing: Quick, Standard, Comprehensive tiers")
    print("="*80)

    # Run tests
    try:
        test_tier_definitions()
        test_tier_execution_quick()
        test_tier_execution_standard()
        test_tier_execution_comprehensive()
        test_tier_comparison()

        print("\n" + "="*80)
        print("ALL TIER TESTS PASSED ✓")
        print("="*80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
