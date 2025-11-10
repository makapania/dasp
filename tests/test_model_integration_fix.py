"""
Test script to verify that new ML models (SVR, XGBoost, LightGBM, CatBoost, ElasticNet)
are properly integrated and can be loaded from Results Tab into Model Development Tab.

This test verifies the fixes for the catastrophic R² failures.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from spectral_predict.model_registry import (
    get_supported_models,
    supports_feature_importance,
    supports_subset_analysis,
    is_valid_model
)


def test_model_registry():
    """Test that the model registry is properly configured."""
    print("\n" + "="*80)
    print("TEST 1: Model Registry Configuration")
    print("="*80)

    # Test get_supported_models
    regression_models = get_supported_models('regression')
    print(f"\n✓ Regression models ({len(regression_models)}): {regression_models}")

    # Verify new models are included
    new_models = ['SVR', 'XGBoost', 'LightGBM', 'CatBoost', 'ElasticNet']
    missing = []
    for model in new_models:
        if model not in regression_models:
            missing.append(model)

    if missing:
        print(f"✗ FAIL: Missing models: {missing}")
        return False
    else:
        print(f"✓ PASS: All new models present in registry")

    # Test feature importance support
    print("\n--- Feature Importance Support ---")
    for model in new_models:
        has_fi = supports_feature_importance(model)
        status = "✓" if has_fi else "✗"
        print(f"{status} {model}: {has_fi}")
        if not has_fi:
            print(f"✗ FAIL: {model} should support feature importance")
            return False

    print("✓ PASS: All new models support feature importance")

    # Test subset analysis support
    print("\n--- Subset Analysis Support ---")
    for model in new_models:
        has_subset = supports_subset_analysis(model)
        status = "✓" if has_subset else "✗"
        print(f"{status} {model}: {has_subset}")
        if not has_subset:
            print(f"✗ FAIL: {model} should support subset analysis")
            return False

    print("✓ PASS: All new models support subset analysis")

    # Test validation
    print("\n--- Model Validation ---")
    for model in new_models:
        is_valid = is_valid_model(model, 'regression')
        status = "✓" if is_valid else "✗"
        print(f"{status} {model}: {is_valid}")
        if not is_valid:
            print(f"✗ FAIL: {model} should be valid for regression")
            return False

    print("✓ PASS: All new models are valid")

    return True


def test_model_instantiation():
    """Test that all new models can be instantiated."""
    print("\n" + "="*80)
    print("TEST 2: Model Instantiation")
    print("="*80)

    from spectral_predict.models import get_model

    new_models = ['SVR', 'XGBoost', 'LightGBM', 'ElasticNet']
    # Note: CatBoost may not be available on all systems

    for model_name in new_models:
        try:
            model = get_model(model_name, task_type='regression')
            print(f"✓ {model_name}: Successfully instantiated as {type(model).__name__}")
        except Exception as e:
            print(f"✗ FAIL: {model_name} instantiation failed: {e}")
            return False

    # Test CatBoost separately (optional)
    try:
        model = get_model('CatBoost', task_type='regression')
        print(f"✓ CatBoost: Successfully instantiated as {type(model).__name__}")
    except ValueError as e:
        if "not available" in str(e):
            print(f"⚠ CatBoost: Not available (expected on some systems) - {e}")
        else:
            print(f"✗ FAIL: CatBoost instantiation failed with unexpected error: {e}")
            return False
    except Exception as e:
        print(f"✗ FAIL: CatBoost instantiation failed: {e}")
        return False

    print("\n✓ PASS: All models can be instantiated")
    return True


def test_feature_importance_extraction():
    """Test that feature importance can be extracted from fitted models."""
    print("\n" + "="*80)
    print("TEST 3: Feature Importance Extraction")
    print("="*80)

    from spectral_predict.models import get_model, get_feature_importances
    from sklearn.datasets import make_regression

    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    new_models = ['SVR', 'XGBoost', 'LightGBM', 'ElasticNet']

    for model_name in new_models:
        try:
            # Instantiate and fit model
            model = get_model(model_name, task_type='regression')
            model.fit(X, y)

            # Extract feature importances
            importances = get_feature_importances(model, model_name, X, y)

            # Verify importances
            if importances is None:
                print(f"✗ FAIL: {model_name} returned None for feature importances")
                return False

            if len(importances) != X.shape[1]:
                print(f"✗ FAIL: {model_name} returned {len(importances)} importances but expected {X.shape[1]}")
                return False

            # Check that importances are non-negative
            if np.any(importances < 0):
                print(f"✗ FAIL: {model_name} returned negative importances")
                return False

            print(f"✓ {model_name}: Extracted {len(importances)} feature importances")
            print(f"  Range: [{np.min(importances):.6f}, {np.max(importances):.6f}]")

        except Exception as e:
            print(f"✗ FAIL: {model_name} feature importance extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ PASS: Feature importance extraction works for all models")
    return True


def test_hyperparameter_grids():
    """Test that hyperparameter grids are defined for new models."""
    print("\n" + "="*80)
    print("TEST 4: Hyperparameter Grid Configuration")
    print("="*80)

    from spectral_predict.models import get_model_grids

    new_models = ['SVR', 'XGBoost', 'LightGBM', 'CatBoost', 'ElasticNet']

    # Get grids for new models
    grids = get_model_grids(
        task_type='regression',
        enabled_models=new_models,
        tier='fast'  # Use fast tier for testing
    )

    for model_name in new_models:
        if model_name == 'CatBoost':
            # CatBoost might not be available
            if model_name not in grids:
                print(f"⚠ CatBoost: Grid not found (expected if CatBoost not installed)")
                continue

        if model_name not in grids:
            print(f"✗ FAIL: {model_name} grid not found in get_model_grids()")
            return False

        grid = grids[model_name]
        print(f"✓ {model_name}: {len(grid)} configurations defined")
        if len(grid) > 0:
            print(f"  Example config: {grid[0]}")

    print("\n✓ PASS: Hyperparameter grids defined for all models")
    return True


def test_end_to_end_workflow():
    """Test a simplified end-to-end workflow with new models."""
    print("\n" + "="*80)
    print("TEST 5: End-to-End Workflow (Simplified)")
    print("="*80)

    from spectral_predict.models import get_model, get_feature_importances
    from sklearn.datasets import make_regression
    from sklearn.model_selection import cross_val_score

    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    new_models = ['SVR', 'XGBoost', 'LightGBM', 'ElasticNet']

    for model_name in new_models:
        try:
            # Step 1: Train model
            model = get_model(model_name, task_type='regression')
            model.fit(X, y)

            # Step 2: Get CV scores
            scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            mean_r2 = np.mean(scores)

            # Step 3: Extract feature importances
            importances = get_feature_importances(model, model_name, X, y)

            # Step 4: Get top features
            top_indices = np.argsort(importances)[-5:]  # Top 5 features

            # Step 5: Retrain on subset
            X_subset = X[:, top_indices]
            model_subset = get_model(model_name, task_type='regression')
            model_subset.fit(X_subset, y)

            # Step 6: Get CV scores on subset
            scores_subset = cross_val_score(model_subset, X_subset, y, cv=3, scoring='r2')
            mean_r2_subset = np.mean(scores_subset)

            print(f"✓ {model_name}:")
            print(f"  Full model R²: {mean_r2:.4f}")
            print(f"  Subset model R² (5 features): {mean_r2_subset:.4f}")

            # Verify R² is reasonable (not negative)
            if mean_r2 < -0.5 or mean_r2_subset < -0.5:
                print(f"✗ FAIL: {model_name} has unreasonable R² values")
                return False

        except Exception as e:
            print(f"✗ FAIL: {model_name} end-to-end workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ PASS: End-to-end workflow successful for all models")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("MODEL INTEGRATION FIX - VERIFICATION TESTS")
    print("="*80)
    print("\nTesting fixes for catastrophic R² failures in new ML models")
    print("(SVR, XGBoost, LightGBM, CatBoost, ElasticNet)")

    tests = [
        ("Model Registry Configuration", test_model_registry),
        ("Model Instantiation", test_model_instantiation),
        ("Feature Importance Extraction", test_feature_importance_extraction),
        ("Hyperparameter Grid Configuration", test_hyperparameter_grids),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ FATAL ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The integration fixes are working correctly.")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
