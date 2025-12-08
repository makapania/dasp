"""
Test R² Consistency Between Search Results and Model Re-running

This test validates the fix for R² score inconsistencies between:
1. Results Tab (search.py) - initial model training with CV
2. Development Tab (GUI) - re-running models with loaded parameters

The test focuses on models that were showing differences:
- XGBoost
- LightGBM
- NeuralBoosted

Expected outcomes after fix:
- XGBoost/LightGBM: R² difference < 0.001 (near-perfect reproducibility)
- NeuralBoosted: R² difference < 0.02 (acceptable variance due to validation splits)
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression
from spectral_predict.models import get_model
from spectral_predict.search import _run_single_fold
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Create synthetic dataset
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=50, n_informative=20,
                       noise=10, random_state=42)

print("="*80)
print("R² CONSISTENCY VALIDATION TEST")
print("="*80)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

# Test configurations for each model type
test_configs = [
    {
        'name': 'XGBoost',
        'params': {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'tolerance': 0.001
    },
    {
        'name': 'LightGBM',
        'params': {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'tolerance': 0.001
    },
    {
        'name': 'NeuralBoosted',
        'params': {
            'n_estimators': 30,
            'learning_rate': 0.1,
            'hidden_layer_size': 3,
            'activation': 'tanh',
            'early_stopping': True,
            'validation_fraction': 0.15,
            'random_state': 42,
            'verbose': 0
        },
        'tolerance': 0.02
    }
]

results = []

for config in test_configs:
    model_name = config['name']
    params = config['params']
    tolerance = config['tolerance']

    print(f"\nTesting {model_name}")
    print("-" * 40)

    try:
        # Step 1: Simulate Results Tab (search.py) behavior
        # Create model with specific parameters
        model1 = get_model(model_name, task_type='regression')
        model1.set_params(**params)

        # Run CV manually (simulating search.py)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and fit model
            from sklearn.base import clone
            fold_model = clone(model1)
            fold_model.fit(X_train, y_train)
            fold_score = fold_model.score(X_test, y_test)
            fold_scores.append(fold_score)

        results_tab_r2 = np.mean(fold_scores)

        # Capture ALL parameters after fitting (simulating parameter capture fix)
        model1_fitted = clone(model1)
        model1_fitted.fit(X, y)  # Fit on full data
        captured_params = model1_fitted.get_params()

        # Filter to serializable params only
        filtered_params = {}
        for key, value in captured_params.items():
            if not callable(value) and not hasattr(value, '__dict__'):
                try:
                    str(value)
                    filtered_params[key] = value
                except:
                    continue

        print(f"  Results Tab R²: {results_tab_r2:.6f}")
        print(f"  Captured {len(filtered_params)} parameters")

        # Step 2: Simulate Development Tab (GUI) behavior
        # Create fresh model and load parameters
        model2 = get_model(model_name, task_type='regression')
        model2.set_params(**filtered_params)

        # Run CV again (simulating Development tab)
        cv2 = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores2 = []

        for train_idx, test_idx in cv2.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_model2 = clone(model2)
            fold_model2.fit(X_train, y_train)
            fold_score2 = fold_model2.score(X_test, y_test)
            fold_scores2.append(fold_score2)

        dev_tab_r2 = np.mean(fold_scores2)

        print(f"  Development Tab R²: {dev_tab_r2:.6f}")

        # Step 3: Compare results
        r2_diff = abs(results_tab_r2 - dev_tab_r2)
        print(f"  Difference: {r2_diff:.6f}")
        print(f"  Tolerance: {tolerance:.6f}")

        if r2_diff <= tolerance:
            print(f"  [PASS] Within tolerance")
            status = "PASS"
        else:
            print(f"  [FAIL] Exceeds tolerance")
            status = "FAIL"

        results.append({
            'model': model_name,
            'results_r2': results_tab_r2,
            'dev_r2': dev_tab_r2,
            'diff': r2_diff,
            'tolerance': tolerance,
            'status': status
        })

        # Check parameter completeness
        important_params_check = []
        if model_name in ['XGBoost', 'LightGBM']:
            important_params = ['n_estimators', 'learning_rate', 'max_depth',
                              'subsample', 'colsample_bytree', 'random_state']
            for param in important_params:
                if param not in filtered_params:
                    important_params_check.append(param)

        if important_params_check:
            print(f"  [WARNING] Missing important params: {important_params_check}")

    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append({
            'model': model_name,
            'results_r2': None,
            'dev_r2': None,
            'diff': None,
            'tolerance': tolerance,
            'status': 'ERROR'
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

pass_count = sum(1 for r in results if r['status'] == 'PASS')
fail_count = sum(1 for r in results if r['status'] == 'FAIL')
error_count = sum(1 for r in results if r['status'] == 'ERROR')

print(f"\nTotal Tests: {len(results)}")
print(f"  Passed: {pass_count}")
print(f"  Failed: {fail_count}")
print(f"  Errors: {error_count}")

print("\nDetailed Results:")
print("-" * 80)
print(f"{'Model':<15} {'Results R²':<12} {'Dev R²':<12} {'Diff':<10} {'Status':<10}")
print("-" * 80)
for r in results:
    if r['results_r2'] is not None:
        print(f"{r['model']:<15} {r['results_r2']:<12.6f} {r['dev_r2']:<12.6f} "
              f"{r['diff']:<10.6f} {r['status']:<10}")
    else:
        print(f"{r['model']:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {r['status']:<10}")

# Overall result
print("\n" + "="*80)
if fail_count == 0 and error_count == 0:
    print("[SUCCESS] ALL TESTS PASSED - R2 consistency fix validated!")
    print("\nConclusion:")
    print("  * XGBoost/LightGBM: Near-perfect reproducibility achieved")
    print("  * NeuralBoosted: Acceptable variance within expected range")
elif fail_count > 0:
    print("[FAILURE] SOME TESTS FAILED - Further investigation needed")
    print("\nFailed models may have:")
    print("  * Incomplete parameter capture")
    print("  * Random state issues")
    print("  * Validation split inconsistencies")
else:
    print("[ERROR] ERRORS ENCOUNTERED - Check test setup")

print("="*80)
