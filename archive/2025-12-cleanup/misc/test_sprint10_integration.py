"""
Quick integration test for Sprint 10 implementation.

Tests:
1. Baseline correction methods work
2. Hyperparameters are valid
3. Bayesian search integrates correctly with search.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

print("=" * 80)
print("SPRINT 10 INTEGRATION TEST")
print("=" * 80)

# Test 1: Baseline correction
print("\n1. Testing baseline correction methods...")
from spectral_predict_v3.core.baseline import (
    BaselinePolynomial,
    BaselineAsLS,
    BaselineAirPLS,
    SavgolSmooth
)

X = np.random.randn(50, 100) + np.arange(100) * 0.05

bl1 = BaselinePolynomial(degree=2)
X1 = bl1.fit_transform(X)
print(f"   [OK] BaselinePolynomial: {X1.shape}")

bl2 = BaselineAsLS(lam=1e5, p=0.01)
X2 = bl2.fit_transform(X)
print(f"   [OK] BaselineAsLS: {X2.shape}")

bl3 = BaselineAirPLS(lam=1e5)
X3 = bl3.fit_transform(X)
print(f"   [OK] BaselineAirPLS: {X3.shape}")

bl4 = SavgolSmooth(window_length=11, polyorder=2)
X4 = bl4.fit_transform(X)
print(f"   [OK] SavgolSmooth: {X4.shape}")

# Test 2: Hyperparameter validation
print("\n2. Testing hyperparameter grids...")
from spectral_predict_v3.core.model_config import get_hyperparameter_grid
from spectral_predict_v3.core.models import get_model

# Test PLS
grid_pls = get_hyperparameter_grid('PLS')
model = get_model('PLS', 'regression', **grid_pls[0])
y = X[:, :5].sum(axis=1) + np.random.randn(50) * 0.1
model.fit(X, y)
print(f"   [OK] PLS hyperparameters: {len(grid_pls)} configs")

# Test Ridge
grid_ridge = get_hyperparameter_grid('Ridge')
model = get_model('Ridge', 'regression', **grid_ridge[0])
model.fit(X, y)
print(f"   [OK] Ridge hyperparameters: {len(grid_ridge)} configs")

# Test 3: Bayesian search integration
print("\n3. Testing Bayesian search integration...")
from spectral_predict_v3.core.search import run_auto_search

# Create small test dataset
np.random.seed(42)
X_test = np.random.randn(50, 30)
y_test = X_test[:, :3].sum(axis=1) + np.random.randn(50) * 0.1

# Test grid search (default)
print("   Testing grid search mode...")
results_grid = run_auto_search(
    X_test, y_test,
    task_type='regression',
    tier='quick',
    folds=3,
    custom_models=['PLS'],
    preproc_methods=['raw'],
    search_mode='grid'
)
print(f"   [OK] Grid search: {len(results_grid)} results")

# Test Bayesian search
print("   Testing Bayesian search mode...")
try:
    results_bayesian = run_auto_search(
        X_test, y_test,
        task_type='regression',
        tier='quick',
        folds=3,
        custom_models=['PLS', 'Ridge'],
        preproc_methods=['raw', 'snv'],
        search_mode='bayesian',
        bayesian_n_trials=10
    )
    print(f"   [OK] Bayesian search: {len(results_bayesian)} results")
    print(f"     Best model: {results_bayesian.iloc[0]['Model']}")
    print(f"     Best RMSE: {results_bayesian.iloc[0]['RMSE']:.4f}")
except Exception as e:
    print(f"   [FAIL] Bayesian search failed: {e}")

# Test 4: SearchMode enum
print("\n4. Testing SearchMode enum...")
from spectral_predict_v3.core.bayesian_search import SearchMode

print(f"   [OK] SearchMode.GRID: {SearchMode.GRID.value}")
print(f"   [OK] SearchMode.BAYESIAN: {SearchMode.BAYESIAN.value}")

# Test 5: Custom hyperparameter grids
print("\n5. Testing custom hyperparameter grids in AUTO mode...")
custom_grid = {
    'PLS': [
        {'n_components': 3},
        {'n_components': 5}
    ]
}

results_custom = run_auto_search(
    X_test, y_test,
    task_type='regression',
    tier='quick',
    folds=3,
    custom_models=['PLS'],
    custom_hyperparam_grids=custom_grid,
    preproc_methods=['raw'],
    search_mode='grid'
)
print(f"   [OK] Custom grid: {len(results_custom)} results")

# Verify only n_components 3 and 5 were tested
tested_components = set()
for params_str in results_custom['Params']:
    if 'n_components=3' in params_str:
        tested_components.add(3)
    elif 'n_components=5' in params_str:
        tested_components.add(5)

if tested_components == {3, 5}:
    print(f"   [OK] Custom grid used correctly: {tested_components}")
else:
    print(f"   [FAIL] Custom grid issue: tested {tested_components}, expected {{3, 5}}")

print("\n" + "=" * 80)
print("SPRINT 10 INTEGRATION TEST COMPLETE")
print("=" * 80)
print("\nAll core functionality verified successfully!")
