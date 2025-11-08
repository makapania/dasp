"""
Test script for Neural Boosted Phase 1 fixes using Python-Julia bridge

This script tests the neural boosting implementation with:
1. Synthetic linear data
2. Synthetic nonlinear data

Phase 1 Fixes Being Tested:
- Adam learning rate at 0.01 (vs previous 0.001)
- Float64 precision throughout (vs Float32)
- Per-learner random seeds for diversity
- Convergence detection with 1e-4 tolerance
- NaN/Inf validation for gradients and predictions
"""

import numpy as np
from spectral_predict_julia_bridge import JuliaBridge
import sys

print("=" * 80)
print("NEURAL BOOSTED PHASE 1 TESTING")
print("=" * 80)
print()

# Initialize Julia bridge
print("Initializing Julia bridge...")
try:
    bridge = JuliaBridge()
    print("✓ Julia bridge initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Julia bridge: {e}")
    sys.exit(1)

print()

def r_squared(y_true, y_pred):
    """Compute R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot)

# =============================================================================
# Test 1: Synthetic Linear Data
# =============================================================================
print("Test 1: Synthetic Linear Data")
print("-" * 80)
print("Description: Simple linear relationship with 2 important features")
print()

np.random.seed(42)
n_samples = 100
n_features = 20

X1 = np.random.randn(n_samples, n_features)
# True relationship: y = 2*X1 + 3*X5 + noise
y1 = 2.0 * X1[:, 0] + 3.0 * X1[:, 4] + 0.1 * np.random.randn(n_samples)

print(f"Data: {n_samples} samples, {n_features} features")
print("True model: y = 2*X[:,0] + 3*X[:,4] + noise")
print()

try:
    # Fit model
    print("Fitting NeuralBoostedRegressor...")
    result = bridge.fit_model(
        X1, y1,
        model_name='NeuralBoosted',
        hyperparameters={
            'n_estimators': 50,
            'learning_rate': 0.1,
            'hidden_layer_size': 3,
            'activation': 'tanh',
            'max_iter': 100,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'random_state': 42
        },
        preprocess_method='raw',
        n_folds=0  # No CV, just fit
    )

    print()
    print(f"Julia output: {result.get('message', 'N/A')}")

    if result['success']:
        # Make predictions
        y1_pred = bridge.predict(X1, result['model_id'])
        r2_train = r_squared(y1, y1_pred)

        print()
        print("Results:")
        print(f"  Training R²: {r2_train:.4f}")
        print(f"  Expected R²: > 0.90 (simple linear problem)")

        if r2_train > 0.90:
            print("  ✓ PASS: Model successfully learned linear relationship")
        elif r2_train > 0.50:
            print("  ⚠ PARTIAL: Model learned something but not great")
        else:
            print("  ✗ FAIL: Model failed to learn")

    else:
        print("✗ FAILED:")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        if 'No weak learners' in result.get('error', ''):
            print("  This is the MAIN PROBLEM: All weak learners are failing!")

except Exception as e:
    print(f"✗ FAILED with exception:")
    print(f"  Error: {e}")

print()
print()

# =============================================================================
# Test 2: Synthetic Nonlinear Data
# =============================================================================
print("Test 2: Synthetic Nonlinear Data")
print("-" * 80)
print("Description: Nonlinear relationship (requires neural network)")
print()

np.random.seed(42)
X2 = np.random.randn(n_samples, n_features)
# Nonlinear relationship: y = X1^2 + sin(X2) + noise
y2 = X2[:, 0]**2 + np.sin(X2[:, 1]) + 0.2 * np.random.randn(n_samples)

print(f"Data: {n_samples} samples, {n_features} features")
print("True model: y = X[:,0]² + sin(X[:,1]) + noise")
print()

try:
    print("Fitting NeuralBoostedRegressor...")
    result = bridge.fit_model(
        X2, y2,
        model_name='NeuralBoosted',
        hyperparameters={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'hidden_layer_size': 5,
            'activation': 'tanh',
            'max_iter': 150,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'random_state': 42
        },
        preprocess_method='raw',
        n_folds=0
    )

    print()
    print(f"Julia output: {result.get('message', 'N/A')}")

    if result['success']:
        y2_pred = bridge.predict(X2, result['model_id'])
        r2_train = r_squared(y2, y2_pred)

        print()
        print("Results:")
        print(f"  Training R²: {r2_train:.4f}")
        print(f"  Expected R²: > 0.60 (nonlinear problem, harder)")

        if r2_train > 0.60:
            print("  ✓ PASS: Model successfully learned nonlinear relationship")
        elif r2_train > 0.30:
            print("  ⚠ PARTIAL: Model learned something but not great")
        else:
            print("  ✗ FAIL: Model failed to learn")

    else:
        print("✗ FAILED:")
        print(f"  Error: {result.get('error', 'Unknown error')}")

except Exception as e:
    print(f"✗ FAILED with exception:")
    print(f"  Error: {e}")

print()
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 80)
print("PHASE 1 TESTING SUMMARY")
print("=" * 80)
print()
print("Phase 1 Fixes Applied:")
print("  ✓ Adam learning rate: 0.01 (was 0.001)")
print("  ✓ Float64 precision (was Float32)")
print("  ✓ Per-learner random seeds for diversity")
print("  ✓ Convergence detection (1e-4 tolerance)")
print("  ✓ NaN/Inf validation")
print()
print("Next Steps:")
print("  - If tests PASS: Phase 1 is sufficient!")
print("  - If tests show <10% success rate: Proceed to Phase 2 (Optim.jl LBFGS)")
print("  - Phase 2 would replace Adam with LBFGS optimizer (like Python/sklearn)")
print()
print("=" * 80)
