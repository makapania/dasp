"""
LightGBM Isolated Test Script

Tests LightGBM in isolation to identify the root cause of negative R² values.
Tests progressively more complex scenarios to pinpoint where the issue occurs.
"""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("="*80)
print("LightGBM Isolated Test Suite")
print("="*80)

# Check LightGBM version
import lightgbm
print(f"\nLightGBM version: {lightgbm.__version__}")

# Test 1: Simple low-dimensional dataset
print("\n" + "="*80)
print("Test 1: Simple Dataset (100 samples, 50 features)")
print("="*80)
try:
    X, y = make_regression(n_samples=100, n_features=50, random_state=42, noise=10)
    model = LGBMRegressor(random_state=42, verbosity=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"[OK] Success: Mean R2 = {np.mean(scores):.4f} (+/-{np.std(scores):.4f})")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in scores]}")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# Test 2: High-dimensional dataset (like spectral data)
print("\n" + "="*80)
print("Test 2: High-Dimensional Dataset (100 samples, 2000 features)")
print("="*80)
try:
    X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
    model = LGBMRegressor(random_state=42, verbosity=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"[OK] Success: Mean R2 = {np.mean(scores):.4f} (+/-{np.std(scores):.4f})")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in scores]}")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# Test 3: High-dimensional with regularization (matching models.py config)
print("\n" + "="*80)
print("Test 3: High-Dimensional with Regularization")
print("="*80)
try:
    X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"[OK] Success: Mean R2 = {np.mean(scores):.4f} (+/-{np.std(scores):.4f})")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in scores]}")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# Test 4: With preprocessing pipeline (like actual code)
print("\n" + "="*80)
print("Test 4: High-Dimensional with Pipeline (StandardScaler)")
print("="*80)
try:
    X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    print(f"[OK] Success: Mean R2 = {np.mean(scores):.4f} (+/-{np.std(scores):.4f})")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in scores]}")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# Test 5: Very small sample size (might be issue with CV splits)
print("\n" + "="*80)
print("Test 5: Small Sample Size (50 samples, 2000 features)")
print("="*80)
try:
    X, y = make_regression(n_samples=50, n_features=2000, random_state=42, noise=10)
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    scores = cross_val_score(model, X, y, cv=3, scoring='r2')  # Use 3-fold for small data
    print(f"[OK] Success: Mean R2 = {np.mean(scores):.4f} (+/-{np.std(scores):.4f})")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in scores]}")
    if np.mean(scores) < 0:
        print("  [WARNING] Negative R2 detected! This is the problem scenario.")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# Test 6: Data type test (float32 vs float64)
print("\n" + "="*80)
print("Test 6: Data Type Test (float32 vs float64)")
print("="*80)
try:
    X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)

    # Test with float64 (default)
    model = LGBMRegressor(random_state=42, verbosity=-1)
    scores_64 = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  float64: Mean R2 = {np.mean(scores_64):.4f}")

    # Test with float32
    X_32 = X.astype(np.float32)
    y_32 = y.astype(np.float32)
    scores_32 = cross_val_score(model, X_32, y_32, cv=5, scoring='r2')
    print(f"  float32: Mean R2 = {np.mean(scores_32):.4f}")

    print("[OK] Both data types tested successfully")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# Test 7: Verbose output to see training details
print("\n" + "="*80)
print("Test 7: Verbose Training (see what LightGBM is doing)")
print("="*80)
try:
    X, y = make_regression(n_samples=100, n_features=2000, random_state=42, noise=10)
    model = LGBMRegressor(
        n_estimators=50,  # Fewer estimators for shorter output
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbosity=1  # Enable verbose output
    )
    # Train on full data (no CV) to see training output
    model.fit(X, y)
    train_r2 = model.score(X, y)
    print(f"\n[OK] Training R2 = {train_r2:.4f}")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("""
If all tests show positive R² values:
  → LightGBM installation is working correctly
  → Issue is likely in the integration with the main codebase
  → Check preprocessing pipeline, data format, or parameter passing

If any test shows negative R²:
  → Note which test failed (simple vs high-dim, with/without pipeline, etc.)
  → This indicates where the problem lies

Next steps based on results:
  1. If isolated tests work → investigate search.py integration
  2. If tests fail → check LightGBM installation, version compatibility
  3. If only small sample size fails → adjust min_child_samples parameter
""")
print("="*80)
