"""Simplified test to isolate LightGBM/XGBoost issues"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

# Generate SIMPLE synthetic data (not high-dimensional)
np.random.seed(42)
n_samples = 100
n_features = 20  # Much smaller feature space

X = np.random.randn(n_samples, n_features)
y = X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

print("=" * 80)
print("SIMPLIFIED TEST - LOW DIMENSIONAL DATA")
print("=" * 80)
print(f"Data: {n_samples} samples × {n_features} features")
print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
print()

# Test 1: LightGBM with conservative settings
print("Test 1: LightGBM with VERY conservative settings")
print("-" * 60)
lgbm_conservative = LGBMRegressor(
    n_estimators=50,
    learning_rate=0.05,
    num_leaves=7,  # Very small
    max_depth=3,  # Limit depth
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
scores = cross_val_score(lgbm_conservative, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
print()

# Test 2: XGBoost with conservative settings
print("Test 2: XGBoost with VERY conservative settings")
print("-" * 60)
xgb_conservative = XGBRegressor(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
scores = cross_val_score(xgb_conservative, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
print()

# Test 3: PLS
print("Test 3: PLS (baseline)")
print("-" * 60)
pls = PLSRegression(n_components=5, scale=False)
scores = cross_val_score(pls, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
print()

# Test 4: RF
print("Test 4: RandomForest (baseline)")
print("-" * 60)
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
print()

print("=" * 80)
print("Now testing HIGH-DIMENSIONAL data (like spectroscopy)")
print("=" * 80)

# Generate high-dimensional data
X_highdim = np.random.randn(n_samples, 500)
y_highdim = np.mean(X_highdim[:, 50:60], axis=1) + np.random.randn(n_samples) * 0.5

print(f"Data: {n_samples} samples × 500 features")
print()

# Test LightGBM on high-dim
print("Test: LightGBM on high-dimensional data")
print("-" * 60)
scores = cross_val_score(lgbm_conservative, X_highdim, y_highdim, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
print()

# Test XGBoost on high-dim
print("Test: XGBoost on high-dimensional data")
print("-" * 60)
scores = cross_val_score(xgb_conservative, X_highdim, y_highdim, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
print()

# Test PLS on high-dim
print("Test: PLS on high-dimensional data")
print("-" * 60)
pls_highdim = PLSRegression(n_components=10, scale=False)
scores = cross_val_score(pls_highdim, X_highdim, y_highdim, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
