"""Debug script to test LightGBM vs XGBoost on sample data."""

import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression

# Create synthetic high-dimensional data similar to spectroscopy
# (small samples, many features)
n_samples = 100
n_features = 1000  # Similar to spectral data

X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=50,  # Only 50 features actually matter
    noise=10.0,
    random_state=42
)

print("=" * 80)
print("Testing LightGBM vs XGBoost on high-dimensional data")
print(f"Samples: {n_samples}, Features: {n_features}")
print("=" * 80)

# Test 1: Current LightGBM defaults (from models.py line 122-135)
print("\n1. LightGBM - Current defaults (NO regularization)")
print("-" * 80)
lgbm_current = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=1.0,  # NO subsampling
    colsample_bytree=1.0,  # Use ALL features
    reg_alpha=0.0,  # NO L1 regularization
    reg_lambda=0.0,  # NO L2 regularization
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
scores = cross_val_score(lgbm_current, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Test 2: XGBoost defaults (from models.py line 106-119)
print("\n2. XGBoost - Current defaults (WITH regularization)")
print("-" * 80)
xgb_current = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,  # Row sampling
    colsample_bytree=0.8,  # Feature sampling
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
scores = cross_val_score(xgb_current, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Test 3: LightGBM with XGBoost-like regularization
print("\n3. LightGBM - WITH regularization (XGBoost-like settings)")
print("-" * 80)
lgbm_regularized = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,  # Add row sampling
    colsample_bytree=0.8,  # Add feature sampling
    reg_alpha=0.1,  # Add L1 regularization
    reg_lambda=1.0,  # Add L2 regularization
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
scores = cross_val_score(lgbm_regularized, X, y, cv=5, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Test 4: LightGBM with even more conservative settings
print("\n4. LightGBM - VERY conservative (small num_leaves)")
print("-" * 80)
lgbm_conservative = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=15,  # Reduced from 31
    max_depth=10,  # Limited depth
    min_child_samples=20,
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
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("The current LightGBM defaults have NO regularization, which causes")
print("severe overfitting on high-dimensional spectral data.")
print("\nRecommended fix: Match XGBoost's regularization approach:")
print("  - subsample=0.8 (row sampling)")
print("  - colsample_bytree=0.8 (feature sampling)")
print("  - reg_alpha=0.1 (L1 regularization)")
print("  - reg_lambda=1.0 (L2 regularization)")
print("=" * 80)
