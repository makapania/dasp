"""Test LightGBM and XGBoost with current configurations"""

import numpy as np
from sklearn.model_selection import cross_val_score
from src.spectral_predict.models import get_model
from src.spectral_predict.model_config import get_hyperparameters

# Generate synthetic spectral data
np.random.seed(42)
n_samples = 100
n_wavelengths = 500

X = np.random.randn(n_samples, n_wavelengths)
for i in range(n_samples):
    X[i] += np.sin(np.linspace(0, 4*np.pi, n_wavelengths)) * (i % 10) / 10

y = np.mean(X[:, 100:200], axis=1) + 0.5 * np.mean(X[:, 300:400], axis=1) + np.random.randn(n_samples) * 0.1

print("=" * 80)
print("TESTING LIGHTGBM AND XGBOOST CONFIGURATIONS")
print("=" * 80)
print(f"\nData: {n_samples} samples × {n_wavelengths} wavelengths")
print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
print()

# Test LightGBM with new defaults
print("Testing LightGBM with FIXED defaults (reg_alpha=0.1, reg_lambda=1.0):")
print("-" * 60)
lgbm_config = get_hyperparameters('LightGBM', 'standard')
print(f"Config: {lgbm_config}")
print()

lgbm_model = get_model('LightGBM', task_type='regression')
lgbm_scores = cross_val_score(lgbm_model, X, y, cv=5, scoring='r2')
print(f"LightGBM R² scores: {lgbm_scores}")
print(f"LightGBM Mean R²: {lgbm_scores.mean():.4f} ± {lgbm_scores.std():.4f}")
print()

# Test XGBoost
print("Testing XGBoost:")
print("-" * 60)
xgb_config = get_hyperparameters('XGBoost', 'standard')
print(f"Config reg_alpha: {xgb_config['reg_alpha']}")
print(f"Config reg_lambda: {xgb_config['reg_lambda']}")
print()

xgb_model = get_model('XGBoost', task_type='regression')
xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
print(f"XGBoost R² scores: {xgb_scores}")
print(f"XGBoost Mean R²: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print()

# Test PLS for comparison (should work fine)
print("Testing PLS (baseline comparison):")
print("-" * 60)
pls_model = get_model('PLS', task_type='regression', n_components=10)
pls_scores = cross_val_score(pls_model, X, y, cv=5, scoring='r2')
print(f"PLS R² scores: {pls_scores}")
print(f"PLS Mean R²: {pls_scores.mean():.4f} ± {pls_scores.std():.4f}")
print()

# Test RandomForest for comparison
print("Testing RandomForest (baseline comparison):")
print("-" * 60)
rf_model = get_model('RandomForest', task_type='regression')
rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"RandomForest R² scores: {rf_scores}")
print(f"RandomForest Mean R²: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"LightGBM: {lgbm_scores.mean():.4f} ± {lgbm_scores.std():.4f}")
print(f"XGBoost:  {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print(f"PLS:      {pls_scores.mean():.4f} ± {pls_scores.std():.4f}")
print(f"RF:       {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print()

if lgbm_scores.mean() > 0.7:
    print("[PASS] LightGBM shows good performance (R² > 0.7)")
else:
    print("[FAIL] LightGBM shows poor performance (R² < 0.7)")

if xgb_scores.mean() > 0.7:
    print("[PASS] XGBoost shows good performance (R² > 0.7)")
else:
    print("[FAIL] XGBoost shows poor performance (R² < 0.7)")
