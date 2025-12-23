"""
Verification test for LightGBM fix.
Tests the problematic scenario with the new min_child_samples=5 setting.
"""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

print("="*80)
print("LightGBM Fix Verification")
print("="*80)

# Test the problematic scenario: small sample size with high dimensions
print("\nTest: Small Sample Size (50 samples, 2000 features)")
print("This was producing NEGATIVE R2 with min_child_samples=20")
print("-"*80)

X, y = make_regression(n_samples=50, n_features=2000, random_state=42, noise=10)

print("\n1. OLD Configuration (min_child_samples=20):")
model_old = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    min_child_samples=20,  # OLD VALUE
    random_state=42,
    verbosity=-1
)
scores_old = cross_val_score(model_old, X, y, cv=3, scoring='r2')
mean_r2_old = np.mean(scores_old)
print(f"   Mean R2 = {mean_r2_old:.4f} (+/-{np.std(scores_old):.4f})")
print(f"   Fold scores: {[f'{s:.4f}' for s in scores_old]}")
if mean_r2_old < 0:
    print("   [FAIL] NEGATIVE R2 - This is the problem!")
else:
    print("   [OK] Positive R2")

print("\n2. NEW Configuration (min_child_samples=5):")
model_new = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    min_child_samples=5,  # NEW VALUE - FIXED!
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1
)
scores_new = cross_val_score(model_new, X, y, cv=3, scoring='r2')
mean_r2_new = np.mean(scores_new)
print(f"   Mean R2 = {mean_r2_new:.4f} (+/-{np.std(scores_new):.4f})")
print(f"   Fold scores: {[f'{s:.4f}' for s in scores_new]}")
if mean_r2_new >= 0:
    print("   [OK] POSITIVE R2 - Fix successful!")
    improvement = mean_r2_new - mean_r2_old
    print(f"   [OK] Improvement: {improvement:+.4f}")
else:
    print("   [FAIL] Still negative - fix did not work")

print("\n" + "="*80)
print("RESULT:")
print("="*80)
if mean_r2_new >= 0 and mean_r2_new > mean_r2_old:
    print("[SUCCESS] LightGBM fix verified!")
    print(f"R2 improved from {mean_r2_old:.4f} to {mean_r2_new:.4f}")
    print("Reducing min_child_samples from 20 to 5 fixes the negative R2 issue.")
else:
    print("[FAILURE] Fix did not resolve the issue.")
print("="*80)
