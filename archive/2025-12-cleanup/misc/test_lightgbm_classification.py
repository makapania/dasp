"""Test LightGBM classification to isolate the issue."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

# Create a small classification dataset similar to spectral data
print("Creating small classification dataset...")
X, y = make_classification(
    n_samples=50,  # Small like spectral datasets
    n_features=100,  # Many features like spectral data
    n_informative=10,
    n_redundant=10,
    n_classes=3,  # Multi-class
    random_state=42
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"Class distribution: {np.bincount(y)}")

# Test 1: LightGBM with bagging_freq (FIXED)
print("\n" + "="*70)
print("Test 1: LightGBM WITH bagging_freq=1 (should work)")
print("="*70)
try:
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.8,
        bagging_freq=1,  # REQUIRED!
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"✓ SUCCESS: 5-fold CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: LightGBM WITHOUT bagging_freq (BROKEN)
print("\n" + "="*70)
print("Test 2: LightGBM WITHOUT bagging_freq (might fail or warn)")
print("="*70)
try:
    model_broken = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.8,  # subsample < 1.0 but NO bagging_freq!
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    scores = cross_val_score(model_broken, X, y, cv=5, scoring='accuracy')
    print(f"Result: 5-fold CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
    print("(May have ignored subsample parameter)")
except Exception as e:
    print(f"✗ FAILED as expected: {e}")

# Test 3: Binary classification (2 classes)
print("\n" + "="*70)
print("Test 3: Binary classification (2 classes)")
print("="*70)
X_binary, y_binary = make_classification(
    n_samples=50,
    n_features=100,
    n_informative=10,
    n_classes=2,  # Binary
    random_state=42
)
try:
    model_binary = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.8,
        bagging_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    scores = cross_val_score(model_binary, X_binary, y_binary, cv=5, scoring='accuracy')
    print(f"✓ SUCCESS: 5-fold CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*70)
print("All tests complete!")
print("="*70)
