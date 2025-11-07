#!/usr/bin/env python3
"""
Minimal test to demonstrate the shuffle=True vs shuffle=False bug.

This test creates synthetic data and shows how CV splitting strategy
affects R² scores, which is the root cause of the Tab 7 bug.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.base import clone

def create_sorted_dataset(n_samples=49, n_features=50):
    """
    Create synthetic dataset that is SORTED by target value.
    This mimics real-world scenarios where data may have systematic ordering.
    """
    np.random.seed(42)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create target with linear relationship + noise
    true_coefficients = np.random.randn(n_features)
    y = X @ true_coefficients + np.random.randn(n_samples) * 0.5

    # CRITICAL: Sort by target value (simulates systematic data ordering)
    sorted_indices = np.argsort(y)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]

    print(f"Created synthetic dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  y range: [{y_sorted.min():.2f}, {y_sorted.max():.2f}]")
    print(f"  Data is SORTED by target value")

    return X_sorted, y_sorted

def test_cv_with_shuffle(X, y, n_folds=5, shuffle=True, random_state=42):
    """
    Test cross-validation with specified shuffle parameter.
    """
    if shuffle:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        print(f"\n{'='*80}")
        print(f"Testing with shuffle=TRUE, random_state={random_state}")
        print(f"{'='*80}")
    else:
        cv = KFold(n_splits=n_folds, shuffle=False)
        print(f"\n{'='*80}")
        print(f"Testing with shuffle=FALSE (sequential splits)")
        print(f"{'='*80}")

    # Model (using Lasso as in user's bug report)
    model = Lasso(alpha=0.01, random_state=42)

    fold_r2 = []
    fold_train_r2 = []

    print(f"\nRunning {n_folds}-fold CV...")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        model_fold = clone(model)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Show fold statistics
        print(f"\n  Fold {fold_idx + 1}:")
        print(f"    Train indices: {train_idx[:5]}...{train_idx[-5:]}")
        print(f"    Test indices:  {test_idx[:5]}...{test_idx[-5:]}")
        print(f"    Train y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        print(f"    Test y range:  [{y_test.min():.2f}, {y_test.max():.2f}]")

        # Train and evaluate
        model_fold.fit(X_train, y_train)

        y_train_pred = model_fold.predict(X_train)
        y_test_pred = model_fold.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        fold_train_r2.append(train_r2)
        fold_r2.append(test_r2)

        print(f"    Train R²: {train_r2:.4f}")
        print(f"    Test R²:  {test_r2:.4f}")

    # Summary
    mean_train_r2 = np.mean(fold_train_r2)
    std_train_r2 = np.std(fold_train_r2)
    mean_test_r2 = np.mean(fold_r2)
    std_test_r2 = np.std(fold_r2)

    print(f"\n{'-'*80}")
    print(f"SUMMARY:")
    print(f"  Train R²: {mean_train_r2:.4f} ± {std_train_r2:.4f}")
    print(f"  Test R²:  {mean_test_r2:.4f} ± {std_test_r2:.4f}")
    print(f"{'-'*80}")

    return {
        'mean_test_r2': mean_test_r2,
        'std_test_r2': std_test_r2,
        'fold_r2': fold_r2
    }

def main():
    """Main test function."""
    print("="*80)
    print("TESTING: Impact of shuffle parameter on CV results")
    print("="*80)
    print("\nThis test demonstrates why Tab 7 gets wrong R² values.")
    print("The bug is: Results tab uses shuffle=True, Tab 7 uses shuffle=False")
    print()

    # Create sorted dataset (mimics real-world data)
    X, y = create_sorted_dataset(n_samples=49, n_features=50)

    # Test 1: shuffle=True (Results tab way)
    results_tab = test_cv_with_shuffle(X, y, n_folds=5, shuffle=True, random_state=42)

    # Test 2: shuffle=False (Tab 7 way - THE BUG!)
    tab7 = test_cv_with_shuffle(X, y, n_folds=5, shuffle=False)

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    print(f"\nResults Tab (shuffle=True):  R² = {results_tab['mean_test_r2']:.4f} ± {results_tab['std_test_r2']:.4f}")
    print(f"Tab 7 (shuffle=False):       R² = {tab7['mean_test_r2']:.4f} ± {tab7['std_test_r2']:.4f}")

    delta = tab7['mean_test_r2'] - results_tab['mean_test_r2']
    print(f"\nDelta (Tab 7 - Results):     ΔR² = {delta:+.4f}")

    print(f"\n{'='*80}")
    if abs(delta) > 0.05:
        print("❌ BUG CONFIRMED: shuffle parameter causes SIGNIFICANT R² difference!")
        print()
        print("EXPLANATION:")
        print("  When data is sorted by target value (common in real-world scenarios),")
        print("  sequential folds (shuffle=False) create train/test distributions that")
        print("  are very different, leading to poor generalization.")
        print()
        print("  Example with 5 folds on sorted data:")
        print("    - Fold 1: Train on samples 10-48, test on samples 0-9 (lowest y values)")
        print("    - Fold 5: Train on samples 0-38, test on samples 39-48 (highest y values)")
        print()
        print("  This creates systematic bias where each fold tests on a different")
        print("  region of the target distribution, leading to unreliable R² estimates.")
        print()
        print("FIX:")
        print("  Change Tab 7 line 2406 in spectral_predict_gui_optimized.py:")
        print("    FROM: cv = KFold(n_splits=n_folds, shuffle=False)")
        print("    TO:   cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)")
    else:
        print("✅ No significant difference - shuffle is not the root cause")
        print("   Need to investigate other potential causes (hyperparameters, etc.)")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
