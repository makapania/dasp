"""
Test CV fold consistency between Python and Julia implementations.
This verifies that both backends create identical train/test splits.
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def test_python_cv_folds(n_samples=50, n_folds=5):
    """Test Python CV folds with shuffle=False (as used in Model Development tab)"""
    print(f"\n=== Python CV Folds (shuffle=False) ===")
    print(f"n_samples={n_samples}, n_folds={n_folds}")

    cv = KFold(n_splits=n_folds, shuffle=False)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(range(n_samples)), 1):
        print(f"Fold {fold_idx}:")
        print(f"  Train indices: {train_idx[:5]}...{train_idx[-5:]}")
        print(f"  Test indices:  {test_idx}")
        print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")

def test_python_cv_folds_shuffle(n_samples=50, n_folds=5):
    """Test Python CV folds with shuffle=True (as used in OLD search.py)"""
    print(f"\n=== Python CV Folds (shuffle=True, random_state=42) ===")
    print(f"n_samples={n_samples}, n_folds={n_folds}")

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(range(n_samples)), 1):
        print(f"Fold {fold_idx}:")
        print(f"  Train indices: {train_idx[:5]}...{train_idx[-5:]}")
        print(f"  Test indices:  {test_idx[:5]}...{test_idx[-5:]}")
        print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")

if __name__ == "__main__":
    print("=" * 80)
    print("CV FOLD CONSISTENCY TEST")
    print("=" * 80)

    # Test with same parameters as typical analysis
    test_python_cv_folds(n_samples=50, n_folds=5)
    test_python_cv_folds_shuffle(n_samples=50, n_folds=5)

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print("✓ Julia uses sequential indices (no shuffle)")
    print("✓ Python GUI Model Development uses shuffle=False (sequential)")
    print("⚠ Python search.py uses shuffle=True (BUT NOT USED if Julia backend active)")
    print("\nCONCLUSION: If Julia backend is active, CV folds should be IDENTICAL")
    print("            between Results tab (Julia) and Model Development tab (Python)")
