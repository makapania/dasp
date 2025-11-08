"""
Diagnostic Script: CV Fold Mismatch Detector

This script helps diagnose whether CV folds in Model Development tab
match those used in Results tab (Julia backend).

Usage:
1. Run a search in Results tab
2. Load a model in Model Development tab
3. Before clicking "Run Refined Model", run this script
4. It will print the first fold's test samples for comparison

Author: Claude Code Investigation
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def diagnose_cv_folds(X_df, y_series, n_folds=5):
    """
    Diagnose CV fold generation to detect index mismatches.

    Parameters:
    -----------
    X_df : pd.DataFrame
        Feature data with potentially non-sequential index
    y_series : pd.Series
        Target data with same index as X_df
    n_folds : int
        Number of CV folds

    Returns:
    --------
    dict : Diagnostic information
    """

    print("="*80)
    print("CV FOLD DIAGNOSTIC REPORT")
    print("="*80)
    print()

    # Check 1: Index sequentiality
    print("CHECK 1: Index Sequentiality")
    print("-"*80)
    print(f"X_df shape: {X_df.shape}")
    print(f"y_series shape: {y_series.shape}")
    print(f"X_df index range: {X_df.index.min()} to {X_df.index.max()}")
    print(f"X_df index[:20]: {X_df.index.tolist()[:20]}")

    is_sequential = (X_df.index == range(len(X_df))).all()
    print(f"Is index sequential (0-based)? {is_sequential}")

    if not is_sequential:
        print("⚠ WARNING: Index is NOT sequential!")
        print("   This will cause different fold assignments than Julia backend!")
        print("   Gaps in index: ", [i for i in range(len(X_df)) if i not in X_df.index.tolist()[:50]])
    else:
        print("✓ Index is sequential - folds should match Julia")
    print()

    # Check 2: Index alignment
    print("CHECK 2: Index Alignment")
    print("-"*80)
    index_match = (X_df.index == y_series.index).all()
    print(f"X and y indices match? {index_match}")
    if not index_match:
        print("✗ ERROR: X and y indices don't match!")
        print(f"   X_df index[:10]: {X_df.index.tolist()[:10]}")
        print(f"   y_series index[:10]: {y_series.index.tolist()[:10]}")
    else:
        print("✓ X and y indices are aligned")
    print()

    # Check 3: Generate CV folds (current behavior - BUGGY if index not sequential)
    print("CHECK 3: CV Fold Generation (Current Behavior)")
    print("-"*80)
    X_values = X_df.values
    y_values = y_series.values

    cv = KFold(n_splits=n_folds, shuffle=False)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_values)):
        if fold_idx == 0:
            print(f"Fold 0 (using .values, non-sequential index):")
            print(f"  Train indices (sklearn): {train_idx[:20]}")
            print(f"  Test indices (sklearn): {test_idx[:10]}")
            print(f"  Test y values: {y_values[test_idx[:10]]}")

            # Map back to original DataFrame index
            original_test_labels = X_df.index[test_idx[:10]].tolist()
            print(f"  Test sample IDs (DataFrame index): {original_test_labels}")
            break
    print()

    # Check 4: Generate CV folds (FIXED behavior - reset index)
    print("CHECK 4: CV Fold Generation (FIXED Behavior - Reset Index)")
    print("-"*80)
    X_df_reset = X_df.reset_index(drop=True)
    y_series_reset = y_series.reset_index(drop=True)

    X_values_reset = X_df_reset.values
    y_values_reset = y_series_reset.values

    cv_reset = KFold(n_splits=n_folds, shuffle=False)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_reset.split(X_values_reset)):
        if fold_idx == 0:
            print(f"Fold 0 (after reset_index):")
            print(f"  Train indices (sklearn): {train_idx[:20]}")
            print(f"  Test indices (sklearn): {test_idx[:10]}")
            print(f"  Test y values: {y_values_reset[test_idx[:10]]}")
            break
    print()

    # Check 5: Compare fold assignments
    print("CHECK 5: Fold Assignment Comparison")
    print("-"*80)

    # Get first fold test samples for both approaches
    cv1 = KFold(n_splits=n_folds, shuffle=False)
    cv2 = KFold(n_splits=n_folds, shuffle=False)

    for fold_idx, (train1, test1) in enumerate(cv1.split(X_values)):
        if fold_idx == 0:
            test_y_buggy = y_values[test1]
            break

    for fold_idx, (train2, test2) in enumerate(cv2.split(X_values_reset)):
        if fold_idx == 0:
            test_y_fixed = y_values_reset[test2]
            break

    folds_match = np.array_equal(test_y_buggy, test_y_fixed)

    if folds_match:
        print("✓ Fold assignments MATCH between buggy and fixed versions")
        print("  (Index was already sequential, no bug present)")
    else:
        print("✗ Fold assignments DIFFER between buggy and fixed versions!")
        print("  This confirms the bug - different samples in test folds:")
        print(f"    Buggy fold 0 test y: {test_y_buggy[:10]}")
        print(f"    Fixed fold 0 test y: {test_y_fixed[:10]}")
        print()
        print("  RECOMMENDATION: Apply the fix (reset_index) in _run_refined_model_thread")
    print()

    # Check 6: Expected Julia behavior
    print("CHECK 6: Expected Julia Backend Behavior")
    print("-"*80)
    print("Julia backend receives data via CSV with sample_id column:")
    print("  - DataFrame index is preserved as 'sample_id' column")
    print("  - Julia reads this and converts to Matrix{Float64}")
    print("  - Matrix conversion creates sequential 1-based indexing")
    print("  - Julia CV uses indices [1:N] sequentially")
    print()
    print("For Julia to match Python, Python must:")
    print("  1. Reset index to sequential before CV")
    print("  2. Use shuffle=False in KFold (already done)")
    print("  3. Ensure same data order as sent to Julia")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    if is_sequential and folds_match:
        print("✓ NO BUG DETECTED - Index is sequential, folds match")
    elif not is_sequential and not folds_match:
        print("✗ BUG CONFIRMED - Non-sequential index causes fold mismatch")
        print()
        print("FIX REQUIRED:")
        print("  File: spectral_predict_gui_optimized.py")
        print("  Location: Line ~3989 (before cv = KFold(...))")
        print("  Add these 2 lines:")
        print("    X_base_df = X_base_df.reset_index(drop=True)")
        print("    y_series = y_series.reset_index(drop=True)")
    else:
        print("⚠ UNCLEAR - Index sequential but folds don't match (investigate further)")
    print("="*80)

    return {
        'is_sequential': is_sequential,
        'indices_aligned': index_match,
        'folds_match': folds_match,
        'test_y_buggy': test_y_buggy[:10] if not folds_match else None,
        'test_y_fixed': test_y_fixed[:10] if not folds_match else None
    }


def main():
    """
    Main diagnostic function - loads data from GUI context if available.
    """
    print("CV Fold Diagnostic Tool")
    print("="*80)
    print()

    # Try to import from GUI
    try:
        import tkinter as tk
        from spectral_predict_gui_optimized import SpectralPredictGUI

        print("Attempting to access GUI data...")
        # Note: This won't work unless run from within GUI context
        # User should call diagnose_cv_folds() manually with their data
        print()
        print("USAGE:")
        print("------")
        print("From GUI or Python console, call:")
        print()
        print("  from diagnose_cv_folds import diagnose_cv_folds")
        print("  diagnose_cv_folds(X_base_df, y_series, n_folds=5)")
        print()
        print("Where:")
        print("  X_base_df = Feature DataFrame (after exclusions/filtering)")
        print("  y_series = Target Series (aligned with X_base_df)")
        print("  n_folds = Number of CV folds (default: 5)")
        print()

    except ImportError:
        print("GUI not available - this is a library function")
        print()
        print("EXAMPLE USAGE:")
        print("--------------")
        print()

        # Create example data with non-sequential index (simulates exclusions)
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        # Simulate excluded samples (remove samples 10, 15, 20, 25, 30)
        all_indices = list(range(n_samples))
        excluded = [10, 15, 20, 25, 30]
        kept_indices = [i for i in all_indices if i not in excluded]

        X_data = np.random.randn(len(kept_indices), n_features)
        y_data = np.random.randn(len(kept_indices))

        X_df = pd.DataFrame(X_data, index=kept_indices)
        y_series = pd.Series(y_data, index=kept_indices)

        print(f"Created example data with {len(excluded)} excluded samples")
        print(f"Index has gaps: {kept_indices[:20]}")
        print()

        # Run diagnostics
        results = diagnose_cv_folds(X_df, y_series, n_folds=5)

        return results


if __name__ == "__main__":
    main()
