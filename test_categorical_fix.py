"""
Test script to validate the categorical target fix in read_combined_excel() and read_combined_csv().

This test demonstrates that:
1. Regression targets (numeric) still work correctly
2. Classification targets (categorical text) are preserved (not converted to NaN)
3. Rows with truly missing values are still filtered out
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from spectral_predict.io import read_combined_excel

def create_test_data():
    """Create test Excel file with both numeric and categorical targets."""

    # Create synthetic spectral data (100 wavelengths from 400-2400nm)
    wavelengths = np.linspace(400, 2400, 100)
    n_samples = 50

    # Generate random reflectance spectra
    spectra = np.random.rand(n_samples, 100) * 0.5 + 0.3  # Values between 0.3 and 0.8

    # Create DataFrame
    data = {}

    # Add specimen IDs
    data['specimen_id'] = [f"Sample_{i+1}" for i in range(n_samples)]

    # Add spectral columns
    for i, wl in enumerate(wavelengths):
        data[str(int(wl))] = spectra[:, i]

    # Add REGRESSION target (numeric with some NaN)
    collagen_values = np.random.rand(n_samples) * 10 + 2  # Values between 2-12
    # Set some to NaN to test filtering
    collagen_values[5] = np.nan
    collagen_values[15] = np.nan
    collagen_values[25] = np.nan
    data['collagen_percent'] = collagen_values

    # Add CLASSIFICATION target (categorical with some empty)
    categories = ['Clean', 'Contaminated', 'Unknown', 'Control']
    classification_values = [categories[i % 4] for i in range(n_samples)]
    # Set some to empty/NaN to test filtering
    classification_values[10] = ''
    classification_values[20] = np.nan
    classification_values[30] = None
    data['classification_tag'] = classification_values

    df = pd.DataFrame(data)

    # Save to Excel
    test_file = Path(__file__).parent / 'test_categorical_data.xlsx'
    df.to_excel(test_file, index=False)

    return test_file, n_samples

def test_regression_target():
    """Test that numeric (regression) targets work correctly."""
    print("\n" + "="*70)
    print("TEST 1: REGRESSION TARGET (numeric 'collagen_percent')")
    print("="*70)

    test_file, n_samples = create_test_data()

    try:
        X, y, metadata = read_combined_excel(
            test_file,
            y_col='collagen_percent'
        )

        # Expected: 50 total - 3 with NaN in collagen = 47 samples
        expected_samples = n_samples - 3

        print(f"Total samples in file: {n_samples}")
        print(f"Samples with NaN in collagen_percent: 3")
        print(f"Expected loaded samples: {expected_samples}")
        print(f"Actually loaded samples: {len(X)}")
        print(f"Target dtype: {y.dtype}")
        print(f"Target sample values: {y[:5].tolist()}")

        # Verify numeric conversion happened
        assert pd.api.types.is_numeric_dtype(y), "Regression target should be numeric"
        assert len(X) == expected_samples, f"Expected {expected_samples} samples, got {len(X)}"
        assert len(y) == expected_samples, f"Expected {expected_samples} targets, got {len(y)}"

        print("\nRESULT: PASS - Regression target works correctly")
        return True

    except Exception as e:
        print(f"\nRESULT: FAIL - {e}")
        return False

def test_classification_target():
    """Test that categorical (classification) targets are preserved."""
    print("\n" + "="*70)
    print("TEST 2: CLASSIFICATION TARGET (categorical 'classification_tag')")
    print("="*70)

    test_file, n_samples = create_test_data()

    try:
        X, y, metadata = read_combined_excel(
            test_file,
            y_col='classification_tag'
        )

        # Expected: 50 total - 3 with empty/NaN in classification = 47 samples
        expected_samples = n_samples - 3

        print(f"Total samples in file: {n_samples}")
        print(f"Samples with empty/NaN in classification_tag: 3")
        print(f"Expected loaded samples: {expected_samples}")
        print(f"Actually loaded samples: {len(X)}")
        print(f"Target dtype: {y.dtype}")
        print(f"Target sample values: {y[:10].tolist()}")
        print(f"Unique categories: {sorted(y.unique())}")

        # THE CRITICAL FIX: Verify categorical data was NOT converted to NaN
        assert len(X) == expected_samples, f"Expected {expected_samples} samples, got {len(X)}"
        assert len(y) == expected_samples, f"Expected {expected_samples} targets, got {len(y)}"

        # BEFORE FIX: This would have been 0 rows because all categorical values became NaN
        # AFTER FIX: Should have 47 rows with categorical values preserved

        # Verify categorical values are preserved
        assert y.dtype == object, f"Classification target should be object/string dtype, got {y.dtype}"
        unique_vals = set(y.unique())
        expected_vals = {'Clean', 'Contaminated', 'Unknown', 'Control'}
        assert unique_vals == expected_vals, f"Expected categories {expected_vals}, got {unique_vals}"

        # Verify no NaN values in the result (they should have been filtered out)
        assert not y.isna().any(), "No NaN values should remain in categorical target"

        print("\nRESULT: PASS - Classification target preserved correctly!")
        print("  - Categorical values were NOT converted to NaN")
        print("  - Only truly missing values were filtered out")
        print("  - All expected categories are present")
        return True

    except Exception as e:
        print(f"\nRESULT: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_scenario():
    """Test the actual user scenario: switching between targets."""
    print("\n" + "="*70)
    print("TEST 3: USER SCENARIO - Switching from regression to classification")
    print("="*70)

    test_file, n_samples = create_test_data()

    try:
        # Step 1: Load with regression target (mimics auto-detection)
        print("\nStep 1: Initial load with 'collagen_percent' (sparse, only 47 values)")
        X1, y1, meta1 = read_combined_excel(test_file, y_col='collagen_percent')
        print(f"  Loaded {len(X1)} samples")

        # Step 2: User switches to classification target and reloads
        print("\nStep 2: User switches to 'classification_tag' (complete, 47 values)")
        print("  BEFORE FIX: Would have loaded 0 rows (all categorical values became NaN)")
        print("  AFTER FIX: Should load 47 rows (categorical values preserved)")

        X2, y2, meta2 = read_combined_excel(test_file, y_col='classification_tag')
        print(f"  Actually loaded {len(X2)} samples")

        # Verify the fix worked
        expected_samples = n_samples - 3  # 3 samples have missing classification
        assert len(X2) == expected_samples, f"Expected {expected_samples} samples, got {len(X2)}"
        assert len(X2) > 0, "CRITICAL BUG: Got ZERO rows when switching to categorical target!"

        print(f"\nRESULT: PASS - Successfully loaded {len(X2)} samples with categorical target")
        print("  BUG IS FIXED: Switching to classification target no longer results in 0 rows")
        return True

    except Exception as e:
        print(f"\nRESULT: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Remove test file."""
    test_file = Path(__file__).parent / 'test_categorical_data.xlsx'
    if test_file.exists():
        test_file.unlink()
        print(f"\nCleaned up test file: {test_file}")

if __name__ == "__main__":
    print("="*70)
    print("CATEGORICAL TARGET FIX VALIDATION")
    print("="*70)
    print("\nTesting fix for bug where categorical targets were converted to NaN")
    print("Bug location: src/spectral_predict/io.py")
    print("  - read_combined_excel() line ~2462")
    print("  - read_combined_csv() line ~1100")

    try:
        results = []
        results.append(test_regression_target())
        results.append(test_classification_target())
        results.append(test_mixed_scenario())

        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)

        if all(results):
            print("ALL TESTS PASSED")
            print("\nThe fix successfully:")
            print("  1. Preserves numeric targets for regression")
            print("  2. Preserves categorical targets for classification")
            print("  3. Only filters truly missing values (NaN/empty)")
            print("  4. Solves the user's bug: switching to classification no longer gives 0 rows")
        else:
            print(f"SOME TESTS FAILED: {sum(results)}/{len(results)} passed")

    finally:
        cleanup()
