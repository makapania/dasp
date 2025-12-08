"""
Test edge cases for data_utils.py
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, r'C:\Users\sponheim\git\dasp')

from spectral_predict_v3.core.data_utils import (
    normalize_filename,
    extract_numeric_id,
    align_xy,
    validate_alignment
)

def test_normalize_filename():
    """Test filename normalization edge cases."""
    print("\n=== Testing normalize_filename ===")

    tests = [
        ("Sample 001.asd", "sample001"),
        ("SPECTRUM_042.CSV", "spectrum042"),
        ("test.SPC", "test"),
        ("NoExtension", "noextension"),
        ("Multiple Spaces Here.txt", "multiplespaceshere"),
        ("Under_Score_Name.sig", "underscorename"),
        ("MixedCase123", "mixedcase123"),
        (123, "123"),  # Non-string input
    ]

    for input_val, expected in tests:
        result = normalize_filename(input_val)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} normalize_filename({input_val!r}) = {result!r} (expected {expected!r})")

    print("normalize_filename tests complete")


def test_extract_numeric_id():
    """Test numeric ID extraction edge cases."""
    print("\n=== Testing extract_numeric_id ===")

    tests = [
        ("Spectrum00001", "1"),
        ("Sample_042.asd", "42"),
        ("test001", "1"),
        ("NoNumbers", None),
        ("123", "123"),
        ("000", "0"),  # All zeros should give "0"
        ("abc123def456", "456"),  # Should get trailing numbers only
        ("Spectrum 0001.asd", "1"),
    ]

    for input_val, expected in tests:
        result = extract_numeric_id(input_val)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} extract_numeric_id({input_val!r}) = {result!r} (expected {expected!r})")

    print("extract_numeric_id tests complete")


def test_align_xy_exact_match():
    """Test exact matching."""
    print("\n=== Testing align_xy: Exact Match ===")

    sample_ids = ["Sample001", "Sample002", "Sample003"]
    y_df = pd.DataFrame({
        'ID': ['Sample001', 'Sample002', 'Sample003'],
        'Value': [1.5, 2.3, 3.7]
    })

    y_values, info = align_xy(sample_ids, y_df, 'ID', 'Value', return_alignment_info=True)

    print(f"Y values: {y_values}")
    print(f"Matched: {info['n_matched']}/{len(sample_ids)}")
    print(f"Strategy: {info['match_strategy_used']}")

    assert info['n_matched'] == 3, f"Expected 3 matches, got {info['n_matched']}"
    assert info['match_strategy_used'] == 'exact', f"Expected exact match, got {info['match_strategy_used']}"
    assert np.allclose(y_values, [1.5, 2.3, 3.7]), f"Unexpected values: {y_values}"
    print("PASS: Exact match test passed")


def test_align_xy_normalized_match():
    """Test normalized matching (extension, spaces, case)."""
    print("\n=== Testing align_xy: Normalized Match ===")

    # Spectral files have extensions, Y file doesn't
    sample_ids = ["Spectrum 001.asd", "Spectrum 002.asd"]
    y_df = pd.DataFrame({
        'ID': ['Spectrum001', 'Spectrum002'],  # No spaces, no extension
        'Concentration': [10.5, 20.3]
    })

    y_values, info = align_xy(sample_ids, y_df, 'ID', 'Concentration', return_alignment_info=True)

    print(f"Y values: {y_values}")
    print(f"Matched: {info['n_matched']}/{len(sample_ids)}")
    print(f"Strategy: {info['match_strategy_used']}")

    assert info['n_matched'] == 2, f"Expected 2 matches, got {info['n_matched']}"
    assert info['match_strategy_used'] == 'normalized', f"Expected normalized match"
    assert np.allclose(y_values, [10.5, 20.3]), f"Unexpected values: {y_values}"
    print("PASS: Normalized match test passed")


def test_align_xy_numeric_match():
    """Test numeric ID matching with leading zeros."""
    print("\n=== Testing align_xy: Numeric Match ===")

    # Leading zeros in spectral files
    sample_ids = ["Spectrum00001.asd", "Spectrum00042.asd"]
    y_df = pd.DataFrame({
        'ID': ['1', '42'],  # Just the numbers
        'Value': [100, 200]
    })

    y_values, info = align_xy(sample_ids, y_df, 'ID', 'Value', return_alignment_info=True)

    print(f"Y values: {y_values}")
    print(f"Matched: {info['n_matched']}/{len(sample_ids)}")
    print(f"Strategy: {info['match_strategy_used']}")

    assert info['n_matched'] == 2, f"Expected 2 matches, got {info['n_matched']}"
    assert info['match_strategy_used'] == 'numeric', f"Expected numeric match"
    assert np.allclose(y_values, [100, 200]), f"Unexpected values: {y_values}"
    print("PASS: Numeric match test passed")


def test_align_xy_empty_y():
    """Test with empty Y DataFrame."""
    print("\n=== Testing align_xy: Empty Y File ===")

    sample_ids = ["Sample001", "Sample002"]
    y_df = pd.DataFrame(columns=['ID', 'Value'])  # Empty

    y_values, info = align_xy(sample_ids, y_df, 'ID', 'Value', return_alignment_info=True)

    print(f"Y values: {y_values}")
    print(f"Matched: {info['n_matched']}")

    assert info['n_matched'] == 0, f"Expected 0 matches"
    assert len(info['unmatched_spectra']) == 2, f"Expected 2 unmatched spectra"
    assert np.all(np.isnan(y_values)), f"Expected all NaN values"
    print("PASS: Empty Y file test passed")


def test_align_xy_partial_match():
    """Test with some matched, some unmatched."""
    print("\n=== Testing align_xy: Partial Match ===")

    sample_ids = ["Sample001", "Sample002", "Sample003"]
    y_df = pd.DataFrame({
        'ID': ['Sample001', 'Sample003', 'Sample999'],  # Missing 002, has extra 999
        'Value': [1.0, 3.0, 9.0]
    })

    y_values, info = align_xy(sample_ids, y_df, 'ID', 'Value', return_alignment_info=True)

    print(f"Y values: {y_values}")
    print(f"Matched: {info['n_matched']}/3")
    print(f"Unmatched spectra: {info['unmatched_spectra']}")
    print(f"Unmatched reference: {info['unmatched_reference']}")

    assert info['n_matched'] == 2, f"Expected 2 matches"
    assert 'Sample002' in info['unmatched_spectra'], "Sample002 should be unmatched"
    assert 'Sample999' in info['unmatched_reference'], "Sample999 should be unmatched"
    assert np.isclose(y_values[0], 1.0), "Sample001 value incorrect"
    assert np.isnan(y_values[1]), "Sample002 should be NaN"
    assert np.isclose(y_values[2], 3.0), "Sample003 value incorrect"
    print("PASS: Partial match test passed")


def test_align_xy_nan_values():
    """Test with NaN target values."""
    print("\n=== Testing align_xy: NaN Target Values ===")

    sample_ids = ["Sample001", "Sample002"]
    y_df = pd.DataFrame({
        'ID': ['Sample001', 'Sample002'],
        'Value': [1.5, np.nan]  # One NaN
    })

    y_values, info = align_xy(sample_ids, y_df, 'ID', 'Value', return_alignment_info=True)

    print(f"Y values: {y_values}")
    print(f"NaN dropped: {info['n_nan_dropped']}")

    assert info['n_matched'] == 2, f"Both should match"
    assert info['n_nan_dropped'] == 1, f"Should report 1 NaN"
    assert np.isclose(y_values[0], 1.5), "First value incorrect"
    assert np.isnan(y_values[1]), "Second value should be NaN"
    print("PASS: NaN values test passed")


def test_align_xy_duplicates():
    """Test with duplicate IDs in Y file."""
    print("\n=== Testing align_xy: Duplicate Y IDs ===")

    sample_ids = ["Sample001"]
    y_df = pd.DataFrame({
        'ID': ['Sample001', 'Sample001', 'Sample001'],  # Triplicates
        'Value': [1.0, 2.0, 3.0]
    })

    y_values, info = align_xy(sample_ids, y_df, 'ID', 'Value', return_alignment_info=True)

    print(f"Y values: {y_values}")

    # Should use first occurrence
    assert np.isclose(y_values[0], 1.0), f"Should use first value (1.0), got {y_values[0]}"
    print("PASS: Duplicate IDs test passed")


def test_validate_alignment():
    """Test alignment validation."""
    print("\n=== Testing validate_alignment ===")

    # Valid alignment
    y_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    is_valid, msg = validate_alignment(y_values, min_samples=3)
    print(f"Valid alignment: {is_valid}, {msg}")
    assert is_valid, "Should be valid"

    # Too few samples
    y_values = np.array([1.0, 2.0])
    is_valid, msg = validate_alignment(y_values, min_samples=10)
    print(f"Too few: {is_valid}, {msg}")
    assert not is_valid, "Should be invalid (too few)"

    # Too many NaN
    y_values = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
    is_valid, msg = validate_alignment(y_values, min_samples=3)
    print(f"Too many NaN: {is_valid}, {msg}")
    assert not is_valid, "Should be invalid (too many NaN)"

    # All NaN
    y_values = np.array([np.nan, np.nan, np.nan])
    is_valid, msg = validate_alignment(y_values, min_samples=1)
    print(f"All NaN: {is_valid}, {msg}")
    assert not is_valid, "Should be invalid (all NaN)"

    print("PASS: validate_alignment tests passed")


def run_all_tests():
    """Run all edge case tests."""
    print("="*60)
    print("RUNNING DATA_UTILS EDGE CASE TESTS")
    print("="*60)

    try:
        test_normalize_filename()
        test_extract_numeric_id()
        test_align_xy_exact_match()
        test_align_xy_normalized_match()
        test_align_xy_numeric_match()
        test_align_xy_empty_y()
        test_align_xy_partial_match()
        test_align_xy_nan_values()
        test_align_xy_duplicates()
        test_validate_alignment()

        print("\n" + "="*60)
        print("ALL TESTS PASSED PASS:")
        print("="*60)
        return True

    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED FAIL:")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
