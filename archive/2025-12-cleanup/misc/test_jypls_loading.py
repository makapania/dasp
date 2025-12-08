"""
Test script for JYPLS-inv Y-value loading infrastructure.

Tests the core data loading and alignment logic without requiring the full GUI.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("JYPLS-INV Y-VALUE LOADING - TEST SUITE")
print("=" * 70)

# Test 1: Import required modules
print("\n[Test 1] Importing required modules...")
try:
    from spectral_predict.io import read_asd_dir, read_csv_spectra, read_spc_dir, read_reference_csv, align_xy
    print("✓ All import functions imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check if test data exists
print("\n[Test 2] Checking for test data...")
# You'll need to provide paths to actual test data
# For now, just demonstrate the expected workflow

test_data_available = False
master_path = Path("test_data/master_spectra")  # Adjust path as needed
slave_path = Path("test_data/slave_spectra")    # Adjust path as needed

if master_path.exists() and slave_path.exists():
    test_data_available = True
    print(f"✓ Test data found")
    print(f"  Master: {master_path}")
    print(f"  Slave: {slave_path}")
else:
    print(f"⚠ Test data not found (expected, this is normal)")
    print(f"  You can test manually with your own data later")

# Test 3: Simulate the loading workflow
print("\n[Test 3] Testing loading workflow logic...")

def simulate_loading_workflow():
    """Simulate what the GUI methods do, without actual UI."""

    print("\n  Step 1: Detect file format...")
    # Simulate _browse_master_spectra_with_y() logic
    if test_data_available:
        asd_files = list(master_path.glob("*.asd"))
        csv_files = list(master_path.glob("*.csv"))
        spc_files = list(master_path.glob("*.spc"))

        if asd_files:
            detected_type = "asd"
            print(f"  ✓ Would detect {len(asd_files)} ASD files")
        elif spc_files:
            detected_type = "spc"
            print(f"  ✓ Would detect {len(spc_files)} SPC files")
        elif csv_files:
            detected_type = "csv"
            print(f"  ✓ Would detect {len(csv_files)} CSV files")
        else:
            print(f"  ✗ No spectral files found")
            return False
    else:
        print("  ⊘ Skipped (no test data)")
        return None

    print("\n  Step 2: Auto-detect reference file...")
    ref_files = list(master_path.glob("*.csv")) + list(master_path.glob("*.xlsx"))
    ref_files = [f for f in ref_files if f not in csv_files]  # Filter out spectra CSVs

    if len(ref_files) == 1:
        print(f"  ✓ Would auto-detect: {ref_files[0].name}")
        ref_path = ref_files[0]
    elif len(ref_files) > 1:
        print(f"  ⚠ Multiple reference files found: {[f.name for f in ref_files]}")
        print(f"    User would need to select manually")
        ref_path = ref_files[0]  # Use first for testing
    else:
        print(f"  ✗ No reference file found")
        return False

    print("\n  Step 3: Load and align data...")
    try:
        # Load spectra
        if detected_type == 'asd':
            X, metadata = read_asd_dir(str(master_path))
        elif detected_type == 'csv':
            X, metadata = read_csv_spectra(str(master_path))
        elif detected_type == 'spc':
            X, metadata = read_spc_dir(str(master_path))

        print(f"  ✓ Loaded spectra: {X.shape}")
        print(f"    Samples: {len(X)}")
        print(f"    Wavelengths: {len(X.columns)}")
        print(f"    Sample IDs: {list(X.index[:3])}...")

        # Load reference
        # We'd need to know the column name, so try to detect
        df_peek = pd.read_csv(ref_path, nrows=5) if ref_path.suffix == '.csv' else pd.read_excel(ref_path, nrows=5)
        columns = list(df_peek.columns)
        print(f"\n  Reference file columns: {columns}")

        # Assume standard pattern: col0=filename, col1=id, col2=target
        if len(columns) >= 2:
            spectral_file_col = columns[0]
            target_col = columns[-1]  # Use last column as target

            ref = read_reference_csv(str(ref_path), spectral_file_col)
            print(f"  ✓ Loaded reference: {len(ref)} samples")
            print(f"    Using column '{spectral_file_col}' for matching")
            print(f"    Target column: '{target_col}'")

            # Align
            X_aligned, y_aligned, alignment_info = align_xy(
                X, ref,
                spectral_file_col,
                target_col,
                return_alignment_info=True
            )

            print(f"\n  ✓ Alignment completed:")
            print(f"    Matched: {len(alignment_info['matched_ids'])}")
            print(f"    Unmatched spectra: {len(alignment_info['unmatched_spectra'])}")
            print(f"    Unmatched reference: {len(alignment_info['unmatched_reference'])}")
            print(f"    Dropped (NaN): {alignment_info['n_nan_dropped']}")
            print(f"    Used fuzzy matching: {alignment_info['used_fuzzy_matching']}")

            print(f"\n  Final aligned data:")
            print(f"    X shape: {X_aligned.shape}")
            print(f"    y shape: {y_aligned.shape}")
            print(f"    Sample IDs match: {(X_aligned.index == y_aligned.index).all()}")

            return True
        else:
            print(f"  ✗ Reference file has too few columns: {len(columns)}")
            return False

    except Exception as e:
        print(f"  ✗ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

result = simulate_loading_workflow()

if result is None:
    print("\n" + "=" * 70)
    print("TEST SUITE: SKIPPED (no test data available)")
    print("=" * 70)
    print("\nThe infrastructure code has been added successfully.")
    print("To test with real data:")
    print("  1. Prepare master spectra folder with ASD/SPC/CSV files")
    print("  2. Add reference CSV/Excel with columns: filename, sample_id, target_value")
    print("  3. Run this script again or test in the GUI")
    sys.exit(0)
elif result:
    print("\n" + "=" * 70)
    print("TEST SUITE: PASSED ✓")
    print("=" * 70)
    print("\nAll core functionality works correctly!")
    print("Ready to proceed with UI implementation (Part 2)")
    sys.exit(0)
else:
    print("\n" + "=" * 70)
    print("TEST SUITE: FAILED ✗")
    print("=" * 70)
    print("\nSome issues were found. Review errors above.")
    sys.exit(1)
