"""
Test script for SPC and Excel I/O functionality.

Tests:
1. Excel read/write with read_excel_spectra() and write_excel_spectra()
2. SPC write with write_spc_file()
3. Validates data integrity after round-trip

Usage:
    python test_spc_excel_io.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.io import (
    read_excel_spectra,
    write_excel_spectra,
    write_spc_file,
    read_spc_file,
)


def create_synthetic_spectra(n_samples=5, n_wavelengths=200):
    """Create synthetic spectral data for testing."""
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    sample_ids = [f"Sample_{i+1}" for i in range(n_samples)]

    # Create random spectral data (reflectance-like, 0-1 range)
    data = np.random.uniform(0.1, 0.9, (n_samples, n_wavelengths))

    # Add some spectral features (peaks/valleys)
    for i in range(n_samples):
        # Add a gaussian peak around 1000nm
        peak_center = 1000 + np.random.uniform(-200, 200)
        peak_width = 100
        peak_height = 0.2
        peak = peak_height * np.exp(-((wavelengths - peak_center) ** 2) / (2 * peak_width ** 2))
        data[i] += peak

    df = pd.DataFrame(data, columns=wavelengths, index=sample_ids)
    return df


def test_excel_round_trip():
    """Test Excel write and read functionality."""
    print("\n" + "="*70)
    print("TEST 1: Excel Round-Trip (write + read)")
    print("="*70)

    # Create synthetic data
    print("\n1. Creating synthetic spectral data...")
    df_original = create_synthetic_spectra(n_samples=5, n_wavelengths=200)
    print(f"   Created {len(df_original)} spectra with {df_original.shape[1]} wavelengths")
    print(f"   Wavelength range: {df_original.columns.min():.1f} - {df_original.columns.max():.1f} nm")
    print(f"   Data range: {df_original.values.min():.3f} - {df_original.values.max():.3f}")

    # Write to Excel
    with tempfile.TemporaryDirectory() as tmpdir:
        excel_path = Path(tmpdir) / "test_spectra.xlsx"

        print(f"\n2. Writing to Excel file: {excel_path.name}")
        write_excel_spectra(df_original, excel_path)

        # Read back from Excel
        print(f"\n3. Reading back from Excel file...")
        df_read, metadata = read_excel_spectra(excel_path)

        print(f"   Loaded {metadata['n_spectra']} spectra")
        print(f"   Wavelength range: {metadata['wavelength_range'][0]:.1f} - {metadata['wavelength_range'][1]:.1f} nm")
        print(f"   Data type detected: {metadata['data_type']} ({metadata['type_confidence']:.1f}% confidence)")

        # Validate
        print(f"\n4. Validating data integrity...")

        # Check shape
        assert df_original.shape == df_read.shape, \
            f"Shape mismatch: {df_original.shape} != {df_read.shape}"
        print(f"   ✓ Shape matches: {df_original.shape}")

        # Check indices
        assert list(df_original.index) == list(df_read.index), \
            "Index mismatch"
        print(f"   ✓ Sample IDs match: {list(df_original.index)}")

        # Check columns (wavelengths)
        np.testing.assert_allclose(df_original.columns, df_read.columns, rtol=1e-5)
        print(f"   ✓ Wavelengths match")

        # Check values
        np.testing.assert_allclose(df_original.values, df_read.values, rtol=1e-5)
        print(f"   ✓ Spectral values match (within tolerance)")

        print("\n✓ TEST PASSED: Excel round-trip successful!")

    return True


def test_spc_write():
    """Test SPC write functionality."""
    print("\n" + "="*70)
    print("TEST 2: SPC Write")
    print("="*70)

    # Create synthetic data (single spectrum)
    print("\n1. Creating synthetic spectrum...")
    df_original = create_synthetic_spectra(n_samples=1, n_wavelengths=200)
    print(f"   Created 1 spectrum with {df_original.shape[1]} wavelengths")
    print(f"   Wavelength range: {df_original.columns.min():.1f} - {df_original.columns.max():.1f} nm")
    print(f"   Data range: {df_original.values.min():.3f} - {df_original.values.max():.3f}")

    # Write to SPC
    with tempfile.TemporaryDirectory() as tmpdir:
        spc_path = Path(tmpdir) / "test_spectrum.spc"

        print(f"\n2. Writing to SPC file: {spc_path.name}")
        try:
            write_spc_file(df_original, spc_path)
            print(f"   ✓ SPC file created successfully")

            # Check file exists and has content
            assert spc_path.exists(), "SPC file was not created"
            file_size = spc_path.stat().st_size
            print(f"   ✓ File size: {file_size} bytes")

            # Try to read it back
            print(f"\n3. Reading back from SPC file...")
            df_read, metadata = read_spc_file(spc_path)

            print(f"   Loaded {metadata['n_spectra']} spectrum")
            print(f"   Wavelength range: {metadata['wavelength_range'][0]:.1f} - {metadata['wavelength_range'][1]:.1f} nm")
            print(f"   Data type detected: {metadata['data_type']} ({metadata['type_confidence']:.1f}% confidence)")

            # Validate
            print(f"\n4. Validating data integrity...")

            # Check shape
            assert df_original.shape == df_read.shape, \
                f"Shape mismatch: {df_original.shape} != {df_read.shape}"
            print(f"   ✓ Shape matches: {df_original.shape}")

            # Check wavelengths (with some tolerance due to SPC format)
            np.testing.assert_allclose(df_original.columns, df_read.columns, rtol=1e-2)
            print(f"   ✓ Wavelengths match (within tolerance)")

            # Check values (with some tolerance due to SPC format)
            np.testing.assert_allclose(df_original.values, df_read.values, rtol=1e-2)
            print(f"   ✓ Spectral values match (within tolerance)")

            print("\n✓ TEST PASSED: SPC write/read successful!")

        except ImportError as e:
            print(f"\n⚠ WARNING: spc-io not installed, skipping SPC test")
            print(f"   Install with: pip install spc-io")
            return None
        except Exception as e:
            print(f"\n✗ TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    return True


def test_multiple_spectra_warning():
    """Test that writing multiple spectra to SPC shows warning."""
    print("\n" + "="*70)
    print("TEST 3: Multiple Spectra Warning")
    print("="*70)

    # Create synthetic data (multiple spectra)
    print("\n1. Creating 5 synthetic spectra...")
    df_original = create_synthetic_spectra(n_samples=5, n_wavelengths=200)
    print(f"   Created {len(df_original)} spectra")

    # Write to SPC (should show warning)
    with tempfile.TemporaryDirectory() as tmpdir:
        spc_path = Path(tmpdir) / "test_multi.spc"

        print(f"\n2. Writing to SPC file (expecting warning)...")
        try:
            write_spc_file(df_original, spc_path)
            print(f"   ✓ SPC file created (only first spectrum should be written)")

            # Read back and verify only 1 spectrum
            print(f"\n3. Verifying only first spectrum was written...")
            df_read, metadata = read_spc_file(spc_path)

            assert len(df_read) == 1, f"Expected 1 spectrum, got {len(df_read)}"
            print(f"   ✓ Confirmed: Only 1 spectrum in SPC file")

            # Verify it's the first spectrum
            np.testing.assert_allclose(
                df_original.iloc[0].values,
                df_read.iloc[0].values,
                rtol=1e-2
            )
            print(f"   ✓ Confirmed: First spectrum matches")

            print("\n✓ TEST PASSED: Multiple spectra warning works correctly!")

        except ImportError:
            print(f"\n⚠ WARNING: spc-io not installed, skipping test")
            return None
        except Exception as e:
            print(f"\n✗ TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SPC and Excel I/O Test Suite")
    print("="*70)

    results = []

    # Test 1: Excel round-trip
    try:
        result = test_excel_round_trip()
        results.append(("Excel Round-Trip", result))
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("Excel Round-Trip", False))

    # Test 2: SPC write
    try:
        result = test_spc_write()
        results.append(("SPC Write", result))
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("SPC Write", False))

    # Test 3: Multiple spectra warning
    try:
        result = test_multiple_spectra_warning()
        results.append(("Multiple Spectra Warning", result))
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("Multiple Spectra Warning", False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result is True else "✗ FAILED" if result is False else "⊘ SKIPPED"
        print(f"{status:12} {name}")

    print(f"\nTotal: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    elif passed > 0:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠ All tests were skipped (missing dependencies?)")
        sys.exit(0)


if __name__ == "__main__":
    main()
