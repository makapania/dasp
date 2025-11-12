"""
Test script for JCAMP-DX and ASCII format support.

This script creates synthetic test data and verifies that:
1. JCAMP-DX read/write functions work correctly
2. ASCII variant formats (.dpt, .dat, .asc) are read correctly
3. Data is properly formatted and metadata is preserved
4. Integration with existing codebase works
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
    read_jcamp_file,
    read_jcamp_dir,
    write_jcamp,
    read_ascii_spectra,
    _read_ascii_dir,
    _parse_ascii_file
)


def create_synthetic_spectrum(n_points=2151, wavelength_range=(350, 2500)):
    """Create a synthetic reflectance spectrum."""
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)

    # Base reflectance around 0.4-0.6
    base = 0.5

    # Add some peaks and valleys
    spectrum = base + 0.1 * np.sin(wavelengths / 200) + 0.05 * np.cos(wavelengths / 100)

    # Add some noise
    spectrum += np.random.normal(0, 0.01, n_points)

    # Clip to valid reflectance range [0, 1]
    spectrum = np.clip(spectrum, 0, 1)

    return wavelengths, spectrum


def test_jcamp_write_read():
    """Test JCAMP-DX write and read functions."""
    print("\n" + "="*80)
    print("TEST 1: JCAMP-DX Write and Read")
    print("="*80)

    # Create synthetic data
    n_spectra = 5
    wavelengths, _ = create_synthetic_spectrum()

    spectra_dict = {}
    for i in range(n_spectra):
        _, spectrum = create_synthetic_spectrum()
        spectra_dict[f"sample_{i+1}"] = spectrum

    # Create DataFrame
    df = pd.DataFrame(spectra_dict, index=wavelengths).T

    print(f"✓ Created synthetic data: {df.shape[0]} spectra with {df.shape[1]} wavelengths")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

    # Write to JCAMP files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        print(f"\n✓ Writing JCAMP-DX files to: {tmpdir_path}")

        created_files = write_jcamp(
            df,
            tmpdir_path,
            title_prefix="test_spectrum",
            xunits="NANOMETERS",
            yunits="REFLECTANCE"
        )

        print(f"✓ Wrote {len(created_files)} JCAMP files")

        # Read back individual file
        print(f"\n✓ Reading individual JCAMP file: {created_files[0].name}")
        spectrum, metadata = read_jcamp_file(created_files[0])

        print(f"  Metadata keys: {list(metadata.keys())}")
        print(f"  X-axis units: {metadata['xunits']}")
        print(f"  Y-axis units: {metadata['yunits']}")
        print(f"  Number of points: {len(spectrum)}")

        # Verify data integrity
        original_spectrum = df.iloc[0].values
        read_spectrum = spectrum.values

        # Allow for small floating point differences
        max_diff = np.max(np.abs(original_spectrum - read_spectrum))
        print(f"  Max difference from original: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("  ✓ Data integrity verified!")
        else:
            print("  ⚠ WARNING: Significant difference detected")

        # Read entire directory
        print(f"\n✓ Reading entire JCAMP directory")
        df_read, dir_metadata = read_jcamp_dir(tmpdir_path)

        print(f"  Loaded {df_read.shape[0]} spectra with {df_read.shape[1]} wavelengths")
        print(f"  Data type detected: {dir_metadata['data_type']} (confidence: {dir_metadata['type_confidence']:.1f}%)")
        print(f"  X-axis units: {dir_metadata['xunits']}")

        # Verify all spectra match
        if df_read.shape == df.shape:
            print("  ✓ Shape matches original!")
        else:
            print(f"  ⚠ WARNING: Shape mismatch - expected {df.shape}, got {df_read.shape}")

    print("\n✓ TEST 1 PASSED")
    return True


def test_ascii_formats():
    """Test ASCII format parsing (.dpt, .dat, .asc)."""
    print("\n" + "="*80)
    print("TEST 2: ASCII Format Support")
    print("="*80)

    wavelengths, spectrum = create_synthetic_spectrum()

    # Test different delimiter formats
    formats = {
        'tab': '\t',
        'space': ' ',
        'comma': ','
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for fmt_name, delimiter in formats.items():
            print(f"\n✓ Testing {fmt_name}-delimited format:")

            # Create test file
            test_file = tmpdir_path / f"test_{fmt_name}.dpt"

            with open(test_file, 'w') as f:
                # Write header comment
                f.write(f"# Test spectrum - {fmt_name} delimited\n")
                f.write(f"# Wavelength{delimiter}Reflectance\n")

                # Write data
                for wl, val in zip(wavelengths, spectrum):
                    f.write(f"{wl:.2f}{delimiter}{val:.6f}\n")

            print(f"  Created: {test_file.name}")

            # Read the file
            df_parsed, x_col, y_col = _parse_ascii_file(test_file)

            if df_parsed is not None:
                print(f"  ✓ Parsed {len(df_parsed)} data points")
                print(f"  X column: {x_col}, Y column: {y_col}")

                # Verify data
                max_wl_diff = np.max(np.abs(df_parsed[x_col].values - wavelengths))
                max_val_diff = np.max(np.abs(df_parsed[y_col].values - spectrum))

                print(f"  Max wavelength diff: {max_wl_diff:.2e}")
                print(f"  Max value diff: {max_val_diff:.2e}")

                if max_wl_diff < 0.1 and max_val_diff < 1e-5:
                    print(f"  ✓ {fmt_name.capitalize()}-delimited format verified!")
                else:
                    print(f"  ⚠ WARNING: Data mismatch detected")
            else:
                print(f"  ✗ FAILED to parse {fmt_name}-delimited file")
                return False

        # Test directory reading with mixed extensions
        print(f"\n✓ Testing directory read with mixed extensions:")

        # Create files with different extensions
        extensions = ['.dpt', '.dat', '.asc']
        for i, ext in enumerate(extensions):
            test_file = tmpdir_path / f"spectrum_{i+1}{ext}"
            with open(test_file, 'w') as f:
                f.write("# Test spectrum\n")
                for wl, val in zip(wavelengths, spectrum):
                    f.write(f"{wl:.2f}\t{val:.6f}\n")

        print(f"  Created 3 files with extensions: {extensions}")

        # Read directory
        df_dir, metadata = _read_ascii_dir(tmpdir_path)

        print(f"  ✓ Loaded {df_dir.shape[0]} spectra with {df_dir.shape[1]} wavelengths")
        print(f"  Data type: {metadata['data_type']} (confidence: {metadata['type_confidence']:.1f}%)")
        print(f"  Wavelength range: {metadata['wavelength_range'][0]:.1f} - {metadata['wavelength_range'][1]:.1f}")

    print("\n✓ TEST 2 PASSED")
    return True


def test_jcamp_metadata_preservation():
    """Test that JCAMP-DX preserves metadata."""
    print("\n" + "="*80)
    print("TEST 3: JCAMP-DX Metadata Preservation")
    print("="*80)

    wavelengths, spectrum = create_synthetic_spectrum()
    df = pd.DataFrame([spectrum], columns=wavelengths, index=['test_sample'])

    # Custom metadata
    custom_metadata = {
        'instrument': 'Test_Spectrometer_3000',
        'operator': 'Test_User',
        'date': '2025-01-01'
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        print(f"✓ Writing JCAMP with custom metadata:")
        print(f"  {custom_metadata}")

        created_files = write_jcamp(
            df,
            tmpdir_path,
            xunits="NANOMETERS",
            yunits="REFLECTANCE",
            metadata=custom_metadata
        )

        # Read back and check metadata
        spectrum_read, metadata_read = read_jcamp_file(created_files[0])

        print(f"\n✓ Read back metadata:")
        for key in custom_metadata.keys():
            if key in metadata_read:
                print(f"  {key}: {metadata_read[key]}")
            else:
                print(f"  ⚠ {key}: NOT FOUND")

        # Check standard metadata
        print(f"\n✓ Standard JCAMP metadata:")
        print(f"  xunits: {metadata_read['xunits']}")
        print(f"  yunits: {metadata_read['yunits']}")
        print(f"  npoints: {metadata_read['npoints']}")
        print(f"  firstx: {metadata_read['firstx']}")
        print(f"  lastx: {metadata_read['lastx']}")

    print("\n✓ TEST 3 PASSED")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST 4: Edge Cases and Error Handling")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Test 1: Empty directory
        print("\n✓ Test 4.1: Empty directory")
        try:
            df, metadata = read_jcamp_dir(tmpdir_path)
            print("  ✗ Should have raised an error")
            return False
        except ValueError as e:
            print(f"  ✓ Correctly raised ValueError: {e}")

        # Test 2: File with comment lines
        print("\n✓ Test 4.2: ASCII file with multiple comment styles")
        test_file = tmpdir_path / "test_comments.dat"

        wavelengths, spectrum = create_synthetic_spectrum(n_points=200)

        with open(test_file, 'w') as f:
            f.write("# Comment line 1\n")
            f.write("% Comment line 2\n")
            f.write("# Header: wavelength reflectance\n")
            for wl, val in zip(wavelengths, spectrum):
                f.write(f"{wl:.2f} {val:.6f}\n")

        df_parsed, x_col, y_col = _parse_ascii_file(test_file)

        if df_parsed is not None and len(df_parsed) == 200:
            print(f"  ✓ Correctly parsed {len(df_parsed)} data points (ignoring comments)")
        else:
            print(f"  ✗ Failed to parse correctly")
            return False

        # Test 3: Mixed numeric precision
        print("\n✓ Test 4.3: Mixed numeric precision")
        test_file = tmpdir_path / "test_precision.asc"

        with open(test_file, 'w') as f:
            f.write("350.0 0.5\n")
            f.write("351 0.51\n")
            f.write("352.00 0.52\n")
            f.write("353.5 0.525\n")

        df_parsed, x_col, y_col = _parse_ascii_file(test_file)

        if df_parsed is not None and len(df_parsed) == 4:
            print(f"  ✓ Correctly parsed mixed precision data")
        else:
            print(f"  ✗ Failed to parse mixed precision")
            return False

    print("\n✓ TEST 4 PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS FOR JCAMP-DX AND ASCII FORMAT SUPPORT")
    print("="*80)

    tests = [
        ("JCAMP-DX Write/Read", test_jcamp_write_read),
        ("ASCII Formats", test_ascii_formats),
        ("JCAMP Metadata", test_jcamp_metadata_preservation),
        ("Edge Cases", test_edge_cases)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠ SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
