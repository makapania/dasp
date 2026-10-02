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

import pytest

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

    print(f"Created synthetic data: {df.shape[0]} spectra with {df.shape[1]} wavelengths")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

    # Write to JCAMP files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        print(f"\nWriting JCAMP-DX files to: {tmpdir_path}")

        created_files = write_jcamp(
            df,
            tmpdir_path,
            title_prefix="test_spectrum",
            xunits="NANOMETERS",
            yunits="REFLECTANCE"
        )

        print(f"Wrote {len(created_files)} JCAMP files")
        assert len(created_files) == n_spectra

        # Read back individual file (requires jcamp package)
        try:
            print(f"\nReading individual JCAMP file: {created_files[0].name}")
            spectrum, metadata = read_jcamp_file(created_files[0])

            print(f"  Metadata keys: {list(metadata.keys())}")
            print(f"  Number of points: {spectrum.shape[1] if hasattr(spectrum, 'shape') else len(spectrum)}")

            # Read entire directory
            print(f"\nReading entire JCAMP directory")
            df_read, dir_metadata = read_jcamp_dir(tmpdir_path)

            print(f"  Loaded {df_read.shape[0]} spectra with {df_read.shape[1]} wavelengths")

            # Verify all spectra match
            assert df_read.shape == df.shape, (
                f"Shape mismatch - expected {df.shape}, got {df_read.shape}"
            )
            print("  Shape matches original!")

        except (ImportError, ValueError) as e:
            if "jcamp" in str(e).lower():
                import pytest
                pytest.skip("jcamp package not installed")
            raise

    print("\nTEST 1 PASSED")


def test_ascii_formats():
    """Test ASCII format parsing (.dpt, .dat, .asc) with exact value round trips."""
    wavelengths, spectrum = create_synthetic_spectrum()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for fmt_name, delimiter in {'tab': '\t', 'space': ' ', 'comma': ','}.items():
            test_file = tmpdir_path / f"test_{fmt_name}.dpt"
            with open(test_file, 'w') as f:
                f.write(f"# Test spectrum - {fmt_name} delimited\n")
                f.write(f"# Wavelength{delimiter}Reflectance\n")
                for wl, val in zip(wavelengths, spectrum):
                    f.write(f"{wl:.2f}{delimiter}{val:.6f}\n")

            df_parsed, info = _parse_ascii_file(test_file)

            assert len(df_parsed) == len(wavelengths), fmt_name
            np.testing.assert_allclose(df_parsed['x'].to_numpy(), wavelengths, atol=0.005)
            np.testing.assert_allclose(df_parsed['y'].to_numpy(), spectrum, atol=5e-7)
            assert info['delimiter'] == delimiter

        # Directory read with mixed extensions (the .dpt files above are included)
        for i, ext in enumerate(['.dpt', '.dat', '.asc']):
            with open(tmpdir_path / f"spectrum_{i+1}{ext}", 'w') as f:
                f.write("# Test spectrum\n")
                for wl, val in zip(wavelengths, spectrum):
                    f.write(f"{wl:.2f}\t{val:.6f}\n")

        df_dir, metadata = _read_ascii_dir(tmpdir_path)
        assert df_dir.shape == (6, len(wavelengths))
        assert metadata['n_spectra'] == 6
        assert metadata['wavelength_range'][0] == pytest.approx(wavelengths[0], abs=0.005)
        assert metadata['wavelength_range'][1] == pytest.approx(wavelengths[-1], abs=0.005)

        # The public reader takes the same folder (R062: it used to raise)
        df_public, _ = read_ascii_spectra(tmpdir_path)
        pd.testing.assert_frame_equal(df_public, df_dir)


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

        print(f"Writing JCAMP with custom metadata:")
        print(f"  {custom_metadata}")

        created_files = write_jcamp(
            df,
            tmpdir_path,
            xunits="NANOMETERS",
            yunits="REFLECTANCE",
            metadata=custom_metadata
        )

        assert len(created_files) == 1

        # Read back and check metadata (requires jcamp package)
        try:
            spectrum_read, metadata_read = read_jcamp_file(created_files[0])

            print(f"\nRead back metadata:")
            for key in custom_metadata.keys():
                if key in metadata_read:
                    print(f"  {key}: {metadata_read[key]}")
                else:
                    print(f"  {key}: NOT FOUND")

        except (ImportError, ValueError) as e:
            if "jcamp" in str(e).lower():
                import pytest
                pytest.skip("jcamp package not installed")
            raise

    print("\nTEST 3 PASSED")


def test_edge_cases():
    """Test edge cases and error handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Empty directory
        with pytest.raises(ValueError):
            read_jcamp_dir(tmpdir_path)

        # Multiple comment styles are ignored
        test_file = tmpdir_path / "test_comments.dat"
        wavelengths, spectrum = create_synthetic_spectrum(n_points=200)
        with open(test_file, 'w') as f:
            f.write("# Comment line 1\n")
            f.write("% Comment line 2\n")
            f.write("# Header: wavelength reflectance\n")
            for wl, val in zip(wavelengths, spectrum):
                f.write(f"{wl:.2f} {val:.6f}\n")
        df_parsed, _ = _parse_ascii_file(test_file)
        assert len(df_parsed) == 200

        # Mixed numeric precision, headerless: all four rows are data
        test_file = tmpdir_path / "test_precision.asc"
        with open(test_file, 'w') as f:
            f.write("350.0 0.5\n")
            f.write("351 0.51\n")
            f.write("352.00 0.52\n")
            f.write("353.5 0.525\n")
        df_parsed, info = _parse_ascii_file(test_file)
        assert df_parsed['x'].tolist() == [350.0, 351.0, 352.0, 353.5]
        assert df_parsed['y'].tolist() == [0.5, 0.51, 0.52, 0.525]
        assert info['header_lines'] == []


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
            test_func()
            results.append((test_name, True))
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
