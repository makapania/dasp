"""Test ASD ASCII file reading."""

import pytest
import numpy as np
from pathlib import Path

from spectral_predict.io import read_asd_dir, _read_single_asd_ascii


def test_read_single_asd_ascii_sig(tmp_path):
    """Test reading a single ASCII .sig file."""
    sig_file = tmp_path / "test.sig"

    # Create synthetic ASCII signature file
    # Format: wavelength reflectance
    wavelengths = np.linspace(400, 2400, 2001)
    reflectances = np.random.rand(2001) * 0.5 + 0.2  # 0.2-0.7 range

    with open(sig_file, "w") as f:
        f.write("# ASD signature file\n")
        f.write("# wavelength reflectance\n")
        for wl, ref in zip(wavelengths, reflectances):
            f.write(f"{wl:.2f} {ref:.6f}\n")

    # Read it
    spectrum = _read_single_asd_ascii(sig_file, reader_mode="auto")

    assert len(spectrum) == 2001
    assert spectrum.index[0] == 400.0
    assert spectrum.index[-1] == 2400.0
    assert all(spectrum >= 0.2)
    assert all(spectrum <= 0.7)


def test_read_single_asd_ascii_multicolumn(tmp_path):
    """Test reading ASCII file with multiple columns (picks last)."""
    sig_file = tmp_path / "test.sig"

    # Create file with wavelength, DN, reflectance columns
    wavelengths = np.linspace(400, 2400, 2001)
    dn_values = np.random.randint(1000, 5000, 2001)
    reflectances = np.random.rand(2001) * 0.5 + 0.2

    with open(sig_file, "w") as f:
        f.write("Wavelength DN Reflectance\n")
        for wl, dn, ref in zip(wavelengths, dn_values, reflectances):
            f.write(f"{wl:.2f} {dn} {ref:.6f}\n")

    # Read it - should pick last column (reflectance)
    spectrum = _read_single_asd_ascii(sig_file, reader_mode="auto")

    assert len(spectrum) == 2001
    # Values should match reflectances (last column), not DN values
    assert all(spectrum >= 0.2)
    assert all(spectrum <= 0.7)


def test_read_asd_dir_multiple_files(tmp_path):
    """Test reading multiple ASCII ASD files from directory."""
    # Create 3 .sig files
    for i in range(3):
        sig_file = tmp_path / f"sample_{i:03d}.sig"

        wavelengths = np.linspace(400, 2400, 2001)
        reflectances = np.random.rand(2001) * 0.5 + 0.2

        with open(sig_file, "w") as f:
            for wl, ref in zip(wavelengths, reflectances):
                f.write(f"{wl:.2f} {ref:.6f}\n")

    # Read directory
    df = read_asd_dir(tmp_path, reader_mode="auto")

    assert df.shape == (3, 2001)
    assert list(df.index) == ["sample_000", "sample_001", "sample_002"]
    assert df.columns[0] == 400.0
    assert df.columns[-1] == 2400.0


def test_read_asd_dir_mixed_sig_asd(tmp_path):
    """Test reading mixed .sig and .asd files."""
    # Create 2 .sig files
    for i in range(2):
        sig_file = tmp_path / f"sample_{i}.sig"
        wavelengths = np.linspace(400, 2400, 2001)
        reflectances = np.random.rand(2001)

        with open(sig_file, "w") as f:
            for wl, ref in zip(wavelengths, reflectances):
                f.write(f"{wl:.2f} {ref:.6f}\n")

    # Create 1 .asd file
    asd_file = tmp_path / "sample_2.asd"
    wavelengths = np.linspace(400, 2400, 2001)
    reflectances = np.random.rand(2001)

    with open(asd_file, "w") as f:
        for wl, ref in zip(wavelengths, reflectances):
            f.write(f"{wl:.2f} {ref:.6f}\n")

    # Read directory
    df = read_asd_dir(tmp_path)

    assert df.shape[0] == 3  # 3 samples
    assert "sample_0" in df.index
    assert "sample_1" in df.index
    assert "sample_2" in df.index


def test_read_asd_dir_no_files(tmp_path):
    """Test error when no ASD files found."""
    with pytest.raises(ValueError, match="No .sig or .asd files found"):
        read_asd_dir(tmp_path)


def test_read_asd_dir_not_directory(tmp_path):
    """Test error when path is not a directory."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError, match="Not a directory"):
        read_asd_dir(file_path)


def test_read_asd_dir_nonexistent(tmp_path):
    """Test error when directory doesn't exist."""
    fake_dir = tmp_path / "nonexistent"

    with pytest.raises(ValueError, match="Directory not found"):
        read_asd_dir(fake_dir)


def test_read_single_asd_with_header_lines(tmp_path):
    """Test reading file with non-numeric header lines."""
    sig_file = tmp_path / "test.sig"

    wavelengths = np.linspace(400, 2400, 2001)
    reflectances = np.random.rand(2001)

    with open(sig_file, "w") as f:
        f.write("ASD Field Spec Pro\n")
        f.write("Date: 2024-01-15\n")
        f.write("Integration time: 17ms\n")
        f.write("===DATA===\n")
        for wl, ref in zip(wavelengths, reflectances):
            f.write(f"{wl:.2f} {ref:.6f}\n")

    # Should skip header lines and read numeric data
    spectrum = _read_single_asd_ascii(sig_file, reader_mode="auto")

    assert len(spectrum) == 2001
    assert spectrum.index[0] == 400.0


def test_read_asd_ascii_wavelength_sorting(tmp_path):
    """Test that wavelengths are sorted correctly."""
    sig_file = tmp_path / "test.sig"

    # Write wavelengths in random order
    wavelengths = np.linspace(400, 2400, 500)
    reflectances = np.random.rand(500)

    # Shuffle
    indices = np.random.permutation(len(wavelengths))

    with open(sig_file, "w") as f:
        for idx in indices:
            f.write(f"{wavelengths[idx]:.2f} {reflectances[idx]:.6f}\n")

    # Read - should be sorted
    spectrum = _read_single_asd_ascii(sig_file, reader_mode="auto")

    # Check sorted
    assert spectrum.index[0] < spectrum.index[-1]
    assert all(spectrum.index[i] < spectrum.index[i + 1] for i in range(len(spectrum) - 1))


def test_binary_asd_detection(tmp_path):
    """Test that binary ASD files are detected and raise appropriate error."""
    # Create a fake binary ASD file
    binary_file = tmp_path / "binary.asd"
    with open(binary_file, "wb") as f:
        # Write some binary data (not UTF-8)
        f.write(b"\x00\x01\x02\x03\x04\x05ASD\x00\x00")
        f.write(b"\xff" * 100)

    # Should detect as binary and raise error
    with pytest.raises((ValueError, NotImplementedError)):
        read_asd_dir(tmp_path, reader_mode="auto")
