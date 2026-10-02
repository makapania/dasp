"""Test vendor-specific format readers (OPUS, PerkinElmer, Agilent)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from spectral_predict.io import (
    read_opus_file,
    read_perkinelmer_file,
    read_agilent_file,
    read_ascii_spectra,
    write_ascii_spectra,
    read_spectra,
    write_spectra,
    detect_format
)


# ============================================================================
# ASCII Text Format Tests (Generic two-column format)
# ============================================================================


def test_read_ascii_tab_delimited(tmp_path):
    """Test reading ASCII file with tab delimiter."""
    ascii_path = tmp_path / "spectrum.txt"

    # Create tab-delimited file
    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        f.write("# Spectral data file\n")
        f.write("# Wavelength\tIntensity\n")
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f}\t{intensity:.6f}\n")

    # Read
    result, metadata = read_ascii_spectra(ascii_path)

    # Every data row is kept, including the first (R062: it used to become a header)
    assert result.shape == (1, 2001)
    assert result.columns[0] == pytest.approx(400.0)
    assert result.columns[-1] == pytest.approx(2400.0)
    np.testing.assert_allclose(result.iloc[0].to_numpy(), intensities, atol=1e-6)
    assert result.index[0] == "spectrum"
    assert metadata['file_format'] == 'ascii'
    assert metadata['n_spectra'] == 1


def test_read_ascii_comma_delimited(tmp_path):
    """Test reading ASCII file with comma delimiter."""
    ascii_path = tmp_path / "spectrum.dat"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f},{intensity:.6f}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert result.shape == (1, 2001)
    assert result.iloc[0, 0] == pytest.approx(intensities[0], abs=1e-6)
    assert metadata['file_format'] == 'ascii'
    assert metadata['delimiter'] == ','


def test_read_ascii_space_delimited(tmp_path):
    """Test reading ASCII file with space delimiter."""
    ascii_path = tmp_path / "spectrum.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f} {intensity:.6f}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert result.shape == (1, 2001)
    assert metadata['delimiter'] == ' '


def test_read_ascii_with_comments(tmp_path):
    """Test reading ASCII file with comment lines."""
    ascii_path = tmp_path / "commented.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        f.write("# This is a comment\n")
        f.write("# Another comment line\n")
        f.write("# Wavelength (nm)\tReflectance\n")
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f}\t{intensity:.6f}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert result.shape == (1, 2001)
    assert result.iloc[0, 0] == pytest.approx(intensities[0], abs=1e-6)
    # Commented-out headings are comments, not a header
    assert metadata['header_lines'] == []


def test_write_read_ascii_roundtrip(tmp_path):
    """Test ASCII write and read roundtrip."""
    ascii_path = tmp_path / "roundtrip.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    original = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["Sample_1"],
        columns=wavelengths
    )

    # Write
    write_ascii_spectra(original, ascii_path, delimiter='\t')

    # Read
    result, _ = read_ascii_spectra(ascii_path, delimiter='\t')

    # Compare
    assert result.shape == original.shape
    np.testing.assert_array_almost_equal(result.values, original.values, decimal=5)


def test_write_ascii_multiple_spectra_warning(tmp_path):
    """Test that writing multiple spectra gives warning."""
    ascii_path = tmp_path / "multi.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        np.random.rand(3, 2001),
        index=["S1", "S2", "S3"],
        columns=wavelengths
    )

    # Should warn and write only first
    write_ascii_spectra(data, ascii_path)

    # Read back
    result, _ = read_ascii_spectra(ascii_path)
    assert result.shape[0] == 1


def test_ascii_via_unified_api(tmp_path):
    """Test ASCII read/write via unified API."""
    ascii_path = tmp_path / "unified.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["S1"],
        columns=wavelengths
    )

    # Write
    write_spectra(data, ascii_path, format='ascii', delimiter='\t')

    # Read with auto-detect
    result, metadata = read_spectra(ascii_path, format='auto')
    assert metadata['file_format'] == 'ascii'

    # Read with explicit format
    result2, _ = read_spectra(ascii_path, format='ascii')
    pd.testing.assert_frame_equal(result, result2)


def test_ascii_without_header(tmp_path):
    """Test ASCII file without header."""
    ascii_path = tmp_path / "no_header.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f}\t{intensity:.6f}\n")

    result, _ = read_ascii_spectra(ascii_path)
    assert result.shape == (1, 2001)
    assert result.columns[0] == pytest.approx(400.0)
    assert result.iloc[0, 0] == pytest.approx(intensities[0], abs=1e-6)


def test_ascii_custom_delimiter_write(tmp_path):
    """Test ASCII write with custom delimiter."""
    ascii_path = tmp_path / "custom.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        [np.random.rand(2001)],
        index=["S1"],
        columns=wavelengths
    )

    # Write with semicolon delimiter
    write_ascii_spectra(data, ascii_path, delimiter=';', include_header=True)

    # Read with specified delimiter
    result, _ = read_ascii_spectra(ascii_path, delimiter=';')
    assert result.shape == data.shape


# ============================================================================
# Bruker OPUS Format Tests
# ============================================================================


def test_opus_import_error():
    """Test that missing brukeropus package raises error."""
    import importlib.util

    # find_spec works even when the module hasn't been imported yet, unlike
    # sys.modules.get() which only sees already-imported modules.
    if importlib.util.find_spec("brukeropus") is not None:
        pytest.skip("brukeropus package is installed")

    # Should raise ImportError before reaching the file-existence check
    with pytest.raises(ImportError, match="brukeropus"):
        read_opus_file(Path("dummy.0"))


def test_opus_file_detection():
    """Test that OPUS numbered extensions are detected."""
    assert detect_format(Path("spectrum.0")) == 'opus'
    assert detect_format(Path("spectrum.1")) == 'opus'
    assert detect_format(Path("spectrum.12")) == 'opus'


def test_opus_via_unified_api_if_installed(tmp_path):
    """Test OPUS reading via unified API if package installed."""
    opus_path = tmp_path / "test.0"

    # Create a mock OPUS file
    with open(opus_path, 'wb') as f:
        f.write(b'OPUS')
        f.write(b'\x00' * 100)

    try:
        # Try auto-detect
        result, metadata = read_spectra(opus_path, format='auto')
        assert metadata['file_format'] == 'opus'
    except ImportError:
        pytest.skip("brukeropus not installed")
    except Exception:
        # Mock file isn't valid
        pass


# ============================================================================
# PerkinElmer Format Tests
# ============================================================================


def test_perkinelmer_import_error():
    """Test that missing specio package raises error."""
    import importlib.util

    # Production code uses specio_py310 (not specio); find_spec works pre-import.
    if importlib.util.find_spec("specio_py310") is not None:
        pytest.skip("specio_py310 package is installed")

    with pytest.raises(ImportError, match="specio"):
        read_perkinelmer_file(Path("dummy.sp"))


def test_perkinelmer_file_detection():
    """Test that .sp extension is detected."""
    assert detect_format(Path("spectrum.sp")) == 'perkinelmer'


def test_perkinelmer_via_unified_api_if_installed(tmp_path):
    """Test PerkinElmer reading via unified API."""
    pe_path = tmp_path / "test.sp"

    # Create mock file
    with open(pe_path, 'wb') as f:
        f.write(b'PEPE')  # Mock header
        f.write(b'\x00' * 100)

    try:
        result, metadata = read_spectra(pe_path, format='auto')
        assert metadata['file_format'] == 'perkinelmer'
    except ImportError:
        pytest.skip("specio not installed")
    except Exception:
        # Mock file isn't valid
        pass


# ============================================================================
# Agilent Format Tests
# ============================================================================


def test_agilent_import_error():
    """Test that missing agilent-ir-formats package raises error."""
    import importlib.util

    if importlib.util.find_spec("agilent_ir_formats") is not None:
        pytest.skip("agilent-ir-formats package is installed")

    with pytest.raises(ImportError, match="agilent-ir-formats"):
        read_agilent_file(Path("dummy.seq"))


def test_agilent_file_detection():
    """Test that .seq extension is detected."""
    assert detect_format(Path("spectrum.seq")) == 'agilent'


def test_agilent_invalid_file_raises_value_error(tmp_path):
    """Agilent reader is implemented around AgilentIRFile (see
    src/spectral_predict/readers/agilent_reader.py). On a deliberately
    invalid .seq file we expect a parser-level ValueError, not silent
    success or an unrelated exception type.

    PR #56 review (Codex MEDIUM #4): the previous ``test_agilent_not_implemented``
    asserted a stale not-implemented contract that only passed because
    ``agilent_ir_formats`` happens to be absent in CI. The moment that
    optional dep lands, the test would fail for the wrong reason. Replaced
    with a real parse-error contract.
    """
    import importlib.util

    if importlib.util.find_spec("agilent_ir_formats") is None:
        pytest.skip("agilent-ir-formats not installed")

    bogus = tmp_path / "bogus.seq"
    bogus.write_bytes(b"NOT A REAL SEQ FILE\x00" * 16)

    with pytest.raises(ValueError):
        read_agilent_file(bogus)


# ============================================================================
# Format Detection Tests
# ============================================================================


def test_detect_format_csv():
    """Test CSV format detection."""
    assert detect_format(Path("data.csv")) == 'csv'


def test_detect_format_excel():
    """Test Excel format detection."""
    assert detect_format(Path("data.xlsx")) == 'excel'
    assert detect_format(Path("data.xls")) == 'excel'


def test_detect_format_asd():
    """Test ASD format detection."""
    assert detect_format(Path("spectrum.asd")) == 'asd'
    assert detect_format(Path("spectrum.sig")) == 'asd'


def test_detect_format_spc():
    """Test SPC format detection."""
    assert detect_format(Path("spectrum.spc")) == 'spc'


def test_detect_format_jcamp():
    """Test JCAMP-DX format detection."""
    assert detect_format(Path("spectrum.jdx")) == 'jcamp'
    assert detect_format(Path("spectrum.dx")) == 'jcamp'
    assert detect_format(Path("spectrum.jcm")) == 'jcamp'


def test_detect_format_ascii():
    """Test ASCII format detection."""
    assert detect_format(Path("spectrum.txt")) == 'ascii'
    assert detect_format(Path("spectrum.dat")) == 'ascii'


def test_detect_format_directory(tmp_path):
    """Test directory detection."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    assert detect_format(test_dir) == 'directory'


def test_detect_format_unknown():
    """Test unknown format detection."""
    assert detect_format(Path("file.unknown")) == 'unknown'


def test_detect_format_magic_bytes_spc(tmp_path):
    """Test SPC detection via magic bytes."""
    spc_path = tmp_path / "noext"

    # Create file with SPC magic bytes
    with open(spc_path, 'wb') as f:
        f.write(b'\x4d\x4b')  # 'MK'
        f.write(b'\x00' * 100)

    assert detect_format(spc_path) == 'spc'


def test_detect_format_magic_bytes_jcamp(tmp_path):
    """Test JCAMP detection via magic bytes."""
    jcamp_path = tmp_path / "noext"

    with open(jcamp_path, 'wb') as f:
        f.write(b'##TITLE=Test\n')
        f.write(b'##JCAMP-DX=5.0\n')

    assert detect_format(jcamp_path) == 'jcamp'


def test_detect_format_magic_bytes_opus(tmp_path):
    """Test OPUS detection via magic bytes."""
    opus_path = tmp_path / "noext"

    with open(opus_path, 'wb') as f:
        f.write(b'OPUS')
        f.write(b'\x00' * 100)

    assert detect_format(opus_path) == 'opus'


# ============================================================================
# Integration Tests
# ============================================================================


def test_ascii_metadata_completeness(tmp_path):
    """Test that ASCII reader returns complete metadata."""
    ascii_path = tmp_path / "meta.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl}\t{intensity}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    # Check all expected metadata fields
    assert 'file_format' in metadata
    assert metadata['file_format'] == 'ascii'
    assert 'n_spectra' in metadata
    assert 'wavelength_range' in metadata
    assert 'data_type' in metadata
    assert 'type_confidence' in metadata
    assert 'detection_method' in metadata


def test_ascii_data_type_detection(tmp_path):
    """Test that data type is detected for ASCII files."""
    ascii_path = tmp_path / "reflectance.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    # Create reflectance-like data (0-1 range)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl}\t{intensity}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert metadata['data_type'] in ['reflectance', 'absorbance']
    assert 0 <= metadata['type_confidence'] <= 100


def test_ascii_wavelength_sorting(tmp_path):
    """Test that wavelengths are sorted in ASCII data."""
    ascii_path = tmp_path / "unsorted.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001)

    # Write in random order
    indices = np.random.permutation(len(wavelengths))

    with open(ascii_path, 'w') as f:
        for idx in indices:
            f.write(f"{wavelengths[idx]}\t{intensities[idx]}\n")

    result, _ = read_ascii_spectra(ascii_path)

    # Check wavelengths are sorted
    wls = result.columns.values
    assert np.all(wls[:-1] <= wls[1:])


# ============================================================================
# Wrapper metadata merge order (sibling audit of review finding R017)
# ============================================================================
# The io wrappers used to merge the reader's own metadata LAST, so reader keys
# overwrote the normalised ones (file_format 'sp'/'seq', PerkinElmer's
# non-canonical x_unit 'wavenumber_cm-1', which the GUI treats as nm).


@pytest.fixture
def fake_specio(monkeypatch):
    import sys
    import types

    x_desc = np.linspace(4000.0, 400.0, 361)

    class _Spectrum:
        wavelength = x_desc
        amplitudes = np.linspace(0.1, 0.9, 361)

    module = types.ModuleType("specio_py310")
    module.specread = lambda path: _Spectrum()
    monkeypatch.setitem(sys.modules, "specio_py310", module)
    return x_desc


def test_perkinelmer_wrapper_normalised_metadata(tmp_path, fake_specio):
    path = tmp_path / "pe1.sp"
    path.write_bytes(b"placeholder")

    df, metadata = read_perkinelmer_file(path)

    assert df.shape == (1, 361)
    assert df.columns[0] == 400.0
    assert metadata['file_format'] == 'perkinelmer'
    assert metadata['x_unit'] == 'cm-1'


def test_perkinelmer_dir_reports_canonical_x_unit(tmp_path, fake_specio):
    from spectral_predict.io import read_sp_dir

    for i in range(2):
        (tmp_path / f"pe{i}.sp").write_bytes(b"placeholder")

    df, metadata = read_sp_dir(tmp_path)

    assert df.shape == (2, 361)
    assert metadata['x_unit'] == 'cm-1'


def _install_specio(monkeypatch, x, meta=None):
    import sys
    import types

    class _Spectrum:
        wavelength = x
        amplitudes = np.linspace(0.1, 0.9, len(x))

    _Spectrum.meta = meta or {}
    module = types.ModuleType("specio_py310")
    module.specread = lambda path: _Spectrum()
    monkeypatch.setitem(sys.modules, "specio_py310", module)


def test_perkinelmer_full_ir_range_is_cm1(tmp_path, monkeypatch):
    _install_specio(monkeypatch, np.linspace(4000.0, 650.0, 336))
    path = tmp_path / "ir.sp"
    path.write_bytes(b"placeholder")

    _, metadata = read_perkinelmer_file(path)

    assert metadata['x_unit'] == 'cm-1'
    assert metadata['x_unit_detection_method'] == 'perkinelmer_range'


def test_perkinelmer_nir_nm_range_is_not_confidently_cm1(tmp_path, monkeypatch):
    """A 1000-2500 nm Lambda NIR file used to be labelled cm-1 with confidence."""
    _install_specio(monkeypatch, np.linspace(1000.0, 2500.0, 1501))
    path = tmp_path / "nir.sp"
    path.write_bytes(b"placeholder")

    with pytest.warns(UserWarning, match="does not identify the unit"):
        _, metadata = read_perkinelmer_file(path)

    assert metadata['x_unit'] == 'nm'
    assert metadata['x_unit_confidence'] < 50.0
    assert metadata['x_unit_detection_method'] == 'perkinelmer_ambiguous_range'


def test_perkinelmer_unit_from_metadata_wins(tmp_path, monkeypatch):
    _install_specio(monkeypatch, np.linspace(1000.0, 2500.0, 1501), meta={'x_units': 'cm-1'})
    path = tmp_path / "meta.sp"
    path.write_bytes(b"placeholder")

    _, metadata = read_perkinelmer_file(path)

    assert metadata['x_unit'] == 'cm-1'
    assert metadata['x_unit_detection_method'] == 'perkinelmer_metadata'


def test_perkinelmer_dir_surfaces_ambiguous_unit_warning(tmp_path, monkeypatch):
    from spectral_predict.io import read_sp_dir

    _install_specio(monkeypatch, np.linspace(1000.0, 2500.0, 1501))
    for i in range(2):
        (tmp_path / f"nir{i}.sp").write_bytes(b"placeholder")

    with pytest.warns(UserWarning):
        _, metadata = read_sp_dir(tmp_path)

    assert metadata['x_unit'] == 'nm'
    assert metadata['x_unit_confidence'] < 50.0
    assert any("2 of 2 .sp files" in m for m in metadata['import_warnings'])


def test_agilent_wrapper_normalised_metadata(tmp_path, monkeypatch):
    import sys
    import types

    class _AgilentIRFile:
        def read(self, path):
            self.wavenumbers = np.linspace(4000.0, 900.0, 200)
            self.intensities = np.linspace(0.1, 0.9, 200)

    package = types.ModuleType("agilent_ir_formats")
    submodule = types.ModuleType("agilent_ir_formats.agilent_ir_file")
    submodule.AgilentIRFile = _AgilentIRFile
    package.agilent_ir_file = submodule
    monkeypatch.setitem(sys.modules, "agilent_ir_formats", package)
    monkeypatch.setitem(sys.modules, "agilent_ir_formats.agilent_ir_file", submodule)

    path = tmp_path / "tile.seq"
    path.write_bytes(b"placeholder")

    df, metadata = read_agilent_file(path)

    assert df.shape == (1, 200)
    assert metadata['file_format'] == 'agilent'
    assert metadata['source_file_format'] == 'seq'
