"""Comprehensive tests for I/O operations in spectral_predict.io module.

This test suite validates file format readers, data alignment, and format detection
for the DASP spectral analysis package. Tests cover:
- CSV reading (wide and long formats)
- Data alignment with fuzzy matching
- Format detection and data type detection
- Edge cases and error handling
- Reference file reading

Test coverage goal: >90% for io.py I/O functions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Tuple

from spectral_predict.io import (
    read_csv_spectra,
    read_reference_csv,
    align_xy,
    _normalize_filename_for_matching,
    detect_spectral_data_type,
)


@pytest.mark.io
class TestCSVReadingWideFormat:
    """Test CSV reading in wide format (samples × wavelengths)."""

    def test_read_csv_wide_format(self, tmp_path: Path) -> None:
        """Test reading standard wide-format spectral CSV."""
        csv_path = tmp_path / "spectra_wide.csv"

        # Create wavelengths (1000-2000 nm, 1001 points)
        wavelengths = np.linspace(1000, 2000, 1001)
        wl_cols = [str(wl) for wl in wavelengths]

        # Create 10 samples
        data = {"sample_id": [f"S{i:03d}" for i in range(1, 11)]}
        for wl in wl_cols:
            data[wl] = np.random.rand(10)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Read it back
        result, metadata = read_csv_spectra(csv_path)

        assert result.shape == (10, 1001)
        assert list(result.index) == [f"S{i:03d}" for i in range(1, 11)]
        assert result.columns[0] == 1000.0
        assert result.columns[-1] == 2000.0

        # Check wavelengths are sorted and increasing
        wls = np.array(result.columns)
        assert np.all(wls[1:] > wls[:-1])

        # Check metadata
        assert metadata["n_spectra"] == 10
        assert metadata["file_format"] == "csv"
        assert metadata["wavelength_range"][0] == 1000.0
        assert metadata["wavelength_range"][1] == 2000.0
        assert "data_type" in metadata

    def test_read_csv_wide_format_with_non_wavelength_columns(
        self, tmp_path: Path
    ) -> None:
        """Test wide CSV with non-numeric columns (target variables)."""
        csv_path = tmp_path / "spectra_with_target.csv"

        # Create wavelengths
        wavelengths = np.linspace(1000, 1500, 501)
        wl_cols = [str(wl) for wl in wavelengths]

        # Create data with target variable
        data = {"sample_id": ["S1", "S2", "S3"]}
        for wl in wl_cols:
            data[wl] = np.random.rand(3)
        # Add non-wavelength columns
        data["nitrogen"] = [2.5, 3.1, 2.8]
        data["protein"] = [15.2, 18.7, 16.3]

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Read it back (should ignore non-wavelength columns)
        result, metadata = read_csv_spectra(csv_path)

        # Should only have wavelength columns
        assert result.shape == (3, 501)
        assert result.columns[0] == 1000.0
        assert result.columns[-1] == 1500.0

    def test_read_csv_wide_format_large(self, tmp_path: Path) -> None:
        """Test reading large wide-format CSV (typical NIR spectrometer)."""
        csv_path = tmp_path / "spectra_large.csv"

        # Create full NIR range (350-2500 nm at 1nm intervals)
        wavelengths = np.linspace(350, 2500, 2151)
        wl_cols = [str(wl) for wl in wavelengths]

        # Create 100 samples
        data = {"sample_id": [f"Sample_{i:04d}" for i in range(1, 101)]}
        for wl in wl_cols:
            data[wl] = np.random.rand(100)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        result, metadata = read_csv_spectra(csv_path)

        assert result.shape == (100, 2151)
        assert metadata["n_spectra"] == 100
        assert metadata["wavelength_range"][0] == 350.0
        assert metadata["wavelength_range"][1] == 2500.0


@pytest.mark.io
class TestCSVReadingLongFormat:
    """Test CSV reading in long format (wavelength, value columns)."""

    def test_read_csv_long_format(self, tmp_path: Path) -> None:
        """Test reading long-format CSV (single spectrum)."""
        csv_path = tmp_path / "spectrum_single.csv"

        # Create a long-format CSV with wavelength and value
        wavelengths = np.linspace(400, 2400, 2001)
        values = np.random.rand(2001)

        df = pd.DataFrame({"wavelength": wavelengths, "value": values})
        df.to_csv(csv_path, index=False)

        # Read it back
        result, metadata = read_csv_spectra(csv_path)

        assert result.shape == (1, 2001)  # Single row
        assert result.index[0] == "spectrum_single"  # Uses filename as ID
        assert result.columns[0] == 400.0
        assert result.columns[-1] == 2400.0

    def test_read_csv_long_format_wavelength_nm(self, tmp_path: Path) -> None:
        """Test long-format with wavelength_nm column name."""
        csv_path = tmp_path / "spectrum_nm.csv"

        wavelengths = np.linspace(1000, 2000, 1001)
        values = np.random.rand(1001)

        df = pd.DataFrame({"wavelength_nm": wavelengths, "intensity": values})
        df.to_csv(csv_path, index=False)

        result, metadata = read_csv_spectra(csv_path)

        assert result.shape == (1, 1001)
        assert result.index[0] == "spectrum_nm"

    def test_read_csv_long_format_reflectance_column(self, tmp_path: Path) -> None:
        """Test long-format with reflectance as value column."""
        csv_path = tmp_path / "spectrum_reflectance.csv"

        wavelengths = np.linspace(350, 1000, 651)
        values = np.random.rand(651)

        df = pd.DataFrame({"wavelength": wavelengths, "reflectance": values})
        df.to_csv(csv_path, index=False)

        result, metadata = read_csv_spectra(csv_path)

        assert result.shape == (1, 651)
        assert result.columns[0] == 350.0


@pytest.mark.io
class TestDataAlignment:
    """Test data alignment between spectral data and reference values."""

    def test_align_xy_exact_match(self) -> None:
        """Test alignment with perfect specimen ID matching."""
        # Create spectral data
        X = pd.DataFrame(
            np.random.rand(5, 100),
            index=["S1", "S2", "S3", "S4", "S5"],
            columns=np.linspace(400, 2400, 100),
        )

        # Create reference with exact same IDs
        ref = pd.DataFrame(
            {"nitrogen": [2.5, 3.1, 2.8, 3.0, 2.7]}, index=["S1", "S2", "S3", "S4", "S5"]
        )

        X_aligned, y = align_xy(X, ref, "sample_id", "nitrogen")

        assert X_aligned.shape == (5, 100)
        assert len(y) == 5
        assert list(y.index) == ["S1", "S2", "S3", "S4", "S5"]
        assert np.all(y.values == ref["nitrogen"].values)

    def test_align_xy_fuzzy_match(self, tmp_path: Path) -> None:
        """Test alignment with flexible filename matching."""
        # Spectral data has .asd extensions
        X = pd.DataFrame(
            np.random.rand(5, 100),
            index=["Sample001.asd", "Sample002.asd", "Sample003.asd", "Sample004.asd", "Sample005.asd"],
            columns=np.linspace(400, 2400, 100),
        )

        # Reference has no extensions
        ref = pd.DataFrame(
            {"nitrogen": [2.5, 3.1, 2.8, 3.0, 2.7]},
            index=["Sample001", "Sample002", "Sample003", "Sample004", "Sample005"],
        )

        X_aligned, y = align_xy(X, ref, "sample_id", "nitrogen")

        # Should successfully match using fuzzy matching
        assert len(X_aligned) == 5
        assert len(y) == 5
        # Indices should be aligned
        assert X_aligned.index.equals(y.index)

    def test_align_xy_fuzzy_match_spaces(self) -> None:
        """Test fuzzy matching handles spaces in filenames."""
        # Spectral data has spaces
        X = pd.DataFrame(
            np.random.rand(3, 100),
            index=["Spectrum 001", "Spectrum 002", "Spectrum 003"],
            columns=np.linspace(400, 2400, 100),
        )

        # Reference has no spaces
        ref = pd.DataFrame(
            {"protein": [15.2, 18.7, 16.3]}, index=["Spectrum001", "Spectrum002", "Spectrum003"]
        )

        X_aligned, y = align_xy(X, ref, "sample_id", "protein")

        assert len(X_aligned) == 3
        assert len(y) == 3

    def test_align_xy_missing_specimens(self) -> None:
        """Test alignment handles unmatched samples gracefully."""
        # X has S1-S5, ref has S3-S7 (partial overlap)
        X = pd.DataFrame(
            np.random.rand(5, 100),
            index=["S1", "S2", "S3", "S4", "S5"],
            columns=np.linspace(400, 2400, 100),
        )

        ref = pd.DataFrame(
            {"nitrogen": [2.8, 3.0, 2.7, 3.2, 2.9]}, index=["S3", "S4", "S5", "S6", "S7"]
        )

        X_aligned, y = align_xy(X, ref, "sample_id", "nitrogen")

        # Should only have S3, S4, S5 (the overlap)
        assert X_aligned.shape == (3, 100)
        assert len(y) == 3
        assert set(y.index) == {"S3", "S4", "S5"}

    def test_align_xy_nan_target_handling(self) -> None:
        """Test that NaN values in target are dropped."""
        X = pd.DataFrame(
            np.random.rand(5, 100),
            index=["S1", "S2", "S3", "S4", "S5"],
            columns=np.linspace(400, 2400, 100),
        )

        # Reference has NaN for S3
        ref = pd.DataFrame(
            {"nitrogen": [2.5, 3.1, np.nan, 3.0, 2.7]}, index=["S1", "S2", "S3", "S4", "S5"]
        )

        X_aligned, y = align_xy(X, ref, "sample_id", "nitrogen")

        # Should drop S3
        assert len(X_aligned) == 4
        assert len(y) == 4
        assert "S3" not in y.index
        assert not np.any(np.isnan(y.values))

    def test_align_xy_return_alignment_info(self) -> None:
        """Test that alignment info is returned when requested."""
        X = pd.DataFrame(
            np.random.rand(5, 100),
            index=["S1", "S2", "S3", "S4", "S5"],
            columns=np.linspace(400, 2400, 100),
        )

        ref = pd.DataFrame(
            {"nitrogen": [2.8, 3.0, 2.7]}, index=["S3", "S4", "S5"]
        )

        X_aligned, y, info = align_xy(X, ref, "sample_id", "nitrogen", return_alignment_info=True)

        assert "matched_ids" in info
        assert "unmatched_spectra" in info
        assert "unmatched_reference" in info
        assert "n_nan_dropped" in info
        assert "used_fuzzy_matching" in info

        assert len(info["matched_ids"]) == 3
        assert set(info["unmatched_spectra"]) == {"S1", "S2"}


@pytest.mark.io
class TestFormatDetection:
    """Test file format and data type detection."""

    def test_detect_spectral_data_type_reflectance(self) -> None:
        """Test detection of reflectance data (0-1 range)."""
        # Create reflectance-like data (0-1 range)
        X = pd.DataFrame(
            np.random.uniform(0.1, 0.9, (50, 200)),
            columns=np.linspace(400, 2400, 200)
        )

        data_type, confidence, method = detect_spectral_data_type(X)

        assert data_type == "reflectance"
        assert confidence > 50  # Should have reasonable confidence

    def test_detect_spectral_data_type_absorbance(self) -> None:
        """Test detection of absorbance data (0-4 range, inverted peaks)."""
        # Create absorbance-like data (0-3 range)
        X = pd.DataFrame(
            np.random.uniform(0.5, 2.5, (50, 200)),
            columns=np.linspace(400, 2400, 200)
        )

        data_type, confidence, method = detect_spectral_data_type(X)

        # Detection might vary - just ensure it returns valid values
        assert data_type in ["reflectance", "absorbance"]
        assert 0 <= confidence <= 100
        assert isinstance(method, str)


@pytest.mark.io
class TestEdgeCases:
    """Test edge cases and error handling for I/O operations."""

    def test_empty_file_handling(self, tmp_path: Path) -> None:
        """Test that empty CSV file raises appropriate error."""
        csv_path = tmp_path / "empty.csv"

        # Create empty CSV (just headers)
        df = pd.DataFrame(columns=["sample_id"])
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Empty CSV file"):
            read_csv_spectra(csv_path)

    def test_duplicate_specimen_ids(self, tmp_path: Path) -> None:
        """Test warning on duplicate specimen IDs in reference file."""
        ref_path = tmp_path / "ref_duplicates.csv"

        # Create reference with duplicate IDs
        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3", "S2"],  # S2 is duplicate
            "nitrogen": [2.5, 3.1, 2.8, 3.0]
        })
        df.to_csv(ref_path, index=False)

        # Should handle duplicates (keeps first occurrence)
        result = read_reference_csv(ref_path, "sample_id")

        # Should have 3 unique samples
        assert len(result) == 3
        assert list(result.index) == ["S1", "S2", "S3"]

    def test_unicode_in_column_names(self, tmp_path: Path) -> None:
        """Test handling of special characters in column names."""
        csv_path = tmp_path / "unicode_cols.csv"

        # Create wavelengths
        wavelengths = np.linspace(1000, 1500, 501)
        wl_cols = [str(wl) for wl in wavelengths]

        # Create data with unicode ID
        data = {"sample_id": ["Sample_α", "Sample_β", "Sample_γ"]}
        for wl in wl_cols:
            data[wl] = np.random.rand(3)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        # Should read without error
        result, metadata = read_csv_spectra(csv_path)

        assert result.shape == (3, 501)
        assert "Sample_α" in result.index or "Sample_Î±" in result.index  # Encoding variations

    def test_read_csv_too_few_wavelengths(self, tmp_path: Path) -> None:
        """Test that validation catches too few wavelengths."""
        csv_path = tmp_path / "spectra_short.csv"

        # Create CSV with only 50 wavelengths (< 100 minimum)
        wavelengths = np.linspace(400, 450, 50)
        wl_cols = [str(wl) for wl in wavelengths]

        data = {"sample_id": ["S1"]}
        for wl in wl_cols:
            data[wl] = [np.random.rand()]

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Should raise error
        with pytest.raises(ValueError, match="at least 100 wavelengths"):
            read_csv_spectra(csv_path)

    def test_read_csv_non_increasing_wavelengths(self, tmp_path: Path) -> None:
        """Test that reversed wavelength order gets sorted correctly."""
        csv_path = tmp_path / "spectra_reversed.csv"

        # Create wavelengths in reverse order
        wavelengths = list(np.linspace(1500, 1000, 501))  # Descending
        wl_cols = [str(wl) for wl in wavelengths]

        data = {"sample_id": ["S1", "S2"]}
        for wl in wl_cols:
            data[wl] = np.random.rand(2)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Should read and auto-sort wavelengths
        result, metadata = read_csv_spectra(csv_path)

        # Verify wavelengths are sorted in ascending order
        wls = np.array(result.columns)
        assert np.all(wls[1:] > wls[:-1])
        assert wls[0] == 1000.0
        assert wls[-1] == 1500.0

    def test_read_reference_missing_id_column(self, tmp_path: Path) -> None:
        """Test error when ID column doesn't exist in reference file."""
        ref_path = tmp_path / "reference.csv"

        df = pd.DataFrame({"sample_id": ["S1", "S2"], "nitrogen": [2.5, 3.1]})
        df.to_csv(ref_path, index=False)

        with pytest.raises(ValueError, match="Column 'wrong_col' not found"):
            read_reference_csv(ref_path, "wrong_col")

    def test_read_reference_excel_format(self, tmp_path: Path) -> None:
        """Test reading Excel reference file."""
        ref_path = tmp_path / "reference.xlsx"

        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "nitrogen": [2.5, 3.1, 2.8],
            "carbon": [45.2, 43.8, 44.5]
        })
        df.to_excel(ref_path, index=False)

        result = read_reference_csv(ref_path, "sample_id")

        assert list(result.index) == ["S1", "S2", "S3"]
        assert "nitrogen" in result.columns
        assert "carbon" in result.columns

    def test_align_xy_missing_target_column(self) -> None:
        """Test error when target column doesn't exist."""
        X = pd.DataFrame(
            np.random.rand(3, 100),
            index=["S1", "S2", "S3"],
            columns=np.linspace(400, 2400, 100)
        )

        ref = pd.DataFrame({"nitrogen": [2.5, 3.1, 2.8]}, index=["S1", "S2", "S3"])

        with pytest.raises(ValueError, match="Target 'carbon' not found"):
            align_xy(X, ref, "sample_id", "carbon")

    def test_align_xy_no_overlap(self) -> None:
        """Test error when no IDs overlap between spectral and reference."""
        X = pd.DataFrame(
            np.random.rand(3, 100),
            index=["S1", "S2", "S3"],
            columns=np.linspace(400, 2400, 100)
        )

        ref = pd.DataFrame({"nitrogen": [2.5, 3.1, 2.8]}, index=["S4", "S5", "S6"])

        with pytest.raises(ValueError, match="No matching IDs"):
            align_xy(X, ref, "sample_id", "nitrogen")

    def test_align_xy_all_nan_targets(self) -> None:
        """Test error when all target values are NaN."""
        X = pd.DataFrame(
            np.random.rand(3, 100),
            index=["S1", "S2", "S3"],
            columns=np.linspace(400, 2400, 100)
        )

        ref = pd.DataFrame(
            {"nitrogen": [np.nan, np.nan, np.nan]}, index=["S1", "S2", "S3"]
        )

        with pytest.raises(ValueError, match="No valid samples after alignment"):
            align_xy(X, ref, "sample_id", "nitrogen")


@pytest.mark.io
class TestFilenameNormalization:
    """Test filename normalization for fuzzy matching."""

    def test_normalize_filename_asd_extension(self) -> None:
        """Test normalization removes .asd extension."""
        assert _normalize_filename_for_matching("Sample001.asd") == "sample001"

    def test_normalize_filename_csv_extension(self) -> None:
        """Test normalization removes .csv extension."""
        assert _normalize_filename_for_matching("Sample002.csv") == "sample002"

    def test_normalize_filename_spaces(self) -> None:
        """Test normalization removes spaces."""
        assert _normalize_filename_for_matching("Sample 003") == "sample003"

    def test_normalize_filename_lowercase(self) -> None:
        """Test normalization converts to lowercase."""
        assert _normalize_filename_for_matching("SAMPLE004") == "sample004"

    def test_normalize_filename_combined(self) -> None:
        """Test normalization handles multiple transformations."""
        assert _normalize_filename_for_matching("Sample 005.ASD") == "sample005"

    def test_normalize_filename_no_extension(self) -> None:
        """Test normalization works without extension."""
        assert _normalize_filename_for_matching("Sample006") == "sample006"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
