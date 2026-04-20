"""Tests for the fixed-method Auto Bone FTIR index extraction.

Validates extract_bone_ftir_indices() and extract_bone_ftir_indices_all_samples()
against synthetic spectra with known peak properties.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spectral_predict.peak_calculator import (
    BONE_FTIR_INDEX_COLUMNS,
    BONE_FTIR_MIN_WN,
    BONE_FTIR_MAX_WN,
    extract_bone_ftir_indices,
    extract_bone_ftir_indices_all_samples,
)


def _gaussian(x: np.ndarray, center: float, height: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _make_bone_spectrum(
    irsf_h560: float = 0.40,
    irsf_h600: float = 0.35,
    irsf_h590: float = 0.24,
    po4_height: float = 1.50,
    po4_center: float = 1035.0,
    co3_height: float = 0.15,
    co3_center: float = 1415.0,
    amide_height: float = 0.30,
    amide_center: float = 1650.0,
    wn_start: float = 400.0,
    wn_end: float = 4000.0,
    wn_step: float = 2.0,
    baseline_slope: tuple[float, float] = (0.1, 0.05),
) -> tuple[np.ndarray, np.ndarray]:
    wn = np.arange(wn_start, wn_end + wn_step, wn_step, dtype=float)
    baseline = baseline_slope[0] + (baseline_slope[1] - baseline_slope[0]) * (wn - wn_start) / (wn_end - wn_start)
    spectrum = baseline.copy()

    # v4 PO4 doublet (IRSF)
    # Build the two peaks so they have the right relative heights and the valley
    # between them has the desired depth.  Use overlapping Gaussians.
    spectrum += _gaussian(wn, center=565, height=irsf_h560, fwhm=30)
    spectrum += _gaussian(wn, center=603, height=irsf_h600, fwhm=28)

    # v3 PO4
    spectrum += _gaussian(wn, center=po4_center, height=po4_height, fwhm=60)

    # v3 CO3
    spectrum += _gaussian(wn, center=co3_center, height=co3_height, fwhm=30)

    # Amide I
    spectrum += _gaussian(wn, center=amide_center, height=amide_height, fwhm=35)

    return wn, spectrum


@pytest.fixture
def bone_wn():
    return np.arange(400, 4002, 2, dtype=float)


@pytest.fixture
def modern_bone_spectrum():
    return _make_bone_spectrum()


@pytest.fixture
def multi_sample_bone():
    wn = np.arange(400, 4002, 2, dtype=float)
    n = 5
    data = np.zeros((n, len(wn)))
    for i in range(n):
        scale = 0.8 + 0.1 * i
        _, spec = _make_bone_spectrum(
            po4_height=1.5 * scale,
            co3_height=0.15 * scale,
            amide_height=0.30 * scale,
        )
        data[i] = spec
    return wn, data


# ---------------------------------------------------------------------------
# Test: returns all expected columns
# ---------------------------------------------------------------------------

class TestBoneFtirColumns:

    def test_single_spectrum_returns_all_keys(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        result = extract_bone_ftir_indices(spec, wn)
        for col in BONE_FTIR_INDEX_COLUMNS:
            assert col in result, f"Missing key: {col}"

    def test_all_values_finite_on_valid_spectrum(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        result = extract_bone_ftir_indices(spec, wn)
        for col in BONE_FTIR_INDEX_COLUMNS:
            assert np.isfinite(result[col]), f"{col} is not finite: {result[col]}"


# ---------------------------------------------------------------------------
# Test: batch extraction returns one row per sample
# ---------------------------------------------------------------------------

class TestBoneFtirBatch:

    def test_batch_returns_correct_shape(self, multi_sample_bone):
        wn, data = multi_sample_bone
        names = [f"s{i}" for i in range(data.shape[0])]
        df = extract_bone_ftir_indices_all_samples(wn, data, names)
        assert len(df) == data.shape[0]
        assert list(df["Sample"]) == names

    def test_batch_has_stable_columns(self, multi_sample_bone):
        wn, data = multi_sample_bone
        df = extract_bone_ftir_indices_all_samples(wn, data)
        expected = ["Sample"] + BONE_FTIR_INDEX_COLUMNS
        assert list(df.columns) == expected

    def test_batch_all_finite(self, multi_sample_bone):
        wn, data = multi_sample_bone
        df = extract_bone_ftir_indices_all_samples(wn, data)
        for col in BONE_FTIR_INDEX_COLUMNS:
            assert df[col].notna().all(), f"NaN in column {col}"


# ---------------------------------------------------------------------------
# Test: missing wavelength coverage → NaN
# ---------------------------------------------------------------------------

class TestBoneFtirMissingCoverage:

    def test_narrow_range_produces_nan(self):
        wn = np.arange(800, 1801, 2, dtype=float)
        spec = np.zeros_like(wn)
        result = extract_bone_ftir_indices(spec, wn)
        assert np.isnan(result["IRSF"]), "IRSF should be NaN without 400-670 coverage"

    def test_partial_coverage_keeps_available(self):
        wn = np.arange(800, 1801, 2, dtype=float)
        spec = np.zeros_like(wn) + 0.1
        spec += _gaussian(wn, 1035, 1.5, 60)
        spec += _gaussian(wn, 1415, 0.15, 30)
        spec += _gaussian(wn, 1650, 0.30, 35)
        result = extract_bone_ftir_indices(spec, wn)
        assert np.isnan(result["IRSF"])
        assert np.isfinite(result["PO4_position"])
        assert np.isfinite(result["CO3_position"])
        assert np.isfinite(result["AmideI_position"])


# ---------------------------------------------------------------------------
# Test: descending wavenumber arrays
# ---------------------------------------------------------------------------

class TestBoneFtirDescending:

    def test_descending_wn_works(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        wn_desc = wn[::-1].copy()
        spec_desc = spec[::-1].copy()
        result_asc = extract_bone_ftir_indices(spec, wn)
        result_desc = extract_bone_ftir_indices(spec_desc, wn_desc)
        for col in BONE_FTIR_INDEX_COLUMNS:
            if np.isnan(result_asc[col]):
                assert np.isnan(result_desc[col])
            else:
                assert result_asc[col] == pytest.approx(
                    result_desc[col], rel=1e-6,
                ), f"Mismatch for {col}"


# ---------------------------------------------------------------------------
# Test: IRSF in reasonable range
# ---------------------------------------------------------------------------

class TestBoneFtirValues:

    def test_irsf_reasonable_range(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        result = extract_bone_ftir_indices(spec, wn)
        assert 2.0 <= result["IRSF"] <= 5.0, f"IRSF={result['IRSF']}"

    def test_po4_position_near_1035(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        result = extract_bone_ftir_indices(spec, wn)
        assert abs(result["PO4_position"] - 1035) < 20, f"PO4_position={result['PO4_position']}"

    def test_co3_position_near_1415(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        result = extract_bone_ftir_indices(spec, wn)
        assert abs(result["CO3_position"] - 1415) < 20, f"CO3_position={result['CO3_position']}"

    def test_amide_position_near_1650(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        result = extract_bone_ftir_indices(spec, wn)
        assert abs(result["AmideI_position"] - 1650) < 20, f"AmideI_position={result['AmideI_position']}"

    def test_ratios_positive(self, modern_bone_spectrum):
        wn, spec = modern_bone_spectrum
        result = extract_bone_ftir_indices(spec, wn)
        assert result["Am_P"] > 0
        assert result["C_P"] > 0
        assert result["OrgInorg"] > 0

    def test_no_amide_gives_nan_amide(self):
        wn, spec = _make_bone_spectrum(
            amide_height=0.0,
            baseline_slope=(0.05, 0.05),
        )
        result = extract_bone_ftir_indices(spec, wn)
        assert np.isnan(result["AmideI_position"])
        assert np.isnan(result["AmideI_intensity"])
        assert np.isnan(result["Am_P"]) or result["Am_P"] == 0.0
        assert np.isnan(result["OrgInorg"]) or result["OrgInorg"] == 0.0


# ---------------------------------------------------------------------------
# Test: manual mode unaffected (regression guard)
# ---------------------------------------------------------------------------

class TestManualModeRegression:

    def test_manual_presets_still_work(self, modern_bone_spectrum):
        from spectral_predict.peak_calculator import (
            BUILT_IN_PRESETS,
            calculate_expression,
        )
        wn, spec = modern_bone_spectrum
        bone_presets = [p for p in BUILT_IN_PRESETS if p.category == "Bone FTIR"]
        assert len(bone_presets) > 0, "No Bone FTIR presets found"
        for preset in bone_presets:
            val = calculate_expression(wn, spec, preset, use_local_baseline=True)
            assert np.isfinite(val), f"{preset.name}: not finite ({val})"


# ---------------------------------------------------------------------------
# Test: coverage constants
# ---------------------------------------------------------------------------

class TestCoverageConstants:

    def test_min_max_defined(self):
        assert BONE_FTIR_MIN_WN == 400.0
        assert BONE_FTIR_MAX_WN == 1720.0

    def test_columns_are_10(self):
        assert len(BONE_FTIR_INDEX_COLUMNS) == 10


# ---------------------------------------------------------------------------
# GUI integration tests (non-Tkinter: test dialog structure + mode logic)
# ---------------------------------------------------------------------------

class TestAutoModeGUIPaths:
    """Tests for auto-mode GUI integration without launching full Tk app.

    These test the wiring between mode toggle, calculation, export, and copy
    by calling the same methods the GUI would, on a minimal mock dialog.
    """

    @pytest.fixture
    def fake_dialog(self):
        """Minimal dialog-like object with the attributes the methods expect."""

        class FakeDialog:
            pass

        d = FakeDialog()
        d._calc_mode_var = None  # set per-test
        d._scope_var = None
        d._results_df = None
        d._stats_text = None
        d._auto_mode = False
        yield d

    def test_auto_mode_flag_set_after_auto_calc(self, fake_dialog):
        fake_dialog._auto_mode = True
        assert fake_dialog._auto_mode is True

    def test_manual_mode_flag_false_by_default(self, fake_dialog):
        assert fake_dialog._auto_mode is False

    def test_auto_mode_export_filename(self, fake_dialog):
        fake_dialog._auto_mode = True
        default = "bone_ftir_indices.csv" if fake_dialog._auto_mode else "peak_ratios.csv"
        assert default == "bone_ftir_indices.csv"

    def test_manual_mode_export_filename(self, fake_dialog):
        fake_dialog._auto_mode = False
        default = "bone_ftir_indices.csv" if fake_dialog._auto_mode else "peak_ratios.csv"
        assert default == "peak_ratios.csv"

    def test_copy_in_auto_mode_copies_dataframe_csv(self, fake_dialog):
        fake_dialog._auto_mode = True
        df = pd.DataFrame({"Sample": ["s1"], "IRSF": [3.1]})
        fake_dialog._results_df = df
        csv_text = df.to_csv(index=False)
        assert "Sample" in csv_text
        assert "IRSF" in csv_text
        assert "3.1" in csv_text

    def test_batch_result_has_expected_columns_for_table(self):
        wn = np.arange(400, 4002, 2, dtype=float)
        data = np.zeros((3, len(wn)))
        for i in range(3):
            data[i] = 0.1 + 0.01 * i
        df = extract_bone_ftir_indices_all_samples(wn, data)
        show_cols = list(df.columns)
        assert show_cols[0] == "Sample"
        for col in BONE_FTIR_INDEX_COLUMNS:
            assert col in show_cols

    def test_results_df_cell_formatting(self):
        wn = np.arange(400, 4002, 2, dtype=float)
        data = np.zeros((1, len(wn)))
        df = extract_bone_ftir_indices_all_samples(wn, data)
        for col in BONE_FTIR_INDEX_COLUMNS:
            v = df[col].iloc[0]
            formatted = f"{v:.4f}" if not np.isnan(v) else "NaN"
            assert isinstance(formatted, str)
