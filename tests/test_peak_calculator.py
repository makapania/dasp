"""Comprehensive tests for spectral_predict.peak_calculator module."""

import json

import numpy as np
import pandas as pd
import pytest

from spectral_predict.peak_calculator import (
    BUILT_IN_PRESETS,
    BaselineRegion,
    PeakDefinition,
    PeakPreset,
    baseline_at_wavenumber,
    calculate_all_samples,
    calculate_expression,
    calculate_expression_detailed,
    find_peak_in_window,
    find_trough_in_window,
    get_baseline_corrected_intensity,
    get_peak_intensity,
    load_user_presets,
    save_user_presets,
    _peak_def_to_dict,
    _peak_def_from_dict,
    _preset_to_dict,
    _preset_from_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_wavelengths():
    """Wavelength axis from 400 to 4000 cm-1 with 1 cm-1 spacing."""
    return np.arange(400, 4001, dtype=float)


@pytest.fixture
def flat_spectrum(simple_wavelengths):
    """Flat spectrum of constant value 1.0."""
    return np.ones_like(simple_wavelengths)


@pytest.fixture
def peaked_spectrum(simple_wavelengths):
    """Spectrum with Gaussian peaks at known positions (1000, 2000, 3000 cm-1)."""
    spectrum = np.zeros_like(simple_wavelengths)
    for center, height in [(1000, 5.0), (2000, 3.0), (3000, 8.0)]:
        spectrum += height * np.exp(-0.5 * ((simple_wavelengths - center) / 10) ** 2)
    return spectrum


@pytest.fixture
def multi_sample_data(simple_wavelengths):
    """Data matrix with 4 samples, each having different peak heights at 1000 cm-1."""
    n_samples = 4
    data = np.zeros((n_samples, len(simple_wavelengths)))
    for i in range(n_samples):
        data[i] = (i + 1) * np.exp(
            -0.5 * ((simple_wavelengths - 1000) / 10) ** 2
        )
        data[i] += (n_samples - i) * np.exp(
            -0.5 * ((simple_wavelengths - 2000) / 10) ** 2
        )
    return data


@pytest.fixture
def basic_preset():
    """Simple two-peak ratio preset (A / B)."""
    return PeakPreset(
        name="Test Ratio",
        peak_a=PeakDefinition(1000, "point", 10, "Peak A"),
        peak_b=PeakDefinition(2000, "point", 10, "Peak B"),
        operator1="/",
        description="Test ratio A/B",
        category="Test",
    )


@pytest.fixture
def three_peak_preset_left():
    """Three-peak preset with left grouping: (A + B) / C."""
    return PeakPreset(
        name="Three Peak Left",
        peak_a=PeakDefinition(1000, "point", 10, "A"),
        peak_b=PeakDefinition(2000, "point", 10, "B"),
        operator1="+",
        peak_c=PeakDefinition(3000, "point", 10, "C"),
        operator2="/",
        grouping="left",
    )


@pytest.fixture
def three_peak_preset_right():
    """Three-peak preset with right grouping: A / (B + C)."""
    return PeakPreset(
        name="Three Peak Right",
        peak_a=PeakDefinition(1000, "point", 10, "A"),
        peak_b=PeakDefinition(2000, "point", 10, "B"),
        operator1="/",
        peak_c=PeakDefinition(3000, "point", 10, "C"),
        operator2="+",
        grouping="right",
    )


# ---------------------------------------------------------------------------
# PeakDefinition dataclass tests
# ---------------------------------------------------------------------------

def test_peak_definition_defaults():
    """PeakDefinition should initialise with documented defaults."""
    p = PeakDefinition()
    assert p.wavenumber == 0.0
    assert p.mode == "point"
    assert p.half_width == 10.0
    assert p.label == ""


def test_peak_definition_custom_values():
    """PeakDefinition should accept and store custom values."""
    p = PeakDefinition(wavenumber=1020.0, mode="range", half_width=30.0, label="Si-O")
    assert p.wavenumber == 1020.0
    assert p.mode == "range"
    assert p.half_width == 30.0
    assert p.label == "Si-O"


# ---------------------------------------------------------------------------
# PeakPreset dataclass tests
# ---------------------------------------------------------------------------

def test_peak_preset_defaults():
    """PeakPreset should initialise with documented defaults."""
    pp = PeakPreset()
    assert pp.name == ""
    assert pp.operator1 == "/"
    assert pp.peak_c is None
    assert pp.operator2 == "/"
    assert pp.grouping == "left"
    assert pp.category == "User"
    assert pp.x_unit == "cm-1"


def test_peak_preset_with_three_peaks():
    """PeakPreset should store a third peak and grouping correctly."""
    pp = PeakPreset(
        name="Crystallinity",
        peak_a=PeakDefinition(604),
        peak_b=PeakDefinition(564),
        operator1="+",
        peak_c=PeakDefinition(590),
        operator2="/",
        grouping="left",
    )
    assert pp.peak_c is not None
    assert pp.peak_c.wavenumber == 590
    assert pp.grouping == "left"


# ---------------------------------------------------------------------------
# BUILT_IN_PRESETS tests
# ---------------------------------------------------------------------------

def test_built_in_presets_is_non_empty_list():
    """BUILT_IN_PRESETS should be a non-empty list of PeakPreset objects."""
    assert isinstance(BUILT_IN_PRESETS, list)
    assert len(BUILT_IN_PRESETS) > 0
    for preset in BUILT_IN_PRESETS:
        assert isinstance(preset, PeakPreset)


def test_built_in_presets_have_required_fields():
    """Every built-in preset must have a name and a category."""
    for preset in BUILT_IN_PRESETS:
        assert preset.name, f"Preset missing name: {preset}"
        assert preset.category, f"Preset missing category: {preset}"
        assert preset.peak_a.wavenumber > 0, f"Peak A wavenumber must be positive: {preset.name}"
        assert preset.peak_b.wavenumber > 0, f"Peak B wavenumber must be positive: {preset.name}"


def test_built_in_presets_contain_expected_categories():
    """BUILT_IN_PRESETS should include at least the Bone FTIR and Mineralogical categories."""
    categories = {p.category for p in BUILT_IN_PRESETS}
    assert "Bone FTIR" in categories
    assert "Mineralogical" in categories


# ---------------------------------------------------------------------------
# get_peak_intensity tests
# ---------------------------------------------------------------------------

def test_get_peak_intensity_point_mode_exact(simple_wavelengths, peaked_spectrum):
    """Point mode should return intensity at nearest wavelength to peak centre."""
    peak_def = PeakDefinition(wavenumber=1000, mode="point")
    val = get_peak_intensity(simple_wavelengths, peaked_spectrum, peak_def)
    # The Gaussian peak at 1000 has height 5.0
    assert pytest.approx(val, abs=0.01) == 5.0


def test_get_peak_intensity_point_mode_off_grid(simple_wavelengths, peaked_spectrum):
    """Point mode at a non-grid position picks nearest grid point."""
    peak_def = PeakDefinition(wavenumber=1000.3, mode="point")
    val = get_peak_intensity(simple_wavelengths, peaked_spectrum, peak_def)
    # Should snap to 1000 and return ~5.0
    assert pytest.approx(val, abs=0.01) == 5.0


def test_get_peak_intensity_range_mode_max(simple_wavelengths, peaked_spectrum):
    """Range mode should return the maximum within the specified window."""
    peak_def = PeakDefinition(wavenumber=1000, mode="range", half_width=20)
    val = get_peak_intensity(simple_wavelengths, peaked_spectrum, peak_def)
    assert pytest.approx(val, abs=0.01) == 5.0


def test_get_peak_intensity_range_mode_fallback_to_nearest(simple_wavelengths):
    """Range mode should fall back to nearest point when no wavelength in range."""
    # Use a linearly increasing spectrum so each index has a distinct value
    spectrum = np.arange(len(simple_wavelengths), dtype=float)
    # Request a range entirely outside the axis (centred at 100, which is below 400)
    peak_def = PeakDefinition(wavenumber=100, mode="range", half_width=5)
    val = get_peak_intensity(simple_wavelengths, spectrum, peak_def)
    # Should snap to 400 (index 0) and return 0.0
    assert val == 0.0


def test_get_peak_intensity_flat_spectrum(simple_wavelengths, flat_spectrum):
    """Any position on a flat spectrum should return the constant value."""
    for wn in [500, 1500, 3500]:
        peak_def = PeakDefinition(wavenumber=wn, mode="point")
        assert get_peak_intensity(simple_wavelengths, flat_spectrum, peak_def) == 1.0


# ---------------------------------------------------------------------------
# calculate_expression tests (indirectly tests _apply_op)
# ---------------------------------------------------------------------------

def test_calculate_expression_two_peak_division(
    simple_wavelengths, peaked_spectrum, basic_preset
):
    """Two-peak division: peak_a(1000)=5.0, peak_b(2000)=3.0 => 5/3."""
    result = calculate_expression(simple_wavelengths, peaked_spectrum, basic_preset)
    assert pytest.approx(result, rel=0.01) == 5.0 / 3.0


def test_calculate_expression_division_by_zero(simple_wavelengths):
    """Division by zero peak should return NaN."""
    # Spectrum is zero everywhere except at 1000
    spectrum = np.zeros_like(simple_wavelengths)
    spectrum[simple_wavelengths == 1000] = 5.0
    preset = PeakPreset(
        name="div0",
        peak_a=PeakDefinition(1000, "point"),
        peak_b=PeakDefinition(2000, "point"),
        operator1="/",
    )
    result = calculate_expression(simple_wavelengths, spectrum, preset)
    assert np.isnan(result)


def test_calculate_expression_three_peak_left_grouping(
    simple_wavelengths, peaked_spectrum, three_peak_preset_left
):
    """Left grouping: (A + B) / C => (5 + 3) / 8 = 1.0."""
    result = calculate_expression(
        simple_wavelengths, peaked_spectrum, three_peak_preset_left
    )
    assert pytest.approx(result, rel=0.01) == (5.0 + 3.0) / 8.0


def test_calculate_expression_three_peak_right_grouping(
    simple_wavelengths, peaked_spectrum, three_peak_preset_right
):
    """Right grouping: A / (B + C) => 5 / (3 + 8)."""
    result = calculate_expression(
        simple_wavelengths, peaked_spectrum, three_peak_preset_right
    )
    assert pytest.approx(result, rel=0.01) == 5.0 / (3.0 + 8.0)


def test_calculate_expression_multiplication():
    """Multiplication operator should work correctly."""
    wl = np.array([100.0, 200.0])
    spec = np.array([2.0, 3.0])
    preset = PeakPreset(
        name="mult",
        peak_a=PeakDefinition(100),
        peak_b=PeakDefinition(200),
        operator1="*",
    )
    assert calculate_expression(wl, spec, preset) == 6.0


def test_calculate_expression_subtraction():
    """Subtraction operator should work correctly."""
    wl = np.array([100.0, 200.0])
    spec = np.array([7.0, 2.0])
    preset = PeakPreset(
        name="sub",
        peak_a=PeakDefinition(100),
        peak_b=PeakDefinition(200),
        operator1="-",
    )
    assert calculate_expression(wl, spec, preset) == 5.0


# ---------------------------------------------------------------------------
# calculate_all_samples tests
# ---------------------------------------------------------------------------

def test_calculate_all_samples_shape(
    simple_wavelengths, multi_sample_data, basic_preset
):
    """Output DataFrame should have one row per sample."""
    df = calculate_all_samples(simple_wavelengths, multi_sample_data, basic_preset)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == multi_sample_data.shape[0]
    assert list(df.columns) == ["Sample", "Value"]


def test_calculate_all_samples_with_names(
    simple_wavelengths, multi_sample_data, basic_preset
):
    """Sample names should be propagated into the output DataFrame."""
    names = ["S1", "S2", "S3", "S4"]
    df = calculate_all_samples(
        simple_wavelengths, multi_sample_data, basic_preset, sample_names=names
    )
    assert list(df["Sample"]) == names


def test_calculate_all_samples_values_vary(
    simple_wavelengths, multi_sample_data, basic_preset
):
    """Different sample spectra should produce different ratio values."""
    df = calculate_all_samples(simple_wavelengths, multi_sample_data, basic_preset)
    values = df["Value"].values
    assert len(np.unique(values)) > 1, "Expected distinct values across samples"


# ---------------------------------------------------------------------------
# User preset I/O tests
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """Saving then loading presets should reproduce the originals."""
    monkeypatch.setattr(
        "spectral_predict.peak_calculator._USER_DIR", tmp_path
    )
    monkeypatch.setattr(
        "spectral_predict.peak_calculator._USER_PRESETS_FILE",
        tmp_path / "peak_presets.json",
    )

    presets = [
        PeakPreset(
            name="MyPreset",
            peak_a=PeakDefinition(1020, "point", 10, "A"),
            peak_b=PeakDefinition(1660, "point", 10, "B"),
            operator1="/",
            description="custom",
            category="User",
        ),
        PeakPreset(
            name="ThreePeak",
            peak_a=PeakDefinition(604, "point", 10, "A"),
            peak_b=PeakDefinition(564, "point", 10, "B"),
            operator1="+",
            peak_c=PeakDefinition(590, "point", 10, "C"),
            operator2="/",
            grouping="left",
            description="crystallinity",
            category="User",
        ),
    ]

    save_user_presets(presets)
    loaded = load_user_presets()

    assert len(loaded) == 2
    assert loaded[0].name == "MyPreset"
    assert loaded[0].peak_a.wavenumber == 1020
    assert loaded[1].peak_c is not None
    assert loaded[1].peak_c.wavenumber == 590


def test_load_user_presets_returns_empty_when_no_file(tmp_path, monkeypatch):
    """load_user_presets should return [] when the file does not exist."""
    monkeypatch.setattr(
        "spectral_predict.peak_calculator._USER_PRESETS_FILE",
        tmp_path / "nonexistent.json",
    )
    assert load_user_presets() == []


def test_load_user_presets_returns_empty_on_corrupt_json(tmp_path, monkeypatch):
    """load_user_presets should return [] for invalid JSON."""
    bad_file = tmp_path / "peak_presets.json"
    bad_file.write_text("NOT VALID JSON {{", encoding="utf-8")
    monkeypatch.setattr(
        "spectral_predict.peak_calculator._USER_PRESETS_FILE", bad_file
    )
    assert load_user_presets() == []


# ---------------------------------------------------------------------------
# Fixtures for local baseline correction tests
# ---------------------------------------------------------------------------

@pytest.fixture
def bone_wavelengths():
    """Wavenumber axis covering the v4 PO4 region (380-800 cm-1) with 1 cm-1 spacing."""
    return np.arange(380, 801, dtype=float)


@pytest.fixture
def bone_spectrum_sloping(bone_wavelengths):
    """Synthetic bone-like spectrum: three peaks at 560, 590, 600 cm-1 on a sloping baseline.

    The sloping baseline runs linearly from 0.5 at 380 cm-1 to 1.5 at 800 cm-1.
    Gaussian peaks of known height sit on top:
      - 560 cm-1: height 3.0 above baseline
      - 590 cm-1: height 4.0 above baseline
      - 600 cm-1: height 3.5 above baseline
    """
    slope = 0.5 + (bone_wavelengths - 380) / (800 - 380)  # 0.5 -> 1.5
    peaks = np.zeros_like(bone_wavelengths)
    for center, height in [(560, 3.0), (590, 4.0), (600, 3.5)]:
        peaks += height * np.exp(-0.5 * ((bone_wavelengths - center) / 8) ** 2)
    return slope + peaks


@pytest.fixture
def bone_baseline_region():
    """Baseline region bracketing the v4 PO4 peaks: troughs near 490-510 and 690-750."""
    return BaselineRegion(left_min=490, left_max=510, right_min=690, right_max=750)


@pytest.fixture
def bone_preset_with_baseline(bone_baseline_region):
    """Crystallinity-like preset with local baseline on all peaks."""
    bl = bone_baseline_region
    return PeakPreset(
        name="Crystallinity BL",
        peak_a=PeakDefinition(600, "point", 10, "600", baseline=bl),
        peak_b=PeakDefinition(560, "point", 10, "560", baseline=bl),
        operator1="+",
        peak_c=PeakDefinition(590, "point", 10, "590", baseline=bl),
        operator2="/",
        grouping="left",
        description="(600 + 560) / 590 with local baseline",
        category="Bone FTIR",
    )


@pytest.fixture
def bone_preset_no_baseline():
    """Same peak positions but without any baseline region."""
    return PeakPreset(
        name="Crystallinity Raw",
        peak_a=PeakDefinition(600, "point", 10, "600"),
        peak_b=PeakDefinition(560, "point", 10, "560"),
        operator1="+",
        peak_c=PeakDefinition(590, "point", 10, "590"),
        operator2="/",
        grouping="left",
        description="(600 + 560) / 590 no baseline",
        category="Bone FTIR",
    )


@pytest.fixture
def bone_multi_sample(bone_wavelengths):
    """Three sample spectra with different peak heights on sloping baselines."""
    n_samples = 3
    data = np.zeros((n_samples, len(bone_wavelengths)))
    for i in range(n_samples):
        slope = 0.5 + (bone_wavelengths - 380) / (800 - 380)
        for center, base_height in [(560, 3.0), (590, 4.0), (600, 3.5)]:
            slope += (i + 1) * base_height * np.exp(
                -0.5 * ((bone_wavelengths - center) / 8) ** 2
            )
        data[i] = slope
    return data


# ---------------------------------------------------------------------------
# 1. BaselineRegion dataclass tests
# ---------------------------------------------------------------------------

def test_baseline_region_defaults():
    """BaselineRegion should initialise with all zeros."""
    bl = BaselineRegion()
    assert bl.left_min == 0.0
    assert bl.left_max == 0.0
    assert bl.right_min == 0.0
    assert bl.right_max == 0.0


def test_baseline_region_custom_values():
    """BaselineRegion should accept and store custom window limits."""
    bl = BaselineRegion(left_min=400, left_max=420, right_min=630, right_max=670)
    assert bl.left_min == 400
    assert bl.left_max == 420
    assert bl.right_min == 630
    assert bl.right_max == 670


def test_peak_definition_with_baseline():
    """PeakDefinition should store an optional BaselineRegion."""
    bl = BaselineRegion(400, 420, 630, 670)
    p = PeakDefinition(wavenumber=604, mode="point", half_width=10, label="604", baseline=bl)
    assert p.baseline is not None
    assert p.baseline.left_min == 400
    assert p.baseline.right_max == 670


def test_peak_definition_baseline_default_is_none():
    """PeakDefinition.baseline should default to None."""
    p = PeakDefinition(wavenumber=604)
    assert p.baseline is None


# ---------------------------------------------------------------------------
# 2. find_trough_in_window() tests
# ---------------------------------------------------------------------------

def test_find_trough_in_window_finds_minimum(bone_wavelengths, bone_spectrum_sloping):
    """find_trough_in_window should locate the minimum within the given window."""
    wn, val = find_trough_in_window(bone_wavelengths, bone_spectrum_sloping, 400, 420)
    assert 400 <= wn <= 420
    # The minimum in this region should be near the low end where the slope is lowest
    # and peaks have not contributed significantly
    mask = (bone_wavelengths >= 400) & (bone_wavelengths <= 420)
    assert val == pytest.approx(np.min(bone_spectrum_sloping[mask]), abs=0.01)


def test_find_trough_in_window_returns_raw_intensity(bone_wavelengths, bone_spectrum_sloping):
    """The returned intensity should be the raw (unsmoothed) value at the trough position."""
    wn, val = find_trough_in_window(bone_wavelengths, bone_spectrum_sloping, 650, 680)
    idx = int(np.argmin(np.abs(bone_wavelengths - wn)))
    assert val == pytest.approx(bone_spectrum_sloping[idx], abs=1e-10)


def test_find_trough_in_window_fallback_no_points():
    """When no data points fall in the window, fall back to the nearest point."""
    wl = np.array([100.0, 200.0, 300.0])
    spec = np.array([5.0, 2.0, 8.0])
    # Window 400-500 has no points; centre = 450 => nearest is 300
    wn, val = find_trough_in_window(wl, spec, 400, 500)
    assert wn == 300.0
    assert val == 8.0


def test_find_trough_in_window_single_point():
    """Window containing a single data point should return that point."""
    wl = np.arange(0, 100, dtype=float)
    spec = np.ones(100) * 10.0
    spec[50] = 3.0
    wn, val = find_trough_in_window(wl, spec, 50, 50)
    assert wn == 50.0
    assert val == 3.0


def test_find_trough_in_window_flat_region():
    """In a flat region all points are equal; should still return a valid result."""
    wl = np.arange(400, 500, dtype=float)
    spec = np.ones_like(wl) * 2.0
    wn, val = find_trough_in_window(wl, spec, 420, 460)
    assert 420 <= wn <= 460
    assert val == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3. baseline_at_wavenumber() tests
# ---------------------------------------------------------------------------

def test_baseline_at_wavenumber_interpolation():
    """Linear interpolation at the midpoint of two anchor points."""
    # Left: (400, 1.0), Right: (600, 2.0), Peak at 500 => 1.5
    result = baseline_at_wavenumber(500, 400, 1.0, 600, 2.0)
    assert result == pytest.approx(1.5)


def test_baseline_at_wavenumber_at_left_anchor():
    """At the left anchor wavenumber the baseline should equal left_val."""
    result = baseline_at_wavenumber(400, 400, 1.0, 600, 2.0)
    assert result == pytest.approx(1.0)


def test_baseline_at_wavenumber_at_right_anchor():
    """At the right anchor wavenumber the baseline should equal right_val."""
    result = baseline_at_wavenumber(600, 400, 1.0, 600, 2.0)
    assert result == pytest.approx(2.0)


def test_baseline_at_wavenumber_coincident_anchors():
    """When both anchors have the same wavenumber, return average of values."""
    result = baseline_at_wavenumber(500, 500, 3.0, 500, 7.0)
    assert result == pytest.approx(5.0)


def test_baseline_at_wavenumber_negative_slope():
    """Baseline with a negative slope should interpolate correctly."""
    # Left: (400, 5.0), Right: (800, 1.0), Peak at 600 => 3.0
    result = baseline_at_wavenumber(600, 400, 5.0, 800, 1.0)
    assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 4. get_baseline_corrected_intensity() tests
# ---------------------------------------------------------------------------

def test_get_baseline_corrected_with_baseline(bone_wavelengths, bone_spectrum_sloping, bone_baseline_region):
    """Corrected intensity should subtract the interpolated baseline from raw peak height."""
    peak_def = PeakDefinition(590, "point", 10, "590", baseline=bone_baseline_region)
    corrected, diag = get_baseline_corrected_intensity(bone_wavelengths, bone_spectrum_sloping, peak_def)
    assert diag is not None
    assert "left_wn" in diag
    assert "right_wn" in diag
    assert "baseline_at_peak" in diag
    assert "raw_intensity" in diag
    assert "found_wn" in diag
    # The corrected value should be raw - baseline
    expected = diag["raw_intensity"] - diag["baseline_at_peak"]
    assert corrected == pytest.approx(expected, abs=1e-10)


def test_get_baseline_corrected_without_baseline(bone_wavelengths, bone_spectrum_sloping):
    """Without a baseline region, return raw intensity and None diagnostics."""
    peak_def = PeakDefinition(590, "point", 10, "590")
    corrected, diag = get_baseline_corrected_intensity(bone_wavelengths, bone_spectrum_sloping, peak_def)
    assert diag is None
    # Should equal raw intensity from get_peak_intensity
    raw = get_peak_intensity(bone_wavelengths, bone_spectrum_sloping, peak_def)
    assert corrected == pytest.approx(raw)


def test_get_baseline_corrected_reduces_sloping_baseline_effect(bone_wavelengths, bone_spectrum_sloping, bone_baseline_region):
    """Corrected peaks on a sloping baseline should differ from raw values."""
    peak_def = PeakDefinition(590, "point", 10, "590", baseline=bone_baseline_region)
    corrected, _diag = get_baseline_corrected_intensity(bone_wavelengths, bone_spectrum_sloping, peak_def)
    raw = get_peak_intensity(bone_wavelengths, bone_spectrum_sloping, peak_def)
    assert corrected < raw, "Baseline correction should reduce the intensity"
    assert corrected > 0, "Peak should still be positive after correction"


def test_get_baseline_corrected_diagnostics_anchor_positions(bone_wavelengths, bone_spectrum_sloping, bone_baseline_region):
    """Diagnostic trough positions should fall within the specified search windows."""
    peak_def = PeakDefinition(590, "point", 10, "590", baseline=bone_baseline_region)
    _, diag = get_baseline_corrected_intensity(bone_wavelengths, bone_spectrum_sloping, peak_def)
    assert 490 <= diag["left_wn"] <= 510
    assert 690 <= diag["right_wn"] <= 750


# ---------------------------------------------------------------------------
# 5. calculate_expression() with use_local_baseline flag
# ---------------------------------------------------------------------------

def test_calculate_expression_with_baseline(
    bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline,
):
    """calculate_expression with use_local_baseline=True should use corrected intensities."""
    result_bl = calculate_expression(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline, use_local_baseline=True,
    )
    result_raw = calculate_expression(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline, use_local_baseline=False,
    )
    # With sloping baseline, the two should differ
    assert result_bl != pytest.approx(result_raw, abs=0.01)
    assert np.isfinite(result_bl)


def test_calculate_expression_without_baseline_flag(
    bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline,
):
    """Setting use_local_baseline=False should ignore baseline regions on peaks."""
    result = calculate_expression(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline, use_local_baseline=False,
    )
    # Should be identical to computing with a no-baseline preset
    no_bl_preset = PeakPreset(
        name="raw",
        peak_a=PeakDefinition(600, "point", 10, "600"),
        peak_b=PeakDefinition(560, "point", 10, "560"),
        operator1="+",
        peak_c=PeakDefinition(590, "point", 10, "590"),
        operator2="/",
        grouping="left",
    )
    result_no_bl = calculate_expression(
        bone_wavelengths, bone_spectrum_sloping, no_bl_preset, use_local_baseline=False,
    )
    assert result == pytest.approx(result_no_bl)


def test_calculate_expression_no_baseline_peaks_unaffected_by_flag(
    bone_wavelengths, bone_spectrum_sloping, bone_preset_no_baseline,
):
    """Peaks without baseline regions give same result regardless of use_local_baseline."""
    result_true = calculate_expression(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_no_baseline, use_local_baseline=True,
    )
    result_false = calculate_expression(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_no_baseline, use_local_baseline=False,
    )
    assert result_true == pytest.approx(result_false)


# ---------------------------------------------------------------------------
# 6. calculate_expression_detailed() tests
# ---------------------------------------------------------------------------

def test_calculate_expression_detailed_returns_diagnostics(
    bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline,
):
    """Detailed result should contain value and per-peak diagnostics."""
    result = calculate_expression_detailed(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline,
    )
    assert "value" in result
    assert np.isfinite(result["value"])
    assert "peak_a_diag" in result
    assert "peak_b_diag" in result
    assert "peak_c_diag" in result
    # All peaks have baselines so all diags should be populated
    for key in ["peak_a_diag", "peak_b_diag", "peak_c_diag"]:
        diag = result[key]
        assert diag is not None
        assert "left_wn" in diag
        assert "right_wn" in diag
        assert "raw_intensity" in diag
        assert "baseline_at_peak" in diag
        assert "found_wn" in diag


def test_calculate_expression_detailed_no_baseline_peaks(
    bone_wavelengths, bone_spectrum_sloping, bone_preset_no_baseline,
):
    """Peaks without baselines should have None diagnostics."""
    result = calculate_expression_detailed(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_no_baseline,
    )
    assert "value" in result
    assert result["peak_a_diag"] is None
    assert result["peak_b_diag"] is None
    assert result["peak_c_diag"] is None


def test_calculate_expression_detailed_two_peak(bone_wavelengths, bone_spectrum_sloping):
    """Two-peak preset should have peak_c_diag = None."""
    preset = PeakPreset(
        name="two peak",
        peak_a=PeakDefinition(600, "point", 10, "600",
                              baseline=BaselineRegion(490, 510, 690, 750)),
        peak_b=PeakDefinition(560, "point", 10, "560",
                              baseline=BaselineRegion(490, 510, 690, 750)),
        operator1="/",
    )
    result = calculate_expression_detailed(bone_wavelengths, bone_spectrum_sloping, preset)
    assert result["peak_c_diag"] is None
    assert result["peak_a_diag"] is not None
    assert result["peak_b_diag"] is not None
    assert np.isfinite(result["value"])


def test_calculate_expression_detailed_matches_calculate_expression(
    bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline,
):
    """Detailed value should equal the regular calculate_expression result."""
    detailed = calculate_expression_detailed(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline,
    )
    simple = calculate_expression(
        bone_wavelengths, bone_spectrum_sloping, bone_preset_with_baseline, use_local_baseline=True,
    )
    assert detailed["value"] == pytest.approx(simple)


# ---------------------------------------------------------------------------
# 7. calculate_all_samples() with baseline regions
# ---------------------------------------------------------------------------

def test_calculate_all_samples_baseline_extra_columns(
    bone_wavelengths, bone_multi_sample, bone_preset_with_baseline,
):
    """When baselines are present, output DataFrame should include diagnostic columns."""
    df = calculate_all_samples(
        bone_wavelengths, bone_multi_sample, bone_preset_with_baseline, use_local_baseline=True,
    )
    assert "Sample" in df.columns
    assert "Value" in df.columns
    # Should have baseline diagnostic columns for peaks A, B, and C
    for label in ["A", "B", "C"]:
        assert f"bl_{label}_left_wn" in df.columns
        assert f"bl_{label}_right_wn" in df.columns
        assert f"bl_{label}_raw" in df.columns
        assert f"bl_{label}_corrected" in df.columns
    assert len(df) == bone_multi_sample.shape[0]


def test_calculate_all_samples_no_baseline_no_extra_columns(
    bone_wavelengths, bone_multi_sample, bone_preset_no_baseline,
):
    """Without baseline regions, output should have only Sample and Value columns."""
    df = calculate_all_samples(
        bone_wavelengths, bone_multi_sample, bone_preset_no_baseline, use_local_baseline=True,
    )
    assert list(df.columns) == ["Sample", "Value"]


def test_calculate_all_samples_baseline_disabled_no_extra_columns(
    bone_wavelengths, bone_multi_sample, bone_preset_with_baseline,
):
    """Disabling local baseline via flag should produce only Sample and Value columns."""
    df = calculate_all_samples(
        bone_wavelengths, bone_multi_sample, bone_preset_with_baseline, use_local_baseline=False,
    )
    assert list(df.columns) == ["Sample", "Value"]


def test_calculate_all_samples_baseline_values_finite(
    bone_wavelengths, bone_multi_sample, bone_preset_with_baseline,
):
    """Baseline-corrected values should be finite (not NaN or Inf)."""
    df = calculate_all_samples(
        bone_wavelengths, bone_multi_sample, bone_preset_with_baseline, use_local_baseline=True,
    )
    values = df["Value"].values
    assert np.all(np.isfinite(values)), "All baseline-corrected values should be finite"
    assert len(values) == bone_multi_sample.shape[0]


def test_calculate_all_samples_baseline_with_names(
    bone_wavelengths, bone_multi_sample, bone_preset_with_baseline,
):
    """Sample names should be propagated when baselines are used."""
    names = ["Bone_A", "Bone_B", "Bone_C"]
    df = calculate_all_samples(
        bone_wavelengths, bone_multi_sample, bone_preset_with_baseline,
        sample_names=names, use_local_baseline=True,
    )
    assert list(df["Sample"]) == names


# ---------------------------------------------------------------------------
# 8. Serialization roundtrip with baseline regions
# ---------------------------------------------------------------------------

def test_peak_def_to_dict_with_baseline():
    """_peak_def_to_dict should include baseline when present."""
    bl = BaselineRegion(400, 420, 630, 670)
    p = PeakDefinition(604, "point", 10, "604", baseline=bl)
    d = _peak_def_to_dict(p)
    assert "baseline" in d
    assert d["baseline"]["left_min"] == 400
    assert d["baseline"]["right_max"] == 670


def test_peak_def_to_dict_without_baseline():
    """_peak_def_to_dict should omit baseline key when None."""
    p = PeakDefinition(604, "point", 10, "604")
    d = _peak_def_to_dict(p)
    assert "baseline" not in d


def test_peak_def_roundtrip_with_baseline():
    """Serialise then deserialise a PeakDefinition with baseline."""
    bl = BaselineRegion(400, 420, 630, 670)
    original = PeakDefinition(604, "point", 10, "v4 PO4", baseline=bl)
    d = _peak_def_to_dict(original)
    restored = _peak_def_from_dict(d)
    assert restored.wavenumber == original.wavenumber
    assert restored.mode == original.mode
    assert restored.half_width == original.half_width
    assert restored.label == original.label
    assert restored.baseline is not None
    assert restored.baseline.left_min == bl.left_min
    assert restored.baseline.left_max == bl.left_max
    assert restored.baseline.right_min == bl.right_min
    assert restored.baseline.right_max == bl.right_max


def test_peak_def_roundtrip_without_baseline():
    """Serialise then deserialise a PeakDefinition without baseline."""
    original = PeakDefinition(1020, "range", 30, "Si-O")
    d = _peak_def_to_dict(original)
    restored = _peak_def_from_dict(d)
    assert restored.wavenumber == original.wavenumber
    assert restored.baseline is None


def test_preset_roundtrip_with_baselines():
    """Full preset with baseline regions should survive a dict roundtrip."""
    bl = BaselineRegion(400, 420, 630, 670)
    preset = PeakPreset(
        name="Crystallinity BL",
        peak_a=PeakDefinition(604, "point", 10, "604", baseline=bl),
        peak_b=PeakDefinition(564, "point", 10, "564", baseline=bl),
        operator1="+",
        peak_c=PeakDefinition(590, "point", 10, "590", baseline=bl),
        operator2="/",
        grouping="left",
        description="test",
        category="Bone FTIR",
    )
    d = _preset_to_dict(preset)
    restored = _preset_from_dict(d)
    assert restored.name == "Crystallinity BL"
    assert restored.peak_a.baseline is not None
    assert restored.peak_a.baseline.left_min == 400
    assert restored.peak_b.baseline is not None
    assert restored.peak_c is not None
    assert restored.peak_c.baseline is not None
    assert restored.peak_c.baseline.right_max == 670


def test_preset_roundtrip_json_with_baselines():
    """Full preset with baselines should survive JSON serialisation."""
    bl = BaselineRegion(880, 900, 1150, 1180)
    preset = PeakPreset(
        name="Acid Phosphate",
        peak_a=PeakDefinition(1128, "point", 10, "HPO4", baseline=bl),
        peak_b=PeakDefinition(1020, "point", 10, "v3 PO4", baseline=bl),
        operator1="/",
        category="Bone FTIR",
    )
    d = _preset_to_dict(preset)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    restored = _preset_from_dict(d2)
    assert restored.peak_a.baseline.left_min == 880
    assert restored.peak_b.baseline.right_max == 1180


def test_save_load_roundtrip_with_baselines(tmp_path, monkeypatch):
    """Saving then loading presets with baselines should reproduce them."""
    monkeypatch.setattr(
        "spectral_predict.peak_calculator._USER_DIR", tmp_path
    )
    monkeypatch.setattr(
        "spectral_predict.peak_calculator._USER_PRESETS_FILE",
        tmp_path / "peak_presets.json",
    )

    bl = BaselineRegion(400, 420, 630, 670)
    presets = [
        PeakPreset(
            name="WithBL",
            peak_a=PeakDefinition(604, "point", 10, "604", baseline=bl),
            peak_b=PeakDefinition(564, "point", 10, "564", baseline=bl),
            operator1="/",
            category="User",
        ),
    ]
    save_user_presets(presets)
    loaded = load_user_presets()
    assert len(loaded) == 1
    assert loaded[0].peak_a.baseline is not None
    assert loaded[0].peak_a.baseline.left_min == 400
    assert loaded[0].peak_b.baseline.right_max == 670


# ---------------------------------------------------------------------------
# 9. Built-in presets with baselines still work correctly
# ---------------------------------------------------------------------------

def test_built_in_presets_with_baselines_have_valid_windows():
    """Every built-in preset peak with a baseline should have left < right windows."""
    for preset in BUILT_IN_PRESETS:
        for peak, label in [(preset.peak_a, "A"), (preset.peak_b, "B")]:
            if peak.baseline is not None:
                bl = peak.baseline
                assert bl.left_min <= bl.left_max, (
                    f"{preset.name} peak {label}: left_min > left_max"
                )
                assert bl.right_min <= bl.right_max, (
                    f"{preset.name} peak {label}: right_min > right_max"
                )
                assert bl.left_max < bl.right_min, (
                    f"{preset.name} peak {label}: left window overlaps right window"
                )
        if preset.peak_c is not None and preset.peak_c.baseline is not None:
            bl = preset.peak_c.baseline
            assert bl.left_min <= bl.left_max
            assert bl.right_min <= bl.right_max
            assert bl.left_max < bl.right_min


def test_built_in_bone_ftir_presets_have_baselines():
    """Most Bone FTIR presets should have baseline regions on at least peak B."""
    bone_presets = [p for p in BUILT_IN_PRESETS if p.category == "Bone FTIR"]
    assert len(bone_presets) > 0
    # At least peak B should always have a baseline (the denominator peak)
    for preset in bone_presets:
        assert preset.peak_b.baseline is not None, (
            f"{preset.name} peak B missing baseline"
        )
    # Most peak A should have baselines too (except isolated peaks like Cyanamide 2010)
    with_bl = [p for p in bone_presets if p.peak_a.baseline is not None]
    assert len(with_bl) >= len(bone_presets) - 2, (
        "Most Bone FTIR presets should have baselines on peak A"
    )


def test_built_in_mineralogical_presets_have_no_baselines():
    """Mineralogical presets should not carry baseline regions (by current design)."""
    mineral_presets = [p for p in BUILT_IN_PRESETS if p.category == "Mineralogical"]
    assert len(mineral_presets) > 0
    for preset in mineral_presets:
        assert preset.peak_a.baseline is None, (
            f"{preset.name} peak A should not have baseline"
        )
        assert preset.peak_b.baseline is None, (
            f"{preset.name} peak B should not have baseline"
        )


# ---------------------------------------------------------------------------
# 10. Backward compatibility — presets without baselines work as before
# ---------------------------------------------------------------------------

def test_backward_compat_two_peak_division(simple_wavelengths, peaked_spectrum):
    """Two-peak preset without baselines should work identically to original behavior."""
    preset = PeakPreset(
        name="Compat",
        peak_a=PeakDefinition(1000, "point", 10, "A"),
        peak_b=PeakDefinition(2000, "point", 10, "B"),
        operator1="/",
    )
    result = calculate_expression(simple_wavelengths, peaked_spectrum, preset, use_local_baseline=True)
    # No baseline on peaks, so use_local_baseline should have no effect
    assert pytest.approx(result, rel=0.01) == 5.0 / 3.0


def test_backward_compat_three_peak_left(simple_wavelengths, peaked_spectrum):
    """Three-peak left grouping without baselines should compute (A + B) / C."""
    preset = PeakPreset(
        name="Compat3",
        peak_a=PeakDefinition(1000, "point", 10, "A"),
        peak_b=PeakDefinition(2000, "point", 10, "B"),
        operator1="+",
        peak_c=PeakDefinition(3000, "point", 10, "C"),
        operator2="/",
        grouping="left",
    )
    result = calculate_expression(simple_wavelengths, peaked_spectrum, preset, use_local_baseline=True)
    assert pytest.approx(result, rel=0.01) == (5.0 + 3.0) / 8.0


def test_backward_compat_calculate_all_samples(simple_wavelengths, multi_sample_data):
    """calculate_all_samples without baselines should return only Sample and Value."""
    preset = PeakPreset(
        name="Compat",
        peak_a=PeakDefinition(1000, "point", 10, "A"),
        peak_b=PeakDefinition(2000, "point", 10, "B"),
        operator1="/",
    )
    df = calculate_all_samples(simple_wavelengths, multi_sample_data, preset, use_local_baseline=True)
    assert list(df.columns) == ["Sample", "Value"]
    assert len(df) == multi_sample_data.shape[0]


def test_backward_compat_detailed_no_diagnostics(simple_wavelengths, peaked_spectrum):
    """calculate_expression_detailed on no-baseline preset returns None diagnostics."""
    preset = PeakPreset(
        name="Compat",
        peak_a=PeakDefinition(1000, "point", 10, "A"),
        peak_b=PeakDefinition(2000, "point", 10, "B"),
        operator1="/",
    )
    result = calculate_expression_detailed(simple_wavelengths, peaked_spectrum, preset)
    assert result["peak_a_diag"] is None
    assert result["peak_b_diag"] is None
    assert result["peak_c_diag"] is None
    assert pytest.approx(result["value"], rel=0.01) == 5.0 / 3.0


def test_backward_compat_peak_def_from_dict_no_baseline_key():
    """_peak_def_from_dict should handle dicts without a 'baseline' key (old format)."""
    d = {"wavenumber": 1020, "mode": "point", "half_width": 10, "label": "test"}
    p = _peak_def_from_dict(d)
    assert p.baseline is None
    assert p.wavenumber == 1020


# ---------------------------------------------------------------------------
# 11. find_peak_in_window() direct tests
# ---------------------------------------------------------------------------

def test_find_peak_in_window_find_max():
    """find_peak_in_window with find_max=True should locate the maximum in the window."""
    wl = np.arange(400, 700, dtype=float)
    spec = np.zeros_like(wl)
    # Place a peak at 550
    spec += 5.0 * np.exp(-0.5 * ((wl - 550) / 5) ** 2)
    wn, val = find_peak_in_window(wl, spec, target_wn=550, half_width=20, find_max=True)
    assert wn == pytest.approx(550.0, abs=1.0)
    assert val == pytest.approx(5.0, abs=0.1)


def test_find_peak_in_window_find_min():
    """find_peak_in_window with find_max=False should locate the minimum in the window."""
    wl = np.arange(400, 700, dtype=float)
    # Create a baseline of 5.0 with a dip at 520
    spec = np.ones_like(wl) * 5.0
    spec -= 3.0 * np.exp(-0.5 * ((wl - 520) / 5) ** 2)
    wn, val = find_peak_in_window(wl, spec, target_wn=520, half_width=20, find_max=False)
    assert wn == pytest.approx(520.0, abs=1.0)
    assert val == pytest.approx(2.0, abs=0.1)


def test_find_peak_in_window_fallback_no_data():
    """When no points fall in window, find_peak_in_window falls back to nearest point."""
    wl = np.array([100.0, 200.0, 300.0])
    spec = np.array([5.0, 2.0, 8.0])
    wn, val = find_peak_in_window(wl, spec, target_wn=450, half_width=10, find_max=True)
    # Nearest to 450 is 300
    assert wn == 300.0
    assert val == 8.0


# ---------------------------------------------------------------------------
# 12. Baseline interpolation fix: uses found_wn, not nominal
# ---------------------------------------------------------------------------

def test_baseline_interpolation_uses_found_wn():
    """Baseline should be interpolated at the actual found peak position, not the nominal.

    A peak nominally at 600 cm-1 is placed at 603 cm-1 in the data.
    With search_max mode, the baseline should be evaluated at 603, not 600.
    """
    wl = np.arange(400, 800, dtype=float)
    # Create a sloping baseline
    baseline = 1.0 + (wl - 400) / (800 - 400)  # 1.0 -> 2.0
    # Add a peak at 603 (not at the nominal 600)
    peak = 5.0 * np.exp(-0.5 * ((wl - 603) / 5) ** 2)
    spec = baseline + peak

    bl_region = BaselineRegion(490, 510, 690, 750)
    peak_def = PeakDefinition(
        wavenumber=600, mode="search_max", half_width=10,
        label="test", baseline=bl_region,
    )
    _corrected, diag = get_baseline_corrected_intensity(wl, spec, peak_def)

    assert diag is not None
    # The found wavenumber should be 603 (the actual peak), not 600 (nominal)
    assert diag["found_wn"] == pytest.approx(603.0, abs=1.0)
    # The baseline_at_peak should be evaluated at 603, not 600
    expected_bl_at_603 = baseline_at_wavenumber(
        603.0, diag["left_wn"], diag["left_val"],
        diag["right_wn"], diag["right_val"],
    )
    assert diag["baseline_at_peak"] == pytest.approx(expected_bl_at_603, abs=0.01)


# ---------------------------------------------------------------------------
# 13. search_max and search_min through get_peak_intensity
# ---------------------------------------------------------------------------

def test_get_peak_intensity_search_max():
    """search_max mode should find the actual local maximum, not the value at nominal."""
    wl = np.arange(400, 700, dtype=float)
    spec = np.zeros_like(wl)
    # Place peak at 553 instead of nominal 550
    spec += 7.0 * np.exp(-0.5 * ((wl - 553) / 5) ** 2)
    peak_def = PeakDefinition(wavenumber=550, mode="search_max", half_width=10)
    val = get_peak_intensity(wl, spec, peak_def)
    # Should return intensity at 553 (the actual max), not 550
    assert val == pytest.approx(7.0, abs=0.1)


def test_get_peak_intensity_search_min():
    """search_min mode should find the actual local minimum, not the value at nominal."""
    wl = np.arange(400, 700, dtype=float)
    # Constant baseline with a dip at 588 instead of nominal 590
    spec = np.ones_like(wl) * 5.0
    spec -= 3.0 * np.exp(-0.5 * ((wl - 588) / 5) ** 2)
    peak_def = PeakDefinition(wavenumber=590, mode="search_min", half_width=10)
    val = get_peak_intensity(wl, spec, peak_def)
    # Should return the trough value at 588 (~2.0), not the value at 590
    assert val == pytest.approx(2.0, abs=0.15)


def test_search_max_calls_find_peak_in_window():
    """search_max mode in get_peak_intensity should delegate to find_peak_in_window."""
    wl = np.arange(400, 700, dtype=float)
    spec = np.zeros_like(wl)
    spec += 4.0 * np.exp(-0.5 * ((wl - 510) / 3) ** 2)
    peak_def = PeakDefinition(wavenumber=510, mode="search_max", half_width=10)
    val = get_peak_intensity(wl, spec, peak_def)
    # Cross-check with find_peak_in_window directly
    _, expected = find_peak_in_window(wl, spec, 510, 10, find_max=True)
    assert val == pytest.approx(expected)


def test_search_min_calls_find_peak_in_window():
    """search_min mode in get_peak_intensity should delegate to find_peak_in_window."""
    wl = np.arange(400, 700, dtype=float)
    spec = np.ones_like(wl) * 10.0
    spec -= 6.0 * np.exp(-0.5 * ((wl - 510) / 3) ** 2)
    peak_def = PeakDefinition(wavenumber=510, mode="search_min", half_width=10)
    val = get_peak_intensity(wl, spec, peak_def)
    _, expected = find_peak_in_window(wl, spec, 510, 10, find_max=False)
    assert val == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 14. Descending wavenumber array (FTIR compatibility)
# ---------------------------------------------------------------------------

def test_find_peak_in_window_descending_wavenumbers():
    """find_peak_in_window should work with descending (FTIR-typical) wavenumber arrays."""
    wl = np.flip(np.arange(400, 700, dtype=float))  # 699, 698, ..., 400
    spec_asc = np.zeros(len(wl))
    # Place peak at 550 (index position will differ from ascending)
    for i, w in enumerate(wl):
        spec_asc[i] = 5.0 * np.exp(-0.5 * ((w - 550) / 5) ** 2)
    wn, val = find_peak_in_window(wl, spec_asc, target_wn=550, half_width=20, find_max=True)
    assert wn == pytest.approx(550.0, abs=1.0)
    assert val == pytest.approx(5.0, abs=0.1)


def test_find_trough_in_window_descending_wavenumbers():
    """find_trough_in_window should work with descending wavenumber arrays."""
    wl = np.flip(np.arange(400, 700, dtype=float))  # 699 -> 400
    spec = np.ones(len(wl)) * 5.0
    for i, w in enumerate(wl):
        spec[i] -= 3.0 * np.exp(-0.5 * ((w - 500) / 5) ** 2)
    wn, val = find_trough_in_window(wl, spec, 490, 510)
    assert wn == pytest.approx(500.0, abs=1.0)
    assert val == pytest.approx(2.0, abs=0.1)


def test_get_peak_intensity_descending_wavenumbers():
    """get_peak_intensity should work correctly with descending wavenumber arrays."""
    wl = np.flip(np.arange(400, 700, dtype=float))
    spec = np.zeros(len(wl))
    for i, w in enumerate(wl):
        spec[i] = 4.0 * np.exp(-0.5 * ((w - 550) / 5) ** 2)

    # Point mode
    peak_def = PeakDefinition(wavenumber=550, mode="point")
    val = get_peak_intensity(wl, spec, peak_def)
    assert val == pytest.approx(4.0, abs=0.1)

    # search_max mode
    peak_def_sm = PeakDefinition(wavenumber=550, mode="search_max", half_width=10)
    val_sm = get_peak_intensity(wl, spec, peak_def_sm)
    assert val_sm == pytest.approx(4.0, abs=0.1)


# ---------------------------------------------------------------------------
# 15. WAMPI and AmI/AmII presets exist in BUILT_IN_PRESETS
# ---------------------------------------------------------------------------

def test_built_in_presets_contain_wampi():
    """BUILT_IN_PRESETS should include the WAMPI preset."""
    names = {p.name for p in BUILT_IN_PRESETS}
    assert "WAMPI" in names


def test_built_in_presets_contain_ami_amii():
    """BUILT_IN_PRESETS should include the AmI/AmII preset."""
    names = {p.name for p in BUILT_IN_PRESETS}
    assert "AmI/AmII" in names


def test_wampi_preset_has_expected_structure():
    """WAMPI preset should have search_max peaks with baselines."""
    wampi = next(p for p in BUILT_IN_PRESETS if p.name == "WAMPI")
    assert wampi.category == "Bone FTIR"
    assert wampi.peak_a.mode == "search_max"
    assert wampi.peak_a.baseline is not None
    assert wampi.peak_b.baseline is not None


def test_ami_amii_preset_has_expected_structure():
    """AmI/AmII preset should have search_max peaks with baselines."""
    ami = next(p for p in BUILT_IN_PRESETS if p.name == "AmI/AmII")
    assert ami.category == "Bone FTIR"
    assert ami.peak_a.mode == "search_max"
    assert ami.peak_b.mode == "search_max"
    assert ami.peak_a.baseline is not None
    assert ami.peak_b.baseline is not None


# ---------------------------------------------------------------------------
# 16. SG guard fix: smoothing triggers when subset >= 5
# ---------------------------------------------------------------------------

def test_sg_smoothing_triggers_with_6_points():
    """SG smoothing should be applied when the window has 6-8 data points (>= 5)."""
    # Create a small wavenumber array with only 7 points in the search window
    wl = np.array([495.0, 497.0, 499.0, 501.0, 503.0, 505.0, 507.0])
    # Add a noisy peak at 501
    spec = np.array([1.0, 1.2, 2.0, 5.0, 2.5, 1.1, 0.9])
    # Window [496, 506] covers 5 points: 497, 499, 501, 503, 505
    wn, val = find_peak_in_window(wl, spec, target_wn=501, half_width=5, find_max=True)
    # Should find the peak at 501 (the maximum)
    assert wn == pytest.approx(501.0, abs=2.0)
    # The raw value returned should be the actual spectrum value, not smoothed
    idx = int(np.argmin(np.abs(wl - wn)))
    assert val == pytest.approx(spec[idx], abs=1e-10)


def test_sg_smoothing_with_exactly_5_points():
    """SG smoothing should apply when there are exactly 5 data points in window."""
    wl = np.array([498.0, 499.0, 500.0, 501.0, 502.0])
    spec = np.array([1.0, 2.0, 5.0, 2.5, 1.0])
    wn, val = find_peak_in_window(wl, spec, target_wn=500, half_width=3, find_max=True)
    assert wn == pytest.approx(500.0, abs=1.0)
    assert val == pytest.approx(5.0, abs=0.1)


def test_sg_smoothing_not_applied_with_4_points():
    """SG smoothing should NOT be applied when there are fewer than 5 data points."""
    wl = np.array([499.0, 500.0, 501.0, 502.0])
    spec = np.array([1.0, 5.0, 2.0, 1.0])
    wn, val = find_peak_in_window(wl, spec, target_wn=500, half_width=2, find_max=True)
    # Without smoothing, should still find the raw max at 500
    assert wn == pytest.approx(500.0)
    assert val == pytest.approx(5.0)
