"""Comprehensive tests for spectral_predict.peak_calculator module."""

import json

import numpy as np
import pandas as pd
import pytest

from spectral_predict.peak_calculator import (
    BUILT_IN_PRESETS,
    PeakDefinition,
    PeakPreset,
    calculate_all_samples,
    calculate_expression,
    get_peak_intensity,
    load_user_presets,
    save_user_presets,
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


def test_get_peak_intensity_range_mode_fallback_to_nearest(simple_wavelengths, flat_spectrum):
    """Range mode should fall back to nearest point when no wavelength in range."""
    # Request a range entirely outside the axis (centred at 100, which is below 400)
    peak_def = PeakDefinition(wavenumber=100, mode="range", half_width=5)
    val = get_peak_intensity(simple_wavelengths, flat_spectrum, peak_def)
    # Should snap to 400 (nearest) and return 1.0
    assert val == 1.0


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
