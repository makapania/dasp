"""Validation tests for peak_calculator against synthetic bone FTIR spectra.

Generates realistic bone FTIR spectra with known peak positions and heights,
then verifies that calculated indices fall within published reference ranges.

Run with:  PYTHONPATH=src pytest tests/test_bone_validation.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spectral_predict.peak_calculator import (
    BUILT_IN_PRESETS,
    PeakPreset,
    calculate_expression,
    calculate_expression_detailed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, center: float, height: float, fwhm: float) -> np.ndarray:
    """Return a Gaussian peak with given center, height, and FWHM."""
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _get_preset(name: str) -> PeakPreset:
    """Look up a built-in preset by name (case-sensitive)."""
    for p in BUILT_IN_PRESETS:
        if p.name == name:
            return p
    raise KeyError(f"Preset {name!r} not found in BUILT_IN_PRESETS")


# ---------------------------------------------------------------------------
# Spectrum generators
# ---------------------------------------------------------------------------

def make_modern_bone_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic modern bone FTIR spectrum (400-4000 cm-1, 2 cm-1 spacing).

    Peak parameters approximate a fresh/modern bone sample with moderate
    crystallinity and intact collagen.
    """
    wn = np.arange(400, 4002, 2, dtype=float)

    # Sloping baseline: 0.1 at 400 cm-1 to 0.05 at 4000 cm-1
    baseline = 0.1 + (0.05 - 0.1) * (wn - 400) / (4000 - 400)

    spectrum = baseline.copy()

    # v4 PO4 doublet — moderate width for modern bone
    # Wider FWHM creates more overlap, filling the trough for realistic IRSF
    spectrum += _gaussian(wn, center=565, height=0.40, fwhm=30)
    spectrum += _gaussian(wn, center=603, height=0.35, fwhm=28)

    # The trough at ~590 cm-1 is naturally formed by the gap between the
    # two overlapping Gaussians. No negative dip needed — the overlap
    # already creates a valley between the peaks.

    # v3 PO4 broad peak
    spectrum += _gaussian(wn, center=1035, height=1.50, fwhm=60)

    # v3 CO3 peak
    spectrum += _gaussian(wn, center=1415, height=0.15, fwhm=30)

    # Amide I peak
    spectrum += _gaussian(wn, center=1650, height=0.30, fwhm=35)

    # Amide II peak
    spectrum += _gaussian(wn, center=1540, height=0.20, fwhm=30)

    return wn, spectrum


def make_burnt_bone_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic burnt/calcined bone with sharper v4 PO4 peaks (IRSF > 4).

    Burning narrows the phosphate peaks and increases crystallinity.  The
    organic (amide) bands are strongly reduced.
    """
    wn = np.arange(400, 4002, 2, dtype=float)

    baseline = 0.08 + (0.03 - 0.08) * (wn - 400) / (4000 - 400)
    spectrum = baseline.copy()

    # Sharper v4 PO4 peaks (narrower FWHM = higher crystallinity)
    # Sharp peaks with less overlap naturally produce a deeper trough
    spectrum += _gaussian(wn, center=565, height=0.55, fwhm=10)
    spectrum += _gaussian(wn, center=603, height=0.50, fwhm=10)

    # v3 PO4
    spectrum += _gaussian(wn, center=1035, height=1.80, fwhm=50)

    # Reduced carbonate
    spectrum += _gaussian(wn, center=1415, height=0.08, fwhm=25)

    # Severely diminished amide bands (collagen destroyed)
    spectrum += _gaussian(wn, center=1650, height=0.04, fwhm=30)
    spectrum += _gaussian(wn, center=1540, height=0.03, fwhm=25)

    return wn, spectrum


def make_steep_baseline_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Spectrum with a steep linear slope to test baseline interpolation.

    The steep slope means that the baseline value at the *found* wavenumber
    differs noticeably from the baseline at the *nominal* wavenumber,
    exercising the found_wn baseline interpolation fix.
    """
    wn = np.arange(400, 4002, 2, dtype=float)

    # Very steep slope: 0.8 at 400 cm-1 falling to 0.05 at 4000 cm-1
    baseline = 0.8 + (0.05 - 0.8) * (wn - 400) / (4000 - 400)
    spectrum = baseline.copy()

    # v4 PO4 peaks on the steep slope
    spectrum += _gaussian(wn, center=565, height=0.40, fwhm=20)
    spectrum += _gaussian(wn, center=603, height=0.35, fwhm=18)

    # v3 PO4
    spectrum += _gaussian(wn, center=1035, height=1.50, fwhm=60)

    # v3 CO3
    spectrum += _gaussian(wn, center=1415, height=0.15, fwhm=30)

    # Amide I
    spectrum += _gaussian(wn, center=1650, height=0.30, fwhm=35)

    # Amide II
    spectrum += _gaussian(wn, center=1540, height=0.20, fwhm=30)

    return wn, spectrum


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def modern_bone():
    """Return (wavenumbers, spectrum) for a modern bone sample."""
    return make_modern_bone_spectrum()


@pytest.fixture
def burnt_bone():
    """Return (wavenumbers, spectrum) for a burnt/calcined bone sample."""
    return make_burnt_bone_spectrum()


@pytest.fixture
def steep_baseline():
    """Return (wavenumbers, spectrum) with a steep linear baseline."""
    return make_steep_baseline_spectrum()


# ---------------------------------------------------------------------------
# Tests — Modern bone index ranges
# ---------------------------------------------------------------------------

class TestModernBoneIndices:
    """Verify that calculated indices on modern bone fall in published ranges."""

    def test_irsf_in_modern_range(self, modern_bone):
        """IRSF (Crystallinity Index) should be ~2.5-3.5 for modern bone."""
        wn, spec = modern_bone
        preset = _get_preset("Crystallinity Index")
        irsf = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert 2.5 <= irsf <= 3.5, f"IRSF = {irsf:.3f}, expected 2.5-3.5 for modern bone"

    def test_cp_in_reasonable_range(self, modern_bone):
        """C/P ratio should be ~0.05-0.3 for bone."""
        wn, spec = modern_bone
        preset = _get_preset("Carbonate:Phosphate")
        cp = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert 0.05 <= cp <= 0.3, f"C/P = {cp:.4f}, expected 0.05-0.3"

    def test_mineral_matrix_positive(self, modern_bone):
        """Mineral:Matrix should be finite and positive."""
        wn, spec = modern_bone
        preset = _get_preset("Mineral:Matrix")
        mm = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert np.isfinite(mm), "Mineral:Matrix is not finite"
        assert mm > 0, f"Mineral:Matrix = {mm:.4f}, expected > 0"

    def test_bpi_positive_and_finite(self, modern_bone):
        """BPI (B-Type Carbonate) should be finite and positive."""
        wn, spec = modern_bone
        preset = _get_preset("B-Type Carbonate (BPI)")
        bpi = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert np.isfinite(bpi), "BPI is not finite"
        assert bpi > 0, f"BPI = {bpi:.4f}, expected > 0"

    def test_amp_positive_and_finite(self, modern_bone):
        """Amide:Phosphate should be finite and positive."""
        wn, spec = modern_bone
        preset = _get_preset("Amide:Phosphate")
        amp = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert np.isfinite(amp), "Am/P is not finite"
        assert amp > 0, f"Am/P = {amp:.4f}, expected > 0"

    def test_wampi_positive_and_finite(self, modern_bone):
        """WAMPI should be finite and positive."""
        wn, spec = modern_bone
        preset = _get_preset("WAMPI")
        wampi = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert np.isfinite(wampi), "WAMPI is not finite"
        assert wampi > 0, f"WAMPI = {wampi:.4f}, expected > 0"

    def test_ami_amii_positive_and_finite(self, modern_bone):
        """AmI/AmII should be finite and positive."""
        wn, spec = modern_bone
        preset = _get_preset("AmI/AmII")
        ratio = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert np.isfinite(ratio), "AmI/AmII is not finite"
        assert ratio > 0, f"AmI/AmII = {ratio:.4f}, expected > 0"


# ---------------------------------------------------------------------------
# Tests — All bone indices are finite and positive
# ---------------------------------------------------------------------------

class TestAllBoneIndicesFinite:
    """Sanity check: every Bone FTIR preset produces a finite positive value."""

    @pytest.fixture
    def bone_presets(self):
        return [p for p in BUILT_IN_PRESETS if p.category == "Bone FTIR"]

    def test_all_finite_positive(self, modern_bone, bone_presets):
        wn, spec = modern_bone
        for preset in bone_presets:
            val = calculate_expression(wn, spec, preset, use_local_baseline=True)
            assert np.isfinite(val), f"{preset.name}: value is not finite ({val})"
            assert val > 0, f"{preset.name}: value = {val:.6f}, expected > 0"


# ---------------------------------------------------------------------------
# Tests — Baseline correction changes values
# ---------------------------------------------------------------------------

class TestBaselineCorrectionEffect:
    """Verify that toggling use_local_baseline actually changes the result."""

    @pytest.mark.parametrize("preset_name", [
        "Crystallinity Index",
        "Carbonate:Phosphate",
        "Mineral:Matrix",
        "Amide:Phosphate",
        "WAMPI",
        "AmI/AmII",
        "B-Type Carbonate (BPI)",
    ])
    def test_baseline_changes_value(self, modern_bone, preset_name):
        """With-baseline and without-baseline should differ for presets that
        have BaselineRegion definitions."""
        wn, spec = modern_bone
        preset = _get_preset(preset_name)

        val_bl = calculate_expression(wn, spec, preset, use_local_baseline=True)
        val_raw = calculate_expression(wn, spec, preset, use_local_baseline=False)

        assert val_bl != pytest.approx(val_raw, rel=1e-6), (
            f"{preset_name}: baseline correction had no effect "
            f"(bl={val_bl:.6f}, raw={val_raw:.6f})"
        )


# ---------------------------------------------------------------------------
# Tests — Diagnostics: found_wn close to but not exactly nominal
# ---------------------------------------------------------------------------

class TestDiagnosticsFoundWn:
    """Verify that search_max/search_min finds wavenumbers near the nominal."""

    @pytest.mark.parametrize("preset_name,peak_key,nominal_wn,tolerance", [
        ("Crystallinity Index", "peak_a_diag", 604, 6),
        ("Crystallinity Index", "peak_b_diag", 564, 6),
        ("Crystallinity Index", "peak_c_diag", 590, 6),
        ("Carbonate:Phosphate", "peak_a_diag", 1415, 10),
        ("Mineral:Matrix", "peak_a_diag", 1035, 15),
    ])
    def test_found_wn_near_nominal(self, modern_bone, preset_name, peak_key,
                                   nominal_wn, tolerance):
        """The found_wn in diagnostics should be close to but not necessarily
        exactly equal to the nominal wavenumber."""
        wn, spec = modern_bone
        preset = _get_preset(preset_name)
        result = calculate_expression_detailed(wn, spec, preset)
        diag = result[peak_key]

        assert diag is not None, f"No diagnostics for {peak_key} in {preset_name}"
        found = diag["found_wn"]

        assert abs(found - nominal_wn) <= tolerance, (
            f"{preset_name} {peak_key}: found_wn={found:.1f}, "
            f"expected within {tolerance} cm-1 of {nominal_wn}"
        )
        # For a synthetic spectrum the peak may land exactly on a grid point
        # that differs from the nominal, so we just check it is close.


# ---------------------------------------------------------------------------
# Tests — Burnt bone: high crystallinity
# ---------------------------------------------------------------------------

class TestBurntBone:
    """Burnt/calcined bone should have much higher IRSF (> 4.0)."""

    def test_irsf_above_four(self, burnt_bone):
        """IRSF for burnt bone should exceed 4.0 due to sharp v4 PO4 peaks."""
        wn, spec = burnt_bone
        preset = _get_preset("Crystallinity Index")
        irsf = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert irsf > 4.0, f"Burnt bone IRSF = {irsf:.3f}, expected > 4.0"

    def test_burnt_irsf_higher_than_modern(self, modern_bone, burnt_bone):
        """Burnt bone IRSF must be strictly higher than modern bone IRSF."""
        preset = _get_preset("Crystallinity Index")
        irsf_modern = calculate_expression(*modern_bone, preset, use_local_baseline=True)
        irsf_burnt = calculate_expression(*burnt_bone, preset, use_local_baseline=True)
        assert irsf_burnt > irsf_modern, (
            f"Burnt IRSF ({irsf_burnt:.3f}) should exceed modern ({irsf_modern:.3f})"
        )

    def test_burnt_mineral_matrix_higher(self, modern_bone, burnt_bone):
        """Burnt bone Mineral:Matrix should be much higher (collagen lost)."""
        preset = _get_preset("Mineral:Matrix")
        mm_modern = calculate_expression(*modern_bone, preset, use_local_baseline=True)
        mm_burnt = calculate_expression(*burnt_bone, preset, use_local_baseline=True)
        assert mm_burnt > mm_modern, (
            f"Burnt Mineral:Matrix ({mm_burnt:.3f}) should exceed modern ({mm_modern:.3f})"
        )

    def test_burnt_all_finite(self, burnt_bone):
        """All Bone FTIR presets should still be finite on burnt bone."""
        wn, spec = burnt_bone
        bone_presets = [p for p in BUILT_IN_PRESETS if p.category == "Bone FTIR"]
        for preset in bone_presets:
            val = calculate_expression(wn, spec, preset, use_local_baseline=True)
            assert np.isfinite(val), f"{preset.name}: not finite on burnt bone ({val})"


# ---------------------------------------------------------------------------
# Tests — Steep baseline: found_wn matters for interpolation
# ---------------------------------------------------------------------------

class TestSteepBaseline:
    """On a steep slope, baseline interpolation at found_wn vs nominal_wn
    should produce different baseline values, confirming the fix works."""

    def test_irsf_finite_on_steep_slope(self, steep_baseline):
        """IRSF should still be computable on a steep-slope spectrum."""
        wn, spec = steep_baseline
        preset = _get_preset("Crystallinity Index")
        irsf = calculate_expression(wn, spec, preset, use_local_baseline=True)
        assert np.isfinite(irsf), f"IRSF not finite on steep slope ({irsf})"
        assert irsf > 0, f"IRSF = {irsf:.3f}, expected positive"

    def test_baseline_at_found_vs_nominal_differs(self, steep_baseline):
        """On a steep slope the baseline value interpolated at found_wn should
        differ from what you would get at the nominal wavenumber, unless the
        found_wn happens to coincide exactly with the nominal."""
        wn, spec = steep_baseline
        preset = _get_preset("Crystallinity Index")
        result = calculate_expression_detailed(wn, spec, preset)

        # Check peak_a (nominal 604): the steep slope means even a 2 cm-1
        # shift changes the baseline noticeably
        diag_a = result["peak_a_diag"]
        assert diag_a is not None

        found_wn = diag_a["found_wn"]
        bl_at_found = diag_a["baseline_at_peak"]

        # Manually compute what the baseline would be at the nominal 604
        from spectral_predict.peak_calculator import baseline_at_wavenumber
        bl_at_nominal = baseline_at_wavenumber(
            604.0,
            diag_a["left_wn"], diag_a["left_val"],
            diag_a["right_wn"], diag_a["right_val"],
        )

        # If found_wn != 604.0, the baselines must differ
        if found_wn != 604.0:
            assert bl_at_found != pytest.approx(bl_at_nominal, abs=1e-9), (
                f"Baselines should differ: found_wn={found_wn}, "
                f"bl_at_found={bl_at_found:.6f}, bl_at_nominal={bl_at_nominal:.6f}"
            )

    def test_steep_all_bone_indices_finite(self, steep_baseline):
        """All Bone FTIR presets remain finite even with a steep slope."""
        wn, spec = steep_baseline
        bone_presets = [p for p in BUILT_IN_PRESETS if p.category == "Bone FTIR"]
        for preset in bone_presets:
            val = calculate_expression(wn, spec, preset, use_local_baseline=True)
            assert np.isfinite(val), f"{preset.name}: not finite on steep slope ({val})"
