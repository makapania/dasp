"""Noise maps and detector joins measured from spectra."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import savgol_filter

from spectral_predict import spectral_diagnostics as sd

REPO = Path(__file__).resolve().parents[1]
WL = np.arange(350.0, 2501.0)


def _smooth_spectra(m: int = 30, seed: int = 0) -> np.ndarray:
    """Smooth absorbance-like spectra with broad bands and sample variation."""
    rng = np.random.default_rng(seed)
    x = (WL - WL[0]) / (WL[-1] - WL[0])
    base = 0.4 + 0.2 * np.sin(5 * x)
    bands = sum(
        a * np.exp(-0.5 * ((WL - c) / w) ** 2)
        for c, w, a in [(1450, 30, 0.2), (1930, 35, 0.3), (2100, 25, 0.1), (2300, 20, 0.08)]
    )
    return base + bands * rng.uniform(0.6, 1.4, (m, 1)) + rng.normal(0, 0.01, (m, 1))


def _interp_noise(rng, m: int, native: float, sd_: float) -> np.ndarray:
    grid = np.arange(WL[0], WL[-1] + native, native)
    e = rng.normal(0, sd_, (m, grid.size))
    return np.array([np.interp(WL, grid, r) for r in e])


@pytest.mark.parametrize("kind", ["white", "box3", "interp2"])
def test_noise_estimates_match_known_noise(kind):
    rng = np.random.default_rng(1)
    A = _smooth_spectra()
    if kind == "white":
        e = rng.normal(0, 1e-3, A.shape)
    elif kind == "box3":
        e = savgol_filter(rng.normal(0, 1e-3, A.shape), 3, 0, axis=1) * np.sqrt(3)
    else:
        e = _interp_noise(rng, A.shape[0], 2.0, 1e-3)
    ne = sd.estimate_noise(A + e, WL)
    true_d2 = sd.second_derivative(e, WL).std(axis=0)
    inner = slice(100, -100)
    ratio = np.median(ne.d2_sd[inner] / true_d2[inner])
    assert 0.7 <= ratio <= 1.3
    assert 0.7 <= np.median(ne.channel_sd[inner]) / e.std() <= 1.3


def test_white_noise_formula_would_underestimate_interpolated_noise():
    """Why the correlation kernel exists: on 2 nm-native noise interpolated to 1 nm the
    derivative noise from a white-noise formula is ~0.45x the truth."""
    rng = np.random.default_rng(2)
    A = _smooth_spectra()
    e = _interp_noise(rng, A.shape[0], 2.0, 1e-3)
    r = (A + e) - savgol_filter(A + e, 11, 2, axis=1)
    from scipy.signal import savgol_coeffs

    c0 = savgol_coeffs(11, 2)[5]
    white = 1.4826 * np.median(np.abs(r), axis=0) / np.sqrt(1 - c0)
    white_d2 = white * np.sqrt((savgol_coeffs(31, 3, deriv=2) ** 2).sum())
    true_d2 = sd.second_derivative(e, WL).std(axis=0)
    assert np.median(white_d2[100:-100] / true_d2[100:-100]) < 0.6


def test_noisy_regions_flag_a_ramp_per_block():
    rng = np.random.default_rng(3)
    A = _smooth_spectra()
    level = np.where(WL > 1800, 4e-4, 1e-4)  # SWIR2 inherently noisier: not flagged
    level = np.where(WL > 2380, 4e-4 * np.exp((WL - 2380) / 25), level)  # 10x by ~2440
    A = A + rng.normal(0, 1, A.shape) * level
    prof = sd.noise_profile(A, WL, space="absorbance", joins=(1000, 1800))
    regions = sd.noisy_regions(prof, WL, (1000, 1800))
    assert len(regions) == 1
    lo, hi = regions[0]
    assert 2400 < lo < 2460 and hi == 2500
    assert not sd.regions_mask(regions, WL)[(WL > 1850) & (WL < 2350)].any()


def test_noise_comparison_floors_quiet_primary():
    rng = np.random.default_rng(4)
    A = _smooth_spectra()
    Ap = A + rng.normal(0, 1e-5, A.shape)
    As = A + rng.normal(0, 1e-5, A.shape) * np.where(WL > 2000, 30, 1)
    comp = sd.noise_comparison(Ap, As, WL, space="absorbance", joins=(1000, 1800))
    assert np.isfinite(comp.ratio).all()
    assert any(lo <= 2100 and hi >= 2400 for lo, hi in comp.regions)
    assert not any(hi < 1900 for lo, hi in comp.regions)


def _with_steps(R: np.ndarray, rng, joins=(1000, 1800), size=0.01) -> np.ndarray:
    R = R.copy()
    for j in joins:
        R[:, WL > j] += rng.normal(size, size / 3, (R.shape[0], 1))
    return R


def test_detects_steps_at_joins():
    rng = np.random.default_rng(5)
    R = 10 ** -_smooth_spectra() + rng.normal(0, 2e-4, (30, WL.size))
    det = sd.detect_detector_joins(_with_steps(R, rng), WL, space="reflectance")
    assert det.confident and det.joins == (1000.0, 1800.0) and det.source == "detected"


def test_no_joins_in_smooth_spectra():
    rng = np.random.default_rng(6)
    R = 10 ** -_smooth_spectra() + rng.normal(0, 2e-4, (30, WL.size))
    det = sd.detect_detector_joins(R, WL, space="reflectance")
    assert not det.confident and det.joins == () and det.source == "none"


def test_sharp_absorption_edge_and_bad_channel_are_not_joins():
    rng = np.random.default_rng(7)
    A = _smooth_spectra()
    A = A + 0.3 / (1 + np.exp(-(WL - 1300) / 2.0))  # steep but smooth edge (~8 nm wide)
    R = 10**-A + rng.normal(0, 2e-4, A.shape)
    R[:, WL == 1600] += 0.02  # one bad channel in every spectrum
    det = sd.detect_detector_joins(R, WL, space="reflectance")
    assert 1300.0 not in det.joins and 1600.0 not in det.joins and 1599.0 not in det.joins


def test_example_asd_files_show_joins_in_data_and_header():
    from spectral_predict.io import read_asd_dir

    X, meta = read_asd_dir(str(REPO / "example"))
    wl = X.columns.astype(float).to_numpy()
    det = sd.detect_detector_joins(X.to_numpy(float), wl, space="reflectance")
    assert det.joins == (1000.0, 1800.0)
    assert meta["detector_joins"] == (1000.0, 1800.0)
    assert sd.read_asd_header_joins(REPO / "example" / "Spectrum00001.asd") == (1000.0, 1800.0)


def _fake_header(path: Path, magic: bytes, s1: float, s2: float) -> Path:
    raw = bytearray(484)
    raw[:3] = magic
    struct.pack_into("<f", raw, 191, 350.0)
    struct.pack_into("<f", raw, 195, 1.0)
    struct.pack_into("<H", raw, 204, 2151)
    struct.pack_into("<f", raw, 444, s1)
    struct.pack_into("<f", raw, 448, s2)
    path.write_bytes(bytes(raw) + bytes(4 * 2151))
    return path


def test_header_joins_plausibility(tmp_path):
    assert sd.read_asd_header_joins(_fake_header(tmp_path / "a.asd", b"as7", 0, 0)) is None
    assert sd.read_asd_header_joins(_fake_header(tmp_path / "b.asd", b"as8", 1800, 1000)) is None
    assert sd.read_asd_header_joins(_fake_header(tmp_path / "c.asd", b"as8", 360, 1800)) is None
    assert sd.read_asd_header_joins(_fake_header(tmp_path / "d.asd", b"ASD", 1000, 1830)) == (
        1000.0,
        1830.0,
    )
    (tmp_path / "e.txt").write_text("wavelength,reflectance\n" * 40)
    assert sd.read_asd_header_joins(tmp_path / "e.txt") is None
    # An ASCII export may start with "ASD": no NUL after the magic, so not a header.
    text = _fake_header(tmp_path / "f.asd", b"ASD", 1000, 1800).read_bytes()
    (tmp_path / "f.asd").write_bytes(b"ASD " + text[4:])
    assert sd.read_asd_header_joins(tmp_path / "f.asd") is None
    short = _fake_header(tmp_path / "g.asd", b"as8", 1000, 1800).read_bytes()[:600]
    (tmp_path / "g.asd").write_bytes(short)
    assert sd.read_asd_header_joins(tmp_path / "g.asd") is None


def test_resolve_joins_priority_and_disagreement():
    rng = np.random.default_rng(8)
    R = _with_steps(10 ** -_smooth_spectra() + rng.normal(0, 2e-4, (30, WL.size)), rng)
    user = sd.resolve_joins(
        WL, user_joins=(990.0,), header_joins=(1000, 1800), X=R, space="reflectance"
    )
    assert user.source == "user" and user.joins == (990.0,)
    head = sd.resolve_joins(WL, header_joins=(1000.0, 1800.0), X=R, space="reflectance")
    assert head.source == "header" and not head.warnings
    off = sd.resolve_joins(WL, header_joins=(1000.0, 1830.0), X=R, space="reflectance")
    assert off.source == "header" and "1800" in off.warnings[0]
    det = sd.resolve_joins(WL, header_joins=(0.0, 0.0), X=R, space="reflectance")
    assert det.source == "detected" and det.joins == (1000.0, 1800.0)


def test_block_slices_and_space_checks():
    assert len(sd.block_slices(WL, np.array([1000.0, 1800.0]))) == 3  # arrays, not just tuples
    assert sd._plausible_joins(np.array([1000.0, 1800.0]), WL) == (1000.0, 1800.0)
    uneven = np.r_[np.arange(400.0, 1000.0, 2.0), np.arange(1000.0, 1400.0, 5.0)]
    with pytest.raises(ValueError, match="spacing varies"):
        sd.check_uniform_blocks(uneven, None)
    sd.check_uniform_blocks(uneven, (998.0,))  # even inside each block
    sl = sd.block_slices(WL, (1000, 1800, 9999))
    assert [(WL[s][0], WL[s][-1]) for s in sl] == [(350, 1000), (1001, 1800), (1801, 2500)]
    with pytest.raises(ValueError, match="percent"):
        sd.to_absorbance(np.full((2, 10), 50.0), "reflectance")
    with pytest.raises(ValueError, match="increasing"):
        sd.check_wavelengths([3, 2, 1])
