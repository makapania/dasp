"""Physical-match calibration transfer: per-block blur, band-depth gain, optional shift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectral_predict import calibration_transfer as ct
from spectral_predict import physical_match as pm
from spectral_predict import spectral_diagnostics as sd

REPO = Path(__file__).resolve().parents[1]
JOINS = (1000.0, 1800.0)
TRUE_SIGMA = (0.0, 3.0, 4.0)
TRUE_GAIN = (0.8, 0.65, 0.6)
TRUE_OFFSET = (0.05, 0.1, 0.15)


@pytest.fixture(scope="module")
def example_reflectance():
    from spectral_predict.io import read_asd_dir

    X, _ = read_asd_dir(str(REPO / "example"))
    return X.columns.astype(float).to_numpy(), X.to_numpy(dtype=float)


def _make_primary(R_sat_clean, wl, sigma=TRUE_SIGMA, gain=TRUE_GAIN, offset=TRUE_OFFSET, shift=0.0):
    """Primary = per block: blur(+shift) the satellite in R, then g*A + c."""
    params = {
        "space": "reflectance",
        "blocks": np.array([[wl[s][0], wl[s][-1]] for s in sd.block_slices(wl, JOINS)]),
        "sigma_nm": np.array(sigma, dtype=float),
        "shift_nm": np.full(3, shift),
        "gain": np.array(gain, dtype=float),
        "offset": np.array(offset, dtype=float),
    }
    return pm.apply_physical_match(R_sat_clean, params, wl)


@pytest.fixture(scope="module")
def noisy_pair(example_reflectance):
    """Satellite noise 10-30x the primary's (the regime that biased a pooled fit)."""
    wl, R = example_reflectance
    rng = np.random.default_rng(1)
    Rp = _make_primary(R, wl) * (1 + rng.normal(0, 2e-5, R.shape))
    Rs = R * (1 + rng.normal(0, 4e-4, R.shape))
    return wl, Rp, Rs, R


def test_blur_identity_and_normalisation():
    wl = np.arange(400.0, 500.0)
    X = np.random.default_rng(0).normal(size=(3, wl.size))
    assert np.array_equal(pm.blur_shift(X, wl, 0.0), X)
    ones = np.ones((1, wl.size))
    np.testing.assert_allclose(pm.blur_shift(ones, wl, 3.0), ones)  # edges renormalised


def test_blur_never_crosses_a_join(example_reflectance):
    wl, R = example_reflectance
    params = {
        "space": "reflectance",
        "blocks": np.array([[wl[s][0], wl[s][-1]] for s in sd.block_slices(wl, JOINS)]),
        "sigma_nm": np.array([5.0, 5.0, 5.0]),
        "shift_nm": np.zeros(3),
        "gain": np.ones(3),
        "offset": np.zeros(3),
    }
    R2 = R[:2].copy()
    R2[:, wl > 1000] *= 0.5  # change only SWIR
    a = pm.apply_physical_match(R[:2], params, wl)
    b = pm.apply_physical_match(R2, params, wl)
    vnir = wl <= 1000
    np.testing.assert_allclose(a[:, vnir], b[:, vnir])


def test_recovers_blur_and_gain_under_satellite_noise(noisy_pair):
    wl, Rp, Rs, _ = noisy_pair
    prm = pm.estimate_physical_match(Rp, Rs, wl, space="reflectance", joins=JOINS)
    for b in (1, 2):  # SWIR blocks carry the band structure
        assert abs(prm["sigma_nm"][b] - TRUE_SIGMA[b]) <= 1.0
        assert prm["gain"][b] == pytest.approx(TRUE_GAIN[b], rel=0.03)
    assert not prm["satellite_coarser"].any()  # the satellite is the sharper one here


def _broadband_pair(n=20, seed=0):
    wl = np.arange(1000.0, 1800.0)
    rng = np.random.default_rng(seed)
    centres = (1150, 1300, 1450, 1600)
    A = 0.3 + sum(
        np.exp(-0.5 * ((wl[None, :] - c) / 20) ** 2) * rng.uniform(0.05, 0.25, (n, 1))
        for c in centres
    )
    As = A + rng.normal(0, 1e-4, A.shape)
    Ap = 0.7 * A + 0.05 + rng.normal(0, 1e-5, A.shape)
    return wl, 10**-Ap, 10**-As


@pytest.mark.parametrize("bad", [-0.02, 0.0, np.nan])
def test_a_dead_satellite_channel_does_not_steer_the_fit(bad):
    """One unusable channel mid-block used to push the learned blur to the top of its
    grid (the blur smeared the absorbance spike the clip created)."""
    wl, Rp, Rs = _broadband_pair()
    Rs = Rs.copy()
    Rs[:, wl == 1400] = bad
    prm = pm.estimate_physical_match(Rp, Rs, wl, space="reflectance")
    assert prm["sigma_nm"][0] <= 1.0
    assert prm["gain"][0] == pytest.approx(0.7, rel=0.02)
    assert not prm["fit_mask"][np.abs(wl - 1400) <= 15].any()
    assert np.isfinite(pm.apply_physical_match(Rs, prm, wl)).all()


def test_nan_in_absorbance_input_is_filled_not_propagated():
    wl, Rp, Rs = _broadband_pair()
    Ap, As = np.log10(1 / Rp), np.log10(1 / Rs)
    Ap[3, wl == 1500] = np.nan
    prm = pm.estimate_physical_match(Ap, As, wl, space="absorbance", sigma=0.0)
    assert np.isfinite(prm["gain"]).all() and prm["gain"][0] == pytest.approx(0.7, rel=0.02)


def test_two_standards_are_flagged_as_rough():
    wl, Rp, Rs = _broadband_pair(n=2)
    prm = pm.estimate_physical_match(Rp, Rs, wl, space="reflectance", sigma=0.0)
    assert "Only 2 standards" in prm["notes"]


def test_mean_fit_beats_pooled_least_squares(example_reflectance):
    """Guards the design choice: with heavy, correlated satellite noise (8 nm native
    spacing, 1%), a pooled per-spectrum LS gain is biased low; the mean-D2 fit is not."""
    wl, R = example_reflectance
    rng = np.random.default_rng(1)
    Rp = _make_primary(R, wl) * (1 + rng.normal(0, 2e-5, R.shape))
    grid = np.arange(wl[0], wl[-1] + 8.0, 8.0)
    e = np.array([np.interp(wl, grid, r) for r in rng.normal(0, 1e-2, (R.shape[0], grid.size))])
    Rs = R * (1 + e)
    sl = sd.block_slices(wl, JOINS)[2]
    blur = pm.blur_shift(Rs[:, sl], wl[sl], TRUE_SIGMA[2])
    ds = sd.second_derivative(np.log10(1 / blur), wl[sl])
    dp = sd.second_derivative(np.log10(1 / Rp[:, sl]), wl[sl])
    m = (wl[sl] > 1830) & (wl[sl] < 2470)
    pooled = float((ds[:, m] * dp[:, m]).sum() / (ds[:, m] ** 2).sum())
    prm = pm.estimate_physical_match(Rp, Rs, wl, space="reflectance", joins=JOINS)
    assert abs(pooled / TRUE_GAIN[2] - 1) > 2 * abs(prm["gain"][2] / TRUE_GAIN[2] - 1)


def test_transfer_reproduces_primary_spectra(noisy_pair):
    wl, Rp, Rs, _ = noisy_pair
    prm = pm.estimate_physical_match(Rp, Rs, wl, space="reflectance", joins=JOINS)
    out = pm.apply_physical_match(Rs, prm, wl)
    swir = (wl > 1100) & (wl < 2300)
    before = np.abs(np.log10(Rs[:, swir] / Rp[:, swir])).mean()
    after = np.abs(np.log10(out[:, swir] / Rp[:, swir])).mean()
    assert after < 0.2 * before


def test_single_gain_pools_blocks(noisy_pair):
    wl, Rp, Rs, _ = noisy_pair
    prm = pm.estimate_physical_match(Rp, Rs, wl, space="reflectance", joins=JOINS, gain="single")
    g = prm["gain"][~prm["featureless"]]
    assert np.ptp(g) == 0.0


def test_datasheet_fwhm_sets_sigma(noisy_pair):
    wl, Rp, Rs, _ = noisy_pair
    prm = pm.estimate_physical_match(
        Rp, Rs, wl, space="reflectance", joins=JOINS, fwhm=([3, 10, 10], [3, 6, 6])
    )
    assert prm["sigma_nm"][1] == pytest.approx(8.0 / 2.3548, rel=1e-3)
    assert prm["sigma_nm"][0] == 0.0


def test_shift_is_recovered(example_reflectance):
    wl, R = example_reflectance
    Rp = _make_primary(R, wl, shift=1.5)
    prm = pm.estimate_physical_match(Rp, R, wl, space="reflectance", joins=JOINS, shift=True)
    assert abs(prm["shift_nm"][1] - 1.5) <= 0.5
    assert abs(prm["shift_nm"][2] - 1.5) <= 0.5


def test_featureless_block_is_left_unscaled():
    wl = np.arange(1000.0, 1400.0)
    rng = np.random.default_rng(0)
    bands = np.exp(-0.5 * ((wl[None, :] - 1200) / 15) ** 2) * rng.uniform(0.1, 0.3, (20, 1))
    A = 0.3 + bands
    A[:, wl < 1150] = 0.3 + rng.normal(0, 1e-5, (20, int((wl < 1150).sum())))
    Rs = 10**-A
    Rp = 10 ** -(0.7 * A + 0.1)
    prm = pm.estimate_physical_match(Rp, Rs, wl, space="reflectance", joins=(1149.0,))
    assert prm["featureless"][0] and prm["gain"][0] == 1.0
    assert "no measurable band structure" in prm["notes"]
    assert prm["gain"][1] == pytest.approx(0.7, rel=0.02)


def test_coarser_satellite_is_not_sharpened(example_reflectance):
    wl, R = example_reflectance
    Rs = _make_primary(R, wl, sigma=(0, 4, 4), gain=(1, 1, 1), offset=(0, 0, 0))
    prm = pm.estimate_physical_match(R, Rs, wl, space="reflectance", joins=JOINS)
    assert prm["satellite_coarser"][1] and prm["satellite_coarser"][2]
    assert prm["sigma_nm"][1] == 0.0 and prm["sigma_nm"][2] == 0.0
    assert "lower-resolution" in prm["notes"]


def test_absorbance_space_round_trip(example_reflectance):
    wl, R = example_reflectance
    A = np.log10(1 / R)
    Ap = 0.7 * A + 0.1
    prm = pm.estimate_physical_match(Ap, A, wl, space="absorbance", joins=JOINS, sigma=0.0)
    out = pm.apply_physical_match(A, prm, wl)
    swir = (wl > 1100) & (wl < 2300)
    np.testing.assert_allclose(out[:, swir], Ap[:, swir], atol=2e-3)


def test_space_mismatches_are_refused(example_reflectance):
    wl, R = example_reflectance
    with pytest.raises(ValueError, match="percent"):
        pm.estimate_physical_match(R * 100, R * 100, wl, space="reflectance")
    prm = pm.estimate_physical_match(R, R, wl, space="reflectance", joins=JOINS, sigma=0.0)
    with pytest.raises(ValueError, match="fitted on reflectance"):
        pm.apply_physical_match(R * 100, prm, wl)
    with pytest.raises(ValueError, match="space"):
        pm.estimate_physical_match(R, R, wl, space="linear")


def test_grid_must_match_blocks(example_reflectance):
    wl, R = example_reflectance
    prm = pm.estimate_physical_match(R, R, wl, space="reflectance", joins=JOINS, sigma=0.0)
    with pytest.raises(ValueError, match="grid"):
        pm.apply_physical_match(R[:, :-50], prm, wl[:-50] + 0.5)


def test_spacing_may_change_only_at_a_join():
    wl = np.r_[np.arange(400.0, 1000.0, 2.0), np.arange(1000.0, 1400.0, 5.0)]
    rng = np.random.default_rng(0)
    A = 0.4 + np.exp(-0.5 * ((wl[None, :] - 1200) / 20) ** 2) * rng.uniform(0.1, 0.3, (15, 1))
    A = A + 0.05 * np.exp(-0.5 * ((wl[None, :] - 700) / 20) ** 2) * rng.uniform(0.5, 1, (15, 1))
    with pytest.raises(ValueError, match="spacing varies"):
        pm.estimate_physical_match(0.8 * A, A, wl, space="absorbance", sigma=0.0)
    prm = pm.estimate_physical_match(0.8 * A, A, wl, space="absorbance", sigma=0.0, joins=(998.0,))
    np.testing.assert_allclose(prm["gain"], 0.8, rtol=1e-4)
    # blur in the 5 nm block is in nm, not channels: a 10 nm sigma spans 2 channels
    blurred = pm.blur_shift(A[:, wl >= 1000], wl[wl >= 1000], 10.0)
    ref = pm.blur_shift(
        np.vstack(
            [np.interp(np.arange(1000.0, 1396.0), wl[wl >= 1000], a) for a in A[:, wl >= 1000]]
        ),
        np.arange(1000.0, 1396.0),
        10.0,
    )
    mid = (wl[wl >= 1000] > 1100) & (wl[wl >= 1000] < 1300)
    np.testing.assert_allclose(
        blurred[:, mid], ref[:, (wl[wl >= 1000][mid] - 1000).astype(int)], atol=2e-3
    )


def test_save_load_round_trip_keeps_every_param(tmp_path, noisy_pair):
    from spectral_predict.transfer_evaluation import TransferCandidate, fit_transfer

    wl, Rp, Rs, _ = noisy_pair
    tm = fit_transfer(
        TransferCandidate.make("physical", sigma="learn", gain="per_block"),
        Rp[:20],
        Rs[:20],
        wavelengths=wl,
        space="reflectance",
        joins=JOINS,
    )
    prefix = ct.save_transfer_model(tm, tmp_path, name="pm")
    back = ct.load_transfer_model(prefix)
    assert set(back.params) == set(tm.params)
    for k, v in tm.params.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(back.params[k], v)
        else:
            assert back.params[k] == v
    np.testing.assert_allclose(
        ct.apply_transfer_dispatch(Rs[20:], back), ct.apply_transfer_dispatch(Rs[20:], tm)
    )
    assert back.meta["format_version"] == pm.PHYSICAL_MATCH_FORMAT_VERSION
    assert back.meta["joins"] == list(JOINS)


def test_old_dispatch_names_the_new_method():
    assert ct.method_display_name("physical_match", short=True) == "Physical match"


def test_smoothing_between_instruments_reads_blur_not_band_depth(example_reflectance):
    from spectral_predict.instrument_profiles import estimate_smoothing_between_instruments

    wl, R = example_reflectance
    keep = (wl >= 1100) & (wl <= 1750)
    A_high = np.log10(1 / R[:, keep])
    A_low = 0.6 * pm.blur_shift(A_high, wl[keep], 3.0) + 0.1  # shallower bands + 3 nm blur
    got = estimate_smoothing_between_instruments(
        wl[keep], A_high, wl[keep], A_low, [0, 1, 2, 3, 4, 6, 8, 10]
    )
    assert got == 3
