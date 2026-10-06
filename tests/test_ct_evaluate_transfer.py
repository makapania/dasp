"""F2 part 1: centred low-rank PDS, dual-form DS, prediction slope/bias from satellite
standards, and leave-one-standard-out comparison of transfer methods."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spectral_predict import calibration_transfer as ct  # noqa: E402
from spectral_predict.transfer_evaluation import (  # noqa: E402
    NONE_LABEL,
    REFERENCE_LABEL,
    TransferCandidate,
    default_transfer_candidates,
    evaluate_transfer,
    fit_transfer,
    pair_standards_by_id,
    predict_fn_from_model,
)

REPO = Path(__file__).resolve().parents[1]


def _pair(n: int = 10, p: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.standard_normal((n, p)), axis=1) * 0.05 + 1.0
    Xs = base + 0.01 * rng.standard_normal((n, p))
    Xp = 0.9 * np.roll(Xs, 1, axis=1) + 0.05 + 0.002 * rng.standard_normal((n, p))
    return Xp, Xs


def _pds_loop_reference(Xp, Xs, window, rank):
    """Plain per-wavelength loop; the batched estimator must match it."""
    n, p = Xs.shape
    h = window // 2
    B = np.zeros((p, window))
    off = np.zeros(p)
    mp, ms = Xp.mean(0), Xs.mean(0)
    scale = max(np.linalg.norm(Xs - ms, 2), np.sqrt(n) * np.abs(Xs).max())
    tol = max(n, window) * np.finfo(float).eps * scale
    for i in range(p):
        a, b = max(0, i - h), min(p, i + h + 1)
        Xw = Xs[:, a:b] - ms[a:b]
        U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
        r = min(rank, n - 1, Xw.shape[1], int(np.sum(s > tol)))
        coef = Vt[:r].T @ ((U[:, :r].T @ (Xp[:, i] - mp[i])) / s[:r])
        B[i, a - (i - h) : a - (i - h) + coef.size] = coef
        off[i] = mp[i] - ms[a:b] @ coef
    return B, off


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def test_dual_ds_equals_primal_centred_ridge():
    Xp, Xs = _pair(n=8, p=30)
    params = ct.estimate_ds_dual(Xp, Xs, lam_rel=1e-2)
    Xc, Pc = Xs - Xs.mean(0), Xp - Xp.mean(0)
    A = np.linalg.solve(Xc.T @ Xc + params["lam"] * np.eye(Xc.shape[1]), Xc.T @ Pc)
    X_new = _pair(n=5, p=30, seed=3)[1]
    expected = Xp.mean(0) + (X_new - Xs.mean(0)) @ A
    np.testing.assert_allclose(ct.apply_ds_dual(X_new, params), expected, atol=1e-10)


def test_uncentred_dual_ds_matches_legacy_ds():
    Xp, Xs = _pair(n=8, p=30)
    params = ct.estimate_ds_dual(Xp, Xs, lam_rel=1e-3, center=False)
    A = ct.estimate_ds(Xp, Xs, lam=params["lam"])
    X_new = _pair(n=4, p=30, seed=5)[1]
    np.testing.assert_allclose(ct.apply_ds_dual(X_new, params), ct.apply_ds(X_new, A), atol=1e-8)


def test_dual_ds_rejects_bad_lambda_and_constant_standards():
    Xp, Xs = _pair(n=5, p=10)
    with pytest.raises(ValueError, match="lam_rel"):
        ct.estimate_ds_dual(Xp, Xs, lam_rel=0.0)
    flat = np.tile(Xs[:1], (5, 1))
    with pytest.raises(ValueError, match="do not vary"):
        ct.estimate_ds_dual(Xp, flat)


@pytest.mark.parametrize("window,rank", [(5, 2), (11, 3), (11, 20), (31, 4), (1, 1)])
def test_batched_centred_pds_matches_loop(window, rank):
    Xp, Xs = _pair(n=9, p=40)
    params = ct.estimate_pds_lowrank(Xp, Xs, window=window, rank=rank)
    B_ref, off_ref = _pds_loop_reference(Xp, Xs, window, rank)
    np.testing.assert_allclose(params["B_centred"], B_ref, atol=1e-9)
    np.testing.assert_allclose(params["offset"], off_ref, atol=1e-9)
    assert params["rank"] == min(rank, 8, window)
    assert isinstance(params["rank"], int)


def test_centred_pds_recovers_exact_local_affine_map():
    rng = np.random.default_rng(1)
    n, p, window = 12, 30, 5
    Xs = rng.standard_normal((n + 6, p))
    coef = rng.standard_normal(3)
    Xp = 0.3 + coef[0] * np.roll(Xs, 1, 1) + coef[1] * Xs + coef[2] * np.roll(Xs, -1, 1)
    inner = slice(2, p - 2)  # np.roll wraps; check interior channels only
    params = ct.estimate_pds_lowrank(Xp[:n], Xs[:n], window=window)
    out = ct.apply_pds_centred(Xs[n:], params)
    np.testing.assert_allclose(out[:, inner], Xp[n:, inner], atol=1e-8)


def test_centred_pds_constant_window_predicts_primary_mean():
    Xp, Xs = _pair(n=6, p=20)
    Xs[:, :] = Xs[0]  # no satellite variation at all
    params = ct.estimate_pds_lowrank(Xp, Xs, window=5, rank=2)
    np.testing.assert_allclose(params["B_centred"], 0.0)
    np.testing.assert_allclose(ct.apply_pds_centred(Xs[:2], params), np.tile(Xp.mean(0), (2, 1)))


def test_new_forms_cannot_be_misread_by_old_apply_code():
    """Older builds read params['B'] / params['A']; the new forms must not have them."""
    Xp, Xs = _pair()
    pds = ct.estimate_pds_lowrank(Xp, Xs, window=5, rank=2)
    ds = ct.estimate_ds_dual(Xp, Xs)
    assert "B" not in pds and "A" not in ds


def test_dispatch_handles_legacy_and_new_forms():
    Xp, Xs = _pair()
    wl = np.arange(Xs.shape[1], dtype=float)
    B = ct.estimate_pds(Xp, Xs, window=5)
    A = ct.estimate_ds(Xp, Xs, lam=1e-3)
    legacy_pds = ct.TransferModel("p", "s", "pds", wl, {"B": B, "window": 5})
    legacy_ds = ct.TransferModel("p", "s", "ds", wl, {"A": A})
    np.testing.assert_array_equal(ct.apply_transfer_dispatch(Xs, legacy_pds), ct.apply_pds(Xs, B))
    np.testing.assert_array_equal(ct.apply_transfer_dispatch(Xs, legacy_ds), ct.apply_ds(Xs, A))
    pds = ct.estimate_pds_lowrank(Xp, Xs, window=5, rank=2)
    ds = ct.estimate_ds_dual(Xp, Xs)
    np.testing.assert_array_equal(
        ct.apply_transfer_dispatch(Xs, ct.TransferModel("p", "s", "pds", wl, pds)),
        ct.apply_pds_centred(Xs, pds),
    )
    np.testing.assert_array_equal(
        ct.apply_transfer_dispatch(Xs, ct.TransferModel("p", "s", "ds", wl, ds)),
        ct.apply_ds_dual(Xs, ds),
    )


@pytest.mark.parametrize(
    "candidate",
    [
        TransferCandidate.make("pds", window=5, rank=2),
        TransferCandidate.make("ds", lam_rel=1e-2),
        TransferCandidate.make("tsr"),
    ],
    ids=lambda c: c.method,
)
def test_save_load_round_trip(tmp_path, candidate):
    Xp, Xs = _pair()
    wl = np.linspace(1000, 1100, Xs.shape[1])
    tm = fit_transfer(candidate, Xp, Xs, wavelengths=wl, primary_id="lab", satellite_id="field")
    prefix = ct.save_transfer_model(tm, tmp_path)
    loaded = ct.load_transfer_model(prefix)
    np.testing.assert_allclose(
        ct.apply_transfer_dispatch(Xs, loaded), ct.apply_transfer_dispatch(Xs, tm), atol=1e-12
    )
    assert (loaded.primary_id, loaded.satellite_id) == ("lab", "field")
    if candidate.method in ("pds", "ds"):
        assert loaded.meta["format_version"] == ct.TRANSFER_FORMAT_VERSION
    if candidate.method == "pds":
        assert loaded.params["rank"] == 2


def test_prediction_correction_slope_bias_and_bias_only():
    y = np.array([2.0, 4.0, 6.0, 8.0])
    yhat = (y - 1.0) / 2.0
    corr = ct.estimate_prediction_correction(y, yhat)
    assert corr["slope"] == pytest.approx(2.0) and corr["bias"] == pytest.approx(1.0)
    assert corr["source"] == "satellite_standards" and corr["fit"] == "slope_bias"
    assert "metrics_original" not in corr and "metrics_corrected" not in corr
    from spectral_predict.bias_correction import apply_correction

    np.testing.assert_allclose(apply_correction(yhat, corr), y)
    bias_only = ct.estimate_prediction_correction(y, y - 0.5, fit_slope=False)
    assert bias_only["slope"] == 1.0 and bias_only["bias"] == pytest.approx(0.5)


def test_prediction_correction_guards():
    with pytest.raises(ValueError, match="at least 3"):
        ct.estimate_prediction_correction([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="do not vary"):
        ct.estimate_prediction_correction([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    narrow = ct.estimate_prediction_correction(
        [5.0, 5.1, 5.2], [5.0, 5.2, 5.1], y_range_reference=(0.0, 20.0)
    )
    assert any("30%" in w for w in narrow["warnings"])


# ---------------------------------------------------------------------------
# Leave-one-standard-out evaluation
# ---------------------------------------------------------------------------


def _linear_predict(p: int, seed: int = 0):
    w = np.random.default_rng(seed).standard_normal(p) / p
    return lambda Z: np.asarray(Z) @ w + 3.0


def test_default_candidates_fit_every_fold():
    for n in range(3, 13):
        for c in default_transfer_candidates(n, has_y=True, has_predict=True, n_wavelengths=40):
            assert c.min_fit_standards <= n - 1, (n, c)
    labels = [c.method for c in default_transfer_candidates(3, has_y=False, has_predict=True)]
    assert "pred_bias" not in labels and labels[0] == "none"


def test_prediction_bias_loso_hand_check():
    Xp, Xs = _pair(n=4, p=20)
    predict = _linear_predict(20)
    y = predict(Xp) + np.array([0.1, -0.2, 0.05, 0.3])
    ev = evaluate_transfer(
        Xp, Xs, y=y, predict=predict, candidates=[TransferCandidate.make("pred_bias")]
    )
    y_sat = predict(Xs)
    expected = np.array([y_sat[i] + np.mean(np.delete(y - y_sat, i)) for i in range(4)])
    np.testing.assert_allclose(ev.oof_predictions["Prediction bias"], expected)


def test_left_out_standard_is_not_used_to_fit_its_own_transfer():
    """Changing standard i's primary spectrum and y must not change its transferred
    spectrum or its corrected prediction."""
    Xp, Xs = _pair(n=6, p=20)
    predict = _linear_predict(20)
    y = predict(Xp)
    cands = [TransferCandidate.make("pds", window=5, rank=2), TransferCandidate.make("pred_bias")]
    ev1 = evaluate_transfer(Xp, Xs, y=y, predict=predict, candidates=cands)
    Xp2, y2 = Xp.copy(), y.copy()
    Xp2[2] += 5.0
    y2[2] += 100.0
    ev2 = evaluate_transfer(Xp2, Xs, y=y2, predict=predict, candidates=cands)
    label = cands[0].label
    Z1 = Xp - ev1.oof_residual_spectra[label]
    Z2 = Xp2 - ev2.oof_residual_spectra[label]
    np.testing.assert_allclose(Z1[2], Z2[2], atol=1e-10)
    assert ev1.oof_predictions["Prediction bias"][2] == pytest.approx(
        ev2.oof_predictions["Prediction bias"][2]
    )


def test_leaderboard_rows_and_order():
    Xp, Xs = _pair(n=8, p=40)
    predict = _linear_predict(40)
    y = predict(Xp)
    ev = evaluate_transfer(Xp, Xs, y=y, predict=predict)
    board = ev.leaderboard
    assert ev.score_column == "RMSD_vs_primary"
    assert NONE_LABEL in set(board["label"])
    assert board.iloc[-1]["label"] == REFERENCE_LABEL
    ranked = board.iloc[:-1]
    scores = ranked["RMSD_vs_primary"].dropna().to_numpy()
    assert np.all(np.diff(scores) >= 0)
    assert ev.best.method != "none"
    assert ev.n_candidates == len(ranked)


def test_rank_by_rmsep_and_invalid_choices():
    Xp, Xs = _pair(n=6, p=20)
    predict = _linear_predict(20)
    ev = evaluate_transfer(Xp, Xs, y=predict(Xp), predict=predict, rank_by="RMSEP")
    scores = ev.leaderboard.iloc[:-1]["RMSEP"].dropna().to_numpy()
    assert ev.score_column == "RMSEP" and np.all(np.diff(scores) >= 0)
    with pytest.raises(ValueError, match="needs y and a model"):
        evaluate_transfer(Xp, Xs, predict=predict, rank_by="RMSEP")
    with pytest.raises(ValueError, match="rank_by must be"):
        evaluate_transfer(Xp, Xs, rank_by="R2")


def test_score_column_without_y_or_model():
    Xp, Xs = _pair(n=5, p=20)
    assert evaluate_transfer(Xp, Xs).score_column == "spectral_RMSE"
    assert evaluate_transfer(Xp, Xs, predict=_linear_predict(20)).score_column == "RMSD_vs_primary"


def test_improvement_guard_when_no_correction_is_perfect():
    Xp, _ = _pair(n=5, p=20)
    ev = evaluate_transfer(Xp, Xp.copy())
    assert ev.leaderboard["improvement"].isna().all()


def test_evaluate_transfer_input_errors():
    Xp, Xs = _pair(n=5, p=20)
    with pytest.raises(ValueError, match="at least 3"):
        evaluate_transfer(Xp[:2], Xs[:2])
    with pytest.raises(ValueError, match="needs a predict"):
        evaluate_transfer(Xp, Xs, y=np.ones(5))
    bad = Xs.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        evaluate_transfer(Xp, bad)
    with pytest.raises(ValueError, match="same shape"):
        evaluate_transfer(Xp, Xs[:, :10])


def test_predict_fn_rejects_classifiers():
    class _Clf:
        classes_ = np.array([0, 1])

    with pytest.raises(NotImplementedError):
        predict_fn_from_model({"model": _Clf(), "metadata": {}}, [1.0, 2.0])
    with pytest.raises(NotImplementedError):
        predict_fn_from_model({"model": None, "metadata": {"task_type": "one_class"}}, [1.0])


# ---------------------------------------------------------------------------
# Pairing by sample ID
# ---------------------------------------------------------------------------


def _frames():
    wl = [1000.0, 1002.0, 1004.0]
    primary = pd.DataFrame(
        np.arange(12.0).reshape(4, 3), index=["A1", "B2", "C3", "D4"], columns=wl
    )
    satellite = pd.DataFrame(
        np.arange(12.0).reshape(4, 3)[[2, 0, 1, 3]] + 100,
        index=["c3.asd", "a1", "B 2", "E5"],
        columns=wl,
    )
    return primary, satellite


def test_pairing_ignores_order_extension_case_and_spaces():
    paired = pair_standards_by_id(*_frames())
    assert paired.ids == ["A1", "B2", "C3"]
    assert paired.satellite_ids == ["a1", "B 2", "c3.asd"]
    np.testing.assert_array_equal(paired.X_satellite - 100, paired.X_primary)
    assert paired.unmatched_primary == ["D4"] and paired.unmatched_satellite == ["E5"]


def test_pairing_reorders_satellite_columns():
    primary, satellite = _frames()
    satellite = satellite[satellite.columns[::-1]]
    paired = pair_standards_by_id(primary, satellite)
    np.testing.assert_array_equal(paired.X_satellite - 100, paired.X_primary)


def test_pairing_refuses_duplicates_and_different_grids():
    primary, satellite = _frames()
    dup = satellite.rename(index={"E5": "A1.asd"})
    with pytest.raises(ValueError, match="match each other"):
        pair_standards_by_id(primary, dup)
    other_grid = satellite.copy()
    other_grid.columns = [1001.0, 1003.0, 1005.0]
    with pytest.raises(ValueError, match="common grid"):
        pair_standards_by_id(primary, other_grid)


# ---------------------------------------------------------------------------
# Apply paths outside the backend use the one dispatcher
# ---------------------------------------------------------------------------


def test_equalization_mapping_applies_any_method():
    from spectral_predict.equalization import build_equalization_mapping_for_instrument

    Xp, Xs = _pair()
    wl = np.arange(Xs.shape[1], dtype=float)
    tm = fit_transfer(TransferCandidate.make("tsr"), Xp, Xs, wavelengths=wl)
    mapping = build_equalization_mapping_for_instrument(None, wl, transfer_model=tm)
    np.testing.assert_allclose(mapping(Xs, wl), ct.apply_tsr(Xs, tm.params), atol=1e-12)


def test_gui_apply_transfer_model_uses_dispatcher():
    import spectral_predict_gui_optimized as gui

    Xp, Xs = _pair()
    wl = np.arange(Xs.shape[1], dtype=float)
    tm = fit_transfer(TransferCandidate.make("ds", lam_rel=1e-2), Xp, Xs, wavelengths=wl)
    out = gui.SpectralPredictApp._apply_transfer_model(None, Xs, tm)
    np.testing.assert_allclose(out, ct.apply_ds_dual(Xs, tm.params))


# ---------------------------------------------------------------------------
# Simulated second instrument from the shipped example data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulated_instruments():
    from scipy.ndimage import gaussian_filter1d

    from spectral_predict.io import align_xy, read_asd_dir, read_reference_csv

    X, _ = read_asd_dir(str(REPO / "example"))
    ref = read_reference_csv(str(REPO / "example" / "BoneCollagen.csv"), "File Number")
    X, y = align_xy(X, ref, "File Number", "%Collagen")
    wl = X.columns.astype(float).to_numpy()
    keep = (wl >= 400) & (wl <= 2450) & (np.round(wl) % 4 == 0)
    wl = wl[keep]
    A = np.log10(1.0 / np.clip(X.to_numpy()[:, keep], 1e-6, None))
    step = wl[1] - wl[0]
    shifted = np.array([np.interp(wl, wl + 6.0, a) for a in A])
    blurred = gaussian_filter1d(shifted, sigma=(8 / 2.355) / step, axis=1)
    t = (wl - wl.min()) / (wl.max() - wl.min())
    sat = blurred * (1 + 0.2 * (t - 0.5)) + 0.05 * A.std() * (t**2 - 0.3 * t)
    sat = sat + np.random.default_rng(0).normal(0, 0.002, sat.shape)
    return wl, A, sat, y.to_numpy(float)


def test_simulated_instrument_ranking(simulated_instruments):
    from sklearn.cross_decomposition import PLSRegression

    from spectral_predict.sample_selection import kennard_stone

    wl, A, sat, y = simulated_instruments
    rng = np.random.default_rng(4)
    test = rng.choice(len(y), 15, replace=False)
    cal = np.setdiff1d(np.arange(len(y)), test)
    pls = PLSRegression(6).fit(A[cal], y[cal])

    def predict(Z):
        return pls.predict(Z).ravel()

    std = cal[kennard_stone(A[cal], n_samples=8)]
    ev = evaluate_transfer(A[std], sat[std], y=y[std], predict=predict)
    board = ev.leaderboard.set_index("label")
    assert ev.best.method != "none"
    assert board.loc[ev.best.label, "RMSD_vs_primary"] < board.loc[NONE_LABEL, "RMSD_vs_primary"]

    def external_rmsep(candidate):
        fitted = fit_transfer(
            candidate, A[std], sat[std], y=y[std], predict=predict, wavelengths=wl
        )
        if candidate.kind == "prediction":
            from spectral_predict.bias_correction import apply_correction

            pred = apply_correction(predict(sat[test]), fitted)
        elif fitted is None:
            pred = predict(sat[test])
        else:
            pred = predict(ct.apply_transfer_dispatch(sat[test], fitted))
        return float(np.sqrt(np.mean((pred - y[test]) ** 2)))

    none_ext = external_rmsep(TransferCandidate.make("none"))
    assert external_rmsep(ev.best) < none_ext
    # Centred PDS and slope/bias per wavelength both help on held-out spectra.
    assert external_rmsep(TransferCandidate.make("pds", window=11, rank=2)) < none_ext
    assert external_rmsep(TransferCandidate.make("tsr")) < none_ext
