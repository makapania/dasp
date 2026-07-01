"""Tests for the T-17 model-agnostic multi-Y foundation (multi_y.py).

Covers the invariants the plan pins:
* FoldYScaler == a hand-built per-column standardization + inverse.
* multi_y_cv_pool shape-awareness and raw-unit pooling.
* multi_y_metrics: joint Q2 == mean per-target Q2; per-target Q2 == legacy
  r2_score at n_targets == 1 (consolidation pin); raw-unit RMSE (not scaled).
* extract_pls_multi_y returns coef_.T with (n_features, n_targets) orientation.
* reduce_multi_y_score, aggregate_importance, inter_target_correlation,
  cap_components.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

from spectral_predict.cv_utils import build_cv_splitter, cross_val_predict_pooled
from spectral_predict.multi_y import (
    FoldYScaler,
    aggregate_importance,
    cap_components,
    extract_pls_multi_y,
    inter_target_correlation,
    multi_y_cv_pool,
    multi_y_metrics,
    reduce_multi_y_score,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260701)


@pytest.fixture
def xy_multi(rng):
    """(n=60, p=25) X with a correlated 3-target Y block of divergent scale."""
    n, p = 60, 25
    X = rng.standard_normal((n, p))
    base = X[:, :3] @ rng.standard_normal((3, 1))
    Y = np.hstack(
        [
            0.1 * base + 0.01 * rng.standard_normal((n, 1)),  # tiny scale
            100.0 * base + 5.0 * rng.standard_normal((n, 1)),  # huge scale
            base + 0.05 * rng.standard_normal((n, 1)),
        ]
    )
    return X, Y


# --------------------------------------------------------------------------- #
# FoldYScaler
# --------------------------------------------------------------------------- #
def test_fold_y_scaler_matches_manual(xy_multi):
    _, Y = xy_multi
    sc = FoldYScaler().fit(Y)
    manual_mean = Y.mean(axis=0)
    manual_std = Y.std(axis=0, ddof=0)
    np.testing.assert_allclose(sc.mean_, manual_mean)
    np.testing.assert_allclose(sc.std_, manual_std)
    Z = sc.transform(Y)
    np.testing.assert_allclose(Z, (Y - manual_mean) / manual_std, atol=1e-12)
    # transformed columns are zero-mean / unit-std
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=0, ddof=0), 1.0, atol=1e-10)


def test_fold_y_scaler_inverse_roundtrip(xy_multi):
    _, Y = xy_multi
    sc = FoldYScaler().fit(Y)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(Y)), Y, atol=1e-9)


def test_fold_y_scaler_1d_becomes_column(rng):
    y = rng.standard_normal(30)
    sc = FoldYScaler().fit(y)
    z = sc.transform(y)
    assert z.shape == (30, 1)
    np.testing.assert_allclose(sc.inverse_transform(z).ravel(), y, atol=1e-9)


def test_fold_y_scaler_zero_variance_column():
    Y = np.column_stack([np.ones(10), np.arange(10.0)])
    sc = FoldYScaler().fit(Y)
    assert sc.std_[0] == 1.0  # no divide-by-zero
    Z = sc.transform(Y)
    np.testing.assert_allclose(Z[:, 0], 0.0)  # constant column centers to 0


def test_fold_y_scaler_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FoldYScaler().transform(np.zeros((3, 2)))


# --------------------------------------------------------------------------- #
# multi_y_cv_pool
# --------------------------------------------------------------------------- #
def test_cv_pool_shape_and_rawunits(xy_multi):
    X, Y = xy_multi
    cv = build_cv_splitter("kfold", 5, "regression", random_state=0)
    yt, yp = multi_y_cv_pool(PLSRegression(n_components=3), X, Y, cv, scale_y=True)
    assert yt.shape == (X.shape[0], 3)
    assert yp.shape == (X.shape[0], 3)
    # raw units: per-target pred magnitude tracks the raw target magnitude,
    # NOT unit-variance scaled output.
    assert yp[:, 1].std() > 10.0  # huge-scale target predicted in raw units
    assert yp[:, 0].std() < 1.0  # tiny-scale target predicted in raw units


def test_cv_pool_accepts_strategy_string(xy_multi):
    X, Y = xy_multi
    yt, yp = multi_y_cv_pool(
        PLSRegression(n_components=3), X, Y, "kfold", scale_y=True, random_state=0
    )
    assert yt.shape == yp.shape == (X.shape[0], 3)


def test_cv_pool_single_target_matches_legacy_pooled(rng):
    """At n_targets==1 with scale_y=False the pooled preds match the legacy
    1-D cross_val_predict_pooled (INDEPENDENT/raw path, no Y scaling)."""
    n, p = 50, 15
    X = rng.standard_normal((n, p))
    y = X[:, :2] @ rng.standard_normal(2) + 0.1 * rng.standard_normal(n)
    cv = build_cv_splitter("kfold", 5, "regression", random_state=7)
    yt, yp = multi_y_cv_pool(PLSRegression(n_components=2), X, y, cv, scale_y=False)
    legacy = cross_val_predict_pooled(PLSRegression(n_components=2), X, y, cv)
    np.testing.assert_allclose(yp.ravel(), np.ravel(legacy), atol=1e-9)
    np.testing.assert_allclose(yt.ravel(), y, atol=1e-12)


# --------------------------------------------------------------------------- #
# multi_y_metrics
# --------------------------------------------------------------------------- #
def test_metrics_joint_q2_is_mean_per_target_q2(xy_multi):
    X, Y = xy_multi
    cv = build_cv_splitter("kfold", 5, "regression", random_state=0)
    yt, yp = multi_y_cv_pool(PLSRegression(n_components=4), X, Y, cv)
    m = multi_y_metrics(yt, yp)
    np.testing.assert_allclose(m["joint_q2"], np.mean(m["q2"]))
    assert m["q2"].shape == (3,)


def test_metrics_per_target_q2_equals_legacy_r2_at_single_target(rng):
    """Consolidation pin: per-target Q2 == r2_score(all_y_test, all_y_pred)."""
    yt = rng.standard_normal((40, 1))
    yp = yt + 0.2 * rng.standard_normal((40, 1))
    m = multi_y_metrics(yt, yp)
    assert m["q2"][0] == pytest.approx(r2_score(yt.ravel(), yp.ravel()))
    assert m["r2"][0] == pytest.approx(r2_score(yt.ravel(), yp.ravel()))
    assert m["joint_q2"] == pytest.approx(r2_score(yt.ravel(), yp.ravel()))


def test_metrics_constant_target_q2_matches_sklearn():
    """Constant-target edge: per-target Q2 must match sklearn
    r2_score(force_finite=True) -- 1.0 for a perfect constant prediction,
    0.0 otherwise -- preserving the n_targets==1 legacy-parity contract."""
    yt = np.full((6, 1), 5.0)
    yp_perfect = np.full((6, 1), 5.0)
    m = multi_y_metrics(yt, yp_perfect)
    assert m["q2"][0] == pytest.approx(r2_score(yt.ravel(), yp_perfect.ravel()))
    assert m["q2"][0] == 1.0
    yp_off = np.array([5.0, 5.1, 4.9, 5.0, 5.0, 5.0]).reshape(-1, 1)
    m_off = multi_y_metrics(yt, yp_off)
    assert m_off["q2"][0] == pytest.approx(r2_score(yt.ravel(), yp_off.ravel()))
    assert m_off["q2"][0] == 0.0


def test_metrics_rmse_is_raw_not_scaled(xy_multi):
    """RMSE must be raw-unit (scale-sensitive): the huge target has a much
    larger RMSE than the tiny target."""
    X, Y = xy_multi
    cv = build_cv_splitter("kfold", 5, "regression", random_state=0)
    yt, yp = multi_y_cv_pool(PLSRegression(n_components=4), X, Y, cv)
    m = multi_y_metrics(yt, yp)
    # recompute RMSE straight from raw pooled arrays -> must match report
    manual_rmse = np.sqrt(((yp - yt) ** 2).mean(axis=0))
    np.testing.assert_allclose(m["rmse"], manual_rmse, atol=1e-9)
    assert m["rmse"][1] > 10.0 * m["rmse"][0]  # raw scale preserved, not unit-var


def test_metrics_rpd_rer_ccc_bias(rng):
    yt = rng.standard_normal((50, 2)) * np.array([1.0, 4.0])
    yp = yt + 0.3 * rng.standard_normal((50, 2))
    m = multi_y_metrics(yt, yp)
    for s in range(2):
        rmse = np.sqrt(((yp[:, s] - yt[:, s]) ** 2).mean())
        assert m["rpd"][s] == pytest.approx(np.std(yt[:, s]) / rmse)
        assert m["rer"][s] == pytest.approx(np.ptp(yt[:, s]) / rmse)
        assert m["bias"][s] == pytest.approx(np.mean(yp[:, s] - yt[:, s]))
    assert len(m["per_target"]) == 2
    assert m["per_target"][0]["target"] == "target_0"


def test_metrics_target_names_and_shape_mismatch():
    yt = np.zeros((5, 2))
    yp = np.zeros((5, 2))
    m = multi_y_metrics(yt, yp, target_names=["a", "b"])
    assert [d["target"] for d in m["per_target"]] == ["a", "b"]
    with pytest.raises(ValueError):
        multi_y_metrics(np.zeros((5, 2)), np.zeros((5, 3)))
    with pytest.raises(ValueError):
        multi_y_metrics(yt, yp, target_names=["only_one"])


# --------------------------------------------------------------------------- #
# reduce_multi_y_score
# --------------------------------------------------------------------------- #
def test_reduce_rules():
    v = [0.2, 0.4, 0.6]
    assert reduce_multi_y_score(v) == pytest.approx(0.4)
    assert reduce_multi_y_score(v, "min") == pytest.approx(0.2)
    assert reduce_multi_y_score(v, "max") == pytest.approx(0.6)
    assert reduce_multi_y_score(v, "sum") == pytest.approx(1.2)
    assert reduce_multi_y_score(v, "median") == pytest.approx(0.4)
    assert reduce_multi_y_score(0.5) == pytest.approx(0.5)


def test_reduce_bad_rule_and_empty():
    with pytest.raises(ValueError):
        reduce_multi_y_score([1, 2], "nope")
    with pytest.raises(ValueError):
        reduce_multi_y_score([])


# --------------------------------------------------------------------------- #
# extract_pls_multi_y  (coef_.T orientation)
# --------------------------------------------------------------------------- #
def test_extract_pls_multi_y_orientation(xy_multi):
    X, Y = xy_multi
    n_features = X.shape[1]
    n_targets = Y.shape[1]
    pls = PLSRegression(n_components=4).fit(X, Y)
    B = extract_pls_multi_y(pls)
    assert B.shape == (n_features, n_targets)
    # It IS coef_.T on pinned sklearn (coef_ is (n_targets, n_features)).
    assert pls.coef_.shape == (n_targets, n_features)
    np.testing.assert_allclose(B, pls.coef_.T)


def test_extract_pls_multi_y_single_target(rng):
    X = rng.standard_normal((40, 10))
    y = rng.standard_normal(40)
    pls = PLSRegression(n_components=3).fit(X, y)
    B = extract_pls_multi_y(pls)
    assert B.shape == (10, 1)
    np.testing.assert_allclose(B, np.asarray(pls.coef_).reshape(1, -1).T)


# --------------------------------------------------------------------------- #
# aggregate_importance
# --------------------------------------------------------------------------- #
def test_aggregate_importance_rules():
    M = np.array([[1.0, 3.0], [0.0, 4.0]])  # (2 features, 2 targets)
    np.testing.assert_allclose(aggregate_importance(M, "mean"), [2.0, 2.0])
    np.testing.assert_allclose(aggregate_importance(M, "sum"), [4.0, 4.0])
    np.testing.assert_allclose(aggregate_importance(M, "max"), [3.0, 4.0])
    np.testing.assert_allclose(aggregate_importance(M, "l2"), [np.sqrt(10.0), 4.0])
    # 1-D input treated as single target column
    np.testing.assert_allclose(aggregate_importance(np.array([2.0, 5.0])), [2.0, 5.0])
    with pytest.raises(ValueError):
        aggregate_importance(M, "bogus")


# --------------------------------------------------------------------------- #
# inter_target_correlation
# --------------------------------------------------------------------------- #
def test_inter_target_correlation_strong(rng):
    base = rng.standard_normal((100, 1))
    Y = np.hstack([base, base * 2 + 0.05 * rng.standard_normal((100, 1))])
    res = inter_target_correlation(Y)
    assert res["mean_abs_corr"] > 0.9
    assert res["is_weak"] is False
    assert res["corr_matrix"].shape == (2, 2)


def test_inter_target_correlation_weak(rng):
    Y = rng.standard_normal((200, 3))  # independent -> near-zero corr
    res = inter_target_correlation(Y)
    assert res["mean_abs_corr"] < 0.35
    assert res["is_weak"] is True


def test_inter_target_correlation_single_target_noop(rng):
    res = inter_target_correlation(rng.standard_normal(50))
    assert res["mean_abs_corr"] == 0.0
    assert res["is_weak"] is False


# --------------------------------------------------------------------------- #
# cap_components
# --------------------------------------------------------------------------- #
def test_cap_components():
    assert cap_components(20, 100) == 19  # min(N-1, p)
    assert cap_components(200, 30) == 30
    assert cap_components(20, 100, requested=5) == 5
    assert cap_components(20, 100, requested=50) == 19  # clamped to cap
    assert cap_components(2, 100, requested=10) == 1  # floored at 1


# --------------------------------------------------------------------------- #
# compute_vip (T-17 canonical multi-Y fix; single-Y byte-identical)
# --------------------------------------------------------------------------- #
from spectral_predict.models import compute_vip


def _manual_vip(pls, ssq_y):
    W = np.asarray(pls.x_weights_)
    T = np.asarray(pls.x_scores_)
    ssy_comp = ssq_y * np.sum(T ** 2, axis=0)
    ssy_total = float(np.sum(ssy_comp))
    n_features = W.shape[0]
    col_norm_sq = np.sum(W ** 2, axis=0)
    col_norm_sq = np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)
    w_norm_sq = (W ** 2) / col_norm_sq
    return np.sqrt(n_features * (w_norm_sq @ ssy_comp) / ssy_total)


def test_compute_vip_multi_y_matches_canonical_formula(xy_multi):
    """Multi-Y VIP: per-component Y-variance term sums squared y_loadings
    across targets (Wold 2001 / Mehmood 2012 Eq. 1)."""
    X, Y = xy_multi
    pls = PLSRegression(n_components=4).fit(X, Y)
    vip = compute_vip(pls, X, Y)
    Q = np.asarray(pls.y_loadings_)
    ssq_y = np.sum(Q ** 2, axis=0)  # sum across targets per component
    np.testing.assert_allclose(vip, _manual_vip(pls, ssq_y))
    # canonical VIP invariant: mean(VIP**2) == 1
    assert np.mean(vip ** 2) == pytest.approx(1.0, abs=1e-9)
    assert vip.shape == (X.shape[1],)


def test_compute_vip_single_y_byte_identical(rng):
    """Single-target VIP must keep the EXACT Q[0, :] result (byte-identical)."""
    X = rng.standard_normal((45, 12))
    y = X[:, :3] @ rng.standard_normal(3) + 0.1 * rng.standard_normal(45)
    pls = PLSRegression(n_components=3).fit(X, y)
    vip = compute_vip(pls, X, y)
    Q = np.asarray(pls.y_loadings_)
    ssq_y = Q[0, :] ** 2  # exact legacy single-target term
    manual = _manual_vip(pls, ssq_y)
    np.testing.assert_array_equal(vip, manual)  # BYTE-identical, not just close


def test_compute_vip_single_target_equals_1col_multi(rng):
    """A single target fit as (n, 1) must give the exact same VIP as the
    canonical multi-Y path collapses to for one target (Q.shape[0] == 1)."""
    X = rng.standard_normal((45, 12))
    y = X[:, :3] @ rng.standard_normal(3) + 0.1 * rng.standard_normal(45)
    pls1 = PLSRegression(n_components=3).fit(X, y)
    pls2 = PLSRegression(n_components=3).fit(X, y.reshape(-1, 1))
    np.testing.assert_array_equal(compute_vip(pls1, X, y),
                                  compute_vip(pls2, X, y.reshape(-1, 1)))
