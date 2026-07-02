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


def test_cv_pool_fold_scaler_fits_train_only_no_leakage(rng):
    """Central no-leakage pin (pr-review finding): the per-fold FoldYScaler
    inside multi_y_cv_pool must be fit on TRAIN-fold Y stats ONLY — never the
    validation fold or the full Y. A regression that fit on full/val Y would
    inflate CV scores yet pass every other (scale-invariant) multi-target test.
    """
    import spectral_predict.multi_y as _my

    n, p = 40, 8
    X = rng.standard_normal((n, p))
    Y = np.column_stack([X[:, 0], 50.0 * X[:, 1]]) + 0.1 * rng.standard_normal((n, 2))

    fold_fit_means: list[np.ndarray] = []
    _Base = _my.FoldYScaler

    class _SpyScaler(_Base):
        def fit(self, Y_arg):
            fold_fit_means.append(np.asarray(Y_arg, dtype=float).mean(axis=0).copy())
            return super().fit(Y_arg)

    splitter = build_cv_splitter("kfold", 4, "regression", random_state=0)
    expected_train_means = [
        Y[tr].mean(axis=0)
        for tr, _ in build_cv_splitter("kfold", 4, "regression", random_state=0).split(X)
    ]

    # multi_y_cv_pool constructs the module-global FoldYScaler(); patch it.
    orig = _my.FoldYScaler
    _my.FoldYScaler = _SpyScaler
    try:
        _my.multi_y_cv_pool(
            PLSRegression(n_components=2, scale=False), X, Y, splitter, scale_y=True
        )
    finally:
        _my.FoldYScaler = orig

    assert len(fold_fit_means) == len(expected_train_means) > 0
    for got, exp in zip(fold_fit_means, expected_train_means):
        np.testing.assert_allclose(got, exp, atol=1e-12)
    # Proves the stats are train-fold, not full-Y: at least one fold mean differs
    # from the global mean (they would all equal it if fit on the full block).
    global_mean = Y.mean(axis=0)
    assert any(not np.allclose(fm, global_mean, atol=1e-9) for fm in fold_fit_means)


def test_cv_pool_repeated_kfold_multitarget_averages(rng):
    """Pooling pin (pr-review finding): repeated_kfold multi-target exercises the
    pred_sum/pred_count averaging branch (pred_count > 1) broadcast over
    (n, n_targets) — plain kfold only ever hits pred_count == 1."""
    n, p = 40, 10
    X = rng.standard_normal((n, p))
    Y = np.column_stack([X[:, 0], X[:, 1]]) + 0.1 * rng.standard_normal((n, 2))
    yt, yp = multi_y_cv_pool(
        PLSRegression(n_components=2, scale=False), X, Y,
        "repeated_kfold", scale_y=True, n_folds=4, n_repeats=3, random_state=0,
    )
    assert yt.shape == yp.shape == (n, 2)
    assert np.all(np.isfinite(yp))


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


# --------------------------------------------------------------------------- #
# get_feature_importances -- multi-target paths (T-17 review, blockers 1 & 2)
# --------------------------------------------------------------------------- #
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

from spectral_predict.models import get_feature_importances


@pytest.mark.parametrize(
    "model_name, ctor",
    [
        ("Ridge", lambda: Ridge(alpha=1.0)),
        ("Lasso", lambda: Lasso(alpha=0.01)),
        ("ElasticNet", lambda: ElasticNet(alpha=0.01)),
    ],
)
def test_get_feature_importances_linear_multi_y_aggregates(xy_multi, model_name, ctor):
    """Blocker 1: multi-output linear coef_ (n_targets, n_features) must be
    aggregated across targets (mean |coef|), NOT collapsed to target 0."""
    X, Y = xy_multi
    model = ctor().fit(X, Y)
    imp = get_feature_importances(model, model_name, X, Y)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))
    # Must equal the mean of per-target |coef| -- and NOT target 0 alone.
    coefs = np.abs(model.coef_)  # (n_targets, n_features)
    np.testing.assert_allclose(imp, coefs.mean(axis=0))
    if coefs.shape[0] > 1 and not np.allclose(coefs[0], coefs.mean(axis=0)):
        assert not np.allclose(imp, coefs[0])  # collapse-to-target-0 is fixed


def test_get_feature_importances_linear_single_y_byte_identical(rng):
    """Guardrail 1: single-Y linear importance must stay byte-identical
    (coef_ is 1-D, aggregation branch skipped)."""
    X = rng.standard_normal((40, 10))
    y = X[:, :3] @ rng.standard_normal(3) + 0.1 * rng.standard_normal(40)
    model = Ridge(alpha=1.0).fit(X, y)
    imp = get_feature_importances(model, "Ridge", X, y)
    np.testing.assert_array_equal(imp, np.abs(model.coef_))


@pytest.mark.parametrize(
    "model_name, base_ctor",
    [
        ("LightGBM", lambda: LGBMRegressor(n_estimators=20, verbose=-1)),
        ("SVR", lambda: SVR(kernel="linear")),
    ],
)
def test_get_feature_importances_mor_wrapped_multi_y(xy_multi, model_name, base_ctor):
    """Blocker 2: a MultiOutputRegressor-wrapped INDEPENDENT model has no
    feature_importances_/coef_ on the wrapper -- must aggregate per-target
    across .estimators_ instead of raising AttributeError."""
    X, Y = xy_multi
    model = MultiOutputRegressor(base_ctor()).fit(X, Y)
    imp = get_feature_importances(model, model_name, X, Y)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))
    # Equals the mean of each target-estimator's own importance vector.
    per_target = np.column_stack(
        [get_feature_importances(est, model_name, X, Y) for est in model.estimators_]
    )
    np.testing.assert_allclose(imp, per_target.mean(axis=1))


def test_get_feature_importances_lightgbm_single_y_unwrapped(rng):
    """Guardrail 1: single-Y LightGBM is NOT MOR-wrapped -> uses native
    feature_importances_ (byte-identical to legacy)."""
    X = rng.standard_normal((40, 10))
    y = X[:, :3] @ rng.standard_normal(3) + 0.1 * rng.standard_normal(40)
    model = LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
    imp = get_feature_importances(model, "LightGBM", X, y)
    np.testing.assert_array_equal(imp, model.feature_importances_)
