"""Tests for the T-17 multi-target orchestrator (multitarget_search.py).

Pins the F2 contracts:
* Table-driven ``resolve_multitarget_strategy``: exact JOINT/INDEPENDENT mode
  for every model in the per-model table; UNKNOWN model RAISES (fails loud);
  the INDEPENDENT precise-note substring is pinned against drift.
* Consolidation pin: at ``n_targets == 1`` the orchestrator's PLS per-target Q2
  equals legacy ``r2_score(all_y_test, all_y_pred)`` within tolerance.
* Grid-engine-only guardrail; PLS-2 multi-target smoke; correlation guardrail;
  not-yet-wired models raise NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

from spectral_predict.cv_utils import build_cv_splitter, cross_val_predict_pooled
from spectral_predict.multitarget_search import (
    INDEPENDENT_PRECISE_NOTE,
    build_multitarget_estimator,
    resolve_multitarget_strategy,
    run_multitarget_search,
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
            0.1 * base + 0.01 * rng.standard_normal((n, 1)),
            100.0 * base + 5.0 * rng.standard_normal((n, 1)),
            base + 0.05 * rng.standard_normal((n, 1)),
        ]
    )
    return X, Y


# --------------------------------------------------------------------------- #
# resolve_multitarget_strategy  (table-driven labeling guardrail)
# --------------------------------------------------------------------------- #
# (model_name, expected_mode, expected_scale_y)
_EXPECTED_STRATEGIES = [
    ("PLS", "JOINT", True),
    ("RandomForest", "JOINT", True),
    ("MLP", "JOINT", True),
    ("CatBoost", "JOINT", True),
    ("XGBoost", "JOINT", True),
    ("Ridge", "INDEPENDENT", False),
    ("Lasso", "INDEPENDENT", False),
    ("ElasticNet", "INDEPENDENT", False),
    ("SVR", "INDEPENDENT", False),
    ("LightGBM", "INDEPENDENT", False),
    ("NeuralBoosted", "INDEPENDENT", False),
]


@pytest.mark.parametrize("model_name,expected_mode,expected_scale_y", _EXPECTED_STRATEGIES)
def test_resolve_strategy_exact_mode(model_name, expected_mode, expected_scale_y):
    strat = resolve_multitarget_strategy(model_name)
    assert strat.model_name == model_name
    assert strat.mode == expected_mode
    assert strat.scale_y is expected_scale_y
    # JOINT couples targets => no precise note; INDEPENDENT carries it verbatim.
    if expected_mode == "JOINT":
        assert strat.precise_note == ""
    else:
        assert strat.precise_note == INDEPENDENT_PRECISE_NOTE


def test_resolve_strategy_scale_y_matches_mode():
    """scale_y is True iff JOINT (fold Y-scaler is JOINT-fitting-only)."""
    for name, mode, _ in _EXPECTED_STRATEGIES:
        strat = resolve_multitarget_strategy(name)
        assert strat.scale_y == (mode == "JOINT")


def test_resolve_strategy_joint_params_for_boosters():
    assert resolve_multitarget_strategy("CatBoost").joint_params == {
        "loss_function": "MultiRMSE"
    }
    assert resolve_multitarget_strategy("XGBoost").joint_params == {
        "multi_strategy": "multi_output_tree"
    }
    # PLS/RF/MLP are native joint with no extra params.
    assert resolve_multitarget_strategy("PLS").joint_params == {}


def test_resolve_strategy_optional_joint_flag():
    """Only Lasso/ElasticNet advertise an optional JOINT MultiTask variant."""
    assert resolve_multitarget_strategy("Lasso").supports_optional_joint is True
    assert resolve_multitarget_strategy("ElasticNet").supports_optional_joint is True
    assert resolve_multitarget_strategy("SVR").supports_optional_joint is False
    assert resolve_multitarget_strategy("Ridge").supports_optional_joint is False


def test_resolve_strategy_svm_alias():
    """SVM (classification-side name) aliases to SVR."""
    assert resolve_multitarget_strategy("SVM").model_name == "SVR"
    assert resolve_multitarget_strategy("SVM").mode == "INDEPENDENT"


def test_resolve_strategy_unknown_raises():
    """UNKNOWN model RAISES -- never silently mislabels coupling."""
    with pytest.raises(ValueError):
        resolve_multitarget_strategy("NotAModel")
    with pytest.raises(ValueError):
        resolve_multitarget_strategy("")
    with pytest.raises(ValueError):
        resolve_multitarget_strategy(None)  # type: ignore[arg-type]


def test_independent_precise_note_substring_pinned():
    """Pin the exact honest-labeling substrings against drift."""
    assert (
        "separate per-target estimators under one shared searched configuration"
        in INDEPENDENT_PRECISE_NOTE
    )
    assert "not a coupled model, no correlation benefit" in INDEPENDENT_PRECISE_NOTE
    assert "run separate single-target searches" in INDEPENDENT_PRECISE_NOTE
    # Every INDEPENDENT strategy carries the SAME note object/string.
    for name, mode, _ in _EXPECTED_STRATEGIES:
        if mode == "INDEPENDENT":
            assert resolve_multitarget_strategy(name).precise_note == INDEPENDENT_PRECISE_NOTE


# --------------------------------------------------------------------------- #
# build_multitarget_estimator
# --------------------------------------------------------------------------- #
def test_build_pls_estimator_scale_false_and_capped():
    strat = resolve_multitarget_strategy("PLS")
    # requested 50 components with n_samples=20 -> capped at min(N-1, p)=19
    est = build_multitarget_estimator(strat, {"n_components": 50}, n_samples=20, n_features=25)
    assert isinstance(est, PLSRegression)
    assert est.n_components == 19
    assert est.scale is False  # matches legacy get_model("PLS")


def test_build_unwired_model_raises():
    # F3 wires the JOINT models; only the INDEPENDENT (F4) models still raise.
    for name in ["Ridge", "Lasso", "ElasticNet", "LightGBM", "SVR", "NeuralBoosted"]:
        strat = resolve_multitarget_strategy(name)
        with pytest.raises(NotImplementedError):
            build_multitarget_estimator(strat, {}, n_samples=30, n_features=10)


# --------------------------------------------------------------------------- #
# F3: JOINT estimator builders (RF, MLP, CatBoost MultiRMSE, XGBoost m.o.tree)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", ["RandomForest", "MLP", "CatBoost", "XGBoost"])
def test_build_joint_estimator_fits_scaled_3target_block(model_name, rng):
    """Each JOINT model fits a 3-target block on fold-scaled Y and predicts (n,3)."""
    n, p = 40, 12
    X = rng.standard_normal((n, p))
    base = X[:, :3] @ rng.standard_normal((3, 3))
    Y = base + 0.05 * rng.standard_normal((n, 3))
    # Fold-scaled Y (per-target zero-mean/unit-std), the block JOINT models fit on.
    Y_scaled = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    strat = resolve_multitarget_strategy(model_name)
    assert strat.mode == "JOINT"
    est = build_multitarget_estimator(strat, {}, n_samples=n, n_features=p)
    est.fit(X, Y_scaled)
    pred = np.asarray(est.predict(X))
    assert pred.shape == (n, 3)
    assert np.all(np.isfinite(pred))


def test_build_catboost_uses_multirmse_and_no_early_stopping():
    strat = resolve_multitarget_strategy("CatBoost")
    est = build_multitarget_estimator(strat, {}, n_samples=40, n_features=12)
    params = est.get_params()
    assert params.get("loss_function") == "MultiRMSE"
    # Booster early-stopping disabled for multi-Y v1: no overfitting-detector wait.
    assert params.get("od_wait") in (None, 0)
    assert params.get("early_stopping_rounds") in (None, 0)


def test_build_xgboost_uses_multi_output_tree_and_no_early_stopping():
    strat = resolve_multitarget_strategy("XGBoost")
    est = build_multitarget_estimator(strat, {}, n_samples=40, n_features=12)
    params = est.get_params()
    assert params.get("multi_strategy") == "multi_output_tree"
    # Booster early-stopping disabled for multi-Y v1.
    assert params.get("early_stopping_rounds") in (None, 0)


@pytest.mark.parametrize("model_name", ["RandomForest", "CatBoost", "XGBoost"])
def test_run_multitarget_joint_models_smoke(model_name):
    """End-to-end orchestrator smoke: JOINT tree/booster models score a 3-target
    block and return per-target metrics of shape (3,)."""
    rng = np.random.default_rng(7)
    n, p = 45, 10
    X = rng.standard_normal((n, p))
    base = X[:, :3] @ rng.standard_normal((3, 3))
    Y = base + 0.05 * rng.standard_normal((n, 3))
    out = run_multitarget_search(
        X,
        Y,
        [{"model_name": model_name, "params": {}}],
        cv="kfold",
        n_folds=3,
        target_names=["a", "b", "c"],
    )
    assert out.n_targets == 3
    best = out.best
    assert best.mode == "JOINT"
    assert best.scale_y is True
    assert best.metrics["q2"].shape == (3,)
    assert best.y_pred_pooled.shape[1] == 3


# --------------------------------------------------------------------------- #
# Consolidation pin: n_targets == 1 per-target Q2 == legacy r2_score
# --------------------------------------------------------------------------- #
def test_consolidation_pin_single_target_q2_equals_legacy_r2(rng):
    """At n_targets==1 the orchestrator's PLS per-target Q2 equals the legacy
    r2_score(all_y_test, all_y_pred) from cross_val_predict_pooled."""
    n, p = 50, 15
    X = rng.standard_normal((n, p))
    y = X[:, :3] @ rng.standard_normal(3) + 0.1 * rng.standard_normal(n)
    cv = build_cv_splitter("kfold", 5, "regression", random_state=7)

    out = run_multitarget_search(
        X, y, [{"model_name": "PLS", "params": {"n_components": 3}}], cv=cv
    )
    assert out.n_targets == 1
    best = out.best
    assert best.model_name == "PLS"
    assert best.mode == "JOINT"

    # Legacy single-Y reference: identical PLS estimator, raw-y pooled preds.
    legacy_pred = cross_val_predict_pooled(
        PLSRegression(n_components=3, scale=False), X, y, cv
    )
    legacy_r2 = r2_score(y, np.ravel(legacy_pred))

    # JOINT path fits on fold-scaled Y then inverse-transforms; PLS-1 is
    # scale-equivariant in Y so this matches the raw fit to numerical tol.
    assert best.metrics["q2"][0] == pytest.approx(legacy_r2, rel=1e-6, abs=1e-6)
    assert best.joint_q2 == pytest.approx(legacy_r2, rel=1e-6, abs=1e-6)


def test_single_target_byte_identity_pooled_preds_and_score(rng):
    """Gold-fixture byte-identity pin (T-17 consolidation guardrail).

    At n_targets==1 the orchestrator MUST route through the exact legacy raw-Y
    pooler, so pooled predictions are ``np.array_equal`` (NOT merely approx) to
    ``cross_val_predict_pooled`` and the per-target Q2 / joint Q2 EXACTLY equal
    the legacy ``r2_score``. pytest.approx cannot catch a ~1e-15 regression from
    the JOINT scale/inverse-transform round-trip; array_equal can.
    """
    n, p = 50, 15
    X = rng.standard_normal((n, p))
    y = X[:, :3] @ rng.standard_normal(3) + 0.1 * rng.standard_normal(n)
    cv = build_cv_splitter("kfold", 5, "regression", random_state=7)

    out = run_multitarget_search(
        X, y, [{"model_name": "PLS", "params": {"n_components": 3}}], cv=cv
    )
    best = out.best

    legacy_pred = cross_val_predict_pooled(
        PLSRegression(n_components=3, scale=False), X, y, cv
    )
    legacy_r2 = r2_score(y, np.ravel(legacy_pred))

    # (1) pooled predictions are BYTE-IDENTICAL, not approx.
    assert best.y_pred_pooled.shape == (n, 1)
    assert np.array_equal(best.y_pred_pooled.ravel(), np.ravel(legacy_pred))
    # pooled truths are the raw targets, byte-identical.
    assert np.array_equal(best.y_true_pooled.ravel(), np.asarray(y, dtype=float))
    # (2) score vectors byte-identical (exact ==, not approx).
    assert best.metrics["q2"][0] == legacy_r2
    assert best.joint_q2 == legacy_r2


# --------------------------------------------------------------------------- #
# Orchestrator: multi-target PLS smoke + ranking + guardrails
# --------------------------------------------------------------------------- #
def test_run_multitarget_pls_smoke(xy_multi):
    X, Y = xy_multi
    out = run_multitarget_search(
        X,
        Y,
        [
            {"model_name": "PLS", "params": {"n_components": 2}},
            {"model_name": "PLS", "params": {"n_components": 4}},
        ],
        cv="kfold",
        target_names=["collagen", "irsf", "cp"],
    )
    assert out.n_targets == 3
    assert len(out.results) == 2
    # ranked by joint_q2 descending
    assert out.results[0].joint_q2 >= out.results[1].joint_q2
    best = out.best
    assert best.mode == "JOINT"
    assert best.scale_y is True
    assert best.metrics["q2"].shape == (3,)
    assert best.metrics["target_names"] == ["collagen", "irsf", "cp"]
    # raw-unit RMSE: huge-scale target dwarfs the tiny-scale one
    assert best.metrics["rmse"][1] > 10.0 * best.metrics["rmse"][0]
    # joint_q2 == mean per-target q2
    assert best.joint_q2 == pytest.approx(np.mean(best.metrics["q2"]))


def test_run_multitarget_correlation_guardrail(xy_multi):
    X, Y = xy_multi
    out = run_multitarget_search(
        X, Y, [{"model_name": "PLS", "params": {"n_components": 3}}], cv="kfold"
    )
    # The 3 targets are all scalings of one base -> strongly correlated.
    assert out.correlation["mean_abs_corr"] > 0.9
    assert out.correlation["is_weak"] is False
    assert out.correlation["corr_matrix"].shape == (3, 3)


def test_run_rejects_non_grid_engine(xy_multi):
    """Bayesian/NSGA-II are 1-D-only; the orchestrator forces grid."""
    X, Y = xy_multi
    for method in ["unified", "nsga2", "bayesian"]:
        with pytest.raises(ValueError):
            run_multitarget_search(
                X,
                Y,
                [{"model_name": "PLS", "params": {"n_components": 2}}],
                optimization_method=method,
            )


def test_run_rejects_empty_configs(xy_multi):
    X, Y = xy_multi
    with pytest.raises(ValueError):
        run_multitarget_search(X, Y, [])


def test_run_high_components_capped_against_fold_size_kfold(rng):
    """Regression pin (Blocker 3): PLS n_components must be capped against the
    SMALLEST fold training set, not the full N. n=20, p=25, 5-fold: full-N cap
    would give 19, but a fold trains on 16 samples (upper bound < 19) and PLS
    raises ValueError inside CV. The orchestrator must not raise."""
    X = rng.standard_normal((20, 25))
    base = X[:, :2] @ rng.standard_normal((2, 1))
    Y = np.hstack([base + 0.05 * rng.standard_normal((20, 1)),
                   2.0 * base + 0.05 * rng.standard_normal((20, 1))])
    out = run_multitarget_search(
        X, Y, [{"model_name": "PLS", "params": {"n_components": 50}}],
        cv="kfold", n_folds=5,
    )
    assert out.best is not None
    assert out.best.metrics["q2"].shape == (2,)


def test_run_high_components_capped_against_fold_size_loo(rng):
    """LOO variant of the component-cap regression pin: LOO train folds are
    n-1, so the cap is min(n-2, p); a huge requested count must still run."""
    X = rng.standard_normal((15, 30))
    y = X[:, :2] @ rng.standard_normal(2) + 0.1 * rng.standard_normal(15)
    out = run_multitarget_search(
        X, y, [{"model_name": "PLS", "params": {"n_components": 99}}],
        cv="loo",
    )
    assert out.best is not None


def test_run_target_names_length_mismatch(xy_multi):
    X, Y = xy_multi
    with pytest.raises(ValueError):
        run_multitarget_search(
            X,
            Y,
            [{"model_name": "PLS", "params": {"n_components": 2}}],
            target_names=["only_one"],
        )
