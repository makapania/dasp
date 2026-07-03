"""Tests for the T-17 multi-target varsel adapters, guards, and grid orchestration."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20260702)


@pytest.fixture
def xy_multi(rng):
    n, p = 50, 40
    X = rng.standard_normal((n, p))
    base = X[:, :4] @ rng.standard_normal((4, 3))
    Y = base + 0.05 * rng.standard_normal((n, 3))
    wl = np.linspace(1000.0, 2000.0, p)
    return X, Y, wl


def test_ipls_selection_rejects_2d_y(xy_multi):
    from spectral_predict.variable_selection import ipls_selection

    X, Y, _wl = xy_multi
    with pytest.raises(NotImplementedError):
        ipls_selection(X, Y)


def test_ipls_selection_single_y_still_works(xy_multi):
    from spectral_predict.variable_selection import ipls_selection

    X, Y, _wl = xy_multi
    # single-column 2-D and 1-D must NOT raise (guard fires only on >1 column).
    out_1d = ipls_selection(X, Y[:, 0])
    out_col = ipls_selection(X, Y[:, [0]])
    assert out_1d is not None
    assert out_col is not None


def test_vcpa_iriv_rejects_2d_y(xy_multi):
    from spectral_predict.wavelength_selection import vcpa_iriv

    X, Y, _wl = xy_multi
    with pytest.raises(NotImplementedError):
        vcpa_iriv(X, Y, n_outer_iterations=1, n_inner_iterations=2, binary_matrix_samples=4)


def test_parse_ipls_subset_limit():
    from spectral_predict.multitarget_grid import _parse_ipls_subset_limit

    assert _parse_ipls_subset_limit("Top 5") == 5
    assert _parse_ipls_subset_limit("Top 20") == 20
    assert _parse_ipls_subset_limit("All") is None


def test_interval_subset_adapter_returns_truncated_subsets(xy_multi):
    from spectral_predict.multitarget_grid import _interval_subset_adapter

    X, Y, wl = xy_multi
    subs = _interval_subset_adapter(
        "ipls_forward", X, Y, wl, ipls_subset_limit="Top 5"
    )
    assert 1 <= len(subs) <= 5
    for s in subs:
        assert s["method"] == "ipls_forward"
        assert isinstance(s["indices"], np.ndarray)
        assert s["indices"].size >= 1
        assert s["indices"].size < X.shape[1] or "iPLS" in s["tag"]
        assert isinstance(s["tag"], str)


def test_verify_spa_multi_y_safe_true_on_informative_block(xy_multi):
    from spectral_predict.multitarget_grid import verify_spa_multi_y_safe

    X, Y, _wl = xy_multi
    assert verify_spa_multi_y_safe(X, Y, n_features=5) is True


def test_verify_spa_multi_y_safe_shape_and_finiteness():
    from spectral_predict.variable_selection import spa_selection

    rng = np.random.default_rng(9)
    X = rng.standard_normal((40, 20))
    Y = X[:, :3] @ rng.standard_normal((3, 2)) + 0.05 * rng.standard_normal((40, 2))
    imp = np.asarray(spa_selection(X, Y, n_features=5), dtype=float)
    assert imp.shape == (20,)
    assert np.all(np.isfinite(imp))


def test_importances_to_subsets_top_n_filtered(xy_multi):
    from spectral_predict.multitarget_grid import _importances_to_subsets

    X, _Y, _wl = xy_multi
    imp = np.arange(X.shape[1], dtype=float)  # 40 features, strictly increasing
    subs = _importances_to_subsets(
        imp, "spa", variable_counts=[5, 10, 999], n_features_sub=X.shape[1]
    )
    # 999 >= 40 is filtered out; 5 and 10 remain.
    sizes = sorted(s["indices"].size for s in subs)
    assert sizes == [5, 10]
    top5 = next(s for s in subs if s["indices"].size == 5)
    assert set(top5["indices"].tolist()) == {35, 36, 37, 38, 39}  # highest importances
    assert top5["method"] == "spa"
    assert "top5" in top5["tag"]


def test_model_independent_importances_spa_and_ga(xy_multi):
    from spectral_predict.multitarget_grid import _model_independent_importances

    X, Y, _wl = xy_multi
    imp_spa = _model_independent_importances("spa", X, Y)
    assert imp_spa is not None and imp_spa.shape == (X.shape[1],)
    imp_ga = _model_independent_importances("ga", X, Y)
    assert imp_ga is not None and imp_ga.shape == (X.shape[1],)


def test_importance_reference_fit_tree_model(xy_multi):
    from spectral_predict.multitarget_grid import _importance_reference_fit

    X, Y, _wl = xy_multi
    imp = _importance_reference_fit("RandomForest", X, Y, min_fold_train=X.shape[0] - 1)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))


def test_fipls_spa_preserves_2d_y_when_spa_safe(xy_multi):
    from spectral_predict.multitarget_grid import verify_spa_multi_y_safe
    from spectral_predict.variable_selection import fipls_spa_selection

    X, Y, wl = xy_multi
    if not verify_spa_multi_y_safe(X, Y, n_features=5):
        pytest.skip("SPA not 2-D-safe in this environment; fipls_spa stays skipped")
    imp = np.asarray(fipls_spa_selection(X, Y, wl), dtype=float)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))
