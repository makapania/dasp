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
