"""Backend contract tests for the T-31 Phase D2 decision-view provider."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.search import (
    build_multiclass_decision_view,
    run_multiclass_simca_search,
)


def _synthetic(K=3, n=45, p=40, seed=0):
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for k in range(K):
        blocks.append(rng.normal(k * 4.0, 1.0, size=(n, p)))
        labels += [f"C{k}"] * n
    X = pd.DataFrame(np.vstack(blocks), columns=[f"w{j}" for j in range(p)])
    y = pd.Series(labels)
    return X, y


_RAW_CFG = {
    "method": "raw", "name": "raw", "deriv": None, "window": None,
    "polyorder": None, "baseline_method": None, "baseline_params": None,
    "smoothing": False, "smoothing_window": 17, "smoothing_polyorder": 2,
}


def test_decision_view_shapes_and_labels():
    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_RAW_CFG, alpha=0.05,
        n_components=0.99,
    )
    assert view["reason"] == ""
    K = len(view["classes"])
    n = len(X)
    assert view["p_values"].shape == (n, K)
    assert view["accept"].shape == (n, K)
    assert view["accept"].dtype == bool
    assert len(view["labels"]) == n
    allowed = set(view["classes"]) | {"multiple", "novel"}
    assert set(view["labels"]).issubset(allowed)
    assert len(view["sample_ids"]) == n


def test_decision_view_wold_aggregates_present():
    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_RAW_CFG, n_components=0.99,
    )
    wold = view["wold"]
    assert wold is not None
    p = X.shape[1]
    assert wold["modeling_power_agg"].shape == (p,)
    assert wold["discriminating_power_agg"].shape == (p,)


def test_decision_view_matches_direct_model_fit():
    """The provider's decision matrix equals a direct MultiClassClassModel fit."""
    from spectral_predict.simca import MultiClassClassModel

    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_RAW_CFG, alpha=0.05,
        n_components=0.99,
    )
    # raw preprocessing => X unchanged, so a direct fit must reproduce P/A.
    m = MultiClassClassModel(
        engine="pca-simca", alpha=0.05, n_components=0.99, scaling="per_class",
    ).fit(X.values.astype(np.float64), y.values)
    P, A = m.decision_matrix(X.values.astype(np.float64))
    np.testing.assert_allclose(view["p_values"], P, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(view["accept"], A)


def test_search_attaches_top_decision_view_when_requested():
    X, y = _synthetic()
    df = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        n_components=0.99, cv_splits=3, compute_top_decision_view=True,
    )
    assert "top_decision_view" in df.attrs
    view = df.attrs["top_decision_view"]
    assert view["p_values"].shape[0] == len(X)
    assert len(view["classes"]) >= 2


def test_search_default_has_no_decision_view_and_returns_dataframe():
    X, y = _synthetic()
    df = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        n_components=0.99, cv_splits=3,
    )
    assert isinstance(df, pd.DataFrame)
    assert "top_decision_view" not in df.attrs
