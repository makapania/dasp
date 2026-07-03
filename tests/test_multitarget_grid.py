"""Tests for the T-17 multi-target grid orchestrator (multitarget_grid.py)."""
from __future__ import annotations

import numpy as np
import pytest


def test_preprocess_configs_raw_and_snv():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({"raw": True, "snv": True})
    names = [c["name"] for c in cfgs]
    assert "raw" in names
    assert "snv" in names
    raw = next(c for c in cfgs if c["name"] == "raw")
    assert raw["deriv"] is None and raw["window"] is None and raw["polyorder"] is None


def test_preprocess_configs_sg_polyorder_pairing():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs(
        {"sg1": True, "sg2": True}, window_sizes=[11]
    )
    derivs = {(c["deriv"], c["polyorder"]) for c in cfgs if c["name"] == "deriv"}
    assert (1, 2) in derivs  # sg1 -> deriv 1 / poly 2
    assert (2, 3) in derivs  # sg2 -> deriv 2 / poly 3


def test_preprocess_configs_autoscale_doubling():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({"raw": True}, autoscale=True)
    names = [c["name"] for c in cfgs]
    assert "raw" in names                 # without-autoscale copy
    assert "raw+autoscale" in names       # with-autoscale copy
    assert any(c.get("autoscale") is True for c in cfgs)
    assert any(c.get("autoscale") is False for c in cfgs)


def test_preprocess_configs_baseline_doubling():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs(
        {"raw": True}, baseline_method="als", baseline_params={"lam": 1e5}
    )
    names = [c["name"] for c in cfgs]
    assert "raw" in names
    assert "als+raw" in names
    without = next(c for c in cfgs if c["name"] == "raw")
    assert without["baseline_method"] is None


def test_preprocess_configs_empty_falls_back_to_raw():
    from spectral_predict.multitarget_grid import build_multitarget_preprocess_configs

    cfgs = build_multitarget_preprocess_configs({})
    assert [c["name"] for c in cfgs] == ["raw"]


@pytest.fixture
def rng():
    return np.random.default_rng(4242)


@pytest.fixture
def grid_xy(rng):
    n, p = 45, 30
    X = rng.standard_normal((n, p))
    base = X[:, :4] @ rng.standard_normal((4, 2))
    Y = base + 0.05 * rng.standard_normal((n, 2))
    wl = np.linspace(1000.0, 2000.0, p)
    return X, Y, wl


def test_grid_search_grid_only_assert(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    with pytest.raises(ValueError):
        run_multitarget_grid_search(
            X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
            preprocessing_methods={"raw": True}, autoscale=False,
            variable_selection_methods=[], tier="quick",
            cv="kfold", n_folds=3, n_repeats=1, optimization_method="unified",
        )


def test_grid_search_end_to_end_ranks_and_skips(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True, "snv": True}, autoscale=False,
        variable_selection_methods=["ipls_forward", "uve"], variable_counts=[5, 10],
        ipls_subset_limit="Top 3", tier="quick",
        cv="kfold", n_folds=3, n_repeats=1, random_state=42,
    )
    # Ranked, NaN-safe: best has a finite joint_q2.
    assert len(out.results) >= 4
    assert np.isfinite(out.results[0].joint_q2)
    # Both preprocessing states and >1 varsel tag appear.
    assert {r.preprocessing for r in out.results} >= {"raw", "snv"}
    assert any(r.varsel_method == "ipls_forward" for r in out.results)
    assert any(r.varsel_method == "full" for r in out.results)
    # UVE skip surfaced.
    assert "uve" in out.skipped


def test_grid_search_progress_callback_dict_shape(grid_xy):
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = grid_xy
    seen = []
    run_multitarget_grid_search(
        X, Y, model_names=["PLS"], target_names=["a", "b"], wavelengths=wl,
        preprocessing_methods={"raw": True}, autoscale=False,
        variable_selection_methods=[], tier="quick",
        cv="kfold", n_folds=3, n_repeats=1,
        progress_callback=lambda info: seen.append(info),
    )
    assert seen
    last = seen[-1]
    assert set(["message", "current", "total"]).issubset(last.keys())
    assert "best_model" in last


def test_dedup_keyset_equals_consumed_no_config_lost():
    from spectral_predict.multitarget_grid import _dedup_model_configs

    # Two RF configs differing only in a consumed key (bootstrap) must NOT collapse.
    class _E: pass
    grids = {"RandomForest": [
        (_E(), {"n_estimators": 50, "bootstrap": True}),
        (_E(), {"n_estimators": 50, "bootstrap": False}),
        (_E(), {"n_estimators": 50, "bootstrap": True}),  # exact dup -> collapses
    ]}
    out = _dedup_model_configs(grids)
    assert len(out) == 2
    assert {c["params"]["bootstrap"] for c in out} == {True, False}
