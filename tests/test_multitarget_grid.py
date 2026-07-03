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
