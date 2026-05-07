"""Tests for OC hyperparameter round-2 additions.

Covers:
- LOF metric checkboxes (euclidean, manhattan, minkowski, cosine)
- LOF contamination 0.01 / 0.1 additions
- IsolationForest max_samples (auto, 256, 512)
- IsolationForest n_estimators 200 / 500
- EllipticEnvelope support_fraction (None, 0.5, 0.75)

Tests are purely backend (no Tkinter required).  The GUI var-map and
default-state machinery is tested via stand-alone dict equivalents rather than
instantiating the full application window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.contamination import get_one_class_model_grids
from spectral_predict.search import (
    _build_ee_custom_grid,
    _build_if_custom_grid,
    _build_lof_custom_grid,
    _resolve_one_class_model_grids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_grids() -> dict:
    return get_one_class_model_grids()


# ---------------------------------------------------------------------------
# Layer 1 — curated grid structure
# ---------------------------------------------------------------------------

class TestCuratedGridStructure:
    def test_if_grid_has_max_samples(self):
        grids = _default_grids()
        for entry in grids["IsolationForest"]:
            assert "max_samples" in entry, f"Missing max_samples in {entry}"

    def test_if_grid_has_200_and_500(self):
        grids = _default_grids()
        n_est_vals = {p["n_estimators"] for p in grids["IsolationForest"]}
        assert 200 in n_est_vals
        assert 500 in n_est_vals

    def test_if_grid_has_256_and_512_max_samples(self):
        grids = _default_grids()
        ms_vals = {p["max_samples"] for p in grids["IsolationForest"]}
        assert 256 in ms_vals
        assert 512 in ms_vals
        assert "auto" in ms_vals

    def test_ee_grid_has_support_fraction_entries(self):
        grids = _default_grids()
        sf_vals = {p.get("support_fraction") for p in grids["EllipticEnvelope"]}
        assert 0.5 in sf_vals
        assert 0.75 in sf_vals

    def test_lof_grid_has_additional_contamination(self):
        grids = _default_grids()
        contam_vals = {p["contamination"] for p in grids["LOF"]}
        assert 0.01 in contam_vals
        assert 0.1 in contam_vals
        assert 0.05 in contam_vals

    def test_lof_grid_has_metric(self):
        grids = _default_grids()
        for entry in grids["LOF"]:
            assert "metric" in entry, f"Missing metric in {entry}"

    def test_lof_grid_has_manhattan(self):
        grids = _default_grids()
        metric_vals = {p["metric"] for p in grids["LOF"]}
        assert "manhattan" in metric_vals
        assert "euclidean" in metric_vals


# ---------------------------------------------------------------------------
# Layer 4 — _build_*_custom_grid wiring tests
# ---------------------------------------------------------------------------

class TestBuildIfCustomGrid:
    def _default(self):
        return _default_grids()["IsolationForest"]

    def test_max_samples_auto_only(self):
        overrides = {"n_estimators": [100], "contamination": [0.05], "max_features": [1.0], "max_samples": ["auto"]}
        grid = _build_if_custom_grid(overrides, self._default())
        assert all(p["max_samples"] == "auto" for p in grid)

    def test_max_samples_256_present(self):
        overrides = {"n_estimators": [100], "contamination": [0.05], "max_features": [1.0], "max_samples": [256]}
        grid = _build_if_custom_grid(overrides, self._default())
        assert any(p["max_samples"] == 256 for p in grid)

    def test_max_samples_512_present(self):
        overrides = {"n_estimators": [100], "contamination": [0.05], "max_features": [1.0], "max_samples": [512]}
        grid = _build_if_custom_grid(overrides, self._default())
        assert any(p["max_samples"] == 512 for p in grid)

    def test_n_estimators_200_present(self):
        overrides = {"n_estimators": [200], "contamination": [0.05], "max_features": [1.0], "max_samples": ["auto"]}
        grid = _build_if_custom_grid(overrides, self._default())
        assert any(p["n_estimators"] == 200 for p in grid)

    def test_n_estimators_500_present(self):
        overrides = {"n_estimators": [500], "contamination": [0.05], "max_features": [1.0], "max_samples": ["auto"]}
        grid = _build_if_custom_grid(overrides, self._default())
        assert any(p["n_estimators"] == 500 for p in grid)

    def test_empty_grid_falls_back_to_default(self):
        grid = _build_if_custom_grid({}, self._default())
        assert len(grid) > 0

    def test_cartesian_product_has_correct_count(self):
        overrides = {
            "n_estimators": [100, 200],
            "contamination": [0.05],
            "max_features": [1.0],
            "max_samples": ["auto", 256],
        }
        grid = _build_if_custom_grid(overrides, self._default())
        assert len(grid) == 2 * 1 * 1 * 2  # 4 combos


class TestBuildEeCustomGrid:
    def _default(self):
        return _default_grids()["EllipticEnvelope"]

    def test_support_fraction_none_produces_entry_without_key(self):
        overrides = {"contamination": [0.05], "support_fraction": [None]}
        grid = _build_ee_custom_grid(overrides, self._default())
        assert len(grid) == 1
        assert "support_fraction" not in grid[0]

    def test_support_fraction_05_present(self):
        overrides = {"contamination": [0.05], "support_fraction": [0.5]}
        grid = _build_ee_custom_grid(overrides, self._default())
        assert any(p.get("support_fraction") == 0.5 for p in grid)

    def test_support_fraction_075_present(self):
        overrides = {"contamination": [0.05], "support_fraction": [0.75]}
        grid = _build_ee_custom_grid(overrides, self._default())
        assert any(p.get("support_fraction") == 0.75 for p in grid)

    def test_none_and_explicit_combo(self):
        overrides = {"contamination": [0.05], "support_fraction": [None, 0.5]}
        grid = _build_ee_custom_grid(overrides, self._default())
        assert len(grid) == 2
        sf_vals = [p.get("support_fraction") for p in grid]
        assert None in sf_vals
        assert 0.5 in sf_vals

    def test_empty_override_uses_default_grid_support_fractions(self):
        default = self._default()
        grid = _build_ee_custom_grid({}, default)
        assert len(grid) > 0
        # Should include all contamination values from default
        contam_vals = {p["contamination"] for p in grid}
        default_contam = {p["contamination"] for p in default}
        assert contam_vals == default_contam


class TestBuildLofCustomGrid:
    def _default(self):
        return _default_grids()["LOF"]

    def test_metric_manhattan_present(self):
        overrides = {"n_neighbors": [10], "contamination": [0.05], "metric": ["manhattan"]}
        grid = _build_lof_custom_grid(overrides, self._default())
        assert all(p["metric"] == "manhattan" for p in grid)

    def test_metric_cosine_present(self):
        overrides = {"n_neighbors": [10], "contamination": [0.05], "metric": ["cosine"]}
        grid = _build_lof_custom_grid(overrides, self._default())
        assert any(p["metric"] == "cosine" for p in grid)

    def test_metric_minkowski_present(self):
        overrides = {"n_neighbors": [10], "contamination": [0.05], "metric": ["minkowski"]}
        grid = _build_lof_custom_grid(overrides, self._default())
        assert any(p["metric"] == "minkowski" for p in grid)

    def test_contamination_001_present(self):
        overrides = {"n_neighbors": [10], "contamination": [0.01], "metric": ["euclidean"]}
        grid = _build_lof_custom_grid(overrides, self._default())
        assert any(p["contamination"] == 0.01 for p in grid)

    def test_contamination_01_present(self):
        overrides = {"n_neighbors": [10], "contamination": [0.1], "metric": ["euclidean"]}
        grid = _build_lof_custom_grid(overrides, self._default())
        assert any(p["contamination"] == 0.1 for p in grid)

    def test_empty_override_uses_default_metrics(self):
        default = self._default()
        grid = _build_lof_custom_grid({}, default)
        metric_vals = {p["metric"] for p in grid}
        # Default grid has euclidean and manhattan
        assert "euclidean" in metric_vals
        assert "manhattan" in metric_vals

    def test_all_combos_have_metric_key(self):
        overrides = {"n_neighbors": [10, 20], "contamination": [0.05], "metric": ["euclidean", "manhattan"]}
        grid = _build_lof_custom_grid(overrides, self._default())
        assert all("metric" in p for p in grid)
        assert len(grid) == 4  # 2 * 1 * 2


# ---------------------------------------------------------------------------
# Default-state preservation pin
# ---------------------------------------------------------------------------

class TestDefaultStatePreservation:
    """When all GUI checkboxes are at their default values, the resolved grid
    must equal the curated grid from get_one_class_model_grids().

    We simulate 'all at default' by passing no oc_model_param_overrides
    (None), which is what _collect_one_class_model_param_overrides() returns
    when every card matches defaults.
    """

    def test_if_default_state_yields_curated_grid(self):
        curated = _default_grids()["IsolationForest"]
        resolved = _resolve_one_class_model_grids(
            enabled_models=["IsolationForest"],
            oc_model_param_overrides=None,
        )
        assert resolved["IsolationForest"] == curated

    def test_ee_default_state_yields_curated_grid(self):
        curated = _default_grids()["EllipticEnvelope"]
        resolved = _resolve_one_class_model_grids(
            enabled_models=["EllipticEnvelope"],
            oc_model_param_overrides=None,
        )
        assert resolved["EllipticEnvelope"] == curated

    def test_lof_default_state_yields_curated_grid(self):
        curated = _default_grids()["LOF"]
        resolved = _resolve_one_class_model_grids(
            enabled_models=["LOF"],
            oc_model_param_overrides=None,
        )
        assert resolved["LOF"] == curated

    def test_ocsvm_default_state_yields_curated_grid(self):
        curated = _default_grids()["OneClassSVM"]
        resolved = _resolve_one_class_model_grids(
            enabled_models=["OneClassSVM"],
            oc_model_param_overrides=None,
        )
        assert resolved["OneClassSVM"] == curated

    def test_simca_default_state_yields_curated_grid(self):
        curated = _default_grids()["PCA-SIMCA"]
        resolved = _resolve_one_class_model_grids(
            enabled_models=["PCA-SIMCA"],
            oc_model_param_overrides=None,
        )
        assert resolved["PCA-SIMCA"] == curated


# ---------------------------------------------------------------------------
# Per-parameter wiring pins (override → resolved grid contains expected config)
# ---------------------------------------------------------------------------

class TestPerParameterWiring:
    def test_if_max_samples_256_wired(self):
        overrides = {
            "IsolationForest": {
                "n_estimators": [100],
                "contamination": [0.05],
                "max_features": [1.0],
                "max_samples": [256],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["IsolationForest"],
            oc_model_param_overrides=overrides,
        )
        assert any(p["max_samples"] == 256 for p in resolved["IsolationForest"])

    def test_if_n_estimators_500_wired(self):
        overrides = {
            "IsolationForest": {
                "n_estimators": [500],
                "contamination": [0.05],
                "max_features": [1.0],
                "max_samples": ["auto"],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["IsolationForest"],
            oc_model_param_overrides=overrides,
        )
        assert any(p["n_estimators"] == 500 for p in resolved["IsolationForest"])

    def test_ee_support_fraction_05_wired(self):
        overrides = {
            "EllipticEnvelope": {
                "contamination": [0.05],
                "support_fraction": [0.5],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["EllipticEnvelope"],
            oc_model_param_overrides=overrides,
        )
        assert any(p.get("support_fraction") == 0.5 for p in resolved["EllipticEnvelope"])

    def test_ee_support_fraction_075_wired(self):
        overrides = {
            "EllipticEnvelope": {
                "contamination": [0.05],
                "support_fraction": [0.75],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["EllipticEnvelope"],
            oc_model_param_overrides=overrides,
        )
        assert any(p.get("support_fraction") == 0.75 for p in resolved["EllipticEnvelope"])

    def test_lof_metric_manhattan_wired(self):
        overrides = {
            "LOF": {
                "n_neighbors": [10],
                "contamination": [0.05],
                "metric": ["manhattan"],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["LOF"],
            oc_model_param_overrides=overrides,
        )
        assert any(p["metric"] == "manhattan" for p in resolved["LOF"])

    def test_lof_metric_cosine_wired(self):
        overrides = {
            "LOF": {
                "n_neighbors": [10],
                "contamination": [0.05],
                "metric": ["cosine"],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["LOF"],
            oc_model_param_overrides=overrides,
        )
        assert any(p["metric"] == "cosine" for p in resolved["LOF"])

    def test_lof_contamination_001_wired(self):
        overrides = {
            "LOF": {
                "n_neighbors": [10],
                "contamination": [0.01],
                "metric": ["euclidean"],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["LOF"],
            oc_model_param_overrides=overrides,
        )
        assert any(p["contamination"] == 0.01 for p in resolved["LOF"])

    def test_lof_contamination_01_wired(self):
        overrides = {
            "LOF": {
                "n_neighbors": [10],
                "contamination": [0.1],
                "metric": ["euclidean"],
            }
        }
        resolved = _resolve_one_class_model_grids(
            enabled_models=["LOF"],
            oc_model_param_overrides=overrides,
        )
        assert any(p["contamination"] == 0.1 for p in resolved["LOF"])


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_oc_data():
    """Synthetic NIR-like one-class fixture.

    Two-class: 'clean' (tight cluster) and 'contaminated' (scattered).
    Enough samples for 3-fold CV.
    """
    rng = np.random.RandomState(42)
    n_features = 50
    X_clean = rng.randn(30, n_features) * 0.3
    X_contam = rng.randn(8, n_features) + 4.0
    X = np.vstack([X_clean, X_contam])
    y = np.array(["clean"] * 30 + ["contaminated"] * 8)
    return X, y


class TestEndToEndSmoke:
    def test_run_one_class_search_with_expanded_grid_completes(self, synthetic_oc_data):
        """run_one_class_search with the new expanded IF/EE/LOF grids completes
        without raising and returns > 0 result rows."""
        from spectral_predict.search import run_one_class_search

        X, y = synthetic_oc_data

        result_df = run_one_class_search(
            X=X,
            y=y,
            inlier_class_label="clean",
            folds=3,
            preprocessing_methods=["raw"],
            tier="quick",
            enabled_models=["IsolationForest", "EllipticEnvelope", "LOF"],
        )
        assert result_df is not None, "run_one_class_search returned None"
        assert len(result_df) > 0, "result_df has 0 rows"

    def test_run_one_class_search_with_lof_manhattan_does_not_crash(self, synthetic_oc_data):
        """LOF with metric=manhattan runs end-to-end without error."""
        from spectral_predict.search import run_one_class_search

        X, y = synthetic_oc_data

        overrides = {
            "LOF": {
                "n_neighbors": [5],
                "contamination": [0.05],
                "metric": ["manhattan"],
            }
        }
        result_df = run_one_class_search(
            X=X,
            y=y,
            inlier_class_label="clean",
            folds=3,
            preprocessing_methods=["raw"],
            tier="quick",
            enabled_models=["LOF"],
            oc_model_param_overrides=overrides,
        )
        assert result_df is not None
        assert len(result_df) > 0

    def test_run_one_class_search_with_ee_support_fraction_does_not_crash(self, synthetic_oc_data):
        """EllipticEnvelope with support_fraction=0.75 runs end-to-end without error."""
        from spectral_predict.search import run_one_class_search

        X, y = synthetic_oc_data

        overrides = {
            "EllipticEnvelope": {
                "contamination": [0.05],
                "support_fraction": [0.75],
            }
        }
        result_df = run_one_class_search(
            X=X,
            y=y,
            inlier_class_label="clean",
            folds=3,
            preprocessing_methods=["raw"],
            tier="quick",
            enabled_models=["EllipticEnvelope"],
            oc_model_param_overrides=overrides,
        )
        assert result_df is not None
        assert len(result_df) > 0
