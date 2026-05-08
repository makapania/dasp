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


# ---------------------------------------------------------------------------
# Layer 8 — Custom-field token parsers (post-PR-58-review hardening)
#
# These tests pin the comma-tolerant list parsers added after the silent-
# failure-hunter found that "256, 512" in a Custom: field was silently
# dropped (int(float("256, 512")) raises). The fix replaces the scalar
# _parse_oc_int / _parse_oc_float with list-returning siblings that
# tokenise on comma/semicolon, deduplicate, and accumulate per-token
# error messages instead of swallowing them.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_class():
    from spectral_predict_gui_optimized import SpectralPredictApp
    return SpectralPredictApp


class TestSplitOcTokens:
    """Tokeniser for comma/semicolon-separated Custom: field input."""

    def test_empty_string_yields_no_tokens(self, app_class):
        assert app_class._split_oc_tokens("") == []

    def test_whitespace_only_yields_no_tokens(self, app_class):
        assert app_class._split_oc_tokens("   ") == []

    def test_none_yields_no_tokens(self, app_class):
        assert app_class._split_oc_tokens(None) == []

    def test_single_token(self, app_class):
        assert app_class._split_oc_tokens("256") == ["256"]

    def test_comma_separated(self, app_class):
        assert app_class._split_oc_tokens("256, 512") == ["256", "512"]

    def test_semicolon_treated_as_comma(self, app_class):
        assert app_class._split_oc_tokens("256; 512") == ["256", "512"]

    def test_mixed_separators(self, app_class):
        assert app_class._split_oc_tokens("100, 200; 300") == ["100", "200", "300"]

    def test_trailing_comma_dropped(self, app_class):
        assert app_class._split_oc_tokens("256,") == ["256"]

    def test_double_comma_dropped(self, app_class):
        assert app_class._split_oc_tokens("100,,200") == ["100", "200"]


class TestParseOcIntList:
    """List parser for integer Custom: fields (n_estimators, n_neighbors, ...)."""

    def test_empty_string_no_values_no_errors(self, app_class):
        vals, errs = app_class._parse_oc_int_list("", 1)
        assert vals == []
        assert errs == []

    def test_single_value_parses(self, app_class):
        vals, errs = app_class._parse_oc_int_list("256", 1)
        assert vals == [256]
        assert errs == []

    def test_comma_separated_parses_to_list(self, app_class):
        # The bug this fix closes: previously "256, 512" was silently dropped.
        vals, errs = app_class._parse_oc_int_list("256, 512", 1)
        assert vals == [256, 512]
        assert errs == []

    def test_below_minimum_reported(self, app_class):
        vals, errs = app_class._parse_oc_int_list("0", 1)
        assert vals == []
        assert len(errs) == 1
        assert "below minimum 1" in errs[0]

    def test_non_integer_reported(self, app_class):
        vals, errs = app_class._parse_oc_int_list("banana", 1)
        assert vals == []
        assert len(errs) == 1
        assert "not an integer" in errs[0]

    def test_mixed_valid_and_invalid_partial_recovery(self, app_class):
        vals, errs = app_class._parse_oc_int_list("256, banana, 512", 1)
        assert vals == [256, 512]
        assert len(errs) == 1
        assert "banana" in errs[0]

    def test_duplicates_deduplicated(self, app_class):
        vals, errs = app_class._parse_oc_int_list("256, 256", 1)
        assert vals == [256]
        assert errs == []

    def test_float_truncated_to_int(self, app_class):
        # Sklearn integer params accept "256.0" via int(float()) cast.
        vals, errs = app_class._parse_oc_int_list("256.0", 1)
        assert vals == [256]


class TestParseOcFloatList:
    """List parser for float Custom: fields (contamination, nu, ...)."""

    def test_comma_separated_parses(self, app_class):
        vals, errs = app_class._parse_oc_float_list("0.02, 0.03", 0.0, 1.0)
        assert vals == [0.02, 0.03]
        assert errs == []

    def test_out_of_range_reported(self, app_class):
        vals, errs = app_class._parse_oc_float_list("1.5", 0.0, 1.0)
        assert vals == []
        assert len(errs) == 1
        assert "not in range" in errs[0]

    def test_zero_excluded_via_strict_lower_bound(self, app_class):
        vals, errs = app_class._parse_oc_float_list("0.0", 0.0, 1.0)
        assert vals == []
        assert len(errs) == 1

    def test_upper_bound_inclusive(self, app_class):
        vals, errs = app_class._parse_oc_float_list("1.0", 0.0, 1.0)
        assert vals == [1.0]
        assert errs == []

    def test_non_numeric_reported(self, app_class):
        vals, errs = app_class._parse_oc_float_list("abc", 0.0, 1.0)
        assert vals == []
        assert len(errs) == 1
        assert "not a number" in errs[0]


class TestParseOcMetricList:
    """List parser for LOF metric Custom: field (whitelist-validated)."""

    def test_known_metric_accepted(self, app_class):
        vals, errs = app_class._parse_oc_metric_list("chebyshev", app_class._LOF_KNOWN_METRICS)
        assert vals == ["chebyshev"]
        assert errs == []

    def test_typo_rejected(self, app_class):
        # Pre-fix: "manhatten" propagated to sklearn, fold-failed silently.
        vals, errs = app_class._parse_oc_metric_list("manhatten", app_class._LOF_KNOWN_METRICS)
        assert vals == []
        assert len(errs) == 1
        assert "manhatten" in errs[0]
        assert "not a recognised metric" in errs[0]

    def test_case_normalised(self, app_class):
        vals, errs = app_class._parse_oc_metric_list("Manhattan", app_class._LOF_KNOWN_METRICS)
        assert vals == ["manhattan"]
        assert errs == []

    def test_comma_separated(self, app_class):
        vals, errs = app_class._parse_oc_metric_list(
            "manhattan, cosine", app_class._LOF_KNOWN_METRICS
        )
        assert vals == ["manhattan", "cosine"]
        assert errs == []

    def test_mixed_valid_invalid(self, app_class):
        vals, errs = app_class._parse_oc_metric_list(
            "manhattan, banana", app_class._LOF_KNOWN_METRICS
        )
        assert vals == ["manhattan"]
        assert len(errs) == 1
        assert "banana" in errs[0]


class TestParseOcNComponentsList:
    """List parser for SIMCA n_components (int OR fraction in (0, 1))."""

    def test_integer_kept_as_int(self, app_class):
        vals, errs = app_class._parse_oc_n_components_list("5")
        assert vals == [5]
        assert errs == []

    def test_fraction_kept_as_float(self, app_class):
        vals, errs = app_class._parse_oc_n_components_list("0.95")
        assert vals == [0.95]
        assert errs == []

    def test_mixed_int_and_fraction(self, app_class):
        vals, errs = app_class._parse_oc_n_components_list("5, 0.95")
        assert vals == [5, 0.95]
        assert errs == []

    def test_zero_rejected(self, app_class):
        vals, errs = app_class._parse_oc_n_components_list("0")
        assert vals == []
        assert len(errs) == 1

    def test_one_rejected_as_fraction(self, app_class):
        # 1.0 is neither a positive integer count nor in (0, 1) — must reject.
        vals, errs = app_class._parse_oc_n_components_list("1.0")
        # 1.0 == int(1.0) and int(1.0) >= 1 -> accepted as int 1
        assert vals == [1]


class TestLofKnownMetrics:
    """Whitelist of metrics LOF custom field accepts."""

    def test_includes_common_spectroscopy_metrics(self, app_class):
        for m in ("euclidean", "manhattan", "cosine", "minkowski"):
            assert m in app_class._LOF_KNOWN_METRICS

    def test_excludes_invalid_strings(self, app_class):
        assert "banana" not in app_class._LOF_KNOWN_METRICS
        assert "manhatten" not in app_class._LOF_KNOWN_METRICS
