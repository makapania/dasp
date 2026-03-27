"""Tests for advanced baseline correction via pybaselines.

Covers:
- Import guard / HAS_PYBASELINES flag
- sklearn estimator contract (fit, clone, get_params/set_params, Pipeline)
- Algorithm correctness for Tier 1 methods
- Input validation
- Shape and determinism
- Degenerate-row robustness
- auto_optimize
- Registry integrity
- Helper functions
"""

from __future__ import annotations

import importlib
import sys
import unittest.mock as mock
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

TIER_1_ALGORITHMS = ["arpls", "iarpls", "drpls", "imor", "snip", "modpoly", "imodpoly"]
REQUIRED_REGISTRY_FIELDS = {"display_name", "category", "tier", "default_params", "param_info", "supports_optimize"}

N_FEATURES = 200
N_SAMPLES = 10


def _make_synthetic_spectrum(n: int = N_FEATURES, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (spectrum_with_baseline, true_signal) where baseline is quadratic."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    # True signal: two narrow Gaussians
    peak1 = 0.8 * np.exp(-((x - 0.3) ** 2) / (2 * 0.015**2))
    peak2 = 0.6 * np.exp(-((x - 0.7) ** 2) / (2 * 0.012**2))
    true_signal = peak1 + peak2
    # Quadratic baseline
    baseline = 0.001 * (np.arange(n) ** 2) / n**2 * n + 0.05 * x + 0.2
    noise = rng.normal(0, 0.005, n)
    spectrum = true_signal + baseline + noise
    return spectrum, true_signal


def _make_batch(n_samples: int = N_SAMPLES, n_features: int = N_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_with_baseline, X_true_signal) batch arrays."""
    spectra = []
    signals = []
    for i in range(n_samples):
        sp, sig = _make_synthetic_spectrum(n_features, seed=i)
        spectra.append(sp)
        signals.append(sig)
    return np.array(spectra), np.array(signals)


# ---------------------------------------------------------------------------
# 1. Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Verify HAS_PYBASELINES flag and ImportError behavior when pybaselines is absent."""

    def test_has_pybaselines_is_bool(self):
        from spectral_predict import baseline_advanced

        assert isinstance(baseline_advanced.HAS_PYBASELINES, bool)

    def test_transform_raises_import_error_when_pybaselines_missing(self):
        """Patching HAS_PYBASELINES=False must make transform raise ImportError."""
        from spectral_predict import baseline_advanced

        transformer = baseline_advanced.BaselineAdvanced(method="arpls")
        X = np.ones((3, 50))
        transformer.fit(X)

        with mock.patch.object(baseline_advanced, "HAS_PYBASELINES", False):
            with pytest.raises(ImportError, match="pybaselines"):
                transformer.transform(X)

    def test_has_pybaselines_false_when_module_absent(self):
        """Re-importing with pybaselines blocked sets HAS_PYBASELINES=False."""
        original_modules = sys.modules.copy()
        try:
            # Remove cached copies so the import block re-runs
            for key in list(sys.modules):
                if "pybaselines" in key or "baseline_advanced" in key:
                    del sys.modules[key]

            with mock.patch.dict(sys.modules, {"pybaselines": None}):
                import spectral_predict.baseline_advanced as ba_fresh

                assert ba_fresh.HAS_PYBASELINES is False
        finally:
            # Restore original sys.modules state
            sys.modules.clear()
            sys.modules.update(original_modules)


# ---------------------------------------------------------------------------
# 2. sklearn contract
# ---------------------------------------------------------------------------


class TestSklearnContract:
    """Verify BaselineAdvanced satisfies the sklearn estimator API."""

    def test_fit_sets_n_features_in(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced

        X = np.ones((5, N_FEATURES))
        est = BaselineAdvanced(method="arpls")
        est.fit(X)
        assert est.n_features_in_ == N_FEATURES

    def test_fit_sets_is_fitted(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced

        X = np.ones((5, N_FEATURES))
        est = BaselineAdvanced(method="arpls")
        est.fit(X)
        assert est._is_fitted is True

    def test_fit_returns_self(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced

        X = np.ones((3, 50))
        est = BaselineAdvanced()
        result = est.fit(X)
        assert result is est

    def test_get_params_returns_method_and_wavenumbers(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced

        wn = np.linspace(400, 4000, N_FEATURES)
        est = BaselineAdvanced(method="snip", wavenumbers=wn)
        params = est.get_params()
        assert params["method"] == "snip"
        np.testing.assert_array_equal(params["wavenumbers"], wn)

    def test_set_params_roundtrip(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced

        est = BaselineAdvanced(method="arpls")
        est.set_params(method="modpoly")
        assert est.method == "modpoly"
        assert est.get_params()["method"] == "modpoly"

    def test_clone_preserves_params(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced

        wn = np.linspace(400, 4000, N_FEATURES)
        est = BaselineAdvanced(method="snip", wavenumbers=wn)
        cloned = clone(est)
        assert cloned.method == est.method
        np.testing.assert_array_equal(cloned.wavenumbers, est.wavenumbers)

    @pytest.mark.integration
    def test_works_inside_pipeline(self):
        """BaselineAdvanced must be composable inside a sklearn Pipeline."""
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=8, n_features=N_FEATURES)
        pipe = Pipeline(
            [
                ("baseline", BaselineAdvanced(method="arpls", lam=1e5)),
                ("scaler", StandardScaler()),
            ]
        )
        X_out = pipe.fit_transform(X)
        assert X_out.shape == X.shape


# ---------------------------------------------------------------------------
# 3. Algorithm correctness — Tier 1
# ---------------------------------------------------------------------------


@pytest.mark.numerical
class TestAlgorithmCorrectness:
    """Verify Tier 1 algorithms reduce baseline MSE vs. the true signal."""

    def _assert_improves(self, method: str, extra_params: dict) -> None:
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, X_true = _make_batch()
        est = BaselineAdvanced(method=method, **extra_params)
        X_corrected = est.fit_transform(X)

        mse_before = float(np.mean((X - X_true) ** 2))
        mse_after = float(np.mean((X_corrected - X_true) ** 2))
        assert mse_after < mse_before, (
            f"{method}: MSE not reduced (before={mse_before:.6f}, after={mse_after:.6f})"
        )

    def test_arpls_reduces_baseline_mse(self):
        self._assert_improves("arpls", {"lam": 1e5})

    def test_imor_reduces_baseline_mse(self):
        self._assert_improves("imor", {"half_window": 30})

    def test_snip_reduces_baseline_mse(self):
        self._assert_improves("snip", {"max_half_window": 30})

    def test_modpoly_reduces_baseline_mse(self):
        self._assert_improves("modpoly", {"poly_order": 3})


# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputValidation:
    """Verify that invalid inputs raise appropriate errors or warnings."""

    def test_unknown_method_raises_value_error_on_transform(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X = np.ones((3, 50))
        est = BaselineAdvanced(method="nonexistent")
        est.fit(X)
        with pytest.raises(ValueError, match="nonexistent"):
            est.transform(X)

    def test_wavenumbers_none_emits_user_warning(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=3)
        est = BaselineAdvanced(method="arpls", wavenumbers=None, lam=1e5)
        est.fit(X)
        with pytest.warns(UserWarning, match="[Ww]avenumber"):
            X_out = est.transform(X)
        # Should still produce valid output
        assert X_out.shape == X.shape
        assert not np.any(np.isnan(X_out))

    def test_wavenumbers_wrong_length_raises_on_fit(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced

        X = np.ones((3, N_FEATURES))
        wrong_wn = np.linspace(400, 4000, N_FEATURES - 10)  # wrong length
        est = BaselineAdvanced(method="arpls", wavenumbers=wrong_wn)
        with pytest.raises(ValueError, match="wavenumbers"):
            est.fit(X)


# ---------------------------------------------------------------------------
# 5. Shape and determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShapeAndDeterminism:
    """Output shape must match input; repeated calls must be identical."""

    @pytest.mark.parametrize(
        "method,extra",
        [
            ("arpls", {"lam": 1e5}),
            ("snip", {"max_half_window": 20}),
            ("modpoly", {"poly_order": 3}),
        ],
    )
    def test_output_shape_matches_input(self, method, extra):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=5)
        est = BaselineAdvanced(method=method, **extra)
        X_out = est.fit_transform(X)
        assert X_out.shape == X.shape

    @pytest.mark.parametrize(
        "method,extra",
        [
            ("arpls", {"lam": 1e5}),
            ("snip", {"max_half_window": 20}),
        ],
    )
    def test_output_is_deterministic(self, method, extra):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=4)
        est = BaselineAdvanced(method=method, **extra)
        X1 = est.fit_transform(X)
        X2 = est.fit_transform(X)
        np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# 6. Row-level robustness — degenerate rows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDegenerateRows:
    """Degenerate rows (all-zeros, constant) must not crash the batch."""

    def test_all_zero_row_does_not_crash(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=5)
        X[2] = 0.0  # degenerate row in the middle
        est = BaselineAdvanced(method="arpls", lam=1e5)
        # Should complete without raising
        X_out = est.fit_transform(X)
        assert X_out.shape == X.shape

    def test_constant_row_does_not_crash(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=5)
        X[0] = 1.0  # all-ones row
        est = BaselineAdvanced(method="snip", max_half_window=20)
        X_out = est.fit_transform(X)
        assert X_out.shape == X.shape


# ---------------------------------------------------------------------------
# 7. auto_optimize
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAutoOptimize:
    """Verify auto_optimize works for supported methods and raises for unsupported."""

    def test_auto_optimize_arpls_returns_dict(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=5)
        result = BaselineAdvanced.auto_optimize(X, method="arpls")
        assert isinstance(result, dict)

    def test_auto_optimize_imor_raises_value_error(self):
        """imor has supports_optimize=False, so auto_optimize must raise ValueError."""
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=5)
        with pytest.raises(ValueError, match="imor"):
            BaselineAdvanced.auto_optimize(X, method="imor")

    def test_auto_optimize_unknown_method_raises_value_error(self):
        from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

        if not HAS_PYBASELINES:
            pytest.skip("pybaselines not installed")

        X, _ = _make_batch(n_samples=3)
        with pytest.raises(ValueError, match="nonexistent"):
            BaselineAdvanced.auto_optimize(X, method="nonexistent")

    def test_auto_optimize_raises_import_error_when_pybaselines_missing(self):
        from spectral_predict import baseline_advanced

        X = np.ones((3, 50))
        with mock.patch.object(baseline_advanced, "HAS_PYBASELINES", False):
            with pytest.raises(ImportError):
                baseline_advanced.BaselineAdvanced.auto_optimize(X, method="arpls")


# ---------------------------------------------------------------------------
# 8. Registry integrity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistryIntegrity:
    """ADVANCED_ALGORITHMS registry must have correct structure for all Tier 1 entries."""

    def test_all_tier1_algorithms_present(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS

        for key in TIER_1_ALGORITHMS:
            assert key in ADVANCED_ALGORITHMS, f"Tier 1 algorithm '{key}' missing from registry"

    def test_all_entries_have_required_fields(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS

        for key, info in ADVANCED_ALGORITHMS.items():
            missing = REQUIRED_REGISTRY_FIELDS - set(info.keys())
            assert not missing, f"Registry entry '{key}' missing fields: {missing}"

    def test_tier_values_are_valid(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS

        valid_tiers = {1, 2, 3}
        for key, info in ADVANCED_ALGORITHMS.items():
            assert info["tier"] in valid_tiers, f"'{key}' has invalid tier: {info['tier']}"

    def test_supports_optimize_is_bool(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS

        for key, info in ADVANCED_ALGORITHMS.items():
            assert isinstance(info["supports_optimize"], bool), (
                f"'{key}': supports_optimize must be bool, got {type(info['supports_optimize'])}"
            )

    def test_default_params_are_dicts(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS

        for key, info in ADVANCED_ALGORITHMS.items():
            assert isinstance(info["default_params"], dict), (
                f"'{key}': default_params must be dict"
            )
            assert len(info["default_params"]) > 0, f"'{key}': default_params must be non-empty"

    def test_imor_does_not_support_optimize(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS

        assert ADVANCED_ALGORITHMS["imor"]["supports_optimize"] is False

    def test_arpls_supports_optimize(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS

        assert ADVANCED_ALGORITHMS["arpls"]["supports_optimize"] is True


# ---------------------------------------------------------------------------
# 9. Helper functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelperFunctions:
    """Verify get_display_to_key_map, is_separator, get_dropdown_values."""

    def test_get_display_to_key_map_roundtrips(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS, get_display_to_key_map

        display_map = get_display_to_key_map()
        # Every key in the registry must be reachable via its display name
        for key, info in ADVANCED_ALGORITHMS.items():
            display = info["display_name"]
            assert display in display_map, f"display_name '{display}' missing from map"
            assert display_map[display] == key, (
                f"display_name '{display}' maps to '{display_map[display]}', expected '{key}'"
            )

    def test_get_display_to_key_map_length_matches_registry(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS, get_display_to_key_map

        assert len(get_display_to_key_map()) == len(ADVANCED_ALGORITHMS)

    def test_is_separator_true_for_tier_labels(self):
        from spectral_predict.baseline_advanced import TIER_LABELS, is_separator

        for _, label in TIER_LABELS:
            assert is_separator(label), f"Expected '{label}' to be a separator"

    def test_is_separator_false_for_real_display_names(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS, is_separator

        for key, info in ADVANCED_ALGORITHMS.items():
            assert not is_separator(info["display_name"]), (
                f"Display name '{info['display_name']}' should not be a separator"
            )

    def test_get_dropdown_values_contains_all_display_names(self):
        from spectral_predict.baseline_advanced import ADVANCED_ALGORITHMS, get_dropdown_values

        dropdown = get_dropdown_values()
        display_names_in_registry = {info["display_name"] for info in ADVANCED_ALGORITHMS.values()}
        display_names_in_dropdown = {v for v in dropdown if not (v.startswith("--") and v.endswith("--"))}
        assert display_names_in_dropdown == display_names_in_registry

    def test_get_dropdown_values_has_separator_strings(self):
        from spectral_predict.baseline_advanced import get_dropdown_values, is_separator

        dropdown = get_dropdown_values()
        separators = [v for v in dropdown if is_separator(v)]
        assert len(separators) > 0, "Dropdown must contain at least one tier separator"

    def test_get_dropdown_values_separators_precede_their_tier(self):
        """Each separator should appear before the algorithms of its tier."""
        from spectral_predict.baseline_advanced import (
            ADVANCED_ALGORITHMS,
            TIER_LABELS,
            get_dropdown_values,
            is_separator,
        )

        dropdown = get_dropdown_values()
        tier_label_map = {label: tier for tier, label in TIER_LABELS}

        last_separator_tier = None
        for value in dropdown:
            if is_separator(value):
                last_separator_tier = tier_label_map.get(value)
            elif last_separator_tier is not None:
                # Find this display name in registry and verify tier matches
                for info in ADVANCED_ALGORITHMS.values():
                    if info["display_name"] == value:
                        assert info["tier"] == last_separator_tier, (
                            f"'{value}' (tier {info['tier']}) appears under wrong separator "
                            f"(tier {last_separator_tier})"
                        )
                        break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
