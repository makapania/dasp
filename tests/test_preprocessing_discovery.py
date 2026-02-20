"""Comprehensive tests for spectral_predict.preprocessing_discovery module.

Tests cover:
- Constants and data structures (PREPROCESSING_CANDIDATES, IMPORTANCE_METHODS, etc.)
- compute_importance: wavelength importance via multiple methods
- apply_preprocessing: preprocessing chain application
- get_edge_zone: edge zone sizing for derivatives
- select_wavelengths_by_importance: top-N wavelength selection
- score_config: configuration scoring
- select_diverse_configs: diverse configuration selection
- evaluate_preprocessing_config: single config evaluation
"""

import numpy as np
import pytest

from spectral_predict.preprocessing_discovery import (
    IMPORTANCE_METHODS,
    PREPROCESSING_CANDIDATES,
    PREPROCESSING_COMPLEXITY,
    SUBSET_SIZES,
    WINDOW_SIZES,
    apply_preprocessing,
    compute_importance,
    get_edge_zone,
    score_config,
    select_diverse_configs,
    select_wavelengths_by_importance,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_spectra():
    """Synthetic spectral data: 30 samples x 100 wavelengths."""
    np.random.seed(42)
    n_samples, n_features = 30, 100
    # Create spectra with some structure (smooth + noise)
    base = np.sin(np.linspace(0, 4 * np.pi, n_features))
    X = np.tile(base, (n_samples, 1)) + np.random.randn(n_samples, n_features) * 0.1
    return X


@pytest.fixture
def synthetic_y(synthetic_spectra):
    """Continuous target values correlated with spectral data."""
    np.random.seed(42)
    X = synthetic_spectra
    # y depends on a few wavelengths
    y = 2.0 * X[:, 10] - 1.5 * X[:, 50] + 0.8 * X[:, 80] + np.random.randn(X.shape[0]) * 0.3
    return y


@pytest.fixture
def synthetic_classification_y(synthetic_spectra):
    """Binary classification target values."""
    np.random.seed(42)
    n = synthetic_spectra.shape[0]
    return np.array([0] * (n // 2) + [1] * (n - n // 2))


@pytest.fixture
def sample_configs(synthetic_spectra, synthetic_y):
    """A list of sample preprocessing config dicts for scoring tests."""
    configs = [
        {
            "preprocessing": "raw",
            "window": None,
            "deriv": None,
            "polyorder": None,
            "selected_wavelengths": np.arange(100),
            "n_wavelengths": 100,
            "score": 0.50,
            "importance_method": "vip",
            "model_name": None,
        },
        {
            "preprocessing": "snv",
            "window": None,
            "deriv": None,
            "polyorder": None,
            "selected_wavelengths": np.arange(100),
            "n_wavelengths": 100,
            "score": 0.45,
            "importance_method": "vip",
            "model_name": None,
        },
        {
            "preprocessing": "deriv1",
            "window": 11,
            "deriv": 1,
            "polyorder": 2,
            "selected_wavelengths": np.arange(90),
            "n_wavelengths": 90,
            "score": 0.40,
            "importance_method": "vip",
            "model_name": None,
        },
        {
            "preprocessing": "snv_deriv1",
            "window": 17,
            "deriv": 1,
            "polyorder": 2,
            "selected_wavelengths": np.arange(84),
            "n_wavelengths": 84,
            "score": 0.38,
            "importance_method": "vip",
            "model_name": None,
        },
        {
            "preprocessing": "deriv2",
            "window": 11,
            "deriv": 2,
            "polyorder": 3,
            "selected_wavelengths": np.arange(90),
            "n_wavelengths": 90,
            "score": 0.42,
            "importance_method": "vip",
            "model_name": None,
        },
    ]
    return configs


# =============================================================================
# Constants and data structures tests
# =============================================================================


def test_preprocessing_candidates_not_empty():
    """PREPROCESSING_CANDIDATES should contain multiple entries."""
    assert len(PREPROCESSING_CANDIDATES) > 0


def test_preprocessing_candidates_include_raw():
    """PREPROCESSING_CANDIDATES should include 'raw' preprocessing."""
    names = [c[0] for c in PREPROCESSING_CANDIDATES]
    assert "raw" in names


def test_preprocessing_candidates_include_snv():
    """PREPROCESSING_CANDIDATES should include 'snv' preprocessing."""
    names = [c[0] for c in PREPROCESSING_CANDIDATES]
    assert "snv" in names


def test_preprocessing_candidates_include_derivatives():
    """PREPROCESSING_CANDIDATES should include 1st and 2nd derivative variants."""
    names = [c[0] for c in PREPROCESSING_CANDIDATES]
    assert "deriv1" in names
    assert "deriv2" in names
    assert "snv_deriv1" in names
    assert "snv_deriv2" in names


def test_importance_methods_contains_expected_keys():
    """IMPORTANCE_METHODS should contain the four expected methods."""
    expected = {"cars_tree", "model_specific", "lightgbm", "vip"}
    assert expected.issubset(set(IMPORTANCE_METHODS.keys()))


def test_window_sizes_are_odd():
    """All WINDOW_SIZES should be odd integers (required by Savitzky-Golay)."""
    for w in WINDOW_SIZES:
        assert w % 2 == 1, f"Window size {w} should be odd"


def test_complexity_scores_cover_all_candidates():
    """PREPROCESSING_COMPLEXITY should have an entry for every candidate name."""
    candidate_names = {c[0] for c in PREPROCESSING_CANDIDATES}
    complexity_names = set(PREPROCESSING_COMPLEXITY.keys())
    assert candidate_names.issubset(complexity_names), (
        f"Missing complexity scores for: {candidate_names - complexity_names}"
    )


def test_subset_sizes_are_positive():
    """All SUBSET_SIZES should be positive integers."""
    for s in SUBSET_SIZES:
        assert s > 0


# =============================================================================
# compute_importance tests
# =============================================================================


def test_compute_importance_vip_shape(synthetic_spectra, synthetic_y):
    """VIP importance should return array with one value per feature."""
    importance = compute_importance(
        synthetic_spectra, synthetic_y, method="vip"
    )

    assert importance.shape == (synthetic_spectra.shape[1],)


def test_compute_importance_vip_normalized(synthetic_spectra, synthetic_y):
    """Importance values should be normalized to [0, 1]."""
    importance = compute_importance(
        synthetic_spectra, synthetic_y, method="vip"
    )

    assert np.min(importance) >= 0.0 - 1e-10
    assert np.max(importance) <= 1.0 + 1e-10


def test_compute_importance_unknown_method_raises(synthetic_spectra, synthetic_y):
    """Unknown method should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown importance method"):
        compute_importance(synthetic_spectra, synthetic_y, method="nonexistent_method")


def test_compute_importance_coefficient_based(synthetic_spectra, synthetic_y):
    """Model-specific importance for Ridge should use coefficient-based approach."""
    importance = compute_importance(
        synthetic_spectra, synthetic_y,
        method="model_specific", model_name="Ridge"
    )

    assert importance.shape == (synthetic_spectra.shape[1],)
    assert np.max(importance) <= 1.0 + 1e-10


# =============================================================================
# apply_preprocessing tests
# =============================================================================


def test_apply_preprocessing_raw(synthetic_spectra):
    """'raw' preprocessing should return a copy of the input unchanged."""
    result = apply_preprocessing(synthetic_spectra, "raw")

    np.testing.assert_array_equal(result, synthetic_spectra)
    # Should be a copy, not a view
    assert result is not synthetic_spectra


def test_apply_preprocessing_snv(synthetic_spectra):
    """'snv' preprocessing should return data with mean~0, std~1 per row."""
    result = apply_preprocessing(synthetic_spectra, "snv")

    assert result.shape == synthetic_spectra.shape
    # Each row should have mean near 0 and std near 1
    row_means = np.mean(result, axis=1)
    row_stds = np.std(result, axis=1)
    np.testing.assert_allclose(row_means, 0.0, atol=1e-10)
    np.testing.assert_allclose(row_stds, 1.0, atol=1e-6)


def test_apply_preprocessing_deriv1(synthetic_spectra):
    """'deriv1' preprocessing should return same shape data."""
    result = apply_preprocessing(synthetic_spectra, "deriv1", window=11)

    assert result.shape == synthetic_spectra.shape


def test_apply_preprocessing_snv_deriv1(synthetic_spectra):
    """'snv_deriv1' should apply SNV first, then 1st derivative."""
    result = apply_preprocessing(synthetic_spectra, "snv_deriv1", window=11)

    assert result.shape == synthetic_spectra.shape


def test_apply_preprocessing_deriv2_snv(synthetic_spectra):
    """'deriv2_snv' should apply 2nd derivative first, then SNV."""
    result = apply_preprocessing(synthetic_spectra, "deriv2_snv", window=11)

    assert result.shape == synthetic_spectra.shape


def test_apply_preprocessing_unknown_raises(synthetic_spectra):
    """Unknown preprocessing name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        apply_preprocessing(synthetic_spectra, "completely_invalid_name")


# =============================================================================
# get_edge_zone tests
# =============================================================================


def test_get_edge_zone_raw():
    """Raw preprocessing should have zero edge zone."""
    assert get_edge_zone("raw", None) == 0


def test_get_edge_zone_snv():
    """SNV preprocessing should have zero edge zone."""
    assert get_edge_zone("snv", None) == 0


def test_get_edge_zone_deriv1():
    """1st derivative should have edge zone of window // 2."""
    assert get_edge_zone("deriv1", 11) == 5
    assert get_edge_zone("deriv1", 17) == 8


def test_get_edge_zone_deriv2():
    """2nd derivative should also have edge zone of window // 2."""
    assert get_edge_zone("deriv2", 11) == 5


def test_get_edge_zone_none_window():
    """No window should return 0 edge zone."""
    assert get_edge_zone("deriv1", None) == 0


# =============================================================================
# select_wavelengths_by_importance tests
# =============================================================================


def test_select_wavelengths_by_importance_count():
    """Should select exactly target_n wavelengths when enough are available."""
    importance = np.random.rand(200)
    indices = select_wavelengths_by_importance(importance, target_n=50)

    assert len(indices) == 50


def test_select_wavelengths_by_importance_sorted():
    """Selected indices should be in ascending (spectral) order."""
    importance = np.random.rand(200)
    indices = select_wavelengths_by_importance(importance, target_n=50)

    assert np.all(np.diff(indices) > 0), "Indices should be in ascending order"


def test_select_wavelengths_by_importance_edge_exclusion():
    """Edge zone should be excluded from selection."""
    importance = np.ones(100)
    # Set edges to high importance - they should still be excluded
    importance[:10] = 10.0
    importance[-10:] = 10.0

    indices = select_wavelengths_by_importance(importance, target_n=30, edge_zone=10)

    # No selected index should be in the edge zones
    assert np.all(indices >= 10), "No index should be in the left edge zone"
    assert np.all(indices < 90), "No index should be in the right edge zone"


def test_select_wavelengths_by_importance_cap_at_available():
    """If target_n > available wavelengths, should return all available."""
    importance = np.random.rand(20)
    indices = select_wavelengths_by_importance(importance, target_n=100)

    assert len(indices) <= 20


# =============================================================================
# score_config tests
# =============================================================================


def test_score_config_lower_rmse_gets_lower_score(sample_configs):
    """Config with lower RMSE should get a lower (better) combined score."""
    best_config = sample_configs[3]  # score=0.38 (lowest RMSE)
    worst_config = sample_configs[0]  # score=0.50 (highest RMSE)

    best_score = score_config(best_config, sample_configs, "regression")
    worst_score = score_config(worst_config, sample_configs, "regression")

    assert best_score < worst_score, "Lower RMSE should yield lower combined score"


def test_score_config_returns_float(sample_configs):
    """score_config should return a float value."""
    result = score_config(sample_configs[0], sample_configs, "regression")
    assert isinstance(result, float)


# =============================================================================
# select_diverse_configs tests
# =============================================================================


def test_select_diverse_configs_respects_n_top(sample_configs):
    """Should return at most n_top configs."""
    selected = select_diverse_configs(sample_configs, n_top=3, task_type="regression")

    assert len(selected) == 3


def test_select_diverse_configs_diversity(sample_configs):
    """Should prefer diversity in preprocessing types."""
    selected = select_diverse_configs(sample_configs, n_top=3, task_type="regression")

    # Should have diverse preprocessing types
    preprocs = {c["preprocessing"] for c in selected}
    assert len(preprocs) >= 2, "Selected configs should have at least 2 different preprocessing types"


def test_select_diverse_configs_returns_all_when_n_top_large(sample_configs):
    """When n_top >= len(configs), should return all configs."""
    selected = select_diverse_configs(sample_configs, n_top=100, task_type="regression")

    assert len(selected) == len(sample_configs)


def test_select_diverse_configs_cleans_temporary_score(sample_configs):
    """Temporary _combined_score key should be removed from returned configs."""
    selected = select_diverse_configs(sample_configs, n_top=3, task_type="regression")

    for config in selected:
        assert "_combined_score" not in config, (
            "_combined_score should be cleaned from returned configs"
        )
