"""Unit tests for model configuration and tier definitions."""

import pytest

from spectral_predict.model_config import (
    MODEL_TIERS,
    CLASSIFICATION_TIERS,
    DEFAULT_TIER,
    PREPROCESSING_DEFAULTS,
    get_tier_models,
    get_hyperparameters,
)


# ---------------------------------------------------------------------------
# Tier model list tests
# ---------------------------------------------------------------------------

REGRESSION_TIERS = ["quick", "standard", "comprehensive", "experimental"]
CLASSIFICATION_TIER_NAMES = ["quick", "standard", "comprehensive", "experimental"]


@pytest.mark.parametrize("tier", REGRESSION_TIERS)
def test_get_tier_models_regression_returns_nonempty_list(tier):
    """Every regression tier should return a non-empty list of model names."""
    models = get_tier_models(tier, task_type="regression")
    assert isinstance(models, list)
    assert len(models) > 0


@pytest.mark.parametrize("tier", CLASSIFICATION_TIER_NAMES)
def test_get_tier_models_classification_returns_nonempty_list(tier):
    """Every classification tier should return a non-empty list of model names."""
    models = get_tier_models(tier, task_type="classification")
    assert isinstance(models, list)
    assert len(models) > 0


def test_get_tier_models_invalid_tier_raises():
    """An unknown tier name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown tier"):
        get_tier_models("nonexistent")


def test_tier_hierarchy_regression():
    """Quick models should be a subset of Standard, Standard of Comprehensive, etc."""
    quick = set(get_tier_models("quick", "regression"))
    standard = set(get_tier_models("standard", "regression"))
    comprehensive = set(get_tier_models("comprehensive", "regression"))
    experimental = set(get_tier_models("experimental", "regression"))

    assert quick.issubset(standard), f"Quick models not in Standard: {quick - standard}"
    assert standard.issubset(comprehensive), (
        f"Standard models not in Comprehensive: {standard - comprehensive}"
    )
    assert comprehensive.issubset(experimental), (
        f"Comprehensive models not in Experimental: {comprehensive - experimental}"
    )


def test_tier_hierarchy_classification():
    """Classification tiers should follow the same subset hierarchy."""
    quick = set(get_tier_models("quick", "classification"))
    standard = set(get_tier_models("standard", "classification"))
    comprehensive = set(get_tier_models("comprehensive", "classification"))
    experimental = set(get_tier_models("experimental", "classification"))

    assert quick.issubset(standard), f"Quick models not in Standard: {quick - standard}"
    assert standard.issubset(comprehensive), (
        f"Standard models not in Comprehensive: {standard - comprehensive}"
    )
    assert comprehensive.issubset(experimental), (
        f"Comprehensive models not in Experimental: {comprehensive - experimental}"
    )


def test_all_model_names_are_nonempty_strings():
    """Every model name across all tiers should be a non-empty string."""
    for tier_name, tier_info in MODEL_TIERS.items():
        for model in tier_info["models"]:
            assert isinstance(model, str) and len(model) > 0, (
                f"Invalid model name in regression tier '{tier_name}': {model!r}"
            )
    for tier_name, tier_info in CLASSIFICATION_TIERS.items():
        for model in tier_info["models"]:
            assert isinstance(model, str) and len(model) > 0, (
                f"Invalid model name in classification tier '{tier_name}': {model!r}"
            )


def test_quick_regression_contains_pls():
    """PLS should always be in the quick regression tier (most basic model)."""
    quick_models = get_tier_models("quick", "regression")
    assert "PLS" in quick_models


def test_default_tier_is_valid():
    """DEFAULT_TIER should be a key in MODEL_TIERS."""
    assert DEFAULT_TIER in MODEL_TIERS


# ---------------------------------------------------------------------------
# Hyperparameter tests
# ---------------------------------------------------------------------------

def test_get_hyperparameters_known_model():
    """Ridge should return a dict with expected keys like 'alpha'."""
    params = get_hyperparameters("Ridge", "standard")
    assert isinstance(params, dict)
    assert "alpha" in params
    assert len(params["alpha"]) > 0


def test_get_hyperparameters_unknown_model_returns_empty():
    """An unknown model name should return an empty dict (no crash)."""
    params = get_hyperparameters("NonExistentModel", "standard")
    assert params == {}


def test_get_hyperparameters_unknown_tier_falls_back_to_standard():
    """An unknown tier should fall back to 'standard' defaults."""
    params_custom = get_hyperparameters("Ridge", "nonexistent_tier")
    params_standard = get_hyperparameters("Ridge", "standard")
    assert params_custom == params_standard


def test_hyperparameter_values_are_lists():
    """All hyperparameter values should be lists (grid-search convention)."""
    for model_name in ["PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest",
                       "LightGBM", "XGBoost", "CatBoost", "SVR", "MLP"]:
        params = get_hyperparameters(model_name, "standard")
        for key, value in params.items():
            assert isinstance(value, list), (
                f"{model_name}.{key} should be a list, got {type(value).__name__}"
            )


def test_pls_max_iter_positive():
    """PLS max_iter values should all be positive integers."""
    params = get_hyperparameters("PLS", "standard")
    if "max_iter" in params:
        for val in params["max_iter"]:
            assert isinstance(val, int) and val > 0


def test_ridge_alpha_values_positive():
    """Ridge alpha values should all be positive floats."""
    params = get_hyperparameters("Ridge", "standard")
    for alpha in params["alpha"]:
        assert alpha > 0, f"Ridge alpha must be positive, got {alpha}"


# ---------------------------------------------------------------------------
# Tier structure tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier_name", REGRESSION_TIERS)
def test_regression_tier_has_required_keys(tier_name):
    """Each regression tier entry should have description, models, recommended_for."""
    tier = MODEL_TIERS[tier_name]
    assert "description" in tier
    assert "models" in tier
    assert "recommended_for" in tier


# ---------------------------------------------------------------------------
# Preprocessing defaults tests
# ---------------------------------------------------------------------------

def test_preprocessing_sg_windows_are_odd():
    """Savitzky-Golay window lengths must be odd numbers."""
    for tier_key in ["quick", "standard", "comprehensive"]:
        windows = PREPROCESSING_DEFAULTS["savitzky_golay"][tier_key]["window_lengths"]
        for w in windows:
            assert w % 2 == 1, f"Window {w} in tier '{tier_key}' is not odd"


def test_preprocessing_methods_include_raw():
    """Every tier's preprocessing methods list should include 'raw'."""
    for tier_key in ["quick", "standard", "comprehensive"]:
        methods = PREPROCESSING_DEFAULTS["methods"][tier_key]
        assert "raw" in methods, f"'raw' missing from methods in tier '{tier_key}'"
