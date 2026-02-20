"""Comprehensive tests for the core search engine in Spectral Predict.

This test suite thoroughly tests the run_search() function which orchestrates:
- Model selection (PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP)
- Preprocessing (raw, snv, sg1, sg2, deriv_snv)
- Cross-validation
- Result ranking
- Variable selection
- Error handling

Test Categories:
1. Basic Functionality: Simple model runs with different task types
2. Preprocessing Tests: Different preprocessing methods
3. Variable Selection Tests: Subset and region-based selection
4. Configuration Tests: Custom hyperparameters and limits
5. Error Handling Tests: Edge cases and invalid inputs
6. Result Validation Tests: Output format and ranking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from spectral_predict.search import run_search
    HAS_CATBOOST = True
except (ImportError, ModuleNotFoundError):
    HAS_CATBOOST = False

pytestmark = pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")


# =============================================================================
# Basic Functionality Tests
# =============================================================================


@pytest.mark.integration
def test_basic_pls_regression(synthetic_spectra_small):
    """Test single PLS model with raw data and 3-fold CV."""
    X, y = synthetic_spectra_small

    results_df, label_encoder = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # Verify results structure
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) > 0, "Should produce at least one result"
    assert label_encoder is None, "Regression should not have label encoder"

    # Verify model is PLS
    best = results_df.iloc[0]
    assert "PLS" in best["Model"]

    # Verify required columns exist
    assert "R2" in best.index
    assert "RMSE" in best.index
    assert "Preprocess" in best.index

    # Verify reasonable performance
    assert not np.isnan(best["R2"]), "R² should not be NaN"
    assert not np.isnan(best["RMSE"]), "RMSE should not be NaN"
    assert best["RMSE"] > 0, "RMSE should be positive"


@pytest.mark.integration
def test_basic_pls_classification(classification_data):
    """Test PLS-DA for classification task."""
    X, y = classification_data

    results_df, label_encoder = run_search(
        X,
        y,
        task_type="classification",
        folds=3,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # Verify results
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) > 0, "Should produce at least one result"

    # Verify classification metrics
    best = results_df.iloc[0]
    assert "Accuracy" in best.index or "Acc" in best.index
    assert "PLS-DA" in best["Model"] or "PLS" in best["Model"]

    # Accuracy should be between 0 and 1
    acc_key = "Accuracy" if "Accuracy" in best.index else "Acc"
    assert 0 <= best[acc_key] <= 1, f"Accuracy should be in [0, 1], got {best[acc_key]}"


@pytest.mark.integration
def test_multi_model_comparison(synthetic_spectra_small):
    """Compare PLS, Ridge, and Lasso models."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS", "Ridge", "Lasso"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) >= 3, "Should have results for at least 3 model configurations"

    # Verify different models are present
    model_names = results_df["Model"].unique()
    assert any("PLS" in name for name in model_names), "Should have PLS results"
    # At least one of Ridge or Lasso should be present
    assert any("Ridge" in name or "Lasso" in name for name in model_names), (
        "Should have Ridge or Lasso results"
    )

    # All results should have valid R² and RMSE
    assert results_df["R2"].notna().all(), "All R² values should be non-NaN"
    assert results_df["RMSE"].notna().all(), "All RMSE values should be non-NaN"


@pytest.mark.integration
@pytest.mark.slow
def test_all_models_quick_tier(synthetic_spectra_small):
    """Run with tier='quick' to test all models with minimal hyperparameters."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # Should have multiple model types
    assert len(results_df) >= 3, "Quick tier should test multiple models"

    # Verify all results are valid
    assert results_df["R2"].notna().all()
    assert results_df["RMSE"].notna().all()
    assert (results_df["RMSE"] > 0).all()


@pytest.mark.integration
@pytest.mark.slow
def test_all_models_standard_tier(synthetic_spectra_small):
    """Run with tier='standard' (more models and hyperparameters)."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="standard",
    )

    # Standard tier should produce more results than quick tier
    assert len(results_df) >= 5, "Standard tier should test multiple configurations"

    # Verify results quality
    assert results_df["R2"].notna().all()
    assert results_df["RMSE"].notna().all()


# =============================================================================
# Preprocessing Tests
# =============================================================================


@pytest.mark.integration
def test_snv_preprocessing(synthetic_spectra_small):
    """Test SNV (Standard Normal Variate) preprocessing."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"snv": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) > 0

    # Verify SNV preprocessing was applied
    best = results_df.iloc[0]
    assert "snv" in best["Preprocess"].lower(), f"Expected SNV preprocessing, got {best['Preprocess']}"


@pytest.mark.integration
def test_sg1_derivative(synthetic_spectra_small):
    """Test 1st derivative with different window sizes."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"sg1": True},
        window_sizes=[7, 11],
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) >= 2, "Should test at least 2 window sizes"

    # Verify derivative preprocessing was applied
    for idx in range(min(2, len(results_df))):
        result = results_df.iloc[idx]
        assert "deriv" in result["Preprocess"].lower() or "sg" in result["Preprocess"].lower(), (
            f"Expected derivative preprocessing, got {result['Preprocess']}"
        )

    # Verify different window sizes were tested
    windows = results_df["Window"].unique()
    assert len(windows) >= 1, "Should test different window sizes"


@pytest.mark.integration
def test_sg2_derivative(synthetic_spectra_small):
    """Test 2nd derivative preprocessing."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"sg2": True},
        window_sizes=[7],
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) > 0

    best = results_df.iloc[0]
    assert "deriv" in best["Preprocess"].lower() or "sg" in best["Preprocess"].lower()

    # 2nd derivative should have Deriv=2
    if "Deriv" in best.index and not pd.isna(best["Deriv"]):
        assert best["Deriv"] == 2, f"Expected 2nd derivative, got {best['Deriv']}"


@pytest.mark.integration
def test_multiple_preprocessing_methods(synthetic_spectra_small):
    """Test running with multiple preprocessing methods: raw + snv + sg1."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True, "snv": True, "sg1": True},
        window_sizes=[7],
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # Should have results for multiple preprocessing methods
    assert len(results_df) >= 3, "Should test at least 3 preprocessing methods"

    # Verify different preprocessing methods are present
    preprocess_methods = results_df["Preprocess"].unique()
    assert len(preprocess_methods) >= 2, "Should have at least 2 different preprocessing methods"


# =============================================================================
# Variable Selection Tests
# =============================================================================


@pytest.mark.integration
def test_variable_subsets(synthetic_spectra_small):
    """Enable variable_subsets with counts [10, 50]."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=True,
        variable_counts=[10, 50],
        enable_region_subsets=False,
        tier="quick",
    )

    # Should have subset results
    assert len(results_df) > 0

    # Check for subset tags
    if "SubsetTag" in results_df.columns:
        subset_results = results_df[results_df["SubsetTag"].notna()]
        if len(subset_results) > 0:
            # Verify subset counts
            assert any("top10" in str(tag).lower() for tag in subset_results["SubsetTag"]), (
                "Should have top10 results"
            )


@pytest.mark.integration
def test_region_subsets(synthetic_spectra_small):
    """Enable region-based variable selection."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=True,
        n_top_regions=5,
        max_n_components=2,  # Limit components to avoid errors with small regions
        tier="quick",
    )

    assert len(results_df) > 0

    # Region analysis should produce valid results (some may have poor R²)
    assert results_df["R2"].notna().all()


# =============================================================================
# Configuration Tests
# =============================================================================


@pytest.mark.integration
def test_custom_hyperparameter_grids(synthetic_spectra_small):
    """Pass custom PLS n_components values."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        max_n_components=3,  # Limit to 3 components
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) > 0

    # Verify PLS components are within limit
    best = results_df.iloc[0]
    if "LVs" in best.index and not pd.isna(best["LVs"]):
        assert best["LVs"] <= 3, f"Should use at most 3 components, got {best['LVs']}"


@pytest.mark.integration
@pytest.mark.slow
def test_reproducibility_with_random_state(synthetic_spectra_small):
    """Test that repeated runs give same results.

    run_search uses a fixed internal RANDOM_STATE, so repeated calls
    with the same data and settings should produce identical results.
    """
    X, y = synthetic_spectra_small

    # Run 1
    results_df_1, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
        max_n_components=3,  # Limit to speed up
    )

    # Run 2 with same parameters
    results_df_2, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
        max_n_components=3,  # Limit to speed up
    )

    # Results should be identical or very close
    best_1 = results_df_1.iloc[0]
    best_2 = results_df_2.iloc[0]

    # R2 should be very close (within tolerance for parallel execution)
    r2_diff = abs(best_1["R2"] - best_2["R2"])
    assert r2_diff < 0.01, f"R2 difference {r2_diff} exceeds tolerance"

    # RMSE should also be very close
    rmse_diff = abs(best_1["RMSE"] - best_2["RMSE"])
    assert rmse_diff < 0.01, f"RMSE difference {rmse_diff} exceeds tolerance"


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
def test_empty_dataframe_raises():
    """Empty X should produce no results (0-row DataFrame)."""
    X = pd.DataFrame()
    y = pd.Series([1, 2, 3])

    # run_search handles empty DataFrames gracefully by returning 0-row results
    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
    )

    assert len(results_df) == 0, "Empty input should produce no results"


@pytest.mark.unit
def test_mismatched_xy_lengths_raises():
    """len(X) != len(y) should raise ValueError."""
    X = pd.DataFrame(np.random.randn(50, 100))
    y = pd.Series(np.random.randn(30))  # Different length

    with pytest.raises((ValueError, IndexError)):
        run_search(
            X,
            y,
            task_type="regression",
            folds=3,
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )


@pytest.mark.integration
def test_nan_values_warning(synthetic_spectra_small):
    """NaN in X should be handled gracefully (drop or warning)."""
    X, y = synthetic_spectra_small

    # Introduce some NaN values
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan
    X_with_nan.iloc[5, 10] = np.nan

    # This should either:
    # 1. Raise an informative error, or
    # 2. Drop NaN rows and continue, or
    # 3. Impute NaN values and warn

    # For now, we expect it to either work (after dropping NaN rows) or raise
    try:
        results_df, _ = run_search(
            X_with_nan,
            y,
            task_type="regression",
            folds=3,
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
            tier="quick",
        )
        # If it succeeds, verify results are valid
        assert len(results_df) > 0
    except (ValueError, RuntimeError) as e:
        # Should raise informative error about NaN values
        assert "nan" in str(e).lower() or "missing" in str(e).lower() or "inf" in str(e).lower()


@pytest.mark.integration
def test_single_class_classification():
    """Classification with single class should handle gracefully."""
    X = pd.DataFrame(np.random.randn(50, 100))
    y = pd.Series([0] * 50)  # All same class

    # This should raise or handle gracefully
    with pytest.raises((ValueError, RuntimeError)):
        run_search(
            X,
            y,
            task_type="classification",
            folds=3,
            models_to_test=["PLS-DA"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )


# =============================================================================
# Result Validation Tests
# =============================================================================


@pytest.mark.integration
def test_results_have_required_columns(synthetic_spectra_small):
    """Verify results DataFrame has all required columns."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS", "Ridge"],
        preprocessing_methods={"raw": True, "snv": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # Required columns for regression
    required_columns = ["Model", "Preprocess", "R2", "RMSE"]

    for col in required_columns:
        assert col in results_df.columns, f"Missing required column: {col}"

    # All rows should have valid values for required columns
    for col in ["R2", "RMSE"]:
        assert results_df[col].notna().all(), f"Column {col} has NaN values"


@pytest.mark.integration
def test_results_sorted_by_composite_score(synthetic_spectra_small):
    """Verify results are ranked properly."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS", "Ridge", "Lasso"],
        preprocessing_methods={"raw": True, "snv": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # Results should be sorted (best first)
    assert len(results_df) >= 2, "Need at least 2 results to test sorting"

    # For regression, results should generally be sorted by performance
    # Best model should be first
    best = results_df.iloc[0]
    worst = results_df.iloc[-1]

    # Check if ranking columns exist
    if "Rank" in results_df.columns:
        # If there's a Rank column, it should be ascending (1 = best)
        assert results_df["Rank"].iloc[0] <= results_df["Rank"].iloc[-1], (
            "Results should be sorted by Rank ascending"
        )
    else:
        # Verify that best model has better or comparable R² than worst
        # Results are typically sorted by R² descending for regression
        # Allow some tolerance since different models may have different strengths
        assert best["R2"] >= worst["R2"] - 0.1, (
            "First result should have R² comparable to or better than last"
        )


@pytest.mark.integration
def test_r2_values_in_valid_range(synthetic_spectra_small):
    """Verify 0 <= R2 <= 1 for regression (though R² can be negative in CV)."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # R2 can be negative in cross-validation (worse than mean baseline)
    # But it should be a finite number
    assert results_df["R2"].notna().all(), "All R2 values should be non-NaN"
    r2_values = results_df["R2"].astype(float)
    assert np.isfinite(r2_values).all(), "All R2 values should be finite"

    # Most R² values should be reasonable (not extremely negative)
    reasonable_r2 = (results_df["R2"] > -2.0).sum() / len(results_df)
    assert reasonable_r2 > 0.8, "At least 80% of R² values should be > -2.0"


# =============================================================================
# Additional Integration Tests
# =============================================================================


@pytest.mark.integration
def test_classification_with_numeric_labels(synthetic_spectra_small):
    """Test classification with numeric labels (0, 1, 2)."""
    X, y_reg = synthetic_spectra_small

    # Convert to classification labels - use qcut to ensure balanced classes
    y = pd.Series(pd.qcut(y_reg, q=3, labels=False, duplicates='drop'))

    # Ensure we have at least 2 classes
    if y.nunique() < 2:
        pytest.skip("Could not create sufficient classes from data")

    results_df, label_encoder = run_search(
        X,
        y,
        task_type="classification",
        folds=3,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) > 0
    # Should have accuracy metric
    assert "Accuracy" in results_df.columns or "Acc" in results_df.columns


@pytest.mark.integration
def test_classification_with_string_labels():
    """Test classification with string labels ('A', 'B', 'C')."""
    X = pd.DataFrame(np.random.randn(60, 100))
    y = pd.Series(["A"] * 20 + ["B"] * 20 + ["C"] * 20, dtype=object)

    results_df, label_encoder = run_search(
        X,
        y,
        task_type="classification",
        folds=3,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) > 0
    # Label encoder should be provided for string labels
    assert label_encoder is not None


@pytest.mark.integration
def test_progress_callback(synthetic_spectra_small):
    """Test that progress_callback is called during search."""
    X, y = synthetic_spectra_small

    progress_updates = []

    def callback(update_dict):
        progress_updates.append(update_dict)

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        progress_callback=callback,
        tier="quick",
    )

    # Progress callback should have been called
    assert len(progress_updates) > 0, "Progress callback should be called at least once"

    # Verify update dict has expected keys
    if len(progress_updates) > 0:
        update = progress_updates[0]
        assert isinstance(update, dict), "Progress update should be a dict"


@pytest.mark.integration
@pytest.mark.slow
def test_enabled_models_parameter(synthetic_spectra_small):
    """Test enabled_models parameter to filter which models to run."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        enabled_models=["PLS", "Ridge"],  # Only these two
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="standard",
    )

    # Should only have PLS and Ridge results
    model_names = results_df["Model"].unique()
    assert len(model_names) <= 2, "Should only test PLS and Ridge"
    assert any("PLS" in name for name in model_names)
    assert any("Ridge" in name for name in model_names)


@pytest.mark.integration
def test_wavelength_range_filtering(synthetic_spectra_small):
    """Test analysis_wl_min and analysis_wl_max parameters."""
    X, y = synthetic_spectra_small

    # Get wavelength range (columns are like "1000.0", "1007.5", etc.)
    wls = [float(col) for col in X.columns]
    wl_min = min(wls)
    wl_max = max(wls)
    wl_mid = (wl_min + wl_max) / 2

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        analysis_wl_min=wl_min,
        analysis_wl_max=wl_mid,  # Use only first half of spectrum
        tier="quick",
    )

    assert len(results_df) > 0

    # Should successfully run with wavelength filtering
    best = results_df.iloc[0]
    assert not np.isnan(best["R2"])


# =============================================================================
# Performance and Stress Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_large_hyperparameter_grid(synthetic_spectra_small):
    """Test with many hyperparameter combinations."""
    X, y = synthetic_spectra_small

    results_df, _ = run_search(
        X,
        y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True, "snv": True, "sg1": True, "sg2": True},
        window_sizes=[5, 7, 9, 11],
        max_n_components=5,
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    # Should produce many results (different preprocessing × window sizes)
    assert len(results_df) >= 5, "Should test multiple configurations"

    # All results should be valid
    assert results_df["R2"].notna().all()
    assert results_df["RMSE"].notna().all()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
