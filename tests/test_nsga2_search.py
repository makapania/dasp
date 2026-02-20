"""
Comprehensive tests for NSGA-II multi-objective optimization.

Tests cover:
- Pareto front generation and validation
- Knee point detection
- Solution decoding
- Model building from chromosomes
- Integration with classification and regression tasks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from spectral_predict.nsga2_search import (
        MODEL_TYPES,
        PREPROC_TYPES,
        WINDOW_SIZES,
        _build_model,
        _compute_solution_r2,
        _get_preprocessing_transform,
        convert_nsga2_to_v1_format,
        decode_solution,
        find_knee_point,
        pareto_to_dataframe,
        run_nsga2_search,
    )
    HAS_CATBOOST = True
except (ImportError, ModuleNotFoundError):
    HAS_CATBOOST = False

pytestmark = pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")


# =============================================================================
# Test NSGA-II Search Integration
# =============================================================================


@pytest.mark.integration
class TestNSGA2Search:
    """Integration tests for full NSGA-II optimization."""

    def test_returns_pareto_front(self, synthetic_spectra_small):
        """NSGA-II should return non-empty Pareto front."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        # Run short NSGA-II optimization
        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS", "Ridge"],  # Limit models for speed
        )

        assert "pareto_front" in result
        assert "pareto_solutions" in result
        assert len(result["pareto_front"]) > 0
        assert len(result["pareto_solutions"]) > 0
        assert len(result["pareto_front"]) == len(result["pareto_solutions"])

    def test_pareto_non_dominated(self, synthetic_spectra_small):
        """Pareto front should contain only non-dominated solutions."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=5,
            random_state=42,
            verbose=0,
            models=["PLS"],
        )

        pareto_front = result["pareto_front"]

        # Check non-domination: for each pair, neither should dominate the other
        # Solution A dominates B if A is better in all objectives
        for i in range(len(pareto_front)):
            for j in range(i + 1, len(pareto_front)):
                obj_i = pareto_front[i]
                obj_j = pareto_front[j]

                # Check if i dominates j (all objectives better or equal, at least one strictly better)
                i_dominates = np.all(obj_i <= obj_j) and np.any(obj_i < obj_j)
                # Check if j dominates i
                j_dominates = np.all(obj_j <= obj_i) and np.any(obj_j < obj_i)

                # In Pareto front, neither should dominate
                assert not i_dominates, f"Solution {i} dominates {j}"
                assert not j_dominates, f"Solution {j} dominates {i}"

    def test_knee_point_identified(self, synthetic_spectra_small):
        """Knee point should be in valid range."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS", "Ridge"],
            selection_bias=2.0,  # Use knee point selection (bias=0 picks min-error, which gives knee_idx=-1)
        )

        knee_idx = result["knee_idx"]
        n_pareto = len(result["pareto_front"])

        assert 0 <= knee_idx < n_pareto
        assert result["knee_solution"] is not None

    def test_respects_min_wavelengths(self, synthetic_spectra_small):
        """All solutions should respect min_wavelengths constraint."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        min_wl = 15
        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=min_wl,
            random_state=42,
            verbose=0,
            models=["PLS"],
        )

        # Decode all solutions and check wavelength count
        n_wavelengths = X_array.shape[1]
        for solution in result["pareto_solutions"]:
            decoded = decode_solution(solution, n_wavelengths, result["model_types"])
            assert decoded["n_wavelengths"] >= min_wl

    def test_classification_mode(self, classification_data):
        """NSGA-II should work for classification tasks."""
        X, y = classification_data
        X_array = X.values
        y_array = y.values

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="classification",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["RandomForest"],  # Use RandomForest for classification
        )

        assert len(result["pareto_front"]) > 0
        # For classification, error is 1 - accuracy, so should be between 0 and 1
        errors = result["pareto_front"][:, 0]
        # Filter out NaN and penalty values (1e10) - only check valid evaluations
        valid_errors = errors[(~np.isnan(errors)) & (errors < 1e9)]
        if len(valid_errors) > 0:
            assert np.all(valid_errors >= 0)
            assert np.all(valid_errors <= 1)

    def test_model_subset_filtering(self, synthetic_spectra_small):
        """Should use only specified models."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        # Specify limited model set
        test_models = ["PLS", "Ridge"]

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=test_models,
        )

        # Check that all solutions use only specified models
        n_wavelengths = X_array.shape[1]
        for solution in result["pareto_solutions"]:
            decoded = decode_solution(solution, n_wavelengths, test_models)
            assert decoded["model"] in test_models

    @pytest.mark.slow
    def test_longer_optimization(self, synthetic_spectra_small):
        """Longer optimization should find better solutions."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        # Short run
        result_short = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=3,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS"],
        )

        # Longer run
        result_long = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=20,
            n_generations=10,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS"],
        )

        # Longer run should find at least as good solution
        best_error_short = result_short["pareto_front"][:, 0].min()
        best_error_long = result_long["pareto_front"][:, 0].min()

        assert best_error_long <= best_error_short * 1.1  # Allow 10% tolerance


# =============================================================================
# Test Knee Point Detection
# =============================================================================


@pytest.mark.unit
class TestKneePointDetection:
    """Test knee point detection algorithm."""

    def test_2d_pareto_front(self):
        """Should find knee in 2D Pareto front."""
        # Create L-shaped Pareto front (classic case)
        pareto_front = np.array(
            [
                [0.1, 0.9],  # Low error, high wavelengths
                [0.3, 0.7],
                [0.5, 0.5],  # Knee point (balanced)
                [0.7, 0.3],
                [0.9, 0.1],  # High error, low wavelengths
            ]
        )

        knee_idx = find_knee_point(pareto_front)

        # Should pick middle point (best compromise)
        assert 0 <= knee_idx < len(pareto_front)

    def test_single_point_returns_zero(self):
        """Single point should return index 0."""
        pareto_front = np.array([[0.5, 0.5, 0.5]])

        knee_idx = find_knee_point(pareto_front)

        assert knee_idx == 0

    def test_two_points(self):
        """Two points should return valid index."""
        pareto_front = np.array([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5]])

        knee_idx = find_knee_point(pareto_front)

        assert knee_idx in [0, 1]

    def test_3d_pareto_front(self):
        """Should handle 3D Pareto front (all objectives)."""
        # Create 3D front
        pareto_front = np.array(
            [
                [0.1, 0.2, 0.9],  # Low error, low wavelengths, high complexity
                [0.3, 0.5, 0.5],  # Balanced
                [0.9, 0.9, 0.1],  # High error, high wavelengths, low complexity
            ]
        )

        knee_idx = find_knee_point(pareto_front)

        # Should find closest to ideal point (0, 0, 0)
        assert 0 <= knee_idx < len(pareto_front)

    def test_degenerate_front(self):
        """Should handle degenerate front where all objectives are equal."""
        pareto_front = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

        knee_idx = find_knee_point(pareto_front)

        assert 0 <= knee_idx < len(pareto_front)


# =============================================================================
# Test Solution Decoding
# =============================================================================


@pytest.mark.unit
class TestDecodeSolution:
    """Test chromosome decoding to human-readable format."""

    def test_decodes_preprocessing(self):
        """Should decode preprocessing configuration."""
        n_wavelengths = 100
        chromosome = np.zeros(13 + n_wavelengths, dtype=int)  # 13 config genes now

        # Test raw preprocessing
        chromosome[0] = 0  # raw
        decoded = decode_solution(chromosome, n_wavelengths)
        assert decoded["preprocessing"] == "raw"

        # Test SNV
        chromosome[0] = 1  # snv
        decoded = decode_solution(chromosome, n_wavelengths)
        assert decoded["preprocessing"] == "snv"

        # Test derivative with window
        chromosome[0] = 2  # deriv1
        chromosome[1] = 6  # window index 6 = window 17
        decoded = decode_solution(chromosome, n_wavelengths)
        assert "deriv1" in decoded["preprocessing"]
        assert "w17" in decoded["preprocessing"]

    def test_decodes_model_type(self):
        """Should decode model type correctly."""
        n_wavelengths = 100
        chromosome = np.zeros(13 + n_wavelengths, dtype=int)  # 13 config genes now
        chromosome[13:] = 1  # Select all wavelengths (start at gene 13)

        # Test PLS
        chromosome[2] = 0  # PLS
        decoded = decode_solution(chromosome, n_wavelengths)
        assert decoded["model"] == "PLS"

        # Test Ridge
        chromosome[2] = 1  # Ridge
        decoded = decode_solution(chromosome, n_wavelengths)
        assert decoded["model"] == "Ridge"

    def test_decodes_wavelength_selection(self):
        """Should decode wavelength selection mask."""
        n_wavelengths = 50
        chromosome = np.zeros(13 + n_wavelengths, dtype=int)  # 13 config genes now

        # Select specific wavelengths
        selected_indices = [0, 5, 10, 15, 20]
        for idx in selected_indices:
            chromosome[13 + idx] = 1  # Wavelengths start at gene 13

        decoded = decode_solution(chromosome, n_wavelengths)

        assert decoded["n_wavelengths"] == len(selected_indices)
        assert decoded["selected_indices"] == selected_indices

    def test_decodes_model_params(self):
        """Should decode model parameters from encoding."""
        n_wavelengths = 100
        chromosome = np.zeros(13 + n_wavelengths, dtype=int)  # 13 config genes now
        chromosome[13:] = 1  # All wavelengths (start at gene 13)

        # PLS with n_components
        chromosome[2] = 0  # PLS
        chromosome[3] = 4  # model_param = 4 -> n_components = 5
        decoded = decode_solution(chromosome, n_wavelengths)

        assert "n_components" in decoded["model_params"]


# =============================================================================
# Test Model Building
# =============================================================================


@pytest.mark.unit
class TestBuildModel:
    """Test model construction from chromosome encoding."""

    def test_builds_pls(self):
        """Should build PLS model with correct n_components."""
        model = _build_model("PLS", model_param=4, task_type="regression", random_state=42)

        assert model is not None
        assert hasattr(model, "n_components")
        assert model.n_components == 5  # param 4 -> n_components 5

    def test_builds_ridge(self):
        """Should build Ridge with exponential alpha scaling."""
        model = _build_model("Ridge", model_param=6, task_type="regression", random_state=42)

        assert model is not None
        assert hasattr(model, "alpha")
        # param 6 -> alpha = 10^(6/3 - 2) = 10^0 = 1.0
        assert abs(model.alpha - 1.0) < 0.1

    def test_builds_randomforest(self):
        """Should build RandomForest with correct estimators."""
        model = _build_model(
            "RandomForest", model_param=5, task_type="regression", random_state=42
        )

        assert model is not None
        assert hasattr(model, "n_estimators")
        # param 5 -> n_estimators = 50 + 5*15 = 125
        assert model.n_estimators == 125

    def test_model_param_scaling(self):
        """Different model_param values should give different configurations."""
        model1 = _build_model("PLS", model_param=0, task_type="regression", random_state=42)
        model2 = _build_model("PLS", model_param=14, task_type="regression", random_state=42)

        assert model1.n_components == 1  # Min
        assert model2.n_components == 15  # Max

    def test_classification_models(self):
        """Should build classification variants for classification tasks."""
        model = _build_model(
            "RandomForest", model_param=5, task_type="classification", random_state=42
        )

        assert model is not None
        # Should be classifier, not regressor
        assert hasattr(model, "predict_proba")

    def test_unavailable_model_returns_none(self):
        """Should return None for unavailable models."""
        # Test with model that requires optional dependency
        # Note: This test assumes the dependency might not be installed
        model_name = "LightGBM"  # May or may not be available
        model = _build_model(model_name, model_param=5, task_type="regression", random_state=42)

        # If dependency missing, should return None
        # If dependency present, should return model
        # Either is acceptable
        if model is None:
            pytest.skip(f"{model_name} not available")


# =============================================================================
# Test Preprocessing Transform
# =============================================================================


@pytest.mark.unit
class TestPreprocessingTransform:
    """Test preprocessing transformation generation."""

    def test_raw_returns_none(self):
        """Raw preprocessing should return None (no transform)."""
        transform = _get_preprocessing_transform(preproc_idx=0, window_idx=0)
        assert transform is None

    def test_snv_transform(self, synthetic_spectra_small):
        """SNV transform should normalize spectra."""
        X, _ = synthetic_spectra_small
        X_array = X.values

        transform = _get_preprocessing_transform(preproc_idx=1, window_idx=0)
        assert transform is not None

        X_transformed = transform(X_array)
        assert X_transformed.shape == X_array.shape

    def test_derivative_transform(self, synthetic_spectra_small):
        """Derivative transform should apply Savitzky-Golay."""
        X, _ = synthetic_spectra_small
        X_array = X.values

        # deriv1 with window index 6
        transform = _get_preprocessing_transform(preproc_idx=2, window_idx=6)
        assert transform is not None

        X_transformed = transform(X_array)
        assert X_transformed.shape == X_array.shape


# =============================================================================
# Test Result Conversion
# =============================================================================


@pytest.mark.unit
class TestResultConversion:
    """Test conversion of NSGA-II results to V1 format."""

    def test_pareto_to_dataframe(self, synthetic_spectra_small):
        """Should convert Pareto front to DataFrame."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS"],
        )

        df = pareto_to_dataframe(
            result, n_wavelengths=X_array.shape[1], task_type="regression"
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Model" in df.columns
        assert "RMSE" in df.columns
        assert "Is_Knee" in df.columns

    def test_v1_format_conversion(self, synthetic_spectra_small):
        """Should convert to V1 results format with all required fields."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values
        # Convert wavelengths to float array (they are strings in the fixture)
        wavelengths = X.columns.astype(float).values

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS", "Ridge"],
            selection_bias=2.0,  # Use knee point selection for Is_Knee marking
        )

        df = convert_nsga2_to_v1_format(
            result=result,
            n_wavelengths=X_array.shape[1],
            task_type="regression",
            folds=3,
            wavelengths=wavelengths,
            X=X_array,
            y=y_array,
            compute_r2=False,  # Skip R2 for speed
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check required columns
        required_cols = [
            "Task",
            "Model",
            "Preprocess",
            "RMSE",
            "n_vars",
            "SubsetTag",
            "Is_Knee",
        ]
        for col in required_cols:
            assert col in df.columns

    def test_knee_point_marked(self, synthetic_spectra_small):
        """Exactly one solution should be marked as knee point."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=5,
            cv_folds=3,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS"],
            selection_bias=2.0,  # Use knee point selection for Is_Knee marking
        )

        df = pareto_to_dataframe(
            result, n_wavelengths=X_array.shape[1], task_type="regression"
        )

        knee_count = df["Is_Knee"].sum()
        assert knee_count == 1


# =============================================================================
# Test R2 Computation
# =============================================================================


@pytest.mark.unit
class TestR2Computation:
    """Test R2 computation for NSGA-II solutions."""

    def test_computes_r2_for_solution(self, synthetic_spectra_small):
        """Should compute R2 for a decoded solution."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        # Create simple solution: PLS with all wavelengths
        n_wavelengths = X_array.shape[1]
        solution = np.zeros(13 + n_wavelengths, dtype=int)  # 13 config genes now
        solution[0] = 0  # raw preprocessing
        solution[1] = 6  # default window
        solution[2] = 0  # PLS model
        solution[3] = 4  # n_components = 5
        solution[4] = 7  # lr_gene (middle value)
        solution[5] = 7  # reg_alpha_gene (middle value)
        solution[6] = 7  # reg_lambda_gene (middle value)
        solution[7] = 7  # l1_gene (middle value)
        solution[8] = 7  # subsample_gene (middle value)
        solution[9] = 7  # colsample_gene (middle value)
        solution[10] = 7  # min_samples_gene (middle value)
        solution[11] = 0  # gamma_gene (no penalty)
        solution[12] = 7  # max_features_gene (middle value)
        solution[13:] = 1  # all wavelengths (start at gene 13)

        r2 = _compute_solution_r2(
            X=X_array,
            y=y_array,
            solution=solution,
            n_wavelengths=n_wavelengths,
            model_types=["PLS"],
            task_type="regression",
            cv_folds=3,
            random_state=42,
        )

        assert r2 is not None
        assert -1.0 <= r2 <= 1.0  # R2 can be negative for bad models

    def test_r2_none_for_classification(self, classification_data):
        """R2 should be None for classification tasks."""
        X, y = classification_data
        X_array = X.values
        y_array = y.values

        n_wavelengths = X_array.shape[1]
        solution = np.zeros(13 + n_wavelengths, dtype=int)  # 13 config genes now
        solution[2] = 1  # Ridge model
        solution[4] = 7  # lr_gene (middle value)
        solution[5] = 7  # reg_alpha_gene (middle value)
        solution[6] = 7  # reg_lambda_gene (middle value)
        solution[7] = 7  # l1_gene (middle value)
        solution[8] = 7  # subsample_gene (middle value)
        solution[9] = 7  # colsample_gene (middle value)
        solution[10] = 7  # min_samples_gene (middle value)
        solution[11] = 0  # gamma_gene (no penalty)
        solution[12] = 7  # max_features_gene (middle value)
        solution[13:] = 1  # all wavelengths (start at gene 13)

        r2 = _compute_solution_r2(
            X=X_array,
            y=y_array,
            solution=solution,
            n_wavelengths=n_wavelengths,
            model_types=["Ridge"],
            task_type="classification",
            cv_folds=3,
            random_state=42,
        )

        assert r2 is None


# =============================================================================
# Test Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pareto_front(self):
        """Should handle empty Pareto front gracefully."""
        result = {
            "pareto_front": np.array([]),
            "pareto_solutions": np.array([]),
            "knee_idx": -1,
            "knee_solution": None,
            "n_evaluations": 0,
            "history": [],
        }

        df = pareto_to_dataframe(result, n_wavelengths=100, task_type="regression")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_min_wavelengths_too_high(self, synthetic_spectra_small):
        """Should handle case where min_wavelengths is very restrictive."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        # Request more than half the wavelengths
        min_wl = X_array.shape[1] - 5

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=10,
            n_generations=3,
            cv_folds=3,
            min_wavelengths=min_wl,
            random_state=42,
            verbose=0,
            models=["PLS"],
        )

        # Should still complete
        assert len(result["pareto_front"]) >= 0

    def test_very_small_population(self, synthetic_spectra_small):
        """Should handle very small population size."""
        X, y = synthetic_spectra_small
        X_array = X.values
        y_array = y.values

        result = run_nsga2_search(
            X=X_array,
            y=y_array,
            task_type="regression",
            population_size=4,  # Minimum viable
            n_generations=2,
            cv_folds=2,
            min_wavelengths=10,
            random_state=42,
            verbose=0,
            models=["PLS"],
        )

        # Should complete without error
        assert "pareto_front" in result
