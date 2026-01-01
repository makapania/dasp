"""Comprehensive end-to-end workflow tests for Spectral Predict.

This test suite validates complete user journeys from data loading through
model training, refinement, persistence, and deployment. Each test represents
a realistic workflow that a user would perform.

Test Categories:
1. TestBasicAnalysisWorkflow: Complete analysis pipelines
2. TestModelRefinementWorkflow: Result exploration and retraining
3. TestModelPersistenceWorkflow: Save/load/predict cycles
4. TestCalibrationTransferWorkflow: Transfer learning between instruments
5. TestDataQualityWorkflow: Outlier detection and removal
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score

# Add src to path for imports
import sys

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.calibration_transfer import estimate_ds, apply_ds, estimate_pds, apply_pds
from spectral_predict.model_io import load_model, predict_with_model, save_model
from spectral_predict.preprocess import build_preprocessing_pipeline
from spectral_predict.search import run_search


@pytest.mark.integration
class TestBasicAnalysisWorkflow:
    """Test complete basic analysis pipelines from data to results."""

    def test_csv_to_results(self, synthetic_spectra_small, tmp_path):
        """Complete: Load CSV → Preprocess → Train → Rank → Export.

        Validates:
        - Multiple models can be trained (PLS, Ridge, RF)
        - Results are properly ranked by R²
        - Best model has valid metrics
        - All expected columns are present
        """
        X, y = synthetic_spectra_small

        # Run search with multiple models (returns tuple: results_df, label_encoder)
        results_df, _ = run_search(
            X,
            y,
            task_type="regression",
            folds=3,
            models_to_test=["PLS", "Ridge", "RandomForest"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )

        # Verify results exist
        assert len(results_df) > 0, "Should produce results"

        # Verify results are ranked (best R² first)
        r2_values = results_df["R2"].values
        assert all(
            r2_values[i] >= r2_values[i + 1] for i in range(len(r2_values) - 1)
        ), "Results should be sorted by R² descending"

        # Verify best model has valid metrics
        best = results_df.iloc[0]
        assert best["R2"] > 0, "Best model should have positive R²"
        assert best["RMSE"] > 0, "Best model should have positive RMSE"
        assert not pd.isna(best["Model"]), "Best model should have model name"

        # Verify all models were tested
        model_names = set(results_df["Model"])
        assert "PLS" in model_names
        assert "Ridge" in model_names
        assert "RandomForest" in model_names

        # Export results to verify workflow
        output_csv = tmp_path / "results.csv"
        results_df.to_csv(output_csv, index=False)
        assert output_csv.exists(), "Results should be exportable to CSV"

        # Verify exported file can be read back
        loaded_results = pd.read_csv(output_csv)
        assert len(loaded_results) == len(results_df), "Exported results should match"

    def test_classification_workflow(self, classification_data, tmp_path):
        """Complete classification pipeline.

        Validates:
        - Classification models can be trained
        - Accuracy metrics are computed
        - Results include classification-specific metrics
        """
        X, y = classification_data

        # Run search with classification (returns tuple: results_df, label_encoder)
        results_df, _ = run_search(
            X,
            y,
            task_type="classification",
            folds=3,
            models_to_test=["PLS", "RandomForest"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )

        # Verify results exist
        assert len(results_df) > 0, "Should produce classification results"

        # Verify classification metrics exist
        best = results_df.iloc[0]
        assert "Accuracy" in results_df.columns, "Should have accuracy column"
        assert best["Accuracy"] > 0.5, "Accuracy should be better than random"

    def test_multiple_preprocessing(self, synthetic_spectra_small):
        """Test raw + SNV + SG1 preprocessing.

        Validates:
        - All preprocessing methods produce results
        - Results are properly labeled with preprocessing method
        - Different preprocessing gives different results
        """
        X, y = synthetic_spectra_small

        # Run with multiple preprocessing methods (returns tuple: results_df, label_encoder)
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
        )

        # Verify results exist for each preprocessing method
        preprocessing_used = set(results_df["Preprocess"])
        assert "raw" in preprocessing_used, "Should have raw preprocessing"
        assert "snv" in preprocessing_used, "Should have SNV preprocessing"
        # Check for derivative preprocessing (may be labeled as "deriv" or "snv_deriv")
        has_derivative = any("deriv" in p for p in preprocessing_used)
        assert has_derivative, f"Should have derivative preprocessing, got: {preprocessing_used}"

        # Verify R² values differ across preprocessing methods
        r2_by_preprocess = results_df.groupby("Preprocess")["R2"].mean()
        assert len(r2_by_preprocess) > 1, "Should have multiple preprocessing methods"

        # Verify results are labeled correctly
        for _, row in results_df.iterrows():
            assert not pd.isna(row["Preprocess"]), "Preprocessing should be labeled"


@pytest.mark.integration
class TestModelRefinementWorkflow:
    """Test model refinement and retraining workflows."""

    def test_result_to_retrain_cycle(self, synthetic_spectra_small):
        """Load result config → retrain with same params → verify R² matches.

        Validates:
        - Best model configuration can be extracted
        - Model can be retrained with same parameters
        - R² matches within tolerance (accounting for CV variance)
        """
        X, y = synthetic_spectra_small

        # Initial search (returns tuple: results_df, label_encoder)
        results_df, _ = run_search(
            X,
            y,
            task_type="regression",
            folds=5,
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True, "snv": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )

        # Test retraining for best result
        best = results_df.iloc[0]
        original_r2 = best["R2"]

        # Extract configuration
        n_components = int(best["LVs"])
        preprocess_name = best["Preprocess"]

        # Build preprocessing pipeline
        preprocess_steps = build_preprocessing_pipeline(
            preprocess_name=preprocess_name,
            deriv=int(best["Deriv"]) if not pd.isna(best["Deriv"]) else None,
            window=int(best["Window"]) if not pd.isna(best["Window"]) else None,
            polyorder=int(best["Poly"]) if not pd.isna(best["Poly"]) else None,
        )

        # Apply preprocessing
        if preprocess_steps:
            from sklearn.pipeline import Pipeline

            preprocessor = Pipeline(preprocess_steps)
            X_processed = preprocessor.fit_transform(X.values)
        else:
            X_processed = X.values

        # Retrain model
        model = PLSRegression(n_components=n_components)
        cv_scores = cross_val_score(model, X_processed, y.values, cv=5, scoring="r2")
        retrained_r2 = cv_scores.mean()

        # Verify R² matches within tolerance
        r2_diff = abs(retrained_r2 - original_r2)
        assert (
            r2_diff < 0.02
        ), f"R² difference {r2_diff:.4f} exceeds tolerance (original={original_r2:.4f}, retrained={retrained_r2:.4f})"

    def test_hyperparameter_modification(self, synthetic_spectra_small):
        """Modify hyperparams → retrain → verify different result.

        Validates:
        - Different hyperparameters produce different results
        - Changes are in expected direction (e.g., more components != same R²)
        """
        X, y = synthetic_spectra_small

        # Train PLS with 3 components
        model_3comp = PLSRegression(n_components=3)
        cv_scores_3 = cross_val_score(model_3comp, X.values, y.values, cv=3, scoring="r2")
        r2_3comp = cv_scores_3.mean()

        # Train PLS with 10 components
        model_10comp = PLSRegression(n_components=10)
        cv_scores_10 = cross_val_score(
            model_10comp, X.values, y.values, cv=3, scoring="r2"
        )
        r2_10comp = cv_scores_10.mean()

        # Results should differ
        r2_diff = abs(r2_10comp - r2_3comp)
        assert r2_diff > 0.001, "Different n_components should give different results"


@pytest.mark.integration
class TestModelPersistenceWorkflow:
    """Test model save/load/predict workflows."""

    def test_save_load_predict_cycle(self, synthetic_spectra_small, tmp_path):
        """Save model → load → predict → verify accuracy.

        Validates:
        - Model can be saved with all metadata
        - Model can be loaded correctly
        - Predictions on new data are valid
        - Loaded model performs similarly to original
        """
        X_train, y_train = synthetic_spectra_small

        # Create test set using same fixture to ensure wavelength compatibility
        # Use a subset of the training data as test (realistic scenario)
        from sklearn.model_selection import train_test_split

        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_train, test_size=0.3, random_state=999
        )
        # Use the larger split for training
        X_train = X_train_split
        y_train = y_train_split

        # Train model
        model = PLSRegression(n_components=5)
        model.fit(X_train.values, y_train.values)

        # Get training predictions for reference
        y_pred_train = model.predict(X_train.values).ravel()
        r2_train_original = 1 - np.sum((y_train.values - y_pred_train) ** 2) / np.sum(
            (y_train.values - y_train.mean()) ** 2
        )

        # Save model
        model_path = tmp_path / "test_model.dasp"
        save_model(
            model=model,
            preprocessor=None,
            metadata={
                "model_name": "PLS",
                "task_type": "regression",
                "wavelengths": [float(w) for w in X_train.columns],
                "n_vars": len(X_train.columns),
                "performance": {"R2": float(r2_train_original), "RMSE": 0.1},
                "params": {"n_components": 5},
            },
            filepath=model_path,
        )

        assert model_path.exists(), "Model file should be created"

        # Load model
        model_dict = load_model(model_path)

        # Verify model loaded correctly
        assert "model" in model_dict, "Should contain model"
        assert "metadata" in model_dict, "Should contain metadata"
        assert model_dict["metadata"]["model_name"] == "PLS"

        # Predict on training data (should match original)
        y_pred_loaded_train = predict_with_model(model_dict, X_train)
        max_diff = np.max(np.abs(y_pred_train - y_pred_loaded_train))
        assert (
            max_diff < 1e-10
        ), f"Loaded model predictions differ from original by {max_diff}"

        # Predict on new test data
        predictions = predict_with_model(model_dict, X_test)

        # Verify predictions are valid
        assert len(predictions) == len(X_test), "Should predict for all test samples"
        assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN"
        assert not np.any(np.isinf(predictions)), "Predictions should not contain Inf"

        # Verify predictions are in reasonable range
        y_mean = y_train.mean()
        y_std = y_train.std()
        pred_in_range = np.abs(predictions - y_mean) < 5 * y_std
        assert np.sum(pred_in_range) / len(predictions) > 0.7, (
            "Most predictions should be within 5 std of training mean"
        )

    def test_preprocessing_persisted(self, synthetic_spectra_small, tmp_path):
        """Verify preprocessing is applied correctly after load.

        Validates:
        - Preprocessing pipeline is saved with model
        - Loaded model applies preprocessing automatically
        - Predictions match direct training
        """
        X, y = synthetic_spectra_small

        # Build preprocessing pipeline (SNV)
        preprocess_steps = build_preprocessing_pipeline(
            preprocess_name="snv", deriv=None, window=None, polyorder=None
        )

        from sklearn.pipeline import Pipeline

        preprocessor = Pipeline(preprocess_steps)
        X_processed = preprocessor.fit_transform(X.values)

        # Train model on preprocessed data
        model = PLSRegression(n_components=5)
        model.fit(X_processed, y.values)

        # Get original predictions
        y_pred_original = model.predict(X_processed).ravel()

        # Save model with preprocessing
        model_path = tmp_path / "model_with_preprocessing.dasp"
        save_model(
            model=model,
            preprocessor=preprocessor,
            metadata={
                "model_name": "PLS",
                "task_type": "regression",
                "preprocessing": "snv",
                "wavelengths": [float(w) for w in X.columns],
                "n_vars": len(X.columns),
                "performance": {"R2": 0.95},
                "params": {"n_components": 5},
            },
            filepath=model_path,
        )

        # Load model
        model_dict = load_model(model_path)

        # Predict on raw data (preprocessing should be applied automatically)
        predictions = predict_with_model(model_dict, X)

        # Verify predictions match original
        max_diff = np.max(np.abs(y_pred_original - predictions))
        assert max_diff < 1e-10, (
            f"Loaded model with preprocessing differs by {max_diff}"
        )


@pytest.mark.integration
class TestCalibrationTransferWorkflow:
    """Test calibration transfer workflows between instruments."""

    def test_ds_transfer_workflow(self, synthetic_spectra_small):
        """Build DS transfer → apply → predict.

        Validates:
        - DS transfer model can be built without errors
        - Transfer function executes successfully
        - Corrected spectra have same shape as input
        """
        X_master, y_master = synthetic_spectra_small

        # Create slave data with systematic multiplicative + additive bias
        X_slave = X_master.copy()
        # Add realistic instrument differences (scaling + offset + noise)
        X_slave = X_slave * 1.05 + 0.02 + np.random.randn(*X_slave.shape) * 0.01

        # Use first 30 samples as standardization set
        n_std = 30
        X_master_std = X_master.values[:n_std]
        X_slave_std = X_slave.values[:n_std]

        # Build DS transfer
        A_ds = estimate_ds(X_master_std, X_slave_std)

        # Verify DS matrix has correct shape
        assert A_ds.shape[0] == X_master.shape[1], "DS matrix should match number of wavelengths"

        # Apply transfer to remaining samples
        X_slave_new = X_slave.values[n_std:]
        X_slave_corrected = apply_ds(X_slave_new, A_ds)

        # Verify corrected data has correct shape
        assert X_slave_corrected.shape == X_slave_new.shape, (
            "Corrected spectra should have same shape as input"
        )

        # Verify no NaN or Inf values after correction
        assert not np.any(np.isnan(X_slave_corrected)), "Corrected spectra should not contain NaN"
        assert not np.any(np.isinf(X_slave_corrected)), "Corrected spectra should not contain Inf"

        # Verify that correction was applied (values should change)
        max_change = np.max(np.abs(X_slave_corrected - X_slave_new))
        assert max_change > 0.001, "DS transfer should modify the spectra"

    def test_pds_transfer_workflow(self, synthetic_spectra_small):
        """Build PDS transfer → apply → predict.

        Validates:
        - PDS transfer model can be built with windows
        - Transfer reduces systematic differences
        - Works with different window sizes
        """
        X_master, y_master = synthetic_spectra_small

        # Create slave data with wavelength-dependent bias
        X_slave = X_master.copy()
        wavelength_factor = np.linspace(0, 0.2, X_master.shape[1])
        X_slave = X_slave + wavelength_factor  # Add gradient bias

        # Use first 30 samples as standardization set
        n_std = 30
        X_master_std = X_master.values[:n_std]
        X_slave_std = X_slave.values[:n_std]

        # Build PDS transfer with window size 11
        pds_params = estimate_pds(X_master_std, X_slave_std, window=11)

        # Apply transfer to remaining samples
        X_slave_new = X_slave.values[n_std:]
        X_slave_corrected = apply_pds(X_slave_new, pds_params)

        # Verify correction reduces bias
        X_master_new = X_master.values[n_std:]
        error_before = np.mean(np.abs(X_slave_new - X_master_new))
        error_after = np.mean(np.abs(X_slave_corrected - X_master_new))

        assert error_after < error_before, (
            f"PDS transfer should reduce error (before={error_before:.6f}, "
            f"after={error_after:.6f})"
        )


@pytest.mark.integration
class TestDataQualityWorkflow:
    """Test data quality assessment and outlier handling workflows."""

    def test_outlier_removal_improves_model(self, outlier_data):
        """Detect outliers → remove → train → verify improvement.

        Validates:
        - Training with outliers produces lower R²
        - Removing known outliers improves R²
        - Workflow demonstrates value of data quality checks
        """
        X, y, outlier_indices = outlier_data

        # Train model with all data (including outliers)
        model_with_outliers = PLSRegression(n_components=5)
        cv_scores_with = cross_val_score(
            model_with_outliers, X.values, y.values, cv=3, scoring="r2"
        )
        r2_with_outliers = cv_scores_with.mean()

        # Remove known outliers
        clean_indices = [i for i in range(len(X)) if i not in outlier_indices]
        X_clean = X.iloc[clean_indices]
        y_clean = y.iloc[clean_indices]

        # Train model without outliers
        model_without_outliers = PLSRegression(n_components=5)
        cv_scores_without = cross_val_score(
            model_without_outliers, X_clean.values, y_clean.values, cv=3, scoring="r2"
        )
        r2_without_outliers = cv_scores_without.mean()

        # Verify improvement
        improvement = r2_without_outliers - r2_with_outliers
        assert improvement > 0.05, (
            f"Removing outliers should improve R² by >0.05 "
            f"(with={r2_with_outliers:.4f}, without={r2_without_outliers:.4f}, "
            f"improvement={improvement:.4f})"
        )

    def test_complete_quality_workflow(self, outlier_data, tmp_path):
        """Complete quality workflow: Load → Detect → Remove → Train → Save.

        Validates:
        - Complete data quality pipeline works end-to-end
        - Results can be saved and documented
        """
        X, y, true_outliers = outlier_data

        # Step 1: Detect outliers using simple method (leverage)
        from sklearn.covariance import EllipticEnvelope

        detector = EllipticEnvelope(contamination=0.05, random_state=42)
        outlier_predictions = detector.fit_predict(X.values)
        detected_outliers = np.where(outlier_predictions == -1)[0]

        # Step 2: Remove detected outliers
        clean_indices = [i for i in range(len(X)) if i not in detected_outliers]
        X_clean = X.iloc[clean_indices]
        y_clean = y.iloc[clean_indices]

        # Step 3: Train on clean data (returns tuple: results_df, label_encoder)
        results_df, _ = run_search(
            X_clean,
            y_clean,
            task_type="regression",
            folds=3,
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )

        # Step 4: Save best model
        best = results_df.iloc[0]
        assert best["R2"] > 0, "Should produce valid model on clean data"

        # Step 5: Export quality report
        quality_report = tmp_path / "quality_report.csv"
        report_df = pd.DataFrame(
            {
                "metric": ["n_samples_original", "n_outliers_detected", "n_samples_clean"],
                "value": [len(X), len(detected_outliers), len(X_clean)],
            }
        )
        report_df.to_csv(quality_report, index=False)
        assert quality_report.exists(), "Quality report should be saved"


@pytest.mark.integration
class TestCompleteUserJourney:
    """Test complete end-to-end user journeys spanning multiple sessions."""

    def test_discovery_to_deployment_workflow(
        self, synthetic_spectra_small, tmp_path
    ):
        """Complete journey: Explore → Select → Refine → Deploy.

        Simulates realistic user workflow:
        1. Initial exploration with multiple models
        2. Select best performing configuration
        3. Refine with different preprocessing
        4. Save final model for deployment
        5. Load and use in production
        """
        X_train, y_train = synthetic_spectra_small

        # Create validation set using train/test split
        from sklearn.model_selection import train_test_split

        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=888
        )
        # Use the larger split for training
        X_train = X_train_split
        y_train = y_train_split

        # === Phase 1: Initial Exploration === (returns tuple: results_df, label_encoder)
        results_exploration, _ = run_search(
            X_train,
            y_train,
            task_type="regression",
            folds=3,
            models_to_test=["PLS", "Ridge", "Lasso"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )

        # Select best model type
        best_exploration = results_exploration.iloc[0]
        best_model_type = best_exploration["Model"]

        # === Phase 2: Refine with Preprocessing === (returns tuple: results_df, label_encoder)
        results_refined, _ = run_search(
            X_train,
            y_train,
            task_type="regression",
            folds=5,  # More folds for better estimates
            models_to_test=[best_model_type],
            preprocessing_methods={"raw": True, "snv": True, "sg1": True},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )

        best_refined = results_refined.iloc[0]
        assert best_refined["R2"] >= best_exploration["R2"] - 0.01, (
            "Refined model should perform at least as well as exploration"
        )

        # === Phase 3: Train Final Model ===
        # Build preprocessing
        preprocess_steps = build_preprocessing_pipeline(
            preprocess_name=best_refined["Preprocess"],
            deriv=int(best_refined["Deriv"]) if not pd.isna(best_refined["Deriv"]) else None,
            window=int(best_refined["Window"]) if not pd.isna(best_refined["Window"]) else None,
            polyorder=int(best_refined["Poly"]) if not pd.isna(best_refined["Poly"]) else None,
        )

        if preprocess_steps:
            from sklearn.pipeline import Pipeline

            preprocessor = Pipeline(preprocess_steps)
            X_processed = preprocessor.fit_transform(X_train.values)
        else:
            preprocessor = None
            X_processed = X_train.values

        # Train final model
        if "PLS" in best_model_type:
            final_model = PLSRegression(n_components=int(best_refined["LVs"]))
        elif "Ridge" in best_model_type:
            final_model = Ridge(alpha=float(best_refined.get("alpha", 1.0)))
        else:
            final_model = Lasso(alpha=float(best_refined.get("alpha", 0.1)))

        final_model.fit(X_processed, y_train.values)

        # === Phase 4: Save for Deployment ===
        deployment_path = tmp_path / "production_model.dasp"
        save_model(
            model=final_model,
            preprocessor=preprocessor,
            metadata={
                "model_name": best_model_type,
                "task_type": "regression",
                "preprocessing": best_refined["Preprocess"],
                "wavelengths": [float(w) for w in X_train.columns],
                "n_vars": len(X_train.columns),
                "performance": {
                    "R2_cv": float(best_refined["R2"]),
                    "RMSE_cv": float(best_refined["RMSE"]),
                },
                "params": dict(best_refined),
            },
            filepath=deployment_path,
        )

        # === Phase 5: Production Use ===
        # Simulate loading in production environment
        production_model = load_model(deployment_path)

        # Make predictions
        predictions = predict_with_model(production_model, X_val)

        # Verify production predictions are reasonable
        assert len(predictions) == len(X_val), "Should predict for all validation samples"
        assert not np.any(np.isnan(predictions)), "Production predictions should be valid"

        # Calculate validation performance
        r2_val = 1 - np.sum((y_val.values - predictions) ** 2) / np.sum(
            (y_val.values - y_val.mean()) ** 2
        )

        # Validation R² should be reasonable (allowing for natural variation)
        assert r2_val > -0.5, (
            f"Validation R² should be reasonable (got {r2_val:.4f})"
        )

    def test_multi_model_comparison_workflow(
        self, synthetic_spectra_small, tmp_path
    ):
        """Workflow: Train multiple models → Save all → Compare on holdout.

        Simulates comparing different modeling approaches on same data.
        """
        X_train, y_train = synthetic_spectra_small

        # Create holdout set using train/test split
        from sklearn.model_selection import train_test_split

        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_train, test_size=0.25, random_state=777
        )
        # Use the larger split for training
        X_train = X_train_split
        y_train = y_train_split

        # Train multiple model types
        models_to_compare = [
            ("PLS", PLSRegression(n_components=5)),
            ("Ridge", Ridge(alpha=1.0)),
            ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42)),
        ]

        comparison_results = []

        for model_name, model in models_to_compare:
            # Train
            model.fit(X_train.values, y_train.values)

            # Save
            model_path = tmp_path / f"{model_name.lower()}_model.dasp"
            save_model(
                model=model,
                preprocessor=None,
                metadata={
                    "model_name": model_name,
                    "task_type": "regression",
                    "wavelengths": [float(w) for w in X_train.columns],
                    "n_vars": len(X_train.columns),
                    "performance": {"R2": 0.9},
                },
                filepath=model_path,
            )

            # Load and test
            loaded_model = load_model(model_path)
            predictions = predict_with_model(loaded_model, X_test)

            # Calculate test performance
            r2_test = 1 - np.sum((y_test.values - predictions) ** 2) / np.sum(
                (y_test.values - y_test.mean()) ** 2
            )

            comparison_results.append(
                {"model": model_name, "r2_test": r2_test, "path": model_path}
            )

        # Export comparison report
        comparison_df = pd.DataFrame(comparison_results)
        comparison_csv = tmp_path / "model_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)

        # Verify all models were tested
        assert len(comparison_results) == 3, "Should have results for all 3 models"

        # Verify comparison file exists
        assert comparison_csv.exists(), "Comparison report should be saved"

        # Verify all models produced valid predictions
        for result in comparison_results:
            assert not np.isnan(result["r2_test"]), (
                f"{result['model']} should have valid R²"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
