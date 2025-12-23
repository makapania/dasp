"""
Integration Tests for Tab 7 Model Development.

This test suite verifies complete end-to-end workflows using Tab 7:
1. Load data → Run analysis → Load to Tab 7 → Refine → Save model
2. Test with different dataset sizes (quick_start and full)
3. Test with different model types and preprocessing methods
4. Verify R² reproducibility across the entire workflow

Test Categories:
- TestTab7QuickStartWorkflow: Complete workflow with quick_start data
- TestTab7FullDatasetWorkflow: Complete workflow with larger datasets
- TestTab7ModelSaveLoad: Test model persistence and reloading
- TestTab7ValidationSetWorkflow: Test with validation set enabled
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import time
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.search import run_search
from spectral_predict.model_io import save_model, load_model
from spectral_predict.models import get_model
from sklearn.model_selection import KFold, cross_val_score

# Import test utilities
from tab7_test_utils import (
    load_quick_start_data,
    create_minimal_synthetic_data,
    run_minimal_analysis,
    run_analysis_with_subsets,
    get_top_result,
    get_subset_result,
    validate_result_fields,
    validate_subset_result_fields,
    extract_hyperparameters,
    compare_r2_values
)


@pytest.fixture
def quick_start_workflow_data():
    """
    Fixture providing quick_start data and minimal analysis results.
    """
    try:
        X, y, ref = load_quick_start_data()
    except FileNotFoundError:
        pytest.skip("Quick start data not available")

    # Run minimal analysis
    results = run_minimal_analysis(X, y, models=['PLS', 'Ridge'], n_folds=3, verbose=False)

    return X, y, results


@pytest.fixture
def full_dataset_workflow_data():
    """
    Fixture providing full synthetic dataset (50 samples, 200 wavelengths).
    """
    X, y = create_minimal_synthetic_data(n_samples=50, n_wavelengths=200, seed=42)

    # Run comprehensive analysis
    results = run_search(
        X=X,
        y=y,
        task_type='regression',
        models=['PLS', 'Ridge', 'RandomForest'],
        preprocessing_methods=['raw', 'sg1', 'snv'],
        n_folds=5,
        subset_methods=['full', 'top'],
        subset_sizes=[50],
        max_n_components=10,
        verbose=False
    )

    return X, y, results


class TestTab7QuickStartWorkflow:
    """Test complete workflow with quick_start example data."""

    def test_complete_workflow_pls(self, quick_start_workflow_data):
        """Test complete workflow: load → analyze → load to Tab 7 → verify R²."""
        X, y, results = quick_start_workflow_data

        # Step 1: Get top PLS result
        top_pls = get_top_result(results, model_type='PLS', preprocessing='raw')
        original_r2 = top_pls['R2']
        n_components = int(top_pls['LVs'])
        n_folds = int(top_pls.get('n_folds', 5))

        print(f"\nOriginal PLS result: R²={original_r2:.6f}, n_components={n_components}")

        # Step 2: Simulate loading into Tab 7 and re-running
        # (This simulates what happens when user double-clicks result and clicks "Run")
        from sklearn.cross_decomposition import PLSRegression

        model = PLSRegression(n_components=n_components)
        cv = KFold(n_splits=n_folds, shuffle=False)
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        reproduced_r2 = scores.mean()

        print(f"Reproduced in Tab 7: R²={reproduced_r2:.6f}")

        # Step 3: Verify R² matches
        r2_diff = compare_r2_values(original_r2, reproduced_r2, tolerance=0.001)
        print(f"R² difference: {r2_diff:.6f} (PASS)")

    def test_complete_workflow_ridge(self, quick_start_workflow_data):
        """Test complete workflow with Ridge model."""
        X, y, results = quick_start_workflow_data

        # Get top Ridge result
        ridge_results = results[results['Model'] == 'Ridge']
        if len(ridge_results) == 0:
            pytest.skip("No Ridge results in quick_start analysis")

        top_ridge = get_top_result(results, model_type='Ridge', preprocessing='raw')
        original_r2 = top_ridge['R2']
        alpha = float(top_ridge['Alpha'])
        n_folds = int(top_ridge.get('n_folds', 5))

        print(f"\nOriginal Ridge result: R²={original_r2:.6f}, alpha={alpha}")

        # Re-run in Tab 7
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=alpha)
        cv = KFold(n_splits=n_folds, shuffle=False)
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        reproduced_r2 = scores.mean()

        print(f"Reproduced in Tab 7: R²={reproduced_r2:.6f}")

        # Verify R² matches
        r2_diff = compare_r2_values(original_r2, reproduced_r2, tolerance=0.001)
        print(f"R² difference: {r2_diff:.6f} (PASS)")

    def test_workflow_with_derivative_preprocessing(self, quick_start_workflow_data):
        """Test workflow with derivative preprocessing."""
        X, y, results = quick_start_workflow_data

        # Get result with derivative preprocessing
        deriv_results = results[results['Preprocess'] == 'deriv']
        if len(deriv_results) == 0:
            pytest.skip("No derivative preprocessing results")

        top_deriv = deriv_results.iloc[0]
        original_r2 = top_deriv['R2']

        print(f"\nDerivative preprocessing result: R²={original_r2:.6f}")

        # This would be reproduced in Tab 7 with preprocessing applied
        # For now, just verify the result exists and has valid data
        validate_result_fields(top_deriv)
        assert original_r2 > 0, "R² should be positive"


class TestTab7FullDatasetWorkflow:
    """Test complete workflow with larger synthetic dataset."""

    def test_full_dataset_pls_workflow(self, full_dataset_workflow_data):
        """Test complete workflow with full dataset (50 samples)."""
        X, y, results = full_dataset_workflow_data

        # Get top result
        top_result = get_top_result(results)
        original_r2 = top_result['R2']

        print(f"\nFull dataset top result: {top_result['Model']}, R²={original_r2:.6f}")

        # Verify result quality
        assert original_r2 > 0.5, f"R² should be reasonable, got {original_r2}"

        # Verify all expected fields present
        validate_result_fields(top_result)

    def test_full_dataset_with_subsets(self, full_dataset_workflow_data):
        """Test workflow with wavelength subsets."""
        X, y, results = full_dataset_workflow_data

        # Get subset results
        subset_results = results[results['SubsetTag'] != 'full']
        if len(subset_results) == 0:
            pytest.skip("No subset results in analysis")

        # Get top subset result
        top_subset = subset_results.iloc[0]
        original_r2 = top_subset['R2']

        print(f"\nTop subset result: R²={original_r2:.6f}, n_vars={top_subset['n_vars']}")

        # Validate subset-specific fields
        validate_subset_result_fields(top_subset)

        # Verify all_vars field is populated
        assert 'all_vars' in top_subset.index, "all_vars field missing"
        assert not pd.isna(top_subset['all_vars']), "all_vars is NaN"

        # Parse all_vars
        all_vars_str = str(top_subset['all_vars'])
        all_vars_list = [w.strip() for w in all_vars_str.split(',') if w.strip()]

        print(f"all_vars contains {len(all_vars_list)} wavelengths")
        assert len(all_vars_list) == int(top_subset['n_vars']), "Wavelength count mismatch"

    def test_multiple_preprocessing_methods(self, full_dataset_workflow_data):
        """Test that multiple preprocessing methods produce valid results."""
        X, y, results = full_dataset_workflow_data

        # Check that we have results from different preprocessing methods
        preprocessing_methods = results['Preprocess'].unique()
        print(f"\nPreprocessing methods tested: {preprocessing_methods}")

        assert len(preprocessing_methods) >= 2, "Should have multiple preprocessing methods"

        # Verify each preprocessing method produces valid results
        for method in preprocessing_methods:
            method_results = results[results['Preprocess'] == method]
            assert len(method_results) > 0, f"No results for {method}"

            top_method = method_results.iloc[0]
            assert top_method['R2'] > 0, f"{method} should produce positive R²"

            print(f"  {method}: R²={top_method['R2']:.4f}")


class TestTab7ModelSaveLoad:
    """Test model saving and loading workflow."""

    def test_save_refined_model(self, quick_start_workflow_data):
        """Test saving a refined model from Tab 7."""
        X, y, results = quick_start_workflow_data

        # Get top result
        top_result = get_top_result(results)

        # Simulate training refined model in Tab 7
        model_name = top_result['Model']
        params = extract_hyperparameters(top_result)

        model = get_model(model_name, task_type='regression', **params)
        model.fit(X.values, y.values)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            temp_path = f.name

        try:
            metadata = {
                'model_name': model_name,
                'task_type': 'regression',
                'preprocessing': top_result['Preprocess'],
                'n_vars': int(top_result['n_vars']),
                'wavelengths': X.columns.tolist(),
                'performance': {
                    'R2': float(top_result['R2']),
                    'RMSE': float(top_result['RMSE'])
                }
            }

            save_model(model, None, metadata, temp_path)

            # Verify file exists
            assert Path(temp_path).exists(), "Model file should exist"
            print(f"\nSaved model to: {temp_path}")

            # Load model back
            loaded = load_model(temp_path)

            # Verify loaded correctly
            assert 'model' in loaded
            assert 'metadata' in loaded
            assert loaded['metadata']['model_name'] == model_name

            print(f"Loaded model successfully: {loaded['metadata']['model_name']}")

        finally:
            # Cleanup
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_load_and_predict_with_saved_model(self, quick_start_workflow_data):
        """Test loading a saved model and making predictions."""
        X, y, results = quick_start_workflow_data

        # Train and save model
        top_result = get_top_result(results)
        model_name = top_result['Model']
        params = extract_hyperparameters(top_result)

        model = get_model(model_name, task_type='regression', **params)
        model.fit(X.values, y.values)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            temp_path = f.name

        try:
            metadata = {
                'model_name': model_name,
                'task_type': 'regression',
                'preprocessing': 'raw',
                'n_vars': X.shape[1],
                'wavelengths': X.columns.tolist()
            }

            save_model(model, None, metadata, temp_path)

            # Load model
            loaded = load_model(temp_path)
            loaded_model = loaded['model']

            # Make predictions
            predictions = loaded_model.predict(X.values[:5])

            # Verify predictions are reasonable
            assert len(predictions) == 5, "Should have 5 predictions"
            assert not np.any(np.isnan(predictions)), "Predictions should not be NaN"

            print(f"\nMade predictions with loaded model: {predictions[:3]}")

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()


class TestTab7ValidationSetWorkflow:
    """Test Tab 7 workflow with validation set enabled."""

    def test_workflow_with_validation_split(self):
        """Test that Tab 7 respects validation set split."""
        # Create data
        X, y = create_minimal_synthetic_data(n_samples=50, n_wavelengths=100)

        # Simulate validation split (20% validation)
        n_samples = len(X)
        n_val = int(0.2 * n_samples)
        validation_indices = list(range(n_samples - n_val, n_samples))
        calibration_indices = list(range(n_samples - n_val))

        print(f"\nTotal samples: {n_samples}")
        print(f"Calibration: {len(calibration_indices)} samples")
        print(f"Validation: {len(validation_indices)} samples")

        # Run analysis on calibration set only
        X_cal = X.iloc[calibration_indices]
        y_cal = y.iloc[calibration_indices]

        results = run_minimal_analysis(X_cal, y_cal, n_folds=3, verbose=False)

        # Get top result
        top_result = get_top_result(results)
        original_r2 = top_result['R2']

        print(f"Calibration R²: {original_r2:.4f}")

        # Simulate Tab 7: should use same calibration set
        model_name = top_result['Model']
        params = extract_hyperparameters(top_result)
        n_folds = int(top_result.get('n_folds', 5))

        model = get_model(model_name, task_type='regression', **params)
        cv = KFold(n_splits=n_folds, shuffle=False)
        scores = cross_val_score(model, X_cal.values, y_cal.values, cv=cv, scoring='r2')
        reproduced_r2 = scores.mean()

        print(f"Tab 7 reproduced R²: {reproduced_r2:.4f}")

        # Verify R² matches
        r2_diff = compare_r2_values(original_r2, reproduced_r2, tolerance=0.001)
        print(f"R² difference: {r2_diff:.6f} (PASS)")

        # Now test on validation set
        model.fit(X_cal.values, y_cal.values)
        X_val = X.iloc[validation_indices]
        y_val = y.iloc[validation_indices]

        val_predictions = model.predict(X_val.values)
        val_r2 = np.corrcoef(y_val.values, val_predictions)[0, 1] ** 2

        print(f"Validation R²: {val_r2:.4f}")

        # Validation R² should be reasonable (may be lower than calibration)
        assert not np.isnan(val_r2), "Validation R² should not be NaN"


class TestTab7EndToEndReproducibility:
    """Test complete end-to-end reproducibility from Results to Tab 7."""

    def test_pls_derivative_subset_reproducibility(self):
        """
        Test the most complex scenario: PLS + derivative preprocessing + wavelength subset.

        This is the scenario most prone to R² discrepancies, so it's critical to test.
        """
        # Create data
        X, y = create_minimal_synthetic_data(n_samples=40, n_wavelengths=150, seed=42)

        # Run analysis with subsets
        results = run_search(
            X=X,
            y=y,
            task_type='regression',
            models=['PLS'],
            preprocessing_methods=['deriv'],
            n_folds=3,
            subset_methods=['top'],
            subset_sizes=[50],
            max_n_components=5,
            verbose=False
        )

        # Get derivative + subset result
        deriv_subset = results[(results['Preprocess'] == 'deriv') & (results['SubsetTag'] != 'full')]
        if len(deriv_subset) == 0:
            pytest.skip("No derivative+subset results")

        top_result = deriv_subset.iloc[0]
        original_r2 = top_result['R2']

        print(f"\nOriginal (derivative+subset): R²={original_r2:.6f}")

        # Validate subset fields
        validate_subset_result_fields(top_result)

        # Extract configuration
        n_components = int(top_result['LVs'])
        n_folds = int(top_result.get('n_folds', 5))
        deriv = int(top_result.get('Deriv', 1))
        window = int(top_result.get('Window', 11))

        # Parse wavelength subset
        all_vars_str = str(top_result['all_vars'])
        subset_wl = [float(w.strip()) for w in all_vars_str.split(',') if w.strip()]

        print(f"Subset wavelengths: {len(subset_wl)} wavelengths")

        # Simulate Tab 7 preprocessing path (derivative + subset)
        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.pipeline import Pipeline

        # Path: Full-spectrum preprocessing, then subset
        prep_steps = build_preprocessing_pipeline('deriv', deriv, window, polyorder=2)
        prep_pipeline = Pipeline(prep_steps)

        # Preprocess full spectrum
        X_full_preprocessed = prep_pipeline.fit_transform(X.values)

        # Find indices of subset wavelengths
        all_wavelengths = X.columns.astype(float).values
        wavelength_indices = []
        for wl in subset_wl:
            idx = np.where(np.abs(all_wavelengths - wl) < 0.01)[0]
            if len(idx) > 0:
                wavelength_indices.append(idx[0])

        # Subset preprocessed data
        X_subset = X_full_preprocessed[:, wavelength_indices]

        print(f"Preprocessed shape: {X_full_preprocessed.shape} -> {X_subset.shape}")

        # Run cross-validation with same settings
        from sklearn.cross_decomposition import PLSRegression

        model = PLSRegression(n_components=n_components)
        cv = KFold(n_splits=n_folds, shuffle=False)
        scores = cross_val_score(model, X_subset, y.values, cv=cv, scoring='r2')
        reproduced_r2 = scores.mean()

        print(f"Reproduced (Tab 7): R²={reproduced_r2:.6f}")

        # Verify R² matches
        r2_diff = compare_r2_values(original_r2, reproduced_r2, tolerance=0.001)
        print(f"R² difference: {r2_diff:.6f} (PASS)")
        print("\nCOMPLEX SCENARIO PASSED: PLS + derivative + subset reproduced correctly!")


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
