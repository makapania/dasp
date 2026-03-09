"""
GUI workflow tests for Spectral Predict V1.

Tests key workflows by simulating user interactions and validating results.
Uses example/BoneCollagen.csv data by default.

Usage:
    pytest tests/gui/test_workflows.py -v              # Run headless
    pytest tests/gui/test_workflows.py -v --visible    # Run with visible window
"""

import pytest
import numpy as np


# ============================================================
# Data Loading Tests
# ============================================================

@pytest.mark.gui
class TestDataLoading:
    """Test that data loads correctly into the application."""

    def test_regression_data_loads(self, loaded_regression_data):
        """Verify regression data (BoneCollagen %Collagen) loads correctly."""
        harness = loaded_regression_data

        # Check X is loaded
        assert harness.app.X is not None, "X (spectral data) not loaded"
        assert len(harness.app.X) > 0, "X has no samples"

        # Check y is loaded
        assert harness.app.y is not None, "y (target) not loaded"
        assert len(harness.app.y) > 0, "y has no values"

        # Check dimensions match
        assert len(harness.app.X) == len(harness.app.y), "X and y have different lengths"

        # Check y is numeric (regression)
        assert np.issubdtype(harness.app.y.dtype, np.number), "y should be numeric for regression"

        # Check y range is reasonable for %Collagen
        assert harness.app.y.min() >= 0, "Collagen % should be >= 0"
        assert harness.app.y.max() <= 100, "Collagen % should be <= 100"

    def test_classification_data_loads(self, loaded_classification_data):
        """Verify classification data (CollagenCat) loads correctly."""
        harness = loaded_classification_data

        # Check X is loaded
        assert harness.app.X is not None, "X (spectral data) not loaded"

        # Check y is loaded with categories
        assert harness.app.y is not None, "y (target) not loaded"

        # Check y has expected categories
        unique_cats = set(harness.app.y.unique())
        expected_cats = {'Low', 'Medium', 'High'}
        assert unique_cats == expected_cats, f"Expected {expected_cats}, got {unique_cats}"

    def test_wavelengths_are_numeric(self, loaded_regression_data):
        """Verify wavelength columns are numeric."""
        harness = loaded_regression_data

        wavelengths = harness.app.X.columns.values
        assert len(wavelengths) > 100, "Should have many wavelength columns"

        # Check first and last wavelengths are reasonable for NIR
        first_wl = float(wavelengths[0])
        last_wl = float(wavelengths[-1])

        assert first_wl > 0, "First wavelength should be positive"
        assert last_wl > first_wl, "Wavelengths should be increasing"


# ============================================================
# Regression Analysis Tests
# ============================================================

@pytest.mark.gui
@pytest.mark.slow
class TestRegressionAnalysis:
    """Test regression analysis workflow."""

    def test_quick_pls_analysis(self, loaded_regression_data):
        """Run a quick PLS analysis and verify results make sense."""
        harness = loaded_regression_data

        # Run analysis directly (bypasses GUI threading issues)
        success = harness.run_analysis_direct(
            models=['PLS'],
            preprocessing=['Raw', 'SNV'],
            cv_folds=3
        )
        assert success, "Analysis did not complete"

        # Validate results
        validation = harness.validate_results()
        assert validation['has_results'], "No results generated"
        assert validation['row_count'] > 0, "Results table is empty"
        assert validation['r2_valid'], f"R2 validation failed: {validation['errors']}"

        # Check results DataFrame
        df = harness.get_results_df()
        assert 'PLS' in df['Model'].values, "PLS model not in results"

    def test_results_have_required_columns(self, loaded_regression_data):
        """Verify results table has all required columns."""
        harness = loaded_regression_data

        # Run analysis directly
        harness.run_analysis_direct(models=['PLS'], cv_folds=3)

        df = harness.get_results_df()
        assert df is not None, "No results"

        # Check for key columns
        required = ['Model', 'Rank']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

        # Should have some metric column
        metric_cols = [c for c in df.columns if 'R2' in c.upper() or 'RMSE' in c.upper()]
        assert len(metric_cols) > 0, "No metric columns found"

    def test_r2_is_reasonable(self, loaded_regression_data):
        """Verify R2 values are within expected range."""
        harness = loaded_regression_data

        harness.run_analysis_direct(models=['PLS'], preprocessing=['SNV'], cv_folds=3)

        df = harness.get_results_df()

        # Find R2 column
        r2_col = None
        for col in df.columns:
            if 'r2' in col.lower():
                r2_col = col
                break

        assert r2_col is not None, "No R2 column found"

        # R2 should be between -1 and 1 (can be negative for bad models)
        r2_values = df[r2_col].dropna()
        assert (r2_values >= -1).all(), f"R2 below -1: {r2_values.min()}"
        assert (r2_values <= 1).all(), f"R2 above 1: {r2_values.max()}"

        # At least one model should have positive R2 on this data
        assert (r2_values > 0).any(), "All R2 values are negative - check data"


# ============================================================
# Classification Analysis Tests
# ============================================================

@pytest.mark.gui
@pytest.mark.slow
class TestClassificationAnalysis:
    """Test classification analysis workflow."""

    def test_quick_plsda_analysis(self, loaded_classification_data):
        """Run a quick PLS-DA classification and verify results."""
        harness = loaded_classification_data

        # Run analysis directly
        success = harness.run_analysis_direct(
            models=['PLS-DA'],
            preprocessing=['Raw', 'SNV'],
            cv_folds=3
        )
        assert success, "Analysis did not complete"

        # Check results
        validation = harness.validate_results()
        assert validation['has_results'], "No results generated"

    def test_accuracy_is_reasonable(self, loaded_classification_data):
        """Verify accuracy values are within expected range."""
        harness = loaded_classification_data

        harness.run_analysis_direct(models=['PLS-DA'], preprocessing=['SNV'], cv_folds=3)

        df = harness.get_results_df()
        if df is None or len(df) == 0:
            pytest.skip("No results - classification may not be configured")

        # Find accuracy column
        acc_col = None
        for col in df.columns:
            if 'acc' in col.lower():
                acc_col = col
                break

        if acc_col is None:
            pytest.skip("No accuracy column found")

        # Accuracy should be between 0 and 1
        acc_values = df[acc_col].dropna()
        assert (acc_values >= 0).all(), f"Accuracy below 0: {acc_values.min()}"
        assert (acc_values <= 1).all(), f"Accuracy above 1: {acc_values.max()}"

        # With 3 classes, random would be ~33%, should do better
        assert acc_values.max() > 0.33, "Best accuracy worse than random"


# ============================================================
# Model Refinement Tests
# ============================================================

@pytest.mark.gui
class TestModelRefinement:
    """Test model refinement workflow."""

    @pytest.mark.slow
    def test_results_populate_after_analysis(self, loaded_regression_data):
        """Verify results populate after analysis."""
        harness = loaded_regression_data

        # Run analysis directly
        harness.run_analysis_direct(models=['PLS'], cv_folds=3)

        # Check results exist
        assert harness.app.results_df is not None, "No results DataFrame"
        assert len(harness.app.results_df) > 0, "Results DataFrame is empty"

    def test_app_state_after_analysis(self, loaded_regression_data):
        """Verify app state is correct after analysis."""
        harness = loaded_regression_data

        harness.run_analysis_direct(models=['PLS'], cv_folds=3)

        # Check results DataFrame is populated
        assert harness.app.results_df is not None
        assert len(harness.app.results_df) > 0


# ============================================================
# Multi-Model Tests
# ============================================================

@pytest.mark.gui
@pytest.mark.slow
class TestMultipleModels:
    """Test analysis with multiple models."""

    def test_multiple_regression_models(self, loaded_regression_data):
        """Run analysis with multiple regression models."""
        harness = loaded_regression_data

        harness.run_analysis_direct(
            models=['PLS', 'Ridge'],
            preprocessing=['SNV'],
            cv_folds=3
        )

        df = harness.get_results_df()
        assert df is not None, "No results"

        models_tested = set(df['Model'].values)
        assert 'PLS' in models_tested, "PLS not in results"
        assert 'Ridge' in models_tested, "Ridge not in results"

    def test_multiple_preprocessing(self, loaded_regression_data):
        """Run analysis with multiple preprocessing methods."""
        harness = loaded_regression_data

        harness.run_analysis_direct(
            models=['PLS'],
            preprocessing=['Raw', 'SNV', 'SG1'],
            cv_folds=3
        )

        df = harness.get_results_df()
        assert df is not None, "No results"

        # Check that we have results with different preprocessing
        if 'Preprocess' in df.columns:
            pp_methods = set(df['Preprocess'].values)
            assert len(pp_methods) > 1, "Only one preprocessing method in results"


# ============================================================
# Baseline Tests (Default Configuration)
# ============================================================

@pytest.mark.gui
@pytest.mark.slow
class TestBaselineAnalysis:
    """
    Baseline tests using default GUI configuration.

    These tests use the standard settings that would be used
    when comparing improvements:
    - Models: PLS, Ridge, ElasticNet
    - Preprocessing: Default (Raw, SNV)
    - CV: 5-fold with SPXY holdout of 8 samples

    This serves as the reference baseline for comparing improvements.
    """

    def test_baseline_regression_with_holdout(self, loaded_regression_data):
        """
        Run baseline regression analysis with SPXY holdout.

        This is the primary baseline test using:
        - PLS, Ridge, ElasticNet models
        - Raw and SNV preprocessing
        - 5-fold CV
        - 8 samples held out using SPXY algorithm
        """
        harness = loaded_regression_data

        # Run baseline models with SPXY holdout
        success = harness.run_analysis_direct(
            models=['PLS', 'Ridge', 'ElasticNet'],
            preprocessing=['Raw', 'SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )
        assert success, "Baseline analysis with holdout failed"

        df = harness.get_results_df()
        assert df is not None, "No results"
        assert len(df) > 0, "Empty results"

        # Verify all baseline models ran
        models_tested = set(df['Model'].values)
        assert 'PLS' in models_tested, "PLS not in baseline results"
        assert 'Ridge' in models_tested, "Ridge not in baseline results"
        assert 'ElasticNet' in models_tested, "ElasticNet not in baseline results"

        # Report baseline R2
        r2_col = None
        for col in df.columns:
            if 'r2' in col.lower():
                r2_col = col
                break

        if r2_col:
            print("\n=== BASELINE RESULTS (SPXY 8-sample holdout) ===")
            print(f"  Calibration samples: {len(harness.app.X) - 8}")
            print(f"  Holdout samples: 8")
            print(f"  CV folds: 5")
            print("")
            for model in ['PLS', 'Ridge', 'ElasticNet']:
                model_results = df[df['Model'] == model]
                if len(model_results) > 0:
                    best_r2 = model_results[r2_col].max()
                    print(f"  {model}: Best R2 = {best_r2:.4f}")
            print("================================================\n")

    def test_baseline_regression(self, loaded_regression_data):
        """
        Run baseline regression analysis without holdout.

        This serves as the reference for comparing improvements.
        Uses PLS, Ridge, ElasticNet with default settings.
        """
        harness = loaded_regression_data

        # Run baseline models
        success = harness.run_analysis_direct(
            models=['PLS', 'Ridge', 'ElasticNet'],
            preprocessing=['Raw', 'SNV'],
            cv_folds=5
        )
        assert success, "Baseline analysis failed"

        df = harness.get_results_df()
        assert df is not None, "No results"
        assert len(df) > 0, "Empty results"

        # Verify all baseline models ran
        models_tested = set(df['Model'].values)
        assert 'PLS' in models_tested, "PLS not in baseline results"
        assert 'Ridge' in models_tested, "Ridge not in baseline results"
        assert 'ElasticNet' in models_tested, "ElasticNet not in baseline results"

        # Report baseline R2
        r2_col = None
        for col in df.columns:
            if 'r2' in col.lower():
                r2_col = col
                break

        if r2_col:
            print("\n=== BASELINE RESULTS (no holdout) ===")
            for model in ['PLS', 'Ridge', 'ElasticNet']:
                model_results = df[df['Model'] == model]
                if len(model_results) > 0:
                    best_r2 = model_results[r2_col].max()
                    print(f"  {model}: Best R2 = {best_r2:.4f}")
            print("=====================================\n")

    def test_baseline_results_quality(self, loaded_regression_data):
        """Verify baseline results are reasonable quality."""
        harness = loaded_regression_data

        harness.run_analysis_direct(
            models=['PLS', 'Ridge', 'ElasticNet'],
            preprocessing=['SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )

        df = harness.get_results_df()

        # Find R2 column
        r2_col = None
        for col in df.columns:
            if 'r2' in col.lower():
                r2_col = col
                break

        assert r2_col is not None, "No R2 column"

        # Baseline should achieve reasonable R2 on BoneCollagen data
        best_r2 = df[r2_col].max()
        assert best_r2 > 0.3, f"Baseline R2 too low: {best_r2:.4f} (expected > 0.3)"


# ============================================================
# State Validation Tests
# ============================================================

@pytest.mark.gui
class TestAppState:
    """Test application state management."""

    def test_initial_state(self, gui_harness):
        """Verify initial app state is clean."""
        harness = gui_harness

        assert harness.app.X is None, "X should be None initially"
        assert harness.app.y is None, "y should be None initially"
        assert harness.app.results_df is None, "results_df should be None initially"

    def test_task_type_variable(self, loaded_regression_data):
        """Test task_type variable."""
        harness = loaded_regression_data

        # Should be set to regression
        task_type = harness.get_var('task_type')
        assert task_type == 'regression', f"Expected 'regression', got '{task_type}'"

    def test_task_type_classification(self, loaded_classification_data):
        """Test task_type is classification."""
        harness = loaded_classification_data

        task_type = harness.get_var('task_type')
        assert task_type == 'classification', f"Expected 'classification', got '{task_type}'"


# ============================================================
# Smoke Tests (Quick validation)
# ============================================================

@pytest.mark.gui
@pytest.mark.smoke
class TestSmoke:
    """Quick smoke tests to verify basic functionality."""

    def test_app_creates(self, gui_harness):
        """App creates without error."""
        assert gui_harness.app is not None
        assert gui_harness.root is not None

    def test_app_has_notebook(self, gui_harness):
        """App has main notebook widget."""
        assert hasattr(gui_harness.app, 'notebook')
        assert gui_harness.app.notebook is not None

    def test_can_load_data(self, loaded_regression_data):
        """Can load example data."""
        assert loaded_regression_data.app.X is not None
        assert loaded_regression_data.app.y is not None

    def test_data_shape_reasonable(self, loaded_regression_data):
        """Loaded data has reasonable shape."""
        harness = loaded_regression_data

        n_samples, n_wavelengths = harness.app.X.shape

        # BoneCollagen has ~50 samples
        assert 40 <= n_samples <= 60, f"Expected ~50 samples, got {n_samples}"

        # NIR spectra typically have many wavelengths
        assert n_wavelengths > 100, f"Expected >100 wavelengths, got {n_wavelengths}"
