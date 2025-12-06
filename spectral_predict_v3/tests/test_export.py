"""
Tests for export functionality in Spectral Predict v3.

Tests cover:
- CSV export with special characters
- Excel export with multiple sheets
- Plot data export
- Large dataset export performance
- Model summary export
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from spectral_predict_v3.core.export import (
    export_results_to_csv,
    export_results_to_excel,
    export_predictions_to_csv,
    export_preprocessed_data_to_csv,
    export_variable_selection_to_csv,
    export_confusion_matrix_to_csv,
    export_model_summary,
    export_all_results
)


class TestCSVExport:
    """Test CSV export functionality."""

    def test_results_to_csv_basic(self):
        """Test basic results export to CSV."""
        # Create sample results
        results = pd.DataFrame({
            'Model': ['PLS', 'Ridge', 'Lasso'],
            'R2': [0.95, 0.92, 0.89],
            'RMSE': [0.05, 0.08, 0.11]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'results.csv'

            # Export
            success = export_results_to_csv(results, str(filepath))

            assert success, "Export should succeed"
            assert filepath.exists(), "CSV file should exist"

            # Read back and verify
            loaded = pd.read_csv(filepath)
            assert len(loaded) == 3, "Should have 3 rows"
            assert list(loaded.columns) == ['Unnamed: 0', 'Model', 'R2', 'RMSE'], \
                "Columns should match (with index)"

            print("✓ Basic CSV export works")

    def test_csv_with_special_characters(self):
        """Test CSV export with special characters."""
        # Create data with special characters
        results = pd.DataFrame({
            'Sample': ['Test, 1', 'Test "2"', 'Test\n3', 'Test;4'],
            'Value': [1.0, 2.0, 3.0, 4.0]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'special_chars.csv'

            success = export_results_to_csv(results, str(filepath))

            assert success, "Export should handle special characters"
            assert filepath.exists(), "File should exist"

            # Read back
            loaded = pd.read_csv(filepath)
            assert len(loaded) == 4, "Should preserve all rows"

            print("✓ CSV export handles special characters")

    def test_csv_auto_extension(self):
        """Test that CSV export adds .csv extension if missing."""
        results = pd.DataFrame({'A': [1, 2, 3]})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'results'  # No extension

            success = export_results_to_csv(results, str(filepath))

            # Should create .csv file
            expected_path = Path(tmpdir) / 'results.csv'
            assert expected_path.exists(), "Should add .csv extension"

            print("✓ CSV export adds .csv extension")


class TestExcelExport:
    """Test Excel export functionality."""

    def test_results_to_excel_basic(self):
        """Test basic Excel export."""
        results = pd.DataFrame({
            'Model': ['PLS', 'Ridge'],
            'R2': [0.95, 0.92]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'results.xlsx'

            success = export_results_to_excel(results, str(filepath))

            assert success, "Excel export should succeed"
            assert filepath.exists(), "Excel file should exist"

            # Read back
            loaded = pd.read_excel(filepath, sheet_name='Results')
            assert len(loaded) == 2, "Should have 2 rows"

            print("✓ Basic Excel export works")

    def test_excel_multiple_sheets(self):
        """Test Excel export with multiple sheets."""
        main_results = pd.DataFrame({'A': [1, 2, 3]})
        extra_data = pd.DataFrame({'B': [4, 5, 6]})

        additional_sheets = {
            'Extra': extra_data
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'multi_sheet.xlsx'

            success = export_results_to_excel(
                main_results,
                str(filepath),
                sheet_name='Main',
                additional_sheets=additional_sheets
            )

            assert success, "Multi-sheet export should succeed"

            # Read both sheets
            loaded_main = pd.read_excel(filepath, sheet_name='Main')
            loaded_extra = pd.read_excel(filepath, sheet_name='Extra')

            assert len(loaded_main) == 3, "Main sheet should have 3 rows"
            assert len(loaded_extra) == 3, "Extra sheet should have 3 rows"

            print("✓ Excel multi-sheet export works")

    def test_excel_auto_extension(self):
        """Test Excel export adds .xlsx extension."""
        results = pd.DataFrame({'A': [1, 2]})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'results'  # No extension

            success = export_results_to_excel(results, str(filepath))

            expected_path = Path(tmpdir) / 'results.xlsx'
            assert expected_path.exists(), "Should add .xlsx extension"

            print("✓ Excel export adds .xlsx extension")


class TestPredictionsExport:
    """Test predictions export."""

    def test_predictions_export_basic(self):
        """Test basic predictions export."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'predictions.csv'

            success = export_predictions_to_csv(y_true, y_pred, str(filepath))

            assert success, "Predictions export should succeed"
            assert filepath.exists(), "File should exist"

            # Read back
            loaded = pd.read_csv(filepath, index_col=0)
            assert 'Actual' in loaded.columns
            assert 'Predicted' in loaded.columns
            assert 'Residual' in loaded.columns
            assert len(loaded) == 5

            print("✓ Predictions export works")

    def test_predictions_with_sample_names(self):
        """Test predictions export with sample names."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        sample_names = ['Sample_A', 'Sample_B', 'Sample_C']

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'predictions_named.csv'

            success = export_predictions_to_csv(
                y_true, y_pred, str(filepath),
                sample_names=sample_names
            )

            assert success, "Export with sample names should succeed"

            # Read back
            loaded = pd.read_csv(filepath, index_col=0)
            assert list(loaded.index) == sample_names, "Should preserve sample names"

            print("✓ Predictions export with sample names works")


class TestPreprocessedDataExport:
    """Test preprocessed data export."""

    def test_preprocessed_data_export(self):
        """Test preprocessed spectral data export."""
        # Create synthetic spectral data
        X = np.random.randn(10, 50)
        wavelengths = np.linspace(400, 2500, 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'preprocessed.csv'

            success = export_preprocessed_data_to_csv(
                X, wavelengths, str(filepath)
            )

            assert success, "Preprocessed data export should succeed"
            assert filepath.exists(), "File should exist"

            # Read back
            loaded = pd.read_csv(filepath, index_col=0)
            assert loaded.shape == (10, 50), "Shape should be preserved"

            print("✓ Preprocessed data export works")

    def test_preprocessed_data_with_target(self):
        """Test preprocessed data export with target values."""
        X = np.random.randn(10, 30)
        wavelengths = np.linspace(400, 2500, 30)
        y = np.random.randn(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'preprocessed_with_y.csv'

            success = export_preprocessed_data_to_csv(
                X, wavelengths, str(filepath),
                y=y, y_name='Protein'
            )

            assert success, "Export with target should succeed"

            # Read back
            loaded = pd.read_csv(filepath, index_col=0)
            assert 'Protein' in loaded.columns, "Should have target column"
            assert loaded.shape == (10, 31), "Should have 30 wavelengths + 1 target"

            print("✓ Preprocessed data export with target works")


class TestVariableSelectionExport:
    """Test variable selection export."""

    def test_variable_selection_export(self):
        """Test selected variables export."""
        selected_indices = np.array([0, 5, 10, 15, 20])
        all_wavelengths = np.linspace(400, 2500, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'selected_vars.csv'

            success = export_variable_selection_to_csv(
                selected_indices, all_wavelengths, filepath=str(filepath)
            )

            assert success, "Variable selection export should succeed"

            # Read back
            loaded = pd.read_csv(filepath)
            assert len(loaded) == 5, "Should have 5 selected variables"
            assert 'Index' in loaded.columns
            assert 'Wavelength' in loaded.columns

            print("✓ Variable selection export works")

    def test_variable_selection_with_importances(self):
        """Test variable selection export with importance scores."""
        selected_indices = np.array([0, 5, 10])
        all_wavelengths = np.linspace(400, 2500, 50)
        all_importances = np.random.rand(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'selected_vars_importance.csv'

            success = export_variable_selection_to_csv(
                selected_indices, all_wavelengths,
                importances=all_importances,
                filepath=str(filepath)
            )

            assert success, "Export with importances should succeed"

            # Read back
            loaded = pd.read_csv(filepath)
            assert 'Importance' in loaded.columns, "Should have importance column"

            print("✓ Variable selection export with importances works")


class TestConfusionMatrixExport:
    """Test confusion matrix export."""

    def test_confusion_matrix_export(self):
        """Test confusion matrix export."""
        cm = np.array([
            [10, 2, 0],
            [1, 15, 1],
            [0, 3, 12]
        ])
        class_names = ['Class_A', 'Class_B', 'Class_C']

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'confusion_matrix.csv'

            success = export_confusion_matrix_to_csv(
                cm, class_names, filepath=str(filepath)
            )

            assert success, "Confusion matrix export should succeed"

            # Read back
            loaded = pd.read_csv(filepath, index_col=0)
            assert loaded.shape == (3, 3), "Should be 3x3"
            assert list(loaded.columns) == class_names, "Columns should match"

            print("✓ Confusion matrix export works")


class TestModelSummaryExport:
    """Test model summary export."""

    def test_model_summary_export(self):
        """Test model summary text export."""
        model_info = {
            'model_name': 'PLS',
            'hyperparameters': {'n_components': 10},
            'performance': {'R2': 0.95, 'RMSE': 0.05},
            'n_samples': 100,
            'n_features': 200
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'model_summary.txt'

            success = export_model_summary(model_info, str(filepath))

            assert success, "Model summary export should succeed"
            assert filepath.exists(), "Summary file should exist"

            # Read back
            with open(filepath, 'r') as f:
                content = f.read()

            assert 'PLS' in content, "Should contain model name"
            assert '0.95' in content or '0.9500' in content, "Should contain R2 value"

            print("✓ Model summary export works")


class TestBatchExport:
    """Test batch export functionality."""

    def test_export_all_results(self):
        """Test exporting all results to directory."""
        results_df = pd.DataFrame({
            'Model': ['PLS', 'Ridge'],
            'R2': [0.95, 0.92]
        })

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])

        X = np.random.randn(10, 30)
        wavelengths = np.linspace(400, 2500, 30)

        model_info = {
            'model_name': 'PLS',
            'performance': {'R2': 0.95}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / 'export'

            success = export_all_results(
                results_dir=str(results_dir),
                results_df=results_df,
                y_true=y_true,
                y_pred=y_pred,
                X=X,
                wavelengths=wavelengths,
                model_info=model_info
            )

            assert success, "Batch export should succeed"

            # Check all files exist
            assert (results_dir / 'results.csv').exists()
            assert (results_dir / 'results.xlsx').exists()
            assert (results_dir / 'predictions.csv').exists()
            assert (results_dir / 'preprocessed_data.csv').exists()
            assert (results_dir / 'model_summary.txt').exists()

            print("✓ Batch export creates all files")


class TestLargeDatasetExport:
    """Test export performance with large datasets."""

    def test_large_csv_export(self):
        """Test CSV export with large dataset."""
        # Create large dataset
        large_df = pd.DataFrame({
            f'Col_{i}': np.random.randn(10000)
            for i in range(50)
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'large_data.csv'

            import time
            start = time.time()

            success = export_results_to_csv(large_df, str(filepath))

            elapsed = time.time() - start

            assert success, "Large dataset export should succeed"
            assert filepath.exists(), "File should exist"

            # Should complete reasonably quickly (< 5 seconds for 10k rows)
            assert elapsed < 5.0, f"Export took too long: {elapsed:.2f}s"

            print(f"✓ Large CSV export (10k rows × 50 cols) in {elapsed:.2f}s")


def test_export_functions_importable():
    """Test that all export functions are importable."""
    from spectral_predict_v3.core.export import (
        export_results_to_csv,
        export_results_to_excel,
        export_predictions_to_csv,
        export_preprocessed_data_to_csv,
        export_variable_selection_to_csv,
        export_confusion_matrix_to_csv,
        export_model_summary,
        export_all_results
    )

    print("✓ All export functions importable")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing Export Functionality")
    print("=" * 60)

    # CSV tests
    print("\n--- CSV Export Tests ---")
    test_csv = TestCSVExport()
    test_csv.test_results_to_csv_basic()
    test_csv.test_csv_with_special_characters()
    test_csv.test_csv_auto_extension()

    # Excel tests
    print("\n--- Excel Export Tests ---")
    test_excel = TestExcelExport()
    test_excel.test_results_to_excel_basic()
    test_excel.test_excel_multiple_sheets()
    test_excel.test_excel_auto_extension()

    # Predictions tests
    print("\n--- Predictions Export Tests ---")
    test_pred = TestPredictionsExport()
    test_pred.test_predictions_export_basic()
    test_pred.test_predictions_with_sample_names()

    # Preprocessed data tests
    print("\n--- Preprocessed Data Export Tests ---")
    test_preproc = TestPreprocessedDataExport()
    test_preproc.test_preprocessed_data_export()
    test_preproc.test_preprocessed_data_with_target()

    # Variable selection tests
    print("\n--- Variable Selection Export Tests ---")
    test_varsel = TestVariableSelectionExport()
    test_varsel.test_variable_selection_export()
    test_varsel.test_variable_selection_with_importances()

    # Confusion matrix tests
    print("\n--- Confusion Matrix Export Tests ---")
    test_cm = TestConfusionMatrixExport()
    test_cm.test_confusion_matrix_export()

    # Model summary tests
    print("\n--- Model Summary Export Tests ---")
    test_summary = TestModelSummaryExport()
    test_summary.test_model_summary_export()

    # Batch export tests
    print("\n--- Batch Export Tests ---")
    test_batch = TestBatchExport()
    test_batch.test_export_all_results()

    # Performance tests
    print("\n--- Performance Tests ---")
    test_perf = TestLargeDatasetExport()
    test_perf.test_large_csv_export()

    # Import test
    print("\n--- Import Tests ---")
    test_export_functions_importable()

    print("\n" + "=" * 60)
    print("All export tests passed!")
    print("=" * 60)
