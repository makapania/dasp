"""
Phase 2 Tests for Spectral Predict v3.

Tests:
1. Column detection from DataFrames
2. Column config dialog creation
3. File preview workflow
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


class TestColumnDetection:
    """Test column auto-detection."""

    def test_detect_wavelength_columns(self):
        """Test detection of wavelength columns."""
        from spectral_predict_v3.core import io_utils

        # Create test DataFrame with wavelength columns
        df = pd.DataFrame({
            'Sample_ID': ['A', 'B', 'C'],
            'Moisture': [10.0, 12.0, 15.0],
            '400.0': [0.1, 0.2, 0.3],
            '500.0': [0.15, 0.25, 0.35],
            '600.0': [0.2, 0.3, 0.4],
        })

        result = io_utils.detect_columns(df)

        assert '400.0' in result['wavelength_columns']
        assert '500.0' in result['wavelength_columns']
        assert '600.0' in result['wavelength_columns']
        assert len(result['wavelength_columns']) == 3

    def test_detect_id_column(self):
        """Test detection of ID column."""
        from spectral_predict_v3.core import io_utils

        df = pd.DataFrame({
            'Sample_ID': ['A', 'B', 'C'],
            'Moisture': [10.0, 12.0, 15.0],
            '400.0': [0.1, 0.2, 0.3],
        })

        result = io_utils.detect_columns(df)

        assert 'Sample_ID' in result['id_candidates']

    def test_detect_target_column(self):
        """Test detection of target column."""
        from spectral_predict_v3.core import io_utils

        df = pd.DataFrame({
            'Sample_ID': ['A', 'B', 'C', 'D'],
            'Moisture': [10.0, 12.0, 15.0, 11.0],
            'Site': ['X', 'X', 'Y', 'Y'],  # Non-unique = metadata
            '400.0': [0.1, 0.2, 0.3, 0.4],
        })

        result = io_utils.detect_columns(df)

        assert 'Moisture' in result['target_candidates']
        assert 'Site' in result['metadata_columns']

    def test_wide_spectral_format(self):
        """Test detection with many wavelength columns."""
        from spectral_predict_v3.core import io_utils

        # Create wide format DataFrame
        wavelengths = [str(w) for w in np.linspace(400, 2500, 100)]
        data = {w: np.random.rand(10) for w in wavelengths}
        data['Sample_ID'] = [f'S{i}' for i in range(10)]
        data['Target'] = np.random.rand(10)

        df = pd.DataFrame(data)
        result = io_utils.detect_columns(df)

        assert len(result['wavelength_columns']) == 100
        assert 'Sample_ID' in result['id_candidates']
        assert 'Target' in result['target_candidates']


class TestColumnConfigDialog:
    """Test column config dialog component."""

    def test_dialog_import(self):
        """Test dialog can be imported."""
        from spectral_predict_v3.ui.components import ColumnConfigDialog
        assert ColumnConfigDialog is not None

    def test_show_column_config_import(self):
        """Test convenience function can be imported."""
        from spectral_predict_v3.ui.components import show_column_config
        assert callable(show_column_config)

    def test_dialog_instantiation(self):
        """Test dialog can be instantiated."""
        from spectral_predict_v3.ui.components import ColumnConfigDialog

        def dummy_callback(config):
            pass

        dialog = ColumnConfigDialog(on_confirm=dummy_callback)
        assert dialog is not None
        assert dialog.on_confirm == dummy_callback


class TestFormatDetection:
    """Test format detection."""

    def test_csv_detection(self):
        """Test CSV format detection."""
        from spectral_predict_v3.core import io_utils

        assert io_utils.detect_format("test.csv") == "csv"
        assert io_utils.detect_format("data/spectra.CSV") == "csv"

    def test_excel_detection(self):
        """Test Excel format detection."""
        from spectral_predict_v3.core import io_utils

        assert io_utils.detect_format("test.xlsx") == "excel"
        assert io_utils.detect_format("data/spectra.xls") == "excel"

    def test_asd_detection(self):
        """Test ASD format detection."""
        from spectral_predict_v3.core import io_utils

        assert io_utils.detect_format("test.asd") == "asd"
        assert io_utils.detect_format("data/sample.sig") == "asd"


def run_tests():
    """Run all tests and report results."""
    test_classes = [
        TestColumnDetection,
        TestColumnConfigDialog,
        TestFormatDetection,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                try:
                    method()
                    print(f"  [PASS] {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed += 1
                    errors.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"  [ERROR] {method_name}: {e}")
                    failed += 1
                    errors.append((test_class.__name__, method_name, str(e)))

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")

    if errors:
        print("\nFailures/Errors:")
        for cls, method, msg in errors:
            print(f"  {cls}.{method}: {msg}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
