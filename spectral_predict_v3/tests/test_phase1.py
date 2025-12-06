"""
Phase 1 Foundation Tests for Spectral Predict v3.

Tests:
1. Core module imports and Engine instantiation
2. Theme creation
3. Application window creation (headless)
4. File loading via Engine
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


class TestCoreImports:
    """Test that core modules import correctly."""

    def test_types_import(self):
        """Test types can be imported."""
        from spectral_predict_v3.core.types import (
            SpectralDataset, LoadResult, MergeResult
        )
        assert SpectralDataset is not None
        assert LoadResult is not None
        assert MergeResult is not None

    def test_engine_import(self):
        """Test Engine can be imported and instantiated."""
        from spectral_predict_v3.core import Engine
        engine = Engine()
        assert engine is not None

    def test_io_utils_import(self):
        """Test io_utils can be imported."""
        from spectral_predict_v3.core import io_utils
        assert hasattr(io_utils, 'detect_format')
        assert hasattr(io_utils, 'read_csv_spectra')
        assert hasattr(io_utils, 'align_with_reference')


class TestSpectralDataset:
    """Test SpectralDataset dataclass."""

    def test_creation(self):
        """Test basic dataset creation."""
        from spectral_predict_v3.core.types import SpectralDataset

        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids
        )

        assert dataset.n_samples == 10
        assert dataset.n_wavelengths == 100
        assert dataset.wavelength_range == (400.0, 2500.0)
        assert not dataset.has_target

    def test_with_target(self):
        """Test dataset with target values."""
        from spectral_predict_v3.core.types import SpectralDataset

        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]
        y = np.random.rand(10)

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            y=y,
            target_name="moisture"
        )

        assert dataset.has_target
        assert dataset.target_name == "moisture"

    def test_copy(self):
        """Test dataset copy is independent."""
        from spectral_predict_v3.core.types import SpectralDataset

        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids
        )

        copy = dataset.copy()

        # Modify copy
        copy.X[0, 0] = -999
        copy.sample_ids[0] = "modified"

        # Original should be unchanged
        assert dataset.X[0, 0] != -999
        assert dataset.sample_ids[0] != "modified"


class TestEngine:
    """Test Engine functionality."""

    def test_instantiation(self):
        """Test Engine can be instantiated."""
        from spectral_predict_v3.core import Engine
        engine = Engine()
        assert engine is not None

    def test_detect_format(self):
        """Test format detection."""
        from spectral_predict_v3.core import io_utils

        # Test various extensions
        assert io_utils.detect_format("test.csv") == "csv"
        assert io_utils.detect_format("test.xlsx") == "excel"
        assert io_utils.detect_format("test.asd") == "asd"


class TestTheme:
    """Test theme module."""

    def test_colors_defined(self):
        """Test color palette is defined."""
        from spectral_predict_v3.ui.theme import COLORS

        assert "bg_base" in COLORS
        assert "text_primary" in COLORS
        assert "accent_primary" in COLORS

        # Colors should be RGB tuples
        assert len(COLORS["bg_base"]) == 3
        assert all(0 <= c <= 255 for c in COLORS["bg_base"])

    def test_create_theme_import(self):
        """Test theme creation function can be imported."""
        from spectral_predict_v3.ui.theme import create_theme
        assert callable(create_theme)


class TestUIImports:
    """Test UI module imports."""

    def test_app_import(self):
        """Test app can be imported."""
        from spectral_predict_v3.ui.app import SpectralPredictApp
        assert SpectralPredictApp is not None

    def test_main_import(self):
        """Test main entry point can be imported."""
        from spectral_predict_v3.main import main
        assert callable(main)


def run_tests():
    """Run all tests and report results."""
    import sys

    # Collect test classes
    test_classes = [
        TestCoreImports,
        TestSpectralDataset,
        TestEngine,
        TestTheme,
        TestUIImports,
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
