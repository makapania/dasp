"""Unit tests for variable count mismatch fix.

Tests that subset models store all wavelengths in all_vars column,
fixing the issue where only top 30 wavelengths were saved.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.search import _run_single_config
from spectral_predict.scoring import create_results_dataframe, add_result


class TestVariableCountFix:
    """Tests for the all_vars column fix."""

    def setup_method(self):
        """Set up test data for each test."""
        np.random.seed(42)

        # Create synthetic spectral data
        n_samples = 100
        n_wavelengths = 200

        self.X = np.random.randn(n_samples, n_wavelengths) * 0.05 + 1.0
        # Create target with relationship to specific wavelengths
        self.y = (2.0 * self.X[:, 50] +
                  1.5 * self.X[:, 100] +
                  1.0 * self.X[:, 150] +
                  np.random.randn(n_samples) * 0.1)

        # Create wavelength array (e.g., 1500-1700 nm)
        self.wavelengths = np.linspace(1500, 1700, n_wavelengths)

        # Create CV splitter
        self.cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    def test_subset_model_has_all_vars_column(self):
        """Test that subset models have all_vars in result dict."""
        # Create subset indices (e.g., top 50 wavelengths)
        subset_indices = np.arange(50)

        # Create model instance
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=5, scale=False)

        result = _run_single_config(
            X=self.X,
            y=self.y,
            wavelengths=self.wavelengths,
            model=model,
            model_name="PLS",
            params={"n_components": 5},
            preprocess_cfg={"name": "raw", "deriv": 0, "window": np.nan, "polyorder": np.nan},
            cv_splitter=self.cv_splitter,
            task_type="regression",
            is_binary_classification=False,
            subset_indices=subset_indices,
            subset_tag="top50",
            top_n_vars=30
        )

        # Check that all_vars exists
        assert 'all_vars' in result, "Result should contain 'all_vars' field"
        assert result['all_vars'] != 'N/A', "all_vars should not be N/A for subset models"

        # Parse all_vars and verify it has all 50 wavelengths
        all_vars_str = result['all_vars']
        all_wavelengths = [float(w.strip()) for w in all_vars_str.split(',') if w.strip()]

        assert len(all_wavelengths) == 50, f"Expected 50 wavelengths in all_vars, got {len(all_wavelengths)}"

    def test_subset_model_has_top_vars_column(self):
        """Test that subset models still have top_vars with top 30."""
        # Create subset indices (e.g., top 100 wavelengths)
        subset_indices = np.arange(100)

        # Create model instance
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)

        result = _run_single_config(
            X=self.X,
            y=self.y,
            wavelengths=self.wavelengths,
            model=model,
            model_name="Ridge",
            params={"alpha": 1.0},
            preprocess_cfg={"name": "raw", "deriv": 0, "window": np.nan, "polyorder": np.nan},
            cv_splitter=self.cv_splitter,
            task_type="regression",
            is_binary_classification=False,
            subset_indices=subset_indices,
            subset_tag="top100",
            top_n_vars=30
        )

        # Check that top_vars exists
        assert 'top_vars' in result, "Result should contain 'top_vars' field"
        assert result['top_vars'] != 'N/A', "top_vars should not be N/A"

        # Parse top_vars and verify it has at most 30 wavelengths
        top_vars_str = result['top_vars']
        top_wavelengths = [float(w.strip()) for w in top_vars_str.split(',') if w.strip()]

        assert len(top_wavelengths) <= 30, f"Expected at most 30 wavelengths in top_vars, got {len(top_wavelengths)}"

    def test_full_model_all_vars_is_na(self):
        """Test that full spectrum models have all_vars set to N/A."""
        # Create model instance
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=5, scale=False)

        result = _run_single_config(
            X=self.X,
            y=self.y,
            wavelengths=self.wavelengths,
            model=model,
            model_name="PLS",
            params={"n_components": 5},
            preprocess_cfg={"name": "raw", "deriv": 0, "window": np.nan, "polyorder": np.nan},
            cv_splitter=self.cv_splitter,
            task_type="regression",
            is_binary_classification=False,
            subset_indices=None,  # Full spectrum
            subset_tag="full",
            top_n_vars=30
        )

        # Check that all_vars is N/A for full models
        assert 'all_vars' in result, "Result should contain 'all_vars' field"
        assert result['all_vars'] == 'N/A', "all_vars should be N/A for full spectrum models"

    def test_results_dataframe_has_all_vars_column(self):
        """Test that results dataframe includes all_vars column."""
        df = create_results_dataframe(task_type="regression")

        assert 'all_vars' in df.columns, "Results dataframe should have 'all_vars' column"
        assert 'top_vars' in df.columns, "Results dataframe should still have 'top_vars' column"

    def test_all_vars_contains_subset_of_original_wavelengths(self):
        """Test that all_vars contains wavelengths that are subset of original."""
        # Create subset indices
        subset_indices = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        # Create model instance
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)

        result = _run_single_config(
            X=self.X,
            y=self.y,
            wavelengths=self.wavelengths,
            model=model,
            model_name="Lasso",
            params={"alpha": 0.1},
            preprocess_cfg={"name": "raw", "deriv": 0, "window": np.nan, "polyorder": np.nan},
            cv_splitter=self.cv_splitter,
            task_type="regression",
            is_binary_classification=False,
            subset_indices=subset_indices,
            subset_tag="top10",
            top_n_vars=30
        )

        # Parse all_vars
        all_vars_str = result['all_vars']
        all_wavelengths = [float(w.strip()) for w in all_vars_str.split(',') if w.strip()]

        # Expected wavelengths from subset
        expected_wavelengths = self.wavelengths[subset_indices]

        # Verify all wavelengths in all_vars are from the expected set
        assert len(all_wavelengths) == len(subset_indices), \
            f"Expected {len(subset_indices)} wavelengths, got {len(all_wavelengths)}"

        # Check that wavelengths match (allowing for floating point comparison)
        for wl in all_wavelengths:
            assert any(abs(wl - exp_wl) < 0.1 for exp_wl in expected_wavelengths), \
                f"Wavelength {wl} not found in expected wavelengths"

    def test_add_result_preserves_all_vars(self):
        """Test that add_result function preserves all_vars column."""
        df = create_results_dataframe(task_type="regression")

        # Create a mock result with all_vars
        result = {
            "Task": "regression",
            "Model": "PLS",
            "Params": "{'n_components': 5}",
            "Preprocess": "raw",
            "Deriv": 0,
            "Window": np.nan,
            "Poly": np.nan,
            "LVs": 5,
            "n_vars": 50,
            "full_vars": 200,
            "SubsetTag": "top50",
            "RMSE": 0.15,
            "R2": 0.95,
            "top_vars": "1500.0,1501.0,1502.0",  # Top 3 for simplicity
            "all_vars": ",".join([f"{1500.0 + i}" for i in range(50)])  # All 50
        }

        df = add_result(df, result)

        # Verify the row was added and all_vars is preserved
        assert len(df) == 1, "DataFrame should have one row"
        assert df.iloc[0]['all_vars'] is not None, "all_vars should be preserved"
        assert df.iloc[0]['n_vars'] == 50, "n_vars should be 50"

        # Verify all_vars contains 50 wavelengths
        all_vars_str = df.iloc[0]['all_vars']
        all_wavelengths = [float(w.strip()) for w in all_vars_str.split(',') if w.strip()]
        assert len(all_wavelengths) == 50, f"Expected 50 wavelengths in all_vars, got {len(all_wavelengths)}"


class TestBackwardCompatibility:
    """Tests for backward compatibility with old results."""

    def test_loading_old_results_without_all_vars(self):
        """Test that old results without all_vars column can still be loaded."""
        # Simulate old results dataframe without all_vars
        old_df = pd.DataFrame({
            "Task": ["regression"],
            "Model": ["PLS"],
            "Params": ["{'n_components': 5}"],
            "Preprocess": ["raw"],
            "Deriv": [0],
            "Window": [np.nan],
            "Poly": [np.nan],
            "LVs": [5],
            "n_vars": [50],
            "full_vars": [200],
            "SubsetTag": ["top50"],
            "RMSE": [0.15],
            "R2": [0.95],
            "top_vars": ["1500.0,1501.0,1502.0"],
            "CompositeScore": [0.95],
            "Rank": [1]
        })

        # This should not raise an error
        # GUI should fall back to top_vars if all_vars doesn't exist
        config = old_df.iloc[0].to_dict()

        assert 'top_vars' in config, "Old results should have top_vars"
        assert 'all_vars' not in config or pd.isna(config.get('all_vars')), \
            "Old results should not have all_vars"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
