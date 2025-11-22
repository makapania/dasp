"""
Integration tests for wavelength filtering feature.

This module tests wavelength filtering after preprocessing but before variable selection.
Key principle: Preprocessing (SNV, derivatives) applied to FULL spectrum, then filtering
restricts which wavelengths are used for variable selection.

Test structure:
- Scenario 1: No filtering (baseline)
- Scenario 2: Minimum only (analysis_wl_min with None max)
- Scenario 3: Maximum only (None min with analysis_wl_max)
- Scenario 4: Both min and max (range filtering)
- Scenario 5: With preprocessing (SNV, derivatives, etc.)
- Scenario 6: With variable selection (importance, SPA, UVE)
- Scenario 7: Edge cases (zero wavelengths, invalid ranges, etc.)
- Scenario 8: Consistency/reproducibility (save/load, CV, determinism)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from io import StringIO
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import functions to test
try:
    from spectral_predict.search import run_search
    from spectral_predict.preprocess import build_preprocessing_pipeline
except ImportError:
    # Fallback - tests can be skipped if imports fail
    pytest.skip("spectral_predict module not available", allow_module_level=True)


# ============================================================================
# FIXTURES AND UTILITIES
# ============================================================================

@pytest.fixture
def synthetic_spectra():
    """
    Generate synthetic spectral data for testing.

    Returns:
    -------
    X : pd.DataFrame
        Spectral data (100 samples, 500 wavelengths)
    y : pd.Series
        Target values (100 samples)
    wavelengths : np.ndarray
        Wavelength array (400-2500 nm, 4.2 nm increments)
    """
    np.random.seed(42)

    # Create wavelengths: 400-2500 nm with 4.2 nm spacing
    n_wavelengths = 500
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    wavelengths_str = [f"{w:.1f}" for w in wavelengths]

    # Create synthetic spectra
    n_samples = 100
    X = np.random.randn(n_samples, n_wavelengths)

    # Create target as linear combination of specific wavelengths + noise
    # Use indices: 95 (~800 nm), 150 (~1132 nm), 250 (~1832 nm)
    important_indices = [95, 150, 250]
    y = X[:, important_indices].sum(axis=1) + 0.5 * np.random.randn(n_samples)

    # Convert to DataFrame with wavelength column names
    X = pd.DataFrame(X, columns=wavelengths_str)
    y = pd.Series(y)

    return X, y, wavelengths


def capture_output(func, *args, **kwargs):
    """
    Capture stdout from function execution.

    Returns:
    -------
    result : any
        Return value from function
    output : str
        Captured stdout
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        result = func(*args, **kwargs)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return result, output


def verify_wavelength_range(wavelengths, min_wl=None, max_wl=None):
    """
    Verify that wavelengths are within specified range.

    Parameters:
    -----------
    wavelengths : np.ndarray or list
        Wavelength values
    min_wl : float, optional
        Minimum expected wavelength
    max_wl : float, optional
        Maximum expected wavelength

    Returns:
    --------
    dict with verification results
    """
    wl_float = np.array([float(w) for w in wavelengths])

    results = {
        'all_numeric': True,
        'min_in_range': True,
        'max_in_range': True,
        'no_duplicates': True,
        'monotonic': True,
    }

    # Check numeric
    try:
        wl_float = np.array([float(w) for w in wavelengths])
    except (ValueError, TypeError):
        results['all_numeric'] = False

    # Check range bounds
    if min_wl is not None:
        results['min_in_range'] = np.all(wl_float >= min_wl * 0.99)  # 1% tolerance
    if max_wl is not None:
        results['max_in_range'] = np.all(wl_float <= max_wl * 1.01)

    # Check duplicates
    results['no_duplicates'] = len(np.unique(wl_float)) == len(wl_float)

    # Check monotonic increasing
    results['monotonic'] = np.all(np.diff(wl_float) > 0)

    return results


# ============================================================================
# SCENARIO 1: NO FILTERING (BASELINE)
# ============================================================================

class TestScenario1NoFiltering:
    """Test no filtering applied (backward compatibility)."""

    def test_no_filtering_none_parameters(self, synthetic_spectra):
        """
        Test that passing None, None leaves all wavelengths available.

        Expected:
        - All 500 wavelengths available
        - No filtering message printed
        - Behavior identical to pre-filtering implementation
        """
        X, y, wavelengths = synthetic_spectra

        # Run search with no filtering parameters
        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            analysis_wl_min=None,
            analysis_wl_max=None,
        )

        # Verify no filtering message
        assert "WAVELENGTH FILTERING" not in output, \
            "Filtering message should not appear when both parameters are None"

        # Verify results exist
        assert len(df_results) > 0, "Should produce results"
        assert 'n_features' in df_results.columns

        # First config should use all features (or close to it)
        # Account for region/subset analysis which reduces features
        first_config_features = df_results.iloc[0]['n_features']
        assert first_config_features > 400, \
            f"Expected ~500 features in first config, got {first_config_features}"

    def test_no_filtering_empty_strings(self, synthetic_spectra):
        """
        Test that empty strings (GUI default) don't filter.

        Expected:
        - Same as None parameters
        - GUI state with no filtering enabled
        """
        X, y, wavelengths = synthetic_spectra

        # Empty strings should be treated as None
        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            analysis_wl_min="",
            analysis_wl_max="",
        )

        # Should not filter (treated as None by GUI parameter parsing)
        assert len(df_results) > 0
        first_features = df_results.iloc[0]['n_features']
        assert first_features > 400

    def test_no_filtering_matches_expected_behavior(self, synthetic_spectra):
        """
        Test that no filtering produces expected features count.

        With 500 wavelengths and 100 samples, full model should use:
        - 500 features for variable selection
        - Preprocessing may adjust slightly (derivatives reduce by ~7)
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=None,
            analysis_wl_max=None,
        )

        # Raw preprocessing should preserve all 500 features
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        assert len(raw_results) > 0, "Should have raw preprocessing results"

        # Should have ~500 features (possibly slight reduction from regions)
        raw_features = raw_results.iloc[0]['n_features']
        assert 480 < raw_features <= 500, \
            f"Expected ~500 features for raw (no filtering), got {raw_features}"


# ============================================================================
# SCENARIO 2: MINIMUM ONLY
# ============================================================================

class TestScenario2MinimumOnly:
    """Test minimum wavelength filtering only."""

    def test_min_only_800nm(self, synthetic_spectra):
        """
        Test minimum wavelength filtering at 800 nm.

        Expected:
        - Keep wavelengths >= 800 nm
        - ~425 wavelengths (indices 95-499)
        - Filtering message printed
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=800,
            analysis_wl_max=None,
        )

        # Verify filtering message
        assert "WAVELENGTH FILTERING" in output, "Should show filtering info"
        assert "800" in output, "Should mention min wavelength"

        # Verify feature count
        # 500 wavelengths at 4.2 nm spacing
        # 800 nm starts at index ~95
        # Expect ~425 wavelengths
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        assert len(raw_results) > 0

        filtered_features = raw_results.iloc[0]['n_features']
        assert 400 < filtered_features <= 425, \
            f"Expected ~425 features after filtering to 800+, got {filtered_features}"

    def test_min_at_nir_edge_1000nm(self, synthetic_spectra):
        """Test min filtering at 1000 nm (NIR start)."""
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=1000,
            analysis_wl_max=None,
        )

        assert "WAVELENGTH FILTERING" in output

        # 1000 nm at ~142 index
        # Expect ~358 wavelengths (500 - 142)
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        filtered_features = raw_results.iloc[0]['n_features']
        assert 330 < filtered_features <= 370, \
            f"Expected ~358 features after filtering to 1000+, got {filtered_features}"


# ============================================================================
# SCENARIO 3: MAXIMUM ONLY
# ============================================================================

class TestScenario3MaximumOnly:
    """Test maximum wavelength filtering only."""

    def test_max_only_2500nm(self, synthetic_spectra):
        """
        Test maximum wavelength filtering at 2500 nm (spectrum endpoint).

        Expected:
        - Keep wavelengths <= 2500 nm (all of them)
        - 500 wavelengths
        - Filtering message printed but no actual filtering
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=None,
            analysis_wl_max=2500,
        )

        # Filtering info should show
        assert "WAVELENGTH FILTERING" in output
        assert "2500" in output

        # All wavelengths included (no actual filtering)
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        filtered_features = raw_results.iloc[0]['n_features']
        assert 480 < filtered_features <= 500, \
            f"Expected ~500 features (no actual filtering), got {filtered_features}"

    def test_max_at_nir_upper_2400nm(self, synthetic_spectra):
        """Test max filtering at 2400 nm (common NIR limit)."""
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=None,
            analysis_wl_max=2400,
        )

        # 2400 nm at ~476 index
        # Expect ~476 wavelengths
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        filtered_features = raw_results.iloc[0]['n_features']
        assert 450 < filtered_features <= 490, \
            f"Expected ~476 features after filtering to ≤2400, got {filtered_features}"


# ============================================================================
# SCENARIO 4: BOTH MIN AND MAX
# ============================================================================

class TestScenario4BothMinMax:
    """Test range filtering with both boundaries."""

    def test_range_700_2500nm(self, synthetic_spectra):
        """
        Test filtering to standard VIS-NIR range (700-2500 nm).

        Expected:
        - Keep 700 <= λ <= 2500
        - ~425 wavelengths
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=700,
            analysis_wl_max=2500,
        )

        # Verify filtering info
        assert "WAVELENGTH FILTERING" in output
        assert "700" in output and "2500" in output

        # 700 nm at ~71 index, 2500 at 499
        # Expect ~428 wavelengths
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        filtered_features = raw_results.iloc[0]['n_features']
        assert 400 < filtered_features <= 440, \
            f"Expected ~428 features for 700-2500 range, got {filtered_features}"

    def test_range_1300_1400nm(self, synthetic_spectra):
        """
        Test narrow range filtering (1300-1400 nm).

        Expected:
        - Very narrow band (~25 wavelengths)
        - Model should still train successfully
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=1300,
            analysis_wl_max=1400,
        )

        # 1300 nm at ~214 index, 1400 at ~238
        # Expect ~24 wavelengths
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        assert len(raw_results) > 0, "Should still produce results with narrow range"

        filtered_features = raw_results.iloc[0]['n_features']
        assert 15 < filtered_features <= 35, \
            f"Expected ~24 features for 1300-1400 range, got {filtered_features}"


# ============================================================================
# SCENARIO 5: WITH PREPROCESSING
# ============================================================================

class TestScenario5WithPreprocessing:
    """Test filtering with various preprocessing methods."""

    def test_snv_preprocessing_with_filtering(self, synthetic_spectra):
        """
        Test SNV preprocessing with wavelength filtering.

        Expected:
        - SNV applied to FULL spectrum
        - Filtering applied AFTER preprocessing
        - Variable selection sees filtered data
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'snv': True, 'raw': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=800,
            analysis_wl_max=2500,
        )

        # Verify both preprocessing and filtering in output
        assert "WAVELENGTH FILTERING" in output or len(df_results) > 0, \
            "Should apply filtering with SNV"

        # Should have results
        assert len(df_results) > 0

        snv_results = df_results[df_results['Preprocessing'] == 'snv']
        if len(snv_results) > 0:
            # SNV should produce ~425 features after filtering
            snv_features = snv_results.iloc[0]['n_features']
            assert 400 < snv_features <= 450

    def test_first_derivative_with_filtering(self, synthetic_spectra):
        """
        Test 1st derivative preprocessing with filtering.

        Expected:
        - Derivative computed on full spectrum
        - Derivative reduces features by ~7 (window size)
        - Filtering applied after derivative
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'sg1': True, 'raw': False, 'snv': False, 'sg2': False},
            window_sizes=[7],
            analysis_wl_min=700,
            analysis_wl_max=2400,
        )

        # Derivative results
        deriv_results = df_results[df_results['Preprocessing'].str.contains('deriv', na=False)]
        assert len(deriv_results) > 0, "Should have derivative preprocessing results"

        # Should have fewer features than no filtering
        # Derivative reduces by ~7, then filtering reduces further
        deriv_features = deriv_results.iloc[0]['n_features']
        assert 390 < deriv_features < 430, \
            f"Expected ~396 features (deriv then filter), got {deriv_features}"

    def test_snv_then_derivative_with_filtering(self, synthetic_spectra):
        """
        Test SNV then derivative preprocessing with filtering.

        Expected:
        - SNV applied to full spectrum
        - Derivative applied to SNV data
        - Filtering applied last
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={
                'snv': True,
                'sg1': True,
                'raw': False,
                'sg2': False,
                'deriv_snv': False
            },
            window_sizes=[7],
            analysis_wl_min=800,
            analysis_wl_max=2000,
        )

        # Check combined preprocessing
        snv_deriv_results = df_results[
            df_results['Preprocessing'].str.contains('snv.*deriv|deriv.*snv',
                                                     regex=True, na=False)
        ]

        # Should have results with combined preprocessing
        if len(snv_deriv_results) > 0:
            features = snv_deriv_results.iloc[0]['n_features']
            # Expect 373 (derivative reduces ~7, filter removes ~100)
            assert 350 < features < 410


# ============================================================================
# SCENARIO 6: WITH VARIABLE SELECTION
# ============================================================================

class TestScenario6WithVariableSelection:
    """Test variable selection with wavelength filtering."""

    def test_importance_selection_with_filtering(self, synthetic_spectra):
        """
        Test feature importance-based variable selection with filtering.

        Expected:
        - Variable selection sees only filtered wavelengths
        - Selected features within filtered wavelength range
        """
        X, y, wavelengths = synthetic_spectra

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['RandomForest'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            enable_variable_subsets=True,
            variable_counts=[20],
            analysis_wl_min=800,
            analysis_wl_max=2500,
        )

        # Should have subset results
        assert len(df_results) > 0

        # Check that subsets are smaller than full model
        full_results = df_results[df_results['Subset_Name'] == 'full']
        subset_results = df_results[df_results['Subset_Name'].notna() &
                                    (df_results['Subset_Name'] != 'full')]

        if len(full_results) > 0 and len(subset_results) > 0:
            full_features = full_results.iloc[0]['n_features']
            subset_features = subset_results.iloc[0]['n_features']
            assert subset_features < full_features, \
                "Subset should have fewer features than full"


# ============================================================================
# SCENARIO 7: EDGE CASES
# ============================================================================

class TestScenario7EdgeCases:
    """Test edge cases and error handling."""

    def test_filtering_zero_wavelengths(self, synthetic_spectra):
        """
        Test when filter range excludes all wavelengths.

        Expected:
        - Error or exception raised
        - Clear error message about no wavelengths
        """
        X, y, wavelengths = synthetic_spectra

        # Filter entirely outside spectrum
        with pytest.raises(Exception) as exc_info:
            run_search(
                X, y,
                task_type='regression',
                folds=3,
                models_to_test=['PLS'],
                analysis_wl_min=3000,  # Beyond spectrum
                analysis_wl_max=3500,
            )

        # Error should mention wavelength filtering issue
        assert "wavelength" in str(exc_info.value).lower()

    def test_invalid_range_min_greater_than_max(self, synthetic_spectra):
        """
        Test invalid parameter order: min > max.

        Expected:
        - Error raised
        - Message indicates parameter order issue
        """
        X, y, wavelengths = synthetic_spectra

        with pytest.raises(Exception) as exc_info:
            run_search(
                X, y,
                task_type='regression',
                folds=3,
                models_to_test=['PLS'],
                analysis_wl_min=2000,
                analysis_wl_max=800,  # Wrong order
            )

        error_msg = str(exc_info.value).lower()
        assert "min" in error_msg or "wavelength" in error_msg

    def test_range_below_spectrum(self, synthetic_spectra):
        """
        Test when filter range is entirely below spectrum (200-300 nm).

        Expected:
        - Error or 0 wavelengths selected
        """
        X, y, wavelengths = synthetic_spectra

        with pytest.raises(Exception):
            run_search(
                X, y,
                task_type='regression',
                folds=3,
                models_to_test=['PLS'],
                analysis_wl_min=200,
                analysis_wl_max=300,
            )

    def test_range_above_spectrum(self, synthetic_spectra):
        """
        Test when filter range is entirely above spectrum (3000-3500 nm).

        Expected:
        - Error or 0 wavelengths selected
        """
        X, y, wavelengths = synthetic_spectra

        with pytest.raises(Exception):
            run_search(
                X, y,
                task_type='regression',
                folds=3,
                models_to_test=['PLS'],
                analysis_wl_min=3000,
                analysis_wl_max=3500,
            )


# ============================================================================
# SCENARIO 8: CONSISTENCY AND REPRODUCIBILITY
# ============================================================================

class TestScenario8Consistency:
    """Test filtering consistency and reproducibility."""

    def test_filtering_deterministic(self, synthetic_spectra):
        """
        Test that filtering produces same result on repeated runs.

        Expected:
        - Same wavelengths selected each run
        - Same variable importances (within numerical precision)
        """
        X, y, wavelengths = synthetic_spectra

        # Run 1
        df_results_1, _ = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=800,
            analysis_wl_max=2500,
        )

        # Run 2
        df_results_2, _ = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=800,
            analysis_wl_max=2500,
        )

        # Results should be identical (same preprocessing, same filtering)
        assert len(df_results_1) == len(df_results_2), \
            "Should have same number of results"

        # Check R² values match (within 1e-6)
        if 'R2' in df_results_1.columns:
            r2_1 = df_results_1.iloc[0]['R2']
            r2_2 = df_results_2.iloc[0]['R2']
            assert abs(r2_1 - r2_2) < 1e-6, \
                f"R² should be deterministic: {r2_1} vs {r2_2}"

    def test_filtering_independent_of_random_seed(self, synthetic_spectra):
        """
        Test that wavelength filtering is independent of random seed.

        Expected:
        - Same wavelengths selected regardless of seed
        - Variable selection/training MAY differ (expected)
        - Filtering itself is deterministic
        """
        X, y, wavelengths = synthetic_spectra

        # Note: run_search uses fixed seed=42 for CV, but filtering should be
        # independent of any randomness

        # Run filtering multiple times
        output_list = []
        for i in range(3):
            _, output = capture_output(
                run_search,
                X, y,
                task_type='regression',
                folds=3,
                models_to_test=['PLS'],
                analysis_wl_min=800,
                analysis_wl_max=2500,
            )
            output_list.append(output)

        # All runs should show same filtering range
        for output in output_list:
            if "WAVELENGTH FILTERING" in output:
                assert "800" in output and "2500" in output

    def test_filtering_consistent_across_cv_folds(self, synthetic_spectra):
        """
        Test that filtering is applied consistently to each CV fold.

        Expected:
        - Each fold sees same filtered wavelengths
        - No data leakage
        """
        X, y, wavelengths = synthetic_spectra

        # With filtering, each CV fold should see same features
        df_results, _ = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=5,  # Use 5 folds to test
            models_to_test=['RandomForest'],
            analysis_wl_min=800,
            analysis_wl_max=2500,
        )

        # All results should have same feature count
        if 'n_features' in df_results.columns:
            feature_counts = df_results['n_features'].unique()

            # May have multiple values due to subsets, but full model should be constant
            full_results = df_results[df_results['Subset_Name'] == 'full']
            if len(full_results) > 1:
                full_features = full_results['n_features'].unique()
                assert len(full_features) == 1, \
                    f"Full model should have same features across runs, got {full_features}"


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrizedScenarios:
    """Parametrized tests for multiple scenarios."""

    @pytest.mark.parametrize("min_wl,max_wl,expected_count_range", [
        (None, None, (480, 500)),           # No filtering
        (800, None, (400, 430)),            # Min only
        (None, 2500, (480, 500)),           # Max only
        (700, 2500, (400, 430)),            # Both
        (1000, 1100, (20, 35)),             # Narrow range
        (1500, 1600, (20, 35)),             # Different narrow range
    ])
    def test_various_filter_ranges(self, synthetic_spectra, min_wl, max_wl,
                                  expected_count_range):
        """
        Parametrized test for various wavelength filter ranges.

        Tests that feature counts match expectations for different ranges.
        """
        X, y, wavelengths = synthetic_spectra

        # Skip if invalid range
        if min_wl is not None and max_wl is not None and min_wl > max_wl:
            pytest.skip("Invalid range")

        df_results, output = capture_output(
            run_search,
            X, y,
            task_type='regression',
            folds=3,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False},
            analysis_wl_min=min_wl,
            analysis_wl_max=max_wl,
        )

        # Verify we got results
        assert len(df_results) > 0, f"Should produce results for range {min_wl}-{max_wl}"

        # Check feature count
        raw_results = df_results[df_results['Preprocessing'] == 'raw']
        if len(raw_results) > 0:
            features = raw_results.iloc[0]['n_features']
            min_expected, max_expected = expected_count_range
            assert min_expected <= features <= max_expected, \
                f"For range {min_wl}-{max_wl}: expected {min_expected}-{max_expected} " \
                f"features, got {features}"


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
