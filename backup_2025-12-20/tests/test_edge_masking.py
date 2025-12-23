"""Tests for edge-effect masking in variable selection.

This module tests the _apply_edge_mask function that prevents variable selection
from choosing wavelengths affected by Savitzky-Golay derivative boundary artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_predict.search import _apply_edge_mask


class TestEdgeMasking:
    """Test suite for edge-effect masking functionality."""

    def test_no_masking_without_derivatives(self):
        """Should not mask when no derivative is specified."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        preprocess_cfg = {"window": 5}  # No deriv key

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Should return original array unchanged
        np.testing.assert_array_equal(result, importances)
        # Should not modify original
        assert result is not importances

    def test_no_masking_with_deriv_zero(self):
        """Should not mask when derivative is 0 (no derivative)."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        preprocess_cfg = {"deriv": 0, "window": 5}

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Should return original array unchanged
        np.testing.assert_array_equal(result, importances)

    def test_no_masking_without_window(self):
        """Should not mask when window is not specified."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        preprocess_cfg = {"deriv": 1}  # No window key

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Should return original array unchanged
        np.testing.assert_array_equal(result, importances)

    def test_masking_with_first_derivative(self):
        """Should mask edges when first derivative is applied."""
        importances = np.array([0.9, 0.8, 0.3, 0.4, 0.5, 0.7, 0.95])
        preprocess_cfg = {"deriv": 1, "window": 5}

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Edge margin = 5 // 2 = 2
        # First 2 and last 2 should be zeroed
        expected = np.array([0.0, 0.0, 0.3, 0.4, 0.5, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_masking_with_second_derivative(self):
        """Should mask edges when second derivative is applied."""
        importances = np.array([0.8, 0.7, 0.3, 0.4, 0.5, 0.6, 0.85])
        preprocess_cfg = {"deriv": 2, "window": 5}

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Edge margin = 5 // 2 = 2
        # First 2 and last 2 should be zeroed
        expected = np.array([0.0, 0.0, 0.3, 0.4, 0.5, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_masking_preserves_middle_values(self):
        """Should not modify importance values in the middle region."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        preprocess_cfg = {"deriv": 1, "window": 7}

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Edge margin = 7 // 2 = 3
        # First 3 and last 3 should be zeroed, middle unchanged
        expected = np.array([0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_masking_with_different_window_sizes(self):
        """Should mask different amounts based on window size."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Window 3: margin = 1
        cfg_3 = {"deriv": 1, "window": 3}
        result_3 = _apply_edge_mask(importances, cfg_3)
        expected_3 = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0])
        np.testing.assert_array_equal(result_3, expected_3)

        # Window 5: margin = 2
        cfg_5 = {"deriv": 1, "window": 5}
        result_5 = _apply_edge_mask(importances, cfg_5)
        expected_5 = np.array([0.0, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0])
        np.testing.assert_array_equal(result_5, expected_5)

        # Window 9: margin = 4
        cfg_9 = {"deriv": 1, "window": 9}
        result_9 = _apply_edge_mask(importances, cfg_9)
        expected_9 = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result_9, expected_9)

    def test_safety_check_prevents_zero_all(self):
        """Should not zero entire array when window is too large."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        preprocess_cfg = {"deriv": 1, "window": 11}  # margin = 5, array len = 5

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Safety check: 2 * 5 >= 5, so return unchanged
        np.testing.assert_array_equal(result, importances)

    def test_safety_check_edge_case_exact_boundary(self):
        """Should not mask when 2*margin exactly equals array length."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        preprocess_cfg = {"deriv": 1, "window": 7}  # margin = 3, array len = 6

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Safety check: 2 * 3 >= 6, so return unchanged
        np.testing.assert_array_equal(result, importances)

    def test_does_not_modify_original_array(self):
        """Should not modify the original importances array."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        original_copy = importances.copy()
        preprocess_cfg = {"deriv": 1, "window": 5}

        _apply_edge_mask(importances, preprocess_cfg)

        # Original should be unchanged
        np.testing.assert_array_equal(importances, original_copy)

    def test_empty_config(self):
        """Should handle empty preprocessing config gracefully."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        preprocess_cfg = {}

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Should return original unchanged
        np.testing.assert_array_equal(result, importances)

    def test_realistic_spectral_scenario(self):
        """Test with realistic spectral data dimensions."""
        # Simulate 100 wavelengths with varying importance
        np.random.seed(42)
        importances = np.random.rand(100)
        importances[0:5] = 0.9  # High importance at edges (artifacts)
        importances[-5:] = 0.9  # High importance at edges (artifacts)
        importances[45:55] = 0.95  # High importance in middle (real signal)

        preprocess_cfg = {"deriv": 1, "window": 11}  # Common SG parameters

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Edge margin = 11 // 2 = 5
        # First 5 and last 5 should be zeroed
        assert np.all(result[:5] == 0.0), "First 5 elements should be zeroed"
        assert np.all(result[-5:] == 0.0), "Last 5 elements should be zeroed"
        # Middle region should be unchanged
        np.testing.assert_array_equal(result[45:55], importances[45:55])

    def test_odd_even_window_sizes(self):
        """Should handle both odd and even window sizes correctly."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        # Odd window: 5 // 2 = 2
        cfg_odd = {"deriv": 1, "window": 5}
        result_odd = _apply_edge_mask(importances, cfg_odd)
        assert result_odd[0] == 0.0 and result_odd[1] == 0.0
        assert result_odd[-1] == 0.0 and result_odd[-2] == 0.0

        # Even window: 6 // 2 = 3
        cfg_even = {"deriv": 1, "window": 6}
        result_even = _apply_edge_mask(importances, cfg_even)
        assert result_even[0] == 0.0 and result_even[1] == 0.0 and result_even[2] == 0.0
        assert result_even[-1] == 0.0 and result_even[-2] == 0.0 and result_even[-3] == 0.0

    def test_small_array_large_window(self):
        """Should handle small arrays with proportionally large windows."""
        importances = np.array([0.1, 0.2, 0.3])
        preprocess_cfg = {"deriv": 1, "window": 5}  # margin = 2, len = 3

        result = _apply_edge_mask(importances, preprocess_cfg)

        # Safety check: 2 * 2 >= 3, return unchanged
        np.testing.assert_array_equal(result, importances)

    def test_minimum_viable_masking(self):
        """Should mask when barely viable (2*margin < len)."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        preprocess_cfg = {"deriv": 1, "window": 3}  # margin = 1, len = 5

        result = _apply_edge_mask(importances, preprocess_cfg)

        # 2 * 1 < 5, so masking should occur
        expected = np.array([0.0, 0.2, 0.3, 0.4, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_none_values_in_config(self):
        """Should handle None values in config gracefully."""
        importances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # None deriv
        cfg_none_deriv = {"deriv": None, "window": 5}
        result = _apply_edge_mask(importances, cfg_none_deriv)
        np.testing.assert_array_equal(result, importances)

        # None window
        cfg_none_window = {"deriv": 1, "window": None}
        result = _apply_edge_mask(importances, cfg_none_window)
        np.testing.assert_array_equal(result, importances)
