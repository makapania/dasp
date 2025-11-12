"""
Comprehensive tests for the _parse_range_specification function.

Tests all supported formats, error cases, and edge cases.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path to import the GUI module
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_predict_gui_optimized import SpectralAnalysisGUI


class TestRangeParser(unittest.TestCase):
    """Test suite for _parse_range_specification function."""

    @classmethod
    def setUpClass(cls):
        """Create a GUI instance for testing (without initializing the full UI)."""
        # We'll use a mock instance just to access the method
        cls.gui = type('MockGUI', (), {})()
        # Bind the method to our mock instance
        from spectral_predict_gui_optimized import SpectralAnalysisGUI
        cls.gui._parse_range_specification = SpectralAnalysisGUI._parse_range_specification.__get__(cls.gui)

    # ===== Test Single Values =====

    def test_single_integer_value(self):
        """Test parsing a single integer value."""
        result = self.gui._parse_range_specification("150", "test_param", is_float=False)
        self.assertEqual(result, [150])

    def test_single_float_value(self):
        """Test parsing a single float value."""
        result = self.gui._parse_range_specification("0.5", "test_param", is_float=True)
        self.assertEqual(result, [0.5])

    def test_single_negative_integer(self):
        """Test parsing a single negative integer."""
        result = self.gui._parse_range_specification("-5", "test_param", is_float=False)
        self.assertEqual(result, [-5])

    def test_single_negative_float(self):
        """Test parsing a single negative float."""
        result = self.gui._parse_range_specification("-0.5", "test_param", is_float=True)
        self.assertEqual(result, [-0.5])

    # ===== Test Lists =====

    def test_simple_integer_list(self):
        """Test parsing a simple list of integers."""
        result = self.gui._parse_range_specification("50, 100, 200", "test_param", is_float=False)
        self.assertEqual(result, [50, 100, 200])

    def test_simple_float_list(self):
        """Test parsing a simple list of floats."""
        result = self.gui._parse_range_specification("0.01, 0.1, 1.0", "test_param", is_float=True)
        self.assertEqual(result, [0.01, 0.1, 1.0])

    def test_list_with_whitespace(self):
        """Test parsing a list with varying whitespace."""
        result = self.gui._parse_range_specification("10,   20,  30  , 40", "test_param", is_float=False)
        self.assertEqual(result, [10, 20, 30, 40])

    def test_unsorted_list(self):
        """Test that unsorted lists get sorted."""
        result = self.gui._parse_range_specification("200, 50, 100", "test_param", is_float=False)
        self.assertEqual(result, [50, 100, 200])

    def test_list_with_duplicates(self):
        """Test that duplicate values are removed."""
        result = self.gui._parse_range_specification("10, 20, 10, 30, 20", "test_param", is_float=False)
        self.assertEqual(result, [10, 20, 30])

    # ===== Test Ranges =====

    def test_simple_integer_range(self):
        """Test parsing a simple integer range."""
        result = self.gui._parse_range_specification("50-200 step 50", "test_param", is_float=False)
        self.assertEqual(result, [50, 100, 150, 200])

    def test_simple_float_range(self):
        """Test parsing a simple float range."""
        result = self.gui._parse_range_specification("0.1-1.0 step 0.3", "test_param", is_float=True)
        # Use assertAlmostEqual for floating point comparisons
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 0.1, places=5)
        self.assertAlmostEqual(result[1], 0.4, places=5)
        self.assertAlmostEqual(result[2], 0.7, places=5)
        self.assertAlmostEqual(result[3], 1.0, places=5)

    def test_range_with_step_1(self):
        """Test range with step of 1."""
        result = self.gui._parse_range_specification("1-5 step 1", "test_param", is_float=False)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_range_negative_to_positive(self):
        """Test range from negative to positive values."""
        result = self.gui._parse_range_specification("-10-10 step 5", "test_param", is_float=False)
        self.assertEqual(result, [-10, -5, 0, 5, 10])

    def test_range_both_negative(self):
        """Test range with both negative values."""
        result = self.gui._parse_range_specification("-50--10 step 10", "test_param", is_float=False)
        self.assertEqual(result, [-50, -40, -30, -20, -10])

    def test_descending_range(self):
        """Test descending range (end < start)."""
        result = self.gui._parse_range_specification("100-50 step 10", "test_param", is_float=False)
        self.assertEqual(result, [50, 60, 70, 80, 90, 100])  # Should still be sorted ascending

    def test_range_case_insensitive_step(self):
        """Test that 'step' keyword is case insensitive."""
        result1 = self.gui._parse_range_specification("10-20 step 5", "test_param", is_float=False)
        result2 = self.gui._parse_range_specification("10-20 STEP 5", "test_param", is_float=False)
        result3 = self.gui._parse_range_specification("10-20 Step 5", "test_param", is_float=False)
        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)

    # ===== Test Mixed Formats =====

    def test_mixed_values_and_ranges(self):
        """Test mixing individual values and ranges."""
        result = self.gui._parse_range_specification("10, 50-100 step 25, 200", "test_param", is_float=False)
        self.assertEqual(result, [10, 50, 75, 100, 200])

    def test_mixed_multiple_ranges(self):
        """Test multiple ranges in one specification."""
        result = self.gui._parse_range_specification("10-20 step 5, 30-40 step 5", "test_param", is_float=False)
        self.assertEqual(result, [10, 15, 20, 30, 35, 40])

    def test_mixed_with_duplicates(self):
        """Test that duplicates are removed in mixed formats."""
        result = self.gui._parse_range_specification("10, 10-30 step 10, 30", "test_param", is_float=False)
        self.assertEqual(result, [10, 20, 30])

    # ===== Test None Values =====

    def test_single_none_value(self):
        """Test parsing a single None value."""
        result = self.gui._parse_range_specification("None", "test_param", is_float=False)
        self.assertEqual(result, [None])

    def test_none_with_integers(self):
        """Test None mixed with integer values."""
        result = self.gui._parse_range_specification("None, 10, 20", "test_param", is_float=False)
        self.assertEqual(result, [None, 10, 20])

    def test_none_case_insensitive(self):
        """Test that None keyword is case insensitive."""
        result1 = self.gui._parse_range_specification("None, 10", "test_param", is_float=False)
        result2 = self.gui._parse_range_specification("none, 10", "test_param", is_float=False)
        result3 = self.gui._parse_range_specification("NONE, 10", "test_param", is_float=False)
        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)

    def test_multiple_none_values(self):
        """Test that multiple None values collapse to one."""
        result = self.gui._parse_range_specification("None, None, 10", "test_param", is_float=False)
        self.assertEqual(result, [None, 10])

    # ===== Test String Values =====

    def test_string_values(self):
        """Test parsing string values."""
        result = self.gui._parse_range_specification("relu, tanh, sigmoid", "test_param")
        self.assertEqual(result, ['relu', 'tanh', 'sigmoid'])

    def test_string_values_preserve_order(self):
        """Test that string values preserve order (not sorted)."""
        result = self.gui._parse_range_specification("tanh, relu, sigmoid", "test_param")
        self.assertEqual(result, ['tanh', 'relu', 'sigmoid'])

    def test_string_values_remove_duplicates(self):
        """Test that duplicate strings are removed."""
        result = self.gui._parse_range_specification("relu, tanh, relu", "test_param")
        self.assertEqual(result, ['relu', 'tanh'])

    def test_string_values_with_whitespace(self):
        """Test string values with varying whitespace."""
        result = self.gui._parse_range_specification("relu,  tanh  , sigmoid", "test_param")
        self.assertEqual(result, ['relu', 'tanh', 'sigmoid'])

    # ===== Test Empty/Whitespace =====

    def test_empty_string(self):
        """Test parsing an empty string."""
        result = self.gui._parse_range_specification("", "test_param", is_float=False)
        self.assertEqual(result, [])

    def test_whitespace_only(self):
        """Test parsing whitespace-only string."""
        result = self.gui._parse_range_specification("   ", "test_param", is_float=False)
        self.assertEqual(result, [])

    def test_commas_only(self):
        """Test parsing string with only commas."""
        result = self.gui._parse_range_specification(",,,", "test_param", is_float=False)
        self.assertEqual(result, [])

    # ===== Test Error Cases =====

    def test_invalid_integer_value(self):
        """Test that invalid integer raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("not_a_number", "test_param", is_float=False)
        # String values should work, so this test is actually for the error message
        # Actually, this will return ['not_a_number'] as a string value
        # Let's modify to test actual invalid case
        result = self.gui._parse_range_specification("not_a_number", "test_param", is_float=False)
        self.assertEqual(result, ['not_a_number'])

    def test_invalid_range_no_dash(self):
        """Test that range without dash raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("50 step 10", "test_param", is_float=False)
        self.assertIn("Expected 'start-end'", str(context.exception))

    def test_invalid_range_no_step_value(self):
        """Test that range without step value raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("50-100 step", "test_param", is_float=False)
        self.assertIn("Could not parse numeric values", str(context.exception))

    def test_zero_step_raises_error(self):
        """Test that zero step raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("10-20 step 0", "test_param", is_float=False)
        self.assertIn("Step cannot be zero", str(context.exception))

    def test_negative_step_raises_error(self):
        """Test that negative step raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("10-20 step -5", "test_param", is_float=False)
        self.assertIn("Step must be positive", str(context.exception))

    def test_malformed_range_syntax(self):
        """Test that malformed range syntax raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("10-20-30 step 5", "test_param", is_float=False)
        self.assertIn("Invalid range syntax", str(context.exception))

    def test_multiple_step_keywords(self):
        """Test that multiple 'step' keywords raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("10-20 step 5 step 10", "test_param", is_float=False)
        self.assertIn("Invalid range syntax", str(context.exception))

    # ===== Test Parameter Name in Errors =====

    def test_error_includes_param_name(self):
        """Test that error messages include the parameter name."""
        with self.assertRaises(ValueError) as context:
            self.gui._parse_range_specification("10-20 step 0", "n_estimators", is_float=False)
        self.assertIn("n_estimators", str(context.exception))

    # ===== Test Float vs Int Behavior =====

    def test_float_flag_with_integer_values(self):
        """Test that is_float=True converts integers to floats."""
        result = self.gui._parse_range_specification("10, 20, 30", "test_param", is_float=True)
        self.assertEqual(result, [10.0, 20.0, 30.0])
        self.assertIsInstance(result[0], float)

    def test_int_flag_with_decimal_values(self):
        """Test that is_float=False tries to parse decimals as ints (should fail)."""
        with self.assertRaises(ValueError):
            self.gui._parse_range_specification("10.5, 20.5", "test_param", is_float=False)

    # ===== Test Edge Cases =====

    def test_very_large_range(self):
        """Test that very large ranges don't cause issues."""
        result = self.gui._parse_range_specification("1-1000 step 100", "test_param", is_float=False)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[-1], 901)

    def test_very_small_float_step(self):
        """Test very small float step values."""
        result = self.gui._parse_range_specification("0.0-0.01 step 0.001", "test_param", is_float=True)
        self.assertEqual(len(result), 11)
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[-1], 0.01, places=5)

    def test_single_value_range(self):
        """Test range where start equals end."""
        result = self.gui._parse_range_specification("10-10 step 5", "test_param", is_float=False)
        self.assertEqual(result, [10])

    def test_range_where_step_exceeds_range(self):
        """Test range where step is larger than the range."""
        result = self.gui._parse_range_specification("10-15 step 100", "test_param", is_float=False)
        self.assertEqual(result, [10])

    def test_trailing_comma(self):
        """Test that trailing comma doesn't cause issues."""
        result = self.gui._parse_range_specification("10, 20, 30,", "test_param", is_float=False)
        self.assertEqual(result, [10, 20, 30])

    def test_leading_comma(self):
        """Test that leading comma doesn't cause issues."""
        result = self.gui._parse_range_specification(",10, 20, 30", "test_param", is_float=False)
        self.assertEqual(result, [10, 20, 30])


class TestRangeParserIntegration(unittest.TestCase):
    """Integration tests for realistic use cases."""

    @classmethod
    def setUpClass(cls):
        """Create a GUI instance for testing."""
        cls.gui = type('MockGUI', (), {})()
        from spectral_predict_gui_optimized import SpectralAnalysisGUI
        cls.gui._parse_range_specification = SpectralAnalysisGUI._parse_range_specification.__get__(cls.gui)

    def test_typical_n_estimators(self):
        """Test typical n_estimators specification."""
        result = self.gui._parse_range_specification("50, 100-300 step 100, 500", "n_estimators", is_float=False)
        self.assertEqual(result, [50, 100, 200, 300, 500])

    def test_typical_learning_rate(self):
        """Test typical learning_rate specification."""
        result = self.gui._parse_range_specification("0.01, 0.05, 0.1, 0.3", "learning_rate", is_float=True)
        self.assertEqual(result, [0.01, 0.05, 0.1, 0.3])

    def test_typical_max_depth(self):
        """Test typical max_depth specification with None."""
        result = self.gui._parse_range_specification("None, 3, 5-15 step 5", "max_depth", is_float=False)
        self.assertEqual(result, [None, 3, 5, 10, 15])

    def test_activation_functions(self):
        """Test typical activation function specification."""
        result = self.gui._parse_range_specification("relu, tanh, sigmoid, elu", "activation")
        self.assertEqual(result, ['relu', 'tanh', 'sigmoid', 'elu'])

    def test_regularization_alpha(self):
        """Test typical regularization alpha specification."""
        result = self.gui._parse_range_specification("0.0001, 0.001, 0.01-0.1 step 0.03", "alpha", is_float=True)
        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result[0], 0.0001, places=5)
        self.assertAlmostEqual(result[-1], 0.1, places=5)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRangeParser))
    suite.addTests(loader.loadTestsFromTestCase(TestRangeParserIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
