"""
Unit tests for hyperparameter widget helper functions.

Tests the _create_parameter_grid_control and _extract_parameter_values
helper functions in spectral_predict_gui_optimized.py.
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch
import tkinter as tk

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MockApp:
    """Mock SpectralPredictApp with necessary methods and attributes."""

    def __init__(self):
        """Initialize mock app with GUI color scheme."""
        self.colors = {
            'bg': '#1e1e1e',
            'card_bg': '#2d2d2d',
            'text': '#ffffff',
            'text_inverse': '#000000',
            'accent': '#0078d4',
            'panel': '#252525'
        }
        self.root = tk.Tk()
        self.root.withdraw()  # Hide window during tests

    def _parse_range_specification(self, spec_string, param_name="parameter", is_float=False):
        """
        Mock implementation of range parser (simplified version).
        In real implementation, this would be imported from the actual GUI.
        """
        if not spec_string or not spec_string.strip():
            return []

        values = []
        segments = [seg.strip() for seg in spec_string.split(',')]

        for segment in segments:
            if ' step ' in segment.lower():
                # Parse range: "start-end step increment"
                parts = segment.lower().split(' step ')
                if len(parts) != 2:
                    raise ValueError(f"Invalid range syntax: {segment}")

                range_part, step_part = parts
                if '-' not in range_part:
                    raise ValueError(f"Invalid range syntax: {segment}")

                # Simple range parsing (not handling negative numbers)
                start_str, end_str = range_part.split('-', 1)
                if is_float:
                    start = float(start_str.strip())
                    end = float(end_str.strip())
                    step = float(step_part.strip())
                else:
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    step = int(step_part.strip())

                # Generate range
                current = start
                while current <= end:
                    values.append(current)
                    current += step

            elif segment.lower() == 'none':
                values.append(None)

            else:
                # Single value
                try:
                    if is_float:
                        values.append(float(segment))
                    else:
                        values.append(int(segment))
                except ValueError:
                    # Might be a string value
                    values.append(segment)

        # Remove duplicates and sort (handle None values)
        none_values = [v for v in values if v is None]
        numeric_values = [v for v in values if v is not None and not isinstance(v, str)]
        string_values = [v for v in values if isinstance(v, str)]

        result = []
        if none_values:
            result.append(None)
        result.extend(sorted(set(numeric_values)))
        result.extend(string_values)

        return result

    def _create_parameter_grid_control(self, parent, param_name, param_label,
                                       checkbox_values, default_checked=None,
                                       is_float=False, allow_string_values=False,
                                       help_text=None):
        """
        Create unified hyperparameter control with checkboxes + custom entry.
        This is the function we're testing.
        """
        # Import the real implementation from GUI
        # For testing purposes, we'll use a simplified version
        from spectral_predict_gui_optimized import SpectralPredictApp

        # Create a real instance temporarily to access the method
        # (This is a bit hacky but necessary for testing)
        real_app = SpectralPredictApp(self.root)

        # Call the real method
        result = real_app._create_parameter_grid_control(
            parent, param_name, param_label, checkbox_values,
            default_checked, is_float, allow_string_values, help_text
        )

        return result

    def _extract_parameter_values(self, control_dict, param_name, is_float=False,
                                  allow_string_values=False):
        """
        Extract final parameter list from control dict.
        This is the function we're testing.
        """
        from spectral_predict_gui_optimized import SpectralPredictApp

        # Create a real instance temporarily
        real_app = SpectralPredictApp(self.root)

        # Call the real method
        result = real_app._extract_parameter_values(
            control_dict, param_name, is_float, allow_string_values
        )

        return result

    def cleanup(self):
        """Clean up test resources."""
        if hasattr(self, 'root') and self.root:
            self.root.destroy()


class TestParameterWidgetCreation(unittest.TestCase):
    """Test _create_parameter_grid_control function."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = MockApp()
        self.parent = tk.Frame(self.app.root, bg=self.app.colors['card_bg'])

    def tearDown(self):
        """Clean up after tests."""
        self.app.cleanup()

    def test_create_widget_basic(self):
        """Test basic widget creation with integer values."""
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500]
        )

        # Verify structure
        self.assertIn('checkboxes', control)
        self.assertIn('custom_entry', control)
        self.assertIn('frame', control)
        self.assertIn('label', control)

        # Verify checkboxes
        self.assertEqual(len(control['checkboxes']), 4)
        self.assertIn(50, control['checkboxes'])
        self.assertIn(500, control['checkboxes'])

        # Verify all checkboxes are unchecked by default
        for var in control['checkboxes'].values():
            self.assertFalse(var.get())

    def test_create_widget_with_defaults(self):
        """Test widget creation with default checked values."""
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500],
            default_checked=[100, 200]
        )

        # Verify default checkboxes are checked
        self.assertTrue(control['checkboxes'][100].get())
        self.assertTrue(control['checkboxes'][200].get())
        self.assertFalse(control['checkboxes'][50].get())
        self.assertFalse(control['checkboxes'][500].get())

    def test_create_widget_float_values(self):
        """Test widget creation with float values."""
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='learning_rate',
            param_label='Learning Rate',
            checkbox_values=[0.001, 0.01, 0.1, 1.0],
            is_float=True
        )

        # Verify float checkboxes exist
        self.assertEqual(len(control['checkboxes']), 4)
        self.assertIn(0.001, control['checkboxes'])
        self.assertIn(1.0, control['checkboxes'])

    def test_create_widget_string_values(self):
        """Test widget creation with string values."""
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='activation',
            param_label='Activation Function',
            checkbox_values=['relu', 'tanh', 'sigmoid'],
            allow_string_values=True
        )

        # Verify string checkboxes exist
        self.assertEqual(len(control['checkboxes']), 3)
        self.assertIn('relu', control['checkboxes'])
        self.assertIn('tanh', control['checkboxes'])

    def test_create_widget_with_none(self):
        """Test widget creation with None value."""
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='max_depth',
            param_label='Max Depth',
            checkbox_values=[None, 5, 10, 20]
        )

        # Verify None checkbox exists
        self.assertIn(None, control['checkboxes'])


class TestParameterValueExtraction(unittest.TestCase):
    """Test _extract_parameter_values function."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = MockApp()
        self.parent = tk.Frame(self.app.root, bg=self.app.colors['card_bg'])

    def tearDown(self):
        """Clean up after tests."""
        self.app.cleanup()

    def test_extract_checkboxes_only(self):
        """Test extraction with only checkboxes selected."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500]
        )

        # Check some boxes
        control['checkboxes'][100].set(True)
        control['checkboxes'][200].set(True)

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'n_estimators', is_float=False
        )

        # Verify
        self.assertEqual(values, [100, 200])

    def test_extract_custom_entry_only(self):
        """Test extraction with only custom entry."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500]
        )

        # Set custom entry (clear placeholder first)
        control['custom_entry'].set('10, 20, 30')

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'n_estimators', is_float=False
        )

        # Verify
        self.assertEqual(values, [10, 20, 30])

    def test_extract_combined(self):
        """Test extraction with both checkboxes and custom entry."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500]
        )

        # Check some boxes
        control['checkboxes'][100].set(True)
        control['checkboxes'][200].set(True)

        # Set custom entry
        control['custom_entry'].set('10, 20, 300')

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'n_estimators', is_float=False
        )

        # Verify - should combine and sort
        self.assertEqual(values, [10, 20, 100, 200, 300])

    def test_extract_removes_duplicates(self):
        """Test that extraction removes duplicate values."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500]
        )

        # Check box for 100
        control['checkboxes'][100].set(True)

        # Also add 100 in custom entry
        control['custom_entry'].set('50, 100, 150')

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'n_estimators', is_float=False
        )

        # Verify - 100 should appear only once
        self.assertEqual(values, [50, 100, 150])

    def test_extract_with_range(self):
        """Test extraction with range specification in custom entry."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500]
        )

        # Set custom entry with range
        control['custom_entry'].set('10-30 step 10')

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'n_estimators', is_float=False
        )

        # Verify - should expand range
        self.assertEqual(values, [10, 20, 30])

    def test_extract_float_values(self):
        """Test extraction with float values."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='learning_rate',
            param_label='Learning Rate',
            checkbox_values=[0.001, 0.01, 0.1, 1.0],
            is_float=True
        )

        # Check some boxes
        control['checkboxes'][0.01].set(True)

        # Set custom entry
        control['custom_entry'].set('0.5, 2.0')

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'learning_rate', is_float=True
        )

        # Verify
        self.assertEqual(values, [0.01, 0.5, 2.0])

    def test_extract_string_values(self):
        """Test extraction with string values."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='activation',
            param_label='Activation Function',
            checkbox_values=['relu', 'tanh', 'sigmoid'],
            allow_string_values=True
        )

        # Check some boxes
        control['checkboxes']['relu'].set(True)
        control['checkboxes']['tanh'].set(True)

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'activation', allow_string_values=True
        )

        # Verify
        self.assertIn('relu', values)
        self.assertIn('tanh', values)

    def test_extract_error_no_values(self):
        """Test that error is raised when no values are specified."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500]
        )

        # Don't check any boxes or enter custom values

        # Extract values - should raise error
        with self.assertRaises(ValueError) as cm:
            self.app._extract_parameter_values(
                control, 'n_estimators', is_float=False
            )

        self.assertIn('No values specified', str(cm.exception))

    def test_extract_with_none(self):
        """Test extraction with None value."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='max_depth',
            param_label='Max Depth',
            checkbox_values=[None, 5, 10, 20]
        )

        # Check None and some numeric values
        control['checkboxes'][None].set(True)
        control['checkboxes'][10].set(True)

        # Extract values
        values = self.app._extract_parameter_values(
            control, 'max_depth', is_float=False
        )

        # Verify - None should come first
        self.assertEqual(values[0], None)
        self.assertIn(10, values)


class TestIntegration(unittest.TestCase):
    """Integration tests for widget creation and value extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = MockApp()
        self.parent = tk.Frame(self.app.root, bg=self.app.colors['card_bg'])

    def tearDown(self):
        """Clean up after tests."""
        self.app.cleanup()

    def test_full_workflow_integer(self):
        """Test full workflow with integer parameters."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='n_estimators',
            param_label='Number of Trees',
            checkbox_values=[50, 100, 200, 500],
            default_checked=[100, 200]
        )

        # Add custom values
        control['custom_entry'].set('10, 300-500 step 100')

        # Extract
        values = self.app._extract_parameter_values(
            control, 'n_estimators', is_float=False
        )

        # Verify - should have: 10, 100, 200, 300, 400, 500
        expected = [10, 100, 200, 300, 400, 500]
        self.assertEqual(values, expected)

    def test_full_workflow_float(self):
        """Test full workflow with float parameters."""
        # Create control
        control = self.app._create_parameter_grid_control(
            self.parent,
            param_name='learning_rate',
            param_label='Learning Rate',
            checkbox_values=[0.001, 0.01, 0.1],
            default_checked=[0.01],
            is_float=True
        )

        # Add custom values
        control['custom_entry'].set('0.5, 1.0')

        # Extract
        values = self.app._extract_parameter_values(
            control, 'learning_rate', is_float=True
        )

        # Verify
        self.assertIn(0.01, values)
        self.assertIn(0.5, values)
        self.assertIn(1.0, values)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestParameterWidgetCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterValueExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
