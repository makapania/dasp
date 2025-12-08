"""
Tests for parameter validation utilities in spectral_predict_gui_optimized.py

Tests cover:
- Grid size estimation with various parameter combinations
- Parameter bound validation
- Warning threshold logic
- Edge cases (empty lists, None values, nested structures)
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import the GUI module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the app class - we'll create a minimal instance for testing
import tkinter as tk
from spectral_predict_gui_optimized import SpectralPredictApp


class TestEstimateGridSize:
    """Test grid size estimation functionality."""

    @pytest.fixture
    def app(self):
        """Create a minimal app instance for testing."""
        root = tk.Tk()
        root.withdraw()  # Hide the window
        app = SpectralPredictApp(root)
        yield app
        root.destroy()

    def test_empty_dict(self, app):
        """Test with empty parameter dictionary."""
        result = app._estimate_grid_size({})
        assert result == 1

    def test_none_input(self, app):
        """Test with None input."""
        result = app._estimate_grid_size(None)
        assert result == 1

    def test_single_parameter_single_value(self, app):
        """Test with single parameter having one value."""
        param_dict = {'n_estimators': [100]}
        result = app._estimate_grid_size(param_dict)
        assert result == 1

    def test_single_parameter_multiple_values(self, app):
        """Test with single parameter having multiple values."""
        param_dict = {'n_estimators': [100, 200, 300]}
        result = app._estimate_grid_size(param_dict)
        assert result == 3

    def test_multiple_parameters(self, app):
        """Test with multiple parameters."""
        param_dict = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, 30],
            'learning_rate': [0.01, 0.1]
        }
        result = app._estimate_grid_size(param_dict)
        assert result == 2 * 3 * 2  # = 12

    def test_empty_list_counts_as_one(self, app):
        """Test that empty lists count as size 1."""
        param_dict = {
            'n_estimators': [],
            'max_depth': [10, 20]
        }
        result = app._estimate_grid_size(param_dict)
        assert result == 1 * 2  # = 2

    def test_none_value_counts_as_one(self, app):
        """Test that None values count as size 1."""
        param_dict = {
            'n_estimators': [100, 200],
            'max_depth': None
        }
        result = app._estimate_grid_size(param_dict)
        assert result == 2 * 1  # = 2

    def test_nested_dict_structure(self, app):
        """Test with nested parameter structure (for two-stage models)."""
        param_dict = {
            'pls_params': {
                'n_components': [2, 3, 4]
            },
            'clf_params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20]
            }
        }
        result = app._estimate_grid_size(param_dict)
        assert result == 3 * 2 * 2  # = 12

    def test_large_grid(self, app):
        """Test with a large parameter grid."""
        param_dict = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [3, 5, 10, 15, 20],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10]
        }
        result = app._estimate_grid_size(param_dict)
        assert result == 4 * 5 * 4 * 3  # = 240

    def test_mixed_types(self, app):
        """Test with mixed parameter types."""
        param_dict = {
            'n_estimators': [100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'random_state': 42  # Single value, not a list
        }
        result = app._estimate_grid_size(param_dict)
        assert result == 2 * 2 * 3 * 1  # = 12


class TestValidateParameterBounds:
    """Test parameter bounds validation functionality."""

    @pytest.fixture
    def app(self):
        """Create a minimal app instance for testing."""
        root = tk.Tk()
        root.withdraw()
        app = SpectralPredictApp(root)
        yield app
        root.destroy()

    def test_valid_range(self, app):
        """Test with values within valid range."""
        is_valid, error = app._validate_parameter_bounds(
            'n_estimators', [50, 100, 200],
            min_val=1, max_val=10000
        )
        assert is_valid is True
        assert error is None

    def test_below_minimum(self, app):
        """Test with values below minimum."""
        is_valid, error = app._validate_parameter_bounds(
            'n_estimators', [0, -5],
            min_val=1
        )
        assert is_valid is False
        assert 'must be >= 1' in error

    def test_above_maximum(self, app):
        """Test with values above maximum."""
        is_valid, error = app._validate_parameter_bounds(
            'learning_rate', [0.5, 1.5],
            max_val=1.0
        )
        assert is_valid is False
        assert 'must be <= 1.0' in error

    def test_type_validation_int(self, app):
        """Test type validation for integers."""
        is_valid, error = app._validate_parameter_bounds(
            'n_estimators', [100, 200],
            allowed_types=[int]
        )
        assert is_valid is True
        assert error is None

    def test_type_validation_float(self, app):
        """Test type validation for floats."""
        is_valid, error = app._validate_parameter_bounds(
            'learning_rate', [0.01, 0.1],
            allowed_types=[float]
        )
        assert is_valid is True
        assert error is None

    def test_type_validation_fail(self, app):
        """Test type validation failure."""
        is_valid, error = app._validate_parameter_bounds(
            'n_estimators', [100, 'invalid'],
            allowed_types=[int]
        )
        assert is_valid is False
        assert 'must be of type' in error

    def test_none_value_allowed(self, app):
        """Test that None values are allowed with type checking."""
        is_valid, error = app._validate_parameter_bounds(
            'max_depth', [None, 10, 20],
            allowed_types=[int],
            min_val=1
        )
        assert is_valid is True
        assert error is None

    def test_empty_list(self, app):
        """Test with empty list."""
        is_valid, error = app._validate_parameter_bounds(
            'n_estimators', [],
            min_val=1
        )
        assert is_valid is True
        assert error is None

    def test_single_value_not_list(self, app):
        """Test with single value instead of list."""
        is_valid, error = app._validate_parameter_bounds(
            'n_estimators', 100,
            min_val=1, max_val=10000
        )
        assert is_valid is True
        assert error is None

    def test_multiple_types_allowed(self, app):
        """Test with multiple allowed types."""
        is_valid, error = app._validate_parameter_bounds(
            'max_depth', [None, 10, 20],
            allowed_types=[int, type(None)]
        )
        assert is_valid is True
        assert error is None

    def test_string_values(self, app):
        """Test with string values (no range checking)."""
        is_valid, error = app._validate_parameter_bounds(
            'criterion', ['gini', 'entropy'],
            allowed_types=[str]
        )
        assert is_valid is True
        assert error is None


class TestCheckGridSizeWarning:
    """Test grid size warning system."""

    @pytest.fixture
    def app(self):
        """Create a minimal app instance for testing."""
        root = tk.Tk()
        root.withdraw()
        app = SpectralPredictApp(root)
        yield app
        root.destroy()

    def test_no_warning_small_grid(self, app):
        """Test with small grid (< 100) - no warning."""
        show_warning, message = app._check_grid_size_warning(50)
        assert show_warning is False
        assert message is None

    def test_info_message_medium_grid(self, app):
        """Test with medium grid (100-999) - info message."""
        show_warning, message = app._check_grid_size_warning(500)
        assert show_warning is True
        assert '500 configurations' in message
        assert 'reasonable time' in message

    def test_warning_large_grid(self, app):
        """Test with large grid (1000-9999) - warning message."""
        show_warning, message = app._check_grid_size_warning(5000)
        assert show_warning is True
        assert '5000 configurations' in message
        assert 'considerable amount of time' in message
        assert 'Consider reducing' in message

    def test_strong_warning_huge_grid(self, app):
        """Test with huge grid (>= 10000) - strong warning."""
        show_warning, message = app._check_grid_size_warning(15000)
        assert show_warning is True
        assert '15000 configurations' in message
        assert 'WARNING' in message
        assert 'very long time' in message
        assert 'strongly recommended' in message

    def test_boundary_100(self, app):
        """Test boundary at 100."""
        show_warning, message = app._check_grid_size_warning(100)
        assert show_warning is True
        assert 'reasonable time' in message

    def test_boundary_1000(self, app):
        """Test boundary at 1000."""
        show_warning, message = app._check_grid_size_warning(1000)
        assert show_warning is True
        assert 'considerable amount of time' in message

    def test_boundary_10000(self, app):
        """Test boundary at 10000."""
        show_warning, message = app._check_grid_size_warning(10000)
        assert show_warning is True
        assert 'WARNING' in message

    def test_custom_threshold(self, app):
        """Test with custom threshold."""
        show_warning, message = app._check_grid_size_warning(500, threshold=500)
        assert show_warning is True


class TestValidationEdgeCases:
    """Test edge cases and integration scenarios."""

    @pytest.fixture
    def app(self):
        """Create a minimal app instance for testing."""
        root = tk.Tk()
        root.withdraw()
        app = SpectralPredictApp(root)
        yield app
        root.destroy()

    def test_real_world_scenario_rf(self, app):
        """Test realistic Random Forest parameter grid."""
        param_dict = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_size = app._estimate_grid_size(param_dict)
        assert grid_size == 3 * 4 * 3 * 3  # = 108

        show_warning, message = app._check_grid_size_warning(grid_size)
        assert show_warning is True
        assert 'reasonable time' in message

    def test_real_world_scenario_xgboost(self, app):
        """Test realistic XGBoost parameter grid."""
        param_dict = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid_size = app._estimate_grid_size(param_dict)
        assert grid_size == 2 * 3 * 3 * 2 * 2  # = 72

        # Validate learning_rate bounds
        is_valid, error = app._validate_parameter_bounds(
            'learning_rate', param_dict['learning_rate'],
            min_val=0.0, max_val=1.0
        )
        assert is_valid is True

    def test_real_world_scenario_pls_da(self, app):
        """Test realistic PLS-DA (two-stage) parameter grid."""
        param_dict = {
            'pls_params': {
                'n_components': [2, 3, 4, 5]
            },
            'clf_params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        }
        grid_size = app._estimate_grid_size(param_dict)
        assert grid_size == 4 * 3 * 2  # = 24

    def test_dangerous_grid_size(self, app):
        """Test with dangerously large grid."""
        param_dict = {
            'param1': list(range(10)),
            'param2': list(range(10)),
            'param3': list(range(10)),
            'param4': list(range(10))
        }
        grid_size = app._estimate_grid_size(param_dict)
        assert grid_size == 10000

        show_warning, message = app._check_grid_size_warning(grid_size)
        assert show_warning is True
        assert 'WARNING' in message
        assert 'strongly recommended' in message


# Example validation rules for common parameters
COMMON_PARAMETER_RULES = {
    'n_estimators': {
        'min_val': 1,
        'max_val': 10000,
        'allowed_types': [int],
        'description': 'Number of trees/estimators in ensemble'
    },
    'max_depth': {
        'min_val': 1,
        'max_val': 100,
        'allowed_types': [int, type(None)],
        'description': 'Maximum depth of trees (None for unlimited)'
    },
    'learning_rate': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float],
        'description': 'Learning rate for gradient boosting'
    },
    'n_components': {
        'min_val': 1,
        'max_val': 100,
        'allowed_types': [int],
        'description': 'Number of components for PLS/PCA'
    },
    'C': {
        'min_val': 0.0,
        'max_val': 1000.0,
        'allowed_types': [float, int],
        'description': 'Regularization parameter for SVM/Logistic'
    },
    'alpha': {
        'min_val': 0.0,
        'max_val': 1000.0,
        'allowed_types': [float, int],
        'description': 'Regularization strength for Ridge/Lasso'
    },
    'min_samples_split': {
        'min_val': 2,
        'max_val': 1000,
        'allowed_types': [int],
        'description': 'Minimum samples required to split a node'
    },
    'min_samples_leaf': {
        'min_val': 1,
        'max_val': 1000,
        'allowed_types': [int],
        'description': 'Minimum samples required in a leaf node'
    },
    'subsample': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float],
        'description': 'Subsample ratio for gradient boosting'
    },
    'colsample_bytree': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float],
        'description': 'Column subsample ratio for XGBoost'
    },
    'hidden_layer_sizes': {
        'allowed_types': [tuple, list],
        'description': 'Hidden layer architecture for MLP'
    },
    'max_iter': {
        'min_val': 1,
        'max_val': 100000,
        'allowed_types': [int],
        'description': 'Maximum number of iterations'
    },
    'random_state': {
        'min_val': 0,
        'allowed_types': [int, type(None)],
        'description': 'Random seed for reproducibility'
    }
}


def test_common_parameter_rules_access():
    """Test that common parameter rules are accessible."""
    assert 'n_estimators' in COMMON_PARAMETER_RULES
    assert 'learning_rate' in COMMON_PARAMETER_RULES
    assert 'max_depth' in COMMON_PARAMETER_RULES

    # Check structure
    n_est_rules = COMMON_PARAMETER_RULES['n_estimators']
    assert 'min_val' in n_est_rules
    assert 'max_val' in n_est_rules
    assert 'allowed_types' in n_est_rules
    assert 'description' in n_est_rules


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
