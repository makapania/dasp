"""
Manual validation test - demonstrates the validation functions work correctly.

This can be run standalone to verify the validation utilities without requiring
the full GUI to be initialized.
"""


class MockApp:
    """Minimal mock of SpectralPredictApp for testing validation functions."""

    def _estimate_grid_size(self, param_dict):
        """Calculate total grid size from parameter dictionary."""
        if not param_dict:
            return 1

        total_size = 1

        for param_name, param_values in param_dict.items():
            # Handle nested parameter structures
            if isinstance(param_values, dict):
                nested_size = self._estimate_grid_size(param_values)
                total_size *= nested_size
            elif isinstance(param_values, list):
                if len(param_values) == 0:
                    list_size = 1
                else:
                    list_size = len(param_values)
                total_size *= list_size
            else:
                total_size *= 1

        return total_size

    def _validate_parameter_bounds(self, param_name, values, min_val=None,
                                   max_val=None, allowed_types=None):
        """Validate parameter values are within acceptable bounds."""
        # Handle non-list inputs
        if not isinstance(values, list):
            values = [values]

        # Skip validation for empty lists
        if len(values) == 0:
            return (True, None)

        # Type checking
        if allowed_types is not None:
            for value in values:
                if value is None:
                    continue

                value_type = type(value)
                if value_type not in allowed_types:
                    type_names = ', '.join([t.__name__ for t in allowed_types])
                    return (False, f"{param_name} must be of type(s): {type_names}. Got {value_type.__name__}.")

        # Range validation
        for value in values:
            if value is None:
                continue

            if not isinstance(value, (int, float)):
                continue

            if min_val is not None and value < min_val:
                return (False, f"{param_name} values must be >= {min_val}. Got {value}.")

            if max_val is not None and value > max_val:
                return (False, f"{param_name} values must be <= {max_val}. Got {value}.")

        return (True, None)

    def _check_grid_size_warning(self, grid_size, threshold=1000):
        """Check if grid size exceeds warning threshold."""
        if grid_size < 100:
            return (False, None)

        elif grid_size < 1000:
            message = (f"Grid search will evaluate {grid_size} configurations.\n"
                      f"This should complete in a reasonable time.")
            return (True, message)

        elif grid_size < 10000:
            message = (f"Grid search will evaluate {grid_size} configurations.\n"
                      f"This may take a considerable amount of time.\n"
                      f"Consider reducing the parameter grid size.")
            return (True, message)

        else:
            message = (f"WARNING: Grid search will evaluate {grid_size} configurations!\n"
                      f"This will likely take a very long time to complete.\n"
                      f"It is strongly recommended to reduce the parameter grid size.\n"
                      f"Consider using random search or Bayesian optimization instead.")
            return (True, message)


def run_tests():
    """Run manual tests of validation functions."""
    app = MockApp()

    print("="*80)
    print("VALIDATION UTILITIES MANUAL TEST")
    print("="*80)
    print()

    # Test 1: Grid Size Estimation
    print("TEST 1: Grid Size Estimation")
    print("-" * 40)

    test_grids = [
        ({}, 1),
        ({'n_estimators': [100, 200]}, 2),
        ({'n_estimators': [100, 200], 'max_depth': [10, 20, 30]}, 6),
        ({'n_estimators': [], 'max_depth': [10, 20]}, 2),
        ({'n_estimators': None, 'max_depth': [10, 20]}, 2),
        ({
            'pls_params': {'n_components': [2, 3, 4]},
            'clf_params': {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        }, 12),
    ]

    for i, (grid, expected) in enumerate(test_grids, 1):
        result = app._estimate_grid_size(grid)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status} Test 1.{i}: {grid} => {result} (expected {expected})")

    print()

    # Test 2: Parameter Bounds Validation
    print("TEST 2: Parameter Bounds Validation")
    print("-" * 40)

    test_validations = [
        ('n_estimators', [50, 100, 200], {'min_val': 1, 'max_val': 10000}, True),
        ('n_estimators', [0, -5], {'min_val': 1}, False),
        ('learning_rate', [0.5, 1.5], {'max_val': 1.0}, False),
        ('n_estimators', [100, 200], {'allowed_types': [int]}, True),
        ('n_estimators', [100, 'invalid'], {'allowed_types': [int]}, False),
        ('max_depth', [None, 10, 20], {'allowed_types': [int], 'min_val': 1}, True),
    ]

    for i, (param, values, kwargs, expected_valid) in enumerate(test_validations, 1):
        is_valid, error = app._validate_parameter_bounds(param, values, **kwargs)
        status = "[PASS]" if is_valid == expected_valid else "[FAIL]"
        print(f"{status} Test 2.{i}: {param}={values} valid={is_valid}")
        if error:
            print(f"         Error: {error}")

    print()

    # Test 3: Grid Size Warnings
    print("TEST 3: Grid Size Warnings")
    print("-" * 40)

    test_warnings = [
        (50, False),
        (150, True),
        (5000, True),
        (15000, True),
    ]

    for i, (size, expected_warning) in enumerate(test_warnings, 1):
        show_warning, message = app._check_grid_size_warning(size)
        status = "[PASS]" if show_warning == expected_warning else "[FAIL]"
        print(f"{status} Test 3.{i}: Grid size {size} => Warning: {show_warning}")
        if message:
            lines = message.split('\n')
            print(f"         Message: {lines[0]}")

    print()

    # Test 4: Real-world Examples
    print("TEST 4: Real-world Examples")
    print("-" * 40)

    # Example 1: Random Forest
    print("\nExample 1: Random Forest Grid")
    rf_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_size = app._estimate_grid_size(rf_grid)
    show_warning, message = app._check_grid_size_warning(grid_size)
    print(f"  Grid size: {grid_size} configurations")
    print(f"  Warning: {show_warning}")
    if message:
        print(f"  Message: {message.split(chr(10))[0]}")

    # Example 2: XGBoost with validation
    print("\nExample 2: XGBoost Grid with Validation")
    xgb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
    }
    grid_size = app._estimate_grid_size(xgb_grid)
    print(f"  Grid size: {grid_size} configurations")

    # Validate learning_rate
    is_valid, error = app._validate_parameter_bounds(
        'learning_rate', xgb_grid['learning_rate'],
        min_val=0.0, max_val=1.0
    )
    print(f"  Learning rate valid: {is_valid}")

    # Example 3: Dangerous grid
    print("\nExample 3: Dangerously Large Grid")
    danger_grid = {
        'param1': list(range(10)),
        'param2': list(range(10)),
        'param3': list(range(10)),
        'param4': list(range(10))
    }
    grid_size = app._estimate_grid_size(danger_grid)
    show_warning, message = app._check_grid_size_warning(grid_size)
    print(f"  Grid size: {grid_size} configurations")
    print(f"  Warning: {show_warning}")
    if message:
        lines = message.split('\n')
        for line in lines[:2]:
            print(f"  {line}")

    print()
    print("="*80)
    print("MANUAL TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    run_tests()
