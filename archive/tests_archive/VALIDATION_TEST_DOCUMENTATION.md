# Validation Utilities Test Documentation

## Overview

This document describes the validation utility functions added to `spectral_predict_gui_optimized.py` and their corresponding tests in `test_validation.py`.

## Functions Added to spectral_predict_gui_optimized.py

### 1. `_estimate_grid_size(self, param_dict)` (Lines 7740-7776)

**Purpose**: Calculate the total number of configurations in a hyperparameter grid search.

**Algorithm**:
- Multiplies the lengths of all parameter value lists
- Handles empty lists as size 1 (default value will be used)
- Handles None values as size 1
- Recursively handles nested dictionaries (for two-stage models like PLS-DA)

**Examples**:
```python
# Simple case
{'n_estimators': [100, 200], 'max_depth': [10, 20, 30]}
# Result: 2 * 3 = 6 configurations

# Nested case (PLS-DA)
{
    'pls_params': {'n_components': [2, 3, 4]},
    'clf_params': {'n_estimators': [100, 200], 'max_depth': [10, 20]}
}
# Result: 3 * 2 * 2 = 12 configurations

# With None and empty lists
{'n_estimators': [100, 200], 'max_depth': None, 'criterion': []}
# Result: 2 * 1 * 1 = 2 configurations
```

**Edge Cases Handled**:
- Empty dictionary → returns 1
- None input → returns 1
- Empty lists → counted as 1
- None values → counted as 1
- Single non-list values → counted as 1
- Nested dictionaries → recursive calculation

---

### 2. `_validate_parameter_bounds(self, param_name, values, min_val=None, max_val=None, allowed_types=None)` (Lines 7778-7837)

**Purpose**: Validate that parameter values are within acceptable bounds and types.

**Returns**: `(is_valid: bool, error_message: str or None)`

**Validation Checks**:
1. **Type validation**: Checks if values match allowed types
2. **Range validation**: Checks min/max bounds for numeric values
3. **None handling**: None values skip most validations (often allowed as defaults)

**Examples**:
```python
# Valid case
_validate_parameter_bounds('n_estimators', [50, 100, 200],
                          min_val=1, max_val=10000)
# Returns: (True, None)

# Below minimum
_validate_parameter_bounds('n_estimators', [0, -5], min_val=1)
# Returns: (False, "n_estimators values must be >= 1. Got 0.")

# Type mismatch
_validate_parameter_bounds('n_estimators', [100, 'invalid'],
                          allowed_types=[int])
# Returns: (False, "n_estimators must be of type(s): int. Got str.")

# None allowed with type checking
_validate_parameter_bounds('max_depth', [None, 10, 20],
                          allowed_types=[int], min_val=1)
# Returns: (True, None)  # None values are skipped
```

**Edge Cases Handled**:
- Non-list inputs → converted to list
- Empty lists → returns (True, None)
- None values → skipped in validation
- Non-numeric types → range checking skipped
- Multiple allowed types → validated against any

---

### 3. `_check_grid_size_warning(self, grid_size, threshold=1000)` (Lines 7839-7873)

**Purpose**: Generate appropriate warning messages based on grid search size.

**Returns**: `(show_warning: bool, warning_message: str or None)`

**Warning Levels**:

| Grid Size | Level | Message Content |
|-----------|-------|-----------------|
| < 100 | None | No warning |
| 100-999 | Info | "Should complete in a reasonable time" |
| 1000-9999 | Warning | "May take considerable time, consider reducing" |
| ≥ 10000 | Strong Warning | "Will take very long time, strongly recommended to reduce" |

**Examples**:
```python
# Small grid - no warning
_check_grid_size_warning(50)
# Returns: (False, None)

# Medium grid - info
_check_grid_size_warning(500)
# Returns: (True, "Grid search will evaluate 500 configurations.\n
#                  This should complete in a reasonable time.")

# Large grid - warning
_check_grid_size_warning(5000)
# Returns: (True, "Grid search will evaluate 5000 configurations.\n
#                  This may take a considerable amount of time.\n
#                  Consider reducing the parameter grid size.")

# Huge grid - strong warning
_check_grid_size_warning(15000)
# Returns: (True, "WARNING: Grid search will evaluate 15000 configurations!\n
#                  This will likely take a very long time to complete.\n
#                  It is strongly recommended to reduce the parameter grid size.\n
#                  Consider using random search or Bayesian optimization instead.")
```

---

## Test Coverage

### Test File: `tests/test_validation.py`

The test file includes 4 test classes with comprehensive coverage:

#### 1. `TestEstimateGridSize` (11 test cases)
- Empty dictionary
- None input
- Single parameter with single/multiple values
- Multiple parameters
- Empty lists
- None values
- Nested dictionaries
- Large grids
- Mixed types

#### 2. `TestValidateParameterBounds` (11 test cases)
- Valid ranges
- Below minimum
- Above maximum
- Type validation (int, float, string)
- Type validation failures
- None values allowed
- Empty lists
- Single values (not in list)
- Multiple allowed types
- String values

#### 3. `TestCheckGridSizeWarning` (8 test cases)
- Small grids (< 100)
- Medium grids (100-999)
- Large grids (1000-9999)
- Huge grids (≥ 10000)
- Boundary conditions (100, 1000, 10000)
- Custom thresholds

#### 4. `TestValidationEdgeCases` (4 integration tests)
- Real-world Random Forest grid
- Real-world XGBoost grid
- Real-world PLS-DA grid
- Dangerously large grid (10000 configurations)

---

## Common Parameter Validation Rules

The test file includes a comprehensive `COMMON_PARAMETER_RULES` dictionary with validation rules for standard ML parameters:

| Parameter | Min | Max | Types | Description |
|-----------|-----|-----|-------|-------------|
| `n_estimators` | 1 | 10000 | int | Number of trees/estimators |
| `max_depth` | 1 | 100 | int, None | Maximum tree depth |
| `learning_rate` | 0.0 | 1.0 | float | Learning rate for boosting |
| `n_components` | 1 | 100 | int | Number of PLS/PCA components |
| `C` | 0.0 | 1000.0 | float, int | SVM/Logistic regularization |
| `alpha` | 0.0 | 1000.0 | float, int | Ridge/Lasso regularization |
| `min_samples_split` | 2 | 1000 | int | Min samples to split node |
| `min_samples_leaf` | 1 | 1000 | int | Min samples in leaf |
| `subsample` | 0.0 | 1.0 | float | Subsample ratio |
| `colsample_bytree` | 0.0 | 1.0 | float | Column subsample ratio |
| `max_iter` | 1 | 100000 | int | Maximum iterations |
| `random_state` | 0 | - | int, None | Random seed |

---

## Usage Examples

### Example 1: Validate and Estimate a Grid Search

```python
# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Estimate grid size
grid_size = self._estimate_grid_size(param_grid)
# Result: 3 * 3 * 3 = 27 configurations

# Check for warnings
show_warning, message = self._check_grid_size_warning(grid_size)
# Result: (False, None) - no warning for 27 configurations

# Validate individual parameters
is_valid, error = self._validate_parameter_bounds(
    'learning_rate', param_grid['learning_rate'],
    min_val=0.0, max_val=1.0, allowed_types=[float]
)
# Result: (True, None) - all values valid
```

### Example 2: Catch Invalid Parameters

```python
# Invalid parameter grid
param_grid = {
    'n_estimators': [0, 100, 200],  # 0 is invalid
    'max_depth': [10, 20, 30]
}

# Validate n_estimators
is_valid, error = self._validate_parameter_bounds(
    'n_estimators', param_grid['n_estimators'],
    min_val=1, max_val=10000
)
# Result: (False, "n_estimators values must be >= 1. Got 0.")

# Show error to user
if not is_valid:
    messagebox.showerror("Parameter Error", error)
```

### Example 3: Warn About Large Grid

```python
# Large parameter grid
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [5, 10, 15, 20, 25],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Estimate grid size
grid_size = self._estimate_grid_size(param_grid)
# Result: 4 * 5 * 4 * 3 * 3 = 720 configurations

# Check for warnings
show_warning, message = self._check_grid_size_warning(grid_size)
# Result: (True, "Grid search will evaluate 720 configurations.\n
#                This should complete in a reasonable time.")

# Show info to user
if show_warning:
    messagebox.showinfo("Grid Size", message)
```

### Example 4: PLS-DA Two-Stage Model

```python
# PLS-DA nested parameter structure
param_grid = {
    'pls_params': {
        'n_components': [2, 3, 4, 5, 6]
    },
    'clf_params': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf', 'poly']
    }
}

# Estimate grid size (handles nested structure)
grid_size = self._estimate_grid_size(param_grid)
# Result: 5 * 4 * 3 = 60 configurations

# Validate PLS components
is_valid, error = self._validate_parameter_bounds(
    'n_components', param_grid['pls_params']['n_components'],
    min_val=1, max_val=100, allowed_types=[int]
)
# Result: (True, None)
```

---

## Running the Tests

To run the validation tests:

```bash
# Run all validation tests
pytest tests/test_validation.py -v

# Run specific test class
pytest tests/test_validation.py::TestEstimateGridSize -v

# Run specific test
pytest tests/test_validation.py::TestEstimateGridSize::test_multiple_parameters -v

# Run with coverage
pytest tests/test_validation.py --cov=spectral_predict_gui_optimized --cov-report=html
```

Expected output:
```
tests/test_validation.py::TestEstimateGridSize::test_empty_dict PASSED
tests/test_validation.py::TestEstimateGridSize::test_none_input PASSED
tests/test_validation.py::TestEstimateGridSize::test_single_parameter_single_value PASSED
tests/test_validation.py::TestEstimateGridSize::test_single_parameter_multiple_values PASSED
tests/test_validation.py::TestEstimateGridSize::test_multiple_parameters PASSED
... (30+ tests total)
```

---

## Integration Points

These validation functions should be integrated at the following points in the GUI:

1. **Before grid search execution**:
   - Estimate grid size
   - Show warning if > 100 configurations
   - Get user confirmation for large grids

2. **When user edits parameters**:
   - Validate bounds in real-time
   - Show error messages for invalid values
   - Highlight invalid fields

3. **On parameter grid submission**:
   - Final validation pass
   - Block execution if any invalid parameters
   - Show comprehensive error report

4. **In help/tooltips**:
   - Display min/max bounds from COMMON_PARAMETER_RULES
   - Show allowed types
   - Provide parameter descriptions

---

## Future Enhancements

Potential improvements to the validation system:

1. **Parameter dependencies**: Validate relationships between parameters (e.g., min_samples_leaf < min_samples_split)
2. **Smart defaults**: Suggest reasonable parameter ranges based on dataset size
3. **Performance estimation**: Estimate time to completion based on grid size and dataset
4. **Auto-reduction**: Automatically reduce grid size by removing redundant combinations
5. **Validation profiles**: Pre-configured validation rules for different model types
6. **Interactive warnings**: Allow user to continue despite warnings with explicit confirmation

---

## File Locations

- **Functions**: `C:\Users\sponheim\git\dasp\spectral_predict_gui_optimized.py` (lines 7740-7873)
- **Tests**: `C:\Users\sponheim\git\dasp\tests\test_validation.py`
- **Documentation**: `C:\Users\sponheim\git\dasp\tests\VALIDATION_TEST_DOCUMENTATION.md`
