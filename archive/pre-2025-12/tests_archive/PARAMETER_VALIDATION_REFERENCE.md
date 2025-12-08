# Parameter Validation Reference Guide

Quick reference for using the validation utilities in Model Development (Tab 7).

## Quick Start

```python
# 1. Estimate grid size
grid_size = self._estimate_grid_size(param_dict)

# 2. Check if warning needed
show_warning, message = self._check_grid_size_warning(grid_size)

# 3. Validate individual parameters
is_valid, error = self._validate_parameter_bounds(
    param_name, values, min_val=X, max_val=Y, allowed_types=[type1, type2]
)
```

---

## Validation Rules by Model Type

### Random Forest / Extra Trees

```python
VALIDATION_RULES = {
    'n_estimators': {
        'min_val': 1,
        'max_val': 10000,
        'allowed_types': [int]
    },
    'max_depth': {
        'min_val': 1,
        'max_val': 100,
        'allowed_types': [int, type(None)]  # None = unlimited
    },
    'min_samples_split': {
        'min_val': 2,
        'max_val': 1000,
        'allowed_types': [int]
    },
    'min_samples_leaf': {
        'min_val': 1,
        'max_val': 1000,
        'allowed_types': [int]
    },
    'max_features': {
        'allowed_types': [int, float, str, type(None)],
        # 'auto', 'sqrt', 'log2', None, or int/float
    }
}
```

### XGBoost / LightGBM / CatBoost

```python
VALIDATION_RULES = {
    'n_estimators': {
        'min_val': 1,
        'max_val': 10000,
        'allowed_types': [int]
    },
    'max_depth': {
        'min_val': 1,
        'max_val': 100,
        'allowed_types': [int]
    },
    'learning_rate': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float]
    },
    'subsample': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float]
    },
    'colsample_bytree': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float]
    },
    'reg_alpha': {
        'min_val': 0.0,
        'max_val': 1000.0,
        'allowed_types': [float, int]
    },
    'reg_lambda': {
        'min_val': 0.0,
        'max_val': 1000.0,
        'allowed_types': [float, int]
    }
}
```

### PLS / PCA

```python
VALIDATION_RULES = {
    'n_components': {
        'min_val': 1,
        'max_val': 100,  # Should not exceed n_features
        'allowed_types': [int]
    },
    'scale': {
        'allowed_types': [bool]
    }
}
```

### Ridge / Lasso / ElasticNet

```python
VALIDATION_RULES = {
    'alpha': {
        'min_val': 0.0,
        'max_val': 1000.0,
        'allowed_types': [float, int]
    },
    'l1_ratio': {  # ElasticNet only
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float]
    },
    'max_iter': {
        'min_val': 1,
        'max_val': 100000,
        'allowed_types': [int]
    }
}
```

### SVM / SVR

```python
VALIDATION_RULES = {
    'C': {
        'min_val': 0.0,
        'max_val': 1000.0,
        'allowed_types': [float, int]
    },
    'epsilon': {  # SVR only
        'min_val': 0.0,
        'max_val': 10.0,
        'allowed_types': [float]
    },
    'gamma': {
        'min_val': 0.0,
        'max_val': 100.0,
        'allowed_types': [float, str]  # 'scale', 'auto', or float
    },
    'kernel': {
        'allowed_types': [str],
        # 'linear', 'poly', 'rbf', 'sigmoid'
    }
}
```

### MLP (Neural Network)

```python
VALIDATION_RULES = {
    'hidden_layer_sizes': {
        'allowed_types': [tuple, list]
        # e.g., (100,), (50, 50), (100, 50, 25)
    },
    'alpha': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float]
    },
    'learning_rate_init': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float]
    },
    'max_iter': {
        'min_val': 1,
        'max_val': 100000,
        'allowed_types': [int]
    }
}
```

### NeuralBoosted

```python
VALIDATION_RULES = {
    'n_estimators': {
        'min_val': 1,
        'max_val': 10000,
        'allowed_types': [int]
    },
    'learning_rate': {
        'min_val': 0.0,
        'max_val': 1.0,
        'allowed_types': [float]
    },
    'hidden_layer_sizes': {
        'allowed_types': [tuple, list]
    },
    'max_iter': {
        'min_val': 1,
        'max_val': 10000,
        'allowed_types': [int]
    }
}
```

---

## Grid Size Guidelines

### Recommended Grid Sizes by Use Case

| Use Case | Grid Size | Example |
|----------|-----------|---------|
| Quick exploration | < 20 | 2-3 values per parameter, 2-3 parameters |
| Standard search | 20-100 | 3-4 values per parameter, 3-4 parameters |
| Thorough search | 100-500 | 4-5 values per parameter, 3-4 parameters |
| Exhaustive search | 500-1000 | 5+ values per parameter, 4+ parameters |
| Too large | > 1000 | Consider random search or Bayesian optimization |

### Grid Size Examples

#### Small Grid (24 configurations)
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.3, 0.5]
}
# 2 * 3 * 4 = 24
```

#### Medium Grid (108 configurations)
```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# 3 * 3 * 3 * 4 = 108
```

#### Large Grid (720 configurations)
```python
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [5, 10, 15, 20, 25],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}
# 4 * 5 * 4 * 3 = 240
```

---

## Common Validation Patterns

### Pattern 1: Validate Before Grid Search

```python
def validate_and_run_grid_search(self, param_grid):
    """Validate parameters and estimate grid size before running."""

    # Step 1: Validate each parameter
    for param_name, values in param_grid.items():
        if param_name in VALIDATION_RULES:
            rules = VALIDATION_RULES[param_name]
            is_valid, error = self._validate_parameter_bounds(
                param_name, values, **rules
            )
            if not is_valid:
                messagebox.showerror("Parameter Error", error)
                return False

    # Step 2: Estimate grid size
    grid_size = self._estimate_grid_size(param_grid)

    # Step 3: Check for warnings
    show_warning, message = self._check_grid_size_warning(grid_size)
    if show_warning:
        # Show warning and ask for confirmation
        result = messagebox.askyesno("Grid Size Warning",
                                     f"{message}\n\nContinue anyway?")
        if not result:
            return False

    # Step 4: Run grid search
    self.run_grid_search(param_grid)
    return True
```

### Pattern 2: Real-time Validation on Input

```python
def on_parameter_changed(self, param_name, value_str):
    """Validate parameter in real-time as user types."""

    try:
        # Parse value
        if param_name in ['n_estimators', 'max_depth', 'min_samples_split']:
            value = int(value_str) if value_str != 'None' else None
        else:
            value = float(value_str) if value_str != 'None' else None

        # Validate
        if param_name in VALIDATION_RULES:
            rules = VALIDATION_RULES[param_name]
            is_valid, error = self._validate_parameter_bounds(
                param_name, [value], **rules
            )

            if is_valid:
                # Clear error highlighting
                self.param_entries[param_name].config(bg='white')
            else:
                # Highlight error
                self.param_entries[param_name].config(bg='#ffcccc')
                self.status_label.config(text=error)

    except ValueError:
        # Invalid format
        self.param_entries[param_name].config(bg='#ffcccc')
        self.status_label.config(text=f"Invalid format for {param_name}")
```

### Pattern 3: Smart Parameter Suggestions

```python
def suggest_parameter_grid(self, model_type, dataset_size):
    """Suggest reasonable parameter grid based on model and dataset."""

    # Base grids for different models
    if model_type == 'RandomForest':
        base_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'XGBoost':
        base_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

    # Adjust based on dataset size
    grid_size = self._estimate_grid_size(base_grid)

    if dataset_size < 100:
        # Small dataset - reduce grid
        for param in base_grid:
            if len(base_grid[param]) > 2:
                base_grid[param] = base_grid[param][:2]

    return base_grid
```

---

## Integration Checklist

When integrating these validation functions:

- [ ] Import validation rules at top of file
- [ ] Add validation before grid search execution
- [ ] Add real-time validation on parameter input
- [ ] Show grid size estimate before running
- [ ] Display warnings for large grids
- [ ] Block execution on invalid parameters
- [ ] Log validation errors for debugging
- [ ] Add tooltips showing min/max bounds
- [ ] Include parameter descriptions in help
- [ ] Test with edge cases (None, empty, nested)

---

## Performance Considerations

### Grid Size vs. Execution Time (Rough Estimates)

Assuming 5-fold CV on 1000 samples:

| Grid Size | Estimated Time | Recommendation |
|-----------|----------------|----------------|
| < 10 | < 1 minute | Optimal for quick testing |
| 10-50 | 1-5 minutes | Good for exploration |
| 50-100 | 5-15 minutes | Reasonable for thorough search |
| 100-500 | 15-60 minutes | Consider parallel processing |
| 500-1000 | 1-3 hours | Use only if necessary |
| > 1000 | > 3 hours | Use random search or Bayesian optimization |

Note: Times vary greatly based on model complexity, dataset size, and hardware.

---

## Error Messages Reference

### Type Errors
```
"{param_name} must be of type(s): {type_names}. Got {actual_type}."
```

### Range Errors
```
"{param_name} values must be >= {min_val}. Got {value}."
"{param_name} values must be <= {max_val}. Got {value}."
```

### Grid Size Info
```
"Grid search will evaluate {size} configurations.
This should complete in a reasonable time."
```

### Grid Size Warning
```
"Grid search will evaluate {size} configurations.
This may take a considerable amount of time.
Consider reducing the parameter grid size."
```

### Grid Size Strong Warning
```
"WARNING: Grid search will evaluate {size} configurations!
This will likely take a very long time to complete.
It is strongly recommended to reduce the parameter grid size.
Consider using random search or Bayesian optimization instead."
```
