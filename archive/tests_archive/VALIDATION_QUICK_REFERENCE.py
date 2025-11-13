"""
Quick Reference: Validation Utilities Usage

Copy-paste examples for common validation scenarios.
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

# 1. Estimate grid size
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30]
}
grid_size = self._estimate_grid_size(param_grid)
# Result: 9 configurations

# 2. Check if warning needed
show_warning, message = self._check_grid_size_warning(grid_size)
if show_warning:
    messagebox.showinfo("Grid Size", message)

# 3. Validate parameter bounds
is_valid, error = self._validate_parameter_bounds(
    'n_estimators', [100, 200, 500],
    min_val=1, max_val=10000, allowed_types=[int]
)
if not is_valid:
    messagebox.showerror("Validation Error", error)


# ============================================================================
# COMPLETE VALIDATION WORKFLOW
# ============================================================================

def validate_and_run_grid_search(self, param_grid):
    """Complete validation before running grid search."""

    # Define validation rules
    validation_rules = {
        'n_estimators': {'min_val': 1, 'max_val': 10000, 'allowed_types': [int]},
        'max_depth': {'min_val': 1, 'max_val': 100, 'allowed_types': [int, type(None)]},
        'learning_rate': {'min_val': 0.0, 'max_val': 1.0, 'allowed_types': [float]},
        'min_samples_split': {'min_val': 2, 'max_val': 1000, 'allowed_types': [int]},
        'min_samples_leaf': {'min_val': 1, 'max_val': 1000, 'allowed_types': [int]},
    }

    # Step 1: Validate each parameter
    errors = []
    for param_name, values in param_grid.items():
        if param_name in validation_rules:
            rules = validation_rules[param_name]
            is_valid, error = self._validate_parameter_bounds(
                param_name, values, **rules
            )
            if not is_valid:
                errors.append(error)

    if errors:
        messagebox.showerror("Validation Errors", "\n".join(errors))
        return False

    # Step 2: Estimate grid size
    grid_size = self._estimate_grid_size(param_grid)

    # Step 3: Check for warnings
    show_warning, message = self._check_grid_size_warning(grid_size)
    if show_warning:
        result = messagebox.askyesno(
            "Grid Size Warning",
            f"{message}\n\nContinue anyway?"
        )
        if not result:
            return False

    # Step 4: Run grid search
    self.run_grid_search(param_grid)
    return True


# ============================================================================
# VALIDATION RULES BY MODEL TYPE
# ============================================================================

# Random Forest
RF_RULES = {
    'n_estimators': {'min_val': 1, 'max_val': 10000, 'allowed_types': [int]},
    'max_depth': {'min_val': 1, 'max_val': 100, 'allowed_types': [int, type(None)]},
    'min_samples_split': {'min_val': 2, 'max_val': 1000, 'allowed_types': [int]},
    'min_samples_leaf': {'min_val': 1, 'max_val': 1000, 'allowed_types': [int]},
}

# XGBoost / LightGBM / CatBoost
BOOSTING_RULES = {
    'n_estimators': {'min_val': 1, 'max_val': 10000, 'allowed_types': [int]},
    'max_depth': {'min_val': 1, 'max_val': 100, 'allowed_types': [int]},
    'learning_rate': {'min_val': 0.0, 'max_val': 1.0, 'allowed_types': [float]},
    'subsample': {'min_val': 0.0, 'max_val': 1.0, 'allowed_types': [float]},
    'colsample_bytree': {'min_val': 0.0, 'max_val': 1.0, 'allowed_types': [float]},
}

# PLS
PLS_RULES = {
    'n_components': {'min_val': 1, 'max_val': 100, 'allowed_types': [int]},
}

# Ridge / Lasso / ElasticNet
LINEAR_RULES = {
    'alpha': {'min_val': 0.0, 'max_val': 1000.0, 'allowed_types': [float, int]},
    'max_iter': {'min_val': 1, 'max_val': 100000, 'allowed_types': [int]},
}

# SVM / SVR
SVM_RULES = {
    'C': {'min_val': 0.0, 'max_val': 1000.0, 'allowed_types': [float, int]},
    'gamma': {'min_val': 0.0, 'max_val': 100.0, 'allowed_types': [float, str]},
}


# ============================================================================
# REAL-TIME VALIDATION ON INPUT
# ============================================================================

def on_parameter_entry_changed(self, param_name, entry_widget):
    """Validate parameter as user types."""

    value_str = entry_widget.get()

    try:
        # Parse value based on parameter type
        if param_name in ['n_estimators', 'max_depth', 'min_samples_split']:
            value = int(value_str) if value_str.lower() != 'none' else None
        elif param_name in ['learning_rate', 'alpha', 'subsample']:
            value = float(value_str) if value_str.lower() != 'none' else None
        else:
            value = value_str

        # Get validation rules
        rules = get_validation_rules(param_name)

        # Validate
        is_valid, error = self._validate_parameter_bounds(
            param_name, [value], **rules
        )

        if is_valid:
            # Clear error
            entry_widget.config(bg='white')
            self.status_label.config(text="Ready")
        else:
            # Show error
            entry_widget.config(bg='#ffcccc')
            self.status_label.config(text=error)

    except ValueError:
        # Invalid format
        entry_widget.config(bg='#ffcccc')
        self.status_label.config(text=f"Invalid format for {param_name}")


# ============================================================================
# GRID SIZE ESTIMATION EXAMPLES
# ============================================================================

# Example 1: Simple grid
simple_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30]
}
# 2 * 3 = 6 configurations

# Example 2: Nested grid (PLS-DA)
nested_grid = {
    'pls_params': {
        'n_components': [2, 3, 4]
    },
    'clf_params': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf']
    }
}
# 3 * 3 * 2 = 18 configurations

# Example 3: With None and empty lists
mixed_grid = {
    'n_estimators': [100, 200],
    'max_depth': None,           # Counts as 1
    'criterion': [],             # Counts as 1
    'min_samples_split': [2, 5]
}
# 2 * 1 * 1 * 2 = 4 configurations


# ============================================================================
# PARAMETER VALIDATION EXAMPLES
# ============================================================================

# Valid cases
self._validate_parameter_bounds('n_estimators', [100, 200], min_val=1, max_val=10000)
# Returns: (True, None)

self._validate_parameter_bounds('max_depth', [None, 10, 20], allowed_types=[int], min_val=1)
# Returns: (True, None) - None values are allowed

# Invalid cases
self._validate_parameter_bounds('n_estimators', [0, 100], min_val=1)
# Returns: (False, "n_estimators values must be >= 1. Got 0.")

self._validate_parameter_bounds('learning_rate', [1.5], max_val=1.0)
# Returns: (False, "learning_rate values must be <= 1.0. Got 1.5.")

self._validate_parameter_bounds('n_estimators', ['invalid'], allowed_types=[int])
# Returns: (False, "n_estimators must be of type(s): int. Got str.")


# ============================================================================
# WARNING SYSTEM EXAMPLES
# ============================================================================

# Small grid - no warning
self._check_grid_size_warning(50)
# Returns: (False, None)

# Medium grid - info
self._check_grid_size_warning(500)
# Returns: (True, "Grid search will evaluate 500 configurations.\nThis should complete in a reasonable time.")

# Large grid - warning
self._check_grid_size_warning(5000)
# Returns: (True, "Grid search will evaluate 5000 configurations.\nThis may take a considerable amount of time.\nConsider reducing the parameter grid size.")

# Huge grid - strong warning
self._check_grid_size_warning(15000)
# Returns: (True, "WARNING: Grid search will evaluate 15000 configurations!\nThis will likely take a very long time to complete.\nIt is strongly recommended to reduce the parameter grid size.\nConsider using random search or Bayesian optimization instead.")


# ============================================================================
# INTEGRATION WITH GUI BUTTON
# ============================================================================

def on_run_grid_search_clicked(self):
    """Handler for 'Run Grid Search' button."""

    # Get parameter grid from UI
    param_grid = self.get_parameter_grid_from_ui()

    # Validate and run
    success = self.validate_and_run_grid_search(param_grid)

    if success:
        self.status_label.config(text="Grid search running...")
    else:
        self.status_label.config(text="Validation failed")


# ============================================================================
# HELPER FUNCTION: GET VALIDATION RULES
# ============================================================================

def get_validation_rules(param_name):
    """Get validation rules for a parameter."""

    RULES = {
        'n_estimators': {'min_val': 1, 'max_val': 10000, 'allowed_types': [int]},
        'max_depth': {'min_val': 1, 'max_val': 100, 'allowed_types': [int, type(None)]},
        'learning_rate': {'min_val': 0.0, 'max_val': 1.0, 'allowed_types': [float]},
        'n_components': {'min_val': 1, 'max_val': 100, 'allowed_types': [int]},
        'C': {'min_val': 0.0, 'max_val': 1000.0, 'allowed_types': [float, int]},
        'alpha': {'min_val': 0.0, 'max_val': 1000.0, 'allowed_types': [float, int]},
        'min_samples_split': {'min_val': 2, 'max_val': 1000, 'allowed_types': [int]},
        'min_samples_leaf': {'min_val': 1, 'max_val': 1000, 'allowed_types': [int]},
        'subsample': {'min_val': 0.0, 'max_val': 1.0, 'allowed_types': [float]},
        'colsample_bytree': {'min_val': 0.0, 'max_val': 1.0, 'allowed_types': [float]},
        'max_iter': {'min_val': 1, 'max_val': 100000, 'allowed_types': [int]},
        'random_state': {'min_val': 0, 'allowed_types': [int, type(None)]},
    }

    return RULES.get(param_name, {})


# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_all_parameters(self, param_grid):
    """Validate all parameters at once."""

    errors = []

    for param_name, values in param_grid.items():
        rules = get_validation_rules(param_name)

        if rules:
            is_valid, error = self._validate_parameter_bounds(
                param_name, values, **rules
            )

            if not is_valid:
                errors.append(f"• {error}")

    if errors:
        return (False, "\n".join(errors))
    else:
        return (True, None)


# ============================================================================
# USAGE IN TAB 7 (MODEL DEVELOPMENT)
# ============================================================================

def run_custom_grid_search(self):
    """Run custom grid search from Tab 7."""

    # Get parameters from UI
    param_grid = {
        'n_estimators': self.parse_list_input(self.n_estimators_entry.get()),
        'max_depth': self.parse_list_input(self.max_depth_entry.get()),
        'learning_rate': self.parse_list_input(self.learning_rate_entry.get()),
    }

    # Remove empty parameters
    param_grid = {k: v for k, v in param_grid.items() if v}

    # Validate all parameters
    is_valid, error = self.validate_all_parameters(param_grid)
    if not is_valid:
        messagebox.showerror("Parameter Validation Failed", error)
        return

    # Estimate grid size
    grid_size = self._estimate_grid_size(param_grid)
    print(f"Grid size: {grid_size} configurations")

    # Check warnings
    show_warning, message = self._check_grid_size_warning(grid_size)
    if show_warning:
        response = messagebox.askyesno("Grid Size Warning", f"{message}\n\nContinue?")
        if not response:
            return

    # Run grid search
    self.execute_grid_search(param_grid)


# ============================================================================
# TOOLTIP INTEGRATION
# ============================================================================

def create_parameter_tooltip(self, widget, param_name):
    """Create tooltip showing parameter constraints."""

    rules = get_validation_rules(param_name)

    if not rules:
        return

    tooltip_text = f"{param_name}:\n"

    if 'min_val' in rules:
        tooltip_text += f"  Min: {rules['min_val']}\n"
    if 'max_val' in rules:
        tooltip_text += f"  Max: {rules['max_val']}\n"
    if 'allowed_types' in rules:
        types = ', '.join([t.__name__ for t in rules['allowed_types']])
        tooltip_text += f"  Types: {types}\n"

    # Create tooltip (using your tooltip library)
    ToolTip(widget, text=tooltip_text)
