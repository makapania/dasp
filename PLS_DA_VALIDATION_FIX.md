# PLS-DA Model Development Tab Fix

## Problem

When running a PLS-DA model from the Results tab, it works fine. However, when loading the same PLS-DA model into the Model Development tab and clicking "Run Refined Model", users encounter TWO issues:

1. **Validation Error**: "Invalid model type selected" error
2. **Categorical Label Error**: "could not convert string to float: 'Parafin'" error

## Root Causes

### Issue 1: Hardcoded Validation for Regression Only

The `_validate_refinement_parameters()` function in `spectral_predict_gui_optimized.py` (line 9883) was **hardcoded to validate against 'regression'** instead of using the actual task type from the Model Development tab.

### Issue 2: Missing Categorical Label Encoding

The `_run_refined_model_thread()` function was missing the label encoding step that converts categorical text labels (like 'Parafin', 'Clean', 'Contaminated') to numeric values before training. The main search in `search.py` has this logic (lines 127-143), but the Model Development tab was missing it.

## Solution

### Fix 1: Validation with Correct Task Type

### Before (Buggy Code)
```python
def _validate_refinement_parameters(self):
    """Validate all refinement parameters before execution."""
    errors = []

    # ... wavelength validation ...

    # Validate model type
    model_type = self.refine_model_type.get()
    if is_valid_model is not None:
        # Use registry validation
        if not is_valid_model(model_type, 'regression'):  # ❌ HARDCODED!
            errors.append(f"Invalid model type selected: '{model_type}'")
```

**Why this failed:**
- PLS-DA is a **classification** model
- The validation was checking if PLS-DA is valid for **regression** (it's not!)
- Result: Validation failed even though PLS-DA is perfectly valid for classification

## Solution

Changed the validation to use the **actual task type** from `self.refine_task_type.get()`:

### After (Fixed Code)
```python
def _validate_refinement_parameters(self):
    """Validate all refinement parameters before execution."""
    errors = []

    # ... wavelength validation ...

    # Validate model type
    model_type = self.refine_model_type.get()
    task_type = self.refine_task_type.get()  # ✅ Get actual task type
    if is_valid_model is not None:
        # Use registry validation with correct task type
        if not is_valid_model(model_type, task_type):  # ✅ Use correct task_type
            errors.append(f"Invalid model type selected: '{model_type}' for {task_type}")
    else:
        # Fallback validation
        valid_models_regression = ['PLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'MLP', 'NeuralBoosted', 'SVR', 'XGBoost', 'LightGBM', 'CatBoost']
        valid_models_classification = ['PLS-DA', 'RandomForest', 'MLP', 'NeuralBoosted', 'SVM', 'XGBoost', 'LightGBM', 'CatBoost']
        valid_models = valid_models_classification if task_type == 'classification' else valid_models_regression
        if model_type not in valid_models:
            errors.append(f"Invalid model type selected: '{model_type}' for {task_type}")
```

### Fix 2: Categorical Label Encoding

Added label encoding logic before cross-validation to match `search.py` behavior:

```python
# Handle categorical labels for classification (must happen BEFORE creating y_array)
# This matches the logic in search.py lines 127-143
local_label_encoder = None
if task_type == "classification":
    # Check if labels are non-numeric (text labels like "Clean", "Contaminated", etc.)
    if y_series.dtype == object or not np.issubdtype(y_series.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        local_label_encoder = LabelEncoder()
        y_original = y_series.copy()  # Keep original for logging
        y_series_encoded = local_label_encoder.fit_transform(y_series)
        # Convert back to Series to maintain index
        y_series = pd.Series(y_series_encoded, index=y_series.index)

        # Log the label mapping
        label_mapping = dict(zip(local_label_encoder.classes_,
                                local_label_encoder.transform(local_label_encoder.classes_)))
        print(f"\nCATEGORICAL LABEL ENCODING (Model Development)")
        print(f"Detected non-numeric classification labels.")
        print(f"Encoding mapping:")
        for label, code in sorted(label_mapping.items(), key=lambda x: x[1]):
            print(f"  '{label}' -> {code}")
```

**Why this works:**
- Text labels like 'Parafin', 'Clean', 'Contaminated' are converted to integers (0, 1, 2, ...)
- PLS and other models can now process numeric targets
- Label encoder is stored for later decoding predictions back to text

## Changes Made

**File:** `spectral_predict_gui_optimized.py`

### Change 1: Validation Function (lines 9870-9898)

1. Added: `task_type = self.refine_task_type.get()` to get the actual task type
2. Changed: `is_valid_model(model_type, 'regression')` → `is_valid_model(model_type, task_type)`
3. Improved: Error messages now show both model and task type
4. Enhanced: Fallback validation now correctly checks against task-specific model lists

### Change 2: Label Encoding (lines 11331-11354)

1. Added: Label encoding logic before preparing cross-validation
2. Detects: Non-numeric labels (dtype == object or not numeric)
3. Encodes: Categorical labels to integers using `LabelEncoder`
4. Logs: Mapping of original labels to encoded values
5. Stores: Label encoder in `self.refined_label_encoder` for later use

## How It Works

The Model Development tab has a task type selector (Regression/Classification):
- `self.refine_task_type` stores the current selection
- When loading a PLS-DA result, the task type is auto-detected and set to 'classification'
- The validation now checks if the model is valid for the **actual task type** being used

## Testing

Created `test_plsda_validation_fix.py` which verifies:
- ✅ PLS-DA validates correctly for classification tasks
- ✅ PLS-DA correctly fails validation for regression tasks
- ✅ PLS validates correctly for regression tasks
- ✅ Other models validate correctly for their respective task types

## Before & After Behavior

### Before (Bugs)
1. Run PLS-DA analysis with categorical labels (e.g., 'Parafin') in Results tab → ✅ Works
2. Double-click PLS-DA result to load in Model Development tab → ✅ Loads
3. Click "Run Refined Model" → ❌ **"Invalid model type selected: 'PLS-DA'"**
4. (If validation was bypassed) → ❌ **"could not convert string to float: 'Parafin'"**

### After (Fixed)
1. Run PLS-DA analysis with categorical labels in Results tab → ✅ Works
2. Double-click PLS-DA result to load in Model Development tab → ✅ Loads (task type set to 'classification')
3. Click "Run Refined Model" → ✅ **Validation passes!**
4. Training begins → ✅ **Labels encoded automatically!**
5. Model trains successfully → ✅ **Shows accuracy, precision, recall, F1 score!**

## Related Files

- `spectral_predict_gui_optimized.py:9870-9898` - Validation fix
- `spectral_predict_gui_optimized.py:11331-11354` - Label encoding fix
- `spectral_predict_gui_optimized.py:11660` - Store label encoder for later use
- `src/spectral_predict/model_registry.py:144-176` - Model validation registry
- `src/spectral_predict/search.py:127-143` - Original label encoding logic (reference)
- `test_plsda_validation_fix.py` - Test to verify validation fix works correctly

## Additional Context

This fix aligns with the correct validation pattern already used in the model loading code (lines 10206-10250) where task type is properly auto-detected before validating the model type.

The Model Development tab already had the infrastructure:
- `self.refine_task_type` - Task type selector variable
- Auto-detection logic when loading models from Results tab
- Task-specific model combo box updates

The validation function just needed to use this existing infrastructure instead of assuming regression.
