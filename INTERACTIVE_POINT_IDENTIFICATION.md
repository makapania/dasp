# Interactive Point Identification Implementation

## Overview

Added interactive click-to-identify functionality across all plot types in the Import, Data Quality, and Model Development pages. When you click on a point in any figure or line in a line graph, you'll see a yellow annotation box showing the specimen ID, Y variable value, and additional relevant information.

## Implementation Summary

### Core Infrastructure (spectral_predict_gui_optimized.py)

#### New Instance Variables
- `self.plot_annotations` - Dictionary tracking annotation objects per canvas
- `self.active_annotation` - Currently visible annotation object
- `self.refined_cv_indices` - Array of CV fold indices for mapping predictions back to specimen IDs

#### New Helper Methods

**`_format_specimen_info(specimen_idx, y_value=None, y_pred=None, extra_info=None)`**
- Formats specimen information for display in annotations
- Handles both regression (continuous values) and classification (text labels)
- Automatically decodes classification labels using `label_encoder`
- Supports additional information via `extra_info` dictionary

**`_create_or_update_annotation(ax, x, y, text, canvas)`**
- Creates or updates the yellow annotation box at clicked point
- Removes previous annotation before creating new one
- Uses arrow to point to exact clicked location
- Positioned with offset to avoid obscuring data

### Import Page (Tab 1)

**Modified Methods:**
- `_on_spectrum_click()` - Enhanced to show specimen ID and Y value when clicking spectra
- `_create_plot_tab()` - Enabled click handlers on all spectral plots (raw, 1st derivative, 2nd derivative)

**Information Displayed:**
- Specimen ID
- Y value (continuous or classification label)

**Behavior:**
- Click toggles spectrum visibility (existing feature)
- Shows annotation at middle point of spectrum
- Works on raw, 1st derivative, and 2nd derivative plots

### Data Quality Page (Tab 3)

**Modified Methods:**

**`_plot_pca_scores()`** - PCA scatter plot
- Click handler uses nearest-point detection
- Displays: Specimen ID, Y value, PC1, PC2, outlier status

**`_plot_hotelling_t2()`** - Hotelling T² bar chart
- Click handler rounds to nearest bar index
- Displays: Specimen ID, Y value, T² value, threshold, outlier status

**`_plot_q_residuals()`** - Q-residuals bar chart
- Click handler rounds to nearest bar index
- Displays: Specimen ID, Y value, Q-residual value, threshold, outlier status

**`_plot_mahalanobis()`** - Mahalanobis distance bar chart
- Click handler rounds to nearest bar index
- Displays: Specimen ID, Y value, Mahalanobis distance, threshold, outlier status

### Model Development Page (Tab 6)

**CV Index Tracking:**
- Added `all_cv_indices` list in cross-validation loop (line 6883)
- Populated during CV with `test_idx` from each fold (line 6901)
- Stored in `self.refined_cv_indices` for mapping predictions back to original specimen IDs (line 7075)

**Modified Methods:**

**`_plot_regression_predictions()`** - Actual vs Predicted scatter plot
- Click handler finds nearest point
- Maps CV index back to original specimen index
- Displays: Specimen ID, actual Y, predicted Y, residual

**`_plot_regression_residual_diagnostics()`** - Three residual plots
- Added handlers to "Residuals vs Fitted" and "Residuals vs Index" plots
- Displays: Specimen ID, actual Y, predicted Y, fitted value, residual

**`_plot_regression_leverage_diagnostics()`** - Leverage scatter plot
- Click handler finds nearest point
- Displays: Specimen ID, Y value, leverage value, leverage category (High/Moderate/Normal)

## Technical Details

### Click Detection Methods

**Line Plots (Import Page):**
- Uses matplotlib's `pick_event` with `line.set_picker(5)`
- Stores sample index in line's GID property
- Annotation placed at spectrum midpoint

**Scatter Plots (PCA, Predictions, Residuals, Leverage):**
- Uses `button_press_event` with nearest-point calculation
- Calculates Euclidean distance to all points
- Shows annotation only if click within 10% threshold of plot range

**Bar Charts (T², Q-residual, Mahalanobis):**
- Uses `button_press_event` with rounding to nearest integer
- Simple x-coordinate rounding to find bar index

### Label Encoding Support

For classification tasks with text labels (e.g., "High", "Low"):
- `_format_specimen_info()` automatically decodes numeric values using `self.label_encoder`
- Falls back to numeric display if decoding fails
- Shows text labels for both actual and predicted values

### Annotation Styling

- Yellow background with black border for visibility
- Arrow pointing to exact clicked point
- **Smart positioning**: Annotation automatically positions itself to stay within plot bounds
  - Points on left side: annotation appears to the right
  - Points on right side: annotation appears to the left
  - Points on top: annotation appears below
  - Points on bottom: annotation appears above
  - Offset adjusts based on point location (±20 pixels)
- High z-order (1000) to appear on top of all plot elements
- Previous annotation automatically removed when clicking new point
- Text alignment adjusts automatically based on position

## Usage

1. **Import Page:** Click any spectrum line to see specimen ID and Y value (also toggles exclusion)
2. **Data Quality Page:** Click any point or bar to identify outliers by specimen ID
3. **Model Development Page:** Click any prediction point to see which specimen it is and compare actual vs predicted values

## Files Modified

- `spectral_predict_gui_optimized.py` - Main GUI implementation

## Benefits

- **Outlier Investigation:** Easily identify which specimens are outliers in PCA, T², Q-residual, or Mahalanobis plots
- **Prediction Analysis:** Understand which specimens have high residuals or poor predictions
- **Quality Control:** Quickly identify problematic specimens across all analysis stages
- **Leverage Assessment:** See which specimens have high influence on the model

## Future Enhancements (Not Implemented)

- Hover tooltips (currently click-based as requested)
- Multi-point selection
- Export list of clicked points
- Jump between tabs to see same specimen across different views
