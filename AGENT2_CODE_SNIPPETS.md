# Agent 2: Tab 9 Visualization - Code Snippets Reference

## Quick Reference for Code Changes

---

## 1. Plot Frame Creation (UI Initialization)

### Section C - Line 6053-6054
```python
# Plot frame for transfer quality visualization
self.ct_transfer_plot_frame = ttk.Frame(section_c, style='TFrame')
self.ct_transfer_plot_frame.pack(fill='both', expand=True, pady=(10, 0))
```

### Section D - Line 6083-6084
```python
# Plot frame for equalization visualization
self.ct_equalize_plot_frame = ttk.Frame(section_d, style='TFrame')
self.ct_equalize_plot_frame.pack(fill='both', expand=True, pady=(10, 0))
```

### Section E - Line 6136-6137
```python
# Plot frame for prediction visualization
self.ct_prediction_plot_frame = ttk.Frame(section_e, style='TFrame')
self.ct_prediction_plot_frame.pack(fill='both', expand=True, pady=(10, 0))
```

---

## 2. Function Calls (Integration Points)

### Section C Call - Line 5800
```python
# In _build_ct_transfer_model() after displaying transfer info
# Generate transfer quality plots
self._plot_transfer_quality(method)
```

### Section D Call - Lines 6058-6062
```python
# In _equalize_and_export() after equalization completes
# Generate equalization quality plots
# Prepare data for plotting: need to split X_equalized back by instrument
equalized_by_instrument = {}
start_idx = 0
for inst_id, (_, X) in self.ct_multiinstrument_data.items():
    n_samples = X.shape[0]
    equalized_by_instrument[inst_id] = X_equalized[start_idx:start_idx + n_samples, :]
    start_idx += n_samples

self._plot_equalization_quality(
    self.ct_multiinstrument_data,
    equalized_by_instrument,
    wavelengths_common
)
```

### Section E Call - Line 6220
```python
# In _load_and_predict_ct() after predictions generated
# Generate prediction plots
self._plot_ct_predictions(y_pred)
```

---

## 3. Plotting Functions (New Methods)

### Function 1: _plot_transfer_quality() - Lines 6251-6371

**Purpose:** Create transfer quality diagnostic plots for Section C

**Parameters:**
- `method` (str): Transfer method used ('ds' or 'pds')

**Creates:**
1. 3-panel spectral comparison plot (Master, Before, After)
2. Transfer scatter plot with R²

**Key Code:**
```python
def _plot_transfer_quality(self, method):
    """Plot transfer quality diagnostics for Section C."""
    if not HAS_MATPLOTLIB:
        return

    try:
        # Clear previous plots
        for widget in self.ct_transfer_plot_frame.winfo_children():
            widget.destroy()

        # Apply transfer to get transferred spectra
        if method == 'ds':
            A = self.ct_transfer_model.params['A']
            X_transferred = apply_ds(self.ct_X_slave_common, A)
        elif method == 'pds':
            B = self.ct_transfer_model.params['B']
            window = self.ct_transfer_model.params['window']
            X_transferred = apply_pds(self.ct_X_slave_common, B, window)

        # Create plots...
        # [Plot 1: 3 subplots showing spectra]
        # [Plot 2: Scatter plot with R²]

    except Exception as e:
        print(f"Error creating transfer quality plots: {str(e)}")
```

---

### Function 2: _plot_equalization_quality() - Lines 6373-6491

**Purpose:** Create equalization diagnostic plots for Section D

**Parameters:**
- `instruments_data` (dict): {instrument_id: (wavelengths, X)} before equalization
- `equalized_data` (dict): {instrument_id: X_equalized} after equalization
- `common_grid` (array): Common wavelength grid

**Creates:**
1. Multi-instrument overlay (before/after)
2. Wavelength grid comparison bar chart

**Key Code:**
```python
def _plot_equalization_quality(self, instruments_data, equalized_data, common_grid):
    """Plot equalization quality diagnostics for Section D."""
    if not HAS_MATPLOTLIB:
        return

    try:
        # Clear previous plots
        for widget in self.ct_equalize_plot_frame.winfo_children():
            widget.destroy()

        # Create plots...
        # [Plot 1: Before/after overlay with different colors per instrument]
        # [Plot 2: Wavelength range bar chart with annotations]

    except Exception as e:
        print(f"Error creating equalization plots: {str(e)}")
```

---

### Function 3: _plot_ct_predictions() - Lines 6493-6586

**Purpose:** Create prediction diagnostic plots for Section E

**Parameters:**
- `y_pred` (array): Predicted values

**Creates:**
1. Prediction distribution histogram
2. Sequential prediction plot

**Key Code:**
```python
def _plot_ct_predictions(self, y_pred):
    """Plot prediction results for Section E."""
    if not HAS_MATPLOTLIB:
        return

    try:
        # Clear previous plots
        for widget in self.ct_prediction_plot_frame.winfo_children():
            widget.destroy()

        # Calculate statistics
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)

        # Create plots...
        # [Plot 1: Histogram with mean/std lines]
        # [Plot 2: Scatter plot with connecting line]

    except Exception as e:
        print(f"Error creating prediction plots: {str(e)}")
```

---

## 4. Common Patterns Used

### Pattern 1: Frame Clearing
```python
# Clear previous plots before creating new ones
for widget in self.plot_frame.winfo_children():
    widget.destroy()
```

### Pattern 2: Figure Creation
```python
# Create matplotlib figure
fig = Figure(figsize=(width, height))
ax = fig.add_subplot(subplot_code)
```

### Pattern 3: Canvas Embedding
```python
# Embed in tkinter
canvas = FigureCanvasTkAgg(fig, self.plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(offset, 0))
```

### Pattern 4: Export Button
```python
# Add export functionality
self._add_plot_export_button(self.plot_frame, fig, "filename")
```

### Pattern 5: Error Handling
```python
try:
    # Plotting code
except Exception as e:
    print(f"Error creating plots: {str(e)}")
```

---

## 5. Matplotlib Styling Examples

### Spectral Line Plot with Shaded Region
```python
ax.plot(wavelengths, mean_spectrum, 'b-', linewidth=2, label='Mean')
ax.fill_between(wavelengths,
               mean_spectrum - std_spectrum,
               mean_spectrum + std_spectrum,
               alpha=0.3, color='b', label='±1 Std')
```

### Scatter Plot with Reference Line
```python
ax.scatter(x_data, y_data, alpha=0.3, s=10, edgecolors='none')
ax.plot([min_val, max_val], [min_val, max_val],
        'r--', linewidth=2, label='1:1 Line')
```

### Histogram with Statistical Lines
```python
ax.hist(data, bins=20, alpha=0.7, color='steelblue',
        edgecolor='black', linewidth=1.2)
ax.axvline(mean, color='red', linestyle='--', linewidth=2,
          label=f'Mean = {mean:.3f}')
ax.axvline(mean - std, color='orange', linestyle=':', linewidth=2)
ax.axvline(mean + std, color='orange', linestyle=':', linewidth=2)
```

### Bar Chart with Annotations
```python
bars = ax.barh(y_pos, widths, left=starts, height=0.6)
bars[-1].set_color('red')
bars[-1].set_alpha(0.7)

for i, (label, start, width) in enumerate(zip(labels, starts, widths)):
    ax.text(start + width/2, i,
           f'{start:.0f}-{start+width:.0f} nm',
           ha='center', va='center', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
```

---

## 6. Data Preparation Examples

### Section C: Apply Transfer
```python
# Apply DS transfer
if method == 'ds':
    A = self.ct_transfer_model.params['A']
    X_transferred = apply_ds(self.ct_X_slave_common, A)

# Apply PDS transfer
elif method == 'pds':
    B = self.ct_transfer_model.params['B']
    window = self.ct_transfer_model.params['window']
    X_transferred = apply_pds(self.ct_X_slave_common, B, window)
```

### Section D: Split Equalized Data
```python
# Split combined equalized data back by instrument
equalized_by_instrument = {}
start_idx = 0
for inst_id, (_, X) in self.ct_multiinstrument_data.items():
    n_samples = X.shape[0]
    equalized_by_instrument[inst_id] = X_equalized[start_idx:start_idx + n_samples, :]
    start_idx += n_samples
```

### Section E: Calculate Statistics
```python
# Calculate prediction statistics
mean_pred = np.mean(y_pred)
std_pred = np.std(y_pred)
sample_indices = np.arange(len(y_pred))
```

---

## 7. Testing Code Snippets

### Test Section C Plots
```python
# In Python console or test script
app = SpectralPredictApp(root)

# Simulate building transfer model
app.ct_X_master_common = master_spectra  # Your master data
app.ct_X_slave_common = slave_spectra    # Your slave data
app.ct_wavelengths_common = wavelengths  # Common grid
app.ct_transfer_model = transfer_model   # Your transfer model

# Test plotting
app._plot_transfer_quality('ds')
```

### Test Section D Plots
```python
# Simulate equalization
instruments_data = {
    'inst1': (wl1, X1),
    'inst2': (wl2, X2)
}
equalized_data = {
    'inst1': X1_eq,
    'inst2': X2_eq
}
common_grid = np.arange(400, 2500, 2)

# Test plotting
app._plot_equalization_quality(instruments_data, equalized_data, common_grid)
```

### Test Section E Plots
```python
# Simulate predictions
y_pred = np.random.randn(50) * 2 + 10  # Example predictions

# Test plotting
app._plot_ct_predictions(y_pred)
```

---

## 8. Imports Required

Already present in file:
```python
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import r2_score  # Used in Section C
```

---

## 9. Dependencies on Existing Code

### Uses Existing Methods
- `self._add_plot_export_button()` - For export functionality
- `apply_ds()`, `apply_pds()` - From calibration_transfer module
- `resample_to_grid()` - From calibration_transfer module

### Uses Existing Flags
- `HAS_MATPLOTLIB` - Check before plotting
- `HAS_CALIBRATION_TRANSFER` - Checked by parent functions

### Uses Existing Data Structures
- `self.ct_X_master_common` - Master spectra on common grid
- `self.ct_X_slave_common` - Slave spectra on common grid
- `self.ct_wavelengths_common` - Common wavelength grid
- `self.ct_transfer_model` - TransferModel object
- `self.ct_multiinstrument_data` - Multi-instrument dataset
- `self.ct_pred_y_pred` - Prediction results

---

## 10. Line Numbers Reference

| Section | Description | Lines |
|---------|-------------|-------|
| C Frame | Create plot frame for Section C | 6053-6054 |
| D Frame | Create plot frame for Section D | 6083-6084 |
| E Frame | Create plot frame for Section E | 6136-6137 |
| C Call | Call transfer quality plot | 5800 |
| D Call | Call equalization quality plot | 6058-6062 |
| E Call | Call prediction plot | 6220 |
| Function 1 | `_plot_transfer_quality()` | 6251-6371 |
| Function 2 | `_plot_equalization_quality()` | 6373-6491 |
| Function 3 | `_plot_ct_predictions()` | 6493-6586 |

---

## Total Code Added

- **Lines Added:** ~350 lines (including docstrings)
- **Functions:** 3 new plotting functions
- **Plots:** 6 total diagnostic plots
- **Frame Widgets:** 3 new plot frames
- **Integration Points:** 3 function calls

---

## File Size Comparison

- **Original:** ~6540 lines
- **Modified:** 6892 lines
- **Increase:** 352 lines (+5.4%)

All changes maintain backward compatibility and follow existing code patterns.
