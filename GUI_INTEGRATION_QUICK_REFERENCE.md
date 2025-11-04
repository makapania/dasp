# Outlier Detection - GUI Integration Quick Reference

**For GUI Developer** | **File:** `spectral_predict_gui_optimized.py`

---

## 1. Import Statement

```python
from src.spectral_predict.outlier_detection import generate_outlier_report
```

---

## 2. Run Detection (Button Handler)

```python
def run_outlier_detection(self):
    """Called when user clicks 'Run Outlier Detection' button."""

    # Get parameters from GUI controls
    n_components = self.pca_components_var.get()  # Spinbox value
    y_min = self.y_min_var.get() or None  # None if empty
    y_max = self.y_max_var.get() or None  # None if empty

    # Run detection
    self.outlier_report = generate_outlier_report(
        X=self.X_data,  # Your loaded spectral data
        y=self.y_data,  # Your loaded reference values
        n_pca_components=n_components,
        y_lower_bound=y_min,
        y_upper_bound=y_max
    )

    # Update GUI displays
    self.update_summary_text()
    self.create_plots()
    self.populate_outlier_table()
```

---

## 3. Display Summary Text

```python
def update_summary_text(self):
    """Update summary text widget with detection results."""

    r = self.outlier_report

    summary = f"""
Detection Results:
  Hotelling T²: {r['pca']['n_outliers']} outliers
  Q-residuals: {r['q_residuals']['n_outliers']} outliers
  Mahalanobis: {r['mahalanobis']['n_outliers']} outliers
  Y-values: {r['y_consistency']['n_outliers']} outliers

Confidence Levels:
  High (3+ methods): {len(r['high_confidence_outliers'])} samples
  Moderate (2 methods): {len(r['moderate_confidence_outliers'])} samples
  Low (1 method): {len(r['low_confidence_outliers'])} samples

PCA Variance: {r['pca']['variance_explained'].sum():.1%} (total)
"""

    self.summary_text.delete('1.0', tk.END)
    self.summary_text.insert('1.0', summary)
```

---

## 4. Create Plots (Matplotlib)

```python
def create_plots(self):
    """Create 5 outlier detection plots."""

    r = self.outlier_report

    # Plot 1: PCA Score Plot (PC1 vs PC2)
    ax1.scatter(
        r['pca']['scores'][:, 0],  # PC1
        r['pca']['scores'][:, 1],  # PC2
        c=self.y_data,              # Color by Y value
        s=r['pca']['hotelling_t2'] * 5,  # Size by T²
        alpha=0.6
    )
    ax1.set_xlabel(f"PC1 ({r['pca']['variance_explained'][0]:.1%})")
    ax1.set_ylabel(f"PC2 ({r['pca']['variance_explained'][1]:.1%})")

    # Plot 2: Hotelling T² Chart
    sample_indices = range(len(r['pca']['hotelling_t2']))
    ax2.bar(sample_indices, r['pca']['hotelling_t2'])
    ax2.axhline(r['pca']['t2_threshold'], color='r', linestyle='--',
                label='95% threshold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Hotelling T²')

    # Plot 3: Q-Residuals Chart
    ax3.bar(sample_indices, r['q_residuals']['q_residuals'])
    ax3.axhline(r['q_residuals']['q_threshold'], color='r', linestyle='--')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Q-Residual')

    # Plot 4: Mahalanobis Distance
    ax4.bar(sample_indices, r['mahalanobis']['distances'])
    ax4.axhline(r['mahalanobis']['threshold'], color='r', linestyle='--')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Mahalanobis Distance')

    # Plot 5: Y Value Distribution (Box Plot)
    ax5.boxplot(self.y_data)
    outlier_indices = r['y_consistency']['outlier_indices']
    ax5.scatter([1] * len(outlier_indices),
                self.y_data[outlier_indices],
                color='red', s=50)
    ax5.set_ylabel('Y Value')
```

---

## 5. Populate Table (Treeview)

```python
def populate_outlier_table(self):
    """Populate Treeview with outlier summary data."""

    # Clear existing items
    for item in self.tree.get_children():
        self.tree.delete(item)

    # Get summary DataFrame
    df = self.outlier_report['outlier_summary']

    # Define columns
    columns = ['Sample', 'Y_Value', 'Total_Flags',
               'T2', 'Q', 'Maha', 'Y']
    self.tree['columns'] = columns

    # Configure column headings
    self.tree.heading('Sample', text='Sample Index')
    self.tree.heading('Y_Value', text='Y Value')
    self.tree.heading('Total_Flags', text='Flags')
    # ... etc

    # Insert rows
    for idx, row in df.iterrows():
        values = (
            row['Sample_Index'],
            f"{row['Y_Value']:.2f}",
            row['Total_Flags'],
            '✓' if row['T2_Outlier'] else '',
            '✓' if row['Q_Outlier'] else '',
            '✓' if row['Maha_Outlier'] else '',
            '✓' if row['Y_Outlier'] else ''
        )

        # Color code by confidence
        if row['Total_Flags'] >= 3:
            tags = ('high_confidence',)
        elif row['Total_Flags'] == 2:
            tags = ('moderate_confidence',)
        else:
            tags = ()

        self.tree.insert('', tk.END, values=values, tags=tags)

    # Configure tag colors
    self.tree.tag_configure('high_confidence', background='#ffcccc')
    self.tree.tag_configure('moderate_confidence', background='#ffffcc')
```

---

## 6. Handle Sample Selection

```python
def mark_samples_for_exclusion(self):
    """Mark selected samples for exclusion from analysis."""

    # Get selected items from Treeview
    selected_items = self.tree.selection()

    # Extract sample indices
    self.excluded_samples = []
    for item in selected_items:
        values = self.tree.item(item, 'values')
        sample_idx = int(values[0])
        self.excluded_samples.append(sample_idx)

    # Create filtered dataset
    mask = np.ones(len(self.X_data), dtype=bool)
    mask[self.excluded_samples] = False

    self.X_filtered = self.X_data[mask]
    self.y_filtered = self.y_data[mask]

    # Update status label
    self.status_label.config(
        text=f"Excluded {len(self.excluded_samples)} samples. "
             f"Using {len(self.X_filtered)} samples for analysis."
    )

def auto_select_high_confidence(self):
    """Auto-select high confidence outliers in table."""

    df = self.outlier_report['high_confidence_outliers']

    # Clear current selection
    for item in self.tree.selection():
        self.tree.selection_remove(item)

    # Select high confidence rows
    for idx, row in df.iterrows():
        sample_idx = row['Sample_Index']
        # Find corresponding tree item and select it
        for item in self.tree.get_children():
            values = self.tree.item(item, 'values')
            if int(values[0]) == sample_idx:
                self.tree.selection_add(item)
                break
```

---

## 7. Export Report

```python
def export_outlier_report(self):
    """Export outlier detection report to CSV."""

    from tkinter import filedialog
    import pandas as pd

    # Ask user for save location
    filepath = filedialog.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
    )

    if not filepath:
        return

    # Create export DataFrame
    df = self.outlier_report['outlier_summary'].copy()

    # Add metadata
    df.insert(0, 'Export_Date', pd.Timestamp.now())
    df.insert(1, 'User', os.getenv('USERNAME', 'unknown'))

    # Save to CSV
    df.to_csv(filepath, index=False)

    messagebox.showinfo('Success', f'Report exported to:\n{filepath}')
```

---

## 8. Data Structure Reference

### Main Report Structure

```python
report = {
    'pca': {
        'pca_model': PCA object,
        'scores': ndarray (n_samples, n_components),
        'loadings': ndarray (n_wavelengths, n_components),
        'variance_explained': ndarray (n_components,),
        'hotelling_t2': ndarray (n_samples,),
        't2_threshold': float,
        'outlier_flags': ndarray (n_samples,) bool,
        'n_outliers': int,
        'outlier_indices': ndarray
    },
    'q_residuals': {
        'q_residuals': ndarray (n_samples,),
        'q_threshold': float,
        'outlier_flags': ndarray (n_samples,) bool,
        'n_outliers': int,
        'outlier_indices': ndarray
    },
    'mahalanobis': {
        'distances': ndarray (n_samples,),
        'median': float,
        'mad': float,
        'threshold': float,
        'outlier_flags': ndarray (n_samples,) bool,
        'n_outliers': int,
        'outlier_indices': ndarray
    },
    'y_consistency': {
        'mean': float,
        'std': float,
        'median': float,
        'min': float,
        'max': float,
        'z_scores': ndarray (n_samples,),
        'z_outliers': ndarray (n_samples,) bool,
        'range_outliers': ndarray (n_samples,) bool,
        'all_outliers': ndarray (n_samples,) bool,
        'n_outliers': int,
        'outlier_indices': ndarray
    },
    'combined_flags': ndarray (n_samples,) bool,
    'outlier_summary': DataFrame,  # See columns below
    'high_confidence_outliers': DataFrame,  # Total_Flags >= 3
    'moderate_confidence_outliers': DataFrame,  # Total_Flags == 2
    'low_confidence_outliers': DataFrame  # Total_Flags == 1
}
```

### Summary DataFrame Columns

```python
columns = [
    'Sample_Index',          # int
    'Y_Value',               # float
    'Hotelling_T2',          # float
    'T2_Outlier',            # bool
    'Q_Residual',            # float
    'Q_Outlier',             # bool
    'Mahalanobis_Distance',  # float
    'Maha_Outlier',          # bool
    'Y_ZScore',              # float
    'Y_Outlier',             # bool
    'Total_Flags'            # int (0-4)
]
```

---

## 9. GUI Layout Suggestion

```python
# In create_data_quality_tab()

# Top frame: Controls
controls_frame = ttk.Frame(self.quality_tab)
controls_frame.pack(fill=tk.X, padx=10, pady=10)

ttk.Button(controls_frame, text="Run Outlier Detection",
           command=self.run_outlier_detection).pack(side=tk.LEFT, padx=5)
ttk.Button(controls_frame, text="Reset",
           command=self.reset_outlier_detection).pack(side=tk.LEFT, padx=5)
ttk.Button(controls_frame, text="Export Report",
           command=self.export_outlier_report).pack(side=tk.LEFT, padx=5)

# Parameters
ttk.Label(controls_frame, text="PCA components:").pack(side=tk.LEFT, padx=5)
self.pca_components_var = tk.IntVar(value=5)
ttk.Spinbox(controls_frame, from_=1, to=20, width=5,
            textvariable=self.pca_components_var).pack(side=tk.LEFT)

# Middle frame: Plots
plots_frame = ttk.Frame(self.quality_tab)
plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create matplotlib figure with 5 subplots
# ... (use FigureCanvasTkAgg)

# Bottom frame: Table and actions
table_frame = ttk.Frame(self.quality_tab)
table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Treeview with scrollbars
self.tree = ttk.Treeview(table_frame)
# ... (add scrollbars, configure columns)

# Actions
actions_frame = ttk.Frame(self.quality_tab)
actions_frame.pack(fill=tk.X, padx=10, pady=10)

ttk.Checkbutton(actions_frame, text="Auto-select high confidence",
                command=self.auto_select_high_confidence).pack(side=tk.LEFT)
ttk.Button(actions_frame, text="Mark for Exclusion",
           command=self.mark_samples_for_exclusion).pack(side=tk.LEFT, padx=5)
```

---

## 10. Error Handling

```python
def run_outlier_detection(self):
    try:
        # Check if data is loaded
        if self.X_data is None or self.y_data is None:
            messagebox.showerror('Error', 'Please load data first.')
            return

        # Validate parameters
        n_comp = self.pca_components_var.get()
        if n_comp < 1 or n_comp > min(self.X_data.shape):
            messagebox.showerror('Error',
                f'PCA components must be 1-{min(self.X_data.shape)}')
            return

        # Run detection
        self.outlier_report = generate_outlier_report(
            X=self.X_data,
            y=self.y_data,
            n_pca_components=n_comp,
            y_lower_bound=self.y_min_var.get() or None,
            y_upper_bound=self.y_max_var.get() or None
        )

        # Update displays
        self.update_summary_text()
        self.create_plots()
        self.populate_outlier_table()

    except Exception as e:
        messagebox.showerror('Error', f'Outlier detection failed:\n{str(e)}')
        import traceback
        traceback.print_exc()
```

---

## Quick Reference Card

| **Task** | **Function/Attribute** |
|----------|------------------------|
| Run detection | `generate_outlier_report(X, y, n_pca_components, y_lower_bound, y_upper_bound)` |
| Get high confidence outliers | `report['high_confidence_outliers']` |
| Get summary table | `report['outlier_summary']` |
| Get PC scores for plot | `report['pca']['scores']` |
| Get T² values | `report['pca']['hotelling_t2']` |
| Get T² threshold | `report['pca']['t2_threshold']` |
| Get Q-residuals | `report['q_residuals']['q_residuals']` |
| Get Mahalanobis distances | `report['mahalanobis']['distances']` |
| Check total flags per sample | `report['outlier_summary']['Total_Flags']` |
| Get outlier indices (any method) | `report['<method>']['outlier_indices']` |

---

## Example: Minimal Working Implementation

```python
import tkinter as tk
from tkinter import ttk
from src.spectral_predict.outlier_detection import generate_outlier_report
import numpy as np

class MinimalOutlierTab:
    def __init__(self, parent, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

        # Button
        ttk.Button(parent, text="Run Detection",
                   command=self.run_detection).pack()

        # Text output
        self.text = tk.Text(parent, height=20, width=60)
        self.text.pack()

    def run_detection(self):
        # Run detection
        report = generate_outlier_report(self.X_data, self.y_data)

        # Display results
        self.text.delete('1.0', tk.END)
        self.text.insert('1.0', f"""
Outlier Detection Results:

High Confidence Outliers: {len(report['high_confidence_outliers'])}
Moderate Confidence: {len(report['moderate_confidence_outliers'])}
Low Confidence: {len(report['low_confidence_outliers'])}

High Confidence Sample Indices:
{report['high_confidence_outliers']['Sample_Index'].tolist()}
""")

# Usage:
# tab = MinimalOutlierTab(notebook_tab, X, y)
```

---

## Need Help?

See full examples in:
- `outlier_detection_usage_example.py` - Complete workflow example
- `OUTLIER_DETECTION_MODULE_SUMMARY.md` - Full documentation
- `test_outlier_detection.py` - Unit tests

Module location: `src/spectral_predict/outlier_detection.py`
