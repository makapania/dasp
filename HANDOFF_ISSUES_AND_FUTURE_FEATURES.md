# DASP Spectral Analysis System - Handoff Document

**Date:** 2025-11-03
**Branch:** gui-redesign
**Status:** 🔧 Issues to Address + 🚀 Future Features
**Last Commit:** e378ab5

---

## 📋 Table of Contents

1. [Current Issues to Fix](#current-issues-to-fix)
2. [Future Features to Implement](#future-features-to-implement)
3. [Technical Clarifications Needed](#technical-clarifications-needed)
4. [System Architecture Overview](#system-architecture-overview)

---

## 🐛 Current Issues to Fix

### Issue 1: Variable Count Mismatch in Model Refinement

**Priority:** HIGH
**Status:** 🔴 Needs Investigation & Fix

#### Problem Description

When a user:
1. Runs analysis with subset models (e.g., top50 variables)
2. Double-clicks a result from the Results tab (e.g., model with 50 variables)
3. Loads it into Custom Model Development tab
4. The wavelength specification only contains 30 variables instead of 50

**Example:**
- Analysis Result shows: `n_vars = 50`
- Custom Model Development loads: Only 30 wavelengths

#### Root Cause Analysis

**Location:** `src/spectral_predict/search.py`, function `_run_single_config()`

```python
def _run_single_config(
    ...
    top_n_vars=30,  # ← HARDCODED LIMIT
    ...
):
```

**The Issue:**
1. During analysis, models are trained on N variables (e.g., 50, 100, 250)
2. Feature importance is computed for all N variables
3. **BUT:** Only the top 30 most important are saved to `top_vars` field (line 656-670)
4. When loading for refinement, only these 30 wavelengths are available

**Code Reference:**
```python
# Line 656-670 in search.py
n_to_select = min(top_n_vars, len(importances))  # Limits to 30
top_indices = np.argsort(importances)[-n_to_select:][::-1]
...
top_vars_str = ','.join([f"{w:.1f}" for w in top_wavelengths])
result['top_vars'] = top_vars_str  # Only contains 30 wavelengths
```

#### Proposed Solutions

**Option A: Store All Wavelengths (RECOMMENDED)**
- Change `top_n_vars` parameter to store ALL wavelengths used in subset models
- Modify line 656 to: `n_to_select = len(importances)` for subset models
- Keep top 30 only for display/summary purposes
- Add new column: `all_vars` (all wavelengths) vs `top_vars` (top 30 for display)

**Option B: Make top_n_vars Configurable**
- Add GUI parameter for "Number of top variables to save"
- Pass this value through the analysis pipeline
- Allow users to choose: 30, 50, 100, or "All"

**Option C: Store Subset Tag Metadata**
- For subset models (top50, top100, etc.), store the subset definition
- Reconstruct wavelengths from original data + subset indices
- More complex but avoids storing large wavelength lists

#### Implementation Steps (Option A - Recommended)

1. **Modify `src/spectral_predict/search.py`:**
   ```python
   # Around line 656-670, change:

   # For subset models: save ALL wavelengths used
   if subset_tag != "full":
       n_to_select_all = len(importances)  # Save all for subsets
       all_indices = np.argsort(importances)[-n_to_select_all:][::-1]
       all_wavelengths = wavelengths[all_indices]
       result['all_vars'] = ','.join([f"{w:.1f}" for w in all_wavelengths])

   # For display: save top 30
   n_to_select = min(30, len(importances))
   top_indices = np.argsort(importances)[-n_to_select:][::-1]
   top_wavelengths = wavelengths[top_indices]
   result['top_vars'] = ','.join([f"{w:.1f}" for w in top_wavelengths])
   ```

2. **Modify `spectral_predict_gui_optimized.py`:**
   ```python
   # Around line 2354-2366, change wavelength loading logic:

   # For subset models: prefer all_vars if available
   if 'all_vars' in config and config['all_vars'] != 'N/A' and config['all_vars']:
       wavelength_strings = [w.strip() for w in config['all_vars'].split(',')]
       model_wavelengths = [float(w) for w in wavelength_strings if w]
   elif 'top_vars' in config and config['top_vars'] != 'N/A' and config['top_vars']:
       # Fallback to top_vars (may be incomplete for large subsets)
       wavelength_strings = [w.strip() for w in config['top_vars'].split(',')]
       model_wavelengths = [float(w) for w in wavelength_strings if w]
   ```

3. **Update Results DataFrame Schema:**
   - Add column: `all_vars` (for subset models)
   - Keep column: `top_vars` (for display, always top 30)

#### Testing Checklist

- [ ] Run analysis with top50 variable subset
- [ ] Verify Results tab shows `n_vars = 50`
- [ ] Double-click result to load in Custom Model Development
- [ ] Verify wavelength spec contains all 50 wavelengths
- [ ] Run refined model - should use same 50 wavelengths
- [ ] Verify performance matches original

**Estimated Time:** 2-3 hours

---

## 🚀 Future Features to Implement

### Feature 1: Model Serialization & Persistence

**Priority:** HIGH
**Status:** 🟡 Planned, Not Started
**Estimated Time:** 8-12 hours

#### Overview

Allow users to save trained models from the Custom Model Development tab to disk, then reload them later for predictions on new data.

#### Requirements

**Save Model Functionality:**
1. Add "Save Model" button in Custom Model Development tab
2. Serialize fitted model + metadata to disk
3. Include all information needed to recreate preprocessing + prediction pipeline
4. Use standard format (e.g., joblib, pickle, or custom JSON + joblib)

**What to Save:**
- Fitted model object (sklearn estimator)
- Preprocessing pipeline configuration
- Wavelengths used for training
- Model hyperparameters
- Performance metrics (R², RMSE, etc.)
- Training data statistics (for validation)
- Task type (regression/classification)
- Timestamp and version info

#### Proposed File Format

**Option A: Joblib + JSON (RECOMMENDED)**

File structure: `model_name_YYYYMMDD_HHMMSS.dasp`

```python
# Format: ZIP archive containing:
{
    "metadata.json": {
        "model_name": "PLS",
        "task_type": "regression",
        "preprocessing": "sg1",
        "window": 17,
        "wavelengths": [1500.0, 1501.0, ...],
        "n_vars": 50,
        "performance": {
            "RMSE": 0.125,
            "R2": 0.987,
            "CV_folds": 5
        },
        "training_stats": {
            "X_mean": [...],
            "X_std": [...],
            "y_mean": 15.2,
            "y_std": 3.4
        },
        "created": "2025-11-03T14:30:00",
        "dasp_version": "2.0",
        "model_id": "uuid-string"
    },
    "model.pkl": <joblib serialized sklearn model>,
    "preprocessor.pkl": <joblib serialized preprocessing pipeline>
}
```

#### Implementation Steps

1. **Create new module: `src/spectral_predict/model_io.py`**

```python
import joblib
import json
import zipfile
from datetime import datetime
from pathlib import Path

def save_model(model, preprocessor, metadata, filepath):
    """
    Save a trained model with all metadata.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model
    preprocessor : sklearn Pipeline or None
        Fitted preprocessing pipeline
    metadata : dict
        Model metadata (wavelengths, params, performance, etc.)
    filepath : str or Path
        Where to save (will create .dasp file)
    """
    # Add timestamp and version
    metadata['created'] = datetime.now().isoformat()
    metadata['dasp_version'] = '2.0'

    # Create temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Save components
        json.dump(metadata, open(tmppath / 'metadata.json', 'w'), indent=2)
        joblib.dump(model, tmppath / 'model.pkl')
        if preprocessor is not None:
            joblib.dump(preprocessor, tmppath / 'preprocessor.pkl')

        # Create zip archive
        with zipfile.ZipFile(filepath, 'w') as zf:
            zf.write(tmppath / 'metadata.json', 'metadata.json')
            zf.write(tmppath / 'model.pkl', 'model.pkl')
            if preprocessor is not None:
                zf.write(tmppath / 'preprocessor.pkl', 'preprocessor.pkl')

def load_model(filepath):
    """
    Load a saved model.

    Returns
    -------
    dict with keys:
        'model': fitted model
        'preprocessor': fitted preprocessing pipeline
        'metadata': dict with all metadata
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Extract archive
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(tmppath)

        # Load components
        metadata = json.load(open(tmppath / 'metadata.json'))
        model = joblib.load(tmppath / 'model.pkl')

        preprocessor = None
        if (tmppath / 'preprocessor.pkl').exists():
            preprocessor = joblib.load(tmppath / 'preprocessor.pkl')

        return {
            'model': model,
            'preprocessor': preprocessor,
            'metadata': metadata
        }

def predict_with_model(model_dict, X_new):
    """
    Make predictions with a loaded model.

    Parameters
    ----------
    model_dict : dict
        Output from load_model()
    X_new : pd.DataFrame
        New spectral data (must have wavelengths in columns)

    Returns
    -------
    predictions : np.ndarray
        Predicted values
    """
    # Validate wavelengths
    required_wl = model_dict['metadata']['wavelengths']
    available_wl = X_new.columns.astype(float).values

    # Check if all required wavelengths are present
    missing_wl = set(required_wl) - set(available_wl)
    if missing_wl:
        raise ValueError(f"Missing {len(missing_wl)} required wavelengths")

    # Select and order wavelengths
    wl_cols = [str(wl) for wl in required_wl if str(wl) in X_new.columns]
    X_selected = X_new[wl_cols].values

    # Apply preprocessing if present
    if model_dict['preprocessor'] is not None:
        X_processed = model_dict['preprocessor'].transform(X_selected)
    else:
        X_processed = X_selected

    # Predict
    predictions = model_dict['model'].predict(X_processed)

    return predictions
```

2. **Add "Save Model" button to GUI** (Custom Model Development tab)

```python
# In _create_tab5_refine_model(), after Run Model button:

ttk.Button(
    button_frame,
    text="💾 Save Model",
    command=self._save_refined_model,
    style='Accent.TButton'
).grid(row=0, column=1, padx=5)
```

3. **Implement save handler:**

```python
def _save_refined_model(self):
    """Save the current refined model to disk."""
    if not hasattr(self, 'refined_model') or self.refined_model is None:
        messagebox.showerror("Error", "No model trained yet. Run the model first.")
        return

    # Ask for save location
    filepath = filedialog.asksaveasfilename(
        defaultextension=".dasp",
        filetypes=[("DASP Model", "*.dasp"), ("All files", "*.*")],
        initialfile=f"model_{self.refine_model_type.get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if not filepath:
        return  # User cancelled

    try:
        # Build metadata
        metadata = {
            'model_name': self.refine_model_type.get(),
            'task_type': self.refine_task_type.get(),
            'preprocessing': self.refine_preprocess.get(),
            'window': self.refine_window.get(),
            'wavelengths': self.refined_wavelengths,
            'n_vars': len(self.refined_wavelengths),
            'performance': self.refined_performance,
            # Add more as needed
        }

        # Save using model_io
        from spectral_predict.model_io import save_model
        save_model(
            self.refined_model,
            self.refined_preprocessor,
            metadata,
            filepath
        )

        messagebox.showinfo("Success", f"Model saved to:\n{filepath}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")
        import traceback
        traceback.print_exc()
```

#### Testing Checklist

- [ ] Train a model in Custom Model Development
- [ ] Click "Save Model" button
- [ ] Choose save location
- [ ] Verify .dasp file created
- [ ] Load model back (next feature)
- [ ] Verify predictions match

---

### Feature 2: Model Prediction Tab

**Priority:** HIGH
**Status:** 🟡 Planned, Not Started
**Estimated Time:** 12-16 hours

#### Overview

Create a new tab where users can:
1. Load one or more saved models
2. Upload new spectral data (ASD, CSV, SPC formats)
3. Apply models to make predictions
4. View predictions in a table
5. Export predictions to CSV

#### GUI Design

**New Tab: "🔮 Model Prediction" (Tab 7)**

```
┌─────────────────────────────────────────────────────────┐
│  Model Prediction                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: Load Saved Models                             │
│  ┌───────────────────────────────────────────────────┐ │
│  │ [📂 Load Model]  [🗑️ Clear All]                  │ │
│  │                                                   │ │
│  │ Loaded Models (3):                                │ │
│  │  ✓ PLS_model_20251103.dasp (R²=0.987)           │ │
│  │  ✓ Ridge_model_20251103.dasp (R²=0.976)         │ │
│  │  ✓ Lasso_model_20251103.dasp (R²=0.981)         │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Step 2: Load New Data for Prediction                  │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Data Source:  ⦿ Directory  ⚪ CSV File            │ │
│  │                                                   │ │
│  │ Directory: [/path/to/new/spectra]  [Browse...]   │ │
│  │                                                   │ │
│  │ [📊 Load & Preview Data]                         │ │
│  │                                                   │ │
│  │ Status: 145 spectra loaded                       │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Step 3: Make Predictions                              │
│  ┌───────────────────────────────────────────────────┐ │
│  │ [🚀 Run All Models]  [📥 Export to CSV]         │ │
│  │                                                   │ │
│  │ Progress: [████████████████████] 100% (145/145)  │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Step 4: View Predictions                              │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Sample   | PLS_pred | Ridge_pred | Lasso_pred |  │ │
│  │ ─────────────────────────────────────────────────│ │
│  │ spec_001 | 15.2     | 15.4       | 15.1       |  │ │
│  │ spec_002 | 18.7     | 18.9       | 18.5       |  │ │
│  │ spec_003 | 12.3     | 12.1       | 12.4       |  │ │
│  │ ...                                               │ │
│  │                                                   │ │
│  │ Statistics:                                       │ │
│  │  PLS:   Mean=15.8, Std=3.2, Range=[10.2-22.1]  │ │
│  │  Ridge: Mean=15.9, Std=3.1, Range=[10.4-22.3]  │ │
│  │  Lasso: Mean=15.7, Std=3.3, Range=[10.1-22.0]  │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Implementation Steps

1. **Create tab structure** (in `__init__` around line 195)

```python
tab7_prediction = self._create_tab7_model_prediction()
self.notebook.add(tab7_prediction, text='  🔮 Model Prediction  ')
```

2. **State variables** (in `__init__` around line 90)

```python
self.loaded_models = []  # List of loaded model dicts
self.prediction_data = None  # New data for prediction (DataFrame)
self.predictions_df = None  # Results dataframe
```

3. **Implement tab creation method:**

```python
def _create_tab7_model_prediction(self):
    """Create the Model Prediction tab."""
    frame = ttk.Frame(self.notebook)

    # Use scrollable canvas
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    content_frame = ttk.Frame(canvas)

    content_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=content_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Step 1: Load Models
    step1_frame = ttk.LabelFrame(content_frame, text="Step 1: Load Saved Models", padding="20")
    step1_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=20, pady=10)

    button_frame = ttk.Frame(step1_frame)
    button_frame.grid(row=0, column=0, sticky=tk.W, pady=5)

    ttk.Button(button_frame, text="📂 Load Model",
               command=self._load_model_for_prediction).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="🗑️ Clear All",
               command=self._clear_loaded_models).pack(side=tk.LEFT, padx=5)

    # Loaded models list
    self.loaded_models_text = tk.Text(step1_frame, height=6, width=80, state='disabled')
    self.loaded_models_text.grid(row=1, column=0, pady=5)

    # Step 2: Load Data
    step2_frame = ttk.LabelFrame(content_frame, text="Step 2: Load New Data for Prediction", padding="20")
    step2_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=20, pady=10)

    # Data source selection
    self.pred_data_source = tk.StringVar(value='directory')
    ttk.Radiobutton(step2_frame, text="Directory (ASD/SPC files)",
                    variable=self.pred_data_source, value='directory').grid(row=0, column=0, sticky=tk.W)
    ttk.Radiobutton(step2_frame, text="CSV File",
                    variable=self.pred_data_source, value='csv').grid(row=0, column=1, sticky=tk.W)

    # Path entry
    ttk.Label(step2_frame, text="Path:").grid(row=1, column=0, sticky=tk.W, pady=5)
    self.pred_data_path = tk.StringVar()
    ttk.Entry(step2_frame, textvariable=self.pred_data_path, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
    ttk.Button(step2_frame, text="Browse...",
               command=self._browse_prediction_data).grid(row=1, column=2, padx=5)

    ttk.Button(step2_frame, text="📊 Load & Preview Data",
               command=self._load_prediction_data).grid(row=2, column=0, columnspan=3, pady=10)

    self.pred_data_status = ttk.Label(step2_frame, text="No data loaded")
    self.pred_data_status.grid(row=3, column=0, columnspan=3, pady=5)

    # Step 3: Run Predictions
    step3_frame = ttk.LabelFrame(content_frame, text="Step 3: Make Predictions", padding="20")
    step3_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=20, pady=10)

    pred_button_frame = ttk.Frame(step3_frame)
    pred_button_frame.grid(row=0, column=0, pady=5)

    ttk.Button(pred_button_frame, text="🚀 Run All Models",
               command=self._run_predictions, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
    ttk.Button(pred_button_frame, text="📥 Export to CSV",
               command=self._export_predictions).pack(side=tk.LEFT, padx=5)

    self.pred_progress = ttk.Progressbar(step3_frame, mode='determinate')
    self.pred_progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

    self.pred_status = ttk.Label(step3_frame, text="Ready")
    self.pred_status.grid(row=2, column=0, pady=5)

    # Step 4: Results
    step4_frame = ttk.LabelFrame(content_frame, text="Step 4: View Predictions", padding="20")
    step4_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=10)

    # Results table
    self.predictions_tree = ttk.Treeview(step4_frame, height=15, show='headings')
    self.predictions_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Scrollbars
    pred_vsb = ttk.Scrollbar(step4_frame, orient="vertical", command=self.predictions_tree.yview)
    pred_vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
    self.predictions_tree.configure(yscrollcommand=pred_vsb.set)

    pred_hsb = ttk.Scrollbar(step4_frame, orient="horizontal", command=self.predictions_tree.xview)
    pred_hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
    self.predictions_tree.configure(xscrollcommand=pred_hsb.set)

    # Statistics display
    self.pred_stats_text = tk.Text(step4_frame, height=5, width=80, state='disabled')
    self.pred_stats_text.grid(row=2, column=0, pady=10)

    return frame
```

4. **Implement handler methods:**

```python
def _load_model_for_prediction(self):
    """Load a saved model file."""
    filepath = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("DASP Model", "*.dasp"), ("All files", "*.*")]
    )

    if not filepath:
        return

    try:
        from spectral_predict.model_io import load_model
        model_dict = load_model(filepath)

        # Add to loaded models list
        model_dict['filepath'] = filepath
        model_dict['filename'] = Path(filepath).name
        self.loaded_models.append(model_dict)

        # Update display
        self._update_loaded_models_display()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

def _update_loaded_models_display(self):
    """Update the loaded models list display."""
    self.loaded_models_text.config(state='normal')
    self.loaded_models_text.delete('1.0', 'end')

    if not self.loaded_models:
        self.loaded_models_text.insert('1.0', "No models loaded.")
    else:
        self.loaded_models_text.insert('1.0', f"Loaded Models ({len(self.loaded_models)}):\n")
        for i, model_dict in enumerate(self.loaded_models):
            meta = model_dict['metadata']
            r2 = meta.get('performance', {}).get('R2', 'N/A')
            line = f"  ✓ {model_dict['filename']} (R²={r2})\n"
            self.loaded_models_text.insert('end', line)

    self.loaded_models_text.config(state='disabled')

def _load_prediction_data(self):
    """Load new spectral data for prediction."""
    # Similar to _load_and_plot_data() but simpler
    # Just load spectra, no reference values needed
    # Implementation similar to existing data loading logic
    pass

def _run_predictions(self):
    """Run all loaded models on the prediction data."""
    if not self.loaded_models:
        messagebox.showerror("Error", "No models loaded. Load at least one model first.")
        return

    if self.prediction_data is None:
        messagebox.showerror("Error", "No prediction data loaded.")
        return

    try:
        from spectral_predict.model_io import predict_with_model

        # Initialize results dataframe
        results = pd.DataFrame()
        results['Sample'] = self.prediction_data.index

        # Run each model
        self.pred_progress['maximum'] = len(self.loaded_models)
        self.pred_progress['value'] = 0

        for i, model_dict in enumerate(self.loaded_models):
            model_name = model_dict['metadata']['model_name']
            self.pred_status.config(text=f"Running {model_name}...")
            self.root.update()

            # Make predictions
            predictions = predict_with_model(model_dict, self.prediction_data)
            results[f"{model_name}_pred"] = predictions

            self.pred_progress['value'] = i + 1
            self.root.update()

        self.predictions_df = results
        self._display_predictions()

        self.pred_status.config(text=f"Complete! {len(results)} predictions made.")

    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
        import traceback
        traceback.print_exc()

def _display_predictions(self):
    """Display predictions in the table."""
    # Clear existing
    for item in self.predictions_tree.get_children():
        self.predictions_tree.delete(item)

    # Set columns
    self.predictions_tree['columns'] = list(self.predictions_df.columns)
    for col in self.predictions_df.columns:
        self.predictions_tree.heading(col, text=col)
        self.predictions_tree.column(col, width=120)

    # Populate rows
    for idx, row in self.predictions_df.iterrows():
        values = [row[col] for col in self.predictions_df.columns]
        self.predictions_tree.insert('', 'end', values=values)

    # Calculate and display statistics
    self._update_prediction_statistics()

def _export_predictions(self):
    """Export predictions to CSV."""
    if self.predictions_df is None:
        messagebox.showerror("Error", "No predictions to export.")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if filepath:
        self.predictions_df.to_csv(filepath, index=False)
        messagebox.showinfo("Success", f"Predictions exported to:\n{filepath}")
```

#### Testing Checklist

- [ ] Create new tab appears
- [ ] Load multiple saved models
- [ ] Load new spectral data (ASD directory)
- [ ] Load new spectral data (CSV file)
- [ ] Run predictions with all models
- [ ] View predictions in table
- [ ] Export predictions to CSV
- [ ] Verify predictions are reasonable
- [ ] Test with models trained on different wavelength ranges
- [ ] Test error handling (missing wavelengths, incompatible data)

**Estimated Time:** 12-16 hours

---

## 🔍 Technical Clarifications Needed

### Clarification 1: Preprocessing Application Order

**Question:** How exactly are derivatives and other preprocessing applied? Is it on the full chosen spectrum or differently for subsets?

**Current Understanding (needs verification):**

#### Case 1: Full Spectrum Models
```
User selects wavelength range: 1500-2300 nm (800 wavelengths)
User selects preprocessing: SNV + 1st derivative (sg1)

Processing order:
1. Load full spectrum → [1500.0, 1501.0, ..., 2300.0] (800 values)
2. Apply SNV → normalize each spectrum
3. Apply Savitzky-Golay 1st derivative (window=17) → [1500.0, ..., 2300.0] (796 values due to edge effects)
4. Model trained on 796 preprocessed features
```

#### Case 2: Subset Models (Top-N Variables)
```
User selects: top50 variable subset
Preprocessing: 1st derivative (sg1)

Processing order:
1. Load full spectrum → [1500.0, ..., 2300.0] (800 values)
2. Apply preprocessing to FULL spectrum → (796 values after derivative)
3. Compute feature importance on all 796 preprocessed features
4. Select top 50 most important features → 50 wavelengths
5. Model trained on these 50 preprocessed features ONLY

IMPORTANT: Feature selection happens AFTER preprocessing
```

#### Case 3: Region-Based Subsets
```
Preprocessing: 2nd derivative (sg2)
Region subset: Identified region [1800-1900 nm]

Processing order:
1. Load full spectrum → [1500.0, ..., 2300.0] (800 values)
2. Apply preprocessing to FULL spectrum → (794 values after 2nd derivative)
3. Identify informative regions using PLS on preprocessed data
4. Select wavelengths in region [1800-1900 nm] from preprocessed features
5. Model trained on regional subset of preprocessed features

IMPORTANT: Region identification happens on preprocessed data
```

#### Key Points to Clarify:

**Question 1:** For subset models, are we:
- ✅ **Option A:** Preprocessing full spectrum, THEN selecting important features? (Current implementation)
- ❌ **Option B:** Selecting raw wavelengths first, THEN preprocessing the subset?

**Question 2:** For derivatives:
- What happens at spectrum edges? (Savgol loses `(window-1)/2` points on each end)
- Do wavelength labels shift or stay the same?
- How is this communicated to users?

**Question 3:** For region subsets:
- Are regions identified on raw data or preprocessed data?
- Current implementation: Preprocessed data (line 240 in search.py)
- Is this the intended behavior?

#### Proposed Documentation (once clarified):

**Add to User Manual:**

```markdown
## Preprocessing and Feature Selection

### General Principle
**Preprocessing is ALWAYS applied to the full selected wavelength range BEFORE any subset selection.**

This ensures that:
1. Derivatives and smoothing have proper context
2. Feature importance reflects actual model inputs
3. Spectral features are captured correctly

### Detailed Workflow

#### For Full Spectrum Models:
1. Select wavelength range (e.g., 1500-2300 nm)
2. Apply preprocessing (e.g., SNV, derivatives) to entire range
3. Train model on all preprocessed features

#### For Subset Models (Top-N Variables):
1. Select wavelength range (e.g., 1500-2300 nm)
2. Apply preprocessing to ENTIRE range
3. Train temporary model on all preprocessed features
4. Compute feature importance
5. Select top N most important features
6. Retrain model using ONLY these N features (already preprocessed)

#### For Region-Based Subsets:
1. Select wavelength range
2. Apply preprocessing to ENTIRE range
3. Use PLS to identify informative spectral regions
4. Select wavelengths within identified regions
5. Train model on regional subset (already preprocessed)

### Important Notes

**Derivative Edge Effects:**
- Savitzky-Golay derivatives lose points at spectrum edges
- Window size 17 → loses 8 points on each end
- Example: 800 wavelengths → 784 after derivative
- Wavelength labels remain at center of derivative window

**Why Preprocess First?**
- Derivatives require neighboring points for calculation
- If we selected 50 wavelengths first, derivatives couldn't be computed properly
- Preprocessing the full spectrum preserves spectral context
```

#### Code Locations to Document:

1. **Full spectrum preprocessing:** `src/spectral_predict/search.py`, line 568-573
2. **Subset preprocessing:** `src/spectral_predict/search.py`, line 376-393
3. **Region preprocessing:** `src/spectral_predict/search.py`, line 226-237

**Request:** Please confirm this understanding is correct, then documentation can be finalized.

---

## 📐 System Architecture Overview

### Current System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DASP Analysis Pipeline                   │
└─────────────────────────────────────────────────────────────┘

1. Data Loading (Tab 1: Import & Preview)
   ├─ Load spectral files (ASD, CSV, SPC)
   ├─ Load reference values (CSV)
   ├─ Align samples
   ├─ Select wavelength range
   └─ Optional: Toggle reflectance/absorbance

2. Data Quality Check (Tab 2: NEW)
   ├─ PCA-based outlier detection
   ├─ Q-residuals, Mahalanobis, Y-checks
   ├─ Interactive visualization
   ├─ Mark outliers for exclusion
   └─ Unified exclusion system

3. Analysis Configuration (Tab 3)
   ├─ Select models (PLS, Ridge, Lasso, RF, MLP, NeuralBoosted)
   ├─ Select preprocessing (raw, SNV, sg1, sg2, deriv_snv)
   ├─ Configure hyperparameters
   ├─ Enable subset analysis (top-N, regions)
   └─ Set CV folds

4. Model Search (Backend: search.py)
   ├─ For each preprocessing method:
   │   ├─ Apply to FULL spectrum
   │   ├─ For each model:
   │   │   ├─ Test hyperparameter grid
   │   │   ├─ Run cross-validation
   │   │   ├─ Compute performance metrics
   │   │   ├─ Extract feature importance
   │   │   └─ Test subsets (top-N, regions)
   │   └─ Record all results
   └─ Rank by performance

5. Results Display (Tab 5)
   ├─ Show ranked model results
   ├─ Performance metrics
   ├─ Top wavelengths used
   └─ Double-click to load in Custom Development

6. Custom Model Development (Tab 6)
   ├─ Load model from results
   ├─ Or create fresh model
   ├─ Adjust parameters
   ├─ Test on custom wavelength ranges
   ├─ View detailed performance
   └─ [FUTURE] Save model to disk

7. [FUTURE] Model Prediction (Tab 7)
   ├─ Load saved models
   ├─ Load new spectral data
   ├─ Apply models to predict
   ├─ View predictions table
   └─ Export to CSV
```

### Data Flow for Subset Models

```
Raw Spectra (1500-2300 nm, 800 wavelengths)
            ↓
    Apply Preprocessing
    (e.g., 1st derivative)
            ↓
Preprocessed Features (796 features after edge effects)
            ↓
    Train Temporary Model
    Compute Importances
            ↓
    Select Top 50 Features
            ↓
Subset: 50 Preprocessed Features
            ↓
    Retrain Model on Subset
            ↓
   Final Model (50 features)
```

**Key Insight:** Feature selection happens in the PREPROCESSED space, not raw space.

---

## 📝 Summary

### Immediate Actions Required

1. ✅ **Fix variable count mismatch** (2-3 hours)
   - Store all wavelengths for subset models, not just top 30
   - Implement Option A from Issue 1

2. 📚 **Clarify preprocessing documentation** (1-2 hours)
   - Confirm understanding with user
   - Document in user manual
   - Add inline comments in code

### Future Development (20-30 hours total)

3. 💾 **Implement model saving** (8-12 hours)
   - Create model_io.py module
   - Add Save Model button
   - Test serialization/deserialization

4. 🔮 **Implement prediction tab** (12-16 hours)
   - Create new tab UI
   - Implement multi-model prediction
   - Add CSV export
   - Comprehensive testing

### Testing Protocol

Before merging to main:
- [ ] Fix Issue 1 and test with top50, top100 models
- [ ] Clarify preprocessing and update documentation
- [ ] Test all existing features with real data
- [ ] Verify no regressions introduced

---

**End of Handoff Document**

Generated: 2025-11-03
Branch: gui-redesign
Commit: e378ab5
