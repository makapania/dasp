# AGENT 5: Quick Start Guide - Tab 7 Diagnostic Plots

**Purpose:** Get Tab 7 diagnostic plots working in 30 minutes

---

## Step 1: Understand What We're Adding (2 min)

**Before:**
- Tab 7 shows predictions in table + text statistics
- NO visual diagnostics

**After:**
- Tab 7 shows predictions in table + text statistics + 3 diagnostic plots
- Plots appear ONLY when using validation set (conditional)

**3 Plots:**
1. **Predictions:** Observed vs Predicted (scatter + 1:1 line + stats)
2. **Residuals:** 4-panel diagnostics (residuals vs fitted, vs index, Q-Q, histogram)
3. **Comparison:** Bar chart comparing multiple models (if >1 model loaded)

---

## Step 2: Backup Current File (1 min)

```bash
cd "C:\Users\sponheim\git\dasp"
cp spectral_predict_gui_optimized.py spectral_predict_gui_optimized.py.backup
```

---

## Step 3: Open Implementation Code (1 min)

Open: `AGENT5_TAB7_PLOT_IMPLEMENTATION.py`

This file contains 5 parts (all copy-paste ready):
- Part 1: UI frames (plot canvases)
- Part 2: Helper methods
- Part 3: Plot methods
- Part 4: Integration method
- Part 5: Execution flow modification

---

## Step 4: Add UI Frames (5 min)

**Location:** `spectral_predict_gui_optimized.py` line ~5368

**Find this:**
```python
self.pred_stats_text.config(yscrollcommand=stats_scrollbar.set)
```

**After it, add this:**
```python
# === Step 5: Diagnostic Plots (Validation Set Only) ===
step5_frame = ttk.LabelFrame(content_frame, text="Step 5: Diagnostic Plots (Validation Set)", padding="20")
step5_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=10)
content_frame.grid_rowconfigure(row, weight=1)
row += 1

# [Copy rest of Part 1 from AGENT5_TAB7_PLOT_IMPLEMENTATION.py]
```

**Copy:** Full code from Part 1 (lines 33-95 in implementation file)

---

## Step 5: Add Helper Methods (3 min)

**Location:** After `_export_predictions()` method (around line 5900)

**Add these 2 methods:**
```python
def _tab7_clear_plots(self):
    """Clear all Tab 7 diagnostic plots."""
    # [Copy from Part 2]

def _tab7_show_plot_placeholder(self, frame, message):
    """Show placeholder message in plot frame."""
    # [Copy from Part 2]
```

**Copy:** Full code from Part 2 (lines 105-134 in implementation file)

---

## Step 6: Add Plot Methods (10 min)

**Location:** After helper methods

**Add these 3 methods:**
```python
def _tab7_plot_predictions(self, y_true, y_pred, model_name="Model"):
    """Plot observed vs predicted for Tab 7 validation set."""
    # [Copy from Part 3]

def _tab7_plot_residuals(self, y_true, y_pred):
    """Plot residual diagnostics for Tab 7 validation set."""
    # [Copy from Part 3]

def _tab7_plot_model_comparison(self, y_true, predictions_dict):
    """Plot model comparison for Tab 7."""
    # [Copy from Part 3]
```

**Copy:** Full code from Part 3 (lines 144-424 in implementation file)

---

## Step 7: Add Integration Method (3 min)

**Location:** After plot methods

**Add this method:**
```python
def _tab7_generate_plots(self):
    """Generate diagnostic plots for Tab 7 validation set."""
    # [Copy from Part 4]
```

**Copy:** Full code from Part 4 (lines 434-495 in implementation file)

---

## Step 8: Modify Execution Flow (5 min)

**Location:** `_update_prediction_statistics()` method, at the END (line ~5829)

**Find this (at end of method):**
```python
self.pred_stats_text.insert('1.0', stats_text)
self.pred_stats_text.config(state='disabled')
```

**After it, add:**
```python
# Generate plots if using validation set
is_validation = (self.pred_data_source.get() == 'validation' and
                self.validation_y is not None)

if is_validation:
    self._tab7_generate_plots()
else:
    self._tab7_clear_plots()
    if hasattr(self, 'tab7_plot1_frame'):
        self._tab7_show_plot_placeholder(self.tab7_plot1_frame,
            "Diagnostic plots available\nonly for validation set")
        self._tab7_show_plot_placeholder(self.tab7_plot2_frame,
            "Diagnostic plots available\nonly for validation set")
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
            "Diagnostic plots available\nonly for validation set")
```

**Copy:** Full code from Part 5 (lines 505-530 in implementation file)

---

## Step 9: Test Basic Functionality (10 min)

### Test 1: UI Check
1. Launch GUI: `python spectral_predict_gui_optimized.py`
2. Navigate to Tab 7
3. Scroll down - you should see "Step 5: Diagnostic Plots"
4. Should see 3 empty frames with placeholders

**Expected:** Clean layout, no errors

---

### Test 2: Validation Set Workflow
1. Go to Tab 1, load data, create validation set (20%)
2. Go to Tab 7
3. Load a saved model (any .dasp file)
4. Select "Use Pre-Selected Validation Set"
5. Click "Load Data"
6. Click "Run All Models"
7. Scroll to plots

**Expected:**
- Plot 1: Scatter with 1:1 line + statistics
- Plot 2: 4 residual plots
- Plot 3: "Load multiple models for comparison"

---

### Test 3: Multiple Models
1. Clear models
2. Load 3 different models
3. Run predictions again
4. Check Plot 3

**Expected:** Bar chart comparing 3 models

---

### Test 4: New Data (Not Validation)
1. Load models
2. Load CSV or directory data (NOT validation set)
3. Run predictions
4. Check plots

**Expected:** All 3 plots show placeholder: "Diagnostic plots available only for validation set"

---

## Troubleshooting

### Error: "name 'HAS_MATPLOTLIB' is not defined"

**Fix:** At top of file, verify:
```python
try:
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
```

---

### Error: "tab7_plot1_frame not found"

**Fix:** Make sure Step 4 (UI frames) was completed
- Check line ~5368 has new code
- Verify `self.tab7_plot1_frame` is created

---

### Plots don't appear

**Check:**
1. Are you using validation set? (plots only for validation)
2. Did predictions run successfully?
3. Check console for errors
4. Verify `_tab7_generate_plots()` is called (add print statement)

---

### Layout looks broken

**Fix:**
- Resize window to larger size
- Check grid weights are set: `content_frame.grid_rowconfigure(row, weight=1)`
- Verify no `pack()` and `grid()` mixing in same container

---

## Success Criteria

✅ **Working correctly if:**
- Tab 7 loads without errors
- Step 5 section visible
- Placeholders show initially
- Plots appear for validation set
- Plots are clear and readable
- Statistics match plot data
- No crashes on edge cases

---

## Next Steps

After basic testing works:

1. **Run full test suite** (see `AGENT5_TESTING_GUIDE.md`)
2. **Test edge cases** (single sample, perfect predictions, etc.)
3. **Get user feedback**
4. **Iterate if needed**

---

## Rollback (If Needed)

If something goes wrong:

```bash
cd "C:\Users\sponheim\git\dasp"
cp spectral_predict_gui_optimized.py.backup spectral_predict_gui_optimized.py
```

---

## Files Reference

- **Implementation code:** `AGENT5_TAB7_PLOT_IMPLEMENTATION.py`
- **Full testing guide:** `AGENT5_TESTING_GUIDE.md`
- **Analysis & rationale:** `AGENT5_DIAGNOSTIC_PLOT_SUITE_ANALYSIS.md`
- **Complete summary:** `AGENT5_COMPLETE_SUMMARY.md`

---

## Estimated Time

- **Integration:** 30 minutes (Steps 1-8)
- **Testing:** 10 minutes (Step 9)
- **Total:** 40 minutes

---

## Getting Help

If stuck:
1. Check console for error messages
2. Review troubleshooting section above
3. Refer to `AGENT5_TESTING_GUIDE.md` for detailed tests
4. Check `AGENT5_TAB7_PLOT_IMPLEMENTATION.py` for code comments

---

**Quick Start Complete!**

After following these steps, Tab 7 will have professional diagnostic plots for validation set analysis.
