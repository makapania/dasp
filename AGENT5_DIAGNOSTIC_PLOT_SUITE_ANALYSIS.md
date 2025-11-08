# AGENT 5: Diagnostic Plot Suite - Analysis & Implementation

**Date:** 2025-11-07
**Status:** Analysis Complete | Implementation Pending
**Mission:** Verify and enhance diagnostic plotting functions for Tab 7

---

## Executive Summary

After thorough analysis of the codebase, I've determined that:

1. **Tab 7 ("Model Prediction") currently has NO plot frames** - it's designed for applying saved models to new data
2. **Tab 6 ("Custom Model Development") has 3 existing plot methods** that are excellent and production-ready
3. **The mission brief appears to reference Tab 6, not Tab 7** - or Tab 7 needs plot frames added
4. **Recommendation:** Add diagnostic plot frames to Tab 7 for validation set analysis

---

## Current State Analysis

### Tab 6: Custom Model Development (COMPLETE)

**Existing Plot Methods:**
- `_plot_refined_predictions()` (Lines 3666-3731) ✅
- `_plot_residual_diagnostics()` (Lines 3733-3942) ✅
- `_plot_leverage_diagnostics()` (Lines 3943-4027) ✅

**Plot Frames:**
- `self.refine_plot_frame` - For prediction plot
- `self.residual_diagnostics_frame` - For 3-panel residual diagnostics
- `self.leverage_plot_frame` - For leverage analysis

**Data Variables:**
- `self.refined_y_true` - Cross-validation actual values
- `self.refined_y_pred` - Cross-validation predictions
- `self.refined_X_cv` - Feature matrix for leverage calculation
- `self.refined_config` - Model configuration
- `self.refined_prediction_intervals` - Jackknife intervals (PLS only)

**Features Implemented:**
1. **Prediction Plot:**
   - Scatter: Observed vs Predicted
   - 1:1 reference line
   - Error bars (±1 SE from jackknife)
   - Statistics box (R², RMSE, MAE, n)
   - Professional styling

2. **Residual Diagnostics (3-panel):**
   - Panel 1: Residuals vs Fitted
   - Panel 2: Residuals vs Index
   - Panel 3: Q-Q Plot (normality)
   - Dynamic assessment box (color-coded)
   - Detects: outliers, non-normality, heteroscedasticity

3. **Leverage Analysis:**
   - Hat values scatter plot
   - 2p/n and 3p/n threshold lines
   - Color-coded points (normal/moderate/high)
   - Labels for high-leverage samples
   - Dynamic assessment box
   - Statistics (counts, percentages)

**Quality:** Production-ready, well-documented, comprehensive

---

### Tab 7: Model Prediction (NEEDS PLOTS)

**Current Functionality:**
- Load saved .dasp model files
- Load new spectral data (directory/CSV/validation set)
- Apply models to make predictions
- Display predictions in treeview table
- Calculate statistics (with validation metrics if using validation set)
- Export predictions to CSV

**Current Display Elements:**
- `self.loaded_models_text` - Shows loaded models
- `self.predictions_tree` - Treeview table of predictions
- `self.pred_stats_text` - Text display of statistics
- `self.pred_progress` - Progress bar
- `self.pred_status` - Status label

**Missing Elements:**
- ❌ No plot frames defined
- ❌ No plotting methods
- ❌ No visual diagnostics

**Key Insight:** Tab 7 would benefit from plots ONLY when using validation set (actual y values available)

---

## Use Case Analysis

### Tab 6 vs Tab 7 Comparison

| Aspect | Tab 6: Custom Model Development | Tab 7: Model Prediction |
|--------|--------------------------------|------------------------|
| **Purpose** | Refine models from Results tab | Apply saved models to new data |
| **Data Source** | Current dataset (calibration) | New data (directory/CSV/validation) |
| **Has y_true?** | Yes (always - CV) | Only if validation set |
| **Model Status** | Training new model | Using pre-trained model |
| **Plots Needed?** | Yes (always) | Only for validation set |
| **Current Plots** | ✅ Complete (3 plots) | ❌ None |

---

## Implementation Options

### Option A: Add Plots to Tab 7 (Recommended)

**When to show plots:**
- ONLY when using validation set (validation data has actual y values)
- When `self.pred_data_source.get() == 'validation'`

**Plot frames needed:**
1. `self.tab7_plot1_frame` - Prediction plot (observed vs predicted)
2. `self.tab7_plot2_frame` - Residual diagnostics (3-panel)
3. `self.tab7_plot3_frame` - Model comparison plot (if multiple models)

**Data sources:**
- y_true: `self.validation_y` (actual validation values)
- y_pred: `self.predictions_df[col]` (predictions from each model)
- X: `self.validation_X` (features for leverage analysis)

**Integration point:**
- Add to `_display_predictions()` method after line 5740
- Check if validation set is being used
- Generate plots if yes

### Option B: Keep Tab 6 Plots Only (Status Quo)

**Rationale:**
- Tab 6 already has comprehensive plotting
- Tab 7 is for new data (often no y_true available)
- Plotting only makes sense for validation set
- Current text statistics in Tab 7 are adequate

**Pros:**
- No additional work needed
- Tab 6 plots are excellent
- Clear separation of concerns

**Cons:**
- No visual validation in Tab 7
- Users must go to Tab 6 for plots

---

## Detailed Implementation Plan (Option A)

### Step 1: Add Plot Frames to Tab 7 UI

Insert after line 5368 (after statistics display):

```python
# === Step 5: Diagnostic Plots (Validation Set Only) ===
step5_frame = ttk.LabelFrame(content_frame, text="Step 5: Diagnostic Plots (Validation Set)", padding="20")
step5_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=20, pady=10)
row += 1

# Info label
ttk.Label(step5_frame,
    text="Diagnostic plots are shown when using validation set with known values.",
    style='Caption.TLabel').grid(row=0, column=0, columnspan=3, pady=(0, 10))

# Create 3 plot frames side-by-side
plots_container = ttk.Frame(step5_frame)
plots_container.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

# Plot 1: Prediction Plot
plot1_container = ttk.LabelFrame(plots_container, text="Predictions", padding="10")
plot1_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
self.tab7_plot1_frame = ttk.Frame(plot1_container, width=350, height=300)
self.tab7_plot1_frame.pack(fill='both', expand=True)

# Plot 2: Residual Diagnostics
plot2_container = ttk.LabelFrame(plots_container, text="Residuals", padding="10")
plot2_container.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
self.tab7_plot2_frame = ttk.Frame(plot2_container, width=350, height=300)
self.tab7_plot2_frame.pack(fill='both', expand=True)

# Plot 3: Model Comparison
plot3_container = ttk.LabelFrame(plots_container, text="Model Comparison", padding="10")
plot3_container.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
self.tab7_plot3_frame = ttk.Frame(plot3_container, width=350, height=300)
self.tab7_plot3_frame.pack(fill='both', expand=True)

# Make plots expandable
plots_container.grid_columnconfigure(0, weight=1)
plots_container.grid_columnconfigure(1, weight=1)
plots_container.grid_columnconfigure(2, weight=1)
plots_container.grid_rowconfigure(0, weight=1)
step5_frame.grid_rowconfigure(1, weight=1)
```

### Step 2: Create Helper Methods

```python
def _tab7_clear_plots(self):
    """Clear all Tab 7 diagnostic plots."""
    if not hasattr(self, 'tab7_plot1_frame'):
        return

    for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
        for widget in frame.winfo_children():
            widget.destroy()

def _tab7_show_plot_placeholder(self, frame, message):
    """Show placeholder message in plot frame."""
    label = ttk.Label(frame, text=message, style='Caption.TLabel',
                     justify='center', anchor='center')
    label.pack(expand=True)
```

### Step 3: Create Plot Methods

#### 3.1: Prediction Plot (Single Model)

```python
def _tab7_plot_predictions(self, y_true, y_pred, model_name="Model"):
    """Plot observed vs predicted for Tab 7 validation set."""
    if not HAS_MATPLOTLIB:
        return

    # Clear existing
    for widget in self.tab7_plot1_frame.winfo_children():
        widget.destroy()

    # Create figure
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidths=0.5, s=50)

    # 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

    # Calculate statistics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Statistics box
    stats_text = f'R² = {r2:.4f}\\nRMSE = {rmse:.4f}\\nMAE = {mae:.4f}\\nn = {len(y_true)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace')

    ax.set_xlabel('Observed Values', fontsize=10)
    ax.set_ylabel('Predicted Values', fontsize=10)
    ax.set_title(f'{model_name}\\nValidation Set Performance', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, self.tab7_plot1_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
```

#### 3.2: Residual Diagnostics (Simplified)

```python
def _tab7_plot_residuals(self, y_true, y_pred):
    """Plot residual diagnostics for Tab 7 validation set."""
    if not HAS_MATPLOTLIB:
        return

    from spectral_predict.diagnostics import compute_residuals, qq_plot_data

    # Clear existing
    for widget in self.tab7_plot2_frame.winfo_children():
        widget.destroy()

    # Compute residuals
    residuals, std_residuals = compute_residuals(y_true, y_pred)

    # Create 2x2 subplot figure (compact for Tab 7)
    fig = Figure(figsize=(6, 5), dpi=100)

    # Plot 1: Residuals vs Fitted (top left)
    ax1 = fig.add_subplot(221)
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=30)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Fitted', fontsize=8)
    ax1.set_ylabel('Residuals', fontsize=8)
    ax1.set_title('Residuals vs Fitted', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)

    # Plot 2: Residuals vs Index (top right)
    ax2 = fig.add_subplot(222)
    indices = np.arange(len(residuals))
    ax2.scatter(indices, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Index', fontsize=8)
    ax2.set_ylabel('Residuals', fontsize=8)
    ax2.set_title('Residuals vs Index', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)

    # Plot 3: Q-Q Plot (bottom left)
    ax3 = fig.add_subplot(223)
    theoretical_q, sample_q = qq_plot_data(residuals)
    ax3.scatter(theoretical_q, sample_q, alpha=0.6, edgecolors='black', linewidths=0.5, s=30)
    min_q = min(theoretical_q.min(), sample_q.min())
    max_q = max(theoretical_q.max(), sample_q.max())
    ax3.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=1.5)
    ax3.set_xlabel('Theoretical Quantiles', fontsize=8)
    ax3.set_ylabel('Sample Quantiles', fontsize=8)
    ax3.set_title('Q-Q Plot', fontsize=9, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=7)

    # Plot 4: Histogram (bottom right)
    ax4 = fig.add_subplot(224)
    ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Residuals', fontsize=8)
    ax4.set_ylabel('Frequency', fontsize=8)
    ax4.set_title('Distribution', fontsize=9, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(labelsize=7)

    fig.tight_layout(pad=1.5)

    canvas = FigureCanvasTkAgg(fig, self.tab7_plot2_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
```

#### 3.3: Model Comparison Plot

```python
def _tab7_plot_model_comparison(self, y_true, predictions_dict):
    """
    Plot model comparison for Tab 7 when multiple models are applied.

    Args:
        y_true: Actual values (validation set)
        predictions_dict: Dict of {model_name: predictions_array}
    """
    if not HAS_MATPLOTLIB:
        return

    # Clear existing
    for widget in self.tab7_plot3_frame.winfo_children():
        widget.destroy()

    if len(predictions_dict) == 0:
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame, "No models to compare")
        return

    # Create figure
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Calculate R² for each model
    from sklearn.metrics import r2_score, mean_squared_error

    models = []
    r2_scores = []
    rmse_scores = []

    for model_name, y_pred in predictions_dict.items():
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        models.append(model_name)
        r2_scores.append(r2)
        rmse_scores.append(rmse)

    # Sort by R² (best first)
    sorted_indices = np.argsort(r2_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    r2_scores = [r2_scores[i] for i in sorted_indices]
    rmse_scores = [rmse_scores[i] for i in sorted_indices]

    # Create bar plot
    x_pos = np.arange(len(models))

    # Plot R² as bars
    bars = ax.bar(x_pos, r2_scores, alpha=0.7, edgecolor='black', linewidth=1)

    # Color bars by performance (green=best, yellow=medium, red=worst)
    max_r2 = max(r2_scores) if r2_scores else 1.0
    for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
        if r2 >= max_r2 * 0.95:  # Within 5% of best
            bar.set_color('#4CAF50')  # Green
        elif r2 >= max_r2 * 0.85:  # Within 15% of best
            bar.set_color('#FFC107')  # Yellow
        else:
            bar.set_color('#F44336')  # Red

    # Add value labels on bars
    for i, (r2, rmse) in enumerate(zip(r2_scores, rmse_scores)):
        ax.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i, r2 - 0.05, f'RMSE:\\n{rmse:.3f}', ha='center', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('R² Score', fontsize=10)
    ax.set_title('Model Comparison\\nValidation Set Performance', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, min(1.1, max(r2_scores) * 1.15) if r2_scores else 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, self.tab7_plot3_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
```

### Step 4: Integration with Execution Flow

Modify `_update_prediction_statistics()` to call plotting:

```python
def _update_prediction_statistics(self):
    """Calculate and display prediction statistics."""
    if self.predictions_df is None or self.predictions_df.empty:
        return

    # ... [existing statistics code] ...

    # Generate plots if using validation set
    if is_validation:
        self._tab7_generate_plots()
    else:
        self._tab7_clear_plots()
        self._tab7_show_plot_placeholder(self.tab7_plot1_frame,
            "Plots available only for validation set")
        self._tab7_show_plot_placeholder(self.tab7_plot2_frame,
            "Plots available only for validation set")
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
            "Plots available only for validation set")

def _tab7_generate_plots(self):
    """Generate diagnostic plots for Tab 7 validation set."""
    try:
        # Get validation data
        y_true = self.validation_y.loc[self.predictions_df['Sample']].values

        # Get prediction columns
        pred_cols = [col for col in self.predictions_df.columns if col != 'Sample']

        if len(pred_cols) == 0:
            return

        # Plot 1 & 2: Use first model for detailed diagnostics
        first_col = pred_cols[0]
        y_pred = self.predictions_df[first_col].values
        model_name = first_col

        self._tab7_plot_predictions(y_true, y_pred, model_name)
        self._tab7_plot_residuals(y_true, y_pred)

        # Plot 3: Model comparison (if multiple models)
        if len(pred_cols) > 1:
            predictions_dict = {col: self.predictions_df[col].values for col in pred_cols}
            self._tab7_plot_model_comparison(y_true, predictions_dict)
        else:
            self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
                "Load multiple models\\nfor comparison")

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
```

---

## Testing Checklist

### UI Testing
- [ ] Tab 7 loads without errors
- [ ] Plot frames are visible and properly sized
- [ ] Layout adjusts properly on window resize
- [ ] Placeholders show when no validation set

### Functional Testing
1. **Validation Set - Single Model:**
   - [ ] Load 1 model
   - [ ] Load validation set
   - [ ] Run predictions
   - [ ] Verify Plot 1 shows observed vs predicted
   - [ ] Verify Plot 2 shows 4-panel residual diagnostics
   - [ ] Verify Plot 3 shows placeholder

2. **Validation Set - Multiple Models:**
   - [ ] Load 3+ models
   - [ ] Load validation set
   - [ ] Run predictions
   - [ ] Verify Plot 1 shows first model predictions
   - [ ] Verify Plot 2 shows first model residuals
   - [ ] Verify Plot 3 shows model comparison bars

3. **New Data (No Validation):**
   - [ ] Load models
   - [ ] Load CSV or directory data
   - [ ] Run predictions
   - [ ] Verify all plots show placeholder message

### Edge Cases
- [ ] Test with 1 sample (should not crash)
- [ ] Test with perfect predictions (R²=1.0)
- [ ] Test with terrible predictions (R²<0)
- [ ] Test with models that fail (some succeed, some fail)
- [ ] Test window resize during plotting

---

## Recommendation

**I recommend implementing Option A** (add plots to Tab 7) because:

1. **User Value:** Visual validation is crucial for model assessment
2. **Complementary:** Tab 6 for development, Tab 7 for validation
3. **Conditional:** Only shows when validation set is used (doesn't clutter for new data)
4. **Reusability:** Can leverage Tab 6's plotting infrastructure
5. **Completeness:** Makes Tab 7 a complete validation workflow

**Alternative:** If you prefer minimal changes, Tab 6 plots are already excellent and sufficient.

---

## Files to Modify

1. `spectral_predict_gui_optimized.py`:
   - Line ~5368: Add plot frames to UI
   - Line ~5830: Add plotting methods
   - Line ~5740: Integrate with statistics display

**Estimated effort:** 2-3 hours for complete implementation and testing

---

**End of Analysis**
