"""
AGENT 5: Tab 7 Diagnostic Plot Suite - Complete Implementation
================================================================

This file contains the complete implementation for adding diagnostic plots
to Tab 7 (Model Prediction) when using validation sets.

INTEGRATION INSTRUCTIONS:
1. Add UI elements to _create_tab7_model_prediction() method
2. Add helper methods to class
3. Add plot methods to class
4. Modify _update_prediction_statistics() to call plotting

Location: spectral_predict_gui_optimized.py
"""

# ============================================================================
# PART 1: UI MODIFICATIONS - Add to _create_tab7_model_prediction()
# ============================================================================

def _add_plot_frames_to_tab7(self, content_frame, row):
    """
    Add plot frames to Tab 7 UI.

    Insert this code in _create_tab7_model_prediction() after line 5368
    (after the statistics display).

    Args:
        content_frame: The main content frame for Tab 7
        row: Current grid row number

    Returns:
        Updated row number
    """
    # === Step 5: Diagnostic Plots (Validation Set Only) ===
    step5_frame = ttk.LabelFrame(content_frame, text="Step 5: Diagnostic Plots (Validation Set)", padding="20")
    step5_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=10)
    content_frame.grid_rowconfigure(row, weight=1)  # Make plots expandable
    row += 1

    # Info label
    ttk.Label(step5_frame,
        text="Diagnostic plots are shown when using validation set with known values.",
        style='Caption.TLabel',
        foreground='#666666').grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)

    # Create 3 plot frames side-by-side
    plots_container = ttk.Frame(step5_frame)
    plots_container.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
    step5_frame.grid_rowconfigure(1, weight=1)

    # Plot 1: Prediction Plot (Observed vs Predicted)
    plot1_container = ttk.LabelFrame(plots_container, text="Predictions", padding="10")
    plot1_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    self.tab7_plot1_frame = ttk.Frame(plot1_container, width=350, height=300)
    self.tab7_plot1_frame.pack(fill='both', expand=True)

    # Plot 2: Residual Diagnostics (4-panel)
    plot2_container = ttk.LabelFrame(plots_container, text="Residuals", padding="10")
    plot2_container.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    self.tab7_plot2_frame = ttk.Frame(plot2_container, width=350, height=300)
    self.tab7_plot2_frame.pack(fill='both', expand=True)

    # Plot 3: Model Comparison (if multiple models)
    plot3_container = ttk.LabelFrame(plots_container, text="Model Comparison", padding="10")
    plot3_container.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    self.tab7_plot3_frame = ttk.Frame(plot3_container, width=350, height=300)
    self.tab7_plot3_frame.pack(fill='both', expand=True)

    # Make plots expandable
    plots_container.grid_columnconfigure(0, weight=1)
    plots_container.grid_columnconfigure(1, weight=1)
    plots_container.grid_columnconfigure(2, weight=1)
    plots_container.grid_rowconfigure(0, weight=1)

    # Initialize with placeholders
    self._tab7_show_plot_placeholder(self.tab7_plot1_frame,
        "Load validation set and\\nrun predictions to see plots")
    self._tab7_show_plot_placeholder(self.tab7_plot2_frame,
        "Load validation set and\\nrun predictions to see plots")
    self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
        "Load validation set and\\nrun predictions to see plots")

    return row


# ============================================================================
# PART 2: HELPER METHODS - Add to class
# ============================================================================

def _tab7_clear_plots(self):
    """Clear all Tab 7 diagnostic plots."""
    if not hasattr(self, 'tab7_plot1_frame'):
        return

    for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
        for widget in frame.winfo_children():
            widget.destroy()


def _tab7_show_plot_placeholder(self, frame, message):
    """
    Show placeholder message in plot frame.

    Args:
        frame: The plot frame widget
        message: Message to display (use \\n for line breaks)
    """
    label = ttk.Label(frame, text=message, style='Caption.TLabel',
                     justify='center', anchor='center',
                     foreground='#999999')
    label.pack(expand=True)


# ============================================================================
# PART 3: PLOT METHODS - Add to class
# ============================================================================

def _tab7_plot_predictions(self, y_true, y_pred, model_name="Model"):
    """
    Plot observed vs predicted for Tab 7 validation set.

    Args:
        y_true: array-like, actual validation values
        y_pred: array-like, predicted values
        model_name: str, name of the model for title
    """
    if not HAS_MATPLOTLIB:
        return

    # Clear existing
    for widget in self.tab7_plot1_frame.winfo_children():
        widget.destroy()

    # Create figure
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidths=0.5, s=50,
              color='steelblue')

    # 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'r--', lw=2, label='1:1 Line', zorder=1)

    # Calculate statistics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate bias
    bias = np.mean(y_pred - y_true)

    # Statistics box
    stats_text = f'R² = {r2:.4f}\\nRMSE = {rmse:.4f}\\nMAE = {mae:.4f}\\nBias = {bias:.4f}\\nn = {len(y_true)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'),
            fontsize=9, family='monospace')

    ax.set_xlabel('Observed Values', fontsize=10)
    ax.set_ylabel('Predicted Values', fontsize=10)
    ax.set_title(f'{model_name}\\nValidation Set Performance', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=8)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, self.tab7_plot1_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def _tab7_plot_residuals(self, y_true, y_pred):
    """
    Plot residual diagnostics for Tab 7 validation set.

    Creates a 2x2 grid of diagnostic plots:
    - Residuals vs Fitted
    - Residuals vs Index
    - Q-Q Plot
    - Histogram

    Args:
        y_true: array-like, actual validation values
        y_pred: array-like, predicted values
    """
    if not HAS_MATPLOTLIB:
        return

    from spectral_predict.diagnostics import compute_residuals, qq_plot_data
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Clear existing
    for widget in self.tab7_plot2_frame.winfo_children():
        widget.destroy()

    # Compute residuals
    residuals, std_residuals = compute_residuals(y_true, y_pred)

    # Create 2x2 subplot figure (compact for Tab 7)
    fig = Figure(figsize=(6, 5), dpi=100)

    # Plot 1: Residuals vs Fitted (top left)
    ax1 = fig.add_subplot(221)
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=30,
               color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero line')
    ax1.set_xlabel('Fitted Values', fontsize=8)
    ax1.set_ylabel('Residuals', fontsize=8)
    ax1.set_title('Residuals vs Fitted', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.tick_params(labelsize=7)

    # Add loess smoothing line to detect patterns
    try:
        # Sort by fitted values for smoother line
        sorted_idx = np.argsort(y_pred)
        from scipy.ndimage import uniform_filter1d
        window_size = max(3, len(y_pred) // 10)
        smoothed = uniform_filter1d(residuals[sorted_idx], size=window_size, mode='nearest')
        ax1.plot(y_pred[sorted_idx], smoothed, 'orange', linewidth=2, alpha=0.7, label='Trend')
        ax1.legend(fontsize=6, loc='best')
    except Exception:
        pass  # Skip smoothing if it fails

    # Plot 2: Residuals vs Index (top right)
    ax2 = fig.add_subplot(222)
    indices = np.arange(len(residuals))
    ax2.scatter(indices, residuals, alpha=0.6, edgecolors='black', linewidths=0.5, s=30,
               color='steelblue')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

    # Highlight large residuals (> 2.5σ)
    large_resid_mask = np.abs(std_residuals) > 2.5
    if np.any(large_resid_mask):
        ax2.scatter(indices[large_resid_mask], residuals[large_resid_mask],
                   color='red', s=50, marker='x', linewidths=2, label='Large residuals')
        ax2.legend(fontsize=6, loc='best')

    ax2.set_xlabel('Sample Index', fontsize=8)
    ax2.set_ylabel('Residuals', fontsize=8)
    ax2.set_title('Residuals vs Index', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.tick_params(labelsize=7)

    # Plot 3: Q-Q Plot (bottom left)
    ax3 = fig.add_subplot(223)
    theoretical_q, sample_q = qq_plot_data(residuals)
    ax3.scatter(theoretical_q, sample_q, alpha=0.6, edgecolors='black', linewidths=0.5, s=30,
               color='steelblue')

    # Reference line
    min_q = min(theoretical_q.min(), sample_q.min())
    max_q = max(theoretical_q.max(), sample_q.max())
    ax3.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=1.5, label='Normal line')

    ax3.set_xlabel('Theoretical Quantiles', fontsize=8)
    ax3.set_ylabel('Sample Quantiles', fontsize=8)
    ax3.set_title('Q-Q Plot (Normality)', fontsize=9, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax3.tick_params(labelsize=7)
    ax3.legend(fontsize=6, loc='best')

    # Plot 4: Histogram (bottom right)
    ax4 = fig.add_subplot(224)
    n_bins = min(20, max(10, len(residuals) // 5))  # Adaptive bins
    counts, bins, patches = ax4.hist(residuals, bins=n_bins, alpha=0.7,
                                     edgecolor='black', color='steelblue')

    # Color bars by distance from center (highlight tails)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    std_resid_value = np.std(residuals)
    for patch, center in zip(patches, bin_centers):
        if abs(center) > 2 * std_resid_value:
            patch.set_facecolor('coral')  # Highlight tail bins

    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    ax4.set_xlabel('Residuals', fontsize=8)
    ax4.set_ylabel('Frequency', fontsize=8)
    ax4.set_title('Distribution', fontsize=9, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
    ax4.tick_params(labelsize=7)
    ax4.legend(fontsize=6, loc='best')

    fig.tight_layout(pad=1.5)

    canvas = FigureCanvasTkAgg(fig, self.tab7_plot2_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def _tab7_plot_model_comparison(self, y_true, predictions_dict):
    """
    Plot model comparison for Tab 7 when multiple models are applied.

    Creates a bar chart comparing R² and RMSE for all models.

    Args:
        y_true: array-like, actual validation values
        predictions_dict: dict, {model_name: predictions_array}
    """
    if not HAS_MATPLOTLIB:
        return

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Clear existing
    for widget in self.tab7_plot3_frame.winfo_children():
        widget.destroy()

    if len(predictions_dict) == 0:
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame, "No models to compare")
        return

    if len(predictions_dict) == 1:
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
            "Load multiple models\\nfor comparison")
        return

    # Calculate R² and RMSE for each model
    from sklearn.metrics import r2_score, mean_squared_error

    models = []
    r2_scores = []
    rmse_scores = []

    for model_name, y_pred in predictions_dict.items():
        try:
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            models.append(model_name)
            r2_scores.append(r2)
            rmse_scores.append(rmse)
        except Exception as e:
            print(f"Warning: Could not compute metrics for {model_name}: {e}")
            continue

    if len(models) == 0:
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
            "No valid predictions\\nfor comparison")
        return

    # Sort by R² (best first)
    sorted_indices = np.argsort(r2_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    r2_scores = [r2_scores[i] for i in sorted_indices]
    rmse_scores = [rmse_scores[i] for i in sorted_indices]

    # Create figure
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Create bar plot
    x_pos = np.arange(len(models))

    # Plot R² as bars
    bars = ax.bar(x_pos, r2_scores, alpha=0.8, edgecolor='black', linewidth=1)

    # Color bars by performance (green=best, yellow=medium, red=worst)
    max_r2 = max(r2_scores) if r2_scores else 1.0
    colors = []
    for r2 in r2_scores:
        if r2 >= max_r2 * 0.95:  # Within 5% of best
            colors.append('#4CAF50')  # Green
        elif r2 >= max_r2 * 0.85:  # Within 15% of best
            colors.append('#FFC107')  # Yellow
        else:
            colors.append('#F44336')  # Red

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels on bars
    for i, (r2, rmse) in enumerate(zip(r2_scores, rmse_scores)):
        # R² on top of bar
        ax.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom',
               fontsize=8, fontweight='bold')

        # RMSE inside bar (if space available)
        if r2 > 0.15:
            ax.text(i, r2 / 2, f'RMSE:\\n{rmse:.3f}', ha='center', va='center',
                   fontsize=7, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('R² Score', fontsize=10)
    ax.set_title('Model Comparison\\n(Validation Set Performance)', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)

    # Truncate long model names for readability
    display_names = [name[:20] + '...' if len(name) > 20 else name for name in models]
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)

    # Set y-axis limits
    y_min = min(0, min(r2_scores) - 0.1)
    y_max = min(1.1, max(r2_scores) * 1.15) if r2_scores else 1.1
    ax.set_ylim([y_min, y_max])

    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', edgecolor='black', label='Top tier (≥95% of best)'),
        Patch(facecolor='#FFC107', edgecolor='black', label='Good (≥85% of best)'),
        Patch(facecolor='#F44336', edgecolor='black', label='Needs improvement (<85%)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, self.tab7_plot3_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# ============================================================================
# PART 4: INTEGRATION WITH EXECUTION FLOW
# ============================================================================

def _tab7_generate_plots(self):
    """
    Generate diagnostic plots for Tab 7 validation set.

    Call this from _update_prediction_statistics() when is_validation == True.
    """
    try:
        # Verify we have validation data
        if self.validation_y is None:
            self._tab7_clear_plots()
            return

        # Get actual values aligned with predictions
        y_true = self.validation_y.loc[self.predictions_df['Sample']].values

        # Get prediction columns (exclude 'Sample')
        pred_cols = [col for col in self.predictions_df.columns if col != 'Sample']

        if len(pred_cols) == 0:
            self._tab7_clear_plots()
            self._tab7_show_plot_placeholder(self.tab7_plot1_frame, "No predictions available")
            self._tab7_show_plot_placeholder(self.tab7_plot2_frame, "No predictions available")
            self._tab7_show_plot_placeholder(self.tab7_plot3_frame, "No predictions available")
            return

        # === Plot 1 & 2: Use first model for detailed diagnostics ===
        first_col = pred_cols[0]
        y_pred = self.predictions_df[first_col].values
        model_name = first_col

        # Shorten model name if too long
        if len(model_name) > 40:
            model_name = model_name[:37] + "..."

        self._tab7_plot_predictions(y_true, y_pred, model_name)
        self._tab7_plot_residuals(y_true, y_pred)

        # === Plot 3: Model comparison (if multiple models) ===
        if len(pred_cols) > 1:
            predictions_dict = {col: self.predictions_df[col].values for col in pred_cols}
            self._tab7_plot_model_comparison(y_true, predictions_dict)
        else:
            # Clear and show placeholder
            for widget in self.tab7_plot3_frame.winfo_children():
                widget.destroy()
            self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
                "Load multiple models\\nfor comparison")

        print("✓ Tab 7 diagnostic plots generated successfully")

    except Exception as e:
        print(f"Error generating Tab 7 plots: {e}")
        import traceback
        traceback.print_exc()

        # Show error message in plots
        self._tab7_clear_plots()
        error_msg = f"Error generating plots:\\n{str(e)}"
        self._tab7_show_plot_placeholder(self.tab7_plot1_frame, error_msg)
        self._tab7_show_plot_placeholder(self.tab7_plot2_frame, error_msg)
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame, error_msg)


# ============================================================================
# PART 5: MODIFICATION TO _update_prediction_statistics()
# ============================================================================

def _update_prediction_statistics_modified(self):
    """
    Modified version of _update_prediction_statistics() that includes plotting.

    Replace the existing method or add this code at the end of it:
    """
    # ... [existing statistics code stays the same] ...

    # === NEW CODE: Generate plots if using validation set ===
    # Add this at the END of _update_prediction_statistics(), after line 5829

    # Check if we're using validation set
    is_validation = (self.pred_data_source.get() == 'validation' and
                    self.validation_y is not None)

    if is_validation:
        # Generate diagnostic plots
        self._tab7_generate_plots()
    else:
        # Clear plots and show placeholders
        self._tab7_clear_plots()
        self._tab7_show_plot_placeholder(self.tab7_plot1_frame,
            "Diagnostic plots available\\nonly for validation set")
        self._tab7_show_plot_placeholder(self.tab7_plot2_frame,
            "Diagnostic plots available\\nonly for validation set")
        self._tab7_show_plot_placeholder(self.tab7_plot3_frame,
            "Diagnostic plots available\\nonly for validation set")


# ============================================================================
# END OF IMPLEMENTATION
# ============================================================================

"""
INTEGRATION SUMMARY:
====================

1. Add UI frames:
   - Insert _add_plot_frames_to_tab7() code into _create_tab7_model_prediction()
   - After line 5368 (after statistics display)
   - Update row = _add_plot_frames_to_tab7(self, content_frame, row)

2. Add helper methods to class:
   - _tab7_clear_plots()
   - _tab7_show_plot_placeholder()

3. Add plot methods to class:
   - _tab7_plot_predictions()
   - _tab7_plot_residuals()
   - _tab7_plot_model_comparison()
   - _tab7_generate_plots()

4. Modify execution flow:
   - Add plotting code to END of _update_prediction_statistics()
   - After line 5829 (after inserting stats text)

TESTING:
========

1. Test validation set workflow:
   - Create validation set in tab 1
   - Load multiple models in tab 7
   - Load validation set
   - Run predictions
   - Verify all 3 plots display correctly

2. Test new data workflow:
   - Load models
   - Load CSV or directory data
   - Run predictions
   - Verify placeholder messages show

3. Test edge cases:
   - Single model (plot 3 shows placeholder)
   - Perfect predictions (R²=1.0)
   - Terrible predictions (R²<0)
   - Very few samples (n<5)
"""
