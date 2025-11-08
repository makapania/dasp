"""
PHASE 5: Tab 7 Diagnostic Plot Suite - Complete Implementation
================================================================

This file contains professional diagnostic plots for Tab 7 (Model Development)
to visualize model performance after training completion.

CONTEXT:
- Three plot frames already created: tab7_plot1_frame, tab7_plot2_frame, tab7_plot3_frame
- Plots shown after model execution completes
- Uses spectral_predict.diagnostics for helper functions

INTEGRATION:
1. Add methods to SpectralPredictGUI class in spectral_predict_gui_optimized.py
2. Add performance history tracking to __init__
3. Call _tab7_generate_plots() from _tab7_update_results() after updating results text

Author: Claude Code Agent 5
Date: 2025-11-07
"""

import numpy as np
import tkinter as tk
from tkinter import ttk

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# PART 1: HELPER METHODS
# ============================================================================

def _tab7_clear_plots(self):
    """
    Clear all Tab 7 diagnostic plots.

    Removes all widgets from the three plot frames.
    """
    if not hasattr(self, 'tab7_plot1_frame'):
        return

    for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
        if frame is not None:
            for widget in frame.winfo_children():
                widget.destroy()


def _tab7_show_plot_placeholder(self, frame, message):
    """
    Show placeholder message in plot frame.

    Args:
        frame: The plot frame widget
        message: Message to display (use \\n for line breaks)
    """
    if frame is None:
        return

    label = ttk.Label(
        frame,
        text=message,
        justify='center',
        anchor='center',
        foreground='#999999',
        font=('Helvetica', 10)
    )
    label.pack(expand=True)


# ============================================================================
# PART 2: PLOT METHOD 1 - PREDICTIONS SCATTER PLOT
# ============================================================================

def _tab7_plot_predictions(self, y_true, y_pred, model_name="Model"):
    """
    Plot observed vs predicted scatter plot for Tab 7 Model Development.

    Creates a scatter plot with 1:1 reference line showing how well
    the model predictions match the observed values.

    Args:
        y_true: array-like, actual values from training/validation
        y_pred: array-like, predicted values from model
        model_name: str, name of the model for plot title

    Features:
        - Scatter plot with proper styling
        - 1:1 reference line (red dashed)
        - Statistics box with R², RMSE, MAE, Bias, n
        - Equal aspect ratio for accurate visualization
        - Professional grid and labels
    """
    if not HAS_MATPLOTLIB:
        self._tab7_show_plot_placeholder(
            self.tab7_plot1_frame,
            "Matplotlib not available\\nCannot generate plots"
        )
        return

    # Clear existing widgets
    for widget in self.tab7_plot1_frame.winfo_children():
        widget.destroy()

    try:
        # Convert to numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Create matplotlib figure
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Scatter plot
        ax.scatter(
            y_true, y_pred,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
            color='steelblue',
            label='Predictions'
        )

        # 1:1 reference line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05

        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'r--',
            lw=2,
            label='1:1 Line',
            zorder=1
        )

        # Calculate statistics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        n = len(y_true)

        # Statistics box in upper left
        stats_text = (
            f'R² = {r2:.4f}\n'
            f'RMSE = {rmse:.4f}\n'
            f'MAE = {mae:.4f}\n'
            f'Bias = {bias:.4f}\n'
            f'n = {n}'
        )

        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='wheat',
                alpha=0.9,
                edgecolor='black'
            ),
            fontsize=9,
            family='monospace'
        )

        # Labels and title
        ax.set_xlabel('Observed Values', fontsize=10)
        ax.set_ylabel('Predicted Values', fontsize=10)
        ax.set_title(
            f'{model_name}\nModel Development Performance',
            fontsize=11,
            fontweight='bold'
        )

        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='lower right', fontsize=8)

        # Equal aspect ratio for proper visualization
        ax.set_aspect('equal', adjustable='box')

        # Tight layout
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.tab7_plot1_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        print(f"Error creating prediction plot: {e}")
        import traceback
        traceback.print_exc()
        self._tab7_show_plot_placeholder(
            self.tab7_plot1_frame,
            f"Error creating plot:\n{str(e)}"
        )


# ============================================================================
# PART 3: PLOT METHOD 2 - RESIDUAL DIAGNOSTICS (4-PANEL)
# ============================================================================

def _tab7_plot_residuals(self, y_true, y_pred):
    """
    Plot residual diagnostics for Tab 7 Model Development.

    Creates a 2x2 grid of diagnostic plots to assess model quality:
    - Top Left: Residuals vs Fitted (detect heteroscedasticity)
    - Top Right: Residuals vs Index (detect systematic errors)
    - Bottom Left: Q-Q Plot (assess normality)
    - Bottom Right: Residual Histogram (assess distribution)

    Args:
        y_true: array-like, actual values from training/validation
        y_pred: array-like, predicted values from model

    Features:
        - Uses compute_residuals() and qq_plot_data() from diagnostics module
        - Highlights outliers (>2.5σ) in red
        - Smooth trend line to detect patterns
        - Color-coded histogram tails
        - Professional styling
    """
    if not HAS_MATPLOTLIB:
        self._tab7_show_plot_placeholder(
            self.tab7_plot2_frame,
            "Matplotlib not available\\nCannot generate plots"
        )
        return

    # Clear existing widgets
    for widget in self.tab7_plot2_frame.winfo_children():
        widget.destroy()

    try:
        from spectral_predict.diagnostics import compute_residuals, qq_plot_data
        from scipy.ndimage import uniform_filter1d

        # Convert to numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Compute residuals
        residuals, std_residuals = compute_residuals(y_true, y_pred)

        # Create 2x2 subplot figure
        fig = Figure(figsize=(5, 4), dpi=100)

        # ====================================================================
        # SUBPLOT 1: Residuals vs Fitted (Top Left)
        # ====================================================================
        ax1 = fig.add_subplot(221)
        ax1.scatter(
            y_pred, residuals,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5,
            s=30,
            color='steelblue'
        )

        # Zero reference line
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero')

        # Add smooth trend line to detect patterns
        try:
            sorted_idx = np.argsort(y_pred)
            window_size = max(3, len(y_pred) // 10)
            smoothed = uniform_filter1d(
                residuals[sorted_idx],
                size=window_size,
                mode='nearest'
            )
            ax1.plot(
                y_pred[sorted_idx],
                smoothed,
                'orange',
                linewidth=2,
                alpha=0.7,
                label='Trend'
            )
        except Exception:
            pass  # Skip smoothing if it fails

        ax1.set_xlabel('Fitted Values', fontsize=8)
        ax1.set_ylabel('Residuals', fontsize=8)
        ax1.set_title('Residuals vs Fitted', fontsize=9, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.tick_params(labelsize=7)
        ax1.legend(fontsize=6, loc='best')

        # ====================================================================
        # SUBPLOT 2: Residuals vs Index (Top Right)
        # ====================================================================
        ax2 = fig.add_subplot(222)
        indices = np.arange(len(residuals))

        ax2.scatter(
            indices, residuals,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5,
            s=30,
            color='steelblue'
        )

        # Zero reference line
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

        # Highlight large residuals (> 2.5σ) in red
        large_resid_mask = np.abs(std_residuals) > 2.5
        if np.any(large_resid_mask):
            ax2.scatter(
                indices[large_resid_mask],
                residuals[large_resid_mask],
                color='red',
                s=50,
                marker='x',
                linewidths=2,
                label=f'Outliers (>{2.5:.1f}σ)',
                zorder=5
            )
            ax2.legend(fontsize=6, loc='best')

        ax2.set_xlabel('Sample Index', fontsize=8)
        ax2.set_ylabel('Residuals', fontsize=8)
        ax2.set_title('Residuals vs Index', fontsize=9, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.tick_params(labelsize=7)

        # ====================================================================
        # SUBPLOT 3: Q-Q Plot (Bottom Left)
        # ====================================================================
        ax3 = fig.add_subplot(223)

        # Get Q-Q plot data
        theoretical_q, sample_q = qq_plot_data(residuals)

        ax3.scatter(
            theoretical_q, sample_q,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5,
            s=30,
            color='steelblue'
        )

        # 45° reference line
        min_q = min(theoretical_q.min(), sample_q.min())
        max_q = max(theoretical_q.max(), sample_q.max())
        ax3.plot(
            [min_q, max_q],
            [min_q, max_q],
            'r--',
            linewidth=1.5,
            label='Normal line'
        )

        ax3.set_xlabel('Theoretical Quantiles', fontsize=8)
        ax3.set_ylabel('Sample Quantiles', fontsize=8)
        ax3.set_title('Q-Q Plot (Normality)', fontsize=9, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax3.tick_params(labelsize=7)
        ax3.legend(fontsize=6, loc='best')

        # ====================================================================
        # SUBPLOT 4: Residual Histogram (Bottom Right)
        # ====================================================================
        ax4 = fig.add_subplot(224)

        # Adaptive number of bins
        n_bins = min(20, max(10, len(residuals) // 5))

        counts, bins, patches = ax4.hist(
            residuals,
            bins=n_bins,
            alpha=0.7,
            edgecolor='black',
            color='steelblue'
        )

        # Color bars by distance from center (highlight tails > 2σ)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        std_resid_value = np.std(residuals)

        for patch, center in zip(patches, bin_centers):
            if abs(center) > 2 * std_resid_value:
                patch.set_facecolor('coral')  # Highlight tail bins

        # Zero reference line
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')

        ax4.set_xlabel('Residuals', fontsize=8)
        ax4.set_ylabel('Frequency', fontsize=8)
        ax4.set_title('Distribution', fontsize=9, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
        ax4.tick_params(labelsize=7)
        ax4.legend(fontsize=6, loc='best')

        # Tight layout with padding
        fig.tight_layout(pad=1.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.tab7_plot2_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        print(f"Error creating residual plots: {e}")
        import traceback
        traceback.print_exc()
        self._tab7_show_plot_placeholder(
            self.tab7_plot2_frame,
            f"Error creating plot:\n{str(e)}"
        )


# ============================================================================
# PART 4: PLOT METHOD 3 - MODEL COMPARISON BAR CHART
# ============================================================================

def _tab7_plot_model_comparison(self, performance_history):
    """
    Plot model comparison bar chart for Tab 7 Model Development.

    Creates a bar chart comparing multiple model runs with color-coded
    performance tiers.

    Args:
        performance_history: list of dict, each containing:
            - 'label': str, model run description
            - 'r2': float, R² score
            - 'rmse': float, RMSE score

    Features:
        - Bars sorted by R² (best first)
        - Color-coded by performance tier:
            * Green: Top tier (≥95% of best)
            * Yellow: Good (≥85% of best)
            * Red: Needs improvement (<85%)
        - R² value labels on top of bars
        - RMSE labels inside bars
        - Legend explaining color coding
        - Shows placeholder if < 2 models
    """
    if not HAS_MATPLOTLIB:
        self._tab7_show_plot_placeholder(
            self.tab7_plot3_frame,
            "Matplotlib not available\\nCannot generate plots"
        )
        return

    # Clear existing widgets
    for widget in self.tab7_plot3_frame.winfo_children():
        widget.destroy()

    # Check if we have enough models to compare
    if len(performance_history) == 0:
        self._tab7_show_plot_placeholder(
            self.tab7_plot3_frame,
            "No model runs to compare"
        )
        return

    if len(performance_history) == 1:
        self._tab7_show_plot_placeholder(
            self.tab7_plot3_frame,
            "Run multiple models\nfor comparison"
        )
        return

    try:
        from sklearn.metrics import r2_score, mean_squared_error

        # Extract data
        labels = [entry['label'] for entry in performance_history]
        r2_scores = [entry['r2'] for entry in performance_history]
        rmse_scores = [entry['rmse'] for entry in performance_history]

        # Sort by R² (best first)
        sorted_indices = np.argsort(r2_scores)[::-1]
        labels = [labels[i] for i in sorted_indices]
        r2_scores = [r2_scores[i] for i in sorted_indices]
        rmse_scores = [rmse_scores[i] for i in sorted_indices]

        # Create figure
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Create bar plot
        x_pos = np.arange(len(labels))
        bars = ax.bar(
            x_pos,
            r2_scores,
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )

        # Color bars by performance tier
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
            ax.text(
                i, r2 + 0.02,
                f'{r2:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )

            # RMSE inside bar (if space available)
            if r2 > 0.15:
                ax.text(
                    i, r2 / 2,
                    f'RMSE:\n{rmse:.3f}',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white',
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='black',
                        alpha=0.6
                    )
                )

        # Labels and title
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('R² Score', fontsize=10)
        ax.set_title(
            'Model Comparison\n(Training Performance)',
            fontsize=11,
            fontweight='bold'
        )

        # X-axis labels (truncate if too long)
        ax.set_xticks(x_pos)
        display_names = [
            name[:20] + '...' if len(name) > 20 else name
            for name in labels
        ]
        ax.set_xticklabels(
            display_names,
            rotation=45,
            ha='right',
            fontsize=8
        )

        # Y-axis limits
        y_min = min(0, min(r2_scores) - 0.1)
        y_max = min(1.1, max(r2_scores) * 1.15) if r2_scores else 1.1
        ax.set_ylim([y_min, y_max])

        # Grid
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

        # Add legend for color coding
        legend_elements = [
            Patch(
                facecolor='#4CAF50',
                edgecolor='black',
                label='Top tier (≥95% of best)'
            ),
            Patch(
                facecolor='#FFC107',
                edgecolor='black',
                label='Good (≥85% of best)'
            ),
            Patch(
                facecolor='#F44336',
                edgecolor='black',
                label='Needs improvement (<85%)'
            )
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=7)

        # Tight layout
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.tab7_plot3_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        import traceback
        traceback.print_exc()
        self._tab7_show_plot_placeholder(
            self.tab7_plot3_frame,
            f"Error creating plot:\n{str(e)}"
        )


# ============================================================================
# PART 5: MAIN ENTRY POINT
# ============================================================================

def _tab7_generate_plots(self, results):
    """
    Generate all diagnostic plots for Tab 7 Model Development.

    Main entry point called after model execution completes.
    Orchestrates the creation of all three plots.

    Args:
        results: dict containing:
            - 'y_true': array-like, actual values
            - 'y_pred': array-like, predicted values
            - 'model_name': str, name of the model
            - 'r2': float, R² score
            - 'rmse': float, RMSE score

    Features:
        - Checks data availability
        - Calls all three plot methods
        - Handles errors gracefully
        - Updates performance history
    """
    try:
        # Check matplotlib availability
        if not HAS_MATPLOTLIB:
            self._tab7_clear_plots()
            for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                self._tab7_show_plot_placeholder(
                    frame,
                    "Matplotlib not available\nInstall matplotlib for plots"
                )
            return

        # Check if we have the required data
        if 'y_true' not in results or 'y_pred' not in results:
            self._tab7_clear_plots()
            for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                self._tab7_show_plot_placeholder(
                    frame,
                    "No data available\nfor diagnostic plots"
                )
            return

        y_true = results['y_true']
        y_pred = results['y_pred']
        model_name = results.get('model_name', 'Model')

        # Validate data
        if len(y_true) == 0 or len(y_pred) == 0:
            self._tab7_clear_plots()
            for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                self._tab7_show_plot_placeholder(
                    frame,
                    "Empty data arrays\nCannot generate plots"
                )
            return

        if len(y_true) != len(y_pred):
            self._tab7_clear_plots()
            for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
                self._tab7_show_plot_placeholder(
                    frame,
                    "Data length mismatch\nCannot generate plots"
                )
            return

        # === PLOT 1: Predictions scatter plot ===
        self._tab7_plot_predictions(y_true, y_pred, model_name)

        # === PLOT 2: Residual diagnostics ===
        self._tab7_plot_residuals(y_true, y_pred)

        # === Update performance history ===
        # Add current run to history
        if not hasattr(self, 'tab7_performance_history'):
            self.tab7_performance_history = []

        # Create entry for history
        history_entry = {
            'label': model_name,
            'r2': results.get('r2', 0.0),
            'rmse': results.get('rmse', 0.0)
        }

        # Append to history
        self.tab7_performance_history.append(history_entry)

        # Keep last 10 runs only
        if len(self.tab7_performance_history) > 10:
            self.tab7_performance_history = self.tab7_performance_history[-10:]

        # === PLOT 3: Model comparison ===
        self._tab7_plot_model_comparison(self.tab7_performance_history)

        print("✓ Tab 7 diagnostic plots generated successfully")

    except Exception as e:
        print(f"Error generating Tab 7 plots: {e}")
        import traceback
        traceback.print_exc()

        # Show error message in all plots
        self._tab7_clear_plots()
        error_msg = f"Error generating plots:\n{str(e)[:50]}"
        for frame in [self.tab7_plot1_frame, self.tab7_plot2_frame, self.tab7_plot3_frame]:
            self._tab7_show_plot_placeholder(frame, error_msg)


# ============================================================================
# PART 6: INTEGRATION INSTRUCTIONS
# ============================================================================

"""
INTEGRATION INTO spectral_predict_gui_optimized.py
===================================================

STEP 1: Add to __init__ method (around line 150):
--------------------------------------------------
# Tab 7 performance tracking
self.tab7_performance_history = []  # Track multiple model runs for comparison

STEP 2: Add all methods to SpectralPredictGUI class:
-----------------------------------------------------
Copy all methods from this file into the class:
- _tab7_clear_plots()
- _tab7_show_plot_placeholder()
- _tab7_plot_predictions()
- _tab7_plot_residuals()
- _tab7_plot_model_comparison()
- _tab7_generate_plots()

STEP 3: Call from _tab7_update_results() method:
-------------------------------------------------
Add this code at the END of _tab7_update_results(), after updating results text:

    # Generate diagnostic plots
    if results is not None:
        self._tab7_generate_plots(results)

STEP 4: Ensure results dict contains required fields:
------------------------------------------------------
In the code that calls _tab7_update_results(), ensure the results dict contains:
- 'y_true': array of actual values
- 'y_pred': array of predicted values
- 'model_name': string name of the model
- 'r2': R² score
- 'rmse': RMSE score

Example:
results = {
    'y_true': y_train,
    'y_pred': y_pred,
    'model_name': f"{model_type} (n_components={n_components})",
    'r2': r2_score(y_train, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
    # ... other fields
}
self._tab7_update_results(results)
"""


# ============================================================================
# PART 7: TESTING CHECKLIST
# ============================================================================

"""
TESTING CHECKLIST
=================

Test Case 1: Perfect Predictions (R²=1.0)
------------------------------------------
- All points should lie exactly on 1:1 line
- Residuals should all be zero
- Q-Q plot should show perfect normality
- Histogram should be single spike at zero

Test Case 2: Poor Predictions (R²<0.5)
---------------------------------------
- Large scatter around 1:1 line
- Large residuals with possible patterns
- Q-Q plot may show deviation from normality
- Wide residual distribution

Test Case 3: Small Dataset (n<10)
----------------------------------
- Plots should still render
- Statistics should be computed
- May have fewer bins in histogram
- Smooth trend line may not work

Test Case 4: Large Dataset (n>100)
-----------------------------------
- Plots should render smoothly
- Statistics should be accurate
- Many bins in histogram
- Smooth trend line should work well

Test Case 5: Single Run (Comparison Placeholder)
-------------------------------------------------
- Plots 1 and 2 should render
- Plot 3 should show placeholder message
- Message: "Run multiple models for comparison"

Test Case 6: Multiple Runs (Comparison Shows Bars)
---------------------------------------------------
- All three plots should render
- Plot 3 should show colored bars
- Bars should be sorted by R² (best first)
- Color coding should be correct

Test Case 7: Extreme Outliers in Residuals
-------------------------------------------
- Outliers should be highlighted in red in plot 2
- Q-Q plot should show deviation in tails
- Histogram should show tail bins in coral color
- Trend line should show patterns

Test Case 8: No Matplotlib Available
-------------------------------------
- All plots should show placeholder
- Message: "Matplotlib not available"
- No errors should occur

Test Case 9: Invalid Data
--------------------------
- Empty arrays: Show "Empty data arrays" message
- Length mismatch: Show "Data length mismatch" message
- No errors should crash the GUI

Test Case 10: Clear Plots
--------------------------
- Calling _tab7_clear_plots() should remove all widgets
- Frames should be empty
- No errors should occur
"""


# ============================================================================
# PART 8: USAGE EXAMPLES
# ============================================================================

"""
USAGE EXAMPLES
==============

Example 1: Basic Integration
-----------------------------
# In _tab7_update_results()
def _tab7_update_results(self, results):
    # Update text display
    # ... [existing code] ...

    # Generate plots
    self._tab7_generate_plots(results)

Example 2: Manual Plot Generation
----------------------------------
# Generate plots manually with custom data
results = {
    'y_true': np.array([1, 2, 3, 4, 5]),
    'y_pred': np.array([1.1, 1.9, 3.2, 3.8, 5.1]),
    'model_name': 'PLS (n=5)',
    'r2': 0.95,
    'rmse': 0.15
}
self._tab7_generate_plots(results)

Example 3: Clearing Plots
--------------------------
# Clear all plots before new run
self._tab7_clear_plots()

Example 4: Showing Placeholder
-------------------------------
# Show custom placeholder message
self._tab7_show_plot_placeholder(
    self.tab7_plot1_frame,
    "No model loaded\nPlease load a model first"
)

Example 5: Multiple Model Runs
-------------------------------
# Track multiple runs for comparison
for model_type in ['PLS', 'Ridge', 'LASSO']:
    # Train model
    y_pred = model.predict(X_test)

    # Create results
    results = {
        'y_true': y_test,
        'y_pred': y_pred,
        'model_name': model_type,
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # Generate plots (will update comparison plot)
    self._tab7_generate_plots(results)
"""


# ============================================================================
# END OF IMPLEMENTATION
# ============================================================================
