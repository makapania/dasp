"""
Visualization utilities for ensemble analysis.

Completely rewritten for maximum readability and self-documentation.
All figures include extensive explanatory text so any user can understand them.

Functions:
- plot_regional_performance: Heatmap showing model performance by region
- plot_ensemble_weights: Bar chart + table showing model contribution weights
- plot_model_specialization_profile: Specialist vs generalist classification
- plot_prediction_comparison: Actual vs predicted scatter plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _auto_scale_figure(n_models, n_regions, base_width=14, base_height=10):
    """Calculate figure size based on content amount."""
    # Scale width with regions (min 12", max 18")
    width = max(12, min(18, base_width + 0.3 * n_regions))
    # Scale height with models (min 10", +0.6" per model beyond 5)
    height = max(10, base_height + 0.6 * max(0, n_models - 5))
    return (width, height)


def _truncate_name(name, max_len=20):
    """Truncate model name to prevent text overflow."""
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + '...'


def _add_title_block(ax, title, description_lines, y_start=0.98):
    """Add a title block with description text at the top of the figure."""
    # Main title
    ax.text(0.5, y_start, title, transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightsteelblue', alpha=0.8))

    # Description lines
    y = y_start - 0.06
    for line in description_lines:
        ax.text(0.5, y, line, transform=ax.transAxes,
                fontsize=11, ha='center', va='top', style='italic')
        y -= 0.035

    return y  # Return the y position after text


def _add_interpretation_box(ax, lines, position='bottom'):
    """Add an interpretation/legend box explaining how to read the figure."""
    if position == 'bottom':
        y_pos = 0.02
        va = 'bottom'
    else:
        y_pos = 0.98
        va = 'top'

    text = "HOW TO READ THIS FIGURE:\n" + "\n".join(f"  - {line}" for line in lines)
    ax.text(0.02, y_pos, text, transform=ax.transAxes,
            fontsize=10, ha='left', va=va, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='orange', alpha=0.9))


def _format_region_boundaries(boundaries, region_idx):
    """Format region boundary text like '(2.5 - 5.0)'."""
    if boundaries is None or len(boundaries) < 2:
        return ""
    low = boundaries[region_idx]
    high = boundaries[region_idx + 1] if region_idx + 1 < len(boundaries) else boundaries[-1]
    return f"({low:.1f} - {high:.1f})"


# =============================================================================
# MAIN VISUALIZATION FUNCTIONS
# =============================================================================

def plot_regional_performance(analyzer, y_true, predictions_dict, metric='rmse',
                               figsize=None, save_path=None):
    """
    Plot model performance across different regions of the target space.

    Creates a large, readable heatmap showing how well each model predicts
    in different target value ranges. Includes extensive explanatory text.

    Parameters
    ----------
    analyzer : RegionBasedAnalyzer
        Fitted analyzer with region definitions
    y_true : array-like
        True target values
    predictions_dict : dict
        Dictionary mapping model names to predictions
    metric : str, default='rmse'
        Metric to plot ('rmse', 'mae', or 'r2')
    figsize : tuple, optional
        Figure size (auto-calculated if None)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax
    """
    # Analyze all models
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    n_regions = analyzer.n_regions

    results = {}
    for model_name, y_pred in predictions_dict.items():
        results[model_name] = analyzer.analyze_model_performance(
            y_true, y_pred, metric=metric
        )

    # Sort models by overall performance (best first)
    sorted_models = sorted(model_names,
                          key=lambda m: results[m]['overall'],
                          reverse=(metric == 'r2'))  # Higher is better for R2

    # Auto-scale figure
    if figsize is None:
        figsize = _auto_scale_figure(n_models, n_regions)

    fig = plt.figure(figsize=figsize)

    # Create grid: title area at top, main heatmap in middle, legend at bottom
    gs = fig.add_gridspec(3, 1, height_ratios=[0.15, 0.7, 0.15], hspace=0.1)

    # === TITLE AREA ===
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')

    # Calculate region sample counts
    regions = analyzer.assign_regions(y_true)
    region_counts = [np.sum(regions == i) for i in range(n_regions)]
    total_samples = len(y_true)

    title_text = "REGIONAL PERFORMANCE ANALYSIS"
    desc_lines = [
        f"This shows how well each of the {n_models} models predicts across {n_regions} different target value ranges.",
        f"{'Lower' if metric != 'r2' else 'Higher'} {metric.upper()} = better predictions. Total samples: {total_samples}",
        f"Models are ranked from best (top) to worst (bottom) based on overall {metric.upper()}."
    ]
    _add_title_block(ax_title, title_text, desc_lines, y_start=0.9)

    # === MAIN HEATMAP ===
    ax_main = fig.add_subplot(gs[1])

    # Build the data matrix (models x regions)
    data_matrix = np.zeros((n_models, n_regions))
    for i, model_name in enumerate(sorted_models):
        data_matrix[i] = results[model_name]['by_region']

    # Handle NaN values for display
    data_matrix_display = np.nan_to_num(data_matrix, nan=0)

    # Choose colormap based on metric
    if metric == 'r2':
        cmap = 'RdYlGn'  # Green = high = good for R2
    else:
        cmap = 'RdYlGn_r'  # Green = low = good for RMSE/MAE

    # Create heatmap
    im = ax_main.imshow(data_matrix_display, aspect='auto', cmap=cmap)

    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax_main, shrink=0.8, pad=0.02)
    cbar.set_label(f'{metric.upper()} ({"higher" if metric == "r2" else "lower"} = better)',
                   fontsize=12, fontweight='bold')

    # Set tick labels
    # Y-axis: Model names with rank and overall score (truncated to prevent overflow)
    y_labels = []
    for i, model_name in enumerate(sorted_models):
        overall = results[model_name]['overall']
        short_name = _truncate_name(model_name, max_len=18)
        y_labels.append(f"{i+1}. {short_name} ({metric.upper()}: {overall:.3f})")

    ax_main.set_yticks(range(n_models))
    ax_main.set_yticklabels(y_labels, fontsize=10)

    # X-axis: Region labels with boundaries and sample counts
    x_labels = []
    boundaries = analyzer.region_boundaries
    for r in range(n_regions):
        bound_text = _format_region_boundaries(boundaries, r)
        x_labels.append(f"Region {r+1}\n{bound_text}\n(n={region_counts[r]})")

    ax_main.set_xticks(range(n_regions))
    ax_main.set_xticklabels(x_labels, fontsize=10, ha='center')
    ax_main.tick_params(axis='x', which='major', pad=5)

    # Add cell annotations with values
    font_size = max(8, min(12, 14 - n_models // 3))  # Scale font with model count
    for i in range(n_models):
        for j in range(n_regions):
            value = data_matrix[i, j]
            if np.isnan(value):
                text = "N/A"
                color = 'gray'
            else:
                text = f"{value:.3f}"
                # Choose text color based on background brightness
                norm_val = (value - np.nanmin(data_matrix)) / (np.nanmax(data_matrix) - np.nanmin(data_matrix) + 1e-10)
                if metric == 'r2':
                    color = 'white' if norm_val > 0.6 else 'black'
                else:
                    color = 'white' if norm_val < 0.4 else 'black'

            ax_main.text(j, i, text, ha='center', va='center',
                        fontsize=font_size, fontweight='bold', color=color)

    # Mark best model for each region with a star
    for j in range(n_regions):
        col_data = data_matrix[:, j]
        if metric == 'r2':
            best_idx = np.nanargmax(col_data)
        else:
            best_idx = np.nanargmin(col_data)

        # Add star marker
        ax_main.plot(j, best_idx, marker='*', markersize=15, color='gold',
                    markeredgecolor='black', markeredgewidth=1)

    ax_main.set_xlabel('Target Value Regions (with boundaries and sample counts)',
                       fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Models (ranked by overall performance)',
                       fontsize=12, fontweight='bold')

    # === INTERPRETATION BOX ===
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')

    interpretation_lines = [
        f"GREEN cells = model performs WELL in that region ({'high' if metric == 'r2' else 'low'} {metric.upper()})",
        f"RED cells = model performs POORLY in that region ({'low' if metric == 'r2' else 'high'} {metric.upper()})",
        "GOLD STAR = best performing model for that region",
        "N/A = insufficient samples in that region to calculate metric",
        f"Models are sorted by overall {metric.upper()} (best at top, worst at bottom)"
    ]
    _add_interpretation_box(ax_legend, interpretation_lines, position='top')

    # Add margins before tight_layout to prevent text clipping
    plt.subplots_adjust(left=0.22, right=0.95, top=0.95, bottom=0.08)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax_main


def plot_ensemble_weights(ensemble, figsize=None, save_path=None):
    """
    Plot ensemble weights by region using grouped bar chart and table.

    Shows how much each model contributes to the final ensemble prediction
    in each region. Includes auto-generated insights about specialists/generalists.

    Parameters
    ----------
    ensemble : RegionAwareWeightedEnsemble or MixtureOfExpertsEnsemble
        Fitted ensemble model with weights
    figsize : tuple, optional
        Figure size (auto-calculated if None)
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    # Get weights
    if hasattr(ensemble, 'regional_weights_'):
        weights = ensemble.regional_weights_
    elif hasattr(ensemble, 'expert_weights_'):
        weights = ensemble.expert_weights_
    else:
        raise ValueError("Ensemble does not have weight information")

    model_names = ensemble.model_names
    n_models, n_regions = weights.shape

    # Auto-scale figure
    if figsize is None:
        figsize = _auto_scale_figure(n_models, n_regions, base_width=16, base_height=12)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.12, 0.58, 0.30],
                          width_ratios=[0.6, 0.4], hspace=0.15, wspace=0.1)

    # === TITLE AREA (spans both columns) ===
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')

    title_text = "ENSEMBLE WEIGHT DISTRIBUTION"
    desc_lines = [
        f"These weights show how much each of the {n_models} models contributes to the final prediction.",
        "Higher weight = more influence on the ensemble output. Weights sum to 1.0 in each region.",
        "Weights vary by region - some models are 'specialists' that excel in specific ranges."
    ]
    _add_title_block(ax_title, title_text, desc_lines, y_start=0.85)

    # === GROUPED BAR CHART (left side) ===
    ax_bars = fig.add_subplot(gs[1, 0])

    # Create grouped bar positions
    x = np.arange(n_regions)
    bar_width = 0.8 / n_models

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))

    # Plot bars for each model (truncate names for legend readability)
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        offset = (i - n_models/2 + 0.5) * bar_width
        short_name = _truncate_name(model_name, max_len=15)
        bars = ax_bars.bar(x + offset, weights[i], bar_width,
                          label=short_name, color=color, edgecolor='black', linewidth=0.5)

        # Add value labels on bars (if not too many models)
        if n_models <= 8:
            for bar, val in zip(bars, weights[i]):
                if val > 0.05:  # Only label if visible
                    ax_bars.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

    ax_bars.set_xlabel('Prediction Region', fontsize=12, fontweight='bold')
    ax_bars.set_ylabel('Weight (contribution to ensemble)', fontsize=12, fontweight='bold')
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f'Region {i+1}' for i in range(n_regions)], fontsize=11)
    ax_bars.set_ylim(0, min(1.1, weights.max() * 1.3))
    # Place legend below the plot to avoid overlapping bar data
    ax_bars.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   fontsize=8, ncol=min(4, n_models), framealpha=0.9)
    ax_bars.grid(axis='y', alpha=0.3)
    ax_bars.set_title('Model Weights by Region', fontsize=13, fontweight='bold', pad=10)

    # === WEIGHT TABLE (right side) ===
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis('off')

    # Create table data
    col_labels = ['Model'] + [f'R{i+1}' for i in range(n_regions)] + ['Avg', 'Std']
    table_data = []

    for i, model_name in enumerate(model_names):
        row = [model_name[:15]]  # Truncate long names
        row.extend([f'{w:.3f}' for w in weights[i]])
        row.append(f'{np.mean(weights[i]):.3f}')
        row.append(f'{np.std(weights[i]):.3f}')
        table_data.append(row)

    # Create the table
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                          loc='center', cellLoc='center',
                          colWidths=[0.25] + [0.1]*n_regions + [0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header row
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('lightsteelblue')
        table[(0, j)].set_text_props(fontweight='bold')

    ax_table.set_title('Exact Weight Values', fontsize=13, fontweight='bold', pad=10)

    # === KEY INSIGHTS (bottom, spans both columns) ===
    ax_insights = fig.add_subplot(gs[2, :])
    ax_insights.axis('off')

    # Calculate insights
    insights = ["KEY INSIGHTS:"]

    # Find highest weight per region
    for r in range(n_regions):
        best_idx = np.argmax(weights[:, r])
        best_weight = weights[best_idx, r]
        insights.append(f"  - Region {r+1}: {model_names[best_idx]} dominates (weight={best_weight:.3f})")

    insights.append("")

    # Find specialists (high variance) and generalists (low variance)
    variances = np.std(weights, axis=1)
    specialist_idx = np.argmax(variances)
    generalist_idx = np.argmin(variances)

    insights.append(f"  - MOST SPECIALIZED model: {model_names[specialist_idx]} (weight std={variances[specialist_idx]:.3f})")
    insights.append(f"    This model's contribution varies most across regions.")
    insights.append(f"  - MOST GENERALIST model: {model_names[generalist_idx]} (weight std={variances[generalist_idx]:.3f})")
    insights.append(f"    This model contributes consistently across all regions.")

    insight_text = "\n".join(insights)
    ax_insights.text(0.02, 0.95, insight_text, transform=ax_insights.transAxes,
                    fontsize=11, ha='left', va='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                             edgecolor='darkgreen', alpha=0.8))

    # Add margins to accommodate legend below plot
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax_bars, ax_table)


def plot_model_specialization_profile(ensemble, figsize=None, save_path=None):
    """
    Create a detailed profile of each model's specialization.

    Shows which models are SPECIALISTS (excel in specific regions) vs
    GENERALISTS (perform consistently across all regions).

    Parameters
    ----------
    ensemble : RegionAwareWeightedEnsemble
        Fitted ensemble with get_model_profiles() method
    figsize : tuple, optional
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    if not hasattr(ensemble, 'get_model_profiles'):
        raise ValueError("Ensemble does not support model profiling")

    profiles = ensemble.get_model_profiles()
    model_names = list(profiles.keys())
    n_models = len(model_names)
    n_regions = len(profiles[model_names[0]]['weights'])

    # Auto-scale figure
    if figsize is None:
        # Height: 3" for header + 1" per model (min 8", max 20")
        height = max(8, min(20, 4 + 1.2 * n_models))
        figsize = (14, height)

    fig = plt.figure(figsize=figsize)

    # Grid: title, summary table, then one row per model
    n_rows = 2 + n_models
    height_ratios = [0.08, 0.15] + [1.0/n_models] * n_models
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

    # === TITLE ===
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')

    title_text = "MODEL SPECIALIZATION PROFILES"
    ax_title.text(0.5, 0.5, title_text, transform=ax_title.transAxes,
                 fontsize=16, fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightsteelblue', alpha=0.8))

    # === SUMMARY TABLE: Specialists vs Generalists ===
    ax_summary = fig.add_subplot(gs[1])
    ax_summary.axis('off')

    # Sort models by specialization (variance)
    model_variance = [(name, profiles[name]['weight_variance']) for name in model_names]
    sorted_by_variance = sorted(model_variance, key=lambda x: x[1], reverse=True)

    # Split into specialists (top half) and generalists (bottom half)
    mid = len(sorted_by_variance) // 2
    specialists = sorted_by_variance[:max(1, mid)]
    generalists = sorted_by_variance[mid:]

    summary_text = "CLASSIFICATION SUMMARY\n\n"
    summary_text += "SPECIALISTS (high variance):          GENERALISTS (low variance):\n"
    summary_text += "-" * 60 + "\n"

    max_rows = max(len(specialists), len(generalists))
    for i in range(max_rows):
        spec_text = ""
        gen_text = ""
        if i < len(specialists):
            name, var = specialists[i]
            short_name = _truncate_name(name, max_len=12)
            spec_text = f"{i+1}. {short_name:<12} (var={var:.3f})"
        if i < len(generalists):
            name, var = generalists[i]
            short_name = _truncate_name(name, max_len=12)
            gen_text = f"{i+1}. {short_name:<12} (var={var:.3f})"
        summary_text += f"{spec_text:<30} {gen_text}\n"

    ax_summary.text(0.02, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, ha='left', va='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # === INDIVIDUAL MODEL PROFILES ===
    # Calculate global max weight for consistent Y-axis
    all_weights = [profiles[name]['weights'] for name in model_names]
    max_weight = max(np.max(w) for w in all_weights)

    # Sort models by variance (specialists first)
    sorted_names = [name for name, _ in sorted_by_variance]

    for idx, model_name in enumerate(sorted_names):
        ax = fig.add_subplot(gs[2 + idx])

        profile = profiles[model_name]
        weights = profile['weights']
        best_regions = profile['best_regions']
        spec_type = profile['specialization'].upper()
        variance = profile['weight_variance']

        # Bar colors: green for best regions, gray for others
        colors = ['forestgreen' if i in best_regions else 'lightgray'
                 for i in range(n_regions)]

        # Create bars
        x = np.arange(n_regions)
        bars = ax.bar(x, weights, color=colors, edgecolor='black', linewidth=0.5)

        # Add average line
        avg_weight = np.mean(weights)
        ax.axhline(avg_weight, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_weight:.3f}')

        # Add value labels on bars
        for i, (bar, w) in enumerate(zip(bars, weights)):
            ax.text(bar.get_x() + bar.get_width()/2, w + 0.02 * max_weight,
                   f'{w:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Configure axes
        ax.set_xlim(-0.5, n_regions - 0.5)
        ax.set_ylim(0, max_weight * 1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f'R{i+1}' for i in range(n_regions)], fontsize=10)

        # Rank indicator
        rank = idx + 1

        # Model label with full info (truncate name to prevent overflow)
        short_name = _truncate_name(model_name, max_len=20)
        best_str = ','.join([f'R{r+1}' for r in best_regions]) if len(best_regions) > 0 else 'None'
        # Split into two lines to prevent horizontal overflow
        title = f"#{rank} {short_name} ({spec_type})\nBest: {best_str} | Var: {variance:.3f}"
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')

        # Only show y-label for first subplot to reduce clutter
        if idx == 0:
            ax.set_ylabel('Weight', fontsize=10)

        # Grid
        ax.grid(axis='y', alpha=0.3)

        # Legend only on first
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    # Add x-label to bottom plot only
    ax.set_xlabel('Prediction Region', fontsize=12, fontweight='bold')

    # Add margins to prevent text clipping
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, None


def plot_prediction_comparison(y_true, predictions_dict, ensemble_pred=None,
                                figsize=(16, 10), save_path=None):
    """
    Compare predictions from individual models and ensemble.

    Creates scatter plots showing actual vs predicted values, with
    clear comparison between ensemble and best individual model.

    Parameters
    ----------
    y_true : array-like
        True values
    predictions_dict : dict
        Dictionary mapping model names to predictions
    ensemble_pred : array-like, optional
        Ensemble predictions
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.1, 0.6, 0.3], hspace=0.2, wspace=0.15)

    # === TITLE ===
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')

    title_text = "PREDICTION COMPARISON: Individual Models vs Ensemble"
    desc_lines = [
        "Each point = one sample. Points closer to the diagonal line = better predictions.",
        f"Comparing {n_models} individual models" + (f" and their ensemble." if ensemble_pred is not None else ".")
    ]
    _add_title_block(ax_title, title_text, desc_lines, y_start=0.8)

    # Calculate RMSE for all models
    rmse_dict = {}
    for name, pred in predictions_dict.items():
        rmse_dict[name] = np.sqrt(np.mean((y_true - pred) ** 2))

    # Sort by RMSE (best first)
    sorted_models = sorted(rmse_dict.items(), key=lambda x: x[1])

    # Value range for diagonal line
    all_preds = list(predictions_dict.values())
    if ensemble_pred is not None:
        all_preds.append(ensemble_pred)
    min_val = min(y_true.min(), min(p.min() for p in all_preds))
    max_val = max(y_true.max(), max(p.max() for p in all_preds))

    # === LEFT PLOT: Individual models ===
    ax_left = fig.add_subplot(gs[1, 0])

    # Limit to top 6 models for readability, group rest as "Other"
    max_show = 6
    colors = plt.cm.Set2(np.linspace(0, 1, min(n_models, max_show)))

    for i, (name, rmse) in enumerate(sorted_models[:max_show]):
        pred = predictions_dict[name]
        short_name = _truncate_name(name, max_len=15)
        ax_left.scatter(y_true, pred, alpha=0.5, s=30,
                       label=f'{short_name} (RMSE={rmse:.3f})',
                       color=colors[i], edgecolors='white', linewidth=0.3)

    if n_models > max_show:
        # Plot remaining models in gray
        for name, rmse in sorted_models[max_show:]:
            pred = predictions_dict[name]
            ax_left.scatter(y_true, pred, alpha=0.2, s=15, color='gray')
        ax_left.scatter([], [], alpha=0.2, s=15, color='gray',
                       label=f'Other {n_models - max_show} models')

    # Diagonal line
    ax_left.plot([min_val, max_val], [min_val, max_val], 'k--',
                linewidth=2, label='Perfect prediction')

    ax_left.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax_left.set_title('Individual Model Predictions', fontsize=13, fontweight='bold')
    # Place legend outside plot area to avoid obscuring data
    ax_left.legend(loc='upper left', bbox_to_anchor=(0, -0.12), fontsize=8, ncol=2, framealpha=0.9)
    ax_left.grid(alpha=0.3)

    # === RIGHT PLOT: Ensemble vs Best ===
    ax_right = fig.add_subplot(gs[1, 1])

    if ensemble_pred is not None:
        # Ensemble
        ensemble_rmse = np.sqrt(np.mean((y_true - ensemble_pred) ** 2))
        ax_right.scatter(y_true, ensemble_pred, alpha=0.6, s=40,
                        label=f'Ensemble (RMSE={ensemble_rmse:.3f})',
                        color='red', edgecolors='darkred', linewidth=0.5)

        # Best individual (truncate name for legend)
        best_name, best_rmse = sorted_models[0]
        best_pred = predictions_dict[best_name]
        short_best = _truncate_name(best_name, max_len=15)
        ax_right.scatter(y_true, best_pred, alpha=0.4, s=30,
                        label=f'Best: {short_best} (RMSE={best_rmse:.3f})',
                        color='blue', edgecolors='darkblue', linewidth=0.5)

        # Diagonal
        ax_right.plot([min_val, max_val], [min_val, max_val], 'k--',
                     linewidth=2, label='Perfect prediction')

        ax_right.set_title('Ensemble vs Best Individual', fontsize=13, fontweight='bold')
    else:
        # Just show best model (truncate name for legend)
        best_name, best_rmse = sorted_models[0]
        best_pred = predictions_dict[best_name]
        short_best = _truncate_name(best_name, max_len=15)
        ax_right.scatter(y_true, best_pred, alpha=0.6, s=40,
                        label=f'{short_best} (RMSE={best_rmse:.3f})',
                        color='blue', edgecolors='darkblue', linewidth=0.5)
        ax_right.plot([min_val, max_val], [min_val, max_val], 'k--',
                     linewidth=2, label='Perfect prediction')
        ax_right.set_title('Best Model', fontsize=13, fontweight='bold')

    ax_right.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax_right.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    # Place legend below plot to avoid obscuring data
    ax_right.legend(loc='upper left', bbox_to_anchor=(0, -0.12), fontsize=9, framealpha=0.9)
    ax_right.grid(alpha=0.3)

    # === SUMMARY STATS ===
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')

    summary_lines = ["SUMMARY STATISTICS:"]
    best_model_name = _truncate_name(sorted_models[0][0], max_len=25)
    worst_model_name = _truncate_name(sorted_models[-1][0], max_len=25)
    summary_lines.append(f"  - Best individual model: {best_model_name} (RMSE = {sorted_models[0][1]:.4f})")
    summary_lines.append(f"  - Worst individual model: {worst_model_name} (RMSE = {sorted_models[-1][1]:.4f})")

    if ensemble_pred is not None:
        improvement = (best_rmse - ensemble_rmse) / best_rmse * 100
        if improvement > 0:
            summary_lines.append(f"  - Ensemble RMSE = {ensemble_rmse:.4f} ({improvement:.1f}% BETTER than best individual)")
        else:
            summary_lines.append(f"  - Ensemble RMSE = {ensemble_rmse:.4f} ({-improvement:.1f}% worse than best individual)")

    summary_lines.append(f"  - Total samples: {len(y_true)}")
    summary_lines.append(f"  - Target range: {min_val:.2f} to {max_val:.2f}")

    summary_text = "\n".join(summary_lines)
    ax_summary.text(0.02, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, ha='left', va='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                            edgecolor='blue', alpha=0.8))

    # Add margins to accommodate legends below plots
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax_left, ax_right)


def create_ensemble_report(analyzer, y_true, predictions_dict, ensemble_pred=None,
                           ensemble_type='Region-Aware Weighted', save_dir=None):
    """
    Create a comprehensive visual report of ensemble performance.

    Generates all visualization figures and optionally saves them.

    Parameters
    ----------
    analyzer : RegionBasedAnalyzer
    y_true : array-like
    predictions_dict : dict
    ensemble_pred : array-like, optional
    ensemble_type : str
    save_dir : str, optional
        Directory to save all plots

    Returns
    -------
    dict of figures
    """
    figures = {}

    # Regional performance
    fig1, _ = plot_regional_performance(
        analyzer, y_true, predictions_dict,
        save_path=f"{save_dir}/regional_performance.png" if save_dir else None
    )
    figures['regional_performance'] = fig1

    # Prediction comparison
    if ensemble_pred is not None:
        fig2, _ = plot_prediction_comparison(
            y_true, predictions_dict, ensemble_pred,
            save_path=f"{save_dir}/prediction_comparison.png" if save_dir else None
        )
        figures['prediction_comparison'] = fig2

    return figures
