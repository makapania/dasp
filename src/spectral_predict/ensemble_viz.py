"""
Visualization utilities for ensemble analysis.

Functions to visualize:
- Regional performance of individual models
- Model specialization patterns
- Ensemble weight distributions
- Prediction comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_regional_performance(analyzer, y_true, predictions_dict, metric='rmse',
                               figsize=(14, 8), save_path=None):
    """
    Plot model performance across different regions of the target space.

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
    figsize : tuple, default=(14, 8)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, axes
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])  # Regional performance
    ax2 = fig.add_subplot(gs[1, 0])   # Overall performance
    ax3 = fig.add_subplot(gs[1, 1])   # Specialization scores

    # Analyze all models
    results = {}
    for model_name, y_pred in predictions_dict.items():
        results[model_name] = analyzer.analyze_model_performance(
            y_true, y_pred, metric=metric
        )

    # Plot 1: Regional performance heatmap
    model_names = list(results.keys())
    n_models = len(model_names)
    n_regions = analyzer.n_regions

    regional_matrix = np.zeros((n_models, n_regions))
    for i, model_name in enumerate(model_names):
        regional_matrix[i] = results[model_name]['by_region']

    # Create heatmap
    im = ax1.imshow(regional_matrix, aspect='auto', cmap='RdYlGn_r')
    ax1.set_yticks(range(n_models))
    ax1.set_yticklabels(model_names)
    ax1.set_xticks(range(n_regions))
    ax1.set_xticklabels([f"R{i}" for i in range(n_regions)])
    ax1.set_xlabel('Prediction Region')
    ax1.set_ylabel('Model')
    ax1.set_title(f'Regional Performance ({metric.upper()}) - Lower is Better')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(metric.upper())

    # Add text annotations
    for i in range(n_models):
        for j in range(n_regions):
            value = regional_matrix[i, j]
            if not np.isnan(value):
                text = ax1.text(j, i, f'{value:.3f}',
                               ha="center", va="center", color="black",
                               fontsize=8)

    # Plot 2: Overall performance comparison
    overall_scores = [results[name]['overall'] for name in model_names]
    bars = ax2.barh(range(n_models), overall_scores)

    # Color bars by performance (best = green, worst = red)
    if metric in ['rmse', 'mae']:
        norm_scores = (overall_scores - np.min(overall_scores)) / (np.max(overall_scores) - np.min(overall_scores) + 1e-10)
        colors = plt.cm.RdYlGn_r(norm_scores)
    else:  # r2
        norm_scores = (overall_scores - np.min(overall_scores)) / (np.max(overall_scores) - np.min(overall_scores) + 1e-10)
        colors = plt.cm.RdYlGn(norm_scores)

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel(f'{metric.upper()}')
    ax2.set_title('Overall Performance')
    ax2.grid(axis='x', alpha=0.3)

    # Plot 3: Specialization scores
    spec_scores = [results[name]['specialization_score'] for name in model_names]
    bars = ax3.barh(range(n_models), spec_scores)

    # Color by specialization (high = specialist, low = generalist)
    norm_spec = (spec_scores - np.min(spec_scores)) / (np.max(spec_scores) - np.min(spec_scores) + 1e-10)
    colors = plt.cm.coolwarm(norm_spec)

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax3.set_yticks(range(n_models))
    ax3.set_yticklabels(model_names)
    ax3.set_xlabel('Specialization Score')
    ax3.set_title('Specialist vs Generalist')
    ax3.grid(axis='x', alpha=0.3)

    # Add legend
    ax3.text(0.95, 0.05, 'High score = Specialist\nLow score = Generalist',
             transform=ax3.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2, ax3)


def plot_ensemble_weights(ensemble, figsize=(12, 6), save_path=None):
    """
    Plot ensemble weights by region.

    Parameters
    ----------
    ensemble : RegionAwareWeightedEnsemble or MixtureOfExpertsEnsemble
        Fitted ensemble model
    figsize : tuple
    save_path : str, optional
    """
    if hasattr(ensemble, 'regional_weights_'):
        weights = ensemble.regional_weights_
    elif hasattr(ensemble, 'expert_weights_'):
        weights = ensemble.expert_weights_
    else:
        raise ValueError("Ensemble does not have weight information")

    model_names = ensemble.model_names
    n_models, n_regions = weights.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Stacked area chart of weights
    x = np.arange(n_regions)
    cumulative = np.zeros(n_regions)

    for i, model_name in enumerate(model_names):
        ax1.fill_between(x, cumulative, cumulative + weights[i],
                         label=model_name, alpha=0.7)
        cumulative += weights[i]

    ax1.set_xlabel('Region')
    ax1.set_ylabel('Weight')
    ax1.set_title('Ensemble Weights by Region (Stacked)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(alpha=0.3)

    # Plot 2: Heatmap of weights
    im = ax2.imshow(weights, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(model_names)
    ax2.set_xticks(range(n_regions))
    ax2.set_xticklabels([f"R{i}" for i in range(n_regions)])
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Model')
    ax2.set_title('Ensemble Weight Heatmap')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Weight')

    # Add text annotations
    for i in range(n_models):
        for j in range(n_regions):
            text = ax2.text(j, i, f'{weights[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if weights[i, j] > 0.5 else "black",
                           fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2)


def plot_model_specialization_profile(ensemble, save_path=None):
    """
    Create a detailed profile of each model's specialization.

    Parameters
    ----------
    ensemble : RegionAwareWeightedEnsemble
        Fitted ensemble with get_model_profiles() method
    save_path : str, optional
    """
    if not hasattr(ensemble, 'get_model_profiles'):
        raise ValueError("Ensemble does not support model profiling")

    profiles = ensemble.get_model_profiles()
    model_names = list(profiles.keys())
    n_models = len(model_names)
    n_regions = len(profiles[model_names[0]]['weights'])

    fig, axes = plt.subplots(n_models, 1, figsize=(10, 2*n_models), sharex=True)

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, profile) in enumerate(profiles.items()):
        ax = axes[idx]

        weights = profile['weights']
        best_regions = profile['best_regions']
        spec_type = profile['specialization']

        # Plot weights as bars
        colors = ['green' if i in best_regions else 'gray' for i in range(n_regions)]
        ax.bar(range(n_regions), weights, color=colors, alpha=0.7)

        # Add horizontal line for average weight
        avg_weight = np.mean(weights)
        ax.axhline(avg_weight, color='red', linestyle='--', alpha=0.5, label='Average')

        ax.set_ylabel('Weight')
        ax.set_title(f'{model_name} ({spec_type.upper()})')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Highlight best regions
        for region in best_regions:
            ax.axvspan(region - 0.4, region + 0.4, alpha=0.2, color='green')

    axes[-1].set_xlabel('Region')
    axes[-1].set_xticks(range(n_regions))
    axes[-1].set_xticklabels([f"R{i}" for i in range(n_regions)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_prediction_comparison(y_true, predictions_dict, ensemble_pred=None,
                                figsize=(14, 6), save_path=None):
    """
    Compare predictions from individual models and ensemble.

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
    """
    fig, axes = plt.subplots(1, 2 if ensemble_pred is not None else 1,
                             figsize=figsize)

    if ensemble_pred is None:
        axes = [axes]

    # Plot 1: All models
    ax = axes[0]

    for model_name, y_pred in predictions_dict.items():
        ax.scatter(y_true, y_pred, alpha=0.3, s=20, label=model_name)

    # Add perfect prediction line
    min_val = min(y_true.min(), min([p.min() for p in predictions_dict.values()]))
    max_val = max(y_true.max(), max([p.max() for p in predictions_dict.values()]))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Individual Model Predictions')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Ensemble comparison
    if ensemble_pred is not None:
        ax = axes[1]

        # Plot ensemble
        ax.scatter(y_true, ensemble_pred, alpha=0.5, s=30, label='Ensemble', color='red')

        # Plot best individual model
        best_model = min(predictions_dict.items(),
                        key=lambda x: np.sqrt(np.mean((y_true - x[1])**2)))
        ax.scatter(y_true, best_model[1], alpha=0.3, s=20,
                  label=f'Best Individual ({best_model[0]})', color='blue')

        # Perfect prediction line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')

        # Compute metrics
        ensemble_rmse = np.sqrt(np.mean((y_true - ensemble_pred)**2))
        best_rmse = np.sqrt(np.mean((y_true - best_model[1])**2))

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Ensemble vs Best Model\nEnsemble RMSE: {ensemble_rmse:.4f} | Best RMSE: {best_rmse:.4f}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def create_ensemble_report(analyzer, y_true, predictions_dict, ensemble_pred=None,
                           ensemble_type='Region-Aware Weighted', save_dir=None):
    """
    Create a comprehensive visual report of ensemble performance.

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
