"""
Visualization templates for generated scripts.
"""

VISUALIZATION_IMPORTS = '''
import matplotlib.pyplot as plt
'''

PRED_VS_ACTUAL_TEMPLATE = '''
# =============================================================================
# VISUALIZATION: Predicted vs Actual
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot
ax.scatter(all_y_true_arr, y_pred_cv, alpha=0.6, edgecolors='k', linewidths=0.5)

# 1:1 line
lims = [min(all_y_true_arr.min(), y_pred_cv.min()), max(all_y_true_arr.max(), y_pred_cv.max())]
ax.plot(lims, lims, 'r--', lw=2, label='1:1 line')

# Labels and title
ax.set_xlabel('Actual Values', fontsize=12)
ax.set_ylabel('Predicted Values', fontsize=12)
ax.set_title(f'Predicted vs Actual\\nRMSE={{rmse:.4f}}, R²={{r2:.4f}}', fontsize=14)
ax.legend()

# Equal aspect ratio
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()
'''

RESIDUALS_TEMPLATE = '''
# =============================================================================
# VISUALIZATION: Residual Plots
# =============================================================================

residuals = all_y_true_arr - y_pred_cv

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs Predicted
ax1 = axes[0]
ax1.scatter(y_pred_cv, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Predicted Values', fontsize=12)
ax1.set_ylabel('Residuals', fontsize=12)
ax1.set_title('Residuals vs Predicted', fontsize=14)
ax1.grid(True, alpha=0.3)

# Residual histogram
ax2 = axes[1]
ax2.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Residual Value', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Residual Distribution', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=150, bbox_inches='tight')
plt.show()
'''

SPECTRA_PLOT_TEMPLATE = '''
# =============================================================================
# VISUALIZATION: Spectra Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Plot a subset of spectra (to avoid clutter)
X_plot = X_processed if 'X_processed' in locals() else X
n_plot = min(50, X_plot.shape[0])
indices_to_plot = np.linspace(0, X_plot.shape[0]-1, n_plot, dtype=int)

# Use wavelengths only if they match the plotted data
if 'wavelengths' in locals() and len(wavelengths) == X_plot.shape[1]:
    x_axis = wavelengths
else:
    x_axis = np.arange(X_plot.shape[1])

# Ensure spectra are plotted in ascending wavelength order
if hasattr(x_axis, '__len__') and len(x_axis) == X_plot.shape[1]:
    sort_idx = np.argsort(x_axis)
    x_axis = np.array(x_axis)[sort_idx]
    X_plot = X_plot[:, sort_idx]

for i in indices_to_plot:
    ax.plot(x_axis, X_plot[i, :], alpha=0.5, lw=0.5)

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Intensity', fontsize=12)
title_tag = 'Processed Spectra' if 'X_processed' in locals() else 'Raw Spectra'
ax.set_title(f'{title_tag} (n={n_plot} of {X_plot.shape[0]})', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectra_plot.png', dpi=150, bbox_inches='tight')
plt.show()
'''


CONFUSION_MATRIX_TEMPLATE = '''
# =============================================================================
# VISUALIZATION: Confusion Matrix (Classification)
# =============================================================================

from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y, y_pred_cv, ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix', fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
'''

ONE_CLASS_SCORE_DISTRIBUTION_TEMPLATE = '''
# =============================================================================
# VISUALIZATION: Decision Score Distribution (One-Class)
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_cv[all_y_true_arr == 1],
        bins=30, alpha=0.6, label='Inlier', color='steelblue', edgecolor='k')
ax.hist(y_pred_cv[all_y_true_arr == -1],
        bins=30, alpha=0.6, label='Outlier', color='coral', edgecolor='k')
ax.axvline(x=0, color='red', linestyle='--', lw=2, label='Decision boundary')
ax.set_xlabel('Decision Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('One-Class Decision Score Distribution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('one_class_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
'''

ONE_CLASS_CONFUSION_TEMPLATE = '''
# =============================================================================
# VISUALIZATION: Confusion Matrix (One-Class)
# =============================================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(all_y_true_arr, y_pred_cv, labels=[1, -1])
disp = ConfusionMatrixDisplay(cm, display_labels=['Inlier (+1)', 'Outlier (-1)'])
disp.plot(ax=ax, cmap='Blues')
ax.set_title('One-Class Confusion Matrix', fontsize=14)

plt.tight_layout()
plt.savefig('one_class_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
'''


def get_visualization_code(task_type: str, include_spectra: bool = False,
                          include_variable_importance: bool = False) -> str:
    """
    Get visualization code based on options.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'
    include_spectra : bool
        Include spectra plot
    include_variable_importance : bool
        Include variable importance plot

    Returns
    -------
    str
        Visualization code
    """
    code_parts = [VISUALIZATION_IMPORTS]

    if task_type == 'regression':
        code_parts.append(PRED_VS_ACTUAL_TEMPLATE)
        code_parts.append(RESIDUALS_TEMPLATE)
    elif task_type == 'one_class':
        code_parts.append(ONE_CLASS_SCORE_DISTRIBUTION_TEMPLATE)
        code_parts.append(ONE_CLASS_CONFUSION_TEMPLATE)
    else:
        code_parts.append(CONFUSION_MATRIX_TEMPLATE)

    if include_spectra:
        code_parts.append(SPECTRA_PLOT_TEMPLATE)

    # Variable importance plot removed for readability in exported notebooks

    return '\n'.join(code_parts)
