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
ax.scatter(y, y_pred_cv, alpha=0.6, edgecolors='k', linewidths=0.5)

# 1:1 line
lims = [min(y.min(), y_pred_cv.min()), max(y.max(), y_pred_cv.max())]
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

residuals = y - y_pred_cv

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
n_plot = min(50, X.shape[0])
indices_to_plot = np.linspace(0, X.shape[0]-1, n_plot, dtype=int)

for i in indices_to_plot:
    ax.plot(wavelengths, X[i, :], alpha=0.5, lw=0.5)

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Intensity', fontsize=12)
ax.set_title(f'Raw Spectra (n={n_plot} of {X.shape[0]})', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectra_plot.png', dpi=150, bbox_inches='tight')
plt.show()
'''

VARIABLE_IMPORTANCE_TEMPLATE = '''
# =============================================================================
# VISUALIZATION: Variable Importance
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 5))

# Create importance array (highlighting selected variables)
importance = np.zeros(len(wavelengths))
importance[selected_indices] = 1

ax.fill_between(wavelengths, 0, importance, alpha=0.3, color='blue', label='Selected')
ax.plot(wavelengths, importance, 'b-', lw=0.5)

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Selected (1) / Not Selected (0)', fontsize=12)
ax.set_title(f'Selected Variables ({len(selected_indices)} of {len(wavelengths)})', fontsize=14)
ax.set_ylim(-0.1, 1.1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('selected_variables.png', dpi=150, bbox_inches='tight')
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
    else:
        code_parts.append(CONFUSION_MATRIX_TEMPLATE)

    if include_spectra:
        code_parts.append(SPECTRA_PLOT_TEMPLATE)

    if include_variable_importance:
        code_parts.append(VARIABLE_IMPORTANCE_TEMPLATE)

    return '\n'.join(code_parts)
