"""
Code templates for reproducible script generation.

This module provides standalone Python code templates that can be
combined to generate complete analysis scripts for reviewers.
All templates use only standard scientific Python libraries.
"""

from .preprocessing import (
    SNV_TEMPLATE,
    SAVGOL_DERIVATIVE_TEMPLATE,
    MSC_TEMPLATE,
    get_preprocessing_template,
)
from .variable_selection import (
    VIP_TEMPLATE,
    SPA_TEMPLATE,
    UVE_TEMPLATE,
    CARS_TEMPLATE,
    get_variable_selection_template,
)
from .models import (
    get_model_template,
    get_model_imports,
)
from .validation import (
    CROSS_VALIDATION_TEMPLATE,
    METRICS_TEMPLATE,
)
from .visualization import (
    PRED_VS_ACTUAL_TEMPLATE,
    RESIDUALS_TEMPLATE,
    SPECTRA_PLOT_TEMPLATE,
)
from .header import (
    HEADER_TEMPLATE,
    DATA_LOADING_TEMPLATE,
)

__all__ = [
    # Preprocessing
    'SNV_TEMPLATE',
    'SAVGOL_DERIVATIVE_TEMPLATE',
    'MSC_TEMPLATE',
    'get_preprocessing_template',
    # Variable selection
    'VIP_TEMPLATE',
    'SPA_TEMPLATE',
    'UVE_TEMPLATE',
    'CARS_TEMPLATE',
    'get_variable_selection_template',
    # Models
    'get_model_template',
    'get_model_imports',
    # Validation
    'CROSS_VALIDATION_TEMPLATE',
    'METRICS_TEMPLATE',
    # Visualization
    'PRED_VS_ACTUAL_TEMPLATE',
    'RESIDUALS_TEMPLATE',
    'SPECTRA_PLOT_TEMPLATE',
    # Header
    'HEADER_TEMPLATE',
    'DATA_LOADING_TEMPLATE',
]
