"""
Central model registry to avoid hardcoded model lists throughout the codebase.

This module provides a single source of truth for:
- All supported regression models
- All supported classification models
- Models that support feature importance extraction
- Models that support subset analysis
"""

# Regression models (ordered by complexity/usage)
REGRESSION_MODELS = [
    'PLS',           # Partial Least Squares
    'Ridge',         # L2-regularized linear regression
    'Lasso',         # L1-regularized linear regression
    'ElasticNet',    # L1+L2 regularized linear regression
    'RandomForest',  # Ensemble of decision trees
    'MLP',           # Multi-Layer Perceptron (neural network)
    'NeuralBoosted', # Custom gradient boosting with neural networks
    'SVR',           # Support Vector Regression
    'XGBoost',       # Extreme Gradient Boosting
    'LightGBM',      # Light Gradient Boosting Machine
    'CatBoost',      # Categorical Boosting
]

# Classification models (ordered by complexity/usage)
CLASSIFICATION_MODELS = [
    'PLS-DA',        # Partial Least Squares Discriminant Analysis
    'PLS',           # PLS can also be used for classification
    'RandomForest',  # Ensemble of decision trees
    'MLP',           # Multi-Layer Perceptron
    'SVM',           # Support Vector Machine
    'XGBoost',       # Extreme Gradient Boosting
    'LightGBM',      # Light Gradient Boosting Machine
    'CatBoost',      # Categorical Boosting
]

# All unique models (union of regression and classification)
ALL_MODELS = sorted(list(set(REGRESSION_MODELS + CLASSIFICATION_MODELS)))

# Models that support feature importance extraction
# (Used for wavelength selection and subset analysis)
MODELS_WITH_FEATURE_IMPORTANCE = [
    'PLS',
    'PLS-DA',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'RandomForest',
    'MLP',
    'NeuralBoosted',
    'SVR',
    'XGBoost',
    'LightGBM',
    'CatBoost',
]

# Models that support variable subset analysis
# (Subset analysis requires feature importance)
MODELS_WITH_SUBSET_SUPPORT = MODELS_WITH_FEATURE_IMPORTANCE


def get_supported_models(task_type='regression'):
    """
    Get list of supported models for a given task type.

    Parameters
    ----------
    task_type : str, default='regression'
        Either 'regression' or 'classification'

    Returns
    -------
    list of str
        List of model names supported for the given task

    Examples
    --------
    >>> get_supported_models('regression')
    ['PLS', 'Ridge', 'Lasso', 'ElasticNet', ...]

    >>> get_supported_models('classification')
    ['PLS-DA', 'PLS', 'RandomForest', 'MLP', ...]
    """
    if task_type == 'regression':
        return REGRESSION_MODELS.copy()
    elif task_type == 'classification':
        return CLASSIFICATION_MODELS.copy()
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'regression' or 'classification'")


def supports_feature_importance(model_name):
    """
    Check if a model supports feature importance extraction.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    bool
        True if model supports feature importance, False otherwise

    Examples
    --------
    >>> supports_feature_importance('XGBoost')
    True

    >>> supports_feature_importance('UnknownModel')
    False
    """
    return model_name in MODELS_WITH_FEATURE_IMPORTANCE


def supports_subset_analysis(model_name):
    """
    Check if a model supports variable subset analysis.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    bool
        True if model supports subset analysis, False otherwise

    Examples
    --------
    >>> supports_subset_analysis('SVR')
    True

    >>> supports_subset_analysis('UnknownModel')
    False
    """
    return model_name in MODELS_WITH_SUBSET_SUPPORT


def is_valid_model(model_name, task_type=None):
    """
    Check if a model name is valid (optionally for a specific task type).

    Parameters
    ----------
    model_name : str
        Name of the model to validate
    task_type : str, optional
        If provided, check if model is valid for this task type
        Must be 'regression' or 'classification'

    Returns
    -------
    bool
        True if model is valid, False otherwise

    Examples
    --------
    >>> is_valid_model('XGBoost')
    True

    >>> is_valid_model('XGBoost', 'regression')
    True

    >>> is_valid_model('UnknownModel')
    False
    """
    if task_type is None:
        return model_name in ALL_MODELS
    else:
        supported = get_supported_models(task_type)
        return model_name in supported
