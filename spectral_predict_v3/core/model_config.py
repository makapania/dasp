"""
Model configuration and tiered defaults for spectral analysis (v3 standalone).

Forked from v1 - tier definitions, hyperparameter grids, and preprocessing defaults.
"""

from typing import List, Dict, Any

# =============================================================================
# MODEL TIERS - Regression
# =============================================================================

MODEL_TIERS = {
    'quick': {
        'description': 'Minimal set for rapid testing',
        'models': ['PLS', 'Ridge', 'ElasticNet'],
        'recommended_for': 'Quick tests, preliminary analysis, daily QC'
    },

    'standard': {
        'description': 'Fast & reliable core models',
        'models': ['PLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'LightGBM'],
        'recommended_for': 'Most users, daily analysis, routine work'
    },

    'comprehensive': {
        'description': 'Advanced analysis with gradient boosting and neural methods',
        'models': ['PLS', 'Ridge', 'ElasticNet', 'RandomForest', 'LightGBM',
                   'XGBoost', 'CatBoost', 'NeuralBoosted', 'MLP'],
        'recommended_for': 'Thorough analysis, research, publications'
    },
}

# =============================================================================
# MODEL TIERS - Classification
# =============================================================================

CLASSIFICATION_TIERS = {
    'quick': {
        'description': 'Minimal set for rapid classification testing',
        'models': ['PLS-DA', 'LightGBM', 'RandomForest'],
        'recommended_for': 'Quick tests, preliminary analysis, daily QC'
    },

    'standard': {
        'description': 'Fast & reliable production classifiers',
        'models': ['PLS-DA', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost'],
        'recommended_for': 'Most users, daily classification, routine work'
    },

    'comprehensive': {
        'description': 'Advanced classifiers for thorough analysis',
        'models': ['PLS-DA', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost', 'SVM', 'MLP'],
        'recommended_for': 'Research, publications, thorough method comparison'
    },
}

DEFAULT_TIER = 'standard'

# =============================================================================
# HYPERPARAMETER GRIDS (same for all tiers - tier only affects which models)
# =============================================================================

HYPERPARAMETER_GRIDS = {
    'PLS': [{'n_components': c} for c in [2, 4, 6, 8, 10, 12, 15]],

    'Ridge': [{'alpha': a} for a in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]],

    'Lasso': [{'alpha': a} for a in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]],

    'ElasticNet': [
        {'alpha': a, 'l1_ratio': r}
        for a in [0.01, 0.1, 0.5, 1.0]
        for r in [0.2, 0.5, 0.8]
    ],

    'RandomForest': [
        {'n_estimators': n, 'max_depth': d}
        for n in [100, 200]
        for d in [None, 20, 30]
    ],

    'LightGBM': [
        {'n_estimators': n, 'learning_rate': lr, 'num_leaves': nl}
        for n in [100, 200]
        for lr in [0.05, 0.1]
        for nl in [31, 63]
    ],

    'XGBoost': [
        {'n_estimators': n, 'learning_rate': lr, 'max_depth': d}
        for n in [100, 200]
        for lr in [0.05, 0.1]
        for d in [3, 6]
    ],

    'CatBoost': [
        {'iterations': n, 'learning_rate': lr, 'depth': d}
        for n in [100, 200]
        for lr in [0.05, 0.1]
        for d in [4, 6]
    ],

    'SVR': [
        {'kernel': k, 'C': c}
        for k in ['rbf', 'linear']
        for c in [0.1, 1.0, 10.0, 100.0]
    ],

    'SVM': [
        {'kernel': k, 'C': c}
        for k in ['rbf', 'linear']
        for c in [0.1, 1.0, 10.0, 100.0]
    ],

    'PLS-DA': [{'n_components': c} for c in [2, 4, 6, 8, 10, 12, 15]],

    'NeuralBoosted': [
        {'n_estimators': n, 'learning_rate': lr, 'hidden_layer_size': hs}
        for n in [50, 100]
        for lr in [0.05, 0.1]
        for hs in [3, 5]
    ],

    'MLP': [
        {'hidden_layer_sizes': hls, 'activation': act, 'alpha': a}
        for hls in [(50,), (100,), (50, 50), (100, 50)]
        for act in ['relu', 'tanh']
        for a in [0.0001, 0.001, 0.01]
    ],
}

# =============================================================================
# PREPROCESSING CONFIGURATION BY TIER
# =============================================================================

PREPROCESSING_CONFIG = {
    'quick': {
        'methods': ['raw', 'snv'],
        'window_sizes': [11],
        'derivative_orders': [1],
    },

    'standard': {
        'methods': ['raw', 'snv', 'deriv1', 'deriv2'],
        'window_sizes': [7, 19],
        'derivative_orders': [1, 2],
    },

    'comprehensive': {
        'methods': ['raw', 'snv', 'deriv1', 'deriv2', 'snv_deriv1', 'snv_deriv2'],
        'window_sizes': [7, 15, 25],
        'derivative_orders': [1, 2],
    },
}

# =============================================================================
# VARIABLE SELECTION CONFIGURATION BY TIER
# =============================================================================

VARIABLE_SELECTION_CONFIG = {
    'quick': {
        'methods': ['importance'],
        'counts': [50, 100],
        'enable_regions': False,
    },

    'standard': {
        'methods': ['importance', 'spa'],
        'counts': [20, 50, 100, 250],
        'enable_regions': True,
        'n_top_regions': 5,
    },

    'comprehensive': {
        'methods': ['importance', 'spa', 'uve', 'uve_spa', 'ipls', 'cars'],
        'counts': [10, 20, 50, 100, 250, 500, 1000],
        'enable_regions': True,
        'n_top_regions': 10,
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tier_models(tier: str = 'standard', task_type: str = 'regression') -> List[str]:
    """
    Get the list of models for a given tier and task type.

    Parameters
    ----------
    tier : str
        One of 'quick', 'standard', 'comprehensive'
    task_type : str
        Either 'regression' or 'classification'

    Returns
    -------
    List[str]
        List of model names for the specified tier and task type
    """
    if task_type == 'classification':
        tier_dict = CLASSIFICATION_TIERS
    else:
        tier_dict = MODEL_TIERS

    if tier not in tier_dict:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {list(tier_dict.keys())}")

    return tier_dict[tier]['models']


def get_hyperparameter_grid(model_name: str, max_lv: int = None) -> List[Dict[str, Any]]:
    """
    Get the hyperparameter grid for a model.

    Note: Hyperparameter grids are the same for all tiers. Tier only affects
    which models are tested, not the hyperparameters used.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'PLS', 'Ridge', 'LightGBM')
    max_lv : int, optional
        For PLS/PLS-DA models, the maximum number of latent variables to test.
        If provided, generates a grid testing all values from 1 to max_lv.

    Returns
    -------
    List[Dict[str, Any]]
        List of hyperparameter dictionaries to test
    """
    # Special handling for PLS/PLS-DA with user-specified max latent variables
    if model_name in ['PLS', 'PLS-DA'] and max_lv is not None:
        # Test all n_components from 1 to max_lv
        return [{'n_components': c} for c in range(1, max_lv + 1)]

    if model_name not in HYPERPARAMETER_GRIDS:
        return [{}]  # No hyperparameters to tune

    return HYPERPARAMETER_GRIDS[model_name]


def get_preprocessing_config(tier: str = 'standard') -> Dict[str, Any]:
    """
    Get the preprocessing configuration for a tier.

    Parameters
    ----------
    tier : str
        Tier level ('quick', 'standard', 'comprehensive')

    Returns
    -------
    Dict[str, Any]
        Preprocessing configuration
    """
    if tier not in PREPROCESSING_CONFIG:
        tier = 'standard'

    return PREPROCESSING_CONFIG[tier]


def get_variable_selection_config(tier: str = 'standard') -> Dict[str, Any]:
    """
    Get the variable selection configuration for a tier.

    Parameters
    ----------
    tier : str
        Tier level ('quick', 'standard', 'comprehensive')

    Returns
    -------
    Dict[str, Any]
        Variable selection configuration
    """
    if tier not in VARIABLE_SELECTION_CONFIG:
        tier = 'standard'

    return VARIABLE_SELECTION_CONFIG[tier]


def estimate_total_configurations(
    tier: str,
    task_type: str = 'regression',
    enable_preprocessing: bool = True,
    enable_variable_selection: bool = True,
    enable_hyperparameter_search: bool = True
) -> int:
    """
    Estimate the total number of model configurations to test.

    Parameters
    ----------
    tier : str
        Tier level
    task_type : str
        'regression' or 'classification'
    enable_preprocessing : bool
        Whether preprocessing variations are enabled
    enable_variable_selection : bool
        Whether variable selection is enabled
    enable_hyperparameter_search : bool
        Whether hyperparameter search is enabled

    Returns
    -------
    int
        Estimated number of configurations
    """
    models = get_tier_models(tier, task_type)
    preproc_config = get_preprocessing_config(tier)
    varsel_config = get_variable_selection_config(tier)

    # Count preprocessing configurations
    n_preproc = 1
    if enable_preprocessing:
        n_methods = len(preproc_config['methods'])
        n_windows = len(preproc_config['window_sizes'])
        # Only derivatives use window sizes
        n_deriv_methods = sum(1 for m in preproc_config['methods'] if 'deriv' in m)
        n_non_deriv = n_methods - n_deriv_methods
        n_preproc = n_non_deriv + (n_deriv_methods * n_windows)

    # Count variable selection configurations
    n_varsel = 1
    if enable_variable_selection:
        n_methods = len(varsel_config['methods'])
        n_counts = len(varsel_config['counts'])
        n_varsel = 1 + (n_methods * n_counts)  # +1 for full spectrum

    # Count hyperparameter configurations per model
    total = 0
    for model_name in models:
        if enable_hyperparameter_search:
            n_hyperparams = len(get_hyperparameter_grid(model_name, tier))
        else:
            n_hyperparams = 1

        total += n_preproc * n_varsel * n_hyperparams

    return total
