"""
Model configuration and tiered defaults for spectral analysis.

This module defines:
1. Model tiers (Quick, Standard, Comprehensive, Experimental)
2. Which models belong to each tier (for regression and classification)
3. Preprocessing defaults (window sizes, derivative orders, methods)

Hyperparameters are controlled by the Analysis Configuration GUI, not hardcoded here.
Tiers only determine which models are run, not how they are configured.
"""

from typing import List

# =============================================================================
# MODEL TIERS
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
        'description': 'Advanced analysis with gradient boosting',
        'models': ['PLS', 'Ridge', 'ElasticNet', 'RandomForest', 'LightGBM',
                   'XGBoost', 'CatBoost', 'NeuralBoosted'],
        'recommended_for': 'Thorough analysis, research, publications'
    },

    'experimental': {
        'description': 'All available models including slow ones',
        'models': ['PLS', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'XGBoost',
                   'LightGBM', 'CatBoost', 'SVR', 'MLP', 'NeuralBoosted'],
        'recommended_for': 'Exploration, method comparison, research, no time constraints'
    }
}

# Classification-specific model tiers (optimized for categorical target variables)
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
        'models': ['PLS-DA', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost', 'SVM', 'MLP', 'NeuralBoosted'],
        'recommended_for': 'Research, publications, thorough method comparison'
    },

    'experimental': {
        'description': 'All available classifiers including experimental',
        'models': ['PLS-DA', 'PLS', 'RandomForest', 'LightGBM', 'XGBoost',
                   'CatBoost', 'SVM', 'MLP', 'NeuralBoosted'],
        'recommended_for': 'Exploration, method comparison, no time constraints'
    }
}

# Default tier
DEFAULT_TIER = 'standard'

# =============================================================================
# PREPROCESSING DEFAULTS
# =============================================================================

PREPROCESSING_DEFAULTS = {
    'savitzky_golay': {
        'standard': {
            'window_lengths': [7, 19],  # Current defaults - good balance
            'polynomial_order': 2,
            'note': 'Window 7: preserves sharp features, Window 19: more smoothing'
        },
        'comprehensive': {
            'window_lengths': [7, 15, 25],  # 3 windows: small/medium/large
            'polynomial_order': 2,
            'note': 'More thorough exploration of smoothing levels'
        },
        'quick': {
            'window_lengths': [11],  # 1 window: middle ground
            'polynomial_order': 2,
            'note': 'Single balanced window'
        },
        'recommended': {
            'window_lengths': [7, 19],  # Keep current as recommended
            'notes': [
                'Window 5-11: Preserve sharp spectral features, more noise',
                'Window 15-25: Balance smoothing and feature preservation',
                'Window 31+: Heavy smoothing, may lose important features',
                'For VIS-NIR: 7-19 is optimal (current default)',
                'For NIR only: 11-25 works well',
                'For Raman: 5-15 preserves sharp peaks'
            ]
        }
    },

    'methods': {
        'standard': ['raw', 'snv', 'deriv1', 'deriv2'],
        'comprehensive': ['raw', 'snv', 'deriv1', 'deriv2', 'snv_deriv1', 'snv_deriv2'],
        'quick': ['raw', 'snv']
    },

    'derivative_orders': {
        'standard': [1, 2],  # 1st and 2nd derivatives
        'comprehensive': [1, 2],  # Same - higher orders rarely useful
        'quick': [1]  # Just 1st derivative
    }
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
        One of 'quick', 'standard', 'comprehensive', 'experimental'
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


def get_hyperparameters(model_name: str, tier: str = 'standard') -> dict:
    """
    Get hyperparameter defaults for a specific model and tier.

    NOTE: This function provides fallback defaults when GUI doesn't provide parameters.
    Primary hyperparameter control is through the Analysis Configuration GUI.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'Ridge', 'RandomForest', 'LightGBM')
    tier : str
        Tier level ('quick', 'standard', 'comprehensive', 'experimental')

    Returns
    -------
    dict
        Dictionary of hyperparameter defaults for the specified model and tier
    """
    # Define fallback hyperparameter defaults
    # These are used only when GUI doesn't provide specific values
    HYPERPARAMETER_DEFAULTS = {
        'NeuralBoosted': {
            'standard': {
                'n_estimators': [100],
                'learning_rate': [0.1],
                'hidden_layer_size': [100],
                'activation': ['relu']
            }
        },
        'PLS': {
            'standard': {
                'max_iter': [500],
                'tol': [1e-6]
            }
        },
        'Ridge': {
            'standard': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'solver': ['auto'],
                'tol': [1e-4]
            }
        },
        'Lasso': {
            'standard': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'selection': ['cyclic'],
                'tol': [1e-4]
            }
        },
        'ElasticNet': {
            'standard': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.3, 0.5, 0.7],
                'selection': ['cyclic'],
                'tol': [1e-4]
            }
        },
        'RandomForest': {
            'standard': {
                'n_estimators': [100],
                'max_depth': [None, 30],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'bootstrap': [True],
                'max_leaf_nodes': [None],
                'min_impurity_decrease': [0.0]
            }
        },
        'XGBoost': {
            'standard': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 6],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'reg_alpha': [0.0],
                'reg_lambda': [1.0]
            }
        },
        'LightGBM': {
            'standard': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63],
                'max_depth': [-1],
                'min_child_samples': [5],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'reg_alpha': [0.1],
                'reg_lambda': [1.0]
            }
        },
        'CatBoost': {
            'standard': {
                'iterations': [100, 200],
                'learning_rate': [0.05, 0.1],
                'depth': [6],
                'l2_leaf_reg': [3.0],
                'border_count': [128],
                'bagging_temperature': [1.0],
                'random_strength': [1.0]
            }
        },
        'SVR': {
            'standard': {
                'kernel': ['rbf', 'linear'],
                'C': [1.0, 10.0],
                'gamma': ['scale'],
                'epsilon': [0.1],
                'degree': [3],
                'coef0': [0.0],
                'shrinking': [True]
            }
        },
        'MLP': {
            'standard': {
                'hidden_layer_sizes': [(64,), (128, 64)],
                'alpha': [0.001],
                'learning_rate_init': [0.001],
                'activation': ['relu'],
                'solver': ['adam'],
                'batch_size': ['auto'],
                'learning_rate': ['constant'],
                'momentum': [0.9]
            }
        }
    }

    # Return defaults for the specified model and tier
    if model_name not in HYPERPARAMETER_DEFAULTS:
        return {}

    model_defaults = HYPERPARAMETER_DEFAULTS[model_name]

    # If tier not found, use 'standard' as fallback
    if tier not in model_defaults:
        tier = 'standard'

    return model_defaults.get(tier, {})


def print_tier_summary():
    """Print a summary of all tiers for user reference."""
    print("=" * 70)
    print("SPECTRAL PREDICT - MODEL TIERS")
    print("=" * 70)

    for tier_name, tier_info in MODEL_TIERS.items():
        print(f"\n{tier_name.upper()}")
        print(f"  Description: {tier_info['description']}")
        print(f"  Models: {', '.join(tier_info['models'])}")
        print(f"  Best for: {tier_info['recommended_for']}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    # Print summary when run directly
    print_tier_summary()

    # Show preprocessing recommendations
    print("\n" + "=" * 70)
    print("PREPROCESSING RECOMMENDATIONS")
    print("=" * 70)
    sg_info = PREPROCESSING_DEFAULTS['savitzky_golay']['recommended']
    print("\nSavitzky-Golay Window Size Guidelines:")
    for note in sg_info['notes']:
        print(f"  • {note}")
