"""
Model configuration and tiered defaults for spectral analysis.

This module defines:
1. Model tiers (Standard, Comprehensive, Experimental)
2. Optimized hyperparameter grids
3. Preprocessing defaults (window sizes, etc.)
4. User-customizable settings
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
        'models': ['PLS-DA', 'RandomForest', 'LightGBM', 'XGBoost', 'CatBoost', 'SVM', 'MLP'],
        'recommended_for': 'Research, publications, thorough method comparison'
    },

    'experimental': {
        'description': 'All available classifiers including experimental',
        'models': ['PLS-DA', 'PLS', 'RandomForest', 'LightGBM', 'XGBoost',
                   'CatBoost', 'SVM', 'MLP'],
        'recommended_for': 'Exploration, method comparison, no time constraints'
    }
}

# Default tier
DEFAULT_TIER = 'standard'

# =============================================================================
# OPTIMIZED HYPERPARAMETER GRIDS
# =============================================================================

# These grids are optimized for 70-80% of best performance with 30-40% of computation time
# Based on benchmarks across spectral datasets

OPTIMIZED_HYPERPARAMETERS = {
    # =========================================================================
    # TIER 1: STANDARD MODELS (Always good defaults)
    # =========================================================================

    'PLS': {
        'standard': {
            'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # 12 configs - same as comprehensive
            'note': 'All tiers use same hyperparams - tier controls models/preprocessing only'
        },
        'comprehensive': {
            'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # 12 configs
            'note': 'All tiers use same hyperparams - tier controls models/preprocessing only'
        },
        'quick': {
            'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # 12 configs - same as comprehensive
            'note': 'All tiers use same hyperparams - tier controls models/preprocessing only'
        }
    },

    'Ridge': {
        'standard': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # 5 configs - same as comprehensive
            'note': 'All tiers use same hyperparams'
        },
        'comprehensive': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # 5 configs
            'note': 'All tiers use same hyperparams'
        },
        'quick': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # 5 configs - same as comprehensive
            'note': 'All tiers use same hyperparams'
        }
    },

    'ElasticNet': {
        'standard': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values - same as comprehensive
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # 5 values - same as comprehensive
            'note': 'Grid size: 4×5 = 20 configs - all tiers use same hyperparams'
        },
        'comprehensive': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # 5 values
            'note': 'Grid size: 4×5 = 20 configs'
        },
        'quick': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values - same as comprehensive
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # 5 values - same as comprehensive
            'note': 'Grid size: 4×5 = 20 configs - all tiers use same hyperparams'
        }
    },

    'XGBoost': {
        'standard': {
            'n_estimators': [100, 200],  # 2 values
            'learning_rate': [0.05, 0.1],  # 2 values
            'max_depth': [3, 6, 9],  # 3 values
            'subsample': [0.7, 0.85, 1.0],  # 3 values
            'colsample_bytree': [0.7, 0.85, 1.0],  # 3 values
            'reg_alpha': [0, 0.1, 0.5],  # 3 values
            'reg_lambda': [1.0, 5.0],  # 2 values
            'note': 'Grid size: 2×2×3×3×3×3×2 = 648 configs - all tiers use same hyperparams'
        },
        'comprehensive': {
            'n_estimators': [100, 200],  # 2 values
            'learning_rate': [0.05, 0.1],  # 2 values
            'max_depth': [3, 6, 9],  # 3 values
            'subsample': [0.7, 0.85, 1.0],  # 3 values
            'colsample_bytree': [0.7, 0.85, 1.0],  # 3 values
            'reg_alpha': [0, 0.1, 0.5],  # 3 values
            'reg_lambda': [1.0, 5.0],  # 2 values
            'note': 'Grid size: 2×2×3×3×3×3×2 = 648 configs'
        },
        'quick': {
            'n_estimators': [100, 200],  # 2 values
            'learning_rate': [0.05, 0.1],  # 2 values
            'max_depth': [3, 6, 9],  # 3 values
            'subsample': [0.7, 0.85, 1.0],  # 3 values
            'colsample_bytree': [0.7, 0.85, 1.0],  # 3 values
            'reg_alpha': [0, 0.1, 0.5],  # 3 values
            'reg_lambda': [1.0, 5.0],  # 2 values
            'note': 'Grid size: 2×2×3×3×3×3×2 = 648 configs - all tiers use same hyperparams'
        }
    },

    # =========================================================================
    # TIER 2: COMPREHENSIVE MODELS (Advanced analysis)
    # =========================================================================

    'LightGBM': {
        'standard': {
            'n_estimators': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'num_leaves': [31, 50, 70],  # 3 values
            'min_child_samples': [5, 10, 20],  # 3 values - min samples per leaf
            'subsample': [0.7, 0.8, 1.0],  # 3 values - row sampling
            'colsample_bytree': [0.7, 0.8, 1.0],  # 3 values - feature sampling
            'reg_alpha': [0.0, 0.1, 0.5],  # 3 values - L1 regularization
            'reg_lambda': [0.5, 1.0, 2.0],  # 3 values - L2 regularization
            'note': 'Grid size: 3×3×3×3×3×3×3×3 = 6561 configs (use tier defaults or subset)'
        },
        'comprehensive': {
            'n_estimators': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'num_leaves': [31, 50, 70],  # 3 values
            'min_child_samples': [5, 10, 20],  # 3 values
            'subsample': [0.7, 0.8, 1.0],  # 3 values
            'colsample_bytree': [0.7, 0.8, 1.0],  # 3 values
            'reg_alpha': [0.0, 0.1, 0.5],  # 3 values
            'reg_lambda': [0.5, 1.0, 2.0],  # 3 values
            'note': 'Grid size: 3×3×3×3×3×3×3×3 = 6561 configs'
        },
        'quick': {
            'n_estimators': [100],  # 1 value for speed
            'learning_rate': [0.1],  # 1 value for speed
            'num_leaves': [31, 50],  # 2 values
            'min_child_samples': [5, 10],  # 2 values
            'subsample': [0.8],  # 1 value for speed
            'colsample_bytree': [0.8],  # 1 value for speed
            'reg_alpha': [0.0, 0.1],  # 2 values
            'reg_lambda': [1.0],  # 1 value for speed
            'note': 'Grid size: 1×1×2×2×1×1×2×1 = 8 configs - optimized for speed'
        }
    },

    'SVR': {
        'standard': {
            'kernel': ['rbf', 'linear'],  # 2 kernels
            'C': [0.1, 1.0, 10.0],  # 3 values
            'gamma': ['scale', 'auto'],  # 2 values for RBF
            'note': 'Grid size: 9 configs - all tiers use same hyperparams'
        },
        'comprehensive': {
            'kernel': ['rbf', 'linear'],  # 2 kernels
            'C': [0.1, 1.0, 10.0],  # 3 values
            'gamma': ['scale', 'auto'],  # 2 values for RBF
            'note': 'Grid size: 9 configs'
        },
        'quick': {
            'kernel': ['rbf', 'linear'],  # 2 kernels
            'C': [0.1, 1.0, 10.0],  # 3 values
            'gamma': ['scale', 'auto'],  # 2 values for RBF
            'note': 'Grid size: 9 configs - all tiers use same hyperparams'
        }
    },

    'NeuralBoosted': {
        # Keep existing intelligent defaults - they're already optimized
        'standard': {
            'n_estimators': [100, 150],  # 2 values
            'learning_rate': [0.1, 0.2, 0.3],  # 3 values
            'hidden_layer_size': [3, 5],  # 2 values
            'activation': ['tanh', 'identity'],  # 2 values
            'note': 'Grid size: 2×3×2×2 = 24 configs - all tiers use same hyperparams'
        },
        'comprehensive': {
            'n_estimators': [100, 150],  # 2 values
            'learning_rate': [0.1, 0.2, 0.3],  # 3 values
            'hidden_layer_size': [3, 5],  # 2 values
            'activation': ['tanh', 'identity'],  # 2 values
            'note': 'Grid size: 2×3×2×2 = 24 configs'
        },
        'quick': {
            'n_estimators': [100, 150],  # 2 values
            'learning_rate': [0.1, 0.2, 0.3],  # 3 values
            'hidden_layer_size': [3, 5],  # 2 values
            'activation': ['tanh', 'identity'],  # 2 values
            'note': 'Grid size: 2×3×2×2 = 24 configs - all tiers use same hyperparams'
        }
    },

    # =========================================================================
    # TIER 3: EXPERIMENTAL MODELS (Use sparingly)
    # =========================================================================

    'CatBoost': {
        'standard': {
            'iterations': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'depth': [4, 6, 8],  # 3 values
            'note': 'Grid size: 3×3×3 = 27 configs - all tiers use same hyperparams'
        },
        'comprehensive': {
            'iterations': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'depth': [4, 6, 8],  # 3 values
            'note': 'Grid size: 3×3×3 = 27 configs'
        },
        'quick': {
            'iterations': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'depth': [4, 6, 8],  # 3 values
            'note': 'Grid size: 3×3×3 = 27 configs - all tiers use same hyperparams'
        }
    },

    'RandomForest': {
        'standard': {
            'n_estimators': [100, 200, 500],  # 3 values
            'max_depth': [None, 15, 30],  # 3 values
            'min_samples_split': [2, 5, 10],  # 3 values - min samples to split node
            'min_samples_leaf': [1, 2, 4],  # 3 values - min samples per leaf
            'max_features': ['sqrt', 'log2', None],  # 3 values - features per split
            'note': 'Grid size: 3×3×3×3×3 = 243 configs'
        },
        'comprehensive': {
            'n_estimators': [100, 200, 500],  # 3 values
            'max_depth': [None, 15, 30],  # 3 values
            'min_samples_split': [2, 5, 10],  # 3 values
            'min_samples_leaf': [1, 2, 4],  # 3 values
            'max_features': ['sqrt', 'log2', None],  # 3 values
            'note': 'Grid size: 3×3×3×3×3 = 243 configs'
        },
        'quick': {
            'n_estimators': [100, 200],  # 2 values for speed
            'max_depth': [None, 30],  # 2 values
            'min_samples_split': [2, 5],  # 2 values
            'min_samples_leaf': [1, 2],  # 2 values
            'max_features': ['sqrt'],  # 1 value for speed
            'note': 'Grid size: 2×2×2×2×1 = 16 configs - optimized for speed'
        }
    },

    'MLP': {
        'standard': {
            'hidden_layer_sizes': [(64,), (128, 64)],  # 2 architectures
            'alpha': [1e-4, 1e-3],  # 2 values
            'learning_rate_init': [1e-3, 1e-2],  # 2 values
            'note': 'Grid size: 2×2×2 = 8 configs - all tiers use same hyperparams'
        },
        'comprehensive': {
            'hidden_layer_sizes': [(64,), (128, 64)],  # 2 architectures
            'alpha': [1e-4, 1e-3],  # 2 values
            'learning_rate_init': [1e-3, 1e-2],  # 2 values
            'note': 'Grid size: 2×2×2 = 8 configs'
        },
        'quick': {
            'hidden_layer_sizes': [(64,), (128, 64)],  # 2 architectures
            'alpha': [1e-4, 1e-3],  # 2 values
            'learning_rate_init': [1e-3, 1e-2],  # 2 values
            'note': 'Grid size: 2×2×2 = 8 configs - all tiers use same hyperparams'
        }
    },

    'Lasso': {
        # Same as Ridge but typically not needed if ElasticNet is used
        'standard': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 configs - all tiers use same hyperparams
            'note': 'All tiers use same hyperparams'
        },
        'comprehensive': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 configs
            'note': 'Full search'
        },
        'quick': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 configs - all tiers use same hyperparams
            'note': 'All tiers use same hyperparams'
        }
    }
}

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
# GRID SIZE SUMMARY
# =============================================================================

def get_grid_size_summary():
    """Return summary of configurations per tier."""
    summary = {
        'standard': {
            'PLS': 8,
            'Ridge': 4,
            'ElasticNet': 9,
            'XGBoost': 8,
            'total': 29,
            'estimated_time': '10-15 min'
        },
        'comprehensive': {
            'PLS': 12,
            'Ridge': 5,
            'ElasticNet': 20,
            'XGBoost': 27,
            'LightGBM': 4,
            'SVR': 5,
            'NeuralBoosted': 8,
            'total': 81,
            'estimated_time': '20-30 min'
        },
        'experimental': {
            'all_models': 'Variable, 100-200+ configs',
            'estimated_time': '45-90 min'
        },
        'quick': {
            'PLS': 3,
            'Ridge': 2,
            'XGBoost': 1,
            'total': 6,
            'estimated_time': '3-5 min'
        }
    }
    return summary


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


def get_hyperparameters(model_name, tier='standard'):
    """
    Get hyperparameters for a model at a given tier.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'PLS', 'XGBoost')
    tier : str
        One of 'quick', 'standard', 'comprehensive'

    Returns
    -------
    dict
        Hyperparameter configuration
    """
    if model_name not in OPTIMIZED_HYPERPARAMETERS:
        raise ValueError(f"Unknown model: {model_name}")

    if tier not in OPTIMIZED_HYPERPARAMETERS[model_name]:
        # Fall back to standard if tier not defined
        tier = 'standard'

    return OPTIMIZED_HYPERPARAMETERS[model_name][tier]


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
    print("GRID SIZE SUMMARY")
    print("=" * 70)

    summary = get_grid_size_summary()
    for tier_name, tier_data in summary.items():
        print(f"\n{tier_name.upper()}:")
        for key, value in tier_data.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)


# =============================================================================
# CLASSIFICATION DEFAULTS
# =============================================================================

# PLS-DA specific parameters (PLS + LogisticRegression pipeline)
PLS_DA_HYPERPARAMETERS = {
    'standard': {
        'pls__n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # PLS components
        'lr__C': [0.1, 1.0, 10.0],  # LogisticRegression regularization strength
        'lr__penalty': ['l2'],  # LogisticRegression penalty (l2 is most stable)
        'lr__solver': ['lbfgs'],  # LogisticRegression solver
        'note': 'Grid size: 12×3 = 36 configs (lr__penalty and lr__solver usually kept as defaults)'
    },
    'comprehensive': {
        'pls__n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # PLS components
        'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Wider range for regularization
        'lr__penalty': ['l2'],  # LogisticRegression penalty
        'lr__solver': ['lbfgs'],  # LogisticRegression solver
        'note': 'Grid size: 12×5 = 60 configs'
    },
    'quick': {
        'pls__n_components': [2, 4, 6, 8, 10],  # Fewer components for speed
        'lr__C': [1.0, 10.0],  # Just 2 values for speed
        'lr__penalty': ['l2'],  # LogisticRegression penalty
        'lr__solver': ['lbfgs'],  # LogisticRegression solver
        'note': 'Grid size: 5×2 = 10 configs - optimized for speed'
    }
}

# For classification, use similar structure but adapted
CLASSIFICATION_HYPERPARAMETERS = {
    'PLS-DA': PLS_DA_HYPERPARAMETERS,  # Now has its own with LogisticRegression params
    'SVM': OPTIMIZED_HYPERPARAMETERS['SVR'],  # Same as SVR
    'XGBoost': OPTIMIZED_HYPERPARAMETERS['XGBoost'],  # Same hyperparams
    'LightGBM': OPTIMIZED_HYPERPARAMETERS['LightGBM'],  # Same hyperparams
    'CatBoost': OPTIMIZED_HYPERPARAMETERS['CatBoost'],  # Same hyperparams
    'RandomForest': OPTIMIZED_HYPERPARAMETERS['RandomForest'],  # Same
    'MLP': OPTIMIZED_HYPERPARAMETERS['MLP']  # Same
}


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
