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
            'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # 12 configs
            'max_iter': [500],  # default (single value)
            'tol': [1e-6],  # default (single value)
            'note': 'Grid size: 12×1×1 = 12 configs (unchanged with defaults)'
        },
        'comprehensive': {
            'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # 12 configs
            'max_iter': [500, 1000],  # explore convergence
            'tol': [1e-7, 1e-6, 1e-5],  # tolerance exploration
            'note': 'Grid size: 12×2×3 = 72 configs (comprehensive exploration)'
        },
        'quick': {
            'n_components': [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50],  # 12 configs
            'max_iter': [500],  # default (single value)
            'tol': [1e-6],  # default (single value)
            'note': 'Grid size: 12×1×1 = 12 configs (unchanged with defaults)'
        }
    },

    'Ridge': {
        'standard': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # 5 configs
            'solver': ['auto'],  # auto-select best solver (single value)
            'tol': [1e-4],  # default tolerance (single value)
            'note': 'Grid size: 5×1×1 = 5 configs (unchanged with defaults)'
        },
        'comprehensive': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # 5 configs
            'solver': ['auto', 'svd', 'cholesky'],  # compare solvers
            'tol': [1e-5, 1e-4, 1e-3],  # tolerance exploration
            'note': 'Grid size: 5×3×3 = 45 configs (comprehensive exploration)'
        },
        'quick': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # 5 configs
            'solver': ['auto'],  # auto-select best solver (single value)
            'tol': [1e-4],  # default tolerance (single value)
            'note': 'Grid size: 5×1×1 = 5 configs (unchanged with defaults)'
        }
    },

    'Lasso': {
        'standard': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values
            'selection': ['cyclic'],  # default coordinate descent (single value)
            'tol': [1e-4],  # default tolerance (single value)
            'note': 'Grid size: 4×1×1 = 4 configs (unchanged with defaults)'
        },
        'comprehensive': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values
            'selection': ['cyclic', 'random'],  # compare selection strategies
            'tol': [1e-5, 1e-4, 1e-3],  # tolerance exploration
            'note': 'Grid size: 4×2×3 = 24 configs (comprehensive exploration)'
        },
        'quick': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values
            'selection': ['cyclic'],  # default coordinate descent (single value)
            'tol': [1e-4],  # default tolerance (single value)
            'note': 'Grid size: 4×1×1 = 4 configs (unchanged with defaults)'
        }
    },

    'ElasticNet': {
        'standard': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # 5 values
            'selection': ['cyclic'],  # default coordinate descent (single value)
            'tol': [1e-4],  # default tolerance (single value)
            'note': 'Grid size: 4×5×1×1 = 20 configs (unchanged with defaults)'
        },
        'comprehensive': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # 5 values
            'selection': ['cyclic', 'random'],  # compare selection strategies
            'tol': [1e-5, 1e-4, 1e-3],  # tolerance exploration
            'note': 'Grid size: 4×5×2×3 = 120 configs (comprehensive exploration)'
        },
        'quick': {
            'alpha': [0.001, 0.01, 0.1, 1.0],  # 4 values
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # 5 values
            'selection': ['cyclic'],  # default coordinate descent (single value)
            'tol': [1e-4],  # default tolerance (single value)
            'note': 'Grid size: 4×5×1×1 = 20 configs (unchanged with defaults)'
        }
    },

    'XGBoost': {
        'standard': {
            'n_estimators': [100, 200],  # 2 values (original)
            'learning_rate': [0.05, 0.1],  # 2 values (original)
            'max_depth': [3, 6],  # 2 values (original - reduced from 3)
            'subsample': [0.8, 1.0],  # 2 values (original)
            'colsample_bytree': [0.8, 1.0],  # 2 values (original)
            'reg_alpha': [0, 0.1],  # 2 values (original)
            'reg_lambda': [1.0],  # Single value default (XGBoost default)
            'min_child_weight': [1],  # Single value default (XGBoost default)
            'gamma': [0],  # Single value default (XGBoost default)
            'note': 'Grid size: 2×2×2×2×2×2×1×1×1 = 64 configs (original working defaults restored)'
        },
        'comprehensive': {
            'n_estimators': [100, 200],  # 2 values
            'learning_rate': [0.05, 0.1],  # 2 values
            'max_depth': [3, 6, 9],  # 3 values
            'subsample': [0.7, 0.85, 1.0],  # 3 values
            'colsample_bytree': [0.7, 0.85, 1.0],  # 3 values
            'reg_alpha': [0, 0.1, 0.5],  # 3 values
            'reg_lambda': [1.0, 5.0],  # 2 values
            'min_child_weight': [1, 3, 5],  # 3 values - minimum sum of instance weight (child)
            'gamma': [0, 0.1, 0.5],  # 3 values - minimum loss reduction for split
            'note': 'Grid size: 2×2×3×3×3×3×2×3×3 = 5832 configs'
        },
        'quick': {
            'n_estimators': [100, 200],  # 2 values
            'learning_rate': [0.05, 0.1],  # 2 values
            'max_depth': [3, 6, 9],  # 3 values
            'subsample': [0.7, 0.85, 1.0],  # 3 values
            'colsample_bytree': [0.7, 0.85, 1.0],  # 3 values
            'reg_alpha': [0, 0.1, 0.5],  # 3 values
            'reg_lambda': [1.0, 5.0],  # 2 values
            'min_child_weight': [1],  # Single value
            'gamma': [0],  # Single value
            'note': 'Grid size: 2×2×3×3×3×3×2×1×1 = 648 configs (unchanged with single-value defaults)'
        }
    },

    # =========================================================================
    # TIER 2: COMPREHENSIVE MODELS (Advanced analysis)
    # =========================================================================

    'LightGBM': {
        'standard': {
            'n_estimators': [100, 200],  # 2 values
            'learning_rate': [0.1],  # 1 value (not very sensitive)
            'num_leaves': [31, 50],  # 2 values
            'max_depth': [-1],  # No limit (controlled by num_leaves)
            'min_child_samples': [20],  # Minimum samples per leaf
            'subsample': [0.8],  # Row sampling to prevent overfitting (like XGBoost)
            'colsample_bytree': [0.8],  # Feature sampling for high-dim data (like XGBoost)
            'reg_alpha': [0.1],  # L1 regularization for feature selection (like XGBoost)
            'reg_lambda': [1.0],  # L2 regularization to prevent overfitting (like XGBoost)
            'note': 'Grid size: 2×1×2×1×1×1×1×1×1 = 4 configs (with regularization to prevent overfitting)'
        },
        'comprehensive': {
            'n_estimators': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'num_leaves': [7, 15, 31, 50],  # 4 values - added smaller values for small datasets
            'max_depth': [-1, 10, 20],  # vary depth limit
            'min_child_samples': [5, 10, 20],  # vary minimum samples (reduced from 5, 20, 50)
            'subsample': [0.8, 1.0],  # with/without subsampling
            'colsample_bytree': [0.8, 1.0],  # vary feature fraction
            'reg_alpha': [0.1, 0.5],  # L1 regularization (removed 0.0 - always use some regularization)
            'reg_lambda': [1.0, 5.0],  # L2 regularization (removed 0.0 - always use some regularization)
            'note': 'Grid size: 3×3×4×3×3×2×2×2×2 = 5184 configs (comprehensive exploration, optimized for small datasets)'
        },
        'quick': {
            'n_estimators': [100],  # 1 value
            'learning_rate': [0.1],  # 1 value
            'num_leaves': [31],  # 1 value
            'max_depth': [-1],  # No limit (controlled by num_leaves)
            'min_child_samples': [20],  # Minimum samples per leaf
            'subsample': [0.8],  # Row sampling to prevent overfitting (like XGBoost)
            'colsample_bytree': [0.8],  # Feature sampling for high-dim data (like XGBoost)
            'reg_alpha': [0.1],  # L1 regularization for feature selection (like XGBoost)
            'reg_lambda': [1.0],  # L2 regularization to prevent overfitting (like XGBoost)
            'note': 'Grid size: 1×1×1×1×1×1×1×1×1 = 1 config (with regularization to prevent overfitting)'
        }
    },

    'SVR': {
        'standard': {
            'kernel': ['rbf', 'linear'],  # 2 kernels
            'C': [0.1, 1.0, 10.0],  # 3 values
            'gamma': ['scale', 'auto'],  # 2 values for RBF
            'epsilon': [0.1],  # Single value default
            'degree': [3],  # Single value default (only for poly kernel)
            'coef0': [0.0],  # Single value default (only for poly/sigmoid kernels)
            'shrinking': [True],  # Single value default
            'note': 'Grid size: 9 configs (unchanged with single-value defaults)'
        },
        'comprehensive': {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],  # Explore all kernels
            'C': [0.1, 1.0, 10.0],  # 3 values
            'gamma': ['scale', 'auto'],  # 2 values for RBF/poly
            'epsilon': [0.01, 0.1, 0.2],  # 3 values
            'degree': [2, 3, 4],  # 3 values (only for poly)
            'coef0': [0.0, 1.0],  # 2 values (only for poly/sigmoid)
            'shrinking': [True, False],  # 2 values
            'note': 'Significantly expanded grid with conditional parameters'
        },
        'quick': {
            'kernel': ['rbf'],  # Single kernel
            'C': [1.0],  # 1 value
            'gamma': ['scale'],  # 1 value
            'epsilon': [0.1],  # Single value
            'degree': [3],  # Single value
            'coef0': [0.0],  # Single value
            'shrinking': [True],  # Single value
            'note': 'Grid size: 1 config - minimal exploration'
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
            'l2_leaf_reg': [3.0],  # Single value default (L2 regularization coefficient)
            'border_count': [128],  # Single value default (number of splits for numerical features)
            'bagging_temperature': [1.0],  # Single value default (controls intensity of Bayesian bagging)
            'random_strength': [1.0],  # Single value default (randomness for scoring splits)
            'note': 'Grid size: 3×3×3×1×1×1×1 = 27 configs (unchanged with single-value defaults)'
        },
        'comprehensive': {
            'iterations': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'depth': [4, 6, 8],  # 3 values
            'l2_leaf_reg': [1.0, 3.0, 5.0],  # 3 values - L2 regularization
            'border_count': [32, 128, 254],  # 3 values - feature split granularity
            'bagging_temperature': [0.0, 1.0, 10.0],  # 3 values - Bayesian bootstrap intensity
            'random_strength': [0.0, 1.0, 2.0],  # 3 values - split scoring randomness
            'note': 'Grid size: 3×3×3×3×3×3×3 = 2187 configs'
        },
        'quick': {
            'iterations': [50, 100, 200],  # 3 values
            'learning_rate': [0.05, 0.1, 0.2],  # 3 values
            'depth': [4, 6, 8],  # 3 values
            'l2_leaf_reg': [3.0],  # Single value
            'border_count': [128],  # Single value
            'bagging_temperature': [1.0],  # Single value
            'random_strength': [1.0],  # Single value
            'note': 'Grid size: 3×3×3×1×1×1×1 = 27 configs (unchanged with single-value defaults)'
        }
    },

    'RandomForest': {
        'standard': {
            'n_estimators': [100, 200, 500],  # 3 values
            'max_depth': [None, 15, 30],  # 3 values
            'min_samples_split': [2],  # sklearn default (single value)
            'min_samples_leaf': [1],  # sklearn default (single value)
            'max_features': ['sqrt'],  # sqrt(n_features) (single value)
            'bootstrap': [True],  # bootstrap sampling (single value)
            'max_leaf_nodes': [None],  # no limit (single value)
            'min_impurity_decrease': [0.0],  # no minimum (single value)
            'note': 'Grid size: 3×3×1×1×1×1×1×1 = 9 configs (unchanged with defaults)'
        },
        'comprehensive': {
            'n_estimators': [100, 200, 500],  # 3 values
            'max_depth': [None, 15, 30],  # 3 values
            'min_samples_split': [2, 5, 10],  # vary minimum samples for split
            'min_samples_leaf': [1, 2, 4],  # vary minimum leaf size
            'max_features': ['sqrt', 'log2', None],  # feature selection strategies
            'bootstrap': [True, False],  # with/without bootstrap
            'max_leaf_nodes': [None, 50, 100],  # limit tree complexity
            'min_impurity_decrease': [0.0, 0.01],  # pruning threshold
            'note': 'Grid size: 3×3×3×3×3×2×3×2 = 2916 configs (comprehensive exploration)'
        },
        'quick': {
            'n_estimators': [100, 200, 500],  # 3 values
            'max_depth': [None, 15, 30],  # 3 values
            'min_samples_split': [2],  # sklearn default (single value)
            'min_samples_leaf': [1],  # sklearn default (single value)
            'max_features': ['sqrt'],  # sqrt(n_features) (single value)
            'bootstrap': [True],  # bootstrap sampling (single value)
            'max_leaf_nodes': [None],  # no limit (single value)
            'min_impurity_decrease': [0.0],  # no minimum (single value)
            'note': 'Grid size: 3×3×1×1×1×1×1×1 = 9 configs (unchanged with defaults)'
        }
    },

    'MLP': {
        'standard': {
            'hidden_layer_sizes': [(64,), (128, 64)],  # 2 architectures
            'alpha': [1e-4, 1e-3],  # 2 values
            'learning_rate_init': [1e-3, 1e-2],  # 2 values
            'activation': ['relu'],  # Single value default
            'solver': ['adam'],  # Single value default
            'batch_size': ['auto'],  # Single value default
            'learning_rate': ['constant'],  # Single value default (schedule)
            'momentum': [0.9],  # Single value default (only used with sgd solver)
            'note': 'Grid size: 2×2×2×1×1×1×1 = 8 configs (unchanged with single-value defaults)'
        },
        'comprehensive': {
            'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32)],  # 3 architectures
            'alpha': [1e-5, 1e-4, 1e-3],  # 3 values
            'learning_rate_init': [1e-4, 1e-3, 1e-2],  # 3 values
            'activation': ['relu', 'tanh', 'logistic'],  # 3 activations
            'solver': ['adam', 'sgd'],  # 2 solvers
            'batch_size': ['auto', 32, 64],  # 3 batch sizes
            'learning_rate': ['constant', 'adaptive'],  # 2 schedules
            'momentum': [0.8, 0.9, 0.95],  # 3 values (only used with sgd solver)
            'note': 'Significantly expanded grid with conditional momentum'
        },
        'quick': {
            'hidden_layer_sizes': [(64,)],  # 1 architecture
            'alpha': [1e-3],  # 1 value
            'learning_rate_init': [1e-3],  # 1 value
            'activation': ['relu'],  # Single value
            'solver': ['adam'],  # Single value
            'batch_size': ['auto'],  # Single value
            'learning_rate': ['constant'],  # Single value
            'momentum': [0.9],  # Single value
            'note': 'Grid size: 1 config - minimal exploration'
        }
    },

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

# For classification, use similar structure but adapted
CLASSIFICATION_HYPERPARAMETERS = {
    'PLS-DA': OPTIMIZED_HYPERPARAMETERS['PLS'],  # Same as PLS
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
