"""
Bayesian hyperparameter optimization search spaces for Optuna.

This module defines continuous search spaces for Bayesian optimization using Optuna.
These spaces are BROADER and CONTINUOUS compared to grid search, allowing
Optuna to explore between grid points and find better hyperparameters.

Key Advantages:
    - Continuous distributions: Can find learning_rate=0.127 instead of just 0.05/0.1/0.2
    - Logarithmic scales: Better exploration of learning rates and regularization
    - Broader ranges: Based on literature and empirical best practices

Integration:
    - Compatible with existing tier system
    - Works with all 11 model types
    - Returns sklearn/xgboost/lightgbm compatible parameter dictionaries
"""

import optuna
from typing import Dict, Any, Optional


def get_bayesian_search_space(
    model_name: str,
    trial: optuna.Trial,
    tier: str = 'standard',
    n_features: int = None,
    max_n_components: int = 8,
    task_type: str = 'regression',
    n_classes: int = 2
) -> Dict[str, Any]:
    """
    Define Optuna search space for a specific model.

    Parameters
    ----------
    model_name : str
        Name of the model ('PLS', 'XGBoost', 'LightGBM', etc.)
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters
    tier : str, default='standard'
        Tier level - affects range of hyperparameters
    n_features : int, optional
        Number of input features (used to constrain PLS components)
    max_n_components : int, default=8
        Maximum PLS components (constrained by dataset size)
    task_type : str, default='regression'
        'regression' or 'classification'
    n_classes : int, default=2
        Number of classes (for classification tasks)

    Returns
    -------
    params : dict
        Dictionary of hyperparameters compatible with sklearn/xgboost/lightgbm

    Notes
    -----
    Search spaces are CONTINUOUS and BROADER than grid search:
        - Grid: learning_rate=[0.05, 0.1, 0.2] (3 values)
        - Bayesian: learning_rate=log-uniform(0.01, 0.3) (infinite values)

    This allows Optuna to find better parameters (e.g., 0.127) that aren't in the grid.
    """
    if model_name == 'PLS' or model_name == 'PLS-DA':
        return _get_pls_space(trial, max_n_components)

    elif model_name == 'Ridge':
        return _get_ridge_space(trial, tier)

    elif model_name == 'Lasso':
        return _get_lasso_space(trial, tier)

    elif model_name == 'ElasticNet':
        return _get_elasticnet_space(trial, tier)

    elif model_name == 'RandomForest':
        return _get_randomforest_space(trial, tier, n_features)

    elif model_name == 'XGBoost':
        return _get_xgboost_space(trial, tier)

    elif model_name == 'LightGBM':
        return _get_lightgbm_space(trial, tier, task_type, n_classes)

    elif model_name == 'CatBoost':
        return _get_catboost_space(trial, tier)

    elif model_name == 'SVR' or model_name == 'SVM':
        return _get_svr_space(trial, tier)

    elif model_name == 'MLP':
        return _get_mlp_space(trial, tier)

    elif model_name == 'NeuralBoosted':
        return _get_neuralboosted_space(trial, tier)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Individual Model Search Spaces
# ============================================================================

def _get_pls_space(trial: optuna.Trial, max_n_components: int = 8) -> Dict:
    """PLS/PLS-DA search space."""
    return {
        'n_components': trial.suggest_int('n_components', 2, max_n_components),
        'max_iter': trial.suggest_categorical('max_iter', [500, 1000, 2000]),
        'tol': trial.suggest_float('tol', 1e-8, 1e-4, log=True)
    }


def _get_ridge_space(trial: optuna.Trial, tier: str) -> Dict:
    """Ridge Regression search space."""
    if tier == 'quick':
        alpha_range = (0.01, 10.0)
    else:
        alpha_range = (0.001, 100.0)

    return {
        'alpha': trial.suggest_float('alpha', *alpha_range, log=True),
        'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
    }


def _get_lasso_space(trial: optuna.Trial, tier: str) -> Dict:
    """Lasso Regression search space."""
    if tier == 'quick':
        alpha_range = (0.01, 10.0)
    else:
        alpha_range = (0.001, 100.0)

    return {
        'alpha': trial.suggest_float('alpha', *alpha_range, log=True),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
    }


def _get_elasticnet_space(trial: optuna.Trial, tier: str) -> Dict:
    """ElasticNet search space."""
    if tier == 'quick':
        alpha_range = (0.01, 10.0)
    else:
        alpha_range = (0.001, 100.0)

    return {
        'alpha': trial.suggest_float('alpha', *alpha_range, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
    }


def _get_randomforest_space(trial: optuna.Trial, tier: str, n_features: int = None) -> Dict:
    """Random Forest search space."""
    if tier == 'quick':
        n_estimators_range = (50, 300)
        max_depth_options = [None, 20, 30]
    elif tier == 'comprehensive':
        n_estimators_range = (50, 500)
        max_depth_options = [None, 10, 20, 30, 50]
    else:  # standard
        n_estimators_range = (100, 300)
        max_depth_options = [None, 20, 30]

    params = {
        'n_estimators': trial.suggest_int('n_estimators', *n_estimators_range),
        'max_depth': trial.suggest_categorical('max_depth', max_depth_options),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # max_features: Use sqrt for high-dimensional spectral data
    if n_features and n_features > 100:
        params['max_features'] = 'sqrt'  # Fixed for spectral data
    else:
        params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    return params


def _get_xgboost_space(trial: optuna.Trial, tier: str) -> Dict:
    """XGBoost search space.

    Grid search: 2×2×2×1×1×1×1 = 8 configs
    Bayesian: Continuous space with broader ranges
    """
    if tier == 'quick':
        n_estimators_range = (50, 200)
        max_depth_range = (3, 6)
    elif tier == 'comprehensive':
        n_estimators_range = (50, 500)
        max_depth_range = (3, 12)
    else:  # standard
        n_estimators_range = (50, 300)
        max_depth_range = (3, 9)

    return {
        'n_estimators': trial.suggest_int('n_estimators', *n_estimators_range),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', *max_depth_range),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0)
    }


def _get_lightgbm_space(trial: optuna.Trial, tier: str, task_type: str, n_classes: int) -> Dict:
    """LightGBM search space.

    Grid search: 2×1×2×1×1×1×1×1×1 = 4 configs
    Bayesian: Continuous space with broader ranges
    """
    if tier == 'quick':
        n_estimators_range = (50, 200)
        num_leaves_range = (15, 63)
    elif tier == 'comprehensive':
        n_estimators_range = (50, 500)
        num_leaves_range = (7, 255)
    else:  # standard
        n_estimators_range = (50, 300)
        # Adapt num_leaves to task complexity
        if task_type == 'classification' and n_classes > 2:
            num_leaves_range = (15, 127)  # Multiclass: more complex
        else:
            num_leaves_range = (15, 63)  # Binary/regression: simpler

    return {
        'n_estimators': trial.suggest_int('n_estimators', *n_estimators_range),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', *num_leaves_range),
        'max_depth': trial.suggest_int('max_depth', -1, 20),  # -1 = no limit
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0)
    }


def _get_catboost_space(trial: optuna.Trial, tier: str) -> Dict:
    """CatBoost search space.

    Grid search: 2×2×1×1×1×1×1 = 4 configs
    Bayesian: Continuous space with broader ranges
    """
    if tier == 'quick':
        iterations_range = (50, 200)
        depth_range = (4, 6)
    elif tier == 'comprehensive':
        iterations_range = (50, 500)
        depth_range = (4, 10)
    else:  # standard
        iterations_range = (50, 300)
        depth_range = (4, 8)

    return {
        'iterations': trial.suggest_int('iterations', *iterations_range),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', *depth_range),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 254]),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0)
    }


def _get_svr_space(trial: optuna.Trial, tier: str) -> Dict:
    """SVR/SVM search space."""
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])

    params = {
        'kernel': kernel,
        'C': trial.suggest_float('C', 0.01, 100.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 0.5)
    }

    # Kernel-specific parameters
    if kernel in ['rbf', 'poly']:
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])

    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
        params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

    return params


def _get_mlp_space(trial: optuna.Trial, tier: str) -> Dict:
    """MLP (Multi-Layer Perceptron) search space."""
    # Hidden layer configurations
    n_layers = trial.suggest_int('n_layers', 1, 3)

    if n_layers == 1:
        hidden_layer_sizes = (trial.suggest_int('layer1_size', 32, 256),)
    elif n_layers == 2:
        hidden_layer_sizes = (
            trial.suggest_int('layer1_size', 64, 256),
            trial.suggest_int('layer2_size', 32, 128)
        )
    else:  # 3 layers
        hidden_layer_sizes = (
            trial.suggest_int('layer1_size', 128, 256),
            trial.suggest_int('layer2_size', 64, 128),
            trial.suggest_int('layer3_size', 32, 64)
        )

    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'batch_size': trial.suggest_categorical('batch_size', ['auto', 32, 64, 128]),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'momentum': trial.suggest_float('momentum', 0.5, 0.99) if trial.params.get('solver') == 'sgd' else 0.9
    }


def _get_neuralboosted_space(trial: optuna.Trial, tier: str) -> Dict:
    """NeuralBoosted search space."""
    if tier == 'quick':
        n_estimators_range = (30, 100)
    elif tier == 'comprehensive':
        n_estimators_range = (50, 200)
    else:  # standard
        n_estimators_range = (50, 150)

    return {
        'n_estimators': trial.suggest_int('n_estimators', *n_estimators_range),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5, log=True),
        'hidden_layer_size': trial.suggest_int('hidden_layer_size', 3, 10),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu', 'identity']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'early_stopping': True,  # Always use for speed
        'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 15)
    }


# ============================================================================
# Utility Functions
# ============================================================================

def validate_params(model_name: str, params: Dict, n_features: int = None) -> bool:
    """
    Validate hyperparameters before training.

    Parameters
    ----------
    model_name : str
        Model name
    params : dict
        Hyperparameters to validate
    n_features : int, optional
        Number of features (for validation)

    Returns
    -------
    valid : bool
        True if parameters are valid, False otherwise
    """
    # PLS: n_components must be <= n_features
    if model_name in ['PLS', 'PLS-DA']:
        if n_features and params.get('n_components', 0) > n_features:
            return False

    # RandomForest: max_depth must be positive if not None
    if model_name == 'RandomForest':
        max_depth = params.get('max_depth')
        if max_depth is not None and max_depth < 1:
            return False

    # LightGBM: num_leaves must be < 2^max_depth if max_depth != -1
    if model_name == 'LightGBM':
        max_depth = params.get('max_depth', -1)
        num_leaves = params.get('num_leaves', 31)
        if max_depth != -1 and num_leaves >= 2**max_depth:
            return False

    return True


def get_search_space_summary(model_name: str, tier: str = 'standard') -> str:
    """
    Get human-readable summary of search space for a model.

    Parameters
    ----------
    model_name : str
        Model name
    tier : str
        Tier level

    Returns
    -------
    summary : str
        Formatted summary of search space
    """
    # Create a dummy trial to inspect ranges
    study = optuna.create_study()
    trial = study.ask()

    try:
        params = get_bayesian_search_space(model_name, trial, tier=tier)

        summary = f"Bayesian Search Space for {model_name} (tier={tier}):\n"
        summary += "=" * 60 + "\n"

        for param_name, param_value in params.items():
            summary += f"  {param_name}: {type(param_value).__name__}\n"

        return summary

    except Exception as e:
        return f"Error getting search space: {e}"


if __name__ == '__main__':
    # Example usage
    print("Bayesian Optimization Search Spaces")
    print("=" * 70)

    models = ['PLS', 'Ridge', 'XGBoost', 'LightGBM', 'NeuralBoosted']

    for model in models:
        print(f"\n{model}:")
        print("-" * 70)

        study = optuna.create_study()
        trial = study.ask()

        try:
            params = get_bayesian_search_space(model, trial, tier='standard')
            for key, value in params.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
