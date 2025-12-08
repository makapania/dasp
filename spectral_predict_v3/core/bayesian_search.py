"""
Bayesian optimization for hyperparameter search using Optuna.

Provides an alternative to grid search with adaptive sampling that can
find good hyperparameters more efficiently, especially for large search spaces.

Key features:
- TPE (Tree-structured Parzen Estimator) sampler for efficient exploration
- Less aggressive pruning to avoid premature trial termination
- Log-scale sampling for learning rates and regularization parameters
- Progress callbacks and optimization history tracking
"""

import numpy as np
from typing import List, Optional, Callable, Dict, Any, Tuple
from enum import Enum
import warnings

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import PercentilePruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None
    TPESampler = None
    PercentilePruner = None

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold


class SearchMode(Enum):
    """Search mode for hyperparameter optimization."""
    GRID = "grid"
    BAYESIAN = "bayesian"


def suggest_hyperparams(trial, model_name: str, task_type: str = 'regression') -> Dict[str, Any]:
    """
    Suggest hyperparameters for a model using Optuna trial.

    Uses appropriate sampling strategies for each parameter type:
    - Log-scale for learning rates, alphas (regularization)
    - Categorical for discrete choices (kernels, activations)
    - Integer for counts (n_estimators, n_components)

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting parameters
    model_name : str
        Model name (e.g., 'PLS', 'Ridge', 'LightGBM')
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    params : dict
        Dictionary of suggested hyperparameters

    Notes
    -----
    The search spaces are designed to:
    1. Cover reasonable ranges for each parameter
    2. Use log-scale where parameters span multiple orders of magnitude
    3. Avoid known incompatible combinations
    4. Balance exploration vs exploitation
    """
    if model_name == 'PLS' or model_name == 'PLS-DA':
        return {
            'n_components': trial.suggest_int('n_components', 1, 20),
            'max_iter': trial.suggest_categorical('max_iter', [500, 1000, 2000]),
            'tol': trial.suggest_float('tol', 1e-8, 1e-4, log=True)
        }

    elif model_name == 'Ridge':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'max_iter': trial.suggest_int('max_iter', 1000, 10000)
        }

    elif model_name == 'Lasso':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'max_iter': trial.suggest_int('max_iter', 1000, 10000)
        }

    elif model_name == 'ElasticNet':
        return {
            'alpha': trial.suggest_float('alpha', 1e-3, 10, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'max_iter': trial.suggest_int('max_iter', 1000, 10000)
        }

    elif model_name == 'RandomForest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

    elif model_name == 'LightGBM':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 255),
            'max_depth': trial.suggest_int('max_depth', -1, 20),  # -1 = no limit
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10.0, log=True),
            'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
        }

    elif model_name == 'XGBoost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 5.0)
        }

    elif model_name == 'CatBoost':
        return {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 254]),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
            'one_hot_max_size': trial.suggest_categorical('one_hot_max_size', [2, 10, 25]),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
        }

    elif model_name == 'SVR' or model_name == 'SVM':
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        params = {
            'kernel': kernel,
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'cache_size': trial.suggest_categorical('cache_size', [200, 500, 1000])
        }
        # SVR-specific parameter
        if model_name == 'SVR':
            params['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.5)
        # Kernel-specific parameters
        if kernel == 'rbf':
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        elif kernel == 'poly':
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            params['degree'] = trial.suggest_int('degree', 2, 5)
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)
        return params

    elif model_name == 'MLP':
        # Suggest architecture
        n_layers = trial.suggest_int('n_layers', 1, 3)
        if n_layers == 1:
            hidden_layer_sizes = (trial.suggest_int('layer_1_size', 20, 150),)
        elif n_layers == 2:
            hidden_layer_sizes = (
                trial.suggest_int('layer_1_size', 50, 150),
                trial.suggest_int('layer_2_size', 20, 100)
            )
        else:  # 3 layers
            hidden_layer_sizes = (
                trial.suggest_int('layer_1_size', 50, 150),
                trial.suggest_int('layer_2_size', 30, 100),
                trial.suggest_int('layer_3_size', 20, 50)
            )

        # Solver selection affects which parameters are relevant
        solver = trial.suggest_categorical('solver', ['adam', 'sgd'])

        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 0.1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 0.1, log=True),
            'solver': solver,
            'batch_size': trial.suggest_categorical('batch_size', ['auto', 32, 64, 128]),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.2),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20)
        }

        # Solver-specific parameters
        if solver == 'sgd':
            params['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
        elif solver == 'adam':
            params['beta_1'] = trial.suggest_float('beta_1', 0.8, 0.999)
            params['beta_2'] = trial.suggest_float('beta_2', 0.9, 0.9999)

        return params

    elif model_name == 'NeuralBoosted':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'hidden_layer_size': trial.suggest_int('hidden_layer_size', 3, 10),
            'activation': trial.suggest_categorical('activation', ['tanh', 'relu', 'identity']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 15)
        }

    else:
        # Unknown model - return empty params
        return {}


def suggest_variable_selection(
    trial,
    include_varsel: bool,
    varsel_methods: List[str],
    varsel_counts: List[int]
) -> Tuple[str, Optional[int]]:
    """
    Suggest variable selection parameters for a Bayesian optimization trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting parameters
    include_varsel : bool
        Whether to include variable selection in the search space
    varsel_methods : list of str
        Available variable selection methods (e.g., ['importance', 'vip', 'spa'])
    varsel_counts : list of int
        Available variable counts to test (e.g., [20, 50, 100, 250])

    Returns
    -------
    method : str
        Selected method ('full' means use all variables)
    n_vars : int or None
        Number of variables to select (None if method is 'full')
    """
    if not include_varsel or not varsel_methods:
        return 'full', None

    # Always include 'full' as an option (use all variables)
    methods_with_full = ['full'] + list(varsel_methods)
    method = trial.suggest_categorical('varsel_method', methods_with_full)

    if method == 'full':
        return 'full', None

    # Suggest variable count only if not 'full'
    n_vars = trial.suggest_categorical('varsel_n_vars', varsel_counts)

    return method, n_vars


def compute_variable_importances(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task_type: str,
    random_state: int,
    method_params: Optional[Dict[str, Any]] = None
) -> Optional[np.ndarray]:
    """
    Compute variable importances using the specified method.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    method : str
        Variable selection method: 'importance', 'vip', 'spa', 'uve', 'uve_spa'
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random seed for reproducibility
    method_params : dict, optional
        Method-specific parameters

    Returns
    -------
    importances : np.ndarray or None
        Importance scores for each variable (higher = more important).
        Returns None if computation fails.
    """
    method_params = method_params or {}
    n_features = X.shape[1]

    try:
        if method == 'importance':
            # RandomForest feature importance
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            if task_type == 'regression':
                rf = RandomForestRegressor(
                    n_estimators=50,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                rf = RandomForestClassifier(
                    n_estimators=50,
                    random_state=random_state,
                    n_jobs=-1
                )
            rf.fit(X, y)
            return rf.feature_importances_

        elif method == 'vip':
            # PLS Variable Importance in Projection
            from .variable_selection import compute_vip
            return compute_vip(X, y)

        elif method == 'spa':
            # Successive Projections Algorithm
            from .variable_selection import spa_selection
            params = method_params.get('spa', {})
            n_spa_features = params.get('n_features', min(100, n_features))
            n_random_starts = params.get('n_random_starts', 5)  # Fewer for speed
            return spa_selection(
                X, y,
                n_features=n_spa_features,
                n_random_starts=n_random_starts,
                random_state=random_state
            )

        elif method == 'uve':
            # Uninformative Variable Elimination
            from .variable_selection import uve_selection
            params = method_params.get('uve', {})
            cutoff = params.get('cutoff_multiplier', 1.0)
            return uve_selection(
                X, y,
                cutoff_multiplier=cutoff,
                random_state=random_state
            )

        elif method == 'uve_spa':
            # UVE-SPA Hybrid
            from .variable_selection import uve_spa_selection
            params = method_params.get('uve_spa', {})
            cutoff = params.get('cutoff_multiplier', 1.0)
            n_spa_features = params.get('n_features', min(100, n_features))
            return uve_spa_selection(
                X, y,
                n_features=n_spa_features,
                cutoff_multiplier=cutoff,
                random_state=random_state
            )

        else:
            # Unknown method
            print(f"Warning: Unknown variable selection method '{method}'")
            return None

    except Exception as e:
        print(f"Variable selection method '{method}' failed: {e}")
        return None


def run_bayesian_search(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = 'regression',
    models_to_try: Optional[List[str]] = None,
    preprocess_configs: Optional[List[Tuple[str, Callable]]] = None,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    optuna_sampler: str = 'tpe',
    pruning_percentile: float = 25.0,
    # Variable selection parameters
    include_variable_selection: bool = False,
    varsel_methods: Optional[List[str]] = None,
    varsel_counts: Optional[List[int]] = None,
    varsel_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[Dict[str, Any], Any]:
    """
    Run Bayesian hyperparameter optimization using Optuna.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data, shape (n_samples, n_features)
    y : np.ndarray
        Target values
    task_type : str
        'regression' or 'classification'
    models_to_try : list of str, optional
        Model names to optimize. If None, uses ['PLS', 'Ridge', 'RandomForest']
    preprocess_configs : list of tuple, optional
        List of (name, transform_func) preprocessing configurations.
        If None, uses raw data only.
    n_trials : int, default=50
        Number of Optuna trials to run
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
    progress_callback : callable, optional
        Callback function called after each trial with progress info
    optuna_sampler : str, default='tpe'
        Optuna sampler: 'tpe', 'random', or 'cmaes'
    pruning_percentile : float, default=25.0
        Percentile for pruning (lower = more aggressive).
        25.0 means prune if worse than 25th percentile of previous trials.
    include_variable_selection : bool, default=False
        If True, includes variable selection method as part of the unified
        Bayesian search space. This allows joint optimization of preprocessing,
        variable selection, model, and hyperparameters.
    varsel_methods : list of str, optional
        Variable selection methods to include in search space.
        Options: 'importance' (RF), 'vip' (PLS), 'spa', 'uve', 'uve_spa'
        If None and include_variable_selection is True, defaults to
        ['importance', 'vip'].
    varsel_counts : list of int, optional
        Number of variables to test. If None, defaults to [20, 50, 100, 250].
        Values that exceed n_features will be automatically filtered.
    varsel_params : dict, optional
        Method-specific parameters for variable selection methods.
        E.g., {'spa': {'n_random_starts': 5}, 'uve': {'cutoff_multiplier': 1.0}}

    Returns
    -------
    best_config : dict
        Best configuration found with keys:
        - 'model': model name
        - 'preprocessing': preprocessing name
        - 'params': hyperparameters
        - 'score': cross-validation score
        - 'trial_number': trial number of best config
        - 'variable_selection': dict with 'method' and 'n_vars' keys
          (only present if include_variable_selection=True)

    study : optuna.Study
        Optuna study object containing optimization history

    Raises
    ------
    ImportError
        If optuna is not installed

    Notes
    -----
    This function addresses issues from v1 implementation:
    1. Uses less aggressive pruning (PercentilePruner at 25th percentile
       instead of MedianPruner) to avoid premature trial termination
    2. Log-scale sampling for learning rates and alphas
    3. Returns study object for optimization history analysis
    4. Broader search ranges for better exploration

    The TPE sampler is generally most effective for:
    - Mixed parameter types (continuous, discrete, categorical)
    - Medium-sized search spaces (10-100 trials)
    - When some parameters are more important than others

    Examples
    --------
    >>> from spectral_predict_v3.core.bayesian_search import run_bayesian_search
    >>> best_config, study = run_bayesian_search(
    ...     X, y,
    ...     task_type='regression',
    ...     models_to_try=['PLS', 'Ridge'],
    ...     n_trials=30,
    ...     cv_folds=5
    ... )
    >>> print(f"Best model: {best_config['model']}")
    >>> print(f"Best score: {best_config['score']:.4f}")
    """
    if not HAS_OPTUNA:
        raise ImportError(
            "Optuna is required for Bayesian search. "
            "Install with: pip install optuna"
        )

    # Default models if not specified
    if models_to_try is None:
        if task_type == 'regression':
            models_to_try = ['PLS', 'Ridge', 'RandomForest']
        else:
            models_to_try = ['PLS-DA', 'RandomForest']

    # Default preprocessing (raw data only)
    if preprocess_configs is None:
        preprocess_configs = [('raw', None)]

    # Default variable selection settings - ONLY set defaults when varsel is enabled
    if varsel_params is None:
        varsel_params = {}

    if include_variable_selection:
        # Set defaults only when variable selection is enabled
        if varsel_methods is None:
            varsel_methods = ['importance', 'vip']  # Conservative default
        if varsel_counts is None:
            varsel_counts = [20, 50, 100, 250]
    else:
        # Ensure no variable selection when disabled (backward compatibility)
        varsel_methods = []
        varsel_counts = []

    # Filter varsel_counts to valid values (must be less than n_features)
    n_features = X.shape[1]
    varsel_counts = [c for c in varsel_counts if c < n_features]
    if include_variable_selection and not varsel_counts:
        print(f"Warning: All variable counts >= n_features ({n_features}). "
              "Disabling variable selection.")
        include_variable_selection = False

    # Initialize importance cache for this study (shared across trials)
    # Key: (preprocessing_name, method) -> importances array
    importance_cache = {}

    # Set up cross-validation
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'accuracy'
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'neg_mean_squared_error'

    # Choose sampler
    if optuna_sampler == 'tpe':
        sampler = TPESampler(seed=random_state)
    elif optuna_sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=random_state)
    elif optuna_sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=random_state)
    else:
        sampler = TPESampler(seed=random_state)

    # Choose pruner (less aggressive than MedianPruner)
    pruner = PercentilePruner(
        percentile=pruning_percentile,
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=0
    )

    # Create study
    direction = 'maximize' if task_type == 'classification' else 'minimize'
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner
    )

    # Track best score
    best_score = None
    if task_type == 'regression':
        best_score = float('inf')  # Lower is better (RMSE)
    else:
        best_score = 0.0  # Higher is better (accuracy)

    # Define objective function
    def objective(trial):
        nonlocal best_score

        # Suggest model
        model_name = trial.suggest_categorical('model', models_to_try)

        # Suggest preprocessing
        preproc_names = [name for name, _ in preprocess_configs]
        preproc_name = trial.suggest_categorical('preprocessing', preproc_names)

        # Get preprocessing function
        preproc_idx = preproc_names.index(preproc_name)
        _, preproc_func = preprocess_configs[preproc_idx]

        # Apply preprocessing
        try:
            X_processed = preproc_func(X) if preproc_func else X
        except Exception as e:
            # Preprocessing failed - skip this trial
            if task_type == 'regression':
                return float('inf')
            else:
                return 0.0

        # Suggest variable selection (if enabled)
        varsel_method, varsel_n_vars = suggest_variable_selection(
            trial, include_variable_selection, varsel_methods, varsel_counts
        )

        # Apply variable selection if not 'full'
        variable_tag = 'full'
        if varsel_method != 'full' and varsel_n_vars is not None:
            # Check cache first
            cache_key = (preproc_name, varsel_method)

            if cache_key not in importance_cache:
                # Compute and cache importances
                importances = compute_variable_importances(
                    X_processed, y, varsel_method, task_type,
                    random_state, varsel_params
                )
                importance_cache[cache_key] = importances
            else:
                importances = importance_cache[cache_key]

            if importances is not None:
                # Select top N variables
                n_to_select = min(varsel_n_vars, len(importances))
                selected_indices = np.argsort(importances)[-n_to_select:]
                X_processed = X_processed[:, selected_indices]
                variable_tag = f'{varsel_method}_top{n_to_select}'
            else:
                # Fallback to full if importance calculation failed
                variable_tag = 'full (varsel failed)'

        # Suggest hyperparameters for this model
        params = suggest_hyperparams(trial, model_name, task_type)

        # Adjust PLS n_components if needed after variable selection
        if model_name in ['PLS', 'PLS-DA'] and 'n_components' in params:
            n_available_features = X_processed.shape[1]
            n_samples = X_processed.shape[0]
            max_components = min(n_available_features, n_samples) - 1
            if params['n_components'] > max_components:
                params['n_components'] = max(1, max_components)

        # Create model
        from .models import get_model
        try:
            model = get_model(model_name, task_type, random_state, **params)
            if model is None:
                # Model not available (e.g., LightGBM not installed)
                if task_type == 'regression':
                    return float('inf')
                else:
                    return 0.0
        except Exception as e:
            # Model creation failed
            if task_type == 'regression':
                return float('inf')
            else:
                return 0.0

        # Cross-validate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    model, X_processed, y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1  # Avoid nested parallelism
                )

            # Compute metric
            if task_type == 'regression':
                # scoring is 'neg_mean_squared_error', convert to RMSE
                rmse = np.sqrt(-scores.mean())
                score = rmse

                # Update best score
                if rmse < best_score:
                    best_score = rmse
            else:
                # scoring is 'accuracy'
                accuracy = scores.mean()
                score = accuracy

                # Update best score
                if accuracy > best_score:
                    best_score = accuracy

            # Progress callback
            if progress_callback:
                # Include variable selection in message if not 'full'
                if variable_tag == 'full':
                    msg = f'{preproc_name} + {model_name}'
                else:
                    msg = f'{preproc_name} + {model_name} ({variable_tag})'

                progress_callback({
                    'stage': 'bayesian_search',
                    'message': msg,
                    'current': trial.number + 1,
                    'total': n_trials,
                    'best_score': best_score,
                    'current_score': score
                })

            return score

        except Exception as e:
            # Cross-validation failed
            print(f"Trial {trial.number} failed: {e}")
            if task_type == 'regression':
                return float('inf')
            else:
                return 0.0

    # Run optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract best configuration
    best_trial = study.best_trial

    # Build best config dict
    best_config = {
        'model': best_trial.params['model'],
        'preprocessing': best_trial.params['preprocessing'],
        'params': {},
        'score': best_trial.value,
        'trial_number': best_trial.number
    }

    # Add variable selection info if enabled
    if include_variable_selection:
        best_config['variable_selection'] = {
            'method': best_trial.params.get('varsel_method', 'full'),
            'n_vars': best_trial.params.get('varsel_n_vars', None)
        }

    # Keys to exclude from hyperparameters
    exclude_keys = ['model', 'preprocessing', 'n_layers', 'varsel_method', 'varsel_n_vars']

    # Extract hyperparameters (exclude model, preprocessing, and varsel params)
    for key, value in best_trial.params.items():
        if key not in exclude_keys:
            # For MLP, combine layer sizes into tuple
            if best_config['model'] == 'MLP' and 'layer' in key:
                continue  # Will be handled separately
            else:
                best_config['params'][key] = value

    # Special handling for MLP hidden_layer_sizes
    if best_config['model'] == 'MLP':
        layer_sizes = []
        n_layers = best_trial.params.get('n_layers', 1)
        for i in range(1, n_layers + 1):
            layer_key = f'layer_{i}_size'
            if layer_key in best_trial.params:
                layer_sizes.append(best_trial.params[layer_key])
        if layer_sizes:
            best_config['params']['hidden_layer_sizes'] = tuple(layer_sizes)

    return best_config, study


def get_optimization_history(study) -> Dict[str, Any]:
    """
    Extract optimization history from Optuna study.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study

    Returns
    -------
    history : dict
        Dictionary with keys:
        - 'trial_numbers': list of trial numbers
        - 'values': list of objective values
        - 'best_values': list of best-so-far values
        - 'params': list of parameter dictionaries
        - 'states': list of trial states

    Examples
    --------
    >>> best_config, study = run_bayesian_search(X, y, n_trials=30)
    >>> history = get_optimization_history(study)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(history['trial_numbers'], history['best_values'])
    >>> plt.xlabel('Trial')
    >>> plt.ylabel('Best Score')
    >>> plt.show()
    """
    if study is None or not HAS_OPTUNA:
        return {}

    trials = study.trials

    history = {
        'trial_numbers': [t.number for t in trials],
        'values': [t.value if t.value is not None else float('nan') for t in trials],
        'best_values': [],
        'params': [t.params for t in trials],
        'states': [t.state.name for t in trials]
    }

    # Compute best-so-far
    if study.direction == optuna.study.StudyDirection.MINIMIZE:
        best_so_far = float('inf')
        for val in history['values']:
            if not np.isnan(val) and val < best_so_far:
                best_so_far = val
            history['best_values'].append(best_so_far)
    else:  # MAXIMIZE
        best_so_far = -float('inf')
        for val in history['values']:
            if not np.isnan(val) and val > best_so_far:
                best_so_far = val
            history['best_values'].append(best_so_far)

    return history
