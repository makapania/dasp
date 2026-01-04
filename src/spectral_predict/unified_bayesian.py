"""Unified Bayesian Optimization for Spectral Modeling.

This module implements TRUE joint optimization of:
- Preprocessing (SNV, derivatives, window sizes)
- Model hyperparameters
- Variable selection method (importance, CARS, regional)
- Subset size

Key Design Principle:
    TPE optimizes the EXACT configuration that produces the final result.
    The SUBSET score is returned to TPE, not full model score.
    This gives TPE a clean learning signal about what actually works.

Why This Beats Grid Search:
    1. Continuous hyperparameters: TPE finds alpha=0.127 instead of grid's [0.1, 0.2]
    2. Joint optimization: TPE learns preprocessing + hyperparams + subsets together
    3. Smarter exploration: 300 trials focus on promising regions
    4. More window sizes: TPE explores 5, 7, 9, ..., 51 vs Grid's [7, 19]
    5. 140,000+ config space intelligently explored vs Grid's ~3000 exhaustively

Example:
    >>> from spectral_predict.unified_bayesian import run_unified_bayesian
    >>> results_df, study = run_unified_bayesian(
    ...     X, y, wavelengths,
    ...     model_name='PLS',
    ...     n_trials=300,
    ...     cv_folds=5
    ... )
    >>> print(f"Best RMSE: {results_df['RMSE'].min():.4f}")
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from typing import Dict, List, Optional, Callable, Tuple, Any

# Import existing infrastructure
from spectral_predict.preprocess import SNV, SavgolDerivative
from spectral_predict.models import build_model, get_feature_importances
from spectral_predict.regions import create_region_subsets
from spectral_predict.variable_selection import (
    spa_selection, uve_selection, cars_selection
)

# Suppress Optuna verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Preprocessing options
PREPROCESSING_OPTIONS = [
    'raw', 'snv',
    'deriv1', 'deriv2', 'deriv3', 'deriv4',
    'snv_deriv1', 'snv_deriv2', 'snv_deriv3', 'snv_deriv4',
    'deriv1_snv', 'deriv2_snv', 'deriv3_snv', 'deriv4_snv'
]

# Subset sizes to explore
SUBSET_SIZES = ['full', 10, 20, 50, 100, 250, 500, 1000]

# Variable selection methods
VAR_METHODS = ['importance', 'cars', 'region']


def suggest_preprocessing(trial: Trial, n_features: int) -> Dict[str, Any]:
    """Suggest preprocessing configuration.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    n_features : int
        Number of spectral features (wavelengths)

    Returns
    -------
    config : dict
        Preprocessing configuration with keys:
        - 'name': Preprocessing name (e.g., 'snv_deriv1')
        - 'deriv': Derivative order (0-4)
        - 'window': Savitzky-Golay window size
        - 'polyorder': Polynomial order
    """
    preprocessing = trial.suggest_categorical('preprocessing', PREPROCESSING_OPTIONS)

    config = {
        'name': preprocessing,
        'deriv': 0,
        'window': 0,
        'polyorder': 0
    }

    if 'deriv' in preprocessing:
        # Extract derivative order
        for i in range(4, 0, -1):
            if f'deriv{i}' in preprocessing:
                config['deriv'] = i
                break

        # Suggest window size (must be odd)
        max_window = min(51, n_features - 1)
        if max_window < 5:
            max_window = 5
        config['window'] = trial.suggest_int('savgol_window', 5, max_window, step=2)

        # Polyorder must be less than window and >= deriv order
        config['polyorder'] = config['deriv'] + 1

        # Ensure window >= polyorder + 2
        if config['window'] < config['polyorder'] + 2:
            config['window'] = config['polyorder'] + 2
            if config['window'] % 2 == 0:
                config['window'] += 1

    return config


def apply_preprocessing(X: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Apply preprocessing to spectral data.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_wavelengths)
    config : dict
        Preprocessing configuration from suggest_preprocessing

    Returns
    -------
    X_processed : np.ndarray
        Preprocessed data
    """
    name = config['name']

    if name == 'raw':
        return X.copy()

    if name == 'snv':
        snv = SNV()
        return snv.fit_transform(X)

    # Handle derivatives
    if 'deriv' in name:
        deriv_order = config['deriv']
        window = config['window']
        polyorder = config['polyorder']

        # Ensure valid parameters
        if window < polyorder + 2:
            window = polyorder + 2
            if window % 2 == 0:
                window += 1

        savgol = SavgolDerivative(deriv=deriv_order, window=window, polyorder=polyorder)

        if name.startswith('snv_deriv'):
            # SNV then derivative
            snv = SNV()
            X_snv = snv.fit_transform(X)
            return savgol.fit_transform(X_snv)
        elif name.startswith('deriv') and '_snv' in name:
            # Derivative then SNV
            X_deriv = savgol.fit_transform(X)
            snv = SNV()
            return snv.fit_transform(X_deriv)
        else:
            # Just derivative
            return savgol.fit_transform(X)

    return X.copy()


def suggest_model_params(
    trial: Trial,
    model_name: str,
    n_features: int,
    task_type: str = 'regression'
) -> Dict[str, Any]:
    """Suggest model hyperparameters (focused set).

    Only optimize parameters that MATTER. Fix convergence params.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    model_name : str
        Model name
    n_features : int
        Number of features (after preprocessing and subsetting)
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    params : dict
        Model hyperparameters
    """
    model_name_lower = model_name.lower()

    if model_name_lower in ('pls', 'pls-da'):
        # PLS: only n_components matters
        # IMPORTANT: Use FIXED range to avoid Optuna's dynamic space error
        # Always suggest from full range, then clamp to actual max_components
        n_components = trial.suggest_int('n_components', 2, 20)
        # Clamp to valid range based on actual feature count
        max_valid = min(20, n_features - 1)
        if max_valid < 2:
            max_valid = 2
        n_components = min(n_components, max_valid)
        return {
            'n_components': n_components,
            'max_iter': 500,
            'tol': 1e-6
        }

    elif model_name_lower == 'ridge':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
            'solver': 'auto',
            'tol': 1e-6,
            'max_iter': 10000
        }

    elif model_name_lower == 'lasso':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
            'max_iter': 10000,
            'tol': 1e-6
        }

    elif model_name_lower == 'elasticnet':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
            'max_iter': 10000,
            'tol': 1e-6
        }

    elif model_name_lower in ('randomforest', 'rf'):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': 'sqrt'
        }

    elif model_name_lower == 'lightgbm':
        # Focus on 4 key params
        max_depth = trial.suggest_int('max_depth', -1, 15)

        # Constrain num_leaves based on max_depth
        if max_depth == -1:
            num_leaves = trial.suggest_int('num_leaves', 15, 127)
        else:
            max_valid = min(2**max_depth - 1, 127)
            if max_valid < 15:
                num_leaves = max_valid
            else:
                num_leaves = trial.suggest_int('num_leaves', 15, max_valid)

        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'min_child_samples': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbosity': -1,
            'n_jobs': 1
        }

    elif model_name_lower == 'xgboost':
        # Focus on 4 key params
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbosity': 0,
            'n_jobs': 1
        }

    elif model_name_lower == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'verbose': False
        }

    elif model_name_lower in ('svr', 'svm'):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'kernel': kernel,
            'max_iter': 10000
        }
        if task_type == 'regression':
            params['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.5)
        if kernel == 'rbf':
            params['gamma'] = 'scale'
        return params

    elif model_name_lower == 'mlp':
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        n_layers = trial.suggest_int('n_layers', 1, 3)

        if n_layers == 1:
            hidden_layer_sizes = (hidden_size,)
        elif n_layers == 2:
            hidden_layer_sizes = (hidden_size, hidden_size // 2)
        else:
            hidden_layer_sizes = (hidden_size, hidden_size // 2, hidden_size // 4)

        return {
            'hidden_layer_sizes': hidden_layer_sizes,
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter': 1000
        }

    else:
        # Default minimal params
        return {}


def compute_importances(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    model_name: str,
    cv_folds: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """Compute feature importances using specified method.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed spectral data
    y : np.ndarray
        Target values
    method : str
        'importance', 'cars', or 'region' (region handled separately)
    model_name : str
        Model name (for importance-based methods)
    cv_folds : int
        Number of CV folds for variable selection methods
    random_state : int
        Random seed

    Returns
    -------
    importances : np.ndarray
        Importance scores for each feature
    """
    n_features = X.shape[1]

    if method == 'importance':
        # Quick model fit to get importances
        from spectral_predict.models import build_model

        # Use simple params for quick importance estimation
        if model_name.lower() in ('pls', 'pls-da'):
            params = {'n_components': min(5, n_features - 1)}
        elif model_name.lower() in ('ridge', 'lasso', 'elasticnet'):
            params = {'alpha': 1.0}
        elif model_name.lower() in ('lightgbm', 'xgboost', 'randomforest', 'rf'):
            params = {'n_estimators': 50, 'max_depth': 5}
        else:
            params = {}

        model = build_model(model_name, params, task_type='regression')
        model.fit(X, y)

        return get_feature_importances(model, model_name, X, y)

    elif method == 'cars':
        # CARS variable selection with model-aware evaluation
        try:
            # For Bayesian optimization context, use reduced iterations
            # (300 trials × 15 iterations = 4,500 CARS runs vs 9,000 with n_iterations=30)
            TREE_MODELS = {'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'rf'}
            use_hybrid = model_name.lower() in {m.lower() for m in TREE_MODELS}

            importances = cars_selection(
                X, y,
                n_iterations=15,  # Reduced for Bayesian optimization context
                pls_components=min(5, n_features - 1),
                cv_folds=cv_folds,
                monte_carlo_samples=60,  # Increased for better sampling with fewer iterations
                random_state=random_state,
                model_type=model_name,  # Enable model-aware evaluation
                use_hybrid_importance=use_hybrid  # Use hybrid importance for tree models
            )
            return importances
        except Exception as e:
            logging.warning(f"CARS failed: {e}, falling back to importance")
            return compute_importances(X, y, 'importance', model_name, cv_folds, random_state)

    else:
        # Default: uniform importances (full model)
        return np.ones(n_features)


def create_unified_objective(
    X_raw: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    model_name: str,
    task_type: str = 'regression',
    cv_folds: int = 5,
    random_state: int = 42,
    n_top_regions: int = 10,
    progress_callback: Optional[Callable] = None
) -> Callable[[Trial], float]:
    """Create objective function for Optuna optimization.

    The objective function:
    1. Suggests preprocessing, hyperparams, variable method, subset size
    2. Applies preprocessing
    3. Computes importances and selects subset (including dynamic regional subsets)
    4. Returns SUBSET score to TPE (critical for learning)

    Parameters
    ----------
    X_raw : np.ndarray
        Raw spectral data (not preprocessed)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values
    model_name : str
        Model name
    task_type : str
        'regression' or 'classification'
    cv_folds : int
        Number of CV folds
    random_state : int
        Random seed
    n_top_regions : int
        Number of top regions for regional subset selection
    progress_callback : callable, optional
        Progress callback function

    Returns
    -------
    objective : callable
        Objective function for Optuna
    """
    n_samples, n_features = X_raw.shape

    # Determine scoring
    if task_type == 'regression':
        scoring = 'neg_root_mean_squared_error'
    else:
        scoring = 'accuracy'

    # Create CV splitter
    if task_type == 'regression':
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Determine available subset types
    # Regional subsets are computed dynamically on preprocessed data
    available_methods = ['importance', 'cars', 'region']

    def objective(trial: Trial) -> float:
        """Objective function for a single trial."""
        try:
            # 1. Suggest preprocessing
            preprocess_config = suggest_preprocessing(trial, n_features)

            # 2. Apply preprocessing
            X_prep = apply_preprocessing(X_raw, preprocess_config)
            n_features_prep = X_prep.shape[1]

            # Validate preprocessing didn't corrupt data
            assert X_prep.shape[0] == X_raw.shape[0], \
                f"Preprocessing changed sample count! {X_raw.shape[0]} -> {X_prep.shape[0]}"

            # 3. Suggest subset type and size
            # IMPORTANT: Always suggest ALL parameters to maintain consistent parameter space
            # Optuna requires the same parameter names to have consistent value spaces
            subset_type = trial.suggest_categorical('subset_type', available_methods)
            subset_size = trial.suggest_categorical('n_vars', SUBSET_SIZES)
            region_idx = trial.suggest_int('region_id', 0, max(0, n_top_regions - 1))

            if subset_type == 'region':
                # Compute regions DYNAMICALLY on preprocessed data
                # This ensures regions are relevant to the current preprocessing
                try:
                    # Create wavelengths for preprocessed data (may have different length)
                    if n_features_prep == len(wavelengths):
                        wl_prep = wavelengths
                    else:
                        # Interpolate wavelengths if preprocessing changed feature count
                        wl_prep = np.linspace(wavelengths[0], wavelengths[-1], n_features_prep)

                    dynamic_regions = create_region_subsets(
                        X_prep, y, wl_prep.astype(float),
                        n_top_regions=n_top_regions
                    )

                    if len(dynamic_regions) > 0:
                        # Clamp region_idx to actual available regions
                        actual_region_idx = min(region_idx, len(dynamic_regions) - 1)

                        top_indices = dynamic_regions[actual_region_idx]['indices']
                        n_vars = len(top_indices)
                        subset_tag = dynamic_regions[actual_region_idx]['tag']
                    else:
                        # Fallback to importance if no regions found
                        actual_subset_size = subset_size if subset_size != 'full' else 100
                        n_vars = min(actual_subset_size, n_features_prep - 1)
                        if n_vars < 5:
                            n_vars = min(5, n_features_prep - 1)
                        importances = compute_importances(
                            X_prep, y, 'importance', model_name, cv_folds, random_state
                        )
                        top_indices = np.argsort(importances)[-n_vars:]
                        subset_tag = f"top{n_vars}_importance_fallback"
                except Exception as e:
                    logging.warning(f"Dynamic region creation failed: {e}, falling back to importance")
                    actual_subset_size = subset_size if subset_size != 'full' else 100
                    n_vars = min(actual_subset_size, n_features_prep - 1)
                    if n_vars < 5:
                        n_vars = min(5, n_features_prep - 1)
                    importances = compute_importances(
                        X_prep, y, 'importance', model_name, cv_folds, random_state
                    )
                    top_indices = np.argsort(importances)[-n_vars:]
                    subset_tag = f"top{n_vars}_importance_fallback"
            else:
                # Importance-based or CARS-based selection (region_idx is ignored)

                if subset_size == 'full':
                    top_indices = None
                    n_vars = n_features_prep
                    subset_tag = 'full'
                else:
                    n_vars = min(subset_size, n_features_prep - 1)
                    if n_vars < 5:
                        n_vars = min(5, n_features_prep - 1)

                    # Compute importances
                    importances = compute_importances(
                        X_prep, y, subset_type, model_name, cv_folds, random_state
                    )

                    # Select top variables
                    top_indices = np.argsort(importances)[-n_vars:]
                    subset_tag = f"top{n_vars}_{subset_type}"

            # 4. Apply subset if selected
            if top_indices is not None and len(top_indices) > 0:
                X_final = X_prep[:, top_indices]
                n_features_final = X_final.shape[1]
            else:
                X_final = X_prep
                n_features_final = n_features_prep
                top_indices = None

            # 5. Suggest model hyperparameters (based on final feature count)
            model_params = suggest_model_params(
                trial, model_name, n_features_final, task_type
            )

            # 6. Build and cross-validate model
            model = build_model(model_name, model_params, task_type=task_type)

            # 7. Compute metrics using cross_validate for proper scoring
            if task_type == 'regression':
                # Use cross_validate to get both RMSE and R² correctly
                cv_results = cross_validate(
                    model, X_final, y,
                    cv=cv,
                    scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
                    n_jobs=1,
                    error_score='raise'
                )
                rmse = -cv_results['test_rmse'].mean()
                r2 = cv_results['test_r2'].mean()
                metric = rmse  # Minimize RMSE
            else:
                # Classification: use accuracy
                scores = cross_val_score(
                    model, X_final, y, cv=cv, scoring='accuracy', n_jobs=1, error_score='raise'
                )
                accuracy = scores.mean()
                metric = -accuracy  # Minimize negative accuracy

            # Store additional info in trial
            trial.set_user_attr('preprocessing', preprocess_config['name'])
            trial.set_user_attr('window', preprocess_config.get('window', 0))
            trial.set_user_attr('deriv', preprocess_config.get('deriv', 0))
            trial.set_user_attr('poly', preprocess_config.get('polyorder', 0))
            trial.set_user_attr('subset_type', subset_type)
            trial.set_user_attr('subset_tag', subset_tag)
            trial.set_user_attr('n_vars', n_vars)
            trial.set_user_attr('model_params', str(model_params))

            if task_type == 'regression':
                trial.set_user_attr('RMSE', rmse)
                trial.set_user_attr('R2', r2)
            else:
                trial.set_user_attr('Accuracy', accuracy)

            # Store selected wavelengths (in SPECTRAL order for Model Development)
            if top_indices is not None:
                # Sort indices to spectral order for storage (training already done)
                sorted_indices = np.sort(top_indices)
                selected_wavelengths = wavelengths[sorted_indices] if len(wavelengths) > max(sorted_indices) else []
                trial.set_user_attr('selected_wavelengths',
                    ','.join([f"{w:.0f}" for w in selected_wavelengths[:50]]))  # First 50
                # Store ALL wavelengths for model reconstruction (all_vars column)
                trial.set_user_attr('all_wavelengths', ','.join([f"{w:.0f}" for w in selected_wavelengths]))
            else:
                # Full spectrum - store all wavelengths
                trial.set_user_attr('all_wavelengths', ','.join([f"{w:.0f}" for w in wavelengths]))

            return metric

        except Exception as e:
            logging.warning(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
            # Return large penalty
            if task_type == 'regression':
                return 1e10
            else:
                return 1e10

    return objective


def run_unified_bayesian(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    model_name: str,
    task_type: str = 'regression',
    n_trials: int = 300,
    cv_folds: int = 5,
    n_top_regions: int = 10,
    random_state: int = 42,
    progress_callback: Optional[Callable] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, optuna.Study]:
    """Run unified Bayesian optimization.

    Jointly optimizes preprocessing, hyperparameters, variable selection,
    and subset size using Optuna's TPE sampler.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    wavelengths : np.ndarray
        Wavelength values
    model_name : str
        Model name ('PLS', 'Ridge', 'LightGBM', etc.)
    task_type : str
        'regression' or 'classification'
    n_trials : int
        Number of Optuna trials (default 300)
    cv_folds : int
        Number of CV folds (default 5)
    n_top_regions : int
        Number of regional subsets to test (default 10)
    random_state : int
        Random seed for reproducibility
    progress_callback : callable, optional
        Function called after each trial with progress info
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    results_df : pd.DataFrame
        Results DataFrame with all trials, compatible with existing format
    study : optuna.Study
        Optuna study object with full optimization history
    """
    # Normalize model name to expected case for build_model
    model_name_map = {
        'pls': 'PLS',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
        'elasticnet': 'ElasticNet',
        'rf': 'RandomForest',
        'randomforest': 'RandomForest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost',
        'svr': 'SVR',
        'svm': 'SVM',
        'mlp': 'MLP',
    }
    model_name = model_name_map.get(model_name.lower(), model_name)

    X = np.asarray(X)
    y = np.asarray(y)
    wavelengths = np.asarray(wavelengths)
    n_samples, n_features = X.shape

    if verbose:
        print(f"\n{'='*70}")
        print(f"Unified Bayesian Optimization")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Task: {task_type}")
        print(f"Trials: {n_trials}")
        print(f"CV Folds: {cv_folds}")
        print(f"Samples: {n_samples}, Features: {n_features}")
        print(f"Regional subsets: dynamically computed ({n_top_regions} regions)")
        print(f"Variable methods: importance, CARS, region")
        print(f"{'='*70}\n")

    # Create objective function
    # Note: Regional subsets are computed DYNAMICALLY inside the objective
    # on preprocessed data, ensuring regions are relevant to the current preprocessing
    objective = create_unified_objective(
        X_raw=X,
        y=y,
        wavelengths=wavelengths,
        model_name=model_name,
        task_type=task_type,
        cv_folds=cv_folds,
        random_state=random_state,
        n_top_regions=n_top_regions,
        progress_callback=progress_callback
    )

    # Create TPE sampler with good defaults
    sampler = TPESampler(
        seed=random_state,
        n_startup_trials=20,  # Random exploration first
        n_ei_candidates=32,   # More candidates for better exploration
        multivariate=True,    # Model parameter interactions
        consider_endpoints=True
    )

    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f"unified_bayesian_{model_name}"
    )

    # Progress callback wrapper
    def progress_wrapper(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if progress_callback:
            progress_info = {
                'stage': 'unified_bayesian',
                'current': trial.number + 1,
                'total': n_trials,
                'message': f'{model_name}: Trial {trial.number + 1}/{n_trials}'
            }

            if trial.value is not None:
                if task_type == 'regression':
                    progress_info['message'] += f" - RMSE: {trial.value:.4f}"
                else:
                    progress_info['message'] += f" - Acc: {-trial.value:.4f}"

            progress_callback(progress_info)

        if verbose and (trial.number + 1) % 10 == 0:
            if trial.value is not None and trial.value < 1e9:
                if task_type == 'regression':
                    print(f"  Trial {trial.number + 1}/{n_trials}: RMSE={trial.value:.4f}")
                else:
                    print(f"  Trial {trial.number + 1}/{n_trials}: Acc={-trial.value:.4f}")

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[progress_wrapper],
        show_progress_bar=verbose and not progress_callback
    )

    # Convert results to DataFrame
    results_df = convert_study_to_dataframe(
        study, model_name, task_type, wavelengths, n_features, cv_folds
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimization Complete")
        print(f"{'='*70}")
        if task_type == 'regression':
            best_rmse = results_df['RMSE'].min()
            best_r2 = results_df.loc[results_df['RMSE'].idxmin(), 'R2']
            print(f"Best RMSE: {best_rmse:.6f}")
            print(f"Best R2: {best_r2:.6f}")
        else:
            best_acc = results_df['Accuracy'].max()
            print(f"Best Accuracy: {best_acc:.6f}")

        # Show best configuration
        if task_type == 'regression':
            best_row = results_df.loc[results_df['RMSE'].idxmin()]
        else:
            best_row = results_df.loc[results_df['Accuracy'].idxmax()]

        print(f"\nBest Configuration:")
        print(f"  Preprocessing: {best_row['Preprocess']}")
        print(f"  Subset: {best_row['SubsetTag']} ({best_row['n_vars']} vars)")
        print(f"  Params: {best_row['Params']}")
        print(f"{'='*70}\n")

    return results_df, study


def convert_study_to_dataframe(
    study: optuna.Study,
    model_name: str,
    task_type: str,
    wavelengths: np.ndarray,
    n_features: int,
    cv_folds: int
) -> pd.DataFrame:
    """Convert Optuna study to results DataFrame.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    model_name : str
        Model name
    task_type : str
        'regression' or 'classification'
    wavelengths : np.ndarray
        Wavelength values
    n_features : int
        Original number of features
    cv_folds : int
        Number of CV folds

    Returns
    -------
    results_df : pd.DataFrame
        Results in standard DASP format
    """
    results = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Skip failed trials (penalty value)
        if trial.value is not None and trial.value >= 1e9:
            continue

        row = {
            'Task': task_type,
            'Model': model_name,
            'Preprocess': trial.user_attrs.get('preprocessing', 'unknown'),
            'Deriv': trial.user_attrs.get('deriv', 0),
            'Window': trial.user_attrs.get('window', 0),
            'Poly': trial.user_attrs.get('poly', 0),
            'Params': trial.user_attrs.get('model_params', '{}'),
            'n_vars': trial.user_attrs.get('n_vars', n_features),
            'full_vars': n_features,
            'SubsetTag': trial.user_attrs.get('subset_tag', 'full'),
            'trial_number': trial.number,
            'Folds': cv_folds,
            'Optimization': 'Unified Bayesian'
        }

        # Add metrics
        if task_type == 'regression':
            row['RMSE'] = trial.user_attrs.get('RMSE', trial.value)
            row['R2'] = trial.user_attrs.get('R2', np.nan)
        else:
            row['Accuracy'] = trial.user_attrs.get('Accuracy', -trial.value if trial.value else np.nan)
            row['ROC_AUC'] = np.nan  # Placeholder for report.py compatibility

        # Add wavelength info
        row['top_vars'] = trial.user_attrs.get('selected_wavelengths', 'N/A')
        row['all_vars'] = trial.user_attrs.get('all_wavelengths', 'N/A')

        # Extract LVs for PLS
        if model_name.lower() in ('pls', 'pls-da'):
            params = trial.params
            row['LVs'] = params.get('n_components', np.nan)
        else:
            row['LVs'] = np.nan

        results.append(row)

    df = pd.DataFrame(results)

    # Sort by performance
    if task_type == 'regression':
        df = df.sort_values('RMSE', ascending=True)
    else:
        df = df.sort_values('Accuracy', ascending=False)

    df = df.reset_index(drop=True)

    # Add Rank column (required for report.py and model selection)
    df.insert(0, 'Rank', range(1, len(df) + 1))

    # Add CompositeScore column (required for report.py compatibility)
    # Lower is better for both regression (RMSE) and classification (negated accuracy)
    if task_type == 'regression':
        df['CompositeScore'] = df['RMSE']
    else:
        df['CompositeScore'] = -df['Accuracy']

    return df


if __name__ == '__main__':
    """Self-test with synthetic data."""
    print("Testing unified_bayesian.py with synthetic data...")

    # Generate synthetic spectral data
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 200

    # Simulate spectra with peaks
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    X = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        baseline = 0.5 + 0.0001 * wavelengths - 0.00000005 * wavelengths ** 2
        peak1 = 0.3 * np.exp(-((wavelengths - 1000) ** 2) / (2 * 50 ** 2))
        peak2 = 0.5 * np.exp(-((wavelengths - 1500) ** 2) / (2 * 80 ** 2))
        noise = 0.02 * np.random.randn(n_wavelengths)
        X[i, :] = baseline + peak1 + peak2 + noise

    # Create target (sum of peak intensities)
    y = (
        X[:, np.argmin(np.abs(wavelengths - 1000))]
        + X[:, np.argmin(np.abs(wavelengths - 1500))]
        + 0.1 * np.random.randn(n_samples)
    )

    # Test with PLS (quick: 20 trials)
    print("\n" + "=" * 60)
    print("Test: PLS with 20 trials")
    print("=" * 60)

    results_df, study = run_unified_bayesian(
        X, y, wavelengths,
        model_name='PLS',
        n_trials=20,
        cv_folds=3,
        n_top_regions=5,
        verbose=True
    )

    print(f"\nResults: {len(results_df)} configurations tested")
    print(f"Best RMSE: {results_df['RMSE'].min():.6f}")
    print(f"Best R2: {results_df.loc[results_df['RMSE'].idxmin(), 'R2']:.6f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
