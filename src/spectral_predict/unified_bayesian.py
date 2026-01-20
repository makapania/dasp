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
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, accuracy_score
from typing import Dict, List, Optional, Callable, Tuple, Any

# Import existing infrastructure
from spectral_predict.preprocess import SNV, SavgolDerivative
from spectral_predict.models import build_model, get_feature_importances
from spectral_predict.regions import create_region_subsets
from spectral_predict.variable_selection import (
    spa_selection, uve_selection, cars_selection
)

# Imbalance handling imports
from imblearn.pipeline import Pipeline as ImbPipeline
from spectral_predict.imbalance import build_imbalance_transformer, validate_classification_config

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


def _needs_resampling_pipeline(imbalance_method: Optional[str], task_type: str) -> bool:
    """Determine if the imbalance method requires imblearn Pipeline.

    Parameters
    ----------
    imbalance_method : str or None
        The imbalance handling method
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    bool
        True if the method requires imblearn Pipeline (uses fit_resample)
    """
    if imbalance_method is None:
        return False
    if imbalance_method == 'class_weight':
        return False

    if task_type == 'classification':
        resampling_methods = {'smote', 'adasyn', 'borderline_smote',
                              'random_undersampler', 'tomek_links', 'smote_tomek'}
        return imbalance_method.lower().replace('-', '_') in resampling_methods
    elif task_type == 'regression':
        resampling_methods = {'undersample', 'oversample', 'smogn', 'smotetomek'}
        return imbalance_method.lower() in resampling_methods
    return False


def _normalize_preprocess_name(name: str) -> str:
    """Convert detailed preprocessing names to standard names for validation.

    Unified Bayesian uses names like 'deriv1', 'snv_deriv2', 'deriv3_snv',
    but build_preprocessing_pipeline() expects 'deriv', 'snv_deriv', 'deriv_snv'.

    Parameters
    ----------
    name : str
        Preprocessing name (e.g., 'deriv1', 'snv_deriv2')

    Returns
    -------
    str
        Normalized name compatible with build_preprocessing_pipeline()
    """
    if name in ('raw', 'snv'):
        return name

    # Strip window suffix like '_w17', '_w43' if present (for consistency with NSGA-II)
    import re
    name_no_window = re.sub(r'_w\d+$', '', name)

    # deriv1, deriv2, deriv3, deriv4 → deriv
    if name_no_window.startswith('deriv') and '_' not in name_no_window:
        return 'deriv'
    # snv_deriv1, snv_deriv2, etc. → snv_deriv
    if name_no_window.startswith('snv_deriv'):
        return 'snv_deriv'
    # deriv1_snv, deriv2_snv, etc. → deriv_snv
    if name_no_window.startswith('deriv') and name_no_window.endswith('_snv'):
        return 'deriv_snv'
    return name  # fallback for unknown names


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
            'bagging_freq': 1,  # Required when subsample < 1.0
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
            # verbosity and n_jobs are set by build_model()
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
            'reg_lambda': 1.0
            # tree_method, verbosity, n_jobs are set by build_model()
        }

    elif model_name_lower == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
            # verbose is set by build_model()
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
    random_state: int = 42,
    task_type: str = 'regression'
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
    task_type : str
        'regression' or 'classification'

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

        model = build_model(model_name, params, task_type=task_type)
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
            return compute_importances(X, y, 'importance', model_name, cv_folds, random_state, task_type)

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
    progress_callback: Optional[Callable] = None,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None
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
    imbalance_method : str, optional
        Imbalance handling method (e.g., 'smote', 'class_weight')
    imbalance_params : dict, optional
        Parameters for the imbalance method

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

    # Guard against None params for imbalance handling
    _imbalance_params = imbalance_params if imbalance_params is not None else {}

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
                            X_prep, y, 'importance', model_name, cv_folds, random_state, task_type
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
                        X_prep, y, 'importance', model_name, cv_folds, random_state, task_type
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
                        X_prep, y, subset_type, model_name, cv_folds, random_state, task_type
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

            # Scale-sensitive models need StandardScaler (matches search.py behavior)
            # For PLS-DA: PLS + StandardScaler + LogisticRegression (search.py lines 3417-3424)
            # For scale-sensitive models: StandardScaler + Model (search.py lines 3427-3429)
            SCALE_SENSITIVE_MODELS = {'SVC', 'SVR', 'MLP', 'NeuralBoosted'}

            # Build pipeline steps with imbalance handling support
            pipe_steps = []

            # Step 1: Add imbalance transformer if specified (and not class_weight)
            if imbalance_method is not None and imbalance_method != 'class_weight':
                imbalance_transformer = build_imbalance_transformer(
                    method=imbalance_method,
                    task_type=task_type,
                    random_state=random_state,
                    **_imbalance_params
                )
                pipe_steps.append(("imbalance", imbalance_transformer))

            # Step 2: Handle class_weight for classification models that support it
            if imbalance_method == 'class_weight' and task_type == 'classification':
                if hasattr(model, 'class_weight'):
                    try:
                        model.set_params(class_weight='balanced')
                    except Exception:
                        pass

            # Step 3: Build model pipeline based on model type
            from sklearn.preprocessing import StandardScaler

            if task_type == 'classification' and model_name.lower() in ('pls', 'pls-da'):
                from sklearn.linear_model import LogisticRegression
                # PLS-DA: Add PLS, scaler, then LogisticRegression
                pipe_steps.append(('pls', model))
                pipe_steps.append(('scaler', StandardScaler()))  # Scale PLS scores for LogisticRegression
                lr = LogisticRegression(max_iter=1000, random_state=random_state)
                # Apply class_weight to LogisticRegression if specified
                if imbalance_method == 'class_weight':
                    lr.set_params(class_weight='balanced')
                pipe_steps.append(('lr', lr))
            elif model_name in SCALE_SENSITIVE_MODELS:
                # Scale-sensitive models: StandardScaler + Model
                pipe_steps.append(('scaler', StandardScaler()))
                pipe_steps.append(('model', model))
            else:
                # Other models don't need scaling
                pipe_steps.append(('model', model))

            # Step 4: Create pipeline with correct class (ImbPipeline for resampling methods)
            needs_resampling = _needs_resampling_pipeline(imbalance_method, task_type)
            if needs_resampling:
                pipeline = ImbPipeline(pipe_steps)
            else:
                pipeline = Pipeline(pipe_steps)

            # Use the constructed pipeline as the model
            model = pipeline

            # 7. Compute metrics
            if task_type == 'regression':
                # Use cross_validate for RMSE (averaging is valid for RMSE)
                cv_results = cross_validate(
                    model, X_final, y,
                    cv=cv,
                    scoring={'rmse': 'neg_root_mean_squared_error'},
                    n_jobs=1,
                    error_score='raise'
                )
                rmse = -cv_results['test_rmse'].mean()

                # R² must use aggregated predictions (not per-fold averages)
                # Averaging per-fold R² is mathematically incorrect due to different SS_tot per fold
                # This matches the method used in search.py for consistency with Model Development
                y_pred = cross_val_predict(model, X_final, y, cv=cv, n_jobs=1)
                r2 = r2_score(y, y_pred)

                metric = rmse  # Minimize RMSE
            else:
                # Classification: use accuracy and ROC_AUC
                scores = cross_val_score(
                    model, X_final, y, cv=cv, scoring='accuracy', n_jobs=1, error_score='raise'
                )
                accuracy = scores.mean()

                # Compute ROC_AUC using cross_val_predict for probability estimates
                try:
                    y_proba = cross_val_predict(
                        model, X_final, y, cv=cv, method='predict_proba', n_jobs=1
                    )
                    n_classes = len(np.unique(y))
                    if n_classes == 2:
                        # Binary classification
                        roc_auc = roc_auc_score(y, y_proba[:, 1])
                    else:
                        # Multiclass - use weighted average
                        roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                except Exception:
                    roc_auc = np.nan

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

            # Fit on full training data for calibration metrics
            model.fit(X_final, y)
            y_pred_cal = model.predict(X_final)

            if task_type == 'regression':
                cal_rmse = np.sqrt(mean_squared_error(y, y_pred_cal))
                cal_r2 = r2_score(y, y_pred_cal)
                trial.set_user_attr('RMSE', cal_rmse)      # Calibration
                trial.set_user_attr('R2', cal_r2)          # Calibration
                trial.set_user_attr('RMSEcv', rmse)        # CV (was RMSE)
                trial.set_user_attr('R2cv', r2)            # CV (was R2)
            else:
                cal_accuracy = accuracy_score(y, y_pred_cal)
                trial.set_user_attr('Accuracy', cal_accuracy)    # Calibration
                trial.set_user_attr('Accuracycv', accuracy)      # CV (was Accuracy)
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba_cal = model.predict_proba(X_final)
                        n_classes = len(np.unique(y))
                        if n_classes == 2:
                            cal_roc_auc = roc_auc_score(y, y_proba_cal[:, 1])
                        else:
                            cal_roc_auc = roc_auc_score(y, y_proba_cal, multi_class='ovr', average='weighted')
                        trial.set_user_attr('ROC_AUC', cal_roc_auc)     # Calibration
                except Exception:
                    trial.set_user_attr('ROC_AUC', np.nan)
                trial.set_user_attr('ROC_AUCcv', roc_auc)          # CV (was ROC_AUC)

            # Store selected wavelengths in TRAINING ORDER (importance order)
            # CRITICAL: Do NOT sort - Model Development expects wavelengths in the same
            # order they were used during training. Grid Search also preserves training
            # order (see search.py line 3201).
            if top_indices is not None:
                selected_wavelengths = wavelengths[top_indices] if len(wavelengths) > max(top_indices) else []
                # Store ALL wavelengths for model reconstruction (training order)
                trial.set_user_attr('all_wavelengths', ','.join([f"{w:.0f}" for w in selected_wavelengths]))
                # Store first 50 for display (also training order - most important first)
                trial.set_user_attr('selected_wavelengths',
                    ','.join([f"{w:.0f}" for w in selected_wavelengths[:50]]))
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
    verbose: bool = True,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
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
    imbalance_method : str, optional
        Imbalance handling method (e.g., 'smote', 'class_weight')
    imbalance_params : dict, optional
        Parameters for the imbalance method

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

    # Label-encode y for classification (string labels -> integers)
    # This matches how search.py handles classification at lines 737-740
    label_encoder = None
    if task_type == 'classification':
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Guard against None params
    if imbalance_params is None:
        imbalance_params = {}

    # Validate imbalance configuration for classification
    if task_type == 'classification' and imbalance_method is not None:
        validate_classification_config(
            y=y,
            imbalance_method=imbalance_method,
            imbalance_params=imbalance_params,
            n_folds=cv_folds
        )

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
        progress_callback=progress_callback,
        imbalance_method=imbalance_method,
        imbalance_params=imbalance_params,
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

            # Add best model tracking for "Best Model So Far" display
            if study.best_trial is not None:
                best = study.best_trial
                best_model = {
                    'Model': model_name,
                    'Preprocess': best.params.get('preprocessing', 'raw'),
                    'n_vars': best.params.get('n_vars', 'N/A'),
                }
                if task_type == 'regression':
                    best_model['RMSE'] = best.value
                    # R² not available (only RMSE optimized), use placeholder
                    best_model['R2'] = 0.0
                else:
                    best_model['Accuracy'] = -best.value
                progress_info['best_model'] = best_model

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

        if len(results_df) == 0:
            print("WARNING: No successful trials completed!")
            print("All trials may have failed. Check the warnings above.")
            print(f"{'='*70}\n")
        else:
            if task_type == 'regression':
                best_rmse_cv = results_df['RMSEcv'].min()
                best_r2_cv = results_df.loc[results_df['RMSEcv'].idxmin(), 'R2cv']
                print(f"Best RMSEcv: {best_rmse_cv:.6f}")
                print(f"Best R2cv: {best_r2_cv:.6f}")
            else:
                best_acc = results_df['Accuracy'].max()
                print(f"Best Accuracy: {best_acc:.6f}")

            # Show best configuration
            if task_type == 'regression':
                best_row = results_df.loc[results_df['RMSEcv'].idxmin()]
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
            'Preprocess': _normalize_preprocess_name(trial.user_attrs.get('preprocessing', 'unknown')),
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

        # Add metrics - both calibration and CV
        if task_type == 'regression':
            row['RMSE'] = trial.user_attrs.get('RMSE', np.nan)       # Calibration
            row['R2'] = trial.user_attrs.get('R2', np.nan)           # Calibration
            row['RMSEcv'] = trial.user_attrs.get('RMSEcv', trial.value)  # CV
            row['R2cv'] = trial.user_attrs.get('R2cv', np.nan)       # CV
        else:
            row['Accuracy'] = trial.user_attrs.get('Accuracy', np.nan)       # Calibration
            accuracycv = trial.user_attrs.get('Accuracycv', -trial.value if trial.value else np.nan)
            row['Accuracycv'] = accuracycv   # CV
            row['CV Error'] = 1 - accuracycv if accuracycv is not None and not np.isnan(accuracycv) else np.nan
            row['ROC_AUC'] = trial.user_attrs.get('ROC_AUC', np.nan)         # Calibration
            row['ROC_AUCcv'] = trial.user_attrs.get('ROC_AUCcv', np.nan)     # CV

        # Add wavelength info
        row['top_vars'] = trial.user_attrs.get('selected_wavelengths', 'N/A')
        row['all_vars'] = trial.user_attrs.get('all_wavelengths', 'N/A')

        # Extract LVs for PLS (store as int, use None for non-PLS to avoid float conversion)
        if model_name.lower() in ('pls', 'pls-da'):
            params = trial.params
            n_comp = params.get('n_components')
            row['LVs'] = int(n_comp) if n_comp is not None else None
        else:
            row['LVs'] = None

        results.append(row)

    df = pd.DataFrame(results)

    # Handle empty results (all trials failed)
    if len(df) == 0:
        cols = ['Rank', 'Task', 'Model', 'Preprocess', 'Params', 'n_vars', 'top_vars',
                'all_vars', 'Window', 'Poly', 'Deriv', 'LVs', 'full_vars', 'SubsetTag',
                'trial_number', 'Folds', 'Optimization']
        if task_type == 'classification':
            cols.extend(['Accuracy', 'Accuracycv', 'CV Error', 'ROC_AUC', 'ROC_AUCcv', 'CompositeScore', 'Score'])
        else:
            cols.extend(['RMSE', 'R2', 'RMSEcv', 'R2cv', 'CompositeScore', 'Score'])
        return pd.DataFrame(columns=cols)

    # Sort by performance (use CV metrics for model selection)
    if task_type == 'regression':
        df = df.sort_values('RMSEcv', ascending=True)
    else:
        df = df.sort_values('CV Error', ascending=True)  # Lower CV Error is better

    df = df.reset_index(drop=True)

    # Add Rank column (required for report.py and model selection)
    df.insert(0, 'Rank', range(1, len(df) + 1))

    # Add CompositeScore column (required for report.py compatibility)
    # Lower is better for both regression (RMSEcv) and classification (CV Error)
    if task_type == 'regression':
        df['CompositeScore'] = df['RMSEcv']
    else:
        df['CompositeScore'] = df['CV Error']

    # Add Score column for compatibility with Coupled format
    if task_type == 'classification':
        df['Score'] = df['CV Error']
    else:
        df['Score'] = df['RMSEcv']

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
