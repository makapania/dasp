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

logger = logging.getLogger(__name__)

import ast
import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_predict

# Import early stopping CV utilities
from spectral_predict.cv_utils import (
    build_cv_splitter,
    cross_val_predict_pooled,
    cross_val_predict_with_early_stopping,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, r2_score, mean_squared_error, accuracy_score, mean_absolute_error,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, log_loss,
    f1_score, precision_score, recall_score, classification_report
)
from typing import Dict, List, Optional, Callable, Tuple, Any

# Import existing infrastructure
from spectral_predict.preprocess import SNV, SavgolDerivative, SavgolSmooth
from spectral_predict.baseline import BaselineALS, BaselineAirPLS, BaselinePolynomial, BaselineRubberBand
from spectral_predict.models import build_model, get_feature_importances
from spectral_predict.regions import create_region_subsets
from spectral_predict.variable_selection import (
    spa_selection, uve_selection, cars_selection
)
from spectral_predict.scoring import compute_specificity, lins_ccc

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


# Pipeline-specific params that should never be passed to model constructors
PIPELINE_PARAMS = {'memory', 'transform_input', 'verbose', 'steps', 'n_jobs'}


def _capture_serializable_params(model) -> Optional[Dict[str, Any]]:
    """Return model params that can round-trip through str() and ast.literal_eval()."""
    try:
        all_params = model.get_params()
    except Exception:
        return None

    filtered_params: Dict[str, Any] = {}
    for key, value in all_params.items():
        # Skip Pipeline-specific params that would break model constructors
        if key in PIPELINE_PARAMS:
            continue
        if callable(value) or hasattr(value, '__dict__'):
            continue

        try:
            if hasattr(value, 'item'):
                value = value.item()

            if isinstance(value, float) and np.isnan(value):
                continue

            test_str = str({key: value})
            ast.literal_eval(test_str)
            filtered_params[key] = value
        except Exception:
            continue

    return filtered_params

# Subset sizes to explore
SUBSET_SIZES = ['full', 10, 20, 50, 100, 250, 500, 1000]

# Variable selection methods
VAR_METHODS = ['importance', 'cars', 'region', 'uve']


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
                              'random_undersampler', 'tomek_links', 'smote_tomek', 'smote_enn'}
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


def _build_display_preprocess_name(
    core_name: str,
    apply_baseline: bool = False,
    baseline_method: str | None = None,
    apply_smoothing: bool = False,
    apply_autoscale: bool = False,
) -> str:
    """Build display name with baseline/smoothing/autoscale affixes matching Grid Search conventions."""
    name = _normalize_preprocess_name(core_name)
    if apply_baseline and baseline_method:
        name = f"{baseline_method}+{name}"
    if apply_smoothing:
        if '+' in name:
            parts = name.split('+', 1)
            name = f"{parts[0]}+sg0+{parts[1]}"
        else:
            name = f"sg0+{name}"
    # T-36: trailing '+autoscale' suffix matches the grid-path doubling block
    # (search.py:1880) so users can filter by Preprocess display name across paths.
    if apply_autoscale:
        name = f"{name}+autoscale"
    return name


def suggest_preprocessing(
    trial: Trial,
    n_features: int,
    baseline_method: str | None = None,
    smoothing: bool = False,
    enable_autoscale: bool = False,
) -> Dict[str, Any]:
    """Suggest preprocessing configuration.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    n_features : int
        Number of spectral features (wavelengths)
    baseline_method : str or None
        If not None, Optuna will suggest whether to apply this baseline method.
    smoothing : bool
        If True, Optuna will suggest whether to apply Savitzky-Golay smoothing.
    enable_autoscale : bool
        If True, Optuna will suggest whether to apply autoscale (UV scaling) per trial.
        T-36: lets TPE learn whether autoscale helps for this dataset.

    Returns
    -------
    config : dict
        Preprocessing configuration with keys:
        - 'name': Preprocessing name (e.g., 'snv_deriv1')
        - 'deriv': Derivative order (0-4)
        - 'window': Savitzky-Golay window size
        - 'polyorder': Polynomial order
        - 'apply_baseline': Whether to apply baseline correction (bool)
        - 'apply_smoothing': Whether to apply smoothing (bool)
        - 'apply_autoscale': Whether to apply autoscale (bool, T-36)
    """
    preprocessing = trial.suggest_categorical('preprocessing', PREPROCESSING_OPTIONS)

    config = {
        'name': preprocessing,
        'deriv': 0,
        'window': 0,
        'polyorder': 0
    }

    # Baseline correction toggle (only when user enabled it in the UI)
    if baseline_method is not None:
        config['apply_baseline'] = trial.suggest_categorical('apply_baseline', [True, False])
    else:
        config['apply_baseline'] = False

    # Smoothing toggle (only when user enabled it in the UI)
    if smoothing:
        config['apply_smoothing'] = trial.suggest_categorical('apply_smoothing', [True, False])
    else:
        config['apply_smoothing'] = False

    # Autoscale (UV scaling) toggle — T-36
    if enable_autoscale:
        config['apply_autoscale'] = trial.suggest_categorical('apply_autoscale', [True, False])
    else:
        config['apply_autoscale'] = False

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


def apply_preprocessing(
    X: np.ndarray,
    config: Dict[str, Any],
    baseline_method: str | None = None,
    baseline_params: dict | None = None,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
) -> np.ndarray:
    """Apply preprocessing to spectral data.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_wavelengths)
    config : dict
        Preprocessing configuration from suggest_preprocessing
    baseline_method : str or None
        Baseline correction method name (e.g., 'als', 'polynomial')
    baseline_params : dict or None
        Parameters for the baseline correction method
    smoothing_window : int
        Savitzky-Golay smoothing window length
    smoothing_polyorder : int
        Savitzky-Golay smoothing polynomial order

    Returns
    -------
    X_processed : np.ndarray
        Preprocessed data
    """
    # Step 1: Baseline correction (if this trial chose to apply it)
    if config.get('apply_baseline') and baseline_method is not None:
        params = baseline_params or {}
        if baseline_method == 'polynomial':
            bl = BaselinePolynomial(degree=params.get('degree', 2))
        elif baseline_method == 'als':
            bl = BaselineALS(lambda_=params.get('lam', 1e5), p=params.get('p', 0.01))
        elif baseline_method == 'rubber_band':
            bl = BaselineRubberBand()
        elif baseline_method == 'airpls':
            bl = BaselineAirPLS(lam=params.get('lam', 1e5))
        else:
            bl = None
        if bl is not None:
            X = bl.fit_transform(X)

    # Step 2: Smoothing (if this trial chose to apply it)
    if config.get('apply_smoothing'):
        X = SavgolSmooth(window_length=smoothing_window, polyorder=smoothing_polyorder).fit_transform(X)

    # Step 3: Core preprocessing (SNV / derivatives).
    # T-36 BUG #1 fix: previously each branch returned early, leaving any
    # trailing autoscale step unreachable. Now each branch ASSIGNS to X,
    # then a single trailing autoscale step + return runs at the end.
    name = config['name']

    if name == 'raw':
        X = X.copy()
    elif name == 'snv':
        X = SNV().fit_transform(X)
    elif 'deriv' in name:
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
            X = savgol.fit_transform(SNV().fit_transform(X))
        elif name.startswith('deriv') and '_snv' in name:
            # Derivative then SNV
            X = SNV().fit_transform(savgol.fit_transform(X))
        else:
            # Just derivative
            X = savgol.fit_transform(X)
    else:
        # Unknown / unhandled name — preserve historic fallback (a defensive copy).
        X = X.copy()

    # Step 4 (T-36): Autoscale (UV scaling) — applied AFTER core preprocessing,
    # so variable selection and the model see scaled features.
    if config.get('apply_autoscale'):
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    return X


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
        # Use categorical to skip invalid values 0 and 1 (can't satisfy num_leaves > 1)
        max_depth = trial.suggest_categorical('max_depth', [-1] + list(range(2, 16)))

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


def suggest_one_class_params(trial: Trial, model_name: str) -> dict:
    """Suggest hyperparameters for one-class models via Optuna TPE.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    model_name : str
        One-class model name (e.g., 'PCA-SIMCA', 'OneClassSVM').

    Returns
    -------
    params : dict
        Hyperparameters for the specified model.

    Raises
    ------
    ValueError
        If model_name is not a recognised one-class model.
    """
    if model_name == 'PCA-SIMCA':
        return {
            'n_components': trial.suggest_int('n_components', 2, 20),
            'alpha': trial.suggest_float('alpha', 0.01, 0.20, log=True),
        }
    elif model_name == 'OneClassSVM':
        return {
            'nu': trial.suggest_float('nu', 0.001, 0.5, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        }
    elif model_name == 'IsolationForest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'contamination': trial.suggest_float('contamination', 0.001, 0.3, log=True),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        }
    elif model_name == 'EllipticEnvelope':
        return {
            'contamination': trial.suggest_float('contamination', 0.001, 0.3, log=True),
            'support_fraction': trial.suggest_float('support_fraction', 0.5, 1.0),
        }
    elif model_name == 'LOF':
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
            'contamination': trial.suggest_float('contamination', 0.001, 0.3, log=True),
        }
    else:
        raise ValueError(f"Unknown one-class model: {model_name}")


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
                use_hybrid_importance=use_hybrid,  # Use hybrid importance for tree models
                task_type=task_type,
            )
            return importances
        except Exception as e:
            logging.warning(f"CARS failed: {e}, falling back to importance")
            return compute_importances(X, y, 'importance', model_name, cv_folds, random_state, task_type)

    elif method == 'uve':
        try:
            importances = uve_selection(
                X, y,
                cutoff_multiplier=1.0,
                cv_folds=cv_folds,
                random_state=random_state,
            )
            return importances
        except Exception as e:
            logging.warning(f"UVE failed: {e}, falling back to importance")
            return compute_importances(X, y, 'importance', model_name, cv_folds, random_state, task_type)

    else:
        # Default: uniform importances (full model)
        return np.ones(n_features)


def _apply_edge_mask_to_data(
    X: np.ndarray,
    wavelengths: np.ndarray,
    preprocess_cfg: dict,
) -> tuple:
    """Remove edge wavelengths affected by Savitzky-Golay derivatives.

    For derivative preprocessing with window W, the first and last W//2
    wavelengths are unreliable due to boundary effects.

    Returns
    -------
    tuple of (X_masked, wavelengths_masked, edge_zone)
    """
    deriv = preprocess_cfg.get("deriv")
    window = preprocess_cfg.get("window")
    if not deriv or not window:
        return X, wavelengths, 0

    edge_zone = window // 2
    if edge_zone == 0:
        return X, wavelengths, 0

    if 2 * edge_zone >= X.shape[1]:
        return X, wavelengths, 0

    return X[:, edge_zone:-edge_zone], wavelengths[edge_zone:-edge_zone], edge_zone


def create_unified_objective(
    X_raw: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    model_name: str,
    task_type: str = 'regression',
    cv_folds: int = 5,
    cv_strategy: str = 'kfold',
    cv_n_repeats: int = 5,
    random_state: int = 42,
    n_top_regions: int = 10,
    progress_callback: Optional[Callable] = None,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: Optional[int] = 40,
    region_test_all_individual: bool = False,
    region_test_pairwise: bool = False,
    baseline_method: str | None = None,
    baseline_params: dict | None = None,
    smoothing: bool = False,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    enable_autoscale: bool = False,  # T-36
    enable_uve: bool = False,
    inlier_class_label=None,
    y_original: np.ndarray | None = None,
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
    early_stopping_rounds : int, optional, default=50
        Number of rounds without improvement before stopping for boosting models.
        Set to None or 0 to disable.
    region_test_all_individual : bool, default=False
        Test all N regions individually (not just top 5)
    region_test_pairwise : bool, default=False
        Test all C(N,2) pairwise region combinations
    baseline_method : str or None
        Baseline correction method (e.g., 'als', 'polynomial', 'rubber_band', 'airpls')
    baseline_params : dict or None
        Parameters for the baseline correction method
    smoothing : bool
        Whether to let Optuna toggle Savitzky-Golay smoothing
    smoothing_window : int
        Smoothing window length
    smoothing_polyorder : int
        Smoothing polynomial order
    enable_uve : bool, default=False
        Include UVE (Uninformative Variable Elimination) as a variable selection method

    Returns
    -------
    objective : callable
        Objective function for Optuna
    """
    n_samples, n_features = X_raw.shape

    # Check if early stopping should be used for this model
    use_early_stopping = (
        early_stopping_rounds is not None and
        early_stopping_rounds > 0 and
        model_name in ('XGBoost', 'LightGBM', 'CatBoost')
    )

    # Determine scoring
    if task_type == 'regression':
        scoring = 'neg_root_mean_squared_error'
    else:
        scoring = 'accuracy'

    # Create CV splitter via factory (supports kfold/repeated_kfold/loo)
    cv = build_cv_splitter(
        strategy=cv_strategy,
        n_folds=cv_folds,
        task_type=task_type,
        n_repeats=cv_n_repeats,
        random_state=random_state,
    )

    # Determine available subset types
    # Regional subsets are computed dynamically on preprocessed data
    available_methods = ['importance', 'cars', 'region']
    if enable_uve:
        available_methods.append('uve')

    # Guard against None params for imbalance handling
    _imbalance_params = imbalance_params if imbalance_params is not None else {}

    # Compute one-class binary labels (+1 inlier, -1 outlier) for OC task
    y_oc: np.ndarray | None = None
    if task_type == 'one_class' and inlier_class_label is not None and y_original is not None:
        # Compare as strings to handle both numeric and text labels consistently
        y_str = np.asarray(y_original, dtype=str)
        y_oc = np.where(y_str == str(inlier_class_label), 1, -1)

    # Cache regions by preprocessing config to avoid redundant computation
    region_cache = {}
    importance_cache = {}  # Cache importances per (preprocessing_config, method, model_proxy)
    preprocessing_cache = {}  # Cache preprocessed data per config

    def objective(trial: Trial) -> float:
        """Objective function for a single trial."""
        try:
            # 1. Suggest preprocessing
            preprocess_config = suggest_preprocessing(
                trial, n_features,
                baseline_method=baseline_method,
                smoothing=smoothing,
                enable_autoscale=enable_autoscale,
            )

            # Create cache key from preprocessing config.
            # T-36 BUG #2 fix: include apply_autoscale so two trials that differ
            # only in autoscale don't collide on the cache (the second would
            # silently receive the first's X_prep).
            cache_key = (
                preprocess_config.get('name', 'raw'),
                preprocess_config.get('deriv', 0),
                preprocess_config.get('window', 0),
                preprocess_config.get('polyorder', 0),
                preprocess_config.get('apply_baseline', False),
                preprocess_config.get('apply_smoothing', False),
                preprocess_config.get('apply_autoscale', False),
            )

            # 2. Apply preprocessing (cached per config — deterministic for same input)
            if cache_key in preprocessing_cache:
                X_prep = preprocessing_cache[cache_key]
            else:
                X_prep = apply_preprocessing(
                    X_raw, preprocess_config,
                    baseline_method=baseline_method,
                    baseline_params=baseline_params,
                    smoothing_window=smoothing_window,
                    smoothing_polyorder=smoothing_polyorder,
                )
                preprocessing_cache[cache_key] = X_prep
            n_features_prep = X_prep.shape[1]

            # Validate preprocessing didn't corrupt data
            assert X_prep.shape[0] == X_raw.shape[0], \
                f"Preprocessing changed sample count! {X_raw.shape[0]} -> {X_prep.shape[0]}"

            # 2b. Apply edge masking for SG derivatives (matches grid search)
            # SG derivatives create boundary artifacts at first/last window//2 wavelengths
            wavelengths_for_trial = wavelengths
            if preprocess_config.get('deriv') and preprocess_config.get('window'):
                X_prep, wavelengths_for_trial, edge_zone_applied = _apply_edge_mask_to_data(
                    X_prep, wavelengths_for_trial, preprocess_config
                )
                n_features_prep = X_prep.shape[1]

            # One-class branch: variable selection + one-class CV
            if task_type == 'one_class':
                from spectral_predict.contamination import run_one_class_cv

                if y_oc is None:
                    return float('inf')

                # --- Variable selection (mirrors regression/classification logic) ---
                subset_type = trial.suggest_categorical('subset_type', available_methods)
                subset_size = trial.suggest_categorical('n_vars', SUBSET_SIZES)
                region_idx = trial.suggest_int('region_id', 0, max(0, n_top_regions - 1))

                # Use y_oc (+1/-1) for all variable selection operations
                y_for_varsel = y_oc

                top_indices = None
                subset_tag = 'full'

                if subset_type == 'region':
                    # Check cache first
                    if cache_key in region_cache:
                        dynamic_regions = region_cache[cache_key]
                    else:
                        # Compute regions DYNAMICALLY on preprocessed data
                        try:
                            if n_features_prep == len(wavelengths_for_trial):
                                wl_prep = wavelengths_for_trial
                            else:
                                wl_prep = np.linspace(
                                    wavelengths_for_trial[0],
                                    wavelengths_for_trial[-1],
                                    n_features_prep,
                                )

                            dynamic_regions = create_region_subsets(
                                X_prep, y_for_varsel, wl_prep.astype(float),
                                n_top_regions=n_top_regions,
                                test_all_individual=region_test_all_individual,
                                test_pairwise=region_test_pairwise,
                            )
                            region_cache[cache_key] = dynamic_regions
                        except Exception as e:
                            logging.warning(
                                f"Dynamic region creation failed for one-class: {e}, "
                                "falling back to empty"
                            )
                            dynamic_regions = []
                            region_cache[cache_key] = dynamic_regions

                    # Use cached region count for index clamping
                    if len(dynamic_regions) > 0:
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
                        imp_cache_key = (cache_key, 'importance', 'LightGBM', 'classification')
                        if imp_cache_key in importance_cache:
                            importances = importance_cache[imp_cache_key]
                        else:
                            importances = compute_importances(
                                X_prep, y_for_varsel, 'importance',
                                'LightGBM', cv_folds, random_state,
                                task_type='classification',
                            )
                            importance_cache[imp_cache_key] = importances
                        top_indices = np.argsort(importances, kind='stable')[-n_vars:]
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

                        # For one-class: use LightGBM as proxy model and
                        # classification task_type so compute_importances builds
                        # a standard classifier on the +1/-1 labels.
                        # Cache per (preprocessing, method) — deterministic for same input.
                        imp_cache_key = (cache_key, subset_type, 'LightGBM', 'classification')
                        if imp_cache_key in importance_cache:
                            importances = importance_cache[imp_cache_key]
                        else:
                            importances = compute_importances(
                                X_prep, y_for_varsel, subset_type,
                                'LightGBM', cv_folds, random_state,
                                task_type='classification',
                            )
                            importance_cache[imp_cache_key] = importances

                        top_indices = np.argsort(importances, kind='stable')[-n_vars:]
                        subset_tag = f"top{n_vars}_{subset_type}"

                # Apply subset
                if top_indices is not None and len(top_indices) > 0:
                    X_for_cv = X_prep[:, top_indices]
                else:
                    X_for_cv = X_prep

                # --- Suggest one-class model params (after subsetting, feature count may differ) ---
                oc_params = suggest_one_class_params(trial, model_name)

                cv_result = run_one_class_cv(
                    X_for_cv, y_oc, model_name, oc_params,
                    n_folds=cv_folds, cv_strategy=cv_strategy,
                    cv_n_repeats=cv_n_repeats, random_state=random_state,
                    y_original=y_original, compute_calibration=True,
                )

                if cv_result.get('skipped', False):
                    # Surface the skip reason so the GUI progress wrapper and
                    # downstream diagnostics can explain WHY the trial bailed
                    # (e.g. "Need at least 3 clean samples to fit DD-SIMCA").
                    skip_reason = cv_result.get('skip_reason', 'unknown')
                    trial.set_user_attr('skip_reason', skip_reason)
                    return float('inf')

                mean_m = cv_result['mean_metrics']
                cal_m = cv_result['cal_metrics']  # Empty when compute_calibration=False

                for key, val in mean_m.items():
                    trial.set_user_attr(f'{key}_cv', val)
                # cal_m is populated only on final calibration runs (not during optimization)
                if cal_m:
                    for key, val in cal_m.items():
                        if key != 'per_contaminant':
                            trial.set_user_attr(key, val)
                    # Store fitted calibration artifacts so any future direct
                    # save-from-results path can persist them without needing a
                    # refinement round-trip. Mirrors search.py:5171-5173 for grid
                    # search. See docs/SESSION_LOG.md 2026-04-10 "One-Class
                    # Prediction Disaster" for context on why this matters.
                    if 'cal_scaler' in cv_result:
                        trial.set_user_attr('cal_scaler', cv_result['cal_scaler'])
                    if 'cal_pca_reducer' in cv_result:
                        trial.set_user_attr('cal_pca_reducer', cv_result['cal_pca_reducer'])
                    if 'oc_score_stats' in cv_result:
                        trial.set_user_attr('oc_score_stats', cv_result['oc_score_stats'])
                trial.set_user_attr('preprocess', preprocess_config)
                trial.set_user_attr('params', oc_params)
                # T-42 Approach C: cv_strategy and cv_n_repeats are now hoisted
                # to study.user_attrs at study creation time (constant per
                # study; previously rewritten on every trial finalize but read
                # by nobody — convert_study_to_dataframe takes them as function
                # parameters).
                # Store preprocessing fields used by convert_study_to_dataframe
                trial.set_user_attr('preprocessing', preprocess_config.get('name', 'raw'))
                trial.set_user_attr('window', preprocess_config.get('window', 0))
                trial.set_user_attr('deriv', preprocess_config.get('deriv', 0))
                trial.set_user_attr('poly', preprocess_config.get('polyorder', 0))
                trial.set_user_attr('apply_baseline', preprocess_config.get('apply_baseline', False))
                trial.set_user_attr('apply_smoothing', preprocess_config.get('apply_smoothing', False))
                # T-36: persist autoscale flag so convert_study_to_dataframe can emit it
                trial.set_user_attr('apply_autoscale', preprocess_config.get('apply_autoscale', False))
                trial.set_user_attr('model_params', str(oc_params))
                trial.set_user_attr('n_vars', X_for_cv.shape[1])
                trial.set_user_attr('full_vars_masked', X_prep.shape[1])
                trial.set_user_attr('subset_tag', subset_tag)
                if top_indices is not None:
                    selected_wl = wavelengths_for_trial[top_indices]
                    trial.set_user_attr('all_wavelengths',
                        ','.join([f"{w:.1f}" for w in selected_wl]))
                    trial.set_user_attr('selected_wavelengths',
                        ','.join([f"{w:.1f}" for w in selected_wl]))
                else:
                    trial.set_user_attr('all_wavelengths',
                        ','.join([f"{w:.1f}" for w in wavelengths_for_trial]))

                balanced_accuracy = mean_m.get('balanced_accuracy', 0.0)
                return -balanced_accuracy

            # 3. Suggest subset type and size
            # IMPORTANT: Always suggest ALL parameters to maintain consistent parameter space
            # Optuna requires the same parameter names to have consistent value spaces
            subset_type = trial.suggest_categorical('subset_type', available_methods)
            subset_size = trial.suggest_categorical('n_vars', SUBSET_SIZES)
            region_idx = trial.suggest_int('region_id', 0, max(0, n_top_regions - 1))

            if subset_type == 'region':
                # Check cache first
                if cache_key in region_cache:
                    dynamic_regions = region_cache[cache_key]
                else:
                    # Compute regions DYNAMICALLY on preprocessed data with full parameters
                    # This ensures regions are relevant to the current preprocessing
                    try:
                        # Create wavelengths for preprocessed data (may have different length)
                        if n_features_prep == len(wavelengths_for_trial):
                            wl_prep = wavelengths_for_trial
                        else:
                            # Interpolate wavelengths if preprocessing changed feature count
                            wl_prep = np.linspace(wavelengths_for_trial[0], wavelengths_for_trial[-1], n_features_prep)

                        dynamic_regions = create_region_subsets(
                            X_prep, y, wl_prep.astype(float),
                            n_top_regions=n_top_regions,
                            test_all_individual=region_test_all_individual,
                            test_pairwise=region_test_pairwise
                        )
                        region_cache[cache_key] = dynamic_regions
                    except Exception as e:
                        logging.warning(f"Dynamic region creation failed: {e}, falling back to empty")
                        dynamic_regions = []
                        region_cache[cache_key] = dynamic_regions

                # Use cached region count for index clamping
                if len(dynamic_regions) > 0:
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
                    imp_cache_key = (cache_key, 'importance', model_name, task_type)
                    if imp_cache_key in importance_cache:
                        importances = importance_cache[imp_cache_key]
                    else:
                        importances = compute_importances(
                            X_prep, y, 'importance', model_name, cv_folds, random_state, task_type
                        )
                        importance_cache[imp_cache_key] = importances
                    top_indices = np.argsort(importances, kind='stable')[-n_vars:]
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

                    # Compute importances (cached per preprocessing + method + model)
                    imp_cache_key = (cache_key, subset_type, model_name, task_type)
                    if imp_cache_key in importance_cache:
                        importances = importance_cache[imp_cache_key]
                    else:
                        importances = compute_importances(
                            X_prep, y, subset_type, model_name, cv_folds, random_state, task_type
                        )
                        importance_cache[imp_cache_key] = importances

                    # Select top variables
                    top_indices = np.argsort(importances, kind='stable')[-n_vars:]
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

            # 5b. Prune invalid PLS trials where n_components > n_features
            # This can happen when variable subset selection reduces features below the
            # suggested n_components. Skip these trials rather than silently clamping.
            if model_name.lower() in ('pls', 'pls-da'):
                n_components = model_params.get('n_components', 2)
                if n_components > n_features_final:
                    logging.debug(
                        f"Trial {trial.number}: Pruning - n_components ({n_components}) > "
                        f"n_features ({n_features_final})"
                    )
                    return 1e10  # Return penalty to skip invalid combination

            # 6. Build and cross-validate model
            model = build_model(model_name, model_params, task_type=task_type)

            # Scale-sensitive models need StandardScaler (matches search.py behavior)
            # For PLS-DA: PLS + StandardScaler + LogisticRegression (search.py lines 3417-3424)
            # For scale-sensitive models: StandardScaler + Model (search.py lines 3427-3429)
            SCALE_SENSITIVE_MODELS = {'SVC', 'SVR', 'MLP', 'NeuralBoosted', 'Ridge', 'Lasso', 'ElasticNet'}

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
                # Extract LogisticRegression parameters (prefixed with lr_) if available
                lr_C = model_params.get('lr_C', 1.0) if model_params else 1.0
                lr_solver = model_params.get('lr_solver', 'lbfgs') if model_params else 'lbfgs'
                lr_max_iter = model_params.get('lr_max_iter', 1000) if model_params else 1000
                lr = LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=random_state)
                # Apply class_weight to LogisticRegression if specified
                if imbalance_method == 'class_weight':
                    lr.set_params(class_weight='balanced')
                pipe_steps.append(('lr', lr))
            elif model_name in SCALE_SENSITIVE_MODELS:
                # Scale-sensitive models: StandardScaler + Model.
                # T-36: skip the per-model scaler when autoscale is already
                # applied at the preprocessing stage — avoids redundant scaling.
                if not preprocess_config.get('apply_autoscale', False):
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

            # Enable CV parallelism (safe - Bayesian trials are sequential)
            from spectral_predict.search import _frozen_needs_threading_fallback

            # Models that are slower with parallel CV (threading conflicts or low overhead)
            # SVM: threading conflicts; PLS/PLS-DA: so fast that joblib overhead dominates
            # Ridge/Lasso/ElasticNet: linear solve is ~5ms, joblib spawn overhead is ~1s on Windows
            models_prefer_serial_cv = {'SVM', 'PLS', 'PLS-DA', 'Ridge', 'Lasso', 'ElasticNet'}
            use_serial = _frozen_needs_threading_fallback() or model_name in models_prefer_serial_cv

            n_jobs_cv = 1 if use_serial else -1

            # 7. Compute metrics
            if task_type == 'regression':
                # Compute pooled CV predictions once and derive both RMSE and R² from them.
                # Averaging per-fold R² is mathematically incorrect (different SS_tot per fold),
                # and averaging per-fold RMSE degenerates to MAE under LOO (1-sample test folds).
                # This matches chemometrics convention (Unscrambler, PLS_Toolbox, SIMCA, IUPAC)
                # and the method used in search.py for consistency with Model Development.
                if use_early_stopping:
                    y_pred_cv = cross_val_predict_with_early_stopping(
                        model, X_final, y, cv=cv,
                        early_stopping_rounds=early_stopping_rounds
                    )
                else:
                    y_pred_cv = cross_val_predict_pooled(model, X_final, y, cv=cv, n_jobs=n_jobs_cv)
                rmse = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
                r2 = r2_score(y, y_pred_cv)

                # Compute additional NIR spectroscopy metrics from CV predictions
                mae_cv = mean_absolute_error(y, y_pred_cv)
                bias_cv = float(np.mean(y_pred_cv - y))
                y_std = float(np.std(y))
                y_range = float(np.ptp(y))
                rpd = y_std / rmse if rmse > 0 else 0.0
                rer = y_range / rmse if rmse > 0 else 0.0
                ccc_cv = lins_ccc(y, y_pred_cv)

                # Compute regional RMSE (per-quartile performance) for coloring in Results tab
                # This enables the same quartile-based highlighting as Grid search
                quartiles = np.percentile(y, [25, 50, 75])
                regional_rmse = {}
                for i, (lower, upper) in enumerate([
                    (-np.inf, quartiles[0]),  # Q1
                    (quartiles[0], quartiles[1]),  # Q2
                    (quartiles[1], quartiles[2]),  # Q3
                    (quartiles[2], np.inf)  # Q4
                ]):
                    # Use true Y values for mask
                    mask = (y >= lower) & (y < upper if i < 3 else y >= lower)
                    if mask.sum() > 0:
                        regional_rmse[f'Q{i+1}'] = float(np.sqrt(mean_squared_error(
                            y[mask], y_pred_cv[mask]
                        )))
                    else:
                        regional_rmse[f'Q{i+1}'] = np.nan
                y_quartiles = quartiles.tolist()

                metric = rmse  # Minimize RMSE
            else:
                # Classification: use accuracy and ROC_AUC
                # Compute pooled CV predictions once and derive accuracy + other metrics from them.
                # Pooled accuracy (sample-weighted across all folds) differs slightly from
                # per-fold-averaged accuracy when folds have unequal sizes — the pooled form
                # matches how RMSE/R² are computed above and matches scikit-learn's
                # cross_val_predict convention. (IUPAC's CV guidance is regression-specific.)
                # This also saves a full CV pass per trial (was 2 passes: score + predict).
                if use_early_stopping:
                    y_pred_cv = cross_val_predict_with_early_stopping(
                        model, X_final, y, cv=cv,
                        early_stopping_rounds=early_stopping_rounds
                    )
                else:
                    y_pred_cv = cross_val_predict_pooled(model, X_final, y, cv=cv, n_jobs=n_jobs_cv)
                accuracy = float(accuracy_score(y, y_pred_cv))

                # Compute ROC_AUC using cross_val_predict for probability estimates
                try:
                    if use_early_stopping:
                        y_proba = cross_val_predict_with_early_stopping(
                            model, X_final, y, cv=cv,
                            early_stopping_rounds=early_stopping_rounds,
                            method='predict_proba'
                        )
                    else:
                        y_proba = cross_val_predict_pooled(
                            model, X_final, y, cv=cv, method='predict_proba', n_jobs=n_jobs_cv
                        )
                    n_classes = len(np.unique(y))
                    if n_classes == 2:
                        # Binary classification
                        roc_auc = roc_auc_score(y, y_proba[:, 1])
                    else:
                        # Multiclass - use weighted average
                        roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')

                    # Compute Log Loss from probabilities
                    try:
                        logloss_cv = log_loss(y, y_proba)
                    except Exception:
                        logloss_cv = np.nan
                except Exception:
                    roc_auc = np.nan
                    logloss_cv = np.nan

                # Compute additional classification metrics from CV predictions
                # Determine averaging method (binary or macro)
                n_classes = len(np.unique(y))
                average_method = 'binary' if n_classes == 2 else 'macro'

                try:
                    f1_cv = f1_score(y, y_pred_cv, average=average_method, zero_division=0)
                except Exception:
                    f1_cv = np.nan

                try:
                    precision_cv = precision_score(y, y_pred_cv, average=average_method, zero_division=0)
                except Exception:
                    precision_cv = np.nan

                try:
                    recall_cv = recall_score(y, y_pred_cv, average=average_method, zero_division=0)
                except Exception:
                    recall_cv = np.nan

                try:
                    specificity_cv = compute_specificity(y, y_pred_cv, average='macro')
                except Exception:
                    specificity_cv = np.nan

                try:
                    kappa_cv = cohen_kappa_score(y, y_pred_cv)
                except Exception:
                    kappa_cv = np.nan

                try:
                    mcc_cv = matthews_corrcoef(y, y_pred_cv)
                except Exception:
                    mcc_cv = np.nan

                try:
                    balanced_acc_cv = balanced_accuracy_score(y, y_pred_cv)
                    ber_cv = 1.0 - balanced_acc_cv
                except Exception:
                    balanced_acc_cv = np.nan
                    ber_cv = np.nan

                # Compute per-class metrics for coloring in Results tab
                # This enables the same class-based highlighting as Grid search
                per_class_metrics = {}
                class_labels = None
                try:
                    report = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
                    class_labels = sorted([k for k in report.keys()
                                           if k not in ['accuracy', 'macro avg', 'weighted avg']])
                    for class_label in class_labels:
                        class_key = str(class_label)
                        if class_key in report:
                            per_class_metrics[class_key] = {
                                'F1': report[class_key]['f1-score'],
                                'Precision': report[class_key]['precision'],
                                'Recall': report[class_key]['recall'],
                                'Support': report[class_key]['support']
                            }
                except Exception:
                    pass

                metric = -accuracy  # Minimize negative accuracy

            # Store additional info in trial
            trial.set_user_attr('preprocessing', preprocess_config['name'])
            trial.set_user_attr('window', preprocess_config.get('window', 0))
            trial.set_user_attr('deriv', preprocess_config.get('deriv', 0))
            trial.set_user_attr('poly', preprocess_config.get('polyorder', 0))
            trial.set_user_attr('apply_baseline', preprocess_config.get('apply_baseline', False))
            trial.set_user_attr('apply_smoothing', preprocess_config.get('apply_smoothing', False))
            # T-36: persist autoscale flag for convert_study_to_dataframe emission
            trial.set_user_attr('apply_autoscale', preprocess_config.get('apply_autoscale', False))
            trial.set_user_attr('subset_type', subset_type)
            trial.set_user_attr('subset_tag', subset_tag)
            trial.set_user_attr('n_vars', n_vars)
            trial.set_user_attr('model_params', str(model_params))
            # T-42 Approach C: early_stopping_rounds + cv_strategy + cv_n_repeats
            # are constant per study; hoisted to study.user_attrs at study
            # creation time so the per-trial finalize commits one fewer write
            # and the SQLite WAL log carries that much less duplication.

            # Fit on full training data for calibration metrics
            model.fit(X_final, y)
            captured_params = _capture_serializable_params(model)
            if captured_params:
                trial.set_user_attr('model_params', str(captured_params))
            y_pred_cal = model.predict(X_final)

            if task_type == 'regression':
                cal_rmse = np.sqrt(mean_squared_error(y, y_pred_cal))
                cal_r2 = r2_score(y, y_pred_cal)
                cal_ccc = lins_ccc(y, y_pred_cal)
                trial.set_user_attr('RMSE', cal_rmse)      # Calibration
                trial.set_user_attr('R2', cal_r2)          # Calibration
                trial.set_user_attr('CCC', cal_ccc)        # Calibration
                trial.set_user_attr('RMSEcv', rmse)        # CV (was RMSE)
                trial.set_user_attr('R2cv', r2)            # CV (was R2)
                trial.set_user_attr('CCCcv', ccc_cv)       # CV
                # NIR-specific metrics (computed from aggregated CV predictions)
                trial.set_user_attr('MAEcv', mae_cv)
                trial.set_user_attr('RPD', rpd)
                trial.set_user_attr('Bias', bias_cv)
                trial.set_user_attr('RER', rer)
                # Regional RMSE for quartile-based coloring in Results tab
                trial.set_user_attr('regional_rmse', regional_rmse)
                trial.set_user_attr('y_quartiles', y_quartiles)
            else:
                # Calibration metrics
                cal_accuracy = accuracy_score(y, y_pred_cal)
                trial.set_user_attr('Accuracy', cal_accuracy)    # Calibration
                trial.set_user_attr('Accuracycv', accuracy)      # CV

                # Calibration ROC AUC and Log Loss
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba_cal = model.predict_proba(X_final)
                        n_classes = len(np.unique(y))
                        if n_classes == 2:
                            cal_roc_auc = roc_auc_score(y, y_proba_cal[:, 1])
                        else:
                            cal_roc_auc = roc_auc_score(y, y_proba_cal, multi_class='ovr', average='weighted')
                        trial.set_user_attr('ROC_AUC', cal_roc_auc)     # Calibration

                        # Calibration Log Loss
                        try:
                            cal_logloss = log_loss(y, y_proba_cal)
                            trial.set_user_attr('LogLoss', cal_logloss)
                        except Exception:
                            trial.set_user_attr('LogLoss', np.nan)
                    else:
                        trial.set_user_attr('LogLoss', np.nan)
                except Exception:
                    trial.set_user_attr('ROC_AUC', np.nan)
                    trial.set_user_attr('LogLoss', np.nan)

                trial.set_user_attr('ROC_AUCcv', roc_auc)          # CV

                # Calibration F1, Precision, Recall
                try:
                    cal_f1 = f1_score(y, y_pred_cal, average='weighted', zero_division=0)
                    trial.set_user_attr('F1', cal_f1)
                except Exception:
                    trial.set_user_attr('F1', np.nan)

                try:
                    cal_precision = precision_score(y, y_pred_cal, average='weighted', zero_division=0)
                    trial.set_user_attr('Precision', cal_precision)
                except Exception:
                    trial.set_user_attr('Precision', np.nan)

                try:
                    cal_recall = recall_score(y, y_pred_cal, average='weighted', zero_division=0)
                    trial.set_user_attr('Recall', cal_recall)
                except Exception:
                    trial.set_user_attr('Recall', np.nan)

                # Calibration additional metrics
                try:
                    cal_specificity = compute_specificity(y, y_pred_cal, average='macro')
                    trial.set_user_attr('Specificity', cal_specificity)
                except Exception:
                    trial.set_user_attr('Specificity', np.nan)

                try:
                    cal_kappa = cohen_kappa_score(y, y_pred_cal)
                    trial.set_user_attr('Kappa', cal_kappa)
                except Exception:
                    trial.set_user_attr('Kappa', np.nan)

                try:
                    cal_mcc = matthews_corrcoef(y, y_pred_cal)
                    trial.set_user_attr('MCC', cal_mcc)
                except Exception:
                    trial.set_user_attr('MCC', np.nan)

                try:
                    cal_balanced_acc = balanced_accuracy_score(y, y_pred_cal)
                    cal_ber = 1.0 - cal_balanced_acc
                    trial.set_user_attr('BalancedAcc', cal_balanced_acc)
                    trial.set_user_attr('BER', cal_ber)
                except Exception:
                    trial.set_user_attr('BalancedAcc', np.nan)
                    trial.set_user_attr('BER', np.nan)

                # CV metrics
                trial.set_user_attr('F1cv', f1_cv)
                trial.set_user_attr('Precisioncv', precision_cv)
                trial.set_user_attr('Recallcv', recall_cv)
                trial.set_user_attr('Specificitycv', specificity_cv)
                trial.set_user_attr('Kappacv', kappa_cv)
                trial.set_user_attr('MCCcv', mcc_cv)
                trial.set_user_attr('BalancedAcccv', balanced_acc_cv)
                trial.set_user_attr('BERcv', ber_cv)
                trial.set_user_attr('LogLosscv', logloss_cv)
                # Per-class metrics for class-based coloring in Results tab
                trial.set_user_attr('per_class_metrics', per_class_metrics)
                trial.set_user_attr('class_labels', class_labels)

            # Store selected wavelengths in TRAINING ORDER (importance order)
            # CRITICAL: Do NOT sort - Model Development expects wavelengths in the same
            # order they were used during training. Grid Search also preserves training
            # order (see search.py line 3201).
            # Use wavelengths_for_trial (edge-masked) to match actual training data
            if top_indices is not None:
                selected_wavelengths = wavelengths_for_trial[top_indices] if len(wavelengths_for_trial) > max(top_indices) else []
                # Store ALL wavelengths for model reconstruction (training order)
                trial.set_user_attr('all_wavelengths', ','.join([f"{w:.1f}" for w in selected_wavelengths]))
                # Store first 50 for display (also training order - most important first)
                trial.set_user_attr('selected_wavelengths',
                    ','.join([f"{w:.1f}" for w in selected_wavelengths[:50]]))
            else:
                # Full spectrum - store all wavelengths (edge-masked)
                trial.set_user_attr('all_wavelengths', ','.join([f"{w:.1f}" for w in wavelengths_for_trial]))
            # Store edge-masked feature count for full_vars
            trial.set_user_attr('full_vars_masked', len(wavelengths_for_trial))

            return metric

        except Exception as e:
            logging.warning(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
            # Return large penalty
            if task_type == 'regression':
                return 1e10
            else:
                return 1e10

    return objective


# ---------------------------------------------------------------------------
# T-41: SQLite WAL helpers + study migration
# ---------------------------------------------------------------------------

def _sqlite_path_from_url(sqlite_url: str) -> str:
    """Extract the filesystem path from an Optuna ``sqlite:///...`` URL.

    Uses urllib.parse so UNC and four-slash forms (``sqlite:////host/share/..``)
    don't silently turn into a different path than what Optuna writes to.
    """
    import os as _os
    import urllib.parse

    parsed = urllib.parse.urlparse(sqlite_url)
    # urlparse on sqlite:///path puts the body in .path with a leading slash.
    # On Windows that becomes "/C:/.../file.sqlite3"; strip the leading slash
    # so sqlite3.connect sees a native path.
    raw = parsed.path
    if _os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return raw


def _apply_wal_pragmas(sqlite_url: str) -> bool:
    """Apply WAL + tuned pragmas to a SQLite Optuna database.

    Returns True iff WAL mode was actually accepted by the engine. A return
    value of False means the filesystem rejected WAL (network share, certain
    AV-shimmed paths) and SQLite silently fell back to its previous mode —
    the user is paying the auto-calc decision cost without getting the WAL
    speedup. Caller should surface that.

    SQLite's ``PRAGMA journal_mode = WAL`` does NOT raise on rejection; it
    returns the journal mode that was actually applied. We must read the
    row and verify, otherwise the silent-fallback case is invisible.
    """
    import sqlite3 as _sqlite3

    db_path = _sqlite_path_from_url(sqlite_url)
    try:
        conn = _sqlite3.connect(db_path)
        try:
            cur = conn.execute("PRAGMA journal_mode = WAL")
            row = cur.fetchone()
            applied = (row[0].lower() if row and row[0] else "?")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA wal_autocheckpoint = 50")
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("T-41: could not apply WAL pragmas to %s: %s", db_path, exc)
        return False

    if applied != "wal":
        logger.warning(
            "T-41: filesystem rejected WAL on %s (got journal_mode=%r); "
            "crash-resume writes will be slower than expected",
            db_path,
            applied,
        )
        return False
    return True


def _make_tpe_sampler(random_state: int) -> TPESampler:
    """Construct a fresh TPE sampler with dasp's standard settings."""
    return TPESampler(
        seed=random_state,
        n_startup_trials=20,
        n_ei_candidates=32,
        multivariate=True,
        consider_endpoints=True,
        warn_independent_sampling=False,
    )


def _migrate_study_to_sqlite(
    study: optuna.Study,
    sqlite_url: str,
    study_name: str,
    random_state: int,
) -> optuna.Study:
    """Migrate a running in-memory Optuna study to a SQLite backend.

    Uses ``optuna.copy_study`` (data copy) followed by
    ``optuna.load_study(sampler=TPESampler(...))`` (sampler attach).

    CRITICAL: ``optuna.create_study(load_if_exists=True, sampler=...)``
    SILENTLY IGNORES the sampler kwarg on existing studies (Optuna 4.8
    empirically verified by DeepSeek V4 Pro Max on 2026-05-01).  The only
    correct way to attach a sampler to an existing study is
    ``optuna.load_study(sampler=...)``.

    After migration TPE rebuilds its internal state from the copied trials on
    the next ``study.optimize()`` call, so exploitation continues seamlessly.

    Args:
        study: The active in-memory study to migrate.
        sqlite_url: Optuna-style ``sqlite:///...`` URL for the target file.
        study_name: Study name to use in the SQLite backend.
        random_state: Random seed for the fresh TPE sampler.

    Returns:
        A new ``optuna.Study`` object backed by SQLite, containing all
        trials from the original in-memory study.
    """
    # Step 1: copy all trials, params, user_attrs, directions to SQLite.
    optuna.copy_study(
        from_study_name=study.study_name,
        to_study_name=study_name,
        from_storage=study._storage,
        to_storage=sqlite_url,
    )

    # Step 2: apply WAL pragmas — copy_study just created the SQLite file.
    _apply_wal_pragmas(sqlite_url)

    # Step 3: load study with a fresh sampler.  This is the ONLY way to attach
    # a sampler to an existing study; TPE will rebuild from copied trials.
    return optuna.load_study(
        study_name=study_name,
        storage=sqlite_url,
        sampler=_make_tpe_sampler(random_state),
    )


def run_unified_bayesian(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    model_name: str,
    task_type: str = 'regression',
    n_trials: int = 300,
    cv_folds: int = 5,
    cv_strategy: str = 'kfold',
    cv_n_repeats: int = 5,
    n_top_regions: int = 10,
    random_state: int = 42,
    progress_callback: Optional[Callable] = None,
    verbose: bool = True,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: Optional[int] = 40,
    region_test_all_individual: bool = False,
    region_test_pairwise: bool = False,
    controller=None,  # For pause/resume/stop support
    baseline_method: str | None = None,
    baseline_params: dict | None = None,
    smoothing: bool = False,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    enable_autoscale: bool = False,  # T-36
    enable_uve: bool = False,
    inlier_class_label=None,
    enable_sqlite_persistence: "PersistenceMode" = "auto",  # T-41: 'auto' | 'always' | 'never'
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
    early_stopping_rounds : int, optional, default=50
        Number of rounds without improvement before stopping for boosting models
        (XGBoost, CatBoost, LightGBM). Set to None or 0 to disable.
        Early stopping typically saves 30-50% training time and often improves
        model quality by preventing overfitting.
    region_test_all_individual : bool, default=False
        Test all N regions individually (not just top 5)
    region_test_pairwise : bool, default=False
        Test all C(N,2) pairwise region combinations
    baseline_method : str or None
        Baseline correction method (e.g., 'als', 'polynomial', 'rubber_band', 'airpls').
        When set, Optuna toggles apply/skip per trial.
    baseline_params : dict or None
        Parameters for the baseline correction method
    smoothing : bool
        Whether to let Optuna toggle Savitzky-Golay smoothing per trial
    smoothing_window : int
        Smoothing window length (used when smoothing=True)
    smoothing_polyorder : int
        Smoothing polynomial order (used when smoothing=True)
    enable_uve : bool, default=False
        Include UVE (Uninformative Variable Elimination) as a variable selection method
    enable_sqlite_persistence : str, default='auto'
        Controls Optuna SQLite crash-resume persistence. T-41 auto-calculator.
        - 'auto'   : (default) first 10 trials in-memory; then decides based on median fit
                     time. If median > 1.0s the study migrates to SQLite+WAL (overhead
                     ~1.2x). If median <= 1.0s stays in-memory (fast models like PLS are 8x
                     slower with persistence, so the auto-calculator keeps them in-memory).
        - 'always' : SQLite+WAL from trial 0 for every model. Accepts the speed cost in
                     exchange for crash-resume on all models.
        - 'never'  : always in-memory; no SQLite file created. Zero overhead but no
                     crash-resume even for slow models.

        The default was 'never' briefly during T-41 ship; flipped back to 'auto' on
        2026-05-02 for resume-path troubleshooting after the user reported the resume
        banner was a no-op (root cause: GUI radio button at 'never' silently ignored the
        loaded SQLite URL). With the resume-override fix and the auto-decision restart
        pattern in place, 'auto' is again a sensible default.

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
        # One-class models
        'pca-simca': 'PCA-SIMCA',
        'oneclasssvm': 'OneClassSVM',
        'isolationforest': 'IsolationForest',
        'ellipticenvelope': 'EllipticEnvelope',
        'lof': 'LOF',
    }
    model_name = model_name_map.get(model_name.lower(), model_name)

    X = np.asarray(X)
    y = np.asarray(y)
    wavelengths = np.asarray(wavelengths)

    # Drop rows where y is NaN (safety net for data with empty rows)
    nan_mask = pd.isna(y)
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(f"Warning: Dropping {n_dropped} sample(s) with NaN target values before optimization.")
        X = X[~nan_mask]
        y = y[~nan_mask]

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

    # Substitute regression sample weighting methods that don't work with cross_val_score
    # These methods (binning, rare_boost, balanced) require manual sample_weight extraction
    # which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
    UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
    if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
        original_method = imbalance_method
        imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
        warn_msg = f"'{original_method}' requires Grid Search. Using 'smogn' instead for Bayesian optimization."
        logger.warning(warn_msg)
        if progress_callback:
            progress_callback({'message': f"[Warning] {warn_msg}"})
        elif verbose:
            print(f"Note: {warn_msg}")

    # Validate imbalance configuration for classification
    if task_type == 'classification' and imbalance_method is not None:
        validate_classification_config(
            y=y,
            imbalance_method=imbalance_method,
            imbalance_params=imbalance_params,
            n_folds=cv_folds
        )

    # Backend guard for CV strategy vs class distribution (runs regardless of
    # imbalance_method — scripted callers can bypass the GUI). Also validates
    # n_repeats for Repeated K-Fold and inlier counts for one-class.
    from .cv_utils import validate_cv_strategy_for_task
    validate_cv_strategy_for_task(
        strategy=cv_strategy,
        task_type=task_type,
        y=y,
        n_folds=cv_folds,
        n_repeats=cv_n_repeats,
        inlier_label=inlier_class_label if task_type == 'one_class' else None,
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
        if task_type == 'one_class':
            print(f"Inlier class: {inlier_class_label}")
            print(f"Variable selection: enabled (importance, CARS, region via LightGBM proxy)")
        else:
            print(f"Regional subsets: dynamically computed ({n_top_regions} regions)")
            methods_str = "importance, CARS, region" + (", UVE" if enable_uve else "")
            print(f"Variable methods: {methods_str}")
        # Show early stopping status for boosting models
        if model_name in ('XGBoost', 'LightGBM', 'CatBoost'):
            if early_stopping_rounds and early_stopping_rounds > 0:
                print(f"Early stopping: enabled ({early_stopping_rounds} rounds)")
            else:
                print(f"Early stopping: disabled")
        if baseline_method:
            print(f"Baseline correction: {baseline_method} (toggled by Optuna)")
        if smoothing:
            print(f"Smoothing: SG window={smoothing_window}, poly={smoothing_polyorder} (toggled by Optuna)")
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
        cv_strategy=cv_strategy,
        cv_n_repeats=cv_n_repeats,
        random_state=random_state,
        n_top_regions=n_top_regions,
        progress_callback=progress_callback,
        imbalance_method=imbalance_method,
        imbalance_params=imbalance_params,
        early_stopping_rounds=early_stopping_rounds,
        region_test_all_individual=region_test_all_individual,
        region_test_pairwise=region_test_pairwise,
        baseline_method=baseline_method,
        baseline_params=baseline_params,
        smoothing=smoothing,
        smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder,
        enable_autoscale=enable_autoscale,  # T-36
        enable_uve=enable_uve,
        inlier_class_label=inlier_class_label,
        y_original=y,
    )

    # Create TPE sampler with good defaults
    sampler = TPESampler(
        seed=random_state,
        n_startup_trials=20,  # Random exploration first
        n_ei_candidates=32,   # More candidates for better exploration
        multivariate=True,    # Model parameter interactions
        consider_endpoints=True,
        warn_independent_sampling=False  # Suppress dynamic space warning
    )

    # Create study. T-11 D: when a run-state storage URL is active, persist
    # to SQLite so a crash/force-quit leaves the trials recoverable.
    # `load_if_exists=True` means re-running the same model picks up where
    # it left off; Optuna skips already-completed trials and continues from
    # the cutoff trial.
    #
    # Codex HIGH (study name): the study_name must be config-sensitive so
    # that changing task_type / CV / random_state / imbalance / preprocessing
    # creates a NEW study instead of silently mixing incompatible trials
    # via load_if_exists=True. Without this fingerprint, a user could
    # change task_type='classification' and inherit RMSEcv-minimizing trials
    # from a prior regression run — the trial values would be wrong but
    # Optuna would happily resume.
    import hashlib as _hashlib
    from spectral_predict.run_state import (
        PersistenceMode,
        _validate_persistence_mode,
        get_storage_url,
    )

    _validate_persistence_mode(enable_sqlite_persistence)

    # Kimi MINOR #6: omit n_trials from the identity hash. If included,
    # changing the trial target (100 → 200) creates a new study and orphans
    # the old one in SQLite. With n_trials excluded, the existing study is
    # reused and the trial-count clamp at the optimize() call computes
    # remaining = max(0, target - already_finished) — the user gets the
    # extra 100 trials added to the existing study rather than a fresh
    # 200-trial study competing with the orphan.
    #
    # Kimi MAJOR #1 partial: include dasp __version__ as a defensive proxy
    # for "code path may have changed." A user upgrading dasp gets a fresh
    # study even if the surface params look identical. Real model-specific
    # search-space changes still need careful attention; documented as a
    # known limitation in T-11 verdict §8.
    try:
        from spectral_predict import __version__ as _dasp_version
    except (ImportError, AttributeError):
        # Codex meta-review: narrowed from `except Exception:` so a real bug
        # in the import chain (circular import, syntax error in __init__) is
        # surfaced rather than silently degrading the fingerprint to
        # "unknown" — that would mix incompatible-version trials into one
        # study via load_if_exists=True.
        _dasp_version = "unknown"

    # Codex meta-review: include caller-visible config that affects the
    # search space or the objective. Two prior reviewers (synthesis +
    # Codex) flagged that omitting these means changing them silently
    # reuses an incompatible study via load_if_exists=True.
    _wavelength_shape = (
        tuple(wavelengths.shape) if hasattr(wavelengths, "shape") else None
    )
    _imbalance_params_str = (
        repr(sorted((imbalance_params or {}).items())) if imbalance_params else "none"
    )
    _baseline_params_str = (
        repr(sorted((baseline_params or {}).items())) if baseline_params else "none"
    )
    config_components = (
        f"version={_dasp_version}|"
        f"task={task_type}|"
        f"cv={cv_strategy}_{cv_folds}_{cv_n_repeats}|"
        f"seed={random_state}|"
        f"imbalance={imbalance_method or 'none'}|"
        f"imbalance_params={_imbalance_params_str}|"
        f"baseline={baseline_method or 'none'}|"
        f"baseline_params={_baseline_params_str}|"
        f"smoothing={smoothing}_{smoothing_window}_{smoothing_polyorder}|"
        f"autoscale={enable_autoscale}|"
        f"uve={enable_uve}|"
        f"inlier={inlier_class_label}|"
        f"wl_shape={_wavelength_shape}|"
        f"n_top_regions={n_top_regions}|"
        f"region_indiv={region_test_all_individual}|"
        f"region_pair={region_test_pairwise}|"
        f"early_stop={early_stopping_rounds}"
    )
    config_hash = _hashlib.sha256(config_components.encode("utf-8")).hexdigest()[:8]
    study_name = f"unified_bayesian_{model_name}_{config_hash}"

    # ---------------------------------------------------------------------------
    # T-41: SQLite auto-calculator — decide storage mode per model.
    # ---------------------------------------------------------------------------
    storage_url = get_storage_url()

    # 'never': pure in-memory; ignore any active storage URL entirely.
    # 'always': SQLite from trial 0 (if a storage URL is available).
    # 'auto': first 10 trials in-memory, then decide based on median fit time.
    _persistence_mode = enable_sqlite_persistence  # already validated above

    if _persistence_mode == "always" and storage_url is not None:
        # Always-on: create SQLite study from trial 0 (T-41 Task 2).
        create_kwargs: dict = {
            "direction": "minimize",
            "study_name": study_name,
            "storage": storage_url,
            "load_if_exists": True,
        }
        # NOTE: sampler kwarg is intentionally NOT passed here — on existing
        # studies, create_study(load_if_exists=True, sampler=...) SILENTLY
        # IGNORES the sampler.  We pass None here so Optuna uses its default
        # for a new study, then immediately re-open via load_study to attach.
        # For a brand-new study this is a two-step no-op; for an existing
        # resume it correctly reattaches TPE.
        study = optuna.create_study(**create_kwargs)
        # Apply WAL pragmas immediately — Optuna may create the file lazily on
        # the first access, so we trigger that by doing a read on the study.
        try:
            _ = len(study.trials)  # forces lazy SQLite init
        except Exception:
            pass
        _apply_wal_pragmas(storage_url)
        # Reattach a fresh TPE sampler (the only safe way per Optuna 4.8).
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            sampler=_make_tpe_sampler(random_state),
        )
        _sqlite_decided = True
    else:
        # 'never' OR 'auto' (warmup window) — start in-memory.
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
        )
        _sqlite_decided = (_persistence_mode == "never")  # 'never' is final

    # T-42 Approach C: hoist the three keys that are constant per study to
    # study.user_attrs so the per-trial finalize doesn't re-emit them. These
    # survive migration via optuna.copy_study (preserves user_attrs along
    # with trials). Applied to both branches above; resume picks up the
    # existing study's user_attrs unchanged.
    _hoist_es = (
        early_stopping_rounds
        if (
            early_stopping_rounds is not None
            and early_stopping_rounds > 0
            and model_name in ('XGBoost', 'LightGBM', 'CatBoost')
        )
        else None
    )
    study.set_user_attr('cv_strategy', cv_strategy)
    study.set_user_attr('cv_n_repeats', cv_n_repeats)
    study.set_user_attr('early_stopping_rounds', _hoist_es)

    # --- auto-decision state ---
    _AUTO_WARMUP = 10       # warmup trials before the auto-calculator decides
    _AUTO_THRESHOLD_S = 1.0  # median fit > 1.0s -> SQLite ON (ratio ~1.2x)
    _auto_migrated = False   # True after the in-memory -> SQLite migration

    # Progress callback wrapper (must reference `study` via mutable container
    # so the auto-migration can swap it after the warmup window).
    _study_ref: list[optuna.Study] = [study]

    def progress_wrapper(cb_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal _sqlite_decided, _auto_migrated

        # Check for stop/pause signal from controller.
        # DeepSeek HIGH #2 fix: stop() must target cb_study (the study Optuna
        # is actively optimizing) not _study_ref[0]. After auto-migration,
        # _study_ref[0] is the migrated SQLite study which is NOT in the
        # optimize loop, and Study.stop() on it would raise RuntimeError.
        if controller is not None:
            if not controller.check_and_wait():
                cb_study.stop()
                return

        # --- T-41 auto-decision logic (only runs during 'auto' warmup) ---
        if (
            _persistence_mode == "auto"
            and not _sqlite_decided
            and storage_url is not None
        ):
            from optuna.trial import TrialState as _TS

            # DeepSeek MEDIUM #1 fix: gate on TOTAL trials seen, not completed
            # ones. Otherwise one-class runs with high CV-skip rates can stall
            # the warmup-window decision past the entire trial budget. The
            # n_completed < 3 fallback is now reachable.
            total_seen = len(cb_study.trials)
            completed = [
                t for t in cb_study.trials
                if t.state == _TS.COMPLETE and t.duration is not None
            ]
            n_completed = len(completed)

            if total_seen >= _AUTO_WARMUP:
                _sqlite_decided = True  # only decide once

                # Fallback: if fewer than 3 trials completed (e.g., one-class
                # with many CV-skip failures), default SQLite ON conservatively
                # — we don't trust an empty-distribution decision.
                if n_completed < 3:
                    use_sqlite = True
                    median_s = float("nan")
                    decision_reason = (
                        f"Auto-enabled SQLite (only {n_completed} of {total_seen} "
                        "trials completed — conservative default)"
                    )
                else:
                    durations = sorted(
                        t.duration.total_seconds() for t in completed
                    )
                    mid = len(durations) // 2
                    median_s = (
                        durations[mid]
                        if len(durations) % 2 == 1
                        else (durations[mid - 1] + durations[mid]) / 2.0
                    )
                    use_sqlite = median_s > _AUTO_THRESHOLD_S
                    if use_sqlite:
                        ratio = 1.0 + 0.200 / median_s  # approx overhead ratio
                        decision_reason = (
                            f"Auto-enabled SQLite (median trial = {median_s:.2f}s, "
                            f"overhead ratio ~{ratio:.2f}x)"
                        )
                    else:
                        slowdown = 1.0 + 0.200 / max(median_s, 1e-6)
                        decision_reason = (
                            f"Auto-disabled SQLite (median trial = {median_s * 1000:.0f}ms, "
                            f"would slow run by {slowdown:.1f}x)"
                        )

                logger.info("T-41 auto-calculator: %s", decision_reason)
                if progress_callback:
                    progress_callback(
                        {
                            "stage": "unified_bayesian",
                            "current": trial.number + 1,
                            "total": n_trials,
                            "message": f"[T-41] {decision_reason}",
                            "t41_decision": decision_reason,
                        }
                    )

                if use_sqlite:
                    # Optuna.optimize() captures the study by reference; swapping
                    # _study_ref here doesn't redirect writes. Stop the in-memory
                    # loop, the outer scope restarts on _study_ref[0] so trials
                    # 11..N land directly in SQLite.
                    try:
                        migrated = _migrate_study_to_sqlite(
                            cb_study, storage_url, study_name, random_state
                        )
                        _study_ref[0] = migrated
                        _auto_migrated = True
                        cb_study.stop()
                        return
                    except Exception as exc:
                        # Partial-success cleanup: copy_study may have created the
                        # study row in SQLite before load_study failed. Use
                        # optuna.delete_study to remove ONLY this study from
                        # the database — multi-model runs share one SQLite file,
                        # so unlinking the file would nuke prior models' trials.
                        # _auto_migrated stays False so the outer scope doesn't
                        # try to restart on a half-broken study.
                        logger.warning(
                            "T-41: SQLite migration failed; staying in-memory (no crash-resume for this run): %s",
                            exc,
                        )
                        try:
                            optuna.delete_study(
                                study_name=study_name,
                                storage=storage_url,
                            )
                        except Exception as cleanup_exc:
                            # KeyError if the study row was never written, or
                            # any other delete failure — log and continue.
                            logger.warning(
                                "T-41: could not delete failed-migration study from SQLite: %s",
                                cleanup_exc,
                            )
                        if progress_callback:
                            progress_callback({
                                "stage": "unified_bayesian",
                                "current": trial.number + 1,
                                "total": n_trials,
                                "message": f"[T-41] WARNING: SQLite migration failed — staying in-memory (no crash-resume). Reason: {exc}",
                                "t41_decision": "migration_failed_inmemory",
                            })
                # else: stay in-memory — no action needed.

        if progress_callback:
            progress_info = {
                'stage': 'unified_bayesian',
                'current': trial.number + 1,
                'total': n_trials,
                'message': f'{model_name}: Trial {trial.number + 1}/{n_trials}'
            }

            if trial.value is not None:
                if task_type == 'regression':
                    progress_info['message'] += f" - RMSEcv: {trial.value:.4f}"
                elif task_type == 'one_class':
                    # For skipped one-class trials, value is +inf (we return
                    # float('inf') from the objective when CV is skipped).
                    # Show the reason instead of a useless "-inf".
                    if np.isinf(trial.value):
                        reason = trial.user_attrs.get('skip_reason', 'skipped')
                        progress_info['message'] += f" - SKIPPED ({reason})"
                    else:
                        progress_info['message'] += f" - BalancedAcccv: {-trial.value:.4f}"
                else:
                    progress_info['message'] += f" - Acccv: {-trial.value:.4f}"

            # Add best model tracking for "Best Model So Far" display
            if _study_ref[0].best_trial is not None:
                best = _study_ref[0].best_trial
                best_model = {
                    'Model': model_name,
                    'Preprocess': _build_display_preprocess_name(
                        best.user_attrs.get('preprocessing', 'raw'),
                        apply_baseline=best.user_attrs.get('apply_baseline', False),
                        baseline_method=baseline_method,
                        apply_smoothing=best.user_attrs.get('apply_smoothing', False),
                        apply_autoscale=best.user_attrs.get('apply_autoscale', False),  # T-36
                    ),
                    'n_vars': best.user_attrs.get('n_vars', 'N/A'),
                }
                if task_type == 'regression':
                    best_model['RMSEcv'] = best.value
                    # R²cv not available (only RMSE optimized), use placeholder
                    best_model['R2cv'] = 0.0
                elif task_type == 'one_class':
                    best_model['BalancedAcccv'] = -best.value
                else:
                    best_model['Accuracycv'] = -best.value
                progress_info['best_model'] = best_model

            progress_callback(progress_info)

        if verbose and (trial.number + 1) % 10 == 0:
            if trial.value is not None and trial.value < 1e9:
                if task_type == 'regression':
                    print(f"  Trial {trial.number + 1}/{n_trials}: RMSEcv={trial.value:.4f}")
                elif task_type == 'one_class':
                    print(f"  Trial {trial.number + 1}/{n_trials}: BalancedAcccv={-trial.value:.4f}")
                else:
                    print(f"  Trial {trial.number + 1}/{n_trials}: Acccv={-trial.value:.4f}")

    # T-11 D / Codex HIGH #3: when resuming a previously-loaded study,
    # study.optimize(n_trials=N) runs N MORE trials, not "until total reaches
    # N." Without this clamp, a crashed 80/100 study resumes with 100 fresh
    # trials (180 total) — silently doubling the user's wait. Compute the
    # remaining count from study.trials filtered to terminal states (Optuna
    # may include RUNNING / FAIL trials in the list, which shouldn't count
    # toward "completed").
    try:
        from optuna.trial import TrialState
        terminal_states = (TrialState.COMPLETE, TrialState.PRUNED)
        already_finished = sum(
            1 for t in study.trials if t.state in terminal_states
        )
    except Exception:
        # Defensive fallback — if Optuna's API surface changes, bias toward
        # running too few rather than too many trials so the user can re-run.
        already_finished = len(study.trials)

    remaining_trials = max(0, n_trials - already_finished)
    if already_finished > 0:
        print(
            f"  Resume detected: {already_finished} trials already complete; "
            f"requesting {remaining_trials} more (target: {n_trials})."
        )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[progress_wrapper],
            show_progress_bar=verbose and not progress_callback
        )
    else:
        print(f"  All {n_trials} trials already complete; skipping optimization.")

    # Auto-migration aborted the in-memory loop; restart on the SQLite study
    # so trials 11..N persist. Skip restart if the user pressed Stop during
    # migration (MEDIUM #1 — without this, one extra trial runs after Stop).
    if _auto_migrated:
        user_stopped_during_migration = (
            controller is not None and not controller.check_and_wait()
        )
        if user_stopped_during_migration:
            logger.info("T-41: user pressed Stop during migration; skipping restart")
        else:
            from optuna.trial import TrialState as _TS_post
            already_done_post = sum(
                1 for t in _study_ref[0].trials
                if t.state in (_TS_post.COMPLETE, _TS_post.PRUNED)
            )
            remaining_post = max(0, n_trials - already_done_post)
            if remaining_post > 0:
                logger.info(
                    "T-41: restarting optimize() on SQLite-backed study for %d remaining trials",
                    remaining_post,
                )
                _study_ref[0].optimize(
                    objective,
                    n_trials=remaining_post,
                    callbacks=[progress_wrapper],
                    show_progress_bar=verbose and not progress_callback,
                )

    # After optimization, resolve the final study object.  If 'auto' migration
    # happened, _study_ref[0] is the SQLite-backed study (now containing all
    # trials post the restart above); otherwise it's the original in-memory
    # study.
    study = _study_ref[0]

    # Convert results to DataFrame
    results_df = convert_study_to_dataframe(
        study, model_name, task_type, wavelengths, n_features, cv_folds,
        imbalance_method=imbalance_method,
        imbalance_params=imbalance_params,
        baseline_method=baseline_method,
        smoothing=smoothing,
        smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder,
        cv_strategy=cv_strategy,
        cv_n_repeats=cv_n_repeats,
        n_samples_used=n_samples,
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
            elif task_type == 'one_class':
                best_ba = results_df['BalancedAcccv'].max()
                print(f"Best BalancedAcccv: {best_ba:.6f}")
            else:
                best_acc = results_df['Accuracy'].max()
                print(f"Best Accuracy: {best_acc:.6f}")

            # Show best configuration
            if task_type == 'regression':
                best_row = results_df.loc[results_df['RMSEcv'].idxmin()]
            elif task_type == 'one_class':
                best_row = results_df.loc[results_df['BalancedAcccv'].idxmax()]
            else:
                best_row = results_df.loc[results_df['Accuracy'].idxmax()]

            print(f"\nBest Configuration:")
            print(f"  Preprocessing: {best_row['Preprocess']}")
            print(f"  Params: {best_row['Params']}")
            print(f"{'='*70}\n")

    return results_df, study


def convert_study_to_dataframe(
    study: optuna.Study,
    model_name: str,
    task_type: str,
    wavelengths: np.ndarray,
    n_features: int,
    cv_folds: int,
    imbalance_method: Optional[str] = None,
    imbalance_params: Optional[Dict[str, Any]] = None,
    baseline_method: str | None = None,
    smoothing: bool = False,
    smoothing_window: int = 17,
    smoothing_polyorder: int = 2,
    cv_strategy: str = 'kfold',
    cv_n_repeats: int = 5,
    n_samples_used: Optional[int] = None,
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
    imbalance_method : str, optional
        Imbalance handling method used
    imbalance_params : dict, optional
        Parameters for the imbalance method
    baseline_method : str or None
        Baseline correction method used (for display name prefixes)
    smoothing : bool
        Whether smoothing was available as an Optuna toggle

    Returns
    -------
    results_df : pd.DataFrame
        Results in standard DASP format
    """
    results = []

    # T-42 Approach C: early_stopping_rounds is hoisted to study.user_attrs.
    # Read once outside the trial loop. Trial-level fallback handles studies
    # finalized before the hoist (e.g., a SQLite from a prior dasp version
    # picked up via the resume flow) — those still have the per-trial value.
    _hoisted_es = study.user_attrs.get('early_stopping_rounds')

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Skip failed trials (penalty value)
        if trial.value is not None and trial.value >= 1e9:
            continue

        row = {
            'Task': task_type,
            'Model': model_name,
            'Preprocess': _build_display_preprocess_name(
                trial.user_attrs.get('preprocessing', 'unknown'),
                apply_baseline=trial.user_attrs.get('apply_baseline', False),
                baseline_method=baseline_method,
                apply_smoothing=trial.user_attrs.get('apply_smoothing', False),
                apply_autoscale=trial.user_attrs.get('apply_autoscale', False),  # T-36
            ),
            'PreprocessBase': _normalize_preprocess_name(trial.user_attrs.get('preprocessing', 'unknown')),
            'Deriv': trial.user_attrs.get('deriv', 0),
            'Window': trial.user_attrs.get('window', 0),
            'Poly': trial.user_attrs.get('poly', 0),
            'Autoscale': trial.user_attrs.get('apply_autoscale', False),  # T-36
            'baseline_method': baseline_method if trial.user_attrs.get('apply_baseline', False) else None,
            # T-36 fix (post-merge review v2): persist baseline_params alongside
            # baseline_method so the validation rebuild path can read both.
            # Without this, non-default ALS / polynomial settings used by the
            # Bayesian search silently snap back to defaults when validation
            # rebuilds the pipeline.
            'baseline_params': baseline_params if trial.user_attrs.get('apply_baseline', False) else None,
            'smoothing': trial.user_attrs.get('apply_smoothing', False),
            'smoothing_window': smoothing_window,
            'smoothing_polyorder': smoothing_polyorder,
            'Params': trial.user_attrs.get('model_params', '{}'),
            'n_vars': trial.user_attrs.get('n_vars', n_features),
            'full_vars': trial.user_attrs.get('full_vars_masked', n_features),
            'SubsetTag': trial.user_attrs.get('subset_tag', 'full'),
            'trial_number': trial.number,
            'Folds': cv_folds,
            'Optimization': 'Unified Bayesian',
            'Imbalance': imbalance_method if imbalance_method else '—',
            'early_stopping_rounds': (
                _hoisted_es
                if _hoisted_es is not None
                else trial.user_attrs.get('early_stopping_rounds', None)
            ),
            'imbalance_method': imbalance_method,
            'imbalance_params': imbalance_params,
            # training_config mirrors search.py so model save/load can restore
            # the exact CV strategy used. `folds` is an integer fallback for
            # old readers that don't know about cv_strategy (LOO reports the
            # effective sample count).
            'training_config': {
                'cv_strategy': cv_strategy,
                'cv_n_repeats': cv_n_repeats,
                'folds': (n_samples_used if cv_strategy == 'loo' and n_samples_used
                          else cv_folds),
                'n_samples_used': n_samples_used,
                'n_features_used': n_features,
                'random_state': 42,
            },
        }

        # Add metrics - both calibration and CV
        if task_type == 'regression':
            row['RMSE'] = trial.user_attrs.get('RMSE', np.nan)       # Calibration
            row['R2'] = trial.user_attrs.get('R2', np.nan)           # Calibration
            row['CCC'] = trial.user_attrs.get('CCC', np.nan)         # Calibration
            row['RMSEcv'] = trial.user_attrs.get('RMSEcv', trial.value)  # CV
            row['R2cv'] = trial.user_attrs.get('R2cv', np.nan)       # CV
            row['CCCcv'] = trial.user_attrs.get('CCCcv', np.nan)     # CV
            # NIR-specific metrics
            row['MAEcv'] = trial.user_attrs.get('MAEcv', np.nan)
            row['RPD'] = trial.user_attrs.get('RPD', np.nan)
            row['Bias'] = trial.user_attrs.get('Bias', np.nan)
            row['RER'] = trial.user_attrs.get('RER', np.nan)
            # Regional RMSE for quartile-based coloring in Results tab
            row['regional_rmse'] = trial.user_attrs.get('regional_rmse', None)
            row['y_quartiles'] = trial.user_attrs.get('y_quartiles', None)
            # Individual quartile columns for display/sorting
            regional_rmse = trial.user_attrs.get('regional_rmse')
            if regional_rmse:
                for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    row[f'RMSE_{q}'] = regional_rmse.get(q, np.nan)
        elif task_type == 'one_class':
            # Calibration metrics (keys from run_one_class_cv cal_metrics via one_class_metrics)
            row['Sensitivity'] = trial.user_attrs.get('sensitivity', np.nan)
            row['Specificity'] = trial.user_attrs.get('specificity', np.nan)
            row['Precision'] = trial.user_attrs.get('precision', np.nan)
            row['F1'] = trial.user_attrs.get('f1', np.nan)
            row['Accuracy'] = trial.user_attrs.get('accuracy', np.nan)
            row['BalancedAcc'] = trial.user_attrs.get('balanced_accuracy', np.nan)
            row['AUC'] = trial.user_attrs.get('auc', np.nan)
            # Cross-validation metrics (stored with _cv suffix by objective)
            row['Sensitivitycv'] = trial.user_attrs.get('sensitivity_cv', np.nan)
            row['Specificitycv'] = trial.user_attrs.get('specificity_cv', np.nan)
            row['Precisioncv'] = trial.user_attrs.get('precision_cv', np.nan)
            row['F1cv'] = trial.user_attrs.get('f1_cv', np.nan)
            row['Accuracycv'] = trial.user_attrs.get('accuracy_cv', np.nan)
            balanced_acc_cv = trial.user_attrs.get('balanced_accuracy_cv', np.nan)
            if balanced_acc_cv is None or (isinstance(balanced_acc_cv, float) and np.isnan(balanced_acc_cv)):
                # Derive from objective value
                balanced_acc_cv = -trial.value if (trial.value is not None and trial.value < 1e9) else np.nan
            row['BalancedAcccv'] = balanced_acc_cv
            row['AUCcv'] = trial.user_attrs.get('auc_cv', np.nan)
            # Fitted calibration artifacts (defensive: refinement re-runs CV, so the
            # only consumer of these today is any future direct-save-from-results
            # path). Column names match search.py:5171-5173 for grid-search parity.
            row['scaler'] = trial.user_attrs.get('cal_scaler')
            row['pca_reducer'] = trial.user_attrs.get('cal_pca_reducer')
            row['oc_score_stats'] = trial.user_attrs.get('oc_score_stats')
        else:
            # Calibration metrics
            row['Accuracy'] = trial.user_attrs.get('Accuracy', np.nan)
            row['ROC_AUC'] = trial.user_attrs.get('ROC_AUC', np.nan)
            row['F1'] = trial.user_attrs.get('F1', np.nan)
            row['Precision'] = trial.user_attrs.get('Precision', np.nan)
            row['Recall'] = trial.user_attrs.get('Recall', np.nan)
            row['Specificity'] = trial.user_attrs.get('Specificity', np.nan)
            row['Kappa'] = trial.user_attrs.get('Kappa', np.nan)
            row['MCC'] = trial.user_attrs.get('MCC', np.nan)
            row['BalancedAcc'] = trial.user_attrs.get('BalancedAcc', np.nan)
            row['BER'] = trial.user_attrs.get('BER', np.nan)
            row['LogLoss'] = trial.user_attrs.get('LogLoss', np.nan)

            # Cross-validation metrics
            accuracycv = trial.user_attrs.get('Accuracycv', -trial.value if trial.value else np.nan)
            row['Accuracycv'] = accuracycv
            row['CV Error'] = 1 - accuracycv if accuracycv is not None and not np.isnan(accuracycv) else np.nan
            row['ROC_AUCcv'] = trial.user_attrs.get('ROC_AUCcv', np.nan)
            row['F1cv'] = trial.user_attrs.get('F1cv', np.nan)
            row['Precisioncv'] = trial.user_attrs.get('Precisioncv', np.nan)
            row['Recallcv'] = trial.user_attrs.get('Recallcv', np.nan)
            row['Specificitycv'] = trial.user_attrs.get('Specificitycv', np.nan)
            row['Kappacv'] = trial.user_attrs.get('Kappacv', np.nan)
            row['MCCcv'] = trial.user_attrs.get('MCCcv', np.nan)
            row['BalancedAcccv'] = trial.user_attrs.get('BalancedAcccv', np.nan)
            row['BERcv'] = trial.user_attrs.get('BERcv', np.nan)
            row['LogLosscv'] = trial.user_attrs.get('LogLosscv', np.nan)
            # Per-class metrics for class-based coloring in Results tab
            row['per_class_metrics'] = trial.user_attrs.get('per_class_metrics', None)
            row['class_labels'] = trial.user_attrs.get('class_labels', None)
            # Individual class F1 columns for display/sorting
            per_class = trial.user_attrs.get('per_class_metrics')
            if per_class:
                for class_label, metrics in per_class.items():
                    row[f'F1_Class{class_label}'] = metrics.get('F1', np.nan)

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
        # Column order aligned with Grid Search: preprocessing cols early, top_vars/all_vars at end
        cols = ['Rank', 'Task', 'Model', 'Params', 'Preprocess', 'Deriv', 'Window',
                'Poly', 'LVs', 'n_vars', 'full_vars', 'SubsetTag', 'Imbalance',
                'early_stopping_rounds', 'trial_number', 'Folds', 'Optimization',
                'imbalance_method', 'imbalance_params']
        if task_type == 'one_class':
            cols.extend([
                'Sensitivity', 'Specificity', 'Precision', 'F1', 'Accuracy', 'BalancedAcc', 'AUC',
                'Sensitivitycv', 'Specificitycv', 'Precisioncv', 'F1cv',
                'Accuracycv', 'BalancedAcccv', 'AUCcv', 'CompositeScore', 'Score',
            ])
        elif task_type == 'classification':
            cols.extend(['Accuracy', 'Accuracycv', 'CV Error', 'ROC_AUC', 'ROC_AUCcv', 'CompositeScore', 'Score'])
        else:
            cols.extend(['RMSE', 'R2', 'RMSEcv', 'R2cv', 'MAEcv', 'RPD', 'Bias', 'RER', 'CompositeScore', 'Score'])
        cols.extend(['top_vars', 'all_vars'])
        return pd.DataFrame(columns=cols)

    # Sort by performance (use CV metrics for model selection)
    if task_type == 'regression':
        df = df.sort_values('RMSEcv', ascending=True)
    elif task_type == 'one_class':
        df = df.sort_values('BalancedAcccv', ascending=False)  # Higher balanced accuracy is better
    else:
        df = df.sort_values('CV Error', ascending=True)  # Lower CV Error is better

    df = df.reset_index(drop=True)

    # Add Rank column (required for report.py and model selection)
    df.insert(0, 'Rank', range(1, len(df) + 1))

    # Add CompositeScore column (required for report.py compatibility)
    # Lower is better for regression/classification; for OC = 1 - BalancedAcccv
    if task_type == 'regression':
        df['CompositeScore'] = df['RMSEcv']
    elif task_type == 'one_class':
        df['CompositeScore'] = 1.0 - df['BalancedAcccv']
    else:
        df['CompositeScore'] = df['CV Error']

    # Add Score column for compatibility with Coupled format
    if task_type == 'classification':
        df['Score'] = df['CV Error']
    elif task_type == 'one_class':
        df['Score'] = df['CompositeScore']
    else:
        df['Score'] = df['RMSEcv']

    # Reorder columns to match Grid Search format
    # Preprocessing columns early, metrics in middle, top_vars/all_vars at end
    base_cols = ['Rank', 'Task', 'Model', 'Params', 'Preprocess', 'Deriv', 'Window',
                 'Poly', 'LVs', 'n_vars', 'full_vars', 'SubsetTag', 'Imbalance',
                 'early_stopping_rounds']

    # Performance metrics
    if task_type == 'regression':
        perf_cols = ['RMSE', 'R2', 'RMSEcv', 'R2cv', 'MAEcv', 'RPD', 'Bias', 'RER', 'CompositeScore', 'Score']
    elif task_type == 'one_class':
        perf_cols = [
            # Calibration metrics
            'Sensitivity', 'Specificity', 'Precision', 'F1', 'Accuracy', 'BalancedAcc', 'AUC',
            # Cross-validation metrics
            'Sensitivitycv', 'Specificitycv', 'Precisioncv', 'F1cv',
            'Accuracycv', 'BalancedAcccv', 'AUCcv',
            # Composite
            'CompositeScore', 'Score',
        ]
    else:
        perf_cols = [
            # Calibration metrics
            'Accuracy', 'ROC_AUC', 'F1', 'Precision', 'Recall',
            'Specificity', 'Kappa', 'MCC', 'BalancedAcc', 'BER', 'LogLoss',
            # Cross-validation metrics
            'Accuracycv', 'ROC_AUCcv', 'F1cv', 'Precisioncv', 'Recallcv',
            'Specificitycv', 'Kappacv', 'MCCcv', 'BalancedAcccv', 'BERcv', 'LogLosscv',
            # Additional columns
            'CV Error', 'CompositeScore', 'Score'
        ]

    # Bayesian-specific and other columns
    other_cols = ['trial_number', 'Folds', 'Optimization', 'imbalance_method', 'imbalance_params']

    # Variable columns at end
    end_cols = ['top_vars', 'all_vars']

    # Build final column order (only include columns that exist in df)
    final_cols = []
    for col_list in [base_cols, perf_cols, other_cols, end_cols]:
        final_cols.extend([col for col in col_list if col in df.columns])

    # Add any remaining columns not in our explicit lists
    remaining = [col for col in df.columns if col not in final_cols]
    final_cols.extend(remaining)

    df = df[final_cols]

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
